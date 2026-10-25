use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::input::PadState;
use crate::kernel::{Event, KObject, SharedMemory};
use crate::process::Process;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const SHARED_SIZE: u64 = 0x40000;
static SUPPORTED_STYLES: AtomicU32 = AtomicU32::new(0);
static SUPPORTED_IDS: AtomicU32 = AtomicU32::new(0);

const STYLE_FULL_KEY: u32 = 1 << 0;
const STYLE_HANDHELD: u32 = 1 << 1;
const STYLE_JOY_DUAL: u32 = 1 << 2;

const NPAD: u64 = 0x9a00;
const NPAD_ENTRY: u64 = 0x5000;
const HANDHELD_INDEX: usize = 8;

const IDLE_LIFOS: [(u64, u64); 4] = [(0x0, 0x28), (0x400, 0x298), (0x3400, 0x30), (0x3800, 0x38)];

const NPAD_CONDITION: u64 = 0x3e200;
const HOLD_HORIZONTAL: u32 = 1;

mod field {
    pub const STYLE_SET: u64 = 0x0;
    pub const JOY_ASSIGNMENT_MODE: u64 = 0x4;
    pub const FULL_KEY_COLOR: u64 = 0x8;
    pub const JOY_COLOR: u64 = 0x14;
    pub const LIFOS: u64 = 0x28;
    pub const LIFO_SIZE: u64 = 0x350;
    pub const SYSTEM_EXT_LIFO: usize = 6;
    pub const SIX_AXIS_LIFOS: u64 = 0x1758;
    pub const SIX_AXIS_LIFO_SIZE: u64 = 0x708;
    pub const DEVICE_TYPE: u64 = 0x4188;
    pub const SYSTEM_PROPERTIES: u64 = 0x4190;
    pub const BATTERY_LEVELS: u64 = 0x419c;
    pub const APPLET_FOOTER_UI_ATTRIBUTE: u64 = 0x41a8;
    pub const APPLET_FOOTER_UI_TYPE: u64 = 0x41ac;
}

const LIFO_ENTRIES: u64 = 17;
const COMMON_STORAGE: u64 = 0x30;
const SIX_AXIS_STORAGE: u64 = 0x68;

const RATE_HZ: u64 = 120;

struct Style {
    tag: u32,
    name: &'static str,
    device_type: u32,
    attributes: u32,
    applet_footer_ui_type: u8,
    lifo: usize,
    six_axis: &'static [usize],
}

const FULL_KEY: Style =
    Style { tag: STYLE_FULL_KEY, name: "Pro Controller", device_type: 1, attributes: 0x3, applet_footer_ui_type: 12, lifo: 0, six_axis: &[0] };
const HANDHELD: Style = Style {
    tag: STYLE_HANDHELD,
    name: "handheld controller",
    device_type: (1 << 2) | (1 << 3),
    attributes: 0x3f,
    applet_footer_ui_type: 4,
    lifo: 1,
    six_axis: &[1],
};
const JOY_DUAL: Style = Style {
    tag: STYLE_JOY_DUAL,
    name: "pair of Joy-Cons",
    device_type: (1 << 4) | (1 << 5),
    attributes: 0x15,
    applet_footer_ui_type: 5,
    lifo: 2,
    six_axis: &[2, 3],
};

fn npad_index(id: u32) -> Option<usize> {
    match id {
        0..=7 => Some(id as usize),
        0x20 => Some(HANDHELD_INDEX),
        0x10 => Some(9),
        _ => None,
    }
}

fn npad_id(index: usize) -> u32 {
    match index {
        HANDHELD_INDEX => 0x20,
        9 => 0x10,
        i => i as u32,
    }
}

fn choose(styles: u32, ids: u32, prefer_handheld: bool) -> Option<(usize, &'static Style)> {
    let styles = if styles == 0 { STYLE_FULL_KEY | STYLE_HANDHELD } else { styles };
    let ids = if ids == 0 { !0 } else { ids };
    let pro = styles & STYLE_FULL_KEY != 0 && ids & 1 != 0;
    let handheld = styles & STYLE_HANDHELD != 0 && ids & (1 << HANDHELD_INDEX) != 0;
    if handheld && (prefer_handheld || !pro) {
        Some((HANDHELD_INDEX, &HANDHELD))
    } else if pro {
        Some((0, &FULL_KEY))
    } else if styles & STYLE_JOY_DUAL != 0 && ids & 1 != 0 {
        Some((0, &JOY_DUAL))
    } else {
        None
    }
}

struct Npads {
    memory: Arc<SharedMemory>,
    handle: u32,
    started: AtomicBool,
    inner: Mutex<Inner>,
    events: Mutex<HashMap<u32, Arc<Event>>>,
}

#[derive(Default)]
struct Inner {
    connected: Option<(usize, &'static Style)>,
    sampling: u64,
    tail: [u64; 10],
    count: [u64; 10],
}

fn put32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn putf(b: &mut [u8], off: usize, v: f32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn storage_sequence(sampling: u64) -> u64 {
    sampling * 2
}

fn common_storage(sampling: u64, pad: &PadState, attributes: u32) -> [u8; COMMON_STORAGE as usize] {
    let mut b = [0; COMMON_STORAGE as usize];
    put64(&mut b, 0, storage_sequence(sampling));
    put64(&mut b, 0x8, sampling);
    put64(&mut b, 0x10, pad.buttons);
    put32(&mut b, 0x18, pad.left.0 as u32);
    put32(&mut b, 0x1c, pad.left.1 as u32);
    put32(&mut b, 0x20, pad.right.0 as u32);
    put32(&mut b, 0x24, pad.right.1 as u32);
    put32(&mut b, 0x28, attributes);
    b
}

fn six_axis_storage(sampling: u64, connected: bool) -> [u8; SIX_AXIS_STORAGE as usize] {
    let mut b = [0; SIX_AXIS_STORAGE as usize];
    put64(&mut b, 0, storage_sequence(sampling));
    let s = &mut b[8..];
    put64(s, 0x8, sampling);
    if !connected {
        return b;
    }
    put64(s, 0x0, 1_000_000_000 / RATE_HZ);
    putf(s, 0x18, -1.0);
    putf(s, 0x34, 1.0);
    putf(s, 0x34 + 0x10, 1.0);
    putf(s, 0x34 + 0x20, 1.0);
    put32(s, 0x58, 1);
    b
}

fn lifo_header(tail: u64, count: u64) -> [u8; 0x20] {
    let mut b = [0; 0x20];
    put64(&mut b, 0x8, LIFO_ENTRIES);
    put64(&mut b, 0x10, tail);
    put64(&mut b, 0x18, count);
    b
}

impl Npads {
    fn update(&self, pad: &PadState, wanted: Option<(usize, &'static Style)>, write: &mut dyn FnMut(u64, &[u8])) -> Vec<u32> {
        let mut st = self.inner.lock().unwrap();
        let mut changed = Vec::new();
        let old = st.connected;
        if old.map(|(i, s)| (i, s.tag)) != wanted.map(|(i, s)| (i, s.tag)) {
            if let Some((i, _)) = old {
                write_header(write, i, None);
                changed.push(npad_id(i));
            }
            if let Some((i, s)) = wanted {
                write_header(write, i, Some(s));
                changed.push(npad_id(i));
            }
            st.connected = wanted;
        }
        st.sampling += 1;
        let sampling = st.sampling;
        let mut entries = Vec::new();
        if let Some((i, _)) = old.filter(|_| !changed.is_empty()) {
            entries.push((i, None));
        }
        if let Some((i, s)) = wanted {
            entries.push((i, Some(s)));
        }
        for (i, style) in entries {
            let tail = (st.tail[i] + 1) % LIFO_ENTRIES;
            let count = (st.count[i] + 1).min(LIFO_ENTRIES - 1);
            st.tail[i] = tail;
            st.count[i] = count;
            let base = NPAD + i as u64 * NPAD_ENTRY;
            let live = common_storage(sampling, pad, style.map_or(0, |s| s.attributes));
            let idle = common_storage(sampling, &PadState::default(), 0);
            for lifo in 0..7 {
                let at = base + field::LIFOS + lifo as u64 * field::LIFO_SIZE;
                let data = match style {
                    Some(s) if lifo == s.lifo || lifo == field::SYSTEM_EXT_LIFO => &live,
                    _ => &idle,
                };
                write(at + 0x20 + tail * COMMON_STORAGE, data);
                write(at, &lifo_header(tail, count));
            }
            for lifo in 0..6 {
                let at = base + field::SIX_AXIS_LIFOS + lifo as u64 * field::SIX_AXIS_LIFO_SIZE;
                let on = style.is_some_and(|s| s.six_axis.contains(&lifo));
                write(at + 0x20 + tail * SIX_AXIS_STORAGE, &six_axis_storage(sampling, on));
                write(at, &lifo_header(tail, count));
            }
        }
        let tail = sampling % LIFO_ENTRIES;
        let count = sampling.min(LIFO_ENTRIES - 1);
        for &(at, size) in &IDLE_LIFOS {
            let mut b = vec![0u8; size as usize];
            put64(&mut b, 0, storage_sequence(sampling));
            put64(&mut b, 8, sampling);
            write(at + 0x20 + tail * size, &b);
            write(at, &lifo_header(tail, count));
        }
        changed
    }
}

fn write_header(write: &mut dyn FnMut(u64, &[u8]), index: usize, style: Option<&Style>) {
    let base = NPAD + index as u64 * NPAD_ENTRY;
    let mut h = [0u8; 0x28];
    if let Some(s) = style {
        put32(&mut h, field::STYLE_SET as usize, s.tag);
        put32(&mut h, field::JOY_ASSIGNMENT_MODE as usize, 0);
        let f = field::FULL_KEY_COLOR as usize;
        put32(&mut h, f + 4, 0x2d2d2d);
        put32(&mut h, f + 8, 0xe6e6e6);
        let j = field::JOY_COLOR as usize;
        put32(&mut h, j + 4, 0x0ab9e6);
        put32(&mut h, j + 8, 0x001e1e);
        put32(&mut h, j + 12, 0xff3c28);
        put32(&mut h, j + 16, 0x1e0a0a);
    } else {
        put32(&mut h, field::FULL_KEY_COLOR as usize, 2);
        put32(&mut h, field::JOY_COLOR as usize, 2);
    }
    write(base, &h);

    let mut t = [0u8; 0x28];
    if let Some(s) = style {
        put32(&mut t, 0, s.device_type);
        put64(&mut t, 0x8, (0b111 << 3) | (1 << 11) | (0b111 << 13));
        for k in 0..3 {
            put32(&mut t, 0x14 + 4 * k, 4);
        }
        put32(&mut t, 0x20, 0);
        t[0x24] = s.applet_footer_ui_type;
    }
    debug_assert_eq!(field::SYSTEM_PROPERTIES - field::DEVICE_TYPE, 0x8);
    debug_assert_eq!(field::BATTERY_LEVELS - field::DEVICE_TYPE, 0x14);
    debug_assert_eq!(field::APPLET_FOOTER_UI_ATTRIBUTE - field::DEVICE_TYPE, 0x20);
    debug_assert_eq!(field::APPLET_FOOTER_UI_TYPE - field::DEVICE_TYPE, 0x24);
    write(base + field::DEVICE_TYPE, &t);
}

fn prefer_handheld() -> bool {
    std::env::var("KIRIKAE_NPAD").is_ok_and(|v| v.eq_ignore_ascii_case("handheld"))
}

fn wanted() -> Option<(usize, &'static Style)> {
    choose(SUPPORTED_STYLES.load(Relaxed), SUPPORTED_IDS.load(Relaxed), prefer_handheld())
}

impl Npads {
    fn set_hold_type(&self, p: &Process, hold: u32) {
        let mut c = [0u8; 0x10];
        put32(&mut c, 0x8, hold);
        c[0xc] = 1;
        self.memory.write(&p.memory, NPAD_CONDITION, &c);
    }

    fn tick(&self, p: &Process) {
        let mut write = |off: u64, bytes: &[u8]| self.memory.write(&p.memory, off, bytes);
        let changed = self.update(&crate::input::state(), wanted(), &mut write);
        if changed.is_empty() {
            return;
        }
        if let Some((i, s)) = self.inner.lock().unwrap().connected {
            p.warn_once(format!("hid: {} connected as NpadId {:#x}", s.name, npad_id(i)));
        }
        let events = self.events.lock().unwrap();
        for id in changed {
            if let Some(e) = events.get(&id) {
                e.signaled.store(true, Relaxed);
            }
        }
    }
}

fn acquire_style_set_event(
    events: &Mutex<HashMap<u32, Arc<Event>>>,
    handles: &mut crate::kernel::HandleTable,
    id: u32,
) -> u32 {
    let event = events
        .lock()
        .unwrap()
        .entry(id)
        .or_insert_with(|| Arc::new(Event { signaled: AtomicBool::new(false) }))
        .clone();
    event.signaled.store(true, Relaxed);
    handles.insert(KObject::Event(event))
}

fn npads(c: &Cpu) -> Arc<Npads> {
    let p = c.process();
    let n = p.service_state("hid.npads", || {
        let (memory, handle) = p.create_shared_memory(SHARED_SIZE);
        Npads { memory, handle, started: AtomicBool::new(false), inner: Mutex::default(), events: Mutex::default() }
    });
    if !n.started.swap(true, Relaxed) {
        n.tick(p);
        n.set_hold_type(p, HOLD_HORIZONTAL);
        start_updates(p, n.clone());
    }
    n
}

fn start_updates(p: &'static Process, n: Arc<Npads>) {
    let _ = std::thread::Builder::new().name("kirikae-hid".into()).spawn(move || {
        let period = std::time::Duration::from_nanos(1_000_000_000 / RATE_HZ);
        let mut next = std::time::Instant::now();
        loop {
            next += period;
            std::thread::sleep(next.saturating_duration_since(std::time::Instant::now()));
            if p.sched.exiting.load(Relaxed) {
                return;
            }
            n.tick(p);
        }
    });
}

pub struct Hid;

impl Service for Hid {
    fn name(&self) -> String {
        "hid".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(AppletResource)),
            1 | 11 | 21 | 31 | 40 | 66 | 67 | 68 | 69 | 82 | 103 | 104 | 109 | 122 | 123 | 124 | 125 | 128 | 129 | 130 | 131
            | 132 | 133 | 201 | 206 | 1000 => crate::services::stub::plain_ok(),
            100 => {
                SUPPORTED_STYLES.store(req.u32(0), Relaxed);
                npads(c).tick(c.process());
                Reply::ok()
            }
            101 => Reply::ok().u32(SUPPORTED_STYLES.load(Relaxed)),
            102 => {
                let p = c.process();
                let ids = req.in_buf(0).and_then(|b| p.memory.read(b.addr, b.size.min(0x100))).unwrap_or_default();
                let mask = ids.as_chunks::<4>().0.iter().filter_map(|w| npad_index(u32::from_le_bytes(*w))).fold(0, |m, i| m | 1 << i);
                SUPPORTED_IDS.store(mask, Relaxed);
                npads(c).tick(p);
                Reply::ok()
            }
            106 => {
                let p = c.process();
                let n = npads(c);
                let mut handles = p.handles.lock().unwrap();
                let handle = acquire_style_set_event(&n.events, &mut handles, req.u32(0));
                Reply::ok().copy_handle(handle)
            }
            108 => {
                const LEDS: [u64; 8] = [0x1, 0x3, 0x7, 0xf, 0x9, 0x5, 0xd, 0x6];
                Reply::ok().u64(LEDS.get(req.u32(0) as usize).copied().unwrap_or(0))
            }
            120 => {
                npads(c).set_hold_type(c.process(), req.u64(8) as u32);
                Reply::ok()
            }
            121 => {
                let p = c.process();
                let b = npads(c).memory.read(&p.memory, NPAD_CONDITION + 8, 4);
                Reply::ok().u64(u32::from_le_bytes(b[..4].try_into().unwrap()) as u64)
            }
            200 => {
                let device = (req.u32(0) >> 16) & 0xff;
                Reply::ok().u32(1).u32(device.min(1) + 1)
            }
            203 => Reply::ok().object(Arc::new(ActiveVibrationDeviceList)),
            _ => unimplemented("hid", c, req),
        }
    }
}

struct ActiveVibrationDeviceList;

impl Service for ActiveVibrationDeviceList {
    fn name(&self) -> String {
        "hid/IActiveVibrationDeviceList".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok(),
            _ => unimplemented("hid/IActiveVibrationDeviceList", c, req),
        }
    }
}

struct AppletResource;

impl Service for AppletResource {
    fn name(&self) -> String {
        "hid/IAppletResource".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().copy_handle(npads(c).handle),
            _ => unimplemented("hid/IAppletResource", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::HandleTable;
    use crate::input::button;

    fn u32_at(m: &[u8], off: u64) -> u32 {
        u32::from_le_bytes(m[off as usize..off as usize + 4].try_into().unwrap())
    }

    fn u64_at(m: &[u8], off: u64) -> u64 {
        u64::from_le_bytes(m[off as usize..off as usize + 8].try_into().unwrap())
    }

    #[test]
    fn chooses_pro_controller_unless_unsupported() {
        assert_eq!(choose(0, 0, false).map(|(i, s)| (i, s.tag)), Some((0, STYLE_FULL_KEY)));
        assert_eq!(choose(STYLE_HANDHELD | STYLE_JOY_DUAL, 0, false).map(|(i, s)| (i, s.tag)), Some((8, STYLE_HANDHELD)));
        assert_eq!(choose(STYLE_FULL_KEY | STYLE_HANDHELD, 0, true).map(|(i, s)| (i, s.tag)), Some((8, STYLE_HANDHELD)));
        assert_eq!(choose(STYLE_HANDHELD | STYLE_JOY_DUAL, 1, false).map(|(i, s)| (i, s.tag)), Some((0, STYLE_JOY_DUAL)));
        assert!(choose(1 << 5, 0, false).is_none());
    }

    #[test]
    fn writes_npad_entry_and_ring_buffers() {
        let n = Npads {
            memory: Arc::new(SharedMemory::new(SHARED_SIZE)),
            handle: 0,
            started: AtomicBool::new(false),
            inner: Mutex::default(),
            events: Mutex::default(),
        };
        let mem = std::cell::RefCell::new(vec![0u8; SHARED_SIZE as usize]);
        let mut write = |off: u64, b: &[u8]| mem.borrow_mut()[off as usize..off as usize + b.len()].copy_from_slice(b);
        let pad = PadState { buttons: button::A, left: (100, -200), right: (0, 0) };
        assert_eq!(n.update(&pad, choose(0, 0, false), &mut write), vec![0]);
        for _ in 0..20 {
            assert!(n.update(&pad, choose(0, 0, false), &mut write).is_empty());
        }
        let e = NPAD;
        assert_eq!(u32_at(&mem.borrow(), e + field::STYLE_SET), STYLE_FULL_KEY);
        assert_eq!(u32_at(&mem.borrow(), e + field::DEVICE_TYPE), 1);
        assert_eq!(mem.borrow()[(e + field::APPLET_FOOTER_UI_TYPE) as usize], 12);
        let lifo = e + field::LIFOS;
        assert_eq!(u64_at(&mem.borrow(), lifo + 0x8), 17);
        let (tail, count) = (u64_at(&mem.borrow(), lifo + 0x10), u64_at(&mem.borrow(), lifo + 0x18));
        assert_eq!((tail, count), (21 % 17, 16));
        let s = lifo + 0x20 + tail * COMMON_STORAGE;
        assert_eq!(u64_at(&mem.borrow(), s), 42);
        assert_eq!(u64_at(&mem.borrow(), s + 0x8), 21);
        assert_eq!(u64_at(&mem.borrow(), s + 0x10), button::A);
        assert_eq!(u32_at(&mem.borrow(), s + 0x18) as i32, 100);
        assert_eq!(u32_at(&mem.borrow(), s + 0x1c) as i32, -200);
        assert_eq!(u32_at(&mem.borrow(), s + 0x28), 0x3);
        let h = e + field::LIFOS + field::LIFO_SIZE + 0x20 + tail * COMMON_STORAGE;
        assert_eq!((u64_at(&mem.borrow(), h), u64_at(&mem.borrow(), h + 0x10)), (42, 0));
        let changed = n.update(&pad, choose(0, 0, true), &mut write);
        assert_eq!(changed, vec![0, 0x20]);
        assert_eq!(u32_at(&mem.borrow(), e + field::STYLE_SET), 0);
        let hh = NPAD + 8 * NPAD_ENTRY;
        assert_eq!(u32_at(&mem.borrow(), hh + field::STYLE_SET), STYLE_HANDHELD);
        let hl = hh + field::LIFOS + field::LIFO_SIZE;
        let t = u64_at(&mem.borrow(), hl + 0x10);
        assert_eq!(u64_at(&mem.borrow(), hl + 0x20 + t * COMMON_STORAGE + 0x10), button::A);
    }

    #[test]
    fn every_acquire_gets_a_fresh_handle_for_the_same_event() {
        let events = Mutex::new(HashMap::new());
        let mut handles = HandleTable::default();
        let first = acquire_style_set_event(&events, &mut handles, 0);
        let second = acquire_style_set_event(&events, &mut handles, 0);
        assert_ne!(first, second, "each client gets its own handle");
        let signaled = |h| matches!(handles.get(h), Some(KObject::Event(e)) if e.signaled.load(Relaxed));
        assert!(signaled(first) && signaled(second));
        assert_eq!(events.lock().unwrap().len(), 1, "one event per npad id");
    }
}
