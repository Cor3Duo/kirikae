use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Cat {
    Native,
    Interp,
    Svc,
    Ipc,
    Gpu,
    Present,
    Video,
    Unlocked,
    LockWait,
    KernelWait,
    Blocked,
    Sleep,
}

pub const CATS: usize = 12;
pub const NAMES: [&str; CATS] = ["native", "interp", "svc", "ipc", "gpu", "present", "video", "unlocked", "lock wait", "kernel wait", "blocked", "sleep"];
pub const HELD: [Cat; 4] = [Cat::Native, Cat::Interp, Cat::Svc, Cat::Ipc];

pub struct Slots {
    pub name: String,
    pub ns: [AtomicU64; CATS],
    pub simd: AtomicU64,
    pub ops: AtomicU64,
}

static ON: AtomicBool = AtomicBool::new(false);
pub static SVC_NS: [AtomicU64; 0x80] = [const { AtomicU64::new(0) }; 0x80];
pub static SVC_CALLS: [AtomicU64; 0x80] = [const { AtomicU64::new(0) }; 0x80];
pub static LOCK_HANDOFF_NS: AtomicU64 = AtomicU64::new(0);
pub static LOCK_HANDOFFS: AtomicU64 = AtomicU64::new(0);
pub static LOCK_IDLE_NS: AtomicU64 = AtomicU64::new(0);

pub fn lock_free(d: std::time::Duration, waiting: bool) {
    let ns = d.as_nanos() as u64;
    if waiting {
        LOCK_HANDOFF_NS.fetch_add(ns, Ordering::Relaxed);
        LOCK_HANDOFFS.fetch_add(1, Ordering::Relaxed);
    } else {
        LOCK_IDLE_NS.fetch_add(ns, Ordering::Relaxed);
    }
}

static THREADS: Mutex<Vec<Arc<Slots>>> = Mutex::new(Vec::new());

struct Local {
    cat: Cell<Cat>,
    svc: Cell<u8>,
    since: Cell<Instant>,
    simd: Cell<u64>,
    ops: Cell<u64>,
    slots: Arc<Slots>,
}

thread_local! {
    static LOCAL: Local = {
        let slots = Arc::new(Slots {
            name: std::thread::current().name().unwrap_or("?").to_string(),
            ns: Default::default(),
            simd: AtomicU64::new(0),
            ops: AtomicU64::new(0),
        });
        THREADS.lock().unwrap().push(slots.clone());
        Local { cat: Cell::new(Cat::Native), svc: Cell::new(0), since: Cell::new(Instant::now()), simd: Cell::new(0), ops: Cell::new(0), slots }
    };
}

pub fn enable() {
    ON.store(true, Ordering::Relaxed);
    if std::env::var_os("KIRIKAE_SIMD_HIST").is_some() {
        SIMD_HIST_ON.store(true, Ordering::Relaxed);
        *SIMD_HIST.lock().unwrap() = Some(Default::default());
    }
}

#[inline(always)]
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

fn switch(l: &Local, to: Cat) -> Cat {
    let from = l.cat.get();
    if from != to {
        let now = Instant::now();
        let ns = now.duration_since(l.since.get()).as_nanos() as u64;
        l.slots.ns[from as usize].fetch_add(ns, Ordering::Relaxed);
        if from == Cat::Svc {
            SVC_NS[l.svc.get() as usize & 0x7f].fetch_add(ns, Ordering::Relaxed);
        }
        l.slots.simd.fetch_add(l.simd.replace(0), Ordering::Relaxed);
        l.slots.ops.fetch_add(l.ops.replace(0), Ordering::Relaxed);
        l.since.set(now);
        l.cat.set(to);
    }
    from
}

#[inline]
pub fn set(to: Cat) -> Cat {
    if !on() {
        return to;
    }
    LOCAL.with(|l| switch(l, to))
}

#[inline]
pub fn enter(to: Cat) -> Guard {
    Guard(on().then(|| set(to)))
}

pub struct Guard(Option<Cat>);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(prev) = self.0 {
            set(prev);
        }
    }
}

pub fn enter_svc(imm: u16) -> Guard {
    if !on() {
        return Guard(None);
    }
    SVC_CALLS[imm as usize & 0x7f].fetch_add(1, Ordering::Relaxed);
    LOCAL.with(|l| {
        let prev = switch(l, Cat::Svc);
        l.svc.set(imm as u8);
        Guard(Some(prev))
    })
}

static SIMD_HIST: Mutex<Option<std::collections::HashMap<u32, u64>>> = Mutex::new(None);

#[inline]
pub fn count_simd(word: u32) {
    if on() {
        LOCAL.with(|l| l.simd.set(l.simd.get() + 1));
        if let Some(h) = &mut *SIMD_HIST.lock().unwrap() {
            *h.entry(word).or_default() += 1;
        }
    }
}

pub fn write_simd_hist() {
    let Some(path) = std::env::var_os("KIRIKAE_SIMD_HIST") else { return };
    let mut rows: Vec<(u32, u64)> = match &*SIMD_HIST.lock().unwrap() {
        Some(h) => h.iter().map(|(w, n)| (*w, *n)).collect(),
        None => return,
    };
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let text: String = rows.iter().map(|(w, n)| format!("{w:08x} {n}
")).collect();
    let _ = std::fs::write(path, text);
}

static OP_HIST: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];

#[inline]
pub fn count_op(op: u32) {
    if on() {
        LOCAL.with(|l| l.ops.set(l.ops.get() + 1));
        if SIMD_HIST_ON.load(Ordering::Relaxed) {
            OP_HIST[op as usize & 0xff].fetch_add(1, Ordering::Relaxed);
        }
    }
}

static SIMD_HIST_ON: AtomicBool = AtomicBool::new(false);

pub fn op_hist() -> Vec<(usize, u64)> {
    let mut rows: Vec<(usize, u64)> = OP_HIST.iter().enumerate().map(|(i, n)| (i, n.load(Ordering::Relaxed))).filter(|r| r.1 > 0).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    rows
}

pub fn snapshot() -> Vec<(String, [u64; CATS], u64, u64)> {
    THREADS
        .lock()
        .unwrap()
        .iter()
        .map(|s| (s.name.clone(), std::array::from_fn(|i| s.ns[i].load(Ordering::Relaxed)), s.simd.load(Ordering::Relaxed), s.ops.load(Ordering::Relaxed)))
        .collect()
}
