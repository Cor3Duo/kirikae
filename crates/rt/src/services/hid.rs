use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::input::PadState;
use crate::kernel::SharedMemory;
use crate::process::Process;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const SHARED_SIZE: u64 = 0x40000;
static SUPPORTED_STYLES: AtomicU32 = AtomicU32::new(0);
static SUPPORTED_IDS: AtomicU32 = AtomicU32::new(0);
static DEBUG_PAD_ACTIVE: AtomicBool = AtomicBool::new(false);
static TOUCH_SCREEN_ACTIVE: AtomicBool = AtomicBool::new(false);

const STYLE_FULL_KEY: u32 = 1 << 0;
const STYLE_HANDHELD: u32 = 1 << 1;
const STYLE_JOY_DUAL: u32 = 1 << 2;

/// `nn::hid::NpadJoyAssignmentMode`.
const JOY_ASSIGNMENT_SINGLE: u32 = 0;
const JOY_ASSIGNMENT_DUAL: u32 = 1;

/// `nn::hid::NpadHandheldActivationMode`: 0 Dual, 1 Single, 2 None. A console
/// starts in Dual, which is what the synthetic handheld pad models.
const HANDHELD_ACTIVATION_DUAL: u32 = 0;
const HANDHELD_ACTIVATION_MAX: u32 = 2;

static HANDHELD_ACTIVATION_MODE: AtomicU32 = AtomicU32::new(HANDHELD_ACTIVATION_DUAL);

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

const FULL_KEY: Style = Style {
    tag: STYLE_FULL_KEY,
    name: "Pro Controller",
    device_type: 1,
    attributes: 0x3,
    applet_footer_ui_type: 12,
    lifo: 0,
    six_axis: &[0],
};
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
    let styles = if styles == 0 {
        STYLE_FULL_KEY | STYLE_HANDHELD
    } else {
        styles
    };
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
    events: Mutex<HashMap<u32, u32>>,
    state: Mutex<State>,
}

#[derive(Default)]
struct Inner {
    connected: Option<(usize, &'static Style)>,
    sampling: u64,
    tail: [u64; 10],
    count: [u64; 10],
    assignment: [u32; 10],
    disconnected: [bool; 10],
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
    fn update(
        &self,
        pad: &PadState,
        wanted: Option<(usize, &'static Style)>,
        write: &mut dyn FnMut(u64, &[u8]),
    ) -> Vec<u32> {
        let mut st = self.inner.lock().unwrap();
        let mut changed = Vec::new();
        let old = st.connected;
        if old.map(|(i, s)| (i, s.tag)) != wanted.map(|(i, s)| (i, s.tag)) {
            if let Some((i, _)) = old {
                write_header(write, i, None, JOY_ASSIGNMENT_SINGLE);
                changed.push(npad_id(i));
            }
            if let Some((i, s)) = wanted {
                write_header(write, i, Some(s), st.assignment[i]);
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
                write(
                    at + 0x20 + tail * SIX_AXIS_STORAGE,
                    &six_axis_storage(sampling, on),
                );
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

fn write_header(
    write: &mut dyn FnMut(u64, &[u8]),
    index: usize,
    style: Option<&Style>,
    assignment_mode: u32,
) {
    let base = NPAD + index as u64 * NPAD_ENTRY;
    let mut h = [0u8; 0x28];
    if let Some(s) = style {
        put32(&mut h, field::STYLE_SET as usize, s.tag);
        put32(&mut h, field::JOY_ASSIGNMENT_MODE as usize, assignment_mode);
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
    choose(
        SUPPORTED_STYLES.load(Relaxed),
        SUPPORTED_IDS.load(Relaxed),
        prefer_handheld(),
    )
}

const VIBRATION_DEVICE_HANDLE_SIZE: usize = 4;
const VIBRATION_VALUE_SIZE: usize = 0x10;
const VIBRATION_HISTORY: usize = 8;
const VIBRATION_DEVICES: usize = 16;
const VIBRATION_HANDLE_LIMIT: u64 = (VIBRATION_DEVICES * VIBRATION_DEVICE_HANDLE_SIZE) as u64;
const VIBRATION_VALUE_LIMIT: u64 = (VIBRATION_DEVICES * VIBRATION_VALUE_SIZE) as u64;
const MAX_SIX_AXIS_SENSORS: usize = 8;
const NPAD_REVISIONS: [u32; 5] = [0, 1, 2, 3, 5];

static NPAD_REVISION: AtomicU32 = AtomicU32::new(0);

fn field_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let b = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn field_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    field_u32(bytes, offset).map(f32::from_bits)
}

fn clamp_amplitude(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

/// Latest amplitudes sent for one `nn::hid::VibrationDeviceHandle`, in the wire order of
/// `nn::hid::VibrationValue`: `amplitude_low` is the low band and `amplitude_high` the
/// high band (the two haptic actuators of the device). Both are clamped to 0.0..=1.0.
#[derive(Clone, Copy, Debug, PartialEq)]
struct VibrationSample {
    amplitude_low: f32,
    amplitude_high: f32,
}

struct VibrationDevice {
    handle: u32,
    samples: u64,
    next: usize,
    history: [VibrationSample; VIBRATION_HISTORY],
}

impl VibrationDevice {
    fn new(handle: u32) -> VibrationDevice {
        VibrationDevice {
            handle,
            samples: 0,
            next: 0,
            history: [VibrationSample {
                amplitude_low: 0.0,
                amplitude_high: 0.0,
            }; VIBRATION_HISTORY],
        }
    }

    fn push(&mut self, sample: VibrationSample) {
        self.history[self.next] = sample;
        self.next = (self.next + 1) % VIBRATION_HISTORY;
        self.samples += 1;
    }
}

struct Vibration {
    devices: [Option<VibrationDevice>; VIBRATION_DEVICES],
    dropped: u64,
}

impl Default for Vibration {
    fn default() -> Vibration {
        Vibration {
            devices: std::array::from_fn(|_| None),
            dropped: 0,
        }
    }
}

impl Vibration {
    fn device(&mut self, handle: u32) -> Option<&mut VibrationDevice> {
        let found = self
            .devices
            .iter()
            .position(|d| d.as_ref().is_some_and(|d| d.handle == handle));
        let index = match found {
            Some(i) => i,
            None => {
                let Some(i) = self.devices.iter().position(Option::is_none) else {
                    self.dropped = self.dropped.saturating_add(1);
                    return None;
                };
                self.devices[i] = Some(VibrationDevice::new(handle));
                i
            }
        };
        self.devices[index].as_mut()
    }

    fn record(&mut self, handle: u32, amplitude_low: f32, amplitude_high: f32) {
        let sample = VibrationSample {
            amplitude_low: clamp_amplitude(amplitude_low),
            amplitude_high: clamp_amplitude(amplitude_high),
        };
        if let Some(device) = self.device(handle) {
            device.push(sample);
        }
    }

    #[cfg(test)]
    fn latest(&self, handle: u32) -> Option<VibrationSample> {
        let device = self.devices.iter().flatten().find(|d| d.handle == handle)?;
        let last = device.samples.checked_sub(1)?;
        Some(device.history[last as usize % VIBRATION_HISTORY])
    }

    #[cfg(test)]
    fn samples(&self, handle: u32) -> u64 {
        self.devices
            .iter()
            .flatten()
            .find(|d| d.handle == handle)
            .map_or(0, |d| d.samples)
    }

    #[cfg(test)]
    fn history(&self, handle: u32) -> Vec<VibrationSample> {
        let Some(device) = self.devices.iter().flatten().find(|d| d.handle == handle) else {
            return Vec::new();
        };
        let stored = device.samples.min(VIBRATION_HISTORY as u64) as usize;
        let oldest = if device.samples < VIBRATION_HISTORY as u64 {
            0
        } else {
            device.next
        };
        (0..stored)
            .map(|i| device.history[(oldest + i) % VIBRATION_HISTORY])
            .collect()
    }
}

#[derive(Default)]
struct SixAxis {
    handles: [u32; MAX_SIX_AXIS_SENSORS],
    len: usize,
}

impl SixAxis {
    fn start(&mut self, handle: u32) {
        if self.handles[..self.len].contains(&handle) || self.len == self.handles.len() {
            return;
        }
        self.handles[self.len] = handle;
        self.len += 1;
    }

    #[cfg(test)]
    fn count(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    fn started(&self, handle: u32) -> bool {
        self.handles[..self.len].contains(&handle)
    }
}

#[derive(Default)]
struct State {
    six_axis: SixAxis,
    vibration: Vibration,
    zero_drift: HashMap<u32, u32>,
    touch_output: Option<(u32, u32)>,
}

/// `nn::hid::GyroscopeZeroDriftMode` value set by `ResetGyroscopeZeroDriftMode` and
/// reported by `GetGyroscopeZeroDriftMode` until the guest sets another one.
const ZERO_DRIFT_STANDARD: u32 = 1;

fn activate(device: crate::input::Device) -> Reply {
    crate::input::activate_device(device);
    Reply::ok()
}

/// `ActivateDebugPad` takes a PID and an `AppletResourceUserId` and has no output.
/// The runtime has no debug pad, but the guest may read the shared-memory entry, which
/// `update` already fills with an empty state.
fn activate_debug_pad() -> Reply {
    DEBUG_PAD_ACTIVE.store(true, Relaxed);
    Reply::ok()
}

/// `ActivateTouchScreen` takes a PID and an `AppletResourceUserId` and has no output.
fn activate_touch_screen() -> Reply {
    TOUCH_SCREEN_ACTIVE.store(true, Relaxed);
    Reply::ok()
}

#[cfg(test)]
fn debug_pad_active() -> bool {
    DEBUG_PAD_ACTIVE.load(Relaxed)
}

#[cfg(test)]
fn touch_screen_active() -> bool {
    TOUCH_SCREEN_ACTIVE.load(Relaxed)
}

fn joy_assignment_offset(index: usize) -> u64 {
    NPAD + index as u64 * NPAD_ENTRY + field::JOY_ASSIGNMENT_MODE
}

fn start_lr_assignment() -> Reply {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Relaxed) {
        eprintln!("kirikae: hid: StartLrAssignmentMode; the NPad assignment is left unchanged");
    }
    Reply::ok()
}

fn activate_npad_with_revision(req: &Request, revision: &AtomicU32) -> Reply {
    revision.store(field_u32(&req.data, 0).unwrap_or(0), Relaxed);
    Reply::ok()
}

/// `SetNpadHandheldActivationMode(AppletResourceUserId, s64)`: stores the mode.
/// Only Dual, Single and None are legal; anything else is rejected and the
/// previous mode is kept, because official user processes abort when
/// `GetNpadHandheldActivationMode` reports a value outside 0..=2.
fn set_handheld_activation_mode(mode: i64) -> bool {
    let Ok(mode) = u32::try_from(mode) else {
        return false;
    };
    if mode > HANDHELD_ACTIVATION_MAX {
        return false;
    }
    HANDHELD_ACTIVATION_MODE.store(mode, Relaxed);
    true
}

/// `GetNpadHandheldActivationMode(AppletResourceUserId)`: the stored
/// `NpadHandheldActivationMode` as an s64.
fn handheld_activation_mode() -> u32 {
    HANDHELD_ACTIVATION_MODE.load(Relaxed)
}

fn handheld_activation_mode_reply() -> Reply {
    Reply::ok().u64(handheld_activation_mode() as u64)
}

fn start_six_axis_sensor(req: &Request, n: &Npads) -> Reply {
    if let Some(handle) = field_u32(&req.data, 0)
        && let Ok(mut state) = n.state.lock()
    {
        state.six_axis.start(handle);
    }
    Reply::ok()
}

/// `SetGyroscopeZeroDriftMode(SixAxisSensorHandle, GyroscopeZeroDriftMode,
/// AppletResourceUserId)`: the mode is stored per handle and has no output.
fn set_gyroscope_zero_drift(req: &Request, n: &Npads) -> Reply {
    let handle = field_u32(&req.data, 0).unwrap_or(0);
    let mode = field_u32(&req.data, 4).unwrap_or(ZERO_DRIFT_STANDARD);
    if let Ok(mut state) = n.state.lock() {
        state.zero_drift.insert(handle, mode);
    }
    Reply::ok()
}

/// `GetGyroscopeZeroDriftMode(SixAxisSensorHandle, AppletResourceUserId)`: returns the
/// stored `GyroscopeZeroDriftMode` as a u32. A handle that was never set reports
/// Standard, the value `ResetGyroscopeZeroDriftMode` installs.
fn get_gyroscope_zero_drift(req: &Request, n: &Npads) -> Reply {
    let handle = field_u32(&req.data, 0).unwrap_or(0);
    let mode = n
        .state
        .lock()
        .ok()
        .and_then(|state| state.zero_drift.get(&handle).copied())
        .unwrap_or(ZERO_DRIFT_STANDARD);
    Reply::ok().u32(mode)
}

fn vibration_pairs<'a>(
    handles: &'a [u8],
    values: &'a [u8],
) -> impl Iterator<Item = (u32, f32, f32)> + 'a {
    let count =
        (handles.len() / VIBRATION_DEVICE_HANDLE_SIZE).min(values.len() / VIBRATION_VALUE_SIZE);
    (0..count).map(move |i| {
        let handle = field_u32(handles, i * VIBRATION_DEVICE_HANDLE_SIZE).unwrap_or(0);
        let low = field_f32(values, i * VIBRATION_VALUE_SIZE).unwrap_or(0.0);
        let high = field_f32(values, i * VIBRATION_VALUE_SIZE + 8).unwrap_or(0.0);
        (handle, low, high)
    })
}

fn send_vibration_values(handles: &[u8], values: &[u8], n: &Npads) -> Reply {
    if let Ok(mut state) = n.state.lock() {
        for (handle, low, high) in vibration_pairs(handles, values) {
            state.vibration.record(handle, low, high);
        }
    }
    Reply::ok()
}

impl Npads {
    /// Records the `NpadJoyAssignmentMode` the guest asked for and returns the shared
    /// memory entry that mirrors it.
    fn assign(&self, id: u32, mode: u32) -> Option<usize> {
        let index = npad_index(id)?;
        self.inner.lock().unwrap().assignment[index] = mode;
        Some(index)
    }

    /// `wanted()` minus the ids the guest disconnected with `DisconnectNpad`.
    fn active_wanted(&self) -> Option<(usize, &'static Style)> {
        let wanted = wanted()?;
        let state = self.inner.lock().unwrap();
        (!state.disconnected[wanted.0]).then_some(wanted)
    }

    /// `DisconnectNpad`: the next `update` blanks the shared-memory entry for the
    /// requested `NpadId` and signals the style-set event. A later `ActivateNpad`
    /// reconnects it.
    fn disconnect(&self, req: &Request) -> Option<usize> {
        let index = npad_index(req.u32(0))?;
        self.inner.lock().unwrap().disconnected[index] = true;
        Some(index)
    }

    fn reconnect(&self) {
        self.inner.lock().unwrap().disconnected = [false; 10];
    }

    fn set_hold_type(&self, p: &Process, hold: u32) {
        let mut c = [0u8; 0x10];
        put32(&mut c, 0x8, hold);
        c[0xc] = 1;
        self.memory.write(&p.memory, NPAD_CONDITION, &c);
    }

    /// `SetTouchScreenOutputRanges` ([17.0.0-19.0.1] `SetTouchScreenResolution`):
    /// records the width and height the guest asked for. The synthetic touch states
    /// keep the fixed platform range, so this only reports what the guest declared.
    fn set_touch_output_ranges(&self, width: u32, height: u32) {
        if let Ok(mut state) = self.state.lock() {
            state.touch_output = Some((width, height));
        }
    }

    fn tick(&self, p: &Process) {
        let mut write = |off: u64, bytes: &[u8]| self.memory.write(&p.memory, off, bytes);
        let changed = self.update(&crate::input::state(), self.active_wanted(), &mut write);
        if changed.is_empty() {
            return;
        }
        if let Some((i, s)) = self.inner.lock().unwrap().connected {
            p.warn_once(format!(
                "hid: {} connected as NpadId {:#x}",
                s.name,
                npad_id(i)
            ));
        }
        let events = self.events.lock().unwrap();
        for id in changed {
            if let Some(&e) = events.get(&id) {
                p.set_event(e, true);
            }
        }
    }
}

fn npads(c: &Cpu) -> Arc<Npads> {
    let p = c.process();
    let n = p.service_state("hid.npads", || {
        let (memory, handle) = p.create_shared_memory(SHARED_SIZE);
        Npads {
            memory,
            handle,
            started: AtomicBool::new(false),
            inner: Mutex::default(),
            events: Mutex::default(),
            state: Mutex::default(),
        }
    });
    if !n.started.swap(true, Relaxed) {
        n.tick(p);
        n.set_hold_type(p, HOLD_HORIZONTAL);
        start_updates(p, n.clone());
    }
    n
}

fn start_updates(p: &'static Process, n: Arc<Npads>) {
    let _ = std::thread::Builder::new()
        .name("kirikae-hid".into())
        .spawn(move || {
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
            // 1 ActivateDebugPad(AppletResourceUserId): enables the guest debug pad,
            // no output.
            1 => activate_debug_pad(),
            // 11 ActivateTouchScreen(AppletResourceUserId): enables the guest touch
            // screen, no output.
            11 => activate_touch_screen(),
            // 21 ActivateMouse(AppletResourceUserId): enables the guest mouse, no output.
            21 => activate(crate::input::Device::Mouse),
            // 31 ActivateKeyboard(AppletResourceUserId): enables the guest keyboard, no output.
            31 => activate(crate::input::Device::Keyboard),
            // 66 StartSixAxisSensor(SixAxisSensorHandle, AppletResourceUserId): no output.
            66 => start_six_axis_sensor(req, &npads(c)),
            // 79 SetGyroscopeZeroDriftMode(SixAxisSensorHandle, GyroscopeZeroDriftMode,
            // AppletResourceUserId): no output.
            79 => set_gyroscope_zero_drift(req, &npads(c)),
            // 80 GetGyroscopeZeroDriftMode(SixAxisSensorHandle, AppletResourceUserId):
            // returns the u32 GyroscopeZeroDriftMode.
            80 => get_gyroscope_zero_drift(req, &npads(c)),
            // 107 DisconnectNpad(NpadId, AppletResourceUserId): no output. The synthetic
            // pad for that id stays disconnected until ActivateNpad runs again.
            107 => {
                let id = req.u32(0);
                match npads(c).disconnect(req) {
                    Some(_) => Reply::ok(),
                    None => {
                        c.process()
                            .warn_once(format!("hid: DisconnectNpad for unknown NpadId {id:#x}"));
                        Reply::ok()
                    }
                }
            }
            // 126 StartLrAssignmentMode(AppletResourceUserId): leaves the NPad layout alone.
            126 => start_lr_assignment(),
            // 128 SetNpadHandheldActivationMode(AppletResourceUserId, s64): stores
            // the mode, no output.
            128 => {
                let mode = req.u64(8) as i64;
                if !set_handheld_activation_mode(mode) {
                    c.process().warn_once(format!(
                        "hid: SetNpadHandheldActivationMode got unknown mode {mode}"
                    ));
                }
                Reply::ok()
            }
            // 129 GetNpadHandheldActivationMode(AppletResourceUserId): returns the
            // s64 NpadHandheldActivationMode.
            129 => handheld_activation_mode_reply(),
            // 1004 SetTouchScreenOutputRanges([17.0.0-19.0.1] SetTouchScreenResolution)
            // (width, height, AppletResourceUserId): no output. The synthetic touch
            // screen keeps the platform range, so the values are only recorded.
            1004 => {
                npads(c).set_touch_output_ranges(req.u32(0), req.u32(4));
                Reply::ok()
            }
            40 | 67 | 68 | 69 | 82 | 104 | 122 | 123 | 125 | 130 | 131 | 132 | 133 | 201 | 1000 => {
                crate::services::stub::plain_ok()
            }
            100 => {
                SUPPORTED_STYLES.store(req.u32(0), Relaxed);
                npads(c).tick(c.process());
                Reply::ok()
            }
            101 => Reply::ok().u32(SUPPORTED_STYLES.load(Relaxed)),
            102 => {
                let p = c.process();
                let ids = req
                    .in_buf(0)
                    .and_then(|b| p.memory.read(b.addr, b.size.min(0x100)))
                    .unwrap_or_default();
                let mask = ids
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter_map(|w| npad_index(u32::from_le_bytes(*w)))
                    .fold(0, |m, i| m | 1 << i);
                SUPPORTED_IDS.store(mask, Relaxed);
                npads(c).tick(p);
                Reply::ok()
            }
            // 103 ActivateNpad(AppletResourceUserId): same as ActivateNpadWithRevision
            // with revision 0, no output.
            103 => {
                NPAD_REVISION.store(0, Relaxed);
                npads(c).reconnect();
                Reply::ok()
            }
            106 => {
                let p = c.process();
                let n = npads(c);
                let e = *n
                    .events
                    .lock()
                    .unwrap()
                    .entry(req.u32(0))
                    .or_insert_with(|| p.create_event(true));
                Reply::ok().copy_handle(e)
            }
            108 => {
                const LEDS: [u64; 8] = [0x1, 0x3, 0x7, 0xf, 0x9, 0x5, 0xd, 0x6];
                Reply::ok().u64(LEDS.get(req.u32(0) as usize).copied().unwrap_or(0))
            }
            // 109 ActivateNpadWithRevision(revision, AppletResourceUserId): no output.
            109 => {
                let reply = activate_npad_with_revision(req, &NPAD_REVISION);
                npads(c).reconnect();
                let revision = NPAD_REVISION.load(Relaxed);
                if !NPAD_REVISIONS.contains(&revision) {
                    c.process().warn_once(format!(
                        "hid: ActivateNpadWithRevision got unknown revision {revision}"
                    ));
                }
                reply
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
            // 124 SetNpadJoyAssignmentModeDual(NpadId, AppletResourceUserId): stores the
            // mode and mirrors it in the shared-memory Npad entry, no output.
            124 => {
                let p = c.process();
                let id = req.u32(0);
                let n = npads(c);
                match n.assign(id, JOY_ASSIGNMENT_DUAL) {
                    Some(index) => n.memory.write(
                        &p.memory,
                        joy_assignment_offset(index),
                        &JOY_ASSIGNMENT_DUAL.to_le_bytes(),
                    ),
                    None => p.warn_once(format!(
                        "hid: SetNpadJoyAssignmentModeDual for unknown NpadId {id:#x}"
                    )),
                }
                Reply::ok()
            }
            200 => {
                let device = (req.u32(0) >> 16) & 0xff;
                Reply::ok().u32(1).u32(device.min(1) + 1)
            }
            203 => Reply::ok().object(Arc::new(ActiveVibrationDeviceList)),
            // 206 SendVibrationValues(AppletResourceUserId, handles[], values[]): no output.
            206 => {
                let p = c.process();
                let n = npads(c);
                let (Some(handles), Some(values)) = (req.in_buf(0), req.in_buf(1)) else {
                    return Reply::ok();
                };
                let handles = p
                    .memory
                    .bytes(handles.addr, handles.size.min(VIBRATION_HANDLE_LIMIT))
                    .unwrap_or_default();
                let values = p
                    .memory
                    .bytes(values.addr, values.size.min(VIBRATION_VALUE_LIMIT))
                    .unwrap_or_default();
                let reply = send_vibration_values(handles, values, &n);
                if n.state.lock().map_or(0, |s| s.vibration.dropped) > 0 {
                    p.warn_once(
                        "hid: vibration devices beyond the tracked handles are ignored".into(),
                    );
                }
                reply
            }
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
    use crate::input::button;

    fn u32_at(m: &[u8], off: u64) -> u32 {
        u32::from_le_bytes(m[off as usize..off as usize + 4].try_into().unwrap())
    }

    fn u64_at(m: &[u8], off: u64) -> u64 {
        u64::from_le_bytes(m[off as usize..off as usize + 8].try_into().unwrap())
    }

    fn test_npads() -> Npads {
        Npads {
            memory: Arc::new(SharedMemory::new(SHARED_SIZE)),
            handle: 0,
            started: AtomicBool::new(false),
            inner: Mutex::default(),
            events: Mutex::default(),
            state: Mutex::default(),
        }
    }

    fn request(cmd: u32, data: &[u8]) -> Request {
        Request {
            cmd,
            data: data.to_vec(),
            ..Request::default()
        }
    }

    fn assert_empty_ok(reply: &Reply) {
        assert_eq!(reply.result, 0);
        assert!(
            reply.data.is_empty(),
            "unexpected reply data {:x?}",
            reply.data
        );
        assert!(
            reply.copy_handles.is_empty()
                && reply.move_handles.is_empty()
                && reply.objects.is_empty()
        );
    }

    fn handles(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|h| h.to_le_bytes()).collect()
    }

    fn vibration_value(low: f32, high: f32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&low.to_le_bytes());
        b.extend_from_slice(&160.0f32.to_le_bytes());
        b.extend_from_slice(&high.to_le_bytes());
        b.extend_from_slice(&160.0f32.to_le_bytes());
        b
    }

    #[test]
    fn chooses_pro_controller_unless_unsupported() {
        assert_eq!(
            choose(0, 0, false).map(|(i, s)| (i, s.tag)),
            Some((0, STYLE_FULL_KEY))
        );
        assert_eq!(
            choose(STYLE_HANDHELD | STYLE_JOY_DUAL, 0, false).map(|(i, s)| (i, s.tag)),
            Some((8, STYLE_HANDHELD))
        );
        assert_eq!(
            choose(STYLE_FULL_KEY | STYLE_HANDHELD, 0, true).map(|(i, s)| (i, s.tag)),
            Some((8, STYLE_HANDHELD))
        );
        assert_eq!(
            choose(STYLE_HANDHELD | STYLE_JOY_DUAL, 1, false).map(|(i, s)| (i, s.tag)),
            Some((0, STYLE_JOY_DUAL))
        );
        assert!(choose(1 << 5, 0, false).is_none());
    }

    #[test]
    fn activation_replies_have_no_output_and_keep_the_npad_layout() {
        let before = wanted().map(|(i, s)| (i, s.tag));
        assert_empty_ok(&activate(crate::input::Device::Mouse));
        assert_empty_ok(&activate(crate::input::Device::Keyboard));
        assert!(crate::input::device_active(crate::input::Device::Mouse));
        assert!(crate::input::device_active(crate::input::Device::Keyboard));
        assert_empty_ok(&start_lr_assignment());
        assert_empty_ok(&start_lr_assignment());
        let revision = AtomicU32::new(0);
        assert_empty_ok(&activate_npad_with_revision(
            &request(109, &3u32.to_le_bytes()),
            &revision,
        ));
        assert_eq!(revision.load(Relaxed), 3);
        assert_empty_ok(&activate_npad_with_revision(&request(109, &[]), &revision));
        assert_eq!(revision.load(Relaxed), 0);
        assert_eq!(wanted().map(|(i, s)| (i, s.tag)), before);
    }

    #[test]
    fn debug_pad_and_touch_screen_activation_is_recorded() {
        assert_empty_ok(&activate_debug_pad());
        assert_empty_ok(&activate_touch_screen());
        assert!(debug_pad_active());
        assert!(touch_screen_active());
    }

    #[test]
    fn handheld_activation_mode_defaults_to_dual_and_round_trips() {
        assert_eq!(handheld_activation_mode(), HANDHELD_ACTIVATION_DUAL);
        let r = handheld_activation_mode_reply();
        assert_eq!(r.result, 0);
        assert_eq!(r.data, 0u64.to_le_bytes());

        assert!(set_handheld_activation_mode(1));
        assert_eq!(handheld_activation_mode(), 1);
        assert_eq!(handheld_activation_mode_reply().data, 1u64.to_le_bytes());
        assert!(set_handheld_activation_mode(2));
        assert_eq!(handheld_activation_mode_reply().data, 2u64.to_le_bytes());

        // Out of range: the previous mode is kept so the guest never reads an
        // invalid value back from GetNpadHandheldActivationMode.
        assert!(!set_handheld_activation_mode(3));
        assert!(!set_handheld_activation_mode(-1));
        assert_eq!(handheld_activation_mode_reply().data, 2u64.to_le_bytes());

        assert!(set_handheld_activation_mode(
            HANDHELD_ACTIVATION_DUAL as i64
        ));
        assert_eq!(handheld_activation_mode(), HANDHELD_ACTIVATION_DUAL);
    }

    #[test]
    fn touch_screen_output_ranges_are_recorded_without_output() {
        let n = test_npads();
        assert_eq!(n.state.lock().unwrap().touch_output, None);
        n.set_touch_output_ranges(1280, 720);
        assert_eq!(n.state.lock().unwrap().touch_output, Some((1280, 720)));
        n.set_touch_output_ranges(1920, 1080);
        assert_eq!(n.state.lock().unwrap().touch_output, Some((1920, 1080)));
    }

    #[test]
    fn joy_assignment_mode_dual_is_stored_and_mirrored_in_the_npad_header() {
        let n = test_npads();
        assert_eq!(n.assign(0, JOY_ASSIGNMENT_DUAL), Some(0));
        assert_eq!(n.assign(0x20, JOY_ASSIGNMENT_DUAL), Some(HANDHELD_INDEX));
        assert_eq!(n.assign(0x31, JOY_ASSIGNMENT_DUAL), None);
        assert_eq!(n.inner.lock().unwrap().assignment[0], JOY_ASSIGNMENT_DUAL);

        let mem = std::cell::RefCell::new(vec![0u8; SHARED_SIZE as usize]);
        let mut write = |off: u64, b: &[u8]| {
            mem.borrow_mut()[off as usize..off as usize + b.len()].copy_from_slice(b)
        };
        assert_eq!(
            n.update(&PadState::default(), choose(0, 0, false), &mut write),
            vec![0]
        );
        assert_eq!(
            u32_at(&mem.borrow(), NPAD + field::JOY_ASSIGNMENT_MODE),
            JOY_ASSIGNMENT_DUAL
        );
        assert_eq!(
            u32_at(&mem.borrow(), joy_assignment_offset(HANDHELD_INDEX)),
            JOY_ASSIGNMENT_SINGLE
        );

        assert_eq!(n.update(&PadState::default(), None, &mut write), vec![0]);
        assert_eq!(u32_at(&mem.borrow(), NPAD + field::STYLE_SET), 0);
        assert_eq!(u32_at(&mem.borrow(), NPAD + field::JOY_ASSIGNMENT_MODE), 0);
    }

    #[test]
    fn six_axis_handles_are_tracked_and_bounded() {
        let n = test_npads();
        assert_empty_ok(&start_six_axis_sensor(
            &request(66, &0x0200_0000u32.to_le_bytes()),
            &n,
        ));
        assert_empty_ok(&start_six_axis_sensor(
            &request(66, &0x0200_0000u32.to_le_bytes()),
            &n,
        ));
        {
            let state = n.state.lock().unwrap();
            assert!(state.six_axis.started(0x0200_0000));
            assert_eq!(state.six_axis.count(), 1);
        }
        assert_empty_ok(&start_six_axis_sensor(&request(66, &[]), &n));
        assert_eq!(n.state.lock().unwrap().six_axis.count(), 1);
        for handle in 0..(MAX_SIX_AXIS_SENSORS as u32 + 2) {
            assert_empty_ok(&start_six_axis_sensor(
                &request(66, &handle.to_le_bytes()),
                &n,
            ));
        }
        let state = n.state.lock().unwrap();
        assert_eq!(state.six_axis.count(), MAX_SIX_AXIS_SENSORS);
        assert!(state.six_axis.started(0x0200_0000));
    }

    /// Payloads as the SDK sends them: the struct followed by the message padding up to
    /// the next 16-byte boundary, which is why the request data is 20 bytes long.
    fn padded_payload(first: u32, second: Option<u32>) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&first.to_le_bytes());
        b.extend_from_slice(&second.unwrap_or(0).to_le_bytes());
        b.extend_from_slice(&0x60u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(b.len(), 20);
        b
    }

    #[test]
    fn gyroscope_zero_drift_mode_round_trips_per_handle() {
        let n = test_npads();
        let handle = 0x0002_0003;
        assert_empty_ok(&set_gyroscope_zero_drift(
            &request(79, &padded_payload(handle, Some(2))),
            &n,
        ));
        assert_eq!(
            n.state.lock().unwrap().zero_drift.get(&handle).copied(),
            Some(2)
        );

        let r = get_gyroscope_zero_drift(&request(80, &padded_payload(handle, None)), &n);
        assert_eq!(r.result, 0);
        assert_eq!(r.data, 2u32.to_le_bytes());

        // A handle that was never set reports Standard, the reset value.
        let r = get_gyroscope_zero_drift(&request(80, &padded_payload(0x0002_0004, None)), &n);
        assert_eq!(r.result, 0);
        assert_eq!(r.data, ZERO_DRIFT_STANDARD.to_le_bytes());
    }

    #[test]
    fn disconnect_npad_clears_the_pad_until_activation() {
        let n = test_npads();
        let connected = n
            .active_wanted()
            .map(|(i, _)| i)
            .expect("a synthetic pad is connected");
        let payload = padded_payload(npad_id(connected), None);
        assert_eq!(n.disconnect(&request(107, &payload)), Some(connected));
        assert_eq!(n.active_wanted().map(|(i, _)| i), None);
        n.reconnect();
        assert_eq!(n.active_wanted().map(|(i, _)| i), Some(connected));

        let mut unknown = payload;
        unknown[..4].copy_from_slice(&0x31u32.to_le_bytes());
        assert_eq!(n.disconnect(&request(107, &unknown)), None);
    }

    #[test]
    fn vibration_amplitudes_are_clamped_and_kept_per_device() {
        let n = test_npads();
        assert_empty_ok(&send_vibration_values(
            &handles(&[7]),
            &vibration_value(-0.5, 2.0),
            &n,
        ));
        assert_eq!(
            n.state.lock().unwrap().vibration.latest(7),
            Some(VibrationSample {
                amplitude_low: 0.0,
                amplitude_high: 1.0
            })
        );
        for i in 0..12 {
            let amplitude = i as f32 / 100.0;
            assert_empty_ok(&send_vibration_values(
                &handles(&[7]),
                &vibration_value(amplitude, amplitude),
                &n,
            ));
        }
        let state = n.state.lock().unwrap();
        assert_eq!(state.vibration.samples(7), 13);
        assert_eq!(
            state.vibration.latest(7),
            Some(VibrationSample {
                amplitude_low: 11.0 / 100.0,
                amplitude_high: 11.0 / 100.0
            })
        );
        let history = state.vibration.history(7);
        assert_eq!(history.len(), VIBRATION_HISTORY);
        assert_eq!(history[0].amplitude_low, 4.0 / 100.0);
        assert_eq!(history[VIBRATION_HISTORY - 1].amplitude_low, 11.0 / 100.0);
        assert_eq!(state.vibration.latest(9), None);
    }

    #[test]
    fn vibration_pairs_handles_with_values() {
        let device_handles = handles(&[3, 9]);
        let mut values = vibration_value(0.25, 0.5);
        values.extend_from_slice(&vibration_value(0.75, 1.0));
        let pairs: Vec<_> = vibration_pairs(&device_handles, &values).collect();
        assert_eq!(pairs, vec![(3, 0.25, 0.5), (9, 0.75, 1.0)]);
        assert_eq!(
            vibration_pairs(&device_handles, &values[..VIBRATION_VALUE_SIZE]).collect::<Vec<_>>(),
            vec![(3, 0.25, 0.5)]
        );
        assert!(vibration_pairs(&[], &values).next().is_none());
    }

    #[test]
    fn vibration_devices_are_bounded() {
        let n = test_npads();
        for handle in 0..(VIBRATION_DEVICES as u32 + 1) {
            assert_empty_ok(&send_vibration_values(
                &handles(&[handle]),
                &vibration_value(0.5, 0.5),
                &n,
            ));
        }
        let state = n.state.lock().unwrap();
        assert_eq!(state.vibration.dropped, 1);
        assert!(
            state
                .vibration
                .latest(VIBRATION_DEVICES as u32 - 1)
                .is_some()
        );
        assert!(state.vibration.latest(VIBRATION_DEVICES as u32).is_none());
    }

    #[test]
    fn writes_npad_entry_and_ring_buffers() {
        let n = test_npads();
        let mem = std::cell::RefCell::new(vec![0u8; SHARED_SIZE as usize]);
        let mut write = |off: u64, b: &[u8]| {
            mem.borrow_mut()[off as usize..off as usize + b.len()].copy_from_slice(b)
        };
        let pad = PadState {
            buttons: button::A,
            left: (100, -200),
            right: (0, 0),
        };
        assert_eq!(n.update(&pad, choose(0, 0, false), &mut write), vec![0]);
        for _ in 0..20 {
            assert!(n.update(&pad, choose(0, 0, false), &mut write).is_empty());
        }
        let e = NPAD;
        assert_eq!(u32_at(&mem.borrow(), e + field::STYLE_SET), STYLE_FULL_KEY);
        assert_eq!(u32_at(&mem.borrow(), e + field::DEVICE_TYPE), 1);
        assert_eq!(
            mem.borrow()[(e + field::APPLET_FOOTER_UI_TYPE) as usize],
            12
        );
        let lifo = e + field::LIFOS;
        assert_eq!(u64_at(&mem.borrow(), lifo + 0x8), 17);
        let (tail, count) = (
            u64_at(&mem.borrow(), lifo + 0x10),
            u64_at(&mem.borrow(), lifo + 0x18),
        );
        assert_eq!((tail, count), (21 % 17, 16));
        let s = lifo + 0x20 + tail * COMMON_STORAGE;
        assert_eq!(u64_at(&mem.borrow(), s), 42);
        assert_eq!(u64_at(&mem.borrow(), s + 0x8), 21);
        assert_eq!(u64_at(&mem.borrow(), s + 0x10), button::A);
        assert_eq!(u32_at(&mem.borrow(), s + 0x18) as i32, 100);
        assert_eq!(u32_at(&mem.borrow(), s + 0x1c) as i32, -200);
        assert_eq!(u32_at(&mem.borrow(), s + 0x28), 0x3);
        let h = e + field::LIFOS + field::LIFO_SIZE + 0x20 + tail * COMMON_STORAGE;
        assert_eq!(
            (u64_at(&mem.borrow(), h), u64_at(&mem.borrow(), h + 0x10)),
            (42, 0)
        );
        let changed = n.update(&pad, choose(0, 0, true), &mut write);
        assert_eq!(changed, vec![0, 0x20]);
        assert_eq!(u32_at(&mem.borrow(), e + field::STYLE_SET), 0);
        let hh = NPAD + 8 * NPAD_ENTRY;
        assert_eq!(u32_at(&mem.borrow(), hh + field::STYLE_SET), STYLE_HANDHELD);
        let hl = hh + field::LIFOS + field::LIFO_SIZE;
        let t = u64_at(&mem.borrow(), hl + 0x10);
        assert_eq!(
            u64_at(&mem.borrow(), hl + 0x20 + t * COMMON_STORAGE + 0x10),
            button::A
        );
    }
}
