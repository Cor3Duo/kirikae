use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::display::{self, Frame};
use crate::host1x::clock::{self, Timestamp};
use crate::process::Process;
use crate::services::stub::{Stub, unimplemented};
use crate::services::{Reply, Request, Service};

/// Display resolution, kept in sync with `am/ICommonStateGetter` command 60.
const DISPLAY_WIDTH: u32 = 1920;
const DISPLAY_HEIGHT: u32 = 1080;
/// Layer size reported before the guest has provided a buffer.
const FALLBACK_WIDTH: u32 = 1280;
const FALLBACK_HEIGHT: u32 = 720;
const BINDER_ID: i32 = 1;
const SLOTS: usize = 16;
/// Display id for the console's built-in panel.
const DEFAULT_DISPLAY: u64 = 1;
/// Layer id `am/ISelfController` command 40 and `CreateStrayLayer` hand to the application;
/// the present path tracks its size in [`State::layer`].
const PRESENTED_LAYER: u64 = 1;
/// Z-order range the display stack advertises. The runtime records the value the guest sets
/// instead of clamping it.
const LAYER_Z_MIN: i64 = 0;
const LAYER_Z_MAX: i64 = 0xFF;
/// Underscan is stubbed on retail units with this result.
const RESULT_UNDERSCAN_STUBBED: u32 = 0xC72;

const LAYER_PRESENTING_EVENT: usize = 0;
const DISPLAY_HOTPLUG_EVENT: usize = 1;
const DISPLAY_MODE_CHANGED_EVENT: usize = 2;
const DISPLAY_ERROR_EVENT: usize = 3;
const DISPLAY_FATAL_ERROR_EVENT: usize = 4;

pub struct Vi {
    pub name: &'static str,
}

impl Service for Vi {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match (self.name, req.cmd) {
            ("vi:u", 0) | ("vi:s", 1) | ("vi:m", 2) => Reply::ok().object(Arc::new(DisplayService)),
            // GetDisplayServiceWithProxyNameExchange on the system and manager roots.
            ("vi:s", 3) | ("vi:m", 3) => Reply::ok().object(Arc::new(DisplayService)),
            _ => unimplemented(self.name, c, req),
        }
    }
}

/// Geometry of a plane that is not the first one.
///
/// The guest's `NvGraphicBuffer` carries an `NvSurface` per plane; the second
/// and third entries describe where the chroma planes live and how they are
/// laid out. A packed buffer (the common NV12 case) leaves them zero and the
/// decoder falls back to the plane following the luma one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PlaneGeometry {
    offset: u64,
    pitch: u32,
    block_linear: bool,
    block_height_log2: u32,
}

/// Rectangle of a queued buffer the guest wants shown, in buffer pixels.
///
/// Android's `QueueBufferInput` carries a crop `Rect`; NVDEC writes coded,
/// macroblock-aligned surfaces (1920x1088 for 1080p, 912x512 for 900x506), so a
/// YUV layer uses this rectangle to leave the padding out of the present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Default)]
struct Slot {
    flattened: Vec<u8>,
    nvmap: u32,
    offset: u64,
    width: u32,
    height: u32,
    format: u32,
    pitch: u32,
    block_linear: bool,
    block_height_log2: u32,
    /// Chroma planes, when the guest describes them separately.
    chroma: [PlaneGeometry; 2],
    /// Visible rectangle inside the buffer, from the queue input.
    crop: Option<Rect>,
}

#[derive(Default)]
struct Queue {
    slots: Vec<Option<Slot>>,
    next: usize,
}

struct State {
    queue: Mutex<Queue>,
    buffer_event: u32,
    vsync_event: u32,
    vsync_started: AtomicBool,
    layer: AtomicU64,
    managers: Mutex<Managers>,
    display_events: Mutex<[Option<u32>; 5]>,
}

fn state(c: &Cpu) -> Arc<State> {
    let p = c.process();
    p.service_state("vi", || State {
        queue: Mutex::new(Queue {
            slots: vec![None; SLOTS],
            next: 0,
        }),
        buffer_event: p.create_event(true),
        vsync_event: p.create_event(false),
        vsync_started: AtomicBool::new(false),
        layer: AtomicU64::new(0),
        managers: Mutex::new(Managers::new()),
        display_events: Mutex::new([None; 5]),
    })
}

// ------------------------------------------------------------------ layer and display model

/// An `nn::vi::Display` as the manager services see it.
#[derive(Clone, Debug)]
struct DisplayState {
    id: u64,
    name: String,
    alpha: f32,
    /// Layer stack the display composes.
    stack: u64,
    /// `nn::vi::PowerState`; the panel starts on.
    power: u32,
    mode: u32,
    magnification: [i32; 4],
}

impl DisplayState {
    fn new(id: u64, name: &str) -> DisplayState {
        DisplayState {
            id,
            name: name.to_string(),
            alpha: 1.0,
            stack: 0,
            power: 1,
            mode: 0,
            magnification: [0, 0, DISPLAY_WIDTH as i32, DISPLAY_HEIGHT as i32],
        }
    }
}

/// A `nn::vi::Layer` owned by the manager services.
///
/// Layers created here share the id space with the present path: [`PRESENTED_LAYER`] is the
/// layer `am/ISelfController` 40 and `CreateStrayLayer` hand out, so a manager resize of it is
/// mirrored into [`State::layer`].
#[derive(Clone, Debug)]
struct LayerState {
    id: u64,
    display: u64,
    stack: u64,
    visible: bool,
    z: i64,
    x: f32,
    y: f32,
    width: i64,
    height: i64,
    alpha: f32,
}

impl LayerState {
    fn new(id: u64, display: u64) -> LayerState {
        LayerState {
            id,
            display,
            stack: 0,
            visible: true,
            z: 0,
            x: 0.0,
            y: 0.0,
            width: FALLBACK_WIDTH as i64,
            height: FALLBACK_HEIGHT as i64,
            alpha: 1.0,
        }
    }
}

/// Display and layer bookkeeping shared by `ISystemDisplayService` and
/// `IManagerDisplayService`. The present path never reads it; it is what the guest reads back
/// (z, size, opacity) and what a later compositor will consume.
struct Managers {
    displays: Vec<DisplayState>,
    layers: Vec<LayerState>,
    next_layer: u64,
    default_display: u64,
    content_visible: bool,
    conductor_layer: u64,
}

impl Managers {
    fn new() -> Managers {
        Managers {
            displays: vec![DisplayState::new(DEFAULT_DISPLAY, "Default")],
            layers: vec![LayerState::new(PRESENTED_LAYER, DEFAULT_DISPLAY)],
            next_layer: PRESENTED_LAYER + 1,
            default_display: DEFAULT_DISPLAY,
            content_visible: true,
            conductor_layer: 0,
        }
    }

    fn display(&self, id: u64) -> Option<&DisplayState> {
        self.displays.iter().find(|d| d.id == id)
    }

    fn display_mut(&mut self, id: u64) -> Option<&mut DisplayState> {
        self.displays.iter_mut().find(|d| d.id == id)
    }

    /// Displays in id order, as `ListDisplays` reports them.
    #[cfg(test)]
    fn enumerate(&self) -> &[DisplayState] {
        &self.displays
    }

    /// Allocates the next layer id. An unknown display falls back to the default one: a title
    /// that passes a stale id keeps a usable layer instead of failing every later present.
    fn create_layer(&mut self, display: u64) -> u64 {
        let display = if self.display(display).is_some() {
            display
        } else {
            self.default_display
        };
        let id = self.next_layer;
        self.next_layer += 1;
        self.layers.push(LayerState::new(id, display));
        id
    }

    fn destroy_layer(&mut self, id: u64) -> bool {
        let Some(at) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        self.layers.remove(at);
        true
    }

    fn layer(&self, id: u64) -> Option<&LayerState> {
        self.layers.iter().find(|l| l.id == id)
    }

    fn layer_mut(&mut self, id: u64) -> Option<&mut LayerState> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    fn set_display_alpha(&mut self, id: u64, alpha: f32) -> bool {
        match self.display_mut(id) {
            Some(display) => {
                display.alpha = alpha;
                true
            }
            None => false,
        }
    }

    fn set_display_stack(&mut self, id: u64, stack: u64) -> bool {
        match self.display_mut(id) {
            Some(display) => {
                display.stack = stack;
                true
            }
            None => false,
        }
    }

    fn set_display_power(&mut self, id: u64, power: u32) -> bool {
        match self.display_mut(id) {
            Some(display) => {
                display.power = power;
                true
            }
            None => false,
        }
    }

    fn set_default_display(&mut self, id: u64) -> bool {
        if self.display(id).is_none() {
            return false;
        }
        self.default_display = id;
        true
    }
}

fn input_bool(data: &[u8], off: usize) -> bool {
    data.get(off).is_some_and(|&b| b != 0)
}

fn input_f32(data: &[u8], off: usize) -> Option<f32> {
    let bytes: [u8; 4] = data.get(off..off + 4)?.try_into().ok()?;
    Some(f32::from_le_bytes(bytes))
}

/// Event owned by the display manager. `ISystemDisplayService` and `IManagerDisplayService`
/// hand the same handle to every caller of a given event command.
fn manager_event(c: &Cpu, st: &State, which: usize) -> u32 {
    let mut slots = st.display_events.lock().unwrap();
    *slots[which].get_or_insert_with(|| c.process().create_event(false))
}

/// Which manager event a command asks for. `ISystemDisplayService` has no events;
/// `IManagerDisplayService` has the four panel/layer notifications.
fn manager_event_index(cmd: u32) -> Option<usize> {
    match cmd {
        2300 => Some(LAYER_PRESENTING_EVENT),
        2302 => Some(DISPLAY_HOTPLUG_EVENT),
        2303 => Some(DISPLAY_MODE_CHANGED_EVENT),
        2601 => Some(DISPLAY_ERROR_EVENT),
        2701 => Some(DISPLAY_FATAL_ERROR_EVENT),
        _ => None,
    }
}

/// Mirrors a manager resize of the presented layer into the size the present path reports.
fn sync_presented_size(slot: &AtomicU64, id: u64, width: i64, height: i64) {
    if id == PRESENTED_LAYER && width > 0 && height > 0 {
        slot.store(pack_layer(width as u32, height as u32), Ordering::Relaxed);
    }
}

/// Size of the layer the guest is producing. Before the first buffer is
/// configured this falls back to 720p, which is also the size the host window
/// starts at.
fn layer_size(st: &State) -> (u32, u32) {
    unpack_layer(st.layer.load(Ordering::Relaxed))
}

fn unpack_layer(packed: u64) -> (u32, u32) {
    if packed == 0 {
        (FALLBACK_WIDTH, FALLBACK_HEIGHT)
    } else {
        ((packed >> 32) as u32, packed as u32)
    }
}

fn set_layer_size(st: &State, width: u32, height: u32) {
    st.layer.store(pack_layer(width, height), Ordering::Relaxed);
}

fn pack_layer(width: u32, height: u32) -> u64 {
    (width as u64) << 32 | height as u64
}

fn start_vsync(p: &'static Process, st: &State) {
    if st.vsync_started.swap(true, Ordering::Relaxed) {
        return;
    }
    let event = st.vsync_event;
    let _ = std::thread::Builder::new()
        .name("kirikae-vsync".into())
        .spawn(move || {
            let period = std::time::Duration::from_nanos(1_000_000_000 / 60);
            let epoch = *VSYNC_EPOCH.get_or_init(std::time::Instant::now);
            let past = (epoch.elapsed().as_nanos() / period.as_nanos()) as u32;
            let mut next = epoch + period * past;
            loop {
                next += period;
                std::thread::sleep(next.saturating_duration_since(std::time::Instant::now()));
                if p.sched.exiting.load(Ordering::Relaxed) {
                    return;
                }
                p.set_event(event, true);
            }
        });
}

struct DisplayService;

impl Service for DisplayService {
    fn name(&self) -> String {
        "vi/IApplicationDisplayService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            100 | 103 => Reply::ok().object(Arc::new(Binder)),
            101 => Reply::ok().object(Arc::new(SystemDisplay)),
            102 => Reply::ok().object(Arc::new(ManagerDisplay)),
            1000 => {
                if let Some(b) = req.out_buf(0) {
                    let mut info = [0u8; 0x60];
                    info[..7].copy_from_slice(b"Default");
                    info[0x40] = 1;
                    info[0x48..0x50].copy_from_slice(&1u64.to_le_bytes());
                    info[0x50..0x58].copy_from_slice(&(DISPLAY_WIDTH as u64).to_le_bytes());
                    info[0x58..0x60].copy_from_slice(&(DISPLAY_HEIGHT as u64).to_le_bytes());
                    p.memory.write(b.addr, &info[..(b.size as usize).min(0x60)]);
                }
                Reply::ok().u64(1)
            }
            1010 | 1011 => Reply::ok().u64(0),
            1020 | 1101 | 2021 | 2031 | 2101 => Reply::ok(),
            1102 => Reply::ok()
                .u64(DISPLAY_WIDTH as u64)
                .u64(DISPLAY_HEIGHT as u64),
            2020 | 2030 => {
                let parcel = native_window();
                if let Some(b) = req.out_buf(0) {
                    p.memory
                        .write(b.addr, &parcel[..parcel.len().min(b.size as usize)]);
                }
                let title = format!("kirikae - {:016x}", p.title_id);
                display::start(p, title);
                start_vsync(p, &state(c));
                p.log_ipc(format!("vi: layer opened (cmd {})", req.cmd));
                let r = Reply::ok();
                let r = if req.cmd == 2030 { r.u64(1) } else { r };
                r.u64(parcel.len() as u64)
            }
            2102 => Reply::ok().u64(req.u32(0).min(4) as u64),
            5202 | 5203 => Reply::ok().copy_handle(state(c).vsync_event),
            _ => unimplemented("vi/IApplicationDisplayService", c, req),
        }
    }
}

// ------------------------------------------------------------------ manager services

/// `nn::visrv::sf::ISystemDisplayService`, returned by
/// `IApplicationDisplayService::GetSystemDisplayService` (cmd 101).
struct SystemDisplay;

impl SystemDisplay {
    fn command(&self, c: &Cpu, req: &Request) -> Option<Reply> {
        let st = state(c);
        let managers = &st.managers;
        let reply = match req.cmd {
            1200 => Reply::ok().u64(LAYER_Z_MIN as u64),
            1202 => Reply::ok().u64(LAYER_Z_MAX as u64),
            // `GetDisplayLogicalResolution` reports a pair of s32.
            1203 => Reply::ok().u32(DISPLAY_WIDTH).u32(DISPLAY_HEIGHT),
            1204 => {
                let [x, y, width, height] = [req.u32(0), req.u32(4), req.u32(8), req.u32(0xc)];
                if let Some(display) = managers.lock().unwrap().display_mut(req.u64(0x10)) {
                    display.magnification = [x as i32, y as i32, width as i32, height as i32];
                }
                Reply::ok()
            }
            // `SetLayerPosition` carries the floats first and the layer id last.
            2201 => {
                let (x, y, id) = (
                    input_f32(&req.data, 0).unwrap_or(0.0),
                    input_f32(&req.data, 4).unwrap_or(0.0),
                    req.u64(8),
                );
                if let Some(layer) = managers.lock().unwrap().layer_mut(id) {
                    layer.x = x;
                    layer.y = y;
                }
                Reply::ok()
            }
            2203 => {
                let (id, width, height) = (req.u64(0), req.u64(8) as i64, req.u64(0x10) as i64);
                if let Some(layer) = managers.lock().unwrap().layer_mut(id) {
                    layer.width = width;
                    layer.height = height;
                }
                sync_presented_size(&st.layer, id, width, height);
                Reply::ok()
            }
            2204 => {
                let z = managers
                    .lock()
                    .unwrap()
                    .layer(req.u64(0))
                    .map_or(0, |layer| layer.z);
                Reply::ok().u64(z as u64)
            }
            2205 => {
                if let Some(layer) = managers.lock().unwrap().layer_mut(req.u64(0)) {
                    layer.z = req.u64(8) as i64;
                }
                Reply::ok()
            }
            2207 => {
                if let Some(layer) = managers.lock().unwrap().layer_mut(req.u64(0)) {
                    layer.visible = input_bool(&req.data, 8);
                }
                Reply::ok()
            }
            2209 => {
                if let Some(layer) = managers.lock().unwrap().layer_mut(req.u64(0)) {
                    layer.alpha = input_f32(&req.data, 8).unwrap_or(0.0);
                }
                Reply::ok()
            }
            2210 => {
                let (id, x, y, width, height) = (
                    req.u64(0),
                    req.u64(8) as i64,
                    req.u64(0x10) as i64,
                    req.u64(0x18) as i64,
                    req.u64(0x20) as i64,
                );
                {
                    let mut managers = managers.lock().unwrap();
                    if let Some(layer) = managers.layer_mut(id) {
                        layer.x = x as f32;
                        layer.y = y as f32;
                        layer.width = width;
                        layer.height = height;
                    }
                }
                sync_presented_size(&st.layer, id, width, height);
                Reply::ok()
            }
            // `GetDisplayMode`: width, height, refresh rate and an unused word.
            3200 => Reply::ok()
                .u32(DISPLAY_WIDTH)
                .u32(DISPLAY_HEIGHT)
                .u32(60.0f32.to_bits())
                .u32(0),
            3201 => {
                if let Some(display) = managers.lock().unwrap().display_mut(req.u64(0)) {
                    display.mode = req.u32(8);
                }
                Reply::ok()
            }
            3202 | 3203 => Reply::err(RESULT_UNDERSCAN_STUBBED),
            9000 => Reply::ok().object(Arc::new(Stub::new("vi/IDp2hdmiController"))),
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for SystemDisplay {
    fn name(&self) -> String {
        "vi/ISystemDisplayService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(c, req) {
            Some(reply) => reply,
            None => unimplemented("vi/ISystemDisplayService", c, req),
        }
    }
}

/// `nn::visrv::sf::IManagerDisplayService`, returned by
/// `IApplicationDisplayService::GetManagerDisplayService` (cmd 102).
struct ManagerDisplay;

impl ManagerDisplay {
    /// Manager events only hand out handles, so they do not need the layer table.
    fn event(&self, c: &Cpu, req: &Request) -> Option<Reply> {
        let which = manager_event_index(req.cmd)?;
        let st = state(c);
        Some(Reply::ok().copy_handle(manager_event(c, &st, which)))
    }

    fn command(&self, c: &Cpu, req: &Request) -> Option<Reply> {
        if let Some(reply) = self.event(c, req) {
            return Some(reply);
        }
        let st = state(c);
        let managers = &st.managers;
        let reply = match req.cmd {
            1102 => Reply::ok()
                .u64(DISPLAY_WIDTH as u64)
                .u64(DISPLAY_HEIGHT as u64),
            // `CreateManagedLayer`: flags, then the display, then the applet resource user id.
            2010 | 2012 | 2050 => {
                let display = if req.cmd == 2050 {
                    DEFAULT_DISPLAY
                } else {
                    req.u64(8)
                };
                let mut managers = managers.lock().unwrap();
                let id = managers.create_layer(display);
                let display = managers.layer(id).map_or(display, |layer| layer.display);
                drop(managers);
                c.process()
                    .log_ipc(format!("vi: layer {id} created on display {display}"));
                Reply::ok().u64(id)
            }
            2011 | 2051 | 2053 | 2055 => {
                managers.lock().unwrap().destroy_layer(req.u64(0));
                Reply::ok()
            }
            2301 | 4207 | 4208 | 8001 | 8100 => Reply::ok(),
            // The panel is always plugged in and on.
            2402 => Reply::ok().u32(1),
            4201 => {
                let (alpha, id) = (input_f32(&req.data, 0).unwrap_or(0.0), req.u64(8));
                managers.lock().unwrap().set_display_alpha(id, alpha);
                Reply::ok()
            }
            // `SetDisplayLayerStack` and `SetDisplayPowerState` carry a small value first.
            4203 => {
                let (stack, id) = (req.u64(0), req.u64(8));
                managers.lock().unwrap().set_display_stack(id, stack);
                Reply::ok()
            }
            4205 => {
                let (power, id) = (req.u32(0), req.u64(8));
                managers.lock().unwrap().set_display_power(id, power);
                Reply::ok()
            }
            4206 => {
                let id = req.u64(0);
                let mut managers = managers.lock().unwrap();
                if managers.set_default_display(id)
                    && let Some(display) = managers.display(id)
                {
                    c.process()
                        .log_ipc(format!("vi: default display is {} ({id})", display.name));
                }
                Reply::ok()
            }
            4209 => Reply::ok().u8(1),
            4300 => Reply::ok().u64(DEFAULT_DISPLAY),
            // `AddToLayerStack`/`RemoveFromLayerStack` carry the layer, then the stack.
            6000 => {
                let (layer, stack) = (req.u64(0), req.u64(8));
                if let Some(layer) = managers.lock().unwrap().layer_mut(layer) {
                    layer.stack = stack;
                }
                Reply::ok()
            }
            6001 => {
                let (id, stack) = (req.u64(0), req.u64(8));
                if let Some(layer) = managers.lock().unwrap().layer_mut(id)
                    && layer.stack == stack
                {
                    layer.stack = 0;
                }
                Reply::ok()
            }
            6002 => {
                if let Some(layer) = managers.lock().unwrap().layer_mut(req.u64(0)) {
                    layer.visible = input_bool(&req.data, 8);
                }
                Reply::ok()
            }
            6013 => {
                if let Some(layer) = managers.lock().unwrap().layer_mut(req.u64(0)) {
                    layer.alpha = input_f32(&req.data, 8).unwrap_or(0.0);
                }
                Reply::ok()
            }
            7000 => {
                managers.lock().unwrap().content_visible = input_bool(&req.data, 0);
                Reply::ok()
            }
            8000 => {
                let id = req.u64(0);
                managers.lock().unwrap().conductor_layer = id;
                Reply::ok()
            }
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for ManagerDisplay {
    fn name(&self) -> String {
        "vi/IManagerDisplayService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(c, req) {
            Some(reply) => reply,
            None => unimplemented("vi/IManagerDisplayService", c, req),
        }
    }
}

fn native_window() -> Vec<u8> {
    let mut w = ParcelWriter::default();
    w.i32(2);
    w.i32(0);
    w.i64(BINDER_ID as i64);
    w.i64(0);
    w.bytes(b"dispdrv\0");
    w.bytes(&[0; 8]);
    w.finish_with_objects(&[0])
}

struct Binder;

impl Service for Binder {
    fn name(&self) -> String {
        "vi/IHOSBinderDriver".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 | 3 => {
                let (id, code) = (req.u32(0) as i32, req.u32(4));
                let input = req
                    .in_buf(0)
                    .and_then(|b| p.memory.read(b.addr, b.size))
                    .unwrap_or_default();
                let reply = transact(c, id, code, &input);
                if let Some(b) = req.out_buf(0) {
                    p.memory
                        .write(b.addr, &reply[..reply.len().min(b.size as usize)]);
                }
                Reply::ok()
            }
            1 => Reply::ok(),
            2 => Reply::ok().copy_handle(state(c).buffer_event),
            _ => unimplemented("vi/IHOSBinderDriver", c, req),
        }
    }
}

const REQUEST_BUFFER: u32 = 1;
const SET_BUFFER_COUNT: u32 = 2;
const DEQUEUE_BUFFER: u32 = 3;
const QUEUE_BUFFER: u32 = 7;
const CANCEL_BUFFER: u32 = 8;
const QUERY: u32 = 9;
const CONNECT: u32 = 10;
const DISCONNECT: u32 = 11;
const SET_PREALLOCATED_BUFFER: u32 = 14;

fn transact(c: &Cpu, id: i32, code: u32, input: &[u8]) -> Vec<u8> {
    let p = c.process();
    let st = state(c);
    let mut r = ParcelReader::new(input);
    r.skip_interface_token();
    let mut w = ParcelWriter::default();
    if id != BINDER_ID {
        p.warn_once(format!("vi: transaction {code} to unknown binder {id}"));
    }
    let known = [
        REQUEST_BUFFER,
        SET_BUFFER_COUNT,
        DEQUEUE_BUFFER,
        QUEUE_BUFFER,
        CANCEL_BUFFER,
        QUERY,
        CONNECT,
        DISCONNECT,
        SET_PREALLOCATED_BUFFER,
    ];
    if known.contains(&code) {
        crate::usage::command(
            c,
            "vi/IGraphicBufferProducer",
            code as u64,
            crate::usage::Status::Implemented,
            "",
            input,
        );
    }
    match code {
        REQUEST_BUFFER => {
            let slot = r.i32() as usize;
            let q = st.queue.lock().unwrap();
            match q.slots.get(slot).and_then(|s| s.as_ref()) {
                Some(s) => {
                    w.i32(1);
                    w.bytes(&s.flattened);
                }
                None => w.i32(0),
            }
        }
        SET_BUFFER_COUNT | DISCONNECT => {}
        DEQUEUE_BUFFER => {
            let mut q = st.queue.lock().unwrap();
            let filled: Vec<usize> = (0..SLOTS).filter(|&i| q.slots[i].is_some()).collect();
            let slot = if filled.is_empty() {
                0
            } else {
                filled[q.next % filled.len()]
            };
            q.next += 1;
            w.i32(slot as i32);
            w.i32(1);
            w.i32(0x24);
            w.i32(0);
            w.bytes(&[0; 0x24]);
        }
        QUEUE_BUFFER => {
            let slot = r.i32() as usize;
            let raw = r.flattenable();
            // One line per run so a trace shows how the guest fills the crop
            // rectangle and the transform, even when the whole buffer is shown.
            static FIRST_QUEUE: std::sync::Once = std::sync::Once::new();
            if crate::host1x::trace() {
                FIRST_QUEUE.call_once(|| {
                    if let Some(raw) = raw.as_deref() {
                        let word = |at: usize| {
                            raw.get(at..at + 4)
                                .and_then(|b| <[u8; 4]>::try_from(b).ok())
                                .map(i32::from_le_bytes)
                                .unwrap_or(0)
                        };
                        eprintln!(
                            "vi: first queue input crop ({},{},{},{}) transform {:#x}",
                            word(20),
                            word(24),
                            word(28),
                            word(32),
                            word(40) as u32
                        );
                    }
                });
            }
            let timestamp = queue_timestamp(raw.as_deref());
            let word = |at: usize| {
                raw.as_ref()
                    .and_then(|raw| raw.get(at..at + 4))
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            };
            let transform = word(40).unwrap_or(0);
            let asked = word(52);
            let interval = asked.unwrap_or(1).min(4);
            static LAST_ASKED: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(u32::MAX);
            if let Some(a) = asked
                && LAST_ASKED.swap(a, Ordering::Relaxed) != a
                && std::env::var_os("KIRIKAE_STATS").is_some()
            {
                eprintln!(
                    "kirikae: vi: swap interval {a} from frame {}",
                    crate::display::FRAMES.load(Ordering::Relaxed)
                );
            }
            let mut s = st.queue.lock().unwrap().slots.get(slot).cloned().flatten();
            if let Some(slot_data) = &mut s {
                slot_data.crop = queue_crop(raw.as_deref(), slot_data.width, slot_data.height);
                if crate::host1x::trace()
                    && let Some(crop) = slot_data.crop
                {
                    eprintln!(
                        "vi: slot {slot} crop {}x{} at {},{} of {}x{} format {}",
                        crop.width,
                        crop.height,
                        crop.x,
                        crop.y,
                        slot_data.width,
                        slot_data.height,
                        slot_data.format
                    );
                }
            }
            let size = match &s {
                Some(s) => {
                    let size = s.visible();
                    set_layer_size(&st, size.0, size.1);
                    size
                }
                None => layer_size(&st),
            };
            match s {
                Some(s) => {
                    let yuv = s.is_video_format();
                    let video = is_video_layer(p, &s);
                    if yuv {
                        clock::yuv_queued();
                    }
                    crate::sched::without_gil(c, || {
                        let prof = crate::prof::enter(crate::prof::Cat::Present);
                        let shown = if video { clock::gate(timestamp) } else { true };
                        if crate::host1x::trace() && video {
                            eprintln!(
                                "vi: video queue slot {slot} {}x{} format {} shown {shown}",
                                s.width, s.height, s.format
                            );
                        }
                        let presented = if shown {
                            present(p, &s, transform)
                        } else {
                            false
                        };
                        drop(prof);
                        // A gated video frame already waited for its presentation
                        // time; adding the generic 60 Hz limiter would make it
                        // late. Only non-video layers keep the old pacing.
                        if shown && !video {
                            let _prof = crate::prof::enter(crate::prof::Cat::Sleep);
                            pace(crate::games::pace_interval(p, interval));
                        }
                        if yuv {
                            if !shown {
                                clock::yuv_dropped();
                            } else if presented {
                                clock::yuv_presented();
                            } else {
                                clock::yuv_failed();
                            }
                        }
                        if shown {
                            crate::games::presented(p);
                            crate::gpu::frame_presented();
                        }
                    })
                }
                None => p.warn_once(format!("vi: queued slot {slot} has no buffer")),
            }
            p.set_event(st.buffer_event, true);
            queue_buffer_output(&mut w, size.0, size.1);
        }
        CANCEL_BUFFER => {}
        QUERY => {
            let (width, height) = layer_size(&st);
            let v = match r.i32() {
                0 => width as i32,
                1 => height as i32,
                2 => 1,
                3 => 1,
                _ => 0,
            };
            w.i32(v);
        }
        CONNECT => {
            let (width, height) = layer_size(&st);
            queue_buffer_output(&mut w, width, height);
        }
        SET_PREALLOCATED_BUFFER => {
            let slot = r.i32() as usize;
            let present = r.i32() != 0;
            let parsed = if present {
                r.flattenable().map(parse_graphic_buffer)
            } else {
                None
            };
            if let Some(s) = &parsed {
                set_layer_size(&st, s.width, s.height);
                p.log_ipc(format!(
                    "vi: slot {slot}: {}x{} format {} pitch {} {} (block height {}) at nvmap {} + {:#x}",
                    s.width,
                    s.height,
                    s.format,
                    s.pitch,
                    if s.block_linear { "block linear" } else { "pitch linear" },
                    1 << s.block_height_log2,
                    s.nvmap,
                    s.offset
                ));
            }
            if let Some(entry) = st.queue.lock().unwrap().slots.get_mut(slot) {
                *entry = parsed;
            }
        }
        _ => {
            p.warn_once(format!(
                "vi: IGraphicBufferProducer transaction {code} is stubbed"
            ));
            crate::usage::command(
                c,
                "vi/IGraphicBufferProducer",
                code as u64,
                crate::usage::Status::Stub,
                "unimplemented",
                input,
            );
        }
    }
    w.i32(0);
    w.finish()
}

fn queue_buffer_output(w: &mut ParcelWriter, width: u32, height: u32) {
    w.i32(width as i32);
    w.i32(height as i32);
    w.i32(0);
    w.i32(0);
}

/// Reads the presentation timestamp from an Android `QueueBufferInput`.
///
/// The first field is the buffer timestamp in nanoseconds; `isAutoTimestamp`
/// follows it. The guest uses `-1` to ask the compositor for a timestamp, and
/// titles that never fill the field leave it at zero. Both cases fall back to
/// the clock's synthesized cadence: using zero for every frame would look like
/// a permanently starved decoder.
fn queue_timestamp(raw: Option<&[u8]>) -> Timestamp {
    let Some(raw) = raw else {
        return Timestamp::Automatic;
    };
    let timestamp = raw
        .get(8..16)
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
        .map_or(-1, i64::from_le_bytes);
    let automatic = raw
        .get(16..20)
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .is_none_or(|b| u32::from_le_bytes(b) != 0);
    if automatic || timestamp <= 0 {
        Timestamp::Automatic
    } else {
        Timestamp::Guest(timestamp)
    }
}

/// Reads the crop rectangle from an Android `QueueBufferInput`.
///
/// The flattened input starts with the timestamp (payload 0), `isAutoTimestamp`
/// (8), the crop `Rect` (12: left, top, right, bottom) and `scalingMode` (28);
/// `transform` follows at 32. The 8-byte flattenable header shifts all of those
/// by 8 in the raw parcel bytes. An empty, inverted or out-of-buffer rectangle
/// means "whole buffer", which is also what an untouched crop holds.
fn queue_crop(raw: Option<&[u8]>, width: u32, height: u32) -> Option<Rect> {
    let raw = raw?;
    let word = |at: usize| {
        raw.get(at..at + 4)
            .and_then(|b| <[u8; 4]>::try_from(b).ok())
            .map(i32::from_le_bytes)
    };
    let (left, top, right, bottom) = (word(12 + 8)?, word(16 + 8)?, word(20 + 8)?, word(24 + 8)?);
    if left < 0 || top < 0 || right <= left || bottom <= top {
        return None;
    }
    let (left, top, right, bottom) = (left as u32, top as u32, right as u32, bottom as u32);
    if right > width || bottom > height {
        return None;
    }
    if left == 0 && top == 0 && right == width && bottom == height {
        return None;
    }
    Some(Rect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

fn parse_graphic_buffer(raw: Vec<u8>) -> Slot {
    let u = |o: usize| {
        raw.get(8 + o..12 + o)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let nv = 0x28;
    let surface = nv + 0x34;
    // `NvSurface` in the marshalled buffer. `NvColorFormat` is a 64-bit enum,
    // which puts the layout word at 0x10 and the block height at 0x24.
    let plane = |index: usize| {
        let at = surface + index * NV_SURFACE_SIZE;
        PlaneGeometry {
            offset: u(at + 0x1c) as u64,
            pitch: u(at + 0x14),
            block_linear: u(at + 0x10) == 3,
            block_height_log2: u(at + 0x24).min(5),
        }
    };
    let planes = [plane(0), plane(1), plane(2)];
    Slot {
        nvmap: u(nv + 0x4),
        offset: planes[0].offset,
        width: u(surface),
        height: u(surface + 0x4),
        format: u(0x10),
        pitch: planes[0].pitch,
        block_linear: planes[0].block_linear,
        block_height_log2: planes[0].block_height_log2,
        chroma: [planes[1], planes[2]],
        flattened: raw,
        crop: None,
    }
}

/// Size of one `NvSurface` in the marshalled `NvGraphicBuffer`.
const NV_SURFACE_SIZE: usize = 0x58;

const TRANSFORM_FLIP_H: u32 = 1;
const TRANSFORM_FLIP_V: u32 = 2;
const TRANSFORM_ROT_90: u32 = 4;
const TRANSFORM_MASK: u32 = TRANSFORM_FLIP_H | TRANSFORM_FLIP_V | TRANSFORM_ROT_90;

/// `android::PixelFormat` values that VI layers can use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PixelFormat {
    Rgba8888,
    Rgbx8888,
    Rgb888,
    Rgb565,
    Bgra8888,
    /// 8-bit 4:2:0 YUV, laid out as described by [`YuvLayout`].
    Yuv(YuvLayout),
}

/// Plane layout of an 8-bit 4:2:0 layer buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum YuvLayout {
    /// Luma then interleaved `U V` (gralloc fourcc `NV12`).
    Nv12,
    /// Luma then interleaved `V U` (`PIXEL_FORMAT_YCrCb_420_SP`).
    Nv21,
    /// Planar `Y U V` (`PIXEL_FORMAT_YCbCr_420_P`).
    I420,
    /// Planar `Y V U` (`PIXEL_FORMAT_YV12`).
    Yv12,
}

/// Pixel-format codes that name a YUV layer even when the runtime cannot decode
/// that particular layout. Used to report an unconvertible layer by format
/// instead of presenting its bytes as RGBA.
fn is_yuv_format_code(format: u32) -> bool {
    matches!(
        format,
        // Android: YCrCb 4:2:0 SP, YCbCr 4:2:0 P, YCbCr 4:2:0 888.
        0x11 | 0x13 | 0x23
            // Android 10-bit semi-planar (P010).
            | 0x36 | 0x38
            // Gralloc fourccs: NV12 and YV12.
            | 0x3231_564E | 0x3231_5659
    )
}

impl PixelFormat {
    fn from_android(format: u32) -> Option<PixelFormat> {
        Some(match format {
            1 => PixelFormat::Rgba8888,
            2 => PixelFormat::Rgbx8888,
            3 => PixelFormat::Rgb888,
            4 => PixelFormat::Rgb565,
            5 => PixelFormat::Bgra8888,
            // AOSP `ui/PixelFormat.h`.
            0x11 => PixelFormat::Yuv(YuvLayout::Nv21),
            0x13 => PixelFormat::Yuv(YuvLayout::I420),
            // Fourcc codes, as `HAL_PIXEL_FORMAT_YV12` and the NV12 fourcc.
            0x3231_564E => PixelFormat::Yuv(YuvLayout::Nv12),
            0x3231_5659 => PixelFormat::Yuv(YuvLayout::Yv12),
            _ => return None,
        })
    }

    fn bytes_per_pixel(self) -> u64 {
        match self {
            PixelFormat::Rgba8888 | PixelFormat::Rgbx8888 | PixelFormat::Bgra8888 => 4,
            PixelFormat::Rgb888 => 3,
            PixelFormat::Rgb565 => 2,
            // The packed path only ever reads the luma plane of a YUV layer.
            PixelFormat::Yuv(_) => 1,
        }
    }

    /// Decodes one pixel into the `0x00RRGGBB` form used by `display::Frame`.
    fn decode(self, px: &[u8]) -> u32 {
        let b = |i: usize| px.get(i).copied().unwrap_or(0) as u32;
        match self {
            PixelFormat::Rgba8888 | PixelFormat::Rgbx8888 | PixelFormat::Rgb888 => {
                b(0) << 16 | b(1) << 8 | b(2)
            }
            PixelFormat::Bgra8888 => b(2) << 16 | b(1) << 8 | b(0),
            PixelFormat::Rgb565 => {
                let v = b(0) | b(1) << 8;
                let r = (v >> 11) & 0x1f;
                let g = (v >> 5) & 0x3f;
                let bl = v & 0x1f;
                (r << 3 | r >> 2) << 16 | (g << 2 | g >> 4) << 8 | (bl << 3 | bl >> 2)
            }
            // YUV layers are converted plane by plane; see `decode_layer`.
            PixelFormat::Yuv(_) => 0,
        }
    }
}

/// BT.601 limited-range YUV to `0x00RRGGBB`.
///
/// Layer buffers carry no colour-space tag, and the studio-swing matrix is what
/// the console's video playback path produces, so that is what we decode.
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> u32 {
    let (c, d, e) = (y as i32 - 16, u as i32 - 128, v as i32 - 128);
    let clamp = |v: i32| v.clamp(0, 255) as u32;
    let r = clamp((298 * c + 409 * e + 128) >> 8);
    let g = clamp((298 * c - 100 * d - 208 * e + 128) >> 8);
    let b = clamp((298 * c + 516 * d + 128) >> 8);
    r << 16 | g << 8 | b
}

/// Mean luma of a converted frame, for `KIRIKAE_TRACE_VIDEO` diagnostics.
fn mean_luma(pixels: &[u32]) -> u64 {
    if pixels.is_empty() {
        return 0;
    }
    let sum: u64 = pixels
        .iter()
        .map(|p| {
            let (r, g, b) = ((p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff);
            (r * 77 + g * 150 + b * 29) >> 8
        })
        .map(u64::from)
        .sum();
    sum / pixels.len() as u64
}

/// Byte geometry of one plane inside a layer buffer.
struct LayerPlane {
    /// Offset from the start of the buffer.
    start: u64,
    /// Row stride in bytes, or the GOB row width for block-linear surfaces.
    row_bytes: u64,
    height: u64,
    /// Bytes per sample.
    bpp: u64,
    block_linear: bool,
    block_height_log2: u32,
}

impl LayerPlane {
    fn new(
        geometry: PlaneGeometry,
        start: u64,
        width: u64,
        height: u64,
        bpp: u64,
        fallback_pitch: u32,
    ) -> LayerPlane {
        let pitch = if geometry.pitch != 0 {
            geometry.pitch
        } else {
            fallback_pitch
        };
        let row_bytes = if geometry.block_linear {
            (width * bpp).div_ceil(64) * 64
        } else {
            (pitch as u64).max(width * bpp)
        };
        LayerPlane {
            start,
            row_bytes,
            height,
            bpp,
            block_linear: geometry.block_linear,
            block_height_log2: geometry.block_height_log2,
        }
    }

    fn size(&self) -> u64 {
        let rows = if self.block_linear {
            let gob = 8u64 << self.block_height_log2;
            self.height.div_ceil(gob) * gob
        } else {
            self.height
        };
        self.row_bytes * rows
    }

    /// Byte offset of the sample at (`x`, `y`) inside the layer buffer.
    fn offset(&self, x: u64, y: u64) -> usize {
        let at = if self.block_linear {
            crate::gpu::block_linear_offset(x * self.bpp, y, self.row_bytes, self.block_height_log2)
        } else {
            y * self.row_bytes + x * self.bpp
        };
        (self.start + at) as usize
    }
}

impl Slot {
    /// Size of the part of the buffer the guest displays.
    fn visible(&self) -> (u32, u32) {
        self.crop
            .map_or((self.width, self.height), |crop| (crop.width, crop.height))
    }

    fn is_video_format(&self) -> bool {
        is_yuv_format_code(self.format)
            || matches!(
                PixelFormat::from_android(self.format),
                Some(PixelFormat::Yuv(_))
            )
    }

    /// Layout of plane `index`, with a packed buffer's planes inheriting the
    /// luma layout when the guest leaves the secondary `NvSurface` empty.
    fn plane(&self, index: usize) -> PlaneGeometry {
        if index == 0 {
            return PlaneGeometry {
                offset: self.offset,
                pitch: self.pitch,
                block_linear: self.block_linear,
                block_height_log2: self.block_height_log2,
            };
        }
        let plane = self.chroma[index - 1];
        if plane == PlaneGeometry::default() {
            PlaneGeometry {
                offset: 0,
                pitch: 0,
                block_linear: self.block_linear,
                block_height_log2: self.block_height_log2,
            }
        } else {
            plane
        }
    }

    /// Start of plane `index` measured from the first byte of the buffer.
    fn plane_start(&self, index: usize, fallback: u64) -> u64 {
        if index == 0 {
            return 0;
        }
        let offset = self.chroma[index - 1].offset;
        if offset == 0 {
            fallback
        } else {
            offset.saturating_sub(self.offset)
        }
    }

    /// Planes of this layer buffer in `format`.
    ///
    /// Chroma follows the luma plane unless the guest's `NvGraphicBuffer`
    /// describes the secondary planes itself, which is what planar 4:2:0
    /// allocations and buffers with a padded luma plane do.
    fn planes(&self, format: PixelFormat) -> Option<Vec<LayerPlane>> {
        let (w, h) = (self.width as u64, self.height as u64);
        if w == 0 || h == 0 {
            return None;
        }
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let luma = LayerPlane::new(self.plane(0), 0, w, h, format.bytes_per_pixel(), self.pitch);
        let after_luma = luma.size();
        Some(match format {
            PixelFormat::Yuv(YuvLayout::Nv12 | YuvLayout::Nv21) => {
                let start = self.plane_start(1, after_luma);
                vec![
                    luma,
                    LayerPlane::new(self.plane(1), start, cw * 2, ch, 1, self.pitch),
                ]
            }
            PixelFormat::Yuv(YuvLayout::I420 | YuvLayout::Yv12) => {
                let start = self.plane_start(1, after_luma);
                let u = LayerPlane::new(self.plane(1), start, cw, ch, 1, self.pitch);
                let after_u = u.start + u.size();
                let start = self.plane_start(2, after_u);
                let v = LayerPlane::new(self.plane(2), start, cw, ch, 1, self.pitch);
                vec![luma, u, v]
            }
            _ => vec![luma],
        })
    }
}

impl YuvLayout {
    /// The U and V samples of the chroma block that covers pixel (`x`, `y`).
    fn chroma(self, bytes: &[u8], planes: &[LayerPlane], x: u64, y: u64) -> (u8, u8) {
        let (cx, cy) = (x / 2, y / 2);
        let at = |plane: &LayerPlane, sample: u64| {
            bytes.get(plane.offset(sample, cy)).copied().unwrap_or(0)
        };
        // Interleaved planes hold one U and one V byte per chroma sample, so the
        // byte index is twice the chroma coordinate.
        match self {
            YuvLayout::Nv12 => (at(&planes[1], 2 * cx), at(&planes[1], 2 * cx + 1)),
            YuvLayout::Nv21 => (at(&planes[1], 2 * cx + 1), at(&planes[1], 2 * cx)),
            YuvLayout::I420 => (at(&planes[1], cx), at(&planes[2], cx)),
            YuvLayout::Yv12 => (at(&planes[2], cx), at(&planes[1], cx)),
        }
    }
}

/// Bytes needed to read the whole layer buffer.
fn buffer_size(planes: &[LayerPlane]) -> u64 {
    planes.iter().map(|p| p.start + p.size()).max().unwrap_or(0)
}

/// Decodes the visible part of a layer buffer into `0x00RRGGBB` pixels.
///
/// The guest's crop rectangle selects the region inside the buffer; without one
/// the whole buffer is decoded. Plane geometry still describes the full
/// allocation, so block-linear offsets are computed at absolute coordinates.
fn decode_layer(bytes: &[u8], slot: &Slot, planes: &[LayerPlane], format: PixelFormat) -> Vec<u32> {
    let (w, h) = slot.visible();
    let (ox, oy) = slot.crop.map_or((0, 0), |crop| (crop.x, crop.y));
    let mut pixels = vec![0u32; (w * h) as usize];
    let luma = &planes[0];
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (ox as u64 + x as u64, oy as u64 + y as u64);
            let at = luma.offset(sx, sy);
            let pixel = match format {
                PixelFormat::Yuv(layout) => {
                    let sample = bytes.get(at).copied().unwrap_or(0);
                    let (u, v) = layout.chroma(bytes, planes, sx, sy);
                    yuv_to_rgb(sample, u, v)
                }
                _ => {
                    let bpp = format.bytes_per_pixel() as usize;
                    match bytes.get(at..at + bpp) {
                        Some(px) => format.decode(px),
                        None => 0,
                    }
                }
            };
            pixels[(y * w + x) as usize] = pixel;
        }
    }
    pixels
}

/// Decodes a whole layer buffer into the frame the presenter uploads.
///
/// Returns `None` instead of a black frame when the buffer is shorter than the
/// planes the guest described, so a truncated or unmapped layer is reported and
/// skipped rather than shown as black video.
fn decode_frame(bytes: &[u8], slot: &Slot, format: PixelFormat, transform: u32) -> Option<Frame> {
    let planes = slot.planes(format)?;
    if buffer_size(&planes) > bytes.len() as u64 {
        return None;
    }
    let pixels = decode_layer(bytes, slot, &planes, format);
    let (width, height) = slot.visible();
    Some(transformed(
        Frame {
            width,
            height,
            pixels,
        },
        transform,
    ))
}

/// Applies the buffer's transform (flips and quarter-turn rotation) while
/// copying it. The rotation is clockwise, matching the Android layer
/// transform; a 90/270 degree rotation swaps the frame dimensions.
fn transformed(frame: Frame, transform: u32) -> Frame {
    let transform = transform & TRANSFORM_MASK;
    if transform == 0 {
        return frame;
    }
    let (w, h) = (frame.width as usize, frame.height as usize);
    let rotate = transform & TRANSFORM_ROT_90 != 0;
    let (ow, oh) = if rotate { (h, w) } else { (w, h) };
    let flip_h = transform & TRANSFORM_FLIP_H != 0;
    let flip_v = transform & TRANSFORM_FLIP_V != 0;
    let mut pixels = vec![0u32; w * h];
    for dy in 0..oh {
        let my = if flip_v { oh - 1 - dy } else { dy };
        for dx in 0..ow {
            let mx = if flip_h { ow - 1 - dx } else { dx };
            let (sx, sy) = if rotate { (my, h - 1 - mx) } else { (mx, my) };
            pixels[dy * ow + dx] = frame.pixels[sy * w + sx];
        }
    }
    Frame {
        width: ow as u32,
        height: oh as u32,
        pixels,
    }
}

fn read_frame(p: &Process, s: &Slot, transform: u32) -> Option<Frame> {
    let m = p.gpu.nvmap(s.nvmap)?;
    let mut pixels = p.gpu.read_pixels(m.addr + s.offset, s.width, s.height)?;
    let (width, height) = s.visible();
    if let Some(crop) = s.crop {
        let mut cropped = Vec::with_capacity((crop.width * crop.height) as usize);
        for y in 0..crop.height {
            let start = ((crop.y + y) * s.width + crop.x) as usize;
            cropped.extend_from_slice(&pixels[start..start + crop.width as usize]);
        }
        pixels = cropped;
    }
    Some(transformed(
        Frame {
            width,
            height,
            pixels,
        },
        transform,
    ))
}

/// Reads and decodes the layer buffer from guest memory. This is the path used
/// for formats the renderer does not understand as RGBA images, for YUV layers
/// and for rotations the swapchain present cannot express.
fn read_frame_from_memory(
    p: &Process,
    cpu: u64,
    s: &Slot,
    format: PixelFormat,
    transform: u32,
) -> Option<Frame> {
    let planes = s.planes(format)?;
    let bytes = p.memory.read(cpu, buffer_size(&planes))?;
    decode_frame(&bytes, s, format, transform)
}

/// Whether a layer carries video.
///
/// YUV layer formats are video by definition. VIC can also compose a decoded
/// YUV frame into an RGBA surface; those surfaces are tagged by the clock for a
/// short while, so the layer is still gated. Layers the GPU renders (game and UI
/// frames) are never tagged and keep the immediate present path.
fn is_video_layer(p: &Process, s: &Slot) -> bool {
    s.is_video_format()
        || p.gpu
            .nvmap(s.nvmap)
            .is_some_and(|m| clock::is_video_surface(m.addr + s.offset))
}

/// Presents one queued layer buffer. Returns whether a frame reached the
/// display path.
fn present(p: &Process, s: &Slot, transform: u32) -> bool {
    if transform & !TRANSFORM_MASK != 0 {
        p.warn_once(format!(
            "vi: buffer transform {transform:#x} is not supported"
        ));
    }
    let format = match PixelFormat::from_android(s.format) {
        Some(format) => format,
        None if is_yuv_format_code(s.format) => {
            // A YUV layer we cannot convert is a gap in the pipeline, not an
            // RGBA layer: show the last frame and name the format.
            p.warn_once(format!(
                "vi: YUV layer format {} cannot be converted; frame not presented",
                s.format
            ));
            return false;
        }
        None => {
            p.warn_once(format!(
                "vi: pixel format {} is shown as RGBA8888",
                s.format
            ));
            PixelFormat::Rgba8888
        }
    };
    let Some(m) = p.gpu.nvmap(s.nvmap) else {
        p.warn_once(format!("vi: buffer uses unknown nvmap handle {}", s.nvmap));
        return false;
    };
    if let Some(dir) = std::env::var_os("KIRIKAE_TARGET_DUMP") {
        let n = display::FRAMES.load(std::sync::atomic::Ordering::Relaxed) + 1;
        let every = std::env::var("KIRIKAE_TARGET_DUMP_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60u64)
            .max(1);
        if n % every == 1 % every {
            p.gpu.dump_images(std::path::Path::new(&dir), n);
        }
    }
    let cpu = m.addr + s.offset;
    if crate::host1x::trace() {
        eprintln!(
            "vi: present {}x{} format {} at {cpu:#x} (nvmap {}+{:#x}) video {} rendered {}",
            s.width,
            s.height,
            s.format,
            s.nvmap,
            s.offset,
            is_video_layer(p, s),
            p.gpu.rendered(cpu, s.width, s.height)
        );
    }
    // A surface VIC composed from a video source is written by DMA, so a GPU
    // image registered for the same memory is stale or never filled: read it
    // back from guest memory instead of presenting that image.
    if !matches!(format, PixelFormat::Yuv(_)) && is_video_layer(p, s) {
        let Some(frame) = read_frame_from_memory(p, cpu, s, format, transform) else {
            p.warn_once(format!(
                "vi: video surface format {} at {cpu:#x} could not be read; frame not presented",
                s.format
            ));
            return false;
        };
        if crate::host1x::trace() {
            eprintln!(
                "vi: composed video {}x{} format {} at {cpu:#x} -> mean luma {}",
                frame.width,
                frame.height,
                s.format,
                mean_luma(&frame.pixels)
            );
        }
        present_frame(p, frame);
        return true;
    }
    // YUV layers never reach the swapchain as an image: decode the planes and
    // upload the BT.601-converted pixels through the RGB present path, applying
    // the layer transform while copying.
    if matches!(format, PixelFormat::Yuv(_)) {
        let Some(frame) = read_frame_from_memory(p, cpu, s, format, transform) else {
            p.warn_once(format!(
                "vi: YUV layer format {} at {cpu:#x} could not be read; frame not presented",
                s.format
            ));
            return false;
        };
        clock::converted();
        if crate::host1x::trace() {
            eprintln!(
                "vi: YUV layer {}x{} format {} at {cpu:#x} -> mean luma {}",
                frame.width,
                frame.height,
                s.format,
                mean_luma(&frame.pixels)
            );
        }
        present_frame(p, frame);
        return true;
    }
    let (flip_h, flip_v) = (
        transform & TRANSFORM_FLIP_H != 0,
        transform & TRANSFORM_FLIP_V != 0,
    );
    let rotation = transform & TRANSFORM_ROT_90 != 0;
    // The swapchain present path only handles RGBA buffers, cannot rotate and
    // cannot crop, so a transformed or cropped layer is decoded on the CPU.
    let readable = format == PixelFormat::Rgba8888;
    let direct = readable && !rotation && s.crop.is_none();
    let dump = display::dump_due() || crate::usage::frame_wanted();
    if !display::headless()
        && direct
        && p.gpu.present(cpu, s.width, s.height, flip_h, flip_v)
            == crate::gpu::vulkan::Presented::Shown
    {
        let frame = if dump {
            read_frame(p, s, transform)
        } else {
            None
        };
        if crate::host1x::trace()
            && let Some(frame) = &frame
        {
            eprintln!(
                "vi: presented {}x{} mean luma {} (format {})",
                frame.width,
                frame.height,
                mean_luma(&frame.pixels),
                s.format
            );
        }
        display::presented(p, frame.as_ref());
        return true;
    }
    if direct && display::headless() && !dump && p.gpu.rendered(cpu, s.width, s.height) {
        display::presented(p, None);
        return true;
    }
    if readable && let Some(frame) = read_frame(p, s, transform) {
        present_frame(p, frame);
        return true;
    }
    let Some(frame) = read_frame_from_memory(p, cpu, s, format, transform) else {
        p.warn_once(format!("vi: buffer at {cpu:#x} is not readable"));
        return false;
    };
    if crate::host1x::trace() {
        eprintln!(
            "vi: layer {}x{} format {} at {cpu:#x} -> mean luma {}",
            frame.width,
            frame.height,
            s.format,
            mean_luma(&frame.pixels)
        );
    }
    present_frame(p, frame);
    true
}

/// Presents a decoded frame: through the swapchain when one is available,
/// otherwise through the CPU window path.
fn present_frame(p: &Process, frame: Frame) {
    if !display::headless()
        && p.gpu
            .present_pixels(frame.width, frame.height, &frame.pixels)
            == crate::gpu::vulkan::Presented::Shown
    {
        display::presented(p, Some(&frame));
        return;
    }
    display::present(p, frame);
}

static VSYNC_EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn pace(interval: u32) {
    static LAST: Mutex<Option<u64>> = Mutex::new(None);
    if std::env::var_os("KIRIKAE_NO_FRAME_LIMIT").is_some() {
        return;
    }
    let epoch = *VSYNC_EPOCH.get_or_init(std::time::Instant::now);
    let blank = |n: u64| epoch + std::time::Duration::from_nanos(n * 1_000_000_000 / 60);
    let mut last = LAST.lock().unwrap();
    let now = std::time::Instant::now();
    let current = (now.duration_since(epoch).as_nanos() * 60 / 1_000_000_000) as u64;
    match last.map(|l| l + interval as u64) {
        Some(target) if blank(target) > now => {
            let wait = blank(target) - now;
            crate::display::PACED_NS
                .fetch_add(wait.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            std::thread::sleep(wait);
            *last = Some(target);
        }
        _ => *last = Some(current),
    }
}

struct ParcelReader<'a> {
    payload: &'a [u8],
    pos: usize,
}

impl<'a> ParcelReader<'a> {
    fn new(parcel: &'a [u8]) -> Self {
        let u = |o: usize| {
            parcel
                .get(o..o + 4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap())) as usize
        };
        let (size, offset) = (u(0), u(4));
        let payload = parcel
            .get(offset..offset.saturating_add(size))
            .unwrap_or(&[]);
        ParcelReader { payload, pos: 0 }
    }

    fn i32(&mut self) -> i32 {
        let v = self
            .payload
            .get(self.pos..self.pos + 4)
            .map_or(0, |b| i32::from_le_bytes(b.try_into().unwrap()));
        self.pos += 4;
        v
    }

    fn skip_interface_token(&mut self) {
        self.i32();
        let len = self.i32();
        if len > 0 {
            self.pos += ((len as usize + 1) * 2).next_multiple_of(4);
        }
    }

    fn flattenable(&mut self) -> Option<Vec<u8>> {
        let start = self.pos;
        let len = self.i32() as usize;
        let fds = self.i32() as usize;
        let end = self.pos + (len + fds * 4).next_multiple_of(4);
        let raw = self.payload.get(start..end)?.to_vec();
        self.pos = end;
        Some(raw)
    }
}

#[derive(Default)]
struct ParcelWriter {
    payload: Vec<u8>,
}

impl ParcelWriter {
    fn i32(&mut self, v: i32) {
        self.payload.extend_from_slice(&v.to_le_bytes());
    }

    fn i64(&mut self, v: i64) {
        self.payload.extend_from_slice(&v.to_le_bytes());
    }

    fn bytes(&mut self, b: &[u8]) {
        self.payload.extend_from_slice(b);
        while !self.payload.len().is_multiple_of(4) {
            self.payload.push(0);
        }
    }

    fn finish(self) -> Vec<u8> {
        self.finish_with_objects(&[])
    }

    fn finish_with_objects(self, objects: &[u32]) -> Vec<u8> {
        let size = self.payload.len() as u32;
        let objects_size = objects.len() as u32 * 4;
        let mut out = Vec::new();
        for v in [size, 0x10, objects_size, 0x10 + size] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.payload);
        for o in objects {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presentation_flips() {
        let frame = || Frame {
            width: 3,
            height: 2,
            pixels: vec![1, 2, 3, 4, 5, 6],
        };
        assert_eq!(transformed(frame(), 0).pixels, [1, 2, 3, 4, 5, 6]);
        assert_eq!(
            transformed(frame(), TRANSFORM_FLIP_V).pixels,
            [4, 5, 6, 1, 2, 3]
        );
        assert_eq!(
            transformed(frame(), TRANSFORM_FLIP_H).pixels,
            [3, 2, 1, 6, 5, 4]
        );
        assert_eq!(
            transformed(frame(), TRANSFORM_FLIP_H | TRANSFORM_FLIP_V).pixels,
            [6, 5, 4, 3, 2, 1]
        );
    }

    #[test]
    fn presentation_rotations() {
        let frame = || Frame {
            width: 3,
            height: 2,
            pixels: vec![1, 2, 3, 4, 5, 6],
        };
        for (transform, pixels) in [
            (TRANSFORM_ROT_90, [4, 1, 5, 2, 6, 3]),
            (TRANSFORM_ROT_90 | TRANSFORM_FLIP_H, [1, 4, 2, 5, 3, 6]),
            (TRANSFORM_ROT_90 | TRANSFORM_FLIP_V, [6, 3, 5, 2, 4, 1]),
            (TRANSFORM_MASK, [3, 6, 2, 5, 1, 4]),
        ] {
            let turned = transformed(frame(), transform);
            assert_eq!((turned.width, turned.height), (2, 3), "{transform}");
            assert_eq!(turned.pixels, pixels, "{transform}");
        }
    }

    #[test]
    fn pixel_formats_decode() {
        assert_eq!(PixelFormat::Rgba8888.decode(&[1, 2, 3, 4]), 0x01_02_03);
        assert_eq!(PixelFormat::Rgbx8888.decode(&[1, 2, 3, 0]), 0x01_02_03);
        assert_eq!(PixelFormat::Rgb888.decode(&[1, 2, 3]), 0x01_02_03);
        assert_eq!(PixelFormat::Bgra8888.decode(&[3, 2, 1, 4]), 0x01_02_03);
        // RGB565: 0xf800 is full red, 0x07e0 full green and 0x001f full blue.
        assert_eq!(PixelFormat::Rgb565.decode(&[0x00, 0xf8]), 0x00_ff_00_00);
        assert_eq!(PixelFormat::Rgb565.decode(&[0xe0, 0x07]), 0x00_00_ff_00);
        assert_eq!(PixelFormat::Rgb565.decode(&[0x1f, 0x00]), 0x00_00_00_ff);
        assert_eq!(PixelFormat::Rgba8888.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Rgb888.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Rgb565.bytes_per_pixel(), 2);
        for (android, want) in [
            (1, PixelFormat::Rgba8888),
            (2, PixelFormat::Rgbx8888),
            (3, PixelFormat::Rgb888),
            (4, PixelFormat::Rgb565),
            (5, PixelFormat::Bgra8888),
            (0x11, PixelFormat::Yuv(YuvLayout::Nv21)),
            (0x13, PixelFormat::Yuv(YuvLayout::I420)),
            (0x3231_564E, PixelFormat::Yuv(YuvLayout::Nv12)),
            (0x3231_5659, PixelFormat::Yuv(YuvLayout::Yv12)),
        ] {
            assert_eq!(PixelFormat::from_android(android), Some(want));
        }
        assert_eq!(PixelFormat::from_android(0x16), None);
    }

    fn yuv_slot(width: u32, height: u32, pitch: u32) -> Slot {
        Slot {
            width,
            height,
            pitch,
            ..Slot::default()
        }
    }

    fn decode(slot: &Slot, format: PixelFormat, bytes: &[u8]) -> Vec<u32> {
        let planes = slot.planes(format).unwrap();
        assert!(buffer_size(&planes) as usize <= bytes.len());
        decode_layer(bytes, slot, &planes, format)
    }

    /// Black, white, and a chroma pair that lands on pure red.
    const LUMA: [u8; 4] = [16, 235, 81, 81];
    const CHROMA: [(u8, u8); 2] = [(128, 128), (90, 240)];

    fn expected() -> [u32; 4] {
        [0x000000, 0x00ff_ffff, 0x00ff_0000, 0x00ff_0000]
    }

    #[test]
    fn nv12_layers_decode_to_rgb() {
        let slot = yuv_slot(4, 1, 4);
        let mut bytes = LUMA.to_vec();
        for (u, v) in CHROMA {
            bytes.extend_from_slice(&[u, v]);
        }
        assert_eq!(
            decode(&slot, PixelFormat::Yuv(YuvLayout::Nv12), &bytes),
            expected()
        );
        // NV21 keeps the same samples in V, U order.
        let mut nv21 = LUMA.to_vec();
        for (u, v) in CHROMA {
            nv21.extend_from_slice(&[v, u]);
        }
        assert_eq!(
            decode(&slot, PixelFormat::Yuv(YuvLayout::Nv21), &nv21),
            expected()
        );
    }

    #[test]
    fn planar_layers_decode_to_rgb() {
        let slot = yuv_slot(4, 1, 4);
        // Planar chroma rows are padded to the layer pitch like the luma ones.
        let mut u: Vec<u8> = CHROMA.iter().map(|c| c.0).collect();
        u.resize(4, 0);
        let mut v: Vec<u8> = CHROMA.iter().map(|c| c.1).collect();
        v.resize(4, 0);
        let mut i420 = LUMA.to_vec();
        i420.extend_from_slice(&u);
        i420.extend_from_slice(&v);
        assert_eq!(
            decode(&slot, PixelFormat::Yuv(YuvLayout::I420), &i420),
            expected()
        );
        // YV12 stores V before U.
        let mut yv12 = LUMA.to_vec();
        yv12.extend_from_slice(&v);
        yv12.extend_from_slice(&u);
        assert_eq!(
            decode(&slot, PixelFormat::Yuv(YuvLayout::Yv12), &yv12),
            expected()
        );
    }

    #[test]
    fn yuv_layers_honour_the_row_pitch() {
        // Two rows with 8 bytes of padding between them.
        let slot = yuv_slot(4, 2, 12);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&LUMA[..4]);
        bytes.extend_from_slice(&[0x55; 8]);
        bytes.extend_from_slice(&[16, 16, 16, 16]);
        bytes.extend_from_slice(&[0x55; 8]);
        // One chroma row covers both luma rows.
        bytes.extend_from_slice(&[128, 128, 90, 240]);
        bytes.extend_from_slice(&[0x55; 8]);
        let out = decode(&slot, PixelFormat::Yuv(YuvLayout::Nv12), &bytes);
        assert_eq!(out[..4], [0x000000, 0x00ff_ffff, 0x00ff_0000, 0x00ff_0000]);
        // The second row shares the chroma row: black luma over the red block
        // still picks up that block's chroma.
        let red = yuv_to_rgb(16, 90, 240);
        assert_eq!(out[4..], [0, 0, red, red]);
    }

    #[test]
    fn yuv_buffer_size_covers_the_chroma_plane() {
        let slot = yuv_slot(4, 2, 8);
        let size = |format| buffer_size(&slot.planes(format).unwrap());
        // Both luma and chroma rows use the layer pitch.
        assert_eq!(size(PixelFormat::Yuv(YuvLayout::Nv12)), 16 + 8);
        assert_eq!(size(PixelFormat::Yuv(YuvLayout::I420)), 16 + 8 + 8);
        assert_eq!(size(PixelFormat::Rgba8888), 16 * 2);
        // Block-linear layers round the plane up to whole GOB rows.
        let block = Slot {
            block_linear: true,
            block_height_log2: 1,
            ..yuv_slot(4, 2, 8)
        };
        assert_eq!(
            buffer_size(&block.planes(PixelFormat::Yuv(YuvLayout::Nv12)).unwrap()),
            64 * 16 + 64 * 16
        );
    }

    /// The bytes `Gpu::present_pixels` uploads: each `0x00RRGGBB` word as a
    /// little-endian BGRA texel.
    fn swapchain_bytes(pixels: &[u32]) -> Vec<u8> {
        pixels.iter().flat_map(|p| p.to_le_bytes()).collect()
    }

    #[test]
    fn a_yuv_frame_reaches_the_swapchain_as_bt601_rgb() {
        let slot = yuv_slot(4, 1, 4);
        let mut bytes = LUMA.to_vec();
        for (u, v) in CHROMA {
            bytes.extend_from_slice(&[u, v]);
        }
        let frame = decode_frame(&bytes, &slot, PixelFormat::Yuv(YuvLayout::Nv12), 0).unwrap();
        assert_eq!(frame.pixels, expected());
        // B8G8R8A8: black keeps alpha at zero, white is all-ones, and the chroma
        // pair lands on pure red.
        assert_eq!(
            swapchain_bytes(&frame.pixels),
            [
                0, 0, 0, 0, // black
                255, 255, 255, 0, // white
                0, 0, 255, 0, // red
                0, 0, 255, 0, // red
            ]
        );
    }

    #[test]
    fn every_yuv_layout_converts_to_the_same_swapchain_bytes() {
        let slot = yuv_slot(4, 1, 4);
        let mut u: Vec<u8> = CHROMA.iter().map(|c| c.0).collect();
        u.resize(4, 0);
        let mut v: Vec<u8> = CHROMA.iter().map(|c| c.1).collect();
        v.resize(4, 0);

        let mut nv12 = LUMA.to_vec();
        for (u, v) in CHROMA {
            nv12.extend_from_slice(&[u, v]);
        }
        let mut nv21 = LUMA.to_vec();
        for (u, v) in CHROMA {
            nv21.extend_from_slice(&[v, u]);
        }
        let mut i420 = LUMA.to_vec();
        i420.extend_from_slice(&u);
        i420.extend_from_slice(&v);
        let mut yv12 = LUMA.to_vec();
        yv12.extend_from_slice(&v);
        yv12.extend_from_slice(&u);

        for (layout, bytes) in [
            (YuvLayout::Nv12, nv12),
            (YuvLayout::Nv21, nv21),
            (YuvLayout::I420, i420),
            (YuvLayout::Yv12, yv12),
        ] {
            let frame = decode_frame(&bytes, &slot, PixelFormat::Yuv(layout), 0).unwrap();
            assert_eq!(frame.pixels, expected(), "{layout:?}");
            assert_eq!(
                swapchain_bytes(&frame.pixels),
                swapchain_bytes(&expected()),
                "{layout:?}"
            );
        }
    }

    #[test]
    fn block_linear_yuv_planes_decode_in_gob_order() {
        let slot = Slot {
            width: 4,
            height: 2,
            block_linear: true,
            block_height_log2: 1,
            ..Slot::default()
        };
        let planes = slot.planes(PixelFormat::Yuv(YuvLayout::Nv12)).unwrap();
        let luma = crate::host1x::to_block_linear(&[16, 235, 81, 81, 16, 16, 16, 16], 4, 4, 2, 1);
        let chroma = crate::host1x::to_block_linear(&[128, 128, 90, 240], 4, 4, 1, 1);
        let mut bytes = vec![0u8; buffer_size(&planes) as usize];
        bytes[..luma.len()].copy_from_slice(&luma);
        let after_luma = planes[0].size() as usize;
        bytes[after_luma..after_luma + chroma.len()].copy_from_slice(&chroma);

        let frame = decode_frame(&bytes, &slot, PixelFormat::Yuv(YuvLayout::Nv12), 0).unwrap();
        let red = yuv_to_rgb(16, 90, 240);
        assert_eq!(
            frame.pixels,
            [
                0x000000,
                0x00ff_ffff,
                0x00ff_0000,
                0x00ff_0000,
                0,
                0,
                red,
                red,
            ]
        );
    }

    #[test]
    fn secondary_plane_geometry_from_the_graphic_buffer_is_honoured() {
        let mut raw = vec![0u8; 8 + 0x140];
        {
            let mut put = |at: usize, value: u32| {
                raw[8 + at..8 + at + 4].copy_from_slice(&value.to_le_bytes());
            };
            put(0x10, 0x11); // android format: YCrCb 420 SP
            put(0x2c, 7); // nvmap id
            put(0x5c, 4); // width
            put(0x60, 2); // height
            put(0x6c, 1); // plane 0 layout: pitch
            put(0x70, 8); // plane 0 pitch
            put(0x78, 0x100); // plane 0 offset
            put(0xb4 + 0x10, 1);
            put(0xb4 + 0x14, 4); // chroma pitch
            put(0xb4 + 0x1c, 0x120);
            put(0x10c + 0x10, 1);
            put(0x10c + 0x14, 4);
            put(0x10c + 0x1c, 0x140);
        }
        let slot = parse_graphic_buffer(raw);
        assert_eq!(slot.nvmap, 7);
        assert_eq!((slot.width, slot.height), (4, 2));
        assert_eq!(slot.format, 0x11);
        assert_eq!((slot.offset, slot.pitch), (0x100, 8));
        assert_eq!((slot.chroma[0].offset, slot.chroma[0].pitch), (0x120, 4));
        assert_eq!((slot.chroma[1].offset, slot.chroma[1].pitch), (0x140, 4));

        // Chroma starts where the guest says, not immediately after luma.
        let nv21 = slot.planes(PixelFormat::Yuv(YuvLayout::Nv21)).unwrap();
        assert_eq!(nv21[1].start, 0x120 - 0x100);
        assert_eq!(nv21[1].row_bytes, 4);
        let i420 = slot.planes(PixelFormat::Yuv(YuvLayout::I420)).unwrap();
        assert_eq!(i420[1].start, 0x20);
        assert_eq!(i420[2].start, 0x40);
    }

    #[test]
    fn rgba_layers_do_not_go_through_the_yuv_conversion() {
        let slot = Slot {
            width: 1,
            height: 1,
            pitch: 4,
            format: 1, // PIXEL_FORMAT_RGBA_8888
            ..Slot::default()
        };
        assert!(!slot.is_video_format());
        assert!(!is_yuv_format_code(slot.format));
        // Luma 16 would be video black; RGBA 16 is a dark grey.
        let frame = decode_frame(
            &[16, 16, 16, 0xff],
            &slot,
            PixelFormat::from_android(slot.format).unwrap(),
            0,
        )
        .unwrap();
        assert_eq!(frame.pixels, [0x0010_1010]);
        // Formats the runtime knows are YUV but cannot decode are still video.
        for format in [0x11, 0x13, 0x23, 0x36, 0x38, 0x3231_564E, 0x3231_5659] {
            assert!(is_yuv_format_code(format), "{format:#x}");
        }
        assert!(!is_yuv_format_code(1));
        assert!(!is_yuv_format_code(0x16));
    }

    #[test]
    fn yuv_frames_apply_the_layer_transform() {
        let slot = yuv_slot(3, 2, 4);
        let mut bytes = vec![10, 20, 30, 0, 40, 50, 60, 0];
        bytes.extend_from_slice(&[128, 128, 128, 128]);

        let plain = decode_frame(&bytes, &slot, PixelFormat::Yuv(YuvLayout::Nv12), 0).unwrap();
        let turned = decode_frame(
            &bytes,
            &slot,
            PixelFormat::Yuv(YuvLayout::Nv12),
            TRANSFORM_ROT_90,
        )
        .unwrap();
        assert_eq!((turned.width, turned.height), (2, 3));
        // Clockwise: bottom-left, top-left, bottom-middle, ...
        assert_eq!(
            turned.pixels,
            [
                plain.pixels[3],
                plain.pixels[0],
                plain.pixels[4],
                plain.pixels[1],
                plain.pixels[5],
                plain.pixels[2],
            ]
        );
    }

    #[test]
    fn a_truncated_yuv_buffer_is_not_presented_as_black() {
        let slot = yuv_slot(4, 2, 8);
        // The luma plane alone is not enough for the NV12 buffer.
        assert!(decode_frame(&[16; 16], &slot, PixelFormat::Yuv(YuvLayout::Nv12), 0).is_none());
    }

    #[test]
    fn queue_crop_reads_the_guest_rectangle() {
        let mut raw = vec![0u8; 0x60];
        let put = |raw: &mut Vec<u8>, at: usize, v: i32| {
            raw[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        // timestamp and isAutoTimestamp are not a crop.
        put(&mut raw, 8, 1234);
        put(&mut raw, 20, 4); // left
        put(&mut raw, 24, 6); // top
        put(&mut raw, 28, 900); // right
        put(&mut raw, 32, 506); // bottom
        assert_eq!(
            queue_crop(Some(&raw), 912, 512),
            Some(Rect {
                x: 4,
                y: 6,
                width: 896,
                height: 500
            })
        );
        // The whole buffer, an empty rectangle, an inverted one, one outside
        // the buffer and a missing queue input all mean "no crop".
        put(&mut raw, 20, 0);
        put(&mut raw, 24, 0);
        put(&mut raw, 28, 912);
        put(&mut raw, 32, 512);
        assert_eq!(queue_crop(Some(&raw), 912, 512), None);
        put(&mut raw, 28, 0);
        assert_eq!(queue_crop(Some(&raw), 912, 512), None);
        put(&mut raw, 20, 10);
        put(&mut raw, 28, 5);
        assert_eq!(queue_crop(Some(&raw), 912, 512), None);
        put(&mut raw, 20, 0);
        put(&mut raw, 28, 913);
        assert_eq!(queue_crop(Some(&raw), 912, 512), None);
        assert_eq!(queue_crop(None, 912, 512), None);
    }

    /// A decoded 1080p picture is coded as 1920x1088 (and 900x506 as 912x512).
    /// The queue input's crop rectangle is what leaves the coded padding out.
    #[test]
    fn a_yuv_layer_crop_leaves_out_the_coded_padding() {
        let slot = Slot {
            width: 4,
            height: 2,
            pitch: 4,
            crop: Some(Rect {
                x: 1,
                y: 0,
                width: 2,
                height: 1,
            }),
            ..Slot::default()
        };
        let mut bytes = vec![16, 235, 81, 81];
        bytes.extend_from_slice(&[16, 16, 16, 16]);
        bytes.extend_from_slice(&[128, 128, 90, 240]);
        let frame = decode_frame(&bytes, &slot, PixelFormat::Yuv(YuvLayout::Nv12), 0).unwrap();
        assert_eq!((frame.width, frame.height), (2, 1));
        // Pixel 1 is luma 235 with neutral chroma; pixel 2 is the red pair.
        assert_eq!(frame.pixels, [0x00ff_ffff, 0x00ff_0000]);
    }

    #[test]
    fn a_rotated_yuv_layer_keeps_its_colours() {
        // Left half red (Y 81, U 90, V 240), right half green (145, 54, 34).
        let slot = yuv_slot(4, 2, 4);
        let mut bytes = vec![81, 81, 145, 145, 81, 81, 145, 145];
        bytes.extend_from_slice(&[90, 240, 54, 34]);
        let frame = decode_frame(
            &bytes,
            &slot,
            PixelFormat::Yuv(YuvLayout::Nv12),
            TRANSFORM_ROT_90,
        )
        .unwrap();
        assert_eq!((frame.width, frame.height), (2, 4));
        let red = yuv_to_rgb(81, 90, 240);
        let green = yuv_to_rgb(145, 54, 34);
        assert_eq!(
            frame.pixels,
            [red, red, red, red, green, green, green, green]
        );
    }

    #[test]
    fn mean_luma_reports_black_and_white() {
        assert_eq!(mean_luma(&[]), 0);
        assert_eq!(mean_luma(&[0, 0]), 0);
        assert_eq!(mean_luma(&[0x00ff_ffff, 0x00ff_ffff]), 255);
        assert_eq!(mean_luma(&[0, 0x00ff_ffff]), 127);
    }

    #[test]
    fn layer_creation_starts_after_the_presented_layer() {
        let mut managers = Managers::new();
        assert_eq!(managers.layers.len(), 1);
        let presented = managers.layer(PRESENTED_LAYER).unwrap();
        assert_eq!(presented.display, DEFAULT_DISPLAY);
        assert!(presented.visible);
        assert_eq!(
            (presented.width, presented.height),
            (FALLBACK_WIDTH as i64, FALLBACK_HEIGHT as i64)
        );

        let id = managers.create_layer(DEFAULT_DISPLAY);
        assert_eq!(id, PRESENTED_LAYER + 1);
        assert_ne!(id, PRESENTED_LAYER);
        // An unknown display id keeps the layer usable on the default display.
        let fallback = managers.create_layer(999);
        assert_eq!(fallback, id + 1);
        assert_eq!(
            managers.layer(fallback).unwrap().display,
            managers.default_display
        );
    }

    #[test]
    fn layer_state_transitions_are_visible_to_the_manager() {
        let mut managers = Managers::new();
        let id = managers.create_layer(DEFAULT_DISPLAY);

        {
            let layer = managers.layer_mut(id).unwrap();
            layer.visible = false;
            layer.z = -3;
            layer.x = 12.5;
            layer.y = -4.0;
            layer.width = 1600;
            layer.height = 900;
            layer.alpha = 0.25;
            layer.stack = 2;
        }
        let layer = managers.layer(id).unwrap();
        assert!(!layer.visible);
        assert_eq!(layer.z, -3);
        assert_eq!((layer.x, layer.y), (12.5, -4.0));
        assert_eq!((layer.width, layer.height), (1600, 900));
        assert_eq!(layer.alpha, 0.25);
        assert_eq!(layer.stack, 2);

        assert!(managers.destroy_layer(id));
        assert!(managers.layer(id).is_none());
        assert!(!managers.destroy_layer(id));
        // The presented layer is not destroyed by a manager call for another id.
        assert!(managers.layer(PRESENTED_LAYER).is_some());
    }

    #[test]
    fn display_enumeration_reports_the_default_panel() {
        let mut managers = Managers::new();
        let displays = managers.enumerate();
        assert_eq!(displays.len(), 1);
        assert_eq!(displays[0].id, DEFAULT_DISPLAY);
        assert_eq!(displays[0].name, "Default");
        assert_eq!(
            displays[0].magnification,
            [0, 0, DISPLAY_WIDTH as i32, DISPLAY_HEIGHT as i32]
        );
        assert_eq!(displays[0].power, 1);

        assert!(managers.display(99).is_none());
        assert!(managers.set_display_alpha(DEFAULT_DISPLAY, 0.5));
        assert!(managers.set_display_power(DEFAULT_DISPLAY, 0));
        assert!(managers.set_display_stack(DEFAULT_DISPLAY, 3));
        assert_eq!(managers.display(DEFAULT_DISPLAY).unwrap().alpha, 0.5);
        assert_eq!(managers.display(DEFAULT_DISPLAY).unwrap().power, 0);
        assert_eq!(managers.display(DEFAULT_DISPLAY).unwrap().stack, 3);
        assert!(!managers.set_display_alpha(99, 0.5));
        assert!(!managers.set_default_display(99));
        assert!(managers.set_default_display(DEFAULT_DISPLAY));
    }

    #[test]
    fn content_visibility_and_conductor_layer_are_recorded() {
        let mut managers = Managers::new();
        assert!(managers.content_visible);
        assert_eq!(managers.conductor_layer, 0);
        managers.content_visible = false;
        managers.conductor_layer = PRESENTED_LAYER;
        assert!(!managers.content_visible);
        assert_eq!(managers.conductor_layer, PRESENTED_LAYER);
    }

    #[test]
    fn a_presented_layer_resize_reaches_the_present_path() {
        let slot = AtomicU64::new(0);
        sync_presented_size(&slot, PRESENTED_LAYER, 1600, 900);
        assert_eq!(unpack_layer(slot.load(Ordering::Relaxed)), (1600, 900));

        // Other layers and empty sizes leave the presented size alone.
        sync_presented_size(&slot, PRESENTED_LAYER + 1, 1280, 720);
        assert_eq!(unpack_layer(slot.load(Ordering::Relaxed)), (1600, 900));
        sync_presented_size(&slot, PRESENTED_LAYER, 0, 900);
        assert_eq!(unpack_layer(slot.load(Ordering::Relaxed)), (1600, 900));
    }

    #[test]
    fn manager_event_commands_map_to_their_slots() {
        assert_eq!(manager_event_index(2300), Some(LAYER_PRESENTING_EVENT));
        assert_eq!(manager_event_index(2302), Some(DISPLAY_HOTPLUG_EVENT));
        assert_eq!(manager_event_index(2303), Some(DISPLAY_MODE_CHANGED_EVENT));
        assert_eq!(manager_event_index(2601), Some(DISPLAY_ERROR_EVENT));
        assert_eq!(manager_event_index(2701), Some(DISPLAY_FATAL_ERROR_EVENT));
        assert_eq!(manager_event_index(2301), None);
        assert_eq!(manager_event_index(1102), None);
        let mut slots = [None; 5];
        for cmd in [2300, 2302, 2303, 2601, 2701] {
            let which = manager_event_index(cmd).unwrap();
            slots[which] = Some(cmd);
        }
        assert!(slots.iter().all(Option::is_some));
    }

    #[test]
    fn layer_size_falls_back_to_720p() {
        assert_eq!(unpack_layer(0), (FALLBACK_WIDTH, FALLBACK_HEIGHT));
        assert_eq!(unpack_layer(0), (1280, 720));
        assert_eq!(unpack_layer(1920u64 << 32 | 1080), (1920, 1080));
    }

    #[test]
    fn parcel_round_trip() {
        let mut w = ParcelWriter::default();
        w.i32(0x100);
        let name: Vec<u16> = "android.gui.IGraphicBufferProducer"
            .encode_utf16()
            .collect();
        w.i32(name.len() as i32);
        let mut s: Vec<u8> = name.iter().flat_map(|c| c.to_le_bytes()).collect();
        s.extend_from_slice(&[0, 0]);
        w.bytes(&s);
        w.i32(7);
        w.i32(0x10);
        w.i32(0);
        w.bytes(&[0xab; 0x10]);
        let parcel = w.finish();
        let mut r = ParcelReader::new(&parcel);
        r.skip_interface_token();
        assert_eq!(r.i32(), 7);
        let f = r.flattenable().unwrap();
        assert_eq!(f.len(), 8 + 0x10);
        assert_eq!(f[8], 0xab);
    }

    #[test]
    fn yuv_layers_are_video_and_rgba_layers_are_not() {
        let mut slot = yuv_slot(4, 2, 4);
        slot.format = 0x11;
        assert!(slot.is_video_format());
        for rgba in [1, 2, 3, 4, 5] {
            slot.format = rgba;
            assert!(!slot.is_video_format());
        }
        for yuv in [0x11, 0x13, 0x3231_564E, 0x3231_5659] {
            slot.format = yuv;
            assert!(slot.is_video_format());
        }
    }

    #[test]
    fn queue_buffer_timestamps_are_read_from_the_android_input() {
        let mut raw = vec![0u8; 0x60];
        raw[8..16].copy_from_slice(&123_456_789i64.to_le_bytes());
        assert_eq!(queue_timestamp(Some(&raw)), Timestamp::Guest(123_456_789));
        raw[16..20].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(queue_timestamp(Some(&raw)), Timestamp::Automatic);
        raw[16..20].copy_from_slice(&0u32.to_le_bytes());
        raw[8..16].copy_from_slice(&(-1i64).to_le_bytes());
        assert_eq!(queue_timestamp(Some(&raw)), Timestamp::Automatic);
        raw[8..16].copy_from_slice(&0i64.to_le_bytes());
        assert_eq!(queue_timestamp(Some(&raw)), Timestamp::Automatic);
        assert_eq!(queue_timestamp(None), Timestamp::Automatic);
    }
}
