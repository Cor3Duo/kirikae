use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::apm::PERFORMANCE_MODE;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const RESULT_NO_MESSAGES: u32 = 0x680;
const RESULT_NOT_AVAILABLE: u32 = 0x480;

const MESSAGE_FOCUS_STATE_CHANGED: u32 = 15;
const OPERATION_MODE: u8 = PERFORMANCE_MODE as u8;
const APPLET_RESOURCE_USER_ID: u64 = 0x60;
const LANGUAGE: &[u8; 5] = b"en-US";
/// Default and logical display resolution, reported by `ICommonStateGetter` command 60
/// and `IAppletCommonFunctions` command 40.
const DISPLAY_WIDTH: u32 = 1920;
const DISPLAY_HEIGHT: u32 = 1080;
/// `nn::am::service::IAppletCommonFunctions` ThemeStorage: 0x400 bytes, prefilled
/// with 0xAA until the guest overwrites it.
const THEME_STORAGE_SIZE: usize = 0x400;
const THEME_STORAGE_DEFAULT: u8 = 0xAA;
/// `nn::am::service::AppletResourceUsageInfo` is 0x20 bytes.
const APPLET_RESOURCE_USAGE_INFO_SIZE: usize = 0x20;
const APPLET_NOTIFICATION_EVENT: usize = 0;
const APPLET_GRC_EVENT: usize = 1;
/// `nn::apm::PerformanceConfiguration` for docked normal performance, matching
/// the configuration `apm/ISession` reports.
const PERFORMANCE_CONFIGURATION: u32 = 0x0002_0003;
/// Default application save and journal sizes. `fs` provisions saves with the same
/// values (see `crates/rt/src/services/fs.rs`).
const SAVE_DATA_SIZE: u64 = 0x200_0000;
const SAVE_JOURNAL_SIZE: u64 = 0x100_0000;
/// `nn::ncm::StorageId::BuiltInUser`: the title is not launched from an NCM
/// storage, but the built-in user partition is the closest non-removable answer.
const STORAGE_ID_BUILT_IN_USER: u8 = 1;
const DEFAULT_PSEUDO_DEVICE_ID: [u8; 16] = [
    0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
];

const COMMON_SLEEP_LOCK_EVENT: usize = 0;
const COMMON_HDCP_EVENT: usize = 1;
const COMMON_RESOLUTION_EVENT: usize = 2;
const APPLICATION_GPU_ERROR_EVENT: usize = 0;
const APPLICATION_FRIEND_EVENT: usize = 1;
const APPLICATION_NOTIFICATION_EVENT: usize = 2;
const APPLICATION_HEALTH_WARNING_EVENT: usize = 3;
const APPLICATION_UNKNOWN_EVENT: usize = 4;

#[derive(Clone, Copy, Debug)]
struct ExpectedMasterVolume {
    main: f32,
    library: f32,
}

impl Default for ExpectedMasterVolume {
    fn default() -> Self {
        ExpectedMasterVolume {
            main: 1.0,
            library: 1.0,
        }
    }
}

/// Flags the application sets through `am/ISelfController`. The runtime does not
/// act on them yet, but the applet system would, and the application can read
/// back the effects (accumulated suspended ticks, exit lock) it asked for.
#[derive(Default)]
struct SelfControllerState {
    exit_locked: AtomicBool,
    /// `Exit` was called while `exit_locked` was set. The exit is performed when
    /// the lock is released.
    exit_pending: AtomicBool,
    fatal_section: AtomicU32,
    operation_mode_changed_notification: AtomicBool,
    performance_mode_changed_notification: AtomicBool,
    /// SetFocusHandlingMode takes three flags packed one after another.
    focus_handling_mode: Mutex<[bool; 3]>,
    restart_message_enabled: AtomicBool,
    out_of_focus_suspending_enabled: AtomicBool,
    idle_time_detection_extension: AtomicU32,
    auto_sleep_disabled: AtomicBool,
}

impl SelfControllerState {
    /// Keeps the exit request until the application releases the exit lock.
    fn request_exit(&self) {
        self.exit_pending.store(true, Ordering::Relaxed);
    }

    fn take_pending_exit(&self) -> bool {
        !self.exit_locked.load(Ordering::Relaxed)
            && self.exit_pending.swap(false, Ordering::Relaxed)
    }
}

/// State owned by `am/ICommonStateGetter`. The applet never sleeps and is
/// always docked, so only the values the application can read back are kept.
#[derive(Default)]
struct CommonState {
    vr_mode_enabled: AtomicBool,
    controller_firmware_update_section: AtomicBool,
}

fn input_bool(data: &[u8], off: usize) -> bool {
    data.get(off).is_some_and(|&b| b != 0)
}

/// Time the applet spent suspended, relative to when the applet service was
/// created.
#[derive(Default)]
struct Suspension {
    accumulated: u64,
    started: Option<u64>,
}

/// State owned by `am/IProcessWindingController`: the context stack and the
/// reservation a library applet makes before winding to another applet.
#[derive(Default)]
struct WindingState {
    contexts: Vec<Vec<u8>>,
    reserved: bool,
}

impl Suspension {
    fn total(&self, now: u64) -> u64 {
        self.accumulated + self.started.map_or(0, |at| now.saturating_sub(at))
    }
}

pub struct AmState {
    messages: Mutex<VecDeque<u32>>,
    message_event: u32,
    launchable_event: u32,
    suspended_tick_event: u32,
    user_popped: AtomicBool,
    /// Launch parameters queued for `IApplicationFunctions::PopLaunchParameter`
    /// kind 0 (`UserChannel`).
    user_channel: Mutex<VecDeque<Vec<u8>>>,
    expected_master_volume: Mutex<ExpectedMasterVolume>,
    self_controller: SelfControllerState,
    common: CommonState,
    common_events: Mutex<[Option<u32>; 3]>,
    application_events: Mutex<[Option<u32>; 5]>,
    suspension: Mutex<Suspension>,
    request_exit_to_library_applet: AtomicBool,
    save_data_size: AtomicU64,
    journal_size: AtomicU64,
    launch_required_version: AtomicU32,
    /// `InitializeGamePlayRecording` transfer-memory size, then
    /// `SetGamePlayRecordingState`'s enable flag.
    recording_size: AtomicU64,
    recording_state: AtomicU32,
    /// `am/IAppletCommonFunctions` ThemeStorage. The runtime keeps it in memory:
    /// a dump run should not leave guest state behind on the host.
    theme: Mutex<Vec<u8>>,
    /// `BeginBlockingHomeButton*` calls that have not been ended yet.
    home_button_blocked: AtomicU32,
    home_button_double_click: AtomicBool,
    winding: Mutex<WindingState>,
    /// Events created on first request: the notification receiver event and the
    /// `GetGrcProcessLaunchedSystemEvent` handle.
    applet_events: Mutex<[Option<u32>; 2]>,
    /// Notifications accepted by `INotificationSender` and not received yet.
    pending_notifications: AtomicU32,
    terminate_result: AtomicU32,
}

fn cached_event<const N: usize>(c: &Cpu, slots: &Mutex<[Option<u32>; N]>, which: usize) -> u32 {
    *slots.lock().unwrap()[which].get_or_insert_with(|| c.process().create_event(false))
}

impl AmState {
    fn update_event(&self, c: &Cpu) {
        let pending = !self.messages.lock().unwrap().is_empty();
        c.process().set_event(self.message_event, pending);
    }

    /// Accumulated suspended ticks in the same unit as `Process::ticks`. The
    /// runtime has no suspend path yet, so this stays at zero while the applet
    /// runs; the accounting lives here so `GetAccumulatedSuspendedTickValue`
    /// reads the runtime tick source instead of a fixed constant.
    fn accumulated_suspended_ticks(&self, now: u64) -> u64 {
        self.suspension.lock().unwrap().total(now)
    }

    #[cfg(test)]
    fn suspend(&self, now: u64) {
        self.suspension.lock().unwrap().started = Some(now);
    }

    #[cfg(test)]
    fn resume(&self, now: u64) {
        let mut suspension = self.suspension.lock().unwrap();
        let total = suspension.total(now);
        suspension.accumulated = total;
        suspension.started = None;
    }

    /// `nn::am::service::IApplicationFunctions::InitializeGamePlayRecording`.
    ///
    /// The runtime records nothing, but it accepts the transfer memory so the SDK's
    /// recording clients are constructed against a valid applet state. The size is kept
    /// so a later build can model the buffer.
    fn initialize_game_play_recording(&self, size: u64) -> Reply {
        self.recording_size.store(size, Ordering::Relaxed);
        Reply::ok()
    }

    fn set_game_play_recording_state(&self, state: u32) -> Reply {
        self.recording_state.store(state, Ordering::Relaxed);
        Reply::ok()
    }

    #[cfg(test)]
    fn recording(&self) -> (u64, u32) {
        (
            self.recording_size.load(Ordering::Relaxed),
            self.recording_state.load(Ordering::Relaxed),
        )
    }

    /// ThemeStorage bytes the guest can read at `offset`, clamped to both the
    /// stored buffer and the output buffer it asked for.
    fn theme_slice(&self, offset: u64, len: usize) -> Vec<u8> {
        let theme = self.theme.lock().unwrap();
        let off = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(theme.len());
        let n = len.min(theme.len() - off);
        theme[off..off + n].to_vec()
    }

    /// Writes ThemeStorage at `offset`; returns the number of bytes stored.
    fn write_theme(&self, offset: u64, bytes: &[u8]) -> usize {
        let mut theme = self.theme.lock().unwrap();
        let off = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(theme.len());
        let n = bytes.len().min(theme.len() - off);
        theme[off..off + n].copy_from_slice(&bytes[..n]);
        n
    }

    /// Queues one notification for the receiver; returns the pending count.
    fn send_notification(&self) -> u32 {
        self.pending_notifications
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1)
    }

    /// Takes one pending notification; returns the remaining count, or `None`
    /// when the queue was empty.
    fn receive_notification(&self) -> Option<u32> {
        let mut pending = self.pending_notifications.load(Ordering::Relaxed);
        while pending != 0 {
            match self.pending_notifications.compare_exchange_weak(
                pending,
                pending - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(pending - 1),
                Err(actual) => pending = actual,
            }
        }
        None
    }

    /// `IApplicationFunctions::SetTerminateResult` and
    /// `IAppletCommonFunctions::SetTerminateResult` share the application result.
    fn set_terminate_result(&self, c: &Cpu, result: u32) {
        self.terminate_result.store(result, Ordering::Relaxed);
        c.process().warn_once(format!(
            "the applet set its terminate result to {result:#x} (module {}, description {})",
            result & 0x1ff,
            result >> 9 & 0x1fff
        ));
    }
}

pub struct AppletOe;

impl Service for AppletOe {
    fn name(&self) -> String {
        "appletOE".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let state = Arc::new(AmState {
                    messages: Mutex::new(VecDeque::from([MESSAGE_FOCUS_STATE_CHANGED])),
                    message_event: c.process().create_event(true),
                    launchable_event: c.process().create_event(true),
                    suspended_tick_event: c.process().create_event(false),
                    user_popped: AtomicBool::new(false),
                    user_channel: Mutex::new(VecDeque::new()),
                    expected_master_volume: Mutex::new(ExpectedMasterVolume::default()),
                    self_controller: SelfControllerState::default(),
                    common: CommonState::default(),
                    common_events: Mutex::new([None; 3]),
                    application_events: Mutex::new([None; 5]),
                    suspension: Mutex::new(Suspension::default()),
                    request_exit_to_library_applet: AtomicBool::new(false),
                    save_data_size: AtomicU64::new(SAVE_DATA_SIZE),
                    journal_size: AtomicU64::new(SAVE_JOURNAL_SIZE),
                    launch_required_version: AtomicU32::new(0),
                    recording_size: AtomicU64::new(0),
                    recording_state: AtomicU32::new(0),
                    theme: Mutex::new(vec![THEME_STORAGE_DEFAULT; THEME_STORAGE_SIZE]),
                    home_button_blocked: AtomicU32::new(0),
                    home_button_double_click: AtomicBool::new(false),
                    winding: Mutex::new(WindingState::default()),
                    applet_events: Mutex::new([None; 2]),
                    pending_notifications: AtomicU32::new(0),
                    terminate_result: AtomicU32::new(0),
                });
                Reply::ok().object(Arc::new(ApplicationProxy { state }))
            }
            _ => unimplemented("appletOE", c, req),
        }
    }
}

struct ApplicationProxy {
    state: Arc<AmState>,
}

impl ApplicationProxy {
    /// The manager object behind an `am/IApplicationProxy` command. Keeping the
    /// table separate from `call` lets the tests check each command's object
    /// without a guest CPU.
    fn object(&self, cmd: u32) -> Option<Arc<dyn Service>> {
        let obj: Arc<dyn Service> = match cmd {
            0 => Arc::new(CommonStateGetter {
                state: self.state.clone(),
            }),
            1 => Arc::new(SelfController {
                state: self.state.clone(),
            }),
            2 => Arc::new(WindowController),
            3 => Arc::new(AudioController {
                state: self.state.clone(),
            }),
            4 => Arc::new(DisplayController),
            10 => Arc::new(ProcessWindingController {
                state: self.state.clone(),
            }),
            11 => Arc::new(LibraryAppletCreator),
            20 => Arc::new(ApplicationFunctions {
                state: self.state.clone(),
            }),
            // `IApplicationProxy` has no GetAppletCommonFunctions on hardware: a regular
            // application reaches it through the system/library/overlay proxies
            // (commands 23/21/21), which the runtime does not implement. The single `am`
            // client is served from this table, and command 21 is the number the library
            // and overlay proxies use.
            21 => Arc::new(AppletCommonFunctions {
                state: self.state.clone(),
            }),
            1000 => Arc::new(DebugFunctions {
                state: self.state.clone(),
            }),
            _ => return None,
        };
        Some(obj)
    }
}

impl Service for ApplicationProxy {
    fn name(&self) -> String {
        "am/IApplicationProxy".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.object(req.cmd) {
            Some(obj) => Reply::ok().object(obj),
            None => unimplemented("am/IApplicationProxy", c, req),
        }
    }
}

struct AudioController {
    state: Arc<AmState>,
}

fn input_f32(data: &[u8], off: usize) -> Option<f32> {
    let bytes: [u8; 4] = data.get(off..off + 4)?.try_into().ok()?;
    Some(f32::from_le_bytes(bytes))
}

impl AudioController {
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            0 => {
                let mut volume = self.state.expected_master_volume.lock().unwrap();
                if let Some(main) = input_f32(&req.data, 0) {
                    volume.main = main;
                }
                if let Some(library) = input_f32(&req.data, 4) {
                    volume.library = library;
                }
                Some(Reply::ok())
            }
            1 => {
                let volume = self.state.expected_master_volume.lock().unwrap().main;
                Some(Reply::ok().u32(volume.to_bits()))
            }
            2 => {
                let volume = self.state.expected_master_volume.lock().unwrap().library;
                Some(Reply::ok().u32(volume.to_bits()))
            }
            _ => None,
        }
    }
}

impl Service for AudioController {
    fn name(&self) -> String {
        "am/IAudioController".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented("am/IAudioController", c, req),
        }
    }
}

/// `nn::am::service::IDisplayController`, returned by `IApplicationProxy` command 4.
///
/// The runtime has no capture pipeline, so no screenshot or capture image exists and no
/// guest buffer is ever filled with invented pixels. The plain image queries report the
/// documented stub error, the `Ex` queries report "no image" through their bool, and the
/// shared-buffer queries report that no buffer is captured.
struct DisplayController;

impl DisplayController {
    fn command(&self, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            // GetLastForegroundCaptureImage, GetLastApplicationCaptureImage,
            // GetCallerAppletCaptureImage: stubbed with an error on hardware.
            0 | 2 | 3 => Reply::err(RESULT_NOT_AVAILABLE),
            // UpdateLastForegroundCaptureImage and UpdateCallerAppletCaptureImage ask the
            // system to refresh an image the runtime never keeps.
            1 | 4 => Reply::ok(),
            // Get*CaptureImageEx: the bool says whether an image was copied. There is
            // none, so the guest buffer is left untouched.
            5..=7 => Reply::ok().u8(0),
            // TakeScreenShotOfOwnLayer/Ex, CopyBetweenCaptureBuffers, ClearCaptureBuffer
            // and ClearAppletTransitionBuffer have nothing to operate on.
            8 | 9 | 20 | 21 | 28 => Reply::ok(),
            // Acquire*CaptureBuffer and Acquire*CaptureBufferEx hand out a handle the
            // runtime cannot back.
            10 | 12 | 14 | 16 | 17 | 18 => Reply::err(RESULT_NOT_AVAILABLE),
            // Release*CaptureBuffer: nothing was acquired.
            11 | 13 | 15 => Reply::err(RESULT_NOT_AVAILABLE),
            // Acquire*CaptureSharedBuffer: (bool captured, s32 shared buffer id).
            22 | 24 | 26 => Reply::ok().u8(0).u32(0),
            // Release*CaptureSharedBuffer: no shared buffer is held.
            23 | 25 | 27 => Reply::err(RESULT_NOT_AVAILABLE),
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for DisplayController {
    fn name(&self) -> String {
        "am/IDisplayController".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented("am/IDisplayController", c, req),
        }
    }
}

/// `nn::am::service::IProcessWindingController`, returned by `IApplicationProxy` command 10.
///
/// Winding is how a library applet hands the foreground to another applet and waits for it.
/// The runtime runs a single applet, so there is never a calling applet to open, and the
/// reserve commands only record a reservation the guest can cancel or consume.
struct ProcessWindingController {
    state: Arc<AmState>,
}

impl ProcessWindingController {
    fn command(&self, req: &Request) -> Option<Reply> {
        let mut winding = self.state.winding.lock().unwrap();
        let reply = match req.cmd {
            // GetLaunchReason: `AppletProcessLaunchReason` (0x4). The flag says whether
            // `OpenCallingLibraryApplet` can be used; nothing launched this applet.
            0 => Reply::ok().u8(0).bytes(&[0; 3]),
            // OpenCallingLibraryApplet: there is no calling applet object in state.
            11 => Reply::err(RESULT_NOT_AVAILABLE),
            // PushContext takes an IStorage. The IPC layer does not resolve input
            // objects for services, so the entry records the push without its contents.
            21 => {
                winding.contexts.push(Vec::new());
                Reply::ok()
            }
            22 => match winding.contexts.pop() {
                Some(data) => Reply::ok().object(Arc::new(Storage::new(data))),
                None => Reply::err(RESULT_NOT_AVAILABLE),
            },
            23 => {
                winding.reserved = false;
                Reply::ok()
            }
            // WindAndDoReserved with nothing reserved is the unsupported combination.
            30 => {
                if winding.reserved {
                    winding.reserved = false;
                    Reply::ok()
                } else {
                    Reply::err(RESULT_NOT_AVAILABLE)
                }
            }
            // ReserveToStartAndWait[AndUnwindThis] take an ILibraryAppletAccessor. The
            // runtime has no second applet to start, so the reservation is only recorded.
            40 | 41 => {
                winding.reserved = true;
                Reply::ok()
            }
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for ProcessWindingController {
    fn name(&self) -> String {
        "am/IProcessWindingController".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented("am/IProcessWindingController", c, req),
        }
    }
}

struct CommonStateGetter {
    state: Arc<AmState>,
}

enum CommonCommand {
    Reply(Reply),
    Event(usize),
}

impl CommonStateGetter {
    fn command(&self, req: &Request) -> Option<CommonCommand> {
        use CommonCommand::{Event, Reply as Data};
        let common = &self.state.common;
        let r = match req.cmd {
            0 => Data(Reply::ok().copy_handle(self.state.message_event)),
            // The applet never enters sleep: every request is accepted, and the
            // acquired-sleep-lock event is already signalled.
            3 | 4 => Data(Reply::ok()),
            5 => Data(Reply::ok().u8(OPERATION_MODE)),
            6 => Data(Reply::ok().u32(PERFORMANCE_MODE)),
            7 => Data(Reply::ok().u8(0)),
            8 => Data(Reply::ok().u8(0)),
            9 => Data(Reply::ok().u8(1)),
            11 | 12 => Data(Reply::ok()),
            13 => Event(COMMON_SLEEP_LOCK_EVENT),
            // The runtime has no wake-up bookkeeping.
            14 => Data(Reply::ok().u64(0)),
            // PushToGeneralChannel takes an IStorage; the applet has no reader.
            20 => Data(Reply::ok()),
            40 => Data(Reply::ok().u32(0).u32(0).u32(0).u32(0)),
            50 => Data(Reply::ok().u8(common.vr_mode_enabled.load(Ordering::Relaxed) as u8)),
            51 => {
                common
                    .vr_mode_enabled
                    .store(input_bool(&req.data, 0), Ordering::Relaxed);
                Data(Reply::ok())
            }
            52 => Data(Reply::ok()),
            53 => {
                common.vr_mode_enabled.store(true, Ordering::Relaxed);
                Data(Reply::ok())
            }
            54 => {
                common.vr_mode_enabled.store(false, Ordering::Relaxed);
                Data(Reply::ok())
            }
            55 => Data(
                Reply::ok().u8(common
                    .controller_firmware_update_section
                    .load(Ordering::Relaxed) as u8),
            ),
            60 => Data(Reply::ok().u32(DISPLAY_WIDTH).u32(DISPLAY_HEIGHT)),
            61 => Event(COMMON_RESOLUTION_EVENT),
            // HDCP is never authenticated and no TV is attached.
            62 => Data(Reply::ok().u32(0)),
            63 => Event(COMMON_HDCP_EVENT),
            64 | 66 | 67 | 80 | 90 | 100 | 400 | 401 | 500 | 501 => Data(Reply::ok()),
            // No content action name maps to an application in an extracted dump.
            65 => Data(Reply::ok().u64(0)),
            // Built-in LCD, never a cradle display.
            68 => Data(Reply::ok().u32(0)),
            91 => Data(Reply::ok().u32(PERFORMANCE_CONFIGURATION)),
            // No operation-mode policy is configured.
            200 => Data(Reply::ok().u32(0)),
            // PlatformRegion::Global.
            300 => Data(Reply::ok().u8(0)),
            // SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled: no input and
            // no output (https://switchbrew.org/wiki/Applet_Manager_services). The
            // applet system would ask the application to exit into the library
            // applet the next time it executes; the runtime only records the request.
            900 => Data(self.enable_request_exit_to_library_applet()),
            _ => return None,
        };
        Some(r)
    }
}

impl Service for CommonStateGetter {
    fn name(&self) -> String {
        "am/ICommonStateGetter".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        if req.cmd == 1 {
            let msg = self.state.messages.lock().unwrap().pop_front();
            self.state.update_event(c);
            return match msg {
                Some(m) => Reply::ok().u32(m),
                None => Reply::err(RESULT_NO_MESSAGES),
            };
        }
        if req.cmd == 10 {
            let event = cached_event(c, &self.state.common_events, COMMON_SLEEP_LOCK_EVENT);
            c.process().set_event(event, true);
            return Reply::ok();
        }
        // Home-button reader/writer locks and the manual GPU error handler are objects, so
        // they need a fresh event handle per accessor.
        if matches!(req.cmd, 30..=32) {
            return Reply::ok().object(Arc::new(LockAccessor {
                locked: Mutex::new(false),
                event: c.process().create_event(false),
            }));
        }
        if req.cmd == 110 {
            return Reply::ok().object(Arc::new(GpuErrorHandler {
                event: c.process().create_event(false),
            }));
        }
        match self.command(req) {
            Some(CommonCommand::Reply(r)) => r,
            Some(CommonCommand::Event(which)) => {
                Reply::ok().copy_handle(cached_event(c, &self.state.common_events, which))
            }
            None => unimplemented("am/ICommonStateGetter", c, req),
        }
    }
}

impl CommonStateGetter {
    fn enable_request_exit_to_library_applet(&self) -> Reply {
        self.state
            .request_exit_to_library_applet
            .store(true, Ordering::Relaxed);
        Reply::ok()
    }
}

/// `nn::am::service::ILockAccessor`, handed out by `ICommonStateGetter` commands 30-32.
///
/// The home-button reader/writer locks are advisory: the console serialises the applets that
/// write the home-button state, and an extracted dump runs one applet, so the runtime only
/// keeps the lock state and its event consistent so `TryLock`/`Unlock` round-trip.
struct LockAccessor {
    locked: Mutex<bool>,
    event: u32,
}

impl LockAccessor {
    fn try_lock(&self) -> bool {
        let mut locked = self.locked.lock().unwrap();
        if *locked {
            return false;
        }
        *locked = true;
        true
    }

    fn unlock(&self) -> bool {
        std::mem::replace(&mut *self.locked.lock().unwrap(), false)
    }

    fn is_locked(&self) -> bool {
        *self.locked.lock().unwrap()
    }

    fn command(&self, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            // `TryLock`: the input flag asks for the event handle (official software passes
            // false and closes the handle without waiting); the output flag says whether the
            // lock was taken.
            1 => {
                let reply = Reply::ok().u8(self.try_lock() as u8);
                if input_bool(&req.data, 0) {
                    reply.copy_handle(self.event)
                } else {
                    reply
                }
            }
            2 => {
                self.unlock();
                Reply::ok()
            }
            3 => Reply::ok().copy_handle(self.event),
            4 => Reply::ok().u8(self.is_locked() as u8),
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for LockAccessor {
    fn name(&self) -> String {
        "am/ILockAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let reply = self.command(req);
        if req.cmd == 2 {
            c.process().set_event(self.event, false);
        }
        match reply {
            Some(reply) => reply,
            None => unimplemented("am/ILockAccessor", c, req),
        }
    }
}

/// `nn::am::service::IGpuErrorHandler`, returned by `ICommonStateGetter` command 110.
///
/// There is no GPU hang to report, so the manual error info is empty and the detection event
/// stays unsignalled.
struct GpuErrorHandler {
    event: u32,
}

impl GpuErrorHandler {
    fn command(&self, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            // GetManualGpuErrorInfoSize / GetManualGpuErrorInfo: no error is pending.
            100 | 101 => Reply::ok().u64(0),
            102 => Reply::ok().copy_handle(self.event),
            103 => Reply::ok(),
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for GpuErrorHandler {
    fn name(&self) -> String {
        "am/IGpuErrorHandler".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(reply) => reply,
            None => unimplemented("am/IGpuErrorHandler", c, req),
        }
    }
}

struct SelfController {
    state: Arc<AmState>,
}

impl SelfController {
    fn command(&self, req: &Request, now: u64) -> Option<Reply> {
        let sc = &self.state.self_controller;
        match req.cmd {
            // Exit is honoured after the reply; if the application holds the exit
            // lock the request waits until UnlockExit.
            0 => {
                sc.request_exit();
                Some(Reply::ok())
            }
            // LockExit/UnlockExit only gate the exit the applet system performs on
            // the application's behalf; the application's own exit path is untouched.
            1 => {
                sc.exit_locked.store(true, Ordering::Relaxed);
                Some(Reply::ok())
            }
            2 => {
                sc.exit_locked.store(false, Ordering::Relaxed);
                Some(Reply::ok())
            }
            3 => {
                sc.fatal_section.fetch_add(1, Ordering::Relaxed);
                Some(Reply::ok())
            }
            4 => {
                let _ = sc
                    .fatal_section
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                        Some(n.saturating_sub(1))
                    });
                Some(Reply::ok())
            }
            9 => Some(Reply::ok().copy_handle(self.state.launchable_event)),
            10 => Some(Reply::ok()),
            11 => {
                sc.operation_mode_changed_notification
                    .store(input_bool(&req.data, 0), Ordering::Relaxed);
                Some(Reply::ok())
            }
            12 => {
                sc.performance_mode_changed_notification
                    .store(input_bool(&req.data, 0), Ordering::Relaxed);
                Some(Reply::ok())
            }
            13 => {
                *sc.focus_handling_mode.lock().unwrap() = [
                    input_bool(&req.data, 0),
                    input_bool(&req.data, 1),
                    input_bool(&req.data, 2),
                ];
                Some(Reply::ok())
            }
            14 => {
                sc.restart_message_enabled
                    .store(input_bool(&req.data, 0), Ordering::Relaxed);
                Some(Reply::ok())
            }
            16 => {
                sc.out_of_focus_suspending_enabled
                    .store(input_bool(&req.data, 0), Ordering::Relaxed);
                Some(Reply::ok())
            }
            // Setters whose effect is not observable on this runtime (screenshot
            // and album policy, keyboard layout, display separation, playback).
            15 | 18 | 19 | 20 | 41 | 45 | 46 | 50 | 51 | 60 | 61 | 64 | 65 | 70 | 72 | 80 | 100
            | 110 | 120 | 130 => Some(Reply::ok()),
            17 => {
                self.state
                    .common
                    .controller_firmware_update_section
                    .store(input_bool(&req.data, 0), Ordering::Relaxed);
                Some(Reply::ok())
            }
            40 => Some(Reply::ok().u64(1)),
            44 => Some(Reply::ok().u64(1).u64(2)),
            62 => {
                sc.idle_time_detection_extension
                    .store(req.u32(0), Ordering::Relaxed);
                Some(Reply::ok())
            }
            63 => Some(Reply::ok().u32(sc.idle_time_detection_extension.load(Ordering::Relaxed))),
            // The application has no ambient-light sensor; report a bright room so
            // titles that gate on it take the normal path.
            66 => Some(Reply::ok().u32(10_000.0f32.to_bits())),
            67 => Some(Reply::ok().u8(1)),
            68 => {
                sc.auto_sleep_disabled
                    .store(input_bool(&req.data, 0), Ordering::Relaxed);
                Some(Reply::ok())
            }
            69 => Some(Reply::ok().u8(sc.auto_sleep_disabled.load(Ordering::Relaxed) as u8)),
            71 => Some(Reply::ok().u8(1).u32(10_000.0f32.to_bits())),
            90 => Some(Reply::ok().u64(self.state.accumulated_suspended_ticks(now))),
            91 => Some(Reply::ok().copy_handle(self.state.suspended_tick_event)),
            _ => None,
        }
    }
}

impl Service for SelfController {
    fn name(&self) -> String {
        "am/ISelfController".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let r = match self.command(req, c.process().ticks()) {
            Some(r) => r,
            None => unimplemented("am/ISelfController", c, req),
        };
        // Exit and UnlockExit can complete a deferred exit request.
        if matches!(req.cmd, 0 | 2) && self.state.self_controller.take_pending_exit() {
            c.process().terminate(0);
        }
        r
    }
}

struct WindowController;

impl Service for WindowController {
    fn name(&self) -> String {
        "am/IWindowController".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            1 => Reply::ok().u64(APPLET_RESOURCE_USER_ID),
            10 => Reply::ok(),
            _ => unimplemented("am/IWindowController", c, req),
        }
    }
}

struct ApplicationFunctions {
    state: Arc<AmState>,
}

const LAUNCH_USER_CHANNEL: u32 = 1;
const LAUNCH_PRESELECTED_USER: u32 = 2;

fn preselected_user() -> Vec<u8> {
    let mut v = 0xc794_97cau32.to_le_bytes().to_vec();
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&crate::services::acc::USER_ID);
    v.resize(0x88, 0);
    v
}

pub struct Storage {
    data: Arc<Mutex<Vec<u8>>>,
}

impl Storage {
    pub fn new(data: Vec<u8>) -> Self {
        Storage {
            data: Arc::new(Mutex::new(data)),
        }
    }
}

impl Service for Storage {
    fn name(&self) -> String {
        "am/IStorage".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(StorageAccessor {
                data: self.data.clone(),
            })),
            _ => unimplemented("am/IStorage", c, req),
        }
    }
}

struct StorageAccessor {
    data: Arc<Mutex<Vec<u8>>>,
}

impl Service for StorageAccessor {
    fn name(&self) -> String {
        "am/IStorageAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(self.data.lock().unwrap().len() as u64),
            10 => {
                let mut data = self.data.lock().unwrap();
                let off = (req.u64(0) as usize).min(data.len());
                if let Some(b) = req.in_buf(0) {
                    let n = (b.size as usize).min(data.len() - off);
                    if let Some(bytes) = c.process().memory.read(b.addr, n as u64) {
                        data[off..off + n].copy_from_slice(&bytes);
                    }
                }
                Reply::ok()
            }
            11 => {
                let data = self.data.lock().unwrap();
                let off = (req.u64(0) as usize).min(data.len());
                if let Some(b) = req.out_buf(0) {
                    let n = (b.size as usize).min(data.len() - off);
                    c.process().memory.write(b.addr, &data[off..off + n]);
                }
                Reply::ok()
            }
            _ => unimplemented("am/IStorageAccessor", c, req),
        }
    }
}

const APPLET_CONTROLLER: u32 = 0xc;
const NPAD_HANDHELD: u32 = 0x20;

struct LibraryAppletCreator;

impl Service for LibraryAppletCreator {
    fn name(&self) -> String {
        "am/ILibraryAppletCreator".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let id = req.u32(0);
                let p = c.process();
                p.log_ipc(format!("am: library applet {id:#x}"));
                Reply::ok().object(Arc::new(LibraryAppletAccessor {
                    id,
                    state_event: p.create_event(false),
                    out_event: p.create_event(false),
                    output: Mutex::new(None),
                }))
            }
            10 => Reply::ok().object(Arc::new(Storage::new(vec![
                0;
                req.u64(0).min(1 << 24) as usize
            ]))),
            _ => unimplemented("am/ILibraryAppletCreator", c, req),
        }
    }
}

struct LibraryAppletAccessor {
    id: u32,
    state_event: u32,
    out_event: u32,
    output: Mutex<Option<Vec<u8>>>,
}

impl LibraryAppletAccessor {
    fn result(&self) -> Vec<u8> {
        match self.id {
            APPLET_CONTROLLER => {
                let mut v = vec![1, 0, 0, 0];
                v.extend_from_slice(&NPAD_HANDHELD.to_le_bytes());
                v.extend_from_slice(&0u32.to_le_bytes());
                v
            }
            _ => Vec::new(),
        }
    }
}

impl Service for LibraryAppletAccessor {
    fn name(&self) -> String {
        "am/ILibraryAppletAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 => Reply::ok().copy_handle(self.state_event),
            1 => Reply::ok().u8(self.output.lock().unwrap().is_some() as u8 | 1),
            10 => {
                let out = self.result();
                if !out.is_empty() {
                    *self.output.lock().unwrap() = Some(out);
                    p.set_event(self.out_event, true);
                }
                p.set_event(self.state_event, true);
                Reply::ok()
            }
            // GetResult has no output for a library applet: the exit reason is
            // reported through the application accessor, not here.
            30 => Reply::ok(),
            20 | 25 | 50 | 60 | 150 => Reply::ok(),
            // PushInData/PushExtraStorage/PushInteractiveInData take an IStorage and
            // produce no output. The built-in applets ignore their input.
            100 | 102 | 103 => Reply::ok(),
            101 | 104 => match self.output.lock().unwrap().take() {
                Some(data) => {
                    p.set_event(self.out_event, false);
                    Reply::ok().object(Arc::new(Storage::new(data)))
                }
                None => Reply::err(RESULT_NOT_AVAILABLE),
            },
            105 => Reply::ok().copy_handle(self.out_event),
            106 => Reply::ok().copy_handle(p.create_event(false)),
            110 => Reply::ok().u8(0),
            _ => unimplemented("am/ILibraryAppletAccessor", c, req),
        }
    }
}

fn language_code() -> u64 {
    let mut b = [0u8; 8];
    b[..LANGUAGE.len()].copy_from_slice(LANGUAGE);
    u64::from_le_bytes(b)
}

enum ApplicationCommand {
    Reply(Reply),
    Event(usize),
    /// The reply is sent and the process ends right after it: the applet system
    /// would power the console down/reboot or tear the application down.
    Exit(Reply),
}

impl ApplicationFunctions {
    fn command(&self, req: &Request) -> Option<ApplicationCommand> {
        use ApplicationCommand::{Event, Exit, Reply as Data};
        let s = &self.state;
        let r = match req.cmd {
            1 => match req.u32(0) {
                LAUNCH_USER_CHANNEL => match s.user_channel.lock().unwrap().pop_front() {
                    Some(data) => Data(Reply::ok().object(Arc::new(Storage::new(data)))),
                    None => Data(Reply::err(RESULT_NOT_AVAILABLE)),
                },
                LAUNCH_PRESELECTED_USER if !s.user_popped.swap(true, Ordering::Relaxed) => {
                    Data(Reply::ok().object(Arc::new(Storage::new(preselected_user()))))
                }
                _ => Data(Reply::err(RESULT_NOT_AVAILABLE)),
            },
            // CreateApplicationAndPushAndRequestToStart variants carry an
            // application id and a storage object. The runtime has no application
            // registry, so the request is accepted without starting anything.
            10 | 11 | 13 | 14 | 15 => Data(Reply::ok()),
            12 => Data(Reply::ok()),
            20 => Data(Reply::ok().u64(s.save_data_size.load(Ordering::Relaxed))),
            21 => Data(Reply::ok().u64(language_code())),
            22 => Data(Reply::ok()),
            23 => Data(Reply::ok().bytes(b"1.0.0\0\0\0\0\0\0\0\0\0\0\0")),
            // GetLaunchStorageInfoForDebug: (application, update) storage ids.
            24 => Data(
                Reply::ok()
                    .u8(STORAGE_ID_BUILT_IN_USER)
                    .u8(STORAGE_ID_BUILT_IN_USER),
            ),
            // ExtendSaveData(save type, uid, save size, journal size).
            25 => {
                s.save_data_size.store(req.u64(0x18), Ordering::Relaxed);
                s.journal_size.store(req.u64(0x20), Ordering::Relaxed);
                Data(Reply::ok().u64(0))
            }
            // GetSaveDataSize and GetSaveDataSizeMax.
            26 | 28 => Data(
                Reply::ok()
                    .u64(s.save_data_size.load(Ordering::Relaxed))
                    .u64(s.journal_size.load(Ordering::Relaxed)),
            ),
            // Begin/EndBlockingHomeButton and the ShortAndLongPressed pair. The count is
            // what `IAppletCommonFunctions::IsHomeButtonShortPressedBlocked` reports.
            30 | 32 => {
                s.home_button_blocked.fetch_add(1, Ordering::Relaxed);
                Data(Reply::ok())
            }
            31 | 33 => {
                let _ =
                    s.home_button_blocked
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                            Some(n.saturating_sub(1))
                        });
                Data(Reply::ok())
            }
            40 => Data(Reply::ok().u8(1)),
            50 => Data(Reply::ok().bytes(&DEFAULT_PSEUDO_DEVICE_ID)),
            60 => Data(Reply::ok()),
            65 => Data(Reply::ok().u8(1)),
            66 => Data(s.initialize_game_play_recording(req.u64(0))),
            67 => Data(s.set_game_play_recording_state(req.u32(0))),
            68 => Data(Reply::ok()),
            70 | 71 | 72 | 80 | 310 => Exit(Reply::ok()),
            90 => Data(Reply::ok()),
            100..=102 => Data(Reply::ok()),
            // No play statistics are recorded, so the query returns an empty list.
            110 | 111 => Data(Reply::ok().u32(0)),
            120 => Data(Reply::ok()),
            121 => {
                s.user_channel.lock().unwrap().clear();
                Data(Reply::ok())
            }
            // UnpopToUserChannel carries an IStorage object; the runtime's IPC layer
            // does not resolve domain objects for services, so the channel push is a
            // no-op (and the channel stays empty for PopLaunchParameter).
            122 => Data(Reply::ok()),
            123 => Data(Reply::ok().u32(0)),
            124 => Data(Reply::ok()),
            130 => Event(APPLICATION_GPU_ERROR_EVENT),
            131 => Data(Reply::ok()),
            140 => Event(APPLICATION_FRIEND_EVENT),
            141 | 151 => Data(Reply::err(RESULT_NOT_AVAILABLE)),
            150 => Event(APPLICATION_NOTIFICATION_EVENT),
            160 => Event(APPLICATION_HEALTH_WARNING_EVENT),
            170 => Data(Reply::ok()),
            180 => Data(Reply::ok().u32(s.launch_required_version.load(Ordering::Relaxed))),
            181 => {
                s.launch_required_version
                    .store(req.u32(0), Ordering::Relaxed);
                Data(Reply::ok())
            }
            200 => Data(Reply::ok().u32(0)),
            210 => Event(APPLICATION_UNKNOWN_EVENT),
            220 => Data(Reply::ok()),
            330 => Data(Reply::ok().u8(0)),
            500 | 1001 => Data(Reply::ok()),
            _ => return None,
        };
        Some(r)
    }
}

impl Service for ApplicationFunctions {
    fn name(&self) -> String {
        "am/IApplicationFunctions".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        if req.cmd == 22 {
            self.state.set_terminate_result(c, req.u32(0));
        } else if matches!(req.cmd, 12 | 120) {
            c.process().log_ipc(format!(
                "am: application requested program start {:#x}",
                req.u64(if req.cmd == 12 { 0 } else { 8 })
            ));
        }
        match self.command(req) {
            Some(ApplicationCommand::Reply(r)) => r,
            Some(ApplicationCommand::Event(which)) => {
                Reply::ok().copy_handle(cached_event(c, &self.state.application_events, which))
            }
            Some(ApplicationCommand::Exit(r)) => {
                c.process().terminate(0);
                r
            }
            None => unimplemented("am/IApplicationFunctions", c, req),
        }
    }
}

/// `nn::am::service::IDebugFunctions`, returned by `IApplicationProxy` command 1000.
///
/// The runtime is not a debug environment: the commands that launch or terminate
/// applications, or that the SDK gates on `am.debug!dev_function`, report the service's
/// "not available" error. The read-only and state-only commands stay implemented.
struct DebugFunctions {
    state: Arc<AmState>,
}

impl DebugFunctions {
    fn command(&self, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            // NotifyMessageToHomeMenuForDebug is stubbed with an error on hardware,
            // OpenMainApplication needs an IApplicationAccessor the runtime does not have,
            // and the launch/terminate commands would start or stop applications.
            0 | 1 | 30 | 31 | 300 => Reply::err(RESULT_NOT_AVAILABLE),
            // PerformSystemButtonPressing and InvalidateTransitionLayer have no system
            // applet to act on.
            10 | 20 => Reply::ok(),
            // GetAppletResourceUsageInfo: (two counters, rights environments, padding).
            40 => Reply::ok().bytes(&[0; APPLET_RESOURCE_USAGE_INFO_SIZE]),
            // The applet-id database and the CPU boost override are debug-mode only.
            50..=53 | 100 | 110 | 111 => Reply::err(RESULT_NOT_AVAILABLE),
            // CancelCpuBoostModeForApplet: the runtime never enters a boost mode.
            101 => Reply::ok(),
            // Alarm and friend-invitation reservations only gate what the applet system
            // pushes when an application launches; accepting them is harmless.
            120..=122 | 130..=132 => Reply::ok(),
            // RestrictPowerOperationForSecureLaunchModeForDebug is DisableSleepTillShutdown.
            140 => Reply::ok(),
            // NX stubs both commands: write 0 and return 0.
            1000 | 1001 => Reply::ok().u64(0),
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for DebugFunctions {
    fn name(&self) -> String {
        "am/IDebugFunctions".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        if req.cmd == 900 {
            return Reply::ok().copy_handle(cached_event(
                c,
                &self.state.applet_events,
                APPLET_GRC_EVENT,
            ));
        }
        if matches!(req.cmd, 30 | 31 | 50..=53 | 100 | 110 | 111 | 300) {
            c.process().warn_once(format!(
                "am/IDebugFunctions command {} requires the debug-mode function flag",
                req.cmd
            ));
        }
        match self.command(req) {
            Some(r) => r,
            None => unimplemented("am/IDebugFunctions", c, req),
        }
    }
}

/// `nn::am::service::IAppletCommonFunctions` (7.0.0+).
///
/// ThemeStorage lives in memory: a dump run should not leave guest state on the host. The
/// default content is 0xAA, as the applet system creates it.
struct AppletCommonFunctions {
    state: Arc<AmState>,
}

impl AppletCommonFunctions {
    fn command(&self, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            // PushToAppletBoundChannel takes an IStorage the IPC layer does not resolve,
            // so the push is accepted without contents; the channel stays empty.
            20 => Reply::ok(),
            21 => Reply::err(RESULT_NOT_AVAILABLE),
            40 => Reply::ok().u32(DISPLAY_WIDTH).u32(DISPLAY_HEIGHT),
            // SetDisplayMagnification crops the composed layer; the runtime has no layer.
            42 => Reply::ok(),
            50 => {
                self.state
                    .home_button_double_click
                    .store(input_bool(&req.data, 0), Ordering::Relaxed);
                Reply::ok()
            }
            51 => Reply::ok().u8(self.state.home_button_double_click.load(Ordering::Relaxed) as u8),
            52 => {
                Reply::ok().u8((self.state.home_button_blocked.load(Ordering::Relaxed) > 0) as u8)
            }
            // The applet is never in VR and never asks to sleep for temperature/battery.
            60..=62 => Reply::ok().u8(0),
            // SetCpuBoostRequestPriority: apm has no priority queue for the guest.
            70 => Reply::ok(),
            80..=82 => Reply::ok(),
            100 => Reply::ok(),
            160 => Reply::ok().object(Arc::new(NotificationReceiverService {
                state: self.state.clone(),
            })),
            161 => Reply::ok().object(Arc::new(NotificationSenderService {
                state: self.state.clone(),
            })),
            // IsSystemAppletHomeMenu and the GPU time-slice setters have no effect here.
            310 => Reply::ok().u8(0),
            320 | 321 => Reply::ok(),
            _ => return None,
        };
        Some(reply)
    }

    fn read_theme(&self, c: &Cpu, req: &Request) -> Reply {
        let bytes = match req.out_buf(0) {
            Some(buf) => {
                let slice = self.state.theme_slice(req.u64(0), buf.size as usize);
                c.process().memory.write(buf.addr, &slice);
                slice
            }
            None => Vec::new(),
        };
        Reply::ok().u64(bytes.len() as u64)
    }

    fn write_theme(&self, c: &Cpu, req: &Request) -> Reply {
        if let Some(buf) = req.in_buf(0)
            && let Some(bytes) = c.process().memory.read(buf.addr, buf.size)
        {
            self.state.write_theme(req.u64(0), &bytes);
        }
        Reply::ok()
    }
}

impl Service for AppletCommonFunctions {
    fn name(&self) -> String {
        "am/IAppletCommonFunctions".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                self.state.set_terminate_result(c, req.u32(0));
                return Reply::ok();
            }
            10 => return self.read_theme(c, req),
            11 => return self.write_theme(c, req),
            300 => return Reply::ok().u64(c.process().title_id),
            370 => {
                return Reply::ok().copy_handle(cached_event(
                    c,
                    &self.state.application_events,
                    APPLICATION_GPU_ERROR_EVENT,
                ));
            }
            _ => {}
        }
        match self.command(req) {
            Some(r) => r,
            None => unimplemented("am/IAppletCommonFunctions", c, req),
        }
    }
}

/// `nn::am::service::IReceiverService` (18.0.0+), returned by
/// `IAppletCommonFunctions` command 160.
struct NotificationReceiverService {
    state: Arc<AmState>,
}

impl Service for NotificationReceiverService {
    fn name(&self) -> String {
        "am/IReceiverService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(NotificationReceiver {
                state: self.state.clone(),
            })),
            _ => unimplemented("am/IReceiverService", c, req),
        }
    }
}

/// `nn::am::service::INotificationReceiver` (18.0.0+).
///
/// The runtime has no notification producer, and the message layout is not public, so a
/// sent message is only counted: the event says one is pending, and `Receive` /
/// `ReceiveWithTick` consume the count without forwarding a payload.
struct NotificationReceiver {
    state: Arc<AmState>,
}

impl NotificationReceiver {
    fn receive(&self, c: &Cpu) -> Reply {
        match self.state.receive_notification() {
            Some(remaining) => {
                let event = cached_event(c, &self.state.applet_events, APPLET_NOTIFICATION_EVENT);
                c.process().set_event(event, remaining > 0);
                Reply::ok()
            }
            None => Reply::err(RESULT_NOT_AVAILABLE),
        }
    }
}

impl Service for NotificationReceiver {
    fn name(&self) -> String {
        "am/INotificationReceiver".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // AddSource/RemoveSource register an applet as a notification producer; the
            // runtime has none, so the registration is accepted and forgotten.
            0 | 1 => Reply::ok(),
            2 => Reply::ok().copy_handle(cached_event(
                c,
                &self.state.applet_events,
                APPLET_NOTIFICATION_EVENT,
            )),
            3 | 4 => self.receive(c),
            _ => unimplemented("am/INotificationReceiver", c, req),
        }
    }
}

/// `nn::am::service::ISenderService` (18.0.0+), returned by
/// `IAppletCommonFunctions` command 161.
struct NotificationSenderService {
    state: Arc<AmState>,
}

impl Service for NotificationSenderService {
    fn name(&self) -> String {
        "am/ISenderService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // `Cmd0` takes the 0x18-byte sender parameters and returns the sender.
            0 => Reply::ok().object(Arc::new(NotificationSender {
                state: self.state.clone(),
            })),
            _ => unimplemented("am/ISenderService", c, req),
        }
    }
}

/// `nn::am::service::INotificationSender` (18.0.0+).
struct NotificationSender {
    state: Arc<AmState>,
}

impl Service for NotificationSender {
    fn name(&self) -> String {
        "am/INotificationSender".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let pending = self.state.send_notification();
                let event = cached_event(c, &self.state.applet_events, APPLET_NOTIFICATION_EVENT);
                c.process().set_event(event, pending > 0);
                Reply::ok()
            }
            1 => Reply::ok().u32(self.state.pending_notifications.load(Ordering::Relaxed)),
            _ => unimplemented("am/INotificationSender", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> Arc<AmState> {
        Arc::new(AmState {
            messages: Mutex::new(VecDeque::new()),
            message_event: 0,
            launchable_event: 1,
            suspended_tick_event: 2,
            user_popped: AtomicBool::new(false),
            user_channel: Mutex::new(VecDeque::new()),
            expected_master_volume: Mutex::new(ExpectedMasterVolume::default()),
            self_controller: SelfControllerState::default(),
            common: CommonState::default(),
            common_events: Mutex::new([None; 3]),
            application_events: Mutex::new([None; 5]),
            suspension: Mutex::new(Suspension::default()),
            request_exit_to_library_applet: AtomicBool::new(false),
            save_data_size: AtomicU64::new(SAVE_DATA_SIZE),
            journal_size: AtomicU64::new(SAVE_JOURNAL_SIZE),
            launch_required_version: AtomicU32::new(0),
            recording_size: AtomicU64::new(0),
            recording_state: AtomicU32::new(0),
            theme: Mutex::new(vec![THEME_STORAGE_DEFAULT; THEME_STORAGE_SIZE]),
            home_button_blocked: AtomicU32::new(0),
            home_button_double_click: AtomicBool::new(false),
            winding: Mutex::new(WindingState::default()),
            applet_events: Mutex::new([None; 2]),
            pending_notifications: AtomicU32::new(0),
            terminate_result: AtomicU32::new(0),
        })
    }

    fn request(cmd: u32, data: Vec<u8>) -> Request {
        Request {
            cmd,
            data,
            ..Default::default()
        }
    }

    fn expected(controller: &AudioController, cmd: u32) -> Vec<u8> {
        controller
            .command(&request(cmd, Vec::new()))
            .expect("expected master volume reply")
            .data
    }

    #[test]
    fn audio_controller_defaults_to_full_volume() {
        let controller = AudioController { state: state() };
        assert_eq!(expected(&controller, 1), 1.0f32.to_bits().to_le_bytes());
        assert_eq!(expected(&controller, 2), 1.0f32.to_bits().to_le_bytes());
    }

    #[test]
    fn audio_controller_stores_both_expected_volumes() {
        let controller = AudioController { state: state() };
        let mut data = 0.25f32.to_le_bytes().to_vec();
        data.extend_from_slice(&0.5f32.to_le_bytes());
        let set = controller
            .command(&request(0, data))
            .expect("set expected master volume");
        assert!(set.data.is_empty());

        assert_eq!(expected(&controller, 1), 0.25f32.to_bits().to_le_bytes());
        assert_eq!(expected(&controller, 2), 0.5f32.to_bits().to_le_bytes());
    }

    #[test]
    fn audio_controller_ignores_unknown_commands() {
        let controller = AudioController { state: state() };
        assert!(controller.command(&request(3, Vec::new())).is_none());
        assert!(controller.command(&request(4, Vec::new())).is_none());
    }

    fn empty_ok(reply: &Reply) {
        assert_eq!(reply.result, 0);
        assert!(
            reply.data.is_empty(),
            "unexpected reply data {:x?}",
            reply.data
        );
    }

    fn suspended_ticks(controller: &SelfController, now: u64) -> Vec<u8> {
        controller
            .command(&request(90, Vec::new()), now)
            .expect("expected suspended tick reply")
            .data
    }

    #[test]
    fn self_controller_lock_exit_records_the_lock_and_unlock_clears_it() {
        let state = state();
        let controller = SelfController {
            state: state.clone(),
        };
        assert!(!state.self_controller.exit_locked.load(Ordering::Relaxed));

        empty_ok(&controller.command(&request(1, Vec::new()), 0).unwrap());
        assert!(state.self_controller.exit_locked.load(Ordering::Relaxed));

        empty_ok(&controller.command(&request(2, Vec::new()), 0).unwrap());
        assert!(!state.self_controller.exit_locked.load(Ordering::Relaxed));
    }

    #[test]
    fn self_controller_stores_the_boolean_notifications() {
        let state = state();
        let controller = SelfController {
            state: state.clone(),
        };
        let sc = &state.self_controller;

        empty_ok(&controller.command(&request(11, vec![1]), 0).unwrap());
        assert!(
            sc.operation_mode_changed_notification
                .load(Ordering::Relaxed)
        );
        empty_ok(&controller.command(&request(12, vec![1]), 0).unwrap());
        assert!(
            sc.performance_mode_changed_notification
                .load(Ordering::Relaxed)
        );
        empty_ok(&controller.command(&request(14, vec![1]), 0).unwrap());
        assert!(sc.restart_message_enabled.load(Ordering::Relaxed));
        empty_ok(&controller.command(&request(16, vec![1]), 0).unwrap());
        assert!(sc.out_of_focus_suspending_enabled.load(Ordering::Relaxed));

        empty_ok(&controller.command(&request(11, vec![0]), 0).unwrap());
        assert!(
            !sc.operation_mode_changed_notification
                .load(Ordering::Relaxed)
        );
        empty_ok(&controller.command(&request(12, vec![0]), 0).unwrap());
        assert!(
            !sc.performance_mode_changed_notification
                .load(Ordering::Relaxed)
        );
        empty_ok(&controller.command(&request(14, vec![0]), 0).unwrap());
        assert!(!sc.restart_message_enabled.load(Ordering::Relaxed));
        empty_ok(&controller.command(&request(16, vec![0]), 0).unwrap());
        assert!(!sc.out_of_focus_suspending_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn self_controller_stores_the_three_focus_handling_flags() {
        let state = state();
        let controller = SelfController {
            state: state.clone(),
        };
        assert_eq!(
            *state.self_controller.focus_handling_mode.lock().unwrap(),
            [false; 3]
        );

        empty_ok(&controller.command(&request(13, vec![1, 0, 1]), 0).unwrap());
        assert_eq!(
            *state.self_controller.focus_handling_mode.lock().unwrap(),
            [true, false, true]
        );
    }

    #[test]
    fn self_controller_reports_the_accumulated_suspended_ticks() {
        let state = state();
        let controller = SelfController {
            state: state.clone(),
        };
        assert_eq!(suspended_ticks(&controller, 42), 0u64.to_le_bytes());

        state.suspend(1_000);
        assert_eq!(suspended_ticks(&controller, 4_000), 3_000u64.to_le_bytes());

        state.resume(4_000);
        assert_eq!(suspended_ticks(&controller, 9_000), 3_000u64.to_le_bytes());

        state.suspend(10_000);
        state.resume(10_500);
        assert_eq!(suspended_ticks(&controller, 12_000), 3_500u64.to_le_bytes());
    }

    #[test]
    fn self_controller_returns_the_applet_events_and_keeps_other_commands() {
        let controller = SelfController { state: state() };

        assert_eq!(
            controller
                .command(&request(9, Vec::new()), 0)
                .unwrap()
                .copy_handles,
            vec![1]
        );
        assert_eq!(
            controller
                .command(&request(91, Vec::new()), 0)
                .unwrap()
                .copy_handles,
            vec![2]
        );
        assert_eq!(
            controller
                .command(&request(44, Vec::new()), 0)
                .unwrap()
                .data,
            [1u64.to_le_bytes(), 2u64.to_le_bytes()].concat()
        );
        assert!(controller.command(&request(200, Vec::new()), 0).is_none());
    }

    #[test]
    fn common_state_getter_records_the_request_exit_to_library_applet_flag() {
        let state = state();
        let getter = CommonStateGetter {
            state: state.clone(),
        };
        assert!(!state.request_exit_to_library_applet.load(Ordering::Relaxed));

        empty_ok(&getter.enable_request_exit_to_library_applet());
        assert!(state.request_exit_to_library_applet.load(Ordering::Relaxed));
    }

    #[test]
    fn game_play_recording_keeps_the_transfer_memory_size_and_state() {
        let state = state();
        assert_eq!(state.recording(), (0, 0));

        empty_ok(&state.initialize_game_play_recording(0x6000000));
        empty_ok(&state.set_game_play_recording_state(1));
        assert_eq!(state.recording(), (0x6000000, 1));

        empty_ok(&state.set_game_play_recording_state(0));
        assert_eq!(state.recording(), (0x6000000, 0));
    }

    fn common_reply(getter: &CommonStateGetter, cmd: u32, data: Vec<u8>) -> Reply {
        match getter.command(&request(cmd, data)) {
            Some(CommonCommand::Reply(r)) => r,
            _ => panic!("command {cmd} should return a data reply"),
        }
    }

    #[test]
    fn common_state_getter_reports_the_applet_environment() {
        let getter = CommonStateGetter { state: state() };
        assert_eq!(common_reply(&getter, 5, Vec::new()).data, [OPERATION_MODE]);
        assert_eq!(
            common_reply(&getter, 6, Vec::new()).data,
            PERFORMANCE_MODE.to_le_bytes()
        );
        assert_eq!(common_reply(&getter, 7, Vec::new()).data, [0]);
        assert_eq!(common_reply(&getter, 8, Vec::new()).data, [0]);
        assert_eq!(common_reply(&getter, 9, Vec::new()).data, [1]);
        assert_eq!(
            common_reply(&getter, 14, Vec::new()).data,
            0u64.to_le_bytes()
        );
        assert_eq!(
            common_reply(&getter, 40, Vec::new()).data,
            [0u32.to_le_bytes(); 4].concat()
        );
        assert_eq!(
            common_reply(&getter, 60, Vec::new()).data,
            [1920u32.to_le_bytes(), 1080u32.to_le_bytes()].concat()
        );
        assert_eq!(
            common_reply(&getter, 62, Vec::new()).data,
            0u32.to_le_bytes()
        );
        assert_eq!(
            common_reply(&getter, 65, Vec::new()).data,
            0u64.to_le_bytes()
        );
        assert_eq!(
            common_reply(&getter, 68, Vec::new()).data,
            0u32.to_le_bytes()
        );
        assert_eq!(
            common_reply(&getter, 91, Vec::new()).data,
            PERFORMANCE_CONFIGURATION.to_le_bytes()
        );
        assert_eq!(
            common_reply(&getter, 200, Vec::new()).data,
            0u32.to_le_bytes()
        );
        let platform = common_reply(&getter, 300, Vec::new());
        assert_eq!(platform.data, [0]);
        assert_eq!(platform.data.len(), 1);
    }

    #[test]
    fn common_state_getter_keeps_the_vr_and_controller_state() {
        let state = state();
        let getter = CommonStateGetter {
            state: state.clone(),
        };
        let controller = SelfController {
            state: state.clone(),
        };
        assert_eq!(common_reply(&getter, 50, Vec::new()).data, [0]);
        empty_ok(&common_reply(&getter, 51, vec![1]));
        assert_eq!(common_reply(&getter, 50, Vec::new()).data, [1]);
        empty_ok(&common_reply(&getter, 54, Vec::new()));
        assert_eq!(common_reply(&getter, 50, Vec::new()).data, [0]);
        empty_ok(&common_reply(&getter, 53, Vec::new()));
        assert_eq!(common_reply(&getter, 50, Vec::new()).data, [1]);

        assert_eq!(common_reply(&getter, 55, Vec::new()).data, [0]);
        empty_ok(&controller.command(&request(17, vec![1]), 0).unwrap());
        assert_eq!(common_reply(&getter, 55, Vec::new()).data, [1]);
    }

    #[test]
    fn common_state_getter_event_commands_name_their_event() {
        let getter = CommonStateGetter { state: state() };
        for (cmd, which) in [
            (13, COMMON_SLEEP_LOCK_EVENT),
            (61, COMMON_RESOLUTION_EVENT),
            (63, COMMON_HDCP_EVENT),
        ] {
            match getter.command(&request(cmd, Vec::new())) {
                Some(CommonCommand::Event(actual)) => assert_eq!(actual, which, "command {cmd}"),
                _ => panic!("command {cmd} should return an event"),
            }
        }
        assert_eq!(common_reply(&getter, 0, Vec::new()).copy_handles, vec![0]);
    }

    #[test]
    fn common_state_getter_accepts_the_sleep_and_vr_requests() {
        let getter = CommonStateGetter { state: state() };
        for cmd in [
            3, 4, 11, 12, 20, 52, 64, 66, 67, 80, 90, 100, 400, 401, 500, 501,
        ] {
            let r = common_reply(&getter, cmd, Vec::new());
            empty_ok(&r);
        }
    }

    #[test]
    fn self_controller_defers_exit_while_the_exit_lock_is_held() {
        let controller = SelfController { state: state() };
        let sc = &controller.state.self_controller;

        empty_ok(&controller.command(&request(0, Vec::new()), 0).unwrap());
        assert!(sc.take_pending_exit());

        empty_ok(&controller.command(&request(1, Vec::new()), 0).unwrap());
        empty_ok(&controller.command(&request(0, Vec::new()), 0).unwrap());
        assert!(!sc.take_pending_exit());
        assert!(sc.exit_pending.load(Ordering::Relaxed));

        empty_ok(&controller.command(&request(2, Vec::new()), 0).unwrap());
        assert!(sc.take_pending_exit());
        assert!(!sc.exit_locked.load(Ordering::Relaxed));
    }

    #[test]
    fn self_controller_counts_fatal_sections() {
        let controller = SelfController { state: state() };
        let sc = &controller.state.self_controller;

        empty_ok(&controller.command(&request(3, Vec::new()), 0).unwrap());
        empty_ok(&controller.command(&request(3, Vec::new()), 0).unwrap());
        assert_eq!(sc.fatal_section.load(Ordering::Relaxed), 2);
        empty_ok(&controller.command(&request(4, Vec::new()), 0).unwrap());
        assert_eq!(sc.fatal_section.load(Ordering::Relaxed), 1);
        empty_ok(&controller.command(&request(4, Vec::new()), 0).unwrap());
        empty_ok(&controller.command(&request(4, Vec::new()), 0).unwrap());
        assert_eq!(sc.fatal_section.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn self_controller_reports_idle_time_auto_sleep_and_illuminance() {
        let controller = SelfController { state: state() };
        let sc = &controller.state.self_controller;

        assert_eq!(
            controller
                .command(&request(63, Vec::new()), 0)
                .unwrap()
                .data,
            0u32.to_le_bytes()
        );
        empty_ok(
            &controller
                .command(&request(62, 7u32.to_le_bytes().to_vec()), 0)
                .unwrap(),
        );
        assert_eq!(sc.idle_time_detection_extension.load(Ordering::Relaxed), 7);
        assert_eq!(
            controller
                .command(&request(63, Vec::new()), 0)
                .unwrap()
                .data,
            7u32.to_le_bytes()
        );

        assert_eq!(
            controller
                .command(&request(69, Vec::new()), 0)
                .unwrap()
                .data,
            [0]
        );
        empty_ok(&controller.command(&request(68, vec![1]), 0).unwrap());
        assert!(
            controller
                .state
                .self_controller
                .auto_sleep_disabled
                .load(Ordering::Relaxed)
        );
        assert_eq!(
            controller
                .command(&request(69, Vec::new()), 0)
                .unwrap()
                .data,
            [1]
        );

        let lux = 10_000.0f32.to_bits();
        assert_eq!(
            controller
                .command(&request(66, Vec::new()), 0)
                .unwrap()
                .data,
            lux.to_le_bytes()
        );
        assert_eq!(
            controller
                .command(&request(67, Vec::new()), 0)
                .unwrap()
                .data,
            [1]
        );
        let ex = controller
            .command(&request(71, Vec::new()), 0)
            .unwrap()
            .data;
        assert_eq!(ex.len(), 8);
        assert_eq!(ex[0], 1);
        assert_eq!(&ex[4..], &lux.to_le_bytes());
    }

    #[test]
    fn self_controller_setters_without_getters_are_accepted() {
        let controller = SelfController { state: state() };
        for cmd in [
            10, 15, 18, 19, 20, 41, 45, 46, 50, 51, 60, 61, 64, 65, 70, 72, 80, 100, 110, 120, 130,
        ] {
            empty_ok(&controller.command(&request(cmd, vec![0; 0x20]), 0).unwrap());
        }
    }

    #[test]
    fn lock_accessor_locks_once_and_reports_its_event() {
        let accessor = LockAccessor {
            locked: Mutex::new(false),
            event: 7,
        };
        assert!(!accessor.is_locked());
        assert!(accessor.try_lock());
        assert!(accessor.is_locked());
        // A second TryLock is refused until the accessor is unlocked.
        assert!(!accessor.try_lock());
        accessor.unlock();

        // TryLock with the handle flag clear takes the lock and returns no handle.
        let taken = accessor.command(&request(1, vec![0])).unwrap();
        assert_eq!(taken.result, 0);
        assert_eq!(taken.data, [1]);
        assert!(taken.copy_handles.is_empty());
        assert!(accessor.is_locked());

        // With the flag set the same handle comes back, even when the lock is refused.
        let refused = accessor.command(&request(1, vec![1])).unwrap();
        assert_eq!(refused.data, [0]);
        assert_eq!(refused.copy_handles, vec![7]);

        empty_ok(&accessor.command(&request(2, Vec::new())).unwrap());
        assert!(!accessor.is_locked());
        assert!(!accessor.unlock());
        assert!(accessor.try_lock());
        assert_eq!(accessor.command(&request(4, Vec::new())).unwrap().data, [1]);
        let event = accessor.command(&request(3, Vec::new())).unwrap();
        assert_eq!(event.copy_handles, vec![7]);
        assert!(accessor.command(&request(5, Vec::new())).is_none());
    }

    #[test]
    fn gpu_error_handler_has_no_pending_error() {
        let handler = GpuErrorHandler { event: 9 };
        assert_eq!(
            handler.command(&request(100, Vec::new())).unwrap().data,
            0u64.to_le_bytes()
        );
        assert_eq!(
            handler.command(&request(101, Vec::new())).unwrap().data,
            0u64.to_le_bytes()
        );
        assert_eq!(
            handler
                .command(&request(102, Vec::new()))
                .unwrap()
                .copy_handles,
            vec![9]
        );
        empty_ok(&handler.command(&request(103, Vec::new())).unwrap());
        assert!(handler.command(&request(104, Vec::new())).is_none());
    }

    fn app_reply(app: &ApplicationFunctions, cmd: u32, data: Vec<u8>) -> Reply {
        match app.command(&request(cmd, data)) {
            Some(ApplicationCommand::Reply(r) | ApplicationCommand::Exit(r)) => r,
            _ => panic!("command {cmd} should return a data reply"),
        }
    }

    #[test]
    fn application_functions_pop_launch_parameter() {
        let state = state();
        let app = ApplicationFunctions {
            state: state.clone(),
        };

        assert_eq!(
            app_reply(&app, 1, LAUNCH_PRESELECTED_USER.to_le_bytes().to_vec()).result,
            0
        );
        assert_eq!(
            app_reply(&app, 1, LAUNCH_PRESELECTED_USER.to_le_bytes().to_vec()).result,
            RESULT_NOT_AVAILABLE
        );

        state.user_channel.lock().unwrap().push_back(vec![1, 2, 3]);
        let stored = app_reply(&app, 1, LAUNCH_USER_CHANNEL.to_le_bytes().to_vec());
        assert_eq!(stored.result, 0);
        assert_eq!(stored.objects.len(), 1);
        assert_eq!(stored.objects[0].name(), "am/IStorage");
        assert_eq!(
            app_reply(&app, 1, LAUNCH_USER_CHANNEL.to_le_bytes().to_vec()).result,
            RESULT_NOT_AVAILABLE
        );
    }

    #[test]
    fn application_functions_report_save_sizes_and_launch_storage() {
        let app = ApplicationFunctions { state: state() };

        assert_eq!(
            app_reply(&app, 20, vec![0; 0x10]).data,
            SAVE_DATA_SIZE.to_le_bytes()
        );
        assert_eq!(app_reply(&app, 24, Vec::new()).data, [1, 1]);

        let mut extend = vec![0u8; 0x28];
        extend[0] = 1;
        extend[0x18..0x20].copy_from_slice(&0x40_0000u64.to_le_bytes());
        extend[0x20..0x28].copy_from_slice(&0x20_0000u64.to_le_bytes());
        assert_eq!(app_reply(&app, 25, extend).data, 0u64.to_le_bytes());

        let expected = [0x40_0000u64.to_le_bytes(), 0x20_0000u64.to_le_bytes()].concat();
        assert_eq!(app_reply(&app, 26, vec![0; 0x18]).data, expected);
        assert_eq!(app_reply(&app, 28, Vec::new()).data, expected);
    }

    #[test]
    fn application_functions_report_the_other_public_values() {
        let app = ApplicationFunctions { state: state() };
        assert_eq!(app_reply(&app, 40, Vec::new()).data, [1]);
        let pseudo = app_reply(&app, 50, Vec::new());
        assert_eq!(pseudo.data, DEFAULT_PSEUDO_DEVICE_ID);
        assert_eq!(app_reply(&app, 110, vec![0; 8]).data, 0u32.to_le_bytes());
        assert_eq!(app_reply(&app, 111, vec![0; 0x18]).data, 0u32.to_le_bytes());
        assert_eq!(app_reply(&app, 123, Vec::new()).data, 0u32.to_le_bytes());
        assert_eq!(app_reply(&app, 200, Vec::new()).data, 0u32.to_le_bytes());
        assert_eq!(app_reply(&app, 330, Vec::new()).data, [0]);

        assert_eq!(app_reply(&app, 180, Vec::new()).data, 0u32.to_le_bytes());
        assert!(
            app_reply(&app, 181, 5u32.to_le_bytes().to_vec())
                .data
                .is_empty()
        );
        assert_eq!(app_reply(&app, 180, Vec::new()).data, 5u32.to_le_bytes());

        for cmd in [
            12, 30, 31, 32, 33, 60, 68, 90, 100, 101, 102, 120, 121, 122, 124, 131, 170, 220, 500,
            1001,
        ] {
            empty_ok(&app_reply(&app, cmd, vec![0; 0x20]));
        }
    }

    #[test]
    fn application_functions_reject_empty_channels() {
        let app = ApplicationFunctions { state: state() };
        for cmd in [141, 151] {
            assert_eq!(
                app_reply(&app, cmd, Vec::new()).result,
                RESULT_NOT_AVAILABLE
            );
        }
    }

    #[test]
    fn application_functions_event_commands_name_their_event() {
        let app = ApplicationFunctions { state: state() };
        for (cmd, which) in [
            (130, APPLICATION_GPU_ERROR_EVENT),
            (140, APPLICATION_FRIEND_EVENT),
            (150, APPLICATION_NOTIFICATION_EVENT),
            (160, APPLICATION_HEALTH_WARNING_EVENT),
            (210, APPLICATION_UNKNOWN_EVENT),
        ] {
            match app.command(&request(cmd, Vec::new())) {
                Some(ApplicationCommand::Event(actual)) => {
                    assert_eq!(actual, which, "command {cmd}")
                }
                _ => panic!("command {cmd} should return an event"),
            }
        }
    }

    #[test]
    fn application_functions_exit_requests_end_the_application() {
        let app = ApplicationFunctions { state: state() };
        for cmd in [70, 71, 72, 80, 310] {
            match app.command(&request(cmd, Vec::new())) {
                Some(ApplicationCommand::Exit(r)) => empty_ok(&r),
                _ => panic!("command {cmd} should exit"),
            }
        }
        assert!(app.command(&request(27, Vec::new())).is_none());
        assert!(app.command(&request(112, Vec::new())).is_none());
    }

    #[test]
    fn application_proxy_returns_the_documented_managers() {
        let proxy = ApplicationProxy { state: state() };
        for (cmd, name) in [
            (0, "am/ICommonStateGetter"),
            (1, "am/ISelfController"),
            (2, "am/IWindowController"),
            (3, "am/IAudioController"),
            (4, "am/IDisplayController"),
            (10, "am/IProcessWindingController"),
            (11, "am/ILibraryAppletCreator"),
            (20, "am/IApplicationFunctions"),
            (21, "am/IAppletCommonFunctions"),
            (1000, "am/IDebugFunctions"),
        ] {
            let object = proxy
                .object(cmd)
                .unwrap_or_else(|| panic!("command {cmd} has no manager"));
            assert_eq!(object.name(), name, "command {cmd}");
        }
        assert!(proxy.object(5).is_none());
    }

    fn display_reply(display: &DisplayController, cmd: u32) -> Reply {
        display
            .command(&request(cmd, vec![0; 0x20]))
            .unwrap_or_else(|| panic!("command {cmd} should reply"))
    }

    #[test]
    fn display_controller_reports_no_capture_image() {
        let display = DisplayController;
        for cmd in [0, 2, 3] {
            let reply = display_reply(&display, cmd);
            assert_eq!(reply.result, RESULT_NOT_AVAILABLE, "command {cmd}");
            assert!(reply.data.is_empty(), "command {cmd}");
        }
        // Get*CaptureImageEx: one bool byte saying the buffer was not filled.
        for cmd in [5, 6, 7] {
            let reply = display_reply(&display, cmd);
            assert_eq!(reply.result, 0, "command {cmd}");
            assert_eq!(reply.data, [0], "command {cmd}");
        }
    }

    #[test]
    fn display_controller_shared_buffers_report_no_capture() {
        let display = DisplayController;
        for cmd in [22, 24, 26] {
            let reply = display_reply(&display, cmd);
            assert_eq!(reply.result, 0, "command {cmd}");
            assert_eq!(reply.data.len(), 8, "command {cmd}");
            assert_eq!(reply.data, [0u8; 8], "command {cmd}");
        }
        for cmd in [10, 11, 12, 13, 14, 15, 16, 17, 18, 23, 25, 27] {
            assert_eq!(
                display_reply(&display, cmd).result,
                RESULT_NOT_AVAILABLE,
                "command {cmd}"
            );
        }
    }

    #[test]
    fn display_controller_accepts_the_no_output_commands() {
        let display = DisplayController;
        for cmd in [1, 4, 8, 9, 20, 21, 28] {
            empty_ok(&display_reply(&display, cmd));
        }
        assert!(display.command(&request(19, Vec::new())).is_none());
    }

    fn winding_reply(winding: &ProcessWindingController, cmd: u32) -> Reply {
        winding
            .command(&request(cmd, Vec::new()))
            .unwrap_or_else(|| panic!("command {cmd} should reply"))
    }

    #[test]
    fn process_winding_controller_reports_the_launch_reason() {
        let winding = ProcessWindingController { state: state() };
        // AppletProcessLaunchReason: caller flag clear plus three zero bytes.
        assert_eq!(winding_reply(&winding, 0).data, [0, 0, 0, 0]);
        assert_eq!(
            winding_reply(&winding, 11).result,
            RESULT_NOT_AVAILABLE,
            "OpenCallingLibraryApplet without a calling applet"
        );
    }

    #[test]
    fn process_winding_controller_pushes_and_pops_contexts() {
        let winding = ProcessWindingController { state: state() };
        assert_eq!(
            winding_reply(&winding, 22).result,
            RESULT_NOT_AVAILABLE,
            "PopContext on an empty stack"
        );

        empty_ok(&winding_reply(&winding, 21));
        empty_ok(&winding_reply(&winding, 21));

        for expected in 0..2usize {
            let pop = winding_reply(&winding, 22);
            assert_eq!(pop.result, 0);
            assert_eq!(pop.objects.len(), 1);
            assert_eq!(pop.objects[0].name(), "am/IStorage");
            assert_eq!(
                winding.state.winding.lock().unwrap().contexts.len(),
                1 - expected
            );
        }
        assert_eq!(winding_reply(&winding, 22).result, RESULT_NOT_AVAILABLE);
    }

    #[test]
    fn process_winding_controller_reservation_state_machine() {
        let winding = ProcessWindingController { state: state() };

        // WindAndDoReserved without a reservation is unsupported.
        assert_eq!(
            winding_reply(&winding, 30).result,
            RESULT_NOT_AVAILABLE,
            "WindAndDoReserved without a reservation"
        );

        // Cancel after reserving leaves nothing to wind.
        empty_ok(&winding_reply(&winding, 40));
        empty_ok(&winding_reply(&winding, 23));
        assert_eq!(winding_reply(&winding, 30).result, RESULT_NOT_AVAILABLE);

        // The other reserve command arms the same transition, which wind consumes once.
        empty_ok(&winding_reply(&winding, 41));
        empty_ok(&winding_reply(&winding, 30));
        assert_eq!(winding_reply(&winding, 30).result, RESULT_NOT_AVAILABLE);

        // Cancel with nothing pending is harmless.
        empty_ok(&winding_reply(&winding, 23));
    }

    fn applet_common_reply(app: &AppletCommonFunctions, cmd: u32, data: Vec<u8>) -> Reply {
        app.command(&request(cmd, data))
            .unwrap_or_else(|| panic!("command {cmd} should reply"))
    }

    #[test]
    fn applet_common_functions_report_the_applet_environment() {
        let app = AppletCommonFunctions { state: state() };
        assert_eq!(
            applet_common_reply(&app, 40, Vec::new()).data,
            [DISPLAY_WIDTH.to_le_bytes(), DISPLAY_HEIGHT.to_le_bytes()].concat()
        );
        assert_eq!(
            applet_common_reply(&app, 40, Vec::new()).data.len(),
            8,
            "logical resolution is two s32"
        );
        empty_ok(&applet_common_reply(&app, 42, vec![0; 0x10]));
        empty_ok(&applet_common_reply(&app, 70, 1u32.to_le_bytes().to_vec()));
        for cmd in [80, 81, 82, 100, 320, 321] {
            empty_ok(&applet_common_reply(&app, cmd, vec![0; 8]));
        }
        for cmd in [60, 61, 62, 310] {
            let reply = applet_common_reply(&app, cmd, Vec::new());
            assert_eq!(reply.data, [0], "command {cmd}");
        }
        assert_eq!(
            applet_common_reply(&app, 21, Vec::new()).result,
            RESULT_NOT_AVAILABLE,
            "TryPopFromAppletBoundChannel with an empty channel"
        );
        assert!(app.command(&request(90, Vec::new())).is_none());
    }

    #[test]
    fn applet_common_functions_tracks_the_home_button_flags() {
        let state = state();
        let app = AppletCommonFunctions {
            state: state.clone(),
        };
        let functions = ApplicationFunctions {
            state: state.clone(),
        };

        assert_eq!(applet_common_reply(&app, 52, Vec::new()).data, [0]);
        empty_ok(&app_reply(&functions, 30, Vec::new()));
        assert_eq!(applet_common_reply(&app, 52, Vec::new()).data, [1]);
        empty_ok(&app_reply(&functions, 32, Vec::new()));
        empty_ok(&app_reply(&functions, 31, Vec::new()));
        assert_eq!(
            applet_common_reply(&app, 52, Vec::new()).data,
            [1],
            "one blocking request is still open"
        );
        empty_ok(&app_reply(&functions, 33, Vec::new()));
        assert_eq!(applet_common_reply(&app, 52, Vec::new()).data, [0]);
        empty_ok(&app_reply(&functions, 33, Vec::new()));
        assert_eq!(applet_common_reply(&app, 52, Vec::new()).data, [0]);

        assert_eq!(applet_common_reply(&app, 51, Vec::new()).data, [0]);
        empty_ok(&applet_common_reply(&app, 50, vec![1]));
        assert_eq!(applet_common_reply(&app, 51, Vec::new()).data, [1]);
        empty_ok(&applet_common_reply(&app, 50, vec![0]));
        assert_eq!(applet_common_reply(&app, 51, Vec::new()).data, [0]);
    }

    #[test]
    fn applet_common_functions_hands_out_the_notification_services() {
        let state = state();
        let app = AppletCommonFunctions {
            state: state.clone(),
        };
        let receiver = applet_common_reply(&app, 160, Vec::new());
        assert_eq!(receiver.objects.len(), 1);
        assert_eq!(receiver.objects[0].name(), "am/IReceiverService");
        let sender = applet_common_reply(&app, 161, Vec::new());
        assert_eq!(sender.objects.len(), 1);
        assert_eq!(sender.objects[0].name(), "am/ISenderService");

        assert_eq!(
            NotificationReceiverService {
                state: state.clone()
            }
            .name(),
            "am/IReceiverService"
        );
        assert_eq!(
            NotificationSenderService {
                state: state.clone()
            }
            .name(),
            "am/ISenderService"
        );
        assert_eq!(
            NotificationReceiver {
                state: state.clone()
            }
            .name(),
            "am/INotificationReceiver"
        );
        assert_eq!(
            NotificationSender {
                state: state.clone()
            }
            .name(),
            "am/INotificationSender"
        );
    }

    #[test]
    fn notification_queue_counts_sends_and_drains_receives() {
        let state = state();
        assert_eq!(state.receive_notification(), None);
        assert_eq!(state.send_notification(), 1);
        assert_eq!(state.send_notification(), 2);
        assert_eq!(state.receive_notification(), Some(1));
        assert_eq!(state.receive_notification(), Some(0));
        assert_eq!(state.receive_notification(), None);
        assert_eq!(state.pending_notifications.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn theme_storage_defaults_to_the_documented_pattern() {
        let state = state();
        let theme = state.theme_slice(0, THEME_STORAGE_SIZE + 1);
        assert_eq!(
            theme.len(),
            THEME_STORAGE_SIZE,
            "the read is clamped to 0x400"
        );
        assert!(theme.iter().all(|&b| b == THEME_STORAGE_DEFAULT));

        assert_eq!(state.theme_slice(0x3FF, 0x10).len(), 1);
        assert!(state.theme_slice(0x400, 0x10).is_empty());
        assert!(state.theme_slice(u64::MAX, 0x10).is_empty());
    }

    #[test]
    fn theme_storage_write_clamps_to_the_state_buffer() {
        let state = state();
        assert_eq!(state.write_theme(0x3FE, &[1, 2, 3, 4]), 2);
        let tail = state.theme_slice(0x3F0, 0x10);
        assert_eq!(&tail[0xE..], &[1, 2]);
        assert_eq!(state.write_theme(0x400, &[1]), 0);
        assert_eq!(state.write_theme(u64::MAX, &[1]), 0);

        assert_eq!(
            state.write_theme(0, &[9; THEME_STORAGE_SIZE]),
            THEME_STORAGE_SIZE
        );
        assert!(
            state
                .theme_slice(0, THEME_STORAGE_SIZE)
                .iter()
                .all(|&b| b == 9)
        );
    }

    fn debug_reply(debug: &DebugFunctions, cmd: u32) -> Reply {
        debug
            .command(&request(cmd, vec![0; 0x20]))
            .unwrap_or_else(|| panic!("command {cmd} should reply"))
    }

    #[test]
    fn debug_functions_deny_the_debug_only_commands() {
        let debug = DebugFunctions { state: state() };
        for cmd in [0, 1, 30, 31, 50, 51, 52, 53, 100, 110, 111, 300] {
            let reply = debug_reply(&debug, cmd);
            assert_eq!(reply.result, RESULT_NOT_AVAILABLE, "command {cmd}");
            assert!(reply.data.is_empty(), "command {cmd}");
        }
        assert!(debug.command(&request(410, Vec::new())).is_none());
    }

    #[test]
    fn debug_functions_keep_the_harmless_commands() {
        let debug = DebugFunctions { state: state() };
        for cmd in [10, 20, 101, 120, 121, 122, 130, 131, 132, 140] {
            empty_ok(&debug_reply(&debug, cmd));
        }

        let usage = debug_reply(&debug, 40);
        assert_eq!(usage.data.len(), APPLET_RESOURCE_USAGE_INFO_SIZE);
        assert!(usage.data.iter().all(|&b| b == 0));

        for cmd in [1000, 1001] {
            assert_eq!(
                debug_reply(&debug, cmd).data,
                0u64.to_le_bytes(),
                "command {cmd}"
            );
        }
    }
}
