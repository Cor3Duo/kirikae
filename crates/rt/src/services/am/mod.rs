//! `nn::am::service` applet management.
//!
//! `appletOE` serves regular applications and `appletAE` serves the system applets
//! (`qlaunch`, `overlay`, `starter`). Each session owns its [`AmState`]: manager
//! objects hold guest event handles, which cannot cross processes, and the runtime
//! runs one applet per process.

mod application;
mod common;
mod library;
mod system;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::apm::PERFORMANCE_MODE;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

use application::ApplicationProxy;
use system::SystemState;

const RESULT_NO_MESSAGES: u32 = 0x680;
/// `nn::am::ResultNotAvailable` (description 0x480). Used for the queries whose
/// backing state only exists once the applet system has done the corresponding
/// hardware action.
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
/// Events created on first request. The indexes are shared by the managers so a
/// command called twice returns the same handle.
const EVENT_NOTIFICATION: usize = 0;
const EVENT_GRC: usize = 1;
const EVENT_GENERAL_CHANNEL: usize = 2;
const EVENT_HDCP_FAILED: usize = 3;
const EVENT_SLOTS: usize = 4;
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
    /// Events created on first request: notification, GRC, the home-menu general
    /// channel and HDCP.
    lazy_events: Mutex<[Option<u32>; EVENT_SLOTS]>,
    /// Notifications accepted by `INotificationSender` and not received yet.
    pending_notifications: AtomicU32,
    terminate_result: AtomicU32,
    /// State owned by the system-applet managers (`IHomeMenuFunctions` and
    /// `IGlobalStateController`).
    system: SystemState,
}

fn cached_event<const N: usize>(c: &Cpu, slots: &Mutex<[Option<u32>; N]>, which: usize) -> u32 {
    *slots.lock().unwrap()[which].get_or_insert_with(|| c.process().create_event(false))
}

impl AmState {
    fn with_events(
        message_event: u32,
        launchable_event: u32,
        suspended_tick_event: u32,
    ) -> AmState {
        AmState {
            messages: Mutex::new(VecDeque::from([MESSAGE_FOCUS_STATE_CHANGED])),
            message_event,
            launchable_event,
            suspended_tick_event,
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
            lazy_events: Mutex::new([None; EVENT_SLOTS]),
            pending_notifications: AtomicU32::new(0),
            terminate_result: AtomicU32::new(0),
            system: SystemState::default(),
        }
    }

    /// State for a fresh applet session. The two autoclear=false events and the
    /// autoclear=true suspended-tick event are created in the order the managers
    /// hand them out.
    fn new(c: &Cpu) -> Arc<AmState> {
        Arc::new(AmState::with_events(
            c.process().create_event(true),
            c.process().create_event(true),
            c.process().create_event(false),
        ))
    }

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

/// `nn::am::service::IApplicationProxyService`, the `appletOE` service used by
/// regular applications.
pub struct AppletOe;

impl Service for AppletOe {
    fn name(&self) -> String {
        "appletOE".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(ApplicationProxy::new(AmState::new(c)))),
            _ => unimplemented("appletOE", c, req),
        }
    }
}

/// `nn::am::service::IAllSystemAppletProxiesService`, the `appletAE` service used
/// by the system titles.
pub struct AppletAe;

impl Service for AppletAe {
    fn name(&self) -> String {
        "appletAE".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match system::open(req.cmd, AmState::new(c)) {
            Some(reply) => reply,
            None => unimplemented("appletAE", c, req),
        }
    }
}

#[cfg(test)]
fn state() -> Arc<AmState> {
    Arc::new(AmState::with_events(0, 1, 2))
}

#[cfg(test)]
fn request(cmd: u32, data: Vec<u8>) -> Request {
    Request {
        cmd,
        data,
        ..Default::default()
    }
}

#[cfg(test)]
fn empty_ok(reply: &Reply) {
    assert_eq!(reply.result, 0);
    assert!(
        reply.data.is_empty(),
        "unexpected reply data {:x?}",
        reply.data
    );
}
