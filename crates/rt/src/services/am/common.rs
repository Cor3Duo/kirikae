//! Managers shared by every applet proxy: common state, self control, display,
//! winding, audio and the lock/error accessors.

use super::library::Storage;
use super::*;

fn input_f32(data: &[u8], off: usize) -> Option<f32> {
    let bytes: [u8; 4] = data.get(off..off + 4)?.try_into().ok()?;
    Some(f32::from_le_bytes(bytes))
}

pub(super) struct AudioController {
    state: Arc<AmState>,
}

impl AudioController {
    pub(super) fn new(state: Arc<AmState>) -> Self {
        AudioController { state }
    }

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
pub(super) struct DisplayController;

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
pub(super) struct ProcessWindingController {
    state: Arc<AmState>,
}

impl ProcessWindingController {
    pub(super) fn new(state: Arc<AmState>) -> Self {
        ProcessWindingController { state }
    }

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

pub(super) struct CommonStateGetter {
    state: Arc<AmState>,
}

enum CommonCommand {
    Reply(Reply),
    Event(usize),
}

impl CommonStateGetter {
    pub(super) fn new(state: Arc<AmState>) -> Self {
        CommonStateGetter { state }
    }

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
            // PushToGeneralChannel takes an IStorage. The IPC layer does not resolve
            // input objects, so the push queues an empty entry: the home menu's
            // `PopFromGeneralChannel` still gets a storage and its event fires.
            20 => Data(self.push_to_general_channel()),
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
            64 | 66 | 67 | 80 | 90 | 100 | 400 | 401 => Data(Reply::ok()),
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
            // DisableSleepTillShutdown / SuppressDisablingSleepTemporarily: the
            // home menu reads the same flag through `IHomeMenuFunctions` command 40.
            500 => {
                self.state.system.set_sleep_enabled(false);
                Data(Reply::ok())
            }
            501 => {
                self.state.system.set_sleep_enabled(true);
                Data(Reply::ok())
            }
            502 => Data(Reply::ok().u8(self.state.system.sleep_enabled() as u8)),
            // The runtime never suppresses a sleep disable request.
            503 => Data(Reply::ok().u8(0)),
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
            return Reply::ok()
                .object(Arc::new(LockAccessor::new(c.process().create_event(false))));
        }
        if req.cmd == 110 {
            return Reply::ok().object(Arc::new(GpuErrorHandler {
                event: c.process().create_event(false),
            }));
        }
        if req.cmd == 20 {
            let reply = self.push_to_general_channel();
            let event = cached_event(c, &self.state.lazy_events, EVENT_GENERAL_CHANNEL);
            c.process().set_event(event, true);
            return reply;
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
    fn push_to_general_channel(&self) -> Reply {
        self.state.system.push_general_channel();
        Reply::ok()
    }

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
pub(super) struct LockAccessor {
    locked: Mutex<bool>,
    event: Mutex<Option<u32>>,
}

impl LockAccessor {
    fn new(event: u32) -> Self {
        LockAccessor {
            locked: Mutex::new(false),
            event: Mutex::new(Some(event)),
        }
    }

    /// Accessor whose event is created the first time the guest uses it. The
    /// home menu hands these out from a command table that has no guest CPU.
    pub(super) fn deferred() -> Self {
        LockAccessor {
            locked: Mutex::new(false),
            event: Mutex::new(None),
        }
    }

    fn event(&self, c: &Cpu) -> u32 {
        *self
            .event
            .lock()
            .unwrap()
            .get_or_insert_with(|| c.process().create_event(false))
    }

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

    fn command(&self, req: &Request, event: u32) -> Option<Reply> {
        let reply = match req.cmd {
            // `TryLock`: the input flag asks for the event handle (official software passes
            // false and closes the handle without waiting); the output flag says whether the
            // lock was taken.
            1 => {
                let reply = Reply::ok().u8(self.try_lock() as u8);
                if input_bool(&req.data, 0) {
                    reply.copy_handle(event)
                } else {
                    reply
                }
            }
            2 => {
                self.unlock();
                Reply::ok()
            }
            3 => Reply::ok().copy_handle(event),
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
        let event = self.event(c);
        let reply = self.command(req, event);
        if req.cmd == 2 {
            c.process().set_event(event, false);
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

pub(super) struct SelfController {
    state: Arc<AmState>,
}

impl SelfController {
    pub(super) fn new(state: Arc<AmState>) -> Self {
        SelfController { state }
    }

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

pub(super) struct WindowController;

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

#[cfg(test)]
mod tests {
    use super::super::{empty_ok, request, state};
    use super::*;

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
    fn common_state_getter_controls_the_sleep_enable_flag() {
        let state = state();
        let getter = CommonStateGetter {
            state: state.clone(),
        };
        assert_eq!(common_reply(&getter, 502, Vec::new()).data, [1]);
        assert_eq!(common_reply(&getter, 503, Vec::new()).data, [0]);

        empty_ok(&common_reply(&getter, 500, Vec::new()));
        assert!(!state.system.sleep_enabled());
        assert_eq!(common_reply(&getter, 502, Vec::new()).data, [0]);

        empty_ok(&common_reply(&getter, 501, Vec::new()));
        assert!(state.system.sleep_enabled());
        assert_eq!(common_reply(&getter, 502, Vec::new()).data, [1]);
    }

    #[test]
    fn common_state_getter_queues_the_general_channel_push() {
        let state = state();
        let getter = CommonStateGetter {
            state: state.clone(),
        };
        assert!(!state.system.general_channel_pending());
        empty_ok(&common_reply(&getter, 20, Vec::new()));
        assert!(state.system.general_channel_pending());
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
        let accessor = LockAccessor::new(7);
        assert!(!accessor.is_locked());
        assert!(accessor.try_lock());
        assert!(accessor.is_locked());
        // A second TryLock is refused until the accessor is unlocked.
        assert!(!accessor.try_lock());
        accessor.unlock();

        // TryLock with the handle flag clear takes the lock and returns no handle.
        let taken = accessor.command(&request(1, vec![0]), 7).unwrap();
        assert_eq!(taken.result, 0);
        assert_eq!(taken.data, [1]);
        assert!(taken.copy_handles.is_empty());
        assert!(accessor.is_locked());

        // With the flag set the same handle comes back, even when the lock is refused.
        let refused = accessor.command(&request(1, vec![1]), 7).unwrap();
        assert_eq!(refused.data, [0]);
        assert_eq!(refused.copy_handles, vec![7]);

        empty_ok(&accessor.command(&request(2, Vec::new()), 7).unwrap());
        assert!(!accessor.is_locked());
        assert!(!accessor.unlock());
        assert!(accessor.try_lock());
        assert_eq!(
            accessor.command(&request(4, Vec::new()), 7).unwrap().data,
            [1]
        );
        let event = accessor.command(&request(3, Vec::new()), 7).unwrap();
        assert_eq!(event.copy_handles, vec![7]);
        assert!(accessor.command(&request(5, Vec::new()), 7).is_none());
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
}
