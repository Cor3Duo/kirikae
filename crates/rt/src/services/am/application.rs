//! Applet-side managers: the objects reached through `am/appletOE` and the
//! `IApplicationProxy` service.

use super::common::{
    AudioController, CommonStateGetter, DisplayController, ProcessWindingController,
    SelfController, WindowController,
};
use super::library::{LibraryAppletCreator, Storage};
use super::*;

/// Managers every applet proxy returns with the same numbers.
pub(super) fn applet_manager(state: &Arc<AmState>, cmd: u32) -> Option<Arc<dyn Service>> {
    let obj: Arc<dyn Service> = match cmd {
        0 => Arc::new(CommonStateGetter::new(state.clone())),
        1 => Arc::new(SelfController::new(state.clone())),
        2 => Arc::new(WindowController),
        3 => Arc::new(AudioController::new(state.clone())),
        4 => Arc::new(DisplayController),
        10 => Arc::new(ProcessWindingController::new(state.clone())),
        11 => Arc::new(LibraryAppletCreator),
        1000 => Arc::new(DebugFunctions::new(state.clone())),
        _ => return None,
    };
    Some(obj)
}

pub(super) struct ApplicationProxy {
    state: Arc<AmState>,
}

impl ApplicationProxy {
    pub(super) fn new(state: Arc<AmState>) -> Self {
        ApplicationProxy { state }
    }

    /// The manager object behind an `am/IApplicationProxy` command. Keeping the
    /// table separate from `call` lets the tests check each command's object
    /// without a guest CPU.
    fn object(&self, cmd: u32) -> Option<Arc<dyn Service>> {
        let obj: Arc<dyn Service> = match cmd {
            20 => Arc::new(ApplicationFunctions::new(self.state.clone())),
            // `IApplicationProxy` has no GetAppletCommonFunctions on hardware: a regular
            // application reaches it through the system/library/overlay proxies
            // (commands 23/21/21). The single `am` client is still served from this
            // table, and command 21 is the number the library and overlay proxies use.
            21 => Arc::new(AppletCommonFunctions::new(self.state.clone())),
            _ => return applet_manager(&self.state, cmd),
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

struct ApplicationFunctions {
    state: Arc<AmState>,
}

impl ApplicationFunctions {
    fn new(state: Arc<AmState>) -> Self {
        ApplicationFunctions { state }
    }
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
pub(super) struct DebugFunctions {
    state: Arc<AmState>,
}

impl DebugFunctions {
    pub(super) fn new(state: Arc<AmState>) -> Self {
        DebugFunctions { state }
    }

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
            return Reply::ok().copy_handle(cached_event(c, &self.state.lazy_events, EVENT_GRC));
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
pub(super) struct AppletCommonFunctions {
    state: Arc<AmState>,
}

impl AppletCommonFunctions {
    pub(super) fn new(state: Arc<AmState>) -> Self {
        AppletCommonFunctions { state }
    }

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
                let event = cached_event(c, &self.state.lazy_events, EVENT_NOTIFICATION);
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
                &self.state.lazy_events,
                EVENT_NOTIFICATION,
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
                let event = cached_event(c, &self.state.lazy_events, EVENT_NOTIFICATION);
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
    use super::super::{empty_ok, request, state};
    use super::*;

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
