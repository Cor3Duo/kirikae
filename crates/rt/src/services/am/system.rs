//! System-applet managers reached through `am/appletAE`: the system, library and
//! overlay proxies, the home menu, the global state controller and the application
//! creator/accessor pair.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::acc::USER_ID;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

use super::application::{AppletCommonFunctions, ApplicationProxy, DebugFunctions, applet_manager};
use super::common::LockAccessor;
use super::library::Storage;
use super::{
    AmState, EVENT_GENERAL_CHANNEL, EVENT_HDCP_FAILED, RESULT_NOT_AVAILABLE, cached_event,
    input_bool,
};

const APPLET_ID_APPLICATION: u32 = 0x01;
const APPLET_ID_SYSTEM_APPLICATION: u32 = 0x04;
/// `nn::am::service::CradleDeviceInfo` is 8 bytes before 13.0.0 and 0xC from
/// 13.0.0 on (https://switchbrew.org/wiki/Applet_Manager_services).
const CRADLE_DEVICE_INFO_SIZE: usize = 0xC;
/// `nn::applet::ApplicationLaunchRequestInfo` is 0x10 bytes.
const APPLICATION_LAUNCH_REQUEST_INFO_SIZE: usize = 0x10;

/// State owned by `IHomeMenuFunctions` and `IGlobalStateController`.
pub(super) struct SystemState {
    foreground_requests: AtomicU32,
    foreground_locked: AtomicBool,
    general_channel: Mutex<VecDeque<Vec<u8>>>,
    sleep_enabled: AtomicBool,
    reboot_enabled: AtomicBool,
    sleep_requested: AtomicBool,
    shutdown_requested: AtomicBool,
    reboot_requested: AtomicBool,
    auto_power_down_requested: AtomicBool,
    idle_policy_applied: AtomicBool,
    home_button_long_press_time: AtomicU64,
    last_application_exit_reason: AtomicU32,
}

impl Default for SystemState {
    fn default() -> Self {
        SystemState {
            foreground_requests: AtomicU32::new(0),
            foreground_locked: AtomicBool::new(false),
            general_channel: Mutex::new(VecDeque::new()),
            // Both power operations are system capabilities. The runtime records
            // the request but never performs the transition itself.
            sleep_enabled: AtomicBool::new(true),
            reboot_enabled: AtomicBool::new(true),
            sleep_requested: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            reboot_requested: AtomicBool::new(false),
            auto_power_down_requested: AtomicBool::new(false),
            idle_policy_applied: AtomicBool::new(false),
            home_button_long_press_time: AtomicU64::new(0),
            last_application_exit_reason: AtomicU32::new(0),
        }
    }
}

impl SystemState {
    pub(super) fn sleep_enabled(&self) -> bool {
        self.sleep_enabled.load(Ordering::Relaxed)
    }

    pub(super) fn set_sleep_enabled(&self, enabled: bool) {
        self.sleep_enabled.store(enabled, Ordering::Relaxed);
    }

    /// `ICommonStateGetter::PushToGeneralChannel`. The IPC layer does not resolve
    /// input IStorage objects, so the queue entry carries no payload.
    pub(super) fn push_general_channel(&self) {
        self.general_channel.lock().unwrap().push_back(Vec::new());
    }

    fn pop_general_channel(&self) -> Option<Vec<u8>> {
        self.general_channel.lock().unwrap().pop_front()
    }

    pub(super) fn general_channel_pending(&self) -> bool {
        !self.general_channel.lock().unwrap().is_empty()
    }

    fn request_foreground(&self) {
        self.foreground_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn lock_foreground(&self) {
        self.foreground_locked.store(true, Ordering::Relaxed);
    }

    fn unlock_foreground(&self) {
        self.foreground_locked.store(false, Ordering::Relaxed);
    }

    fn request_sleep(&self) {
        self.sleep_requested.store(true, Ordering::Relaxed);
    }

    fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Relaxed);
    }

    fn request_reboot(&self) {
        self.reboot_requested.store(true, Ordering::Relaxed);
    }

    fn apply_idle_policy(&self) {
        self.idle_policy_applied.store(true, Ordering::Relaxed);
    }

    fn set_home_button_long_press_time(&self, time: u64) {
        self.home_button_long_press_time
            .store(time, Ordering::Relaxed);
    }

    fn set_last_application_exit_reason(&self, reason: u32) {
        self.last_application_exit_reason
            .store(reason, Ordering::Relaxed);
    }
}

/// The `appletAE` object table. `None` means the command number is not part of
/// `nn::am::service::IAllSystemAppletProxiesService`.
pub(super) fn open(cmd: u32, state: Arc<AmState>) -> Option<Reply> {
    let reply = match cmd {
        100 | 110 => Reply::ok().object(Arc::new(SystemAppletProxy::new(state))),
        200 | 201 => Reply::ok().object(Arc::new(LibraryAppletProxy::new(state))),
        300 => Reply::ok().object(Arc::new(OverlayAppletProxy::new(state))),
        350 => Reply::ok().object(Arc::new(ApplicationProxy::new(state))),
        1000 => Reply::ok().object(Arc::new(DebugFunctions::new(state))),
        // CreateSelfLibraryAppletCreatorForDevelop and GetSystemAppletControllerForDebug
        // need the debug-mode flag and `am.debug!dev_function`; this runtime does not
        // expose a debug environment. GetSystemProcessCommonFunctions and
        // IAppletAlternativeFunctions are not modelled.
        400 | 410 | 450 | 460 => Reply::err(RESULT_NOT_AVAILABLE),
        // Switch 2 only.
        310 | 500 => Reply::err(RESULT_NOT_AVAILABLE),
        _ => return None,
    };
    Some(reply)
}

/// `nn::am::service::ISystemAppletProxy`.
pub(super) struct SystemAppletProxy {
    state: Arc<AmState>,
}

impl SystemAppletProxy {
    fn new(state: Arc<AmState>) -> Self {
        SystemAppletProxy { state }
    }

    fn object(&self, cmd: u32) -> Option<Arc<dyn Service>> {
        let obj: Arc<dyn Service> = match cmd {
            20 => Arc::new(HomeMenuFunctions::new(self.state.clone())),
            21 => Arc::new(GlobalStateController::new(self.state.clone())),
            22 => Arc::new(ApplicationCreator),
            23 => Arc::new(AppletCommonFunctions::new(self.state.clone())),
            _ => return applet_manager(&self.state, cmd),
        };
        Some(obj)
    }
}

impl Service for SystemAppletProxy {
    fn name(&self) -> String {
        "am/ISystemAppletProxy".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.object(req.cmd) {
            Some(obj) => Reply::ok().object(obj),
            None => unimplemented("am/ISystemAppletProxy", c, req),
        }
    }
}

/// `nn::am::service::ILibraryAppletProxy`.
///
/// The library applet self accessor (command 20) is not implemented: the runtime
/// has no library applet process to expose.
pub(super) struct LibraryAppletProxy {
    state: Arc<AmState>,
}

impl LibraryAppletProxy {
    fn new(state: Arc<AmState>) -> Self {
        LibraryAppletProxy { state }
    }

    fn object(&self, cmd: u32) -> Option<Arc<dyn Service>> {
        let obj: Arc<dyn Service> = match cmd {
            // 7.0.0+ GetAppletCommonFunctions, the same number the appletOE
            // shortcut in `ApplicationProxy` uses.
            21 => Arc::new(AppletCommonFunctions::new(self.state.clone())),
            22 => Arc::new(HomeMenuFunctions::new(self.state.clone())),
            23 => Arc::new(GlobalStateController::new(self.state.clone())),
            _ => return applet_manager(&self.state, cmd),
        };
        Some(obj)
    }
}

impl Service for LibraryAppletProxy {
    fn name(&self) -> String {
        "am/ILibraryAppletProxy".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.object(req.cmd) {
            Some(obj) => Reply::ok().object(obj),
            None => unimplemented("am/ILibraryAppletProxy", c, req),
        }
    }
}

/// `nn::am::service::IOverlayAppletProxy`.
///
/// `IOverlayFunctions` (command 20) is not modelled: the runtime has no overlay
/// layer or overlay-specific power path.
pub(super) struct OverlayAppletProxy {
    state: Arc<AmState>,
}

impl OverlayAppletProxy {
    fn new(state: Arc<AmState>) -> Self {
        OverlayAppletProxy { state }
    }

    fn object(&self, cmd: u32) -> Option<Arc<dyn Service>> {
        let obj: Arc<dyn Service> = match cmd {
            21 => Arc::new(AppletCommonFunctions::new(self.state.clone())),
            23 => Arc::new(GlobalStateController::new(self.state.clone())),
            _ => return applet_manager(&self.state, cmd),
        };
        Some(obj)
    }
}

impl Service for OverlayAppletProxy {
    fn name(&self) -> String {
        "am/IOverlayAppletProxy".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.object(req.cmd) {
            Some(obj) => Reply::ok().object(obj),
            None => unimplemented("am/IOverlayAppletProxy", c, req),
        }
    }
}

/// `nn::am::service::IHomeMenuFunctions`.
///
/// The home menu runs alone in this runtime, so the foreground requests only move
/// the recorded lock state, and the system applet/starter launches are accepted
/// without starting a second process. The general channel is the real round trip
/// the IPC layer allows: pushes carry no payload, pops return an empty storage.
pub(super) struct HomeMenuFunctions {
    state: Arc<AmState>,
    writer_locks: [Arc<LockAccessor>; 4],
}

impl HomeMenuFunctions {
    fn new(state: Arc<AmState>) -> Self {
        HomeMenuFunctions {
            state,
            writer_locks: std::array::from_fn(|_| Arc::new(LockAccessor::deferred())),
        }
    }

    /// `GetWriterLockAccessorEx` only accepts 0-3, where 0 is the home button.
    /// `GetHomeButtonWriterLockAccessor` is the same lock as index 0.
    fn writer_lock(&self, index: u32) -> Option<Arc<dyn Service>> {
        self.writer_locks
            .get(index as usize)
            .map(|lock| lock.clone() as Arc<dyn Service>)
    }

    fn command(&self, req: &Request) -> Option<Reply> {
        let system = &self.state.system;
        let reply = match req.cmd {
            // The home menu would come forward; there is no other applet to yield to.
            10 => {
                system.request_foreground();
                Reply::ok()
            }
            11 => {
                system.lock_foreground();
                Reply::ok()
            }
            12 => {
                system.unlock_foreground();
                Reply::ok()
            }
            20 => match system.pop_general_channel() {
                Some(data) => Reply::ok().object(Arc::new(Storage::new(data))),
                None => Reply::err(RESULT_NOT_AVAILABLE),
            },
            40 => Reply::ok().u8(system.sleep_enabled() as u8),
            41 => Reply::ok().u8(system.reboot_enabled.load(Ordering::Relaxed) as u8),
            // These wrap ns LaunchSystemApplet / LaunchLibraryApplet. The runtime has
            // no ns service and runs one applet, so they are documented no-ops that
            // let the home menu continue its own flow.
            50 | 51 => Reply::ok(),
            // The 19.0.0+ commands have no public behaviour in Switchbrew.
            60 | 61 => Reply::err(RESULT_NOT_AVAILABLE),
            // No application launch request was queued for the debug pop.
            100 => Reply::err(RESULT_NOT_AVAILABLE),
            // Debug mode is off, so force-terminating applications is not disabled.
            110 => Reply::ok().u8(0),
            // LaunchDevMenu requires the debug-mode flag, which the runtime never sets.
            200 => Reply::err(RESULT_NOT_AVAILABLE),
            1000 => {
                system.set_last_application_exit_reason(req.u32(0));
                Reply::ok()
            }
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for HomeMenuFunctions {
    fn name(&self) -> String {
        "am/IHomeMenuFunctions".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        if req.cmd == 30 {
            return Reply::ok().object(self.writer_locks[0].clone());
        }
        if req.cmd == 31 {
            return match self.writer_lock(req.u32(0)) {
                Some(lock) => Reply::ok().object(lock),
                None => Reply::err(RESULT_NOT_AVAILABLE),
            };
        }
        if req.cmd == 21 {
            return Reply::ok().copy_handle(cached_event(
                c,
                &self.state.lazy_events,
                EVENT_GENERAL_CHANNEL,
            ));
        }
        let reply = self.command(req);
        if req.cmd == 20 {
            let event = cached_event(c, &self.state.lazy_events, EVENT_GENERAL_CHANNEL);
            c.process()
                .set_event(event, self.state.system.general_channel_pending());
        }
        match reply {
            Some(reply) => reply,
            None => unimplemented("am/IHomeMenuFunctions", c, req),
        }
    }
}

enum GlobalCommand {
    Reply(Reply),
    HdcpFailedEvent,
    CradleUpdater,
}

/// `nn::am::service::IGlobalStateController`.
///
/// Sleep and power transitions are recorded or rejected: the runtime cannot put the
/// host machine to sleep, shut it down or reboot it on behalf of a guest. The idle
/// policy, CEC notification and cradle updater are accepted as system bookkeeping.
pub(super) struct GlobalStateController {
    state: Arc<AmState>,
}

impl GlobalStateController {
    fn new(state: Arc<AmState>) -> Self {
        GlobalStateController { state }
    }

    fn command(&self, req: &Request) -> Option<GlobalCommand> {
        use GlobalCommand::{CradleUpdater, HdcpFailedEvent, Reply as Data};
        let system = &self.state.system;
        let reply = match req.cmd {
            // Stubbed on hardware too: the system applet must use the sleep sequence.
            0 | 1 => Data(Reply::err(RESULT_NOT_AVAILABLE)),
            // The u8 says whether the sleep is a "real" sleep; the runtime records the
            // request and never suspends the process.
            2 => {
                system.request_sleep();
                Data(Reply::ok())
            }
            3 => {
                system.request_shutdown();
                Data(Reply::ok())
            }
            4 => {
                system.request_reboot();
                Data(Reply::ok())
            }
            // No idle policy is configured and the console never auto powers down.
            9 => {
                Data(Reply::ok().u8(system.auto_power_down_requested.load(Ordering::Relaxed) as u8))
            }
            10 => {
                system.apply_idle_policy();
                Data(Reply::ok())
            }
            11 => Data(Reply::ok()),
            12 => {
                system.set_home_button_long_press_time(req.u64(0));
                Data(Reply::ok())
            }
            13 => Data(Reply::ok()),
            // The console boots to the home menu, never to sleep.
            14 => Data(Reply::ok().u8(0)),
            15 => HdcpFailedEvent,
            // 23.0.0+ command whose behaviour Switchbrew does not document.
            16 => Data(Reply::err(RESULT_NOT_AVAILABLE)),
            30 => CradleUpdater,
            // Switch 2 only.
            100 => Data(Reply::err(RESULT_NOT_AVAILABLE)),
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for GlobalStateController {
    fn name(&self) -> String {
        "am/IGlobalStateController".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(GlobalCommand::Reply(reply)) => reply,
            Some(GlobalCommand::HdcpFailedEvent) => {
                Reply::ok().copy_handle(cached_event(c, &self.state.lazy_events, EVENT_HDCP_FAILED))
            }
            Some(GlobalCommand::CradleUpdater) => Reply::ok().object(Arc::new(
                CradleFirmwareUpdater::new(c.process().create_event(false)),
            )),
            None => unimplemented("am/IGlobalStateController", c, req),
        }
    }
}

/// `nn::am::service::ICradleFirmwareUpdater` (12.0.0+).
///
/// No cradle is attached, so there is no device info and no firmware image to
/// update. `FinishUpdate` stays accepted because it only ends an update.
pub(super) struct CradleFirmwareUpdater {
    event: u32,
}

impl CradleFirmwareUpdater {
    fn new(event: u32) -> Self {
        CradleFirmwareUpdater { event }
    }

    fn command(&self, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            // StartUpdate: the runtime has neither a cradle nor a firmware image.
            0 => Reply::err(RESULT_NOT_AVAILABLE),
            1 => Reply::ok(),
            // CradleDeviceInfo is all-zero without a cradle. The 0xC size matches
            // 13.0.0+ firmware, which is what this runtime targets.
            2 => Reply::ok().bytes(&[0; CRADLE_DEVICE_INFO_SIZE]),
            3 => Reply::ok().copy_handle(self.event),
            // Switchbrew documents neither reply, and no update ran.
            4 | 5 => Reply::err(RESULT_NOT_AVAILABLE),
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for CradleFirmwareUpdater {
    fn name(&self) -> String {
        "am/ICradleFirmwareUpdater".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(reply) => reply,
            None => unimplemented("am/ICradleFirmwareUpdater", c, req),
        }
    }
}

/// `nn::am::service::IApplicationCreator`.
pub(super) struct ApplicationCreator;

enum CreatorCommand {
    Create { applet_id: u32, application_id: u64 },
    Reject(Reply),
}

impl ApplicationCreator {
    fn command(&self, req: &Request) -> Option<CreatorCommand> {
        let reply = match req.cmd {
            // CreateApplication uses AppletId 0x01; the initial launch request info is
            // 0x3/0x3 (ApplicationLaunchRequestInfo offsets 0x0 and 0x4).
            0 => CreatorCommand::Create {
                applet_id: APPLET_ID_APPLICATION,
                application_id: req.u64(0),
            },
            // CreateSystemApplication uses AppletId 0x04 and an all-zero request info.
            10 => CreatorCommand::Create {
                applet_id: APPLET_ID_SYSTEM_APPLICATION,
                application_id: req.u64(0),
            },
            // The runtime never queues an application launch or a floating app.
            1 | 100 => CreatorCommand::Reject(Reply::err(RESULT_NOT_AVAILABLE)),
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for ApplicationCreator {
    fn name(&self) -> String {
        "am/IApplicationCreator".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(CreatorCommand::Create {
                applet_id,
                application_id,
            }) => Reply::ok().object(Arc::new(ApplicationAccessor::new(
                application_id,
                ApplicationAccessor::launch_request_info(applet_id),
                c.process().create_event(false),
            ))),
            Some(CreatorCommand::Reject(reply)) => reply,
            None => unimplemented("am/IApplicationCreator", c, req),
        }
    }
}

/// `nn::am::service::IApplicationAccessor`.
///
/// The accessor describes an application the home menu created but that this runtime
/// will never start: `Start` is the only command that would launch it and answers
/// the documented "not available" result instead of pretending the process exists.
/// Everything else answers from the launch request the creator built.
pub(super) struct ApplicationAccessor {
    application_id: u64,
    state_event: u32,
    completed: AtomicBool,
    exit_requested: AtomicBool,
    result: AtomicU32,
    users_available: AtomicBool,
    launch_request: [u8; APPLICATION_LAUNCH_REQUEST_INFO_SIZE],
}

impl ApplicationAccessor {
    fn new(
        application_id: u64,
        launch_request: [u8; APPLICATION_LAUNCH_REQUEST_INFO_SIZE],
        state_event: u32,
    ) -> Self {
        ApplicationAccessor {
            application_id,
            state_event,
            completed: AtomicBool::new(false),
            exit_requested: AtomicBool::new(false),
            result: AtomicU32::new(0),
            users_available: AtomicBool::new(true),
            launch_request,
        }
    }

    fn launch_request_info(applet_id: u32) -> [u8; APPLICATION_LAUNCH_REQUEST_INFO_SIZE] {
        let mut raw = [0u8; APPLICATION_LAUNCH_REQUEST_INFO_SIZE];
        if applet_id == APPLET_ID_APPLICATION {
            raw[..4].copy_from_slice(&3u32.to_le_bytes());
            raw[4..8].copy_from_slice(&3u32.to_le_bytes());
        }
        raw
    }

    fn command(&self, req: &Request, current_title_id: u64) -> Option<Reply> {
        let reply = match req.cmd {
            0 => Reply::ok().copy_handle(self.state_event),
            1 => Reply::ok().u8(self.completed.load(Ordering::Relaxed) as u8),
            // Start would launch the application; the runtime runs one application and
            // has no NS launch path, so it must not fake success.
            10 => Reply::err(RESULT_NOT_AVAILABLE),
            // The applet system would forward the exit request to the process. There is
            // none, so only the request is recorded.
            20 => {
                self.exit_requested.store(true, Ordering::Relaxed);
                Reply::ok()
            }
            // Terminate ends an application that is not running: mark the accessor
            // completed so the state-changed event and GetResult become consistent.
            25 => {
                self.completed.store(true, Ordering::Relaxed);
                Reply::ok()
            }
            // The application result is only meaningful once the accessor completed.
            30 => {
                if self.completed.load(Ordering::Relaxed) {
                    Reply::ok().u32(self.result.load(Ordering::Relaxed))
                } else {
                    Reply::err(RESULT_NOT_AVAILABLE)
                }
            }
            101 => Reply::ok(),
            110 => Reply::ok(),
            111 => Reply::ok().u8(0),
            // No library applet was started by this accessor.
            112 => Reply::err(RESULT_NOT_AVAILABLE),
            120 => Reply::ok().u64(self.application_id),
            // The parameter is for the target application's user channel; the target
            // never runs, so it is accepted without being queued.
            121 => Reply::ok(),
            // The runtime has no control.nacp or launch property for a title that is
            // not loaded. Switchbrew also documents an error for system applications.
            122 | 123 => Reply::err(RESULT_NOT_AVAILABLE),
            124 => Reply::ok().bytes(&self.launch_request),
            130 => {
                if input_bool(&req.data, 0) {
                    self.users_available.store(false, Ordering::Relaxed);
                }
                Reply::ok()
            }
            // No rights environment was assigned because no process was created.
            131 => Reply::ok().u8(0),
            132 => Reply::err(RESULT_NOT_AVAILABLE),
            150 => Reply::ok(),
            160 => Reply::ok(),
            161 | 300 | 301 | 310 => Reply::err(RESULT_NOT_AVAILABLE),
            // Only the running title has provisioned save data.
            170 => Reply::ok().u8((self.application_id == current_title_id) as u8),
            180 | 190 => Reply::ok(),
            200 | 201 => Reply::ok(),
            _ => return None,
        };
        Some(reply)
    }

    /// Number of uids `GetDesirableUids` reports for an output buffer of `size`
    /// bytes: one local user, unless `SetUsers(true)` cleared the availability.
    fn desirable_uid_count(&self, size: u64) -> u32 {
        if self.users_available.load(Ordering::Relaxed) {
            (size / 16).min(1) as u32
        } else {
            0
        }
    }

    /// `GetDesirableUids`: the runtime has one local user, and `SetUsers(true)`
    /// makes it unavailable. The uid array goes into the type-0x6 output buffer.
    fn desirable_uids(&self, c: &Cpu, req: &Request) -> Reply {
        let Some(buf) = req.out_buf(0) else {
            return Reply::ok().u32(0);
        };
        let count = self.desirable_uid_count(buf.size);
        if count == 1 {
            c.process().memory.write(buf.addr, &USER_ID);
        }
        Reply::ok().u32(count)
    }
}

impl Service for ApplicationAccessor {
    fn name(&self) -> String {
        "am/IApplicationAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        // GetDesirableUids is the only command that writes guest memory.
        if req.cmd == 140 {
            return self.desirable_uids(c, req);
        }
        let reply = self.command(req, c.process().title_id);
        if req.cmd == 25 {
            c.process().set_event(self.state_event, true);
        }
        match reply {
            Some(reply) => reply,
            None => unimplemented("am/IApplicationAccessor", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{empty_ok, request, state};
    use super::*;

    fn home_reply(menu: &HomeMenuFunctions, cmd: u32, data: Vec<u8>) -> Reply {
        menu.command(&request(cmd, data))
            .unwrap_or_else(|| panic!("command {cmd} should reply"))
    }

    fn global_reply(controller: &GlobalStateController, cmd: u32, data: Vec<u8>) -> Reply {
        match controller.command(&request(cmd, data)) {
            Some(GlobalCommand::Reply(reply)) => reply,
            _ => panic!("command {cmd} should return a data reply"),
        }
    }

    fn accessor() -> ApplicationAccessor {
        ApplicationAccessor::new(
            0x0100_0000_0000_1000,
            ApplicationAccessor::launch_request_info(APPLET_ID_APPLICATION),
            5,
        )
    }

    fn accessor_reply(accessor: &ApplicationAccessor, cmd: u32, current_title_id: u64) -> Reply {
        accessor
            .command(&request(cmd, Vec::new()), current_title_id)
            .unwrap_or_else(|| panic!("command {cmd} should reply"))
    }

    #[test]
    fn applet_ae_returns_the_documented_proxies() {
        for (cmd, name) in [
            (100, "am/ISystemAppletProxy"),
            (110, "am/ISystemAppletProxy"),
            (200, "am/ILibraryAppletProxy"),
            (201, "am/ILibraryAppletProxy"),
            (300, "am/IOverlayAppletProxy"),
            (350, "am/IApplicationProxy"),
            (1000, "am/IDebugFunctions"),
        ] {
            let reply = open(cmd, state()).unwrap_or_else(|| panic!("command {cmd} should reply"));
            assert_eq!(reply.result, 0, "command {cmd}");
            assert_eq!(reply.objects.len(), 1, "command {cmd}");
            assert_eq!(reply.objects[0].name(), name, "command {cmd}");
        }

        // Debug-only and unmodelled proxies answer the AM "not available" result.
        for cmd in [400, 410, 450, 460, 310, 500] {
            let reply = open(cmd, state()).unwrap_or_else(|| panic!("command {cmd} should reply"));
            assert_eq!(reply.result, RESULT_NOT_AVAILABLE, "command {cmd}");
            assert!(reply.objects.is_empty(), "command {cmd}");
        }
        assert!(open(999, state()).is_none());
    }

    #[test]
    fn system_proxy_returns_the_documented_managers() {
        let proxy = SystemAppletProxy::new(state());
        for (cmd, name) in [
            (0, "am/ICommonStateGetter"),
            (1, "am/ISelfController"),
            (2, "am/IWindowController"),
            (3, "am/IAudioController"),
            (4, "am/IDisplayController"),
            (10, "am/IProcessWindingController"),
            (11, "am/ILibraryAppletCreator"),
            (20, "am/IHomeMenuFunctions"),
            (21, "am/IGlobalStateController"),
            (22, "am/IApplicationCreator"),
            (23, "am/IAppletCommonFunctions"),
            (1000, "am/IDebugFunctions"),
        ] {
            let object = proxy
                .object(cmd)
                .unwrap_or_else(|| panic!("command {cmd} has no manager"));
            assert_eq!(object.name(), name, "command {cmd}");
        }
        assert!(proxy.object(24).is_none());
    }

    #[test]
    fn library_and_overlay_proxies_share_the_common_managers() {
        let library = LibraryAppletProxy::new(state());
        assert_eq!(
            library.object(21).unwrap().name(),
            "am/IAppletCommonFunctions"
        );
        assert_eq!(library.object(22).unwrap().name(), "am/IHomeMenuFunctions");
        assert_eq!(
            library.object(23).unwrap().name(),
            "am/IGlobalStateController"
        );
        assert_eq!(library.object(0).unwrap().name(), "am/ICommonStateGetter");
        // The library applet self accessor is not modelled.
        assert!(library.object(20).is_none());

        let overlay = OverlayAppletProxy::new(state());
        assert_eq!(
            overlay.object(21).unwrap().name(),
            "am/IAppletCommonFunctions"
        );
        assert_eq!(
            overlay.object(23).unwrap().name(),
            "am/IGlobalStateController"
        );
        assert_eq!(overlay.object(0).unwrap().name(), "am/ICommonStateGetter");
        // IOverlayFunctions and HomeMenuFunctions are not part of this proxy.
        assert!(overlay.object(20).is_none());
        assert!(overlay.object(22).is_none());
    }

    #[test]
    fn home_menu_tracks_the_foreground_lock() {
        let state = state();
        let menu = HomeMenuFunctions::new(state.clone());
        assert!(!state.system.foreground_locked.load(Ordering::Relaxed));
        assert_eq!(state.system.foreground_requests.load(Ordering::Relaxed), 0);

        empty_ok(&home_reply(&menu, 10, Vec::new()));
        empty_ok(&home_reply(&menu, 10, Vec::new()));
        assert_eq!(state.system.foreground_requests.load(Ordering::Relaxed), 2);

        empty_ok(&home_reply(&menu, 11, Vec::new()));
        assert!(state.system.foreground_locked.load(Ordering::Relaxed));
        empty_ok(&home_reply(&menu, 12, Vec::new()));
        assert!(!state.system.foreground_locked.load(Ordering::Relaxed));
    }

    #[test]
    fn home_menu_reports_the_power_capabilities() {
        let state = state();
        let menu = HomeMenuFunctions::new(state.clone());

        assert_eq!(home_reply(&menu, 40, Vec::new()).data, [1]);
        assert_eq!(home_reply(&menu, 41, Vec::new()).data, [1]);

        state.system.set_sleep_enabled(false);
        assert_eq!(home_reply(&menu, 40, Vec::new()).data, [0]);
        assert_eq!(home_reply(&menu, 41, Vec::new()).data, [1]);
    }

    #[test]
    fn home_menu_pops_the_general_channel() {
        let state = state();
        let menu = HomeMenuFunctions::new(state.clone());

        assert_eq!(
            home_reply(&menu, 20, Vec::new()).result,
            RESULT_NOT_AVAILABLE,
            "PopFromGeneralChannel with an empty channel"
        );

        state.system.push_general_channel();
        assert!(state.system.general_channel_pending());
        let popped = home_reply(&menu, 20, Vec::new());
        assert_eq!(popped.result, 0);
        assert_eq!(popped.objects.len(), 1);
        assert_eq!(popped.objects[0].name(), "am/IStorage");
        assert!(!state.system.general_channel_pending());
    }

    #[test]
    fn home_menu_writer_locks_cover_the_documented_indexes() {
        let menu = HomeMenuFunctions::new(state());
        for index in 0..4 {
            let lock = menu
                .writer_lock(index)
                .unwrap_or_else(|| panic!("index {index} has no lock"));
            assert_eq!(lock.name(), "am/ILockAccessor", "index {index}");
        }
        assert!(menu.writer_lock(4).is_none());
    }

    #[test]
    fn home_menu_launches_are_no_ops_and_debug_queries_answer_offline() {
        let state = state();
        let menu = HomeMenuFunctions::new(state.clone());

        // LaunchSystemApplet and LaunchStarter accept the request without a process.
        for cmd in [50, 51] {
            empty_ok(&home_reply(&menu, cmd, Vec::new()));
        }

        // The debug-only commands are unavailable with the debug flag off.
        for cmd in [100, 200, 60, 61] {
            assert_eq!(
                home_reply(&menu, cmd, vec![0; 0x20]).result,
                RESULT_NOT_AVAILABLE,
                "command {cmd}"
            );
        }
        assert_eq!(home_reply(&menu, 110, Vec::new()).data, [0]);

        empty_ok(&home_reply(
            &menu,
            1000,
            (-5i32 as u32).to_le_bytes().to_vec(),
        ));
        assert_eq!(
            state
                .system
                .last_application_exit_reason
                .load(Ordering::Relaxed),
            -5i32 as u32
        );
        assert!(menu.command(&request(999, Vec::new())).is_none());
    }

    #[test]
    fn global_state_controller_records_the_power_requests() {
        let state = state();
        let controller = GlobalStateController::new(state.clone());

        // RequestToEnterSleep and EnterSleep are stubbed on hardware too.
        for cmd in [0, 1] {
            assert_eq!(
                global_reply(&controller, cmd, Vec::new()).result,
                RESULT_NOT_AVAILABLE,
                "command {cmd}"
            );
        }

        empty_ok(&global_reply(&controller, 2, vec![1]));
        assert!(state.system.sleep_requested.load(Ordering::Relaxed));
        empty_ok(&global_reply(&controller, 3, Vec::new()));
        assert!(state.system.shutdown_requested.load(Ordering::Relaxed));
        empty_ok(&global_reply(&controller, 4, Vec::new()));
        assert!(state.system.reboot_requested.load(Ordering::Relaxed));

        empty_ok(&global_reply(&controller, 10, Vec::new()));
        assert!(state.system.idle_policy_applied.load(Ordering::Relaxed));
        empty_ok(&global_reply(&controller, 11, Vec::new()));
        empty_ok(&global_reply(
            &controller,
            12,
            9000u64.to_le_bytes().to_vec(),
        ));
        assert_eq!(
            state
                .system
                .home_button_long_press_time
                .load(Ordering::Relaxed),
            9000
        );
        empty_ok(&global_reply(&controller, 13, Vec::new()));

        assert_eq!(global_reply(&controller, 9, Vec::new()).data, [0]);
        assert_eq!(global_reply(&controller, 14, Vec::new()).data, [0]);
        for cmd in [16, 100] {
            assert_eq!(
                global_reply(&controller, cmd, Vec::new()).result,
                RESULT_NOT_AVAILABLE,
                "command {cmd}"
            );
        }
        assert!(controller.command(&request(31, Vec::new())).is_none());
    }

    #[test]
    fn global_state_controller_opens_the_hdcp_event_and_cradle_updater() {
        let state = state();
        let controller = GlobalStateController::new(state);
        match controller.command(&request(15, Vec::new())) {
            Some(GlobalCommand::HdcpFailedEvent) => {}
            _ => panic!("command 15 should name the HDCP event"),
        }
        match controller.command(&request(30, Vec::new())) {
            Some(GlobalCommand::CradleUpdater) => {}
            _ => panic!("command 30 should open the cradle updater"),
        }
    }

    #[test]
    fn cradle_firmware_updater_has_no_cradle_or_update() {
        let updater = CradleFirmwareUpdater::new(9);
        for cmd in [0, 4, 5] {
            let reply = updater
                .command(&request(cmd, Vec::new()))
                .unwrap_or_else(|| panic!("command {cmd} should reply"));
            assert_eq!(reply.result, RESULT_NOT_AVAILABLE, "command {cmd}");
        }
        empty_ok(&updater.command(&request(1, Vec::new())).unwrap());

        let info = updater.command(&request(2, Vec::new())).unwrap();
        assert_eq!(info.result, 0);
        assert_eq!(info.data.len(), CRADLE_DEVICE_INFO_SIZE);
        assert!(info.data.iter().all(|&b| b == 0));

        assert_eq!(
            updater
                .command(&request(3, Vec::new()))
                .unwrap()
                .copy_handles,
            vec![9]
        );
        assert!(updater.command(&request(6, Vec::new())).is_none());
    }

    #[test]
    fn application_creator_selects_the_initial_launch_request_info() {
        let creator = ApplicationCreator;
        assert_eq!(APPLET_ID_APPLICATION, 1);
        assert_eq!(APPLET_ID_SYSTEM_APPLICATION, 4);

        let application = ApplicationAccessor::launch_request_info(APPLET_ID_APPLICATION);
        assert_eq!(application.len(), APPLICATION_LAUNCH_REQUEST_INFO_SIZE);
        assert_eq!(&application[..4], &3u32.to_le_bytes());
        assert_eq!(&application[4..8], &3u32.to_le_bytes());
        assert_eq!(&application[8..], &[0; 8]);

        let system = ApplicationAccessor::launch_request_info(APPLET_ID_SYSTEM_APPLICATION);
        assert_eq!(system, [0; APPLICATION_LAUNCH_REQUEST_INFO_SIZE]);

        match creator.command(&request(0, 0x0100_0000_0000_1000u64.to_le_bytes().to_vec())) {
            Some(CreatorCommand::Create {
                applet_id,
                application_id,
            }) => {
                assert_eq!(applet_id, APPLET_ID_APPLICATION);
                assert_eq!(application_id, 0x0100_0000_0000_1000);
            }
            _ => panic!("CreateApplication should create an accessor"),
        }
        match creator.command(&request(
            10,
            0x0100_0000_0000_1012u64.to_le_bytes().to_vec(),
        )) {
            Some(CreatorCommand::Create {
                applet_id,
                application_id,
            }) => {
                assert_eq!(applet_id, APPLET_ID_SYSTEM_APPLICATION);
                assert_eq!(application_id, 0x0100_0000_0000_1012);
            }
            _ => panic!("CreateSystemApplication should create an accessor"),
        }

        // PopLaunchRequestedApplication and PopFloatingApplicationForDevelopment have
        // nothing queued and must not return a fake accessor.
        for cmd in [1, 100] {
            match creator.command(&request(cmd, Vec::new())) {
                Some(CreatorCommand::Reject(reply)) => {
                    assert_eq!(reply.result, RESULT_NOT_AVAILABLE, "command {cmd}");
                    assert!(reply.objects.is_empty(), "command {cmd}");
                }
                _ => panic!("command {cmd} should be rejected"),
            }
        }
        assert!(creator.command(&request(2, Vec::new())).is_none());
    }

    #[test]
    fn application_accessor_reports_identity_and_launch_parameters() {
        let accessor = accessor();

        assert_eq!(
            accessor_reply(&accessor, 120, 0).data,
            0x0100_0000_0000_1000u64.to_le_bytes()
        );
        assert_eq!(
            accessor_reply(&accessor, 124, 0).data,
            ApplicationAccessor::launch_request_info(APPLET_ID_APPLICATION)
        );
        assert_eq!(accessor_reply(&accessor, 0, 0).copy_handles, vec![5]);
        assert_eq!(accessor_reply(&accessor, 1, 0).data, [0]);

        // No library applet or rights environment exists for an unstarted application.
        assert_eq!(
            accessor_reply(&accessor, 112, 0).result,
            RESULT_NOT_AVAILABLE
        );
        assert_eq!(accessor_reply(&accessor, 131, 0).data, [0]);
        assert_eq!(
            accessor_reply(&accessor, 132, 0).result,
            RESULT_NOT_AVAILABLE
        );

        for cmd in [101, 110, 121, 150, 160, 180, 190, 200, 201] {
            empty_ok(&accessor_reply(&accessor, cmd, 0));
        }
        assert_eq!(accessor_reply(&accessor, 111, 0).data, [0]);
    }

    #[test]
    fn application_accessor_start_does_not_fake_a_launch() {
        let accessor = accessor();
        for cmd in [10, 122, 123, 161, 300, 301, 310] {
            let reply = accessor_reply(&accessor, cmd, 0);
            assert_eq!(reply.result, RESULT_NOT_AVAILABLE, "command {cmd}");
            assert!(reply.objects.is_empty(), "command {cmd}");
        }
    }

    #[test]
    fn application_accessor_completes_on_terminate() {
        let accessor = accessor();
        assert_eq!(
            accessor_reply(&accessor, 30, 0).result,
            RESULT_NOT_AVAILABLE,
            "GetResult before completion"
        );

        empty_ok(&accessor_reply(&accessor, 20, 0));
        assert!(accessor.exit_requested.load(Ordering::Relaxed));
        assert!(!accessor.completed.load(Ordering::Relaxed));

        empty_ok(&accessor_reply(&accessor, 25, 0));
        assert!(accessor.completed.load(Ordering::Relaxed));
        assert_eq!(accessor_reply(&accessor, 1, 0).data, [1]);
        assert_eq!(accessor_reply(&accessor, 30, 0).data, 0u32.to_le_bytes());
    }

    #[test]
    fn application_accessor_filters_users_and_save_data() {
        let accessor = accessor();
        assert_eq!(
            accessor_reply(&accessor, 170, 0x0100_0000_0000_1000).data,
            [1]
        );
        assert_eq!(
            accessor_reply(&accessor, 170, 0x0100_0000_0000_2000).data,
            [0]
        );

        assert_eq!(accessor.desirable_uid_count(0), 0);
        assert_eq!(accessor.desirable_uid_count(8), 0);
        assert_eq!(accessor.desirable_uid_count(16), 1);
        assert_eq!(accessor.desirable_uid_count(0x100), 1);

        // SetUsers(true) clears users_available and makes the list empty.
        let mut data = vec![0u8; 0x10];
        data[0] = 1;
        empty_ok(
            &accessor
                .command(&request(130, data), 0)
                .expect("SetUsers should reply"),
        );
        assert_eq!(accessor.desirable_uid_count(0x100), 0);
    }
}
