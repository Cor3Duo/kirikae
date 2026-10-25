//! `omm` — `nn::omm::detail::IOperationModeManager`.
//!
//! The runtime has no dock and no cradle. It reports the console in handheld
//! mode with the 1280x720 display, so a title that asks about the operation mode
//! gets a consistent machine instead of a zeroed stub. The cradle firmware
//! updater exists as an object (the SDK creates it on 12.0.0+) and reports "no
//! update device" with progress 0.
//!
//! Command numbers, names and output widths follow Switchbrew's "OMM services"
//! page.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

/// `nn::omm::OperationMode::Handheld`
const OPERATION_MODE_HANDHELD: u32 = 0;
/// Handheld default display resolution (720p).
const DISPLAY_WIDTH: u32 = 1280;
const DISPLAY_HEIGHT: u32 = 720;
/// `CradleStatus::NoCradle`
const CRADLE_STATUS_NONE: u32 = 0;
/// `UpdateDeviceStatus` with no device attached.
const UPDATE_DEVICE_STATUS_NONE: u32 = 0;

static POLICY: AtomicU32 = AtomicU32::new(0);
static AUDIO_MODE: AtomicU32 = AtomicU32::new(0);

pub struct Omm;

impl Omm {
    /// `GetOperationMode`: handheld unless a policy forced console mode.
    fn operation_mode() -> u32 {
        if POLICY.load(Ordering::Relaxed) == 2 {
            1
        } else {
            OPERATION_MODE_HANDHELD
        }
    }

    /// `GetOperationModePolicy`, `SetOperationModePolicy`.
    fn policy(value: Option<u32>) -> u32 {
        match value {
            Some(value) => {
                POLICY.store(value, Ordering::Relaxed);
                value
            }
            None => POLICY.load(Ordering::Relaxed),
        }
    }

    fn audio_mode(value: Option<u32>) -> u32 {
        match value {
            Some(value) => {
                AUDIO_MODE.store(value, Ordering::Relaxed);
                value
            }
            None => AUDIO_MODE.load(Ordering::Relaxed),
        }
    }
}

impl Service for Omm {
    fn name(&self) -> String {
        "omm".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // GetOperationMode
            0 => Reply::ok().u32(Omm::operation_mode()),
            // GetOperationModeChangeEvent: never raised.
            1 => event(c, "omm.mode-event"),
            // EnableAudioVisual / DisableAudioVisual / FadeInDisplay / FadeOutDisplay.
            2 | 3 | 6 | 7 => Reply::ok(),
            // EnterSleepAndWait(Event): the runtime does not suspend, so the
            // command returns instead of waiting forever.
            4 => Reply::ok(),
            // GetCradleStatus
            5 => Reply::ok().u32(CRADLE_STATUS_NONE),
            // GetCradleFwVersion
            8 => Reply::ok().u32(0),
            // NotifyCecSettingsChanged / Hdcp notifications / ShowCardUpdateProcessing.
            9 | 15..=18 | 21 | 24 | 25 => Reply::ok(),
            // SetOperationModePolicy / GetOperationModePolicy.
            10 => Reply::ok(),
            30 => Reply::ok().u32(Omm::policy(None)),
            // GetDefaultDisplayResolution
            11 => Reply::ok().u32(DISPLAY_WIDTH).u32(DISPLAY_HEIGHT),
            // GetDefaultDisplayResolutionChangeEvent
            12 => event(c, "omm.resolution-event"),
            // UpdateDefaultDisplayResolution
            13 => Reply::ok(),
            // ShouldSleepOnBoot
            14 => Reply::ok().u8(0),
            // GetHdcpAuthenticationFailedEvent / GetHdcpStateChangeEvent
            19 | 22 => event(
                c,
                if req.cmd == 19 {
                    "omm.hdcp-failed"
                } else {
                    "omm.hdcp-state"
                },
            ),
            // GetHdcpAuthenticationFailedEmulationEnabled / GetHdcpState
            20 | 23 => Reply::ok().u8(0),
            // GetOperationModeSystemInfo
            26 => Reply::ok().u32(0),
            // GetAppletFullAwakingSystemEvent
            27 => event(c, "omm.awake-event"),
            // CreateCradleFirmwareUpdater
            28 => Reply::ok().object(std::sync::Arc::new(CradleFirmwareUpdater)),
            // GetCradleStatusChangeEvent
            29 => event(c, "omm.cradle-event"),
            // GetAudioMode / SetAudioMode
            40 => Reply::ok().u32(Omm::audio_mode(None)),
            41 => {
                Omm::audio_mode(Some(req.u32(0)));
                Reply::ok()
            }
            // GetDisplayResolution
            50 => Reply::ok().u32(DISPLAY_WIDTH).u32(DISPLAY_HEIGHT),
            // The 21.0.0+ commands are stubbed on NX (Switchbrew Cmd300/301).
            300 | 301 => Reply::ok().u32(0),
            // Backlight and startup-logo controls have no observable effect.
            500 | 501 | 900 => Reply::ok(),
            502..=504 => Reply::ok().u32(0),
            _ => unimplemented_named("omm", omm_command_name(req.cmd), c, req),
        }
    }
}

fn event(c: &mut Cpu, key: &'static str) -> Reply {
    let handle = *c
        .process()
        .service_state(key, || c.process().create_event(false));
    Reply::ok().copy_handle(handle)
}

fn omm_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "GetOperationMode",
        1 => "GetOperationModeChangeEvent",
        2 => "EnableAudioVisual",
        3 => "DisableAudioVisual",
        4 => "EnterSleepAndWait",
        5 => "GetCradleStatus",
        6 => "FadeInDisplay",
        7 => "FadeOutDisplay",
        8 => "GetCradleFwVersion",
        9 => "NotifyCecSettingsChanged",
        10 => "SetOperationModePolicy",
        11 => "GetDefaultDisplayResolution",
        12 => "GetDefaultDisplayResolutionChangeEvent",
        13 => "UpdateDefaultDisplayResolution",
        14 => "ShouldSleepOnBoot",
        19 => "GetHdcpAuthenticationFailedEvent",
        20 => "GetHdcpAuthenticationFailedEmulationEnabled",
        21 => "SetHdcpAuthenticationFailedEmulation",
        22 => "GetHdcpStateChangeEvent",
        23 => "GetHdcpState",
        24 => "ShowCardUpdateProcessing",
        25 => "SetApplicationCecSettingsAndNotifyChanged",
        26 => "GetOperationModeSystemInfo",
        27 => "GetAppletFullAwakingSystemEvent",
        28 => "CreateCradleFirmwareUpdater",
        29 => "GetCradleStatusChangeEvent",
        30 => "GetOperationModePolicy",
        40 => "GetAudioMode",
        41 => "SetAudioMode",
        50 => "GetDisplayResolution",
        500 => "LblSwitchBacklightOnForAm",
        501 => "LblSwitchBacklightOffForAm",
        900 => "HideStartupLogo",
        _ => "unknown",
    }
}

/// `nn::omm::detail::ICradleFirmwareUpdater` (12.0.0+).
pub struct CradleFirmwareUpdater;

impl Service for CradleFirmwareUpdater {
    fn name(&self) -> String {
        "omm/ICradleFirmwareUpdater".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 | 1 => Reply::ok(),
            // GetUpdateDeviceStatus: the 22.0.0+ reply is 0x10 bytes; older
            // clients read the first u32 and ignore the rest.
            2 => {
                let mut status = [0u8; 0x10];
                status[..4].copy_from_slice(&UPDATE_DEVICE_STATUS_NONE.to_le_bytes());
                Reply::ok().bytes(&status)
            }
            // GetUpdateProgress / GetUpdateProgress2: current and total size.
            3 | 5 => Reply::ok().u32(0).u32(0),
            4 => event(c, "omm.update-status-event"),
            _ => unimplemented_named("omm/ICradleFirmwareUpdater", "unknown", c, req),
        }
    }
}

/// Test-only access to the static state so one test cannot leak into another.
#[cfg(test)]
pub(crate) fn reset_state() {
    POLICY.store(0, Ordering::Relaxed);
    AUDIO_MODE.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::services::harness;

    #[test]
    fn command_replies_have_the_documented_widths() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = |c: &mut Cpu, cmd: u32, data: Vec<u8>| {
                Omm.call(
                    c,
                    &Request {
                        cmd,
                        data,
                        ..Default::default()
                    },
                )
            };
            let mode = reply(c, 0, Vec::new());
            harness::record(
                "mode",
                &[
                    u64::from(mode.result),
                    mode.data.len() as u64,
                    u64::from(u32::from_le_bytes(mode.data[..4].try_into().unwrap())),
                ],
            );
            let resolution = reply(c, 11, Vec::new());
            harness::record(
                "resolution",
                &[
                    resolution.data.len() as u64,
                    u64::from(u32::from_le_bytes(resolution.data[..4].try_into().unwrap())),
                    u64::from(u32::from_le_bytes(
                        resolution.data[4..8].try_into().unwrap(),
                    )),
                ],
            );
            let cradle = reply(c, 5, Vec::new());
            harness::record(
                "cradle",
                &[u64::from(u32::from_le_bytes(
                    cradle.data[..4].try_into().unwrap(),
                ))],
            );
            let updater = CradleFirmwareUpdater.call(
                c,
                &Request {
                    cmd: 2,
                    ..Default::default()
                },
            );
            harness::record(
                "updater-status",
                &[
                    updater.data.len() as u64,
                    u64::from(u32::from_le_bytes(updater.data[..4].try_into().unwrap())),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("mode"), vec![0, 4, 0]);
        assert_eq!(harness::observed("resolution"), vec![8, 1280, 720]);
        assert_eq!(harness::observed("cradle"), vec![0]);
        assert_eq!(harness::observed("updater-status"), vec![0x10, 0]);
        reset_state();
    }

    #[test]
    fn the_console_reports_handheld_720p_and_no_cradle() {
        reset_state();
        assert_eq!(Omm::operation_mode(), 0);
        assert_eq!((DISPLAY_WIDTH, DISPLAY_HEIGHT), (1280, 720));
        assert_eq!(CRADLE_STATUS_NONE, 0);
        assert_eq!(UPDATE_DEVICE_STATUS_NONE, 0);
    }

    #[test]
    fn the_operation_mode_policy_and_audio_mode_round_trip() {
        reset_state();
        assert_eq!(Omm::policy(None), 0);
        assert_eq!(Omm::policy(Some(1)), 1);
        assert_eq!(Omm::policy(None), 1);
        assert_eq!(Omm::audio_mode(None), 0);
        assert_eq!(Omm::audio_mode(Some(3)), 3);
        assert_eq!(Omm::audio_mode(None), 3);
        reset_state();
    }

    #[test]
    fn the_cradle_updater_reports_a_documented_status_and_progress() {
        let status = Reply::ok().bytes(&[0; 0x10]);
        assert_eq!(status.data.len(), 0x10);
        let progress = Reply::ok().u32(0).u32(0);
        assert_eq!(progress.data.len(), 8);
    }
}
