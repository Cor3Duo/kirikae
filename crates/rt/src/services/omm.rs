use std::sync::atomic::{AtomicU32, Ordering};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const OPERATION_MODE_HANDHELD: u32 = 0;
const DISPLAY_WIDTH: u32 = 1280;
const DISPLAY_HEIGHT: u32 = 720;
const CRADLE_STATUS_NONE: u32 = 0;
const UPDATE_DEVICE_STATUS_NONE: u32 = 0;

static POLICY: AtomicU32 = AtomicU32::new(0);
static AUDIO_MODE: AtomicU32 = AtomicU32::new(0);

pub struct Omm;

impl Omm {
    fn operation_mode() -> u32 {
        if POLICY.load(Ordering::Relaxed) == 2 {
            1
        } else {
            OPERATION_MODE_HANDHELD
        }
    }

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
            0 => Reply::ok().u32(Omm::operation_mode()),
            1 => event(c, "omm.mode-event"),
            2 | 3 | 6 | 7 => Reply::ok(),
            4 => Reply::ok(),
            5 => Reply::ok().u32(CRADLE_STATUS_NONE),
            8 => Reply::ok().u32(0),
            9 | 15..=18 | 21 | 24 | 25 => Reply::ok(),
            10 => Reply::ok(),
            30 => Reply::ok().u32(Omm::policy(None)),
            11 => Reply::ok().u32(DISPLAY_WIDTH).u32(DISPLAY_HEIGHT),
            12 => event(c, "omm.resolution-event"),
            13 => Reply::ok(),
            14 => Reply::ok().u8(0),
            19 | 22 => event(
                c,
                if req.cmd == 19 {
                    "omm.hdcp-failed"
                } else {
                    "omm.hdcp-state"
                },
            ),
            20 | 23 => Reply::ok().u8(0),
            26 => Reply::ok().u32(0),
            27 => event(c, "omm.awake-event"),
            28 => Reply::ok().object(std::sync::Arc::new(CradleFirmwareUpdater)),
            29 => event(c, "omm.cradle-event"),
            40 => Reply::ok().u32(Omm::audio_mode(None)),
            41 => {
                Omm::audio_mode(Some(req.u32(0)));
                Reply::ok()
            }
            50 => Reply::ok().u32(DISPLAY_WIDTH).u32(DISPLAY_HEIGHT),
            300 | 301 => Reply::ok().u32(0),
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

pub struct CradleFirmwareUpdater;

impl Service for CradleFirmwareUpdater {
    fn name(&self) -> String {
        "omm/ICradleFirmwareUpdater".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 | 1 => Reply::ok(),
            2 => {
                let mut status = [0u8; 0x10];
                status[..4].copy_from_slice(&UPDATE_DEVICE_STATUS_NONE.to_le_bytes());
                Reply::ok().bytes(&status)
            }
            3 | 5 => Reply::ok().u32(0).u32(0),
            4 => event(c, "omm.update-status-event"),
            _ => unimplemented_named("omm/ICradleFirmwareUpdater", "unknown", c, req),
        }
    }
}

#[cfg(test)]
pub(crate) fn reset_state() {
    POLICY.store(0, Ordering::Relaxed);
    AUDIO_MODE.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_operation_mode_policy_round_trips() {
        reset_state();
        assert_eq!(Omm::policy(Some(2)), 2);
        assert_eq!(Omm::operation_mode(), 1);
        assert_eq!(Omm::policy(Some(0)), 0);
        assert_eq!(Omm::operation_mode(), 0);
    }

    #[test]
    fn the_audio_mode_round_trips() {
        reset_state();
        assert_eq!(Omm::audio_mode(Some(1)), 1);
        assert_eq!(Omm::audio_mode(None), 1);
        assert_eq!(Omm::audio_mode(Some(0)), 0);
    }
}
