//! `nfc:user` / `nfc:sys` / `nfc:mf:u` — NFC and Mii-mifare queries.
//!
//! The runtime has no NFC controller. The services still answer the queries a
//! game can ask offline: the interface initializes, `GetState` reports
//! initialized, `IsNfcEnabled` reports disabled, `ListDevices` is an empty list,
//! and a device state is `Unavailable` (5). Detection, tag reads and mifare
//! writes need hardware and return `InvalidArgument` instead of a fake tag.
//!
//! Command numbers follow Switchbrew's "NFC services" page and libnx's `nfc.c`.

use std::sync::Mutex;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

/// `NfcState::NonInitialized` / `Initialized`.
const STATE_NON_INITIALIZED: u32 = 0;
const STATE_INITIALIZED: u32 = 1;
/// `DeviceState::Unavailable`.
const DEVICE_STATE_UNAVAILABLE: u32 = 5;
/// `ResultInvalidArgument`.
const INVALID_ARGUMENT: u32 = 0x2EE202;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    User,
    System,
    Mifare,
}

pub struct Nfc {
    name: &'static str,
    kind: Kind,
}

impl Nfc {
    pub fn user() -> Nfc {
        Nfc {
            name: "nfc:user",
            kind: Kind::User,
        }
    }

    pub fn system() -> Nfc {
        Nfc {
            name: "nfc:sys",
            kind: Kind::System,
        }
    }

    pub fn mifare() -> Nfc {
        Nfc {
            name: "nfc:mf:u",
            kind: Kind::Mifare,
        }
    }
}

impl Service for Nfc {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        // Every manager has one creator command (0).
        if req.cmd == 0 {
            return Reply::ok().object(std::sync::Arc::new(NfcSession::new(self.kind)));
        }
        unimplemented_named(self.name, "unknown", c, req)
    }
}

struct NfcState {
    initialized: bool,
    enabled: bool,
}

struct NfcSession {
    kind: Kind,
    state: Mutex<NfcState>,
}

impl NfcSession {
    fn new(kind: Kind) -> NfcSession {
        NfcSession {
            kind,
            state: Mutex::new(NfcState {
                initialized: false,
                enabled: false,
            }),
        }
    }

    fn set_initialized(&self, value: bool) -> Reply {
        self.state.lock().unwrap().initialized = value;
        Reply::ok()
    }

    fn set_enabled(&self, value: bool) -> Reply {
        self.state.lock().unwrap().enabled = value;
        Reply::ok()
    }

    fn state(&self) -> Reply {
        let state = self.state.lock().unwrap();
        Reply::ok().u32(if state.initialized {
            STATE_INITIALIZED
        } else {
            STATE_NON_INITIALIZED
        })
    }

    fn is_enabled(&self) -> Reply {
        Reply::ok().u8(u8::from(self.state.lock().unwrap().enabled))
    }

    fn refuse_detection(&self) -> Reply {
        // No controller is attached, so detection cannot start even after the
        // interface was initialized.
        Reply::err(INVALID_ARGUMENT)
    }
}

impl Service for NfcSession {
    fn name(&self) -> String {
        "nfc/ISession".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        // Legacy aliases (0-3 / 100) and the 4.0.0+ numbering (400+) are the
        // same commands. 100/500 are SetNfcEnabled. For mifare the commands are
        // renumbered, so fold them into the same internal numbers.
        let cmd = match self.kind {
            Kind::User | Kind::System => match req.cmd {
                0 | 400 => 0,
                1 | 401 => 1,
                2 | 402 => 2,
                3 | 403 => 3,
                100 | 500 => 100,
                404 => 4,
                405 => 5,
                406 => 6,
                407 => 7,
                408 => 8,
                409 => 9,
                410 => 10,
                411 => 11,
                412 => 12,
                510 => 510,
                1000 => 1000,
                1001 => 1001,
                1300 => 1300,
                1301 => 1301,
                1302 => 1302,
                _ => req.cmd,
            },
            Kind::Mifare => match req.cmd {
                0 => 0,
                1 => 1,
                2 => 4,
                3 => 8,
                4 => 9,
                5 => 1000,
                6 => 1001,
                7 => 10,
                8 => 11,
                9 => 12,
                10 => 2,
                11 => 5,
                12 => 6,
                13 => 7,
                _ => req.cmd,
            },
        };
        match cmd {
            // Initialize
            0 => self.set_initialized(true),
            // Finalize
            1 => self.set_initialized(false),
            // GetState
            2 => self.state(),
            // IsNfcEnabled
            3 => self.is_enabled(),
            // SetNfcEnabled, bool input
            100 => self.set_enabled(req.data.first().is_some_and(|&b| b != 0)),
            // ListDevices: empty list, out count 0.
            4 => Reply::ok().u32(0),
            // GetDeviceState
            5 => Reply::ok().u32(DEVICE_STATE_UNAVAILABLE),
            // GetNpadId: no controller carries a tag.
            6 => Reply::err(INVALID_ARGUMENT),
            // GetAvailabilityChangeEventHandle
            7 => event(c, "nfc.availability"),
            // StartDetection / StopDetection
            8 | 9 => self.refuse_detection(),
            // GetTagInfo
            10 => Reply::err(INVALID_ARGUMENT),
            // GetActivateEventHandle / GetDeactivateEventHandle
            11 => event(c, "nfc.activate"),
            12 => event(c, "nfc.deactivate"),
            // OutputTestWave
            510 => Reply::ok(),
            // ReadMifare / WriteMifare
            1000 | 1001 => Reply::err(INVALID_ARGUMENT),
            // SendCommandByPassThrough / KeepPassThroughSession / Release.
            1300 => Reply::err(INVALID_ARGUMENT),
            1301 | 1302 => Reply::ok(),
            _ => unimplemented_named("nfc", nfc_command_name(req.cmd), c, req),
        }
    }
}

fn event(c: &mut Cpu, key: &'static str) -> Reply {
    let handle = *c
        .process()
        .service_state(key, || c.process().create_event(false));
    Reply::ok().copy_handle(handle)
}

fn nfc_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "Initialize",
        1 => "Finalize",
        2 => "GetState",
        3 => "IsNfcEnabled",
        4 => "ListDevices",
        5 => "GetDeviceState",
        6 => "GetNpadId",
        7 => "GetAvailabilityChangeEventHandle",
        8 => "StartDetection",
        9 => "StopDetection",
        10 => "GetTagInfo",
        11 => "GetActivateEventHandle",
        12 => "GetDeactivateEventHandle",
        100 => "SetNfcEnabled",
        510 => "OutputTestWave",
        1000 => "ReadMifare",
        1001 => "WriteMifare",
        1300 => "SendCommandByPassThrough",
        1301 => "KeepPassThroughSession",
        1302 => "ReleasePassThroughSession",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::services::harness;

    #[test]
    fn the_session_reports_offline_values() {
        extern "C" fn probe(c: &mut Cpu) {
            let manager = Nfc::user().call(c, &Request::default());
            harness::record("object", &[manager.objects.len() as u64]);
            let session = NfcSession::new(Kind::User);
            let init = session.call(
                c,
                &Request {
                    cmd: 400,
                    ..Default::default()
                },
            );
            let state = session.call(
                c,
                &Request {
                    cmd: 402,
                    ..Default::default()
                },
            );
            let devices = session.call(
                c,
                &Request {
                    cmd: 404,
                    ..Default::default()
                },
            );
            let device = session.call(
                c,
                &Request {
                    cmd: 405,
                    ..Default::default()
                },
            );
            harness::record(
                "session",
                &[
                    u64::from(init.result),
                    u64::from(u32::from_le_bytes(state.data[..4].try_into().unwrap())),
                    u64::from(u32::from_le_bytes(devices.data[..4].try_into().unwrap())),
                    u64::from(u32::from_le_bytes(device.data[..4].try_into().unwrap())),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("object"), vec![1]);
        assert_eq!(harness::observed("session"), vec![0, 1, 0, 5]);
    }

    #[test]
    fn the_session_starts_uninitialized_and_disabled() {
        let session = NfcSession::new(Kind::User);
        assert_eq!(session.state().data, STATE_NON_INITIALIZED.to_le_bytes());
        assert_eq!(session.is_enabled().data, [0]);
        session.set_initialized(true);
        assert_eq!(session.state().data, STATE_INITIALIZED.to_le_bytes());
        session.set_enabled(true);
        assert_eq!(session.is_enabled().data, [1]);
    }

    #[test]
    fn no_controller_means_unavailable_and_detection_is_refused() {
        let session = NfcSession::new(Kind::Mifare);
        assert_eq!(
            Reply::ok().u32(DEVICE_STATE_UNAVAILABLE).data,
            5u32.to_le_bytes()
        );
        assert_eq!(session.refuse_detection().result, INVALID_ARGUMENT);
    }
}
