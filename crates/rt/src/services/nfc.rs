use std::sync::Mutex;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const STATE_NON_INITIALIZED: u32 = 0;
const STATE_INITIALIZED: u32 = 1;
const DEVICE_STATE_UNAVAILABLE: u32 = 5;
const DEVICE_NOT_FOUND: u32 = (64 << 9) | 127;
const UNMAPPED: u32 = u32::MAX;

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
                enabled: true,
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

    fn remap(&self, cmd: u32) -> u32 {
        match self.kind {
            Kind::User | Kind::System => match cmd {
                0 | 400 => 0,
                1 | 401 => 1,
                2 | 402 => 2,
                3 | 403 => 3,
                100 | 500 if self.kind == Kind::System => 100,
                404 => 4,
                405 => 5,
                406 => 6,
                407 => 7,
                408 => 8,
                409 => 9,
                410 => 10,
                411 => 11,
                412 => 12,
                510 if self.kind == Kind::System => 510,
                1000 => 1000,
                1001 => 1001,
                1300 => 1300,
                1301 => 1301,
                1302 => 1302,
                _ => UNMAPPED,
            },
            Kind::Mifare => match cmd {
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
                _ => UNMAPPED,
            },
        }
    }

    fn refuse_detection(&self) -> Reply {
        Reply::err(DEVICE_NOT_FOUND)
    }
}

impl Service for NfcSession {
    fn name(&self) -> String {
        "nfc/ISession".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.remap(req.cmd) {
            0 => self.set_initialized(true),
            1 => self.set_initialized(false),
            2 => self.state(),
            3 => self.is_enabled(),
            100 => self.set_enabled(req.data.first().is_some_and(|&b| b != 0)),
            4 => Reply::ok().u32(0),
            5 => Reply::ok().u32(DEVICE_STATE_UNAVAILABLE),
            6 => Reply::err(DEVICE_NOT_FOUND),
            7 => event(c, "nfc.availability"),
            8 | 9 => self.refuse_detection(),
            10 => Reply::err(DEVICE_NOT_FOUND),
            11 => event(c, "nfc.activate"),
            12 => event(c, "nfc.deactivate"),
            510 => Reply::ok(),
            1000 | 1001 => Reply::err(DEVICE_NOT_FOUND),
            1300 => Reply::err(DEVICE_NOT_FOUND),
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
    use crate::process::{RunOptions, run_executable};

    static OPENED: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());

    extern "C" fn open_all(c: &mut Cpu) {
        for name in ["nfc:user", "nfc:sys", "nfc:mf:u"] {
            let service = crate::services::open(name);
            let reply = service.call(c, &Request { cmd: 0, ..Default::default() });
            OPENED.lock().unwrap().push((service.name(), reply.objects.len()));
        }
        c.halt();
    }

    #[test]
    fn every_port_opens_a_session() {
        let entries: &'static [Entry] = Box::leak(vec![Entry { addr: 0x1000_0000, func: open_all }].into_boxed_slice());
        let exe = kirikae_loader::Executable {
            entry: entries[0].addr,
            ..Default::default()
        };
        let result = run_executable(&exe, entries, RunOptions { capture_output: true }).expect("the fake guest runs");
        assert_eq!(result.exit_code, 0);
        let opened = OPENED.lock().unwrap().clone();
        assert_eq!(opened, vec![("nfc:user".to_string(), 1), ("nfc:sys".to_string(), 1), ("nfc:mf:u".to_string(), 1)]);
    }

    fn u32_of(reply: &Reply) -> u32 {
        u32::from_le_bytes(reply.data[..4].try_into().unwrap())
    }

    #[test]
    fn a_session_reports_its_state_and_switch() {
        let session = NfcSession::new(Kind::User);
        assert_eq!(u32_of(&session.state()), STATE_NON_INITIALIZED);
        session.set_initialized(true);
        assert_eq!(u32_of(&session.state()), STATE_INITIALIZED);
        session.set_initialized(false);
        assert_eq!(u32_of(&session.state()), STATE_NON_INITIALIZED);
        assert_eq!(session.is_enabled().data, vec![1], "NFC is on by default on a real console");
        session.set_enabled(false);
        assert_eq!(session.is_enabled().data, vec![0]);
    }

    #[test]
    fn detection_is_refused_with_the_documented_error() {
        let session = NfcSession::new(Kind::Mifare);
        session.set_initialized(true);
        assert_eq!(session.refuse_detection().result, DEVICE_NOT_FOUND);
        assert_eq!(DEVICE_NOT_FOUND, 0x807f);
    }

    #[test]
    fn each_interface_only_answers_its_own_commands() {
        let user = NfcSession::new(Kind::User);
        let system = NfcSession::new(Kind::System);
        assert_eq!(user.remap(405), 5);
        assert_eq!(user.remap(5), UNMAPPED, "IUser has no command 5");
        assert_eq!(user.remap(100), UNMAPPED, "SetNfcEnabled is a system command");
        assert_eq!(user.remap(510), UNMAPPED, "OutputTestWave is a system command");
        assert_eq!(system.remap(510), 510);
        assert_eq!(system.remap(500), 100);
        assert_eq!(NfcSession::new(Kind::Mifare).remap(2), 4);
        assert_eq!(NfcSession::new(Kind::Mifare).remap(99), UNMAPPED);
    }
}
