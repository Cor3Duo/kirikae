use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

use super::state::RESULT_NOT_AVAILABLE;
use super::storage::Storage;

const APPLET_CONTROLLER: u32 = 0xc;
const NPAD_HANDHELD: u32 = 0x20;

pub(super) struct LibraryAppletCreator;

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
            20 | 25 | 30 | 50 | 60 | 100 | 150 => crate::services::stub::plain_ok(),
            101 => match self.output.lock().unwrap().take() {
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
