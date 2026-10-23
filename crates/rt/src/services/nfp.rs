use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const DEVICE: u64 = 0;
const NPAD_NO1: u32 = 0;

const STATE_NONE: u32 = 0;
const STATE_INITIALIZED: u32 = 1;
const DEVICE_INITIALIZED: u32 = 0;
const DEVICE_SEARCHING: u32 = 1;
const APPLICATION_AREA_SIZE: u32 = 0xd8;

const RESULT_DEVICE_NOT_FOUND: u32 = (64 << 9) | 115;

pub struct Nfp;

impl Service for Nfp {
    fn name(&self) -> String {
        "nfp:user".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(User::default())),
            _ => unimplemented("nfp:user", c, req),
        }
    }
}

#[derive(Default)]
struct User {
    state: AtomicU32,
    device_state: AtomicU32,
    events: Mutex<[Option<u32>; 3]>,
}

impl User {
    fn event(&self, c: &Cpu, which: usize) -> u32 {
        *self.events.lock().unwrap()[which].get_or_insert_with(|| c.process().create_event(false))
    }
}

impl Service for User {
    fn name(&self) -> String {
        "nfp:user/IUser".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        if matches!(req.cmd, 3..=18 | 20..=22 | 24) && req.u64(0) != DEVICE {
            return Reply::err(RESULT_DEVICE_NOT_FOUND);
        }
        match req.cmd {
            0 => {
                self.state.store(STATE_INITIALIZED, Relaxed);
                self.device_state.store(DEVICE_INITIALIZED, Relaxed);
                Reply::ok()
            }
            1 => {
                self.state.store(STATE_NONE, Relaxed);
                Reply::ok()
            }
            2 => {
                if let Some(out) = req.out_buf(0).filter(|b| b.size >= 8) {
                    c.process().memory.write(out.addr, &DEVICE.to_le_bytes());
                    return Reply::ok().u32(1);
                }
                Reply::ok().u32(0)
            }
            3 => {
                self.device_state.store(DEVICE_SEARCHING, Relaxed);
                Reply::ok()
            }
            4 => {
                self.device_state.store(DEVICE_INITIALIZED, Relaxed);
                Reply::ok()
            }
            17 => Reply::ok().copy_handle(self.event(c, 0)),
            18 => Reply::ok().copy_handle(self.event(c, 1)),
            19 => Reply::ok().u32(self.state.load(Relaxed)),
            20 => Reply::ok().u32(self.device_state.load(Relaxed)),
            21 => Reply::ok().u32(NPAD_NO1),
            22 => Reply::ok().u32(APPLICATION_AREA_SIZE),
            23 => Reply::ok().copy_handle(self.event(c, 2)),
            _ => unimplemented("nfp:user/IUser", c, req),
        }
    }
}
