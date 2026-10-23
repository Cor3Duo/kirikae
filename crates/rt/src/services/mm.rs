use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const OLD: u32 = 1 << 31;

#[derive(Default)]
struct State {
    next_id: AtomicU32,
    settings: Mutex<HashMap<u32, u32>>,
}

pub struct Mm;

impl Service for Mm {
    fn name(&self) -> String {
        "mm:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let st = c.process().service_state("mm:u", State::default);
        let mut settings = st.settings.lock().unwrap();
        let key = if req.cmd < 4 { req.u32(0) | OLD } else { req.u32(0) };
        match req.cmd {
            0 | 1 => Reply::ok(),
            2 | 6 => {
                settings.insert(key, req.u32(4));
                Reply::ok()
            }
            3 | 7 => Reply::ok().u32(settings.get(&key).copied().unwrap_or(0)),
            4 => {
                let id = st.next_id.fetch_add(1, Ordering::Relaxed) + 1;
                settings.insert(id, 0);
                Reply::ok().u32(id)
            }
            5 => {
                settings.remove(&req.u32(0));
                Reply::ok()
            }
            _ => {
                drop(settings);
                unimplemented("mm:u", c, req)
            }
        }
    }
}
