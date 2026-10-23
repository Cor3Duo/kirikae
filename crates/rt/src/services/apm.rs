use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

pub const PERFORMANCE_MODE: u32 = 1;
const DOCKED_CONFIGURATION: u32 = 0x0002_0003;

pub struct Apm;

impl Service for Apm {
    fn name(&self) -> String {
        "apm".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(Session)),
            1 => Reply::ok().u32(PERFORMANCE_MODE),
            6 => Reply::ok().u8(0),
            _ => unimplemented("apm", c, req),
        }
    }
}

pub struct Session;

impl Service for Session {
    fn name(&self) -> String {
        "apm/ISession".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok(),
            1 => Reply::ok().u32(DOCKED_CONFIGURATION),
            2 => Reply::ok(),
            _ => unimplemented("apm/ISession", c, req),
        }
    }
}
