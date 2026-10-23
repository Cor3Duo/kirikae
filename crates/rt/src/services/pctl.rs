use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::{unimplemented, Stub};
use crate::services::{Reply, Request, Service};

pub struct Pctl {
    pub name: &'static str,
}

impl Service for Pctl {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 | 1 => Reply::ok().object(Arc::new(Stub::new("pctl/IParentalControlService"))),
            _ => unimplemented(self.name, c, req),
        }
    }
}
