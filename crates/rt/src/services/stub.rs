use crate::cpu::Cpu;
use crate::services::{Reply, Request, Service, result};

const STUB_OUTPUT: usize = 0x20;

fn strict() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KIRIKAE_STRICT_IPC").is_some())
}

pub fn unimplemented(service: &str, c: &mut Cpu, req: &Request) -> Reply {
    crate::usage::mark_stub("unimplemented");
    let p = c.process();
    p.warn_once(format!(
        "{service} command {} is stubbed (in: {} bytes, A={} B={} X={} C={}, called from {})",
        req.cmd,
        req.data.len(),
        req.a.len(),
        req.b.len(),
        req.x.len(),
        req.c.len(),
        p.describe(c.x[30])
    ));
    if strict() {
        return Reply::err(result::CMIF_UNKNOWN_COMMAND);
    }
    Reply::ok().bytes(&[0; STUB_OUTPUT])
}

pub fn unimplemented_named(service: &str, command: &str, c: &mut Cpu, req: &Request) -> Reply {
    crate::usage::mark_stub("unimplemented");
    let p = c.process();
    p.warn_once(format!(
        "{service} command {} ({command}) is not implemented (in: {} bytes, A={} B={} X={} C={}, called from {})",
        req.cmd,
        req.data.len(),
        req.a.len(),
        req.b.len(),
        req.x.len(),
        req.c.len(),
        p.describe(c.x[30])
    ));
    Reply::err(result::CMIF_UNKNOWN_COMMAND)
}

pub fn plain_ok() -> Reply {
    crate::usage::mark_stub("plain ok");
    Reply::ok()
}

pub struct Stub {
    name: String,
}

impl Stub {
    pub fn new(name: &str) -> Stub {
        Stub {
            name: name.to_string(),
        }
    }
}

impl Service for Stub {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let r = unimplemented(&self.name, c, req);
        crate::usage::mark_stub("service stubbed");
        r
    }
}
