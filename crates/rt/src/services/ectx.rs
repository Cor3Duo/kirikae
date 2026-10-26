use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

pub struct Ectx {
    name: String,
}

impl Ectx {
    pub fn new(name: &str) -> Ectx {
        Ectx {
            name: name.to_string(),
        }
    }
}

impl Service for Ectx {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match (self.name.as_str(), req.cmd) {
            ("ectx:r", 1) => Reply::ok().u32(0).u32(0).u32(0),
            ("ectx:aw", 0) => Reply::ok().object(std::sync::Arc::new(ContextRegistrar)),
            _ => unimplemented_named(&self.name, ectx_command_name(req.cmd), c, req),
        }
    }
}

struct ContextRegistrar;

impl Service for ContextRegistrar {
    fn name(&self) -> String {
        "ectx:aw/IContextRegistrar".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u32(0),
            _ => unimplemented_named("ectx:aw/IContextRegistrar", "unknown", c, req),
        }
    }
}

fn ectx_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "CreateContextRegistrar",
        1 => "PullContext",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ectx_ports_are_opened_from_the_registry() {
        let pulled = crate::services::harness::call_service("ectx:r", None, Request { cmd: 1, ..Default::default() });
        assert_eq!((pulled.result, pulled.data), (0, vec![0; 12]));
        let registrar = crate::services::harness::call_service("ectx:aw", None, Request { cmd: 0, ..Default::default() });
        assert_eq!((registrar.result, registrar.objects), (0, 1));
    }

    extern "C" fn complete(c: &mut Cpu) {
        let r = ContextRegistrar.call(c, &Request { cmd: 0, ..Default::default() });
        crate::services::harness::record("ectx/complete", &[u64::from(r.result), r.data.len() as u64]);
        c.halt();
    }

    #[test]
    fn the_registrar_completes_a_context() {
        crate::services::harness::run(vec![crate::abi::Entry { addr: crate::services::harness::ENTRY, func: complete }]);
        assert_eq!(crate::services::harness::observed("ectx/complete"), vec![0, 4]);
    }
    use crate::abi::Entry;
    use crate::services::harness;

    #[test]
    fn pull_context_returns_the_documented_header() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = Ectx::new("ectx:r").call(
                c,
                &Request {
                    cmd: 1,
                    data: 1u32
                        .to_le_bytes()
                        .into_iter()
                        .chain(2u32.to_le_bytes())
                        .collect(),
                    ..Default::default()
                },
            );
            harness::record("ectx/pull", &[u64::from(reply.result), reply.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("ectx/pull"), vec![0, 12]);
    }

    #[test]
    fn pull_context_returns_a_zeroed_context_header() {
        let reply = Reply::ok().u32(0).u32(0).u32(0);
        assert_eq!(reply.result, 0);
        assert_eq!(reply.data, [0u8; 12]);
    }
}
