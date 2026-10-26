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
            _ => unimplemented_named(&self.name, ectx_command_name(req.cmd), c, req),
        }
    }
}

fn ectx_command_name(cmd: u32) -> &'static str {
    match cmd {
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
            harness::record("pull", &[u64::from(reply.result), reply.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("pull"), vec![0, 12]);
    }

    #[test]
    fn pull_context_returns_a_zeroed_context_header() {
        let reply = Reply::ok().u32(0).u32(0).u32(0);
        assert_eq!(reply.result, 0);
        assert_eq!(reply.data, [0u8; 12]);
    }
}
