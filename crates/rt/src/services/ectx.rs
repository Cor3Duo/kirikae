//! `ectx` — error context services.
//!
//! `ectx:r` is the only interface with a public command layout (libnx
//! `ectxrPullContext`): command 1 `PullContext(descriptor, result)` returns
//! `(s32 context_id, u32 total_size, u32 size)` plus an output buffer. The
//! runtime does not persist error reports, so a pull succeeds with no context
//! (`0, 0, 0`), which is what the SDK stores when no report exists for the pair.
//!
//! `ectx:aw` / `ectx:w` are writer/wake-up interfaces with no public command
//! list; they answer with the documented CMIF unknown-command result and log the
//! command name once instead of pretending the write succeeded.

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
            // ectx:r PullContext: no stored context for this descriptor/result.
            ("ectx:r", 1) => Reply::ok().u32(0).u32(0).u32(0),
            _ => unimplemented_named(&self.name, ectx_command_name(req.cmd), c, req),
        }
    }
}

fn ectx_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "unknown",
        1 => "PullContext",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
