//! `ngc:u` — `nn::ngc::detail::IService` (content/bad-word filtering).
//!
//! Switchbrew's "NGC services" page documents the command list: 0
//! `GetContentVersion`, 1 `Check`, 2 `Mask`, 3 `Reload` and, on 21.0.0+, 4/5
//! `Check2`/`Mask2`. Only the version query has an unambiguous reply (`u32`); the
//! filtering commands exchange opaque content packs this runtime does not ship,
//! so they answer with the documented unknown-command error instead of an empty
//! "clean" verdict.
//!
//! All four measured titles resolve `ngc:u` during SDK start-up; 010015100b514000, 0100965017338000
//! and 0100c2500fc20000 do so without calling a command in the first 25 seconds.

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

/// Version of the built-in content filter. The runtime does not filter, so 0 is
/// "no content pack loaded"; the SDK accepts it and skips the check.
const CONTENT_VERSION: u32 = 0;

pub struct Ngc;

impl Service for Ngc {
    fn name(&self) -> String {
        "ngc:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u32(CONTENT_VERSION),
            _ => unimplemented_named("ngc:u", ngc_command_name(req.cmd), c, req),
        }
    }
}

fn ngc_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "GetContentVersion",
        1 => "Check",
        2 => "Mask",
        3 => "Reload",
        4 => "Check2",
        5 => "Mask2",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_content_version_is_zero_and_the_commands_are_named() {
        assert_eq!(CONTENT_VERSION, 0);
        assert_eq!(ngc_command_name(0), "GetContentVersion");
        assert_eq!(ngc_command_name(1), "Check");
        assert_eq!(ngc_command_name(99), "unknown");
    }
}
