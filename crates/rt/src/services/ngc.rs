use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

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
