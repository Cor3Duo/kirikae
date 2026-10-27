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
            1 => Reply::ok().u32(0),
            2 => mask(c, req),
            _ => unimplemented_named("ngc:u", ngc_command_name(req.cmd), c, req),
        }
    }
}

fn mask(c: &Cpu, req: &Request) -> Reply {
    if let (Some(input), Some(output)) = (req.in_buf(0), req.out_buf(0)) {
        let n = input.size.min(output.size);
        if let Some(text) = c.process().memory.read(input.addr, n) {
            c.process().memory.write(output.addr, &text);
        }
    }
    Reply::ok().u32(0)
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

    static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

    extern "C" fn check_and_mask(c: &mut Cpu) {
        let (input, output) = (0x1_2000_0000u64, 0x1_2000_1000u64);
        c.process().memory.map_as(input, 0x2000, crate::mem::state::NORMAL, crate::mem::perm::RW).unwrap();
        c.process().memory.write(input, b"hello");
        let check = Ngc.call(c, &Request { cmd: 1, x: vec![crate::services::Buf { addr: input, size: 5 }], ..Default::default() });
        let masked = Ngc.call(
            c,
            &Request {
                cmd: 2,
                x: vec![crate::services::Buf { addr: input, size: 5 }],
                b: vec![crate::services::Buf { addr: output, size: 0x100 }],
                ..Default::default()
            },
        );
        let copied = c.process().memory.read(output, 5).unwrap() == b"hello";
        SEEN.lock().unwrap().extend([u64::from(check.result), u64::from(check.data == [0; 4]), u64::from(masked.result), u64::from(copied)]);
        c.halt();
    }

    #[test]
    fn check_finds_nothing_and_mask_returns_the_text_unchanged() {
        let entries: &'static [crate::abi::Entry] = Box::leak(vec![crate::abi::Entry { addr: 0x1000_0000, func: check_and_mask }].into_boxed_slice());
        let exe = kirikae_loader::Executable { entry: entries[0].addr, ..Default::default() };
        let result = crate::process::run_executable(&exe, entries, crate::process::RunOptions { capture_output: true }).expect("the fake guest runs");
        assert_eq!(result.exit_code, 0);
        assert_eq!(*SEEN.lock().unwrap(), vec![0, 1, 0, 1]);
    }
}
