use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const CONTENT_VERSION: u32 = 0;
pub const WORD_LIST_DATA: u64 = 0x0100_0000_0000_0823;
const EMPTY_AUTOMATON: usize = 195;
const LANGUAGES: u32 = 32;

pub fn word_list_archive() -> Vec<u8> {
    let table = gzip(&[0; EMPTY_AUTOMATON]);
    let mut files = vec![("version.dat".to_string(), CONTENT_VERSION.to_be_bytes().to_vec())];
    for set in ["common".to_string()].into_iter().chain((0..LANGUAGES).map(|l| l.to_string())) {
        for kind in ["b1", "b2", "not_b"] {
            files.push((format!("ac_{set}_{kind}_nx"), table.clone()));
        }
    }
    kirikae_loader::romfs::from_files(&files)
}

fn gzip(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
    out.extend(miniz_oxide::deflate::compress_to_vec(data, 6));
    out.extend_from_slice(&crate::usage::crc32(&[data]).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

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

    #[test]
    fn the_word_list_archive_holds_gzipped_empty_tables() {
        let path = std::env::temp_dir().join(format!("kirikae-ngc-{}.romfs", std::process::id()));
        let archive = word_list_archive();
        std::fs::write(&path, &archive).unwrap();
        let rom = kirikae_loader::romfs::RomFs { path: path.clone(), offset: 0, size: archive.len() as u64, compression: None };
        let mut r = kirikae_loader::romfs::Reader::open(&rom).unwrap();
        let mut read = |name: &str| {
            let f = kirikae_loader::romfs::lookup(&mut r, name).unwrap().unwrap_or_else(|| panic!("{name} is missing"));
            let mut data = vec![0u8; f.size as usize];
            r.read_at(f.offset, &mut data).unwrap();
            data
        };
        assert_eq!(read("/version.dat"), CONTENT_VERSION.to_be_bytes());
        for name in ["/ac_common_b1_nx", "/ac_common_b2_nx", "/ac_common_not_b_nx", "/ac_1_b1_nx", "/ac_31_not_b_nx"] {
            let gz = read(name);
            assert_eq!(gz[..3], [0x1f, 0x8b, 8], "{name}");
            let (body, trailer) = gz[10..].split_at(gz.len() - 18);
            let table = miniz_oxide::inflate::decompress_to_vec(body).unwrap();
            assert_eq!(table, [0; EMPTY_AUTOMATON], "{name}");
            assert_eq!(trailer[..4], crate::usage::crc32(&[&table]).to_le_bytes(), "{name}");
            assert_eq!(trailer[4..], (EMPTY_AUTOMATON as u32).to_le_bytes(), "{name}");
        }
        let _ = std::fs::remove_file(&path);
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
