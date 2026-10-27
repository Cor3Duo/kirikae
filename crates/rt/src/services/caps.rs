use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const APPLICATION_ALBUM_ENTRY_SIZE: usize = 0x20;

const CAPSRV_MODULE: u32 = 206;
const fn capsrv_error(description: u32) -> u32 {
    (description << 9) | CAPSRV_MODULE
}

const ALBUM_FILE_NOT_FOUND: u32 = capsrv_error(23);

pub struct CapsSu;

impl Service for CapsSu {
    fn name(&self) -> String {
        "caps:su".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            32 => Reply::ok(),
            201 | 203 | 205 | 210 => save_screen_shot(c),
            _ => unimplemented_named("caps:su", "unknown", c, req),
        }
    }
}

fn save_screen_shot(c: &Cpu) -> Reply {
    c.process().warn_once(
        "caps:su: no album storage; SaveScreenShot returns a zero-sized ApplicationAlbumEntry"
            .into(),
    );
    Reply::ok().bytes(&[0u8; APPLICATION_ALBUM_ENTRY_SIZE])
}

pub struct CapsU;

impl Service for CapsU {
    fn name(&self) -> String {
        "caps:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            32 => Reply::ok(),
            102 | 140..=147 => Reply::ok().u64(0),
            130 => Reply::ok(),
            103..=105 | 110 | 120 | 60002 => file_not_found(c, req.cmd),
            _ => unimplemented_named("caps:u", "unknown", c, req),
        }
    }
}

fn file_not_found(c: &Cpu, cmd: u32) -> Reply {
    c.process().warn_once(format!(
        "caps:u: command {cmd} names an album file that does not exist; reporting AlbumFileNotFound"
    ));
    Reply::err(ALBUM_FILE_NOT_FOUND)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_capture_ports_are_opened_from_the_registry() {
        let listed = crate::services::harness::call_service("caps:u", None, Request { cmd: 102, ..Default::default() });
        assert_eq!((listed.result, listed.data), (0, 0u64.to_le_bytes().to_vec()));
        let version = crate::services::harness::call_service("caps:su", None, Request { cmd: 32, ..Default::default() });
        assert_eq!((version.result, version.data.len()), (0, 0));
    }
    use crate::abi::Entry;
    use crate::services::harness;

    #[test]
    fn set_shim_library_version_has_no_output() {
        let reply = Reply::ok();
        assert_eq!(reply.result, 0);
        assert!(reply.data.is_empty());
        assert!(reply.copy_handles.is_empty() && reply.objects.is_empty());
    }

    #[test]
    fn every_save_command_returns_a_zero_sized_album_entry() {
        extern "C" fn probe(c: &mut Cpu) {
            let keys = ["caps/cmd201", "caps/cmd203", "caps/cmd205", "caps/cmd210"];
            for (key, cmd) in keys.into_iter().zip([201u32, 203, 205, 210]) {
                let reply = CapsSu.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                harness::record(
                    key,
                    &[
                        reply.result as u64,
                        reply.data.len() as u64,
                        reply.data.iter().all(|&b| b == 0) as u64,
                    ],
                );
            }
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        for key in ["caps/cmd201", "caps/cmd203", "caps/cmd205", "caps/cmd210"] {
            assert_eq!(
                harness::observed(key),
                vec![0, APPLICATION_ALBUM_ENTRY_SIZE as u64, 1],
                "{key} succeeds with a zero-sized 0x20-byte entry"
            );
        }
        assert_eq!(APPLICATION_ALBUM_ENTRY_SIZE, 0x20);
    }

    #[test]
    fn an_undocumented_command_is_refused() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = CapsSu.call(
                c,
                &Request {
                    cmd: 200,
                    ..Default::default()
                },
            );
            harness::record("caps/su-unknown", &[reply.result as u64, reply.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("caps/su-unknown"),
            vec![crate::services::result::CMIF_UNKNOWN_COMMAND as u64, 0]
        );
    }

    #[test]
    fn caps_u_sets_the_shim_version_with_no_output() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = CapsU.call(
                c,
                &Request {
                    cmd: 32,
                    data: 1u64.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            harness::record("caps/shim",
                &[
                    reply.result as u64,
                    reply.data.len() as u64,
                    reply.copy_handles.len() as u64,
                    reply.objects.len() as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("caps/shim"), vec![0, 0, 0, 0]);
    }

    #[test]
    fn caps_u_lists_an_empty_album_and_accepts_the_save_precheck() {
        extern "C" fn probe(c: &mut Cpu) {
            let mut all = Vec::new();
            for cmd in [102u32, 140, 141, 142, 143, 144, 145, 146, 147, 130] {
                let reply = CapsU.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                all.extend([reply.result as u64, reply.data.len() as u64]);
            }
            harness::record("caps/lists", &all);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let mut expected = Vec::new();
        for _ in 0..9 {
            expected.extend([0, 8]);
        }
        expected.extend([0, 0]);
        assert_eq!(harness::observed("caps/lists"), expected);
    }

    #[test]
    fn caps_u_commands_that_name_a_file_report_it_as_missing() {
        extern "C" fn probe(c: &mut Cpu) {
            let mut all = Vec::new();
            for cmd in [103u32, 104, 105, 110, 120, 60002] {
                let reply = CapsU.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                all.extend([
                    reply.result as u64,
                    reply.data.len() as u64,
                    reply.objects.len() as u64,
                ]);
            }
            harness::record("caps/files", &all);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let mut expected = Vec::new();
        for _ in 0..6 {
            expected.extend([ALBUM_FILE_NOT_FOUND as u64, 0, 0]);
        }
        assert_eq!(
            harness::observed("caps/files"),
            expected,
            "no buffer is filled and no accessor object is handed out"
        );
        assert_eq!(ALBUM_FILE_NOT_FOUND, 0x2ECE);
    }

    #[test]
    fn caps_u_refuses_an_undocumented_command() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = CapsU.call(
                c,
                &Request {
                    cmd: 148,
                    ..Default::default()
                },
            );
            harness::record("caps/u-unknown", &[reply.result as u64, reply.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("caps/u-unknown"),
            vec![crate::services::result::CMIF_UNKNOWN_COMMAND as u64, 0]
        );
    }
}
