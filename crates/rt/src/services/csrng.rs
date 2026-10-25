use crate::cpu::Cpu;
use crate::mem::GuestMemory;
use crate::services::stub::unimplemented;
use crate::services::{Buf, Reply, Request, Service};

const SPL_MODULE: u32 = 26;
const ENTROPY_UNAVAILABLE: u32 = (1 << 9) | SPL_MODULE;

const CHUNK: usize = 0x1000;

pub struct Csrng;

impl Service for Csrng {
    fn name(&self) -> String {
        "csrng".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let Some(buf) = req.out_buf(0) else {
                    c.process()
                        .warn_once("csrng: GenerateRandomBytes without an output buffer".into());
                    return Reply::ok();
                };
                if fill(&c.process().memory, buf, os_entropy) {
                    Reply::ok()
                } else {
                    c.process().warn_once(
                        "csrng: no host entropy source; random bytes are unavailable".into(),
                    );
                    Reply::err(ENTROPY_UNAVAILABLE)
                }
            }
            _ => unimplemented("csrng", c, req),
        }
    }
}

fn fill(memory: &GuestMemory, buf: Buf, mut source: impl FnMut(&mut [u8]) -> bool) -> bool {
    let mut chunk = [0u8; CHUNK];
    let mut done = 0u64;
    while done < buf.size {
        let n = (buf.size - done).min(CHUNK as u64) as usize;
        if !source(&mut chunk[..n]) {
            return false;
        }
        memory.write(buf.addr + done, &chunk[..n]);
        done += n as u64;
    }
    true
}

#[cfg(unix)]
fn os_entropy(bytes: &mut [u8]) -> bool {
    use std::io::Read;

    let Ok(mut file) = std::fs::File::open("/dev/urandom") else {
        return false;
    };
    file.read_exact(bytes).is_ok()
}

#[cfg(windows)]
fn os_entropy(bytes: &mut [u8]) -> bool {
    use std::ffi::{OsStr, c_char, c_void};
    use std::os::windows::ffi::OsStrExt;
    use std::sync::OnceLock;

    type GenRandom = unsafe extern "system" fn(*mut u8, u32) -> u8;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    static ENTRY: OnceLock<Option<GenRandom>> = OnceLock::new();
    let entry = *ENTRY.get_or_init(|| {
        for dll in ["bcryptprimitives.dll", "advapi32.dll"] {
            let name: Vec<u16> = OsStr::new(dll).encode_wide().chain(Some(0)).collect();
            // SAFETY: `name` is a NUL-terminated UTF-16 string that outlives the
            // call and `LoadLibraryW` only reads it.
            let module = unsafe { LoadLibraryW(name.as_ptr()) };
            if module.is_null() {
                continue;
            }
            // SAFETY: `module` is the HMODULE returned above and the symbol name
            // is NUL-terminated.
            let proc = unsafe { GetProcAddress(module, c"SystemFunction036".as_ptr()) };
            if proc.is_null() {
                continue;
            }
            // SAFETY: every documented export of `SystemFunction036` is
            // `BOOLEAN WINAPI SystemFunction036(PVOID, ULONG)`, which is this
            // signature. The function never stores a Rust non-trivial value.
            return Some(unsafe { std::mem::transmute::<*mut c_void, GenRandom>(proc) });
        }
        None
    });
    let Some(entry) = entry else {
        return false;
    };
    let mut done = 0;
    while done < bytes.len() {
        let n = (bytes.len() - done).min(u32::MAX as usize) as u32;
        // SAFETY: the pointer ranges over `n` writable bytes of `bytes`, which is
        // what the function fills. A zero return means the call failed.
        if unsafe { entry(bytes[done..].as_mut_ptr(), n) } == 0 {
            return false;
        }
        done += n as usize;
    }
    true
}

#[cfg(not(any(unix, windows)))]
fn os_entropy(_bytes: &mut [u8]) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(len: u64) -> GuestMemory {
        let m = GuestMemory::new().unwrap();
        m.map(0x1000_0000, len.next_multiple_of(0x1000)).unwrap();
        m
    }

    #[test]
    fn generate_random_bytes_fills_the_output_buffer_without_a_return_value() {
        let m = memory(0x2000);
        let buf = Buf {
            addr: 0x1000_0000,
            size: 0x2000,
        };
        assert!(fill(&m, buf, os_entropy));
        let bytes = m.read(buf.addr, buf.size).unwrap();
        assert_eq!(bytes.len(), buf.size as usize);
        assert!(
            bytes[..CHUNK].iter().any(|&b| b != 0),
            "the first chunk must not be left as zeros"
        );
        assert!(
            bytes[CHUNK..].iter().any(|&b| b != 0),
            "chunks past the first one are filled too"
        );
    }

    #[test]
    fn two_requests_return_different_bytes() {
        let m = memory(0x40);
        let buf = Buf {
            addr: 0x1000_0000,
            size: 0x40,
        };
        assert!(fill(&m, buf, os_entropy));
        let first = m.read(buf.addr, buf.size).unwrap();
        assert!(fill(&m, buf, os_entropy));
        let second = m.read(buf.addr, buf.size).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn a_failing_source_stops_and_is_reported() {
        let m = memory(0x2000);
        let buf = Buf {
            addr: 0x1000_0000,
            size: 0x2000,
        };
        let mut calls = 0;
        let ok = fill(&m, buf, |out| {
            calls += 1;
            if calls == 2 {
                return false;
            }
            out.fill(0xab);
            true
        });
        assert!(!ok);
        assert_eq!(calls, 2);
        let bytes = m.read(buf.addr, buf.size).unwrap();
        assert_eq!(&bytes[..CHUNK], &[0xab; CHUNK][..]);
        assert_eq!(&bytes[CHUNK..], &vec![0u8; CHUNK][..]);
    }

    #[test]
    fn a_zero_sized_buffer_needs_no_entropy() {
        let m = memory(0x1000);
        let buf = Buf {
            addr: 0x1000_0000,
            size: 0,
        };
        assert!(fill(&m, buf, |_| false));
    }

    #[test]
    fn the_entropy_error_is_an_spl_result() {
        assert_eq!(ENTROPY_UNAVAILABLE & 0x1ff, SPL_MODULE);
        assert_ne!(ENTROPY_UNAVAILABLE, 0);
    }
}
