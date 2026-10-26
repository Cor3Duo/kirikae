use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Buf, Reply, Request, Service};

pub const SSL_NOT_AVAILABLE: u32 = 0x0021_2a0a;

const HOST_NAME_MAX: usize = 0xff;
#[cfg(test)]
const CIPHER_INFO_SIZE: usize = 0x48;
const DTLS_TIMEOUT_SIZE: usize = 8;
const OPTION_TYPES: usize = 4;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn read_guest(c: &Cpu, buf: Buf) -> Option<Vec<u8>> {
    c.process().memory.read(buf.addr, buf.size)
}

fn write_guest(c: &Cpu, buf: Buf, data: &[u8]) -> bool {
    let n = data.len().min(buf.size as usize);
    c.process().memory.write(buf.addr, &data[..n])
}

pub struct Ssl {
    name: &'static str,
    interface_version: AtomicU32,
    context_count: AtomicU32,
    core_mask: AtomicU64,
    debug_option: AtomicU32,
}

impl Ssl {
    pub fn new(name: &'static str) -> Ssl {
        Ssl {
            name,
            interface_version: AtomicU32::new(0),
            context_count: AtomicU32::new(0),
            core_mask: AtomicU64::new(0x7),
            debug_option: AtomicU32::new(0),
        }
    }

    fn set_interface_version(&self, req: &Request) -> Reply {
        self.interface_version.store(req.u32(0), Relaxed);
        Reply::ok()
    }

    fn create_context(&self) -> Reply {
        self.context_count.fetch_add(1, Relaxed);
        Reply::ok().object(Arc::new(SslContext::new()))
    }

    fn get_certificates(&self, c: &Cpu, req: &Request) -> Reply {
        let _ = (c, req);
        if self.interface_version.load(Relaxed) >= 1 {
            Reply::ok().u32(0)
        } else {
            Reply::ok()
        }
    }

    fn flush_session_cache(&self) -> Reply {
        Reply::ok().u32(0)
    }

    fn set_debug_option(&self, req: &Request) -> Reply {
        let kind = req.u32(0);
        if kind != 0 {
            return Reply::err(SSL_NOT_AVAILABLE);
        }
        let Some(buf) = req.in_buf(0) else {
            return Reply::err(SSL_NOT_AVAILABLE);
        };
        if buf.size == 0 {
            return Reply::err(SSL_NOT_AVAILABLE);
        }
        self.debug_option.store(1, Relaxed);
        Reply::ok()
    }

    fn get_debug_option(&self, c: &Cpu, req: &Request) -> Reply {
        if req.u32(0) != 0 {
            return Reply::err(SSL_NOT_AVAILABLE);
        }
        if let Some(buf) = req.out_buf(0) {
            let value = [self.debug_option.load(Relaxed) as u8];
            let _ = write_guest(c, buf, &value);
        }
        Reply::ok()
    }

    #[cfg(test)]
    fn interface_version(&self) -> u32 {
        self.interface_version.load(Relaxed)
    }
}

impl Service for Ssl {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => self.create_context(),
            1 => Reply::ok().u32(self.context_count.load(Relaxed)),
            2 => self.get_certificates(c, req),
            3 => Reply::ok().u32(0),
            4 => Reply::err(SSL_NOT_AVAILABLE),
            5 => self.set_interface_version(req),
            6 => self.flush_session_cache(),
            7 => self.set_debug_option(req),
            8 => self.get_debug_option(c, req),
            9 => Reply::ok(),
            10 => Reply::err(SSL_NOT_AVAILABLE),
            11 => Reply::ok().u32(0),
            100 => self.create_context(),
            101 => {
                self.core_mask.store(req.u64(0), Relaxed);
                Reply::ok()
            }
            102 => Reply::ok().u64(self.core_mask.load(Relaxed)),
            103 => Reply::err(SSL_NOT_AVAILABLE),
            104 | 105 => Reply::err(SSL_NOT_AVAILABLE),
            _ => unimplemented_named(self.name, "unknown ssl command", c, req),
        }
    }
}

struct SslContext {
    crl_import_date_check: AtomicU32,
    connections: AtomicU32,
}

impl SslContext {
    fn new() -> SslContext {
        SslContext {
            crl_import_date_check: AtomicU32::new(1),
            connections: AtomicU32::new(0),
        }
    }

    fn set_option(&self, req: &Request) -> Reply {
        match req.u32(0) {
            1 => {
                self.crl_import_date_check.store(req.u32(4), Relaxed);
                Reply::ok()
            }
            0 => Reply::ok(),
            _ => Reply::err(SSL_NOT_AVAILABLE),
        }
    }
}

impl Service for SslContext {
    fn name(&self) -> String {
        "ssl/ISslContext".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => self.set_option(req),
            1 => {
                let value = if req.u32(0) == 1 {
                    self.crl_import_date_check.load(Relaxed)
                } else {
                    0
                };
                Reply::ok().u32(value)
            }
            2 => {
                self.connections.fetch_add(1, Relaxed);
                Reply::ok().object(Arc::new(SslConnection::new()))
            }
            3 => Reply::ok().u32(self.connections.load(Relaxed)),
            4 | 5 | 8 | 10 | 12 | 13 => Reply::err(SSL_NOT_AVAILABLE),
            6 | 7 | 11 => Reply::ok(),
            9 => Reply::ok(),
            _ => unimplemented_named("ssl/ISslContext", "unknown ssl context command", c, req),
        }
    }
}

struct SslConnection {
    socket: AtomicI32,
    host_name: Mutex<Vec<u8>>,
    verify: AtomicU32,
    io_mode: AtomicU32,
    io_timeout: AtomicU32,
    session_cache: AtomicU32,
    renegotiation: AtomicU32,
    options: Mutex<[bool; OPTION_TYPES]>,
    alpn: Mutex<Vec<u8>>,
}

impl SslConnection {
    fn new() -> SslConnection {
        SslConnection {
            socket: AtomicI32::new(-1),
            host_name: Mutex::new(Vec::new()),
            verify: AtomicU32::new(0x3),
            io_mode: AtomicU32::new(1),
            io_timeout: AtomicU32::new(0),
            session_cache: AtomicU32::new(0),
            renegotiation: AtomicU32::new(0),
            options: Mutex::new([false; OPTION_TYPES]),
            alpn: Mutex::new(Vec::new()),
        }
    }

    fn set_host_name(&self, c: &Cpu, req: &Request) -> Reply {
        let Some(buf) = req.in_buf(0) else {
            return Reply::err(SSL_NOT_AVAILABLE);
        };
        let Some(bytes) = read_guest(c, buf) else {
            return Reply::err(SSL_NOT_AVAILABLE);
        };
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let mut name = bytes[..end.min(HOST_NAME_MAX)].to_vec();
        name.truncate(HOST_NAME_MAX);
        *lock(&self.host_name) = name;
        Reply::ok()
    }

    fn get_host_name(&self, c: &Cpu, req: &Request) -> Reply {
        let name = lock(&self.host_name).clone();
        if let Some(buf) = req.out_buf(0) {
            let _ = write_guest(c, buf, &name);
        }
        Reply::ok().u32(name.len() as u32)
    }

    fn set_alpn(&self, c: &Cpu, req: &Request) -> Reply {
        if let Some(buf) = req.in_buf(0)
            && let Some(bytes) = read_guest(c, buf)
        {
            *lock(&self.alpn) = bytes;
        }
        Reply::ok()
    }

    fn get_alpn(&self, c: &Cpu, req: &Request) -> Reply {
        if let Some(buf) = req.out_buf(0) {
            let _ = write_guest(c, buf, &[]);
        }
        Reply::ok().u32(0).u32(0)
    }

    fn set_option(&self, req: &Request) -> Reply {
        let value = req.data.first().copied().unwrap_or(0) != 0;
        let kind = req.u32(4) as usize;
        if kind >= OPTION_TYPES {
            return Reply::err(SSL_NOT_AVAILABLE);
        }
        lock(&self.options)[kind] = value;
        Reply::ok()
    }

    fn get_option(&self, req: &Request) -> Reply {
        let kind = req.u32(0) as usize;
        if kind >= OPTION_TYPES {
            return Reply::err(SSL_NOT_AVAILABLE);
        }
        Reply::ok().u8(u8::from(lock(&self.options)[kind]))
    }
}

impl Service for SslConnection {
    fn name(&self) -> String {
        "ssl/ISslConnection".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let fd = req.u32(0) as i32;
                self.socket.store(fd, Relaxed);
                Reply::ok().u32(fd as u32)
            }
            1 => self.set_host_name(c, req),
            2 => {
                self.verify.store(req.u32(0), Relaxed);
                Reply::ok()
            }
            3 => {
                self.io_mode.store(req.u32(0), Relaxed);
                Reply::ok()
            }
            4 => Reply::ok().u32(self.socket.load(Relaxed) as u32),
            5 => self.get_host_name(c, req),
            6 => Reply::ok().u32(self.verify.load(Relaxed)),
            7 => Reply::ok().u32(self.io_mode.load(Relaxed)),
            8 | 9 | 10 | 11 | 13 | 14 | 25 | 31 | 32 | 33 | 36 | 37 => {
                Reply::err(SSL_NOT_AVAILABLE)
            }
            12 => Reply::ok().u32(0),
            15 => Reply::ok(),
            16 => Reply::ok().u32(0),
            17 => {
                self.session_cache.store(req.u32(0), Relaxed);
                Reply::ok()
            }
            18 => Reply::ok().u32(self.session_cache.load(Relaxed)),
            19 => Reply::ok(),
            20 => {
                self.renegotiation.store(req.u32(0), Relaxed);
                Reply::ok()
            }
            21 => Reply::ok().u32(self.renegotiation.load(Relaxed)),
            22 => self.set_option(req),
            23 => self.get_option(req),
            24 => Reply::ok().u32(0).u32(0),
            26 => self.set_alpn(c, req),
            27 => self.get_alpn(c, req),
            28 => {
                let fd = req.u32(0) as i32;
                self.socket.store(fd, Relaxed);
                Reply::ok().u32(fd as u32)
            }
            29 => {
                if let Some(buf) = req.out_buf(0) {
                    let _ = write_guest(c, buf, &[0u8; DTLS_TIMEOUT_SIZE]);
                }
                Reply::ok()
            }
            30 => Reply::ok(),
            34 => {
                self.io_timeout.store(req.u32(0), Relaxed);
                Reply::ok()
            }
            35 => Reply::ok().u32(self.io_timeout.load(Relaxed)),
            _ => unimplemented_named(
                "ssl/ISslConnection",
                "unknown ssl connection command",
                c,
                req,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::process::{RunOptions, run_executable};

    static OBSERVED: Mutex<Vec<(&'static str, Vec<u64>)>> = Mutex::new(Vec::new());

    fn record(name: &'static str, values: &[u64]) {
        OBSERVED.lock().unwrap().push((name, values.to_vec()));
    }

    fn observed(name: &'static str) -> Vec<u64> {
        let mut all = OBSERVED.lock().unwrap();
        let i = all
            .iter()
            .position(|(n, _)| *n == name)
            .expect("the fake guest recorded nothing");
        all.remove(i).1
    }

    fn run(probe: extern "C" fn(&mut Cpu)) {
        let entries: &'static [Entry] = Box::leak(
            vec![Entry {
                addr: 0x1000_0000,
                func: probe,
            }]
            .into_boxed_slice(),
        );
        let exe = kirikae_loader::Executable {
            modules: Vec::new(),
            entry: entries[0].addr,
            title_id: None,
            main_stack_size: None,
            system_resource_size: 0,
            romfs: None,
        };
        let result = run_executable(
            &exe,
            entries,
            RunOptions {
                capture_output: true,
            },
        )
        .expect("the fake guest runs");
        assert_eq!(result.exit_code, 0);
    }

    fn map(c: &mut Cpu, addr: u64, size: u64) -> u64 {
        c.process()
            .memory
            .map_as(addr, size, crate::mem::state::NORMAL, crate::mem::perm::RW)
            .expect("the guest buffer maps");
        addr
    }

    fn object(reply: &Reply, index: usize) -> Arc<dyn Service> {
        Arc::clone(&reply.objects[index])
    }

    #[test]
    fn set_interface_version_has_no_output_and_is_kept() {
        let ssl = Ssl::new("ssl");
        assert_eq!(ssl.interface_version(), 0);

        let reply = ssl.set_interface_version(&Request {
            data: 2u32.to_le_bytes().to_vec(),
            ..Request::default()
        });
        assert_eq!(reply.result, 0);
        assert!(
            reply.data.is_empty(),
            "unexpected reply data {:x?}",
            reply.data
        );
        assert_eq!(ssl.interface_version(), 2);
    }

    #[test]
    fn a_truncated_version_reads_as_zero() {
        let ssl = Ssl::new("ssl");
        ssl.set_interface_version(&Request {
            data: vec![2, 0],
            ..Request::default()
        });
        assert_eq!(ssl.interface_version(), 0);
    }

    #[test]
    fn create_context_and_connection_keep_the_documented_widths() {
        extern "C" fn probe(c: &mut Cpu) {
            let ssl = Ssl::new("ssl");
            let create = ssl.call(
                c,
                &Request {
                    cmd: 0,
                    ..Request::default()
                },
            );
            let context = object(&create, 0);
            let count = ssl.call(
                c,
                &Request {
                    cmd: 1,
                    ..Request::default()
                },
            );
            let get_option = context.call(
                c,
                &Request {
                    cmd: 1,
                    data: 1u32.to_le_bytes().to_vec(),
                    ..Request::default()
                },
            );
            let connection = context.call(
                c,
                &Request {
                    cmd: 2,
                    ..Request::default()
                },
            );
            let conn = object(&connection, 0);
            let conn_count = context.call(
                c,
                &Request {
                    cmd: 3,
                    ..Request::default()
                },
            );

            let host = map(c, 0x1_6000_0000, 0x1000);
            let out = map(c, 0x1_6000_1000, 0x1000);
            c.process()
                .memory
                .write(host, b"example.com\0")
                .then_some(())
                .expect("the host name writes");
            let set_host = conn.call(
                c,
                &Request {
                    cmd: 1,
                    a: vec![Buf {
                        addr: host,
                        size: 12,
                    }],
                    ..Request::default()
                },
            );
            let get_host = conn.call(
                c,
                &Request {
                    cmd: 5,
                    b: vec![Buf {
                        addr: out,
                        size: 0x100,
                    }],
                    ..Request::default()
                },
            );
            let host_name = c.process().memory.read(out, 11).expect("the host name");

            let set_alpn = conn.call(
                c,
                &Request {
                    cmd: 26,
                    a: vec![Buf {
                        addr: host,
                        size: 12,
                    }],
                    ..Request::default()
                },
            );
            let get_alpn = conn.call(
                c,
                &Request {
                    cmd: 27,
                    b: vec![Buf {
                        addr: out,
                        size: 0x100,
                    }],
                    ..Request::default()
                },
            );
            let handshake = conn.call(
                c,
                &Request {
                    cmd: 8,
                    ..Request::default()
                },
            );
            let descriptor = conn.call(
                c,
                &Request {
                    cmd: 0,
                    data: 5i32.to_le_bytes().to_vec(),
                    ..Request::default()
                },
            );
            let get_descriptor = conn.call(
                c,
                &Request {
                    cmd: 4,
                    ..Request::default()
                },
            );
            let io_mode = conn.call(
                c,
                &Request {
                    cmd: 7,
                    ..Request::default()
                },
            );
            let verify = conn.call(
                c,
                &Request {
                    cmd: 6,
                    ..Request::default()
                },
            );

            record(
                "ssl_context",
                &[
                    create.result as u64,
                    create.objects.len() as u64,
                    u32::from_le_bytes(count.data[..4].try_into().unwrap()) as u64,
                    u32::from_le_bytes(get_option.data[..4].try_into().unwrap()) as u64,
                    connection.result as u64,
                    connection.objects.len() as u64,
                    u32::from_le_bytes(conn_count.data[..4].try_into().unwrap()) as u64,
                    set_host.result as u64,
                    u32::from_le_bytes(get_host.data[..4].try_into().unwrap()) as u64,
                    host_name
                        .iter()
                        .fold(0u64, |acc, &b| acc << 8 | u64::from(b)),
                    set_alpn.result as u64,
                    u32::from_le_bytes(get_alpn.data[..4].try_into().unwrap()) as u64,
                    u32::from_le_bytes(get_alpn.data[4..8].try_into().unwrap()) as u64,
                    handshake.result as u64,
                    u32::from_le_bytes(descriptor.data[..4].try_into().unwrap()) as u64,
                    u32::from_le_bytes(get_descriptor.data[..4].try_into().unwrap()) as u64,
                    u32::from_le_bytes(io_mode.data[..4].try_into().unwrap()) as u64,
                    u32::from_le_bytes(verify.data[..4].try_into().unwrap()) as u64,
                ],
            );
            c.halt();
        }
        run(probe);
        let mut host = 0u64;
        for &b in b"example.com" {
            host = host << 8 | u64::from(b);
        }
        assert_eq!(
            observed("ssl_context"),
            vec![
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                11,
                host,
                0,
                0,
                0,
                SSL_NOT_AVAILABLE as u64,
                5,
                5,
                1,
                3,
            ]
        );
    }

    #[test]
    fn the_offline_certificate_store_is_empty_and_tls_is_refused() {
        extern "C" fn probe(c: &mut Cpu) {
            let ssl = Ssl::new("ssl:s");
            let _ = ssl.call(
                c,
                &Request {
                    cmd: 5,
                    data: 2u32.to_le_bytes().to_vec(),
                    ..Request::default()
                },
            );
            let out = map(c, 0x1_6100_0000, 0x1000);
            let certificates = ssl.call(
                c,
                &Request {
                    cmd: 2,
                    b: vec![Buf {
                        addr: out,
                        size: 0x100,
                    }],
                    ..Request::default()
                },
            );
            let buffer_size = ssl.call(
                c,
                &Request {
                    cmd: 3,
                    ..Request::default()
                },
            );
            let trusted = ssl.call(
                c,
                &Request {
                    cmd: 11,
                    ..Request::default()
                },
            );
            let system = ssl.call(
                c,
                &Request {
                    cmd: 100,
                    ..Request::default()
                },
            );
            let core = ssl.call(
                c,
                &Request {
                    cmd: 102,
                    ..Request::default()
                },
            );
            let verify = ssl.call(
                c,
                &Request {
                    cmd: 103,
                    ..Request::default()
                },
            );

            let context = object(&system, 0);
            let import = context.call(
                c,
                &Request {
                    cmd: 4,
                    ..Request::default()
                },
            );
            let connection = context.call(
                c,
                &Request {
                    cmd: 2,
                    ..Request::default()
                },
            );
            let conn = object(&connection, 0);
            let read = conn.call(
                c,
                &Request {
                    cmd: 10,
                    b: vec![Buf {
                        addr: out,
                        size: 0x40,
                    }],
                    ..Request::default()
                },
            );
            let write = conn.call(
                c,
                &Request {
                    cmd: 11,
                    a: vec![Buf {
                        addr: out,
                        size: 0x40,
                    }],
                    ..Request::default()
                },
            );
            let cipher = conn.call(
                c,
                &Request {
                    cmd: 25,
                    data: 1u32.to_le_bytes().to_vec(),
                    b: vec![Buf {
                        addr: out,
                        size: CIPHER_INFO_SIZE as u64,
                    }],
                    ..Request::default()
                },
            );
            let pending = conn.call(
                c,
                &Request {
                    cmd: 12,
                    ..Request::default()
                },
            );
            let needed = conn.call(
                c,
                &Request {
                    cmd: 16,
                    ..Request::default()
                },
            );
            let cert_error = conn.call(
                c,
                &Request {
                    cmd: 15,
                    ..Request::default()
                },
            );
            let dtls = conn.call(
                c,
                &Request {
                    cmd: 29,
                    b: vec![Buf {
                        addr: out,
                        size: DTLS_TIMEOUT_SIZE as u64,
                    }],
                    ..Request::default()
                },
            );
            let dtls_bytes = c
                .process()
                .memory
                .read(out, DTLS_TIMEOUT_SIZE as u64)
                .expect("the DTLS timeout");

            record(
                "ssl_offline",
                &[
                    certificates.result as u64,
                    u32::from_le_bytes(certificates.data[..4].try_into().unwrap()) as u64,
                    buffer_size.result as u64,
                    u32::from_le_bytes(buffer_size.data[..4].try_into().unwrap()) as u64,
                    trusted.result as u64,
                    u32::from_le_bytes(trusted.data[..4].try_into().unwrap()) as u64,
                    system.result as u64,
                    u64::from_le_bytes(core.data[..8].try_into().unwrap()),
                    verify.result as u64,
                    import.result as u64,
                    read.result as u64,
                    write.result as u64,
                    cipher.result as u64,
                    pending.result as u64,
                    u32::from_le_bytes(pending.data[..4].try_into().unwrap()) as u64,
                    needed.result as u64,
                    u32::from_le_bytes(needed.data[..4].try_into().unwrap()) as u64,
                    cert_error.result as u64,
                    dtls.result as u64,
                    dtls_bytes.iter().map(|&b| u64::from(b)).sum::<u64>(),
                ],
            );
            c.halt();
        }
        run(probe);
        assert_eq!(
            observed("ssl_offline"),
            vec![
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0x7,
                SSL_NOT_AVAILABLE as u64,
                SSL_NOT_AVAILABLE as u64,
                SSL_NOT_AVAILABLE as u64,
                SSL_NOT_AVAILABLE as u64,
                SSL_NOT_AVAILABLE as u64,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        );
    }

    #[test]
    fn the_documented_sizes_are_the_ones_the_sysmodule_uses() {
        assert_eq!(SSL_NOT_AVAILABLE, 0x212a0a);
        assert_eq!(CIPHER_INFO_SIZE, 0x48);
        assert_eq!(DTLS_TIMEOUT_SIZE, 8);
        assert_eq!(HOST_NAME_MAX, 0xff);
    }
}
