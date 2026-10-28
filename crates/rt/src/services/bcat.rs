use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const MODULE: u32 = 122;
const NOT_FOUND: u32 = (2 << 9) | MODULE;

pub struct Bcat {
    name: String,
}

impl Bcat {
    pub fn new(name: &str) -> Bcat {
        Bcat { name: name.to_string() }
    }
}

impl Service for Bcat {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(BcatService)),
            1 | 2 => Reply::ok().object(Arc::new(DeliveryCacheStorage)),
            _ => unimplemented_named(&self.name, creator_command_name(req.cmd), c, req),
        }
    }
}

struct BcatService;

impl Service for BcatService {
    fn name(&self) -> String {
        "bcat/IBcatService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        unimplemented_named("bcat/IBcatService", "unknown", c, req)
    }
}

struct DeliveryCacheStorage;

impl Service for DeliveryCacheStorage {
    fn name(&self) -> String {
        "bcat/IDeliveryCacheStorageService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(DeliveryCacheFile)),
            1 => Reply::ok().object(Arc::new(DeliveryCacheDirectory)),
            10 => Reply::ok().u32(0),
            _ => unimplemented_named("bcat/IDeliveryCacheStorageService", "unknown", c, req),
        }
    }
}

struct DeliveryCacheDirectory;

impl Service for DeliveryCacheDirectory {
    fn name(&self) -> String {
        "bcat/IDeliveryCacheDirectoryService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::err(NOT_FOUND),
            1 | 2 => Reply::ok().u32(0),
            _ => unimplemented_named("bcat/IDeliveryCacheDirectoryService", "unknown", c, req),
        }
    }
}

struct DeliveryCacheFile;

impl Service for DeliveryCacheFile {
    fn name(&self) -> String {
        "bcat/IDeliveryCacheFileService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::err(NOT_FOUND),
            _ => unimplemented_named("bcat/IDeliveryCacheFileService", file_command_name(req.cmd), c, req),
        }
    }
}

fn creator_command_name(cmd: u32) -> &'static str {
    match cmd {
        3 => "CreateDeliveryCacheProgressService",
        4 => "CreateDeliveryCacheProgressServiceWithApplicationId",
        _ => "unknown",
    }
}

fn file_command_name(cmd: u32) -> &'static str {
    match cmd {
        1 => "Read",
        2 => "GetSize",
        3 => "GetDigest",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::services::harness;

    fn call(c: &mut Cpu, service: &dyn Service, cmd: u32) -> Reply {
        service.call(c, &Request { cmd, ..Default::default() })
    }

    #[test]
    fn every_bcat_port_creates_a_delivery_cache_storage() {
        assert_eq!(NOT_FOUND, 0x47a);
        for name in ["bcat:u", "bcat:s", "bcat:m", "bcat:a"] {
            let called = harness::call_service(name, None, Request { cmd: 1, ..Default::default() });
            assert_eq!((called.result, called.objects), (0, 1), "{name}");
        }
    }

    #[test]
    fn the_delivery_cache_is_empty() {
        extern "C" fn probe(c: &mut Cpu) {
            let storage = DeliveryCacheStorage;
            let directories = call(c, &storage, 10);
            let file = call(c, &storage, 0);
            let directory = call(c, &storage, 1);
            harness::record(
                "bcat/empty",
                &[
                    directories.result as u64,
                    u64::from(u32::from_le_bytes(directories.data[..4].try_into().unwrap())),
                    file.objects.len() as u64,
                    directory.objects.len() as u64,
                    call(c, &DeliveryCacheFile, 0).result as u64,
                    call(c, &DeliveryCacheDirectory, 0).result as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        assert_eq!(
            harness::observed("bcat/empty"),
            vec![0, 0, 1, 1, NOT_FOUND as u64, NOT_FOUND as u64],
            "no directories, and opening any file or directory is not found"
        );
    }

    #[test]
    fn an_unimplemented_command_is_refused() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = call(c, &DeliveryCacheFile, 2);
            harness::record("bcat/unknown", &[reply.result as u64, reply.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        assert_eq!(
            harness::observed("bcat/unknown"),
            vec![crate::services::result::CMIF_UNKNOWN_COMMAND as u64, 0]
        );
    }
}
