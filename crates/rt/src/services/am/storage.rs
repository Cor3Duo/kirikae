use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::kernel::KObject;
use crate::mem::GuestMemory;
use crate::services::stub::unimplemented;
use crate::services::{Buf, Reply, Request, Service};

pub(super) const RESULT_OBJECT_INVALID: u32 = 0x3e880;
pub(super) const RESULT_OUT_OF_BOUNDS: u32 = 0x3ee80;
pub(super) const RESULT_INVALID_PARAMETERS: u32 = 0x3f480;
pub(super) const RESULT_OPENED_AS_WRONG_TYPE: u32 = 0x3fe80;
pub(super) const RESULT_NULL_OBJECT: u32 = 0x40c80;
pub(super) const RESULT_MEMORY_ALLOCATION_FAILED: u32 = 0x4b080;

pub(super) const MAX_STORAGE_SIZE: u64 = 0x400_0000;

#[derive(Clone)]
enum Bytes {
    Owned(Arc<Mutex<Vec<u8>>>),
    Mapped { addr: u64, size: u64 },
}

enum Backing {
    Bytes { bytes: Bytes, read_only: bool },
    Handle { size: u64, object: KObject },
}

pub struct Storage {
    backing: Backing,
}

impl Storage {
    pub fn new(data: Vec<u8>) -> Self {
        Storage {
            backing: Backing::Bytes {
                bytes: Bytes::Owned(Arc::new(Mutex::new(data))),
                read_only: false,
            },
        }
    }

    pub(super) fn transfer_memory(addr: u64, size: u64, read_only: bool) -> Self {
        Storage {
            backing: Backing::Bytes {
                bytes: Bytes::Mapped { addr, size },
                read_only,
            },
        }
    }

    pub(super) fn handle(size: u64, object: KObject) -> Self {
        Storage {
            backing: Backing::Handle { size, object },
        }
    }
}

impl Service for Storage {
    fn name(&self) -> String {
        "am/IStorage".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match (req.cmd, &self.backing) {
            (0, Backing::Bytes { bytes, read_only }) => Reply::ok().object(Arc::new(StorageAccessor {
                bytes: bytes.clone(),
                read_only: *read_only,
            })),
            (1, Backing::Handle { size, object }) => Reply::ok().object(Arc::new(TransferStorageAccessor {
                size: *size,
                object: object.clone(),
            })),
            (0 | 1, _) => Reply::err(RESULT_OPENED_AS_WRONG_TYPE),
            _ => unimplemented("am/IStorage", c, req),
        }
    }
}

struct StorageAccessor {
    bytes: Bytes,
    read_only: bool,
}

fn out_buffer(req: &Request) -> Option<Buf> {
    req.b
        .first()
        .filter(|b| b.size > 0)
        .or(req.c.first().filter(|b| b.size > 0))
        .copied()
}

impl StorageAccessor {
    fn len(&self) -> u64 {
        match &self.bytes {
            Bytes::Owned(data) => data.lock().unwrap().len() as u64,
            Bytes::Mapped { size, .. } => *size,
        }
    }

    fn range(&self, offset: u64, buf: Option<Buf>) -> Result<Option<(u64, Buf)>, u32> {
        let len = self.len();
        let size = buf.map_or(0, |b| b.size);
        if offset > len || size > len - offset {
            return Err(RESULT_OUT_OF_BOUNDS);
        }
        Ok(buf.map(|b| (offset, b)))
    }

    fn write(&self, memory: &GuestMemory, offset: u64, buf: Option<Buf>) -> Reply {
        if self.read_only {
            return Reply::err(RESULT_OBJECT_INVALID);
        }
        let (offset, buf) = match self.range(offset, buf) {
            Ok(Some(range)) => range,
            Ok(None) => return Reply::ok(),
            Err(result) => return Reply::err(result),
        };
        let Some(bytes) = memory.read(buf.addr, buf.size) else {
            return Reply::err(RESULT_INVALID_PARAMETERS);
        };
        let written = match &self.bytes {
            Bytes::Owned(data) => {
                data.lock().unwrap()[offset as usize..(offset + buf.size) as usize].copy_from_slice(&bytes);
                true
            }
            Bytes::Mapped { addr, .. } => memory.write(addr + offset, &bytes),
        };
        if written { Reply::ok() } else { Reply::err(RESULT_INVALID_PARAMETERS) }
    }

    fn read(&self, memory: &GuestMemory, offset: u64, buf: Option<Buf>) -> Reply {
        let (offset, buf) = match self.range(offset, buf) {
            Ok(Some(range)) => range,
            Ok(None) => return Reply::ok(),
            Err(result) => return Reply::err(result),
        };
        let bytes = match &self.bytes {
            Bytes::Owned(data) => Some(data.lock().unwrap()[offset as usize..(offset + buf.size) as usize].to_vec()),
            Bytes::Mapped { addr, .. } => memory.read(addr + offset, buf.size),
        };
        match bytes {
            Some(bytes) if memory.write(buf.addr, &bytes) => Reply::ok(),
            _ => Reply::err(RESULT_INVALID_PARAMETERS),
        }
    }
}

impl Service for StorageAccessor {
    fn name(&self) -> String {
        "am/IStorageAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(self.len()),
            10 => self.write(&c.process().memory, req.u64(0), req.in_buf(0)),
            11 => self.read(&c.process().memory, req.u64(0), out_buffer(req)),
            _ => unimplemented("am/IStorageAccessor", c, req),
        }
    }
}

struct TransferStorageAccessor {
    size: u64,
    object: KObject,
}

impl Service for TransferStorageAccessor {
    fn name(&self) -> String {
        "am/ITransferStorageAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(self.size),
            1 => {
                let handle = c.process().handles.lock().unwrap().insert(self.object.clone());
                Reply::ok().u64(self.size).copy_handle(handle)
            }
            _ => unimplemented("am/ITransferStorageAccessor", c, req),
        }
    }
}
