use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
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

pub(super) enum Backing {
    Buffer {
        data: Arc<Mutex<Vec<u8>>>,
        read_only: bool,
    },
    Handle { size: u64, handle: u32 },
}

pub struct Storage {
    pub(super) backing: Backing,
}

impl Storage {
    pub fn new(data: Vec<u8>) -> Self {
        Storage::buffer(data, false)
    }

    pub(super) fn buffer(data: Vec<u8>, read_only: bool) -> Self {
        Storage {
            backing: Backing::Buffer {
                data: Arc::new(Mutex::new(data)),
                read_only,
            },
        }
    }

    pub(super) fn handle(size: u64, handle: u32) -> Self {
        Storage {
            backing: Backing::Handle { size, handle },
        }
    }
}

impl Service for Storage {
    fn name(&self) -> String {
        "am/IStorage".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match (req.cmd, &self.backing) {
            (0, Backing::Buffer { data, read_only }) => {
                Reply::ok().object(Arc::new(StorageAccessor {
                    data: data.clone(),
                    read_only: *read_only,
                }))
            }
            (0, Backing::Handle { .. }) => Reply::err(RESULT_OPENED_AS_WRONG_TYPE),
            (1, Backing::Handle { size, handle }) => {
                Reply::ok().object(Arc::new(TransferStorageAccessor {
                    size: *size,
                    handle: *handle,
                }))
            }
            (1, Backing::Buffer { .. }) => Reply::err(RESULT_OPENED_AS_WRONG_TYPE),
            _ => unimplemented("am/IStorage", c, req),
        }
    }
}

struct StorageAccessor {
    data: Arc<Mutex<Vec<u8>>>,
    read_only: bool,
}

fn out_buffer(req: &Request) -> Option<Buf> {
    req.b
        .first()
        .filter(|b| b.size > 0)
        .or(req.c.first().filter(|b| b.size > 0))
        .or(req.x.first().filter(|b| b.size > 0))
        .copied()
}

impl StorageAccessor {
    fn offset(&self, data: &[u8], offset: u64) -> Result<usize, u32> {
        let offset = usize::try_from(offset).map_err(|_| RESULT_OUT_OF_BOUNDS)?;
        if offset > data.len() {
            return Err(RESULT_OUT_OF_BOUNDS);
        }
        Ok(offset)
    }

    fn write(&self, memory: &GuestMemory, offset: u64, buf: Option<Buf>) -> Reply {
        if self.read_only {
            return Reply::err(RESULT_OBJECT_INVALID);
        }
        let mut data = self.data.lock().unwrap();
        let offset = match self.offset(&data, offset) {
            Ok(offset) => offset,
            Err(result) => return Reply::err(result),
        };
        let Some(buf) = buf else {
            return Reply::ok();
        };
        let len = (buf.size as usize).min(data.len() - offset);
        let Some(bytes) = memory.read(buf.addr, len as u64) else {
            return Reply::err(RESULT_INVALID_PARAMETERS);
        };
        data[offset..offset + len].copy_from_slice(&bytes);
        Reply::ok()
    }

    fn read(&self, memory: &GuestMemory, offset: u64, buf: Option<Buf>) -> Reply {
        let data = self.data.lock().unwrap();
        let offset = match self.offset(&data, offset) {
            Ok(offset) => offset,
            Err(result) => return Reply::err(result),
        };
        let Some(buf) = buf else {
            return Reply::ok();
        };
        let len = (buf.size as usize).min(data.len() - offset);
        if !memory.write(buf.addr, &data[offset..offset + len]) {
            return Reply::err(RESULT_INVALID_PARAMETERS);
        }
        Reply::ok()
    }
}

impl Service for StorageAccessor {
    fn name(&self) -> String {
        "am/IStorageAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(self.data.lock().unwrap().len() as u64),
            10 => self.write(&c.process().memory, req.u64(0), req.in_buf(0)),
            11 => self.read(&c.process().memory, req.u64(0), out_buffer(req)),
            _ => unimplemented("am/IStorageAccessor", c, req),
        }
    }
}

struct TransferStorageAccessor {
    size: u64,
    handle: u32,
}

impl Service for TransferStorageAccessor {
    fn name(&self) -> String {
        "am/ITransferStorageAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(self.size),
            1 => Reply::ok().u64(self.size).copy_handle(self.handle),
            _ => unimplemented("am/ITransferStorageAccessor", c, req),
        }
    }
}
