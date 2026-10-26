use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

pub struct Storage {
    data: Arc<Mutex<Vec<u8>>>,
}

impl Storage {
    pub fn new(data: Vec<u8>) -> Self {
        Storage {
            data: Arc::new(Mutex::new(data)),
        }
    }
}

impl Service for Storage {
    fn name(&self) -> String {
        "am/IStorage".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(StorageAccessor {
                data: self.data.clone(),
            })),
            _ => unimplemented("am/IStorage", c, req),
        }
    }
}

struct StorageAccessor {
    data: Arc<Mutex<Vec<u8>>>,
}

impl Service for StorageAccessor {
    fn name(&self) -> String {
        "am/IStorageAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(self.data.lock().unwrap().len() as u64),
            10 => {
                let mut data = self.data.lock().unwrap();
                let off = (req.u64(0) as usize).min(data.len());
                if let Some(b) = req.in_buf(0) {
                    let n = (b.size as usize).min(data.len() - off);
                    if let Some(bytes) = c.process().memory.read(b.addr, n as u64) {
                        data[off..off + n].copy_from_slice(&bytes);
                    }
                }
                Reply::ok()
            }
            11 => {
                let data = self.data.lock().unwrap();
                let off = (req.u64(0) as usize).min(data.len());
                if let Some(b) = req.out_buf(0) {
                    let n = (b.size as usize).min(data.len() - off);
                    c.process().memory.write(b.addr, &data[off..off + n]);
                }
                Reply::ok()
            }
            _ => unimplemented("am/IStorageAccessor", c, req),
        }
    }
}
