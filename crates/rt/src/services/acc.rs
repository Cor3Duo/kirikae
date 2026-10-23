use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

pub const USER_ID: [u8; 16] = [0x01, 0, 0, 0, 0, 0, 0, 0, 0x64, 0x6e, 0x73, 0x70, 0, 0, 0, 0];
pub const NICKNAME: &str = "kirikae";

fn profile_base() -> Vec<u8> {
    let mut v = USER_ID.to_vec();
    v.extend_from_slice(&0u64.to_le_bytes());
    let mut name = [0u8; 0x20];
    name[..NICKNAME.len()].copy_from_slice(NICKNAME.as_bytes());
    v.extend_from_slice(&name);
    v
}

pub struct Account {
    pub name: &'static str,
}

impl Service for Account {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 => Reply::ok().u32(1),
            1 => Reply::ok().u8((req.data.get(..16) == Some(&USER_ID[..])) as u8),
            2 | 3 | 141 => {
                if let Some(out) = req.out_buf(0) {
                    let mut ids = vec![0u8; out.size.min(0x80) as usize];
                    ids[..16].copy_from_slice(&USER_ID);
                    p.memory.write(out.addr, &ids);
                }
                Reply::ok()
            }
            4 | 51 => Reply::ok().bytes(&USER_ID),
            5 => Reply::ok().object(Arc::new(Profile)),
            50 => Reply::ok().u8(0),
            100 | 140 | 160 => Reply::ok(),
            101 => Reply::ok().object(Arc::new(crate::services::stub::Stub::new("acc/IManagerForApplication"))),
            150 => Reply::ok().u8(0),
            _ => unimplemented(self.name, c, req),
        }
    }
}

struct Profile;

impl Service for Profile {
    fn name(&self) -> String {
        "acc/IProfile".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                if let Some(b) = req.b.first() {
                    c.process().memory.write(b.addr, &vec![0u8; b.size.min(0x80) as usize]);
                }
                Reply::ok().bytes(&profile_base())
            }
            1 => Reply::ok().bytes(&profile_base()),
            10 => Reply::ok().u32(0),
            11 => Reply::ok().u32(0),
            _ => unimplemented("acc/IProfile", c, req),
        }
    }
}
