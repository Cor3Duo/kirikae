//! `mii:u` / `mii:e` — Mii database queries.
//!
//! The runtime has no Mii database, which is a valid offline console state: no
//! user Miis, no special Miis and no full database. `OpenDatabase` creates the
//! SDK object, and its queries answer zero counts / "not updated" / "not full".
//! `BuildRandom` still returns a zeroed `MiiCharInfo` of the documented 0x58
//! bytes so a caller that needs a placeholder gets a well-formed buffer.
//!
//! Command numbers follow libnx's `mii.c` (the SDK wrapper for this service).

use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

/// `MiiCharInfo` is 0x58 bytes.
const MII_CHAR_INFO_SIZE: usize = 0x58;

pub struct Mii {
    name: String,
}

impl Mii {
    pub fn new(name: &str) -> Mii {
        Mii {
            name: name.to_string(),
        }
    }
}

impl Service for Mii {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // OpenDatabase(MiiSpecialKeyCode)
            0 => Reply::ok().object(Arc::new(MiiDatabase)),
            _ => unimplemented_named(&self.name, mii_command_name(req.cmd), c, req),
        }
    }
}

/// `nn::mii::detail::IDatabase`.
struct MiiDatabase;

impl Service for MiiDatabase {
    fn name(&self) -> String {
        "mii/IDatabase".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // IsUpdated(source flag): an empty database never changes.
            0 => Reply::ok().u8(0),
            // IsFull
            1 => Reply::ok().u8(0),
            // GetCount(source flag)
            2 => Reply::ok().u32(0),
            // Get1(source flag, output MiiCharInfo array): no entries.
            4 => Reply::ok().u32(0),
            // BuildRandom(age, gender, face color): a placeholder Mii.
            6 => Reply::ok().bytes(&[0; MII_CHAR_INFO_SIZE]),
            _ => unimplemented_named("mii/IDatabase", mii_database_command_name(req.cmd), c, req),
        }
    }
}

fn mii_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "OpenDatabase",
        _ => "unknown",
    }
}

fn mii_database_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "IsUpdated",
        1 => "IsFull",
        2 => "GetCount",
        3 => "Get",
        4 => "Get1",
        5 => "BuildDefault",
        6 => "BuildRandom",
        7 => "Delete",
        8 => "Destroy",
        9 => "GetIndex",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::services::harness;

    #[test]
    fn database_queries_answer_an_empty_offline_database() {
        extern "C" fn probe(c: &mut Cpu) {
            let manager = Mii::new("mii:u").call(c, &Request::default());
            harness::record("object", &[manager.objects.len() as u64]);
            let database = MiiDatabase;
            let count = database.call(
                c,
                &Request {
                    cmd: 2,
                    data: 1u32.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            harness::record(
                "count",
                &[u64::from(u32::from_le_bytes(
                    count.data[..4].try_into().unwrap(),
                ))],
            );
            let random = database.call(
                c,
                &Request {
                    cmd: 6,
                    ..Default::default()
                },
            );
            harness::record("random", &[random.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("object"), vec![1]);
        assert_eq!(harness::observed("count"), vec![0]);
        assert_eq!(harness::observed("random"), vec![0x58]);
    }

    #[test]
    fn a_placeholder_mii_is_the_documented_size() {
        assert_eq!(Reply::ok().bytes(&[0; MII_CHAR_INFO_SIZE]).data.len(), 0x58);
    }
}
