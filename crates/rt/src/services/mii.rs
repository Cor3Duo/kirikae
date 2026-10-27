use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

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
            0 => Reply::ok().object(Arc::new(MiiDatabase)),
            _ => unimplemented_named(&self.name, mii_command_name(req.cmd), c, req),
        }
    }
}

struct MiiDatabase;

impl Service for MiiDatabase {
    fn name(&self) -> String {
        "mii/IDatabase".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u8(0),
            1 => Reply::ok().u8(0),
            2 => Reply::ok().u32(0),
            4 => Reply::ok().u32(0),
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
    use crate::process::{RunOptions, run_executable};
    use std::sync::Mutex;

    static OPENED: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());

    extern "C" fn open_both(c: &mut Cpu) {
        for name in ["mii:u", "mii:e"] {
            let service = crate::services::open(name);
            let reply = service.call(c, &Request { cmd: 0, ..Default::default() });
            OPENED.lock().unwrap().push((service.name(), reply.objects.len()));
        }
        c.halt();
    }

    #[test]
    fn both_ports_open_the_database() {
        let entries: &'static [Entry] = Box::leak(vec![Entry { addr: 0x1000_0000, func: open_both }].into_boxed_slice());
        let exe = kirikae_loader::Executable { entry: entries[0].addr, ..Default::default() };
        let result = run_executable(&exe, entries, RunOptions { capture_output: true }).expect("the fake guest runs");
        assert_eq!(result.exit_code, 0);
        assert_eq!(*OPENED.lock().unwrap(), vec![("mii:u".to_string(), 1), ("mii:e".to_string(), 1)]);
    }

    #[test]
    fn the_character_info_has_the_documented_size() {
        assert_eq!(MII_CHAR_INFO_SIZE, 0x58);
    }

    #[test]
    fn the_service_keeps_the_name_it_was_opened_with() {
        assert_eq!(Mii::new("mii:u").name(), "mii:u");
        assert_eq!(Mii::new("mii:e").name(), "mii:e");
        assert_eq!(mii_command_name(0), "OpenDatabase");
        assert_eq!(mii_command_name(999), "unknown");
    }
}
