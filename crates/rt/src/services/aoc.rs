use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const AOC_BASE_ID_OFFSET: u64 = 0x1000;

const AOC_LOST_ERROR: u64 = 0x7d0a4;
const AOC_LOST_ERROR_BASE: u64 = 2000;

enum Command {
    Reply(Reply),
    Event(bool),
}

pub struct AddOnContentManager {
    events: Mutex<[Option<u32>; 2]>,
}

impl AddOnContentManager {
    pub fn new() -> AddOnContentManager {
        AddOnContentManager {
            events: Mutex::new([None; 2]),
        }
    }

    fn command(&self, req: &Request, title_id: u64) -> Option<Command> {
        use Command::{Event, Reply as Data};
        let r = match req.cmd {
            0..=3 => Data(Reply::ok().u32(0)),
            4 => Data(Reply::ok().u64(aoc_base_id(req.u64(0)))),
            5 => Data(Reply::ok().u64(aoc_base_id(title_id))),
            6 | 7 => Data(Reply::ok()),
            8 | 10 => Event(req.cmd == 10),
            9 => Data(Reply::ok().u64(aoc_lost_error())),
            11 | 12 => Data(Reply::ok()),
            50 => Data(Reply::ok()),
            _ => return None,
        };
        Some(r)
    }

    fn event(&self, c: &Cpu, which: usize) -> u32 {
        *self.events.lock().unwrap()[which].get_or_insert_with(|| c.process().create_event(false))
    }
}

fn aoc_base_id(title_id: u64) -> u64 {
    title_id.wrapping_add(AOC_BASE_ID_OFFSET)
}

fn aoc_lost_error() -> u64 {
    ((AOC_LOST_ERROR & 0x1ff) | (((AOC_LOST_ERROR >> 9) & 0x1fff) << 32)) + AOC_LOST_ERROR_BASE
}

impl Default for AddOnContentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for AddOnContentManager {
    fn name(&self) -> String {
        "aoc:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        if let 100 | 101 = req.cmd {
            return Reply::ok().object(Arc::new(PurchaseEventManager::new(c)));
        }
        match self.command(req, c.process().title_id) {
            Some(Command::Reply(r)) => r,
            Some(Command::Event(with_pid)) => {
                Reply::ok().copy_handle(self.event(c, usize::from(with_pid)))
            }
            None => unimplemented("aoc:u", c, req),
        }
    }
}

struct PurchaseEventManager {
    purchased: u32,
}

impl PurchaseEventManager {
    fn new(c: &Cpu) -> PurchaseEventManager {
        PurchaseEventManager {
            purchased: c.process().create_event(false),
        }
    }
}

impl Service for PurchaseEventManager {
    fn name(&self) -> String {
        "aoc:u/IPurchaseEventManager".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 | 1 => Reply::ok(),
            2 => Reply::ok().copy_handle(self.purchased),
            _ => unimplemented("aoc:u/IPurchaseEventManager", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TITLE_ID: u64 = 0x0100_0000_0000_0000;

    fn command(cmd: u32) -> Option<Command> {
        AddOnContentManager::new().command(
            &Request {
                cmd,
                data: vec![0; 0x10],
                ..Default::default()
            },
            TITLE_ID,
        )
    }

    fn reply(cmd: u32) -> Reply {
        match command(cmd) {
            Some(Command::Reply(r)) => r,
            _ => panic!("command {cmd} should return a data reply"),
        }
    }

    #[test]
    fn counts_and_lists_no_add_on_content() {
        for cmd in 0..=3 {
            let r = reply(cmd);
            assert_eq!(r.data, 0u32.to_le_bytes());
            assert!(r.copy_handles.is_empty());
            assert!(r.objects.is_empty());
        }
    }

    #[test]
    fn base_id_is_the_application_id_plus_0x1000() {
        let manager = AddOnContentManager::new();
        let by_id = manager.command(
            &Request {
                cmd: 4,
                data: (TITLE_ID + 0x2000).to_le_bytes().to_vec(),
                ..Default::default()
            },
            TITLE_ID,
        );
        let Some(Command::Reply(by_id)) = by_id else {
            panic!("expected a data reply");
        };
        assert_eq!(by_id.data, (TITLE_ID + 0x3000).to_le_bytes());

        let r = reply(5);
        assert_eq!(r.data, (TITLE_ID + 0x1000).to_le_bytes());
    }

    #[test]
    fn prepare_and_mount_notifications_have_no_output() {
        for cmd in [6, 7, 11, 12, 50] {
            let r = reply(cmd);
            assert!(r.data.is_empty(), "command {cmd}");
            assert!(r.copy_handles.is_empty());
            assert!(r.objects.is_empty());
        }
    }

    #[test]
    fn list_changed_commands_return_their_events() {
        for (cmd, with_pid) in [(8, false), (10, true)] {
            match command(cmd) {
                Some(Command::Event(actual)) => assert_eq!(actual, with_pid, "command {cmd}"),
                _ => panic!("command {cmd} should return an event"),
            }
        }
    }

    #[test]
    fn lost_error_code_maps_to_2164_1000() {
        let r = reply(9);
        assert_eq!(r.data.len(), 8);
        let value = u64::from_le_bytes(r.data[..8].try_into().unwrap());
        assert_eq!(value & 0xffff, 2164);
        assert_eq!((value >> 32) & 0x1fff, 1000);
    }

    #[test]
    fn aoc_u_is_opened_from_the_registry() {
        let called = crate::services::harness::call_service("aoc:u", Some(TITLE_ID), Request { cmd: 5, ..Default::default() });
        assert_eq!((called.result, called.data), (0, (TITLE_ID + 0x1000).to_le_bytes().to_vec()));
    }

    #[test]
    fn both_purchase_event_managers_are_objects_with_a_real_event() {
        for cmd in [100, 101] {
            let created = crate::services::harness::call_service("aoc:u", Some(TITLE_ID), Request { cmd, ..Default::default() });
            assert_eq!((created.result, created.objects), (0, 1), "command {cmd}");
        }
    }

    extern "C" fn purchase_event(c: &mut Cpu) {
        let manager = AddOnContentManager::new().call(c, &Request { cmd: 100, ..Default::default() });
        let event = manager.objects[0].call(c, &Request { cmd: 2, ..Default::default() });
        let handle = event.copy_handles.first().copied().unwrap_or(0);
        let is_event = matches!(c.process().handles.lock().unwrap().get(handle), Some(crate::kernel::KObject::Event(_)));
        crate::services::harness::record("aoc/purchase_event", &[u64::from(event.result), u64::from(is_event)]);
        c.halt();
    }

    #[test]
    fn the_purchase_event_manager_hands_out_a_kernel_event() {
        crate::services::harness::run(vec![crate::abi::Entry { addr: crate::services::harness::ENTRY, func: purchase_event }]);
        assert_eq!(crate::services::harness::observed("aoc/purchase_event"), vec![0, 1]);
    }

    #[test]
    fn unknown_commands_are_not_handled() {
        for cmd in [13, 102, 301] {
            assert!(command(cmd).is_none(), "command {cmd}");
        }
    }
}
