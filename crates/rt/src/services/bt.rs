use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const MODULE: u32 = 113;
const DEVICE_ALREADY_DISCONNECTED: u32 = (3202 << 9) | MODULE;

const LE_EVENT_INFO_SIZE: u64 = 0x400;
const BLE_EVENT_NONE: u32 = 0;

pub struct Bt;

impl Service for Bt {
    fn name(&self) -> String {
        "bt".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0..=7 => Reply::err(DEVICE_ALREADY_DISCONNECTED),
            8 => get_le_event_info(c, req),
            9 => register_ble_event(c),
            _ => unimplemented_named("bt", "unknown", c, req),
        }
    }
}

fn get_le_event_info(c: &mut Cpu, req: &Request) -> Reply {
    if let Some(buf) = req.out_buf(0) {
        let size = buf.size.min(LE_EVENT_INFO_SIZE) as usize;
        let _ = c.process().memory.write(buf.addr, &vec![0u8; size]);
    }
    Reply::ok().u32(BLE_EVENT_NONE)
}

fn register_ble_event(c: &Cpu) -> Reply {
    let handle = *c
        .process()
        .service_state("bt.ble_event", || c.process().create_event(false));
    Reply::ok().copy_handle(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bt_is_opened_from_the_registry() {
        let called = crate::services::harness::call_service("bt", None, Request { cmd: 0, ..Default::default() });
        assert_eq!(called.result, DEVICE_ALREADY_DISCONNECTED);
    }
    use crate::abi::Entry;
    use crate::kernel::KObject;
    use crate::services::harness;

    #[test]
    fn gatt_operations_without_a_device_are_refused_with_no_output() {
        extern "C" fn probe(c: &mut Cpu) {
            let mut all = Vec::new();
            for cmd in 0..=7u32 {
                let reply = Bt.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                all.push(reply.result as u64);
                all.push(reply.data.len() as u64);
                all.push(reply.copy_handles.len() as u64 + reply.objects.len() as u64);
            }
            harness::record("bt/gatt", &all);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(DEVICE_ALREADY_DISCONNECTED, 0x190471);
        let mut expected = Vec::new();
        for _ in 0..8 {
            expected.extend([DEVICE_ALREADY_DISCONNECTED as u64, 0, 0]);
        }
        assert_eq!(
            harness::observed("bt/gatt"),
            expected,
            "GATT commands name a disconnected device and carry no output"
        );
    }

    #[test]
    fn get_le_event_info_zeroes_the_documented_buffer_and_reports_no_event() {
        extern "C" fn probe(c: &mut Cpu) {
            let addr = 0x1_2000_0000u64;
            c.process()
                .memory
                .map_as(
                    addr,
                    0x1000,
                    crate::mem::state::NORMAL,
                    crate::mem::perm::RW,
                )
                .unwrap();
            assert!(c.process().memory.write(addr, &[0xaa; 0x400]));
            let reply = Bt.call(
                c,
                &Request {
                    cmd: 8,
                    b: vec![crate::services::Buf {
                        addr,
                        size: LE_EVENT_INFO_SIZE,
                    }],
                    ..Default::default()
                },
            );
            let bytes = c.process().memory.read(addr, LE_EVENT_INFO_SIZE).unwrap();
            harness::record(
                "bt/event_info",
                &[
                    reply.result as u64,
                    u64::from(u32::from_le_bytes(reply.data[..4].try_into().unwrap())),
                    bytes.iter().all(|&b| b == 0) as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("bt/event_info"),
            vec![0, BLE_EVENT_NONE as u64, 1],
            "GetLeEventInfo returns BleEventType 0 and leaves no event in the buffer"
        );
    }

    #[test]
    fn register_ble_event_returns_a_never_signalled_kernel_event() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = register_ble_event(c);
            let handle = reply.copy_handles.first().copied().unwrap_or(0);
            let signaled = match c.process().handles.lock().unwrap().get(handle) {
                Some(KObject::Event(e)) => e.signaled.load(std::sync::atomic::Ordering::Relaxed),
                _ => true,
            };
            harness::record(
                "bt/event",
                &[
                    reply.result as u64,
                    handle as u64,
                    (!signaled) as u64,
                    reply.data.len() as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let event = harness::observed("bt/event");
        assert_eq!(event[0], 0, "RegisterBleEvent succeeds");
        assert_ne!(event[1], 0, "it returns a handle");
        assert_eq!(
            event[2], 1,
            "the event names a real kernel event that is not signalled"
        );
        assert_eq!(event[3], 0, "the reply carries no data, only the handle");
    }

    #[test]
    fn an_undocumented_command_is_refused() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = Bt.call(
                c,
                &Request {
                    cmd: 10,
                    ..Default::default()
                },
            );
            harness::record("bt/unknown", &[reply.result as u64, reply.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("bt/unknown"),
            vec![crate::services::result::CMIF_UNKNOWN_COMMAND as u64, 0]
        );
    }
}
