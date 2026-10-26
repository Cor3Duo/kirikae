use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const STATE_NONE: u32 = 0;
const STATE_ERROR: u32 = 6;

const LDN_MODULE: u32 = 203;
const fn ldn_error(description: u32) -> u32 {
    (description << 9) | LDN_MODULE
}

const DEVICE_DISABLED: u32 = ldn_error(22);
const INVALID_STATE: u32 = ldn_error(32);
const INVALID_ARGUMENT: u32 = ldn_error(96);

const NETWORK_INFO_SIZE: u64 = 0x480;

pub struct Ldn;

impl Service for Ldn {
    fn name(&self) -> String {
        "ldn:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(std::sync::Arc::new(LdnSession::default())),
            1 => unimplemented_named("ldn:u", "CreateClientProcessMonitor", c, req),
            _ => unimplemented_named("ldn:u", "unknown", c, req),
        }
    }
}

#[derive(Default)]
struct LdnSession {
    disabled: AtomicBool,
}

impl LdnSession {
    fn offline_or(&self, result: u32) -> Reply {
        Reply::err(if self.disabled.load(Relaxed) { DEVICE_DISABLED } else { result })
    }
}

impl Service for LdnSession {
    fn name(&self) -> String {
        "ldn:u/IUserLocalCommunicationService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u32(if self.disabled.load(Relaxed) { STATE_ERROR } else { STATE_NONE }),
            1 | 101 => {
                if let Some(b) = req.c.first().filter(|b| b.size > 0) {
                    c.process().memory.write(b.addr, &vec![0; b.size.min(NETWORK_INFO_SIZE) as usize]);
                }
                self.offline_or(INVALID_STATE)
            }
            2 => self.offline_or(INVALID_ARGUMENT),
            3 => Reply::ok().bytes(&0u16.to_le_bytes()),
            4 | 5 | 102 | 103 | 200..=209 | 300..=304 => self.offline_or(INVALID_STATE),
            100 => {
                let p = c.process();
                let event = *p.service_state("ldn.state_change", || p.create_event(false));
                let Some(object) = p.handles.lock().unwrap().get(event) else {
                    return Reply::err(INVALID_STATE);
                };
                let handle = p.handles.lock().unwrap().insert(object);
                Reply::ok().copy_handle(handle)
            }
            104 if req.u32(0) > 1 => Reply::err(INVALID_ARGUMENT),
            104 => Reply::err(INVALID_STATE),
            400 | 402 => {
                self.disabled.store(true, Relaxed);
                Reply::err(DEVICE_DISABLED)
            }
            401 if self.disabled.load(Relaxed) => Reply::err(DEVICE_DISABLED),
            401 => Reply::ok(),
            _ => unimplemented_named("ldn:u", "unknown", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::kernel::KObject;
    use crate::services::{Buf, harness};

    fn call(session: &LdnSession, c: &mut Cpu, cmd: u32) -> Reply {
        session.call(c, &Request { cmd, ..Default::default() })
    }

    #[test]
    fn ldn_u_is_opened_from_the_registry() {
        let opened = harness::call_service("ldn:u", None, Request { cmd: 0, ..Default::default() });
        assert_eq!((opened.result, opened.objects), (0, 1));
    }

    #[test]
    fn without_a_network_initialize_fails_and_the_session_reports_device_disabled() {
        extern "C" fn probe(c: &mut Cpu) {
            let session = LdnSession::default();
            let before = call(&session, c, 0);
            let info_before = call(&session, c, 1);
            let init = call(&session, c, 400);
            let state = call(&session, c, 0);
            let reason = call(&session, c, 3);
            let mut values = vec![
                u64::from(before.data[0]),
                u64::from(info_before.result),
                u64::from(init.result),
                u64::from(state.data[0]),
                u64::from(reason.result),
                reason.data.len() as u64,
            ];
            for cmd in [1, 2, 4, 5, 102, 200, 300, 401, 201, 302] {
                values.push(u64::from(call(&session, c, cmd).result));
            }
            harness::record("ldn/offline", &values);
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        let (dd, state) = (u64::from(DEVICE_DISABLED), u64::from(INVALID_STATE));
        assert_eq!(dd, 0x2ccb, "DeviceDisabled is 2203-0022");
        assert_eq!(
            harness::observed("ldn/offline"),
            vec![0, state, dd, 6, 0, 2, dd, dd, dd, dd, dd, dd, dd, dd, dd, dd]
        );
    }

    #[test]
    fn network_info_is_written_to_the_output_buffer_and_never_to_an_input_buffer() {
        extern "C" fn probe(c: &mut Cpu) {
            let (input, output) = (0x1_2000_0000u64, 0x1_2000_1000u64);
            c.process().memory.map_as(input, 0x2000, crate::mem::state::NORMAL, crate::mem::perm::RW).unwrap();
            c.process().memory.write(input, &[0xaa; 0x2000]);
            let session = LdnSession::default();
            let mut results = Vec::new();
            for cmd in [1, 101] {
                c.process().memory.write(output, &[0xbb; 0x480]);
                let req = Request {
                    cmd,
                    x: vec![Buf { addr: input, size: 0x480 }],
                    c: vec![Buf { addr: output, size: 0x480 }],
                    ..Default::default()
                };
                results.push(u64::from(session.call(c, &req).result));
            }
            let kept = c.process().memory.read(input, 0x480).unwrap();
            let zeroed = c.process().memory.read(output, 0x480).unwrap();
            harness::record(
                "ldn/buffers",
                &[
                    results[0],
                    results[1],
                    u64::from(kept.iter().all(|&b| b == 0xaa)),
                    u64::from(zeroed.iter().all(|&b| b == 0)),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        assert_eq!(
            harness::observed("ldn/buffers"),
            vec![u64::from(INVALID_STATE), u64::from(INVALID_STATE), 1, 1]
        );
    }

    #[test]
    fn each_request_gets_its_own_handle_to_one_unsignalled_state_change_event() {
        extern "C" fn probe(c: &mut Cpu) {
            let first = call(&LdnSession::default(), c, 100);
            let second = call(&LdnSession::default(), c, 100);
            let (a, b) = (first.copy_handles[0], second.copy_handles[0]);
            let handles = c.process().handles.lock().unwrap();
            let (same, signaled) = match (handles.get(a), handles.get(b)) {
                (Some(KObject::Event(x)), Some(KObject::Event(y))) => {
                    (u64::from(std::sync::Arc::ptr_eq(&x, &y)), u64::from(x.signaled.load(Relaxed)))
                }
                _ => (0, 2),
            };
            drop(handles);
            harness::record(
                "ldn/event",
                &[u64::from(first.result), u64::from(a != b), same, signaled],
            );
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        assert_eq!(harness::observed("ldn/event"), vec![0, 1, 1, 0]);
    }

    #[test]
    fn an_unknown_command_is_the_documented_cmif_error() {
        extern "C" fn probe(c: &mut Cpu) {
            let unknown = call(&LdnSession::default(), c, 9000);
            harness::record("ldn/unknown", &[u64::from(unknown.result)]);
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        assert_eq!(
            harness::observed("ldn/unknown"),
            vec![u64::from(crate::services::result::CMIF_UNKNOWN_COMMAND)]
        );
    }
}
