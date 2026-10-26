use std::sync::Mutex;

use crate::cpu::Cpu;
use crate::process::Process;
use crate::services::stub::unimplemented_named;
use crate::services::{Buf, Reply, Request, Service};

const STATE_NONE: u32 = 0;
const STATE_INITIALIZED: u32 = 1;
const STATE_ACCESS_POINT: u32 = 2;
const STATE_ACCESS_POINT_CREATED: u32 = 3;
const STATE_STATION: u32 = 4;
const STATE_STATION_CONNECTED: u32 = 5;
const STATE_ERROR: u32 = 6;

const DISCONNECT_REASON_NONE: i16 = 0;
const DISCONNECT_REASON_BY_USER: i16 = 1;

const LDN_MODULE: u32 = 203;
const fn ldn_error(description: u32) -> u32 {
    (description << 9) | LDN_MODULE
}

const DEVICE_NOT_AVAILABLE: u32 = ldn_error(16);
const DEVICE_DISABLED: u32 = ldn_error(22);
const INVALID_STATE: u32 = ldn_error(32);
const NODE_NOT_FOUND: u32 = ldn_error(48);
const CONNECT_NOT_FOUND: u32 = ldn_error(65);
const INVALID_ARGUMENT: u32 = ldn_error(96);

const NETWORK_INFO_SIZE: usize = 0x480;
const NODE_LATEST_UPDATE_SIZE: usize = 8;
const SECURITY_PARAMETER_SIZE: usize = 0x20;
const NETWORK_CONFIG_SIZE: usize = 0x20;
const IPV4_ADDRESS_SIZE: usize = 8;

pub struct Ldn;

impl Service for Ldn {
    fn name(&self) -> String {
        "ldn:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(std::sync::Arc::new(LdnSession::new(c.process()))),
            1 => unimplemented_named("ldn:u", "CreateClientProcessMonitor", c, req),
            _ => unimplemented_named("ldn:u", "unknown", c, req),
        }
    }
}

#[derive(Clone, Copy)]
struct SessionState {
    state: u32,
    disconnect_reason: i16,
}

struct LdnSession {
    state: Mutex<SessionState>,
    state_change_event: u32,
}

impl LdnSession {
    fn new(p: &Process) -> LdnSession {
        LdnSession {
            state: Mutex::new(SessionState {
                state: STATE_NONE,
                disconnect_reason: DISCONNECT_REASON_NONE,
            }),
            state_change_event: p.create_event(false),
        }
    }

    fn state(&self) -> u32 {
        self.state.lock().unwrap().state
    }

    fn set_state(&self, value: u32) {
        self.state.lock().unwrap().state = value;
    }

    fn require_state(&self, allowed: &[u32]) -> Result<u32, Reply> {
        let state = self.state();
        if allowed.contains(&state) {
            Ok(state)
        } else {
            Err(Reply::err(INVALID_STATE))
        }
    }

    fn initialize(&self, req: &Request) -> Reply {
        if req.cmd == 402 {
            let version = req
                .data
                .get(..4)
                .map_or(0, |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            if version < 0 {
                return Reply::err(INVALID_ARGUMENT);
            }
        }
        match self.state() {
            STATE_NONE | STATE_ERROR | STATE_INITIALIZED => {
                self.set_state(STATE_INITIALIZED);
                Reply::ok()
            }
            _ => Reply::err(INVALID_STATE),
        }
    }
}

impl Service for LdnSession {
    fn name(&self) -> String {
        "ldn:u/IUserLocalCommunicationService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let state = self.state();
        match req.cmd {
            0 => Reply::ok().u32(state),
            1 => {
                zero_buffer(c, req, NETWORK_INFO_SIZE);
                Reply::ok()
            }
            2 => Reply::ok().bytes(&[0; IPV4_ADDRESS_SIZE]),
            3 => Reply::ok().bytes(&self.state.lock().unwrap().disconnect_reason.to_le_bytes()),
            4 => Reply::ok().bytes(&[0; SECURITY_PARAMETER_SIZE]),
            5 => Reply::ok().bytes(&[0; NETWORK_CONFIG_SIZE]),
            100 => Reply::ok().copy_handle(self.state_change_event),
            101 => {
                zero_buffer(c, req, NETWORK_INFO_SIZE);
                zero_buffer_named(c, req, 1, NODE_LATEST_UPDATE_SIZE);
                Reply::ok()
            }
            102 | 103 => scan(c, req, state),
            104 => {
                if req.u32(0) > 1 {
                    return Reply::err(INVALID_ARGUMENT);
                }
                self.require_state(&[STATE_INITIALIZED])
                    .map_or_else(|error| error, |_| Reply::ok())
            }
            105 => {
                if req.u32(0) > 1 {
                    return Reply::err(INVALID_ARGUMENT);
                }
                Reply::ok()
            }
            106 => {
                if !matches!(req.u32(0), 1 | 3) {
                    return Reply::err(INVALID_ARGUMENT);
                }
                self.require_state(&[STATE_INITIALIZED])
                    .map_or_else(|error| error, |_| Reply::ok())
            }
            200 => self
                .require_state(&[STATE_INITIALIZED])
                .map_or_else(|error| error, |_| Reply::err(DEVICE_NOT_AVAILABLE)),
            201 => {
                let result = self.require_state(&[STATE_ACCESS_POINT, STATE_ACCESS_POINT_CREATED]);
                match result {
                    Ok(_) => {
                        self.set_state(STATE_INITIALIZED);
                        Reply::ok()
                    }
                    Err(error) => error,
                }
            }
            202 | 203 => self
                .require_state(&[STATE_ACCESS_POINT])
                .map_or_else(|error| error, |_| Reply::err(DEVICE_NOT_AVAILABLE)),
            204 => {
                let result = self.require_state(&[STATE_ACCESS_POINT_CREATED]);
                match result {
                    Ok(_) => {
                        self.set_state(STATE_ACCESS_POINT);
                        Reply::ok()
                    }
                    Err(error) => error,
                }
            }
            205 => self
                .require_state(&[STATE_ACCESS_POINT_CREATED])
                .map_or_else(|error| error, |_| Reply::err(NODE_NOT_FOUND)),
            206 => self
                .require_state(&[STATE_ACCESS_POINT, STATE_ACCESS_POINT_CREATED])
                .map_or_else(|error| error, |_| Reply::ok()),
            207 => {
                if req.data.first().copied().unwrap_or(0) > 3 {
                    return Reply::err(INVALID_ARGUMENT);
                }
                self.require_state(&[STATE_ACCESS_POINT, STATE_ACCESS_POINT_CREATED])
                    .map_or_else(|error| error, |_| Reply::ok())
            }
            208 | 209 => self
                .require_state(&[STATE_ACCESS_POINT, STATE_ACCESS_POINT_CREATED])
                .map_or_else(|error| error, |_| Reply::ok()),
            300 => {
                let result = self.require_state(&[STATE_INITIALIZED]);
                match result {
                    Ok(_) => {
                        self.set_state(STATE_STATION);
                        Reply::ok()
                    }
                    Err(error) => error,
                }
            }
            301 => {
                let result = self.require_state(&[STATE_STATION, STATE_STATION_CONNECTED]);
                match result {
                    Ok(_) => {
                        self.set_state(STATE_INITIALIZED);
                        Reply::ok()
                    }
                    Err(error) => error,
                }
            }
            302 | 303 => self
                .require_state(&[STATE_STATION])
                .map_or_else(|error| error, |_| Reply::err(CONNECT_NOT_FOUND)),
            304 => {
                let result = self.require_state(&[STATE_STATION_CONNECTED]);
                match result {
                    Ok(_) => {
                        self.state.lock().unwrap().disconnect_reason = DISCONNECT_REASON_BY_USER;
                        self.set_state(STATE_STATION);
                        Reply::ok()
                    }
                    Err(error) => error,
                }
            }
            400 | 402 => self.initialize(req),
            401 => {
                let mut state = self.state.lock().unwrap();
                state.state = STATE_NONE;
                state.disconnect_reason = DISCONNECT_REASON_NONE;
                Reply::ok()
            }
            403 => {
                if req.u32(0) > 1 {
                    return Reply::err(INVALID_ARGUMENT);
                }
                self.require_state(&[STATE_INITIALIZED])
                    .map_or_else(|error| error, |_| Reply::ok())
            }
            500 => self
                .require_state(&[STATE_INITIALIZED])
                .map_or_else(|error| error, |_| Reply::err(DEVICE_NOT_AVAILABLE)),
            501 => self
                .require_state(&[STATE_INITIALIZED])
                .map_or_else(|error| error, |_| Reply::ok()),
            502 | 503 => self
                .require_state(&[
                    STATE_ACCESS_POINT_CREATED,
                    STATE_STATION,
                    STATE_STATION_CONNECTED,
                ])
                .map_or_else(|error| error, |_| Reply::err(DEVICE_NOT_AVAILABLE)),
            505 => self
                .require_state(&[STATE_STATION])
                .map_or_else(|error| error, |_| Reply::err(DEVICE_NOT_AVAILABLE)),
            600 | 601 => self
                .require_state(&[
                    STATE_ACCESS_POINT,
                    STATE_ACCESS_POINT_CREATED,
                    STATE_STATION,
                    STATE_STATION_CONNECTED,
                ])
                .map_or_else(
                    |error| error,
                    |_| {
                        if req.cmd == 600 && req.data.len() < 2 {
                            Reply::err(INVALID_ARGUMENT)
                        } else {
                            Reply::ok()
                        }
                    },
                ),
            _ => unimplemented_named("ldn:u", command_name(req.cmd), c, req),
        }
    }
}

fn output_buffer(req: &Request, index: usize) -> Option<Buf> {
    req.b
        .get(index)
        .filter(|b| b.size > 0)
        .or_else(|| req.x.get(index).filter(|b| b.size > 0))
        .or_else(|| req.c.get(index).filter(|b| b.size > 0))
        .copied()
}

fn zero_buffer_named(c: &mut Cpu, req: &Request, index: usize, size: usize) {
    if let Some(b) = output_buffer(req, index) {
        let len = (b.size as usize).min(size);
        c.process().memory.write(b.addr, &vec![0; len]);
    }
}

fn zero_buffer(c: &mut Cpu, req: &Request, size: usize) {
    zero_buffer_named(c, req, 0, size);
}

fn scan(c: &mut Cpu, req: &Request, state: u32) -> Reply {
    match state {
        STATE_ACCESS_POINT_CREATED | STATE_STATION | STATE_STATION_CONNECTED => {}
        STATE_NONE | STATE_ERROR => return Reply::err(DEVICE_DISABLED),
        _ => return Reply::err(INVALID_STATE),
    }
    if let Some(b) = output_buffer(req, 0) {
        c.process().memory.write(b.addr, &vec![0; b.size as usize]);
    }
    Reply::ok().bytes(&0u16.to_le_bytes())
}

fn command_name(cmd: u32) -> &'static str {
    match cmd {
        404 => "InitializeWithPriority (system service only)",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::kernel::KObject;
    use crate::services::harness;
    use std::sync::atomic::Ordering::Relaxed;

    #[test]
    fn ldn_u_is_opened_from_the_registry() {
        let opened = crate::services::harness::call_service("ldn:u", None, Request { cmd: 0, ..Default::default() });
        assert_eq!((opened.result, opened.objects), (0, 1));
    }

    #[test]
    fn initialize_reports_initialized_and_not_connected() {
        extern "C" fn probe(c: &mut Cpu) {
            let creator = Ldn;
            let created = creator.call(c, &Request::default());
            let session = created.objects.first().unwrap().clone();

            let before = session.call(
                c,
                &Request {
                    cmd: 0,
                    ..Default::default()
                },
            );

            let out = 0x1_2000_0000u64;
            c.process()
                .memory
                .map_as(out, 0x1000, crate::mem::state::NORMAL, crate::mem::perm::RW)
                .unwrap();
            c.process().memory.write(out, &[0xaa; 0x1000]);

            let init = session.call(
                c,
                &Request {
                    cmd: 400,
                    ..Default::default()
                },
            );
            let state = session.call(
                c,
                &Request {
                    cmd: 0,
                    ..Default::default()
                },
            );
            let info = session.call(
                c,
                &Request {
                    cmd: 1,
                    x: vec![Buf {
                        addr: out,
                        size: NETWORK_INFO_SIZE as u64,
                    }],
                    ..Default::default()
                },
            );
            let network_info = c
                .process()
                .memory
                .read(out, NETWORK_INFO_SIZE as u64)
                .unwrap();
            let ip = session.call(
                c,
                &Request {
                    cmd: 2,
                    ..Default::default()
                },
            );
            let reason = session.call(
                c,
                &Request {
                    cmd: 3,
                    ..Default::default()
                },
            );
            let parameter = session.call(
                c,
                &Request {
                    cmd: 4,
                    ..Default::default()
                },
            );
            let config = session.call(
                c,
                &Request {
                    cmd: 5,
                    ..Default::default()
                },
            );

            harness::record(
                "initialize",
                &[
                    u64::from(before.data[0]) | u64::from(before.data[1]) << 8,
                    u64::from(init.result),
                    init.data.len() as u64,
                    u64::from(state.data[0]),
                    u64::from(info.result),
                    info.data.len() as u64,
                    network_info.iter().filter(|&&b| b == 0).count() as u64,
                    u64::from(ip.result),
                    ip.data.len() as u64,
                    u64::from(reason.result),
                    reason.data.len() as u64,
                    u64::from(parameter.result),
                    parameter.data.len() as u64,
                    u64::from(config.result),
                    config.data.len() as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("initialize"),
            vec![0, 0, 0, 1, 0, 0, 0x480, 0, 8, 0, 2, 0, 0x20, 0, 0x20]
        );
    }

    #[test]
    fn a_scan_with_the_radio_off_is_device_disabled() {
        extern "C" fn probe(c: &mut Cpu) {
            let session = LdnSession::new(c.process());
            let scan = session.call(
                c,
                &Request {
                    cmd: 102,
                    ..Default::default()
                },
            );
            harness::record("scan", &[u64::from(scan.result)]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("scan"), vec![u64::from(DEVICE_DISABLED)]);
    }

    #[test]
    fn the_state_change_event_is_a_real_unsignalled_kernel_event() {
        extern "C" fn probe(c: &mut Cpu) {
            let session = LdnSession::new(c.process());
            let event = session.call(
                c,
                &Request {
                    cmd: 100,
                    ..Default::default()
                },
            );
            let handle = event.copy_handles[0];
            let (is_event, signaled) = {
                let handles = c.process().handles.lock().unwrap();
                match handles.get(handle) {
                    Some(KObject::Event(e)) => (true, e.signaled.load(Relaxed)),
                    _ => (false, true),
                }
            };
            harness::record(
                "event",
                &[
                    u64::from(event.result),
                    event.copy_handles.len() as u64,
                    u64::from(is_event),
                    u64::from(signaled),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("event"), vec![0, 1, 1, 0]);
    }

    #[test]
    fn an_unknown_command_is_the_documented_cmif_error() {
        extern "C" fn probe(c: &mut Cpu) {
            let session = LdnSession::new(c.process());
            let unknown = session.call(
                c,
                &Request {
                    cmd: 9000,
                    ..Default::default()
                },
            );
            harness::record(
                "unknown",
                &[u64::from(unknown.result), unknown.data.len() as u64],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("unknown"),
            vec![u64::from(crate::services::result::CMIF_UNKNOWN_COMMAND), 0]
        );
    }

    #[test]
    fn offline_link_commands_refuse_instead_of_faking_a_link() {
        extern "C" fn probe(c: &mut Cpu) {
            fn call(session: &LdnSession, c: &mut Cpu, cmd: u32) -> Reply {
                session.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                )
            }

            let session = LdnSession::new(c.process());

            let scan_off = call(&session, c, 102);
            let open_off = call(&session, c, 200);
            let destroy_off = call(&session, c, 204);

            let init = call(&session, c, 400);
            let open = call(&session, c, 200);
            let create = call(&session, c, 202);
            let destroy = call(&session, c, 204);

            let station = call(&session, c, 300);
            let scan = call(&session, c, 102);
            let connect = call(&session, c, 302);
            let disconnect = call(&session, c, 304);
            let close = call(&session, c, 301);

            let finalize = call(&session, c, 401);
            let state = call(&session, c, 0);

            harness::record(
                "link",
                &[
                    u64::from(scan_off.result),
                    u64::from(open_off.result),
                    u64::from(destroy_off.result),
                    u64::from(init.result),
                    u64::from(open.result),
                    u64::from(create.result),
                    u64::from(destroy.result),
                    u64::from(station.result),
                    u64::from(scan.result),
                    scan.data.len() as u64,
                    scan.data
                        .get(..2)
                        .map_or(1, |b| u64::from(u16::from_le_bytes([b[0], b[1]]))),
                    u64::from(connect.result),
                    u64::from(disconnect.result),
                    u64::from(close.result),
                    u64::from(finalize.result),
                    u64::from(state.data[0]),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let observed = harness::observed("link");
        assert_eq!(
            &observed[..],
            &[
                u64::from(DEVICE_DISABLED),
                u64::from(INVALID_STATE),
                u64::from(INVALID_STATE),
                0,
                u64::from(DEVICE_NOT_AVAILABLE),
                u64::from(INVALID_STATE),
                u64::from(INVALID_STATE),
                0,
                0,
                2,
                0,
                u64::from(CONNECT_NOT_FOUND),
                u64::from(INVALID_STATE),
                0,
                0,
                0,
            ]
        );
    }
}
