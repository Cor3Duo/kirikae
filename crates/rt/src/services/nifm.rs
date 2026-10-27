use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const MODULE: u32 = 110;
const NETWORK_NOT_AVAILABLE: u32 = (1100 << 9) | MODULE;

const REQUEST_STATE_FREE: u32 = 1;

const CLIENT_ID: u32 = 1;

pub struct Nifm {
    name: String,
}

impl Nifm {
    pub fn new(name: &str) -> Nifm {
        Nifm { name: name.to_string() }
    }
}

impl Service for Nifm {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            4 | 5 => Reply::ok().object(Arc::new(GeneralService)),
            _ => unimplemented_named(&self.name, static_command_name(req.cmd), c, req),
        }
    }
}

struct GeneralService;

impl Service for GeneralService {
    fn name(&self) -> String {
        "nifm/IGeneralService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            1 => {
                if let Some(out) = req.out_buf(0).filter(|b| b.size >= 4) {
                    c.process().memory.write(out.addr, &CLIENT_ID.to_le_bytes());
                }
                Reply::ok()
            }
            4 => Reply::ok().object(Arc::new(NetworkRequest::default())),
            5 | 12 | 15 | 18 => Reply::err(NETWORK_NOT_AVAILABLE),
            17 => Reply::ok().u8(1),
            20..=22 => Reply::ok().u8(0),
            _ => unimplemented_named("nifm/IGeneralService", general_command_name(req.cmd), c, req),
        }
    }
}

#[derive(Default)]
struct NetworkRequest {
    submitted: AtomicBool,
    events: Mutex<Option<[u32; 2]>>,
}

impl NetworkRequest {
    fn events(&self, c: &Cpu) -> [u32; 2] {
        let p = c.process();
        *self.events.lock().unwrap().get_or_insert_with(|| [p.create_event(false), p.create_event(false)])
    }
}

impl Service for NetworkRequest {
    fn name(&self) -> String {
        "nifm/IRequest".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u32(REQUEST_STATE_FREE),
            1 if self.submitted.load(Relaxed) => Reply::err(NETWORK_NOT_AVAILABLE),
            1 | 3 => Reply::ok(),
            2 => {
                let [state, other] = self.events(c);
                Reply::ok().copy_handle(state).copy_handle(other)
            }
            4 => {
                self.submitted.store(true, Relaxed);
                let [state, _] = self.events(c);
                c.process().set_event(state, true);
                Reply::ok()
            }
            5 | 6 | 8..=18 => Reply::ok(),
            _ => unimplemented_named("nifm/IRequest", request_command_name(req.cmd), c, req),
        }
    }
}

fn static_command_name(cmd: u32) -> &'static str {
    match cmd {
        4 => "CreateGeneralServiceOld",
        5 => "CreateGeneralService",
        _ => "unknown",
    }
}

fn general_command_name(cmd: u32) -> &'static str {
    match cmd {
        2 => "CreateScanRequest",
        6 => "EnumerateNetworkInterfaces",
        7 => "EnumerateNetworkProfiles",
        8 => "GetNetworkProfile",
        33 => "ConfirmSystemAvailability",
        _ => "unknown",
    }
}

fn request_command_name(cmd: u32) -> &'static str {
    match cmd {
        19 => "GetRequirement",
        20 => "GetRevision",
        21 => "GetAppletInfo",
        22 => "GetAdditionalInfo",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::kernel::KObject;
    use crate::services::harness;

    fn call(c: &mut Cpu, service: &dyn Service, cmd: u32) -> Reply {
        service.call(c, &Request { cmd, ..Default::default() })
    }

    fn signaled(c: &Cpu, handle: u32) -> bool {
        match c.process().handles.lock().unwrap().get(handle) {
            Some(KObject::Event(e)) => e.signaled.load(Relaxed),
            _ => false,
        }
    }

    #[test]
    fn every_nifm_port_creates_a_general_service() {
        assert_eq!(NETWORK_NOT_AVAILABLE, 0x8986e);
        for name in ["nifm:u", "nifm:s", "nifm:a"] {
            let called = harness::call_service(name, None, Request { cmd: 5, ..Default::default() });
            assert_eq!((called.result, called.objects), (0, 1), "{name}");
        }
    }

    #[test]
    fn a_submitted_request_ends_free_with_network_not_available() {
        extern "C" fn probe(c: &mut Cpu) {
            let request = NetworkRequest::default();
            let events = call(c, &request, 2);
            let state_event = events.copy_handles[0];
            let before = [call(c, &request, 1).result as u64, signaled(c, state_event) as u64];
            let submit = call(c, &request, 4).result as u64;
            let state = call(c, &request, 0);
            harness::record(
                "nifm/request",
                &[
                    events.copy_handles.len() as u64,
                    before[0],
                    before[1],
                    submit,
                    signaled(c, state_event) as u64,
                    u64::from(u32::from_le_bytes(state.data[..4].try_into().unwrap())),
                    call(c, &request, 1).result as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        assert_eq!(
            harness::observed("nifm/request"),
            vec![2, 0, 0, 0, 1, REQUEST_STATE_FREE as u64, NETWORK_NOT_AVAILABLE as u64],
            "two events, nothing to report before Submit, then the state event fires and the request is Free and refused"
        );
    }

    #[test]
    fn the_general_service_reports_no_connection() {
        extern "C" fn probe(c: &mut Cpu) {
            let general = GeneralService;
            let mut all = Vec::new();
            for cmd in [5, 12, 15, 18] {
                all.push(call(c, &general, cmd).result as u64);
            }
            for cmd in [17, 20, 21, 22] {
                let reply = call(c, &general, cmd);
                all.push(reply.result as u64);
                all.push(reply.data[0] as u64);
            }
            let created = call(c, &general, 4);
            all.push(created.result as u64);
            all.push(created.objects.len() as u64);
            harness::record("nifm/general", &all);
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        let refused = NETWORK_NOT_AVAILABLE as u64;
        assert_eq!(
            harness::observed("nifm/general"),
            vec![refused, refused, refused, refused, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            "no profile, address or status; wireless on, nothing accepted; CreateRequest returns a request"
        );
    }

    #[test]
    fn an_unimplemented_command_is_refused() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = call(c, &NetworkRequest::default(), 21);
            harness::record("nifm/unknown", &[reply.result as u64, reply.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry { addr: harness::ENTRY, func: probe }]);
        assert_eq!(
            harness::observed("nifm/unknown"),
            vec![crate::services::result::CMIF_UNKNOWN_COMMAND as u64, 0]
        );
    }
}
