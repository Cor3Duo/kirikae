use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::cpu::Cpu;
use crate::process::Process;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

/// `nn::nifm::RequestState`.
const REQUEST_STATE_FREE: u32 = 1;
const REQUEST_STATE_ACCEPTED: u32 = 3;

/// `nn::nifm::detail::sf::NetworkProfileData` as exchanged with the service.
const NETWORK_PROFILE_SIZE: usize = 0x17c;
/// `nn::nifm::IpAddressSetting` (0xd) followed by `nn::nifm::DnsSetting` (0x9).
const IP_CONFIG_INFO_SIZE: usize = 0xd + 0x9;
/// `nn::nifm::Requirement`, plus the 4 bytes [3.0.0+] appends in `SetRequirement`.
const REQUIREMENT_SIZE: usize = 0x24;

/// nifm:u/nifm:s/nifm:a (`nn::nifm::detail::IStaticService`).
///
/// The runtime has no network stack, so the services report a valid machine that is
/// offline: every address is zero, no interface is up and no internet request is
/// accepted. `nn::nifm::detail::IRequest` still goes through a real state transition
/// (Free -> Accepted on `Submit`) and hands the SDK two events, because the SDK builds
/// its client objects from those replies and dereferences whatever comes back.
pub struct Nifm {
    name: &'static str,
}

impl Nifm {
    pub fn new(name: &'static str) -> Nifm {
        Nifm { name }
    }
}

impl Service for Nifm {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // 4 CreateGeneralServiceOld (1.0.0-2.3.0), 5 CreateGeneralService (3.0.0+).
            4 | 5 => Reply::ok().object(Arc::new(GeneralService)),
            _ => unimplemented(self.name, c, req),
        }
    }
}

/// `nn::nifm::detail::IGeneralService`.
struct GeneralService;

fn no_connection_status() -> Reply {
    // NetworkInterfaceType::Invalid, wifiStrength 0, connectionStatus 0 (disconnected).
    Reply::ok().u8(0).u8(0).u8(0)
}

fn ip_config_info() -> Reply {
    Reply::ok().bytes(&[0; IP_CONFIG_INFO_SIZE])
}

/// `GetCurrentNetworkProfile` (5) and `GetNetworkProfile` (8) hand the profile
/// back through a type-0x1A output buffer, not in the message's raw data: libnx
/// sends them with `SfBufferAttr_HipcPointer | SfBufferAttr_FixedSize |
/// SfBufferAttr_Out`. A zeroed profile is the offline machine's "no network
/// profile stored", and it has to land in the caller's buffer or the SDK keeps
/// whatever it had there.
fn network_profile(c: &mut Cpu, req: &Request) -> Reply {
    let Some(b) = req.x.first().or_else(|| req.b.first()) else {
        return Reply::ok();
    };
    let n = (b.size as usize).min(NETWORK_PROFILE_SIZE);
    c.process()
        .memory
        .write(b.addr, &[0; NETWORK_PROFILE_SIZE][..n]);
    Reply::ok()
}

impl Service for GeneralService {
    fn name(&self) -> String {
        "nifm/IGeneralService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // GetClientId: type-0x1A output buffer holding a 4-byte ClientId.
            1 => {
                if let Some(b) = req.x.first().or_else(|| req.c.first())
                    && b.size >= 4
                {
                    c.process().memory.write(b.addr, &0u32.to_le_bytes());
                }
                Reply::ok()
            }
            // CreateRequest(RequirementPreset) -> IRequest.
            4 => Reply::ok().object(Arc::new(NetworkRequest::new(c.process()))),
            // GetCurrentNetworkProfile / GetNetworkProfile: both fill the
            // caller's output buffer with an empty SfNetworkProfileData.
            5 | 8 => network_profile(c, req),
            // EnumerateNetworkProfiles: no profile is stored.
            7 => Reply::ok().u32(0),
            12 => Reply::ok().u32(0),
            15 => ip_config_info(),
            17 => Reply::ok().u8(0),
            18 => no_connection_status(),
            // IsAnyInternetRequestAccepted / IsAnyForegroundRequestAccepted.
            21 | 22 => Reply::ok().u8(0),
            _ => unimplemented("nifm/IGeneralService", c, req),
        }
    }
}

/// State shared by the replies of one `nn::nifm::detail::IRequest`.
struct RequestState {
    state: AtomicU32,
    result: AtomicU32,
    requirement: Mutex<[u8; REQUIREMENT_SIZE]>,
}

impl RequestState {
    fn new() -> RequestState {
        RequestState {
            state: AtomicU32::new(REQUEST_STATE_FREE),
            result: AtomicU32::new(0),
            requirement: Mutex::new([0; REQUIREMENT_SIZE]),
        }
    }

    fn state(&self) -> u32 {
        self.state.load(Relaxed)
    }

    /// Moves the request to Accepted and returns whether the state changed.
    fn submit(&self) -> bool {
        let previous = self.state.swap(REQUEST_STATE_ACCEPTED, Relaxed);
        previous != REQUEST_STATE_ACCEPTED
    }

    fn cancel(&self) -> bool {
        let previous = self.state.swap(REQUEST_STATE_FREE, Relaxed);
        previous != REQUEST_STATE_FREE
    }

    fn store_requirement(&self, data: &[u8]) {
        let mut requirement = self.requirement.lock().unwrap();
        let n = data.len().min(REQUIREMENT_SIZE);
        requirement[..n].copy_from_slice(&data[..n]);
    }

    fn requirement(&self) -> [u8; REQUIREMENT_SIZE] {
        *self.requirement.lock().unwrap()
    }
}

struct NetworkRequest {
    state: RequestState,
    state_event: u32,
    aux_event: u32,
}

impl NetworkRequest {
    fn new(p: &Process) -> NetworkRequest {
        NetworkRequest {
            state: RequestState::new(),
            state_event: p.create_event(false),
            aux_event: p.create_event(false),
        }
    }

    fn signal_state_change(&self, p: &Process) {
        p.set_event(self.state_event, true);
    }
}

impl Service for NetworkRequest {
    fn name(&self) -> String {
        "nifm/IRequest".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 => Reply::ok().u32(self.state.state()),
            // GetResult: the request completed with success.
            1 => Reply::ok().u32(self.state.result.load(Relaxed)),
            2 => Reply::ok()
                .copy_handle(self.state_event)
                .copy_handle(self.aux_event),
            3 => {
                if self.state.cancel() {
                    self.signal_state_change(p);
                }
                Reply::ok()
            }
            4 => {
                if self.state.submit() {
                    self.signal_state_change(p);
                }
                Reply::ok()
            }
            5 => {
                self.state.store_requirement(&req.data);
                Reply::ok()
            }
            6 | 8..=18 | 23..=25 => Reply::ok(),
            19 => Reply::ok().bytes(&self.state.requirement()),
            20 => Reply::ok().u32(0),
            21 => Reply::ok().u32(0).u32(0).u32(0),
            22 => Reply::ok().u32(0).u32(0),
            _ => unimplemented("nifm/IRequest", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_request_is_free_with_no_error() {
        let state = RequestState::new();
        assert_eq!(state.state(), REQUEST_STATE_FREE);
        assert_eq!(state.result.load(Relaxed), 0);
    }

    #[test]
    fn submit_moves_the_request_to_accepted_only_once() {
        let state = RequestState::new();
        assert!(state.submit());
        assert_eq!(state.state(), REQUEST_STATE_ACCEPTED);
        assert!(!state.submit());
    }

    #[test]
    fn cancel_returns_the_request_to_free() {
        let state = RequestState::new();
        state.submit();
        assert!(state.cancel());
        assert_eq!(state.state(), REQUEST_STATE_FREE);
        assert!(!state.cancel());
    }

    #[test]
    fn requirements_are_stored_and_read_back_at_the_documented_size() {
        let state = RequestState::new();
        assert_eq!(state.requirement(), [0; REQUIREMENT_SIZE]);

        let data: Vec<u8> = (0..REQUIREMENT_SIZE as u8 + 8).collect();
        state.store_requirement(&data);
        let stored = state.requirement();
        assert_eq!(stored.len(), REQUIREMENT_SIZE);
        assert_eq!(&stored[..], &data[..REQUIREMENT_SIZE]);

        state.store_requirement(&[0xaa; 4]);
        assert_eq!(state.requirement()[..4], [0xaa; 4]);
    }

    #[test]
    fn offline_replies_have_the_documented_sizes() {
        let status = no_connection_status();
        assert_eq!(status.result, 0);
        assert_eq!(status.data, [0, 0, 0]);

        let ip = ip_config_info();
        assert_eq!(ip.data.len(), 0x16);
        assert!(ip.data.iter().all(|&b| b == 0));
        assert_eq!(NETWORK_PROFILE_SIZE, 0x17c);
    }

    #[test]
    fn the_profile_reads_land_in_the_callers_pointer_buffer() {
        use crate::abi::Entry;
        use crate::services::{Buf, harness};

        extern "C" fn probe(c: &mut Cpu) {
            let out = 0x1_2000_0000u64;
            c.process()
                .memory
                .map_as(out, 0x1000, crate::mem::state::NORMAL, crate::mem::perm::RW)
                .unwrap();
            c.process().memory.write(out, &[0xaa; 0x1000]);

            let service = GeneralService;
            let full = service.call(
                c,
                &Request {
                    cmd: 5,
                    x: vec![Buf {
                        addr: out,
                        size: NETWORK_PROFILE_SIZE as u64,
                    }],
                    ..Default::default()
                },
            );
            let profile = c
                .process()
                .memory
                .read(out, NETWORK_PROFILE_SIZE as u64)
                .unwrap();

            // A buffer shorter than the struct takes only what fits; the bytes
            // past it keep their old contents.
            let short = out + 0x200;
            c.process().memory.write(short, &[0xbb; 0x10]);
            let partial = service.call(
                c,
                &Request {
                    cmd: 8,
                    x: vec![Buf {
                        addr: short,
                        size: 8,
                    }],
                    ..Default::default()
                },
            );
            let tail = c.process().memory.read(short, 0x10).unwrap();

            harness::record(
                "profiles",
                &[
                    u64::from(full.result),
                    full.data.len() as u64,
                    profile.iter().filter(|&&b| b == 0).count() as u64,
                    u64::from(partial.result),
                    partial.data.len() as u64,
                    tail.iter().filter(|&&b| b == 0).count() as u64,
                    u64::from(tail[8]),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("profiles"),
            vec![0, 0, NETWORK_PROFILE_SIZE as u64, 0, 0, 8, 0xbb]
        );
    }

    #[test]
    fn request_replies_carry_the_state_the_sdk_reads() {
        let state = RequestState::new();
        assert_eq!(
            state.state().to_le_bytes(),
            REQUEST_STATE_FREE.to_le_bytes()
        );
        state.submit();
        assert_eq!(
            state.state().to_le_bytes(),
            REQUEST_STATE_ACCEPTED.to_le_bytes()
        );
    }
}
