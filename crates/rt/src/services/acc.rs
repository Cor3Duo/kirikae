use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

pub const USER_ID: [u8; 16] = [
    0x01, 0, 0, 0, 0, 0, 0, 0, 0x64, 0x6e, 0x73, 0x70, 0, 0, 0, 0,
];
pub const NICKNAME: &str = "kirikae";

/// `nn::account::NetworkServiceAccountId`. The runtime models one local user with no
/// Nintendo Account linked, so the baas manager has no id and no cache is available.
pub const NETWORK_ACCOUNT_ID: u64 = 0;
pub const NETWORK_ACCOUNT_AVAILABLE: bool = false;

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

impl Account {
    /// Commands that only create an object, kept apart from `call` so the object
    /// lifecycle can be tested without a guest CPU.
    fn object_command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            // GetBaasAccountManagerForApplication(userID): the manager for the current
            // user. The runtime has a single local user, so the input Uid is not checked.
            101 => Some(Reply::ok().object(Arc::new(ManagerForApplication))),
            // LoadOpenContext: loads the baas manager for the user context that is
            // already open, i.e. another IManagerForApplication. Switchbrew lists the
            // reply as unknown; that object is what the SDK keeps from this command
            // (Ryujinx implements 130 the same way, gdkchan/Ryujinx#4359).
            130 => Some(Reply::ok().object(Arc::new(ManagerForApplication))),
            _ => None,
        }
    }
}

impl Service for Account {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        if let Some(r) = self.object_command(req) {
            return r;
        }
        let p = c.process();
        match req.cmd {
            0 => Reply::ok().u32(1),
            1 => Reply::ok().u8((req.data.get(..16) == Some(&USER_ID[..])) as u8),
            2 | 3 | 131 | 141 => {
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
            150 => Reply::ok().u8(0),
            _ => unimplemented(self.name, c, req),
        }
    }
}

/// `nn::account::baas::IManagerForApplication`. Only the commands the application can
/// answer offline are implemented; the id-token and authorization flows need a network
/// account and stay in `unimplemented`.
struct ManagerForApplication;

impl ManagerForApplication {
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            // CheckAvailability: no linked network service account.
            0 => Some(Reply::ok().u8(NETWORK_ACCOUNT_AVAILABLE as u8)),
            // GetAccountId: NetworkServiceAccountId, a u64.
            1 => Some(Reply::ok().u64(NETWORK_ACCOUNT_ID)),
            // LoadIdTokenCache: type-0x6 output buffer, returns the u32 byte count.
            // There is no token cache, so nothing is written and the count is zero.
            3 => Some(Reply::ok().u32(0)),
            // StoreOpenContext(Uid): no output.
            160 => Some(Reply::ok()),
            _ => None,
        }
    }
}

impl Service for ManagerForApplication {
    fn name(&self) -> String {
        "acc/IManagerForApplication".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented("acc/IManagerForApplication", c, req),
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
                    c.process()
                        .memory
                        .write(b.addr, &vec![0u8; b.size.min(0x80) as usize]);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn request(cmd: u32, data: &[u8]) -> Request {
        Request {
            cmd,
            data: data.to_vec(),
            ..Request::default()
        }
    }

    fn manager(cmd: u32) -> Reply {
        ManagerForApplication
            .command(&request(cmd, &[]))
            .unwrap_or_else(|| panic!("IManagerForApplication command {cmd} is handled"))
    }

    #[test]
    fn check_availability_is_a_single_false_byte() {
        let r = manager(0);
        assert_eq!(r.result, 0);
        assert_eq!(r.data, vec![0]);
        assert!(r.copy_handles.is_empty() && r.move_handles.is_empty() && r.objects.is_empty());
    }

    #[test]
    fn account_id_is_a_zero_u64() {
        let r = manager(1);
        assert_eq!(r.result, 0);
        assert_eq!(r.data, 0u64.to_le_bytes());
        assert!(r.objects.is_empty());
    }

    #[test]
    fn id_token_cache_reports_no_bytes() {
        let r = manager(3);
        assert_eq!(r.result, 0);
        assert_eq!(r.data, 0u32.to_le_bytes());
    }

    #[test]
    fn store_open_context_has_no_output() {
        let r = ManagerForApplication
            .command(&request(160, &USER_ID))
            .expect("StoreOpenContext");
        assert_eq!(r.result, 0);
        assert!(r.data.is_empty());
        assert!(r.objects.is_empty() && r.copy_handles.is_empty());
    }

    #[test]
    fn network_account_commands_stay_unimplemented() {
        for cmd in [2, 150, 170] {
            assert!(ManagerForApplication.command(&request(cmd, &[])).is_none());
        }
    }

    #[test]
    fn load_open_context_returns_a_manager_object() {
        let account = Account { name: "acc:u0" };
        let r = account
            .object_command(&request(130, &[]))
            .expect("LoadOpenContext");
        assert_eq!(r.result, 0);
        assert_eq!(r.objects.len(), 1);
        assert_eq!(r.objects[0].name(), "acc/IManagerForApplication");
        assert!(r.data.is_empty() && r.copy_handles.is_empty() && r.move_handles.is_empty());
    }

    #[test]
    fn account_101_returns_a_manager_object_and_keeps_profile_separate() {
        let account = Account { name: "acc:u0" };
        let first = account
            .object_command(&request(101, &USER_ID))
            .expect("GetBaasAccountManagerForApplication");
        let second = account
            .object_command(&request(101, &USER_ID))
            .expect("GetBaasAccountManagerForApplication");
        assert_eq!(first.result, 0);
        assert_eq!(first.objects.len(), 1);
        assert_eq!(first.objects[0].name(), "acc/IManagerForApplication");
        assert_eq!(second.objects.len(), 1);
        assert!(
            !Arc::ptr_eq(&first.objects[0], &second.objects[0]),
            "every call creates a fresh manager object"
        );
        assert!(
            account.object_command(&request(0, &[])).is_none(),
            "plain account commands are not object commands"
        );
    }
}
