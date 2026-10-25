use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::cpu::Cpu;
use crate::kernel::{ConnectError, KObject, Port, Session};
use crate::services::{
    Reply, Request, Service, host_served, open,
    result::{
        CMIF_UNKNOWN_COMMAND, SM_ALREADY_REGISTERED, SM_INVALID_NAME, SM_MAX_SESSIONS,
        SM_NOT_REGISTERED, SM_OUT_OF_SERVICES,
    },
};

/// The runtime's guest registration table accepts 0x40 live names, the same
/// ceiling the kernel puts on guest named ports (`kernel::MAX_NAMED_PORTS`).
/// Horizon's service table is a fixed list too; refusing past a bound keeps a
/// guest from growing an unbounded registry.
pub const MAX_SERVICES: usize = 0x40;

/// `sm`'s session ceiling for one service. The kernel caps a port at 0x3f
/// sessions (`kernel::MAX_PORT_SESSIONS`); `-1` means unlimited.
pub const MAX_SESSIONS: i32 = crate::kernel::MAX_PORT_SESSIONS;

pub struct Sm;

/// A name the guest registered with `sm:`. The port is weak: closing the
/// registration handle frees the name, exactly like closing a server port does
/// for `svcManageNamedPort`.
struct Registration {
    port: Weak<Port>,
    /// Kept for diagnostics and tests; the port enforces the ceiling itself.
    #[allow(dead_code)]
    max_sessions: i32,
    #[allow(dead_code)]
    is_light: bool,
}

impl Registration {
    fn live(&self) -> Option<Arc<Port>> {
        self.port.upgrade()
    }
}

static GUEST_SERVICES: OnceLock<Mutex<BTreeMap<String, Registration>>> = OnceLock::new();

fn guest_services() -> &'static Mutex<BTreeMap<String, Registration>> {
    GUEST_SERVICES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Drops the entries whose port is gone and returns how many live ones remain.
fn reap_services(services: &mut BTreeMap<String, Registration>) -> usize {
    services.retain(|_, r| r.port.strong_count() > 0);
    services.len()
}

/// The service name in `raw`, or the documented `sm` error.
///
/// A `ServiceName` is eight bytes, a zero-terminated string padded with zeroes.
/// All-zero is the documented invalid name (0xc15); so is a non-printable byte
/// or a non-zero byte after the terminator, which the console's decoder never
/// produces and would otherwise become part of a lookup key.
pub fn parse_service_name(raw: u64) -> Result<String, u32> {
    let bytes = raw.to_le_bytes();
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
    if len == 0 || !bytes[..len].iter().all(|b| (0x20..=0x7e).contains(b)) {
        return Err(SM_INVALID_NAME);
    }
    if bytes[len..].iter().any(|&b| b != 0) {
        return Err(SM_INVALID_NAME);
    }
    // Validated as printable ASCII above, so this cannot fail.
    Ok(String::from_utf8_lossy(&bytes[..len]).into_owned())
}

/// The printable form of a name for diagnostics. Unlike
/// [`parse_service_name`] it never refuses: it is only used in trace lines.
pub fn service_name(raw: u64) -> String {
    parse_service_name(raw).unwrap_or_else(|_| format!("{raw:#018x}"))
}

/// Registers a guest port under `name`.
fn register_guest(
    name: &str,
    port: &Arc<Port>,
    max_sessions: i32,
    is_light: bool,
) -> Result<(), u32> {
    let mut services = guest_services().lock().unwrap();
    reap_services(&mut services);
    if services.contains_key(name) || host_served(name) {
        return Err(SM_ALREADY_REGISTERED);
    }
    if services.len() >= MAX_SERVICES {
        return Err(SM_OUT_OF_SERVICES);
    }
    services.insert(
        name.to_string(),
        Registration {
            port: Arc::downgrade(port),
            max_sessions,
            is_light,
        },
    );
    Ok(())
}

/// The live guest port registered under `name`.
fn lookup_guest(name: &str) -> Option<Arc<Port>> {
    let mut services = guest_services().lock().unwrap();
    let port = services.get(name)?.live();
    if port.is_none() {
        services.remove(name);
    }
    port
}

/// Removes a guest registration. Returns whether the name was live.
fn unregister_guest(name: &str) -> bool {
    guest_services().lock().unwrap().remove(name).is_some()
}

/// How many guest names are live (diagnostics and tests).
pub fn guest_service_count() -> usize {
    reap_services(&mut guest_services().lock().unwrap())
}

#[cfg(test)]
fn reset_guest_services() {
    guest_services().lock().unwrap().clear();
}

impl Service for Sm {
    fn name(&self) -> String {
        "sm:".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // RegisterClient. It used to be called Initialize; either way there
            // is no state to set up on this side.
            0 => Reply::ok(),
            1 => get_service_handle(c, req),
            2 => register_service(c, req),
            3 => unregister_service(c, req),
            // DetachClient (11.0.0-11.0.1). The client drops its own handle.
            4 => Reply::ok(),
            _ => Reply::err(CMIF_UNKNOWN_COMMAND),
        }
    }
}

fn get_service_handle(c: &mut Cpu, req: &Request) -> Reply {
    let name = match parse_service_name(req.u64(0)) {
        Ok(name) => name,
        Err(result) => return Reply::err(result),
    };
    let p = c.process();
    p.log_ipc(format!("sm: GetServiceHandle({name:?})"));
    if let Some(port) = lookup_guest(&name) {
        return match port.connect() {
            Ok(link) => {
                let handle = p
                    .handles
                    .lock()
                    .unwrap()
                    .insert(KObject::Session(Session::guest_client(link)));
                p.log_ipc(format!("sm: GetServiceHandle({name:?}) -> {handle:#x}"));
                Reply::ok().move_handle(handle)
            }
            Err(ConnectError::Limit) => {
                p.warn_once(format!(
                    "sm: GetServiceHandle({name:?}): the service is at its session limit"
                ));
                Reply::err(SM_MAX_SESSIONS)
            }
            Err(ConnectError::Closed) => {
                unregister_guest(&name);
                Reply::err(SM_NOT_REGISTERED)
            }
        };
    }
    if host_served(&name) {
        return Reply::ok().object(open(&name));
    }
    p.warn_once(format!(
        "sm: GetServiceHandle({name:?}): no service is registered under that name"
    ));
    Reply::err(SM_NOT_REGISTERED)
}

/// `RegisterService`'s arguments differ between the two serializations:
/// CMIF lays out `{ name, is_light, max_sessions }` (the libnx struct with its
/// padding), TIPC as `{ name, max_sessions, is_light }`.
fn register_arguments(req: &Request) -> (i32, bool) {
    if req.tipc {
        (req.u32(8) as i32, req.u32(12) & 0xff != 0)
    } else {
        (req.u32(12) as i32, req.u32(8) & 0xff != 0)
    }
}

fn register_service(c: &mut Cpu, req: &Request) -> Reply {
    let name = match parse_service_name(req.u64(0)) {
        Ok(name) => name,
        Err(result) => return Reply::err(result),
    };
    let (max_sessions, is_light) = register_arguments(req);
    if !(-1..=MAX_SESSIONS).contains(&max_sessions) {
        c.process().warn_once(format!(
            "sm: RegisterService({name:?}, {max_sessions}): invalid limit"
        ));
        return Reply::err(SM_MAX_SESSIONS);
    }
    if host_served(&name) {
        c.process().warn_once(format!(
            "sm: RegisterService({name:?}): the runtime already serves it"
        ));
        return Reply::err(SM_ALREADY_REGISTERED);
    }
    let p = c.process();
    let port = Port::with_wake(0, is_light, max_sessions, move || crate::sched::wake(p));
    let handle = p
        .handles
        .lock()
        .unwrap()
        .insert(KObject::Port(port.clone()));
    match register_guest(&name, &port, max_sessions, is_light) {
        Ok(()) => {
            p.log_ipc(format!(
                "sm: RegisterService({name:?}, max_sessions={max_sessions}, light={is_light}) -> {handle:#x}"
            ));
            Reply::ok().move_handle(handle)
        }
        Err(result) => {
            // The registration was refused: do not leave the caller holding a
            // port that no name points at.
            p.handles.lock().unwrap().close(handle);
            p.warn_once(format!("sm: RegisterService({name:?}) was refused"));
            Reply::err(result)
        }
    }
}

fn unregister_service(c: &mut Cpu, req: &Request) -> Reply {
    let name = match parse_service_name(req.u64(0)) {
        Ok(name) => name,
        Err(result) => return Reply::err(result),
    };
    if !unregister_guest(&name) {
        return Reply::err(SM_NOT_REGISTERED);
    }
    c.process()
        .log_ipc(format!("sm: UnregisterService({name:?})"));
    Reply::ok()
}

/// Whether a registration exists (diagnostics and tests).
#[cfg(test)]
fn registration(name: &str) -> Option<(i32, bool)> {
    let services = guest_services().lock().unwrap();
    let r = services.get(name)?;
    Some((r.max_sessions, r.is_light))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::ipc::client;
    use crate::services::harness;
    use std::sync::Mutex;

    static OBSERVED: Mutex<Vec<(&'static str, Vec<u64>)>> = Mutex::new(Vec::new());
    static REGISTRY_LOCK: Mutex<()> = Mutex::new(());

    fn record(name: &'static str, values: &[u64]) {
        OBSERVED.lock().unwrap().push((name, values.to_vec()));
    }

    fn observed(name: &'static str) -> Vec<u64> {
        let mut all = OBSERVED.lock().unwrap();
        let i = all
            .iter()
            .position(|(n, _)| *n == name)
            .expect("the guest recorded nothing under that name");
        all.remove(i).1
    }

    fn name(raw: &str) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[..raw.len()].copy_from_slice(raw.as_bytes());
        u64::from_le_bytes(bytes)
    }

    fn port(max_sessions: i32) -> Arc<Port> {
        Port::with_wake(0, false, max_sessions, || {})
    }

    #[test]
    fn service_names_follow_the_encoding() {
        assert_eq!(parse_service_name(name("fsp-srv")), Ok("fsp-srv".into()));
        assert_eq!(
            parse_service_name(name("appletOE")),
            Ok("appletOE".into()),
            "eight bytes with no terminator are a valid name"
        );
        assert_eq!(parse_service_name(0), Err(SM_INVALID_NAME));
        assert_eq!(
            parse_service_name(u64::from_le_bytes(*b"ab\0cdefg")),
            Err(SM_INVALID_NAME),
            "a byte after the terminator is not part of any console name"
        );
        assert_eq!(
            parse_service_name(u64::from_le_bytes(*b"bad\nname")),
            Err(SM_INVALID_NAME)
        );
    }

    #[test]
    fn the_registry_enforces_its_limits_and_frees_names() {
        let _guard = REGISTRY_LOCK.lock().unwrap();
        reset_guest_services();
        let first = port(1);
        assert_eq!(register_guest("test:a", &first, 1, false), Ok(()));
        assert_eq!(
            register_guest("test:a", &first, 1, false),
            Err(SM_ALREADY_REGISTERED)
        );
        assert_eq!(registration("test:a"), Some((1, false)));
        assert_eq!(lookup_guest("test:a").map(|p| p.name), Some(0));

        let mut live = vec![first.clone()];
        for i in 0..MAX_SERVICES - 1 {
            let p = port(1);
            let name = format!("test:{i:03}");
            assert_eq!(register_guest(&name, &p, 1, false), Ok(()));
            // Keep the port alive for the lifetime of the registry entry.
            live.push(p);
        }
        assert_eq!(live.len(), MAX_SERVICES);
        assert_eq!(guest_service_count(), MAX_SERVICES);
        let extra = port(1);
        assert_eq!(
            register_guest("test:full", &extra, 1, false),
            Err(SM_OUT_OF_SERVICES)
        );
        assert!(unregister_guest("test:a"));
        assert!(lookup_guest("test:a").is_none());
        assert!(!unregister_guest("test:a"));
        reset_guest_services();
    }

    #[test]
    fn a_dead_port_frees_its_name() {
        let _guard = REGISTRY_LOCK.lock().unwrap();
        reset_guest_services();
        let p = port(1);
        assert_eq!(register_guest("test:gone", &p, 1, false), Ok(()));
        drop(p);
        assert!(lookup_guest("test:gone").is_none());
        assert_eq!(guest_service_count(), 0);
    }

    fn sm_handle(c: &mut Cpu) -> u32 {
        harness::session(c, Arc::new(Sm))
    }

    fn send_cmif(c: &mut Cpu, handle: u32, cmd: u32, args: &[u8]) -> (u32, Vec<u8>, Vec<u32>) {
        let msg = client::cmif(crate::ipc::TYPE_REQUEST, cmd, args, None, false, &[]);
        let reply = harness::send(c, handle, &msg);
        let (result, data, hipc) = client::cmif_response(&reply, false);
        (result, data, hipc.move_handles)
    }

    fn send_tipc(c: &mut Cpu, handle: u32, cmd: u32, args: &[u8]) -> (u32, Vec<u8>, Vec<u32>) {
        let msg = client::tipc(cmd, args, &[]);
        let reply = harness::send(c, handle, &msg);
        let (result, data, hipc) = client::tipc_response(&reply);
        (result, data, hipc.move_handles)
    }

    fn register_args_tipc(name: u64, max_sessions: i32, is_light: bool) -> Vec<u8> {
        let mut args = name.to_le_bytes().to_vec();
        args.extend_from_slice(&max_sessions.to_le_bytes());
        args.push(is_light as u8);
        args.resize(16, 0);
        args
    }

    fn register_args_cmif(name: u64, max_sessions: i32, is_light: bool) -> Vec<u8> {
        let mut args = name.to_le_bytes().to_vec();
        args.push(is_light as u8);
        args.resize(12, 0);
        args.extend_from_slice(&max_sessions.to_le_bytes());
        args.resize(16, 0);
        args
    }

    extern "C" fn register_lookup(c: &mut Cpu) {
        let sm = sm_handle(c);
        // TIPC RegisterService (12.0.0+ layout: name, max_sessions, is_light).
        let (result, _, handles) =
            send_tipc(c, sm, 2, &register_args_tipc(name("test:r"), 2, false));
        record("register_tipc", &[result as u64, handles.len() as u64]);

        // A duplicate is refused.
        let (result, _, _) = send_tipc(c, sm, 2, &register_args_tipc(name("test:r"), 2, false));
        record("register_duplicate", &[result as u64]);

        // All-zero names are the documented invalid name.
        let (result, _, _) = send_tipc(c, sm, 2, &register_args_tipc(0, 2, false));
        record("register_invalid", &[result as u64]);

        // CMIF GetServiceHandle (the legacy path every SDK version can use).
        let (result, _, handles) = send_cmif(c, sm, 1, &name("test:r").to_le_bytes());
        record(
            "lookup_cmif",
            &[result as u64, handles.len() as u64, handles[0] as u64],
        );
        // The second connection is within the limit of two...
        let (result, _, handles) = send_cmif(c, sm, 1, &name("test:r").to_le_bytes());
        record("lookup_second", &[result as u64, handles[0] as u64]);
        // ...and the third is refused by the port's ceiling.
        let (result, _, _) = send_tipc(c, sm, 1, &name("test:r").to_le_bytes());
        record("lookup_limit", &[result as u64]);

        // A name nobody registered.
        let (result, _, _) = send_cmif(c, sm, 1, &name("test:no").to_le_bytes());
        record("lookup_missing", &[result as u64]);

        // Unregister frees the name.
        let (result, _, _) = send_cmif(c, sm, 3, &name("test:r").to_le_bytes());
        record("unregister", &[result as u64]);
        let (result, _, _) = send_cmif(c, sm, 1, &name("test:r").to_le_bytes());
        record("lookup_after_unregister", &[result as u64]);
        let (result, _, _) = send_cmif(c, sm, 3, &name("test:r").to_le_bytes());
        record("unregister_twice", &[result as u64]);

        // A light registration is looked up through the same command.
        let (result, _, handles) =
            send_cmif(c, sm, 2, &register_args_cmif(name("test:l"), 1, true));
        record("register_light", &[result as u64, handles[0] as u64]);
        let (result, _, handles) = send_tipc(c, sm, 1, &name("test:l").to_le_bytes());
        record("lookup_light", &[result as u64, handles[0] as u64]);

        // A host name cannot be taken over or shadowed.
        let (result, _, _) = send_cmif(c, sm, 2, &register_args_cmif(name("fsp-srv"), 1, false));
        record("register_host", &[result as u64]);

        c.halt();
    }

    #[test]
    fn guest_messages_register_look_up_and_unregister() {
        let _guard = REGISTRY_LOCK.lock().unwrap();
        reset_guest_services();
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: register_lookup,
        }]);

        assert_eq!(observed("register_tipc"), vec![0, 1]);
        assert_eq!(
            observed("register_duplicate"),
            vec![SM_ALREADY_REGISTERED as u64]
        );
        assert_eq!(observed("register_invalid"), vec![SM_INVALID_NAME as u64]);
        let lookup = observed("lookup_cmif");
        assert_eq!((lookup[0], lookup[1]), (0, 1));
        assert_ne!(lookup[2], 0, "the response carries a client session handle");
        let second = observed("lookup_second");
        assert_eq!(second[0], 0);
        assert_ne!(second[1], 0);
        assert_eq!(observed("lookup_limit"), vec![SM_MAX_SESSIONS as u64]);
        assert_eq!(observed("lookup_missing"), vec![SM_NOT_REGISTERED as u64]);
        assert_eq!(observed("unregister"), vec![0]);
        assert_eq!(
            observed("lookup_after_unregister"),
            vec![SM_NOT_REGISTERED as u64]
        );
        assert_eq!(observed("unregister_twice"), vec![SM_NOT_REGISTERED as u64]);
        assert_eq!(observed("register_light")[0], 0);
        assert_eq!(observed("lookup_light")[0], 0);
        assert_eq!(
            observed("register_host"),
            vec![SM_ALREADY_REGISTERED as u64]
        );
        reset_guest_services();
    }
}
