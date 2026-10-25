use std::io::Write;
use std::sync::Arc;

use crate::arbiter;
use crate::cpu::Cpu;
use crate::kernel::{
    ClientPort, CodeMemory, ConnectError, Event, GuestSession, KObject, NamedPortError, Port,
    ResourceLimit, Session, SharedMapError, SharedMemory, SharedMemoryRole, TransferMemory,
};
use crate::mem::{ADDR_MASK, MirrorError, PAGE, PhysicalError, perm, state};
use crate::process::Process;
use crate::services::sm::Sm;
use crate::threads;

pub const TICK_HZ: u64 = 19_200_000;

pub const RESULT_SUCCESS: u64 = 0;
pub const RESULT_NOT_IMPLEMENTED: u64 = 0x4201;
pub const RESULT_INVALID_SIZE: u64 = 0xca01;
pub const RESULT_INVALID_ENUM: u64 = 0xf001;
pub const RESULT_INVALID_ADDRESS: u64 = 0xcc01;
pub const RESULT_INVALID_MEMORY_REGION: u64 = 0xdc01;
pub const RESULT_LIMIT_REACHED: u64 = 0x10801;
pub const RESULT_INVALID_NEW_MEMORY_PERMISSION: u64 = 0xd801;
pub const RESULT_INVALID_POINTER: u64 = 0xe601;
pub const RESULT_BUSY: u64 = 0xf401;
pub const RESULT_TOO_MANY_HANDLES: u64 = 0xee01;
pub const RESULT_REMOTE_DEAD: u64 = 0xf601;

pub const ASLR_BASE: u64 = 0x0800_0000;
pub const ASLR_SIZE: u64 = 0x7f_f800_0000;
pub const HEAP_BASE: u64 = 0x2_0000_0000;
pub const HEAP_REGION_SIZE: u64 = 0x1_8000_0000;
pub const STACK_REGION_BASE: u64 = 0x4_0000_0000;
pub const STACK_REGION_SIZE: u64 = 0x1_0000_0000;
pub const ALIAS_BASE: u64 = 0x10_0000_0000;
pub const ALIAS_SIZE: u64 = 0x10_0000_0000;
pub const TOTAL_MEMORY: u64 = 0xcd50_0000;
const HEAP_ALIGN: u64 = 0x20_0000;

/// Guest regions a process asks the kernel about. A 32-bit process cannot see
/// anything above 4 GiB, so its heap, stack, TLS and alias live in the low half
/// of the address space; the A64 layout is unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub aslr_base: u64,
    pub aslr_size: u64,
    pub heap_base: u64,
    pub heap_region_size: u64,
    pub stack_base: u64,
    pub stack_region_size: u64,
    pub alias_base: u64,
    pub alias_size: u64,
    pub tls_base: u64,
}

pub const fn layout(arch: kirikae_loader::Arch) -> Layout {
    match arch {
        kirikae_loader::Arch::A64 => Layout {
            aslr_base: ASLR_BASE,
            aslr_size: ASLR_SIZE,
            heap_base: HEAP_BASE,
            heap_region_size: HEAP_REGION_SIZE,
            stack_base: STACK_REGION_BASE,
            stack_region_size: STACK_REGION_SIZE,
            alias_base: ALIAS_BASE,
            alias_size: ALIAS_SIZE,
            tls_base: STACK_REGION_BASE + 0x8000_0000,
        },
        kirikae_loader::Arch::A32 => Layout {
            aslr_base: ASLR_BASE,
            aslr_size: 0x1800_0000,
            heap_base: 0x2000_0000,
            heap_region_size: 0x2000_0000,
            stack_base: 0x4000_0000,
            stack_region_size: 0x2000_0000,
            alias_base: 0x8000_0000,
            alias_size: 0x8000_0000,
            tls_base: 0x6000_0000,
        },
    }
}

pub const MAIN_THREAD_HANDLE: u64 = 0x0001_0001;
pub const CURRENT_PROCESS: u64 = 0xffff_8001;

/// The HIPC message buffer at `tpidrro_el0` holds one 0x100-byte message.
const IPC_BUFFER_SIZE: u64 = 0x100;

fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KIRIKAE_TRACE_SVC").is_some())
}

pub fn svc(c: &mut Cpu, imm: u16) {
    let _prof = crate::prof::enter_svc(imm);
    crate::usage::svc(imm);
    crate::sched::kernel_enter(c);
    dispatch(c, imm);
    if trace() {
        eprintln!("svc {imm:#04x} -> x0={:#x} x1={:#x}", c.x[0], c.x[1]);
    }
    crate::sched::kernel_exit();
    crate::sched::reschedule(c);
    if c.process()
        .sched
        .exiting
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        c.halt();
    }
}

fn dispatch(c: &mut Cpu, imm: u16) {
    if trace() {
        let p = c.process();
        eprintln!(
            "svc {imm:#04x} from {} x0={:#x} x1={:#x} x2={:#x} x3={:#x}",
            p.describe(c.pc - 4),
            c.x[0],
            c.x[1],
            c.x[2],
            c.x[3]
        );
    }
    match imm {
        0x01 => set_heap_size(c),
        0x03 => ok(c),
        0x02 => set_memory_permission(c),
        0x04 => map_memory(c),
        0x05 => unmap_memory(c),
        0x06 => query_memory(c),
        0x07 => {
            c.process().terminate(0);
            c.halt();
        }
        0x08 => threads::create_thread(c),
        0x09 => threads::start_thread(c),
        0x0a => c.halt(),
        0x0b => threads::sleep_thread(c),
        0x0c => threads::get_priority(c),
        0x0d => threads::set_priority(c),
        0x0e => threads::get_core_mask(c),
        0x0f => threads::set_core_mask(c),
        0x10 => threads::current_processor(c),
        0x11 => event_signal(c, true),
        0x12 | 0x17 => event_signal(c, false),
        0x15 => create_transfer_memory(c),
        0x13 => map_shared_memory(c),
        0x14 => unmap_shared_memory(c),
        0x16 => close_handle(c),
        0x18 => threads::wait_synchronization(c),
        0x19 => threads::cancel_synchronization(c),
        0x1a => arbiter::arbitrate_lock(c),
        0x1b => arbiter::arbitrate_unlock(c),
        0x1c => arbiter::wait_process_wide_key(c),
        0x1d => arbiter::signal_process_wide_key(c),
        0x23 => send_async_request(c),
        0x24 => threads::get_process_id(c),
        0x25 => threads::get_thread_id(c),
        0x28 => return_from_exception(c),
        0x2e => get_debug_future_thread_info(c),
        0x32 => threads::set_thread_activity(c),
        0x33 => threads::get_thread_context3(c),
        0x34 => arbiter::wait_for_address(c),
        0x35 => arbiter::signal_to_address(c),
        0x37 => get_resource_limit_peak_value(c),
        0x40 => create_session(c),
        0x41 => accept_session(c),
        0x42 => reply_and_receive_light(c),
        0x43 => reply_and_receive(c, c.x[1], c.x[2], c.x[3] as u32, c.x[4] as i64),
        0x44 => reply_and_receive_with_user_buffer(c),
        0x45 => threads::create_event(c),
        0x4b => create_code_memory(c),
        0x4c => control_code_memory(c),
        0x4e => read_write_register(c),
        0x50 => create_shared_memory(c),
        0x51 => map_transfer_memory(c),
        0x52 => unmap_transfer_memory(c),
        0x53 => create_interrupt_event(c),
        0x54 => query_physical_address(c),
        0x5d..=0x5f => process_data_cache(c, imm),
        0x6f => get_system_info(c),
        0x70 => create_port(c),
        0x71 => manage_named_port(c),
        0x72 => connect_to_port(c),
        0x7c => get_process_info(c),
        0x7d => create_resource_limit(c),
        0x7e => set_resource_limit_limit_value(c),
        0x1f => connect_to_named_port(c),
        0x20 | 0x21 => send_sync_request(c),
        0x22 => c.x[0] = crate::ipc::send_sync_request(c, c.x[2] as u32, c.x[0], c.x[1]),
        0x1e => c.x[0] = c.process().ticks(),
        0x26 => break_(c),
        0x27 => output_debug_string(c),
        0x29 => get_info(c),
        0x2a => flush_entire_data_cache(c),
        0x2b => flush_data_cache(c),
        0x2c => map_physical_memory(c),
        0x2d => unmap_physical_memory(c),
        0x2f => get_last_thread_info(c),
        0x30 => get_resource_limit_value(c, false),
        0x31 => get_resource_limit_value(c, true),
        0x36 => synchronize_preemption_state(c),
        // IoPool/IoRegion (0x39 CreateIoPool, 0x3A CreateIoRegion, 0x46
        // MapIoRegion, 0x47 UnmapIoRegion) stay unimplemented on purpose. The
        // runtime has no device physical address space: `mem::map_physical`
        // commits zero-filled guest RAM in the alias region and charges it to
        // the memory budget, while device registers are modelled per block in
        // gpu/host1x. An IoRegion over the PCIe aperture (0x12000000-0x1FFFFFFF)
        // would therefore fabricate RAM instead of mapping device memory.
        //
        // For the same reason the unsafe pool-0 physical calls stay missing:
        // 0x48 MapPhysicalMemoryUnsafe and 0x49 UnmapPhysicalMemoryUnsafe would
        // hand the guest device pages that only exist as fabricated RAM, and 0x4A
        // SetUnsafeLimit grants nothing without them (the regular pool ceiling is
        // already LimitableResource_PhysicalMemoryMax).
        //
        // The device address space calls (0x55 QueryMemoryMapping, 0x56
        // CreateDeviceAddressSpace .. 0x5C UnmapDeviceAddressSpace) depend on
        // the same missing physical map and on an IOMMU, so they keep reporting
        // 0x4201 rather than handing out synthetic physical addresses.
        //
        // The debug (0x60-0x6D) and multi-process calls (0x4F, 0x73-0x7B, and
        // the 0x7F secure monitor) need debugger objects and a kernel process
        // table; the runtime has one process and no attached debugger.
        _ => {
            c.process().warn_once(format!(
                "svc {imm:#04x} is not implemented (first call from {})",
                c.process().describe(c.pc - 4)
            ));
            crate::usage::svc_missing(c, imm);
            c.x[0] = RESULT_NOT_IMPLEMENTED;
        }
    }
}

pub const MAIN_THREAD_PRIORITY: u64 = 44;
pub const MAIN_THREAD_ID: u64 = 0x51;

pub fn ok(c: &mut Cpu) {
    c.x[0] = RESULT_SUCCESS;
}

pub fn ok1(c: &mut Cpu, value: u64) {
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = value;
}

pub const RESULT_NOT_FOUND: u64 = 0xf201;

/// The kernel's port names are NUL-terminated strings of at most 0xb bytes
/// (`svcConnectToNamedPort` copies 12 bytes, `svcManageNamedPort` reads one).
const PORT_NAME_LEN: u64 = 12;

/// The name a port SVC takes from guest memory at `addr`. `None` means the
/// bytes are not readable or no NUL appears within the kernel's 12-byte limit.
fn port_name(p: &Process, addr: u64) -> Option<String> {
    let raw = p.memory.read(addr, PORT_NAME_LEN)?;
    let end = raw.iter().position(|&b| b == 0)?;
    Some(String::from_utf8_lossy(&raw[..end]).into_owned())
}

/// `svcConnectToNamedPort` (0x1F): a client session to the port registered
/// under the 12-byte name at X1. `sm:` is host-served and answers through the
/// runtime's own `Sm` service; every other name is a guest port registered with
/// `svcManageNamedPort`. A name nobody owns is the documented `0xf201`, and a
/// port at its `max_sessions` ceiling (or a process at its
/// `LimitableResource_Sessions` ceiling) reports `0x10801`.
fn connect_to_named_port(c: &mut Cpu) {
    let p = c.process();
    let Some(name) = port_name(p, c.x[1]) else {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    };
    p.log_ipc(format!("ConnectToNamedPort({name:?})"));
    if name == "sm:" {
        let session = Session::new(Arc::new(Sm));
        let handle = p.handles.lock().unwrap().insert(KObject::Session(session));
        ok1(c, handle as u64);
        return;
    }
    let Some(port) = crate::kernel::find_named_port(&name) else {
        p.warn_once(format!(
            "svcConnectToNamedPort({name:?}): no guest port is registered under that name"
        ));
        c.x[0] = RESULT_NOT_FOUND;
        return;
    };
    // A connection is a client session plus the server end queued on the port,
    // so it charges the limit the same way svcCreateSession does.
    let (used, ceiling) = session_usage(p);
    if used + 2 > ceiling {
        c.x[0] = RESULT_LIMIT_REACHED;
        return;
    }
    match port.connect() {
        Ok(link) => {
            let handle = p
                .handles
                .lock()
                .unwrap()
                .insert(KObject::Session(Session::guest_client(link)));
            p.log_ipc(format!("ConnectToNamedPort({name:?}) -> {handle:#x}"));
            ok1(c, handle as u64);
        }
        Err(ConnectError::Limit) => c.x[0] = RESULT_LIMIT_REACHED,
        Err(ConnectError::Closed) => c.x[0] = RESULT_NOT_FOUND,
    }
}

/// `svcCreatePort` (0x70): a server port with `max_sessions` connections
/// (`-1` unlimited) plus, in W2, its client end. W3 is `is_light` and X4 is the
/// port's numeric tag (not a string; `svcManageNamedPort` is what publishes a
/// name). The client end is a port, not a session, so it costs no
/// `LimitableResource_Sessions` until `svcConnectToPort` uses it.
fn create_port(c: &mut Cpu) {
    let (max_sessions, is_light, name) = (c.x[2] as i32, c.x[3], c.x[4]);
    if is_light > 1 {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    if !(-1..=crate::kernel::MAX_PORT_SESSIONS).contains(&max_sessions) {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    let p = c.process();
    let port = Port::with_wake(name, is_light == 1, max_sessions, move || {
        crate::sched::wake(p)
    });
    let client = ClientPort::new(port.clone());
    let (server, client) = {
        let mut table = p.handles.lock().unwrap();
        (
            table.insert(KObject::Port(port)),
            table.insert(KObject::ClientPort(client)),
        )
    };
    p.log_ipc(format!(
        "CreatePort(name={name:#x}, max_sessions={max_sessions}, light={is_light}) -> {server:#x}, {client:#x}"
    ));
    ok1(c, server as u64);
    c.x[2] = client as u64;
}

/// `svcManageNamedPort` (0x71): registers a new server port under the name at
/// X1 so `svcConnectToNamedPort` can find it, and returns it in W1. The name is
/// free again when the port handle is closed. `sm:` is host-served, so the
/// guest cannot take it; a name already registered is refused (the kernel does
/// not publish a result for it, the runtime reports `0xfa01`), and a full
/// named-port table reports `0x10801`.
fn manage_named_port(c: &mut Cpu) {
    let (name_ptr, max_sessions) = (c.x[1], c.x[2] as i32);
    if !(-1..=crate::kernel::MAX_PORT_SESSIONS).contains(&max_sessions) {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    let p = c.process();
    let Some(name) = port_name(p, name_ptr) else {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    };
    if name.is_empty() {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    if name == "sm:" {
        p.warn_once(
            "svcManageNamedPort(\"sm:\"): the runtime serves sm: itself, the name is taken".into(),
        );
        c.x[0] = RESULT_INVALID_STATE;
        return;
    }
    let port = Port::with_wake(0, false, max_sessions, move || crate::sched::wake(p));
    let handle = p
        .handles
        .lock()
        .unwrap()
        .insert(KObject::Port(port.clone()));
    match crate::kernel::register_named_port(&name, &port) {
        Ok(()) => {
            p.log_ipc(format!(
                "ManageNamedPort({name:?}, max_sessions={max_sessions}) -> {handle:#x}"
            ));
            ok1(c, handle as u64);
        }
        Err(error) => {
            // The registration is taken from the caller: drop the port it never
            // received rather than leaving an unnamed handle behind.
            p.handles.lock().unwrap().close(handle);
            let message = match error {
                NamedPortError::Taken => {
                    format!("svcManageNamedPort({name:?}): the name is already registered")
                }
                NamedPortError::Full => "svcManageNamedPort: the named-port table is full".into(),
            };
            p.warn_once(message);
            c.x[0] = match error {
                NamedPortError::Taken => RESULT_INVALID_STATE,
                NamedPortError::Full => RESULT_LIMIT_REACHED,
            };
        }
    }
}

/// `svcConnectToPort` (0x72): the client session for the client end of a port
/// (`svcCreatePort`'s W2). A closed port has nothing to accept the connection,
/// so it reports `0xf201`; the port's `max_sessions` ceiling and the process
/// session ceiling report `0x10801`.
fn connect_to_port(c: &mut Cpu) {
    let handle = c.x[1] as u32;
    let p = c.process();
    let Some(client) = p.handles.lock().unwrap().client_port(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    let (used, ceiling) = session_usage(p);
    if used + 2 > ceiling {
        c.x[0] = RESULT_LIMIT_REACHED;
        return;
    }
    match client.port().connect() {
        Ok(link) => {
            let session = p
                .handles
                .lock()
                .unwrap()
                .insert(KObject::Session(Session::guest_client(link)));
            p.log_ipc(format!("ConnectToPort(client {handle:#x}) -> {session:#x}"));
            ok1(c, session as u64);
        }
        Err(ConnectError::Limit) => c.x[0] = RESULT_LIMIT_REACHED,
        Err(ConnectError::Closed) => c.x[0] = RESULT_NOT_FOUND,
    }
}

pub const RESULT_TIMED_OUT: u64 = 0xea01;

fn send_sync_request(c: &mut Cpu) {
    // svcSendSyncRequestLight (0x20) sends a 7-word light message in x1-x7
    // without handles or buffers, while svcSendSyncRequest (0x21) uses the
    // HIPC buffer at tpidrro_el0. The runtime serves both from the HIPC buffer:
    // the light register form is not modelled, and every service the runtime
    // itself provides replies inline. A guest session pair made with
    // svcCreateSession goes through the rendezvous instead.
    c.x[0] = crate::ipc::send_sync_request(c, c.x[0] as u32, c.tpidrro_el0, IPC_BUFFER_SIZE);
}

/// `svcSendAsyncRequestWithUserBuffer` (0x23): the kernel copies the request in
/// the user buffer to the server, returns a readable event and signals it when
/// the reply lands in that same buffer.
///
/// Every service in the runtime is a host object that replies inline, so the
/// request is served through the synchronous IPC path before the event handle
/// is returned. A guest session pair blocks here until the guest server replies,
/// exactly as the synchronous path does. The observable client shape is the one
/// games use: send, wait for the event, parse the reply from the buffer. The
/// event is a plain `Event` (signaled at once), so it is also accepted by
/// `svcSignalEvent`/`svcClearEvent`, which a real readable event would refuse.
fn send_async_request(c: &mut Cpu) {
    let (addr, size, handle) = (c.x[1], c.x[2], c.x[3] as u32);
    if !addr.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    let p = c.process();
    if p.handles.lock().unwrap().session(handle).is_none() {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    if !p.memory.is_mapped(addr, size) {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    let result = crate::ipc::send_sync_request(c, handle, addr, size);
    if result != RESULT_SUCCESS {
        c.x[0] = result;
        return;
    }
    let event = Arc::new(Event {
        signaled: std::sync::atomic::AtomicBool::new(true),
    });
    let event_handle = c
        .process()
        .handles
        .lock()
        .unwrap()
        .insert(KObject::Event(event));
    ok1(c, event_handle as u64);
}

/// `svcCreateSession` (0x40): a session pair. The kernel takes the arguments as
/// W2 = is_light and X3 = name and returns the server handle in W1 and the
/// client handle in W2 (the library wrapper stores them in the caller's two out
/// pointers); the server handle is a port whose queue holds the pair until
/// `svcAcceptSession`.
fn create_session(c: &mut Cpu) {
    let (is_light, name) = (c.x[2], c.x[3]);
    if is_light > 1 {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    let p = c.process();
    let (used, ceiling) = session_usage(p);
    if used + 2 > ceiling {
        c.x[0] = RESULT_LIMIT_REACHED;
        return;
    }
    let link = GuestSession::with_wake(move || crate::sched::wake(p));
    let port = Port::new(name, is_light == 1, link.clone());
    let (server, client) = {
        let mut table = p.handles.lock().unwrap();
        (
            table.insert(KObject::Port(port)),
            table.insert(KObject::Session(Session::guest_client(link))),
        )
    };
    p.log_ipc(format!(
        "CreateSession(name={name:#x}, light={is_light}) -> {server:#x}, {client:#x}"
    ));
    ok1(c, server as u64);
    c.x[2] = client as u64;
}

/// Sessions and ports the process holds, and the `LimitableResource_Sessions`
/// ceiling of its resource limit (`u64::MAX` while the guest never asked for
/// one).
fn session_usage(p: &'static Process) -> (u64, u64) {
    let table = p.handles.lock().unwrap();
    let used = table.sessions_in_use();
    let ceiling = table
        .process_resource_limit()
        .and_then(|h| table.get(h))
        .and_then(|o| match o {
            KObject::ResourceLimit(limit) => limit.limit(4),
            _ => None,
        })
        .unwrap_or(u64::MAX);
    (used, ceiling)
}

/// `svcAcceptSession` (0x41): pops the next queued client of a port and returns
/// a server session handle in W1. With no connection waiting the kernel reports
/// `0xf201`, which is why a server waits on the port (`svcWaitSynchronization`
/// or the port entries of `svcReplyAndReceive`) before accepting. The session
/// ceiling was charged when `svcCreateSession` queued the pair, so accepting
/// moves the server end out of the port without accounting it twice.
fn accept_session(c: &mut Cpu) {
    let port_handle = c.x[1] as u32;
    let p = c.process();
    let Some(port) = p.handles.lock().unwrap().port(port_handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    let Some(link) = port.accept() else {
        c.x[0] = RESULT_NOT_FOUND;
        return;
    };
    let handle = p
        .handles
        .lock()
        .unwrap()
        .insert(KObject::Session(Session::guest_server(link)));
    p.log_ipc(format!(
        "AcceptSession(port {port_handle:#x}) -> {handle:#x}"
    ));
    ok1(c, handle as u64);
}

/// What a `svcReplyAndReceive` handle list entry waits for.
enum Waitable {
    /// A host service session: it replies inline and never has a request.
    Host,
    /// A server session of a guest pair.
    Guest(Arc<GuestSession>),
    /// A port with a connection waiting to be accepted.
    Port(Arc<Port>),
}

/// What ended a `ReplyAndReceive` wait: the index that signalled with an
/// optional request, or the result code and the index it applies to.
type Received = Result<(usize, Option<Vec<u8>>), (u64, usize)>;

/// Handles for `svcReplyAndReceive` (0x43) and its user-buffer form (0x44).
/// Horizon validates the whole list before waiting, so an object that is not a
/// session or a port is `0xe401` even when another entry is already signalled.
fn read_reply_handles(c: &mut Cpu, addr: u64, count: u64) -> Option<Vec<(u32, Waitable)>> {
    if count > 0x40 {
        c.x[0] = RESULT_TOO_MANY_HANDLES;
        return None;
    }
    let p = c.process();
    let Some(raw) = p.memory.read(addr, count * 4) else {
        c.x[0] = RESULT_INVALID_POINTER;
        return None;
    };
    let handles: Vec<u32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let table = p.handles.lock().unwrap();
    let mut waitable = Vec::with_capacity(handles.len());
    for handle in handles {
        match table.get(handle) {
            Some(KObject::Session(s)) => {
                let entry = match s.guest_server_link() {
                    Some(link) => Waitable::Guest(link),
                    None => Waitable::Host,
                };
                waitable.push((handle, entry));
            }
            Some(KObject::Port(port)) => waitable.push((handle, Waitable::Port(port))),
            _ => {
                c.x[0] = RESULT_INVALID_HANDLE;
                return None;
            }
        }
    }
    Some(waitable)
}

/// Sends the reply the calling thread has in `buffer` to the session in
/// service, if any. A host session replies inline, so there is nothing to send.
fn send_reply(c: &Cpu, session: &Arc<Session>, buffer: (u64, u64)) {
    let Some(link) = session.guest_server_link() else {
        return;
    };
    if !link.in_service() {
        return;
    }
    let p = c.process();
    let Some(bytes) = p.memory.read(buffer.0, buffer.1) else {
        p.warn_once(format!(
            "ReplyAndReceive: the reply buffer {:#x} is not mapped",
            buffer.0
        ));
        return;
    };
    // The pair wakes the process itself, so a host service posting from its own
    // thread reaches the blocked client too.
    link.post_reply(bytes);
}

/// Where `svcReplyAndReceive` receives: the TLS message buffer or the user
/// buffer of the 0x44 form.
#[derive(Clone, Copy)]
struct ReceiveBuffer {
    addr: u64,
    size: u64,
}

impl ReceiveBuffer {
    fn tls(c: &Cpu) -> ReceiveBuffer {
        ReceiveBuffer {
            addr: c.tpidrro_el0,
            size: IPC_BUFFER_SIZE,
        }
    }
}

fn reply_and_receive(c: &mut Cpu, addr: u64, count: u64, target: u32, timeout: i64) {
    let Some(handles) = read_reply_handles(c, addr, count) else {
        return;
    };
    if !send_reply_to_target(c, target, (c.tpidrro_el0, IPC_BUFFER_SIZE)) {
        return;
    }
    receive_request(c, handles, ReceiveBuffer::tls(c), timeout);
}

fn reply_and_receive_with_user_buffer(c: &mut Cpu) {
    let (addr, size, list, count, target, timeout) =
        (c.x[1], c.x[2], c.x[3], c.x[4], c.x[5] as u32, c.x[6] as i64);
    if !addr.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    if !c.process().memory.is_mapped(addr, size) {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    let Some(handles) = read_reply_handles(c, list, count) else {
        return;
    };
    if !send_reply_to_target(c, target, (addr, size)) {
        return;
    }
    receive_request(c, handles, ReceiveBuffer { addr, size }, timeout);
}

/// Sends the pending reply to `target` when it is not zero. `false` means the
/// target handle is not a session and the SVC already stored its result.
fn send_reply_to_target(c: &mut Cpu, target: u32, buffer: (u64, u64)) -> bool {
    if target == 0 {
        return true;
    }
    // The handle table is not held across the wakeup: a thread blocked in
    // `sched::block` holds the scheduler lock and may touch the table.
    let session = c.process().handles.lock().unwrap().session(target);
    match session {
        Some(session) => {
            send_reply(c, &session, buffer);
            true
        }
        None => {
            c.x[0] = RESULT_INVALID_HANDLE;
            false
        }
    }
}

/// `svcReplyAndReceiveLight` (0x42): replies on `handle` and receives the next
/// request on it, with no timeout. A port handle only waits until a connection
/// is ready to accept. The light message lives in registers on hardware; the
/// runtime keeps both ends on the HIPC buffer, so a light pair behaves like a
/// synchronous one.
fn reply_and_receive_light(c: &mut Cpu) {
    let handle = c.x[0] as u32;
    let p = c.process();
    let entry = p.handles.lock().unwrap().get(handle);
    let waitable = match entry {
        Some(KObject::Session(session)) => {
            let Some(link) = session.guest_server_link() else {
                // The argument is a port or a server session; a client session
                // has nothing to receive.
                c.x[0] = RESULT_INVALID_HANDLE;
                return;
            };
            send_reply(c, &session, (c.tpidrro_el0, IPC_BUFFER_SIZE));
            Waitable::Guest(link)
        }
        Some(KObject::Port(port)) => Waitable::Port(port),
        _ => {
            c.x[0] = RESULT_INVALID_HANDLE;
            return;
        }
    };
    receive_request(c, vec![(handle, waitable)], ReceiveBuffer::tls(c), -1);
}

/// Waits for a request (or a pending connection) on `handles` and copies it into
/// `buffer`. Horizon enumerates the list in order: an incoming message wins over
/// a closed handle that appears later, and the index of the entry that ended the
/// wait is returned in W1. A session whose peer closed is `0xf601` with its
/// index; nothing else can end the wait, since the runtime has no host server
/// that messages a guest.
fn receive_request(
    c: &mut Cpu,
    handles: Vec<(u32, Waitable)>,
    buffer: ReceiveBuffer,
    timeout: i64,
) {
    let p = c.process();
    let me = c.thread.clone();
    let lr = c.x[30];
    let ids: Vec<u32> = handles.iter().map(|(h, _)| *h).collect();
    let what = || format!("ReplyAndReceive({ids:x?}) from {}", p.describe(lr));
    let ended: Option<Received> = crate::sched::block(c, timeout, what, |_| {
        if me
            .cancelled
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            return Some(Err((RESULT_CANCELLED, 0)));
        }
        for (i, (_, waitable)) in handles.iter().enumerate() {
            match waitable {
                Waitable::Guest(link) => {
                    if let Some(bytes) = link.take_request() {
                        return Some(Ok((i, Some(bytes))));
                    }
                    if !link.client_open() {
                        return Some(Err((RESULT_REMOTE_DEAD, i)));
                    }
                }
                Waitable::Port(port) => {
                    if port.pending() > 0 {
                        return Some(Ok((i, None)));
                    }
                    // A closed port cannot produce a connection, so a server
                    // waiting on it is released instead of parking forever.
                    if port.closed() {
                        return Some(Err((RESULT_REMOTE_DEAD, i)));
                    }
                }
                Waitable::Host => {}
            }
        }
        None
    });
    match ended {
        Some(Ok((index, request))) => {
            if let Some(bytes) = request {
                let n = bytes.len().min(buffer.size as usize);
                if !p.memory.write(buffer.addr, &bytes[..n]) {
                    c.x[0] = RESULT_INVALID_POINTER;
                    return;
                }
            }
            c.x[0] = RESULT_SUCCESS;
            c.x[1] = index as u64;
        }
        Some(Err((result, index))) => {
            c.x[0] = result;
            c.x[1] = index as u64;
        }
        None => c.x[0] = RESULT_TIMED_OUT,
    }
}

/// `svcGetDebugFutureThreadInfo` (0x2E) returns the schedule context in x1-x4
/// and the thread id in x5, which is the register block libnx and the SDK
/// unpack. The runtime schedules no future thread, so the debug target is
/// reported; `svcGetLastThreadInfo` (0x2F) keeps reporting the caller. This is
/// the 6.0.0+ layout (X2 is a debug handle); the 5.0.0-5.1.0
/// `GetFutureThreadInfo` form is not distinguished from it.
fn get_debug_future_thread_info(c: &mut Cpu) {
    let handle = c.x[2] as u32;
    let Some(t) = threads::thread_of(c, handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    let (fp, sp, lr, pc) = (c.x[29], c.sp, c.x[30], c.pc);
    ok1(c, fp);
    c.x[2] = sp;
    c.x[3] = lr;
    c.x[4] = pc;
    c.x[5] = t.id;
}

pub const RESULT_INVALID_HANDLE: u64 = 0xe401;
pub const RESULT_CANCELLED: u64 = 0xec01;
pub const RESULT_INVALID_PRIORITY: u64 = 0xe001;
pub const RESULT_INVALID_CORE: u64 = 0xe201;
pub const RESULT_INVALID_COMBINATION: u64 = 0xe801;
pub const RESULT_INVALID_STATE: u64 = 0xfa01;

fn event_signal(c: &mut Cpu, signaled: bool) {
    c.x[0] = if c.process().set_event(c.x[0] as u32, signaled) {
        RESULT_SUCCESS
    } else {
        RESULT_INVALID_HANDLE
    };
}

fn close_handle(c: &mut Cpu) {
    let handle = c.x[0] as u32;
    let p = c.process();
    let closed = {
        let mut table = p.handles.lock().unwrap();
        let object = table.get(handle);
        let closed = table.close(handle);
        drop(table);
        // Closing one end of a guest session pair wakes the peer: a client
        // waiting for a reply learns the server is gone and the other way
        // round, so neither side can block on a dead peer. The pair does the
        // wake itself, so a host service reaches a blocked guest too.
        match object {
            Some(KObject::Session(session)) if session.is_guest() => {
                session.close_guest_end();
            }
            Some(KObject::Port(port)) => {
                port.close();
                crate::sched::wake(p);
            }
            _ => {}
        }
        closed
    };
    if !closed && handle < 0xffff_8000 {
        p.log_ipc(format!("CloseHandle({handle:#x}) of an unknown handle"));
    }
    ok(c);
}

fn set_heap_size(c: &mut Cpu) {
    let size = c.x[1];
    let layout = c.process().layout;
    if !size.is_multiple_of(HEAP_ALIGN) || size > layout.heap_region_size {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    let p = c.process();
    let mut heap = p.heap.lock().unwrap();
    if size > *heap {
        if p.memory
            .map_as(
                layout.heap_base + *heap,
                size - *heap,
                state::HEAP,
                perm::RW,
            )
            .is_err()
        {
            drop(heap);
            fatal(c, format!("could not map a {size:#x}-byte heap"));
            return;
        }
        *heap = size;
    }
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = layout.heap_base;
}

pub const RESULT_INVALID_MEMORY_STATE: u64 = 0xd401;

/// `svcMapMemory` (0x04): maps `src` at `dst`. Horizon borrows the source
/// pages, so the source becomes read-only to the guest until `svcUnmapMemory`.
/// The runtime keeps one host view of the pages and copies them at both ends
/// (`mem::GuestMemory::map_mirror`); the destination is reported as the stack
/// mapping the kernel makes (both states share the value) and the source is
/// refused as a source twice.
fn map_memory(c: &mut Cpu) {
    let (dst, src, size) = (c.x[0], c.x[1], c.x[2]);
    let result = match c.process().memory.map_mirror(dst, src, size) {
        Ok(()) => RESULT_SUCCESS,
        Err(MirrorError::Size) => RESULT_INVALID_SIZE,
        Err(MirrorError::Address) => RESULT_INVALID_ADDRESS,
        Err(MirrorError::State) => RESULT_INVALID_MEMORY_STATE,
    };
    if result != RESULT_SUCCESS {
        let p = c.process();
        let s = p.memory.query(src);
        let d = p.memory.query(dst);
        p.warn_once(format!(
            "MapMemory({dst:#x}, {src:#x}, {size:#x}) failed {result:#x}:              src state {:#x} perm {:#x} in {:#x}..{:#x}, dst state {:#x} perm {:#x}",
            s.state, s.perm, s.range.start, s.range.end, d.state, d.perm
        ));
    }
    c.x[0] = result;
}

fn unmap_memory(c: &mut Cpu) {
    let (dst, src, size) = (c.x[0], c.x[1], c.x[2]);
    c.x[0] = match c.process().memory.unmap_mirror(dst, src, size) {
        Ok(()) => RESULT_SUCCESS,
        Err(MirrorError::Size) => RESULT_INVALID_SIZE,
        Err(MirrorError::Address) => RESULT_INVALID_ADDRESS,
        Err(MirrorError::State) => RESULT_INVALID_MEMORY_STATE,
    };
}

/// The memory states a transfer memory may take its pages from, and that a
/// mirror may borrow. Horizon gates these behind state bits
/// (`MemState_TransferMemAllowed`, `MemState_MapAllowed`); the runtime keeps the
/// equivalent list.
const LENDABLE: [u32; 4] = [state::NORMAL, state::HEAP, state::STACK, state::SHARED];

/// The permissions a transfer memory may have: no execute (the kernel refuses
/// it) and no write-only.
fn transfer_perm(perm: u32) -> bool {
    matches!(perm, perm::NONE | perm::R | perm::RW)
}

/// The permissions a shared memory block may have: data only, so no execute.
fn shared_perm(permission: u32) -> bool {
    matches!(permission, perm::NONE | perm::R | perm::RW)
}

/// `svcCreateSharedMemory` (0x50): a block of `size` zeroed bytes whose creator
/// maps it with `owner_perm` and whose borrower maps it with `remote_perm`
/// (`0x10000000` accepts any permission). The runtime runs one process, so the
/// role lives on the handle; the block itself keeps both permissions.
fn create_shared_memory(c: &mut Cpu) {
    let (size, owner_perm, remote_perm) = (c.x[1], c.x[2] as u32, c.x[3] as u32);
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    if !shared_perm(owner_perm) {
        c.x[0] = RESULT_INVALID_NEW_MEMORY_PERMISSION;
        return;
    }
    if remote_perm != crate::kernel::MEMORY_PERMISSION_DONT_CARE && !shared_perm(remote_perm) {
        c.x[0] = RESULT_INVALID_NEW_MEMORY_PERMISSION;
        return;
    }
    let p = c.process();
    // The kernel allocates shared memory from the process pool partition, so a
    // block that does not fit the remaining budget is the memory limit result.
    if size > TOTAL_MEMORY.saturating_sub(p.used_memory()) {
        c.x[0] = RESULT_LIMIT_REACHED;
        return;
    }
    let Some(memory) = SharedMemory::try_new(size, owner_perm, remote_perm) else {
        p.warn_once("svcCreateSharedMemory: the host refused the block's backing store".into());
        c.x[0] = RESULT_LIMIT_REACHED;
        return;
    };
    let handle = p
        .handles
        .lock()
        .unwrap()
        .insert_shared_memory(Arc::new(memory), SharedMemoryRole::Owner);
    p.log_ipc(format!(
        "CreateSharedMemory(size={size:#x}, owner={owner_perm:#x}, remote={remote_perm:#x}) -> {handle:#x}"
    ));
    ok1(c, handle as u64);
}

/// `svcMapSharedMemory` (0x13): maps the block at `addr` with `perm`. The size
/// must be the block's size and the permission the one the caller's side was
/// created with, which is how a creator and a borrower differ.
fn map_shared_memory(c: &mut Cpu) {
    let (handle, addr, size, perm) = (c.x[0] as u32, c.x[1], c.x[2], c.x[3] as u32);
    let p = c.process();
    let Some((memory, role)) = p.handles.lock().unwrap().shared_memory(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    c.x[0] = match memory.map(&p.memory, addr, size, perm, role) {
        Ok(()) => RESULT_SUCCESS,
        Err(SharedMapError::Size) => RESULT_INVALID_SIZE,
        Err(SharedMapError::Address) => RESULT_INVALID_ADDRESS,
        Err(SharedMapError::Permission) => RESULT_INVALID_NEW_MEMORY_PERMISSION,
        Err(SharedMapError::State) => RESULT_INVALID_MEMORY_STATE,
    };
}

/// `svcUnmapSharedMemory` (0x14): writes the mapping back into the block and
/// releases the pages. The size must be the block's size and the address the
/// one it was mapped at.
fn unmap_shared_memory(c: &mut Cpu) {
    let (handle, addr, size) = (c.x[0] as u32, c.x[1], c.x[2]);
    let p = c.process();
    let Some((memory, _)) = p.handles.lock().unwrap().shared_memory(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    c.x[0] = match memory.unmap(&p.memory, addr, size) {
        Ok(()) => RESULT_SUCCESS,
        Err(SharedMapError::Size) => RESULT_INVALID_SIZE,
        Err(SharedMapError::Address) => RESULT_INVALID_ADDRESS,
        Err(SharedMapError::Permission) => RESULT_INVALID_NEW_MEMORY_PERMISSION,
        Err(SharedMapError::State) => RESULT_INVALID_MEMORY_STATE,
    };
}

/// `svcCreateTransferMemory` (0x15): reprotects `addr..addr+size` with `perm`
/// and marks it as a transfer memory, then returns the object in X1. The kernel
/// keeps the pages with the creator, so they stay mapped and only their state
/// and permission change; `svcMapTransferMemory` aliases them elsewhere.
fn create_transfer_memory(c: &mut Cpu) {
    let (addr, size, perm) = (c.x[1], c.x[2], c.x[3] as u32);
    if !addr.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    let Some(end) = addr.checked_add(size) else {
        c.x[0] = RESULT_INVALID_MEMORY_REGION;
        return;
    };
    if !transfer_perm(perm) {
        c.x[0] = RESULT_INVALID_NEW_MEMORY_PERMISSION;
        return;
    }
    let p = c.process();
    if !p.memory.retag(addr, end, state::TRANSFER, perm, &LENDABLE) {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    }
    let tmem = TransferMemory::new(addr, size, perm);
    let handle = p
        .handles
        .lock()
        .unwrap()
        .insert(KObject::TransferMemory(tmem));
    ok1(c, handle as u64);
}

/// `svcMapTransferMemory` (0x51): maps the block at `addr`. The size and the
/// permission must be the ones the block was created with, and only one mapping
/// exists at a time.
fn map_transfer_memory(c: &mut Cpu) {
    let (handle, addr, size, perm) = (c.x[0] as u32, c.x[1], c.x[2], c.x[3] as u32);
    let p = c.process();
    let Some(tmem) = p.handles.lock().unwrap().transfer_memory(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    c.x[0] = if size == 0 || size != tmem.size {
        RESULT_INVALID_SIZE
    } else if !addr.is_multiple_of(PAGE) {
        RESULT_INVALID_ADDRESS
    } else if addr.checked_add(size).is_none() {
        RESULT_INVALID_MEMORY_REGION
    } else if perm != tmem.perm {
        RESULT_INVALID_NEW_MEMORY_PERMISSION
    } else if tmem.mapped_at().is_some() {
        RESULT_INVALID_STATE
    } else if tmem.map(&p.memory, addr, perm) {
        RESULT_SUCCESS
    } else {
        RESULT_INVALID_MEMORY_STATE
    };
}

/// `svcUnmapTransferMemory` (0x52): writes the mapping back to the creator's
/// pages and drops it.
fn unmap_transfer_memory(c: &mut Cpu) {
    let (handle, addr, size) = (c.x[0] as u32, c.x[1], c.x[2]);
    let p = c.process();
    let Some(tmem) = p.handles.lock().unwrap().transfer_memory(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    c.x[0] = if size != tmem.size {
        RESULT_INVALID_SIZE
    } else if !addr.is_multiple_of(PAGE) {
        RESULT_INVALID_ADDRESS
    } else if tmem.unmap(&p.memory, addr) {
        RESULT_SUCCESS
    } else {
        RESULT_INVALID_STATE
    };
}

/// `svcCreateInterruptEvent` (0x53). The event object is real, but the runtime
/// has no interrupt controller wired to guest IRQs, so nothing ever signals it;
/// a guest that waits on it blocks like it would on an interrupt that never
/// fires.
fn create_interrupt_event(c: &mut Cpu) {
    let (irq, flag) = (c.x[1], c.x[2] as u32);
    if flag > 1 {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    c.process()
        .warn_once(format!(
            "svcCreateInterruptEvent(irq {irq}): the runtime has no interrupt source, the event never signals"
        ));
    let event = Arc::new(Event {
        signaled: std::sync::atomic::AtomicBool::new(false),
    });
    let handle = c
        .process()
        .handles
        .lock()
        .unwrap()
        .insert(KObject::Event(event));
    ok1(c, handle as u64);
}

/// `svcCreateCodeMemory` (0x4B): locks `src..src+size` into a code memory
/// object, which leaves the process and starts `0xFF`-filled, as the kernel
/// does.
fn create_code_memory(c: &mut Cpu) {
    let (src, size) = (c.x[1], c.x[2]);
    let p = c.process();
    if !src.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    let Some(end) = src.checked_add(size) else {
        c.x[0] = RESULT_INVALID_MEMORY_REGION;
        return;
    };
    let source = p.memory.query(src);
    if source.range.start != src
        || source.range.end < end
        || !matches!(source.state, state::NORMAL | state::HEAP)
    {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    }
    if p.memory.unmap(src, size).is_err() {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    }
    let memory = CodeMemory::new(size);
    let handle = p
        .handles
        .lock()
        .unwrap()
        .insert(KObject::CodeMemory(memory));
    ok1(c, handle as u64);
}

/// `svcControlCodeMemory` (0x4C). Only the owner mapping exists in the runtime:
/// there is one process, so `CodeMemoryOperation_MapSlave` has no second address
/// space to map into and, since 5.0.0, the kernel reports `0xe401` when the
/// owner is the calling process. A JIT maps the range RW, fills it and maps it
/// RX, which is why both are accepted for `MapOwner`; the documented `MapSlave`
/// permissions (R or RX) are validated first so the guest sees the right error.
fn control_code_memory(c: &mut Cpu) {
    let (handle, op, dst, size, perm) = (c.x[0] as u32, c.x[1], c.x[2], c.x[3], c.x[4] as u32);
    let p = c.process();
    let Some(memory) = p.handles.lock().unwrap().code_memory(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    if size != memory.size {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    if !dst.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    c.x[0] = match op {
        0 => {
            if !matches!(perm, perm::RW | perm::RX) {
                RESULT_INVALID_NEW_MEMORY_PERMISSION
            } else if memory.map_owner(&p.memory, dst, perm) {
                RESULT_SUCCESS
            } else {
                RESULT_INVALID_MEMORY_STATE
            }
        }
        1 => {
            if !matches!(perm, perm::R | perm::RX) {
                RESULT_INVALID_NEW_MEMORY_PERMISSION
            } else {
                p.warn_once(
                    "svcControlCodeMemory(MapSlave): the runtime has one process, so the owner is the caller"
                        .into(),
                );
                RESULT_INVALID_HANDLE
            }
        }
        2 => {
            if memory.unmap_owner(&p.memory, dst) {
                RESULT_SUCCESS
            } else {
                RESULT_INVALID_STATE
            }
        }
        3 => RESULT_INVALID_STATE,
        _ => RESULT_INVALID_ENUM,
    };
}

/// `svcInvalidateProcessDataCache`, `svcStoreProcessDataCache` and
/// `svcFlushProcessDataCache` (0x5D-0x5F). The host caches are coherent with
/// the guest, so only the process handle and the range are validated; the
/// runtime has no handle for another process.
fn process_data_cache(c: &mut Cpu, imm: u16) {
    let (handle, addr, size) = (c.x[0] as u32, c.x[1], c.x[2]);
    let p = c.process();
    if handle != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    c.x[0] = if size == 0 {
        RESULT_SUCCESS
    } else if addr.checked_add(size).is_none_or(|end| end > ADDR_MASK + 1)
        || !p.memory.is_mapped(addr, size)
    {
        p.warn_once(format!(
            "svc {imm:#04x} on {addr:#x}+{size:#x} which is not mapped"
        ));
        RESULT_INVALID_MEMORY_STATE
    } else {
        RESULT_SUCCESS
    };
}

fn set_memory_permission(c: &mut Cpu) {
    let (addr, size, new) = (c.x[0], c.x[1], c.x[2] as u32);
    c.x[0] = if !addr.is_multiple_of(PAGE) {
        RESULT_INVALID_ADDRESS
    } else if size == 0 || !size.is_multiple_of(PAGE) {
        RESULT_INVALID_SIZE
    } else if addr.checked_add(size).is_none() {
        RESULT_INVALID_MEMORY_REGION
    } else if ![perm::NONE, perm::R, perm::RW].contains(&new) {
        RESULT_INVALID_NEW_MEMORY_PERMISSION
    } else if c.process().memory.protect(
        addr,
        addr + size,
        new,
        &[
            state::HEAP,
            state::CODE_MUTABLE,
            state::CODE_WRITABLE,
            state::CODE_READONLY,
        ],
    ) {
        RESULT_SUCCESS
    } else {
        RESULT_INVALID_MEMORY_STATE
    };
}

fn physical_range(c: &mut Cpu) -> Option<(u64, u64)> {
    let (addr, size) = (c.x[0], c.x[1]);
    let layout = c.process().layout;
    let error = if !addr.is_multiple_of(PAGE) {
        RESULT_INVALID_ADDRESS
    } else if size == 0 || !size.is_multiple_of(PAGE) {
        RESULT_INVALID_SIZE
    } else if addr.checked_add(size).is_none() {
        RESULT_INVALID_MEMORY_REGION
    } else if c.process().system_resource_size == 0 {
        RESULT_INVALID_STATE
    } else if addr < layout.alias_base || addr + size > layout.alias_base + layout.alias_size {
        RESULT_INVALID_MEMORY_REGION
    } else {
        return Some((addr, addr + size));
    };
    c.x[0] = error;
    None
}

fn physical_result(
    c: &mut Cpu,
    name: &str,
    r: Result<u64, PhysicalError>,
    update: impl FnOnce(u64),
) {
    c.x[0] = match r {
        Ok(bytes) => {
            update(bytes);
            RESULT_SUCCESS
        }
        Err(PhysicalError::State) => RESULT_INVALID_MEMORY_STATE,
        Err(PhysicalError::Limit) => RESULT_LIMIT_REACHED,
        Err(PhysicalError::Host(e)) => {
            return fatal(c, format!("{name}: the host refused the mapping ({e})"));
        }
    };
}

fn map_physical_memory(c: &mut Cpu) {
    let Some((start, end)) = physical_range(c) else {
        return;
    };
    let p = c.process();
    let budget = TOTAL_MEMORY.saturating_sub(p.used_memory());
    let r = p.memory.map_physical(start, end, budget);
    physical_result(c, "svcMapPhysicalMemory", r, |n| {
        p.physical
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    });
}

fn unmap_physical_memory(c: &mut Cpu) {
    let Some((start, end)) = physical_range(c) else {
        return;
    };
    let p = c.process();
    let r = p.memory.unmap_physical(start, end);
    physical_result(c, "svcUnmapPhysicalMemory", r, |n| {
        p.physical
            .fetch_sub(n, std::sync::atomic::Ordering::Relaxed);
    });
}

/// `svcQueryPhysicalAddress` (0x54): the kernel returns the physical address of
/// the page, the virtual base of the mapping that contains it and the mapping's
/// size in X1-X3 (the SDK wrapper stores them in a `PhysicalMemoryInfo`).
///
/// The runtime keeps one host reservation for all guest RAM and has no second
/// physical address space, so a guest page's physical address is its virtual
/// address: the guest RAM the guest sees is the same flat space the runtime's
/// device models (GPU, host1x) consume. The address must be mapped; there is no
/// physical page to report for a hole, which is the documented invalid-address
/// result.
fn query_physical_address(c: &mut Cpu) {
    let addr = c.x[1];
    let p = c.process();
    let region = p.memory.query(addr);
    if region.state == state::FREE || addr > ADDR_MASK {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = addr & !(PAGE - 1);
    c.x[2] = region.range.start;
    c.x[3] = region.range.end - region.range.start;
}

/// The MC/MC0/MC1 register offsets `svcReadWriteRegister` accepts. The three
/// pages share the whitelist (Switchbrew's table, including the 2.0.0+
/// additions).
const MC_REGISTERS: &[u16] = &[
    0x054, 0x090, 0x094, 0x098, 0x09c, 0x0a0, 0x0a4, 0x0a8, 0x0ac, 0x0b0, 0x0b4, 0x0b8, 0x0bc,
    0x0c0, 0x0c4, 0x0c8, 0x0d0, 0x0d4, 0x0d8, 0x0dc, 0x0e0, 0x100, 0x108, 0x10c, 0x118, 0x11c,
    0x124, 0x128, 0x12c, 0x130, 0x134, 0x138, 0x13c, 0x158, 0x15c, 0x164, 0x168, 0x16c, 0x170,
    0x174, 0x178, 0x17c, 0x200, 0x204, 0x2e4, 0x2e8, 0x2ec, 0x2f4, 0x2f8, 0x310, 0x314, 0x320,
    0x328, 0x344, 0x348, 0x370, 0x374, 0x37c, 0x380, 0x390, 0x394, 0x398, 0x3ac, 0x3b8, 0x3bc,
    0x3c0, 0x3c4, 0x3d8, 0x3e8, 0x41c, 0x420, 0x424, 0x428, 0x42c, 0x430, 0x44c, 0x47c, 0x480,
    0x484, 0x4c4, 0x4c8, 0x4cc, 0x50c, 0x554, 0x558, 0x55c, 0x584, 0x588, 0x58c, 0x670, 0x674,
    0x690, 0x694, 0x698, 0x69c, 0x6a0, 0x6a4, 0x6c0, 0x6c4, 0x6f0, 0x6f4, 0x960, 0x970, 0x974,
    0xa20, 0xa24, 0xb88, 0xb8c, 0xbc4, 0xbc8, 0xbcc, 0xbd0, 0xbd4, 0xbd8, 0xbdc, 0xbe0, 0xbe4,
    0xbe8, 0xbec, 0xc00, 0xc5c, 0xcac,
];

/// `svcReadWriteRegister` (0x4E): the only addresses the kernel accepts are the
/// Memory Controller pages (0x70019000, 0x7001C000, 0x7001D000) at one of the
/// documented register offsets, and, since 2.0.0, the PMC block
/// (0x7000E400..0x7000F000), 4-byte aligned. The runtime has no MC/PMC model:
/// device registers are modelled per block in gpu/host1x and none of those
/// blocks exposes the controller state, so a register read or write is refused
/// rather than answered with a made-up value. An address outside the documented
/// registers is the kernel's invalid-address result.
fn read_write_register(c: &mut Cpu) {
    let addr = c.x[1];
    let page = addr & !(PAGE - 1);
    let offset = (addr & (PAGE - 1)) as u16;
    let memory_controller = matches!(page, 0x7001_9000 | 0x7001_c000 | 0x7001_d000)
        && MC_REGISTERS.binary_search(&offset).is_ok();
    // The PMC block skips the whitelist and asks the secure monitor, which the
    // runtime does not have either.
    let pmc = (0x7000_e400..0x7000_f000).contains(&addr);
    if !addr.is_multiple_of(4) || (!memory_controller && !pmc) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    c.process().warn_once(format!(
        "svcReadWriteRegister({addr:#x}): the runtime has no Memory Controller model"
    ));
    c.x[0] = RESULT_NOT_IMPLEMENTED;
}

fn query_memory(c: &mut Cpu) {
    let (out, addr) = (c.x[0], c.x[2]);
    let r = c.process().memory.query(addr);
    let mut info = [0u8; 0x28];
    info[0..8].copy_from_slice(&r.range.start.to_le_bytes());
    info[8..16].copy_from_slice(&(r.range.end - r.range.start).to_le_bytes());
    info[16..20].copy_from_slice(&r.state.to_le_bytes());
    info[24..28].copy_from_slice(&r.perm.to_le_bytes());
    if !c.process().memory.write(out, &info) {
        // A guest that hands the kernel an unmapped output pointer gets the
        // documented result code; it is guest data, not a host failure.
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = 0;
}

fn get_info(c: &mut Cpu) {
    let (id, subid) = (c.x[1], c.x[3]);
    let p = c.process();
    let layout = p.layout;
    let value = match id {
        0 => 0b1111,
        1 => 0xfc00_0000_0000_0000,
        2 => layout.alias_base,
        3 => layout.alias_size,
        4 => layout.heap_base,
        5 => layout.heap_region_size,
        6 => TOTAL_MEMORY,
        7 => p.used_memory(),
        8 => 0,
        9 => resource_limit_handle(p),
        10 => 0,
        11 => p.entropy(subid),
        12 => layout.aslr_base,
        13 => layout.aslr_size,
        14 => layout.stack_base,
        15 => layout.stack_region_size,
        16 => p.system_resource_size,
        17 => (p.physical.load(std::sync::atomic::Ordering::Relaxed) / 512)
            .next_multiple_of(PAGE)
            .min(p.system_resource_size),
        18 => p.title_id,
        // InfoType_UserExceptionContextAddress: the exception frame lives at
        // the start of the thread's TLS region, which is TPIDRRO_EL0.
        20 => c.tpidrro_el0,
        21 => TOTAL_MEMORY - p.system_resource_size,
        22 => p.used_memory() - p.system_resource_size,
        23 => 1,
        24 => 0x60,
        26 => 1,
        28 => 0,
        _ => {
            p.warn_once(format!(
                "svcGetInfo id {id} (subid {subid}) is not implemented"
            ));
            c.x[0] = RESULT_INVALID_ENUM;
            return;
        }
    };
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = value;
}

/// `svcGetSystemInfo` (0x6F), the 5.0.0+ system-wide counterpart of
/// `svcGetInfo`. X1 is the `SystemInfoType`, X2 is unused (the kernel's table
/// says "Zero") and X3 the subtype. The runtime has one flat application pool,
/// so every partition reports `TOTAL_MEMORY` and the live usage, and the
/// initial process id range collapses to the only process.
fn get_system_info(c: &mut Cpu) {
    let (id, sub) = (c.x[1], c.x[3]);
    let max_sub = if id == 2 { 1 } else { 3 };
    if id > 2 || sub > max_sub {
        c.process().warn_once(format!(
            "svcGetSystemInfo type {id} (subtype {sub}) is not implemented"
        ));
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    let p = c.process();
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = match id {
        0 => TOTAL_MEMORY,
        1 => p.used_memory(),
        _ => threads::PROCESS_ID,
    };
}

/// `svcGetProcessInfo` (0x7C): X0 is the output pointer the wrapper keeps, X1
/// the process handle and W2 the `ProcessInfoType`; the value comes back in W1.
/// The runtime's only process is the caller, and it is running, so type 0
/// (`ProcessState`) reports `ProcessState_Running` (2).
fn get_process_info(c: &mut Cpu) {
    if c.x[1] as u32 != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    if c.x[2] != 0 {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    ok1(c, 2);
}

/// Resource-limit handle reported by `svcGetInfo(9)`.
fn resource_limit_handle(p: &'static Process) -> u64 {
    p.handles
        .lock()
        .unwrap()
        .resource_limit(|| KObject::ResourceLimit(Arc::new(ResourceLimit::new(TOTAL_MEMORY))))
        as u64
}

fn get_resource_limit_value(c: &mut Cpu, current: bool) {
    let (handle, which) = (c.x[1] as u32, c.x[2]);
    let p = c.process();
    let handles = p.handles.lock().unwrap();
    let Some(KObject::ResourceLimit(limit)) = handles.get(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    // Only the process resource limit accounts the live handle table; a limit
    // made with svcCreateResourceLimit has no usage attributed to it.
    let process_limit = handles.process_resource_limit() == Some(handle);
    drop(handles);
    let value = if current {
        if process_limit {
            current_resource_value(p, which)
        } else {
            limit.limit(which).map(|_| 0)
        }
    } else {
        limit.limit(which)
    };
    match value {
        Some(value) => ok1(c, value),
        None => c.x[0] = RESULT_INVALID_ENUM,
    }
}

fn get_resource_limit_peak_value(c: &mut Cpu) {
    let (handle, which) = (c.x[1] as u32, c.x[2]);
    let p = c.process();
    let handles = p.handles.lock().unwrap();
    let Some(KObject::ResourceLimit(limit)) = handles.get(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    let process_limit = handles.process_resource_limit() == Some(handle);
    drop(handles);
    let Some(_) = limit.limit(which) else {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    };
    let current = if process_limit {
        current_resource_value(p, which).unwrap_or(0)
    } else {
        0
    };
    match limit.record_peak(which, current) {
        Some(peak) => ok1(c, peak),
        None => c.x[0] = RESULT_INVALID_ENUM,
    }
}

fn create_resource_limit(c: &mut Cpu) {
    let limit = Arc::new(ResourceLimit::empty());
    let handle = c
        .process()
        .handles
        .lock()
        .unwrap()
        .insert(KObject::ResourceLimit(limit));
    ok1(c, handle as u64);
}

fn set_resource_limit_limit_value(c: &mut Cpu) {
    let (handle, which, value) = (c.x[0] as u32, c.x[1], c.x[2] as i64);
    let p = c.process();
    let handles = p.handles.lock().unwrap();
    let Some(KObject::ResourceLimit(limit)) = handles.get(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    let process_limit = handles.process_resource_limit() == Some(handle);
    drop(handles);
    let Some(_) = limit.limit(which) else {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    };
    let current = if process_limit {
        current_resource_value(p, which).unwrap_or(0)
    } else {
        0
    };
    c.x[0] = match limit.set_limit(which, value, current) {
        Some(true) => RESULT_SUCCESS,
        Some(false) => RESULT_INVALID_STATE,
        None => RESULT_INVALID_ENUM,
    };
}

fn current_resource_value(p: &'static Process, which: u64) -> Option<u64> {
    if which == 0 {
        return Some(p.used_memory());
    }
    let handles = p.handles.lock().unwrap();
    match which {
        1 => Some(handles.threads().len() as u64),
        // svcCreateEvent publishes a read and a write handle for one event.
        2 => {
            let mut seen = std::collections::HashSet::new();
            Some(
                handles
                    .objects()
                    .filter_map(|o| match o {
                        KObject::Event(e) => Some(Arc::as_ptr(e) as usize),
                        _ => None,
                    })
                    .filter(|id| seen.insert(*id))
                    .count() as u64,
            )
        }
        3 => Some(
            handles
                .objects()
                .filter(|o| matches!(o, KObject::TransferMemory(_)))
                .count() as u64,
        ),
        // Both ends of a session pair, as `LimitableResource_Sessions` counts
        // them, including a server end still queued on its port.
        4 => Some(handles.sessions_in_use()),
        _ => None,
    }
}

fn get_last_thread_info(c: &mut Cpu) {
    // The kernel reports the thread that ran before the caller; the runtime
    // keeps no previous-thread context, so the caller's own registers are
    // reported (LastThreadContext: fp, sp, lr, pc, then TLS and flags).
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = c.x[29];
    c.x[2] = c.sp;
    c.x[3] = c.x[30];
    c.x[4] = c.pc;
    c.x[5] = c.thread.tls;
    c.x[6] = 0;
}

fn flush_entire_data_cache(c: &mut Cpu) {
    // The host caches are coherent with the guest, so there is nothing to
    // flush. The SVC returns nothing; success is left in x0.
    c.x[0] = RESULT_SUCCESS;
}

fn flush_data_cache(c: &mut Cpu) {
    let (addr, size) = (c.x[0], c.x[1]);
    c.x[0] = if size == 0 {
        RESULT_SUCCESS
    } else if addr.checked_add(size).is_none_or(|end| end > ADDR_MASK + 1) {
        RESULT_INVALID_MEMORY_STATE
    } else {
        RESULT_SUCCESS
    };
}

fn synchronize_preemption_state(c: &mut Cpu) {
    // Threads are never pinned across a system call in the runtime, so the
    // preemption state is always synchronized. The SVC returns nothing;
    // success is left in x0.
    c.x[0] = RESULT_SUCCESS;
}

fn break_(c: &mut Cpu) {
    if c.x[0] & 0x8000_0000 != 0 {
        c.x[0] = 0;
        return;
    }
    let p = c.process();
    p.terminate(1);
    let _ = writeln!(
        std::io::stderr(),
        "kirikae: guest called svcBreak(reason={:#x}, {:#x}, {:#x}) at {}",
        c.x[0],
        c.x[1],
        c.x[2],
        p.describe(c.pc - 4)
    );
    // The SDK passes a pointer to the assert/abort information in x1. Print the
    // readable strings it contains: that is the guest's own message, which is
    // usually the only way to know which assertion fired.
    // The info pointer can sit at the top of the stack, so a fixed forward read can
    // cross the mapping end. Read a window centred on it instead and print both the
    // strings and the raw bytes: the guest's abort record is a small struct.
    let info = c.x[1];
    let window = info.saturating_sub(0x80);
    if let Some(bytes) = p.memory.read(window, 0x180) {
        let _ = writeln!(
            std::io::stderr(),
            "kirikae:   info at {info:#x}: {}",
            bytes
                .chunks(16)
                .map(|c| c.iter().map(|b| format!("{b:02x}")).collect::<String>())
                .collect::<Vec<_>>()
                .join(" ")
        );
        let mut start = None;
        for (i, &b) in bytes.iter().enumerate() {
            let printable = (0x20..0x7f).contains(&b) || b == b'\n' || b == b'\t';
            match (printable, start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    if i - s >= 4 {
                        let text = String::from_utf8_lossy(&bytes[s..i]);
                        let _ =
                            writeln!(std::io::stderr(), "kirikae:   message: {}", text.trim_end());
                    }
                    start = None;
                }
                _ => {}
            }
        }
    }
    let mut fp = c.x[29];
    for _ in 0..16 {
        let (Some(next), Some(lr)) = (p.memory.read(fp, 8), p.memory.read(fp + 8, 8)) else {
            break;
        };
        let (next, lr) = (
            u64::from_le_bytes(next.try_into().unwrap()),
            u64::from_le_bytes(lr.try_into().unwrap()),
        );
        if lr < 4 || next <= fp {
            break;
        }
        let _ = writeln!(
            std::io::stderr(),
            "kirikae:   called from {}",
            p.describe(lr - 4)
        );
        fp = next;
    }
    c.halt();
}

fn output_debug_string(c: &mut Cpu) {
    let (addr, len) = (c.x[0], c.x[1]);
    match c.process().memory.read(addr, len) {
        Some(bytes) => {
            c.process().output(&bytes);
            c.x[0] = RESULT_SUCCESS;
        }
        None => fatal(
            c,
            format!("svcOutputDebugString read unmapped memory at {addr:#x}+{len:#x}"),
        ),
    }
}

/// Layout of the user exception context at TPIDRRO_EL0, from the Horizon
/// exception path: a 0x148-byte exception stack followed by the A64 frame.
pub const EXCEPTION_FRAME_OFFSET: u64 = 0x148;
const EXCEPTION_FRAME_SIZE: u64 = 0x78;

/// Description codes the kernel passes to the process entry point in X0.
pub const EXCEPTION_INSTRUCTION_ABORT: u64 = 0x100;
pub const EXCEPTION_UNCATEGORIZED: u64 = 0x104;

thread_local! {
    /// Set while the thread is handling a guest exception. A fault raised
    /// before `svcReturnFromException` replays the exception to the debugger,
    /// which the runtime does not have, so it terminates instead of looping.
    static EXCEPTION_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The process entry point (the crt0 the kernel re-enters on an exception).
fn process_entry(p: &'static Process) -> Option<u64> {
    p.handles
        .lock()
        .unwrap()
        .threads()
        .into_iter()
        .find(|t| t.id == MAIN_THREAD_ID)
        .map(|t| t.entry)
        .filter(|entry| *entry != 0)
}

/// Stores the interrupted context at the thread's TLS and returns to the
/// process entry point with X0 = description and X1 = SP = the frame, which is
/// what the Horizon userland reporting path does.
///
/// A host access violation cannot take this path: the host signal/VEH handler
/// only receives the faulting host address and unwinding back into the guest
/// thread would need a non-local jump, so `fault::report` still exits. See the
/// note there.
fn deliver_exception(c: &mut Cpu, desc: u64, fault_pc: u64) -> bool {
    if EXCEPTION_ACTIVE.with(|a| a.get()) {
        return false;
    }
    let p = c.process();
    let Some(entry) = process_entry(p) else {
        return false;
    };
    let frame_at = c.tpidrro_el0 + EXCEPTION_FRAME_OFFSET;
    if !p.memory.is_mapped(frame_at, EXCEPTION_FRAME_SIZE) {
        return false;
    }
    let mut frame = [0u8; EXCEPTION_FRAME_SIZE as usize];
    for (i, reg) in c.x[..9].iter().enumerate() {
        frame[i * 8..i * 8 + 8].copy_from_slice(&reg.to_le_bytes());
    }
    frame[0x48..0x50].copy_from_slice(&c.x[30].to_le_bytes());
    frame[0x50..0x58].copy_from_slice(&c.sp.to_le_bytes());
    frame[0x58..0x60].copy_from_slice(&fault_pc.to_le_bytes());
    // The runtime keeps no syndrome registers; PSTATE is the flag word.
    frame[0x60..0x64].copy_from_slice(&(c.nzcv as u32).to_le_bytes());
    if !p.memory.write(frame_at, &frame) {
        return false;
    }
    EXCEPTION_ACTIVE.with(|a| a.set(true));
    c.x[0] = desc;
    c.x[1] = frame_at;
    c.sp = frame_at;
    c.pc = entry;
    true
}

/// `svcReturnFromException` (0x28). With a zero result the frame the guest
/// updated at the TLS is restored, which is how crt0 pivots into the user
/// handler. A non-zero result replays the exception to KDebug; without a
/// debugger that is fatal. The kernel also writes 1 to TPIDR_EL0 on this path,
/// which the runtime leaves alone: guests use it as a thread pointer.
fn return_from_exception(c: &mut Cpu) {
    if !EXCEPTION_ACTIVE.with(|a| a.replace(false)) {
        fatal(
            c,
            "svcReturnFromException ran on a thread that is not handling an exception".into(),
        );
        return;
    }
    if c.x[0] != 0 {
        c.process().terminate(3);
        c.halt();
        return;
    }
    let frame_at = c.tpidrro_el0 + EXCEPTION_FRAME_OFFSET;
    let Some(frame) = c.process().memory.read(frame_at, EXCEPTION_FRAME_SIZE) else {
        fatal(
            c,
            format!("svcReturnFromException cannot read the frame at {frame_at:#x}"),
        );
        return;
    };
    let word = |at: usize| u64::from_le_bytes(frame[at..at + 8].try_into().unwrap_or_default());
    for (i, reg) in c.x[..9].iter_mut().enumerate() {
        *reg = word(i * 8);
    }
    c.x[30] = word(0x48);
    c.sp = word(0x50);
    c.pc = word(0x58);
    c.nzcv = u32::from_le_bytes(frame[0x60..0x64].try_into().unwrap_or_default()) as u64;
}

/// A guest exception the interpreter or a lifted instruction could not
/// execute. It is reported to the process entry point when that is possible.
pub fn undefined(c: &mut Cpu, addr: u64, word: u32) {
    let at = c.process().describe(addr);
    let _ = writeln!(
        std::io::stderr(),
        "kirikae: instruction {word:#010x} at {at} is not implemented"
    );
    if !deliver_exception(c, EXCEPTION_UNCATEGORIZED, addr) {
        terminate(c);
    }
}

/// A guest fault that is not an instruction trap in an SVC (a jump outside
/// executable memory, for example). Routed to the process entry point when the
/// guest can handle it.
pub fn fault(c: &mut Cpu, message: String) {
    let _ = writeln!(std::io::stderr(), "kirikae: {message}");
    if !deliver_exception(c, EXCEPTION_INSTRUCTION_ABORT, c.pc) {
        terminate(c);
    }
}

/// A host or runtime failure with no guest exception equivalent (a refused
/// mapping, a host thread that cannot start). Always fatal.
pub fn fatal(c: &mut Cpu, message: String) {
    let _ = writeln!(std::io::stderr(), "kirikae: {message}");
    terminate(c);
}

fn terminate(c: &mut Cpu) {
    let p = c.process();
    let _ = writeln!(
        std::io::stderr(),
        "kirikae: thread {} lr={} sp={:#x}",
        c.thread.id,
        p.describe(c.x[30]),
        c.sp
    );
    p.terminate(2);
    c.halt();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{Entry, GuestFn};
    use crate::kernel::{KThread, ThreadContext};
    use crate::process::{RunOptions, run_executable};
    use crate::services::{Reply, Request, Service};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};

    const GUEST_ENTRY: u64 = 0x1000_0000;

    static OBSERVED: Mutex<Vec<(&'static str, Vec<u64>)>> = Mutex::new(Vec::new());

    struct Echo;

    impl Service for Echo {
        fn name(&self) -> String {
            "echo".into()
        }

        fn call(&self, _c: &mut Cpu, req: &Request) -> Reply {
            Reply::ok().u32(req.cmd)
        }
    }

    fn record(name: &'static str, values: &[u64]) {
        OBSERVED.lock().unwrap().push((name, values.to_vec()));
    }

    fn observed(name: &'static str) -> Vec<u64> {
        let mut all = OBSERVED.lock().unwrap();
        let i = all
            .iter()
            .position(|(n, _)| *n == name)
            .expect("the fake guest recorded nothing");
        all.remove(i).1
    }

    /// Runs an entry point that drives `dispatch` on a real `Cpu` with a
    /// process whose guest memory and handle table are live, then halts.
    fn run_entries(entries: Vec<Entry>, expected_exit: i32) -> crate::process::RunResult {
        let entries: &'static [Entry] = Box::leak(entries.into_boxed_slice());
        let exe = kirikae_loader::Executable {
            modules: Vec::new(),
            entry: entries[0].addr,
            arch: kirikae_loader::Arch::A64,
            title_id: None,
            main_stack_size: None,
            system_resource_size: 0,
            romfs: None,
        };
        let result = run_executable(
            &exe,
            entries,
            RunOptions {
                capture_output: true,
            },
        )
        .expect("the fake guest runs");
        assert_eq!(result.exit_code, expected_exit);
        result
    }

    fn run_guest(f: GuestFn) {
        run_entries(
            vec![Entry {
                addr: GUEST_ENTRY,
                func: f,
            }],
            0,
        );
    }

    fn parked_thread(id: u64) -> Arc<KThread> {
        Arc::new(KThread {
            id,
            handle: id as u32,
            priority: AtomicI32::new(44),
            core: AtomicI32::new(0),
            ideal_core: AtomicI32::new(0),
            affinity: AtomicU64::new(1),
            inherited: AtomicI32::new(i32::MAX),
            entry: 0,
            arg: 0,
            stack_top: 0x9000,
            tls: 0xa000,
            started: AtomicBool::new(true),
            exited: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            last_pc: AtomicU64::new(0),
            last_lr: AtomicU64::new(0),
            paused: AtomicBool::new(false),
            context: Default::default(),
        })
    }

    extern "C" fn light_request(c: &mut Cpu) {
        let session = Session::new(Arc::new(Echo));
        let handle = c
            .process()
            .handles
            .lock()
            .unwrap()
            .insert(KObject::Session(session));
        let tls = c.tpidrro_el0;
        let mut out = Vec::new();
        for imm in [0x20u16, 0x21] {
            let request =
                crate::ipc::client::cmif(crate::ipc::TYPE_REQUEST, 7, &[], None, false, &[]);
            out.push(c.process().memory.write(tls, &request) as u64);
            c.x[0] = handle as u64;
            dispatch(c, imm);
            if trace() {
                eprintln!("svc {imm:#04x} -> x0={:#x} x1={:#x}", c.x[0], c.x[1]);
            }
            out.push(c.x[0]);
            let response = c
                .process()
                .memory
                .read(tls, IPC_BUFFER_SIZE)
                .unwrap_or_default();
            let (result, data, _) = crate::ipc::client::cmif_response(&response, false);
            out.push(result as u64);
            out.push(
                data.get(..4)
                    .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
                    as u64,
            );
        }
        c.x[0] = 0xdead;
        dispatch(c, 0x20);
        out.push(c.x[0]);
        record("light", &out);
        c.halt();
    }

    #[test]
    fn send_sync_request_light_uses_the_ipc_path() {
        run_guest(light_request);
        assert_eq!(
            observed("light"),
            vec![1, 0, 0, 7, 1, 0, 0, 7, RESULT_INVALID_HANDLE]
        );
    }

    extern "C" fn whole_cache(c: &mut Cpu) {
        c.x[0] = 0xdead_beef;
        dispatch(c, 0x2a);
        record("whole_cache", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn flush_entire_data_cache_succeeds() {
        run_guest(whole_cache);
        assert_eq!(observed("whole_cache"), vec![RESULT_SUCCESS]);
    }

    extern "C" fn range_cache(c: &mut Cpu) {
        let probes = [(0x1000, 0x4000), (ADDR_MASK, 2), (0x1234, 0)];
        let mut out = Vec::new();
        for (addr, size) in probes {
            c.x[0] = addr;
            c.x[1] = size;
            dispatch(c, 0x2b);
            out.push(c.x[0]);
        }
        record("range_cache", &out);
        c.halt();
    }

    #[test]
    fn flush_data_cache_validates_the_range() {
        run_guest(range_cache);
        assert_eq!(
            observed("range_cache"),
            vec![RESULT_SUCCESS, RESULT_INVALID_MEMORY_STATE, RESULT_SUCCESS]
        );
    }

    extern "C" fn last_thread(c: &mut Cpu) {
        c.pc = 0x1111_2222_3333_4444;
        c.sp = 0x5555_6666_7777_8888;
        c.x[29] = 0x9999_aaaa_bbbb_cccc;
        c.x[30] = 0xdddd_eeee_ffff_0000;
        dispatch(c, 0x2f);
        record("last_thread", &c.x[..7]);
        record("last_thread_tls", &[c.tpidrro_el0, c.thread.tls]);
        c.halt();
    }

    #[test]
    fn get_last_thread_info_reports_the_calling_thread() {
        run_guest(last_thread);
        let tls = observed("last_thread_tls");
        assert_eq!(tls[0], tls[1], "the TLS base is TPIDRRO_EL0");
        assert_eq!(
            observed("last_thread"),
            vec![
                RESULT_SUCCESS,
                0x9999_aaaa_bbbb_cccc,
                0x5555_6666_7777_8888,
                0xdddd_eeee_ffff_0000,
                0x1111_2222_3333_4444,
                tls[0],
                0,
            ]
        );
    }

    extern "C" fn limits(c: &mut Cpu) {
        c.x[1] = 9;
        c.x[3] = 0;
        dispatch(c, 0x29);
        let handle = c.x[1];
        let mut out = vec![c.x[0], handle];
        c.x[1] = 9;
        c.x[3] = 0;
        dispatch(c, 0x29);
        out.push(c.x[1]);
        for which in 0..=4 {
            for imm in [0x30u16, 0x31] {
                c.x[1] = handle;
                c.x[2] = which;
                dispatch(c, imm);
                if trace() {
                    eprintln!("svc {imm:#04x} -> x0={:#x} x1={:#x}", c.x[0], c.x[1]);
                }
                out.push(c.x[0]);
                out.push(c.x[1]);
            }
        }
        c.x[1] = handle;
        c.x[2] = 5;
        dispatch(c, 0x30);
        out.push(c.x[0]);
        c.x[1] = 0xdead;
        c.x[2] = 0;
        dispatch(c, 0x31);
        out.push(c.x[0]);
        out.push(c.process().used_memory());
        record("limits", &out);
        c.halt();
    }

    #[test]
    fn resource_limits_report_memory_and_threads() {
        run_guest(limits);
        let got = observed("limits");
        let (handle, used) = (got[1], got[25]);
        assert_eq!(got[0], RESULT_SUCCESS);
        assert_ne!(handle, 0);
        assert_eq!(got[2], handle, "svcGetInfo(9) caches the handle");
        let pair = |which: usize| {
            let i = 3 + which * 4;
            (got[i], got[i + 1], got[i + 2], got[i + 3])
        };
        assert_eq!(
            pair(0),
            (RESULT_SUCCESS, TOTAL_MEMORY, RESULT_SUCCESS, used)
        );
        assert_eq!(pair(1), (RESULT_SUCCESS, 0x40, RESULT_SUCCESS, 1));
        assert_eq!(pair(2), (RESULT_SUCCESS, 0x100, RESULT_SUCCESS, 0));
        assert_eq!(pair(3), (RESULT_SUCCESS, 0x100, RESULT_SUCCESS, 0));
        assert_eq!(pair(4), (RESULT_SUCCESS, 0x200, RESULT_SUCCESS, 0));
        assert_eq!(got[23], RESULT_INVALID_ENUM);
        assert_eq!(got[24], RESULT_INVALID_HANDLE);
    }

    extern "C" fn preemption(c: &mut Cpu) {
        c.x[0] = 0xdead_beef;
        dispatch(c, 0x36);
        record("preemption", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn synchronize_preemption_state_is_a_no_op() {
        run_guest(preemption);
        assert_eq!(observed("preemption"), vec![RESULT_SUCCESS]);
    }

    extern "C" fn thread_activity(c: &mut Cpu) {
        let t = parked_thread(0x99);
        let handle = c
            .process()
            .handles
            .lock()
            .unwrap()
            .insert(KObject::Thread(t.clone()));
        let mut out = Vec::new();
        for activity in [2u64, 1, 1, 0] {
            c.x[0] = handle as u64;
            c.x[1] = activity;
            dispatch(c, 0x32);
            out.push(c.x[0]);
        }
        out.push(t.paused.load(Ordering::Relaxed) as u64);
        record("thread_activity", &out);
        c.halt();
    }

    #[test]
    fn set_thread_activity_pauses_and_resumes() {
        run_guest(thread_activity);
        assert_eq!(
            observed("thread_activity"),
            vec![
                RESULT_INVALID_ENUM,
                RESULT_SUCCESS,
                RESULT_INVALID_STATE,
                RESULT_SUCCESS,
                0,
            ]
        );
    }

    extern "C" fn thread_context(c: &mut Cpu) {
        let mut expected = ThreadContext {
            sp: 0x1111,
            pc: 0x2222,
            psr: 0x6000_0000,
            tpidr: 0x3333,
            ..Default::default()
        };
        expected.x[29] = 0x4444;
        expected.x[30] = 0x5555;
        let bytes = expected.to_bytes();
        let runnable = parked_thread(0x9a);
        let paused = parked_thread(0x9b);
        paused.paused.store(true, Ordering::Relaxed);
        *paused.context.lock().unwrap() = expected;
        let (a, b) = {
            let mut handles = c.process().handles.lock().unwrap();
            (
                handles.insert(KObject::Thread(runnable)),
                handles.insert(KObject::Thread(paused)),
            )
        };
        let out = c.sp - 0x400;
        let mut results = Vec::new();
        for handle in [a, b] {
            c.x[0] = out;
            c.x[1] = handle as u64;
            dispatch(c, 0x33);
            results.push(c.x[0]);
        }
        record("thread_context_results", &results);
        let got = c
            .process()
            .memory
            .read(out, ThreadContext::SIZE as u64)
            .unwrap_or_default();
        record("thread_context", &words(&got));
        record("thread_context_expected", &words(&bytes));
        c.halt();
    }

    fn words(bytes: &[u8]) -> Vec<u64> {
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|w| u64::from_le_bytes(*w))
            .collect()
    }

    #[test]
    fn get_thread_context3_reports_a_paused_thread() {
        run_guest(thread_context);
        assert_eq!(
            observed("thread_context_results"),
            vec![RESULT_INVALID_STATE, RESULT_SUCCESS]
        );
        assert_eq!(
            observed("thread_context"),
            observed("thread_context_expected"),
            "the whole ThreadContext is written"
        );
    }

    extern "C" fn unknown(c: &mut Cpu) {
        c.x[0] = 0;
        dispatch(c, 0x3e);
        record("unknown", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn unknown_svcs_still_report_not_implemented() {
        run_guest(unknown);
        assert_eq!(observed("unknown"), vec![RESULT_NOT_IMPLEMENTED]);
    }

    const IPC_PAGE: u64 = 0x1000_0000;

    extern "C" fn async_ipc(c: &mut Cpu) {
        c.process()
            .memory
            .map_as(IPC_PAGE, PAGE, state::HEAP, perm::RW)
            .expect("the IPC buffer page maps");
        let session = Session::new(Arc::new(Echo));
        let handle = c
            .process()
            .handles
            .lock()
            .unwrap()
            .insert(KObject::Session(session));
        let request = crate::ipc::client::cmif(crate::ipc::TYPE_REQUEST, 7, &[], None, false, &[]);
        c.process().memory.write(IPC_PAGE, &request);

        // svcSendAsyncRequestWithUserBuffer(out_event, buffer, size, session):
        // the kernel returns the event in X1, the SDK wrapper stores it.
        c.x[0] = 0;
        c.x[1] = IPC_PAGE;
        c.x[2] = PAGE;
        c.x[3] = handle as u64;
        dispatch(c, 0x23);
        let event = c.x[1];
        record("async", &[c.x[0], event]);
        let response = c
            .process()
            .memory
            .read(IPC_PAGE, IPC_BUFFER_SIZE)
            .unwrap_or_default();
        let (result, data, _) = crate::ipc::client::cmif_response(&response, false);
        record(
            "async_response",
            &[
                result as u64,
                data.get(..4)
                    .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()) as u64),
            ],
        );
        let signaled = matches!(
            c.process().handles.lock().unwrap().get(event as u32),
            Some(KObject::Event(e)) if e.signaled.load(Ordering::Relaxed)
        );
        record("async_event", &[signaled as u64]);

        let probes = [
            (IPC_PAGE + 0x800, PAGE, handle as u64),
            (IPC_PAGE, 0x800, handle as u64),
            (IPC_PAGE, PAGE, 0xdead),
            (0x2000_0000, PAGE, handle as u64),
        ];
        let mut errors = Vec::new();
        for (addr, size, session) in probes {
            c.x[0] = 0;
            c.x[1] = addr;
            c.x[2] = size;
            c.x[3] = session;
            dispatch(c, 0x23);
            errors.push(c.x[0]);
        }
        record("async_errors", &errors);

        let list = c.sp - 0x40;
        c.process().memory.write(list, &handle.to_le_bytes());
        let mut replies = Vec::new();
        // No guest server object exists, so a valid list can only time out.
        for (count, addr, target) in [
            (1u64, list, 0u64),
            (0x41, list, 0),
            (1, 0x2000_0000, 0),
            (1, list, 0xdead),
        ] {
            c.x[0] = 0;
            c.x[1] = addr;
            c.x[2] = count;
            c.x[3] = target;
            c.x[4] = 0;
            dispatch(c, 0x43);
            replies.push(c.x[0]);
        }
        // svcReplyAndReceiveWithUserBuffer validates its buffer first.
        for (addr, size) in [
            (IPC_PAGE + 0x800, PAGE),
            (IPC_PAGE, 0x800),
            (IPC_PAGE, PAGE),
        ] {
            c.x[1] = addr;
            c.x[2] = size;
            c.x[3] = list;
            c.x[4] = 1;
            c.x[5] = 0;
            c.x[6] = 0;
            dispatch(c, 0x44);
            replies.push(c.x[0]);
        }
        record("async_replies", &replies);
        c.halt();
    }

    #[test]
    fn async_request_is_served_into_the_user_buffer() {
        run_guest(async_ipc);
        let got = observed("async");
        assert_eq!(got[0], RESULT_SUCCESS);
        assert_ne!(got[1], 0, "a readable event handle is returned in X1");
        assert_eq!(
            observed("async_response"),
            vec![RESULT_SUCCESS, 7],
            "the reply is in the user buffer"
        );
        assert_eq!(observed("async_event"), vec![1], "the event is signaled");
        assert_eq!(
            observed("async_errors"),
            vec![
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_POINTER,
            ]
        );
        assert_eq!(
            observed("async_replies"),
            vec![
                RESULT_TIMED_OUT,
                RESULT_TOO_MANY_HANDLES,
                RESULT_INVALID_POINTER,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_SIZE,
                RESULT_TIMED_OUT,
            ]
        );
    }

    /// The guest-visible pieces a two-thread session test shares: the client
    /// leaves the port handle here and the server reads it.
    const SESSION_MAILBOX: u64 = 0x0F00_0000;
    const SESSION_LIST: u64 = 0x0F00_1000;
    const SERVER_STACK: u64 = 0x0F01_0000;
    const SERVER_ENTRY: u64 = GUEST_ENTRY + 0x100;

    /// The result and first data word of a CMIF response, or `u32::MAX` twice
    /// when the buffer does not hold one. Parsing here keeps a wrong reply from
    /// panicking inside a guest entry point, which aborts the test binary.
    fn reply_summary(msg: &[u8]) -> (u32, u32) {
        let Ok(m) = crate::ipc::parse_hipc(msg) else {
            return (u32::MAX, u32::MAX);
        };
        let pad = m.data_offset.next_multiple_of(16) - m.data_offset;
        let Some(p) = m.data.get(pad..) else {
            return (u32::MAX, u32::MAX);
        };
        if p.get(..4) != Some(crate::ipc::SFCO.to_le_bytes().as_slice()) {
            return (u32::MAX, u32::MAX);
        }
        let word = |at: usize| {
            p.get(at..at + 4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap_or_default()))
        };
        (word(8), word(16))
    }

    /// The client of a guest session pair: create it, start the server thread,
    /// send a request and block until the reply lands in its own TLS.
    extern "C" fn session_client(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(SESSION_MAILBOX, PAGE, state::HEAP, perm::RW)
            .expect("the mailbox page maps");
        p.memory
            .map_as(SESSION_LIST, PAGE, state::HEAP, perm::RW)
            .expect("the handle list page maps");
        p.memory
            .map_as(SERVER_STACK, PAGE, state::STACK, perm::RW)
            .expect("the server stack maps");

        c.x[2] = 0;
        c.x[3] = 0x3a6d73;
        dispatch(c, 0x40);
        let (port, client) = (c.x[1] as u32, c.x[2] as u32);
        record("session_create", &[c.x[0], port as u64, client as u64]);
        p.memory.write(SESSION_MAILBOX, &port.to_le_bytes());

        c.x[1] = SERVER_ENTRY;
        c.x[2] = 0;
        c.x[3] = SERVER_STACK + PAGE;
        c.x[4] = 44;
        c.x[5] = 0;
        dispatch(c, 0x08);
        let thread = c.x[1] as u32;
        c.x[0] = thread as u64;
        dispatch(c, 0x09);
        record("session_start", &[c.x[0]]);

        let tls = c.tpidrro_el0;
        let request = crate::ipc::client::cmif(crate::ipc::TYPE_REQUEST, 7, &[], None, false, &[]);
        p.memory.write(tls, &request);
        c.x[0] = client as u64;
        dispatch(c, 0x21);
        let result = c.x[0];
        let response = p.memory.read(tls, IPC_BUFFER_SIZE).unwrap_or_default();
        let mut words = Vec::new();
        words.push(result);
        words.extend(
            response
                .as_chunks::<4>()
                .0
                .iter()
                .take(4)
                .map(|w| u32::from_le_bytes(*w) as u64),
        );
        record("session_client_raw", &words);
        let (reply_result, value) = reply_summary(&response);
        record(
            "session_client",
            &[result, reply_result as u64, value as u64],
        );
        c.x[0] = client as u64;
        dispatch(c, 0x16);
        c.halt();
    }

    /// The server: accept the queued client, take the request with
    /// `ReplyAndReceiveLight`, reply through `ReplyAndReceive` and stop waiting
    /// at once.
    extern "C" fn session_server(c: &mut Cpu) {
        let p = c.process();
        let raw = p.memory.read(SESSION_MAILBOX, 4).unwrap_or_default();
        let port = u32::from_le_bytes(raw.try_into().unwrap_or_default());
        // A server can wait for a connection on the port before accepting it.
        c.x[0] = port as u64;
        dispatch(c, 0x42);
        record("session_port_wait", &[c.x[0], c.x[1]]);
        c.x[1] = port as u64;
        dispatch(c, 0x41);
        let server = c.x[1] as u32;
        record("session_accept", &[c.x[0], server as u64]);

        c.x[0] = server as u64;
        dispatch(c, 0x42);
        let result = c.x[0];
        let tls = c.tpidrro_el0;
        let request = p.memory.read(tls, IPC_BUFFER_SIZE).unwrap_or_default();
        let cmd = crate::ipc::parse_hipc(&request)
            .and_then(|m| crate::ipc::parse_cmif(&m, false))
            .map_or(u32::MAX, |m| m.cmd);
        record("session_server", &[result, cmd as u64]);

        let response = crate::ipc::encode_cmif(0, &(cmd + 1).to_le_bytes(), &[], &[], None);
        p.memory.write(tls, &response);
        p.memory.write(SESSION_LIST, &server.to_le_bytes());
        c.x[1] = SESSION_LIST;
        c.x[2] = 1;
        c.x[3] = server as u64;
        c.x[4] = 0;
        dispatch(c, 0x43);
        record("session_reply", &[c.x[0]]);
        c.x[0] = server as u64;
        dispatch(c, 0x16);
        c.halt();
    }

    #[test]
    fn a_guest_session_pair_moves_a_request_and_a_reply() {
        run_entries(
            vec![
                Entry {
                    addr: GUEST_ENTRY,
                    func: session_client,
                },
                Entry {
                    addr: SERVER_ENTRY,
                    func: session_server,
                },
            ],
            0,
        );
        let create = observed("session_create");
        assert_eq!(create[0], RESULT_SUCCESS);
        assert_ne!(create[1], 0, "the server port handle is returned in W1");
        assert_ne!(create[2], 0, "the client session handle is returned in W2");
        assert_ne!(create[1], create[2]);
        assert_eq!(observed("session_start"), vec![RESULT_SUCCESS]);
        assert_eq!(
            observed("session_port_wait"),
            vec![RESULT_SUCCESS, 0],
            "a waiting server port reports the connection with its index"
        );
        let accept = observed("session_accept");
        assert_eq!(accept[0], RESULT_SUCCESS);
        assert_ne!(accept[1], 0, "accepting makes a server session handle");
        assert_ne!(accept[1], create[1]);
        assert_ne!(accept[1], create[2]);
        assert_eq!(
            observed("session_server"),
            vec![RESULT_SUCCESS, 7],
            "the server receives the client's request"
        );
        assert_eq!(
            observed("session_reply"),
            vec![RESULT_TIMED_OUT],
            "the reply is sent and no second request is waited for"
        );
        assert_eq!(
            observed("session_client"),
            vec![RESULT_SUCCESS, RESULT_SUCCESS, 8],
            "the client blocks until the reply arrives (raw {:x?})",
            observed("session_client_raw")
        );
    }

    /// The server end of the pair is handed to a plain host thread, which is how
    /// a host-implemented service is served through a guest port.
    extern "C" fn host_served_client(c: &mut Cpu) {
        let p = c.process();
        c.x[2] = 0;
        c.x[3] = 0x686f_7374;
        dispatch(c, 0x40);
        let port = c.x[1] as u32;
        let client = c.x[2] as u32;
        c.x[1] = port as u64;
        dispatch(c, 0x41);
        let server = c.x[1] as u32;
        let link = p
            .handles
            .lock()
            .unwrap()
            .session(server)
            .and_then(|s| s.guest_server_link())
            .expect("the accepted session is a guest pair");

        let host = std::thread::spawn(move || {
            let request = link.take_request_blocking(std::time::Duration::from_secs(5));
            let cmd = request
                .as_deref()
                .and_then(|b| crate::ipc::parse_hipc(b).ok())
                .and_then(|m| crate::ipc::parse_cmif(&m, false).ok())
                .map_or(u32::MAX, |m| m.cmd);
            let reply = crate::ipc::encode_cmif(0, &(cmd + 100).to_le_bytes(), &[], &[], None);
            let posted = link.post_reply(reply);
            record("host_service", &[cmd as u64, posted as u64]);
        });

        let request = crate::ipc::client::cmif(crate::ipc::TYPE_REQUEST, 3, &[], None, false, &[]);
        p.memory.write(c.tpidrro_el0, &request);
        c.x[0] = client as u64;
        dispatch(c, 0x21);
        let response = p
            .memory
            .read(c.tpidrro_el0, IPC_BUFFER_SIZE)
            .unwrap_or_default();
        let (reply_result, value) = reply_summary(&response);
        record("host_client", &[c.x[0], reply_result as u64, value as u64]);
        host.join().expect("the host service thread joins");
        c.halt();
    }

    #[test]
    fn a_host_service_serves_a_guest_through_the_port() {
        run_guest(host_served_client);
        assert_eq!(
            observed("host_service"),
            vec![3, 1],
            "the host thread takes the request and its reply is posted"
        );
        assert_eq!(
            observed("host_client"),
            vec![RESULT_SUCCESS, RESULT_SUCCESS, 103],
            "the guest client wakes with the host service's reply"
        );
    }

    /// A server that takes the request and then closes without replying.
    extern "C" fn dying_server(c: &mut Cpu) {
        let p = c.process();
        let raw = p.memory.read(SESSION_MAILBOX, 4).unwrap_or_default();
        let port = u32::from_le_bytes(raw.try_into().unwrap_or_default());
        c.x[1] = port as u64;
        dispatch(c, 0x41);
        let server = c.x[1] as u32;
        c.x[0] = server as u64;
        dispatch(c, 0x42);
        record("dying_server", &[c.x[0]]);
        c.x[0] = server as u64;
        dispatch(c, 0x16);
        c.halt();
    }

    extern "C" fn waiting_client(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(SESSION_MAILBOX, PAGE, state::HEAP, perm::RW)
            .expect("the mailbox page maps");
        p.memory
            .map_as(SERVER_STACK, PAGE, state::STACK, perm::RW)
            .expect("the server stack maps");
        c.x[2] = 0;
        c.x[3] = 0;
        dispatch(c, 0x40);
        let (port, client) = (c.x[1] as u32, c.x[2] as u32);
        p.memory.write(SESSION_MAILBOX, &port.to_le_bytes());
        c.x[1] = SERVER_ENTRY;
        c.x[2] = 0;
        c.x[3] = SERVER_STACK + PAGE;
        c.x[4] = 44;
        c.x[5] = 0;
        dispatch(c, 0x08);
        let thread = c.x[1] as u32;
        c.x[0] = thread as u64;
        dispatch(c, 0x09);

        let request = crate::ipc::client::cmif(crate::ipc::TYPE_REQUEST, 1, &[], None, false, &[]);
        p.memory.write(c.tpidrro_el0, &request);
        c.x[0] = client as u64;
        dispatch(c, 0x21);
        record("waiting_client", &[c.x[0]]);
        // The server end is closed, so the pair refuses further requests
        // without blocking, even though the request was left in service.
        c.x[0] = client as u64;
        dispatch(c, 0x21);
        record("waiting_client_again", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn a_dead_server_wakes_its_client() {
        run_entries(
            vec![
                Entry {
                    addr: GUEST_ENTRY,
                    func: waiting_client,
                },
                Entry {
                    addr: SERVER_ENTRY,
                    func: dying_server,
                },
            ],
            0,
        );
        assert_eq!(
            observed("dying_server"),
            vec![RESULT_SUCCESS],
            "the server took the request"
        );
        assert_eq!(
            observed("waiting_client"),
            vec![RESULT_REMOTE_DEAD],
            "closing the server end wakes the waiting client with 0xf601"
        );
        assert_eq!(
            observed("waiting_client_again"),
            vec![RESULT_REMOTE_DEAD],
            "a request to a dead peer is refused at once"
        );
    }

    extern "C" fn session_errors(c: &mut Cpu) {
        c.x[2] = 2;
        c.x[3] = 0;
        dispatch(c, 0x40);
        let bad_light = c.x[0];

        c.x[2] = 0;
        c.x[3] = 0x1234;
        dispatch(c, 0x40);
        let create = c.x[0];
        let (port, client) = (c.x[1] as u32, c.x[2] as u32);

        c.x[1] = client as u64;
        dispatch(c, 0x41);
        let accept_on_client = c.x[0];
        c.x[1] = 0xdead;
        dispatch(c, 0x41);
        let accept_unknown = c.x[0];
        c.x[1] = port as u64;
        dispatch(c, 0x41);
        let accept = c.x[0];
        let server = c.x[1] as u32;
        c.x[1] = port as u64;
        dispatch(c, 0x41);
        let accept_empty = c.x[0];
        c.x[0] = client as u64;
        dispatch(c, 0x42);
        let light_on_client = c.x[0];

        // Closing the client end is what the server sees as a dead peer.
        c.x[0] = client as u64;
        dispatch(c, 0x16);
        c.x[0] = server as u64;
        dispatch(c, 0x42);
        let dead = c.x[0];
        let dead_index = c.x[1];

        // LimitableResource_Sessions: a pair needs two session objects.
        c.x[1] = 9;
        c.x[3] = 0;
        dispatch(c, 0x29);
        let limit = c.x[1];
        record("session_limit_handle", &[c.x[0], limit]);
        c.x[0] = limit;
        c.x[1] = 4;
        c.x[2] = 1;
        dispatch(c, 0x7e);
        let lowered = c.x[0];
        c.x[2] = 0;
        c.x[3] = 0;
        dispatch(c, 0x40);
        let limited = c.x[0];
        c.x[0] = limit;
        c.x[1] = 4;
        c.x[2] = 0x200;
        dispatch(c, 0x7e);
        let raised = c.x[0];
        c.x[2] = 0;
        c.x[3] = 0;
        dispatch(c, 0x40);
        let created = c.x[0];

        record(
            "session_errors",
            &[
                bad_light,
                create,
                accept_on_client,
                accept_unknown,
                accept,
                accept_empty,
                light_on_client,
                dead,
                dead_index,
            ],
        );
        record("session_limits", &[lowered, limited, raised, created]);
        c.halt();
    }

    #[test]
    fn session_handles_and_limits_report_the_documented_errors() {
        run_guest(session_errors);
        let errors = observed("session_errors");
        assert_eq!(errors[0], RESULT_INVALID_ENUM, "is_light is a bool");
        assert_eq!(errors[1], RESULT_SUCCESS);
        assert_eq!(
            errors[2], RESULT_INVALID_HANDLE,
            "AcceptSession needs a port, not a session"
        );
        assert_eq!(errors[3], RESULT_INVALID_HANDLE);
        assert_eq!(errors[4], RESULT_SUCCESS);
        assert_eq!(
            errors[5], RESULT_NOT_FOUND,
            "no connection is waiting to be accepted"
        );
        assert_eq!(
            errors[6], RESULT_INVALID_HANDLE,
            "ReplyAndReceiveLight needs a port or a server session"
        );
        assert_eq!(
            errors[7], RESULT_REMOTE_DEAD,
            "a closed client makes the server session dead"
        );
        assert_eq!(errors[8], 0, "the index of the dead entry is returned");
        assert_eq!(
            observed("session_limits"),
            vec![
                RESULT_SUCCESS,
                RESULT_LIMIT_REACHED,
                RESULT_SUCCESS,
                RESULT_SUCCESS
            ],
            "a ceiling below the two ends of a pair refuses CreateSession"
        );
    }

    const TMEM_PAGE: u64 = 0x0F02_0000;
    const TMEM_MAP: u64 = 0x0F02_1000;
    const CODE_SRC: u64 = 0x0F03_0000;
    const CODE_MAP: u64 = 0x0F03_1000;

    /// The first eight bytes of a recorded buffer as one word.
    fn eight(bytes: &[u8]) -> u64 {
        let mut word = [0u8; 8];
        let n = bytes.len().min(8);
        word[..n].copy_from_slice(&bytes[..n]);
        u64::from_le_bytes(word)
    }

    extern "C" fn memory_family(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(TMEM_PAGE, PAGE, state::HEAP, perm::RW)
            .expect("the transfer memory page maps");
        p.memory.write(TMEM_PAGE, b"transfer");
        p.memory
            .map_as(CODE_SRC, PAGE, state::HEAP, perm::RW)
            .expect("the code memory page maps");
        p.memory.write(CODE_SRC, b"source");

        c.x[1] = TMEM_PAGE;
        c.x[2] = PAGE;
        c.x[3] = perm::NONE as u64;
        dispatch(c, 0x15);
        let created = c.x[0];
        let tmem = c.x[1] as u32;
        let retagged = p.memory.query(TMEM_PAGE).state;
        record("tmem_create", &[created, tmem as u64, retagged as u64]);

        c.x[1] = TMEM_PAGE;
        c.x[2] = PAGE;
        c.x[3] = perm::RX as u64;
        dispatch(c, 0x15);
        let create_exec = c.x[0];
        c.x[1] = TMEM_PAGE + 1;
        c.x[2] = PAGE;
        c.x[3] = perm::NONE as u64;
        dispatch(c, 0x15);
        let create_unaligned = c.x[0];
        c.x[1] = TMEM_PAGE;
        c.x[2] = 0;
        dispatch(c, 0x15);
        let create_empty = c.x[0];
        record(
            "tmem_create_errors",
            &[create_exec, create_unaligned, create_empty],
        );

        // MapTransferMemory wants the creation's size and permission.
        let mut map_errors = Vec::new();
        for (addr, size, map_perm) in [
            (TMEM_MAP, PAGE, perm::RW as u64),
            (TMEM_MAP, PAGE * 2, perm::NONE as u64),
            (TMEM_MAP + 1, PAGE, perm::NONE as u64),
        ] {
            c.x[0] = tmem as u64;
            c.x[1] = addr;
            c.x[2] = size;
            c.x[3] = map_perm;
            dispatch(c, 0x51);
            map_errors.push(c.x[0]);
        }
        record("tmem_map_errors", &map_errors);

        c.x[0] = tmem as u64;
        c.x[1] = TMEM_MAP;
        c.x[2] = PAGE;
        c.x[3] = perm::NONE as u64;
        dispatch(c, 0x51);
        let mapped = c.x[0];
        let contents = p.memory.read(TMEM_MAP, 8).unwrap_or_default();
        p.memory.write(TMEM_MAP, b"CHANGED");
        c.x[0] = tmem as u64;
        c.x[1] = TMEM_MAP + PAGE;
        c.x[2] = PAGE;
        c.x[3] = perm::NONE as u64;
        dispatch(c, 0x51);
        let twice = c.x[0];
        c.x[0] = tmem as u64;
        c.x[1] = TMEM_MAP;
        c.x[2] = PAGE * 2;
        dispatch(c, 0x52);
        let bad_unmap = c.x[0];
        c.x[0] = tmem as u64;
        c.x[1] = TMEM_MAP;
        c.x[2] = PAGE;
        dispatch(c, 0x52);
        let unmapped = c.x[0];
        let written_back = p.memory.read(TMEM_PAGE, 7).unwrap_or_default();
        record(
            "tmem_map",
            &[
                mapped,
                twice,
                bad_unmap,
                unmapped,
                eight(&contents),
                eight(&written_back),
            ],
        );

        c.x[1] = CODE_SRC;
        c.x[2] = PAGE;
        dispatch(c, 0x4b);
        let code_created = c.x[0];
        let code = c.x[1] as u32;
        let source_gone = !p.memory.is_mapped(CODE_SRC, 1);
        record(
            "code_create",
            &[code_created, code as u64, source_gone as u64],
        );

        let mut ops = Vec::new();
        for (op, addr, size, op_perm) in [
            (1u64, CODE_MAP, PAGE, perm::R as u64),
            (1, CODE_MAP, PAGE, perm::RW as u64),
            (0, CODE_MAP, PAGE, perm::R as u64),
            (0, CODE_MAP, PAGE * 2, perm::RW as u64),
            (0, CODE_MAP + 1, PAGE, perm::RW as u64),
            (9, CODE_MAP, PAGE, perm::RW as u64),
        ] {
            c.x[0] = code as u64;
            c.x[1] = op;
            c.x[2] = addr;
            c.x[3] = size;
            c.x[4] = op_perm;
            dispatch(c, 0x4c);
            ops.push(c.x[0]);
        }
        record("code_op_errors", &ops);

        c.x[0] = code as u64;
        c.x[1] = 0;
        c.x[2] = CODE_MAP;
        c.x[3] = PAGE;
        c.x[4] = perm::RW as u64;
        dispatch(c, 0x4c);
        let owner = c.x[0];
        let initial = p.memory.read(CODE_MAP, 4).unwrap_or_default();
        p.memory.write(CODE_MAP, b"code!");
        c.x[0] = code as u64;
        c.x[1] = 0;
        c.x[2] = CODE_MAP;
        c.x[3] = PAGE;
        c.x[4] = perm::RX as u64;
        dispatch(c, 0x4c);
        let executable = c.x[0];
        let code_perm = p.memory.query(CODE_MAP).perm;
        c.x[0] = code as u64;
        c.x[1] = 2;
        c.x[2] = CODE_MAP;
        c.x[3] = PAGE;
        dispatch(c, 0x4c);
        let unmapped_code = c.x[0];
        c.x[0] = code as u64;
        c.x[1] = 0;
        c.x[2] = CODE_MAP;
        c.x[3] = PAGE;
        c.x[4] = perm::RX as u64;
        dispatch(c, 0x4c);
        let remapped = c.x[0];
        let kept = p.memory.read(CODE_MAP, 5).unwrap_or_default();
        record(
            "code_map",
            &[
                owner,
                executable,
                code_perm as u64,
                unmapped_code,
                remapped,
                eight(&initial),
                eight(&kept),
            ],
        );
        c.halt();
    }

    #[test]
    fn transfer_and_code_memory_validate_size_and_permission() {
        run_guest(memory_family);
        let create = observed("tmem_create");
        assert_eq!(create[0], RESULT_SUCCESS);
        assert_ne!(create[1], 0, "the object handle is returned in X1");
        assert_eq!(
            create[2],
            state::TRANSFER as u64,
            "the creator's pages become transfer memory"
        );
        assert_eq!(
            observed("tmem_create_errors"),
            vec![
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_SIZE,
            ]
        );
        assert_eq!(
            observed("tmem_map_errors"),
            vec![
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_ADDRESS,
            ]
        );
        let map = observed("tmem_map");
        assert_eq!(map[0], RESULT_SUCCESS);
        assert_eq!(map[1], RESULT_INVALID_STATE, "one mapping at a time");
        assert_eq!(map[2], RESULT_INVALID_SIZE);
        assert_eq!(map[3], RESULT_SUCCESS);
        assert_eq!(map[4], eight(b"transfer"), "the block maps");
        assert_eq!(
            map[5],
            eight(b"CHANGED"),
            "the mapping writes back to the creator's pages"
        );

        let code = observed("code_create");
        assert_eq!(code[0], RESULT_SUCCESS);
        assert_ne!(code[1], 0);
        assert_eq!(code[2], 1, "the locked pages leave the address space");
        assert_eq!(
            observed("code_op_errors"),
            vec![
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_ENUM,
            ]
        );
        let mapped = observed("code_map");
        assert_eq!(mapped[0], RESULT_SUCCESS);
        assert_eq!(mapped[1], RESULT_SUCCESS, "the owner can go from RW to RX");
        assert_eq!(mapped[2], perm::RX as u64);
        assert_eq!(mapped[3], RESULT_SUCCESS);
        assert_eq!(mapped[4], RESULT_SUCCESS);
        assert_eq!(
            mapped[5],
            eight(&[0xff; 4]),
            "code memory starts 0xFF-filled"
        );
        assert_eq!(mapped[6], eight(b"code!"), "the contents survive the unmap");
    }

    const MIRROR_SRC: u64 = 0x0F04_0000;
    const MIRROR_DST: u64 = 0x0F04_2000;
    const MIRROR_BAD: u64 = 0x0F04_3000;

    extern "C" fn mirrored_memory(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(MIRROR_SRC, PAGE * 2, state::HEAP, perm::RW)
            .expect("the mirror source maps");
        p.memory.write(MIRROR_SRC, b"guard");

        c.x[0] = MIRROR_DST;
        c.x[1] = MIRROR_SRC;
        c.x[2] = PAGE;
        dispatch(c, 0x04);
        let map = c.x[0];
        let borrowed = p.memory.query(MIRROR_SRC).perm;
        let dst_state = p.memory.query(MIRROR_DST).state;
        let copied = p.memory.read(MIRROR_DST, 5).unwrap_or_default();

        c.x[0] = MIRROR_DST;
        c.x[1] = MIRROR_SRC;
        c.x[2] = PAGE;
        dispatch(c, 0x04);
        let twice = c.x[0];
        c.x[0] = MIRROR_BAD;
        c.x[1] = MIRROR_SRC;
        c.x[2] = 0;
        dispatch(c, 0x04);
        let empty = c.x[0];
        c.x[0] = MIRROR_BAD + 1;
        c.x[1] = MIRROR_SRC;
        c.x[2] = PAGE;
        dispatch(c, 0x04);
        let unaligned = c.x[0];

        p.memory.write(MIRROR_DST, b"GUARD");
        c.x[0] = MIRROR_DST;
        c.x[1] = MIRROR_SRC;
        c.x[2] = PAGE;
        dispatch(c, 0x05);
        let unmap = c.x[0];
        let restored = p.memory.query(MIRROR_SRC).perm;
        let written_back = p.memory.read(MIRROR_SRC, 5).unwrap_or_default();
        let gone = !p.memory.is_mapped(MIRROR_DST, 1);
        c.x[0] = MIRROR_DST;
        c.x[1] = MIRROR_SRC;
        c.x[2] = PAGE;
        dispatch(c, 0x05);
        let twice_unmap = c.x[0];

        record(
            "mirror",
            &[
                map,
                borrowed as u64,
                dst_state as u64,
                twice,
                empty,
                unaligned,
                unmap,
                restored as u64,
                gone as u64,
                twice_unmap,
            ],
        );
        record("mirror_contents", &[eight(&copied), eight(&written_back)]);
        c.halt();
    }

    #[test]
    fn map_memory_borrows_the_source_and_unmap_restores_it() {
        run_guest(mirrored_memory);
        let got = observed("mirror");
        assert_eq!(got[0], RESULT_SUCCESS);
        assert_eq!(got[1], perm::RW as u64, "the source keeps its permission");
        assert_eq!(got[2], state::STACK as u64, "the view is a stack mapping");
        assert_eq!(
            got[3], RESULT_INVALID_MEMORY_STATE,
            "the destination is already mapped"
        );
        assert_eq!(got[4], RESULT_INVALID_SIZE);
        assert_eq!(got[5], RESULT_INVALID_ADDRESS);
        assert_eq!(got[6], RESULT_SUCCESS);
        assert_eq!(
            got[7],
            perm::RW as u64,
            "the source gets its permission back"
        );
        assert_eq!(got[8], 1, "the view is gone");
        assert_eq!(got[9], RESULT_INVALID_MEMORY_STATE);
        let contents = observed("mirror_contents");
        assert_eq!(contents[0], eight(b"guard"));
        assert_eq!(contents[1], eight(b"GUARD"), "writes come back");
    }

    const CACHE_PAGE: u64 = 0x0F05_0000;

    extern "C" fn data_cache(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(CACHE_PAGE, PAGE, state::HEAP, perm::RW)
            .expect("the cache range maps");
        let mut out = Vec::new();
        for imm in [0x5du16, 0x5e, 0x5f] {
            for (handle, addr, size) in [
                (CURRENT_PROCESS, CACHE_PAGE, PAGE),
                (CURRENT_PROCESS, CACHE_PAGE, 0),
                (CURRENT_PROCESS + 1, CACHE_PAGE, PAGE),
                (CURRENT_PROCESS, 0x2000_0000, PAGE),
            ] {
                c.x[0] = handle;
                c.x[1] = addr;
                c.x[2] = size;
                dispatch(c, imm);
                if trace() {
                    eprintln!("svc {imm:#04x} -> x0={:#x} x1={:#x}", c.x[0], c.x[1]);
                }
                out.push(c.x[0]);
            }
        }
        record("data_cache", &out);
        c.halt();
    }

    #[test]
    fn process_data_cache_validates_the_process_and_the_range() {
        run_guest(data_cache);
        assert_eq!(
            observed("data_cache"),
            vec![
                RESULT_SUCCESS,
                RESULT_SUCCESS,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_MEMORY_STATE,
                RESULT_SUCCESS,
                RESULT_SUCCESS,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_MEMORY_STATE,
                RESULT_SUCCESS,
                RESULT_SUCCESS,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_MEMORY_STATE,
            ]
        );
    }

    const FAULT_PC: u64 = 0x1000_2000;
    const HANDLER_ENTRY: u64 = GUEST_ENTRY + 0x100;

    extern "C" fn exception_handler(c: &mut Cpu) {
        record("handler", &[c.x[0], c.x[1], c.sp, c.pc]);
        c.halt();
    }

    extern "C" fn exception_guest(c: &mut Cpu) {
        if c.x[0] == 0 {
            record("exception_tls", &[c.tpidrro_el0]);
            c.pc = FAULT_PC;
            c.sp = 0x3333;
            c.x[1] = 0x11;
            c.x[7] = 0x77;
            c.x[8] = 0x88;
            c.x[29] = 0x1111;
            c.x[30] = 0x2222;
            undefined(c, FAULT_PC, 0xdead_beef);
            record("delivered", &[c.x[0], c.x[1], c.sp, c.pc]);
            return;
        }
        // The kernel re-entered the entry point with X0 = description and
        // X1 = the frame; patch ELR at the user handler and return.
        let frame_at = c.x[1];
        record(
            "frame_at",
            &[frame_at, c.tpidrro_el0 + EXCEPTION_FRAME_OFFSET],
        );
        let frame = c
            .process()
            .memory
            .read(frame_at, EXCEPTION_FRAME_SIZE)
            .unwrap_or_default();
        record("frame", &words(&frame));
        let mut patched = frame;
        patched[0x0..0x8].copy_from_slice(&0x77u64.to_le_bytes());
        patched[0x58..0x60].copy_from_slice(&HANDLER_ENTRY.to_le_bytes());
        c.process().memory.write(frame_at, &patched);
        c.x[0] = 0;
        dispatch(c, 0x28);
    }

    #[test]
    fn guest_faults_reach_the_process_entry_and_return_from_exception() {
        run_entries(
            vec![
                Entry {
                    addr: GUEST_ENTRY,
                    func: exception_guest,
                },
                Entry {
                    addr: HANDLER_ENTRY,
                    func: exception_handler,
                },
            ],
            0,
        );
        let tls = observed("exception_tls")[0];
        assert_eq!(
            observed("delivered"),
            vec![
                EXCEPTION_UNCATEGORIZED,
                tls + EXCEPTION_FRAME_OFFSET,
                tls + EXCEPTION_FRAME_OFFSET,
                GUEST_ENTRY,
            ],
            "X0 = description, X1 = SP = frame, PC = entry"
        );
        assert_eq!(
            observed("frame_at"),
            vec![tls + EXCEPTION_FRAME_OFFSET, tls + EXCEPTION_FRAME_OFFSET]
        );
        let frame = observed("frame");
        assert_eq!(
            &frame[..12],
            &[
                0, 0x11, 0, 0, 0, 0, 0, 0x77, 0x88, 0x2222, // lr
                0x3333, // sp
                FAULT_PC,
            ],
            "ExceptionFrameA64 holds x0-x8, lr, sp and pc"
        );
        assert_eq!(&frame[12..], &[0, 0, 0], "PSTATE and syndrome are zero");
        assert_eq!(
            observed("handler"),
            vec![0x77, 0x11, 0x3333, HANDLER_ENTRY],
            "ReturnFromException restores the frame and jumps to the handler"
        );
    }

    extern "C" fn exception_abort(c: &mut Cpu) {
        if c.x[0] == 0 {
            undefined(c, FAULT_PC, 0xdead_beef);
            return;
        }
        // A non-zero result replays the exception to KDebug; there is no
        // debugger, so the runtime stops the process.
        c.x[0] = 0x10001;
        dispatch(c, 0x28);
    }

    #[test]
    fn return_from_exception_with_a_result_terminates() {
        run_entries(
            vec![Entry {
                addr: GUEST_ENTRY,
                func: exception_abort,
            }],
            3,
        );
    }

    extern "C" fn exception_replays(c: &mut Cpu) {
        if c.x[0] == 0 {
            undefined(c, FAULT_PC, 0xdead_beef);
            return;
        }
        // A second fault before ReturnFromException replays the exception.
        undefined(c, FAULT_PC + 0x1000, 0xdead_beef);
        c.halt();
    }

    #[test]
    fn a_second_fault_while_handling_terminates() {
        run_entries(
            vec![Entry {
                addr: GUEST_ENTRY,
                func: exception_replays,
            }],
            2,
        );
    }

    extern "C" fn stray_return(c: &mut Cpu) {
        c.x[0] = 0;
        dispatch(c, 0x28);
        c.halt();
    }

    #[test]
    fn return_from_exception_without_a_fault_terminates() {
        run_entries(
            vec![Entry {
                addr: GUEST_ENTRY,
                func: stray_return,
            }],
            2,
        );
    }

    extern "C" fn future_thread(c: &mut Cpu) {
        c.pc = 0x4444;
        c.sp = 0x3333;
        c.x[29] = 0x1111;
        c.x[30] = 0x2222;
        c.x[2] = crate::kernel::PSEUDO_CURRENT_THREAD as u64;
        dispatch(c, 0x2e);
        record("future", &c.x[..6]);
        c.x[2] = 0xdead;
        dispatch(c, 0x2e);
        record("future_bad", &[c.x[0]]);
        c.x[1] = 20;
        c.x[3] = 0;
        dispatch(c, 0x29);
        record("future_context", &[c.x[0], c.x[1], c.tpidrro_el0]);
        c.halt();
    }

    #[test]
    fn get_debug_future_thread_info_reports_the_target_thread() {
        run_guest(future_thread);
        assert_eq!(
            observed("future"),
            vec![
                RESULT_SUCCESS,
                0x1111,
                0x3333,
                0x2222,
                0x4444,
                MAIN_THREAD_ID,
            ]
        );
        assert_eq!(observed("future_bad"), vec![RESULT_INVALID_HANDLE]);
        let context = observed("future_context");
        assert_eq!(context[0], RESULT_SUCCESS);
        assert_eq!(
            context[1], context[2],
            "svcGetInfo(20) reports the TLS base the exception frame uses"
        );
    }

    extern "C" fn limits_created(c: &mut Cpu) {
        c.x[1] = 9;
        c.x[3] = 0;
        dispatch(c, 0x29);
        let process_limit = c.x[1];
        record("limit_process", &[c.x[0], process_limit]);

        c.x[0] = 0;
        dispatch(c, 0x7d);
        let limit = c.x[1];
        record("limit_created", &[c.x[0], limit]);

        // A fresh limit starts at zero for every LimitableResource.
        let mut limits = Vec::new();
        for which in 0..=4 {
            c.x[0] = 0;
            c.x[1] = limit;
            c.x[2] = which;
            dispatch(c, 0x30);
            limits.push(c.x[0]);
            limits.push(c.x[1]);
        }
        record("limit_zeros", &limits);

        // Raise the thread ceiling; nothing is attributed to a fresh limit.
        c.x[0] = limit;
        c.x[1] = 1;
        c.x[2] = 8;
        dispatch(c, 0x7e);
        record("limit_set", &[c.x[0]]);
        let mut after = Vec::new();
        for imm in [0x30u16, 0x31, 0x37] {
            c.x[1] = limit;
            c.x[2] = 1;
            dispatch(c, imm);
            if trace() {
                eprintln!("svc {imm:#04x} -> x0={:#x} x1={:#x}", c.x[0], c.x[1]);
            }
            after.push(c.x[0]);
            after.push(c.x[1]);
        }
        record("limit_after", &after);

        // The process limit accounts the live handle table and refuses a
        // ceiling below the current usage.
        c.x[0] = process_limit;
        c.x[1] = 1;
        c.x[2] = 0;
        dispatch(c, 0x7e);
        record("limit_low", &[c.x[0]]);
        c.x[0] = process_limit;
        c.x[1] = 1;
        c.x[2] = 4;
        dispatch(c, 0x7e);
        record("limit_raised", &[c.x[0]]);

        let mut peaks = Vec::new();
        for (handle, which) in [
            (process_limit, 1u64),
            (process_limit, 0),
            (limit, 5),
            (0xdead, 0),
        ] {
            c.x[1] = handle;
            c.x[2] = which;
            dispatch(c, 0x37);
            peaks.push(c.x[0]);
            peaks.push(c.x[1]);
        }
        record("limit_peaks", &peaks);

        c.x[0] = limit;
        c.x[1] = 5;
        c.x[2] = 1;
        dispatch(c, 0x7e);
        record("limit_bad_which", &[c.x[0]]);
        c.x[0] = 0xdead;
        c.x[1] = 0;
        c.x[2] = 1;
        dispatch(c, 0x7e);
        record("limit_bad_handle", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn resource_limits_create_set_and_peak() {
        run_guest(limits_created);
        let process = observed("limit_process");
        assert_eq!(process[0], RESULT_SUCCESS);
        let created = observed("limit_created");
        assert_eq!(created[0], RESULT_SUCCESS);
        assert_ne!(created[1], 0);
        assert_ne!(created[1], process[1], "a separate limit object");
        assert_eq!(
            observed("limit_zeros"),
            vec![
                RESULT_SUCCESS,
                0,
                RESULT_SUCCESS,
                0,
                RESULT_SUCCESS,
                0,
                RESULT_SUCCESS,
                0,
                RESULT_SUCCESS,
                0,
            ]
        );
        assert_eq!(observed("limit_set"), vec![RESULT_SUCCESS]);
        assert_eq!(
            observed("limit_after"),
            vec![RESULT_SUCCESS, 8, RESULT_SUCCESS, 0, RESULT_SUCCESS, 0]
        );
        assert_eq!(observed("limit_low"), vec![RESULT_INVALID_STATE]);
        assert_eq!(observed("limit_raised"), vec![RESULT_SUCCESS]);
        let peaks = observed("limit_peaks");
        assert_eq!(peaks[0], RESULT_SUCCESS);
        assert_eq!(peaks[1], 1, "the main thread is the peak thread count");
        assert_eq!(peaks[2], RESULT_SUCCESS);
        assert_ne!(peaks[3], 0, "peak memory is the used memory");
        assert_eq!(peaks[4], RESULT_INVALID_ENUM);
        assert_eq!(peaks[6], RESULT_INVALID_HANDLE);
        assert_eq!(observed("limit_bad_which"), vec![RESULT_INVALID_ENUM]);
        assert_eq!(observed("limit_bad_handle"), vec![RESULT_INVALID_HANDLE]);
    }

    const PORT_MAILBOX: u64 = 0x0F10_0000;
    const PORT_LIST: u64 = 0x0F10_1000;
    const PORT_STACK: u64 = 0x0F11_0000;
    const PORT_NAME: u64 = 0x0F10_0040;
    const PORT_ENTRY: u64 = GUEST_ENTRY + 0x200;

    /// The server of a named port: wait on the port, accept the connection,
    /// take the request and reply through `ReplyAndReceive`.
    extern "C" fn named_port_server(c: &mut Cpu) {
        let p = c.process();
        let raw = p.memory.read(PORT_MAILBOX, 4).unwrap_or_default();
        let port = u32::from_le_bytes(raw.try_into().unwrap_or_default());
        let list = c.sp - 8;
        p.memory.write(list, &port.to_le_bytes());
        c.x[1] = list;
        c.x[2] = 1;
        c.x[3] = -1i64 as u64;
        dispatch(c, 0x18);
        record("named_wait", &[c.x[0], c.x[1]]);

        c.x[1] = port as u64;
        dispatch(c, 0x41);
        let server = c.x[1] as u32;
        record("named_accept", &[c.x[0], server as u64]);

        c.x[0] = server as u64;
        dispatch(c, 0x42);
        let tls = c.tpidrro_el0;
        let request = p.memory.read(tls, IPC_BUFFER_SIZE).unwrap_or_default();
        let cmd = crate::ipc::parse_hipc(&request)
            .and_then(|m| crate::ipc::parse_cmif(&m, false))
            .map_or(u32::MAX, |m| m.cmd);
        record("named_server", &[c.x[0], cmd as u64]);

        let response = crate::ipc::encode_cmif(0, &(cmd + 100).to_le_bytes(), &[], &[], None);
        p.memory.write(tls, &response);
        p.memory.write(PORT_LIST, &server.to_le_bytes());
        c.x[1] = PORT_LIST;
        c.x[2] = 1;
        c.x[3] = server as u64;
        c.x[4] = 0;
        dispatch(c, 0x43);
        record("named_reply", &[c.x[0]]);
        c.x[0] = server as u64;
        dispatch(c, 0x16);
        c.halt();
    }

    extern "C" fn named_port_client(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(PORT_MAILBOX, PAGE, state::HEAP, perm::RW)
            .expect("the port mailbox page maps");
        p.memory
            .map_as(PORT_LIST, PAGE, state::HEAP, perm::RW)
            .expect("the handle list page maps");
        p.memory
            .map_as(PORT_STACK, PAGE, state::STACK, perm::RW)
            .expect("the server stack maps");
        p.memory.write(PORT_NAME, b"ws9:p1\0");

        c.x[1] = PORT_NAME;
        c.x[2] = 1;
        dispatch(c, 0x71);
        let port = c.x[1] as u32;
        record("named_create", &[c.x[0], port as u64]);
        p.memory.write(PORT_MAILBOX, &port.to_le_bytes());

        c.x[1] = PORT_ENTRY;
        c.x[2] = 0;
        c.x[3] = PORT_STACK + PAGE;
        c.x[4] = 44;
        c.x[5] = 0;
        dispatch(c, 0x08);
        let thread = c.x[1] as u32;
        c.x[0] = thread as u64;
        dispatch(c, 0x09);
        record("named_start", &[c.x[0]]);

        c.x[1] = PORT_NAME;
        dispatch(c, 0x1f);
        let client = c.x[1] as u32;
        record("named_connect", &[c.x[0], client as u64]);

        // The port accepts one connection, so the second is refused.
        c.x[1] = PORT_NAME;
        dispatch(c, 0x1f);
        record("named_limit", &[c.x[0]]);

        let request = crate::ipc::client::cmif(crate::ipc::TYPE_REQUEST, 5, &[], None, false, &[]);
        p.memory.write(c.tpidrro_el0, &request);
        c.x[0] = client as u64;
        dispatch(c, 0x21);
        let response = p
            .memory
            .read(c.tpidrro_el0, IPC_BUFFER_SIZE)
            .unwrap_or_default();
        let (reply_result, value) = reply_summary(&response);
        record("named_client", &[c.x[0], reply_result as u64, value as u64]);

        c.x[0] = client as u64;
        dispatch(c, 0x16);
        // Closing the port frees the name: `sm:` keeps answering, but the
        // guest name is gone.
        c.x[0] = port as u64;
        dispatch(c, 0x16);
        record("named_close", &[c.x[0]]);
        c.x[1] = PORT_NAME;
        dispatch(c, 0x1f);
        record("named_gone", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn a_named_port_connects_accepts_and_limits_connections() {
        run_entries(
            vec![
                Entry {
                    addr: GUEST_ENTRY,
                    func: named_port_client,
                },
                Entry {
                    addr: PORT_ENTRY,
                    func: named_port_server,
                },
            ],
            0,
        );
        let create = observed("named_create");
        assert_eq!(create[0], RESULT_SUCCESS);
        assert_ne!(
            create[1], 0,
            "ManageNamedPort returns the server port in W1"
        );
        assert_eq!(observed("named_start"), vec![RESULT_SUCCESS]);
        assert_eq!(
            observed("named_wait"),
            vec![RESULT_SUCCESS, 0],
            "a connection signals the waiting server port"
        );
        let accept = observed("named_accept");
        assert_eq!(accept[0], RESULT_SUCCESS);
        assert_ne!(accept[1], 0, "accepting makes a server session handle");
        assert_eq!(
            observed("named_server"),
            vec![RESULT_SUCCESS, 5],
            "the server receives the client's request"
        );
        assert_eq!(observed("named_reply"), vec![RESULT_TIMED_OUT]);
        let connect = observed("named_connect");
        assert_eq!(connect[0], RESULT_SUCCESS);
        assert_ne!(connect[1], 0, "ConnectToNamedPort returns a session in W1");
        assert_eq!(
            observed("named_limit"),
            vec![RESULT_LIMIT_REACHED],
            "max_sessions is the port's connection ceiling"
        );
        assert_eq!(
            observed("named_client"),
            vec![RESULT_SUCCESS, RESULT_SUCCESS, 105],
            "the named client blocks until the server replies"
        );
        assert_eq!(observed("named_close"), vec![RESULT_SUCCESS]);
        assert_eq!(
            observed("named_gone"),
            vec![RESULT_NOT_FOUND],
            "closing the port releases its name"
        );
    }

    /// A server that waits on a port which the other thread then closes.
    extern "C" fn closing_port_server(c: &mut Cpu) {
        let p = c.process();
        let raw = p.memory.read(PORT_MAILBOX, 4).unwrap_or_default();
        let port = u32::from_le_bytes(raw.try_into().unwrap_or_default());
        let list = c.sp - 8;
        p.memory.write(list, &port.to_le_bytes());
        c.x[1] = list;
        c.x[2] = 1;
        // A timeout keeps a missing wake from hanging the run: it would report
        // 0xea01 instead of the closed-port 0xf601.
        c.x[3] = 2_000_000_000;
        dispatch(c, 0x18);
        record("closed_wait", &[c.x[0], c.x[1]]);
        c.halt();
    }

    extern "C" fn closing_port_client(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(PORT_MAILBOX, PAGE, state::HEAP, perm::RW)
            .expect("the port mailbox page maps");
        p.memory
            .map_as(PORT_STACK, PAGE, state::STACK, perm::RW)
            .expect("the server stack maps");
        p.memory.write(PORT_NAME, b"ws9:closep\0");

        c.x[1] = PORT_NAME;
        c.x[2] = 4;
        dispatch(c, 0x71);
        let port = c.x[1] as u32;
        record("closed_create", &[c.x[0], port as u64]);
        p.memory.write(PORT_MAILBOX, &port.to_le_bytes());

        c.x[1] = PORT_ENTRY;
        c.x[2] = 0;
        c.x[3] = PORT_STACK + PAGE;
        c.x[4] = 44;
        c.x[5] = 0;
        dispatch(c, 0x08);
        let thread = c.x[1] as u32;
        c.x[0] = thread as u64;
        dispatch(c, 0x09);

        // Sleep off the CPU so the server reaches the wait, then close the port
        // it is waiting on.
        c.x[0] = 20_000_000;
        dispatch(c, 0x0b);
        c.x[0] = port as u64;
        dispatch(c, 0x16);
        record("closed_close", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn closing_a_port_releases_its_waiter() {
        run_entries(
            vec![
                Entry {
                    addr: GUEST_ENTRY,
                    func: closing_port_client,
                },
                Entry {
                    addr: PORT_ENTRY,
                    func: closing_port_server,
                },
            ],
            0,
        );
        let create = observed("closed_create");
        assert_eq!(create[0], RESULT_SUCCESS);
        assert_eq!(observed("closed_close"), vec![RESULT_SUCCESS]);
        assert_eq!(
            observed("closed_wait")[0],
            RESULT_REMOTE_DEAD,
            "closing the port wakes the waiting server with 0xf601"
        );
    }

    extern "C" fn client_port_sessions(c: &mut Cpu) {
        let p = c.process();
        let mut errors = Vec::new();
        for (max, light) in [(-2i64, 0u64), (0x40, 0), (2, 2)] {
            c.x[2] = max as u64;
            c.x[3] = light;
            c.x[4] = 0;
            dispatch(c, 0x70);
            errors.push(c.x[0]);
        }
        record("cp_create_errors", &errors);

        c.x[2] = 2;
        c.x[3] = 0;
        c.x[4] = 0x1234;
        dispatch(c, 0x70);
        let server = c.x[1] as u32;
        let client_port = c.x[2] as u32;
        record("cp_create", &[c.x[0], server as u64, client_port as u64]);

        c.x[1] = server as u64;
        dispatch(c, 0x72);
        record("cp_server_as_client", &[c.x[0]]);
        c.x[1] = 0xdead;
        dispatch(c, 0x72);
        record("cp_unknown", &[c.x[0]]);

        let mut connects = Vec::new();
        for _ in 0..2 {
            c.x[1] = client_port as u64;
            dispatch(c, 0x72);
            connects.push(c.x[0]);
            connects.push(c.x[1]);
        }
        record("cp_connect", &connects);
        c.x[1] = client_port as u64;
        dispatch(c, 0x72);
        record("cp_limit", &[c.x[0]]);

        let mut accepts = Vec::new();
        for _ in 0..2 {
            c.x[1] = server as u64;
            dispatch(c, 0x41);
            accepts.push(c.x[0]);
            accepts.push(c.x[1]);
        }
        record("cp_accept", &accepts);
        c.x[1] = server as u64;
        dispatch(c, 0x41);
        record("cp_accept_empty", &[c.x[0]]);

        // Both ends of a connection live on one rendezvous.
        let paired = {
            let table = p.handles.lock().unwrap();
            let c = table
                .session(connects[1] as u32)
                .and_then(|s| s.guest_client_link());
            let s = table
                .session(accepts[1] as u32)
                .and_then(|s| s.guest_server_link());
            matches!((c, s), (Some(c), Some(s)) if Arc::ptr_eq(&c, &s))
        };
        record("cp_pair", &[paired as u64]);

        c.x[0] = server as u64;
        dispatch(c, 0x16);
        c.x[1] = client_port as u64;
        dispatch(c, 0x72);
        record("cp_closed", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn a_client_port_connects_the_documented_number_of_sessions() {
        run_guest(client_port_sessions);
        assert_eq!(
            observed("cp_create_errors"),
            vec![
                RESULT_INVALID_ENUM,
                RESULT_INVALID_ENUM,
                RESULT_INVALID_ENUM
            ],
            "max_sessions is -1 or 0..=0x3f and is_light is a bool"
        );
        let create = observed("cp_create");
        assert_eq!(create[0], RESULT_SUCCESS);
        assert_ne!(create[1], 0, "the server port is returned in W1");
        assert_ne!(create[2], 0, "the client port is returned in W2");
        assert_eq!(observed("cp_server_as_client"), vec![RESULT_INVALID_HANDLE]);
        assert_eq!(observed("cp_unknown"), vec![RESULT_INVALID_HANDLE]);
        let connects = observed("cp_connect");
        assert_eq!(
            connects,
            vec![RESULT_SUCCESS, connects[1], RESULT_SUCCESS, connects[3]]
        );
        assert_ne!(connects[1], 0);
        assert_ne!(connects[3], 0);
        assert_ne!(
            connects[1], connects[3],
            "each connection is its own session"
        );
        assert_eq!(
            observed("cp_limit"),
            vec![RESULT_LIMIT_REACHED],
            "the port's max_sessions is reached"
        );
        let accepts = observed("cp_accept");
        assert_eq!(accepts[0], RESULT_SUCCESS);
        assert_eq!(accepts[2], RESULT_SUCCESS);
        assert_ne!(accepts[1], 0);
        assert_ne!(accepts[3], 0);
        assert_eq!(observed("cp_accept_empty"), vec![RESULT_NOT_FOUND]);
        assert_eq!(
            observed("cp_pair"),
            vec![1],
            "the client and server handles share one session pair"
        );
        assert_eq!(
            observed("cp_closed"),
            vec![RESULT_NOT_FOUND],
            "a closed port cannot connect"
        );
    }

    const SHM_A: u64 = 0x0F40_0000;
    const SHM_B: u64 = 0x0F40_1000;

    extern "C" fn shared_memory_blocks(c: &mut Cpu) {
        let p = c.process();
        let mut errors = Vec::new();
        for (size, owner, remote) in [
            (0u64, perm::RW as u64, perm::RW as u64),
            (PAGE + 1, perm::RW as u64, perm::RW as u64),
            (PAGE, perm::RX as u64, perm::R as u64),
            (PAGE, perm::RW as u64, perm::RX as u64),
        ] {
            c.x[1] = size;
            c.x[2] = owner;
            c.x[3] = remote;
            dispatch(c, 0x50);
            errors.push(c.x[0]);
        }
        record("shm_create_errors", &errors);

        c.x[1] = PAGE;
        c.x[2] = perm::RW as u64;
        c.x[3] = perm::R as u64;
        dispatch(c, 0x50);
        let owner = c.x[1] as u32;
        record("shm_create", &[c.x[0], owner as u64]);

        // The creator maps with its own permission, the block's size, a page
        // aligned address, and only once.
        let mut owner_errors = Vec::new();
        for (addr, size, map_perm) in [
            (SHM_A, PAGE, perm::R as u64),
            (SHM_A, PAGE * 2, perm::RW as u64),
            (SHM_A + 1, PAGE, perm::RW as u64),
            (SHM_A, PAGE, perm::RX as u64),
        ] {
            c.x[0] = owner as u64;
            c.x[1] = addr;
            c.x[2] = size;
            c.x[3] = map_perm;
            dispatch(c, 0x13);
            owner_errors.push(c.x[0]);
        }
        record("shm_owner_errors", &owner_errors);

        c.x[0] = owner as u64;
        c.x[1] = SHM_A;
        c.x[2] = PAGE;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x13);
        let mapped = c.x[0];
        p.memory.write(SHM_A, b"shared");
        c.x[0] = owner as u64;
        c.x[1] = SHM_A;
        c.x[2] = PAGE;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x13);
        let twice = c.x[0];
        c.x[0] = owner as u64;
        c.x[1] = SHM_A;
        c.x[2] = PAGE * 2;
        dispatch(c, 0x14);
        let bad_size = c.x[0];
        c.x[0] = owner as u64;
        c.x[1] = SHM_A;
        c.x[2] = PAGE;
        dispatch(c, 0x14);
        let unmapped = c.x[0];
        let gone = !p.memory.is_mapped(SHM_A, 1);

        // A borrower's handle is what a handle copy from another process
        // inserts; the same block then maps with the remote permission.
        let block = p
            .handles
            .lock()
            .unwrap()
            .shared_memory(owner)
            .map(|(m, _)| m)
            .expect("the owner handle is a shared memory block");
        let borrower = p
            .handles
            .lock()
            .unwrap()
            .insert_shared_memory(block, SharedMemoryRole::Borrower);
        c.x[0] = borrower as u64;
        c.x[1] = SHM_A;
        c.x[2] = PAGE;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x13);
        let borrower_rw = c.x[0];
        c.x[0] = borrower as u64;
        c.x[1] = SHM_A;
        c.x[2] = PAGE;
        c.x[3] = perm::R as u64;
        dispatch(c, 0x13);
        let borrower_r = c.x[0];
        let contents = p.memory.read(SHM_A, 6).unwrap_or_default();
        c.x[0] = borrower as u64;
        c.x[1] = SHM_A;
        c.x[2] = PAGE;
        dispatch(c, 0x14);
        let unmapped_borrower = c.x[0];

        // The block keeps the contents for the next mapping.
        c.x[0] = owner as u64;
        c.x[1] = SHM_B;
        c.x[2] = PAGE;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x13);
        let remapped = c.x[0];
        let kept = p.memory.read(SHM_B, 6).unwrap_or_default();
        c.x[0] = owner as u64;
        c.x[1] = SHM_B;
        c.x[2] = PAGE;
        dispatch(c, 0x14);

        // MemoryPermission_DontCare lets the borrower choose.
        c.x[1] = PAGE;
        c.x[2] = perm::RW as u64;
        c.x[3] = crate::kernel::MEMORY_PERMISSION_DONT_CARE as u64;
        dispatch(c, 0x50);
        let flexible = c.x[1] as u32;
        let block = p
            .handles
            .lock()
            .unwrap()
            .shared_memory(flexible)
            .map(|(m, _)| m)
            .expect("the second block is a shared memory object");
        let flexible_borrower = p
            .handles
            .lock()
            .unwrap()
            .insert_shared_memory(block, SharedMemoryRole::Borrower);
        c.x[0] = flexible_borrower as u64;
        c.x[1] = SHM_B;
        c.x[2] = PAGE;
        c.x[3] = perm::NONE as u64;
        dispatch(c, 0x13);
        let dont_care = c.x[0];
        c.x[0] = flexible_borrower as u64;
        c.x[1] = SHM_B;
        c.x[2] = PAGE;
        dispatch(c, 0x14);

        c.x[0] = 0xdead;
        c.x[1] = SHM_B;
        c.x[2] = PAGE;
        c.x[3] = perm::R as u64;
        dispatch(c, 0x13);
        let bad_handle = c.x[0];

        record(
            "shm",
            &[
                mapped,
                twice,
                bad_size,
                unmapped,
                gone as u64,
                borrower_rw,
                borrower_r,
                unmapped_borrower,
                remapped,
                dont_care,
                bad_handle,
            ],
        );
        record("shm_contents", &[eight(&contents), eight(&kept)]);
        c.halt();
    }

    #[test]
    fn shared_memory_checks_creator_and_borrower_permissions() {
        run_guest(shared_memory_blocks);
        assert_eq!(
            observed("shm_create_errors"),
            vec![
                RESULT_INVALID_SIZE,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
            ]
        );
        let create = observed("shm_create");
        assert_eq!(create[0], RESULT_SUCCESS);
        assert_ne!(create[1], 0, "the block handle is returned in X1");
        assert_eq!(
            observed("shm_owner_errors"),
            vec![
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
            ]
        );
        let shm = observed("shm");
        assert_eq!(shm[0], RESULT_SUCCESS);
        assert_eq!(shm[1], RESULT_INVALID_MEMORY_STATE, "one mapping at a time");
        assert_eq!(shm[2], RESULT_INVALID_SIZE);
        assert_eq!(shm[3], RESULT_SUCCESS);
        assert_eq!(shm[4], 1, "unmapping releases the pages");
        assert_eq!(
            shm[5], RESULT_INVALID_NEW_MEMORY_PERMISSION,
            "a borrower maps with the remote permission"
        );
        assert_eq!(shm[6], RESULT_SUCCESS);
        assert_eq!(shm[7], RESULT_SUCCESS);
        assert_eq!(
            shm[8], RESULT_SUCCESS,
            "the owner can map the block again at another address"
        );
        assert_eq!(
            shm[9], RESULT_SUCCESS,
            "MemoryPermission_DontCare accepts any valid permission"
        );
        assert_eq!(shm[10], RESULT_INVALID_HANDLE);
        let contents = observed("shm_contents");
        assert_eq!(contents[0], eight(b"shared"), "the block maps its contents");
        assert_eq!(
            contents[1],
            eight(b"shared"),
            "and keeps them across mappings"
        );
    }

    const INFO_PAGE: u64 = 0x0F50_0000;

    extern "C" fn system_reports(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(INFO_PAGE, PAGE, state::HEAP, perm::RW)
            .expect("the info page maps");
        let used = p.used_memory();
        let mut info = Vec::new();
        for (id, sub) in [(0u64, 0u64), (0, 1), (0, 3), (1, 0), (2, 0), (2, 1)] {
            c.x[1] = id;
            c.x[2] = 0;
            c.x[3] = sub;
            dispatch(c, 0x6f);
            info.push(c.x[0]);
            info.push(c.x[1]);
        }
        record("sys_info", &info);
        record("sys_used", &[used]);

        let mut errors = Vec::new();
        for (id, sub) in [(3u64, 0u64), (0, 4), (2, 2)] {
            c.x[1] = id;
            c.x[3] = sub;
            dispatch(c, 0x6f);
            errors.push(c.x[0]);
        }
        record("sys_info_errors", &errors);

        c.x[0] = 0; // the wrapper's output pointer; the kernel ignores it
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = 0;
        dispatch(c, 0x7c);
        let (state_result, state) = (c.x[0], c.x[1]);
        c.x[0] = 0;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = 1;
        dispatch(c, 0x7c);
        let bad_type = c.x[0];
        c.x[0] = 0;
        c.x[1] = 0xdead;
        c.x[2] = 0;
        dispatch(c, 0x7c);
        record("process_info", &[state_result, state, bad_type, c.x[0]]);

        c.x[1] = INFO_PAGE + 0x34;
        dispatch(c, 0x54);
        record("phys_query", &[c.x[0], c.x[1], c.x[2], c.x[3]]);
        c.x[1] = INFO_PAGE + PAGE;
        dispatch(c, 0x54);
        record("phys_unmapped", &[c.x[0]]);

        let mut registers = Vec::new();
        for addr in [
            0x7001_9054u64,
            0x7001_9052,
            0x7001_9010,
            0x7000_e400,
            0x1234_5678,
        ] {
            c.x[1] = addr;
            c.x[2] = 0;
            c.x[3] = 0;
            dispatch(c, 0x4e);
            registers.push(c.x[0]);
        }
        record("registers", &registers);
        c.halt();
    }

    #[test]
    fn system_info_process_info_and_physical_queries() {
        run_guest(system_reports);
        let info = observed("sys_info");
        let used = observed("sys_used")[0];
        assert_eq!(
            &info[..],
            &[
                RESULT_SUCCESS,
                TOTAL_MEMORY,
                RESULT_SUCCESS,
                TOTAL_MEMORY,
                RESULT_SUCCESS,
                TOTAL_MEMORY,
                RESULT_SUCCESS,
                used,
                RESULT_SUCCESS,
                threads::PROCESS_ID,
                RESULT_SUCCESS,
                threads::PROCESS_ID,
            ],
            "one pool reports its size and usage; the id range collapses"
        );
        assert_eq!(
            observed("sys_info_errors"),
            vec![
                RESULT_INVALID_ENUM,
                RESULT_INVALID_ENUM,
                RESULT_INVALID_ENUM
            ]
        );
        assert_eq!(
            observed("process_info"),
            vec![
                RESULT_SUCCESS,
                2,
                RESULT_INVALID_ENUM,
                RESULT_INVALID_HANDLE
            ],
            "the process is in ProcessState_Running"
        );
        assert_eq!(
            observed("phys_query"),
            vec![RESULT_SUCCESS, INFO_PAGE, INFO_PAGE, PAGE],
            "the physical address is the guest address in the flat model"
        );
        assert_eq!(observed("phys_unmapped"), vec![RESULT_INVALID_ADDRESS]);
        assert_eq!(
            observed("registers"),
            vec![
                RESULT_NOT_IMPLEMENTED,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_ADDRESS,
                RESULT_NOT_IMPLEMENTED,
                RESULT_INVALID_ADDRESS,
            ],
            "MC/PMC registers are unmodelled; anything else is an invalid address"
        );
    }
}
