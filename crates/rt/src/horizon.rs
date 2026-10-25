use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::arbiter;
use crate::cpu::Cpu;
use crate::kernel::{
    ClientPort, CodeMemory, ConnectError, Event, GuestSession, KObject, KThread, NamedPortError,
    Port, ResourceLimit, Session, SharedMapError, SharedMemory, SharedMemoryRole, ThreadContext,
    TransferMemory,
};
use crate::mem::{
    ADDR_MASK, DEVICE_SPACE_LIMIT, DeviceAddressSpace, DeviceError, IO_POOL_BASE, IO_POOL_END,
    IoPool, IoRegion, MirrorError, PAGE, PhysicalError, device_perm, io_mapping, perm, state,
};
use crate::process::Process;
use crate::services::sm::Sm;
use crate::threads;

pub const TICK_HZ: u64 = 19_200_000;

pub const RESULT_SUCCESS: u64 = 0;
pub const RESULT_NOT_IMPLEMENTED: u64 = 0x4201;
pub const RESULT_INVALID_SIZE: u64 = 0xca01;
/// `ResultOutOfMemory`; `svcSetHeapSize` reports it when the request does not
/// fit the heap region (the misaligned and above-4-GiB cases are `0xca01`).
pub const RESULT_OUT_OF_MEMORY: u64 = 0xd001;
pub const RESULT_INVALID_ENUM: u64 = 0xf001;
pub const RESULT_INVALID_ADDRESS: u64 = 0xcc01;
pub const RESULT_INVALID_MEMORY_REGION: u64 = 0xdc01;
pub const RESULT_LIMIT_REACHED: u64 = 0x10801;
pub const RESULT_INVALID_NEW_MEMORY_PERMISSION: u64 = 0xd801;
pub const RESULT_INVALID_POINTER: u64 = 0xe601;
pub const RESULT_BUSY: u64 = 0xf401;
pub const RESULT_TOO_MANY_HANDLES: u64 = 0xee01;
pub const RESULT_REMOTE_DEAD: u64 = 0xf601;
pub const RESULT_OUT_OF_RANGE: u64 = 0xee01;
pub const RESULT_INVALID_THREAD_ID: u64 = 0x40c01;
pub const RESULT_INVALID_ID: u64 = 0x40e01;
pub const RESULT_TERMINATION_REQUESTED: u64 = 0x7601;

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
/// `svcSetHeapSize` accepts a size of at most 4 GiB (`0xca01` above it).
const MAX_HEAP_SIZE: u64 = 0x1_0000_0000;

/// Guest regions a process asks the kernel about.
///
/// The A64 layout is the 39-bit one the runtime has always used and does not
/// change. A 32-bit process cannot see anything above 4 GiB, so its regions
/// live in the low half and the heap region is the last one: `svcSetHeapSize`
/// caps a request at 4 GiB and a 32-bit title asks for up to that, which only
/// fits above the map region. The 32-bit ASLR layout documented for 1.0.0 puts
/// the map region at `0x4000_0000` (1 GiB) with the heap region after it, so the
/// runtime keeps that order:
///
/// ```text
/// 0x0800_0000 code (the loader's LOAD_BASE) .. 0x2000_0000
/// 0x2000_0000 stack region (256 MiB)
/// 0x3000_0000 TLS (grows up to the map region)
/// 0x4000_0000 map/alias region (1 GiB)
/// 0x8000_0000 heap region (2 GiB, up to 4 GiB)
/// ```
///
/// `tls_base` is the only region the kernel does not expose through
/// `svcGetInfo`; it needs its own hole so a guest that maps the whole alias
/// region cannot take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            heap_base: 0x8000_0000,
            heap_region_size: 0x8000_0000,
            stack_base: 0x2000_0000,
            stack_region_size: 0x1000_0000,
            alias_base: 0x4000_0000,
            alias_size: 0x4000_0000,
            tls_base: 0x3000_0000,
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
        // `svcKernelDebug` (0x3C) and `svcChangeKernelTraceState` (0x3D) are
        // void calls that the retail kernel stubs out: they neither return a
        // result nor write guest memory. The runtime has no kernel trace
        // buffer, so ignoring them is the documented behavior.
        0x39 => create_io_pool(c),
        0x3a => create_io_region(c),
        0x3c | 0x3d => {}
        0x40 => create_session(c),
        0x41 => accept_session(c),
        0x42 => reply_and_receive_light(c),
        0x43 => reply_and_receive(c, c.x[1], c.x[2], c.x[3] as u32, c.x[4] as i64),
        0x44 => reply_and_receive_with_user_buffer(c),
        0x45 => threads::create_event(c),
        0x46 => map_io_region(c),
        0x47 => unmap_io_region(c),
        0x48 => map_physical_memory_unsafe(c),
        0x49 => unmap_physical_memory_unsafe(c),
        0x4a => set_unsafe_limit(c),
        0x4b => create_code_memory(c),
        0x4c => control_code_memory(c),
        0x4d => sleep_system(c),
        0x4e => read_write_register(c),
        0x4f => set_process_activity(c),
        0x50 => create_shared_memory(c),
        0x51 => map_transfer_memory(c),
        0x52 => unmap_transfer_memory(c),
        0x53 => create_interrupt_event(c),
        0x54 => query_physical_address(c),
        0x55 => query_memory_mapping(c),
        0x56 => create_device_address_space(c),
        0x57 | 0x58 => attach_device_address_space(c, imm),
        0x59..=0x5b => map_device_address_space(c, imm),
        0x5c => unmap_device_address_space(c),
        0x5d..=0x5f => process_data_cache(c, imm),
        // The debug SVCs that need a `KDebug` object to attach to, stop or
        // touch another process keep the retail answers: a console without
        // debug mode reports 0x4201 and one without a debug object reports an
        // invalid handle. The calls that only observe or rewrite the caller's
        // own process are served below (`debug_target`): the runtime is
        // single-process and its host is also its debugger, so they expose the
        // only process there is. Hardware breakpoints are recorded but never
        // armed.
        0x60 => debug_needs_debugger(c, "svcDebugActiveProcess"),
        0x61 => debug_needs_debugger(c, "svcBreakDebugProcess"),
        0x62 => debug_needs_debugger(c, "svcTerminateDebugProcess"),
        0x63 => debug_needs_handle(c, "svcGetDebugEvent"),
        0x64 => debug_needs_debugger(c, "svcContinueDebugEvent"),
        0x65 => get_process_list(c),
        0x66 => get_thread_list(c),
        0x67 => get_debug_thread_context(c),
        0x68 => set_debug_thread_context(c),
        0x69 => debug_needs_handle(c, "svcQueryDebugProcessMemory"),
        0x6a => debug_needs_handle(c, "svcReadDebugProcessMemory"),
        0x6b => debug_needs_debugger(c, "svcWriteDebugProcessMemory"),
        0x6c => set_hardware_break_point(c),
        0x6d => get_debug_thread_param(c),
        0x6f => get_system_info(c),
        0x70 => create_port(c),
        0x71 => manage_named_port(c),
        0x72 => connect_to_port(c),
        0x73 => set_process_memory_permission(c),
        0x74 | 0x75 => map_unmap_process_memory(c, imm),
        0x76 => query_process_memory(c),
        0x77 | 0x78 => map_unmap_process_code_memory(c, imm),
        0x79 => create_process(c),
        0x7a => start_process(c),
        0x7b => terminate_process(c),
        0x7c => get_process_info(c),
        0x7d => create_resource_limit(c),
        0x7e => set_resource_limit_limit_value(c),
        0x7f => call_secure_monitor(c),
        0x90 => map_insecure_physical_memory(c),
        0x91 => unmap_insecure_physical_memory(c),
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
        // The device families are served by the device memory model in `mem`:
        // an IO pool is the PCIe A2 aperture (0x39/0x3A), an IO region is a
        // claimed page range with a real host buffer behind it (0x46/0x47), and
        // a device address space is an IOMMU page table that aliases guest
        // pages of this process (0x55-0x5C). Register ranges
        // (`MemoryMapping_IoRegister`) and any range the model cannot back are
        // refused with the documented error instead of being faked; see
        // `create_io_region` and `query_memory_mapping`.
        //
        // 0x90/0x91 are the insecure physical family: they allocate new guest
        // pages, so they are served from the ASLR region as MemType_Insecure
        // (see `map_insecure_physical_memory`). Nothing here invents device
        // memory.
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
/// `Err` is the `svcReplyAndReceive` result: an unmapped reply buffer is
/// `0xe601` and a reply with Send/Receive/Exchange descriptors is `0xe801`
/// (HIPC page, Server).
fn send_reply(c: &Cpu, session: &Arc<Session>, buffer: (u64, u64)) -> Result<(), u64> {
    let Some(link) = session.guest_server_link() else {
        return Ok(());
    };
    if !link.in_service() {
        return Ok(());
    }
    let p = c.process();
    let Some(bytes) = p.memory.read(buffer.0, buffer.1) else {
        p.warn_once(format!(
            "ReplyAndReceive: the reply buffer {:#x} is not mapped",
            buffer.0
        ));
        return Err(RESULT_INVALID_POINTER);
    };
    if crate::ipc::reply_result(&bytes) != 0 {
        p.warn_once("ReplyAndReceive: a reply may not carry Send/Receive/Exchange buffers".into());
        return Err(RESULT_INVALID_COMBINATION);
    }
    // The pair wakes the process itself, so a host service posting from its own
    // thread reaches the blocked client too.
    link.post_reply(bytes);
    Ok(())
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
        Some(session) => match send_reply(c, &session, buffer) {
            Ok(()) => true,
            Err(result) => {
                c.x[0] = result;
                false
            }
        },
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
            if let Err(result) = send_reply(c, &session, (c.tpidrro_el0, IPC_BUFFER_SIZE)) {
                c.x[0] = result;
                return;
            }
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

fn release_transfer_memory(p: &Process, handle: u32, tmem: &TransferMemory) {
    if !tmem.release(&p.memory) {
        p.warn_once(format!(
            "svcCloseHandle({handle:#x}): the transfer memory at {:#x} could not release its pages",
            tmem.addr
        ));
    }
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
            // Transfer memory is a lease on the creator's pages: the last handle
            // clears the lock and restores their state and permission, so the
            // heap can adopt them again (the kernel does the same).
            Some(KObject::TransferMemory(tmem)) => {
                release_transfer_memory(p, handle, &tmem);
            }
            // An IoRegion owns guest pages while it is mapped. Closing the last
            // handle writes them back and releases them, so a guest that closes
            // the handle before unmapping does not leak them.
            Some(KObject::IoRegion(region)) => {
                let released = region.release(&p.memory);
                if !released {
                    p.warn_once(format!(
                        "svcCloseHandle({handle:#x}): the IO region at {:#x} could not release its guest mapping",
                        region.physical
                    ));
                }
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
    // SVC page, SetHeapSize: a misaligned size or one above 4 GiB is 0xca01;
    // a size that does not fit the heap region is 0xd001 (OutOfMemory).
    if !size.is_multiple_of(HEAP_ALIGN) || size > MAX_HEAP_SIZE {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    if size > layout.heap_region_size {
        c.x[0] = RESULT_OUT_OF_MEMORY;
        return;
    }
    let p = c.process();
    let mut heap = p.heap.lock().unwrap();
    if size > *heap {
        if let Err(e) = p.memory.map_as(
            layout.heap_base + *heap,
            size - *heap,
            state::HEAP,
            perm::RW,
        ) {
            drop(heap);
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                // A live mapping sits where the heap would grow; the kernel
                // reports the region's state, not a host failure.
                p.warn_once(format!(
                    "svcSetHeapSize({size:#x}): the heap region is not free: {e}"
                ));
                c.x[0] = RESULT_INVALID_MEMORY_STATE;
            } else {
                fatal(c, format!("could not map a {size:#x}-byte heap: {e}"));
            }
            return;
        }
    } else if size < *heap {
        // Shrinking gives the tail back, as the kernel does.
        if let Err(e) = p.memory.unmap(layout.heap_base + size, *heap - size) {
            drop(heap);
            fatal(c, format!("could not shrink the heap to {size:#x}: {e}"));
            return;
        }
    }
    *heap = size;
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
/// and permission change; `svcMapTransferMemory` aliases them elsewhere. The
/// retag is a lease: closing the handle restores the pages (see `close_handle`).
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
    let Some(saved) = p
        .memory
        .retag_leased(addr, end, state::TRANSFER, perm, &LENDABLE)
    else {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    };
    let tmem = TransferMemory::new(addr, size, perm, saved);
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

/// `svcMapPhysicalMemoryUnsafe` and `svcUnmapPhysicalMemoryUnsafe` take the same
/// alias-region address as `svcMapPhysicalMemory`, but charge pool partition 0
/// and are gated by `svcSetUnsafeLimit`. The runtime has one pool partition, so
/// the bytes a guest maps here are ordinary host RAM it can back, never the
/// device aperture: the range must lie inside the alias region and outside the
/// PCIe window. A guest that tries the aperture gets the kernel's invalid
/// range result rather than fabricated device memory.
fn unsafe_physical_range(c: &mut Cpu) -> Option<(u64, u64)> {
    let (addr, size) = (c.x[0], c.x[1]);
    let layout = c.process().layout;
    let error = if !addr.is_multiple_of(PAGE) {
        RESULT_INVALID_ADDRESS
    } else if size == 0 || !size.is_multiple_of(PAGE) {
        RESULT_INVALID_SIZE
    } else if addr.checked_add(size).is_none()
        || addr < layout.alias_base
        || addr + size > layout.alias_base + layout.alias_size
        // The A64 alias region starts above the aperture, so this only rejects
        // a layout whose alias covers 0x1200_0000..0x2000_0000.
        || (addr < IO_POOL_END && addr + size > IO_POOL_BASE)
    {
        RESULT_INVALID_MEMORY_REGION
    } else {
        return Some((addr, addr + size));
    };
    c.x[0] = error;
    None
}

fn map_physical_memory_unsafe(c: &mut Cpu) {
    let Some((start, end)) = unsafe_physical_range(c) else {
        return;
    };
    let size = end - start;
    let p = c.process();
    let (limit, used) = {
        let table = p.handles.lock().unwrap();
        (table.unsafe_limit(), table.unsafe_mapped())
    };
    if used.checked_add(size).is_none_or(|n| n > limit) {
        p.warn_once(format!(
            "svcMapPhysicalMemoryUnsafe({start:#x}, {size:#x}): {used:#x} bytes are already mapped through the unsafe calls and the limit is {limit:#x} (svcSetUnsafeLimit)"
        ));
        c.x[0] = RESULT_LIMIT_REACHED;
        return;
    }
    let budget = TOTAL_MEMORY.saturating_sub(p.used_memory());
    let r = p.memory.map_physical(start, end, budget);
    physical_result(c, "svcMapPhysicalMemoryUnsafe", r, |_| {
        // The allowance counts the range the guest asked for, not only the
        // pages this call had to commit: it is a pool-0 budget, not a host
        // allocation log.
        p.physical
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        p.handles.lock().unwrap().charge_unsafe(size);
    });
}

fn unmap_physical_memory_unsafe(c: &mut Cpu) {
    let Some((start, end)) = unsafe_physical_range(c) else {
        return;
    };
    let p = c.process();
    let r = p.memory.unmap_physical(start, end);
    physical_result(c, "svcUnmapPhysicalMemoryUnsafe", r, |held| {
        p.physical
            .fetch_sub(held, std::sync::atomic::Ordering::Relaxed);
        p.handles.lock().unwrap().release_unsafe(held);
    });
}

/// `svcSetUnsafeLimit` (0x4A): the ceiling for the unsafe physical calls. On
/// hardware it starts at zero, so a guest that never calls this cannot use
/// `svcMapPhysicalMemoryUnsafe`; the runtime keeps that rule.
fn set_unsafe_limit(c: &mut Cpu) {
    let limit = c.x[0];
    c.process().handles.lock().unwrap().set_unsafe_limit(limit);
    c.process().log_ipc(format!("SetUnsafeLimit({limit:#x})"));
    ok(c);
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

/// The largest `DeviceName` the SVC page documents (`TSECB1` = 0x25). Any other
/// value is the kernel's invalid-enum result.
const MAX_DEVICE_NAME: u32 = 0x25;

/// The kernel frame `svcMapDeviceAddressSpaceAligned` (and the aligned path of
/// `svcMapDeviceAddressSpace`) requires the process and device addresses to
/// share.
const DEVICE_ALIGN_MASK: u64 = (1 << 22) - 1;

/// Turns a device-model refusal into the kernel's result code.
fn device_error(p: &Process, name: &str, e: DeviceError) -> u64 {
    match e {
        DeviceError::Size => RESULT_INVALID_SIZE,
        DeviceError::Address => RESULT_INVALID_ADDRESS,
        DeviceError::Range | DeviceError::Registers => RESULT_INVALID_MEMORY_REGION,
        DeviceError::State => RESULT_INVALID_MEMORY_STATE,
        DeviceError::Permission => RESULT_INVALID_NEW_MEMORY_PERMISSION,
        DeviceError::Host => {
            p.warn_once(format!("{name}: the host refused the device backing store"));
            RESULT_LIMIT_REACHED
        }
    }
}

/// `svcCreateIoPool` (0x39): the handle of the PCIe A2 pool, the aperture
/// 0x1200_0000..0x2000_0000. The pool claims no bytes until regions are created
/// inside it, so creating it costs nothing; closing the handle is the destroy
/// path (there is no separate destroy SVC). The runtime models one aperture per
/// process, so a second pool is refused while one, or any region of one, is
/// still alive: the claim could otherwise be backed twice.
fn create_io_pool(c: &mut Cpu) {
    if c.x[1] as u32 != 0 {
        // IoPoolType_PcieA2 is the only documented pool type.
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    let p = c.process();
    let mut table = p.handles.lock().unwrap();
    if table
        .objects()
        .any(|o| matches!(o, KObject::IoPool(_) | KObject::IoRegion(_)))
    {
        drop(table);
        p.warn_once(
            "svcCreateIoPool: the runtime models one PCIe A2 aperture per process; close every pool and region first"
                .into(),
        );
        c.x[0] = RESULT_INVALID_STATE;
        return;
    }
    let handle = table.insert(KObject::IoPool(IoPool::pcie_a2()));
    drop(table);
    p.log_ipc(format!("CreateIoPool(PcieA2) -> {handle:#x}"));
    ok1(c, handle as u64);
}

/// `svcCreateIoRegion` (0x3A): claims `physical..physical+size` inside the pool
/// with `mapping` and `perm`. A memory range gets a zero-filled host buffer, so
/// the guest can map and use it; an `IoRegister` range is refused with the
/// kernel's invalid-range result and a log, because there are no registers to
/// map.
fn create_io_region(c: &mut Cpu) {
    let (handle, physical, size) = (c.x[1] as u32, c.x[2], c.x[3]);
    let (mapping, permission) = (c.x[4] as u32, c.x[5] as u32);
    let p = c.process();
    let Some(pool) = p.handles.lock().unwrap().io_pool(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    if mapping > io_mapping::MEMORY {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    match IoRegion::new(pool, physical, size, mapping, permission) {
        Ok(region) => {
            let handle = p.handles.lock().unwrap().insert(KObject::IoRegion(region));
            p.log_ipc(format!(
                "CreateIoRegion({physical:#x}+{size:#x}, mapping={mapping}, perm={permission:#x}) -> {handle:#x}"
            ));
            ok1(c, handle as u64);
        }
        Err(DeviceError::Registers) => {
            p.warn_once(format!(
                "svcCreateIoRegion({physical:#x}+{size:#x}): MemoryMapping_IoRegister asks for device registers, which only the console has; the runtime backs Memory and Uncached regions only"
            ));
            c.x[0] = RESULT_INVALID_MEMORY_REGION;
        }
        Err(e) => c.x[0] = device_error(p, "svcCreateIoRegion", e),
    }
}

/// `svcMapIoRegion` (0x46): exposes the region at `addr` with `perm`, which must
/// be the permission the region was created with. The guest pages become
/// `MemType_Io` and hold the region's contents; only one mapping exists per
/// region at a time.
fn map_io_region(c: &mut Cpu) {
    let (handle, addr, size, permission) = (c.x[0] as u32, c.x[1], c.x[2], c.x[3] as u32);
    let p = c.process();
    let Some(region) = p.handles.lock().unwrap().io_region(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    c.x[0] = match region.map(&p.memory, addr, size, permission) {
        Ok(()) => {
            p.log_ipc(format!(
                "MapIoRegion({handle:#x}, {addr:#x}+{size:#x}, perm={permission:#x})"
            ));
            RESULT_SUCCESS
        }
        Err(e) => device_error(p, "svcMapIoRegion", e),
    };
}

/// `svcUnmapIoRegion` (0x47): copies the guest pages back into the region and
/// frees them.
fn unmap_io_region(c: &mut Cpu) {
    let (handle, addr, size) = (c.x[0] as u32, c.x[1], c.x[2]);
    let p = c.process();
    let Some(region) = p.handles.lock().unwrap().io_region(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    c.x[0] = match region.unmap(&p.memory, addr, size) {
        Ok(()) => RESULT_SUCCESS,
        Err(e) => device_error(p, "svcUnmapIoRegion", e),
    };
}

/// `svcQueryMemoryMapping` (0x55, 10.0.0+; it replaced `svcQueryIoMapping`):
/// the virtual address behind a physical range, plus the size the caller asked
/// for. The runtime's physical space is the guest's virtual space (guest RAM is
/// identity mapped), so a mapped physical page resolves to its mapping base. In
/// the PCIe aperture only a live `svcMapIoRegion` mapping can answer, which is
/// the one place a device physical address has a guest view. Anything else is
/// the kernel's not-found result; the kernel maps firmware regions here that
/// the runtime does not have.
fn query_memory_mapping(c: &mut Cpu) {
    let (out_addr, out_size, physical, size) = (c.x[1], c.x[2], c.x[3], c.x[4]);
    let p = c.process();
    if out_addr == 0 || out_size == 0 {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    // A modern kernel only accepts a page-aligned physical address (the
    // sub-page values name firmware regions), so an unaligned one is not found.
    if !physical.is_multiple_of(PAGE) || physical.checked_add(size).is_none() {
        c.x[0] = RESULT_NOT_FOUND;
        return;
    }
    let found = if (IO_POOL_BASE..IO_POOL_END).contains(&physical) {
        io_region_view(p, physical, size)
    } else {
        let r = p.memory.query(physical);
        (r.state != state::FREE && r.range.end >= physical + size).then_some(r.range.start)
    };
    let Some(address) = found else {
        p.warn_once(format!(
            "svcQueryMemoryMapping({physical:#x}+{size:#x}): nothing in the runtime backs that physical range"
        ));
        c.x[0] = RESULT_NOT_FOUND;
        return;
    };
    if !p.memory.is_mapped(out_addr, 8) || !p.memory.is_mapped(out_size, 8) {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    if !p.memory.write(out_addr, &address.to_le_bytes())
        || !p.memory.write(out_size, &size.to_le_bytes())
    {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    c.x[0] = RESULT_SUCCESS;
}

/// The guest view of a pool-backed physical range, when a live IoRegion covers
/// it and is currently mapped.
fn io_region_view(p: &Process, physical: u64, size: u64) -> Option<u64> {
    let handles = p.handles.lock().unwrap();
    handles.objects().find_map(|object| {
        let KObject::IoRegion(region) = object else {
            return None;
        };
        let at = region.mapped_at()?;
        let end = physical.checked_add(size)?;
        (physical >= region.physical && end <= region.physical + region.size)
            .then(|| at + (physical - region.physical))
    })
}

/// `svcCreateDeviceAddressSpace` (0x56): an IOMMU window `base..base+size` for
/// `svcMapDeviceAddressSpace*`. The runtime caps it at
/// [`DEVICE_SPACE_LIMIT`] and backs it with the guest's own pages when they are
/// mapped into it.
fn create_device_address_space(c: &mut Cpu) {
    let (base, size) = (c.x[1], c.x[2]);
    let error = if !base.is_multiple_of(PAGE) {
        RESULT_INVALID_ADDRESS
    } else if size == 0 || !size.is_multiple_of(PAGE) {
        RESULT_INVALID_SIZE
    } else if base
        .checked_add(size)
        .is_none_or(|end| end > DEVICE_SPACE_LIMIT)
    {
        RESULT_INVALID_MEMORY_REGION
    } else {
        RESULT_SUCCESS
    };
    if error != RESULT_SUCCESS {
        c.x[0] = error;
        return;
    }
    let p = c.process();
    let handle = p
        .handles
        .lock()
        .unwrap()
        .insert(KObject::DeviceAddressSpace(DeviceAddressSpace::new(
            base, size,
        )));
    p.log_ipc(format!(
        "CreateDeviceAddressSpace({base:#x}+{size:#x}) -> {handle:#x}"
    ));
    ok1(c, handle as u64);
}

/// `svcAttachDeviceAddressSpace` (0x57) and `svcDetachDeviceAddressSpace`
/// (0x58). The attachment is recorded per device name and duplicate attach or
/// detach of a device that is not attached is the kernel's invalid state. No
/// hardware device exists in the runtime, so nothing reads the address space
/// until a synthetic device uses it.
fn attach_device_address_space(c: &mut Cpu, imm: u16) {
    let (device, handle) = (c.x[0] as u32, c.x[1] as u32);
    if device > MAX_DEVICE_NAME {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    let p = c.process();
    let Some(das) = p.handles.lock().unwrap().device_address_space(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    let attach = imm == 0x57;
    let name = if attach {
        "svcAttachDeviceAddressSpace"
    } else {
        "svcDetachDeviceAddressSpace"
    };
    let r = if attach {
        das.attach(device)
    } else {
        das.detach(device)
    };
    c.x[0] = match r {
        Ok(()) => {
            if attach {
                p.warn_once(format!(
                    "svcAttachDeviceAddressSpace(device {device}): the runtime records the attachment, but no hardware device reads this address space"
                ));
            }
            p.log_ipc(format!("{name}(device={device}, handle={handle:#x})"));
            RESULT_SUCCESS
        }
        Err(DeviceError::State) => RESULT_INVALID_STATE,
        Err(e) => device_error(p, name, e),
    };
}

/// `svcMapDeviceAddressSpaceByForce` (0x59), `...Aligned` (0x5A) and
/// `MapDeviceAddressSpace` (0x5B): alias a range of the current process at a
/// device address. Only the current process exists, so only the
/// current-process alias is a valid process handle.
///
/// 0x59 and 0x5A pass the permission in the low 16 bits of an option word; the
/// high bits select an IO mapping the runtime cannot model, so they are refused
/// as an invalid enum. 0x5A and 0x5B require the process and device addresses to
/// share their 4 MiB frame, as the kernel does; 0x5B returns the mapped size in
/// X1.
fn map_device_address_space(c: &mut Cpu, imm: u16) {
    let (handle, process, process_addr, size, device_addr, permission) = if imm == 0x5b {
        (
            c.x[1] as u32,
            c.x[2] as u32,
            c.x[3],
            c.x[4],
            c.x[5],
            c.x[6] as u32,
        )
    } else {
        let option = c.x[5] as u32;
        (
            c.x[0] as u32,
            c.x[1] as u32,
            c.x[2],
            c.x[3],
            c.x[4],
            option & 0xffff,
        )
    };
    let p = c.process();
    if imm != 0x5b && (c.x[5] as u32) & !0xffff != 0 {
        p.warn_once(format!(
            "svc {imm:#04x}: the option word {:#x} asks for an IO mapping indicator the runtime cannot model",
            c.x[5] as u32
        ));
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    if process != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    if !process_addr.is_multiple_of(PAGE) || !device_addr.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    if process_addr.checked_add(size).is_none() {
        c.x[0] = RESULT_INVALID_MEMORY_REGION;
        return;
    }
    if imm != 0x59 && (process_addr & DEVICE_ALIGN_MASK) != (device_addr & DEVICE_ALIGN_MASK) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if !device_perm(permission) {
        c.x[0] = RESULT_INVALID_NEW_MEMORY_PERMISSION;
        return;
    }
    let Some(das) = p.handles.lock().unwrap().device_address_space(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    match das.map(&p.memory, process_addr, size, device_addr, permission) {
        Ok(()) => {
            p.log_ipc(format!(
                "MapDeviceAddressSpace({device_addr:#x} <- {process_addr:#x}+{size:#x}, perm={permission:#x})"
            ));
            if imm == 0x5b {
                ok1(c, size);
            } else {
                ok(c);
            }
        }
        Err(e) => c.x[0] = device_error(p, "svcMapDeviceAddressSpace", e),
    }
}

/// `svcUnmapDeviceAddressSpace` (0x5C): removes the device pages in the range
/// from the address space. The guest pages are not touched; the device side
/// simply stops resolving them. The device table is keyed by device address, so
/// the guest range is only validated for alignment and overflow, not for still
/// being mapped: unmapping a device range whose guest pages went away is valid
/// cleanup, and refusing it would leave the device window allocated.
fn unmap_device_address_space(c: &mut Cpu) {
    let (handle, process, process_addr, size, device_addr) =
        (c.x[0] as u32, c.x[1] as u32, c.x[2], c.x[3], c.x[4]);
    let p = c.process();
    if process != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    if !process_addr.is_multiple_of(PAGE) || !device_addr.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    if process_addr.checked_add(size).is_none() {
        c.x[0] = RESULT_INVALID_MEMORY_REGION;
        return;
    }
    let Some(das) = p.handles.lock().unwrap().device_address_space(handle) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    c.x[0] = match das.unmap(device_addr, size) {
        Ok(()) => RESULT_SUCCESS,
        Err(e) => device_error(p, "svcUnmapDeviceAddressSpace", e),
    };
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
    write_memory_info(c, out, addr);
}

/// The `MemoryInfo`/`PageInfo` pair `svcQueryMemory` (0x06) and
/// `svcQueryProcessMemory` (0x76) report: a 0x28-byte `MemoryInfo` at `out` and
/// the 32-bit `PageInfo` in W1 (the wrappers store it in their out pointer).
///
/// `State` carries the runtime's page tag, which is the low byte of the
/// documented `MemoryType` (Switchbrew's table) and not the whole
/// `MemoryState` word: reporting the documented flag bits makes MK8D's 32-bit
/// rtld reject its own module scan (see
/// `docs/plan/findings/switchbrew-core.md`). `Attribute`, `IpcRefCount` and
/// `DeviceRefCount` are zero: the runtime does not track them.
fn write_memory_info(c: &mut Cpu, out: u64, addr: u64) {
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
        // InfoType_SystemResourceSizeUsed: the pages of the system resource the
        // process has mapped so far, in bytes (`p.physical`). The old code
        // divided by 512, which reported an eighth of the real usage.
        17 => p.physical.load(std::sync::atomic::Ordering::Relaxed),
        18 => p.title_id,
        19 => threads::PROCESS_ID,
        // InfoType_UserExceptionContextAddress: the exception frame lives at
        // the start of the thread's TLS region, which is TPIDRRO_EL0.
        20 => c.tpidrro_el0,
        21 => TOTAL_MEMORY - p.system_resource_size,
        // InfoType_SystemResourceSizeUsed: the memory in use outside the system
        // resource. Subtracting `physical` here made SMBW compute a heap 32 MiB
        // smaller and abort while loading its heap definition, so the value is
        // the used memory minus the system resource size.
        22 => p.used_memory().saturating_sub(p.system_resource_size),
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

/// `svcSleepSystem` (0x4D) is a void call that asks the system to suspend. The
/// runtime models no system power state and the host keeps running, so the
/// request is reported once and dropped; there is no result code to return.
fn sleep_system(c: &mut Cpu) {
    c.process().warn_once(
        "svcSleepSystem: the runtime has no system sleep state; the guest keeps running".into(),
    );
}

/// `svcSetProcessActivity` (0x4F): W0 is a process handle and W1 the
/// `ProcessActivity` (0 = none, 1 = runnable). The kernel validates the enum,
/// resolves the handle and refuses the current process with `ResultBusy`; with
/// one process and no debug handle for another, those three answers cover every
/// call.
fn set_process_activity(c: &mut Cpu) {
    if c.x[1] > 1 {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    if c.x[0] as u32 != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    c.x[0] = RESULT_BUSY;
}

/// The debug SVCs that require a `KDebug` object. A retail console only exposes
/// them on development units and answers `0x4201` (`NotImplemented`) from an
/// application; the runtime creates no such object, so it reports the same code
/// instead of pretending to attach, stop or continue a process.
fn debug_needs_debugger(c: &mut Cpu, name: &str) {
    c.process()
        .warn_once(format!("{name}: the runtime has no debug object"));
    c.x[0] = RESULT_NOT_IMPLEMENTED;
}

/// Debug SVCs that take a debug handle but are not gated on debug mode on
/// hardware: with no `KDebug` object, every handle is invalid.
fn debug_needs_handle(c: &mut Cpu, name: &str) {
    c.process().warn_once(format!(
        "{name}: the runtime creates no debug objects, so the handle is invalid"
    ));
    c.x[0] = RESULT_INVALID_HANDLE;
}

/// The largest element count a process/thread list accepts. The kernel checks
/// `0 <= max_out_count <= i32::MAX / sizeof(u64)` and reports `0xEE01`
/// (`OutOfRange`) otherwise.
const MAX_DEBUG_LIST: i32 = i32::MAX / 8;

fn debug_list_max(c: &mut Cpu) -> Option<u64> {
    let max = c.x[2] as i32;
    if !(0..=MAX_DEBUG_LIST).contains(&max) {
        c.x[0] = RESULT_OUT_OF_RANGE;
        return None;
    }
    let max = max as u64;
    let buffer = c.x[1];
    if max > 0 && !c.process().memory.is_mapped(buffer, max * 8) {
        c.x[0] = RESULT_INVALID_POINTER;
        return None;
    }
    Some(max)
}

fn write_debug_list(c: &mut Cpu, buffer: u64, ids: &[u64], max: u64) {
    let take = ids.len().min(max as usize);
    let mut bytes = Vec::with_capacity(take * 8);
    for id in &ids[..take] {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    if take > 0 && !c.process().memory.write(buffer, &bytes) {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    // The count is the total number of objects, not how many fit in the array.
    ok1(c, ids.len() as u64);
}

/// `svcGetProcessList` (0x65): X1 is the output array, W2 its element count and
/// W1 returns the number of live processes (all of them, even when the array is
/// shorter). The runtime runs one process, so the list is just its id.
///
/// A retail console gates this on debug mode and would answer `0x4201`; the
/// runtime's host is also its debugger, so hiding the only process there is
/// would serve nothing. The calls that need a debug object keep that answer.
fn get_process_list(c: &mut Cpu) {
    let buffer = c.x[1];
    let Some(max) = debug_list_max(c) else {
        return;
    };
    write_debug_list(c, buffer, &[threads::PROCESS_ID], max);
}

/// `svcGetThreadList` (0x66): X1 = ids, W2 = count, W3 = debug handle, W1 =
/// total threads. A zero handle asks for the global list, which is this
/// process's list because it is the only process; a `KDebug` handle or another
/// process's handle cannot exist here and is the kernel's invalid handle.
fn get_thread_list(c: &mut Cpu) {
    let buffer = c.x[1];
    let Some(max) = debug_list_max(c) else {
        return;
    };
    if c.x[3] as u32 != 0 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    // Not while writing guest memory: the handle table is held only to copy the
    // ids, and a thread that exits meanwhile must not keep it locked.
    let ids: Vec<u64> = c
        .process()
        .handles
        .lock()
        .unwrap()
        .threads()
        .iter()
        .map(|t| t.id)
        .collect();
    write_debug_list(c, buffer, &ids, max);
}

/// The runtime has no `KDebug` object, so the debug calls that only ever inspect
/// or rewrite the caller's own process take the current-process pseudo handle in
/// the debug-handle slot. Every other value is the kernel's invalid handle.
fn debug_target(handle: u32) -> bool {
    handle == CURRENT_PROCESS as u32
}

fn thread_by_id(p: &Process, id: u64) -> Option<Arc<KThread>> {
    p.handles
        .lock()
        .unwrap()
        .threads()
        .into_iter()
        .find(|t| t.id == id)
}

fn live_thread_context(c: &Cpu) -> ThreadContext {
    ThreadContext {
        x: c.x,
        sp: c.sp,
        pc: c.pc,
        psr: c.nzcv as u32,
        v: c.v,
        fpcr: c.fpcr as u32,
        fpsr: c.fpsr as u32,
        tpidr: c.tpidr_el0,
    }
}

/// Writes the register groups selected by `flags` back into the calling Cpu.
/// x0 is skipped: the SVC result lands there after the context is applied.
fn apply_thread_context(c: &mut Cpu, context: &ThreadContext, flags: u32) {
    if flags & ThreadContext::FLAG_GPRS != 0 {
        c.x[1..29].copy_from_slice(&context.x[1..29]);
    }
    if flags & ThreadContext::FLAG_CONTROL != 0 {
        c.x[29] = context.x[29];
        c.x[30] = context.x[30];
        c.sp = context.sp;
        c.pc = context.pc;
        c.nzcv = context.psr as u64;
        c.tpidr_el0 = context.tpidr;
    }
    if flags & ThreadContext::FLAG_VECTOR != 0 {
        c.v = context.v;
    }
    if flags & ThreadContext::FLAG_FP_CONTROL != 0 {
        c.fpcr = context.fpcr as u64;
        c.fpsr = context.fpsr as u64;
    }
}

/// `svcGetDebugThreadContext` (0x67): X0 is the output `ThreadContext`, X1 the
/// debug handle, X2 the thread id and W3 the `ThreadContextFlags`.
///
/// Only the caller's own process is reachable (see [`debug_target`]). The
/// caller's registers are live; a thread the guest paused with
/// `svcSetThreadActivity` reports its saved context. A running thread cannot be
/// inspected because the runtime never stops it behind its back — the kernel
/// would suspend it for the debugger — which is `ResultBusy`.
fn get_debug_thread_context(c: &mut Cpu) {
    let (out, debug, id, flags) = (c.x[0], c.x[1] as u32, c.x[2], c.x[3] as u32);
    if flags & !ThreadContext::FLAGS_ALL != 0 {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    if !debug_target(debug) {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    let p = c.process();
    let Some(t) = thread_by_id(p, id) else {
        c.x[0] = RESULT_INVALID_ID;
        return;
    };
    let context = if Arc::ptr_eq(&t, &c.thread) {
        live_thread_context(c)
    } else if t.exited.load(Ordering::Relaxed) {
        c.x[0] = RESULT_TERMINATION_REQUESTED;
        return;
    } else if t.paused.load(Ordering::Relaxed) {
        crate::sched::wait_stopped(c, &t);
        t.context.lock().unwrap().clone()
    } else {
        c.x[0] = RESULT_BUSY;
        return;
    };
    let Some(mut bytes) = p.memory.read(out, ThreadContext::SIZE as u64) else {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    };
    context.merge_into(&mut bytes, flags);
    if !p.memory.write(out, &bytes) {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    ok(c);
}

/// `svcSetDebugThreadContext` (0x68): W0 the debug handle, X1 the thread id, X2
/// the input `ThreadContext` and W3 the flags.
///
/// The live register set of the calling thread is the only one the runtime can
/// really rewrite. Another thread's saved context is a snapshot taken when it
/// parked; the thread resumes from its live `Cpu`, so writing the snapshot
/// would be a lie, and the runtime does not suspend a thread to make it true.
/// That case is `ResultBusy`.
fn set_debug_thread_context(c: &mut Cpu) {
    let (debug, id, ptr, flags) = (c.x[0] as u32, c.x[1], c.x[2], c.x[3] as u32);
    if flags & !ThreadContext::FLAGS_ALL != 0 {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    if !debug_target(debug) {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    let p = c.process();
    let Some(t) = thread_by_id(p, id) else {
        c.x[0] = RESULT_INVALID_ID;
        return;
    };
    if t.exited.load(Ordering::Relaxed) {
        c.x[0] = RESULT_TERMINATION_REQUESTED;
        return;
    }
    if !Arc::ptr_eq(&t, &c.thread) {
        c.x[0] = RESULT_BUSY;
        return;
    }
    let Some(bytes) = p.memory.read(ptr, ThreadContext::SIZE as u64) else {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    };
    let Some(context) = ThreadContext::from_bytes(&bytes) else {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    };
    apply_thread_context(c, &context, flags);
    ok(c);
}

/// `svcGetDebugThreadParam` (0x6D): X2 is the debug handle, X3 the thread id
/// and W4 the `DebugThreadParam`; the two outputs are X1 and W2.
///
/// The parameters are `0` priority, `1` state, `2` ideal core, `3` current core
/// and `4` affinity mask. The state is `{suspended, idle, running, queued}`
/// reported as `{5, 0, 1, 4}` with bit 0 of W2 marking a debug suspend (never
/// set here) and bit 1 a user suspend (`svcSetThreadActivity`).
fn get_debug_thread_param(c: &mut Cpu) {
    let (debug, id, param) = (c.x[2] as u32, c.x[3], c.x[4]);
    if !debug_target(debug) {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    let p = c.process();
    let Some(t) = thread_by_id(p, id) else {
        c.x[0] = RESULT_INVALID_THREAD_ID;
        return;
    };
    match param {
        0 => {
            c.x[0] = RESULT_SUCCESS;
            c.x[2] = t.effective_priority() as u32 as u64;
        }
        1 => {
            let suspended = t.paused.load(Ordering::Relaxed);
            let state = if t.exited.load(Ordering::Relaxed) {
                4
            } else if suspended {
                5
            } else if t.started.load(Ordering::Relaxed) {
                1
            } else {
                0
            };
            c.x[0] = RESULT_SUCCESS;
            c.x[1] = state;
            c.x[2] = (suspended as u64) << 1;
        }
        2 => {
            c.x[0] = RESULT_SUCCESS;
            c.x[2] = t.ideal_core.load(Ordering::Relaxed) as u32 as u64;
        }
        3 => {
            c.x[0] = RESULT_SUCCESS;
            c.x[2] = t.core.load(Ordering::Relaxed) as u32 as u64;
        }
        4 => {
            c.x[0] = RESULT_SUCCESS;
            c.x[2] = t.affinity.load(Ordering::Relaxed) & 0xffff_ffff;
        }
        _ => c.x[0] = RESULT_INVALID_ENUM,
    }
}

/// `svcSetHardwareBreakPoint` (0x6C): W0 names the register (0-5 are the six
/// breakpoints, 0x10-0x13 the four watchpoints), X1 the flag bits the guest may
/// set and X2 the address. The kernel masks the flags and ORs in its own
/// security bits (0x4004 for a breakpoint, 0x104004 for a watchpoint).
///
/// The runtime records the register and never arms it: guest code runs as
/// recompiled host functions, not under a per-instruction debug unit, so there
/// is nothing that could trap at `value`.
fn set_hardware_break_point(c: &mut Cpu) {
    let (name, flags, value) = (c.x[0] as u32, c.x[1], c.x[2]);
    let (slot, allowed, forced) = match name {
        0..=5 => (name as usize, 0x7f01e1u64, 0x4004u64),
        0x10..=0x13 => (6 + (name - 0x10) as usize, 0xff0f1ff9u64, 0x104004u64),
        _ => {
            c.x[0] = RESULT_INVALID_ENUM;
            return;
        }
    };
    if flags & !allowed != 0 {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    if !crate::kernel::set_hardware_breakpoint(slot, flags | forced, value) {
        c.x[0] = RESULT_INVALID_ENUM;
        return;
    }
    ok(c);
}

/// `svcSetProcessMemoryPermission` (0x73): X1 address, X2 size, W3 the new
/// permission and X0 the process handle. The kernel only accepts the code
/// states here — the heap belongs to `svcSetMemoryPermission` — and reports
/// `0xD401` for anything else. Making `CodeStatic` writable turns it into
/// `CodeMutable`, as the loader's transitions expect.
fn set_process_memory_permission(c: &mut Cpu) {
    let (handle, addr, size, perm) = (c.x[0] as u32, c.x[1], c.x[2], c.x[3] as u32);
    if !addr.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    let Some(end) = addr.checked_add(size) else {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    };
    if !matches!(perm, perm::NONE | perm::R | perm::RW | perm::RX) {
        c.x[0] = RESULT_INVALID_NEW_MEMORY_PERMISSION;
        return;
    }
    if handle != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    let p = c.process();
    let code = [
        state::CODE_STATIC,
        state::CODE_MUTABLE,
        state::CODE_READONLY,
        state::CODE_WRITABLE,
    ];
    let writable = perm & perm::RW == perm::RW;
    let result = if writable && p.memory.query(addr).state == state::CODE_STATIC {
        p.memory
            .retag(addr, end, state::CODE_MUTABLE, perm, &[state::CODE_STATIC])
    } else {
        p.memory.protect(addr, end, perm, &code)
    };
    c.x[0] = if result {
        RESULT_SUCCESS
    } else {
        RESULT_INVALID_MEMORY_STATE
    };
}

/// `svcMapProcessMemory` (0x74) and `svcUnmapProcessMemory` (0x75) map pages of
/// *another* process into the caller. The kernel resolves the handle without
/// the current-process pseudo alias, and the runtime has no second process and
/// no handle for one, so after the address/size validation every call is the
/// kernel's invalid handle.
fn map_unmap_process_memory(c: &mut Cpu, imm: u16) {
    let (dst, src, size) = (c.x[0], c.x[2], c.x[3]);
    if !dst.is_multiple_of(PAGE) || !src.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    if dst.checked_add(size).is_none() || src.checked_add(size).is_none() {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    }
    let name = if imm == 0x74 {
        "svcMapProcessMemory"
    } else {
        "svcUnmapProcessMemory"
    };
    c.process().warn_once(format!(
        "{name}: the runtime has one process and the current-process alias is not accepted"
    ));
    c.x[0] = RESULT_INVALID_HANDLE;
}

/// `svcMapProcessCodeMemory` (0x77) and `svcUnmapProcessCodeMemory` (0x78):
/// same rule, and the kernel documents that this pair does not accept the
/// current-process alias at all.
fn map_unmap_process_code_memory(c: &mut Cpu, imm: u16) {
    let (dst, src, size) = (c.x[1], c.x[2], c.x[3]);
    if !dst.is_multiple_of(PAGE) || !src.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return;
    }
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    if dst.checked_add(size).is_none() || src.checked_add(size).is_none() {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    }
    let name = if imm == 0x77 {
        "svcMapProcessCodeMemory"
    } else {
        "svcUnmapProcessCodeMemory"
    };
    c.process().warn_once(format!(
        "{name}: the runtime has one process and refuses the current-process alias"
    ));
    c.x[0] = RESULT_INVALID_HANDLE;
}

/// `svcQueryProcessMemory` (0x76): like `svcQueryMemory`, with the process
/// handle in W2 and the address in X3. The only process the handle can name is
/// the caller, whose memory the query already reads.
fn query_process_memory(c: &mut Cpu) {
    if c.x[2] as u32 != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    let (out, addr) = (c.x[0], c.x[3]);
    write_memory_info(c, out, addr);
}

/// The A64 `CreateProcessParameter` is 0x30 bytes: a 12-byte process name, the
/// process category, the title id, the code address, the code page count, the
/// flags (bit 0 AArch64, bits 1-3 address space, bit 4 `EnableDebug`, bit 5
/// ASLR, bit 6 application, bits 7-10 memory region, bit 11 optimized
/// allocation), the resource-limit handle and the system-resource page count.
const CREATE_PROCESS_PARAMETER_SIZE: u64 = 0x30;

/// `svcCreateProcess` (0x79): X1 points at the parameter block, X2 at the
/// capability array and W3 is its length. A second process needs its own
/// address space, memory pool and scheduler, which the runtime does not build;
/// the pointers are validated so a malformed call is reported as such, and then
/// the call is refused instead of handing back a handle with nothing behind it.
fn create_process(c: &mut Cpu) {
    let (params, caps, num_caps) = (c.x[1], c.x[2], c.x[3] as i32);
    if num_caps < 0 {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    let cap_bytes = (num_caps as u64) * 4;
    if cap_bytes > 0 && !c.process().memory.is_mapped(caps, cap_bytes) {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    if !c
        .process()
        .memory
        .is_mapped(params, CREATE_PROCESS_PARAMETER_SIZE)
    {
        c.x[0] = RESULT_INVALID_POINTER;
        return;
    }
    c.process()
        .warn_once("svcCreateProcess: the runtime builds one process per run".into());
    c.x[0] = RESULT_NOT_IMPLEMENTED;
}

/// `svcStartProcess` (0x7A): starts a process made by `svcCreateProcess`. No
/// such process can exist, and the only process handle left is the caller's own
/// pseudo handle, which is already running; the kernel's answer for starting a
/// running process is `ResultInvalidState`.
fn start_process(c: &mut Cpu) {
    if c.x[0] as u32 != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    c.x[0] = RESULT_INVALID_STATE;
}

/// `svcTerminateProcess` (0x7B): the kernel accepts a process handle, including
/// the current-process alias, and terminating the caller goes through the same
/// exit path as `svcExitProcess`.
fn terminate_process(c: &mut Cpu) {
    if c.x[0] as u32 != CURRENT_PROCESS as u32 {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    }
    c.process().terminate(0);
    c.halt();
}

/// `SmcResult_NotImplemented` from the secure monitor's own result space.
const SMC_NOT_IMPLEMENTED: u64 = 1;

/// `svcCallSecureMonitor` (0x7F): X0 is the SMC function id and X1-X7 its
/// arguments; the monitor's result comes back in the same registers. The
/// runtime has no secure monitor (there is no TrustZone behind it), so the call
/// is answered with the monitor's documented `NotImplemented` result instead of
/// inventing the output of an SMC function.
fn call_secure_monitor(c: &mut Cpu) {
    c.process().warn_once(
        "svcCallSecureMonitor: the runtime has no secure monitor; returning SMC NotImplemented"
            .into(),
    );
    c.x[0] = SMC_NOT_IMPLEMENTED;
}

/// `svcMapInsecurePhysicalMemory` (0x90) and
/// `svcUnmapInsecurePhysicalMemory` (0x91), 15.0.0+: the guest asks for a
/// mapping of freshly allocated pages from the insecure pool. The runtime has
/// no separate pool, so the pages come from the ASLR region as zeroed guest RAM
/// with `MemType_Insecure` and are charged to the process's memory budget the
/// same way `svcMapPhysicalMemory` charges the alias region.
fn insecure_physical_range(c: &mut Cpu) -> Option<(u64, u64)> {
    let (addr, size) = (c.x[0], c.x[1]);
    let layout = c.process().layout;
    if size == 0 || !size.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_SIZE;
        return None;
    }
    if !addr.is_multiple_of(PAGE) {
        c.x[0] = RESULT_INVALID_ADDRESS;
        return None;
    }
    let Some(end) = addr.checked_add(size) else {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return None;
    };
    if addr < layout.aslr_base || end > layout.aslr_base + layout.aslr_size {
        c.x[0] = RESULT_INVALID_MEMORY_REGION;
        return None;
    }
    Some((addr, end))
}

fn map_insecure_physical_memory(c: &mut Cpu) {
    let Some((start, end)) = insecure_physical_range(c) else {
        return;
    };
    let p = c.process();
    let budget = TOTAL_MEMORY.saturating_sub(p.used_memory());
    match p.memory.map_insecure(start, end, budget) {
        Ok(bytes) => {
            p.physical.fetch_add(bytes, Ordering::Relaxed);
            ok(c);
        }
        Err(PhysicalError::State) => c.x[0] = RESULT_INVALID_MEMORY_REGION,
        Err(PhysicalError::Limit) => c.x[0] = RESULT_LIMIT_REACHED,
        Err(PhysicalError::Host(e)) => fatal(
            c,
            format!("svcMapInsecurePhysicalMemory: the host refused the mapping ({e})"),
        ),
    }
}

fn unmap_insecure_physical_memory(c: &mut Cpu) {
    let Some((start, end)) = insecure_physical_range(c) else {
        return;
    };
    let p = c.process();
    match p.memory.unmap_insecure(start, end) {
        Ok(bytes) => {
            p.physical.fetch_sub(bytes, Ordering::Relaxed);
            ok(c);
        }
        Err(PhysicalError::State) => c.x[0] = RESULT_INVALID_MEMORY_STATE,
        Err(PhysicalError::Limit) => c.x[0] = RESULT_LIMIT_REACHED,
        Err(PhysicalError::Host(e)) => fatal(
            c,
            format!("svcUnmapInsecurePhysicalMemory: the host refused the mapping ({e})"),
        ),
    }
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
/// exception path: a 0x148-byte exception stack followed by the register
/// frame, which is 0x78 bytes on A64 and 0x44 on A32.
pub const EXCEPTION_FRAME_OFFSET: u64 = 0x148;
/// `ExceptionFrameA64`: GPRs 0-8, lr, sp, pc, pstate, afsr0, afsr1, esr, far.
const EXCEPTION_FRAME_SIZE_A64: u64 = 0x78;
/// `ExceptionFrameA32`: GPRs 0-7, sp, lr, pc, tpidr_el0 (= 1), cpsr, afsr0,
/// afsr1, esr, far.
const EXCEPTION_FRAME_SIZE_A32: u64 = 0x44;

/// The frame size the kernel writes for `arch`.
fn exception_frame_size(arch: kirikae_loader::Arch) -> u64 {
    match arch {
        kirikae_loader::Arch::A64 => EXCEPTION_FRAME_SIZE_A64,
        kirikae_loader::Arch::A32 => EXCEPTION_FRAME_SIZE_A32,
    }
}

/// The `ExceptionFrameA64`/`ExceptionFrameA32` the kernel stores at
/// `EXCEPTION_FRAME_OFFSET`. The 32-bit frame keeps the guest's registers in
/// 4-byte fields and, as the SVC page documents, stores `tpidr_el0 = 1`.
fn encode_exception_frame(arch: kirikae_loader::Arch, c: &Cpu, fault_pc: u64) -> Vec<u8> {
    let mut frame = vec![0u8; exception_frame_size(arch) as usize];
    let (gprs, lr_at, sp_at, pc_at, cpsr_at) = match arch {
        kirikae_loader::Arch::A64 => (9, 0x48, 0x50, 0x58, 0x60),
        kirikae_loader::Arch::A32 => (8, 0x24, 0x20, 0x28, 0x30),
    };
    let width = if arch == kirikae_loader::Arch::A32 {
        4
    } else {
        8
    };
    for (i, reg) in c.x[..gprs].iter().enumerate() {
        let at = i * width;
        if width == 4 {
            frame[at..at + 4].copy_from_slice(&(*reg as u32).to_le_bytes());
        } else {
            frame[at..at + 8].copy_from_slice(&reg.to_le_bytes());
        }
    }
    let put = |frame: &mut Vec<u8>, at: usize, value: u64| {
        if width == 4 {
            frame[at..at + 4].copy_from_slice(&(value as u32).to_le_bytes());
        } else {
            frame[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
    };
    // The A32 lr is r14; the runtime keeps it in x30 for both architectures.
    put(&mut frame, lr_at, c.x[30]);
    put(&mut frame, sp_at, c.sp);
    put(&mut frame, pc_at, fault_pc);
    put(&mut frame, cpsr_at, (c.nzcv as u32) as u64);
    if arch == kirikae_loader::Arch::A32 {
        // tpidr_el0 = 1 and far = the faulting address.
        put(&mut frame, 0x2c, 1);
        put(&mut frame, 0x40, fault_pc);
    }
    frame
}

/// Applies a frame written by [`encode_exception_frame`] back to `c`, which is
/// what `svcReturnFromException` does with the crt0's edits. The runtime keeps
/// TPIDR_EL0 alone: guests use it as their thread pointer and the frame carries
/// no usable value for it.
fn decode_exception_frame(arch: kirikae_loader::Arch, c: &mut Cpu, frame: &[u8]) {
    let (gprs, lr_at, sp_at, pc_at, cpsr_at) = match arch {
        kirikae_loader::Arch::A64 => (9, 0x48, 0x50, 0x58, 0x60),
        kirikae_loader::Arch::A32 => (8, 0x24, 0x20, 0x28, 0x30),
    };
    let width = if arch == kirikae_loader::Arch::A32 {
        4
    } else {
        8
    };
    let get = |at: usize| -> u64 {
        if width == 4 {
            u32::from_le_bytes(frame[at..at + 4].try_into().unwrap_or_default()) as u64
        } else {
            u64::from_le_bytes(frame[at..at + 8].try_into().unwrap_or_default())
        }
    };
    for (i, reg) in c.x[..gprs].iter_mut().enumerate() {
        *reg = get(i * width);
    }
    c.x[30] = get(lr_at);
    c.sp = get(sp_at);
    c.pc = get(pc_at);
    c.nzcv = get(cpsr_at);
}

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
    let frame = encode_exception_frame(p.arch, c, fault_pc);
    if !p.memory.is_mapped(frame_at, frame.len() as u64) {
        return false;
    }
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
    let p = c.process();
    let frame_at = c.tpidrro_el0 + EXCEPTION_FRAME_OFFSET;
    let size = exception_frame_size(p.arch);
    let Some(frame) = c.process().memory.read(frame_at, size) else {
        fatal(
            c,
            format!("svcReturnFromException cannot read the frame at {frame_at:#x}"),
        );
        return;
    };
    decode_exception_frame(p.arch, c, &frame);
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
        run_entries_with(entries, expected_exit, 0)
    }

    fn run_entries_with(
        entries: Vec<Entry>,
        expected_exit: i32,
        system_resource_size: u64,
    ) -> crate::process::RunResult {
        run_entries_arch(
            kirikae_loader::Arch::A64,
            entries,
            expected_exit,
            system_resource_size,
        )
    }

    fn run_entries_arch(
        arch: kirikae_loader::Arch,
        entries: Vec<Entry>,
        expected_exit: i32,
        system_resource_size: u64,
    ) -> crate::process::RunResult {
        let entries: &'static [Entry] = Box::leak(entries.into_boxed_slice());
        let exe = kirikae_loader::Executable {
            modules: Vec::new(),
            entry: entries[0].addr,
            arch,
            title_id: None,
            main_stack_size: None,
            system_resource_size,
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

    extern "C" fn info_type22(c: &mut Cpu) {
        let p = c.process();
        p.physical
            .store(0x1000, std::sync::atomic::Ordering::Relaxed);
        let used = p.used_memory();
        let system_resource = p.system_resource_size;
        c.x[1] = 22;
        c.x[2] = 0xffff_8001;
        dispatch(c, 0x29);
        record("info22", &[c.x[0], c.x[1], used, system_resource]);
        c.halt();
    }

    #[test]
    fn info_type22_reports_the_memory_outside_the_system_resource() {
        run_guest(info_type22);
        let got = observed("info22");
        assert_eq!(got[0], RESULT_SUCCESS);
        assert_eq!(
            got[1],
            got[2].saturating_sub(got[3]),
            "type 22 is used memory minus the system resource size"
        );
        assert_ne!(
            got[1],
            got[2].saturating_sub(0x1000),
            "a physical mapping must not be subtracted here"
        );
    }

    fn run_a32_guest(f: GuestFn) {
        run_entries_arch(
            kirikae_loader::Arch::A32,
            vec![Entry {
                addr: GUEST_ENTRY,
                func: f,
            }],
            0,
            0,
        );
    }

    /// The 32-bit layout: every region below 4 GiB, no overlaps, and a heap
    /// region that fits the 0x7800_0000 (1920 MiB) request MK8D makes.
    #[test]
    fn the_32_bit_layout_keeps_every_region_below_4_gib() {
        let l = layout(kirikae_loader::Arch::A32);
        assert!(l.heap_region_size >= 0x7800_0000);
        assert_eq!(l.heap_base + l.heap_region_size, 0x1_0000_0000);
        let code_end = l.aslr_base + l.aslr_size;
        let stack_end = l.stack_base + l.stack_region_size;
        let alias_end = l.alias_base + l.alias_size;
        assert_eq!(l.aslr_base, ASLR_BASE, "the loader loads at LOAD_BASE");
        assert_eq!(code_end, l.stack_base, "the stack region follows the code");
        assert!(
            l.tls_base >= stack_end && l.tls_base < l.alias_base,
            "TLS has its own hole between the stack and the map region"
        );
        assert_eq!(alias_end, l.heap_base, "the heap follows the map region");
        for (name, start, end) in [
            ("code", l.aslr_base, code_end),
            ("stack", l.stack_base, stack_end),
            ("tls", l.tls_base, l.alias_base),
            ("alias", l.alias_base, alias_end),
            ("heap", l.heap_base, l.heap_base + l.heap_region_size),
        ] {
            assert!(start < end, "{name} is not empty");
            assert!(end <= 0x1_0000_0000, "{name} leaves the 32-bit space");
        }
    }

    #[test]
    fn the_64_bit_layout_is_unchanged() {
        assert_eq!(
            layout(kirikae_loader::Arch::A64),
            Layout {
                aslr_base: 0x0800_0000,
                aslr_size: 0x7f_f800_0000,
                heap_base: 0x2_0000_0000,
                heap_region_size: 0x1_8000_0000,
                stack_base: 0x4_0000_0000,
                stack_region_size: 0x1_0000_0000,
                alias_base: 0x10_0000_0000,
                alias_size: 0x10_0000_0000,
                tls_base: 0x4_8000_0000,
            }
        );
    }

    /// The 1920 MiB heap a 32-bit title asks for, and both SetHeapSize error
    /// paths: misaligned/above 4 GiB is 0xca01, over the heap region is 0xd001.
    extern "C" fn a32_heap(c: &mut Cpu) {
        let p = c.process();
        c.x[1] = 0x7800_0000;
        dispatch(c, 0x01);
        let base = c.x[1];
        let mut grow = vec![c.x[0], base];
        grow.push(p.memory.is_mapped(base, 1) as u64);
        grow.push(p.memory.is_mapped(base + 0x7800_0000 - 1, 1) as u64);
        grow.push(p.memory.is_mapped(base + 0x7800_0000, 1) as u64);
        let r = p.memory.query(base + 0x1000);
        grow.extend([r.state as u64, r.perm as u64]);
        record("a32_grow", &grow);

        // Shrinking gives the tail back.
        c.x[1] = 0x20_0000;
        dispatch(c, 0x01);
        record(
            "a32_shrink",
            &[
                c.x[0],
                p.memory.is_mapped(base, 1) as u64,
                p.memory.is_mapped(base + 0x20_0000 - 1, 1) as u64,
                p.memory.is_mapped(base + 0x20_0000, 1) as u64,
            ],
        );

        let mut errors = Vec::new();
        for size in [0x78_0001u64, 0x1_0000_0000, 0x2_0000_0000, 0] {
            c.x[1] = size;
            dispatch(c, 0x01);
            errors.push(c.x[0]);
        }
        record("a32_errors", &errors);
        c.halt();
    }

    #[test]
    fn a_32_bit_guest_can_take_the_heap_mk8d_asks_for() {
        run_a32_guest(a32_heap);
        let grow = observed("a32_grow");
        assert_eq!(grow[0], RESULT_SUCCESS, "1920 MiB fits the 32-bit heap");
        assert_eq!(
            grow[1], 0x8000_0000,
            "the heap region starts at 0x8000_0000"
        );
        assert_eq!(
            &grow[2..5],
            &[1, 1, 0],
            "exactly the requested range is mapped"
        );
        assert_eq!(
            (grow[5], grow[6]),
            (state::HEAP as u64, perm::RW as u64),
            "the heap is Normal/SHARED-able RW memory"
        );
        assert_eq!(
            observed("a32_shrink"),
            vec![RESULT_SUCCESS, 1, 1, 0],
            "a smaller heap releases its tail"
        );
        assert_eq!(
            observed("a32_errors"),
            vec![
                RESULT_INVALID_SIZE,  // not 2 MiB aligned
                RESULT_OUT_OF_MEMORY, // 4 GiB: aligned, over the heap region
                RESULT_INVALID_SIZE,  // above 4 GiB
                RESULT_SUCCESS,       // a zero heap is a valid shrink
            ]
        );
    }

    const INFO_OUT: u64 = 0x0F60_0000;
    const INFO_HEAP: u64 = 0x0F61_0000;
    const INFO_CODE: u64 = 0x0F62_0000;

    extern "C" fn memory_info_states(c: &mut Cpu) {
        let p = c.process();
        p.memory
            .map_as(INFO_OUT, PAGE, state::HEAP, perm::RW)
            .expect("the info page maps");
        p.memory
            .map_as(INFO_HEAP, PAGE, state::HEAP, perm::RW)
            .expect("the heap page maps");
        p.memory
            .map_as(INFO_CODE, PAGE, state::CODE_STATIC, perm::RX)
            .expect("the code page maps");
        let mut out = Vec::new();
        for addr in [INFO_HEAP, INFO_CODE, INFO_HEAP + PAGE] {
            c.x[0] = INFO_OUT;
            c.x[2] = addr;
            dispatch(c, 0x06);
            let info = p.memory.read(INFO_OUT, 0x28).unwrap_or_default();
            let word =
                |at: usize| u32::from_le_bytes(info[at..at + 4].try_into().unwrap_or_default());
            let wide =
                |at: usize| u64::from_le_bytes(info[at..at + 8].try_into().unwrap_or_default());
            out.extend([c.x[0], c.x[1], wide(0), wide(8)]);
            out.extend([word(0x10) as u64, word(0x18) as u64]);
        }
        record("memory_info", &out);
        c.halt();
    }

    #[test]
    fn query_memory_reports_the_memory_info_layout() {
        run_guest(memory_info_states);
        let got = observed("memory_info");
        // `MemoryInfo`: base at 0, size at 8, state at 0x10, attribute at 0x14,
        // permission at 0x18, reference counts at 0x1C/0x20, padding at 0x24.
        // The state carries only the runtime's page tag (the MemoryType low
        // byte); the documented flag bits are a known gap (findings doc).
        assert_eq!(
            &got[..6],
            &[
                RESULT_SUCCESS,
                0,
                INFO_HEAP,
                PAGE,
                state::HEAP as u64,
                perm::RW as u64,
            ],
            "a Normal page with RW permission"
        );
        assert_eq!(
            &got[6..12],
            &[
                RESULT_SUCCESS,
                0,
                INFO_CODE,
                PAGE,
                state::CODE_STATIC as u64,
                perm::RX as u64,
            ],
            "a Code page with RX permission"
        );
        assert_eq!(
            &got[12..18],
            &[
                RESULT_SUCCESS,
                0,
                INFO_HEAP + PAGE,
                0xF000, // up to the next mapping
                state::FREE as u64,
                0,
            ],
            "an unmapped range reports the whole free region"
        );
    }

    /// The exception frame is 0x78 bytes on A64 and 0x44 on A32, at the
    /// documented offsets (`ExceptionFrameA64`/`ExceptionFrameA32`).
    extern "C" fn exception_frames(c: &mut Cpu) {
        let set = |c: &mut Cpu| {
            c.x[0] = 0x10;
            c.x[7] = 0x77;
            c.x[8] = 0x88;
            c.x[30] = 0x2222;
            c.sp = 0x3333;
            c.nzcv = 0x6000_0000;
        };
        let u32_at = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        let u64_at = |b: &[u8], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
        let mut out = Vec::new();

        set(c);
        let frame = encode_exception_frame(kirikae_loader::Arch::A64, c, 0x4444);
        out.push(frame.len() as u64);
        out.extend([
            u64_at(&frame, 0),
            u64_at(&frame, 0x40),
            u64_at(&frame, 0x48),
            u64_at(&frame, 0x50),
            u64_at(&frame, 0x58),
            u64_at(&frame, 0x60),
        ]);
        c.x = [0; 31];
        decode_exception_frame(kirikae_loader::Arch::A64, c, &frame);
        out.extend([c.x[8], c.sp, c.x[30], c.pc, c.nzcv]);

        set(c);
        let frame = encode_exception_frame(kirikae_loader::Arch::A32, c, 0x4444);
        out.push(frame.len() as u64);
        out.extend([
            u32_at(&frame, 0) as u64,
            u32_at(&frame, 0x1c) as u64,
            u32_at(&frame, 0x20) as u64,
            u32_at(&frame, 0x24) as u64,
            u32_at(&frame, 0x28) as u64,
            u32_at(&frame, 0x2c) as u64,
            u32_at(&frame, 0x30) as u64,
            u32_at(&frame, 0x40) as u64,
        ]);
        c.x = [0; 31];
        decode_exception_frame(kirikae_loader::Arch::A32, c, &frame);
        out.extend([c.x[0], c.x[7], c.sp, c.x[30], c.pc, c.nzcv]);
        record("frames", &out);
        c.halt();
    }

    #[test]
    fn exception_frames_follow_the_process_architecture() {
        run_guest(exception_frames);
        let got = observed("frames");
        assert_eq!(
            &got[..7],
            &[0x78, 0x10, 0x88, 0x2222, 0x3333, 0x4444, 0x6000_0000],
            "ExceptionFrameA64: 0x78 bytes, GPRs, lr, sp, pc, pstate"
        );
        assert_eq!(
            &got[7..12],
            &[0x88, 0x3333, 0x2222, 0x4444, 0x6000_0000],
            "the A64 frame restores x8, sp, lr, pc and pstate"
        );
        assert_eq!(
            &got[12..21],
            &[
                0x44,
                0x10,
                0x77,
                0x3333,
                0x2222,
                0x4444,
                1,
                0x6000_0000,
                0x4444
            ],
            "ExceptionFrameA32: 0x44 bytes, r0, r7, sp, lr, pc, tpidr_el0 = 1, cpsr, far"
        );
        assert_eq!(
            &got[21..],
            &[0x10, 0x77, 0x3333, 0x2222, 0x4444, 0x6000_0000],
            "the A32 frame restores r0, r7, sp, lr, pc and cpsr"
        );
    }

    const REPLY_LIST: u64 = 0x0F70_1000;

    /// A guest server whose reply carries a Send buffer: the HIPC page says
    /// `svcReplyAndReceive` refuses it with 0xe801.
    extern "C" fn reply_with_send_buffer(c: &mut Cpu) {
        let p = c.process();
        let link = GuestSession::new();
        link.send(vec![0u8; 0x10]).expect("the request queues");
        assert_eq!(
            link.take_request().map(|r| r.len()),
            Some(0x10),
            "the pair is in service"
        );
        let handle = p
            .handles
            .lock()
            .unwrap()
            .insert(KObject::Session(Session::guest_server(link)));
        p.memory
            .map_as(REPLY_LIST, PAGE, state::HEAP, perm::RW)
            .expect("the handle list maps");
        p.memory.write(REPLY_LIST, &handle.to_le_bytes());
        let h0 = crate::ipc::TYPE_REQUEST as u32 | 1 << 20;
        let mut reply = h0.to_le_bytes().to_vec();
        reply.extend_from_slice(&0u32.to_le_bytes());
        p.memory.write(c.tpidrro_el0, &reply);
        c.x[1] = REPLY_LIST;
        c.x[2] = 1;
        c.x[3] = handle as u64;
        c.x[4] = 0;
        dispatch(c, 0x43);
        record("bad_reply", &[c.x[0]]);
        c.halt();
    }

    #[test]
    fn a_reply_cannot_carry_transfer_buffers() {
        run_guest(reply_with_send_buffer);
        assert_eq!(
            observed("bad_reply"),
            vec![RESULT_INVALID_COMBINATION],
            "svcReplyAndReceive refuses Send/Receive/Exchange descriptors"
        );
    }

    const WAIT_LIST: u64 = 0x0F71_1000;

    extern "C" fn wait_counts(c: &mut Cpu) {
        c.process()
            .memory
            .map_as(WAIT_LIST, PAGE, state::HEAP, perm::RW)
            .expect("the handle list maps");
        let mut out = Vec::new();
        for count in [65i64, 0x41, -1] {
            c.x[1] = WAIT_LIST;
            c.x[2] = count as u64;
            c.x[3] = 0;
            dispatch(c, 0x18);
            out.push(c.x[0]);
        }
        record("wait_counts", &out);
        c.halt();
    }

    #[test]
    fn wait_synchronization_refuses_more_than_0x40_handles() {
        run_guest(wait_counts);
        assert_eq!(
            observed("wait_counts"),
            vec![RESULT_TOO_MANY_HANDLES; 3],
            "the SVC page caps HandlesNum at 0x40 (0xee01 OutOfRange)"
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

        // The closing handle releases the lease; the creator's heap tags return.
        c.x[0] = tmem as u64;
        dispatch(c, 0x16);
        let closed = c.x[0];
        let after = p.memory.query(TMEM_PAGE);
        record(
            "tmem_close",
            &[closed, after.state as u64, after.perm as u64],
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
        assert_eq!(
            observed("tmem_close"),
            vec![RESULT_SUCCESS, state::HEAP as u64, perm::RW as u64],
            "the last handle puts the creator's pages back as heap"
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

    /// The Brilliant Diamond shape: the heap commits a physical range, carves a
    /// transfer block out of it for bsd, closes the block, and then adopts the
    /// whole range. The live block is a conflict; the released one is heap.
    extern "C" fn transfer_memory_then_physical(c: &mut Cpu) {
        let p = c.process();
        let base = p.layout.alias_base + 0x1000_0000;
        let alias = base + 0x1_0000;
        let mut out = Vec::new();

        c.x[0] = base;
        c.x[1] = PAGE * 2;
        dispatch(c, 0x2c);
        out.push(c.x[0]);

        c.x[1] = base + PAGE;
        c.x[2] = PAGE;
        c.x[3] = perm::NONE as u64;
        dispatch(c, 0x15);
        let tmem = c.x[1] as u32;
        out.push(c.x[0]);

        c.x[0] = tmem as u64;
        c.x[1] = alias;
        c.x[2] = PAGE;
        c.x[3] = perm::NONE as u64;
        dispatch(c, 0x51);
        out.push(c.x[0]);
        p.memory.write(alias, b"alias");
        c.x[0] = tmem as u64;
        c.x[1] = alias;
        c.x[2] = PAGE;
        dispatch(c, 0x52);
        out.push(c.x[0]);
        out.push(eight(&p.memory.read(base + PAGE, 5).unwrap_or_default()));

        // A live transfer memory still refuses the adoption.
        c.x[0] = base;
        c.x[1] = PAGE * 4;
        dispatch(c, 0x2c);
        out.push(c.x[0]);
        out.push(p.memory.is_mapped(base + PAGE * 2, 1) as u64);
        c.x[0] = base;
        c.x[1] = PAGE * 4;
        dispatch(c, 0x2d);
        out.push(c.x[0]);

        c.x[0] = tmem as u64;
        dispatch(c, 0x16);
        out.push(c.x[0]);
        out.push(p.memory.query(base + PAGE).state as u64);

        c.x[0] = base;
        c.x[1] = PAGE * 4;
        dispatch(c, 0x2c);
        out.push(c.x[0]);
        let r = p.memory.query(base);
        out.push(r.state as u64);
        out.push(r.range.end - base);
        out.push(eight(&p.memory.read(base + PAGE, 5).unwrap_or_default()));
        record("tmem_physical", &out);
        c.halt();
    }

    #[test]
    fn a_guest_releases_transfer_memory_before_adopting_physical_memory() {
        run_entries_with(
            vec![Entry {
                addr: GUEST_ENTRY,
                func: transfer_memory_then_physical,
            }],
            0,
            0x400_0000,
        );
        assert_eq!(
            observed("tmem_physical"),
            vec![
                RESULT_SUCCESS,
                RESULT_SUCCESS,
                RESULT_SUCCESS,
                RESULT_SUCCESS,
                eight(b"alias"),
                RESULT_INVALID_MEMORY_STATE,
                0,
                RESULT_INVALID_MEMORY_STATE,
                RESULT_SUCCESS,
                state::HEAP as u64,
                RESULT_SUCCESS,
                state::HEAP as u64,
                PAGE * 4,
                eight(b"alias"),
            ],
            "the live block conflicts, the released one is adopted with its bytes"
        );
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
            .read(frame_at, EXCEPTION_FRAME_SIZE_A64)
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

    extern "C" fn process_activity_and_kernel_stubs(c: &mut Cpu) {
        let mut out = Vec::new();
        for (handle, activity) in [(CURRENT_PROCESS, 2u64), (CURRENT_PROCESS, 1), (0xdead, 1)] {
            c.x[0] = handle;
            c.x[1] = activity;
            dispatch(c, 0x4f);
            out.push(c.x[0]);
        }

        // SleepSystem is void; the runtime has no system state to change.
        c.x[0] = 0x1234;
        dispatch(c, 0x4d);
        out.push(c.x[0]);
        // KernelDebug and ChangeKernelTraceState are void and stubbed in retail.
        c.x[0] = 0x5678;
        dispatch(c, 0x3c);
        out.push(c.x[0]);
        c.x[0] = 0x9abc;
        dispatch(c, 0x3d);
        out.push(c.x[0]);

        record("process_activity", &out);
        c.halt();
    }

    #[test]
    fn process_activity_sleep_and_kernel_debug_are_safe() {
        run_guest(process_activity_and_kernel_stubs);
        assert_eq!(
            observed("process_activity"),
            vec![
                RESULT_INVALID_ENUM,
                RESULT_BUSY,
                RESULT_INVALID_HANDLE,
                0x1234,
                0x5678,
                0x9abc,
            ],
            "the enum, the handle and the current-process refusal are the kernel's"
        );
    }

    extern "C" fn process_and_thread_lists(c: &mut Cpu) {
        let buffer = c.sp - 0x200;
        let extra = parked_thread(0x99);
        c.process()
            .handles
            .lock()
            .unwrap()
            .insert(KObject::Thread(extra));

        let mut out = Vec::new();
        // A count of zero writes no ids but still reports the total.
        c.x[1] = buffer;
        c.x[2] = 0;
        dispatch(c, 0x65);
        out.push(c.x[0]);
        out.push(c.x[1]);
        // Room for one id.
        c.x[1] = buffer;
        c.x[2] = 1;
        dispatch(c, 0x65);
        out.push(c.x[0]);
        out.push(c.x[1]);
        out.push(u64::from_le_bytes(
            c.process()
                .memory
                .read(buffer, 8)
                .unwrap()
                .try_into()
                .unwrap(),
        ));
        // Out-of-range counts and unmapped arrays.
        c.x[1] = buffer;
        c.x[2] = 0x8000_0000;
        dispatch(c, 0x65);
        out.push(c.x[0]);
        c.x[1] = 0xdead_0000;
        c.x[2] = 1;
        dispatch(c, 0x65);
        out.push(c.x[0]);

        // The global thread list is this process's list.
        c.x[1] = buffer;
        c.x[2] = 8;
        c.x[3] = 0;
        dispatch(c, 0x66);
        out.push(c.x[0]);
        out.push(c.x[1]);
        out.extend(words(
            &c.process().memory.read(buffer, 16).unwrap_or_default(),
        ));
        // Only a debug object or another process's handle can be non-zero.
        c.x[1] = buffer;
        c.x[2] = 8;
        c.x[3] = 1;
        dispatch(c, 0x66);
        out.push(c.x[0]);
        // Zero capacity is still a valid count-only query.
        c.x[1] = 0;
        c.x[2] = 0;
        c.x[3] = 0;
        dispatch(c, 0x66);
        out.push(c.x[0]);
        out.push(c.x[1]);

        record("lists", &out);
        c.halt();
    }

    #[test]
    fn process_and_thread_lists_report_the_running_process() {
        run_guest(process_and_thread_lists);
        let got = observed("lists");
        assert_eq!(
            &got[..4],
            &[RESULT_SUCCESS, 1, RESULT_SUCCESS, 1],
            "one process; the count does not depend on the array size"
        );
        assert_eq!(got[4], threads::PROCESS_ID);
        assert_eq!(got[5], RESULT_OUT_OF_RANGE);
        assert_eq!(got[6], RESULT_INVALID_POINTER);
        assert_eq!(&got[7..9], &[RESULT_SUCCESS, 2], "main thread plus 0x99");
        assert_eq!(&got[9..11], &[MAIN_THREAD_ID, 0x99]);
        assert_eq!(got[11], RESULT_INVALID_HANDLE);
        assert_eq!(&got[12..14], &[RESULT_SUCCESS, 2]);
    }

    extern "C" fn process_memory_svcs(c: &mut Cpu) {
        const CODE: u64 = 0x3_0000_0000;
        const HEAP: u64 = 0x2_0000_0000;
        let p = c.process();
        p.memory
            .map_as(CODE, PAGE, state::CODE_STATIC, perm::RX)
            .expect("the code page maps");
        p.memory
            .map_as(HEAP, PAGE, state::HEAP, perm::RW)
            .expect("the heap page maps");

        let mut out = Vec::new();
        // 0x73: address, size, permission, handle.
        for (addr, size, perm_) in [
            (CODE + 1, PAGE, perm::RW as u64),
            (CODE, 0, perm::RW as u64),
            (CODE, PAGE, 7),
            (CODE, PAGE, 2),
        ] {
            c.x[0] = CURRENT_PROCESS;
            c.x[1] = addr;
            c.x[2] = size;
            c.x[3] = perm_;
            dispatch(c, 0x73);
            out.push(c.x[0]);
        }
        for handle in [0xdead, CURRENT_PROCESS] {
            c.x[0] = handle;
            c.x[1] = HEAP;
            c.x[2] = PAGE;
            c.x[3] = perm::RX as u64;
            dispatch(c, 0x73);
            out.push(c.x[0]);
        }
        // CodeStatic becomes CodeMutable when written, then stays mutable when
        // made executable again.
        c.x[0] = CURRENT_PROCESS;
        c.x[1] = CODE;
        c.x[2] = PAGE;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x73);
        out.push(c.x[0]);
        out.push(p.memory.query(CODE).state as u64);
        out.push(p.memory.query(CODE).perm as u64);
        c.x[0] = CURRENT_PROCESS;
        c.x[1] = CODE;
        c.x[2] = PAGE;
        c.x[3] = perm::RX as u64;
        dispatch(c, 0x73);
        out.push(c.x[0]);
        out.push(p.memory.query(CODE).perm as u64);

        // 0x74/0x75 validate the range before rejecting the handle.
        for imm in [0x74u16, 0x75] {
            c.x[0] = CODE + 1;
            c.x[1] = CURRENT_PROCESS;
            c.x[2] = CODE;
            c.x[3] = PAGE;
            dispatch(c, imm);
            out.push(c.x[0]);
            c.x[0] = CODE;
            c.x[1] = CURRENT_PROCESS;
            c.x[2] = CODE;
            c.x[3] = 0;
            dispatch(c, imm);
            out.push(c.x[0]);
            c.x[0] = CODE;
            c.x[1] = CURRENT_PROCESS;
            c.x[2] = CODE;
            c.x[3] = PAGE;
            dispatch(c, imm);
            out.push(c.x[0]);
        }

        // 0x76: the current process's memory is the caller's memory.
        let info = c.sp - 0x400;
        c.x[0] = info;
        c.x[2] = CURRENT_PROCESS;
        c.x[3] = CODE;
        dispatch(c, 0x76);
        out.push(c.x[0]);
        out.push(c.x[1]);
        let bytes = p.memory.read(info, 0x28).unwrap();
        out.push(u64::from_le_bytes(bytes[0..8].try_into().unwrap()));
        out.push(u64::from_le_bytes(bytes[8..16].try_into().unwrap()));
        out.push(u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as u64);
        c.x[0] = info;
        c.x[2] = 0xdead;
        c.x[3] = CODE;
        dispatch(c, 0x76);
        out.push(c.x[0]);

        // 0x77/0x78: the current-process alias is not accepted at all.
        for imm in [0x77u16, 0x78] {
            c.x[0] = CURRENT_PROCESS;
            c.x[1] = CODE + 1;
            c.x[2] = HEAP;
            c.x[3] = PAGE;
            dispatch(c, imm);
            out.push(c.x[0]);
            c.x[0] = CURRENT_PROCESS;
            c.x[1] = CODE;
            c.x[2] = HEAP;
            c.x[3] = PAGE;
            dispatch(c, imm);
            out.push(c.x[0]);
        }

        // 0x79: pointer validation, then the single-process refusal.
        let params = c.sp - 0x500;
        assert!(p.memory.write(params, &[0; 0x30]));
        c.x[1] = 0xdead_0000;
        c.x[2] = 0;
        c.x[3] = 0;
        dispatch(c, 0x79);
        out.push(c.x[0]);
        c.x[1] = params;
        c.x[3] = u32::MAX as u64;
        dispatch(c, 0x79);
        out.push(c.x[0]);
        c.x[1] = params;
        c.x[3] = 0;
        dispatch(c, 0x79);
        out.push(c.x[0]);

        // 0x7a: the only process is already running.
        c.x[0] = CURRENT_PROCESS;
        dispatch(c, 0x7a);
        out.push(c.x[0]);
        c.x[0] = 0xdead;
        dispatch(c, 0x7a);
        out.push(c.x[0]);
        // 0x7b: another process cannot exist; terminating self is a separate test.
        c.x[0] = 0xdead;
        dispatch(c, 0x7b);
        out.push(c.x[0]);

        // 0x7f: the SMC result space says NotImplemented, arguments untouched.
        for (i, r) in c.x.iter_mut().take(8).enumerate() {
            *r = 0x100 + i as u64;
        }
        dispatch(c, 0x7f);
        out.extend(c.x[..8].iter().copied());

        record("process_memory", &out);
        c.halt();
    }

    #[test]
    fn process_memory_svcs_use_the_documented_errors() {
        run_guest(process_memory_svcs);
        let got = observed("process_memory");
        assert_eq!(
            &got[..9],
            &[
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_MEMORY_STATE,
                RESULT_SUCCESS,
                0x04,
                perm::RW as u64,
            ],
            "0x73 validates its arguments and only code states are writable"
        );
        assert_eq!(
            &got[9..11],
            &[RESULT_SUCCESS, perm::RX as u64],
            "CodeMutable stays mutable when it is made executable again"
        );
        assert_eq!(
            &got[11..17],
            &[
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_HANDLE,
            ],
            "0x74/0x75 validate before refusing the pseudo process handle"
        );
        assert_eq!(
            &got[17..22],
            &[
                RESULT_SUCCESS,
                0,
                0x3_0000_0000,
                PAGE,
                state::CODE_MUTABLE as u64
            ],
            "svcQueryProcessMemory reports the page tag"
        );
        assert_eq!(got[22], RESULT_INVALID_HANDLE);
        assert_eq!(
            &got[23..27],
            &[
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_HANDLE,
            ]
        );
        assert_eq!(
            &got[27..30],
            &[
                RESULT_INVALID_POINTER,
                RESULT_INVALID_POINTER,
                RESULT_NOT_IMPLEMENTED
            ]
        );
        assert_eq!(&got[30..32], &[RESULT_INVALID_STATE, RESULT_INVALID_HANDLE]);
        assert_eq!(got[32], RESULT_INVALID_HANDLE);
        assert_eq!(
            &got[33..41],
            &[1, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107]
        );
    }

    extern "C" fn debug_context_round_trip(c: &mut Cpu) {
        let out = c.sp - 0x500;
        c.tpidr_el0 = 0xfeed;
        let mut got = Vec::new();

        // The caller is a valid debug target through the current-process handle.
        c.x[0] = out;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = c.thread.id;
        c.x[3] = ThreadContext::FLAGS_ALL as u64;
        dispatch(c, 0x67);
        got.push(c.x[0]);
        let original = c
            .process()
            .memory
            .read(out, ThreadContext::SIZE as u64)
            .unwrap();
        got.push(u64::from_le_bytes(original[8..16].try_into().unwrap()));

        // Rewrite x1 through the GPR group.
        let mut patched = original.clone();
        patched[8..16].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        assert!(c.process().memory.write(out, &patched));
        c.x[0] = CURRENT_PROCESS;
        c.x[1] = c.thread.id;
        c.x[2] = out;
        c.x[3] = ThreadContext::FLAG_GPRS as u64;
        dispatch(c, 0x68);
        got.push(c.x[0]);
        got.push(c.x[1]);
        let sp = c.sp;

        // A control-only read refreshes the control group and leaves x1 alone.
        assert!(c.process().memory.write(out, &patched));
        c.x[0] = out;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = c.thread.id;
        c.x[3] = ThreadContext::FLAG_CONTROL as u64;
        dispatch(c, 0x67);
        got.push(c.x[0]);
        let control = c
            .process()
            .memory
            .read(out, ThreadContext::SIZE as u64)
            .unwrap();
        got.push(u64::from_le_bytes(control[0xf8..0x100].try_into().unwrap()));
        got.push(u64::from_le_bytes(
            control[0x318..0x320].try_into().unwrap(),
        ));
        got.push(u64::from_le_bytes(control[8..16].try_into().unwrap()));

        // Error paths: unknown flags, handle and thread, and an unmapped output.
        c.x[0] = out;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = c.thread.id;
        c.x[3] = 0x10;
        dispatch(c, 0x67);
        got.push(c.x[0]);
        c.x[0] = 0xdead;
        c.x[1] = c.thread.id;
        c.x[2] = out;
        c.x[3] = ThreadContext::FLAG_GPRS as u64;
        dispatch(c, 0x68);
        got.push(c.x[0]);
        c.x[0] = CURRENT_PROCESS;
        c.x[1] = 0x7777;
        c.x[2] = out;
        c.x[3] = ThreadContext::FLAG_GPRS as u64;
        dispatch(c, 0x68);
        got.push(c.x[0]);
        c.x[0] = 0xdead_0000;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = c.thread.id;
        c.x[3] = ThreadContext::FLAGS_ALL as u64;
        dispatch(c, 0x67);
        got.push(c.x[0]);

        got.push(sp);
        record("debug_context", &got);
        c.halt();
    }

    #[test]
    fn debug_thread_context_round_trips_on_the_calling_thread() {
        run_guest(debug_context_round_trip);
        let got = observed("debug_context");
        assert_eq!(got[0], RESULT_SUCCESS);
        assert_eq!(got[1], CURRENT_PROCESS, "the live x1 was reported");
        assert_eq!(&got[2..4], &[RESULT_SUCCESS, 0x1234_5678], "Set applied x1");
        assert_eq!(got[4], RESULT_SUCCESS);
        assert_eq!(got[5], got[12], "the control read reported the live sp");
        assert_eq!(got[6], 0xfeed, "TPIDR_EL0 round-trips");
        assert_eq!(got[7], 0x1234_5678, "an unselected group is left alone");
        assert_eq!(
            &got[8..12],
            &[
                RESULT_INVALID_ENUM,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_ID,
                RESULT_INVALID_POINTER,
            ]
        );
    }

    extern "C" fn debug_misc(c: &mut Cpu) {
        let mut out = Vec::new();
        // The six-argument debug family with no KDebug object.
        for imm in [0x60u16, 0x61, 0x62, 0x64, 0x6b] {
            c.x[0] = 0;
            dispatch(c, imm);
            out.push(c.x[0]);
        }
        for imm in [0x63u16, 0x69, 0x6a] {
            c.x[0] = 0;
            dispatch(c, imm);
            out.push(c.x[0]);
        }

        // 0x6C records breakpoint 0 and watchpoint 0x10, rejecting bad names
        // and flag bits.
        let mut bp = Vec::new();
        for (name, flags, value) in [
            (0u64, 0u64, 0x1000u64),
            (6, 0, 0x2000),
            (0x10, 0, 0x3000),
            (0x14, 0, 0x4000),
            (0, 0x2, 0x5000),
        ] {
            c.x[0] = name;
            c.x[1] = flags;
            c.x[2] = value;
            dispatch(c, 0x6c);
            bp.push(c.x[0]);
        }
        bp.push(
            crate::kernel::hardware_breakpoint(0)
                .map(|b| b.flags)
                .unwrap_or_default(),
        );
        bp.push(
            crate::kernel::hardware_breakpoint(6)
                .map(|b| b.flags)
                .unwrap_or_default(),
        );
        record("debug_breakpoints", &bp);

        // 0x6D: priority, state, cores and affinity of the caller, then a
        // paused thread, an unknown thread and an invalid parameter.
        let mut params = Vec::new();
        for param in 0..=5u64 {
            c.x[1] = 0;
            c.x[2] = CURRENT_PROCESS;
            c.x[3] = c.thread.id;
            c.x[4] = param;
            dispatch(c, 0x6d);
            params.push(c.x[0]);
            params.push(c.x[1]);
            params.push(c.x[2]);
        }
        let paused = parked_thread(0x9b);
        paused.paused.store(true, Ordering::Relaxed);
        *paused.context.lock().unwrap() = ThreadContext {
            sp: 0xaaaa,
            ..Default::default()
        };
        let running = parked_thread(0x9c);
        let exited = parked_thread(0x9d);
        exited.exited.store(true, Ordering::Relaxed);
        {
            let mut handles = c.process().handles.lock().unwrap();
            handles.insert(KObject::Thread(paused));
            handles.insert(KObject::Thread(running));
            handles.insert(KObject::Thread(exited));
        }
        let out_ctx = c.sp - 0x500;
        c.x[1] = 0;
        c.x[2] = CURRENT_PROCESS;
        c.x[3] = 0x9b;
        c.x[4] = 1;
        dispatch(c, 0x6d);
        params.push(c.x[0]);
        params.push(c.x[1]);
        params.push(c.x[2]);
        c.x[0] = out_ctx;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = 0x9b;
        c.x[3] = ThreadContext::FLAGS_ALL as u64;
        dispatch(c, 0x67);
        params.push(c.x[0]);
        let saved = c
            .process()
            .memory
            .read(out_ctx, ThreadContext::SIZE as u64)
            .unwrap();
        params.push(u64::from_le_bytes(saved[0xf8..0x100].try_into().unwrap()));
        c.x[0] = CURRENT_PROCESS;
        c.x[1] = 0x9b;
        c.x[2] = out_ctx;
        c.x[3] = ThreadContext::FLAG_GPRS as u64;
        dispatch(c, 0x68);
        params.push(c.x[0]);
        c.x[0] = out_ctx;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = 0x9c;
        c.x[3] = ThreadContext::FLAGS_ALL as u64;
        dispatch(c, 0x67);
        params.push(c.x[0]);
        c.x[0] = out_ctx;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = 0x9d;
        c.x[3] = ThreadContext::FLAGS_ALL as u64;
        dispatch(c, 0x67);
        params.push(c.x[0]);
        c.x[1] = 0;
        c.x[2] = CURRENT_PROCESS;
        c.x[3] = 0x9d;
        c.x[4] = 1;
        dispatch(c, 0x6d);
        params.push(c.x[0]);
        params.push(c.x[1]);
        params.push(c.x[2]);
        c.x[0] = CURRENT_PROCESS;
        c.x[1] = 0x9d;
        c.x[2] = out_ctx;
        c.x[3] = ThreadContext::FLAG_GPRS as u64;
        dispatch(c, 0x68);
        params.push(c.x[0]);
        c.x[1] = 0;
        c.x[2] = CURRENT_PROCESS;
        c.x[3] = 0x7777;
        c.x[4] = 1;
        dispatch(c, 0x6d);
        params.push(c.x[0]);
        c.x[2] = 0xdead;
        dispatch(c, 0x6d);
        params.push(c.x[0]);
        record("debug_params", &params);
        c.halt();
    }

    #[test]
    fn debug_svcs_report_the_documented_errors_and_params() {
        run_guest(debug_misc);
        assert_eq!(
            observed("debug_breakpoints"),
            vec![
                RESULT_SUCCESS,
                RESULT_INVALID_ENUM,
                RESULT_SUCCESS,
                RESULT_INVALID_ENUM,
                RESULT_INVALID_ENUM,
                0x4004,
                0x104004,
            ],
            "the kernel's forced security bits are recorded, never armed"
        );
        let got = observed("debug_params");
        assert_eq!(
            &got[..3],
            &[RESULT_SUCCESS, 0, MAIN_THREAD_PRIORITY],
            "priority is reported in W2"
        );
        assert_eq!(&got[3..6], &[RESULT_SUCCESS, 1, 0], "the caller is running");
        assert_eq!(&got[6..9], &[RESULT_SUCCESS, 0, 0], "ideal core");
        assert_eq!(&got[9..12], &[RESULT_SUCCESS, 0, 0], "current core");
        assert_eq!(&got[12..15], &[RESULT_SUCCESS, 0, 1], "affinity mask");
        assert_eq!(got[15], RESULT_INVALID_ENUM);
        assert_eq!(&got[18..21], &[RESULT_SUCCESS, 5, 2], "user-suspended");
        assert_eq!(got[21], RESULT_SUCCESS);
        assert_eq!(got[22], 0xaaaa, "the paused thread's saved context is read");
        assert_eq!(got[23], RESULT_BUSY, "a stopped thread is not rewritten");
        assert_eq!(got[24], RESULT_BUSY, "a running thread is not inspected");
        assert_eq!(got[25], RESULT_TERMINATION_REQUESTED);
        assert_eq!(
            &got[26..29],
            &[RESULT_SUCCESS, 4, 0],
            "the exited thread reports the terminating state"
        );
        assert_eq!(got[29], RESULT_TERMINATION_REQUESTED);
        assert_eq!(got[30], RESULT_INVALID_THREAD_ID);
        assert_eq!(got[31], RESULT_INVALID_HANDLE);
    }

    extern "C" fn insecure_memory(c: &mut Cpu) {
        let base = c.process().layout.aslr_base + 0x1000_0000;
        let mut out = Vec::new();
        c.x[0] = base;
        c.x[1] = PAGE + 1;
        dispatch(c, 0x90);
        out.push(c.x[0]);
        c.x[0] = base + 1;
        c.x[1] = PAGE;
        dispatch(c, 0x90);
        out.push(c.x[0]);
        c.x[0] = 0;
        c.x[1] = PAGE;
        dispatch(c, 0x90);
        out.push(c.x[0]);
        c.x[0] = base + 0x2000_0000;
        c.x[1] = TOTAL_MEMORY;
        dispatch(c, 0x90);
        out.push(c.x[0]);

        c.x[0] = base;
        c.x[1] = PAGE;
        dispatch(c, 0x90);
        out.push(c.x[0]);
        out.push(c.process().memory.query(base).state as u64);
        out.push(c.process().memory.query(base).perm as u64);
        out.push(u64::from_le_bytes(
            c.process()
                .memory
                .read(base, 8)
                .unwrap()
                .try_into()
                .unwrap(),
        ));
        c.x[0] = base;
        c.x[1] = PAGE;
        dispatch(c, 0x91);
        out.push(c.x[0]);
        out.push(c.process().memory.query(base).state as u64);
        c.x[0] = base;
        c.x[1] = PAGE;
        dispatch(c, 0x91);
        out.push(c.x[0]);
        record("insecure", &out);
        c.halt();
    }

    #[test]
    fn insecure_physical_memory_allocates_zeroed_insecure_pages() {
        run_guest(insecure_memory);
        assert_eq!(
            observed("insecure"),
            vec![
                RESULT_INVALID_SIZE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_MEMORY_REGION,
                RESULT_LIMIT_REACHED,
                RESULT_SUCCESS,
                state::INSECURE as u64,
                perm::RW as u64,
                0,
                RESULT_SUCCESS,
                state::FREE as u64,
                RESULT_INVALID_MEMORY_STATE,
            ]
        );
    }

    extern "C" fn terminate_self(c: &mut Cpu) {
        c.x[0] = CURRENT_PROCESS;
        dispatch(c, 0x7b);
        record(
            "terminate_self",
            &[
                c.halted as u64,
                c.process().sched.exiting.load(Ordering::Relaxed) as u64,
            ],
        );
        c.halt();
    }

    #[test]
    fn terminate_process_exits_the_caller() {
        run_guest(terminate_self);
        assert_eq!(
            observed("terminate_self"),
            vec![1, 1],
            "svcTerminateProcess on the current process exits and halts it"
        );
    }

    const IO_VA: u64 = 0x3_0000_0000;
    const DEV_VA: u64 = 0x0040_0000;
    const DEV_VA_B: u64 = 0x0040_1000;
    const DEV_VA_C: u64 = 0x0040_2000;

    /// Gives the guest a 2 MiB heap so the device tests have real guest pages.
    fn map_io_heap(c: &mut Cpu) -> u64 {
        c.x[1] = 0x20_0000;
        dispatch(c, 0x01);
        c.x[1]
    }

    extern "C" fn io_pool_lifecycle(c: &mut Cpu) {
        let p = c.process();
        let mut out = Vec::new();
        c.x[1] = 1;
        dispatch(c, 0x39);
        out.push(c.x[0]);
        c.x[1] = 0;
        dispatch(c, 0x39);
        let pool = c.x[1] as u32;
        out.push(c.x[0]);
        out.push(pool as u64);
        c.x[1] = 0;
        dispatch(c, 0x39);
        out.push(c.x[0]);
        record("io_pool", &out);

        let mut errors = Vec::new();
        for (handle, physical, size, mapping, permission) in [
            (0xdead_u32, IO_POOL_BASE, PAGE, io_mapping::MEMORY, perm::RW),
            (pool, IO_POOL_BASE, PAGE, 3, perm::RW),
            (
                pool,
                IO_POOL_BASE - PAGE,
                PAGE,
                io_mapping::MEMORY,
                perm::RW,
            ),
            (pool, IO_POOL_BASE, PAGE + 1, io_mapping::MEMORY, perm::RW),
            (pool, IO_POOL_BASE + 1, PAGE, io_mapping::MEMORY, perm::RW),
            (
                pool,
                IO_POOL_END - PAGE,
                PAGE * 2,
                io_mapping::MEMORY,
                perm::RW,
            ),
            (pool, IO_POOL_BASE, PAGE, io_mapping::IO_REGISTER, perm::RW),
            (pool, IO_POOL_BASE, PAGE, io_mapping::MEMORY, perm::RX),
        ] {
            c.x[1] = handle as u64;
            c.x[2] = physical;
            c.x[3] = size;
            c.x[4] = mapping as u64;
            c.x[5] = permission as u64;
            dispatch(c, 0x3a);
            errors.push(c.x[0]);
        }
        record("io_region_errors", &errors);

        let mut regions = Vec::new();
        for (physical, size) in [(IO_POOL_BASE, PAGE), (IO_POOL_BASE + PAGE, PAGE * 2)] {
            c.x[1] = pool as u64;
            c.x[2] = physical;
            c.x[3] = size;
            c.x[4] = io_mapping::MEMORY as u64;
            c.x[5] = perm::RW as u64;
            dispatch(c, 0x3a);
            regions.push((c.x[0], c.x[1] as u32));
        }
        // A range that crosses both live claims is refused.
        c.x[1] = pool as u64;
        c.x[2] = IO_POOL_BASE;
        c.x[3] = PAGE * 3;
        c.x[4] = io_mapping::MEMORY as u64;
        c.x[5] = perm::RW as u64;
        dispatch(c, 0x3a);
        let overlap = c.x[0];
        let claimed = p
            .handles
            .lock()
            .unwrap()
            .io_pool(pool)
            .expect("the pool handle is live")
            .claimed();
        record("io_claims", &[claimed]);
        // Closing a region releases its claim.
        c.x[0] = regions[0].1 as u64;
        dispatch(c, 0x16);
        let claimed = p
            .handles
            .lock()
            .unwrap()
            .io_pool(pool)
            .expect("the pool handle is live")
            .claimed();
        record("io_claims_closed", &[claimed]);
        // Closing the pool handle while a region lives keeps the aperture busy:
        // a second pool would be backed twice, so it is refused.
        c.x[0] = pool as u64;
        dispatch(c, 0x16);
        c.x[1] = 0;
        dispatch(c, 0x39);
        let busy = c.x[0];
        c.x[0] = regions[1].1 as u64;
        dispatch(c, 0x16);
        c.x[1] = 0;
        dispatch(c, 0x39);
        record("io_recreate", &[overlap, busy, c.x[0], c.x[1]]);
        c.halt();
    }

    #[test]
    fn io_pools_claim_regions_and_are_destroyed_by_closing_the_handles() {
        run_guest(io_pool_lifecycle);
        let pool = observed("io_pool");
        assert_eq!(pool[0], RESULT_INVALID_ENUM, "only IoPoolType_PcieA2");
        assert_eq!(pool[1], RESULT_SUCCESS);
        assert_ne!(pool[2], 0, "the pool handle is returned in X1");
        assert_eq!(
            pool[3], RESULT_INVALID_STATE,
            "one aperture per process until its regions are gone"
        );
        assert_eq!(
            observed("io_region_errors"),
            vec![
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_ENUM,
                RESULT_INVALID_MEMORY_REGION,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_MEMORY_REGION,
                RESULT_INVALID_MEMORY_REGION,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
            ]
        );
        assert_eq!(observed("io_claims"), vec![PAGE * 3]);
        assert_eq!(observed("io_claims_closed"), vec![PAGE * 2]);
        let recreate = observed("io_recreate");
        assert_eq!(
            recreate[0], RESULT_INVALID_MEMORY_STATE,
            "overlapping claim"
        );
        assert_eq!(
            recreate[1], RESULT_INVALID_STATE,
            "the aperture is still held"
        );
        assert_eq!(recreate[2], RESULT_SUCCESS, "the region released it");
        assert_ne!(recreate[3], 0);
    }

    extern "C" fn io_region_memory(c: &mut Cpu) {
        let p = c.process();
        c.x[1] = 0;
        dispatch(c, 0x39);
        let pool = c.x[1] as u32;
        c.x[1] = pool as u64;
        c.x[2] = IO_POOL_BASE;
        c.x[3] = PAGE * 2;
        c.x[4] = io_mapping::MEMORY as u64;
        c.x[5] = perm::RW as u64;
        dispatch(c, 0x3a);
        let region = c.x[1] as u32;
        record("io_region_create", &[c.x[0], region as u64]);

        let mut errors = Vec::new();
        for (addr, size, permission) in [
            (IO_VA, PAGE, perm::RW),
            (IO_VA + 1, PAGE * 2, perm::RW),
            (IO_VA, PAGE * 2, perm::R),
        ] {
            c.x[0] = region as u64;
            c.x[1] = addr;
            c.x[2] = size;
            c.x[3] = permission as u64;
            dispatch(c, 0x46);
            errors.push(c.x[0]);
        }
        record("io_map_errors", &errors);

        c.x[0] = region as u64;
        c.x[1] = IO_VA;
        c.x[2] = PAGE * 2;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x46);
        let mapped = c.x[0];
        let state_at = p.memory.query(IO_VA);
        let zeros = p.memory.read(IO_VA, 4).unwrap_or_default();
        assert!(p.memory.write(IO_VA, b"pool"));
        assert!(p.memory.write(IO_VA + 4, b"page"));
        let object = p
            .handles
            .lock()
            .unwrap()
            .io_region(region)
            .expect("the region handle is live");
        let device_read = object.read(&p.memory, 0, 8).unwrap_or_default();
        let wrote = object.write(&p.memory, 4, b"DEV!");
        let guest_sees = p.memory.read(IO_VA + 4, 4).unwrap_or_default();
        record(
            "io_region_device",
            &[
                mapped,
                state_at.state as u64,
                state_at.perm as u64,
                eight(&zeros),
                eight(&device_read),
                wrote as u64,
                eight(&guest_sees),
            ],
        );

        c.x[0] = region as u64;
        c.x[1] = IO_VA;
        c.x[2] = PAGE * 2;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x46);
        let twice = c.x[0];
        c.x[0] = region as u64;
        c.x[1] = IO_VA;
        c.x[2] = PAGE;
        dispatch(c, 0x47);
        let bad_size = c.x[0];
        c.x[0] = region as u64;
        c.x[1] = IO_VA + PAGE;
        c.x[2] = PAGE * 2;
        dispatch(c, 0x47);
        let bad_address = c.x[0];
        c.x[0] = region as u64;
        c.x[1] = IO_VA;
        c.x[2] = PAGE * 2;
        dispatch(c, 0x47);
        let unmapped = c.x[0];
        let gone = p.memory.is_mapped(IO_VA, 1);
        record(
            "io_region_unmap",
            &[twice, bad_size, bad_address, unmapped, gone as u64],
        );

        // The contents survive to the next mapping.
        c.x[0] = region as u64;
        c.x[1] = IO_VA + 0x4000;
        c.x[2] = PAGE * 2;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x46);
        let remapped = c.x[0];
        let kept = p.memory.read(IO_VA + 0x4000, 8).unwrap_or_default();
        // Closing the handle while mapped releases the guest pages.
        c.x[0] = region as u64;
        dispatch(c, 0x16);
        let released = !p.memory.is_mapped(IO_VA + 0x4000, 1);
        record(
            "io_region_persist",
            &[remapped, eight(&kept), released as u64],
        );
        c.halt();
    }

    #[test]
    fn an_io_region_maps_real_memory_and_survives_unmapping() {
        run_guest(io_region_memory);
        assert_eq!(
            observed("io_region_create")[0],
            RESULT_SUCCESS,
            "a memory region inside the pool is backed"
        );
        assert_eq!(
            observed("io_map_errors"),
            vec![
                RESULT_INVALID_SIZE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
            ]
        );
        assert_eq!(
            observed("io_region_device"),
            vec![
                RESULT_SUCCESS,
                1, // MemType_Io
                perm::RW as u64,
                0, // zero-filled on the first mapping
                eight(b"poolpage"),
                1, // the device wrote through the guest mapping
                eight(b"DEV!"),
            ]
        );
        assert_eq!(
            observed("io_region_unmap"),
            vec![
                RESULT_INVALID_MEMORY_STATE, // a second mapping
                RESULT_INVALID_SIZE,
                RESULT_INVALID_MEMORY_STATE, // the mapping is not at that address
                RESULT_SUCCESS,
                0, // the guest pages are free again
            ]
        );
        assert_eq!(
            observed("io_region_persist"),
            vec![RESULT_SUCCESS, eight(b"poolDEV!"), 1]
        );
    }

    extern "C" fn device_address_space(c: &mut Cpu) {
        let p = c.process();
        let heap = map_io_heap(c);
        assert!(p.memory.write(heap, b"device page"));
        assert!(p.memory.write(heap + PAGE, b"copy me"));

        let mut create_errors = Vec::new();
        for (base, size) in [
            (1u64, PAGE),
            (0, 0),
            (0, PAGE + 1),
            (0, DEVICE_SPACE_LIMIT + PAGE),
        ] {
            c.x[1] = base;
            c.x[2] = size;
            dispatch(c, 0x56);
            create_errors.push(c.x[0]);
        }
        record("das_create_errors", &create_errors);
        c.x[1] = 0;
        c.x[2] = DEVICE_SPACE_LIMIT;
        dispatch(c, 0x56);
        let das = c.x[1] as u32;
        record("das_create", &[c.x[0], das as u64]);

        let mut attach = Vec::new();
        c.x[0] = 8;
        c.x[1] = das as u64;
        dispatch(c, 0x57);
        attach.push(c.x[0]);
        c.x[0] = 8;
        c.x[1] = das as u64;
        dispatch(c, 0x57);
        attach.push(c.x[0]);
        c.x[0] = 0x26;
        c.x[1] = das as u64;
        dispatch(c, 0x57);
        attach.push(c.x[0]);
        c.x[0] = 29;
        c.x[1] = das as u64;
        dispatch(c, 0x58);
        attach.push(c.x[0]);
        c.x[0] = 8;
        c.x[1] = 0xdead;
        dispatch(c, 0x58);
        attach.push(c.x[0]);
        c.x[0] = 8;
        c.x[1] = das as u64;
        dispatch(c, 0x58);
        attach.push(c.x[0]);
        c.x[0] = 8;
        c.x[1] = das as u64;
        dispatch(c, 0x58);
        attach.push(c.x[0]);
        record("das_attach", &attach);

        let mut errors = Vec::new();
        for (imm, handle, process, guest, size, device, option) in [
            // The option word's high bits ask for an IO mapping indicator.
            (
                0x59u16,
                das as u64,
                CURRENT_PROCESS,
                heap,
                PAGE,
                DEV_VA,
                (perm::RW as u64) | 0x1_0000,
            ),
            (
                0x59,
                das as u64,
                0xdead,
                heap,
                PAGE,
                DEV_VA,
                perm::RW as u64,
            ),
            (
                0x59,
                das as u64,
                CURRENT_PROCESS,
                heap + 1,
                PAGE,
                DEV_VA,
                perm::RW as u64,
            ),
            (
                0x59,
                das as u64,
                CURRENT_PROCESS,
                heap,
                PAGE,
                DEV_VA + 1,
                perm::RW as u64,
            ),
            (
                0x59,
                das as u64,
                CURRENT_PROCESS,
                heap,
                0,
                DEV_VA,
                perm::RW as u64,
            ),
            (
                0x59,
                das as u64,
                CURRENT_PROCESS,
                IO_VA,
                PAGE,
                DEV_VA,
                perm::RW as u64,
            ),
            (
                0x59,
                das as u64,
                CURRENT_PROCESS,
                heap,
                PAGE,
                DEV_VA,
                perm::RX as u64,
            ),
            // The aligned variant wants matching 4 MiB frames; the heap and
            // 0x2000 do not share one.
            (
                0x5a,
                das as u64,
                CURRENT_PROCESS,
                heap,
                PAGE,
                DEV_VA_C,
                perm::RW as u64,
            ),
        ] {
            c.x[0] = handle;
            c.x[1] = process;
            c.x[2] = guest;
            c.x[3] = size;
            c.x[4] = device;
            c.x[5] = option;
            dispatch(c, imm);
            errors.push(c.x[0]);
        }
        // 0x5B reads its arguments from a different register layout.
        c.x[0] = 0;
        c.x[1] = 0xdead;
        c.x[2] = CURRENT_PROCESS;
        c.x[3] = heap;
        c.x[4] = PAGE;
        c.x[5] = DEV_VA;
        c.x[6] = perm::RW as u64;
        dispatch(c, 0x5b);
        errors.push(c.x[0]);
        record("das_map_errors", &errors);

        // By force: no alignment requirement.
        c.x[0] = das as u64;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = heap;
        c.x[3] = PAGE;
        c.x[4] = DEV_VA;
        c.x[5] = perm::RW as u64;
        dispatch(c, 0x59);
        let forced = c.x[0];
        // Aligned: both frames are 0x1000.
        c.x[0] = das as u64;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = heap + PAGE;
        c.x[3] = PAGE;
        c.x[4] = DEV_VA_B;
        c.x[5] = perm::R as u64;
        dispatch(c, 0x5a);
        let aligned = c.x[0];
        // 0x5B maps too and reports the size.
        c.x[1] = das as u64;
        c.x[2] = CURRENT_PROCESS;
        c.x[3] = heap + PAGE * 2;
        c.x[4] = PAGE;
        c.x[5] = DEV_VA_C;
        c.x[6] = perm::RW as u64;
        dispatch(c, 0x5b);
        let sized = (c.x[0], c.x[1]);
        record("das_map", &[forced, aligned, sized.0, sized.1]);

        let object = p
            .handles
            .lock()
            .unwrap()
            .device_address_space(das)
            .expect("the device address space handle is live");
        let read = object.read(&p.memory, DEV_VA, 11).unwrap_or_default();
        let readonly_write = object.write(&p.memory, DEV_VA_B, b"x");
        let copied = object.copy(&p.memory, DEV_VA_C, DEV_VA_B, PAGE);
        let dst = p.memory.read(heap + PAGE * 2, 7).unwrap_or_default();
        let mapped_bytes = object.mapped_bytes();
        record(
            "das_device",
            &[
                eight(&read),
                readonly_write as u64,
                copied as u64,
                eight(&dst),
                mapped_bytes,
            ],
        );

        let mut unmap = Vec::new();
        c.x[0] = das as u64;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = heap + PAGE;
        c.x[3] = PAGE;
        c.x[4] = DEV_VA_B;
        dispatch(c, 0x5c);
        unmap.push(c.x[0]);
        unmap.push(
            p.handles
                .lock()
                .unwrap()
                .device_address_space(das)
                .expect("live")
                .mapped_bytes(),
        );
        c.x[0] = das as u64;
        c.x[1] = CURRENT_PROCESS;
        c.x[2] = heap + PAGE;
        c.x[3] = PAGE;
        c.x[4] = DEV_VA_B;
        dispatch(c, 0x5c);
        unmap.push(c.x[0]);
        c.x[0] = das as u64;
        c.x[1] = 0xdead;
        dispatch(c, 0x5c);
        unmap.push(c.x[0]);
        record("das_unmap", &unmap);
        c.halt();
    }

    #[test]
    fn a_device_address_space_aliases_guest_pages_for_a_synthetic_device() {
        run_guest(device_address_space);
        assert_eq!(
            observed("das_create_errors"),
            vec![
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_MEMORY_REGION,
            ]
        );
        let create = observed("das_create");
        assert_eq!(create[0], RESULT_SUCCESS);
        assert_ne!(create[1], 0);
        assert_eq!(
            observed("das_attach"),
            vec![
                RESULT_SUCCESS,
                RESULT_INVALID_STATE,
                RESULT_INVALID_ENUM,
                RESULT_INVALID_STATE,
                RESULT_INVALID_HANDLE,
                RESULT_SUCCESS,
                RESULT_INVALID_STATE,
            ]
        );
        assert_eq!(
            observed("das_map_errors"),
            vec![
                RESULT_INVALID_ENUM,
                RESULT_INVALID_HANDLE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_MEMORY_REGION,
                RESULT_INVALID_NEW_MEMORY_PERMISSION,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_HANDLE,
            ]
        );
        assert_eq!(
            observed("das_map"),
            vec![RESULT_SUCCESS, RESULT_SUCCESS, RESULT_SUCCESS, PAGE],
            "0x5B reports the mapped size in X1"
        );
        assert_eq!(
            observed("das_device"),
            vec![
                eight(b"device page"),
                0, // the read-only mapping refuses a device write
                1, // the synthetic device copied through the device window
                eight(b"copy me"),
                PAGE * 3,
            ]
        );
        assert_eq!(
            observed("das_unmap"),
            vec![
                RESULT_SUCCESS,
                PAGE * 2,
                RESULT_INVALID_MEMORY_STATE,
                RESULT_INVALID_HANDLE
            ]
        );
    }

    extern "C" fn query_memory_mapping(c: &mut Cpu) {
        let p = c.process();
        let heap = map_io_heap(c);
        c.x[1] = 0;
        dispatch(c, 0x39);
        let pool = c.x[1] as u32;
        c.x[1] = pool as u64;
        c.x[2] = IO_POOL_BASE;
        c.x[3] = PAGE;
        c.x[4] = io_mapping::MEMORY as u64;
        c.x[5] = perm::RW as u64;
        dispatch(c, 0x3a);
        let region = c.x[1] as u32;
        let out_addr = heap + 0x1000;
        let out_size = heap + 0x2000;
        let word = |p: &Process, at: u64| {
            p.memory
                .read(at, 8)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap_or([0; 8])))
                .unwrap_or(0)
        };
        let mut out = Vec::new();
        // Guest RAM: physical == virtual in this runtime.
        c.x[1] = out_addr;
        c.x[2] = out_size;
        c.x[3] = heap + PAGE;
        c.x[4] = PAGE;
        dispatch(c, 0x55);
        out.push(c.x[0]);
        out.push(word(p, out_addr));
        out.push(word(p, out_size));
        // The aperture is not backed before a region is mapped there.
        c.x[1] = out_addr;
        c.x[2] = out_size;
        c.x[3] = IO_POOL_BASE;
        c.x[4] = PAGE;
        dispatch(c, 0x55);
        out.push(c.x[0]);
        // Map the region: the pool range now resolves to its guest view.
        c.x[0] = region as u64;
        c.x[1] = IO_VA;
        c.x[2] = PAGE;
        c.x[3] = perm::RW as u64;
        dispatch(c, 0x46);
        c.x[1] = out_addr;
        c.x[2] = out_size;
        c.x[3] = IO_POOL_BASE;
        c.x[4] = PAGE;
        dispatch(c, 0x55);
        out.push(c.x[0]);
        out.push(word(p, out_addr));
        // Validation and not-found paths.
        for (physical, size) in [
            (heap, 0),
            (heap + 1, PAGE),
            (u64::MAX - PAGE, PAGE * 2),
            (0x3_4000_0000, PAGE),
        ] {
            c.x[1] = out_addr;
            c.x[2] = out_size;
            c.x[3] = physical;
            c.x[4] = size;
            dispatch(c, 0x55);
            out.push(c.x[0]);
        }
        // Output pointers the guest cannot write.
        c.x[1] = 0x3_4000_0000;
        c.x[2] = out_size;
        c.x[3] = heap;
        c.x[4] = PAGE;
        dispatch(c, 0x55);
        out.push(c.x[0]);
        c.x[1] = 0;
        dispatch(c, 0x55);
        out.push(c.x[0]);
        record("query_mapping", &out);
        c.halt();
    }

    #[test]
    fn query_memory_mapping_reports_the_guest_view_of_a_physical_range() {
        run_guest(query_memory_mapping);
        let heap = HEAP_BASE;
        assert_eq!(
            observed("query_mapping"),
            vec![
                RESULT_SUCCESS,
                heap,
                PAGE,
                RESULT_NOT_FOUND,
                RESULT_SUCCESS,
                IO_VA,
                RESULT_INVALID_SIZE,
                RESULT_NOT_FOUND, // unaligned physical address
                RESULT_NOT_FOUND, // overflow
                RESULT_NOT_FOUND, // unmapped RAM
                RESULT_INVALID_POINTER,
                RESULT_INVALID_POINTER,
            ]
        );
    }

    extern "C" fn unsafe_physical(c: &mut Cpu) {
        let p = c.process();
        let at = ALIAS_BASE + 0x1000_0000;
        let mut out = Vec::new();
        // The allowance starts at zero, as on hardware.
        c.x[0] = at;
        c.x[1] = PAGE * 2;
        dispatch(c, 0x48);
        out.push(c.x[0]);
        out.push(p.memory.is_mapped(at, PAGE) as u64);
        c.x[0] = PAGE * 2;
        dispatch(c, 0x4a);
        out.push(c.x[0]);
        c.x[0] = at;
        c.x[1] = PAGE * 2;
        dispatch(c, 0x48);
        out.push(c.x[0]);
        out.push(p.memory.query(at).state as u64);
        // One page more than the allowance.
        c.x[0] = at + PAGE * 2;
        c.x[1] = PAGE;
        dispatch(c, 0x48);
        out.push(c.x[0]);
        // The documented validation.
        c.x[0] = at;
        c.x[1] = PAGE + 1;
        dispatch(c, 0x48);
        out.push(c.x[0]);
        c.x[0] = at + 1;
        c.x[1] = PAGE;
        dispatch(c, 0x48);
        out.push(c.x[0]);
        c.x[0] = 0x1000;
        c.x[1] = PAGE;
        dispatch(c, 0x48);
        out.push(c.x[0]);
        // Unmap, remap a page, and refuse to unmap a page that is not heap.
        c.x[0] = at;
        c.x[1] = PAGE * 2;
        dispatch(c, 0x49);
        out.push(c.x[0]);
        out.push(p.memory.query(at).state as u64);
        c.x[0] = at;
        c.x[1] = PAGE;
        dispatch(c, 0x48);
        out.push(c.x[0]);
        // A page mapped with another state is not the unsafe family's to unmap.
        assert!(
            p.memory
                .map_as(at + PAGE, PAGE, state::NORMAL, perm::RW)
                .is_ok()
        );
        c.x[0] = at + PAGE;
        c.x[1] = PAGE;
        dispatch(c, 0x49);
        out.push(c.x[0]);
        record("unsafe", &out);
        c.halt();
    }

    #[test]
    fn unsafe_physical_memory_needs_the_limit_and_only_backs_guest_ram() {
        run_guest(unsafe_physical);
        assert_eq!(
            observed("unsafe"),
            vec![
                RESULT_LIMIT_REACHED,
                0,              // nothing was mapped
                RESULT_SUCCESS, // svcSetUnsafeLimit
                RESULT_SUCCESS,
                state::HEAP as u64,
                RESULT_LIMIT_REACHED,
                RESULT_INVALID_SIZE,
                RESULT_INVALID_ADDRESS,
                RESULT_INVALID_MEMORY_REGION, // outside the alias region
                RESULT_SUCCESS,
                state::FREE as u64,
                RESULT_SUCCESS,
                RESULT_INVALID_MEMORY_STATE, // that page is not heap
            ]
        );
    }
}
