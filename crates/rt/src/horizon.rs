use std::io::Write;
use std::sync::Arc;

use crate::arbiter;
use crate::cpu::Cpu;
use crate::kernel::{KObject, Session};
use crate::threads;
use crate::mem::{attr, perm, state, PhysicalError, PAGE};
use crate::services::sm::Sm;

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

pub const MAIN_THREAD_HANDLE: u64 = 0x0001_0001;
pub const CURRENT_PROCESS: u64 = 0xffff_8001;

fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KIRIKAE_TRACE_SVC").is_some())
}

pub fn svc(c: &mut Cpu, imm: u16) {
    let _prof = crate::prof::enter_svc(imm);
    crate::usage::svc(imm);
    crate::sched::kernel_enter(c);
    dispatch(c, imm);
    crate::sched::kernel_exit();
    crate::sched::reschedule(c);
    if c.process().sched.exiting.load(std::sync::atomic::Ordering::Relaxed) {
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
        0x03 => set_memory_attribute(c),
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
        0x15 => {
            let (addr, size) = (c.x[1], c.x[2]);
            let h = c.process().handles.lock().unwrap().insert(KObject::TransferMemory { addr, size });
            ok1(c, h as u64);
        }
        0x13 | 0x14 => {
            let (h, addr, perm) = (c.x[0] as u32, c.x[1], c.x[3] as u32);
            let p = c.process();
            let obj = p.handles.lock().unwrap().get(h);
            let done = match obj {
                Some(KObject::SharedMemory(s)) if imm == 0x13 => s.map(&p.memory, addr, perm & 3),
                Some(KObject::SharedMemory(s)) => s.unmap(&p.memory, addr),
                _ => {
                    c.x[0] = RESULT_INVALID_HANDLE;
                    return;
                }
            };
            c.x[0] = if done { RESULT_SUCCESS } else { RESULT_INVALID_MEMORY_STATE };
        }
        0x16 => close_handle(c),
        0x18 => threads::wait_synchronization(c),
        0x19 => threads::cancel_synchronization(c),
        0x1a => arbiter::arbitrate_lock(c),
        0x1b => arbiter::arbitrate_unlock(c),
        0x1c => arbiter::wait_process_wide_key(c),
        0x1d => arbiter::signal_process_wide_key(c),
        0x24 => threads::get_process_id(c),
        0x25 => threads::get_thread_id(c),
        0x32 => threads::set_thread_activity(c),
        0x33 => threads::get_thread_context3(c),
        0x34 => arbiter::wait_for_address(c),
        0x35 => arbiter::signal_to_address(c),
        0x45 => threads::create_event(c),
        0x1f => connect_to_named_port(c),
        0x21 => c.x[0] = crate::ipc::send_sync_request(c, c.x[0] as u32, c.tpidrro_el0, 0x100),
        0x22 => c.x[0] = crate::ipc::send_sync_request(c, c.x[2] as u32, c.x[0], c.x[1]),
        0x1e => c.x[0] = c.process().ticks(),
        0x26 => break_(c),
        0x27 => output_debug_string(c),
        0x29 => get_info(c),
        0x2c => map_physical_memory(c),
        0x2d => unmap_physical_memory(c),
        _ => {
            c.process().warn_once(format!("svc {imm:#04x} is not implemented (first call from {})", c.process().describe(c.pc - 4)));
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

fn connect_to_named_port(c: &mut Cpu) {
    let p = c.process();
    let raw = p.memory.read(c.x[1], 12).unwrap_or_default();
    let name = String::from_utf8_lossy(raw.split(|&b| b == 0).next().unwrap_or(&[])).into_owned();
    p.log_ipc(format!("ConnectToNamedPort({name:?})"));
    if name != "sm:" {
        p.warn_once(format!("svcConnectToNamedPort({name:?}): unknown port"));
        c.x[0] = RESULT_NOT_FOUND;
        return;
    }
    let session = Session::new(Arc::new(Sm));
    let handle = p.handles.lock().unwrap().insert(KObject::Session(session));
    ok1(c, handle as u64);
}

pub const RESULT_TIMED_OUT: u64 = 0xea01;
pub const RESULT_INVALID_HANDLE: u64 = 0xe401;
pub const RESULT_CANCELLED: u64 = 0xec01;
pub const RESULT_INVALID_PRIORITY: u64 = 0xe001;
pub const RESULT_INVALID_CORE: u64 = 0xe201;
pub const RESULT_INVALID_COMBINATION: u64 = 0xe801;
pub const RESULT_INVALID_STATE: u64 = 0xfa01;

fn event_signal(c: &mut Cpu, signaled: bool) {
    c.x[0] = if c.process().set_event(c.x[0] as u32, signaled) { RESULT_SUCCESS } else { RESULT_INVALID_HANDLE };
}

fn close_handle(c: &mut Cpu) {
    let handle = c.x[0] as u32;
    let closed = c.process().handles.lock().unwrap().close(handle);
    if !closed && handle < 0xffff_8000 {
        c.process().log_ipc(format!("CloseHandle({handle:#x}) of an unknown handle"));
    }
    ok(c);
}

fn set_heap_size(c: &mut Cpu) {
    let size = c.x[1];
    if !size.is_multiple_of(HEAP_ALIGN) || size > HEAP_REGION_SIZE {
        c.x[0] = RESULT_INVALID_SIZE;
        return;
    }
    let p = c.process();
    let mut heap = p.heap.lock().unwrap();
    if size > *heap {
        if p.memory.map_as(HEAP_BASE + *heap, size - *heap, state::HEAP, perm::RW).is_err() {
            drop(heap);
            fault(c, format!("could not map a {size:#x}-byte heap"));
            return;
        }
        *heap = size;
    } else if size < *heap {
        if let Err(e) = p.memory.unmap(HEAP_BASE + size, *heap - size) {
            drop(heap);
            fault(c, format!("could not shrink the heap to {size:#x}: {e}"));
            return;
        }
        *heap = size;
    }
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = HEAP_BASE;
}

pub const RESULT_INVALID_MEMORY_STATE: u64 = 0xd401;

fn map_memory(c: &mut Cpu) {
    let (dst, src, size) = (c.x[0], c.x[1], c.x[2]);
    let m = &c.process().memory;
    if m.map_as(dst, size, state::STACK, perm::RW).is_err() || !m.copy(dst, src, size) {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    }
    ok(c);
}

fn unmap_memory(c: &mut Cpu) {
    let (dst, src, size) = (c.x[0], c.x[1], c.x[2]);
    let m = &c.process().memory;
    if !m.copy(src, dst, size) || m.unmap(dst, size).is_err() {
        c.x[0] = RESULT_INVALID_MEMORY_STATE;
        return;
    }
    ok(c);
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
    } else if c.process().memory.protect(addr, addr + size, new, &[state::HEAP, state::CODE_MUTABLE]) {
        RESULT_SUCCESS
    } else {
        RESULT_INVALID_MEMORY_STATE
    };
}

fn set_memory_attribute(c: &mut Cpu) {
    let (addr, size, mask, value) = (c.x[0], c.x[1], c.x[2] as u32, c.x[3] as u32);
    c.x[0] = if !addr.is_multiple_of(PAGE) {
        RESULT_INVALID_ADDRESS
    } else if size == 0 || !size.is_multiple_of(PAGE) {
        RESULT_INVALID_SIZE
    } else if addr.checked_add(size).is_none() {
        RESULT_INVALID_MEMORY_STATE
    } else if (mask | value) & !attr::SETTABLE != 0 || value & !mask != 0 || (mask ^ value) & attr::PERMISSION_LOCKED != 0 {
        RESULT_INVALID_COMBINATION
    } else if c.process().memory.set_attribute(addr, addr + size, mask, value, &[state::HEAP, state::CODE_MUTABLE]) {
        RESULT_SUCCESS
    } else {
        RESULT_INVALID_MEMORY_STATE
    };
}

fn physical_range(c: &mut Cpu) -> Option<(u64, u64)> {
    let (addr, size) = (c.x[0], c.x[1]);
    let error = if !addr.is_multiple_of(PAGE) {
        RESULT_INVALID_ADDRESS
    } else if size == 0 || !size.is_multiple_of(PAGE) {
        RESULT_INVALID_SIZE
    } else if addr.checked_add(size).is_none() {
        RESULT_INVALID_MEMORY_REGION
    } else if c.process().system_resource_size == 0 {
        RESULT_INVALID_STATE
    } else if addr < ALIAS_BASE || addr + size > ALIAS_BASE + ALIAS_SIZE {
        RESULT_INVALID_MEMORY_REGION
    } else {
        return Some((addr, addr + size));
    };
    c.x[0] = error;
    None
}

fn physical_result(c: &mut Cpu, name: &str, r: Result<u64, PhysicalError>, update: impl FnOnce(u64)) {
    c.x[0] = match r {
        Ok(bytes) => {
            update(bytes);
            RESULT_SUCCESS
        }
        Err(PhysicalError::State) => RESULT_INVALID_MEMORY_STATE,
        Err(PhysicalError::Limit) => RESULT_LIMIT_REACHED,
        Err(PhysicalError::Host(e)) => return fault(c, format!("{name}: the host refused the mapping ({e})")),
    };
}

fn map_physical_memory(c: &mut Cpu) {
    let Some((start, end)) = physical_range(c) else { return };
    let p = c.process();
    let budget = TOTAL_MEMORY.saturating_sub(p.used_memory());
    let r = p.memory.map_physical(start, end, budget);
    physical_result(c, "svcMapPhysicalMemory", r, |n| {
        p.physical.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    });
}

fn unmap_physical_memory(c: &mut Cpu) {
    let Some((start, end)) = physical_range(c) else { return };
    let p = c.process();
    let r = p.memory.unmap_physical(start, end);
    physical_result(c, "svcUnmapPhysicalMemory", r, |n| {
        p.physical.fetch_sub(n, std::sync::atomic::Ordering::Relaxed);
    });
}

fn query_memory(c: &mut Cpu) {
    let (out, addr) = (c.x[0], c.x[2]);
    let r = c.process().memory.query(addr);
    let mut info = [0u8; 0x28];
    info[0..8].copy_from_slice(&r.range.start.to_le_bytes());
    info[8..16].copy_from_slice(&(r.range.end - r.range.start).to_le_bytes());
    info[16..20].copy_from_slice(&r.state.to_le_bytes());
    info[20..24].copy_from_slice(&r.attr.to_le_bytes());
    info[24..28].copy_from_slice(&r.perm.to_le_bytes());
    if !c.process().memory.write(out, &info) {
        fault(c, format!("svcQueryMemory output {out:#x} is not mapped"));
        return;
    }
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = 0;
}

fn get_info(c: &mut Cpu) {
    let (id, subid) = (c.x[1], c.x[3]);
    let p = c.process();
    let value = match id {
        0 => 0b1111,
        1 => 0xfc00_0000_0000_0000,
        2 => ALIAS_BASE,
        3 => ALIAS_SIZE,
        4 => HEAP_BASE,
        5 => HEAP_REGION_SIZE,
        6 => TOTAL_MEMORY,
        7 => p.used_memory(),
        8 => 0,
        9 => 0,
        10 => 0,
        11 => p.entropy(subid),
        12 => ASLR_BASE,
        13 => ASLR_SIZE,
        14 => STACK_REGION_BASE,
        15 => STACK_REGION_SIZE,
        16 => p.system_resource_size,
        17 => (p.physical.load(std::sync::atomic::Ordering::Relaxed) / 512).next_multiple_of(PAGE).min(p.system_resource_size),
        18 => p.title_id,
        20 => *p.exception_context.get_or_init(|| p.alloc_tls()),
        21 => TOTAL_MEMORY - p.system_resource_size,
        22 => p.used_memory() - p.system_resource_size,
        23 => 1,
        24 => 0x60,
        26 => 1,
        28 => 0,
        _ => {
            p.warn_once(format!("svcGetInfo id {id} (subid {subid}) is not implemented"));
            c.x[0] = RESULT_INVALID_ENUM;
            return;
        }
    };
    c.x[0] = RESULT_SUCCESS;
    c.x[1] = value;
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
    let mut fp = c.x[29];
    for _ in 0..16 {
        let (Some(next), Some(lr)) = (p.memory.read(fp, 8), p.memory.read(fp + 8, 8)) else { break };
        let (next, lr) = (u64::from_le_bytes(next.try_into().unwrap()), u64::from_le_bytes(lr.try_into().unwrap()));
        if lr < 4 || next <= fp {
            break;
        }
        let _ = writeln!(std::io::stderr(), "kirikae:   called from {}", p.describe(lr - 4));
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
        None => fault(c, format!("svcOutputDebugString read unmapped memory at {addr:#x}+{len:#x}")),
    }
}

pub fn undefined(c: &mut Cpu, addr: u64, word: u32) {
    let at = c.process().describe(addr);
    fault(c, format!("instruction {word:#010x} at {at} is not implemented"));
}

pub fn fault(c: &mut Cpu, message: String) {
    let p = c.process();
    let _ = writeln!(std::io::stderr(), "kirikae: {message}");
    let _ = writeln!(std::io::stderr(), "kirikae: thread {} lr={} sp={:#x}", c.thread.id, p.describe(c.x[30]), c.sp);
    p.terminate(2);
    c.halt();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{Entry, GuestFn};
    use crate::process::{RunOptions, run_executable};
    use std::sync::Mutex;

    static OBSERVED: Mutex<Vec<(&'static str, Vec<u64>)>> = Mutex::new(Vec::new());

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

    fn run_guest(f: GuestFn) {
        let entries: &'static [Entry] = Box::leak(
            vec![Entry {
                addr: 0x1000_0000,
                func: f,
            }]
            .into_boxed_slice(),
        );
        let exe = kirikae_loader::Executable {
            modules: Vec::new(),
            entry: entries[0].addr,
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
        assert_eq!(result.exit_code, 0);
    }

    #[derive(Clone, Copy)]
    struct Info {
        start: u64,
        size: u64,
        state: u32,
        attr: u32,
        perm: u32,
    }

    fn query_info(c: &mut Cpu, out: u64, addr: u64) -> Info {
        c.x[0] = out;
        c.x[2] = addr;
        dispatch(c, 0x06);
        assert_eq!(c.x[0], RESULT_SUCCESS, "svcQueryMemory succeeds");
        let b = c.process().memory.read(out, 0x28).expect("the MemoryInfo output is mapped");
        Info {
            start: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            size: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            state: u32::from_le_bytes(b[16..20].try_into().unwrap()),
            attr: u32::from_le_bytes(b[20..24].try_into().unwrap()),
            perm: u32::from_le_bytes(b[24..28].try_into().unwrap()),
        }
    }

    fn set_attr(c: &mut Cpu, addr: u64, size: u64, mask: u32, value: u32) -> u64 {
        c.x[0] = addr;
        c.x[1] = size;
        c.x[2] = u64::from(mask);
        c.x[3] = u64::from(value);
        dispatch(c, 0x03);
        c.x[0]
    }

    extern "C" fn shrink(c: &mut Cpu) {
        let size = |c: &mut Cpu, bytes: u64| {
            c.x[1] = bytes;
            dispatch(c, 0x01);
            c.x[0]
        };
        let grown = size(c, HEAP_ALIGN * 2);
        let mapped = c.process().memory.query(HEAP_BASE + HEAP_ALIGN).state;
        let shrunk = size(c, HEAP_ALIGN);
        let tail = c.process().memory.query(HEAP_BASE + HEAP_ALIGN).state;
        record("heap", &[grown, u64::from(mapped), shrunk, u64::from(tail)]);
        c.halt();
    }

    #[test]
    fn shrinking_the_heap_unmaps_the_tail() {
        run_guest(shrink);
        let got = observed("heap");
        assert_eq!(got[0], RESULT_SUCCESS);
        assert_eq!(got[1], u64::from(state::HEAP), "the grown page is heap");
        assert_eq!(got[2], RESULT_SUCCESS, "shrinking succeeds");
        assert_eq!(
            got[3],
            u64::from(state::FREE),
            "the returned tail is unmapped, not left as heap"
        );
    }

    extern "C" fn attribute_round_trip(c: &mut Cpu) {
        let base = HEAP_BASE;
        let out = base + 0x3000;
        c.process().memory.map_as(base, 0x4000, state::HEAP, perm::RW).unwrap();
        assert!(c.process().memory.write(base + PAGE, b"kept"));
        let set = set_attr(c, base + PAGE, PAGE, attr::UNCACHED, attr::UNCACHED);
        let mid = query_info(c, out, base + PAGE);
        let left = query_info(c, out, base);
        let right = query_info(c, out, base + 3 * PAGE);
        let kept = c.process().memory.read(base + PAGE, 4).expect("the page is still mapped");
        let clear = set_attr(c, base + PAGE, PAGE, attr::UNCACHED, 0);
        let cleared = query_info(c, out, base + PAGE);
        let merged = query_info(c, out, base);
        record(
            "attribute_round_trip",
            &[
                set,
                u64::from(mid.attr),
                u64::from(mid.state),
                u64::from(mid.perm),
                u64::from(left.attr),
                u64::from(right.attr),
                clear,
                u64::from(cleared.attr),
                merged.start,
                merged.size,
                u64::from(u32::from_le_bytes(kept.try_into().unwrap())),
            ],
        );
        c.halt();
    }

    #[test]
    fn the_uncached_attribute_round_trips_through_query_memory() {
        run_guest(attribute_round_trip);
        let got = observed("attribute_round_trip");
        assert_eq!(got[0], RESULT_SUCCESS, "setting the uncached bit succeeds");
        assert_eq!(got[1], u64::from(attr::UNCACHED), "QueryMemory reports the attribute");
        assert_eq!(got[2], u64::from(state::HEAP), "the state is untouched");
        assert_eq!(got[3], u64::from(perm::RW), "the permission is untouched");
        assert_eq!(got[4], 0, "only the requested page reports the attribute");
        assert_eq!(got[5], 0);
        assert_eq!(got[6], RESULT_SUCCESS, "clearing the uncached bit succeeds");
        assert_eq!(got[7], 0, "QueryMemory reports the cleared bit");
        assert_eq!(got[8], HEAP_BASE, "clearing merges the page back into its neighbours");
        assert_eq!(got[9], 0x4000);
        assert_eq!(got[10], u64::from(u32::from_le_bytes(*b"kept")), "the change leaves the data alone");
    }

    extern "C" fn permission_locked(c: &mut Cpu) {
        let base = HEAP_BASE;
        let out = base + 0x2000;
        c.process().memory.map_as(base, 0x3000, state::HEAP, perm::RW).unwrap();
        let set = set_attr(c, base, PAGE, attr::PERMISSION_LOCKED, attr::PERMISSION_LOCKED);
        let after_set = query_info(c, out, base);
        let clear = set_attr(c, base, PAGE, attr::PERMISSION_LOCKED, 0);
        let after_clear = query_info(c, out, base);
        let set_both = set_attr(c, base, PAGE, attr::SETTABLE, attr::SETTABLE);
        let after_both = query_info(c, out, base);
        let clear_lock = set_attr(c, base, PAGE, attr::SETTABLE, attr::UNCACHED);
        let after_refused = query_info(c, out, base);
        record(
            "attribute_permission_locked",
            &[
                set,
                u64::from(after_set.attr),
                clear,
                u64::from(after_clear.attr),
                set_both,
                u64::from(after_both.attr),
                clear_lock,
                u64::from(after_refused.attr),
            ],
        );
        c.halt();
    }

    #[test]
    fn permission_locked_can_be_set_but_not_cleared() {
        run_guest(permission_locked);
        let got = observed("attribute_permission_locked");
        assert_eq!(got[0], RESULT_SUCCESS, "setting PermissionLocked succeeds");
        assert_eq!(got[1], u64::from(attr::PERMISSION_LOCKED));
        assert_eq!(got[2], RESULT_INVALID_COMBINATION, "clearing PermissionLocked is refused");
        assert_eq!(got[3], u64::from(attr::PERMISSION_LOCKED), "the refused call changes nothing");
        assert_eq!(got[4], RESULT_SUCCESS, "setting both bits succeeds");
        assert_eq!(got[5], u64::from(attr::SETTABLE));
        assert_eq!(got[6], RESULT_INVALID_COMBINATION, "uncached cannot be set while dropping the lock");
        assert_eq!(got[7], u64::from(attr::SETTABLE), "the refused call changes nothing");
    }

    extern "C" fn bad_arguments(c: &mut Cpu) {
        let base = HEAP_BASE;
        let out = base;
        c.process().memory.map_as(base, 0x2000, state::HEAP, perm::RW).unwrap();
        let misaligned = set_attr(c, base + 1, PAGE, attr::UNCACHED, attr::UNCACHED);
        let bad_size = set_attr(c, base, 0x800, attr::UNCACHED, attr::UNCACHED);
        let zero_size = set_attr(c, base, 0, attr::UNCACHED, attr::UNCACHED);
        let not_subset = set_attr(c, base, PAGE, attr::UNCACHED, attr::UNCACHED | attr::PERMISSION_LOCKED);
        let unsupported = set_attr(c, base, PAGE, 0x1, 0x1);
        let unmapped = set_attr(c, base + 0x2000, PAGE, attr::UNCACHED, attr::UNCACHED);
        let partial = set_attr(c, base + PAGE, 2 * PAGE, attr::UNCACHED, attr::UNCACHED);
        let first = query_info(c, out, base);
        let second = query_info(c, out, base + PAGE);
        record(
            "attribute_bad_arguments",
            &[
                misaligned,
                bad_size,
                zero_size,
                not_subset,
                unsupported,
                unmapped,
                partial,
                u64::from(first.attr),
                u64::from(second.attr),
                u64::from(first.state),
                u64::from(second.perm),
            ],
        );
        c.halt();
    }

    #[test]
    fn bad_arguments_are_refused_with_the_documented_results() {
        run_guest(bad_arguments);
        let got = observed("attribute_bad_arguments");
        assert_eq!(got[0], RESULT_INVALID_ADDRESS);
        assert_eq!(got[1], RESULT_INVALID_SIZE);
        assert_eq!(got[2], RESULT_INVALID_SIZE);
        assert_eq!(got[3], RESULT_INVALID_COMBINATION, "value is not a subset of mask");
        assert_eq!(got[4], RESULT_INVALID_COMBINATION, "the locked bit cannot be set by this SVC");
        assert_eq!(got[5], RESULT_INVALID_MEMORY_STATE, "an unmapped page is refused");
        assert_eq!(got[6], RESULT_INVALID_MEMORY_STATE, "a range with an unmapped page is refused");
        assert_eq!(got[7], 0, "the mapped pages keep their attribute");
        assert_eq!(got[8], 0);
        assert_eq!(got[9], u64::from(state::HEAP));
        assert_eq!(got[10], u64::from(perm::RW));
    }
}
