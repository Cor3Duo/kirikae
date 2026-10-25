use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::time::Duration;

use crate::cpu::Cpu;
use crate::horizon::{
    MAIN_THREAD_ID, RESULT_BUSY, RESULT_CANCELLED, RESULT_INVALID_COMBINATION, RESULT_INVALID_CORE,
    RESULT_INVALID_ENUM, RESULT_INVALID_HANDLE, RESULT_INVALID_POINTER, RESULT_INVALID_PRIORITY,
    RESULT_INVALID_STATE, RESULT_TIMED_OUT, ok, ok1,
};
use crate::kernel::{Event, KObject, KThread, PSEUDO_CURRENT_PROCESS, PSEUDO_CURRENT_THREAD};
use crate::sched;

pub const PROCESS_ID: u64 = 0x81;
const DEFAULT_CORE: i32 = 0;
const CORE_MASK: u64 = 0b1111;

static NEXT_THREAD_ID: AtomicU64 = AtomicU64::new(MAIN_THREAD_ID + 1);

pub fn thread_of(c: &Cpu, handle: u32) -> Option<Arc<KThread>> {
    if handle == PSEUDO_CURRENT_THREAD {
        return Some(c.thread.clone());
    }
    match c.process().handles.lock().unwrap().get(handle) {
        Some(KObject::Thread(t)) => Some(t),
        _ => None,
    }
}

pub fn create_thread(c: &mut Cpu) {
    let (entry, arg, stack_top) = (c.x[1], c.x[2], c.x[3]);
    let (priority, core) = (c.x[4] as i32, c.x[5] as i32);
    if !(0..=63).contains(&priority) {
        c.x[0] = RESULT_INVALID_PRIORITY;
        return;
    }
    let core = match core {
        -2 => DEFAULT_CORE,
        0..=3 => core,
        _ => {
            c.x[0] = RESULT_INVALID_CORE;
            return;
        }
    };
    let p = c.process();
    let tls = p.alloc_tls();
    let mut handles = p.handles.lock().unwrap();
    let handle = handles.reserve();
    let thread = Arc::new(KThread {
        id: NEXT_THREAD_ID.fetch_add(1, Ordering::Relaxed),
        handle,
        priority: AtomicI32::new(priority),
        core: AtomicI32::new(core),
        ideal_core: AtomicI32::new(core),
        affinity: AtomicU64::new(1 << core),
        inherited: AtomicI32::new(i32::MAX),
        entry,
        arg,
        stack_top,
        tls,
        started: AtomicBool::new(false),
        exited: AtomicBool::new(false),
        cancelled: AtomicBool::new(false),
        last_pc: Default::default(),
        last_lr: Default::default(),
        paused: AtomicBool::new(false),
        context: Default::default(),
    });
    handles.insert_at(handle, KObject::Thread(thread));
    drop(handles);
    p.log_ipc(format!(
        "CreateThread(entry={}, stack={stack_top:#x}, prio={priority}, core={core}) -> {handle:#x}",
        p.describe(entry)
    ));
    ok1(c, handle as u64);
}

pub fn start_thread(c: &mut Cpu) {
    let Some(t) = thread_of(c, c.x[0] as u32) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    if t.started.swap(true, Ordering::Relaxed) {
        c.x[0] = RESULT_INVALID_STATE;
        return;
    }
    let (entry, arg, sp, handle) = (t.entry, t.arg, t.stack_top, t.handle);
    let started = sched::spawn(c.process(), t, move |c| {
        c.x[0] = arg;
        c.x[1] = handle as u64;
        c.sp = sp;
        c.pc = entry;
    });
    match started {
        Ok(_) => ok(c),
        Err(e) => crate::horizon::fatal(c, format!("could not start a host thread: {e}")),
    }
}

pub fn sleep_thread(c: &mut Cpu) {
    let ns = c.x[0] as i64;
    sched::sleep(c, Duration::from_nanos(ns.max(0) as u64));
}

pub fn get_priority(c: &mut Cpu) {
    match thread_of(c, c.x[1] as u32) {
        Some(t) => ok1(c, t.priority.load(Ordering::Relaxed) as u64),
        None => c.x[0] = RESULT_INVALID_HANDLE,
    }
}

pub fn set_priority(c: &mut Cpu) {
    let prio = c.x[1] as i32;
    match thread_of(c, c.x[0] as u32) {
        Some(_) if !(0..=63).contains(&prio) => c.x[0] = RESULT_INVALID_PRIORITY,
        Some(t) => {
            t.priority.store(prio, Ordering::Relaxed);
            c.process().sched.gil.changed();
            ok(c);
        }
        None => c.x[0] = RESULT_INVALID_HANDLE,
    }
}

pub fn get_core_mask(c: &mut Cpu) {
    match thread_of(c, c.x[2] as u32) {
        Some(t) => {
            ok1(c, t.ideal_core.load(Ordering::Relaxed) as u32 as u64);
            c.x[2] = t.affinity.load(Ordering::Relaxed);
        }
        None => c.x[0] = RESULT_INVALID_HANDLE,
    }
}

pub fn set_core_mask(c: &mut Cpu) {
    let (core, mask) = (c.x[1] as i32, c.x[2]);
    let Some(t) = thread_of(c, c.x[0] as u32) else {
        c.x[0] = RESULT_INVALID_HANDLE;
        return;
    };
    let (core, mask) = match core {
        -2 => (DEFAULT_CORE, 1 << DEFAULT_CORE),
        _ if mask == 0 || mask & !CORE_MASK != 0 => {
            c.x[0] = if mask == 0 {
                RESULT_INVALID_COMBINATION
            } else {
                RESULT_INVALID_CORE
            };
            return;
        }
        -3 => (t.ideal_core.load(Ordering::Relaxed), mask),
        -1..=3 => (core, mask),
        _ => {
            c.x[0] = RESULT_INVALID_CORE;
            return;
        }
    };
    if core >= 0 && mask & (1 << core) == 0 {
        c.x[0] = RESULT_INVALID_COMBINATION;
        return;
    }
    t.ideal_core.store(core, Ordering::Relaxed);
    t.affinity.store(mask, Ordering::Relaxed);
    if mask & (1 << t.core.load(Ordering::Relaxed)) == 0 {
        let to = if core >= 0 {
            core
        } else {
            63 - mask.leading_zeros() as i32
        };
        t.core.store(to, Ordering::Relaxed);
    }
    c.process().sched.gil.changed();
    c.process().log_ipc(format!(
        "SetThreadCoreMask(thread {}, core {core}, mask {mask:#x})",
        t.id
    ));
    ok(c);
}

pub fn current_processor(c: &mut Cpu) {
    c.x[0] = c.thread.core.load(Ordering::Relaxed) as u64;
}

pub fn get_thread_id(c: &mut Cpu) {
    match thread_of(c, c.x[1] as u32) {
        Some(t) => ok1(c, t.id),
        None => c.x[0] = RESULT_INVALID_HANDLE,
    }
}

pub fn get_process_id(c: &mut Cpu) {
    let h = c.x[1] as u32;
    if h == PSEUDO_CURRENT_PROCESS || thread_of(c, h).is_some() {
        ok1(c, PROCESS_ID);
    } else {
        c.x[0] = RESULT_INVALID_HANDLE;
    }
}

pub fn cancel_synchronization(c: &mut Cpu) {
    match thread_of(c, c.x[0] as u32) {
        Some(t) => {
            let s = &c.process().sched;
            let _st = s.lock();
            t.cancelled.store(true, Ordering::Relaxed);
            s.notify();
            drop(_st);
            ok(c);
        }
        None => c.x[0] = RESULT_INVALID_HANDLE,
    }
}

fn other_thread(c: &mut Cpu, handle: u32) -> Option<Arc<KThread>> {
    let t = (handle != PSEUDO_CURRENT_THREAD)
        .then(|| thread_of(c, handle))
        .flatten();
    match t {
        Some(t) if Arc::ptr_eq(&t, &c.thread) => c.x[0] = RESULT_BUSY,
        Some(t) => return Some(t),
        None if handle == PSEUDO_CURRENT_THREAD => c.x[0] = RESULT_BUSY,
        None => c.x[0] = RESULT_INVALID_HANDLE,
    }
    None
}

pub fn set_thread_activity(c: &mut Cpu) {
    let pause = match c.x[1] {
        0 => false,
        1 => true,
        _ => {
            c.x[0] = RESULT_INVALID_ENUM;
            return;
        }
    };
    let Some(t) = other_thread(c, c.x[0] as u32) else {
        return;
    };
    let s = &c.process().sched;
    let st = s.lock();
    if t.exited.load(Ordering::Relaxed) || t.paused.load(Ordering::Relaxed) == pause {
        c.x[0] = RESULT_INVALID_STATE;
        return;
    }
    t.paused.store(pause, Ordering::Relaxed);
    s.notify();
    drop(st);
    if pause {
        sched::wait_stopped(c, &t);
    }
    ok(c);
}

pub fn get_thread_context3(c: &mut Cpu) {
    let out = c.x[0];
    let Some(t) = other_thread(c, c.x[1] as u32) else {
        return;
    };
    if !t.paused.load(Ordering::Relaxed) {
        c.x[0] = RESULT_INVALID_STATE;
        return;
    }
    sched::wait_stopped(c, &t);
    let bytes = t.context.lock().unwrap().to_bytes();
    if c.process().memory.write(out, &bytes) {
        ok(c);
    } else {
        c.x[0] = RESULT_INVALID_POINTER;
    }
}

pub fn create_event(c: &mut Cpu) {
    let event = Arc::new(Event {
        signaled: AtomicBool::new(false),
    });
    let mut handles = c.process().handles.lock().unwrap();
    let w = handles.insert(KObject::Event(event.clone()));
    let r = handles.insert(KObject::Event(event));
    drop(handles);
    ok1(c, w as u64);
    c.x[2] = r as u64;
}

pub fn wait_synchronization(c: &mut Cpu) {
    let poll = c.x[3] == 0;
    wait_for_objects(c);
    if poll && c.x[0] != crate::horizon::RESULT_SUCCESS {
        sched::yield_now(c);
    }
}

fn wait_for_objects(c: &mut Cpu) {
    let p = c.process();
    let (list, count, timeout) = (c.x[1], c.x[2].min(64), c.x[3] as i64);
    let raw = p.memory.read(list, count * 4).unwrap_or_default();
    let handles: Vec<u32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let mut objects = Vec::new();
    for h in &handles {
        let obj = if *h == PSEUDO_CURRENT_THREAD {
            Some(KObject::Thread(c.thread.clone()))
        } else {
            p.handles.lock().unwrap().get(*h)
        };
        match obj {
            Some(o) => objects.push(o),
            None => {
                p.warn_once(format!(
                    "WaitSynchronization on unknown handle {h:#x} from {}",
                    p.describe(c.x[30])
                ));
                c.x[0] = RESULT_INVALID_HANDLE;
                return;
            }
        }
    }
    let me = c.thread.clone();
    let what = || {
        format!(
            "WaitSynchronization({handles:x?}) from {}",
            p.describe(c.x[30])
        )
    };
    let result = sched::block(c, timeout, what, |_| {
        if me.cancelled.swap(false, Ordering::Relaxed) {
            return Some(Err(RESULT_CANCELLED));
        }
        let signaled = objects.iter().position(|o| match o {
            KObject::Event(e) => e.signaled.load(Ordering::Relaxed),
            KObject::Thread(t) => t.exited.load(Ordering::Relaxed),
            // A port is signalled while a connection waits to be accepted.
            KObject::Port(p) => p.pending() > 0,
            // A client port is signalled while the server port can accept a
            // connection.
            KObject::ClientPort(p) => p.can_connect(),
            // A server session is signalled by a request or a closed peer.
            KObject::Session(s) => s
                .guest_server_link()
                .is_some_and(|l| l.has_request() || !l.client_open()),
            KObject::TransferMemory(_)
            | KObject::CodeMemory(_)
            | KObject::SharedMemory(_)
            | KObject::ResourceLimit(_) => false,
        });
        if signaled.is_some() {
            return signaled.map(Ok);
        }
        // A port that was closed cannot produce another connection, so a
        // server waiting on it is released instead of parking forever. This is
        // the `0xf601` of a session whose peer is gone, applied to the port.
        if objects
            .iter()
            .any(|o| matches!(o, KObject::Port(p) if p.closed()))
        {
            return Some(Err(crate::horizon::RESULT_REMOTE_DEAD));
        }
        None
    });
    match result {
        Some(Ok(i)) => ok1(c, i as u64),
        Some(Err(r)) => c.x[0] = r,
        None => c.x[0] = RESULT_TIMED_OUT,
    }
}
