use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::cpu::Cpu;
use crate::horizon::{ok, RESULT_INVALID_STATE, RESULT_TIMED_OUT};
use crate::kernel::{KObject, KThread};
use crate::sched::{self, SyncState, WaitKind};

pub const HAS_WAITERS: u32 = 0x4000_0000;

fn priority(c: &Cpu) -> i32 {
    c.thread.priority.load(Ordering::Relaxed)
}

fn unlock_mutex(c: &Cpu, st: &mut SyncState, addr: u64) {
    let queue = st.queue(WaitKind::Mutex, addr);
    match queue.first() {
        None => c.st32(addr, 0),
        Some(next) => {
            let more = if queue.len() > 1 { HAS_WAITERS } else { 0 };
            c.st32(addr, (next.tag | more) as u64);
            st.wake(next.id, 0);
        }
    }
}

fn inherit(c: &Cpu, st: &mut SyncState) {
    let handles = c.process().handles.lock().unwrap();
    let thread = |tag: u32| match handles.get(tag) {
        Some(KObject::Thread(t)) => Some(t),
        _ => None,
    };
    let edges: Vec<(Arc<KThread>, Arc<KThread>)> = st
        .waiters
        .iter()
        .filter(|w| w.kind == WaitKind::Mutex)
        .filter_map(|w| Some((thread(w.tag)?, thread(c.ld32(w.key) as u32 & !HAS_WAITERS)?)))
        .collect();
    drop(handles);
    let before: Vec<(u64, i32)> = st.boosted.iter().map(|t| (t.id, t.inherited.load(Ordering::Relaxed))).collect();
    let mut lent: HashMap<u64, i32> = HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for (waiter, owner) in &edges {
            let p = waiter.priority.load(Ordering::Relaxed).min(lent.get(&waiter.id).copied().unwrap_or(i32::MAX));
            if p < owner.priority.load(Ordering::Relaxed) && p < lent.get(&owner.id).copied().unwrap_or(i32::MAX) {
                lent.insert(owner.id, p);
                changed = true;
            }
        }
    }
    for t in st.boosted.drain(..) {
        t.inherited.store(i32::MAX, Ordering::Relaxed);
    }
    for (_, owner) in edges {
        if let Some(&p) = lent.get(&owner.id)
            && owner.inherited.swap(p, Ordering::Relaxed) == i32::MAX
        {
            st.boosted.push(owner);
        }
    }
    let after: Vec<(u64, i32)> = st.boosted.iter().map(|t| (t.id, t.inherited.load(Ordering::Relaxed))).collect();
    if before != after {
        c.process().sched.gil.changed();
    }
}

pub fn arbitrate_lock(c: &mut Cpu) {
    let (owner, addr, tag) = (c.x[0] as u32, c.x[1], c.x[2] as u32);
    let p = c.process();
    let id = {
        let mut st = p.sched.lock();
        if c.ld32(addr) as u32 != owner | HAS_WAITERS {
            drop(st);
            ok(c);
            return;
        }
        let id = st.add_waiter(WaitKind::Mutex, addr, 0, tag, priority(c));
        inherit(c, &mut st);
        id
    };
    let what = || format!("ArbitrateLock({addr:#x}, owner {owner:#x}) from {}", p.describe(c.x[30]));
    match sched::block(c, -1, what, |st| st.woken.remove(&id)) {
        Some(r) => c.x[0] = r,
        None => {
            let mut st = p.sched.lock();
            st.remove(id);
            inherit(c, &mut st);
            drop(st);
            c.halt();
        }
    }
}

pub fn arbitrate_unlock(c: &mut Cpu) {
    let addr = c.x[0];
    let p = c.process();
    let mut st = p.sched.lock();
    unlock_mutex(c, &mut st, addr);
    inherit(c, &mut st);
    p.sched.touch();
    drop(st);
    ok(c);
}

pub fn wait_process_wide_key(c: &mut Cpu) {
    let (mutex, key, tag, timeout) = (c.x[0], c.x[1], c.x[2] as u32, c.x[3] as i64);
    let p = c.process();
    let id = {
        let mut st = p.sched.lock();
        c.st32(key, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        unlock_mutex(c, &mut st, mutex);
        inherit(c, &mut st);
        p.sched.touch();
        st.add_waiter(WaitKind::CondVar, key, mutex, tag, priority(c))
    };
    let what = || format!("WaitProcessWideKeyAtomic(cv {key:#x}, mutex {mutex:#x}) from {}", p.describe(c.x[30]));
    if let Some(r) = sched::block(c, timeout, what, |st| st.woken.remove(&id)) {
        c.x[0] = r;
        return;
    }
    let still_cv = {
        let mut st = p.sched.lock();
        match st.waiters.iter().find(|w| w.id == id).map(|w| w.kind) {
            Some(WaitKind::CondVar) => {
                st.remove(id);
                true
            }
            Some(_) => false,
            None => match st.woken.remove(&id) {
                Some(r) => {
                    drop(st);
                    c.x[0] = r;
                    return;
                }
                None => true,
            },
        }
    };
    if still_cv {
        c.x[0] = RESULT_TIMED_OUT;
        return;
    }
    let what = || format!("WaitProcessWideKeyAtomic re-lock of {mutex:#x}");
    match sched::block(c, -1, what, |st| st.woken.remove(&id)) {
        Some(r) => c.x[0] = r,
        None => c.halt(),
    }
}

pub fn signal_process_wide_key(c: &mut Cpu) {
    let (key, count) = (c.x[0], c.x[1] as i32);
    let p = c.process();
    let mut st = p.sched.lock();
    let queue = st.queue(WaitKind::CondVar, key);
    let n = if count <= 0 { queue.len() } else { (count as usize).min(queue.len()) };
    for w in &queue[..n] {
        let old = c.update32(w.mutex, |word| Some(if word == 0 { w.tag } else { word | HAS_WAITERS }));
        if old == 0 {
            st.wake(w.id, 0);
        } else {
            if let Some(entry) = st.waiters.iter_mut().find(|e| e.id == w.id) {
                entry.kind = WaitKind::Mutex;
                entry.key = w.mutex;
            }
        }
    }
    if n == queue.len() {
        c.st32(key, 0);
    }
    inherit(c, &mut st);
    p.sched.touch();
    drop(st);
    ok(c);
}

pub fn wait_for_address(c: &mut Cpu) {
    let (addr, kind, value, timeout) = (c.x[0], c.x[1] as u32, c.x[2] as i32, c.x[3] as i64);
    let p = c.process();
    let id = {
        let mut st = p.sched.lock();
        let cur = c.ld32(addr) as i32;
        let wait = match kind {
            0 => cur < value,
            1 => {
                let old = c.update32(addr, |cur| ((cur as i32) < value).then(|| (cur as i32).wrapping_sub(1) as u32));
                (old as i32) < value
            }
            2 => cur == value,
            3 => c.ld64(addr) == c.x[2],
            _ => false,
        };
        if !wait {
            drop(st);
            c.x[0] = RESULT_INVALID_STATE;
            return;
        }
        st.add_waiter(WaitKind::Address, addr, 0, c.thread.handle, priority(c))
    };
    let what = || format!("WaitForAddress({addr:#x}) from {}", p.describe(c.x[30]));
    match sched::block(c, timeout, what, |st| st.woken.remove(&id)) {
        Some(r) => c.x[0] = r,
        None => {
            let mut st = p.sched.lock();
            st.remove(id);
            c.x[0] = st.woken.remove(&id).unwrap_or(RESULT_TIMED_OUT);
        }
    }
}

pub fn signal_to_address(c: &mut Cpu) {
    let (addr, kind, value, count) = (c.x[0], c.x[1] as u32, c.x[2] as i32, c.x[3] as i32);
    let p = c.process();
    let mut st = p.sched.lock();
    let queue = st.queue(WaitKind::Address, addr);
    let n = if count <= 0 { queue.len() } else { (count as usize).min(queue.len()) };
    if kind != 0 {
        let new = match kind {
            1 => value.wrapping_add(1),
            _ if queue.is_empty() => value.wrapping_add(1),
            _ if count <= 0 => value.wrapping_sub(2),
            _ if (queue.len() as i32) < count => value.wrapping_sub(1),
            _ => value,
        };
        if c.update32(addr, |cur| (cur as i32 == value).then_some(new as u32)) as i32 != value {
            drop(st);
            c.x[0] = RESULT_INVALID_STATE;
            return;
        }
    }
    for w in &queue[..n] {
        st.wake(w.id, 0);
    }
    p.sched.touch();
    drop(st);
    ok(c);
}
