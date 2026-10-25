use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::cpu::{Cpu, HALT_PC};
use crate::kernel::KThread;
use crate::process::{Process, run_until};

pub const POLL_INTERVAL: u32 = 2048;
const HOST_STACK: usize = 256 << 20;
const CORES: usize = 4;
const MIGRATION_MIN_PRIORITY: i32 = 2;

struct Ready {
    thread: Arc<KThread>,
    seq: u64,
    asked: Option<u64>,
    turn: Arc<Condvar>,
    running: bool,
}

impl Ready {
    fn core(&self) -> usize {
        self.thread.core.load(Ordering::Relaxed) as usize
    }

    fn priority(&self) -> i32 {
        self.thread.effective_priority()
    }

    fn key(&self) -> (i32, u64) {
        (self.priority(), self.seq)
    }
}

#[derive(Default)]
struct GilState {
    holder: Option<u64>,
    ready: Vec<Ready>,
    counter: u64,
    freed: Option<(Instant, bool)>,
    suggested: Vec<usize>,
    tops: Vec<usize>,
}

impl GilState {
    fn note_free(&mut self) {
        if crate::prof::on() {
            let waiting = self.ready.iter().any(|r| r.asked.is_some());
            self.freed = Some((Instant::now(), waiting));
        }
    }

    fn note_taken(&mut self) {
        if let Some((at, waiting)) = self.freed.take() {
            crate::prof::lock_free(at.elapsed(), waiting);
        }
    }

    fn tick(&mut self) -> u64 {
        self.counter += 1;
        self.counter
    }

    fn schedule(&mut self) -> [Option<usize>; CORES] {
        let mut top: [Option<usize>; CORES] = [None; CORES];
        for (i, r) in self.ready.iter().enumerate() {
            let c = r.core();
            if top[c].is_none_or(|t| r.key() < self.ready[t].key()) {
                top[c] = Some(i);
            }
        }
        for core in 0..CORES {
            if top[core].is_some() {
                continue;
            }
            let allowed = |r: &Ready| {
                r.core() != core && r.thread.affinity.load(Ordering::Relaxed) & (1 << core) != 0
            };
            let mut suggested = std::mem::take(&mut self.suggested);
            let mut tops = std::mem::take(&mut self.tops);
            suggested.clear();
            tops.clear();
            suggested.extend((0..self.ready.len()).filter(|&i| allowed(&self.ready[i])));
            suggested.sort_by_key(|&i| self.ready[i].key());
            let mut pick = None;
            let mut stuck = false;
            for &i in &suggested {
                let from = self.ready[i].core();
                match top[from] {
                    Some(t) if t == i => tops.push(from),
                    t => {
                        stuck =
                            t.is_some_and(|t| self.ready[t].priority() < MIGRATION_MIN_PRIORITY);
                        pick = (!stuck).then_some(i);
                        break;
                    }
                }
            }
            if pick.is_none() && !stuck {
                for &from in &tops {
                    let t = top[from].unwrap();
                    let next = (0..self.ready.len())
                        .filter(|&i| i != t && self.ready[i].core() == from)
                        .min_by_key(|&i| self.ready[i].key());
                    if next.is_some() {
                        top[from] = next;
                        pick = Some(t);
                        break;
                    }
                }
            }
            self.suggested = suggested;
            self.tops = tops;
            if let Some(i) = pick {
                let seq = self.tick();
                let r = &mut self.ready[i];
                r.thread.core.store(core as i32, Ordering::Relaxed);
                r.seq = seq;
                top[core] = Some(i);
            }
        }
        top
    }

    fn next(&mut self) -> Option<usize> {
        let top = self.schedule();
        top.into_iter()
            .flatten()
            .filter(|&i| self.ready[i].asked.is_some())
            .min_by_key(|&i| self.ready[i].asked)
    }

    fn wake_next(&mut self) -> Option<u64> {
        let i = self.next()?;
        self.ready[i].turn.notify_one();
        Some(self.ready[i].thread.id)
    }

    fn core_busy(&self, core: usize, id: u64) -> bool {
        self.ready
            .iter()
            .any(|r| r.running && r.thread.id != id && r.core() == core)
    }

    fn may_run(&mut self, id: u64) -> bool {
        let top = self.schedule();
        let Some(i) = self.ready.iter().position(|r| r.thread.id == id) else {
            return false;
        };
        let core = self.ready[i].core();
        top[core] == Some(i) && !self.core_busy(core, id)
    }

    fn wake_runnable(&mut self) {
        let top = self.schedule();
        for (core, t) in top.into_iter().enumerate() {
            let Some(i) = t else { continue };
            let r = &self.ready[i];
            if r.asked.is_some() && !self.core_busy(core, r.thread.id) {
                r.turn.notify_one();
            }
        }
    }

    fn lost_core(&mut self, id: u64) -> bool {
        let top = self.schedule();
        !top.iter().flatten().any(|&i| self.ready[i].thread.id == id)
    }
}

pub fn parallel() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::env_var!("KIRIKAE_PARALLEL").is_none_or(|v| v != "0"))
}

pub struct Gil {
    state: Mutex<GilState>,
    parallel: bool,
    epoch: AtomicU64,
}

impl Default for Gil {
    fn default() -> Gil {
        Gil {
            state: Mutex::default(),
            parallel: parallel(),
            epoch: AtomicU64::new(0),
        }
    }
}

impl Gil {
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    fn publish(&self, st: &GilState) {
        self.epoch.store(st.counter, Ordering::Release);
    }

    pub fn acquire(&self, me: &Arc<KThread>) {
        let _prof = crate::prof::enter(crate::prof::Cat::LockWait);
        let mut st = self.state.lock().unwrap();
        let seq = st.tick();
        st.ready.push(Ready {
            thread: me.clone(),
            seq,
            asked: None,
            turn: TURN.with(|t| t.clone()),
            running: false,
        });
        drop(self.queue(st, me.id));
    }

    fn queue<'a>(&'a self, st: MutexGuard<'a, GilState>, id: u64) -> MutexGuard<'a, GilState> {
        if self.parallel {
            return self.queue_parallel(st, id);
        }
        self.queue_global(st, id)
    }

    fn queue_parallel<'a>(
        &'a self,
        mut st: MutexGuard<'a, GilState>,
        id: u64,
    ) -> MutexGuard<'a, GilState> {
        let asked = st.tick();
        let r = st
            .ready
            .iter_mut()
            .find(|r| r.thread.id == id)
            .expect("a queued thread is runnable");
        r.asked = Some(asked);
        r.running = false;
        let turn = r.turn.clone();
        st.wake_runnable();
        self.publish(&st);
        while !st.may_run(id) {
            st = turn.wait(st).unwrap();
        }
        st.tick();
        if let Some(r) = st.ready.iter_mut().find(|r| r.thread.id == id) {
            r.asked = None;
            r.running = true;
        }
        self.publish(&st);
        st
    }

    fn queue_global<'a>(
        &'a self,
        mut st: MutexGuard<'a, GilState>,
        id: u64,
    ) -> MutexGuard<'a, GilState> {
        let asked = st.tick();
        let r = st
            .ready
            .iter_mut()
            .find(|r| r.thread.id == id)
            .expect("a queued thread is runnable");
        r.asked = Some(asked);
        let turn = r.turn.clone();
        while st.holder.is_some() || st.wake_next() != Some(id) {
            st = turn.wait(st).unwrap();
        }
        st.holder = Some(id);
        st.note_taken();
        if let Some(r) = st.ready.iter_mut().find(|r| r.thread.id == id) {
            r.asked = None;
        }
        st
    }

    pub fn release(&self, me: &KThread) {
        let mut st = self.state.lock().unwrap();
        if self.parallel {
            st.ready.retain(|r| r.thread.id != me.id);
            st.tick();
            st.wake_runnable();
            self.publish(&st);
            return;
        }
        if let Some(id) = st.holder.take() {
            st.ready.retain(|r| r.thread.id != id);
        }
        st.note_free();
        st.wake_next();
    }

    pub fn give_way(&self, me: &Arc<KThread>) {
        let _prof = crate::prof::enter(crate::prof::Cat::LockWait);
        let mut st = self.state.lock().unwrap();
        if !self.parallel {
            st.holder = None;
            st.note_free();
        }
        drop(self.queue(st, me.id));
    }

    pub fn displaced(&self, me: &Arc<KThread>) -> bool {
        let mut st = self.state.lock().unwrap();
        if self.parallel {
            let lost = st.lost_core(me.id);
            self.publish(&st);
            return lost;
        }
        if st.ready.iter().all(|r| r.asked.is_none()) {
            return false;
        }
        let top = st.schedule();
        !top.iter()
            .flatten()
            .any(|&i| st.ready[i].thread.id == me.id)
    }

    pub fn preempts(&self, me: &Arc<KThread>) -> bool {
        let mut st = self.state.lock().unwrap();
        if st.ready.iter().all(|r| r.asked.is_none()) {
            return false;
        }
        let top = st.schedule();
        top.iter()
            .flatten()
            .any(|&i| st.ready[i].asked.is_some() && st.ready[i].thread.id != me.id)
    }

    pub fn displaced_since(&self, me: &Arc<KThread>, seen: &mut u64) -> bool {
        if self.epoch.load(Ordering::Acquire) == *seen {
            return false;
        }
        let mut st = self.state.lock().unwrap();
        let lost = st.lost_core(me.id);
        self.publish(&st);
        *seen = st.counter;
        lost
    }

    pub fn describe(&self) -> Vec<String> {
        let mut st = self.state.lock().unwrap();
        let top = st.schedule();
        let mut v = vec![format!(
            "cores run {:?}, lock holder {:?}",
            top.map(|t| t.map(|i| st.ready[i].thread.id)),
            st.holder
        )];
        for r in &st.ready {
            v.push(format!(
                "thread {}: core {} (mask {:#x}), priority {}, {}",
                r.thread.id,
                r.core(),
                r.thread.affinity.load(Ordering::Relaxed),
                r.priority(),
                if r.running {
                    "running"
                } else if r.asked.is_some() {
                    "waiting for its core"
                } else {
                    "holds the lock"
                }
            ));
        }
        v
    }

    pub fn is_running(&self, id: u64) -> bool {
        let st = self.state.lock().unwrap();
        if self.parallel {
            st.ready.iter().any(|r| r.thread.id == id && r.running)
        } else {
            st.holder == Some(id)
        }
    }

    pub fn changed(&self) {
        if !self.parallel {
            return;
        }
        let mut st = self.state.lock().unwrap();
        st.tick();
        st.wake_runnable();
        self.publish(&st);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitKind {
    Mutex,
    CondVar,
    Address,
}

#[derive(Clone, Debug)]
pub struct Waiter {
    pub id: u64,
    pub kind: WaitKind,
    pub key: u64,
    pub mutex: u64,
    pub tag: u32,
    pub priority: i32,
    cond: Arc<Condvar>,
}

#[derive(Default)]
pub struct SyncState {
    pub waiters: Vec<Waiter>,
    pub woken: HashMap<u64, u64>,
    next_id: u64,
    blocked_forever: HashMap<u64, String>,
    blocked: HashMap<u64, String>,
    pub boosted: Vec<Arc<KThread>>,
}

impl SyncState {
    pub fn add_waiter(
        &mut self,
        kind: WaitKind,
        key: u64,
        mutex: u64,
        tag: u32,
        priority: i32,
    ) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        self.waiters.push(Waiter {
            id,
            kind,
            key,
            mutex,
            tag,
            priority,
            cond: my_cond(),
        });
        id
    }

    pub fn queue(&self, kind: WaitKind, key: u64) -> Vec<Waiter> {
        let mut q: Vec<Waiter> = self
            .waiters
            .iter()
            .filter(|w| w.kind == kind && w.key == key)
            .cloned()
            .collect();
        q.sort_by_key(|w| w.priority);
        q
    }

    pub fn remove(&mut self, id: u64) -> Option<Waiter> {
        let i = self.waiters.iter().position(|w| w.id == id)?;
        Some(self.waiters.remove(i))
    }

    pub fn wake(&mut self, id: u64, result: u64) {
        if let Some(w) = self.remove(id) {
            w.cond.notify_one();
        }
        self.woken.insert(id, result);
    }
}

#[derive(Default)]
pub struct Scheduler {
    pub gil: Gil,
    pub sync: Mutex<SyncState>,
    pub changed: Condvar,
    pub live: AtomicUsize,
    pub exiting: AtomicBool,
    pub epoch: AtomicU64,
    sleepers: Mutex<HashMap<u64, Arc<Condvar>>>,
    kernel: Mutex<()>,
}

impl Scheduler {
    pub fn lock(&self) -> MutexGuard<'_, SyncState> {
        self.sync.lock().unwrap()
    }

    pub fn notify(&self) {
        self.touch();
        self.changed.notify_all();
        for cond in self.sleepers.lock().unwrap().values() {
            cond.notify_one();
        }
    }

    pub fn kernel_busy(&self) -> bool {
        matches!(
            self.kernel.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        )
    }

    pub fn touch(&self) {
        self.epoch.fetch_add(1, Ordering::Relaxed);
    }
}

thread_local! {
    static COND: Arc<Condvar> = Arc::new(Condvar::new());
    static TURN: Arc<Condvar> = Arc::new(Condvar::new());
    static KERNEL: RefCell<Option<MutexGuard<'static, ()>>> = const { RefCell::new(None) };
}

pub fn kernel_enter(c: &Cpu) {
    let s = &c.process().sched;
    if !s.gil.parallel {
        return;
    }
    let _prof = crate::prof::enter(crate::prof::Cat::KernelWait);
    let guard = s.kernel.lock().unwrap_or_else(|e| e.into_inner());
    KERNEL.with(|k| *k.borrow_mut() = Some(guard));
}

pub fn kernel_exit() -> bool {
    KERNEL.with(|k| k.borrow_mut().take().is_some())
}

fn my_cond() -> Arc<Condvar> {
    COND.with(|c| c.clone())
}

/// Wakes every thread blocked in [`block`] or waiting in [`sleep`]. Taking the
/// scheduler lock is what makes the wakeup reliable: a waiter holds it from its
/// readiness check to its wait, so a notifier that holds it either sees the
/// waiter before it sleeps or after it is asleep. Never call this while holding
/// the handle table: [`block`] calls the readiness closure with the scheduler
/// lock held and that closure may read the table.
pub fn wake(p: &Process) {
    let s = &p.sched;
    let _st = s.lock();
    s.notify();
}

const DEADLOCK_TICK: Duration = Duration::from_millis(500);

pub fn block<T>(
    c: &Cpu,
    timeout_ns: i64,
    what: impl FnOnce() -> String,
    mut ready: impl FnMut(&mut SyncState) -> Option<T>,
) -> Option<T> {
    let p = c.process();
    let s = &p.sched;
    let mut st = s.lock();
    if let Some(v) = ready(&mut st) {
        return Some(v);
    }
    if timeout_ns == 0 || s.exiting.load(Ordering::Relaxed) {
        return None;
    }
    let deadline =
        (timeout_ns > 0).then(|| Instant::now() + Duration::from_nanos(timeout_ns as u64));
    if deadline.is_none() {
        st.blocked_forever.insert(c.thread.id, String::new());
    }
    let mut what = Some(what);
    let mut describe = |st: &mut SyncState| {
        if let Some(what) = what.take() {
            let description = what();
            let pcs = format!("{} (lr {})", p.describe(c.pc), p.describe(c.x[30]));
            st.blocked.insert(
                c.thread.id,
                format!(
                    "{description}{} at {pcs}",
                    if deadline.is_some() {
                        " (with timeout)"
                    } else {
                        ""
                    }
                ),
            );
            if deadline.is_none() {
                st.blocked_forever.insert(c.thread.id, description);
            }
        }
    };
    let kernel = park(c);
    let prof = crate::prof::enter(crate::prof::Cat::Blocked);
    let cond = my_cond();
    s.sleepers.lock().unwrap().insert(c.thread.id, cond.clone());
    let mut quiet_ticks = 0;
    let mut seen_epoch = s.epoch.load(Ordering::Relaxed);
    let result = loop {
        if s.exiting.load(Ordering::Relaxed) {
            break None;
        }
        st = match deadline {
            None => {
                let (mut guard, waited) = cond.wait_timeout(st, DEADLOCK_TICK).unwrap();
                if waited.timed_out() {
                    describe(&mut guard);
                    let epoch = s.epoch.load(Ordering::Relaxed);
                    quiet_ticks = if epoch == seen_epoch {
                        quiet_ticks + 1
                    } else {
                        0
                    };
                    seen_epoch = epoch;
                    if quiet_ticks >= 2 {
                        check_deadlock(p, &guard);
                    }
                }
                guard
            }
            Some(d) => {
                let now = Instant::now();
                if now >= d {
                    break ready(&mut st);
                }
                let (mut guard, waited) =
                    cond.wait_timeout(st, (d - now).min(DEADLOCK_TICK)).unwrap();
                if waited.timed_out() {
                    describe(&mut guard);
                }
                guard
            }
        };
        if s.exiting.load(Ordering::Relaxed) {
            break None;
        }
        if let Some(v) = ready(&mut st) {
            break Some(v);
        }
    };
    st.blocked_forever.remove(&c.thread.id);
    st.blocked.remove(&c.thread.id);
    s.sleepers.lock().unwrap().remove(&c.thread.id);
    drop(st);
    drop(prof);
    unpark(c, kernel);
    result
}

fn park(c: &Cpu) -> bool {
    save_context(c);
    let kernel = kernel_exit();
    c.process().sched.gil.release(&c.thread);
    kernel
}

fn save_context(c: &Cpu) {
    *c.thread.context.lock().unwrap() = crate::kernel::ThreadContext {
        x: c.x,
        sp: c.sp,
        pc: c.pc,
        psr: c.nzcv as u32,
        v: c.v,
        fpcr: c.fpcr as u32,
        fpsr: c.fpsr as u32,
        tpidr: c.tpidr_el0,
    };
}

fn unpark(c: &Cpu, kernel: bool) {
    c.process().sched.gil.acquire(&c.thread);
    wait_while_paused(c);
    if kernel {
        kernel_enter(c);
    }
}

pub fn without_gil<T>(c: &Cpu, work: impl FnOnce() -> T) -> T {
    let kernel = park(c);
    let prof = crate::prof::enter(crate::prof::Cat::Unlocked);
    let r = work();
    drop(prof);
    unpark(c, kernel);
    r
}

fn wait_while_paused(c: &Cpu) {
    let s = &c.process().sched;
    while c.thread.paused.load(Ordering::Relaxed) && !s.exiting.load(Ordering::Relaxed) {
        s.gil.release(&c.thread);
        let _prof = crate::prof::enter(crate::prof::Cat::Blocked);
        let mut st = s.lock();
        while c.thread.paused.load(Ordering::Relaxed) && !s.exiting.load(Ordering::Relaxed) {
            st = s.changed.wait(st).unwrap();
        }
        drop(st);
        s.gil.acquire(&c.thread);
    }
}

pub fn blocked_threads(p: &Process) -> Vec<String> {
    let st = p.sched.lock();
    let mut v: Vec<String> = st
        .blocked
        .iter()
        .map(|(id, what)| format!("thread {id}: {what}"))
        .collect();
    v.sort();
    v
}

pub fn check_deadlock(p: &Process, st: &SyncState) {
    let live = p.sched.live.load(Ordering::Relaxed);
    if live == 0 || st.blocked_forever.len() < live || p.sched.exiting.load(Ordering::Relaxed) {
        return;
    }
    let mut msg = String::from("kirikae: deadlock: every guest thread is waiting forever\n");
    let mut waits: Vec<_> = st.blocked_forever.iter().collect();
    waits.sort();
    for (id, what) in waits {
        msg.push_str(&format!("  thread {id}: {what}\n"));
    }
    eprint!("{msg}");
    p.set_exit_code(5);
    p.sched.exiting.store(true, Ordering::Relaxed);
    p.sched.notify();
}

pub fn yield_now(c: &mut Cpu) -> bool {
    c.thread.last_pc.store(c.pc, Ordering::Relaxed);
    c.thread.last_lr.store(c.x[30], Ordering::Relaxed);
    let s = &c.process().sched;
    if s.exiting.load(Ordering::Relaxed) {
        c.halt();
        return true;
    }
    if s.gil.parallel {
        if c.thread.paused.load(Ordering::Relaxed) {
            let kernel = kernel_exit();
            save_context(c);
            wait_while_paused(c);
            if kernel {
                kernel_enter(c);
            }
        } else {
            keep_core(c);
        }
    } else if s.gil.preempts(&c.thread) {
        save_context(c);
        s.gil.give_way(&c.thread);
        wait_while_paused(c);
    }
    s.exiting.load(Ordering::Relaxed) && {
        c.halt();
        true
    }
}

fn keep_core(c: &mut Cpu) {
    let s = &c.process().sched;
    if s.gil.displaced_since(&c.thread, &mut c.sched_seen) {
        let kernel = kernel_exit();
        save_context(c);
        s.gil.give_way(&c.thread);
        wait_while_paused(c);
        if kernel {
            kernel_enter(c);
        }
    }
}

pub fn reschedule(c: &mut Cpu) {
    let s = &c.process().sched;
    if s.exiting.load(Ordering::Relaxed) {
        return;
    }
    if s.gil.parallel {
        keep_core(c);
    } else if s.gil.displaced(&c.thread) {
        save_context(c);
        s.gil.give_way(&c.thread);
        wait_while_paused(c);
    }
}

pub fn wait_stopped(c: &mut Cpu, t: &KThread) {
    let s = &c.process().sched;
    while s.gil.parallel && s.gil.is_running(t.id) && !s.exiting.load(Ordering::Relaxed) {
        sleep(c, Duration::from_micros(50));
    }
}

pub fn sleep(c: &mut Cpu, d: Duration) {
    let kernel = park(c);
    {
        let _prof = crate::prof::enter(crate::prof::Cat::Sleep);
        if d.is_zero() {
            std::thread::yield_now()
        } else {
            std::thread::sleep(d)
        }
    }
    unpark(c, kernel);
}

pub fn spawn(
    p: &'static Process,
    thread: Arc<KThread>,
    setup: impl FnOnce(&mut Cpu) + Send + 'static,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    p.sched.live.fetch_add(1, Ordering::Relaxed);
    std::thread::Builder::new()
        .name(format!("guest-{}", thread.id))
        .stack_size(HOST_STACK)
        .spawn(move || {
            p.sched.gil.acquire(&thread);
            let mut c = Cpu::new(p, thread.clone());
            c.x[30] = HALT_PC;
            c.tpidrro_el0 = thread.tls;
            setup(&mut c);
            crate::fault::install(&mut c);
            wait_while_paused(&c);
            if !p.sched.exiting.load(Ordering::Relaxed) {
                let sp = c.sp;
                run_until(&mut c, HALT_PC, sp);
            }
            p.stats.flush();
            {
                let st = p.sched.lock();
                thread.exited.store(true, Ordering::Relaxed);
                p.sched.live.fetch_sub(1, Ordering::Relaxed);
                p.sched.notify();
                drop(st);
            }
            kernel_exit();
            p.sched.gil.release(&thread);
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;

    fn thread(id: u64, priority: i32, core: i32, affinity: u64) -> Arc<KThread> {
        Arc::new(KThread {
            id,
            handle: id as u32,
            priority: AtomicI32::new(priority),
            core: AtomicI32::new(core),
            ideal_core: AtomicI32::new(core),
            affinity: AtomicU64::new(affinity),
            inherited: AtomicI32::new(i32::MAX),
            entry: 0,
            arg: 0,
            stack_top: 0,
            tls: 0,
            started: AtomicBool::new(true),
            exited: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            last_pc: AtomicU64::new(0),
            last_lr: AtomicU64::new(0),
            paused: AtomicBool::new(false),
            context: Default::default(),
        })
    }

    fn state(threads: &[Arc<KThread>]) -> GilState {
        let mut st = GilState::default();
        for t in threads {
            let seq = st.tick();
            st.ready.push(Ready {
                thread: t.clone(),
                seq,
                asked: Some(seq),
                turn: Arc::new(Condvar::new()),
                running: false,
            });
        }
        st
    }

    fn next(st: &mut GilState) -> Option<u64> {
        let i = st.next()?;
        Some(st.ready[i].thread.id)
    }

    fn chosen(st: &mut GilState) -> Vec<Option<u64>> {
        st.schedule()
            .iter()
            .map(|i| i.map(|i| st.ready[i].thread.id))
            .collect()
    }

    #[test]
    fn equal_priorities_on_a_core_run_in_order() {
        let mut st = state(&[
            thread(1, 40, 0, 1),
            thread(2, 40, 0, 1),
            thread(3, 44, 1, 2),
        ]);
        assert_eq!(chosen(&mut st), [Some(1), Some(3), None, None]);
        assert_eq!(next(&mut st), Some(1));
    }

    #[test]
    fn better_priority_takes_the_core() {
        let mut st = state(&[thread(1, 40, 0, 1), thread(2, 28, 0, 1)]);
        assert_eq!(chosen(&mut st), [Some(2), None, None, None]);
    }

    #[test]
    fn idle_cores_take_threads_their_mask_allows() {
        let (a, b, c) = (
            thread(1, 40, 0, 0b1111),
            thread(2, 40, 0, 0b0110),
            thread(3, 40, 0, 0b0001),
        );
        let mut st = state(&[a, b.clone(), c]);
        assert_eq!(chosen(&mut st), [Some(3), Some(2), Some(1), None]);
        assert_eq!(b.core.load(Ordering::Relaxed), 1);
        assert_eq!(chosen(&mut st), [Some(3), Some(2), Some(1), None]);
    }

    #[test]
    fn a_best_thread_moves_when_its_core_has_another() {
        let a = thread(1, 40, 0, 0b0011);
        let mut st = state(&[a.clone(), thread(2, 40, 0, 0b0001)]);
        assert_eq!(chosen(&mut st), [Some(2), Some(1), None, None]);
        assert_eq!(a.core.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn parallel_cores_run_their_threads_together() {
        let mut st = state(&[
            thread(1, 44, 0, 1),
            thread(2, 44, 1, 2),
            thread(3, 30, 2, 4),
            thread(4, 44, 2, 4),
        ]);
        assert!(st.may_run(1) && st.may_run(2) && st.may_run(3));
        assert!(!st.may_run(4));
    }

    #[test]
    fn parallel_core_waits_for_the_thread_that_lost_it() {
        let mut st = state(&[thread(1, 44, 0, 1), thread(2, 40, 0, 1)]);
        st.ready[0].asked = None;
        st.ready[0].running = true;
        assert!(st.lost_core(1));
        assert!(!st.may_run(2), "1 still runs on core 0");
        st.ready[0].running = false;
        st.ready[0].asked = Some(st.tick());
        assert!(st.may_run(2));
        assert!(!st.may_run(1));
    }

    #[test]
    fn lock_goes_to_the_earliest_chosen_waiter() {
        let mut st = state(&[
            thread(1, 40, 1, 2),
            thread(2, 40, 0, 1),
            thread(3, 40, 0, 1),
        ]);
        assert_eq!(next(&mut st), Some(1));
        st.ready[0].asked = None;
        assert_eq!(next(&mut st), Some(2));
    }
}
