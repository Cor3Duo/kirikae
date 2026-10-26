use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use kirikae_a64::Insn;
use kirikae_loader::{Executable, Module, LOAD_BASE};

use crate::cpu::Cpu;
use crate::horizon::{MAIN_THREAD_HANDLE, MAIN_THREAD_ID, MAIN_THREAD_PRIORITY, STACK_REGION_BASE, TICK_HZ};
use crate::kernel::{KObject, KThread};
use crate::mem::{perm, state, GuestMemory, PAGE};

pub use crate::abi::{Entry, GuestFn};

pub const DEFAULT_STACK_SIZE: u64 = 0x10_0000;
pub const TLS_BASE: u64 = STACK_REGION_BASE + 0x8000_0000;
pub const TLS_SIZE: u64 = 0x200;
const LOOKUP_SLOTS: usize = 1 << 16;

#[derive(Default)]
pub struct Stats {
    pub interpreted: AtomicU64,
    pub native_calls: AtomicU64,
}

#[derive(Default)]
struct Pending {
    native: std::cell::Cell<u64>,
    interpreted: std::cell::Cell<u64>,
}

thread_local! {
    static PENDING: Pending = Pending::default();
}

const PENDING_BATCH: u64 = 1024;

impl Stats {
    #[inline]
    pub fn count_native(&self) {
        PENDING.with(|p| {
            let n = p.native.get() + 1;
            if n == PENDING_BATCH {
                self.native_calls.fetch_add(n, Ordering::Relaxed);
                p.native.set(0);
            } else {
                p.native.set(n);
            }
        });
    }

    #[inline]
    pub fn count_interpreted(&self) {
        PENDING.with(|p| {
            let n = p.interpreted.get() + 1;
            if n == PENDING_BATCH {
                self.interpreted.fetch_add(n, Ordering::Relaxed);
                p.interpreted.set(0);
            } else {
                p.interpreted.set(n);
            }
        });
    }

    pub fn flush(&self) {
        PENDING.with(|p| {
            self.native_calls.fetch_add(p.native.replace(0), Ordering::Relaxed);
            self.interpreted.fetch_add(p.interpreted.replace(0), Ordering::Relaxed);
        });
    }
}

enum Sink {
    Stdout,
    Capture(Vec<u8>),
}

pub struct Process {
    pub memory: GuestMemory,
    modules: Vec<(String, std::ops::Range<u64>)>,
    symbols: Vec<(u64, String)>,
    pub title_id: u64,
    code_size: u64,
    stack_size: u64,
    entries: &'static [Entry],
    lookup_cache: Box<[AtomicU32]>,
    pub(crate) interp_cache: std::sync::RwLock<HashMap<u64, Arc<Insn>>>,
    pub(crate) heap: Mutex<u64>,
    pub(crate) physical: AtomicU64,
    pub system_resource_size: u64,
    pub handles: Mutex<crate::kernel::HandleTable>,
    pub sched: crate::sched::Scheduler,
    tls_next: Mutex<u64>,
    pub exception_context: std::sync::OnceLock<u64>,
    pub romfs: Option<Mutex<kirikae_loader::romfs::Reader>>,
    pub gpu: crate::gpu::Gpu,
    pub hle: crate::hle::Table,
    service_state: Mutex<HashMap<&'static str, Arc<dyn std::any::Any + Send + Sync>>>,
    pub stats: Stats,
    sink: Mutex<Sink>,
    warned: Mutex<HashSet<String>>,
    exit_code: AtomicI32,
    start: Instant,
}

impl Process {
    pub fn lookup(&self, pc: u64) -> Option<GuestFn> {
        let slot = &self.lookup_cache[((pc >> 2) ^ (pc >> 18)) as usize & (LOOKUP_SLOTS - 1)];
        let i = slot.load(Ordering::Relaxed);
        if i != 0 {
            let e = &self.entries[i as usize - 1];
            if e.addr == pc {
                return Some(e.func);
            }
        }
        if self.hle.owns(pc) {
            return Some(crate::hle::trampoline);
        }
        let i = self.entries.binary_search_by_key(&pc, |e| e.addr).ok()?;
        slot.store(i as u32 + 1, Ordering::Relaxed);
        Some(self.entries[i].func)
    }

    pub fn lookup_compiled(&self, pc: u64) -> Option<GuestFn> {
        let i = self.entries.binary_search_by_key(&pc, |e| e.addr).ok()?;
        Some(self.entries[i].func)
    }

    pub fn symbol(&self, name: &str) -> Option<u64> {
        self.symbols.iter().find(|s| s.1 == name).map(|s| s.0)
    }

    pub fn ticks(&self) -> u64 {
        (self.start.elapsed().as_nanos() * TICK_HZ as u128 / 1_000_000_000) as u64
    }

    pub fn output(&self, bytes: &[u8]) {
        match &mut *self.sink.lock().unwrap() {
            Sink::Stdout => {
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(bytes);
                let _ = out.flush();
            }
            Sink::Capture(buf) => buf.extend_from_slice(bytes),
        }
    }

    pub fn create_event(&self, signaled: bool) -> u32 {
        let event = Arc::new(crate::kernel::Event { signaled: AtomicBool::new(signaled) });
        self.handles.lock().unwrap().insert(KObject::Event(event))
    }

    pub fn set_event(&self, handle: u32, signaled: bool) -> bool {
        let Some(KObject::Event(e)) = self.handles.lock().unwrap().get(handle) else { return false };
        self.sched.store_event(&e, signaled);
        true
    }

    pub fn service_state<T: std::any::Any + Send + Sync>(&self, key: &'static str, init: impl FnOnce() -> T) -> Arc<T> {
        let mut map = self.service_state.lock().unwrap();
        if let Some(s) = map.get(key) {
            return s.clone().downcast::<T>().expect("service state type");
        }
        let s = Arc::new(init());
        map.insert(key, s.clone());
        s
    }

    pub fn create_shared_memory(&self, size: u64) -> (Arc<crate::kernel::SharedMemory>, u32) {
        let s = Arc::new(crate::kernel::SharedMemory::new(size));
        let h = self.handles.lock().unwrap().insert(KObject::SharedMemory(s.clone()));
        (s, h)
    }

    pub fn terminate(&self, code: i32) {
        if code != 0 {
            self.set_exit_code(code);
        }
        let _st = self.sched.lock();
        self.sched.exiting.store(true, Ordering::Relaxed);
        self.sched.notify();
    }

    pub fn alloc_tls(&self) -> u64 {
        let mut next = self.tls_next.lock().unwrap();
        let addr = *next;
        *next += TLS_SIZE;
        if addr.is_multiple_of(PAGE) {
            let _ = self.memory.map_as(addr, PAGE, state::THREAD_LOCAL, perm::RW);
        }
        addr
    }

    pub fn log_ipc(&self, message: String) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var_os("KIRIKAE_TRACE_IPC").is_some()) {
            let _ = writeln!(std::io::stderr(), "ipc: {message}");
        }
    }

    pub fn warn_once(&self, message: String) {
        if self.warned.lock().unwrap().insert(message.clone()) {
            let _ = writeln!(std::io::stderr(), "kirikae: {message}");
        }
    }

    pub fn set_exit_code(&self, code: i32) {
        self.exit_code.store(code, Ordering::Relaxed);
    }

    pub fn module_base(&self, name: &str) -> Option<u64> {
        self.modules.iter().find(|(n, _)| n == name).map(|(_, r)| r.start)
    }

    pub fn describe(&self, addr: u64) -> String {
        match self.modules.iter().find(|(_, r)| r.contains(&addr)) {
            Some((name, r)) => {
                let i = self.symbols.partition_point(|s| s.0 <= addr);
                match i.checked_sub(1).map(|i| &self.symbols[i]).filter(|s| r.contains(&s.0) && addr - s.0 < 0x4000) {
                    Some((at, sym)) => format!("{name}+{:#x} ({sym}+{:#x})", addr - r.start, addr - at),
                    None => format!("{name}+{:#x}", addr - r.start),
                }
            }
            None => format!("{addr:#x}"),
        }
    }

    pub fn used_memory(&self) -> u64 {
        self.code_size + *self.heap.lock().unwrap() + self.physical.load(Ordering::Relaxed) + self.stack_size + self.system_resource_size
    }

    pub fn entropy(&self, subid: u64) -> u64 {
        let mut z = 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(subid + 1) ^ self.title_id;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn start_stats(p: &'static Process) {
    let _ = std::thread::Builder::new().name("kirikae-stats".into()).spawn(move || {
        let mut last_frames = 0;
        let mut last_paced = 0;
        let mut last_times = HashMap::new();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            if p.sched.exiting.load(Ordering::Relaxed) {
                return;
            }
            let g = &p.gpu.stats;
            let frames = crate::display::FRAMES.load(Ordering::Relaxed);
            let paced = crate::display::PACED_NS.load(Ordering::Relaxed);
            eprintln!(
                "kirikae: [{:>4}s] {frames} frames ({:.1} fps, {:.1} without the limit, longest {:.0} ms), GPU: {} submits, {} draws, {} clears, {} copies, {} dispatches; {} native calls, {} interpreted",
                p.start.elapsed().as_secs(),
                (frames - last_frames) as f64 / 10.0,
                crate::display::uncapped_fps(frames - last_frames, 10.0, paced - last_paced),
                crate::display::take_longest_frame().as_secs_f64() * 1000.0,
                g.submits.load(Ordering::Relaxed),
                g.draws.load(Ordering::Relaxed),
                g.clears.load(Ordering::Relaxed),
                g.copies.load(Ordering::Relaxed),
                g.dispatches.load(Ordering::Relaxed),
                p.stats.native_calls.load(Ordering::Relaxed),
                p.stats.interpreted.load(Ordering::Relaxed),
            );
            if let Some(r) = p.gpu.render_stats() {
                eprintln!(
                    "kirikae:         vulkan: {} draws, {} skipped, {} writing nothing, {} with the descriptors of the one before, {} dispatches, {} skipped, {} uploads ({} refreshed), {} pipelines; {:.1}s drawing, {:.1}s waiting for the GPU, {:.1}s reading back, {:.1}s translating shaders",
                    r.draws,
                    r.skipped,
                    r.noop_draws,
                    r.pushes_reused,
                    r.dispatches,
                    r.skipped_dispatches,
                    r.uploads,
                    r.refreshes,
                    r.pipelines,
                    r.draw_time.as_secs_f64(),
                    r.flush_time.as_secs_f64(),
                    r.readback_time.as_secs_f64(),
                    r.shader_time.as_secs_f64()
                );
                let draws = r.draws.max(1) as f64;
                let parts: Vec<String> = crate::gpu::vulkan::DRAW_PARTS
                    .iter()
                    .zip(r.draw_parts)
                    .map(|(name, t)| format!("{} {:.1}", name.trim(), t.as_secs_f64() * 1e6 / draws))
                    .collect();
                if r.draw_parts.iter().any(|t| !t.is_zero()) {
                    eprintln!("kirikae:         µs per draw: {}; crops {} of {} calls copied, {} crop images made", parts.join(", "), r.crop_copies, r.crop_calls, r.crop_images);
                }
                eprintln!(
                    "kirikae:         vulkan: {:.1}s of GPU work, {} frames through the swapchain, {} batches, {} fence waits, {} storage resolves, {} storage patches, {} MiB uploaded ({} MiB constant buffers, {} MiB vertex streams), {} MiB from the buffer cache ({} of {} MiB in VRAM), {} MiB written back",
                    r.gpu_time.as_secs_f64(),
                    r.presents,
                    r.batches,
                    r.waits,
                    r.resolves,
                    r.patches,
                    r.ring_bytes >> 20,
                    r.cbuf_bytes >> 20,
                    r.vertex_bytes >> 20,
                    r.cached_bytes >> 20,
                    r.cache_memory.0 >> 20,
                    r.cache_memory.1 >> 20,
                    r.written_back >> 20
                );
                let mut reasons: Vec<_> = r.skip_reasons.iter().collect();
                reasons.sort_by(|a, b| b.1.cmp(a.1));
                for (why, n) in reasons.iter().take(6) {
                    eprintln!("kirikae:           {n:>7} skipped: {why}");
                }
                let mut placeholders: Vec<_> = r.placeholders.iter().collect();
                placeholders.sort_by(|a, b| b.1.cmp(a.1));
                for (why, n) in placeholders.iter().take(4) {
                    eprintln!("kirikae:           {n:>7} {why}");
                }
            }
            print_times(&mut last_times, frames - last_frames);
            crate::nvn::report(frames - last_frames);
            crate::prof::write_simd_hist();
            if let Some(path) = std::env::var_os("KIRIKAE_SIMD_HIST") {
                let text: String = crate::prof::op_hist().iter().map(|(i, n)| format!("{} {n}
", kirikae_a64::Op::ALL.get(*i).map_or("?", |o| o.name()))).collect();
                let mut path = std::path::PathBuf::from(path).into_os_string();
                path.push(".ops");
                let _ = std::fs::write(path, text);
            }
            if frames == last_frames && frames > 0 {
                let live = p.sched.live.load(Ordering::Relaxed);
                eprintln!("kirikae:         no new frame; {live} live threads, blocked:");
                let blocked = crate::sched::blocked_threads(p);
                for line in &blocked {
                    eprintln!("kirikae:           {line}");
                }
                eprintln!("kirikae:         scheduler (kernel lock {}):", if p.sched.kernel_busy() { "held" } else { "free" });
                for line in p.sched.gil.describe() {
                    eprintln!("kirikae:           {line}");
                }
                let threads: Vec<std::sync::Arc<KThread>> = p.handles.lock().unwrap().threads();
                for t in threads {
                    let tag = format!("thread {}:", t.id);
                    if t.started.load(Ordering::Relaxed) && !t.exited.load(Ordering::Relaxed) && !blocked.iter().any(|b| b.starts_with(&tag)) {
                        eprintln!(
                            "kirikae:           thread {} running, last at {} (lr {})",
                            t.id,
                            p.describe(t.last_pc.load(Ordering::Relaxed)),
                            p.describe(t.last_lr.load(Ordering::Relaxed))
                        );
                    }
                }
            }
            last_frames = frames;
            last_paced = paced;
        }
    });
}

fn print_times(last: &mut HashMap<String, [u64; crate::prof::CATS]>, frames: u64) {
    use crate::prof::{Cat, CATS, HELD, NAMES};
    let mut total = [0u64; CATS];
    let mut rows = Vec::new();
    for (i, (name, ns, simd, ops)) in crate::prof::snapshot().into_iter().enumerate() {
        let key = format!("{name}#{i}");
        let prev = last.insert(key, ns).unwrap_or([0; CATS]);
        let d: [u64; CATS] = std::array::from_fn(|k| ns[k] - prev[k]);
        for k in 0..CATS {
            total[k] += d[k];
        }
        let busy: u64 = HELD.iter().map(|&k| d[k as usize]).sum::<u64>() + d[Cat::Gpu as usize] + d[Cat::Present as usize] + d[Cat::Video as usize];
        rows.push((busy, name, d, simd, ops));
    }
    let per = |ns: u64| ns as f64 / 1e6 / frames.max(1) as f64;
    let held: u64 = HELD.iter().map(|&k| total[k as usize]).sum();
    let line: Vec<String> = (0..CATS).filter(|&k| total[k] > 0).map(|k| format!("{} {:.1}", NAMES[k], per(total[k]))).collect();
    eprintln!("kirikae:         time per frame (ms, all threads): lock held {:.1} = {}", per(held), line.join(", "));
    let lock = [&crate::prof::LOCK_HANDOFF_NS, &crate::prof::LOCK_HANDOFFS, &crate::prof::LOCK_IDLE_NS].map(|a| a.load(Ordering::Relaxed));
    let prev = last.insert("lock".into(), std::array::from_fn(|k| lock.get(k).copied().unwrap_or(0))).unwrap_or([0; CATS]);
    let n = lock[1] - prev[1];
    eprintln!(
        "kirikae:           lock free per frame: {:.1} ms handing over ({:.0} hand-overs, {:.1} µs each), {:.1} ms with nobody waiting",
        per(lock[0] - prev[0]),
        n as f64 / frames.max(1) as f64,
        (lock[0] - prev[0]) as f64 / 1e3 / n.max(1) as f64,
        per(lock[2] - prev[2])
    );
    let mut svcs: Vec<(u64, usize, u64)> = (0..0x80)
        .map(|i| {
            let (ns, n) = (crate::prof::SVC_NS[i].load(Ordering::Relaxed), crate::prof::SVC_CALLS[i].load(Ordering::Relaxed));
            let prev = last.insert(format!("svc{i}"), std::array::from_fn(|k| [ns, n].get(k).copied().unwrap_or(0))).unwrap_or([0; CATS]);
            (ns - prev[0], i, n - prev[1])
        })
        .filter(|s| s.0 > 0)
        .collect();
    svcs.sort_by_key(|r| std::cmp::Reverse(r.0));
    let line: Vec<String> = svcs.iter().take(8).map(|(ns, i, n)| format!("{i:#x} {:.1} ms ({:.0} calls)", per(*ns), *n as f64 / frames.max(1) as f64)).collect();
    eprintln!("kirikae:           SVCs per frame: {}", line.join(", "));
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    for (busy, name, d, simd, ops) in rows.iter().take(6) {
        if *busy == 0 {
            break;
        }
        let line: Vec<String> = (0..CATS).filter(|&k| d[k] as f64 > 1e5 * frames as f64).map(|k| format!("{} {:.1}", NAMES[k], per(d[k]))).collect();
        eprintln!("kirikae:           {name}: {} ({simd} SIMD/FP helper calls, {ops} op helper calls in total)", line.join(", "));
    }
}

pub fn run_until(c: &mut Cpu, stop_pc: u64, stop_sp: u64) {
    let p = c.process();
    let prof = crate::prof::on().then(|| crate::prof::set(crate::prof::Cat::Native));
    let mut entering = true;
    loop {
        if c.halted || (c.pc == stop_pc && c.sp >= stop_sp) {
            break;
        }
        match p.lookup(c.pc) {
            Some(f) => {
                p.stats.count_native();
                if prof.is_some() {
                    crate::prof::set(crate::prof::Cat::Native);
                }
                let ret = c.x[30];
                f(c);
                entering = c.pc != ret;
            }
            None => {
                let pc = c.pc;
                if prof.is_some() {
                    crate::prof::set(crate::prof::Cat::Interp);
                }
                if entering && crate::coverage::on() {
                    crate::coverage::record(pc);
                }
                entering = false;
                crate::interp::step(c);
                if c.pc <= pc && c.poll() {
                    break;
                }
            }
        }
    }
    if let Some(prev) = prof {
        crate::prof::set(prev);
    }
}

#[inline(never)]
pub fn call_indirect(c: &mut Cpu, ret: u64) {
    let sp = c.sp;
    run_until(c, ret, sp);
}

pub struct RunOptions {
    pub capture_output: bool,
}

pub struct RunResult {
    pub exit_code: i32,
    pub output: Vec<u8>,
    pub interpreted: u64,
    pub native_calls: u64,
}

pub fn run(file: &[u8], entries: &'static [Entry], opts: RunOptions) -> Result<RunResult, String> {
    let image = kirikae_loader::load(file, LOAD_BASE).map_err(|e| e.to_string())?;
    let entry = image.entry;
    let exe = Executable { modules: vec![Module { name: "main".into(), image }], entry, title_id: None, main_stack_size: None, system_resource_size: 0, romfs: None };
    run_executable(&exe, entries, opts)
}

fn map_module(memory: &GuestMemory, m: &Module) -> Result<(), String> {
    let i = &m.image;
    let page = |v: u64| v.div_ceil(PAGE) * PAGE;
    let err = |e: std::io::Error| format!("mapping {}: {e}", m.name);
    memory.map_as(i.text.start, page(i.text.end) - i.text.start, state::CODE_STATIC, perm::RX).map_err(err)?;
    if i.ro.end > i.ro.start {
        memory.map_as(i.ro.start, page(i.ro.end) - i.ro.start, state::CODE_STATIC, perm::R).map_err(err)?;
    }
    let rw_end = page(i.bss.end.max(i.data.end));
    if rw_end > i.data.start {
        memory.map_as(i.data.start, rw_end - i.data.start, state::CODE_MUTABLE, perm::RW).map_err(err)?;
    }
    memory.write(i.base, &i.memory);
    Ok(())
}

pub fn run_executable(exe: &Executable, entries: &'static [Entry], opts: RunOptions) -> Result<RunResult, String> {
    if !entries.windows(2).all(|w| w[0].addr < w[1].addr) {
        return Err("the recompiled function table is not sorted".into());
    }
    let memory = GuestMemory::new().map_err(|e| format!("could not reserve guest memory: {e}"))?;
    for m in &exe.modules {
        map_module(&memory, m)?;
    }
    let stack_size = exe.main_stack_size.unwrap_or(DEFAULT_STACK_SIZE).max(PAGE).div_ceil(PAGE) * PAGE;
    let stack_top = STACK_REGION_BASE + stack_size;
    memory.map_as(STACK_REGION_BASE, stack_size, state::STACK, perm::RW).map_err(|e| e.to_string())?;

    let process: &'static Process = Box::leak(Box::new(Process {
        memory,
        modules: exe.modules.iter().map(|m| (m.name.clone(), m.image.base..m.image.end())).collect(),
        symbols: {
            let mut all: Vec<(u64, String)> = exe.modules.iter().flat_map(|m| kirikae_loader::dynamic::function_symbols(&m.image)).collect();
            all.sort_unstable();
            all
        },
        title_id: exe.title_id.unwrap_or(0),
        code_size: exe.modules.iter().map(|m| m.image.memory.len() as u64).sum(),
        stack_size,
        entries,
        lookup_cache: (0..LOOKUP_SLOTS).map(|_| AtomicU32::new(0)).collect(),
        interp_cache: std::sync::RwLock::new(HashMap::new()),
        heap: Mutex::new(0),
        physical: AtomicU64::new(0),
        system_resource_size: exe.system_resource_size,
        handles: Mutex::new(Default::default()),
        sched: Default::default(),
        tls_next: Mutex::new(TLS_BASE),
        exception_context: std::sync::OnceLock::new(),
        gpu: Default::default(),
        hle: Default::default(),
        service_state: Mutex::new(HashMap::new()),
        romfs: exe.romfs.as_ref().and_then(|r| match kirikae_loader::romfs::Reader::open(r) {
            Ok(reader) => Some(Mutex::new(reader)),
            Err(e) => {
                eprintln!("kirikae: the RomFS at {} cannot be read ({e}); the game will not find its assets", r.path.display());
                None
            }
        }),
        stats: Stats::default(),
        sink: Mutex::new(if opts.capture_output { Sink::Capture(Vec::new()) } else { Sink::Stdout }),
        warned: Mutex::new(HashSet::new()),
        exit_code: AtomicI32::new(0),
        start: Instant::now(),
    }));

    if std::env::var_os("KIRIKAE_STATS").is_some() {
        crate::prof::enable();
        start_stats(process);
    }
    crate::nvn::install(process);

    let entry = exe.entry;
    let horizon_abi = exe.modules.first().is_some_and(|m| m.name == "rtld");
    let main = Arc::new(KThread {
        id: MAIN_THREAD_ID,
        handle: MAIN_THREAD_HANDLE as u32,
        priority: AtomicI32::new(MAIN_THREAD_PRIORITY as i32),
        core: AtomicI32::new(0),
        ideal_core: AtomicI32::new(0),
        affinity: AtomicU64::new(1),
        inherited: AtomicI32::new(i32::MAX),
        entry,
        arg: 0,
        stack_top,
        tls: process.alloc_tls(),
        started: AtomicBool::new(true),
        exited: AtomicBool::new(false),
        cancelled: AtomicBool::new(false),
        last_pc: AtomicU64::new(0),
        last_lr: AtomicU64::new(0),
        paused: AtomicBool::new(false),
        context: Default::default(),
    });
    process.handles.lock().unwrap().insert_at(main.handle, KObject::Thread(main.clone()));
    let thread = crate::sched::spawn(process, main, move |c| {
        c.x[0] = 0;
        c.x[1] = if horizon_abi { MAIN_THREAD_HANDLE } else { u64::MAX };
        c.sp = stack_top;
        c.pc = entry;
    })
    .map_err(|e| e.to_string())?;
    thread.join().map_err(|_| "the guest thread panicked".to_string())?;
    {
        let s = &process.sched;
        let mut st = s.lock();
        while s.live.load(Ordering::Relaxed) > 0 && !s.exiting.load(Ordering::Relaxed) {
            st = s.changed.wait(st).unwrap();
        }
    }

    let output = match &mut *process.sink.lock().unwrap() {
        Sink::Capture(buf) => std::mem::take(buf),
        Sink::Stdout => Vec::new(),
    };
    Ok(RunResult {
        exit_code: process.exit_code.load(Ordering::Relaxed),
        output,
        interpreted: process.stats.interpreted.load(Ordering::Relaxed),
        native_calls: process.stats.native_calls.load(Ordering::Relaxed),
    })
}

pub fn program_main(package: &'static [u8], entries: &'static [Entry]) -> ! {
    let exe = match kirikae_loader::package::unpack(package) {
        Ok(exe) => exe,
        Err(e) => {
            eprintln!("kirikae: the embedded executable is damaged: {e}");
            std::process::exit(3)
        }
    };
    crate::coverage::start();
    crate::usage::start(exe.title_id.unwrap_or(0));
    let result = run_executable(&exe, entries, RunOptions { capture_output: false });
    crate::usage::finish();
    match result {
        Ok(r) => {
            if std::env::var_os("KIRIKAE_STATS").is_some() {
                eprintln!(
                    "kirikae: {} recompiled function calls, {} interpreted instructions, exit code {}",
                    r.native_calls, r.interpreted, r.exit_code
                );
            }
            std::process::exit(r.exit_code)
        }
        Err(e) => {
            eprintln!("kirikae: {e}");
            std::process::exit(3)
        }
    }
}
