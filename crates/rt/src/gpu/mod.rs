use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::mem::GuestMemory;

pub mod astc;
pub mod buffer_cache;
pub mod compute;
pub mod formats;
pub mod fxhash;
pub mod mme;
pub mod threed;
pub mod tic;
pub mod twod;
pub mod vulkan;

pub const SYNCPOINTS: usize = 192;

#[derive(Clone, Copy, Debug, Default)]
pub struct NvMap {
    pub size: u64,
    pub align: u32,
    pub kind: u8,
    pub addr: u64,
}

#[derive(Clone, Copy, Debug)]
struct Mapping {
    cpu: u64,
    size: u64,
}

static MAPS_CHANGED: AtomicU64 = AtomicU64::new(0);

type Found = (u64, usize, u64, Mapping);

fn mapping_slot(va: u64) -> usize {
    ((va >> 12).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 55) as usize
}

thread_local! {
    static LAST_MAPPINGS: std::cell::RefCell<[Option<Found>; 512]> = const { std::cell::RefCell::new([None; 512]) };
}

struct Engines {
    classes: [u32; 8],
    regs: fxhash::FastMap<u32, Vec<u32>>,
    inline: Option<(u64, u64, Vec<u8>)>,
    cb_upload: (u64, Vec<u8>),
    threed: threed::State,
}

pub const CLASS_3D: u32 = 0xb197;
pub const CLASS_COMPUTE: u32 = 0xb1c0;
pub const CLASS_INLINE: u32 = 0xa140;
pub const CLASS_DMA: u32 = 0xb0b5;
pub const CLASS_2D: u32 = 0x902d;

#[derive(Default)]
pub struct Stats {
    pub submits: AtomicU64,
    pub draws: AtomicU64,
    pub macros: AtomicU64,
    pub clears: AtomicU64,
    pub dispatches: AtomicU64,
    pub copies: AtomicU64,
}

pub struct Gpu {
    nvmaps: Mutex<HashMap<u32, NvMap>>,
    next_handle: AtomicU32,
    maps: Mutex<BTreeMap<u64, Mapping>>,
    next_va: Mutex<u64>,
    reserved: Mutex<(u64, u64)>,
    pub syncpoints: [AtomicU32; SYNCPOINTS],
    next_syncpoint: AtomicU32,
    engines: Mutex<Engines>,
    renderer: Mutex<(Option<vulkan::Renderer>, bool)>,
    written: Mutex<Vec<(u64, u64)>>,
    gpu_writes: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub stats: Stats,
}

impl Default for Gpu {
    fn default() -> Self {
        Gpu {
            nvmaps: Mutex::new(HashMap::new()),
            next_handle: AtomicU32::new(1),
            maps: Mutex::new(BTreeMap::new()),
            next_va: Mutex::new(0x1_0000_0000),
            reserved: Mutex::new((0, 0)),
            syncpoints: std::array::from_fn(|_| AtomicU32::new(0)),
            next_syncpoint: AtomicU32::new(16),
            engines: Mutex::new(Engines { classes: [0; 8], regs: fxhash::FastMap::default(), inline: None, cb_upload: (0, Vec::new()), threed: Default::default() }),
            renderer: Mutex::new((None, false)),
            written: Mutex::new(Vec::new()),
            gpu_writes: Default::default(),
            stats: Stats::default(),
        }
    }
}

const BIG_PAGE: u64 = 0x2_0000;

impl Gpu {
    pub fn nvmap_create(&self, size: u64) -> u32 {
        let h = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.nvmaps.lock().unwrap().insert(h, NvMap { size, ..NvMap::default() });
        h
    }

    pub fn nvmap(&self, handle: u32) -> Option<NvMap> {
        self.nvmaps.lock().unwrap().get(&handle).copied()
    }

    pub fn nvmap_alloc(&self, handle: u32, addr: u64, align: u32, kind: u8) -> bool {
        match self.nvmaps.lock().unwrap().get_mut(&handle) {
            Some(m) => {
                m.addr = addr;
                m.align = align.max(0x1000);
                m.kind = kind;
                true
            }
            None => false,
        }
    }

    pub fn nvmap_free(&self, handle: u32) -> Option<NvMap> {
        self.nvmaps.lock().unwrap().remove(&handle)
    }

    pub fn reserve_va(&self, size: u64, align: u64, fixed: Option<u64>) -> u64 {
        let va = self.alloc_va(size, align, fixed);
        let mut r = self.reserved.lock().unwrap();
        *r = (r.0 + 1, va);
        va
    }

    pub fn last_reserved(&self) -> (u64, u64) {
        *self.reserved.lock().unwrap()
    }

    pub fn alloc_va(&self, size: u64, align: u64, fixed: Option<u64>) -> u64 {
        if let Some(va) = fixed {
            return va;
        }
        let mut next = self.next_va.lock().unwrap();
        let align = align.max(BIG_PAGE);
        let va = next.div_ceil(align) * align;
        *next = va + size.div_ceil(BIG_PAGE) * BIG_PAGE;
        va
    }

    pub fn map(&self, va: u64, cpu: u64, size: u64) {
        let mut maps = self.maps.lock().unwrap();
        MAPS_CHANGED.fetch_add(1, Ordering::Relaxed);
        let end = va + size;
        let first = maps.range(..va).next_back().filter(|(s, m)| *s + m.size > va).map(|(s, _)| *s).unwrap_or(va);
        let overlapping: Vec<(u64, Mapping)> = maps.range(first..end).map(|(s, m)| (*s, *m)).collect();
        for (start, m) in overlapping {
            maps.remove(&start);
            if start < va {
                maps.insert(start, Mapping { cpu: m.cpu, size: va - start });
            }
            let m_end = start + m.size;
            if m_end > end {
                maps.insert(end, Mapping { cpu: m.cpu + (end - start), size: m_end - end });
            }
        }
        maps.insert(va, Mapping { cpu, size });
    }

    pub fn unmap(&self, va: u64) {
        let mut maps = self.maps.lock().unwrap();
        MAPS_CHANGED.fetch_add(1, Ordering::Relaxed);
        maps.remove(&va);
    }

    pub fn translate(&self, va: u64) -> Option<u64> {
        let (changes, me) = (MAPS_CHANGED.load(Ordering::Acquire), self as *const Self as usize);
        let slot = mapping_slot(va);
        let hit = LAST_MAPPINGS.with_borrow(|last| {
            last[slot].filter(|&(n, gpu, start, m)| (n, gpu) == (changes, me) && (start..start + m.size).contains(&va)).map(|(_, _, start, m)| m.cpu + (va - start))
        });
        if hit.is_some() {
            return hit;
        }
        let maps = self.maps.lock().unwrap();
        let (&start, &m) = maps.range(..=va).next_back()?;
        if va >= start + m.size {
            return None;
        }
        let entry = (MAPS_CHANGED.load(Ordering::Acquire), me, start, m);
        LAST_MAPPINGS.with_borrow_mut(|last| last[slot] = Some(entry));
        Some(m.cpu + (va - start))
    }

    fn pieces(&self, va: u64, len: u64) -> Option<Vec<(u64, u64)>> {
        let maps = self.maps.lock().unwrap();
        let mut out: Vec<(u64, u64)> = Vec::new();
        let mut at = va;
        while at < va + len {
            let (&start, &m) = maps.range(..=at).next_back()?;
            if at >= start + m.size {
                return None;
            }
            let n = (start + m.size - at).min(va + len - at);
            match out.last_mut() {
                Some((cpu, size)) if *cpu + *size == m.cpu + (at - start) => *size += n,
                _ => out.push((m.cpu + (at - start), n)),
            }
            at += n;
        }
        Some(out)
    }

    pub fn gpu_addresses_of(&self, cpu: u64) -> Vec<u64> {
        let maps = self.maps.lock().unwrap();
        maps.iter().filter(|(_, m)| m.cpu == cpu).map(|(va, _)| *va).collect()
    }

    pub fn gpu_mappings_of(&self, cpu: u64) -> Vec<(u64, u64, u64, u64)> {
        let maps = self.maps.lock().unwrap();
        maps.iter().filter(|(_, m)| (m.cpu..m.cpu + m.size).contains(&cpu)).map(|(va, m)| (va + (cpu - m.cpu), *va, m.cpu, m.size)).collect()
    }

    fn read(&self, mem: &GuestMemory, va: u64, len: u64) -> Option<Vec<u8>> {
        mem.read(self.translate(va)?, len)
    }

    fn bytes<'m>(&self, mem: &'m GuestMemory, va: u64, len: u64) -> Option<&'m [u8]> {
        mem.bytes(self.translate(va)?, len)
    }

    fn write(&self, mem: &GuestMemory, va: u64, bytes: &[u8]) -> bool {
        self.cpu_access(va, bytes.len() as u64);
        let Some(pieces) = self.pieces(va, bytes.len() as u64) else { return false };
        let mut done = 0;
        for (cpu, n) in pieces {
            if let Some(r) = self.renderer.lock().unwrap().0.as_mut() {
                r.note_write(cpu, n);
            }
            if !mem.write(cpu, &bytes[done..done + n as usize]) {
                return false;
            }
            done += n as usize;
        }
        true
    }

    fn cpu_access(&self, va: u64, len: u64) {
        if self.gpu_writes.load(Ordering::Relaxed)
            && let Some(cpu) = self.translate(va)
        {
            self.with_renderer(|r| r.resolve(cpu, len));
        }
    }

    pub fn wait_idle(&self) {
        if self.gpu_writes.load(Ordering::Relaxed) {
            self.with_renderer(|r| r.finish());
        }
    }

    pub fn alloc_syncpoint(&self) -> u32 {
        self.next_syncpoint.fetch_add(1, Ordering::Relaxed) % SYNCPOINTS as u32
    }

    pub fn syncpoint(&self, id: u32) -> u32 {
        self.syncpoints[id as usize % SYNCPOINTS].load(Ordering::Relaxed)
    }

    pub fn increment(&self, id: u32, by: u32) -> u32 {
        self.syncpoints[id as usize % SYNCPOINTS].fetch_add(by, Ordering::Relaxed) + by
    }

    pub fn submit(&self, mem: &GuestMemory, entries: &[u64]) -> Result<(), String> {
        self.stats.submits.fetch_add(1, Ordering::Relaxed);
        self.forget_written();
        for &e in entries {
            let va = (e & 0xffff_fffc) | ((e >> 32) & 0xff) << 32;
            let words = (e >> 42) & 0x1f_ffff;
            if words == 0 {
                continue;
            }
            self.cpu_access(va, words * 4);
            let plan = if crate::nvn::shadow::ON.load(Ordering::Relaxed) { crate::nvn::shadow::cp_entry(va, words) } else { None };
            if let Some(p) = plan.as_ref().filter(|_| crate::nvn::exec::ON.load(Ordering::Relaxed))
                && self.execute(mem, p)
            {
                continue;
            }
            let simulated = plan.as_ref().filter(|_| crate::nvn::exec::CHECK.load(Ordering::Relaxed)).and_then(|p| self.simulate(mem, p));
            let before = (self.stats.draws.load(Ordering::Relaxed), self.stats.clears.load(Ordering::Relaxed));
            let bytes = self.bytes(mem, va, words * 4).ok_or_else(|| format!("pushbuffer at GPU address {va:#x} is not mapped"))?;
            let w: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
            self.run_pushbuffer(mem, &w);
            if crate::nvn::exec::CHECK.load(Ordering::Relaxed) {
                crate::nvn::exec::segment_done();
            }
            if let Some((regs, cbs, writes)) = simulated {
                let eng = self.engines.lock().unwrap();
                if let Some(actual) = eng.regs.get(&CLASS_3D) {
                    crate::nvn::exec::checked(&regs, actual, &cbs, &eng.threed.constant_buffers);
                }
                drop(eng);
                let differ = writes.iter().filter(|(at, w)| self.read(mem, **at, 4).is_none_or(|b| u32::from_le_bytes(b[..4].try_into().unwrap()) != **w)).count();
                crate::nvn::exec::writes_checked(writes.len(), differ);
            }
            if let Some(plan) = plan {
                let (draws, clears) = (self.stats.draws.load(Ordering::Relaxed) - before.0, self.stats.clears.load(Ordering::Relaxed) - before.1);
                crate::nvn::shadow::cp_entry_done(plan, draws, clears);
            }
        }
        let mut r = self.renderer.lock().unwrap();
        if let Some(r) = r.0.as_mut() {
            r.flush();
        }
        Ok(())
    }

    pub fn with_renderer<T>(&self, f: impl FnOnce(&mut vulkan::Renderer) -> T) -> Option<T> {
        let mut guard = self.renderer.lock().unwrap();
        if !guard.1 {
            guard.1 = true;
            if std::env::var_os("KIRIKAE_NO_VULKAN").is_none() {
                match vulkan::Renderer::new() {
                    Ok(mut r) => {
                        r.pending_flag = self.gpu_writes.clone();
                        eprintln!("kirikae: rendering with Vulkan on {}", r.name);
                        guard.0 = Some(r);
                    }
                    Err(e) => eprintln!("kirikae: Vulkan is not available ({e}); draws are not rendered"),
                }
            }
        }
        guard.0.as_mut().map(f)
    }

    pub fn guest_memory_written(&self, cpu: u64, len: u64) {
        let mut written = self.written.lock().unwrap();
        if !written.contains(&(cpu, len)) {
            written.push((cpu, len));
        }
    }

    fn written_behind(&self, va: u64, len: u64) {
        if let Some(cpu) = self.translate(va) {
            self.with_renderer(|r| r.forget_guest_range(cpu, len));
        }
    }

    fn forget_written(&self) {
        let written = std::mem::take(&mut *self.written.lock().unwrap());
        if !written.is_empty()
            && let Some(r) = self.renderer.lock().unwrap().0.as_mut()
        {
            for (cpu, len) in written {
                r.forget_guest_range(cpu, len);
            }
        }
    }

    pub fn read_pixels(&self, cpu: u64, width: u32, height: u32) -> Option<Vec<u32>> {
        let mut guard = self.renderer.lock().unwrap();
        guard.0.as_mut()?.read_pixels(cpu, width, height)
    }

    pub fn present(&self, cpu: u64, width: u32, height: u32, flip_h: bool, flip_v: bool) -> vulkan::Presented {
        let mut guard = self.renderer.lock().unwrap();
        guard.0.as_mut().map_or(vulkan::Presented::NoSwapchain, |r| r.present(cpu, width, height, flip_h, flip_v))
    }

    pub fn present_pixels(&self, width: u32, height: u32, pixels: &[u32]) -> vulkan::Presented {
        let mut guard = self.renderer.lock().unwrap();
        guard.0.as_mut().map_or(vulkan::Presented::NoSwapchain, |r| r.present_pixels(width, height, pixels))
    }

    pub fn rendered(&self, cpu: u64, width: u32, height: u32) -> bool {
        let guard = self.renderer.lock().unwrap();
        guard.0.as_ref().is_some_and(|r| r.rendered(cpu, width, height))
    }

    pub fn dump_images(&self, dir: &std::path::Path, tag: u64) {
        if let Some(r) = self.renderer.lock().unwrap().0.as_mut() {
            r.dump_images(dir, tag);
        }
    }

    pub fn render_stats(&self) -> Option<vulkan::RenderStats> {
        self.renderer.lock().unwrap().0.as_ref().map(|r| r.stats.clone())
    }

    fn run_pushbuffer(&self, mem: &GuestMemory, w: &[u32]) {
        let mut guard = self.engines.lock().unwrap();
        let eng = &mut *guard;
        let mut i = 0;
        while i < w.len() {
            let hdr = w[i];
            i += 1;
            let kind = hdr >> 29;
            let count = (hdr >> 16) & 0x1fff;
            let sub = (hdr >> 13) & 7;
            let method = hdr & 0xfff;
            match kind {
                1 | 3 | 5 => {
                    for k in 0..count as usize {
                        let Some(&v) = w.get(i + k) else { break };
                        let m = match kind {
                            1 => method + k as u32,
                            3 => method,
                            _ => method + (k > 0) as u32,
                        };
                        self.method(mem, eng, sub, m, v);
                    }
                    i += count as usize;
                }
                4 => self.method(mem, eng, sub, method, count),
                _ => {}
            }
        }
        if let Some(p) = eng.threed.macros.pending.take() {
            self.run_macro(mem, eng, p);
        }
        self.flush_cb_upload(mem, eng);
    }

    #[allow(clippy::type_complexity)]
    fn simulate(&self, mem: &GuestMemory, plan: &crate::nvn::shadow::Plan) -> Option<(Vec<u32>, [[Option<(u64, u32)>; threed::CB_SLOTS]; 5], std::collections::BTreeMap<u64, u32>)> {
        use crate::nvn::exec::Step;
        let eng = self.engines.lock().unwrap();
        let mut regs = eng.regs.get(&CLASS_3D)?.clone();
        let mut cbs = eng.threed.constant_buffers;
        let steps = crate::nvn::shadow::exec_steps(plan, &regs, &|va, len| self.read(mem, va, len))?;
        let mut writes = std::collections::BTreeMap::new();
        for step in steps {
            match step {
                Step::Regs(values) => {
                    for (r, mask, v) in values {
                        regs[r] = regs[r] & !mask | v;
                    }
                }
                Step::Bind { group, slot, binding } => cbs[group][slot] = binding,
                Step::Draw(d) => crate::nvn::exec::expect_draw(regs.clone(), cbs, d),
                Step::Clear(v) => crate::nvn::exec::expect_draw(regs.clone(), [[None; threed::CB_SLOTS]; 5], threed::Draw { topology: crate::nvn::exec::CLEAR, first: v, count: 0, index_size: None, instance: 0 }),
                Step::Write { at, words } => {
                    for (i, w) in words.into_iter().enumerate() {
                        writes.insert(at + 4 * i as u64, w);
                    }
                }
            }
        }
        Some((regs, cbs, writes))
    }

    fn execute(&self, mem: &GuestMemory, plan: &crate::nvn::shadow::Plan) -> bool {
        use crate::nvn::exec::Step;
        let mut guard = self.engines.lock().unwrap();
        let eng = &mut *guard;
        let Some(regs) = eng.regs.get(&CLASS_3D) else { return false };
        let Some(steps) = crate::nvn::shadow::exec_steps(plan, regs, &|va, len| self.read(mem, va, len)) else { return false };
        for step in steps {
            match step {
                Step::Regs(values) => {
                    let regs = eng.regs.get_mut(&CLASS_3D).expect("3D registers");
                    for (r, mask, v) in values {
                        regs[r] = regs[r] & !mask | v;
                    }
                }
                Step::Bind { group, slot, binding } => eng.threed.constant_buffers[group][slot] = binding,
                Step::Write { at, words } => {
                    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                    self.write(mem, at, &bytes);
                }
                Step::Draw(d) => {
                    let cbs = eng.threed.constant_buffers;
                    self.draw(mem, &eng.regs[&CLASS_3D], &cbs, d);
                }
                Step::Clear(v) => self.clear(mem, &eng.regs[&CLASS_3D], v),
            }
        }
        true
    }

    fn flush_cb_upload(&self, mem: &GuestMemory, eng: &mut Engines) {
        if !eng.cb_upload.1.is_empty() {
            if crate::nvn::shadow::ON.load(Ordering::Relaxed) {
                crate::nvn::shadow::cp_upload(eng.cb_upload.0, &eng.cb_upload.1);
            }
            self.write(mem, eng.cb_upload.0, &eng.cb_upload.1);
            eng.cb_upload.1.clear();
        }
    }

    fn method(&self, mem: &GuestMemory, eng: &mut Engines, sub: u32, m: u32, v: u32) {
        if m == 0 {
            eng.classes[sub as usize] = v & 0xffff;
            return;
        }
        if m >= threed::MACRO_CALL && eng.classes[sub as usize] == CLASS_3D {
            let index = (m - threed::MACRO_CALL) / 2;
            if m % 2 == 1
                && let Some(p) = eng.threed.macros.pending.as_mut().filter(|p| p.index == index)
            {
                p.params.push(v);
                return;
            }
            let mut params = std::mem::take(&mut eng.threed.macros.spare);
            params.clear();
            params.push(v);
            let previous = eng.threed.macros.pending.replace(threed::PendingMacro { sub, index, params });
            if let Some(p) = previous {
                self.run_macro(mem, eng, p);
            }
            return;
        }
        if let Some(p) = eng.threed.macros.pending.take() {
            self.run_macro(mem, eng, p);
        }
        self.method_direct(mem, eng, sub, m, v);
    }

    fn run_macro(&self, mem: &GuestMemory, eng: &mut Engines, call: threed::PendingMacro) {
        let (code, start) = (eng.threed.macros.code.clone(), eng.threed.macros.starts.get(call.index as usize).copied());
        self.stats.macros.fetch_add(1, Ordering::Relaxed);
        struct Host<'a> {
            gpu: &'a Gpu,
            mem: &'a GuestMemory,
            eng: &'a mut Engines,
            sub: u32,
        }
        impl mme::Host for Host<'_> {
            fn write(&mut self, method: u32, value: u32) {
                if method < threed::MACRO_CALL {
                    self.gpu.method_direct(self.mem, self.eng, self.sub, method, value);
                }
            }
            fn read(&mut self, method: u32) -> u32 {
                self.eng.regs.get(&CLASS_3D).and_then(|r| r.get(method as usize)).copied().unwrap_or(0)
            }
        }
        if let Some(start) = start {
            let mut host = Host { gpu: self, mem, eng, sub: call.sub };
            if let Err(e) = mme::run(&code, start as usize, &call.params, &mut host)
                && trace()
            {
                eprintln!("gpu: macro {} failed: {e:?} (params {:x?})", call.index, &call.params[..call.params.len().min(16)]);
                if let Some(dir) = std::env::var_os("KIRIKAE_SHADER_DUMP") {
                    let bytes: Vec<u8> = code[start as usize..].iter().take(0x200).flat_map(|w| w.to_le_bytes()).collect();
                    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("macro{}.bin", call.index)), bytes);
                }
            }
        }
        eng.threed.macros.spare = call.params;
    }

    fn method_direct(&self, mem: &GuestMemory, eng: &mut Engines, sub: u32, m: u32, v: u32) {
        let class = eng.classes[sub as usize];
        if !eng.cb_upload.1.is_empty() && !(class == CLASS_3D && threed::reg::LOAD_CONSTANT_BUFFER.contains(&m)) {
            self.flush_cb_upload(mem, eng);
        }
        if m < 0x40 {
            let regs = eng.regs.entry(0).or_insert_with(|| vec![0; 0x40]);
            regs[m as usize] = v;
            match m {
                0x07 => {
                    let addr = (regs[0x04] as u64 & 0xff) << 32 | regs[0x05] as u64;
                    let payload = regs[0x06];
                    if v & 7 == 2 {
                        self.write(mem, addr, &payload.to_le_bytes());
                    }
                }
                0x1d if v & 1 == 1 => {
                    self.increment(v >> 8 & 0xff, 1);
                }
                _ => {}
            }
            return;
        }
        let regs = eng.regs.entry(class).or_insert_with(|| vec![0; 0x1000]);
        if let Some(r) = regs.get_mut(m as usize) {
            *r = v;
        }
        if matches!(class, CLASS_3D | CLASS_COMPUTE | CLASS_INLINE) {
            match m {
                0x6d => {
                    if let Some((_, len, data)) = eng.inline.as_mut() {
                        data.extend_from_slice(&v.to_le_bytes());
                        if data.len() as u64 >= *len {
                            let (dst, len, data) = eng.inline.take().unwrap();
                            self.write(mem, dst, &data[..len as usize]);
                        }
                    }
                    return;
                }
                0x6c => {
                    let dst = (regs[0x62] as u64 & 0xff) << 32 | regs[0x63] as u64;
                    let len = regs[0x60] as u64 * regs[0x61].max(1) as u64;
                    eng.inline = Some((dst, len, Vec::new()));
                    return;
                }
                _ => {}
            }
        }
        match (class, m) {
            (CLASS_3D, _) => self.threed(mem, eng, m, v),
            (CLASS_2D, twod::PIXELS_FROM_MEMORY_SRC_Y0_INT) => self.blit_2d(mem, regs),
            (CLASS_COMPUTE, 0x6c3) => self.report(mem, regs),
            (CLASS_COMPUTE, compute::reg::LAUNCH) => self.dispatch(mem, regs),
            (CLASS_DMA, 0xc0) => self.dma(mem, regs, v),
            _ => {}
        }
    }

    fn dma(&self, mem: &GuestMemory, regs: &[u32], launch: u32) {
        self.stats.copies.fetch_add(1, Ordering::Relaxed);
        let src = (regs[0x100] as u64 & 0xff) << 32 | regs[0x101] as u64;
        let dst = (regs[0x102] as u64 & 0xff) << 32 | regs[0x103] as u64;
        let (pitch_in, pitch_out) = (regs[0x104] as u64, regs[0x105] as u64);
        let (len, lines) = (regs[0x106] as u64, regs[0x107].max(1) as u64);
        let multi_line = launch >> 9 & 1 == 1;
        let lines = if multi_line { lines } else { 1 };
        let remap = launch >> 10 & 1 == 1;
        let src_pitch_layout = launch >> 7 & 1 == 1;
        let dst_pitch_layout = launch >> 8 & 1 == 1;
        if !(src_pitch_layout && dst_pitch_layout) {
            if trace() {
                eprintln!("gpu: dma block-linear launch {launch:#x} src {src:#x} dst {dst:#x} pitch {pitch_in:#x}/{pitch_out:#x} len {len:#x} lines {lines} regs {:x?}", &regs[0x1c0..0x1d0]);
            }
            if remap {
                return;
            }
            let side = |pitch_layout: bool, base: u64, at: usize, pitch: u64| {
                if pitch_layout {
                    Side::Pitch { base, pitch: if multi_line { pitch } else { 0 } }
                } else {
                    let origin = regs[at + 5];
                    Side::BlockLinear { base, block_height_log2: regs[at] >> 4 & 0xf, width: regs[at + 1] as u64, x: (origin & 0xffff) as u64, y: (origin >> 16) as u64 }
                }
            };
            let from = side(src_pitch_layout, src, 0x1ca, pitch_in);
            let to = side(dst_pitch_layout, dst, 0x1c3, pitch_out);
            let Some(rows) = from.read(self, mem, len, lines) else { return };
            to.write(self, mem, &rows, len, lines);
            return;
        }
        for l in 0..lines {
            let out = dst + l * if multi_line { pitch_out } else { 0 };
            if remap {
                let c = regs[0x1c0].to_le_bytes();
                let fill: Vec<u8> = c.iter().copied().cycle().take((len * 4) as usize).collect();
                self.write(mem, out, &fill);
            } else if let Some(line) = {
                let from = src + l * if multi_line { pitch_in } else { 0 };
                self.cpu_access(from, len);
                self.read(mem, from, len)
            } {
                self.write(mem, out, &line);
            }
        }
    }
}

fn runs(x: u64, len: u64) -> impl Iterator<Item = (u64, usize)> {
    let mut i = 0;
    std::iter::from_fn(move || {
        (i < len).then(|| {
            let n = (16 - (x + i) % 16).min(len - i);
            let at = i;
            i += n;
            (at, n as usize)
        })
    })
}

enum Side {
    Pitch { base: u64, pitch: u64 },
    BlockLinear { base: u64, block_height_log2: u32, width: u64, x: u64, y: u64 },
}

impl Side {
    fn block_rows(block_height_log2: u32, width: u64, y: u64, lines: u64) -> (u64, u64) {
        let rows = 8u64 << block_height_log2;
        let per_block_row = width.div_ceil(64) * (1 << block_height_log2) * 512;
        let (first, last) = (y / rows, (y + lines).div_ceil(rows));
        (first * per_block_row, (last - first) * per_block_row)
    }

    fn read(&self, gpu: &Gpu, mem: &GuestMemory, len: u64, lines: u64) -> Option<Vec<u8>> {
        match *self {
            Side::Pitch { base, pitch } => {
                let mut out = Vec::with_capacity((len * lines) as usize);
                for l in 0..lines {
                    gpu.cpu_access(base + l * pitch, len);
                    out.extend_from_slice(gpu.bytes(mem, base + l * pitch, len)?);
                }
                Some(out)
            }
            Side::BlockLinear { base, block_height_log2, width, x, y } => {
                let (start, size) = Self::block_rows(block_height_log2, width, y, lines);
                gpu.cpu_access(base + start, size);
                let surface = gpu.bytes(mem, base + start, size)?;
                let mut out = vec![0u8; (len * lines) as usize];
                for l in 0..lines {
                    for (i, n) in runs(x, len) {
                        let o = (block_linear_offset(x + i, y + l, width, block_height_log2) - start) as usize;
                        let d = (l * len + i) as usize;
                        if let Some(src) = surface.get(o..o + n) {
                            out[d..d + n].copy_from_slice(src);
                        }
                    }
                }
                Some(out)
            }
        }
    }

    fn write(&self, gpu: &Gpu, mem: &GuestMemory, rows: &[u8], len: u64, lines: u64) {
        match *self {
            Side::Pitch { base, pitch } => {
                for (l, row) in rows.chunks(len as usize).enumerate() {
                    gpu.write(mem, base + l as u64 * pitch, row);
                }
                gpu.written_behind(base, pitch * lines.saturating_sub(1) + len);
            }
            Side::BlockLinear { base, block_height_log2, width, x, y } => {
                let (start, size) = Self::block_rows(block_height_log2, width, y, lines);
                gpu.cpu_access(base + start, size);
                let Some(mut surface) = gpu.read(mem, base + start, size) else { return };
                for l in 0..lines {
                    for (i, n) in runs(x, len) {
                        let o = (block_linear_offset(x + i, y + l, width, block_height_log2) - start) as usize;
                        let d = (l * len + i) as usize;
                        if let Some(dst) = surface.get_mut(o..o + n) {
                            dst.copy_from_slice(&rows[d..d + n]);
                        }
                    }
                }
                gpu.write(mem, base + start, &surface);
                gpu.written_behind(base + start, size);
            }
        }
    }
}

pub(crate) fn trace() -> bool {
    static RANGE: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    let range = *RANGE.get_or_init(|| {
        std::env::var_os("KIRIKAE_TRACE_GPU")?;
        let num = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok());
        let from = num("KIRIKAE_TRACE_GPU_FROM").unwrap_or(0);
        Some((from, num("KIRIKAE_TRACE_GPU_COUNT").map_or(u64::MAX, |n| from.saturating_add(n))))
    });
    range.is_some_and(|(from, to)| (from..to).contains(&crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed)))
}

pub fn report_ticks() -> u64 {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let forced = *MODE.get_or_init(|| match std::env::var("KIRIKAE_GPU_TIME").as_deref() {
        Ok("frame") => Some(true),
        Ok("live") => Some(false),
        _ => None,
    });
    if forced.unwrap_or_else(|| FROZEN.load(std::sync::atomic::Ordering::Relaxed)) {
        FRAME_TICKS.load(std::sync::atomic::Ordering::Relaxed)
    } else {
        gpu_ticks()
    }
}

static FROZEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FRAME_TICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn frame_presented() {
    FRAME_TICKS.store(gpu_ticks(), std::sync::atomic::Ordering::Relaxed);
}

pub fn freeze_frame_time() {
    FROZEN.store(true, std::sync::atomic::Ordering::Relaxed);
}

pub fn gpu_ticks() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

pub fn block_linear_offset(x: u64, y: u64, width_bytes: u64, block_height_log2: u32) -> u64 {
    let gobs_high = 1u64 << block_height_log2;
    let gobs_wide = width_bytes.div_ceil(64);
    let block_row = y / (8 * gobs_high);
    let gob_in_block = (y % (8 * gobs_high)) / 8;
    let (gx, gy) = (x % 64, y % 8);
    let inner = (gx / 32) * 256 + (gy / 2) * 64 + ((gx % 32) / 16) * 32 + (gy % 2) * 16 + (gx % 16);
    block_row * gobs_wide * gobs_high * 512 + (x / 64) * gobs_high * 512 + gob_in_block * 512 + inner
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_linear_runs_are_contiguous() {
        for x in [0u64, 5, 16, 30] {
            for (i, n) in runs(x, 100) {
                let o = block_linear_offset(x + i, 3, 256, 2);
                for k in 0..n as u64 {
                    assert_eq!(block_linear_offset(x + i + k, 3, 256, 2), o + k);
                }
            }
            assert_eq!(runs(x, 100).map(|r| r.1).sum::<usize>(), 100);
        }
    }

    #[test]
    fn gob_layout() {
        assert_eq!(block_linear_offset(0, 0, 64, 0), 0);
        assert_eq!(block_linear_offset(15, 0, 64, 0), 15);
        assert_eq!(block_linear_offset(0, 1, 64, 0), 16);
        assert_eq!(block_linear_offset(16, 0, 64, 0), 32);
        assert_eq!(block_linear_offset(32, 0, 64, 0), 256);
        assert_eq!(block_linear_offset(0, 8, 64, 1), 512);
        let mut seen = std::collections::HashSet::new();
        for y in 0..16 {
            for x in 0..128 {
                assert!(seen.insert(block_linear_offset(x, y, 128, 1)));
            }
        }
        assert_eq!(*seen.iter().max().unwrap(), 128 * 16 - 1);
    }

    #[test]
    fn pushbuffer_semaphore_release() {
        let mem = GuestMemory::new().unwrap();
        mem.map(0x1000_0000, 0x1000).unwrap();
        let gpu = Gpu::default();
        gpu.map(0x5_0000_0000, 0x1000_0000, 0x1000);
        let pb = [0x2004_0004u32, 0x5, 0x0000_0100, 0xcafe, 0x2];
        gpu.run_pushbuffer(&mem, &pb);
        assert_eq!(mem.read(0x1000_0100, 4).unwrap(), 0xcafeu32.to_le_bytes());
    }

    #[test]
    fn remap_inside_a_mapping() {
        let gpu = Gpu::default();
        gpu.map(0x10_0000, 0x5000_0000, 0x10_0000);
        gpu.map(0x14_0000, 0x7000_0000, 0x1_0000);
        assert_eq!(gpu.translate(0x10_0010), Some(0x5000_0010));
        assert_eq!(gpu.translate(0x14_0010), Some(0x7000_0010));
        assert_eq!(gpu.translate(0x15_0000), Some(0x5005_0000));
        assert_eq!(gpu.translate(0x1f_ffff), Some(0x500f_ffff));
        assert_eq!(gpu.translate(0x20_0000), None);
    }

    #[test]
    fn writes_split_at_mappings() {
        let mem = GuestMemory::new().unwrap();
        mem.map(0x1000_0000, 0x2000).unwrap();
        mem.map(0x2000_0000, 0x1000).unwrap();
        let gpu = Gpu::default();
        gpu.map(0x5_0000_0000, 0x1000_0000, 0x1000);
        gpu.map(0x5_0000_1000, 0x2000_0000, 0x1000);
        assert!(gpu.write(&mem, 0x5_0000_0ff0, &[0xaa; 0x20]));
        assert_eq!(mem.read(0x1000_0ff0, 0x10).unwrap(), [0xaa; 0x10]);
        assert_eq!(mem.read(0x2000_0000, 0x10).unwrap(), [0xaa; 0x10]);
        assert_eq!(mem.read(0x1000_1000, 0x10).unwrap(), [0; 0x10]);
        assert!(!gpu.write(&mem, 0x5_0000_1ff0, &[0xaa; 0x20]));
    }
}
