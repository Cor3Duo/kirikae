pub mod h264;
pub mod nvdec;
pub mod vic;
pub mod vp9;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

use crate::gpu::{block_linear_offset, Gpu, NvMap};
use crate::mem::GuestMemory;
use crate::process::Process;

pub const CLASS_HOST1X: u32 = 0x01;
pub const CLASS_VIC: u32 = 0x5d;
pub const CLASS_NVDEC: u32 = 0xf0;

const THI_INCR_SYNCPT: u32 = 0x00;
const THI_METHOD0: u32 = 0x10;
const THI_METHOD1: u32 = 0x11;

const IOVA_BASE: u64 = 0x10_0000;
const IOVA_PAGE: u64 = 0x1000;

pub(crate) fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KIRIKAE_TRACE_VIDEO").is_some())
}

#[derive(Default)]
struct Iova {
    handles: HashMap<u32, u64>,
    maps: BTreeMap<u64, (u64, u64)>,
    next: u64,
}

impl Iova {
    fn map(&mut self, handle: u32, m: NvMap) -> Option<u32> {
        if let Some(&iova) = self.handles.get(&handle) {
            let size = m.size.div_ceil(IOVA_PAGE) * IOVA_PAGE;
            if self.maps.get(&iova).is_some_and(|&(_, s)| s >= size) {
                self.maps.insert(iova, (m.addr, size));
                return Some(iova as u32);
            }
        }
        let size = m.size.div_ceil(IOVA_PAGE).max(1) * IOVA_PAGE;
        let iova = self.next.max(IOVA_BASE);
        if iova + size > u32::MAX as u64 {
            return None;
        }
        self.next = iova + size;
        self.handles.insert(handle, iova);
        self.maps.insert(iova, (m.addr, size));
        Some(iova as u32)
    }

    fn translate(&self, iova: u64, len: u64) -> Option<u64> {
        let (&start, &(cpu, size)) = self.maps.range(..=iova).next_back()?;
        (cpu != 0 && iova + len <= start + size).then(|| cpu + (iova - start))
    }
}

pub struct Dma<'a> {
    iova: &'a Iova,
    mem: &'a GuestMemory,
    gpu: &'a Gpu,
    later: Option<Later<'a>>,
}

type Later<'a> = (&'static Process, &'a Arc<Mutex<Iova>>, &'a Arc<Worker>);

impl Dma<'_> {
    pub fn defer(&self, job: impl FnOnce(&Dma) -> Result<(), String> + Send + 'static) -> Result<(), String> {
        static ASYNC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let deferred = *ASYNC.get_or_init(|| std::env::var_os("KIRIKAE_NVDEC_ASYNC").is_some());
        match self.later {
            Some((p, iova, worker)) if deferred => {
                let iova = iova.clone();
                worker.push(p, Box::new(move || {
                    let iova = iova.lock().unwrap();
                    job(&Dma { iova: &iova, mem: &p.memory, gpu: &p.gpu, later: None })
                }));
                Ok(())
            }
            _ => job(self),
        }
    }

    pub fn read(&self, offset: u32, len: usize) -> Option<Vec<u8>> {
        let cpu = self.iova.translate((offset as u64) << 8, len as u64)?;
        self.mem.read(cpu, len as u64)
    }

    pub fn write(&self, offset: u32, bytes: &[u8]) -> bool {
        let Some(cpu) = self.iova.translate((offset as u64) << 8, bytes.len() as u64) else { return false };
        let ok = self.mem.write(cpu, bytes);
        self.gpu.guest_memory_written(cpu, bytes.len() as u64);
        ok
    }
}

#[derive(Default)]
struct Engines {
    nvdec: nvdec::Nvdec,
    vic: vic::Vic,
    nvdec_method: u32,
    vic_method: u32,
}

type Job = Box<dyn FnOnce() -> Result<(), String> + Send>;

#[derive(Default)]
pub struct Worker {
    state: Mutex<(VecDeque<Job>, bool, bool)>,
    changed: Condvar,
}

impl Worker {
    fn push(self: &Arc<Self>, p: &'static Process, job: Job) {
        let mut st = self.state.lock().unwrap();
        st.0.push_back(job);
        if !st.2 {
            st.2 = true;
            let w = self.clone();
            let _ = std::thread::Builder::new().name("kirikae-host1x".into()).spawn(move || w.run(p));
        }
        self.changed.notify_all();
    }

    fn run(&self, p: &'static Process) {
        loop {
            let mut st = self.state.lock().unwrap();
            while st.0.is_empty() {
                st = self.changed.wait(st).unwrap();
            }
            let job = st.0.pop_front().unwrap();
            st.1 = true;
            drop(st);
            if let Err(e) = job() {
                p.warn_once(e);
            }
            self.state.lock().unwrap().1 = false;
            self.changed.notify_all();
        }
    }

    fn drain(&self) {
        let mut st = self.state.lock().unwrap();
        while !st.0.is_empty() || st.1 {
            st = self.changed.wait(st).unwrap();
        }
    }
}

#[derive(Default)]
pub struct Host1x {
    iova: Arc<Mutex<Iova>>,
    engines: Mutex<Engines>,
    worker: Arc<Worker>,
}

impl Host1x {
    pub fn map_buffer(&self, handle: u32, m: NvMap) -> Option<u32> {
        self.iova.lock().unwrap().map(handle, m)
    }

    pub fn drain(&self) {
        self.worker.drain();
    }

    pub fn submit(&self, p: &'static Process, context: u32, class: u32, words: &[u32]) -> Vec<String> {
        if class != CLASS_NVDEC {
            self.drain();
        }
        let (gpu, mem) = (&p.gpu, &p.memory);
        let iova = self.iova.lock().unwrap();
        let dma = Dma { iova: &iova, mem, gpu, later: Some((p, &self.iova, &self.worker)) };
        let mut engines = self.engines.lock().unwrap();
        let mut errors = Vec::new();
        let (mut class, mut offset, mut count, mut mask, mut incrementing) = (class, 0u32, 0u32, 0u32, false);
        for &w in words {
            let reg = if mask != 0 {
                let bit = mask.trailing_zeros();
                mask &= mask - 1;
                offset + bit
            } else if count != 0 {
                count -= 1;
                let r = offset;
                if incrementing {
                    offset += 1;
                }
                r
            } else {
                match w >> 28 {
                    0 => {
                        offset = (w >> 16) & 0xfff;
                        class = (w >> 6) & 0x3ff;
                        mask = w & 0x3f;
                    }
                    op @ (1 | 2) => {
                        offset = (w >> 16) & 0xfff;
                        count = w & 0xffff;
                        incrementing = op == 1;
                    }
                    3 => {
                        offset = (w >> 16) & 0xfff;
                        mask = w & 0xffff;
                    }
                    4 => {
                        let r = (w >> 16) & 0xfff;
                        engines.write(&dma, context, class, r, w & 0xfff, &mut errors);
                    }
                    op => errors.push(format!("host1x: opcode {op} is not supported")),
                }
                continue;
            };
            engines.write(&dma, context, class, reg, w, &mut errors);
        }
        errors
    }
}

impl Engines {
    fn write(&mut self, dma: &Dma, context: u32, class: u32, reg: u32, value: u32, errors: &mut Vec<String>) {
        let method = match class {
            CLASS_NVDEC => &mut self.nvdec_method,
            CLASS_VIC => &mut self.vic_method,
            CLASS_HOST1X => return,
            _ => {
                errors.push(format!("host1x: engine class {class:#x} is not emulated"));
                return;
            }
        };
        match reg {
            THI_INCR_SYNCPT => {
                dma.gpu.increment(value & 0xff, 1);
            }
            THI_METHOD0 => *method = value,
            THI_METHOD1 => {
                let m = *method;
                let r = match class {
                    CLASS_NVDEC => self.nvdec.write(dma, context, m, value),
                    _ => self.vic.write(dma, m, value),
                };
                if let Err(e) = r {
                    errors.push(e);
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn bits(b: &[u8], lsb: usize, len: usize) -> u32 {
    let mut v = 0u64;
    for i in 0..8 {
        v |= (b.get(lsb / 8 + i).copied().unwrap_or(0) as u64) << (8 * i);
    }
    ((v >> (lsb % 8)) & ((1u64 << len) - 1)) as u32
}

pub(crate) fn u32_at(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}

pub fn block_linear_size(width_bytes: usize, height: usize, block_height_log2: u32) -> usize {
    let rows = 8usize << block_height_log2;
    width_bytes.div_ceil(64) * 64 * height.div_ceil(rows) * rows
}

pub fn to_block_linear(src: &[u8], pitch: usize, width_bytes: usize, height: usize, block_height_log2: u32) -> Vec<u8> {
    let mut out = vec![0u8; block_linear_size(width_bytes, height, block_height_log2)];
    for y in 0..height {
        let row = &src[(y * pitch).min(src.len())..];
        for x in (0..width_bytes).step_by(16) {
            let n = 16.min(width_bytes - x).min(row.len().saturating_sub(x));
            if n == 0 {
                break;
            }
            let o = block_linear_offset(x as u64, y as u64, width_bytes as u64, block_height_log2) as usize;
            out[o..o + n].copy_from_slice(&row[x..x + n]);
        }
    }
    out
}

pub fn from_block_linear(raw: &[u8], width_bytes: usize, height: usize, block_height_log2: u32) -> Vec<u8> {
    let mut out = vec![0u8; width_bytes * height];
    for y in 0..height {
        for x in (0..width_bytes).step_by(16) {
            let n = 16.min(width_bytes - x);
            let o = block_linear_offset(x as u64, y as u64, width_bytes as u64, block_height_log2) as usize;
            if let Some(s) = raw.get(o..o + n) {
                out[y * width_bytes + x..y * width_bytes + x + n].copy_from_slice(s);
            }
        }
    }
    out
}

pub fn pitch(width_bytes: usize) -> usize {
    width_bytes.div_ceil(256) * 256
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_linear_round_trip() {
        let (w, h) = (200usize, 37usize);
        let src: Vec<u8> = (0..w * h).map(|i| (i * 7 + i / w) as u8).collect();
        for bh in 0..4 {
            let bl = to_block_linear(&src, w, w, h, bh);
            assert_eq!(bl.len(), block_linear_size(w, h, bh));
            assert_eq!(from_block_linear(&bl, w, h, bh), src);
        }
    }

    #[test]
    fn bit_fields() {
        let b = [0xffu8, 0x0f, 0, 0, 0, 0, 0, 0x80, 0x34, 0x12];
        assert_eq!(bits(&b, 0, 12), 0xfff);
        assert_eq!(bits(&b, 4, 8), 0xff);
        assert_eq!(bits(&b, 63, 1), 1);
        assert_eq!(bits(&b, 64, 16), 0x1234);
    }

    #[test]
    fn iova_follows_nvmap() {
        let mut io = Iova::default();
        let a = io.map(7, NvMap { size: 0x1800, addr: 0x8000_0000, ..NvMap::default() }).unwrap();
        let b = io.map(9, NvMap { size: 0x100, addr: 0x9000_0000, ..NvMap::default() }).unwrap();
        assert_eq!(io.map(7, NvMap { size: 0x1800, addr: 0x8000_0000, ..NvMap::default() }), Some(a));
        assert!(b as u64 >= a as u64 + 0x2000);
        assert_eq!(io.translate(a as u64 + 0x10, 0x10), Some(0x8000_0010));
        assert_eq!(io.translate(a as u64 + 0x1ff0, 0x20), None);
    }
}
