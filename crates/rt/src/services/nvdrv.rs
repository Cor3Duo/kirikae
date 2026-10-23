use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use crate::cpu::Cpu;
use crate::gpu::{Gpu, CLASS_2D, CLASS_3D, CLASS_COMPUTE, CLASS_DMA, CLASS_INLINE};
use crate::host1x::{Host1x, CLASS_NVDEC, CLASS_VIC};
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const NV_SUCCESS: u32 = 0;
const NV_NOT_IMPLEMENTED: u32 = 1;
const NV_BAD_PARAMETER: u32 = 4;
const NV_BAD_VALUE: u32 = 0xb;

#[derive(Clone, Debug)]
enum Device {
    Nvmap,
    AsGpu,
    Ctrl,
    CtrlGpu,
    Channel { syncpoint: u32 },
    Engine { class: u32 },
    Other(String),
}

#[derive(Default)]
struct State {
    fds: Mutex<HashMap<u32, Device>>,
    next_fd: AtomicU32,
    events: Mutex<HashMap<(u32, u32), u32>>,
    engine_syncpoints: Mutex<HashMap<(u32, u32), u32>>,
}

fn state(c: &Cpu) -> std::sync::Arc<State> {
    c.process().service_state("nvdrv", || State { next_fd: AtomicU32::new(1), ..State::default() })
}

pub struct Nvdrv {
    pub name: &'static str,
}

struct Args<'a>(&'a mut [u8]);

impl Args<'_> {
    fn u32(&self, o: usize) -> u32 {
        self.0.get(o..o + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn u64(&self, o: usize) -> u64 {
        self.0.get(o..o + 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn set32(&mut self, o: usize, v: u32) {
        if let Some(b) = self.0.get_mut(o..o + 4) {
            b.copy_from_slice(&v.to_le_bytes());
        }
    }
    fn set64(&mut self, o: usize, v: u64) {
        if let Some(b) = self.0.get_mut(o..o + 8) {
            b.copy_from_slice(&v.to_le_bytes());
        }
    }
}

impl Service for Nvdrv {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let st = state(c);
        match req.cmd {
            0 => {
                let path = req.in_buf(0).and_then(|b| p.memory.read(b.addr, b.size)).unwrap_or_default();
                let path = String::from_utf8_lossy(path.split(|&b| b == 0).next().unwrap_or(&[])).into_owned();
                let dev = match path.as_str() {
                    "/dev/nvmap" => Device::Nvmap,
                    "/dev/nvhost-as-gpu" => Device::AsGpu,
                    "/dev/nvhost-ctrl" => Device::Ctrl,
                    "/dev/nvhost-ctrl-gpu" => Device::CtrlGpu,
                    "/dev/nvhost-gpu" => Device::Channel { syncpoint: p.gpu.alloc_syncpoint() },
                    "/dev/nvhost-nvdec" => Device::Engine { class: CLASS_NVDEC },
                    "/dev/nvhost-vic" => Device::Engine { class: CLASS_VIC },
                    _ => Device::Other(path.clone()),
                };
                let fd = st.next_fd.fetch_add(1, Ordering::Relaxed);
                p.log_ipc(format!("nvdrv Open({path:?}) -> fd {fd}"));
                st.fds.lock().unwrap().insert(fd, dev);
                Reply::ok().u32(fd).u32(NV_SUCCESS)
            }
            1 | 11 | 12 => {
                let (fd, cmd) = (req.u32(0), req.u32(4));
                let Some(dev) = st.fds.lock().unwrap().get(&fd).cloned() else { return Reply::ok().u32(NV_BAD_PARAMETER) };
                let input = req.in_buf(0).and_then(|b| p.memory.read(b.addr, b.size)).unwrap_or_default();
                let extra = if req.cmd == 11 { req.in_buf(1).and_then(|b| p.memory.read(b.addr, b.size)).unwrap_or_default() } else { Vec::new() };
                let out_buf = req.out_buf(0);
                let size = ((cmd >> 16) & 0x3fff) as usize;
                let mut args = input.clone();
                args.resize(args.len().max(size).max(out_buf.map_or(0, |b| b.size as usize)), 0);
                let mut extra_out = Vec::new();
                let r = ioctl(c, fd, &dev, cmd, &mut Args(&mut args), &extra, &mut extra_out);
                if r == NV_SUCCESS {
                    p.log_ipc(format!("nvdrv ioctl {dev:?} {cmd:#010x}"));
                } else {
                    p.log_ipc(format!("nvdrv ioctl {dev:?} {cmd:#010x} -> {r:#x}, args {:02x?}", &input[..input.len().min(0x40)]));
                }
                if let Some(b) = out_buf {
                    p.memory.write(b.addr, &args[..(b.size as usize).min(args.len())]);
                }
                if let (12, Some(b)) = (req.cmd, req.out_buf(1)) {
                    p.memory.write(b.addr, &extra_out[..(b.size as usize).min(extra_out.len())]);
                }
                Reply::ok().u32(r)
            }
            2 => {
                p.service_state("host1x", Host1x::default).drain();
                st.fds.lock().unwrap().remove(&req.u32(0));
                Reply::ok().u32(NV_SUCCESS)
            }
            3 | 8 | 13 => Reply::ok().u32(NV_SUCCESS),
            4 => {
                let key = (req.u32(0), req.u32(4));
                let h = *st.events.lock().unwrap().entry(key).or_insert_with(|| p.create_event(false));
                Reply::ok().copy_handle(h).u32(NV_SUCCESS)
            }
            _ => unimplemented(self.name, c, req),
        }
    }
}

fn ioctl(c: &Cpu, fd: u32, dev: &Device, cmd: u32, a: &mut Args, extra: &[u8], extra_out: &mut Vec<u8>) -> u32 {
    let p = c.process();
    let gpu = &p.gpu;
    let nr = cmd & 0xff;
    let r = match dev {
        Device::Nvmap => {
            if nr == 0x05 {
                p.service_state("host1x", Host1x::default).drain();
            }
            nvmap(gpu, nr, a)
        }
        Device::AsGpu => as_gpu(gpu, nr, a, extra_out),
        Device::Ctrl => ctrl(gpu, nr, a),
        Device::CtrlGpu => ctrl_gpu(nr, a, extra_out),
        Device::Channel { syncpoint } => channel(c, gpu, *syncpoint, nr, a, extra),
        Device::Engine { class } => engine(c, fd, *class, cmd, a),
        Device::Other(path) => {
            p.warn_once(format!("nvdrv: device {path} is not emulated (ioctl {cmd:#010x})"));
            Some(NV_SUCCESS)
        }
    };
    if crate::usage::on() {
        let path = match dev {
            Device::Nvmap => "/dev/nvmap",
            Device::AsGpu => "/dev/nvhost-as-gpu",
            Device::Ctrl => "/dev/nvhost-ctrl",
            Device::CtrlGpu => "/dev/nvhost-ctrl-gpu",
            Device::Channel { .. } => "/dev/nvhost-gpu",
            Device::Engine { class } if *class == CLASS_NVDEC => "/dev/nvhost-nvdec",
            Device::Engine { .. } => "/dev/nvhost-vic",
            Device::Other(path) => path.as_str(),
        };
        crate::usage::ioctl(c, path, cmd, r.is_some() && !matches!(dev, Device::Other(_)), a.0);
    }
    r.unwrap_or_else(|| {
        p.warn_once(format!("nvdrv: {dev:?} ioctl {cmd:#010x} is stubbed"));
        NV_SUCCESS
    })
}

fn nvmap(gpu: &Gpu, nr: u32, a: &mut Args) -> Option<u32> {
    Some(match nr {
        0x01 => {
            let h = gpu.nvmap_create(a.u32(0) as u64);
            a.set32(4, h);
            NV_SUCCESS
        }
        0x03 => {
            let id = a.u32(0);
            if gpu.nvmap(id).is_none() {
                return Some(NV_BAD_VALUE);
            }
            a.set32(4, id);
            NV_SUCCESS
        }
        0x04 => {
            let (h, align, kind, addr) = (a.u32(0), a.u32(12), a.0.get(16).copied().unwrap_or(0), a.u64(24));
            if gpu.nvmap_alloc(h, addr, align, kind) { NV_SUCCESS } else { NV_BAD_VALUE }
        }
        0x05 => {
            match gpu.nvmap_free(a.u32(0)) {
                Some(m) => {
                    a.set64(8, m.addr);
                    a.set32(16, m.size as u32);
                    a.set32(20, 0);
                    NV_SUCCESS
                }
                None => NV_BAD_VALUE,
            }
        }
        0x09 => {
            let Some(m) = gpu.nvmap(a.u32(0)) else { return Some(NV_BAD_VALUE) };
            let v = match a.u32(4) {
                1 => m.size as u32,
                2 => m.align,
                3 => m.addr as u32,
                4 => 0x4000_0000,
                5 => m.kind as u32,
                6 => 0,
                _ => return Some(NV_BAD_VALUE),
            };
            a.set32(8, v);
            NV_SUCCESS
        }
        0x0e => {
            let h = a.u32(4);
            a.set32(0, h);
            NV_SUCCESS
        }
        _ => return None,
    })
}

fn as_gpu(gpu: &Gpu, nr: u32, a: &mut Args, extra_out: &mut Vec<u8>) -> Option<u32> {
    const FIXED: u32 = 1;
    const REMAP: u32 = 0x100;
    Some(match nr {
        0x01 | 0x09 => NV_SUCCESS,
        0x02 => {
            let (pages, page, flags) = (a.u32(0) as u64, a.u32(4) as u64, a.u32(8));
            let fixed = (flags & FIXED != 0).then(|| a.u64(16));
            let va = gpu.reserve_va(pages * page, page, fixed);
            a.set64(16, va);
            NV_SUCCESS
        }
        0x03 | 0x05 => {
            gpu.unmap(a.u64(0));
            NV_SUCCESS
        }
        0x06 => {
            let (flags, handle) = (a.u32(0), a.u32(8));
            let (buffer_offset, mut size) = (a.u64(16), a.u64(24));
            if flags & REMAP != 0 {
                let va = a.u64(32);
                let Some(cpu) = gpu.translate(va) else { return Some(NV_BAD_VALUE) };
                gpu.map(va + buffer_offset, cpu + buffer_offset, size);
                return Some(NV_SUCCESS);
            }
            let Some(m) = gpu.nvmap(handle) else { return Some(NV_BAD_VALUE) };
            if size == 0 {
                size = m.size - buffer_offset.min(m.size);
            }
            let fixed = (flags & FIXED != 0).then(|| a.u64(32));
            let va = gpu.alloc_va(size, a.u32(12).max(0x1000) as u64, fixed);
            gpu.map(va, m.addr + buffer_offset, size);
            a.set64(32, va);
            NV_SUCCESS
        }
        0x08 => {
            a.set64(16, 0x0400_0000);
            a.set32(24, 0x1000);
            a.set64(32, 0x3fbfff);
            a.set64(40, 0x4_0000_0000);
            a.set32(48, 0x2_0000);
            a.set64(56, 0x1bffff);
            *extra_out = a.0.get(16..64).unwrap_or(&[]).to_vec();
            NV_SUCCESS
        }
        0x14 => {
            for e in a.0.as_chunks::<0x14>().0 {
                let u = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap());
                let (handle, map_off, gpu_off, pages) = (u(4), (u(8) as u64) << 16, (u(12) as u64) << 16, (u(16) as u64) << 16);
                if let Some(m) = gpu.nvmap(handle).filter(|_| handle != 0) {
                    gpu.map(gpu_off, m.addr + map_off, pages);
                }
            }
            NV_SUCCESS
        }
        _ => return None,
    })
}

fn ctrl(gpu: &Gpu, nr: u32, a: &mut Args) -> Option<u32> {
    Some(match nr {
        0x14 => {
            let v = gpu.syncpoint(a.u32(0));
            a.set32(4, v);
            NV_SUCCESS
        }
        0x15 => {
            gpu.increment(a.u32(0), 1);
            NV_SUCCESS
        }
        0x16 | 0x19 | 0x1d | 0x1e => {
            gpu.wait_idle();
            if nr != 0x16 {
                let v = gpu.syncpoint(a.u32(0));
                a.set32(12, v);
            }
            NV_SUCCESS
        }
        0x1b => NV_NOT_IMPLEMENTED,
        0x1f..=0x21 => NV_SUCCESS,
        _ => return None,
    })
}

fn characteristics() -> Vec<u8> {
    let mut v = Vec::with_capacity(0xa0);
    let u32s = |xs: &[u32], v: &mut Vec<u8>| xs.iter().for_each(|x| v.extend_from_slice(&x.to_le_bytes()));
    u32s(&[0x120, 0xb, 0xa1, 1], &mut v);
    v.extend_from_slice(&0x40000u64.to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes());
    u32s(&[2, 0x20, 0x20000, 0x20000, 0x1b, 0x30000, 1, 0x503, 0x503, 0x80], &mut v);
    v.extend_from_slice(&0x28u16.to_le_bytes());
    v.extend_from_slice(&0u16.to_le_bytes());
    u32s(&[0x55, CLASS_2D, CLASS_3D, CLASS_COMPUTE, 0xb06f, CLASS_INLINE, CLASS_DMA, 1, 0, 2, 1, 0, 1, 0x21d70, 0], &mut v);
    v.extend_from_slice(&0x6230326d67u64.to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes());
    v.resize(0xa0, 0);
    v
}

fn ctrl_gpu(nr: u32, a: &mut Args, extra_out: &mut Vec<u8>) -> Option<u32> {
    Some(match nr {
        0x01 => {
            a.set32(0, 1);
            NV_SUCCESS
        }
        0x02 => {
            for (k, v) in [0x20u32, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10].iter().enumerate() {
                a.set32(4 * k, *v);
            }
            NV_SUCCESS
        }
        0x03 | 0x04 | 0x07 | 0x0b => NV_SUCCESS,
        0x05 => {
            a.set64(0, 0xa0);
            let ch = characteristics();
            if let Some(dst) = a.0.get_mut(16..16 + 0xa0) {
                dst.copy_from_slice(&ch);
            }
            *extra_out = ch;
            NV_SUCCESS
        }
        0x06 => {
            if a.u32(0) != 0 {
                a.set64(16, 3);
                *extra_out = 3u64.to_le_bytes().to_vec();
            }
            NV_SUCCESS
        }
        0x12 => {
            a.set32(0, 2);
            NV_SUCCESS
        }
        0x13 => {
            *extra_out = vec![0, 0, 0, 1];
            NV_SUCCESS
        }
        0x14 => {
            a.set32(0, 7);
            a.set32(4, 1);
            NV_SUCCESS
        }
        0x1c => {
            a.set64(0, crate::gpu::gpu_ticks());
            NV_SUCCESS
        }
        _ => return None,
    })
}

fn channel(c: &Cpu, gpu: &Gpu, syncpoint: u32, nr: u32, a: &mut Args, extra: &[u8]) -> Option<u32> {
    const FENCE_INCREMENT: u32 = 2;
    const INCREMENT_WITH_VALUE: u32 = 0x100;
    let p = c.process();
    Some(match nr {
        0x01 | 0x03 | 0x0b | 0x0c | 0x0d | 0x1d | 0x14 => NV_SUCCESS,
        0x08 | 0x1b => {
            let (count, flags) = (a.u32(8) as usize, a.u32(12));
            let src: &[u8] = if extra.len() >= count * 8 { extra } else { a.0.get(0x18..).unwrap_or(&[]) };
            let entries: Vec<u64> = src.as_chunks::<8>().0.iter().take(count).map(|b| u64::from_le_bytes(*b)).collect();
            if let Err(e) = crate::sched::without_gil(c, || {
                let _prof = crate::prof::enter(crate::prof::Cat::Gpu);
                gpu.submit(&p.memory, &entries)
            }) {
                p.warn_once(format!("GPU: {e}"));
            }
            if flags & FENCE_INCREMENT != 0 {
                let by = if flags & INCREMENT_WITH_VALUE != 0 { a.u32(20) } else { 1 };
                let value = gpu.increment(syncpoint, by);
                a.set32(16, syncpoint);
                a.set32(20, value);
            }
            NV_SUCCESS
        }
        0x09 => {
            a.set64(8, a.u32(0) as u64);
            NV_SUCCESS
        }
        0x1a => {
            a.set32(12, syncpoint);
            a.set32(16, gpu.syncpoint(syncpoint));
            NV_SUCCESS
        }
        0x16 => NV_SUCCESS,
        _ => return None,
    })
}

fn engine(c: &Cpu, fd: u32, class: u32, cmd: u32, a: &mut Args) -> Option<u32> {
    let p = c.process();
    let gpu = &p.gpu;
    let host1x = p.service_state("host1x", Host1x::default);
    let st = state(c);
    let nr = cmd & 0xff;
    match (cmd >> 8) & 0xff {
        0x00 => {}
        0x48 if matches!(nr, 0x01 | 0x03) => return Some(NV_SUCCESS),
        0x47 if nr == 0x14 => return Some(NV_SUCCESS),
        _ => return None,
    }
    Some(match nr {
        0x07 => NV_SUCCESS,
        0x01 => {
            let (buffers, relocs, incrs, fences) = (a.u32(0) as usize, a.u32(4) as usize, a.u32(8) as usize, a.u32(12) as usize);
            let incr_at = 16 + 12 * buffers + 20 * relocs;
            let fence_at = incr_at + 20 * incrs;
            let mut thresholds: Vec<(u32, u32)> = Vec::new();
            for i in 0..incrs {
                let (id, n) = (a.u32(incr_at + 20 * i), a.u32(incr_at + 20 * i + 4));
                let before = thresholds.iter().rev().find(|t| t.0 == id).map_or(gpu.syncpoint(id), |t| t.1);
                thresholds.push((id, before.wrapping_add(n)));
            }
            if crate::host1x::trace() {
                let reloc_at = 16 + 12 * buffers;
                let relocs: Vec<[u32; 5]> = (0..relocs)
                    .map(|i| [0, 4, 8, 12].map(|o| a.u32(reloc_at + 16 * i + o)))
                    .enumerate()
                    .map(|(i, r)| [r[0], r[1], r[2], r[3], a.u32(reloc_at + 16 * relocs + 4 * i)])
                    .collect();
                eprintln!("nvdrv: class {class:#x} SUBMIT {buffers} buffer(s), relocs {relocs:x?}, increments {thresholds:?}");
            }
            for i in 0..buffers {
                let (handle, offset, words) = (a.u32(16 + 12 * i), a.u32(20 + 12 * i), a.u32(24 + 12 * i));
                let Some(m) = gpu.nvmap(handle).filter(|m| m.addr != 0) else { return Some(NV_BAD_VALUE) };
                let Some(bytes) = p.memory.read(m.addr + offset as u64, words as u64 * 4) else { return Some(NV_BAD_VALUE) };
                let mut words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
                let reloc_at = 16 + 12 * buffers;
                for r in 0..relocs {
                    let e = |o: usize| a.u32(reloc_at + 16 * r + o);
                    let (buffer, at, target, target_offset, shift) = (e(0), e(4), e(8), e(12), a.u32(reloc_at + 16 * relocs + 4 * r));
                    let Some(index) = at.checked_sub(offset).map(|o| o as usize / 4).filter(|&i| buffer == handle && i < words.len()) else { continue };
                    if let Some(iova) = gpu.nvmap(target).and_then(|m| host1x.map_buffer(target, m)) {
                        words[index] = (iova.wrapping_add(target_offset) as u64 >> (shift & 31)) as u32;
                    }
                }
                let errors = crate::sched::without_gil(c, || {
                    let _prof = crate::prof::enter(crate::prof::Cat::Video);
                    host1x.submit(p, fd, class, &words)
                });
                for e in errors {
                    p.warn_once(e);
                }
            }
            for &(id, threshold) in &thresholds {
                let behind = threshold.wrapping_sub(gpu.syncpoint(id));
                if (behind as i32) > 0 {
                    gpu.increment(id, behind);
                }
            }
            for (i, &(_, threshold)) in thresholds.iter().take(fences).enumerate() {
                a.set32(fence_at + 4 * i, threshold);
            }
            NV_SUCCESS
        }
        0x02 => {
            let index = a.u32(0);
            let id = *st.engine_syncpoints.lock().unwrap().entry((class, index)).or_insert_with(|| gpu.alloc_syncpoint());
            a.set32(4, id);
            NV_SUCCESS
        }
        0x03 => {
            a.set32(4, 0);
            NV_SUCCESS
        }
        0x09 => {
            for i in 0..a.u32(0) as usize {
                let handle = a.u32(12 + 8 * i);
                let Some(m) = gpu.nvmap(handle) else { return Some(NV_BAD_VALUE) };
                let Some(iova) = host1x.map_buffer(handle, m) else { return Some(NV_BAD_VALUE) };
                a.set32(16 + 8 * i, iova);
            }
            NV_SUCCESS
        }
        0x0a => NV_SUCCESS,
        _ => return None,
    })
}
