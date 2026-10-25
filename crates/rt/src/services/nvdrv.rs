use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::cpu::Cpu;
use crate::host1x::{CLASS_NVDEC, CLASS_VIC, Host1x};
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

use self::args::{Args, Env};
use self::names::{characteristics, ioctl_name};
#[cfg(test)]
use self::submit::{SubmitLayout, gpfifo_entry};
use self::submit::{engine_submit, legacy_submit, submit_gpfifo};

const NV_SUCCESS: u32 = 0;
const NV_NOT_IMPLEMENTED: u32 = 1;
const NV_NOT_SUPPORTED: u32 = 2;
const NV_BAD_PARAMETER: u32 = 4;
const NV_TIMEOUT: u32 = 5;
const NV_BAD_VALUE: u32 = 0xb;

const PAGE_SMALL: u64 = 0x1000;
const PAGE_BIG: u64 = 0x2_0000;
const GPU_CLK_RATE: u32 = 768_000_000;
/// The two GPU virtual address regions `/dev/nvhost-as-gpu` reports, as the
/// console does: a small-page window below 4 GiB and a big-page window above it.
const VA_REGIONS: [(u64, u32, u64); 2] = [
    (0x0400_0000, PAGE_SMALL as u32, 0x3fbfff),
    (0x40_0000_0000, PAGE_BIG as u32, 0x1bffff),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Outcome {
    Ok,
    Fail(u32),
    Unimplemented,
}

#[derive(Clone, Debug)]
enum Device {
    Nvmap,
    AsGpu,
    Ctrl,
    CtrlGpu,
    Channel { syncpoint: u32 },
    Engine { class: u32 },
}

#[derive(Clone, Copy, Default)]
struct Zbc {
    words: [u32; 11],
}

#[derive(Default)]
struct State {
    fds: Mutex<HashMap<u32, Device>>,
    next_fd: AtomicU32,
    events: Mutex<HashMap<(u32, u32), u32>>,
    engine_syncpoints: Mutex<HashMap<(u32, u32), u32>>,
    /// `NVGPU_IOCTL_CHANNEL_SET_USER_DATA`, keyed by the channel's syncpoint.
    user_data: Mutex<HashMap<u32, u64>>,
    /// `NVGPU_GPU_IOCTL_ZBC_SET_TABLE`, indexed by kind (color, depth).
    zbc: Mutex<[Option<Zbc>; 2]>,
    /// `NVMAP_IOC_EXPORT_FOR_ARUID`, mapping a nvmap handle to its aruid.
    exports: Mutex<HashMap<u32, u64>>,
    /// Address space reservations made with `NVGPU_AS_IOCTL_ALLOC_SPACE`.
    reservations: Mutex<Vec<(u64, u64)>>,
}

fn state(c: &Cpu) -> std::sync::Arc<State> {
    c.process().service_state("nvdrv", || State {
        next_fd: AtomicU32::new(1),
        ..State::default()
    })
}

mod args;
mod names;
mod submit;

pub struct Nvdrv {
    pub name: &'static str,
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
                let path = req
                    .in_buf(0)
                    .and_then(|b| p.memory.read(b.addr, b.size))
                    .unwrap_or_default();
                let path = String::from_utf8_lossy(path.split(|&b| b == 0).next().unwrap_or(&[]))
                    .into_owned();
                let dev = match path.as_str() {
                    "/dev/nvmap" => Some(Device::Nvmap),
                    "/dev/nvhost-as-gpu" => Some(Device::AsGpu),
                    "/dev/nvhost-ctrl" => Some(Device::Ctrl),
                    "/dev/nvhost-ctrl-gpu" => Some(Device::CtrlGpu),
                    "/dev/nvhost-gpu" => Some(Device::Channel {
                        syncpoint: p.gpu.alloc_syncpoint(),
                    }),
                    "/dev/nvhost-nvdec" => Some(Device::Engine { class: CLASS_NVDEC }),
                    "/dev/nvhost-vic" => Some(Device::Engine { class: CLASS_VIC }),
                    _ => None,
                };
                let Some(dev) = dev else {
                    p.warn_once(format!(
                        "nvdrv: Open({path:?}) is not a device this runtime implements"
                    ));
                    return Reply::ok().u32(0).u32(NV_NOT_SUPPORTED);
                };
                let fd = st.next_fd.fetch_add(1, Ordering::Relaxed);
                p.log_ipc(format!("nvdrv Open({path:?}) -> fd {fd}"));
                st.fds.lock().unwrap().insert(fd, dev);
                Reply::ok().u32(fd).u32(NV_SUCCESS)
            }
            1 | 11 | 12 => {
                let (fd, cmd) = (req.u32(0), req.u32(4));
                let Some(dev) = st.fds.lock().unwrap().get(&fd).cloned() else {
                    return Reply::ok().u32(NV_BAD_PARAMETER);
                };
                let input = req
                    .in_buf(0)
                    .and_then(|b| p.memory.read(b.addr, b.size))
                    .unwrap_or_default();
                let extra = if req.cmd == 11 {
                    req.in_buf(1)
                        .and_then(|b| p.memory.read(b.addr, b.size))
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let out_buf = req.out_buf(0);
                let size = declared_size(cmd);
                let mut args = input.clone();
                args.resize(
                    args.len()
                        .max(size)
                        .max(out_buf.map_or(0, |b| b.size as usize)),
                    0,
                );
                let host1x = matches!(dev, Device::Engine { .. })
                    .then(|| p.service_state("host1x", Host1x::default));
                let env = Env {
                    gpu: &p.gpu,
                    mem: &p.memory,
                    st: &st,
                    cpu: Some(c),
                    host1x: host1x.as_deref(),
                };
                let mut extra_out = Vec::new();
                let (outcome, overrun) = if !args_supplied(cmd, input.len(), args.len()) {
                    p.warn_once(format!(
                        "nvdrv: {dev:?} ioctl {cmd:#010x} declares {size} argument bytes and the buffers do not carry them"
                    ));
                    (Outcome::Fail(NV_BAD_PARAMETER), false)
                } else {
                    let mut a = Args::new(&mut args);
                    let outcome = dispatch(&env, fd, &dev, cmd, &mut a, &extra, &mut extra_out);
                    (outcome, a.overrun)
                };
                let outcome = if overrun {
                    p.warn_once(format!(
                        "nvdrv: {dev:?} ioctl {cmd:#010x} wrote past its {size}-byte argument struct"
                    ));
                    Outcome::Fail(NV_BAD_VALUE)
                } else {
                    outcome
                };
                let code = match outcome {
                    Outcome::Ok => NV_SUCCESS,
                    Outcome::Fail(code) => code,
                    Outcome::Unimplemented => {
                        p.warn_once(format!(
                            "nvdrv: {} ({cmd:#010x}, {} bytes) on {} is not implemented",
                            ioctl_name(&dev, cmd),
                            size,
                            device_path(&dev)
                        ));
                        NV_NOT_IMPLEMENTED
                    }
                };
                if code == NV_SUCCESS {
                    p.log_ipc(format!("nvdrv ioctl {dev:?} {cmd:#010x}"));
                } else {
                    p.log_ipc(format!(
                        "nvdrv ioctl {dev:?} {cmd:#010x} -> {code:#x}, args {:02x?}",
                        &input[..input.len().min(0x40)]
                    ));
                }
                if crate::usage::on() {
                    crate::usage::ioctl(
                        c,
                        device_path(&dev),
                        cmd,
                        !matches!(outcome, Outcome::Unimplemented),
                        &args,
                    );
                }
                if let Some(b) = out_buf {
                    p.memory
                        .write(b.addr, &args[..(b.size as usize).min(args.len())]);
                }
                if let (12, Some(b)) = (req.cmd, req.out_buf(1)) {
                    p.memory
                        .write(b.addr, &extra_out[..(b.size as usize).min(extra_out.len())]);
                }
                Reply::ok().u32(code)
            }
            2 => {
                p.service_state("host1x", Host1x::default).drain();
                st.fds.lock().unwrap().remove(&req.u32(0));
                Reply::ok().u32(NV_SUCCESS)
            }
            3 | 8 | 13 => Reply::ok().u32(NV_SUCCESS),
            4 => {
                let key = (req.u32(0), req.u32(4));
                let h = *st
                    .events
                    .lock()
                    .unwrap()
                    .entry(key)
                    .or_insert_with(|| p.create_event(false));
                Reply::ok().copy_handle(h).u32(NV_SUCCESS)
            }
            _ => unimplemented(self.name, c, req),
        }
    }
}

fn declared_size(cmd: u32) -> usize {
    ((cmd >> 16) & 0x3fff) as usize
}

/// An ioctl number says whether its argument struct travels in the input buffer
/// (`_IOW`), in the output buffer (`_IOR`) or in both (`_IOWR`, the common case).
fn args_supplied(cmd: u32, input: usize, output: usize) -> bool {
    let size = declared_size(cmd);
    let supplied = if (cmd >> 30) & 1 != 0 { input } else { output };
    supplied >= size
}

fn device_path(dev: &Device) -> &'static str {
    match dev {
        Device::Nvmap => "/dev/nvmap",
        Device::AsGpu => "/dev/nvhost-as-gpu",
        Device::Ctrl => "/dev/nvhost-ctrl",
        Device::CtrlGpu => "/dev/nvhost-ctrl-gpu",
        Device::Channel { .. } => "/dev/nvhost-gpu",
        Device::Engine { class } if *class == CLASS_NVDEC => "/dev/nvhost-nvdec",
        Device::Engine { .. } => "/dev/nvhost-vic",
    }
}

fn dispatch(
    env: &Env,
    fd: u32,
    dev: &Device,
    cmd: u32,
    a: &mut Args,
    extra: &[u8],
    extra_out: &mut Vec<u8>,
) -> Outcome {
    let nr = cmd & 0xff;
    match dev {
        Device::Nvmap => {
            if nr == 0x05
                && let Some(c) = env.cpu
            {
                c.process().service_state("host1x", Host1x::default).drain();
            }
            nvmap(env, nr, a)
        }
        Device::AsGpu => as_gpu(env, nr, a, extra_out),
        Device::Ctrl => ctrl(env, nr, a),
        Device::CtrlGpu => ctrl_gpu(env, nr, a, extra_out),
        Device::Channel { syncpoint } => channel(env, *syncpoint, cmd, a, extra),
        Device::Engine { class } => engine(env, fd, *class, cmd, a),
    }
}

fn nvmap(env: &Env, nr: u32, a: &mut Args) -> Outcome {
    let gpu = env.gpu;
    match nr {
        0x01 => {
            let h = gpu.nvmap_create(a.u32(0) as u64);
            a.set32(4, h);
            Outcome::Ok
        }
        // The console answers these with NotSupported (Switchbrew, NV_services).
        0x02 | 0x06 | 0x07 | 0x08 | 0x0a | 0x0b | 0x0c | 0x0d | 0x0f | 0x10 | 0x11 => {
            Outcome::Fail(NV_NOT_SUPPORTED)
        }
        0x03 => {
            let id = a.u32(0);
            if gpu.nvmap(id).is_none() {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            a.set32(4, id);
            Outcome::Ok
        }
        0x04 => {
            let h = a.u32(0);
            let align = a.u32(12).max(PAGE_SMALL as u32);
            if !gpu.nvmap_alloc(h, a.u64(24), align, a.u8(16)) {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            a.set32(12, align);
            Outcome::Ok
        }
        0x05 => match gpu.nvmap_free(a.u32(0)) {
            Some(m) => {
                a.set64(8, m.addr);
                a.set32(16, m.size as u32);
                a.set32(20, 0);
                Outcome::Ok
            }
            None => Outcome::Fail(NV_BAD_VALUE),
        },
        0x09 => {
            let Some(m) = gpu.nvmap(a.u32(0)) else {
                return Outcome::Fail(NV_BAD_VALUE);
            };
            let value = match a.u32(4) {
                1 => m.size as u32,
                2 => m.align,
                3 => return Outcome::Fail(NV_BAD_VALUE),
                4 => 0x4000_0000,
                5 => m.kind as u32,
                6 => 0,
                _ => return Outcome::Fail(NV_BAD_VALUE),
            };
            a.set32(8, value);
            Outcome::Ok
        }
        0x0e => {
            let h = a.u32(4);
            if gpu.nvmap(h).is_none() {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            a.set32(0, h);
            Outcome::Ok
        }
        0x12 => {
            let (aruid, handle) = (a.u64(0), a.u32(8));
            if gpu.nvmap(handle).is_none() {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            env.st.exports.lock().unwrap().insert(handle, aruid);
            Outcome::Ok
        }
        0x13 => {
            let (aruid, handle) = (a.u64(0), a.u32(8));
            if env.st.exports.lock().unwrap().get(&handle) == Some(&aruid) {
                Outcome::Ok
            } else {
                Outcome::Fail(NV_BAD_VALUE)
            }
        }
        0x14 => {
            let handle = a.u32(8);
            if env.st.exports.lock().unwrap().remove(&handle).is_some() {
                Outcome::Ok
            } else {
                Outcome::Fail(NV_BAD_VALUE)
            }
        }
        _ => Outcome::Unimplemented,
    }
}

fn as_gpu(env: &Env, nr: u32, a: &mut Args, extra_out: &mut Vec<u8>) -> Outcome {
    let gpu = env.gpu;
    const FIXED: u32 = 1;
    match nr {
        0x01 => {
            let fd = a.u32(0);
            if !env.st.fds.lock().unwrap().contains_key(&fd) {
                return Outcome::Fail(NV_BAD_PARAMETER);
            }
            Outcome::Ok
        }
        0x02 => {
            let (pages, page, flags) = (a.u32(0) as u64, a.u32(4) as u64, a.u32(8));
            if pages == 0 {
                return Outcome::Fail(NV_BAD_PARAMETER);
            }
            if !matches!(page, 0x1000 | 0x1_0000 | 0x2_0000) {
                return Outcome::Fail(NV_BAD_PARAMETER);
            }
            let fixed = (flags & FIXED != 0).then(|| a.u64(16));
            let size = pages * page;
            let va = gpu.reserve_va(size, page, fixed);
            a.set64(16, va);
            env.st.reservations.lock().unwrap().push((va, size));
            Outcome::Ok
        }
        0x03 => {
            let offset = a.u64(0);
            let size = a.u32(8) as u64 * a.u32(12) as u64;
            let mut reservations = env.st.reservations.lock().unwrap();
            let Some(index) = reservations
                .iter()
                .position(|&(o, s)| o == offset && (size == 0 || s == size))
            else {
                return Outcome::Fail(NV_BAD_VALUE);
            };
            reservations.remove(index);
            gpu.unmap(offset);
            Outcome::Ok
        }
        0x04 => {
            let flags = a.u32(0);
            let Some(m) = gpu.nvmap(a.u32(8)) else {
                return Outcome::Fail(NV_BAD_VALUE);
            };
            if m.size == 0 {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            let align = a.u64(16);
            let fixed = (flags & FIXED != 0).then_some(align);
            let va = gpu.alloc_va(m.size, align.max(PAGE_SMALL), fixed);
            gpu.map(va, m.addr, m.size);
            a.set64(16, va);
            Outcome::Ok
        }
        0x05 => {
            // NVN also unmaps ranges it reserved but never mapped, and passes
            // `u64::MAX` for "the whole space"; an unknown start is a no-op.
            gpu.unmap(a.u64(0));
            Outcome::Ok
        }
        0x06 => {
            let flags = a.u32(0);
            let Some(m) = gpu.nvmap(a.u32(8)) else {
                return Outcome::Fail(NV_BAD_VALUE);
            };
            let buffer_offset = a.u64(16);
            let mut size = a.u64(24);
            if size == 0 {
                size = m.size.saturating_sub(buffer_offset);
            }
            if size == 0 {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            let fixed = (flags & FIXED != 0).then(|| a.u64(32));
            let va = gpu.alloc_va(size, a.u32(12).max(PAGE_SMALL as u32) as u64, fixed);
            gpu.map(va, m.addr + buffer_offset, size);
            a.set64(32, va);
            Outcome::Ok
        }
        // ALLOC_AS only carries hints: `as_fd` is ignored and the page size is
        // chosen by the driver, so the console accepts it without a reply.
        0x07 => Outcome::Ok,
        0x08 => {
            a.set64(0, 0);
            a.set32(8, 0x30);
            a.set32(12, 0);
            for (i, &(offset, page_size, pages)) in VA_REGIONS.iter().enumerate() {
                let base = 16 + i * 24;
                a.set64(base, offset);
                a.set32(base + 8, page_size);
                a.set32(base + 12, 0);
                a.set64(base + 16, pages);
            }
            *extra_out = a.bytes(16, 0x30).to_vec();
            Outcome::Ok
        }
        0x09 => Outcome::Ok,
        0x0a => {
            let flags = a.u32(0);
            let Some(m) = gpu.nvmap(a.u32(8)) else {
                return Outcome::Fail(NV_BAD_VALUE);
            };
            let buffer_offset = a.u64(16);
            let mut size = a.u64(24);
            if size == 0 {
                size = m.size.saturating_sub(buffer_offset);
            }
            if size == 0 {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            let vma = a.u64(40);
            let fixed = if vma != 0 {
                Some(vma)
            } else {
                (flags & FIXED != 0).then(|| a.u64(32))
            };
            let va = gpu.alloc_va(size, a.u32(12).max(PAGE_SMALL as u32) as u64, fixed);
            gpu.map(va, m.addr + buffer_offset, size);
            a.set64(32, va);
            Outcome::Ok
        }
        0x14 => {
            let entries = a.data();
            if !entries.len().is_multiple_of(0x14) {
                return Outcome::Fail(NV_BAD_PARAMETER);
            }
            for e in entries.as_chunks::<0x14>().0 {
                let u = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap());
                let handle = u(4);
                let map_offset = (u(8) as u64) << 16;
                let gpu_offset = (u(12) as u64) << 16;
                let pages = (u(16) as u64) << 16;
                if handle == 0 {
                    continue;
                }
                let Some(m) = gpu.nvmap(handle) else {
                    return Outcome::Fail(NV_BAD_VALUE);
                };
                gpu.map(gpu_offset, m.addr + map_offset, pages);
            }
            Outcome::Ok
        }
        _ => Outcome::Unimplemented,
    }
}

fn syncpt_reached(env: &Env, id: u32, threshold: u32) -> bool {
    env.gpu.wait_idle();
    env.gpu.syncpoint(id).wrapping_sub(threshold) < 0x8000_0000
}

fn ctrl(env: &Env, nr: u32, a: &mut Args) -> Outcome {
    let gpu = env.gpu;
    match nr {
        0x14 | 0x1a => {
            a.set32(4, gpu.syncpoint(a.u32(0)));
            Outcome::Ok
        }
        0x15 => {
            gpu.increment(a.u32(0), 1);
            Outcome::Ok
        }
        0x16 => {
            let (id, threshold) = (a.u32(0), a.u32(4));
            if syncpt_reached(env, id, threshold) {
                Outcome::Ok
            } else {
                Outcome::Fail(NV_TIMEOUT)
            }
        }
        0x17 => {
            if a.u32(4) > 1 {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            Outcome::Ok
        }
        // `NVHOST_IOCTL_CTRL_MODULE_REGRDWR` is documented as not working and
        // `GET_CONFIG` is not available in production mode.
        0x18 | 0x1b => Outcome::Fail(NV_NOT_SUPPORTED),
        0x19 => {
            let (id, threshold) = (a.u32(0), a.u32(4));
            let value = gpu.syncpoint(id);
            a.set32(12, value);
            if syncpt_reached(env, id, threshold) {
                Outcome::Ok
            } else {
                Outcome::Fail(NV_TIMEOUT)
            }
        }
        0x1c => {
            if a.u32(0) >= 64 {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            Outcome::Ok
        }
        0x1d => {
            let (id, threshold) = (a.u32(0), a.u32(4));
            if syncpt_reached(env, id, threshold) {
                a.set32(12, gpu.syncpoint(id));
                Outcome::Ok
            } else {
                a.set32(12, id & 0xfff);
                Outcome::Fail(NV_TIMEOUT)
            }
        }
        0x1e => {
            let (id, threshold) = (a.u32(0), a.u32(4));
            let slot = a.u32(12) & 0x3f;
            if syncpt_reached(env, id, threshold) {
                a.set32(12, gpu.syncpoint(id));
                Outcome::Ok
            } else {
                a.set32(12, slot | (id << 4));
                Outcome::Fail(NV_TIMEOUT)
            }
        }
        0x1f | 0x20 => {
            if a.u32(0) >= 64 {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            Outcome::Ok
        }
        0x21 => Outcome::Ok,
        0x22 => {
            a.set32(0, 0);
            Outcome::Ok
        }
        _ => Outcome::Unimplemented,
    }
}

fn ctrl_gpu(env: &Env, nr: u32, a: &mut Args, extra_out: &mut Vec<u8>) -> Outcome {
    match nr {
        0x01 => {
            a.set32(0, 0x1000);
            Outcome::Ok
        }
        0x02 => {
            for (k, v) in [
                0x20u32, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10,
            ]
            .iter()
            .enumerate()
            {
                a.set32(4 * k, *v);
            }
            Outcome::Ok
        }
        0x03 => {
            let kind = a.u32(40);
            if !matches!(kind, 1 | 2) {
                return Outcome::Fail(NV_BAD_VALUE);
            }
            let mut words = [0u32; 11];
            for (i, w) in words.iter_mut().enumerate() {
                *w = a.u32(4 * i);
            }
            env.st.zbc.lock().unwrap()[kind as usize - 1] = Some(Zbc { words });
            Outcome::Ok
        }
        0x04 => {
            let zbc = env.st.zbc.lock().unwrap();
            if let Some(t) = zbc[0].or(zbc[1]) {
                for i in 0..8 {
                    a.set32(4 * i, t.words[i]);
                }
                a.set32(32, t.words[8]);
                a.set32(36, 1);
                a.set32(40, t.words[9]);
                a.set32(44, t.words[10]);
            }
            a.set32(48, 0);
            Outcome::Ok
        }
        0x05 => {
            a.set64(0, 0xa0);
            let ch = characteristics();
            a.set(16, &ch);
            *extra_out = ch;
            Outcome::Ok
        }
        0x06 => {
            a.set64(16, 0x3);
            Outcome::Ok
        }
        0x07 | 0x0d | 0x0e | 0x0f | 0x16 | 0x18 => Outcome::Ok,
        0x10 => {
            if a.len() >= 48 {
                a.fill(0, 48, 0);
            }
            Outcome::Ok
        }
        0x11 => {
            a.set64(0, 0);
            Outcome::Ok
        }
        0x12 => {
            a.set32(0, 2);
            a.set32(4, 0);
            Outcome::Ok
        }
        0x13 => {
            a.set32(0, 0x0100_0000);
            Outcome::Ok
        }
        0x14 => {
            a.set32(0, 7);
            a.set32(4, 1);
            Outcome::Ok
        }
        0x15 => {
            a.set32(0, 0);
            Outcome::Ok
        }
        0x17 | 0x19 => {
            a.set32(4, 0);
            Outcome::Ok
        }
        0x1a => {
            a.set64(0, 0);
            a.set64(8, 0);
            a.set64(16, 0);
            Outcome::Ok
        }
        0x1b => {
            a.set64(0, 0);
            Outcome::Ok
        }
        0x1c => {
            a.set64(0, crate::gpu::gpu_ticks());
            a.set64(8, 0);
            Outcome::Ok
        }
        0x1d => {
            let count = (a.u32(256) as usize).min(16);
            for i in 0..count {
                let ticks = crate::gpu::gpu_ticks();
                a.set64(16 * i, ticks / 52);
                a.set64(16 * i + 8, ticks);
            }
            Outcome::Ok
        }
        _ => Outcome::Unimplemented,
    }
}

fn channel(env: &Env, syncpoint: u32, cmd: u32, a: &mut Args, extra: &[u8]) -> Outcome {
    let gpu = env.gpu;
    let group = (cmd >> 8) & 0xff;
    let nr = cmd & 0xff;
    match nr {
        0x01 if group == 0x48 => Outcome::Ok,
        0x01 | 0x24 => legacy_submit(env, a),
        0x02 => {
            a.set32(4, gpu.syncpoint(syncpoint));
            Outcome::Ok
        }
        0x03 if group == 0x48 => Outcome::Ok,
        0x03 | 0x04 => {
            a.set32(4, 0);
            Outcome::Ok
        }
        0x05 if group == 0x48 => Outcome::Ok,
        0x06 if group == 0x48 => channel_wait(env, a),
        0x07 => Outcome::Ok,
        0x08 if group == 0x48 => submit_gpfifo(env, syncpoint, a, extra),
        0x08 => Outcome::Ok,
        0x09 if group == 0x48 => alloc_obj_ctx(a),
        0x09 | 0x25 => map_cmd_buffer(env, a),
        0x0a if group == 0x48 => Outcome::Fail(NV_NOT_SUPPORTED),
        0x0a | 0x26 => unmap_cmd_buffer(env, a),
        0x0b if group == 0x48 => Outcome::Ok,
        0x0c => Outcome::Ok,
        0x0d => Outcome::Ok,
        0x0e | 0x0f => Outcome::Ok,
        0x10 | 0x11 => {
            gpu.wait_idle();
            Outcome::Ok
        }
        0x12 => Outcome::Ok,
        0x13 => Outcome::Ok,
        0x14 if group == 0x47 => {
            env.st.user_data.lock().unwrap().insert(syncpoint, a.u64(0));
            Outcome::Ok
        }
        0x14 | 0x23 => {
            a.set32(0, GPU_CLK_RATE);
            Outcome::Ok
        }
        0x15 if group == 0x47 => {
            a.set64(
                0,
                env.st
                    .user_data
                    .lock()
                    .unwrap()
                    .get(&syncpoint)
                    .copied()
                    .unwrap_or(0),
            );
            Outcome::Ok
        }
        0x16 => {
            a.fill(0, 128, 0);
            Outcome::Ok
        }
        0x17 => {
            a.set64(0, crate::gpu::gpu_ticks());
            a.set32(8, 0);
            a.set16(0xc, 0);
            a.set16(0xe, 0xffff);
            Outcome::Ok
        }
        0x18 => alloc_gpfifo_ex(env, syncpoint, a),
        0x19 | 0x1b | 0x1c => submit_gpfifo(env, syncpoint, a, extra),
        0x1a => alloc_gpfifo_ex2(env, syncpoint, a),
        0x1d => Outcome::Ok,
        _ => Outcome::Unimplemented,
    }
}

fn alloc_gpfifo_ex(env: &Env, syncpoint: u32, a: &mut Args) -> Outcome {
    if a.u32(0) == 0 {
        return Outcome::Fail(NV_BAD_VALUE);
    }
    a.set32(12, syncpoint);
    a.set32(16, env.gpu.syncpoint(syncpoint));
    Outcome::Ok
}

fn alloc_gpfifo_ex2(env: &Env, syncpoint: u32, a: &mut Args) -> Outcome {
    alloc_gpfifo_ex(env, syncpoint, a)
}

fn alloc_obj_ctx(a: &mut Args) -> Outcome {
    let class = a.u32(0);
    if class == 0 {
        return Outcome::Fail(NV_BAD_VALUE);
    }
    a.set64(8, class as u64);
    Outcome::Ok
}

fn channel_wait(env: &Env, a: &mut Args) -> Outcome {
    let (kind, handle, offset, payload) = (a.u32(0), a.u32(8), a.u32(12) as u64, a.u32(16));
    if kind != 1 {
        return Outcome::Ok;
    }
    let Some(m) = env.gpu.nvmap(handle) else {
        return Outcome::Fail(NV_BAD_VALUE);
    };
    env.gpu.wait_idle();
    let Some(bytes) = env.mem.read(m.addr + offset, 4) else {
        return Outcome::Fail(NV_BAD_VALUE);
    };
    if u32::from_le_bytes(bytes.try_into().unwrap()) == payload {
        Outcome::Fail(NV_TIMEOUT)
    } else {
        Outcome::Ok
    }
}

fn map_cmd_buffer(env: &Env, a: &mut Args) -> Outcome {
    let count = a.u32(0) as usize;
    if a.len() < 12 + 8 * count {
        return Outcome::Fail(NV_BAD_PARAMETER);
    }
    for i in 0..count {
        let handle = a.u32(12 + 8 * i);
        let Some(m) = env.gpu.nvmap(handle) else {
            return Outcome::Fail(NV_BAD_VALUE);
        };
        let address = env
            .gpu
            .gpu_addresses_of(m.addr)
            .first()
            .copied()
            .unwrap_or(m.addr);
        a.set32(16 + 8 * i, address as u32);
    }
    Outcome::Ok
}

fn unmap_cmd_buffer(env: &Env, a: &mut Args) -> Outcome {
    let count = a.u32(0) as usize;
    if a.len() < 12 + 8 * count {
        return Outcome::Fail(NV_BAD_PARAMETER);
    }
    for i in 0..count {
        if env.gpu.nvmap(a.u32(12 + 8 * i)).is_none() {
            return Outcome::Fail(NV_BAD_VALUE);
        }
    }
    Outcome::Ok
}

fn engine(env: &Env, fd: u32, class: u32, cmd: u32, a: &mut Args) -> Outcome {
    let nr = cmd & 0xff;
    let group = (cmd >> 8) & 0xff;
    match group {
        0x00 => {}
        0x48 if matches!(nr, 0x01 | 0x03) => return Outcome::Ok,
        0x48 if nr == 0x06 => return channel_wait(env, a),
        0x47 if nr == 0x14 => return Outcome::Ok,
        _ => return Outcome::Unimplemented,
    }
    let gpu = env.gpu;
    match nr {
        0x01 | 0x24 => {
            let Some(c) = env.cpu else {
                return Outcome::Unimplemented;
            };
            engine_submit(env, c, fd, class, a)
        }
        0x02 => {
            let id = *env
                .st
                .engine_syncpoints
                .lock()
                .unwrap()
                .entry((class, 0))
                .or_insert_with(|| gpu.alloc_syncpoint());
            a.set32(4, id);
            Outcome::Ok
        }
        0x03 | 0x04 => {
            a.set32(4, 0);
            Outcome::Ok
        }
        0x07 | 0x08 | 0x13 => Outcome::Ok,
        0x09 | 0x25 => {
            for i in 0..a.u32(0) as usize {
                let handle = a.u32(12 + 8 * i);
                let Some(m) = gpu.nvmap(handle) else {
                    return Outcome::Fail(NV_BAD_VALUE);
                };
                let address = match env.host1x.and_then(|h| h.map_buffer(handle, m)) {
                    Some(iova) => iova,
                    None => gpu
                        .gpu_addresses_of(m.addr)
                        .first()
                        .copied()
                        .unwrap_or(m.addr) as u32,
                };
                a.set32(16 + 8 * i, address);
            }
            Outcome::Ok
        }
        0x0a | 0x26 => {
            let count = a.u32(0) as usize;
            if a.len() < 12 + 8 * count {
                return Outcome::Fail(NV_BAD_PARAMETER);
            }
            for i in 0..count {
                if gpu.nvmap(a.u32(12 + 8 * i)).is_none() {
                    return Outcome::Fail(NV_BAD_VALUE);
                }
            }
            Outcome::Ok
        }
        0x23 => {
            a.set32(0, GPU_CLK_RATE);
            Outcome::Ok
        }
        _ => Outcome::Unimplemented,
    }
}

#[cfg(test)]
mod tests;
