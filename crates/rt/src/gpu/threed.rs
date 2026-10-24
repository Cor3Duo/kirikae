use std::sync::atomic::Ordering;

use super::{Engines, Gpu, trace};
use crate::mem::GuestMemory;

pub const MACRO_CALL: u32 = 0xe00;

pub mod reg {
    pub const LOAD_MME_INSTRUCTION_RAM_POINTER: u32 = 0x045;
    pub const LOAD_MME_INSTRUCTION_RAM: u32 = 0x046;
    pub const LOAD_MME_START_ADDRESS_RAM_POINTER: u32 = 0x047;
    pub const LOAD_MME_START_ADDRESS_RAM: u32 = 0x048;
    pub const VIEW_VOLUME_CLIP_CTRL: usize = 0x64f;
    pub const VB_ELEMENT_BASE: usize = 0x50d;
    pub const ALPHA_TEST_ENABLE: usize = 0x4bb;
    pub const ALPHA_TEST_REF: usize = 0x4c4;
    pub const ALPHA_TEST_FUNC: usize = 0x4c5;
    pub const COLOR_TARGET: usize = 0x200;
    pub const VIEWPORT_SCALE: usize = 0x280;
    pub const VIEWPORT_CLIP: usize = 0x300;
    pub const VERTEX_ARRAY_START: usize = 0x35d;
    pub const DRAW_VERTEX_ARRAY: u32 = 0x35e;
    pub const CLEAR_COLOR: usize = 0x360;
    pub const SCISSOR: usize = 0x380;
    pub const ZT_A: usize = 0x3f8;
    pub const VERTEX_ATTRIBUTE: usize = 0x458;
    pub const DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST: u32 = 0x485;
    pub const DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT: u32 = 0x486;
    pub const CT_SELECT: usize = 0x487;
    pub const TEX_SAMPLER_POOL: usize = 0x557;
    pub const TEX_HEADER_POOL: usize = 0x55d;
    pub const PROGRAM_REGION: usize = 0x582;
    pub const END: u32 = 0x585;
    pub const BEGIN: u32 = 0x586;
    pub const INDEX_BUFFER: usize = 0x5f2;
    pub const PRIMITIVE_RESTART: usize = 0x591;
    pub const DRAW_INDEX_BUFFER: u32 = 0x5f8;
    pub const DRAW_INDEX_BUFFER_INLINE: std::ops::RangeInclusive<u32> = 0x5f9..=0x5fe;
    pub const CLEAR_SURFACE: u32 = 0x674;
    pub const REPORT_SEMAPHORE: usize = 0x6c0;
    pub const REPORT_SEMAPHORE_D: u32 = 0x6c3;
    pub const VERTEX_STREAM: usize = 0x700;
    pub const VERTEX_STREAM_LIMIT: usize = 0x7c0;
    pub const PIPELINE: usize = 0x800;
    pub const CONSTANT_BUFFER_SELECTOR: usize = 0x8e0;
    pub const LOAD_CONSTANT_BUFFER_OFFSET: usize = 0x8e3;
    pub const LOAD_CONSTANT_BUFFER: std::ops::RangeInclusive<u32> = 0x8e4..=0x8f3;
    pub const BIND_GROUP_CONSTANT_BUFFER: u32 = 0x904;
    pub const BINDLESS_TEXTURE: usize = 0x982;
}

const OCCLUSION_VISIBLE: u64 = 1 << 24;

pub const STAGES: usize = 6;
pub const STAGE_NAMES: [&str; STAGES] = ["vertex A", "vertex B", "tess control", "tess eval", "geometry", "fragment"];

pub fn stage_enabled(regs: &[u32], stage: usize) -> bool {
    stage == 1 || regs[reg::PIPELINE + 16 * stage] & 1 == 1
}
pub const CB_SLOTS: usize = 18;

pub struct Macros {
    pub code: std::sync::Arc<Vec<u32>>,
    pub starts: Vec<u32>,
    code_pointer: usize,
    start_pointer: usize,
    pub pending: Option<PendingMacro>,
    pub spare: Vec<u32>,
}

impl Default for Macros {
    fn default() -> Self {
        Macros { code: std::sync::Arc::new(vec![0; 0x2000]), starts: vec![0; 0x80], code_pointer: 0, start_pointer: 0, pending: None, spare: Vec::new() }
    }
}

pub struct PendingMacro {
    pub sub: u32,
    pub index: u32,
    pub params: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub topology: u32,
    pub first: u32,
    pub count: u32,
    pub index_size: Option<u32>,
    pub instance: u32,
}

#[derive(Default)]
pub struct DrawTracker {
    topology: u32,
    pending: Option<(u32, u32, Option<u32>)>,
    instance: u32,
}

impl DrawTracker {
    fn step_instance(&mut self, mode: u32) -> u32 {
        match mode {
            0 => self.instance = 0,
            1 => self.instance = self.instance.wrapping_add(1),
            _ => {}
        }
        self.instance
    }
}

#[derive(Default)]
pub struct State {
    pub macros: Macros,
    pub draw: DrawTracker,
    pub constant_buffers: [[Option<(u64, u32)>; CB_SLOTS]; 5],
}

pub fn address(regs: &[u32], at: usize) -> u64 {
    (regs[at] as u64 & 0xff) << 32 | regs[at + 1] as u64
}

impl Gpu {
    pub(super) fn threed(&self, mem: &GuestMemory, e: &mut Engines, m: u32, v: u32) {
        match m {
            reg::LOAD_MME_INSTRUCTION_RAM_POINTER => e.threed.macros.code_pointer = v as usize,
            reg::LOAD_MME_INSTRUCTION_RAM => {
                let ptr = e.threed.macros.code_pointer;
                let code = std::sync::Arc::make_mut(&mut e.threed.macros.code);
                if let Some(w) = code.get_mut(ptr) {
                    *w = v;
                }
                e.threed.macros.code_pointer += 1;
            }
            reg::LOAD_MME_START_ADDRESS_RAM_POINTER => e.threed.macros.start_pointer = v as usize,
            reg::LOAD_MME_START_ADDRESS_RAM => {
                let ptr = e.threed.macros.start_pointer;
                if let Some(s) = e.threed.macros.starts.get_mut(ptr) {
                    *s = v;
                }
                e.threed.macros.start_pointer += 1;
            }
            m if reg::LOAD_CONSTANT_BUFFER.contains(&m) => {
                let regs = e.regs.get_mut(&super::CLASS_3D).expect("3D registers");
                let base = address(regs, reg::CONSTANT_BUFFER_SELECTOR + 1);
                let offset = regs[reg::LOAD_CONSTANT_BUFFER_OFFSET];
                regs[reg::LOAD_CONSTANT_BUFFER_OFFSET] = offset.wrapping_add(4);
                let at = base + offset as u64;
                let (start, data) = &mut e.cb_upload;
                if !data.is_empty() && *start + data.len() as u64 != at {
                    self.flush_cb_upload(mem, e);
                }
                let (start, data) = &mut e.cb_upload;
                if data.is_empty() {
                    *start = at;
                }
                data.extend_from_slice(&v.to_le_bytes());
            }
            m if (reg::BIND_GROUP_CONSTANT_BUFFER..reg::BIND_GROUP_CONSTANT_BUFFER + 5 * 8).contains(&m) && (m - reg::BIND_GROUP_CONSTANT_BUFFER).is_multiple_of(8) => {
                let group = ((m - reg::BIND_GROUP_CONSTANT_BUFFER) / 8) as usize;
                let slot = ((v >> 4) & 0x1f) as usize;
                let regs = &e.regs[&super::CLASS_3D];
                let binding = (v & 1 == 1).then(|| (address(regs, reg::CONSTANT_BUFFER_SELECTOR + 1), regs[reg::CONSTANT_BUFFER_SELECTOR]));
                if let Some(s) = e.threed.constant_buffers[group].get_mut(slot) {
                    *s = binding;
                }
            }
            reg::BEGIN => {
                e.threed.draw.topology = v & 0xffff;
                e.threed.draw.pending = None;
                e.threed.draw.step_instance(v >> 26 & 3);
            }
            reg::DRAW_VERTEX_ARRAY => {
                let first = e.regs[&super::CLASS_3D][reg::VERTEX_ARRAY_START];
                e.threed.draw.pending = Some((first, v, None));
            }
            reg::DRAW_INDEX_BUFFER => {
                let regs = &e.regs[&super::CLASS_3D];
                let size = 1 << (regs[reg::INDEX_BUFFER + 4] & 3);
                e.threed.draw.pending = Some((regs[reg::INDEX_BUFFER + 5], v, Some(size)));
            }
            reg::END => {
                if let Some((first, count, index_size)) = e.threed.draw.pending.take() {
                    let draw = Draw { topology: e.threed.draw.topology, first, count, index_size, instance: e.threed.draw.instance };
                    let cbs = e.threed.constant_buffers;
                    self.draw(mem, &e.regs[&super::CLASS_3D], &cbs, draw);
                }
            }
            reg::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST | reg::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT => {
                let instance = e.threed.draw.step_instance((m == reg::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT) as u32);
                let draw = Draw { topology: v >> 28, first: v & 0xffff, count: (v >> 16) & 0xfff, index_size: None, instance };
                let cbs = e.threed.constant_buffers;
                self.draw(mem, &e.regs[&super::CLASS_3D], &cbs, draw);
            }
            m if reg::DRAW_INDEX_BUFFER_INLINE.contains(&m) => {
                let size = [4, 2, 1][((m - 0x5f9) % 3) as usize];
                let instance = e.threed.draw.step_instance((m - 0x5f9 >= 3) as u32);
                let draw = Draw { topology: v >> 28, first: v & 0xffff, count: (v >> 16) & 0xfff, index_size: Some(size), instance };
                let cbs = e.threed.constant_buffers;
                self.draw(mem, &e.regs[&super::CLASS_3D], &cbs, draw);
            }
            reg::CLEAR_SURFACE => {
                self.clear(mem, &e.regs[&super::CLASS_3D], v);
            }
            reg::REPORT_SEMAPHORE_D => {
                self.report(mem, &e.regs[&super::CLASS_3D]);
            }
            _ => {}
        }
    }

    pub(super) fn draw(&self, mem: &GuestMemory, regs: &[u32], cbs: &[[Option<(u64, u32)>; CB_SLOTS]; 5], draw: Draw) {
        let n = self.stats.draws.fetch_add(1, Ordering::Relaxed);
        if crate::nvn::exec::CHECK.load(Ordering::Relaxed) {
            crate::nvn::exec::cp_draw(regs, cbs, &draw);
        }
        if trace() {
            let base = address(regs, reg::PROGRAM_REGION);
            let stages: Vec<String> = (0..STAGES)
                .filter(|&s| stage_enabled(regs, s))
                .map(|s| format!("{}@{:#x}", STAGE_NAMES[s], base + regs[reg::PIPELINE + 16 * s + 1] as u64))
                .collect();
            let rt = reg::COLOR_TARGET;
            eprintln!(
                "gpu: draw #{n} {draw:?} rt0 {:#x} {}x{} fmt {:#x} ct_select {:#x}; shaders {}; cb[fragment] {:?}",
                address(regs, rt),
                regs[rt + 2],
                regs[rt + 3],
                regs[rt + 4],
                regs[reg::CT_SELECT],
                stages.join(", "),
                cbs[4].iter().enumerate().filter_map(|(i, b)| b.map(|b| (i, b))).collect::<Vec<_>>()
            );
            if crate::env_var!("KIRIKAE_TRACE_REGS").is_some() {
                let nz: Vec<String> = regs.iter().enumerate().filter(|(_, v)| **v != 0).map(|(i, v)| format!("{i:x}={v:x}")).collect();
                eprintln!("gpu:   regs {}", nz.join(" "));
            }
            let ct = regs[reg::CT_SELECT];
            for k in 1..(ct & 0xf).min(8) {
                let base = reg::COLOR_TARGET + 16 * (ct >> (4 + 3 * k) & 7) as usize;
                eprintln!("gpu:   target {k}: {:#x} regs {:x?}", address(regs, base), &regs[base + 2..base + 9]);
            }
            if regs[reg::COLOR_TARGET + 6] & 0xffff > 1 || regs[reg::COLOR_TARGET + 8] != 0 {
                eprintln!("gpu:   target 0 layered: regs {:x?}", &regs[reg::COLOR_TARGET + 2..reg::COLOR_TARGET + 9]);
            }
        }
        if let Some(dir) = crate::env_var!("KIRIKAE_SHADER_DUMP") {
            let base = address(regs, reg::PROGRAM_REGION);
            for s in (0..STAGES).filter(|&s| stage_enabled(regs, s)) {
                let va = base + regs[reg::PIPELINE + 16 * s + 1] as u64;
                let path = std::path::Path::new(&dir).join(format!("{va:010x}_{}.bin", STAGE_NAMES[s].replace(' ', "_")));
                if !path.exists()
                    && let Some(code) = self.read(mem, va, 0x8000)
                {
                    let _ = std::fs::write(path, code);
                }
            }
        }
        if crate::nvn::shadow::ON.load(Ordering::Relaxed) {
            crate::nvn::shadow::cp_draw(regs, cbs, &|va, len| self.read(mem, va, len), &draw);
        }
        if crate::nvn::encode::ON.load(Ordering::Relaxed) {
            crate::nvn::encode::observe(regs);
        }
        self.with_renderer(|r| r.draw(self, mem, regs, cbs, draw));
    }

    pub(super) fn clear(&self, mem: &GuestMemory, regs: &[u32], v: u32) {
        if crate::nvn::exec::CHECK.load(Ordering::Relaxed) {
            let cbs = [[None; CB_SLOTS]; 5];
            crate::nvn::exec::cp_draw(regs, &cbs, &Draw { topology: crate::nvn::exec::CLEAR, first: v, count: 0, index_size: None, instance: 0 });
        }
        self.stats.clears.fetch_add(1, Ordering::Relaxed);
        if crate::nvn::shadow::ON.load(Ordering::Relaxed) {
            crate::nvn::shadow::cp_clear(regs, v);
        }
        if v & 3 != 0 {
            self.with_renderer(|r| r.clear_depth(self, regs, v & 1 == 1, v & 2 == 2));
        }
        if v >> 2 & 0xf == 0 {
            return;
        }
        let rt = (v >> 6 & 0xf) as usize;
        let base = reg::COLOR_TARGET + rt * 0x10;
        let addr = address(regs, base);
        let (width, height, format) = (regs[base + 2] as u64, regs[base + 3] as u64, regs[base + 4]);
        let color: Vec<f32> = (0..4).map(|k| f32::from_bits(regs[reg::CLEAR_COLOR + k])).collect();
        if trace() {
            eprintln!(
                "gpu: clear rt{rt} at {addr:#x} (guest {}) {width}x{height} format {format:#x} color {color:?} mask {:#x} layer {} flags {:#x} scissor {:#x} {:#x} {:#x}",
                self.translate(addr).map_or("unmapped".into(), |a| format!("{a:#x}")),
                v >> 2 & 0xf,
                v >> 10 & 0x7ff,
                regs[0x43e],
                regs[reg::SCISSOR],
                regs[reg::SCISSOR + 1],
                regs[reg::SCISSOR + 2]
            );
        }
        let surface = super::vulkan::color_surface(regs, base);
        let bits = [regs[reg::CLEAR_COLOR], regs[reg::CLEAR_COLOR + 1], regs[reg::CLEAR_COLOR + 2], regs[reg::CLEAR_COLOR + 3]];
        let rendered = self.with_renderer(|r| r.clear(self, mem, &surface, bits)).is_some();
        if rendered && crate::env_var!("KIRIKAE_CPU_CLEARS").is_none() {
            return;
        }
        let to8 = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let pixel: [u8; 4] = match format {
            0xd5 | 0xd6 => [to8(color[0]), to8(color[1]), to8(color[2]), to8(color[3])],
            0xcf | 0xd0 => [to8(color[2]), to8(color[1]), to8(color[0]), to8(color[3])],
            _ => return,
        };
        let block_h = 8u64 << (regs[base + 5] >> 4 & 7);
        let size = (width * 4).div_ceil(64) * 64 * height.div_ceil(block_h) * block_h;
        if size == 0 || size > 64 << 20 {
            return;
        }
        let fill: Vec<u8> = pixel.iter().copied().cycle().take(size as usize).collect();
        self.write(mem, addr, &fill);
    }

    pub(super) fn report(&self, mem: &GuestMemory, regs: &[u32]) {
        let addr = address(regs, reg::REPORT_SEMAPHORE);
        let (payload, control) = (regs[reg::REPORT_SEMAPHORE + 2], regs[reg::REPORT_SEMAPHORE + 3]);
        let value = match control & 3 {
            0 => payload as u64,
            2 => match control >> 23 & 0x1f {
                0x15 => OCCLUSION_VISIBLE,
                _ => 0,
            },
            _ => return,
        };
        if control >> 28 & 1 == 1 {
            self.write(mem, addr, &(value as u32).to_le_bytes());
        } else {
            let mut b = value.to_le_bytes().to_vec();
            b.extend_from_slice(&super::report_ticks().to_le_bytes());
            self.write(mem, addr, &b);
        }
    }
}
