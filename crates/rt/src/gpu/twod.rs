use std::sync::atomic::Ordering;

use super::vulkan::Surface;
use super::{Gpu, trace};
use crate::mem::GuestMemory;

pub const PIXELS_FROM_MEMORY_SRC_Y0_INT: u32 = 0x237;

fn surface(regs: &[u32], base: usize) -> Surface {
    let (pitch, width, height, upper) = if base == 0x80 { (0x85, 0x86, 0x87, 0x88) } else { (0x91, 0x92, 0x93, 0x94) };
    let pitch_layout = regs[base + 1] & 1 == 1;
    Surface {
        va: (regs[upper] as u64 & 0xff) << 32 | regs[upper + 1] as u64,
        format: regs[base],
        width: regs[width],
        height: regs[height],
        block_height_log2: (regs[base + 2] >> 4) & 7,
        pitch: pitch_layout.then_some(regs[pitch]),
    }
}

fn fixed(int: u32, frac: u32) -> f64 {
    int as i32 as f64 + frac as f64 / 4294967296.0
}

impl Gpu {
    pub(super) fn blit_2d(&self, mem: &GuestMemory, regs: &[u32]) {
        self.stats.copies.fetch_add(1, Ordering::Relaxed);
        let dst = surface(regs, 0x80);
        let src = surface(regs, 0x8c);
        let (dx, dy, dw, dh) = (regs[0x22c] as i32, regs[0x22d] as i32, regs[0x22e] as i32, regs[0x22f] as i32);
        let du_dx = fixed(regs[0x231], regs[0x230]);
        let dv_dy = fixed(regs[0x233], regs[0x232]);
        let sx = fixed(regs[0x235], regs[0x234]);
        let sy = fixed(regs[0x237], regs[0x236]);
        let src_rect = [sx as f32, sy as f32, (sx + dw as f64 * du_dx) as f32, (sy + dh as f64 * dv_dy) as f32];
        let dst_rect = [dx, dy, dx + dw, dy + dh];
        let linear = regs[0x223] >> 4 & 1 == 1;
        if trace() {
            eprintln!("gpu: 2D blit {src:x?} {src_rect:?} -> {dst:x?} {dst_rect:?} (filter {})", if linear { "linear" } else { "point" });
        }
        let done = self.with_renderer(|r| r.blit(self, mem, &src, &dst, src_rect, dst_rect, linear)).unwrap_or(false);
        if !done && trace() {
            eprintln!("gpu: 2D blit not performed (surface format {:#x} -> {:#x})", src.format, dst.format);
        }
    }
}
