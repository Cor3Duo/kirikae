use std::sync::atomic::Ordering;

use super::threed::address;
use super::{trace, Gpu};
use crate::mem::GuestMemory;

pub mod reg {
    pub const QMD_ADDRESS: usize = 0xad;
    pub const LAUNCH: u32 = 0xaf;
    pub const LOCAL_MEMORY: usize = 0x1e4;
    pub const TEX_SAMPLER_POOL: usize = 0x557;
    pub const TEX_HEADER_POOL: usize = 0x55d;
    pub const PROGRAM_REGION: usize = 0x582;
    pub const BINDLESS_TEXTURE: usize = 0x982;
}

pub const QMD_CBUFS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qmd {
    pub program: u32,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared: u32,
    pub local: u32,
    pub cbufs: [Option<(u64, u32)>; QMD_CBUFS],
}

fn bits(w: &[u32], lo: usize, hi: usize) -> u32 {
    let n = hi - lo + 1;
    let word = w[lo / 32] as u64 | (w.get(lo / 32 + 1).copied().unwrap_or(0) as u64) << 32;
    ((word >> (lo % 32)) & ((1u64 << n) - 1)) as u32
}

impl Qmd {
    pub fn parse(w: &[u32]) -> Qmd {
        let mut cbufs = [None; QMD_CBUFS];
        for (i, cb) in cbufs.iter_mut().enumerate() {
            if bits(w, 640 + i, 640 + i) == 1 {
                let lo = bits(w, 928 + i * 64, 959 + i * 64) as u64;
                let hi = bits(w, 960 + i * 64, 967 + i * 64) as u64;
                let size = bits(w, 975 + i * 64, 991 + i * 64);
                *cb = Some((hi << 32 | lo, size));
            }
        }
        Qmd {
            program: bits(w, 256, 287),
            grid: [bits(w, 384, 415), bits(w, 416, 431), bits(w, 432, 447)],
            block: [bits(w, 592, 607), bits(w, 608, 623), bits(w, 624, 639)],
            shared: bits(w, 544, 561),
            local: bits(w, 1440, 1463) + bits(w, 1472, 1495),
            cbufs,
        }
    }

    pub fn threads(&self) -> u64 {
        self.block.iter().map(|&b| b.max(1) as u64).product()
    }
}

impl Gpu {
    pub(super) fn dispatch(&self, mem: &GuestMemory, regs: &[u32]) {
        self.stats.dispatches.fetch_add(1, Ordering::Relaxed);
        let qmd_va = (regs[reg::QMD_ADDRESS] as u64) << 8;
        let Some(raw) = self.read(mem, qmd_va, 0x100) else { return };
        let words: Vec<u32> = raw.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        let qmd = Qmd::parse(&words);
        if crate::nvn::shadow::ON.load(Ordering::Relaxed) {
            crate::nvn::shadow::cp_dispatch(qmd.grid);
        }
        let va = address(regs, reg::PROGRAM_REGION) + qmd.program as u64;
        if trace() {
            eprintln!("gpu: dispatch program {va:#x} grid {:?} block {:?} shared {:#x} local {:#x} cbufs {:x?} bindless slot {} tic {:#x} tsc {:#x}", qmd.grid, qmd.block, qmd.shared, qmd.local, qmd.cbufs, regs[reg::BINDLESS_TEXTURE], address(regs, reg::TEX_HEADER_POOL), address(regs, reg::TEX_SAMPLER_POOL));
        }
        if let Some(dir) = crate::env_var!("KIRIKAE_SHADER_DUMP") {
            let path = std::path::Path::new(&dir).join(format!("{va:010x}_compute.bin"));
            if !path.exists()
                && let Some(code) = self.read(mem, va, 0x8000)
            {
                let mut out = vec![0u8; 0x50];
                out.extend_from_slice(&code);
                let _ = std::fs::write(path, out);
            }
        }
        self.with_renderer(|r| r.dispatch(self, mem, regs, &qmd, va));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qmd_fields() {
        let mut w = [0u32; 64];
        w[8] = 0x1230;
        w[12] = 40;
        w[13] = 3 | 2 << 16;
        w[17] = 0x800;
        w[18] = 8 << 16;
        w[19] = 4 | 1 << 16;
        w[20] = 0b101;
        w[29] = 0x4000_0000;
        w[30] = 0x2 | 0x100 << 15;
        w[33] = 0x5000;
        w[34] = 0x40 << 15;
        let q = Qmd::parse(&w);
        assert_eq!(q.program, 0x1230);
        assert_eq!(q.grid, [40, 3, 2]);
        assert_eq!(q.block, [8, 4, 1]);
        assert_eq!(q.shared, 0x800);
        assert_eq!(q.cbufs[0], Some((0x2_4000_0000, 0x100)));
        assert_eq!(q.cbufs[1], None);
        assert_eq!(q.cbufs[2], Some((0x5000, 0x40)));
    }
}
