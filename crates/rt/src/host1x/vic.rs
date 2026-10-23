use super::{bits, block_linear_size, from_block_linear, pitch, to_block_linear, trace, Dma};

mod reg {
    pub const EXECUTE: u32 = 0x300 / 4;
    pub const SURFACES: usize = 0x400 / 4;
    pub const CONFIG_STRUCT: usize = 0x708 / 4;
    pub const OUTPUT_SURFACE: usize = 0x720 / 4;
    pub const COUNT: usize = 0x800 / 4;
}

mod config {
    pub const OUTPUT: usize = 0x10;
    pub const OUTPUT_SURFACE: usize = 0x20;
    pub const SLOTS: usize = 0x90;
    pub const SLOT_SIZE: usize = 0xb0;
    pub const SIZE: usize = SLOTS + 8 * SLOT_SIZE;
    pub const SLOT_SURFACE: usize = 0x40;
    pub const SLOT_COLOR_MATRIX: usize = 0x60;
}

mod format {
    pub const A8B8G8R8: u32 = 31;
    pub const A8R8G8B8: u32 = 32;
    pub const X8B8G8R8: u32 = 35;
    pub const X8R8G8B8: u32 = 36;
    pub const Y8_V8U8_N420: u32 = 68;
}

#[cfg(test)]
type Pixel = [u16; 4];

pub struct Vic {
    regs: Vec<u32>,
    compositions: u64,
}

impl Default for Vic {
    fn default() -> Self {
        Vic { regs: vec![0; reg::COUNT], compositions: 0 }
    }
}

struct SurfaceConfig {
    format: u32,
    block_linear: bool,
    block_height_log2: u32,
    luma: (usize, usize),
    chroma: (usize, usize),
}

impl SurfaceConfig {
    fn parse(b: &[u8]) -> SurfaceConfig {
        SurfaceConfig {
            format: bits(b, 0, 7),
            block_linear: bits(b, 11, 4) != 0,
            block_height_log2: bits(b, 15, 4),
            luma: (bits(b, 64, 14) as usize + 1, bits(b, 78, 14) as usize + 1),
            chroma: (bits(b, 96, 14) as usize + 1, bits(b, 110, 14) as usize + 1),
        }
    }

    fn read_plane(&self, dma: &Dma, offset: u32, width_bytes: usize, height: usize) -> Option<Vec<u8>> {
        if self.block_linear {
            let raw = dma.read(offset, block_linear_size(width_bytes, height, self.block_height_log2))?;
            Some(from_block_linear(&raw, width_bytes, height, self.block_height_log2))
        } else {
            let p = pitch(width_bytes);
            let raw = dma.read(offset, p * (height - 1) + width_bytes)?;
            Some(raw.chunks(p).flat_map(|r| &r[..width_bytes.min(r.len())]).copied().collect())
        }
    }

    fn write_plane(&self, dma: &Dma, offset: u32, rows: &[u8], width_bytes: usize, height: usize) -> bool {
        if self.block_linear {
            dma.write(offset, &to_block_linear(rows, width_bytes, width_bytes, height, self.block_height_log2))
        } else {
            let p = pitch(width_bytes);
            let mut out = vec![0u8; p * height];
            for (d, s) in out.chunks_mut(p).zip(rows.chunks(width_bytes)) {
                d[..width_bytes].copy_from_slice(s);
            }
            dma.write(offset, &out)
        }
    }
}

struct Matrix {
    m: [[i64; 4]; 3],
    shift: u32,
    enabled: bool,
}

impl Matrix {
    fn parse(b: &[u8]) -> Matrix {
        let s = |lsb: usize| ((bits(b, lsb, 20) << 12) as i32 >> 12) as i64;
        Matrix {
            m: [[s(0), s(64), s(128), s(192)], [s(20), s(84), s(148), s(212)], [s(40), s(104), s(168), s(232)]],
            shift: bits(b, 60, 4),
            enabled: bits(b, 127, 1) != 0,
        }
    }

    #[cfg(test)]
    fn apply(&self, p: Pixel, lo: u16, hi: u16) -> Pixel {
        if !self.enabled {
            return [p[0].clamp(lo, hi), p[1].clamp(lo, hi), p[2].clamp(lo, hi), p[3]];
        }
        let (x, y, z) = (p[0] as i64, p[1] as i64, p[2] as i64);
        let c = |r: &[i64; 4]| ((((r[0] * x + r[1] * y + r[2] * z) >> self.shift) + r[3]) >> 8).clamp(lo as i64, hi as i64) as u16;
        [c(&self.m[0]), c(&self.m[1]), c(&self.m[2]), p[3]]
    }
}

impl Vic {
    pub fn write(&mut self, dma: &Dma, method: u32, value: u32) -> Result<(), String> {
        if let Some(r) = self.regs.get_mut(method as usize) {
            *r = value;
        }
        if method == reg::EXECUTE {
            return self.execute(dma);
        }
        Ok(())
    }

    fn execute(&mut self, dma: &Dma) -> Result<(), String> {
        let at = self.regs[reg::CONFIG_STRUCT];
        let cfg = dma.read(at, config::SIZE).ok_or_else(|| format!("VIC: config struct {at:#x} is not mapped"))?;
        let out_cfg = &cfg[config::OUTPUT..config::OUTPUT + 0x10];
        let out_surface = &cfg[config::OUTPUT_SURFACE..config::OUTPUT_SURFACE + 0x10];
        let surface = SurfaceConfig::parse(out_surface);
        let order: [usize; 4] = match surface.format {
            format::A8B8G8R8 | format::X8B8G8R8 | format::Y8_V8U8_N420 => [0, 1, 2, 3],
            format::A8R8G8B8 | format::X8R8G8B8 => [2, 1, 0, 3],
            f => return Err(format!("VIC: output format {f} is not supported")),
        };
        let (width, height) = (bits(out_surface, 32, 14) as usize + 1, bits(out_surface, 46, 14) as usize + 1);
        let background = [bits(out_cfg, 16, 10), bits(out_cfg, 26, 10), bits(out_cfg, 36, 10), bits(out_cfg, 6, 10)].map(|c| (c >> 2) as u8);
        let mut out: Vec<u8> = order.map(|i| background[i]).repeat(width * height);
        let (l, r) = (bits(out_cfg, 64, 14) as usize, bits(out_cfg, 80, 14) as usize + 1);
        let (t, b) = (bits(out_cfg, 96, 14) as usize, bits(out_cfg, 112, 14) as usize + 1);
        let (tx, ty) = (l.min(r), t.min(b));
        let (tw, th) = (width.saturating_sub(tx).min(l.abs_diff(r)), height.saturating_sub(ty).min(t.abs_diff(b)));
        let mut slots = 0;
        let mut luma_mean = 0u64;
        for slot in 0..8 {
            let s = &cfg[config::SLOTS + slot * config::SLOT_SIZE..][..config::SLOT_SIZE];
            if bits(s, 0, 1) == 0 {
                continue;
            }
            slots += 1;
            let src = self.read_slot(dma, slot, &SurfaceConfig::parse(&s[config::SLOT_SURFACE..]))?;
            if trace() {
                luma_mean = src.y.iter().map(|&v| v as u64).sum::<u64>() / src.y.len().max(1) as u64;
            }
            let (lo, hi) = (bits(s, 128, 10) as u16, bits(s, 138, 10) as u16);
            let hi = if hi == 0 { 0x3ff } else { hi };
            let convert = Converter::new(&Matrix::parse(&s[config::SLOT_COLOR_MATRIX..]), lo, hi);
            let (x_end, y_end) = ((tx + tw).min(src.width), (ty + th).min(src.height));
            for y in ty..y_end {
                let luma = &src.y[y * src.width..][..src.width];
                let chroma = &src.uv[(y / 2).min(src.chroma.1 - 1) * src.chroma.0 * 2..][..src.chroma.0 * 2];
                let row = &mut out[y * width * 4..][..width * 4];
                for x in tx..x_end {
                    let c = (x / 2).min(src.chroma.0 - 1) * 2;
                    let [r, g, b] = convert.pixel(luma[x], chroma[c], chroma[c + 1]);
                    let rgba = [r, g, b, 0xff];
                    row[x * 4..x * 4 + 4].copy_from_slice(&order.map(|i| rgba[i]));
                }
            }
        }
        if bits(out_cfg, 48, 1) != 0 {
            for row in out.chunks_exact_mut(width * 4) {
                row.as_chunks_mut::<4>().0.reverse();
            }
        }
        if bits(out_cfg, 49, 1) != 0 {
            out = out.chunks_exact(width * 4).rev().flatten().copied().collect();
        }
        let offset = self.regs[reg::OUTPUT_SURFACE];
        self.compositions += 1;
        if trace() {
            eprintln!(
                "vic: {:.3} s: composition {}: {slots} slot(s), mean luma {luma_mean} -> {width}x{height} format {} {} (block height {}) at {:#x}",
                crate::gpu::gpu_ticks() as f64 / 1e9,
                self.compositions,
                surface.format,
                if surface.block_linear { "block-linear" } else { "pitch" },
                1 << surface.block_height_log2,
                (offset as u64) << 8
            );
        }
        let written = if surface.format == format::Y8_V8U8_N420 {
            let (y, uv) = to_nv12(&out, width, height, surface.chroma);
            let (cw, ch) = surface.chroma;
            surface.write_plane(dma, offset, &y, width, height) && surface.write_plane(dma, self.regs[reg::OUTPUT_SURFACE + 1], &uv, cw * 2, ch)
        } else {
            surface.write_plane(dma, offset, &out, width * 4, height)
        };
        if !written {
            return Err(format!("VIC: output surface {:#x} is not mapped", (offset as u64) << 8));
        }
        Ok(())
    }

    fn read_slot(&self, dma: &Dma, slot: usize, s: &SurfaceConfig) -> Result<Nv12, String> {
        if s.format != format::Y8_V8U8_N420 {
            return Err(format!("VIC: slot format {} is not supported", s.format));
        }
        let base = reg::SURFACES + slot * 8 * 3;
        let (luma_at, chroma_at) = (self.regs[base], self.regs[base + 1]);
        let ((width, height), chroma) = (s.luma, s.chroma);
        let missing = || format!("VIC: slot {slot} surface {:#x} is not mapped", (luma_at as u64) << 8);
        let y = s.read_plane(dma, luma_at, width, height).ok_or_else(missing)?;
        let uv = s.read_plane(dma, chroma_at, chroma.0 * 2, chroma.1).ok_or_else(missing)?;
        Ok(Nv12 { width, height, chroma, y, uv })
    }
}

fn to_nv12(yuva: &[u8], width: usize, height: usize, chroma: (usize, usize)) -> (Vec<u8>, Vec<u8>) {
    let y: Vec<u8> = yuva.chunks_exact(4).map(|p| p[0]).collect();
    let (cw, ch) = chroma;
    let mut uv = vec![0u8; cw * ch * 2];
    for cy in 0..ch {
        let sy = (2 * cy).min(height - 1);
        for cx in 0..cw {
            let p = &yuva[(sy * width + (2 * cx).min(width - 1)) * 4..][..4];
            uv[(cy * cw + cx) * 2] = p[1];
            uv[(cy * cw + cx) * 2 + 1] = p[2];
        }
    }
    (y, uv)
}

struct Nv12 {
    width: usize,
    height: usize,
    chroma: (usize, usize),
    y: Vec<u8>,
    uv: Vec<u8>,
}

struct Converter {
    products: [[[i64; 256]; 3]; 3],
    offsets: [i64; 3],
    shift: u32,
    enabled: bool,
    lo: i64,
    hi: i64,
}

impl Converter {
    fn new(m: &Matrix, lo: u16, hi: u16) -> Converter {
        let products = std::array::from_fn(|c| std::array::from_fn(|i| std::array::from_fn(|v| m.m[c][i] * ((v as i64) << 2))));
        Converter { products, offsets: [m.m[0][3], m.m[1][3], m.m[2][3]], shift: m.shift, enabled: m.enabled, lo: lo as i64, hi: hi as i64 }
    }

    #[inline]
    fn pixel(&self, y: u8, u: u8, v: u8) -> [u8; 3] {
        let (y, u, v) = (y as usize, u as usize, v as usize);
        if !self.enabled {
            return [y, u, v].map(|s| (((s as i64) << 2).clamp(self.lo, self.hi) >> 2) as u8);
        }
        std::array::from_fn(|c| {
            let p = &self.products[c];
            (((((p[0][y] + p[1][u] + p[2][v]) >> self.shift) + self.offsets[c]) >> 8).clamp(self.lo, self.hi) >> 2) as u8
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_output_planes() {
        let px: Vec<u8> = (0..2).flat_map(|y| (0..4).flat_map(move |x| [x as u8, 100 + x as u8, 200 + y as u8, 255])).collect();
        let (luma, uv) = to_nv12(&px, 4, 2, (2, 1));
        assert_eq!(luma, [0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(uv, [100, 200, 102, 200]);
    }

    #[test]
    fn color_matrix() {
        let mut b = [0u8; 32];
        let mut put = |lsb: usize, v: i64| {
            let v = (v as u64) & 0xfffff;
            for i in 0..20 {
                if v >> i & 1 != 0 {
                    b[(lsb + i) / 8] |= 1 << ((lsb + i) % 8);
                }
            }
        };
        let k = |f: f64| (f * 1024.0).round() as i64;
        put(0, k(1.164));
        put(64, 0);
        put(128, k(1.596));
        put(192, ((-1.164 * 64.0 - 1.596 * 512.0) * 256.0) as i64);
        put(20, k(1.164));
        put(84, k(-0.392));
        put(148, k(-0.813));
        put(212, ((-1.164 * 64.0 + 0.392 * 512.0 + 0.813 * 512.0) * 256.0) as i64);
        put(40, k(1.164));
        put(104, k(2.017));
        put(168, 0);
        put(232, ((-1.164 * 64.0 - 2.017 * 512.0) * 256.0) as i64);
        b[7] |= 2 << 4;
        b[15] |= 0x80;
        let m = Matrix::parse(&b);
        assert!(m.enabled);
        assert_eq!(m.shift, 2);
        let white = m.apply([940, 512, 512, 0x3ff], 0, 0x3ff);
        assert!(white[..3].iter().all(|&c| c >= 1015), "{white:?}");
        let black = m.apply([64, 512, 512, 0x3ff], 0, 0x3ff);
        assert!(black[..3].iter().all(|&c| c <= 2), "{black:?}");
        let conv = Converter::new(&m, 16, 1000);
        for (y, u, v) in [(0u8, 0u8, 0u8), (235, 128, 128), (81, 90, 240), (145, 54, 34), (255, 255, 255), (16, 200, 30)] {
            let want = m.apply([(y as u16) << 2, (u as u16) << 2, (v as u16) << 2, 0x3ff], 16, 1000);
            assert_eq!(conv.pixel(y, u, v), [0, 1, 2].map(|i| (want[i] >> 2) as u8));
        }
    }
}
