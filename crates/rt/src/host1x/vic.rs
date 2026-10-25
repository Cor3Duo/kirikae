use super::surface::BitDepth;
use super::{
    Dma, bits, block_linear_size, clock, from_block_linear, pitch, to_block_linear, trace,
};

mod reg {
    pub const EXECUTE: u32 = 0x300 / 4;
    pub const SURFACES: usize = 0x400 / 4;
    pub const CONFIG_STRUCT: usize = 0x708 / 4;
    pub const OUTPUT_SURFACE: usize = 0x720 / 4;
    pub const COUNT: usize = 0x800 / 4;
}

mod config {
    /// Output configuration: background colour, transform and target rectangle.
    pub const OUTPUT: usize = 0x10;
    pub const OUTPUT_SURFACE: usize = 0x20;
    pub const SLOTS: usize = 0x90;
    pub const SLOT_SIZE: usize = 0xb0;
    pub const SIZE: usize = SLOTS + 8 * SLOT_SIZE;
    pub const SLOT_SURFACE: usize = 0x40;
    pub const SLOT_COLOR_MATRIX: usize = 0x60;
}

/// Bits of the output configuration word block.
mod output {
    /// Background alpha (10 bits), then the three colour channels.
    pub const ALPHA: usize = 6;
    pub const RED: usize = 16;
    pub const GREEN: usize = 26;
    pub const BLUE: usize = 36;
    pub const CHANNEL_BITS: usize = 10;
    /// Orientation: two flip bits followed by a 2-bit clockwise rotation count.
    pub const FLIP_H: usize = 48;
    pub const FLIP_V: usize = 49;
    pub const ROTATION: usize = 50;
    /// Destination rectangle in output pixels. Right/bottom are inclusive.
    pub const LEFT: usize = 64;
    pub const RIGHT: usize = 80;
    pub const TOP: usize = 96;
    pub const BOTTOM: usize = 112;
    pub const RECT_BITS: usize = 14;
}

/// Bits of a slot configuration (slot base is `config::SLOTS + slot * SLOT_SIZE`).
mod slot {
    pub const ENABLE: usize = 0;
    /// Selects nearest-neighbour scaling; bilinear is the default.
    pub const NEAREST: usize = 1;
    /// Luma/chroma range clamp, 10 bits each.
    pub const RANGE_LOW: usize = 128;
    pub const RANGE_HIGH: usize = 138;
    /// Composition: blend flag and 10-bit coverage alpha. Without the blend flag
    /// the slot is opaque, which is also what the hardware does for alpha-less
    /// sources such as NV12.
    pub const BLEND: usize = 0x80 * 8;
    pub const ALPHA: usize = BLEND + 1;
    pub const ALPHA_BITS: usize = 10;
}

mod format {
    pub const A8B8G8R8: u32 = 31;
    pub const A8R8G8B8: u32 = 32;
    pub const R5G6B5: u32 = 33;
    pub const A1R5G5B5: u32 = 34;
    pub const X8B8G8R8: u32 = 35;
    pub const X8R8G8B8: u32 = 36;
    pub const B8G8R8: u32 = 37;
    pub const R8G8B8: u32 = 38;
    /// Semi-planar 4:2:0: luma followed by interleaved U/V bytes.
    pub const Y8_V8U8_N420: u32 = 68;
    /// Planar 4:2:0: luma, then U, then V.
    ///
    /// The 4:2:0 codes come in planar/semi-planar pairs (68/69, 70/71, 72/73 in
    /// the pixel-format list). Only 68 has been observed in a trace; this one
    /// and the 10-bit code below are wired and covered by synthetic tests until
    /// a title asks for them.
    pub const Y8_U8_V8_N420: u32 = 69;
    /// 10-bit semi-planar 4:2:0 (P010), the 10-bit block's luma+interleaved
    /// chroma code.
    pub const Y10_V10U10_N420: u32 = 92;
}

/// Pixel formats a slot can read from its source surface.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SourceFormat {
    /// `Y8_V8U8_N420`: NV12, one byte per luma sample plus interleaved chroma.
    Nv12,
    /// `A8B8G8R8`: memory bytes `R G B A`.
    Rgba8,
    /// `A8R8G8B8`: memory bytes `B G R A`.
    Bgra8,
    /// `X8B8G8R8`: memory bytes `R G B X`, opaque.
    Rgbx8,
    /// `X8R8G8B8`: memory bytes `B G R X`, opaque.
    Bgrx8,
}

impl SourceFormat {
    fn parse(f: u32) -> Option<SourceFormat> {
        Some(match f {
            format::Y8_V8U8_N420 => SourceFormat::Nv12,
            format::A8B8G8R8 => SourceFormat::Rgba8,
            format::A8R8G8B8 => SourceFormat::Bgra8,
            format::X8B8G8R8 => SourceFormat::Rgbx8,
            format::X8R8G8B8 => SourceFormat::Bgrx8,
            _ => return None,
        })
    }

    /// Bytes between horizontally adjacent pixels.
    fn stride(self) -> usize {
        match self {
            SourceFormat::Nv12 => 1,
            _ => 4,
        }
    }

    /// Byte offsets of the red, green and blue components.
    fn channels(self) -> [usize; 3] {
        match self {
            // The YUV path never uses these; luma is channel 0.
            SourceFormat::Nv12 => [0, 1, 2],
            SourceFormat::Rgba8 | SourceFormat::Rgbx8 => [0, 1, 2],
            SourceFormat::Bgra8 | SourceFormat::Bgrx8 => [2, 1, 0],
        }
    }

    /// Whether byte 3 is a real alpha channel instead of padding.
    fn has_alpha(self) -> bool {
        matches!(self, SourceFormat::Rgba8 | SourceFormat::Bgra8)
    }
}

#[cfg(test)]
type Pixel = [u16; 4];

pub struct Vic {
    regs: Vec<u32>,
    compositions: u64,
}

impl Default for Vic {
    fn default() -> Self {
        Vic {
            regs: vec![0; reg::COUNT],
            compositions: 0,
        }
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

    fn read_plane(
        &self,
        dma: &Dma,
        offset: u32,
        width_bytes: usize,
        height: usize,
    ) -> Option<Vec<u8>> {
        if self.block_linear {
            let raw = dma.read(
                offset,
                block_linear_size(width_bytes, height, self.block_height_log2),
            )?;
            Some(from_block_linear(
                &raw,
                width_bytes,
                height,
                self.block_height_log2,
            ))
        } else {
            let p = pitch(width_bytes);
            let raw = dma.read(offset, p * (height - 1) + width_bytes)?;
            Some(
                raw.chunks(p)
                    .flat_map(|r| &r[..width_bytes.min(r.len())])
                    .copied()
                    .collect(),
            )
        }
    }

    fn write_plane(
        &self,
        dma: &Dma,
        offset: u32,
        rows: &[u8],
        width_bytes: usize,
        height: usize,
    ) -> bool {
        if self.block_linear {
            dma.write(
                offset,
                &to_block_linear(
                    rows,
                    width_bytes,
                    width_bytes,
                    height,
                    self.block_height_log2,
                ),
            )
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

/// Output pixel formats this engine can write. The numbers follow the VIC
/// pixel-format list; every other value is rejected instead of guessed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OutputFormat {
    /// `A8B8G8R8`: memory bytes `R G B A`.
    Rgba8,
    /// `A8R8G8B8`: memory bytes `B G R A`.
    Bgra8,
    /// `R5G6B5`, little-endian 16-bit word.
    Rgb565,
    /// `A1R5G5B5`, little-endian 16-bit word.
    Argb1555,
    /// `X8B8G8R8`: memory bytes `R G B X`.
    Rgbx8,
    /// `X8R8G8B8`: memory bytes `B G R X`.
    Bgrx8,
    /// `B8G8R8`: memory bytes `R G B`.
    Rgb8,
    /// `R8G8B8`: memory bytes `B G R`.
    Bgr8,
    /// `Y8_V8U8_N420`: NV12 luma plus interleaved chroma.
    Nv12,
    /// `Y8_U8_V8_N420`: planar I420, written to three surfaces.
    I420,
    /// `Y10_V10U10_N420`: P010, two bytes per sample.
    P010,
}

impl OutputFormat {
    fn parse(f: u32) -> Option<OutputFormat> {
        Some(match f {
            format::A8B8G8R8 => OutputFormat::Rgba8,
            format::A8R8G8B8 => OutputFormat::Bgra8,
            format::R5G6B5 => OutputFormat::Rgb565,
            format::A1R5G5B5 => OutputFormat::Argb1555,
            format::X8B8G8R8 => OutputFormat::Rgbx8,
            format::X8R8G8B8 => OutputFormat::Bgrx8,
            format::B8G8R8 => OutputFormat::Rgb8,
            format::R8G8B8 => OutputFormat::Bgr8,
            format::Y8_V8U8_N420 => OutputFormat::Nv12,
            format::Y8_U8_V8_N420 => OutputFormat::I420,
            format::Y10_V10U10_N420 => OutputFormat::P010,
            _ => return None,
        })
    }

    fn bytes_per_pixel(self) -> usize {
        match self {
            OutputFormat::Rgba8
            | OutputFormat::Bgra8
            | OutputFormat::Rgbx8
            | OutputFormat::Bgrx8 => 4,
            OutputFormat::Rgb8 | OutputFormat::Bgr8 => 3,
            OutputFormat::Rgb565 | OutputFormat::Argb1555 => 2,
            OutputFormat::Nv12 | OutputFormat::I420 => 1,
            OutputFormat::P010 => 2,
        }
    }
}

fn encode(format: OutputFormat, rgba: &[u8]) -> Vec<u8> {
    let pixels = rgba.as_chunks::<4>().0.iter();
    match format {
        OutputFormat::Rgba8 => rgba.to_vec(),
        OutputFormat::Bgra8 => pixels.flat_map(|p| [p[2], p[1], p[0], p[3]]).collect(),
        OutputFormat::Rgbx8 => pixels.flat_map(|p| [p[0], p[1], p[2], 0xff]).collect(),
        OutputFormat::Bgrx8 => pixels.flat_map(|p| [p[2], p[1], p[0], 0xff]).collect(),
        OutputFormat::Rgb8 => pixels.flat_map(|p| [p[0], p[1], p[2]]).collect(),
        OutputFormat::Bgr8 => pixels.flat_map(|p| [p[2], p[1], p[0]]).collect(),
        OutputFormat::Rgb565 => pixels
            .flat_map(|p| {
                let v =
                    u16::from(p[0] >> 3) << 11 | u16::from(p[1] >> 2) << 5 | u16::from(p[2] >> 3);
                v.to_le_bytes()
            })
            .collect(),
        OutputFormat::Argb1555 => pixels
            .flat_map(|p| {
                let alpha = if p[3] >= 0x80 { 1u16 } else { 0 };
                let v = alpha << 15
                    | u16::from(p[0] >> 3) << 10
                    | u16::from(p[1] >> 3) << 5
                    | u16::from(p[2] >> 3);
                v.to_le_bytes()
            })
            .collect(),
        OutputFormat::Nv12 | OutputFormat::I420 | OutputFormat::P010 => Vec::new(),
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
            m: [
                [s(0), s(64), s(128), s(192)],
                [s(20), s(84), s(148), s(212)],
                [s(40), s(104), s(168), s(232)],
            ],
            shift: bits(b, 60, 4),
            enabled: bits(b, 127, 1) != 0,
        }
    }

    #[cfg(test)]
    fn apply(&self, p: Pixel, lo: u16, hi: u16) -> Pixel {
        if !self.enabled {
            return [
                p[0].clamp(lo, hi),
                p[1].clamp(lo, hi),
                p[2].clamp(lo, hi),
                p[3],
            ];
        }
        let (x, y, z) = (p[0] as i64, p[1] as i64, p[2] as i64);
        let c = |r: &[i64; 4]| {
            ((((r[0] * x + r[1] * y + r[2] * z) >> self.shift) + r[3]) >> 8)
                .clamp(lo as i64, hi as i64) as u16
        };
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
        let cfg = dma
            .read(at, config::SIZE)
            .ok_or_else(|| format!("VIC: config struct {at:#x} is not mapped"))?;
        let out_cfg = &cfg[config::OUTPUT..config::OUTPUT + 0x10];
        let out_surface = &cfg[config::OUTPUT_SURFACE..config::OUTPUT_SURFACE + 0x10];
        let surface = SurfaceConfig::parse(out_surface);
        let format = OutputFormat::parse(surface.format)
            .ok_or_else(|| format!("VIC: output format {} is not supported", surface.format))?;
        let (width, height) = (
            bits(out_surface, 32, 14) as usize + 1,
            bits(out_surface, 46, 14) as usize + 1,
        );
        let channel = |lsb: usize| (bits(out_cfg, lsb, output::CHANNEL_BITS) >> 2) as u8;
        let background = [
            channel(output::RED),
            channel(output::GREEN),
            channel(output::BLUE),
            channel(output::ALPHA),
        ];
        let mut out: Vec<u8> = background.repeat(width * height);
        let (l, r) = (
            bits(out_cfg, output::LEFT, output::RECT_BITS) as usize,
            bits(out_cfg, output::RIGHT, output::RECT_BITS) as usize + 1,
        );
        let (t, b) = (
            bits(out_cfg, output::TOP, output::RECT_BITS) as usize,
            bits(out_cfg, output::BOTTOM, output::RECT_BITS) as usize + 1,
        );
        let (tx, ty) = (l.min(r), t.min(b));
        let (tw, th) = (
            width.saturating_sub(tx).min(l.abs_diff(r)),
            height.saturating_sub(ty).min(t.abs_diff(b)),
        );
        let transform = Transform {
            flip_h: bits(out_cfg, output::FLIP_H, 1) != 0,
            flip_v: bits(out_cfg, output::FLIP_V, 1) != 0,
            rotation: bits(out_cfg, output::ROTATION, 2) as u8,
        };
        let mut slots = 0;
        let mut video = false;
        let mut luma_mean = 0u64;
        for index in 0..8 {
            let s = &cfg[config::SLOTS + index * config::SLOT_SIZE..][..config::SLOT_SIZE];
            if bits(s, slot::ENABLE, 1) == 0 {
                continue;
            }
            slots += 1;
            let src = self.read_slot(
                dma,
                index,
                &SurfaceConfig::parse(&s[config::SLOT_SURFACE..]),
            )?;
            if matches!(src.kind, SourceKind::Yuv { .. }) {
                video = true;
            }
            if trace()
                && let SourceKind::Yuv { y, .. } = &src.kind
                && !y.data.is_empty()
            {
                luma_mean = y.data.iter().map(|&v| v as u64).sum::<u64>() / y.data.len() as u64;
            }
            if tw == 0 || th == 0 || src.width == 0 || src.height == 0 {
                continue;
            }
            let (lo, hi) = (
                bits(s, slot::RANGE_LOW, 10) as u16,
                bits(s, slot::RANGE_HIGH, 10) as u16,
            );
            let hi = if hi == 0 { 0x3ff } else { hi };
            let convert = Converter::new(&Matrix::parse(&s[config::SLOT_COLOR_MATRIX..]), lo, hi);
            let geometry = Geometry {
                size: (tw, th),
                source: (src.width, src.height),
                transform,
            };
            let scaled = transform.rotation % 2 == 1;
            let native = if scaled {
                (src.height, src.width)
            } else {
                (src.width, src.height)
            };
            let filter = if bits(s, slot::NEAREST, 1) != 0 || (tw, th) == native {
                Filter::Nearest
            } else {
                Filter::Bilinear
            };
            // A source alpha only matters when the slot blends: without the
            // blend flag the hardware composites the slot as opaque.
            let blending = bits(s, slot::BLEND, 1) != 0;
            let slot_alpha = bits(s, slot::ALPHA, slot::ALPHA_BITS).min(0x3ff);
            for dy in 0..th {
                let row = &mut out[((ty + dy) * width + tx) * 4..][..tw * 4];
                for dx in 0..tw {
                    let (fx, fy) = geometry.map(dx, dy);
                    let (rgb, source_alpha) = src.sample(&convert, fx, fy, filter);
                    let alpha = match (blending, source_alpha) {
                        (true, Some(a)) => (slot_alpha * a as u32 + 0x7f) / 0xff,
                        (true, None) => slot_alpha,
                        (false, _) => 0x3ff,
                    };
                    let px = &mut row[dx * 4..dx * 4 + 4];
                    let dst = [px[0], px[1], px[2], px[3]];
                    let blended = blend(dst, rgb, alpha);
                    px.copy_from_slice(&blended);
                }
            }
        }
        let offset = self.regs[reg::OUTPUT_SURFACE];
        self.compositions += 1;
        if trace() {
            eprintln!(
                "vic: {:.3} s: composition {}: {slots} slot(s), mean luma {luma_mean} -> {width}x{height} format {} {} (block height {}) rot {} flip {}{} at {:#x}",
                crate::gpu::gpu_ticks() as f64 / 1e9,
                self.compositions,
                surface.format,
                if surface.block_linear {
                    "block-linear"
                } else {
                    "pitch"
                },
                1 << surface.block_height_log2,
                transform.rotation * 90,
                transform.flip_h as u8,
                transform.flip_v as u8,
                (offset as u64) << 8
            );
        }
        let written = match format {
            OutputFormat::Nv12 => {
                let (y, uv) = to_nv12(&out, width, height, surface.chroma);
                let (cw, ch) = surface.chroma;
                surface.write_plane(dma, offset, &y, width, height)
                    && surface.write_plane(dma, self.regs[reg::OUTPUT_SURFACE + 1], &uv, cw * 2, ch)
            }
            OutputFormat::I420 => {
                let (y, u, v) = to_i420(&out, width, height, surface.chroma);
                let (cw, ch) = surface.chroma;
                surface.write_plane(dma, offset, &y, width, height)
                    && surface.write_plane(dma, self.regs[reg::OUTPUT_SURFACE + 1], &u, cw, ch)
                    && surface.write_plane(dma, self.regs[reg::OUTPUT_SURFACE + 2], &v, cw, ch)
            }
            OutputFormat::P010 => {
                let (y, uv) = to_p010(&out, width, height, surface.chroma);
                let (cw, ch) = surface.chroma;
                surface.write_plane(dma, offset, &y, width * 2, height)
                    && surface.write_plane(dma, self.regs[reg::OUTPUT_SURFACE + 1], &uv, cw * 4, ch)
            }
            _ => {
                let encoded = encode(format, &out);
                surface.write_plane(
                    dma,
                    offset,
                    &encoded,
                    width * format.bytes_per_pixel(),
                    height,
                )
            }
        };
        if !written {
            return Err(format!(
                "VIC: output surface {:#x} is not mapped",
                (offset as u64) << 8
            ));
        }
        if video {
            let (cw, ch) = surface.chroma;
            let len = match format {
                OutputFormat::Nv12 => width * height + cw * 2 * ch,
                OutputFormat::I420 => width * height + 2 * cw * ch,
                OutputFormat::P010 => width * height * 2 + cw * 4 * ch,
                _ => width * height * format.bytes_per_pixel(),
            };
            if let Some(cpu) = dma.cpu(offset, len) {
                clock::composed_surface(cpu, len);
            }
        }
        Ok(())
    }

    fn read_slot(&self, dma: &Dma, slot: usize, s: &SurfaceConfig) -> Result<Source, String> {
        let format = SourceFormat::parse(s.format)
            .ok_or_else(|| format!("VIC: slot format {} is not supported", s.format))?;
        let base = reg::SURFACES + slot * 8 * 3;
        let (luma_at, chroma_at) = (self.regs[base], self.regs[base + 1]);
        let (width, height) = s.luma;
        let missing = || {
            format!(
                "VIC: slot {slot} surface {:#x} is not mapped",
                (luma_at as u64) << 8
            )
        };
        if format == SourceFormat::Nv12 {
            let chroma = s.chroma;
            let y = s
                .read_plane(dma, luma_at, width, height)
                .ok_or_else(missing)?;
            let uv = s
                .read_plane(dma, chroma_at, chroma.0 * 2, chroma.1)
                .ok_or_else(missing)?;
            return Ok(Source {
                width,
                height,
                kind: SourceKind::Yuv {
                    y: Plane {
                        width,
                        height,
                        step: 1,
                        data: y,
                    },
                    uv: Plane {
                        width: chroma.0,
                        height: chroma.1,
                        step: 2,
                        data: uv,
                    },
                },
            });
        }
        let stride = format.stride();
        let data = s
            .read_plane(dma, luma_at, width * stride, height)
            .ok_or_else(missing)?;
        Ok(Source {
            width,
            height,
            kind: SourceKind::Rgb {
                pixels: Plane {
                    width,
                    height,
                    step: stride,
                    data,
                },
                channels: format.channels(),
                alpha: format.has_alpha(),
            },
        })
    }
}

/// Composites a source pixel over the current destination value with a 10-bit
/// coverage alpha.
fn blend(dst: [u8; 4], src: [u8; 3], alpha: u32) -> [u8; 4] {
    if alpha >= 0x3ff {
        return [src[0], src[1], src[2], 0xff];
    }
    let inverse = 0x3ff - alpha;
    let mix = |d: u8, s: u8| ((d as u32 * inverse + s as u32 * alpha + 0x1ff) / 0x3ff) as u8;
    [
        mix(dst[0], src[0]),
        mix(dst[1], src[1]),
        mix(dst[2], src[2]),
        ((alpha * 0xff + 0x1ff) / 0x3ff) as u8,
    ]
}

#[derive(Clone, Copy)]
struct Transform {
    flip_h: bool,
    flip_v: bool,
    /// Clockwise rotation in quarter turns (0..=3).
    rotation: u8,
}

/// Maps output pixels of the target rectangle back to source surface
/// coordinates (Q16.16, pixel centres at integer values). Rotation and flips
/// describe the orientation of the source inside the rectangle, so the output
/// surface keeps its declared dimensions.
#[derive(Clone, Copy)]
struct Geometry {
    size: (usize, usize),
    source: (usize, usize),
    transform: Transform,
}

impl Geometry {
    fn map(&self, dx: usize, dy: usize) -> (i64, i64) {
        let (dw, dh) = self.size;
        let (sw, sh) = self.source;
        let rotate = self.transform.rotation & 3;
        let (rw, rh) = if rotate % 2 == 1 { (sh, sw) } else { (sw, sh) };
        let scale = |offset: usize, dest: usize, size: usize| {
            ((2 * offset as i64 + 1) * size as i64 * 0x1_0000) / (2 * dest as i64) - 0x8000
        };
        let mut u = scale(dx, dw, rw);
        let mut v = scale(dy, dh, rh);
        if self.transform.flip_h {
            u = (rw as i64 - 1) * 0x1_0000 - u;
        }
        if self.transform.flip_v {
            v = (rh as i64 - 1) * 0x1_0000 - v;
        }
        match rotate {
            0 => (u, v),
            1 => (v, (sh as i64 - 1) * 0x1_0000 - u),
            2 => (
                (sw as i64 - 1) * 0x1_0000 - u,
                (sh as i64 - 1) * 0x1_0000 - v,
            ),
            _ => ((sw as i64 - 1) * 0x1_0000 - v, u),
        }
    }
}

#[derive(Clone, Copy)]
enum Filter {
    Nearest,
    Bilinear,
}

struct Plane {
    data: Vec<u8>,
    width: usize,
    height: usize,
    /// Bytes between horizontally adjacent samples (2 for interleaved chroma).
    step: usize,
}

impl Plane {
    fn at(&self, x: isize, y: isize, channel: usize) -> u8 {
        if self.width == 0 || self.height == 0 {
            return 0;
        }
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data
            .get((y * self.width + x) * self.step + channel)
            .copied()
            .unwrap_or(0)
    }

    /// Samples at Q16.16 source coordinates; nearest rounds to the closest
    /// pixel centre, bilinear weights the four surrounding samples.
    fn sample(&self, x: i64, y: i64, filter: Filter, channel: usize) -> u8 {
        match filter {
            Filter::Nearest => self.at(
                ((x + 0x8000) >> 16) as isize,
                ((y + 0x8000) >> 16) as isize,
                channel,
            ),
            Filter::Bilinear => {
                let x0 = (x >> 16) as isize;
                let y0 = (y >> 16) as isize;
                let tx = x & 0xffff;
                let ty = y & 0xffff;
                let top = self.at(x0, y0, channel) as i64 * (0x1_0000 - tx)
                    + self.at(x0 + 1, y0, channel) as i64 * tx;
                let bottom = self.at(x0, y0 + 1, channel) as i64 * (0x1_0000 - tx)
                    + self.at(x0 + 1, y0 + 1, channel) as i64 * tx;
                ((top * (0x1_0000 - ty) + bottom * ty + (1 << 31)) >> 32) as u8
            }
        }
    }
}

/// Where a slot's pixels come from and how they map to colour.
struct Source {
    width: usize,
    height: usize,
    kind: SourceKind,
}

enum SourceKind {
    /// YUV 4:2:0, converted through the slot's colour matrix.
    Yuv { y: Plane, uv: Plane },
    /// 32-bit RGB(A) pixels, already in a display colour space: the matrix is
    /// bypassed, as the hardware does for RGB surfaces.
    Rgb {
        pixels: Plane,
        /// Byte offsets of the red, green and blue components.
        channels: [usize; 3],
        /// Whether byte 3 is a source alpha instead of padding.
        alpha: bool,
    },
}

impl Source {
    /// Samples the source at Q16.16 coordinates, returning the colour and, for
    /// formats with a real alpha channel, the source alpha.
    fn sample(&self, convert: &Converter, x: i64, y: i64, filter: Filter) -> ([u8; 3], Option<u8>) {
        match &self.kind {
            SourceKind::Yuv { y: luma, uv } => {
                let luma = luma.sample(x, y, filter, 0);
                let (cx, cy) = (x >> 1, y >> 1);
                let u = uv.sample(cx, cy, filter, 0);
                let v = uv.sample(cx, cy, filter, 1);
                (convert.pixel(luma, u, v), None)
            }
            SourceKind::Rgb {
                pixels,
                channels,
                alpha,
            } => {
                let rgb = std::array::from_fn(|i| pixels.sample(x, y, filter, channels[i]));
                let a = alpha.then(|| pixels.sample(x, y, filter, 3));
                (rgb, a)
            }
        }
    }
}

fn to_nv12(yuva: &[u8], width: usize, height: usize, chroma: (usize, usize)) -> (Vec<u8>, Vec<u8>) {
    let y: Vec<u8> = yuva.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
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

/// Splits the composed staging buffer into planar 4:2:0 planes (I420).
fn to_i420(
    yuva: &[u8],
    width: usize,
    height: usize,
    chroma: (usize, usize),
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y: Vec<u8> = yuva.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
    let (cw, ch) = chroma;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for cy in 0..ch {
        let sy = (2 * cy).min(height - 1);
        for cx in 0..cw {
            let p = &yuva[(sy * width + (2 * cx).min(width - 1)) * 4..][..4];
            u[cy * cw + cx] = p[1];
            v[cy * cw + cx] = p[2];
        }
    }
    (y, u, v)
}

/// Packs the composed staging buffer as P010.
///
/// The compositor works at 8 bits per sample, so the samples are promoted to 10
/// bits with [`BitDepth::rescale`] (the same policy NVDEC uses for an 8-bit
/// picture in a 10-bit surface) and then stored in the high bits of the
/// little-endian word.
fn to_p010(yuva: &[u8], width: usize, height: usize, chroma: (usize, usize)) -> (Vec<u8>, Vec<u8>) {
    let (y, uv) = to_nv12(yuva, width, height, chroma);
    let pack = |samples: &[u8]| -> Vec<u8> {
        samples
            .iter()
            .flat_map(|&v| (BitDepth::Eight.rescale(v as u16, BitDepth::Ten) << 6).to_le_bytes())
            .collect()
    };
    (pack(&y), pack(&uv))
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
        let products = std::array::from_fn(|c| {
            std::array::from_fn(|i| std::array::from_fn(|v| m.m[c][i] * ((v as i64) << 2)))
        });
        Converter {
            products,
            offsets: [m.m[0][3], m.m[1][3], m.m[2][3]],
            shift: m.shift,
            enabled: m.enabled,
            lo: lo as i64,
            hi: hi as i64,
        }
    }

    #[inline]
    fn pixel(&self, y: u8, u: u8, v: u8) -> [u8; 3] {
        let (y, u, v) = (y as usize, u as usize, v as usize);
        if !self.enabled {
            return [y, u, v].map(|s| (((s as i64) << 2).clamp(self.lo, self.hi) >> 2) as u8);
        }
        std::array::from_fn(|c| {
            let p = &self.products[c];
            (((((p[0][y] + p[1][u] + p[2][v]) >> self.shift) + self.offsets[c]) >> 8)
                .clamp(self.lo, self.hi)
                >> 2) as u8
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{Gpu, NvMap};
    use crate::host1x::Iova;
    use crate::mem::GuestMemory;

    #[test]
    fn nv12_output_planes() {
        let px: Vec<u8> = (0..2)
            .flat_map(|y| (0..4).flat_map(move |x| [x as u8, 100 + x as u8, 200 + y as u8, 255]))
            .collect();
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
        put(
            212,
            ((-1.164 * 64.0 + 0.392 * 512.0 + 0.813 * 512.0) * 256.0) as i64,
        );
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
        for (y, u, v) in [
            (0u8, 0u8, 0u8),
            (235, 128, 128),
            (81, 90, 240),
            (145, 54, 34),
            (255, 255, 255),
            (16, 200, 30),
        ] {
            let want = m.apply(
                [(y as u16) << 2, (u as u16) << 2, (v as u16) << 2, 0x3ff],
                16,
                1000,
            );
            assert_eq!(conv.pixel(y, u, v), [0, 1, 2].map(|i| (want[i] >> 2) as u8));
        }
    }

    #[test]
    fn nearest_and_bilinear_scaling() {
        let plane = Plane {
            data: vec![0, 100, 200, 255],
            width: 2,
            height: 2,
            step: 1,
        };
        // Pixel centres sit at integer coordinates.
        assert_eq!(plane.sample(0, 0, Filter::Nearest, 0), 0);
        assert_eq!(plane.sample(0x1_0000, 0, Filter::Nearest, 0), 100);
        assert_eq!(plane.sample(0, 0x1_0000, Filter::Nearest, 0), 200);
        // A quarter of the way between the left and right samples.
        assert_eq!(plane.sample(0x4000, 0, Filter::Bilinear, 0), 25);
        assert_eq!(
            plane.sample(0x8000, 0x8000, Filter::Bilinear, 0),
            ((100 + 200 + 255) as f64 / 4.0 + 0.5) as u8
        );
    }

    #[test]
    fn scaling_geometry_hits_known_pixels() {
        let geometry = Geometry {
            size: (4, 1),
            source: (2, 1),
            transform: Transform {
                flip_h: false,
                flip_v: false,
                rotation: 0,
            },
        };
        let plane = Plane {
            data: vec![10, 20],
            width: 2,
            height: 1,
            step: 1,
        };
        // Dest pixels 0..1 map to the left source pixel and 2..3 to the right
        // one; the exact boundaries are in the middle of the source pixels.
        let samples: Vec<u8> = (0..4)
            .map(|x| {
                let (fx, fy) = geometry.map(x, 0);
                plane.sample(fx, fy, Filter::Nearest, 0)
            })
            .collect();
        assert_eq!(samples, [10, 10, 20, 20]);
    }

    #[test]
    fn rotation_maps_corners() {
        // 2x3 source, rotated 90 degrees clockwise into a 3x2 rectangle.
        let geometry = Geometry {
            size: (3, 2),
            source: (2, 3),
            transform: Transform {
                flip_h: false,
                flip_v: false,
                rotation: 1,
            },
        };
        let plane = Plane {
            data: (0..6).collect(),
            width: 2,
            height: 3,
            step: 1,
        };
        let at = |x: usize, y: usize| {
            let (fx, fy) = geometry.map(x, y);
            plane.sample(fx, fy, Filter::Nearest, 0)
        };
        // Clockwise: top-left <- bottom-left, top-right <- top-left,
        // bottom-right <- top-right.
        assert_eq!(at(0, 0), 4);
        assert_eq!(at(2, 0), 0);
        assert_eq!(at(2, 1), 1);
    }

    #[test]
    fn flips_mirror_the_target_rectangle() {
        let geometry = |flip_h, flip_v| Geometry {
            size: (2, 1),
            source: (2, 1),
            transform: Transform {
                flip_h,
                flip_v,
                rotation: 0,
            },
        };
        let plane = Plane {
            data: vec![10, 20],
            width: 2,
            height: 1,
            step: 1,
        };
        let sample = |g: Geometry, x: usize| {
            let (fx, fy) = g.map(x, 0);
            plane.sample(fx, fy, Filter::Nearest, 0)
        };
        assert_eq!(sample(geometry(false, false), 0), 10);
        assert_eq!(sample(geometry(true, false), 0), 20);
        assert_eq!(sample(geometry(true, false), 1), 10);
    }

    #[test]
    fn alpha_blends_towards_the_source() {
        assert_eq!(
            blend([0, 0, 0, 255], [255, 255, 255], 0x3ff),
            [255, 255, 255, 255]
        );
        assert_eq!(blend([0, 0, 0, 0], [255, 255, 255], 0), [0, 0, 0, 0]);
        assert_eq!(
            blend([0, 0, 0, 0], [255, 255, 255], 0x200),
            [128, 128, 128, 128]
        );
        assert_eq!(
            blend([200, 100, 0, 0], [0, 100, 200], 0x1ff),
            [100, 100, 100, 127]
        );
    }

    #[test]
    fn output_formats_encode_expected_bytes() {
        let px = [10, 20, 30, 40];
        assert_eq!(encode(OutputFormat::Rgba8, &px), [10, 20, 30, 40]);
        assert_eq!(encode(OutputFormat::Bgra8, &px), [30, 20, 10, 40]);
        assert_eq!(encode(OutputFormat::Rgbx8, &px), [10, 20, 30, 0xff]);
        assert_eq!(encode(OutputFormat::Bgrx8, &px), [30, 20, 10, 0xff]);
        assert_eq!(encode(OutputFormat::Rgb8, &px), [10, 20, 30]);
        assert_eq!(encode(OutputFormat::Bgr8, &px), [30, 20, 10]);
        assert_eq!(
            encode(OutputFormat::Rgb565, &px),
            (u16::from(10u8 >> 3) << 11 | u16::from(20u8 >> 2) << 5 | u16::from(30u8 >> 3))
                .to_le_bytes()
        );
        assert_eq!(
            encode(OutputFormat::Argb1555, &px),
            (u16::from(10u8 >> 3) << 10 | u16::from(20u8 >> 3) << 5 | u16::from(30u8 >> 3))
                .to_le_bytes()
        );
        for yuv in [OutputFormat::Nv12, OutputFormat::I420, OutputFormat::P010] {
            assert!(encode(yuv, &px).is_empty());
        }
        assert_eq!(OutputFormat::P010.bytes_per_pixel(), 2);
        assert_eq!(OutputFormat::I420.bytes_per_pixel(), 1);
        for (bits, want) in [
            (format::A8B8G8R8, OutputFormat::Rgba8),
            (format::A8R8G8B8, OutputFormat::Bgra8),
            (format::R5G6B5, OutputFormat::Rgb565),
            (format::A1R5G5B5, OutputFormat::Argb1555),
            (format::X8B8G8R8, OutputFormat::Rgbx8),
            (format::X8R8G8B8, OutputFormat::Bgrx8),
            (format::B8G8R8, OutputFormat::Rgb8),
            (format::R8G8B8, OutputFormat::Bgr8),
            (format::Y8_V8U8_N420, OutputFormat::Nv12),
            (format::Y8_U8_V8_N420, OutputFormat::I420),
            (format::Y10_V10U10_N420, OutputFormat::P010),
            (0, OutputFormat::Rgba8),
        ] {
            assert_eq!(OutputFormat::parse(bits), (bits != 0).then_some(want));
        }
        for (source, want) in [
            (format::Y8_V8U8_N420, SourceFormat::Nv12),
            (format::A8B8G8R8, SourceFormat::Rgba8),
            (format::A8R8G8B8, SourceFormat::Bgra8),
            (format::X8B8G8R8, SourceFormat::Rgbx8),
            (format::X8R8G8B8, SourceFormat::Bgrx8),
        ] {
            assert_eq!(SourceFormat::parse(source), Some(want));
        }
        assert_eq!(SourceFormat::parse(format::R5G6B5), None);
        // Memory byte order of the RGB sources.
        assert_eq!(SourceFormat::Rgba8.channels(), [0, 1, 2]);
        assert_eq!(SourceFormat::Bgra8.channels(), [2, 1, 0]);
        assert!(SourceFormat::Rgba8.has_alpha());
        assert!(!SourceFormat::Rgbx8.has_alpha());
        assert_eq!(SourceFormat::Rgba8.stride(), 4);
        assert_eq!(SourceFormat::Nv12.stride(), 1);
    }

    #[test]
    fn planar_and_ten_bit_output_pack_the_staging_buffer() {
        let px: Vec<u8> = (0..2)
            .flat_map(|y| (0..4).flat_map(move |x| [x as u8, 100 + x as u8, 200 + y as u8, 255]))
            .collect();
        let (y, u, v) = to_i420(&px, 4, 2, (2, 1));
        assert_eq!(y, [0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(u, [100, 102]);
        assert_eq!(v, [200, 200]);
        let (y, uv) = to_p010(&px, 4, 2, (2, 1));
        let words = |b: &[u8]| -> Vec<u16> {
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|w| u16::from_le_bytes(*w) >> 6)
                .collect()
        };
        assert_eq!(words(&y), [0u16, 4, 8, 12, 0, 4, 8, 12]);
        assert_eq!(words(&uv), [400u16, 800, 408, 800]);
    }

    fn put(b: &mut [u8], lsb: usize, len: usize, value: u32) {
        for i in 0..len {
            if value >> i & 1 != 0 {
                b[(lsb + i) / 8] |= 1 << ((lsb + i) % 8);
            }
        }
    }

    struct Harness {
        mem: GuestMemory,
        gpu: Gpu,
        iova: Iova,
        guest: u64,
        vic: Vic,
        config: usize,
        output: usize,
        /// Second and third output surfaces (chroma for the YUV layouts).
        output_u: usize,
        output_v: usize,
        luma: usize,
        chroma: usize,
    }

    impl Harness {
        fn new(width: usize, height: usize, format: u32) -> Harness {
            let mem = GuestMemory::new().unwrap();
            mem.map(0x8000_0000, 0x10_0000).unwrap();
            let mut iova = Iova::default();
            let base = iova
                .map(
                    7,
                    NvMap {
                        size: 0x10_0000,
                        addr: 0x8000_0000,
                        ..NvMap::default()
                    },
                )
                .unwrap() as u64;
            let (config, output, luma, chroma): (usize, usize, usize, usize) =
                (0x0, 0x1000, 0x2000, 0x3000);
            let (output_u, output_v) = (0x4000, 0x5000);
            let guest = 0x8000_0000u64;
            let mut vic = Vic::default();
            vic.regs[reg::CONFIG_STRUCT] = ((base + config as u64) >> 8) as u32;
            vic.regs[reg::OUTPUT_SURFACE] = ((base + output as u64) >> 8) as u32;
            vic.regs[reg::OUTPUT_SURFACE + 1] = ((base + output_u as u64) >> 8) as u32;
            vic.regs[reg::OUTPUT_SURFACE + 2] = ((base + output_v as u64) >> 8) as u32;
            vic.regs[reg::SURFACES] = ((base + luma as u64) >> 8) as u32;
            vic.regs[reg::SURFACES + 1] = ((base + chroma as u64) >> 8) as u32;
            let mut h = Harness {
                mem,
                gpu: Gpu::default(),
                iova,
                guest,
                vic,
                config,
                output,
                output_u,
                output_v,
                luma,
                chroma,
            };
            let cfg = vec![0u8; config::SIZE];
            assert!(h.mem.write(h.guest + h.config as u64, &cfg));
            h.set_output(width, height, format);
            h.configure_surface(width, height);
            h
        }

        fn set_output(&mut self, width: usize, height: usize, format: u32) {
            let at = (self.guest + self.config as u64) as usize;
            let mut out_cfg = [0u8; 16];
            put(
                &mut out_cfg,
                output::RIGHT,
                output::RECT_BITS,
                width as u32 - 1,
            );
            put(
                &mut out_cfg,
                output::BOTTOM,
                output::RECT_BITS,
                height as u32 - 1,
            );
            assert!(self.mem.write((at + config::OUTPUT) as u64, &out_cfg));
            let mut surface = [0u8; 16];
            put(&mut surface, 0, 7, format);
            put(&mut surface, 32, 14, width as u32 - 1);
            put(&mut surface, 46, 14, height as u32 - 1);
            put(&mut surface, 96, 14, width.div_ceil(2).max(1) as u32 - 1);
            put(&mut surface, 110, 14, height.div_ceil(2).max(1) as u32 - 1);
            assert!(
                self.mem
                    .write((at + config::OUTPUT_SURFACE) as u64, &surface)
            );
            let mut slot = [0u8; config::SLOT_SIZE];
            put(&mut slot, slot::ENABLE, 1, 1);
            assert!(self.mem.write((at + config::SLOTS) as u64, &slot));
        }

        fn output_config(&mut self, lsb: usize, len: usize, value: u32) {
            let at = self.guest + (self.config + config::OUTPUT) as u64;
            let mut b = self.mem.read(at, 16).unwrap();
            put(&mut b, lsb, len, value);
            assert!(self.mem.write(at, &b));
        }

        fn set_source(&mut self, width: usize, height: usize) {
            self.configure_surface(width, height);
        }

        fn slot_config(&mut self, lsb: usize, len: usize, value: u32) {
            let at = self.guest + (self.config + config::SLOTS) as u64;
            let mut b = self.mem.read(at, config::SLOT_SIZE as u64).unwrap();
            put(&mut b, lsb, len, value);
            assert!(self.mem.write(at, &b));
        }

        fn set_source_layout(&mut self, block_linear: bool, block_height_log2: u32) {
            let at = self.guest + (config::SLOTS + config::SLOT_SURFACE) as u64;
            let mut b = self.mem.read(at, 16).unwrap();
            put(&mut b, 11, 4, if block_linear { 1 } else { 0 });
            put(&mut b, 15, 4, block_height_log2);
            assert!(self.mem.write(at, &b));
        }

        fn write_plane_block(&self, at: usize, width_bytes: usize, rows: &[&[u8]], bh: u32) {
            let flat: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();
            let raw = to_block_linear(&flat, width_bytes, width_bytes, rows.len(), bh);
            assert!(self.mem.write(self.guest + at as u64, &raw));
        }

        /// Declares the first slot's source surface with an explicit format.
        fn set_source_format(&mut self, format: u32, width: usize, height: usize) {
            self.configure_surface_format(format, width, height);
        }

        fn configure_surface(&mut self, width: usize, height: usize) {
            self.configure_surface_format(format::Y8_V8U8_N420, width, height);
        }

        fn configure_surface_format(&mut self, format: u32, width: usize, height: usize) {
            let mut surface = [0u8; 16];
            put(&mut surface, 0, 7, format);
            put(&mut surface, 64, 14, width as u32 - 1);
            put(&mut surface, 78, 14, height as u32 - 1);
            put(&mut surface, 96, 14, width.div_ceil(2).max(1) as u32 - 1);
            put(&mut surface, 110, 14, height.div_ceil(2).max(1) as u32 - 1);
            assert!(self.mem.write(
                self.guest + (config::SLOTS + config::SLOT_SURFACE) as u64,
                &surface
            ));
        }

        fn compose(&mut self) -> Result<(), String> {
            let dma = Dma {
                iova: &self.iova,
                mem: &self.mem,
                gpu: &self.gpu,
                later: None,
            };
            self.vic.write(&dma, reg::EXECUTE, 0)
        }

        /// Writes a pitch-linear plane using the same default pitch the DMA
        /// reader assumes between rows.
        fn write_plane(&self, at: usize, width_bytes: usize, rows: &[&[u8]]) {
            for (y, row) in rows.iter().enumerate() {
                assert!(row.len() <= width_bytes);
                assert!(self.mem.write(
                    self.guest + at as u64 + (pitch(width_bytes) * y) as u64,
                    row
                ));
            }
        }

        fn write_luma(&self, width_bytes: usize, rows: &[&[u8]]) {
            self.write_plane(self.luma, width_bytes, rows);
        }

        fn write_chroma(&self, width_bytes: usize, rows: &[&[u8]]) {
            self.write_plane(self.chroma, width_bytes, rows);
        }

        /// Reads back a pitch-linear output surface, dropping the row padding
        /// the runtime models between the rows.
        fn output(&self, row_bytes: usize, height: usize) -> Vec<u8> {
            self.read_plane(self.output, row_bytes, height)
        }

        fn read_plane(&self, at: usize, row_bytes: usize, height: usize) -> Vec<u8> {
            let stride = pitch(row_bytes);
            let raw = self
                .mem
                .read(self.guest + at as u64, (stride * height) as u64)
                .unwrap();
            raw.chunks(stride)
                .take(height)
                .flat_map(|r| &r[..row_bytes])
                .copied()
                .collect()
        }
    }

    #[test]
    fn composes_nv12_into_rgba() {
        let mut h = Harness::new(4, 2, format::A8B8G8R8);
        h.write_luma(4, &[&[0, 40, 80, 120], &[160, 200, 240, 20]]);
        h.write_chroma(4, &[&[128, 128, 128, 128]]);
        h.compose().unwrap();
        let out = h.output(4 * 4, 2);
        let reds: Vec<u8> = out.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        assert_eq!(reds, [0, 40, 80, 120, 160, 200, 240, 20]);
        assert!(out.as_chunks::<4>().0.iter().all(|p| p[3] == 0xff));
    }

    #[test]
    fn composes_scaled_nv12_with_nearest_filter() {
        let mut h = Harness::new(4, 2, format::A8B8G8R8);
        h.set_source(2, 1);
        h.write_luma(2, &[&[10, 20]]);
        h.write_chroma(2, &[&[0, 0]]);
        h.slot_config(slot::NEAREST, 1, 1);
        h.compose().unwrap();
        let out = h.output(4 * 4, 2);
        let reds: Vec<u8> = out.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        assert_eq!(reds, [10, 10, 20, 20, 10, 10, 20, 20]);
    }

    #[test]
    fn composes_scaled_nv12_with_bilinear_filter() {
        let mut h = Harness::new(4, 1, format::A8B8G8R8);
        h.set_source(2, 1);
        h.write_luma(2, &[&[0, 100]]);
        h.write_chroma(2, &[&[0, 0]]);
        h.compose().unwrap();
        let out = h.output(4 * 4, 1);
        let reds: Vec<u8> = out.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        assert_eq!(reds, [0, 25, 75, 100]);
    }

    #[test]
    fn composes_block_linear_input() {
        let mut h = Harness::new(4, 2, format::A8B8G8R8);
        h.set_source_layout(true, 1);
        h.write_plane_block(h.luma, 4, &[&[0, 40, 80, 120], &[160, 200, 240, 20]], 1);
        h.write_plane_block(h.chroma, 4, &[&[128, 128, 128, 128]], 1);
        h.compose().unwrap();
        let out = h.output(4 * 4, 2);
        let reds: Vec<u8> = out.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        assert_eq!(reds, [0, 40, 80, 120, 160, 200, 240, 20]);
    }

    #[test]
    fn composes_with_alpha_over_background() {
        let mut h = Harness::new(2, 2, format::A8B8G8R8);
        h.write_luma(2, &[&[0, 0], &[0, 0]]);
        h.write_chroma(2, &[&[0, 0]]);
        for channel in [output::RED, output::GREEN, output::BLUE] {
            h.output_config(channel, 10, 0x3ff);
        }
        h.slot_config(slot::BLEND, 1, 1);
        h.slot_config(slot::ALPHA, 10, 0x200);
        h.compose().unwrap();
        let out = h.output(2 * 4, 2);
        assert_eq!(&out[..4], &[127, 127, 127, 128]);
    }

    #[test]
    fn composes_rgb565_with_rotation() {
        // 4x2 source, 2x4 output: the rectangle has the rotated aspect ratio.
        let mut h = Harness::new(2, 4, format::R5G6B5);
        h.set_source(4, 2);
        h.write_luma(4, &[&[0, 10, 20, 30], &[40, 50, 60, 70]]);
        h.write_chroma(4, &[&[128, 128, 128, 128]]);
        h.output_config(output::ROTATION, 2, 1);
        h.compose().unwrap();
        let out = h.output(2 * 2, 4);
        let pixels: Vec<u16> = out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_le_bytes(*p))
            .collect();
        let reds: Vec<u16> = pixels.iter().map(|p| p >> 11).collect();
        assert_eq!(
            reds,
            [
                40 >> 3,
                0,
                50 >> 3,
                10 >> 3,
                60 >> 3,
                20 >> 3,
                70 >> 3,
                30 >> 3
            ]
        );
    }

    #[test]
    fn composes_rgba_slot_sources() {
        let mut h = Harness::new(2, 2, format::A8B8G8R8);
        h.set_source_format(format::A8B8G8R8, 2, 2);
        h.write_luma(
            2 * 4,
            &[
                &[10, 20, 30, 40, 50, 60, 70, 80],
                &[90, 100, 110, 120, 130, 140, 150, 160],
            ],
        );
        h.compose().unwrap();
        let out = h.output(2 * 4, 2);
        // RGB sources bypass the colour matrix and are copied channel for
        // channel; without the blend flag the slot is opaque.
        assert_eq!(
            out,
            [
                10, 20, 30, 255, 50, 60, 70, 255, 90, 100, 110, 255, 130, 140, 150, 255
            ]
        );
    }

    #[test]
    fn composes_bgra_slot_sources() {
        let mut h = Harness::new(1, 1, format::A8B8G8R8);
        h.set_source_format(format::A8R8G8B8, 1, 1);
        // Bytes are B, G, R, A.
        h.write_luma(4, &[&[30, 20, 10, 200]]);
        h.compose().unwrap();
        assert_eq!(h.output(4, 1), [10, 20, 30, 255]);
    }

    #[test]
    fn source_alpha_modulates_the_slot_coverage() {
        let mut h = Harness::new(1, 1, format::A8B8G8R8);
        h.set_source_format(format::A8B8G8R8, 1, 1);
        h.output_config(output::RED, 10, 0);
        h.output_config(output::GREEN, 10, 0);
        h.output_config(output::BLUE, 10, 0);
        h.slot_config(slot::BLEND, 1, 1);
        h.slot_config(slot::ALPHA, 10, 0x3ff);
        h.write_luma(4, &[&[255, 255, 255, 128]]);
        h.compose().unwrap();
        // Half the source alpha over black: the colour and the written alpha
        // land near 128 (the exact values come from the 10-bit coverage).
        assert_eq!(h.output(4, 1), [128, 128, 128, 128]);
    }

    #[test]
    fn without_blending_source_alpha_is_ignored() {
        let mut h = Harness::new(1, 1, format::A8B8G8R8);
        h.set_source_format(format::A8B8G8R8, 1, 1);
        h.write_luma(4, &[&[255, 255, 255, 0]]);
        h.compose().unwrap();
        assert_eq!(h.output(4, 1), [255, 255, 255, 255]);
    }

    #[test]
    fn writes_i420_output() {
        let mut h = Harness::new(4, 2, format::Y8_U8_V8_N420);
        h.write_luma(4, &[&[0, 40, 80, 120], &[160, 200, 240, 20]]);
        h.write_chroma(4, &[&[100, 110, 120, 130]]);
        h.compose().unwrap();
        assert_eq!(h.output(4, 2), [0, 40, 80, 120, 160, 200, 240, 20]);
        // Planar chroma: U in the second surface, V in the third.
        assert_eq!(h.read_plane(h.output_u, 2, 1), [100, 120]);
        assert_eq!(h.read_plane(h.output_v, 2, 1), [110, 130]);
    }

    #[test]
    fn writes_p010_output() {
        let mut h = Harness::new(4, 2, format::Y10_V10U10_N420);
        h.write_luma(4, &[&[0, 40, 80, 120], &[160, 200, 240, 20]]);
        h.write_chroma(4, &[&[100, 110, 120, 130]]);
        h.compose().unwrap();
        let words = |b: &[u8]| -> Vec<u16> {
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|w| u16::from_le_bytes(*w) >> 6)
                .collect()
        };
        // 8-bit composition promoted to 10 bits, two bytes per sample.
        assert_eq!(
            words(&h.output(4 * 2, 2)),
            [0u16, 160, 320, 480, 640, 800, 960, 80]
        );
        assert_eq!(
            words(&h.read_plane(h.output_u, 2 * 2 * 2, 1)),
            [400u16, 440, 480, 520]
        );
    }
}
