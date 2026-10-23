use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

use super::h264::{self, PictureInfo};
use super::vp9;
use super::{block_linear_size, to_block_linear, trace, Dma};

mod reg {
    pub const APPLICATION_ID: u32 = 0x200 / 4;
    pub const EXECUTE: u32 = 0x300 / 4;
    pub const DRV_PIC_SETUP: u32 = 0x404 / 4;
    pub const IN_BUF_BASE: u32 = 0x408 / 4;
    pub const PICTURE_LUMA: u32 = 0x430 / 4;
    pub const PICTURE_CHROMA: u32 = 0x474 / 4;
    pub const VP9_PROB_TAB: u32 = 0x5c0 / 4;
    pub const VP9_CTX_COUNTER: u32 = 0x5c4 / 4;
    pub const COUNT: usize = 0x800 / 4;
}

const CODEC_H264: u32 = 3;
const CODEC_VP8: u32 = 5;
const CODEC_VP9: u32 = 9;

const SURFACE_BLOCK_HEIGHT_LOG2: u32 = 1;

pub struct Nvdec {
    regs: Vec<u32>,
    h264: HashMap<u32, h264::Decoder>,
    vp9: Arc<Mutex<Vp9State>>,
    pictures: u64,
}

#[derive(Default)]
struct Vp9State {
    decoders: HashMap<u32, vp9::Decoder>,
    pictures: vp9::Pictures,
    decoded: u64,
}

impl Default for Nvdec {
    fn default() -> Self {
        Nvdec { regs: vec![0; reg::COUNT], h264: HashMap::new(), vp9: Arc::default(), pictures: 0 }
    }
}

impl Nvdec {
    pub fn write(&mut self, dma: &Dma, context: u32, method: u32, value: u32) -> Result<(), String> {
        if let Some(r) = self.regs.get_mut(method as usize) {
            *r = value;
        }
        if method == reg::EXECUTE {
            return match self.regs[reg::APPLICATION_ID as usize] {
                CODEC_H264 => self.decode_h264(dma, context),
                CODEC_VP8 => Err("NVDEC: VP8 is not supported".into()),
                CODEC_VP9 => self.decode_vp9(dma, context),
                id => Err(format!("NVDEC: codec {id} is not supported")),
            };
        }
        Ok(())
    }

    fn decode_h264(&mut self, dma: &Dma, context: u32) -> Result<(), String> {
        let setup = self.regs[reg::DRV_PIC_SETUP as usize];
        let raw = dma.read(setup, h264::PICTURE_INFO_SIZE).ok_or_else(|| format!("NVDEC: picture setup {setup:#x} is not mapped"))?;
        let info = PictureInfo::parse(&raw);
        let base = self.regs[reg::IN_BUF_BASE as usize];
        let slices = dma.read(base, info.bitstream_size as usize).ok_or_else(|| format!("NVDEC: bitstream {base:#x} is not mapped"))?;
        if info.field_pic || !info.frame_mbs_only {
            return Err("NVDEC: interlaced H.264 is not supported".into());
        }
        if let Some(dir) = std::env::var_os("KIRIKAE_VIDEO_DUMP") {
            let path = std::path::Path::new(&dir).join(format!("nvdec-{context}.h264"));
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = f.write_all(&h264::parameter_sets(&info));
                let _ = f.write_all(&slices);
            }
        }
        let decoder = match self.h264.entry(context) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(h264::Decoder::new()?),
        };
        let frame = decoder.decode(&info, &slices)?;
        self.pictures += 1;
        let surface = (info.output_surface as usize).min(16);
        let luma = self.regs[reg::PICTURE_LUMA as usize + surface].wrapping_add(info.luma_frame_offset);
        let chroma = self.regs[reg::PICTURE_CHROMA as usize + surface].wrapping_add(info.chroma_frame_offset);
        if trace() {
            eprintln!(
                "nvdec: H.264 picture {} on channel {context}: {}x{} MBs, {} bytes in {} slices, frame_num {}, surface {surface} (luma {:#x}, chroma {:#x}) -> {}",
                self.pictures,
                info.width_mbs,
                info.height_mbs,
                info.bitstream_size,
                info.num_slices,
                info.frame_num,
                (luma as u64) << 8,
                (chroma as u64) << 8,
                if frame.is_some() { "decoded" } else { "no output" }
            );
        }
        let Some(f) = frame else { return Ok(()) };
        let y = to_block_linear(&f.y, f.width, f.width, f.height, SURFACE_BLOCK_HEIGHT_LOG2);
        let uv_width = f.width.div_ceil(2) * 2;
        let uv = to_block_linear(&f.uv, uv_width, uv_width, f.height.div_ceil(2), SURFACE_BLOCK_HEIGHT_LOG2);
        if !dma.write(luma, &y) || !dma.write(chroma, &uv) {
            return Err(format!("NVDEC: output surface {surface} is not mapped"));
        }
        Ok(())
    }

    fn decode_vp9(&mut self, dma: &Dma, context: u32) -> Result<(), String> {
        let reg = |r: u32| self.regs[r as usize];
        let setup = reg(reg::DRV_PIC_SETUP);
        let raw = dma.read(setup, vp9::PICTURE_INFO_SIZE).ok_or_else(|| format!("NVDEC: picture setup {setup:#x} is not mapped"))?;
        let info = vp9::PictureInfo::parse(&raw);
        let probs = dma.read(reg(reg::VP9_PROB_TAB), vp9::PROBS_SIZE).ok_or("NVDEC: VP9 probability tables are not mapped")?;
        let tiles = dma.read(reg(reg::IN_BUF_BASE), info.bitstream_size as usize).ok_or("NVDEC: VP9 bitstream is not mapped")?;
        let luma: [u32; 4] = std::array::from_fn(|i| self.regs[reg::PICTURE_LUMA as usize + i]);
        let chroma: [u32; 4] = std::array::from_fn(|i| self.regs[reg::PICTURE_CHROMA as usize + i]);
        let counter = reg(reg::VP9_CTX_COUNTER);
        let state = self.vp9.clone();
        let queued = std::time::Instant::now();
        dma.defer(move |dma| state.lock().unwrap().decode(dma, context, &info, &probs, &tiles, luma, chroma, counter, queued))
    }
}

impl Vp9State {
    #[allow(clippy::too_many_arguments)]
    fn decode(&mut self, dma: &Dma, context: u32, info: &vp9::PictureInfo, probs: &[u8], tiles: &[u8], luma: [u32; 4], chroma: [u32; 4], counter: u32, queued: std::time::Instant) -> Result<(), String> {
        let refs: [Option<vp9::Picture>; 3] = std::array::from_fn(|i| {
            self.pictures.get(luma[i]).or_else(|| {
                let (w, h) = (info.sizes[i].0 as usize, info.sizes[i].1 as usize);
                if w == 0 || h == 0 {
                    return None;
                }
                let y = dma.read(luma[i], block_linear_size(w, h, SURFACE_BLOCK_HEIGHT_LOG2))?;
                let uv = dma.read(chroma[i], block_linear_size(w.div_ceil(2) * 2, h.div_ceil(2), SURFACE_BLOCK_HEIGHT_LOG2))?;
                Some(vp9::picture_from_nv12(&y, &uv, w, h))
            })
        });
        let start = std::time::Instant::now();
        let decoded = self.decoders.entry(context).or_default().decode(info, probs, refs, tiles)?;
        let took = start.elapsed();
        self.decoded += 1;
        dma.write(counter, &decoded.counts);
        let (w, h) = (info.width() as usize, info.height() as usize);
        let (y, uv) = vp9::nv12(&decoded.picture);
        if let Some(dir) = std::env::var_os("KIRIKAE_VIDEO_DUMP").filter(|_| self.decoded.is_multiple_of(100)) {
            let path = std::path::Path::new(&dir).join(format!("nvdec-{}-{w}x{h}.nv12", self.decoded));
            let _ = std::fs::write(path, [y.as_slice(), uv.as_slice()].concat());
        }
        let y = to_block_linear(&y, w, w, h, SURFACE_BLOCK_HEIGHT_LOG2);
        let uv = to_block_linear(&uv, w.div_ceil(2) * 2, w.div_ceil(2) * 2, h.div_ceil(2), SURFACE_BLOCK_HEIGHT_LOG2);
        if !dma.write(luma[3], &y) || !dma.write(chroma[3], &uv) {
            return Err("NVDEC: VP9 output surface is not mapped".into());
        }
        self.pictures.put(luma[3], decoded.picture);
        if trace() {
            eprintln!(
                "nvdec: {:.3} s: VP9 picture {} on channel {context}: {}x{}{}, flags {:#x}, {} bytes -> surface {:#x} in {took:.1?} (+{:.1?} output, {:.1?} after submission)",
                crate::gpu::gpu_ticks() as f64 / 1e9,
                self.decoded,
                info.width(),
                info.height(),
                if info.key_frame() { " key frame" } else { "" },
                info.flags,
                info.bitstream_size,
                (luma[3] as u64) << 8,
                start.elapsed() - took,
                queued.elapsed()
            );
        }
        Ok(())
    }
}
