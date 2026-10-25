//! NVDEC: H.264 and VP9 picture decode.
//!
//! # Codec coverage and missing backends
//!
//! The engine implements the two codecs the built-in decoders can produce:
//! H.264 through OpenH264 (feature `h264`) and VP9 through `vp9dec`. VP8 and AV1
//! are rejected with an error that names the codec; they never return a black or
//! partially written surface, so a title that uses them fails loudly instead of
//! presenting garbage.
//!
//! ## Design for a VP8 / AV1 backend
//!
//! Both codecs are entropy-coded end to end (VP8 has no out-of-band probability
//! tables and AV1 is self-describing), so neither needs the picture-info walking
//! that H.264 and VP9 do: the whole `IN_BUF_BASE` buffer goes to libavcodec and
//! the decoder manages its own reference frames.
//!
//! The intended wiring, kept out of the tree because it cannot be verified here
//! (no FFmpeg development libraries are part of the build, and Tegra X1's NVDEC
//! has no AV1 decoder at all, so an AV1 path could not be exercised against the
//! hardware it claims to emulate):
//!
//! 1. `Cargo.toml`: `ffmpeg-next = { version = "7", optional = true }` and
//!    `ffmpeg = ["dep:ffmpeg-next"]`, off by default. The build needs
//!    libavcodec's headers and import libraries; `pkg-config` locates them.
//! 2. `host1x/ffmpeg.rs`, `#[cfg(feature = "ffmpeg")]`: one `ffmpeg::decoder::Video`
//!    per NVDEC channel, created with `decoder::find(Id::VP8 | Id::AV1)` and
//!    flushed per picture the same way the H.264 path does.
//! 3. Codec dispatch below: read the bitstream once (the size still has to be
//!    established from the VP8/AV1 picture setup; the offsets differ from the
//!    H.264 and VP9 blocks — that field has not been observed in a trace yet),
//!    hand the packet over, then write the returned `Yuv420` with
//!    [`write_picture`], which already handles both NV12 and P010.
//! 4. Gate the reply on the feature: without `ffmpeg`, the error text points at
//!    the missing feature rather than at a codec the runtime cannot decode.
//!
//! # H.264 picture to surface mapping
//!
//! NVDEC writes each decoded picture into the surface its own picture setup
//! names. OpenH264 hands pictures back in display order and one submission
//! after the access unit that produced them (it cannot be flushed per picture:
//! see `h264::Decoder::new`), so the engine keeps the surface addresses of the
//! pending access units and matches each output to the entry with the lowest
//! picture order count. Entries of a previous IDR period are drained first,
//! because the decoder still owes their pictures when a new period starts. A
//! picture whose access unit fails is dropped with its entry; nothing is
//! written to the guest until a full picture is available.

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::sync::{Arc, Mutex};

use super::h264::{self, PictureInfo};
use super::surface::{BitDepth, Yuv420};
use super::vp9;
use super::{Dma, block_linear_size, clock, to_block_linear, trace};

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
const CODEC_AV1: u32 = 10;

const SURFACE_BLOCK_HEIGHT_LOG2: u32 = 1;

/// Access units waiting for their decoded picture, oldest submission first.
///
/// The decoder emits pictures in display order, so the surface an output has to
/// be written to is not necessarily the one of the submission that produced it:
/// with B-frames a picture is output after a later submission. Each entry keeps
/// the surface addresses captured when the access unit was submitted, and the
/// output is matched to the entry with the lowest picture order count (the next
/// one in display order). Entries of an earlier IDR period are consumed first
/// because the decoder still owes their pictures when a new period starts.
const MAX_PENDING_PICTURES: usize = 8;

struct PendingPicture {
    luma: u32,
    chroma: u32,
    poc: i32,
    period: u64,
}

impl PendingPicture {
    fn take_for_output(
        pending: &mut VecDeque<PendingPicture>,
        period: u64,
    ) -> Option<PendingPicture> {
        let oldest_period = pending.front()?.period;
        if oldest_period < period {
            return pending.pop_front();
        }
        let index = pending
            .iter()
            .enumerate()
            .min_by_key(|(_, p)| p.poc)
            .map(|(i, _)| i)?;
        pending.remove(index)
    }
}

struct H264Channel {
    decoder: h264::Decoder,
    pending: VecDeque<PendingPicture>,
    period: u64,
}

impl H264Channel {
    fn new() -> Result<H264Channel, String> {
        Ok(H264Channel {
            decoder: h264::Decoder::new()?,
            pending: VecDeque::new(),
            period: 0,
        })
    }
}

pub struct Nvdec {
    regs: Vec<u32>,
    h264: HashMap<u32, H264Channel>,
    vp9: Arc<Mutex<Vp9State>>,
    pictures: u64,
}

#[derive(Default)]
struct Vp9State {
    decoders: HashMap<u32, vp9::Decoder>,
    pictures: vp9::Pictures,
    decoded: u64,
    /// Depth of the last picture decoded on any channel. Reference surfaces are
    /// reloaded from guest memory at this depth: the decoder writes 8-bit NV12
    /// or 10-bit P010 depending on the picture it produced.
    bit_depth: BitDepth,
}

impl Default for Nvdec {
    fn default() -> Self {
        Nvdec {
            regs: vec![0; reg::COUNT],
            h264: HashMap::new(),
            vp9: Arc::default(),
            pictures: 0,
        }
    }
}

impl Nvdec {
    pub fn write(
        &mut self,
        dma: &Dma,
        context: u32,
        method: u32,
        value: u32,
    ) -> Result<(), String> {
        if let Some(r) = self.regs.get_mut(method as usize) {
            *r = value;
        }
        if method == reg::EXECUTE {
            return match self.regs[reg::APPLICATION_ID as usize] {
                CODEC_H264 => self.decode_h264(dma, context),
                CODEC_VP8 => Err(
                    "NVDEC: VP8 is not supported: the built-in decoders cover H.264 and VP9 only"
                        .into(),
                ),
                CODEC_VP9 => self.decode_vp9(dma, context),
                CODEC_AV1 => Err(
                    "NVDEC: AV1 is not supported: the built-in decoders cover H.264 and VP9 only"
                        .into(),
                ),
                id => Err(format!("NVDEC: codec {id} is not supported")),
            };
        }
        Ok(())
    }

    fn decode_h264(&mut self, dma: &Dma, context: u32) -> Result<(), String> {
        let setup = self.regs[reg::DRV_PIC_SETUP as usize];
        let raw = dma
            .read(setup, h264::PICTURE_INFO_SIZE)
            .ok_or_else(|| format!("NVDEC: picture setup {setup:#x} is not mapped"))?;
        let info = PictureInfo::parse(&raw);
        let base = self.regs[reg::IN_BUF_BASE as usize];
        let slices = dma
            .read(base, info.bitstream_size as usize)
            .ok_or_else(|| format!("NVDEC: bitstream {base:#x} is not mapped"))?;
        if info.field_pic || !info.frame_mbs_only {
            return Err("NVDEC: interlaced H.264 is not supported".into());
        }
        if slices.is_empty() {
            if trace() {
                eprintln!("nvdec: H.264 submit with an empty bitstream on channel {context}");
            }
            return Ok(());
        }
        if let Some(dir) = std::env::var_os("KIRIKAE_VIDEO_DUMP") {
            let path = std::path::Path::new(&dir).join(format!("nvdec-{context}.h264"));
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = f.write_all(&h264::parameter_sets(&info));
                let _ = f.write_all(&slices);
            }
        }
        // The surface table is rewritten by the guest between submissions, so
        // the addresses have to be captured with the access unit, not when its
        // picture finally comes out.
        let surface = (info.output_surface as usize).min(16);
        let luma =
            self.regs[reg::PICTURE_LUMA as usize + surface].wrapping_add(info.luma_frame_offset);
        let chroma = self.regs[reg::PICTURE_CHROMA as usize + surface]
            .wrapping_add(info.chroma_frame_offset);

        let channel = match self.h264.entry(context) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(H264Channel::new()?),
        };
        if contains_idr(&slices) {
            channel.period += 1;
            let keep_from = channel.period.saturating_sub(1);
            channel.pending.retain(|p| p.period >= keep_from);
        }
        channel.pending.push_back(PendingPicture {
            luma,
            chroma,
            poc: info.field_order_cnt[0],
            period: channel.period,
        });
        while channel.pending.len() > MAX_PENDING_PICTURES {
            channel.pending.pop_front();
        }
        let frame = match channel.decoder.decode(&info, &slices) {
            Ok(frame) => frame,
            Err(e) => {
                channel.pending.pop_back();
                return Err(e);
            }
        };
        self.pictures += 1;
        let Some(picture) = frame else {
            if trace() {
                eprintln!(
                    "nvdec: H.264 picture {} on channel {context}: {}x{} MBs, {} bytes in {} slices, frame_num {}, poc {}, surface {surface} -> no output yet",
                    self.pictures,
                    info.width_mbs,
                    info.height_mbs,
                    info.bitstream_size,
                    info.num_slices,
                    info.frame_num,
                    info.field_order_cnt[0],
                );
            }
            return Ok(());
        };
        let target = PendingPicture::take_for_output(&mut channel.pending, channel.period)
            .ok_or_else(|| "NVDEC: decoded picture has no pending surface".to_string())?;
        if trace() {
            eprintln!(
                "nvdec: H.264 picture {} on channel {context}: {}x{} MBs, {} bytes in {} slices, frame_num {}, poc {}, surface {surface} (poc {} at luma {:#x}, chroma {:#x}) -> decoded (8-bit)",
                self.pictures,
                info.width_mbs,
                info.height_mbs,
                info.bitstream_size,
                info.num_slices,
                info.frame_num,
                info.field_order_cnt[0],
                target.poc,
                (target.luma as u64) << 8,
                (target.chroma as u64) << 8,
            );
        }
        if !write_picture(dma, target.luma, target.chroma, &picture) {
            return Err(format!(
                "NVDEC: output surface at luma {:#x} / chroma {:#x} is not mapped",
                (target.luma as u64) << 8,
                (target.chroma as u64) << 8
            ));
        }
        clock::decoded();
        Ok(())
    }

    fn decode_vp9(&mut self, dma: &Dma, context: u32) -> Result<(), String> {
        let reg = |r: u32| self.regs[r as usize];
        let setup = reg(reg::DRV_PIC_SETUP);
        let raw = dma
            .read(setup, vp9::PICTURE_INFO_SIZE)
            .ok_or_else(|| format!("NVDEC: picture setup {setup:#x} is not mapped"))?;
        let info = vp9::PictureInfo::parse(&raw);
        let probs = dma
            .read(reg(reg::VP9_PROB_TAB), vp9::PROBS_SIZE)
            .ok_or("NVDEC: VP9 probability tables are not mapped")?;
        let tiles = dma
            .read(reg(reg::IN_BUF_BASE), info.bitstream_size as usize)
            .ok_or("NVDEC: VP9 bitstream is not mapped")?;
        let luma: [u32; 4] = std::array::from_fn(|i| self.regs[reg::PICTURE_LUMA as usize + i]);
        let chroma: [u32; 4] = std::array::from_fn(|i| self.regs[reg::PICTURE_CHROMA as usize + i]);
        let counter = reg(reg::VP9_CTX_COUNTER);
        let state = self.vp9.clone();
        let queued = std::time::Instant::now();
        dma.defer(move |dma| {
            state.lock().unwrap().decode(
                dma, context, &info, &probs, &tiles, luma, chroma, counter, queued,
            )
        })
    }
}

/// True when the access unit carries an IDR slice. Its picture order count
/// starts a new period, so pictures still queued from the previous period have
/// to keep their own entries (the decoder may still owe them).
fn contains_idr(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i + 3 < bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 {
            if bytes[i + 3] & 0x1f == 5 {
                return true;
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    false
}

/// Writes a decoded picture into the guest surface the picture setup points at.
///
/// The surface layout follows the picture's sample depth: NV12 for 8-bit
/// pictures and P010 for 10-bit ones (see [`super::surface`]). The luma and
/// chroma planes are block-linear with the NVDEC surface block height.
fn write_picture(dma: &Dma, luma: u32, chroma: u32, picture: &Yuv420) -> bool {
    let depth = picture.bit_depth;
    let (w, h) = (picture.width, picture.height);
    let (cw, ch) = picture.chroma();
    let (y, uv) = picture.pack(depth);
    let (luma_bytes, chroma_bytes) = (w * depth.bytes(), cw * 2 * depth.bytes());
    let y = to_block_linear(&y, luma_bytes, luma_bytes, h, SURFACE_BLOCK_HEIGHT_LOG2);
    let uv = to_block_linear(
        &uv,
        chroma_bytes,
        chroma_bytes,
        ch,
        SURFACE_BLOCK_HEIGHT_LOG2,
    );
    dma.write(luma, &y) && dma.write(chroma, &uv)
}

impl Vp9State {
    /// Reads the reference surfaces back from guest memory at `depth`.
    fn reference(
        &self,
        dma: &Dma,
        luma: u32,
        chroma: u32,
        size: (u32, u32),
        depth: BitDepth,
    ) -> Option<vp9::Picture> {
        if let Some(p) = self.pictures.get(luma) {
            return Some(p);
        }
        let (w, h) = (size.0 as usize, size.1 as usize);
        if w == 0 || h == 0 {
            return None;
        }
        let y = dma.read(
            luma,
            block_linear_size(w * depth.bytes(), h, SURFACE_BLOCK_HEIGHT_LOG2),
        )?;
        let uv = dma.read(
            chroma,
            block_linear_size(
                w.div_ceil(2) * 2 * depth.bytes(),
                h.div_ceil(2),
                SURFACE_BLOCK_HEIGHT_LOG2,
            ),
        )?;
        Some(vp9::picture_from_surface(&y, &uv, w, h, depth))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &mut self,
        dma: &Dma,
        context: u32,
        info: &vp9::PictureInfo,
        probs: &[u8],
        tiles: &[u8],
        luma: [u32; 4],
        chroma: [u32; 4],
        counter: u32,
        queued: std::time::Instant,
    ) -> Result<(), String> {
        let depth = self.bit_depth;
        let refs: [Option<vp9::Picture>; 3] =
            std::array::from_fn(|i| self.reference(dma, luma[i], chroma[i], info.sizes[i], depth));
        let start = std::time::Instant::now();
        let decoded = self
            .decoders
            .entry(context)
            .or_default()
            .decode(info, probs, refs, tiles)?;
        let took = start.elapsed();
        self.decoded += 1;
        dma.write(counter, &decoded.counts);
        let (w, h) = (info.width() as usize, info.height() as usize);
        let picture = vp9::yuv420(&decoded.picture)?;
        self.bit_depth = picture.bit_depth;
        if let Some(dir) =
            std::env::var_os("KIRIKAE_VIDEO_DUMP").filter(|_| self.decoded.is_multiple_of(100))
        {
            let path = std::path::Path::new(&dir).join(format!(
                "nvdec-{}-{w}x{h}-{}.yuv",
                self.decoded,
                picture.bit_depth.bits()
            ));
            let (y, uv) = picture.pack(picture.bit_depth);
            let _ = std::fs::write(path, [y.as_slice(), uv.as_slice()].concat());
        }
        if !write_picture(dma, luma[3], chroma[3], &picture) {
            return Err("NVDEC: VP9 output surface is not mapped".into());
        }
        self.pictures.put(luma[3], decoded.picture);
        clock::decoded();
        if trace() {
            eprintln!(
                "nvdec: {:.3} s: VP9 picture {} on channel {context}: {}x{}{}, {}-bit, flags {:#x}, {} bytes -> surface {:#x} in {took:.1?} (+{:.1?} output, {:.1?} after submission)",
                crate::gpu::gpu_ticks() as f64 / 1e9,
                self.decoded,
                info.width(),
                info.height(),
                if info.key_frame() { " key frame" } else { "" },
                picture.bit_depth.bits(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{Gpu, NvMap};
    use crate::host1x::{Iova, from_block_linear};
    use crate::mem::GuestMemory;

    const LUMA: usize = 0x2000;
    const CHROMA: usize = 0x4000;

    struct Harness {
        mem: GuestMemory,
        gpu: Gpu,
        iova: Iova,
        base: u64,
    }

    impl Harness {
        fn new() -> Harness {
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
            Harness {
                mem,
                gpu: Gpu::default(),
                iova,
                base,
            }
        }

        fn dma(&self) -> Dma<'_> {
            Dma {
                iova: &self.iova,
                mem: &self.mem,
                gpu: &self.gpu,
                later: None,
            }
        }

        /// IOVA offset of a buffer, in the 8-bit units the engine registers use.
        fn at(&self, offset: usize) -> u32 {
            ((self.base + offset as u64) >> 8) as u32
        }

        fn read_plane(&self, offset: usize, width_bytes: usize, height: usize) -> Vec<u8> {
            let raw = self
                .mem
                .read(
                    0x8000_0000 + offset as u64,
                    block_linear_size(width_bytes, height, SURFACE_BLOCK_HEIGHT_LOG2) as u64,
                )
                .unwrap();
            from_block_linear(&raw, width_bytes, height, SURFACE_BLOCK_HEIGHT_LOG2)
        }
    }

    fn words(bytes: &[u8]) -> Vec<u16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| u16::from_le_bytes(*w) >> 6)
            .collect()
    }

    #[test]
    fn writes_nv12_for_eight_bit_pictures() {
        let h = Harness::new();
        let picture = Yuv420 {
            width: 4,
            height: 2,
            bit_depth: BitDepth::Eight,
            y: vec![16, 32, 64, 128, 235, 200, 100, 0],
            uv: vec![128, 128, 90, 240],
        };
        assert!(write_picture(&h.dma(), h.at(LUMA), h.at(CHROMA), &picture));
        assert_eq!(
            h.read_plane(LUMA, 4, 2),
            picture.y.iter().map(|&v| v as u8).collect::<Vec<_>>()
        );
        assert_eq!(
            h.read_plane(CHROMA, 4, 1),
            picture.uv.iter().map(|&v| v as u8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn writes_p010_for_ten_bit_pictures() {
        let h = Harness::new();
        let picture = Yuv420 {
            width: 4,
            height: 2,
            bit_depth: BitDepth::Ten,
            y: vec![64, 940, 324, 580, 0, 1023, 512, 512],
            uv: vec![512, 512, 360, 960],
        };
        assert!(write_picture(&h.dma(), h.at(LUMA), h.at(CHROMA), &picture));
        // Two bytes per sample, so the block-linear planes are twice as wide.
        assert_eq!(words(&h.read_plane(LUMA, 8, 2)), picture.y);
        assert_eq!(words(&h.read_plane(CHROMA, 8, 1)), picture.uv);
        // The low six bits of every word are padding: the value sits in the top.
        let raw = h.read_plane(LUMA, 8, 2);
        assert!(raw.as_chunks::<2>().0.iter().all(|w| w[0] & 0x3f == 0));
    }

    #[test]
    fn pictures_come_out_in_display_order() {
        let mut pending = VecDeque::from(vec![
            PendingPicture {
                luma: 1,
                chroma: 2,
                poc: 0,
                period: 0,
            },
            PendingPicture {
                luma: 3,
                chroma: 4,
                poc: 3,
                period: 0,
            },
            PendingPicture {
                luma: 5,
                chroma: 6,
                poc: 1,
                period: 0,
            },
            PendingPicture {
                luma: 7,
                chroma: 8,
                poc: 2,
                period: 0,
            },
        ]);
        let mut order = Vec::new();
        while let Some(p) = PendingPicture::take_for_output(&mut pending, 0) {
            order.push((p.poc, p.luma));
        }
        assert_eq!(order, [(0, 1), (1, 5), (2, 7), (3, 3)]);
        assert!(pending.is_empty());
    }

    #[test]
    fn an_idr_period_keeps_what_the_decoder_still_owes() {
        let mut pending = VecDeque::from(vec![
            PendingPicture {
                luma: 1,
                chroma: 2,
                poc: 60,
                period: 1,
            },
            PendingPicture {
                luma: 3,
                chroma: 4,
                poc: 0,
                period: 2,
            },
        ]);
        let first = PendingPicture::take_for_output(&mut pending, 2).unwrap();
        assert_eq!(first.poc, 60, "the previous period's picture comes first");
        let second = PendingPicture::take_for_output(&mut pending, 2).unwrap();
        assert_eq!(second.poc, 0);
    }

    #[test]
    fn finds_the_idr_slice_in_an_access_unit() {
        assert!(contains_idr(&[0, 0, 0, 1, 0x65, 0x88, 0x80]));
        assert!(contains_idr(&[0, 0, 1, 0x65, 0x88]));
        assert!(!contains_idr(&[0, 0, 0, 1, 0x41, 0x9a]));
        assert!(!contains_idr(&[0, 0, 0, 1, 0x67, 0x64]));
    }

    #[test]
    fn unsupported_codecs_fail_loudly() {
        let h = Harness::new();
        let mut nvdec = Nvdec::default();
        for (codec, name) in [(CODEC_VP8, "VP8"), (CODEC_AV1, "AV1")] {
            nvdec
                .write(&h.dma(), 0, reg::APPLICATION_ID, codec)
                .unwrap();
            let error = nvdec.write(&h.dma(), 0, reg::EXECUTE, 0).unwrap_err();
            assert!(error.contains(name), "{error}");
            assert!(error.contains("not supported"), "{error}");
        }
        nvdec.write(&h.dma(), 0, reg::APPLICATION_ID, 42).unwrap();
        assert_eq!(
            nvdec.write(&h.dma(), 0, reg::EXECUTE, 0).unwrap_err(),
            "NVDEC: codec 42 is not supported"
        );
    }

    /// End-to-end path: bitstream extraction from `IN_BUF_BASE` at the size the
    /// picture setup reports, parameter set reconstruction, decode, and the
    /// block-linear NV12 write into the surface the setup points at. The
    /// decoder emits a picture one submission after the access unit that
    /// produced it, so the surface is checked after the follow-up submission.
    #[cfg(feature = "h264")]
    #[test]
    fn decodes_an_encoder_stream_into_the_surface() {
        use crate::host1x::h264::support::{setup_from, write_setup};
        use openh264::decoder::Decoder as OpenH264;
        use openh264::encoder::Encoder;
        use openh264::formats::{RgbSliceU8, YUVBuffer, YUVSource};

        const SETUP: usize = 0x0000;
        const BITSTREAM: usize = 0x1000;
        let (w, h) = (96usize, 64usize);
        let mut encoder = Encoder::new().unwrap();
        let frames: Vec<Vec<u8>> = (0..2)
            .map(|t| {
                let rgb: Vec<u8> = (0..w * h)
                    .flat_map(|i| {
                        [
                            (i % w * 2 + t * 11) as u8,
                            (i / w * 4 + t * 7) as u8,
                            (i * 5 + t * 3 % 251) as u8,
                        ]
                    })
                    .collect();
                encoder
                    .encode(&YUVBuffer::from_rgb8_source(RgbSliceU8::new(&rgb, (w, h))))
                    .unwrap()
                    .to_vec()
            })
            .collect();
        let nal_type = |n: &[u8]| n[n.iter().position(|&b| b == 1).unwrap() + 1] & 0x1f;
        let body = |n: &[u8]| n[n.iter().position(|&b| b == 1).unwrap() + 2..].to_vec();
        let nals: Vec<&[u8]> = openh264::nal_units(&frames[0]).collect();
        let sps = body(nals.iter().find(|n| nal_type(n) == 7).unwrap());
        let pps = body(nals.iter().find(|n| nal_type(n) == 8).unwrap());
        let slices = |frame: &[u8]| -> Vec<u8> {
            openh264::nal_units(frame)
                .filter(|n| matches!(nal_type(n), 1 | 5))
                .flatten()
                .copied()
                .collect()
        };
        let mut setup = setup_from(&sps, &pps);
        setup.num_slices = 1;
        setup.output_surface = 0;

        let harness = Harness::new();
        let mut nvdec = Nvdec::default();
        let dma = harness.dma();
        nvdec
            .write(&dma, 0, reg::APPLICATION_ID, CODEC_H264)
            .unwrap();
        nvdec
            .write(&dma, 0, reg::DRV_PIC_SETUP, harness.at(SETUP))
            .unwrap();
        nvdec
            .write(&dma, 0, reg::IN_BUF_BASE, harness.at(BITSTREAM))
            .unwrap();
        nvdec
            .write(&dma, 0, reg::PICTURE_LUMA, harness.at(LUMA))
            .unwrap();
        nvdec
            .write(&dma, 0, reg::PICTURE_CHROMA, harness.at(CHROMA))
            .unwrap();

        for (i, frame) in frames.iter().enumerate() {
            let au = slices(frame);
            setup.bitstream_size = au.len() as u32;
            assert!(
                harness
                    .mem
                    .write(0x8000_0000 + SETUP as u64, &write_setup(&setup))
            );
            assert!(harness.mem.write(0x8000_0000 + BITSTREAM as u64, &au));
            nvdec.write(&dma, 0, reg::EXECUTE, 0).unwrap();
            if i == 0 {
                assert_eq!(
                    harness.read_plane(LUMA, w, h),
                    vec![0u8; w * h],
                    "the priming submission writes nothing"
                );
            }
        }

        let mut reference = OpenH264::new().unwrap();
        let want = reference
            .decode(&frames[0])
            .unwrap()
            .expect("reference picture");
        let (sy, su, sv) = want.strides();
        let mut expected = Vec::new();
        for y in 0..h {
            expected.extend_from_slice(&want.y()[y * sy..y * sy + w]);
        }
        let mut expected_uv = Vec::new();
        for y in 0..h / 2 {
            let u = &want.u()[y * su..y * su + w / 2];
            let v = &want.v()[y * sv..y * sv + w / 2];
            for x in 0..w / 2 {
                expected_uv.extend_from_slice(&[u[x], v[x]]);
            }
        }
        assert_eq!(harness.read_plane(LUMA, w, h), expected);
        assert_eq!(harness.read_plane(CHROMA, w, h / 2), expected_uv);
    }
}
