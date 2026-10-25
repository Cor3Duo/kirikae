use std::collections::HashMap;
use std::sync::Arc;

use vp9dec::compressed_header::{CompressedHeader, CompressedHeaderProbs};
use vp9dec::counts::{Counts, TokenCounts};
use vp9dec::dpb::RefFrameData;
use vp9dec::framebuffer::Plane;
use vp9dec::header::{
    BILINEAR, CS_UNKNOWN, ColorConfig, EIGHTTAP, EIGHTTAP_SHARP, EIGHTTAP_SMOOTH, FrameType,
    LoopFilterParams, NewFrameHeader, QuantizationParams, SWITCHABLE, SegmentationParams,
};
use vp9dec::tile::{MiGrid, TileDecoder};

use super::surface::{BitDepth, Yuv420};
use super::{from_block_linear, u32_at};

pub const PICTURE_INFO_SIZE: usize = 0x100;
pub const PROBS_SIZE: usize = 0xea0;

const KEY_FRAME: u32 = 1 << 0;
const FRAME_SIZE_CHANGED: u32 = 1 << 2;
const ERROR_RESILIENT: u32 = 1 << 3;
const LAST_SHOW_FRAME: u32 = 1 << 4;
const INTRA_ONLY: u32 = 1 << 5;

#[derive(Clone, Debug)]
pub struct PictureInfo {
    pub bitstream_size: u32,
    pub sizes: [(u32, u32); 4],
    pub flags: u32,
    header: NewFrameHeader,
    tx_mode: u8,
    reference_mode: u8,
    comp_fixed_ref: u8,
    comp_var_ref: [u8; 2],
}

impl PictureInfo {
    pub fn parse(b: &[u8]) -> PictureInfo {
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as u32;
        let i8_at = |o: usize| b[o] as i8;
        let sizes = std::array::from_fn(|i| (u16_at(0x48 + 8 * i), u16_at(0x4a + 8 * i)));
        let flags = u32_at(b, 0x68);
        let intra_only = flags & INTRA_ONLY != 0;
        let key = flags & KEY_FRAME != 0;
        let (width, height) = sizes[3];
        let mut feature_enabled = [[false; 4]; 8];
        let mut feature_data = [[0i32; 4]; 8];
        for s in 0..8 {
            let mask = u32_at(b, 0x84 + 4 * s);
            for f in 0..4 {
                feature_enabled[s][f] = mask >> f & 1 != 0;
                feature_data[s][f] =
                    i16::from_le_bytes([b[0xa4 + 8 * s + 2 * f], b[0xa5 + 8 * s + 2 * f]]) as i32;
            }
        }
        let (base_q_idx, y_dc, uv_ac, uv_dc) = (
            b[0x72],
            i8_at(0x73) as i32,
            i8_at(0x74) as i32,
            i8_at(0x75) as i32,
        );
        let header = NewFrameHeader {
            profile: 0,
            frame_type: if key {
                FrameType::KeyFrame
            } else {
                FrameType::NonKeyFrame
            },
            show_frame: true,
            error_resilient_mode: flags & ERROR_RESILIENT != 0,
            frame_is_intra: key || intra_only,
            intra_only,
            reset_frame_context: 0,
            // Tegra X1's NVDEC decodes VP9 Profile 0 only (8-bit 4:2:0), and the
            // picture info we parse from the guest carries no profile field, so
            // every stream we can be handed is Profile 0. The surface writer and
            // the reference reload still handle 10-bit pictures for a backend
            // that reports Profile 2 (`vp9dec` decodes 8, 10 and 12 bits).
            color_config: Some(ColorConfig {
                bit_depth: 8,
                color_space: CS_UNKNOWN,
                color_range: false,
                subsampling_x: 1,
                subsampling_y: 1,
            }),
            width,
            height,
            render_width: width,
            render_height: height,
            refresh_frame_flags: 0,
            ref_frame_idx: [0, 1, 2],
            ref_frame_sign_bias: std::array::from_fn(|i| b[0x6c + i] != 0),
            allow_high_precision_mv: b[0x78] != 0,
            interpolation_filter: interpolation_filter(b[0x79]),
            refresh_frame_context: false,
            frame_parallel_decoding_mode: true,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                level: b[0x70],
                sharpness: b[0x71],
                delta_enabled: b[0xe4] != 0,
                ref_deltas: std::array::from_fn(|i| i8_at(0xe5 + i)),
                mode_deltas: std::array::from_fn(|i| i8_at(0xe9 + i)),
            },
            quantization: QuantizationParams {
                base_q_idx,
                delta_q_y_dc: y_dc,
                delta_q_uv_dc: uv_dc,
                delta_q_uv_ac: uv_ac,
                lossless: b[0x76] != 0,
            },
            segmentation: SegmentationParams {
                enabled: b[0x80] != 0,
                update_map: b[0x81] != 0,
                temporal_update: b[0x82] != 0,
                abs_or_delta_update: b[0x83] != 0,
                feature_enabled,
                feature_data,
                tree_probs: [255; 7],
                pred_prob: [255; 3],
            },
            tile_cols_log2: b[0x7e] as u32,
            tile_rows_log2: b[0x7f] as u32,
            header_size_in_bytes: 0,
        };
        PictureInfo {
            bitstream_size: u32_at(b, 0x30),
            sizes,
            flags,
            header,
            tx_mode: b[0x77],
            reference_mode: b[0x7a],
            comp_fixed_ref: b[0x7b],
            comp_var_ref: [b[0x7c], b[0x7d]],
        }
    }

    pub fn key_frame(&self) -> bool {
        self.flags & KEY_FRAME != 0
    }

    pub fn width(&self) -> u32 {
        self.header.width
    }

    pub fn height(&self) -> u32 {
        self.header.height
    }
}

fn interpolation_filter(literal: u8) -> u8 {
    match literal {
        0 => EIGHTTAP_SMOOTH,
        1 => EIGHTTAP,
        2 => EIGHTTAP_SHARP,
        3 => BILINEAR,
        _ => SWITCHABLE,
    }
}

struct Fields<'a> {
    b: &'a [u8],
    at: usize,
}

impl Fields<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let v = self.b[self.at..self.at + N].try_into().unwrap();
        self.at += N;
        v
    }

    fn skip(&mut self, n: usize) {
        self.at += n;
    }

    fn rows<const R: usize, const N: usize>(&mut self) -> [[u8; N]; R] {
        std::array::from_fn(|_| self.take::<N>())
    }
}

pub fn parse_probs(b: &[u8]) -> (CompressedHeaderProbs, [u8; 7], [u8; 3]) {
    let mut f = Fields { b, at: 0 };
    let mut p = CompressedHeaderProbs::default();
    f.skip(10 * 10 * 8 + 10 * 10 + 3);
    let seg_tree = f.take::<7>();
    let seg_pred = f.take::<3>();
    f.skip(15 + 10 * 8 + 10 + 6);
    let inter_mode: [[u8; 4]; 7] = f.rows();
    p.inter_mode_probs = inter_mode.map(|r| [r[0], r[1], r[2]]);
    p.is_inter_prob = f.take();
    let uv_mode: [[u8; 8]; 10] = f.rows();
    let tx8: [[u8; 1]; 2] = f.rows();
    let tx16: [[u8; 2]; 2] = f.rows();
    let tx32: [[u8; 3]; 2] = f.rows();
    for ctx in 0..2 {
        p.tx_probs[1][ctx][0] = tx8[ctx][0];
        p.tx_probs[2][ctx][..2].copy_from_slice(&tx16[ctx]);
        p.tx_probs[3][ctx] = tx32[ctx];
    }
    let y_mode_e8: [u8; 4] = f.take();
    let y_mode: [[u8; 8]; 4] = f.rows();
    for i in 0..4 {
        p.y_mode_probs[i][..8].copy_from_slice(&y_mode[i]);
        p.y_mode_probs[i][8] = y_mode_e8[i];
    }
    f.skip(16 * 4);
    let partition: [[u8; 4]; 16] = f.rows();
    p.partition_probs = partition.map(|r| [r[0], r[1], r[2]]);
    let uv_mode_e8: [u8; 10] = f.take();
    for i in 0..10 {
        p.uv_mode_probs[i][..8].copy_from_slice(&uv_mode[i]);
        p.uv_mode_probs[i][8] = uv_mode_e8[i];
    }
    p.interp_filter_probs = f.rows();
    p.comp_mode_prob = f.take();
    let skip: [u8; 4] = f.take();
    p.skip_prob = [skip[0], skip[1], skip[2]];
    p.mv_joint_probs = f.take();
    p.mv_sign_prob = f.take();
    p.mv_class0_bit_prob = f.take();
    p.mv_fr_probs = f.rows();
    p.mv_class0_hp_prob = f.take();
    p.mv_hp_prob = f.take();
    p.mv_class_probs = f.rows();
    p.mv_class0_fr_probs = [f.rows(), f.rows()];
    p.mv_bits_prob = f.rows();
    p.single_ref_prob = f.rows();
    p.comp_ref_prob = f.take();
    f.skip(17);
    for tx in &mut p.coef_probs {
        for plane in tx {
            for reference in plane {
                for band in reference {
                    for ctx in band {
                        let v: [u8; 4] = f.take();
                        *ctx = [v[0], v[1], v[2]];
                    }
                }
            }
        }
    }
    debug_assert_eq!(f.at, PROBS_SIZE);
    (p, seg_tree, seg_pred)
}

pub fn backward_updates(c: &Counts) -> Vec<u8> {
    let mut w: Vec<u32> = Vec::with_capacity(3316);
    for m in &c.inter_mode {
        w.extend_from_slice(&[m[2], m[0] + m[1] + m[3], m[0], m[1] + m[3], m[1], m[3]]);
    }
    w.extend(c.intra_mode.as_flattened());
    w.extend(c.uv_mode.as_flattened());
    w.extend(c.partition.as_flattened());
    w.extend(c.interp_filter.as_flattened());
    w.extend(c.is_inter.as_flattened());
    w.extend(c.comp_mode.as_flattened());
    w.extend(c.single_ref.as_flattened().as_flattened());
    w.extend(c.comp_ref.as_flattened());
    w.extend(c.tx_size[3].as_flattened());
    for ctx in &c.tx_size[2] {
        w.extend(&ctx[..3]);
    }
    for ctx in &c.tx_size[1] {
        w.extend(&ctx[..2]);
    }
    w.extend(c.skip.as_flattened());
    w.extend(&c.mv_joint);
    w.extend(c.mv_sign.as_flattened());
    w.extend(c.mv_class.as_flattened());
    w.extend(c.mv_class0_bit.as_flattened());
    w.extend(c.mv_bits.as_flattened().as_flattened());
    w.extend(c.mv_class0_fr.as_flattened().as_flattened());
    w.extend(c.mv_fr.as_flattened());
    w.extend(c.mv_class0_hp.as_flattened());
    w.extend(c.mv_hp.as_flattened());
    let flat = |t: &TokenCounts| -> Vec<[u32; 3]> {
        t.iter()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .copied()
            .collect()
    };
    let more: Vec<[u32; 2]> = c
        .more_coefs
        .iter()
        .flatten()
        .flatten()
        .flatten()
        .flatten()
        .copied()
        .collect();
    for (t, m) in flat(&c.token).iter().zip(&more) {
        w.extend_from_slice(&[t[0], t[1], t[2], m[0]]);
    }
    w.extend(more.iter().map(|m| m[0] + m[1]));
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub type Picture = Arc<RefFrameData>;

/// Rebuilds a reference picture from the guest surface it was written to.
///
/// `depth` is the depth the surface was written at, which is the depth of the
/// picture that produced it: 8-bit surfaces are NV12 and 10-bit ones are P010.
pub fn picture_from_surface(
    y: &[u8],
    uv: &[u8],
    width: usize,
    height: usize,
    depth: BitDepth,
) -> Picture {
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let (cw, ch) = (cw.max(1), ch.max(1));
    let mut py = Plane::new(width, height);
    for (d, s) in py
        .as_mut_slice()
        .iter_mut()
        .zip(samples(y, width, height, depth))
    {
        *d = s;
    }
    let uv = samples(uv, cw * 2, ch, depth);
    let (mut pu, mut pv) = (Plane::new(cw, ch), Plane::new(cw, ch));
    for (i, (u, v)) in pu
        .as_mut_slice()
        .iter_mut()
        .zip(pv.as_mut_slice().iter_mut())
        .enumerate()
    {
        *u = uv[2 * i];
        *v = uv[2 * i + 1];
    }
    Arc::new(RefFrameData {
        width: width as u32,
        height: height as u32,
        subsampling_x: 1,
        subsampling_y: 1,
        bit_depth: depth.bits() as u8,
        y: py,
        u: pu,
        v: pv,
    })
}

/// Reads a block-linear plane into samples at `depth`.
fn samples(raw: &[u8], width: usize, height: usize, depth: BitDepth) -> Vec<u16> {
    let rows = from_block_linear(raw, width * depth.bytes(), height, 1);
    match depth {
        BitDepth::Eight => rows.iter().map(|&v| v as u16).collect(),
        BitDepth::Ten => rows
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| u16::from_le_bytes(*w) >> 6)
            .collect(),
    }
}

/// Flattens a decoded picture into the shared 4:2:0 form used for the guest
/// surface. `vp9dec` keeps real samples for 10- and 12-bit streams, so the depth
/// travels with the picture instead of being assumed to be 8.
pub fn yuv420(p: &RefFrameData) -> Result<Yuv420, String> {
    let depth = BitDepth::from_bits(p.bit_depth).ok_or_else(|| {
        format!(
            "VP9 bit depth {} is not supported (8 and 10 bits only)",
            p.bit_depth
        )
    })?;
    Ok(Yuv420 {
        width: p.width as usize,
        height: p.height as usize,
        bit_depth: depth,
        y: p.y.as_slice().to_vec(),
        uv: p
            .u
            .as_slice()
            .iter()
            .zip(p.v.as_slice())
            .flat_map(|(&u, &v)| [u, v])
            .collect(),
    })
}

#[derive(Default)]
pub struct Decoder {
    prev_grid: Option<Arc<MiGrid>>,
    prev_size: (u32, u32),
    prev_segment_ids: Arc<Vec<u8>>,
}

pub struct Decoded {
    pub picture: Picture,
    pub counts: Vec<u8>,
}

impl Decoder {
    pub fn decode(
        &mut self,
        info: &PictureInfo,
        probs: &[u8],
        refs: [Option<Picture>; 3],
        tiles: &[u8],
    ) -> Result<Decoded, String> {
        let mut header = info.header.clone();
        let (probs, seg_tree, seg_pred) = parse_probs(probs);
        header.segmentation.tree_probs = seg_tree;
        header.segmentation.pred_prob = seg_pred;
        let (width, height) = (header.width, header.height);
        if width == 0 || height == 0 || width > 8192 || height > 8192 {
            return Err(format!("VP9 frame size {width}x{height}"));
        }
        let independent = header.frame_is_intra || header.error_resilient_mode;
        let use_prev_mvs = !independent
            && info.flags & (FRAME_SIZE_CHANGED | LAST_SHOW_FRAME) == LAST_SHOW_FRAME
            && self.prev_size == (width, height)
            && self.prev_grid.is_some();
        let mi_count = (width.div_ceil(8) * height.div_ceil(8)) as usize;
        if independent
            || self.prev_size != (width, height)
            || self.prev_segment_ids.len() != mi_count
        {
            self.prev_segment_ids = Arc::new(vec![0; mi_count]);
        }
        let refs = if header.frame_is_intra {
            [None, None, None]
        } else {
            refs
        };
        if !header.frame_is_intra && refs.iter().any(Option::is_none) {
            return Err("VP9 reference picture missing".into());
        }
        let compressed = CompressedHeader {
            tx_mode: info.tx_mode,
            probs: Arc::new(probs),
            reference_mode: info.reference_mode,
            comp_fixed_ref: info.comp_fixed_ref,
            comp_var_ref: info.comp_var_ref,
        };
        let color = header.color_config.unwrap();
        let prev_grid = if use_prev_mvs {
            self.prev_grid.clone()
        } else {
            None
        };
        let mut tiles_decoder = TileDecoder::new_with_prev(
            &header,
            color,
            &compressed,
            use_prev_mvs,
            prev_grid,
            refs,
            self.prev_segment_ids.clone(),
        );
        tiles_decoder
            .decode_tiles(tiles)
            .map_err(|e| format!("VP9 tiles: {e:?}"))?;
        if header.loop_filter.level != 0 {
            tiles_decoder.apply_loop_filter(&header.loop_filter);
        }
        let counts = backward_updates(tiles_decoder.counts());
        let planes = tiles_decoder.planes();
        let (w, h) = (width as usize, height as usize);
        let picture = Arc::new(RefFrameData {
            width,
            height,
            subsampling_x: 1,
            subsampling_y: 1,
            bit_depth: color.bit_depth,
            y: planes[0].crop_to_plane(w, h),
            u: planes[1].crop_to_plane(w.div_ceil(2), h.div_ceil(2)),
            v: planes[2].crop_to_plane(w.div_ceil(2), h.div_ceil(2)),
        });
        if header.segmentation.enabled && header.segmentation.update_map {
            let grid = tiles_decoder.mi_grid();
            let (cols, rows) = (width.div_ceil(8), height.div_ceil(8));
            self.prev_segment_ids = Arc::new(
                (0..rows)
                    .flat_map(|r| (0..cols).map(move |c| (r, c)))
                    .map(|(r, c)| grid.get(r, c).segment_id)
                    .collect(),
            );
        }
        self.prev_grid = Some(Arc::new(tiles_decoder.into_mi_grid()));
        self.prev_size = (width, height);
        Ok(Decoded { picture, counts })
    }
}

#[derive(Default)]
pub struct Pictures {
    by_surface: HashMap<u32, (u64, Picture)>,
    clock: u64,
}

impl Pictures {
    const KEEP: usize = 24;

    pub fn get(&self, luma: u32) -> Option<Picture> {
        self.by_surface.get(&luma).map(|(_, p)| p.clone())
    }

    pub fn put(&mut self, luma: u32, p: Picture) {
        self.clock += 1;
        self.by_surface.insert(luma, (self.clock, p));
        if self.by_surface.len() > Self::KEEP
            && let Some(&oldest) = self
                .by_surface
                .iter()
                .min_by_key(|(_, (t, _))| *t)
                .map(|(k, _)| k)
        {
            self.by_surface.remove(&oldest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts() {
        let d = CompressedHeaderProbs::default();
        let mut b = vec![0u8; PROBS_SIZE];
        b[0x400..0x41c]
            .copy_from_slice(&d.inter_mode_probs.map(|r| [r[0], r[1], r[2], 0]).concat());
        b[0x5a0..].copy_from_slice(
            &d.coef_probs
                .as_flattened()
                .as_flattened()
                .as_flattened()
                .as_flattened()
                .iter()
                .flat_map(|c| [c[0], c[1], c[2], 0])
                .collect::<Vec<_>>(),
        );
        b[0x4e0..0x520].copy_from_slice(&d.partition_probs.map(|r| [r[0], r[1], r[2], 0]).concat());
        b[0x56c..0x580].copy_from_slice(&d.mv_bits_prob.concat());
        let (p, _, _) = parse_probs(&b);
        assert_eq!(p.inter_mode_probs, d.inter_mode_probs);
        assert_eq!(p.coef_probs, d.coef_probs);
        assert_eq!(p.partition_probs, d.partition_probs);
        assert_eq!(p.mv_bits_prob, d.mv_bits_prob);
        assert_eq!(backward_updates(&Counts::new()).len(), 3316 * 4);
    }

    #[test]
    fn nv12_round_trip() {
        let (w, h) = (34usize, 18usize);
        let y: Vec<u8> = (0..w * h).map(|i| i as u8).collect();
        let uv: Vec<u8> = (0..w.div_ceil(2) * 2 * h.div_ceil(2))
            .map(|i| (i * 3) as u8)
            .collect();
        let p = picture_from_surface(
            &super::super::to_block_linear(&y, w, w, h, 1),
            &super::super::to_block_linear(&uv, w, w, h / 2, 1),
            w,
            h,
            BitDepth::Eight,
        );
        let picture = yuv420(&p).unwrap();
        assert_eq!(picture.bit_depth, BitDepth::Eight);
        assert_eq!(picture.pack(BitDepth::Eight), (y, uv));
    }

    #[test]
    fn p010_reference_round_trip() {
        let (w, h) = (18usize, 10usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y: Vec<u16> = (0..w * h).map(|i| (i * 7 % 1024) as u16).collect();
        let uv: Vec<u16> = (0..cw * 2 * ch).map(|i| (i * 11 % 1024) as u16).collect();
        let picture = Yuv420 {
            width: w,
            height: h,
            bit_depth: BitDepth::Ten,
            y: y.clone(),
            uv: uv.clone(),
        };
        let (packed_y, packed_uv) = picture.pack(BitDepth::Ten);
        let p = picture_from_surface(
            &super::super::to_block_linear(&packed_y, w * 2, w * 2, h, 1),
            &super::super::to_block_linear(&packed_uv, cw * 4, cw * 4, ch, 1),
            w,
            h,
            BitDepth::Ten,
        );
        assert_eq!(p.bit_depth, 10);
        assert_eq!(p.y.as_slice(), &y[..]);
        assert_eq!(
            p.u.as_slice()
                .iter()
                .zip(p.v.as_slice())
                .flat_map(|(&u, &v)| [u, v])
                .collect::<Vec<_>>(),
            uv
        );
    }
}
