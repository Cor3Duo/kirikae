#[cfg(feature = "h264")]
use openh264::OpenH264API;
#[cfg(feature = "h264")]
use openh264::decoder::{Decoder as OpenH264, DecoderConfig, Flush};
#[cfg(feature = "h264")]
use openh264::formats::YUVSource;

#[cfg(feature = "h264")]
use super::surface::BitDepth;
use super::surface::Yuv420;
use super::u32_at;

/// Frame cropping rectangle from the H.264 sequence parameter set, in luma
/// samples.
///
/// The coded picture is macroblock aligned (1920x1088 for 1080p, 912x512 for a
/// 900x506 stream); the SPS crop removes the padding so the picture has the
/// size the stream actually carries. Offsets are zero for a coded-size picture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Crop {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

impl Crop {
    pub const NONE: Crop = Crop {
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
    };

    pub fn is_zero(&self) -> bool {
        *self == Crop::NONE
    }

    /// Visible size of a `coded` picture, or `None` when the rectangle does not
    /// leave a non-empty picture inside it.
    pub fn apply(&self, coded: (u32, u32)) -> Option<(u32, u32)> {
        let (width, height) = coded;
        if self.left + self.right >= width || self.top + self.bottom >= height {
            return None;
        }
        Some((
            width - self.left - self.right,
            height - self.top - self.bottom,
        ))
    }
}

/// Sequence parameters the hardware picture setup does not carry and that have
/// to come from the stream's own SPS when it is in band.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpsParams {
    /// Cropping rectangle in luma samples.
    pub crop: Crop,
    /// `level_idc` from the stream, or 0 to derive one from the picture size.
    pub level_idc: u32,
    /// `max_num_ref_frames` from the stream, or 0 to derive it from the level.
    pub max_num_ref_frames: u32,
    /// `bit_depth_luma_minus8 + 8`, or 0 when no in-band SPS said otherwise.
    pub bit_depth_luma: u32,
    /// `bit_depth_chroma_minus8 + 8`, or 0 when no in-band SPS said otherwise.
    pub bit_depth_chroma: u32,
    /// `chroma_format_idc` from the stream, or 0 when no in-band SPS said so.
    pub chroma_format_idc: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PictureInfo {
    pub bitstream_size: u32,
    pub num_slices: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub delta_pic_order_always_zero: bool,
    pub frame_mbs_only: bool,
    pub width_mbs: u32,
    pub height_mbs: u32,
    pub entropy_coding_mode: bool,
    pub pic_order_present: bool,
    pub num_ref_idx_l0_active_minus1: u32,
    pub num_ref_idx_l1_active_minus1: u32,
    pub deblocking_filter_control_present: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    pub luma_frame_offset: u32,
    pub chroma_frame_offset: u32,
    pub mb_adaptive_frame_field: bool,
    pub direct_8x8_inference: bool,
    pub weighted_pred: bool,
    pub constrained_intra_pred: bool,
    pub field_pic: bool,
    pub log2_max_frame_num_minus4: u32,
    pub chroma_format_idc: u32,
    pub pic_order_cnt_type: u32,
    pub pic_init_qp_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub second_chroma_qp_index_offset: i32,
    pub weighted_bipred_idc: u32,
    pub output_surface: u32,
    pub frame_num: u32,
    /// Picture order counts the hardware parsed from the slice, one per field.
    /// For a frame picture the first entry is the display order of the picture.
    pub field_order_cnt: [i32; 2],
    pub scaling_4x4: [[u8; 16]; 6],
    pub scaling_8x8: [[u8; 64]; 2],
    pub qpprime_y_zero_transform_bypass: bool,
    /// Parameters only an in-band SPS can supply; see [`in_band_sps`].
    pub sps: SpsParams,
}

pub const PICTURE_INFO_SIZE: usize = 0x2fc;

impl PictureInfo {
    pub fn parse(b: &[u8]) -> PictureInfo {
        let flag = |o: usize| u32_at(b, o) != 0;
        let flags = u32_at(b, 0xb0) as u64 | (u32_at(b, 0xb4) as u64) << 32;
        let field = |lsb: u32, len: u32| ((flags >> lsb) & ((1 << len) - 1)) as u32;
        let signed = |lsb: u32, len: u32| ((flags << (64 - lsb - len)) as i64 >> (64 - len)) as i32;
        let mut scaling_4x4 = [[0u8; 16]; 6];
        for (i, l) in scaling_4x4.iter_mut().enumerate() {
            l.copy_from_slice(&b[0x1c0 + 16 * i..0x1d0 + 16 * i]);
        }
        let mut scaling_8x8 = [[0u8; 64]; 2];
        for (i, l) in scaling_8x8.iter_mut().enumerate() {
            l.copy_from_slice(&b[0x220 + 64 * i..0x260 + 64 * i]);
        }
        PictureInfo {
            bitstream_size: u32_at(b, 0x48),
            num_slices: u32_at(b, 0x4c),
            log2_max_pic_order_cnt_lsb_minus4: u32_at(b, 0x58),
            delta_pic_order_always_zero: flag(0x5c),
            frame_mbs_only: flag(0x60),
            width_mbs: u32_at(b, 0x64),
            height_mbs: u32_at(b, 0x68),
            entropy_coding_mode: flag(0x70),
            pic_order_present: flag(0x74),
            num_ref_idx_l0_active_minus1: u32_at(b, 0x78),
            num_ref_idx_l1_active_minus1: u32_at(b, 0x7c),
            deblocking_filter_control_present: flag(0x80),
            redundant_pic_cnt_present: flag(0x84),
            transform_8x8_mode: flag(0x88),
            luma_frame_offset: u32_at(b, 0x9c),
            chroma_frame_offset: u32_at(b, 0xa8),
            mb_adaptive_frame_field: field(0, 1) != 0,
            direct_8x8_inference: field(1, 1) != 0,
            weighted_pred: field(2, 1) != 0,
            constrained_intra_pred: field(3, 1) != 0,
            field_pic: field(5, 1) != 0,
            log2_max_frame_num_minus4: field(8, 4),
            chroma_format_idc: field(12, 2),
            pic_order_cnt_type: field(14, 2),
            pic_init_qp_minus26: signed(16, 6),
            chroma_qp_index_offset: signed(22, 5),
            second_chroma_qp_index_offset: signed(27, 5),
            weighted_bipred_idc: field(32, 2),
            output_surface: field(34, 7),
            frame_num: field(46, 16),
            field_order_cnt: [u32_at(b, 0xb8) as i32, u32_at(b, 0xbc) as i32],
            scaling_4x4,
            scaling_8x8,
            qpprime_y_zero_transform_bypass: u32_at(b, 0x2d0) & 2 != 0,
            sps: SpsParams::default(),
        }
    }

    /// Coded picture size in luma samples.
    pub fn coded_size(&self) -> (u32, u32) {
        (
            self.width_mbs.max(1) * 16,
            self.height_mbs.max(1) * 16 * (2 - self.frame_mbs_only as u32),
        )
    }

    /// Display size after the SPS crop, or the coded size without one.
    pub fn display_size(&self) -> (u32, u32) {
        self.sps
            .crop
            .apply(self.coded_size())
            .unwrap_or_else(|| self.coded_size())
    }
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bits: u32,
    acc: u8,
}

impl BitWriter {
    fn u(&mut self, value: u32, len: u32) {
        for i in (0..len).rev() {
            self.acc = self.acc << 1 | ((value >> i) & 1) as u8;
            self.bits += 1;
            if self.bits == 8 {
                self.bytes.push(self.acc);
                (self.acc, self.bits) = (0, 0);
            }
        }
    }

    fn flag(&mut self, b: bool) {
        self.u(b as u32, 1);
    }

    fn ue(&mut self, v: u32) {
        let len = 32 - (v + 1).leading_zeros();
        self.u(0, len - 1);
        self.u(v + 1, len);
    }

    fn se(&mut self, v: i32) {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            2 * v.unsigned_abs()
        });
    }

    fn nal(mut self, header: u8, out: &mut Vec<u8>) {
        self.u(1, 1);
        if self.bits > 0 {
            self.u(0, 8 - self.bits);
        }
        out.extend_from_slice(&[0, 0, 0, 1, header]);
        let mut zeros = 0;
        for b in self.bytes {
            if zeros >= 2 && b <= 3 {
                out.push(3);
                zeros = 0;
            }
            out.push(b);
            zeros = if b == 0 { zeros + 1 } else { 0 };
        }
    }
}

const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

fn scaling_list(w: &mut BitWriter, list: &[u8], scan: &[usize]) {
    let mut last = 8i32;
    for &i in scan {
        let v = list[i] as i32;
        let delta = (v - last + 128).rem_euclid(256) - 128;
        w.se(delta);
        last = v;
    }
}

/// OpenH264's `MAX_REF_PIC_COUNT`. Its SPS parser rejects anything larger and
/// its picture pool is `max_num_ref_frames + 2`.
pub const MAX_REFERENCE_FRAMES: u32 = 16;
/// Highest `level_idc` in OpenH264's level table (5.2). Higher levels are
/// rejected by its SPS parser.
const MAX_LEVEL_IDC: u32 = 52;
/// Annex A DPB limits: (`level_idc`, MaxFS in macroblocks, MaxDpbMbs).
const LEVEL_LIMITS: [(u32, u32, u32); 16] = [
    (10, 99, 396),
    (11, 396, 900),
    (12, 396, 2376),
    (13, 396, 2376),
    (20, 396, 2376),
    (21, 792, 4752),
    (22, 1620, 8100),
    (30, 1620, 8100),
    (31, 3600, 18000),
    (32, 5120, 20480),
    (40, 8192, 32768),
    (41, 8192, 32768),
    (42, 8704, 34816),
    (50, 22080, 110400),
    (51, 36864, 184320),
    (52, 36864, 184320),
];

fn level_limits(level_idc: u32) -> Option<(u32, u32)> {
    LEVEL_LIMITS
        .iter()
        .find(|(level, _, _)| *level == level_idc)
        .map(|(_, max_fs, max_dpb)| (*max_fs, *max_dpb))
}

/// Lowest level that allows [`MAX_REFERENCE_FRAMES`] for a picture of
/// `size_mbs` macroblocks, capped at 5.2 (OpenH264's last level).
fn level_for(size_mbs: u32) -> u32 {
    let wanted = size_mbs.saturating_mul(MAX_REFERENCE_FRAMES);
    LEVEL_LIMITS
        .iter()
        .find(|(_, max_fs, max_dpb)| size_mbs <= *max_fs && wanted <= *max_dpb)
        .map_or(MAX_LEVEL_IDC, |(level, _, _)| *level)
}

/// `max_num_ref_frames` for `size_mbs` macroblocks at `level_idc`: the level's
/// DPB budget divided by the picture size, clamped to OpenH264's 16 frames.
fn reference_frames(level_idc: u32, size_mbs: u32) -> Result<u32, String> {
    let (max_fs, max_dpb) = level_limits(level_idc)
        .ok_or_else(|| format!("H.264 level {level_idc} is not in OpenH264's level table"))?;
    if size_mbs > max_fs {
        return Err(format!(
            "H.264 picture of {size_mbs} macroblocks exceeds level {level_idc} ({max_fs} macroblocks)"
        ));
    }
    let frames = max_dpb / size_mbs;
    if frames == 0 {
        return Err(format!(
            "H.264 level {level_idc} DPB ({max_dpb} MBs) cannot hold a {size_mbs}-MB picture"
        ));
    }
    Ok(frames.min(MAX_REFERENCE_FRAMES))
}

/// Chroma subsampling of the crop offsets: (horizontal, vertical) luma samples
/// per SPS crop unit.
fn crop_unit(chroma_format_idc: u32, frame_mbs_only: bool) -> (u32, u32) {
    let (sub_width, sub_height) = match chroma_format_idc {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    let unit_x = if chroma_format_idc == 0 { 1 } else { sub_width };
    let unit_y = if chroma_format_idc == 0 {
        1
    } else {
        sub_height
    };
    (unit_x, unit_y * (2 - frame_mbs_only as u32))
}

/// The `level_idc` and `max_num_ref_frames` the rebuilt SPS declares.
///
/// A level from an in-band SPS is kept, and so is the reference count it names
/// as long as OpenH264 can hold it. Without a level, the lowest one that keeps
/// [`MAX_REFERENCE_FRAMES`] is used, so the reference count never silently drops
/// below what the encoder may rely on.
pub fn level_and_refs(p: &PictureInfo) -> Result<(u32, u32), String> {
    let size_mbs = p.width_mbs.max(1) * p.height_mbs.max(1);
    let level = if p.sps.level_idc != 0 {
        p.sps.level_idc
    } else {
        level_for(size_mbs)
    };
    if level_limits(level).is_none() {
        return Err(format!(
            "H.264 level {level} is not in OpenH264's level table (max {MAX_LEVEL_IDC})"
        ));
    }
    let refs = if p.sps.max_num_ref_frames != 0 {
        if p.sps.max_num_ref_frames > MAX_REFERENCE_FRAMES {
            return Err(format!(
                "H.264 stream asks for {} reference frames, more than OpenH264's {MAX_REFERENCE_FRAMES}",
                p.sps.max_num_ref_frames
            ));
        }
        p.sps.max_num_ref_frames
    } else {
        reference_frames(level, size_mbs)?
    };
    Ok((level, refs))
}

pub fn parameter_sets(p: &PictureInfo) -> Result<Vec<u8>, String> {
    let (coded_width, coded_height) = p.coded_size();
    let (level, refs) = level_and_refs(p)?;
    let crop = p.sps.crop;
    let (unit_x, unit_y) = crop_unit(p.chroma_format_idc, p.frame_mbs_only);
    if !crop.is_zero() {
        if crop.apply((coded_width, coded_height)).is_none() {
            return Err(format!(
                "H.264 crop {crop:?} does not fit the {coded_width}x{coded_height} coded picture"
            ));
        }
        if !crop.left.is_multiple_of(unit_x)
            || !crop.right.is_multiple_of(unit_x)
            || !crop.top.is_multiple_of(unit_y)
            || !crop.bottom.is_multiple_of(unit_y)
        {
            return Err(format!(
                "H.264 crop {crop:?} is not a multiple of its {unit_x}x{unit_y} crop unit"
            ));
        }
    }

    let mut out = Vec::new();
    let mut w = BitWriter::default();
    w.u(100, 8);
    w.u(0, 8);
    w.u(level, 8);
    w.ue(0);
    w.ue(p.chroma_format_idc);
    if p.chroma_format_idc == 3 {
        w.flag(false);
    }
    // bit_depth_luma_minus8 and bit_depth_chroma_minus8: OpenH264 decodes 8-bit
    // profiles only, so the rebuilt SPS always declares 8 bits.
    w.ue(0);
    w.ue(0);
    w.flag(p.qpprime_y_zero_transform_bypass);
    w.flag(false);
    w.ue(p.log2_max_frame_num_minus4);
    w.ue(p.pic_order_cnt_type);
    match p.pic_order_cnt_type {
        0 => w.ue(p.log2_max_pic_order_cnt_lsb_minus4),
        1 => {
            w.flag(p.delta_pic_order_always_zero);
            w.se(0);
            w.se(0);
            w.ue(0);
        }
        _ => {}
    }
    w.ue(refs);
    w.flag(false);
    w.ue(p.width_mbs.max(1) - 1);
    w.ue((p.height_mbs >> !p.frame_mbs_only as u32).max(1) - 1);
    w.flag(p.frame_mbs_only);
    if !p.frame_mbs_only {
        w.flag(p.mb_adaptive_frame_field);
    }
    w.flag(p.direct_8x8_inference);
    w.flag(!crop.is_zero());
    if !crop.is_zero() {
        w.ue(crop.left / unit_x);
        w.ue(crop.right / unit_x);
        w.ue(crop.top / unit_y);
        w.ue(crop.bottom / unit_y);
    }
    w.flag(false);
    w.nal(0x67, &mut out);

    let mut w = BitWriter::default();
    w.ue(0);
    w.ue(0);
    w.flag(p.entropy_coding_mode);
    w.flag(p.pic_order_present);
    w.ue(0);
    w.ue(p.num_ref_idx_l0_active_minus1);
    w.ue(p.num_ref_idx_l1_active_minus1);
    w.flag(p.weighted_pred);
    w.u(p.weighted_bipred_idc, 2);
    w.se(p.pic_init_qp_minus26);
    w.se(0);
    w.se(p.chroma_qp_index_offset);
    w.flag(p.deblocking_filter_control_present);
    w.flag(p.constrained_intra_pred);
    w.flag(p.redundant_pic_cnt_present);
    w.flag(p.transform_8x8_mode);
    w.flag(true);
    for l in &p.scaling_4x4 {
        w.flag(true);
        scaling_list(&mut w, l, &ZIGZAG_4X4);
    }
    if p.transform_8x8_mode {
        for l in &p.scaling_8x8 {
            w.flag(true);
            scaling_list(&mut w, l, &ZIGZAG_8X8);
        }
    }
    w.se(p.second_chroma_qp_index_offset);
    w.nal(0x68, &mut out);
    Ok(out)
}

/// Reads the first SPS in an access unit, if the stream carries one in band.
///
/// The hardware picture setup supplies most sequence parameters, but not the
/// cropping rectangle, the level or `max_num_ref_frames`; when the title passes
/// its own parameter sets alongside the slices these values are available.
/// `None` means the access unit has no SPS (the usual case: parameter sets go
/// through the picture setup only).
pub fn in_band_sps(bytes: &[u8]) -> Option<SpsParams> {
    let mut i = 0;
    while i + 3 < bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 {
            let header = bytes[i + 3];
            let start = i + 4;
            if header & 0x1f == 7 {
                let mut end = bytes.len();
                let mut j = start;
                while j + 3 < bytes.len() {
                    if bytes[j] == 0 && bytes[j + 1] == 0 && bytes[j + 2] == 1 {
                        end = j;
                        break;
                    }
                    j += 1;
                }
                let rbsp = unescape(&bytes[start..end]);
                return parse_sps(&rbsp);
            }
            i = start;
        } else {
            i += 1;
        }
    }
    None
}

/// Removes the emulation-prevention bytes (0x03 after two zeros) from an EBSP.
fn unescape(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0;
    for &b in ebsp {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

struct SpsBits<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl SpsBits<'_> {
    fn u(&mut self, n: usize) -> Option<u32> {
        let mut v = 0;
        for _ in 0..n {
            let b = (*self.bytes.get(self.bit / 8)? >> (7 - self.bit % 8)) & 1;
            self.bit += 1;
            v = v << 1 | b as u32;
        }
        Some(v)
    }

    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.u(1)? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        ((1u32 << zeros) - 1).checked_add(self.u(zeros as usize)?)
    }

    fn se(&mut self) -> Option<i32> {
        let v = self.ue()? as i32;
        Some(if v & 1 == 1 { (v + 1) / 2 } else { -v / 2 })
    }

    /// Skips one of the scaling lists `seq_scaling_matrix_present_flag` announces.
    fn scaling_list(&mut self, size: usize) -> Option<()> {
        let mut last = 8i32;
        let mut next = 8i32;
        for _ in 0..size {
            if next != 0 {
                next = (last + self.se()? + 256) % 256;
            }
            last = if next == 0 { last } else { next };
        }
        Some(())
    }
}

fn parse_sps(rbsp: &[u8]) -> Option<SpsParams> {
    let mut r = SpsBits {
        bytes: rbsp,
        bit: 0,
    };
    let profile = r.u(8)?;
    r.u(8)?; // constraint_set flags and reserved bits
    let level_idc = r.u(8)?;
    r.ue()?; // seq_parameter_set_id
    let mut chroma_format_idc = 1;
    let (mut bit_depth_luma, mut bit_depth_chroma) = (8, 8);
    if matches!(
        profile,
        44 | 83 | 86 | 100 | 110 | 118 | 122 | 128 | 134 | 135 | 138 | 139 | 244
    ) {
        chroma_format_idc = r.ue()?;
        if chroma_format_idc > 3 {
            return None;
        }
        if chroma_format_idc == 3 {
            r.u(1)?; // separate_colour_plane_flag
        }
        bit_depth_luma = r.ue()? + 8;
        bit_depth_chroma = r.ue()? + 8;
        r.u(1)?; // qpprime_y_zero_transform_bypass_flag
        if r.u(1)? == 1 {
            let lists = if chroma_format_idc == 3 { 12 } else { 8 };
            for i in 0..lists {
                if r.u(1)? == 1 {
                    r.scaling_list(if i < 6 { 16 } else { 64 })?;
                }
            }
        }
    }
    r.ue()?; // log2_max_frame_num_minus4
    match r.ue()? {
        0 => {
            r.ue()?; // log2_max_pic_order_cnt_lsb_minus4
        }
        1 => {
            r.u(1)?; // delta_pic_order_always_zero_flag
            r.se()?; // offset_for_non_ref_pic
            r.se()?; // offset_for_top_to_bottom_field
            for _ in 0..r.ue()? {
                r.se()?;
            }
        }
        _ => {}
    }
    let max_num_ref_frames = r.ue()?;
    r.u(1)?; // gaps_in_frame_num_value_allowed_flag
    let width_mbs = r.ue()? + 1;
    let height_map_units = r.ue()? + 1;
    let frame_mbs_only = r.u(1)? == 1;
    if !frame_mbs_only {
        r.u(1)?; // mb_adaptive_frame_field_flag
    }
    r.u(1)?; // direct_8x8_inference_flag
    let mut crop = Crop::NONE;
    if r.u(1)? == 1 {
        let raw = Crop {
            left: r.ue()?,
            right: r.ue()?,
            top: r.ue()?,
            bottom: r.ue()?,
        };
        let (unit_x, unit_y) = crop_unit(chroma_format_idc, frame_mbs_only);
        crop = Crop {
            left: raw.left * unit_x,
            right: raw.right * unit_x,
            top: raw.top * unit_y,
            bottom: raw.bottom * unit_y,
        };
        let coded = (
            width_mbs * 16,
            height_map_units * 16 * (2 - frame_mbs_only as u32),
        );
        if crop.apply(coded).is_none() {
            crop = Crop::NONE;
        }
    }
    Some(SpsParams {
        crop,
        level_idc,
        max_num_ref_frames,
        bit_depth_luma,
        bit_depth_chroma,
        chroma_format_idc,
    })
}

#[cfg(not(feature = "h264"))]
pub struct Decoder;

#[cfg(not(feature = "h264"))]
impl Decoder {
    pub fn new() -> Result<Decoder, String> {
        Err("H.264 support was not built in".into())
    }

    pub fn decode(
        &mut self,
        _info: &PictureInfo,
        _slices: &[u8],
    ) -> Result<Option<Yuv420>, String> {
        Err("H.264 support was not built in".into())
    }

    pub fn flush(&mut self) -> Result<Vec<Yuv420>, String> {
        Err("H.264 support was not built in".into())
    }
}

#[cfg(feature = "h264")]
fn open_decoder() -> Result<OpenH264, String> {
    // Pictures are *not* flushed after every submission. `FlushFrame` keeps
    // the flushed picture marked as a DPB reference (OpenH264 never drops it
    // through the sliding window), so a flush per picture exhausts the
    // `max_num_ref_frames + 2` picture buffer after a handful of GOPs: the
    // decoder then reports ERR_INFO_REF_COUNT_OVERFLOW as dsOutOfMemory
    // (16384) and every later call fails with dsNoParamSets (16).
    //
    // Without the flush the decoder emits the picture one submission later
    // and in display order, and no references leak: 213 submissions of a
    // 1920x1088 High stream with B-frames decode with no error. `None`
    // therefore means "no picture this submission", not a lost frame. The
    // caller matches each picture to the access unit that produced it by
    // picture order count (`Nvdec` keeps the surface queue).
    let config = DecoderConfig::new().flush_after_decode(Flush::NoFlush);
    OpenH264::with_api_config(OpenH264API::from_source(), config)
        .map_err(|e| format!("OpenH264: {e}"))
}

#[cfg(feature = "h264")]
pub struct Decoder {
    inner: OpenH264,
    headers: Vec<u8>,
    /// Access units accepted by the decoder whose picture has not come out.
    owed: usize,
}

#[cfg(feature = "h264")]
impl Decoder {
    pub fn new() -> Result<Decoder, String> {
        Ok(Decoder {
            inner: open_decoder()?,
            headers: Vec::new(),
            owed: 0,
        })
    }

    pub fn decode(&mut self, info: &PictureInfo, slices: &[u8]) -> Result<Option<Yuv420>, String> {
        // The picture setup describes the chroma format the slices were encoded
        // with. Only 4:2:0 (idc 1) has an NV12/P010 surface; anything else would
        // be written into a surface of the wrong shape. Idc 0 is what the setup
        // block reads as before it is filled in, so it is treated as 4:2:0 too.
        // An in-band SPS is authoritative when the setup left the field unset.
        let chroma = if info.sps.chroma_format_idc != 0 {
            info.sps.chroma_format_idc
        } else {
            info.chroma_format_idc
        };
        if chroma > 1 {
            return Err(format!(
                "H.264 chroma format {chroma} is not supported (4:2:0 only)"
            ));
        }
        // OpenH264 only decodes 8-bit profiles. Its SPS parser accepts High 10
        // but the reconstruction would be wrong, so a 10-bit stream is rejected
        // here. A 10-bit backend would have to come from a permissive decoder
        // (FFmpeg/libavcodec, LGPL, behind its own feature); none is linked in.
        if info.sps.bit_depth_luma > 8 || info.sps.bit_depth_chroma > 8 {
            return Err(format!(
                "H.264 {}-bit pictures are not supported: OpenH264 decodes 8-bit 4:2:0 only (a 10-bit FFmpeg/libavcodec backend is not built in)",
                info.sps.bit_depth_luma.max(info.sps.bit_depth_chroma)
            ));
        }
        let headers = parameter_sets(info)?;
        let mut packet = Vec::with_capacity(headers.len() + slices.len());
        if headers != self.headers {
            packet.extend_from_slice(&headers);
            self.headers = headers;
        }
        packet.extend_from_slice(slices);
        let counted = !slices.is_empty();
        let decoded = self.inner.decode(&packet);
        if counted {
            self.owed += 1;
        }
        let result = match decoded {
            Ok(Some(yuv)) => Ok(Some(picture(&yuv))),
            Ok(None) => Ok(None),
            Err(e) => Err(format!("H.264 picture not decoded: {e}")),
        };
        if counted && (matches!(result, Ok(Some(_))) || result.is_err()) {
            self.owed = self.owed.saturating_sub(1);
        }
        result
    }

    /// Drains the pictures OpenH264 still holds for this stream.
    ///
    /// The decoder emits a picture one submission after the access unit that
    /// produced it, so a stream that ends without another submission leaves its
    /// last picture (up to a whole reorder delay) in the decoder. Only the
    /// pictures this decoder knows it owes are returned: `flush_remaining` also
    /// keeps flushing until OpenH264 reports no frames left in its buffer, which
    /// includes reference pictures and can repeat the last output.
    ///
    /// This is only called when the stream ends. Flushing keeps the flushed
    /// pictures marked as references, which is harmless once nothing else is
    /// decoded; a stream that resumes is expected to start with an IDR and gets
    /// the parameter sets again (`headers` is cleared).
    pub fn flush(&mut self) -> Result<Vec<Yuv420>, String> {
        if self.owed == 0 {
            return Ok(Vec::new());
        }
        let wanted = self.owed;
        let pictures: Vec<Yuv420> = self
            .inner
            .flush_remaining()
            .map_err(|e| format!("H.264 flush failed: {e}"))?
            .iter()
            .take(wanted)
            .map(picture)
            .collect();
        // `FlushFrame` leaves the flushed pictures in the DPB as references, and
        // OpenH264 re-emits the last one with the next decode. A stream that
        // resumes after its end starts with an IDR, so a fresh decoder is both
        // correct and the only way to avoid presenting a frame twice. `headers`
        // is empty so the next access unit carries its parameter sets again.
        self.inner = open_decoder()?;
        self.headers.clear();
        self.owed = 0;
        Ok(pictures)
    }
}

/// Copies a decoded picture, widening its samples to `u16`.
///
/// OpenH264 hands back the cropped picture: `dimensions()` already excludes the
/// SPS cropping rectangle and the plane pointers are advanced to its top-left,
/// while `strides()` still describe the coded surface, so reading
/// `width`/`height` samples from each row yields the display picture.
#[cfg(feature = "h264")]
fn picture(yuv: &openh264::decoder::DecodedYUV<'_>) -> Yuv420 {
    let (width, height) = yuv.dimensions();
    let (sy, su, sv) = yuv.strides();
    let mut y = Vec::with_capacity(width * height);
    for row in yuv.y().chunks(sy).take(height) {
        y.extend(row[..width.min(row.len())].iter().map(|&v| v as u16));
    }
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let mut uv = vec![0u16; cw * 2 * ch];
    for r in 0..ch {
        let u = &yuv.u()[r * su..r * su + cw];
        let v = &yuv.v()[r * sv..r * sv + cw];
        for (i, d) in uv[r * cw * 2..(r + 1) * cw * 2]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .enumerate()
        {
            d[0] = u[i] as u16;
            d[1] = v[i] as u16;
        }
    }
    Yuv420 {
        width,
        height,
        // OpenH264 only decodes 8-bit profiles; a 10-bit stream (High 10)
        // is rejected in `decode` before the decoder sees it.
        bit_depth: BitDepth::Eight,
        y,
        uv,
    }
}

#[cfg(all(test, not(feature = "h264")))]
mod no_backend {
    use super::*;

    #[test]
    fn reports_the_missing_backend() {
        let error = match Decoder::new() {
            Ok(_) => panic!("the decoder exists without the h264 feature"),
            Err(e) => e,
        };
        assert!(error.contains("not built in"), "{error}");
        let mut decoder = Decoder;
        let info = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        assert!(
            decoder
                .decode(&info, &[])
                .unwrap_err()
                .contains("not built in")
        );
    }
}

#[cfg(all(test, feature = "h264"))]
pub(crate) mod support {
    use super::*;

    pub(crate) struct Bits<'a>(pub(crate) &'a [u8], pub(crate) usize);

    impl Bits<'_> {
        pub(crate) fn u(&mut self, n: usize) -> u32 {
            (0..n).fold(0, |v, _| {
                let b = (self.0[self.1 / 8] >> (7 - self.1 % 8)) & 1;
                self.1 += 1;
                v << 1 | b as u32
            })
        }
        pub(crate) fn ue(&mut self) -> u32 {
            let mut zeros = 0;
            while self.u(1) == 0 {
                zeros += 1;
            }
            (1 << zeros) - 1 + self.u(zeros)
        }
        pub(crate) fn se(&mut self) -> i32 {
            let v = self.ue() as i32;
            if v & 1 == 1 { (v + 1) / 2 } else { -v / 2 }
        }
    }

    pub(crate) fn rbsp(nal: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for &b in nal {
            if b == 3 && out.ends_with(&[0, 0]) {
                continue;
            }
            out.push(b);
        }
        out
    }

    /// Builds the picture setup the hardware would report for an SPS/PPS pair.
    pub(crate) fn setup_from(sps: &[u8], pps: &[u8]) -> PictureInfo {
        let mut p = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        let sps = rbsp(sps);
        let mut r = Bits(&sps, 0);
        assert_eq!(r.u(8), 66, "Baseline profile expected");
        r.u(16);
        assert_eq!(r.ue(), 0);
        p.chroma_format_idc = 1;
        p.log2_max_frame_num_minus4 = r.ue();
        p.pic_order_cnt_type = r.ue();
        match p.pic_order_cnt_type {
            0 => p.log2_max_pic_order_cnt_lsb_minus4 = r.ue(),
            1 => panic!("POC type 1"),
            _ => {}
        }
        r.ue();
        r.u(1);
        p.width_mbs = r.ue() + 1;
        p.height_mbs = r.ue() + 1;
        p.frame_mbs_only = r.u(1) == 1;
        assert!(p.frame_mbs_only);
        p.direct_8x8_inference = r.u(1) == 1;
        let pps = rbsp(pps);
        let mut r = Bits(&pps, 0);
        assert_eq!((r.ue(), r.ue()), (0, 0));
        p.entropy_coding_mode = r.u(1) == 1;
        p.pic_order_present = r.u(1) == 1;
        assert_eq!(r.ue(), 0);
        p.num_ref_idx_l0_active_minus1 = r.ue();
        p.num_ref_idx_l1_active_minus1 = r.ue();
        p.weighted_pred = r.u(1) == 1;
        p.weighted_bipred_idc = r.u(2);
        p.pic_init_qp_minus26 = r.se();
        r.se();
        p.chroma_qp_index_offset = r.se();
        p.second_chroma_qp_index_offset = p.chroma_qp_index_offset;
        p.deblocking_filter_control_present = r.u(1) == 1;
        p.constrained_intra_pred = r.u(1) == 1;
        p.redundant_pic_cnt_present = r.u(1) == 1;
        p.scaling_4x4 = [[16; 16]; 6];
        p.scaling_8x8 = [[16; 64]; 2];
        p
    }

    /// Serializes a [`PictureInfo`] back into the NVDEC picture setup layout.
    pub(crate) fn write_setup(p: &PictureInfo) -> Vec<u8> {
        fn put(b: &mut [u8], off: usize, v: u32) {
            b[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let mut b = vec![0u8; PICTURE_INFO_SIZE];
        put(&mut b, 0x48, p.bitstream_size);
        put(&mut b, 0x4c, p.num_slices);
        put(&mut b, 0x58, p.log2_max_pic_order_cnt_lsb_minus4);
        put(&mut b, 0x5c, p.delta_pic_order_always_zero as u32);
        put(&mut b, 0x60, p.frame_mbs_only as u32);
        put(&mut b, 0x64, p.width_mbs);
        put(&mut b, 0x68, p.height_mbs);
        put(&mut b, 0x70, p.entropy_coding_mode as u32);
        put(&mut b, 0x74, p.pic_order_present as u32);
        put(&mut b, 0x78, p.num_ref_idx_l0_active_minus1);
        put(&mut b, 0x7c, p.num_ref_idx_l1_active_minus1);
        put(&mut b, 0x80, p.deblocking_filter_control_present as u32);
        put(&mut b, 0x84, p.redundant_pic_cnt_present as u32);
        put(&mut b, 0x88, p.transform_8x8_mode as u32);
        put(&mut b, 0x9c, p.luma_frame_offset);
        put(&mut b, 0xa8, p.chroma_frame_offset);
        put(&mut b, 0xb8, p.field_order_cnt[0] as u32);
        put(&mut b, 0xbc, p.field_order_cnt[1] as u32);
        let mut flags = p.mb_adaptive_frame_field as u64;
        flags |= (p.direct_8x8_inference as u64) << 1;
        flags |= (p.weighted_pred as u64) << 2;
        flags |= (p.constrained_intra_pred as u64) << 3;
        flags |= 1 << 4; // IsReference
        flags |= (p.field_pic as u64) << 5;
        flags |= ((p.log2_max_frame_num_minus4 as u64) & 0xf) << 8;
        flags |= ((p.chroma_format_idc as u64) & 3) << 12;
        flags |= ((p.pic_order_cnt_type as u64) & 3) << 14;
        flags |= ((p.pic_init_qp_minus26 as u32 as u64) & 0x3f) << 16;
        flags |= ((p.chroma_qp_index_offset as u32 as u64) & 0x1f) << 22;
        flags |= ((p.second_chroma_qp_index_offset as u32 as u64) & 0x1f) << 27;
        flags |= ((p.weighted_bipred_idc as u64) & 3) << 32;
        flags |= ((p.output_surface as u64) & 0x7f) << 34;
        flags |= ((p.frame_num as u64) & 0xffff) << 46;
        put(&mut b, 0xb0, flags as u32);
        put(&mut b, 0xb4, (flags >> 32) as u32);
        for (i, list) in p.scaling_4x4.iter().enumerate() {
            b[0x1c0 + 16 * i..0x1d0 + 16 * i].copy_from_slice(list);
        }
        for (i, list) in p.scaling_8x8.iter().enumerate() {
            b[0x220 + 64 * i..0x260 + 64 * i].copy_from_slice(list);
        }
        if p.qpprime_y_zero_transform_bypass {
            put(&mut b, 0x2d0, 2);
        }
        b
    }
}

#[cfg(all(test, feature = "h264"))]
mod tests {
    use super::support::setup_from;
    use super::*;

    #[test]
    fn exp_golomb() {
        let mut w = BitWriter::default();
        w.ue(0);
        w.ue(3);
        w.se(-2);
        w.se(1);
        let mut out = Vec::new();
        w.nal(0x67, &mut out);
        assert_eq!(out, [0, 0, 0, 1, 0x67, 0b1001_0000, 0b1010_1010]);
    }

    #[test]
    fn emulation_prevention() {
        let mut w = BitWriter::default();
        w.u(0, 24);
        w.u(1, 8);
        let mut out = Vec::new();
        w.nal(0x68, &mut out);
        assert_eq!(out, [0, 0, 0, 1, 0x68, 0, 0, 3, 0, 1, 0x80]);
    }

    #[test]
    fn flat_scaling_list() {
        let mut w = BitWriter::default();
        scaling_list(&mut w, &[16; 16], &ZIGZAG_4X4);
        let mut out = Vec::new();
        w.nal(0x68, &mut out);
        assert_eq!(&out[5..], [0b0000_1000, 0b0111_1111, 0xff, 0x80]);
    }

    fn encode_clip(w: usize, h: usize, count: usize) -> Vec<Vec<u8>> {
        use openh264::encoder::Encoder;
        use openh264::formats::{RgbSliceU8, YUVBuffer};
        let mut encoder = Encoder::new().unwrap();
        (0..count)
            .map(|t| {
                let rgb: Vec<u8> = (0..w * h)
                    .flat_map(|i| {
                        [
                            (i % w * 2 + t * 9) as u8,
                            (i / w * 3 + t * 5) as u8,
                            (i * 7 + t * 13 % 251) as u8,
                        ]
                    })
                    .collect();
                encoder
                    .encode(&YUVBuffer::from_rgb8_source(RgbSliceU8::new(&rgb, (w, h))))
                    .unwrap()
                    .to_vec()
            })
            .collect()
    }

    fn slices_of(frame: &[u8]) -> Vec<u8> {
        let nal_type = |n: &[u8]| n[n.iter().position(|&b| b == 1).unwrap() + 1] & 0x1f;
        openh264::nal_units(frame)
            .filter(|n| matches!(nal_type(n), 1 | 5))
            .flatten()
            .copied()
            .collect()
    }

    fn parameter_setup(frame: &[u8]) -> PictureInfo {
        let nal_type = |n: &[u8]| n[n.iter().position(|&b| b == 1).unwrap() + 1] & 0x1f;
        let body = |n: &[u8]| n[n.iter().position(|&b| b == 1).unwrap() + 2..].to_vec();
        let nals: Vec<&[u8]> = openh264::nal_units(frame).collect();
        let sps = body(nals.iter().find(|n| nal_type(n) == 7).unwrap());
        let pps = body(nals.iter().find(|n| nal_type(n) == 8).unwrap());
        setup_from(&sps, &pps)
    }

    #[test]
    fn rebuilt_sps_carries_the_crop_level_and_reference_count() {
        let mut info = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        info.chroma_format_idc = 1;
        info.frame_mbs_only = true;
        info.width_mbs = 120;
        info.height_mbs = 68;
        info.sps.crop = Crop {
            bottom: 8,
            ..Crop::NONE
        };
        assert_eq!(info.coded_size(), (1920, 1088));
        assert_eq!(info.display_size(), (1920, 1080));

        let sets = parameter_sets(&info).unwrap();
        let sps = in_band_sps(&sets).expect("the rebuilt parameter sets start with an SPS");
        assert_eq!(sps.crop, info.sps.crop);
        // 120x68 MBs needs level 5.1 to keep 16 references; the rebuilt SPS
        // used to hard-code both.
        assert_eq!(sps.level_idc, 51);
        assert_eq!(sps.max_num_ref_frames, 16);
        assert_eq!((sps.bit_depth_luma, sps.bit_depth_chroma), (8, 8));
        assert_eq!(sps.chroma_format_idc, 1);

        // A 900x506 stream is coded as 912x512 and crops 12 columns and 6 rows.
        let mut small = info.clone();
        small.width_mbs = 57;
        small.height_mbs = 32;
        small.sps.crop = Crop {
            right: 12,
            bottom: 6,
            ..Crop::NONE
        };
        assert_eq!(small.display_size(), (900, 506));
        let sps = in_band_sps(&parameter_sets(&small).unwrap()).expect("SPS");
        assert_eq!(sps.crop, small.sps.crop);
        assert_eq!(sps.level_idc, 40);
        assert_eq!(reference_frames(sps.level_idc, 57 * 32).unwrap(), 16);
    }

    #[test]
    fn the_reference_pool_follows_the_level() {
        // 1080p: only level 5.1 has a DPB large enough for 16 references.
        assert_eq!(level_for(120 * 68), 51);
        assert_eq!(reference_frames(51, 120 * 68).unwrap(), 16);
        // Level 4.0 holds four 1080p pictures.
        assert_eq!(reference_frames(40, 120 * 68).unwrap(), 4);
        // A 4K picture cannot fit 16 references in any OpenH264 level.
        assert_eq!(level_for(240 * 136), 52);
        assert_eq!(reference_frames(52, 240 * 136).unwrap(), 5);
        // Levels outside the table and pictures larger than MaxFS are refused.
        assert!(reference_frames(99, 8160).is_err());
        assert!(reference_frames(51, 40000).is_err());

        let mut info = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        info.chroma_format_idc = 1;
        info.frame_mbs_only = true;
        info.width_mbs = 120;
        info.height_mbs = 68;
        info.sps.max_num_ref_frames = MAX_REFERENCE_FRAMES + 1;
        let error = parameter_sets(&info).unwrap_err();
        assert!(error.contains("reference frames"), "{error}");
    }

    #[test]
    fn an_invalid_crop_is_refused() {
        let mut info = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        info.chroma_format_idc = 1;
        info.frame_mbs_only = true;
        info.width_mbs = 120;
        info.height_mbs = 68;
        info.sps.crop = Crop {
            bottom: 1088,
            ..Crop::NONE
        };
        let error = parameter_sets(&info).unwrap_err();
        assert!(error.contains("does not fit"), "{error}");
        // 4:2:0 crops in luma-sample multiples of two.
        info.sps.crop = Crop {
            bottom: 3,
            ..Crop::NONE
        };
        let error = parameter_sets(&info).unwrap_err();
        assert!(error.contains("crop unit"), "{error}");
    }

    #[test]
    fn finds_a_parameter_set_between_slices() {
        let mut info = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        info.chroma_format_idc = 1;
        info.frame_mbs_only = true;
        info.width_mbs = 120;
        info.height_mbs = 68;
        info.sps.crop = Crop {
            bottom: 4,
            ..Crop::NONE
        };
        let sets = parameter_sets(&info).unwrap();
        let mut access_unit = vec![0, 0, 0, 1, 0x41, 0x9a, 0x80];
        access_unit.extend_from_slice(&sets);
        access_unit.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9a, 0x80]);
        assert_eq!(in_band_sps(&access_unit).unwrap().crop, info.sps.crop);
        assert!(in_band_sps(&[0, 0, 0, 1, 0x41, 0x9a, 0x80]).is_none());
    }

    #[test]
    fn the_sps_crop_shapes_the_decoded_picture() {
        // The encoder codes 96x80 exactly; the SPS then crops the last 16 rows,
        // as a 1080p stream crops the 1088-row coded picture down to 1080.
        let (w, h) = (96usize, 80usize);
        let frames = encode_clip(w, h, 2);
        let mut setup = parameter_setup(&frames[0]);
        assert_eq!(setup.height_mbs, 5);
        setup.sps.crop = Crop {
            bottom: 16,
            ..Crop::NONE
        };
        assert_eq!(setup.display_size(), (96, 64));

        let mut reference = OpenH264::new().unwrap();
        let want = reference.decode(&frames[0]).unwrap().expect("reference");
        assert_eq!(want.dimensions(), (96, 80));

        let mut ours = Decoder::new().unwrap();
        assert!(
            ours.decode(&setup, &slices_of(&frames[0]))
                .unwrap()
                .is_none()
        );
        let got = ours
            .decode(&setup, &slices_of(&frames[1]))
            .unwrap()
            .expect("picture");
        assert_eq!((got.width, got.height), (96, 64));
        let (sy, su, sv) = want.strides();
        let mut luma = Vec::new();
        for y in 0..64 {
            luma.extend(want.y()[y * sy..y * sy + w].iter().map(|&v| v as u16));
        }
        assert_eq!(got.y, luma);
        let mut uv = Vec::new();
        for y in 0..32 {
            let (u, v) = (
                &want.u()[y * su..y * su + w / 2],
                &want.v()[y * sv..y * sv + w / 2],
            );
            for x in 0..w / 2 {
                uv.extend_from_slice(&[u[x] as u16, v[x] as u16]);
            }
        }
        assert_eq!(got.uv, uv);
    }

    #[test]
    fn flushing_the_stream_drains_the_last_picture() {
        let frames = encode_clip(96, 64, 3);
        let setup = parameter_setup(&frames[0]);
        let mut decoder = Decoder::new().unwrap();
        let mut pictures = 0;
        for frame in &frames {
            if decoder.decode(&setup, &slices_of(frame)).unwrap().is_some() {
                pictures += 1;
            }
        }
        assert_eq!(pictures, frames.len() - 1);

        let drained = decoder.flush().unwrap();
        assert_eq!(drained.len(), 1, "the last picture was still owed");
        assert_eq!((drained[0].width, drained[0].height), (96, 64));
        assert!(decoder.flush().unwrap().is_empty());

        // The drained picture is the last one of the stream, not a repeat.
        let mut reference = OpenH264::new().unwrap();
        let mut last = None;
        for frame in &frames {
            last = reference.decode(frame).unwrap();
        }
        let last = last.expect("last picture");
        let stride = last.strides().0;
        let want: Vec<u16> = (0..64)
            .flat_map(|y| {
                last.y()[y * stride..y * stride + 96]
                    .iter()
                    .map(|&v| v as u16)
            })
            .collect();
        assert_eq!(drained[0].y, want);

        // A stream that starts again with an IDR keeps decoding after the drain.
        let more = encode_clip(96, 64, 2);
        let mut resumed = 0;
        for frame in &more {
            if decoder
                .decode(&setup, &slices_of(frame))
                .unwrap_or_else(|e| panic!("after flush: {e}"))
                .is_some()
            {
                resumed += 1;
            }
        }
        assert_eq!(resumed, more.len() - 1);
    }

    #[test]
    fn ten_bit_streams_are_rejected_with_the_missing_backend() {
        let mut decoder = Decoder::new().unwrap();
        let mut info = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        info.sps.bit_depth_luma = 10;
        info.sps.bit_depth_chroma = 10;
        let error = decoder.decode(&info, &[]).unwrap_err();
        assert!(error.contains("10-bit"), "{error}");
        assert!(error.contains("OpenH264"), "{error}");
        assert!(error.contains("FFmpeg"), "{error}");
        // 4:2:2 is rejected from the setup or from an in-band SPS.
        let mut info = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        info.sps.chroma_format_idc = 2;
        let error = decoder.decode(&info, &[]).unwrap_err();
        assert!(error.contains("4:2:0 only"), "{error}");
    }

    #[test]
    fn rebuilt_parameter_sets_decode_like_the_originals() {
        let (w, h) = (96usize, 64usize);
        let frames = encode_clip(w, h, 4);
        let setup = parameter_setup(&frames[0]);
        // The reference is flushed per frame so each call returns the picture it
        // decoded; ours is not flushed, so the picture arrives one submission
        // later but is still the same picture.
        let mut reference = OpenH264::new().unwrap();
        let mut want = Vec::new();
        for frame in &frames {
            let yuv = reference.decode(frame).unwrap().expect("reference picture");
            let stride = yuv.strides().0;
            let rows: Vec<u16> = (0..h)
                .flat_map(|y| {
                    yuv.y()[y * stride..y * stride + w]
                        .iter()
                        .map(|&v| v as u16)
                })
                .collect();
            want.push((yuv.dimensions(), rows));
        }
        assert_eq!(want.len(), frames.len());
        let mut ours = Decoder::new().unwrap();
        let mut got = Vec::new();
        for frame in &frames {
            if let Some(picture) = ours.decode(&setup, &slices_of(frame)).unwrap() {
                got.push(picture);
            }
        }
        assert_eq!(got.len(), frames.len() - 1, "one submission of latency");
        for (i, (got, (dims, rows))) in got.iter().zip(&want).enumerate() {
            assert_eq!((got.width, got.height), *dims, "picture {i}");
            assert_eq!(got.bit_depth, BitDepth::Eight);
            assert_eq!(&got.y[..], &rows[..], "picture {i}");
        }
    }

    /// A flush after every picture makes OpenH264 keep each output marked as a
    /// DPB reference. The picture buffer (`max_num_ref_frames + 2`) then runs
    /// out after a few GOPs and every later decode fails with dsOutOfMemory
    /// (16384) and then dsNoParamSets (16). Decoding a stream longer than the
    /// buffer has to keep going.
    #[test]
    fn decodes_more_pictures_than_the_reference_buffer_holds() {
        let frames = encode_clip(96, 64, 64);
        assert!(frames.len() > 2 * (16 + 2), "needs more than one refill");
        let setup = parameter_setup(&frames[0]);
        let mut decoder = Decoder::new().unwrap();
        let mut pictures = 0;
        for (i, frame) in frames.iter().enumerate() {
            let decoded = decoder
                .decode(&setup, &slices_of(frame))
                .unwrap_or_else(|e| panic!("frame {i}: {e}"));
            if decoded.is_some() {
                pictures += 1;
            }
        }
        assert_eq!(pictures, frames.len() - 1);
    }

    #[test]
    fn first_submission_only_primes_the_decoder() {
        let frames = encode_clip(96, 64, 2);
        let setup = parameter_setup(&frames[0]);
        let mut decoder = Decoder::new().unwrap();
        assert!(
            decoder
                .decode(&setup, &slices_of(&frames[0]))
                .unwrap()
                .is_none()
        );
        let picture = decoder
            .decode(&setup, &slices_of(&frames[1]))
            .unwrap()
            .expect("the first picture comes out with the second submission");
        assert_eq!((picture.width, picture.height), (96, 64));
        // An empty submission asks for nothing and is not an error.
        assert!(decoder.decode(&setup, &[]).unwrap().is_none());
    }

    #[test]
    fn rejects_non_420_chroma() {
        let mut decoder = Decoder::new().unwrap();
        let mut info = PictureInfo::parse(&[0; PICTURE_INFO_SIZE]);
        info.chroma_format_idc = 2;
        let error = decoder.decode(&info, &[]).unwrap_err();
        assert!(error.contains("4:2:0 only"), "{error}");
        // 4:4:4 is rejected before the decoder looks at the (empty) bitstream.
        info.chroma_format_idc = 3;
        assert!(decoder.decode(&info, &[]).is_err());
    }
}
