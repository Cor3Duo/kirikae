#[cfg(feature = "h264")]
use openh264::OpenH264API;
#[cfg(feature = "h264")]
use openh264::decoder::{DecodeOptions, Decoder as OpenH264, DecoderConfig, Flush};
#[cfg(feature = "h264")]
use openh264::formats::YUVSource;

#[cfg(feature = "h264")]
use super::surface::BitDepth;
use super::surface::Yuv420;
use super::u32_at;

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
    pub scaling_4x4: [[u8; 16]; 6],
    pub scaling_8x8: [[u8; 64]; 2],
    pub qpprime_y_zero_transform_bypass: bool,
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
            scaling_4x4,
            scaling_8x8,
            qpprime_y_zero_transform_bypass: u32_at(b, 0x2d0) & 2 != 0,
        }
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

pub fn parameter_sets(p: &PictureInfo) -> Vec<u8> {
    let mut out = Vec::new();
    let mut w = BitWriter::default();
    w.u(100, 8);
    w.u(0, 8);
    w.u(51, 8);
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
    w.ue(16);
    w.flag(false);
    w.ue(p.width_mbs.max(1) - 1);
    w.ue((p.height_mbs >> !p.frame_mbs_only as u32).max(1) - 1);
    w.flag(p.frame_mbs_only);
    if !p.frame_mbs_only {
        w.flag(p.mb_adaptive_frame_field);
    }
    w.flag(p.direct_8x8_inference);
    w.flag(false);
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
    out
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
}

#[cfg(feature = "h264")]
pub struct Decoder {
    inner: OpenH264,
    headers: Vec<u8>,
}

#[cfg(feature = "h264")]
impl Decoder {
    pub fn new() -> Result<Decoder, String> {
        let config = DecoderConfig::new().flush_after_decode(Flush::Flush);
        let inner = OpenH264::with_api_config(OpenH264API::from_source(), config)
            .map_err(|e| format!("OpenH264: {e}"))?;
        Ok(Decoder {
            inner,
            headers: Vec::new(),
        })
    }

    pub fn decode(&mut self, info: &PictureInfo, slices: &[u8]) -> Result<Option<Yuv420>, String> {
        // The picture setup describes the chroma format the slices were encoded
        // with. Only 4:2:0 (idc 1) has an NV12/P010 surface; anything else would
        // be written into a surface of the wrong shape. Idc 0 is what the setup
        // block reads as before it is filled in, so it is treated as 4:2:0 too.
        if info.chroma_format_idc > 1 {
            return Err(format!(
                "H.264 chroma format {} is not supported (4:2:0 only)",
                info.chroma_format_idc
            ));
        }
        let headers = parameter_sets(info);
        let mut packet = Vec::with_capacity(headers.len() + slices.len());
        if headers != self.headers {
            packet.extend_from_slice(&headers);
            self.headers = headers;
        }
        packet.extend_from_slice(slices);
        let options = DecodeOptions::new().flush_after_decode(Flush::Flush);
        let yuv = match self.inner.decode_with_options(&packet, options) {
            Ok(Some(yuv)) => yuv,
            Ok(None) => return Ok(None),
            Err(e) => return Err(format!("H.264 picture not decoded: {e}")),
        };
        let (width, height) = yuv.dimensions();
        let (sy, su, sv) = yuv.strides();
        let mut y = Vec::with_capacity(width * height);
        for row in yuv.y().chunks(sy).take(height) {
            y.extend(row[..width].iter().map(|&v| v as u16));
        }
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let mut uv = vec![0u16; width.div_ceil(2) * 2 * ch];
        for r in 0..ch {
            let (u, v) = (&yuv.u()[r * su..r * su + cw], &yuv.v()[r * sv..r * sv + cw]);
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
        Ok(Some(Yuv420 {
            width,
            height,
            // OpenH264 only decodes 8-bit profiles; a 10-bit stream (High 10)
            // is rejected by the decoder above rather than decoded as garbage.
            bit_depth: BitDepth::Eight,
            y,
            uv,
        }))
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
mod tests {
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

    struct Bits<'a>(&'a [u8], usize);

    impl Bits<'_> {
        fn u(&mut self, n: usize) -> u32 {
            (0..n).fold(0, |v, _| {
                let b = (self.0[self.1 / 8] >> (7 - self.1 % 8)) & 1;
                self.1 += 1;
                v << 1 | b as u32
            })
        }
        fn ue(&mut self) -> u32 {
            let mut zeros = 0;
            while self.u(1) == 0 {
                zeros += 1;
            }
            (1 << zeros) - 1 + self.u(zeros)
        }
        fn se(&mut self) -> i32 {
            let v = self.ue() as i32;
            if v & 1 == 1 { (v + 1) / 2 } else { -v / 2 }
        }
    }

    fn rbsp(nal: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for &b in nal {
            if b == 3 && out.ends_with(&[0, 0]) {
                continue;
            }
            out.push(b);
        }
        out
    }

    fn setup_from(sps: &[u8], pps: &[u8]) -> PictureInfo {
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

    #[test]
    fn rebuilt_parameter_sets_decode_like_the_originals() {
        use openh264::encoder::Encoder;
        use openh264::formats::{RgbSliceU8, YUVBuffer};
        let (w, h) = (96usize, 64usize);
        let mut encoder = Encoder::new().unwrap();
        let frames: Vec<Vec<u8>> = (0..4)
            .map(|t| {
                let rgb: Vec<u8> = (0..w * h)
                    .flat_map(|i| {
                        [
                            (i % w * 2 + t * 9) as u8,
                            (i / w * 3) as u8,
                            (i * 7 % 251) as u8,
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
        let first: Vec<&[u8]> = openh264::nal_units(&frames[0]).collect();
        let sps = body(first.iter().find(|n| nal_type(n) == 7).unwrap());
        let pps = body(first.iter().find(|n| nal_type(n) == 8).unwrap());
        let setup = setup_from(&sps, &pps);
        let mut reference = OpenH264::new().unwrap();
        let mut ours = Decoder::new().unwrap();
        for frame in &frames {
            let slices: Vec<u8> = openh264::nal_units(frame)
                .filter(|n| matches!(nal_type(n), 1 | 5))
                .flatten()
                .copied()
                .collect();
            let want = reference.decode(frame).unwrap().expect("reference picture");
            let got = ours.decode(&setup, &slices).unwrap().expect("picture");
            assert_eq!((got.width, got.height), want.dimensions());
            assert_eq!(got.bit_depth, BitDepth::Eight);
            let stride = want.strides().0;
            for y in 0..h {
                let row: Vec<u16> = want.y()[y * stride..y * stride + w]
                    .iter()
                    .map(|&v| v as u16)
                    .collect();
                assert_eq!(&got.y[y * w..(y + 1) * w], &row[..], "row {y}");
            }
        }
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
