//! Sample depth and 4:2:0 guest surfaces, shared by NVDEC and VIC.
//!
//! NVDEC writes decoded pictures into guest surfaces whose layout follows the
//! picture's sample depth: 8-bit pictures go out as NV12 (`Y8_V8U8_N420`, one
//! byte per sample) and 10-bit pictures as P010 (two bytes per sample, the
//! 10-bit value in the high bits of a little-endian 16-bit word). VIC reads and
//! writes the same two layouts, so both engines share the conversion policy
//! described on [`BitDepth::rescale`].

/// Sample depth of a decoded picture or of a video surface.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BitDepth {
    /// 8 bits per sample.
    #[default]
    Eight,
    /// 10 bits per sample, carried in the high bits of a 16-bit word (P010).
    Ten,
}

impl BitDepth {
    /// The depth of `bits`-bit samples, or `None` for depths no surface uses.
    pub fn from_bits(bits: u8) -> Option<BitDepth> {
        match bits {
            8 => Some(BitDepth::Eight),
            10 => Some(BitDepth::Ten),
            _ => None,
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            BitDepth::Eight => 8,
            BitDepth::Ten => 10,
        }
    }

    /// Bytes one sample occupies in a surface.
    pub fn bytes(self) -> usize {
        match self {
            BitDepth::Eight => 1,
            BitDepth::Ten => 2,
        }
    }

    /// Rescales `value` from this depth to `target`.
    ///
    /// Gaining bits shifts left, so 8-bit code 235 becomes 10-bit 940 and the
    /// mid-point stays the mid-point. Dropping bits rounds to nearest
    /// (`(v + 2) >> 2` for 10 bits to 8) instead of truncating, and the result
    /// saturates: 10-bit 1023 maps to 8-bit 255, not 256.
    pub fn rescale(self, value: u16, target: BitDepth) -> u16 {
        let scaled = match (self, target) {
            (BitDepth::Eight, BitDepth::Ten) => value << 2,
            (BitDepth::Ten, BitDepth::Eight) => (value + 2) >> 2,
            _ => value,
        };
        scaled.min((1u16 << target.bits()) - 1)
    }
}

/// A decoded 4:2:0 picture: one luma sample per pixel and interleaved U/V pairs
/// at half resolution, both expressed at `bit_depth`.
///
/// Samples are kept widened to `u16` at every depth so decoders, the packers and
/// the reference pictures all share one code path; narrowing happens only when
/// the picture is packed for a guest surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420 {
    pub width: usize,
    pub height: usize,
    pub bit_depth: BitDepth,
    /// One sample per luma position, `width * height` entries.
    pub y: Vec<u16>,
    /// Interleaved U and V, two samples per chroma position.
    pub uv: Vec<u16>,
}

impl Yuv420 {
    /// Wraps an 8-bit NV12 picture.
    pub fn from_nv12(y: &[u8], uv: &[u8], width: usize, height: usize) -> Yuv420 {
        Yuv420 {
            width,
            height,
            bit_depth: BitDepth::Eight,
            y: y.iter().map(|&v| v as u16).collect(),
            uv: uv.iter().map(|&v| v as u16).collect(),
        }
    }

    /// Chroma plane size in samples.
    pub fn chroma(&self) -> (usize, usize) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }

    /// Packs the picture for a guest surface at `depth`, converting the samples
    /// according to [`BitDepth::rescale`].
    ///
    /// The luma and interleaved chroma bytes are returned separately; each is
    /// laid out for `block_linear_size`/`to_block_linear` with
    /// `width * depth.bytes()` bytes per row.
    pub fn pack(&self, depth: BitDepth) -> (Vec<u8>, Vec<u8>) {
        let convert = |samples: &[u16]| -> Vec<u8> {
            match depth {
                BitDepth::Eight => samples
                    .iter()
                    .map(|&v| self.bit_depth.rescale(v, depth) as u8)
                    .collect(),
                BitDepth::Ten => samples
                    .iter()
                    .flat_map(|&v| (self.bit_depth.rescale(v, depth) << 6).to_le_bytes())
                    .collect(),
            }
        };
        (convert(&self.y), convert(&self.uv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescale_rounds_when_dropping_bits() {
        assert_eq!(BitDepth::Ten.rescale(1023, BitDepth::Eight), 255);
        assert_eq!(BitDepth::Ten.rescale(940, BitDepth::Eight), 235);
        assert_eq!(BitDepth::Ten.rescale(64, BitDepth::Eight), 16);
        // Truncation would give 0 here; rounding to nearest gives 1.
        assert_eq!(BitDepth::Ten.rescale(2, BitDepth::Eight), 1);
        assert_eq!(BitDepth::Ten.rescale(1, BitDepth::Eight), 0);
        assert_eq!(BitDepth::Ten.rescale(0, BitDepth::Eight), 0);
        assert_eq!(BitDepth::Eight.rescale(255, BitDepth::Ten), 1020);
        assert_eq!(BitDepth::Eight.rescale(235, BitDepth::Ten), 940);
        assert_eq!(BitDepth::Eight.rescale(7, BitDepth::Eight), 7);
        // 10-bit values above the 8-bit range that would wrap if unmasked.
        assert_eq!(BitDepth::Ten.rescale(1023, BitDepth::Ten), 1023);
    }

    #[test]
    fn packs_nv12_and_p010() {
        let y = [16u16, 235, 81, 145];
        let uv = [128u16, 128, 90, 240];
        let picture = Yuv420 {
            width: 2,
            height: 2,
            bit_depth: BitDepth::Eight,
            y: y.to_vec(),
            uv: uv.to_vec(),
        };
        assert_eq!(
            picture.pack(BitDepth::Eight),
            (y.map(|v| v as u8).to_vec(), uv.map(|v| v as u8).to_vec())
        );

        let (p010_y, p010_uv) = picture.pack(BitDepth::Ten);
        assert_eq!(p010_y.len(), 8);
        // 10-bit samples live in the high bits of the little-endian word.
        assert_eq!(
            p010_y
                .as_chunks::<2>()
                .0
                .iter()
                .map(|w| u16::from_le_bytes(*w) >> 6)
                .collect::<Vec<_>>(),
            [64u16, 940, 324, 580]
        );
        assert_eq!(
            p010_uv
                .as_chunks::<2>()
                .0
                .iter()
                .map(|w| u16::from_le_bytes(*w) >> 6)
                .collect::<Vec<_>>(),
            [512u16, 512, 360, 960]
        );
    }

    #[test]
    fn ten_bit_pictures_lose_two_bits_in_nv12() {
        let picture = Yuv420 {
            width: 2,
            height: 2,
            bit_depth: BitDepth::Ten,
            y: vec![1023, 940, 2, 0],
            uv: vec![512, 512, 512, 512],
        };
        let (y, uv) = picture.pack(BitDepth::Eight);
        assert_eq!(y, [255, 235, 1, 0]);
        assert_eq!(uv, [128, 128, 128, 128]);
        let (y, _) = picture.pack(BitDepth::Ten);
        assert_eq!(
            y.as_chunks::<2>()
                .0
                .iter()
                .map(|w| u16::from_le_bytes(*w) >> 6)
                .collect::<Vec<_>>(),
            [1023, 940, 2, 0]
        );
    }
}
