fn bits(words: &[u32; 8], lo: usize, n: usize) -> u32 {
    let mut v = 0u64;
    for k in 0..n {
        let b = lo + k;
        v |= ((words[b / 32] >> (b % 32) & 1) as u64) << k;
    }
    v as u32
}

pub fn words(bytes: &[u8]) -> Option<[u32; 8]> {
    let b = bytes.get(..32)?;
    Some(std::array::from_fn(|i| {
        u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    BlockLinear {
        block_height_log2: u32,
        block_depth_log2: u32,
    },
    Pitch {
        pitch: u32,
    },
    Buffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHeader {
    pub components: u32,
    pub data_type: u32,
    pub data_types: [u32; 4],
    pub swizzle: [u32; 4],
    pub address: u64,
    pub layout: Layout,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub texture_type: u32,
    pub levels: u32,
    pub min_level: u32,
    pub max_level: u32,
    pub srgb: bool,
}

pub fn texture_header(w: &[u32; 8]) -> TextureHeader {
    let version = bits(w, 85, 3);
    let layout = match version {
        3 | 4 => Layout::BlockLinear {
            block_height_log2: bits(w, 99, 3),
            block_depth_log2: bits(w, 102, 3),
        },
        1 | 2 => Layout::Pitch {
            pitch: bits(w, 96, 16) << 5,
        },
        _ => Layout::Buffer,
    };
    let address = match layout {
        Layout::BlockLinear { .. } => {
            (bits(w, 41, 23) as u64) << 9 | (bits(w, 64, 16) as u64) << 32
        }
        _ => (bits(w, 37, 27) as u64) << 5 | (bits(w, 64, 16) as u64) << 32,
    };
    TextureHeader {
        components: bits(w, 0, 7),
        data_type: bits(w, 7, 3),
        data_types: [
            bits(w, 7, 3),
            bits(w, 10, 3),
            bits(w, 13, 3),
            bits(w, 16, 3),
        ],
        swizzle: [
            bits(w, 19, 3),
            bits(w, 22, 3),
            bits(w, 25, 3),
            bits(w, 28, 3),
        ],
        address,
        layout,
        width: bits(w, 128, 16) + 1,
        height: bits(w, 160, 16) + 1,
        depth: bits(w, 176, 14) + 1,
        texture_type: bits(w, 151, 4),
        levels: bits(w, 124, 4) + 1,
        min_level: bits(w, 224, 4),
        max_level: bits(w, 228, 4),
        srgb: bits(w, 150, 1) == 1,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Sampler {
    pub address: [u32; 3],
    pub compare: Option<u32>,
    pub mag_linear: bool,
    pub min_linear: bool,
    pub mip: u32,
    pub max_anisotropy: u32,
    pub border: [u32; 4],
    pub min_lod: u32,
    pub max_lod: u32,
    pub lod_bias: i32,
    pub reduction: u32,
}

pub fn sampler(w: &[u32; 8]) -> Sampler {
    Sampler {
        address: [bits(w, 0, 3), bits(w, 3, 3), bits(w, 6, 3)],
        compare: (bits(w, 9, 1) == 1).then(|| bits(w, 10, 3)),
        mag_linear: bits(w, 32, 3) == 2,
        min_linear: bits(w, 36, 2) >= 2,
        mip: bits(w, 38, 2),
        max_anisotropy: 1 << bits(w, 20, 3),
        border: [w[4], w[5], w[6], w[7]],
        min_lod: bits(w, 96, 12),
        max_lod: bits(w, 108, 12),
        lod_bias: ((bits(w, 44, 13) << 19) as i32) >> 19,
        reduction: bits(w, 42, 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_linear_header() {
        let mut w = [0u32; 8];
        w[0] = 0x08 | 2 << 7 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28;
        w[1] = 0x1234 << 9;
        w[2] = 0x56 | 3 << 21;
        w[3] = 4 << 3;
        w[4] = 1919 | 1 << 23;
        w[5] = 1079;
        let h = texture_header(&w);
        assert_eq!((h.components, h.data_type, h.swizzle), (8, 2, [2, 3, 4, 5]));
        assert_eq!(h.address, 0x56_0000_0000 | 0x1234 << 9);
        assert_eq!(
            h.layout,
            Layout::BlockLinear {
                block_height_log2: 4,
                block_depth_log2: 0
            }
        );
        assert_eq!((h.width, h.height, h.texture_type), (1920, 1080, 1));
    }
}
