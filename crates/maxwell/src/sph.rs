pub const SIZE: usize = 0x50;

#[derive(Clone, Debug)]
pub struct Header {
    words: [u32; 20],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Unused,
    Constant,
    Perspective,
    ScreenLinear,
}

impl Header {
    pub fn parse(bytes: &[u8]) -> Option<Header> {
        let b = bytes.get(..SIZE)?;
        let words = std::array::from_fn(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()));
        Some(Header { words })
    }

    fn bits(&self, lo: usize, n: usize) -> u32 {
        let mut v = 0u64;
        for k in 0..n {
            let b = lo + k;
            v |= ((self.words[b / 32] >> (b % 32) & 1) as u64) << k;
        }
        v as u32
    }

    pub fn sph_type(&self) -> u32 {
        self.bits(0, 5)
    }

    pub fn shader_type(&self) -> u32 {
        self.bits(10, 4)
    }

    pub fn kills_pixels(&self) -> bool {
        self.bits(15, 1) == 1
    }

    pub fn local_memory_size(&self) -> u32 {
        self.bits(32, 24) + self.bits(64, 24)
    }

    pub fn vtg_input(&self, i: usize, c: usize) -> bool {
        self.bits(192 + i * 4 + c, 1) == 1
    }

    pub fn vtg_output(&self, i: usize, c: usize) -> bool {
        self.bits(432 + i * 4 + c, 1) == 1
    }

    pub fn ps_input(&self, i: usize, c: usize) -> Interpolation {
        match self.bits(192 + i * 8 + c * 2, 2) {
            0 => Interpolation::Unused,
            1 => Interpolation::Constant,
            2 => Interpolation::Perspective,
            _ => Interpolation::ScreenLinear,
        }
    }

    pub fn ps_output(&self, t: usize, c: usize) -> bool {
        self.bits(576 + t * 4 + c, 1) == 1
    }

    pub fn ps_writes_depth(&self) -> bool {
        self.bits(609, 1) == 1
    }

    pub fn ps_writes_sample_mask(&self) -> bool {
        self.bits(608, 1) == 1
    }
}
