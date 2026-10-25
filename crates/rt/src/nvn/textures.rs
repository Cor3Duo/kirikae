//! NVN textures and texture views, CPU side.
//!
//! The builder is the SDK driver's struct; the host object keeps the values decoded from it (target, size, format,
//! levels, samples, stride, storage) plus the layout the real driver would compute: storage size, alignment and the
//! level-0 block height in GOBs. No image is created here (that is milestone 4.5).
//!
//! Block-linear layout (`nvn_texture_layout.cpp:42-82`, `docs/nvn/TILING.md`): a GOB is 64 bytes by 8 rows, a level
//! is a whole number of GOB-tall blocks, levels are back to back and a layer is a mip chain. Packaged (BNTX)
//! textures carry their own block height; the builder's packaged-layout value is that exponent when it is 1..=5.

use super::error::Error;
use super::formats;
use super::pools::Pool;

/// Per-channel swizzle (0 zero, 1 one, 2 R, 3 G, 4 B, 5 A) in R, G, B, A order.
pub const DEFAULT_SWIZZLE: [u8; 4] = [2, 3, 4, 5];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureBuilder {
    pub flags: u32,
    pub target: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub levels: u32,
    pub format: u32,
    pub samples: u32,
    pub stride: u64,
    pub pool: u64,
    pub offset: u64,
    /// `nvnTextureBuilderSetPackagedTextureLayout`: 0, or the block height exponent of a BNTX layout.
    pub packaged: u64,
    pub swizzle: [u8; 4],
}

impl Default for TextureBuilder {
    fn default() -> Self {
        TextureBuilder {
            flags: 0,
            target: 1,
            width: 1,
            height: 1,
            depth: 1,
            levels: 1,
            format: 0,
            samples: 1,
            stride: 0,
            pool: 0,
            offset: 0,
            packaged: 0,
            swizzle: DEFAULT_SWIZZLE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub size: u64,
    pub alignment: u64,
    /// Level 0 block height, log2 in GOBs (512-byte rows of 8). At most 4 (16 GOBs) unless packaged.
    pub block_height_log2: u32,
}

/// Block height in GOBs for `rows` block rows: the smallest power of two >= rows / 8, at most 16.
fn block_height_log2(rows: u32) -> u32 {
    let gobs = rows.div_ceil(8).max(1);
    let mut log2 = 0;
    while (1u32 << log2) < gobs && log2 < 4 {
        log2 += 1;
    }
    log2
}

impl TextureBuilder {
    /// Rows of texels the texture has, ignoring the layer count a 1D array keeps in `height`.
    fn rows(&self) -> u32 {
        if matches!(self.target, 0 | 3) {
            1
        } else {
            self.height.max(1)
        }
    }

    /// Array/cubemap layers, from the target rules the driver uses.
    pub fn layers(&self) -> u32 {
        match self.target {
            3 => self.height.max(1),
            4 | 6 | 9 => self.depth.max(1),
            8 => 6,
            _ => 1,
        }
    }

    /// Slices for a 3D texture, 1 otherwise.
    fn slices(&self) -> u32 {
        if self.target == 2 {
            self.depth.max(1)
        } else {
            1
        }
    }

    /// Storage size and alignment of the texture the builder describes.
    pub fn layout(&self) -> Result<Layout, Error> {
        let (bytes, block_w, block_h) = formats::block(self.format);
        if bytes == 0 {
            return Err(Error::UnknownFormat);
        }
        let height = self.rows();
        let layers = u64::from(self.layers());
        if self.stride != 0 {
            let rows = u64::from(height.div_ceil(block_h));
            let size = self
                .stride
                .checked_mul(rows)
                .and_then(|size| size.checked_mul(layers))
                .ok_or(Error::Overflow)?;
            return Ok(Layout {
                size,
                alignment: 0x100,
                block_height_log2: 0,
            });
        }

        let levels = self.levels.max(1);
        if levels > 32 {
            return Err(Error::OutOfRange);
        }
        let width = self.width.max(1);
        let mut base_block_height = block_height_log2(height.div_ceil(block_h));
        if (1..=5).contains(&self.packaged) {
            base_block_height = (self.packaged as u32).min(4);
        }
        let mut layer = 0u64;
        for level in 0..levels {
            let width = width.checked_shr(level).unwrap_or(0).max(1);
            let height = height.checked_shr(level).unwrap_or(0).max(1);
            let columns = u64::from(width.div_ceil(block_w));
            let rows = height.div_ceil(block_h);
            let block_height = base_block_height.min(block_height_log2(rows));
            let pitch = columns
                .checked_mul(u64::from(bytes))
                .ok_or(Error::Overflow)?
                .div_ceil(64)
                .checked_mul(64)
                .ok_or(Error::Overflow)?;
            let gobs = 8u64 << block_height;
            let tall = u64::from(rows)
                .div_ceil(gobs)
                .checked_mul(gobs)
                .ok_or(Error::Overflow)?;
            layer = layer
                .checked_add(pitch.checked_mul(tall).ok_or(Error::Overflow)?)
                .ok_or(Error::Overflow)?;
        }
        let alignment = 512u64 << base_block_height;
        let layer = layer
            .div_ceil(alignment)
            .checked_mul(alignment)
            .ok_or(Error::Overflow)?;
        let size = layer
            .checked_mul(u64::from(self.slices()))
            .and_then(|size| size.checked_mul(layers))
            .and_then(|size| size.checked_mul(u64::from(self.samples.max(1))))
            .ok_or(Error::Overflow)?;
        Ok(Layout {
            size,
            alignment,
            block_height_log2: base_block_height,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Texture {
    pub host: u64,
    pub gpu: u64,
    pub pool: u64,
    pub offset: u64,
    pub flags: u32,
    pub target: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layers: u32,
    pub levels: u32,
    pub samples: u32,
    pub stride: u64,
    pub packaged: u64,
    pub swizzle: [u8; 4],
    pub storage_size: u64,
    pub alignment: u64,
    pub block_height_log2: u32,
}

impl Texture {
    pub fn create(
        pool_address: u64,
        pool: &Pool,
        builder: &TextureBuilder,
        host: u64,
    ) -> Result<Texture, Error> {
        let layout = builder.layout()?;
        pool.contains(builder.offset, layout.size)?;
        if host == 0 {
            return Err(Error::Unmapped);
        }
        let gpu = pool
            .gpu
            .checked_add(builder.offset)
            .ok_or(Error::Overflow)?;
        Ok(Texture {
            host,
            gpu,
            pool: pool_address,
            offset: builder.offset,
            flags: builder.flags,
            target: builder.target,
            format: builder.format,
            width: builder.width,
            height: builder.height,
            depth: builder.depth,
            layers: builder.layers(),
            levels: builder.levels.max(1),
            samples: builder.samples.max(1),
            stride: builder.stride,
            packaged: builder.packaged,
            swizzle: builder.swizzle,
            storage_size: layout.size,
            alignment: layout.alignment,
            block_height_log2: layout.block_height_log2,
        })
    }
}

/// What an `NVNtextureView` selects. 0 levels/layers means "all of the texture's"; format 0 the texture's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureView {
    pub base_level: u32,
    pub levels: u32,
    pub min_layer: u32,
    pub layers: u32,
    pub format: u32,
    pub target: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    const STORAGE: u64 = 0x8_0000_0000;

    fn builder(format: u32, width: u32, height: u32, levels: u32) -> TextureBuilder {
        TextureBuilder {
            format,
            width,
            height,
            levels,
            ..TextureBuilder::default()
        }
    }

    #[test]
    fn storage_sizes_follow_block_linear_layout() {
        let rgba8 = builder(0x25, 256, 256, 1).layout().expect("RGBA8 256");
        assert_eq!(rgba8.size, 0x40000);
        assert_eq!(rgba8.alignment, 0x2000);
        assert_eq!(rgba8.block_height_log2, 4);

        let bc1 = builder(0x41, 128, 128, 1).layout().expect("BC1 128");
        assert_eq!(bc1.size, 0x2000);
        assert_eq!(bc1.alignment, 0x800);
        assert_eq!(bc1.block_height_log2, 2);

        let mips = builder(0x25, 256, 256, 3)
            .layout()
            .expect("RGBA8 with mips");
        assert_eq!(mips.size, 0x40000 + 0x10000 + 0x4000);

        let array = TextureBuilder {
            target: 4,
            depth: 4,
            ..builder(0x25, 64, 64, 1)
        }
        .layout()
        .expect("2D array");
        assert_eq!(array.size, 4 * (64 * 4 * 64));
    }

    #[test]
    fn pitch_linear_uses_stride_and_alignment() {
        let layout = TextureBuilder {
            stride: 0x100,
            ..builder(0x01, 128, 128, 1)
        }
        .layout()
        .expect("pitch linear");
        assert_eq!(layout.size, 0x8000);
        assert_eq!(layout.alignment, 0x100);
    }

    #[test]
    fn packaged_block_height_is_kept() {
        let layout = TextureBuilder {
            packaged: 2,
            ..builder(0x25, 256, 256, 1)
        }
        .layout()
        .expect("packaged");
        assert_eq!(layout.block_height_log2, 2);
        assert_eq!(layout.alignment, 0x800);
        assert_eq!(layout.size, 0x40000);
    }

    #[test]
    fn unknown_format_has_no_layout() {
        assert_eq!(builder(0, 64, 64, 1).layout(), Err(Error::UnknownFormat));
        assert_eq!(builder(0x25, 64, 64, 64).layout(), Err(Error::OutOfRange));
    }

    #[test]
    fn texture_rejects_ranges_outside_its_pool() {
        let pool = Pool {
            storage: STORAGE,
            size: 0x10_0000,
            gpu: 0x1_0000_0000,
            ..Pool::default()
        };
        let mut b = builder(0x25, 64, 64, 1);
        b.pool = 0x1000;
        b.offset = 0x1000;
        let t = Texture::create(0x1000, &pool, &b, STORAGE + 0x1000).expect("texture");
        assert_eq!(
            (t.gpu, t.host, t.storage_size),
            (0x1_0000_1000, STORAGE + 0x1000, 0x4000)
        );
        assert_eq!(t.layers, 1);
        assert_eq!(t.swizzle, DEFAULT_SWIZZLE);

        b.offset = 0xfd000;
        assert_eq!(
            Texture::create(0x1000, &pool, &b, STORAGE + b.offset),
            Err(Error::OutOfRange)
        );
        b.offset = 0x1000;
        assert_eq!(Texture::create(0x1000, &pool, &b, 0), Err(Error::Unmapped));
    }

    #[test]
    fn target_rules_give_layers_and_rows() {
        let array = TextureBuilder {
            target: 3,
            height: 8,
            ..builder(0x25, 16, 8, 1)
        };
        assert_eq!((array.layers(), array.rows()), (8, 1));
        let cube = TextureBuilder {
            target: 8,
            ..builder(0x25, 16, 16, 1)
        };
        assert_eq!(cube.layers(), 6);
        let cube_array = TextureBuilder {
            target: 9,
            depth: 12,
            ..builder(0x25, 16, 16, 1)
        };
        assert_eq!(cube_array.layers(), 12);
        let volume = TextureBuilder {
            target: 2,
            depth: 8,
            ..builder(0x25, 16, 16, 1)
        };
        assert_eq!((volume.layers(), volume.slices()), (1, 8));
    }
}
