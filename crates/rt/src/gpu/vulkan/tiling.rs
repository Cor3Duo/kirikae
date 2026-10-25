use crate::gpu::block_linear_offset;
use crate::gpu::formats::TexFormat;
use crate::gpu::tic::{Layout, TextureHeader};

pub fn deswizzle(raw: &[u8], width: u32, height: u32, bpp: u32, block_height_log2: u32) -> Vec<u8> {
    let row = width as u64 * bpp as u64;
    let width_bytes = row.div_ceil(64) * 64;
    let mut out = vec![0u8; (row * height as u64) as usize];
    for y in 0..height as u64 {
        let dst = &mut out[(y * row) as usize..][..row as usize];
        let mut x = 0u64;
        while x < row {
            let o = block_linear_offset(x, y, width_bytes, block_height_log2) as usize;
            let n = (16 - (x % 16)).min(row - x) as usize;
            if let Some(src) = raw.get(o..o + n) {
                dst[x as usize..x as usize + n].copy_from_slice(src);
            }
            x += n as u64;
        }
    }
    out
}

pub(super) fn fit_block(log2: u32, extent: u64) -> u32 {
    let mut l = log2;
    while l > 0 && (1u64 << (l - 1)) >= extent {
        l -= 1;
    }
    l
}

pub(super) fn volume_size(
    width: u32,
    height: u32,
    depth: u32,
    bpp: u32,
    block_height_log2: u32,
    block_depth_log2: u32,
) -> u64 {
    let gobs_wide = (width as u64 * bpp as u64).div_ceil(64);
    let gobs_high = (height as u64).div_ceil(8);
    let bh = fit_block(block_height_log2, gobs_high);
    let bd = fit_block(block_depth_log2, depth as u64);
    let blocks_high = gobs_high.div_ceil(1 << bh);
    let blocks_deep = (depth as u64).div_ceil(1 << bd);
    gobs_wide * blocks_high * blocks_deep * (512 << (bh + bd))
}

pub(super) fn deswizzle_3d(
    raw: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    bpp: u32,
    block_height_log2: u32,
    block_depth_log2: u32,
) -> Vec<u8> {
    let row = width as u64 * bpp as u64;
    let gobs_wide = row.div_ceil(64);
    let gobs_high = (height as u64).div_ceil(8);
    let bh = fit_block(block_height_log2, gobs_high);
    let bd = fit_block(block_depth_log2, depth as u64);
    let blocks_high = gobs_high.div_ceil(1 << bh);
    let block_size = 512u64 << (bh + bd);
    let mut out = vec![0u8; (row * height as u64 * depth as u64) as usize];
    for z in 0..depth as u64 {
        for y in 0..height as u64 {
            let dst = ((z * height as u64 + y) * row) as usize;
            let mut x = 0u64;
            while x < row {
                let block = ((z >> bd) * blocks_high + ((y / 8) >> bh)) * gobs_wide + x / 64;
                let gob = ((z & ((1 << bd) - 1)) << bh) + ((y / 8) & ((1 << bh) - 1));
                let (gx, gy) = (x % 64, y % 8);
                let inner = (gx / 32) * 256
                    + (gy / 2) * 64
                    + ((gx % 32) / 16) * 32
                    + (gy % 2) * 16
                    + (gx % 16);
                let o = (block * block_size + gob * 512 + inner) as usize;
                let n = (16 - (x % 16)).min(row - x) as usize;
                if let Some(src) = raw.get(o..o + n) {
                    out[dst + x as usize..dst + x as usize + n].copy_from_slice(src);
                }
                x += n as u64;
            }
        }
    }
    out
}

pub(super) fn layer_alignment(h: &TextureHeader, f: &TexFormat) -> Option<u64> {
    let Layout::BlockLinear {
        block_height_log2, ..
    } = h.layout
    else {
        return None;
    };
    let gobs_high = (h.height.div_ceil(f.block_h) as u64).div_ceil(8);
    let mut bhl = block_height_log2;
    while bhl > 0 && (1u64 << (bhl - 1)) >= gobs_high {
        bhl -= 1;
    }
    Some(512 << bhl)
}

pub(super) fn layer_size(h: &TextureHeader, f: &TexFormat, levels: u32) -> Option<u64> {
    let Layout::BlockLinear { .. } = h.layout else {
        return None;
    };
    let size = level_offset(h, f, levels);
    let align = layer_alignment(h, f)?;
    Some(size.div_ceil(align) * align)
}

pub(super) fn level_offset(h: &TextureHeader, f: &TexFormat, level: u32) -> u64 {
    let Layout::BlockLinear {
        block_height_log2, ..
    } = h.layout
    else {
        return 0;
    };
    let mut size = 0u64;
    for level in 0..level {
        let (bw, bh) = (
            (h.width >> level).max(1).div_ceil(f.block_w),
            (h.height >> level).max(1).div_ceil(f.block_h),
        );
        let gobs_high = (bh as u64).div_ceil(8);
        let mut bhl = block_height_log2;
        while bhl > 0 && (1u64 << (bhl - 1)) >= gobs_high {
            bhl -= 1;
        }
        let rows = 8u64 << bhl;
        size += (bw as u64 * f.block_bytes as u64).div_ceil(64)
            * 64
            * (bh as u64).div_ceil(rows)
            * rows;
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::formats;

    #[test]
    fn volume_slices() {
        let (w, h, d, bpp) = (32u32, 24u32, 3u32, 4u32);
        let slice = volume_size(w, h, 1, bpp, 1, 0);
        assert_eq!(volume_size(w, h, d, bpp, 1, 0), slice * d as u64);
        let mut raw = vec![0u8; (slice * d as u64) as usize];
        for z in 0..d as u64 {
            for y in 0..h as u64 {
                for x in 0..(w * bpp) as u64 {
                    raw[(z * slice + block_linear_offset(x, y, (w * bpp) as u64, 1)) as usize] =
                        (x + 7 * y + 31 * z) as u8;
                }
            }
        }
        let lin = deswizzle_3d(&raw, w, h, d, bpp, 1, 0);
        let row = (w * bpp) as usize;
        for z in 0..d as usize {
            for y in 0..h as usize {
                for x in 0..row {
                    assert_eq!(
                        lin[(z * h as usize + y) * row + x],
                        (x + 7 * y + 31 * z) as u8
                    );
                }
            }
        }
        let raw: Vec<u8> = (0..1024).map(|i| (i / 512) as u8).collect();
        let lin = deswizzle_3d(&raw, 16, 8, 2, 4, 0, 1);
        assert_eq!((lin[0], lin[16 * 4 * 8]), (0, 1));
    }

    fn header(width: u32, height: u32, levels: u32, block_height_log2: u32) -> TextureHeader {
        TextureHeader {
            components: 8,
            data_type: 2,
            data_types: [2; 4],
            swizzle: [2, 3, 4, 5],
            address: 0,
            layout: Layout::BlockLinear {
                block_height_log2,
                block_depth_log2: 0,
            },
            width,
            height,
            depth: 6,
            texture_type: 1,
            levels,
            min_level: 0,
            max_level: levels - 1,
            srgb: false,
        }
    }

    #[test]
    fn cube_layer_stride() {
        let f = formats::texture(8, 2, false).unwrap();
        assert_eq!(layer_size(&header(64, 64, 7, 3), &f, 7), Some(0x6000));
        assert_eq!(layer_size(&header(64, 64, 1, 3), &f, 1), Some(0x4000));
    }
}
