//! NVN texture format blocks: bytes per block, block width and block height.
//!
//! The real driver reads this from its own format table (`nvn_texture_layout.cpp:5-6, 19-38`); the table is
//! public in the NVN format list, so the host keeps a copy instead of reading the SDK binary. `block` returns
//! zeros for a format without a layout, which callers must reject rather than size a texture at zero.

/// `(bytes per block, block width, block height)` for an NVN format.
pub fn block(format: u32) -> (u32, u32, u32) {
    let c = |bytes| (bytes, 1, 1);
    match format {
        0x01..=0x04 | 0x31 | 0x51 | 0x52 => c(1),
        0x05..=0x09
        | 0x0d..=0x10
        | 0x32
        | 0x39..=0x3c
        | 0x53
        | 0x54
        | 0x57
        | 0x58
        | 0x95..=0x97 => c(2),
        0x19..=0x1c | 0x5d | 0x5e => c(3),
        0x0a..=0x0c
        | 0x11..=0x15
        | 0x25..=0x28
        | 0x33..=0x35
        | 0x37
        | 0x38
        | 0x3d..=0x40
        | 0x55
        | 0x56
        | 0x59
        | 0x5a
        | 0x63
        | 0x64
        | 0x69..=0x70
        | 0x98..=0x9b => c(4),
        0x1d..=0x21 | 0x5f | 0x60 => c(6),
        0x16..=0x18 | 0x29..=0x2d | 0x36 | 0x5b | 0x5c | 0x65 | 0x66 | 0x71..=0x75 => c(8),
        0x22..=0x24 | 0x61 | 0x62 => c(12),
        0x2e..=0x30 | 0x67 | 0x68 | 0x76..=0x78 => c(16),
        0x41 | 0x42 | 0x45 | 0x46 | 0x49 | 0x4a => (8, 4, 4),
        0x43 | 0x44 | 0x47 | 0x48 | 0x4b..=0x50 => (16, 4, 4),
        0x79..=0x94 => {
            const ASTC: [(u32, u32); 14] = [
                (4, 4),
                (5, 4),
                (5, 5),
                (6, 5),
                (6, 6),
                (8, 5),
                (8, 6),
                (8, 8),
                (10, 5),
                (10, 6),
                (10, 8),
                (10, 10),
                (12, 10),
                (12, 12),
            ];
            let (w, h) = ASTC[((format - 0x79) % 14) as usize];
            (16, w, h)
        }
        _ => (0, 1, 1),
    }
}

/// The Vulkan format an NVN format names, for a render target, a vertex attribute or a texture view.
///
/// The numbers are the SDK driver's own table order (`docs/nvn/CONTRACT.md` §6): R8 R16 R32 RG8 RG16 RG32 RGB8 RGB16
/// RGB32 RGBA8 RGBA16 RGBA32 (normalized, signed, integer), stencil and depth, sRGB RGBA8, 16-bit packed, RGB10A2,
/// R11G11B10F, RGB9E5, BC1-BC7. Vertex-only formats follow at 0x51; some have no Vulkan equivalent and read as their
/// texture format. This is the mapping, not a claim about the SDK; the driver's table remains the authority and the
/// TIC entries it writes are what sampling uses.
pub fn vk(format: u32) -> Option<ash::vk::Format> {
    use ash::vk::Format as F;
    let f = match format {
        0x01 => F::R8_UNORM,
        0x02 => F::R8_SNORM,
        0x03 => F::R8_UINT,
        0x04 => F::R8_SINT,
        0x05 => F::R16_SFLOAT,
        0x06 => F::R16_UNORM,
        0x07 => F::R16_SNORM,
        0x08 => F::R16_UINT,
        0x09 => F::R16_SINT,
        0x0a => F::R32_SFLOAT,
        0x0b => F::R32_UINT,
        0x0c => F::R32_SINT,
        0x0d => F::R8G8_UNORM,
        0x0e => F::R8G8_SNORM,
        0x0f => F::R8G8_UINT,
        0x10 => F::R8G8_SINT,
        0x11 => F::R16G16_SFLOAT,
        0x12 => F::R16G16_UNORM,
        0x13 => F::R16G16_SNORM,
        0x14 => F::R16G16_UINT,
        0x15 => F::R16G16_SINT,
        0x16 => F::R32G32_SFLOAT,
        0x17 => F::R32G32_UINT,
        0x18 => F::R32G32_SINT,
        0x19 => F::R8G8B8_UNORM,
        0x1a => F::R8G8B8_SNORM,
        0x1b => F::R8G8B8_UINT,
        0x1c => F::R8G8B8_SINT,
        0x1d => F::R16G16B16_SFLOAT,
        0x1e => F::R16G16B16_UNORM,
        0x1f => F::R16G16B16_SNORM,
        0x20 => F::R16G16B16_UINT,
        0x21 => F::R16G16B16_SINT,
        0x22 => F::R32G32B32_SFLOAT,
        0x23 => F::R32G32B32_UINT,
        0x24 => F::R32G32B32_SINT,
        0x25 => F::R8G8B8A8_UNORM,
        0x26 => F::R8G8B8A8_SNORM,
        0x27 => F::R8G8B8A8_UINT,
        0x28 => F::R8G8B8A8_SINT,
        0x29 => F::R16G16B16A16_SFLOAT,
        0x2a => F::R16G16B16A16_UNORM,
        0x2b => F::R16G16B16A16_SNORM,
        0x2c => F::R16G16B16A16_UINT,
        0x2d => F::R16G16B16A16_SINT,
        0x2e => F::R32G32B32A32_SFLOAT,
        0x2f => F::R32G32B32A32_UINT,
        0x30 => F::R32G32B32A32_SINT,
        0x31 => F::S8_UINT,
        0x32 => F::D16_UNORM,
        // 0x33 is D24 in the driver's table; a depth attachment is widened to D32 as the Maxwell path does.
        0x33 | 0x34 => F::D32_SFLOAT,
        0x35 | 0x36 => F::D32_SFLOAT_S8_UINT,
        0x37 | 0x38 => F::R8G8B8A8_SRGB,
        0x39 => F::A4B4G4R4_UNORM_PACK16,
        0x3a | 0x3b => F::A1R5G5B5_UNORM_PACK16,
        0x3c => F::R5G6B5_UNORM_PACK16,
        0x3d => F::A2B10G10R10_UNORM_PACK32,
        0x3e => F::A2B10G10R10_UINT_PACK32,
        0x3f => F::R16G16B16A16_SFLOAT,
        0x40 => F::E5B9G9R9_UFLOAT_PACK32,
        0x41 => F::BC1_RGB_UNORM_BLOCK,
        0x42 => F::BC1_RGBA_UNORM_BLOCK,
        0x43 => F::BC2_UNORM_BLOCK,
        0x44 => F::BC3_UNORM_BLOCK,
        0x45 => F::BC1_RGB_SRGB_BLOCK,
        0x46 => F::BC1_RGBA_SRGB_BLOCK,
        0x47 => F::BC2_SRGB_BLOCK,
        0x48 => F::BC3_SRGB_BLOCK,
        0x49 => F::BC4_UNORM_BLOCK,
        0x4a => F::BC4_SNORM_BLOCK,
        0x4b => F::BC5_UNORM_BLOCK,
        0x4c => F::BC5_SNORM_BLOCK,
        0x4d => F::BC7_UNORM_BLOCK,
        0x4e => F::BC7_SRGB_BLOCK,
        0x4f => F::BC6H_SFLOAT_BLOCK,
        0x50 => F::BC6H_UFLOAT_BLOCK,
        0x51 => F::R8_USCALED,
        0x52 => F::R8_SSCALED,
        0x53 => F::R16_USCALED,
        0x54 => F::R16_SSCALED,
        0x55 => F::R32_UINT,
        0x56 => F::R32_SINT,
        0x57 => F::R8G8_USCALED,
        0x58 => F::R8G8_SSCALED,
        0x59 => F::R16G16_USCALED,
        0x5a => F::R16G16_SSCALED,
        0x5b => F::R32G32_UINT,
        0x5c => F::R32G32_SINT,
        0x5d => F::R8G8B8_USCALED,
        0x5e => F::R8G8B8_SSCALED,
        0x5f => F::R16G16B16_USCALED,
        0x60 => F::R16G16B16_SSCALED,
        0x61 => F::R32G32B32_UINT,
        0x62 => F::R32G32B32_SINT,
        0x63 => F::R8G8B8A8_USCALED,
        0x64 => F::R8G8B8A8_SSCALED,
        0x65 => F::R16G16B16A16_USCALED,
        0x66 => F::R16G16B16A16_SSCALED,
        0x67 => F::R32G32B32A32_UINT,
        0x68 => F::R32G32B32A32_SINT,
        0x69 => F::A2B10G10R10_SNORM_PACK32,
        0x6a => F::A2B10G10R10_SINT_PACK32,
        0x6b => F::A2B10G10R10_USCALED_PACK32,
        0x6c => F::A2B10G10R10_SSCALED_PACK32,
        0x95 => F::B5G6R5_UNORM_PACK16,
        0x96 | 0x97 => F::B5G5R5A1_UNORM_PACK16,
        0x98 => F::A1R5G5B5_UNORM_PACK16,
        0x99 | 0x9a => F::B8G8R8A8_UNORM,
        0x9b | 0x9c => F::B8G8R8A8_SRGB,
        0x79..=0x94 => return astc(format),
        _ => return None,
    };
    Some(f)
}

fn astc(format: u32) -> Option<ash::vk::Format> {
    use ash::vk::Format as F;
    const LINEAR: [F; 14] = [
        F::ASTC_4X4_UNORM_BLOCK,
        F::ASTC_5X4_UNORM_BLOCK,
        F::ASTC_5X5_UNORM_BLOCK,
        F::ASTC_6X5_UNORM_BLOCK,
        F::ASTC_6X6_UNORM_BLOCK,
        F::ASTC_8X5_UNORM_BLOCK,
        F::ASTC_8X6_UNORM_BLOCK,
        F::ASTC_8X8_UNORM_BLOCK,
        F::ASTC_10X5_UNORM_BLOCK,
        F::ASTC_10X6_UNORM_BLOCK,
        F::ASTC_10X8_UNORM_BLOCK,
        F::ASTC_10X10_UNORM_BLOCK,
        F::ASTC_12X10_UNORM_BLOCK,
        F::ASTC_12X12_UNORM_BLOCK,
    ];
    const SRGB: [F; 14] = [
        F::ASTC_4X4_SRGB_BLOCK,
        F::ASTC_5X4_SRGB_BLOCK,
        F::ASTC_5X5_SRGB_BLOCK,
        F::ASTC_6X5_SRGB_BLOCK,
        F::ASTC_6X6_SRGB_BLOCK,
        F::ASTC_8X5_SRGB_BLOCK,
        F::ASTC_8X6_SRGB_BLOCK,
        F::ASTC_8X8_SRGB_BLOCK,
        F::ASTC_10X5_SRGB_BLOCK,
        F::ASTC_10X6_SRGB_BLOCK,
        F::ASTC_10X8_SRGB_BLOCK,
        F::ASTC_10X10_SRGB_BLOCK,
        F::ASTC_12X10_SRGB_BLOCK,
        F::ASTC_12X12_SRGB_BLOCK,
    ];
    let index = ((format - 0x79) % 14) as usize;
    Some(if format < 0x87 {
        LINEAR[index]
    } else {
        SRGB[index]
    })
}

/// Which aspect Vulkan render-pass and copy work must name for this format.
pub fn aspect(format: u32) -> ash::vk::ImageAspectFlags {
    use ash::vk::ImageAspectFlags as A;
    match format {
        0x31 => A::STENCIL,
        0x32..=0x34 => A::DEPTH,
        0x35 | 0x36 => A::DEPTH | A::STENCIL,
        _ => A::COLOR,
    }
}

/// Whether shaders sample this format as floating point, signed or unsigned integer: the type the Maxwell translation
/// needs for a fragment output or a vertex attribute.
pub fn kind(format: u32) -> kirikae_maxwell::Kind {
    use ash::vk::Format as F;
    use kirikae_maxwell::Kind;
    match vk(format) {
        Some(
            F::R8_UINT
            | F::R16_UINT
            | F::R32_UINT
            | F::R8G8_UINT
            | F::R16G16_UINT
            | F::R32G32_UINT
            | F::R8G8B8_UINT
            | F::R16G16B16_UINT
            | F::R32G32B32_UINT
            | F::R8G8B8A8_UINT
            | F::R16G16B16A16_UINT
            | F::R32G32B32A32_UINT
            | F::A2B10G10R10_UINT_PACK32,
        ) => Kind::Uint,
        Some(
            F::R8_SNORM
            | F::R8_SINT
            | F::R16_SNORM
            | F::R16_SINT
            | F::R32_SINT
            | F::R8G8_SNORM
            | F::R8G8_SINT
            | F::R16G16_SNORM
            | F::R16G16_SINT
            | F::R32G32_SINT
            | F::R8G8B8_SNORM
            | F::R8G8B8_SINT
            | F::R16G16B16_SNORM
            | F::R16G16B16_SINT
            | F::R32G32B32_SINT
            | F::R8G8B8A8_SNORM
            | F::R8G8B8A8_SINT
            | F::R16G16B16A16_SNORM
            | F::R16G16B16A16_SINT
            | F::R32G32B32A32_SINT
            | F::A2B10G10R10_SINT_PACK32
            | F::A2B10G10R10_SNORM_PACK32,
        ) => Kind::Sint,
        _ => Kind::Float,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_formats_follow_the_driver_table() {
        use ash::vk::Format as F;
        assert_eq!(vk(0x25), Some(F::R8G8B8A8_UNORM));
        assert_eq!(vk(0x37), Some(F::R8G8B8A8_SRGB));
        assert_eq!(vk(0x32), Some(F::D16_UNORM));
        assert_eq!(vk(0x35), Some(F::D32_SFLOAT_S8_UINT));
        assert_eq!(vk(0x41), Some(F::BC1_RGB_UNORM_BLOCK));
        assert_eq!(vk(0x79), Some(F::ASTC_4X4_UNORM_BLOCK));
        assert_eq!(vk(0x87), Some(F::ASTC_4X4_SRGB_BLOCK));
        assert_eq!(vk(0x94), Some(F::ASTC_12X12_SRGB_BLOCK));
        assert_eq!(vk(0x6c), Some(F::A2B10G10R10_SSCALED_PACK32));
        assert_eq!(vk(0x9b), Some(F::B8G8R8A8_SRGB));
        assert_eq!(vk(0x00), None);
        assert_eq!(vk(0x6d), None, "0x6d has no Vulkan equivalent in the table");

        assert_eq!(aspect(0x25), ash::vk::ImageAspectFlags::COLOR);
        assert_eq!(aspect(0x32), ash::vk::ImageAspectFlags::DEPTH);
        assert_eq!(aspect(0x31), ash::vk::ImageAspectFlags::STENCIL);
        assert_eq!(
            aspect(0x35),
            ash::vk::ImageAspectFlags::DEPTH | ash::vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(kind(0x25), kirikae_maxwell::Kind::Float);
        assert_eq!(kind(0x27), kirikae_maxwell::Kind::Uint);
        assert_eq!(kind(0x26), kirikae_maxwell::Kind::Sint);
    }

    #[test]
    fn known_blocks() {
        assert_eq!(block(0x25), (4, 1, 1));
        assert_eq!(block(0x01), (1, 1, 1));
        assert_eq!(block(0x41), (8, 4, 4));
        assert_eq!(block(0x79), (16, 4, 4));
        assert_eq!(block(0x94), (16, 12, 12));
        assert_eq!(block(0), (0, 1, 1));
    }
}
