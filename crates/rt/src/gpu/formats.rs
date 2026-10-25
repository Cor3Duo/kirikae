use ash::vk;
use kirikae_maxwell::Kind;

#[derive(Clone, Copy, Debug)]
pub struct ColorFormat {
    pub vk: vk::Format,
    pub bytes: u32,
    pub kind: Kind,
}

pub fn color_target(v: u32) -> Option<ColorFormat> {
    use vk::Format as F;
    let (vk, bytes, kind) = match v {
        0xc0 => (F::R32G32B32A32_SFLOAT, 16, Kind::Float),
        0xc1 => (F::R32G32B32A32_SINT, 16, Kind::Sint),
        0xc2 => (F::R32G32B32A32_UINT, 16, Kind::Uint),
        0xc3 => (F::R32G32B32A32_SFLOAT, 16, Kind::Float),
        0xc6 => (F::R16G16B16A16_UNORM, 8, Kind::Float),
        0xc7 => (F::R16G16B16A16_SNORM, 8, Kind::Float),
        0xc8 => (F::R16G16B16A16_SINT, 8, Kind::Sint),
        0xc9 => (F::R16G16B16A16_UINT, 8, Kind::Uint),
        0xca | 0xce => (F::R16G16B16A16_SFLOAT, 8, Kind::Float),
        0xcb => (F::R32G32_SFLOAT, 8, Kind::Float),
        0xcc => (F::R32G32_SINT, 8, Kind::Sint),
        0xcd => (F::R32G32_UINT, 8, Kind::Uint),
        0xcf | 0xe6 => (F::B8G8R8A8_UNORM, 4, Kind::Float),
        0xd0 | 0xe7 => (F::B8G8R8A8_SRGB, 4, Kind::Float),
        0xd1 => (F::A2B10G10R10_UNORM_PACK32, 4, Kind::Float),
        0xd2 => (F::A2B10G10R10_UINT_PACK32, 4, Kind::Uint),
        0xd5 | 0xf9 => (F::R8G8B8A8_UNORM, 4, Kind::Float),
        0xd6 | 0xfa => (F::R8G8B8A8_SRGB, 4, Kind::Float),
        0xd7 => (F::R8G8B8A8_SNORM, 4, Kind::Float),
        0xd8 => (F::R8G8B8A8_SINT, 4, Kind::Sint),
        0xd9 => (F::R8G8B8A8_UINT, 4, Kind::Uint),
        0xda => (F::R16G16_UNORM, 4, Kind::Float),
        0xdb => (F::R16G16_SNORM, 4, Kind::Float),
        0xdc => (F::R16G16_SINT, 4, Kind::Sint),
        0xdd => (F::R16G16_UINT, 4, Kind::Uint),
        0xde => (F::R16G16_SFLOAT, 4, Kind::Float),
        0xdf => (F::A2R10G10B10_UNORM_PACK32, 4, Kind::Float),
        0xe0 => (F::B10G11R11_UFLOAT_PACK32, 4, Kind::Float),
        0xe3 => (F::R32_SINT, 4, Kind::Sint),
        0xe4 => (F::R32_UINT, 4, Kind::Uint),
        0xe5 => (F::R32_SFLOAT, 4, Kind::Float),
        0xe8 => (F::R5G6B5_UNORM_PACK16, 2, Kind::Float),
        0xe9 | 0xf8 => (F::A1R5G5B5_UNORM_PACK16, 2, Kind::Float),
        0xea => (F::R8G8_UNORM, 2, Kind::Float),
        0xeb => (F::R8G8_SNORM, 2, Kind::Float),
        0xec => (F::R8G8_SINT, 2, Kind::Sint),
        0xed => (F::R8G8_UINT, 2, Kind::Uint),
        0xee => (F::R16_UNORM, 2, Kind::Float),
        0xef => (F::R16_SNORM, 2, Kind::Float),
        0xf0 => (F::R16_SINT, 2, Kind::Sint),
        0xf1 => (F::R16_UINT, 2, Kind::Uint),
        0xf2 => (F::R16_SFLOAT, 2, Kind::Float),
        0xf3 | 0xf7 => (F::R8_UNORM, 1, Kind::Float),
        0xf4 => (F::R8_SNORM, 1, Kind::Float),
        0xf5 => (F::R8_SINT, 1, Kind::Sint),
        0xf6 => (F::R8_UINT, 1, Kind::Uint),
        _ => return None,
    };
    Some(ColorFormat { vk, bytes, kind })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexFormat {
    pub vk: vk::Format,
    pub block_bytes: u32,
    pub block_w: u32,
    pub block_h: u32,
    pub astc: bool,
}

const fn plain(vk: vk::Format, bytes: u32) -> TexFormat {
    TexFormat {
        vk,
        block_bytes: bytes,
        block_w: 1,
        block_h: 1,
        astc: false,
    }
}
const fn block(vk: vk::Format, bytes: u32) -> TexFormat {
    TexFormat {
        vk,
        block_bytes: bytes,
        block_w: 4,
        block_h: 4,
        astc: false,
    }
}

pub fn texture(components: u32, data_type: u32, srgb: bool) -> Option<TexFormat> {
    use vk::Format as F;
    const SNORM: u32 = 1;
    const UNORM: u32 = 2;
    const SINT: u32 = 3;
    const UINT: u32 = 4;
    const FLOAT: u32 = 7;
    let f = match (components, data_type) {
        (0x01, FLOAT) => plain(F::R32G32B32A32_SFLOAT, 16),
        (0x01, SINT) => plain(F::R32G32B32A32_SINT, 16),
        (0x01, UINT) => plain(F::R32G32B32A32_UINT, 16),
        (0x02, FLOAT) => plain(F::R32G32B32A32_SFLOAT, 12),
        (0x03, FLOAT) => plain(F::R16G16B16A16_SFLOAT, 8),
        (0x03, UNORM) => plain(F::R16G16B16A16_UNORM, 8),
        (0x03, SNORM) => plain(F::R16G16B16A16_SNORM, 8),
        (0x03, SINT) => plain(F::R16G16B16A16_SINT, 8),
        (0x03, UINT) => plain(F::R16G16B16A16_UINT, 8),
        (0x04, FLOAT) => plain(F::R32G32_SFLOAT, 8),
        (0x04, SINT) => plain(F::R32G32_SINT, 8),
        (0x04, UINT) => plain(F::R32G32_UINT, 8),
        (0x07 | 0x08, UNORM) if srgb => plain(F::R8G8B8A8_SRGB, 4),
        (0x07 | 0x08, UNORM) => plain(F::R8G8B8A8_UNORM, 4),
        (0x08, SNORM) => plain(F::R8G8B8A8_SNORM, 4),
        (0x08, SINT) => plain(F::R8G8B8A8_SINT, 4),
        (0x08, UINT) => plain(F::R8G8B8A8_UINT, 4),
        (0x09, UNORM) => plain(F::A2B10G10R10_UNORM_PACK32, 4),
        (0x09, UINT) => plain(F::A2B10G10R10_UINT_PACK32, 4),
        (0x0c, FLOAT) => plain(F::R16G16_SFLOAT, 4),
        (0x0c, UNORM) => plain(F::R16G16_UNORM, 4),
        (0x0c, SNORM) => plain(F::R16G16_SNORM, 4),
        (0x0c, SINT) => plain(F::R16G16_SINT, 4),
        (0x0c, UINT) => plain(F::R16G16_UINT, 4),
        (0x0f, FLOAT) => plain(F::R32_SFLOAT, 4),
        (0x0f, SINT) => plain(F::R32_SINT, 4),
        (0x0f, UINT) => plain(F::R32_UINT, 4),
        (0x12, _) => plain(F::R4G4B4A4_UNORM_PACK16, 2),
        (0x13, _) => plain(F::A1R5G5B5_UNORM_PACK16, 2),
        (0x14, _) => plain(F::A1R5G5B5_UNORM_PACK16, 2),
        (0x15, _) => plain(F::R5G6B5_UNORM_PACK16, 2),
        (0x18, UNORM) => plain(F::R8G8_UNORM, 2),
        (0x18, SNORM) => plain(F::R8G8_SNORM, 2),
        (0x18, SINT) => plain(F::R8G8_SINT, 2),
        (0x18, UINT) => plain(F::R8G8_UINT, 2),
        (0x1b, FLOAT) => plain(F::R16_SFLOAT, 2),
        (0x1b, UNORM) => plain(F::R16_UNORM, 2),
        (0x1b, SNORM) => plain(F::R16_SNORM, 2),
        (0x1b, SINT) => plain(F::R16_SINT, 2),
        (0x1b, UINT) => plain(F::R16_UINT, 2),
        (0x1d, UNORM) => plain(F::R8_UNORM, 1),
        (0x1d, SNORM) => plain(F::R8_SNORM, 1),
        (0x1d, SINT) => plain(F::R8_SINT, 1),
        (0x1d, UINT) => plain(F::R8_UINT, 1),
        (0x20, _) => plain(F::E5B9G9R9_UFLOAT_PACK32, 4),
        (0x21, _) => plain(F::B10G11R11_UFLOAT_PACK32, 4),
        (0x24, _) if srgb => block(F::BC1_RGBA_SRGB_BLOCK, 8),
        (0x24, _) => block(F::BC1_RGBA_UNORM_BLOCK, 8),
        (0x25, _) if srgb => block(F::BC2_SRGB_BLOCK, 16),
        (0x25, _) => block(F::BC2_UNORM_BLOCK, 16),
        (0x26, _) if srgb => block(F::BC3_SRGB_BLOCK, 16),
        (0x26, _) => block(F::BC3_UNORM_BLOCK, 16),
        (0x27, SNORM) => block(F::BC4_SNORM_BLOCK, 8),
        (0x27, _) => block(F::BC4_UNORM_BLOCK, 8),
        (0x28, SNORM) => block(F::BC5_SNORM_BLOCK, 16),
        (0x28, _) => block(F::BC5_UNORM_BLOCK, 16),
        (0x10, _) => block(F::BC6H_SFLOAT_BLOCK, 16),
        (0x11, _) => block(F::BC6H_UFLOAT_BLOCK, 16),
        (0x17, _) if srgb => block(F::BC7_SRGB_BLOCK, 16),
        (0x17, _) => block(F::BC7_UNORM_BLOCK, 16),
        (0x29 | 0x2b, _) => plain(F::D24_UNORM_S8_UINT, 4),
        (0x2a, _) => plain(F::X8_D24_UNORM_PACK32, 4),
        (0x2f, _) => plain(F::D32_SFLOAT, 4),
        (0x30, _) => plain(F::D32_SFLOAT_S8_UINT, 8),
        (0x3a, _) => plain(F::D16_UNORM, 2),
        (c @ (0x40..=0x46 | 0x50..=0x57), _) => {
            let (w, h) = astc_block(c)?;
            TexFormat {
                vk: if srgb {
                    F::R8G8B8A8_SRGB
                } else {
                    F::R8G8B8A8_UNORM
                },
                block_bytes: 16,
                block_w: w,
                block_h: h,
                astc: true,
            }
        }
        _ => return None,
    };
    Some(f)
}

pub fn astc_block(components: u32) -> Option<(u32, u32)> {
    Some(match components {
        0x40 => (4, 4),
        0x50 => (5, 4),
        0x41 => (5, 5),
        0x51 => (6, 5),
        0x42 => (6, 6),
        0x55 => (8, 5),
        0x52 => (8, 6),
        0x44 => (8, 8),
        0x56 => (10, 5),
        0x57 => (10, 6),
        0x53 => (10, 8),
        0x45 => (10, 10),
        0x54 => (12, 10),
        0x46 => (12, 12),
        _ => return None,
    })
}

pub fn vertex_attribute(sizes: u32, ty: u32) -> Option<(vk::Format, Kind)> {
    use vk::Format as F;
    const SNORM: u32 = 1;
    const UNORM: u32 = 2;
    const SINT: u32 = 3;
    const UINT: u32 = 4;
    const USCALED: u32 = 5;
    const SSCALED: u32 = 6;
    const FLOAT: u32 = 7;
    let kind = match ty {
        SINT => Kind::Sint,
        UINT => Kind::Uint,
        _ => Kind::Float,
    };
    let pick = |f: [F; 7]| f[(ty.clamp(1, 7) - 1) as usize];
    let f = match sizes {
        0x01 => pick([
            F::UNDEFINED,
            F::UNDEFINED,
            F::R32G32B32A32_SINT,
            F::R32G32B32A32_UINT,
            F::UNDEFINED,
            F::UNDEFINED,
            F::R32G32B32A32_SFLOAT,
        ]),
        0x02 => pick([
            F::UNDEFINED,
            F::UNDEFINED,
            F::R32G32B32_SINT,
            F::R32G32B32_UINT,
            F::UNDEFINED,
            F::UNDEFINED,
            F::R32G32B32_SFLOAT,
        ]),
        0x03 => pick([
            F::R16G16B16A16_SNORM,
            F::R16G16B16A16_UNORM,
            F::R16G16B16A16_SINT,
            F::R16G16B16A16_UINT,
            F::R16G16B16A16_USCALED,
            F::R16G16B16A16_SSCALED,
            F::R16G16B16A16_SFLOAT,
        ]),
        0x04 => pick([
            F::UNDEFINED,
            F::UNDEFINED,
            F::R32G32_SINT,
            F::R32G32_UINT,
            F::UNDEFINED,
            F::UNDEFINED,
            F::R32G32_SFLOAT,
        ]),
        0x05 => pick([
            F::R16G16B16_SNORM,
            F::R16G16B16_UNORM,
            F::R16G16B16_SINT,
            F::R16G16B16_UINT,
            F::R16G16B16_USCALED,
            F::R16G16B16_SSCALED,
            F::R16G16B16_SFLOAT,
        ]),
        0x0a | 0x2f => pick([
            F::R8G8B8A8_SNORM,
            F::R8G8B8A8_UNORM,
            F::R8G8B8A8_SINT,
            F::R8G8B8A8_UINT,
            F::R8G8B8A8_USCALED,
            F::R8G8B8A8_SSCALED,
            F::UNDEFINED,
        ]),
        0x0f => pick([
            F::R16G16_SNORM,
            F::R16G16_UNORM,
            F::R16G16_SINT,
            F::R16G16_UINT,
            F::R16G16_USCALED,
            F::R16G16_SSCALED,
            F::R16G16_SFLOAT,
        ]),
        0x12 => pick([
            F::UNDEFINED,
            F::UNDEFINED,
            F::R32_SINT,
            F::R32_UINT,
            F::UNDEFINED,
            F::UNDEFINED,
            F::R32_SFLOAT,
        ]),
        0x13 => pick([
            F::R8G8B8_SNORM,
            F::R8G8B8_UNORM,
            F::R8G8B8_SINT,
            F::R8G8B8_UINT,
            F::R8G8B8_USCALED,
            F::R8G8B8_SSCALED,
            F::UNDEFINED,
        ]),
        0x18 | 0x32 => pick([
            F::R8G8_SNORM,
            F::R8G8_UNORM,
            F::R8G8_SINT,
            F::R8G8_UINT,
            F::R8G8_USCALED,
            F::R8G8_SSCALED,
            F::UNDEFINED,
        ]),
        0x1b => pick([
            F::R16_SNORM,
            F::R16_UNORM,
            F::R16_SINT,
            F::R16_UINT,
            F::R16_USCALED,
            F::R16_SSCALED,
            F::R16_SFLOAT,
        ]),
        0x1d | 0x34 => pick([
            F::R8_SNORM,
            F::R8_UNORM,
            F::R8_SINT,
            F::R8_UINT,
            F::R8_USCALED,
            F::R8_SSCALED,
            F::UNDEFINED,
        ]),
        0x30 => pick([
            F::A2B10G10R10_SNORM_PACK32,
            F::A2B10G10R10_UNORM_PACK32,
            F::A2B10G10R10_SINT_PACK32,
            F::A2B10G10R10_UINT_PACK32,
            F::A2B10G10R10_USCALED_PACK32,
            F::A2B10G10R10_SSCALED_PACK32,
            F::UNDEFINED,
        ]),
        0x31 => F::B10G11R11_UFLOAT_PACK32,
        _ => F::UNDEFINED,
    };
    let _ = (USCALED, SSCALED, FLOAT, SNORM, UNORM);
    (f != F::UNDEFINED).then_some((f, kind))
}

pub fn topology(op: u32) -> vk::PrimitiveTopology {
    use vk::PrimitiveTopology as T;
    match op {
        0 => T::POINT_LIST,
        1 => T::LINE_LIST,
        2 | 3 => T::LINE_STRIP,
        5 | 8 => T::TRIANGLE_STRIP,
        6 | 9 => T::TRIANGLE_FAN,
        10 => T::LINE_LIST_WITH_ADJACENCY,
        11 => T::LINE_STRIP_WITH_ADJACENCY,
        12 => T::TRIANGLE_LIST_WITH_ADJACENCY,
        13 => T::TRIANGLE_STRIP_WITH_ADJACENCY,
        14 => T::PATCH_LIST,
        _ => T::TRIANGLE_LIST,
    }
}

pub fn blend_factor(v: u32) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match v {
        0x4000 | 0x1 => B::ZERO,
        0x4001 | 0x2 => B::ONE,
        0x4300 | 0x3 => B::SRC_COLOR,
        0x4301 | 0x4 => B::ONE_MINUS_SRC_COLOR,
        0x4302 | 0x5 => B::SRC_ALPHA,
        0x4303 | 0x6 => B::ONE_MINUS_SRC_ALPHA,
        0x4304 | 0x7 => B::DST_ALPHA,
        0x4305 | 0x8 => B::ONE_MINUS_DST_ALPHA,
        0x4306 | 0x9 => B::DST_COLOR,
        0x4307 | 0xa => B::ONE_MINUS_DST_COLOR,
        0x4308 | 0xb => B::SRC_ALPHA_SATURATE,
        0xc001 | 0xe => B::CONSTANT_COLOR,
        0xc002 | 0xf => B::ONE_MINUS_CONSTANT_COLOR,
        0xc003 => B::CONSTANT_ALPHA,
        0xc004 => B::ONE_MINUS_CONSTANT_ALPHA,
        0xc900 | 0x10 => B::SRC1_COLOR,
        0xc901 | 0x11 => B::ONE_MINUS_SRC1_COLOR,
        0xc902 | 0x12 => B::SRC1_ALPHA,
        0xc903 | 0x13 => B::ONE_MINUS_SRC1_ALPHA,
        _ => B::ONE,
    }
}

pub fn blend_op(v: u32) -> vk::BlendOp {
    use vk::BlendOp as O;
    match v {
        0x800a | 2 => O::SUBTRACT,
        0x800b | 3 => O::REVERSE_SUBTRACT,
        0x8007 | 4 => O::MIN,
        0x8008 | 5 => O::MAX,
        _ => O::ADD,
    }
}

pub fn compare_op(v: u32) -> vk::CompareOp {
    use vk::CompareOp as C;
    let n = if v >= 0x200 {
        v - 0x200
    } else {
        v.saturating_sub(1)
    };
    match n & 0xf {
        0 => C::NEVER,
        1 => C::LESS,
        2 => C::EQUAL,
        3 => C::LESS_OR_EQUAL,
        4 => C::GREATER,
        5 => C::NOT_EQUAL,
        6 => C::GREATER_OR_EQUAL,
        _ => C::ALWAYS,
    }
}

pub fn stencil_op(v: u32) -> vk::StencilOp {
    use vk::StencilOp as S;
    match v {
        0 | 2 => S::ZERO,
        0x1e01 | 3 => S::REPLACE,
        0x1e02 | 4 => S::INCREMENT_AND_CLAMP,
        0x1e03 | 5 => S::DECREMENT_AND_CLAMP,
        0x150a | 6 => S::INVERT,
        0x8507 | 7 => S::INCREMENT_AND_WRAP,
        0x8508 | 8 => S::DECREMENT_AND_WRAP,
        _ => S::KEEP,
    }
}
