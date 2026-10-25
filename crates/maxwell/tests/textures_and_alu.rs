mod common;

use kirikae_maxwell::{Options, Stage, TexDim, translate};

const PT: u64 = 7 << 16;
const NOP: u64 = 0x50b0_0000_0000_0000 | PT | (0xf << 8);
const EXIT: u64 = 0xe300_0000_0000_0000 | PT | 0xf;

fn fragment_header(targets: u32) -> Vec<u8> {
    let mut words = [0u32; 20];
    words[0] = 2 | 5 << 10;
    for t in 0..targets {
        words[18] |= 0xf << (t * 4);
    }
    words[6] = 0b1010;
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn program(header: Vec<u8>, insns: &[u64]) -> Vec<u8> {
    let mut out = header;
    for chunk in insns.chunks(3) {
        out.extend_from_slice(&0u64.to_le_bytes());
        for k in 0..3 {
            let w = chunk.get(k).copied().unwrap_or(NOP);
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

fn mov32i(rd: u64, imm: u32) -> u64 {
    0x0100_0000_0000_0000 | (imm as u64) << 20 | PT | rd
}
fn ipa(rd: u64, attr: u64, rb: u64) -> u64 {
    0xe000_0000_0000_ff00 | 1 << 54 | attr << 28 | 0xff << 39 | rb << 20 | PT | rd & 0xff
}
fn mov_cbuf(rd: u64, slot: u64, offset: u64) -> u64 {
    0x4c98_0000_0000_0000 | slot << 34 | (offset >> 2) << 20 | PT | rd
}
fn tex_bindless(rd: u64, ra: u64, rb: u64, mask: u64, lod: u64) -> u64 {
    0xdeb8_0000_0000_0000 | rb << 20 | mask << 31 | lod << 37 | 1 << 29 | PT | ra << 8 | rd
}
fn isetp_lt(p: u64, ra: u64, rb: u64) -> u64 {
    0x5b60_0000_0000_0000 | 1 << 49 | 1 << 48 | 7 << 39 | p << 3 | 7 | rb << 20 | PT | ra << 8
}
fn tld_word(base: u64, rd: u64, ra: u64, rb: u64, handle: u64, dim: u64, array: bool) -> u64 {
    base | handle << 36
        | 0xf << 31
        | dim << 29
        | (array as u64) << 28
        | PT
        | rb << 20
        | ra << 8
        | rd
}
fn txd_word(base: u64, rd: u64, ra: u64, rb: u64, handle: u64, dim: u64, array: bool) -> u64 {
    base | handle << 36
        | 0xf << 31
        | dim << 29
        | (array as u64) << 28
        | PT
        | rb << 20
        | ra << 8
        | rd
}
fn imad_word(rd: u64, ra: u64, rb: u64, rc: u64, signed_a: bool, signed_b: bool, hi: bool) -> u64 {
    0x5a00_0000_0000_0000
        | (signed_a as u64) << 40
        | (signed_b as u64) << 41
        | (hi as u64) << 54
        | rc << 39
        | rb << 20
        | PT
        | ra << 8
        | rd
}
fn imul_word(rd: u64, ra: u64, rb: u64, signed_a: bool, signed_b: bool, hi: bool) -> u64 {
    0x5c38_0000_0000_0000
        | (signed_a as u64) << 40
        | (signed_b as u64) << 41
        | (hi as u64) << 39
        | rb << 20
        | PT
        | ra << 8
        | rd
}
fn imad32i_word(rd: u64, ra: u64, imm: u32, signed_a: bool, signed_b: bool, hi: bool) -> u64 {
    0x1000_0000_0000_0000
        | (signed_a as u64) << 54
        | (signed_b as u64) << 57
        | (hi as u64) << 53
        | (imm as u64) << 20
        | PT
        | ra << 8
        | rd
}
fn imul32i_word(rd: u64, ra: u64, imm: u32, signed_a: bool, signed_b: bool, hi: bool) -> u64 {
    0x1f00_0000_0000_0000
        | (signed_a as u64) << 54
        | (signed_b as u64) << 55
        | (hi as u64) << 53
        | (imm as u64) << 20
        | PT
        | ra << 8
        | rd
}

fn options(stage: Stage) -> Options {
    Options {
        local_size: [8, 8, 1],
        ..Options::new(stage, 0)
    }
}

fn disassemble(spirv: &[u32]) -> String {
    use rspirv::binary::Disassemble;
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(spirv, &mut loader).unwrap();
    loader.module().disassemble()
}

fn validate(spirv: &[u32]) {
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    let module =
        match naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default()) {
            Ok(m) => m,
            Err(e) => panic!(
                "SPIR-V does not parse: {e:?}
{}",
                disassemble(spirv)
            ),
        };
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("SPIR-V validates");
    common::check_structure(spirv);
}

#[test]
fn tld_fetches_texels() {
    // 2D, level zero: coordinates in r0/r1, level in r3, table index 5.
    let code = program(
        fragment_header(1),
        &[
            mov32i(0, 0),
            mov32i(1, 0),
            mov32i(3, 0),
            tld_word(0xdc38_0000_0000_0000, 2, 0, 3, 5, 1, false),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.mnemonics.iter().any(|(m, _)| *m == "tld"));
    assert_eq!(shader.textures.len(), 1);
    assert_eq!(shader.textures[0].dim, TexDim::D2);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageFetch"), "{text}");
    validate(&shader.spirv);

    // Explicit LOD level from rb (bit 55).
    let code = program(
        fragment_header(1),
        &[
            mov32i(0, 0),
            mov32i(1, 0),
            mov32i(3, 1),
            tld_word(0xdc38_0000_0000_0000, 2, 0, 3, 5, 1, false) | 1 << 55,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageFetch"), "{text}");
    validate(&shader.spirv);

    // Array: the layer is the first source, then the coordinates.
    let code = program(
        fragment_header(1),
        &[
            mov32i(0, 2),
            mov32i(1, 0),
            mov32i(2, 0),
            mov32i(4, 0),
            tld_word(0xdc38_0000_0000_0000, 3, 0, 4, 6, 1, true),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.textures[0].array);
    validate(&shader.spirv);

    // Bindless register form: the handle comes from a constant-buffer word.
    let code = program(
        fragment_header(1),
        &[
            mov32i(0, 0),
            mov32i(1, 0),
            mov_cbuf(5, 4, 0xb0),
            tld_word(0xdd38_0000_0000_0000, 2, 0, 5, 0, 1, false),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(
        (shader.textures[0].slot, shader.textures[0].handle),
        (Some(4), 0xb0 / 4)
    );
    validate(&shader.spirv);
}

#[test]
fn tld_offsets_and_rejections() {
    // `aoffi` (bit 35) consumes the register after the optional LOD.
    let code = program(
        fragment_header(1),
        &[
            mov32i(0, 0),
            mov32i(1, 0),
            mov32i(3, 0x11),
            mov32i(4, 0),
            tld_word(0xdc38_0000_0000_0000, 2, 0, 3, 5, 1, false) | 1 << 35,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 1);
    validate(&shader.spirv);

    // Bindless: handle in rb, offset in rb + 1.
    let code = program(
        fragment_header(1),
        &[
            mov32i(0, 0),
            mov32i(1, 0),
            mov_cbuf(5, 4, 0xb0),
            mov32i(6, 0x11),
            tld_word(0xdd38_0000_0000_0000, 2, 0, 5, 0, 1, false) | 1 << 35,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 1);
    validate(&shader.spirv);

    let unsupported = [
        (
            tld_word(0xdc38_0000_0000_0000, 2, 0, 3, 5, 1, false) | 1 << 50,
            "tld (multisample)",
        ),
        (
            tld_word(0xdc38_0000_0000_0000, 2, 0, 3, 5, 1, false) | 1 << 54,
            "tld (cl)",
        ),
        (
            tld_word(0xdd38_0000_0000_0000, 2, 0, 9, 0, 1, false),
            "tld (bindless)",
        ),
    ];
    for (tld, label) in unsupported {
        let code = program(
            fragment_header(1),
            &[mov32i(0, 0), mov32i(1, 0), mov32i(3, 0), tld, EXIT],
        );
        let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
        assert_eq!(shader.unsupported, [label]);
        assert!(shader.textures.is_empty(), "{label}");
        validate(&shader.spirv);
    }
}

#[test]
fn bindless_texture_lod_offset_and_depth_subforms() {
    let coords = [ipa(0, 0x80, 0xff), ipa(1, 0x84, 0xff), mov_cbuf(5, 4, 0xb0)];
    let with = |extra: &[u64]| {
        let mut insns: Vec<u64> = coords.to_vec();
        insns.extend_from_slice(extra);
        insns.push(EXIT);
        program(fragment_header(1), &insns)
    };

    // LOD bias (mode 2) reads rb + 1.
    let code = with(&[mov32i(6, 0.5f32.to_bits()), tex_bindless(2, 0, 5, 0xf, 2)]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageSampleImplicitLod"), "{text}");
    assert!(text.contains("Bias"), "{text}");
    validate(&shader.spirv);

    // LOD level (mode 3).
    let code = with(&[mov32i(6, 1.0f32.to_bits()), tex_bindless(2, 0, 5, 0xf, 3)]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageSampleExplicitLod"), "{text}");
    validate(&shader.spirv);

    // Per-texel offsets (aoffi, bit 36) read rb + 1.
    let code = with(&[mov32i(6, 0x11), tex_bindless(2, 0, 5, 0xf, 0) | 1 << 36]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageQuerySizeLod"), "{text}");
    validate(&shader.spirv);

    // Depth compare (dc, bit 50) reads rb + 1.
    let code = with(&[
        mov32i(6, 0.5f32.to_bits()),
        tex_bindless(2, 0, 5, 0xf, 0) | 1 << 50,
    ]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.textures[0].shadow);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageSampleDrefImplicitLod"), "{text}");
    validate(&shader.spirv);

    // Combined aoffi + dc: offset in rb + 1, depth in rb + 2.
    let code = with(&[
        mov32i(6, 0x11),
        mov32i(7, 0.5f32.to_bits()),
        tex_bindless(2, 0, 5, 0xf, 0) | 1 << 36 | 1 << 50,
    ]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageSampleDrefImplicitLod"), "{text}");
    validate(&shader.spirv);

    for (lod, label) in [(6u64, "tex (bindless lba)"), (7, "tex (bindless lla)")] {
        let code = with(&[mov32i(6, 0.5f32.to_bits()), tex_bindless(2, 0, 5, 0xf, lod)]);
        let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
        assert_eq!(shader.unsupported, [label]);
        assert!(shader.textures.is_empty(), "{label}");
        validate(&shader.spirv);
    }
}

#[test]
fn immediate_texture_offset_and_depth() {
    let tex = 0xc038_0000_0000_0000u64 | 7 << 36 | 0xf << 31 | 1 << 29 | 1 << 54 | 1 << 50 | PT | 2;
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov32i(3, 0x11),
            mov32i(4, 0.5f32.to_bits()),
            tex,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.textures[0].shadow);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageQuerySizeLod"), "{text}");
    assert!(text.contains("OpImageSampleDrefImplicitLod"), "{text}");
    validate(&shader.spirv);
}

#[test]
fn texture_derivatives() {
    // Immediate: coordinates in ra, derivatives in rb.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov32i(2, 0x3c00_0000),
            mov32i(3, 0),
            mov32i(4, 0x3c00_0000),
            mov32i(5, 0),
            txd_word(0xde38_0000_0000_0000, 2, 0, 4, 5, 1, false),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.mnemonics.iter().any(|(m, _)| *m == "txd"));
    assert_eq!(shader.textures.len(), 1);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("Grad"), "{text}");
    validate(&shader.spirv);

    // Per-texel offsets share the packed register with the array layer and
    // start at bit 16.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov32i(2, 0x0001_0000),
            mov32i(3, 0),
            mov32i(4, 0x3c00_0000),
            mov32i(5, 0),
            txd_word(0xde38_0000_0000_0000, 2, 0, 4, 5, 1, false) | 1 << 35,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageQuerySizeLod"), "{text}");
    validate(&shader.spirv);

    // Bindless: the handle is the first source (ra), then coordinates and
    // the packed array/offset register.
    let code = program(
        fragment_header(1),
        &[
            mov_cbuf(4, 4, 0xb0),
            mov32i(5, 0x3c00_0000),
            mov32i(6, 0x3c00_0000),
            mov32i(7, 0),
            mov32i(0, 0x3c00_0000),
            mov32i(1, 0),
            mov32i(2, 0x3c00_0000),
            mov32i(3, 0),
            txd_word(0xde78_0000_0000_0000, 2, 4, 0, 0, 1, false),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(
        (shader.textures[0].slot, shader.textures[0].handle),
        (Some(4), 0xb0 / 4)
    );
    validate(&shader.spirv);

    // Arrayed bindless: the layer is packed in the low 16 bits of the
    // register after the coordinates.
    let code = program(
        fragment_header(1),
        &[
            mov_cbuf(4, 4, 0xb0),
            mov32i(5, 0x3c00_0000),
            mov32i(6, 0x3c00_0000),
            mov32i(7, 1),
            mov32i(0, 0x3c00_0000),
            mov32i(1, 0),
            mov32i(2, 0x3c00_0000),
            mov32i(3, 0),
            txd_word(0xde78_0000_0000_0000, 2, 4, 0, 0, 1, true),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.textures[0].array);
    validate(&shader.spirv);

    // The bindless form needs the handle in a constant-buffer word.
    let code = program(
        fragment_header(1),
        &[
            mov32i(4, 3),
            mov32i(5, 0),
            mov32i(6, 0),
            mov32i(7, 0),
            mov32i(0, 0),
            mov32i(1, 0),
            mov32i(2, 0),
            mov32i(3, 0),
            txd_word(0xde78_0000_0000_0000, 2, 4, 0, 0, 1, false),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["txd (bindless)"]);
    assert!(shader.textures.is_empty());
    validate(&shader.spirv);
}

#[test]
fn integer_multiply_and_conversion_mnemonics() {
    let code = program(
        fragment_header(1),
        &[
            mov32i(0, 7),
            mov32i(1, 3),
            mov32i(2, 5),
            imad_word(3, 0, 1, 2, false, false, false),
            imad_word(4, 0, 1, 2, true, true, false),
            imad_word(5, 0, 1, 2, false, true, false),
            imad_word(6, 0, 1, 2, true, false, true),
            imul_word(7, 0, 1, false, false, false),
            imul_word(8, 0, 1, true, true, true),
            imad32i_word(9, 0, 3, false, false, false),
            imad32i_word(10, 0, 3, true, true, true),
            imul32i_word(11, 0, 3, false, false, false),
            imul32i_word(12, 0, 3, true, true, true),
            0x50d0_0000_0000_0000 | PT | 13,
            0xef60_0000_0000_0000 | PT,
            0xef80_0000_0000_0000 | PT,
            0xebf0_0000_0000_0000 | PT,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("Int64"), "{text}");
    assert!(text.contains("OpShiftRightLogical"), "{text}");
    let names: Vec<_> = shader.mnemonics.iter().map(|(m, _)| *m).collect();
    for m in [
        "imad", "imul", "imad32i", "imul32i", "lepc", "cctl", "cctll", "cctlt",
    ] {
        assert!(names.contains(&m), "{m} missing from {names:?}");
    }
    validate(&shader.spirv);

    // Sub-forms that would change the result must stay explicit.
    for (imad, label) in [
        (
            imad_word(3, 0, 1, 2, false, false, false) | 1 << 49,
            "imad (extended)",
        ),
        (
            imad_word(3, 0, 1, 2, false, false, false) | 1 << 50,
            "imad (saturate)",
        ),
    ] {
        let code = program(
            fragment_header(1),
            &[mov32i(0, 7), mov32i(1, 3), mov32i(2, 5), imad, EXIT],
        );
        let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
        assert_eq!(shader.unsupported, [label]);
        validate(&shader.spirv);
    }
}

#[test]
fn cset_writes_one_or_all_ones() {
    for (bf, value) in [(false, u32::MAX), (true, 1.0f32.to_bits())] {
        let code = program(
            fragment_header(1),
            &[
                mov32i(0, 1),
                mov32i(1, 2),
                isetp_lt(7, 0, 1),
                0x5098_0000_0000_0000 | (bf as u64) << 44 | 7 << 39 | 1 << 8 | PT | 3,
                mov32i(4, value),
                EXIT,
            ],
        );
        let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
        assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
        assert!(shader.mnemonics.iter().any(|(m, _)| *m == "cset"));
        validate(&shader.spirv);
    }
}

fn mov_reg(rd: u64, rs: u64) -> u64 {
    0x5c98_0000_0000_0000 | rs << 20 | PT | rd
}

fn ldc_word(rd: u64, slot: u64, offset: u64) -> u64 {
    0xef90_0000_0000_0000 | 4 << 48 | slot << 36 | offset << 20 | PT | 0xff << 8 | rd
}

fn tld4_bindless(rd: u64, ra: u64, rb: u64, mask: u64, component: u64, offset: u64) -> u64 {
    0xdef8_0000_0000_0000
        | component << 38
        | offset << 36
        | mask << 31
        | 1 << 29
        | PT
        | rb << 20
        | ra << 8
        | rd
}

#[test]
fn bindless_handle_through_a_register_copy() {
    // Jamboree copies the handle word out of `c[0x0][0x7b8]` through r34
    // before the sample; the descriptor is still that constant-buffer word.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov_cbuf(5, 0, 0x7b8),
            mov_reg(7, 5),
            tex_bindless(2, 0, 7, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 1);
    let t = shader.textures[0];
    assert_eq!((t.slot, t.handle, t.sampler), (Some(0), 0x7b8 / 4, None));
    validate(&shader.spirv);
}

#[test]
fn bindless_handle_through_a_constant_load() {
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            ldc_word(8, 3, 0x2c0),
            tex_bindless(2, 0, 8, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let t = shader.textures[0];
    assert_eq!((t.slot, t.handle, t.sampler), (Some(3), 0x2c0 / 4, None));
    validate(&shader.spirv);

    // An indexed constant load has no static location and stays labelled.
    let indexed = 0xef90_0000_0000_0000 | 4 << 48 | 3 << 36 | 0x40 << 20 | PT | 1 << 8 | 8;
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            indexed,
            tex_bindless(2, 0, 8, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tex (bindless)"]);
    assert!(shader.textures.is_empty());
    validate(&shader.spirv);
}

#[test]
fn bindless_tld4_reads_the_handle_from_the_constant_bank() {
    // The register form of `tld4` (0xdef8) keeps the texture index in `rb`;
    // the component moves to bits 38-39. This is the gather at 0x00b0 of
    // Jamboree's shader 0x...0230, minus its per-texel offset sibling.
    let word = tld4_bindless(3, 14, 0, 0x2, 0, 0);
    let code = program(
        fragment_header(1),
        &[
            ipa(14, 0x80, 0xff),
            ipa(15, 0x84, 0xff),
            mov_cbuf(0, 0, 0x7b8),
            word,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 1);
    let t = shader.textures[0];
    assert_eq!((t.slot, t.handle), (Some(0), 0x7b8 / 4));
    assert_eq!((t.dim, t.array, t.shadow), (TexDim::D2, false, false));
    assert!(shader.mnemonics.iter().any(|(m, _)| *m == "tld4"));
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageGather"), "{text}");
    validate(&shader.spirv);
}

#[test]
fn bindless_tld4_subforms_that_are_not_expressed_stay_labelled() {
    // Per-texel offsets (PTP) cannot be one `OpImageGather`.
    let ptp = tld4_bindless(4, 14, 0, 0xf, 0, 2);
    let code = program(
        fragment_header(1),
        &[
            ipa(14, 0x80, 0xff),
            ipa(15, 0x84, 0xff),
            mov_cbuf(0, 0, 0x7b8),
            ptp,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tld4 (per-texel offsets)"]);
    validate(&shader.spirv);
    assert!(shader.textures.is_empty());

    // A handle that is not a constant-buffer word is not guessed.
    let code = program(
        fragment_header(1),
        &[
            ipa(14, 0x80, 0xff),
            ipa(15, 0x84, 0xff),
            mov32i(9, 3),
            tld4_bindless(4, 14, 9, 0xf, 0, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tld4 (bindless)"]);
    assert!(shader.textures.is_empty());
    validate(&shader.spirv);
}

fn s2r_word(rd: u64, sr: u64) -> u64 {
    0xf0c8_0000_0000_0000 | sr << 20 | PT | rd
}

#[test]
fn the_labels_jamboree_reported_disappear_for_the_translated_forms() {
    // The words the coverage run reported, in the order and shape the game
    // uses them: a handle copied through a register, a bindless gather with
    // the handle at c[0x0][0x7b8], and SR_WSCALEFACTOR_XY/Z.
    let code = program(
        fragment_header(1),
        &[
            ipa(16, 0x80, 0xff),
            ipa(17, 0x84, 0xff),
            mov_cbuf(34, 4, 0x3e0),
            mov_reg(26, 34),
            tex_bindless(27, 16, 26, 0xf, 0),
            mov_cbuf(0, 0, 0x7b8),
            tld4_bindless(3, 16, 0, 0x2, 0, 0),
            s2r_word(5, 0x1e),
            s2r_word(6, 0x1f),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 2);
    assert_eq!(
        (shader.textures[0].slot, shader.textures[0].handle),
        (Some(4), 0x3e0 / 4)
    );
    assert_eq!(
        (shader.textures[1].slot, shader.textures[1].handle),
        (Some(0), 0x7b8 / 4)
    );
    for m in ["tex", "tld4", "mov"] {
        assert!(
            shader.mnemonics.iter().any(|(name, _)| *name == m),
            "{m} missing"
        );
    }
    validate(&shader.spirv);

    // The forms that cannot be expressed exactly keep their labels: the
    // per-texel gather, the vote-to-global form and the conditional stop-mask
    // condition code.
    let ptp = program(
        fragment_header(1),
        &[
            ipa(16, 0x80, 0xff),
            ipa(17, 0x84, 0xff),
            mov_cbuf(0, 0, 0x7b8),
            tld4_bindless(4, 16, 0, 0xf, 0, 2),
            EXIT,
        ],
    );
    let shader = translate(&ptp, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tld4 (per-texel offsets)"]);
    validate(&shader.spirv);

    let vote = program(
        fragment_header(1),
        &[0x50e2_4321_1117_0000, 0xe300_0000_0007_001c, EXIT],
    );
    let shader = translate(&vote, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    validate(&shader.spirv);
    assert_eq!(
        shader.unsupported,
        ["conditional stop mask", "vote (vtg)"],
        "the vote and its stop-mask exit stay reported"
    );
}

fn lop_cbuf(rd: u64, ra: u64, slot: u64, offset: u64, op: u64) -> u64 {
    0x4c40_0000_0000_0000
        | 7 << 48
        | op << 41
        | slot << 34
        | (offset >> 2) << 20
        | PT
        | ra << 8
        | rd
}

fn ldc_indexed(rd: u64, ra: u64, slot: u64, offset: u64) -> u64 {
    0xef90_0000_0000_0000 | 4 << 48 | slot << 36 | offset << 20 | PT | ra << 8 | rd
}

#[test]
fn bindless_handle_from_an_indexed_constant_load() {
    // An indexed constant load whose index is itself a constant keeps a
    // static address: byte 0x40 + 0x10 is word 0x14 of c[3].
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov32i(7, 0x10),
            ldc_indexed(8, 7, 3, 0x40),
            tex_bindless(2, 0, 8, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(
        (shader.textures[0].slot, shader.textures[0].handle),
        (Some(3), (0x40 + 0x10) / 4)
    );
    validate(&shader.spirv);

    // The same load with a per-draw index keeps no static address.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            ipa(7, 0x88, 0xff),
            ldc_indexed(8, 7, 3, 0x40),
            tex_bindless(2, 0, 8, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tex (bindless)"]);
    assert!(shader.textures.is_empty());
    validate(&shader.spirv);
}

#[test]
fn tex_pairs_nvn_texture_and_sampler_words_into_one_binding() {
    // BD builds the handle from NVN's 128-slot layout: the texture index in
    // one word of the driver's texture bank and the sampler descriptor in
    // another, combined with an OR. The binding names both words, so the
    // runtime takes the texture id from one and the sampler id from the other.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov_cbuf(6, 2, 0x170),
            lop_cbuf(6, 6, 2, 0x568, 1),
            tex_bindless(2, 0, 6, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 1);
    let t = shader.textures[0];
    assert_eq!(
        (t.slot, t.handle, t.sampler),
        (Some(2), 0x170 / 4, Some(0x568 / 4))
    );
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageSampleImplicitLod"), "{text}");
    validate(&shader.spirv);

    // The same two words reached through indexed constant loads whose index is
    // a constant keep a static binding; the sampler table comes first here, so
    // the pair is recognised in either order.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov32i(7, 0),
            ldc_indexed(8, 7, 2, 0x568),
            ldc_indexed(9, 7, 2, 0x168),
            lop_cbuf(10, 8, 2, 0x168, 1),
            tex_bindless(2, 0, 10, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let t = shader.textures[0];
    assert_eq!(
        (t.slot, t.handle, t.sampler),
        (Some(2), 0x168 / 4, Some(0x568 / 4))
    );
    validate(&shader.spirv);
}

#[test]
fn tex_handle_shapes_that_are_not_the_separate_layout_stay_labelled() {
    // The other side of the OR changing per draw cannot be a binding.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            ipa(6, 0x88, 0xff),
            lop_cbuf(7, 6, 2, 0x568, 1),
            tex_bindless(2, 0, 7, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tex (bindless dynamic handle)"]);
    assert!(shader.textures.is_empty());
    assert!(!disassemble(&shader.spirv).contains("OpImageSample"));
    validate(&shader.spirv);

    // A pair that is not one texture and one sampler table entry is not the
    // layout either, and it emits no sample.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov_cbuf(6, 2, 0x170),
            lop_cbuf(6, 6, 2, 0x178, 1),
            tex_bindless(2, 0, 6, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tex (bindless handle or)"]);
    assert!(shader.textures.is_empty());
    assert!(!disassemble(&shader.spirv).contains("OpImageSample"));
    validate(&shader.spirv);

    // The pair in a bank that is not the driver's texture bank is not the
    // layout NVN documents.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov_cbuf(6, 4, 0x170),
            lop_cbuf(6, 6, 4, 0x568, 1),
            tex_bindless(2, 0, 6, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tex (bindless handle or)"]);
    assert!(shader.textures.is_empty());
    validate(&shader.spirv);

    // A per-draw index into the texture table has no static word either.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            ipa(7, 0x88, 0xff),
            ldc_indexed(8, 7, 2, 0x168),
            lop_cbuf(10, 8, 2, 0x568, 1),
            tex_bindless(2, 0, 10, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tex (bindless dynamic handle)"]);
    assert!(shader.textures.is_empty());
    validate(&shader.spirv);

    // The register form of `lop` (0x5c40) is not the driver's handle build.
    let reg_lop = 0x5c40_0000_0000_0000 | 7 << 48 | 1 << 41 | 2 << 20 | PT | 2 << 8 | 6;
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov_cbuf(2, 2, 0x170),
            reg_lop,
            tex_bindless(2, 0, 6, 0xf, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["tex (bindless)"]);
    assert!(shader.textures.is_empty());
    validate(&shader.spirv);
}

#[test]
fn register_lod_bias_with_a_depth_compare_stays_valid() {
    // A depth comparison cannot carry a bias in SPIR-V, so the level stays
    // implicit. The bias register comes before the depth one.
    let code = program(
        fragment_header(1),
        &[
            ipa(0, 0x80, 0xff),
            ipa(1, 0x84, 0xff),
            mov_cbuf(5, 4, 0xb0),
            mov32i(6, 0.5f32.to_bits()),
            mov32i(7, 0.5f32.to_bits()),
            tex_bindless(2, 0, 5, 0xf, 2) | 1 << 50,
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.textures[0].shadow);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageSampleDrefImplicitLod"), "{text}");
    assert!(!text.contains("Bias"), "no bias with a dref sample: {text}");
    validate(&shader.spirv);
}
