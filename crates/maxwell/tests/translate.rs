use kirikae_maxwell::{Options, Stage, translate};

const PT: u64 = 7 << 16;

fn fragment_header(targets: u32) -> Vec<u8> {
    let mut words = [0u32; 20];
    words[0] = 2 | 5 << 10;
    for t in 0..targets {
        words[18] |= 0xf << (t * 4);
    }
    words[6] = 0b1010;
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn vertex_header() -> Vec<u8> {
    let mut words = [0u32; 20];
    words[0] = 1 | 1 << 10;
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

const NOP: u64 = 0x50b0_0000_0000_0000 | PT | (0xf << 8);
const EXIT: u64 = 0xe300_0000_0000_0000 | PT | 0xf;
const KIL: u64 = 0xe330_0000_0000_0000 | PT | 0xf;

fn mov32i(rd: u64, imm: u32) -> u64 {
    0x0100_0000_0000_0000 | (imm as u64) << 20 | PT | rd
}
fn fadd(rd: u64, ra: u64, rb: u64) -> u64 {
    0x5c58_0000_0000_0000 | rb << 20 | PT | ra << 8 | rd
}
fn ffma(rd: u64, ra: u64, rb: u64, rc: u64) -> u64 {
    0x5980_0000_0000_0000 | rc << 39 | rb << 20 | PT | ra << 8 | rd
}
fn ipa(rd: u64, attr: u64, rb: u64) -> u64 {
    0xe000_0000_0000_ff00 | 1 << 54 | attr << 28 | 0xff << 39 | rb << 20 | PT | rd & 0xff
}
fn isetp_lt(p: u64, ra: u64, rb: u64) -> u64 {
    0x5b60_0000_0000_0000 | 1 << 49 | 1 << 48 | 7 << 39 | p << 3 | 7 | rb << 20 | PT | ra << 8
}
fn bra(offset: i32, pred: u64) -> u64 {
    0xe240_0000_0000_0000 | ((offset as u32 as u64) & 0xff_ffff) << 20 | pred << 16 | 0xf
}
fn iadd32i(rd: u64, ra: u64, imm: u32) -> u64 {
    0x1c00_0000_0000_0000 | (imm as u64) << 20 | PT | ra << 8 | rd
}
fn texs_2d(rd: u64, ra: u64, rb: u64, handle: u64) -> u64 {
    0xd800_0000_0000_0000 | 1 << 53 | 4 << 50 | handle << 36 | (rd + 2) << 28 | rb << 20 | PT | ra << 8 | rd
}
fn ald(rd: u64, attr: u64, words: u64) -> u64 {
    0xefd8_0000_0000_0000 | (words - 1) << 47 | 0xff << 39 | attr << 20 | PT | 0xff << 8 | rd
}
fn ast(rs: u64, attr: u64, words: u64) -> u64 {
    0xeff0_0000_0000_0000 | (words - 1) << 47 | 0xff << 39 | attr << 20 | PT | 0xff << 8 | rs
}
fn ldc(rd: u64, slot: u64, offset: u64) -> u64 {
    0xef90_0000_0000_0000 | 4 << 48 | slot << 36 | offset << 20 | PT | 0xff << 8 | rd
}

fn options(stage: Stage) -> Options {
    Options { local_size: [8, 8, 1], ..Options::new(stage, 0) }
}

fn disassemble(spirv: &[u32]) -> String {
    use rspirv::binary::Disassemble;
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(spirv, &mut loader).unwrap();
    loader.module().disassemble()
}

fn validate(spirv: &[u32]) {
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    let module = match naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default()) {
        Ok(m) => m,
        Err(e) => panic!("SPIR-V does not parse: {e:?}
{}", disassemble(spirv)),
    };
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).expect("SPIR-V validates");
    variables_first(spirv);
}

fn variables_first(spirv: &[u32]) {
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(spirv, &mut loader).unwrap();
    let module = loader.module();
    let is_var = |i: &rspirv::dr::Instruction| i.class.opcode == rspirv::spirv::Op::Variable;
    for f in &module.functions {
        for (k, block) in f.blocks.iter().enumerate() {
            let leading = block.instructions.iter().take_while(|i| is_var(i)).count();
            let late = block.instructions.iter().skip(if k == 0 { leading } else { 0 }).any(is_var);
            assert!(!late, "OpVariable after other instructions in block {k}\n{}", disassemble(spirv));
        }
    }
}

#[test]
fn straight_line_fragment_shader() {
    let code = program(
        fragment_header(1),
        &[mov32i(0, 1.0f32.to_bits()), mov32i(1, 0.5f32.to_bits()), fadd(2, 0, 1), ffma(3, 0, 1, 2), ipa(4, 0x80, 0xff), EXIT],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.inputs, 1);
    validate(&shader.spirv);
}

#[test]
fn kil_leaves_the_dispatcher() {
    for kil in [KIL, KIL & !PT] {
        let code = program(fragment_header(1), &[isetp_lt(0, 0, 1), bra(0x8, 0), kil, EXIT]);
        let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
        let text = disassemble(&shader.spirv);
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        let at = lines.iter().position(|l| l.starts_with("OpKill")).expect("OpKill");
        assert!(lines[at - 1].starts_with("OpStore"), "{text}");
    }
}

#[test]
fn loop_with_branches() {
    let code = program(
        fragment_header(1),
        &[
            mov32i(0, 0),
            mov32i(1, 10),
            iadd32i(0, 0, 1),
            isetp_lt(0, 0, 1),
            bra(-0x20, 0),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    validate(&shader.spirv);
    assert!(disassemble(&shader.spirv).contains("OpLoopMerge"), "a backward branch needs the dispatcher loop");
}

#[test]
fn forward_branches_need_no_loop() {
    let code = program(fragment_header(1), &[mov32i(1, 10), isetp_lt(0, 0, 1), bra(0x10, 0), mov32i(2, 1), EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    validate(&shader.spirv);
    let text = disassemble(&shader.spirv);
    assert!(!text.contains("OpLoopMerge") && text.contains("OpBranchConditional"), "{text}");
}

#[test]
fn discard_keeps_a_shader_straight_line() {
    const KIL_P0: u64 = 0xe330_0000_0000_000f;
    let code = program(fragment_header(1), &[mov32i(1, 10), isetp_lt(0, 0, 1), KIL_P0, mov32i(2, 0), EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    validate(&shader.spirv);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpKill") && !text.contains("OpLoopMerge"), "{text}");
}

#[test]
fn texture_and_constant_buffer() {
    let code = program(fragment_header(1), &[ipa(0, 0x80, 0xff), ipa(1, 0x84, 0xff), texs_2d(4, 0, 1, 3), ldc(8, 1, 0x10), EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 1);
    assert_eq!(shader.textures[0].handle, 3);
    assert_eq!(shader.cbufs, 1 << 1);
    validate(&shader.spirv);
}

#[test]
fn point_sprite_coordinates() {
    let code = program(fragment_header(1), &[ipa(0, 0x2e0, 0xff), ipa(1, 0x2e4, 0xff), EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("BuiltIn PointCoord"), "{text}");
    validate(&shader.spirv);
}

fn tmml_2d_array(rd: u64, ra: u64, handle: u64) -> u64 {
    0xdf58_0000_0000_0000 | handle << 36 | 0b11 << 31 | 1 << 29 | 1 << 28 | PT | ra << 8 | rd
}

#[test]
fn texture_lod_query() {
    let code = program(fragment_header(1), &[mov32i(0, 0), ipa(1, 0x80, 0xff), ipa(2, 0x84, 0xff), tmml_2d_array(4, 0, 5), EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 1);
    assert_eq!(shader.textures[0].handle, 5);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageQueryLod"), "{text}");
    let code = program(vertex_header(), &[mov32i(0, 0), mov32i(1, 0), mov32i(2, 0), tmml_2d_array(4, 0, 5), EXIT]);
    let shader = translate(&code, &options(Stage::Vertex), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.textures.is_empty());
    validate(&shader.spirv);
}

#[test]
fn texture_gathers() {
    let tld4s = 0xdf00_00a0_00a7_0b06;
    let tld4 = 0xc838_0000_0000_0000u64 | 1 << 56 | 1 << 54 | 1 << 50 | 7 << 36 | 0xf << 31 | 1 << 29 | PT | 2 << 20 | 4;
    let code = program(fragment_header(1), &[ipa(11, 0x80, 0xff), ipa(10, 0x84, 0xff), tld4s, EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.textures.len(), 1);
    validate(&shader.spirv);
    let code = program(fragment_header(1), &[mov32i(0, 0), mov32i(1, 0), mov32i(2, 0x11), mov32i(3, 0), tld4, EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.textures.iter().any(|t| t.handle == 7 && t.shadow));
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageDrefGather") && text.contains("Offset"), "{text}");
}

#[test]
fn texture_gather_per_texel_offsets() {
    let tld4 = 0xc838_0000_0000_0000u64 | 2 << 54 | 7 << 36 | 0xf << 31 | 1 << 29 | PT | 2 << 20 | 4;
    let code = program(fragment_header(1), &[mov32i(0, 0), mov32i(1, 0), mov32i(2, 0x01ff_0100), mov32i(3, 0xff01_0000), tld4, EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert_eq!(text.matches("OpImageGather").count(), 4, "{text}");
    assert_eq!(text.matches("Offset %").count(), 4, "{text}");
}

#[test]
fn shared_memory_atomics_return_the_old_value() {
    let atoms = |op: u64, size: u64| 0xec00_0000_0000_0000u64 | op << 52 | 4 << 30 | size << 28 | 2 << 20 | PT | 5;
    let translated = |word: u64| {
        let code = program(vec![0; kirikae_maxwell::sph::SIZE], &[mov32i(0, 0), mov32i(2, 1), word, EXIT]);
        translate(&code, &Options { shared_memory: 0x100, ..options(Stage::Compute) }, &kirikae_maxwell::NoEnv).unwrap()
    };
    for (op, size, spirv) in [(0, 0, "OpAtomicIAdd"), (1, 1, "OpAtomicSMin"), (2, 0, "OpAtomicUMax"), (6, 0, "OpAtomicOr"), (8, 0, "OpAtomicExchange")] {
        let shader = translated(atoms(op, size));
        assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
        let text = disassemble(&shader.spirv);
        assert!(text.contains(spirv) && text.contains("Workgroup"), "{spirv}: {text}");
        validate(&shader.spirv);
    }
    assert_eq!(translated(atoms(0, 2)).unsupported, ["atoms (64-bit)"]);
    assert_eq!(translated(atoms(3, 0)).unsupported, ["atoms (inc/dec)"]);
}

#[test]
fn compute_shared_memory() {
    let sts = 0xef58_0000_0000_0000u64 | 5 << 48 | 0x10 << 20 | PT | 2;
    let bar = 0xf0a8_1000_0000_0000u64 | PT;
    let lds = 0xef48_0000_0000_0000u64 | 6 << 48 | PT | 4;
    let code = program(vec![0; kirikae_maxwell::sph::SIZE], &[mov32i(0, 0), mov32i(2, 1), mov32i(3, 2), sts, bar, lds, EXIT]);
    let opts = Options { shared_memory: 0x100, ..options(Stage::Compute) };
    let shader = translate(&code, &opts, &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("Workgroup") && text.contains("OpControlBarrier"), "{text}");
    validate(&shader.spirv);
}

#[test]
fn vertex_shader_passthrough() {
    let code = program(vertex_header(), &[ald(0, 0x80, 4), ast(0, 0x70, 4), ald(4, 0x90, 4), ast(4, 0x80, 4), EXIT]);
    let shader = translate(&code, &options(Stage::Vertex), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!((shader.inputs, shader.outputs), (0b11, 0b1));
    validate(&shader.spirv);
}

#[test]
fn dumped_shaders() {
    let Some(dir) = std::env::var_os("KIRIKAE_SHADER_DIR") else { return };
    let mut missing = std::collections::BTreeMap::<&str, usize>::new();
    let (mut total, mut clean) = (0, 0);
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.ends_with(".bin") || name.starts_with("macro") {
            continue;
        }
        let stage = if name.contains("fragment") {
            Stage::Fragment
        } else if name.contains("compute") {
            Stage::Compute
        } else {
            Stage::Vertex
        };
        let code = std::fs::read(&path).unwrap();
        let shader = translate(&code, &options(stage), &kirikae_maxwell::NoEnv).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let bytes: Vec<u8> = shader.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        let module = naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default());
        let valid = module.as_ref().ok().map(|m| naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(m).map(|_| ()).map_err(|e| format!("{e:?}")));
        match (&module, &valid) {
            (Err(naga::front::spv::Error::UnsupportedCapability(_) | naga::front::spv::Error::UnsupportedInstruction(..)), _) => {}
            (Err(e), _) => println!("{name}: parse error {e:?}"),
            (_, Some(Err(e))) => println!("{name}: invalid {e:?}"),
            _ => {}
        }
        let program = kirikae_maxwell::cfg::scan(&code[kirikae_maxwell::sph::SIZE..]);
        for (pc, w) in &program.unknown {
            println!("{name}: unknown word {w:#018x} at {pc:#x}");
        }
        total += 1;
        let checked_ok = matches!(valid, Some(Ok(_))) || matches!(module, Err(naga::front::spv::Error::UnsupportedCapability(_) | naga::front::spv::Error::UnsupportedInstruction(..)));
        if shader.unsupported.is_empty() && checked_ok {
            clean += 1;
        }
        if !shader.unsupported.is_empty() {
            println!("{name}: missing {:?}", shader.unsupported);
        }
        for u in shader.unsupported {
            *missing.entry(u).or_default() += 1;
        }
    }
    println!("{clean}/{total} shaders translate completely; missing: {missing:?}");
}

#[test]
fn half_pair_operands() {
    use kirikae_maxwell::decode::{HalfOperand, HalfSrc, Insn};
    let reg = |reg, swizzle, neg| HalfOperand { src: HalfSrc::Reg { reg, swizzle }, neg, abs: false };
    let cbuf = |slot, offset, neg| HalfOperand { src: HalfSrc::Cbuf { slot, offset }, neg, abs: false };
    let ops = |w: u64| Insn::decode(w).unwrap().half_operands();

    let o = ops(0x61a2_8484_0007_090f);
    assert_eq!((o.a, o.b, o.c), (reg(9, 1, false), reg(9, 1, true), Some(cbuf(1, 0, false))));
    assert_eq!((o.mode, o.sat), (1, false));
    let o = ops(0x71c2_8b04_0047_190e);
    assert_eq!((o.a, o.b, o.c), (reg(25, 1, false), cbuf(1, 0x10, true), Some(reg(22, 2, false))));
    let o = ops(0x5d06_8b10_90b7_0601);
    assert_eq!((o.a, o.b, o.c), (reg(6, 1, false), reg(11, 1, true), Some(reg(22, 2, false))));
    assert_eq!(o.mode, 3);
    let o = ops(0x7a85_0804_0047_ff16);
    assert_eq!((o.a, o.b, o.c), (reg(255, 2, true), cbuf(1, 0x10, false), None));
    let o = ops(0x5d0a_8000_11c7_0909);
    assert_eq!((o.a, o.b, o.c, o.mode), (reg(9, 1, false), reg(28, 1, false), None, 1));

    let code = program(fragment_header(1), &[0x61a2_8484_0007_090f, 0x71c2_8b04_0047_190e, 0x5d06_8b10_90b7_0601, 0x7a85_0804_0047_ff16, EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    validate(&shader.spirv);
}

#[test]
fn a_128_bit_constant_load_fills_four_registers() {
    let sized = |size: u64| ldc(4, 2, 0x10) & !(7 << 48) | size << 48;
    let stores = |size: u64| {
        let code = program(fragment_header(1), &[sized(size), EXIT]);
        let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
        assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
        validate(&shader.spirv);
        disassemble(&shader.spirv).matches("OpStore").count()
    };
    assert_eq!(stores(5), stores(4) + 1);
    assert_eq!(stores(6), stores(4) + 3);
}
