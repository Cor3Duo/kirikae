//! Integer misc forms, the integer video ALU and the control-flow forms the
//! dispatcher can model (call/return and continue stacks).
//!
//! Coverage of this file, one test per form:
//!
//! * integer misc: `prmt` (generic, `f4e`, `b4e`, `rc8`, `ecl`, `ecr`,
//!   `rc16`), `idp` (`dp4a`, all signedness mixes), `iscadd32i`, `cs2r`,
//!   `getcrsptr`/`setcrsptr`, `getlmembase`/`setlmembase`.
//! * video: `vadd` (merge/accumulate/min/max write-backs, `avg`, `sat`),
//!   `vmad` (`>> 7`/`>> 15`, `sat`), `vabsdiff`, `vsetp`.
//! * half pairs with a 32-bit immediate: `hadd2_32i`, `hmul2_32i`,
//!   `hfma2_32i`.
//! * control: `pcnt`/`cont`, `cal`/`jcal`/`ret`, `jmp`.
//!
//! Left out on purpose and reported by name: `b2r`, `r2b`, `jmx`, `pexit`,
//! `pret`, `longjmp`, `plongjmp`, `bpt`, `stp`, `imadsp`, `sam`, `ram`,
//! `rtt`, `out`, `vset`, `vshl`, `vshr`, `vabsdiff4`, `hset2` and the FP64
//! forms (`dadd`, `dmul`, `dfma`, `dmnmx`, `dset`, `dsetp`, `fchk`).

mod common;

use kirikae_maxwell::{NoEnv, Options, Stage, translate};

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

/// Address of the `n`-th instruction of a program built by [`program`].
fn pc(n: usize) -> u64 {
    (8 + 32 * (n / 3) + 8 * (n % 3)) as u64
}

fn mov32i(rd: u64, imm: u32) -> u64 {
    0x0100_0000_0000_0000 | (imm as u64) << 20 | PT | rd
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

fn fragment(insns: &[u64]) -> Vec<u8> {
    program(fragment_header(1), insns)
}

/// Translates a fragment program and checks that every instruction was
/// decoded and translated.
fn translated(insns: &[u64], mnemonics: &[&str]) -> kirikae_maxwell::Shader {
    let shader = translate(
        &fragment(insns),
        &options(Stage::Fragment),
        &kirikae_maxwell::NoEnv,
    )
    .expect("translation");
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    for m in mnemonics {
        assert!(
            shader.mnemonics.iter().any(|(name, _)| name == m),
            "{m} not decoded: {:?}",
            shader.mnemonics
        );
    }
    validate(&shader.spirv);
    shader
}

/// Translates a program expected to leave exactly one precise label.
fn rejected(insns: &[u64], label: &str) {
    let shader = translate(
        &fragment(insns),
        &options(Stage::Fragment),
        &kirikae_maxwell::NoEnv,
    )
    .expect("translation");
    assert_eq!(shader.unsupported, [label]);
    validate(&shader.spirv);
}

fn prmt(mode: u64, rd: u64, ra: u64, rb: u64, rc: u64) -> u64 {
    0x5bc0_0000_0000_0000 | mode << 48 | rc << 39 | rb << 20 | PT | ra << 8 | rd
}

fn prmt_imm(mode: u64, rd: u64, ra: u64, imm: u64, rc: u64) -> u64 {
    0x36c0_0000_0000_0000 | mode << 48 | rc << 39 | imm << 20 | PT | ra << 8 | rd
}

fn idp(rd: u64, ra: u64, rb: u64, rc: u64, a_signed: bool, b_signed: bool) -> u64 {
    0x53f8_0000_0000_0000
        | (a_signed as u64) << 49
        | (b_signed as u64) << 47
        | rc << 39
        | rb << 20
        | PT
        | ra << 8
        | rd
}

fn iscadd32i(rd: u64, ra: u64, imm: u32, shift: u64) -> u64 {
    0x1400_0000_0000_0000 | shift << 53 | (imm as u64) << 20 | PT | ra << 8 | rd
}

fn cs2r(rd: u64, sr: u64) -> u64 {
    0x50c8_0000_0000_0000 | sr << 20 | PT | rd
}

#[allow(clippy::too_many_arguments)]
fn vadd(
    rd: u64,
    ra: u64,
    rb: u64,
    rc: u64,
    a_sel: u64,
    b_sel: u64,
    avg: u64,
    sat: bool,
    op: u64,
) -> u64 {
    0x2000_0000_0000_0000
        | avg << 56
        | (sat as u64) << 55
        | op << 51
        | 1 << 50
        | (b_sel & 0x8) << 46
        | (a_sel & 0x8) << 45
        | a_sel << 36
        | b_sel << 28
        | rc << 39
        | rb << 20
        | PT
        | ra << 8
        | rd
}

#[allow(clippy::too_many_arguments)]
fn vmad(
    rd: u64,
    ra: u64,
    rb: u64,
    rc: u64,
    a_sel: u64,
    b_sel: u64,
    avg: u64,
    scale: u64,
    sat: bool,
) -> u64 {
    0x5f00_0000_0000_0000
        | (sat as u64) << 55
        | avg << 53
        | scale << 51
        | 1 << 50
        | (b_sel & 0x8) << 46
        | (a_sel & 0x8) << 45
        | a_sel << 36
        | b_sel << 28
        | rc << 39
        | rb << 20
        | PT
        | ra << 8
        | rd
}

fn vabsdiff(rd: u64, ra: u64, rb: u64, rc: u64, a_sel: u64, b_sel: u64, sat: bool) -> u64 {
    0x5400_0000_0000_0000
        | (sat as u64) << 55
        | 1 << 50
        | (b_sel & 0x8) << 46
        | (a_sel & 0x8) << 45
        | a_sel << 36
        | b_sel << 28
        | rc << 39
        | rb << 20
        | PT
        | ra << 8
        | rd
}

fn vsetp(ra: u64, rb: u64, cmp: u64, p: u64, pd: u64) -> u64 {
    0x50f0_0000_0000_0000
        | (cmp & 0x4) << 45
        | (cmp & 0x3) << 43
        | 1 << 50
        | p << 39
        | pd << 3
        | PT
        | rb << 20
        | ra << 8
        | pd
}

fn hadd2_32i(rd: u64, ra: u64, imm: u32, swizzle: u64, neg_a: bool, sat: bool) -> u64 {
    0x2c00_0000_0000_0000
        | swizzle << 53
        | (sat as u64) << 52
        | (neg_a as u64) << 56
        | (imm as u64) << 20
        | PT
        | ra << 8
        | rd
}

fn hmul2_32i(rd: u64, ra: u64, imm: u32, swizzle: u64, sat: bool) -> u64 {
    0x2a00_0000_0000_0000
        | swizzle << 53
        | (sat as u64) << 52
        | (imm as u64) << 20
        | PT
        | ra << 8
        | rd
}

fn hfma2_32i(rd: u64, ra: u64, imm: u32, swizzle: u64, neg_c: bool) -> u64 {
    0x2800_0000_0000_0000
        | swizzle << 47
        | (neg_c as u64) << 51
        | (imm as u64) << 20
        | PT
        | ra << 8
        | rd
}

fn cal(target: u64, at: u64) -> u64 {
    let imm = (target as i64 - at as i64 - 8) as u64 & 0xff_ffff;
    0xe260_0000_0000_0000 | imm << 20
}

fn jcal(target: u64) -> u64 {
    0xe220_0000_0000_0000 | target << 20
}

fn jmp(target: u64) -> u64 {
    0xe210_0000_0000_0000 | target << 20 | PT | 0xf
}

fn pcnt(target: u64, at: u64) -> u64 {
    let imm = (target as i64 - at as i64 - 8) as u64 & 0xff_ffff;
    0xe2b0_0000_0000_0000 | imm << 20
}

const CONT: u64 = 0xe350_0000_0000_0000 | PT | 0xf;
const RET: u64 = 0xe320_0000_0000_0000 | PT | 0xf;

#[test]
fn prmt_modes_and_operand_forms() {
    for mode in 0..=6 {
        let shader = translated(
            &[
                mov32i(0, 0x0403_0201),
                mov32i(1, 0x0807_0605),
                mov32i(2, 0x3010_3210),
                prmt(mode, 3, 0, 1, 2),
                EXIT,
            ],
            &["prmt"],
        );
        let text = disassemble(&shader.spirv);
        assert!(text.contains("OpShiftLeftLogical"), "{text}");
        assert!(text.contains("OpVectorExtractDynamic"), "{text}");
    }

    // The permute control may also be an immediate.
    let shader = translated(
        &[mov32i(0, 0x0403_0201), prmt_imm(0, 1, 0, 0x3012, 2), EXIT],
        &["prmt"],
    );
    assert!(!shader.spirv.is_empty());

    // Modes above `rc16` are reserved.
    rejected(&[prmt(7, 1, 0, 0, 2), EXIT], "prmt (mode)");
}

#[test]
fn idp_dot_product_forms() {
    for (a_signed, b_signed) in [(false, false), (true, true), (true, false), (false, true)] {
        translated(
            &[
                mov32i(0, 0x0403_0201),
                mov32i(1, 0x0807_0605),
                mov32i(2, 7),
                idp(3, 0, 1, 2, a_signed, b_signed),
                EXIT,
            ],
            &["idp"],
        );
    }

    // Two-way (`dp2a`) and high-half forms are not translated.
    rejected(
        &[idp(3, 0, 1, 2, false, false) | 1 << 48, EXIT],
        "idp (dp2a)",
    );
    rejected(
        &[idp(3, 0, 1, 2, false, false) | 1 << 50, EXIT],
        "idp (high half)",
    );
}

#[test]
fn scaled_integer_add_immediate() {
    let shader = translated(
        &[
            mov32i(0, 5),
            iscadd32i(1, 0, 0x20, 3),
            iscadd32i(2, 0, 0xffff_ffff, 1),
            EXIT,
        ],
        &["iscadd32i"],
    );
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpShiftLeftLogical"), "{text}");
}

#[test]
fn special_register_reads() {
    // SRZ, the lane id and the constant one.
    translated(
        &[
            cs2r(0, 0xff),
            cs2r(1, 0x00),
            cs2r(2, 0x12),
            mov32i(3, 0),
            EXIT,
        ],
        &["cs2r"],
    );
    // Thread ids come from the compute builtins.
    let code = program(
        vec![0; kirikae_maxwell::sph::SIZE],
        &[cs2r(0, 0x21), cs2r(1, 0x20), EXIT],
    );
    let shader = translate(&code, &options(Stage::Compute), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    validate(&shader.spirv);
    // Counters and ids the shader cannot observe here stay labelled.
    rejected(&[cs2r(0, 0x04), EXIT], "cs2r (special register)");
}

#[test]
fn opaque_stack_and_local_base_pointers() {
    // These encodings carry no guard predicate: bits 16-19 stay clear, so the
    // operation must not be wrapped in a conditional block.
    let set_crs = 0xe2e0_0000_0000_0000u64;
    let get_crs = 0xe2c0_0000_0000_0000u64 | 2;
    let set_lmem = 0xe2f0_0000_0000_0000u64 | 1 << 8;
    let get_lmem = 0xe2d0_0000_0000_0000u64 | 3;
    let shader = translated(
        &[
            mov32i(0, 0x1234),
            mov32i(1, 0x5678),
            set_crs,
            get_crs,
            set_lmem,
            get_lmem,
            EXIT,
        ],
        &["setcrsptr", "getcrsptr", "setlmembase", "getlmembase"],
    );
    let text = disassemble(&shader.spirv);
    assert!(!text.contains("OpSelectionMerge"), "{text}");
}

#[test]
fn video_add_modes() {
    // u8 lanes, merge into the high half of `c`.
    translated(
        &[
            mov32i(0, 0x0403_0201),
            mov32i(1, 0x0807_0605),
            mov32i(2, 0x0c0b_0a09),
            vadd(3, 0, 1, 2, 0, 0, 0, false, 0),
            EXIT,
        ],
        &["vadd"],
    );

    // s16 lanes with saturation and the accumulate write-back.
    let shader = translated(
        &[
            mov32i(0, 0x8000_0001),
            mov32i(1, 0x7fff_ffff),
            mov32i(2, 3),
            vadd(3, 0, 1, 2, 0xc, 0x8 | 0xc, 0, true, 4),
            EXIT,
        ],
        &["vadd"],
    );
    assert!(!shader.spirv.is_empty());

    // `avg` negates an operand / adds one.
    for avg in 1..=3 {
        translated(
            &[
                mov32i(0, 0x0403_0201),
                mov32i(1, 0x0807_0605),
                mov32i(2, 9),
                vadd(3, 0, 1, 2, 0, 0, avg, false, 0),
                EXIT,
            ],
            &["vadd"],
        );
    }
}

#[test]
fn video_multiply_add_scaling() {
    for (scale, sat) in [(0, false), (1, true), (2, true)] {
        let shader = translated(
            &[
                mov32i(0, 0x0000_0100),
                mov32i(1, 0x0000_0200),
                mov32i(2, 0x0000_1000),
                vmad(3, 0, 1, 2, 0xa, 0xa, 0, scale, sat),
                EXIT,
            ],
            &["vmad"],
        );
        let text = disassemble(&shader.spirv);
        assert!(text.contains("OpIMul"), "{text}");
    }
}

#[test]
fn video_absolute_difference_and_compare() {
    translated(
        &[
            mov32i(0, 40),
            mov32i(1, 7),
            mov32i(2, 1),
            vabsdiff(3, 0, 1, 2, 0, 0, true),
            EXIT,
        ],
        &["vabsdiff"],
    );
    translated(
        &[
            mov32i(0, 0xffff_fff0),
            mov32i(1, 0x10),
            mov32i(2, 0),
            vabsdiff(3, 0, 1, 2, 0x8 | 0x3, 0x8 | 0x3, false),
            EXIT,
        ],
        &["vabsdiff"],
    );
    // Mixed signedness has no documented ordering for the difference.
    rejected(
        &[vabsdiff(3, 0, 1, 2, 0x8, 0, false), EXIT],
        "vabsdiff (mixed signedness)",
    );

    translated(
        &[mov32i(0, 1), mov32i(1, 2), vsetp(0, 1, 1, 7, 1), EXIT],
        &["vsetp"],
    );
}

#[test]
fn half_pair_32_bit_immediates() {
    let imm = 0x3c00_3c00;
    translated(
        &[
            mov32i(0, 0x3c00_4000),
            hadd2_32i(1, 0, imm, 0, false, true),
            hadd2_32i(2, 0, imm, 2, true, false),
            hmul2_32i(3, 0, imm, 1, true),
            hfma2_32i(4, 0, imm, 2, true),
            EXIT,
        ],
        &["hadd2_32i", "hmul2_32i", "hfma2_32i"],
    );
}

#[test]
fn continue_stack_round_trip() {
    let target = pc(3);
    let code = [mov32i(0, 1), pcnt(target, pc(1)), CONT, mov32i(1, 2), EXIT];
    let shader = translated(&code, &["pcnt", "cont"]);
    // The continue target becomes a block of the dispatcher state machine.
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpSwitch"), "{text}");
}

#[test]
fn calls_push_and_pop_the_return_address() {
    let ret_pc = pc(5);
    let code = [
        mov32i(0, 1),
        cal(ret_pc, pc(1)),
        mov32i(1, 2),
        NOP,
        NOP,
        RET,
        EXIT,
    ];
    let shader = translated(&code, &["cal", "ret"]);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpSwitch"), "{text}");

    // Absolute call, returning to the instruction after it.
    let code = [mov32i(0, 1), jcal(pc(4)), mov32i(1, 2), NOP, RET, EXIT];
    let shader = translated(&code, &["jcal", "ret"]);
    assert!(!shader.spirv.is_empty());
}

#[test]
fn absolute_jump_to_a_known_block() {
    let code = [mov32i(0, 1), jmp(pc(3)), NOP, mov32i(1, 2), EXIT];
    translated(&code, &["jmp"]);

    // A target outside the program cannot be resolved statically.
    rejected(&[jmp(0x4000), EXIT], "jmp (target outside the program)");
}

#[test]
fn control_and_video_forms_outside_the_model_are_reported_by_name() {
    // These are left out whole: coverage reports the mnemonic and the draw is
    // skipped instead of running with a guessed operation.
    let words = [
        0xf0b8_0000_0000_0000 | PT | 1,                              // b2r
        0xf0c0_0000_0000_0000 | PT | 1 << 28,                        // r2b
        0xe200_0000_0000_0000 | PT | 0xf,                            // jmx
        0xe230_0000_0000_0000 | PT,                                  // pexit
        0xe270_0000_0000_0000 | PT,                                  // pret
        0xe310_0000_0000_0000 | PT | 0xf,                            // longjmp
        0xe280_0000_0000_0000 | PT | 0xf,                            // plongjmp
        0xe3a0_0000_0000_0000 | 0x100,                               // bpt
        0xeea0_0000_0000_0000 | PT | 1 << 20,                        // stp
        0x5a80_0000_0000_0000 | 2 << 39 | 2 << 20 | PT | 1 << 8 | 1, // imadsp
        0xe370_0000_0000_0000,                                       // sam
        0xe380_0000_0000_0000,                                       // ram
        0xe360_0000_0000_0000 | 1,                                   // rtt
        0xfbe0_0000_0000_0000 | PT | 1 << 8,                         // out
        0x4000_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1,          // vset
        0x5700_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1, // vshl
        0x5600_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1, // vshr
        0x5000_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1, // vabsdiff4
        0x7c00_0000_0000_0000 | 1 << 20 | PT | 1 << 8 | 1,           // hset2
    ];
    for word in words {
        let shader = translate(
            &fragment(&[word, EXIT]),
            &options(Stage::Fragment),
            &kirikae_maxwell::NoEnv,
        )
        .expect("translation");
        assert_eq!(shader.unsupported.len(), 1, "{:?}", shader.unsupported);
        let label = shader.unsupported[0];
        // A label names the mnemonic it belongs to, bare or with the reason in
        // parentheses, so the coverage map still attributes the gap.
        let mnemonic = label.split(" (").next().unwrap_or(label);
        assert!(
            shader.mnemonics.iter().any(|(name, _)| *name == mnemonic),
            "{label} not decoded: {:?}",
            shader.mnemonics
        );
        validate(&shader.spirv);
    }
}

#[test]
fn video_multiply_add_rejects_the_undefined_scale() {
    rejected(
        &[mov32i(0, 1), vmad(1, 0, 0, 0, 0, 0, 0, 3, false), EXIT],
        "vmad (scale)",
    );
}

#[test]
fn untranslated_fp64_forms_are_reported_by_name() {
    // FP64 would need the `Float64` capability, which this pipeline does not
    // request: the Vulkan device would have to expose `shaderFloat64`.
    for (word, name) in [
        (
            0x5c70_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1,
            "dadd",
        ),
        (
            0x5c80_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1,
            "dmul",
        ),
        (
            0x5b70_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1,
            "dfma",
        ),
        (
            0x5c50_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1,
            "dmnmx",
        ),
        (
            0x5900_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1,
            "dset",
        ),
        (
            0x5b80_0000_0000_0000 | 2 << 39 | 1 << 20 | PT | 1 << 8 | 1,
            "dsetp",
        ),
        (0x5c88_0000_0000_0000 | 2 << 20 | PT | 1 << 8 | 1, "fchk"),
    ] {
        let shader = translate(
            &fragment(&[word, EXIT]),
            &options(Stage::Fragment),
            &kirikae_maxwell::NoEnv,
        )
        .expect("translation");
        assert_eq!(shader.unsupported, [name]);
        assert!(
            shader.mnemonics.iter().any(|(m, _)| *m == name),
            "{name} not decoded: {:?}",
            shader.mnemonics
        );
        validate(&shader.spirv);
    }
}

fn s2r(rd: u64, sr: u64) -> u64 {
    0xf0c8_0000_0000_0000 | sr << 20 | PT | rd
}

#[test]
fn s2r_reads_the_view_scale_factors_as_identity() {
    // Jamboree's vertex shaders read SR_WSCALEFACTOR_XY (0x1e) and
    // SR_WSCALEFACTOR_Z (0x1f) and multiply W by them; with one view both are
    // the identity, which is the only scale the runtime can have.
    let code = program(
        fragment_header(1),
        &[s2r(0, 0x1e), s2r(1, 0x1f), mov32i(2, 0), EXIT],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert!(shader.mnemonics.iter().any(|(m, _)| *m == "mov"));
    validate(&shader.spirv);

    // A system value the runtime cannot stand in for stays reported. The
    // global timer has no SPIR-V equivalent and changes every cycle.
    let code = program(fragment_header(1), &[s2r(0, 0x52), EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["s2r (system value)"]);
    validate(&shader.spirv);
}

#[test]
fn csetp_accepts_every_expressible_condition_code() {
    // `csetp` writes two predicates from the 5-bit condition code. The low
    // 24 codes are the flag tests; 0x18..0x1f name the warp's conditional
    // stop mask, which the translator cannot express.
    for cc in 0..0x20u64 {
        let words = [0x50a0_0000_0000_0000 | cc << 8 | 7 << 39 | PT | 7, EXIT];
        let shader = translate(
            &program(fragment_header(1), &words),
            &options(Stage::Fragment),
            &NoEnv,
        )
        .unwrap();
        if cc < 0x18 {
            assert!(
                shader.unsupported.is_empty(),
                "{cc:#x}: {:?}",
                shader.unsupported
            );
        } else {
            assert_eq!(shader.unsupported, ["conditional stop mask"], "{cc:#x}");
        }
        validate(&shader.spirv);
    }
}
