mod common;

use kirikae_maxwell::{Options, Stage, sph, translate};

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

fn mov_cbuf(rd: u64, slot: u64, offset: u64) -> u64 {
    0x4c98_0000_0000_0000 | slot << 34 | (offset >> 2) << 20 | PT | rd
}

/// `vote.{all,any,eq} Rd, Psrc, Pdst`: type in bits 48-49, source predicate in
/// 45-47 (negated in 50), destination predicate in 39-41 (negated in 42).
fn vote_word(kind: u64, rd: u64, src: u64, src_not: bool, dst: u64, dst_not: bool) -> u64 {
    0x50d8_0000_0000_0000
        | kind << 48
        | src << 45
        | dst << 39
        | (src_not as u64) << 50
        | (dst_not as u64) << 42
        | PT
        | rd
}

/// `atoms.{add,min,max,inc,dec,and,or,xor,exch}.{u32,s32,u64,s64} Rd, [s[Ra]+imm], Rb`.
fn atoms_word(op: u64, size: u64, rd: u64, ra: u64, rb: u64, imm_units: u64) -> u64 {
    0xec00_0000_0000_0000 | op << 52 | size << 28 | imm_units << 30 | rb << 20 | PT | ra << 8 | rd
}

/// `atoms.{cast,cast.spin,cas} Rd, [s[Ra]+imm], Rb`: sub-operation in bits 53-54
/// and the 64-bit flag in bit 52.
fn atoms_pair_word(subop: u64, wide: bool, rd: u64, ra: u64, rb: u64) -> u64 {
    0xee00_0000_0000_0000 | subop << 53 | (wide as u64) << 52 | rb << 20 | PT | ra << 8 | rd
}

/// A surface instruction (`suld` 0xeb00, `sust` 0xeb20, `sured` 0xeb40,
/// `suatom` 0xea60/0xeac0) against the descriptor register `desc`.
fn surface_word(base: u64, desc: u64, rd: u64, ra: u64, dim: u64, mask: u64) -> u64 {
    base | 1 << 52 // `.D`: the surface descriptor form
        | desc << 39
        | dim << 33
        | mask << 20
        | PT
        | ra << 8
        | rd
}

#[test]
fn vote_ballots_cover_all_any_and_uniform() {
    // `vote.all R0, PT, PT` is how the compiler materialises the active mask;
    // the other two types write the same ballot and a different condition.
    let code = program(
        fragment_header(1),
        &[
            vote_word(0, 0, 7, false, 7, false),
            vote_word(1, 1, 7, false, 7, false),
            vote_word(2, 2, 0x84, false, 7, false),
            EXIT,
        ],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpGroupNonUniformBallot"), "{text}");
    // all/any/eq each compare the ballot with the active mask or zero.
    assert!(text.matches("OpIEqual").count() >= 3, "{text}");
    assert!(text.contains("OpINotEqual"), "{text}");
    validate(&shader.spirv);
}

#[test]
fn vote_reads_the_source_predicate_and_writes_the_destination_one() {
    // The word a dumped shader uses: `@P2 vote.all R71, !P3, P4`, i.e. the
    // ballot of the lanes where P3 is clear, stored in R71, with the condition
    // written to P4. `pred39` is the source (with its `not` bit) and `pred45`
    // the destination.
    let code = program(fragment_header(1), &[0x50dc_9598_e9f2_1f47, EXIT]);
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpGroupNonUniformBallot"), "{text}");
    // The source predicate is `!P3`, so the ballot is fed by a negation.
    assert!(text.contains("OpLogicalNot"), "{text}");
    validate(&shader.spirv);
}

#[test]
fn vote_forms_outside_the_three_ballots_stay_labelled() {
    // Bit 49 set is undocumented for `vote`; nothing is emitted for it.
    let code = program(
        fragment_header(1),
        &[vote_word(3, 0, 7, false, 7, false), EXIT],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["vote (type 3)"]);
    let text = disassemble(&shader.spirv);
    assert!(!text.contains("OpGroupNonUniformBallot"), "{text}");
    validate(&shader.spirv);
}

#[test]
fn vote_to_governor_and_the_stop_mask_exit_stay_labelled() {
    // The exact pair Jamboree reports: `vote.vtg ra, 0x321111` followed by
    // `exit` with the conditional-stop-mask code 0x1c. Neither the vote's
    // thread-group state nor the stop mask it feeds exists here, so both keep a
    // label and emit no operation of their own.
    let code = program(
        fragment_header(1),
        &[0x50e2_4321_1117_0000, 0xe300_0000_0007_001c, EXIT],
    );
    let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(
        shader.unsupported,
        ["conditional stop mask", "vote (vtg)"],
        "the vote and its stop-mask exit stay reported"
    );
    validate(&shader.spirv);
}

#[test]
fn shared_atomics_reduce_shared_memory_for_32_bit_integers() {
    // One `atoms` per op: add, min, max, and, or, xor, exch. None of them reads
    // or writes a global buffer, so the shader needs no storage binding.
    let word = |op: u64, size: u64, rd: u64, imm: u64| atoms_word(op, size, rd, 4, 5, imm);
    let code = program(
        vec![0; sph::SIZE],
        &[
            word(0, 0, 0, 0), // add.u32
            word(1, 1, 1, 1), // min.s32
            word(1, 0, 2, 2), // min.u32
            word(2, 0, 3, 3), // max.u32
            word(2, 1, 4, 4), // max.s32
            word(5, 0, 5, 5), // and
            word(6, 0, 6, 6), // or
            word(7, 0, 7, 7), // xor
            word(8, 0, 8, 8), // exch
            EXIT,
        ],
    );
    let opts = Options {
        shared_memory: 0x100,
        ..options(Stage::Compute)
    };
    let shader = translate(&code, &opts, &kirikae_maxwell::NoEnv).unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.storage, 0, "atomics are not global buffer writes");
    assert!(shader.images.is_empty());
    let text = disassemble(&shader.spirv);
    for op in [
        "OpAtomicIAdd",
        "OpAtomicUMin",
        "OpAtomicSMin",
        "OpAtomicUMax",
        "OpAtomicSMax",
        "OpAtomicAnd",
        "OpAtomicOr",
        "OpAtomicXor",
        "OpAtomicExchange",
    ] {
        assert!(text.contains(op), "{op} missing from {text}");
    }
    validate(&shader.spirv);
}

#[test]
fn shared_atomic_forms_that_are_not_32_bit_integers_stay_labelled() {
    // 64-bit lanes, the increment/decrement pair (a compare-and-swap loop on
    // hardware) and the compare-and-swap/`cast` encodings are not expressible
    // with a single workgroup atomic.
    let cases = [
        (atoms_word(0, 2, 0, 4, 5, 0), "atoms (64-bit)"),
        (atoms_word(0, 3, 0, 4, 5, 0), "atoms (64-bit)"),
        (atoms_word(3, 0, 0, 4, 5, 0), "atoms (inc/dec)"),
        (atoms_word(4, 0, 0, 4, 5, 0), "atoms (inc/dec)"),
        (atoms_pair_word(0, false, 0, 4, 5), "atoms (cast)"),
        (atoms_pair_word(1, false, 0, 4, 5), "atoms (cast spin)"),
        (atoms_pair_word(2, false, 0, 4, 5), "atoms (cas)"),
        (atoms_pair_word(2, true, 0, 4, 5), "atoms (cas)"),
    ];
    for (word, label) in cases {
        let code = program(vec![0; sph::SIZE], &[word, EXIT]);
        let opts = Options {
            shared_memory: 0x100,
            ..options(Stage::Compute)
        };
        let shader = translate(&code, &opts, &kirikae_maxwell::NoEnv).unwrap();
        assert_eq!(shader.unsupported, [label], "{word:016x}");
        let text = disassemble(&shader.spirv);
        assert!(
            !text.contains("OpAtomic"),
            "{label} emitted an atomic: {text}"
        );
        validate(&shader.spirv);
    }
}

#[test]
fn shared_atomics_scale_the_immediate_by_four_bytes() {
    // The offset is a signed 22-bit field at bit 30 in four-byte units: 0x1555
    // words is 21844 bytes, and -1 is -4.
    for (units, bytes) in [(0x1555u64, 21844u32), ((-1i64) as u64, 0xffff_fffc)] {
        let code = program(
            vec![0; sph::SIZE],
            &[atoms_word(0, 0, 0, 4, 5, units & 0x3f_ffff), EXIT],
        );
        let opts = Options {
            shared_memory: 0x100,
            ..options(Stage::Compute)
        };
        let shader = translate(&code, &opts, &kirikae_maxwell::NoEnv).unwrap();
        assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
        let text = disassemble(&shader.spirv);
        assert!(
            text.lines()
                .any(|line| line.contains("OpConstant") && line.ends_with(&format!(" {bytes}"))),
            "offset {bytes} missing from {text}"
        );
        validate(&shader.spirv);
    }
}

#[test]
fn shared_atomics_without_shared_memory_stay_labelled() {
    // A stage with no shared allocation has nothing to reduce, so the gap is
    // reported instead of being written into the local-memory array.
    let code = program(vec![0; sph::SIZE], &[atoms_word(0, 0, 0, 4, 5, 0), EXIT]);
    let opts = Options {
        shared_memory: 0,
        ..options(Stage::Compute)
    };
    let shader = translate(&code, &opts, &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["atoms (no shared memory)"]);
    let text = disassemble(&shader.spirv);
    assert!(!text.contains("OpAtomic"), "{text}");
    validate(&shader.spirv);
}

#[test]
fn surface_load_stores_and_atomics_keep_their_labels() {
    // `suld`, `sured` and `suatom` resolve the same surface descriptor as
    // `sust`, which the runtime can only bind as a storage image with an
    // unknown format: reading it or making it an atomic target needs a device
    // feature that is not enabled, so no operation is emitted.
    let cases = [
        (
            surface_word(0xeb00_0000_0000_0000, 39, 0, 4, 6, 0xf),
            "suld (surface read of an unknown-format image)",
        ),
        (
            surface_word(0xeb40_0000_0000_0000, 39, 0, 4, 6, 0),
            "sured (surface atomic of an unknown-format image)",
        ),
        (
            surface_word(0xea60_0000_0000_0000, 39, 0, 4, 6, 0),
            "suatom (surface atomic of an unknown-format image)",
        ),
    ];
    for (word, label) in cases {
        let code = program(vertex_header(), &[mov_cbuf(39, 0, 0x110), word, EXIT]);
        let shader = translate(&code, &options(Stage::Vertex), &kirikae_maxwell::NoEnv).unwrap();
        assert_eq!(shader.unsupported, [label], "{word:016x}");
        assert!(shader.images.is_empty(), "{label} bound an image");
        let text = disassemble(&shader.spirv);
        assert!(
            !text.contains("OpImageRead") && !text.contains("OpImageTexelPointer"),
            "{label} emitted an image access: {text}"
        );
        validate(&shader.spirv);
    }
}

#[test]
fn surface_store_keeps_the_descriptor_form_and_labels_the_physical_one() {
    // Bit 52 picks `.D` (a surface descriptor, which `sust` can write through)
    // over `.P`, whose address is physical and has no descriptor to resolve.
    let descriptor = program(
        vertex_header(),
        &[
            mov_cbuf(39, 0, 0x110),
            surface_word(0xeb20_0000_0000_0000, 39, 0, 4, 3, 1),
            EXIT,
        ],
    );
    let shader = translate(
        &descriptor,
        &options(Stage::Vertex),
        &kirikae_maxwell::NoEnv,
    )
    .unwrap();
    assert!(shader.unsupported.is_empty(), "{:?}", shader.unsupported);
    assert_eq!(shader.images.len(), 1);
    assert_eq!(
        (shader.images[0].slot, shader.images[0].handle),
        (Some(0), 0x110 / 4)
    );
    // naga's frontend rejects `StorageImageWriteWithoutFormat`, which is why the
    // module is only checked structurally here; `spirv-val` accepts the
    // capability.
    let text = disassemble(&shader.spirv);
    assert!(text.contains("OpImageWrite"), "{text}");
    common::check_structure(&shader.spirv);

    let physical = program(
        vertex_header(),
        &[
            mov_cbuf(39, 0, 0x110),
            surface_word(0xeb20_0000_0000_0000, 39, 0, 4, 3, 1) & !(1 << 52),
            EXIT,
        ],
    );
    let shader = translate(&physical, &options(Stage::Vertex), &kirikae_maxwell::NoEnv).unwrap();
    assert_eq!(shader.unsupported, ["sust (physical address)"]);
    assert!(shader.images.is_empty());
    validate(&shader.spirv);
}

#[test]
fn instructions_whose_inputs_are_not_in_the_shader_keep_a_reason() {
    // Each of these reads driver or rasteriser state the shader interface does
    // not carry, so it reports what is missing instead of approximating.
    let words: &[(u64, &str)] = &[
        (
            0xdf40_0000_0000_0000 | PT,
            "txa (texture descriptor is not a shader input)",
        ),
        (
            0xefe8_0000_0000_0000 | PT | 0xf,
            "pixld (rasteriser coverage is not a shader input)",
        ),
        (
            0xefa0_0000_0000_0000 | PT,
            "al2p (attribute address is not a shader input)",
        ),
        (
            0xefd0_0000_0000_0000 | PT,
            "isberd (primitive attribute stream)",
        ),
        (0xf0b8_0000_0000_0000 | PT | 3, "b2r (barrier register)"),
        (0xf0c0_0000_0000_0000 | PT | 3, "r2b (barrier register)"),
        (
            0xe200_0000_0000_0000 | PT | 0xf,
            "jmx (computed target outside the scanned program)",
        ),
        (0xfbe0_0000_0000_0000 | PT, "out (geometry output stream)"),
    ];
    for &(word, label) in words {
        let code = program(fragment_header(1), &[word, EXIT]);
        let shader = translate(&code, &options(Stage::Fragment), &kirikae_maxwell::NoEnv).unwrap();
        assert_eq!(shader.unsupported, [label], "{word:016x}");
        validate(&shader.spirv);
    }
}
