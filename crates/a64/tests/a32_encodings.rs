use kirikae_a64::a32::{DecodeError, decode};

fn check(cases: &[(u32, &str)]) {
    for &(word, text) in cases {
        match decode(0x1000, word) {
            Ok(insn) => assert_eq!(insn.to_string(), text, "{word:#010x}"),
            Err(e) => panic!("{word:#010x} ({text}) did not decode: {e:?}"),
        }
    }
}

#[test]
fn data_processing_matches_the_assembler() {
    check(&[
        (0xe0810002, "add r0, r1, r2"),
        (0xe29430ff, "adds r3, r4, #0xff"),
        (0xe0410182, "sub r0, r1, r2, lsl #3"),
        (0xe2665000, "rsb r5, r6, #0x0"),
        (0xe0010372, "and r0, r1, r2, ror r3"),
        (0xe38874ff, "orr r7, r8, #0xff000000"),
        (0xe0221fc3, "eor r1, r2, r3, asr #31"),
        (0xe3c54001, "bic r4, r5, #0x1"),
        (0xe1e00001, "mvn r0, r1"),
        (0xe3a00000, "mov r0, #0x0"),
        (0xe1b010a2, "movs r1, r2, lsr #1"),
        (0xe350000a, "cmp r0, #0xa"),
        (0xe1710002, "cmn r1, r2"),
        (0xe3130004, "tst r3, #0x4"),
        (0xe1340005, "teq r4, r5"),
        (0xe3010234, "movw r0, #0x1234"),
        (0xe34a0bcd, "movt r0, #0xabcd"),
        (0xe0a10002, "adc r0, r1, r2"),
        (0xe0c43005, "sbc r3, r4, r5"),
    ]);
}

#[test]
fn multiply_matches_the_assembler() {
    check(&[
        (0xe0000291, "mul r0, r1, r2"),
        (0xe0203291, "mla r0, r1, r2, r3"),
        (0xe0810392, "umull r0, r1, r2, r3"),
        (0xe0c10392, "smull r0, r1, r2, r3"),
        (0xe0a10392, "umlal r0, r1, r2, r3"),
        (0xe0e10392, "smlal r0, r1, r2, r3"),
        (0xe710f211, "sdiv r0, r1, r2"),
        (0xe733f514, "udiv r3, r4, r5"),
        (0xe0603291, "mls r0, r1, r2, r3"),
    ]);
}

#[test]
fn load_store_matches_the_assembler() {
    check(&[
        (0xe5910000, "ldr r0, [r1]"),
        (0xe5910004, "ldr r0, [r1, #0x4]"),
        (0xe5310004, "ldr r0, [r1, #-0x4]!"),
        (0xe4910008, "ldr r0, [r1], #0x8"),
        (0xe58d2010, "str r2, [sp, #0x10]"),
        (0xe7d10002, "ldrb r0, [r1, r2]"),
        (0xe7410102, "strb r0, [r1, -r2, lsl #2]"),
        (0xe1d100b2, "ldrh r0, [r1, #0x2]"),
        (0xe04100b2, "strh r0, [r1], #-0x2"),
        (0xe1d100d0, "ldrsb r0, [r1]"),
        (0xe1d100f6, "ldrsh r0, [r1, #0x6]"),
        (0xe1c200d8, "ldrd r0, r1, [r2, #0x8]"),
        (0xe16d20f8, "strd r2, r3, [sp, #-0x8]!"),
        (0xe92d4030, "stmdb sp!, {r4, r5, lr}"),
        (0xe8bd8030, "ldmia sp!, {r4, r5, pc}"),
        (0xe8b0000e, "ldmia r0!, {r1-r3}"),
        (0xe92d0003, "stmdb sp!, {r0, r1}"),
    ]);
}

#[test]
fn exclusive_and_ordered_matches_the_assembler() {
    check(&[
        (0xe1910f9f, "ldrex r0, [r1]"),
        (0xe1812f90, "strex r2, r0, [r1]"),
        (0xe1d10f9f, "ldrexb r0, [r1]"),
        (0xe1c12f90, "strexb r2, r0, [r1]"),
        (0xe1f10f9f, "ldrexh r0, [r1]"),
        (0xe1e12f90, "strexh r2, r0, [r1]"),
        (0xe1b20f9f, "ldrexd r0, r1, [r2]"),
        (0xe1a24f90, "strexd r4, r0, r1, [r2]"),
        (0xe1910e9f, "ldaex r0, [r1]"),
        (0xe1812e90, "stlex r2, r0, [r1]"),
        (0xe1d10e9f, "ldaexb r0, [r1]"),
        (0xe1c12e90, "stlexb r2, r0, [r1]"),
        (0xe1f10e9f, "ldaexh r0, [r1]"),
        (0xe1e12e90, "stlexh r2, r0, [r1]"),
        (0xe1b20e9f, "ldaexd r0, r1, [r2]"),
        (0xe1a24e90, "stlexd r4, r0, r1, [r2]"),
        (0xe1910c9f, "lda r0, [r1]"),
        (0xe181fc90, "stl r0, [r1]"),
        (0xe1d10c9f, "ldab r0, [r1]"),
        (0xe1e1fc90, "stlh r0, [r1]"),
    ]);
}

#[test]
fn barriers_and_hints_matches_the_assembler() {
    check(&[
        (0xf57ff01f, "clrex"),
        (0xf57ff05b, "dmb #0xb"),
        (0xf57ff04f, "dsb #0xf"),
        (0xf57ff06f, "isb #0xf"),
        (0xf57ff070, "sb"),
        (0xe320f000, "nop"),
        (0xe320f001, "yield"),
        (0xe320f002, "wfe"),
        (0xe320f003, "wfi"),
        (0xe320f004, "sev"),
    ]);
}

#[test]
fn register_branches_matches_the_assembler() {
    check(&[
        (0xe12fff1e, "bx lr"),
        (0xe12fff33, "blx r3"),
    ]);
}

#[test]
fn vfp_matches_the_assembler() {
    check(&[
        (0xee300a81, "vadd.f32 s0, s1, s2"),
        (0xee310b02, "vadd.f64 d0, d1, d2"),
        (0xee621a22, "vmul.f32 s3, s4, s5"),
        (0xee710be2, "vsub.f64 d16, d17, d18"),
        (0xee800a81, "vdiv.f32 s0, s1, s2"),
        (0xeeb10bc1, "vsqrt.f64 d0, d1"),
        (0xeef10a41, "vneg.f32 s1, s2"),
        (0xeeb03bc4, "vabs.f64 d3, d4"),
        (0xeeb00a60, "vmov.f32 s0, s1"),
        (0xee100a90, "vmov r0, s1"),
        (0xee011a10, "vmov s2, r1"),
        (0xec410b10, "vmov d0, r0, r1"),
        (0xeeb40a60, "vcmp.f32 s0, s1"),
        (0xeeb40bc1, "vcmpe.f64 d0, d1"),
        (0xeebd0ae0, "vcvt.s32.f32 s0, s1"),
        (0xeeb70ae0, "vcvt.f64.f32 d0, s1"),
        (0xed900a01, "vldr s0, [r0, #0x4]"),
        (0xed111b02, "vldr d1, [r1, #-0x8]"),
        (0xedcd1a00, "vstr s3, [sp]"),
        (0xed6d8a01, "vpush {s17}"),
        (0xed2d8a04, "vpush {s16-s19}"),
        (0xecfd0a01, "vpop {s1}"),
        (0xed2d8b10, "vpush {d8-d15}"),
        (0xecfd0b04, "vpop {d16, d17}"),
        (0xecb00a04, "vldmia r0!, {s0-s3}"),
        (0xed210b04, "vstmdb r1!, {d0, d1}"),
        (0xeef10a10, "vmrs r0, fpscr"),
        (0xeee11a10, "vmsr fpscr, r1"),
    ]);
}

#[test]
fn neon_matches_the_assembler() {
    check(&[
        (0xf2220844, "vadd.i32 q0, q1, q2"),
        (0xf3010802, "vsub.i8 d0, d1, d2"),
        (0xf2010112, "vand d0, d1, d2"),
        (0xf2220154, "vorr q0, q1, q2"),
        (0xf3010112, "veor d0, d1, d2"),
        (0xf3120154, "vbsl q0, q1, q2"),
        (0xf2810010, "vmov.i32 d0, #0x10"),
        (0xf2810110, "vorr.i32 d0, #0x10"),
        (0xf3870e5f, "vmov.i8 q0, #0xff"),
        (0xf2800030, "vmvn.i32 d0, #0x0"),
        (0xf2800931, "vbic.i16 d0, #0x1"),
        (0xf2b10302, "vext.8 d0, d1, d2, #0x3"),
        (0xf3b80001, "vrev64.32 d0, d1"),
        (0xf3b400c2, "vrev32.16 q0, q1"),
        (0xf3b00101, "vrev16.8 d0, d1"),
        (0xf3b50301, "vabs.s16 d0, d1"),
        (0xf3b103c2, "vneg.s8 q0, q1"),
        (0xf3b90701, "vabs.f32 d0, d1"),
        (0xf3b907c2, "vneg.f32 q0, q1"),
        (0xf3b00581, "vmvn d0, d1"),
    ]);
}

#[test]
fn shift_by_immediate_and_crypto_are_reported_unsupported_not_misdecoded() {
    for word in [0xf38d0011, 0xf2bf0052, 0xf2920511, 0xf3880a11, 0xf28c0812, 0xf29d0912, 0xf3a00191, 0xf3b80011, 0xf3b80052, 0xf3b00302, 0xf3b00382] {
        assert_eq!(decode(0x1000, word).err(), Some(DecodeError::Unsupported), "{word:#010x}");
    }
}

#[test]
fn hints_are_conditional_and_unknown_hints_are_nops() {
    check(&[(0x1320f002, "wfene"), (0x0320f000, "nopeq"), (0xe320f005, "sevl"), (0xe320f014, "nop")]);
    for word in [0xf57ff030u32, 0xf57ff035] {
        assert_eq!(decode(0x1000, word).err(), Some(DecodeError::Unallocated), "{word:#010x}");
    }
}

#[test]
fn cases_found_by_the_llvm_differential_match_the_assembler() {
    check(&[
        (0xf3878d5f, "vmov.i32 q4, #0xffffff"),
        (0xf2c15d38, "vmvn.i32 d21, #0x18ffff"),
        (0xe7430714, "smlald r0, r3, r4, r7"),
        (0xeef20b70, "vmov.u8 r0, d2[7]"),
        (0xec8d8b1e, "vstmia sp, {d8-d22}"),
        (0xe10f0000, "mrs r0, apsr"),
        (0xe128f000, "msr APSR_nzcvq, r0"),
        (0xeef80a10, "vmrs r0, fpexc"),
        (0xeef30b70, "vmov.u8 r0, d3[7]"),
        (0xeee40b10, "vdup.8 q2, r0"),
        (0xf420071f, "vld1.8 {d0}, [r0:64]"),
        (0xec9d8b04, "vldmia sp, {d8, d9}"),
    ]);
}

#[test]
fn undefined_encodings_found_by_the_llvm_differential_are_refused() {
    let cases = [
        (0xf3774618u32, DecodeError::Unallocated),
        (0xf2f5680c, DecodeError::Unallocated),
        (0xf37a4340, DecodeError::Unallocated),
        (0xf23468b6, DecodeError::Unallocated),
        (0xf279399b, DecodeError::Unallocated),
        (0xf3b80080, DecodeError::Unallocated),
        (0xf3bd0301, DecodeError::Unallocated),
        (0xf43d5a3f, DecodeError::Unallocated),
        (0xf420072f, DecodeError::Unallocated),
        (0xf42007af, DecodeError::Unallocated),
        (0xf420083f, DecodeError::Unallocated),
        (0xeea7ab19, DecodeError::Unpredictable),
        (0xeef1baf0, DecodeError::Unpredictable),
        (0xeef2baf0, DecodeError::Unsupported),
        (0xee734b16, DecodeError::Unpredictable),
        (0xecbd8b22, DecodeError::Unpredictable),
    ];
    for (word, error) in cases {
        assert_eq!(decode(0x1000, word).err(), Some(error), "{word:#010x}");
    }
}

#[test]
fn two_register_neighbours_are_reported_unsupported_not_misdecoded() {
    for word in [0xf3b10802u32, 0xf3bc0c01, 0xf3b10000, 0xf3b10100, 0xf3b00780, 0xf3b00500, 0xf3b902c0, 0xf3b20000] {
        assert_eq!(decode(0x1000, word).err(), Some(DecodeError::Unsupported), "{word:#010x}");
    }
}
