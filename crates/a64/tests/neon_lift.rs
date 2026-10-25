//! Differential tests for the native integer NEON lifting (`a64::lift_neon`, plus the
//! forms `a64::lift_fp` already lifted).
//!
//! Every word is executed twice from the same state: once by the runtime helper in
//! `kirikae_a64::simd` (the reference) and once by evaluating the lifted IR the way the
//! interpreter in `kirikae-rt` does (the same `Op::eval` table, `Expr::V` halves and
//! `Stmt::SetV`). The whole V register file and the flags must match.

use kirikae_a64::asm;
use kirikae_a64::simd::oracle::{State, mem_pattern};
use kirikae_a64::{Control, Expr, Insn, Op, Stmt, SysReg, lift};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn state(&mut self) -> State {
        let mut v = [0u128; 32];
        for q in v.iter_mut() {
            *q = match self.next() % 4 {
                0 => 0,
                1 => u128::MAX,
                _ => self.next() as u128 | (self.next() as u128) << 64,
            };
        }
        State {
            x: [0; 31],
            sp: 0,
            v,
            nzcv: (self.next() & 0xf) << 28,
            fpcr: 0,
            // Keeps the saturation flag sometimes already set, so a lifting that replaced
            // `FPSR` instead of adding `QC` would be caught.
            fpsr: (self.next() & 0x1f) << 27,
            mem: mem_pattern(),
        }
    }
}

fn eval(s: &State, e: &Expr, tmp: &[u64; 8]) -> u64 {
    match e {
        Expr::Const(v) => *v,
        Expr::Tmp(n) => tmp[*n as usize],
        Expr::V { reg, hi } => {
            let q = s.v[*reg as usize];
            if *hi { (q >> 64) as u64 } else { q as u64 }
        }
        Expr::Op(op, args) => {
            let mut vals = [0u64; 5];
            for (slot, a) in vals.iter_mut().zip(args) {
                *slot = eval(s, a, tmp);
            }
            op.eval(&vals[..args.len()])
        }
        Expr::SysReg(SysReg::Fpsr) => s.fpsr,
        Expr::SysReg(SysReg::Fpcr) => s.fpcr,
        other => panic!("unexpected operand in a lifted NEON word: {other:?}"),
    }
}

/// Mirrors `kirikae_rt::interp::exec` for the statements these liftings emit.
fn exec_ir(s: &mut State, insn: &Insn) {
    let mut tmp = [0u64; 8];
    for st in &insn.stmts {
        match st {
            Stmt::Let(n, e) => tmp[*n as usize] = eval(s, e, &tmp),
            Stmt::SetV { reg, hi, value } => {
                let x = eval(s, value, &tmp) as u128;
                let q = &mut s.v[*reg as usize];
                *q = if *hi {
                    (*q & u64::MAX as u128) | x << 64
                } else {
                    (*q & !(u64::MAX as u128)) | x
                };
            }
            Stmt::SetSysReg(SysReg::Fpsr, e) => s.fpsr = eval(s, e, &tmp),
            other => panic!("unexpected statement in a lifted NEON word: {other:?}"),
        }
    }
}

fn lifted(word: u32) -> bool {
    let insn = lift(0, word);
    !insn.stmts.iter().any(|s| matches!(s, Stmt::Simd(_))) && insn.control() != Control::Undefined
}

fn vector_expr(e: &Expr) -> bool {
    match e {
        Expr::Const(_) | Expr::Tmp(_) | Expr::V { .. } | Expr::SysReg(_) => true,
        Expr::Op(_, args) => args.iter().all(vector_expr),
        _ => false,
    }
}

/// True when the lifted instruction only reads and writes V registers, so the test
/// executor below can run it without a full `Cpu`.
fn vector_only(insn: &Insn) -> bool {
    insn.stmts.iter().all(|s| match s {
        Stmt::Let(_, e) | Stmt::SetV { value: e, .. } => vector_expr(e),
        _ => false,
    })
}

/// Runs the word through the helper and through the lifted IR and compares everything.
fn check(word: u32, rng: &mut Rng) {
    let insn = lift(0, word);
    assert!(!insn.stmts.is_empty(), "{word:#010x} lifted to nothing");
    assert!(
        !insn.stmts.iter().any(|s| matches!(s, Stmt::Simd(_))),
        "{word:#010x} still goes through the SIMD helper: {:?}",
        insn.stmts
    );
    for _ in 0..8 {
        let mut want = rng.state();
        let mut got = want.clone();
        assert!(
            kirikae_a64::simd::exec(&mut want, word),
            "{word:#010x} rejected by the helper"
        );
        exec_ir(&mut got, &insn);
        for i in 0..32 {
            assert_eq!(
                got.v[i], want.v[i],
                "{word:#010x}: v{i} mismatch (helper {:#034x} vs IR {:#034x})",
                want.v[i], got.v[i]
            );
        }
        assert_eq!(got.nzcv, want.nzcv, "{word:#010x}: nzcv mismatch");
        assert_eq!(got.fpsr, want.fpsr, "{word:#010x}: fpsr mismatch");
    }
}

/// Every encoding the lifters claim, tagged with the family it belongs to.
fn covered() -> Vec<(u32, &'static str)> {
    let mut words = Vec::new();
    // Bitwise three-same: already lifted by `lift_fp`, kept as a regression check.
    for q in [false, true] {
        for u in [false, true] {
            for size in 0..3 {
                for (rm, rn, rd) in [(1, 2, 3), (5, 6, 7), (31, 0, 31), (4, 4, 4)] {
                    words.push((asm::neon_logic(q, u, size, rm, rn, rd), "bitwise"));
                }
            }
        }
    }
    // SHL, USHR and SSHR by immediate, every lane width and shift amount.
    for q in [false, true] {
        for es in [8u32, 16, 32, 64] {
            if !q && es == 64 {
                continue;
            }
            for n in 1..=es {
                words.push((
                    asm::neon_shift_imm(q, false, 2 * es - n, 0b00000, 1, 0),
                    "sshr",
                ));
                words.push((
                    asm::neon_shift_imm(q, true, 2 * es - n, 0b00000, 1, 0),
                    "ushr",
                ));
            }
            for n in 0..es {
                words.push((asm::neon_shift_imm(q, false, es + n, 0b01010, 2, 0), "shl"));
            }
        }
    }
    // DUP (element) and INS (element).
    for q in [false, true] {
        let datasize = if q { 128 } else { 64 };
        for es in [8u32, 16, 32, 64] {
            if !q && es == 64 {
                continue;
            }
            for index in 0..datasize / es {
                let w = asm::neon_copy(q, false, asm::neon_imm5(es, index), 0, 1, 0);
                words.push((w, if es <= 16 { "dup narrow" } else { "dup wide" }));
            }
        }
    }
    for es in [8u32, 16, 32, 64] {
        for index in 0..128 / es {
            for src in 0..128 / es {
                let imm4 = src << (es.trailing_zeros() - 3);
                let w = asm::neon_copy(true, true, asm::neon_imm5(es, index), imm4, 3, 0);
                words.push((w, if es <= 16 { "ins narrow" } else { "ins wide" }));
            }
        }
    }
    // Three-same integer arithmetic. ADD/SUB come from `lift_fp`; MUL and the four
    // min/max forms from `lift_neon`.
    let regs = [(1, 2, 3), (5, 6, 7), (31, 0, 31), (4, 4, 4)];
    for q in [false, true] {
        for size in 0..4 {
            if !q && size == 3 {
                continue;
            }
            for (rm, rn, rd) in regs {
                for u in [false, true] {
                    words.push((
                        asm::neon_three_same(q, u, size, 0b10000, rm, rn, rd),
                        "add/sub",
                    ));
                }
            }
        }
        for size in 0..3 {
            for (rm, rn, rd) in regs {
                words.push((
                    asm::neon_three_same(q, false, size, 0b10011, rm, rn, rd),
                    "mul",
                ));
                for u in [false, true] {
                    words.push((asm::neon_three_same(q, u, size, 0b01100, rm, rn, rd), "max"));
                    words.push((asm::neon_three_same(q, u, size, 0b01101, rm, rn, rd), "min"));
                }
            }
        }
        for size in 0..4 {
            if !q && size == 3 {
                continue;
            }
            for (rn, rd) in [(1, 0), (31, 15), (5, 5)] {
                for u in [false, true] {
                    words.push((
                        asm::neon_two_misc(q, u, size, 0b01011, rn, rd),
                        if u { "neg" } else { "abs" },
                    ));
                    words.push((
                        asm::neon_two_misc(q, u, size, 0b00111, rn, rd),
                        if u { "sqneg" } else { "sqabs" },
                    ));
                }
            }
        }
    }
    // Saturating three-same arithmetic and the multiply-accumulate forms, every lane
    // width, `Q`, signedness and aliasing pattern.
    for q in [false, true] {
        for size in 0..4 {
            if !q && size == 3 {
                continue;
            }
            for (rm, rn, rd) in regs {
                for u in [false, true] {
                    words.push((
                        asm::neon_three_same(q, u, size, 0b00001, rm, rn, rd),
                        "sqadd",
                    ));
                    words.push((
                        asm::neon_three_same(q, u, size, 0b00101, rm, rn, rd),
                        "sqsub",
                    ));
                }
            }
        }
        // `MLA`/`MLS` and the pairwise forms have no 64-bit lane encoding in the helper.
        for size in 0..3 {
            for (rm, rn, rd) in regs {
                for u in [false, true] {
                    words.push((
                        asm::neon_three_same(q, u, size, 0b10010, rm, rn, rd),
                        "mla/mls",
                    ));
                    words.push((
                        asm::neon_three_same(q, u, size, 0b10100, rm, rn, rd),
                        "maxp",
                    ));
                    words.push((
                        asm::neon_three_same(q, u, size, 0b10101, rm, rn, rd),
                        "minp",
                    ));
                }
            }
        }
    }
    words
}

#[test]
fn covered_words_match_the_helper() {
    let mut rng = Rng(0x5eed_0f17_0000_0001);
    let words = covered();
    assert!(words.len() > 900);
    for (w, _) in &words {
        check(*w, &mut rng);
    }
}

#[test]
fn known_encodings_are_lifted() {
    // Sanity check the test assembler against hand-known words.
    assert_eq!(asm::neon_logic(true, false, 0, 2, 1, 0), 0x4e22_1c20); // and v0.16b, v1.16b, v2.16b
    assert_eq!(asm::neon_logic(true, false, 2, 2, 1, 0), 0x4ea2_1c20); // orr v0.16b, v1.16b, v2.16b
    assert_eq!(asm::neon_logic(true, true, 0, 2, 1, 0), 0x6e22_1c20); // eor v0.16b, v1.16b, v2.16b
    assert_eq!(
        asm::neon_shift_imm(true, false, 11, 0b01010, 1, 0),
        0x4f0b_5420
    ); // shl v0.16b, v1.16b, #3
    assert_eq!(
        asm::neon_shift_imm(true, true, 13, 0b00000, 1, 0),
        0x6f0d_0420
    ); // ushr v0.16b, v1.16b, #3
    assert_eq!(
        asm::neon_shift_imm(true, false, 13, 0b00000, 1, 0),
        0x4f0d_0420
    ); // sshr v0.16b, v1.16b, #3
    assert_eq!(asm::neon_copy(true, false, 7, 0, 1, 0), 0x4e07_0420); // dup v0.16b, v1.b[3]
    assert_eq!(asm::neon_copy(true, true, 5, 5, 1, 0), 0x6e05_2c20); // ins v0.b[2], v1.b[5]
    assert_eq!(
        asm::neon_three_same(true, false, 2, 0b10000, 17, 1, 1),
        0x4eb1_8421
    ); // add v1.4s, v1.4s, v17.4s
    assert_eq!(
        asm::neon_three_same(true, false, 2, 0b10011, 2, 1, 0),
        0x4ea2_9c20
    ); // mul v0.4s, v1.4s, v2.4s
    assert_eq!(
        asm::neon_three_same(true, false, 0, 0b01100, 2, 1, 0),
        0x4e22_6420
    ); // smax v0.16b, v1.16b, v2.16b
    assert_eq!(
        asm::neon_two_misc(true, false, 0, 0b01011, 1, 0),
        0x4e20_b820
    );
    // abs v0.16b, v1.16b
    assert_eq!(
        asm::neon_two_misc(true, true, 0, 0b01011, 1, 0),
        0x6e20_b820
    );
    // neg v0.16b, v1.16b
    assert_eq!(
        asm::neon_three_same(true, false, 0, 0b00001, 2, 1, 0),
        0x4e22_0c20
    ); // sqadd v0.16b, v1.16b, v2.16b
    assert_eq!(
        asm::neon_three_same(true, false, 2, 0b10010, 2, 1, 0),
        0x4ea2_9420
    ); // mla v0.4s, v1.4s, v2.4s
    assert_eq!(
        asm::neon_three_same(true, false, 1, 0b10100, 2, 1, 0),
        0x4e62_a420
    ); // smaxp v0.8h, v1.8h, v2.8h
    assert_eq!(
        asm::neon_two_misc(true, false, 2, 0b00111, 1, 0),
        0x4ea0_7820
    ); // sqabs v0.4s, v1.4s
    for w in [
        0x4e22_1c20,
        0x4ea2_1c20,
        0x6e22_1c20,
        0x4e62_1c20,
        0x4ee2_1c20,
        0x4f0b_5420,
        0x6f0d_0420,
        0x4f0d_0420,
        0x4e07_0420,
        0x6e05_2c20,
        0x4eb1_8421,
        0x4ea2_9c20,
        0x4e22_6420,
        0x4e20_b820,
        0x6e20_b820,
        0x4e22_0c20,
        0x4ea2_9420,
        0x4e62_a420,
        0x4ea0_7820,
    ] {
        assert!(lifted(w), "{w:#010x} should be lifted");
    }
    for w in [
        asm::neon_three_same(true, true, 0, 0b01100, 2, 1, 0), // umax v0.16b, v1.16b, v2.16b
        asm::neon_three_same(true, false, 0, 0b01101, 2, 1, 0), // smin v0.16b, v1.16b, v2.16b
        asm::neon_three_same(false, true, 1, 0b01101, 2, 1, 0), // umin v0.4h, v1.4h, v2.4h
        asm::neon_two_misc(true, true, 3, 0b01011, 1, 0),      // neg v0.2d, v1.2d
        asm::neon_three_same(true, false, 3, 0b00101, 2, 1, 0), // sqsub v0.2d, v1.2d, v2.2d
        asm::neon_three_same(false, true, 2, 0b10010, 2, 1, 0), // mls v0.2s, v1.2s, v2.2s
        asm::neon_two_misc(true, false, 3, 0b00111, 1, 0),     // sqabs v0.2d, v1.2d
    ] {
        assert!(lifted(w), "{w:#010x} should be lifted");
    }
}

#[test]
fn other_neon_stays_on_the_helper() {
    let words = [
        0x4e02_0020u32,                                       // tbl v0.16b, {v1.16b}, v2.16b
        asm::neon_shift_imm(true, true, 13, 0b01000, 1, 0),   // sri v0.16b, v1.16b, #3
        asm::neon_shift_imm(true, true, 11, 0b01010, 1, 0),   // sli v0.16b, v1.16b, #3
        asm::neon_shift_imm(false, false, 64, 0b00000, 1, 0), // 64-bit lane into a D register
        0x4e28_4800u32,                                       // aes group: not claimed here
        asm::neon_copy(false, false, asm::neon_imm5(64, 0), 0, 1, 0), // dup d0, v1.d[0]
        asm::neon_copy(false, true, asm::neon_imm5(16, 2), 3, 1, 0), // ins v0.h[2], v1.h[3] into a D register
        // Polynomial multiply, and the saturating forms the helper does not implement at
        // 64-bit lanes into a D register.
        asm::neon_three_same(true, true, 0, 0b10011, 2, 1, 0), // pmul v0.16b, v1.16b, v2.16b
        asm::neon_three_same(false, false, 3, 0b00001, 2, 1, 0), // sqadd v0.2d into a D register
        asm::neon_two_misc(false, false, 3, 0b00111, 1, 0),    // sqabs d0, d1 (unallocated)
        // The 64-bit lane arithmetic encodings do not exist (or need `Q` = 0).
        asm::neon_three_same(true, false, 3, 0b01100, 2, 1, 0), // smax v0.2d
        asm::neon_three_same(true, true, 3, 0b01101, 2, 1, 0),  // umin v0.2d
        asm::neon_three_same(true, false, 3, 0b10011, 2, 1, 0), // mul v0.2d
        asm::neon_three_same(true, false, 3, 0b10010, 2, 1, 0), // mla v0.2d
        asm::neon_three_same(true, false, 3, 0b10100, 2, 1, 0), // smaxp v0.2d (helper lacks it)
        asm::neon_three_same(true, true, 3, 0b10101, 2, 1, 0),  // uminp v0.2d (helper lacks it)
        asm::neon_two_misc(false, false, 3, 0b01011, 1, 0),     // abs d0, d1 (unallocated)
    ];
    for w in words {
        assert!(
            !lifted(w),
            "{w:#010x} should not be lifted natively: {:?}",
            lift(0, w).stmts
        );
    }
}

/// Random words inside the arithmetic groups: anything lifted must be something the
/// helper also executes (no over-accepting decoder), and must agree with it.
#[test]
fn random_arithmetic_words_match_the_helper() {
    let mut rng = Rng(0x5137_0000_0000_0001);
    let mut checked = 0;
    for _ in 0..6000 {
        let r = rng.next() as u32;
        let w = match r % 2 {
            0 => (r & !0x9f20_0400) | 0x0e20_0400,
            _ => (r & !0x9f3e_0c00) | 0x0e20_0800,
        };
        let insn = lift(0, w);
        if !lifted(w) || insn.stmts.is_empty() {
            continue;
        }
        assert!(
            kirikae_a64::simd::supported(w),
            "{w:#010x} lifted but the helper rejects it: {:?}",
            insn.stmts
        );
        if !vector_only(&insn) {
            continue;
        }
        check(w, &mut rng);
        checked += 1;
    }
    assert!(
        checked > 500,
        "only {checked} random arithmetic words were liftable"
    );
}

/// Random words inside the three groups this slice touches: whatever is lifted must
/// still match the helper, so an over-accepting decoder cannot slip through.
#[test]
fn random_group_words_match_the_helper() {
    let mut rng = Rng(0xabcd_1234_0000_0001);
    let mut checked = 0;
    for _ in 0..4000 {
        let r = rng.next() as u32;
        let w = match r % 3 {
            0 => (r & !0x9f20_0400) | 0x0e20_0400,
            1 => (r & !0x9f80_0400) | 0x0f00_0400,
            _ => (r & !0x9fe0_8400) | 0x0e00_0400,
        };
        let insn = lift(0, w);
        if !lifted(w)
            || insn.stmts.is_empty()
            || !vector_only(&insn)
            || !kirikae_a64::simd::supported(w)
        {
            continue;
        }
        check(w, &mut rng);
        checked += 1;
    }
    assert!(checked > 100, "only {checked} random words were liftable");
}

/// Reports the before/after helper counts on the synthetic corpus, in the terms
/// `kirikae info` uses, and on a random slice of NEON space.
#[test]
fn helper_count_report() {
    let words = covered();
    let mut families: Vec<&'static str> = Vec::new();
    for (_, f) in &words {
        if !families.contains(f) {
            families.push(f);
        }
    }
    let (mut before, mut after) = (0, 0);
    for (w, f) in &words {
        let helper = kirikae_a64::simd::supported(*w);
        let still = lift(0, *w).stmts.iter().any(|s| matches!(s, Stmt::Simd(_)));
        assert!(helper, "{w:#010x} ({f}) is not supported by the helper");
        assert!(!still, "{w:#010x} ({f}) still calls the helper");
        before += helper as usize;
        after += still as usize;
    }
    let mut rng = Rng(0xc0ff_ee00_0000_0001);
    let sample: Vec<u32> = (0..4000)
        .map(|_| {
            let r = rng.next() as u32;
            ((r | 0b111 << 25) & !(1 << 31)) | 0x0e00_0000
        })
        .collect();
    let sample_before = sample
        .iter()
        .filter(|w| kirikae_a64::simd::supported(**w))
        .count();
    let sample_after = sample
        .iter()
        .filter(|w| {
            lift(0, **w)
                .stmts
                .iter()
                .any(|s| matches!(s, Stmt::Simd(_)))
        })
        .count();
    println!(
        "covered set: {} words, helper before {before}, after {after}",
        words.len()
    );
    for f in &families {
        println!("  {f}: {}", words.iter().filter(|(_, x)| x == f).count());
    }
    println!(
        "random NEON sample: {} words, helper before {sample_before}, after {sample_after}",
        sample.len()
    );
}

#[test]
fn lane_helpers() {
    use kirikae_a64::ops::{lane_keep, lane_keep_left, lane_mask, splat, vshl, vsshr, vushr};
    assert_eq!(lane_mask(8), 0xff);
    assert_eq!(lane_keep(8, 1), 0x7f7f_7f7f_7f7f_7f7f);
    assert_eq!(lane_keep(8, 8), 0);
    assert_eq!(lane_keep(64, 0), u64::MAX);
    assert_eq!(lane_keep(64, 64), 0);
    assert_eq!(lane_keep_left(8, 1), 0xfefe_fefe_fefe_fefe);
    assert_eq!(lane_keep_left(8, 0), u64::MAX);
    assert_eq!(lane_keep_left(64, 63), 1 << 63);
    assert_eq!(vshl(0x0102_0304_0506_0708, 8, 1), 0x0204_0608_0a0c_0e10);
    assert_eq!(vshl(0x0102_0304_0506_0708, 8, 8), 0);
    assert_eq!(vshl(0x9017_56fe_2364_a494, 8, 1), 0x202e_acfc_46c8_4828);
    assert_eq!(vushr(0x8000_0000_0000_0001, 8, 1), 0x4000_0000_0000_0000);
    assert_eq!(vushr(0x8000_0000_0000_0001, 8, 8), 0);
    assert_eq!(vsshr(0x8000_0000_0000_0001, 8, 1), 0xc000_0000_0000_0000);
    assert_eq!(vsshr(0x80ff_0000_0000_8000, 8, 8), 0xffff_0000_0000_ff00);
    assert_eq!(vsshr(0x8000_0000_0000_0001, 64, 63), u64::MAX);
    assert_eq!(splat(0xab, 16), 0x00ab_00ab_00ab_00ab);
    assert_eq!(splat(0xdead_beef, 32), 0xdead_beef_dead_beef);
    assert_eq!(splat(0x1234_5678_9abc_def0, 64), 0x1234_5678_9abc_def0);
    assert_eq!(Op::Vushr.arity(), 3);
    assert_eq!(Op::Splat.name(), "splat");
}
