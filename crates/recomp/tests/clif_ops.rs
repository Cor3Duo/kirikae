use std::collections::BTreeSet;

use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::FuncId;
use kirikae_a64::bits::decode_bit_masks;
use kirikae_a64::{Expr, Insn, Op, Stmt};
use kirikae_recomp::clif::{Compiler, host_isa};
use kirikae_recomp::{Block, Function};
use kirikae_rt::abi;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        match r % 8 {
            0 => [
                0,
                1,
                u64::MAX,
                1 << 63,
                (1 << 63) - 1,
                0xffff_ffff,
                0x8000_0000,
                0x7fff_ffff,
            ][(r >> 8) as usize % 8],
            1 => r & 0xff,
            _ => r,
        }
    }
}

struct Case {
    op: Op,
    args: Vec<Expr>,
    inputs: [u64; 5],
}

impl Case {
    fn expected(&self) -> u64 {
        let vals: Vec<u64> = self
            .args
            .iter()
            .map(|a| match a {
                Expr::Const(v) => *v,
                Expr::Reg(n) => self.inputs[*n as usize],
                _ => unreachable!(),
            })
            .collect();
        self.op.eval(&vals)
    }
}

fn jit() -> Compiler<JITModule> {
    let mut b = JITBuilder::with_isa(
        host_isa().unwrap(),
        cranelift_module::default_libcall_names(),
    );
    for (name, ptr) in abi::helpers() {
        b.symbol(name, ptr);
    }
    Compiler::new(JITModule::new(b)).unwrap()
}

fn function(addr: u64, case: &Case) -> Function {
    let stmts = vec![
        Stmt::SetReg(5, Expr::Op(case.op, case.args.clone())),
        Stmt::Return(Expr::Const(0)),
    ];
    Function {
        entry: addr,
        blocks: vec![Block {
            start: addr,
            insns: vec![Insn {
                addr,
                word: 0,
                stmts,
            }],
            next: addr + 4,
        }],
    }
}

fn run_cases(cases: &[Case]) {
    let mut c = jit();
    let mut ids: Vec<FuncId> = Vec::new();
    let entries: BTreeSet<u64> = (0..cases.len() as u64).map(|i| 0x1000 + 4 * i).collect();
    for (i, case) in cases.iter().enumerate() {
        ids.push(
            c.compile(&function(0x1000 + 4 * i as u64, case), &entries, &|_| true)
                .unwrap(),
        );
    }
    c.module.finalize_definitions().unwrap();
    let mut cpu = vec![0u64; 512];
    for (case, id) in cases.iter().zip(ids) {
        let f: extern "C" fn(*mut u64) =
            unsafe { std::mem::transmute(c.module.get_finalized_function(id)) };
        for (k, v) in case.inputs.iter().enumerate() {
            cpu[abi::OFF_X / 8 + k] = *v;
        }
        f(cpu.as_mut_ptr());
        let got = cpu[abi::OFF_X / 8 + 5];
        assert_eq!(
            got,
            case.expected(),
            "{:?}({:x?}) inputs {:x?}",
            case.op,
            case.args,
            case.inputs
        );
    }
}

fn regs(n: usize) -> Vec<Expr> {
    (0..n as u8).map(Expr::Reg).collect()
}

#[test]
fn register_operands_match_shared_semantics() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut cases = Vec::new();
    for &op in Op::ALL {
        for _ in 0..64 {
            let inputs = [rng.next(), rng.next(), rng.next(), rng.next(), rng.next()];
            cases.push(Case {
                op,
                args: regs(op.arity()),
                inputs,
            });
        }
    }
    run_cases(&cases);
}

#[test]
fn immediate_operands_match_shared_semantics() {
    let mut rng = Rng(0xfeed_beef_cafe_f00d);
    let mut cases = Vec::new();
    let c = Expr::Const;
    for _ in 0..400 {
        let inputs = [rng.next(), rng.next(), rng.next(), rng.next(), rng.next()];
        for is64 in [true, false] {
            let size = if is64 { 64 } else { 32 };
            let (immr, imms) = ((rng.next() % size) as u32, (rng.next() % size) as u32);
            let Some((wm, tm)) = decode_bit_masks(is64 as u32, imms, immr, false, size as u32)
            else {
                continue;
            };
            let (r, s) = (immr as u64, imms as u64);
            let (u, sb, b) = if is64 {
                (Op::Ubfm64, Op::Sbfm64, Op::Bfm64)
            } else {
                (Op::Ubfm32, Op::Sbfm32, Op::Bfm32)
            };
            cases.push(Case {
                op: u,
                args: vec![Expr::Reg(0), c(r), c(wm), c(tm)],
                inputs,
            });
            cases.push(Case {
                op: sb,
                args: vec![Expr::Reg(0), c(r), c(s), c(wm), c(tm)],
                inputs,
            });
            cases.push(Case {
                op: b,
                args: vec![Expr::Reg(0), Expr::Reg(1), c(r), c(wm), c(tm)],
                inputs,
            });
            let lsb = rng.next() % size;
            let e = if is64 { Op::Extr64 } else { Op::Extr32 };
            cases.push(Case {
                op: e,
                args: vec![Expr::Reg(0), Expr::Reg(1), c(lsb)],
                inputs,
            });
        }
        let cc = rng.next() % 16;
        cases.push(Case {
            op: Op::Cond,
            args: vec![Expr::Reg(0), c(cc)],
            inputs,
        });
        for op in [Op::FlagsAdd64, Op::FlagsAdd32] {
            cases.push(Case {
                op,
                args: vec![Expr::Reg(0), Expr::Reg(1), c(1)],
                inputs,
            });
        }
    }
    run_cases(&cases);
}

/// The vector ops with immediate lane parameters, exercised at every lane width the
/// shared semantics accepts (and a few it rejects). `register_operands` only feeds
/// registers there, so the folded constants of these lowerings need their own cases.
#[test]
fn vector_immediates_match_shared_semantics() {
    let mut rng = Rng(0x1111_2222_3333_4444);
    let mut cases = Vec::new();
    let c = Expr::Const;
    for op in [
        Op::Vsat,
        Op::VsatFlag,
        Op::Vmla,
        Op::Vmls,
        Op::Vpmax,
        Op::Vpmin,
    ] {
        for esize in [1u64, 2, 4, 7, 8, 16, 32, 64, 65, 128] {
            for mode in 0..=7u64 {
                for _ in 0..8 {
                    let inputs = [rng.next(), rng.next(), rng.next(), rng.next(), rng.next()];
                    let args = match op {
                        Op::Vmla | Op::Vmls => {
                            vec![Expr::Reg(0), Expr::Reg(1), Expr::Reg(2), c(esize)]
                        }
                        _ => vec![Expr::Reg(0), Expr::Reg(1), c(esize), c(mode)],
                    };
                    cases.push(Case { op, args, inputs });
                }
            }
        }
    }
    run_cases(&cases);
}

fn float(rng: &mut Rng, double: bool) -> u64 {
    let r = rng.next();
    let pick = |l: &[u64]| l[(r >> 8) as usize % l.len()];
    if double {
        match r % 4 {
            0 => pick(&[
                0,
                1 << 63,
                0x7ff0 << 48,
                0xfff0 << 48,
                0x7ff8 << 48,
                0x7ff0_0000_0000_0001,
                1,
                0x8000_0000_0000_0001,
                0x3ff0 << 48,
                0x0010 << 48,
            ]),
            1 => r,
            _ => ((r >> 20) % 0x300 + 0x200) << 52 | (r >> 12) & ((1 << 52) - 1) | (r & 1) << 63,
        }
    } else {
        match r % 4 {
            0 => pick(&[
                0,
                0x8000_0000,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
                0x7f80_0001,
                0xffc0_1234,
                1,
                0x807f_ffff,
                0x3f80_0000,
                0x0080_0000,
                0x4f00_0000,
                0xcf00_0001,
            ]),
            1 => r & 0xffff_ffff,
            _ => ((r >> 20) % 0x60 + 0x40) << 23 | (r >> 32) & 0x7f_ffff | (r & 1) << 31,
        }
    }
}

#[test]
fn fp_operands_match_shared_semantics() {
    let mut rng = Rng(0x5eed_f10a_7000_0001);
    let fpcrs = [0u64, 0, 1 << 24, 1 << 25, 3 << 24, 1 << 22, 3 << 22];
    let mut cases = Vec::new();
    for i in 0..3000 {
        let fpcr = fpcrs[i % fpcrs.len()];
        for (ops, double) in [
            (
                &[
                    Op::FAdd32,
                    Op::FSub32,
                    Op::FMul32,
                    Op::FDiv32,
                    Op::FMulAdd32,
                    Op::FCmp32,
                    Op::FMax32,
                    Op::FMin32,
                    Op::FMinNm32,
                    Op::FSqrt32,
                    Op::FCvtSD,
                ][..],
                false,
            ),
            (
                &[
                    Op::FAdd64,
                    Op::FSub64,
                    Op::FMul64,
                    Op::FDiv64,
                    Op::FMulAdd64,
                    Op::FCmp64,
                    Op::FMax64,
                    Op::FMin64,
                    Op::FSqrt64,
                    Op::FCvtDS,
                ][..],
                true,
            ),
        ] {
            for &op in ops {
                let n = op.arity() - 1;
                let mut inputs = [0u64; 5];
                for x in inputs.iter_mut().take(n) {
                    *x = float(&mut rng, double);
                }
                inputs[n] = fpcr;
                cases.push(Case {
                    op,
                    args: regs(op.arity()),
                    inputs,
                });
            }
        }
        for op in [Op::FAdd32x2, Op::FSub32x2, Op::FMul32x2, Op::FMulAdd32x2] {
            let n = op.arity() - 1;
            let mut inputs = [0u64; 5];
            for x in inputs.iter_mut().take(n) {
                *x = float(&mut rng, false) | float(&mut rng, false) << 32;
            }
            inputs[n] = fpcr;
            cases.push(Case {
                op,
                args: regs(op.arity()),
                inputs,
            });
        }
        let round = [
            kirikae_a64::simd::fp::Round::Nearest,
            kirikae_a64::simd::fp::Round::Zero,
            kirikae_a64::simd::fp::Round::MinusInf,
        ][i % 3];
        for (op, double) in [(Op::FToInt32, false), (Op::FToInt64, true)] {
            for (signed, width) in [(true, 32), (false, 32), (true, 64), (false, 64)] {
                let inputs = [float(&mut rng, double), 0, fpcr, 0, 0];
                cases.push(Case {
                    op,
                    args: vec![
                        Expr::Reg(0),
                        Expr::Const(kirikae_a64::ops::fp_mode(round, signed, width)),
                        Expr::Reg(2),
                    ],
                    inputs,
                });
            }
        }
        for op in [Op::FFromInt32, Op::FFromInt64] {
            for (signed, width) in [(true, 32), (false, 32), (true, 64), (false, 64)] {
                let inputs = [rng.next(), 0, fpcr, 0, 0];
                cases.push(Case {
                    op,
                    args: vec![
                        Expr::Reg(0),
                        Expr::Const(kirikae_a64::ops::fp_mode(round, signed, width)),
                        Expr::Reg(2),
                    ],
                    inputs,
                });
            }
        }
    }
    run_cases(&cases);
}

#[test]
fn flush_to_zero_edges_match_shared_semantics() {
    let mut rng = Rng(0x0f2e_d6e5_0000_0001);
    let mut cases = Vec::new();
    for i in 0..600 {
        let fpcr = [1u64 << 24, 3 << 24, 0][i % 3];
        let r = rng.next();
        let tiny32 =
            |r: u64, e: u64| (e << 23 | (r >> 8) & 0x7f_ffff | (r & 1) << 31) & 0xffff_ffff;
        let tiny64 = |r: u64, e: u64| e << 52 | (r >> 8) & ((1 << 52) - 1) | (r & 1) << 63;
        let (e1, e2) = if r >> 60 & 1 == 0 {
            (1 + r % 4, 1 + (r >> 4) % 4)
        } else {
            (40 + r % 50, 40 + (r >> 8) % 50)
        };
        let (f1, f2, f3) = (rng.next(), rng.next(), rng.next());
        for op in [
            Op::FAdd32,
            Op::FSub32,
            Op::FMul32,
            Op::FDiv32,
            Op::FMulAdd32,
            Op::FCmp32,
        ] {
            let inputs = if op == Op::FMulAdd32 {
                [
                    tiny32(f3, 1 + r % 3),
                    tiny32(f1, e1),
                    tiny32(f2, e2),
                    fpcr,
                    0,
                ]
            } else {
                [
                    tiny32(f1, e1),
                    tiny32(f2, e2 + 64 * (op == Op::FDiv32) as u64),
                    fpcr,
                    0,
                    0,
                ]
            };
            cases.push(Case {
                op,
                args: regs(op.arity()),
                inputs,
            });
        }
        let (d1, d2) = if r >> 61 & 1 == 0 {
            (1 + r % 4, 1 + (r >> 4) % 4)
        } else {
            (400 + r % 300, 400 + (r >> 8) % 300)
        };
        let narrow = tiny64(f1, 895 + r % 5);
        cases.push(Case {
            op: Op::FCvtDS,
            args: regs(2),
            inputs: [narrow, fpcr, 0, 0, 0],
        });
        cases.push(Case {
            op: Op::FCvtSD,
            args: regs(2),
            inputs: [tiny32(f2, r % 3), fpcr, 0, 0, 0],
        });
        for op in [Op::FSqrt32, Op::FMax32, Op::FMin32] {
            let inputs = [tiny32(f1, r % 3), tiny32(f2, (r >> 3) % 3), fpcr, 0, 0];
            let inputs = if op == Op::FSqrt32 {
                [inputs[0] & 0x7fff_ffff, fpcr, 0, 0, 0]
            } else {
                inputs
            };
            cases.push(Case {
                op,
                args: regs(op.arity()),
                inputs,
            });
        }
        for op in [
            Op::FAdd64,
            Op::FSub64,
            Op::FMul64,
            Op::FDiv64,
            Op::FMulAdd64,
            Op::FCmp64,
        ] {
            let inputs = if op == Op::FMulAdd64 {
                [
                    tiny64(f3, 1 + r % 3),
                    tiny64(f1, d1),
                    tiny64(f2, d2),
                    fpcr,
                    0,
                ]
            } else {
                [
                    tiny64(f1, d1),
                    tiny64(f2, d2 + 1000 * (op == Op::FDiv64) as u64),
                    fpcr,
                    0,
                    0,
                ]
            };
            cases.push(Case {
                op,
                args: regs(op.arity()),
                inputs,
            });
        }
        for op in [Op::FAdd32x2, Op::FSub32x2, Op::FMul32x2, Op::FMulAdd32x2] {
            let n = op.arity() - 1;
            let mut inputs = [0u64; 5];
            for (k, x) in inputs.iter_mut().take(n).enumerate() {
                let (a, b) = (rng.next(), rng.next());
                let e = if k == 0 && op == Op::FMulAdd32x2 {
                    1 + a % 3
                } else {
                    [e1, e2][k % 2]
                };
                *x = tiny32(a, e) | tiny32(b, e) << 32;
            }
            inputs[n] = fpcr;
            cases.push(Case {
                op,
                args: regs(op.arity()),
                inputs,
            });
        }
    }
    run_cases(&cases);
}

fn neon_state(rng: &mut Rng) -> kirikae_a64::simd::oracle::State {
    use kirikae_a64::simd::oracle::{State, mem_pattern};
    let mut v = [0u128; 32];
    for q in v.iter_mut() {
        *q = match rng.next() % 4 {
            0 => 0,
            1 => u128::MAX,
            _ => rng.next() as u128 | (rng.next() as u128) << 64,
        };
    }
    State {
        x: [0; 31],
        sp: 0,
        v,
        nzcv: (rng.next() & 0xf) << 28,
        fpcr: 0,
        fpsr: (rng.next() & 0x1f) << 27,
        mem: mem_pattern(),
    }
}

/// A representative word per lifted form: the interpreter path is covered
/// exhaustively in `crates/a64/tests/neon_lift.rs`; here the point is the codegen.
fn neon_words() -> Vec<u32> {
    use kirikae_a64::asm;
    let mut words = Vec::new();
    for q in [false, true] {
        for (es, shifts) in [(8u32, [1, 7]), (16, [1, 15]), (32, [1, 31]), (64, [1, 63])] {
            if !q && es == 64 {
                continue;
            }
            for n in shifts {
                words.push(asm::neon_shift_imm(q, false, 2 * es - n, 0b00000, 1, 0));
                words.push(asm::neon_shift_imm(q, true, 2 * es - n, 0b00000, 1, 0));
                words.push(asm::neon_shift_imm(q, false, es + n - 1, 0b01010, 2, 0));
            }
            for index in [0, 1] {
                words.push(asm::neon_copy(q, false, asm::neon_imm5(es, index), 0, 1, 0));
            }
        }
        for u in [false, true] {
            for size in 0..3 {
                words.push(asm::neon_logic(q, u, size, 1, 2, 3));
            }
        }
    }
    for es in [8u32, 16] {
        for (index, src) in [(0, 0), (1, 2), (2, 3)] {
            words.push(asm::neon_copy(
                true,
                true,
                asm::neon_imm5(es, index),
                src << (es.trailing_zeros() - 3),
                3,
                0,
            ));
        }
    }
    // Three-same integer arithmetic and the two-register ABS/NEG forms, including
    // destinations that alias a source.
    for q in [false, true] {
        for size in 0..4 {
            if !q && size == 3 {
                continue;
            }
            for u in [false, true] {
                words.push(asm::neon_three_same(q, u, size, 0b10000, 2, 1, 3));
                words.push(asm::neon_three_same(q, u, size, 0b10000, 1, 1, 1));
            }
        }
        for size in 0..3 {
            for (rm, rn, rd) in [(2, 1, 3), (1, 1, 1)] {
                words.push(asm::neon_three_same(q, false, size, 0b10011, rm, rn, rd));
                for u in [false, true] {
                    words.push(asm::neon_three_same(q, u, size, 0b01100, rm, rn, rd));
                    words.push(asm::neon_three_same(q, u, size, 0b01101, rm, rn, rd));
                }
            }
        }
        for size in 0..4 {
            if !q && size == 3 {
                continue;
            }
            for (rn, rd) in [(1, 3), (1, 1)] {
                for u in [false, true] {
                    words.push(asm::neon_two_misc(q, u, size, 0b01011, rn, rd));
                }
            }
        }
    }
    // Saturating, multiply-accumulate and pairwise forms, including the aliasing cases
    // (`rd` = `rn`, and `rd` = `rm` for the pairwise forms, whose high half reads `rm`).
    for q in [false, true] {
        for size in 0..4 {
            if !q && size == 3 {
                continue;
            }
            for u in [false, true] {
                words.push(asm::neon_three_same(q, u, size, 0b00001, 2, 1, 3));
                words.push(asm::neon_three_same(q, u, size, 0b00101, 2, 1, 0));
                words.push(asm::neon_three_same(q, u, size, 0b00001, 1, 1, 1));
                words.push(asm::neon_two_misc(q, u, size, 0b00111, 1, 3));
                words.push(asm::neon_two_misc(q, u, size, 0b00111, 2, 2));
            }
        }
        for size in 0..3 {
            for u in [false, true] {
                words.push(asm::neon_three_same(q, u, size, 0b10010, 2, 1, 3));
                words.push(asm::neon_three_same(q, u, size, 0b10010, 1, 1, 1));
                words.push(asm::neon_three_same(q, u, size, 0b10100, 2, 1, 3));
                words.push(asm::neon_three_same(q, u, size, 0b10100, 1, 2, 1));
                words.push(asm::neon_three_same(q, u, size, 0b10101, 2, 1, 1));
            }
        }
    }
    words
}

/// Compiles the lifted IR of each word and compares the whole V file (plus the flags)
/// against the runtime helper running the same word on the same state.
#[test]
fn lifted_neon_matches_the_helper() {
    use kirikae_a64::simd::{exec, oracle::State};
    let mut rng = Rng(0x1020_3040_5060_7080);
    let words = neon_words();
    let mut c = jit();
    let mut ids = Vec::new();
    let entries: BTreeSet<u64> = (0..words.len() as u64).map(|i| 0x80_0000 + 4 * i).collect();
    for (i, w) in words.iter().enumerate() {
        let addr = 0x80_0000 + 4 * i as u64;
        let insn = kirikae_a64::lift(addr, *w);
        assert!(
            !insn.stmts.iter().any(|s| matches!(s, Stmt::Simd(_))),
            "{w:#010x} still calls the helper"
        );
        let mut stmts = insn.stmts;
        stmts.push(Stmt::Return(Expr::Const(0)));
        let f = Function {
            entry: addr,
            blocks: vec![Block {
                start: addr,
                insns: vec![Insn {
                    addr,
                    word: *w,
                    stmts,
                }],
                next: addr + 4,
            }],
        };
        ids.push(c.compile(&f, &entries, &|_| true).unwrap());
    }
    c.module.finalize_definitions().unwrap();

    let mut cpu = vec![0u64; 512];
    let mut compared = 0;
    for (w, id) in words.iter().zip(ids) {
        for _ in 0..4 {
            let input = neon_state(&mut rng);
            let mut want: State = input.clone();
            cpu.fill(0);
            for reg in 0..32 {
                cpu[abi::OFF_V / 8 + 2 * reg] = input.v[reg] as u64;
                cpu[abi::OFF_V / 8 + 2 * reg + 1] = (input.v[reg] >> 64) as u64;
            }
            cpu[abi::OFF_NZCV / 8] = input.nzcv;
            cpu[abi::OFF_FPSR / 8] = input.fpsr;
            let f: extern "C" fn(*mut u64) =
                unsafe { std::mem::transmute(c.module.get_finalized_function(id)) };
            f(cpu.as_mut_ptr());
            assert!(exec(&mut want, *w), "{w:#010x} rejected by the helper");
            for reg in 0..32 {
                let got = cpu[abi::OFF_V / 8 + 2 * reg] as u128
                    | (cpu[abi::OFF_V / 8 + 2 * reg + 1] as u128) << 64;
                assert_eq!(
                    got, want.v[reg],
                    "{w:#010x}: v{reg} mismatch (helper {:#034x} vs JIT {:#034x})",
                    want.v[reg], got
                );
            }
            assert_eq!(
                cpu[abi::OFF_NZCV / 8],
                want.nzcv,
                "{w:#010x}: nzcv mismatch"
            );
            assert_eq!(
                cpu[abi::OFF_FPSR / 8],
                want.fpsr,
                "{w:#010x}: fpsr mismatch (QC)"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, words.len() * 4);
}
