use std::collections::BTreeSet;

use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::FuncId;
use kirikae_a64::bits::decode_bit_masks;
use kirikae_a64::{Expr, Insn, Op, Stmt};
use kirikae_recomp::clif::{host_isa, Compiler};
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
            0 => [0, 1, u64::MAX, 1 << 63, (1 << 63) - 1, 0xffff_ffff, 0x8000_0000, 0x7fff_ffff][(r >> 8) as usize % 8],
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
    let mut b = JITBuilder::with_isa(host_isa().unwrap(), cranelift_module::default_libcall_names());
    for (name, ptr) in abi::helpers() {
        b.symbol(name, ptr);
    }
    Compiler::new(JITModule::new(b)).unwrap()
}

fn function(addr: u64, case: &Case) -> Function {
    let stmts = vec![Stmt::SetReg(5, Expr::Op(case.op, case.args.clone())), Stmt::Return(Expr::Const(0))];
    Function { entry: addr, blocks: vec![Block { start: addr, insns: vec![Insn { addr, word: 0, stmts }], next: addr + 4 }] }
}

fn run_cases(cases: &[Case]) {
    let mut c = jit();
    let mut ids: Vec<FuncId> = Vec::new();
    let entries: BTreeSet<u64> = (0..cases.len() as u64).map(|i| 0x1000 + 4 * i).collect();
    for (i, case) in cases.iter().enumerate() {
        ids.push(c.compile(&function(0x1000 + 4 * i as u64, case), &entries).unwrap());
    }
    c.module.finalize_definitions().unwrap();
    let mut cpu = vec![0u64; 512];
    for (case, id) in cases.iter().zip(ids) {
        let f: extern "C" fn(*mut u64) = unsafe { std::mem::transmute(c.module.get_finalized_function(id)) };
        for (k, v) in case.inputs.iter().enumerate() {
            cpu[abi::OFF_X / 8 + k] = *v;
        }
        f(cpu.as_mut_ptr());
        let got = cpu[abi::OFF_X / 8 + 5];
        assert_eq!(got, case.expected(), "{:?}({:x?}) inputs {:x?}", case.op, case.args, case.inputs);
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
            cases.push(Case { op, args: regs(op.arity()), inputs });
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
            let Some((wm, tm)) = decode_bit_masks(is64 as u32, imms, immr, false, size as u32) else { continue };
            let (r, s) = (immr as u64, imms as u64);
            let (u, sb, b) = if is64 { (Op::Ubfm64, Op::Sbfm64, Op::Bfm64) } else { (Op::Ubfm32, Op::Sbfm32, Op::Bfm32) };
            cases.push(Case { op: u, args: vec![Expr::Reg(0), c(r), c(wm), c(tm)], inputs });
            cases.push(Case { op: sb, args: vec![Expr::Reg(0), c(r), c(s), c(wm), c(tm)], inputs });
            cases.push(Case { op: b, args: vec![Expr::Reg(0), Expr::Reg(1), c(r), c(wm), c(tm)], inputs });
            let lsb = rng.next() % size;
            let e = if is64 { Op::Extr64 } else { Op::Extr32 };
            cases.push(Case { op: e, args: vec![Expr::Reg(0), Expr::Reg(1), c(lsb)], inputs });
        }
        let cc = rng.next() % 16;
        cases.push(Case { op: Op::Cond, args: vec![Expr::Reg(0), c(cc)], inputs });
        for op in [Op::FlagsAdd64, Op::FlagsAdd32] {
            cases.push(Case { op, args: vec![Expr::Reg(0), Expr::Reg(1), c(1)], inputs });
        }
    }
    run_cases(&cases);
}

fn float(rng: &mut Rng, double: bool) -> u64 {
    let r = rng.next();
    let pick = |l: &[u64]| l[(r >> 8) as usize % l.len()];
    if double {
        match r % 4 {
            0 => pick(&[0, 1 << 63, 0x7ff0 << 48, 0xfff0 << 48, 0x7ff8 << 48, 0x7ff0_0000_0000_0001, 1, 0x8000_0000_0000_0001, 0x3ff0 << 48, 0x0010 << 48]),
            1 => r,
            _ => ((r >> 20) % 0x300 + 0x200) << 52 | (r >> 12) & ((1 << 52) - 1) | (r & 1) << 63,
        }
    } else {
        match r % 4 {
            0 => pick(&[0, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001, 0xffc0_1234, 1, 0x807f_ffff, 0x3f80_0000, 0x0080_0000, 0x4f00_0000, 0xcf00_0001]),
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
            (&[Op::FAdd32, Op::FSub32, Op::FMul32, Op::FDiv32, Op::FMulAdd32, Op::FCmp32, Op::FMax32, Op::FMin32, Op::FMinNm32, Op::FSqrt32, Op::FCvtSD][..], false),
            (&[Op::FAdd64, Op::FSub64, Op::FMul64, Op::FDiv64, Op::FMulAdd64, Op::FCmp64, Op::FMax64, Op::FMin64, Op::FSqrt64, Op::FCvtDS][..], true),
        ] {
            for &op in ops {
                let n = op.arity() - 1;
                let mut inputs = [0u64; 5];
                for x in inputs.iter_mut().take(n) {
                    *x = float(&mut rng, double);
                }
                inputs[n] = fpcr;
                cases.push(Case { op, args: regs(op.arity()), inputs });
            }
        }
        for op in [Op::FAdd32x2, Op::FSub32x2, Op::FMul32x2, Op::FMulAdd32x2] {
            let n = op.arity() - 1;
            let mut inputs = [0u64; 5];
            for x in inputs.iter_mut().take(n) {
                *x = float(&mut rng, false) | float(&mut rng, false) << 32;
            }
            inputs[n] = fpcr;
            cases.push(Case { op, args: regs(op.arity()), inputs });
        }
        let round = [kirikae_a64::simd::fp::Round::Nearest, kirikae_a64::simd::fp::Round::Zero, kirikae_a64::simd::fp::Round::MinusInf][i % 3];
        for (op, double) in [(Op::FToInt32, false), (Op::FToInt64, true)] {
            for (signed, width) in [(true, 32), (false, 32), (true, 64), (false, 64)] {
                let inputs = [float(&mut rng, double), 0, fpcr, 0, 0];
                cases.push(Case { op, args: vec![Expr::Reg(0), Expr::Const(kirikae_a64::ops::fp_mode(round, signed, width)), Expr::Reg(2)], inputs });
            }
        }
        for op in [Op::FFromInt32, Op::FFromInt64] {
            for (signed, width) in [(true, 32), (false, 32), (true, 64), (false, 64)] {
                let inputs = [rng.next(), 0, fpcr, 0, 0];
                cases.push(Case { op, args: vec![Expr::Reg(0), Expr::Const(kirikae_a64::ops::fp_mode(round, signed, width)), Expr::Reg(2)], inputs });
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
        let tiny32 = |r: u64, e: u64| (e << 23 | (r >> 8) & 0x7f_ffff | (r & 1) << 31) & 0xffff_ffff;
        let tiny64 = |r: u64, e: u64| e << 52 | (r >> 8) & ((1 << 52) - 1) | (r & 1) << 63;
        let (e1, e2) = if r >> 60 & 1 == 0 { (1 + r % 4, 1 + (r >> 4) % 4) } else { (40 + r % 50, 40 + (r >> 8) % 50) };
        let (f1, f2, f3) = (rng.next(), rng.next(), rng.next());
        for op in [Op::FAdd32, Op::FSub32, Op::FMul32, Op::FDiv32, Op::FMulAdd32, Op::FCmp32] {
            let inputs = if op == Op::FMulAdd32 { [tiny32(f3, 1 + r % 3), tiny32(f1, e1), tiny32(f2, e2), fpcr, 0] } else { [tiny32(f1, e1), tiny32(f2, e2 + 64 * (op == Op::FDiv32) as u64), fpcr, 0, 0] };
            cases.push(Case { op, args: regs(op.arity()), inputs });
        }
        let (d1, d2) = if r >> 61 & 1 == 0 { (1 + r % 4, 1 + (r >> 4) % 4) } else { (400 + r % 300, 400 + (r >> 8) % 300) };
        let narrow = tiny64(f1, 895 + r % 5);
        cases.push(Case { op: Op::FCvtDS, args: regs(2), inputs: [narrow, fpcr, 0, 0, 0] });
        cases.push(Case { op: Op::FCvtSD, args: regs(2), inputs: [tiny32(f2, r % 3), fpcr, 0, 0, 0] });
        for op in [Op::FSqrt32, Op::FMax32, Op::FMin32] {
            let inputs = [tiny32(f1, r % 3), tiny32(f2, (r >> 3) % 3), fpcr, 0, 0];
            let inputs = if op == Op::FSqrt32 { [inputs[0] & 0x7fff_ffff, fpcr, 0, 0, 0] } else { inputs };
            cases.push(Case { op, args: regs(op.arity()), inputs });
        }
        for op in [Op::FAdd64, Op::FSub64, Op::FMul64, Op::FDiv64, Op::FMulAdd64, Op::FCmp64] {
            let inputs = if op == Op::FMulAdd64 { [tiny64(f3, 1 + r % 3), tiny64(f1, d1), tiny64(f2, d2), fpcr, 0] } else { [tiny64(f1, d1), tiny64(f2, d2 + 1000 * (op == Op::FDiv64) as u64), fpcr, 0, 0] };
            cases.push(Case { op, args: regs(op.arity()), inputs });
        }
        for op in [Op::FAdd32x2, Op::FSub32x2, Op::FMul32x2, Op::FMulAdd32x2] {
            let n = op.arity() - 1;
            let mut inputs = [0u64; 5];
            for (k, x) in inputs.iter_mut().take(n).enumerate() {
                let (a, b) = (rng.next(), rng.next());
                let e = if k == 0 && op == Op::FMulAdd32x2 { 1 + a % 3 } else { [e1, e2][k % 2] };
                *x = tiny32(a, e) | tiny32(b, e) << 32;
            }
            inputs[n] = fpcr;
            cases.push(Case { op, args: regs(op.arity()), inputs });
        }
    }
    run_cases(&cases);
}
