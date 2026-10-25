use std::collections::BTreeSet;

use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::FuncId;
use kirikae_a64::{Expr, Insn, Op, Stmt, SysReg};
use kirikae_recomp::clif::{Compiler, host_isa};
use kirikae_recomp::{Block, Function};
use kirikae_rt::abi;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const COUNTER: u8 = 28;
const TARGET: u8 = 27;
const REGS: u64 = 8;
const CALLEE: u64 = 0x1000_0000;
const OUTSIDE: u64 = 0x2000_0000;
const MEM_BYTES: usize = 4096;
const SIMD: &[u32] = &[
    0x1e62_2820,
    0x9e66_0023,
    0x9e67_0082,
    0x1e62_2020,
    0x1e62_0c23,
    0x0e0c_3c25,
    0x4ea2_8420,
    0x9e78_0026,
];

fn c(v: u64) -> Expr {
    Expr::Const(v)
}

fn op(o: Op, args: Vec<Expr>) -> Expr {
    Expr::Op(o, args)
}

fn addr(r: &mut Rng, tmps: u8) -> Expr {
    op(Op::And, vec![expr(r, 1, tmps), c(0x7f8)])
}

fn leaf(r: &mut Rng, tmps: u8) -> Expr {
    match r.below(9) {
        0 => c(r.next()),
        1 => c(r.below(16)),
        2 => Expr::Sp,
        3 => Expr::Nzcv,
        4 => Expr::V {
            reg: r.below(4) as u8,
            hi: r.below(2) == 1,
        },
        5 if tmps > 0 => Expr::Tmp(r.below(tmps as u64) as u8),
        6 => Expr::SysReg(SysReg::TpidrEl0),
        _ => Expr::Reg(r.below(REGS) as u8),
    }
}

fn expr(r: &mut Rng, depth: u32, tmps: u8) -> Expr {
    if depth == 0 || r.below(3) == 0 {
        return leaf(r, tmps);
    }
    let mut sub = || expr(r, depth - 1, tmps);
    let (a, b, d) = (sub(), sub(), sub());
    match r.below(10) {
        0 => op(Op::Add, vec![a, b]),
        1 => op(Op::Sub, vec![a, b]),
        2 => op(Op::Xor, vec![a, b]),
        3 => op(Op::Mul, vec![a, b]),
        4 => op(Op::Lsr64, vec![a, b]),
        5 => op(Op::FlagsAdd64, vec![a, b, d]),
        6 => op(Op::Select, vec![op(Op::Bit, vec![a, c(3)]), b, d]),
        7 => op(Op::Cond, vec![Expr::Nzcv, c(r.below(16))]),
        8 => Expr::Load {
            size: [1, 2, 4, 8][r.below(4) as usize],
            addr: Box::new(addr(r, tmps)),
        },
        _ => op(Op::Umulh, vec![a, b]),
    }
}

fn body(r: &mut Rng) -> Vec<Stmt> {
    let mut out = Vec::new();
    let mut tmps = 0;
    for _ in 0..1 + r.below(3) {
        let e = expr(r, 2, tmps);
        out.push(match r.below(9) {
            0 if tmps < 8 => {
                tmps += 1;
                Stmt::Let(tmps - 1, e)
            }
            1 => Stmt::SetSp(e),
            2 => Stmt::SetNzcv(op(Op::And, vec![e, c(0xf000_0000)])),
            3 => Stmt::SetV {
                reg: r.below(4) as u8,
                hi: r.below(2) == 1,
                value: e,
            },
            4 => Stmt::Store {
                size: [1, 2, 4, 8][r.below(4) as usize],
                addr: addr(r, tmps),
                value: e,
            },
            5 if r.below(4) == 0 => Stmt::SetSysReg(SysReg::TpidrEl0, e),
            6 => Stmt::Simd(SIMD[r.below(SIMD.len() as u64) as usize]),
            _ => Stmt::SetReg(r.below(REGS) as u8, e),
        });
    }
    out
}

fn random_function(r: &mut Rng, entry: u64) -> Function {
    let n = 2 + r.below(7) as usize;
    let lens: Vec<u64> = (0..n).map(|_| 1 + r.below(5)).collect();
    let mut starts = vec![entry];
    for len in &lens {
        starts.push(starts.last().unwrap() + 4 * len + 4);
    }
    let mut blocks = Vec::new();
    for i in 0..n {
        let start = starts[i];
        let alive = op(Op::Ne, vec![Expr::Reg(COUNTER), c(0)]);
        let decrement = Stmt::SetReg(COUNTER, op(Op::Sub, vec![Expr::Reg(COUNTER), alive]));
        let mut insns = vec![Insn {
            addr: start,
            word: 0,
            stmts: vec![decrement],
        }];
        for k in 0..lens[i] {
            insns.push(Insn {
                addr: start + 4 + 4 * k,
                word: 0,
                stmts: body(r),
            });
        }
        let last = insns.last_mut().unwrap();
        let next = last.addr + 4;
        let any = r.below(n as u64) as usize;
        let forward = starts[(i + 1 + r.below((n - i) as u64) as usize).min(n - 1)];
        if i == n - 1 {
            last.stmts.push(Stmt::Return(Expr::Reg(30)));
        } else {
            match r.below(8) {
                0 | 1 => {
                    let cond = op(Op::Bit, vec![expr(r, 1, 0), c(r.below(64))]);
                    let alive = op(Op::Ne, vec![Expr::Reg(COUNTER), c(0)]);
                    let cond = if starts[any] <= start {
                        op(Op::And, vec![cond, alive])
                    } else {
                        cond
                    };
                    last.stmts.push(Stmt::JumpIf(cond, starts[any]));
                }
                2 => last.stmts.push(Stmt::Jump(forward)),
                5 => last.stmts.push(Stmt::JumpIf(
                    op(Op::Bit, vec![expr(r, 1, 0), c(r.below(64))]),
                    OUTSIDE,
                )),
                6 if r.below(3) == 0 => last.stmts.push(Stmt::Jump(OUTSIDE)),
                3 => {
                    last.stmts.push(Stmt::SetReg(30, c(next)));
                    last.stmts.push(Stmt::Call(CALLEE));
                }
                4 => {
                    let target = if r.below(4) == 0 { OUTSIDE } else { forward };
                    last.stmts.push(Stmt::SetReg(TARGET, c(target)));
                    last.stmts.push(Stmt::JumpIndirect(Expr::Reg(TARGET)));
                }
                _ => {}
            }
        }
        blocks.push(Block {
            start,
            insns,
            next: starts[i + 1],
        });
    }
    Function { entry, blocks }
}

fn callee() -> Function {
    let stmts = vec![
        Stmt::SetReg(
            0,
            op(
                Op::Add,
                vec![op(Op::Mul, vec![Expr::Reg(0), c(3)]), Expr::Reg(1)],
            ),
        ),
        Stmt::SetReg(2, op(Op::Xor, vec![Expr::Reg(2), Expr::Sp])),
        Stmt::SetV {
            reg: 1,
            hi: true,
            value: op(Op::Add, vec![Expr::V { reg: 1, hi: false }, c(5)]),
        },
        Stmt::SetNzcv(c(0x6000_0000)),
        Stmt::SetSp(op(Op::Sub, vec![Expr::Sp, c(16)])),
        Stmt::Return(Expr::Reg(30)),
    ];
    Function {
        entry: CALLEE,
        blocks: vec![Block {
            start: CALLEE,
            insns: vec![Insn {
                addr: CALLEE,
                word: 0,
                stmts,
            }],
            next: CALLEE + 4,
        }],
    }
}

fn jit(fast_regs: bool) -> Compiler<JITModule> {
    let mut b = JITBuilder::with_isa(
        host_isa().unwrap(),
        cranelift_module::default_libcall_names(),
    );
    for (name, ptr) in abi::helpers() {
        b.symbol(name, ptr);
    }
    let mut c = Compiler::new(JITModule::new(b)).unwrap();
    c.set_fast_regs(fast_regs);
    c
}

struct State {
    cpu: Vec<u64>,
    mem: Vec<u8>,
}

impl State {
    fn new(r: &mut Rng) -> State {
        let mut cpu = vec![0u64; 512];
        for v in &mut cpu[abi::OFF_X / 8..abi::OFF_NZCV / 8 + 1] {
            *v = r.next();
        }
        cpu[abi::OFF_NZCV / 8] &= 0xf000_0000;
        cpu[abi::OFF_X / 8 + COUNTER as usize] = 12;
        cpu[abi::OFF_X / 8 + 30] = 0xdead_0000;
        let mem = (0..MEM_BYTES).map(|_| r.next() as u8).collect();
        State { cpu, mem }
    }

    fn run(&mut self, f: extern "C" fn(*mut u64)) {
        let base = self.cpu.as_mut_ptr() as *mut u8;
        // SAFETY: the offsets are inside the 4 KiB buffer.
        unsafe {
            (base.add(abi::OFF_MEM) as *mut *mut u8).write_unaligned(self.mem.as_mut_ptr());
            (base.add(abi::OFF_BUDGET) as *mut u32).write_unaligned(1 << 30);
            base.add(abi::OFF_HALTED).write(0);
        }
        f(self.cpu.as_mut_ptr());
    }

    fn observed(&self) -> (Vec<u64>, Vec<u8>) {
        (
            self.cpu[abi::OFF_X / 8..abi::OFF_NZCV / 8 + 2].to_vec(),
            self.mem.clone(),
        )
    }
}

#[test]
fn fast_registers_match_memory_registers() {
    assert!(SIMD.iter().all(|w| kirikae_a64::simd::supported(*w)));
    let mut r = Rng(0x0dd5_eed5_1234_5678);
    let funcs: Vec<Function> = (0..400)
        .map(|i| random_function(&mut r, 0x1000 + 0x1000 * i))
        .collect();
    let mut entries: BTreeSet<u64> = funcs.iter().map(|f| f.entry).collect();
    entries.insert(CALLEE);

    let compile = |fast| {
        let mut c = jit(fast);
        c.compile(&callee(), &entries, &|_| true).unwrap();
        let ids: Vec<FuncId> = funcs
            .iter()
            .map(|f| c.compile(f, &entries, &|_| true).unwrap())
            .collect();
        c.module.finalize_definitions().unwrap();
        let ptrs: Vec<extern "C" fn(*mut u64)> = ids
            .iter()
            .map(|id| unsafe { std::mem::transmute(c.module.get_finalized_function(*id)) })
            .collect();
        (c, ptrs)
    };
    let (_slow_module, slow) = compile(false);
    let (_fast_module, fast) = compile(true);

    for (i, f) in funcs.iter().enumerate() {
        for run in 0..4 {
            let mut a = State::new(&mut r);
            let mut b = State {
                cpu: a.cpu.clone(),
                mem: a.mem.clone(),
            };
            a.run(slow[i]);
            b.run(fast[i]);
            assert!(
                a.observed() == b.observed(),
                "function {i} run {run} differs:\n{f:#x?}"
            );
        }
    }
}
