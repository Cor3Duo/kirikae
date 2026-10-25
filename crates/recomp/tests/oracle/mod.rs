//! A concrete ARM64 stepper: the independent oracle for jump-table recognition.
//!
//! Unicorn would be the usual tool for this, but it is GPL-2.0 and this workspace does not link
//! GPL code, not even in the test graph without a licence decision (see
//! `docs/plan/findings/jump-tables.md`). The stepper below is the part of an emulator a table
//! dispatch needs: a controlled register file, the image mapped as memory, and one instruction
//! at a time through the same lifter the runtime uses. It executes; it does not infer. An
//! instruction outside the dispatch shapes stops the run, so the oracle never invents a target.
//!
//! It is a test-only module: no crate ships it.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use kirikae_a64::{Expr, Stmt};
use kirikae_loader::Image;
use kirikae_recomp::discover::CodeSpace;

/// Register file and memory writes of one path. Registers are the architectural `x0..x30`;
/// `sp` is separate, as on the hardware.
#[derive(Clone, Default)]
pub struct Cpu {
    pub regs: [u64; 31],
    pub sp: u64,
    pub nzcv: u64,
    writes: BTreeMap<u64, u8>,
}

impl Cpu {
    pub fn new() -> Cpu {
        Cpu::default()
    }

    /// The same register file with `reg` set to `value`, which is how the oracle sweeps the
    /// unknown index of a dispatch.
    pub fn with(&self, reg: u8, value: u64) -> Cpu {
        let mut cpu = self.clone();
        if reg < 31 {
            cpu.regs[reg as usize] = value;
        }
        cpu
    }
}

pub enum Step {
    /// Ran one instruction; the next one is at `pc + 4`.
    Next,
    /// Took a conditional branch to this target. The fall-through is still live.
    Branch(u64),
    /// Took an unconditional branch to this target.
    Jump(u64),
    /// Jumped through a computed destination, which is the value of interest.
    Indirect(u64),
    /// Called through a computed destination, which is the value of interest.
    Call(u64),
    Return,
    /// A shape the oracle does not execute: a call, a syscall, vector state, an unmapped load.
    Unsupported,
}

pub struct Oracle<'a> {
    space: &'a CodeSpace<'a>,
    images: &'a [&'a Image],
}

impl<'a> Oracle<'a> {
    pub fn new(space: &'a CodeSpace<'a>, images: &'a [&'a Image]) -> Oracle<'a> {
        Oracle { space, images }
    }

    fn byte(&self, cpu: &Cpu, addr: u64) -> Option<u8> {
        if let Some(b) = cpu.writes.get(&addr) {
            return Some(*b);
        }
        for m in self.images {
            if addr >= m.base && addr < m.end() {
                return m.memory.get((addr - m.base) as usize).copied();
            }
        }
        None
    }

    fn read(&self, cpu: &Cpu, addr: u64, size: u8) -> Option<u64> {
        let mut value = 0;
        for k in 0..size as u64 {
            value |= (self.byte(cpu, addr.checked_add(k)?)? as u64) << (8 * k);
        }
        Some(value)
    }

    fn eval(&self, cpu: &Cpu, e: &Expr, tmp: &[Option<u64>]) -> Option<u64> {
        Some(match e {
            Expr::Const(v) => *v,
            Expr::Reg(n) => *cpu.regs.get(*n as usize)?,
            Expr::Sp => cpu.sp,
            Expr::Nzcv => cpu.nzcv,
            Expr::Tmp(n) => (*tmp.get(*n as usize)?)?,
            Expr::Op(op, args) => {
                let values: Vec<u64> = args
                    .iter()
                    .map(|a| self.eval(cpu, a, tmp))
                    .collect::<Option<_>>()?;
                op.eval(&values)
            }
            Expr::Load { size, addr } => self.read(cpu, self.eval(cpu, addr, tmp)?, *size)?,
            Expr::SysReg(_) | Expr::V { .. } => return None,
        })
    }

    /// Execute one instruction at `pc`, mutating `cpu` as the hardware would.
    pub fn step(&self, cpu: &mut Cpu, pc: u64) -> Step {
        let Some(insn) = self.space.lift(pc) else {
            return Step::Unsupported;
        };
        let mut tmp: Vec<Option<u64>> = vec![None; 8];
        for s in &insn.stmts {
            match s {
                Stmt::Let(n, e) => {
                    let Some(value) = self.eval(cpu, e, &tmp) else {
                        return Step::Unsupported;
                    };
                    let Some(slot) = tmp.get_mut(*n as usize) else {
                        return Step::Unsupported;
                    };
                    *slot = Some(value);
                }
                Stmt::SetReg(n, e) => {
                    let Some(value) = self.eval(cpu, e, &tmp) else {
                        return Step::Unsupported;
                    };
                    let Some(reg) = cpu.regs.get_mut(*n as usize) else {
                        return Step::Unsupported;
                    };
                    *reg = value;
                }
                Stmt::SetSp(e) => {
                    let Some(value) = self.eval(cpu, e, &tmp) else {
                        return Step::Unsupported;
                    };
                    cpu.sp = value;
                }
                Stmt::SetNzcv(e) => {
                    let Some(value) = self.eval(cpu, e, &tmp) else {
                        return Step::Unsupported;
                    };
                    cpu.nzcv = value;
                }
                Stmt::Store { size, addr, value } => {
                    let (Some(addr), Some(value)) =
                        (self.eval(cpu, addr, &tmp), self.eval(cpu, value, &tmp))
                    else {
                        return Step::Unsupported;
                    };
                    for k in 0..*size as u64 {
                        cpu.writes.insert(addr + k, (value >> (8 * k)) as u8);
                    }
                }
                Stmt::Fence | Stmt::Simd(_) => {}
                Stmt::Jump(t) => return Step::Jump(*t),
                Stmt::JumpIf(cond, t) => {
                    let Some(taken) = self.eval(cpu, cond, &tmp) else {
                        return Step::Unsupported;
                    };
                    if taken != 0 {
                        return Step::Branch(*t);
                    }
                }
                Stmt::JumpIndirect(e) => {
                    let Some(target) = self.eval(cpu, e, &tmp) else {
                        return Step::Unsupported;
                    };
                    return Step::Indirect(target);
                }
                Stmt::Return(_) => return Step::Return,
                Stmt::CallIndirect(e) => {
                    let Some(target) = self.eval(cpu, e, &tmp) else {
                        return Step::Unsupported;
                    };
                    return Step::Call(target);
                }
                Stmt::Call(_)
                | Stmt::Monitor { .. }
                | Stmt::StoreExclusive { .. }
                | Stmt::SetSysReg(..)
                | Stmt::SetV { .. }
                | Stmt::Svc(_)
                | Stmt::Undefined(_) => return Step::Unsupported,
            }
        }
        Step::Next
    }

    /// The destinations the hardware actually takes from `entry` with this state, following
    /// the path as the branch conditions decide it, up to `budget` executed instructions.
    pub fn targets(&self, entry: u64, cpu: &Cpu, budget: usize) -> BTreeSet<u64> {
        let mut out = BTreeSet::new();
        let mut cpu = cpu.clone();
        let mut pc = entry;
        for _ in 0..budget {
            match self.step(&mut cpu, pc) {
                Step::Next => pc = pc.wrapping_add(4),
                Step::Branch(t) | Step::Jump(t) => {
                    out.insert(t);
                    pc = t;
                }
                Step::Indirect(t) | Step::Call(t) => {
                    out.insert(t);
                    break;
                }
                Step::Return | Step::Unsupported => break,
            }
        }
        out
    }

    /// The computed destination of the straight-line run at `start`, or `None` when the path
    /// leaves through a branch or the run is not executable.
    pub fn dispatch(&self, start: u64, cpu: &Cpu, budget: usize) -> Option<u64> {
        let mut cpu = cpu.clone();
        let mut pc = start;
        for _ in 0..budget {
            match self.step(&mut cpu, pc) {
                Step::Next => pc = pc.wrapping_add(4),
                Step::Indirect(t) | Step::Call(t) => return Some(t),
                Step::Branch(_) | Step::Jump(_) | Step::Return | Step::Unsupported => return None,
            }
        }
        None
    }

    /// The start of the straight-line run that ends at `br`: the address after the previous
    /// control instruction. Shared with the detector on purpose; the oracle checks the targets
    /// and the shape, not where the run begins.
    pub fn run_start(&self, br: u64) -> u64 {
        let mut start = br;
        while let Some(prev) = start.checked_sub(4) {
            if br - prev > 4 * 64 {
                break;
            }
            match self.space.lift(prev).map(|i| i.control()) {
                Some(kirikae_a64::Control::Fallthrough) => start = prev,
                _ => break,
            }
        }
        start
    }
}
