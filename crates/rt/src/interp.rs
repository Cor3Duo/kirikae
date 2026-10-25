use std::sync::Arc;

use kirikae_a64::{Expr, Insn, Stmt, lift};

use crate::cpu::Cpu;
use crate::horizon;

pub fn eval(c: &Cpu, e: &Expr, tmp: &[u64; 8]) -> u64 {
    match e {
        Expr::Const(v) => *v,
        Expr::Reg(n) => c.x[*n as usize],
        Expr::Sp => c.sp,
        Expr::Nzcv => c.nzcv,
        Expr::Tmp(n) => tmp[*n as usize],
        Expr::Op(op, args) => {
            let mut vals = [0u64; 5];
            for (slot, a) in vals.iter_mut().zip(args) {
                *slot = eval(c, a, tmp);
            }
            op.eval(&vals[..args.len()])
        }
        Expr::Load { size, addr } => c.load(*size, eval(c, addr, tmp)),
        Expr::SysReg(r) => c.sysreg(*r),
        Expr::V { reg, hi } => c.v[*reg as usize][*hi as usize],
    }
}

pub fn exec(c: &mut Cpu, insn: &Insn) {
    let mut tmp = [0u64; 8];
    for s in &insn.stmts {
        match s {
            Stmt::Let(n, e) => tmp[*n as usize] = eval(c, e, &tmp),
            Stmt::SetReg(n, e) => c.x[*n as usize] = eval(c, e, &tmp),
            Stmt::SetSp(e) => c.sp = eval(c, e, &tmp),
            Stmt::SetNzcv(e) => c.nzcv = eval(c, e, &tmp),
            Stmt::SetSysReg(r, e) => {
                let v = eval(c, e, &tmp);
                c.set_sysreg(*r, v);
            }
            Stmt::SetV { reg, hi, value } => {
                c.v[*reg as usize][*hi as usize] = eval(c, value, &tmp)
            }
            Stmt::Store { size, addr, value } => {
                let (a, v) = (eval(c, addr, &tmp), eval(c, value, &tmp));
                c.store(*size, a, v);
            }
            Stmt::Monitor { addr, lo, hi } => {
                let (a, l, h) = (eval(c, addr, &tmp), eval(c, lo, &tmp), eval(c, hi, &tmp));
                c.monitor(a, l, h);
            }
            Stmt::StoreExclusive {
                size,
                addr,
                lo,
                hi,
                status,
            } => {
                let (a, l, h) = (eval(c, addr, &tmp), eval(c, lo, &tmp), eval(c, hi, &tmp));
                let r = c.store_exclusive(*size, a, l, h);
                if *status != 31 {
                    c.x[*status as usize] = r;
                }
            }
            Stmt::Fence => std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst),
            Stmt::Simd(word) => crate::simd::exec(c, *word),
            Stmt::Svc(imm) => {
                c.pc = insn.next();
                horizon::svc(c, *imm);
                return;
            }
            Stmt::Jump(t) | Stmt::Call(t) => {
                c.pc = *t;
                return;
            }
            Stmt::JumpIf(cond, t) => {
                c.pc = if eval(c, cond, &tmp) != 0 {
                    *t
                } else {
                    insn.next()
                };
                return;
            }
            Stmt::JumpIndirect(e) | Stmt::CallIndirect(e) => {
                c.pc = eval(c, e, &tmp);
                if crate::coverage::on() && c.process().lookup(c.pc).is_none() {
                    crate::coverage::record(c.pc);
                }
                return;
            }
            Stmt::Return(e) => {
                c.pc = eval(c, e, &tmp);
                return;
            }
            Stmt::Undefined(word) => {
                horizon::undefined(c, insn.addr, *word);
                return;
            }
        }
    }
    c.pc = insn.next();
}

pub fn step(c: &mut Cpu) {
    if c.process().arch == kirikae_loader::Arch::A32 {
        crate::interp_a32::step(c);
        return;
    }
    step_a64(c);
}

fn step_a64(c: &mut Cpu) {
    let p = c.process();
    p.stats.count_interpreted();
    let pc = c.pc;
    let cached = p.interp_cache.read().unwrap().get(&pc).cloned();
    let insn = match cached {
        Some(i) => i,
        None => {
            if !pc.is_multiple_of(4) || !p.memory.is_mapped(pc, 4) {
                horizon::fault(
                    c,
                    format!("jumped to unmapped or misaligned address {pc:#x}"),
                );
                return;
            }
            let insn = Arc::new(lift(pc, c.ld32(pc) as u32));
            p.interp_cache.write().unwrap().insert(pc, insn.clone());
            insn
        }
    };
    exec(c, &insn);
}
