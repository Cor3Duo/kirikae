//! Jump tables: the `br` sites whose destination is read from a table of code addresses.
//!
//! A `br` is only compiled natively if discovery knows where it can go. Everything it cannot
//! resolve falls to the interpreter and is recovered later by the profile loop, so every table
//! shape we fail to read here costs a full build-and-run cycle.
//!
//! The shape is a straight-line run of instructions that computes the destination and ends in
//! `br`. This module evaluates that run symbolically with one free register — the index — and
//! reads the entries out of constant data. Three things make it wider than a bound check plus a
//! `cmp` immediate:
//!
//! * the index register is found from the expression, not assumed to be the one the bound check
//!   compares (a tag check right before the dispatch is common and is not the table bound);
//! * the bound may be a register holding a constant (`mov w9, #5` then `cmp w0, w9`);
//! * a table without any bound check is followed while its entries stay inside constant data and
//!   are code, requiring a short run so random data cannot pass for a table.
//!
//! The result is a set of targets, never a promise that every case of the switch was found: a
//! missing target is still a hole the profile loop closes.

use std::collections::BTreeSet;

use kirikae_a64::{Control, Expr, Stmt};

use crate::discover::CodeSpace;

/// Instructions the destination expression may span between the last control instruction and the
/// `br`. Longer runs stop the walk, which makes the expression symbolic and the site unresolved.
const MAX_RUN: u64 = 32;
/// Largest entry count a bound check may imply.
const MAX_CASES: u64 = 4096;
/// A table without a bound check must have at least this many code entries. A single computed
/// address that happens to land in constant data is not a table.
const MIN_UNBOUNDED: usize = 4;
/// A table without a bound check is followed no further than this. Constant data holds many
/// arrays of code pointers back to back, so an unguarded scan that runs long is reading the next
/// array; the first [`MAX_UNBOUNDED`] entries are still real targets.
const MAX_UNBOUNDED: usize = 64;
/// Farthest a case label may be from its `br`.
const MAX_SPAN: u64 = 1 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    /// Code addresses the `br` can reach through the table, ascending and deduplicated.
    pub targets: Vec<u64>,
    /// Code addresses the destination expression names directly.
    pub anchors: Vec<u64>,
    /// Whether a bound check capped the number of entries. An unbounded table stops at the first
    /// entry that is not code.
    pub bounded: bool,
}

fn subst(e: &Expr, regs: &[Expr], tmp: &[Option<Expr>]) -> Option<Expr> {
    Some(match e {
        Expr::Reg(n) => regs[*n as usize].clone(),
        Expr::Tmp(n) => tmp.get(*n as usize)?.clone()?,
        Expr::Op(op, args) => Expr::Op(
            *op,
            args.iter()
                .map(|a| subst(a, regs, tmp))
                .collect::<Option<_>>()?,
        ),
        Expr::Load { size, addr } => Expr::Load {
            size: *size,
            addr: Box::new(subst(addr, regs, tmp)?),
        },
        other => other.clone(),
    })
}

fn eval(space: &CodeSpace, e: &Expr, index: u8, value: u64) -> Option<u64> {
    match e {
        Expr::Const(v) => Some(*v),
        Expr::Reg(n) if *n == index => Some(value),
        Expr::Op(op, args) => {
            let vals: Vec<u64> = args
                .iter()
                .map(|a| eval(space, a, index, value))
                .collect::<Option<_>>()?;
            Some(op.eval(&vals))
        }
        Expr::Load { size, addr } => space.read_const(eval(space, addr, index, value)?, *size),
        _ => None,
    }
}

/// Values a leaf can be folded to without touching a register or memory.
fn fold(e: &Expr) -> Option<u64> {
    match e {
        Expr::Const(v) => Some(*v),
        Expr::Op(op, args) => Some(op.eval(&args.iter().map(fold).collect::<Option<Vec<_>>>()?)),
        _ => None,
    }
}

fn constants(e: &Expr, out: &mut Vec<u64>) -> bool {
    match e {
        Expr::Const(v) => out.push(*v),
        Expr::Op(_, args) => return args.iter().fold(false, |load, a| constants(a, out) | load),
        Expr::Load { addr, .. } => {
            constants(addr, out);
            return true;
        }
        _ => {}
    }
    false
}

/// The straight-line run that ends at `br`, and the expression it computes for the destination.
///
/// The run starts after the previous control instruction (or [`MAX_RUN`] instructions back) so a
/// table without a bound check is still visible. Side effects that do not feed the destination
/// are ignored; anything that could produce a new destination (a call, a return, another
/// indirect jump) makes the site symbolic and rejects it.
fn destination(space: &CodeSpace, br: u64) -> Option<(u64, Expr)> {
    let mut start = br;
    while let Some(prev) = start.checked_sub(4) {
        if br - prev > 4 * MAX_RUN {
            break;
        }
        let Some(insn) = space.lift(prev) else {
            break;
        };
        if insn.control() != Control::Fallthrough {
            break;
        }
        start = prev;
    }
    let mut regs: Vec<Expr> = (0..31).map(Expr::Reg).collect();
    let mut target = None;
    for addr in (start..=br).step_by(4) {
        let insn = space.lift(addr)?;
        let mut tmp: Vec<Option<Expr>> = vec![None; 8];
        for s in &insn.stmts {
            match s {
                Stmt::Let(n, e) => *tmp.get_mut(*n as usize)? = Some(subst(e, &regs, &tmp)?),
                Stmt::SetReg(n, e) => regs[*n as usize] = subst(e, &regs, &tmp)?,
                Stmt::JumpIndirect(e) if addr == br => target = Some(subst(e, &regs, &tmp)?),
                Stmt::SetSp(_)
                | Stmt::SetNzcv(_)
                | Stmt::SetV { .. }
                | Stmt::Store { .. }
                | Stmt::Simd(_)
                | Stmt::Fence => {}
                _ => return None,
            }
        }
    }
    Some((start, target?))
}

/// Every register the expression still names.
fn registers(e: &Expr, out: &mut BTreeSet<u8>) {
    match e {
        Expr::Reg(n) => {
            out.insert(*n);
        }
        Expr::Op(_, args) => args.iter().for_each(|a| registers(a, out)),
        Expr::Load { addr, .. } => registers(addr, out),
        _ => {}
    }
}

/// The only register the expression still depends on, if it depends on exactly one.
fn index_register(space: &CodeSpace, expr: &Expr) -> Option<u8> {
    let mut named = BTreeSet::new();
    registers(expr, &mut named);
    let mut found = None;
    for r in named {
        // The expression evaluates only when every register it names is either substituted or
        // absent, so a register that evaluates is the only free one. Probing two values would
        // miss a table whose first two entries are equal (duplicate cases, a zero byte-offset
        // table), which is common.
        let Some(_) = eval(space, expr, r, 0) else {
            continue;
        };
        if found.is_some() {
            return None;
        }
        found = Some(r);
    }
    found
}

/// A constant held by `reg` in a straight line before `at`, if the assignment is visible.
fn constant_reg(space: &CodeSpace, at: u64, reg: u8) -> Option<u64> {
    let mut addr = at;
    for _ in 0..3 {
        addr = addr.checked_sub(4)?;
        let insn = space.lift(addr)?;
        if insn.control() != Control::Fallthrough {
            return None;
        }
        let mut regs: Vec<Expr> = (0..31).map(Expr::Reg).collect();
        let mut tmp: Vec<Option<Expr>> = vec![None; 8];
        for s in &insn.stmts {
            match s {
                Stmt::Let(n, e) => *tmp.get_mut(*n as usize)? = Some(subst(e, &regs, &tmp)?),
                Stmt::SetReg(n, e) => {
                    let value = subst(e, &regs, &tmp)?;
                    if *n == reg {
                        return fold(&value);
                    }
                    regs[*n as usize] = value;
                }
                _ => {}
            }
        }
    }
    None
}

/// The entry count an out-of-range branch implies, when the instruction before the run is a
/// bound check on an index register.
///
/// `cmp wN, #k` followed by `b.hi` (unsigned above) or `b.hs` (unsigned above or equal) branches
/// away the values outside the table, so the table holds `k + 1` or `k` entries. The signed
/// forms behave the same for a non-negative index. Any other condition leaves the table on the
/// branch side, a different shape that this pass does not follow.
fn bound(space: &CodeSpace, at: u64) -> Option<(u8, u64)> {
    let b = space.word(at)?;
    if b & 0xff00_0010 != 0x5400_0000 {
        return None;
    }
    let cc = b & 0xf;
    let cmp = space.word(at.checked_sub(4)?)?;
    let (index, limit) = if cmp & 0x7fc0_001f == 0x7100_001f {
        (((cmp >> 5) & 31) as u8, ((cmp >> 10) & 0xfff) as u64)
    } else if cmp & 0x7fe0_001f == 0x6b00_001f {
        let rm = ((cmp >> 16) & 31) as u8;
        (((cmp >> 5) & 31) as u8, constant_reg(space, at - 4, rm)?)
    } else {
        return None;
    };
    if index >= 31 {
        return None;
    }
    let count = match cc {
        8 | 12 => limit.checked_add(1)?,
        2 | 10 => limit,
        _ => return None,
    };
    (count > 0 && count <= MAX_CASES).then_some((index, count))
}

/// Read the table behind `br`, or `None` when the site is not a table this pass can follow.
pub fn jump_table(space: &CodeSpace, br: u64) -> Option<Table> {
    let (start, expr) = destination(space, br)?;
    let index = index_register(space, &expr);
    let Some(index) = index else {
        // No free register: either a constant destination or an expression that still depends on
        // more than one register, which evaluates to nothing.
        let target = eval(space, &expr, 0, 0)?;
        let mut anchors = Vec::new();
        constants(&expr, &mut anchors);
        anchors.retain(|a| space.is_code(*a));
        return (space.is_code(target) && target.abs_diff(br) <= MAX_SPAN).then_some(Table {
            targets: vec![target],
            anchors,
            bounded: false,
        });
    };
    let guard = start.checked_sub(4).and_then(|at| bound(space, at));
    let (mut targets, bounded) = match guard {
        Some((checked, count)) if checked == index => {
            let mut targets = Vec::with_capacity(count as usize);
            for i in 0..count {
                let target = eval(space, &expr, index, i)?;
                if !space.is_code(target) || target.abs_diff(br) > MAX_SPAN {
                    return None;
                }
                targets.push(target);
            }
            (targets, true)
        }
        // A bound on a value that is not the index does not say how long the table is. Reading
        // past its end would run into the next code-pointer array in constant data, so the site
        // stays unresolved rather than guessing.
        Some(_) => return None,
        None => {
            let mut targets = Vec::new();
            for i in 0..MAX_UNBOUNDED as u64 {
                let Some(target) = eval(space, &expr, index, i) else {
                    break;
                };
                if !space.is_code(target) || target.abs_diff(br) > MAX_SPAN {
                    break;
                }
                targets.push(target);
            }
            if targets.len() < MIN_UNBOUNDED {
                return None;
            }
            (targets, false)
        }
    };
    targets.sort_unstable();
    targets.dedup();
    let mut anchors = Vec::new();
    constants(&expr, &mut anchors);
    anchors.retain(|a| space.is_code(*a));
    Some(Table {
        targets,
        anchors,
        bounded,
    })
}
