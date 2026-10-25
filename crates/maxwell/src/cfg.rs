use std::collections::{BTreeMap, BTreeSet};

use crate::decode::{Insn, PT};

pub const MAX_SIZE: u32 = 0x10_0000;

#[derive(Clone, Debug)]
pub struct Program {
    pub insns: BTreeMap<u32, Insn>,
    pub leaders: BTreeSet<u32>,
    pub unknown: Vec<(u32, u64)>,
    pub brx: BTreeMap<u32, Vec<u32>>,
}

pub const ENTRY: u32 = 8;

pub fn next(pc: u32) -> u32 {
    if (pc + 8).is_multiple_of(32) {
        pc + 16
    } else {
        pc + 8
    }
}

/// Absolute instruction address used by `jmp`/`jcal`: a bundle address
/// denotes its first slot, matching the relative branch targets.
pub fn absolute_target(addr: u32) -> u32 {
    if addr.is_multiple_of(32) {
        addr.wrapping_add(8)
    } else {
        addr
    }
}

const CC_TRUE: u64 = 0xf;

pub fn unconditional(i: &Insn) -> bool {
    i.guard() == (PT, false) && i.bits(0, 5) == CC_TRUE
}

pub fn scan(code: &[u8]) -> Program {
    scan_with(code, &|_, _| None)
}

pub fn scan_with(code: &[u8], cbuf: &dyn Fn(u32, u32) -> Option<u32>) -> Program {
    let word = |pc: u32| {
        code.get(pc as usize..pc as usize + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    };
    let mut insns = BTreeMap::new();
    let mut leaders = BTreeSet::from([ENTRY]);
    let mut unknown = Vec::new();
    let mut brx = BTreeMap::new();
    let mut work = vec![ENTRY];
    while let Some(start) = work.pop() {
        let mut pc = start;
        loop {
            if insns.contains_key(&pc) || pc >= MAX_SIZE {
                break;
            }
            let Some(w) = word(pc) else { break };
            let Some(i) = Insn::decode(w) else {
                unknown.push((pc, w));
                break;
            };
            insns.insert(pc, i);
            let fall = next(pc);
            match i.name() {
                "bra" if i.bit(5) => {
                    break;
                }
                "bra" => {
                    let target = i.branch_target(pc);
                    leaders.insert(target);
                    work.push(target);
                    leaders.insert(fall);
                    if unconditional(&i) {
                        break;
                    }
                }
                "ssy" | "pbk" | "pcnt" if !i.bit(5) => {
                    let target = i.branch_target(pc);
                    leaders.insert(target);
                    work.push(target);
                }
                "cal" if !i.bit(5) && !i.bit(6) => {
                    let target = i.branch_target(pc);
                    leaders.insert(target);
                    work.push(target);
                    leaders.insert(fall);
                    work.push(fall);
                    break;
                }
                "jcal" if !i.bit(5) && !i.bit(6) => {
                    let target = absolute_target(i.imm32());
                    leaders.insert(target);
                    work.push(target);
                    leaders.insert(fall);
                    work.push(fall);
                    break;
                }
                "jmp" if !i.bit(5) => {
                    let target = absolute_target(i.imm32());
                    leaders.insert(target);
                    work.push(target);
                    leaders.insert(fall);
                    work.push(fall);
                    if unconditional(&i) {
                        break;
                    }
                }
                "cont" | "ret" => {
                    leaders.insert(fall);
                    work.push(fall);
                    if unconditional(&i) {
                        break;
                    }
                }
                "brx" => {
                    if let Some(targets) = jump_table(&insns, pc, &i, cbuf, code.len() as u32) {
                        for &t in &targets {
                            leaders.insert(t);
                            work.push(t);
                        }
                        brx.insert(pc, targets);
                    }
                    leaders.insert(fall);
                    if unconditional(&i) {
                        break;
                    }
                }
                "exit" | "sync" | "brk" | "kil" => {
                    leaders.insert(fall);
                    if unconditional(&i) && i.name() != "kil" {
                        break;
                    }
                }
                _ => {}
            }
            pc = fall;
        }
    }
    leaders.retain(|l| insns.contains_key(l));
    Program {
        insns,
        leaders,
        unknown,
        brx,
    }
}

fn writer(insns: &BTreeMap<u32, Insn>, pc: u32, reg: u8) -> Option<(u32, Insn)> {
    insns
        .range(..pc)
        .rev()
        .take(32)
        .find(|(_, i)| i.rd() == reg && !matches!(i.name(), "st" | "stg" | "sync" | "bra"))
        .map(|(&p, &i)| (p, i))
}

fn jump_table(
    insns: &BTreeMap<u32, Insn>,
    pc: u32,
    i: &Insn,
    cbuf: &dyn Fn(u32, u32) -> Option<u32>,
    size: u32,
) -> Option<Vec<u32>> {
    let (ldc_pc, ldc) = writer(insns, pc, i.ra())?;
    if ldc.op.value != 0xef90000000000000 {
        return None;
    }
    let slot = ldc.bits(36, 5) as u32;
    let offset = (((ldc.bits(20, 16) as u32) << 16) as i32 >> 16) as u32;
    let mut count = 64;
    if let Some((shl_pc, shl)) = writer(insns, ldc_pc, ldc.ra())
        && shl.name() == "shl"
        && let Some((_, mnmx)) = writer(insns, shl_pc, shl.ra())
        && mnmx.name() == "imnmx"
        && mnmx.form() == crate::opcodes::Form::Imm
    {
        count = (mnmx.simm20().max(0) as u32 + 1).min(256);
    }
    let base = (pc as i32 + 8 + (((i.bits(20, 24) as u32) << 8) as i32 >> 8)) as u32;
    let mut targets = Vec::new();
    for k in 0..count {
        let Some(entry) = cbuf(slot, offset.wrapping_add(4 * k)) else {
            break;
        };
        let mut t = base.wrapping_add(entry);
        if t % 32 == 0 {
            t += 8;
        }
        if t >= size || t % 8 != 0 {
            break;
        }
        if !targets.contains(&t) {
            targets.push(t);
        }
    }
    (!targets.is_empty()).then_some(targets)
}

impl Program {
    pub fn branches(&self) -> bool {
        self.leaders.len() > 1
    }

    pub fn blocks(&self) -> Vec<(u32, Vec<(u32, Insn)>)> {
        let mut out: Vec<(u32, Vec<(u32, Insn)>)> = Vec::new();
        for (&pc, &i) in &self.insns {
            if self.leaders.contains(&pc) || out.is_empty() {
                out.push((pc, Vec::new()));
            }
            out.last_mut().unwrap().1.push((pc, i));
        }
        out
    }
}
