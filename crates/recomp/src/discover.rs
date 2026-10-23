use std::collections::{BTreeMap, BTreeSet};

use kirikae_a64::{lift, Control, Insn};
use kirikae_loader::Image;

#[derive(Debug)]
pub struct Block {
    pub start: u64,
    pub insns: Vec<Insn>,
    pub next: u64,
}

impl Block {
    pub fn last(&self) -> &Insn {
        self.insns.last().expect("blocks are never empty")
    }
}

#[derive(Debug)]
pub struct Function {
    pub entry: u64,
    pub blocks: Vec<Block>,
}

impl Function {
    pub fn has_block(&self, addr: u64) -> bool {
        self.blocks.binary_search_by_key(&addr, |b| b.start).is_ok()
    }

    pub fn insn_count(&self) -> usize {
        self.blocks.iter().map(|b| b.insns.len()).sum()
    }
}

#[derive(Debug)]
pub struct Program {
    pub entries: BTreeSet<u64>,
    pub functions: Vec<Function>,
}

pub struct CodeSpace<'a> {
    images: Vec<&'a Image>,
}

impl<'a> CodeSpace<'a> {
    pub fn new(mut images: Vec<&'a Image>) -> CodeSpace<'a> {
        images.sort_by_key(|i| i.base);
        CodeSpace { images }
    }

    fn image(&self, addr: u64) -> Option<(usize, &'a Image)> {
        let i = self.images.partition_point(|m| m.base <= addr).checked_sub(1)?;
        let m = self.images[i];
        m.is_code(addr).then_some((i, m))
    }

    pub fn is_code(&self, addr: u64) -> bool {
        self.image(addr).is_some()
    }

    pub fn word(&self, addr: u64) -> Option<u32> {
        self.image(addr)?.1.word(addr)
    }

    pub fn lift(&self, addr: u64) -> Option<Insn> {
        self.word(addr).map(|w| lift(addr, w))
    }

    pub fn words(&self) -> u64 {
        self.images.iter().map(|m| (m.text.end - m.text.start) / 4).sum()
    }
}

fn adr_target(insn: &Insn) -> Option<u64> {
    if insn.word & 0x9f00_0000 != 0x1000_0000 {
        return None;
    }
    match insn.stmts.first() {
        Some(kirikae_a64::Stmt::SetReg(_, kirikae_a64::Expr::Const(v))) => Some(*v),
        _ => None,
    }
}

fn adrp_add_target(space: &CodeSpace, insn: &Insn) -> Option<u64> {
    if insn.word & 0x9f00_0000 != 0x9000_0000 {
        return None;
    }
    let Some(kirikae_a64::Stmt::SetReg(rd, kirikae_a64::Expr::Const(page))) = insn.stmts.first() else { return None };
    let next = space.word(insn.next())?;
    if next & 0xff80_0000 != 0x9100_0000 || (next >> 5) & 31 != *rd as u32 || next & 31 != *rd as u32 {
        return None;
    }
    let imm = ((next >> 10) & 0xfff) as u64;
    Some(page + if next & (1 << 22) != 0 { imm << 12 } else { imm })
}

struct Seen {
    bits: Vec<Vec<u64>>,
}

impl Seen {
    fn new(space: &CodeSpace) -> Seen {
        Seen { bits: space.images.iter().map(|m| vec![0u64; ((m.text.end - m.text.start) / 4).div_ceil(64) as usize]).collect() }
    }

    fn insert(&mut self, space: &CodeSpace, addr: u64) -> bool {
        let Some((i, m)) = space.image(addr) else { return false };
        let k = ((addr - m.text.start) / 4) as usize;
        let (word, bit) = (k / 64, 1u64 << (k % 64));
        let fresh = self.bits[i][word] & bit == 0;
        self.bits[i][word] |= bit;
        fresh
    }

    fn count(&self) -> u64 {
        self.bits.iter().flatten().map(|w| w.count_ones() as u64).sum()
    }
}

pub fn find_entries(space: &CodeSpace, seeds: &[u64]) -> (BTreeSet<u64>, u64) {
    let mut entries: BTreeSet<u64> = seeds.iter().copied().filter(|a| space.is_code(*a)).collect();
    let mut seen = Seen::new(space);
    let mut work: Vec<u64> = entries.iter().copied().collect();
    while let Some(addr) = work.pop() {
        if !seen.insert(space, addr) {
            continue;
        }
        let Some(insn) = space.lift(addr) else { continue };
        if let Some(t) = adr_target(&insn).or_else(|| adrp_add_target(space, &insn)).filter(|t| space.is_code(*t))
            && entries.insert(t)
        {
            work.push(t);
        }
        match insn.control() {
            Control::Fallthrough | Control::CallIndirect => work.push(insn.next()),
            Control::Jump(t) => work.push(t),
            Control::JumpIf(t) => work.extend([t, insn.next()]),
            Control::Call(t) => {
                if space.is_code(t) {
                    entries.insert(t);
                }
                work.extend([t, insn.next()]);
            }
            Control::JumpIndirect | Control::Return | Control::Undefined => {}
        }
    }
    (entries, seen.count())
}

pub fn discover(image: &Image) -> Program {
    let space = CodeSpace::new(vec![image]);
    let (entries, _) = find_entries(&space, &[image.entry]);
    let functions = entries.iter().map(|&e| build_function(&space, e, &entries)).collect();
    Program { entries, functions }
}

pub fn build_function(space: &CodeSpace, entry: u64, entries: &BTreeSet<u64>) -> Function {
    let internal = |t: u64| space.is_code(t) && (t == entry || !entries.contains(&t));
    let mut insns = BTreeMap::new();
    let mut leaders = BTreeSet::from([entry]);
    let mut work = vec![entry];
    while let Some(addr) = work.pop() {
        if insns.contains_key(&addr) {
            continue;
        }
        let Some(insn) = space.lift(addr) else { continue };
        match insn.control() {
            Control::Fallthrough => work.push(insn.next()),
            Control::Jump(t) => {
                if internal(t) {
                    leaders.insert(t);
                    work.push(t);
                }
            }
            Control::JumpIf(t) => {
                if internal(t) {
                    leaders.insert(t);
                    work.push(t);
                }
                leaders.insert(insn.next());
                work.push(insn.next());
            }
            Control::Call(_) | Control::CallIndirect => {
                leaders.insert(insn.next());
                work.push(insn.next());
            }
            Control::JumpIndirect | Control::Return | Control::Undefined => {}
        }
        insns.insert(addr, insn);
    }

    let mut blocks: Vec<Block> = Vec::new();
    let mut open = false;
    for (addr, insn) in insns {
        let starts_new = !open || leaders.contains(&addr) || blocks.last().is_some_and(|b| b.next != addr);
        if starts_new {
            blocks.push(Block { start: addr, insns: Vec::new(), next: addr });
        }
        let block = blocks.last_mut().unwrap();
        block.next = insn.next();
        open = insn.control() == Control::Fallthrough;
        block.insns.push(insn);
    }
    Function { entry, blocks }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kirikae_loader::LOAD_BASE;

    #[test]
    fn finds_the_test_program_functions() {
        let image = kirikae_loader::load(&kirikae_testkit::hello_nro(), LOAD_BASE).unwrap();
        let prog = discover(&image);
        assert_eq!(prog.entries.len(), 5, "{:x?}", prog.entries);
        let entry = &prog.functions[0];
        assert_eq!(entry.entry, LOAD_BASE);
        let classify = prog.functions.iter().find(|f| f.insn_count() == 5).expect("classify");
        assert!(matches!(classify.blocks.last().unwrap().last().control(), Control::JumpIndirect));
    }
}
