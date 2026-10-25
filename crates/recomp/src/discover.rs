use std::collections::{BTreeMap, BTreeSet};

use kirikae_a64::{Control, Insn, lift};
use kirikae_loader::Image;

use crate::jump::{Table, jump_table};

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

/// What discovery did, for `entry_density` and the finding notes.
///
/// The source counts overlap: an entry can be a seed and a `bl` target. They answer "how did
/// this address become an entry", not "how many entries are there of each kind".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiscoveryStats {
    /// Reachable instructions, before promotion.
    pub reachable: u64,
    /// Entries that were roots: image entry, `.eh_frame_hdr` FDE, MOD0 code pointer or profile.
    pub seeds: u64,
    /// Entries that are `bl` targets.
    pub calls: u64,
    /// Entries named by an `adr` or `adrp+add`.
    pub references: u64,
    /// Entries added so a jump across a segment boundary stays native.
    pub promoted: u64,
    /// Of [`promoted`](Self::promoted), entries reached as a jump-table case.
    pub promoted_cases: u64,
    /// Entries another entry's body would have run into by falling through.
    pub interior: u64,
    /// Reachable `blr`. Its targets only become entries through a reference or a profile.
    pub indirect_calls: u64,
    /// Reachable `br`, and how many of them had a recognised jump table.
    pub indirect_jumps: u64,
    pub tables: u64,
    /// Of [`tables`](Self::tables), how many had no bound check: the entry count came from the
    /// entries themselves.
    pub unbounded_tables: u64,
    /// Reachable `br` with no recognised table. Its targets fall to the interpreter and are
    /// recovered by the profile loop.
    pub unresolved_jumps: u64,
    /// Fixed-point rounds promotion took.
    pub rounds: u32,
}

/// The entry whose segment holds `addr`: the closest entry at or below it.
///
/// A block belongs to exactly one entry, so [`build_function`] emits it once. When it treated
/// every entry only as a stop for branches, a function walked through the following entries'
/// bodies and each shared tail target was lifted once per caller: Brilliant Diamond lifted
/// 74.6 M instructions for 13.3 M reachable (5.6x).
fn owner(entries: &BTreeSet<u64>, addr: u64) -> Option<u64> {
    entries.range(..=addr).next_back().copied()
}

/// Every explicit edge to one target: the lowest and highest source, and whether one of them
/// came from a jump table.
#[derive(Clone, Copy, Debug)]
struct Reach {
    min: u64,
    max: u64,
    table: bool,
}

impl Reach {
    fn note(&mut self, source: u64, table: bool) {
        self.min = self.min.min(source);
        self.max = self.max.max(source);
        self.table |= table;
    }
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
        let i = self
            .images
            .partition_point(|m| m.base <= addr)
            .checked_sub(1)?;
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
        self.images
            .iter()
            .map(|m| (m.text.end - m.text.start) / 4)
            .sum()
    }

    /// A little-endian word from constant data: `.rodata` or `.data`.
    ///
    /// `.data` is read as it sits in the image. Absolute pointers there are the loader's
    /// business (see `loader::dynamic`), so a relocated slot may read as zero; a relative offset
    /// table reads correctly and is what most titles use.
    pub(crate) fn read_const(&self, addr: u64, size: u8) -> Option<u64> {
        let last = addr.checked_add(size as u64 - 1)?;
        let m = self.images.iter().find(|m| {
            (m.ro.contains(&addr) && m.ro.contains(&last))
                || (m.data.contains(&addr) && m.data.contains(&last))
        })?;
        let at = (addr - m.base) as usize;
        let bytes = m.memory.get(at..at + size as usize)?;
        Some(bytes.iter().rev().fold(0, |v, &b| v << 8 | b as u64))
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
    let Some(kirikae_a64::Stmt::SetReg(rd, kirikae_a64::Expr::Const(page))) = insn.stmts.first()
    else {
        return None;
    };
    let next = space.word(insn.next())?;
    if next & 0xff80_0000 != 0x9100_0000
        || (next >> 5) & 31 != *rd as u32
        || next & 31 != *rd as u32
    {
        return None;
    }
    let imm = ((next >> 10) & 0xfff) as u64;
    Some(
        page + if next & (1 << 22) != 0 {
            imm << 12
        } else {
            imm
        },
    )
}

struct Seen {
    bits: Vec<Vec<u64>>,
}

impl Seen {
    fn new(space: &CodeSpace) -> Seen {
        Seen {
            bits: space
                .images
                .iter()
                .map(|m| vec![0u64; ((m.text.end - m.text.start) / 4).div_ceil(64) as usize])
                .collect(),
        }
    }

    fn insert(&mut self, space: &CodeSpace, addr: u64) -> bool {
        let Some((i, m)) = space.image(addr) else {
            return false;
        };
        let k = ((addr - m.text.start) / 4) as usize;
        let (word, bit) = (k / 64, 1u64 << (k % 64));
        let fresh = self.bits[i][word] & bit == 0;
        self.bits[i][word] |= bit;
        fresh
    }

    fn count(&self) -> u64 {
        self.bits
            .iter()
            .flatten()
            .map(|w| w.count_ones() as u64)
            .sum()
    }

    fn contains(&self, space: &CodeSpace, addr: u64) -> bool {
        let Some((i, m)) = space.image(addr) else {
            return false;
        };
        let k = ((addr - m.text.start) / 4) as usize;
        self.bits[i][k / 64] & (1u64 << (k % 64)) != 0
    }
}

/// A jump out of the entry's segment can only run natively if its target is an entry, so every
/// such target becomes one. This is what lets [`build_function`] stop at a segment boundary
/// without dropping reachable code into the interpreter. Conditional fall-through needs no
/// promotion: `addr + 4` can only leave its segment through an address that is already an
/// entry.
///
/// An edge `source -> target` is inside one segment exactly while no entry lies between the two
/// addresses. With several edges into the same target, the strictest is the union of their
/// ranges, which is the single interval `(min, max]` below: `min` is the lowest source (or the
/// target itself, when every source is above it) and `max` the highest. Adding an entry can only
/// make more of those intervals contain one, so the pass repeats until it adds nothing. A round
/// is two binary searches per candidate and no lifting.
fn promote(entries: &mut BTreeSet<u64>, reach: &BTreeMap<u64, Reach>) -> (u64, u64, u32) {
    // `(target, min, max, from a table)`, ascending.
    let mut candidates: Vec<(u64, u64, u64, bool)> = reach
        .iter()
        .filter(|&(&target, _)| !entries.contains(&target))
        .map(|(&target, r)| (target, target.min(r.min), target.max(r.max), r.table))
        .collect();
    candidates.sort_unstable_by_key(|c| c.0);

    let start = entries.len();
    let mut promoted_cases = 0;
    let mut rounds = 0;
    loop {
        let sorted: Vec<u64> = entries.iter().copied().collect();
        let mut added = Vec::new();
        for &(target, min, max, table) in &candidates {
            if max <= min || sorted.binary_search(&target).is_ok() {
                continue;
            }
            let at = sorted.partition_point(|&e| e <= min);
            if sorted.get(at).is_some_and(|&e| e <= max) {
                added.push((target, table));
            }
        }
        if added.is_empty() {
            return ((entries.len() - start) as u64, promoted_cases, rounds);
        }
        rounds += 1;
        for (target, table) in added {
            if entries.insert(target) {
                promoted_cases += u64::from(table);
            }
        }
    }
}

pub fn find_entries(space: &CodeSpace, seeds: &[u64]) -> (BTreeSet<u64>, u64) {
    let (entries, reachable, _) = find_entries_measured(space, seeds);
    (entries, reachable)
}

/// [`find_entries`] with the counters the discovery notes quote.
pub fn find_entries_measured(
    space: &CodeSpace,
    seeds: &[u64],
) -> (BTreeSet<u64>, u64, DiscoveryStats) {
    let mut entries: BTreeSet<u64> = seeds
        .iter()
        .copied()
        .filter(|a| space.is_code(*a))
        .collect();
    let seeded = entries.clone();
    let (mut calls, mut cases, mut refs) =
        (BTreeSet::new(), BTreeSet::<u64>::new(), BTreeSet::new());
    let mut reach: BTreeMap<u64, Reach> = BTreeMap::new();
    let mut seen = Seen::new(space);
    // Addresses another entry's body would reach by falling through, which is exactly the
    // overlap the per-segment ownership removes.
    let mut fall = Seen::new(space);
    let (
        mut indirect_calls,
        mut indirect_jumps,
        mut tables,
        mut unbounded_tables,
        mut unresolved_jumps,
    ) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut work: Vec<u64> = entries.iter().copied().collect();
    while let Some(addr) = work.pop() {
        if !seen.insert(space, addr) {
            continue;
        }
        let Some(insn) = space.lift(addr) else {
            continue;
        };
        let edge = |reach: &mut BTreeMap<u64, Reach>, target: u64, table: bool| {
            if space.is_code(target) {
                reach
                    .entry(target)
                    .or_insert(Reach {
                        min: addr,
                        max: addr,
                        table,
                    })
                    .note(addr, table);
            }
        };
        if let Some(t) = adr_target(&insn)
            .or_else(|| adrp_add_target(space, &insn))
            .filter(|t| space.is_code(*t))
        {
            refs.insert(t);
            if entries.insert(t) {
                work.push(t);
            }
        }
        match insn.control() {
            Control::Fallthrough => work.push(insn.next()),
            Control::CallIndirect => {
                indirect_calls += 1;
                work.push(insn.next());
            }
            Control::Jump(t) => {
                edge(&mut reach, t, false);
                work.push(t);
            }
            Control::JumpIf(t) => {
                edge(&mut reach, t, false);
                work.extend([t, insn.next()]);
            }
            Control::Call(t) => {
                if space.is_code(t) {
                    entries.insert(t);
                    calls.insert(t);
                }
                work.extend([t, insn.next()]);
            }
            Control::JumpIndirect => {
                indirect_jumps += 1;
                if let Some(Table {
                    targets,
                    anchors,
                    bounded,
                }) = jump_table(space, addr)
                {
                    tables += 1;
                    unbounded_tables += u64::from(!bounded);
                    cases.extend(anchors.iter().chain(&targets));
                    targets.iter().for_each(|&t| edge(&mut reach, t, true));
                    work.extend(targets);
                } else {
                    unresolved_jumps += 1;
                }
            }
            Control::Return | Control::Undefined => {}
        }
        if matches!(
            insn.control(),
            Control::Fallthrough | Control::JumpIf(_) | Control::Call(_) | Control::CallIndirect
        ) {
            fall.insert(space, insn.next());
        }
    }
    entries.retain(|e| seeded.contains(e) || calls.contains(e) || !cases.contains(e));
    let reachable = seen.count();
    let (promoted, promoted_cases, rounds) = promote(&mut entries, &reach);
    // Counted over the final entry set: a promoted entry can also be one that another body runs
    // into, and it is an entry all the same.
    let interior = entries.iter().filter(|&&e| fall.contains(space, e)).count() as u64;
    let count = |set: &BTreeSet<u64>| set.iter().filter(|a| entries.contains(a)).count() as u64;
    let stats = DiscoveryStats {
        reachable,
        seeds: count(&seeded),
        calls: count(&calls),
        references: count(&refs),
        promoted,
        promoted_cases,
        interior,
        indirect_calls,
        indirect_jumps,
        tables,
        unbounded_tables,
        unresolved_jumps,
        rounds,
    };
    (entries, reachable, stats)
}

pub fn discover(image: &Image) -> Program {
    let space = CodeSpace::new(vec![image]);
    let mut roots = vec![image.entry];
    match crate::ehframe::functions(image) {
        Ok(functions) => roots.extend(functions.iter().map(|f| f.start)),
        Err(crate::ehframe::Error::Missing) => {}
        Err(e) => eprintln!("kirikae: ignoring malformed .eh_frame_hdr: {e}"),
    }
    let (entries, _) = find_entries(&space, &roots);
    let functions = entries
        .iter()
        .map(|&e| build_function(&space, e, &entries))
        .collect();
    Program { entries, functions }
}

pub fn build_function(space: &CodeSpace, entry: u64, entries: &BTreeSet<u64>) -> Function {
    // Only this entry's own segment. Anything else is left to the runtime: a conditional
    // fall-through, a call return or a branch into another segment sets the PC and returns, and
    // the dispatcher runs that entry (the same path `blr` already takes). Promotion made every
    // such branch target an entry, so it is native code, not the interpreter.
    let internal = |t: u64| space.is_code(t) && owner(entries, t) == Some(entry);
    let mut insns = BTreeMap::new();
    let mut leaders = BTreeSet::from([entry]);
    let mut work = vec![entry];
    while let Some(addr) = work.pop() {
        if insns.contains_key(&addr) {
            continue;
        }
        let Some(insn) = space.lift(addr) else {
            continue;
        };
        match insn.control() {
            Control::Fallthrough => {
                let next = insn.next();
                if internal(next) {
                    work.push(next);
                }
            }
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
                let next = insn.next();
                if internal(next) {
                    leaders.insert(next);
                    work.push(next);
                }
            }
            Control::Call(_) | Control::CallIndirect => {
                let next = insn.next();
                if internal(next) {
                    leaders.insert(next);
                    work.push(next);
                }
            }
            Control::JumpIndirect => {
                for t in jump_table(space, addr).map_or_else(Vec::new, |table| table.targets) {
                    if internal(t) {
                        leaders.insert(t);
                        work.push(t);
                    }
                }
            }
            Control::Return | Control::Undefined => {}
        }
        insns.insert(addr, insn);
    }

    let mut blocks: Vec<Block> = Vec::new();
    let mut open = false;
    for (addr, insn) in insns {
        let starts_new =
            !open || leaders.contains(&addr) || blocks.last().is_some_and(|b| b.next != addr);
        if starts_new {
            blocks.push(Block {
                start: addr,
                insns: Vec::new(),
                next: addr,
            });
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
        let classify = prog
            .functions
            .iter()
            .find(|f| {
                f.blocks
                    .iter()
                    .any(|b| b.last().control() == Control::JumpIndirect)
            })
            .expect("classify");
        assert_eq!(
            classify.blocks.len(),
            7,
            "guard, table jump, four cases and the default: {classify:x?}"
        );
    }

    #[test]
    fn unguarded_indirect_jumps_are_not_tables() {
        let image =
            kirikae_loader::load(&kirikae_testkit::unguarded_jump_nro(), LOAD_BASE).unwrap();
        let prog = discover(&image);
        let f = &prog.functions[0];
        let jump = f.blocks.last().unwrap();
        assert_eq!(jump.last().control(), Control::JumpIndirect);
        assert!(!f.has_block(jump.next), "{f:x?}");
    }

    #[test]
    fn promotion_reaches_its_fixed_point() {
        // 0x1020 is promoted because an entry (0x1100) sits between it and its farthest source.
        // Once it is an entry it falls inside 0x1030's interval, so 0x1030 follows in a second
        // round even though nothing between 0x1015 and 0x1030 was an entry at first.
        let mut entries = BTreeSet::from([0x1000, 0x1100]);
        let mut reach = BTreeMap::new();
        reach.insert(
            0x1020,
            Reach {
                min: 0x1400,
                max: 0x1400,
                table: false,
            },
        );
        reach.insert(
            0x1030,
            Reach {
                min: 0x1015,
                max: 0x1015,
                table: false,
            },
        );
        let (promoted, cases, rounds) = promote(&mut entries, &reach);
        assert_eq!(promoted, 2);
        assert_eq!(cases, 0);
        assert_eq!(rounds, 2);
        assert!(entries.contains(&0x1020) && entries.contains(&0x1030));
    }

    #[test]
    fn promotion_keeps_edges_inside_one_segment() {
        // A forward and a backward edge that both stay between the same entries need no
        // promotion, and neither does a one-word self jump (an empty interval).
        let mut entries = BTreeSet::from([0x1000, 0x1100]);
        let mut reach = BTreeMap::new();
        reach.insert(
            0x1080,
            Reach {
                min: 0x1020,
                max: 0x1090,
                table: false,
            },
        );
        reach.insert(
            0x10c0,
            Reach {
                min: 0x10c0,
                max: 0x10c0,
                table: false,
            },
        );
        let (promoted, _, rounds) = promote(&mut entries, &reach);
        assert_eq!(promoted, 0);
        assert_eq!(rounds, 0);
        assert_eq!(entries, BTreeSet::from([0x1000, 0x1100]));
    }

    #[test]
    fn seeds_stay_entries_when_they_are_cases() {
        let image = kirikae_loader::load(&kirikae_testkit::hello_nro(), LOAD_BASE).unwrap();
        let space = CodeSpace::new(vec![&image]);
        let prog = discover(&image);
        let classify = prog
            .functions
            .iter()
            .find(|f| {
                f.blocks
                    .iter()
                    .any(|b| b.last().control() == Control::JumpIndirect)
            })
            .unwrap();
        let br = classify
            .blocks
            .iter()
            .find(|b| b.last().control() == Control::JumpIndirect)
            .unwrap()
            .last()
            .addr;
        let case = jump_table(&space, br).unwrap().targets[0];

        let (entries, _) = find_entries(&space, &[image.entry]);
        assert!(!entries.contains(&case), "{entries:x?}");
        let (entries, _) = find_entries(&space, &[image.entry, case]);
        assert!(entries.contains(&case), "{entries:x?}");
    }
}
