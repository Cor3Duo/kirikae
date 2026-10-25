//! How much code an unresolved indirect control transfer leaves to the interpreter.
//!
//! `br` sites that `jump_table` does not recognise and every `blr` are the two holes in the
//! static CFG. This measurement asks the independent stepper in `tests/oracle` what the code
//! actually reaches at those sites: it executes the dispatch with one register at a time set to
//! a case index and records the destinations. Targets that the compiler did not reach are then
//! walked as a CFG and counted, so the number is instructions, not sites.
//!
//! `#[ignore]`d because it needs a dumped title; it keeps only counters, never game bytes:
//!
//! ```text
//! KIRIKAE_DUMP=path/to/exefs cargo test --release -p kirikae-recomp --test jump_gaps -- --ignored --nocapture
//! ```
//!
//! Behind `docs/plan/findings/jump-tables.md`.

mod oracle;

use std::collections::BTreeSet;
use std::path::Path;

use kirikae_a64::Control;
use kirikae_recomp::discover::{CodeSpace, Function, build_function, find_entries_measured};
use kirikae_recomp::jump::jump_table;
use kirikae_recomp::project::seeds;
use oracle::{Cpu, Oracle};

/// Case indices the oracle tries once a register looks like an index.
const PROBES: u64 = 64;

/// The addresses the compiler reached, one bit per word per module.
struct Reachable {
    ranges: Vec<(u64, u64)>,
    bits: Vec<Vec<u64>>,
}

impl Reachable {
    fn new(ranges: Vec<(u64, u64)>) -> Reachable {
        Reachable {
            bits: ranges
                .iter()
                .map(|(start, end)| vec![0u64; ((end - start) / 4).div_ceil(64) as usize])
                .collect(),
            ranges,
        }
    }

    fn index(&self, addr: u64) -> Option<(usize, usize)> {
        let i = self
            .ranges
            .partition_point(|(start, _)| *start <= addr)
            .checked_sub(1)?;
        (addr < self.ranges[i].1).then(|| (i, ((addr - self.ranges[i].0) / 4) as usize))
    }

    fn contains(&self, addr: u64) -> bool {
        self.index(addr)
            .is_some_and(|(i, k)| self.bits[i][k / 64] & (1u64 << (k % 64)) != 0)
    }

    fn insert(&mut self, addr: u64) -> bool {
        let Some((i, k)) = self.index(addr) else {
            return false;
        };
        let fresh = self.bits[i][k / 64] & (1u64 << (k % 64)) == 0;
        self.bits[i][k / 64] |= 1u64 << (k % 64);
        fresh
    }

    /// Walk the CFG from a target the compiler did not know and count the words it did not
    /// already reach, marking them. Stops at an unresolved indirect jump: that is another hole
    /// of its own.
    fn walk_new(&mut self, space: &CodeSpace, start: u64) -> u64 {
        let mut count = 0;
        let mut work = vec![start];
        let mut local = BTreeSet::new();
        while let Some(addr) = work.pop() {
            if self.contains(addr) || !local.insert(addr) {
                continue;
            }
            let Some(insn) = space.lift(addr) else {
                continue;
            };
            self.insert(addr);
            count += 1;
            match insn.control() {
                Control::Fallthrough => work.push(insn.next()),
                Control::Jump(t) => work.push(t),
                Control::JumpIf(t) => work.extend([t, insn.next()]),
                Control::Call(t) => work.extend([t, insn.next()]),
                Control::CallIndirect => work.push(insn.next()),
                Control::JumpIndirect | Control::Return | Control::Undefined => {}
            }
        }
        count
    }
}

/// What the oracle found at one site.
#[derive(Default)]
struct Found {
    targets: BTreeSet<u64>,
    /// A register that produced a code destination when set to 0.
    index: Option<u8>,
    /// The run was not executable with any register set: a call, a syscall, an unmapped load.
    unreadable: bool,
}

/// Probe one indirect site: set each register to 0 in turn, keep the ones that produce a code
/// destination, then sweep that register over [`PROBES`] case indices.
fn probe(oracle: &Oracle, space: &CodeSpace, site: u64) -> Found {
    let start = oracle.run_start(site);
    let mut found = Found {
        unreadable: true,
        ..Found::default()
    };
    let mut indexes = Vec::new();
    for r in 0..31u8 {
        if let Some(target) = oracle.dispatch(start, &Cpu::new().with(r, 0), 64) {
            found.unreadable = false;
            if space.is_code(target) {
                indexes.push(r);
            }
        }
    }
    for r in indexes {
        found.index.get_or_insert(r);
        for i in 0..PROBES {
            if let Some(target) = oracle.dispatch(start, &Cpu::new().with(r, i), 64)
                && space.is_code(target)
            {
                found.targets.insert(target);
            }
        }
    }
    found
}

#[test]
#[ignore = "needs a dumped title in KIRIKAE_DUMP"]
fn jump_gaps() {
    let Ok(path) = std::env::var("KIRIKAE_DUMP") else {
        eprintln!("set KIRIKAE_DUMP to an extracted dump to measure it");
        return;
    };
    let exe = kirikae_loader::open(Path::new(&path)).expect("dump");
    let mut images = Vec::new();
    for m in &exe.modules {
        images.push(&m.image);
    }
    let space = CodeSpace::new(images.clone());
    let (entries, reachable, stats) = find_entries_measured(&space, &seeds(&exe));

    let ranges: Vec<(u64, u64)> = images.iter().map(|m| (m.text.start, m.text.end)).collect();
    let mut known = Reachable::new(ranges);
    let mut functions: Vec<Function> = Vec::with_capacity(entries.len());
    for &e in &entries {
        functions.push(build_function(&space, e, &entries));
    }
    for f in &functions {
        for b in &f.blocks {
            for insn in &b.insns {
                known.insert(insn.addr);
            }
        }
    }
    let mut jumps = Vec::new();
    let mut calls = Vec::new();
    for f in &functions {
        for b in &f.blocks {
            match b.last().control() {
                Control::JumpIndirect => jumps.push(b.last().addr),
                Control::CallIndirect => calls.push(b.last().addr),
                _ => {}
            }
        }
    }
    jumps.sort_unstable();
    jumps.dedup();
    calls.sort_unstable();
    calls.dedup();

    println!(
        "modules {} · entries {} · reachable {reachable} · {} `br` ({} tables: {} bounded, {} \
         without a bound check, {} unresolved) · {} `blr`",
        images.len(),
        entries.len(),
        stats.indirect_jumps,
        stats.tables,
        stats.tables - stats.unbounded_tables,
        stats.unbounded_tables,
        stats.unresolved_jumps,
        stats.indirect_calls,
    );

    let oracle = Oracle::new(&space, &images);
    let (mut unresolved, mut unreadable, mut with_targets) = (0u64, 0u64, 0u64);
    let mut targets = BTreeSet::new();
    for &site in &jumps {
        if jump_table(&space, site).is_some() {
            continue;
        }
        unresolved += 1;
        let found = probe(&oracle, &space, site);
        if found.unreadable {
            unreadable += 1;
        }
        if !found.targets.is_empty() {
            with_targets += 1;
            targets.extend(found.targets);
        }
    }
    let mut new_jump_targets = 0;
    let mut new_jump_insns = 0;
    for &t in &targets {
        if !known.contains(t) {
            new_jump_targets += 1;
            new_jump_insns += known.walk_new(&space, t);
        }
    }
    println!(
        "unresolved `br` ({unresolved} sites): the stepper cannot run {unreadable} of them \
         (calls, syscalls, unmapped loads) · {with_targets} reach code · {} distinct targets, \
         {new_jump_targets} not already reached · {new_jump_insns} instructions missed",
        targets.len()
    );

    // `blr`: the same probe. A target becomes an entry only through an `adr` reference or the
    // profile, so any code destination here is a hole even when it looks obvious.
    let (mut resolved_calls, mut call_targets) = (0u64, BTreeSet::new());
    for &site in &calls {
        let found = probe(&oracle, &space, site);
        if !found.targets.is_empty() {
            resolved_calls += 1;
            call_targets.extend(found.targets);
        }
    }
    let mut new_call_targets = 0;
    let mut new_call_insns = 0;
    let mut entry_calls = 0;
    for &t in &call_targets {
        if entries.contains(&t) {
            entry_calls += 1;
        } else if !known.contains(t) {
            new_call_targets += 1;
            new_call_insns += known.walk_new(&space, t);
        }
    }
    println!(
        "`blr` ({} sites): the stepper resolves {resolved_calls} · {} distinct targets, \
         {entry_calls} already entries, {new_call_targets} not reached · {new_call_insns} \
         instructions missed",
        calls.len(),
        call_targets.len(),
    );
}
