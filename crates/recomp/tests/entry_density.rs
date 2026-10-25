//! Entry-point density, where the entries come from, and how much of the reachable code the
//! per-entry CFG walk lifts more than once.
//!
//! The three [`Image`] tests at the bottom run anywhere. The density measurement is `#[ignore]`d
//! because it needs a dumped title; run it against an extracted dump (the folder the CLI accepts,
//! or a single NSO/NRO) with:
//!
//! ```text
//! KIRIKAE_DUMP=path/to/exefs cargo test -p kirikae-recomp --test entry_density -- --ignored --nocapture
//! ```
//!
//! `KIRIKAE_DENSITY_SAMPLE` caps the entries the old (overlapping) walk visits; `0` measures every
//! entry, which is what a full `kirikae build` did and can take minutes. It keeps no game data,
//! only the counters, so it can be rerun on any dump. Behind
//! `docs/plan/findings/instruction-duplication.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use kirikae_a64::Control;
use kirikae_loader::{Image, LOAD_BASE};
use kirikae_recomp::discover::{CodeSpace, DiscoveryStats, build_function, find_entries_measured};
use kirikae_recomp::project::seeds;

/// Per-address counters over every image's `text` range.
struct Counters {
    counts: Vec<Vec<u16>>,
    ranges: Vec<(u64, u64)>,
}

impl Counters {
    fn new(ranges: &[(u64, u64)]) -> Counters {
        Counters {
            counts: ranges
                .iter()
                .map(|(start, end)| vec![0u16; ((end - start) / 4) as usize])
                .collect(),
            ranges: ranges.to_vec(),
        }
    }

    fn index(&self, addr: u64) -> Option<(usize, usize)> {
        let i = self
            .ranges
            .partition_point(|(start, _)| *start <= addr)
            .checked_sub(1)?;
        (addr < self.ranges[i].1).then(|| (i, ((addr - self.ranges[i].0) / 4) as usize))
    }

    fn bump(&mut self, addr: u64) {
        if let Some((i, k)) = self.index(addr) {
            self.counts[i][k] = self.counts[i][k].saturating_add(1);
        }
    }

    fn per_module(&self) -> Vec<(u64, u64)> {
        self.counts
            .iter()
            .map(|c| {
                (
                    c.iter().filter(|&&n| n != 0).count() as u64,
                    c.iter().filter(|&&n| n > 1).count() as u64,
                )
            })
            .collect()
    }

    fn histogram(&self) -> BTreeMap<usize, u64> {
        let mut h = BTreeMap::new();
        for &c in self.counts.iter().flatten() {
            if c != 0 {
                let bucket = match c {
                    1..=8 => c as usize,
                    9..=16 => 16,
                    17..=64 => 64,
                    65..=256 => 256,
                    _ => 1000,
                };
                *h.entry(bucket).or_default() += 1;
            }
        }
        h
    }
}

/// The original `build_function`: it stopped at another entry only on a branch, so a fall-through
/// walked through the following entries' bodies and lifted them once per caller. Kept here to
/// measure the duplication the ownership rule removes; not used by the compiler.
fn old_walk(space: &CodeSpace, entry: u64, entries: &BTreeSet<u64>) -> u64 {
    let internal = |t: u64| space.is_code(t) && (t == entry || !entries.contains(&t));
    let mut insns = BTreeSet::new();
    let mut work = vec![entry];
    while let Some(addr) = work.pop() {
        if insns.contains(&addr) {
            continue;
        }
        let Some(insn) = space.lift(addr) else {
            continue;
        };
        match insn.control() {
            Control::Fallthrough => work.push(insn.next()),
            Control::Jump(t) => {
                if internal(t) {
                    work.push(t);
                }
            }
            Control::JumpIf(t) => {
                if internal(t) {
                    work.push(t);
                }
                work.push(insn.next());
            }
            Control::Call(_) | Control::CallIndirect => work.push(insn.next()),
            // `old_walk` cannot see jump-table targets (the detector is private to discovery), so
            // it treats every indirect jump as a leaf. That makes the old figure a lower bound;
            // the exact count is in the build log quoted by the finding note.
            Control::JumpIndirect => {}
            Control::Return | Control::Undefined => {}
        }
        insns.insert(addr);
    }
    insns.len() as u64
}

/// `true` when the entry's function is only a trampoline: at most three instructions ending in a
/// jump or an indirect jump (the import/PLT shape, not a real function).
fn is_thunk(func: &kirikae_recomp::Function) -> bool {
    let insns = func.insn_count();
    insns <= 3
        && matches!(
            func.blocks.last().map(|b| b.last().control()),
            Some(Control::Jump(_) | Control::JumpIndirect)
        )
}

#[test]
#[ignore = "needs a dumped title in KIRIKAE_DUMP"]
fn entry_density() {
    let Ok(path) = std::env::var("KIRIKAE_DUMP") else {
        eprintln!("set KIRIKAE_DUMP to an extracted dump to measure it");
        return;
    };
    let exe = kirikae_loader::open(Path::new(&path)).expect("dump");
    let space = CodeSpace::new(exe.modules.iter().map(|m| &m.image).collect());
    let roots = seeds(&exe);

    let started = std::time::Instant::now();
    let (entries, reachable, stats) = find_entries_measured(&space, &roots);
    let discovery = started.elapsed().as_secs_f64();

    let words = space.words();
    let bytes = words * 4;
    let per_4k = |n: usize, bytes: u64| n as f64 / (bytes as f64 / 4096.0);
    println!(
        "modules {} · text {words} words ({:.1} MB) · {} entries ({:.1} per 4 KiB) · {reachable} reachable ({:.1}% of text) · {} seeds",
        exe.modules.len(),
        bytes as f64 / (1024.0 * 1024.0),
        entries.len(),
        per_4k(entries.len(), bytes),
        100.0 * reachable as f64 / words.max(1) as f64,
        roots.len(),
    );
    println!(
        "discovery {discovery:.1}s · promoted {} entries over {} rounds · {} interior entries",
        stats.promoted, stats.rounds, stats.interior
    );
    print_sources(&stats, entries.len());

    let ranges: Vec<(u64, u64)> = exe
        .modules
        .iter()
        .map(|m| (m.image.text.start, m.image.text.end))
        .collect();
    let mut covered = Counters::new(&ranges);
    let mut lifted = 0u64;
    let mut per_entry: Vec<(u64, u64)> = Vec::with_capacity(entries.len());
    let mut thunks = 0u64;
    let mut call_insns = 0u64;
    let mut br_insns = 0u64;
    let mut blr_insns = 0u64;
    for &e in &entries {
        let f = build_function(&space, e, &entries);
        let n = f.insn_count() as u64;
        lifted += n;
        per_entry.push((n, e));
        thunks += u64::from(is_thunk(&f));
        for b in &f.blocks {
            for insn in &b.insns {
                covered.bump(insn.addr);
                match insn.control() {
                    Control::Call(_) => call_insns += 1,
                    Control::CallIndirect => blr_insns += 1,
                    Control::JumpIndirect => br_insns += 1,
                    _ => {}
                }
            }
        }
    }
    let union: u64 = covered.counts.iter().flatten().filter(|&&c| c != 0).count() as u64;
    println!(
        "lifted {lifted} · union {union} · duplication {:.2}x · {:.1}% of reachable compiled",
        lifted as f64 / union.max(1) as f64,
        100.0 * union as f64 / reachable.max(1) as f64,
    );
    println!(
        "coverage multiplicity histogram: {:?} · {} thunk entries (≤3 insns ending in a jump) · {call_insns} `bl` · {blr_insns} `blr` · {br_insns} `br` compiled",
        covered.histogram(),
        thunks,
    );

    println!("per module:");
    let per_module = covered.per_module();
    for (i, m) in exe.modules.iter().enumerate() {
        let (start, end) = ranges[i];
        let in_module = entries.range(start..end).count();
        let (mine, shared) = per_module[i];
        println!(
            "  {:<8} {:>6.1} MB text · {in_module:>7} entries ({:.1}/4 KiB) · {mine:>8} words compiled ({:.1}% of reachable) · {shared} shared",
            m.name,
            (end - start) as f64 / (1024.0 * 1024.0),
            per_4k(in_module, end - start),
            100.0 * mine as f64 / reachable.max(1) as f64,
        );
    }

    per_entry.sort_unstable_by_key(|&(n, _)| n);
    let p = |q: f64| per_entry[(per_entry.len() as f64 * q) as usize].0;
    println!(
        "per-entry lifted instructions: p50 {} p90 {} p99 {} max {}",
        p(0.50),
        p(0.90),
        p(0.99),
        per_entry.last().copied().unwrap_or((0, 0)).0
    );
    per_entry.sort_unstable_by_key(|&(n, _)| std::cmp::Reverse(n));
    let top: u64 = per_entry.iter().take(10).map(|&(n, _)| n).sum();
    println!(
        "largest: {:x?}",
        per_entry
            .iter()
            .take(10)
            .map(|&(n, e)| (n, format!("{e:#x}")))
            .collect::<Vec<_>>()
    );
    println!(
        "top 10 hold {top} ({:.0}% of lifted)",
        100.0 * top as f64 / lifted.max(1) as f64
    );

    // The measurement that motivated ownership: the old walk duplicates, the new one does not.
    let sample: usize = std::env::var("KIRIKAE_DENSITY_SAMPLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let step = if sample == 0 {
        1
    } else {
        entries.len().div_ceil(sample).max(1)
    };
    let sampled: Vec<u64> = entries.iter().step_by(step).copied().collect();
    let started = std::time::Instant::now();
    let old_lifted: u64 = sampled.iter().map(|&e| old_walk(&space, e, &entries)).sum();
    let old_mean = old_lifted as f64 / sampled.len().max(1) as f64;
    let projected = old_mean * entries.len() as f64;
    println!(
        "old walk on {} of {} entries ({:.1}s): {old_lifted} lifted · mean {old_mean:.1} per entry · \
         projected {projected:.0} total · {:.2}x the reachable set · {:.1}% of the entries were interior",
        sampled.len(),
        entries.len(),
        started.elapsed().as_secs_f64(),
        projected / reachable.max(1) as f64,
        100.0 * stats.interior as f64 / entries.len().max(1) as f64,
    );
}

fn print_sources(stats: &DiscoveryStats, entries: usize) {
    let pct = |n: u64| 100.0 * n as f64 / entries.max(1) as f64;
    println!(
        "entry sources: {} seeds ({:.1}%) · {} `bl` targets ({:.1}%) · {} `adr`/`adrp` references ({:.1}%) · {} promoted ({:.1}%, {} from a jump table) · {} data pointers",
        stats.seeds,
        pct(stats.seeds),
        stats.calls,
        pct(stats.calls),
        stats.references,
        pct(stats.references),
        stats.promoted,
        pct(stats.promoted),
        stats.promoted_cases,
        stats.data_pointers,
    );
    println!(
        "indirect control flow reached: {} `blr` · {} `br`, {} with a resolved jump table ({} with no bound check, {} unresolved)",
        stats.indirect_calls,
        stats.indirect_jumps,
        stats.tables,
        stats.unbounded_tables,
        stats.unresolved_jumps,
    );
}

// ---------------------------------------------------------------------------------------------
// Synthetic programs: the properties the real dumps only suggest.

const NOP: u32 = 0xd503_201f;
const RET: u32 = 0xd65f_03c0;

fn b(from: u64, to: u64) -> u32 {
    0x1400_0000 | b_imm(from, to)
}

fn bl(from: u64, to: u64) -> u32 {
    0x9400_0000 | b_imm(from, to)
}

fn b_imm(from: u64, to: u64) -> u32 {
    (((to as i64 - from as i64) / 4) as u32) & 0x03ff_ffff
}

/// One 64-bit image whose text starts at `LOAD_BASE` and holds `words`, plus `entry`.
fn image(entry: u64, words: &[(u64, u32)]) -> Image {
    let end = words.iter().map(|(a, _)| *a + 4).max().unwrap_or(LOAD_BASE);
    let mut memory = vec![0u8; (end - LOAD_BASE) as usize];
    for &(addr, word) in words {
        let at = (addr - LOAD_BASE) as usize;
        memory[at..at + 4].copy_from_slice(&word.to_le_bytes());
    }
    Image {
        base: LOAD_BASE,
        memory,
        text: LOAD_BASE..end,
        ro: 0..0,
        data: 0..0,
        bss: 0..0,
        entry,
    }
}

fn discover_words(
    entry: u64,
    words: &[(u64, u32)],
    seeds: &[u64],
) -> (Image, BTreeSet<u64>, DiscoveryStats) {
    let image = image(entry, words);
    let space = CodeSpace::new(vec![&image]);
    let (entries, _, stats) = find_entries_measured(&space, seeds);
    (image, entries, stats)
}

#[test]
fn overlapping_entries_compile_a_block_once() {
    // A is the entry at 0x1000; B is a seed at 0x1008 that sits inside A's fall-through body.
    // The old walk gave A all five instructions and B the last three (8 lifted for 5 words).
    let words = [
        (LOAD_BASE, NOP),
        (LOAD_BASE + 4, NOP),
        (LOAD_BASE + 8, NOP),
        (LOAD_BASE + 12, NOP),
        (LOAD_BASE + 16, RET),
    ];
    let seeds = [LOAD_BASE, LOAD_BASE + 8];
    let (image, entries, stats) = discover_words(LOAD_BASE, &words, &seeds);
    assert_eq!(
        entries,
        BTreeSet::from([LOAD_BASE, LOAD_BASE + 8]),
        "both seeds stay entries"
    );
    assert_eq!(stats.interior, 1, "0x1008 is inside 0x1000's body");

    let space = CodeSpace::new(vec![&image]);
    let a = build_function(&space, LOAD_BASE, &entries);
    let b = build_function(&space, LOAD_BASE + 8, &entries);
    assert_eq!(a.blocks.len(), 1);
    assert_eq!(a.insn_count(), 2, "A stops where B starts");
    assert_eq!(b.insn_count(), 3);
    assert_eq!(
        a.insn_count() + b.insn_count(),
        5,
        "each of the five reachable words is lifted exactly once"
    );
}

#[test]
fn jump_across_an_entry_becomes_an_entry() {
    // 0x1000 jumps over the entry at 0x1008 to 0x1030. Without promotion 0x1030 would belong to
    // 0x1008's segment and be dropped from native code; with it, it is an entry of its own.
    let words = [
        (LOAD_BASE, b(LOAD_BASE, LOAD_BASE + 0x30)),
        (LOAD_BASE + 8, RET),
        (LOAD_BASE + 0x30, RET),
    ];
    let seeds = [LOAD_BASE, LOAD_BASE + 8];
    let (image, entries, stats) = discover_words(LOAD_BASE, &words, &seeds);
    assert_eq!(stats.promoted, 1);
    assert!(entries.contains(&(LOAD_BASE + 0x30)), "{entries:x?}");

    let space = CodeSpace::new(vec![&image]);
    let caller = build_function(&space, LOAD_BASE, &entries);
    assert_eq!(caller.insn_count(), 1, "the jump leaves the segment");
    let target = build_function(&space, LOAD_BASE + 0x30, &entries);
    assert_eq!(target.entry, LOAD_BASE + 0x30);
    assert_eq!(target.insn_count(), 1);
}

#[test]
fn backward_jump_into_another_segment_becomes_an_entry() {
    // The last entry jumps back to 0x1004, which is inside the segment of the first entry (0x1000)
    // and past the second (0x1008). Promotion cuts 0x1004 out of 0x1000's segment.
    let words = [
        (LOAD_BASE, RET),
        (LOAD_BASE + 4, NOP),
        (LOAD_BASE + 8, RET),
        (LOAD_BASE + 12, RET),
        (LOAD_BASE + 16, b(LOAD_BASE + 16, LOAD_BASE + 4)),
    ];
    let seeds = [LOAD_BASE, LOAD_BASE + 8, LOAD_BASE + 12, LOAD_BASE + 16];
    let (image, entries, stats) = discover_words(LOAD_BASE, &words, &seeds);
    assert_eq!(stats.promoted, 1);
    assert!(entries.contains(&(LOAD_BASE + 4)), "{entries:x?}");

    let space = CodeSpace::new(vec![&image]);
    let total: usize = entries
        .iter()
        .map(|&e| build_function(&space, e, &entries).insn_count())
        .sum();
    assert_eq!(total, 5, "five reachable words, five lifted");
}

#[test]
fn call_target_gets_its_own_function() {
    // `bl` to 0x1020, which no seed names. Discovery must make it an entry so the callee is a
    // function of its own and the caller's body stops at the return address.
    let words = [
        (LOAD_BASE, bl(LOAD_BASE, LOAD_BASE + 0x20)),
        (LOAD_BASE + 4, RET),
        (LOAD_BASE + 0x20, RET),
    ];
    let (image, entries, stats) = discover_words(LOAD_BASE, &words, &[LOAD_BASE]);
    assert_eq!(stats.calls, 1);
    assert_eq!(entries, BTreeSet::from([LOAD_BASE, LOAD_BASE + 0x20]));

    let space = CodeSpace::new(vec![&image]);
    let caller = build_function(&space, LOAD_BASE, &entries);
    assert_eq!(
        caller.insn_count(),
        2,
        "the return address stays in the caller"
    );
    assert!(
        !caller.has_block(LOAD_BASE + 0x20),
        "the callee is not inlined"
    );
    let callee = build_function(&space, LOAD_BASE + 0x20, &entries);
    assert_eq!(callee.insn_count(), 1);
    assert_eq!(caller.insn_count() + callee.insn_count(), 3);
}

#[test]
fn call_targets_stay_entries_and_resolve() {
    let image = kirikae_loader::load(&kirikae_testkit::hello_nro(), LOAD_BASE).unwrap();
    let prog = kirikae_recomp::discover(&image);
    for f in &prog.functions {
        for block in &f.blocks {
            for insn in &block.insns {
                for stmt in &insn.stmts {
                    let kirikae_a64::Stmt::Call(target) = stmt else {
                        continue;
                    };
                    assert!(
                        prog.entries.contains(target),
                        "{target:#x} is called but not an entry"
                    );
                    let callee = prog
                        .functions
                        .iter()
                        .find(|c| c.entry == *target)
                        .unwrap_or_else(|| panic!("no function for call target {target:#x}"));
                    assert_eq!(callee.entry, *target);
                    assert!(!callee.blocks.is_empty());
                }
            }
        }
    }
}
