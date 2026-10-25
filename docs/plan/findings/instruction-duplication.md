# Per-entry instruction duplication

Session of 2026-10-25, worktree `ws2-discover` (base `8b925a3`). Question: why did Brilliant
Diamond lift 74.6 M instructions for 13.3 M reachable words (5.6x), and what is the smallest
change that makes every reachable block compile exactly once without breaking the entry table,
`Call` lowering or Super Mario Bros. Wonder?

Answer: entries now **own** an address range, so `build_function` stops where the next entry
starts, and every jump target that would be cut off is **promoted** to an entry. Duplication is
1.00x on all three dumps measured, the entry table stays a superset of the old one, and SMBW
keeps 0 interpreted instructions and ~30 fps.

## The measurement harness

`crates/recomp/tests/entry_density.rs` holds both the measurement and the synthetic regression
tests. The measurement is `#[ignore]`d, keeps no game data and prints, for any dump:

```text
KIRIKAE_DUMP=path/to/exefs cargo test --release -p kirikae-recomp --test entry_density -- --ignored --nocapture
```

It reports text size, entries (total, per 4 KiB and per module), reachable words, discovery time,
the source of each entry, how many entries another body falls into, the per-entry block count,
the multiplicity histogram of the compiled union, thunk entries, `bl`/`blr`/`br` counts, and a
sampled run of the **old** walk (`KIRIKAE_DENSITY_SAMPLE`, default 2048 entries, `0` = all) to
project what the old compiler would have lifted.

## What the dumps say

`2026-10-25`, `kirikae-recomp` built from this worktree, `CARGO_TARGET_DIR=.../kt-ws2-discover`.
Addresses are irrelevant to the finding and are not reproduced.

| | BD | SMBW | Mystery Dungeon |
| --- | ---: | ---: | ---: |
| text | 52.4 MB (13.7 M words) | 38.0 MB (9.96 M words) | 49.9 MB (13.1 M words) |
| entries | 268,038 | 260,412 | 235,699 |
| entries per 4 KiB | 20.0 | 26.8 | 18.4 |
| reachable | 13,307,769 (97.0%) | 9,748,969 (97.9%) | 12,389,733 (94.7%) |
| promoted | 6,297 (2.3%) | 13,002 (5.0%) | 1,300 (0.6%) |
| entries inside another body | 28,365 (10.6%) | 8,752 (3.4%) | 3,383 (1.4%) |
| lifted, new | 13,307,769 | 9,748,969 | 12,389,733 |
| duplication, new | **1.00x** | **1.00x** | **1.00x** |
| duplication, old (exact build log) | 5.607x | 1.085x | 1.026x |
| duplication, old (sampled harness) | 5.47x | 1.01x | 1.03x |

The sampled harness under-projects because it cannot see jump-table targets (the detector is
private to discovery): BD 5.47x vs 5.607x, SMBW 1.01x vs 1.085x, MD 1.03x vs 1.026x. It is close
enough to show the cause; use the build log when the exact figure matters.

The entry sets confirm the rule exactly: old entries plus promoted equals the new entry count on
all three dumps (261,741 + 6,297 = 268,038; 247,410 + 13,002 = 260,412; 234,399 + 1,300 =
235,699). Promotion is the only change to what becomes an entry.

`interior` is the count of entries whose address another entry's body reaches by **falling
through** (a non-branch flow edge). That is precisely the set the old walk duplicated: the old
`build_function` stopped at another entry only on a branch (`Control::Jump`/`JumpIf`), and kept
walking on `Control::Fallthrough`, conditional fall-through, call return and `blr` return. BD's
10.6% interior entries are long functions that `.eh_frame` splits into many FDEs; a caller that
entered the first one walked through all of them and lifted them again.

## Where the entries come from

BD, one entry can be in more than one row:

| source | BD | SMBW | MD |
| --- | ---: | ---: | ---: |
| seeds (image entry, `.eh_frame_hdr` FDE, MOD0 code pointer, profile) | 248,483 (92.7%) | 238,241 (91.5%) | 223,317 (94.7%) |
| `bl` targets | 73,312 (27.4%) | 74,186 (28.5%) | 65,280 (27.7%) |
| `adr` / `adrp+add` references | 3,828 (1.4%) | 9,435 (3.6%) | 2,869 (1.2%) |
| promoted (of which, jump-table cases) | 6,297 (354) | 13,002 (391) | 1,300 (0) |

Reachable indirect control flow: BD 96,648 `blr` and 30,136 `br` (2,658 with a resolved table);
SMBW 102,480 `blr` and 25,113 `br` (1,627); MD 103,226 `blr` and 30,894 `br` (2,311). `blr` never
creates an entry by itself: its targets only become entries through an `adr`/`adrp` reference or
the profile. That is why the profile loop still matters, and why the first build of a title
interprets a few hundred entry points (see *Runtime*).

PLT-style thunks are visible but not a source of density: BD has 19,190 entries whose whole
function is at most three instructions ending in a jump (SMBW 14,848). They are `bl` targets and
were already entries before this change.

## The rule: one segment per entry

`owner(entries, addr)` is the closest entry at or below `addr`. `build_function(entry)` walks only
addresses whose owner is `entry`, so every address belongs to exactly one function and is lifted
exactly once. The entry remains the function symbol in the entry table and the target of
`Stmt::Call`, so the dispatcher and `Call` lowering are unchanged.

Leaving a segment is not a hole: a block that ends in a branch to another segment does what
`blr` already did — it writes the target PC and returns, and the dispatcher runs that entry.
`Lower::goto` already handled a target without a block that way (`set_pc_const` + `leave`), so
the runtime needed no change.

### Promotion

A jump whose target is not an entry would land in the interpreter. Discovery therefore promotes
such targets. For every explicit edge `source -> target` (branches, conditional branches and
resolved jump-table cases, collected during the discovery walk itself) the target must stay in
one segment with every source. For one target that is the union of the per-edge ranges:

```text
min = min over edges of min(source, target)
max = max over edges of max(source, target)
```

and the target can stay only while `(min, max]` holds no entry. So the rule is: for each target
not already an entry, if any entry lies in `(min, max]`, make it an entry. Adding entries can only
make more ranges contain one, so `promote` repeats the scan until it adds nothing. A round is two
binary searches per candidate over an `entries` snapshot and touches no instruction; measured
depth is 2–16 rounds and discovery went from **37.6 s to 4.5 s** on SMBW compared with an earlier
prototype that re-lifted the whole program each round.

Why the interval is exact: `owner` is the largest entry `<= addr`. `owner(source) ==
owner(target)` iff no entry lies strictly between the two after the source, i.e. in
`(min(source,target), max(source,target)]`; a target that is itself an entry violates it. Taking
the union over all sources of the same target gives the single interval above, so one range query
per target decides every edge into it.

An earlier version of this change re-lifted the whole program on every round; it took 37.6 s of
discovery on SMBW. Collecting the ranges during the walk discovery already does and iterating over
the candidates instead brought that to 4.5 s.

### Why it is sound

* **Coverage.** Take a reachable address `X` and a path from a seed. Cut the path at its last
  cross-segment edge. That edge is a branch (promoted target, so an entry), a call (target is
  already an entry) or a fall-through/return into `addr + 4`. The latter can only cross a segment
  if `addr + 4` is itself an entry, because entries are addresses and the owner of `addr + 4` is
  either the owner of `addr` or `addr + 4`. So the remaining path stays inside the owner's
  segment, and the owner's walk follows the same within-segment edges. Measured: union ==
  reachable == lifted on all three dumps, with a multiplicity histogram of `{1: n}`.
* **Call targets.** `bl` targets are entries (unchanged) and their functions start at that
  address. A cross-segment call is declared as a direct call to that entry, exactly as before.
* **Termination.** The walk is confined to a segment, and the fixed point only adds entries.
* **No behaviour change in the IR.** Blocks and instructions are the same; only which function
  contains them changed. `Function::has_block` and the register planner see a subset of the old
  blocks, and every exit they no longer see sets the PC before returning.

### Why not the alternatives

* **Keep the old walk and deduplicate blocks globally.** Each block would need one owner and each
  function would have to jump into another function's body, i.e. one CFG per translation unit.
  That is a much larger change to `Lower` (register cache, block params, dispatch) for the same
  result.
* **Compile one function per 64 KiB page.** Cuts the duplication but coalesces unrelated
  functions and still needs the same promotion rule at every page boundary.
* **Leave gaps to the interpreter and let the profile fix them.** Correct but slower: BD would
  interpret every shared tail once before the profile pass, and a profile rebuild would then turn
  those into entries anyway — which is this promotion, but with a full build in between.

## Results on real titles

`kirikae build`, same `CARGO_TARGET_DIR`, 2026-10-25. "Old" is the current `main` build log for the
same dump without a profile; "new" is this worktree without a profile. Sizes are the sum of
`SizeOfRawData` of the `.text` sections in the emitted COFF objects (per-DLL entry tables
excluded).

**Pokémon Brilliant Diamond** (`myswitch/games/pokemon-brilliant-diamond`):

| | before | after |
| --- | ---: | ---: |
| functions / entries | 261,741 | 268,038 |
| lifted instructions | 74,618,780 | 13,307,769 |
| duplication | 5.607x | **1.000x** |
| object `.text` | 2,733 MB | **566 MB** (4.83x smaller) |
| object files | 2,951 MB (257) | 626 MB (263) |
| code generation | 425.6 s | **39.0 s** |
| whole build | 490 s, two DLLs (1.36 + 1.42 GB) | **66 s, one 630 MB exe** |
| output | 2 code DLLs + exe | single image, no split |

The estimate now predicts `13,307,769 × 52 = 692 MB`, below the 1.8 GiB split threshold, so BD no
longer takes the DLL path at all. The average object byte per lifted instruction went from 36.6 to
42.6, so the saving is 4.83x rather than the 5.6x of the lifted count: the duplicated instructions
were cheaper than the unique mix (the old total minus the new one puts the removed 61.3 M
instructions at ~35 bytes each), not merely fewer. The total is what matters for the PE limit.

**Super Mario Bros. Wonder** (`myswitch/extracted`):

| | before | after |
| --- | ---: | ---: |
| functions / entries | 247,410 | 260,412 |
| lifted instructions | 10,577,033 | 9,748,969 |
| duplication | 1.085x | **1.000x** |
| object `.text` | 421.4 MB | 387.2 MB |
| object files | 467.5 MB (243) | 433.4 MB (257) |
| code generation | 15.1 s | 13.9 s |
| whole build | 28 s | 30 s (first: 80 s, includes rebuilding `kirikae-rt`) |

SMBW is the no-regression check: the old duplication was only 8.5%, and the new build is 8% smaller
in `.text` with three times as many dispatcher entries (13,002 promoted) and unchanged bytes per
instruction (39.8 → 39.7).

## Runtime

Headless, 25 s, `KIRIKAE_HEADLESS=1 KIRIKAE_STATS=1`, RTX 3080, same profile loop for both builds:

| | before (profile-seeded exe) | after (first build, then rebuild with its profile) |
| --- | ---: | ---: |
| frames @10 s | 284 (28.4 fps) | 325 (32.5 fps) |
| frames @20 s | 573 (28.9 fps) | 625 (30.0 fps) |
| interpreted instructions | 0 | **0** |
| native calls @10 s | 41.5 M | 113.6 M |

Native calls grow 2.7x because a branch that leaves its segment is now a dispatcher call instead
of duplicated inline code; at ~11 M dispatches/s that is cheaper than the duplicated i-cache
footprint. A build *without* the profile interprets 17.9 M instructions (269 addresses); those
are exactly the addresses the profile adds, and rebuilding with it returns to 0. That is the
existing discovery loop, not a regression from this change: the entry set here is a strict
superset of the old one for the same roots.

BD boots to the same pre-existing fault as before, with identical address, call stack and
thread: `nn::audio::GetAudioOutBufferDataPointer` reading `0x30` (`sdk+0x3ac80`, called from
`main+0x1334b7c`). The interpreter reaches it too (see `large-executables.md`), so it is the
audio service gap, not discovery.

## Tests

`crates/recomp` suite (`--skip register_operands`), 2026-10-25:

* `entry_density`: the measurement above.
* `overlapping_entries_compile_a_block_once`: a seed inside another body's fall-through; the two
  functions lift 2 + 3 of 5 words (the old walk lifted 8).
* `jump_across_an_entry_becomes_an_entry`: the caller keeps one block, the far target becomes an
  entry and is compiled from there.
* `backward_jump_into_another_segment_becomes_an_entry`: promotion works backwards too, and five
  reachable words are lifted five times.
* `call_target_gets_its_own_function` and `call_targets_stay_entries_and_resolve`: every
  `Stmt::Call` target in the testkit programs is an entry with a non-empty function.
* `discover::tests::promotion_reaches_its_fixed_point` and
  `promotion_keeps_edges_inside_one_segment`: the interval rule and its fixed point.

## Remaining risks

* **Dispatcher cost.** Every branch that crosses an entry boundary is now a dispatcher call.
  SMBW is faster despite 2.7x more calls, but a title whose entries are dense *and* whose code is
  hot across those boundaries could lose. There is no per-function jump table yet; a per-DLL
  table (already listed in `large-executables.md`) would remove the lookup.
* **Ownership is address-based, not CFG-based.** A function whose body is split by an interior
  FDE becomes several native functions. It is correct and each block is lifted once, but locals
  live in the `Cpu` struct on every boundary, so `--fast-regs` benefits less on BD than on SMBW.
* **Promotion only knows explicit edges.** An unresolved `br` (`jump_table` returned `None`)
  still goes to the interpreter as before, but the detector has since widened (see
  [`jump-tables.md`](jump-tables.md)): on SMBW 264 more `br` sites resolve, and what remains is
  measured at 2,344 instructions that the profile loop recovers.
* **`interior` is measured for reporting only.** It costs one extra bitset of the text size per
  build (1.7 MB for BD), and it is computed for every `kirikae build`, not only for the harness.
* **The old sampled figure under-counts** jump-table-only duplicates by 2–5%; use the build log
  when the exact old number matters.
