# Jump tables: what an unresolved `br` costs, and a wider reader

Session of 2026-10-25, worktree `ws2-unicorn` (base `a8ea531`, on top of the per-entry ownership
change `be34a9c` in `instruction-duplication.md`). Question: **how many reachable instructions
does discovery miss because `JumpIndirect` and `CallIndirect` are unresolved, and can a wider
jump-table reader close part of that hole without a game-specific rule or an unbounded entry
explosion?**

Answer: on the two titles measured, `blr` misses **0** instructions: of 102,584 (SMBW) and
96,881 (BD) indirect calls, the code only ever sets up 5 and 2 statically visible targets, and
all of them were already entries or the target is runtime state (an import slot, a struct
field). `br` misses a small tail: **2,344** instructions on SMBW and **5,812** on BD after this
change; before it the same measurement plus what this change recovered gives **22,412** and
**46,201**. The wider reader resolves 264 (SMBW) and 341 (BD) more `br` sites, which lifts
20,068 and 40,389 more instructions. There is no production dependency: Unicorn is GPL-2.0 and
is not linked at all, not even as a dev-dependency.

## Method

Two independent measurements:

1. **Missed instructions.** `crates/recomp/tests/jump_gaps.rs` walks the compiled program and
   takes every `br` that `jump::jump_table` does not resolve and every `blr`. For each site the
   oracle (`crates/recomp/tests/oracle/mod.rs`) executes the straight-line run that computes the
   destination with one register at a time set to 0. A register that produces a code address is
   the index; it is then swept over 64 case indices and every code destination is recorded. Each
   destination the compiler did not already reach is walked as a CFG and its new words are
   counted. The oracle never resolves a call/syscall/vector/unmapped-load path, so a site it
   cannot run is reported separately instead of being counted as a table.
2. **What the reader changed.** `crates/recomp/tests/entry_density.rs` on the same dump, with
   the reader before (`a8ea531`) and after this change.

Both counts are lower bounds: the walk from a newly found target stops at its own unresolved
`br`, and the probe needs the index register to produce a code address at value 0 before it
sweeps the rest. Both effects are small on these dumps (the nested-table case is what the
profile loop keeps finding), but they are why the figures are "at least", not "exactly".

```text
KIRIKAE_DUMP=path/to/exefs cargo test --release -p kirikae-recomp --test jump_gaps -- --ignored --nocapture
KIRIKAE_DUMP=path/to/exefs cargo test --release -p kirikae-recomp --test entry_density -- --ignored --nocapture
```

Both print counters only and keep no game bytes.

## Why Unicorn is not used

`unicorn` is **GPL-2.0**. This workspace is MIT OR Apache-2.0 and `AGENTS.md` rule 3 accepts no
GPL code; a dev-dependency is still GPL code in the repository's build graph and its test
binaries would be GPL, which is a licence decision this change is not allowed to make on its own.
The workspace did not have it, and it stays out of `crates/recomp/Cargo.toml` — production *and*
dev.

Instead, `tests/oracle/mod.rs` is a ~250-line stepper over the IR that the runtime already uses:
a controlled register file, the image as memory, one instruction at a time, `Op::eval` for the
same arithmetic the interpreter runs. It executes only the shapes a dispatch is made of and
stops on anything else, so it cannot silently invent a target. It is a test module, not a crate.
If a future change wants Unicorn anyway, it needs an explicit licence review first.

## What the unresolved sites actually are

SMBW, reachable, after this change (BD is the same shape):

| | SMBW | BD |
| --- | ---: | ---: |
| `br` | 25,131 | 30,164 |
| resolved tables | 1,891 | 2,999 |
| unresolved `br` | 23,240 | 27,165 |
| the stepper cannot run the site (call, syscall, unmapped load) | 12,603 | 13,563 |
| runnable and reaches code through one register | 29 | 46 |
| distinct targets there | 815 | 1,455 |
| targets the compiler had not reached | 100 | 290 |
| **instructions missed** | **2,344** | **5,812** |
| `blr` | 102,584 | 96,881 |
| `blr` resolved by the stepper | 5 | 2 |
| **instructions missed through `blr`** | **0** | **0** |

The unresolved `br` population is not mostly switches:

* ~22.5 k (SMBW) are an `adrp`/`ldr` from a slot followed by `br xN`: an import/PLT thunk. The
  slot is filled by rtld at load time — every single one read from the image is zero or not a
  code address (measured: 11,798 unreadable, 835 zero, 0 code among 22,504 such sites before the
  change). Nothing static can be followed, and the target is outside the module.
* The remaining ~10.6 k (SMBW) / 13.6 k (BD) are runnable but no single register produces a code
  destination: a `br` through a register that was never assigned in the block (a tail call
  through an argument or a struct field), or an import stub whose slot holds a non-code value.
  The first is a whole-function dataflow problem, not a table one.
* The 29/46 sites above are real tables the reader still declines (a bound check on a different
  value, or a table longer than the unguarded cap).

`blr` is the same story with no table: rtld-resolved imports and `blr` through a field. The code
that does have a visible target (`adr xN, f; blr xN`) already becomes an entry through the `adr`
reference, which is why the stepper finds 5/2 and every one is already an entry. That is what the
`instruction-duplication.md` note calls the profile loop's job; this measurement puts the hole at
zero *instructions* on both dumps.

## The reader

`crates/recomp/src/jump.rs` replaces the private detector in `discover.rs`. The shape is still a
straight-line run that computes a destination and ends in `br`, evaluated symbolically with one
free register, in `.rodata` or `.data` (absolute pointers in `.data` read as they sit in the
image; the loader's RELA pointers are seeded separately by `loader::dynamic`). Four changes:

1. **The index register comes from the expression, not from the bound check.** The old reader
   assumed the register compared by `cmp` was the index. It usually is, but a tag check right
   before the dispatch is not: of 1,593 indexed loads, 197 had a guard on another value (SMBW).
   The register is now found by evaluation — the expression evaluates exactly when every register
   it names is substituted, so a register that evaluates is the only free one. Probing two
   values would miss tables whose first two entries are equal, which is common (duplicate cases,
   zero byte offsets); this was a real bug in the first version of the change.
2. **The bound may be in a register.** `mov w9, #5; cmp w0, w9; b.hi` is the same bound; the
   constant is followed for up to three instructions.
3. **A guard on a value that is not the index rejects the site.** That bound says nothing about
   the table's length, and reading past the end runs into the next array of code pointers in
   constant data. Without that refusal the unbounded scan merged neighbouring tables and
   promoted 145,609 entries on SMBW (+50% over the old reader) for 8,343 more instructions than
   the capped version below.
4. **A table with no bound check at all is followed for up to 64 entries**, and only while every
   entry is code. Below four entries it is not a table. 64 is a judgement call from the same
   measurement: without the cap, 37 sites with 1,024–4,096 "entries" added 124 k entries for
   8 k instructions; the cap keeps them out.

Bounded tables still require every entry to be code, so a misidentified count is rejected whole
rather than partly followed.

## Results on the real titles

`kirikae build`, same `CARGO_TARGET_DIR`, 2026-10-25. "Before" is `a8ea531` (per-entry ownership,
old detector), "after" is this worktree. Function count is the entry table; lifted instructions
are the union of the emitted blocks (`entry_density`, each reachable word exactly once).

**Super Mario Bros. Wonder** (`myswitch/extracted`):

| | before | after |
| --- | ---: | ---: |
| functions / entries | 260,412 | 268,443 |
| lifted instructions | 9,749,760 | **9,769,828** (+20,068, +0.21%) |
| reachable | 9,749,760 | 9,769,828 |
| `br` | 25,113 | 25,131 |
| resolved tables | 1,627 | **1,891** |
| unresolved `br` | 23,486 | 23,240 |
| promoted entries | 13,002 (391 from tables) | 21,258 (2,735 from tables) |
| discovery | 2.6 s | 2.5 s |
| duplication | 1.00x | 1.00x |

**Pokémon Brilliant Diamond** (`myswitch/games/pokemon-brilliant-diamond`):

| | before | after |
| --- | ---: | ---: |
| functions / entries | 268,038 | 272,020 |
| lifted instructions | 13,308,590 | **13,348,979** (+40,389, +0.30%) |
| reachable | 13,308,590 | 13,348,979 |
| `br` | 30,136 | 30,164 |
| resolved tables | 2,658 | **2,999** |
| unresolved `br` | 27,478 | 27,165 |
| promoted entries | 6,297 (354 from tables) | 10,556 (2,387 from tables) |
| discovery | 4.5 s | 4.0 s |
| duplication | 1.00x | 1.00x |

Entry growth is 3.1% (SMBW) and 1.5% (BD) for 0.2–0.3% more instructions: a switch case that is
inside another entry's body has to become an entry to be compiled natively (`build_function`
only emits its own segment, and the dispatcher needs an entry to land on), so targets that were
already reachable through another path still cost an entry. The alternative — following the
table without promoting — leaves exactly those targets to the interpreter, which is the hole
this change is measuring.

The "interpreter addresses recovered" are the 264/341 newly resolved sites and their targets.
Before the change, the same oracle-plus-walk measurement, reading the new reachable set as the
baseline, gives 2,344 + 20,068 = **22,412** (SMBW) and 5,812 + 40,389 = **46,201** (BD)
instructions that the first build would have interpreted until a profile rebuild.

The builds themselves (no profile, `kirikae build <dump>`): SMBW 268,443 functions and
9,769,828 of 9,956,704 instructions reachable, 94 s including the generated project's runtime;
BD 272,020 functions and 13,348,979 of 13,725,765 reachable, 99 s.

A headless SMBW run of the SMBW build (`KIRIKAE_HEADLESS=1 KIRIKAE_STATS=1`, 25 s) presents a
frame and holds the frame rate: 323 frames at 10 s (32.3 fps, 80.7 unlimited), matching the
32.5 fps of the profiled build in `instruction-duplication.md`. It reports 2,690,440
**executions** through the interpreter, not 2,344 distinct words: the remaining holes are import
thunks and callback dispatch in hot code, so a handful of addresses run many times. That is the
dynamic side of the measurement above and what the profile loop exists for; the static hole this
change closes is 22,412 words.

Discovery is unchanged within noise (SMBW 2.6 → 2.5 s, BD 4.5 → 4.0 s). The first version of
the reader probed all 31 registers for every site and took 4.3 / 7.7 s; the expression already
names its registers, so sweeping only those removed the cost.

## Tests

`crates/recomp`, `--skip register_operands`, 2026-10-25:

* `jump_tables`: synthetic programs assembled in the test, table in `.rodata` after the code.
  * `absolute_eight_byte_table`, `relative_four_byte_table`, `byte_offset_table` — the three
    layouts, each checked against the stepper: for case `i` the stepper sees exactly that case,
    and out of range exactly the default.
  * `unbounded_absolute_table`, `unbounded_relative_table` — no bound check, followed until the
    first non-code entry.
  * `the_bound_may_be_a_register_constant` — `mov w9, #3; cmp w0, w9; b.hi`.
  * `a_table_whose_entries_are_not_code_is_rejected`, `a_table_of_zeros_is_rejected`,
    `a_short_unguarded_run_is_not_a_table` — the bogus shapes, none may produce targets.
  * `a_guard_on_another_value_does_not_guess_the_length` — the table is real, the length is not
    knowable, and the reader declines rather than reading into the next array.
  * `a_constant_pointer_jump_is_a_single_target` — `br` through a fixed `.rodata` pointer is one
    edge, not a table.
  * `case_targets_are_lifted_exactly_once` — entry ownership still holds with table targets: the
    sum of lifted blocks equals the reachable set.
  * `arbitrary_bytes_never_panic` — fixed-seed noise in text and constant data: every address is
    asked, discovery runs on the same bytes, and any reported target must be code.
* `jump_gaps`: the measurement above, `#[ignore]`d.
* `entry_density`: unchanged measurement, now also prints bounded/unbounded/unresolved tables.

## Remaining gaps

* **Import thunks are still holes by construction.** 22.5 k `br` sites read a slot rtld fills at
  load time. Their target is resolved at runtime, so the dispatcher already handles them; the
  interpreter is not involved. A util that maps an import slot to its module export would turn
  those into direct edges, and is a separate change.
* **`blr` through a field is not followed.** The target is `[x0+0x18]` or a callback argument;
  only whole-function dataflow or the profile finds it. Measured at 0 missed *instructions* on
  both dumps, so it is not urgent.
* **Unbounded tables over 64 entries.** The cap trades coverage for entry count; the 37 big
  arrays on SMBW are mostly function-pointer arrays whose targets are already entries, so the
  trade is good there, but a title whose switches are large and unguarded would want the profile
  loop.
* **Absolute tables in `.data`.** The reader reads the file image; a RELA-relocated slot may be
  zero. Those targets are already seeds through `loader::dynamic::code_pointers`, so they are
  compiled, but the `br` itself is not linked to them and promotion does not see the edge.
* **Discovery asks twice.** The reader runs for a site in discovery and again in
  `build_function`; the measured cost is in the noise today, but a cache keyed by `br` would
  remove the duplicate work if the reader grows.
