# WS12 slice: an A32 decoder and IR for the frontend

Date: 2026-10-25
Branch: `ws12-a32` (not merged)
Status: **proof of concept complete**; full frontend not implemented

## Question

[`aarch32.md`](aarch32.md) and [`aarch32-confirmation.md`](aarch32-confirmation.md) show
that Mario Kart 8 Deluxe 1.0 and New Super Mario Bros. U Deluxe ship an AArch32 `main`
module. WS12 is estimated at 6–8 weeks; before committing to it, does a useful A32
decoder fit the shape of `crates/a64`, and does a small dedicated IR give the later
interpreter/backend what it needs without touching the A64 pipeline?

The answer from this slice is yes, with the caveats in "What remains".

## What the slice covers

`crates/a64/src/a32/` is self-contained: it does not import the A64 lifter, the
recompiler or the runtime, and nothing calls it yet. The crate only gained
`pub mod a32;` in `crates/a64/src/lib.rs`.

| Class | Forms decoded |
|---|---|
| Condition | All 15 condition codes, including evaluation against packed NZCV |
| Data processing (immediate, register, register-shifted register) | `and`, `eor`, `sub`, `rsb`, `add`, `adc`, `sbc`, `rsc`, `tst`, `teq`, `cmp`, `cmn`, `orr`, `mov`, `bic`, `mvn`; `S` bit; rotated immediates |
| Wide moves | `movw`, `movt` |
| Branches | `b`, `bl`, `bx`, `blx` (register and immediate), with interworking state |
| Single transfers | `ldr`/`str` word and byte, immediate or register offset, pre/post-indexed, writeback, PC-relative and PC-destination forms |
| Block transfers | `ldm`/`stm` in `ia`/`ib`/`da`/`db`, writeback, register lists |
| Multiply | `mul`, `mla`, `S` bit |
| Stop points | `svc`, `bkpt` |

Everything else is rejected with a typed error; the decoder never panics, never guesses
and never reads outside its arguments:

- `Unsupported` — allocated, but outside the slice: coprocessor 0–15 (VFP, NEON, MCR/MRC),
  media/parallel and halfword forms, multiply-long and swap, extra load/store (`ldrh`,
  `ldrd`, ...), exclusives, `mrs`/`msr`, hints, `cps`/`setend`, `pld`, unprivileged
  `ldrt`/`strt`, and `ldm`/`stm` with the `S` bit.
- `Unallocated` — no instruction in that encoding (for example the `msr` immediate space
  with neither a hint nor a valid mask field).
- `Unpredictable` — allocated but architecturally UNPREDICTABLE, rejected instead of
  given invented semantics: empty register list, base in its own writeback list, `pc` as
  base of a block transfer, store of `pc` in a block transfer, `pc` as index register,
  byte transfer to `pc`, writeback over the base or destination, SBZ fields that are not
  zero (`mov`/`mvn` Rn, `mul` accumulator, test `Rd`), `pc` operands of `mul`/`mla`/
  `movw`/`movt`, `bkpt` with a condition other than `al`.

The decoder follows the ARM ARM ARMv7-A/R encoding tables: class from bits 27:25, then
the opcode, then the selector bits (bit 4 for the data-processing/miscellaneous and
media spaces, bit 4 and bit 7 for register shifts). Branch targets use the architectural
`pc + 8` and wrap modulo 2^32. `blx` immediate adds `H` as bit 1 of the target, as the
pseudocode does.

Two deliberate syntax choices: `push`/`pop` are decoded as `stmdb sp!`/`ldmia sp!`
instead of being aliased, and `rrx` is kept as `mov ..., rrx`. Both are unambiguous and
reversible; aliasing belongs to a printer, not to the IR.

## IR shape

`crates/a64/src/a32/ir.rs` has one `Insn` per 32-bit word:

```rust
pub struct Insn {
    pub pc: u32,            // address of the word
    pub word: u32,          // original encoding
    pub cond: Cond,         // 15 condition codes, no 0b1111
    pub opcode: Opcode,     // mnemonic, 30 variants
    pub sets_flags: bool,   // S bit, or implicit for tst/teq/cmp/cmn
    pub operands: Vec<Operand>,
}
```

`Operand` is `Reg`, `Imm`, `Shifted { reg, shift }`, `Mem`, `Block` or `Target`.
`Mem` keeps base, offset, `add`, `pre_indexed`, `writeback` and `Byte`/`Word`; `Offset` is
an immediate or a scaled index register. `Block` keeps base, `RegList`, addressing mode
and writeback. `Target` is an absolute address plus `Arm`/`Thumb` state.
`Shift` normalises the architectural edge cases (`Lsr(32)`, `Asr(32)`, `Rrx`) so no
consumer has to remember that `LSR #0` means 32.

Operand order is fixed per opcode and documented on `Opcode`; for example
`and`…`bic` are `[Rd, Rn, op2]`, `mov`/`mvn` are `[Rd, op2]`, `tst`…`cmn` are `[Rn, op2]`,
`ldr`/`str` are `[Rt, Mem]`, `ldm`/`stm` are `[Block]`, and `mla` is `[Rd, Rm, Rs, Ra]`
with `Rd = Rm * Rs + Ra`, matching gas/LLVM assembly order.

`Insn::mnemonic()` rebuilds the full assembly mnemonic (`addsne`, `ldrbne`, `ldmiane`,
`stmdb`) and `Display` renders the canonical text; that is what the tests pin. The IR
itself is structured, so a lowering pass never re-reads the encoding.

## Intended integration (not wired)

1. **Architecture tag.** `crates/loader/src/package.rs` already classifies A32 (META
   `Flags0` bit 0, with a code-sample fallback) and rejects the title. WS12 turns the
   rejection into a tag on `Executable`; `crates/recomp` picks the frontend per module.
2. **Discovery.** A32 seeds are `bl`/`b` targets, `bx`/`blx` register targets resolved
   through literal pools (`ldr rN, [pc, #imm]`), and `ldr pc, [...]`/`add pc, pc, ...`
   tables. The IR already exposes the target and the literal address, so the existing
   discovery walk can be parameterised instead of duplicated.
3. **Lowering.** Map to the existing `crate::ir::Stmt`/`Op`: results to
   `SetReg`/`SetSp`, flag updates to the `Flags*` ops, PC writes to
   `Jump`/`JumpIf`/`JumpIndirect`, `svc` to `Stmt::Svc`, `bkpt` to a trap.
   `Cond::field()` uses the same four-bit encoding as `Op::Cond`, so conditional
   execution can reuse the existing predicate evaluator (predicate the instruction, or
   emit one `JumpIf` per block).
4. **Runtime.** `crates/rt/src/cpu.rs` already keeps `x: [u64; 31]`, `sp`, `pc`, `nzcv`,
   so `r0`–`r12` fit in the low 32 bits of `x0`–`x12` without a second register file.
   The decisions left for the interpreter work item are the 32-bit wrap of results and
   addresses, `pc` reading as `address + 8`, and the banked `sp`/`lr` of the privileged
   modes.

## Evidence

- `cargo test --release -p kirikae-a64`: **54 tests pass** (26 unit, of which 2 are A32
  helper tests; 16 A32 integration; 7 assembler; 5 SIMD). The A32 suite pins known
  encodings per class, condition handling and evaluation, immediate rotation, branch
  targets (including the `blx` `H` bit and 32-bit wrap), LDM/STM lists and modes, memory
  addressing modes, the unpredictable/unallocated rejections, and 200,000 generated
  words that must decode or reject without panicking.
- The expected encodings and texts were checked against the ARM ARM A8.8 pages for
  ARMv7-A and cross-checked with an independent disassembler. Three encodings were
  fixed during review: the register form of `ldr`/`str` requires bit 4 = 0 (bit 4 = 1
  is the media space), `P = 0 && W = 1` is the unprivileged `ldrt`/`strt` form rather
  than an unpredictable one, and `mla`'s accumulator is bits 15:12 while the assembler
  order is `Rd, Rm, Rs, Ra`.
- Local differential against Capstone 5.0.7 (ARM mode) on **1,080,000 words**
  (1,000,000 uniform random plus 16 × 5,000 condition-swept words), 2026-10-25:
  - 505,324 decoded by both. After normalising `push`/`pop`, the `ldm`/`stm` mode
    aliases and `cs/cc` vs `hs/lo`, **0 mnemonic disagreements** remain except 190
    words where Capstone prints a shift alias (`asrs`, `lsls`, `lsrs`, `rors`) and this
    IR keeps `movs` plus a shift operand;
  - **0 words accepted here and rejected by Capstone** (no false accepts);
  - 432,375 words Capstone decodes and this slice rejects, all either outside the slice
    (coprocessor, VFP/NEON, media, unprivileged, `S`-bit block transfers, exclusives) or
    architecturally unpredictable (SBZ fields, `pc` operands, writeback over the base,
    byte transfers of `pc`).
  The differential script needs Capstone and lives outside the repository, like the
  scripts in `aarch32-confirmation.md`; it is not a test dependency and adds no crates.

These numbers measure the decoder, not the frontend: 99.8 % of the real MK8D `.text`
decodes as A32 in the sweep of `aarch32-confirmation.md`, but how much of that falls
inside this slice is a question for the full decoder, run over a local dump (step 1 of
"What remains").

## What remains

| Area | Work |
|---|---|
| ARM decoder completion | Media/parallel add/sub, packing/saturation (`pkh`, `ssat`, `usat`), byte reversal, halfword and DSP multiplies (`smul*`, `smla*`, `smlal*`), multiply-long, swap, exclusives (`ldrex`/`strex`), extra and doubleword loads (`ldrh`, `ldrsb`, `ldrd`), barriers/hints, `mrs`/`msr`, `cps`, `setend` |
| Thumb-2 | 16-bit and 32-bit encodings, `it` blocks, `cbz`/`tbb`, interworking; only needed if the measurement shows Thumb code (MK8D shows no `blx #imm` and all branch targets word-aligned, so its `main` is consistent with ARM-only; NSMBUDX is unmeasured) |
| VFP/NEON | Coprocessor 10/11: single/double arithmetic, `vmov` between core and FP registers, conversions. Likely needed (Wii U-era engines are float-heavy), and it is a second register file with its own short vector forms |
| Function discovery | A32 seeds and ranges, literal-pool resolution, jump tables, ARM/Thumb interworking, coverage/`.profile` integration, plugin hooks |
| Interpreter | 16 registers with `pc + 8` reads, CPSR condition execution, banked `sp`/`lr`, aligned/unaligned access rules, exclusive monitor, `svc`/`bkpt` traps |
| Backend | Lowering to `Stmt`/Cranelift, flag computation, predication, PC writes as control flow, 32-bit wrap |
| Bring-up | Architecture tag end to end, headless boot of MK8D, triage loop, NSMBUDX |

Not required: big-endian support (the Switch is little-endian; `setend` is rejected).
A second address space is not needed either: a 32-bit process uses the low 4 GiB of the
guest's existing map, and the only extra rule is the 32-bit wrap already listed for the
interpreter and the decoder.

## Estimate

6–8 weeks for one experienced developer (≈33–42 person-days), on the assumption that the
two titles are a product goal:

| Week | Work | Days |
|---|---|---|
| 1 | Finish the ARM decoder (table above) and run it over the MK8D `.text` to measure in-slice coverage; differential harness against Capstone | 5 |
| 2 | Thumb-2 if the measurement demands it (otherwise this week goes to VFP) | 4 |
| 3 | VFP/NEON, sized by a count of VFP instructions in MK8D | 5 |
| 4 | A32 interpreter and memory semantics, synthetic programs from `testkit` | 5 |
| 5 | Discovery, architecture tag through loader/recomp, coverage integration | 5 |
| 6–7 | Lowering to `Stmt` and Cranelift, flags, predication, PC writes | 9 |
| 8 | Bring-up, triage, documentation | 4 |

Confidence is moderate: the decoder is the best-understood part and this slice shows it
is small (≈1,360 lines including documentation, plus ≈520 lines of tests). The variance
sits in VFP/NEON scope (unmeasured), Thumb-2 presence (unmeasured for NSMBUDX), and how
much of the A64-era runtime assumes 64-bit pointers and ABI details when the first A32
title boots. GPU/services gaps that appear after boot are outside WS12 and will dominate
the effort if the goal is "playable", not "reaches the menu".

## Reproduce

```
cargo test --release -p kirikae-a64
cargo clippy --release -p kirikae-a64 --all-targets -- -D warnings
cargo fmt --all --check
```

The differential needs a local Python with Capstone 5.x and a small binary that prints
`decode(pc, word)` results; both live in the session directory, not in the repository.
The same harness run over a local `main` dump is the first task of week 1.
