//! AArch32 (ARM mode) decoder and IR — proof of concept for WS12.
//!
//! Two titles in the collection (Mario Kart 8 Deluxe 1.0 and New Super Mario Bros. U
//! Deluxe) ship an A32 `main` module, so the A64 frontend cannot see their code at all.
//! See `docs/plan/findings/aarch32.md` and `docs/plan/findings/aarch32-frontend.md`.
//!
//! This module is self-contained on purpose: it does not import the A64 lifter, the
//! recompiler or the runtime, and nothing calls it yet. It answers one question —
//! whether a small, exact A32 decoder and a matching IR fit the shape of the project —
//! before committing to a full frontend.
//!
//! # Scope
//!
//! ARM mode only. The slice covers:
//!
//! - the condition field, including `blx` immediate from the unconditional space;
//! - data processing (`and`, `eor`, `sub`, `rsb`, `add`, `adc`, `sbc`, `rsc`, `tst`,
//!   `teq`, `cmp`, `cmn`, `orr`, `mov`, `bic`, `mvn`) with immediate or register/shifted
//!   operands, and `movw`/`movt`;
//! - `b`, `bl`, `bx`, `blx`;
//! - `ldr`/`str` (byte and word, immediate or register offset, pre/post-indexed);
//! - `ldm`/`stm` in all four addressing modes;
//! - `mul`/`mla`;
//! - `svc` and `bkpt` so the discovery loop can stop at a supervisor call or a trap.
//!
//! Everything else is rejected with a [`DecodeError`], never guessed and never panics.
//!
//! # Intended integration (not wired yet)
//!
//! 1. **Architecture tag.** `crates/loader/src/package.rs` already classifies the image
//!    (`npdm_is_aarch32`: `META.Flags0` bit 0 with a code-sample fallback) and today
//!    rejects the title with a clear error. WS12 turns that rejection into an
//!    architecture tag on `Executable`, so `crates/recomp` picks a frontend per module
//!    instead of assuming A64.
//! 2. **Discovery.** `crates/recomp/src/discover.rs` seeds functions from the entry
//!    point, exports and `bl` targets. The A32 seeds are `bl`/`b` targets, `bx`/`blx`
//!    register targets resolved through literal pools (`ldr rN, [pc, #imm]`), and
//!    `ldr pc, [...]` tables. The A32 IR exposes [`Insn::target`] and the `pc`-relative
//!    `Mem` of a literal load (base `pc` plus offset), so the same walk can be written
//!    once.
//! 3. **Lowering.** The IR is shaped to lower to the existing `crate::ir::Stmt` list:
//!    `Stmt::SetReg`/`SetSp` for results, an op producing NZCV for `sets_flags`,
//!    `Stmt::Jump`/`JumpIf`/`JumpIndirect` for PC writes, `Stmt::Svc` for `svc` and a
//!    trap for `bkpt`. [`Cond`] uses the same four-bit field as `Op::Cond`, so
//!    conditional execution can reuse the existing predicate evaluator.
//! 4. **Runtime.** `crates/rt/src/cpu.rs` keeps `x: [u64; 31]`, `sp`, `pc` and `nzcv`,
//!    so A32 `r0`-`r12` map onto the low 32 bits of `x0`-`x12` without a second register
//!    file. The open questions are the 32-bit wrap of results and addresses, the banked
//!    `sp`/`lr` of the privileged modes, and `pc` reading as `address + 8`; those need a
//!    decision in the interpreter work item, not in the decoder.
//!
//! A32 is a 32-bit architecture: addresses and immediates here are `u32`, and branch
//! targets wrap modulo 2^32.

pub mod decode;
pub mod ir;

pub use decode::{DecodeError, decode};
pub use ir::{
    Block, BlockMode, Cond, ExecState, Insn, Mem, MemSize, Offset, Opcode, Operand, Reg, RegList,
    Shift, Target,
};
