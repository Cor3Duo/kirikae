//! AArch32 (ARM mode) decoder and IR — proof of concept for WS12.
//!
//! Two titles in the collection (Mario Kart 8 Deluxe 1.0 and New Super Mario Bros. U
//! Deluxe) ship an A32 `main` module, so the A64 frontend cannot see their code at all.
//! See `docs/plan/findings/aarch32.md` and `docs/plan/findings/aarch32-frontend.md`.
//!
//! This module is self-contained on purpose: it does not import the A64 lifter, the
//! recompiler or the runtime. `crates/rt/src/interp_a32.rs` executes the slice as an
//! interpreter; the Cranelift lowering is still a work item, so discovery and the
//! generated project do not know about A32 yet.
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
//! - `ldr`/`str` (byte and word), `ldrh`/`strh`, `ldrsb`/`ldrsh`, `ldrd`/`strd`, and
//!   `ldrex`/`strex` with their byte and halfword forms, plus the ARMv8 acquire/release
//!   loads and stores (`lda`/`ldab`/`ldah`, `stl`/`stlb`/`stlh`);
//! - `ldm`/`stm` in all four addressing modes;
//! - `mul`/`mla`, `umull`/`umlal`/`smull`/`smlal`, and the halfword multiplies
//!   (`smla<x><y>`, `smlaw<y>`, `smul<x><y>`, `smulw<y>`, `smlal<x><y>`);
//! - the media dual multiplies (`smuad`/`smusd`/`smlad`/`smlsd` with their `x` forms,
//!   `smlald`/`smlsld`, and `smmul`/`smmla`/`smmls` with their rounding forms) and the
//!   integer divide (`udiv`/`sdiv`);
//! - the media instructions: `uxtb`/`uxth`/`sxtb`/`sxth`, their 16-bit and accumulating
//!   forms, `rev`/`rev16`/`revsh`/`rbit`/`clz`, `pkhbt`/`pkhtb`, `sel`, `ssat`/`usat`
//!   and their 16-bit forms, `bfc`/`bfi`/`sbfx`/`ubfx`, and the whole parallel
//!   add/subtract family (`sadd16` … `uhsub8`);
//! - `qadd`/`qsub`/`qdadd`/`qdsub` and the system instructions `mrs`, `msr` of the
//!   flags byte, the hints (`nop`, `yield`, `wfe`, `wfi`, `sev`) and `mrc`/`mcr` of the
//!   two `cp15` thread pointers;
//! - `svc` and `bkpt` so the discovery loop can stop at a supervisor call or a trap;
//! - the unconditional-space hints and barriers (`nop`, `yield`, `wfe`, `wfi`, `sev`,
//!   `sevl`, `dsb`, `dmb`, `isb`, `clrex`) and `pld`, which the interpreter models as
//!   the no-ops they are for the host, and `mrrc` of the two 64-bit counters;
//! - VFP (`vfp.rs`): `vldr`/`vstr`, `vldm`/`vstm`/`vpush`/`vpop`, `vmov` in every form
//!   (register, immediate, core register, core pair and 8/16/32-bit element), `vdup`,
//!   `vmrs`/`vmsr`, the fused and non-fused arithmetic, `vabs`/`vneg`/`vsqrt`, `vcmp`/
//!   `vcmpe` with FPSCR flags, `vcvt` between precisions and to and from integers, and
//!   `vrintr`/`vrintz`/`vrintx`;
//! - Advanced SIMD (`neon.rs`): `vld1`…`vld4`/`vst1`…`vst4` structure transfers,
//!   three-register integer and float arithmetic, comparisons and bitwise logic,
//!   `vabs`/`vneg`/`vrev`, `vext` and the modified immediates.
//!
//! Everything else is rejected with a [`DecodeError`], never guessed and never panics.
//! That includes the deprecated `swp`, any `msr` of the control or status fields, the
//! VFP fixed-point conversions and two-single `vmov`, the Advanced SIMD table lookups,
//! lane transfers, shift-by-immediate, widening/narrowing, zip/unzip/transpose and
//! `vswp`, and the `cp15` registers other than the thread pointers, the counters and
//! `fpscr`.
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
pub mod neon;
pub mod opcode;
pub mod vfp;

pub use decode::{DecodeError, decode, describe_unsupported};
pub use ir::{
    A32SysReg, Block, BlockMode, Cond, Convert, DReg, ExecState, Insn, Mem, MemSize, NeonMem,
    Offset, Operand, PostUpdate, Reg, RegList, SReg, Shift, Target, VBlock, VecShape, VfpDest,
    VfpSys,
};
pub use opcode::Opcode;
