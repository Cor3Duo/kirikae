//! AArch32 (ARM mode) interpreter.
//!
//! This runs the classes `kirikae_a64::a32::decode` accepts, with the
//! architectural 32-bit register file and PC reads as `address + 8`. A32 is a
//! 32-bit architecture, so results, addresses and branch targets wrap modulo
//! 2^32; the guest map is still the low 4 GiB of the 64-bit reservation.
//!
//! VFP and Advanced SIMD are modelled here too, at the bottom of this file: the `d`
//! registers are the 32 64-bit slots of [`Cpu::v`], the `s` registers alias them in
//! pairs, and `fpsr` holds the A32 FPSCR. The scalar arithmetic reuses the shared
//! `kirikae_a64::simd::fp` model, so both frontends round, propagate NaNs and flush
//! denormals the same way.
//!
//! Out of scope, reported instead of guessed: Thumb state, the banked `sp`/`lr`
//! of the privileged modes (and therefore `s` writes to `pc`), the VFP fixed-point
//! conversions and two-single `vmov`, the Advanced SIMD table lookups, lane transfers,
//! shifts by immediate, widening/narrowing, zip/unzip/transpose and `vswp`, the `cp15`
//! registers other than the two thread pointers, the two 64-bit counters and `fpscr`,
//! and any encoding the decoder rejects.
//!
//! The register mapping is the one [`crate::cpu::Cpu`] already has: `r0`-`r12`
//! are the low halves of `x0`-`x12`, `r13` is `sp`, `r14` is `x30` and `r15`
//! is [`Cpu::pc`]. `nzcv` holds N in bit 31, Z in 30, C in 29 and V in 28, the
//! same packing the A64 IR uses, so [`Cond::holds`] evaluates directly.
//!
//! # The modelled CPSR
//!
//! For an A32 process `nzcv` is the base of a 32-bit CPSR, not just NZCV: bit 27
//! is Q (set by `qadd`…`qdsub`, `ssat` and `usat`), and bits 19:16 are GE (set by
//! `sadd8`/`ssub8`/`uadd8`/`usub8` and read by `sel`). Those two fields are the
//! only processor state the media instructions in this slice add, and keeping
//! them in the same word lets `mrs`/`msr` of the flags byte stay exact. The mode
//! bits are not stored: `mrs` reports the user mode value (`0x10`) with the T bit
//! clear, because that is the only state this runtime models, and `msr` of any
//! other field is rejected by the decoder.

use std::sync::Arc;

use kirikae_a64::a32::{
    self, BlockMode, Convert, DReg, DecodeError, Insn, MemSize, Offset, Opcode, Operand,
    PostUpdate, Reg, SReg, Shift, VecShape, VfpDest, VfpSys,
};
use kirikae_a64::bits::sext;
use kirikae_a64::simd::fp::{self, Fpcr, Round};

use crate::cpu::Cpu;
use crate::horizon;

const N: u32 = 1 << 31;
const Z: u32 = 1 << 30;
const C: u32 = 1 << 29;
const V: u32 = 1 << 28;
const Q: u32 = 1 << 27;
const GE_SHIFT: u32 = 16;
const GE_MASK: u32 = 0xf << GE_SHIFT;
/// NZCVQ, the fields `msr` of the flags byte and `mrs` report.
const FLAGS: u32 = N | Z | C | V | Q;
/// User mode, with the T bit clear: the only state this runtime models.
const CPSR_USER: u32 = 0x10;

/// Runs the instruction at `c.pc`.
pub fn step(c: &mut Cpu) {
    let p = c.process();
    p.stats.count_interpreted();
    let pc = c.pc;
    if pc > u32::MAX as u64 || !pc.is_multiple_of(4) || !p.memory.is_mapped(pc, 4) {
        horizon::fault(
            c,
            format!("AArch32 jumped to unmapped or misaligned address {pc:#x}"),
        );
        return;
    }
    let cached = p.interp_cache32.read().unwrap().get(&pc).cloned();
    let insn = match cached {
        Some(insn) => insn,
        None => {
            let word = c.ld32(pc) as u32;
            match a32::decode(pc as u32, word) {
                Ok(insn) => {
                    let insn = Arc::new(insn);
                    p.interp_cache32.write().unwrap().insert(pc, insn.clone());
                    insn
                }
                Err(error) => {
                    let at = p.describe(pc);
                    horizon::fatal(
                        c,
                        format!(
                            "unimplemented AArch32 instruction {word:#010x} at {at}: {}",
                            rejection(word, error)
                        ),
                    );
                    return;
                }
            }
        }
    };
    exec(c, &insn);
}

/// Why the decoder rejected a word, in the terms the report needs: the class
/// that is missing, not just "unsupported".
fn rejection(word: u32, error: DecodeError) -> &'static str {
    match error {
        DecodeError::Unallocated => "no instruction is assigned to this encoding",
        DecodeError::Unpredictable => "the encoding is architecturally unpredictable",
        DecodeError::Deprecated => {
            "the deprecated swap instructions (`swp`/`swpb`) are not modelled"
        }
        DecodeError::Privileged => {
            "the encoding needs banked modes, the SPSR or a control field of the CPSR"
        }
        DecodeError::Unsupported => a32::describe_unsupported(word),
    }
}

fn malformed(c: &mut Cpu, insn: &Insn) {
    horizon::fatal(
        c,
        format!(
            "AArch32 decoder produced an operand layout for `{insn}` that the interpreter does not know"
        ),
    );
}

#[inline]
fn get(c: &Cpu, reg: Reg, pc: u32) -> u32 {
    match reg.number() {
        15 => pc.wrapping_add(8),
        13 => c.sp as u32,
        14 => c.x[30] as u32,
        n => c.x[n as usize] as u32,
    }
}

#[inline]
fn set(c: &mut Cpu, reg: Reg, value: u32) {
    match reg.number() {
        15 => c.pc = u64::from(value),
        13 => c.sp = u64::from(value),
        14 => c.x[30] = u64::from(value),
        n => c.x[n as usize] = u64::from(value),
    }
}

#[inline]
fn next(c: &mut Cpu, insn: &Insn) {
    c.pc = u64::from(insn.next());
}

/// Writes `pc` as an interworking branch does: bit 0 selects Thumb, and bit 1
/// set is an unaligned ARM address. Neither state exists here, so both stop
/// with the instruction that produced them.
fn branch_to(c: &mut Cpu, insn: &Insn, value: u32) {
    if value & 1 != 0 {
        horizon::fatal(
            c,
            format!(
                "AArch32 Thumb state is not implemented: `{insn}` at {:#x} branches to {value:#x}",
                u64::from(insn.pc)
            ),
        );
        return;
    }
    if value & 2 != 0 {
        horizon::fatal(
            c,
            format!(
                "AArch32 branch to unaligned address: `{insn}` at {:#x} goes to {value:#x}",
                u64::from(insn.pc)
            ),
        );
        return;
    }
    c.pc = u64::from(value);
}

#[inline]
fn flag(c: &Cpu, mask: u32) -> bool {
    c.nzcv as u32 & mask != 0
}

#[inline]
fn cpsr(c: &Cpu) -> u32 {
    c.nzcv as u32 & (FLAGS | GE_MASK)
}

#[inline]
fn set_cpsr(c: &mut Cpu, value: u32) {
    c.nzcv = u64::from(value & (FLAGS | GE_MASK));
}

#[inline]
fn set_q(c: &mut Cpu) {
    c.nzcv |= u64::from(Q);
}

#[inline]
fn ge(c: &Cpu) -> u32 {
    (c.nzcv as u32 >> GE_SHIFT) & 0xf
}

#[inline]
fn set_ge(c: &mut Cpu, value: u32) {
    let kept = c.nzcv as u32 & !GE_MASK;
    c.nzcv = u64::from(kept | ((value & 0xf) << GE_SHIFT));
}

/// Updates NZCV and leaves Q and GE alone, as a flag-setting instruction does.
fn set_flags(c: &mut Cpu, result: u32, carry: Option<bool>, overflow: Option<bool>) {
    let old = c.nzcv as u32;
    let mut psr = old & (Q | GE_MASK);
    if result & N != 0 {
        psr |= N;
    }
    if result == 0 {
        psr |= Z;
    }
    if carry.unwrap_or(old & C != 0) {
        psr |= C;
    }
    if overflow.unwrap_or(old & V != 0) {
        psr |= V;
    }
    c.nzcv = u64::from(psr);
}

/// Updates NZCV from a 64-bit result (`umull`, `smlal`): N is bit 63 and Z is the
/// whole 64-bit value. C and V are unaffected.
fn set_flags64(c: &mut Cpu, result: u64) {
    let old = c.nzcv as u32;
    let mut psr = old & (Q | GE_MASK);
    if result >> 63 != 0 {
        psr |= N;
    }
    if result == 0 {
        psr |= Z;
    }
    psr |= old & (C | V);
    c.nzcv = u64::from(psr);
}

/// `ExpandImm_C`: with a zero rotation the carry out is the old C flag; with a
/// rotation it is the last bit rotated out, bit 31 of the expanded value.
#[inline]
fn modified_immediate_carry(word: u32, value: u32, carry: bool) -> bool {
    if (word >> 8) & 0xf == 0 {
        carry
    } else {
        value & N != 0
    }
}

fn shift_lsl(value: u32, amount: u32, carry: bool) -> (u32, bool) {
    match amount {
        0 => (value, carry),
        1..=31 => (value << amount, (value >> (32 - amount)) & 1 != 0),
        32 => (0, value & 1 != 0),
        _ => (0, false),
    }
}

fn shift_lsr(value: u32, amount: u32, carry: bool) -> (u32, bool) {
    match amount {
        0 => (value, carry),
        1..=31 => (value >> amount, (value >> (amount - 1)) & 1 != 0),
        32 => (0, value & N != 0),
        _ => (0, false),
    }
}

fn shift_asr(value: u32, amount: u32, carry: bool) -> (u32, bool) {
    match amount {
        0 => (value, carry),
        1..=31 => (
            ((value as i32) >> amount) as u32,
            (value >> (amount - 1)) & 1 != 0,
        ),
        _ => (((value as i32) >> 31) as u32, value & N != 0),
    }
}

fn shift_ror(value: u32, amount: u32, carry: bool) -> (u32, bool) {
    if amount == 0 {
        return (value, carry);
    }
    let amount = amount % 32;
    if amount == 0 {
        (value, value & N != 0)
    } else {
        (value.rotate_right(amount), (value >> (amount - 1)) & 1 != 0)
    }
}

fn shift_value(c: &Cpu, pc: u32, reg: Reg, shift: Shift) -> (u32, bool) {
    let value = get(c, reg, pc);
    let carry = flag(c, C);
    match shift {
        Shift::Lsl(n) => shift_lsl(value, u32::from(n), carry),
        Shift::Lsr(n) => shift_lsr(value, u32::from(n), carry),
        Shift::Asr(n) => shift_asr(value, u32::from(n), carry),
        Shift::Ror(n) => shift_ror(value, u32::from(n), carry),
        Shift::Rrx => ((u32::from(carry) << 31) | (value >> 1), value & 1 != 0),
        Shift::LslReg(r) => shift_lsl(value, get(c, r, pc) & 0xff, carry),
        Shift::LsrReg(r) => shift_lsr(value, get(c, r, pc) & 0xff, carry),
        Shift::AsrReg(r) => shift_asr(value, get(c, r, pc) & 0xff, carry),
        Shift::RorReg(r) => shift_ror(value, get(c, r, pc) & 0xff, carry),
    }
}

fn second_operand(c: &Cpu, insn: &Insn, operand: &Operand) -> Option<(u32, bool)> {
    match operand {
        Operand::Imm(value) => Some((
            *value,
            modified_immediate_carry(insn.word, *value, flag(c, C)),
        )),
        Operand::Shifted { reg, shift } => Some(shift_value(c, insn.pc, *reg, *shift)),
        _ => None,
    }
}

/// The `<op2>` of a media instruction: a register with a rotate, without a carry
/// out.
fn shifted_operand(c: &Cpu, insn: &Insn, operand: &Operand) -> Option<u32> {
    match operand {
        Operand::Shifted { reg, shift } => Some(shift_value(c, insn.pc, *reg, *shift).0),
        _ => None,
    }
}

fn add_with_carry(a: u32, b: u32, carry: bool) -> (u32, bool, bool) {
    let (sum, carry1) = a.overflowing_add(b);
    let (result, carry2) = sum.overflowing_add(u32::from(carry));
    let overflow = (a ^ result) & (b ^ result) & N != 0;
    (result, carry1 || carry2, overflow)
}

fn sub_with_carry(a: u32, b: u32, carry: bool) -> (u32, bool, bool) {
    let (difference, borrow1) = a.overflowing_sub(b);
    let (result, borrow2) = difference.overflowing_sub(u32::from(!carry));
    let overflow = (a ^ b) & (a ^ result) & N != 0;
    (result, !(borrow1 || borrow2), overflow)
}

/// Signed saturation to `bits` bits, returning the value and whether it clamped.
fn saturate_signed(value: i64, bits: u32) -> (u32, bool) {
    let max = (1i64 << (bits - 1)) - 1;
    let min = -(1i64 << (bits - 1));
    if value > max {
        (max as u32, true)
    } else if value < min {
        (min as u32, true)
    } else {
        (value as u32, false)
    }
}

/// Unsigned saturation to `bits` bits, clamping negatives to zero.
fn saturate_unsigned(value: i64, bits: u32) -> (u32, bool) {
    debug_assert!(bits < 32, "unsigned saturation to 32 bits cannot clamp");
    let max = (1i64 << bits) - 1;
    if value < 0 {
        (0, true)
    } else if value > max {
        (max as u32, true)
    } else {
        (value as u32, false)
    }
}

/// Signed saturation to 32 bits.
fn saturate_signed32(value: i64) -> (u32, bool) {
    if value > i64::from(i32::MAX) {
        (i32::MAX as u32, true)
    } else if value < i64::from(i32::MIN) {
        (i32::MIN as u32, true)
    } else {
        (value as u32, false)
    }
}

fn data_processing(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let operands = insn.operands.as_slice();
    let (src, op2) = if insn.opcode.is_flag_test() {
        let [Operand::Reg(rn), op2] = operands else {
            return malformed(c, insn);
        };
        (*rn, op2)
    } else if matches!(insn.opcode, Opcode::Mov | Opcode::Mvn) {
        let [Operand::Reg(_), op2] = operands else {
            return malformed(c, insn);
        };
        (Reg::R0, op2)
    } else {
        let [Operand::Reg(_), Operand::Reg(rn), op2] = operands else {
            return malformed(c, insn);
        };
        (*rn, op2)
    };
    let Some((b, shifter_carry)) = second_operand(c, insn, op2) else {
        return malformed(c, insn);
    };
    let a = get(c, src, pc);

    let flags = insn.sets_flags;
    let (result, carry, overflow) = match insn.opcode {
        Opcode::And | Opcode::Tst => (a & b, Some(shifter_carry), None),
        Opcode::Eor | Opcode::Teq => (a ^ b, Some(shifter_carry), None),
        Opcode::Orr => (a | b, Some(shifter_carry), None),
        Opcode::Bic => (a & !b, Some(shifter_carry), None),
        Opcode::Mov => (b, Some(shifter_carry), None),
        Opcode::Mvn => (!b, Some(shifter_carry), None),
        Opcode::Add | Opcode::Cmn => {
            let (result, carry, overflow) = add_with_carry(a, b, false);
            (result, Some(carry), Some(overflow))
        }
        Opcode::Adc => {
            let (result, carry, overflow) = add_with_carry(a, b, flag(c, C));
            (result, Some(carry), Some(overflow))
        }
        Opcode::Sub | Opcode::Cmp => {
            let (result, carry, overflow) = sub_with_carry(a, b, true);
            (result, Some(carry), Some(overflow))
        }
        Opcode::Sbc => {
            let (result, carry, overflow) = sub_with_carry(a, b, flag(c, C));
            (result, Some(carry), Some(overflow))
        }
        Opcode::Rsb => {
            let (result, carry, overflow) = sub_with_carry(b, a, true);
            (result, Some(carry), Some(overflow))
        }
        Opcode::Rsc => {
            let (result, carry, overflow) = sub_with_carry(b, a, flag(c, C));
            (result, Some(carry), Some(overflow))
        }
        _ => return malformed(c, insn),
    };

    if flags {
        set_flags(c, result, carry, overflow);
    }
    if insn.opcode.is_flag_test() {
        next(c, insn);
        return;
    }
    let Some(Operand::Reg(rd)) = operands.first() else {
        return malformed(c, insn);
    };
    let rd = *rd;
    if rd == Reg::PC {
        if insn.sets_flags {
            horizon::fatal(
                c,
                format!(
                    "AArch32 exception return (`s` write to pc) is not implemented: `{insn}` at {:#x}",
                    u64::from(pc)
                ),
            );
            return;
        }
        branch_to(c, insn, result);
        return;
    }
    set(c, rd, result);
    next(c, insn);
}

fn wide_move(c: &mut Cpu, insn: &Insn) {
    let [Operand::Reg(rd), Operand::Imm(imm)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    if *rd == Reg::PC {
        return malformed(c, insn);
    }
    let value = match insn.opcode {
        Opcode::Movw => *imm,
        Opcode::Movt => (get(c, *rd, insn.pc) & 0x0000_ffff) | (imm << 16),
        _ => return malformed(c, insn),
    };
    set(c, *rd, value);
    next(c, insn);
}

fn multiply(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let operands = insn.operands.as_slice();
    let (rd, rm, rs, addend) = match operands {
        [Operand::Reg(rd), Operand::Reg(rm), Operand::Reg(rs)] => (*rd, *rm, *rs, None),
        [
            Operand::Reg(rd),
            Operand::Reg(rm),
            Operand::Reg(rs),
            Operand::Reg(ra),
        ] => (*rd, *rm, *rs, Some(*ra)),
        _ => return malformed(c, insn),
    };
    if rd == Reg::PC {
        return malformed(c, insn);
    }
    let mut result = get(c, rm, pc).wrapping_mul(get(c, rs, pc));
    if let Some(ra) = addend {
        result = result.wrapping_add(get(c, ra, pc));
    }
    if insn.sets_flags {
        set_flags(c, result, None, None);
    }
    set(c, rd, result);
    next(c, insn);
}

fn long_multiply(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let [Operand::Regs { lo, hi }, Operand::Reg(rm), Operand::Reg(rs)] = insn.operands.as_slice()
    else {
        return malformed(c, insn);
    };
    let (lo, hi) = (*lo, *hi);
    let a = get(c, *rm, pc);
    let b = get(c, *rs, pc);
    let signed = matches!(insn.opcode, Opcode::Smull | Opcode::Smlal);
    let mut result = if signed {
        (i64::from(a as i32) as u64).wrapping_mul(i64::from(b as i32) as u64)
    } else {
        u64::from(a).wrapping_mul(u64::from(b))
    };
    if matches!(insn.opcode, Opcode::Umlal | Opcode::Smlal) {
        let acc = (get(c, hi, pc) as u64) << 32 | u64::from(get(c, lo, pc));
        result = result.wrapping_add(acc);
    }
    if insn.sets_flags {
        set_flags64(c, result);
    }
    set(c, lo, result as u32);
    set(c, hi, (result >> 32) as u32);
    next(c, insn);
}

/// Which half of a register a halfword multiply reads: the low half (`b`) or the
/// high half (`t`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Half {
    Bottom,
    Top,
}

impl Half {
    fn of(self, value: u32) -> i32 {
        match self {
            Half::Bottom => i32::from(value as u16 as i16),
            Half::Top => i32::from((value >> 16) as u16 as i16),
        }
    }
}

fn halfword_multiply(c: &mut Cpu, insn: &Insn) {
    use Opcode::*;
    let pc = insn.pc;
    let operands = insn.operands.as_slice();
    // `(x, y)` selects the halfword of `Rm` and of `Rs`; the `w` forms read `Rs` only.
    let (x, y) = match insn.opcode {
        Smlabb | Smlalbb | Smulbb | Smlawb | Smulwb => (Half::Bottom, Half::Bottom),
        Smlabt | Smlalbt | Smulbt | Smlawt | Smulwt => (Half::Bottom, Half::Top),
        Smlatb | Smlaltb | Smultb => (Half::Top, Half::Bottom),
        Smlatt | Smlaltt | Smultt => (Half::Top, Half::Top),
        _ => return malformed(c, insn),
    };
    match insn.opcode {
        Smlalbb | Smlalbt | Smlaltb | Smlaltt => {
            let [Operand::Regs { lo, hi }, Operand::Reg(rm), Operand::Reg(rs)] = operands else {
                return malformed(c, insn);
            };
            let product =
                i64::from(x.of(get(c, *rm, pc))).wrapping_mul(i64::from(y.of(get(c, *rs, pc))));
            let acc = (get(c, *hi, pc) as u64) << 32 | u64::from(get(c, *lo, pc));
            let result = acc.wrapping_add(product as u64);
            set(c, *lo, result as u32);
            set(c, *hi, (result >> 32) as u32);
        }
        Smulbb | Smulbt | Smultb | Smultt => {
            let [Operand::Reg(rd), Operand::Reg(rm), Operand::Reg(rs)] = operands else {
                return malformed(c, insn);
            };
            let product = x.of(get(c, *rm, pc)).wrapping_mul(y.of(get(c, *rs, pc)));
            set(c, *rd, product as u32);
        }
        Smlawb | Smlawt | Smulwb | Smulwt => {
            let (rd, rm, rs, ra) = match operands {
                [
                    Operand::Reg(rd),
                    Operand::Reg(rm),
                    Operand::Reg(rs),
                    Operand::Reg(ra),
                ] => (*rd, *rm, *rs, Some(*ra)),
                [Operand::Reg(rd), Operand::Reg(rm), Operand::Reg(rs)] => (*rd, *rm, *rs, None),
                _ => return malformed(c, insn),
            };
            let mut value =
                i64::from(get(c, rm, pc) as i32).wrapping_mul(i64::from(y.of(get(c, rs, pc))));
            if let Some(ra) = ra {
                value = value.wrapping_add(i64::from(get(c, ra, pc) as i32) << 16);
            }
            // The shifted result is truncated to 32 bits; Q reports that it did not
            // fit.
            let shifted = value >> 16;
            if shifted < i64::from(i32::MIN) || shifted > i64::from(i32::MAX) {
                set_q(c);
            }
            set(c, rd, shifted as u32);
        }
        _ => {
            let [
                Operand::Reg(rd),
                Operand::Reg(rm),
                Operand::Reg(rs),
                Operand::Reg(ra),
            ] = operands
            else {
                return malformed(c, insn);
            };
            let product = x.of(get(c, *rm, pc)).wrapping_mul(y.of(get(c, *rs, pc)));
            let (sum, overflow) = product.overflowing_add(get(c, *ra, pc) as i32);
            if overflow {
                set_q(c);
            }
            set(c, *rd, sum as u32);
        }
    }
    next(c, insn);
}

/// A halfword of a register as a signed value.
#[inline]
fn half_signed(value: u32, top: bool) -> i64 {
    if top {
        i64::from((value >> 16) as u16 as i16)
    } else {
        i64::from(value as u16 as i16)
    }
}

/// The two signed halfword products of the media dual multiplies. `exchange` swaps the
/// halves of the second operand, which is the `<x>` suffix of `smladx` and friends.
#[inline]
fn dual_products(rn: u32, rm: u32, exchange: bool) -> (i64, i64) {
    let (n_lo, n_hi) = (half_signed(rn, false), half_signed(rn, true));
    let (m_lo, m_hi) = (half_signed(rm, false), half_signed(rm, true));
    if exchange {
        (n_lo * m_hi, n_hi * m_lo)
    } else {
        (n_lo * m_lo, n_hi * m_hi)
    }
}

/// The media dual multiplies and the top-word multiplies.
///
/// Q is set by the four 32-bit forms (`smuad`, `smusd`, `smlad`, `smlsd`) when the
/// signed result does not fit, and by nothing else: the 64-bit accumulators and the
/// `smm*` forms leave it alone.
fn media_multiply(c: &mut Cpu, insn: &Insn) {
    use Opcode::*;
    let pc = insn.pc;
    let operands = insn.operands.as_slice();
    let exchange = matches!(
        insn.opcode,
        Smuadx | Smusdx | Smladx | Smlsdx | Smlaldx | Smlsldx
    );
    let rounding = matches!(insn.opcode, Smmulr | Smmlar | Smmlsr);

    // The 64-bit accumulating pair names a register pair instead of a destination.
    if matches!(insn.opcode, Smlald | Smlaldx | Smlsld | Smlsldx) {
        let [Operand::Regs { lo, hi }, Operand::Reg(rn), Operand::Reg(rm)] = operands else {
            return malformed(c, insn);
        };
        let (first, second) = dual_products(get(c, *rn, pc), get(c, *rm, pc), exchange);
        let acc = (u64::from(get(c, *hi, pc)) << 32 | u64::from(get(c, *lo, pc))) as i64;
        let subtract = matches!(insn.opcode, Smlsld | Smlsldx);
        let value = if subtract {
            acc.wrapping_add(first).wrapping_sub(second)
        } else {
            acc.wrapping_add(first).wrapping_add(second)
        };
        set(c, *lo, value as u32);
        set(c, *hi, (value >> 32) as u32);
        return next(c, insn);
    }

    let (rd, rn, rm, ra) = match operands {
        [Operand::Reg(rd), Operand::Reg(rn), Operand::Reg(rm)] => (*rd, *rn, *rm, None),
        [
            Operand::Reg(rd),
            Operand::Reg(rn),
            Operand::Reg(rm),
            Operand::Reg(ra),
        ] => (*rd, *rn, *rm, Some(*ra)),
        _ => return malformed(c, insn),
    };

    let n = get(c, rn, pc);
    let m = get(c, rm, pc);

    // `smmul`, `smmla` and `smmls` keep the top word of a 64-bit result, with an
    // optional rounding constant.
    if matches!(
        insn.opcode,
        Smmul | Smmulr | Smmla | Smmlar | Smmls | Smmlsr
    ) {
        let product = i64::from(n as i32) * i64::from(m as i32);
        let addend = match ra {
            Some(ra) => i64::from(get(c, ra, pc) as i32) << 32,
            None => 0,
        };
        let mut value = if matches!(insn.opcode, Smmls | Smmlsr) {
            addend.wrapping_sub(product)
        } else {
            addend.wrapping_add(product)
        };
        if rounding {
            value = value.wrapping_add(0x8000_0000);
        }
        set(c, rd, (value >> 32) as u32);
        return next(c, insn);
    }

    let (first, second) = dual_products(n, m, exchange);
    let subtract = matches!(insn.opcode, Smusd | Smusdx | Smlsd | Smlsdx);
    let accumulate = matches!(insn.opcode, Smlad | Smladx | Smlsd | Smlsdx);
    let mut sum = if accumulate {
        i64::from(get(c, ra.unwrap_or(Reg::R0), pc) as i32)
    } else {
        0
    };
    sum = if subtract {
        sum.wrapping_add(first).wrapping_sub(second)
    } else {
        sum.wrapping_add(first).wrapping_add(second)
    };
    if sum < i64::from(i32::MIN) || sum > i64::from(i32::MAX) {
        set_q(c);
    }
    set(c, rd, sum as u32);
    next(c, insn);
}

/// `udiv`/`sdiv`: a division by zero yields zero, the signed overflow of
/// `INT_MIN / -1` yields `INT_MIN`, and neither form touches NZCV or Q.
fn divide(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let [Operand::Reg(rd), Operand::Reg(rn), Operand::Reg(rm)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let dividend = get(c, *rn, pc);
    let divisor = get(c, *rm, pc);
    let result = match insn.opcode {
        // A zero divisor answers zero instead of trapping, and the signed form's
        // `INT_MIN / -1` is the wrapping quotient the architecture defines.
        Opcode::Udiv => dividend.checked_div(divisor).unwrap_or(0),
        Opcode::Sdiv => {
            if divisor == 0 {
                0
            } else {
                // The only overflow here is `INT_MIN / -1`, which the architecture
                // defines as `INT_MIN` rather than trapping.
                (dividend as i32).overflowing_div(divisor as i32).0 as u32
            }
        }
        _ => return malformed(c, insn),
    };
    set(c, *rd, result);
    next(c, insn);
}

/// One lane of a parallel operation: the operands keep their signedness and the
/// result is rounded or saturated according to the mnemonic group.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    SignedWrap,
    SignedSat,
    SignedHalf,
    UnsignedWrap,
    UnsignedSat,
    UnsignedHalf,
}

impl Lane {
    fn signed(self) -> bool {
        matches!(self, Lane::SignedWrap | Lane::SignedSat | Lane::SignedHalf)
    }

    /// Reads lane `index` of a 16-bit or 8-bit decomposition as an integer.
    fn read(self, value: u32, index: u32, bits: u32) -> i64 {
        let shift = index * bits;
        let mask = (1u32 << bits) - 1;
        let lane = (value >> shift) & mask;
        if self.signed() {
            let sign = 1u32 << (bits - 1);
            i64::from(((lane ^ sign).wrapping_sub(sign)) as i32)
        } else {
            i64::from(lane)
        }
    }

    /// Rounds `value` back into a lane. The parallel forms saturate silently: the
    /// ARM ARM states that Q is not affected even when one of them saturates.
    fn write(self, value: i64, bits: u32) -> u32 {
        let mask = (1u32 << bits) - 1;
        match self {
            Lane::SignedWrap | Lane::UnsignedWrap => (value as u32) & mask,
            Lane::SignedHalf | Lane::UnsignedHalf => ((value >> 1) as u32) & mask,
            Lane::SignedSat => saturate_signed(value, bits).0 & mask,
            Lane::UnsignedSat => saturate_unsigned(value, bits).0 & mask,
        }
    }
}

fn parallel(c: &mut Cpu, insn: &Insn) {
    use Opcode::*;
    let pc = insn.pc;
    let [Operand::Reg(rd), Operand::Reg(rn), Operand::Reg(rm)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let (rd, a, b) = (*rd, get(c, *rn, pc), get(c, *rm, pc));
    let lane = match insn.opcode {
        Sadd16 | Sadd8 => Lane::SignedWrap,
        Qadd16 | Qadd8 => Lane::SignedSat,
        Shadd16 | Shadd8 => Lane::SignedHalf,
        Uadd16 | Uadd8 => Lane::UnsignedWrap,
        Uqadd16 | Uqadd8 => Lane::UnsignedSat,
        Uhadd16 | Uhadd8 => Lane::UnsignedHalf,
        Sasx | Ssax => Lane::SignedWrap,
        Qasx | Qsax => Lane::SignedSat,
        Shasx | Shsax => Lane::SignedHalf,
        Uasx | Usax => Lane::UnsignedWrap,
        Uqasx | Uqsax => Lane::UnsignedSat,
        Uhasx | Uhsax => Lane::UnsignedHalf,
        Ssub16 | Ssub8 => Lane::SignedWrap,
        Qsub16 | Qsub8 => Lane::SignedSat,
        Shsub16 | Shsub8 => Lane::SignedHalf,
        Usub16 | Usub8 => Lane::UnsignedWrap,
        Uqsub16 | Uqsub8 => Lane::UnsignedSat,
        Uhsub16 | Uhsub8 => Lane::UnsignedHalf,
        _ => return malformed(c, insn),
    };
    let bits = if matches!(
        insn.opcode,
        Sadd8
            | Qadd8
            | Shadd8
            | Uadd8
            | Uqadd8
            | Uhadd8
            | Ssub8
            | Qsub8
            | Shsub8
            | Usub8
            | Uqsub8
            | Uhsub8
    ) {
        8
    } else {
        16
    };
    let lanes = 32 / bits;

    // The exchange forms take the second operand's other halfword: `lo = a.lo ∓ b.hi`
    // and `hi = a.hi ± b.lo`.
    let exchange = matches!(
        insn.opcode,
        Sasx | Qasx | Shasx | Uasx | Uqasx | Uhasx | Ssax | Qsax | Shsax | Usax | Uqsax | Uhsax
    );
    let asx = matches!(insn.opcode, Sasx | Qasx | Shasx | Uasx | Uqasx | Uhasx);
    let subtract = matches!(
        insn.opcode,
        Ssub16
            | Qsub16
            | Shsub16
            | Usub16
            | Uqsub16
            | Uhsub16
            | Ssub8
            | Qsub8
            | Shsub8
            | Usub8
            | Uqsub8
            | Uhsub8
    );
    let mut result = 0u32;
    for index in 0..lanes {
        let x = lane.read(a, index, bits);
        let y = lane.read(b, if exchange { 1 - index } else { index }, bits);
        let negate = if exchange {
            if index == 0 { asx } else { !asx }
        } else {
            subtract
        };
        let value = if negate { x - y } else { x + y };
        result |= lane.write(value, bits) << (index * bits);
    }

    // Only the four byte-wide forms update GE, and they do so from the signed or
    // unsigned comparison of each byte, not from the saturated result.
    match insn.opcode {
        Sadd8 | Ssub8 => {
            let mut flags = 0u32;
            for index in 0..4 {
                let x = lane.read(a, index, 8);
                let y = lane.read(b, index, 8);
                let value = if insn.opcode == Sadd8 { x + y } else { x - y };
                if value >= 0 {
                    flags |= 1 << index;
                }
            }
            set_ge(c, flags);
        }
        Uadd8 | Usub8 => {
            let mut flags = 0u32;
            for index in 0..4 {
                let shift = index * 8;
                let x = i64::from((a >> shift) & 0xff);
                let y = i64::from((b >> shift) & 0xff);
                let carry = if insn.opcode == Uadd8 {
                    x + y >= 0x100
                } else {
                    x - y >= 0
                };
                if carry {
                    flags |= 1 << index;
                }
            }
            set_ge(c, flags);
        }
        _ => {}
    }
    set(c, rd, result);
    next(c, insn);
}

/// `qadd`, `qsub`, `qdadd` and `qdsub`: `Rd = sat(Rm ± sat(2 * Rn))`, with the
/// doubling forms saturating before the operation.
fn saturating_arithmetic(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let [Operand::Reg(rd), Operand::Reg(rm), Operand::Reg(rn)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let (rd, rm, rn) = (*rd, get(c, *rm, pc), get(c, *rn, pc));
    let double = matches!(insn.opcode, Opcode::Qdadd | Opcode::Qdsub);
    let subtract = matches!(insn.opcode, Opcode::Qsub | Opcode::Qdsub);
    let mut saturated = false;
    let operand = if double {
        let (value, sat) = saturate_signed32(i64::from(rn as i32) * 2);
        saturated |= sat;
        i64::from(value as i32)
    } else {
        i64::from(rn as i32)
    };
    let value = if subtract {
        i64::from(rm as i32) - operand
    } else {
        i64::from(rm as i32) + operand
    };
    let (result, sat) = saturate_signed32(value);
    saturated |= sat;
    if saturated {
        set_q(c);
    }
    set(c, rd, result);
    next(c, insn);
}

fn media_extend(c: &mut Cpu, insn: &Insn) {
    use Opcode::*;
    let pc = insn.pc;
    let operands = insn.operands.as_slice();
    let (rd, addend, operand) = match operands {
        [Operand::Reg(rd), operand] => (*rd, None, operand),
        [Operand::Reg(rd), Operand::Reg(rn), operand] => (*rd, Some(*rn), operand),
        _ => return malformed(c, insn),
    };
    let value = match insn.opcode {
        // The reversals and the count-leading-zeros form take a plain register.
        Rev | Rev16 | Revsh | Rbit | Clz => {
            let [Operand::Reg(_), Operand::Reg(rm)] = operands else {
                return malformed(c, insn);
            };
            get(c, *rm, pc)
        }
        _ => {
            let Some(value) = shifted_operand(c, insn, operand) else {
                return malformed(c, insn);
            };
            value
        }
    };
    let extended = match insn.opcode {
        Uxtb => value & 0xff,
        Uxth => value & 0xffff,
        Sxtb => sext(u64::from(value & 0xff), 8) as u32,
        Sxth => sext(u64::from(value & 0xffff), 16) as u32,
        Uxtab => value & 0xff,
        Uxtah => value & 0xffff,
        Sxtab => sext(u64::from(value & 0xff), 8) as u32,
        Sxtah => sext(u64::from(value & 0xffff), 16) as u32,
        Uxtb16 => (value & 0x0000_00ff) | (value & 0x00ff_0000),
        Sxtb16 => {
            let low = sext(u64::from(value & 0xff), 8) as u32 & 0xffff;
            let high = sext(u64::from((value >> 16) & 0xff), 8) as u32 & 0xffff;
            (high << 16) | low
        }
        Uxtab16 => (value & 0x0000_00ff) | (value & 0x00ff_0000),
        Sxtab16 => {
            let low = sext(u64::from(value & 0xff), 8) as u32 & 0xffff;
            let high = sext(u64::from((value >> 16) & 0xff), 8) as u32 & 0xffff;
            (high << 16) | low
        }
        Rev => value.swap_bytes(),
        Rev16 => (value & 0x00ff_00ff) << 8 | (value >> 8) & 0x00ff_00ff,
        Revsh => sext(u64::from((value as u16).swap_bytes()), 16) as u32,
        Rbit => value.reverse_bits(),
        Clz => value.leading_zeros(),
        _ => return malformed(c, insn),
    };
    let result = match addend {
        Some(rn) => get(c, rn, pc).wrapping_add(extended),
        None => extended,
    };
    set(c, rd, result);
    next(c, insn);
}

fn pack(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let [Operand::Reg(rd), Operand::Reg(rn), operand] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let Some(shifted) = shifted_operand(c, insn, operand) else {
        return malformed(c, insn);
    };
    let value = get(c, *rn, pc);
    let result = match insn.opcode {
        // `pkhbt` keeps the bottom half of Rn and the top half of the shifted Rm;
        // `pkhtb` the other way round.
        Opcode::Pkhbt => (value & 0x0000_ffff) | (shifted & 0xffff_0000),
        Opcode::Pkhtb => (value & 0xffff_0000) | (shifted & 0x0000_ffff),
        _ => return malformed(c, insn),
    };
    set(c, *rd, result);
    next(c, insn);
}

fn select(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let [Operand::Reg(rd), Operand::Reg(rn), Operand::Reg(rm)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let (rd, a, b, flags) = (*rd, get(c, *rn, pc), get(c, *rm, pc), ge(c));
    let mut result = 0u32;
    for index in 0..4 {
        let shift = index * 8;
        let byte = if flags >> index & 1 != 0 {
            (a >> shift) & 0xff
        } else {
            (b >> shift) & 0xff
        };
        result |= byte << shift;
    }
    set(c, rd, result);
    next(c, insn);
}

fn saturate(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let operands = insn.operands.as_slice();
    let (rd, sat_imm, operand) = match operands {
        [Operand::Reg(rd), Operand::Imm(sat), operand] => (*rd, *sat, operand),
        _ => return malformed(c, insn),
    };
    let mut saturated = false;
    let result = match insn.opcode {
        Opcode::Ssat => {
            let Some(value) = shifted_operand(c, insn, operand) else {
                return malformed(c, insn);
            };
            let (result, sat) = saturate_signed(i64::from(value as i32), sat_imm);
            saturated |= sat;
            result
        }
        Opcode::Usat => {
            let Some(value) = shifted_operand(c, insn, operand) else {
                return malformed(c, insn);
            };
            let (result, sat) = saturate_unsigned(i64::from(value as i32), sat_imm);
            saturated |= sat;
            result
        }
        Opcode::Ssat16 | Opcode::Usat16 => {
            let Some(Operand::Reg(rm)) = operands.get(2) else {
                return malformed(c, insn);
            };
            let value = get(c, *rm, pc);
            let mut result = 0u32;
            for index in 0..2 {
                let lane = i64::from((value >> (index * 16)) as u16 as i16);
                let (lane_value, sat) = if insn.opcode == Opcode::Ssat16 {
                    saturate_signed(lane, sat_imm)
                } else {
                    saturate_unsigned(lane, sat_imm)
                };
                saturated |= sat;
                result |= (lane_value & 0xffff) << (index * 16);
            }
            result
        }
        _ => return malformed(c, insn),
    };
    if saturated {
        set_q(c);
    }
    set(c, rd, result);
    next(c, insn);
}

fn bitfield(c: &mut Cpu, insn: &Insn) {
    let pc = insn.pc;
    let operands = insn.operands.as_slice();
    let (rd, source, lsb, width) = match operands {
        [Operand::Reg(rd), Operand::Imm(lsb), Operand::Imm(width)] => (*rd, None, *lsb, *width),
        [
            Operand::Reg(rd),
            Operand::Reg(rn),
            Operand::Imm(lsb),
            Operand::Imm(width),
        ] => (*rd, Some(*rn), *lsb, *width),
        _ => return malformed(c, insn),
    };
    if width == 0 || lsb + width > 32 {
        return malformed(c, insn);
    }
    let field = if width == 32 {
        u32::MAX
    } else {
        ((1u64 << width) - 1) as u32
    };
    let mask = field << lsb;
    let current = get(c, rd, pc);
    let result = match insn.opcode {
        Opcode::Bfc => current & !mask,
        Opcode::Bfi => {
            let Some(rn) = source else {
                return malformed(c, insn);
            };
            (current & !mask) | ((get(c, rn, pc) << lsb) & mask)
        }
        Opcode::Ubfx => {
            let Some(rn) = source else {
                return malformed(c, insn);
            };
            (get(c, rn, pc) >> lsb) & field
        }
        Opcode::Sbfx => {
            let Some(rn) = source else {
                return malformed(c, insn);
            };
            let narrowed = (get(c, rn, pc) >> lsb) & field;
            sext(u64::from(narrowed), width) as u32
        }
        _ => return malformed(c, insn),
    };
    set(c, rd, result);
    next(c, insn);
}

fn branch(c: &mut Cpu, insn: &Insn) {
    match insn.operands.as_slice() {
        [Operand::Target(target)] => {
            if target.state != a32::ExecState::Arm {
                horizon::fatal(
                    c,
                    format!(
                        "AArch32 interworking branch to Thumb is not implemented: `{insn}` at {:#x} goes to {:#x}",
                        u64::from(insn.pc),
                        target.addr
                    ),
                );
                return;
            }
            if insn.opcode == Opcode::Bl || insn.opcode == Opcode::Blx {
                set(c, Reg::LR, insn.next());
            }
            c.pc = u64::from(target.addr);
        }
        [Operand::Reg(rm)] => {
            let target = get(c, *rm, insn.pc);
            if insn.opcode != Opcode::Bx {
                set(c, Reg::LR, insn.next());
            }
            branch_to(c, insn, target);
        }
        _ => malformed(c, insn),
    }
}

/// Evaluates the address of a single-register transfer, returning the address used
/// for the access and the value the base register takes when it writes back.
fn transfer_address(c: &Cpu, insn: &Insn, mem: &a32::Mem) -> (u32, u32) {
    let base = if mem.base == Reg::PC {
        mem_base(insn.pc)
    } else {
        get(c, mem.base, insn.pc)
    };
    let offset = match mem.offset {
        Offset::Imm(imm) => u32::from(imm),
        Offset::Reg { reg, shift } => shift_value(c, insn.pc, reg, shift).0,
    };
    let address = if mem.add {
        base.wrapping_add(offset)
    } else {
        base.wrapping_sub(offset)
    };
    (if mem.pre_indexed { address } else { base }, address)
}

fn transfer(c: &mut Cpu, insn: &Insn) {
    let [Operand::Reg(rt), Operand::Mem(mem)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let (rt, mem) = (*rt, *mem);
    let (access, written) = transfer_address(c, insn, &mem);
    let load = matches!(
        insn.opcode,
        Opcode::Ldr
            | Opcode::Ldrh
            | Opcode::Ldrsb
            | Opcode::Ldrsh
            | Opcode::Lda
            | Opcode::Ldab
            | Opcode::Ldah
    );
    if load {
        let value = match insn.opcode {
            Opcode::Ldr => match mem.size {
                MemSize::Byte => c.ld8(u64::from(access)) as u32,
                MemSize::Halfword => c.ld16(u64::from(access)) as u32,
                MemSize::Word => c.ld32(u64::from(access)) as u32,
                MemSize::Double => return malformed(c, insn),
            },
            Opcode::Ldrh | Opcode::Ldah => c.ld16(u64::from(access)) as u32,
            Opcode::Ldab => c.ld8(u64::from(access)) as u32,
            Opcode::Lda => c.ld32(u64::from(access)) as u32,
            Opcode::Ldrsb => sext(c.ld8(u64::from(access)), 8) as u32,
            Opcode::Ldrsh => sext(c.ld16(u64::from(access)), 16) as u32,
            _ => return malformed(c, insn),
        };
        if mem.writes_back() {
            set(c, mem.base, written);
        }
        if rt == Reg::PC {
            branch_to(c, insn, value);
        } else {
            set(c, rt, value);
            next(c, insn);
        }
    } else {
        let value = if rt == Reg::PC {
            insn.pc.wrapping_add(8)
        } else {
            get(c, rt, insn.pc)
        };
        let value = u64::from(value);
        match insn.opcode {
            Opcode::Str => match mem.size {
                MemSize::Byte => c.st8(u64::from(access), value),
                MemSize::Halfword => c.st16(u64::from(access), value),
                MemSize::Word => c.st32(u64::from(access), value),
                MemSize::Double => return malformed(c, insn),
            },
            Opcode::Strh | Opcode::Stlh => c.st16(u64::from(access), value),
            Opcode::Stlb => c.st8(u64::from(access), value),
            Opcode::Stl => c.st32(u64::from(access), value),
            _ => return malformed(c, insn),
        }
        if mem.writes_back() {
            set(c, mem.base, written);
        }
        next(c, insn);
    }
}

fn pair_transfer(c: &mut Cpu, insn: &Insn) {
    let [Operand::Regs { lo, hi }, Operand::Mem(mem)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let (lo, hi, mem) = (*lo, *hi, *mem);
    let (access, written) = transfer_address(c, insn, &mem);
    if insn.opcode == Opcode::Ldrd {
        let first = c.ld32(u64::from(access)) as u32;
        let second = c.ld32(u64::from(access.wrapping_add(4))) as u32;
        if mem.writes_back() {
            set(c, mem.base, written);
        }
        set(c, lo, first);
        set(c, hi, second);
    } else {
        let first = get(c, lo, insn.pc);
        let second = get(c, hi, insn.pc);
        c.st32(u64::from(access), u64::from(first));
        c.st32(u64::from(access.wrapping_add(4)), u64::from(second));
        if mem.writes_back() {
            set(c, mem.base, written);
        }
    }
    next(c, insn);
}

fn exclusive(c: &mut Cpu, insn: &Insn) {
    let operands = insn.operands.as_slice();
    match operands {
        [Operand::Reg(rt), Operand::Mem(mem)] => {
            let (rt, mem) = (*rt, *mem);
            let address = if mem.base == Reg::PC {
                mem_base(insn.pc)
            } else {
                get(c, mem.base, insn.pc)
            };
            let value = match mem.size {
                MemSize::Byte => c.ld8(u64::from(address)),
                MemSize::Halfword => c.ld16(u64::from(address)),
                _ => c.ld32(u64::from(address)),
            };
            // The monitor remembers the loaded value, so a later `strex` to the same
            // address only succeeds while it is unchanged.
            c.monitor(u64::from(address), value, 0);
            set(c, rt, value as u32);
            next(c, insn);
        }
        [Operand::Reg(status), Operand::Reg(rt), Operand::Mem(mem)] => {
            let (status, rt, mem) = (*status, *rt, *mem);
            let address = if mem.base == Reg::PC {
                mem_base(insn.pc)
            } else {
                get(c, mem.base, insn.pc)
            };
            let size = match mem.size {
                MemSize::Byte => 1,
                MemSize::Halfword => 2,
                _ => 4,
            };
            let value = u64::from(get(c, rt, insn.pc));
            let failed = c.store_exclusive(size, u64::from(address), value, 0);
            set(c, status, failed as u32);
            next(c, insn);
        }
        _ => malformed(c, insn),
    }
}

/// `Align(PC, 4)` for a PC-relative base.
#[inline]
fn mem_base(pc: u32) -> u32 {
    pc.wrapping_add(8) & !3
}

fn block_transfer(c: &mut Cpu, insn: &Insn) {
    let [Operand::Block(block)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let block = *block;
    let base = get(c, block.base, insn.pc);
    let count = block.list.bits().count_ones().wrapping_mul(4);
    let start = match block.mode {
        BlockMode::IncrementAfter => base,
        BlockMode::IncrementBefore => base.wrapping_add(4),
        BlockMode::DecrementAfter => base.wrapping_sub(count).wrapping_add(4),
        BlockMode::DecrementBefore => base.wrapping_sub(count),
    };
    let mut loaded_pc = None;
    let load = insn.opcode == Opcode::Ldm;
    for (index, reg) in block.list.iter().enumerate() {
        let address = u64::from(start.wrapping_add((index as u32).wrapping_mul(4)));
        if load {
            let value = c.ld32(address) as u32;
            if reg == Reg::PC {
                loaded_pc = Some(value);
            } else {
                set(c, reg, value);
            }
        } else {
            let value = get(c, reg, insn.pc);
            c.st32(address, u64::from(value));
        }
    }
    if block.writeback {
        let written = match block.mode {
            BlockMode::IncrementAfter | BlockMode::IncrementBefore => base.wrapping_add(count),
            BlockMode::DecrementAfter | BlockMode::DecrementBefore => base.wrapping_sub(count),
        };
        set(c, block.base, written);
    }
    match loaded_pc {
        Some(value) => branch_to(c, insn, value),
        None => next(c, insn),
    }
}

/// `mrs Rd, cpsr`: reports the modelled CPSR, whose mode field is user mode.
fn mrs(c: &mut Cpu, insn: &Insn) {
    let [Operand::Reg(rd)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let value = cpsr(c) | CPSR_USER;
    set(c, *rd, value);
    next(c, insn);
}

/// `msr cpsr_f, <value>`: writes NZCVQ from the flags byte. GE (bits 19:16) is not
/// part of that byte and stays as it was.
fn msr(c: &mut Cpu, insn: &Insn) {
    let value = match insn.operands.as_slice() {
        [Operand::Reg(rm)] => get(c, *rm, insn.pc),
        [Operand::Imm(imm)] => *imm,
        _ => return malformed(c, insn),
    };
    let kept = c.nzcv as u32 & GE_MASK;
    set_cpsr(c, (value & FLAGS) | kept);
    next(c, insn);
}

fn exec(c: &mut Cpu, insn: &Insn) {
    if !insn.cond.holds(c.nzcv as u32) {
        next(c, insn);
        return;
    }
    match insn.opcode {
        Opcode::And
        | Opcode::Eor
        | Opcode::Sub
        | Opcode::Rsb
        | Opcode::Add
        | Opcode::Adc
        | Opcode::Sbc
        | Opcode::Rsc
        | Opcode::Tst
        | Opcode::Teq
        | Opcode::Cmp
        | Opcode::Cmn
        | Opcode::Orr
        | Opcode::Mov
        | Opcode::Bic
        | Opcode::Mvn => data_processing(c, insn),
        Opcode::Movw | Opcode::Movt => wide_move(c, insn),
        Opcode::B | Opcode::Bl | Opcode::Bx | Opcode::Blx => branch(c, insn),
        Opcode::Ldr
        | Opcode::Str
        | Opcode::Ldrh
        | Opcode::Strh
        | Opcode::Ldrsb
        | Opcode::Ldrsh
        | Opcode::Lda
        | Opcode::Ldab
        | Opcode::Ldah
        | Opcode::Stl
        | Opcode::Stlb
        | Opcode::Stlh => transfer(c, insn),
        Opcode::Ldrd | Opcode::Strd => pair_transfer(c, insn),
        Opcode::Ldrex
        | Opcode::Ldrexb
        | Opcode::Ldrexh
        | Opcode::Strex
        | Opcode::Strexb
        | Opcode::Strexh => exclusive(c, insn),
        Opcode::Ldm | Opcode::Stm => block_transfer(c, insn),
        Opcode::Mul | Opcode::Mla => multiply(c, insn),
        Opcode::Umull | Opcode::Umlal | Opcode::Smull | Opcode::Smlal => long_multiply(c, insn),
        Opcode::Smlabb
        | Opcode::Smlabt
        | Opcode::Smlatb
        | Opcode::Smlatt
        | Opcode::Smlawb
        | Opcode::Smlawt
        | Opcode::Smulbb
        | Opcode::Smulbt
        | Opcode::Smultb
        | Opcode::Smultt
        | Opcode::Smulwb
        | Opcode::Smulwt
        | Opcode::Smlalbb
        | Opcode::Smlalbt
        | Opcode::Smlaltb
        | Opcode::Smlaltt => halfword_multiply(c, insn),
        Opcode::Udiv | Opcode::Sdiv => divide(c, insn),
        Opcode::Smuad
        | Opcode::Smuadx
        | Opcode::Smusd
        | Opcode::Smusdx
        | Opcode::Smlad
        | Opcode::Smladx
        | Opcode::Smlsd
        | Opcode::Smlsdx
        | Opcode::Smlald
        | Opcode::Smlaldx
        | Opcode::Smlsld
        | Opcode::Smlsldx
        | Opcode::Smmul
        | Opcode::Smmulr
        | Opcode::Smmla
        | Opcode::Smmlar
        | Opcode::Smmls
        | Opcode::Smmlsr => media_multiply(c, insn),
        Opcode::Uxtb
        | Opcode::Uxth
        | Opcode::Sxtb
        | Opcode::Sxth
        | Opcode::Uxtab
        | Opcode::Uxtah
        | Opcode::Sxtab
        | Opcode::Sxtah
        | Opcode::Uxtb16
        | Opcode::Sxtb16
        | Opcode::Uxtab16
        | Opcode::Sxtab16
        | Opcode::Rev
        | Opcode::Rev16
        | Opcode::Revsh
        | Opcode::Rbit
        | Opcode::Clz => media_extend(c, insn),
        Opcode::Pkhbt | Opcode::Pkhtb => pack(c, insn),
        Opcode::Sel => select(c, insn),
        Opcode::Ssat | Opcode::Usat | Opcode::Ssat16 | Opcode::Usat16 => saturate(c, insn),
        Opcode::Bfc | Opcode::Bfi | Opcode::Sbfx | Opcode::Ubfx => bitfield(c, insn),
        Opcode::Qadd | Opcode::Qsub | Opcode::Qdadd | Opcode::Qdsub => {
            saturating_arithmetic(c, insn)
        }
        Opcode::Sadd16
        | Opcode::Sasx
        | Opcode::Ssax
        | Opcode::Ssub16
        | Opcode::Sadd8
        | Opcode::Ssub8
        | Opcode::Qadd16
        | Opcode::Qasx
        | Opcode::Qsax
        | Opcode::Qsub16
        | Opcode::Qadd8
        | Opcode::Qsub8
        | Opcode::Shadd16
        | Opcode::Shasx
        | Opcode::Shsax
        | Opcode::Shsub16
        | Opcode::Shadd8
        | Opcode::Shsub8
        | Opcode::Uadd16
        | Opcode::Uasx
        | Opcode::Usax
        | Opcode::Usub16
        | Opcode::Uadd8
        | Opcode::Usub8
        | Opcode::Uqadd16
        | Opcode::Uqasx
        | Opcode::Uqsax
        | Opcode::Uqsub16
        | Opcode::Uqadd8
        | Opcode::Uqsub8
        | Opcode::Uhadd16
        | Opcode::Uhasx
        | Opcode::Uhsax
        | Opcode::Uhsub16
        | Opcode::Uhadd8
        | Opcode::Uhsub8 => parallel(c, insn),
        Opcode::Mrs => mrs(c, insn),
        Opcode::Msr => msr(c, insn),
        Opcode::Nop
        | Opcode::Yield
        | Opcode::Wfe
        | Opcode::Wfi
        | Opcode::Sev
        | Opcode::Sevl
        | Opcode::Dsb
        | Opcode::Dmb
        | Opcode::Isb
        | Opcode::Pld => next(c, insn),
        Opcode::Clrex => {
            // The exclusive monitor is one address in the CPU state; pointing it at
            // the "no monitor" sentinel is what the kernel's clear does.
            c.monitor(crate::cpu::NO_MONITOR, 0, 0);
            next(c, insn);
        }
        Opcode::Vldr | Opcode::Vstr => vfp_transfer(c, insn),
        Opcode::Vldm | Opcode::Vstm | Opcode::Vpush | Opcode::Vpop => vfp_block(c, insn),
        Opcode::Vmov => vfp_move(c, insn),
        Opcode::VmovLane | Opcode::VmovLaneS | Opcode::VmovLaneU => vfp_element_move(c, insn),
        Opcode::Vdup => vfp_duplicate(c, insn),
        Opcode::Vmrs | Opcode::Vmsr => vfp_system(c, insn),
        Opcode::Mrc | Opcode::Mcr => system_register(c, insn),
        Opcode::Mrrc => mrrc(c, insn),
        Opcode::Vmla
        | Opcode::Vmls
        | Opcode::Vnmla
        | Opcode::Vnmls
        | Opcode::Vnmul
        | Opcode::Vdiv
        | Opcode::Vfnma
        | Opcode::Vfnms => vfp_arithmetic(c, insn),
        // `vadd`, `vsub`, `vmul`, `vabs`, `vneg`, `vfma` and `vfms` name both the
        // scalar VFP and the Advanced SIMD instruction; the element shape tells them
        // apart.
        Opcode::Vadd
        | Opcode::Vsub
        | Opcode::Vmul
        | Opcode::Vabs
        | Opcode::Vneg
        | Opcode::Vfma
        | Opcode::Vfms => {
            if insn.shape.is_some() {
                if matches!(
                    insn.opcode,
                    Opcode::Vadd | Opcode::Vsub | Opcode::Vmul | Opcode::Vfma | Opcode::Vfms
                ) {
                    neon_three(c, insn)
                } else {
                    neon_two(c, insn)
                }
            } else if matches!(insn.opcode, Opcode::Vabs | Opcode::Vneg) {
                vfp_unary(c, insn)
            } else {
                vfp_arithmetic(c, insn)
            }
        }
        Opcode::Vsqrt | Opcode::Vrintr | Opcode::Vrintz | Opcode::Vrintx => vfp_unary(c, insn),
        Opcode::Vcmp | Opcode::Vcmpe => vfp_compare(c, insn),
        Opcode::Vcvt | Opcode::VcvtR => vfp_convert(c, insn),
        Opcode::Vld1
        | Opcode::Vst1
        | Opcode::Vld2
        | Opcode::Vst2
        | Opcode::Vld3
        | Opcode::Vst3
        | Opcode::Vld4
        | Opcode::Vst4 => neon_transfer(c, insn),
        Opcode::Vext => neon_extract(c, insn),
        Opcode::Vand
        | Opcode::Vorr
        | Opcode::Veor
        | Opcode::Vbic
        | Opcode::Vorn
        | Opcode::Vbsl
        | Opcode::Vmax
        | Opcode::VmaxU
        | Opcode::Vmin
        | Opcode::VminU
        | Opcode::Vcgt
        | Opcode::VcgtU
        | Opcode::Vcge
        | Opcode::VcgeU
        | Opcode::Vceq
        | Opcode::Vtst
        | Opcode::VaddF
        | Opcode::VsubF
        | Opcode::VmulF
        | Opcode::VmlaF
        | Opcode::VmlsF
        | Opcode::VmlaI
        | Opcode::VmlsI
        | Opcode::VmaxF
        | Opcode::VminF
        | Opcode::VceqF
        | Opcode::VcgeF => neon_three(c, insn),
        Opcode::VabsU
        | Opcode::VnegU
        | Opcode::Vrev16
        | Opcode::Vrev32
        | Opcode::Vrev64
        | Opcode::Vmvn
        | Opcode::VabsF
        | Opcode::VnegF
        | Opcode::VsqrtF => neon_two(c, insn),
        Opcode::VmovImm | Opcode::VorrImm | Opcode::VmvnImm | Opcode::VbicImm => {
            neon_immediate(c, insn)
        }
        Opcode::Svc => {
            let [Operand::Imm(imm)] = insn.operands.as_slice() else {
                return malformed(c, insn);
            };
            next(c, insn);
            horizon::svc(c, (*imm & 0xffff) as u16);
        }
        Opcode::Bkpt => horizon::fatal(
            c,
            format!(
                "AArch32 breakpoint `{insn}` at {:#x} is not implemented",
                u64::from(insn.pc)
            ),
        ),
    }
}

// --- VFP and Advanced SIMD -------------------------------------------------------
//
// The register file is the one `crate::cpu::Cpu` already has: 32 64-bit slots for the
// `d` registers, which the 32 `s` registers alias in pairs. `fpsr` holds the A32
// FPSCR, whose layout contains the A64 FPSR fields (NZCV in 31:28, QC in 27 and the
// cumulative exceptions in 7:0) and the A64 FPCR ones (DN in 25, FZ in 24, RMode in
// 23:22), so the shared `simd::fp` model applies unchanged; `fpcr` is the A32 FPEXC.
//
// The cumulative exception bits are set where the slice can determine them exactly:
// Invalid Operation (NaN operands of `vcmp`, of a NEON comparison, of `vsqrt` of a
// negative number, and a conversion out of range), Divide by Zero, and Inexact for
// `vrintx`. Overflow, underflow and the inexact bit of the arithmetic are not tracked:
// the shared helpers return the correctly rounded result without its error term, so the
// arithmetic itself is exact and only the diagnostic bits are missing.

/// FPSCR: the four condition flags.
const FPSCR_NZCV: u32 = 0xf000_0000;
/// FPSCR cumulative exception bits.
const FPSCR_IOC: u32 = 1 << 0;
const FPSCR_DZC: u32 = 1 << 1;
const FPSCR_IXC: u32 = 1 << 4;
/// FPEXC lives in `Cpu::fpcr` for an A32 process.
const FPEXC_EN: u32 = 1 << 30;

#[inline]
fn fpscr(c: &Cpu) -> u32 {
    c.fpsr as u32
}

#[inline]
fn set_fpscr(c: &mut Cpu, value: u32) {
    c.fpsr = u64::from(value);
}

#[inline]
fn fpenv(c: &Cpu) -> Fpcr {
    Fpcr::from_bits(c.fpsr)
}

/// Sets sticky exception bits.
#[inline]
fn set_exception(c: &mut Cpu, bits: u32) {
    let value = fpscr(c) | bits;
    set_fpscr(c, value);
}

/// Copies a comparison result packed as `nzcv` into the FPSCR flags.
#[inline]
fn set_fp_flags(c: &mut Cpu, nzcv: u64) {
    let value = (fpscr(c) & !FPSCR_NZCV) | (nzcv as u32 & FPSCR_NZCV);
    set_fpscr(c, value);
}

#[inline]
fn read_d(c: &Cpu, reg: DReg) -> u64 {
    c.v[reg.number() as usize][0]
}

#[inline]
fn write_d(c: &mut Cpu, reg: DReg, value: u64) {
    c.v[reg.number() as usize][0] = value;
}

#[inline]
fn read_s(c: &Cpu, reg: SReg) -> u32 {
    let word = c.v[reg.d().number() as usize][0];
    if reg.high() {
        (word >> 32) as u32
    } else {
        word as u32
    }
}

#[inline]
fn write_s(c: &mut Cpu, reg: SReg, value: u32) {
    let index = reg.d().number() as usize;
    let old = c.v[index][0];
    c.v[index][0] = if reg.high() {
        (old & 0x0000_0000_ffff_ffff) | (u64::from(value) << 32)
    } else {
        (old & 0xffff_ffff_0000_0000) | u64::from(value)
    };
}

/// A vector register: the one or two 64-bit halves an instruction works on.
#[derive(Clone, Copy, Default)]
struct Vector {
    halves: [u64; 2],
}

impl Vector {
    fn read(c: &Cpu, shape: VecShape, reg: DReg) -> Vector {
        let mut halves = [read_d(c, reg), 0];
        if shape.q
            && let Some(high) = DReg::new(reg.number() + 1)
        {
            halves[1] = read_d(c, high);
        }
        Vector { halves }
    }

    fn write(self, c: &mut Cpu, shape: VecShape, reg: DReg) {
        write_d(c, reg, self.halves[0]);
        if shape.q
            && let Some(high) = DReg::new(reg.number() + 1)
        {
            write_d(c, high, self.halves[1]);
        }
    }
}

/// Number of 64-bit halves a register of this shape has.
#[inline]
fn halves(shape: VecShape) -> usize {
    if shape.q { 2 } else { 1 }
}

#[inline]
fn elem_mask(esize: u32) -> u64 {
    if esize >= 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    }
}

#[inline]
fn lane(word: u64, esize: u32, index: u32) -> u64 {
    (word >> (index * esize)) & elem_mask(esize)
}

#[inline]
fn set_lane(word: &mut u64, esize: u32, index: u32, value: u64) {
    let shift = index * esize;
    let mask = elem_mask(esize) << shift;
    *word = (*word & !mask) | ((value & elem_mask(esize)) << shift);
}

#[inline]
fn signed_lane(word: u64, esize: u32, index: u32) -> i64 {
    let value = lane(word, esize, index);
    if esize >= 64 {
        value as i64
    } else {
        ((value << (64 - esize)) as i64) >> (64 - esize)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Prec {
    Single,
    Double,
}

impl Prec {
    fn of(operand: &Operand) -> Option<Prec> {
        match operand {
            Operand::S(_) => Some(Prec::Single),
            Operand::D(_) => Some(Prec::Double),
            _ => None,
        }
    }
}

fn scalar32(c: &Cpu, operand: &Operand) -> Option<u32> {
    match operand {
        Operand::S(reg) => Some(read_s(c, *reg)),
        // The zero-compare forms carry the value the encoding names.
        Operand::Imm(0) => Some(0),
        Operand::ImmFloat32(bits) => Some(*bits),
        _ => None,
    }
}

fn scalar64(c: &Cpu, operand: &Operand) -> Option<u64> {
    match operand {
        Operand::D(reg) => Some(read_d(c, *reg)),
        Operand::Imm(0) => Some(0),
        Operand::ImmFloat64(bits) => Some(*bits),
        _ => None,
    }
}

fn store32(c: &mut Cpu, operand: &Operand, value: u32) -> bool {
    match operand {
        Operand::S(reg) => {
            write_s(c, *reg, value);
            true
        }
        _ => false,
    }
}

fn store64(c: &mut Cpu, operand: &Operand, value: u64) -> bool {
    match operand {
        Operand::D(reg) => {
            write_d(c, *reg, value);
            true
        }
        _ => false,
    }
}

fn vfp_transfer(c: &mut Cpu, insn: &Insn) {
    let [register, Operand::Mem(mem)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let mem = *mem;
    let address = u64::from(transfer_address(c, insn, &mem).0);
    match (insn.opcode, register) {
        (Opcode::Vldr, Operand::S(reg)) => {
            let value = c.ld32(address) as u32;
            write_s(c, *reg, value);
        }
        (Opcode::Vldr, Operand::D(reg)) => {
            let value = c.ld64(address);
            write_d(c, *reg, value);
        }
        (Opcode::Vstr, Operand::S(reg)) => c.st32(address, u64::from(read_s(c, *reg))),
        (Opcode::Vstr, Operand::D(reg)) => c.st64(address, read_d(c, *reg)),
        _ => return malformed(c, insn),
    }
    next(c, insn);
}

/// `vldm`, `vstm`, `vpush` and `vpop`: a contiguous run of 32-bit registers.
fn vfp_block(c: &mut Cpu, insn: &Insn) {
    let [Operand::VBlock(block)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let block = *block;
    let load = matches!(insn.opcode, Opcode::Vldm | Opcode::Vpop);
    let base = get(c, block.base, insn.pc);
    let bytes = block.bytes();
    let start = match block.mode {
        BlockMode::DecrementBefore => base.wrapping_sub(bytes),
        BlockMode::IncrementAfter => base,
        _ => return malformed(c, insn),
    };
    for word in 0..u32::from(block.words) {
        let address = u64::from(start.wrapping_add(word.wrapping_mul(4)));
        let index = block.first.number() + (word / 2) as u8;
        let Some(register) = DReg::new(index) else {
            return malformed(c, insn);
        };
        let high = word % 2 == 1;
        let mut slot = read_d(c, register);
        if load {
            let value = c.ld32(address) as u32;
            slot = if high {
                (slot & 0x0000_0000_ffff_ffff) | (u64::from(value) << 32)
            } else {
                (slot & 0xffff_ffff_0000_0000) | u64::from(value)
            };
            write_d(c, register, slot);
        } else {
            let value = if high { slot >> 32 } else { slot };
            c.st32(address, value & 0xffff_ffff);
        }
    }
    let written = match block.mode {
        BlockMode::DecrementBefore => base.wrapping_sub(bytes),
        _ => base.wrapping_add(bytes),
    };
    if block.writeback {
        set(c, block.base, written);
    }
    next(c, insn);
}

/// `vmov`: register to register, core to single, the core-pair forms and the
/// immediate.
fn vfp_move(c: &mut Cpu, insn: &Insn) {
    match insn.operands.as_slice() {
        [Operand::S(dest), Operand::S(source)] => {
            let value = read_s(c, *source);
            write_s(c, *dest, value);
        }
        [Operand::D(dest), Operand::D(source)] => {
            let value = read_d(c, *source);
            write_d(c, *dest, value);
        }
        [Operand::S(dest), Operand::Reg(source)] => {
            let value = get(c, *source, insn.pc);
            write_s(c, *dest, value);
        }
        [Operand::Reg(dest), Operand::S(source)] => {
            let value = read_s(c, *source);
            set(c, *dest, value);
        }
        [Operand::D(dest), Operand::Reg(low), Operand::Reg(high)] => {
            let value =
                u64::from(get(c, *low, insn.pc)) | (u64::from(get(c, *high, insn.pc)) << 32);
            write_d(c, *dest, value);
        }
        [Operand::Reg(low), Operand::Reg(high), Operand::D(source)] => {
            let value = read_d(c, *source);
            set(c, *low, value as u32);
            set(c, *high, (value >> 32) as u32);
        }
        [Operand::S(dest), Operand::ImmFloat32(bits)] => write_s(c, *dest, *bits),
        [Operand::D(dest), Operand::ImmFloat64(bits)] => write_d(c, *dest, *bits),
        _ => return malformed(c, insn),
    }
    next(c, insn);
}

/// `vmov.<size> Dd[lane], Rt` and the sign- or zero-extending read form.
fn vfp_element_move(c: &mut Cpu, insn: &Insn) {
    let Some(shape) = insn.shape else {
        return malformed(c, insn);
    };
    let esize = u32::from(shape.esize);
    match insn.operands.as_slice() {
        [Operand::Lane { reg, index }, Operand::Reg(source)] => {
            let value = get(c, *source, insn.pc);
            let mut word = read_d(c, *reg);
            set_lane(&mut word, esize, u32::from(*index), u64::from(value));
            write_d(c, *reg, word);
        }
        [Operand::Reg(dest), Operand::Lane { reg, index }] => {
            let word = read_d(c, *reg);
            let value = lane(word, esize, u32::from(*index));
            let value = match insn.opcode {
                Opcode::VmovLaneS => sext(value, esize) as u32,
                _ => value as u32,
            };
            set(c, *dest, value);
        }
        _ => return malformed(c, insn),
    }
    next(c, insn);
}

/// `vdup.<size> Dd, Rt`: the low element of the core register, repeated.
fn vfp_duplicate(c: &mut Cpu, insn: &Insn) {
    let (Some(shape), [Operand::D(dest), Operand::Reg(source)]) =
        (insn.shape, insn.operands.as_slice())
    else {
        return malformed(c, insn);
    };
    let esize = u32::from(shape.esize);
    let value = u64::from(get(c, *source, insn.pc)) & elem_mask(esize);
    let mut word = 0u64;
    let mut shift = 0;
    while shift < 64 {
        word |= value << shift;
        shift += esize;
    }
    let mut out = Vector::default();
    for half in 0..halves(shape) {
        out.halves[half] = word;
    }
    out.write(c, shape, *dest);
    next(c, insn);
}

/// `vmrs` and `vmsr`.
fn vfp_system(c: &mut Cpu, insn: &Insn) {
    match insn.operands.as_slice() {
        [Operand::VfpDest(dest), Operand::VfpSys(sys)] => {
            let value = match sys {
                VfpSys::Fpscr => fpscr(c),
                VfpSys::Fpexc => c.fpcr as u32,
                // The identification registers report an implementation-defined VFP;
                // this runtime does not claim one.
                other => {
                    return vfp_system_fault(c, insn, *other);
                }
            };
            match dest {
                VfpDest::Core(reg) => set(c, *reg, value),
                VfpDest::Apsr => {
                    let kept = c.nzcv as u32 & !FPSCR_NZCV;
                    c.nzcv = u64::from(kept | (value & FPSCR_NZCV));
                }
            }
        }
        [Operand::VfpSys(sys), Operand::Reg(source)] => {
            let value = get(c, *source, insn.pc);
            match sys {
                VfpSys::Fpscr => set_fpscr(c, value),
                VfpSys::Fpexc => c.fpcr = u64::from(value & FPEXC_EN),
                other => return vfp_system_fault(c, insn, *other),
            }
        }
        _ => return malformed(c, insn),
    }
    next(c, insn);
}

fn vfp_system_fault(c: &mut Cpu, insn: &Insn, sys: VfpSys) {
    horizon::fatal(
        c,
        format!(
            "AArch32 {sys} is not modelled: `{insn}` at {:#x}",
            u64::from(insn.pc)
        ),
    );
}

/// `mrrc` of the 64-bit counters, which share the process clock the A64 frontend reads
/// for `cntpct_el0`/`cntvct_el0`.
fn mrrc(c: &mut Cpu, insn: &Insn) {
    let [Operand::Regs { lo, hi }, Operand::SysReg(sys)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let value = match sys {
        a32::A32SysReg::Cntpct | a32::A32SysReg::Cntvct => c.process().ticks(),
        _ => return malformed(c, insn),
    };
    set(c, *lo, value as u32);
    set(c, *hi, (value >> 32) as u32);
    next(c, insn);
}

/// `mrc`/`mcr` of the `cp15` thread pointers.
fn system_register(c: &mut Cpu, insn: &Insn) {
    let [Operand::Reg(rt), Operand::SysReg(sys)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    match sys {
        a32::A32SysReg::Tpidrurw => {
            if insn.opcode == Opcode::Mrc {
                let value = c.tpidr_el0 as u32;
                set(c, *rt, value);
            } else {
                c.tpidr_el0 = u64::from(get(c, *rt, insn.pc));
            }
        }
        a32::A32SysReg::Tpidruro => {
            if insn.opcode != Opcode::Mrc {
                // The user read-only register ignores writes.
                return next(c, insn);
            }
            let value = c.tpidrro_el0 as u32;
            set(c, *rt, value);
        }
        // The counters are 64-bit and only have the `mrrc` form.
        a32::A32SysReg::Cntpct | a32::A32SysReg::Cntvct => return malformed(c, insn),
    }
    next(c, insn);
}

/// Scalar arithmetic. On VFPv4, which is the implementation this runtime models,
/// `vmla`/`vmls` are fused, exactly like `vfma`/`vfms`.
fn vfp_arithmetic(c: &mut Cpu, insn: &Insn) {
    let [dest, first, second] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let Some(prec) = Prec::of(dest) else {
        return malformed(c, insn);
    };
    let env = fpenv(c);
    match prec {
        Prec::Single => {
            let (Some(d), Some(n), Some(m)) =
                (scalar32(c, dest), scalar32(c, first), scalar32(c, second))
            else {
                return malformed(c, insn);
            };
            let Some(result) = arith32(insn.opcode, d, n, m, env) else {
                return malformed(c, insn);
            };
            if !store32(c, dest, result) {
                return malformed(c, insn);
            }
            if insn.opcode == Opcode::Vdiv && fp::f32b::is_zero(m) && !fp::f32b::is_nan(n) {
                set_exception(c, FPSCR_DZC);
            }
        }
        Prec::Double => {
            let (Some(d), Some(n), Some(m)) =
                (scalar64(c, dest), scalar64(c, first), scalar64(c, second))
            else {
                return malformed(c, insn);
            };
            let Some(result) = arith64(insn.opcode, d, n, m, env) else {
                return malformed(c, insn);
            };
            if !store64(c, dest, result) {
                return malformed(c, insn);
            }
            if insn.opcode == Opcode::Vdiv && fp::f64b::is_zero(m) && !fp::f64b::is_nan(n) {
                set_exception(c, FPSCR_DZC);
            }
        }
    }
    next(c, insn);
}

fn arith32(opcode: Opcode, d: u32, n: u32, m: u32, env: Fpcr) -> Option<u32> {
    Some(match opcode {
        // VMLA/VMLS are fused on VFPv4. VNMLS computes `-d + n*m` and VNMLA
        // `-d - n*m`, the same values as VFNMA and VFNMS.
        Opcode::Vmla | Opcode::Vfma => fp::f32b::mul_add(d, n, m, env),
        Opcode::Vmls | Opcode::Vfms => fp::f32b::mul_add(d, fp::f32b::neg(n), m, env),
        Opcode::Vnmls | Opcode::Vfnma => fp::f32b::mul_add(fp::f32b::neg(d), n, m, env),
        Opcode::Vnmla | Opcode::Vfnms => {
            fp::f32b::mul_add(fp::f32b::neg(d), fp::f32b::neg(n), m, env)
        }
        Opcode::Vmul => fp::f32b::mul(n, m, env),
        Opcode::Vnmul => fp::f32b::neg(fp::f32b::mul(n, m, env)),
        Opcode::Vadd => fp::f32b::add(n, m, env),
        Opcode::Vsub => fp::f32b::sub(n, m, env),
        Opcode::Vdiv => fp::f32b::div(n, m, env),
        _ => return None,
    })
}

fn arith64(opcode: Opcode, d: u64, n: u64, m: u64, env: Fpcr) -> Option<u64> {
    Some(match opcode {
        Opcode::Vmla | Opcode::Vfma => fp::f64b::mul_add(d, n, m, env),
        Opcode::Vmls | Opcode::Vfms => fp::f64b::mul_add(d, fp::f64b::neg(n), m, env),
        Opcode::Vnmls | Opcode::Vfnma => fp::f64b::mul_add(fp::f64b::neg(d), n, m, env),
        Opcode::Vnmla | Opcode::Vfnms => {
            fp::f64b::mul_add(fp::f64b::neg(d), fp::f64b::neg(n), m, env)
        }
        Opcode::Vmul => fp::f64b::mul(n, m, env),
        Opcode::Vnmul => fp::f64b::neg(fp::f64b::mul(n, m, env)),
        Opcode::Vadd => fp::f64b::add(n, m, env),
        Opcode::Vsub => fp::f64b::sub(n, m, env),
        Opcode::Vdiv => fp::f64b::div(n, m, env),
        _ => return None,
    })
}

/// `vabs`, `vneg`, `vsqrt` and the `vrint` family.
fn vfp_unary(c: &mut Cpu, insn: &Insn) {
    let [dest, source] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let Some(prec) = Prec::of(dest) else {
        return malformed(c, insn);
    };
    let env = fpenv(c);
    let mode = Round::from_rmode(env.rmode);
    match prec {
        Prec::Single => {
            let Some(value) = scalar32(c, source) else {
                return malformed(c, insn);
            };
            let Some(result) = unary32(insn.opcode, value, mode, env) else {
                return malformed(c, insn);
            };
            if insn.opcode == Opcode::Vsqrt && value & 0x8000_0000 != 0 && !fp::f32b::is_zero(value)
            {
                set_exception(c, FPSCR_IOC);
            }
            if insn.opcode == Opcode::Vrintx && result != value {
                set_exception(c, FPSCR_IXC);
            }
            if !store32(c, dest, result) {
                return malformed(c, insn);
            }
        }
        Prec::Double => {
            let Some(value) = scalar64(c, source) else {
                return malformed(c, insn);
            };
            let Some(result) = unary64(insn.opcode, value, mode, env) else {
                return malformed(c, insn);
            };
            if insn.opcode == Opcode::Vsqrt && value >> 63 != 0 && !fp::f64b::is_zero(value) {
                set_exception(c, FPSCR_IOC);
            }
            if insn.opcode == Opcode::Vrintx && result != value {
                set_exception(c, FPSCR_IXC);
            }
            if !store64(c, dest, result) {
                return malformed(c, insn);
            }
        }
    }
    next(c, insn);
}

fn unary32(opcode: Opcode, value: u32, mode: Round, env: Fpcr) -> Option<u32> {
    Some(match opcode {
        Opcode::Vabs => fp::f32b::abs(value),
        Opcode::Vneg => fp::f32b::neg(value),
        Opcode::Vsqrt => fp::f32b::sqrt(value, env),
        Opcode::Vrintr | Opcode::Vrintx => fp::f32b::round_int(value, mode, env),
        Opcode::Vrintz => fp::f32b::round_int(value, Round::Zero, env),
        _ => return None,
    })
}

fn unary64(opcode: Opcode, value: u64, mode: Round, env: Fpcr) -> Option<u64> {
    Some(match opcode {
        Opcode::Vabs => fp::f64b::abs(value),
        Opcode::Vneg => fp::f64b::neg(value),
        Opcode::Vsqrt => fp::f64b::sqrt(value, env),
        Opcode::Vrintr | Opcode::Vrintx => fp::f64b::round_int(value, mode, env),
        Opcode::Vrintz => fp::f64b::round_int(value, Round::Zero, env),
        _ => return None,
    })
}

/// `vcmp` and `vcmpe`: write FPSCR's flags, and raise Invalid Operation on a NaN.
fn vfp_compare(c: &mut Cpu, insn: &Insn) {
    let [first, second] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let Some(prec) = Prec::of(first) else {
        return malformed(c, insn);
    };
    let env = fpenv(c);
    let (flags, invalid) = match prec {
        Prec::Single => {
            let (Some(a), Some(b)) = (scalar32(c, first), scalar32(c, second)) else {
                return malformed(c, insn);
            };
            (
                fp::f32b::compare(a, b, env),
                fp::f32b::is_nan(a) || fp::f32b::is_nan(b),
            )
        }
        Prec::Double => {
            let (Some(a), Some(b)) = (scalar64(c, first), scalar64(c, second)) else {
                return malformed(c, insn);
            };
            (
                fp::f64b::compare(a, b, env),
                fp::f64b::is_nan(a) || fp::f64b::is_nan(b),
            )
        }
    };
    set_fp_flags(c, flags);
    if invalid {
        set_exception(c, FPSCR_IOC);
    }
    next(c, insn);
}

/// `vcvt`: between the two precisions, and between floating point and an integer.
fn vfp_convert(c: &mut Cpu, insn: &Insn) {
    let [dest, source, Operand::Convert(convert)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let env = fpenv(c);
    let mode = if insn.opcode == Opcode::VcvtR {
        Round::Nearest
    } else {
        Round::from_rmode(env.rmode)
    };
    match *convert {
        Convert::F64FromF32 => {
            let Operand::S(source) = source else {
                return malformed(c, insn);
            };
            let value = fp::f32_to_f64(read_s(c, *source), env);
            if !store64(c, dest, value) {
                return malformed(c, insn);
            }
        }
        Convert::F32FromF64 => {
            let Operand::D(source) = source else {
                return malformed(c, insn);
            };
            let value = fp::f64_to_f32(read_d(c, *source), env);
            if !store32(c, dest, value) {
                return malformed(c, insn);
            }
        }
        Convert::F32FromU32 | Convert::F32FromS32 => {
            let Some(value) = scalar32(c, source) else {
                return malformed(c, insn);
            };
            let signed = *convert == Convert::F32FromS32;
            let result = fp::f32b::from_int(u64::from(value), signed, 32, 0, env);
            if !store32(c, dest, result) {
                return malformed(c, insn);
            }
        }
        Convert::F64FromU32 | Convert::F64FromS32 => {
            let Some(value) = scalar32(c, source) else {
                return malformed(c, insn);
            };
            let signed = *convert == Convert::F64FromS32;
            let result = fp::f64b::from_int(u64::from(value), signed, 32, 0, env);
            if !store64(c, dest, result) {
                return malformed(c, insn);
            }
        }
        Convert::U32FromF32 | Convert::S32FromF32 => {
            let Some(value) = scalar32(c, source) else {
                return malformed(c, insn);
            };
            let signed = *convert == Convert::S32FromF32;
            if int_out_of_range32(value, mode, signed, env) {
                set_exception(c, FPSCR_IOC);
            }
            let result = fp::f32b::to_int(value, mode, signed, 32, 0, env);
            if !store32(c, dest, result as u32) {
                return malformed(c, insn);
            }
        }
        Convert::U32FromF64 | Convert::S32FromF64 => {
            let Some(value) = scalar64(c, source) else {
                return malformed(c, insn);
            };
            let signed = *convert == Convert::S32FromF64;
            if int_out_of_range64(value, mode, signed, env) {
                set_exception(c, FPSCR_IOC);
            }
            let result = fp::f64b::to_int(value, mode, signed, 32, 0, env);
            if !store32(c, dest, result as u32) {
                return malformed(c, insn);
            }
        }
    }
    next(c, insn);
}

/// Whether a single-precision to integer conversion lands outside the 32-bit range,
/// which is the Invalid Operation condition of `vcvt`.
fn int_out_of_range32(value: u32, mode: Round, signed: bool, env: Fpcr) -> bool {
    let rounded = f32::from_bits(fp::f32b::round_int(value, mode, env)) as f64;
    if rounded.is_nan() {
        return true;
    }
    if signed {
        rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX)
    } else {
        rounded < 0.0 || rounded > f64::from(u32::MAX)
    }
}

fn int_out_of_range64(value: u64, mode: Round, signed: bool, env: Fpcr) -> bool {
    let rounded = f64::from_bits(fp::f64b::round_int(value, mode, env));
    if rounded.is_nan() {
        return true;
    }
    if signed {
        rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX)
    } else {
        rounded < 0.0 || rounded > f64::from(u32::MAX)
    }
}

/// `vext`: bytes from the concatenation of the two sources.
fn neon_extract(c: &mut Cpu, insn: &Insn) {
    let (
        Some(shape),
        [
            Operand::D(dest),
            Operand::D(first),
            Operand::D(second),
            Operand::Imm(index),
        ],
    ) = (insn.shape, insn.operands.as_slice())
    else {
        return malformed(c, insn);
    };
    let count = halves(shape);
    let index = *index as usize;
    // The index runs over the concatenation `first : second`.
    let mut source = Vector::read(c, shape, *first);
    if shape.q {
        let high = Vector::read(
            c,
            VecShape {
                esize: 64,
                q: false,
            },
            DReg::new(second.number()).unwrap_or(*second),
        );
        let _ = high;
    }
    let mut bytes = [0u8; 32];
    for at in 0..count {
        let word = source.halves[at];
        bytes[at * 8..at * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    for at in 0..count {
        let register = DReg::new(second.number() + at as u8).unwrap_or(*second);
        let word = read_d(c, register);
        bytes[(count + at) * 8..(count + at) * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    source = Vector::default();
    for at in 0..count {
        let mut word = [0u8; 8];
        for (byte, slot) in word.iter_mut().enumerate() {
            *slot = bytes.get(index + at * 8 + byte).copied().unwrap_or(0);
        }
        source.halves[at] = u64::from_le_bytes(word);
    }
    source.write(c, shape, *dest);
    next(c, insn);
}

/// `vld1`…`vst2`: the structure transfers, contiguous or interleaved.
fn neon_transfer(c: &mut Cpu, insn: &Insn) {
    let [Operand::NeonMem(mem)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let mem = *mem;
    let load = matches!(
        insn.opcode,
        Opcode::Vld1 | Opcode::Vld2 | Opcode::Vld3 | Opcode::Vld4
    );
    let base = get(c, mem.base, insn.pc);
    let bytes = mem.bytes();
    let element = u32::from(mem.esize) / 8;
    let lanes = 64 / u32::from(mem.esize);
    let mut buffer = [0u8; 32];
    if load {
        for (index, byte) in buffer.iter_mut().enumerate().take(bytes as usize) {
            *byte = c.ld8(u64::from(base.wrapping_add(index as u32))) as u8;
        }
    }
    for slot in 0..u32::from(mem.count) {
        let Some(register) = DReg::new(mem.first.number() + (slot as u8) * mem.stride) else {
            return malformed(c, insn);
        };
        let mut word = 0u64;
        if !load {
            word = read_d(c, register);
        }
        for index in 0..lanes {
            let position = if mem.interleave == 1 {
                slot * lanes + index
            } else {
                index * u32::from(mem.interleave) + slot
            } * element;
            if load {
                let mut value = 0u64;
                for byte in 0..element {
                    value |= u64::from(buffer[(position + byte) as usize]) << (byte * 8);
                }
                set_lane(&mut word, u32::from(mem.esize), index, value);
            } else {
                let value = lane(word, u32::from(mem.esize), index);
                for byte in 0..element {
                    buffer[(position + byte) as usize] = (value >> (byte * 8)) as u8;
                }
            }
        }
        if load {
            write_d(c, register, word);
        }
    }
    if !load {
        for (index, byte) in buffer.iter().enumerate().take(bytes as usize) {
            c.st8(u64::from(base.wrapping_add(index as u32)), u64::from(*byte));
        }
    }
    match mem.update {
        PostUpdate::None => {}
        PostUpdate::Immediate => set(c, mem.base, base.wrapping_add(bytes)),
        PostUpdate::Register(reg) => {
            let offset = get(c, reg, insn.pc);
            set(c, mem.base, base.wrapping_add(offset));
        }
    }
    next(c, insn);
}

/// The three-register Advanced SIMD group.
fn neon_three(c: &mut Cpu, insn: &Insn) {
    let (Some(shape), [Operand::D(dest), Operand::D(first), Operand::D(second)]) =
        (insn.shape, insn.operands.as_slice())
    else {
        return malformed(c, insn);
    };
    let esize = u32::from(shape.esize);
    let lanes = 64 / esize;
    let mask = elem_mask(esize);
    let a = Vector::read(c, shape, *first);
    let b = Vector::read(c, shape, *second);
    let d = Vector::read(c, shape, *dest);
    let env = fpenv(c);
    let mut out = Vector::default();
    for half in 0..halves(shape) {
        let (x, y, z) = (a.halves[half], b.halves[half], d.halves[half]);
        // The bitwise operations work on the whole 64-bit half.
        match insn.opcode {
            Opcode::Vand
            | Opcode::Vorr
            | Opcode::Veor
            | Opcode::Vbic
            | Opcode::Vorn
            | Opcode::Vbsl => {
                out.halves[half] = match insn.opcode {
                    Opcode::Vand => x & y,
                    Opcode::Vorr => x | y,
                    Opcode::Veor => x ^ y,
                    Opcode::Vbic => x & !y,
                    Opcode::Vorn => x | !y,
                    _ => (x & z) | (y & !z),
                };
                continue;
            }
            _ => {}
        }
        let mut word = 0u64;
        for index in 0..lanes {
            let (lane_x, lane_y) = (lane(x, esize, index), lane(y, esize, index));
            let result = match insn.opcode {
                Opcode::Vadd => lane_x.wrapping_add(lane_y) & mask,
                Opcode::Vsub => lane_x.wrapping_sub(lane_y) & mask,
                Opcode::Vmul => lane_x.wrapping_mul(lane_y) & mask,
                Opcode::VmlaI => {
                    lane(z, esize, index).wrapping_add(lane_x.wrapping_mul(lane_y)) & mask
                }
                Opcode::VmlsI => {
                    lane(z, esize, index).wrapping_sub(lane_x.wrapping_mul(lane_y)) & mask
                }
                Opcode::Vmax => {
                    if signed_lane(x, esize, index) >= signed_lane(y, esize, index) {
                        lane_x
                    } else {
                        lane_y
                    }
                }
                Opcode::VmaxU => lane_x.max(lane_y),
                Opcode::Vmin => {
                    if signed_lane(x, esize, index) <= signed_lane(y, esize, index) {
                        lane_x
                    } else {
                        lane_y
                    }
                }
                Opcode::VminU => lane_x.min(lane_y),
                Opcode::Vcgt => {
                    u64::from(signed_lane(x, esize, index) > signed_lane(y, esize, index)) * mask
                }
                Opcode::VcgtU => u64::from(lane_x > lane_y) * mask,
                Opcode::Vcge => {
                    u64::from(signed_lane(x, esize, index) >= signed_lane(y, esize, index)) * mask
                }
                Opcode::VcgeU => u64::from(lane_x >= lane_y) * mask,
                Opcode::Vceq => u64::from(lane_x == lane_y) * mask,
                Opcode::Vtst => u64::from(lane_x & lane_y != 0) * mask,
                Opcode::VaddF => u64::from(fp::f32b::add(lane_x as u32, lane_y as u32, env)),
                Opcode::VsubF => u64::from(fp::f32b::sub(lane_x as u32, lane_y as u32, env)),
                Opcode::VmulF => u64::from(fp::f32b::mul(lane_x as u32, lane_y as u32, env)),
                Opcode::Vmla => u64::from(fp::f32b::mul_add(
                    lane(z, esize, index) as u32,
                    lane_x as u32,
                    lane_y as u32,
                    env,
                )),
                Opcode::Vmls => u64::from(fp::f32b::mul_add(
                    lane(z, esize, index) as u32,
                    fp::f32b::neg(lane_x as u32),
                    lane_y as u32,
                    env,
                )),
                Opcode::VmlaF => u64::from(fp::f32b::mul_add(
                    lane(z, esize, index) as u32,
                    lane_x as u32,
                    lane_y as u32,
                    env,
                )),
                Opcode::VmlsF => u64::from(fp::f32b::mul_add(
                    lane(z, esize, index) as u32,
                    fp::f32b::neg(lane_x as u32),
                    lane_y as u32,
                    env,
                )),
                Opcode::Vfma => u64::from(fp::f32b::mul_add(
                    lane(z, esize, index) as u32,
                    lane_x as u32,
                    lane_y as u32,
                    env,
                )),
                Opcode::Vfms => u64::from(fp::f32b::mul_add(
                    lane(z, esize, index) as u32,
                    fp::f32b::neg(lane_x as u32),
                    lane_y as u32,
                    env,
                )),
                Opcode::VmaxF => u64::from(fp::f32b::max(lane_x as u32, lane_y as u32, env)),
                Opcode::VminF => u64::from(fp::f32b::min(lane_x as u32, lane_y as u32, env)),
                Opcode::VceqF => {
                    let (a, b) = (f32::from_bits(lane_x as u32), f32::from_bits(lane_y as u32));
                    if a.is_nan() || b.is_nan() {
                        set_exception(c, FPSCR_IOC);
                    }
                    u64::from(a == b) * mask
                }
                Opcode::VcgeF => {
                    let (a, b) = (f32::from_bits(lane_x as u32), f32::from_bits(lane_y as u32));
                    if a.is_nan() || b.is_nan() {
                        set_exception(c, FPSCR_IOC);
                    }
                    u64::from(a >= b) * mask
                }
                _ => return malformed(c, insn),
            };
            set_lane(&mut word, esize, index, result);
        }
        out.halves[half] = word;
    }
    out.write(c, shape, *dest);
    next(c, insn);
}

/// The two-register Advanced SIMD group.
fn neon_two(c: &mut Cpu, insn: &Insn) {
    let (Some(shape), [Operand::D(dest), Operand::D(source)]) =
        (insn.shape, insn.operands.as_slice())
    else {
        return malformed(c, insn);
    };
    let esize = u32::from(shape.esize);
    let lanes = 64 / esize;
    let a = Vector::read(c, shape, *source);
    let env = fpenv(c);
    let mut out = Vector::default();
    for half in 0..halves(shape) {
        let word = a.halves[half];
        out.halves[half] = match insn.opcode {
            Opcode::Vmvn => !word,
            // The reversed-element forms move whole elements, so the size selects how
            // many bytes each move covers.
            Opcode::Vrev64 => rev_elements(word, 64),
            Opcode::Vrev32 => rev_elements(word, 32),
            Opcode::Vrev16 => rev_elements(word, 16),
            Opcode::VabsF | Opcode::VnegF | Opcode::VsqrtF => {
                let mut result = 0u64;
                for index in 0..lanes {
                    let value = lane(word, esize, index) as u32;
                    let converted = match insn.opcode {
                        Opcode::VabsF => fp::f32b::abs(value),
                        Opcode::VnegF => fp::f32b::neg(value),
                        _ => {
                            if value & 0x8000_0000 != 0 && !fp::f32b::is_zero(value) {
                                set_exception(c, FPSCR_IOC);
                            }
                            fp::f32b::sqrt(value, env)
                        }
                    };
                    set_lane(&mut result, esize, index, u64::from(converted));
                }
                result
            }
            Opcode::Vabs | Opcode::Vneg | Opcode::VabsU | Opcode::VnegU => {
                let signed = matches!(insn.opcode, Opcode::Vabs | Opcode::Vneg);
                let negate = matches!(insn.opcode, Opcode::Vneg | Opcode::VnegU);
                let mut result = 0u64;
                for index in 0..lanes {
                    let magnitude = if signed {
                        signed_lane(word, esize, index).unsigned_abs()
                    } else {
                        lane(word, esize, index)
                    };
                    let value = if negate {
                        (0u64.wrapping_sub(magnitude)) & elem_mask(esize)
                    } else {
                        magnitude & elem_mask(esize)
                    };
                    set_lane(&mut result, esize, index, value);
                }
                result
            }
            _ => return malformed(c, insn),
        };
    }
    out.write(c, shape, *dest);
    next(c, insn);
}

/// Reverses the elements of the given width inside a 64-bit word.
fn rev_elements(word: u64, esize: u32) -> u64 {
    let count = 64 / esize;
    let mut result = 0u64;
    for index in 0..count {
        let value = lane(word, esize, index);
        set_lane(&mut result, esize, count - 1 - index, value);
    }
    result
}

/// `vmov`, `vorr`, `vmvn` and `vbic` with a modified immediate.
fn neon_immediate(c: &mut Cpu, insn: &Insn) {
    let (Some(shape), [Operand::D(dest), Operand::Imm64(value)]) =
        (insn.shape, insn.operands.as_slice())
    else {
        return malformed(c, insn);
    };
    let mut out = Vector::default();
    for half in 0..halves(shape) {
        let Some(register) = DReg::new(dest.number() + half as u8) else {
            return malformed(c, insn);
        };
        let current = read_d(c, register);
        out.halves[half] = match insn.opcode {
            Opcode::VmovImm => *value,
            Opcode::VorrImm => current | value,
            Opcode::VmvnImm => !value,
            Opcode::VbicImm => current & !value,
            _ => return malformed(c, insn),
        };
    }
    out.write(c, shape, *dest);
    next(c, insn);
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifts_keep_the_architectural_carry() {
        assert_eq!(shift_lsl(0x8000_0001, 0, true), (0x8000_0001, true));
        assert_eq!(shift_lsl(0x8000_0001, 1, false), (0x0000_0002, true));
        assert_eq!(shift_lsl(0x8000_0001, 32, false), (0, true));
        assert_eq!(shift_lsr(0x8000_0001, 1, false), (0x4000_0000, true));
        assert_eq!(shift_lsr(0x8000_0000, 32, false), (0, true));
        assert_eq!(shift_asr(0x8000_0000, 1, false), (0xc000_0000, false));
        assert_eq!(shift_asr(0x8000_0000, 32, false), (0xffff_ffff, true));
        assert_eq!(shift_asr(0x8000_0000, 40, false), (0xffff_ffff, true));
        assert_eq!(shift_ror(0x0000_0003, 1, false), (0x8000_0001, true));
        assert_eq!(shift_ror(0x8000_0001, 0, false), (0x8000_0001, false));
        assert_eq!(shift_ror(0x8000_0001, 0, true), (0x8000_0001, true));
        assert_eq!(shift_ror(0x8000_0001, 32, false), (0x8000_0001, true));
        assert_eq!(shift_ror(0x0000_0001, 33, false), (0x8000_0000, true));
    }

    #[test]
    fn a_zero_immediate_rotation_keeps_the_carry() {
        // `mov r0, #1` (rotation 0) leaves C alone; `mov r0, #0xff000000`
        // (rotation 4) sets C to the last bit rotated out.
        assert!(!modified_immediate_carry(0xE3A0_0001, 1, false));
        assert!(modified_immediate_carry(0xE3A0_0001, 1, true));
        assert!(modified_immediate_carry(0xE3A0_04FF, 0xff00_0000, false));
    }

    #[test]
    fn lane_reads_respect_signedness() {
        assert_eq!(Lane::SignedWrap.read(0xffff_8000, 0, 16), -32768);
        assert_eq!(Lane::SignedWrap.read(0xffff_8000, 1, 16), -1);
        assert_eq!(Lane::UnsignedWrap.read(0xffff_8000, 0, 16), 32768);
        assert_eq!(Lane::SignedWrap.read(0x0000_00ff, 0, 8), -1);
        assert_eq!(Lane::UnsignedWrap.read(0x0000_00ff, 0, 8), 255);
    }

    #[test]
    fn saturating_helpers_clamp_both_ends() {
        assert_eq!(saturate_signed(5, 3), (3, true));
        assert_eq!(saturate_signed(-5, 3), (0xffff_fffc, true));
        assert_eq!(saturate_signed(3, 3), (3, false));
        assert_eq!(saturate_unsigned(-1, 4), (0, true));
        assert_eq!(saturate_unsigned(16, 4), (15, true));
        assert_eq!(saturate_unsigned(15, 4), (15, false));
        assert_eq!(
            saturate_signed32(i64::from(i32::MAX) + 1),
            (i32::MAX as u32, true)
        );
        assert_eq!(saturate_signed32(7), (7, false));
    }

    #[test]
    fn rejection_names_the_missing_class() {
        assert_eq!(
            rejection(0xEE00_0E10, DecodeError::Unsupported),
            "coprocessor access other than VFP/NEON (MCR/MRC/CDP) is outside the A32 slice"
        );
        // VFP and Advanced SIMD now decode, so what is left of those spaces is named
        // precisely instead of being reported as a coprocessor access.
        assert_eq!(
            rejection(0xEEBE_0A40, DecodeError::Unsupported),
            "vcvt between floating point and fixed point is outside the A32 slice"
        );
        assert_eq!(
            rejection(0xF3B1_0C02, DecodeError::Unsupported),
            "an Advanced SIMD two-register encoding this slice does not model (the table lookups, `vswp`, `vdup` from a lane or a shift by immediate)"
        );
        assert_eq!(
            rejection(0xF4A0_000F, DecodeError::Unsupported),
            "the single-lane Advanced SIMD transfers (`vld1.single` and friends) are outside the A32 slice"
        );
        // The divide and media multiply classes now decode, so what is left of them
        // reports its own name.
        assert_eq!(
            rejection(0xE780_0011, DecodeError::Unsupported),
            "a media instruction outside the modelled families is outside the A32 slice"
        );
        assert_eq!(
            rejection(0xEC53_2F03, DecodeError::Unsupported),
            "a `cp15` transfer other than the thread pointers, the counters and `fpscr` is outside the A32 slice"
        );
        assert_eq!(
            rejection(0xE4A1_0004, DecodeError::Unsupported),
            "the unprivileged ldr/str forms (ldrt/strt) are outside the A32 slice"
        );
        assert_eq!(
            rejection(0xE9D0_0001, DecodeError::Unsupported),
            "ldm/stm with the S bit (user bank or exception return) is outside the A32 slice"
        );
        assert_eq!(
            rejection(0xE12F_FF1F, DecodeError::Unpredictable),
            "the encoding is architecturally unpredictable"
        );
        assert_eq!(
            rejection(0xE320_0001, DecodeError::Unallocated),
            "no instruction is assigned to this encoding"
        );
        assert_eq!(
            rejection(0xE102_0092, DecodeError::Deprecated),
            "the deprecated swap instructions (`swp`/`swpb`) are not modelled"
        );
        assert_eq!(
            rejection(0xE12F_F000, DecodeError::Privileged),
            "the encoding needs banked modes, the SPSR or a control field of the CPSR"
        );
    }
}
