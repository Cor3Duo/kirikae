//! AArch32 (ARM mode) interpreter.
//!
//! This runs the classes `kirikae_a64::a32::decode` accepts, with the
//! architectural 32-bit register file and PC reads as `address + 8`. A32 is a
//! 32-bit architecture, so results, addresses and branch targets wrap modulo
//! 2^32; the guest map is still the low 4 GiB of the 64-bit reservation.
//!
//! Out of scope, reported instead of guessed: Thumb state, the banked `sp`/`lr`
//! of the privileged modes (and therefore `s` writes to `pc`), VFP/NEON and the
//! rest of the coprocessor space, and any encoding the decoder rejects.
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
    self, BlockMode, DecodeError, Insn, MemSize, Offset, Opcode, Operand, Reg, Shift,
};
use kirikae_a64::bits::sext;

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
    let class = (word >> 25) & 0b111;
    match error {
        DecodeError::Unallocated => "no instruction is assigned to this encoding",
        DecodeError::Unpredictable => "the encoding is architecturally unpredictable",
        DecodeError::Deprecated => {
            "the deprecated swap instructions (`swp`/`swpb`) are not modelled"
        }
        DecodeError::Privileged => {
            "the encoding needs banked modes, the SPSR or a control field of the CPSR"
        }
        DecodeError::Unsupported if word >> 28 == 0b1111 => {
            "the unconditional space (barriers, hints, PLD or Advanced SIMD) is outside the A32 slice"
        }
        DecodeError::Unsupported => match class {
            0b110 | 0b111 => "coprocessor access (VFP/NEON or MCR/MRC) is outside the A32 slice",
            0b011 => {
                "the media multiply-accumulate forms (smlad/smlsd/smmul) are outside the A32 slice"
            }
            0b100 => {
                "ldm/stm with the S bit (user bank or exception return) is outside the A32 slice"
            }
            0b010 => "the unprivileged ldr/str forms (ldrt/strt) are outside the A32 slice",
            0b001 => "the msr immediate or hint form is outside the A32 slice",
            _ => {
                "the miscellaneous space (umaal, the unprivileged forms or a reserved encoding) is outside the A32 slice"
            }
        },
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
        Opcode::Ldr | Opcode::Ldrh | Opcode::Ldrsb | Opcode::Ldrsh
    );
    if load {
        let value = match insn.opcode {
            Opcode::Ldr => match mem.size {
                MemSize::Byte => c.ld8(u64::from(access)) as u32,
                MemSize::Halfword => c.ld16(u64::from(access)) as u32,
                MemSize::Word => c.ld32(u64::from(access)) as u32,
                MemSize::Double => return malformed(c, insn),
            },
            Opcode::Ldrh => c.ld16(u64::from(access)) as u32,
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
            Opcode::Strh => c.st16(u64::from(access), value),
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
        Opcode::Ldr | Opcode::Str | Opcode::Ldrh | Opcode::Strh | Opcode::Ldrsb | Opcode::Ldrsh => {
            transfer(c, insn)
        }
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
        Opcode::Nop | Opcode::Yield | Opcode::Wfe | Opcode::Wfi | Opcode::Sev => next(c, insn),
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
            rejection(0xEE00_0010, DecodeError::Unsupported),
            "coprocessor access (VFP/NEON or MCR/MRC) is outside the A32 slice"
        );
        assert_eq!(
            rejection(0xE700_0010, DecodeError::Unsupported),
            "the media multiply-accumulate forms (smlad/smlsd/smmul) are outside the A32 slice"
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
