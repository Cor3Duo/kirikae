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

use std::sync::Arc;

use kirikae_a64::a32::{
    self, BlockMode, DecodeError, Insn, MemSize, Offset, Opcode, Operand, Reg, Shift,
};

use crate::cpu::Cpu;
use crate::horizon;

const N: u32 = 1 << 31;
const Z: u32 = 1 << 30;
const C: u32 = 1 << 29;
const V: u32 = 1 << 28;

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
        DecodeError::Unsupported if word >> 28 == 0b1111 => {
            "the unconditional space (barriers, hints, PLD or Advanced SIMD) is outside the A32 slice"
        }
        DecodeError::Unsupported => match class {
            0b110 | 0b111 => "coprocessor access (VFP/NEON or MCR/MRC) is outside the A32 slice",
            0b011 => "the media/parallel class is outside the A32 slice",
            0b100 => {
                "ldm/stm with the S bit (user bank or exception return) is outside the A32 slice"
            }
            0b010 => "the unprivileged ldr/str forms (ldrt/strt) are outside the A32 slice",
            _ => {
                "the miscellaneous space (mrs/msr/clz and the extra load/store forms) is outside the A32 slice"
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

fn set_flags(c: &mut Cpu, result: u32, carry: Option<bool>, overflow: Option<bool>) {
    let old = c.nzcv as u32;
    let mut psr = 0u32;
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

fn transfer(c: &mut Cpu, insn: &Insn) {
    let [Operand::Reg(rt), Operand::Mem(mem)] = insn.operands.as_slice() else {
        return malformed(c, insn);
    };
    let (rt, mem) = (*rt, *mem);
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
    let access = if mem.pre_indexed { address } else { base };

    if insn.opcode == Opcode::Ldr {
        let value = match mem.size {
            MemSize::Byte => c.ld8(u64::from(access)) as u32,
            MemSize::Word => c.ld32(u64::from(access)) as u32,
        };
        if mem.writes_back() {
            set(c, mem.base, address);
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
        match mem.size {
            MemSize::Byte => c.st8(u64::from(access), value as u64),
            MemSize::Word => c.st32(u64::from(access), value as u64),
        }
        if mem.writes_back() {
            set(c, mem.base, address);
        }
        next(c, insn);
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
        Opcode::Ldr | Opcode::Str => transfer(c, insn),
        Opcode::Ldm | Opcode::Stm => block_transfer(c, insn),
        Opcode::Mul | Opcode::Mla => multiply(c, insn),
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
    fn rejection_names_the_missing_class() {
        assert_eq!(
            rejection(0xEE00_0010, DecodeError::Unsupported),
            "coprocessor access (VFP/NEON or MCR/MRC) is outside the A32 slice"
        );
        assert_eq!(
            rejection(0xE6BF_0F30, DecodeError::Unsupported),
            "the media/parallel class is outside the A32 slice"
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
    }
}
