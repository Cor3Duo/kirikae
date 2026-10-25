//! A32 (ARM mode) decoder.
//!
//! [`decode`] takes the address of the instruction and its 32-bit word and returns
//! either a decoded [`Insn`] or a [`DecodeError`]. The decoder never panics and never
//! reads outside its arguments, so the discovery loop can run it over every word of a
//! module, including literal pools and padding.
//!
//! Decoding follows the A32 encoding tables of the ARM ARM (ARMv7-A/R, A5.1 "Format of
//! the instruction set" and the A8.8 per-instruction encodings). The class comes from
//! bits 27:25, the form from the opcode bits and from the register/offset selector bits
//! (bit 4 in the data-processing and media classes), and every restriction the
//! architecture states as UNPREDICTABLE is rejected instead of given invented
//! semantics.

use super::ir::{
    Block, BlockMode, Cond, ExecState, Insn, Mem, MemSize, Offset, Opcode, Operand, Reg, RegList,
    Shift, Target,
};
use crate::bits::sext;

/// Why a word is not a decoded instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The encoding belongs to an instruction class outside this slice: coprocessor,
    /// VFP/NEON, media (parallel and halfword forms), the unprivileged `ldrt`/`strt`
    /// forms, `mrs`/`msr`, hints, or the unconditional space other than `blx`.
    Unsupported,
    /// No instruction is assigned to this encoding.
    Unallocated,
    /// The encoding is allocated, but the architecture makes this form unpredictable
    /// (for example an empty register list, a writeback base inside its own list, `pc`
    /// as an index register, or a byte load into `pc`).
    Unpredictable,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DecodeError::Unsupported => "encoding outside the A32 decoder slice",
            DecodeError::Unallocated => "unallocated encoding",
            DecodeError::Unpredictable => "architecturally unpredictable encoding",
        })
    }
}

impl std::error::Error for DecodeError {}

/// Decodes one A32 instruction. `pc` is the address of `word` and is used to resolve
/// direct branch targets.
pub fn decode(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    // Condition field 0b1111 selects the unconditional space, which only contributes
    // `blx` immediate to this slice.
    if word >> 28 == 0b1111 {
        return unconditional(pc, word);
    }
    let cond = Cond::from_field(word >> 28).ok_or(DecodeError::Unsupported)?;

    // `cond 1111 imm24`: supervisor call. Checked before the class dispatch because the
    // opcode nibble 0b1111 is also the coprocessor class.
    if (word >> 24) & 0xf == 0xf {
        return Ok(insn(
            pc,
            word,
            cond,
            Opcode::Svc,
            false,
            vec![Operand::Imm(word & 0x00ff_ffff)],
        ));
    }

    match (word >> 25) & 0b111 {
        0b000 => data_processing_register(pc, word, cond),
        0b001 => data_processing_immediate(pc, word, cond),
        0b010 => transfer_immediate(pc, word, cond),
        // Bit 4 separates the register forms of `ldr`/`str` (0) from the media
        // instructions (1).
        0b011 if word & 0x10 == 0 => transfer_register(pc, word, cond),
        0b011 => Err(DecodeError::Unsupported),
        0b100 => block_transfer(pc, word, cond),
        0b101 => branch(pc, word, cond),
        // 0b110 coprocessor load/store, 0b111 coprocessor data operations.
        _ => Err(DecodeError::Unsupported),
    }
}

fn insn(
    pc: u32,
    word: u32,
    cond: Cond,
    opcode: Opcode,
    sets_flags: bool,
    operands: Vec<Operand>,
) -> Insn {
    Insn {
        pc,
        word,
        cond,
        opcode,
        sets_flags,
        operands,
    }
}

fn reg(bits: u32) -> Reg {
    Reg::from_encoding(bits)
}

/// Normalises the immediate shift of a register operand. `LSR #0` and `ASR #0` encode a
/// shift by 32 and `ROR #0` encodes `RRX`.
fn imm_shift(kind: u32, amount: u32) -> Shift {
    match kind & 3 {
        0 => Shift::Lsl(amount as u8),
        1 => Shift::Lsr(if amount == 0 { 32 } else { amount as u8 }),
        2 => Shift::Asr(if amount == 0 { 32 } else { amount as u8 }),
        _ => {
            if amount == 0 {
                Shift::Rrx
            } else {
                Shift::Ror(amount as u8)
            }
        }
    }
}

fn reg_shift(kind: u32, reg: Reg) -> Shift {
    match kind & 3 {
        0 => Shift::LslReg(reg),
        1 => Shift::LsrReg(reg),
        2 => Shift::AsrReg(reg),
        _ => Shift::RorReg(reg),
    }
}

/// Sign-extends a 24-bit branch offset and scales it by four.
fn branch_offset(imm24: u32) -> u32 {
    sext(((imm24 & 0x00ff_ffff) as u64) << 2, 26) as u32
}

fn unconditional(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    if (word >> 25) & 0b111 != 0b101 {
        // `cps`, `setend`, `pld`/`pli`, Advanced SIMD, barriers and hints, coprocessor.
        return Err(DecodeError::Unsupported);
    }
    // `blx` immediate: Align(PC, 4) + SignExtend(imm24) << 2 + H * 2. H contributes
    // bit 1 of the target, so the branch lands on a halfword boundary inside Thumb
    // code; the address keeps bit 0 clear and the state is implied.
    let half = (word >> 24) & 1;
    let addr = (pc.wrapping_add(8) & !3)
        .wrapping_add(branch_offset(word & 0x00ff_ffff))
        .wrapping_add(half * 2);
    Ok(insn(
        pc,
        word,
        Cond::Al,
        Opcode::Blx,
        false,
        vec![Operand::Target(Target {
            addr,
            state: ExecState::Thumb,
        })],
    ))
}

/// Data-processing instructions with an immediate operand, including `movw`/`movt`.
fn data_processing_immediate(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let opcode_field = (word >> 20) & 0xff;
    if opcode_field == 0x30 || opcode_field == 0x34 {
        let rd = reg((word >> 12) & 0xf);
        if rd == Reg::PC {
            return Err(DecodeError::Unpredictable);
        }
        let imm = (((word >> 16) & 0xf) << 12) | (word & 0xfff);
        let opcode = if opcode_field == 0x30 {
            Opcode::Movw
        } else {
            Opcode::Movt
        };
        return Ok(insn(
            pc,
            word,
            cond,
            opcode,
            false,
            vec![Operand::Reg(rd), Operand::Imm(imm)],
        ));
    }

    let sets_flags = (word >> 20) & 1 != 0;
    let rd = reg((word >> 12) & 0xf);
    let rn = reg((word >> 16) & 0xf);
    let opcode = Opcode::from_data_processing((word >> 21) & 0xf);
    if opcode.is_flag_test() {
        if !sets_flags {
            // Opcodes 0b1000 and 0b1010 with S = 0 are `movw`/`movt` and were handled
            // above. The remaining forms of 0b1001 and 0b1011 are `msr` immediate and
            // the hints (`nop`, `yield`, `wfe`, ...), which put 0b1111 in Rd.
            return Err(if rd == Reg::PC {
                DecodeError::Unsupported
            } else {
                DecodeError::Unallocated
            });
        }
        if rd != Reg::R0 {
            return Err(DecodeError::Unpredictable);
        }
    } else if matches!(opcode, Opcode::Mov | Opcode::Mvn) && rn != Reg::R0 {
        // Rn is SBZ in the `mov`/`mvn` encodings.
        return Err(DecodeError::Unpredictable);
    }
    let rotate = 2 * ((word >> 8) & 0xf);
    let imm = (word & 0xff).rotate_right(rotate);
    let operands = data_processing_operands(opcode, rd, rn, Operand::Imm(imm));
    Ok(insn(
        pc,
        word,
        cond,
        opcode,
        sets_flags || opcode.is_flag_test(),
        operands,
    ))
}

fn data_processing_operands(opcode: Opcode, rd: Reg, rn: Reg, op2: Operand) -> Vec<Operand> {
    if opcode.is_flag_test() {
        vec![Operand::Reg(rn), op2]
    } else if matches!(opcode, Opcode::Mov | Opcode::Mvn) {
        vec![Operand::Reg(rd), op2]
    } else {
        vec![Operand::Reg(rd), Operand::Reg(rn), op2]
    }
}

/// Data-processing instructions with a register operand: the register-shift forms, the
/// miscellaneous space (`bx`, `blx`, `bkpt`) and the multiplies.
fn data_processing_register(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if word & 0x0fc0_00f0 == 0x90 {
        return multiply(pc, word, cond);
    }

    let sets_flags = (word >> 20) & 1 != 0;
    let rd = reg((word >> 12) & 0xf);
    let opcode = Opcode::from_data_processing((word >> 21) & 0xf);
    if opcode.is_flag_test() {
        // `tst`/`teq`/`cmp`/`cmn` require S = 1 and Rd = 0. The other words with these
        // opcodes are the miscellaneous space (`mrs`, `msr`, `clz`, ...), which uses
        // bit 4 and bit 7 to select its own forms.
        let shift_form = word & 0x10 == 0 || word & 0x80 == 0;
        if !sets_flags || !shift_form {
            return misc(pc, word, cond);
        }
        if rd != Reg::R0 {
            return Err(DecodeError::Unpredictable);
        }
    }
    let rn = reg((word >> 16) & 0xf);
    if word & 0x90 == 0x90 {
        // Multiply long, swap, halfword multiply and the extra load/store forms live in
        // this space: outside the slice.
        return Err(DecodeError::Unsupported);
    }
    if matches!(opcode, Opcode::Mov | Opcode::Mvn) && rn != Reg::R0 {
        // Rn is SBZ in the `mov`/`mvn` encodings.
        return Err(DecodeError::Unpredictable);
    }

    let rm = reg(word & 0xf);
    let shift = if word & 0x10 == 0 {
        imm_shift((word >> 5) & 3, (word >> 7) & 0x1f)
    } else {
        reg_shift((word >> 5) & 3, reg((word >> 8) & 0xf))
    };
    let op2 = Operand::Shifted { reg: rm, shift };
    let operands = data_processing_operands(opcode, rd, rn, op2);
    Ok(insn(
        pc,
        word,
        cond,
        opcode,
        sets_flags || opcode.is_flag_test(),
        operands,
    ))
}

/// The miscellaneous space reachable from data-processing opcodes 0b1000-0b1011:
/// `bx`, `blx` (register) and `bkpt`. Everything else (`mrs`, `msr`, `clz`, `qadd`, ...)
/// is reported as unsupported.
fn misc(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let rm = reg(word & 0xf);
    if word & 0x0fff_fff0 == 0x012f_ff10 {
        if rm == Reg::PC {
            return Err(DecodeError::Unpredictable);
        }
        return Ok(insn(
            pc,
            word,
            cond,
            Opcode::Bx,
            false,
            vec![Operand::Reg(rm)],
        ));
    }
    if word & 0x0fff_fff0 == 0x012f_ff30 {
        if rm == Reg::PC {
            return Err(DecodeError::Unpredictable);
        }
        return Ok(insn(
            pc,
            word,
            cond,
            Opcode::Blx,
            false,
            vec![Operand::Reg(rm)],
        ));
    }
    if word & 0x0ff0_00f0 == 0x0120_0070 {
        // `bkpt` is only defined with cond = 0b1110.
        if cond != Cond::Al {
            return Err(DecodeError::Unpredictable);
        }
        let imm = (((word >> 8) & 0xfff) << 4) | (word & 0xf);
        return Ok(insn(
            pc,
            word,
            cond,
            Opcode::Bkpt,
            false,
            vec![Operand::Imm(imm)],
        ));
    }
    Err(DecodeError::Unsupported)
}

fn multiply(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let accumulate = (word >> 21) & 1 != 0;
    let sets_flags = (word >> 20) & 1 != 0;
    let rd = reg((word >> 16) & 0xf);
    let rn = reg((word >> 12) & 0xf);
    let rs = reg((word >> 8) & 0xf);
    let rm = reg(word & 0xf);
    if rd == Reg::PC || rs == Reg::PC || rm == Reg::PC || (accumulate && rn == Reg::PC) {
        return Err(DecodeError::Unpredictable);
    }
    if !accumulate && rn != Reg::R0 {
        // The accumulate field is SBZ in `mul`.
        return Err(DecodeError::Unpredictable);
    }
    // Operand order follows the assembler syntax: `mla Rd, Rm, Rs, Ra` computes
    // `Rd = Rm * Rs + Ra`, with the accumulator encoded in bits 15:12.
    let (opcode, operands) = if accumulate {
        (
            Opcode::Mla,
            vec![
                Operand::Reg(rd),
                Operand::Reg(rm),
                Operand::Reg(rs),
                Operand::Reg(rn),
            ],
        )
    } else {
        (
            Opcode::Mul,
            vec![Operand::Reg(rd), Operand::Reg(rm), Operand::Reg(rs)],
        )
    };
    Ok(insn(pc, word, cond, opcode, sets_flags, operands))
}

/// Addressing and type bits shared by the immediate and register forms of `ldr`/`str`.
struct Transfer {
    pre_indexed: bool,
    add: bool,
    byte: bool,
    writeback: bool,
    load: bool,
}

impl Transfer {
    fn from_word(word: u32) -> Transfer {
        Transfer {
            pre_indexed: (word >> 24) & 1 != 0,
            add: (word >> 23) & 1 != 0,
            byte: (word >> 22) & 1 != 0,
            writeback: (word >> 21) & 1 != 0,
            load: (word >> 20) & 1 != 0,
        }
    }

    fn size(&self) -> MemSize {
        if self.byte {
            MemSize::Byte
        } else {
            MemSize::Word
        }
    }

    fn opcode(&self) -> Opcode {
        if self.load { Opcode::Ldr } else { Opcode::Str }
    }

    /// `P = 0` writes the base back even though `W` must be zero.
    fn effective_writeback(&self) -> bool {
        !self.pre_indexed || self.writeback
    }
}

fn check_transfer(transfer: &Transfer, base: Reg, rt: Reg) -> Result<(), DecodeError> {
    // `if P == '0' && W == '1' then SEE "LDRT"/"STRT"`: the unprivileged forms are
    // outside the slice.
    if !transfer.pre_indexed && transfer.writeback {
        return Err(DecodeError::Unsupported);
    }
    if transfer.effective_writeback() && (base == Reg::PC || base == rt) {
        return Err(DecodeError::Unpredictable);
    }
    // A word load to PC is an interworking branch; a byte load to PC is unpredictable.
    if rt == Reg::PC && transfer.size() == MemSize::Byte {
        return Err(DecodeError::Unpredictable);
    }
    Ok(())
}

fn transfer_immediate(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let transfer = Transfer::from_word(word);
    let base = reg((word >> 16) & 0xf);
    let rt = reg((word >> 12) & 0xf);
    check_transfer(&transfer, base, rt)?;
    let mem = Mem {
        base,
        offset: Offset::Imm((word & 0xfff) as u16),
        add: transfer.add,
        pre_indexed: transfer.pre_indexed,
        writeback: transfer.writeback,
        size: transfer.size(),
    };
    Ok(insn(
        pc,
        word,
        cond,
        transfer.opcode(),
        false,
        vec![Operand::Reg(rt), Operand::Mem(mem)],
    ))
}

fn transfer_register(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let transfer = Transfer::from_word(word);
    let index = reg(word & 0xf);
    if index == Reg::PC {
        return Err(DecodeError::Unpredictable);
    }
    let base = reg((word >> 16) & 0xf);
    let rt = reg((word >> 12) & 0xf);
    check_transfer(&transfer, base, rt)?;
    let mem = Mem {
        base,
        offset: Offset::Reg {
            reg: index,
            shift: imm_shift((word >> 5) & 3, (word >> 7) & 0x1f),
        },
        add: transfer.add,
        pre_indexed: transfer.pre_indexed,
        writeback: transfer.writeback,
        size: transfer.size(),
    };
    Ok(insn(
        pc,
        word,
        cond,
        transfer.opcode(),
        false,
        vec![Operand::Reg(rt), Operand::Mem(mem)],
    ))
}

fn block_transfer(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let pre_indexed = (word >> 24) & 1 != 0;
    let add = (word >> 23) & 1 != 0;
    let user_bank = (word >> 22) & 1 != 0;
    let writeback = (word >> 21) & 1 != 0;
    let load = (word >> 20) & 1 != 0;
    if user_bank {
        // `ldm`/`stm` with the S bit transfer the user bank or return from an
        // exception; both are outside the slice.
        return Err(DecodeError::Unsupported);
    }
    let list = RegList::new((word & 0xffff) as u16).ok_or(DecodeError::Unpredictable)?;
    let base = reg((word >> 16) & 0xf);
    if base == Reg::PC {
        return Err(DecodeError::Unpredictable);
    }
    if writeback && list.contains(base) {
        return Err(DecodeError::Unpredictable);
    }
    if !load && list.contains(Reg::PC) {
        return Err(DecodeError::Unpredictable);
    }
    let mode = match (pre_indexed, add) {
        (false, true) => BlockMode::IncrementAfter,
        (true, true) => BlockMode::IncrementBefore,
        (false, false) => BlockMode::DecrementAfter,
        (true, false) => BlockMode::DecrementBefore,
    };
    let block = Block {
        base,
        list,
        mode,
        writeback,
    };
    Ok(insn(
        pc,
        word,
        cond,
        if load { Opcode::Ldm } else { Opcode::Stm },
        false,
        vec![Operand::Block(block)],
    ))
}

fn branch(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let link = (word >> 24) & 1 != 0;
    let addr = pc
        .wrapping_add(8)
        .wrapping_add(branch_offset(word & 0x00ff_ffff));
    Ok(insn(
        pc,
        word,
        cond,
        if link { Opcode::Bl } else { Opcode::B },
        false,
        vec![Operand::Target(Target {
            addr,
            state: ExecState::Arm,
        })],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_immediates_are_normalised() {
        assert_eq!(imm_shift(0, 0), Shift::Lsl(0));
        assert_eq!(imm_shift(0, 31), Shift::Lsl(31));
        assert_eq!(imm_shift(1, 0), Shift::Lsr(32));
        assert_eq!(imm_shift(2, 0), Shift::Asr(32));
        assert_eq!(imm_shift(3, 0), Shift::Rrx);
        assert_eq!(imm_shift(3, 1), Shift::Ror(1));
    }

    #[test]
    fn branch_offsets_sign_extend_and_scale() {
        assert_eq!(branch_offset(0), 0);
        assert_eq!(branch_offset(1), 4);
        assert_eq!(branch_offset(0x00ff_ffff), 0xffff_fffc);
    }
}
