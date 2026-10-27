use super::ir::{
    Block, BlockMode, Cond, ExecState, Insn, Mem, MemSize, Offset, Operand, Reg, RegList, Shift,
    Target,
};
use super::opcode::Opcode;
use super::{neon, vfp};
use crate::bits::sext;

mod media;
mod multiply;
mod transfer;

use media::*;
use multiply::*;
use transfer::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    Unsupported,
    Unallocated,
    Unpredictable,
    Deprecated,
    Privileged,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DecodeError::Unsupported => "encoding outside the A32 decoder slice",
            DecodeError::Unallocated => "unallocated encoding",
            DecodeError::Unpredictable => "architecturally unpredictable encoding",
            DecodeError::Deprecated => "architecturally deprecated encoding",
            DecodeError::Privileged => "encoding needs processor state outside user mode",
        })
    }
}

impl std::error::Error for DecodeError {}

pub fn describe_unsupported(word: u32) -> &'static str {
    if word >> 28 == 0b1111 {
        return match (word >> 25) & 0b111 {
            0b001 if word & 0x0fb0_0000 == 0x03b0_0000 => {
                "an Advanced SIMD two-register encoding this slice does not model (the table lookups, `vswp`, `vdup` from a lane or a shift by immediate)"
            }
            0b001 => "an Advanced SIMD data-processing encoding this slice does not model",
            0b010 if word & 0x0100_0000 == 0 => {
                if word & 0x0080_0000 != 0 {
                    "the single-lane Advanced SIMD transfers (`vld1.single` and friends) are outside the A32 slice"
                } else {
                    "an Advanced SIMD structure transfer this slice does not model"
                }
            }
            0b010 => {
                "the unconditional space (`cps`, `setend`, `pli` or a reserved hint) is outside the A32 slice"
            }
            _ => {
                "the unconditional space (barriers, hints, PLD or Advanced SIMD) is outside the A32 slice"
            }
        };
    }
    match (word >> 25) & 0b111 {
        0b110 | 0b111 => {
            let cp = (word >> 8) & 0xf;
            let data_processing = (word >> 24) & 0xf == 0xe;
            let single = cp == 0b1010;
            if cp == 0b1111 {
                "a `cp15` transfer other than the thread pointers, the counters and `fpscr` is outside the A32 slice"
            } else if cp != 0b1010 && cp != 0b1011 {
                "coprocessor access other than VFP/NEON (MCR/MRC/CDP) is outside the A32 slice"
            } else if data_processing && word & 0x10 == 0 && (word >> 16) & 0xf == 0xe {
                "vcvt between floating point and fixed point is outside the A32 slice"
            } else if data_processing && word & 0x10 != 0 && (word >> 21) & 0b111 == 0b000 && single
            {
                "the two-single vmov forms (`vmov Sd, Sd+1, Rt, Rt2`) are outside the A32 slice"
            } else if !data_processing && (word >> 24) & 1 == 0 && word & 0x0010_0000 == 0 {
                "this 64-bit VFP transfer is outside the A32 slice"
            } else {
                "this VFP encoding is outside the A32 slice"
            }
        }
        0b011 => "a media instruction outside the modelled families is outside the A32 slice",
        0b100 => "ldm/stm with the S bit (user bank or exception return) is outside the A32 slice",
        0b010 => "the unprivileged ldr/str forms (ldrt/strt) are outside the A32 slice",
        0b001 => "the msr immediate or hint form is outside the A32 slice",
        _ => {
            "the miscellaneous space (umaal, the unprivileged forms or a reserved encoding) is outside the A32 slice"
        }
    }
}

pub fn decode(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    if word >> 28 == 0b1111 {
        return unconditional(pc, word);
    }
    let cond = Cond::from_field(word >> 28).ok_or(DecodeError::Unsupported)?;

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
        0b011 if word & 0x10 == 0 => transfer_register(pc, word, cond),
        0b011 => media(pc, word, cond),
        0b100 => block_transfer(pc, word, cond),
        0b101 => branch(pc, word, cond),
        _ => vfp::decode(pc, word, cond),
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
        shape: None,
        operands,
    }
}

pub(super) fn vinsn(
    pc: u32,
    word: u32,
    cond: Cond,
    opcode: Opcode,
    shape: Option<super::ir::VecShape>,
    operands: Vec<Operand>,
) -> Insn {
    Insn {
        pc,
        word,
        cond,
        opcode,
        sets_flags: false,
        shape,
        operands,
    }
}

fn reg(bits: u32) -> Reg {
    Reg::from_encoding(bits)
}

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

fn branch_offset(imm24: u32) -> u32 {
    sext(((imm24 & 0x00ff_ffff) as u64) << 2, 26) as u32
}

fn rotated_immediate(word: u32) -> u32 {
    (word & 0xff).rotate_right(2 * ((word >> 8) & 0xf))
}

fn require_not_pc(reg: Reg) -> Result<Reg, DecodeError> {
    if reg == Reg::PC {
        Err(DecodeError::Unpredictable)
    } else {
        Ok(reg)
    }
}

fn unconditional(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    match (word >> 25) & 0b111 {
        0b101 => {
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
        0b001 => neon::decode(pc, word),
        0b010 if word & 0x0110_0000 == 0 => neon::decode(pc, word),
        0b010 => hint_space(pc, word),
        _ => Err(DecodeError::Unsupported),
    }
}

fn hint_space(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    if word & 0x0fff_ff00 == 0x057f_f000 {
        let low = word & 0xf;
        let opcode = match (word >> 4) & 0xf {
            0b0001 if low == 0xf => Opcode::Clrex,
            0b0100 => Opcode::Dsb,
            0b0101 => Opcode::Dmb,
            0b0110 => Opcode::Isb,
            0b0111 if low == 0 => Opcode::Sb,
            _ => return Err(DecodeError::Unallocated),
        };
        let operands = if matches!(opcode, Opcode::Dsb | Opcode::Dmb | Opcode::Isb) {
            vec![Operand::Imm(low)]
        } else {
            Vec::new()
        };
        return Ok(insn(pc, word, Cond::Al, opcode, false, operands));
    }
    let op1 = (word >> 20) & 0x1f;
    if op1 & 0b00111 == 0b00101 && (word >> 12) & 0xf == 0xf {
        let mem = Mem {
            base: reg((word >> 16) & 0xf),
            offset: Offset::Imm((word & 0xfff) as u16),
            add: (word >> 23) & 1 != 0,
            pre_indexed: true,
            writeback: false,
            size: MemSize::Word,
        };
        return Ok(insn(
            pc,
            word,
            Cond::Al,
            if word & 0x0100_0000 == 0 { Opcode::Pli } else { Opcode::Pld },
            false,
            vec![Operand::Mem(mem)],
        ));
    }
    Err(DecodeError::Unsupported)
}

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
            if rd != Reg::PC {
                return Err(DecodeError::Unallocated);
            }
            return match opcode {
                Opcode::Teq | Opcode::Cmn => system_immediate(pc, word, cond),
                _ => Err(DecodeError::Unallocated),
            };
        }
        if rd != Reg::R0 {
            return Err(DecodeError::Unpredictable);
        }
    } else if matches!(opcode, Opcode::Mov | Opcode::Mvn) && rn != Reg::R0 {
        return Err(DecodeError::Unpredictable);
    }
    let imm = rotated_immediate(word);
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

fn system_immediate(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let spsr = (word >> 22) & 1 != 0;
    let mask = (word >> 16) & 0xf;
    let rotate = (word >> 8) & 0xf;
    let imm8 = word & 0xff;
    if mask == 0 && rotate == 0 && !spsr {
        let opcode = match imm8 {
            0 => Opcode::Nop,
            1 => Opcode::Yield,
            2 => Opcode::Wfe,
            3 => Opcode::Wfi,
            4 => Opcode::Sev,
            5 => Opcode::Sevl,
            _ => Opcode::Nop,
        };
        return Ok(insn(pc, word, cond, opcode, false, Vec::new()));
    }
    if mask == 0 {
        return Err(DecodeError::Unpredictable);
    }
    if spsr || mask != 0b1000 {
        return Err(DecodeError::Privileged);
    }
    Ok(insn(
        pc,
        word,
        cond,
        Opcode::Msr,
        false,
        vec![Operand::Imm(rotated_immediate(word))],
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

fn data_processing_register(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if word & 0x0fc0_00f0 == 0x90 {
        return multiply(pc, word, cond);
    }
    if word & 0x0ff0_00f0 == 0x0060_0090 {
        return multiply_subtract(pc, word, cond);
    }

    if word & 0x90 == 0x90 {
        return multiply_and_transfer(pc, word, cond);
    }

    let sets_flags = (word >> 20) & 1 != 0;
    let rd = reg((word >> 12) & 0xf);
    let opcode = Opcode::from_data_processing((word >> 21) & 0xf);
    if opcode.is_flag_test() {
        let shift_form = word & 0x10 == 0 || word & 0x80 == 0;
        if !sets_flags || !shift_form {
            return misc(pc, word, cond);
        }
        if rd != Reg::R0 {
            return Err(DecodeError::Unpredictable);
        }
    }
    let rn = reg((word >> 16) & 0xf);
    if matches!(opcode, Opcode::Mov | Opcode::Mvn) && rn != Reg::R0 {
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

    let opcode_field = (word >> 21) & 0xf;
    let nibble = (word >> 4) & 0xf;
    match (opcode_field, nibble) {
        (0b1000 | 0b1010, 0b0000) => mrs(pc, word, cond),
        (0b1000, 0b0101) => qadd(pc, word, cond),
        (0b1001 | 0b1011, 0b0000) => msr_register(pc, word, cond),
        (0b1001, 0b0101) => qadd(pc, word, cond),
        (0b1010, 0b0101) => qadd(pc, word, cond),
        (0b1010, 0b1000 | 0b1010 | 0b1100 | 0b1110) => halfword_multiply(pc, word, cond),
        (0b1011, 0b0001) => clz(pc, word, cond),
        (0b1011, 0b0101) => qadd(pc, word, cond),
        (0b1011, 0b1000 | 0b1010 | 0b1100 | 0b1110) => halfword_multiply(pc, word, cond),
        (0b1000, 0b1000 | 0b1010 | 0b1100 | 0b1110) => halfword_multiply(pc, word, cond),
        (0b1001, 0b1000 | 0b1100) => halfword_multiply(pc, word, cond),
        _ => Err(DecodeError::Unsupported),
    }
}

fn mrs(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if (word >> 16) & 0xf != 0xf || word & 0xfff != 0 {
        return Err(DecodeError::Unpredictable);
    }
    let rd = require_not_pc(reg((word >> 12) & 0xf))?;
    if (word >> 22) & 1 != 0 {
        return Err(DecodeError::Privileged);
    }
    Ok(insn(
        pc,
        word,
        cond,
        Opcode::Mrs,
        false,
        vec![Operand::Reg(rd)],
    ))
}

fn msr_register(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if (word >> 12) & 0xf != 0xf || (word >> 4) & 0xff != 0 {
        return Err(DecodeError::Unpredictable);
    }
    let rm = require_not_pc(reg(word & 0xf))?;
    let mask = (word >> 16) & 0xf;
    if (word >> 22) & 1 != 0 || mask != 0b1000 {
        return Err(if mask == 0 && (word >> 22) & 1 == 0 {
            DecodeError::Unpredictable
        } else {
            DecodeError::Privileged
        });
    }
    Ok(insn(
        pc,
        word,
        cond,
        Opcode::Msr,
        false,
        vec![Operand::Reg(rm)],
    ))
}

fn clz(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if (word >> 16) & 0xf != 0xf || (word >> 8) & 0xf != 0xf {
        return Err(DecodeError::Unpredictable);
    }
    let rd = require_not_pc(reg((word >> 12) & 0xf))?;
    let rm = require_not_pc(reg(word & 0xf))?;
    Ok(insn(
        pc,
        word,
        cond,
        Opcode::Clz,
        false,
        vec![Operand::Reg(rd), Operand::Reg(rm)],
    ))
}

fn qadd(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if (word >> 8) & 0xf != 0 {
        return Err(DecodeError::Unpredictable);
    }
    let rd = require_not_pc(reg((word >> 12) & 0xf))?;
    let rn = require_not_pc(reg((word >> 16) & 0xf))?;
    let rm = require_not_pc(reg(word & 0xf))?;
    let double = (word >> 22) & 1 != 0;
    let subtract = (word >> 21) & 1 != 0;
    let opcode = match (double, subtract) {
        (false, false) => Opcode::Qadd,
        (false, true) => Opcode::Qsub,
        (true, false) => Opcode::Qdadd,
        (true, true) => Opcode::Qdsub,
    };
    Ok(insn(
        pc,
        word,
        cond,
        opcode,
        false,
        vec![Operand::Reg(rd), Operand::Reg(rm), Operand::Reg(rn)],
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

    #[test]
    fn rotated_immediates_rotate_right() {
        assert_eq!(rotated_immediate(0xE3A0_00FF), 0xff);
        assert_eq!(rotated_immediate(0xE3A0_04FF), 0xff00_0000);
    }
}
