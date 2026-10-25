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
    Block, BlockMode, Cond, ExecState, Insn, Mem, MemSize, Offset, Operand, Reg, RegList, Shift,
    Target,
};
use super::opcode::Opcode;
use super::{neon, vfp};
use crate::bits::sext;

/// Why a word is not a decoded instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The encoding belongs to an instruction class outside this slice: coprocessor,
    /// VFP/NEON, the multiply-accumulate media forms (`smlad`, `smmla`, ...), `umaal`,
    /// the unprivileged `ldrt`/`strt` forms, or the unconditional space other than
    /// `blx`.
    Unsupported,
    /// No instruction is assigned to this encoding.
    Unallocated,
    /// The encoding is allocated, but the architecture makes this form unpredictable
    /// (for example an empty register list, a writeback base inside its own list, `pc`
    /// as an index register, a byte load into `pc`, or a non-zero SBZ field).
    Unpredictable,
    /// An architecturally deprecated encoding this slice does not model (`swp`/`swpb`).
    Deprecated,
    /// An encoding that reaches processor state this runtime does not model: the SPSR,
    /// a `msr` of any control or status field other than the flags, or the banked
    /// register file.
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

/// Names the family a rejected word belongs to, in the terms the report needs: the
/// class that is missing, not just "unsupported".
pub fn describe_unsupported(word: u32) -> &'static str {
    if word >> 28 == 0b1111 {
        return match (word >> 25) & 0b111 {
            // The two-register group holds the table lookups, `vswp`, `vdup` from a lane
            // and the shift-by-immediate forms, none of which are modelled.
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
        // The divide, the parallel groups, the extends and the dual multiplies are all
        // decoded; what is left of the media class is `umaal` and the reserved groups.
        0b011 => "a media instruction outside the modelled families is outside the A32 slice",
        0b100 => "ldm/stm with the S bit (user bank or exception return) is outside the A32 slice",
        0b010 => "the unprivileged ldr/str forms (ldrt/strt) are outside the A32 slice",
        0b001 => "the msr immediate or hint form is outside the A32 slice",
        _ => {
            "the miscellaneous space (umaal, the unprivileged forms or a reserved encoding) is outside the A32 slice"
        }
    }
}

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
        0b011 => media(pc, word, cond),
        0b100 => block_transfer(pc, word, cond),
        0b101 => branch(pc, word, cond),
        // 0b110 coprocessor load/store and 0b111 coprocessor data operations, which
        // for coprocessors 10 and 11 are VFP.
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

/// Builds an instruction of the VFP or Advanced SIMD classes: they never set the ARM
/// flags implicitly and carry the element shape when they have one.
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

/// `ROR(imm8, 2 * rotate)` of an immediate data-processing operand.
fn rotated_immediate(word: u32) -> u32 {
    (word & 0xff).rotate_right(2 * ((word >> 8) & 0xf))
}

/// The `Rd` field of the encoding must be a real destination, never `pc`.
fn require_not_pc(reg: Reg) -> Result<Reg, DecodeError> {
    if reg == Reg::PC {
        Err(DecodeError::Unpredictable)
    } else {
        Ok(reg)
    }
}

fn unconditional(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    match (word >> 25) & 0b111 {
        // `blx` immediate: Align(PC, 4) + SignExtend(imm24) << 2 + H * 2. H contributes
        // bit 1 of the target, so the branch lands on a halfword boundary inside Thumb
        // code; the address keeps bit 0 clear and the state is implied.
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
        // The Advanced SIMD data-processing space (`0xF2`/`0xF3`, where bit 24 is the
        // `i` bit) and, with bit 24 clear, the element and structure load/store space
        // (`0xF4`).
        0b001 => neon::decode(pc, word),
        0b010 if word & 0x0100_0000 == 0 => neon::decode(pc, word),
        // `0xF5`: `pld`, the barriers and the hints.
        0b010 => hint_space(pc, word),
        // `cps`, `setend`, `pli` and the coprocessor.
        _ => Err(DecodeError::Unsupported),
    }
}

/// `1111 0101`: `pld` and the barrier and hint block.
///
/// The barriers promise an ordering this interpreter does not reorder, and the hints
/// have no architectural effect at all, so both decode to their mnemonic and execute as
/// no-ops. Decoding them keeps the report exact about what the guest ran.
fn hint_space(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    // `1111 0101 0111 1111 1111 0000 01xx xxxx`: bits 7:4 select the barrier or the
    // hint, and the low nibble is the barrier option or the hint number.
    if word & 0x0fff_ff00 == 0x057f_f000 {
        let low = word & 0xf;
        let opcode = match (word >> 4) & 0xf {
            0b0011 => match low {
                0 => Opcode::Nop,
                1 => Opcode::Yield,
                2 => Opcode::Wfe,
                3 => Opcode::Wfi,
                4 => Opcode::Sev,
                5 => Opcode::Sevl,
                _ => return Err(DecodeError::Unallocated),
            },
            0b0100 => Opcode::Dsb,
            0b0101 => Opcode::Dmb,
            0b0110 => Opcode::Isb,
            // The `clrex` option field is SBZ.
            0b0111 if low == 0 => Opcode::Clrex,
            _ => return Err(DecodeError::Unallocated),
        };
        return Ok(insn(
            pc,
            word,
            Cond::Al,
            opcode,
            false,
            vec![Operand::Imm(low)],
        ));
    }
    // `1111 0101 U101 Rn 1111 imm12`: `pld`, a prefetch hint. It has no architectural
    // effect and never faults, so the address is decoded for the report and the
    // interpreter ignores it.
    let op1 = (word >> 20) & 0x1f;
    if op1 & 0b10111 == 0b10101 && (word >> 12) & 0xf == 0xf {
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
            Opcode::Pld,
            false,
            vec![Operand::Mem(mem)],
        ));
    }
    Err(DecodeError::Unsupported)
}

/// Data-processing instructions with an immediate operand, including `movw`/`movt`,
/// `msr` immediate and the hints.
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
            // Bits 24:21 = 0b1001 with S = 0 is the `msr` immediate space, and its
            // all-zero mask/rotate encoding is the hint space. Everything else with
            // `pc` in Rd is unallocated.
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
        // Rn is SBZ in the `mov`/`mvn` encodings.
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

/// `msr` immediate (flags only) and the hints (`nop`, `yield`, `wfe`, `wfi`, `sev`),
/// which share the `0011 0010` encoding space of the immediate `teq`.
fn system_immediate(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let spsr = (word >> 22) & 1 != 0;
    let mask = (word >> 16) & 0xf;
    let rotate = (word >> 8) & 0xf;
    let imm8 = word & 0xff;
    if mask == 0 && rotate == 0 && !spsr {
        // Hints are only defined with cond = 0b1110; imm8 selects the hint.
        let opcode = match imm8 {
            0 => Opcode::Nop,
            1 => Opcode::Yield,
            2 => Opcode::Wfe,
            3 => Opcode::Wfi,
            4 => Opcode::Sev,
            _ => return Err(DecodeError::Unsupported),
        };
        if cond != Cond::Al {
            return Err(DecodeError::Unpredictable);
        }
        return Ok(insn(
            pc,
            word,
            cond,
            opcode,
            false,
            vec![Operand::Imm(imm8)],
        ));
    }
    // Only the flags byte is writable here; an empty mask is unpredictable, and the
    // SPSR and every control or status field belong to modes this interpreter does not
    // model.
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

/// Data-processing instructions with a register operand: the register-shift forms, the
/// miscellaneous space (`bx`, `blx`, `mrs`, `msr`, `qadd`, the halfword multiplies) and
/// the extra load/store, swap and exclusive forms.
fn data_processing_register(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if word & 0x0fc0_00f0 == 0x90 {
        return multiply(pc, word, cond);
    }

    // Bits 7 = 1 and 4 = 1 select the extra load/store, swap, exclusive and long
    // multiply space. It is tested before the data-processing restrictions, which do
    // not apply to it (`ldrh r0, [pc, #4]` has 0b1111 in Rn).
    if word & 0x90 == 0x90 {
        return multiply_and_transfer(pc, word, cond);
    }

    let sets_flags = (word >> 20) & 1 != 0;
    let rd = reg((word >> 12) & 0xf);
    let opcode = Opcode::from_data_processing((word >> 21) & 0xf);
    if opcode.is_flag_test() {
        // `tst`/`teq`/`cmp`/`cmn` require S = 1 and Rd = 0. The other words with these
        // opcodes are the miscellaneous space, which uses bit 4 and bit 7 to select its
        // own forms.
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

/// The miscellaneous space reachable from data-processing opcodes 0b1000-0b1011 with
/// S = 0: `bx`, `blx` (register), `bkpt`, `mrs`/`msr`, `clz`, `qadd`/`qsub`/`qdadd`/
/// `qdsub` and the halfword multiplies.
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

    let opcode_field = (word >> 21) & 0xf;
    let nibble = (word >> 4) & 0xf;
    match (opcode_field, nibble) {
        // `mrs` and `qadd` share the `tst` space; the `spsr` form of `mrs` uses the
        // `cmp` space.
        (0b1000 | 0b1010, 0b0000) => mrs(pc, word, cond),
        (0b1000, 0b0101) => qadd(pc, word, cond),
        // `msr` (register) and `qsub` share the `teq` space; `bx`/`blx`/`bkpt` above,
        // and the `spsr` form of `msr` uses the `cmn` space.
        (0b1001 | 0b1011, 0b0000) => msr_register(pc, word, cond),
        (0b1001, 0b0101) => qadd(pc, word, cond),
        // `smlal<x><y>` and `qdadd`.
        (0b1010, 0b0101) => qadd(pc, word, cond),
        (0b1010, 0b1000 | 0b1010 | 0b1100 | 0b1110) => halfword_multiply(pc, word, cond),
        // `clz`, `qdsub` and `smul<x><y>`.
        (0b1011, 0b0001) => clz(pc, word, cond),
        (0b1011, 0b0101) => qadd(pc, word, cond),
        (0b1011, 0b1000 | 0b1010 | 0b1100 | 0b1110) => halfword_multiply(pc, word, cond),
        // `smla<x><y>`.
        (0b1000, 0b1000 | 0b1010 | 0b1100 | 0b1110) => halfword_multiply(pc, word, cond),
        // `smlaw<y>` (only the bottom and top selectors are allocated).
        (0b1001, 0b1000 | 0b1100) => halfword_multiply(pc, word, cond),
        _ => Err(DecodeError::Unsupported),
    }
}

fn mrs(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    // `cond 0001 0 R 00 1111 Rd 0000 0000 0000`.
    if (word >> 16) & 0xf != 0xf || word & 0xfff != 0 {
        return Err(DecodeError::Unpredictable);
    }
    let rd = require_not_pc(reg((word >> 12) & 0xf))?;
    if (word >> 22) & 1 != 0 {
        // The SPSR belongs to the banked modes, which this runtime does not model.
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
    // `cond 0001 0 R 10 mask 1111 0000 0000 Rm`.
    if (word >> 12) & 0xf != 0xf || (word >> 4) & 0xff != 0 {
        return Err(DecodeError::Unpredictable);
    }
    let rm = require_not_pc(reg(word & 0xf))?;
    let mask = (word >> 16) & 0xf;
    if (word >> 22) & 1 != 0 || mask != 0b1000 {
        // Only the flags byte is writable here. An empty mask is unpredictable, and
        // the SPSR and every control or status field belong to modes this interpreter
        // does not model.
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

/// `clz Rd, Rm`: bits 19:16 and 11:8 are SBO.
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

/// `qadd`, `qsub`, `qdadd` and `qdsub`: `Rd = sat(Rm ± Rn)` with the doubling forms
/// saturating `2 * Rn` first. Bits 22:21 select the form.
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

/// `smla<x><y>`, `smlaw<y>`, `smlal<x><y>` and `smul<x><y>`. Bits 6:5 select the
/// halfwords: bit 6 is `<y>` (top half of `Rs`) and bit 5 is `<x>` (top half of `Rm`).
fn halfword_multiply(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let x = (word >> 5) & 1 != 0;
    let y = (word >> 6) & 1 != 0;
    let rd = reg((word >> 16) & 0xf);
    let ra = reg((word >> 12) & 0xf);
    let rs = reg((word >> 8) & 0xf);
    let rm = reg(word & 0xf);
    if rs == Reg::PC || rm == Reg::PC || rd == Reg::PC {
        return Err(DecodeError::Unpredictable);
    }
    let opcode = match ((word >> 21) & 0xf, x, y) {
        (0b1000, false, false) => Opcode::Smlabb,
        (0b1000, false, true) => Opcode::Smlabt,
        (0b1000, true, false) => Opcode::Smlatb,
        (0b1000, true, true) => Opcode::Smlatt,
        (0b1001, _, false) => Opcode::Smlawb,
        (0b1001, _, true) => Opcode::Smlawt,
        (0b1010, false, false) => Opcode::Smlalbb,
        (0b1010, false, true) => Opcode::Smlalbt,
        (0b1010, true, false) => Opcode::Smlaltb,
        (0b1010, true, true) => Opcode::Smlaltt,
        (0b1011, false, false) => Opcode::Smulbb,
        (0b1011, false, true) => Opcode::Smulbt,
        (0b1011, true, false) => Opcode::Smultb,
        (0b1011, true, true) => Opcode::Smultt,
        _ => return Err(DecodeError::Unallocated),
    };
    let operands = match opcode {
        // `smlal<x><y>` accumulates into the 64-bit pair RdHi:RdLo, which the encoding
        // names in reverse order: RdHi in bits 19:16, RdLo in bits 15:12.
        Opcode::Smlalbb | Opcode::Smlalbt | Opcode::Smlaltb | Opcode::Smlaltt => {
            if ra == Reg::PC || rd == ra {
                return Err(DecodeError::Unpredictable);
            }
            vec![
                Operand::Regs { lo: ra, hi: rd },
                Operand::Reg(rm),
                Operand::Reg(rs),
            ]
        }
        Opcode::Smulbb | Opcode::Smulbt | Opcode::Smultb | Opcode::Smultt => {
            // Ra is SBZ in the `smul<x><y>` encodings.
            if ra != Reg::R0 {
                return Err(DecodeError::Unpredictable);
            }
            vec![Operand::Reg(rd), Operand::Reg(rm), Operand::Reg(rs)]
        }
        _ => {
            if ra == Reg::PC {
                return Err(DecodeError::Unpredictable);
            }
            vec![
                Operand::Reg(rd),
                Operand::Reg(rm),
                Operand::Reg(rs),
                Operand::Reg(ra),
            ]
        }
    };
    Ok(insn(pc, word, cond, opcode, false, operands))
}

/// The classes that live behind bits 7 = 1 and 4 = 1 in class 0b000: multiply long,
/// swap, the exclusive accesses and the extra load/store forms.
fn multiply_and_transfer(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let nibble = (word >> 4) & 0xf;
    let group = (word >> 23) & 0x1f;
    if nibble == 0b1001 {
        // Bits 6:5 = 0b00: a multiply form or `swap`.
        return match group {
            0b00001 => long_multiply(pc, word, cond),
            0b00010 => Err(DecodeError::Deprecated),
            // `umaal` (0b00000 with bit 22) is outside the slice.
            0b00011 => exclusive(pc, word, cond),
            _ => Err(DecodeError::Unsupported),
        };
    }
    // Any other combination of bits 6:5 selects the extra load/store forms.
    extra_transfer(pc, word, cond)
}

/// `umull`, `umlal`, `smull` and `smlal`: bits 22 and 21 select signedness and
/// accumulation, bit 20 the flag update.
fn long_multiply(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let signed = (word >> 22) & 1 != 0;
    let accumulate = (word >> 21) & 1 != 0;
    let sets_flags = (word >> 20) & 1 != 0;
    let rd_hi = reg((word >> 16) & 0xf);
    let rd_lo = reg((word >> 12) & 0xf);
    let rs = reg((word >> 8) & 0xf);
    let rm = reg(word & 0xf);
    if rd_hi == Reg::PC || rd_lo == Reg::PC || rs == Reg::PC || rm == Reg::PC || rd_hi == rd_lo {
        return Err(DecodeError::Unpredictable);
    }
    let opcode = match (signed, accumulate) {
        (false, false) => Opcode::Umull,
        (false, true) => Opcode::Umlal,
        (true, false) => Opcode::Smull,
        (true, true) => Opcode::Smlal,
    };
    Ok(insn(
        pc,
        word,
        cond,
        opcode,
        sets_flags,
        vec![
            Operand::Regs {
                lo: rd_lo,
                hi: rd_hi,
            },
            Operand::Reg(rm),
            Operand::Reg(rs),
        ],
    ))
}

/// `ldrex`, `strex` and their byte and halfword forms. The base carries no offset: the
/// encoding's bit 11:8 and 3:0 fields are fixed.
/// The exclusive accesses and the ARMv8 acquire/release forms, which share this space:
/// bits 9:8 select the family (`11` `ldrex`/`strex`, `10` `ldaex`/`stlex`, `00` the
/// plain `lda`/`ldab`/`ldah`). The ordering those promise is not modelled, because the
/// interpreter does not reorder the guest's own accesses.
fn exclusive(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let exclusive = match (word >> 8) & 0xf {
        0b1111 | 0b1110 => true,
        0b1100 => false,
        _ => return Err(DecodeError::Unpredictable),
    };
    let load = (word >> 20) & 1 != 0;
    let byte = (word >> 22) & 1 != 0;
    let half = (word >> 21) & 1 != 0;
    let base = require_not_pc(reg((word >> 16) & 0xf))?;
    let size = match (byte, half) {
        (false, false) => MemSize::Word,
        (true, false) => MemSize::Byte,
        (true, true) => MemSize::Halfword,
        (false, true) => return Err(DecodeError::Unallocated),
    };
    let mem = Mem {
        base,
        offset: Offset::Imm(0),
        add: true,
        pre_indexed: true,
        writeback: false,
        size,
    };
    if !exclusive {
        // The plain forms name the data register in bits 3:0 and leave bits 15:12 as
        // an SBZ field for the status register the exclusive form would use.
        if load {
            if word & 0xf != 0xf {
                return Err(DecodeError::Unpredictable);
            }
            let rt = require_not_pc(reg((word >> 12) & 0xf))?;
            let opcode = match size {
                MemSize::Word => Opcode::Lda,
                MemSize::Byte => Opcode::Ldab,
                _ => Opcode::Ldah,
            };
            return Ok(insn(
                pc,
                word,
                cond,
                opcode,
                false,
                vec![Operand::Reg(rt), Operand::Mem(mem)],
            ));
        }
        if (word >> 12) & 0xf != 0xf {
            return Err(DecodeError::Unpredictable);
        }
        let rt = require_not_pc(reg(word & 0xf))?;
        let opcode = match size {
            MemSize::Word => Opcode::Stl,
            MemSize::Byte => Opcode::Stlb,
            _ => Opcode::Stlh,
        };
        return Ok(insn(
            pc,
            word,
            cond,
            opcode,
            false,
            vec![Operand::Reg(rt), Operand::Mem(mem)],
        ));
    }
    let (status, rt) = if load {
        if word & 0xf != 0xf {
            return Err(DecodeError::Unpredictable);
        }
        (None, require_not_pc(reg((word >> 12) & 0xf))?)
    } else {
        let status = require_not_pc(reg((word >> 12) & 0xf))?;
        let rt = require_not_pc(reg(word & 0xf))?;
        if status == rt {
            return Err(DecodeError::Unpredictable);
        }
        (Some(status), rt)
    };
    let opcode = match (load, size) {
        (true, MemSize::Word) => Opcode::Ldrex,
        (true, MemSize::Byte) => Opcode::Ldrexb,
        (true, _) => Opcode::Ldrexh,
        (false, MemSize::Word) => Opcode::Strex,
        (false, MemSize::Byte) => Opcode::Strexb,
        (false, _) => Opcode::Strexh,
    };
    let operands = match status {
        Some(status) => vec![Operand::Reg(status), Operand::Reg(rt), Operand::Mem(mem)],
        None => vec![Operand::Reg(rt), Operand::Mem(mem)],
    };
    Ok(insn(pc, word, cond, opcode, false, operands))
}

/// `ldrh`/`strh`, `ldrsb`/`ldrsh` and the `ldrd`/`strd` pair.
fn extra_transfer(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let pre_indexed = (word >> 24) & 1 != 0;
    let add = (word >> 23) & 1 != 0;
    let immediate = (word >> 22) & 1 != 0;
    let writeback = (word >> 21) & 1 != 0;
    let load = (word >> 20) & 1 != 0;
    let signed = (word >> 6) & 1 != 0;
    let half = (word >> 5) & 1 != 0;
    let base = reg((word >> 16) & 0xf);
    let rt = reg((word >> 12) & 0xf);
    let double = signed && !load;

    if !pre_indexed && writeback {
        // The unprivileged `ldrht`/`strht` forms; `ldrd`/`strd` have no such form.
        return Err(if double {
            DecodeError::Unpredictable
        } else {
            DecodeError::Unsupported
        });
    }
    if (!pre_indexed || writeback) && base == Reg::PC {
        return Err(DecodeError::Unpredictable);
    }
    let offset = if immediate {
        Offset::Imm((((word >> 8) & 0xf) << 4) as u16 | (word & 0xf) as u16)
    } else {
        if (word >> 8) & 0xf != 0 {
            // The register form has 0b0000 in bits 11:8.
            return Err(DecodeError::Unpredictable);
        }
        let index = reg(word & 0xf);
        if index == Reg::PC {
            return Err(DecodeError::Unpredictable);
        }
        Offset::Reg {
            reg: index,
            shift: Shift::Lsl(0),
        }
    };
    let size = match (signed, half) {
        (false, true) => MemSize::Halfword,
        (true, false) if load => MemSize::Byte,
        (true, false) => MemSize::Double,
        (true, true) if load => MemSize::Halfword,
        (true, true) => MemSize::Double,
        (false, false) => return Err(DecodeError::Unallocated),
    };
    let mem = Mem {
        base,
        offset,
        add,
        pre_indexed,
        writeback,
        size,
    };

    if double {
        // `Rt` and `Rt + 1`; the encoding cannot name `pc` or `lr` as the low half.
        if rt.number() >= 14 {
            return Err(DecodeError::Unpredictable);
        }
        let rt_hi = Reg::new(rt.number() + 1).ok_or(DecodeError::Unpredictable)?;
        if mem.writes_back() && (base == rt || base == rt_hi) {
            return Err(DecodeError::Unpredictable);
        }
        // With (S, H) = (1, 0) the L bit is clear for `ldrd`; with (1, 1) it is clear
        // for `strd`, so the pair as a whole is selected by H, not by L.
        return Ok(insn(
            pc,
            word,
            cond,
            if half { Opcode::Strd } else { Opcode::Ldrd },
            false,
            vec![Operand::Regs { lo: rt, hi: rt_hi }, Operand::Mem(mem)],
        ));
    }

    if rt == Reg::PC {
        return Err(DecodeError::Unpredictable);
    }
    if mem.writes_back() && base == rt {
        return Err(DecodeError::Unpredictable);
    }
    let opcode = match (signed, half, load) {
        (false, true, true) => Opcode::Ldrh,
        (false, true, false) => Opcode::Strh,
        (true, false, true) => Opcode::Ldrsb,
        (true, true, true) => Opcode::Ldrsh,
        _ => return Err(DecodeError::Unallocated),
    };
    Ok(insn(
        pc,
        word,
        cond,
        opcode,
        false,
        vec![Operand::Reg(rt), Operand::Mem(mem)],
    ))
}

/// Media instructions: the parallel add/subtract family, packing, extension, reversal,
/// saturation and the bitfield operations.
fn media(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let nibble = (word >> 4) & 0xf;
    if nibble & 1 == 0 {
        // Bit 4 selects the media class.
        return Err(DecodeError::Unallocated);
    }
    let group = (word >> 20) & 0x1f;

    // The integer divide instructions share the media class: `op1` is `10011` for
    // `udiv` and `10001` for `sdiv`, bits 15:12 are SBO and bits 7:4 are `0001`.
    if matches!(group, 0b10001 | 0b10011) {
        return divide(pc, word, cond, group == 0b10011);
    }

    // The dual multiplies and their multiply-accumulate forms: `0111 0000` for the
    // 32-bit ones, `0111 0100` for the 64-bit accumulators and `0111 0101` for the
    // top-word multiplies.
    if matches!(group, 0b10000 | 0b10100 | 0b10101) {
        return media_multiply(pc, word, cond, group);
    }

    if let Some(opcode) = Opcode::parallel(group, nibble) {
        // Bits 11:8 are SBO in every parallel form.
        if (word >> 8) & 0xf != 0xf {
            return Err(DecodeError::Unpredictable);
        }
        let rd = require_not_pc(reg((word >> 12) & 0xf))?;
        let rn = require_not_pc(reg((word >> 16) & 0xf))?;
        let rm = require_not_pc(reg(word & 0xf))?;
        return Ok(insn(
            pc,
            word,
            cond,
            opcode,
            false,
            vec![Operand::Reg(rd), Operand::Reg(rn), Operand::Reg(rm)],
        ));
    }

    if matches!(
        group,
        0b00001 | 0b00010 | 0b00011 | 0b00101 | 0b00110 | 0b00111
    ) {
        // A parallel group with a selector other than the six allocated ones.
        return Err(DecodeError::Unallocated);
    }

    match group {
        0b01000 => pack_and_select(pc, word, cond),
        0b01010 | 0b01011 => saturate_or_extend(pc, word, cond, ExtendFamily::Signed),
        0b01100 => saturate_or_extend(pc, word, cond, ExtendFamily::Uxtb16),
        0b01110 | 0b01111 => saturate_or_extend(pc, word, cond, ExtendFamily::Unsigned),
        0b11010..=0b11111 => bitfield(pc, word, cond),
        _ => Err(DecodeError::Unsupported),
    }
}

/// `udiv`/`sdiv`: `cond 0111 00x1 Rd 1111 Rm 0001 Rn`. The 32-bit divide is a standard
/// instruction of the architecture from ARMv7-A with the Divide extension on, which the
/// Switch core has; it reads `Rn` as the dividend and `Rm` as the divisor and leaves the
/// flags and Q untouched.
fn divide(pc: u32, word: u32, cond: Cond, unsigned: bool) -> Result<Insn, DecodeError> {
    if (word >> 12) & 0xf != 0xf || (word >> 4) & 0xf != 0b0001 {
        return Err(DecodeError::Unpredictable);
    }
    let rd = require_not_pc(reg((word >> 16) & 0xf))?;
    let rn = require_not_pc(reg(word & 0xf))?;
    let rm = require_not_pc(reg((word >> 8) & 0xf))?;
    Ok(insn(
        pc,
        word,
        cond,
        if unsigned { Opcode::Udiv } else { Opcode::Sdiv },
        false,
        vec![Operand::Reg(rd), Operand::Reg(rn), Operand::Reg(rm)],
    ))
}

/// The media dual multiplies: `smuad`, `smlad`, `smlald`, `smmul` and their exchange,
/// rounding and subtract forms.
///
/// `0111 0000` holds the 32-bit dual forms (with `Ra = 1111` selecting the plain
/// multiplies), `0111 0100` the 64-bit accumulating pair and `0111 0101` the top-word
/// multiplies. In all three, bits 7:6 pick the operation and bit 5 the exchange or
/// rounding modifier.
fn media_multiply(pc: u32, word: u32, cond: Cond, group: u32) -> Result<Insn, DecodeError> {
    let form = (word >> 6) & 3;
    let modifier = (word >> 5) & 1 != 0;
    let rd = reg((word >> 16) & 0xf);
    let ra = reg((word >> 12) & 0xf);
    let rm = require_not_pc(reg((word >> 8) & 0xf))?;
    let rn = require_not_pc(reg(word & 0xf))?;
    match (group, form) {
        // The plain multiplies name no accumulator.
        (0b10000, 0 | 1) if ra == Reg::PC => {
            let opcode = match (form, modifier) {
                (0, false) => Opcode::Smuad,
                (0, true) => Opcode::Smuadx,
                (_, false) => Opcode::Smusd,
                (_, true) => Opcode::Smusdx,
            };
            Ok(insn(
                pc,
                word,
                cond,
                opcode,
                false,
                vec![
                    Operand::Reg(require_not_pc(rd)?),
                    Operand::Reg(rn),
                    Operand::Reg(rm),
                ],
            ))
        }
        (0b10000, 0 | 1) => {
            let opcode = match (form, modifier) {
                (0, false) => Opcode::Smlad,
                (0, true) => Opcode::Smladx,
                (_, false) => Opcode::Smlsd,
                (_, true) => Opcode::Smlsdx,
            };
            Ok(insn(
                pc,
                word,
                cond,
                opcode,
                false,
                vec![
                    Operand::Reg(require_not_pc(rd)?),
                    Operand::Reg(rn),
                    Operand::Reg(rm),
                    Operand::Reg(ra),
                ],
            ))
        }
        // The 64-bit accumulators name `RdLo` in bits 19:16 and `RdHi` in 15:12.
        (0b10100, 0 | 1) => {
            if rd == Reg::PC || ra == Reg::PC || rd == ra {
                return Err(DecodeError::Unpredictable);
            }
            let opcode = match (form, modifier) {
                (0, false) => Opcode::Smlald,
                (0, true) => Opcode::Smlaldx,
                (_, false) => Opcode::Smlsld,
                (_, true) => Opcode::Smlsldx,
            };
            Ok(insn(
                pc,
                word,
                cond,
                opcode,
                false,
                vec![
                    Operand::Regs { lo: rd, hi: ra },
                    Operand::Reg(rn),
                    Operand::Reg(rm),
                ],
            ))
        }
        // `smmul` leaves the accumulator field as SBZ; `smmla` and `smmls` use it.
        (0b10101, 0) if ra == Reg::PC => Ok(insn(
            pc,
            word,
            cond,
            if modifier {
                Opcode::Smmulr
            } else {
                Opcode::Smmul
            },
            false,
            vec![
                Operand::Reg(require_not_pc(rd)?),
                Operand::Reg(rn),
                Operand::Reg(rm),
            ],
        )),
        (0b10101, 0) => Ok(insn(
            pc,
            word,
            cond,
            if modifier {
                Opcode::Smmlar
            } else {
                Opcode::Smmla
            },
            false,
            vec![
                Operand::Reg(require_not_pc(rd)?),
                Operand::Reg(rn),
                Operand::Reg(rm),
                Operand::Reg(ra),
            ],
        )),
        (0b10101, 3) if ra != Reg::PC => Ok(insn(
            pc,
            word,
            cond,
            if modifier {
                Opcode::Smmlsr
            } else {
                Opcode::Smmls
            },
            false,
            vec![
                Operand::Reg(require_not_pc(rd)?),
                Operand::Reg(rn),
                Operand::Reg(rm),
                Operand::Reg(ra),
            ],
        )),
        _ => Err(DecodeError::Unallocated),
    }
}

/// The `0100 0` media group: `pkhbt`, `pkhtb`, `sxtb16`, `sxtab16` and `sel`.
fn pack_and_select(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let rd = require_not_pc(reg((word >> 12) & 0xf))?;
    let rn = reg((word >> 16) & 0xf);
    let rm = require_not_pc(reg(word & 0xf))?;
    let imm5 = (word >> 7) & 0x1f;
    match (word >> 4) & 0xf {
        0b0001 | 0b1001 => Ok(insn(
            pc,
            word,
            cond,
            Opcode::Pkhbt,
            false,
            vec![
                Operand::Reg(rd),
                Operand::Reg(require_not_pc(rn)?),
                Operand::Shifted {
                    reg: rm,
                    shift: Shift::Lsl(imm5 as u8),
                },
            ],
        )),
        0b0101 | 0b1101 => Ok(insn(
            pc,
            word,
            cond,
            Opcode::Pkhtb,
            false,
            vec![
                Operand::Reg(rd),
                Operand::Reg(require_not_pc(rn)?),
                Operand::Shifted {
                    reg: rm,
                    shift: Shift::Asr(if imm5 == 0 { 32 } else { imm5 as u8 }),
                },
            ],
        )),
        0b0111 => {
            let rotate = ((word >> 10) & 3) * 8;
            if (word >> 8) & 3 != 0 {
                return Err(DecodeError::Unpredictable);
            }
            let plain = rn == Reg::PC;
            let shift = if rotate == 0 {
                Shift::Lsl(0)
            } else {
                Shift::Ror(rotate as u8)
            };
            let opcode = if plain {
                Opcode::Sxtb16
            } else {
                Opcode::Sxtab16
            };
            let mut operands = vec![Operand::Reg(rd)];
            if !plain {
                operands.push(Operand::Reg(rn));
            }
            operands.push(Operand::Shifted { reg: rm, shift });
            Ok(insn(pc, word, cond, opcode, false, operands))
        }
        0b1011 => {
            // `sel`: bits 11:8 are SBO.
            if (word >> 8) & 0xf != 0xf {
                return Err(DecodeError::Unpredictable);
            }
            Ok(insn(
                pc,
                word,
                cond,
                Opcode::Sel,
                false,
                vec![
                    Operand::Reg(rd),
                    Operand::Reg(require_not_pc(rn)?),
                    Operand::Reg(rm),
                ],
            ))
        }
        _ => Err(DecodeError::Unallocated),
    }
}

/// Which of the three media families a `0101 x`/`0110 x`/`0111 x` group belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtendFamily {
    /// `0b01010`/`0b01011`: `ssat`, `ssat16`, `sxtb`, `sxth`, `rev`, `rev16`.
    Signed,
    /// `0b01100`: `uxtb16`/`uxtab16`.
    Uxtb16,
    /// `0b01110`/`0b01111`: `usat`, `usat16`, `uxtb`, `uxth`, `rbit`, `revsh`.
    Unsigned,
}

/// The `0101 x`, `0110 0` and `0111 x` groups: `ssat`/`usat` (with or without a
/// shift), `ssat16`/`usat16`, the extension instructions and the byte reversals.
fn saturate_or_extend(
    pc: u32,
    word: u32,
    cond: Cond,
    family: ExtendFamily,
) -> Result<Insn, DecodeError> {
    let nibble = (word >> 4) & 0xf;
    let low_bit = (word >> 20) & 1 != 0;
    let rd = require_not_pc(reg((word >> 12) & 0xf))?;
    let rm = require_not_pc(reg(word & 0xf))?;

    // `ssat`/`usat`: bits 5:4 = 0b01, bits 20:16 are the saturation width and bit 6
    // selects an arithmetic shift. The shift amount is a signed 5-bit field.
    if nibble & 0b0011 == 0b0001 {
        // Bits 20:16 are the saturation width minus one for `ssat` and the width
        // itself for `usat`; the IR carries the width, which is what the assembly
        // syntax prints.
        let sat_imm = (word >> 16) & 0x1f;
        let width = match family {
            ExtendFamily::Signed => sat_imm + 1,
            _ => sat_imm,
        };
        let imm5 = (word >> 7) & 0x1f;
        let shift = if (word >> 6) & 1 != 0 {
            Shift::Asr(if imm5 == 0 { 32 } else { imm5 as u8 })
        } else {
            Shift::Lsl(imm5 as u8)
        };
        let opcode = match family {
            ExtendFamily::Signed => Opcode::Ssat,
            ExtendFamily::Unsigned => Opcode::Usat,
            ExtendFamily::Uxtb16 => return Err(DecodeError::Unallocated),
        };
        return Ok(insn(
            pc,
            word,
            cond,
            opcode,
            false,
            vec![
                Operand::Reg(rd),
                Operand::Imm(width),
                Operand::Shifted { reg: rm, shift },
            ],
        ));
    }

    // Bits 11:8 are SBO for the 16-bit saturating forms and for the reversals.
    let rn = (word >> 16) & 0xf;
    match nibble {
        0b0011 => {
            if (word >> 8) & 0xf != 0xf {
                return Err(DecodeError::Unpredictable);
            }
            let opcode = match (family, low_bit) {
                (ExtendFamily::Signed, false) => Opcode::Ssat16,
                (ExtendFamily::Signed, true) => Opcode::Rev,
                (ExtendFamily::Unsigned, false) => Opcode::Usat16,
                (ExtendFamily::Unsigned, true) => Opcode::Rbit,
                (ExtendFamily::Uxtb16, _) => return Err(DecodeError::Unallocated),
            };
            let operands = match opcode {
                Opcode::Rev | Opcode::Rbit => {
                    if rn != 0xf {
                        return Err(DecodeError::Unpredictable);
                    }
                    vec![Operand::Reg(rd), Operand::Reg(rm)]
                }
                Opcode::Ssat16 => vec![Operand::Reg(rd), Operand::Imm(rn + 1), Operand::Reg(rm)],
                _ => vec![Operand::Reg(rd), Operand::Imm(rn), Operand::Reg(rm)],
            };
            Ok(insn(pc, word, cond, opcode, false, operands))
        }
        0b0111 => {
            let rotate = ((word >> 10) & 3) * 8;
            if (word >> 8) & 3 != 0 {
                return Err(DecodeError::Unpredictable);
            }
            let plain = rn == 0xf;
            let shift = if rotate == 0 {
                Shift::Lsl(0)
            } else {
                Shift::Ror(rotate as u8)
            };
            let opcode = match (family, low_bit, plain) {
                (ExtendFamily::Signed, false, true) => Opcode::Sxtb,
                (ExtendFamily::Signed, false, false) => Opcode::Sxtab,
                (ExtendFamily::Signed, true, true) => Opcode::Sxth,
                (ExtendFamily::Signed, true, false) => Opcode::Sxtah,
                (ExtendFamily::Unsigned, false, true) => Opcode::Uxtb,
                (ExtendFamily::Unsigned, false, false) => Opcode::Uxtab,
                (ExtendFamily::Unsigned, true, true) => Opcode::Uxth,
                (ExtendFamily::Unsigned, true, false) => Opcode::Uxtah,
                (ExtendFamily::Uxtb16, _, true) => Opcode::Uxtb16,
                (ExtendFamily::Uxtb16, _, false) => Opcode::Uxtab16,
            };
            let mut operands = vec![Operand::Reg(rd)];
            if !plain {
                operands.push(Operand::Reg(require_not_pc(reg(rn))?));
            }
            operands.push(Operand::Shifted { reg: rm, shift });
            Ok(insn(pc, word, cond, opcode, false, operands))
        }
        0b1011 => {
            if (word >> 8) & 0xf != 0xf || rn != 0xf {
                return Err(DecodeError::Unpredictable);
            }
            let opcode = match (family, low_bit) {
                (ExtendFamily::Signed, true) => Opcode::Rev16,
                (ExtendFamily::Unsigned, true) => Opcode::Revsh,
                _ => return Err(DecodeError::Unallocated),
            };
            Ok(insn(
                pc,
                word,
                cond,
                opcode,
                false,
                vec![Operand::Reg(rd), Operand::Reg(rm)],
            ))
        }
        _ => Err(DecodeError::Unallocated),
    }
}

/// `bfc`/`bfi` (`op1 = 1110x`) and `sbfx`/`ubfx` (`op1 = 1101x`/`1111x`). Bits 11:7
/// hold the least significant bit; bits 20:16 hold the most significant bit for the
/// inserting forms and the width minus one for the extracting ones.
fn bitfield(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let group = (word >> 20) & 0x1f;
    let field = (word >> 16) & 0x1f;
    let lsb = (word >> 7) & 0x1f;
    let rn = reg(word & 0xf);
    let rd = reg((word >> 12) & 0xf);
    let insert = matches!(group, 0b11100 | 0b11101);
    // Bits 6:4 select the form; bit 7 is part of the least significant bit.
    if (word >> 4) & 0b111 != if insert { 0b001 } else { 0b101 } {
        return Err(DecodeError::Unallocated);
    }
    // `bfi`/`bfc` encode the most significant bit; `sbfx`/`ubfx` encode the width
    // minus one.
    let width = if insert {
        if field < lsb {
            // `width = msb - lsb + 1` would not be a legal bitfield.
            return Err(DecodeError::Unpredictable);
        }
        field - lsb + 1
    } else {
        if lsb + field + 1 > 32 {
            return Err(DecodeError::Unpredictable);
        }
        field + 1
    };
    let rd = require_not_pc(rd)?;
    if insert {
        // `bfc` is `bfi` with the source field set to 0b1111.
        if rn == Reg::PC {
            return Ok(insn(
                pc,
                word,
                cond,
                Opcode::Bfc,
                false,
                vec![Operand::Reg(rd), Operand::Imm(lsb), Operand::Imm(width)],
            ));
        }
        return Ok(insn(
            pc,
            word,
            cond,
            Opcode::Bfi,
            false,
            vec![
                Operand::Reg(rd),
                Operand::Reg(require_not_pc(rn)?),
                Operand::Imm(lsb),
                Operand::Imm(width),
            ],
        ));
    }
    let opcode = if group & 0b00100 != 0 {
        Opcode::Ubfx
    } else {
        Opcode::Sbfx
    };
    Ok(insn(
        pc,
        word,
        cond,
        opcode,
        false,
        vec![
            Operand::Reg(rd),
            Operand::Reg(require_not_pc(rn)?),
            Operand::Imm(lsb),
            Operand::Imm(width),
        ],
    ))
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

    #[test]
    fn rotated_immediates_rotate_right() {
        assert_eq!(rotated_immediate(0xE3A0_00FF), 0xff);
        assert_eq!(rotated_immediate(0xE3A0_04FF), 0xff00_0000);
    }
}
