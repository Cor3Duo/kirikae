use super::*;

pub(super) fn media(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let nibble = (word >> 4) & 0xf;
    if nibble & 1 == 0 {
        return Err(DecodeError::Unallocated);
    }
    let group = (word >> 20) & 0x1f;

    if matches!(group, 0b10001 | 0b10011) {
        return divide(pc, word, cond, group == 0b10011);
    }

    if matches!(group, 0b10000 | 0b10100 | 0b10101) {
        return media_multiply(pc, word, cond, group);
    }

    if let Some(opcode) = Opcode::parallel(group, nibble) {
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

fn media_multiply(pc: u32, word: u32, cond: Cond, group: u32) -> Result<Insn, DecodeError> {
    let form = (word >> 6) & 3;
    let modifier = (word >> 5) & 1 != 0;
    let rd = reg((word >> 16) & 0xf);
    let ra = reg((word >> 12) & 0xf);
    let rm = require_not_pc(reg((word >> 8) & 0xf))?;
    let rn = require_not_pc(reg(word & 0xf))?;
    match (group, form) {
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtendFamily {
    Signed,
    Uxtb16,
    Unsigned,
}

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

    if nibble & 0b0011 == 0b0001 {
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

fn bitfield(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let group = (word >> 20) & 0x1f;
    let field = (word >> 16) & 0x1f;
    let lsb = (word >> 7) & 0x1f;
    let rn = reg(word & 0xf);
    let rd = reg((word >> 12) & 0xf);
    let insert = matches!(group, 0b11100 | 0b11101);
    if (word >> 4) & 0b111 != if insert { 0b001 } else { 0b101 } {
        return Err(DecodeError::Unallocated);
    }
    let width = if insert {
        if field < lsb {
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
