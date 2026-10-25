use super::*;

pub(super) fn halfword_multiply(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
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

pub(super) fn multiply_and_transfer(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let nibble = (word >> 4) & 0xf;
    let group = (word >> 23) & 0x1f;
    if nibble == 0b1001 {
        return match group {
            0b00001 => long_multiply(pc, word, cond),
            0b00010 => Err(DecodeError::Deprecated),
            0b00011 => exclusive(pc, word, cond),
            _ => Err(DecodeError::Unsupported),
        };
    }
    extra_transfer(pc, word, cond)
}

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

pub(super) fn multiply(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
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
        return Err(DecodeError::Unpredictable);
    }
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
