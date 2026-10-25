use super::*;

pub(super) fn exclusive(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
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

pub(super) fn extra_transfer(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
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
        if rt.number() >= 14 {
            return Err(DecodeError::Unpredictable);
        }
        let rt_hi = Reg::new(rt.number() + 1).ok_or(DecodeError::Unpredictable)?;
        if mem.writes_back() && (base == rt || base == rt_hi) {
            return Err(DecodeError::Unpredictable);
        }
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

    fn effective_writeback(&self) -> bool {
        !self.pre_indexed || self.writeback
    }
}

fn check_transfer(transfer: &Transfer, base: Reg, rt: Reg) -> Result<(), DecodeError> {
    if !transfer.pre_indexed && transfer.writeback {
        return Err(DecodeError::Unsupported);
    }
    if transfer.effective_writeback() && (base == Reg::PC || base == rt) {
        return Err(DecodeError::Unpredictable);
    }
    if rt == Reg::PC && transfer.size() == MemSize::Byte {
        return Err(DecodeError::Unpredictable);
    }
    Ok(())
}

pub(super) fn transfer_immediate(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
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

pub(super) fn transfer_register(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
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

pub(super) fn block_transfer(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let pre_indexed = (word >> 24) & 1 != 0;
    let add = (word >> 23) & 1 != 0;
    let user_bank = (word >> 22) & 1 != 0;
    let writeback = (word >> 21) & 1 != 0;
    let load = (word >> 20) & 1 != 0;
    if user_bank {
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
