use super::decode::{DecodeError, vinsn};
use super::ir::{
    BlockMode, Cond, Convert, DReg, Insn, Mem, MemSize, Offset, Operand, Reg, SReg, VBlock,
    VecShape, VfpDest, VfpSys,
};
use super::opcode::Opcode;

fn reg(bits: u32) -> Reg {
    Reg::from_encoding(bits)
}

fn s(field: u32, high: bool) -> SReg {
    SReg::from_parts(field as u8, high)
}

pub(super) fn d(field: u32, high: bool) -> DReg {
    DReg::from_parts(field as u8, high)
}

fn precision(word: u32) -> Option<bool> {
    match (word >> 8) & 0xf {
        0b1010 => Some(false),
        0b1011 => Some(true),
        _ => None,
    }
}

pub(super) fn decode(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    match (word >> 24) & 0xf {
        0xc | 0xd => transfer(pc, word, cond),
        0xe => {
            if word & 0x10 == 0 {
                data_processing(pc, word, cond)
            } else {
                register_transfer(pc, word, cond)
            }
        }
        _ => Err(DecodeError::Unsupported),
    }
}

fn mrrc(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let opc = (word >> 4) & 0xf;
    let crm = word & 0xf;
    let sys = match (crm, opc) {
        (14, 0) => super::ir::A32SysReg::Cntpct,
        (14, 1) => super::ir::A32SysReg::Cntvct,
        _ => return Err(DecodeError::Unsupported),
    };
    let rt2 = reg((word >> 16) & 0xf);
    let rt = reg((word >> 12) & 0xf);
    if rt == Reg::PC || rt2 == Reg::PC || rt == rt2 {
        return Err(DecodeError::Unpredictable);
    }
    Ok(vinsn(
        pc,
        word,
        cond,
        Opcode::Mrrc,
        None,
        vec![Operand::Regs { lo: rt, hi: rt2 }, Operand::SysReg(sys)],
    ))
}

fn system_register(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let opc1 = (word >> 21) & 0b111;
    let read = (word >> 20) & 1 != 0;
    let crn = (word >> 16) & 0xf;
    let rt = reg((word >> 12) & 0xf);
    let opc2 = (word >> 5) & 0b111;
    let crm = word & 0xf;
    let sys = match (opc1, crn, crm, opc2) {
        (0, 13, 0, 2) => super::ir::A32SysReg::Tpidrurw,
        (0, 13, 0, 3) => super::ir::A32SysReg::Tpidruro,
        _ => return Err(DecodeError::Unsupported),
    };
    if rt == Reg::PC {
        return Err(DecodeError::Unpredictable);
    }
    Ok(vinsn(
        pc,
        word,
        cond,
        if read { Opcode::Mrc } else { Opcode::Mcr },
        None,
        vec![Operand::Reg(rt), Operand::SysReg(sys)],
    ))
}

fn transfer(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if (word >> 8) & 0xf == 0b1111 {
        return if (word >> 21) & 0b111 == 0b010 && (word >> 20) & 1 == 1 {
            mrrc(pc, word, cond)
        } else {
            Err(DecodeError::Unsupported)
        };
    }
    let Some(double) = precision(word) else {
        return Err(DecodeError::Unsupported);
    };
    let pre = (word >> 24) & 1 != 0;
    let add = (word >> 23) & 1 != 0;
    let high = (word >> 22) & 1 != 0;
    let writeback = (word >> 21) & 1 != 0;
    let load = (word >> 20) & 1 != 0;
    let base = reg((word >> 16) & 0xf);
    let vd = (word >> 12) & 0xf;
    let imm8 = word & 0xff;

    match (pre, add, writeback) {
        (false, false, false) => core_transfer(pc, word, cond, double),
        (false, true, _) => vblock(pc, word, cond, double, BlockMode::IncrementAfter),
        (true, false, true) => vblock(pc, word, cond, double, BlockMode::DecrementBefore),
        (true, true, true) => Err(DecodeError::Unallocated),
        (false, false, true) => Err(DecodeError::Unallocated),
        _ => {
            let mem = Mem {
                base,
                offset: Offset::Imm((imm8 * 4) as u16),
                add,
                pre_indexed: true,
                writeback: false,
                size: if double {
                    MemSize::Double
                } else {
                    MemSize::Word
                },
            };
            let register = if double {
                Operand::D(d(vd, high))
            } else {
                Operand::S(s(vd, high))
            };
            Ok(vinsn(
                pc,
                word,
                cond,
                if load { Opcode::Vldr } else { Opcode::Vstr },
                None,
                vec![register, Operand::Mem(mem)],
            ))
        }
    }
}

fn core_transfer(pc: u32, word: u32, cond: Cond, double: bool) -> Result<Insn, DecodeError> {
    if !double || (word >> 21) & 0b111 != 0b010 || (word >> 4) & 0b1101 != 0b0001 {
        return Err(DecodeError::Unsupported);
    }
    let to_core = (word >> 20) & 1 != 0;
    let rt2 = reg((word >> 16) & 0xf);
    let rt = reg((word >> 12) & 0xf);
    if rt == Reg::PC || rt2 == Reg::PC || rt == rt2 {
        return Err(DecodeError::Unpredictable);
    }
    let dm = d(word & 0xf, (word >> 5) & 1 != 0);
    let operands = if to_core {
        vec![Operand::Reg(rt), Operand::Reg(rt2), Operand::D(dm)]
    } else {
        vec![Operand::D(dm), Operand::Reg(rt), Operand::Reg(rt2)]
    };
    Ok(vinsn(pc, word, cond, Opcode::Vmov, None, operands))
}

fn vblock(
    pc: u32,
    word: u32,
    cond: Cond,
    double: bool,
    mode: BlockMode,
) -> Result<Insn, DecodeError> {
    let load = (word >> 20) & 1 != 0;
    let base = reg((word >> 16) & 0xf);
    let (field, high) = ((word >> 12) & 0xf, (word >> 22) & 1 != 0);
    let first = if double { d(field, high).number() } else { s(field, high).number() };
    let writeback = (word >> 21) & 1 != 0;
    let words = (word & 0xff) as u8;
    if words == 0 {
        return Err(DecodeError::Unpredictable);
    }
    if double && !words.is_multiple_of(2) {
        return Err(DecodeError::Unpredictable);
    }
    let end = if double {
        u16::from(first) + u16::from(words / 2)
    } else {
        u16::from(first) + u16::from(words)
    };
    if end > 32 {
        return Err(DecodeError::Unpredictable);
    }
    if base == Reg::PC {
        return Err(DecodeError::Unpredictable);
    }
    let stack = base == Reg::SP
        && writeback
        && matches!((mode, load), (BlockMode::DecrementBefore, false) | (BlockMode::IncrementAfter, true));
    let block = VBlock {
        first,
        words,
        double,
        base,
        mode,
        writeback,
        implicit_base: stack,
    };
    let opcode = match (stack, mode, load) {
        (true, BlockMode::DecrementBefore, false) => Opcode::Vpush,
        (true, BlockMode::IncrementAfter, true) => Opcode::Vpop,
        (_, _, true) => Opcode::Vldm,
        (_, _, false) => Opcode::Vstm,
    };
    Ok(vinsn(
        pc,
        word,
        cond,
        opcode,
        None,
        vec![Operand::VBlock(block)],
    ))
}

fn data_processing(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    if (word >> 9) & 0b111 != 0b101 {
        return Err(DecodeError::Unsupported);
    }
    let Some(double) = precision(word) else {
        return Err(DecodeError::Unsupported);
    };
    let dbit = (word >> 22) & 1 != 0;
    let nbit = (word >> 7) & 1 != 0;
    let mbit = (word >> 5) & 1 != 0;
    let vd = (word >> 12) & 0xf;
    let vn = (word >> 16) & 0xf;
    let vm = word & 0xf;
    let bit6 = (word >> 6) & 1 != 0;
    let op = (((word >> 23) & 1) << 2) | (((word >> 21) & 1) << 1) | ((word >> 20) & 1);

    if op == 0b111 {
        return other(pc, word, cond, double, dbit, mbit);
    }
    let (dst, first, second) = if double {
        (
            Operand::D(d(vd, dbit)),
            Operand::D(d(vn, nbit)),
            Operand::D(d(vm, mbit)),
        )
    } else {
        (
            Operand::S(s(vd, dbit)),
            Operand::S(s(vn, nbit)),
            Operand::S(s(vm, mbit)),
        )
    };
    let opcode = match (op, bit6) {
        (0b000, false) => Opcode::Vmla,
        (0b000, true) => Opcode::Vmls,
        (0b001, false) => Opcode::Vnmls,
        (0b001, true) => Opcode::Vnmla,
        (0b010, false) => Opcode::Vmul,
        (0b010, true) => Opcode::Vnmul,
        (0b011, false) => Opcode::Vadd,
        (0b011, true) => Opcode::Vsub,
        (0b100, false) => Opcode::Vdiv,
        (0b101, false) => Opcode::Vfnms,
        (0b101, true) => Opcode::Vfnma,
        (0b110, false) => Opcode::Vfma,
        (0b110, true) => Opcode::Vfms,
        _ => return Err(DecodeError::Unallocated),
    };
    Ok(vinsn(
        pc,
        word,
        cond,
        opcode,
        None,
        vec![dst, first, second],
    ))
}

fn other(
    pc: u32,
    word: u32,
    cond: Cond,
    double: bool,
    dbit: bool,
    mbit: bool,
) -> Result<Insn, DecodeError> {
    let vd = (word >> 12) & 0xf;
    let vn = (word >> 16) & 0xf;
    let vm = word & 0xf;
    let bit7 = (word >> 7) & 1 != 0;
    let bit6 = (word >> 6) & 1 != 0;
    let dst = if double {
        Operand::D(d(vd, dbit))
    } else {
        Operand::S(s(vd, dbit))
    };
    let source = if double {
        Operand::D(d(vm, mbit))
    } else {
        Operand::S(s(vm, mbit))
    };
    let single = |field: u32, high: bool| Operand::S(s(field, high));

    if !bit6 {
        if bit7 || (word >> 4) & 0xf != 0 {
            return Err(DecodeError::Unallocated);
        }
        let imm8 = (vn << 4) | vm;
        let bits = super::super::simd::fp_expand_imm8(imm8, if double { 64 } else { 32 });
        let value = if double {
            Operand::ImmFloat64(bits)
        } else {
            Operand::ImmFloat32(bits as u32)
        };
        return Ok(vinsn(pc, word, cond, Opcode::Vmov, None, vec![dst, value]));
    }

    let (opcode, operands) = match (vn, bit7) {
        (0, false) => (Opcode::Vmov, vec![dst, source]),
        (0, true) => (Opcode::Vabs, vec![dst, source]),
        (1, false) => (Opcode::Vneg, vec![dst, source]),
        (1, true) => (Opcode::Vsqrt, vec![dst, source]),
        (4, false) => (Opcode::Vcmp, vec![dst, source]),
        (4, true) => (Opcode::Vcmpe, vec![dst, source]),
        (5, _) => {
            if vm != 0 || mbit {
                return Err(DecodeError::Unpredictable);
            }
            let opcode = if bit7 { Opcode::Vcmpe } else { Opcode::Vcmp };
            let zero = if double {
                Operand::ImmFloat64(0)
            } else {
                Operand::ImmFloat32(0)
            };
            (opcode, vec![dst, zero])
        }
        (6, false) => (Opcode::Vrintr, vec![dst, source]),
        (6, true) => (Opcode::Vrintz, vec![dst, source]),
        (7, false) => (Opcode::Vrintx, vec![dst, source]),
        (7, true) => {
            if double {
                (
                    Opcode::Vcvt,
                    vec![
                        single(vd, dbit),
                        Operand::D(d(vm, mbit)),
                        Operand::Convert(Convert::F32FromF64),
                    ],
                )
            } else {
                (
                    Opcode::Vcvt,
                    vec![
                        Operand::D(d(vd, dbit)),
                        single(vm, mbit),
                        Operand::Convert(Convert::F64FromF32),
                    ],
                )
            }
        }
        (8, _) => {
            let (convert, dest) = match (double, bit7) {
                (false, false) => (Convert::F32FromU32, single(vd, dbit)),
                (false, true) => (Convert::F32FromS32, single(vd, dbit)),
                (true, false) => (Convert::F64FromU32, Operand::D(d(vd, dbit))),
                (true, true) => (Convert::F64FromS32, Operand::D(d(vd, dbit))),
            };
            (
                Opcode::Vcvt,
                vec![dest, single(vm, mbit), Operand::Convert(convert)],
            )
        }
        (0xc | 0xd, _) => {
            let signed = vn == 0xd;
            let (convert, src) = match (double, signed) {
                (false, false) => (Convert::U32FromF32, single(vm, mbit)),
                (false, true) => (Convert::S32FromF32, single(vm, mbit)),
                (true, false) => (Convert::U32FromF64, Operand::D(d(vm, mbit))),
                (true, true) => (Convert::S32FromF64, Operand::D(d(vm, mbit))),
            };
            let opcode = if bit7 { Opcode::Vcvt } else { Opcode::VcvtR };
            (
                opcode,
                vec![single(vd, dbit), src, Operand::Convert(convert)],
            )
        }
        (0xe, _) => return Err(DecodeError::Unsupported),
        _ => return Err(DecodeError::Unallocated),
    };
    Ok(vinsn(pc, word, cond, opcode, None, operands))
}

fn register_transfer(pc: u32, word: u32, cond: Cond) -> Result<Insn, DecodeError> {
    let op = (word >> 20) & 1 != 0;
    let cp = (word >> 8) & 0xf;
    let rt = reg((word >> 12) & 0xf);
    let vn = (word >> 16) & 0xf;
    if (word >> 21) & 0b111 == 0b111 {
        let field = (word >> 16) & 0xf;
        let Some(sys) = VfpSys::from_field(field) else {
            return Err(DecodeError::Unsupported);
        };
        if cp != 0b1010 {
            return element_transfer(pc, word, cond, rt, vn);
        }
        if word & 0xef != 0 {
            return Err(DecodeError::Unpredictable);
        }
        if op {
            let dest = if rt == Reg::PC {
                VfpDest::Apsr
            } else {
                VfpDest::Core(rt)
            };
            return Ok(vinsn(
                pc,
                word,
                cond,
                Opcode::Vmrs,
                None,
                vec![Operand::VfpDest(dest), Operand::VfpSys(sys)],
            ));
        }
        if rt == Reg::PC {
            return Err(DecodeError::Unpredictable);
        }
        return Ok(vinsn(
            pc,
            word,
            cond,
            Opcode::Vmsr,
            None,
            vec![Operand::VfpSys(sys), Operand::Reg(rt)],
        ));
    }
    match cp {
        0b1010 => {
            if (word >> 21) & 0b111 != 0 || word & 0x6f != 0 {
                return Err(DecodeError::Unpredictable);
            }
            let sd = s(vn, (word >> 7) & 1 != 0);
            let operands = if op {
                vec![Operand::Reg(rt), Operand::S(sd)]
            } else {
                vec![Operand::S(sd), Operand::Reg(rt)]
            };
            Ok(vinsn(pc, word, cond, Opcode::Vmov, None, operands))
        }
        0b1011 => element_transfer(pc, word, cond, rt, vn),
        0b1111 => system_register(pc, word, cond),
        _ => Err(DecodeError::Unsupported),
    }
}

fn element_transfer(pc: u32, word: u32, cond: Cond, rt: Reg, vn: u32) -> Result<Insn, DecodeError> {
    let to_core = (word >> 20) & 1 != 0;
    let bit23 = (word >> 23) & 1 != 0;
    let bit22 = (word >> 22) & 1 != 0;
    let bit21 = (word >> 21) & 1 != 0;
    let bit7 = (word >> 7) & 1 != 0;
    let bit6 = (word >> 6) & 1 != 0;
    let bit5 = (word >> 5) & 1 != 0;
    if word & 0xf != 0 {
        return Err(DecodeError::Unpredictable);
    }
    let dm = DReg::from_parts(vn as u8, bit7);
    let (esize, lane) = if bit22 {
        (
            8,
            (u32::from(bit21) << 2) | (u32::from(bit6) << 1) | u32::from(bit5),
        )
    } else if bit5 {
        (16, (u32::from(bit21) << 1) | u32::from(bit6))
    } else {
        if bit6 {
            return Err(DecodeError::Unallocated);
        }
        (32, u32::from(bit21))
    };

    if !to_core {
        if bit23 {
            if bit6 || bit5 != (esize == 16) {
                return Err(DecodeError::Unpredictable);
            }
            if rt == Reg::PC || (bit21 && dm.number() & 1 != 0) {
                return Err(DecodeError::Unpredictable);
            }
            return Ok(vinsn(
                pc,
                word,
                cond,
                Opcode::Vdup,
                Some(VecShape { esize, q: bit21 }),
                vec![Operand::D(dm), Operand::Reg(rt)],
            ));
        }
        if rt == Reg::PC {
            return Err(DecodeError::Unpredictable);
        }
        return Ok(vinsn(
            pc,
            word,
            cond,
            Opcode::VmovLane,
            Some(VecShape { esize, q: false }),
            vec![
                Operand::Lane {
                    reg: dm,
                    index: lane as u8,
                },
                Operand::Reg(rt),
            ],
        ));
    }

    if esize == 32 && bit23 {
        return Err(DecodeError::Unallocated);
    }
    if rt == Reg::PC {
        return Err(DecodeError::Unpredictable);
    }
    let opcode = match (esize, bit23) {
        (32, _) => Opcode::VmovLane,
        (_, false) => Opcode::VmovLaneS,
        (_, true) => Opcode::VmovLaneU,
    };
    Ok(vinsn(
        pc,
        word,
        cond,
        opcode,
        Some(VecShape { esize, q: false }),
        vec![
            Operand::Reg(rt),
            Operand::Lane {
                reg: dm,
                index: lane as u8,
            },
        ],
    ))
}
