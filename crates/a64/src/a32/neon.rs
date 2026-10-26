use super::decode::{DecodeError, vinsn};
use super::ir::{Cond, Insn, NeonMem, Operand, PostUpdate, Reg, VecShape};
use super::opcode::Opcode;
use super::vfp::d as double;

pub(super) fn decode(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    match (word >> 25) & 0b111 {
        0b001 => data_processing(pc, word),
        0b010 => structure(pc, word),
        _ => Err(DecodeError::Unsupported),
    }
}

fn reg(bits: u32) -> Reg {
    Reg::from_encoding(bits)
}

fn data_processing(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    if word & 0x0fb0_0000 == 0x03b0_0000 {
        return two_register(pc, word);
    }
    if word & 0x0fb0_0010 == 0x02b0_0000 {
        return extract(pc, word);
    }
    if word & 0x0eb8_0090 == 0x0280_0010 {
        return modified_immediate(pc, word);
    }
    if word & 0x0080_0000 == 0 {
        return three_same(pc, word);
    }
    Err(DecodeError::Unsupported)
}

fn extract(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    let q = word & 0x40 != 0;
    let index = (word >> 8) & 0xf;
    if !q && index & 0b1000 != 0 {
        return Err(DecodeError::Unallocated);
    }
    let vd = double((word >> 12) & 0xf, word & 0x0040_0000 != 0);
    let vn = double((word >> 16) & 0xf, word & 0x80 != 0);
    let vm = double(word & 0xf, word & 0x20 != 0);
    if q && (vd.number() & 1 != 0 || vn.number() & 1 != 0 || vm.number() & 1 != 0) {
        return Err(DecodeError::Unpredictable);
    }
    Ok(vinsn(
        pc,
        word,
        Cond::Al,
        Opcode::Vext,
        Some(VecShape { esize: 8, q }),
        vec![
            Operand::D(vd),
            Operand::D(vn),
            Operand::D(vm),
            Operand::Imm(index),
        ],
    ))
}

fn three_same(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    let u = word & 0x0100_0000 != 0;
    let size = (word >> 20) & 3;
    let vd = (word >> 12) & 0xf;
    let vn = (word >> 16) & 0xf;
    let vm = word & 0xf;
    let nbit = word & 0x80 != 0;
    let mbit = word & 0x20 != 0;
    let q = word & 0x40 != 0;
    let bit4 = word & 0x10 != 0;
    let opc1 = (word >> 8) & 0xf;
    let esize = 8u8 << size;

    let (opcode, style) = match (bit4, u, size, opc1) {
        (true, false, 0, 0b0001) => (Opcode::Vand, 8),
        (true, false, 1, 0b0001) => (Opcode::Vbic, 8),
        (true, false, 2, 0b0001) => (Opcode::Vorr, 8),
        (true, false, 3, 0b0001) => (Opcode::Vorn, 8),
        (true, true, 0, 0b0001) => (Opcode::Veor, 8),
        (true, true, 1, 0b0001) => (Opcode::Vbsl, 8),
        (false, false, _, 0b1000) => (Opcode::Vadd, esize),
        (false, true, _, 0b1000) => (Opcode::Vsub, esize),
        (false, false, 0, 0b1101) => (Opcode::VaddF, 32),
        (false, false, 2, 0b1101) => (Opcode::VsubF, 32),
        (false, false, _, 0b1001) => (Opcode::VmlaI, esize),
        (false, true, _, 0b1001) => (Opcode::VmlsI, esize),
        (true, false, _, 0b1001) => (Opcode::Vmul, esize),
        (true, true, 0, 0b1101) => (Opcode::VmulF, 32),
        (true, false, 0, 0b1100) => (Opcode::Vfma, 32),
        (true, false, 2, 0b1100) => (Opcode::Vfms, 32),
        (true, false, 0, 0b1101) => (Opcode::VmlaF, 32),
        (true, false, 2, 0b1101) => (Opcode::VmlsF, 32),
        (false, false, _, 0b0110) => (Opcode::Vmax, esize),
        (false, true, _, 0b0110) => (Opcode::VmaxU, esize),
        (true, false, _, 0b0110) => (Opcode::Vmin, esize),
        (true, true, _, 0b0110) => (Opcode::VminU, esize),
        (false, false, 0, 0b1111) => (Opcode::VmaxF, 32),
        (false, false, 2, 0b1111) => (Opcode::VminF, 32),
        (false, false, _, 0b0011) => (Opcode::Vcgt, esize),
        (false, true, _, 0b0011) => (Opcode::VcgtU, esize),
        (true, false, _, 0b0011) => (Opcode::Vcge, esize),
        (true, true, _, 0b0011) => (Opcode::VcgeU, esize),
        (false, false, 0, 0b1110) => (Opcode::VceqF, 32),
        (false, true, 0, 0b1110) => (Opcode::VcgeF, 32),
        (true, false, _, 0b1000) => (Opcode::Vtst, esize),
        (true, true, _, 0b1000) => (Opcode::Vceq, esize),
        _ => return Err(DecodeError::Unsupported),
    };
    if size == 3 && !matches!(opcode, Opcode::Vadd | Opcode::Vsub) && style != 8 {
        return Err(DecodeError::Unallocated);
    }
    let dest = double(vd, word & 0x0040_0000 != 0);
    let first = double(vn, nbit);
    let second = double(vm, mbit);
    if q && (dest.number() & 1 != 0 || first.number() & 1 != 0 || second.number() & 1 != 0) {
        return Err(DecodeError::Unpredictable);
    }
    Ok(vinsn(
        pc,
        word,
        Cond::Al,
        opcode,
        Some(VecShape { esize: style, q }),
        vec![Operand::D(dest), Operand::D(first), Operand::D(second)],
    ))
}

fn two_register(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    let vd = (word >> 12) & 0xf;
    let vm = word & 0xf;
    let mbit = word & 0x20 != 0;
    let q = word & 0x40 != 0;
    if word & 0x0810 != 0 {
        return Err(DecodeError::Unsupported);
    }
    let size = (word >> 18) & 3;
    let opc1 = (word >> 16) & 3;
    let opc2 = (word >> 7) & 0xf;
    let esize = 8u8 << size;
    let (opcode, style) = match (opc1, opc2, size) {
        (0, 0b0000, _) => (Opcode::Vrev64, esize),
        (0, 0b0001, _) => (Opcode::Vrev32, esize),
        (0, 0b0010, _) => (Opcode::Vrev16, esize),
        (0, 0b1011, 0) => (Opcode::Vmvn, 8),
        (1, 0b0110, _) => (Opcode::Vabs, esize),
        (1, 0b0111, _) => (Opcode::Vneg, esize),
        (1, 0b1110, 2) => (Opcode::VabsF, 32),
        (1, 0b1111, 2) => (Opcode::VnegF, 32),
        _ => return Err(DecodeError::Unsupported),
    };
    let undefined = match opcode {
        Opcode::Vrev64 | Opcode::Vrev32 | Opcode::Vrev16 => opc2 + size >= 3,
        Opcode::Vabs | Opcode::Vneg => size == 3,
        _ => false,
    };
    if undefined {
        return Err(DecodeError::Unallocated);
    }
    let dest = double(vd, word & 0x0040_0000 != 0);
    let source = double(vm, mbit);
    if q && (dest.number() & 1 != 0 || source.number() & 1 != 0) {
        return Err(DecodeError::Unpredictable);
    }
    Ok(vinsn(
        pc,
        word,
        Cond::Al,
        opcode,
        Some(VecShape { esize: style, q }),
        vec![Operand::D(dest), Operand::D(source)],
    ))
}

fn modified_immediate(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    let vd = (word >> 12) & 0xf;
    let q = word & 0x40 != 0;
    let cmode = (word >> 8) & 0xf;
    let complement = word & 0x20 != 0;
    let imm8 = (((word >> 16) & 0xf) << 4)
        | (word & 0xf)
        | (if word & 0x0100_0000 != 0 { 0x80 } else { 0 });
    if cmode == 0b1111 || (cmode == 0b1110 && complement) {
        return Err(DecodeError::Unsupported);
    }
    let (esize, value) = match cmode & 0b1110 {
        0b0000 => (32u8, u64::from(imm8)),
        0b0010 => (32, u64::from(imm8) << 8),
        0b0100 => (32, u64::from(imm8) << 16),
        0b0110 => (32, u64::from(imm8) << 24),
        0b1000 => (16, u64::from(imm8)),
        0b1010 => (16, u64::from(imm8) << 8),
        0b1100 if cmode & 1 == 0 => (32, (u64::from(imm8) << 8) | 0xff),
        0b1100 => (32, (u64::from(imm8) << 16) | 0xffff),
        _ => (8, u64::from(imm8)),
    };
    let logical = cmode & 1 == 1 && cmode < 0b1100;
    let opcode = match (logical, complement) {
        (false, false) => Opcode::VmovImm,
        (false, true) => Opcode::VmvnImm,
        (true, false) => Opcode::VorrImm,
        (true, true) => Opcode::VbicImm,
    };
    let mut expanded = 0u64;
    let mut shift = 0u32;
    while shift < 64 {
        expanded |= value << shift;
        shift += u32::from(esize);
    }
    let dest = double(vd, word & 0x0040_0000 != 0);
    if q && dest.number() & 1 != 0 {
        return Err(DecodeError::Unpredictable);
    }
    Ok(vinsn(
        pc,
        word,
        Cond::Al,
        opcode,
        Some(VecShape { esize, q }),
        vec![Operand::D(dest), Operand::Imm64(expanded)],
    ))
}

struct Structure {
    count: u8,
    stride: u8,
    interleave: u8,
}

fn structure_type(kind: u32) -> Option<Structure> {
    let one = |count, stride| {
        Some(Structure {
            count,
            stride,
            interleave: 1,
        })
    };
    let many = |count, stride, streams| {
        Some(Structure {
            count,
            stride,
            interleave: streams,
        })
    };
    match kind {
        0b0000 => many(4, 1, 4),
        0b0001 => many(4, 2, 4),
        0b0010 => one(4, 1),
        0b0011 => many(4, 1, 2),
        0b0100 => many(3, 1, 3),
        0b0101 => many(3, 2, 3),
        0b0110 => one(3, 1),
        0b0111 => one(1, 1),
        0b1000 => many(2, 1, 2),
        0b1001 => many(2, 2, 2),
        0b1010 => one(2, 1),
        _ => None,
    }
}

fn structure(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    if word & 0x0080_0000 != 0 {
        return Err(DecodeError::Unsupported);
    }
    let load = word & 0x0020_0000 != 0;
    if word & 0x0010_0000 != 0 {
        return Err(DecodeError::Unallocated);
    }
    let kind = (word >> 8) & 0xf;
    let Some(structure) = structure_type(kind) else {
        return Err(DecodeError::Unsupported);
    };
    let size = (word >> 6) & 3;
    let alignment = (word >> 4) & 3;
    let undefined = match kind {
        0b0111 | 0b0110 | 0b0100 | 0b0101 => alignment & 0b10 != 0,
        0b1010 | 0b1000 | 0b1001 => alignment == 0b11,
        _ => false,
    };
    if undefined {
        return Err(DecodeError::Unallocated);
    }
    let esize = match (size, structure.interleave) {
        (0, _) => 8,
        (1, _) => 16,
        (2, _) => 32,
        (_, 1) => 64,
        _ => return Err(DecodeError::Unallocated),
    };
    let align = (alignment != 0).then(|| (esize / 8) << alignment);
    let base = reg((word >> 16) & 0xf);
    let first = double((word >> 12) & 0xf, word & 0x0040_0000 != 0);
    let last =
        u16::from(first.number()) + u16::from(structure.count - 1) * u16::from(structure.stride);
    if base == Reg::PC || last >= 32 {
        return Err(DecodeError::Unpredictable);
    }
    let update = match word & 0xf {
        0b1111 => PostUpdate::None,
        0b1101 => PostUpdate::Immediate,
        _ => {
            let offset = reg(word & 0xf);
            if offset == Reg::PC {
                return Err(DecodeError::Unpredictable);
            }
            PostUpdate::Register(offset)
        }
    };
    let mem = NeonMem {
        base,
        first,
        count: structure.count,
        stride: structure.stride,
        esize,
        interleave: structure.interleave,
        align,
        update,
    };
    let opcode = match (load, structure.interleave) {
        (true, 1) => Opcode::Vld1,
        (false, 1) => Opcode::Vst1,
        (true, 2) => Opcode::Vld2,
        (false, 2) => Opcode::Vst2,
        (true, 3) => Opcode::Vld3,
        (false, 3) => Opcode::Vst3,
        (true, _) => Opcode::Vld4,
        (false, _) => Opcode::Vst4,
    };
    Ok(vinsn(
        pc,
        word,
        Cond::Al,
        opcode,
        Some(VecShape { esize, q: false }),
        vec![Operand::NeonMem(mem)],
    ))
}
