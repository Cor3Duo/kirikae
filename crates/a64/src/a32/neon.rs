//! Advanced SIMD (NEON) decoding for the A32 slice.
//!
//! NEON lives in the unconditional space (`cond = 0b1111`): bits 27:25 = `001` is the
//! data-processing space (`0xF2`/`0xF3`) and `010` is the load/store space (`0xF4`).
//! The slice covers the structure transfers, the three-register arithmetic and logic
//! groups, the two-register absolute/negate/reverse group, `vext` and the modified
//! immediates. Everything else is rejected with a [`DecodeError`].
//!
//! # Register fields
//!
//! A `d` register is `D:Vd` and a `q` register is the pair that starts at that `d`; for
//! the 128-bit forms the low bit of the register field is zero, so `q(n)` is named by
//! `d(2n)`. `N` (bit 7) and `M` (bit 5) extend `Vn` and `Vm` the same way `D` extends
//! `Vd`.

use super::decode::{DecodeError, vinsn};
use super::ir::{Cond, Insn, NeonMem, Operand, PostUpdate, Reg, VecShape};
use super::opcode::Opcode;
use super::vfp::d as double;

/// Decodes the two Advanced SIMD spaces.
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

/// The three-register and two-register data-processing groups.
fn data_processing(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    // `1111 001i 1D11 ...`: the two-register group.
    if word & 0x0fb0_0000 == 0x03b0_0000 {
        return two_register(pc, word);
    }
    // `1111 0010 1D11 ...`: `vext`.
    if word & 0x0fb0_0000 == 0x02b0_0000 {
        if word & 0x10 != 0 {
            return Err(DecodeError::Unallocated);
        }
        return extract(pc, word);
    }
    // `1111 001i 1D00 0imm3 ... 1 imm4`: modified immediate.
    if word & 0x0eb0_0010 == 0x0280_0010 {
        return modified_immediate(pc, word);
    }
    // `1111 001U 0Dsize ...`: three registers of the same length.
    if word & 0x0080_0000 == 0 {
        return three_same(pc, word);
    }
    Err(DecodeError::Unsupported)
}

/// `vext`: bytes taken from the concatenation of `Vn` and `Vm`.
fn extract(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    let q = word & 0x40 != 0;
    let index = (word >> 8) & 0xf;
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

/// `1111 001U 0Dsize Vn Vd opc1 N Q M x Vm`: the three-register groups.
///
/// Bit 4 splits the table in two halves; the operation is selected by `U`, the element
/// size and `opc1`. The bitwise mnemonics use the size field as their own selector,
/// which is why they carry no element suffix and a fixed element size here.
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
        // Logic: the size field selects the operation, not the element width.
        (true, false, 0, 0b0001) => (Opcode::Vand, 8),
        (true, false, 1, 0b0001) => (Opcode::Vbic, 8),
        (true, false, 2, 0b0001) => (Opcode::Vorr, 8),
        (true, false, 3, 0b0001) => (Opcode::Vorn, 8),
        (true, true, 0, 0b0001) => (Opcode::Veor, 8),
        (true, true, 1, 0b0001) => (Opcode::Vbsl, 8),
        // Integer and float addition and subtraction.
        (false, false, _, 0b1000) => (Opcode::Vadd, esize),
        (false, true, _, 0b1000) => (Opcode::Vsub, esize),
        (false, false, 0, 0b1101) => (Opcode::VaddF, 32),
        (false, false, 2, 0b1101) => (Opcode::VsubF, 32),
        // Integer multiply and multiply-accumulate, and the float multipliers, which
        // share opc1; bit 4 and U select the form.
        (false, false, _, 0b1001) => (Opcode::VmlaI, esize),
        (false, true, _, 0b1001) => (Opcode::VmlsI, esize),
        (true, false, _, 0b1001) => (Opcode::Vmul, esize),
        (true, true, 0, 0b1101) => (Opcode::VmulF, 32),
        (true, false, 0, 0b1100) => (Opcode::Vfma, 32),
        (true, false, 2, 0b1100) => (Opcode::Vfms, 32),
        (true, false, 0, 0b1101) => (Opcode::VmlaF, 32),
        (true, false, 2, 0b1101) => (Opcode::VmlsF, 32),
        // Maximum and minimum, signed and unsigned, plus the float forms. Bit 4
        // selects between the two.
        (false, false, _, 0b0110) => (Opcode::Vmax, esize),
        (false, true, _, 0b0110) => (Opcode::VmaxU, esize),
        (true, false, _, 0b0110) => (Opcode::Vmin, esize),
        (true, true, _, 0b0110) => (Opcode::VminU, esize),
        (false, false, 0, 0b1111) => (Opcode::VmaxF, 32),
        (false, false, 2, 0b1111) => (Opcode::VminF, 32),
        // Comparisons.
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
    // A quadword names an even `d` register.
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

/// `1111 0011 1D11 size Vd opc(5) Q M 0 Vm`: the two-register group.
///
/// Only the absolute value, the negation, the reversed-element forms and `vmvn` are
/// modelled; the table lookups and the `vdup` lane form are rejected.
fn two_register(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    let vd = (word >> 12) & 0xf;
    let vm = word & 0xf;
    let mbit = word & 0x20 != 0;
    let q = word & 0x40 != 0;
    let size = (word >> 18) & 3;
    let opc = (word >> 7) & 0x1f;
    let u = word & 0x0001_0000 != 0;
    let esize = 8u8 << size;
    let (opcode, style) = match (u, size, opc) {
        (false, _, 0b00000) => (Opcode::Vrev64, esize),
        (false, _, 0b00001) => (Opcode::Vrev32, esize),
        (false, _, 0b00010) => (Opcode::Vrev16, esize),
        (false, _, 0b01011) => (Opcode::Vmvn, 8),
        (true, _, 0b00110) => (Opcode::Vabs, esize),
        (false, _, 0b00110) => (Opcode::VabsU, esize),
        (true, _, 0b00111) => (Opcode::Vneg, esize),
        (false, _, 0b00111) => (Opcode::VnegU, esize),
        (true, 2, 0b01110) => (Opcode::VabsF, 32),
        (true, 2, 0b01111) => (Opcode::VnegF, 32),
        (true, 2, 0b11111) => (Opcode::VsqrtF, 32),
        _ => return Err(DecodeError::Unsupported),
    };
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

/// The modified immediate: `vmov`, `vorr`, `vmvn` and `vbic` with a value expanded from
/// eight bits by `cmode`. The float and 64-bit encodings are not modelled.
fn modified_immediate(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    let vd = (word >> 12) & 0xf;
    let q = word & 0x40 != 0;
    let cmode = (word >> 8) & 0xf;
    let complement = word & 0x20 != 0;
    let imm8 = (((word >> 16) & 0xf) << 4)
        | (word & 0xf)
        | (if word & 0x0100_0000 != 0 { 0x80 } else { 0 });
    if cmode == 0b1111 || (cmode == 0b1110 && complement) {
        // A float immediate, and the 64-bit form of `vmov`.
        return Err(DecodeError::Unsupported);
    }
    let (esize, value) = match cmode & 0b1110 {
        0b0000 => (32u8, u64::from(imm8)),
        0b0010 => (32, u64::from(imm8) << 8),
        0b0100 => (32, u64::from(imm8) << 16),
        0b0110 => (32, u64::from(imm8) << 24),
        0b1000 => (16, u64::from(imm8)),
        0b1010 => (16, u64::from(imm8) << 8),
        0b1100 => (32, (u64::from(imm8) << 8) | 0xff),
        _ => (8, u64::from(imm8)),
    };
    let opcode = match (cmode & 1 == 1, complement) {
        (false, false) => Opcode::VmovImm,
        (false, true) => Opcode::VmvnImm,
        (true, false) => Opcode::VorrImm,
        (true, true) => Opcode::VbicImm,
    };
    // The encoded value fills the lane and repeats over the register.
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

/// Register count, register stride and interleaved stream count of a structure
/// transfer, from the `type` field.
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

/// `1111 0100 0D10 Rn Vd type size align Rm` (load) and the same with bit 21 clear
/// (store).
///
/// The block transfer (`0D00 … 1011`) and the single-lane forms are rejected: the
/// first is the VFP `vldm`/`vstm` encoding, which `vfp.rs` already decodes, and the
/// second names an element index in the register list.
fn structure(pc: u32, word: u32) -> Result<Insn, DecodeError> {
    if word & 0x0080_0000 != 0 {
        return Err(DecodeError::Unsupported);
    }
    let load = word & 0x0020_0000 != 0;
    if !load && word & 0x0010_0000 != 0 {
        return Err(DecodeError::Unallocated);
    }
    let kind = (word >> 8) & 0xf;
    let Some(structure) = structure_type(kind) else {
        return Err(DecodeError::Unsupported);
    };
    let size = (word >> 6) & 3;
    let esize = match (size, structure.interleave) {
        (0, _) => 8,
        (1, _) => 16,
        (2, _) => 32,
        // 64-bit elements only exist for the non-interleaved forms.
        (_, 1) => 64,
        _ => return Err(DecodeError::Unallocated),
    };
    let alignment = (word >> 4) & 3;
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
