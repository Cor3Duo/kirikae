//! Native lifting for the integer NEON forms that `lift_fp` does not cover yet.
//!
//! `lift_fp` already lifts the bitwise three-same group (`AND`/`BIC`/`ORR`/`ORN`/
//! `EOR`/`BSL`/`BIT`/`BIF`), the wrapping `ADD`/`SUB`/`ADDP` three-same group and the
//! 32/64-bit lane `DUP`/`INS`, so those stay there. This module adds, bit-exact with
//! the runtime helper in `crate::simd`:
//!
//! - shift by immediate, vector form (`immh != 0`), lanes of 8/16/32/64 bits:
//!   `SHL` (`opcode` = `0b01010`, `U` = 0), `USHR` (`0b00000`, `U` = 1) and
//!   `SSHR` (`0b00000`, `U` = 0), with `immhb` inside `[esize, 2*esize)` and no
//!   64-bit lane into a `D` register (`Q` = 0);
//! - `DUP` (element to vector) and `INS` (element) with 8- and 16-bit lanes, the two
//!   `copy` forms that do not move data through general registers;
//! - the remaining three-same integer arithmetic: `MUL` (`opcode` = `0b10011`,
//!   `U` = 0, 8/16/32-bit lanes), `SMAX`/`UMAX`/`SMIN`/`UMIN` (`0b01100`/
//!   `0b01101`, `U` picks signed or unsigned, 8/16/32-bit lanes), `MLA`/`MLS`
//!   (`0b10010`, `U` picks the subtract form) and `SMAXP`/`UMAXP`/`SMINP`/`UMINP`
//!   (`0b10100`/`0b10101`, `U` picks unsigned, 8/16/32-bit lanes: the helper does not
//!   implement the 64-bit encoding);
//! - the saturating three-same forms `SQADD`/`SQSUB` (`0b00001`/`0b00101`, `U` picks
//!   unsigned, 8/16/32-bit lanes and 64-bit lanes with `Q` = 1). Saturation also sets
//!   the cumulative `QC` bit of `FPSR`, which the lifting writes explicitly;
//! - `ABS`/`NEG` from the two-register miscellaneous group (`opcode` = `0b01011`,
//!   `U` = 0/1) and the saturating `SQABS`/`SQNEG` (`0b00111`, `U` = 0/1), 8/16/32-bit
//!   lanes and 64-bit lanes with `Q` = 1.
//!
//! Everything else (`SRI`/`SLI`, the polynomial `PMUL`, the 64-bit `MUL`/`MAX`/`MIN`
//! encodings the architecture does not define, the 64-bit `MAXP`/`MINP` the helper does
//! not implement, the scalar `Q` = 1 forms, `DUP`/`INS` from or to a general register,
//! 64-bit `DUP` into a `D` register, `SMOV`/`UMOV`, ...) keeps going through
//! `Stmt::Simd`, exactly as before.
//!
//! Every lane width used here divides 64, so a 128-bit register can be treated as two
//! independent `u64` halves: the IR stores V registers as two halves (`Expr::V`,
//! `Stmt::SetV`) and these operations are lane-wise `u64` arithmetic. No lane ever
//! crosses the half boundary. When `Q = 0` the architecture zeroes the upper half,
//! which the lowering writes explicitly.

use crate::ir::*;
use crate::ops::Op;

fn bits(w: u32, lo: u32, n: u32) -> u32 {
    (w >> lo) & ((1 << n) - 1)
}

fn mask(esize: u32) -> u64 {
    if esize >= 64 {
        u64::MAX
    } else {
        (1 << esize) - 1
    }
}

fn v(reg: u32, hi: bool) -> Expr {
    if hi { vhi(reg) } else { vlo(reg) }
}

pub fn lift(w: u32) -> Option<Vec<Stmt>> {
    if w & 0x9f80_0400 == 0x0f00_0400 {
        return shift_imm(w);
    }
    if w & 0x9fe0_8400 == 0x0e00_0400 {
        return copy(w);
    }
    if w & 0x9f20_0400 == 0x0e20_0400 {
        return three_same(w);
    }
    if w & 0x9f3e_0c00 == 0x0e20_0800 {
        return two_reg_misc(w);
    }
    None
}

/// Writes both halves of the destination; with `Q = 0` the upper half becomes zero.
fn put(out: &mut Vec<Stmt>, q: bool, rd: u32, lo: Expr, hi: Expr) {
    out.push(Stmt::SetV {
        reg: rd as u8,
        hi: false,
        value: lo,
    });
    out.push(Stmt::SetV {
        reg: rd as u8,
        hi: true,
        value: if q { hi } else { c(0) },
    });
}

fn both_halves(q: bool, rd: u32, f: impl Fn(bool) -> Expr) -> Vec<Stmt> {
    let mut out = Vec::new();
    put(&mut out, q, rd, f(false), f(true));
    out
}

/// Sets the cumulative saturation bit (`QC`, `FPSR` bit 27) when any lane the instruction
/// processes saturated. It is written **before** the destination so a source/destination
/// alias cannot hide the original operands.
fn set_qc(out: &mut Vec<Stmt>, q: bool, flag: impl Fn(bool) -> Expr) {
    let flag = if q {
        op2(Op::Or, flag(false), flag(true))
    } else {
        flag(false)
    };
    out.push(Stmt::SetSysReg(
        SysReg::Fpsr,
        op2(
            Op::Or,
            Expr::SysReg(SysReg::Fpsr),
            op2(Op::Lsl64, flag, c(27)),
        ),
    ));
}

/// `SQADD`/`SQSUB`/`SQABS`/`SQNEG` lanes plus their `QC` update. `rm` is absent for the
/// two-register forms, whose second operand the helper ignores.
fn saturating(out: &mut Vec<Stmt>, q: bool, rd: u32, rn: u32, rm: Option<u32>, es: u32, mode: u64) {
    let second = |hi: bool| rm.map_or(c(0), |r| v(r, hi));
    let lane = |hi: bool| op(Op::Vsat, vec![v(rn, hi), second(hi), c(es as u64), c(mode)]);
    let flag = |hi: bool| {
        op(
            Op::VsatFlag,
            vec![v(rn, hi), second(hi), c(es as u64), c(mode)],
        )
    };
    set_qc(out, q, flag);
    put(out, q, rd, lane(false), lane(true));
}

fn shift_imm(w: u32) -> Option<Vec<Stmt>> {
    let q = bits(w, 30, 1) == 1;
    let u = bits(w, 29, 1) == 1;
    let immh = bits(w, 19, 4);
    let immhb = bits(w, 16, 7);
    let opcode = bits(w, 11, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    if immh == 0 {
        return None;
    }
    let esize = 8 << (31 - immh.leading_zeros());
    if esize > 64 || (!q && esize == 64) {
        return None;
    }
    // `immhb` outside [esize, 2*esize) would be an unallocated encoding; the helper does
    // not check it, so we do not claim those words either.
    if immhb < esize || immhb >= 2 * esize {
        return None;
    }
    let (op, shift) = match (opcode, u) {
        (0b01010, false) => (Op::Vshl, immhb - esize),
        (0b00000, false) => (Op::Vsshr, 2 * esize - immhb),
        (0b00000, true) => (Op::Vushr, 2 * esize - immhb),
        _ => return None,
    };
    Some(both_halves(q, rd, |hi| {
        op3(op, v(rn, hi), c(esize as u64), c(shift as u64))
    }))
}

/// `(esize, index)` from the `imm5` field of the copy group, as the helper decodes it.
fn imm5_elem(imm5: u32) -> Option<(u32, u32)> {
    let size = imm5.trailing_zeros();
    (imm5 != 0 && size <= 3).then(|| (8 << size, imm5 >> (size + 1)))
}

/// The half of `reg` holding element `index` of `esize` bits, and its bit offset there.
fn elem(reg: u32, esize: u32, index: u32) -> (Expr, u32) {
    let bit = index * esize;
    (v(reg, bit >= 64), bit % 64)
}

fn copy(w: u32) -> Option<Vec<Stmt>> {
    let q = bits(w, 30, 1) == 1;
    let op = bits(w, 29, 1);
    let imm5 = bits(w, 16, 5);
    let imm4 = bits(w, 11, 4);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let (esize, index) = imm5_elem(imm5)?;
    if esize > 16 {
        return None;
    }
    let size = esize.trailing_zeros() - 3;
    let datasize = if q { 128 } else { 64 };
    if index * esize >= datasize {
        return None;
    }
    match (op, imm4) {
        (0, 0b0000) => {
            let (half, off) = elem(rn, esize, index);
            let mut out = Vec::new();
            let src = if off == 0 {
                half
            } else {
                out.push(Stmt::Let(0, op2(Op::Lsr64, half, c(off as u64))));
                t(0)
            };
            let spread = op2(Op::Splat, src, c(esize as u64));
            put(&mut out, q, rd, spread.clone(), spread);
            Some(out)
        }
        (1, _) if q => {
            let src = imm4 >> size;
            if src * esize >= datasize {
                return None;
            }
            let (from, from_off) = elem(rn, esize, src);
            let (to, to_off) = elem(rd, esize, index);
            let mut out = vec![Stmt::Let(
                0,
                if from_off == 0 {
                    from
                } else {
                    op2(Op::Lsr64, from, c(from_off as u64))
                },
            )];
            out.push(Stmt::Let(1, op2(Op::And, t(0), c(mask(esize)))));
            let cleared = op2(Op::And, to, c(!(mask(esize) << to_off)));
            let value = op2(Op::Or, cleared, op2(Op::Lsl64, t(1), c(to_off as u64)));
            out.push(Stmt::SetV {
                reg: rd as u8,
                hi: index * esize >= 64,
                value,
            });
            Some(out)
        }
        _ => None,
    }
}

/// The three-same integer arithmetic forms with a per-lane result: `MUL`, `SMAX`/
/// `UMAX`, `SMIN`/`UMIN`, `MLA`/`MLS`, `SMAXP`/`UMAXP`/`SMINP`/`UMINP` and the
/// saturating `SQADD`/`SQSUB`. `ADD`/`SUB`/`ADDP` stay in `lift_fp`, which already
/// lifts them; the helper rejects the 64-bit lane encodings of `MUL`/`MAX`/`MIN`/`ML*`
/// and of `MAXP`/`MINP`.
fn three_same(w: u32) -> Option<Vec<Stmt>> {
    let q = bits(w, 30, 1) == 1;
    let u = bits(w, 29, 1) == 1;
    let size = bits(w, 22, 2);
    let (rm, opcode, rn, rd) = (bits(w, 16, 5), bits(w, 11, 5), bits(w, 5, 5), bits(w, 0, 5));
    let es = 8 << size;
    let signed = (!u) as u64;
    let mut out = Vec::new();
    match (opcode, u) {
        (0b10011, false) if size < 3 => Some(both_halves(q, rd, |hi| {
            op3(Op::Vmul, v(rn, hi), v(rm, hi), c(es as u64))
        })),
        // `U` = 0 is the signed form (`SMAX`/`SMIN`), `U` = 1 the unsigned one.
        (0b01100, _) if size < 3 => Some(both_halves(q, rd, |hi| {
            op(
                Op::Vmax,
                vec![v(rn, hi), v(rm, hi), c(es as u64), c(signed)],
            )
        })),
        (0b01101, _) if size < 3 => Some(both_halves(q, rd, |hi| {
            op(
                Op::Vmin,
                vec![v(rn, hi), v(rm, hi), c(es as u64), c(signed)],
            )
        })),
        // `U` = 1 turns the multiply-accumulate into `MLS`.
        (0b10010, _) if size < 3 => {
            let o = if u { Op::Vmls } else { Op::Vmla };
            Some(both_halves(q, rd, |hi| {
                op(o, vec![v(rn, hi), v(rm, hi), v(rd, hi), c(es as u64)])
            }))
        }
        // `MAXP`/`MINP`: the helper has no 64-bit lane form.
        (0b10100, _) if size < 3 => Some(pairwise(q, rd, rn, rm, es, Op::Vpmax, signed)),
        (0b10101, _) if size < 3 => Some(pairwise(q, rd, rn, rm, es, Op::Vpmin, signed)),
        (0b00001, _) if size < 3 || q => {
            saturating(&mut out, q, rd, rn, Some(rm), es, u as u64);
            Some(out)
        }
        (0b00101, _) if size < 3 || q => {
            saturating(&mut out, q, rd, rn, Some(rm), es, 2 + u as u64);
            Some(out)
        }
        _ => None,
    }
}

/// One output half of `ADDP`-style pairing reduces one source register: the two halves of
/// that register are the op's operands. With `Q` = 0 both sources share the single output
/// half, which is why the second operand is `rn`'s and `rm`'s low halves there.
fn pairwise(q: bool, rd: u32, rn: u32, rm: u32, es: u32, kind: Op, signed: u64) -> Vec<Stmt> {
    let pair = |lo: Expr, hi: Expr| op(kind, vec![lo, hi, c(es as u64), c(signed)]);
    let mut out = Vec::new();
    if !q {
        put(&mut out, false, rd, pair(vlo(rn), vlo(rm)), c(0));
        return out;
    }
    // The high half reduces `rm`, so writing the low half first would hide `rm.lo` when
    // `rd` and `rm` are the same register. Both halves are computed before either write.
    out.push(Stmt::Let(0, pair(vlo(rn), vhi(rn))));
    out.push(Stmt::Let(1, pair(vlo(rm), vhi(rm))));
    put(&mut out, true, rd, t(0), t(1));
    out
}

/// `ABS` (`U` = 0) and `NEG` (`U` = 1) from the two-register miscellaneous group
/// (`opcode` = `0b01011`, non-saturating) and `SQABS`/`SQNEG` (`0b00111`).
fn two_reg_misc(w: u32) -> Option<Vec<Stmt>> {
    let q = bits(w, 30, 1) == 1;
    let u = bits(w, 29, 1) == 1;
    let size = bits(w, 22, 2);
    let opcode = bits(w, 12, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    if !matches!(opcode, 0b01011 | 0b00111) || (size == 3 && !q) {
        return None;
    }
    let es = 8 << size;
    if opcode == 0b00111 {
        let mut out = Vec::new();
        saturating(&mut out, q, rd, rn, None, es, 4 + u as u64);
        return Some(out);
    }
    let op = if u { Op::Vneg } else { Op::Vabs };
    Some(both_halves(q, rd, |hi| op2(op, v(rn, hi), c(es as u64))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm;

    #[test]
    fn covered_forms_are_lifted() {
        for w in [
            0x4f0b_5420,                                             // shl v0.16b, v1.16b, #3
            0x0f0b_5420,                                             // shl v0.8b, v1.8b, #3
            0x6f0d_0420,                                             // ushr v0.16b, v1.16b, #3
            0x4f0d_0420,                                             // sshr v0.16b, v1.16b, #3
            0x4e07_0420,                                             // dup v0.16b, v1.b[3]
            0x0e0f_0420,                                             // dup v0.8b, v1.b[7]
            0x4e1f_0420,                                             // dup v0.8h, v1.h[3]
            0x6e05_2c20,                                             // ins v0.b[2], v1.b[5]
            0x6e0b_2c20,                                             // ins v0.h[1], v1.h[2]
            asm::neon_three_same(true, false, 2, 0b10011, 2, 1, 0),  // mul v0.4s, v1.4s, v2.4s
            asm::neon_three_same(false, false, 0, 0b10011, 2, 1, 0), // mul v0.8b, v1.8b, v2.8b
            asm::neon_three_same(true, false, 0, 0b01100, 2, 1, 0),  // smax v0.16b, v1.16b, v2.16b
            asm::neon_three_same(true, true, 1, 0b01100, 2, 1, 0),   // umax v0.8h, v1.8h, v2.8h
            asm::neon_three_same(true, false, 2, 0b01101, 2, 1, 0),  // smin v0.4s, v1.4s, v2.4s
            asm::neon_three_same(false, true, 0, 0b01101, 2, 1, 0),  // umin v0.8b, v1.8b, v2.8b
            asm::neon_two_misc(true, false, 0, 0b01011, 1, 0),       // abs v0.16b, v1.16b
            asm::neon_two_misc(false, true, 2, 0b01011, 1, 0),       // neg v0.2s, v1.2s
            asm::neon_two_misc(true, true, 3, 0b01011, 1, 0),        // neg v0.2d, v1.2d
            asm::neon_three_same(true, false, 0, 0b00001, 2, 1, 0),  // sqadd v0.16b
            asm::neon_three_same(false, true, 1, 0b00101, 2, 1, 0),  // sqsub v0.4h
            asm::neon_three_same(true, false, 3, 0b00001, 2, 1, 0),  // sqadd v0.2d
            asm::neon_three_same(true, true, 2, 0b10010, 2, 1, 0),   // mls v0.4s, v1.4s, v2.4s
            asm::neon_three_same(true, false, 1, 0b10100, 2, 1, 0),  // smaxp v0.8h
            asm::neon_three_same(false, true, 0, 0b10101, 2, 1, 0),  // uminp v0.8b
            asm::neon_two_misc(true, false, 2, 0b00111, 1, 0),       // sqabs v0.4s, v1.4s
            asm::neon_two_misc(true, true, 3, 0b00111, 1, 0),        // sqneg v0.2d, v1.2d
        ] {
            assert!(lift(w).is_some(), "{w:#010x} should be lifted");
        }
    }

    #[test]
    fn other_forms_stay_on_the_helper() {
        for w in [
            0x5f0d_0420, // sshr d0, d1, #3 (scalar shift by immediate)
            0x6f0d_2420, // sri v0.16b, v1.16b, #3
            0x6f0b_5420, // sli v0.16b, v1.16b, #3
            0x4e02_0020, // tbl v0.16b, {v1.16b}, v2.16b
            0x0f00_0420, // modified immediate (immh == 0)
            // `ADD`/`SUB`/`ADDP` are lifted by `lift_fp`, not here.
            asm::neon_three_same(true, false, 2, 0b10000, 2, 1, 0), // add v0.4s, v1.4s, v2.4s
            // 64-bit `MUL`, `MAX` and `MIN` three-same encodings are unallocated.
            asm::neon_three_same(true, false, 3, 0b10011, 2, 1, 0),
            asm::neon_three_same(true, false, 3, 0b01100, 2, 1, 0),
            asm::neon_three_same(true, true, 3, 0b01101, 2, 1, 0),
            // Polynomial multiply; the helper has no 64-bit `MAXP`/`MINP`.
            asm::neon_three_same(true, true, 0, 0b10011, 2, 1, 0), // pmul v0.16b, v1.16b, v2.16b
            asm::neon_three_same(true, false, 3, 0b10100, 2, 1, 0), // smaxp v0.2d
            asm::neon_three_same(true, true, 3, 0b10101, 2, 1, 0), // uminp v0.2d
            // 64-bit lane ABS/NEG into a D register is unallocated.
            asm::neon_two_misc(false, false, 3, 0b01011, 1, 0),
            asm::neon_two_misc(false, false, 3, 0b00111, 1, 0),
        ] {
            assert!(lift(w).is_none(), "{w:#010x} should not be lifted here");
        }
    }
}
