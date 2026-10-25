use crate::ir::*;
use crate::ops::{Op, fp_mode};
use crate::simd::fp::Round;

const SIGN32: u64 = 0x8000_0000;
const SIGN64: u64 = 1 << 63;
const LO32: u64 = 0xffff_ffff;

fn bits(w: u32, lo: u32, n: u32) -> u32 {
    (w >> lo) & ((1 << n) - 1)
}

fn fpcr() -> Expr {
    Expr::SysReg(SysReg::Fpcr)
}

fn scalar(n: u32, double: bool) -> Expr {
    if double {
        vlo(n)
    } else {
        op1(Op::Zext32, vlo(n))
    }
}

fn lane32(n: u32, i: u32) -> Expr {
    let half = if i >= 2 { vhi(n) } else { vlo(n) };
    if i % 2 == 1 {
        op2(Op::Lsr64, half, c(32))
    } else {
        op1(Op::Zext32, half)
    }
}

fn write_scalar(out: &mut Vec<Stmt>, d: u32, value: Expr) {
    out.push(Stmt::SetV {
        reg: d as u8,
        hi: false,
        value,
    });
    out.push(Stmt::SetV {
        reg: d as u8,
        hi: true,
        value: c(0),
    });
}

fn write_halves(out: &mut Vec<Stmt>, d: u32, lo: Expr, hi: Option<Expr>) {
    out.push(Stmt::Let(0, lo));
    let upper = match hi {
        Some(hi) => {
            out.push(Stmt::Let(1, hi));
            t(1)
        }
        None => c(0),
    };
    out.push(Stmt::SetV {
        reg: d as u8,
        hi: false,
        value: t(0),
    });
    out.push(Stmt::SetV {
        reg: d as u8,
        hi: true,
        value: upper,
    });
}

fn neg(e: Expr, double: bool) -> Expr {
    op2(Op::Xor, e, c(if double { SIGN64 } else { SIGN32 }))
}

pub fn lift(w: u32) -> Option<Vec<Stmt>> {
    let mut out = Vec::new();
    if w & 0xff20_0c00 == 0x1e20_0800 {
        fp_dp2(&mut out, w)?;
    } else if w & 0xff00_0000 == 0x1f00_0000 {
        fp_dp3(&mut out, w)?;
    } else if w & 0xff20_7c00 == 0x1e20_4000 {
        fp_dp1(&mut out, w)?;
    } else if w & 0xff20_3c00 == 0x1e20_2000 {
        fp_compare(&mut out, w)?;
    } else if w & 0xff20_0c00 == 0x1e20_0c00 {
        fp_select(&mut out, w)?;
    } else if w & 0xff20_1c00 == 0x1e20_1000 {
        fp_immediate(&mut out, w)?;
    } else if w & 0x7f20_fc00 == 0x1e20_0000 {
        fp_int(&mut out, w)?;
    } else if w & 0xdf3e_0c00 == 0x5e20_0800 {
        scalar_int_conv(&mut out, w)?;
    } else if w & 0x9f20_0400 == 0x0e20_0400 {
        three_same(&mut out, w)?;
    } else if w & 0x9f00_0400 == 0x0f00_0000 || w & 0xdf00_0400 == 0x5f00_0000 {
        by_element(&mut out, w)?;
    } else if w & 0xffe0_8400 == 0x6e00_0400 {
        ins_element(&mut out, w)?;
    } else if w & 0xbfe0_fc00 == 0x0e00_0400 || w & 0xffe0_fc00 == 0x5e00_0400 {
        dup_element(&mut out, w)?;
    } else if w & 0xbff8_fc00 == 0x2f00_e400 {
        movi_bytes(&mut out, w);
    } else if w & 0xff20_0c00 == 0x1e20_0400 {
        fp_cond_compare(&mut out, w)?;
    } else if w & 0xbfe0_8400 == 0x2e00_0000 {
        ext(&mut out, w)?;
    } else if w & 0xffe0_fc00 == 0x4e00_1c00 {
        ins_general(&mut out, w)?;
    } else if w & 0xbfe0_ec00 == 0x0e00_2c00 {
        mov_to_general(&mut out, w)?;
    } else if w & 0x9f80_fc00 == 0x0f00_a400 {
        shift_left_long(&mut out, w)?;
    } else if w & 0xbf20_8c00 == 0x0e00_0800 {
        permute(&mut out, w)?;
    } else if w & 0x9f3e_0c00 == 0x0e20_0800 {
        two_reg_misc(&mut out, w)?;
    } else if w & 0xbfe0_fc00 == 0x0e00_0c00 {
        dup_general(&mut out, w)?;
    } else if w & 0x9ff8_9c00 == 0x0f00_0400 {
        movi_words(&mut out, w)?;
    } else {
        return None;
    }
    Some(out)
}

fn fp_dp2(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let ty = bits(w, 22, 2);
    if ty > 1 {
        return None;
    }
    let double = ty == 1;
    let (rm, rn, rd) = (bits(w, 16, 5), bits(w, 5, 5), bits(w, 0, 5));
    let o = |single: Op, double_op: Op| if double { double_op } else { single };
    let (a, b) = (scalar(rn, double), scalar(rm, double));
    let r = match bits(w, 12, 4) {
        0 => op3(o(Op::FMul32, Op::FMul64), a, b, fpcr()),
        1 => op3(o(Op::FDiv32, Op::FDiv64), a, b, fpcr()),
        2 => op3(o(Op::FAdd32, Op::FAdd64), a, b, fpcr()),
        3 => op3(o(Op::FSub32, Op::FSub64), a, b, fpcr()),
        4 => op3(o(Op::FMax32, Op::FMax64), a, b, fpcr()),
        5 => op3(o(Op::FMin32, Op::FMin64), a, b, fpcr()),
        6 => op3(o(Op::FMaxNm32, Op::FMaxNm64), a, b, fpcr()),
        7 => op3(o(Op::FMinNm32, Op::FMinNm64), a, b, fpcr()),
        8 => neg(op3(o(Op::FMul32, Op::FMul64), a, b, fpcr()), double),
        _ => return None,
    };
    write_scalar(out, rd, r);
    Some(())
}

fn fp_dp3(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let ty = bits(w, 22, 2);
    if ty > 1 {
        return None;
    }
    let double = ty == 1;
    let (rm, ra, rn, rd) = (bits(w, 16, 5), bits(w, 10, 5), bits(w, 5, 5), bits(w, 0, 5));
    let (neg_a, neg_n) = match (bits(w, 21, 1), bits(w, 15, 1)) {
        (0, 0) => (false, false),
        (0, _) => (false, true),
        (_, 0) => (true, true),
        _ => (true, false),
    };
    let a = if neg_a {
        neg(scalar(ra, double), double)
    } else {
        scalar(ra, double)
    };
    let n = if neg_n {
        neg(scalar(rn, double), double)
    } else {
        scalar(rn, double)
    };
    let f = if double { Op::FMulAdd64 } else { Op::FMulAdd32 };
    write_scalar(out, rd, op(f, vec![a, n, scalar(rm, double), fpcr()]));
    Some(())
}

fn fp_dp1(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let ty = bits(w, 22, 2);
    if ty > 1 {
        return None;
    }
    let double = ty == 1;
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let a = scalar(rn, double);
    let sign = if double { SIGN64 } else { SIGN32 };
    let r = match bits(w, 15, 6) {
        0 => a,
        1 => op2(Op::And, a, c(!sign)),
        2 => op2(Op::Xor, a, c(sign)),
        3 => op2(if double { Op::FSqrt64 } else { Op::FSqrt32 }, a, fpcr()),
        4 if double => op2(Op::FCvtDS, a, fpcr()),
        5 if !double => op2(Op::FCvtSD, a, fpcr()),
        _ => return None,
    };
    write_scalar(out, rd, r);
    Some(())
}

fn fp_compare(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let ty = bits(w, 22, 2);
    if ty > 1 || bits(w, 0, 3) != 0 || bits(w, 14, 2) != 0 {
        return None;
    }
    let double = ty == 1;
    let (rm, rn) = (bits(w, 16, 5), bits(w, 5, 5));
    let b = if bits(w, 3, 1) == 1 {
        c(0)
    } else {
        scalar(rm, double)
    };
    let f = if double { Op::FCmp64 } else { Op::FCmp32 };
    out.push(Stmt::SetNzcv(op3(f, scalar(rn, double), b, fpcr())));
    Some(())
}

fn fp_select(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let ty = bits(w, 22, 2);
    if ty > 1 {
        return None;
    }
    let double = ty == 1;
    let (rm, rn, rd) = (bits(w, 16, 5), bits(w, 5, 5), bits(w, 0, 5));
    let cond = op2(Op::Cond, Expr::Nzcv, c(bits(w, 12, 4) as u64));
    write_scalar(
        out,
        rd,
        op3(Op::Select, cond, scalar(rn, double), scalar(rm, double)),
    );
    Some(())
}

fn fp_immediate(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let ty = bits(w, 22, 2);
    if ty > 1 || bits(w, 5, 5) != 0 {
        return None;
    }
    let width = if ty == 1 { 64 } else { 32 };
    write_scalar(
        out,
        bits(w, 0, 5),
        c(crate::simd::fp_expand_imm8(bits(w, 13, 8), width)),
    );
    Some(())
}

fn rounding(rmode: u32) -> Round {
    [Round::Nearest, Round::PlusInf, Round::MinusInf, Round::Zero][rmode as usize]
}

fn fp_int(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let sf = bits(w, 31, 1) == 1;
    let ty = bits(w, 22, 2);
    let rmode = bits(w, 19, 2);
    let opcode = bits(w, 16, 3);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let iw = if sf { 64 } else { 32 };
    let x = |n: u32| if n == 31 { c(0) } else { Expr::Reg(n as u8) };
    let set_x = |out: &mut Vec<Stmt>, e: Expr| {
        if rd != 31 {
            out.push(Stmt::SetReg(rd as u8, e));
        }
    };
    match opcode {
        0b110 | 0b111 => {
            let to_fp = opcode == 0b111;
            match (sf, ty, rmode) {
                (false, 0, 0) | (true, 1, 0) => {
                    if to_fp {
                        let v = if sf { x(rn) } else { op1(Op::Zext32, x(rn)) };
                        write_scalar(out, rd, v);
                    } else {
                        set_x(out, scalar(rn, sf));
                    }
                }
                (true, 2, 1) => {
                    if to_fp {
                        out.push(Stmt::SetV {
                            reg: rd as u8,
                            hi: true,
                            value: x(rn),
                        });
                    } else {
                        set_x(out, vhi(rn));
                    }
                }
                _ => return None,
            }
        }
        _ if ty > 1 => return None,
        0b000 | 0b001 | 0b100 | 0b101 => {
            let signed = opcode & 1 == 0;
            let mode = if opcode & 0b100 != 0 {
                if rmode != 0 {
                    return None;
                }
                Round::Away
            } else {
                rounding(rmode)
            };
            let f = if ty == 1 { Op::FToInt64 } else { Op::FToInt32 };
            set_x(
                out,
                op3(f, scalar(rn, ty == 1), c(fp_mode(mode, signed, iw)), fpcr()),
            );
        }
        0b010 | 0b011 if rmode == 0 => {
            let f = if ty == 1 {
                Op::FFromInt64
            } else {
                Op::FFromInt32
            };
            write_scalar(
                out,
                rd,
                op3(
                    f,
                    x(rn),
                    c(fp_mode(Round::Nearest, opcode == 0b010, iw)),
                    fpcr(),
                ),
            );
        }
        _ => return None,
    }
    Some(())
}

fn scalar_int_conv(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let opcode = bits(w, 12, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let double = size & 1 == 1;
    let es = if double { 64 } else { 32 };
    let r = match (opcode, size >> 1) {
        (0b11101, 0) => op3(
            if double {
                Op::FFromInt64
            } else {
                Op::FFromInt32
            },
            scalar(rn, double),
            c(fp_mode(Round::Nearest, u == 0, es)),
            fpcr(),
        ),
        (0b11011, 1) => op3(
            if double { Op::FToInt64 } else { Op::FToInt32 },
            scalar(rn, double),
            c(fp_mode(Round::Zero, u == 0, es)),
            fpcr(),
        ),
        _ => return None,
    };
    write_scalar(out, rd, r);
    Some(())
}

fn three_same(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let q = bits(w, 30, 1) == 1;
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let (rm, opcode, rn, rd) = (bits(w, 16, 5), bits(w, 11, 5), bits(w, 5, 5), bits(w, 0, 5));
    let hi = |e: &dyn Fn(bool) -> Expr| if q { Some(e(true)) } else { None };
    let half = |n: u32, h: bool| if h { vhi(n) } else { vlo(n) };
    if opcode == 0b00011 {
        let f = |h: bool| -> Option<Expr> {
            let (n, m) = (half(rn, h), half(rm, h));
            let d = half(rd, h);
            Some(match (u, size) {
                (0, 0) => op2(Op::And, n, m),
                (0, 1) => op2(Op::And, n, op1(Op::Not, m)),
                (0, 2) => op2(Op::Or, n, m),
                (0, _) => op2(Op::Or, n, op1(Op::Not, m)),
                (_, 0) => op2(Op::Xor, n, m),
                (_, 1) => op2(
                    Op::Or,
                    op2(Op::And, n, d.clone()),
                    op2(Op::And, m, op1(Op::Not, d)),
                ),
                (_, 2) => op2(
                    Op::Or,
                    op2(Op::And, n, m.clone()),
                    op2(Op::And, d, op1(Op::Not, m)),
                ),
                _ => op2(
                    Op::Or,
                    op2(Op::And, n, op1(Op::Not, m.clone())),
                    op2(Op::And, d, m),
                ),
            })
        };
        let lo = f(false)?;
        let h = if q { Some(f(true)?) } else { None };
        write_halves(out, rd, lo, h);
        return Some(());
    }
    if opcode == 0b10000 || (u == 0 && opcode == 0b10111) {
        if size == 3 && !q {
            return None;
        }
        let es = 8u32 << size;
        let n = if q { 128 } else { 64 } / es;
        let mask = if es == 64 { u64::MAX } else { (1u64 << es) - 1 };
        let sum = |x: Expr, y: Expr| {
            let r = if u == 1 {
                op2(Op::Sub, x, y)
            } else {
                op2(Op::Add, x, y)
            };
            if es == 64 {
                r
            } else {
                op2(Op::And, r, c(mask))
            }
        };
        let elems: Vec<Expr> = if opcode == 0b10000 {
            (0..n)
                .map(|i| sum(elem(rn, i, es), elem(rm, i, es)))
                .collect()
        } else {
            (0..n)
                .map(|k| {
                    let (src, j) = if k < n / 2 {
                        (rn, 2 * k)
                    } else {
                        (rm, 2 * (k - n / 2))
                    };
                    sum(elem(src, j, es), elem(src, j + 1, es))
                })
                .collect()
        };
        write_elems(out, rd, &elems, es, q);
        return Some(());
    }
    if size & 1 != 0 {
        return None;
    }
    let a_bit = size >> 1;
    let fpop = |o: Op, h: bool| op3(o, half(rn, h), half(rm, h), fpcr());
    match (u, a_bit, opcode) {
        (0, 0, 0b11010) => write_halves(
            out,
            rd,
            fpop(Op::FAdd32x2, false),
            hi(&|h| fpop(Op::FAdd32x2, h)),
        ),
        (0, 1, 0b11010) => write_halves(
            out,
            rd,
            fpop(Op::FSub32x2, false),
            hi(&|h| fpop(Op::FSub32x2, h)),
        ),
        (1, 0, 0b11011) => write_halves(
            out,
            rd,
            fpop(Op::FMul32x2, false),
            hi(&|h| fpop(Op::FMul32x2, h)),
        ),
        (0, _, 0b11001) => {
            let fma = |h: bool| {
                let n = if a_bit == 1 {
                    op2(Op::Xor, half(rn, h), c(SIGN32 | SIGN32 << 32))
                } else {
                    half(rn, h)
                };
                op(Op::FMulAdd32x2, vec![half(rd, h), n, half(rm, h), fpcr()])
            };
            write_halves(out, rd, fma(false), hi(&fma));
        }
        (_, _, 0b11110) | (_, _, 0b11000) => {
            let o = match (a_bit, opcode) {
                (0, 0b11110) => Op::FMax32,
                (_, 0b11110) => Op::FMin32,
                (0, _) => Op::FMaxNm32,
                _ => Op::FMinNm32,
            };
            let n = if q { 4 } else { 2 };
            let elems: Vec<Expr> = if u == 0 {
                (0..n)
                    .map(|i| op3(o, lane32(rn, i), lane32(rm, i), fpcr()))
                    .collect()
            } else {
                (0..n)
                    .map(|k| {
                        let (src, j) = if k < n / 2 {
                            (rn, 2 * k)
                        } else {
                            (rm, 2 * (k - n / 2))
                        };
                        op3(o, lane32(src, j), lane32(src, j + 1), fpcr())
                    })
                    .collect()
            };
            write_elems(out, rd, &elems, 32, q);
        }
        (_, _, 0b11100) => {
            let test = match (u, a_bit) {
                (0, 0) => Cmp::Eq,
                (1, 0) => Cmp::Ge,
                (1, _) => Cmp::Gt,
                _ => return None,
            };
            let n = if q { 4 } else { 2 };
            let elems: Vec<Expr> = (0..n)
                .map(|i| fcmp_lane(lane32(rn, i), lane32(rm, i), test))
                .collect();
            write_elems(out, rd, &elems, 32, q);
        }
        (1, 0, 0b11010) => {
            let pair =
                |n: u32, i: u32| op3(Op::FAdd32, lane32(n, 2 * i), lane32(n, 2 * i + 1), fpcr());
            let pack = |a: Expr, b: Expr| op2(Op::Or, a, op2(Op::Lsl64, b, c(32)));
            if q {
                write_halves(
                    out,
                    rd,
                    pack(pair(rn, 0), pair(rn, 1)),
                    Some(pack(pair(rm, 0), pair(rm, 1))),
                );
            } else {
                write_halves(out, rd, pack(pair(rn, 0), pair(rm, 0)), None);
            }
        }
        _ => return None,
    }
    Some(())
}

fn by_element(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let is_scalar = w >> 28 & 1 == 1;
    let q = bits(w, 30, 1) == 1;
    let (u, size, opcode) = (bits(w, 29, 1), bits(w, 22, 2), bits(w, 12, 4));
    if u != 0 || size != 0b10 {
        return None;
    }
    let index = bits(w, 11, 1) << 1 | bits(w, 21, 1);
    let rm = bits(w, 16, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    if !matches!(opcode, 0b0001 | 0b0101 | 0b1001) {
        return None;
    }
    out.push(Stmt::Let(2, lane32(rm, index)));
    if is_scalar {
        let n = scalar(rn, false);
        let r = match opcode {
            0b1001 => op3(Op::FMul32, n, t(2), fpcr()),
            0b0001 => op(Op::FMulAdd32, vec![scalar(rd, false), n, t(2), fpcr()]),
            _ => op(
                Op::FMulAdd32,
                vec![scalar(rd, false), neg(n, false), t(2), fpcr()],
            ),
        };
        write_scalar(out, rd, r);
        return Some(());
    }
    out.push(Stmt::Let(3, op2(Op::Or, t(2), op2(Op::Lsl64, t(2), c(32)))));
    let half = |n: u32, h: bool| if h { vhi(n) } else { vlo(n) };
    let f = |h: bool| match opcode {
        0b1001 => op3(Op::FMul32x2, half(rn, h), t(3), fpcr()),
        0b0001 => op(
            Op::FMulAdd32x2,
            vec![half(rd, h), half(rn, h), t(3), fpcr()],
        ),
        _ => op(
            Op::FMulAdd32x2,
            vec![
                half(rd, h),
                op2(Op::Xor, half(rn, h), c(SIGN32 | SIGN32 << 32)),
                t(3),
                fpcr(),
            ],
        ),
    };
    write_halves(out, rd, f(false), if q { Some(f(true)) } else { None });
    Some(())
}

fn ins_element(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let imm5 = bits(w, 16, 5);
    let imm4 = bits(w, 11, 4);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    if imm5 & 0b1111 == 0b1000 {
        let (to, from) = (imm5 >> 4 == 1, imm4 >> 3 == 1);
        let src = if from { vhi(rn) } else { vlo(rn) };
        out.push(Stmt::SetV {
            reg: rd as u8,
            hi: to,
            value: src,
        });
        return Some(());
    }
    if imm5 & 0b111 != 0b100 {
        return None;
    }
    let (to, from) = (imm5 >> 3, imm4 >> 2);
    let shift = 32 * (to % 2) as u64;
    let dst = if to >= 2 { vhi(rd) } else { vlo(rd) };
    out.push(Stmt::Let(0, lane32(rn, from)));
    let kept = op2(Op::And, dst, c(!(LO32 << shift)));
    out.push(Stmt::SetV {
        reg: rd as u8,
        hi: to >= 2,
        value: op2(Op::Or, kept, op2(Op::Lsl64, t(0), c(shift))),
    });
    Some(())
}

fn dup_element(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let is_scalar = w >> 28 & 1 == 1;
    let q = bits(w, 30, 1) == 1;
    let imm5 = bits(w, 16, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let elem = if imm5 & 0b111 == 0b100 {
        let e = lane32(rn, imm5 >> 3);
        if is_scalar {
            e
        } else {
            op2(Op::Or, e.clone(), op2(Op::Lsl64, e, c(32)))
        }
    } else if imm5 & 0b1111 == 0b1000 {
        if !is_scalar && !q {
            return None;
        }
        if imm5 >> 4 == 1 { vhi(rn) } else { vlo(rn) }
    } else {
        return None;
    };
    out.push(Stmt::Let(0, elem));
    out.push(Stmt::SetV {
        reg: rd as u8,
        hi: false,
        value: t(0),
    });
    out.push(Stmt::SetV {
        reg: rd as u8,
        hi: true,
        value: if q && !is_scalar { t(0) } else { c(0) },
    });
    Some(())
}

fn movi_bytes(out: &mut Vec<Stmt>, w: u32) {
    let q = bits(w, 30, 1) == 1;
    let imm8 = bits(w, 16, 3) << 5 | bits(w, 5, 5);
    let imm: u64 = (0..8)
        .filter(|i| imm8 >> i & 1 == 1)
        .map(|i| 0xff << (8 * i))
        .sum();
    let rd = bits(w, 0, 5) as u8;
    out.push(Stmt::SetV {
        reg: rd,
        hi: false,
        value: c(imm),
    });
    out.push(Stmt::SetV {
        reg: rd,
        hi: true,
        value: c(if q { imm } else { 0 }),
    });
}

fn elem(n: u32, i: u32, es: u32) -> Expr {
    let bit = i * es;
    let half = if bit >= 64 { vhi(n) } else { vlo(n) };
    let v = if bit.is_multiple_of(64) {
        half
    } else {
        op2(Op::Lsr64, half, c((bit % 64) as u64))
    };
    match es {
        8 => op1(Op::Zext8, v),
        16 => op1(Op::Zext16, v),
        32 => op1(Op::Zext32, v),
        _ => v,
    }
}

fn pack(elems: &[Expr], es: u32) -> Expr {
    let mut r = elems[0].clone();
    for (i, e) in elems.iter().enumerate().skip(1) {
        r = op2(
            Op::Or,
            r,
            op2(Op::Lsl64, e.clone(), c(i as u64 * es as u64)),
        );
    }
    r
}

fn write_elems(out: &mut Vec<Stmt>, d: u32, elems: &[Expr], es: u32, q: bool) {
    let per = (64 / es) as usize;
    let hi = q.then(|| pack(&elems[per..], es));
    write_halves(out, d, pack(&elems[..per], es), hi);
}

fn sext(e: Expr, es: u32) -> Expr {
    match es {
        8 => op1(Op::Sext8, e),
        16 => op1(Op::Sext16, e),
        _ => op1(Op::Sext32, e),
    }
}

#[derive(Clone, Copy)]
enum Cmp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
}

fn fcmp_lane(a: Expr, b: Expr, test: Cmp) -> Expr {
    let f = op3(Op::FCmp32, a, b, fpcr());
    let is = |nzcv: u64| op2(Op::Eq, f.clone(), c(nzcv << 28));
    let cond = match test {
        Cmp::Eq => is(0b0110),
        Cmp::Gt => is(0b0010),
        Cmp::Lt => is(0b1000),
        Cmp::Ge => op2(Op::Or, is(0b0010), is(0b0110)),
        Cmp::Le => op2(Op::Or, is(0b1000), is(0b0110)),
    };
    op3(Op::Select, cond, c(LO32), c(0))
}

fn fp_cond_compare(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let ty = bits(w, 22, 2);
    if ty > 1 {
        return None;
    }
    let double = ty == 1;
    let (rm, cond, rn, nzcv) = (bits(w, 16, 5), bits(w, 12, 4), bits(w, 5, 5), bits(w, 0, 4));
    let f = if double { Op::FCmp64 } else { Op::FCmp32 };
    let holds = op2(Op::Cond, Expr::Nzcv, c(cond as u64));
    out.push(Stmt::SetNzcv(op3(
        Op::Select,
        holds,
        op3(f, scalar(rn, double), scalar(rm, double), fpcr()),
        c((nzcv as u64) << 28),
    )));
    Some(())
}

fn ext(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let q = bits(w, 30, 1) == 1;
    let (rm, imm4, rn, rd) = (bits(w, 16, 5), bits(w, 11, 4), bits(w, 5, 5), bits(w, 0, 5));
    if !q && imm4 >= 8 {
        return None;
    }
    let words = if q {
        vec![vlo(rn), vhi(rn), vlo(rm), vhi(rm)]
    } else {
        vec![vlo(rn), vlo(rm)]
    };
    let at = |b: u32| -> Expr {
        let (i, s) = ((b / 8) as usize, (b % 8) * 8);
        if s == 0 {
            words[i].clone()
        } else {
            op2(
                Op::Or,
                op2(Op::Lsr64, words[i].clone(), c(s as u64)),
                op2(Op::Lsl64, words[i + 1].clone(), c(64 - s as u64)),
            )
        }
    };
    write_halves(out, rd, at(imm4), q.then(|| at(imm4 + 8)));
    Some(())
}

fn ins_general(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let imm5 = bits(w, 16, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let size = imm5.trailing_zeros();
    if size > 3 {
        return None;
    }
    let es = 8u32 << size;
    let bit = (imm5 >> (size + 1)) * es;
    let (hi, sh) = (bit >= 64, (bit % 64) as u64);
    let x = if rn == 31 { c(0) } else { Expr::Reg(rn as u8) };
    let value = if es == 64 {
        x
    } else {
        let mask = (1u64 << es) - 1;
        let dst = if hi { vhi(rd) } else { vlo(rd) };
        op2(
            Op::Or,
            op2(Op::And, dst, c(!(mask << sh))),
            op2(Op::Lsl64, op2(Op::And, x, c(mask)), c(sh)),
        )
    };
    out.push(Stmt::SetV {
        reg: rd as u8,
        hi,
        value,
    });
    Some(())
}

fn mov_to_general(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let q = bits(w, 30, 1) == 1;
    let unsigned = bits(w, 12, 1) == 1;
    let imm5 = bits(w, 16, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let size = imm5.trailing_zeros();
    let valid = if unsigned {
        size <= 3 && q == (size == 3)
    } else {
        size <= if q { 2 } else { 1 }
    };
    if !valid {
        return None;
    }
    let es = 8u32 << size;
    let e = elem(rn, imm5 >> (size + 1), es);
    let v = if unsigned {
        e
    } else if q {
        sext(e, es)
    } else {
        op1(Op::Zext32, sext(e, es))
    };
    if rd != 31 {
        out.push(Stmt::SetReg(rd as u8, v));
    }
    Some(())
}

fn shift_left_long(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let (q, u) = (bits(w, 30, 1) == 1, bits(w, 29, 1));
    let (immh, immb) = (bits(w, 19, 4), bits(w, 16, 3));
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let es = match immh {
        1 => 8,
        2 | 3 => 16,
        4..=7 => 32,
        _ => return None,
    };
    let shift = ((immh << 3 | immb) - es) as u64;
    let (per, wide) = (64 / es, 2 * es);
    let elems: Vec<Expr> = (0..per)
        .map(|i| {
            let e = elem(rn, i + if q { per } else { 0 }, es);
            let x = if u == 1 { e } else { sext(e, es) };
            let x = if shift == 0 {
                x
            } else {
                op2(Op::Lsl64, x, c(shift))
            };
            if wide == 64 {
                x
            } else {
                op2(Op::And, x, c((1u64 << wide) - 1))
            }
        })
        .collect();
    write_elems(out, rd, &elems, wide, true);
    Some(())
}

fn permute(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let q = bits(w, 30, 1) == 1;
    let (size, rm, opc, rn, rd) = (
        bits(w, 22, 2),
        bits(w, 16, 5),
        bits(w, 12, 3),
        bits(w, 5, 5),
        bits(w, 0, 5),
    );
    if (size == 3 && !q) || opc & 3 == 0 {
        return None;
    }
    let es = 8u32 << size;
    let n = if q { 128 } else { 64 } / es;
    let half = n / 2;
    let (a, b) = (|i: u32| elem(rn, i, es), |i: u32| elem(rm, i, es));
    let elems: Vec<Expr> = (0..n)
        .map(|k| match opc {
            0b001 => {
                if k < half {
                    a(2 * k)
                } else {
                    b(2 * (k - half))
                }
            }
            0b101 => {
                if k < half {
                    a(2 * k + 1)
                } else {
                    b(2 * (k - half) + 1)
                }
            }
            0b010 => {
                if k % 2 == 0 {
                    a(k)
                } else {
                    b(k - 1)
                }
            }
            0b110 => {
                if k % 2 == 0 {
                    a(k + 1)
                } else {
                    b(k)
                }
            }
            0b011 => {
                if k % 2 == 0 {
                    a(k / 2)
                } else {
                    b(k / 2)
                }
            }
            _ => {
                if k % 2 == 0 {
                    a(half + k / 2)
                } else {
                    b(half + k / 2)
                }
            }
        })
        .collect();
    write_elems(out, rd, &elems, es, q);
    Some(())
}

fn dup_general(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let q = bits(w, 30, 1) == 1;
    let (imm5, rn, rd) = (bits(w, 16, 5), bits(w, 5, 5), bits(w, 0, 5));
    let size = imm5.trailing_zeros();
    if size > 3 || (size == 3 && !q) {
        return None;
    }
    let es = 8u32 << size;
    let x = if rn == 31 { c(0) } else { Expr::Reg(rn as u8) };
    let e = if es == 64 {
        x
    } else {
        op2(Op::And, x, c((1u64 << es) - 1))
    };
    out.push(Stmt::Let(2, e));
    let elems: Vec<Expr> = (0..64 / es).map(|_| t(2)).collect();
    let v = pack(&elems, es);
    write_halves(out, rd, v.clone(), q.then_some(v));
    Some(())
}

fn movi_words(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let q = bits(w, 30, 1) == 1;
    let invert = bits(w, 29, 1) == 1;
    let cmode = bits(w, 12, 4);
    let imm8 = (bits(w, 16, 3) << 5 | bits(w, 5, 5)) as u64;
    let word = imm8 << (8 * (cmode >> 1 & 3));
    let word = if invert { !word & LO32 } else { word };
    let v = word | word << 32;
    let rd = bits(w, 0, 5) as u8;
    out.push(Stmt::SetV {
        reg: rd,
        hi: false,
        value: c(v),
    });
    out.push(Stmt::SetV {
        reg: rd,
        hi: true,
        value: c(if q { v } else { 0 }),
    });
    Some(())
}

fn two_reg_misc(out: &mut Vec<Stmt>, w: u32) -> Option<()> {
    let (q, u) = (bits(w, 30, 1) == 1, bits(w, 29, 1));
    let (size, opcode, rn, rd) = (bits(w, 22, 2), bits(w, 12, 5), bits(w, 5, 5), bits(w, 0, 5));
    let half = |h: bool| if h { vhi(rn) } else { vlo(rn) };
    match (u, opcode) {
        (_, 0b00000) => {
            let f = |x: Expr| -> Option<Expr> {
                Some(match (u, size) {
                    (0, 0) => op1(Op::Rev64, x),
                    (0, 1) => op1(Op::Rev16x4, op1(Op::Rev64, x)),
                    (0, 2) => op2(Op::Ror64, x, c(32)),
                    (1, 0) => op1(Op::Rev32x2, x),
                    (1, 1) => op1(Op::Rev16x4, op1(Op::Rev32x2, x)),
                    _ => return None,
                })
            };
            let lo = f(half(false))?;
            let hi = if q { Some(f(half(true))?) } else { None };
            write_halves(out, rd, lo, hi);
        }
        (0, 0b10010) => {
            if size == 3 {
                return None;
            }
            let es = 8u32 << size;
            let elems: Vec<Expr> = (0..64 / es)
                .map(|i| op2(Op::And, elem(rn, i, 2 * es), c((1u64 << es) - 1)))
                .collect();
            let v = pack(&elems, es);
            if q {
                out.push(Stmt::Let(0, v));
                out.push(Stmt::SetV {
                    reg: rd as u8,
                    hi: true,
                    value: t(0),
                });
            } else {
                write_halves(out, rd, v, None);
            }
        }
        (_, 0b01111) if size >= 2 => {
            let double = size == 3;
            if double && !q {
                return None;
            }
            let sign = if double {
                SIGN64
            } else {
                SIGN32 | SIGN32 << 32
            };
            let f = |x: Expr| {
                if u == 1 {
                    op2(Op::Xor, x, c(sign))
                } else {
                    op2(Op::And, x, c(!sign))
                }
            };
            write_halves(out, rd, f(half(false)), q.then(|| f(half(true))));
        }
        (_, 0b01100..=0b01110) if size == 2 => {
            let test = match (u, opcode) {
                (0, 0b01100) => Cmp::Gt,
                (1, 0b01100) => Cmp::Ge,
                (0, 0b01101) => Cmp::Eq,
                (1, 0b01101) => Cmp::Le,
                (0, _) => Cmp::Lt,
                _ => return None,
            };
            let n = if q { 4 } else { 2 };
            let elems: Vec<Expr> = (0..n)
                .map(|i| fcmp_lane(lane32(rn, i), c(0), test))
                .collect();
            write_elems(out, rd, &elems, 32, q);
        }
        _ => return None,
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::Env;

    #[derive(Clone, PartialEq, Debug)]
    struct State {
        x: [u64; 32],
        v: [u128; 32],
        nzcv: u64,
        fpcr: u64,
        fpsr: u64,
    }

    impl Env for State {
        fn x(&self, n: u32) -> u64 {
            if n == 31 { 0 } else { self.x[n as usize] }
        }
        fn set_x(&mut self, n: u32, v: u64) {
            if n != 31 {
                self.x[n as usize] = v;
            }
        }
        fn sp(&self) -> u64 {
            0
        }
        fn set_sp(&mut self, _: u64) {}
        fn v(&self, n: u32) -> u128 {
            self.v[n as usize]
        }
        fn set_v(&mut self, n: u32, v: u128) {
            self.v[n as usize] = v;
        }
        fn nzcv(&self) -> u64 {
            self.nzcv
        }
        fn set_nzcv(&mut self, v: u64) {
            self.nzcv = v;
        }
        fn fpcr(&self) -> u64 {
            self.fpcr
        }
        fn fpsr(&self) -> u64 {
            self.fpsr
        }
        fn set_fpsr(&mut self, v: u64) {
            self.fpsr = v;
        }
        fn load(&mut self, _: u64, _: u8) -> u64 {
            unreachable!()
        }
        fn store(&mut self, _: u64, _: u8, _: u64) {
            unreachable!()
        }
    }

    fn eval(s: &State, e: &Expr, tmp: &[u64; 8]) -> u64 {
        match e {
            Expr::Const(v) => *v,
            Expr::Reg(n) => s.x[*n as usize],
            Expr::Nzcv => s.nzcv,
            Expr::Tmp(n) => tmp[*n as usize],
            Expr::Op(o, args) => o.eval(&args.iter().map(|a| eval(s, a, tmp)).collect::<Vec<_>>()),
            Expr::SysReg(SysReg::Fpcr) => s.fpcr,
            Expr::V { reg, hi } => (s.v[*reg as usize] >> if *hi { 64 } else { 0 }) as u64,
            other => panic!("unexpected {other:?}"),
        }
    }

    fn run(s: &mut State, stmts: &[Stmt]) {
        let mut tmp = [0u64; 8];
        for st in stmts {
            match st {
                Stmt::Let(n, e) => tmp[*n as usize] = eval(s, e, &tmp),
                Stmt::SetReg(n, e) => s.x[*n as usize] = eval(s, e, &tmp),
                Stmt::SetNzcv(e) => s.nzcv = eval(s, e, &tmp),
                Stmt::SetV { reg, hi, value } => {
                    let v = eval(s, value, &tmp) as u128;
                    let r = &mut s.v[*reg as usize];
                    *r = if *hi {
                        (*r & u64::MAX as u128) | v << 64
                    } else {
                        (*r & !(u64::MAX as u128)) | v
                    };
                }
                other => panic!("unexpected {other:?}"),
            }
        }
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn f32(&mut self) -> u64 {
            let r = self.next();
            let special = [
                0,
                SIGN32,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
                0x7f80_0001,
                0xffc0_1234,
                0x0000_0001,
                0x807f_ffff,
                0x3f80_0000,
                0x0080_0000,
            ];
            match r % 4 {
                0 => special[(r >> 8) as usize % special.len()],
                1 => (r >> 16) & LO32,
                _ => ((r >> 20) % 0x60 + 0x40) << 23 | (r >> 32) & 0x7f_ffff | (r >> 63) << 31,
            }
        }

        fn f64(&mut self) -> u64 {
            let r = self.next();
            match r % 4 {
                0 => [
                    0,
                    SIGN64,
                    0x7ff0 << 48,
                    0xfff0 << 48,
                    0x7ff8 << 48,
                    0x7ff0_0000_0000_0001,
                    1,
                    0x3ff0 << 48,
                ][(r >> 8) as usize % 8],
                1 => r,
                _ => ((r >> 20) % 0x300 + 0x200) << 52 | (r >> 12) & ((1 << 52) - 1),
            }
        }

        fn state(&mut self) -> State {
            let mut s = State {
                x: [0; 32],
                v: [0; 32],
                nzcv: (self.next() >> 60) << 28,
                fpcr: 0,
                fpsr: 0,
            };
            for i in 0..32 {
                s.x[i] = if self.next().is_multiple_of(2) {
                    self.next()
                } else {
                    self.f32()
                };
                let (a, b) = (
                    self.f32() | self.f32() << 32,
                    if self.next().is_multiple_of(3) {
                        self.f64()
                    } else {
                        self.f32() | self.f32() << 32
                    },
                );
                s.v[i] = a as u128 | (b as u128) << 64;
                if self.next().is_multiple_of(4) {
                    s.v[i] = self.f64() as u128 | (self.f64() as u128) << 64;
                }
            }
            s.fpcr =
                [0, 0, 0, 1 << 24, 1 << 25, 3 << 24, 1 << 22, 3 << 22][(self.next() % 8) as usize];
            s
        }
    }

    // One entry per form `lift_fp` still claims. The `SHL` by-immediate form
    // (`0x0f00_5400`) moved to `lift_neon`, which covers it in its own tests.
    const FORMS: &[(u32, u32)] = &[
        (0x1e20_0800, 0x00df_f3ff),
        (0x1f00_0000, 0x00ff_ffff),
        (0x1e20_4000, 0x00c1_83ff),
        (0x1e20_2000, 0x00df_03f8),
        (0x1e20_0c00, 0x00df_f3ff),
        (0x1e20_1000, 0x00df_e01f),
        (0x1e20_0000, 0x80df_03ff),
        (0x5e20_0800, 0x20c1_f3ff),
        (0x0e20_0400, 0x60df_fbff),
        (0x0f80_0000, 0x607f_fbff),
        (0x5f80_0000, 0x207f_fbff),
        (0x6e00_0400, 0x001f_7bff),
        (0x0e00_0400, 0x401f_03ff),
        (0x5e00_0400, 0x001f_03ff),
        (0x2f00_e400, 0x4007_03ff),
        (0x1e22_4000, 0x00c1_83ff),
        (0x1e20_0400, 0x00df_f3ff),
        (0x2e00_0000, 0x401f_7bff),
        (0x4e00_1c00, 0x001f_03ff),
        (0x0e00_2c00, 0x401f_13ff),
        (0x0f00_a400, 0x607f_03ff),
        (0x0e00_0800, 0x40df_73ff),
        (0x0e20_0800, 0x60c1_f3ff),
        (0x0e00_0c00, 0x401f_03ff),
        (0x0f00_0400, 0x6007_63ff),
        (0x0e20_8400, 0x60df_03ff),
        (0x0e20_bc00, 0x40df_03ff),
    ];

    #[test]
    fn lifted_forms_match_simd_exec() {
        let mut rng = Rng(0x0dd5_eed5_1234_5678);
        for &(base, random) in FORMS {
            let mut lifted = 0;
            for _ in 0..4000 {
                let w = base | (rng.next() as u32 & random);
                let Some(stmts) = lift(w) else { continue };
                assert!(
                    crate::simd::supported(w),
                    "{w:08x} lifted but not supported by simd::exec"
                );
                lifted += 1;
                for _ in 0..4 {
                    let start = rng.state();
                    let mut want = start.clone();
                    assert!(crate::simd::exec(&mut want, w));
                    let mut got = start.clone();
                    run(&mut got, &stmts);
                    assert_eq!(got, want, "{w:08x}: {stmts:?}");
                }
            }
            assert!(lifted >= 100, "{base:08x}: only {lifted} words lifted");
        }
    }

    #[test]
    fn common_forms_are_lifted() {
        for w in [
            0x1e20_0820u32,
            0x1f03_4e33,
            0x1e20_20a8,
            0x6e08_0420,
            0x6e0c_0422,
            0x6e33_dce7,
            0x4fa3_10a1,
            0x5e1c_0606,
            0x6e07_40f0,
            0x4e1c_1d03,
            0x0e22_f462,
            0x2ea1_f423,
            0x1e20_0424,
            0x0f10_a463,
            0x0e84_3863,
            0x0ea0_08a5,
            0x6e76_1c27,
            0x1e21_c020,
            0x1e62_4000,
            0x1e22_c000,
            0x6ea0_f821,
            0x2ea1_e400,
            0x0e61_2884,
            0x4ea0_d884,
            0x0e02_3c88,
            0x4f00_0420,
            0x0ea6_bca5,
            0x4e08_0d20,
            0x4e04_0d03,
            0x4eb1_8421,
        ] {
            assert!(lift(w).is_some(), "{w:08x}");
        }
    }
}
