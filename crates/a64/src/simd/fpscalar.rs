use super::fp::{self, f32b, f64b, Fpcr, Round};
use super::{bits, fp_expand_imm8, Env};

const SINGLE: u32 = 0;
const DOUBLE: u32 = 1;
const HALF: u32 = 3;

fn width(ty: u32) -> u32 {
    match ty {
        SINGLE => 32,
        DOUBLE => 64,
        _ => 16,
    }
}

fn rd(e: &dyn Env, n: u32, ty: u32) -> u64 {
    let v = e.v(n);
    match ty {
        SINGLE => v as u32 as u64,
        DOUBLE => v as u64,
        _ => v as u16 as u64,
    }
}

fn wr(e: &mut dyn Env, n: u32, x: u64) {
    e.set_v(n, x as u128);
}

fn fpcr(e: &dyn Env) -> Fpcr {
    Fpcr::from_bits(e.fpcr())
}

macro_rules! by_type {
    ($ty:expr, $m:ident => $body:expr) => {
        match $ty {
            SINGLE => {
                use f32b as $m;
                ($body) as u64
            }
            _ => {
                use f64b as $m;
                ($body) as u64
            }
        }
    };
}

pub fn exec(e: &mut dyn Env, w: u32) -> bool {
    if w & 0x7f20_0000 == 0x1e00_0000 {
        return fixed_conv(e, w);
    }
    if w & 0x7f20_fc00 == 0x1e20_0000 {
        return int_conv(e, w);
    }
    if w & 0xff20_7c00 == 0x1e20_4000 {
        return dp1(e, w);
    }
    if w & 0xff20_3c00 == 0x1e20_2000 {
        return compare(e, w);
    }
    if w & 0xff20_1c00 == 0x1e20_1000 {
        return immediate(e, w);
    }
    if w & 0xff20_0c00 == 0x1e20_0400 {
        return cond_compare(e, w);
    }
    if w & 0xff20_0c00 == 0x1e20_0800 {
        return dp2(e, w);
    }
    if w & 0xff20_0c00 == 0x1e20_0c00 {
        return cond_select(e, w);
    }
    if w & 0xff00_0000 == 0x1f00_0000 {
        return dp3(e, w);
    }
    false
}

fn dp1(e: &mut dyn Env, w: u32) -> bool {
    let ty = bits(w, 22, 2);
    let opcode = bits(w, 15, 6);
    let (rn, rd_) = (bits(w, 5, 5), bits(w, 0, 5));
    let c = fpcr(e);
    let a = rd(e, rn, ty);
    if ty == 2 {
        return false;
    }
    let r = match opcode {
        0b000100 | 0b000101 | 0b000111 => {
            let to = opcode & 3;
            if to == ty {
                return false;
            }
            match (ty, to) {
                (SINGLE, DOUBLE) => fp::f32_to_f64(a as u32, c),
                (DOUBLE, SINGLE) => fp::f64_to_f32(a, c) as u64,
                (SINGLE, HALF) => fp::f32_to_f16(a as u32, c) as u64,
                (DOUBLE, HALF) => fp::f64_to_f16(a, c) as u64,
                (HALF, SINGLE) => fp::f16_to_f32(a as u16, c) as u64,
                (HALF, DOUBLE) => fp::f32_to_f64(fp::f16_to_f32(a as u16, c), c),
                _ => return false,
            }
        }
        _ if ty == HALF => return false,
        0b000000 => a,
        0b000001 => by_type!(ty, m => m::abs(a as _)),
        0b000010 => by_type!(ty, m => m::neg(a as _)),
        0b000011 => by_type!(ty, m => m::sqrt(a as _, c)),
        0b001000..=0b001111 => {
            let mode = match opcode & 7 {
                0 => Round::Nearest,
                1 => Round::PlusInf,
                2 => Round::MinusInf,
                3 => Round::Zero,
                4 => Round::Away,
                6 | 7 => Round::from_rmode(c.rmode),
                _ => return false,
            };
            by_type!(ty, m => m::round_int(a as _, mode, c))
        }
        _ => return false,
    };
    wr(e, rd_, r);
    true
}

fn dp2(e: &mut dyn Env, w: u32) -> bool {
    let ty = bits(w, 22, 2);
    if ty > DOUBLE {
        return false;
    }
    let (rm, rn, rd_) = (bits(w, 16, 5), bits(w, 5, 5), bits(w, 0, 5));
    let c = fpcr(e);
    let (a, b) = (rd(e, rn, ty), rd(e, rm, ty));
    let r = match bits(w, 12, 4) {
        0 => by_type!(ty, m => m::mul(a as _, b as _, c)),
        1 => by_type!(ty, m => m::div(a as _, b as _, c)),
        2 => by_type!(ty, m => m::add(a as _, b as _, c)),
        3 => by_type!(ty, m => m::sub(a as _, b as _, c)),
        4 => by_type!(ty, m => m::max(a as _, b as _, c)),
        5 => by_type!(ty, m => m::min(a as _, b as _, c)),
        6 => by_type!(ty, m => m::maxnm(a as _, b as _, c)),
        7 => by_type!(ty, m => m::minnm(a as _, b as _, c)),
        8 => by_type!(ty, m => m::neg(m::mul(a as _, b as _, c))),
        _ => return false,
    };
    wr(e, rd_, r);
    true
}

fn dp3(e: &mut dyn Env, w: u32) -> bool {
    let ty = bits(w, 22, 2);
    if ty > DOUBLE {
        return false;
    }
    let (rm, ra, rn, rd_) = (bits(w, 16, 5), bits(w, 10, 5), bits(w, 5, 5), bits(w, 0, 5));
    let c = fpcr(e);
    let (a, n, m2) = (rd(e, ra, ty), rd(e, rn, ty), rd(e, rm, ty));
    let (neg_a, neg_n) = match (bits(w, 21, 1), bits(w, 15, 1)) {
        (0, 0) => (false, false),
        (0, _) => (false, true),
        (_, 0) => (true, true),
        _ => (true, false),
    };
    let r = by_type!(ty, f => {
        let a = if neg_a { f::neg(a as _) } else { a as _ };
        let n = if neg_n { f::neg(n as _) } else { n as _ };
        f::mul_add(a, n, m2 as _, c)
    });
    wr(e, rd_, r);
    true
}

fn compare(e: &mut dyn Env, w: u32) -> bool {
    let ty = bits(w, 22, 2);
    if ty > DOUBLE || bits(w, 0, 3) != 0 || bits(w, 14, 2) != 0 {
        return false;
    }
    let with_zero = bits(w, 3, 1) == 1;
    let (rm, rn) = (bits(w, 16, 5), bits(w, 5, 5));
    let c = fpcr(e);
    let a = rd(e, rn, ty);
    let b = if with_zero { 0 } else { rd(e, rm, ty) };
    let flags = by_type!(ty, m => m::compare(a as _, b as _, c));
    e.set_nzcv(flags);
    true
}

fn cond_compare(e: &mut dyn Env, w: u32) -> bool {
    let ty = bits(w, 22, 2);
    if ty > DOUBLE {
        return false;
    }
    let (rm, rn) = (bits(w, 16, 5), bits(w, 5, 5));
    let cc = bits(w, 12, 4) as u64;
    if crate::ops::cond(e.nzcv(), cc) == 1 {
        let c = fpcr(e);
        let (a, b) = (rd(e, rn, ty), rd(e, rm, ty));
        let flags = by_type!(ty, m => m::compare(a as _, b as _, c));
        e.set_nzcv(flags);
    } else {
        e.set_nzcv((bits(w, 0, 4) as u64) << 28);
    }
    true
}

fn cond_select(e: &mut dyn Env, w: u32) -> bool {
    let ty = bits(w, 22, 2);
    if ty > DOUBLE {
        return false;
    }
    let (rm, rn, rd_) = (bits(w, 16, 5), bits(w, 5, 5), bits(w, 0, 5));
    let cc = bits(w, 12, 4) as u64;
    let r = if crate::ops::cond(e.nzcv(), cc) == 1 { rd(e, rn, ty) } else { rd(e, rm, ty) };
    wr(e, rd_, r);
    true
}

fn immediate(e: &mut dyn Env, w: u32) -> bool {
    let ty = bits(w, 22, 2);
    if ty > DOUBLE || bits(w, 5, 5) != 0 {
        return false;
    }
    wr(e, bits(w, 0, 5), fp_expand_imm8(bits(w, 13, 8), width(ty)));
    true
}

fn rounding(rmode: u32) -> Round {
    [Round::Nearest, Round::PlusInf, Round::MinusInf, Round::Zero][rmode as usize]
}

fn int_conv(e: &mut dyn Env, w: u32) -> bool {
    let sf = bits(w, 31, 1) == 1;
    let ty = bits(w, 22, 2);
    let rmode = bits(w, 19, 2);
    let opcode = bits(w, 16, 3);
    let (rn, rd_) = (bits(w, 5, 5), bits(w, 0, 5));
    let iw = if sf { 64 } else { 32 };
    let c = fpcr(e);
    match opcode {
        0b110 | 0b111 => {
            let to_fp = opcode == 0b111;
            match (sf, ty, rmode) {
                (false, SINGLE, 0) | (true, DOUBLE, 0) => {
                    if to_fp {
                        let x = e.x(rn) & super::mask(iw);
                        wr(e, rd_, x);
                    } else {
                        let x = rd(e, rn, ty);
                        e.set_x(rd_, x);
                    }
                }
                (true, 2, 1) => {
                    if to_fp {
                        let v = e.v(rd_);
                        let x = e.x(rn);
                        e.set_v(rd_, (v & u64::MAX as u128) | ((x as u128) << 64));
                    } else {
                        let x = (e.v(rn) >> 64) as u64;
                        e.set_x(rd_, x);
                    }
                }
                _ => return false,
            }
            true
        }
        _ if ty > DOUBLE => false,
        0b000 | 0b001 | 0b100 | 0b101 => {
            let signed = opcode & 1 == 0;
            let mode = if opcode & 0b100 != 0 {
                if rmode != 0 {
                    return false;
                }
                Round::Away
            } else {
                rounding(rmode)
            };
            let a = rd(e, rn, ty);
            let r = by_type!(ty, m => m::to_int(a as _, mode, signed, iw, 0, c));
            e.set_x(rd_, r);
            true
        }
        0b010 | 0b011 if rmode == 0 => {
            let x = e.x(rn);
            let r = by_type!(ty, m => m::from_int(x, opcode == 0b010, iw, 0, c));
            wr(e, rd_, r);
            true
        }
        _ => false,
    }
}

fn fixed_conv(e: &mut dyn Env, w: u32) -> bool {
    let sf = bits(w, 31, 1) == 1;
    let ty = bits(w, 22, 2);
    let scale = bits(w, 10, 6);
    let (rn, rd_) = (bits(w, 5, 5), bits(w, 0, 5));
    if ty > DOUBLE || (!sf && scale < 32) {
        return false;
    }
    let fbits = 64 - scale;
    let iw = if sf { 64 } else { 32 };
    let c = fpcr(e);
    match (bits(w, 19, 2), bits(w, 16, 3)) {
        (3, op @ (0 | 1)) => {
            let a = rd(e, rn, ty);
            let r = by_type!(ty, m => m::to_int(a as _, Round::Zero, op == 0, iw, fbits, c));
            e.set_x(rd_, r);
            true
        }
        (0, op @ (2 | 3)) => {
            let x = e.x(rn);
            let r = by_type!(ty, m => m::from_int(x, op == 2, iw, fbits, c));
            wr(e, rd_, r);
            true
        }
        _ => false,
    }
}
