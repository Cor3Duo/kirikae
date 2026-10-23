#![allow(clippy::too_many_arguments)]

use super::fp::{f32b, f64b, Fpcr, Round};
use super::{bits, fp_expand_imm8, lane, mask, sat_s, sat_u, set_lane, set_qc, sized, sx, Env};

macro_rules! by_esize {
    ($es:expr, $m:ident => $body:expr) => {
        if $es == 32 {
            use f32b as $m;
            ($body) as u64
        } else {
            use f64b as $m;
            ($body) as u64
        }
    };
}

fn fpcr(e: &dyn Env) -> Fpcr {
    Fpcr::from_bits(e.fpcr())
}

fn ones(b: bool, esize: u32) -> u64 {
    if b { mask(esize) } else { 0 }
}

#[derive(Clone, Copy)]
struct Shape {
    esize: u32,
    datasize: u32,
}

impl Shape {
    fn elems(self) -> u32 {
        self.datasize / self.esize
    }
}

fn map3(e: &mut dyn Env, rd: u32, rn: u32, rm: u32, s: Shape, mut f: impl FnMut(u64, u64, u64) -> Option<u64>) -> bool {
    let (vn, vm, vd) = (e.v(rn), e.v(rm), e.v(rd));
    let mut out = 0u128;
    for i in 0..s.elems() {
        let Some(r) = f(lane(vn, s.esize, i), lane(vm, s.esize, i), lane(vd, s.esize, i)) else { return false };
        set_lane(&mut out, s.esize, i, r);
    }
    e.set_v(rd, sized(out, s.datasize));
    true
}

fn pairwise(e: &mut dyn Env, rd: u32, rn: u32, rm: u32, s: Shape, f: impl Fn(u64, u64) -> u64) -> bool {
    let (vn, vm) = (e.v(rn), e.v(rm));
    let n = s.elems();
    let get = |k: u32| if k < n { lane(vn, s.esize, k) } else { lane(vm, s.esize, k - n) };
    let mut out = 0u128;
    for i in 0..n {
        set_lane(&mut out, s.esize, i, f(get(2 * i), get(2 * i + 1)));
    }
    e.set_v(rd, sized(out, s.datasize));
    true
}

fn shift_reg(a: u64, shift_byte: u64, esize: u32, signed: bool, round: bool) -> u64 {
    let shift = sx(shift_byte & 0xff, 8);
    let v: i128 = if signed { sx(a, esize) as i128 } else { a as i128 };
    let r = if shift >= 0 {
        if shift >= 128 { 0 } else { v << shift }
    } else {
        let s = (-shift).min(127);
        let rounding = if round { 1i128 << (s - 1) } else { 0 };
        (v + rounding) >> s
    };
    r as u64 & mask(esize)
}

fn int3(u: u32, opcode: u32, size: u32, es: u32, a: u64, b: u64, d: u64, sat: &mut bool) -> Option<u64> {
    let m = mask(es);
    let (sa, sb) = (sx(a, es) as i128, sx(b, es) as i128);
    let (ua, ub) = (a as i128, b as i128);
    Some(match (opcode, u) {
        (0b00000, 0) => ((sa + sb) >> 1) as u64 & m,
        (0b00000, _) => ((ua + ub) >> 1) as u64 & m,
        (0b00001, 0) => sat_s(sa + sb, es, sat),
        (0b00001, _) => sat_u(ua + ub, es, sat),
        (0b00010, 0) => ((sa + sb + 1) >> 1) as u64 & m,
        (0b00010, _) => ((ua + ub + 1) >> 1) as u64 & m,
        (0b00100, 0) => ((sa - sb) >> 1) as u64 & m,
        (0b00100, _) => ((ua - ub) >> 1) as u64 & m,
        (0b00101, 0) => sat_s(sa - sb, es, sat),
        (0b00101, _) => sat_u(ua - ub, es, sat),
        (0b00110, 0) => ones(sa > sb, es),
        (0b00110, _) => ones(ua > ub, es),
        (0b00111, 0) => ones(sa >= sb, es),
        (0b00111, _) => ones(ua >= ub, es),
        (0b01000, 0) => shift_reg(a, b, es, true, false),
        (0b01000, _) => shift_reg(a, b, es, false, false),
        (0b01001 | 0b01011, uns) => {
            let shift = sx(b & 0xff, 8);
            let v = if uns == 0 { sa } else { ua };
            let r = if shift >= 0 {
                if shift >= 64 { v.signum() << 100 } else { v << shift }
            } else {
                let s = (-shift).min(127);
                let round = if opcode == 0b01011 { 1i128 << (s - 1) } else { 0 };
                (v + round) >> s
            };
            if uns == 0 { sat_s(r, es, sat) } else { sat_u(r, es, sat) }
        }
        (0b01010, 0) => shift_reg(a, b, es, true, true),
        (0b01010, _) => shift_reg(a, b, es, false, true),
        (0b01100, 0) => sa.max(sb) as u64 & m,
        (0b01100, _) => ua.max(ub) as u64,
        (0b01101, 0) => sa.min(sb) as u64 & m,
        (0b01101, _) => ua.min(ub) as u64,
        (0b01110, 0) => (sa - sb).unsigned_abs() as u64 & m,
        (0b01110, _) => (ua - ub).unsigned_abs() as u64 & m,
        (0b01111, 0) => (d as i128 + (sa - sb).abs()) as u64 & m,
        (0b01111, _) => (d as i128 + (ua - ub).abs()) as u64 & m,
        (0b10000, 0) => a.wrapping_add(b) & m,
        (0b10000, _) => a.wrapping_sub(b) & m,
        (0b10001, 0) => ones(a & b != 0, es),
        (0b10001, _) => ones(a == b, es),
        (0b10010, 0) => d.wrapping_add(a.wrapping_mul(b)) & m,
        (0b10010, _) => d.wrapping_sub(a.wrapping_mul(b)) & m,
        (0b10011, 0) => a.wrapping_mul(b) & m,
        (0b10011, _) if size == 0 => {
            let mut r = 0u64;
            for i in 0..8 {
                if b >> i & 1 == 1 {
                    r ^= a << i;
                }
            }
            r & 0xff
        }
        (0b10110, 0) if es == 16 || es == 32 => sat_s((2 * sa * sb) >> es, es, sat),
        (0b10110, _) if es == 16 || es == 32 => sat_s((2 * sa * sb + (1 << (es - 1))) >> es, es, sat),
        _ => return None,
    })
}

fn fp3(u: u32, a_bit: u32, opcode: u32, es: u32, x: u64, y: u64, d: u64, c: Fpcr) -> Option<u64> {
    Some(match (u, a_bit, opcode) {
        (0, 0, 0b11000) => by_esize!(es, m => m::maxnm(x as _, y as _, c)),
        (0, 1, 0b11000) => by_esize!(es, m => m::minnm(x as _, y as _, c)),
        (0, 0, 0b11001) => by_esize!(es, m => m::mul_add(d as _, x as _, y as _, c)),
        (0, 1, 0b11001) => by_esize!(es, m => m::mul_add(d as _, m::neg(x as _), y as _, c)),
        (0, 0, 0b11010) => by_esize!(es, m => m::add(x as _, y as _, c)),
        (0, 1, 0b11010) => by_esize!(es, m => m::sub(x as _, y as _, c)),
        (0, 0, 0b11011) => by_esize!(es, m => m::mulx(x as _, y as _, c)),
        (0, 0, 0b11100) => ones(by_esize!(es, m => m::cmp_eq(x as _, y as _, c)) != 0, es),
        (0, 0, 0b11110) => by_esize!(es, m => m::max(x as _, y as _, c)),
        (0, 1, 0b11110) => by_esize!(es, m => m::min(x as _, y as _, c)),
        (0, 0, 0b11111) => by_esize!(es, m => m::recip_step(x as _, y as _, c)),
        (0, 1, 0b11111) => by_esize!(es, m => m::rsqrt_step(x as _, y as _, c)),
        (1, 0, 0b11011) => by_esize!(es, m => m::mul(x as _, y as _, c)),
        (1, 0, 0b11100) => ones(by_esize!(es, m => m::cmp_ge(x as _, y as _, c)) != 0, es),
        (1, 1, 0b11100) => ones(by_esize!(es, m => m::cmp_gt(x as _, y as _, c)) != 0, es),
        (1, 0, 0b11101) => ones(by_esize!(es, m => m::cmp_ge(m::abs(x as _), m::abs(y as _), c)) != 0, es),
        (1, 1, 0b11101) => ones(by_esize!(es, m => m::cmp_gt(m::abs(x as _), m::abs(y as _), c)) != 0, es),
        (1, 1, 0b11010) => by_esize!(es, m => m::abs(m::sub(x as _, y as _, c))),
        (1, 0, 0b11111) => by_esize!(es, m => m::div(x as _, y as _, c)),
        _ => return None,
    })
}

fn fp_pair_op(a_bit: u32, opcode: u32, es: u32, c: Fpcr) -> Option<impl Fn(u64, u64) -> u64> {
    let op = match (a_bit, opcode) {
        (0, 0b11000) => 0,
        (1, 0b11000) => 1,
        (0, 0b11010) => 2,
        (0, 0b11110) => 3,
        (1, 0b11110) => 4,
        _ => return None,
    };
    Some(move |x: u64, y: u64| match op {
        0 => by_esize!(es, m => m::maxnm(x as _, y as _, c)),
        1 => by_esize!(es, m => m::minnm(x as _, y as _, c)),
        2 => by_esize!(es, m => m::add(x as _, y as _, c)),
        3 => by_esize!(es, m => m::max(x as _, y as _, c)),
        _ => by_esize!(es, m => m::min(x as _, y as _, c)),
    })
}

fn three_same(e: &mut dyn Env, w: u32) -> bool {
    let q = bits(w, 30, 1);
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let (rm, opcode, rn, rd) = (bits(w, 16, 5), bits(w, 11, 5), bits(w, 5, 5), bits(w, 0, 5));
    let datasize = 64 << q;
    if opcode == 0b00011 {
        let (n, m, d) = (e.v(rn), e.v(rm), e.v(rd));
        let r = match (u, size) {
            (0, 0) => n & m,
            (0, 1) => n & !m,
            (0, 2) => n | m,
            (0, _) => n | !m,
            (_, 0) => n ^ m,
            (_, 1) => (d & n) | (!d & m),
            (_, 2) => (d & !m) | (n & m),
            _ => (d & m) | (n & !m),
        };
        e.set_v(rd, sized(r, datasize));
        return true;
    }
    if opcode >= 0b11000 {
        let sz = bits(w, 22, 1);
        if sz == 1 && q == 0 {
            return false;
        }
        let es = 32 << sz;
        let a_bit = bits(w, 23, 1);
        let c = fpcr(e);
        let s = Shape { esize: es, datasize };
        if u == 1
            && let Some(f) = fp_pair_op(a_bit, opcode, es, c)
        {
            return pairwise(e, rd, rn, rm, s, f);
        }
        return map3(e, rd, rn, rm, s, |x, y, d| fp3(u, a_bit, opcode, es, x, y, d, c));
    }
    let es = 8 << size;
    if size == 3 && (q == 0 || !matches!(opcode, 0b00001 | 0b00101 | 0b00110 | 0b00111 | 0b01000..=0b01011 | 0b10000 | 0b10001 | 0b10111)) {
        return false;
    }
    if size == 3 && opcode == 0b10111 && u == 1 {
        return false;
    }
    let s = Shape { esize: es, datasize };
    match (opcode, u) {
        (0b10111, 0) => return pairwise(e, rd, rn, rm, s, |x, y| x.wrapping_add(y) & mask(es)),
        (0b10100, _) | (0b10101, _) => {
            let max = opcode == 0b10100;
            return pairwise(e, rd, rn, rm, s, |x, y| {
                let (p, q) = if u == 0 { (sx(x, es) as i128, sx(y, es) as i128) } else { (x as i128, y as i128) };
                (if max { p.max(q) } else { p.min(q) }) as u64 & mask(es)
            });
        }
        _ => {}
    }
    let mut sat = false;
    let ok = map3(e, rd, rn, rm, s, |x, y, d| int3(u, opcode, size, es, x, y, d, &mut sat));
    if sat {
        set_qc(e);
    }
    ok
}

fn narrow(x: u64, es: u32, kind: u32, sat: &mut bool) -> u64 {
    let wide = 2 * es;
    match kind {
        0 => x & mask(es),
        1 => sat_s(sx(x, wide) as i128, es, sat),
        2 => sat_u(x as i128, es, sat),
        _ => sat_u(sx(x, wide) as i128, es, sat),
    }
}

fn write_narrow(e: &mut dyn Env, rd: u32, q: u32, es: u32, vals: &[u64]) {
    let mut out = if q == 1 { e.v(rd) & u64::MAX as u128 } else { 0 };
    let base = if q == 1 { 64 / es } else { 0 };
    for (i, v) in vals.iter().enumerate() {
        set_lane(&mut out, es, base + i as u32, *v);
    }
    e.set_v(rd, out);
}

fn misc(e: &mut dyn Env, w: u32, scalar: bool) -> bool {
    let q = bits(w, 30, 1);
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let opcode = bits(w, 12, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let a_bit = bits(w, 23, 1);
    let datasize = if scalar { 8 << size } else { 64 << q };
    let es = if opcode == 0b00101 { 8 } else { 8 << size };
    let vn = e.v(rn);
    let c = fpcr(e);

    let narrow_kind = match (opcode, u) {
        (0b10010, 0) if !scalar => Some(0),
        (0b10100, 0) => Some(1),
        (0b10100, 1) => Some(2),
        (0b10010, 1) => Some(3),
        _ => None,
    };
    if let Some(kind) = narrow_kind {
        if size == 3 {
            return false;
        }
        let n = if scalar { 1 } else { 64 / es };
        let mut sat = false;
        let vals: Vec<u64> = (0..n).map(|i| narrow(lane(vn, 2 * es, i), es, kind, &mut sat)).collect();
        if scalar {
            e.set_v(rd, vals[0] as u128);
        } else {
            write_narrow(e, rd, q, es, &vals);
        }
        if sat {
            set_qc(e);
        }
        return true;
    }

    if !scalar && (opcode == 0b10110 || opcode == 0b10111) && u == 0 {
        if a_bit == 1 {
            return false;
        }
        let sz = bits(w, 22, 1);
        if opcode == 0b10111 {
            let base = if q == 1 { if sz == 0 { 4 } else { 2 } } else { 0 };
            let mut out = 0u128;
            if sz == 0 {
                for i in 0..4 {
                    set_lane(&mut out, 32, i, super::fp::f16_to_f32(lane(vn, 16, base + i) as u16, c) as u64);
                }
            } else {
                for i in 0..2 {
                    set_lane(&mut out, 64, i, super::fp::f32_to_f64(lane(vn, 32, base + i) as u32, c));
                }
            }
            e.set_v(rd, out);
        } else {
            let vals: Vec<u64> = if sz == 0 {
                (0..4).map(|i| super::fp::f32_to_f16(lane(vn, 32, i) as u32, c) as u64).collect()
            } else {
                (0..2).map(|i| super::fp::f64_to_f32(lane(vn, 64, i), c) as u64).collect()
            };
            write_narrow(e, rd, q, if sz == 0 { 16 } else { 32 }, &vals);
        }
        return true;
    }

    let fp = (opcode >= 0b11000) || ((0b01100..=0b01111).contains(&opcode) && a_bit == 1);
    if fp {
        let sz = bits(w, 22, 1);
        if !scalar && sz == 1 && q == 0 {
            return false;
        }
        let es = 32 << sz;
        let datasize = if scalar { es } else { 64 << q };
        let s = Shape { esize: es, datasize };
        let rounding = |op: u32, a: u32| match (op & 1, a, u) {
            (0, 0, 0) => Round::Nearest,
            (0, 1, _) => Round::PlusInf,
            (1, 0, _) => Round::MinusInf,
            (1, 1, _) => Round::Zero,
            _ => Round::Away,
        };
        let op = |x: u64| -> Option<u64> {
            Some(match (opcode, u, a_bit) {
                (0b01100, 0, 1) => ones(by_esize!(es, m => m::cmp_gt(x as _, 0, c)) != 0, es),
                (0b01100, 1, 1) => ones(by_esize!(es, m => m::cmp_ge(x as _, 0, c)) != 0, es),
                (0b01101, 0, 1) => ones(by_esize!(es, m => m::cmp_eq(x as _, 0, c)) != 0, es),
                (0b01101, 1, 1) => ones(by_esize!(es, m => m::cmp_ge(0, x as _, c)) != 0, es),
                (0b01110, 0, 1) => ones(by_esize!(es, m => m::cmp_gt(0, x as _, c)) != 0, es),
                (0b01111, 0, 1) if !scalar => by_esize!(es, m => m::abs(x as _)),
                (0b01111, 1, 1) if !scalar => by_esize!(es, m => m::neg(x as _)),
                (0b11111, 1, 1) if !scalar => by_esize!(es, m => m::sqrt(x as _, c)),
                (0b11000, 0, a) if !scalar => by_esize!(es, m => m::round_int(x as _, if a == 0 { Round::Nearest } else { Round::PlusInf }, c)),
                (0b11001, 0, a) if !scalar => by_esize!(es, m => m::round_int(x as _, if a == 0 { Round::MinusInf } else { Round::Zero }, c)),
                (0b11000, 1, 0) if !scalar => by_esize!(es, m => m::round_int(x as _, Round::Away, c)),
                (0b11001, 1, _) if !scalar => by_esize!(es, m => m::round_int(x as _, Round::from_rmode(c.rmode), c)),
                (0b11010 | 0b11011, _, a) => {
                    let mode = rounding(opcode, a);
                    let mode = if opcode == 0b11010 && a == 0 { Round::Nearest } else { mode };
                    by_esize!(es, m => m::to_int(x as _, mode, u == 0, es, 0, c))
                }
                (0b11100, _, 0) => by_esize!(es, m => m::to_int(x as _, Round::Away, u == 0, es, 0, c)),
                (0b11101, _, 0) => by_esize!(es, m => m::from_int(x, u == 0, es, 0, c)),
                (0b11101, 0, 1) => by_esize!(es, m => m::recip_estimate(x as _, c)),
                (0b11101, 1, 1) => by_esize!(es, m => m::rsqrt_estimate(x as _, c)),
                (0b11111, 0, 1) if scalar => {
                    by_esize!(es, m => {
                        let v = m::flush(x as _, c);
                        match m::process_nan1(v, c) {
                            Some(n) => n,
                            None => {
                                let exp = v & m::EXP;
                                let new_exp = if exp == 0 { m::EXP - (1 << m::FRAC) } else { !exp & m::EXP };
                                (v & m::SIGN) | new_exp
                            }
                        }
                    })
                }
                _ => return None,
            })
        };
        let mut out = 0u128;
        for i in 0..s.elems() {
            let Some(r) = op(lane(vn, es, i)) else { return false };
            set_lane(&mut out, es, i, r);
        }
        e.set_v(rd, sized(out, datasize));
        return true;
    }

    if !scalar && size == 3 && (q == 0 || matches!(opcode, 0b00100)) && !matches!(opcode, 0b00101) {
        return false;
    }
    if scalar && !matches!(opcode, 0b00011 | 0b00111 | 0b01000 | 0b01001 | 0b01010 | 0b01011) {
        return false;
    }
    if scalar && size != 3 && !matches!(opcode, 0b00011 | 0b00111) {
        return false;
    }
    let s = Shape { esize: es, datasize };
    match (opcode, u) {
        (0b00000, _) | (0b00001, 0) => {
            let container = match (opcode, u) {
                (0b00000, 0) => 64,
                (0b00000, _) => 32,
                _ => 16,
            };
            if es >= container {
                return false;
            }
            let per = container / es;
            let mut out = 0u128;
            for i in 0..s.elems() {
                let base = i / per * per;
                set_lane(&mut out, es, base + (per - 1 - (i - base)), lane(vn, es, i));
            }
            e.set_v(rd, sized(out, datasize));
            return true;
        }
        (0b00010, _) | (0b00110, _) => {
            if size == 3 {
                return false;
            }
            let wide = 2 * es;
            let vd = e.v(rd);
            let mut out = 0u128;
            for i in 0..datasize / wide {
                let (x, y) = (lane(vn, es, 2 * i), lane(vn, es, 2 * i + 1));
                let sum = if u == 0 { (sx(x, es) + sx(y, es)) as u64 } else { x + y };
                let acc = if opcode == 0b00110 { lane(vd, wide, i) } else { 0 };
                set_lane(&mut out, wide, i, sum.wrapping_add(acc));
            }
            e.set_v(rd, sized(out, datasize));
            return true;
        }
        (0b10011, 1) if !scalar => {
            if size == 3 {
                return false;
            }
            let base = if q == 1 { 64 / es } else { 0 };
            let mut out = 0u128;
            for i in 0..64 / es {
                set_lane(&mut out, 2 * es, i, lane(vn, es, base + i) << es);
            }
            e.set_v(rd, out);
            return true;
        }
        _ => {}
    }
    let mut sat = false;
    let vd = e.v(rd);
    let lanes_d: Vec<u64> = (0..s.elems()).map(|i| lane(vd, es, i)).collect();
    let current = std::cell::Cell::new(0usize);
    let d0 = |_: u64| lanes_d[current.get()];
    let f = |x: u64, sat: &mut bool| -> Option<u64> {
        let sxv = sx(x, es) as i128;
        Some(match (opcode, u) {
            (0b00100, 0) => cls(x, es),
            (0b00100, _) => (x.leading_zeros() - (64 - es)) as u64,
            (0b00101, 0) if size == 0 => (x as u8).count_ones() as u64,
            (0b00101, 1) if size == 0 => !x & 0xff,
            (0b00101, 1) if size == 1 => (x as u8).reverse_bits() as u64,
            (0b00011, 0) => sat_s(sx(d0(x), es) as i128 + x as i128, es, sat),
            (0b00011, _) => sat_u(d0(x) as i128 + sxv, es, sat),
            (0b00111, 0) => sat_s(sxv.abs(), es, sat),
            (0b00111, _) => sat_s(-sxv, es, sat),
            (0b01000, 0) => ones(sxv > 0, es),
            (0b01000, _) => ones(sxv >= 0, es),
            (0b01001, 0) => ones(x == 0, es),
            (0b01001, _) => ones(sxv <= 0, es),
            (0b01010, 0) => ones(sxv < 0, es),
            (0b01011, 0) => sxv.unsigned_abs() as u64 & mask(es),
            (0b01011, _) => (sxv.wrapping_neg()) as u64 & mask(es),
            _ => return None,
        })
    };
    let mut out = 0u128;
    for i in 0..s.elems() {
        current.set(i as usize);
        let Some(r) = f(lane(vn, es, i), &mut sat) else { return false };
        set_lane(&mut out, es, i, r);
    }
    e.set_v(rd, sized(out, datasize));
    if sat {
        set_qc(e);
    }
    true
}

fn cls(x: u64, es: u32) -> u64 {
    let v = sx(x, es);
    let t = (v ^ (v >> 63)) as u64;
    (t.leading_zeros() - (64 - es) - 1) as u64
}

fn reduce(vals: &[u64], f: &dyn Fn(u64, u64) -> u64) -> u64 {
    if vals.len() == 1 {
        return vals[0];
    }
    let half = vals.len() / 2;
    f(reduce(&vals[..half], f), reduce(&vals[half..], f))
}

fn across(e: &mut dyn Env, w: u32) -> bool {
    let q = bits(w, 30, 1);
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let opcode = bits(w, 12, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let vn = e.v(rn);
    let datasize = 64 << q;
    if opcode == 0b01100 || opcode == 0b01111 {
        if u != 1 || bits(w, 22, 1) != 0 || q != 1 {
            return false;
        }
        let c = fpcr(e);
        let min = bits(w, 23, 1) == 1;
        let vals: Vec<u64> = (0..4).map(|i| lane(vn, 32, i)).collect();
        let f = move |x: u64, y: u64| -> u64 {
            let (x, y) = (x as u32, y as u32);
            (match (opcode, min) {
                (0b01100, false) => f32b::maxnm(x, y, c),
                (0b01100, true) => f32b::minnm(x, y, c),
                (_, false) => f32b::max(x, y, c),
                _ => f32b::min(x, y, c),
            }) as u64
        };
        e.set_v(rd, reduce(&vals, &f) as u128);
        return true;
    }
    let es = 8 << size;
    if size == 3 || (size == 2 && q == 0) {
        return false;
    }
    let n = datasize / es;
    let vals: Vec<u64> = (0..n).map(|i| lane(vn, es, i)).collect();
    let r = match (opcode, u) {
        (0b00011, 0) => vals.iter().map(|x| sx(*x, es)).sum::<i64>() as u64 & mask(2 * es),
        (0b00011, _) => vals.iter().sum::<u64>() & mask(2 * es),
        (0b11011, 0) => vals.iter().fold(0u64, |a, x| a.wrapping_add(*x)) & mask(es),
        (0b01010, 0) => vals.iter().map(|x| sx(*x, es)).max().unwrap() as u64 & mask(es),
        (0b01010, _) => *vals.iter().max().unwrap(),
        (0b11010, 0) => vals.iter().map(|x| sx(*x, es)).min().unwrap() as u64 & mask(es),
        (0b11010, _) => *vals.iter().min().unwrap(),
        _ => return false,
    };
    e.set_v(rd, r as u128);
    true
}

fn shift_imm(e: &mut dyn Env, w: u32, scalar: bool) -> bool {
    let q = bits(w, 30, 1);
    let u = bits(w, 29, 1);
    let immh = bits(w, 19, 4);
    let immhb = bits(w, 16, 7);
    let opcode = bits(w, 11, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    if immh == 0 {
        return false;
    }
    let es = 8 << (31 - immh.leading_zeros());
    let vn = e.v(rn);
    let vd = e.v(rd);
    let c = fpcr(e);

    if (0b10000..=0b10011).contains(&opcode) {
        if immh >> 3 == 1 {
            return false;
        }
        let shift = 2 * es - immhb;
        let round = opcode & 1 == 1;
        let kind = match (opcode >> 1 & 1, u) {
            (0, 0) => 0,
            (0, _) => 3,
            (_, 0) => 1,
            _ => 2,
        };
        if scalar && kind == 0 {
            return false;
        }
        let wide = 2 * es;
        let n = if scalar { 1 } else { 64 / es };
        let mut sat = false;
        let vals: Vec<u64> = (0..n)
            .map(|i| {
                let x = lane(vn, wide, i);
                let v: i128 = if kind == 2 || (kind == 0 && u == 1) { x as i128 } else { sx(x, wide) as i128 };
                let rb = if round { 1i128 << (shift - 1) } else { 0 };
                let r = (v + rb) >> shift;
                match kind {
                    0 => r as u64 & mask(es),
                    1 => sat_s(r, es, &mut sat),
                    2 => sat_u(r, es, &mut sat),
                    _ => sat_u(r, es, &mut sat),
                }
            })
            .collect();
        if scalar {
            e.set_v(rd, vals[0] as u128);
        } else {
            write_narrow(e, rd, q, es, &vals);
        }
        if sat {
            set_qc(e);
        }
        return true;
    }
    if opcode == 0b10100 {
        if scalar || immh >> 3 == 1 {
            return false;
        }
        let shift = immhb - es;
        let base = if q == 1 { 64 / es } else { 0 };
        let mut out = 0u128;
        for i in 0..64 / es {
            let x = lane(vn, es, base + i);
            let v = if u == 0 { sx(x, es) as u64 } else { x };
            set_lane(&mut out, 2 * es, i, v << shift);
        }
        e.set_v(rd, out);
        return true;
    }

    if !scalar && es == 64 && q == 0 {
        return false;
    }
    let datasize = if scalar { es } else { 64 << q };
    let s = Shape { esize: es, datasize };
    let right = 2 * es - immhb;
    let left = immhb - es;
    if scalar && es != 64 && !matches!(opcode, 0b01100 | 0b01110 | 0b11100 | 0b11111) {
        return false;
    }
    let mut sat = false;
    let op = |x: u64, d: u64, sat: &mut bool| -> Option<u64> {
        let m = mask(es);
        let sv = sx(x, es) as i128;
        let rsh = |v: i128, round: bool| -> i128 {
            let rb = if round { 1i128 << (right - 1) } else { 0 };
            (v + rb) >> right
        };
        Some(match (opcode, u) {
            (0b00000, 0) => rsh(sv, false) as u64 & m,
            (0b00000, _) => rsh(x as i128, false) as u64 & m,
            (0b00010, 0) => d.wrapping_add(rsh(sv, false) as u64) & m,
            (0b00010, _) => d.wrapping_add(rsh(x as i128, false) as u64) & m,
            (0b00100, 0) => rsh(sv, true) as u64 & m,
            (0b00100, _) => rsh(x as i128, true) as u64 & m,
            (0b00110, 0) => d.wrapping_add(rsh(sv, true) as u64) & m,
            (0b00110, _) => d.wrapping_add(rsh(x as i128, true) as u64) & m,
            (0b01000, 1) => {
                let keep = if right >= es { m } else { !(m >> right) & m };
                (d & keep) | if right >= es { 0 } else { x >> right }
            }
            (0b01010, 0) => (x << left) & m,
            (0b01010, _) => {
                let keep = (1u64 << left) - 1;
                (d & keep) | ((x << left) & m)
            }
            (0b01100, 1) => sat_u(sv << left, es, sat),
            (0b01110, 0) => sat_s(sv << left, es, sat),
            (0b01110, _) => sat_u((x as i128) << left, es, sat),
            (0b11100, _) if es >= 32 => by_esize!(es, m => m::from_int(x, u == 0, es, right, c)),
            (0b11111, _) if es >= 32 => by_esize!(es, m => m::to_int(x as _, Round::Zero, u == 0, es, right, c)),
            _ => return None,
        })
    };
    let mut out = 0u128;
    for i in 0..s.elems() {
        let Some(r) = op(lane(vn, es, i), lane(vd, es, i), &mut sat) else { return false };
        set_lane(&mut out, es, i, r);
    }
    e.set_v(rd, sized(out, datasize));
    if sat {
        set_qc(e);
    }
    true
}

fn three_diff(e: &mut dyn Env, w: u32, scalar: bool) -> bool {
    let q = bits(w, 30, 1);
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let (rm, opcode, rn, rd) = (bits(w, 16, 5), bits(w, 12, 4), bits(w, 5, 5), bits(w, 0, 5));
    let (vn, vm, vd) = (e.v(rn), e.v(rm), e.v(rd));
    if opcode == 0b1110 && u == 0 && size == 3 && !scalar {
        let k = if q == 1 { 1 } else { 0 };
        e.set_v(rd, pmull64(lane(vn, 64, k), lane(vm, 64, k)));
        return true;
    }
    if size == 3 {
        return false;
    }
    if scalar && !(u == 0 && matches!(opcode, 0b1001 | 0b1011 | 0b1101) && (size == 1 || size == 2)) {
        return false;
    }
    let es = 8 << size;
    let wide = 2 * es;
    let n = if scalar { 1 } else { 64 / es };
    let base = if q == 1 && !scalar { n } else { 0 };
    let ext = |x: u64| -> i128 { if u == 0 { sx(x, es) as i128 } else { x as i128 } };
    let ext_w = |x: u64| -> i128 { if u == 0 { sx(x, wide) as i128 } else { x as i128 } };

    if opcode == 0b0100 || opcode == 0b0110 {
        let vals: Vec<u64> = (0..n)
            .map(|i| {
                let (a, b) = (lane(vn, wide, i), lane(vm, wide, i));
                let mut r = if opcode == 0b0100 { a.wrapping_add(b) } else { a.wrapping_sub(b) } & mask(wide);
                if u == 1 {
                    r = r.wrapping_add(1 << (es - 1)) & mask(wide);
                }
                r >> es
            })
            .collect();
        write_narrow(e, rd, q, es, &vals);
        return true;
    }
    let wide_first = opcode == 0b0001 || opcode == 0b0011;
    let mut sat = false;
    let mut out = 0u128;
    for i in 0..n {
        let a = if wide_first { lane(vn, wide, i) } else { lane(vn, es, base + i) };
        let b = lane(vm, es, base + i);
        let d = lane(vd, wide, i);
        let r: u64 = match opcode {
            0b0000 => (ext(a) + ext(b)) as u64,
            0b0001 => (ext_w(a) + ext(b)) as u64,
            0b0010 => (ext(a) - ext(b)) as u64,
            0b0011 => (ext_w(a) - ext(b)) as u64,
            0b0101 => d.wrapping_add((ext(a) - ext(b)).unsigned_abs() as u64),
            0b0111 => (ext(a) - ext(b)).unsigned_abs() as u64,
            0b1000 => d.wrapping_add((ext(a) * ext(b)) as u64),
            0b1010 => d.wrapping_sub((ext(a) * ext(b)) as u64),
            0b1100 => (ext(a) * ext(b)) as u64,
            0b1001 | 0b1011 | 0b1101 if u == 0 && (es == 16 || es == 32) => {
                let prod = sat_s(2 * ext(a) * ext(b), wide, &mut sat);
                match opcode {
                    0b1101 => prod,
                    0b1001 => sat_s(sx(d, wide) as i128 + sx(prod, wide) as i128, wide, &mut sat),
                    _ => sat_s(sx(d, wide) as i128 - sx(prod, wide) as i128, wide, &mut sat),
                }
            }
            0b1110 if u == 0 && es == 8 => {
                let mut r = 0u64;
                for k in 0..8 {
                    if b >> k & 1 == 1 {
                        r ^= a << k;
                    }
                }
                r
            }
            _ => return false,
        };
        set_lane(&mut out, wide, i, r & mask(wide));
    }
    e.set_v(rd, out);
    if sat {
        set_qc(e);
    }
    true
}

fn pmull64(a: u64, b: u64) -> u128 {
    let mut r = 0u128;
    for k in 0..64 {
        if b >> k & 1 == 1 {
            r ^= (a as u128) << k;
        }
    }
    r
}

fn indexed(e: &mut dyn Env, w: u32, scalar: bool) -> bool {
    let q = bits(w, 30, 1);
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let (l, mbit, h) = (bits(w, 21, 1), bits(w, 20, 1), bits(w, 11, 1));
    let opcode = bits(w, 12, 4);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let fp_op = matches!((opcode, u), (0b0001, 0) | (0b0101, 0) | (0b1001, 0) | (0b1001, 1));
    if scalar && !fp_op && !(u == 0 && matches!(opcode, 0b0011 | 0b0111 | 0b1011 | 0b1100 | 0b1101)) {
        return false;
    }
    let c = fpcr(e);
    if fp_op {
        let sz = bits(w, 22, 1);
        if size >> 1 != 1 || (sz == 1 && l == 1) || (!scalar && sz == 1 && q == 0) {
            return false;
        }
        let es = 32 << sz;
        let index = if sz == 0 { h << 1 | l } else { h };
        let rm = mbit << 4 | bits(w, 16, 4);
        let elem = lane(e.v(rm), es, index);
        let datasize = if scalar { es } else { 64 << q };
        let s = Shape { esize: es, datasize };
        return map3(e, rd, rn, rn, s, |x, _, d| {
            Some(match (opcode, u) {
                (0b0001, _) => by_esize!(es, m => m::mul_add(d as _, x as _, elem as _, c)),
                (0b0101, _) => by_esize!(es, m => m::mul_add(d as _, m::neg(x as _), elem as _, c)),
                (0b1001, 0) => by_esize!(es, m => m::mul(x as _, elem as _, c)),
                _ => by_esize!(es, m => m::mulx(x as _, elem as _, c)),
            })
        });
    }
    let (es, index, rm) = match size {
        1 => (16, h << 2 | l << 1 | mbit, bits(w, 16, 4)),
        2 => (32, h << 1 | l, mbit << 4 | bits(w, 16, 4)),
        _ => return false,
    };
    let elem = lane(e.v(rm), es, index);
    let long = matches!(opcode, 0b0010 | 0b0011 | 0b0110 | 0b0111 | 0b1010 | 0b1011);
    if long {
        let wide = 2 * es;
        let n = if scalar { 1 } else { 64 / es };
        let base = if q == 1 && !scalar { n } else { 0 };
        let (vn, vd) = (e.v(rn), e.v(rd));
        let ext = |x: u64| -> i128 { if u == 0 { sx(x, es) as i128 } else { x as i128 } };
        let mut sat = false;
        let mut out = 0u128;
        for i in 0..n {
            let a = lane(vn, es, base + i);
            let d = lane(vd, wide, i);
            let r = match (opcode, u) {
                (0b0010, _) => d.wrapping_add((ext(a) * ext(elem)) as u64),
                (0b0110, _) => d.wrapping_sub((ext(a) * ext(elem)) as u64),
                (0b1010, _) => (ext(a) * ext(elem)) as u64,
                (0b0011 | 0b0111 | 0b1011, 0) => {
                    let p = sat_s(2 * ext(a) * ext(elem), wide, &mut sat);
                    match opcode {
                        0b1011 => p,
                        0b0011 => sat_s(sx(d, wide) as i128 + sx(p, wide) as i128, wide, &mut sat),
                        _ => sat_s(sx(d, wide) as i128 - sx(p, wide) as i128, wide, &mut sat),
                    }
                }
                _ => return false,
            };
            set_lane(&mut out, wide, i, r & mask(wide));
        }
        e.set_v(rd, out);
        if sat {
            set_qc(e);
        }
        return true;
    }
    let datasize = if scalar { es } else { 64 << q };
    let s = Shape { esize: es, datasize };
    let mut sat = false;
    let ok = map3(e, rd, rn, rn, s, |x, _, d| {
        let (sa, sb) = (sx(x, es) as i128, sx(elem, es) as i128);
        Some(match (opcode, u) {
            (0b1000, 0) if !scalar => x.wrapping_mul(elem) & mask(es),
            (0b0000, 1) if !scalar => d.wrapping_add(x.wrapping_mul(elem)) & mask(es),
            (0b0100, 1) if !scalar => d.wrapping_sub(x.wrapping_mul(elem)) & mask(es),
            (0b1100, 0) => sat_s((2 * sa * sb) >> es, es, &mut sat),
            (0b1101, 0) => sat_s((2 * sa * sb + (1 << (es - 1))) >> es, es, &mut sat),
            _ => return None,
        })
    });
    if sat {
        set_qc(e);
    }
    ok
}

fn permute(e: &mut dyn Env, w: u32) -> bool {
    let q = bits(w, 30, 1);
    let size = bits(w, 22, 2);
    let (rm, opcode, rn, rd) = (bits(w, 16, 5), bits(w, 12, 3), bits(w, 5, 5), bits(w, 0, 5));
    if size == 3 && q == 0 {
        return false;
    }
    let es = 8 << size;
    let datasize = 64 << q;
    let n = datasize / es;
    let (vn, vm) = (e.v(rn), e.v(rm));
    let part = (opcode >> 2) & 1;
    let mut out = 0u128;
    for i in 0..n {
        let r = match opcode & 3 {
            1 => {
                let k = 2 * i + part;
                if k < n { lane(vn, es, k) } else { lane(vm, es, k - n) }
            }
            2 => {
                let pair = i / 2;
                if i % 2 == 0 { lane(vn, es, 2 * pair + part) } else { lane(vm, es, 2 * pair + part) }
            }
            3 => {
                let base = part * (n / 2);
                if i % 2 == 0 { lane(vn, es, base + i / 2) } else { lane(vm, es, base + i / 2) }
            }
            _ => return false,
        };
        set_lane(&mut out, es, i, r);
    }
    e.set_v(rd, sized(out, datasize));
    true
}

fn ext(e: &mut dyn Env, w: u32) -> bool {
    let q = bits(w, 30, 1);
    let imm4 = bits(w, 11, 4);
    let (rm, rn, rd) = (bits(w, 16, 5), bits(w, 5, 5), bits(w, 0, 5));
    if q == 0 && imm4 >> 3 == 1 {
        return false;
    }
    let bytes = 8 << q;
    let (vn, vm) = (e.v(rn), e.v(rm));
    let mut out = 0u128;
    for i in 0..bytes {
        let k = i + imm4;
        let b = if k < bytes { lane(vn, 8, k) } else { lane(vm, 8, k - bytes) };
        set_lane(&mut out, 8, i, b);
    }
    e.set_v(rd, out);
    true
}

fn tbl(e: &mut dyn Env, w: u32) -> bool {
    let q = bits(w, 30, 1);
    if bits(w, 22, 2) != 0 {
        return false;
    }
    let (rm, len, tbx, rn, rd) = (bits(w, 16, 5), bits(w, 13, 2), bits(w, 12, 1), bits(w, 5, 5), bits(w, 0, 5));
    let regs = len + 1;
    let table: Vec<u128> = (0..regs).map(|k| e.v((rn + k) % 32)).collect();
    let (idx, vd) = (e.v(rm), e.v(rd));
    let bytes = 8 << q;
    let mut out = 0u128;
    for i in 0..bytes {
        let j = lane(idx, 8, i) as u32;
        let b = if j < 16 * regs { lane(table[(j / 16) as usize], 8, j % 16) } else if tbx == 1 { lane(vd, 8, i) } else { 0 };
        set_lane(&mut out, 8, i, b);
    }
    e.set_v(rd, sized(out, 64 << q));
    true
}

fn imm5_elem(imm5: u32) -> Option<(u32, u32)> {
    let size = imm5.trailing_zeros();
    (size <= 3).then(|| (8 << size, imm5 >> (size + 1)))
}

fn copy(e: &mut dyn Env, w: u32) -> bool {
    let q = bits(w, 30, 1);
    let op = bits(w, 29, 1);
    let imm5 = bits(w, 16, 5);
    let imm4 = bits(w, 11, 4);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let Some((es, index)) = imm5_elem(imm5) else { return false };
    let datasize = 64 << q;
    match (op, imm4) {
        (0, 0b0000) => {
            if es == 64 && q == 0 {
                return false;
            }
            let x = lane(e.v(rn), es, index);
            let mut out = 0u128;
            for i in 0..datasize / es {
                set_lane(&mut out, es, i, x);
            }
            e.set_v(rd, out);
        }
        (0, 0b0001) => {
            if es == 64 && q == 0 {
                return false;
            }
            let x = e.x(rn);
            let mut out = 0u128;
            for i in 0..datasize / es {
                set_lane(&mut out, es, i, x);
            }
            e.set_v(rd, out);
        }
        (0, 0b0011) if q == 1 => {
            let mut v = e.v(rd);
            set_lane(&mut v, es, index, e.x(rn));
            e.set_v(rd, v);
        }
        (0, 0b0101) | (0, 0b0111) => {
            let x = lane(e.v(rn), es, index);
            let signed = imm4 == 0b0101;
            let r = match (signed, q) {
                (true, 0) if es < 32 => sx(x, es) as u64 & 0xffff_ffff,
                (true, 1) if es < 64 => sx(x, es) as u64,
                (false, 0) if es < 64 => x,
                (false, 1) if es == 64 => x,
                _ => return false,
            };
            e.set_x(rd, r);
        }
        (1, _) if q == 1 => {
            let src = imm4 >> (es.trailing_zeros() - 3);
            let x = lane(e.v(rn), es, src);
            let mut v = e.v(rd);
            set_lane(&mut v, es, index, x);
            e.set_v(rd, v);
        }
        _ => return false,
    }
    true
}

fn modified_imm(e: &mut dyn Env, w: u32) -> bool {
    let q = bits(w, 30, 1);
    let op = bits(w, 29, 1);
    let cmode = bits(w, 12, 4);
    let o2 = bits(w, 11, 1);
    let rd = bits(w, 0, 5);
    let imm8 = (bits(w, 16, 3) << 5) | bits(w, 5, 5);
    if o2 == 1 {
        return false;
    }
    let i = imm8 as u64;
    let rep32 = |x: u64| x | (x << 32);
    let rep16 = |x: u64| rep32(x | (x << 16));
    let (imm, kind) = match cmode {
        0b0000 | 0b0010 | 0b0100 | 0b0110 => (rep32(i << (8 * (cmode >> 1))), 0),
        0b0001 | 0b0011 | 0b0101 | 0b0111 => (rep32(i << (8 * (cmode >> 1))), 1),
        0b1000 | 0b1010 => (rep16(i << (8 * ((cmode >> 1) & 1))), 0),
        0b1001 | 0b1011 => (rep16(i << (8 * ((cmode >> 1) & 1))), 1),
        0b1100 => (rep32((i << 8) | 0xff), 0),
        0b1101 => (rep32((i << 16) | 0xffff), 0),
        0b1110 if op == 0 => (i * 0x0101_0101_0101_0101, 2),
        0b1110 => {
            let mut x = 0u64;
            for b in 0..8 {
                if i >> b & 1 == 1 {
                    x |= 0xff << (8 * b);
                }
            }
            (x, 2)
        }
        0b1111 if op == 0 => (rep32(fp_expand_imm8(imm8, 32)), 2),
        0b1111 if q == 1 => (fp_expand_imm8(imm8, 64), 2),
        _ => return false,
    };
    let imm128 = (imm as u128) | ((imm as u128) << 64);
    let datasize = 64 << q;
    let r = match (kind, op) {
        (0, 1) => !imm128,
        (1, 0) => e.v(rd) | imm128,
        (1, _) => e.v(rd) & !imm128,
        _ => imm128,
    };
    e.set_v(rd, sized(r, datasize));
    true
}

fn scalar_copy(e: &mut dyn Env, w: u32) -> bool {
    if bits(w, 29, 1) != 0 || bits(w, 11, 4) != 0 {
        return false;
    }
    let Some((es, index)) = imm5_elem(bits(w, 16, 5)) else { return false };
    let x = lane(e.v(bits(w, 5, 5)), es, index);
    e.set_v(bits(w, 0, 5), x as u128);
    true
}

fn scalar_pairwise(e: &mut dyn Env, w: u32) -> bool {
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let opcode = bits(w, 12, 5);
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let vn = e.v(rn);
    if u == 0 {
        if opcode == 0b11011 && size == 3 {
            e.set_v(rd, lane(vn, 64, 0).wrapping_add(lane(vn, 64, 1)) as u128);
            return true;
        }
        return false;
    }
    let es = 32 << bits(w, 22, 1);
    let c = fpcr(e);
    let opcode = match opcode {
        0b01100 => 0b11000,
        0b01101 => 0b11010,
        0b01111 => 0b11110,
        _ => return false,
    };
    let Some(f) = fp_pair_op(bits(w, 23, 1), opcode, es, c) else { return false };
    e.set_v(rd, f(lane(vn, es, 0), lane(vn, es, 1)) as u128);
    true
}

fn scalar_three_same(e: &mut dyn Env, w: u32) -> bool {
    let u = bits(w, 29, 1);
    let size = bits(w, 22, 2);
    let (rm, opcode, rn, rd) = (bits(w, 16, 5), bits(w, 11, 5), bits(w, 5, 5), bits(w, 0, 5));
    let (a, b) = (e.v(rn), e.v(rm));
    if opcode >= 0b11000 {
        let es = 32 << bits(w, 22, 1);
        let a_bit = bits(w, 23, 1);
        let allowed = matches!((u, a_bit, opcode), (0, 0, 0b11011) | (0, 0, 0b11100) | (0, _, 0b11111) | (1, _, 0b11100) | (1, _, 0b11101) | (1, 1, 0b11010));
        if !allowed {
            return false;
        }
        let c = fpcr(e);
        let Some(r) = fp3(u, a_bit, opcode, es, lane(a, es, 0), lane(b, es, 0), 0, c) else { return false };
        e.set_v(rd, r as u128);
        return true;
    }
    let es = 8 << size;
    let saturating = matches!(opcode, 0b00001 | 0b00101 | 0b01001 | 0b01011 | 0b10110);
    if !saturating && size != 3 {
        return false;
    }
    if !matches!(opcode, 0b00001 | 0b00101 | 0b00110 | 0b00111 | 0b01000 | 0b01001 | 0b01010 | 0b01011 | 0b10000 | 0b10001 | 0b10110) {
        return false;
    }
    let mut sat = false;
    let Some(r) = int3(u, opcode, size, es, lane(a, es, 0), lane(b, es, 0), 0, &mut sat) else { return false };
    e.set_v(rd, r as u128);
    if sat {
        set_qc(e);
    }
    true
}

pub fn exec(e: &mut dyn Env, w: u32) -> bool {
    if w & 0xdfe0_8400 == 0x5e00_0400 {
        return scalar_copy(e, w);
    }
    if w & 0xdf3e_0c00 == 0x5e30_0800 {
        return scalar_pairwise(e, w);
    }
    if w & 0xdf3e_0c00 == 0x5e20_0800 {
        return misc(e, w, true);
    }
    if w & 0xdf20_0c00 == 0x5e20_0000 {
        return three_diff(e, w, true);
    }
    if w & 0xdf20_0400 == 0x5e20_0400 {
        return scalar_three_same(e, w);
    }
    if w & 0xdf80_0400 == 0x5f00_0400 {
        return shift_imm(e, w, true);
    }
    if w & 0xdf00_0400 == 0x5f00_0000 {
        return indexed(e, w, true);
    }
    if w & 0xbf20_8c00 == 0x0e00_0000 {
        return tbl(e, w);
    }
    if w & 0xbf20_8c00 == 0x0e00_0800 {
        return permute(e, w);
    }
    if w & 0xbfe0_8400 == 0x2e00_0000 {
        return ext(e, w);
    }
    if w & 0x9fe0_8400 == 0x0e00_0400 {
        return copy(e, w);
    }
    if w & 0x9ff8_0400 == 0x0f00_0400 {
        return modified_imm(e, w);
    }
    if w & 0x9f80_0400 == 0x0f00_0400 {
        return shift_imm(e, w, false);
    }
    if w & 0x9f3e_0c00 == 0x0e30_0800 {
        return across(e, w);
    }
    if w & 0x9f3e_0c00 == 0x0e20_0800 {
        return misc(e, w, false);
    }
    if w & 0x9f20_0c00 == 0x0e20_0000 {
        return three_diff(e, w, false);
    }
    if w & 0x9f20_0400 == 0x0e20_0400 {
        return three_same(e, w);
    }
    if w & 0x9f00_0400 == 0x0f00_0000 {
        return indexed(e, w, false);
    }
    false
}
