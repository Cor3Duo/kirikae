#[derive(Clone, Copy, Debug, Default)]
pub struct Fpcr {
    pub dn: bool,
    pub fz: bool,
    pub rmode: u32,
}

impl Fpcr {
    pub fn from_bits(fpcr: u64) -> Fpcr {
        Fpcr { dn: fpcr >> 25 & 1 == 1, fz: fpcr >> 24 & 1 == 1, rmode: (fpcr >> 22 & 3) as u32 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    Nearest,
    Away,
    PlusInf,
    MinusInf,
    Zero,
}

impl Round {
    pub fn from_rmode(rmode: u32) -> Round {
        [Round::Nearest, Round::PlusInf, Round::MinusInf, Round::Zero][rmode as usize & 3]
    }
}

macro_rules! fp_format {
    ($m:ident, $f:ty, $u:ty, $i:ty, $bits:expr, $frac:expr) => {
        pub mod $m {
            use super::{Fpcr, Round};

            pub const BITS: u32 = $bits;
            pub const FRAC: u32 = $frac;
            pub const SIGN: $u = 1 << (BITS - 1);
            pub const EXP: $u = ((1 << (BITS - 1 - FRAC)) - 1) << FRAC;
            pub const FRAC_MASK: $u = (1 << FRAC) - 1;
            pub const QUIET: $u = 1 << (FRAC - 1);
            pub const DEFAULT_NAN: $u = EXP | QUIET;
            pub const INF: $u = EXP;
            pub const ONE: $u = (1.0 as $f).to_bits();

            pub fn f(b: $u) -> $f {
                <$f>::from_bits(b)
            }
            pub fn b(x: $f) -> $u {
                x.to_bits()
            }
            pub fn is_nan(x: $u) -> bool {
                x & EXP == EXP && x & FRAC_MASK != 0
            }
            pub fn is_snan(x: $u) -> bool {
                is_nan(x) && x & QUIET == 0
            }
            pub fn is_inf(x: $u) -> bool {
                x & !SIGN == INF
            }
            pub fn is_zero(x: $u) -> bool {
                x & !SIGN == 0
            }
            pub fn is_denormal(x: $u) -> bool {
                x & EXP == 0 && x & FRAC_MASK != 0
            }
            pub fn neg(x: $u) -> $u {
                x ^ SIGN
            }
            pub fn abs(x: $u) -> $u {
                x & !SIGN
            }

            pub fn flush(x: $u, c: Fpcr) -> $u {
                if c.fz && is_denormal(x) { x & SIGN } else { x }
            }

            fn nan_result(x: $u, c: Fpcr) -> $u {
                if c.dn { DEFAULT_NAN } else { x | QUIET }
            }

            pub fn process_nan1(a: $u, c: Fpcr) -> Option<$u> {
                is_nan(a).then(|| nan_result(a, c))
            }

            pub fn process_nans(ops: &[$u], c: Fpcr) -> Option<$u> {
                ops.iter().find(|x| is_snan(**x)).or_else(|| ops.iter().find(|x| is_nan(**x))).map(|x| nan_result(*x, c))
            }

            fn finish(r: $f, err: f64, c: Fpcr) -> $u {
                let rb = b(r);
                if is_nan(rb) {
                    return DEFAULT_NAN;
                }
                if c.fz {
                    let min = <$f>::MIN_POSITIVE;
                    let ar = r.abs();
                    let below = ar < min && ar != 0.0;
                    let rounded_up = ar == min && ((r > 0.0 && err < 0.0) || (r < 0.0 && err > 0.0));
                    if below || rounded_up {
                        return rb & SIGN;
                    }
                }
                rb
            }

            fn binary(a: $u, b2: $u, c: Fpcr, op: impl Fn($f, $f) -> ($f, $f)) -> $u {
                let (a, b2) = (flush(a, c), flush(b2, c));
                if let Some(n) = process_nans(&[a, b2], c) {
                    return n;
                }
                let (r, err) = op(f(a), f(b2));
                finish(r, err as f64, c)
            }

            fn sum_err(p: $f, q: $f, s: $f) -> $f {
                let bb = s - p;
                (p - (s - bb)) + (q - bb)
            }

            pub fn add(a: $u, x: $u, c: Fpcr) -> $u {
                binary(a, x, c, |p, q| {
                    let s = p + q;
                    (s, if s.is_finite() { sum_err(p, q, s) } else { 0.0 })
                })
            }
            pub fn sub(a: $u, x: $u, c: Fpcr) -> $u {
                binary(a, x, c, |p, q| {
                    let s = p - q;
                    (s, if s.is_finite() { sum_err(p, -q, s) } else { 0.0 })
                })
            }
            pub fn mul(a: $u, x: $u, c: Fpcr) -> $u {
                binary(a, x, c, |p, q| {
                    let m = p * q;
                    (m, if m.is_finite() { p.mul_add(q, -m) } else { 0.0 })
                })
            }
            pub fn div(a: $u, x: $u, c: Fpcr) -> $u {
                binary(a, x, c, |p, q| {
                    let d = p / q;
                    let rem = if d.is_finite() && q != 0.0 { (-d).mul_add(q, p) } else { 0.0 };
                    (d, if q < 0.0 { -rem } else { rem })
                })
            }

            pub fn mulx(a: $u, x: $u, c: Fpcr) -> $u {
                let (fa, fx) = (flush(a, c), flush(x, c));
                if process_nans(&[fa, fx], c).is_none() && ((is_inf(fa) && is_zero(fx)) || (is_zero(fa) && is_inf(fx))) {
                    return b(2.0) | ((fa ^ fx) & SIGN);
                }
                mul(a, x, c)
            }

            pub fn sqrt(a: $u, c: Fpcr) -> $u {
                let a = flush(a, c);
                if let Some(n) = process_nan1(a, c) {
                    return n;
                }
                finish(f(a).sqrt(), 0.0, c)
            }

            pub fn mul_add(addend: $u, op1: $u, op2: $u, c: Fpcr) -> $u {
                let (a, x, y) = (flush(addend, c), flush(op1, c), flush(op2, c));
                let inf_times_zero = (is_inf(x) && is_zero(y)) || (is_zero(x) && is_inf(y));
                if is_nan(a) && !is_snan(a) && !is_snan(x) && !is_snan(y) && inf_times_zero {
                    return DEFAULT_NAN;
                }
                if let Some(n) = process_nans(&[a, x, y], c) {
                    return n;
                }
                let r = f(x).mul_add(f(y), f(a));
                finish(r, super::fma_err(f(x) as f64, f(y) as f64, f(a) as f64, r as f64), c)
            }

            pub fn recip_step(a: $u, x: $u, c: Fpcr) -> $u {
                let (fa, fx) = (flush(a, c), flush(x, c));
                if process_nans(&[fa, fx], c).is_none() && ((is_inf(fa) && is_zero(fx)) || (is_zero(fa) && is_inf(fx))) {
                    return b(2.0);
                }
                mul_add(b(2.0), neg(fa), fx, c)
            }

            pub fn rsqrt_step(a: $u, x: $u, c: Fpcr) -> $u {
                let (fa, fx) = (neg(flush(a, c)), flush(x, c));
                if let Some(n) = process_nans(&[fa, fx], c) {
                    return n;
                }
                if (is_inf(fa) && is_zero(fx)) || (is_zero(fa) && is_inf(fx)) {
                    return b(1.5);
                }
                finish(f(fa).mul_add(f(fx), 3.0) / 2.0, 0.0, c)
            }

            fn max_core(a: $u, x: $u, want_max: bool) -> $u {
                if is_zero(a) && is_zero(x) {
                    return if want_max { a & x } else { a | x };
                }
                let gt = f(a) > f(x);
                if gt == want_max { a } else { x }
            }

            pub fn max(a: $u, x: $u, c: Fpcr) -> $u {
                let (a, x) = (flush(a, c), flush(x, c));
                process_nans(&[a, x], c).unwrap_or_else(|| flush(max_core(a, x, true), c))
            }
            pub fn min(a: $u, x: $u, c: Fpcr) -> $u {
                let (a, x) = (flush(a, c), flush(x, c));
                process_nans(&[a, x], c).unwrap_or_else(|| flush(max_core(a, x, false), c))
            }

            fn num(a: $u, x: $u, c: Fpcr, want_max: bool) -> $u {
                let (mut a, mut x) = (flush(a, c), flush(x, c));
                let loser = if want_max { INF | SIGN } else { INF };
                let quiet = |v: $u| is_nan(v) && !is_snan(v);
                if quiet(a) && !is_nan(x) {
                    a = loser;
                } else if quiet(x) && !is_nan(a) {
                    x = loser;
                }
                if want_max { max(a, x, c) } else { min(a, x, c) }
            }
            pub fn maxnm(a: $u, x: $u, c: Fpcr) -> $u {
                num(a, x, c, true)
            }
            pub fn minnm(a: $u, x: $u, c: Fpcr) -> $u {
                num(a, x, c, false)
            }

            pub fn compare(a: $u, x: $u, c: Fpcr) -> u64 {
                let (a, x) = (flush(a, c), flush(x, c));
                if is_nan(a) || is_nan(x) {
                    0b0011 << 28
                } else if f(a) == f(x) {
                    0b0110 << 28
                } else if f(a) < f(x) {
                    0b1000 << 28
                } else {
                    0b0010 << 28
                }
            }

            pub fn cmp_eq(a: $u, x: $u, c: Fpcr) -> bool {
                f(flush(a, c)) == f(flush(x, c))
            }
            pub fn cmp_ge(a: $u, x: $u, c: Fpcr) -> bool {
                f(flush(a, c)) >= f(flush(x, c))
            }
            pub fn cmp_gt(a: $u, x: $u, c: Fpcr) -> bool {
                f(flush(a, c)) > f(flush(x, c))
            }

            fn round_float(x: $f, mode: Round) -> $f {
                match mode {
                    Round::Nearest => x.round_ties_even(),
                    Round::Away => x.round(),
                    Round::PlusInf => x.ceil(),
                    Round::MinusInf => x.floor(),
                    Round::Zero => x.trunc(),
                }
            }

            pub fn round_int(a: $u, mode: Round, c: Fpcr) -> $u {
                let a = flush(a, c);
                if let Some(n) = process_nan1(a, c) {
                    return n;
                }
                b(round_float(f(a), mode))
            }

            pub fn to_int(a: $u, mode: Round, signed: bool, width: u32, fbits: u32, c: Fpcr) -> u64 {
                let a = flush(a, c);
                if is_nan(a) {
                    return 0;
                }
                let scaled = f(a) * (2.0 as $f).powi(fbits as i32);
                let r = round_float(scaled, mode);
                match (signed, width) {
                    (true, 64) => r as i64 as u64,
                    (true, _) => r as i32 as u32 as u64,
                    (false, 64) => r as u64,
                    (false, _) => r as u32 as u64,
                }
            }

            pub fn from_int(v: u64, signed: bool, width: u32, fbits: u32, c: Fpcr) -> $u {
                let x: $f = match (signed, width) {
                    (true, 64) => v as i64 as $f,
                    (true, _) => v as u32 as i32 as $f,
                    (false, 64) => v as $f,
                    (false, _) => v as u32 as $f,
                };
                flush(b(x / (2.0 as $f).powi(fbits as i32)), c)
            }

            fn max_normal(sign: $u) -> $u {
                sign | (EXP - (1 << FRAC)) | FRAC_MASK
            }

            fn fraction52(x: $u) -> (u64, i64) {
                (((x & FRAC_MASK) as u64) << (52 - FRAC), ((x & EXP) >> FRAC) as i64)
            }

            pub fn recip_estimate(a: $u, c: Fpcr) -> $u {
                let a = flush(a, c);
                if let Some(n) = process_nan1(a, c) {
                    return n;
                }
                let sign = a & SIGN;
                if is_inf(a) {
                    return sign;
                }
                if is_zero(a) {
                    return sign | INF;
                }
                let value = (f(abs(a)) as f64).abs();
                let (tiny, huge) = if BITS == 32 { (2f64.powi(-128), 2f64.powi(126)) } else { (2f64.powi(-1024), 2f64.powi(1022)) };
                if value < tiny {
                    let overflow_to_inf = match c.rmode {
                        0 => true,
                        1 => sign == 0,
                        2 => sign != 0,
                        _ => false,
                    };
                    return if overflow_to_inf { sign | INF } else { max_normal(sign) };
                }
                if c.fz && value >= huge {
                    return sign;
                }
                let (mut fraction, mut exp) = fraction52(a);
                if exp == 0 {
                    if fraction >> 51 & 1 == 0 {
                        exp = -1;
                        fraction = (fraction << 2) & ((1 << 52) - 1);
                    } else {
                        fraction = (fraction << 1) & ((1 << 52) - 1);
                    }
                }
                let scaled = 0x100 | (fraction >> 44) as u32;
                let mut result_exp: i64 = if BITS == 32 { 253 - exp } else { 2045 - exp };
                let estimate = super::recip_estimate_int(scaled) as u64;
                let mut fraction = (estimate & 0xff) << 44;
                if result_exp == 0 {
                    fraction = (1 << 51) | (fraction >> 1);
                } else if result_exp == -1 {
                    fraction = (1 << 50) | (fraction >> 2);
                    result_exp = 0;
                }
                sign | ((result_exp as $u) << FRAC) | ((fraction >> (52 - FRAC)) as $u)
            }

            pub fn rsqrt_estimate(a: $u, c: Fpcr) -> $u {
                let a = flush(a, c);
                if let Some(n) = process_nan1(a, c) {
                    return n;
                }
                if is_zero(a) {
                    return (a & SIGN) | INF;
                }
                if a & SIGN != 0 {
                    return DEFAULT_NAN;
                }
                if is_inf(a) {
                    return 0;
                }
                let (mut fraction, mut exp) = fraction52(a);
                if exp == 0 {
                    while fraction >> 51 & 1 == 0 {
                        fraction = (fraction << 1) & ((1 << 52) - 1);
                        exp -= 1;
                    }
                    fraction = (fraction << 1) & ((1 << 52) - 1);
                }
                let scaled = if exp & 1 == 1 { 0x80 | (fraction >> 45) as u32 } else { 0x100 | (fraction >> 44) as u32 };
                let result_exp: i64 = if BITS == 32 { (380 - exp).div_euclid(2) } else { (3068 - exp).div_euclid(2) };
                let estimate = super::rsqrt_estimate_int(scaled) as u64;
                ((result_exp as $u) << FRAC) | ((((estimate & 0xff) << 44) >> (52 - FRAC)) as $u)
            }

            #[allow(dead_code)]
            pub type Int = $i;
        }
    };
}

fp_format!(f32b, f32, u32, i32, 32, 23);
fp_format!(f64b, f64, u64, i64, 64, 52);

pub fn fma_err(x: f64, y: f64, z: f64, r: f64) -> f64 {
    let p = x * y;
    let pe = x.mul_add(y, -p);
    let s = p + z;
    let bb = s - p;
    let se = (p - (s - bb)) + (z - bb);
    if !s.is_finite() || !r.is_finite() {
        return 0.0;
    }
    (s - r) + (se + pe)
}

pub fn recip_estimate_int(a: u32) -> u32 {
    let a = a * 2 + 1;
    let b = (1 << 19) / a;
    b.div_ceil(2)
}

pub fn rsqrt_estimate_int(a: u32) -> u32 {
    let a = if a < 256 { a * 2 + 1 } else { (((a >> 1) << 1) + 1) * 2 } as u64;
    let mut b: u64 = 512;
    while a * (b + 1) * (b + 1) < (1 << 28) {
        b += 1;
    }
    b.div_ceil(2) as u32
}

pub fn f16_to_f32(h: u16, c: Fpcr) -> u32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = (h >> 10 & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    match exp {
        0 if frac == 0 => sign,
        0 => {
            let shift = frac.leading_zeros() - 21;
            let frac = (frac << shift) & 0x3ff;
            sign | ((127 - 15 + 1 - shift) << 23) | (frac << 13)
        }
        31 if frac == 0 => sign | 0x7f80_0000,
        31 => {
            let n = sign | 0x7f80_0000 | (frac << 13);
            if c.dn { f32b::DEFAULT_NAN } else { n | f32b::QUIET }
        }
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    }
}

pub fn f64_to_f16(d: u64, c: Fpcr) -> u16 {
    let sign = ((d >> 48) & 0x8000) as u16;
    if f64b::is_nan(d) {
        if c.dn {
            return 0x7e00;
        }
        return sign | 0x7e00 | ((d >> 42) & 0x1ff) as u16;
    }
    let x = f64::from_bits(d).abs();
    if x.is_infinite() {
        return sign | 0x7c00;
    }
    if x == 0.0 {
        return sign;
    }
    let exp = x.log2().floor() as i32;
    let e = exp.clamp(-14, 15);
    let ulp = 2f64.powi(e - 10);
    let q = x / ulp;
    let r = match Round::from_rmode(c.rmode) {
        Round::Nearest => q.round_ties_even(),
        Round::PlusInf if sign == 0 => q.ceil(),
        Round::MinusInf if sign != 0 => q.ceil(),
        _ => q.trunc(),
    } as u64;
    let (mut e, mut m) = (e, r);
    if m >= 2048 {
        m >>= 1;
        e += 1;
    }
    if e > 15 {
        return sign | 0x7c00;
    }
    if m < 1024 {
        return sign | m as u16;
    }
    sign | (((e + 15) as u16) << 10) | (m as u16 & 0x3ff)
}

pub fn f32_to_f16(s: u32, c: Fpcr) -> u16 {
    if f32b::is_nan(s) {
        let sign = ((s >> 16) & 0x8000) as u16;
        return if c.dn { 0x7e00 } else { sign | 0x7e00 | ((s >> 13) & 0x1ff) as u16 };
    }
    f64_to_f16((f32::from_bits(f32b::flush(s, c)) as f64).to_bits(), c)
}

pub fn f32_to_f64(s: u32, c: Fpcr) -> u64 {
    let s = f32b::flush(s, c);
    if f32b::is_nan(s) {
        if c.dn {
            return f64b::DEFAULT_NAN;
        }
        let sign = ((s as u64) & 0x8000_0000) << 32;
        return sign | f64b::EXP | f64b::QUIET | (((s & 0x003f_ffff) as u64) << 29);
    }
    (f32::from_bits(s) as f64).to_bits()
}

pub fn f64_to_f32(d: u64, c: Fpcr) -> u32 {
    let d = f64b::flush(d, c);
    if f64b::is_nan(d) {
        if c.dn {
            return f32b::DEFAULT_NAN;
        }
        let sign = ((d >> 32) & 0x8000_0000) as u32;
        return sign | f32b::EXP | f32b::QUIET | ((d >> 29) & 0x003f_ffff) as u32;
    }
    f32b::flush((f64::from_bits(d) as f32).to_bits(), c)
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: Fpcr = Fpcr { dn: false, fz: false, rmode: 0 };

    #[test]
    fn nan_rules_match_the_hardware() {
        assert_eq!(f32b::add(0x7fc0_0001, 0x7f80_0002, C), 0x7fc0_0002);
        assert_eq!(f32b::sub(0x7f80_0000, 0x7f80_0000, C), 0x7fc0_0000);
        assert_eq!(f32b::add(1.5f32.to_bits(), 2.25f32.to_bits(), C), 3.75f32.to_bits());
        assert_eq!(f32b::to_int(3e9f32.to_bits(), Round::Zero, true, 32, 0, C), 0x7fff_ffff);
    }

    #[test]
    fn min_max_zeros() {
        let (p, n) = (0u32, 0x8000_0000u32);
        assert_eq!(f32b::max(p, n, C), p);
        assert_eq!(f32b::min(p, n, C), n);
        assert_eq!(f32b::maxnm(0x7fc0_0000, 1.0f32.to_bits(), C), 1.0f32.to_bits());
    }

    #[test]
    fn estimates() {
        assert_eq!(recip_estimate_int(256), 511);
        assert_eq!(rsqrt_estimate_int(128), 511);
    }

    #[test]
    fn half() {
        assert_eq!(f32_to_f16(1.0f32.to_bits(), C), 0x3c00);
        assert_eq!(f16_to_f32(0x3c00, C), 1.0f32.to_bits());
        assert_eq!(f32_to_f16(65504.0f32.to_bits(), C), 0x7bff);
        assert_eq!(f32_to_f16(1e6f32.to_bits(), C), 0x7c00);
        assert_eq!(f16_to_f32(0x0001, C), 2f32.powi(-24).to_bits());
    }
}
