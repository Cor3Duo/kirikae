mod crypto;
pub mod fp;
mod fpscalar;
mod ldst;
mod neon;
pub mod oracle;

pub trait Env {
    fn x(&self, n: u32) -> u64;
    fn set_x(&mut self, n: u32, v: u64);
    fn sp(&self) -> u64;
    fn set_sp(&mut self, v: u64);
    fn v(&self, n: u32) -> u128;
    fn set_v(&mut self, n: u32, v: u128);
    fn nzcv(&self) -> u64;
    fn set_nzcv(&mut self, v: u64);
    fn fpcr(&self) -> u64;
    fn fpsr(&self) -> u64;
    fn set_fpsr(&mut self, v: u64);
    fn load(&mut self, addr: u64, size: u8) -> u64;
    fn store(&mut self, addr: u64, size: u8, v: u64);
}

pub fn exec(e: &mut dyn Env, w: u32) -> bool {
    if crypto::supported(w) {
        return crypto::exec(e, w);
    }
    if (w >> 25) & 0b111 == 0b111 {
        if w >> 28 & 1 == 1 && w >> 30 & 1 == 0 {
            fpscalar::exec(e, w)
        } else {
            neon::exec(e, w)
        }
    } else {
        ldst::exec(e, w)
    }
}

pub fn supported(w: u32) -> bool {
    exec(&mut Probe, w)
}

#[derive(Default)]
struct Probe;

impl Env for Probe {
    fn x(&self, _: u32) -> u64 {
        0
    }
    fn set_x(&mut self, _: u32, _: u64) {}
    fn sp(&self) -> u64 {
        0
    }
    fn set_sp(&mut self, _: u64) {}
    fn v(&self, _: u32) -> u128 {
        0
    }
    fn set_v(&mut self, _: u32, _: u128) {}
    fn nzcv(&self) -> u64 {
        0
    }
    fn set_nzcv(&mut self, _: u64) {}
    fn fpcr(&self) -> u64 {
        0
    }
    fn fpsr(&self) -> u64 {
        0
    }
    fn set_fpsr(&mut self, _: u64) {}
    fn load(&mut self, _: u64, _: u8) -> u64 {
        0
    }
    fn store(&mut self, _: u64, _: u8, _: u64) {}
}

pub(crate) fn bits(w: u32, lo: u32, n: u32) -> u32 {
    (w >> lo) & ((1 << n) - 1)
}

pub(crate) fn mask(esize: u32) -> u64 {
    if esize >= 64 {
        u64::MAX
    } else {
        (1 << esize) - 1
    }
}

pub(crate) fn lane(v: u128, esize: u32, i: u32) -> u64 {
    ((v >> (i * esize)) as u64) & mask(esize)
}

pub(crate) fn set_lane(v: &mut u128, esize: u32, i: u32, x: u64) {
    let m = (mask(esize) as u128) << (i * esize);
    *v = (*v & !m) | (((x & mask(esize)) as u128) << (i * esize));
}

pub(crate) fn sx(x: u64, esize: u32) -> i64 {
    if esize >= 64 {
        x as i64
    } else {
        ((x << (64 - esize)) as i64) >> (64 - esize)
    }
}

pub(crate) fn set_qc(e: &mut dyn Env) {
    let f = e.fpsr();
    e.set_fpsr(f | 1 << 27);
}

pub(crate) fn sat_s(x: i128, esize: u32, sat: &mut bool) -> u64 {
    let (lo, hi) = (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1);
    if x < lo {
        *sat = true;
        lo as u64 & mask(esize)
    } else if x > hi {
        *sat = true;
        hi as u64
    } else {
        x as u64 & mask(esize)
    }
}

pub(crate) fn sat_u(x: i128, esize: u32, sat: &mut bool) -> u64 {
    let hi = mask(esize) as i128;
    if x < 0 {
        *sat = true;
        0
    } else if x > hi {
        *sat = true;
        hi as u64
    } else {
        x as u64
    }
}

pub(crate) fn sized(v: u128, datasize: u32) -> u128 {
    if datasize >= 128 {
        v
    } else {
        v & ((1u128 << datasize) - 1)
    }
}

pub(crate) fn fp_expand_imm8(imm8: u32, width: u32) -> u64 {
    let sign = (imm8 >> 7) as u64;
    let b6 = (imm8 >> 6 & 1) as u64;
    let low = (imm8 & 0x3f) as u64;
    match width {
        16 => (sign << 15) | ((b6 ^ 1) << 14) | (if b6 == 1 { 0b11 } else { 0 } << 12) | (low << 6),
        32 => {
            (sign << 31)
                | ((b6 ^ 1) << 30)
                | (if b6 == 1 { 0b1_1111 } else { 0 } << 25)
                | (low << 19)
        }
        _ => (sign << 63) | ((b6 ^ 1) << 62) | (if b6 == 1 { 0xff } else { 0 } << 54) | (low << 48),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp_immediates() {
        assert_eq!(fp_expand_imm8(0x70, 32), 1.0f32.to_bits() as u64);
        assert_eq!(fp_expand_imm8(0x80, 64), (-2.0f64).to_bits());
        assert_eq!(fp_expand_imm8(0x60, 32), 0.5f32.to_bits() as u64);
    }

    #[test]
    fn lanes() {
        let mut v = 0u128;
        set_lane(&mut v, 16, 3, 0xbeef);
        assert_eq!(lane(v, 16, 3), 0xbeef);
        assert_eq!(v, 0xbeef << 48);
        assert_eq!(sx(0x80, 8), -128);
    }
}
