pub fn sext(v: u64, bits: u32) -> u64 {
    let shift = 64 - bits;
    (((v << shift) as i64) >> shift) as u64
}

fn ones(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

fn ror_elem(x: u64, r: u32, esize: u32) -> u64 {
    if r == 0 {
        x
    } else {
        ((x >> r) | (x << (esize - r))) & ones(esize)
    }
}

fn replicate(x: u64, esize: u32, datasize: u32) -> u64 {
    let mut out = 0;
    let mut i = 0;
    while i < datasize {
        out |= x << i;
        i += esize;
    }
    out
}

pub fn decode_bit_masks(
    n: u32,
    imms: u32,
    immr: u32,
    immediate: bool,
    datasize: u32,
) -> Option<(u64, u64)> {
    let combined = (n << 6) | (!imms & 0x3f);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 || (1u32 << len) > datasize {
        return None;
    }
    let levels = (1u32 << len) - 1;
    if immediate && (imms & levels) == levels {
        return None;
    }
    let s = imms & levels;
    let r = immr & levels;
    let d = s.wrapping_sub(r) & levels;
    let esize = 1u32 << len;
    let welem = ones(s + 1);
    let telem = ones(d + 1);
    let wmask = replicate(ror_elem(welem, r, esize), esize, datasize);
    let tmask = replicate(telem, esize, datasize);
    Some((wmask, tmask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_immediates() {
        assert_eq!(decode_bit_masks(1, 7, 0, true, 64).unwrap().0, 0xff);
        assert_eq!(decode_bit_masks(0, 0, 0, true, 32).unwrap().0, 1);
        assert_eq!(
            decode_bit_masks(0, 0b111100, 0, true, 64).unwrap().0,
            0x5555_5555_5555_5555
        );
        assert_eq!(
            decode_bit_masks(0, 0b001111, 16, true, 64).unwrap().0,
            0xffff_0000_ffff_0000
        );
        assert!(decode_bit_masks(1, 0b111111, 0, true, 64).is_none());
    }

    #[test]
    fn sign_extend() {
        assert_eq!(sext(0x1ff, 9), u64::MAX);
        assert_eq!(sext(0x0ff, 9), 0xff);
    }
}
