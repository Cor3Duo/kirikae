macro_rules! define_ops {
    ($( $variant:ident => $func:ident ( $($arg:ident),+ ); )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Op { $($variant),* }

        impl Op {
            pub const ALL: &'static [Op] = &[$(Op::$variant),*];

            pub fn index(self) -> u32 {
                self as u32
            }

            pub fn name(self) -> &'static str {
                match self { $(Op::$variant => stringify!($func)),* }
            }

            pub fn arity(self) -> usize {
                match self { $(Op::$variant => define_ops!(@count $($arg)+)),* }
            }

            pub fn eval(self, a: &[u64]) -> u64 {
                debug_assert_eq!(a.len(), self.arity(), "{self:?}");
                match self {
                    $(Op::$variant => {
                        let mut i = 0usize;
                        $func($({ let _ = stringify!($arg); i += 1; a[i - 1] }),+)
                    }),*
                }
            }
        }
    };
    (@count $h:ident) => { 1usize };
    (@count $h:ident $($t:ident)+) => { 1usize + define_ops!(@count $($t)+) };
}

define_ops! {
    Add => add(a, b);
    Sub => sub(a, b);
    And => and(a, b);
    Or => or(a, b);
    Xor => xor(a, b);
    Not => not(a);
    Mul => mul(a, b);
    Zext8 => zext8(a);
    Zext16 => zext16(a);
    Zext32 => zext32(a);
    Sext8 => sext8(a);
    Sext16 => sext16(a);
    Sext32 => sext32(a);
    Lsl64 => lsl64(a, s);
    Lsr64 => lsr64(a, s);
    Asr64 => asr64(a, s);
    Ror64 => ror64(a, s);
    Lsl32 => lsl32(a, s);
    Lsr32 => lsr32(a, s);
    Asr32 => asr32(a, s);
    Ror32 => ror32(a, s);
    Udiv64 => udiv64(a, b);
    Sdiv64 => sdiv64(a, b);
    Udiv32 => udiv32(a, b);
    Sdiv32 => sdiv32(a, b);
    Smulh => smulh(a, b);
    Umulh => umulh(a, b);
    Clz64 => clz64(a);
    Clz32 => clz32(a);
    Cls64 => cls64(a);
    Cls32 => cls32(a);
    Rbit64 => rbit64(a);
    Rbit32 => rbit32(a);
    Rev64 => rev64(a);
    Rev32 => rev32(a);
    Rev32x2 => rev32x2(a);
    Rev16x4 => rev16x4(a);
    Rev16x2 => rev16x2(a);
    Ubfm64 => ubfm64(src, r, wmask, tmask);
    Ubfm32 => ubfm32(src, r, wmask, tmask);
    Sbfm64 => sbfm64(src, r, s, wmask, tmask);
    Sbfm32 => sbfm32(src, r, s, wmask, tmask);
    Bfm64 => bfm64(dst, src, r, wmask, tmask);
    Bfm32 => bfm32(dst, src, r, wmask, tmask);
    Extr64 => extr64(hi, lo, lsb);
    Extr32 => extr32(hi, lo, lsb);
    FlagsAdd64 => flags_add64(a, b, carry);
    FlagsAdd32 => flags_add32(a, b, carry);
    FlagsLogic64 => flags_logic64(r);
    FlagsLogic32 => flags_logic32(r);
    Carry => carry(nzcv);
    Cond => cond(nzcv, cc);
    Select => select(c, a, b);
    Eq => eq(a, b);
    Ne => ne(a, b);
    Bit => bit(a, n);
    Vshl => vshl(a, esize, shift);
    Vushr => vushr(a, esize, shift);
    Vsshr => vsshr(a, esize, shift);
    Splat => splat(a, esize);
    Vmul => vmul(a, b, esize);
    Vmax => vmax(a, b, esize, signed);
    Vmin => vmin(a, b, esize, signed);
    Vabs => vabs(a, esize);
    Vneg => vneg(a, esize);
    Vsat => vsat(a, b, esize, mode);
    VsatFlag => vsat_flag(a, b, esize, mode);
    Vmla => vmla(a, b, d, esize);
    Vmls => vmls(a, b, d, esize);
    Vpmax => vpmax(a, b, esize, signed);
    Vpmin => vpmin(a, b, esize, signed);
    Vcnt => vcnt(a, esize);
    Vaddlp => vaddlp(a, esize);
    Vaddv => vaddv(a, esize);
    FAdd32 => fadd32(a, b, fpcr);
    FSub32 => fsub32(a, b, fpcr);
    FMul32 => fmul32(a, b, fpcr);
    FDiv32 => fdiv32(a, b, fpcr);
    FMax32 => fmax32(a, b, fpcr);
    FMin32 => fmin32(a, b, fpcr);
    FMaxNm32 => fmaxnm32(a, b, fpcr);
    FMinNm32 => fminnm32(a, b, fpcr);
    FMulAdd32 => fmuladd32(addend, a, b, fpcr);
    FCmp32 => fcmp32(a, b, fpcr);
    FToInt32 => ftoint32(a, mode, fpcr);
    FFromInt32 => ffromint32(v, mode, fpcr);
    FAdd64 => fadd64(a, b, fpcr);
    FSub64 => fsub64(a, b, fpcr);
    FMul64 => fmul64(a, b, fpcr);
    FDiv64 => fdiv64(a, b, fpcr);
    FMax64 => fmax64(a, b, fpcr);
    FMin64 => fmin64(a, b, fpcr);
    FMaxNm64 => fmaxnm64(a, b, fpcr);
    FMinNm64 => fminnm64(a, b, fpcr);
    FMulAdd64 => fmuladd64(addend, a, b, fpcr);
    FCmp64 => fcmp64(a, b, fpcr);
    FToInt64 => ftoint64(a, mode, fpcr);
    FFromInt64 => ffromint64(v, mode, fpcr);
    FAdd32x2 => fadd32x2(a, b, fpcr);
    FSub32x2 => fsub32x2(a, b, fpcr);
    FMul32x2 => fmul32x2(a, b, fpcr);
    FMulAdd32x2 => fmuladd32x2(addend, a, b, fpcr);
    Crc32 => crc32(acc, data, size);
    Crc32c => crc32c(acc, data, size);
    FSqrt32 => fsqrt32(a, fpcr);
    FSqrt64 => fsqrt64(a, fpcr);
    FCvtSD => fcvtsd(a, fpcr);
    FCvtDS => fcvtds(a, fpcr);
}

#[inline(always)]
pub fn add(a: u64, b: u64) -> u64 {
    a.wrapping_add(b)
}
#[inline(always)]
pub fn sub(a: u64, b: u64) -> u64 {
    a.wrapping_sub(b)
}
#[inline(always)]
pub fn and(a: u64, b: u64) -> u64 {
    a & b
}
#[inline(always)]
pub fn or(a: u64, b: u64) -> u64 {
    a | b
}
#[inline(always)]
pub fn xor(a: u64, b: u64) -> u64 {
    a ^ b
}
#[inline(always)]
pub fn not(a: u64) -> u64 {
    !a
}
#[inline(always)]
pub fn mul(a: u64, b: u64) -> u64 {
    a.wrapping_mul(b)
}

#[inline(always)]
pub fn zext8(a: u64) -> u64 {
    a & 0xff
}
#[inline(always)]
pub fn zext16(a: u64) -> u64 {
    a & 0xffff
}
#[inline(always)]
pub fn zext32(a: u64) -> u64 {
    a & 0xffff_ffff
}
#[inline(always)]
pub fn sext8(a: u64) -> u64 {
    a as u8 as i8 as i64 as u64
}
#[inline(always)]
pub fn sext16(a: u64) -> u64 {
    a as u16 as i16 as i64 as u64
}
#[inline(always)]
pub fn sext32(a: u64) -> u64 {
    a as u32 as i32 as i64 as u64
}

#[inline(always)]
pub fn lsl64(a: u64, s: u64) -> u64 {
    a << (s & 63)
}
#[inline(always)]
pub fn lsr64(a: u64, s: u64) -> u64 {
    a >> (s & 63)
}
#[inline(always)]
pub fn asr64(a: u64, s: u64) -> u64 {
    ((a as i64) >> (s & 63)) as u64
}
#[inline(always)]
pub fn ror64(a: u64, s: u64) -> u64 {
    a.rotate_right((s & 63) as u32)
}
#[inline(always)]
pub fn lsl32(a: u64, s: u64) -> u64 {
    ((a as u32) << (s & 31)) as u64
}
#[inline(always)]
pub fn lsr32(a: u64, s: u64) -> u64 {
    ((a as u32) >> (s & 31)) as u64
}
#[inline(always)]
pub fn asr32(a: u64, s: u64) -> u64 {
    ((a as u32 as i32) >> (s & 31)) as u32 as u64
}
#[inline(always)]
pub fn ror32(a: u64, s: u64) -> u64 {
    (a as u32).rotate_right((s & 31) as u32) as u64
}

#[inline(always)]
pub fn udiv64(a: u64, b: u64) -> u64 {
    a.checked_div(b).unwrap_or(0)
}
#[inline(always)]
pub fn sdiv64(a: u64, b: u64) -> u64 {
    if b == 0 {
        0
    } else {
        (a as i64).wrapping_div(b as i64) as u64
    }
}
#[inline(always)]
pub fn udiv32(a: u64, b: u64) -> u64 {
    (a as u32).checked_div(b as u32).unwrap_or(0) as u64
}
#[inline(always)]
pub fn sdiv32(a: u64, b: u64) -> u64 {
    if b as u32 == 0 {
        0
    } else {
        (a as u32 as i32).wrapping_div(b as u32 as i32) as u32 as u64
    }
}
#[inline(always)]
pub fn smulh(a: u64, b: u64) -> u64 {
    (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64
}
#[inline(always)]
pub fn umulh(a: u64, b: u64) -> u64 {
    (((a as u128) * (b as u128)) >> 64) as u64
}

#[inline(always)]
pub fn clz64(a: u64) -> u64 {
    a.leading_zeros() as u64
}
#[inline(always)]
pub fn clz32(a: u64) -> u64 {
    (a as u32).leading_zeros() as u64
}
#[inline(always)]
pub fn cls64(a: u64) -> u64 {
    let x = a as i64;
    (((x ^ (x >> 63)) as u64).leading_zeros() - 1) as u64
}
#[inline(always)]
pub fn cls32(a: u64) -> u64 {
    let x = a as u32 as i32;
    (((x ^ (x >> 31)) as u32).leading_zeros() - 1) as u64
}
#[inline(always)]
pub fn rbit64(a: u64) -> u64 {
    a.reverse_bits()
}
#[inline(always)]
pub fn rbit32(a: u64) -> u64 {
    (a as u32).reverse_bits() as u64
}
#[inline(always)]
pub fn rev64(a: u64) -> u64 {
    a.swap_bytes()
}
#[inline(always)]
pub fn rev32(a: u64) -> u64 {
    (a as u32).swap_bytes() as u64
}
#[inline(always)]
pub fn rev32x2(a: u64) -> u64 {
    ((a as u32).swap_bytes() as u64) | ((((a >> 32) as u32).swap_bytes() as u64) << 32)
}
#[inline(always)]
pub fn rev16x4(a: u64) -> u64 {
    ((a >> 8) & 0x00ff_00ff_00ff_00ff) | ((a & 0x00ff_00ff_00ff_00ff) << 8)
}
#[inline(always)]
pub fn rev16x2(a: u64) -> u64 {
    rev16x4(a) & 0xffff_ffff
}

#[inline(always)]
pub fn ubfm64(src: u64, r: u64, wmask: u64, tmask: u64) -> u64 {
    src.rotate_right(r as u32) & wmask & tmask
}
#[inline(always)]
pub fn ubfm32(src: u64, r: u64, wmask: u64, tmask: u64) -> u64 {
    ((src as u32).rotate_right(r as u32) as u64) & wmask & tmask
}
#[inline(always)]
pub fn sbfm64(src: u64, r: u64, s: u64, wmask: u64, tmask: u64) -> u64 {
    let bot = src.rotate_right(r as u32) & wmask;
    let top = if (src >> (s & 63)) & 1 == 1 {
        u64::MAX
    } else {
        0
    };
    (top & !tmask) | (bot & tmask)
}
#[inline(always)]
pub fn sbfm32(src: u64, r: u64, s: u64, wmask: u64, tmask: u64) -> u64 {
    let bot = ((src as u32).rotate_right(r as u32) as u64) & wmask;
    let top = if (src >> (s & 31)) & 1 == 1 {
        0xffff_ffff
    } else {
        0
    };
    ((top & !tmask) | (bot & tmask)) & 0xffff_ffff
}
#[inline(always)]
pub fn bfm64(dst: u64, src: u64, r: u64, wmask: u64, tmask: u64) -> u64 {
    let bot = (dst & !wmask) | (src.rotate_right(r as u32) & wmask);
    (dst & !tmask) | (bot & tmask)
}
#[inline(always)]
pub fn bfm32(dst: u64, src: u64, r: u64, wmask: u64, tmask: u64) -> u64 {
    let rot = (src as u32).rotate_right(r as u32) as u64;
    let bot = (dst & !wmask) | (rot & wmask);
    ((dst & !tmask) | (bot & tmask)) & 0xffff_ffff
}
#[inline(always)]
pub fn extr64(hi: u64, lo: u64, lsb: u64) -> u64 {
    let lsb = lsb & 63;
    if lsb == 0 {
        lo
    } else {
        (lo >> lsb) | (hi << (64 - lsb))
    }
}
#[inline(always)]
pub fn extr32(hi: u64, lo: u64, lsb: u64) -> u64 {
    let (h, l, lsb) = (hi as u32, lo as u32, lsb & 31);
    (if lsb == 0 {
        l
    } else {
        (l >> lsb) | (h << (32 - lsb))
    }) as u64
}

pub const N: u64 = 1 << 31;
pub const Z: u64 = 1 << 30;
pub const C: u64 = 1 << 29;
pub const V: u64 = 1 << 28;

#[inline(always)]
fn pack(n: bool, z: bool, c: bool, v: bool) -> u64 {
    ((n as u64) << 31) | ((z as u64) << 30) | ((c as u64) << 29) | ((v as u64) << 28)
}

#[inline(always)]
pub fn flags_add64(a: u64, b: u64, carry: u64) -> u64 {
    let (r1, c1) = a.overflowing_add(b);
    let (r, c2) = r1.overflowing_add(carry & 1);
    let v = ((a ^ r) & (b ^ r)) >> 63 == 1;
    pack(r >> 63 == 1, r == 0, c1 | c2, v)
}
#[inline(always)]
pub fn flags_add32(a: u64, b: u64, carry: u64) -> u64 {
    let (a, b) = (a as u32, b as u32);
    let (r1, c1) = a.overflowing_add(b);
    let (r, c2) = r1.overflowing_add((carry & 1) as u32);
    let v = ((a ^ r) & (b ^ r)) >> 31 == 1;
    pack(r >> 31 == 1, r == 0, c1 | c2, v)
}
#[inline(always)]
pub fn flags_logic64(r: u64) -> u64 {
    pack(r >> 63 == 1, r == 0, false, false)
}
#[inline(always)]
pub fn flags_logic32(r: u64) -> u64 {
    let r = r as u32;
    pack(r >> 31 == 1, r == 0, false, false)
}
#[inline(always)]
pub fn carry(nzcv: u64) -> u64 {
    (nzcv >> 29) & 1
}

#[inline(always)]
pub fn cond(nzcv: u64, cc: u64) -> u64 {
    let n = nzcv & N != 0;
    let z = nzcv & Z != 0;
    let c = nzcv & C != 0;
    let v = nzcv & V != 0;
    let base = match (cc >> 1) & 7 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    let invert = cc & 1 == 1 && cc != 15;
    (base != invert) as u64
}
#[inline(always)]
pub fn select(c: u64, a: u64, b: u64) -> u64 {
    if c != 0 { a } else { b }
}
#[inline(always)]
pub fn eq(a: u64, b: u64) -> u64 {
    (a == b) as u64
}
#[inline(always)]
pub fn ne(a: u64, b: u64) -> u64 {
    (a != b) as u64
}
#[inline(always)]
pub fn bit(a: u64, n: u64) -> u64 {
    (a >> (n & 63)) & 1
}

/// Low `esize` bits set, zero elsewhere. Total for any `esize`, including 0 and >64.
#[inline(always)]
pub fn lane_mask(esize: u64) -> u64 {
    if esize >= 64 {
        u64::MAX
    } else if esize == 0 {
        0
    } else {
        (1u64 << esize) - 1
    }
}

#[inline(always)]
fn repeat_lane(unit: u64, esize: u64) -> u64 {
    let step = esize.clamp(1, 64);
    let mut m = 0u64;
    let mut i = 0u64;
    while i < 64 {
        m |= unit << i;
        i += step;
    }
    m
}

/// Bits of a lane that survive a shift of `shift` bits to the **right**.
///
/// AArch64 lane shifts never move bits across lane boundaries, and every lane width
/// here (8/16/32/64) divides the 64-bit halves of a register, so both halves can be
/// shifted as plain `u64` and masked with this.
#[inline(always)]
pub fn lane_keep(esize: u64, shift: u64) -> u64 {
    if esize == 0 || esize > 64 || shift >= esize {
        return 0;
    }
    repeat_lane(lane_mask(esize) >> shift, esize)
}

/// Bits of a lane that survive a shift of `shift` bits to the **left**: the low `shift`
/// bits of each lane come from the lane below and must be cleared.
#[inline(always)]
pub fn lane_keep_left(esize: u64, shift: u64) -> u64 {
    if esize == 0 || esize > 64 || shift >= esize {
        return 0;
    }
    repeat_lane((lane_mask(esize) << shift) & lane_mask(esize), esize)
}

#[inline(always)]
pub fn vshl(a: u64, esize: u64, shift: u64) -> u64 {
    let keep = lane_keep_left(esize, shift);
    if keep == 0 {
        return 0;
    }
    (a << shift) & keep
}

#[inline(always)]
pub fn vushr(a: u64, esize: u64, shift: u64) -> u64 {
    let keep = lane_keep(esize, shift);
    if keep == 0 {
        return 0;
    }
    (a >> shift) & keep
}

/// Per-lane arithmetic shift right. The low bits of every lane come from a logical
/// shift; the top `shift` bits of each lane are that lane's sign, replicated. The sign
/// bit is moved to the lane base (0/1), multiplied by a lane-local pattern of the top
/// bits and masked in, so no lane borrows from its neighbour.
#[inline(always)]
pub fn vsshr(a: u64, esize: u64, shift: u64) -> u64 {
    if esize == 0 || esize > 64 {
        return 0;
    }
    let low = lane_keep(esize, shift);
    let shifted = if low == 0 { 0 } else { (a >> shift) & low };
    let unit = if shift == 0 {
        0
    } else if shift >= esize {
        lane_mask(esize)
    } else {
        (lane_mask(shift) << (esize - shift)) & lane_mask(esize)
    };
    if unit == 0 {
        return shifted;
    }
    let sign = (a & splat(1 << (esize - 1), esize)) >> (esize - 1);
    shifted | sign.wrapping_mul(unit)
}

/// Repeats the low `esize` bits of `a` in every lane (DUP element).
#[inline(always)]
pub fn splat(a: u64, esize: u64) -> u64 {
    if esize == 0 || esize > 64 {
        return 0;
    }
    repeat_lane(a & lane_mask(esize), esize)
}

/// Multiplier that turns `splat(1, esize)` into a broadcast, for codegen.
#[inline(always)]
pub fn lane_repeat_ones(esize: u64) -> u64 {
    splat(1, esize)
}

/// True for the lane widths the vector arithmetic below accepts: a width that divides
/// the 64-bit halves of a register, so no lane crosses a half boundary.
#[inline(always)]
fn lane_width(esize: u64) -> bool {
    esize != 0 && esize <= 64 && 64 % esize == 0
}

#[inline(always)]
fn lane_sx(x: u64, esize: u64) -> i64 {
    if esize >= 64 {
        x as i64
    } else {
        ((x << (64 - esize)) as i64) >> (64 - esize)
    }
}

/// Lane-wise wrapping multiply. A plain 64-bit multiply mixes neighbouring lanes (the
/// high half of one lane's product lands in the next), so every lane is isolated.
#[inline(always)]
pub fn vmul(a: u64, b: u64, esize: u64) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let m = lane_mask(esize);
    let mut out = 0u64;
    let mut i = 0u64;
    while i < 64 {
        let product = ((a >> i) & m).wrapping_mul((b >> i) & m) & m;
        out |= product << i;
        i += esize;
    }
    out
}

/// Per-lane `a >= b` as a full-lane mask.
///
/// For an unsigned comparison the subtraction `(a | H) - (b & !H)` cannot borrow across
/// lanes (`H` is the top bit of each lane), and its top bit is set exactly when the low
/// bits of `a` are not below those of `b`. The top bits themselves decide the comparison
/// when they differ, and a signed comparison is the unsigned one with both sign bits
/// flipped.
#[inline(always)]
fn vge_mask(a: u64, b: u64, esize: u64, signed: bool) -> u64 {
    let h = 1u64 << (esize - 1);
    let hrep = splat(h, esize);
    let (x, y) = if signed { (a ^ hrep, b ^ hrep) } else { (a, b) };
    let t = (x | hrep).wrapping_sub(y & !hrep);
    let differ = ((x ^ y) & hrep) >> (esize - 1);
    let d = differ.wrapping_mul(lane_mask(esize));
    let ge = ((x & hrep) & d) | ((t & hrep) & !d);
    (ge >> (esize - 1)).wrapping_mul(lane_mask(esize))
}

/// `signed` is 0 for unsigned lanes and non-zero for signed lanes, as the lifter writes it.
#[inline(always)]
pub fn vmax(a: u64, b: u64, esize: u64, signed: u64) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let mask = vge_mask(a, b, esize, signed != 0);
    b ^ ((a ^ b) & mask)
}

#[inline(always)]
pub fn vmin(a: u64, b: u64, esize: u64, signed: u64) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let mask = vge_mask(a, b, esize, signed != 0);
    a ^ ((a ^ b) & mask)
}

/// Lane-wise absolute value, non-saturating: the absolute value of the most negative
/// lane is itself (the helper's `unsigned_abs`). `(x ^ m) + sign` negates only the
/// negative lanes, and adding the lane-base bit cannot carry out because `!x` has its
/// sign bit clear in those lanes.
#[inline(always)]
pub fn vabs(a: u64, esize: u64) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let sign = (a >> (esize - 1)) & splat(1, esize);
    let m = sign.wrapping_mul(lane_mask(esize));
    (a ^ m).wrapping_add(sign)
}

/// Lane-wise wrapping negation. Adding one at every lane base would carry into the next
/// lane when a lane is zero, so the low bits and the sign bit are handled separately.
#[inline(always)]
pub fn vneg(a: u64, esize: u64) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let hrep = splat(1 << (esize - 1), esize);
    let low = !hrep;
    let v = ((!a) & low).wrapping_add(splat(1, esize));
    (v & low) | (((!a) & hrep) ^ (v & hrep))
}

/// One saturating lane: the clamped value and whether it had to clamp. `mode` is the
/// operation the lifter encodes: 0/1 are `SQADD` signed/unsigned, 2/3 `SQSUB`
/// signed/unsigned, 4 `SQABS` and 5 `SQNEG` (those two ignore `y`).
#[inline(always)]
fn sat_lane(x: u64, y: u64, esize: u64, mode: u64) -> (u64, bool) {
    let signed = mode != 1 && mode != 3;
    let exact: i128 = match mode {
        1 => x as i128 + y as i128,
        2 => lane_sx(x, esize) as i128 - lane_sx(y, esize) as i128,
        3 => x as i128 - y as i128,
        4 => (lane_sx(x, esize) as i128).abs(),
        5 => -(lane_sx(x, esize) as i128),
        _ => lane_sx(x, esize) as i128 + lane_sx(y, esize) as i128,
    };
    let (lo, hi) = if signed {
        let bits = 1i128 << (esize - 1);
        (-bits, bits - 1)
    } else {
        (0, lane_mask(esize) as i128)
    };
    let clamped = exact.clamp(lo, hi);
    (clamped as u64 & lane_mask(esize), clamped != exact)
}

/// Lane-wise saturating arithmetic. Saturation is per lane, so the lane loop keeps every
/// result inside its own bits; `esize` divides 64, so the halves stay independent.
#[inline(always)]
pub fn vsat(a: u64, b: u64, esize: u64, mode: u64) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let m = lane_mask(esize);
    let mut out = 0u64;
    let mut i = 0u64;
    while i < 64 {
        let (r, _) = sat_lane((a >> i) & m, (b >> i) & m, esize, mode);
        out |= r << i;
        i += esize;
    }
    out
}

/// `1` when any lane of `Vsat` would clamp, `0` otherwise: the cumulative `QC` bit.
#[inline(always)]
pub fn vsat_flag(a: u64, b: u64, esize: u64, mode: u64) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let m = lane_mask(esize);
    let mut sat = false;
    let mut i = 0u64;
    while i < 64 {
        sat |= sat_lane((a >> i) & m, (b >> i) & m, esize, mode).1;
        i += esize;
    }
    sat as u64
}

/// Lane-wise wrapping `d + a * b` (`MLA`) or `d - a * b` (`MLS`). Isolating every lane
/// keeps the high half of one product out of the next lane.
#[inline(always)]
fn accumulate(a: u64, b: u64, d: u64, esize: u64, sub: bool) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let m = lane_mask(esize);
    let mut out = 0u64;
    let mut i = 0u64;
    while i < 64 {
        let product = ((a >> i) & m).wrapping_mul((b >> i) & m);
        let acc = (d >> i) & m;
        let r = if sub {
            acc.wrapping_sub(product)
        } else {
            acc.wrapping_add(product)
        };
        out |= (r & m) << i;
        i += esize;
    }
    out
}

#[inline(always)]
pub fn vmla(a: u64, b: u64, d: u64, esize: u64) -> u64 {
    accumulate(a, b, d, esize, false)
}

#[inline(always)]
pub fn vmls(a: u64, b: u64, d: u64, esize: u64) -> u64 {
    accumulate(a, b, d, esize, true)
}

/// Input lane `index` of the 128-bit value formed by `a` (low) and `b` (high).
#[inline(always)]
fn pair_input(a: u64, b: u64, esize: u64, index: u64) -> u64 {
    let bit = index * esize;
    let (word, shift) = if bit < 64 { (a, bit) } else { (b, bit - 64) };
    (word >> shift) & lane_mask(esize)
}

/// Pairwise maximum or minimum of adjacent lanes over `a | b << 64`, as `ADDP` does for
/// the two sources: the 64-bit result holds one lane per input pair.
#[inline(always)]
fn pair_sel(a: u64, b: u64, esize: u64, signed: bool, max: bool) -> u64 {
    if !lane_width(esize) {
        return 0;
    }
    let mut out = 0u64;
    let mut j = 0u64;
    while j < 64 {
        let first = 2 * (j / esize);
        let (x, y) = (
            pair_input(a, b, esize, first),
            pair_input(a, b, esize, first + 1),
        );
        let r = match (signed, max) {
            (true, true) if lane_sx(x, esize) >= lane_sx(y, esize) => x,
            (true, false) if lane_sx(x, esize) <= lane_sx(y, esize) => x,
            (false, true) if x >= y => x,
            (false, false) if x <= y => x,
            _ => y,
        };
        out |= r << j;
        j += esize;
    }
    out
}

#[inline(always)]
pub fn vpmax(a: u64, b: u64, esize: u64, signed: u64) -> u64 {
    pair_sel(a, b, esize, signed != 0, true)
}

#[inline(always)]
pub fn vpmin(a: u64, b: u64, esize: u64, signed: u64) -> u64 {
    pair_sel(a, b, esize, signed != 0, false)
}

/// Population count of every lane of a 64-bit half (`CNT`). The four partial sums are
/// built with the usual SWAR word trick, which never carries across a byte.
#[inline(always)]
pub fn vcnt(a: u64, esize: u64) -> u64 {
    let x = a - ((a >> 1) & 0x5555_5555_5555_5555);
    let x = (x & 0x3333_3333_3333_3333) + ((x >> 2) & 0x3333_3333_3333_3333);
    let x = (x + (x >> 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    match esize {
        8 => x,
        16 => (x & 0x00ff_00ff_00ff_00ff) + ((x >> 8) & 0x00ff_00ff_00ff_00ff),
        32 => {
            let y = (x & 0x00ff_00ff_00ff_00ff) + ((x >> 8) & 0x00ff_00ff_00ff_00ff);
            (y & 0x0000_ffff_0000_ffff) + ((y >> 16) & 0x0000_ffff_0000_ffff)
        }
        _ => x,
    }
}

/// Pairwise add of the lanes of one 64-bit half into twice as wide lanes (`SADDLP` with
/// the signed form of the caller). The result of a lane never leaves the half, so the
/// two halves of a register can be computed independently.
#[inline(always)]
pub fn vaddlp(a: u64, esize: u64) -> u64 {
    match esize {
        8 => (a & 0x00ff_00ff_00ff_00ff) + ((a >> 8) & 0x00ff_00ff_00ff_00ff),
        16 => (a & 0x0000_ffff_0000_ffff) + ((a >> 16) & 0x0000_ffff_0000_ffff),
        32 => (a & 0xffff_ffff) + (a >> 32),
        _ => a,
    }
}

/// Sum of every lane of one 64-bit half into a single value (`ADDV`/`SADDLV`/`UADDLV`,
/// reduced by the caller to the result width). Adding the halves of a `Q` = 1 register
/// and masking gives the architectural result.
#[inline(always)]
pub fn vaddv(a: u64, esize: u64) -> u64 {
    match esize {
        8 => {
            let x = (a & 0x00ff_00ff_00ff_00ff) + ((a >> 8) & 0x00ff_00ff_00ff_00ff);
            let x = (x & 0x0000_ffff_0000_ffff) + ((x >> 16) & 0x0000_ffff_0000_ffff);
            (x & 0xffff_ffff) + (x >> 32)
        }
        16 => {
            let x = (a & 0x0000_ffff_0000_ffff) + ((a >> 16) & 0x0000_ffff_0000_ffff);
            (x & 0xffff_ffff) + (x >> 32)
        }
        32 => (a & 0xffff_ffff) + (a >> 32),
        _ => a,
    }
}

use crate::simd::fp::{Fpcr, Round, f32b, f64b};

pub fn fp_mode(round: Round, signed: bool, width: u32) -> u64 {
    let r = match round {
        Round::Nearest => 0,
        Round::Away => 1,
        Round::PlusInf => 2,
        Round::MinusInf => 3,
        Round::Zero => 4,
    };
    r | (signed as u64) << 3 | (width as u64) << 8
}

fn fp_mode_parts(mode: u64) -> (Round, bool, u32) {
    let round = [
        Round::Nearest,
        Round::Away,
        Round::PlusInf,
        Round::MinusInf,
        Round::Zero,
    ][(mode & 7) as usize % 5];
    (round, mode >> 3 & 1 == 1, (mode >> 8 & 0xff) as u32)
}

macro_rules! fp_ops {
    ($m:ident, $u:ty, $add:ident, $sub:ident, $mul:ident, $div:ident, $max:ident, $min:ident, $maxnm:ident, $minnm:ident, $fma:ident, $cmp:ident, $toint:ident, $fromint:ident) => {
        pub fn $add(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::add(a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $sub(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::sub(a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $mul(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::mul(a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $div(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::div(a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $max(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::max(a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $min(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::min(a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $maxnm(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::maxnm(a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $minnm(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::minnm(a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $fma(addend: u64, a: u64, b: u64, fpcr: u64) -> u64 {
            $m::mul_add(addend as $u, a as $u, b as $u, Fpcr::from_bits(fpcr)) as u64
        }
        pub fn $cmp(a: u64, b: u64, fpcr: u64) -> u64 {
            $m::compare(a as $u, b as $u, Fpcr::from_bits(fpcr))
        }
        pub fn $toint(a: u64, mode: u64, fpcr: u64) -> u64 {
            let (round, signed, width) = fp_mode_parts(mode);
            $m::to_int(a as $u, round, signed, width, 0, Fpcr::from_bits(fpcr))
        }
        pub fn $fromint(v: u64, mode: u64, fpcr: u64) -> u64 {
            let (_, signed, width) = fp_mode_parts(mode);
            $m::from_int(v, signed, width, 0, Fpcr::from_bits(fpcr)) as u64
        }
    };
}

fp_ops!(
    f32b, u32, fadd32, fsub32, fmul32, fdiv32, fmax32, fmin32, fmaxnm32, fminnm32, fmuladd32,
    fcmp32, ftoint32, ffromint32
);
fp_ops!(
    f64b, u64, fadd64, fsub64, fmul64, fdiv64, fmax64, fmin64, fmaxnm64, fminnm64, fmuladd64,
    fcmp64, ftoint64, ffromint64
);

pub fn fsqrt32(a: u64, fpcr: u64) -> u64 {
    f32b::sqrt(a as u32, Fpcr::from_bits(fpcr)) as u64
}
pub fn fsqrt64(a: u64, fpcr: u64) -> u64 {
    f64b::sqrt(a, Fpcr::from_bits(fpcr))
}
pub fn fcvtsd(a: u64, fpcr: u64) -> u64 {
    crate::simd::fp::f32_to_f64(a as u32, Fpcr::from_bits(fpcr))
}
pub fn fcvtds(a: u64, fpcr: u64) -> u64 {
    crate::simd::fp::f64_to_f32(a, Fpcr::from_bits(fpcr)) as u64
}

fn lanes2(a: u64, b: u64, f: impl Fn(u64, u64) -> u64) -> u64 {
    f(a & 0xffff_ffff, b & 0xffff_ffff) | f(a >> 32, b >> 32) << 32
}

pub fn fadd32x2(a: u64, b: u64, fpcr: u64) -> u64 {
    lanes2(a, b, |x, y| fadd32(x, y, fpcr))
}
pub fn fsub32x2(a: u64, b: u64, fpcr: u64) -> u64 {
    lanes2(a, b, |x, y| fsub32(x, y, fpcr))
}
pub fn fmul32x2(a: u64, b: u64, fpcr: u64) -> u64 {
    lanes2(a, b, |x, y| fmul32(x, y, fpcr))
}
pub fn fmuladd32x2(addend: u64, a: u64, b: u64, fpcr: u64) -> u64 {
    let lo = fmuladd32(addend & 0xffff_ffff, a & 0xffff_ffff, b & 0xffff_ffff, fpcr);
    let hi = fmuladd32(addend >> 32, a >> 32, b >> 32, fpcr);
    lo | hi << 32
}

#[inline(always)]
fn crc_reflected(acc: u64, data: u64, size: u64, poly: u32) -> u64 {
    let mut crc = acc as u32;
    for i in 0..1u64 << size {
        crc ^= (data >> (8 * i)) as u8 as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (poly & (crc & 1).wrapping_neg());
        }
    }
    crc as u64
}
#[inline(always)]
pub fn crc32(acc: u64, data: u64, size: u64) -> u64 {
    crc_reflected(acc, data, size, 0xEDB8_8320)
}
#[inline(always)]
pub fn crc32c(acc: u64, data: u64, size: u64) -> u64 {
    crc_reflected(acc, data, size, 0x82F6_3B78)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_matches_direct_calls() {
        assert_eq!(Op::Add.eval(&[2, 3]), 5);
        assert_eq!(Op::Ubfm64.arity(), 4);
        assert_eq!(Op::Bfm32.name(), "bfm32");
        assert_eq!(Op::Select.eval(&[1, 7, 9]), 7);
        for (i, op) in Op::ALL.iter().enumerate() {
            assert_eq!(op.index() as usize, i);
        }
    }

    #[test]
    fn subtraction_flags() {
        let f = flags_add64(5, !5, 1);
        assert_eq!(f, Z | C);
        let f = flags_add64(3, !5, 1);
        assert_eq!(f, N);
        let f = flags_add64(1 << 63, !1, 1);
        assert_eq!(f & V, V);
        assert_eq!(flags_add32(7, !7u64, 1), Z | C);
    }

    #[test]
    fn conditions() {
        let eq_flags = flags_add64(5, !5, 1);
        assert_eq!(cond(eq_flags, 0), 1);
        assert_eq!(cond(eq_flags, 1), 0);
        assert_eq!(cond(eq_flags, 2), 1);
        let lt_flags = flags_add64(3, !5, 1);
        assert_eq!(cond(lt_flags, 11), 1);
        assert_eq!(cond(lt_flags, 12), 0);
        assert_eq!(cond(0, 14), 1);
        assert_eq!(cond(0, 15), 1);
    }

    #[test]
    fn crc() {
        let run = |f: fn(u64, u64, u64) -> u64| {
            !(b"123456789"
                .iter()
                .fold(!0u32 as u64, |c, &b| f(c, b as u64, 0)) as u32)
        };
        assert_eq!(run(crc32), 0xCBF4_3926);
        assert_eq!(run(crc32c), 0xE306_9283);
        let bytes = 0x0807_0605_0403_0201u64;
        let narrow = (0..8).fold(0x1234_5678, |c, i| crc32c(c, bytes >> (8 * i), 0));
        assert_eq!(crc32c(0x1234_5678, bytes, 3), narrow);
        assert_eq!(
            crc32(0x1234_5678, bytes, 2),
            (0..4).fold(0x1234_5678, |c, i| crc32(c, bytes >> (8 * i), 0))
        );
    }

    #[test]
    fn vector_arithmetic_matches_per_lane_reference() {
        let lanes = |a: u64, esize: u64| -> Vec<u64> {
            (0..64 / esize)
                .map(|i| (a >> (i * esize)) & lane_mask(esize))
                .collect()
        };
        let values = [
            0,
            u64::MAX,
            0x8000_0000_0000_0000,
            0x7fff_ffff_ffff_ffff,
            0x0102_0304_0506_0708,
            0xf0e0_d0c0_b0a0_9080,
            0x0000_0000_0000_00ff,
            0x00ff_00ff_00ff_00ff,
        ];
        for esize in [8u64, 16, 32, 64] {
            let m = lane_mask(esize);
            for &a in &values {
                for &b in &values {
                    let (la, lb) = (lanes(a, esize), lanes(b, esize));
                    let (mut mul, mut smax, mut umax) = (0, 0, 0);
                    let (mut smin, mut umin) = (0, 0);
                    let (mut abs, mut neg) = (0, 0);
                    for (i, (&x, &y)) in la.iter().zip(&lb).enumerate() {
                        let at = i as u64 * esize;
                        let (sx, sy) = (lane_sx(x, esize), lane_sx(y, esize));
                        mul |= (x.wrapping_mul(y) & m) << at;
                        smax |= (if sx >= sy { x } else { y }) << at;
                        smin |= (if sx <= sy { x } else { y }) << at;
                        umax |= x.max(y) << at;
                        umin |= x.min(y) << at;
                        abs |= (sx.unsigned_abs() & m) << at;
                        neg |= (x.wrapping_neg() & m) << at;
                    }
                    assert_eq!(vmul(a, b, esize), mul, "mul {esize} {a:#x} {b:#x}");
                    assert_eq!(vmax(a, b, esize, 1), smax, "smax {esize} {a:#x} {b:#x}");
                    assert_eq!(vmax(a, b, esize, 0), umax, "umax {esize} {a:#x} {b:#x}");
                    assert_eq!(vmin(a, b, esize, 1), smin, "smin {esize} {a:#x} {b:#x}");
                    assert_eq!(vmin(a, b, esize, 0), umin, "umin {esize} {a:#x} {b:#x}");
                    assert_eq!(vabs(a, esize), abs, "abs {esize} {a:#x}");
                    assert_eq!(vneg(a, esize), neg, "neg {esize} {a:#x}");
                }
            }
        }
        assert_eq!(Op::Vmul.arity(), 3);
        assert_eq!(Op::Vmax.arity(), 4);
        assert_eq!(Op::Vabs.name(), "vabs");
    }

    #[test]
    fn reductions() {
        // CNT counts bits inside each lane, so no lane can spill into its neighbour.
        assert_eq!(vcnt(0, 8), 0);
        assert_eq!(vcnt(u64::MAX, 8), 0x0808_0808_0808_0808);
        assert_eq!(vcnt(0x0102_0304_0506_0708, 8), 0x0101_0201_0202_0301);
        assert_eq!(vcnt(0x00ff_00ff_00ff_00ff, 16), 0x0008_0008_0008_0008);
        assert_eq!(vcnt(0xffff_ffff_0000_0000, 32), 0x0000_0020_0000_0000);
        // Pairwise add: 2 lanes of 8 bits become 1 lane of 16, and so on.
        assert_eq!(vaddlp(0x00ff_00ff_00ff_00ff, 8), 0x00ff_00ff_00ff_00ff);
        assert_eq!(vaddlp(0x0001_0002_0003_0004, 16), 0x0000_0003_0000_0007);
        assert_eq!(vaddlp(0x0000_0001_0000_0002, 32), 3);
        // ADDV adds every lane of the half; the caller masks to the result width.
        assert_eq!(vaddv(0x0101_0101_0101_0101, 8), 8);
        assert_eq!(vaddv(0x0001_0002_0003_0004, 16), 10);
        assert_eq!(vaddv(0x0000_0001_0000_0002, 32), 3);
        assert_eq!(vaddv(0x1234_5678_9abc_def0, 64), 0x1234_5678_9abc_def0);
        assert_eq!(Op::Vcnt.arity(), 2);
        assert_eq!(Op::Vaddlp.name(), "vaddlp");
    }

    #[test]
    fn misc() {
        assert_eq!(cls64(0), 63);
        assert_eq!(cls64(u64::MAX), 63);
        assert_eq!(cls64(1), 62);
        assert_eq!(cls32(0xffff_ffff), 31);
        assert_eq!(sdiv64((i64::MIN) as u64, (-1i64) as u64), i64::MIN as u64);
        assert_eq!(udiv64(7, 0), 0);
        assert_eq!(rev16x4(0x0102_0304_0506_0708), 0x0201_0403_0605_0807);
        assert_eq!(rev32x2(0x0102_0304_0506_0708), 0x0403_0201_0807_0605);
        assert_eq!(extr64(0xAAAA, 0x1234, 0), 0x1234);
        assert_eq!(extr64(1, 0, 1), 1 << 63);
    }
}
