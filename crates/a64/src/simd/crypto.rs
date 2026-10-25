use super::{Env, bits};

const CRYPTO3_MASK: u32 = 0xffe0_fc00;
const CRYPTO2_MASK: u32 = 0xffff_fc00;

const CRC32_POLY: u32 = 0xedb8_8320;
const CRC32C_POLY: u32 = 0x82f6_3b78;

const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    let mut i = 0;
    while i < 8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
        i += 1;
    }
    p
}

const fn gf_inv(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    let mut x = 1u16;
    while x < 256 {
        if gf_mul(x as u8, a) == 1 {
            return x as u8;
        }
        x += 1;
    }
    0
}

const fn rotl8(v: u8, n: u32) -> u8 {
    v.rotate_left(n)
}

const fn build_sbox() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let inv = gf_inv(i as u8);
        t[i] = inv ^ rotl8(inv, 1) ^ rotl8(inv, 2) ^ rotl8(inv, 3) ^ rotl8(inv, 4) ^ 0x63;
        i += 1;
    }
    t
}

const fn build_inv_sbox(sbox: &[u8; 256]) -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[sbox[i] as usize] = i as u8;
        i += 1;
    }
    t
}

const SBOX: [u8; 256] = build_sbox();
const INV_SBOX: [u8; 256] = build_inv_sbox(&SBOX);

fn to_bytes(v: u128) -> [u8; 16] {
    let mut b = [0u8; 16];
    for (i, slot) in b.iter_mut().enumerate() {
        *slot = (v >> (8 * i)) as u8;
    }
    b
}

fn from_bytes(b: [u8; 16]) -> u128 {
    let mut v = 0u128;
    for (i, byte) in b.iter().enumerate() {
        v |= (*byte as u128) << (8 * i);
    }
    v
}

fn aese(state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
    let mut s = [0u8; 16];
    for i in 0..16 {
        s[i] = state[i] ^ key[i];
    }
    let mut out = [0u8; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[row + 4 * col] = SBOX[s[row + 4 * ((col + row) & 3)] as usize];
        }
    }
    out
}

fn aesd(state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
    let mut s = [0u8; 16];
    for i in 0..16 {
        s[i] = state[i] ^ key[i];
    }
    let mut out = [0u8; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[row + 4 * col] = INV_SBOX[s[row + 4 * ((col + 4 - row) & 3)] as usize];
        }
    }
    out
}

fn mix_columns(state: [u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for col in 0..4 {
        let a = [
            state[4 * col],
            state[4 * col + 1],
            state[4 * col + 2],
            state[4 * col + 3],
        ];
        out[4 * col] = gf_mul(a[0], 2) ^ gf_mul(a[1], 3) ^ a[2] ^ a[3];
        out[4 * col + 1] = a[0] ^ gf_mul(a[1], 2) ^ gf_mul(a[2], 3) ^ a[3];
        out[4 * col + 2] = a[0] ^ a[1] ^ gf_mul(a[2], 2) ^ gf_mul(a[3], 3);
        out[4 * col + 3] = gf_mul(a[0], 3) ^ a[1] ^ a[2] ^ gf_mul(a[3], 2);
    }
    out
}

fn inv_mix_columns(state: [u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for col in 0..4 {
        let a = [
            state[4 * col],
            state[4 * col + 1],
            state[4 * col + 2],
            state[4 * col + 3],
        ];
        out[4 * col] = gf_mul(a[0], 14) ^ gf_mul(a[1], 11) ^ gf_mul(a[2], 13) ^ gf_mul(a[3], 9);
        out[4 * col + 1] = gf_mul(a[0], 9) ^ gf_mul(a[1], 14) ^ gf_mul(a[2], 11) ^ gf_mul(a[3], 13);
        out[4 * col + 2] = gf_mul(a[0], 13) ^ gf_mul(a[1], 9) ^ gf_mul(a[2], 14) ^ gf_mul(a[3], 11);
        out[4 * col + 3] = gf_mul(a[0], 11) ^ gf_mul(a[1], 13) ^ gf_mul(a[2], 9) ^ gf_mul(a[3], 14);
    }
    out
}

fn aesmc(state: [u8; 16]) -> [u8; 16] {
    mix_columns(state)
}

fn aesimc(state: [u8; 16]) -> [u8; 16] {
    inv_mix_columns(state)
}

fn lane32(v: u128, i: u32) -> u32 {
    (v >> (32 * i)) as u32
}

fn v4(lanes: [u32; 4]) -> u128 {
    let mut v = 0u128;
    for (i, lane) in lanes.iter().enumerate() {
        v |= (*lane as u128) << (32 * i);
    }
    v
}

fn sha1_choose(b: u32, c: u32, d: u32) -> u32 {
    (b & c) | (!b & d)
}

fn sha1_parity(b: u32, c: u32, d: u32) -> u32 {
    b ^ c ^ d
}

fn sha1_majority(b: u32, c: u32, d: u32) -> u32 {
    (b & c) | (b & d) | (c & d)
}

fn sha1_rounds(x: u128, e: u32, w: u128, f: fn(u32, u32, u32) -> u32) -> u128 {
    let mut x = x;
    let mut y = e;
    for i in 0..4 {
        let a = lane32(x, 0);
        let b = lane32(x, 1);
        let c = lane32(x, 2);
        let d = lane32(x, 3);
        let t = a
            .rotate_left(5)
            .wrapping_add(f(b, c, d))
            .wrapping_add(y)
            .wrapping_add(lane32(w, i));
        x = v4([t, a, b.rotate_left(30), c]);
        y = d;
    }
    x
}

fn sha1_su0(d: u128, n: u128, m: u128) -> u128 {
    let rotated = v4([lane32(d, 2), lane32(d, 3), lane32(n, 0), lane32(n, 1)]);
    d ^ m ^ rotated
}

fn sha1_su1(d: u128, n: u128) -> u128 {
    let t = v4([
        lane32(d, 0) ^ lane32(n, 1),
        lane32(d, 1) ^ lane32(n, 2),
        lane32(d, 2) ^ lane32(n, 3),
        lane32(d, 3),
    ]);
    v4([
        lane32(t, 0).rotate_left(1),
        lane32(t, 1).rotate_left(1),
        lane32(t, 2).rotate_left(1),
        lane32(t, 3).rotate_left(1) ^ lane32(t, 0).rotate_left(2),
    ])
}

fn sha256_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

fn sha256_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

fn sha256_schedule0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

fn sha256_schedule1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

fn sha256_rounds(x_in: u128, y_in: u128, w: u128) -> (u128, u128) {
    let mut x = x_in;
    let mut y = y_in;
    for i in 0..4 {
        let a = lane32(x, 0);
        let b = lane32(x, 1);
        let c = lane32(x, 2);
        let d = lane32(x, 3);
        let e = lane32(y, 0);
        let f = lane32(y, 1);
        let g = lane32(y, 2);
        let h = lane32(y, 3);
        let ch = sha1_choose(e, f, g);
        let maj = sha1_majority(a, b, c);
        let t = h
            .wrapping_add(sha256_sigma1(e))
            .wrapping_add(ch)
            .wrapping_add(lane32(w, i));
        x = v4([a, b, c, d.wrapping_add(t)]);
        y = v4([e, f, g, t.wrapping_add(sha256_sigma0(a)).wrapping_add(maj)]);
        let (x3, y3) = (lane32(x, 3), lane32(y, 3));
        x = v4([y3, lane32(x, 0), lane32(x, 1), lane32(x, 2)]);
        y = v4([x3, lane32(y, 0), lane32(y, 1), lane32(y, 2)]);
    }
    (x, y)
}

fn sha256_su0(d: u128, n: u128) -> u128 {
    v4([
        sha256_schedule0(lane32(d, 1)).wrapping_add(lane32(d, 0)),
        sha256_schedule0(lane32(d, 2)).wrapping_add(lane32(d, 1)),
        sha256_schedule0(lane32(d, 3)).wrapping_add(lane32(d, 2)),
        sha256_schedule0(lane32(n, 0)).wrapping_add(lane32(d, 3)),
    ])
}

fn sha256_su1(d: u128, n: u128, m: u128) -> u128 {
    let t0 = [lane32(n, 1), lane32(n, 2), lane32(n, 3), lane32(m, 0)];
    let r0 = sha256_schedule1(lane32(m, 2))
        .wrapping_add(lane32(d, 0))
        .wrapping_add(t0[0]);
    let r1 = sha256_schedule1(lane32(m, 3))
        .wrapping_add(lane32(d, 1))
        .wrapping_add(t0[1]);
    let r2 = sha256_schedule1(r0)
        .wrapping_add(lane32(d, 2))
        .wrapping_add(t0[2]);
    let r3 = sha256_schedule1(r1)
        .wrapping_add(lane32(d, 3))
        .wrapping_add(t0[3]);
    v4([r0, r1, r2, r3])
}

fn crc_update(mut crc: u32, data: u64, bytes: u32, poly: u32) -> u32 {
    for i in 0..bytes {
        crc ^= (data >> (8 * i)) as u8 as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (poly & (crc & 1).wrapping_neg());
        }
    }
    crc
}

fn crc_kind(w: u32) -> Option<(u32, u32)> {
    let kind = match w & CRYPTO3_MASK {
        0x1ac0_4000 => (CRC32_POLY, 1),
        0x1ac0_4400 => (CRC32_POLY, 2),
        0x1ac0_4800 => (CRC32_POLY, 4),
        0x9ac0_4c00 => (CRC32_POLY, 8),
        0x1ac0_5000 => (CRC32C_POLY, 1),
        0x1ac0_5400 => (CRC32C_POLY, 2),
        0x1ac0_5800 => (CRC32C_POLY, 4),
        0x9ac0_5c00 => (CRC32C_POLY, 8),
        _ => return None,
    };
    Some(kind)
}

pub fn exec<E: Env + ?Sized>(e: &mut E, w: u32) -> bool {
    if !supported(w) {
        return false;
    }
    let (rn, rd) = (bits(w, 5, 5), bits(w, 0, 5));
    let rm = bits(w, 16, 5);

    if let Some((poly, bytes)) = crc_kind(w) {
        let crc = crc_update(e.x(rn) as u32, e.x(rm), bytes, poly);
        e.set_x(rd, crc as u64);
        return true;
    }

    let aes = w & CRYPTO2_MASK;
    if matches!(aes, 0x4e28_4800 | 0x4e28_5800 | 0x4e28_6800 | 0x4e28_7800) {
        let out = match aes {
            0x4e28_4800 => aese(to_bytes(e.v(rd)), to_bytes(e.v(rn))),
            0x4e28_5800 => aesd(to_bytes(e.v(rd)), to_bytes(e.v(rn))),
            0x4e28_6800 => aesmc(to_bytes(e.v(rn))),
            _ => aesimc(to_bytes(e.v(rn))),
        };
        e.set_v(rd, from_bytes(out));
        return true;
    }

    let result = match w & CRYPTO3_MASK {
        0x5e00_0000 => sha1_rounds(e.v(rd), lane32(e.v(rn), 0), e.v(rm), sha1_choose),
        0x5e00_1000 => sha1_rounds(e.v(rd), lane32(e.v(rn), 0), e.v(rm), sha1_parity),
        0x5e00_2000 => sha1_rounds(e.v(rd), lane32(e.v(rn), 0), e.v(rm), sha1_majority),
        0x5e00_3000 => sha1_su0(e.v(rd), e.v(rn), e.v(rm)),
        0x5e00_4000 => sha256_rounds(e.v(rd), e.v(rn), e.v(rm)).0,
        0x5e00_5000 => sha256_rounds(e.v(rn), e.v(rd), e.v(rm)).1,
        0x5e00_6000 => sha256_su1(e.v(rd), e.v(rn), e.v(rm)),
        _ => match w & CRYPTO2_MASK {
            0x5e28_0800 => lane32(e.v(rn), 0).rotate_left(30) as u128,
            0x5e28_1800 => sha1_su1(e.v(rd), e.v(rn)),
            0x5e28_2800 => sha256_su0(e.v(rd), e.v(rn)),
            _ => return false,
        },
    };
    e.set_v(rd, result);
    true
}

pub fn supported(w: u32) -> bool {
    let aes = matches!(
        w & CRYPTO2_MASK,
        0x4e28_4800 | 0x4e28_5800 | 0x4e28_6800 | 0x4e28_7800
    );
    aes || matches!(
        w & CRYPTO3_MASK,
        0x5e00_0000
            | 0x5e00_1000
            | 0x5e00_2000
            | 0x5e00_3000
            | 0x5e00_4000
            | 0x5e00_5000
            | 0x5e00_6000
            | 0x1ac0_4000
            | 0x1ac0_4400
            | 0x1ac0_4800
            | 0x9ac0_4c00
            | 0x1ac0_5000
            | 0x1ac0_5400
            | 0x1ac0_5800
            | 0x9ac0_5c00
    ) || matches!(w & CRYPTO2_MASK, 0x5e28_0800 | 0x5e28_1800 | 0x5e28_2800)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestEnv {
        x: [u64; 32],
        v: [u128; 32],
    }

    impl Env for TestEnv {
        fn x(&self, n: u32) -> u64 {
            self.x[n as usize]
        }
        fn set_x(&mut self, n: u32, v: u64) {
            self.x[n as usize] = v;
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

    fn aese_word(rd: u32, rn: u32) -> u32 {
        0x4e28_4800 | rn << 5 | rd
    }

    #[test]
    fn aes_round_matches_the_fips_197_intermediate_state() {
        let key = [
            0x00u8, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        let plain = [
            0x00u8, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let after_sub_bytes = [
            0x63u8, 0x53, 0xe0, 0x8c, 0x09, 0x60, 0xe1, 0x04, 0xcd, 0x70, 0xb7, 0x51, 0xba, 0xca,
            0xd0, 0xe7,
        ];
        let after_mix_columns = [
            0x5fu8, 0x72, 0x64, 0x15, 0x57, 0xf5, 0xbc, 0x92, 0xf7, 0xbe, 0x3b, 0x29, 0x1d, 0xb9,
            0xf9, 0x1a,
        ];

        let mut e = TestEnv::default();
        e.v[0] = from_bytes(plain);
        e.v[1] = from_bytes(key);
        assert!(exec(&mut e, aese_word(0, 1)));
        assert_eq!(to_bytes(e.v[0]), after_sub_bytes);

        assert!(exec(&mut e, 0x4e28_6800));
        assert_eq!(to_bytes(e.v[0]), after_mix_columns);
    }

    #[test]
    fn mix_columns_and_inverse_undo_each_other() {
        let state = [
            0xdb, 0x13, 0x53, 0x45, 0xf2, 0x0a, 0x22, 0x5c, 0x2c, 0x26, 0x12, 0x0a, 0x8c, 0x63,
            0x55, 0x55,
        ];
        assert_eq!(inv_mix_columns(mix_columns(state)), state);
    }

    #[test]
    fn the_inverse_sbox_undoes_the_sbox() {
        for i in 0..256 {
            assert_eq!(INV_SBOX[SBOX[i] as usize] as usize, i);
        }
    }

    #[test]
    fn aesd_and_aesimc_are_not_dropped_by_the_decoder() {
        let state = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let key = [0x0fu8; 16];
        let expected = aesd(state, key);
        let mut e = TestEnv::default();
        e.v[0] = from_bytes(state);
        e.v[1] = from_bytes(key);
        assert!(exec(&mut e, 0x4e28_5800 | 1 << 5));
        assert_eq!(to_bytes(e.v[0]), expected);

        let expected = aesimc(state);
        let mut e = TestEnv::default();
        e.v[1] = from_bytes(state);
        assert!(exec(&mut e, 0x4e28_7800 | 1 << 5));
        assert_eq!(to_bytes(e.v[0]), expected);
    }

    #[test]
    fn unsupported_words_are_rejected() {
        assert!(!supported(0x4e28_4800 & !(1 << 30)));
        assert!(!supported(0x0e20_0000));
        assert!(supported(0x4e28_4800));
        assert!(supported(0x4e28_7800));
    }

    fn words4(v: u128) -> [u32; 4] {
        [lane32(v, 0), lane32(v, 1), lane32(v, 2), lane32(v, 3)]
    }

    fn three(base: u32, rd: u32, rn: u32, rm: u32) -> u32 {
        base | rm << 16 | rn << 5 | rd
    }

    fn two(base: u32, rd: u32, rn: u32) -> u32 {
        base | rn << 5 | rd
    }

    const SHA1C: u32 = 0x5e00_0000;
    const SHA1P: u32 = 0x5e00_1000;
    const SHA1M: u32 = 0x5e00_2000;
    const SHA1SU0: u32 = 0x5e00_3000;
    const SHA256H: u32 = 0x5e00_4000;
    const SHA256H2: u32 = 0x5e00_5000;
    const SHA256SU1: u32 = 0x5e00_6000;
    const SHA1H: u32 = 0x5e28_0800;
    const SHA1SU1: u32 = 0x5e28_1800;
    const SHA256SU0: u32 = 0x5e28_2800;

    const CRC32B: u32 = 0x1ac0_4000;
    const CRC32H: u32 = 0x1ac0_4400;
    const CRC32W: u32 = 0x1ac0_4800;
    const CRC32X: u32 = 0x9ac0_4c00;
    const CRC32CB: u32 = 0x1ac0_5000;
    const CRC32CH: u32 = 0x1ac0_5400;
    const CRC32CW: u32 = 0x1ac0_5800;
    const CRC32CX: u32 = 0x9ac0_5c00;

    #[test]
    fn arm_arm_encodings() {
        for base in [SHA1C, SHA1P, SHA1M, SHA1SU0, SHA256H, SHA256H2, SHA256SU1] {
            let w = three(base, 3, 2, 1);
            assert_eq!(w, base | 1 << 16 | 2 << 5 | 3);
            assert!(supported(w), "{w:#010x} should be supported");
            assert!(supported(base | 0x1f << 16), "{base:#010x} must ignore Rm");
        }
        for base in [SHA1H, SHA1SU1, SHA256SU0] {
            let w = two(base, 3, 2);
            assert_eq!(w, base | 2 << 5 | 3);
            assert!(supported(w), "{w:#010x} should be supported");
        }
        for base in [
            CRC32B, CRC32H, CRC32W, CRC32X, CRC32CB, CRC32CH, CRC32CW, CRC32CX,
        ] {
            let w = three(base, 3, 2, 1);
            assert!(supported(w), "{w:#010x} should be supported");
        }

        assert!(!supported(0x5e00_7000));
        assert!(!supported(SHA1H - 0x800));
        assert!(!supported(SHA1SU1 - 0x800));
        assert!(!supported(SHA256SU0 - 0x800));
        assert!(!supported(0x1ac0_6000));
        assert!(!supported(0x1ac0_4c00));
    }

    const SHA1_INIT: [u32; 5] = [
        0x6745_2301,
        0xefcd_ab89,
        0x98ba_dcfe,
        0x1032_5476,
        0xc3d2_e1f0,
    ];
    const SHA1_K: [u32; 4] = [0x5a82_7999, 0x6ed9_eba1, 0x8f1b_bcdc, 0xca62_c1d6];

    const SHA256_INIT: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const SHA256_K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    fn padded(msg: &[u8]) -> Vec<u8> {
        let mut out = msg.to_vec();
        out.push(0x80);
        while out.len() % 64 != 56 {
            out.push(0);
        }
        out.extend_from_slice(&((msg.len() as u64) * 8).to_be_bytes());
        out
    }

    fn msg_words(block: &[u8]) -> [u32; 16] {
        let mut w = [0u32; 16];
        for (i, word) in w.iter_mut().enumerate() {
            *word = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
        }
        w
    }

    fn sha1_compress(e: &mut TestEnv, block: &[u8], state: &[u32; 5]) -> [u32; 5] {
        let w = msg_words(block);
        let ring = [4u32, 5, 6, 7];
        for (i, reg) in ring.iter().enumerate() {
            e.v[*reg as usize] = v4(w[4 * i..4 * i + 4].try_into().unwrap());
        }
        let mut schedule = vec![
            [w[0], w[1], w[2], w[3]],
            [w[4], w[5], w[6], w[7]],
            [w[8], w[9], w[10], w[11]],
            [w[12], w[13], w[14], w[15]],
        ];
        for step in 0..16 {
            let d = ring[step % 4];
            let n = ring[(step + 1) % 4];
            let m = ring[(step + 2) % 4];
            let last = ring[(step + 3) % 4];
            assert!(exec(e, three(SHA1SU0, d, n, m)));
            assert!(exec(e, two(SHA1SU1, d, last)));
            schedule.push(words4(e.v[d as usize]));
        }

        e.v[0] = v4(state[..4].try_into().unwrap());
        e.v[1] = state[4] as u128;
        for (group, words) in schedule.iter().enumerate() {
            let k = SHA1_K[group / 5];
            e.v[8] = v4([
                words[0].wrapping_add(k),
                words[1].wrapping_add(k),
                words[2].wrapping_add(k),
                words[3].wrapping_add(k),
            ]);
            let (cur, next) = if group % 2 == 0 { (1, 2) } else { (2, 1) };
            assert!(exec(e, two(SHA1H, next, 0)));
            let op = match group / 5 {
                0 => SHA1C,
                1 => SHA1P,
                2 => SHA1M,
                _ => SHA1P,
            };
            assert!(exec(e, three(op, 0, cur, 8)));
        }

        let abcd = words4(e.v[0]);
        let e_word = e.v[1] as u32;
        [abcd[0], abcd[1], abcd[2], abcd[3], e_word]
    }

    fn sha1_digest(msg: &[u8]) -> [u32; 5] {
        let mut state = SHA1_INIT;
        for block in padded(msg).chunks(64) {
            let mut e = TestEnv::default();
            let out = sha1_compress(&mut e, block, &state);
            for (s, word) in state.iter_mut().zip(out) {
                *s = s.wrapping_add(word);
            }
        }
        state
    }

    fn sha256_compress(e: &mut TestEnv, block: &[u8], state: &[u32; 8]) -> [u32; 8] {
        let w = msg_words(block);
        let ring = [4u32, 5, 6, 7];
        for (i, reg) in ring.iter().enumerate() {
            e.v[*reg as usize] = v4(w[4 * i..4 * i + 4].try_into().unwrap());
        }
        let mut schedule = vec![
            [w[0], w[1], w[2], w[3]],
            [w[4], w[5], w[6], w[7]],
            [w[8], w[9], w[10], w[11]],
            [w[12], w[13], w[14], w[15]],
        ];
        for step in 0..12 {
            let d = ring[step % 4];
            let n = ring[(step + 1) % 4];
            let m = ring[(step + 2) % 4];
            let last = ring[(step + 3) % 4];
            assert!(exec(e, two(SHA256SU0, d, n)));
            assert!(exec(e, three(SHA256SU1, d, m, last)));
            schedule.push(words4(e.v[d as usize]));
        }

        e.v[0] = v4(state[..4].try_into().unwrap());
        e.v[1] = v4(state[4..].try_into().unwrap());
        for (group, words) in schedule.iter().enumerate() {
            let k = &SHA256_K[4 * group..4 * group + 4];
            e.v[8] = v4([
                words[0].wrapping_add(k[0]),
                words[1].wrapping_add(k[1]),
                words[2].wrapping_add(k[2]),
                words[3].wrapping_add(k[3]),
            ]);
            e.v[2] = e.v[0];
            assert!(exec(e, three(SHA256H, 0, 1, 8)));
            assert!(exec(e, three(SHA256H2, 1, 2, 8)));
        }

        let mut out = [0u32; 8];
        out[..4].copy_from_slice(&words4(e.v[0]));
        out[4..].copy_from_slice(&words4(e.v[1]));
        out
    }

    fn sha256_digest(msg: &[u8]) -> [u32; 8] {
        let mut state = SHA256_INIT;
        for block in padded(msg).chunks(64) {
            let mut e = TestEnv::default();
            let out = sha256_compress(&mut e, block, &state);
            for (s, word) in state.iter_mut().zip(out) {
                *s = s.wrapping_add(word);
            }
        }
        state
    }

    fn hex(words: &[u32]) -> String {
        words.iter().map(|w| format!("{w:08x}")).collect()
    }

    #[test]
    fn sha1_matches_fips_180() {
        assert_eq!(
            hex(&sha1_digest(b"")),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
        assert_eq!(
            hex(&sha1_digest(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            hex(&sha1_digest(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test]
    fn sha256_matches_fips_180() {
        assert_eq!(
            hex(&sha256_digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256_digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256_digest(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    fn crc_checksum(op: u32, tail: u32, width: u32, msg: &[u8]) -> u32 {
        let mut e = TestEnv::default();
        e.x[1] = 0xffff_ffff;
        let mut consumed = 0;
        while consumed + width as usize <= msg.len() {
            let mut data = 0u64;
            for i in 0..width as usize {
                data |= (msg[consumed + i] as u64) << (8 * i);
            }
            e.x[2] = data;
            assert!(exec(&mut e, three(op, 1, 1, 2)));
            consumed += width as usize;
        }
        for &byte in &msg[consumed..] {
            e.x[2] = byte as u64;
            assert!(exec(&mut e, three(tail, 1, 1, 2)));
        }
        e.x[1] as u32 ^ 0xffff_ffff
    }

    #[test]
    fn crc32_matches_the_known_check_value_across_widths() {
        for (op, tail, width) in [
            (CRC32B, CRC32B, 1),
            (CRC32H, CRC32B, 2),
            (CRC32W, CRC32B, 4),
            (CRC32X, CRC32B, 8),
        ] {
            assert_eq!(
                crc_checksum(op, tail, width, b"123456789"),
                0xcbf4_3926,
                "CRC-32 at width {width}"
            );
        }
        for (op, tail, width) in [
            (CRC32CB, CRC32CB, 1),
            (CRC32CH, CRC32CB, 2),
            (CRC32CW, CRC32CB, 4),
            (CRC32CX, CRC32CB, 8),
        ] {
            assert_eq!(
                crc_checksum(op, tail, width, b"123456789"),
                0xe306_9283,
                "CRC-32C at width {width}"
            );
        }
    }

    #[test]
    fn crc32_accumulates_least_significant_byte_first() {
        let bytes = 0x8070_6050_4030_2010u64;
        assert_eq!(
            crc_update(0x1234_5678, bytes, 8, CRC32_POLY),
            (0..8).fold(0x1234_5678, |c, i| crc_update(
                c,
                bytes >> (8 * i),
                1,
                CRC32_POLY
            ))
        );
        assert_eq!(
            crc_update(0x1234_5678, bytes, 4, CRC32C_POLY),
            (0..4).fold(0x1234_5678, |c, i| crc_update(
                c,
                bytes >> (8 * i),
                1,
                CRC32C_POLY
            ))
        );
    }
}
