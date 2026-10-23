const ERROR: [u8; 4] = [0xff, 0, 0xff, 0xff];

pub fn decode(blocks: &[u8], width: u32, height: u32, bw: u32, bh: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let (bw, bh) = (bw as usize, bh as usize);
    let blocks_x = w.div_ceil(bw);
    let mut out = vec![0u8; w * h * 4];
    let mut texels = [[0u8; 4]; 144];
    for by in 0..h.div_ceil(bh) {
        for bx in 0..blocks_x {
            let i = (by * blocks_x + bx) * 16;
            let Some(block) = blocks.get(i..i + 16) else { return out };
            decode_block(block.try_into().unwrap(), bw, bh, &mut texels);
            for y in 0..bh.min(h - by * bh) {
                for x in 0..bw.min(w - bx * bw) {
                    let o = ((by * bh + y) * w + bx * bw + x) * 4;
                    out[o..o + 4].copy_from_slice(&texels[y * bw + x]);
                }
            }
        }
    }
    out
}

pub fn decode_block(block: &[u8; 16], bw: usize, bh: usize, out: &mut [[u8; 4]; 144]) {
    let texels = bw * bh;
    if decode_inner(u128::from_le_bytes(*block), bw, bh, &mut out[..texels]).is_none() {
        out[..texels].fill(ERROR);
    }
}

fn bits(v: u128, start: u32, count: u32) -> u32 {
    ((v >> start) & ((1u128 << count) - 1)) as u32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    trits: bool,
    quints: bool,
    bits: u32,
}

impl Range {
    const fn new(trits: bool, quints: bool, bits: u32) -> Self {
        Range { trits, quints, bits }
    }
    fn ise_bits(self, n: u32) -> u32 {
        n * self.bits + if self.trits { (8 * n).div_ceil(5) } else if self.quints { (7 * n).div_ceil(3) } else { 0 }
    }
}

const COLOR_RANGES: [Range; 21] = [
    Range::new(false, false, 1),
    Range::new(true, false, 0),
    Range::new(false, false, 2),
    Range::new(false, true, 0),
    Range::new(true, false, 1),
    Range::new(false, false, 3),
    Range::new(false, true, 1),
    Range::new(true, false, 2),
    Range::new(false, false, 4),
    Range::new(false, true, 2),
    Range::new(true, false, 3),
    Range::new(false, false, 5),
    Range::new(false, true, 3),
    Range::new(true, false, 4),
    Range::new(false, false, 6),
    Range::new(false, true, 4),
    Range::new(true, false, 5),
    Range::new(false, false, 7),
    Range::new(false, true, 5),
    Range::new(true, false, 6),
    Range::new(false, false, 8),
];

fn weight_range(r: u32, high: bool) -> Option<Range> {
    Some(match (r, high) {
        (2, false) => Range::new(false, false, 1),
        (3, false) => Range::new(true, false, 0),
        (4, false) => Range::new(false, false, 2),
        (5, false) => Range::new(false, true, 0),
        (6, false) => Range::new(true, false, 1),
        (7, false) => Range::new(false, false, 3),
        (2, true) => Range::new(false, true, 1),
        (3, true) => Range::new(true, false, 2),
        (4, true) => Range::new(false, false, 4),
        (5, true) => Range::new(false, true, 2),
        (6, true) => Range::new(true, false, 3),
        (7, true) => Range::new(false, false, 5),
        _ => return None,
    })
}

#[derive(Debug, PartialEq, Eq)]
struct BlockMode {
    grid_w: u32,
    grid_h: u32,
    dual_plane: bool,
    range: Range,
}

fn block_mode(mode: u32) -> Option<BlockMode> {
    let bit = |n: u32| (mode >> n) & 1;
    let a = (mode >> 5) & 3;
    let (mut dual, mut high) = (bit(10) == 1, bit(9) == 1);
    let (r, w, h);
    if mode & 3 != 0 {
        r = bit(4) | (mode & 3) << 1;
        let b = (mode >> 7) & 3;
        (w, h) = match (mode >> 2) & 3 {
            0 => (b + 4, a + 2),
            1 => (b + 8, a + 2),
            2 => (a + 2, b + 8),
            _ if bit(8) == 0 => (a + 2, (b & 1) + 6),
            _ => ((b & 1) + 2, a + 2),
        };
    } else {
        r = bit(4) | ((mode >> 2) & 3) << 1;
        if mode & 0xf == 0 {
            return None;
        }
        (w, h) = match (mode >> 7) & 3 {
            0 => (12, a + 2),
            1 => (a + 2, 12),
            2 => {
                (dual, high) = (false, false);
                (a + 6, ((mode >> 9) & 3) + 6)
            }
            _ => match a {
                0 => (6, 10),
                1 => (10, 6),
                _ => return None,
            },
        };
    }
    Some(BlockMode { grid_w: w, grid_h: h, dual_plane: dual, range: weight_range(r, high)? })
}

fn decode_inner(block: u128, bw: usize, bh: usize, out: &mut [[u8; 4]]) -> Option<()> {
    let mode = bits(block, 0, 11);
    if mode & 0x1ff == 0x1fc {
        if bits(block, 9, 1) == 1 {
            return None;
        }
        let c = |i: u32| (bits(block, 64 + 16 * i, 16) >> 8) as u8;
        out.fill([c(0), c(1), c(2), c(3)]);
        return Some(());
    }
    let m = block_mode(mode)?;
    let (gw, gh) = (m.grid_w as usize, m.grid_h as usize);
    if gw > bw || gh > bh {
        return None;
    }
    let planes = if m.dual_plane { 2 } else { 1 };
    let weight_count = (gw * gh * planes) as u32;
    let weight_bits = m.range.ise_bits(weight_count);
    if weight_count > 64 || !(24..=96).contains(&weight_bits) {
        return None;
    }
    let partitions = bits(block, 11, 2) + 1;
    if m.dual_plane && partitions == 4 {
        return None;
    }

    let mut below_weights = 128 - weight_bits;
    let mut cems = [0u32; 4];
    let color_start;
    if partitions == 1 {
        cems[0] = bits(block, 13, 4);
        color_start = 17;
    } else {
        color_start = 29;
        let low = bits(block, 23, 6);
        if low & 3 == 0 {
            cems[..partitions as usize].fill(low >> 2);
        } else {
            let extra = 3 * partitions - 4;
            below_weights = below_weights.checked_sub(extra)?;
            let enc = low | bits(block, below_weights, extra) << 6;
            let base = (low & 3) - 1;
            for (i, cem) in cems.iter_mut().enumerate().take(partitions as usize) {
                let class = (enc >> (2 + i as u32)) & 1;
                let m = (enc >> (2 + partitions + 2 * i as u32)) & 3;
                *cem = (base + class) << 2 | m;
            }
        }
    }
    let ccs = if m.dual_plane {
        below_weights = below_weights.checked_sub(2)?;
        Some(bits(block, below_weights, 2) as usize)
    } else {
        None
    };
    let cems = &cems[..partitions as usize];
    let color_values: u32 = cems.iter().map(|c| (c >> 2) * 2 + 2).sum();
    if color_values > 18 {
        return None;
    }
    let color_bits = below_weights.checked_sub(color_start)?;
    let color_level = COLOR_RANGES.iter().rposition(|r| r.ise_bits(color_values) <= color_bits)?;
    if color_level < 4 {
        return None;
    }
    let color_range = COLOR_RANGES[color_level];
    let mut colors = [0u32; 18];
    ise(block, color_start, color_range, &mut colors[..color_values as usize]);
    for c in &mut colors[..color_values as usize] {
        *c = unquantize_color(*c, color_range);
    }
    let mut endpoints = [[[0u32; 4]; 2]; 4];
    let mut at = 0;
    for (e, &cem) in endpoints.iter_mut().zip(cems) {
        *e = endpoint(cem, &colors[at..])?;
        at += ((cem >> 2) * 2 + 2) as usize;
    }

    let mut grid = [0u32; 64];
    ise(block.reverse_bits(), 0, m.range, &mut grid[..weight_count as usize]);
    for w in &mut grid[..weight_count as usize] {
        *w = unquantize_weight(*w, m.range);
    }

    let seed = bits(block, 13, 10);
    let small = bw * bh < 31;
    let ds = (1024 + bw / 2) / (bw - 1).max(1);
    let dt = (1024 + bh / 2) / (bh - 1).max(1);
    for t in 0..bh {
        for s in 0..bw {
            let gs = (ds * s * (gw - 1) + 32) >> 6;
            let gt = (dt * t * (gh - 1) + 32) >> 6;
            let (js, fs, jt, ft) = (gs >> 4, gs & 0xf, gt >> 4, gt & 0xf);
            let w11 = (fs * ft + 8) >> 4;
            let taps = [(0, 0, 16 + w11 - fs - ft), (1, 0, fs - w11), (0, 1, ft - w11), (1, 1, w11)];
            let weight = |plane: usize| -> u32 {
                let mut sum = 0;
                for &(dx, dy, f) in &taps {
                    if f != 0 {
                        sum += grid.get(((jt + dy) * gw + js + dx) * planes + plane).copied().unwrap_or(0) * f as u32;
                    }
                }
                (sum + 8) >> 4
            };
            let w0 = weight(0);
            let w1 = if ccs.is_some() { weight(1) } else { w0 };
            let part = if partitions == 1 { 0 } else { select_partition(seed, s as u32, t as u32, 0, partitions, small) };
            let [e0, e1] = endpoints[part as usize];
            let mut texel = [0u8; 4];
            for (c, v) in texel.iter_mut().enumerate() {
                let w = if ccs == Some(c) { w1 } else { w0 };
                let (c0, c1) = (e0[c] << 8 | 0x80, e1[c] << 8 | 0x80);
                *v = ((c0 * (64 - w) + c1 * w + 32) >> 6 >> 8) as u8;
            }
            out[t * bw + s] = texel;
        }
    }
    Some(())
}

fn ise(data: u128, start: u32, range: Range, out: &mut [u32]) {
    let end = start + range.ise_bits(out.len() as u32);
    let mut pos = start;
    let mut read = |n: u32| -> u32 {
        let mut v = 0;
        for i in 0..n {
            if pos < end.min(128) {
                v |= bits(data, pos, 1) << i;
            }
            pos += 1;
        }
        v
    };
    let b = range.bits;
    if range.trits {
        for chunk in out.chunks_mut(5) {
            let mut m = [0u32; 5];
            let mut t = 0;
            for (mi, (at, n)) in m.iter_mut().zip([(0, 2), (2, 2), (4, 1), (5, 2), (7, 1)]) {
                *mi = read(b);
                t |= read(n) << at;
            }
            let tr = trits(t);
            for (i, v) in chunk.iter_mut().enumerate() {
                *v = tr[i] << b | m[i];
            }
        }
    } else if range.quints {
        for chunk in out.chunks_mut(3) {
            let mut m = [0u32; 3];
            let mut q = 0;
            for (mi, (at, n)) in m.iter_mut().zip([(0, 3), (3, 2), (5, 2)]) {
                *mi = read(b);
                q |= read(n) << at;
            }
            let qu = quints(q);
            for (i, v) in chunk.iter_mut().enumerate() {
                *v = qu[i] << b | m[i];
            }
        }
    } else {
        for v in out {
            *v = read(b);
        }
    }
}

fn trits(t: u32) -> [u32; 5] {
    let bit = |v: u32, n: u32| (v >> n) & 1;
    let (c, t4, t3);
    if (t >> 2) & 7 == 7 {
        c = (t >> 5) << 2 | (t & 3);
        (t4, t3) = (2, 2);
    } else {
        c = t & 0x1f;
        if (t >> 5) & 3 == 3 {
            (t4, t3) = (2, bit(t, 7));
        } else {
            (t4, t3) = (bit(t, 7), (t >> 5) & 3);
        }
    }
    let (t2, t1, t0);
    if c & 3 == 3 {
        t2 = 2;
        t1 = bit(c, 4);
        t0 = bit(c, 3) << 1 | (bit(c, 2) & !bit(c, 3) & 1);
    } else if (c >> 2) & 3 == 3 {
        (t2, t1, t0) = (2, 2, c & 3);
    } else {
        t2 = bit(c, 4);
        t1 = (c >> 2) & 3;
        t0 = bit(c, 1) << 1 | (bit(c, 0) & !bit(c, 1) & 1);
    }
    [t0, t1, t2, t3, t4]
}

fn quints(q: u32) -> [u32; 3] {
    let bit = |v: u32, n: u32| (v >> n) & 1;
    if (q >> 1) & 3 == 3 && (q >> 5) & 3 == 0 {
        let nq0 = !bit(q, 0) & 1;
        let q2 = bit(q, 0) << 2 | (bit(q, 4) & nq0) << 1 | (bit(q, 3) & nq0);
        return [4, 4, q2];
    }
    let (q2, c);
    if (q >> 1) & 3 == 3 {
        q2 = 4;
        c = ((q >> 3) & 3) << 3 | (!(q >> 5) & 3) << 1 | (q & 1);
    } else {
        q2 = (q >> 5) & 3;
        c = q & 0x1f;
    }
    let (q1, q0) = if c & 7 == 5 { (4, (c >> 3) & 3) } else { ((c >> 3) & 3, c & 7) };
    [q0, q1, q2]
}

fn replicate(v: u32, from: u32, to: u32) -> u32 {
    if from == 0 {
        return 0;
    }
    let mut out = 0;
    let mut have = 0;
    while have < to {
        out = out << from | v;
        have += from;
    }
    out >> (have - to)
}

fn unquantize_color(v: u32, r: Range) -> u32 {
    if !r.trits && !r.quints {
        return replicate(v, r.bits, 8);
    }
    let m = v & ((1 << r.bits) - 1);
    let d = v >> r.bits;
    let a = if m & 1 != 0 { 0x1ff } else { 0 };
    let bit = |n: u32| (m >> n) & 1;
    let (b, c) = match (r.trits, r.bits) {
        (true, 1) => (0, 204),
        (true, 2) => (bit(1) * 0x116, 93),
        (true, 3) => ((bit(2) << 8 | bit(1) << 7) | (bit(2) << 3 | bit(1) << 2) | (bit(2) << 1 | bit(1)), 44),
        (true, 4) => ((m >> 1) << 6 | (m >> 1), 22),
        (true, 5) => ((m >> 1) << 5 | (m >> 3), 11),
        (true, 6) => ((m >> 1) << 4 | (m >> 5), 5),
        (false, 1) => (0, 113),
        (false, 2) => (bit(1) * 0x10c, 54),
        (false, 3) => ((m >> 1) << 7 | (m >> 1) << 1 | (m >> 2), 26),
        (false, 4) => ((m >> 1) << 6 | (m >> 2), 13),
        (false, 5) => ((m >> 1) << 5 | (m >> 4), 6),
        _ => return 0,
    };
    let t = (d * c + b) ^ a;
    (a & 0x80) | (t >> 2)
}

fn unquantize_weight(v: u32, r: Range) -> u32 {
    let t = if !r.trits && !r.quints {
        replicate(v, r.bits, 6)
    } else if r.bits == 0 {
        if r.trits {
            [0, 32, 63][v as usize]
        } else {
            [0, 16, 32, 47, 63][v as usize]
        }
    } else {
        let m = v & ((1 << r.bits) - 1);
        let d = v >> r.bits;
        let a = if m & 1 != 0 { 0x7f } else { 0 };
        let bit = |n: u32| (m >> n) & 1;
        let (b, c) = match (r.trits, r.bits) {
            (true, 1) => (0, 50),
            (true, 2) => (bit(1) * 0x45, 23),
            (true, _) => ((m >> 1) << 5 | (m >> 1), 11),
            (false, 1) => (0, 28),
            (false, _) => (bit(1) * 0x42, 13),
        };
        let t = (d * c + b) ^ a;
        (a & 0x20) | (t >> 2)
    };
    if t > 32 { t + 1 } else { t }
}

fn bit_transfer_signed(a: u32, b: u32) -> (i32, i32) {
    let b = (b >> 1) | (a & 0x80);
    let mut a = ((a >> 1) & 0x3f) as i32;
    if a & 0x20 != 0 {
        a -= 0x40;
    }
    (a, b as i32)
}

fn blue_contract(c: [i32; 4]) -> [i32; 4] {
    [(c[0] + c[2]) >> 1, (c[1] + c[2]) >> 1, c[2], c[3]]
}

fn endpoint(cem: u32, v: &[u32]) -> Option<[[u32; 4]; 2]> {
    let v = |i: usize| v[i] as i32;
    let (e0, e1): ([i32; 4], [i32; 4]) = match cem {
        0 => ([v(0), v(0), v(0), 255], [v(1), v(1), v(1), 255]),
        1 => {
            let l0 = (v(0) >> 2) | (v(1) & 0xc0);
            let l1 = (l0 + (v(1) & 0x3f)).min(255);
            ([l0, l0, l0, 255], [l1, l1, l1, 255])
        }
        4 => ([v(0), v(0), v(0), v(2)], [v(1), v(1), v(1), v(3)]),
        5 => {
            let (o0, b0) = bit_transfer_signed(v(1) as u32, v(0) as u32);
            let (o1, b1) = bit_transfer_signed(v(3) as u32, v(2) as u32);
            ([b0, b0, b0, b1], [b0 + o0, b0 + o0, b0 + o0, b1 + o1])
        }
        6 => {
            let s = |i: usize| (v(i) * v(3)) >> 8;
            ([s(0), s(1), s(2), 255], [v(0), v(1), v(2), 255])
        }
        8 | 12 => {
            let (a0, a1) = if cem == 12 { (v(6), v(7)) } else { (255, 255) };
            if v(1) + v(3) + v(5) >= v(0) + v(2) + v(4) {
                ([v(0), v(2), v(4), a0], [v(1), v(3), v(5), a1])
            } else {
                (blue_contract([v(1), v(3), v(5), a1]), blue_contract([v(0), v(2), v(4), a0]))
            }
        }
        9 | 13 => {
            let (or, r) = bit_transfer_signed(v(1) as u32, v(0) as u32);
            let (og, g) = bit_transfer_signed(v(3) as u32, v(2) as u32);
            let (ob, b) = bit_transfer_signed(v(5) as u32, v(4) as u32);
            let (oa, a) = if cem == 13 { bit_transfer_signed(v(7) as u32, v(6) as u32) } else { (0, 255) };
            if or + og + ob >= 0 {
                ([r, g, b, a], [r + or, g + og, b + ob, a + oa])
            } else {
                (blue_contract([r + or, g + og, b + ob, a + oa]), blue_contract([r, g, b, a]))
            }
        }
        10 => {
            let s = |i: usize| (v(i) * v(3)) >> 8;
            ([s(0), s(1), s(2), v(4)], [v(0), v(1), v(2), v(5)])
        }
        _ => return None,
    };
    let clamp = |e: [i32; 4]| e.map(|c| c.clamp(0, 255) as u32);
    Some([clamp(e0), clamp(e1)])
}

fn hash52(mut p: u32) -> u32 {
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p
}

fn select_partition(seed: u32, mut x: u32, mut y: u32, mut z: u32, count: u32, small: bool) -> u32 {
    if small {
        (x, y, z) = (x << 1, y << 1, z << 1);
    }
    let seed = seed + (count - 1) * 1024;
    let r = hash52(seed);
    let mut s = [
        r & 0xf,
        (r >> 4) & 0xf,
        (r >> 8) & 0xf,
        (r >> 12) & 0xf,
        (r >> 16) & 0xf,
        (r >> 20) & 0xf,
        (r >> 24) & 0xf,
        (r >> 28) & 0xf,
        (r >> 18) & 0xf,
        (r >> 22) & 0xf,
        (r >> 26) & 0xf,
        r.rotate_left(2) & 0xf,
    ];
    for v in &mut s {
        *v *= *v;
    }
    let (sh1, sh2) = if seed & 1 != 0 {
        (if seed & 2 != 0 { 4 } else { 5 }, if count == 3 { 6 } else { 5 })
    } else {
        (if count == 3 { 6 } else { 5 }, if seed & 2 != 0 { 4 } else { 5 })
    };
    let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
    for (i, v) in s.iter_mut().enumerate() {
        *v >>= if i >= 8 { sh3 } else if i % 2 == 0 { sh1 } else { sh2 };
    }
    let a = (s[0] * x + s[1] * y + s[10] * z + (r >> 14)) & 0x3f;
    let b = (s[2] * x + s[3] * y + s[11] * z + (r >> 10)) & 0x3f;
    let c = if count >= 3 { (s[4] * x + s[5] * y + s[8] * z + (r >> 6)) & 0x3f } else { 0 };
    let d = if count >= 4 { (s[6] * x + s[7] * y + s[9] * z + (r >> 2)) & 0x3f } else { 0 };
    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

#[cfg(test)]
#[allow(clippy::unusual_byte_groupings)]
mod tests {
    use super::*;

    fn pack(fields: &[(u128, u32)]) -> u128 {
        let mut v = 0u128;
        let mut at = 0;
        for &(f, n) in fields {
            v |= f << at;
            at += n;
        }
        v
    }

    fn decode_one(block: u128, bw: usize, bh: usize) -> Vec<[u8; 4]> {
        let mut out = [[0u8; 4]; 144];
        decode_block(&block.to_le_bytes(), bw, bh, &mut out);
        out[..bw * bh].to_vec()
    }

    #[test]
    fn void_extent_is_constant() {
        let block = pack(&[(0x1fc, 9), (0, 1), (3, 2), ((1 << 52) - 1, 52), (0x1234, 16), (0xabcd, 16), (0x0080, 16), (0xffff, 16)]);
        for (bw, bh) in [(4, 4), (8, 5), (12, 12)] {
            assert!(decode_one(block, bw, bh).iter().all(|&t| t == [0x12, 0xab, 0x00, 0xff]));
        }
        let hdr = block | 1 << 9;
        assert!(decode_one(hdr, 4, 4).iter().all(|&t| t == ERROR));
    }

    #[test]
    fn block_modes() {
        assert_eq!(block_mode(MODE_4X4_2BIT), Some(BlockMode { grid_w: 4, grid_h: 4, dual_plane: false, range: Range::new(false, false, 2) }));
        let m = 0b1_1_11_00_1_11_00;
        assert_eq!(block_mode(m), Some(BlockMode { grid_w: 6, grid_h: 10, dual_plane: true, range: Range::new(false, false, 5) }));
        let m = 0b11_10_01_1_11_00;
        assert_eq!(block_mode(m), Some(BlockMode { grid_w: 7, grid_h: 9, dual_plane: false, range: Range::new(false, false, 3) }));
        let m = 0b0_0_01_00_1_01_01;
        assert_eq!(block_mode(m), Some(BlockMode { grid_w: 9, grid_h: 2, dual_plane: false, range: Range::new(true, false, 0) }));
        assert_eq!(block_mode(0b000_0000_0000), None);
    }

    #[test]
    fn trit_and_quint_packing_covers_all_tuples() {
        let tuples: std::collections::HashSet<_> = (0..256).map(trits).collect();
        assert_eq!(tuples.len(), 243);
        let tuples: std::collections::HashSet<_> = (0..128).map(quints).collect();
        assert_eq!(tuples.len(), 125);
        assert_eq!(trits(0), [0; 5]);
        assert_eq!(quints(0), [0; 3]);
    }

    fn levels(r: Range) -> u32 {
        (if r.trits { 3 } else if r.quints { 5 } else { 1 }) << r.bits
    }

    #[test]
    fn unquantization_spans_the_range() {
        for r in &COLOR_RANGES[4..] {
            let mut v: Vec<u32> = (0..levels(*r)).map(|x| unquantize_color(x, *r)).collect();
            v.sort();
            v.dedup();
            assert_eq!(v.len() as u32, levels(*r), "{r:?}");
            assert_eq!((v[0], *v.last().unwrap()), (0, 255), "{r:?}");
        }
        for rr in 2..=7 {
            for h in [false, true] {
                let r = weight_range(rr, h).unwrap();
                let mut v: Vec<u32> = (0..levels(r)).map(|x| unquantize_weight(x, r)).collect();
                v.sort();
                v.dedup();
                assert_eq!(v.len() as u32, levels(r), "{r:?}");
                assert_eq!((v[0], *v.last().unwrap()), (0, 64), "{r:?}");
            }
        }
        let r = Range::new(true, false, 1);
        let got: Vec<u32> = (0..6).map(|v| unquantize_color(v, r)).collect();
        assert_eq!(got, [0, 255, 51, 204, 102, 153]);
        let r = Range::new(false, true, 0);
        let got: Vec<u32> = (0..5).map(|v| unquantize_weight(v, r)).collect();
        assert_eq!(got, [0, 16, 32, 48, 64]);
    }

    const MODE_4X4_2BIT: u32 = 0b0_0_00_10_0_00_10;

    fn single_partition_block(colors: [u128; 6], weights: [u8; 16]) -> u128 {
        let mut b = pack(&[(MODE_4X4_2BIT as u128, 11), (0, 2), (8, 4)]);
        for (i, &c) in colors.iter().enumerate() {
            b |= c << (17 + 8 * i);
        }
        for (i, &w) in weights.iter().enumerate() {
            for k in 0..2 {
                if w >> k & 1 != 0 {
                    b |= 1u128 << (127 - 2 * i - k);
                }
            }
        }
        b
    }

    const COLORS: [u128; 6] = [10, 250, 20, 240, 30, 230];

    #[test]
    fn single_partition_rgb() {
        let weights = [0, 3, 1, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 0, 3, 0];
        let texels = decode_one(single_partition_block(COLORS, weights), 4, 4);
        let lerp = |a: u32, b: u32, w: u32| {
            let (a, b) = (a << 8 | 0x80, b << 8 | 0x80);
            ((a * (64 - w) + b * w + 32) >> 6 >> 8) as u8
        };
        for (i, t) in texels.iter().enumerate() {
            let w = [0, 21, 43, 64][weights[i] as usize];
            let want = [lerp(10, 250, w), lerp(20, 240, w), lerp(30, 230, w), 255];
            assert_eq!(*t, want, "texel {i}");
        }
        assert_eq!(texels[0], [10, 20, 30, 255]);
        assert_eq!(texels[1], [250, 240, 230, 255]);
    }

    #[test]
    fn single_partition_blue_contract() {
        let b = single_partition_block([200, 100, 200, 100, 200, 100], [0; 16]);
        assert!(decode_one(b, 4, 4).iter().all(|&t| t == [100, 100, 100, 255]));
    }

    #[test]
    fn weight_infill_interpolates() {
        let b = single_partition_block(COLORS, [0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3]);
        let texels = decode_one(b, 8, 8);
        for y in 0..8 {
            let row: Vec<u8> = (0..8).map(|x| texels[y * 8 + x][0]).collect();
            assert!(row.windows(2).all(|w| w[0] <= w[1]), "{row:?}");
            assert_eq!((row[0], row[7]), (10, 250));
        }
        assert!(decode_one(b, 4, 3).iter().all(|&t| t == ERROR));
    }

    #[test]
    fn partition_selection_stays_in_range() {
        let mut both = 0;
        for seed in 0..1024 {
            let mut used = [false; 2];
            for y in 0..4 {
                for x in 0..4 {
                    used[select_partition(seed, x, y, 0, 2, true) as usize] = true;
                }
            }
            both += (used[0] && used[1]) as u32;
            for y in 0..12 {
                for x in 0..12 {
                    assert!(select_partition(seed, x, y, 0, 3, false) < 3);
                    assert!(select_partition(seed, x, y, 0, 4, false) < 4);
                }
            }
        }
        assert!(both > 512, "{both}");
    }

    #[test]
    fn decode_crops_partial_blocks() {
        let block = pack(&[(0x1fc, 9), (0, 1), (3, 2), ((1 << 52) - 1, 52), (0xff00, 16), (0, 16), (0, 16), (0xffff, 16)]);
        let data: Vec<u8> = (0..4).flat_map(|_| block.to_le_bytes()).collect();
        let out = decode(&data, 7, 5, 4, 4);
        assert_eq!(out.len(), 7 * 5 * 4);
        assert!(out.chunks(4).all(|t| t == [0xff, 0, 0, 0xff]));
    }
}
