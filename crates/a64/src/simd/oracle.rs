use super::{Env, exec, supported};

pub const CODE: u64 = 0x1000;
pub const MEM_BASE: u64 = 0x10_0000;
pub const MEM_SIZE: usize = 0x1_0000;
pub const LDST_BASE: u64 = MEM_BASE + 0x4000;

pub fn mem_pattern() -> Vec<u8> {
    (0..MEM_SIZE)
        .map(|i| ((i * 167 + 13) & 0xff) as u8)
        .collect()
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct State {
    pub x: [u64; 31],
    pub sp: u64,
    pub v: [u128; 32],
    pub nzcv: u64,
    pub fpcr: u64,
    pub fpsr: u64,
    pub mem: Vec<u8>,
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
        self.sp
    }
    fn set_sp(&mut self, v: u64) {
        self.sp = v;
    }
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
    fn load(&mut self, addr: u64, size: u8) -> u64 {
        let mut r = 0u64;
        for k in 0..size as u64 {
            let o = addr.wrapping_add(k).wrapping_sub(MEM_BASE) as usize;
            r |= (*self.mem.get(o).unwrap_or(&0) as u64) << (8 * k);
        }
        r
    }
    fn store(&mut self, addr: u64, size: u8, v: u64) {
        for k in 0..size as u64 {
            let o = addr.wrapping_add(k).wrapping_sub(MEM_BASE) as usize;
            if let Some(b) = self.mem.get_mut(o) {
                *b = (v >> (8 * k)) as u8;
            }
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn f32ish(&mut self) -> u64 {
        const SPECIAL: [u32; 16] = [
            0,
            0x8000_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0x7f80_0001,
            0xffc0_1234,
            0x0000_0001,
            0x8000_0010,
            0x0080_0000,
            0x7f7f_ffff,
            0x4f00_0000,
            0x3f00_0000,
            0x4b00_0001,
        ];
        let r = self.next();
        match r % 4 {
            0 => SPECIAL[(r >> 8) as usize % 16] as u64,
            1 => (((r >> 8) % 2000) as f32 - 1000.0).to_bits() as u64,
            _ => r >> 32,
        }
    }

    fn f64ish(&mut self) -> u64 {
        const SPECIAL: [u64; 12] = [
            0,
            1 << 63,
            0x3ff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x7ff0_0000_0000_0001,
            1,
            0x0010_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0x43e0_0000_0000_0000,
            0xc1e0_0000_0020_0000,
        ];
        let r = self.next();
        match r % 4 {
            0 => SPECIAL[(r >> 8) as usize % 12],
            1 => (((r >> 8) % 4000) as f64 / 4.0 - 500.0).to_bits(),
            2 => self.f32ish() | (self.f32ish() << 32),
            _ => self.next(),
        }
    }
}

pub fn words(n: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| {
            let w = r.next() as u32;
            match r.next() % 10 {
                0..=2 => (w & !(1 << 30) & !(1 << 29)) | 0b1111 << 25,
                3 => {
                    let f = |r: &mut Rng, n: u64| (r.next() % n) as u32;
                    let (q, l, rn, rt, rm) = (
                        f(&mut r, 2),
                        f(&mut r, 2),
                        f(&mut r, 32),
                        f(&mut r, 32),
                        f(&mut r, 32),
                    );
                    let post = if f(&mut r, 2) == 1 {
                        1 << 23 | rm << 16
                    } else {
                        0
                    };
                    let body = if f(&mut r, 2) == 0 {
                        0x0c00_0000 | f(&mut r, 16) << 12 | f(&mut r, 4) << 10
                    } else {
                        0x0d00_0000
                            | f(&mut r, 2) << 21
                            | f(&mut r, 8) << 13
                            | f(&mut r, 2) << 12
                            | f(&mut r, 4) << 10
                    };
                    body | q << 30 | l << 22 | rn << 5 | rt | post
                }
                _ => (w | 0b111 << 25) & !(1 << 31),
            }
        })
        .collect()
}

pub fn initial(word: u32, seed: u64, index: u64) -> State {
    let mut r = Rng(seed ^ index.wrapping_mul(0x2545_f491_4f6c_dd1d));
    let mut x = [0u64; 31];
    for v in x.iter_mut() {
        *v = match r.next() % 3 {
            0 => r.next() & 0xff,
            1 => r.f64ish(),
            _ => r.next(),
        };
    }
    let mut v = [0u128; 32];
    for q in v.iter_mut() {
        let (lo, hi) = if r.next() & 1 == 0 {
            (r.f32ish() | r.f32ish() << 32, r.f32ish() | r.f32ish() << 32)
        } else {
            (r.f64ish(), r.f64ish())
        };
        *q = lo as u128 | (hi as u128) << 64;
    }
    let is_ldst = (word >> 25) & 0b111 != 0b111;
    if is_ldst {
        let (rn, rm) = ((word >> 5) & 31, (word >> 16) & 31);
        if rm != 31 {
            x[rm as usize] = 16 * (r.next() % 8);
        }
        if rn != 31 {
            x[rn as usize] = LDST_BASE;
        }
    }
    let fpcr = [0u64, 1 << 25, 1 << 24, 3 << 24][(index % 4) as usize];
    State {
        x,
        sp: MEM_BASE + 0x8000,
        v,
        nzcv: (r.next() & 0xf) << 28,
        fpcr,
        fpsr: 0,
        mem: mem_pattern(),
    }
}

fn put_state(out: &mut Vec<u8>, s: &State) {
    out.extend_from_slice(&s.fpcr.to_le_bytes());
    out.extend_from_slice(&s.nzcv.to_le_bytes());
    for x in s.x {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(&s.sp.to_le_bytes());
    for q in s.v {
        out.extend_from_slice(&(q as u64).to_le_bytes());
        out.extend_from_slice(&((q >> 64) as u64).to_le_bytes());
    }
}

pub fn generate(n: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, w) in words(n, seed).into_iter().enumerate() {
        out.extend_from_slice(&w.to_le_bytes());
        put_state(&mut out, &initial(w, seed, i as u64));
    }
    out
}

const STATE_BYTES: usize = 8 + 8 + 31 * 8 + 8 + 32 * 16;

fn get_state(b: &[u8], mem: Vec<u8>) -> State {
    let u = |k: usize| u64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap());
    let mut x = [0u64; 31];
    for (i, v) in x.iter_mut().enumerate() {
        *v = u(2 + i);
    }
    let mut v = [0u128; 32];
    for (i, q) in v.iter_mut().enumerate() {
        *q = u(34 + 2 * i) as u128 | (u(35 + 2 * i) as u128) << 64;
    }
    State {
        x,
        sp: u(33),
        v,
        nzcv: u(1),
        fpcr: u(0),
        fpsr: 0,
        mem,
    }
}

#[derive(Default, Debug)]
pub struct Report {
    pub compared: usize,
    pub matched: usize,
    pub missing: Vec<(String, usize, u32)>,
    pub over_accepting: Vec<u32>,
    pub mismatches: Vec<(u32, String)>,
}

pub fn group(w: u32) -> String {
    if (w >> 25) & 0b111 != 0b111 {
        return format!("ldst {:03b}", (w >> 23) & 0b111);
    }
    if w >> 28 & 1 == 1 && w >> 30 & 1 == 0 {
        return format!("fp {:#010x}", w & 0xff20_fc00);
    }
    format!("simd {:#010x}", w & 0xbf20_fc00)
}

fn first_difference(want: &State, got: &State) -> Option<String> {
    for i in 0..31 {
        if want.x[i] != got.x[i] {
            return Some(format!("x{i}: want {:#x} got {:#x}", want.x[i], got.x[i]));
        }
    }
    if want.sp != got.sp {
        return Some(format!("sp: want {:#x} got {:#x}", want.sp, got.sp));
    }
    for i in 0..32 {
        if want.v[i] != got.v[i] {
            return Some(format!(
                "v{i}: want {:#034x} got {:#034x}",
                want.v[i], got.v[i]
            ));
        }
    }
    if want.nzcv != got.nzcv {
        return Some(format!("nzcv: want {:#x} got {:#x}", want.nzcv, got.nzcv));
    }
    if want.fpsr & (1 << 27) != got.fpsr & (1 << 27) {
        return Some(format!(
            "fpsr.qc: want {} got {}",
            want.fpsr >> 27 & 1,
            got.fpsr >> 27 & 1
        ));
    }
    if want.mem != got.mem {
        let k = want
            .mem
            .iter()
            .zip(&got.mem)
            .position(|(a, b)| a != b)
            .unwrap();
        return Some(format!(
            "mem+{k:#x}: want {:#04x} got {:#04x}",
            want.mem[k], got.mem[k]
        ));
    }
    None
}

pub fn check(cases: &[u8], results: &[u8]) -> Report {
    let mut report = Report::default();
    let mut missing = std::collections::BTreeMap::<String, (usize, u32)>::new();
    let (mut ci, mut ri) = (0, 0);
    while ci + 4 + STATE_BYTES <= cases.len() && ri < results.len() {
        let word = u32::from_le_bytes(cases[ci..ci + 4].try_into().unwrap());
        let mut input = get_state(&cases[ci + 4..ci + 4 + STATE_BYTES], mem_pattern());
        ci += 4 + STATE_BYTES;
        let status = results[ri];
        let mut want = get_state(&results[ri + 1..ri + 1 + STATE_BYTES], mem_pattern());
        let fpsr_at = ri + 1 + STATE_BYTES;
        want.fpsr = u64::from_le_bytes(results[fpsr_at..fpsr_at + 8].try_into().unwrap());
        let mut k = fpsr_at + 8;
        let count = u32::from_le_bytes(results[k..k + 4].try_into().unwrap()) as usize;
        k += 4;
        for _ in 0..count {
            let off = u32::from_le_bytes(results[k..k + 4].try_into().unwrap()) as usize;
            want.mem[off] = results[k + 4];
            k += 5;
        }
        ri = k;

        let ours = supported(word);
        match (status, ours) {
            (0, false) => continue,
            (0, true) => {
                report.over_accepting.push(word);
                continue;
            }
            (_, false) => {
                missing.entry(group(word)).or_insert((0, word)).0 += 1;
                continue;
            }
            _ => {}
        }
        report.compared += 1;
        let (vn, vm) = (
            input.v[(word >> 5 & 31) as usize],
            input.v[(word >> 16 & 31) as usize],
        );
        let fpcr = input.fpcr;
        exec(&mut input, word);
        match first_difference(&want, &input) {
            None => report.matched += 1,
            Some(d) => report.mismatches.push((
                word,
                format!("{d} [in: vn={vn:#034x} vm={vm:#034x} fpcr={fpcr:#x}]"),
            )),
        }
    }
    let mut m: Vec<(String, usize, u32)> =
        missing.into_iter().map(|(g, (n, w))| (g, n, w)).collect();
    m.sort_by_key(|g| std::cmp::Reverse(g.1));
    report.missing = m;
    report
}
