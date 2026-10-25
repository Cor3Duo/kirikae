use kirikae_a64::simd::{self, Env};

use crate::cpu::Cpu;

// Every method is a field (or a few shifts) on `Cpu`, so the helper monomorphises the
// whole decoder for `Cpu` and inlines these; a trait object could not.
impl Env for Cpu {
    #[inline]
    fn x(&self, n: u32) -> u64 {
        if n == 31 { 0 } else { self.x[n as usize] }
    }
    #[inline]
    fn set_x(&mut self, n: u32, v: u64) {
        if n != 31 {
            self.x[n as usize] = v;
        }
    }
    #[inline]
    fn sp(&self) -> u64 {
        self.sp
    }
    #[inline]
    fn set_sp(&mut self, v: u64) {
        self.sp = v;
    }
    #[inline]
    fn v(&self, n: u32) -> u128 {
        let [lo, hi] = self.v[n as usize];
        lo as u128 | (hi as u128) << 64
    }
    #[inline]
    fn set_v(&mut self, n: u32, v: u128) {
        self.v[n as usize] = [v as u64, (v >> 64) as u64];
    }
    #[inline]
    fn nzcv(&self) -> u64 {
        self.nzcv
    }
    #[inline]
    fn set_nzcv(&mut self, v: u64) {
        self.nzcv = v;
    }
    #[inline]
    fn fpcr(&self) -> u64 {
        self.fpcr
    }
    #[inline]
    fn fpsr(&self) -> u64 {
        self.fpsr
    }
    #[inline]
    fn set_fpsr(&mut self, v: u64) {
        self.fpsr = v;
    }
    #[inline]
    fn load(&mut self, addr: u64, size: u8) -> u64 {
        Cpu::load(self, size, addr)
    }
    #[inline]
    fn store(&mut self, addr: u64, size: u8, v: u64) {
        Cpu::store(self, size, addr, v);
    }
}

pub fn exec(c: &mut Cpu, word: u32) {
    exec_words(c, &[word]);
}

/// Runs consecutive SIMD/FP words. The caller owns the program counter, so the address
/// reported for a rejected word is the one `Cpu::pc` holds on entry.
pub fn exec_words(c: &mut Cpu, words: &[u32]) {
    let done = simd::exec_run(c, words);
    if let Some(&word) = words.get(done) {
        let pc = c.pc;
        crate::horizon::undefined(c, pc, word);
    }
}
