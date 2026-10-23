use kirikae_a64::simd::{self, Env};

use crate::cpu::Cpu;

impl Env for Cpu {
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
        let [lo, hi] = self.v[n as usize];
        lo as u128 | (hi as u128) << 64
    }
    fn set_v(&mut self, n: u32, v: u128) {
        self.v[n as usize] = [v as u64, (v >> 64) as u64];
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
        Cpu::load(self, size, addr)
    }
    fn store(&mut self, addr: u64, size: u8, v: u64) {
        Cpu::store(self, size, addr, v);
    }
}

pub fn exec(c: &mut Cpu, word: u32) {
    if !simd::exec(c, word) {
        let pc = c.pc;
        crate::horizon::undefined(c, pc, word);
    }
}
