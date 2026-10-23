use std::mem::offset_of;

use kirikae_a64::{Op, SysReg};

use crate::cpu::Cpu;

pub type GuestFn = extern "C" fn(&mut Cpu);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Entry {
    pub addr: u64,
    pub func: GuestFn,
}

pub const OFF_X: usize = offset_of!(Cpu, x);
pub const OFF_SP: usize = offset_of!(Cpu, sp);
pub const OFF_V: usize = offset_of!(Cpu, v);
pub const OFF_PC: usize = offset_of!(Cpu, pc);
pub const OFF_NZCV: usize = offset_of!(Cpu, nzcv);
pub const OFF_HALTED: usize = offset_of!(Cpu, halted);
pub const OFF_FPCR: usize = offset_of!(Cpu, fpcr);
pub const OFF_EXCL_ADDR: usize = offset_of!(Cpu, excl_addr);
pub const OFF_EXCL_LO: usize = offset_of!(Cpu, excl_lo);
pub const OFF_EXCL_HI: usize = offset_of!(Cpu, excl_hi);
pub const OFF_BUDGET: usize = Cpu::OFF_BUDGET;
pub const OFF_MEM: usize = Cpu::OFF_MEM;

pub fn helpers() -> Vec<(&'static str, *const u8)> {
    vec![
        ("kirikae_svc", kirikae_svc as *const u8),
        ("kirikae_undefined", kirikae_undefined as *const u8),
        ("kirikae_call_indirect", kirikae_call_indirect as *const u8),
        ("kirikae_poll_slow", kirikae_poll_slow as *const u8),
        ("kirikae_sysreg", kirikae_sysreg as *const u8),
        ("kirikae_set_sysreg", kirikae_set_sysreg as *const u8),
        ("kirikae_op", kirikae_op as *const u8),
        ("kirikae_simd", kirikae_simd as *const u8),
        ("kirikae_store_exclusive", kirikae_store_exclusive as *const u8),
    ]
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_svc(c: &mut Cpu, imm: u32) {
    crate::horizon::svc(c, imm as u16);
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_undefined(c: &mut Cpu, addr: u64, word: u32) {
    crate::horizon::undefined(c, addr, word);
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_call_indirect(c: &mut Cpu, ret: u64) {
    crate::process::call_indirect(c, ret);
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_poll_slow(c: &mut Cpu) -> u32 {
    c.refill_budget();
    crate::sched::yield_now(c) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_store_exclusive(c: &mut Cpu, size: u32, addr: u64, lo: u64, hi: u64) -> u64 {
    c.store_exclusive(size as u8, addr, lo, hi)
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_sysreg(c: &mut Cpu, id: u32) -> u64 {
    c.sysreg(SysReg::ALL[id as usize])
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_set_sysreg(c: &mut Cpu, id: u32, value: u64) {
    c.set_sysreg(SysReg::ALL[id as usize], value);
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_simd(c: &mut Cpu, word: u32) {
    crate::prof::count_simd(word);
    crate::simd::exec(c, word);
}

#[unsafe(no_mangle)]
pub extern "C" fn kirikae_op(op: u32, a: u64, b: u64, c: u64, d: u64, e: u64) -> u64 {
    crate::prof::count_op(op);
    let op = Op::ALL[op as usize];
    op.eval(&[a, b, c, d, e][..op.arity()])
}
