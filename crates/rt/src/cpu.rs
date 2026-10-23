use std::sync::Arc;

use crate::kernel::KThread;
use crate::mem::ADDR_MASK;
use crate::process::Process;
use kirikae_a64::SysReg;

pub const HALT_PC: u64 = 0xFFFF_FFFF_FFFF_0000;
pub const NO_MONITOR: u64 = u64::MAX;

#[repr(C)]
pub struct Cpu {
    pub x: [u64; 31],
    pub sp: u64,
    pub v: [[u64; 2]; 32],
    pub pc: u64,
    pub nzcv: u64,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    pub fpcr: u64,
    pub fpsr: u64,
    pub excl_addr: u64,
    pub excl_lo: u64,
    pub excl_hi: u64,
    pub halted: bool,
    budget: u32,
    pub(crate) sched_seen: u64,
    mem: *mut u8,
    pub(crate) process: &'static Process,
    pub thread: Arc<KThread>,
}

macro_rules! accessors {
    ($($ld:ident, $st:ident, $t:ty;)*) => {$(
        #[inline(always)]
        pub fn $ld(&self, addr: u64) -> u64 {
            // SAFETY: `host` stays inside the guest reservation (see mem.rs).
            unsafe { (self.host(addr) as *const $t).read_unaligned() as u64 }
        }
        #[inline(always)]
        pub fn $st(&self, addr: u64, value: u64) {
            // SAFETY: as above.
            unsafe { (self.host(addr) as *mut $t).write_unaligned(value as $t) }
        }
    )*};
}

impl Cpu {
    pub(crate) fn new(process: &'static Process, thread: Arc<KThread>) -> Cpu {
        Cpu {
            x: [0; 31],
            sp: 0,
            v: [[0; 2]; 32],
            pc: 0,
            nzcv: 0,
            tpidr_el0: 0,
            tpidrro_el0: 0,
            fpcr: 0,
            fpsr: 0,
            excl_addr: NO_MONITOR,
            excl_lo: 0,
            excl_hi: 0,
            halted: false,
            budget: crate::sched::POLL_INTERVAL,
            sched_seen: 0,
            mem: process.memory.base(),
            process,
            thread,
        }
    }

    #[inline(always)]
    fn host(&self, addr: u64) -> *mut u8 {
        // SAFETY: the mask bounds the offset by the reservation size.
        unsafe { self.mem.add((addr & ADDR_MASK) as usize) }
    }

    accessors! {
        ld8, st8, u8;
        ld16, st16, u16;
        ld32, st32, u32;
        ld64, st64, u64;
    }

    pub fn load(&self, size: u8, addr: u64) -> u64 {
        match size {
            1 => self.ld8(addr),
            2 => self.ld16(addr),
            4 => self.ld32(addr),
            _ => self.ld64(addr),
        }
    }

    pub fn store(&self, size: u8, addr: u64, value: u64) {
        match size {
            1 => self.st8(addr, value),
            2 => self.st16(addr, value),
            4 => self.st32(addr, value),
            _ => self.st64(addr, value),
        }
    }

    pub fn monitor(&mut self, addr: u64, lo: u64, hi: u64) {
        self.excl_addr = addr;
        self.excl_lo = lo;
        self.excl_hi = hi;
    }

    pub fn store_exclusive(&mut self, size: u8, addr: u64, lo: u64, hi: u64) -> u64 {
        if std::mem::replace(&mut self.excl_addr, NO_MONITOR) != addr {
            return 1;
        }
        let (old_lo, old_hi) = (self.excl_lo, self.excl_hi);
        let p = self.host(addr);
        let stored = if !addr.is_multiple_of(size as u64) {
            let same = self.load(size.min(8), addr) == old_lo && (size < 16 || self.ld64(addr + 8) == old_hi);
            if same {
                self.store(size.min(8), addr, lo);
                if size == 16 {
                    self.st64(addr + 8, hi);
                }
            }
            same
        } else {
            use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering::SeqCst};
            // SAFETY: `p` is inside the guest reservation and aligned for the size.
            unsafe {
                match size {
                    1 => AtomicU8::from_ptr(p).compare_exchange(old_lo as u8, lo as u8, SeqCst, SeqCst).is_ok(),
                    2 => AtomicU16::from_ptr(p.cast()).compare_exchange(old_lo as u16, lo as u16, SeqCst, SeqCst).is_ok(),
                    4 => AtomicU32::from_ptr(p.cast()).compare_exchange(old_lo as u32, lo as u32, SeqCst, SeqCst).is_ok(),
                    8 => AtomicU64::from_ptr(p.cast()).compare_exchange(old_lo, lo, SeqCst, SeqCst).is_ok(),
                    _ => cas128(p, [old_lo, old_hi], [lo, hi]),
                }
            }
        };
        !stored as u64
    }

    pub fn update32(&self, addr: u64, mut f: impl FnMut(u32) -> Option<u32>) -> u32 {
        use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
        if !addr.is_multiple_of(4) {
            let old = self.ld32(addr) as u32;
            if let Some(new) = f(old) {
                self.st32(addr, new as u64);
            }
            return old;
        }
        // SAFETY: inside the guest reservation, aligned.
        let word = unsafe { AtomicU32::from_ptr(self.host(addr).cast()) };
        match word.fetch_update(SeqCst, SeqCst, f) {
            Ok(old) | Err(old) => old,
        }
    }

    pub fn sysreg(&self, r: SysReg) -> u64 {
        match r {
            SysReg::TpidrEl0 => self.tpidr_el0,
            SysReg::TpidrroEl0 => self.tpidrro_el0,
            SysReg::CntvctEl0 | SysReg::CntpctEl0 => self.process.ticks(),
            SysReg::CntfrqEl0 => crate::horizon::TICK_HZ,
            SysReg::Fpcr => self.fpcr,
            SysReg::Fpsr => self.fpsr,
            SysReg::DczidEl0 => 4,
            SysReg::CtrEl0 => 0x8444_c004,
        }
    }

    pub fn set_sysreg(&mut self, r: SysReg, value: u64) {
        match r {
            SysReg::TpidrEl0 => self.tpidr_el0 = value,
            SysReg::Fpcr => self.fpcr = value,
            SysReg::Fpsr => self.fpsr = value,
            _ => {}
        }
    }

    pub(crate) const OFF_BUDGET: usize = std::mem::offset_of!(Cpu, budget);
    pub(crate) const OFF_MEM: usize = std::mem::offset_of!(Cpu, mem);

    pub(crate) fn refill_budget(&mut self) {
        self.budget = crate::sched::POLL_INTERVAL;
    }

    #[inline(always)]
    pub fn poll(&mut self) -> bool {
        self.budget -= 1;
        if self.budget != 0 {
            return false;
        }
        self.budget = crate::sched::POLL_INTERVAL;
        crate::sched::yield_now(self)
    }

    pub fn halt(&mut self) {
        self.halted = true;
        self.pc = HALT_PC;
    }

    pub fn process(&self) -> &'static Process {
        self.process
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn cas128(p: *mut u8, old: [u64; 2], new: [u64; 2]) -> bool {
    let ok: u8;
    // SAFETY: the caller passes an aligned pointer into guest memory. RBX is
    // reserved by the compiler, so the low new half goes through it and is
    // restored after.
    unsafe {
        std::arch::asm!(
            "xchg {nlo}, rbx",
            "lock cmpxchg16b xmmword ptr [{p}]",
            "sete {ok}",
            "mov rbx, {nlo}",
            p = in(reg) p,
            nlo = inout(reg) new[0] => _,
            ok = out(reg_byte) ok,
            inout("rax") old[0] => _,
            inout("rdx") old[1] => _,
            in("rcx") new[1],
            options(nostack),
        );
    }
    ok != 0
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn cas128(p: *mut u8, old: [u64; 2], new: [u64; 2]) -> bool {
    let p = p as *mut [u64; 2];
    // SAFETY: as above; not atomic on this host.
    unsafe {
        if p.read() != old {
            return false;
        }
        p.write(new);
    }
    true
}
