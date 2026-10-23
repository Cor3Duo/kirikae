use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cpu::Cpu;
use crate::mem::ADDR_MASK;

thread_local! {
    static CPU: Cell<*mut Cpu> = const { Cell::new(std::ptr::null_mut()) };
}
static BASE: AtomicUsize = AtomicUsize::new(0);

pub fn install(cpu: &mut Cpu) {
    BASE.store(cpu.process().memory.base() as usize, Ordering::Relaxed);
    CPU.with(|c| c.set(cpu as *mut Cpu));
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(sys::install);
}

pub fn current_cpu() -> Option<*const Cpu> {
    let c = CPU.with(|c| c.get());
    (!c.is_null()).then_some(c as *const Cpu)
}

fn report(host_addr: usize) {
    let base = BASE.load(Ordering::Relaxed);
    if base == 0 || host_addr < base || host_addr as u64 - base as u64 > ADDR_MASK + 0x10000 {
        return;
    }
    let guest = (host_addr - base) as u64;
    let cpu = CPU.with(|c| c.get());
    // SAFETY: the guest thread is stopped inside this fault; only fields are read.
    let msg = match unsafe { cpu.as_ref() } {
        Some(c) => {
            let p = c.process();
            let mut msg = format!(
                "kirikae: guest access to unmapped address {guest:#x} on thread {} near pc={} (lr={}, sp={:#x})\n",
                c.thread.id,
                p.describe(c.pc),
                p.describe(c.x[30]),
                c.sp
            );
            let mut fp = c.x[29];
            for _ in 0..16 {
                let Some(rec) = p.memory.read(fp, 16) else { break };
                let next = u64::from_le_bytes(rec[..8].try_into().unwrap());
                let lr = u64::from_le_bytes(rec[8..].try_into().unwrap());
                if lr == 0 {
                    break;
                }
                msg += &format!("    called from {}\n", p.describe(lr));
                if next <= fp {
                    break;
                }
                fp = next;
            }
            msg
        }
        None => format!("kirikae: guest access to unmapped address {guest:#x}\n"),
    };
    use std::io::Write;
    let _ = std::io::stderr().write_all(msg.as_bytes());
    std::process::exit(4);
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;

    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut ExceptionRecord,
        address: *mut c_void,
        parameters: u32,
        information: [usize; 15],
    }

    #[repr(C)]
    struct ExceptionPointers {
        record: *mut ExceptionRecord,
        context: *mut c_void,
    }

    const ACCESS_VIOLATION: u32 = 0xC000_0005;
    const CONTINUE_SEARCH: i32 = 0;

    unsafe extern "system" {
        fn AddVectoredExceptionHandler(first: u32, handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32) -> *mut c_void;
    }

    unsafe extern "system" fn handler(info: *mut ExceptionPointers) -> i32 {
        // SAFETY: Windows passes valid exception pointers to the handler.
        let rec = unsafe { &*(*info).record };
        if rec.code == ACCESS_VIOLATION && rec.parameters >= 2 {
            super::report(rec.information[1]);
        }
        CONTINUE_SEARCH
    }

    pub fn install() {
        unsafe { AddVectoredExceptionHandler(1, handler) };
    }
}

#[cfg(unix)]
mod sys {
    extern "C" fn handler(_sig: libc::c_int, info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
        // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
        let addr = unsafe { (*info).si_addr() } as usize;
        super::report(addr);
        unsafe {
            libc::signal(libc::SIGSEGV, libc::SIG_DFL);
            libc::signal(libc::SIGBUS, libc::SIG_DFL);
        }
    }

    pub fn install() {
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO;
            libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut());
            libc::sigaction(libc::SIGBUS, &sa, std::ptr::null_mut());
        }
    }
}
