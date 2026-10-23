#[macro_export]
macro_rules! env_var {
    ($name:literal) => {{
        static VALUE: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
        VALUE.get_or_init(|| std::env::var_os($name)).as_deref()
    }};
}

pub mod abi;
pub mod arbiter;
pub mod coverage;
pub mod cpu;
pub mod display;
pub mod fault;
pub mod games;
pub mod gpu;
pub mod hle;
pub mod horizon;
pub mod host1x;
pub mod input;
pub mod interp;
pub mod ipc;
pub mod kernel;
pub mod mem;
pub mod nvn;
pub mod process;
pub mod prof;
pub mod sched;
pub mod services;
pub mod simd;
pub mod threads;
pub mod usage;

pub use cpu::{Cpu, HALT_PC};
pub use kirikae_a64::{ops, SysReg};
pub use horizon::{svc, undefined};
pub use process::{call_indirect, program_main, run, run_executable, run_until, Entry, GuestFn, RunOptions, RunResult};
