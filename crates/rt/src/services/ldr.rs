//! `ldr:dmnt` (`nn::ldr::detail::IDebugMonitorInterface`).
//!
//! The only command this runtime implements is `GetProcessModuleInfo` (2), the
//! documented way a debugger — and `nn::ro`'s own initialisation path — asks
//! which modules the process has. The service reads the same registry `ldr:ro`
//! fills, so `RegisterModuleInfo`, `MapManualLoadModuleMemory` and the modules
//! laid out at boot all appear in one list.

use std::sync::Mutex;

use crate::cpu::Cpu;
use crate::services::ro::{Registry, STATE};
use crate::services::{Reply, Request, Service};

/// `ModuleInfo` in `ldr:dmnt`: ModuleId (0x20), Address (0x8), Size (0x8).
pub const MODULE_INFO_SIZE: u64 = 0x30;

/// `ldr` module (9), `InvalidAddress` (81) and `InvalidSize` (82).
const RESULT_INVALID_ADDRESS: u32 = (81 << 9) | 9;
const RESULT_INVALID_SIZE: u32 = (82 << 9) | 9;

pub struct LdrDmnt;

impl Service for LdrDmnt {
    fn name(&self) -> String {
        "ldr:dmnt".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            2 => {
                let Some(out) = req.out_buf(0) else {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                };
                if out.size < MODULE_INFO_SIZE {
                    return Reply::err(RESULT_INVALID_SIZE);
                }
                let registry = c
                    .process()
                    .service_state(STATE, || Mutex::new(Registry::default()));
                let registry = registry.lock().unwrap();
                let max = (out.size / MODULE_INFO_SIZE) as usize;
                let mut bytes = Vec::with_capacity(registry.modules.len().min(max) * 0x30);
                let mut count = 0u32;
                for m in registry.modules.iter().take(max) {
                    bytes.extend_from_slice(&m.module_id);
                    bytes.extend_from_slice(&m.base.to_le_bytes());
                    bytes.extend_from_slice(&m.size.to_le_bytes());
                    count += 1;
                }
                if !c.process().memory.write(out.addr, &bytes) {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                }
                Reply::ok().u32(count)
            }
            _ => Reply::err(crate::services::result::CMIF_UNKNOWN_COMMAND),
        }
    }
}
