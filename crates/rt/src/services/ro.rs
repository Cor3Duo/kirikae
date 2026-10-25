use std::collections::HashMap;
use std::sync::Mutex;

use crate::cpu::Cpu;
use crate::mem::{perm, state};
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const SEARCH_BASE: u64 = 0x6_0000_0000;
const SEARCH_END: u64 = 0x80_0000_0000;
const ALIGN: u64 = 0x20_0000;

const RESULT_INVALID_ADDRESS: u32 = (0x1a << 9) | 0x16;
const RESULT_OUT_OF_ADDRESS_SPACE: u32 = (0x16 << 9) | 0x16;

#[derive(Default)]
struct Modules(Mutex<HashMap<u64, u64>>);

pub struct Ro;

impl Service for Ro {
    fn name(&self) -> String {
        "ldr:ro".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let modules = p.service_state("ldr:ro.modules", Modules::default);
        match req.cmd {
            0 => {
                let (nro, nro_size, bss_size) = (req.u64(8), req.u64(16), req.u64(32));
                let size = nro_size + bss_size;
                if nro_size == 0 || nro_size & 0xfff != 0 || bss_size & 0xfff != 0 {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                }
                let Some(image) = p.memory.read(nro, nro_size) else {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                };
                let word =
                    |o: usize| u32::from_le_bytes(image[o..o + 4].try_into().unwrap()) as u64;
                if &image[0x10..0x14] != b"NRO0" {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                }
                let (ro_offset, data_offset) = (word(0x28), word(0x30));
                if !(ro_offset <= data_offset && data_offset <= nro_size) {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                }
                let Some(at) = free_range(c, size) else {
                    return Reply::err(RESULT_OUT_OF_ADDRESS_SPACE);
                };
                let segments = [
                    (0, ro_offset, state::CODE_STATIC, perm::RX),
                    (ro_offset, data_offset, state::CODE_STATIC, perm::R),
                    (data_offset, size, state::CODE_MUTABLE, perm::RW),
                ];
                for (start, end, st, pm) in segments {
                    if end > start && p.memory.map_as(at + start, end - start, st, pm).is_err() {
                        let _ = p.memory.unmap(at, size);
                        return Reply::err(RESULT_OUT_OF_ADDRESS_SPACE);
                    }
                }
                if !p.memory.write(at, &image) {
                    let _ = p.memory.unmap(at, size);
                    return Reply::err(RESULT_OUT_OF_ADDRESS_SPACE);
                }
                modules.0.lock().unwrap().insert(at, size);
                p.warn_once(format!("ldr:ro: NRO of {nro_size:#x} bytes (+{bss_size:#x} bss) mapped at {at:#x}; its code runs on the interpreter"));
                Reply::ok().u64(at)
            }
            1 => {
                let at = req.u64(8);
                match modules.0.lock().unwrap().remove(&at) {
                    Some(size) => {
                        let _ = p.memory.unmap(at, size);
                        Reply::ok()
                    }
                    None => Reply::err(RESULT_INVALID_ADDRESS),
                }
            }
            2..=4 | 10 => Reply::ok(),
            _ => unimplemented("ldr:ro", c, req),
        }
    }
}

fn free_range(c: &Cpu, size: u64) -> Option<u64> {
    let mem = &c.process().memory;
    let mut at = SEARCH_BASE;
    while at + size <= SEARCH_END {
        let r = mem.query(at);
        if r.state != state::FREE {
            at = r.range.end.next_multiple_of(ALIGN);
            continue;
        }
        if r.range.end >= at + size {
            return Some(at);
        }
        at = r.range.end.next_multiple_of(ALIGN);
    }
    None
}
