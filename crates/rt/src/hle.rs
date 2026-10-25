use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::cpu::Cpu;

pub const BASE: u64 = 0x7f_ff00_0000;
pub const SLOTS: u64 = 0x10_0000 / 16;
const RETURN: u64 = BASE + 16 * (SLOTS - 1);

pub type HostFn = Arc<dyn Fn(&mut Cpu) + Send + Sync>;

#[derive(Default)]
pub struct Table {
    slots: RwLock<Vec<(String, HostFn)>>,
    replaced: RwLock<HashMap<u64, HostFn>>,
    any_replaced: std::sync::atomic::AtomicBool,
}

impl Table {
    pub fn add(&self, name: &str, f: HostFn) -> u64 {
        let mut slots = self.slots.write().unwrap();
        assert!((slots.len() as u64) < SLOTS - 1, "too many host functions");
        slots.push((name.to_string(), f));
        BASE + 16 * (slots.len() as u64 - 1)
    }

    pub fn replace(&self, addr: u64, f: HostFn) {
        self.replaced.write().unwrap().insert(addr, f);
        self.any_replaced
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn owns(&self, pc: u64) -> bool {
        (BASE..BASE + 16 * SLOTS).contains(&pc)
            || (self.any_replaced.load(std::sync::atomic::Ordering::Relaxed)
                && self.replaced.read().unwrap().contains_key(&pc))
    }

    fn get(&self, pc: u64) -> Option<HostFn> {
        if (BASE..BASE + 16 * SLOTS).contains(&pc) {
            return self
                .slots
                .read()
                .unwrap()
                .get(((pc - BASE) / 16) as usize)
                .map(|s| s.1.clone());
        }
        self.replaced.read().unwrap().get(&pc).cloned()
    }

    pub fn name(&self, pc: u64) -> Option<String> {
        (BASE..BASE + 16 * SLOTS)
            .contains(&pc)
            .then(|| {
                self.slots
                    .read()
                    .unwrap()
                    .get(((pc - BASE) / 16) as usize)
                    .map(|s| s.0.clone())
            })
            .flatten()
    }
}

pub extern "C" fn trampoline(c: &mut Cpu) {
    let pc = c.pc;
    match c.process().hle.get(pc) {
        Some(f) => f(c),
        None => {
            c.process()
                .warn_once(format!("hle: no host function at {pc:#x}"));
            c.pc = c.x[30];
        }
    }
}

pub fn call_guest(c: &mut Cpu, addr: u64, compiled: bool) {
    let (lr, sp) = (c.x[30], c.sp);
    c.x[30] = RETURN;
    c.pc = addr;
    if compiled {
        match c.process().lookup_compiled(addr) {
            Some(f) => f(c),
            None => c
                .process()
                .warn_once(format!("hle: no recompiled function at {addr:#x} to call")),
        }
    }
    crate::process::run_until(c, RETURN, sp);
    c.x[30] = lr;
}

pub fn guest_str(c: &Cpu, addr: u64) -> String {
    let mem = &c.process().memory;
    let mut out = Vec::new();
    for i in 0..256 {
        match mem.read(addr + i, 1) {
            Some(b) if b[0] != 0 => out.push(b[0]),
            _ => break,
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
