use std::sync::Mutex;

use kirikae_loader::Image;
use kirikae_loader::nro::{self, Header};

use crate::cpu::Cpu;
use crate::mem::{PAGE, perm, state};
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

/// Where the module images are placed. The region is above the heap and the
/// stack so it does not collide with the process layout.
const SEARCH_BASE: u64 = 0x6_0000_0000;
const SEARCH_END: u64 = 0x80_0000_0000;
const ALIGN: u64 = 0x20_0000;

/// The registry is stored per process under this key. `ldr:dmnt` reads the same
/// state so the guest sees one list, not one per service session.
pub const STATE: &str = "ldr:ro.modules";

/// The documented registration limits of `nn::ro`: 0x40 NRRs and 0x40 loaded
/// NROs per process.
pub const MAX_NRRS: usize = 0x40;
pub const MAX_MODULES: usize = 0x40;

/// An NRR is at most 64 KiB here: enough for 0x40 page-sized registrations and
/// for the larger hash lists the SDK builds, while keeping a malformed size from
/// asking the guest to read an arbitrary amount of memory.
pub const MAX_NRR_SIZE: u64 = 0x1_0000;

const NRR_HEADER_SIZE: u64 = 0x350;
const NRR_HASH_SIZE: u64 = 0x20;

// Result codes from the ro module (module 22). The SDK reacts to any of them
// with a description in 1..=0x3ff, so the values identify the module while the
// descriptions keep the failure kinds apart.
const RESULT_OUT_OF_ADDRESS_SPACE: u32 = (2 << 9) | 22;
const RESULT_ALREADY_LOADED: u32 = (3 << 9) | 22;
const RESULT_INVALID_NRO: u32 = (4 << 9) | 22;
const RESULT_INVALID_NRR: u32 = (6 << 9) | 22;
const RESULT_TOO_MANY_NRO: u32 = (7 << 9) | 22;
const RESULT_TOO_MANY_NRR: u32 = (8 << 9) | 22;
const RESULT_INVALID_NRR_KIND: u32 = (10 << 9) | 22;
const RESULT_INVALID_ADDRESS: u32 = (1025 << 9) | 22;
const RESULT_INVALID_SIZE: u32 = (1026 << 9) | 22;
const RESULT_NOT_LOADED: u32 = (1028 << 9) | 22;
const RESULT_NOT_REGISTERED: u32 = (1029 << 9) | 22;

/// One module the process may resolve symbols in. The ranges are the mapped
/// text, rodata and data/bss windows; `size` covers the whole image including
/// the zero-filled tail, as `ldr:dmnt` reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleInfo {
    pub name: String,
    pub base: u64,
    pub size: u64,
    pub text: (u64, u64),
    pub ro: (u64, u64),
    pub data: (u64, u64),
    pub module_id: [u8; 32],
    /// Address of the NRR that authorized this module, or 0 for a boot module.
    pub nrr: u64,
    /// True for an NRO this service mapped, false for a boot module.
    pub mapped: bool,
}

impl ModuleInfo {
    fn from_image(
        name: String,
        image: &Image,
        module_id: [u8; 32],
        nrr: u64,
        mapped: bool,
    ) -> ModuleInfo {
        ModuleInfo {
            name,
            base: image.base,
            size: image.end() - image.base,
            text: (image.text.start, image.text.end),
            ro: (image.ro.start, image.ro.end),
            data: (image.data.start, image.data.end.max(image.bss.end)),
            module_id,
            nrr,
            mapped,
        }
    }

    pub fn contains(&self, addr: u64) -> bool {
        (self.base..self.base + self.size).contains(&addr)
    }
}

/// The NRR registered through `RegisterModuleInfo`: the hash list the loader
/// checks before it lets an NRO map. The runtime does not verify hashes (it
/// only sees decrypted dumps), but keeping the table lets the guest unregister
/// by address and makes the limits observable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NrrInfo {
    pub address: u64,
    pub program_id: u64,
    pub kind: u8,
    pub hashes: Vec<[u8; 32]>,
}

#[derive(Default)]
pub struct Registry {
    pub modules: Vec<ModuleInfo>,
    pub nrrs: Vec<NrrInfo>,
}

impl Registry {
    /// Adds a module that was already laid out at boot (`rtld`, `main`,
    /// `subsdk*`, `sdk`). Boot modules have no NRR.
    pub fn add_boot(&mut self, name: &str, image: &Image, module_id: [u8; 32]) -> Result<(), u32> {
        if self.modules.len() >= MAX_MODULES {
            return Err(RESULT_TOO_MANY_NRO);
        }
        if self.modules.iter().any(|m| m.base == image.base) {
            return Err(RESULT_ALREADY_LOADED);
        }
        self.modules.push(ModuleInfo::from_image(
            name.into(),
            image,
            module_id,
            0,
            false,
        ));
        Ok(())
    }

    /// Adds an NRO mapped by `MapManualLoadModuleMemory`.
    pub fn add_nro(&mut self, header: &Header, image: &Image, nrr: u64) -> Result<(), u32> {
        if self.modules.len() >= MAX_MODULES {
            return Err(RESULT_TOO_MANY_NRO);
        }
        if self.modules.iter().any(|m| m.base == image.base) {
            return Err(RESULT_ALREADY_LOADED);
        }
        self.modules.push(ModuleInfo::from_image(
            nro_name(header, image.base),
            image,
            header.module_id,
            nrr,
            true,
        ));
        Ok(())
    }

    pub fn remove(&mut self, base: u64) -> Result<ModuleInfo, u32> {
        let at = self
            .modules
            .iter()
            .position(|m| m.base == base && m.mapped)
            .ok_or(RESULT_NOT_LOADED)?;
        Ok(self.modules.remove(at))
    }

    pub fn module_at(&self, addr: u64) -> Option<&ModuleInfo> {
        self.modules.iter().find(|m| m.contains(addr))
    }

    pub fn module_named(&self, name: &str) -> Option<&ModuleInfo> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// Validates and records an NRR. `kind` is the value the session accepts:
    /// `ldr:ro` only takes `User` (0), `ro:1` only `JitPlugin` (1).
    pub fn register_nrr(&mut self, address: u64, bytes: &[u8], kind: u8) -> Result<(), u32> {
        let size = bytes.len() as u64;
        if !address.is_multiple_of(PAGE)
            || size == 0
            || !size.is_multiple_of(PAGE)
            || !(NRR_HEADER_SIZE..=MAX_NRR_SIZE).contains(&size)
        {
            return Err(RESULT_INVALID_SIZE);
        }
        if &bytes[..4] != b"NRR0" {
            return Err(RESULT_INVALID_NRR);
        }
        if u32_at(bytes, 0x338) as u64 != size {
            return Err(RESULT_INVALID_SIZE);
        }
        if bytes[0x33c] != kind {
            return Err(RESULT_INVALID_NRR_KIND);
        }
        let offset = u32_at(bytes, 0x340) as u64;
        let count = u32_at(bytes, 0x344) as u64;
        let Some(end) = offset
            .checked_add(count.saturating_mul(NRR_HASH_SIZE))
            .filter(|&end| end <= size)
        else {
            return Err(RESULT_INVALID_NRR);
        };
        if offset < NRR_HEADER_SIZE || end > size {
            return Err(RESULT_INVALID_NRR);
        }
        if self.nrrs.len() >= MAX_NRRS {
            return Err(RESULT_TOO_MANY_NRR);
        }
        if self.nrrs.iter().any(|n| n.address == address) {
            return Err(RESULT_ALREADY_LOADED);
        }
        let mut hashes = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let at = offset as usize + i * NRR_HASH_SIZE as usize;
            hashes.push(bytes[at..at + NRR_HASH_SIZE as usize].try_into().unwrap());
        }
        self.nrrs.push(NrrInfo {
            address,
            program_id: u64_at(bytes, 0x330),
            kind,
            hashes,
        });
        Ok(())
    }

    pub fn unregister_nrr(&mut self, address: u64) -> Result<NrrInfo, u32> {
        let at = self
            .nrrs
            .iter()
            .position(|n| n.address == address)
            .ok_or(RESULT_NOT_REGISTERED)?;
        Ok(self.nrrs.remove(at))
    }
}

/// A stable name for an NRO. The NRO header carries a build id, not a name;
/// use its first bytes so two modules are distinguishable, and fall back to the
/// address for a homebrew image whose build id is zero.
fn nro_name(header: &Header, base: u64) -> String {
    if header.module_id.iter().any(|&b| b != 0) {
        let id: String = header.module_id[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!("nro-{id}")
    } else {
        format!("nro-{base:x}")
    }
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

pub struct Ro;

impl Service for Ro {
    fn name(&self) -> String {
        "ldr:ro".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let registry = p.service_state(STATE, || Mutex::new(Registry::default()));
        match req.cmd {
            0 => {
                // MapManualLoadModuleMemory(nro_address, nro_size, bss_address,
                // bss_size). The client keeps the NRO and its zeroed `.bss`
                // buffers in its own memory; the service places both
                // contiguously in the module region and returns the base.
                let (nro, nro_size, bss, bss_size) =
                    (req.u64(8), req.u64(16), req.u64(24), req.u64(32));
                if nro == 0
                    || nro_size == 0
                    || !nro_size.is_multiple_of(PAGE)
                    || !bss_size.is_multiple_of(PAGE)
                    || (bss_size != 0 && (bss == 0 || !bss.is_multiple_of(PAGE)))
                {
                    return Reply::err(RESULT_INVALID_SIZE);
                }
                let Some(bytes) = p.memory.read(nro, nro_size) else {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                };
                let Ok(header) = Header::parse(&bytes) else {
                    return Reply::err(RESULT_INVALID_NRO);
                };
                // The header is the authority for the image and its `.bss`:
                // every size is page aligned and the segments are contiguous
                // (text starts at 0 and data ends at the file size).
                let data_end = header.data.offset as u64 + header.data.size as u64;
                if header.size as u64 != nro_size
                    || header.bss_size as u64 != bss_size
                    || data_end != nro_size
                {
                    return Reply::err(RESULT_INVALID_NRO);
                }
                let total = nro_size + bss_size;
                let Some(at) = free_range(c, total) else {
                    return Reply::err(RESULT_OUT_OF_ADDRESS_SPACE);
                };
                let Ok(image) = nro::load(&bytes, at) else {
                    return Reply::err(RESULT_INVALID_NRO);
                };
                if image.memory.len() as u64 > total {
                    return Reply::err(RESULT_INVALID_NRO);
                }
                // text: RX, ro: R, data + bss: RW. The NRO header is part of
                // the text segment.
                let regions = [
                    (
                        image.text.start,
                        image.text.end,
                        state::CODE_STATIC,
                        perm::RX,
                    ),
                    (image.ro.start, image.ro.end, state::CODE_STATIC, perm::R),
                    (
                        at + header.data.offset as u64,
                        at + total,
                        state::CODE_MUTABLE,
                        perm::RW,
                    ),
                ];
                for (start, end, st, pm) in regions {
                    if end > start && p.memory.map_as(start, end - start, st, pm).is_err() {
                        let _ = p.memory.unmap(at, total);
                        return Reply::err(RESULT_OUT_OF_ADDRESS_SPACE);
                    }
                }
                if !p.memory.write(at, &image.memory) {
                    let _ = p.memory.unmap(at, total);
                    return Reply::err(RESULT_OUT_OF_ADDRESS_SPACE);
                }
                // The `.bss` pages must never expose whatever the caller's
                // buffer held, and the module object lives at the `.bss` base.
                if bss_size != 0 && !p.memory.write(at + nro_size, &vec![0u8; bss_size as usize]) {
                    let _ = p.memory.unmap(at, total);
                    return Reply::err(RESULT_OUT_OF_ADDRESS_SPACE);
                }
                let nrr = registry
                    .lock()
                    .unwrap()
                    .nrrs
                    .last()
                    .map_or(0, |n| n.address);
                if let Err(e) = registry.lock().unwrap().add_nro(&header, &image, nrr) {
                    let _ = p.memory.unmap(at, total);
                    return Reply::err(e);
                }
                forget_lifted_code(p);
                p.warn_once(format!(
                    "ldr:ro: mapped {nro_size:#x} bytes (+{bss_size:#x} bss) at {at:#x}, entry {:#x}; its code runs on the interpreter",
                    image.entry
                ));
                Reply::ok().u64(at)
            }
            1 => {
                let at = req.u64(8);
                let removed = registry.lock().unwrap().remove(at);
                match removed {
                    Ok(module) => {
                        forget_lifted_code(p);
                        let _ = p.memory.unmap(at, module.size);
                        p.warn_once(format!(
                            "ldr:ro: unmapped the {:#x}-byte module at {at:#x}",
                            module.size
                        ));
                        Reply::ok()
                    }
                    Err(e) => Reply::err(e),
                }
            }
            2 | 10 => {
                // RegisterModuleInfo(nrr_address, nrr_size). The 4th word of
                // command 10 is the user process handle, which this runtime
                // ignores: there is one process.
                let (address, size) = (req.u64(8), req.u64(16));
                if address == 0 || size == 0 || !size.is_multiple_of(PAGE) {
                    return Reply::err(RESULT_INVALID_SIZE);
                }
                if !address.is_multiple_of(PAGE) {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                }
                let max = size.min(MAX_NRR_SIZE);
                let Some(bytes) = p.memory.read(address, max) else {
                    return Reply::err(RESULT_INVALID_ADDRESS);
                };
                if bytes.len() as u64 != size {
                    return Reply::err(RESULT_INVALID_SIZE);
                }
                match registry.lock().unwrap().register_nrr(address, &bytes, 0) {
                    Ok(()) => Reply::ok(),
                    Err(e) => Reply::err(e),
                }
            }
            3 => {
                let address = req.u64(8);
                match registry.lock().unwrap().unregister_nrr(address) {
                    Ok(_) => Reply::ok(),
                    Err(e) => Reply::err(e),
                }
            }
            4 => Reply::ok(),
            _ => unimplemented("ldr:ro", c, req),
        }
    }
}

/// Seeds the registry with the modules the loader laid out before the first
/// instruction. Called once by `run_executable`; `ldr:dmnt` and the ro
/// registration commands share this state.
pub fn register_boot_modules(p: &crate::process::Process, modules: &[kirikae_loader::Module]) {
    let registry = p.service_state(STATE, || Mutex::new(Registry::default()));
    let mut registry = registry.lock().unwrap();
    for m in modules {
        let _ = registry.add_boot(&m.name, &m.image, m.module_id);
    }
}

/// Lifted instructions are keyed by guest address. A module unloaded and
/// another one mapped at the same base would otherwise keep executing the old
/// decode, so drop the cache whenever the mapped code changes.
fn forget_lifted_code(p: &crate::process::Process) {
    p.interp_cache.write().unwrap().clear();
    p.interp_cache32.write().unwrap().clear();
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::abi::Entry;
    use crate::process::{RunOptions, run_executable};
    use crate::services::Buf;
    use kirikae_a64::lift;

    const GUEST: u64 = 0x1000_0000;
    const MODULE_BASE: u64 = SEARCH_BASE;

    fn put(file: &mut [u8], at: u32, value: u32) {
        file[at as usize..at as usize + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put64(file: &mut [u8], at: u32, value: u64) {
        file[at as usize..at as usize + 8].copy_from_slice(&value.to_le_bytes());
    }

    /// A synthetic NRO: 0x1000 text, 0x1000 ro (MOD0 + dynamic table),
    /// 0x1000 data (with a RELATIVE relocation) and 0x1000 of `.bss` after it.
    /// `DT_INIT` points at text+0x80. The last 0x800 bytes of data are the
    /// module's `.bss` and are filled with a marker in the file.
    fn image() -> Vec<u8> {
        let mut f = vec![0u8; 0x3000];
        put(&mut f, 0x00, 1);
        put(&mut f, 0x04, 0x1020);
        put(&mut f, 0x08, 0x103c);
        f[0x10..0x14].copy_from_slice(b"NRO0");
        put(&mut f, 0x18, 0x3000);
        put(&mut f, 0x20, 0x0000);
        put(&mut f, 0x24, 0x1000);
        put(&mut f, 0x28, 0x1000);
        put(&mut f, 0x2c, 0x1000);
        put(&mut f, 0x30, 0x2000);
        put(&mut f, 0x34, 0x1000);
        put(&mut f, 0x38, 0x1000);
        put(&mut f, 0x60, 0x2f00);
        put(&mut f, 0x80, 0xd503201f);
        // MOD0 at 0x1020.
        f[0x1020..0x1024].copy_from_slice(b"MOD0");
        put(&mut f, 0x1024, 0x60);
        put(&mut f, 0x1028, 0x17e0);
        put(&mut f, 0x102c, 0x1fe0);
        put(&mut f, 0x1038, 0x17e0);
        // Dynamic table at 0x1080: DT_INIT = 0x80, DT_RELA = 0x2080.
        put64(&mut f, 0x1080, 12);
        put64(&mut f, 0x1088, 0x80);
        put64(&mut f, 0x1090, 7);
        put64(&mut f, 0x1098, 0x2080);
        put64(&mut f, 0x10a0, 8);
        put64(&mut f, 0x10a8, 24);
        // RELA at 0x2080: target 0x2100, R_AARCH64_RELATIVE, addend 0x80.
        put64(&mut f, 0x2080, 0x2100);
        put64(&mut f, 0x2088, 0x403);
        put64(&mut f, 0x2090, 0x80);
        put64(&mut f, 0x2100, 0x80);
        // The module's `.bss` overlaps the tail of the stored data.
        for b in &mut f[0x2800..] {
            *b = 0xaa;
        }
        f
    }

    fn nrr(size: u64, kind: u8, hashes: u32) -> Vec<u8> {
        let mut b = vec![0u8; size as usize];
        b[..4].copy_from_slice(b"NRR0");
        b[0x330..0x338].copy_from_slice(&0x0100_0000_0000_0000u64.to_le_bytes());
        b[0x338..0x33c].copy_from_slice(&(size as u32).to_le_bytes());
        b[0x33c] = kind;
        b[0x340..0x344].copy_from_slice(&0x350u32.to_le_bytes());
        b[0x344..0x348].copy_from_slice(&hashes.to_le_bytes());
        for i in 0..hashes as usize {
            for (j, x) in b[0x350 + i * 32..0x370 + i * 32].iter_mut().enumerate() {
                *x = (i * 32 + j) as u8;
            }
        }
        b
    }

    fn args(src: u64, nro_size: u64, bss: u64, bss_size: u64) -> Vec<u8> {
        let mut data = vec![0u8; 40];
        data[8..16].copy_from_slice(&src.to_le_bytes());
        data[16..24].copy_from_slice(&nro_size.to_le_bytes());
        data[24..32].copy_from_slice(&bss.to_le_bytes());
        data[32..40].copy_from_slice(&bss_size.to_le_bytes());
        data
    }

    fn read_u64(c: &Cpu, at: u64) -> u64 {
        let b = c.process().memory.read(at, 8).unwrap();
        u64::from_le_bytes(b.try_into().unwrap())
    }

    fn read_u32(c: &Cpu, at: u64) -> u32 {
        let b = c.process().memory.read(at, 4).unwrap();
        u32::from_le_bytes(b.try_into().unwrap())
    }

    fn zeros(c: &Cpu, start: u64, len: u64) -> bool {
        c.process()
            .memory
            .read(start, len)
            .is_some_and(|b| b.iter().all(|&x| x == 0))
    }

    fn region(c: &Cpu, at: u64) -> (u32, u32) {
        let r = c.process().memory.query(at);
        (r.state, r.perm)
    }

    #[derive(Default, Debug)]
    struct Seen {
        result: u32,
        base: u64,
        reused_base: bool,
        text: (u32, u32),
        ro: (u32, u32),
        data: (u32, u32),
        bss: (u32, u32),
        entry_word: u32,
        entry_offset: u32,
        mod0_bss_zero: bool,
        bss_zero: bool,
        rela: u64,
        target: u64,
        lifted_after_map: usize,
        unload: u32,
        after_unload: u32,
        lifted_after_unload: usize,
        bad_size: u32,
        bad_layout: u32,
        registered: Option<ModuleInfo>,
    }

    static SEEN: Mutex<Option<Seen>> = Mutex::new(None);

    fn seed_cache(c: &Cpu) {
        c.process().interp_cache.write().unwrap().insert(
            MODULE_BASE + 0x80,
            Arc::new(lift(MODULE_BASE + 0x80, 0xd503201f)),
        );
    }

    extern "C" fn load_and_unload(c: &mut Cpu) {
        let src = 0x1_0000_0000u64;
        let bss = 0x1_1000_0000u64;
        let bytes = image();
        c.process()
            .memory
            .map_as(src, 0x3000, state::NORMAL, perm::RW)
            .unwrap();
        assert!(c.process().memory.write(src, &bytes));
        c.process()
            .memory
            .map_as(bss, 0x1000, state::NORMAL, perm::RW)
            .unwrap();
        assert!(c.process().memory.write(bss, &[0xaa; 0x1000]));
        seed_cache(c);

        let reply = Ro.call(
            c,
            &Request {
                cmd: 0,
                data: args(src, 0x3000, bss, 0x1000),
                ..Request::default()
            },
        );
        let base = u64::from_le_bytes(reply.data[..8].try_into().unwrap());
        let lifted_after_map = c.process().interp_cache.read().unwrap().len();
        let registered = c
            .process()
            .service_state(STATE, || Mutex::new(Registry::default()))
            .lock()
            .unwrap()
            .module_at(base)
            .cloned();

        let unload = Ro
            .call(
                c,
                &Request {
                    cmd: 1,
                    data: args(base, 0, 0, 0),
                    ..Request::default()
                },
            )
            .result;
        let after_unload = c.process().memory.query(base + 0x80).state;
        let lifted_after_unload = c.process().interp_cache.read().unwrap().len();

        // Re-map for the remaining checks and leave the process clean.
        let reply2 = Ro.call(
            c,
            &Request {
                cmd: 0,
                data: args(src, 0x3000, bss, 0x1000),
                ..Request::default()
            },
        );
        let base2 = u64::from_le_bytes(reply2.data[..8].try_into().unwrap());

        let bad_size = Ro
            .call(
                c,
                &Request {
                    cmd: 0,
                    data: args(src, 0x2000, 0, 0),
                    ..Request::default()
                },
            )
            .result;
        let bad_layout = Ro
            .call(
                c,
                &Request {
                    cmd: 0,
                    data: args(src, 0x3000, 0, 0x1001),
                    ..Request::default()
                },
            )
            .result;
        *SEEN.lock().unwrap() = Some(Seen {
            result: reply.result,
            base,
            reused_base: base2 == base,
            text: region(c, base2 + 0x10),
            ro: region(c, base2 + 0x1000),
            data: region(c, base2 + 0x2000),
            bss: region(c, base2 + 0x3500),
            entry_word: read_u32(c, base2 + 0x80),
            entry_offset: nro::entry_offset(&bytes, &Header::parse(&bytes).unwrap()).unwrap(),
            mod0_bss_zero: zeros(c, base2 + 0x2800, 0x800),
            bss_zero: zeros(c, base2 + 0x3000, 0x1000),
            rela: read_u64(c, base2 + 0x2080),
            target: read_u64(c, base2 + 0x2100),
            lifted_after_map,
            unload,
            after_unload,
            lifted_after_unload,
            bad_size,
            bad_layout,
            registered,
        });
        c.halt();
    }

    fn run(f: crate::abi::GuestFn) {
        let entries: &'static [Entry] = Box::leak(
            vec![Entry {
                addr: GUEST,
                func: f,
            }]
            .into_boxed_slice(),
        );
        let exe = kirikae_loader::Executable {
            modules: Vec::new(),
            entry: entries[0].addr,
            arch: kirikae_loader::Arch::A64,
            title_id: None,
            main_stack_size: None,
            system_resource_size: 0,
            romfs: None,
        };
        run_executable(
            &exe,
            entries,
            RunOptions {
                capture_output: true,
            },
        )
        .unwrap();
    }

    #[test]
    fn maps_the_segments_zeroes_the_bss_and_forgets_lifted_code() {
        run(load_and_unload);
        let s = SEEN.lock().unwrap().take().expect("the guest ran");
        assert_eq!(s.result, 0, "the NRO maps");
        assert_eq!(s.base, MODULE_BASE);
        assert!(s.reused_base, "the freed range is reused");
        assert_eq!(s.text, (state::CODE_STATIC, perm::RX));
        assert_eq!(s.ro, (state::CODE_STATIC, perm::R));
        assert_eq!(s.data, (state::CODE_MUTABLE, perm::RW));
        assert_eq!(s.bss, (state::CODE_MUTABLE, perm::RW));
        assert_eq!(s.entry_word, 0xd503201f, "the text is copied verbatim");
        assert_eq!(s.entry_offset, 0x80, "DT_INIT is the module entry");
        assert!(s.mod0_bss_zero, "the module's .bss inside data is cleared");
        assert!(s.bss_zero, "the mapped .bss is cleared");
        assert_eq!(s.rela, 0x2100, "the RELA table reaches the mapping");
        assert_eq!(
            s.target, 0x80,
            "the RELA target keeps its addend for the caller to relocate once"
        );
        assert_eq!(s.lifted_after_map, 0, "mapping forgets lifted code");
        assert_eq!(s.unload, 0);
        assert_eq!(s.after_unload, state::FREE, "unmapping releases the range");
        assert_eq!(s.lifted_after_unload, 0, "unmapping forgets lifted code");
        assert_eq!(s.bad_size, RESULT_INVALID_NRO, "a short image is refused");
        assert_eq!(
            s.bad_layout, RESULT_INVALID_SIZE,
            "an unaligned bss size is refused"
        );
        let m = s.registered.expect("the NRO is in the registry");
        assert_eq!(m.base, MODULE_BASE);
        assert_eq!(m.text, (MODULE_BASE, MODULE_BASE + 0x1000));
        assert_eq!(m.ro, (MODULE_BASE + 0x1000, MODULE_BASE + 0x2000));
        assert_eq!(m.data, (MODULE_BASE + 0x2000, MODULE_BASE + 0x4000));
    }

    #[test]
    fn registry_enforces_limits_and_removes_by_base() {
        let bytes = image();
        let header = Header::parse(&bytes).unwrap();
        let image = nro::load(&bytes, 0x1000_0000).unwrap();
        let mut registry = Registry::default();
        registry.add_nro(&header, &image, 0).unwrap();
        assert!(registry.module_at(0x1000_0010).is_some());
        assert!(registry.module_at(0x1000_0000 + 0x4000).is_none());
        assert_eq!(registry.remove(0x1000_0000).unwrap().base, 0x1000_0000);
        assert_eq!(registry.remove(0x1000_0000), Err(RESULT_NOT_LOADED));

        for i in 0..MAX_MODULES {
            let mut r = Registry::default();
            for j in 0..i {
                let img = nro::load(&bytes, 0x1000_0000 + j as u64 * 0x4000).unwrap();
                r.add_nro(&header, &img, 0).unwrap();
            }
            let img = nro::load(&bytes, 0x1000_0000 + i as u64 * 0x4000).unwrap();
            assert_eq!(r.add_nro(&header, &img, 0), Ok(()));
            let img = nro::load(&bytes, 0x1000_0000 + (i as u64 + 1) * 0x4000).unwrap();
            if i + 1 >= MAX_MODULES {
                assert_eq!(r.add_nro(&header, &img, 0), Err(RESULT_TOO_MANY_NRO));
            }
        }
    }

    #[test]
    fn nrr_registration_validates_shape_kind_and_limits() {
        let mut registry = Registry::default();
        let good = nrr(0x1000, 0, 2);
        registry.register_nrr(0x2000_0000, &good, 0).unwrap();
        assert_eq!(registry.nrrs[0].hashes.len(), 2);
        assert_eq!(registry.nrrs[0].program_id, 0x0100_0000_0000_0000);
        assert_eq!(
            registry.register_nrr(0x2000_0000, &good, 0),
            Err(RESULT_ALREADY_LOADED)
        );
        assert_eq!(
            registry.unregister_nrr(0x2000_0000).unwrap().address,
            0x2000_0000
        );
        assert_eq!(
            registry.unregister_nrr(0x2000_0000),
            Err(RESULT_NOT_REGISTERED)
        );
        assert_eq!(
            registry.register_nrr(0x2000_0000, &good, 1),
            Err(RESULT_INVALID_NRR_KIND)
        );
        assert_eq!(
            registry.register_nrr(0x2000_0000, &good[..0x34f], 0),
            Err(RESULT_INVALID_SIZE)
        );
        assert_eq!(
            registry.register_nrr(0x2000_0001, &good, 0),
            Err(RESULT_INVALID_SIZE)
        );
        let mut bad = good.clone();
        bad[..4].copy_from_slice(b"NRR1");
        assert_eq!(
            registry.register_nrr(0x2000_0000, &bad, 0),
            Err(RESULT_INVALID_NRR)
        );
        let mut bad = good.clone();
        bad[0x340..0x344].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
        assert_eq!(
            registry.register_nrr(0x2000_0000, &bad, 0),
            Err(RESULT_INVALID_NRR)
        );
    }

    #[test]
    fn two_modules_are_looked_up_by_name_and_address() {
        let bytes = image();
        let header = Header::parse(&bytes).unwrap();
        let a = nro::load(&bytes, 0x1000_0000).unwrap();
        let b = nro::load(&bytes, 0x2000_0000).unwrap();
        let mut registry = Registry::default();
        registry.add_nro(&header, &a, 0).unwrap();
        registry.add_nro(&header, &b, 0).unwrap();
        assert_eq!(registry.modules.len(), 2);
        assert_eq!(registry.module_at(0x1000_0010).unwrap().base, 0x1000_0000);
        assert_eq!(registry.module_at(0x2000_0010).unwrap().base, 0x2000_0000);
        assert_eq!(
            registry
                .module_named(&nro_name(&header, 0x2000_0000))
                .unwrap()
                .base,
            0x2000_0000
        );
    }

    #[test]
    fn module_info_commands_register_and_unregister_the_nrr() {
        static CMD: Mutex<Option<(u32, u32, u32, u32)>> = Mutex::new(None);
        extern "C" fn call(c: &mut Cpu) {
            let at = 0x1_3000_0000u64;
            let bytes = nrr(0x1000, 0, 1);
            c.process()
                .memory
                .map_as(at, 0x1000, state::NORMAL, perm::RW)
                .unwrap();
            assert!(c.process().memory.write(at, &bytes));
            let mut data = vec![0u8; 24];
            data[8..16].copy_from_slice(&at.to_le_bytes());
            data[16..24].copy_from_slice(&0x1000u64.to_le_bytes());
            let ok = Ro
                .call(
                    c,
                    &Request {
                        cmd: 2,
                        data: data.clone(),
                        ..Request::default()
                    },
                )
                .result;
            let duplicate = Ro
                .call(
                    c,
                    &Request {
                        cmd: 2,
                        data,
                        ..Request::default()
                    },
                )
                .result;
            let mut data = vec![0u8; 16];
            data[8..16].copy_from_slice(&at.to_le_bytes());
            let unregister = Ro
                .call(
                    c,
                    &Request {
                        cmd: 3,
                        data: data.clone(),
                        ..Request::default()
                    },
                )
                .result;
            let missing = Ro
                .call(
                    c,
                    &Request {
                        cmd: 3,
                        data,
                        ..Request::default()
                    },
                )
                .result;
            *CMD.lock().unwrap() = Some((ok, duplicate, unregister, missing));
            c.halt();
        }
        run(call);
        assert_eq!(
            *CMD.lock().unwrap(),
            Some((0, RESULT_ALREADY_LOADED, 0, RESULT_NOT_REGISTERED))
        );
    }

    #[test]
    fn ldr_dmnt_reports_the_registry_in_the_guest_buffer() {
        static OUT: Mutex<Option<Vec<u8>>> = Mutex::new(None);
        extern "C" fn fill(c: &mut Cpu) {
            let bytes = image();
            let header = Header::parse(&bytes).unwrap();
            let img = nro::load(&bytes, 0x1000_0000).unwrap();
            let registry = c
                .process()
                .service_state(STATE, || Mutex::new(Registry::default()));
            registry
                .lock()
                .unwrap()
                .add_boot("main", &img, header.module_id)
                .unwrap();
            let out = 0x1_2000_0000u64;
            c.process()
                .memory
                .map_as(out, 0x1000, state::NORMAL, perm::RW)
                .unwrap();
            let reply = crate::services::ldr::LdrDmnt.call(
                c,
                &Request {
                    cmd: 2,
                    data: 1u64.to_le_bytes().to_vec(),
                    b: vec![Buf {
                        addr: out,
                        size: 0x1000,
                    }],
                    ..Request::default()
                },
            );
            let count = u32::from_le_bytes(reply.data[..4].try_into().unwrap());
            let mut got = c.process().memory.read(out, 0x30 * count as u64).unwrap();
            got.extend_from_slice(&count.to_le_bytes());
            *OUT.lock().unwrap() = Some(got);
            c.halt();
        }
        run(fill);
        let got = OUT.lock().unwrap().take().unwrap();
        let count = u32::from_le_bytes(got[got.len() - 4..].try_into().unwrap());
        assert_eq!(count, 1);
        assert_eq!(&got[..0x20], &[0u8; 0x20]);
        let base = u64::from_le_bytes(got[0x20..0x28].try_into().unwrap());
        let size = u64::from_le_bytes(got[0x28..0x30].try_into().unwrap());
        assert_eq!(base, 0x1000_0000);
        assert_eq!(size, 0x4000);
    }
}
