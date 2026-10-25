use std::ops::Range;

use crate::Image;

const DT_NULL: u64 = 0;
const DT_HASH: u64 = 4;
const DT_STRTAB: u64 = 5;
const DT_SYMTAB: u64 = 6;
const DT_RELA: u64 = 7;
const DT_RELASZ: u64 = 8;
const DT_PLTRELSZ: u64 = 2;
const DT_JMPREL: u64 = 23;

const R_AARCH64_ABS64: u32 = 257;
const R_AARCH64_GLOB_DAT: u32 = 1025;
const R_AARCH64_JUMP_SLOT: u32 = 1026;
const R_AARCH64_RELATIVE: u32 = 1027;

const STT_FUNC: u8 = 2;
const SYM_SIZE: u64 = 24;
const RELA_SIZE: u64 = 24;

struct View<'a>(&'a [u8]);

impl View<'_> {
    fn u32(&self, off: u64) -> Option<u32> {
        let o = usize::try_from(off).ok()?;
        Some(u32::from_le_bytes(self.0.get(o..o + 4)?.try_into().ok()?))
    }
    fn u64(&self, off: u64) -> Option<u64> {
        let o = usize::try_from(off).ok()?;
        Some(u64::from_le_bytes(self.0.get(o..o + 8)?.try_into().ok()?))
    }
    fn u8(&self, off: u64) -> Option<u8> {
        self.0.get(usize::try_from(off).ok()?).copied()
    }
}

struct Symbol {
    kind: u8,
    defined: bool,
    value: u64,
}

fn symbol(m: &View, symtab: u64, index: u64) -> Option<Symbol> {
    let at = symtab + index * SYM_SIZE;
    Some(Symbol {
        kind: m.u8(at + 4)? & 0xf,
        defined: m.u32(at + 4)? >> 16 != 0,
        value: m.u64(at + 8)?,
    })
}

#[derive(Clone, Copy, Default)]
struct Tables {
    rela: u64,
    relasz: u64,
    jmprel: u64,
    pltrelsz: u64,
    symtab: u64,
    strtab: u64,
    hash: u64,
}

impl Tables {
    fn read(m: &View) -> Option<Tables> {
        let mod0 = u64::from(m.u32(4)?);
        if m.u32(mod0) != Some(u32::from_le_bytes(*b"MOD0")) {
            return None;
        }
        let dynamic = mod0.wrapping_add(m.u32(mod0 + 4)? as i32 as i64 as u64);
        let mut t = Tables::default();
        for i in 0..4096u64 {
            let (Some(tag), Some(val)) = (m.u64(dynamic + i * 16), m.u64(dynamic + i * 16 + 8))
            else {
                break;
            };
            match tag {
                DT_NULL => break,
                DT_RELA => t.rela = val,
                DT_RELASZ => t.relasz = val,
                DT_JMPREL => t.jmprel = val,
                DT_PLTRELSZ => t.pltrelsz = val,
                DT_SYMTAB => t.symtab = val,
                DT_STRTAB => t.strtab = val,
                DT_HASH => t.hash = val,
                _ => {}
            }
        }
        Some(t)
    }

    fn symbol_count(&self, m: &View) -> u64 {
        let n = match (self.hash, self.strtab > self.symtab) {
            (h, _) if h != 0 => m.u32(h + 4).unwrap_or(0) as u64,
            (_, true) => (self.strtab - self.symtab) / SYM_SIZE,
            _ => 0,
        };
        if self.symtab == 0 { 0 } else { n.min(1 << 20) }
    }
}

/// Guest address range of the module's `.eh_frame_hdr`, as recorded in MOD0.
///
/// Both MOD0 fields are signed offsets from MOD0 itself; a zero start means the
/// module has no unwind header. Ranges outside the loaded image are rejected.
pub fn eh_frame_hdr(image: &Image) -> Option<Range<u64>> {
    let m = View(&image.memory);
    let mod0 = u64::from(m.u32(4)?);
    if m.u32(mod0)? != u32::from_le_bytes(*b"MOD0") {
        return None;
    }
    let rel = |field: u64| Some(mod0 as i64 + i64::from(m.u32(mod0 + field)? as i32));
    let (start, end) = (rel(0x10)?, rel(0x14)?);
    if start <= 0 || end <= start {
        return None;
    }
    let start = usize::try_from(start).ok()?;
    let end = usize::try_from(end).ok()?;
    (end <= image.memory.len()).then(|| image.base + start as u64..image.base + end as u64)
}

pub fn function_symbols(image: &Image) -> Vec<(u64, String)> {
    let m = View(&image.memory);
    let Some(t) = Tables::read(&m) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for i in 0..t.symbol_count(&m) {
        let at = t.symtab + i * SYM_SIZE;
        let (Some(s), Some(name_off)) = (symbol(&m, t.symtab, i), m.u32(at)) else {
            continue;
        };
        if s.kind != STT_FUNC || !s.defined {
            continue;
        }
        let start = (t.strtab + name_off as u64) as usize;
        let Some(rest) = image.memory.get(start..) else {
            continue;
        };
        let name = &rest[..rest.iter().position(|&b| b == 0).unwrap_or(rest.len())];
        out.push((
            image.base.wrapping_add(s.value),
            String::from_utf8_lossy(name).into_owned(),
        ));
    }
    out.sort_unstable();
    out.dedup_by_key(|s| s.0);
    out
}

pub fn code_pointers(image: &Image) -> Vec<u64> {
    let m = View(&image.memory);
    let base = image.base;
    let mut out = Vec::new();
    let Some(t) = Tables::read(&m) else {
        return out;
    };
    let Tables {
        rela,
        relasz,
        jmprel,
        pltrelsz,
        symtab,
        ..
    } = t;
    let mut push = |addr: u64| {
        if image.is_code(addr) {
            out.push(addr);
        }
    };

    for (start, size) in [(rela, relasz), (jmprel, pltrelsz)] {
        for k in 0..size / RELA_SIZE {
            let at = start + k * RELA_SIZE;
            let (Some(info), Some(addend)) = (m.u64(at + 8), m.u64(at + 16)) else {
                break;
            };
            match info as u32 {
                R_AARCH64_RELATIVE => push(base.wrapping_add(addend)),
                R_AARCH64_ABS64 | R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT if symtab != 0 => {
                    if let Some(s) = symbol(&m, symtab, info >> 32)
                        && s.defined
                    {
                        push(base.wrapping_add(s.value).wrapping_add(addend));
                    }
                }
                _ => {}
            }
        }
    }

    for i in 0..t.symbol_count(&m) {
        if let Some(s) = symbol(&m, symtab, i)
            && s.kind == STT_FUNC
            && s.defined
        {
            push(base.wrapping_add(s.value));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}
