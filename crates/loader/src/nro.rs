use crate::{Error, Image, align_up};

pub const MAGIC: &[u8; 4] = b"NRO0";
pub const HEADER_SIZE: usize = 0x80;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Segment {
    pub offset: u32,
    pub size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub mod0_offset: u32,
    pub version: u32,
    pub size: u32,
    pub flags: u32,
    pub text: Segment,
    pub ro: Segment,
    pub data: Segment,
    pub bss_size: u32,
    pub module_id: [u8; 32],
    pub dso_handle_offset: u32,
    pub api_info: Segment,
    pub dynstr: Segment,
    pub dynsym: Segment,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn seg_at(b: &[u8], off: usize) -> Segment {
    Segment {
        offset: u32_at(b, off),
        size: u32_at(b, off + 4),
    }
}

pub fn is_nro(file: &[u8]) -> bool {
    file.len() >= HEADER_SIZE && &file[0x10..0x14] == MAGIC
}

/// The module header ("MOD0") inside the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleHeader {
    /// Offset of MOD0 from the image base (the NRO start structure points here).
    pub offset: u32,
    /// Dynamic table, relative to MOD0.
    pub dynamic: u32,
    /// Start of `.bss`, relative to MOD0. It may fall inside the stored data.
    pub bss_start: u32,
    /// End of `.bss`, relative to MOD0.
    pub bss_end: u32,
}

impl Header {
    pub fn parse(file: &[u8]) -> Result<Header, Error> {
        if file.len() < HEADER_SIZE {
            return Err(Error::TooSmall);
        }
        if &file[0x10..0x14] != MAGIC {
            return Err(Error::BadMagic);
        }
        Ok(Header {
            mod0_offset: u32_at(file, 0x04),
            version: u32_at(file, 0x14),
            size: u32_at(file, 0x18),
            flags: u32_at(file, 0x1c),
            text: seg_at(file, 0x20),
            ro: seg_at(file, 0x28),
            data: seg_at(file, 0x30),
            bss_size: u32_at(file, 0x38),
            module_id: file[0x40..0x60].try_into().unwrap(),
            dso_handle_offset: u32_at(file, 0x60),
            api_info: seg_at(file, 0x68),
            dynstr: seg_at(file, 0x70),
            dynsym: seg_at(file, 0x78),
        })
    }

    pub fn write(&self, out: &mut [u8]) {
        let mut put = |off: usize, v: u32| out[off..off + 4].copy_from_slice(&v.to_le_bytes());
        put(0x04, self.mod0_offset);
        put(0x14, self.version);
        put(0x18, self.size);
        put(0x1c, self.flags);
        for (off, s) in [
            (0x20, self.text),
            (0x28, self.ro),
            (0x30, self.data),
            (0x68, self.api_info),
            (0x70, self.dynstr),
            (0x78, self.dynsym),
        ] {
            put(off, s.offset);
            put(off + 4, s.size);
        }
        put(0x38, self.bss_size);
        put(0x60, self.dso_handle_offset);
        out[0x10..0x14].copy_from_slice(MAGIC);
        out[0x40..0x60].copy_from_slice(&self.module_id);
    }
}

pub fn load(file: &[u8], base: u64) -> Result<Image, Error> {
    let h = Header::parse(file)?;
    if h.text.offset != 0 {
        return Err(Error::Malformed("text segment does not start at 0"));
    }
    if !(h.text.offset <= h.ro.offset && h.ro.offset <= h.data.offset) {
        return Err(Error::Malformed("segments out of order"));
    }
    if h.text.offset as u64 + h.text.size as u64 != h.ro.offset as u64
        || h.ro.offset as u64 + h.ro.size as u64 != h.data.offset as u64
    {
        return Err(Error::Malformed("segments are not contiguous"));
    }
    let data_end = h.data.offset as u64 + h.data.size as u64;
    if data_end > file.len() as u64 {
        return Err(Error::Malformed("segments run past the end of the file"));
    }
    let bss_start = data_end;
    let total = align_up(data_end + h.bss_size as u64, 0x1000);
    let mut memory = vec![0u8; total as usize];
    for s in [h.text, h.ro, h.data] {
        let (o, n) = (s.offset as usize, s.size as usize);
        memory[o..o + n].copy_from_slice(&file[o..o + n]);
    }
    // The linker puts the runtime-generated module object at the start of the
    // module's `.bss`, which may reach back into the last page of the stored
    // data. Those bytes are zero-initialised at run time even though the file
    // carries them, so clear the range the module header declares.
    if let Ok(m) = ModuleHeader::parse(file, &h) {
        let bss = m.bss_range();
        if bss.start < bss.end && bss.end <= memory.len() {
            memory[bss].fill(0);
        }
    }
    let range = |s: Segment| base + s.offset as u64..base + s.offset as u64 + s.size as u64;
    let entry = match entry_offset(file, &h) {
        Some(off) if off != 0 && off as u64 + 4 <= total && off.is_multiple_of(4) => {
            base + off as u64
        }
        _ => base,
    };
    Ok(Image {
        base,
        text: range(h.text),
        ro: range(h.ro),
        data: range(h.data),
        bss: base + bss_start..base + bss_start + h.bss_size as u64,
        entry,
        memory,
    })
}

impl ModuleHeader {
    pub fn parse(file: &[u8], h: &Header) -> Result<ModuleHeader, Error> {
        let bad = || Error::Malformed("missing MOD0 module header");
        let off = usize::try_from(h.mod0_offset).map_err(|_| bad())?;
        let m = file.get(off..off + 0x1c).ok_or_else(bad)?;
        if &m[..4] != b"MOD0" {
            return Err(bad());
        }
        Ok(ModuleHeader {
            offset: h.mod0_offset,
            dynamic: u32_at(m, 4),
            bss_start: u32_at(m, 8),
            bss_end: u32_at(m, 12),
        })
    }

    /// `.bss` as an offset range from the image base, relative to MOD0.
    pub fn bss_range(&self) -> std::ops::Range<usize> {
        let base = self.offset as usize;
        base + self.bss_start as usize..base + self.bss_end as usize
    }
}

const DT_NULL: u64 = 0;
const DT_INIT: u64 = 12;

/// The module's initialisation entry point, relative to the image base.
///
/// Official NROs are DLL-like modules: `nn::ro::LoadModule` builds the module
/// object and calls its `DT_INIT` function. Homebrew NROs have no dynamic
/// table and start at the image base, where the loader placed their branch to
/// the payload.
pub fn entry_offset(file: &[u8], h: &Header) -> Option<u32> {
    let m = ModuleHeader::parse(file, h).ok()?;
    if m.dynamic == 0 {
        return None;
    }
    let at = usize::try_from(m.offset).ok()? + usize::try_from(m.dynamic).ok()?;
    for i in 0..4096 {
        let e = file.get(at + i * 16..at + i * 16 + 16)?;
        let tag = u64::from_le_bytes(e[..8].try_into().ok()?);
        let val = u64::from_le_bytes(e[8..].try_into().ok()?);
        match tag {
            DT_NULL => break,
            DT_INIT => return u32::try_from(val).ok(),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x0800_0000;

    fn put(f: &mut [u8], at: u32, v: u32) {
        f[at as usize..at as usize + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn put64(f: &mut [u8], at: u32, v: u64) {
        f[at as usize..at as usize + 8].copy_from_slice(&v.to_le_bytes());
    }

    /// text 0x0000..0x1000, ro 0x1000..0x2000 (MOD0 at 0x1020, dynamic table at
    /// 0x1080), data 0x2000..0x3000 (RELA at 0x2080, target 0x2100) and 0x1000
    /// of `.bss` after the image. MOD0's `.bss` starts at 0x2800, inside data.
    fn image() -> Vec<u8> {
        let mut f = vec![0u8; 0x3000];
        put(&mut f, 0x00, 1);
        put(&mut f, 0x04, 0x1020);
        put(&mut f, 0x08, 0x103c);
        f[0x10..0x14].copy_from_slice(b"NRO0");
        put(&mut f, 0x18, 0x3000);
        put(&mut f, 0x24, 0x1000);
        put(&mut f, 0x28, 0x1000);
        put(&mut f, 0x2c, 0x1000);
        put(&mut f, 0x30, 0x2000);
        put(&mut f, 0x34, 0x1000);
        put(&mut f, 0x38, 0x1000);
        put(&mut f, 0x60, 0x2f00);
        put(&mut f, 0x80, 0xd503201f);
        f[0x1020..0x1024].copy_from_slice(b"MOD0");
        put(&mut f, 0x1024, 0x60);
        put(&mut f, 0x1028, 0x17e0);
        put(&mut f, 0x102c, 0x1fe0);
        put(&mut f, 0x1038, 0x17e0);
        put64(&mut f, 0x1080, 12);
        put64(&mut f, 0x1088, 0x80);
        put64(&mut f, 0x1090, 7);
        put64(&mut f, 0x1098, 0x2080);
        put64(&mut f, 0x10a0, 8);
        put64(&mut f, 0x10a8, 24);
        put64(&mut f, 0x2080, 0x2100);
        put64(&mut f, 0x2088, 0x403);
        put64(&mut f, 0x2090, 0x80);
        put64(&mut f, 0x2100, 0x80);
        for b in &mut f[0x2800..] {
            *b = 0xaa;
        }
        f
    }

    #[test]
    fn maps_segments_zeroes_bss_and_resolves_the_entry() {
        let f = image();
        let h = Header::parse(&f).unwrap();
        let m = ModuleHeader::parse(&f, &h).unwrap();
        assert_eq!(m.bss_range(), 0x2800..0x3000);
        assert_eq!(entry_offset(&f, &h), Some(0x80));

        let i = load(&f, BASE).unwrap();
        assert_eq!(i.text, BASE..BASE + 0x1000);
        assert_eq!(i.ro, BASE + 0x1000..BASE + 0x2000);
        assert_eq!(i.data, BASE + 0x2000..BASE + 0x3000);
        assert_eq!(i.bss, BASE + 0x3000..BASE + 0x4000);
        assert_eq!(i.entry, BASE + 0x80);
        assert_eq!(i.memory[0x80..0x84], 0xd503201f_u32.to_le_bytes());
        assert!(i.memory[0x2800..0x3000].iter().all(|&b| b == 0));
        assert!(i.memory[0x3000..0x4000].iter().all(|&b| b == 0));
        // The relocation stays in the image for the module loader to apply.
        assert_eq!(
            u64::from_le_bytes(i.memory[0x2080..0x2088].try_into().unwrap()),
            0x2100
        );
        assert_eq!(
            u64::from_le_bytes(i.memory[0x2088..0x2090].try_into().unwrap()),
            0x403
        );
        assert_eq!(
            u64::from_le_bytes(i.memory[0x2100..0x2108].try_into().unwrap()),
            0x80
        );
    }

    #[test]
    fn a_module_without_a_dynamic_table_starts_at_the_base() {
        let mut f = image();
        // MOD0 + 4 is the dynamic table offset.
        put(&mut f, 0x1024, 0);
        let h = Header::parse(&f).unwrap();
        assert_eq!(entry_offset(&f, &h), None);
        assert_eq!(load(&f, BASE).unwrap().entry, BASE);
    }

    #[test]
    fn rejects_segments_that_are_not_contiguous() {
        let mut f = image();
        put(&mut f, 0x2c, 0x2000); // ro.size now overlaps data
        let err = load(&f, BASE).unwrap_err();
        assert!(matches!(err, Error::Malformed(_)));
    }

    #[test]
    fn rejects_a_truncated_image() {
        let f = image();
        assert!(matches!(
            load(&f[..0x2800], BASE).unwrap_err(),
            Error::Malformed(_)
        ));
    }
}
