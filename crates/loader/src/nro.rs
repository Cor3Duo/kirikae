use crate::{align_up, Error, Image};

pub const MAGIC: &[u8; 4] = b"NRO0";
pub const HEADER_SIZE: usize = 0x80;
pub const ASSET_MAGIC: &[u8; 4] = b"ASET";
const ASSET_ROMFS: usize = 0x28;

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
    Segment { offset: u32_at(b, off), size: u32_at(b, off + 4) }
}

pub fn is_nro(file: &[u8]) -> bool {
    file.len() >= HEADER_SIZE && &file[0x10..0x14] == MAGIC
}

pub fn asset_romfs(file: &[u8]) -> Option<(u64, u64)> {
    let base = u32_at(file, 0x18) as usize;
    let assets = file.get(base..)?;
    if assets.get(..4)? != ASSET_MAGIC {
        return None;
    }
    let u64_at = |off: usize| assets.get(off..off + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    let (offset, size) = (u64_at(ASSET_ROMFS)?, u64_at(ASSET_ROMFS + 8)?);
    let start = base as u64 + offset;
    (size > 0 && start.checked_add(size)? <= file.len() as u64).then_some((start, size))
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
        for (off, s) in [(0x20, self.text), (0x28, self.ro), (0x30, self.data), (0x68, self.api_info), (0x70, self.dynstr), (0x78, self.dynsym)] {
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
    let range = |s: Segment| base + s.offset as u64..base + s.offset as u64 + s.size as u64;
    Ok(Image {
        base,
        text: range(h.text),
        ro: range(h.ro),
        data: range(h.data),
        bss: base + bss_start..base + bss_start + h.bss_size as u64,
        entry: base,
        memory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_assets(romfs: (u64, u64), tail: usize) -> Vec<u8> {
        let mut file = vec![0u8; 0x100];
        file[0x10..0x14].copy_from_slice(MAGIC);
        file[0x18..0x1c].copy_from_slice(&0x100u32.to_le_bytes());
        let mut assets = vec![0u8; 0x38];
        assets[..4].copy_from_slice(ASSET_MAGIC);
        assets[ASSET_ROMFS..ASSET_ROMFS + 8].copy_from_slice(&romfs.0.to_le_bytes());
        assets[ASSET_ROMFS + 8..ASSET_ROMFS + 16].copy_from_slice(&romfs.1.to_le_bytes());
        file.extend_from_slice(&assets);
        file.resize(file.len() + tail, 0);
        file
    }

    #[test]
    fn finds_the_asset_romfs() {
        assert_eq!(asset_romfs(&with_assets((0x38, 0x40), 0x40)), Some((0x138, 0x40)));
        assert_eq!(asset_romfs(&with_assets((0x38, 0x80), 0x40)), None);
        assert_eq!(asset_romfs(&with_assets((0x38, 0), 0)), None);
        let mut plain = with_assets((0x38, 0x40), 0x40);
        plain.truncate(0x100);
        assert_eq!(asset_romfs(&plain), None);
    }
}
