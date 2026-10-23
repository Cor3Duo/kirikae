use crate::{align_up, lz4, Error, Image};

pub const MAGIC: &[u8; 4] = b"NSO0";

#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub file_offset: u32,
    pub mem_offset: u32,
    pub size: u32,
    pub compressed_size: u32,
    pub compressed: bool,
}

#[derive(Clone, Debug)]
pub struct Header {
    pub flags: u32,
    pub segments: [Segment; 3],
    pub bss_size: u32,
    pub module_id: [u8; 32],
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

pub fn is_nso(file: &[u8]) -> bool {
    file.len() >= 0x100 && &file[..4] == MAGIC
}

impl Header {
    pub fn parse(file: &[u8]) -> Result<Header, Error> {
        if file.len() < 0x100 {
            return Err(Error::TooSmall);
        }
        if &file[..4] != MAGIC {
            return Err(Error::BadMagic);
        }
        let flags = u32_at(file, 0x0c);
        let seg = |i: usize| Segment {
            file_offset: u32_at(file, 0x10 + i * 0x10),
            mem_offset: u32_at(file, 0x14 + i * 0x10),
            size: u32_at(file, 0x18 + i * 0x10),
            compressed_size: u32_at(file, 0x60 + i * 4),
            compressed: flags & (1 << i) != 0,
        };
        Ok(Header {
            flags,
            segments: [seg(0), seg(1), seg(2)],
            bss_size: u32_at(file, 0x3c),
            module_id: file[0x40..0x60].try_into().unwrap(),
        })
    }
}

pub fn load(file: &[u8], base: u64) -> Result<Image, Error> {
    let h = Header::parse(file)?;
    let [text, ro, data] = h.segments;
    let data_end = data.mem_offset as u64 + data.size as u64;
    let total = align_up(data_end + h.bss_size as u64, 0x1000);
    let mut memory = vec![0u8; total as usize];
    for s in h.segments {
        let stored = if s.compressed { s.compressed_size } else { s.size } as usize;
        let start = s.file_offset as usize;
        let raw = file.get(start..start + stored).ok_or(Error::Malformed("segment runs past the end of the file"))?;
        let dst = memory
            .get_mut(s.mem_offset as usize..s.mem_offset as usize + s.size as usize)
            .ok_or(Error::Malformed("segment outside the module"))?;
        if s.compressed {
            dst.copy_from_slice(&lz4::decompress(raw, s.size as usize)?);
        } else {
            dst.copy_from_slice(raw);
        }
    }
    let range = |s: Segment| base + s.mem_offset as u64..base + s.mem_offset as u64 + s.size as u64;
    Ok(Image {
        base,
        text: range(text),
        ro: range(ro),
        data: range(data),
        bss: base + data_end..base + data_end + h.bss_size as u64,
        entry: base + text.mem_offset as u64,
        memory,
    })
}
