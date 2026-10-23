use std::io::{Read, Seek, SeekFrom};

use crate::Error;

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

pub fn parse<R: Read + Seek>(r: &mut R, base: u64) -> Result<Vec<Entry>, Error> {
    r.seek(SeekFrom::Start(base))?;
    let mut head = [0u8; 16];
    r.read_exact(&mut head)?;
    if &head[..4] != b"PFS0" {
        return Err(Error::BadMagic);
    }
    let count = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
    let strings_size = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
    if count > 0x10000 || strings_size > 0x100_0000 {
        return Err(Error::Malformed("PFS0 header is implausible"));
    }
    let mut table = vec![0u8; count * 0x18];
    r.read_exact(&mut table)?;
    let mut strings = vec![0u8; strings_size];
    r.read_exact(&mut strings)?;
    let data_start = base + 16 + table.len() as u64 + strings_size as u64;

    table
        .as_chunks::<0x18>()
        .0
        .iter()
        .map(|e| {
            let offset = u64::from_le_bytes(e[0..8].try_into().unwrap());
            let size = u64::from_le_bytes(e[8..16].try_into().unwrap());
            let name_at = u32::from_le_bytes(e[16..20].try_into().unwrap()) as usize;
            let tail = strings.get(name_at..).ok_or(Error::Malformed("PFS0 name offset"))?;
            let len = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
            let name = String::from_utf8_lossy(&tail[..len]).into_owned();
            Ok(Entry { name, offset: data_start + offset, size })
        })
        .collect()
}

pub fn read_entry<R: Read + Seek>(r: &mut R, e: &Entry) -> Result<Vec<u8>, Error> {
    if e.size > 1 << 32 {
        return Err(Error::Malformed("PFS0 entry too large to read into memory"));
    }
    r.seek(SeekFrom::Start(e.offset))?;
    let mut buf = vec![0u8; e.size as usize];
    r.read_exact(&mut buf)?;
    Ok(buf)
}
