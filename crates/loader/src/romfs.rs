use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use crate::{lz4, Error};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RomFs {
    pub path: PathBuf,
    pub offset: u64,
    pub size: u64,
    pub compression: Option<(u64, u64)>,
}

const NODE: usize = 0x4000;
const ENTRY: usize = 0x18;
const CACHE_BLOCKS: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Entry {
    virt: u64,
    phys: u64,
    kind: u8,
    psize: u32,
}

pub struct Reader {
    file: File,
    base: u64,
    size: u64,
    entries: Vec<Entry>,
    cache: VecDeque<(usize, Vec<u8>)>,
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn parse_table(t: &[u8]) -> Result<(Vec<Entry>, u64), Error> {
    let bad = Error::Malformed("bad compression table");
    if t.len() < NODE {
        return Err(bad);
    }
    let sets = u32_at(t, 4) as usize;
    let end = u64_at(t, 8);
    if sets > (NODE - 16) / 8 {
        return Err(Error::Malformed("compression tables with an L2 level are not supported yet"));
    }
    let mut entries = Vec::new();
    for k in 0..sets {
        let node = t.get(NODE * (1 + k)..NODE * (2 + k)).ok_or(Error::Malformed("truncated compression table"))?;
        let count = u32_at(node, 4) as usize;
        if 16 + count * ENTRY > NODE {
            return Err(bad);
        }
        for i in 0..count {
            let e = &node[16 + i * ENTRY..16 + (i + 1) * ENTRY];
            entries.push(Entry { virt: u64_at(e, 0), phys: u64_at(e, 8), kind: e[16], psize: u32_at(e, 20) });
        }
    }
    if !entries.windows(2).all(|w| w[0].virt < w[1].virt) {
        return Err(bad);
    }
    Ok((entries, end))
}

impl Reader {
    pub fn open(r: &RomFs) -> Result<Reader, Error> {
        let mut file = File::open(&r.path)?;
        let (entries, size) = match r.compression {
            Some((offset, len)) => {
                file.seek(SeekFrom::Start(offset))?;
                let mut t = vec![0u8; len as usize];
                file.read_exact(&mut t)?;
                parse_table(&t)?
            }
            None => (Vec::new(), r.size),
        };
        Ok(Reader { file, base: r.offset, size, entries, cache: VecDeque::new() })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    fn block_end(&self, i: usize) -> u64 {
        self.entries.get(i + 1).map_or(self.size, |e| e.virt)
    }

    fn block(&mut self, i: usize) -> Result<&[u8], Error> {
        if let Some(pos) = self.cache.iter().position(|(k, _)| *k == i) {
            let hit = self.cache.remove(pos).unwrap();
            self.cache.push_front(hit);
            return Ok(&self.cache[0].1);
        }
        let e = self.entries[i];
        let vsize = (self.block_end(i) - e.virt) as usize;
        let data = match e.kind {
            0 => {
                let mut b = vec![0u8; vsize];
                self.file.seek(SeekFrom::Start(self.base + e.phys))?;
                self.file.read_exact(&mut b)?;
                b
            }
            1 => vec![0u8; vsize],
            3 => {
                let mut raw = vec![0u8; e.psize as usize];
                self.file.seek(SeekFrom::Start(self.base + e.phys))?;
                self.file.read_exact(&mut raw)?;
                lz4::decompress(&raw, vsize)?
            }
            _ => return Err(Error::Malformed("unknown compression type")),
        };
        if self.cache.len() == CACHE_BLOCKS {
            self.cache.pop_back();
        }
        self.cache.push_front((i, data));
        Ok(&self.cache[0].1)
    }

    pub fn read_at(&mut self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        if offset >= self.size {
            return Ok(0);
        }
        let len = out.len().min((self.size - offset) as usize);
        if self.entries.is_empty() {
            self.file.seek(SeekFrom::Start(self.base + offset))?;
            self.file.read_exact(&mut out[..len])?;
            return Ok(len);
        }
        let mut done = 0;
        while done < len {
            let at = offset + done as u64;
            let i = self.entries.partition_point(|e| e.virt <= at) - 1;
            let start = self.entries[i].virt;
            let block = self.block(i)?;
            let skip = (at - start) as usize;
            let n = (block.len() - skip).min(len - done);
            out[done..done + n].copy_from_slice(&block[skip..skip + n]);
            done += n;
        }
        Ok(len)
    }
}

#[derive(Clone, Debug)]
pub struct FileEntry {
    pub path: String,
    pub offset: u64,
    pub size: u64,
}

pub fn list(r: &mut Reader, limit: usize) -> Result<Vec<FileEntry>, Error> {
    let mut header = [0u8; 0x50];
    r.read_at(0, &mut header)?;
    let field = |k: usize| u64_at(&header, 8 * k);
    let (dir_off, dir_len, file_off, file_len, data_off) = (field(3), field(4), field(7), field(8), field(9));
    let mut dirs = vec![0u8; dir_len as usize];
    r.read_at(dir_off, &mut dirs)?;
    let mut files = vec![0u8; file_len as usize];
    r.read_at(file_off, &mut files)?;
    let name = |t: &[u8], at: usize, len: usize| String::from_utf8_lossy(t.get(at..at + len).unwrap_or(&[])).into_owned();
    let mut out = Vec::new();
    let mut stack = vec![(0u32, String::new())];
    while let Some((d, prefix)) = stack.pop() {
        let d = d as usize;
        if d + 0x18 > dirs.len() {
            continue;
        }
        let mut f = u32_at(&dirs, d + 12);
        while f != u32::MAX && (f as usize) + 0x20 <= files.len() && out.len() < limit {
            let e = f as usize;
            let n = u32_at(&files, e + 0x1c) as usize;
            out.push(FileEntry { path: format!("{prefix}/{}", name(&files, e + 0x20, n)), offset: data_off + u64_at(&files, e + 8), size: u64_at(&files, e + 16) });
            f = u32_at(&files, e + 4);
        }
        let mut c = u32_at(&dirs, d + 8);
        while c != u32::MAX && (c as usize) + 0x18 <= dirs.len() {
            let e = c as usize;
            let n = u32_at(&dirs, e + 0x14) as usize;
            stack.push((c, format!("{prefix}/{}", name(&dirs, e + 0x18, n))));
            c = u32_at(&dirs, e + 4);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_across_stored_zero_and_lz4_blocks() {
        let dir = std::env::temp_dir().join(format!("kirikae-romfs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("image.bin");
        let mut phys = b"stored!!".to_vec();
        phys.push(0x80);
        phys.extend_from_slice(b"lz4data!");
        std::fs::write(&path, &phys).unwrap();

        let mut table = vec![0u8; NODE * 2];
        table[4..8].copy_from_slice(&1u32.to_le_bytes());
        table[8..16].copy_from_slice(&24u64.to_le_bytes());
        let set = &mut table[NODE..];
        set[4..8].copy_from_slice(&3u32.to_le_bytes());
        let entries: [(u64, u64, u8, u32); 3] = [(0, 0, 0, 0), (8, 0, 1, 0), (16, 8, 3, 9)];
        for (i, (v, p, k, s)) in entries.iter().enumerate() {
            let e = &mut set[16 + i * ENTRY..16 + (i + 1) * ENTRY];
            e[0..8].copy_from_slice(&v.to_le_bytes());
            e[8..16].copy_from_slice(&p.to_le_bytes());
            e[16] = *k;
            e[20..24].copy_from_slice(&s.to_le_bytes());
        }
        let (entries, size) = parse_table(&table).unwrap();
        let mut r = Reader { file: File::open(&path).unwrap(), base: 0, size, entries, cache: VecDeque::new() };
        let mut out = [0u8; 24];
        assert_eq!(r.read_at(0, &mut out).unwrap(), 24);
        assert_eq!(&out, b"stored!!\0\0\0\0\0\0\0\0lz4data!");
        let mut mid = [0u8; 6];
        r.read_at(5, &mut mid).unwrap();
        assert_eq!(&mid, b"d!!\0\0\0");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
