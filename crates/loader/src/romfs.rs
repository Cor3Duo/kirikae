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

const EMPTY: u32 = u32::MAX;
const HEADER: u64 = 0x50;
const DATA_START: u64 = 0x200;

pub fn hash(parent: u32, name: &[u8]) -> u32 {
    let mut h = parent ^ 123456789;
    for &b in name {
        h = h.rotate_right(5) ^ b as u32;
    }
    h
}

fn table_size(entries: usize) -> usize {
    if entries < 3 {
        return 3;
    }
    if entries < 19 {
        return entries | 1;
    }
    let mut n = entries;
    while [2, 3, 5, 7, 11, 13, 17].iter().any(|p| n.is_multiple_of(*p)) {
        n += 1;
    }
    n
}

fn padded(name: &[u8]) -> usize {
    name.len().div_ceil(4) * 4
}

pub fn lookup(r: &mut Reader, path: &str) -> Result<Option<FileEntry>, Error> {
    let mut header = [0u8; 0x50];
    r.read_at(0, &mut header)?;
    let field = |k: usize| u64_at(&header, 8 * k);
    let read = |r: &mut Reader, off: u64, len: u64| -> Result<Vec<u8>, Error> {
        let mut b = vec![0u8; len as usize];
        r.read_at(off, &mut b)?;
        Ok(b)
    };
    let dir_hash = read(r, field(1), field(2))?;
    let dirs = read(r, field(3), field(4))?;
    let file_hash = read(r, field(5), field(6))?;
    let files = read(r, field(7), field(8))?;
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').filter(|p| !p.is_empty()).collect();
    let Some((file, folders)) = parts.split_last() else { return Ok(None) };
    let find = |table: &[u8], meta: &[u8], parent: u32, name: &[u8], next_at: usize, name_at: usize| -> Option<u32> {
        let buckets = table.len() / 4;
        let mut at = u32_at(table, 4 * (hash(parent, name) as usize % buckets));
        while at != EMPTY {
            let e = at as usize;
            let len = u32_at(meta, e + name_at - 4) as usize;
            if u32_at(meta, e) == parent && meta.get(e + name_at..e + name_at + len) == Some(name) {
                return Some(at);
            }
            at = u32_at(meta, e + next_at);
        }
        None
    };
    let mut dir = 0u32;
    for f in folders {
        match find(&dir_hash, &dirs, dir, f.as_bytes(), 0x10, 0x18) {
            Some(d) => dir = d,
            None => return Ok(None),
        }
    }
    let Some(at) = find(&file_hash, &files, dir, file.as_bytes(), 0x18, 0x20) else { return Ok(None) };
    let e = at as usize;
    Ok(Some(FileEntry { path: format!("/{}", parts.join("/")), offset: field(9) + u64_at(&files, e + 8), size: u64_at(&files, e + 16) }))
}

struct Dir {
    name: Vec<u8>,
    parent: usize,
    path: std::path::PathBuf,
    dirs: Vec<usize>,
    files: Vec<usize>,
    offset: u32,
}

struct FileItem {
    name: Vec<u8>,
    path: std::path::PathBuf,
    size: u64,
    offset: u32,
    data: u64,
}

fn sorted_entries(dir: &std::path::Path) -> std::io::Result<Vec<std::fs::DirEntry>> {
    let mut v: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    v.sort_by_key(|e| e.file_name());
    Ok(v)
}

pub fn empty() -> Vec<u8> {
    from_files(&[])
}

pub fn from_files(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut sorted: Vec<&(String, Vec<u8>)> = entries.iter().collect();
    sorted.sort_by_key(|(name, _)| name);
    let mut dirs = vec![Dir { name: Vec::new(), parent: 0, path: Default::default(), dirs: Vec::new(), files: (0..sorted.len()).collect(), offset: 0 }];
    let mut files: Vec<FileItem> = sorted.iter().map(|(name, data)| FileItem { name: name.as_bytes().to_vec(), path: Default::default(), size: data.len() as u64, offset: 0, data: 0 }).collect();
    let (header, order, tables_at, tables) = layout(&mut dirs, &mut files);
    let mut out = header;
    for &i in &order {
        out.resize((DATA_START + files[i].data) as usize, 0);
        out.extend_from_slice(&sorted[i].1);
    }
    out.resize(tables_at as usize, 0);
    out.extend(tables);
    out
}

pub fn build(dir: &std::path::Path, out: &std::path::Path) -> Result<u64, Error> {
    use std::io::Write;
    let mut dirs = vec![Dir { name: Vec::new(), parent: 0, path: dir.to_path_buf(), dirs: Vec::new(), files: Vec::new(), offset: 0 }];
    let mut files: Vec<FileItem> = Vec::new();
    let mut k = 0;
    while k < dirs.len() {
        for e in sorted_entries(&dirs[k].path.clone())? {
            let name = e.file_name().to_string_lossy().into_owned().into_bytes();
            if e.file_type()?.is_dir() {
                let i = dirs.len();
                dirs.push(Dir { name, parent: k, path: e.path(), dirs: Vec::new(), files: Vec::new(), offset: 0 });
                dirs[k].dirs.push(i);
            } else {
                let i = files.len();
                files.push(FileItem { name, path: e.path(), size: e.metadata()?.len(), offset: 0, data: 0 });
                dirs[k].files.push(i);
            }
        }
        k += 1;
    }
    let (header, order, tables, table_bytes) = layout(&mut dirs, &mut files);
    let end = tables + table_bytes.len() as u64;
    let tmp = out.with_extension("part");
    let mut w = std::io::BufWriter::with_capacity(8 << 20, File::create(&tmp)?);
    w.write_all(&header)?;
    let mut written = 0u64;
    for &i in &order {
        let f = &files[i];
        w.write_all(&vec![0u8; (f.data - written) as usize])?;
        let n = std::io::copy(&mut File::open(&f.path)?, &mut w)?;
        if n != f.size {
            return Err(Error::Malformed("a RomFS file changed while it was read"));
        }
        written = f.data + f.size;
    }
    w.write_all(&vec![0u8; (tables - DATA_START - written) as usize])?;
    w.write_all(&table_bytes)?;
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, out)?;
    Ok(end)
}

fn layout(dirs: &mut [Dir], files: &mut [FileItem]) -> (Vec<u8>, Vec<usize>, u64, Vec<u8>) {
    let mut at = 0u32;
    for d in dirs.iter_mut() {
        d.offset = at;
        at += (0x18 + padded(&d.name)) as u32;
    }
    let dir_meta_size = at as u64;
    let mut order = Vec::new();
    for d in dirs.iter() {
        order.extend(d.files.iter().copied());
    }
    let (mut at, mut data) = (0u32, 0u64);
    for &i in &order {
        let f = &mut files[i];
        f.offset = at;
        at += (0x20 + padded(&f.name)) as u32;
        data = data.div_ceil(0x10) * 0x10;
        f.data = data;
        data += f.size;
    }
    let file_meta_size = at as u64;
    let dir_buckets = table_size(dirs.len());
    let file_buckets = table_size(files.len());
    let mut dir_hash = vec![EMPTY; dir_buckets];
    let mut file_hash = vec![EMPTY; file_buckets];
    let mut dir_meta = vec![0u8; dir_meta_size as usize];
    let mut file_meta = vec![0u8; file_meta_size as usize];
    let put32 = |b: &mut [u8], at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let put64 = |b: &mut [u8], at: usize, v: u64| b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    for (i, d) in dirs.iter().enumerate() {
        let parent = dirs[d.parent].offset;
        let siblings = if i == 0 { &[][..] } else { &dirs[d.parent].dirs[..] };
        let sibling = siblings.iter().position(|&s| s == i).and_then(|p| siblings.get(p + 1)).map_or(EMPTY, |&s| dirs[s].offset);
        let bucket = hash(parent, &d.name) as usize % dir_buckets;
        let e = d.offset as usize;
        put32(&mut dir_meta, e, parent);
        put32(&mut dir_meta, e + 4, sibling);
        put32(&mut dir_meta, e + 8, d.dirs.first().map_or(EMPTY, |&c| dirs[c].offset));
        put32(&mut dir_meta, e + 12, d.files.first().map_or(EMPTY, |&c| files[c].offset));
        put32(&mut dir_meta, e + 16, dir_hash[bucket]);
        put32(&mut dir_meta, e + 20, d.name.len() as u32);
        dir_meta[e + 24..e + 24 + d.name.len()].copy_from_slice(&d.name);
        dir_hash[bucket] = d.offset;
    }
    for d in dirs.iter() {
        for (j, &i) in d.files.iter().enumerate() {
            let f = &files[i];
            let sibling = d.files.get(j + 1).map_or(EMPTY, |&s| files[s].offset);
            let bucket = hash(d.offset, &f.name) as usize % file_buckets;
            let e = f.offset as usize;
            put32(&mut file_meta, e, d.offset);
            put32(&mut file_meta, e + 4, sibling);
            put64(&mut file_meta, e + 8, f.data);
            put64(&mut file_meta, e + 16, f.size);
            put32(&mut file_meta, e + 24, file_hash[bucket]);
            put32(&mut file_meta, e + 28, f.name.len() as u32);
            file_meta[e + 32..e + 32 + f.name.len()].copy_from_slice(&f.name);
            file_hash[bucket] = f.offset;
        }
    }
    let tables = (DATA_START + data).div_ceil(4) * 4;
    let dir_hash_at = tables;
    let dir_meta_at = dir_hash_at + 4 * dir_buckets as u64;
    let file_hash_at = dir_meta_at + dir_meta_size;
    let file_meta_at = file_hash_at + 4 * file_buckets as u64;
    let mut header = vec![0u8; DATA_START as usize];
    for (k, v) in [HEADER, dir_hash_at, 4 * dir_buckets as u64, dir_meta_at, dir_meta_size, file_hash_at, 4 * file_buckets as u64, file_meta_at, file_meta_size, DATA_START].iter().enumerate() {
        put64(&mut header, 8 * k, *v);
    }
    let mut table_bytes: Vec<u8> = dir_hash.iter().flat_map(|v| v.to_le_bytes()).collect();
    table_bytes.extend_from_slice(&dir_meta);
    table_bytes.extend(file_hash.iter().flat_map(|v| v.to_le_bytes()));
    table_bytes.extend_from_slice(&file_meta);
    (header, order, tables, table_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_matches_a_built_empty_directory() {
        let dir = std::env::temp_dir().join(format!("kirikae-romfs-empty-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let image = dir.join("image.bin");
        let size = build(&dir.join("src"), &image).unwrap();
        let built = std::fs::read(&image).unwrap();
        assert_eq!(built.len() as u64, size);
        assert_eq!(empty(), built);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_files_matches_a_built_directory() {
        let dir = std::env::temp_dir().join(format!("kirikae-romfs-files-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let entries = [("version.dat".to_string(), vec![0, 0, 0, 7]), ("b_list".to_string(), b"seventeen bytes!!".to_vec()), ("a".to_string(), Vec::new())];
        for (name, data) in &entries {
            std::fs::write(dir.join("src").join(name), data).unwrap();
        }
        let image = dir.join("image.bin");
        let size = build(&dir.join("src"), &image).unwrap();
        assert_eq!(from_files(&entries), std::fs::read(&image).unwrap());
        let mut r = Reader::open(&RomFs { path: image.clone(), offset: 0, size, compression: None }).unwrap();
        let f = lookup(&mut r, "/b_list").unwrap().expect("b_list is in the image");
        let mut data = vec![0u8; f.size as usize];
        r.read_at(f.offset, &mut data).unwrap();
        assert_eq!(data, b"seventeen bytes!!");
        let _ = std::fs::remove_dir_all(&dir);
    }

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

    #[test]
    fn built_images_read_back() {
        let dir = std::env::temp_dir().join(format!("kirikae-romfs-build-{}", std::process::id()));
        let src = dir.join("romfs");
        std::fs::create_dir_all(src.join("Model/Sub")).unwrap();
        std::fs::create_dir_all(src.join("Empty")).unwrap();
        std::fs::write(src.join("top.txt"), b"top level").unwrap();
        std::fs::write(src.join("Model/a.bin"), vec![7u8; 1000]).unwrap();
        std::fs::write(src.join("Model/Sub/b.txt"), b"nested file").unwrap();
        for k in 0..40 {
            std::fs::write(src.join(format!("Model/f{k}.dat")), format!("file {k}")).unwrap();
        }
        let image = dir.join("romfs.bin");
        let size = build(&src, &image).unwrap();
        assert_eq!(std::fs::metadata(&image).unwrap().len(), size);
        let mut r = Reader::open(&RomFs { path: image.clone(), offset: 0, size, compression: None }).unwrap();
        let all = list(&mut r, usize::MAX).unwrap();
        assert_eq!(all.len(), 43);
        for (path, want) in [("/top.txt", b"top level".to_vec()), ("/Model/Sub/b.txt", b"nested file".to_vec()), ("/Model/f33.dat", b"file 33".to_vec())] {
            let f = lookup(&mut r, path).unwrap().unwrap_or_else(|| panic!("{path} not found"));
            let mut got = vec![0u8; f.size as usize];
            r.read_at(f.offset, &mut got).unwrap();
            assert_eq!(got, want, "{path}");
        }
        assert!(lookup(&mut r, "/Model/missing.bin").unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
