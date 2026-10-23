use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Buf, Reply, Request, Service};

const PATH_NOT_FOUND: u32 = 0x202;
const PATH_ALREADY_EXISTS: u32 = 0x402;
const TARGET_NOT_FOUND: u32 = 0x7d402;
const DIRECTORY_NOT_EMPTY: u32 = 0x4d002;
const OUT_OF_RANGE: u32 = 0x3e402;

const ENTRY_DIRECTORY: u8 = 0;
const ENTRY_FILE: u8 = 1;
const DIRECTORY_ENTRY_SIZE: usize = 0x310;
const SAVE_SPACE: u64 = 4 << 30;

pub fn saves_root(title: u64) -> PathBuf {
    if let Some(dir) = std::env::var_os("KIRIKAE_SAVES") {
        return PathBuf::from(dir).join(format!("{title:016x}"));
    }
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("kirikae").join("saves").join(format!("{title:016x}"))
}

fn host_path(c: &Cpu, root: &Path, x: Option<&Buf>) -> Option<PathBuf> {
    let b = x?;
    let raw = c.process().memory.read(b.addr, b.size.min(0x301))?;
    let s = String::from_utf8_lossy(raw.split(|&b| b == 0).next()?).into_owned();
    let rel = Path::new(s.trim_start_matches('/'));
    if rel.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_))) {
        return None;
    }
    Some(root.join(rel))
}

pub struct FsProxy;

impl Service for FsProxy {
    fn name(&self) -> String {
        "fsp-srv".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            1 => Reply::ok(),
            18 => {
                let root = saves_root(p.title_id).join("sdcard");
                Reply::ok().object(Arc::new(HostFs::new(root)))
            }
            22 | 23 | 25 | 26 | 27 | 28 => crate::services::stub::plain_ok(),
            51..=53 => {
                let a = &req.data;
                let kind = a.get(0x28).copied().unwrap_or(1);
                let user = req.u64(0x10) ^ req.u64(0x18);
                let dir = format!("{}-{user:016x}", ["system", "account", "bcat", "device", "temporary", "cache"].get(kind as usize).unwrap_or(&"other"));
                Reply::ok().object(Arc::new(HostFs::new(saves_root(p.title_id).join(dir))))
            }
            60..=62 => Reply::ok().object(Arc::new(SaveDataInfoReader)),
            200 => match &p.romfs {
                Some(_) => Reply::ok().object(Arc::new(RomStorage)),
                None => Reply::err(TARGET_NOT_FOUND),
            },
            202 | 203 => Reply::err(TARGET_NOT_FOUND),
            1003 | 1004 | 1006 => Reply::ok(),
            1005 => Reply::ok().u32(0),
            1011 => Reply::ok().u32(0).u32(1),
            _ => unimplemented("fsp-srv", c, req),
        }
    }
}

struct RomStorage;

impl Service for RomStorage {
    fn name(&self) -> String {
        "fs/IStorage(RomFS)".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let Some(romfs) = &p.romfs else { return Reply::err(TARGET_NOT_FOUND) };
        match req.cmd {
            0 => {
                let (offset, size) = (req.u64(0), req.u64(8));
                let Some(b) = req.b.first() else { return Reply::err(OUT_OF_RANGE) };
                let mut buf = vec![0u8; size.min(b.size) as usize];
                match romfs.lock().unwrap().read_at(offset, &mut buf) {
                    Ok(n) => {
                        p.memory.write(b.addr, &buf[..n]);
                        Reply::ok()
                    }
                    Err(e) => {
                        p.warn_once(format!("RomFS read failed: {e}"));
                        Reply::err(OUT_OF_RANGE)
                    }
                }
            }
            4 => Reply::ok().u64(romfs.lock().unwrap().size()),
            _ => unimplemented("fs/IStorage(RomFS)", c, req),
        }
    }
}

struct SaveDataInfoReader;

impl Service for SaveDataInfoReader {
    fn name(&self) -> String {
        "fs/ISaveDataInfoReader".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(0),
            _ => unimplemented("fs/ISaveDataInfoReader", c, req),
        }
    }
}

pub struct HostFs {
    root: PathBuf,
}

impl HostFs {
    pub fn new(root: PathBuf) -> HostFs {
        let _ = fs::create_dir_all(&root);
        HostFs { root }
    }
}

fn io_result(r: std::io::Result<()>) -> Reply {
    match r {
        Ok(()) => Reply::ok(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Reply::err(PATH_NOT_FOUND),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Reply::err(PATH_ALREADY_EXISTS),
        Err(_) => Reply::err(DIRECTORY_NOT_EMPTY),
    }
}

impl Service for HostFs {
    fn name(&self) -> String {
        "fs/IFileSystem".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let path = |k: usize| host_path(c, &self.root, req.x.get(k));
        let Some(p0) = path(0) else {
            return match req.cmd {
                10 => Reply::ok(),
                _ => Reply::err(PATH_NOT_FOUND),
            };
        };
        match req.cmd {
            0 => {
                if p0.exists() {
                    return Reply::err(PATH_ALREADY_EXISTS);
                }
                io_result(File::create(&p0).and_then(|f| f.set_len(req.u64(8))))
            }
            1 => io_result(fs::remove_file(&p0)),
            2 => {
                if p0.exists() {
                    return Reply::err(PATH_ALREADY_EXISTS);
                }
                io_result(fs::create_dir(&p0))
            }
            3 => io_result(fs::remove_dir(&p0)),
            4 => io_result(fs::remove_dir_all(&p0)),
            5 | 6 => match path(1) {
                Some(to) => io_result(fs::rename(&p0, to)),
                None => Reply::err(PATH_NOT_FOUND),
            },
            7 => {
                match fs::metadata(&p0) {
                    Ok(m) => Reply::ok().u32(if m.is_dir() { ENTRY_DIRECTORY } else { ENTRY_FILE } as u32),
                    Err(_) => Reply::err(PATH_NOT_FOUND),
                }
            }
            8 => {
                let mode = req.u32(0);
                match OpenOptions::new().read(true).write(mode & 6 != 0).open(&p0) {
                    Ok(f) => Reply::ok().object(Arc::new(HostFile { file: Mutex::new(f), append: mode & 4 != 0 })),
                    Err(_) => Reply::err(PATH_NOT_FOUND),
                }
            }
            9 => {
                let filter = req.u32(0);
                let Ok(rd) = fs::read_dir(&p0) else { return Reply::err(PATH_NOT_FOUND) };
                let mut entries: Vec<(String, u8, u64)> = rd
                    .flatten()
                    .filter_map(|e| {
                        let m = e.metadata().ok()?;
                        let kind = if m.is_dir() { ENTRY_DIRECTORY } else { ENTRY_FILE };
                        let wanted = if kind == ENTRY_DIRECTORY { filter & 1 != 0 } else { filter & 2 != 0 };
                        wanted.then(|| (e.file_name().to_string_lossy().into_owned(), kind, m.len()))
                    })
                    .collect();
                entries.sort();
                Reply::ok().object(Arc::new(HostDir { entries, next: Mutex::new(0) }))
            }
            10 => Reply::ok(),
            11 | 12 => Reply::ok().u64(SAVE_SPACE),
            13 => {
                let r = fs::read_dir(&p0).map(|rd| {
                    for e in rd.flatten() {
                        let path = e.path();
                        let _ = if path.is_dir() { fs::remove_dir_all(&path) } else { fs::remove_file(&path) };
                    }
                });
                io_result(r)
            }
            14 => Reply::ok().bytes(&[0u8; 0x20]),
            15 => Reply::ok(),
            _ => unimplemented("fs/IFileSystem", c, req),
        }
    }
}

struct HostFile {
    file: Mutex<File>,
    append: bool,
}

impl Service for HostFile {
    fn name(&self) -> String {
        "fs/IFile".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let mut f = self.file.lock().unwrap();
        match req.cmd {
            0 => {
                let (offset, size) = (req.u64(8), req.u64(16));
                let Some(b) = req.b.first() else { return Reply::err(OUT_OF_RANGE) };
                let mut buf = vec![0u8; size.min(b.size) as usize];
                let n = f.seek(SeekFrom::Start(offset)).and_then(|_| read_full(&mut f, &mut buf)).unwrap_or(0);
                p.memory.write(b.addr, &buf[..n]);
                Reply::ok().u64(n as u64)
            }
            1 => {
                let (offset, size) = (req.u64(8), req.u64(16));
                let Some(a) = req.a.first() else { return Reply::err(OUT_OF_RANGE) };
                let data = p.memory.read(a.addr, size.min(a.size)).unwrap_or_default();
                let at = if self.append { SeekFrom::End(0) } else { SeekFrom::Start(offset) };
                io_result(f.seek(at).and_then(|_| f.write_all(&data)))
            }
            2 => io_result(f.flush()),
            3 => io_result(f.set_len(req.u64(0))),
            4 => Reply::ok().u64(f.metadata().map_or(0, |m| m.len())),
            _ => unimplemented("fs/IFile", c, req),
        }
    }
}

fn read_full(f: &mut File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

struct HostDir {
    entries: Vec<(String, u8, u64)>,
    next: Mutex<usize>,
}

impl Service for HostDir {
    fn name(&self) -> String {
        "fs/IDirectory".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let Some(b) = req.b.first() else { return Reply::ok().u64(0) };
                let room = b.size as usize / DIRECTORY_ENTRY_SIZE;
                let mut next = self.next.lock().unwrap();
                let batch = &self.entries[(*next).min(self.entries.len())..];
                let n = batch.len().min(room);
                let mut out = vec![0u8; n * DIRECTORY_ENTRY_SIZE];
                for (i, (name, kind, size)) in batch[..n].iter().enumerate() {
                    let e = &mut out[i * DIRECTORY_ENTRY_SIZE..(i + 1) * DIRECTORY_ENTRY_SIZE];
                    let bytes = name.as_bytes();
                    e[..bytes.len().min(0x300)].copy_from_slice(&bytes[..bytes.len().min(0x300)]);
                    e[0x304] = *kind;
                    e[0x308..0x310].copy_from_slice(&size.to_le_bytes());
                }
                c.process().memory.write(b.addr, &out);
                *next += n;
                Reply::ok().u64(n as u64)
            }
            1 => Reply::ok().u64(self.entries.len() as u64),
            _ => unimplemented("fs/IDirectory", c, req),
        }
    }
}
