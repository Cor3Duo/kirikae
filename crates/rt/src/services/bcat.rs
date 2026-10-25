//! BCAT (Background Content Asymmetric synchronized delivery and Transmission).
//!
//! The runtime never talks to Nintendo's delivery servers. The delivery cache is backed
//! by the save-data model: the object graph mounts `<KIRIKAE_SAVES>/<title>/bcat` (or the
//! `bcat-*` save `fs` already created for the title) and lists what is really there, so a
//! title with a pre-populated cache sees its files and a title that never synced gets an
//! empty cache instead of an error. Command numbers, names and reply shapes follow
//! Switchbrew's "BCAT services" page; the result codes use module 122.
//!
//! `IDeliveryCacheFileService` and `IDeliveryCacheDirectoryService` have no close command:
//! the guest drops the session object, which drops the open entity here. A later
//! `CreateFileService`/`CreateDirectoryService` therefore starts closed, and a sync request
//! is a no-op that immediately reports `Done`.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::fs::saves_root;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

/// `nn::bcat` result module (Switchbrew's error-code table).
const MODULE: u32 = 122;
const SHIFT: u32 = 9;

const INVALID_ARGUMENT: u32 = (1 << SHIFT) | MODULE;
const ENTITY_NOT_FOUND: u32 = (2 << SHIFT) | MODULE;
const ENTITY_ALREADY_OPEN: u32 = (6 << SHIFT) | MODULE;
const NO_OPEN_ENTITY: u32 = (7 << SHIFT) | MODULE;
const PERMISSION_DENIED: u32 = (90 << SHIFT) | MODULE;

const DIRECTORY_NAME_SIZE: usize = 0x20;
const FILE_NAME_SIZE: usize = 0x20;
const DIRECTORY_ENTRY_SIZE: usize = 0x38;
const DIGEST_SIZE: usize = 0x10;
const PROGRESS_IMPL_SIZE: usize = 0x200;
const STORAGE_USAGE_SIZE: usize = 0x30;
const PASSPHRASE_SIZE: u64 = 0x40;
/// Size `fs` provisions for a save with the default creation parameters, which is also
/// the quota the storage-info query reports when the cache directory is not there.
const SAVE_SIZE: u64 = 0x200_0000;
/// `DeliveryCacheProgressImpl::Status::Done`.
const PROGRESS_DONE: u8 = 9;

// ------------------------------------------------------------------ delivery cache

/// The delivery-cache tree mounted for one title.
struct Cache {
    root: PathBuf,
}

impl Cache {
    fn at(root: PathBuf) -> Cache {
        Cache { root }
    }

    /// Mounts the title's BCAT save. Nothing is created: a title without a cache gets an
    /// empty view, which is what `bcat` reports for a cache that has never synced.
    fn mount(title: u64) -> Cache {
        Cache::at(find_cache_root(&saves_root(title)))
    }

    /// Cache directories in the order `EnumerateDeliveryCacheDirectory` hands them out.
    fn directories(&self) -> Vec<String> {
        let Ok(read) = fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut dirs: Vec<String> = read
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        dirs.sort();
        dirs
    }

    fn directory(&self, name: &str) -> Option<PathBuf> {
        let path = self.root.join(name);
        path.is_dir().then_some(path)
    }

    /// Files of an opened cache directory, sorted by name.
    fn files(&self, directory: &str) -> Vec<(String, u64)> {
        let Some(dir) = self.directory(directory) else {
            return Vec::new();
        };
        let Ok(read) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut files: Vec<(String, u64)> = read
            .flatten()
            .filter_map(|entry| {
                let meta = entry.metadata().ok()?;
                meta.is_file()
                    .then(|| (entry.file_name().to_string_lossy().into_owned(), meta.len()))
            })
            .collect();
        files.sort();
        files
    }

    fn open(&self, directory: &str, file: &str) -> Option<OpenFile> {
        let path = self.root.join(directory).join(file);
        let meta = fs::metadata(&path).ok()?;
        meta.is_file().then_some(OpenFile {
            path,
            size: meta.len(),
        })
    }

    /// Bytes used by the cache tree, zero when nothing is mounted.
    fn usage(&self) -> u64 {
        tree_usage(&self.root)
    }

    fn clear(&self) -> std::io::Result<()> {
        let Ok(read) = fs::read_dir(&self.root) else {
            return Ok(());
        };
        for entry in read.flatten() {
            if entry.path().is_dir() {
                fs::remove_dir_all(entry.path())?;
            } else {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }
}

/// `fs` names account saves `bcat-<id>` and keeps the committed tree in `data/`. The
/// runtime's own cache is the plain `bcat/` directory, so both layouts are accepted.
fn find_cache_root(title_root: &Path) -> PathBuf {
    let direct = title_root.join("bcat");
    if direct.is_dir() {
        return direct;
    }
    let Ok(read) = fs::read_dir(title_root) else {
        return direct;
    };
    let mut candidates: Vec<PathBuf> = read
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("bcat-"))
        .map(|entry| {
            let data = entry.path().join("data");
            if data.is_dir() { data } else { entry.path() }
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next().unwrap_or(direct)
}

fn tree_usage(path: &Path) -> u64 {
    let Ok(read) = fs::read_dir(path) else {
        return 0;
    };
    read.flatten()
        .map(|entry| match entry.metadata() {
            Ok(meta) if meta.is_dir() => tree_usage(&entry.path()),
            Ok(meta) => meta.len(),
            Err(_) => 0,
        })
        .sum()
}

// ------------------------------------------------------------------ names and digests

/// A `DirectoryName`/`FileName`: ASCII letters, digits, `_`, the separator for that kind
/// and a NUL terminator in the last byte.
fn valid_name(raw: &[u8], separator: u8) -> Option<&str> {
    if raw.len() != FILE_NAME_SIZE || raw[FILE_NAME_SIZE - 1] != 0 {
        return None;
    }
    let text = raw.split(|&b| b == 0).next()?;
    if text.is_empty()
        || !text
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == separator)
    {
        return None;
    }
    std::str::from_utf8(text).ok()
}

fn valid_directory(data: &[u8], off: usize) -> Option<&str> {
    valid_name(data.get(off..off + DIRECTORY_NAME_SIZE)?, b'-')
}

fn valid_file(data: &[u8], off: usize) -> Option<&str> {
    valid_name(data.get(off..off + FILE_NAME_SIZE)?, b'.')
}

fn write_name(out: &mut [u8], name: &str) {
    let bytes = name.as_bytes();
    let len = bytes.len().min(out.len().saturating_sub(1));
    out[..len].copy_from_slice(&bytes[..len]);
}

/// `nn::bcat::Digest`, the MD5 of the file. The digest only marks a file as different
/// from another one with the same name, so MD5 is what the console reports.
fn digest(bytes: &[u8]) -> [u8; DIGEST_SIZE] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76a_a478,
        0xe8c7_b756,
        0x2420_70db,
        0xc1bd_ceee,
        0xf57c_0faf,
        0x4787_c62a,
        0xa830_4613,
        0xfd46_9501,
        0x6980_98d8,
        0x8b44_f7af,
        0xffff_5bb1,
        0x895c_d7be,
        0x6b90_1122,
        0xfd98_7193,
        0xa679_438e,
        0x49b4_0821,
        0xf61e_2562,
        0xc040_b340,
        0x265e_5a51,
        0xe9b6_c7aa,
        0xd62f_105d,
        0x0244_1453,
        0xd8a1_e681,
        0xe7d3_fbc8,
        0x21e1_cde6,
        0xc337_07d6,
        0xf4d5_0d87,
        0x455a_14ed,
        0xa9e3_e905,
        0xfcef_a3f8,
        0x676f_02d9,
        0x8d2a_4c8a,
        0xfffa_3942,
        0x8771_f681,
        0x6d9d_6122,
        0xfde5_380c,
        0xa4be_ea44,
        0x4bde_cfa9,
        0xf6bb_4b60,
        0xbebf_bc70,
        0x289b_7ec6,
        0xeaa1_27fa,
        0xd4ef_3085,
        0x0488_1d05,
        0xd9d4_d039,
        0xe6db_99e5,
        0x1fa2_7cf8,
        0xc4ac_5665,
        0xf429_2244,
        0x432a_ff97,
        0xab94_23a7,
        0xfc93_a039,
        0x655b_59c3,
        0x8f0c_cc92,
        0xffef_f47d,
        0x8584_5dd1,
        0x6fa8_7e4f,
        0xfe2c_e6e0,
        0xa301_4314,
        0x4e08_11a1,
        0xf753_7e82,
        0xbd3a_f235,
        0x2ad7_d2bb,
        0xeb86_d391,
    ];

    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let (mut a0, mut b0, mut c0, mut d0) =
        (0x6745_2301u32, 0xefcd_ab89u32, 0x98ba_dcfe, 0x1032_5476);
    for chunk in msg.chunks(64) {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let at = i * 4;
            *word = u32::from_le_bytes(chunk[at..at + 4].try_into().unwrap_or_default());
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let sum = a
                .wrapping_add(f)
                .wrapping_add(K[i])
                .wrapping_add(m[g])
                .rotate_left(S[i])
                .wrapping_add(b);
            (a, b, c, d) = (d, sum, b, c);
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = [0u8; DIGEST_SIZE];
    for (i, word) in [a0, b0, c0, d0].into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

// ------------------------------------------------------------------ service creator

pub struct Bcat {
    pub name: &'static str,
}

impl Service for Bcat {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let cache = Arc::new(Cache::mount(c.process().title_id));
        let object: Arc<dyn Service> = match req.cmd {
            // CreateBcatService(ProcessId) and the two storage-service creators. The
            // application id only selects the cache on a real console; here the title is
            // fixed by the process.
            0 => Arc::new(BcatService { cache }),
            1 | 2 => Arc::new(Storage::new(cache)),
            // CreateDeliveryCacheProgressService{,WithApplicationId}, 2.0.0-2.3.0 only.
            3 | 4 => Arc::new(Progress::new(c, "")),
            _ => return unimplemented(self.name, c, req),
        };
        Reply::ok().object(object)
    }
}

// ------------------------------------------------------------------ nn::bcat::ipc::IBcatService

struct BcatService {
    cache: Arc<Cache>,
}

impl BcatService {
    fn command(&self, c: &Cpu, req: &Request) -> Option<Reply> {
        let passphrase_too_long = req.in_buf(0).is_some_and(|b| b.size > PASSPHRASE_SIZE);
        let reply = match req.cmd {
            // A sync completes immediately: there is no network transfer, so the progress
            // object reports `Done` and its event is already signalled.
            10100 | 20100 => Reply::ok().object(Arc::new(Progress::new(c, ""))),
            10101 => {
                let name = valid_directory(&req.data, 0).unwrap_or("");
                Reply::ok().object(Arc::new(Progress::new(c, name)))
            }
            20101 => {
                let name = valid_directory(&req.data, 8).unwrap_or("");
                Reply::ok().object(Arc::new(Progress::new(c, name)))
            }
            10200 => Reply::ok(),
            20300 => Reply::ok().object(Arc::new(EventHolder::new(c, "bcat/INotifierService"))),
            20301 => Reply::ok().object(Arc::new(EventHolder::new(
                c,
                "bcat/IDeliveryTaskSuspensionService",
            ))),
            20400
            | 20401
            | 20410
            | 20500
            | 30101
            | 30102
            | 30200..=30203
            | 30210
            | 30300
            | 90202 => Reply::ok(),
            30100 => {
                if req.u64(0) == 0 || passphrase_too_long {
                    Reply::err(INVALID_ARGUMENT)
                } else {
                    Reply::ok()
                }
            }
            // No delivery tasks are registered, so every list is empty.
            90100 | 90101 | 90300 => Reply::ok().u32(0),
            90200 => Reply::ok().u32(0).u32(0),
            90201 => {
                if req.u64(0) == 0 {
                    Reply::err(INVALID_ARGUMENT)
                } else {
                    match self.cache.clear() {
                        Ok(()) => Reply::ok(),
                        Err(_) => Reply::err(PERMISSION_DENIED),
                    }
                }
            }
            90301 => self.storage_info(),
            _ => return None,
        };
        Some(reply)
    }

    /// `GetDeliveryCacheStorageUsage` (the delivery-cache storage info query): used bytes,
    /// total quota and free bytes as `u64`, followed by reserved space to the documented
    /// 0x30-byte output.
    fn storage_info(&self) -> Reply {
        let used = self.cache.usage();
        let total = SAVE_SIZE.max(used);
        let reply = Reply::ok()
            .u64(used)
            .u64(total)
            .u64(total - used)
            .u64(0)
            .u64(0)
            .u64(0);
        debug_assert_eq!(reply.data.len(), STORAGE_USAGE_SIZE);
        reply
    }
}

impl Service for BcatService {
    fn name(&self) -> String {
        "bcat/IBcatService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(c, req) {
            Some(reply) => reply,
            None => unimplemented("bcat/IBcatService", c, req),
        }
    }
}

/// `IDeliveryCacheProgressService`: the event and the 0x200-byte progress state.
struct Progress {
    event: u32,
    imp: [u8; PROGRESS_IMPL_SIZE],
}

/// The 0x200-byte `DeliveryCacheProgressImpl`: `status`, `result`, the current
/// directory/file names and the byte counters. This runtime finishes every sync at once,
/// so the state is always `Done` with zero bytes moved.
fn progress_impl(directory: &str) -> [u8; PROGRESS_IMPL_SIZE] {
    let mut imp = [0u8; PROGRESS_IMPL_SIZE];
    imp[0] = PROGRESS_DONE;
    write_name(&mut imp[8..8 + DIRECTORY_NAME_SIZE], directory);
    imp
}

impl Progress {
    fn new(c: &Cpu, directory: &str) -> Progress {
        Progress {
            event: c.process().create_event(true),
            imp: progress_impl(directory),
        }
    }

    fn command(&self, c: &Cpu, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            0 => Reply::ok().copy_handle(self.event),
            1 => {
                if let Some(b) = req.out_buf(0) {
                    let n = (b.size as usize).min(self.imp.len());
                    c.process().memory.write(b.addr, &self.imp[..n]);
                }
                Reply::ok()
            }
            _ => return None,
        };
        Some(reply)
    }
}

impl Service for Progress {
    fn name(&self) -> String {
        "bcat/IDeliveryCacheProgressService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(c, req) {
            Some(reply) => reply,
            None => unimplemented("bcat/IDeliveryCacheProgressService", c, req),
        }
    }
}

/// `INotifierService` and `IDeliveryTaskSuspensionService` both only expose `GetEvent`.
struct EventHolder {
    name: &'static str,
    event: u32,
}

impl EventHolder {
    fn new(c: &Cpu, name: &'static str) -> EventHolder {
        EventHolder {
            name,
            event: c.process().create_event(false),
        }
    }
}

impl Service for EventHolder {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().copy_handle(self.event),
            _ => unimplemented(self.name, c, req),
        }
    }
}

// ------------------------------------------------------------------ IDeliveryCacheStorageService

struct Storage {
    cache: Arc<Cache>,
    next: Mutex<usize>,
}

impl Storage {
    fn new(cache: Arc<Cache>) -> Storage {
        Storage {
            cache,
            next: Mutex::new(0),
        }
    }

    fn command(&self, c: &Cpu, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            0 => Reply::ok().object(Arc::new(FileService {
                cache: self.cache.clone(),
                open: Mutex::new(None),
            })),
            1 => Reply::ok().object(Arc::new(DirectoryService {
                cache: self.cache.clone(),
                open: Mutex::new(None),
            })),
            10 => {
                let room = req
                    .out_buf(0)
                    .map_or(0, |b| b.size as usize / DIRECTORY_NAME_SIZE);
                let batch = self.batch(room);
                if let Some(b) = req.out_buf(0) {
                    let mut out = vec![0u8; batch.len() * DIRECTORY_NAME_SIZE];
                    for (i, name) in batch.iter().enumerate() {
                        write_name(
                            &mut out[i * DIRECTORY_NAME_SIZE..(i + 1) * DIRECTORY_NAME_SIZE],
                            name,
                        );
                    }
                    c.process().memory.write(b.addr, &out);
                }
                Reply::ok().u32(batch.len() as u32)
            }
            _ => return None,
        };
        Some(reply)
    }

    /// Next `room` directory names. The console keeps the cursor on the service object, so
    /// two calls with a one-entry buffer walk the list instead of repeating it.
    fn batch(&self, room: usize) -> Vec<String> {
        let names = self.cache.directories();
        let mut next = self.next.lock().unwrap();
        let start = (*next).min(names.len());
        let n = (names.len() - start).min(room);
        *next += n;
        names[start..start + n].to_vec()
    }
}

impl Service for Storage {
    fn name(&self) -> String {
        "bcat/IDeliveryCacheStorageService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(c, req) {
            Some(reply) => reply,
            None => unimplemented("bcat/IDeliveryCacheStorageService", c, req),
        }
    }
}

// ------------------------------------------------------------------ IDeliveryCacheFileService

struct OpenFile {
    path: PathBuf,
    size: u64,
}

struct FileService {
    cache: Arc<Cache>,
    open: Mutex<Option<OpenFile>>,
}

impl FileService {
    fn command(&self, c: &Cpu, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            0 => self.open_entity(&req.data),
            1 => {
                let size = req.out_buf(0).map_or(0, |b| {
                    // `Read(offset, size, out_buffer)`. A guest that omits the explicit size
                    // reads as much as its buffer holds.
                    let asked = if req.data.len() >= 16 {
                        req.u64(8)
                    } else {
                        b.size
                    };
                    asked.min(b.size)
                });
                match self.read_bytes(req.u64(0), size) {
                    Ok(bytes) => {
                        if let Some(b) = req.out_buf(0) {
                            c.process().memory.write(b.addr, &bytes);
                        }
                        Reply::ok().u64(bytes.len() as u64)
                    }
                    Err(code) => Reply::err(code),
                }
            }
            2 => match self.open.lock().unwrap().as_ref() {
                Some(file) => Reply::ok().u64(file.size),
                None => Reply::err(NO_OPEN_ENTITY),
            },
            3 => match self.file_digest() {
                Ok(digest) => Reply::ok().bytes(&digest),
                Err(code) => Reply::err(code),
            },
            _ => return None,
        };
        Some(reply)
    }

    /// `Open(DirectoryName, FileName)`: a second `Open` on the same object is refused with
    /// `AlreadyOpen`, as on the console.
    fn open_entity(&self, data: &[u8]) -> Reply {
        let (Some(directory), Some(file)) = (valid_directory(data, 0), valid_file(data, 0x20))
        else {
            return Reply::err(INVALID_ARGUMENT);
        };
        let mut open = self.open.lock().unwrap();
        if open.is_some() {
            return Reply::err(ENTITY_ALREADY_OPEN);
        }
        match self.cache.open(directory, file) {
            Some(entity) => {
                *open = Some(entity);
                Reply::ok()
            }
            None => Reply::err(ENTITY_NOT_FOUND),
        }
    }

    fn read_bytes(&self, offset: u64, size: u64) -> Result<Vec<u8>, u32> {
        let open = self.open.lock().unwrap();
        let Some(entity) = open.as_ref() else {
            return Err(NO_OPEN_ENTITY);
        };
        let offset = offset.min(entity.size);
        let len = size.min(entity.size - offset) as usize;
        let mut file = fs::File::open(&entity.path).map_err(|_| ENTITY_NOT_FOUND)?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|_| ENTITY_NOT_FOUND)?;
        let mut out = vec![0u8; len];
        file.read_exact(&mut out).map_err(|_| ENTITY_NOT_FOUND)?;
        Ok(out)
    }

    fn file_digest(&self) -> Result<[u8; DIGEST_SIZE], u32> {
        let open = self.open.lock().unwrap();
        let Some(entity) = open.as_ref() else {
            return Err(NO_OPEN_ENTITY);
        };
        let bytes = fs::read(&entity.path).map_err(|_| ENTITY_NOT_FOUND)?;
        Ok(digest(&bytes))
    }
}

impl Service for FileService {
    fn name(&self) -> String {
        "bcat/IDeliveryCacheFileService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(c, req) {
            Some(reply) => reply,
            None => unimplemented("bcat/IDeliveryCacheFileService", c, req),
        }
    }
}

// ------------------------------------------------------------------ IDeliveryCacheDirectoryService

struct DirectoryService {
    cache: Arc<Cache>,
    open: Mutex<Option<String>>,
}

impl DirectoryService {
    fn command(&self, c: &Cpu, req: &Request) -> Option<Reply> {
        let reply = match req.cmd {
            0 => {
                let Some(name) = valid_directory(&req.data, 0) else {
                    return Some(Reply::err(INVALID_ARGUMENT));
                };
                match self.open_directory(name) {
                    Ok(()) => Reply::ok(),
                    Err(code) => Reply::err(code),
                }
            }
            1 => {
                let Some(b) = req.out_buf(0) else {
                    return Some(Reply::ok().u32(0));
                };
                let room = b.size as usize / DIRECTORY_ENTRY_SIZE;
                // `Read(offset, out_buffer)`: a second call continues where the first
                // stopped instead of repeating the beginning of the list.
                match self.entry_page(req.u64(0) as usize, room) {
                    Ok(entries) => {
                        let mut out = vec![0u8; entries.len() * DIRECTORY_ENTRY_SIZE];
                        for (i, entry) in entries.iter().enumerate() {
                            out[i * DIRECTORY_ENTRY_SIZE..(i + 1) * DIRECTORY_ENTRY_SIZE]
                                .copy_from_slice(entry);
                        }
                        c.process().memory.write(b.addr, &out);
                        Reply::ok().u32(entries.len() as u32)
                    }
                    Err(code) => Reply::err(code),
                }
            }
            2 => match self.entries() {
                Ok(entries) => Reply::ok().u32(entries.len() as u32),
                Err(code) => Reply::err(code),
            },
            _ => return None,
        };
        Some(reply)
    }

    fn open_directory(&self, name: &str) -> Result<(), u32> {
        let mut open = self.open.lock().unwrap();
        if open.is_some() {
            return Err(ENTITY_ALREADY_OPEN);
        }
        if self.cache.directory(name).is_none() {
            return Err(ENTITY_NOT_FOUND);
        }
        *open = Some(name.to_string());
        Ok(())
    }

    /// One page of `Read`: skip `offset` records, then take up to `room`.
    fn entry_page(
        &self,
        offset: usize,
        room: usize,
    ) -> Result<Vec<[u8; DIRECTORY_ENTRY_SIZE]>, u32> {
        Ok(self
            .entries()?
            .into_iter()
            .skip(offset)
            .take(room)
            .collect())
    }

    /// `DeliveryCacheDirectoryEntry` records (`FileName`, size, digest), one per file.
    fn entries(&self) -> Result<Vec<[u8; DIRECTORY_ENTRY_SIZE]>, u32> {
        let open = self.open.lock().unwrap();
        let Some(name) = open.as_ref() else {
            return Err(NO_OPEN_ENTITY);
        };
        let mut out = Vec::new();
        for (file, size) in self.cache.files(name) {
            let mut entry = [0u8; DIRECTORY_ENTRY_SIZE];
            write_name(&mut entry[..FILE_NAME_SIZE], &file);
            entry[FILE_NAME_SIZE..FILE_NAME_SIZE + 8].copy_from_slice(&size.to_le_bytes());
            let data = fs::read(self.cache.root.join(name).join(&file)).unwrap_or_default();
            entry[FILE_NAME_SIZE + 8..].copy_from_slice(&digest(&data));
            out.push(entry);
        }
        Ok(out)
    }
}

impl Service for DirectoryService {
    fn name(&self) -> String {
        "bcat/IDeliveryCacheDirectoryService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(c, req) {
            Some(reply) => reply,
            None => unimplemented("bcat/IDeliveryCacheDirectoryService", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("kirikae-bcat-{tag}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn name(text: &str) -> [u8; FILE_NAME_SIZE] {
        let mut out = [0u8; FILE_NAME_SIZE];
        out[..text.len()].copy_from_slice(text.as_bytes());
        out
    }

    fn mounted(title: &Path) -> Cache {
        Cache::at(find_cache_root(title))
    }

    /// `<title>/bcat` with two directories and 13 bytes in three files.
    fn populated(tag: &str) -> PathBuf {
        let title = temp_root(tag);
        let cache = title.join("bcat");
        fs::create_dir_all(cache.join("Dir-1")).unwrap();
        fs::create_dir_all(cache.join("Dir-2")).unwrap();
        fs::write(cache.join("Dir-1/a.bin"), b"alpha").unwrap();
        fs::write(cache.join("Dir-1/b.dat"), b"beta!").unwrap();
        fs::write(cache.join("Dir-2/c.bin"), b"one").unwrap();
        title
    }

    #[test]
    fn md5_matches_the_rfc_1321_vectors() {
        let hex = |d: [u8; DIGEST_SIZE]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(digest(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(digest(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            hex(digest(b"message digest")),
            "f96b697d7cb7938d525a2f31aaf161d0"
        );
        // Longer than one block, so the length padding path is covered too.
        assert_eq!(
            hex(digest(
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            )),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    #[test]
    fn a_missing_cache_is_empty() {
        let title = temp_root("empty");
        let cache = mounted(&title);
        assert_eq!(cache.root, title.join("bcat"));
        assert!(!cache.root.exists());
        assert!(cache.directories().is_empty());
        assert!(cache.files("Dir-1").is_empty());
        assert_eq!(cache.usage(), 0);
        let _ = fs::remove_dir_all(&title);
    }

    #[test]
    fn directories_and_files_are_listed_from_the_mounted_tree() {
        let title = populated("list");
        let cache = mounted(&title);
        assert_eq!(cache.directories(), ["Dir-1", "Dir-2"]);
        assert_eq!(
            cache.files("Dir-1"),
            [("a.bin".to_string(), 5), ("b.dat".to_string(), 5)]
        );
        assert_eq!(cache.files("Dir-2"), [("c.bin".to_string(), 3)]);
        assert_eq!(cache.usage(), 13);
        cache.clear().unwrap();
        assert!(cache.directories().is_empty());
        assert_eq!(cache.usage(), 0);
        let _ = fs::remove_dir_all(&title);
    }

    #[test]
    fn result_codes_use_the_bcat_module() {
        // Module 122 at bit 8, as in Switchbrew's error-code table.
        assert_eq!(INVALID_ARGUMENT, 0x27A);
        assert_eq!(ENTITY_NOT_FOUND, 0x47A);
        assert_eq!(ENTITY_ALREADY_OPEN, 0xC7A);
        assert_eq!(NO_OPEN_ENTITY, 0xE7A);
        assert_eq!(PERMISSION_DENIED, 0xB47A);
    }

    #[test]
    fn the_service_family_is_registered() {
        for name in ["bcat:u", "bcat:a", "bcat:m", "bcat:s"] {
            assert_eq!(crate::services::open(name).name(), name);
        }
    }

    #[test]
    fn the_fs_created_bcat_save_data_tree_is_mounted() {
        let title = temp_root("save-data");
        let data = title.join("bcat-0000000000000000").join("data");
        fs::create_dir_all(data.join("Dir-9")).unwrap();
        fs::write(data.join("Dir-9/file.bin"), b"payload").unwrap();
        let cache = mounted(&title);
        assert_eq!(cache.root, data);
        assert_eq!(cache.directories(), ["Dir-9"]);
        assert_eq!(cache.usage(), 7);
        let _ = fs::remove_dir_all(&title);
    }

    #[test]
    fn names_accept_only_the_documented_characters() {
        assert_eq!(valid_name(&name("Dir-1_2"), b'-'), Some("Dir-1_2"));
        assert_eq!(valid_name(&name("a.bin"), b'.'), Some("a.bin"));
        assert_eq!(valid_name(&name("a.bin"), b'-'), None);
        assert_eq!(valid_name(&name("a/b"), b'.'), None);
        assert_eq!(valid_name(&name(""), b'.'), None);
        // No terminator in the last byte.
        assert_eq!(valid_name(&[b'a'; FILE_NAME_SIZE], b'.'), None);
        // Interior NULs short-circuit at the first one.
        assert_eq!(valid_name(&name("ab"), b'.'), Some("ab"));
    }

    fn open_request(directory: &str, file: &str) -> Vec<u8> {
        let mut request = name(directory).to_vec();
        request.extend_from_slice(&name(file));
        request
    }

    #[test]
    fn file_service_reports_the_documented_errors() {
        let title = populated("errors");
        let service = FileService {
            cache: Arc::new(mounted(&title)),
            open: Mutex::new(None),
        };

        assert_eq!(service.open_entity(&[]).result, INVALID_ARGUMENT);
        assert_eq!(service.file_digest().unwrap_err(), NO_OPEN_ENTITY);
        assert_eq!(service.read_bytes(0, 4).unwrap_err(), NO_OPEN_ENTITY);

        let request = open_request("Dir-1", "a.bin");
        assert_eq!(service.open_entity(&request).result, 0);
        assert_eq!(service.open_entity(&request).result, ENTITY_ALREADY_OPEN);
        assert_eq!(service.file_digest().unwrap(), digest(b"alpha"));
        assert_eq!(service.read_bytes(0, 0x100).unwrap(), b"alpha");
        assert_eq!(service.read_bytes(4, 0x100).unwrap(), b"a");
        assert_eq!(service.read_bytes(5, 0x100).unwrap(), b"");
        assert_eq!(service.read_bytes(100, 0x100).unwrap(), b"");

        let missing = FileService {
            cache: Arc::new(mounted(&title)),
            open: Mutex::new(None),
        };
        assert_eq!(
            missing
                .open_entity(&open_request("Dir-1", "missing.bin"))
                .result,
            ENTITY_NOT_FOUND
        );
        assert_eq!(
            missing.open_entity(&open_request("nope", "a.bin")).result,
            ENTITY_NOT_FOUND
        );
        let _ = fs::remove_dir_all(&title);
    }

    #[test]
    fn directory_service_lists_entries_with_size_and_digest() {
        let title = populated("entries");
        let service = DirectoryService {
            cache: Arc::new(mounted(&title)),
            open: Mutex::new(None),
        };
        assert_eq!(service.entries().unwrap_err(), NO_OPEN_ENTITY);
        assert_eq!(service.open_directory("Dir-9"), Err(ENTITY_NOT_FOUND));
        assert_eq!(service.open_directory("Dir-1"), Ok(()));
        assert_eq!(service.open_directory("Dir-1"), Err(ENTITY_ALREADY_OPEN));

        let entries = service.entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(&entries[0][..5], b"a.bin");
        assert_eq!(
            u64::from_le_bytes(
                entries[0][FILE_NAME_SIZE..FILE_NAME_SIZE + 8]
                    .try_into()
                    .unwrap()
            ),
            5
        );
        assert_eq!(&entries[0][FILE_NAME_SIZE + 8..], &digest(b"alpha"));
        assert_eq!(&entries[1][..5], b"b.dat");
        assert_eq!(&entries[1][FILE_NAME_SIZE + 8..], &digest(b"beta!"));

        // A page continues from the requested offset instead of repeating the start.
        assert_eq!(service.entry_page(0, 1).unwrap().len(), 1);
        let second = service.entry_page(1, 1).unwrap();
        assert_eq!(&second[0][..5], b"b.dat");
        assert!(service.entry_page(2, 1).unwrap().is_empty());
        assert!(service.entry_page(99, 1).unwrap().is_empty());
        let _ = fs::remove_dir_all(&title);
    }

    #[test]
    fn clearing_the_cache_also_removes_root_files() {
        let title = populated("clear");
        fs::write(title.join("bcat/stray.bin"), b"x").unwrap();
        let cache = mounted(&title);
        assert_eq!(cache.usage(), 14);
        cache.clear().unwrap();
        assert!(!title.join("bcat/stray.bin").exists());
        assert_eq!(cache.usage(), 0);
        let _ = fs::remove_dir_all(&title);
    }

    #[test]
    fn storage_enumerates_directories_in_batches() {
        let title = populated("enumerate");
        let service = Storage::new(Arc::new(mounted(&title)));
        assert_eq!(service.batch(1), ["Dir-1"]);
        assert_eq!(service.batch(1), ["Dir-2"]);
        assert!(service.batch(4).is_empty());
        let _ = fs::remove_dir_all(&title);
    }

    #[test]
    fn progress_reports_done_and_names_the_directory() {
        let imp = progress_impl("Dir-1");
        assert_eq!(imp[0], PROGRESS_DONE);
        assert_eq!(&imp[8..13], b"Dir-1");
        assert_eq!(imp.len(), PROGRESS_IMPL_SIZE);
        assert!(imp[13..].iter().all(|&b| b == 0));
    }

    #[test]
    fn storage_info_is_thirty_bytes_of_usage() {
        let title = populated("usage");
        let service = BcatService {
            cache: Arc::new(mounted(&title)),
        };
        let reply = service.storage_info();
        assert_eq!(reply.result, 0);
        assert_eq!(reply.data.len(), STORAGE_USAGE_SIZE);
        let used = u64::from_le_bytes(reply.data[0..8].try_into().unwrap());
        let total = u64::from_le_bytes(reply.data[8..16].try_into().unwrap());
        let free = u64::from_le_bytes(reply.data[16..24].try_into().unwrap());
        assert_eq!((used, total, free), (13, SAVE_SIZE, SAVE_SIZE - 13));
        let _ = fs::remove_dir_all(&title);
    }
}
