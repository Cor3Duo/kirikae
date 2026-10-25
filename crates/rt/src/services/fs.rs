use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Buf, Reply, Request, Service};

mod cache;
mod device;
mod meta;

const PATH_NOT_FOUND: u32 = 0x202;
const PATH_ALREADY_EXISTS: u32 = 0x402;
const TARGET_LOCKED: u32 = 0xe02;
const TARGET_NOT_FOUND: u32 = 0x7d402;
const DIRECTORY_NOT_EMPTY: u32 = 0x4d002;
const OUT_OF_RANGE: u32 = 0x3e402;
const USABLE_SPACE_NOT_ENOUGH: u32 = 0x4e02;
const INVALID_ARGUMENT: u32 = 0x2ee202;
const PRECONDITION_VIOLATION: u32 = 0x2ee002;
const READ_ONLY_FILE_SYSTEM: u32 = 0x31c202;
const READ_ONLY_FILE: u32 = 0x31c802;
/// A write past the end of a file opened without the append bit.
const WRITE_BEYOND_END: u32 = 0x307202;

const ENTRY_DIRECTORY: u8 = 0;
const ENTRY_FILE: u8 = 1;
const DIRECTORY_ENTRY_SIZE: usize = 0x310;
const SAVE_SPACE: u64 = 4 << 30;

/// `SaveDataType` values the runtime keys per application instead of per user.
const DEVICE: u8 = 3;
const TEMPORARY: u8 = 4;
const CACHE: u8 = 5;

/// Smallest block size the save image format accepts.
const MIN_BLOCK_SIZE: u64 = 0x200;

const DEFAULT_SAVE_SIZE: u64 = 0x200_0000;
const DEFAULT_JOURNAL_SIZE: u64 = 0x100_0000;
const DEFAULT_BLOCK_SIZE: u64 = 0x4000;

const SAVE_META: &str = ".kirikae-save";
const SAVE_META_TMP: &str = ".kirikae-save.tmp";
const SAVE_DATA: &str = "data";
const SAVE_PENDING: &str = "pending";
const SAVE_PREV: &str = "data.prev";
const SAVE_JOURNAL: &str = ".commit-journal";
const SAVE_JOURNAL_TMP: &str = ".commit-journal.tmp";

/// `nn::fs::SaveDataInfo`: the fields end at 0x3C and the struct carries 0x24
/// bytes of reserved space, so a reader's output array has a 0x60-byte stride.
const SAVE_INFO_SIZE: usize = 0x60;
const SAVE_EXTRA_SIZE: usize = 0x200;

static DEFAULT_SIZE: AtomicU64 = AtomicU64::new(DEFAULT_SAVE_SIZE);
static DEFAULT_JOURNAL: AtomicU64 = AtomicU64::new(DEFAULT_JOURNAL_SIZE);
static AUTO_CREATE: AtomicBool = AtomicBool::new(true);
/// `SetSdCardAccessibility` / `IsSdCardAccessible`. The runtime serves the SD
/// card from a host directory, so it starts out accessible.
static SD_ACCESSIBLE: AtomicBool = AtomicBool::new(true);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn saves_root(title: u64) -> PathBuf {
    if let Some(dir) = std::env::var_os("KIRIKAE_SAVES") {
        return PathBuf::from(dir).join(format!("{title:016x}"));
    }
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("kirikae")
        .join("saves")
        .join(format!("{title:016x}"))
}

fn host_path(c: &Cpu, root: &Path, x: Option<&Buf>) -> Option<PathBuf> {
    let b = x?;
    let raw = c.process().memory.read(b.addr, b.size.min(0x301))?;
    let s = String::from_utf8_lossy(raw.split(|&b| b == 0).next()?).into_owned();
    let rel = Path::new(s.trim_start_matches('/'));
    if rel
        .components()
        .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
    {
        return None;
    }
    Some(root.join(rel))
}

pub struct FsProxy;

fn reply(result: Result<(), u32>) -> Reply {
    match result {
        Ok(()) => Reply::ok(),
        Err(code) => Reply::err(code),
    }
}

impl Service for FsProxy {
    fn name(&self) -> String {
        "fsp-srv".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            1 => Reply::ok(),
            2 => match &p.romfs {
                Some(romfs) => {
                    match kirikae_loader::romfs::list(&mut romfs.lock().unwrap(), usize::MAX) {
                        Ok(files) => Reply::ok().object(Arc::new(RomFileSystem::new(files))),
                        Err(e) => {
                            p.warn_once(format!("RomFS listing failed: {e}"));
                            Reply::err(TARGET_NOT_FOUND)
                        }
                    }
                }
                None => Reply::err(TARGET_NOT_FOUND),
            },
            18 => {
                let root = saves_root(p.title_id).join("sdcard");
                Reply::ok().object(Arc::new(HostFs::new(root)))
            }
            21 => match device::delete_application(&saves_root(p.title_id), req.u64(0)) {
                Ok(()) => Reply::ok(),
                Err(code) => Reply::err(code),
            },
            22 | 23 => {
                let Some(attr) = SaveAttr::parse(&req.data, 0) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                let mut creation = SaveCreation::parse(&req.data, 0x40);
                if req.cmd == 22 {
                    creation.meta_size = req.u32(0x80);
                }
                let space = if req.cmd == 23 { 0 } else { creation.space };
                reply(create_save(
                    &saves_root(p.title_id),
                    p.title_id,
                    space,
                    attr,
                    creation,
                ))
            }
            24 => unimplemented_named(
                "fsp-srv",
                "RegisterSaveDataFileSystemAtomicDeletion",
                c,
                req,
            ),
            25 => {
                let space = req.data.first().copied().unwrap_or(1);
                match delete_save(&saves_root(p.title_id), req.u64(8), Some(space)) {
                    Ok(()) => Reply::ok(),
                    Err(code) => Reply::err(code),
                }
            }
            26 => crate::services::stub::plain_ok(),
            // IsExFatSupported: the runtime reads host directories, so it never
            // needs the console's exFAT driver.
            27 => Reply::ok().u8(0),
            28 => {
                let space = req.data.first().copied().unwrap_or(1);
                let Some(attr) = SaveAttr::parse(&req.data, 8) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                match delete_save_by_attr(
                    &saves_root(p.title_id),
                    space,
                    normalize_attr(attr, p.title_id),
                ) {
                    Ok(()) => Reply::ok(),
                    Err(code) => Reply::err(code),
                }
            }
            32 => {
                let space = req.data.first().copied().unwrap_or(1);
                let id = req.u64(8);
                let size = req.u64(0x10);
                let journal = req.u64(0x18);
                match extend_save(&saves_root(p.title_id), space, id, size, journal) {
                    Ok(()) => Reply::ok(),
                    Err(code) => Reply::err(code),
                }
            }
            33 => {
                let index = u16::from_le_bytes([req.u32(0) as u8, (req.u32(0) >> 8) as u8]);
                reply(cache::delete(&saves_root(p.title_id), index))
            }
            34 => {
                let index = u16::from_le_bytes([req.u32(0) as u8, (req.u32(0) >> 8) as u8]);
                match cache::size(&saves_root(p.title_id), index) {
                    Ok((free, total)) => Reply::ok().u64(free).u64(total),
                    Err(code) => Reply::err(code),
                }
            }
            35 => {
                let Some(attr) = SaveAttr::parse(&req.data, 0) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                let mut creation = SaveCreation::parse(&req.data, 0x40);
                creation.meta_size = req.u32(0x90);
                reply(create_save(
                    &saves_root(p.title_id),
                    p.title_id,
                    creation.space,
                    attr,
                    creation,
                ))
            }
            37 => {
                let Some(attr) = SaveAttr::parse(&req.data, 8) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                let creation = SaveCreation {
                    size: req.u64(0x48),
                    journal_size: req.u64(0x50),
                    block_size: req.u64(0x58),
                    owner_id: req.u64(0x60),
                    flags: req.u32(0x68),
                    space: req.data.get(0x6c).copied().unwrap_or(1),
                    meta_size: req.u32(0x98),
                };
                reply(create_save(
                    &saves_root(p.title_id),
                    p.title_id,
                    creation.space,
                    attr,
                    creation,
                ))
            }
            51..=53 => {
                let space = req.data.first().copied().unwrap_or(1);
                let Some(attr) = SaveAttr::parse(&req.data, 8) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                let read_only = req.cmd == 53;
                let allow_create = AUTO_CREATE.load(Ordering::Relaxed);
                match SaveData::mount(
                    saves_root(p.title_id),
                    p.title_id,
                    space,
                    attr,
                    read_only,
                    allow_create,
                ) {
                    Ok(save) => Reply::ok().object(Arc::new(SaveFs { save })),
                    Err(code) => {
                        p.warn_once(format!("fs: save mount failed with {code:#x}"));
                        Reply::err(code)
                    }
                }
            }
            57 => {
                let space = req.data.first().copied().unwrap_or(1);
                let id = req.u64(8);
                read_save_extra(c, req, &saves_root(p.title_id), id, Some(space))
            }
            58 => read_save_extra(c, req, &saves_root(p.title_id), req.u64(0), None),
            59 => {
                let id = req.u64(0);
                write_save_extra(c, req, &saves_root(p.title_id), id, None)
            }
            60..=62 => {
                let filter = match req.cmd {
                    60 => InfoFilter::All,
                    61 => InfoFilter::Space(req.data.first().copied().unwrap_or(1)),
                    _ => InfoFilter::Cache,
                };
                Reply::ok().object(Arc::new(SaveDataInfoReader::new(
                    saves_root(p.title_id),
                    filter,
                )))
            }
            67 => {
                let Some(mut filter) = SaveDataFilter::parse(&req.data, 8) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                filter.attr = normalize_attr(filter.attr, p.title_id);
                let space = req.data.first().copied().unwrap_or(1);
                find_save_with_filter(c, req, &saves_root(p.title_id), space, filter)
            }
            68 => {
                let Some(mut filter) = SaveDataFilter::parse(&req.data, 8) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                filter.attr = normalize_attr(filter.attr, p.title_id);
                Reply::ok().object(Arc::new(SaveDataInfoReader::new(
                    saves_root(p.title_id),
                    InfoFilter::Filter(filter),
                )))
            }
            69 => {
                let space = req.data.first().copied().unwrap_or(1);
                let Some(attr) = SaveAttr::parse(&req.data, 8) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                read_extra_by_attr(
                    c,
                    req,
                    &saves_root(p.title_id),
                    space,
                    normalize_attr(attr, p.title_id),
                )
            }
            70 | 71 => {
                let space = req.data.first().copied().unwrap_or(1);
                let Some(attr) = SaveAttr::parse(&req.data, 8) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                write_extra_by_attr(
                    c,
                    req,
                    &saves_root(p.title_id),
                    space,
                    normalize_attr(attr, p.title_id),
                )
            }
            80 => {
                let space = req.data.first().copied().unwrap_or(1);
                let Some(attr) = SaveAttr::parse(&req.data, 8) else {
                    return Reply::err(INVALID_ARGUMENT);
                };
                let meta_type = req.data.get(0x48).copied().unwrap_or(0);
                match meta::SaveMetaFile::open(
                    &saves_root(p.title_id),
                    space,
                    normalize_attr(attr, p.title_id),
                    meta_type,
                ) {
                    Ok(file) => Reply::ok().object(Arc::new(file)),
                    Err(code) => Reply::err(code),
                }
            }
            200 => match &p.romfs {
                Some(_) => Reply::ok().object(Arc::new(RomStorage)),
                None => Reply::err(TARGET_NOT_FOUND),
            },
            202 | 203 => Reply::err(TARGET_NOT_FOUND),
            // OpenDeviceOperator: the only device query the runtime can answer is
            // whether a card is inserted, and it serves SD from a host directory.
            400 => Reply::ok().object(Arc::new(DeviceOperator)),
            // Detection notifiers return an IEventNotifier. The events exist but
            // nothing in the runtime removes the card or updates system data.
            500 => Reply::ok().object(Arc::new(EventNotifier {
                key: "fs.sd-card-detect",
            })),
            501 => Reply::ok().object(Arc::new(EventNotifier {
                key: "fs.game-card-detect",
            })),
            510 => Reply::ok().object(Arc::new(EventNotifier {
                key: "fs.system-data-update",
            })),
            630 => {
                SD_ACCESSIBLE.store(req.u32(0) & 1 != 0, Ordering::Relaxed);
                Reply::ok()
            }
            631 => Reply::ok().u8(u8::from(SD_ACCESSIBLE.load(Ordering::Relaxed))),
            700 => Reply::ok().object(Arc::new(EventNotifier {
                key: "fs.access-failure",
            })),
            601 => Reply::ok().u64(req.u64(0).saturating_add(req.u64(8))),
            612 => {
                let used = load_saves(&saves_root(p.title_id))
                    .iter()
                    .map(|(dir, _)| tree_usage(&dir.join(SAVE_DATA)).unwrap_or(0))
                    .sum::<u64>();
                Reply::ok().u64(SAVE_SPACE.saturating_sub(used))
            }
            616 => {
                let space = req.data.first().copied().unwrap_or(1);
                let id = req.u64(8);
                match find_save(&saves_root(p.title_id), id, Some(space)) {
                    Some((_, meta)) => Reply::ok().u64(meta.commit_id),
                    None => Reply::err(TARGET_NOT_FOUND),
                }
            }
            1001 => {
                let size = req.u64(0);
                let journal = req.u64(8);
                if size != 0 {
                    DEFAULT_SIZE.store(size, Ordering::Relaxed);
                }
                if journal != 0 {
                    DEFAULT_JOURNAL.store(journal, Ordering::Relaxed);
                }
                Reply::ok()
            }
            1003 => {
                AUTO_CREATE.store(false, Ordering::Relaxed);
                Reply::ok()
            }
            1004 | 1006 => Reply::ok(),
            1005 => Reply::ok().u32(0),
            1011 => Reply::ok().u32(0).u32(1),
            _ => unimplemented_named("fsp-srv", proxy_command_name(req.cmd), c, req),
        }
    }
}

/// `nn::fssrv::sf::IEventNotifier`: the object `OpenSdCardDetectionEventNotifier`
/// and its siblings return. The event is created on first use and stays
/// non-signalled, because nothing in the runtime removes a card or updates the
/// system data partition.
struct EventNotifier {
    key: &'static str,
}

impl Service for EventNotifier {
    fn name(&self) -> String {
        "fs/IEventNotifier".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let key = self.key;
                let handle = *c
                    .process()
                    .service_state(key, || c.process().create_event(false));
                Reply::ok().copy_handle(handle)
            }
            _ => unimplemented_named(
                "fs/IEventNotifier",
                event_notifier_command_name(req.cmd),
                c,
                req,
            ),
        }
    }
}

fn event_notifier_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "GetEventHandle",
        _ => "unknown",
    }
}

/// A name table is not a command table: keeping it out of a `match` keeps the
/// coverage report from counting diagnostic names as handled commands.
fn command_name(table: &[(u32, &'static str)], cmd: u32) -> &'static str {
    table
        .iter()
        .find(|(id, _)| *id == cmd)
        .map_or("unknown", |(_, name)| name)
}

/// `nn::fssrv::sf::IDeviceOperator`. The runtime serves the SD card from a host
/// directory, so a card is always inserted. Everything else in the interface
/// reads CID/CSD registers or error logs from hardware the runtime does not
/// model, and stays unimplemented.
struct DeviceOperator;

impl Service for DeviceOperator {
    fn name(&self) -> String {
        "fs/IDeviceOperator".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u8(1),
            _ => unimplemented_named(
                "fs/IDeviceOperator",
                device_operator_command_name(req.cmd),
                c,
                req,
            ),
        }
    }
}

fn device_operator_command_name(cmd: u32) -> &'static str {
    command_name(
        &[
            (0, "IsSdCardInserted"),
            (1, "GetSdCardSpeedMode"),
            (2, "GetSdCardCid"),
            (3, "GetSdCardUserAreaSize"),
            (4, "GetSdCardProtectedAreaSize"),
            (5, "GetAndClearSdCardErrorInfo"),
            (100, "GetMmcCid"),
            (101, "GetMmcSpeedMode"),
            (111, "GetMmcPartitionSize"),
        ],
        cmd,
    )
}

/// `nn::fs::FileSystemAttribute` is 0xC0 bytes: thirteen "has value" flags, a
/// reserved gap, thirteen s32 limits and a reserved tail.
const FILE_SYSTEM_ATTRIBUTE_SIZE: usize = 0xc0;
/// Longest path or name the runtime accepts (`FsPath`, and `DirectoryEntry.name`).
const FS_MAX_PATH: u32 = 0x300;

fn file_system_attribute() -> [u8; FILE_SYSTEM_ATTRIBUTE_SIZE] {
    let mut out = [0u8; FILE_SYSTEM_ATTRIBUTE_SIZE];
    // Every limit below is meaningful for this filesystem.
    out[..13].fill(1);
    for i in 0..13 {
        put_u32(&mut out, 0x28 + i * 4, FS_MAX_PATH);
    }
    out
}

/// Name of an `fsp-srv` command for the "not implemented" diagnostic. Only the
/// commands the runtime does not answer need an entry; the rest use "unknown".
fn proxy_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "OpenFileSystem",
        3..=6 => "legacy file system opener",
        7 => "OpenFileSystemWithPatch",
        8 | 10 => "OpenFileSystemWithId",
        9 => "OpenDataFileSystemByProgramId",
        11 => "OpenBisFileSystem",
        12 => "OpenBisStorage",
        13 => "InvalidateBisCache",
        17 => "OpenHostFileSystem",
        19 => "FormatSdCardFileSystem",
        29 => "IsInitializationMode",
        30 => "OpenGameCardStorage",
        31 => "OpenGameCardFileSystem",
        36 => "OpenHostFileSystemWithOption",
        38 => "DetachSdCard",
        39 => "ShutdownStorage",
        40 => "FixSaveDataCertificate",
        54 => "ReadSaveDataFileSystemExtraData",
        55 | 56 => "save data extra data",
        63..=66 => "save data extra data",
        81 => "OpenSaveDataTransferManager",
        82 => "OpenSaveDataTransferManagerVersion2",
        83 => "OpenSaveDataTransferProhibiter",
        84 => "ListAccessibleSaveDataOwnerId",
        85 => "OpenSaveDataTransferManagerForSaveDataRepair",
        86 => "OpenSaveDataMover",
        87 => "OpenSaveDataTransferManagerForRepair",
        88 => "OpenSaveDataTransferManagerForSaveDataRepairWithPolicy",
        100 => "OpenImageDirectoryFileSystem",
        101 => "OpenBaseFileSystem",
        102 => "FormatBaseFileSystem",
        110 => "OpenContentStorageFileSystem",
        120 => "OpenCloudBackupWorkStorageFileSystem",
        130 => "OpenCustomStorageFileSystem",
        201 => "OpenDataStorageByProgramId",
        202 => "OpenDataStorageByDataId",
        203 => "OpenPatchDataStorageByCurrentProcess",
        204 => "OpenDataFileSystemWithProgramIndex",
        205 => "OpenDataStorageWithProgramIndex",
        206 => "OpenDataStorageByPath",
        207 => "OpenDataFileSystemByDataId",
        502 | 503 => "wake-up event notifier",
        511 => "NotifySystemDataUpdateEvent",
        520 => "SimulateDeviceDetectionEvent",
        602 => "VerifySaveDataFileSystem",
        603 => "CorruptSaveDataFileSystem",
        604 => "CreatePaddingFile",
        605 => "DeleteAllPaddingFiles",
        606 => "GetRightsId",
        607 => "RegisterExternalKey",
        608 | 617 => "UnregisterExternalKey",
        609 => "GetRightsIdByPath",
        610 => "GetRightsIdAndKeyGenerationByPath",
        611 => "SetCurrentPosixTimeWithTimeDifference",
        613 => "VerifySaveDataFileSystemBySaveDataSpaceId",
        614 => "CorruptSaveDataFileSystemBySaveDataSpaceId",
        615 => "QuerySaveDataInternalStorageTotalSize",
        618 => "GetProgramId",
        620 => "SetSdCardEncryptionSeed",
        701 | 702 | 710 | 720 => "access failure notifier",
        800 => "GetAndClearErrorInfo",
        810 => "RegisterProgramIndexMapInfo",
        820 => "GetContentStorageInfoIndex",
        1002 => "SetSaveDataRootPath",
        1007 | 1008 => "registered update partition",
        1009 => "GetAndClearMemoryReportInfo",
        1012 => "GetFsStackUsage",
        1013 => "UnsetSaveDataRootPath",
        1016 | 1017 => "access log",
        1018 | 1019 => "debug option",
        1020 => "OpenMemoryStressor",
        1021 => "GetFsVariant",
        1200 => "OpenMultiCommitManager",
        1300 => "OpenBisWiper",
        1400 => "NotifyErrorContextServiceReady",
        _ => "unknown",
    }
}

struct RomStorage;

impl Service for RomStorage {
    fn name(&self) -> String {
        "fs/IStorage(RomFS)".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let Some(romfs) = &p.romfs else {
            return Reply::err(TARGET_NOT_FOUND);
        };
        match req.cmd {
            0 => {
                let (offset, size) = (req.u64(0), req.u64(8));
                let Some(b) = req.b.first() else {
                    return Reply::err(OUT_OF_RANGE);
                };
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
            // The SDK's RomFsFileSystem::Initialize rejects a header extent past this size.
            4 => Reply::ok().u64(romfs.lock().unwrap().size()),
            _ => unimplemented_named("fs/IStorage(RomFS)", storage_command_name(req.cmd), c, req),
        }
    }
}

fn rom_path(c: &Cpu, x: Option<&Buf>) -> Option<String> {
    let b = x?;
    let raw = c.process().memory.read(b.addr, b.size.min(0x301))?;
    let s = String::from_utf8_lossy(raw.split(|&b| b == 0).next()?).into_owned();
    Some(
        s.split('/')
            .filter(|p| !p.is_empty() && *p != ".")
            .collect::<Vec<_>>()
            .join("/"),
    )
}

struct RomFileSystem {
    files: HashMap<String, (u64, u64)>,
    dirs: HashMap<String, Vec<(String, u8, u64)>>,
}

impl RomFileSystem {
    fn new(list: Vec<kirikae_loader::romfs::FileEntry>) -> RomFileSystem {
        let mut files = HashMap::new();
        let mut dirs: HashMap<String, Vec<(String, u8, u64)>> =
            HashMap::from([(String::new(), Vec::new())]);
        for f in list {
            let key = f.path.trim_start_matches('/').to_string();
            let mut parent = String::new();
            let parts: Vec<&str> = key.split('/').collect();
            for (i, part) in parts.iter().enumerate() {
                let path = if parent.is_empty() {
                    part.to_string()
                } else {
                    format!("{parent}/{part}")
                };
                if i + 1 == parts.len() {
                    dirs.entry(parent.clone()).or_default().push((
                        part.to_string(),
                        ENTRY_FILE,
                        f.size,
                    ));
                } else if !dirs.contains_key(&path) {
                    dirs.insert(path.clone(), Vec::new());
                    dirs.entry(parent.clone()).or_default().push((
                        part.to_string(),
                        ENTRY_DIRECTORY,
                        0,
                    ));
                }
                parent = path;
            }
            files.insert(key, (f.offset, f.size));
        }
        for entries in dirs.values_mut() {
            entries.sort();
        }
        RomFileSystem { files, dirs }
    }

    fn lookup(&self, req: &Request, path: &str) -> Reply {
        match req.cmd {
            7 => {
                if self.files.contains_key(path) {
                    Reply::ok().u32(ENTRY_FILE as u32)
                } else if self.dirs.contains_key(path) {
                    Reply::ok().u32(ENTRY_DIRECTORY as u32)
                } else {
                    Reply::err(PATH_NOT_FOUND)
                }
            }
            8 => match self.files.get(path) {
                Some(&(offset, size)) => Reply::ok().object(Arc::new(RomFile { offset, size })),
                None => Reply::err(PATH_NOT_FOUND),
            },
            _ => {
                let filter = req.u32(0);
                let Some(all) = self.dirs.get(path) else {
                    return Reply::err(PATH_NOT_FOUND);
                };
                let entries = all
                    .iter()
                    .filter(|(_, kind, _)| {
                        if *kind == ENTRY_DIRECTORY {
                            filter & 1 != 0
                        } else {
                            filter & 2 != 0
                        }
                    })
                    .cloned()
                    .collect();
                Reply::ok().object(Arc::new(HostDir {
                    entries,
                    next: Mutex::new(0),
                }))
            }
        }
    }
}

impl Service for RomFileSystem {
    fn name(&self) -> String {
        "fs/IFileSystem(RomFS)".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            7..=9 => match rom_path(c, req.x.first()) {
                Some(path) => self.lookup(req, &path),
                None => Reply::err(PATH_NOT_FOUND),
            },
            10 => Reply::ok(),
            11 => Reply::ok().u64(0),
            12 => Reply::ok().u64(
                c.process()
                    .romfs
                    .as_ref()
                    .map_or(0, |r| r.lock().unwrap().size()),
            ),
            14 => Reply::ok().bytes(&[0u8; 0x20]),
            16 => Reply::ok().bytes(&file_system_attribute()),
            _ => unimplemented_named(
                "fs/IFileSystem(RomFS)",
                file_system_command_name(req.cmd),
                c,
                req,
            ),
        }
    }
}

struct RomFile {
    offset: u64,
    size: u64,
}

impl Service for RomFile {
    fn name(&self) -> String {
        "fs/IFile(RomFS)".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let Some(romfs) = &p.romfs else {
            return Reply::err(TARGET_NOT_FOUND);
        };
        match req.cmd {
            0 => {
                let (offset, size) = (req.u64(8), req.u64(16));
                let Some(b) = req.b.first() else {
                    return Reply::err(OUT_OF_RANGE);
                };
                let len = size.min(b.size).min(self.size.saturating_sub(offset));
                let mut buf = vec![0u8; len as usize];
                match romfs
                    .lock()
                    .unwrap()
                    .read_at(self.offset + offset, &mut buf)
                {
                    Ok(n) => {
                        p.memory.write(b.addr, &buf[..n]);
                        Reply::ok().u64(n as u64)
                    }
                    Err(e) => {
                        p.warn_once(format!("RomFS read failed: {e}"));
                        Reply::err(OUT_OF_RANGE)
                    }
                }
            }
            2 => Reply::ok(),
            4 => Reply::ok().u64(self.size),
            _ => unimplemented_named("fs/IFile(RomFS)", file_command_name(req.cmd), c, req),
        }
    }
}

// ------------------------------------------------------------------ save data model

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SaveAttr {
    reference_id: u64,
    user_id: [u8; 16],
    system_id: u64,
    data_type: u8,
    rank: u8,
    index: u16,
}

impl SaveAttr {
    fn parse(data: &[u8], off: usize) -> Option<SaveAttr> {
        let raw = data.get(off..off + 0x24)?;
        let mut user_id = [0u8; 16];
        user_id.copy_from_slice(&raw[8..0x18]);
        Some(SaveAttr {
            reference_id: u64::from_le_bytes(raw[0..8].try_into().ok()?),
            user_id,
            system_id: u64::from_le_bytes(raw[0x18..0x20].try_into().ok()?),
            data_type: raw[0x20],
            rank: raw[0x21],
            index: u16::from_le_bytes(raw[0x22..0x24].try_into().ok()?),
        })
    }

    fn user_fold(&self) -> u64 {
        u64::from_le_bytes(self.user_id[..8].try_into().unwrap_or_default())
            ^ u64::from_le_bytes(self.user_id[8..].try_into().unwrap_or_default())
    }

    fn is_system(&self) -> bool {
        self.system_id != 0 || matches!(self.data_type, 0 | 6)
    }

    fn write(&self, out: &mut [u8]) {
        out.fill(0);
        put_u64(out, 0, self.reference_id);
        out[8..0x18].copy_from_slice(&self.user_id);
        put_u64(out, 0x18, self.system_id);
        out[0x20] = self.data_type;
        out[0x21] = self.rank;
        put_u16(out, 0x22, self.index);
    }
}

#[derive(Clone, Copy, Debug)]
struct SaveCreation {
    size: u64,
    journal_size: u64,
    block_size: u64,
    owner_id: u64,
    flags: u32,
    space: u8,
    /// `SaveDataMetaInfo.size` from the creation command. The meta file can grow
    /// up to this many bytes; 0 means "not specified" and the open path falls
    /// back to [`meta::META_DEFAULT_SIZE`].
    meta_size: u32,
}

impl Default for SaveCreation {
    fn default() -> SaveCreation {
        SaveCreation {
            size: DEFAULT_SIZE.load(Ordering::Relaxed),
            journal_size: DEFAULT_JOURNAL.load(Ordering::Relaxed),
            block_size: DEFAULT_BLOCK_SIZE,
            owner_id: 0,
            flags: 0,
            space: 1,
            meta_size: 0,
        }
    }
}

impl SaveCreation {
    fn parse(data: &[u8], off: usize) -> SaveCreation {
        let mut creation = SaveCreation::default();
        if let Some(size) = data.get(off..off + 8) {
            creation.size = u64::from_le_bytes(size.try_into().unwrap_or_default());
        }
        if let Some(journal) = data.get(off + 8..off + 16) {
            creation.journal_size = u64::from_le_bytes(journal.try_into().unwrap_or_default());
        }
        if let Some(block) = data.get(off + 0x10..off + 0x18) {
            creation.block_size = u64::from_le_bytes(block.try_into().unwrap_or_default());
        }
        if let Some(owner) = data.get(off + 0x18..off + 0x20) {
            creation.owner_id = u64::from_le_bytes(owner.try_into().unwrap_or_default());
        }
        if let Some(flags) = data.get(off + 0x20..off + 0x24) {
            creation.flags = u32::from_le_bytes(flags.try_into().unwrap_or_default());
        }
        if let Some(space) = data.get(off + 0x24) {
            creation.space = *space;
        }
        if creation.size == 0 {
            creation.size = DEFAULT_SIZE.load(Ordering::Relaxed);
        }
        if creation.block_size == 0 {
            creation.block_size = DEFAULT_BLOCK_SIZE;
        }
        creation
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SaveMeta {
    title: u64,
    space: u8,
    attr: SaveAttr,
    save_id: u64,
    size: u64,
    journal_size: u64,
    block_size: u64,
    owner_id: u64,
    flags: u32,
    timestamp: u64,
    commit_id: u64,
    generation: u64,
    meta_size: u32,
}

impl SaveMeta {
    fn new(title: u64, space: u8, attr: SaveAttr) -> SaveMeta {
        SaveMeta {
            title,
            space,
            attr,
            save_id: derive_save_id(title, space, attr),
            size: initial_size(attr.data_type),
            journal_size: DEFAULT_JOURNAL.load(Ordering::Relaxed),
            block_size: DEFAULT_BLOCK_SIZE,
            owner_id: 0,
            flags: 0,
            timestamp: now_secs(),
            commit_id: 0,
            generation: 0,
            meta_size: meta::META_DEFAULT_SIZE as u32,
        }
    }

    fn apply(&mut self, creation: &SaveCreation) {
        self.size = creation.size.max(1);
        self.journal_size = creation.journal_size;
        self.block_size = creation.block_size.max(1);
        self.owner_id = creation.owner_id;
        self.flags = creation.flags;
        if creation.meta_size != 0 {
            self.meta_size = creation.meta_size;
        }
    }

    fn load(dir: &Path) -> Option<SaveMeta> {
        let text = fs::read_to_string(dir.join(SAVE_META)).ok()?;
        let value: Value = serde_json::from_str(&text).ok()?;
        SaveMeta::from_json(&value)
    }

    fn save(&self, dir: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.to_json()).map_err(std::io::Error::other)?;
        write_atomic(&dir.join(SAVE_META), &dir.join(SAVE_META_TMP), &bytes)
    }

    fn to_json(&self) -> Value {
        json!({
            "version": 1,
            "title": self.title,
            "space": self.space,
            "save_id": self.save_id,
            "reference": self.attr.reference_id,
            "user": hex(&self.attr.user_id),
            "system": self.attr.system_id,
            "type": self.attr.data_type,
            "rank": self.attr.rank,
            "index": self.attr.index,
            "size": self.size,
            "journal": self.journal_size,
            "block": self.block_size,
            "owner": self.owner_id,
            "flags": self.flags,
            "timestamp": self.timestamp,
            "commit": self.commit_id,
            "generation": self.generation,
            "meta_size": self.meta_size,
        })
    }

    fn from_json(v: &Value) -> Option<SaveMeta> {
        if v.get("version")?.as_u64()? != 1 {
            return None;
        }
        let field = |name: &str| v.get(name)?.as_u64();
        let user = parse_hex(v.get("user")?.as_str()?)?;
        Some(SaveMeta {
            title: field("title")?,
            space: field("space")? as u8,
            attr: SaveAttr {
                reference_id: field("reference")?,
                user_id: user,
                system_id: field("system")?,
                data_type: field("type")? as u8,
                rank: field("rank")? as u8,
                index: field("index")? as u16,
            },
            save_id: field("save_id")?,
            size: field("size")?,
            journal_size: field("journal")?,
            block_size: field("block")?,
            owner_id: field("owner")?,
            flags: field("flags")? as u32,
            timestamp: field("timestamp")?,
            commit_id: field("commit")?,
            generation: field("generation")?,
            meta_size: field("meta_size")? as u32,
        })
    }

    fn extra_bytes(&self, used: u64) -> [u8; SAVE_EXTRA_SIZE] {
        let mut out = [0u8; SAVE_EXTRA_SIZE];
        self.attr.write(&mut out[..0x40]);
        put_u64(&mut out, 0x40, self.owner_id);
        put_u64(&mut out, 0x48, self.timestamp);
        put_u32(&mut out, 0x50, self.flags);
        put_u64(&mut out, 0x58, self.size.saturating_sub(used));
        put_u64(&mut out, 0x60, self.journal_size);
        put_u64(&mut out, 0x68, self.commit_id);
        out
    }

    fn info_bytes(&self) -> [u8; SAVE_INFO_SIZE] {
        let mut out = [0u8; SAVE_INFO_SIZE];
        put_u64(&mut out, 0, self.save_id);
        out[8] = self.space;
        out[9] = self.attr.data_type;
        out[0x10..0x20].copy_from_slice(&self.attr.user_id);
        put_u64(&mut out, 0x20, self.attr.system_id);
        put_u64(&mut out, 0x28, self.attr.reference_id);
        put_u64(&mut out, 0x30, self.size);
        put_u16(&mut out, 0x38, self.attr.index);
        out[0x3a] = self.attr.rank;
        out
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn valid_space(space: u8) -> bool {
    matches!(space, 0..=4 | 100..=102)
}

fn data_type_name(kind: u8) -> &'static str {
    match kind {
        0 => "system",
        1 => "account",
        2 => "bcat",
        3 => "device",
        4 => "temporary",
        5 => "cache",
        6 => "system-bcat",
        _ => "other",
    }
}

fn save_dir_name(attr: &SaveAttr) -> String {
    let kind = data_type_name(attr.data_type);
    let id = if attr.is_system() {
        attr.system_id
    } else if matches!(attr.data_type, DEVICE | TEMPORARY | CACHE) {
        attr.reference_id
    } else {
        attr.user_fold()
    };
    if attr.index == 0 {
        format!("{kind}-{id:016x}")
    } else {
        format!("{kind}-{id:016x}-{}", attr.index)
    }
}

fn hash_bytes(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn normalize_attr(mut attr: SaveAttr, title: u64) -> SaveAttr {
    // A zero reference id means "the application owning this FS session". System
    // save data has no application id at all.
    if attr.reference_id == 0 && !attr.is_system() {
        attr.reference_id = title;
    }
    // Device, temporary and cache images belong to the application and the
    // hardware, not to an account: Horizon's SaveDataAttribute is all-zero for
    // them except for the type (and the optional application id). Folding the
    // user fields here keeps one image per application even when a title passes
    // the current user out of habit.
    if matches!(attr.data_type, DEVICE | TEMPORARY | CACHE) {
        attr.user_id = [0; 16];
        attr.system_id = 0;
        attr.index = if attr.data_type == DEVICE {
            0
        } else {
            attr.index
        };
    }
    attr
}

fn initial_size(data_type: u8) -> u64 {
    match data_type {
        CACHE => cache::CACHE_STORAGE_SIZE,
        _ => DEFAULT_SIZE.load(Ordering::Relaxed),
    }
}

/// Total bytes reserved by the save images of `root`. The quota counts the data
/// image and its journal, which is what `GetTotalSpaceSize` reports too.
fn reserved(root: &Path) -> u64 {
    load_saves(root)
        .into_iter()
        .map(|(_, meta)| meta.size.saturating_add(meta.journal_size))
        .sum()
}

/// Checks that adding `size` bytes (or extending `existing` to `size`) still fits
/// in the title's save space.
fn check_quota(root: &Path, size: u64, existing: u64) -> Result<(), u32> {
    let used = reserved(root).saturating_sub(existing);
    if used.saturating_add(size) > SAVE_SPACE {
        return Err(USABLE_SPACE_NOT_ENOUGH);
    }
    Ok(())
}

fn valid_block_size(block: u64) -> bool {
    block >= MIN_BLOCK_SIZE && block.is_power_of_two()
}

fn derive_save_id(title: u64, space: u8, attr: SaveAttr) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325;
    hash_bytes(&mut h, &title.to_le_bytes());
    hash_bytes(&mut h, &[space, attr.data_type, attr.rank]);
    hash_bytes(&mut h, &attr.index.to_le_bytes());
    hash_bytes(&mut h, &attr.reference_id.to_le_bytes());
    hash_bytes(&mut h, &attr.system_id.to_le_bytes());
    hash_bytes(&mut h, &attr.user_id);
    h.max(1)
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn parse_hex(text: &str) -> Option<[u8; 16]> {
    if text.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

fn put_u16(out: &mut [u8], off: usize, value: u16) {
    out[off..off + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut [u8], off: usize, value: u32) {
    out[off..off + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut [u8], off: usize, value: u64) {
    out[off..off + 8].copy_from_slice(&value.to_le_bytes());
}

fn io_code(e: std::io::Error) -> u32 {
    match e.kind() {
        std::io::ErrorKind::NotFound => PATH_NOT_FOUND,
        std::io::ErrorKind::AlreadyExists => PATH_ALREADY_EXISTS,
        std::io::ErrorKind::PermissionDenied => TARGET_LOCKED,
        _ => PRECONDITION_VIOLATION,
    }
}

fn is_internal(name: &OsString) -> bool {
    name.to_str().is_some_and(|n| {
        matches!(
            n,
            SAVE_DATA
                | SAVE_PENDING
                | SAVE_PREV
                | SAVE_META
                | SAVE_META_TMP
                | SAVE_JOURNAL
                | SAVE_JOURNAL_TMP
                | meta::META_DIR
        )
    })
}

fn remove_dir_all_if_exists(path: &Path) -> std::io::Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

fn write_atomic(path: &Path, tmp: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = File::create(tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    fs::rename(tmp, path)?;
    if let Some(parent) = path.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}

fn sync_tree(path: &Path) -> std::io::Result<()> {
    let Ok(rd) = fs::read_dir(path) else {
        return Ok(());
    };
    for entry in rd {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_dir() {
            sync_tree(&entry.path())?;
            sync_dir(&entry.path())?;
        } else if ft.is_file() {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(entry.path())
                .or_else(|_| File::open(entry.path()))?;
            file.sync_all()?;
        }
    }
    Ok(())
}

fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    let Ok(rd) = fs::read_dir(from) else {
        return Ok(());
    };
    for entry in rd {
        let entry = entry?;
        let ft = entry.file_type()?;
        let dst = to.join(entry.file_name());
        if ft.is_dir() {
            copy_tree(&entry.path(), &dst)?;
        } else if ft.is_file() {
            fs::copy(entry.path(), &dst)?;
        }
    }
    Ok(())
}

fn tree_usage(path: &Path) -> std::io::Result<u64> {
    let Ok(rd) = fs::read_dir(path) else {
        return Ok(0);
    };
    let mut total = 0;
    for entry in rd {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_dir() {
            total += tree_usage(&entry.path())?;
        } else if ft.is_file() {
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

fn dir_has_files(dir: &Path) -> bool {
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    rd.flatten().any(|e| !is_internal(&e.file_name()))
}

fn ensure_data_layout(root: &Path) -> std::io::Result<PathBuf> {
    let data = root.join(SAVE_DATA);
    if data.exists() {
        return Ok(data);
    }
    fs::create_dir_all(&data)?;
    let Ok(rd) = fs::read_dir(root) else {
        return Ok(data);
    };
    for entry in rd {
        let entry = entry?;
        let name = entry.file_name();
        if is_internal(&name) {
            continue;
        }
        fs::rename(entry.path(), data.join(name))?;
    }
    Ok(data)
}

fn read_journal_generation(root: &Path) -> Option<u64> {
    let text = fs::read_to_string(root.join(SAVE_JOURNAL)).ok()?;
    text.lines()
        .find_map(|line| line.trim().strip_prefix("generation "))
        .and_then(|v| v.trim().parse().ok())
}

fn write_journal(root: &Path, generation: u64) -> std::io::Result<()> {
    let text = format!("kirikae-commit 1\ngeneration {generation}\n");
    write_atomic(
        &root.join(SAVE_JOURNAL),
        &root.join(SAVE_JOURNAL_TMP),
        text.as_bytes(),
    )
}

/// Puts a save directory back into a state that only contains committed data.
///
/// A pending tree is never promoted automatically: the guest never saw the commit
/// reply, so dropping it is safer than keeping writes it does not know about.
fn recover_save(root: &Path) -> std::io::Result<()> {
    let data = root.join(SAVE_DATA);
    let prev = root.join(SAVE_PREV);
    let pending = root.join(SAVE_PENDING);
    let journal = root.join(SAVE_JOURNAL);

    if journal.exists() {
        let generation = read_journal_generation(root);
        let committed = SaveMeta::load(root).map_or(0, |m| m.generation);
        let finished = data.exists()
            && generation.is_some_and(|generation| generation != 0 && committed >= generation);
        if prev.exists() {
            if finished {
                remove_dir_all_if_exists(&prev)?;
            } else {
                remove_dir_all_if_exists(&data)?;
                fs::rename(&prev, &data)?;
            }
        }
        remove_dir_all_if_exists(&pending)?;
        let _ = fs::remove_file(&journal);
    } else {
        if !data.exists() && prev.exists() {
            fs::rename(&prev, &data)?;
        } else {
            remove_dir_all_if_exists(&prev)?;
        }
        remove_dir_all_if_exists(&pending)?;
    }
    Ok(())
}

fn load_saves(root: &Path) -> Vec<(PathBuf, SaveMeta)> {
    let Ok(rd) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut saves: Vec<(PathBuf, SaveMeta)> = rd
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|ft| ft.is_dir()))
        .filter_map(|entry| {
            let dir = entry.path();
            let meta = SaveMeta::load(&dir)?;
            Some((dir, meta))
        })
        .collect();
    saves.sort_by(|a, b| a.0.cmp(&b.0));
    saves
}

fn find_save(root: &Path, id: u64, space: Option<u8>) -> Option<(PathBuf, SaveMeta)> {
    load_saves(root)
        .into_iter()
        .find(|(_, meta)| meta.save_id == id && space.is_none_or(|space| meta.space == space))
}

fn find_save_by_attr(root: &Path, attr: SaveAttr) -> Option<(PathBuf, SaveMeta)> {
    let dir = root.join(save_dir_name(&attr));
    if let Some(meta) = SaveMeta::load(&dir)
        && meta.attr == attr
    {
        return Some((dir, meta));
    }
    load_saves(root)
        .into_iter()
        .find(|(_, meta)| meta.attr == attr)
}

// ------------------------------------------------------------------ save journals

struct SaveState {
    live: PathBuf,
}

struct SaveData {
    root: PathBuf,
    data: PathBuf,
    pending: PathBuf,
    prev: PathBuf,
    meta: Mutex<SaveMeta>,
    state: Mutex<SaveState>,
    commit: Mutex<()>,
    read_only: bool,
}

impl SaveData {
    fn mount(
        root: PathBuf,
        title: u64,
        space: u8,
        attr: SaveAttr,
        read_only: bool,
        allow_create: bool,
    ) -> Result<Arc<SaveData>, u32> {
        if !valid_space(space) || attr.data_type > 6 {
            return Err(INVALID_ARGUMENT);
        }
        let (dir, meta) = ensure_save(&root, title, space, attr, allow_create && !read_only)?;
        let data = dir.join(SAVE_DATA);
        let pending = dir.join(SAVE_PENDING);
        fs::create_dir_all(&data).map_err(io_code)?;
        if !read_only {
            remove_dir_all_if_exists(&pending).map_err(io_code)?;
        }
        Ok(Arc::new(SaveData {
            prev: dir.join(SAVE_PREV),
            root: dir,
            data: data.clone(),
            pending,
            meta: Mutex::new(meta),
            state: Mutex::new(SaveState { live: data }),
            commit: Mutex::new(()),
            read_only,
        }))
    }

    fn live_root(&self) -> PathBuf {
        lock(&self.state).live.clone()
    }

    #[cfg(test)]
    fn meta(&self) -> SaveMeta {
        lock(&self.meta).clone()
    }

    fn limit(&self) -> Option<u64> {
        (!self.read_only).then(|| lock(&self.meta).size)
    }

    fn space(&self) -> (u64, u64) {
        let used = tree_usage(&self.live_root()).unwrap_or(0);
        let meta = lock(&self.meta);
        (
            meta.size.saturating_sub(used),
            meta.size.saturating_add(meta.journal_size),
        )
    }

    fn ensure_pending(&self) -> Result<(), u32> {
        if self.read_only {
            return Err(READ_ONLY_FILE_SYSTEM);
        }
        let _commit = lock(&self.commit);
        let mut state = lock(&self.state);
        if state.live == self.pending {
            return Ok(());
        }
        remove_dir_all_if_exists(&self.pending).map_err(io_code)?;
        copy_tree(&self.data, &self.pending).map_err(io_code)?;
        state.live = self.pending.clone();
        Ok(())
    }

    fn commit(&self) -> Result<(), u32> {
        if self.read_only {
            return Err(READ_ONLY_FILE_SYSTEM);
        }
        let _commit = lock(&self.commit);
        let mut state = lock(&self.state);
        if state.live != self.pending || !self.pending.exists() {
            return Ok(());
        }

        let used = tree_usage(&self.pending).map_err(io_code)?;
        let mut meta = lock(&self.meta);
        if used > meta.size {
            return Err(USABLE_SPACE_NOT_ENOUGH);
        }
        sync_tree(&self.pending).map_err(io_code)?;

        let generation = meta.generation.wrapping_add(1).max(1);
        write_journal(&self.root, generation).map_err(io_code)?;
        remove_dir_all_if_exists(&self.prev).map_err(io_code)?;
        if self.data.exists() {
            fs::rename(&self.data, &self.prev).map_err(io_code)?;
        }
        if let Err(e) = fs::rename(&self.pending, &self.data) {
            let _ = fs::rename(&self.prev, &self.data);
            let _ = fs::remove_file(self.root.join(SAVE_JOURNAL));
            return Err(io_code(e));
        }

        meta.generation = generation;
        meta.commit_id = generation;
        meta.timestamp = now_secs();
        if let Err(e) = meta.save(&self.root) {
            let _ = fs::remove_dir_all(&self.data);
            let _ = fs::rename(&self.prev, &self.data);
            let _ = fs::remove_file(self.root.join(SAVE_JOURNAL));
            return Err(io_code(e));
        }

        let _ = fs::remove_dir_all(&self.prev);
        let _ = fs::remove_file(self.root.join(SAVE_JOURNAL));
        state.live = self.data.clone();
        Ok(())
    }

    #[cfg(test)]
    fn extra_bytes(&self) -> [u8; SAVE_EXTRA_SIZE] {
        let used = tree_usage(&self.live_root()).unwrap_or(0);
        lock(&self.meta).extra_bytes(used)
    }
}

fn ensure_save(
    root: &Path,
    title: u64,
    space: u8,
    attr: SaveAttr,
    allow_create: bool,
) -> Result<(PathBuf, SaveMeta), u32> {
    let attr = normalize_attr(attr, title);
    if !valid_space(space) || attr.data_type > 6 {
        return Err(INVALID_ARGUMENT);
    }
    if attr.data_type == CACHE && !cache::valid_index(attr.index) {
        return Err(INVALID_ARGUMENT);
    }
    let dir = root.join(save_dir_name(&attr));
    if dir.exists() {
        recover_save(&dir).map_err(io_code)?;
    }
    if let Some(meta) = SaveMeta::load(&dir) {
        return Ok((dir, meta));
    }
    if !dir.exists() && !allow_create {
        return Err(TARGET_NOT_FOUND);
    }
    if !dir.join(SAVE_DATA).exists() && !dir_has_files(&dir) && !allow_create {
        return Err(TARGET_NOT_FOUND);
    }
    fs::create_dir_all(&dir).map_err(io_code)?;
    recover_save(&dir).map_err(io_code)?;
    let data = ensure_data_layout(&dir).map_err(io_code)?;
    let used = tree_usage(&data).map_err(io_code)?;
    let mut meta = SaveMeta::new(title, space, attr);
    if attr.data_type == CACHE {
        meta.size = cache::default_size(root);
        if meta.size == 0 {
            return Err(USABLE_SPACE_NOT_ENOUGH);
        }
        cache::check_budget(root, meta.size, 0)?;
    }
    check_quota(root, meta.size.saturating_add(meta.journal_size), 0)?;
    meta.size = meta.size.max(used.max(1));
    meta.save(&dir).map_err(io_code)?;
    sync_dir(&dir).map_err(io_code)?;
    Ok((dir, meta))
}

fn create_save(
    root: &Path,
    title: u64,
    space: u8,
    attr: SaveAttr,
    creation: SaveCreation,
) -> Result<(), u32> {
    let attr = normalize_attr(attr, title);
    if !valid_space(space) || attr.data_type > 6 {
        return Err(INVALID_ARGUMENT);
    }
    if attr.data_type == CACHE && !cache::valid_index(attr.index) {
        return Err(INVALID_ARGUMENT);
    }
    if !valid_block_size(creation.block_size) {
        return Err(INVALID_ARGUMENT);
    }
    let mut creation = creation;
    if creation.size == 0 {
        creation.size = initial_size(attr.data_type);
    }
    if attr.data_type == CACHE {
        cache::check_budget(root, creation.size, 0)?;
    }
    check_quota(root, creation.size.saturating_add(creation.journal_size), 0)?;
    let dir = root.join(save_dir_name(&attr));
    if SaveMeta::load(&dir).is_some() {
        return Err(PATH_ALREADY_EXISTS);
    }
    fs::create_dir_all(&dir).map_err(io_code)?;
    recover_save(&dir).map_err(io_code)?;
    let data = ensure_data_layout(&dir).map_err(io_code)?;
    let used = tree_usage(&data).map_err(io_code)?;
    let mut meta = SaveMeta::new(title, space, attr);
    meta.apply(&creation);
    meta.size = meta.size.max(used.max(1));
    meta.save(&dir).map_err(io_code)?;
    sync_dir(&dir).map_err(io_code)
}

fn delete_save(root: &Path, id: u64, space: Option<u8>) -> Result<(), u32> {
    match find_save(root, id, space) {
        Some((dir, _)) => fs::remove_dir_all(dir).map_err(io_code),
        None => Err(TARGET_NOT_FOUND),
    }
}

fn delete_save_by_attr(root: &Path, space: u8, attr: SaveAttr) -> Result<(), u32> {
    let Some((dir, meta)) = find_save_by_attr(root, attr) else {
        return Err(TARGET_NOT_FOUND);
    };
    if meta.space != space {
        return Err(TARGET_NOT_FOUND);
    }
    fs::remove_dir_all(dir).map_err(io_code)
}

fn extend_save(root: &Path, space: u8, id: u64, size: u64, journal: u64) -> Result<(), u32> {
    let Some((dir, mut meta)) = find_save(root, id, Some(space)) else {
        return Err(TARGET_NOT_FOUND);
    };
    let used = tree_usage(&dir.join(SAVE_DATA)).unwrap_or(0);
    let old = meta.size.saturating_add(meta.journal_size);
    if size != 0 {
        if size < used {
            return Err(USABLE_SPACE_NOT_ENOUGH);
        }
        if meta.attr.data_type == CACHE {
            cache::check_budget(root, size, meta.size)?;
        }
        meta.size = size;
    }
    if journal != 0 {
        meta.journal_size = journal;
    }
    let new = meta.size.saturating_add(meta.journal_size);
    check_quota(root, new, old)?;
    meta.save(&dir).map_err(io_code)
}

fn read_save_extra(c: &mut Cpu, req: &Request, root: &Path, id: u64, space: Option<u8>) -> Reply {
    let Some((dir, meta)) = find_save(root, id, space) else {
        return Reply::err(TARGET_NOT_FOUND);
    };
    let Some(b) = req.out_buf(0) else {
        return Reply::err(INVALID_ARGUMENT);
    };
    let used = tree_usage(&dir.join(SAVE_DATA)).unwrap_or(0);
    let extra = meta.extra_bytes(used);
    let n = (b.size as usize).min(extra.len());
    c.process().memory.write(b.addr, &extra[..n]);
    Reply::ok()
}

fn write_save_extra(c: &mut Cpu, req: &Request, root: &Path, id: u64, space: Option<u8>) -> Reply {
    let Some((dir, mut meta)) = find_save(root, id, space) else {
        return Reply::err(TARGET_NOT_FOUND);
    };
    let Some(a) = req.in_buf(0) else {
        return Reply::err(INVALID_ARGUMENT);
    };
    let raw = c
        .process()
        .memory
        .read(a.addr, a.size.min(SAVE_EXTRA_SIZE as u64))
        .unwrap_or_default();
    if raw.len() < 0x54 {
        return Reply::err(INVALID_ARGUMENT);
    }
    meta.timestamp = u64::from_le_bytes(raw[0x48..0x50].try_into().unwrap_or_default());
    meta.flags = u32::from_le_bytes(raw[0x50..0x54].try_into().unwrap_or_default());
    match meta.save(&dir) {
        Ok(()) => Reply::ok(),
        Err(e) => Reply::err(io_code(e)),
    }
}

fn read_extra_by_attr(c: &mut Cpu, req: &Request, root: &Path, space: u8, attr: SaveAttr) -> Reply {
    let Some((dir, meta)) = find_save_by_attr(root, attr) else {
        return Reply::err(TARGET_NOT_FOUND);
    };
    if meta.space != space {
        return Reply::err(TARGET_NOT_FOUND);
    }
    let Some(b) = req.out_buf(0) else {
        return Reply::err(INVALID_ARGUMENT);
    };
    let used = tree_usage(&dir.join(SAVE_DATA)).unwrap_or(0);
    let extra = meta.extra_bytes(used);
    let n = (b.size as usize).min(extra.len());
    c.process().memory.write(b.addr, &extra[..n]);
    Reply::ok()
}

fn write_extra_by_attr(
    c: &mut Cpu,
    req: &Request,
    root: &Path,
    space: u8,
    attr: SaveAttr,
) -> Reply {
    let Some((dir, mut meta)) = find_save_by_attr(root, attr) else {
        return Reply::err(TARGET_NOT_FOUND);
    };
    if meta.space != space {
        return Reply::err(TARGET_NOT_FOUND);
    }
    let Some(a) = req.in_buf(0) else {
        return Reply::err(INVALID_ARGUMENT);
    };
    let raw = c
        .process()
        .memory
        .read(a.addr, a.size.min(SAVE_EXTRA_SIZE as u64))
        .unwrap_or_default();
    if raw.len() >= 0x54 {
        meta.timestamp = u64::from_le_bytes(raw[0x48..0x50].try_into().unwrap_or_default());
        meta.flags = u32::from_le_bytes(raw[0x50..0x54].try_into().unwrap_or_default());
    }
    match meta.save(&dir) {
        Ok(()) => Reply::ok(),
        Err(e) => Reply::err(io_code(e)),
    }
}

// ------------------------------------------------------------------ ISaveDataInfoReader

#[derive(Clone, Copy)]
struct SaveDataFilter {
    filter_application: bool,
    filter_type: bool,
    filter_user: bool,
    filter_system: bool,
    filter_index: bool,
    attr: SaveAttr,
}

impl SaveDataFilter {
    fn parse(data: &[u8], off: usize) -> Option<SaveDataFilter> {
        let raw = data.get(off..off + 0x48)?;
        Some(SaveDataFilter {
            filter_application: raw[0] != 0,
            filter_type: raw[1] != 0,
            filter_user: raw[2] != 0,
            filter_system: raw[3] != 0,
            filter_index: raw[4] != 0,
            attr: SaveAttr::parse(raw, 8)?,
        })
    }

    fn matches(&self, meta: &SaveMeta) -> bool {
        (!self.filter_application || meta.attr.reference_id == self.attr.reference_id)
            && (!self.filter_type || meta.attr.data_type == self.attr.data_type)
            && (!self.filter_user || meta.attr.user_id == self.attr.user_id)
            && (!self.filter_system || meta.attr.system_id == self.attr.system_id)
            && (!self.filter_index || meta.attr.index == self.attr.index)
    }
}

#[derive(Clone, Copy)]
enum InfoFilter {
    All,
    Space(u8),
    Cache,
    Filter(SaveDataFilter),
}

impl InfoFilter {
    fn matches(&self, meta: &SaveMeta) -> bool {
        match self {
            InfoFilter::All => true,
            InfoFilter::Space(space) => meta.space == *space,
            InfoFilter::Cache => meta.attr.data_type == 5,
            InfoFilter::Filter(filter) => filter.matches(meta),
        }
    }
}

fn collect_saves(root: &Path, filter: InfoFilter) -> Vec<SaveMeta> {
    load_saves(root)
        .into_iter()
        .map(|(_, meta)| meta)
        .filter(|meta| filter.matches(meta))
        .collect()
}

struct SaveDataInfoReader {
    entries: Vec<SaveMeta>,
    next: Mutex<usize>,
}

impl SaveDataInfoReader {
    fn new(root: PathBuf, filter: InfoFilter) -> SaveDataInfoReader {
        SaveDataInfoReader {
            entries: collect_saves(&root, filter),
            next: Mutex::new(0),
        }
    }
}

impl Service for SaveDataInfoReader {
    fn name(&self) -> String {
        "fs/ISaveDataInfoReader".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let Some(b) = req.b.first().or_else(|| req.c.first()) else {
                    return Reply::ok().u64(0);
                };
                let room = b.size as usize / SAVE_INFO_SIZE;
                let mut next = lock(&self.next);
                let batch = &self.entries[(*next).min(self.entries.len())..];
                let n = batch.len().min(room);
                let mut out = vec![0u8; n * SAVE_INFO_SIZE];
                for (i, meta) in batch[..n].iter().enumerate() {
                    out[i * SAVE_INFO_SIZE..(i + 1) * SAVE_INFO_SIZE]
                        .copy_from_slice(&meta.info_bytes());
                }
                c.process().memory.write(b.addr, &out);
                *next += n;
                Reply::ok().u64(n as u64)
            }
            _ => unimplemented_named(
                "fs/ISaveDataInfoReader",
                info_reader_command_name(req.cmd),
                c,
                req,
            ),
        }
    }
}

fn find_save_with_filter(
    c: &mut Cpu,
    req: &Request,
    root: &Path,
    space: u8,
    filter: SaveDataFilter,
) -> Reply {
    let hit = collect_saves(root, InfoFilter::Filter(filter))
        .into_iter()
        .find(|meta| meta.space == space);
    match hit {
        Some(meta) => {
            if let Some(b) = req.out_buf(0) {
                let info = meta.info_bytes();
                let n = (b.size as usize).min(info.len());
                c.process().memory.write(b.addr, &info[..n]);
            }
            Reply::ok().u64(1)
        }
        None => Reply::ok().u64(0),
    }
}

// ------------------------------------------------------------------ host filesystem

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

fn host_command(
    root: &Path,
    c: &mut Cpu,
    req: &Request,
    limit: Option<u64>,
    read_only: bool,
) -> Reply {
    let path = |k: usize| host_path(c, root, req.x.get(k));
    let Some(p0) = path(0) else {
        return match req.cmd {
            10 => Reply::ok(),
            _ => Reply::err(PATH_NOT_FOUND),
        };
    };
    match req.cmd {
        0 => {
            if read_only {
                return Reply::err(READ_ONLY_FILE_SYSTEM);
            }
            if limit.is_some_and(|limit| req.u64(8) > limit) {
                return Reply::err(USABLE_SPACE_NOT_ENOUGH);
            }
            if p0.exists() {
                return Reply::err(PATH_ALREADY_EXISTS);
            }
            io_result(File::create(&p0).and_then(|f| f.set_len(req.u64(8))))
        }
        1 => io_result(fs::remove_file(&p0)),
        2 => {
            if read_only {
                return Reply::err(READ_ONLY_FILE_SYSTEM);
            }
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
        7 => match fs::metadata(&p0) {
            Ok(m) => Reply::ok().u32(if m.is_dir() {
                ENTRY_DIRECTORY
            } else {
                ENTRY_FILE
            } as u32),
            Err(_) => Reply::err(PATH_NOT_FOUND),
        },
        8 => {
            let mode = req.u32(0);
            let write = mode & 6 != 0;
            if write && read_only {
                return Reply::err(READ_ONLY_FILE_SYSTEM);
            }
            match OpenOptions::new().read(true).write(write).open(&p0) {
                Ok(f) => Reply::ok().object(Arc::new(HostFile {
                    file: Mutex::new(f),
                    append: mode & 4 != 0,
                    limit,
                    read_only,
                })),
                Err(_) => Reply::err(PATH_NOT_FOUND),
            }
        }
        9 => {
            let filter = req.u32(0);
            let Ok(rd) = fs::read_dir(&p0) else {
                return Reply::err(PATH_NOT_FOUND);
            };
            let mut entries: Vec<(String, u8, u64)> = rd
                .flatten()
                .filter_map(|e| {
                    let m = e.metadata().ok()?;
                    let kind = if m.is_dir() {
                        ENTRY_DIRECTORY
                    } else {
                        ENTRY_FILE
                    };
                    let wanted = if kind == ENTRY_DIRECTORY {
                        filter & 1 != 0
                    } else {
                        filter & 2 != 0
                    };
                    wanted.then(|| (e.file_name().to_string_lossy().into_owned(), kind, m.len()))
                })
                .collect();
            entries.sort();
            Reply::ok().object(Arc::new(HostDir {
                entries,
                next: Mutex::new(0),
            }))
        }
        10 => Reply::ok(),
        11 | 12 => Reply::ok().u64(SAVE_SPACE),
        13 => {
            let r = fs::read_dir(&p0).map(|rd| {
                for e in rd.flatten() {
                    let path = e.path();
                    let _ = if path.is_dir() {
                        fs::remove_dir_all(&path)
                    } else {
                        fs::remove_file(&path)
                    };
                }
            });
            io_result(r)
        }
        14 => match fs::metadata(&p0) {
            // `FileTimeStampRaw`: created, modified, accessed and a validity
            // byte. The host keeps one modification time here, so all three
            // report it and the entry is marked valid.
            Ok(m) => {
                let modified = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                let mut out = [0u8; 0x20];
                put_u64(&mut out, 0, modified);
                put_u64(&mut out, 8, modified);
                put_u64(&mut out, 0x10, modified);
                out[0x18] = 1;
                Reply::ok().bytes(&out)
            }
            Err(_) => Reply::err(PATH_NOT_FOUND),
        },
        15 => Reply::ok(),
        _ => unimplemented_named("fs/IFileSystem", file_system_command_name(req.cmd), c, req),
    }
}

impl Service for HostFs {
    fn name(&self) -> String {
        "fs/IFileSystem".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // GetFileSystemAttribute describes the filesystem, so it takes no
            // path and does not go through the path-guarded host commands.
            16 => Reply::ok().bytes(&file_system_attribute()),
            _ => host_command(&self.root, c, req, None, false),
        }
    }
}

struct SaveFs {
    save: Arc<SaveData>,
}

fn is_write_command(req: &Request) -> bool {
    match req.cmd {
        0..=6 | 13 => true,
        8 => req.u32(0) & 6 != 0,
        _ => false,
    }
}

impl Service for SaveFs {
    fn name(&self) -> String {
        "fs/IFileSystem(save)".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // A save image has the same name and path limits as a host folder.
            16 => Reply::ok().bytes(&file_system_attribute()),
            10 => match self.save.commit() {
                Ok(()) => Reply::ok(),
                Err(code) => {
                    c.process()
                        .warn_once(format!("fs: save commit failed with {code:#x}"));
                    Reply::err(code)
                }
            },
            11 | 12 => {
                let (free, total) = self.save.space();
                Reply::ok().u64(if req.cmd == 11 { free } else { total })
            }
            _ => {
                if is_write_command(req)
                    && let Err(code) = self.save.ensure_pending()
                {
                    return Reply::err(code);
                }
                host_command(
                    &self.save.live_root(),
                    c,
                    req,
                    self.save.limit(),
                    self.save.read_only,
                )
            }
        }
    }
}

struct HostFile {
    file: Mutex<File>,
    append: bool,
    limit: Option<u64>,
    read_only: bool,
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
                let Some(b) = req.b.first() else {
                    return Reply::err(OUT_OF_RANGE);
                };
                let mut buf = vec![0u8; size.min(b.size) as usize];
                let n = f
                    .seek(SeekFrom::Start(offset))
                    .and_then(|_| read_full(&mut f, &mut buf))
                    .unwrap_or(0);
                p.memory.write(b.addr, &buf[..n]);
                Reply::ok().u64(n as u64)
            }
            1 => {
                if self.read_only {
                    return Reply::err(READ_ONLY_FILE);
                }
                let (offset, size) = (req.u64(8), req.u64(16));
                let Some(a) = req.a.first() else {
                    return Reply::err(OUT_OF_RANGE);
                };
                let data = p.memory.read(a.addr, size.min(a.size)).unwrap_or_default();
                let at = if self.append {
                    f.seek(SeekFrom::End(0)).unwrap_or(0)
                } else {
                    offset
                };
                // Without the append bit a write may not extend the file; the
                // documented result is 0x307202. Append mode writes at the end.
                if !self.append {
                    let len = f.metadata().map_or(0, |m| m.len());
                    if at.saturating_add(data.len() as u64) > len {
                        return Reply::err(WRITE_BEYOND_END);
                    }
                }
                if self
                    .limit
                    .is_some_and(|limit| at.saturating_add(data.len() as u64) > limit)
                {
                    return Reply::err(USABLE_SPACE_NOT_ENOUGH);
                }
                io_result(f.seek(SeekFrom::Start(at)).and_then(|_| f.write_all(&data)))
            }
            2 => io_result(f.flush()),
            3 => {
                if self.read_only {
                    return Reply::err(READ_ONLY_FILE);
                }
                let size = req.u64(0);
                if self.limit.is_some_and(|limit| size > limit) {
                    return Reply::err(USABLE_SPACE_NOT_ENOUGH);
                }
                io_result(f.set_len(size))
            }
            4 => Reply::ok().u64(f.metadata().map_or(0, |m| m.len())),
            _ => unimplemented_named("fs/IFile", file_command_name(req.cmd), c, req),
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

fn storage_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "Read",
        1 => "Write",
        2 => "Flush",
        3 => "SetSize",
        4 => "GetSize",
        5 => "OperateRange",
        _ => "unknown",
    }
}

fn file_system_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "CreateFile",
        1 => "DeleteFile",
        2 => "CreateDirectory",
        3 => "DeleteDirectory",
        4 => "DeleteDirectoryRecursively",
        5 => "RenameFile",
        6 => "RenameDirectory",
        7 => "GetEntryType",
        8 => "OpenFile",
        9 => "OpenDirectory",
        10 => "Commit",
        11 => "GetFreeSpaceSize",
        12 => "GetTotalSpaceSize",
        13 => "CleanDirectoryRecursively",
        14 => "GetFileTimeStampRaw",
        15 => "QueryEntry",
        16 => "GetFileSystemAttribute",
        _ => "unknown",
    }
}

fn file_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "Read",
        1 => "Write",
        2 => "Flush",
        3 => "SetSize",
        4 => "GetSize",
        5 => "OperateRange",
        _ => "unknown",
    }
}

fn directory_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "Read",
        1 => "GetEntryCount",
        _ => "unknown",
    }
}

fn info_reader_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "ReadSaveDataInfo",
        _ => "unknown",
    }
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
                let Some(b) = req.b.first() else {
                    return Reply::ok().u64(0);
                };
                let room = b.size as usize / DIRECTORY_ENTRY_SIZE;
                let mut next = self.next.lock().unwrap();
                let batch = &self.entries[(*next).min(self.entries.len())..];
                let n = batch.len().min(room);
                let mut out = vec![0u8; n * DIRECTORY_ENTRY_SIZE];
                for (i, (name, kind, size)) in batch[..n].iter().enumerate() {
                    let e = &mut out[i * DIRECTORY_ENTRY_SIZE..(i + 1) * DIRECTORY_ENTRY_SIZE];
                    let bytes = name.as_bytes();
                    e[..bytes.len().min(0x300)].copy_from_slice(&bytes[..bytes.len().min(0x300)]);
                    // 0x301 is the attribute byte: bit 0 marks a directory. The
                    // archive bit (bit 1) is a FAT concept the runtime does not track.
                    e[0x301] = u8::from(*kind == ENTRY_DIRECTORY);
                    e[0x304] = *kind;
                    e[0x308..0x310].copy_from_slice(&size.to_le_bytes());
                }
                c.process().memory.write(b.addr, &out);
                *next += n;
                Reply::ok().u64(n as u64)
            }
            1 => Reply::ok().u64(self.entries.len() as u64),
            _ => unimplemented_named("fs/IDirectory", directory_command_name(req.cmd), c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kirikae_loader::romfs;

    fn temp_root(tag: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("kirikae-fs-{tag}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn user_id(user: u64) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&user.to_le_bytes());
        id
    }

    fn attr(user: u64, data_type: u8) -> SaveAttr {
        SaveAttr {
            reference_id: 0x0100_0000_0001_0000,
            user_id: user_id(user),
            system_id: 0,
            data_type,
            rank: 0,
            index: 0,
        }
    }

    fn write_pending(save: &SaveData, rel: &str, data: &[u8]) {
        save.ensure_pending().unwrap();
        let path = save.live_root().join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, data).unwrap();
    }

    #[test]
    fn write_commit_and_reopen_keeps_the_save() {
        let root = temp_root("commit");
        let a = attr(0x1122_3344_5566_7788, 1);
        let save = SaveData::mount(root.clone(), 0x0100_0000_0001_0000, 1, a, false, true).unwrap();
        let before = save.meta();
        assert_ne!(before.save_id, 0);
        assert!(save.data.is_dir());

        write_pending(&save, "slot1/data.bin", b"pending payload");
        assert_eq!(
            fs::read(save.live_root().join("slot1/data.bin")).unwrap(),
            b"pending payload"
        );
        assert!(!save.data.join("slot1/data.bin").exists());

        save.commit().unwrap();
        let after = save.meta();
        assert_eq!(after.commit_id, before.commit_id + 1);
        assert_eq!(after.generation, before.generation + 1);
        assert_eq!(
            fs::read(save.data.join("slot1/data.bin")).unwrap(),
            b"pending payload"
        );
        assert!(!save.pending.exists());
        drop(save);

        let reopened =
            SaveData::mount(root.clone(), 0x0100_0000_0001_0000, 1, a, false, true).unwrap();
        assert_eq!(
            fs::read(reopened.live_root().join("slot1/data.bin")).unwrap(),
            b"pending payload"
        );
        assert_eq!(reopened.meta().commit_id, after.commit_id);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_pending_tree_is_dropped_on_the_next_mount() {
        let root = temp_root("rollback");
        let a = attr(0xabcdef, 1);
        let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
        write_pending(&save, "save.bin", b"committed");
        save.commit().unwrap();
        let dir = save.root.clone();
        drop(save);

        // A crash leaves the pending tree behind: the committed file must win.
        fs::create_dir_all(dir.join(SAVE_PENDING)).unwrap();
        fs::write(dir.join(SAVE_PENDING).join("save.bin"), b"uncommitted").unwrap();
        fs::write(dir.join(SAVE_PENDING).join("new.bin"), b"new").unwrap();

        let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
        assert_eq!(
            fs::read(save.live_root().join("save.bin")).unwrap(),
            b"committed"
        );
        assert!(!save.live_root().join("new.bin").exists());
        assert!(!dir.join(SAVE_PENDING).exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recovery_rolls_back_a_commit_that_never_wrote_its_metadata() {
        let root = temp_root("journal-rollback");
        let a = attr(0x5150, 1);
        let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
        write_pending(&save, "save.bin", b"v1");
        save.commit().unwrap();
        let dir = save.root.clone();
        let generation = save.meta().generation;
        drop(save);

        // Crash after data -> data.prev but before pending -> data, with the
        // commit journal already on disk and the metadata still at generation N.
        fs::create_dir_all(dir.join(SAVE_PREV)).unwrap();
        fs::write(dir.join(SAVE_PREV).join("save.bin"), b"v1").unwrap();
        fs::remove_dir_all(dir.join(SAVE_DATA)).unwrap();
        fs::create_dir_all(dir.join(SAVE_PENDING)).unwrap();
        fs::write(dir.join(SAVE_PENDING).join("save.bin"), b"v2").unwrap();
        write_journal(&dir, generation + 1).unwrap();

        let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
        assert_eq!(save.meta().generation, generation);
        assert_eq!(fs::read(save.live_root().join("save.bin")).unwrap(), b"v1");
        assert!(!dir.join(SAVE_PREV).exists());
        assert!(!dir.join(SAVE_JOURNAL).exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recovery_finishes_a_commit_whose_metadata_is_already_written() {
        let root = temp_root("journal-finish");
        let a = attr(0x6161, 1);
        let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
        write_pending(&save, "save.bin", b"v1");
        save.commit().unwrap();
        write_pending(&save, "save.bin", b"v2");
        save.commit().unwrap();
        let dir = save.root.clone();
        let generation = save.meta().generation;
        drop(save);

        // Crash after the metadata sidecar was replaced, before the old tree
        // and the journal were removed.
        fs::create_dir_all(dir.join(SAVE_PREV)).unwrap();
        fs::write(dir.join(SAVE_PREV).join("save.bin"), b"v1").unwrap();
        write_journal(&dir, generation).unwrap();

        let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
        assert_eq!(fs::read(save.live_root().join("save.bin")).unwrap(), b"v2");
        assert!(!dir.join(SAVE_PREV).exists());
        assert!(!dir.join(SAVE_JOURNAL).exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn enumeration_reports_real_metadata_and_filters() {
        let root = temp_root("enumerate");
        let alice = attr(0x1111, 1);
        let bob = attr(0x2222, 1);
        let cache = attr(0x3333, 5);
        for (a, bytes) in [(alice, 4u64), (bob, 8), (cache, 16)] {
            let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
            write_pending(&save, "file.bin", &vec![0u8; bytes as usize]);
            save.commit().unwrap();
        }

        let all = collect_saves(&root, InfoFilter::All);
        assert_eq!(all.len(), 3);
        let alice_meta = all.iter().find(|m| m.attr == alice).unwrap();
        assert_eq!(alice_meta.attr.user_id, user_id(0x1111));
        assert_eq!(alice_meta.size, DEFAULT_SAVE_SIZE);
        assert_eq!(alice_meta.journal_size, DEFAULT_JOURNAL_SIZE);
        assert_eq!(alice_meta.commit_id, 1);
        assert_ne!(
            alice_meta.save_id,
            all.iter().find(|m| m.attr == bob).unwrap().save_id
        );

        let info = alice_meta.info_bytes();
        assert_eq!(
            u64::from_le_bytes(info[0..8].try_into().unwrap()),
            alice_meta.save_id
        );
        assert_eq!(info[8], 1);
        assert_eq!(info[9], 1);
        assert_eq!(&info[0x10..0x20], &user_id(0x1111));
        assert_eq!(
            u64::from_le_bytes(info[0x28..0x30].try_into().unwrap()),
            0x0100_0000_0001_0000
        );
        assert_eq!(
            u64::from_le_bytes(info[0x30..0x38].try_into().unwrap()),
            alice_meta.size
        );

        assert_eq!(collect_saves(&root, InfoFilter::Cache).len(), 1);
        assert_eq!(
            collect_saves(
                &root,
                InfoFilter::Filter(SaveDataFilter {
                    filter_application: false,
                    filter_type: false,
                    filter_user: true,
                    filter_system: false,
                    filter_index: false,
                    attr: bob,
                })
            )
            .len(),
            1
        );

        let save = SaveData::mount(root.clone(), 0x100, 1, alice, false, true).unwrap();
        let extra = save.extra_bytes();
        assert_eq!(&extra[8..0x18], &user_id(0x1111));
        assert_eq!(
            u64::from_le_bytes(extra[0x60..0x68].try_into().unwrap()),
            DEFAULT_JOURNAL_SIZE
        );
        assert_eq!(u64::from_le_bytes(extra[0x68..0x70].try_into().unwrap()), 1);
        assert_eq!(
            u64::from_le_bytes(extra[0x58..0x60].try_into().unwrap()),
            DEFAULT_SAVE_SIZE - 4
        );

        let text = fs::read_to_string(save.root.join(SAVE_META)).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["save_id"], alice_meta.save_id);
        assert_eq!(value["user"], hex(&user_id(0x1111)));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn create_delete_and_size_limits_use_horizon_errors() {
        let root = temp_root("limits");
        let a = attr(0x7777, 1);
        let creation = SaveCreation {
            size: 0x100,
            journal_size: 0,
            ..SaveCreation::default()
        };
        assert_eq!(create_save(&root, 0x100, 1, a, creation), Ok(()));
        assert_eq!(
            create_save(&root, 0x100, 1, a, creation),
            Err(PATH_ALREADY_EXISTS)
        );

        let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
        assert_eq!(save.meta().size, 0x100);
        write_pending(&save, "too-big.bin", &vec![0u8; 0x200]);
        assert_eq!(save.commit(), Err(USABLE_SPACE_NOT_ENOUGH));
        assert!(!save.data.join("too-big.bin").exists());

        fs::remove_file(save.live_root().join("too-big.bin")).unwrap();
        write_pending(&save, "fits.bin", b"ok");
        save.commit().unwrap();
        let id = save.meta().save_id;
        drop(save);

        assert_eq!(
            SaveData::mount(root.clone(), 0x100, 1, a, false, false)
                .unwrap()
                .meta()
                .save_id,
            id
        );
        assert_eq!(delete_save(&root, id, Some(1)), Ok(()));
        assert_eq!(delete_save(&root, id, Some(1)), Err(TARGET_NOT_FOUND));

        let missing = attr(0x8888, 1);
        assert!(matches!(
            SaveData::mount(root.clone(), 0x100, 1, missing, true, true),
            Err(TARGET_NOT_FOUND)
        ));
        assert!(matches!(
            SaveData::mount(root.clone(), 0x100, 0xff, missing, false, true),
            Err(INVALID_ARGUMENT)
        ));
        let mut bad_type = missing;
        bad_type.data_type = 9;
        assert!(matches!(
            SaveData::mount(root.clone(), 0x100, 1, bad_type, false, true),
            Err(INVALID_ARGUMENT)
        ));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn legacy_host_directories_are_adopted_as_committed_data() {
        let root = temp_root("legacy");
        let a = attr(0x9999, 1);
        let dir = root.join(save_dir_name(&a));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("old.bin"), b"old save").unwrap();

        let save = SaveData::mount(root.clone(), 0x100, 1, a, false, true).unwrap();
        assert_eq!(
            fs::read(save.live_root().join("old.bin")).unwrap(),
            b"old save"
        );
        assert!(save.meta().size >= 8);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn read_only_mounts_reject_writes_and_keep_creation_metadata() {
        let root = temp_root("readonly");
        let a = attr(0xabcd, 1);
        let creation = SaveCreation {
            size: 0x4000,
            journal_size: 0x1000,
            block_size: 0x4000,
            owner_id: 0x1234_5678,
            flags: 4,
            space: 1,
            meta_size: 0,
        };
        create_save(&root, 0x100, 1, a, creation).unwrap();

        let save = SaveData::mount(root.clone(), 0x100, 1, a, true, true).unwrap();
        let meta = save.meta();
        assert_eq!(
            (meta.size, meta.journal_size, meta.block_size),
            (0x4000, 0x1000, 0x4000)
        );
        assert_eq!((meta.owner_id, meta.flags), (0x1234_5678, 4));
        assert_eq!(save.ensure_pending(), Err(READ_ONLY_FILE_SYSTEM));
        assert_eq!(save.commit(), Err(READ_ONLY_FILE_SYSTEM));
        assert_eq!(save.space(), (0x4000, 0x5000));

        let (dir, _) = find_save(&root, meta.save_id, Some(1)).unwrap();
        assert_eq!(extend_save(&root, 1, meta.save_id, 0x8000, 0x2000), Ok(()));
        let meta = SaveMeta::load(&dir).unwrap();
        assert_eq!((meta.size, meta.journal_size), (0x8000, 0x2000));
        assert_eq!(
            extend_save(&root, 1, 0xdead_beef, 0x100, 0),
            Err(TARGET_NOT_FOUND)
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn serves_a_built_romfs() {
        let dir = std::env::temp_dir().join(format!("kirikae-fs-romfs-{}", std::process::id()));
        let src = dir.join("romfs");
        fs::create_dir_all(src.join("Model/Sub")).unwrap();
        fs::write(src.join("top.txt"), b"top level").unwrap();
        fs::write(src.join("Model/a.bin"), vec![7u8; 1000]).unwrap();
        fs::write(src.join("Model/Sub/b.txt"), b"nested file").unwrap();
        let image = dir.join("romfs.bin");
        let size = romfs::build(&src, &image).unwrap();
        let mut r = romfs::Reader::open(&romfs::RomFs {
            path: image,
            offset: 0,
            size,
            compression: None,
        })
        .unwrap();
        let rom = RomFileSystem::new(romfs::list(&mut r, usize::MAX).unwrap());

        let req = |cmd: u32, filter: u32| Request {
            cmd,
            data: filter.to_le_bytes().to_vec(),
            ..Default::default()
        };
        let kind = |path: &str| {
            let reply = rom.lookup(&req(7, 0), path);
            (reply.result, reply.data)
        };
        assert_eq!(kind("top.txt"), (0, vec![ENTRY_FILE, 0, 0, 0]));
        assert_eq!(kind("Model/Sub"), (0, vec![ENTRY_DIRECTORY, 0, 0, 0]));
        assert_eq!(kind(""), (0, vec![ENTRY_DIRECTORY, 0, 0, 0]));
        assert_eq!(kind("Model/missing").0, PATH_NOT_FOUND);

        assert_eq!(rom.lookup(&req(8, 0), "Model/Sub/b.txt").objects.len(), 1);
        assert_eq!(rom.lookup(&req(8, 0), "Model").result, PATH_NOT_FOUND);
        assert_eq!(rom.lookup(&req(9, 3), "Model").objects.len(), 1);
        assert_eq!(rom.lookup(&req(9, 3), "top.txt").result, PATH_NOT_FOUND);

        assert_eq!(
            rom.dirs[""],
            [
                ("Model".to_string(), ENTRY_DIRECTORY, 0),
                ("top.txt".to_string(), ENTRY_FILE, 9)
            ]
        );
        assert_eq!(
            rom.dirs["Model"],
            [
                ("Sub".to_string(), ENTRY_DIRECTORY, 0),
                ("a.bin".to_string(), ENTRY_FILE, 1000)
            ]
        );
        let (offset, size) = rom.files["Model/Sub/b.txt"];
        let mut got = vec![0u8; size as usize];
        r.read_at(offset, &mut got).unwrap();
        assert_eq!(got, b"nested file");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn device_saves_are_one_image_per_application() {
        let root = temp_root("device");
        let title = 0x0100_0000_0002_0000;
        let mut device = attr(0x1111, DEVICE);
        device.reference_id = title;

        let save = SaveData::mount(root.clone(), title, 1, device, false, true).unwrap();
        write_pending(&save, "device.bin", b"device");
        save.commit().unwrap();
        assert_eq!(save.meta().attr.user_id, [0; 16]);
        drop(save);

        // A title that passes a user id still resolves the same device image.
        let mut with_user = device;
        with_user.user_id = user_id(0x2222);
        let save = SaveData::mount(root.clone(), title, 1, with_user, false, true).unwrap();
        assert_eq!(
            fs::read(save.live_root().join("device.bin")).unwrap(),
            b"device"
        );
        assert_eq!(device::device_images(&root, title), 1);
        drop(save);

        let creation = SaveCreation {
            size: 0x1000,
            ..SaveCreation::default()
        };
        assert_eq!(
            create_save(&root, title, 1, device, creation),
            Err(PATH_ALREADY_EXISTS)
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn deleting_an_application_drops_its_saves_and_not_its_neighbours() {
        let root = temp_root("app-delete");
        let title = 0x0100_0000_0003_0000;
        let (owner, neighbour) = (0x0100_0000_0003_0000, 0x0100_0000_0004_0000);
        let creation = SaveCreation {
            size: 0x1000,
            ..SaveCreation::default()
        };
        for mut a in [attr(0x1111, 1), attr(0x1111, DEVICE), attr(0x1111, CACHE)] {
            a.reference_id = owner;
            create_save(&root, title, 1, a, creation).unwrap();
        }
        let mut other = attr(0x1111, DEVICE);
        other.reference_id = neighbour;
        create_save(&root, title, 1, other, creation).unwrap();

        assert_eq!(device::delete_application(&root, owner), Ok(()));
        assert!(
            load_saves(&root)
                .iter()
                .all(|(_, meta)| meta.attr.reference_id != owner)
        );
        assert_eq!(device::device_images(&root, neighbour), 1);
        assert_eq!(
            device::delete_application(&root, owner),
            Err(TARGET_NOT_FOUND)
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_storage_has_a_fixed_quota_and_round_trips() {
        let root = temp_root("cache");
        let title = 0x0100_0000_0005_0000;
        let mut storage = attr(0x1111, CACHE);
        storage.reference_id = title;
        storage.index = 0;

        let save = SaveData::mount(root.clone(), title, 1, storage, false, true).unwrap();
        assert_eq!(save.meta().size, cache::CACHE_STORAGE_SIZE);
        write_pending(&save, "shader/cache.bin", &[7u8; 0x100]);
        save.commit().unwrap();
        drop(save);

        let (free, total) = cache::size(&root, 0).unwrap();
        assert_eq!(total, cache::CACHE_STORAGE_SIZE);
        assert_eq!(free, total - 0x100);

        // A second index gets its own image and byte budget.
        let mut second = storage;
        second.index = 1;
        let save = SaveData::mount(root.clone(), title, 1, second, false, true).unwrap();
        assert_eq!(cache::reserved(&root), 2 * cache::CACHE_STORAGE_SIZE);
        drop(save);

        assert_eq!(cache::delete(&root, 0), Ok(()));
        assert_eq!(cache::delete(&root, 0), Err(TARGET_NOT_FOUND));
        assert_eq!(cache::size(&root, 0).unwrap_err(), TARGET_NOT_FOUND);

        let mut invalid = storage;
        invalid.index = cache::CACHE_STORAGE_COUNT;
        assert!(matches!(
            SaveData::mount(root.clone(), title, 1, invalid, false, true),
            Err(INVALID_ARGUMENT)
        ));
        assert_eq!(cache::delete(&root, 0xffff), Err(INVALID_ARGUMENT));

        // Fill the 32 indices, then prove the total budget has no room left.
        for index in 0..cache::CACHE_STORAGE_COUNT {
            let mut storage = storage;
            storage.index = index;
            let _ = SaveData::mount(root.clone(), title, 1, storage, false, true).unwrap();
        }
        assert_eq!(
            cache::check_budget(&root, 1, 0),
            Err(USABLE_SPACE_NOT_ENOUGH)
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_writes_beyond_the_index_quota_are_refused() {
        let root = temp_root("cache-quota");
        let title = 0x0100_0000_0006_0000;
        let mut storage = attr(0x2222, CACHE);
        storage.reference_id = title;
        let creation = SaveCreation {
            size: 0x1000,
            journal_size: 0,
            ..SaveCreation::default()
        };
        create_save(&root, title, 1, storage, creation).unwrap();
        assert_eq!(
            create_save(&root, title, 1, storage, creation),
            Err(PATH_ALREADY_EXISTS)
        );

        let save = SaveData::mount(root.clone(), title, 1, storage, false, true).unwrap();
        write_pending(&save, "big.bin", &vec![0u8; 0x2000]);
        assert_eq!(save.commit(), Err(USABLE_SPACE_NOT_ENOUGH));
        // The per-index cap is 32 MiB even when creation asks for more.
        let huge = SaveCreation {
            size: cache::CACHE_STORAGE_SIZE + 1,
            journal_size: 0,
            ..SaveCreation::default()
        };
        let mut other = storage;
        other.index = 1;
        assert_eq!(
            create_save(&root, title, 1, other, huge),
            Err(USABLE_SPACE_NOT_ENOUGH)
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn meta_files_round_trip_within_the_creation_quota() {
        let root = temp_root("meta");
        let title = 0x0100_0000_0007_0000;
        let a = attr(0x3333, 1);
        let creation = SaveCreation {
            size: 0x1000,
            meta_size: 0x20,
            ..SaveCreation::default()
        };
        create_save(&root, title, 1, a, creation).unwrap();
        assert_eq!(
            SaveMeta::load(&root.join(save_dir_name(&a)))
                .unwrap()
                .meta_size,
            0x20
        );

        let file = meta::SaveMetaFile::open(&root, 1, a, meta::META_THUMBNAIL).unwrap();
        assert_eq!(file.limit(), 0x20);
        file.write_at(0, b"abc").unwrap();
        file.flush().unwrap();
        assert_eq!(file.len(), 3);
        assert_eq!(file.set_len(0x40), Err(USABLE_SPACE_NOT_ENOUGH));
        assert_eq!(file.write_at(0x1f, b"xy"), Err(USABLE_SPACE_NOT_ENOUGH));
        drop(file);

        let file = meta::SaveMetaFile::open(&root, 1, a, meta::META_THUMBNAIL).unwrap();
        let mut buf = [0u8; 8];
        let n = file.read_at(1, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"bc");
        // The extension context is a separate file with the same quota.
        let other = meta::SaveMetaFile::open(&root, 1, a, meta::META_EXTENSION_CONTEXT).unwrap();
        assert_eq!(other.len(), 0);
        drop(other);
        drop(file);

        assert!(matches!(
            meta::SaveMetaFile::open(&root, 1, a, 0),
            Err(INVALID_ARGUMENT)
        ));
        let missing = attr(0x4444, 1);
        assert!(matches!(
            meta::SaveMetaFile::open(&root, 1, missing, meta::META_THUMBNAIL),
            Err(TARGET_NOT_FOUND)
        ));
        assert!(matches!(
            meta::SaveMetaFile::open(&root, 0, a, meta::META_THUMBNAIL),
            Err(TARGET_NOT_FOUND)
        ));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_write_past_the_end_without_the_append_bit_is_refused() {
        use crate::abi::Entry;
        use crate::services::{Buf, harness};

        extern "C" fn probe(c: &mut Cpu) {
            let dir = temp_root("write-beyond");
            let path = dir.join("f.bin");
            fs::write(&path, b"abcd").unwrap();
            let open = |append: bool| HostFile {
                file: Mutex::new(
                    OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&path)
                        .unwrap(),
                ),
                append,
                limit: None,
                read_only: false,
            };
            let out = guest_buffer(c, 0x1_2000_0000, 0x1000);
            c.process().memory.write(out, b"12345678");
            let write = |c: &mut Cpu, file: &HostFile, offset: u64, size: u64| {
                let mut data = 0u32.to_le_bytes().to_vec();
                data.extend_from_slice(&0u32.to_le_bytes());
                data.extend_from_slice(&offset.to_le_bytes());
                data.extend_from_slice(&size.to_le_bytes());
                file.call(
                    c,
                    &Request {
                        cmd: 1,
                        data,
                        a: vec![Buf { addr: out, size }],
                        ..Default::default()
                    },
                )
            };

            let past = write(c, &open(false), 0, 8);
            let fits = write(c, &open(false), 0, 4);
            let appended = write(c, &open(true), 0, 4);
            let contents = fs::read(&path).unwrap();
            let len = contents.len() as u64;
            harness::record(
                "write",
                &[
                    u64::from(past.result),
                    u64::from(fits.result),
                    u64::from(appended.result),
                    len,
                ],
            );
            let _ = fs::remove_dir_all(&dir);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("write"),
            vec![u64::from(WRITE_BEYOND_END), 0, 0, 8]
        );
    }

    #[test]
    fn file_timestamps_come_from_the_host_and_missing_paths_are_refused() {
        use crate::abi::Entry;
        use crate::services::{Buf, harness};

        extern "C" fn probe(c: &mut Cpu) {
            let root = temp_root("timestamp");
            let path = root.join("f.bin");
            fs::write(&path, b"x").unwrap();
            let modified = fs::metadata(&path)
                .unwrap()
                .modified()
                .unwrap()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let service = HostFs::new(root.clone());
            let name = 0x1_2000_0000u64;
            c.process()
                .memory
                .map_as(
                    name,
                    0x1000,
                    crate::mem::state::NORMAL,
                    crate::mem::perm::RW,
                )
                .unwrap();

            c.process().memory.write(name, b"f.bin\0");
            let found = service.call(
                c,
                &Request {
                    cmd: 14,
                    x: vec![Buf {
                        addr: name,
                        size: 6,
                    }],
                    ..Default::default()
                },
            );
            let stamp = found.data.clone();

            c.process().memory.write(name, b"gone.bin\0");
            let missing = service.call(
                c,
                &Request {
                    cmd: 14,
                    x: vec![Buf {
                        addr: name,
                        size: 9,
                    }],
                    ..Default::default()
                },
            );

            harness::record(
                "timestamp",
                &[
                    u64::from(found.result),
                    stamp.len() as u64,
                    u64::from_le_bytes(stamp[0..8].try_into().unwrap()),
                    u64::from(stamp[0x18]),
                    u64::from(missing.result),
                    modified,
                ],
            );
            let _ = fs::remove_dir_all(&root);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let seen = harness::observed("timestamp");
        assert_eq!(seen[0], 0);
        assert_eq!(seen[1], 0x20);
        assert_eq!(seen[2], seen[5], "the modified time is the host's");
        assert_eq!(seen[3], 1, "the timestamps are marked valid");
        assert_eq!(seen[4], u64::from(PATH_NOT_FOUND));
    }

    #[test]
    fn the_filesystem_attribute_is_the_documented_0xc0_bytes() {
        use crate::abi::Entry;
        use crate::services::harness;

        extern "C" fn probe(c: &mut Cpu) {
            let service = HostFs::new(temp_root("attribute"));
            let reply = service.call(
                c,
                &Request {
                    cmd: 16,
                    ..Default::default()
                },
            );
            let data = reply.data.clone();
            let limits: Vec<u64> = (0..13)
                .map(|i| {
                    let at = 0x28 + i * 4;
                    u64::from(u32::from_le_bytes(data[at..at + 4].try_into().unwrap()))
                })
                .collect();
            harness::record(
                "attribute",
                &[
                    u64::from(reply.result),
                    data.len() as u64,
                    data[..13].iter().filter(|&&b| b == 1).count() as u64,
                    data[0x5c..].iter().filter(|&&b| b == 0).count() as u64,
                    limits.iter().filter(|&&v| v == 0x300).count() as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("attribute"),
            vec![
                0,
                FILE_SYSTEM_ATTRIBUTE_SIZE as u64,
                13,
                (FILE_SYSTEM_ATTRIBUTE_SIZE - 0x5c) as u64,
                13
            ]
        );
    }

    #[test]
    fn creation_and_extend_enforce_the_documented_quotas() {
        let root = temp_root("quota");
        let title = 0x0100_0000_0008_0000;
        let a = attr(0x5555, 1);
        let bad_block = SaveCreation {
            block_size: 0x123,
            ..SaveCreation::default()
        };
        assert_eq!(
            create_save(&root, title, 1, a, bad_block),
            Err(INVALID_ARGUMENT)
        );
        let too_big = SaveCreation {
            size: SAVE_SPACE,
            journal_size: DEFAULT_JOURNAL_SIZE,
            ..SaveCreation::default()
        };
        assert_eq!(
            create_save(&root, title, 1, a, too_big),
            Err(USABLE_SPACE_NOT_ENOUGH)
        );

        let creation = SaveCreation {
            size: 0x1000,
            journal_size: 0x1000,
            ..SaveCreation::default()
        };
        create_save(&root, title, 1, a, creation).unwrap();
        let save = SaveData::mount(root.clone(), title, 1, a, false, false).unwrap();
        let id = save.meta().save_id;
        drop(save);
        assert_eq!(
            extend_save(&root, 1, id, SAVE_SPACE + 1, 0),
            Err(USABLE_SPACE_NOT_ENOUGH)
        );
        assert_eq!(extend_save(&root, 1, id, 0x800, 0x800), Ok(()));
        let meta = SaveMeta::load(&root.join(save_dir_name(&a))).unwrap();
        assert_eq!((meta.size, meta.journal_size), (0x800, 0x800));
        let _ = fs::remove_dir_all(&root);
    }

    /// Maps a scratch buffer in the guest and returns its address.
    fn guest_buffer(c: &mut Cpu, addr: u64, size: u64) -> u64 {
        c.process()
            .memory
            .map_as(addr, size, crate::mem::state::NORMAL, crate::mem::perm::RW)
            .unwrap();
        addr
    }

    #[test]
    fn save_data_info_records_have_the_documented_0x60_byte_stride() {
        use crate::abi::Entry;
        use crate::services::{Buf, harness};

        static ROOT: Mutex<Option<PathBuf>> = Mutex::new(None);

        let root = temp_root("info-stride");
        let creation = SaveCreation {
            size: 0x1000,
            journal_size: 0,
            ..SaveCreation::default()
        };
        for a in [attr(0x1111, 1), attr(0x2222, 1)] {
            create_save(&root, 0x100, 1, a, creation).unwrap();
        }
        let expected: Vec<u64> = collect_saves(&root, InfoFilter::All)
            .iter()
            .map(|meta| meta.save_id)
            .collect();
        assert_eq!(expected.len(), 2);
        *ROOT.lock().unwrap() = Some(root.clone());

        // 0xAA in the buffer proves the record covers its whole 0x60 bytes: a
        // 0x40-byte stride would leave 0xAA in the second record and its tail.
        extern "C" fn probe(c: &mut Cpu) {
            let root = ROOT.lock().unwrap().clone().expect("the root is set");
            let out = guest_buffer(c, 0x1_2000_0000, 0x1000);
            c.process().memory.write(out, &[0xaa; 0x1000]);
            let reply = SaveDataInfoReader::new(root, InfoFilter::All).call(
                c,
                &Request {
                    cmd: 0,
                    b: vec![Buf {
                        addr: out,
                        size: 2 * SAVE_INFO_SIZE as u64,
                    }],
                    ..Default::default()
                },
            );
            let count = u64::from_le_bytes(reply.data[..8].try_into().unwrap());
            let bytes = c
                .process()
                .memory
                .read(out, 2 * SAVE_INFO_SIZE as u64)
                .unwrap();
            let id = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            harness::record(
                "info",
                &[
                    count,
                    id(0),
                    id(SAVE_INFO_SIZE),
                    u64::from(bytes[SAVE_INFO_SIZE - 4]),
                    u64::from(bytes[SAVE_INFO_SIZE - 0x24]),
                    u64::from(bytes[0x3b]),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let info = harness::observed("info");
        assert_eq!(info.len(), 6);
        assert_eq!(info[0], 2, "two saves are enumerated");
        assert_eq!(info[1], expected[0]);
        assert_eq!(info[2], expected[1], "the second record starts at 0x60");
        assert_eq!(info[3], 0, "the reserved tail is zeroed");
        assert_eq!(info[4], 0, "the reserved bytes before State are zeroed");
        assert_eq!(info[5], 0, "State is zero (normal)");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn directory_entries_mark_a_directory_in_the_attribute_byte() {
        use crate::abi::Entry;
        use crate::services::{Buf, harness};

        extern "C" fn probe(c: &mut Cpu) {
            let out = guest_buffer(c, 0x1_2000_0000, 0x1000);
            let dir = HostDir {
                entries: vec![
                    ("Models".to_string(), ENTRY_DIRECTORY, 0),
                    ("data.bin".to_string(), ENTRY_FILE, 0x1234),
                ],
                next: Mutex::new(0),
            };
            let reply = dir.call(
                c,
                &Request {
                    cmd: 0,
                    b: vec![Buf {
                        addr: out,
                        size: 2 * DIRECTORY_ENTRY_SIZE as u64,
                    }],
                    ..Default::default()
                },
            );
            let count = u64::from_le_bytes(reply.data[..8].try_into().unwrap());
            let bytes = c
                .process()
                .memory
                .read(out, 2 * DIRECTORY_ENTRY_SIZE as u64)
                .unwrap();
            let at = |i: usize, off: usize| u64::from(bytes[i * DIRECTORY_ENTRY_SIZE + off]);
            let size_at = |i: usize| {
                let at = i * DIRECTORY_ENTRY_SIZE + 0x308;
                u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
            };
            harness::record(
                "entries",
                &[
                    count,
                    at(0, 0x301),
                    at(0, 0x304),
                    size_at(0),
                    at(1, 0x301),
                    at(1, 0x304),
                    size_at(1),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("entries"),
            vec![
                2,
                1,
                ENTRY_DIRECTORY as u64,
                0,
                0,
                ENTRY_FILE as u64,
                0x1234
            ]
        );
    }

    #[test]
    fn sd_card_accessibility_round_trips_through_the_proxy() {
        use crate::abi::Entry;
        use crate::services::harness;

        extern "C" fn probe(c: &mut Cpu) {
            let read = |c: &mut Cpu| {
                let reply = FsProxy.call(
                    c,
                    &Request {
                        cmd: 631,
                        ..Default::default()
                    },
                );
                u64::from(reply.data[0])
            };
            let initial = read(c);
            let hidden = FsProxy.call(
                c,
                &Request {
                    cmd: 630,
                    data: 0u32.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            let while_hidden = read(c);
            FsProxy.call(
                c,
                &Request {
                    cmd: 630,
                    data: 1u32.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            let restored = read(c);
            harness::record(
                "sd",
                &[initial, u64::from(hidden.result), while_hidden, restored],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(harness::observed("sd"), vec![1, 0, 0, 1]);
    }

    #[test]
    fn detection_notifiers_share_one_event_and_hand_out_its_handle() {
        use crate::abi::Entry;
        use crate::services::harness;

        extern "C" fn probe(c: &mut Cpu) {
            let open = |c: &mut Cpu, cmd: u32| {
                let reply = FsProxy.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                assert_eq!(reply.result, 0, "command {cmd} opens");
                let object = reply.objects.first().cloned().expect("a notifier object");
                let handle = object.call(
                    c,
                    &Request {
                        cmd: 0,
                        ..Default::default()
                    },
                );
                assert_eq!(handle.result, 0);
                assert_eq!(handle.copy_handles.len(), 1);
                handle.copy_handles[0]
            };
            let first = open(c, 500);
            let second = open(c, 500);
            let game_card = open(c, 501);
            let system = open(c, 510);
            let failure = open(c, 700);
            harness::record(
                "notifiers",
                &[
                    u64::from(first),
                    u64::from(second),
                    u64::from(game_card),
                    u64::from(system),
                    u64::from(failure),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let seen = harness::observed("notifiers");
        assert_eq!(seen.len(), 5);
        assert_eq!(seen[0], seen[1], "the SD event is one process event");
        assert_ne!(seen[0], seen[2], "game card has its own event");
        assert_ne!(seen[0], seen[3], "system data has its own event");
        assert_ne!(seen[0], seen[4], "access failure has its own event");
        assert_ne!(seen[2], seen[3]);
        assert_ne!(seen[3], seen[4]);
    }

    #[test]
    fn the_device_operator_reports_the_served_sd_card_and_refuses_the_rest() {
        use crate::abi::Entry;
        use crate::services::harness;
        use crate::services::result::CMIF_UNKNOWN_COMMAND;

        extern "C" fn probe(c: &mut Cpu) {
            let reply = FsProxy.call(
                c,
                &Request {
                    cmd: 400,
                    ..Default::default()
                },
            );
            let operator = reply
                .objects
                .first()
                .cloned()
                .expect("OpenDeviceOperator returns an object");
            let inserted = operator.call(
                c,
                &Request {
                    cmd: 0,
                    ..Default::default()
                },
            );
            let speed = operator.call(
                c,
                &Request {
                    cmd: 1,
                    ..Default::default()
                },
            );
            harness::record(
                "operator",
                &[
                    u64::from(reply.result),
                    u64::from(inserted.result),
                    u64::from(inserted.data[0]),
                    u64::from(speed.result),
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("operator"),
            vec![0, 0, 1, u64::from(CMIF_UNKNOWN_COMMAND)]
        );
    }
}
