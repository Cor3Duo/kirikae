//! `OpenSaveDataMetaFile` and the meta file it returns.
//!
//! A save data image can carry a small auxiliary file next to its main tree: the
//! thumbnail (`SaveDataMetaType` 1) or the extension context (2). The size is
//! fixed when the save is created (`SaveDataMetaInfo.size`, 0x40060 for the
//! thumbnail case), and `IFile` writes beyond it are rejected with
//! `UsableSpaceNotEnough` just like the main tree.
//!
//! The payload lives in `<save>/meta/<type>.bin`, outside `data/`, so the guest
//! cannot see it while enumerating the save filesystem and the commit journal
//! does not have to copy it. The file is flushed with `IFile::Flush`.

use super::*;

/// `SaveDataMetaType::Thumbnail`
pub(super) const META_THUMBNAIL: u8 = 1;
/// `SaveDataMetaType::ExtensionContext`
pub(super) const META_EXTENSION_CONTEXT: u8 = 2;

/// Directory under the save root that holds meta files. Hidden from the guest
/// filesystem, like `data/` and the journal.
pub(super) const META_DIR: &str = "meta";

/// Meta image size the SDK asks for when it creates a save with a thumbnail.
pub(super) const META_DEFAULT_SIZE: u64 = 0x40060;

/// One open meta file. `limit` is the size the save was created with.
pub(super) struct SaveMetaFile {
    file: Mutex<File>,
    limit: u64,
}

impl SaveMetaFile {
    /// Opens (and creates, when missing) the meta file of `meta_type` for the
    /// save described by `attr`. A save that does not exist is `PathNotFound`.
    pub(super) fn open(
        root: &Path,
        space: u8,
        attr: SaveAttr,
        meta_type: u8,
    ) -> Result<SaveMetaFile, u32> {
        if !matches!(meta_type, META_THUMBNAIL | META_EXTENSION_CONTEXT) {
            return Err(INVALID_ARGUMENT);
        }
        let Some((dir, meta)) = find_save_by_attr(root, attr) else {
            return Err(TARGET_NOT_FOUND);
        };
        if meta.space != space {
            return Err(TARGET_NOT_FOUND);
        }
        let meta_dir = dir.join(META_DIR);
        fs::create_dir_all(&meta_dir).map_err(io_code)?;
        let path = meta_dir.join(format!("{meta_type}.bin"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(io_code)?;
        let limit = match meta.meta_size {
            0 => META_DEFAULT_SIZE,
            size => u64::from(size),
        };
        Ok(SaveMetaFile {
            file: Mutex::new(file),
            limit,
        })
    }

    pub(super) fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize, u32> {
        let mut f = lock(&self.file);
        f.seek(SeekFrom::Start(offset)).map_err(io_code)?;
        read_full(&mut f, buf).map_err(io_code)
    }

    pub(super) fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), u32> {
        if offset.saturating_add(data.len() as u64) > self.limit {
            return Err(USABLE_SPACE_NOT_ENOUGH);
        }
        let mut f = lock(&self.file);
        f.seek(SeekFrom::Start(offset)).map_err(io_code)?;
        f.write_all(data).map_err(io_code)
    }

    pub(super) fn flush(&self) -> Result<(), u32> {
        lock(&self.file).flush().map_err(io_code)
    }

    pub(super) fn set_len(&self, size: u64) -> Result<(), u32> {
        if size > self.limit {
            return Err(USABLE_SPACE_NOT_ENOUGH);
        }
        lock(&self.file).set_len(size).map_err(io_code)
    }

    pub(super) fn len(&self) -> u64 {
        lock(&self.file)
            .metadata()
            .map_or(0, |metadata| metadata.len())
    }

    /// The size the save image was created with.
    #[cfg(test)]
    pub(super) fn limit(&self) -> u64 {
        self.limit
    }
}

impl Service for SaveMetaFile {
    fn name(&self) -> String {
        "fs/IFile(save meta)".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 => {
                let (offset, size) = (req.u64(8), req.u64(16));
                let Some(b) = req.b.first() else {
                    return Reply::err(OUT_OF_RANGE);
                };
                let mut buf = vec![0u8; size.min(b.size) as usize];
                let n = self.read_at(offset, &mut buf).unwrap_or(0);
                p.memory.write(b.addr, &buf[..n]);
                Reply::ok().u64(n as u64)
            }
            1 => {
                let (offset, size) = (req.u64(8), req.u64(16));
                let Some(a) = req.a.first() else {
                    return Reply::err(OUT_OF_RANGE);
                };
                let data = p.memory.read(a.addr, size.min(a.size)).unwrap_or_default();
                reply(self.write_at(offset, &data))
            }
            2 => reply(self.flush()),
            3 => reply(self.set_len(req.u64(0))),
            4 => Reply::ok().u64(self.len()),
            _ => crate::services::stub::unimplemented_named(
                "fs/IFile(save meta)",
                file_command_name(req.cmd),
                c,
                req,
            ),
        }
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
