//! Cache storage: the per-application scratch area whose indices a title can
//! clear from Data Management.
//!
//! Cache storage is a savedata type (`SaveData.type == 5`) keyed by
//! `(application id, save data index)` and *not* by user: it belongs to the
//! application, like device save data. Horizon reserves a fixed amount for each
//! index and a bounded number of indices; the runtime keeps that shape so a
//! title that walks its cache indices sees a finite, consistent quota instead of
//! an unlimited host directory.

use super::*;

/// `SaveDataIndex` values the system accepts for cache storage. The console
/// exposes 32 slots to a title; the API rejects anything above that.
pub(super) const CACHE_STORAGE_COUNT: u16 = 0x20;

/// Size the runtime provisions for one cache index. 32 MiB matches the default
/// save image the runtime already uses and keeps a whole cache small enough to
/// be created during boot.
pub(super) const CACHE_STORAGE_SIZE: u64 = 0x200_0000;

/// Total a title may hold across every cache index (1 GiB). This is a runtime
/// policy: it bounds what a single title can write without touching the host's
/// free space.
pub(super) const CACHE_STORAGE_TOTAL: u64 = 0x4000_0000;

/// Whether `index` names a cache storage the console would accept.
pub(super) fn valid_index(index: u16) -> bool {
    index < CACHE_STORAGE_COUNT
}

/// Bytes reserved by every cache storage already created for the title. The
/// cache budget counts the data image; the journal lives in the save space
/// budget instead.
pub(super) fn reserved(root: &Path) -> u64 {
    load_saves(root)
        .into_iter()
        .filter(|(_, meta)| meta.attr.data_type == CACHE)
        .map(|(_, meta)| meta.size)
        .sum()
}

/// Size a cache storage created by an implicit mount gets: the per-index size,
/// clamped to the title's remaining cache budget.
pub(super) fn default_size(root: &Path) -> u64 {
    CACHE_STORAGE_SIZE.min(CACHE_STORAGE_TOTAL.saturating_sub(reserved(root)))
}

/// Checks a new or extended cache storage against the per-index and total
/// budgets. `existing` is the storage's current reservation when it is being
/// extended.
pub(super) fn check_budget(root: &Path, size: u64, existing: u64) -> Result<(), u32> {
    if size == 0 || size > CACHE_STORAGE_SIZE {
        return Err(USABLE_SPACE_NOT_ENOUGH);
    }
    let used = reserved(root).saturating_sub(existing);
    if used.saturating_add(size) > CACHE_STORAGE_TOTAL {
        return Err(USABLE_SPACE_NOT_ENOUGH);
    }
    Ok(())
}

/// `DeleteCacheStorage`: removes every cache storage with `index`. Cache
/// storages are application-scoped, so one title has at most one per index.
pub(super) fn delete(root: &Path, index: u16) -> Result<(), u32> {
    if !valid_index(index) {
        return Err(INVALID_ARGUMENT);
    }
    let mut found = false;
    for (dir, meta) in load_saves(root) {
        if meta.attr.data_type == CACHE && meta.attr.index == index {
            fs::remove_dir_all(dir).map_err(io_code)?;
            found = true;
        }
    }
    if found { Ok(()) } else { Err(TARGET_NOT_FOUND) }
}

/// `GetCacheStorageSize`: `(free, total)` bytes of the index, matching the order
/// `IFileSystem::GetFreeSpaceSize` and `GetTotalSpaceSize` use separately.
pub(super) fn size(root: &Path, index: u16) -> Result<(u64, u64), u32> {
    if !valid_index(index) {
        return Err(INVALID_ARGUMENT);
    }
    let Some((dir, meta)) = load_saves(root)
        .into_iter()
        .find(|(_, meta)| meta.attr.data_type == CACHE && meta.attr.index == index)
    else {
        return Err(TARGET_NOT_FOUND);
    };
    let used = tree_usage(&dir.join(SAVE_DATA)).unwrap_or(0);
    Ok((meta.size.saturating_sub(used), meta.size))
}
