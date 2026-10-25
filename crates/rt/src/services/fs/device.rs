//! Device save data lifetime.
//!
//! Device save data (`SaveData.type == 3`) belongs to the hardware/application,
//! not to an account: `SaveDataAttribute` is all-zero except the type and the
//! optional application id (Switchbrew "Filesystem services", Notes). The runtime
//! therefore keys it by application id only and treats the application's save
//! data deletion as a device reset: the device image is dropped together with
//! the account, cache and BCAT images of the same application.
//!
//! This module also owns the application-wide delete used by
//! `fsp-srv::DeleteSaveDataFileSystem` (command 21), whose input is an
//! application id and not a `SaveDataId`.

use super::*;

/// Removes every save image owned by `application_id` (account, device, cache,
/// BCAT and temporary data). Returns `PathNotFound` when the application owns
/// nothing, matching the console's answer for an unknown application.
pub(super) fn delete_application(root: &Path, application_id: u64) -> Result<(), u32> {
    if application_id == 0 {
        return Err(INVALID_ARGUMENT);
    }
    let mut found = false;
    for (dir, meta) in load_saves(root) {
        if meta.attr.reference_id == application_id && meta.attr.data_type != DEVICE {
            fs::remove_dir_all(dir).map_err(io_code)?;
            found = true;
        }
    }
    if delete_device(root, application_id).is_ok() {
        found = true;
    }
    if found { Ok(()) } else { Err(TARGET_NOT_FOUND) }
}

/// Drops the device save of `application_id`, if it exists. `DeleteSaveDataFileSystem`
/// uses this so uninstalling an application resets the device data it left on the
/// console; the account/cache images go through their own attribute deletes.
pub(super) fn delete_device(root: &Path, application_id: u64) -> Result<(), u32> {
    let Some((dir, _)) = load_saves(root).into_iter().find(|(_, meta)| {
        meta.attr.data_type == DEVICE && meta.attr.reference_id == application_id
    }) else {
        return Err(TARGET_NOT_FOUND);
    };
    fs::remove_dir_all(dir).map_err(io_code)
}

/// Number of device save images an application owns. Horizon allows one; the
/// commit tests use this to prove a second user id does not create a second
/// image.
#[cfg(test)]
pub(super) fn device_images(root: &Path, application_id: u64) -> usize {
    load_saves(root)
        .into_iter()
        .filter(|(_, meta)| {
            meta.attr.data_type == DEVICE && meta.attr.reference_id == application_id
        })
        .count()
}
