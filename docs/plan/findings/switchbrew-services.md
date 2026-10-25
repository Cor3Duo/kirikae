# Switchbrew audit: services, filesystem and loader metadata

**Date:** 2026-10-25
**Branch:** `ws-wiki-services`
**Question:** the dashboard reports 73 % of Horizon service commands; which of those commands
really exist in the runtime, which only look implemented, and which gaps can be closed from the
pages without guessing?

**Method:** every claim below was checked by reading the runtime source, the Switchbrew page
named in the table, and (where it exists) the libnx wrapper for the same command, which fixes the
buffer attributes and the reply width of the client side. Nothing was taken from a game dump.
Encrypted content, keys, signatures and save-transfer crypto were left alone.

Reference pages:

- [Services API](https://switchbrew.org/wiki/Services_API) — service names, hosts, page map
- [Filesystem services](https://switchbrew.org/wiki/Filesystem_services) — `fsp-srv`, `IFileSystem`,
  `IFile`, `IDirectory`, `IStorage`, `ISaveDataInfoReader`, `IDeviceOperator`, `IEventNotifier`
- [NCA Format](https://switchbrew.org/wiki/NCA_Format), [NCM services](https://switchbrew.org/wiki/NCM_services)
  (`StorageId`, `ProgramId`, `RightsId`), [CNMT](https://switchbrew.org/wiki/CNMT)
- [Network Interface services](https://switchbrew.org/wiki/Network_Interface_services) (`nifm`)
- [Sockets services](https://switchbrew.org/wiki/Sockets_services) (`bsd`), [SSL services](https://switchbrew.org/wiki/SSL_services)
- [NS services](https://switchbrew.org/wiki/NS_Services) (`ns`, `aoc`, `mii`, `pdm`),
  [Shared Database services](https://switchbrew.org/wiki/Shared_Database_services) (`mii`, `pdm`, `pl`)
- [NFC services](https://switchbrew.org/wiki/NFC_services) (`nfc`, `nfp`), [NGC services](https://switchbrew.org/wiki/NGC_services),
  [OMM services](https://switchbrew.org/wiki/OMM_services), [BCAT services](https://switchbrew.org/wiki/BCAT_services)
  (`bcat`, `prepo`), [Parental Control services](https://switchbrew.org/wiki/Parental_Control_services),
  [Glue services](https://switchbrew.org/wiki/Glue_services) (`arp`, `ectx`),
  [Applet Manager services](https://switchbrew.org/wiki/Applet_Manager_services) (`am`)

---

## 1. The 73 % is not coverage

`tools/progress.py` scans `crates/rt/src/services` and compares the command arms it finds with
`tools/reference/horizon.txt`. Two properties of that scan inflate the published number:

1. **55 of the 106 service groups have no entry in the reference file.** For those, the
   denominator is *the number of arms the scanner found*, so the group is always 100 % by
   construction. They contribute 560 arms — a third of the published total — and include groups
   whose real coverage is far lower: `ns:am` is published as 51/51, `ngc:u` as 6/6, `mii/IDatabase`
   as 10/10, `ectx` as 2/2, although the runtime handles 10, 1, 5 and 0 commands respectively.
2. **Diagnostic name tables are counted as implementations.** When a command falls through to
   `unimplemented_named(..., proxy_command_name(req.cmd), ...)`, the scanner follows the call into
   `proxy_command_name`, sees a `match cmd { ... }`, and records every name in it as a handled
   command. `fsp-srv` publishes 131/146, but only **50** of those 131 are arms of the real
   dispatch; the other 81 are names that answer with the CMIF unknown-command result.

Applying only the second correction to the 51 documented groups gives **632 of 1,171 commands
(54 %)** with a real dispatch arm, not the 729 the dashboard reports. Both corrections are
mechanical to make in `callee_bodies`/`scan_code` (skip callees whose name ends in
`_command_name`), but `tools/progress.py` was outside the file scope of this change, so the
dashboard is left as it is and this document states the real figures. The two new name tables
added here (`arp`, `fs/IDeviceOperator`) are lookup tables precisely so they do not add new
phantoms.

Per group, for the pages in scope:

| Page | Group | Dashboard | Real dispatch arms |
| --- | --- | ---: | ---: |
| Filesystem | `fsp-srv` | 131/146 | 50/146 |
| Filesystem | `fs/IFileSystem` | 17/17 | 17/17 |
| Filesystem | `fs/IFile` | 6/7 | 5/7 |
| Filesystem | `fs/IStorage` | 6/6 | 2/6 |
| Filesystem | `fs/IDeviceOperator` | 1/1 (not in reference) | 1/59 |
| NS | `ns/IApplicationManagerInterface` | 40/40 (not in reference) | 10/485 |
| NS | `ns:am` | 51/51 (not in reference) | 10/485 |
| Sockets | `bsd:u` | 2/2 (not in reference) | 2/47 |
| SSL | `ssl` | 1/1 (not in reference) | 1/12 |

---

## 2. Inventory

`Handled` counts arms that answer the command (some with a documented error, e.g.
`InvalidArgument` when the runtime has no data). "Named only" means the only trace of the command
is a diagnostic name; the reply is the CMIF unknown-command result.

### 2.1 Filesystem (`fsp-srv`)

| # | Command | Status |
| ---: | --- | --- |
| 1 | SetCurrentProcess | ok |
| 2 | OpenDataFileSystemByCurrentProcess | returns the RomFS filesystem (this is what `MountRom` mounts) |
| 18 | OpenSdCardFileSystem | host directory per title |
| 21, 22, 23, 25, 28, 32, 35, 37 | save-data create/delete/extend | modelled with Horizon errors |
| 24 | RegisterSaveDataFileSystemAtomicDeletion | refused |
| 26, 27 | `FormatSdCardDryRun`, `IsExFATSupported` | fixed replies |
| 33, 34 | cache storage delete/size | modelled |
| 51–53, 57–62, 67–71, 80 | save mounts, extra data, info readers, meta file | modelled |
| 29, 30, 31, 36, 38, 39, 40 | BIS/game-card/detach/access log | refused (no such device) |
| 400 | OpenDeviceOperator | **added**: `IDeviceOperator` (1 of its 59 commands) |
| 500, 501, 510, 700 | event notifiers | **added**: `IEventNotifier` |
| 502, 503, 511, 520 | wake-up notifiers, NotifySystemDataUpdateEvent, SimulateDeviceDetectionEvent | refused |
| 600–618, 620 | rights ids, verify/corrupt, program id, encryption seed | refused; rights/key commands need keys and stay out of scope |
| 630, 631 | SetSdCardAccessibility / IsSdCardAccessible | **added** (state, default accessible) |
| 601, 612, 616 | save size / commit id queries | ok |
| 800, 810, 820 | error info, program index map, content storage info | refused |
| 1001–1021 | debug/access-log/save-root/stack/report options | the four no-output setters and `GetGlobalAccessLogMode` answer; the rest refused |
| 1200, 1300, 1400 | multi-commit, BIS wiper, error-context ready | refused |
| 210–213 | batch-read data storages (23.0.0+) | refused |

Missing for retail titles, in order of likely impact: `IMultiCommitManager` (1200),
`OpenSaveDataInfoReaderWithFilter` is handled but `FindSaveDataWithFilter` only returns the first
hit, `GetRightsId` (606, add-on content), and the transfer/export interfaces (81–88, 830/831)
which exist to move save data off-device and are built around encryption.

### 2.2 Filesystem sub-interfaces

| Interface | Page commands | Handled | Gaps |
| --- | ---: | ---: | --- |
| `IFileSystem` | 17 | 17 | `QueryEntry` (15) accepts and writes nothing; `GetFileSystemAttribute` (16) added here |
| `IFile` | 7 | 5 | `OperateRange`/`OperateRangeWithBuffer` have no public layout and stay refused |
| `IDirectory` | 2 | 2 | — |
| `IStorage` | 6 | 2 | only Read/GetSize; RomFS is read-only |
| `ISaveDataInfoReader` | 1 | 1 | record stride fixed here |
| `IDeviceOperator` | 59 | 1 | only `IsSdCardInserted`; the rest reads CID/CSD/error logs |
| `IEventNotifier` | 1 | 1 | added here |
| `IStorageForBatchRead` | 11 | 0 | used by 23.0.0+ batch reads |
| `IMultiCommitManager` | 2 | 0 | used when a game commits several saves atomically |
| `IMemoryStressor` | 1 | 0 | debug |
| `IWiper` | 2 | 0 | BIS only |
| `ISaveDataTransferManager` | 4 | 0 | needs keys, out of scope |

### 2.3 Save-data layouts (verified against the page)

`SaveDataAttribute` 0x40, `SaveDataCreationInfo` 0x40, `SaveDataCreationInfo2` 0xA0,
`SaveDataInfo` **0x60** (not 0x40), `SaveDataExtraData` 0x200, `SaveDataFilter` 0x48 after the
space byte, `DirectoryEntry` 0x310 with the attribute byte at 0x301 and the type at 0x304,
`FileTimeStampRaw` 0x20, `FileSystemAttribute` 0xC0. The reader's 0x40 stride, the missing
attribute byte and the zeroed timestamps were real reply-shape bugs, fixed in §3.

### 2.4 Cryptosystem, NCA and CNMT (loader, metadata only)

`crates/loader/src/nca.rs` reads the NCA header at the documented offsets: magic at 0x200,
distribution type at 0x204, content type at 0x205, program id at 0x210, the four `FsEntry`
records at 0x240 whose start/end are 0x200-byte blocks, and each `FsHeader` at 0x400 with the
IVFC/SHA-256 extents, the crypto type and the BKTR compression marker. The four section hashes
and the key area are not needed to locate plaintext. Encrypted sections are refused
(`Error::Encrypted`), never decrypted.

What the loader does **not** read, and therefore what the runtime cannot answer:

- **CNMT**: `exefs_from_nsp` takes the first program NCA that contains an ExeFS with `main`. A
  bundle that carries both base and patch NCAs is resolved by file order, not by content-meta
  version; the update NCA's title version is never read.
- **NACP / control NCA**: `control.nacp` lives in the Control content, which
  `read_source` skips. That is why `ns:am` `GetApplicationControlData` and `arp:r`
  `GetApplicationControlProperty` return `InvalidArgument` instead of data, and why
  `GetDisplayVersion` is the fixed `"1.0.0"` string.
- **Certificates / rights ids**: signed material and key-dependent, deliberately untouched.

### 2.5 Network services a title can answer offline

| Interface | Page commands | Handled | Notes |
| --- | ---: | ---: | --- |
| `nifm:u` | 2 | 2 | `CreateGeneralService`, `CreateGeneralServiceOld` |
| `nifm/IGeneralService` | 56 | 11 | client id, request creation, current profile, profile by uuid, enumerate (empty), current IP, IP config, wireless flag, connection status, internet/foreground accepted |
| `nifm/IRequest` | 26 | 25 | full state machine (Free 1 → Accepted 3), two events, requirement storage; `GetNetworkAccessStatus` (26) missing |
| `bsd` | 47 | 2 | `RegisterClient`, `StartMonitoring` only; there is no socket stack |
| `ssl` | 12 | 1 | `SetInterfaceVersion`; no TLS |

The nifm profile commands were replying through the message's raw data. libnx sends them with a
fixed-size pointer output buffer (`SfBufferAttr_HipcPointer | FixedSize | Out`), so the SDK read
whatever its stack held; fixed in §3.

### 2.6 Remaining retail-reachable families

| Group | Page commands | Handled | Honest status |
| --- | ---: | ---: | --- |
| `ns:IApplicationManagerInterface` (`ns:am`/`ns:am2`) | 485 | 10 | record list/count, desired language, language-code conversion, launch version check, control data (refused), `HasApplicationRecord`, `NeedsUpdateVulnerability`, `IsAnyApplicationEntityInstalled`, `CountApplicationContentMeta` |
| `aoc:u` | 22 | 14 | counts/list are empty, base id is app+0x1000, both list-changed events exist; the EC purchase managers and 16.0.0+ registration are refused |
| `mii:u` + `IDatabase` | 1 + 29 | 1 + 5 | `IsUpdated`, `IsFullDatabase`, `GetCount`, `Get1`, `BuildRandom`; no Mii database exists |
| `nfc` | 35 ids (`IUser`, `ISystem`, mifare) | 35 | every alias answers; detection/tag reads refuse with `InvalidArgument` because there is no controller |
| `ngc:u` | 6 | 1 | only `GetContentVersion` (0); filtering commands have opaque content packs |
| `pdm:qry` | 31 | 12 | empty play log with the 16.0.0 record sizes |
| `prepo:u` | 17 | 17 | reports are accepted and dropped; status is idle |
| `bcat:u` + sub-services | 5 + 27 + 12 | all | empty delivery cache, no network |
| `omm` | 42 | 42 | handheld 720p, no cradle, one event per manager |
| `pctl` | 140 | 83 | parental-control queries answer the defaults; the 57 missing are settings/updates a retail game does not drive |
| `arp:r` | 10 | 10 arms (7 with data, 3 `InvalidArgument`) | launch/process property, instance id, list, unregistration notifier; NACP and certificate unavailable |
| `arp:w` | 5 | 0 | instance registration belongs to the ARP sysmodule and `ns` |
| `ectx:r` / `ectx:aw` | 1 / 0 | 1 / 0 | `PullContext` returns an empty context |

---

## 3. What this change fixes

Each item has a test that fails on the previous behaviour.

1. **`SaveDataInfo` stride (0x40 → 0x60).** `fs/ISaveDataInfoReader::ReadSaveDataInfo` wrote 0x40
   bytes per record while the guest reads an array of 0x60-byte records: the second and later
   saves were written inside the reserved tail of the first one. Test
   `save_data_info_records_have_the_documented_0x60_byte_stride` enumerates two saves, checks the
   ids at 0 and 0x60 and that the reserved bytes and `State` are zeroed.
2. **`DirectoryEntry` attribute byte.** Byte 0x301 (bit 0 = directory) was left zero, so a guest
   that reads attributes saw every directory as a file. `directory_entries_mark_a_directory_in_the_attribute_byte`
   checks both a directory and a file.
3. **`FileTimeStampRaw`.** It returned 0x20 zero bytes for every path, including paths that do not
   exist. It now returns the host modification time in all three fields with the validity byte set,
   and `PathNotFound` for a missing path (`file_timestamps_come_from_the_host_and_missing_paths_are_refused`).
4. **`GetFileSystemAttribute` (15.0.0+).** Was unimplemented on every filesystem. It now returns
   the 0xC0-byte struct with all limits valid at 0x300 (`the_filesystem_attribute_is_the_documented_0xc0_bytes`).
   It is answered before the path guard, because the command takes no path.
5. **`IFile::Write` past end of file.** The page documents that a write may not extend a file
   unless the append bit (0x4) of `OpenFile` was set, and that the failure is `0x307202`. The
   runtime extended the file silently. Test `a_write_past_the_end_without_the_append_bit_is_refused`
   checks the error, a write that fits, and the append case.
6. **`OpenDeviceOperator` (400) and `fs/IDeviceOperator`.** The interface was missing entirely.
   `IsSdCardInserted` reports the SD card the runtime serves; the other 58 commands stay refused
   rather than fabricated (`the_device_operator_reports_the_served_sd_card_and_refuses_the_rest`).
7. **Detection notifiers (500, 501, 510, 700) and `fs/IEventNotifier`.** They returned the
   unknown-command result, so the SDK could not build its notifier. Each now returns an object
   whose `GetEventHandle` hands out one non-signalled process event
   (`detection_notifiers_share_one_event_and_hand_out_its_handle`).
8. **`SetSdCardAccessibility` / `IsSdCardAccessible` (630/631).** Missing; a title asking whether
   its storage is mounted got an error. Now a state that starts accessible and round-trips
   (`sd_card_accessibility_round_trips_through_the_proxy`).
9. **nifm `GetCurrentNetworkProfile` / `GetNetworkProfile` (5/8).** They returned the zeroed
   profile as raw reply data instead of filling the caller's pointer buffer, which is how libnx
   (and the page) describe the command. Now the profile goes into the buffer, only as many bytes
   as the buffer holds (`the_profile_reads_land_in_the_callers_pointer_buffer`).
10. **`ns` record list and language conversion.**
    `ListApplicationRecord` (0) ignored the `entry_offset` argument, so a caller looping until the
    count reaches zero would never stop; offset 0 returns the single record, any offset past it
    returns 0. `ConvertLanguageCodeToApplicationLanguage` (60) returned the index as raw data
    instead of writing it through the caller's output pointer, and its test now asserts the
    buffer, the empty reply and both error/success results.
11. **`StorageId` for the running title.** `am`'s `GetLaunchStorageInfoForDebug` reported 1 for
    application and update storage; the enum defines 1 as `Host` and 4 as `BuiltInUser`. The
    constant (and the new `arp` property) now report 4, which is also what the machine model
    (`ns` lists the title as installed, `am` reports `"1.0.0"`) says.
12. **`arp:r`.** The module claimed the page has no command list; the Glue services page documents
    `nn::arp::detail::IReader`. Launch property (id, storage, kind), process property, instance id,
    `ListApplicationInstanceId` and the unregistration notifier are implemented; the NACP and
    certificate queries return `InvalidArgument` instead of being faked. `arp:w` is now served (its
    five writer commands stay refused: instance registration is done by the ARP sysmodule and
    `ns`), and `arp:u` — a name the Services API page does not list — is no longer registered, so
    `sm:` answers `SM_NOT_REGISTERED` for it as the console would. The module comment was wrong and
    is rewritten.

## 4. What remains, and why

- **fsp-srv**: 15 documented commands are unimplemented and 81 are named only. The next useful
  ones are `OpenMultiCommitManager` (1200, atomic multi-save commit), `OpenSaveDataInfoReaderWithFilter`
  is present but `FindSaveDataWithFilter` stops at the first match, and `GetRightsId` (606) for
  add-on content. The transfer/export interfaces and `RegisterExternalKey`/`GetRightsId` are
  key-dependent and stay out by rule 2.
- **`IFile::OperateRange`** has no published structure for the range reply, so guessing it would
  be worse than the current refusal.
- **`IDeviceOperator`** needs CID/CSD registers, erase counters and error logs. Only insertion has
  a truthful answer for a host directory.
- **`ns` control data** and **`arp` control property / certificate** need the Control NCA and its
  NACP (and, for the certificate, signed material). This is the single highest-value loader gap
  for those two services: a `control.nacp` reader over the same PFS0 code the loader already has.
- **BISO/GcApp/RegUpdate mounts** (`OpenBisFileSystem`, `OpenGameCardFileSystem`,
  `OpenRegisteredUpdatePartition`) are system/title-specific storage; a retail application cannot
  request them, so their absence costs nothing.
- **Sockets and TLS** need a host stack; `bsd`'s `RegisterClient`/`StartMonitoring` exist so SDK
  initialisation succeeds, and every socket call reports failure instead of a fake connection.
- **Big files**: `CreateFile` ignores `FsCreateOption_BigFile` (the concatenation-file flag); a
  title that writes a >4 GiB file would create a plain file.
- **Dashboard**: `tools/progress.py` should ignore `match` arms of `*_command_name` helpers and
  should not use the arm count as the denominator for groups missing from
  `tools/reference/horizon.txt`. Until then, use the "real dispatch arms" column in §1.

## 5. Progress delta

`python tools/progress.py` (read-only run, dashboard regenerated but not committed as part of this
change):

| Metric | Before | After |
| --- | ---: | ---: |
| Horizon services implemented | 1289 | 1302 |
| Counted (silent stubs excluded) | 1256 | 1269 |
| Horizon services total | 1731 | 1744 |
| Percentage | 73 % | 73 % |
| Kernel SVCs | 124/124 | 124/124 |
| nvdrv ioctls | 122/138 | 122/138 |
| NVN procs | 128/154 | 128/154 |

The +13 are all in groups without a reference entry: `arp:r` (10), `arp/IUnregistrationNotifier`
(1), `fs/IEventNotifier` (1) and `fs/IDeviceOperator` (1). The seven `fsp-srv` commands that are
now really implemented (400, 500, 501, 510, 630, 631, 700) do not move the dashboard because they
were already counted through their diagnostic names; the change removes those names from
`proxy_command_name`, so the count stays the same while the implementation stops being fictional.
`fs/IFileSystem` is now 17/17 with a real arm for every command.
