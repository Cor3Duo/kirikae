# Switchbrew core audit

A page-by-page comparison of the Horizon core in `crates/rt` with Switchbrew, done on
2026-10-25 against these revisions:

| Page | Revision |
| --- | --- |
| [Memory layout](https://switchbrew.org/wiki/Memory_layout) | 12866 (2024-07-31) |
| [SVC](https://switchbrew.org/wiki/SVC) | 15307 |
| [HIPC](https://switchbrew.org/wiki/HIPC) | 14205 (2026-01-09) |
| [Kernel](https://switchbrew.org/wiki/Kernel) | 12654 |
| [Error codes](https://switchbrew.org/wiki/Error_codes) | 14808 |
| [Thread Local Region](https://switchbrew.org/wiki/Thread_Local_Region) | 14389 |

"Matches" below means the runtime produces the value, layout or result the page documents
for the case the code handles; a fact the page leaves open is listed as *not documented*
instead of being counted as a mismatch. The facts are paraphrased; the pages are the
source of the exact wording and tables.

## Audit table

| Page | Fact | Code | Status | Evidence |
| --- | --- | --- | --- | --- |
| Memory layout | A 32-bit process has a 32-bit virtual address space; every user region has to stay below 4 GiB | `horizon::layout(Arch::A32)` | fixed | `the_32_bit_layout_keeps_every_region_below_4_gib` |
| Memory layout | The 32-bit ASLR pseudocode for 1.0.0 starts the map region at `0x4000_0000` with a 1 GiB size and puts the heap region after it | `horizon::layout` | matches in order | same test (`alias_base`, `heap_base`) |
| Memory layout | The main binary region is laid out `rtld`, `main`, `subsdk*`, `sdk`, consecutively | `loader::package::LOAD_ORDER`, `process::map_module` | matches | `kirikae info` on MK8D: rtld `0x0800_0000`, main `0x0800_6000`, … sdk `0x09a9_c000` |
| Memory layout | A binary's R-X section is followed by R-- and then RW- with no guard pages | `process::map_module` | matches | segment map above |
| Memory layout | Reserved regions are ASLR'd (heap, map, new map, and a kernel-only TLS/IO region); binaries are randomized | `horizon::layout` is a fixed table | missing (deterministic runtime) | no `MapBaseAddr`/`HeapRegionBaseAddr` randomization; `GetInfo(11)` still returns entropy |
| Memory layout | Kernel page tables and KASLR (TTBR0/TTBR1, carveouts) | — | not modelled (the runtime is a host process) | no kernel VA in the tree |
| Kernel | KThread/KProcess/KSessionRequest object layouts and slab counts | `kernel` models host objects; `ResourceLimit` caps threads/events/transfer memory/sessions | approximates (limits are the runtime's own, not the slab table) | `kernel::ResourceLimit::new`, `kernel::MAX_PORT_SESSIONS` |
| Kernel | Handle values are kernel-assigned small indices; `0xFFFF8000`/`0xFFFF8001` are the current-thread/process aliases | `kernel::HandleTable` (`next = 0x2_0001`), `PSEUDO_CURRENT_*` | partial (aliases match; numbering differs, guests treat handles as opaque) | `HandleTable::default` |
| SVC `SetHeapSize` | Size is a multiple of 2 MiB, at most 4 GiB; `0xCA01` for misaligned/above 4 GiB, `0xD001` when it does not fit the heap region; shrinking is allowed; the heap base comes back in X1 | `horizon::set_heap_size` | fixed | `a_32_bit_guest_can_take_the_heap_mk8d_asks_for` |
| SVC `SetMemoryPermission` | Bit 2 (execute) and write-only are refused; only `---`, `r--`, `rw-` are accepted | `horizon::set_memory_permission` | matches | existing `0xD801` path |
| SVC `SetMemoryAttribute` | Only `mask == value == 0x8` is allowed (cache attribute) | `horizon` dispatch: `0x03 => ok(c)` | missing (always succeeds) | dispatch table |
| SVC `MapMemory` | The source becomes inaccessible (`---`) with the Locked attribute; 2.0.0+ only maps into the stack region; partial unmaps are allowed | `mem::map_mirror`, `horizon::unmap_memory` | partial (copy model, stack state, no attribute, any destination, exact-pair unmap) | `map_memory_borrows_the_source_and_unmap_restores_it` |
| SVC `QueryMemory` | `MemoryInfo` is 0x28 bytes: base 0, size 8, state 0x10, attribute 0x14, permission 0x18, IPC refcount 0x1C, device refcount 0x20, padding 0x24; `PageInfo` in W1 | `horizon::write_memory_info` | layout matches; attribute/refcounts zero; `PageInfo` 0 | `query_memory_reports_the_memory_info_layout` |
| SVC `QueryMemory` | `State` is a `MemoryState` word (`Code 0x00dc7e03`, `CodeData 0x03febd04`, `Normal 0x037ebd05`, …) | `horizon::write_memory_info` writes the runtime's page tag (the type's low byte) | mismatched (kept on purpose, see below) | MK8D experiment below |
| SVC `CreateThread` | Priority `0..0x3F`; core `-2` or `0..3` | `threads::create_thread` | matches | existing tests |
| SVC `WaitSynchronization` | At most 0x40 handles; more is `0xEE01` | `threads::wait_for_objects` | fixed (used to clamp to 64) | `wait_synchronization_refuses_more_than_0x40_handles` |
| SVC `GetInfo` | Types 0-9, 11-18, 20-24, 26, 28 are per-process values; type 17 is the used system resource in bytes | `horizon::get_info` | fixed type 17 and 19; 10/25/27/29-38 still `0xF001` | `system_info_process_info_and_physical_queries` |
| SVC `GetSystemInfo` | Types 0/1 (pool sizes/usage per subtype) and 2 (initial process id range) | `horizon::get_system_info` | matches (one pool, one process) | same test |
| SVC `GetProcessInfo` | Type 0 is `ProcessState`; a running process reports 2 | `horizon::get_process_info` | matches | same test |
| SVC `ReplyAndReceive` | A reply must not carry Send/Receive/Exchange buffers; the call returns `0xE801` | `horizon::send_reply` + `ipc::reply_result` | fixed | `a_reply_cannot_carry_transfer_buffers` |
| SVC exception path | The frame at `TPIDRRO_EL0 + 0x148` is `ExceptionFrameA64` (0x78) on A64 and `ExceptionFrameA32` (0x44: r0-r7, sp, lr, pc, tpidr=1, cpsr, afsr0/1, esr, far) on A32 | `horizon::encode_exception_frame` / `decode_exception_frame` | fixed (was A64-only) | `exception_frames_follow_the_process_architecture` |
| SVC `GetThreadContext3` | `ThreadContext` with flag groups (GPRs 0-28 or 0-12, control, vector, FP control) | `kernel::ThreadContext` (0x320, A64 offsets) | A64 matches; the A32 context is not documented and not modelled (always written as A64) | `thread_context_layout` |
| SVC `CreateIoPool` / `CreateIoRegion` | `IoPoolType_PcieA2` is the only pool type; the pool aperture is `0x1200_0000..0x2000_0000`; mappings are `IoRegister`, `Uncached`, `Memory` | `mem::IoPool`, `mem::IoRegion` | matches for the aperture/enums; one pool per process and IoRegister is refused (no MMIO) | existing device tests |
| SVC `ReadWriteRegister` | MC/MC0/MC1 whitelist plus the 2.0.0 additions, and the PMC block | `MC_REGISTERS`, `horizon::read_write_register` | whitelist matches; the runtime returns `0x4201` instead of inventing register values | `system_info_process_info_and_physical_queries` |
| HIPC | `HeaderData` 8 bytes, `MapData` 0xC, `PointerData` 8, `ReceiveListData` 8 | `ipc::parse_hipc` | matches | `descriptor_tags_match_the_hipc_page_bitfields` |
| HIPC | Header0 counts (Pointer 16-19, Send 20-23, Receive 24-27, Exchange 28-31); Header1 (Raw 0-9, ReceiveListCount 10-13, ReceiveListOffset 20-30, Special 31); Special (Pid 0, Copy 1-4, Move 5-8) | `ipc::parse_hipc`, `ipc::frame` | matches | same test |
| HIPC | Map0/1/2, Pointer0/1 and ReceiveList0/1 bitfields | `ipc::{encode,decode}_map`, `_pointer`, `_receive_list` | matches | same test |
| HIPC | ReceiveListCount 0 disables the list, 1 is inline, 2 is one descriptor, otherwise count-2 descriptors | `ipc::parse_hipc` | descriptor forms match; ReceiveIndex matching is not modelled | `a_message_carries_pointer_map_and_receive_list_descriptors` |
| HIPC | Pointer data is copied into the server's pointer buffer and the reply writes Address/Size/ReceiveIndex back | `ipc` passes the client addresses through | missing (one process, no pointer buffer) | `PointerBufferSize` is only advertised |
| HIPC | Handle value 0 means "no handle" | handle lookups treat 0 as invalid | missing (same result as an invalid handle in most paths) | `HandleTable::get(0)` |
| HIPC | CMIF headers are 0x10 bytes; the in-header version is 1 on 5.0.0+; domains prefix `CmifDomainMessageIn/OutHeader` (0x10) | `ipc::parse_cmif`, `ipc::encode_cmif` | layout matches; out-header version is 0 | could not verify the out version from the page |
| HIPC | TIPC keeps the request id in the low 16 bits of Header0 and has no domains | `ipc::{parse_tipc,encode_tipc}` uses `16 + cmd` | partial (the `+16` base is the runtime's convention, not documented) | `tipc_request_and_reply` |
| Error codes | Bits 0-8 module, 9-21 description; `Svc` is module 1 | constants in `horizon` and `ipc` | matches (`0xCA01`, `0xD001`, `0xD401`, `0xD801`, `0xDC01`, `0xE401`, `0xE601`, `0xE801`, `0xEE01`, `0xF201`, `0xF601`, `0x10801`, …) | tests above |
| Error codes | `0xCE01` OutOfResource for allocation failures | the runtime uses `0x10801` for host refusals and `fatal` for some mappings | partial | `create_shared_memory`, `set_heap_size` |

## The A32 layout

Before, the A32 layout was a 512 MiB heap at `0x2000_0000`, a 512 MiB stack at
`0x4000_0000`, TLS at `0x6000_0000` and a 2 GiB alias at `0x8000_0000`. MK8D asks for
`0x7800_0000` (1920 MiB), so `svcSetHeapSize` returned `0xCA01` and the title terminated
with that result:

```text
svc 0x01 from sdk+0xb42c0 x0=0x200fffcc x1=0x78000000
svc 0x01 -> x0=0xca01 x1=0x78000000
kirikae: the applet set its terminate result to 0xca01
```

The new layout keeps the loader's code base and the documented order (map region, then the
heap region) and leaves a hole between the code region and the stack so the runtime's
service regions are not mistaken for loaded binaries:

| Region | A64 (unchanged) | A32 (new) |
| --- | --- | --- |
| code / ASLR | `0x0800_0000` + `0x7f_f800_0000` | `0x0800_0000` + `0x1800_0000` |
| stack | `0x4_0000_0000` + `0x1_0000_0000` | `0x2000_0000` + `0x1000_0000` |
| TLS | `0x4_8000_0000` | `0x3000_0000` |
| map / alias | `0x10_0000_0000` + `0x10_0000_0000` | `0x4000_0000` + `0x4000_0000` |
| heap | `0x2_0000_0000` + `0x1_8000_0000` | `0x8000_0000` + `0x8000_0000` |

`0x8000_0000 + 0x8000_0000` ends exactly at 4 GiB, which is the SetHeapSize ceiling. The
A64 layout is tested field by field (`the_64_bit_layout_is_unchanged`).

MK8D after the change (same command, same dump):

```text
svc 0x01 from sdk+0xb42c0 x0=0x200fffcc x1=0x78000000
svc 0x01 -> x0=0x0 x1=0x80000000
...
kirikae: unimplemented AArch32 instruction 0xee07cfba at subsdk0+0xccb8: a `cp15` transfer
        other than the thread pointers, the counters and `fpscr` is outside the A32 slice
```

That is 326 traced SVCs (against 180 before) and the next wall is in the A32 interpreter
(`crates/a32` / `interp_a32.rs`), outside this audit.

## What was fixed

1. **A32 layout** (`horizon::layout`): the heap region is `0x8000_0000..0x1_0000_0000` and
   the stack, TLS and alias regions sit below it; the A64 layout is unchanged.
2. **`svcSetHeapSize`** (`horizon::set_heap_size`): the 4 GiB ceiling is enforced with
   `0xCA01`, a request over the heap region with `0xD001` (`ResultOutOfMemory`), a live
   mapping in the way with `0xD401`, and shrinking now unmaps the tail instead of only
   updating the accounting.
3. **`svcWaitSynchronization`** (`threads::wait_for_objects`): more than 0x40 handles (or a
   negative count) is `0xEE01`; it used to be silently clamped to 64.
4. **Exception frames** (`horizon::encode_exception_frame` / `decode_exception_frame`): A32
   processes get `ExceptionFrameA32` (0x44) with r0-r7, sp, lr, pc, `tpidr_el0 = 1`, cpsr
   and far; A64 keeps `ExceptionFrameA64` (0x78).
5. **HIPC reply validation** (`ipc::reply_result`, `horizon::send_reply`): a reply with
   Send/Receive/Exchange descriptors returns `0xE801` and is not delivered; an unmapped
   reply buffer returns `0xE601`.
6. **`svcGetInfo`** (`horizon::get_info`): type 17 reports the bytes of system resource in
   use (it divided by 512 and rounded to a page), type 22 saturates instead of underflowing
   when the system resource is larger than the used memory, and type 19 (initial process id
   range) reports the only process.

## Remaining mismatches, and why

- **`MemoryInfo.State` flag bits.** The runtime reports the `MemoryType` low byte, not the
  whole documented `MemoryState`. Encoding the documented words was implemented and
  measured: MK8D then failed inside rtld with `[rtld] Unresolved symbol:
  '__nnDetailInitLibc0'` followed by a jump to 0. Bisecting the encoding showed the trigger
  is the Code state on module text pages: with `Code 0x00dc7e03` (or the Alias/CodeData/Normal
  variants) rtld rejects its own module scan, and only the small type value lets it continue.
  The page does not document a 32-bit variant of `MemoryInfo`, so the runtime keeps the type
  byte until the A32 check is understood. `Attribute` (Locked/Uncached), `IpcRefCount` and
  `DeviceRefCount` are always zero because the runtime does not track them.
- **`MapMemory` attributes and range.** The runtime models the borrow as a copy, reports the
  destination as Stack, accepts any destination (not only the stack region) and only unmaps
  the exact `(dst, src, size)` triplet. The Locked attribute and partial unmaps are missing.
- **`SetMemoryAttribute`** is a no-op that always succeeds; the mask/value validation
  (`0x8`) and the cache attribute are not modelled.
- **`ThreadContext` on A32.** `GetThreadContext3` always writes the A64 0x320-byte layout.
  The page does not document the A32 struct, so it was left alone.
- **`SleepThread`** maps 0, -1 and -2 to a zero-nanosecond sleep; whether the scheduler
  treats those as the three documented yields has not been measured.
- **`GetInfo`** types 10 (idle ticks), 25 (thread ticks), 27 (IO region hint) and 29-38
  still return `0xF001`. The `GetProcessList`/`GetThreadList` self-process view is an
  extension: a retail console gates them on debug mode.
- **HIPC pointer buffers.** There is one process, so pointer (type 0x9) buffers are passed
  through with the client's addresses and no `ReceiveIndex`; the kernel's copy into the
  server's pointer buffer, the reply write-back and the `0x11A0B` size check are not
  modelled. ReceiveListCount values >= 3 (one buffer per index) are parsed but not routed.
- **HIPC/CMIF details.** Handle value 0 is treated as an ordinary invalid handle; the CMIF
  out-header version is 0 (the page only documents the in-header bump to 1 on 5.0.0+); the
  TIPC request id is encoded as `16 + cmd`, which is the runtime's convention.
- **Error codes.** `0xCE01` (OutOfResource) is not used for allocation failures; host
  refusals surface as `0x10801` or as a fatal stop.
- **ASLR, KASLR, kernel page tables, slab limits, IoRegister mappings and MC/PMC register
  state** are not modelled. The runtime is deterministic, single-process, and has no
  hardware behind the SMC.

## What the pages do not document

- The sizes and addresses of the 32-bit heap, map, stack and TLS regions beyond the 1.0.0
  ASLR pseudocode (map region 1 GiB at `0x4000_0000`, heap region after it).
- The A32 `ThreadContext` and any A32-specific `MemoryInfo`.
- The `PageInfo` bitfield returned in W1 of `QueryMemory`.
- The CMIF out-header version and the TIPC message-type base used above.

## Reproducing

```bash
CARGO_TARGET_DIR=…/kt-ws-wiki-core cargo test --release -p kirikae-rt
CARGO_TARGET_DIR=…/kt-ws-wiki-core cargo build --release -p kirikae
KIRIKAE_HEADLESS=1 KIRIKAE_TRACE_SVC=1 …/kirikae.exe run <mk8d exefs>
```

The title is AArch32, so `kirikae build` refuses it (`only recompiles AArch64`); the run
above is the interpreter path. The dump and its traces stay outside the repository.

## Correction after a real title (2026-10-25)

`GetInfo` type 22 was implemented as `used_memory - physical` from the page's
`SystemResourceSizeUsed` wording. SMBW then requested a heap 32 MiB smaller and aborted
while loading `System/HeapSize/Product.Nin_NX_NVN.HeapDef.byml.zs`. Reverting the whole
commit fixed it; bisecting the value showed the 16 MiB difference and the downstream
32 MiB request. Type 22 is now `used_memory - system_resource_size` again, with a test
that the two families stay distinct. The page text is ambiguous for a single process
with no physical mappings; the title is the stronger evidence.
