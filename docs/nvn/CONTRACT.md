# NVN on the host: the contract

Working note for WS4. It records what a host implementation of NVN has to preserve, anchored to the
MIT-licensed smwr tree (a local checkout, read-only reference) and to what
`crates/rt/src/nvn/` already models. Public Switchbrew pages are named where a definition is generic
(`nvnBootstrapLoader`, `NVNdevice`, `NVNmemoryPool`, `NVNbuffer`, `NVNtexture`, `NVNcommandBuffer`,
`NVNsync`, `NVNwindow`). Nothing here is game data.

The short version: implement the **API**, not the Maxwell command processor. The game's structs stay
guest memory; the recompiled SDK driver is the authority on their layout; our job is to give each
proc a host meaning, record commands into guest command memory, and replay them into Vulkan.

## 1. The boundary

One title imports exactly two NVN symbols, `nvnBootstrapLoader` and `nvnPrivate0`
(`docs/nvn/WONDER.md:43`). `nvnBootstrapLoader(name)` returns a generic proc pointer, and the game
then asks `nvnDeviceGetProcAddress(device, name)` for every other function, the same shape as
`vkGetInstanceProcAddr` (`docs/NVN.md:17-21`, `runtime/src/nvn/nvn_trace.cpp:97-118`).

- The name set is finite: 536 distinct `nvn*` strings in one title's `main`, broken down per object
  (149 CommandBuffer, 71 Texture, 40 Sampler, 33 Device, 33 Queue, 24 Window, ...)
  `docs/nvn/WONDER.md:63-70`. Generate the lookup table from those strings; do not hand-write 536 arms.
- Resolution order in smwr (`nvn_trace.cpp:58-95`): an implemented HLE proc; otherwise a pure struct
  proc (builders, state objects, texture views) served by the **recompiled SDK driver**; otherwise a
  tracing stub. kirikae already has the same three-way split in `crates/rt/src/nvn/mod.rs:50-105`.
- A stub returns 0 but `*Initialize*` returns 1 (NVNboolean), so the engine does not give up at the
  first object (`nvn_trace.cpp:74-77`). Trace the first 32 calls with arguments and LR; kirikae only
  counts calls today (`mod.rs:107-135`).
- `nvnPrivate0` is the bootstrap's second entry point; treat it as an alias unless the trace shows
  otherwise. Not confirmed in smwr either.

The boundary rule: a stub is allowed only while it is visibly missing. No stub may silently change
control flow, and every stub event is reported per frame (`shadow::unsupported` already does this).

## 2. Object model and ownership

The game allocates every NVN object itself, at the real driver's size; we never redefine those
structs. Builders and state objects are plain structs the SDK's driver fills and reads, so their procs
go straight to the recompiled driver and we read values back with its **getters** at recording time.
That is the "SDK driver as struct oracle" pattern (`runtime/src/nvn/nvn.h:1-9`, `nvn_trace.cpp:79-85`);
the same getters supply enums and computed values (storage size, alignment).

Host objects are keyed by the guest object's address in one table. `make()` publishes a new object and
`destroy()` removes it; both **defer** destruction, so an object is freed only when no replay can still
name it (`nvn.h:53-163`, `nvn_trace.cpp:183-208`).

| Guest object | Host object holds | Source |
| --- | --- | --- |
| `NVNmemoryPool` | storage VA, size, flags, GPU VA, virtual mappings | `nvn_objects.cpp:99-167`, `nvn.h:167-177` |
| `NVNbuffer` | pool + offset, host pointer, GPU VA, size | `nvn_objects.cpp:199-246` |
| `NVNtexture` | builder fields, storage size, block height, swizzle, host image | `nvn_objects.cpp:268-340` |
| `NVNtextureView` | base level/count, min layer/count, format, swizzle | `nvn_objects.cpp:68-82` |
| `NVNsampler` | min/mag filter, wrap[3], compare, lod range/bias, aniso, border | `nvn.h:229-232`, `nvn_sampler_objects.cpp` |
| `NVNtexturePool` / `NVNsamplerPool` | id → texture, snapshot, view / `SamplerInfo` | `nvn.h:242-249` |
| `NVNprogram` | stage list; SHA-1 of each code block → `ShaderInfo` | `nvn_shaders.h:207-223` |
| `NVNwindow` | texture list, active count, present interval, crop | `nvn.h:221-225`, `nvn_queue.cpp:184-209` |
| `NVNqueue`, `NVNsync`, `NVNevent` | submit sequence, fence/timeline, signaled flag | `nvn_queue.cpp:98-124` |

**Pools.** A pool exposes a *pitch-linear* GPU address, a *block-linear* one and a *compressed* one;
which one a texture uses depends on its flags. Virtual pools own no storage: `nvnMemoryPoolMapVirtual`
maps `[virtualOffset, +size)` onto `(physical pool, physicalOffset)` in 0x28-byte requests, and host
resolution walks those mappings (`nvn_objects.cpp:84-93, 142-167`). GPU addresses are 40-bit, 64 KiB
aligned, with a guard gap; 0 and low values stay invalid (`nvn_objects.cpp:25-34`).

**Descriptor pools.** A texture handle packs a 20-bit texture id and a 12-bit sampler id
(`nvn_sets.cpp:35-37`). Switchbrew treats the handle as opaque; keep the split as a decode rule, not
as an API guarantee, and validate it per title.

**Slots and snapshots.** Registering a texture in a pool must preserve what the slot named at
registration. smwr keeps the texture and a copy/snapshot plus the view (`nvn.h:242-249`); re-init of
the same guest struct must not rewrite an old slot (`nvn_objects.cpp` `keep_for_slots`,
`hand_image_to_snapshots`, DEBUG-2 #63). One title re-initializes ~180 texture structs per frame, so
this is not an edge case.

## 3. Command recording

A command buffer is guest memory the game hands over with
`AddCommandMemory(pool, offset, size)` / `AddControlMemory(ptr, size)`. Recording appends to a write
pointer. kirikae's shadow already reads that pointer from the guest header: write count at `+0x18`,
base at `+0x50`, GPU base at `+0x60`, position = `gpu_base + write - base`
(`crates/rt/src/nvn/shadow.rs:339-346`).

Contract:

- **Fixed-layout records.** Write one tagged, 8/16-byte-aligned record per command at the write
  pointer: op tag, small counts, up to 9 64-bit words, 4 floats, 4 ints, and an index into a side blob
  for decoded state or inline data. This is the shape of smwr's host-side `Cmd`
  (`runtime/src/nvn/nvn.h:299-316`) and of kirikae's `Cmd` (`shadow.rs:55-74`). Do not allocate per
  command.
- **Handle = guest address.** `EndRecording` returns the address of the list header in command
  memory. The game treats a handle as opaque, so the value is ours; the lifetime is not. smwr proves
  the lifetime rule: a newer list recorded over the same bytes drops the older one, and finalizing a
  pool frees every list recorded in it (`nvn_cmd.cpp:47-88, 244-300, 449-468`). smwr estimates usage
  at 16 bytes/command and 4 bytes/control (`nvn_cmd.cpp:41-45`); storing the records in guest memory
  makes the rule exact instead of estimated, and gives the handle its natural lifetime.
- **Copy vs call.** `CallCommands` appends a reference (the handle). `CopyCommands` must append a
  *copy* of the callee's records, so overwriting the callee's memory does not change the copy. smwr
  does not implement it; kirikae's shadow maps both to `CallCommands` (`shadow.rs:406, 528`) — the one
  semantics point still unconfirmed (§10).
- **State by value.** Every `Bind*State` and every setter decodes through the driver's getters and
  stores the *values* in the record or its blob: the game may reuse its struct immediately after the
  call (`nvn.h:295-297`, `nvn_cmd.cpp:304-319`). Never keep a pointer to a builder.
- **Deferred binds.** `BindTextures*Deferred`, `BindUniformBuffers*Deferred`, etc. carry the address of
  an array. Store address + count + element width, and snapshot the array **at submit**
  (`nvn_binds.cpp:33-60, 193-213`); reading it late without a snapshot picked up two-frames-old handles
  (DEBUG.md #41).
- **Clears keep their masks.** Record value, target index, channel mask, depth/stencil masks and the
  current scissor; the pass decision happens at replay (`nvn_cmd.cpp:321-335`).
- **Discard.** Finalizing a command buffer or a pool invalidates its handles; never reuse a handle after
  its memory is gone. A handle can be submitted many times while it lives.

## 4. Replay

- **One worker, FIFO.** A single replay thread with a job queue, not a pool: NVN submission order must
  equal Vulkan submission order (`nvn_worker.cpp:1-6, 58-88`). Inline execution stays a debug switch.
- **One `VkCommandBuffer` per submit**, split into at most a few parts when the submit contains
  compute dispatches. Record, queue, fence; a slot is reusable only after its fence
  (`nvn_frames.cpp:54-78, 209-257`). Keep the per-part fence array and the upload-ring tail.
- **CallList nesting**: replay recursively, depth limit 8, and cache a handle's list for the duration
  of the submit so a list re-recorded meanwhile is not mixed in (`nvn_replay.cpp:14-36, 64-77`).
- **Targets and passes**: `SetRenderTargets` only updates state. A pass begins at the next draw/clear
  that needs it and ends when a draw, clear or copy needs other targets (`nvn_replay.cpp:51-60`,
  `nvn_frames.cpp:283-295`). Use dynamic rendering; a pre-baked render pass is only useful as a cache.
- **Clears**: a full-target clear becomes the next pass's `loadOp = CLEAR`. A scissored clear, or a
  channel-masked colour clear, stays inside the pass: `vkCmdClearAttachments` on the scissor rect, or
  a fullscreen draw with `colorWriteMask` for partial channels (`nvn_clears.cpp:1-10, 196-290`,
  `nvn_mask_clear.cpp`). Depth/stencil aspect masks are independent.
- **Barriers**: start every part with an all-commands memory barrier, and end the submit with a
  `MEMORY_WRITE → HOST_READ` barrier so compute write-backs are visible
  (`nvn_frames.cpp:209-257`).
- **Dynamic state** for viewport, scissor, blend constants, stencil ref/masks, depth bounds, depth
  bias, line width and cull where the device supports it; the rest goes in the pipeline. smwr's key is
  a byte-comparable struct (shader pointers, attributes/streams, blend per target, colour/channel/
  depth/polygon state, target formats, alpha-to-coverage, depth clamp), ~704 bytes, 4-lane hash, never
  evicted (`nvn_pipeline.h:17-79`, `nvn_pipeline.cpp:253-262`). The size is `sizeof(PipelineKey)`, not
  an API constant.

## 5. Shader bank contract

Every stage is Maxwell machine code plus a control section. smwr translated all of one title's 12,717
blocks offline and verified the bank layout against the reflection of all 47,797 stages
(`docs/nvn/TRANSLATOR.md:250-262`):

| Bank | Content | Where |
| --- | --- | --- |
| `c[0]` | driver constants: storage-buffer descriptors, and compute uniform descriptors | graphics `0x110 + 0x100*stage + 0x10*n` (vertex stage 0, fragment 4); compute uniform `0x210 + 0x10*n`, storage `0x310 + 0x10*n`. A descriptor is `{u64 GPU address, u32 size}`. Max point size at `0x610` |
| `c[1]` | the block's constant data | control `+0x700` offset, `+0x6fc` size, 0x100-aligned |
| `c[2]` | texture handles | sampler binding *n* = the 64-bit word at byte `0x20 + 8n`; storage images at `0x120 + 8n` (inferred) |
| `c[3+n]` | uniform block *n* (graphics) | what `BindUniformBuffer(stage, n)` bound |

Compute shaders reach uniform blocks through the `c[0]` descriptor table and the translator turns them
into storage buffers; agl/gsys compute reads `c[3]` directly. Bindless is the exception: the game writes
the handle into a uniform block, typically `c[7]` word 32, so the replay reads that word from the bound
UBO (`TRANSLATOR.md:269-273`, `nvn_shaders.h:88-109`).

The support buffer (set 0, binding 0, std140, 0x570 bytes) is shared by all stages:
`alpha_test` 0x000, `is_bgra[8]` 0x010, `viewport_inverse` 0x090, `viewport_size` 0x0a0,
`frag_scale_count` 0x0b0, `render_scale[73]` 0x0c0, `tfe_offset` 0x550, `tfe_vertex_count` 0x560.
With render scale 1 and no transform feedback most reads are identities
(`TRANSLATOR.md:415-420`, `nvn_shaders.h:184-203`).

Required SPIR-V reflection per stage: stage and local size; every descriptor with set/binding/type,
array count, image shape (dim, depth, arrayed, MS, sampled, format), and its **NVN meaning** (bank,
binding, word or `c[0]` byte offset); vertex inputs and fragment outputs with location/component type;
built-ins; whether the module uses the support buffer (`nvn_shaders.h:60-140`). The SPIR-V does not
carry a storage buffer's `c[0]` offset; smwr reads it from the harness's `results.tsv` and serializes it
into the shader cache (`nvn_shaders.h:145-168`). Cache key: SHA-1 of the whole code block, first 16 hex
digits; load once when the program is set, not per draw (`nvn_shaders.h:205-223`).

## 6. Texture layout

The format table comes from the driver binary, not from a hand-written enum: SDK 16.2 has it at
`sdk+0xd34198`, 0x30 bytes per format, 0xd0 entries; word 2 byte 0 = block width, byte 1 = block height,
word 4 byte 0 = bytes per block (`nvn_texture_layout.cpp:5-6, 19-38`). Treat the offset as
version-specific: derive it per SDK build ID or verify the table against known formats before use.

Maxwell block-linear, in blocks (one texel for plain formats, 4x4 or ASTC NxM for compressed):

- GOB = 64 bytes x 8 rows = 512 bytes; byte at (x, y) is
  `(x/32)*256 + (y/2)*64 + ((x/16)%2)*32 + (y%2)*16 + x%16` (`docs/nvn/TILING.md:13-21`).
- Level size = `div_up(row_bytes, 64) * div_up(rows, 8*block_height) * 512 * block_height`.
- Default block height = smallest power of two ≥ `rows/8`, at most 16 GOBs; each smaller mip uses
  `min(level0, smallest power of two ≥ level_rows/8)` (`nvn_texture_layout.cpp:42-47, 68-77`).
- Array layers are one mip chain apart, aligned to `512 * block_height0`; levels are back to back with
  no extra alignment (`TILING.md:23-31`, `nvn_texture_layout.cpp:78-82`).
- BNTX carries its own block height: `block_height = 1 << (BRTI+0x34 & 7)`. Across one title's 13,171
  block-linear textures it equals the default for 12,731 and is one step smaller for 440, so packaged
  textures must be untiled with the **file's** value (`TILING.md:54-69`). The builder's packaged-layout
  field reads 1 for every texture in that title; smwr treats a non-zero value as "packaged" and uses
  the fit `smallest power of two ≥ rows*3/32, at most 16` (`nvn_objects.cpp:303-315`).
- Untiling walks one GOB at a time, source order, no per-texel division; ~0.4 ms for 1920x1080 RGBA8,
  ~2 ms for a 2048² BC7 chain on an M1 (`TILING.md:46-52`).
- Colour space and compression: the **view's** format decides sRGB, not the texture's (view format 0
  means the texture's own; `nvn_objects.cpp:68-82`, `nvn.h:236-239`). ASTC is native when the device has
  `textureCompressionASTC_LDR`, otherwise decoded to RGBA8 at upload (linear and sRGB halves
  separately); BC1–BC7 are native on desktop (`nvn_format.cpp:76-109`, `nvn_upload.cpp` ASTC path).
- Uploads go through a ring owned by the replay slot; a large or scene-load upload gets its own
  staging buffer instead (over a quarter of the ring in one submit, or one upload over an eighth)
  (`nvn_upload.cpp` `staging_wanted`; `nvn_frames.cpp:104-108`). Copies are bounds-checked against what
  was staged; a short upload clears the image instead of reading another upload (`copies_fit`).

Pitch-linear textures (`stride` set) need no untiling; their size is
`stride * div_up(height, block_h) * depth`, alignment 0x100 (`nvn_texture_layout.cpp:63-67`).

## 7. Temporal model

The game records frame N+1 while frame N is replayed (98.5% of one title's pool registrations happen
during a replay, DEBUG.md #56). Three mechanisms make that safe:

- **Incarnations.** A texture re-initialization or finalization records the object it retires with a new
  epoch; a replay resolves each guest address at its mark, and retired objects live until no replay can
  name them (`nvn_incarnation.cpp:1-14, 48-74, 141-163`).
- **Mark = end of recording**, not start of submit: each list is stamped when recording ends and a
  submit takes the newest stamp, so a recording thread that runs ahead is not included (DEBUG.md #57).
- **Pool snapshots.** A slot change records the entry it replaces; a slot or object that was empty at
  the mark resolves to the current one, because NVN reads descriptors when the GPU runs (DEBUG-2
  #63, #64).

Only compute results flow back to guest memory. Copy back only the words the shader wrote: keep the
guest bytes a range was made from and compare after the fence, so a concurrent CPU write in the same
range is not overwritten (`DEBUG.md:593-617`, `apply_write_back`); whole-range copies are the old,
wrong behaviour. Fences use two counters (submits called, done): a fence records the submits made
before it, while `nvnQueueFinish`, pool invalidate and pool finalize wait for everything made so far
(DEBUG-2 #59, `nvn_queue.cpp:132-179`). A wait that blocks the game's thread is correctness, not a
performance detail.

## 8. Present

`nvnQueuePresentTexture(queue, window, index)` presents the window's texture at that index and
`nvnWindowSetCrop` selects the part shown (`nvn_queue.cpp:384-392`). The host present pass rotates by
the swapchain's `preTransform` (a pre-rotated image goes to the display as is), fits with bars (16:9 on
a wider screen keeps its shape), writes alpha 1 (a surface may treat alpha as transparency), and crops;
handheld renders 1280x720 into 1920x1080 window textures and crops to it
(`runtime/src/nvn/nvn_present.cpp:1-8, 346-381`).

Pacing: a present returns no earlier than `presentInterval` vsyncs (16.666 ms each) after the previous
one, without snapping a late frame to the next vsync. Prefer MAILBOX, then IMMEDIATE, then FIFO, so the
game's vsync clock is not fighting the display's (DEBUG-2 #62, `nvn_queue.cpp:340-357`); queue at most
two presents ahead of the replay.

Kirikae's host present (`nvn/window.rs`, `nvn/host.rs`, `gpu/vulkan/nvn.rs`) records
`WindowBuilderSetTextures`, `WindowSetCrop` and the present interval by value, runs the **driver's**
`nvnQueuePresentTexture`/`nvnWindowPresentTexture` first (its fences, vsync clock and native-window
queue stay its own), then waits for the submit serial the frame's replay was queued under and blits the
window texture's replayed image into the swapchain, fitted with bars. A present whose window or texture
the host never recorded, or whose replay has no image, stays with the driver and is counted as a
fallback: the frame is never dropped. The counter is `display::FRAMES`, the same one `trace` uses; the
host credits the present only when the driver's own present did not advance it.

Measured 2026-10-25 (SMBW, `KIRIKAE_NVN=record`, 20 s): 989 presents, every one resolved its window
texture and submit; 25,141 draws were recorded and 20,027 replayed, with 0 skipped. No present found a
replayed image, so all 989 stayed with the driver. The window pass binds every window texture as a
render target, then a driver-internal `EndRecording` splits its state from the draw that follows: the
host holds the internal-pointer chunk as pending and drops it when the next game handle arrives, so the
window target is set but never drawn. Keeping that chunk and merging it into the game handle, in
recording order, is the next gap; until then the driver keeps the frame.

## 9. Mapping to kirikae

| Piece | Where | Reuse |
| --- | --- | --- |
| Bootstrap interception and HLE thunks | `mod.rs:23-105` | keep; add per-name argument tracing |
| Command model and recording hooks | `shadow.rs:55-74, 353-453, 488-716` | promote `Cmd` to the real record; keep the hook table as the op map |
| Command-memory position and list ranges | `shadow.rs:339-346, 497-527`, `SEGMENTS` | keep the position math; replace the estimate with real records |
| Pools, buffers, textures, views | `resources.rs` | keep the field decode (builder getters, block height, view offsets); add host storage |
| Render state vocabulary | `state.rs` | keep; it is the NVN→Vulkan state mapping the pipeline key needs |
| Differential oracle | `shadow.rs` model vs `gpu/threed.rs` decode, `encode.rs` | keep running while the host path is built; it is the cheapest regression test |
| Vulkan side | `gpu/vulkan/{pipeline,tiling,images,submit,present}.rs` | reuse the device, ring, cache and untiling; add an NVN replay that feeds them |

Missing for a host implementation, in order:

1. Host object registry keyed by guest address, with deferred destruction and **incarnations/epochs**.
2. GPU VA allocator + pool registry; virtual pools; pool→host memory (import guest pages with
   `VK_EXT_external_memory_host` where available, otherwise copy through the ring).
3. Real command records **in guest command memory**, handle = address, copy/call/discard rules.
4. NVN replay worker: one command buffer per submit, CallList, dynamic rendering, deferred/masked
   clears, pass barriers.
5. NVN format table + storage layout + untiling + sRGB views + ASTC/BC fallback + upload ring.
6. Pipeline key/cache from decoded NVN state and reflected SPIR-V; `c[0..3+n]` bindings; support buffer.
7. Incarnation/pool-slot marks and selective compute write-back.
8. Present pass with preTransform, bars, alpha 1, crop and pacing.

Milestones (from `docs/plan/MASTER-PLAN.md:395-414`) and the main risk of each:

| Milestone | Deliverable | Risk |
| --- | --- | --- |
| 4.1 Interception | name table, HLE registration, tracing thunks for the rest | a title asks for a proc whose signature is wrong; keep stubs neutral and trace args |
| 4.2 Objects | device/queue/sync/event, pools incl. virtual, buffers, textures/views, samplers, descriptor pools, programs, windows | object struct size/field assumptions; always go through the driver's getters |
| 4.3 Recording | fixed records in guest memory, handle = address, state decoded at record time, copy/call | `CopyCommands` semantics unconfirmed; overflow/usage accounting differs from the real driver |
| 4.4 Replay | one worker, one command buffer per submit, CallList, deferred/masked clears, pass barriers | pass count explodes if targets break too eagerly; measure with `KIRIKAE_STATS` |
| 4.5 Textures | format table, GOB layout, BNTX block height, untiling, sRGB, ASTC/BC, upload ring | format-table offset is SDK-version-specific; small BNTX block heights garble if ignored |
| 4.6 Pipelines | 704-byte key, disk cache, sets 0..3+n, dynamic state | variant explosion from attribute/output types; bake the defaults the title uses |
| 4.7 Temporal | epochs/incarnations, pool snapshots, selective write-back | a missed mark shows as one-frame-old textures; expensive to diagnose |
| 4.8 Present | preTransform, bars, alpha 1, crop, pacing | rotation/crop mistakes look like gameplay bugs; test handheld and docked |

Ordering note: 4.1–4.3 are cheap and unblock a trace-driven 4.4; 4.5/4.6 can run in parallel once the
first clear presents; do not start 4.7 until 4.4 replays two frames, since its bug needs recording ahead.

## 10. Sources and license

Consulted in smwr (MIT, local checkout): `docs/NVN.md`,
`docs/nvn/WONDER.md`, `docs/nvn/TRANSLATOR.md`, `docs/nvn/TILING.md`, `docs/nvn/DEBUG.md`
(#41, #55–#57), `docs/nvn/DEBUG-2.md` (#59, #61–#64); and under `runtime/src/nvn/`: `nvn.h`,
`nvn_trace.cpp`, `nvn_objects.cpp`, `nvn_cmd.cpp`, `nvn_replay.cpp`, `nvn_worker.cpp`,
`nvn_frames.cpp`, `nvn_incarnation.cpp`, `nvn_queue.cpp`, `nvn_binds.cpp`, `nvn_sets.cpp`,
`nvn_clears.cpp`, `nvn_mask_clear.cpp`, `nvn_present.cpp`, `nvn_texture_layout.cpp`,
`nvn_format.cpp`, `nvn_upload.cpp`, `nvn_pipeline.{h,cpp}`, `nvn_shaders.h`,
`nvn_sampler_objects.cpp`. kirikae side: `crates/rt/src/nvn/{mod,shadow,resources,state,encode}.rs`,
`crates/rt/src/hle.rs`, `crates/rt/src/gpu/vulkan/*`, `docs/plan/MASTER-PLAN.md`.

MIT attribution to keep in any derived file or doc section: smwr, MIT License, Copyright (c) 2026
BelmanteGu. Ryujinx's support-buffer layout and binding scheme are also MIT (Copyright (c) Ryujinx Team
and Contributors). No code is copied here; identifiers and measurements only.

Not confirmed: `CopyCommands` copy semantics (smwr does not implement it); `nvnPrivate0`; the exact
meaning of the virtual-pool mapping request's storage class; whether the descriptor-handle bit split
holds on every title; the format-table offset for SDK builds other than 16.2.
