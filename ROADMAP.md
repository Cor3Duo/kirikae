# Roadmap

This roadmap describes where Kirikae is, what is being worked on, and what comes next.
It is a living document: open an issue or a discussion before starting a large item so
the work does not collide.

Status vocabulary:

- **done** — implemented, covered by tests and exercised by at least one title.
- **partial** — implemented but either not complete for the whole surface, or not yet
  exercised by a title.
- **next** — designed, not implemented.
- **research** — the shape of the solution is not settled yet.

## Where the project stands

Kirikae loads a decrypted dump, lays the modules out at their guest addresses, discovers
the reachable functions, lifts every instruction to a small IR and compiles it with
Cranelift. Whatever static analysis misses runs in an interpreter, and the addresses it
touched are written to a `.profile`; the next build compiles them. In practice one play
session is enough for the interpreter to disappear.

The runtime implements the Horizon kernel, the services, the Maxwell GPU (to Vulkan,
with shaders translated to SPIR-V), audio, video and input.

### Titles exercised so far

| Title | Build | Run |
| --- | --- | --- |
| Super Mario Bros. Wonder | single executable, ~490 MB | 30 fps, 0 interpreted instructions with a profile, no coverage gaps |
| Super Mario Odyssey | ~260 MB | ~10 s to the first frame with a profile, then 60 fps, no coverage gaps |
| Splatoon 3 | ~800 MB single executable | ~33 fps, compute dispatches, no coverage gaps |
| Super Mario Party Jamboree | ~390 MB | 35-38 fps and 4,896 draws/10 s, but the window stays black and it aborts on `brx` at ~frame 241 |
| Pokémon Brilliant Diamond / Shining Pearl | links as two code DLLs | no frame without a profile (Unity loading); ~30 fps after a profile-guided rebuild |
| Mario Kart 8 Deluxe, NSMBU Deluxe | — | AArch32: the interpreter runs 138 M instructions; VFP/NEON in progress |

These numbers come from headless runs on Windows with an RTX 3080. They prove the
pipeline, not visual correctness: image comparison is still manual.

## Workstreams

### 1. Recompiler quality (discovery and linking)

**Done**

- `.eh_frame_hdr`/FDE seeds, indirect-call targets and profile seeds.
- One block segment per entry: each address is lifted once. On Brilliant Diamond this
  removed a 5.6x duplication (74.6 M lifted instructions for 13.3 M reachable) and took
  the build from 490 s and two 1.4 GB DLLs to 66 s and one 630 MB executable.
- Sources over the PE 2 GiB limit are split into one executable plus size-balanced code
  DLLs with a bound entry table, and the default target directory is short enough for
  the MSVC linker (the previous failure was a 261-character path, not the image size).

**Next**

- Jump tables reached through `br`/`brx` that the heuristic does not resolve. They fall
  to the interpreter today; the profile loop (`kirikae build --profile-loop`) recovers
  them, and Brilliant Diamond needed three iterations and 37 addresses to reach zero
  interpreted instructions.
- A per-DLL jump table for cross-segment branches: deduplication made every
  cross-boundary branch a dispatcher lookup, which is correct but not free.
- Jump tables are now recognised in more shapes (index register inferred, register-held
  bounds, unbounded tables); 2.3 k (SMBW) and 5.8 k (BD) static instructions remain
  reachable only through indirect branch sites.
- Function-level sections (`function_sections`) to let the linker drop unreachable
  blocks, and `/OPT:ICF` once symbols allow it.

### 2. AArch32 frontend

**Partial**

- The loader detects 32-bit titles from `META.Flags0` bit 0 (`Is64BitInstruction`) and
  rejects them with an explicit message; ACID flags are not architecture.
- An AArch32 decoder and IR exist for ARM mode: condition field, data-processing,
  moves, branches, loads/stores, block transfers, multiplies and `svc`/`bkpt`, with
  typed rejections. It is validated against Capstone over 1.08 M random words with no
  false acceptances.
- The interpreter executes that slice end to end with a 32-bit address layout, including
  the media/parallel classes and byte/halfword transfer sizes. Mario Kart 8 Deluxe now
  runs 137 M interpreted instructions (its `rtld` hash loop no longer spins) and stops on
  the first VFP/NEON instruction.

**Next** (largest single workstream)

- VFP/NEON (the current wall), then the remaining coprocessor space.
- Thumb-2, if measurement on the two titles shows it is needed; VFP/NEON is the largest
  unknown.
- Function discovery, predication in the IR, and lowering to Cranelift (or the Rust
  backend) with the A32 register file.

### 3. NVN host

**Partial**

- NVN objects: pools (including virtual), buffers, textures, views and samplers, with
  generations and 40-bit guest VAs.
- Command recording: lists are recorded into guest command memory and the recorded
  lists are replayed into Vulkan in submission order, with per-submit snapshots,
  `Call`/`Copy` handling and the Maxwell pipeline helpers.
- Lists the driver writes without `BeginRecording` are captured in a separate host
  arena keyed by the handle the submit names, so replay sees the game's per-frame
  lists without touching the driver's pushbuffer.

**Next**

- Present: blit the replayed window render target to the swapchain and present in submit
  order, so `KIRIKAE_NVN=record` produces frames instead of only draws.
- Temporal state: read texture and pool rows at submit time rather than at replay time.
- Storage buffers and images, instanced vertex streams, and compute dispatch replay.
- Shader `brx` (indirect branch) inside shaders: those draws are skipped today.

### 4. Maxwell GPU

**Partial**

- 112 of 146 mnemonics translate to SPIR-V with naga validation in the tests, including
  the texture forms (`tex`, `tld`, `txd`, `tld4`, bindless LOD, per-texel offsets and
  depth compare), the integer, video, half2 and control groups, and the bindless handle
  sources resolved through register copies and constant-bank loads.
- Draw state, pipelines, descriptors, render passes, clears and compute dispatches are
  driven from the decoded command stream.

**Next**

- Surface and atomics (`suld`, `sured`, `suatom`, `atoms`, `txa`, `pixld`, `al2p`).
- FP64 is intentionally labelled: the pipeline does not declare `Float64`, and enabling
  it would make 64-bit floating point a hard device requirement for every shader.
- Barriers and the remaining control forms, plus geometry and tessellation stages.
- Image comparison tooling: coverage is not correctness.

### 5. Kernel and services

**Kernel: complete surface (124 of 124 SVCs)**

- Threads, scheduler, events, arbiters, memory, mapping, transfer/code memory,
  sessions (create/accept/reply), async IPC, resource limits and guest exceptions
  (`svcBreak` reports the guest assert message and backtrace, `ReturnFromException`).
- Next: the device address space family (which needs a device physical map the runtime
  does not have yet), physical memory pools, process lifecycle and the debug SVCs.

**Services: partial (55 % of the documented command surface)**

- Account, applet manager (application/self/common state, display, winding, debug,
  `appletAE` and every system proxy, home menu, global state, cradle, application
  creator/accessor), audio (`audren` effects 1-9, real Opus including multistream,
  six-channel output), BCAT with a delivery cache backed by save data, filesystem with
  journaled save data, HID, NIFM/BSD/SSL handshakes, `pctl`, `aoc`, `prepo`, `time`,
  `csrng`, `irs`, `vi` managers, `sm` with guest registration and typed IPC.
- Next: the remaining `am` proxies (`IHomeMenuFunctions`, `IGlobalStateController`,
  `IApplicationCreator`), `sm` domains and session closing, BCAT transfer, device and
  cache saves, and the system-only services titles rarely reach.

### 6. Audio and video

**Done**

- `audren` effects 1-8 and the compressor (9), limiter statistics, real Opus
  (mono/stereo and multistream up to 5.1), and a documented 5.1 downmix or a real
  six-channel stream when the device supports it.
- H.264 and VP9 decoding with 8/10-bit surfaces (NV12/P010), VIC composition with
  scaling/rotation/alpha and RGBA/I420/P010 outputs, VI with YUV layer formats.
- A shared presentation clock for video: NVDEC -> VIC -> VI is paced against the audio
  clock (or a monotonic clock when muted), with drop/repeat limits, seek handling and
  drift counters in `KIRIKAE_STATS`.

**Next**

- VP8 and AV1 backends. AV1 is not in the Tegra X1 NVDEC, so this is for completeness of
  the command surface, not for this hardware.
- 10-bit H.264 decoding needs an FFmpeg-class backend; the surface and conversion paths
  exist.
- Real A/V validation with a title that plays video, and audio latency compensation.

### 7. Tooling and diagnostics

**Done**

- `kirikae info` (modules, segments, decoder coverage).
- Coverage files (`KIRIKAE_COVERAGE`) with per-interface status, and the profile loop
  (`KIRIKAE_PROFILE`) that feeds the next build.
- A batch runner with a bounded boot phase, per-game timeouts, coverage and a
  compatibility matrix; triage and a progress dashboard generator.

**Next**

- Image comparison (golden frames) and a deterministic input script format.
- Automatic per-game profile iteration in the batch runner.
- A dashboard section generated from the batch matrix (schema already emitted).

## Visibility (2026-10-25)

Every extracted title was built, booted headless and its presented frame classified
(`KIRIKAE_CAPTURE` + `tools/frame_diff.py`). Ties are "the frame has visible content",
not "the game is playable".

| Result | Titles |
| --- | --- |
| Visible | Super Mario Bros. Wonder, Super Mario Odyssey, New Super Lucky's Tale, Pokémon Mystery Dungeon DX, Zelda: Breath of the Wild, Zelda: Tears of the Kingdom |
| Black but running | Super Mario Party Jamboree (draws, but the target stays cleared), Pokémon Sword, Pokémon Shield |
| No frame in 100 s | Brilliant Diamond / Shining Pearl (Unity loading without a profile) |
| Crashes before a frame | Splatoon 3, Smash Ultimate, 3D All-Stars, Final Fantasy Resonance, Onimusha 2, Legends Arceus/Z-A, Scarlet/Violet, Let's Go Eevee/Pikachu |
| AArch32 | Mario Kart 8 Deluxe, NSMBU Deluxe |

The crash list is the next triage queue: several titles stop on `Sha1Impl::ProcessBlock`
(the ARMv8 SHA-1 instructions are not implemented yet), one on the socket `GetHErrno`,
and the rest are SDK aborts during initialisation.

## Compatibility matrix

The matrix is generated by `tools/discovery/batch.py`; game data and artifacts stay
outside the repository.

| Title | Compiles | Boots | Coverage gaps |
| --- | --- | --- | --- |
| Super Mario Bros. Wonder | yes | yes, 30 fps | 0 |
| Super Mario Odyssey | yes | yes, 60 fps | 0 |
| Splatoon 3 | yes | yes | 0 |
| Super Mario Party Jamboree | yes | 35-38 fps, 1,024x1080 | 0 services (5 shader forms) |
| Pokémon Brilliant Diamond / Shining Pearl | yes (split) | ~30 fps after a profile-guided rebuild | `tex` forms |
| Pokémon Legends Arceus, Z-A, Scarlet, Violet, Sword, Shield, Let's Go | yes | not measured | — |
| Super Smash Bros. Ultimate, BotW, TotK, 3D All-Stars | yes | not measured | — |
| Mario Kart 8 Deluxe, NSMBU Deluxe | AArch32 | — | — |

"Boots" means a headless run reached the first presented frame or the SDK
initialisation; it does not mean the game is playable.

## How to help

- Pick an item marked **next** and open an issue describing the title or the interface
  you are going to exercise.
- Every behavioural change needs a test that fails before and passes after.
- Services, SVCs and ioctls must follow the documented Horizon semantics; when the
  hardware behaviour is unclear, stop and ask instead of guessing.
- Game data, keys, firmware, shaders, textures and captures never enter the repository.
- `docs/progress` is generated after merge, not in pull requests.

## Known limitations

- No image has been compared against hardware yet; "no coverage gaps" means no
  unimplemented interface was hit, not that the output is correct.
- AArch32 titles cannot run yet: the interpreter covers ARM, VFP and most of NEON, but
  there is no backend and no discovery for them.
- Firmware/SDK versions are not modelled: services accept the superset of commands and
  answer with the newest documented shape. A per-title version model is the next
  foundation item.
- Titles not listed above have only been compiled.
- The interpreter is slow by design: a first run without a profile pays for it.
- The split executable is a workaround for the PE 2 GiB limit; deduplication removed
  the need for it on the current titles, but a future title may need it again.
