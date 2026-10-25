# Architecture

A map of the code base for contributors: what each crate does, where things live, and how a frame
travels from the game to the screen. The code keeps comments to the required ones, so this page is the
place to start.

## The big picture

```
                 build time                                         run time
 ┌──────────────────────────────────────────────┐   ┌──────────────────────────────────────────────┐
 │ loader → discover → lift → clif → cargo      │   │ game.exe                                     │
 │                                              │   │  ├─ recompiled functions (native code)       │
 │ the dump becomes game.exe:                   │──►│  ├─ kirikae-rt: memory, threads, kernel,     │
 │ native code + kirikae-rt + the dump embedded │   │  │   services, GPU, audio, video, input      │
 └──────────────────────────────────────────────┘   │  └─ interpreter for code not found at build  │
                                                    └──────────────────────────────────────────────┘
```

## Crates

| Crate | Role | Start reading at |
| --- | --- | --- |
| `loader` | Reads NRO, NSO (LZ4), PFS0, NCA, RomFS and the folder layout into an `Executable`: modules laid out at their guest addresses (`LOAD_BASE` = `0x0800_0000`), the title ID and where the RomFS is. Also packs the source that gets embedded into `game.exe`, and builds RomFS images from folders. | `package.rs` (`read_source`, `from_exefs`, `pack`) |
| `a64` | ARM64 decoding. `lift.rs` / `lift_fp.rs` / `lift_neon.rs` turn one instruction into the small IR of `ir.rs`; every operation is a function in `ops.rs`. `simd/` holds the SIMD/FP semantics and an oracle to compare against a real Cortex-A57 model. | `ir.rs`, then `lift.rs` |
| `recomp` | Function discovery (`discover.rs`: module entries, `.eh_frame_hdr`/FDE bounds from `ehframe.rs`, relocations, dynamic symbols, branch targets, ADRP+ADD), code generation with Cranelift (`clif.rs`, 1024 functions per object, in parallel) or as Rust source (`rust_backend.rs`, the reference), and the generated Cargo project (`project.rs`). | `project.rs` (`build`) |
| `rt` | The runtime linked into every game. Everything a game needs from the console. | see below |
| `maxwell` | Maxwell (SM 5.x) shader decoding (`decode.rs`, `opcodes.rs` from envytools), control flow (`cfg.rs`), the shader program header (`sph.rs`) and the SPIR-V translator (`translate.rs`). | `translate.rs` |
| `kirikae` | The command line tool. | `main.rs` |
| `gui` | The egui front end; it runs `kirikae.exe` and can take converters and plugins from a wrapper binary. | `lib.rs` |
| `testkit` | Builds small synthetic NROs for the end-to-end tests. | `lib.rs` |

## The runtime (`crates/rt/src`)

### CPU and memory

- `mem.rs`: reserves the whole 39-bit guest address space once; every guest access is
  `base + (addr & mask)`, so a bad pointer faults instead of touching host memory. Also the
  `svcMapPhysicalMemory` regions.
- `cpu.rs`: the `Cpu` state (X/V registers, flags, TLS, exclusive monitor) shared with generated code;
  `abi.rs` holds the field offsets and the helpers generated code calls.
- `process.rs`: the `Process` (memory, modules, GPU, services, HLE table) and `run_until`, the
  dispatcher: an indirect branch looks up the recompiled function for its target, or falls back to
  `interp.rs`. Addresses the interpreter had to run are recorded in `coverage.rs` and written to the
  `.profile` every 10 s and when the program exits, for the next build.
- `simd.rs`: SIMD/FP instructions the backend does not inline run through the `a64` semantics
  (`kirikae info` counts them apart from words lifted to IR, not as missing).
  V registers are two `u64` halves, so the integer NEON forms in `a64::lift_neon` (SHL/USHR/SSHR
  by immediate on 8/16/32/64-bit lanes, 8/16-bit `DUP`/`INS`, `MUL`, `SMAX`/`UMAX`/`SMIN`/`UMIN`,
  `MLA`/`MLS`, `SMAXP`/`UMAXP`/`SMINP`/`UMINP`, `ABS`/`NEG` and the saturating
  `SQADD`/`SQSUB`/`SQABS`/`SQNEG`) are lifted to one `u64` op per half
  (`Op::Vshl`/`Vushr`/`Vsshr`/`Splat`/`Vmul`/`Vmax`/`Vmin`/`Vmla`/`Vmls`/`Vpmax`/`Vpmin`/`Vabs`/
  `Vneg`/`Vsat`); `lift_fp` keeps the bitwise three-same group, the wrapping `ADD`/`SUB`/`ADDP`
  forms and the 32/64-bit `DUP`/`INS`.
  The Cranelift lowering folds the lane shift and keep mask into `ishl`/`ushr`/`band` plus
  immediates, unrolls `mul`/`mla` per lane, turns `max`/`min` into a borrow-free lane comparison
  (`(x | H) - (y & !H)` with `H` the top bit of each lane) plus a mask select, `abs`/`neg` into
  masks and lane-base additions, and the saturating forms into per-lane clamps (explicit overflow
  rules for 64-bit lanes) whose cumulative `QC` update is an `FPSR` read-modify-write fed by
  `Op::VsatFlag`. The interpreter shares the same ops, so both paths match the helper bit for bit
  (differential tests in `crates/a64/tests/neon_lift.rs` and `crates/recomp/tests/clif_ops.rs`).
  Everything else, including the scalar `Q = 1` forms, `SRI`/`SLI`, `PMUL` and the
  general-register forms, still calls the helper.
- `fault.rs`: turns host access violations into a guest backtrace with module offsets and symbols.

### Kernel

- `horizon.rs`: the SVC table (`svc`), from memory management to synchronization and IPC.
- `threads.rs`, `sched.rs`: each guest thread is a host thread. `sched.rs` models Horizon's
  scheduler (priorities, cores, affinity masks); guest threads run in parallel by default, with a
  global lock as a fallback (`KIRIKAE_PARALLEL=0`).
- `arbiter.rs`: mutexes, condition variables and address arbiters (`ArbitrateLock`,
  `WaitProcessWideKey`, `WaitForAddress`, `SignalToAddress`).
- `kernel.rs`, `ipc.rs`: sessions, domains and the HIPC/CMIF message format.

### Services (`services/`)

One file per service family (`am`, `fs`, `hid`, `vi`, `nvdrv`, `audio`, `time`, `set`, ...), each a
`match` on the command ID. Anything not handled falls to `stub.rs`, which answers zeros and warns
once; `usage.rs` records what each game used for the coverage reports, and `tools/progress.py` counts
the arms against Switchbrew.

### GPU (`gpu/`)

```
nvdrv SUBMIT_GPFIFO ──► gpu/mod.rs: pushbuffer, methods by class ──► threed.rs (3D draws, clears)
                                       │                             compute.rs (QMD dispatches)
                                       │                             twod.rs, DMA copies (mod.rs)
                                       │                             mme.rs (macros)
                                       ▼
                              vulkan/: the renderer (mod.rs holds the Renderer and its setup)
                              ├─ draw.rs: draws and dispatches; descriptors.rs: what each shader binds
                              ├─ pipeline.rs: maxwell::translate → SPIR-V → pipelines (cached on disk)
                              ├─ textures.rs, volumes.rs, tiling.rs: tic.rs headers, formats.rs,
                              │  astc.rs, block-linear detiling, 3D textures
                              ├─ buffers.rs: vertex and index buffers (buffer_cache.rs keeps them on the GPU)
                              ├─ images.rs: render targets as Vulkan images; readback.rs writes them back
                              │  to guest memory when read; guest_writes.rs tracks CPU writes over them
                              ├─ submit.rs: command buffers, the upload ring, batches in flight
                              └─ present.rs: the swapchain
vi QueueBuffer ──► display.rs: the window and the swapchain, frame pacing
```

GPU addresses are translated through the `nvdrv` address space mappings (`Gpu::translate`). Data the
CPU writes (textures, constant buffers) is compared or hashed so only what changed is uploaded.

### NVN on the host (`hle.rs`, `nvn/`)

`hle.rs` lets host functions live at guest addresses (a trampoline the dispatcher recognizes).
`nvn/` wraps the game's NVN driver: `KIRIKAE_NVN=trace` counts calls, `shadow` builds a host model
(`shadow.rs` for command lists and draws, `state.rs` for render state, `resources.rs` for pools,
buffers, textures and views) and checks it against what the command processor decodes. This is the
groundwork for drawing straight from NVN calls. Alongside the shadow model, `host.rs` fills the host
object table from the builder procs: `objects.rs` keys objects by guest address with deferred
destruction and incarnations, `va.rs` allocates GPU virtual addresses and pool ranges, and
`pools.rs`, `buffers.rs`, `textures.rs` and `samplers.rs` hold their decoded state. No command is
recorded or replayed from them yet.

### Video and audio

- `host1x/`: the multimedia engines. `nvdec.rs` (VP9 in `vp9.rs`, H.264 in `h264.rs` through
  OpenH264, behind the `h264` feature) and `vic.rs` (color conversion and composition).
- `services/audio/`: the `audren` renderer (`mixer.rs`, `dsp.rs`, `effect.rs`) and the host output
  (`output.rs`). `effect.rs` runs the mix effects (biquad, delay and the four-FDN reverb) after a mix
  is built and before it is sent to its destination or the sink.

### Game-specific code

Kirikae aims to emulate the console, not to patch games. When a game needs help that is not how the
hardware behaves, it goes into a plugin (`kirikae build --plugin`), which can hook presentation through
`games.rs`.

## How a build works

1. `loader::package::read_source` reads the dump; `from_exefs` lays the modules out.
2. `recomp::discover` finds functions starting from module entries, the unwind header
   (`.eh_frame_hdr`, with FDE sizes), relocations and symbols, plus the addresses a previous run had
   to interpret (`<name>.profile`).
3. Each function is lifted to IR and compiled by Cranelift into object files, in parallel.
4. `project.rs` writes a Cargo project that links the objects, depends on `kirikae-rt` and embeds the
   packed dump; `cargo build` produces `game.exe`. A PE image cannot exceed 2 GiB and the Windows
   loader gives up below that, so when the objects would not fit the code goes into
   `<name>_chunk<N>.dll` files next to a thin `game.exe` that loads them and joins their entry
   tables (`docs/plan/findings/large-executables.md`).

## Tests

- `crates/recomp/tests`: the Cranelift backend against the IR semantics on random inputs, and
  end-to-end builds of the testkit programs.
- `crates/a64`: decoding and SIMD semantics (with an optional Unicorn oracle, `tools/simd_oracle.py`).
- `crates/loader`, `crates/rt`, `crates/maxwell`: unit tests next to the code.

`cargo test --release --workspace -- --skip register_operands` is what CI runs (that one test is very slow).
