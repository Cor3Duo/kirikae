# Executables over the PE 2 GiB limit

Session of 2026-10-24/25, worktree `ws10-link`. Question: why do Brilliant Diamond / Shining
Pearl fail to link with `LNK1248`, is Mystery Dungeon's `LNK1104` the same problem, and what is
the smallest change that makes the large titles linkable without slowing the small ones.

## What the failure is

`kirikae build` compiles every reachable guest function into COFF objects (1024 functions per
object) and links all of them, plus `kirikae-rt` and the packed dump, into one `.exe`. The MSVC
linker refuses a PE image over 2 GiB:

```
LINK : fatal error LNK1248: image size (A6A7A000) exceeds the maximum allowed size (80000000)
```

`0xA6A7A000` is 2.79 GB. That is not the object total: the objects are 2.696 GB, the rest is the
runtime, the entry table and section padding.

## What the 2.8 GB is made of

Measured on the 256 objects of the failed `pokemon-brilliant-diamond` build
(`%LOCALAPPDATA%\kirikae-discovery\batch-all\pokemon-brilliant-diamond\out\...\obj`, 2026-10-24),
reading each COFF section table:

| Section | Size |
| --- | --- |
| `.text` | **2.696 GB** |
| `.pdata` / `.xdata` | 0 |
| `.debug*` | 0 |
| `.rdata` / other | 0 |

Every object has exactly one section, `.text`, with characteristics `0x60400020` (code, execute,
read, 16-byte aligned, **not COMDAT**). So:

- There is no debug information in the image (rustc keeps it in the PDB) and no unwind tables:
  Cranelift emits neither for these objects. There is nothing to strip.
- `/OPT:REF,ICF` (already passed by rustc) cannot help: both options only fold or drop COMDAT
  sections, and Cranelift emits neither COMDATs nor function-level sections. The linker keeps
  every byte.
- `panic = "abort"`, `opt-level`, `codegen-units` and `strip` concern the Rust runtime crate
  (tens of MB), not the 2.7 GB of generated code. None of them is a lever here.

Ratio: 2.696 GB / 13.3 M reachable guest instructions = **202 bytes of host code per guest
instruction**, versus 483 MB / 9.7 M = 50 bytes in Super Mario Bros. Wonder and
573 MB / 12.4 M = 46 bytes in Mystery Dungeon.

The difference is not the lowering, it is discovery: the report line counts the functions it
*lifted*, and for BD/SP that is 74.6 M instructions for 13.3 M reachable (5.6x), while Mystery
Dungeon lifts 12.7 M for 12.4 M. BD/SP has hundreds of thousands of entry points that sit
*inside* other functions' bodies, and `discover::build_function` compiles the overlapping blocks
once per entry (`Control::Fallthrough` keeps walking past another entry point). 5.6x is the
duplication factor, so the same game would be ~480 MB if each block were emitted once.

That is the real long-term reduction, but it changes what a "function" is in the IR
(`discover.rs`, out of scope for this session) and a game with a 3x duplication would still be
over the limit. The structural fix stays necessary.

## Mystery Dungeon is a path problem, not a size problem

Its objects are 573 MB (`obj/`, 229 objects) and the link fails on a different file:

```
LNK1104: cannot open '...\out\pokemon-mystery-dungeon-rescue-team-dx-build\target\release\build\
pokemon-mystery-dungeon-rescue-team-dx-a4fc8b77cf1bc0b3\build_script_build-a4fc8b77cf1bc0b3.exe'
```

That path is 261 characters. MSVC's linker opens paths with the ANSI API, so 260 is the hard
limit. The file is the generated `build.rs`, whose only job is to print the object paths as
`cargo:rustc-link-arg-bins`.

Reproduced and confirmed: relinking the same objects with a short `CARGO_TARGET_DIR` (`C:/k/md`)
succeeds (604,887,040-byte exe) because the build-script path is then short. The fix is to keep
the build in a short directory: when neither `BuildOptions::target_dir` nor `CARGO_TARGET_DIR` is
set, the default is `%TEMP%/kirikae-build/<hash of the project>` instead of `<project>/target`,
which sits under the (possibly very deep) output folder.

## The split

`kirikae build` decides, before code generation, whether one image can hold the code:

1. `clif::estimate` lifts every function and sums the instructions it will emit (one pass, no
   Cranelift). Bytes are estimated at 52 per lifted instruction, the highest ratio measured
   (BD 36, Mystery Dungeon 45, SMBW 50), so the estimate errs towards a split.
2. Over 1.8 GiB (2 GiB minus the runtime and padding) the objects are compiled with
   `Calls::Dispatch`: a call to a function this object does not hold is **not** declared as a
   link-time import but goes to `kirikae_call_indirect`, the same dispatcher path already used
   for `BLR`. Calls inside the object stay direct.
3. After code generation the real object sizes pick the DLLs: `partition` fills one DLL up to
   1.5 GiB and starts the next. BD/SP: two DLLs, 1.36 GB and 1.42 GB.
4. The generated project becomes an executable with no guest objects (just `package.bin` +
   `kirikae-rt`) and one `cdylib` crate per group. Each crate links its objects through the same
   tiny `build.rs` the single-image build uses (`cargo:rustc-link-arg-*`). The executable loads
   `<name>_chunk<N>.dll` from its own directory, calls each one's `kirikae_bind` with the
   runtime's ABI helper pointers, joins the `kirikae_entries` tables, sorts them by address and
   hands them to `program_main`.

The build script is not a stylistic choice. Passing the objects as `-C link-arg` on the rustc
command line overruns the 32767-character limit on a deep output folder (230 objects with a
~200-character path are 46 KB), and the obvious fix — a linker response file passed as
`-C link-arg=@file` — only works while rustc keeps the linker invocation short enough to pass it
directly: past that point rustc writes its own response file, and **link.exe does not expand a
nested `@file`**, so the link dies with `LNK1104: cannot open '@...rsp'`. Measured with the same
project and rsp at `C:/k/rsp-test` (links) and with a 90-character target directory (fails). A
build script puts the paths through rustc as ordinary arguments, which link.exe reads from
rustc's response file correctly.

Because the build script prints the same argument list on every build, `kirikae build` deletes
the previous artifact before invoking cargo; otherwise cargo considers the target fresh and
leaves the old image in place when only the generated objects changed.

The chunk DLLs have no Rust dependency and no import from each other: the nine ABI helpers
(`kirikae_svc`, `kirikae_op`, ...) are forwarded by tiny functions in the generated `lib.rs`
through a table the executable fills before any generated code runs (`abi::helper_table`). The
loader side is `LoadLibraryW`/`GetProcAddress` declared by hand, so no new crate dependency.

Games whose estimate stays under 1.8 GiB keep the old single-image path; SMBW and Odyssey are
in that group.

Using it: `kirikae build <dump> -o <dir>` decides on its own; `--split <n>` forces `n` DLLs
(used by the hello test). The estimate costs one extra lifting pass — about 4 s for SMBW and
30 s for BD/SP, against 500 s of Cranelift — and it is what keeps codegen from producing an
image that cannot be linked.

## The loader refuses large images before the linker does

Building BD first with a 50/50 split put 2.06 GB of code (image size `0x7B132000`, 1.92 GiB)
into `chunk0.dll`. It **linked**, and Windows refused it:

```
kirikae: could not load ...\pokemon_brilliant_diamond_chunk0.dll:
%1 is not a valid Win32 application (os error 193 = ERROR_BAD_EXE_FORMAT)
```

`chunk1.dll` (717 MB, image `0x2AC14000`) loaded fine, and `LoadLibraryEx` with
`LOAD_LIBRARY_AS_DATAFILE` accepted the file, so the file was intact and the check is on the
image layout. Synthetic DLLs isolate the limit (all x64, `SizeOfImage` = one huge `.rdata`):

| `SizeOfImage` | bytes | load |
| --- | --- | --- |
| `0x71401000` | 1,900,111,872 | ok |
| `0x742bc000` | 1,949,024,256 | ok |
| `0x7726b000` | 1,999,024,128 | **193** |
| `0x7B09E000` | 2,064,154,624 | **193** |

So there is a second ceiling between 1.95 GB and 2.00 GB, stricter than `LNK1248`'s 2 GiB. A
DLL with a `.bss`-only image at 1.89 GiB did load, so it is the image size, not the file size.
Chunk DLLs are therefore kept at 1.5 GiB (1.61 GB) of objects, which links to a ~1.45 GiB image,
well under the measured ceiling.

Consequence for the split: the groups cannot be cut by "the entry whose cumulative byte offset
crosses 50%", because one object in BD is **741 MB on its own** (a chunk of 1024 functions where
one guest function expands enormously). The first split put that object plus 1.4 GB of others
into one DLL and produced a 1.92 GiB image. `partition` now fills a DLL greedily up to the
target and starts a new one, so an oversized object gets a module of its own.

## Results

Kirikae binary: worktree `ws10-link`, `cargo build --release`, 2026-10-24/25,
`CARGO_TARGET_DIR=C:/Users/eleaz/AppData/Local/Temp/opencode/kt-ws10-link`.

**Brilliant Diamond** (`-o C:/k/bd`, 261,741 functions, 13,307,769 of 13,725,765 instructions
reachable):

| | before | after |
| --- | --- | --- |
| estimate pass | — | 74,618,780 lifted instructions, "3700 MB of code predicted" |
| code generation | 497 s (74,618,780 instructions) | 497 s |
| link | `LNK1248` after 2.79 GB | 2 DLLs: 1,362,507,264 + 1,419,619,840 bytes |
| exe | — | 51 MB (runtime + embedded dump) |
| total build | fails | 490 s |

Headless boot (`KIRIKAE_HEADLESS=1`, 20 s): both DLLs load, the guest starts threads, brings up
HID and `nn::audio`, and then faults on an unmapped read at `0x30` inside
`nn::audio::GetAudioOutBufferDataPointer` (`sdk+0x3ac80`, called from `main+0x1334b7c`). The
**interpreter reaches the same instruction with the same fault** (`kirikae run` on the same
dump), so it is a service gap in BD's audio path, not the split. That is the first thing to fix
before BD presents a frame.

**Mystery Dungeon** (234,399 functions, 12,389,733 of 13,086,374 reachable): built with
`-o ...\out\long-path-test2` under the same deep batch folder that produced the 261-character
build-script path, with no `CARGO_TARGET_DIR` override. Links in 81 s (604 MB image); before, it
died on `LNK1104`. The relevant change is the default target directory:
`%TEMP%/kirikae-build/<hash>` keeps the build-script path at ~110 characters whatever the output
folder is.

**Super Mario Bros. Wonder** (`myswitch/extracted`, 247,410 functions, 9,748,969 reachable): no
split (its objects are 408 MB), 470,152,704-byte exe, 337 frames in 10 s (33.7 fps, 77.0 without
the frame limit), 20,253 draws — the same behaviour as before the change (324 frames / 32.4 fps
on the pre-change build).

**Hello testkit**, forced `--split 2`: two DLLs of 13 KB, prints `HELLO_EXPECTED` with
"0 interpreted instructions" (`cranelift_split_hello` in `crates/kirikae/tests/end_to_end.rs`).

## Remaining work

- **The 5.6x overlap is the real fix.** If `build_function` stopped past another entry point and
  emitted a call instead (`discover.rs`), BD/SP would be around 480 MB and would not need a split
  at all. That is the change to make next; it also shrinks every other title.
- A chunk (1024 functions) is the smallest unit that can move to a DLL. One BD chunk is 741 MB
  on its own; a chunk over the ceiling could not be split further.
- The estimate (52 bytes per lifted instruction) is deliberately high. A title whose real total
  lands under 1.8 GiB after a "large" estimate is linked as one image but keeps the dispatcher
  call in cross-chunk calls; recompiling with `Calls::Import` would cost another codegen pass.
  No current title is in that band (largest after BD/SP: Legends Arceus, 836 MB of objects).
- Cross-DLL calls go through `run_until` (hash lookup, then an indirect call). A per-DLL jump
  table in the entry table would avoid the dispatcher, at the cost of a second symbol per
  function.
- BD's audio fault (`nn::audio::GetAudioOutBufferDataPointer` reading `0x30`) needs a look in
  `services/audio` before a frame can be judged.
