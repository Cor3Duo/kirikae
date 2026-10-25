# AArch32 titles on the Switch

Date: 2026-10-24. Updated: 2026-10-25.

Status: **confirmed**. Mario Kart 8 Deluxe 1.0 (`0100152000022000`) and New Super
Mario Bros. U Deluxe are 32-bit ARM titles. Kirikae has no AArch32 frontend yet;
the loader rejects them with an explicit message.

## What was seen

`kirikae info` on the extracted ExeFS reported an anomalous coverage:

```
all code: 4,814,793 words, 12.47 % lifted in IR, 5.17 % SIMD/FP unsupported, 82.35 % other
```

and `kirikae build` discovered **8 functions / 9 instructions**. The `.eh_frame_hdr`
parser also read nonsense versions (204, 84, 164), because the image is not the code it
expects.

## Evidence

On the `main` module dumped with `kirikae dev module-dump`:

| Pattern | Occurrences |
| --- | ---: |
| `push {r11, lr}` (`00 48 2d e9`) | **5,457** |
| `push {r4-r11, lr}` (`f0 4f 2d e9`) | **3,790** |
| `bx lr` (`1e ff 2f e1`) | **13,456** |
| `stp x29, x30, [sp, #-16]!` (A64) | **0** |

Capstone decodes 5,404 valid A32 instructions in the 0x1000–0x2000 window and one A64
instruction. The binary is ARM32.

### Independent confirmation

The check was repeated without the project's decoder: the loader's LZ4 output is
byte-identical to an independent `lz4.block` decompression and to the SHA-256 values in
the NSO header, and Capstone was run over the whole text segment (2,715,215 words):

| Metric | A32 (`CS_ARCH_ARM`) | A64 (`CS_ARCH_ARM64`) |
| --- | ---: | ---: |
| words decoded | 2,709,682 (**99.80 %**) | 883,470 (32.54 %) |
| words not decoded | 5,533 (0.20 %) | 1,831,745 (**67.46 %**) |
| `push {…, lr}` / prologues | 32,129 | 0 |
| returns (`bx lr` / `pop {…, pc}`) | 40,902 (`bx lr` = 13,456) | 0 |
| `ret` | 0 | **0** |
| `stp` with `x29, x30` | 0 | 221 |
| `bl` | 156,927 | 749 |
| `udf` | 0 | 9,508 |

Control flow: **27,107** prologue→return ranges, of which 26,837 (99.0 %) decode their
whole body; 156,927 `bl` with 19,584 distinct targets, **100 %** inside `.text`;
350,181 of 350,206 immediate branches (99.993 %) land inside `.text` on instruction
boundaries; no `blx #imm`, so no Thumb interworking appears in this title.

Positive control: the same analysis on Pokémon Sword, a known A64 title, decodes
99.99 % as A64 and 71.22 % as A32, with 96,672 `ret` and zero A32 prologues.

## Metadata: which field says the architecture

The ACID flags do **not** encode the architecture (bit 0 is `ProductionFlag`, bit 1
`UnqualifiedApproval`). The first hypothesis used them and was wrong: MK8D and Sword
both have ACID flags `0x1`, and one is A32 while the other is A64.

`META.Flags0` at offset `0x0C` does: bit 0 is `Is64BitInstruction` and bits 1-3 are
`ProcessAddressSpace`.

| Title | META `0x0C` | ACID flags | `bx lr` | `ret` | Architecture |
| --- | ---: | ---: | ---: | ---: | --- |
| Mario Kart 8 Deluxe | 0x04 | 0x1 | 13,456 | 0 | **A32** |
| New Super Mario Bros. U Deluxe | 0x04 | 0x3 | 15,238 | 0 | **A32** |
| Pokémon Sword / Shield | 0x07 | 0x1 | 0 | 96,672 | A64 |
| Pokémon Let's Go Eevee / Pikachu | 0x07 | 0x1 | 0 | 52,592 | A64 |
| Super Mario Odyssey | 0x07 | 0x3 | 0 | 58,667 | A64 |
| Super Mario 3D All-Stars | 0x07 | 0x3 | 0 | 22,122 | A64 |
| Super Smash Bros. Ultimate | 0x07 | 0x3 | 0 | 96,939 | A64 |
| Zelda: Breath of the Wild | 0x03 | 0x3 | 0 | 82,465 | A64 |
| Brilliant Diamond / Shining Pearl | 0x37 | 0x3 | 0 | 153,029 | A64 |
| Legends Arceus | 0x37 | 0x3 | 0 | 134,575 | A64 |
| Legends Z-A | 0x37 | 0x3 | 0 | 200,748 | A64 |
| Scarlet / Violet | 0x37 | 0x3 | 0 | 152,455 | A64 |

The loader now reads `META.Flags0` and, as a fallback for metadata-less dumps, samples
the first words of the text segment. A 32-bit title is rejected with
`this title's main module is AArch32 (32-bit)` instead of being misparsed as AArch64.

## Implications

- The first Switch titles (SDK 0.12, Wii U ports such as MK8D) can be AArch32. This is
  not an extraction failure: PFS0, IVFC and RomFS validate.
- Kirikae only has an AArch64 frontend. These titles need a complete A32 frontend
  (decoder, IR, interpreter and lowering), not a patch.
- Until then the title is blocked by architecture, not by missing services.

## Current state of the AArch32 work

- Decoder and IR for ARM mode exist in `crates/a64/src/a32`, with typed rejections and
  a differential run against Capstone over 1.08 M random words (no false acceptances).
- The loader detection and message are done.
- Next: an interpreter path, then Thumb-2/VFP measurement, discovery, predication and
  lowering. See [`aarch32-frontend.md`](aarch32-frontend.md) and the repository
  [`ROADMAP.md`](../../../ROADMAP.md).

## References

- `crates/loader/src/package.rs` (`npdm_is_aarch32`, `looks_aarch32`).
- Switchbrew: NPDM, ACID/ACI0, process address space.
