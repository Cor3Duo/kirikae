//! A32 instruction mnemonics.
//!
//! One variant per assembly mnemonic, so a decoded [`super::Insn`] can be matched on
//! the operation alone. [`Opcode::ALL`] lists every variant; the decoder and the
//! interpreter both have to cover it, which the compiler enforces with exhaustive
//! `match`.

/// Decoded instruction mnemonic, without the condition suffix.
///
/// | mnemonic | operands |
/// |---|---|
/// | `and` ... `bic`, except `mov`, `mvn` | `[Rd, Rn, op2]` |
/// | `mov`, `mvn` | `[Rd, op2]` |
/// | `tst`, `teq`, `cmp`, `cmn` | `[Rn, op2]` |
/// | `movw`, `movt` | `[Rd, Imm(imm16)]` |
/// | `b`, `bl`, `blx` (immediate) | `[Target]` |
/// | `bx`, `blx` (register) | `[Reg]` |
/// | `ldr`, `str`, `ldrh`, `strh`, `ldrsb`, `ldrsh`, `ldrex*` | `[Reg(rt), Mem]` |
/// | `ldrd`, `strd` | `[Regs { lo: Rt, hi: Rt + 1 }, Mem]` |
/// | `strex`, `strexb`, `strexh` | `[Reg(status), Reg(rt), Mem]` |
/// | `ldm`, `stm` | `[Block]` |
/// | `mul` | `[Rd, Rm, Rs]` |
/// | `mla`, `smla<x><y>`, `smlaw<y>` | `[Rd, Rm, Rs, Ra]` |
/// | `smul<x><y>`, `smulw<y>` | `[Rd, Rm, Rs]` |
/// | `umull`, `umlal`, `smull`, `smlal`, `smlal<x><y>` | `[Regs { lo: RdLo, hi: RdHi }, Rm, Rs]` |
/// | `uxtb`, `sxth`, `uxtb16`, `sxtb16` (plain) | `[Rd, Shifted { reg: Rm, shift }]` |
/// | `uxtab`, `sxtah`, `uxtab16`, `sxtab16` (accumulating) | `[Rd, Rn, Shifted { reg: Rm, shift }]` |
/// | `rev`, `rev16`, `revsh`, `rbit`, `clz` | `[Rd, Rm]` |
/// | `pkhbt`, `pkhtb` | `[Rd, Rn, Shifted { reg: Rm, shift }]` |
/// | `sel`, parallel `*add`/`*sub`, `*asx`/`*sax` | `[Rd, Rn, Rm]` |
/// | `ssat`, `usat` | `[Rd, Imm(sat_imm), Shifted { reg: Rm, shift }]` |
/// | `ssat16`, `usat16` | `[Rd, Imm(sat_imm), Rm]` |
/// | `bfc` | `[Rd, Imm(lsb), Imm(width)]` |
/// | `bfi`, `sbfx`, `ubfx` | `[Rd, Rn, Imm(lsb), Imm(width)]` |
/// | `qadd`, `qsub`, `qdadd`, `qdsub` | `[Rd, Rm, Rn]` |
/// | `mrs` | `[Rd]` |
/// | `msr` | `[Reg(Rm)]` or `[Imm(value)]` |
/// | `nop`, `yield`, `wfe`, `wfi`, `sev` | `[Imm(hint)]` |
/// | `svc` | `[Imm(imm24)]` |
/// | `bkpt` | `[Imm(imm16)]` |
///
/// `op2` is either `Operand::Imm` (already rotated) or `Operand::Shifted`.
///
/// `ldrd`/`strd` show the pair as two registers, which is more explicit than the
/// assembly syntax: the encoding names only `Rt`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Opcode {
    // Data processing.
    And,
    Eor,
    Sub,
    Rsb,
    Add,
    Adc,
    Sbc,
    Rsc,
    Tst,
    Teq,
    Cmp,
    Cmn,
    Orr,
    Mov,
    Bic,
    Mvn,
    Movw,
    Movt,
    // Branches and supervisor calls.
    B,
    Bl,
    Bx,
    Blx,
    Bkpt,
    Svc,
    // Single-register and block transfers.
    Ldr,
    Str,
    Ldm,
    Stm,
    // Multiplies.
    Mul,
    Mla,
    Umull,
    Umlal,
    Smull,
    Smlal,
    Smlabb,
    Smlabt,
    Smlatb,
    Smlatt,
    Smlawb,
    Smlawt,
    Smulbb,
    Smulbt,
    Smultb,
    Smultt,
    Smulwb,
    Smulwt,
    Smlalbb,
    Smlalbt,
    Smlaltb,
    Smlaltt,
    // Extra loads and stores.
    Ldrh,
    Strh,
    Ldrsb,
    Ldrsh,
    Ldrd,
    Strd,
    // Exclusive access.
    Ldrex,
    Strex,
    Ldrexb,
    Strexb,
    Ldrexh,
    Strexh,
    // Media: extend, reverse, pack and select.
    Uxtb,
    Uxth,
    Sxtb,
    Sxth,
    Uxtab,
    Uxtah,
    Sxtab,
    Sxtah,
    Uxtb16,
    Sxtb16,
    Uxtab16,
    Sxtab16,
    Rev,
    Rev16,
    Revsh,
    Rbit,
    Clz,
    Pkhbt,
    Pkhtb,
    Sel,
    // Media: saturate and bitfield.
    Ssat,
    Usat,
    Ssat16,
    Usat16,
    Bfc,
    Bfi,
    Sbfx,
    Ubfx,
    // Parallel addition and subtraction, signed.
    Sadd16,
    Sasx,
    Ssax,
    Ssub16,
    Sadd8,
    Ssub8,
    Qadd16,
    Qasx,
    Qsax,
    Qsub16,
    Qadd8,
    Qsub8,
    Shadd16,
    Shasx,
    Shsax,
    Shsub16,
    Shadd8,
    Shsub8,
    // Parallel addition and subtraction, unsigned.
    Uadd16,
    Uasx,
    Usax,
    Usub16,
    Uadd8,
    Usub8,
    Uqadd16,
    Uqasx,
    Uqsax,
    Uqsub16,
    Uqadd8,
    Uqsub8,
    Uhadd16,
    Uhasx,
    Uhsax,
    Uhsub16,
    Uhadd8,
    Uhsub8,
    // Saturating scalar arithmetic and system.
    Qadd,
    Qsub,
    Qdadd,
    Qdsub,
    Mrs,
    Msr,
    Nop,
    Yield,
    Wfe,
    Wfi,
    Sev,
}

impl Opcode {
    pub const ALL: &'static [Opcode] = &[
        // Data processing.
        Opcode::And,
        Opcode::Eor,
        Opcode::Sub,
        Opcode::Rsb,
        Opcode::Add,
        Opcode::Adc,
        Opcode::Sbc,
        Opcode::Rsc,
        Opcode::Tst,
        Opcode::Teq,
        Opcode::Cmp,
        Opcode::Cmn,
        Opcode::Orr,
        Opcode::Mov,
        Opcode::Bic,
        Opcode::Mvn,
        Opcode::Movw,
        Opcode::Movt,
        // Branches and supervisor calls.
        Opcode::B,
        Opcode::Bl,
        Opcode::Bx,
        Opcode::Blx,
        Opcode::Bkpt,
        Opcode::Svc,
        // Single-register and block transfers.
        Opcode::Ldr,
        Opcode::Str,
        Opcode::Ldm,
        Opcode::Stm,
        // Multiplies.
        Opcode::Mul,
        Opcode::Mla,
        Opcode::Umull,
        Opcode::Umlal,
        Opcode::Smull,
        Opcode::Smlal,
        Opcode::Smlabb,
        Opcode::Smlabt,
        Opcode::Smlatb,
        Opcode::Smlatt,
        Opcode::Smlawb,
        Opcode::Smlawt,
        Opcode::Smulbb,
        Opcode::Smulbt,
        Opcode::Smultb,
        Opcode::Smultt,
        Opcode::Smulwb,
        Opcode::Smulwt,
        Opcode::Smlalbb,
        Opcode::Smlalbt,
        Opcode::Smlaltb,
        Opcode::Smlaltt,
        // Extra loads and stores.
        Opcode::Ldrh,
        Opcode::Strh,
        Opcode::Ldrsb,
        Opcode::Ldrsh,
        Opcode::Ldrd,
        Opcode::Strd,
        // Exclusive access.
        Opcode::Ldrex,
        Opcode::Strex,
        Opcode::Ldrexb,
        Opcode::Strexb,
        Opcode::Ldrexh,
        Opcode::Strexh,
        // Media: extend, reverse, pack and select.
        Opcode::Uxtb,
        Opcode::Uxth,
        Opcode::Sxtb,
        Opcode::Sxth,
        Opcode::Uxtab,
        Opcode::Uxtah,
        Opcode::Sxtab,
        Opcode::Sxtah,
        Opcode::Uxtb16,
        Opcode::Sxtb16,
        Opcode::Uxtab16,
        Opcode::Sxtab16,
        Opcode::Rev,
        Opcode::Rev16,
        Opcode::Revsh,
        Opcode::Rbit,
        Opcode::Clz,
        Opcode::Pkhbt,
        Opcode::Pkhtb,
        Opcode::Sel,
        // Media: saturate and bitfield.
        Opcode::Ssat,
        Opcode::Usat,
        Opcode::Ssat16,
        Opcode::Usat16,
        Opcode::Bfc,
        Opcode::Bfi,
        Opcode::Sbfx,
        Opcode::Ubfx,
        // Parallel addition and subtraction, signed.
        Opcode::Sadd16,
        Opcode::Sasx,
        Opcode::Ssax,
        Opcode::Ssub16,
        Opcode::Sadd8,
        Opcode::Ssub8,
        Opcode::Qadd16,
        Opcode::Qasx,
        Opcode::Qsax,
        Opcode::Qsub16,
        Opcode::Qadd8,
        Opcode::Qsub8,
        Opcode::Shadd16,
        Opcode::Shasx,
        Opcode::Shsax,
        Opcode::Shsub16,
        Opcode::Shadd8,
        Opcode::Shsub8,
        // Parallel addition and subtraction, unsigned.
        Opcode::Uadd16,
        Opcode::Uasx,
        Opcode::Usax,
        Opcode::Usub16,
        Opcode::Uadd8,
        Opcode::Usub8,
        Opcode::Uqadd16,
        Opcode::Uqasx,
        Opcode::Uqsax,
        Opcode::Uqsub16,
        Opcode::Uqadd8,
        Opcode::Uqsub8,
        Opcode::Uhadd16,
        Opcode::Uhasx,
        Opcode::Uhsax,
        Opcode::Uhsub16,
        Opcode::Uhadd8,
        Opcode::Uhsub8,
        // Saturating scalar arithmetic and system.
        Opcode::Qadd,
        Opcode::Qsub,
        Opcode::Qdadd,
        Opcode::Qdsub,
        Opcode::Mrs,
        Opcode::Msr,
        Opcode::Nop,
        Opcode::Yield,
        Opcode::Wfe,
        Opcode::Wfi,
        Opcode::Sev,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Opcode::And => "and",
            Opcode::Eor => "eor",
            Opcode::Sub => "sub",
            Opcode::Rsb => "rsb",
            Opcode::Add => "add",
            Opcode::Adc => "adc",
            Opcode::Sbc => "sbc",
            Opcode::Rsc => "rsc",
            Opcode::Tst => "tst",
            Opcode::Teq => "teq",
            Opcode::Cmp => "cmp",
            Opcode::Cmn => "cmn",
            Opcode::Orr => "orr",
            Opcode::Mov => "mov",
            Opcode::Bic => "bic",
            Opcode::Mvn => "mvn",
            Opcode::Movw => "movw",
            Opcode::Movt => "movt",
            Opcode::B => "b",
            Opcode::Bl => "bl",
            Opcode::Bx => "bx",
            Opcode::Blx => "blx",
            Opcode::Bkpt => "bkpt",
            Opcode::Svc => "svc",
            Opcode::Ldr => "ldr",
            Opcode::Str => "str",
            Opcode::Ldm => "ldm",
            Opcode::Stm => "stm",
            Opcode::Mul => "mul",
            Opcode::Mla => "mla",
            Opcode::Umull => "umull",
            Opcode::Umlal => "umlal",
            Opcode::Smull => "smull",
            Opcode::Smlal => "smlal",
            Opcode::Smlabb => "smlabb",
            Opcode::Smlabt => "smlabt",
            Opcode::Smlatb => "smlatb",
            Opcode::Smlatt => "smlatt",
            Opcode::Smlawb => "smlawb",
            Opcode::Smlawt => "smlawt",
            Opcode::Smulbb => "smulbb",
            Opcode::Smulbt => "smulbt",
            Opcode::Smultb => "smultb",
            Opcode::Smultt => "smultt",
            Opcode::Smulwb => "smulwb",
            Opcode::Smulwt => "smulwt",
            Opcode::Smlalbb => "smlalbb",
            Opcode::Smlalbt => "smlalbt",
            Opcode::Smlaltb => "smlaltb",
            Opcode::Smlaltt => "smlaltt",
            Opcode::Ldrh => "ldrh",
            Opcode::Strh => "strh",
            Opcode::Ldrsb => "ldrsb",
            Opcode::Ldrsh => "ldrsh",
            Opcode::Ldrd => "ldrd",
            Opcode::Strd => "strd",
            Opcode::Ldrex => "ldrex",
            Opcode::Strex => "strex",
            Opcode::Ldrexb => "ldrexb",
            Opcode::Strexb => "strexb",
            Opcode::Ldrexh => "ldrexh",
            Opcode::Strexh => "strexh",
            Opcode::Uxtb => "uxtb",
            Opcode::Uxth => "uxth",
            Opcode::Sxtb => "sxtb",
            Opcode::Sxth => "sxth",
            Opcode::Uxtab => "uxtab",
            Opcode::Uxtah => "uxtah",
            Opcode::Sxtab => "sxtab",
            Opcode::Sxtah => "sxtah",
            Opcode::Uxtb16 => "uxtb16",
            Opcode::Sxtb16 => "sxtb16",
            Opcode::Uxtab16 => "uxtab16",
            Opcode::Sxtab16 => "sxtab16",
            Opcode::Rev => "rev",
            Opcode::Rev16 => "rev16",
            Opcode::Revsh => "revsh",
            Opcode::Rbit => "rbit",
            Opcode::Clz => "clz",
            Opcode::Pkhbt => "pkhbt",
            Opcode::Pkhtb => "pkhtb",
            Opcode::Sel => "sel",
            Opcode::Ssat => "ssat",
            Opcode::Usat => "usat",
            Opcode::Ssat16 => "ssat16",
            Opcode::Usat16 => "usat16",
            Opcode::Bfc => "bfc",
            Opcode::Bfi => "bfi",
            Opcode::Sbfx => "sbfx",
            Opcode::Ubfx => "ubfx",
            Opcode::Sadd16 => "sadd16",
            Opcode::Sasx => "sasx",
            Opcode::Ssax => "ssax",
            Opcode::Ssub16 => "ssub16",
            Opcode::Sadd8 => "sadd8",
            Opcode::Ssub8 => "ssub8",
            Opcode::Qadd16 => "qadd16",
            Opcode::Qasx => "qasx",
            Opcode::Qsax => "qsax",
            Opcode::Qsub16 => "qsub16",
            Opcode::Qadd8 => "qadd8",
            Opcode::Qsub8 => "qsub8",
            Opcode::Shadd16 => "shadd16",
            Opcode::Shasx => "shasx",
            Opcode::Shsax => "shsax",
            Opcode::Shsub16 => "shsub16",
            Opcode::Shadd8 => "shadd8",
            Opcode::Shsub8 => "shsub8",
            Opcode::Uadd16 => "uadd16",
            Opcode::Uasx => "uasx",
            Opcode::Usax => "usax",
            Opcode::Usub16 => "usub16",
            Opcode::Uadd8 => "uadd8",
            Opcode::Usub8 => "usub8",
            Opcode::Uqadd16 => "uqadd16",
            Opcode::Uqasx => "uqasx",
            Opcode::Uqsax => "uqsax",
            Opcode::Uqsub16 => "uqsub16",
            Opcode::Uqadd8 => "uqadd8",
            Opcode::Uqsub8 => "uqsub8",
            Opcode::Uhadd16 => "uhadd16",
            Opcode::Uhasx => "uhasx",
            Opcode::Uhsax => "uhsax",
            Opcode::Uhsub16 => "uhsub16",
            Opcode::Uhadd8 => "uhadd8",
            Opcode::Uhsub8 => "uhsub8",
            Opcode::Qadd => "qadd",
            Opcode::Qsub => "qsub",
            Opcode::Qdadd => "qdadd",
            Opcode::Qdsub => "qdsub",
            Opcode::Mrs => "mrs",
            Opcode::Msr => "msr",
            Opcode::Nop => "nop",
            Opcode::Yield => "yield",
            Opcode::Wfe => "wfe",
            Opcode::Wfi => "wfi",
            Opcode::Sev => "sev",
        }
    }
    /// Data-processing instructions whose only result is NZCV.
    pub const fn is_flag_test(self) -> bool {
        matches!(self, Opcode::Tst | Opcode::Teq | Opcode::Cmp | Opcode::Cmn)
    }

    /// Data-processing instructions (`and` ... `mvn`), including `movw`/`movt`.
    pub const fn is_data_processing(self) -> bool {
        matches!(
            self,
            Opcode::And
                | Opcode::Eor
                | Opcode::Sub
                | Opcode::Rsb
                | Opcode::Add
                | Opcode::Adc
                | Opcode::Sbc
                | Opcode::Rsc
                | Opcode::Tst
                | Opcode::Teq
                | Opcode::Cmp
                | Opcode::Cmn
                | Opcode::Orr
                | Opcode::Mov
                | Opcode::Bic
                | Opcode::Mvn
                | Opcode::Movw
                | Opcode::Movt
        )
    }

    /// Whether the mnemonic takes the `S` suffix (`adds`, `movs`, `muls`). The media,
    /// load/store and system instructions have no `S` form.
    pub const fn takes_s_suffix(self) -> bool {
        matches!(
            self,
            Opcode::And
                | Opcode::Eor
                | Opcode::Sub
                | Opcode::Rsb
                | Opcode::Add
                | Opcode::Adc
                | Opcode::Sbc
                | Opcode::Rsc
                | Opcode::Orr
                | Opcode::Mov
                | Opcode::Bic
                | Opcode::Mvn
                | Opcode::Mul
                | Opcode::Mla
                | Opcode::Umull
                | Opcode::Umlal
                | Opcode::Smull
                | Opcode::Smlal
                | Opcode::Smlabb
                | Opcode::Smlabt
                | Opcode::Smlatb
                | Opcode::Smlatt
                | Opcode::Smlawb
                | Opcode::Smlawt
                | Opcode::Smulbb
                | Opcode::Smulbt
                | Opcode::Smultb
                | Opcode::Smultt
                | Opcode::Smulwb
                | Opcode::Smulwt
                | Opcode::Smlalbb
                | Opcode::Smlalbt
                | Opcode::Smlaltb
                | Opcode::Smlaltt
        )
    }

    /// Maps the four-bit data-processing opcode, bits 24:21.
    pub const fn from_data_processing(field: u32) -> Opcode {
        match field & 0xf {
            0 => Opcode::And,
            1 => Opcode::Eor,
            2 => Opcode::Sub,
            3 => Opcode::Rsb,
            4 => Opcode::Add,
            5 => Opcode::Adc,
            6 => Opcode::Sbc,
            7 => Opcode::Rsc,
            8 => Opcode::Tst,
            9 => Opcode::Teq,
            10 => Opcode::Cmp,
            11 => Opcode::Cmn,
            12 => Opcode::Orr,
            13 => Opcode::Mov,
            14 => Opcode::Bic,
            _ => Opcode::Mvn,
        }
    }

    /// The parallel add/subtract instruction encoded by `group` (bits 24:20) and
    /// `selector` (bits 7:4), or `None` if the pair is not one of the twelve forms.
    ///
    /// `group` selects signedness and propagation: `0b00001` signed wrapping,
    /// `0b00010` signed saturating, `0b00011` signed halving, `0b00101` unsigned
    /// wrapping, `0b00110` unsigned saturating and `0b00111` unsigned halving.
    pub const fn parallel(group: u32, selector: u32) -> Option<Opcode> {
        let index = match selector & 0xf {
            0b0001 => 0,
            0b0011 => 1,
            0b0101 => 2,
            0b0111 => 3,
            0b1001 => 4,
            0b1111 => 5,
            _ => return None,
        };
        let table: [Opcode; 6] = match group & 0x1f {
            0b00001 => [
                Opcode::Sadd16,
                Opcode::Sasx,
                Opcode::Ssax,
                Opcode::Ssub16,
                Opcode::Sadd8,
                Opcode::Ssub8,
            ],
            0b00010 => [
                Opcode::Qadd16,
                Opcode::Qasx,
                Opcode::Qsax,
                Opcode::Qsub16,
                Opcode::Qadd8,
                Opcode::Qsub8,
            ],
            0b00011 => [
                Opcode::Shadd16,
                Opcode::Shasx,
                Opcode::Shsax,
                Opcode::Shsub16,
                Opcode::Shadd8,
                Opcode::Shsub8,
            ],
            0b00101 => [
                Opcode::Uadd16,
                Opcode::Uasx,
                Opcode::Usax,
                Opcode::Usub16,
                Opcode::Uadd8,
                Opcode::Usub8,
            ],
            0b00110 => [
                Opcode::Uqadd16,
                Opcode::Uqasx,
                Opcode::Uqsax,
                Opcode::Uqsub16,
                Opcode::Uqadd8,
                Opcode::Uqsub8,
            ],
            0b00111 => [
                Opcode::Uhadd16,
                Opcode::Uhasx,
                Opcode::Uhsax,
                Opcode::Uhsub16,
                Opcode::Uhadd8,
                Opcode::Uhsub8,
            ],
            _ => return None,
        };
        Some(table[index])
    }
}
