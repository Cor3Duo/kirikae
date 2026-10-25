//! A32 instruction mnemonics.
//!
//! One variant per assembly mnemonic, so a decoded [`super::Insn`] can be matched on
//! the operation alone. [`Opcode::ALL`] lists every variant; the decoder and the
//! interpreter both have to cover it, which the compiler enforces with exhaustive
//! `match`.

/// How the element size of a NEON instruction appears in its mnemonic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ElemStyle {
    /// No suffix (`vand`, `vorr`, `veor`, `vmov` between registers).
    Hidden,
    /// `.8`, `.16`, `.32`, `.64` (`vld1`, `vst1`, `vext`, `vdup`).
    Plain,
    /// `.i8` … (three-register integer arithmetic, `vceq`, `vtst`).
    Int,
    /// `.s8` … (signed comparisons and saturating forms).
    Signed,
    /// `.u8` … (unsigned comparisons and saturating forms).
    Unsigned,
    /// `.f32` (Advanced SIMD floating point, which has no 64-bit form).
    Float,
}

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
/// | `smuad`, `smuadx`, `smusd`, `smusdx` | `[Rd, Rn, Rm]` |
/// | `smlad`, `smladx`, `smlsd`, `smlsdx` | `[Rd, Rn, Rm, Ra]` |
/// | `smlald`, `smlaldx`, `smlsld`, `smlsldx` | `[Regs { lo: RdLo, hi: RdHi }, Rn, Rm]` |
/// | `smmul`, `smmulr` | `[Rd, Rn, Rm]` |
/// | `smmla`, `smmlar`, `smmls`, `smmlsr` | `[Rd, Rn, Rm, Ra]` |
/// | `ldr`, `str`, `ldrh`, `strh`, `ldrsb`, `ldrsh`, `ldrex*` | `[Reg(rt), Mem]` |
/// | `ldrd`, `strd` | `[Regs { lo: Rt, hi: Rt + 1 }, Mem]` |
/// | `lda`, `ldab`, `ldah` | `[Reg(t), Mem]` |
/// | `stl`, `stlb`, `stlh` | `[Reg(t), Mem]` |
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
/// | `udiv`, `sdiv` | `[Rd, Rn, Rm]` |
/// | `mrs` | `[Rd]` |
/// | `msr` | `[Reg(Rm)]` or `[Imm(value)]` |
/// | `nop`, `yield`, `wfe`, `wfi`, `sev` | `[Imm(hint)]` |
/// | `sevl` | `[Imm(hint)]` |
/// | `dsb`, `dmb`, `isb`, `clrex` | `[Imm(option)]` |
/// | `pld` | `[Mem]` |
/// | `svc` | `[Imm(imm24)]` |
/// | `bkpt` | `[Imm(imm16)]` |
/// | `vldr`, `vstr` | `[S or D, Mem]` |
/// | `vldm`, `vstm` (also `vpush`, `vpop`) | `[VBlock]` |
/// | `vmov` | `[S, S]`, `[D, D]`, `[S, Reg]`, `[Reg, S]`, `[D, Reg, Reg]`, `[Reg, Reg, D]` |
/// | `vmov` immediate | `[S, ImmFloat32]` or `[D, ImmFloat64]` |
/// | `vmov` element | `[D, Imm(lane), Reg]` or `[Reg, D, Imm(lane)]` |
/// | `vdup` | `[D, Reg]` |
/// | `vmrs` | `[VfpDest, VfpSys]` |
/// | `vmsr` | `[VfpSys, Reg]` |
/// | `mrc`, `mcr` | `[Reg(rt), SysReg]` |
/// | `mrrc` | `[Regs { lo: Rt, hi: Rt2 }, SysReg]` |
/// | `vmla` … `vdiv`, `vfma` … `vfnms` | `[S|D(dest), S|D(n), S|D(m)]` |
/// | `vabs`, `vneg`, `vsqrt`, `vrint*` | `[S|D(dest), S|D(m)]` |
/// | `vcmp`, `vcmpe` | `[S|D(n), S|D(m)]` or `[S|D(n), Imm(0)]` |
/// | `vcvt` | `[S|D(dest), S|D(src), Convert]` |
/// | `vld1`/`vst1`/`vld2`/`vst2` | `[NeonMem]` |
/// | `vext` | `[D, D, D, Imm(byte_index)]` |
/// | NEON `vadd`, `vsub`, `vmul`, `vand`, `vorr`, `veor`, `vbic`, `vorn` | `[D, D, D]` |
/// | NEON `vabs`, `vneg` | `[D, D]` |
/// | NEON `vtbl` | `[D, D, D]` |
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
    // Media: dual multiply and multiply-accumulate.
    Smuad,
    Smuadx,
    Smusd,
    Smusdx,
    Smlad,
    Smladx,
    Smlsd,
    Smlsdx,
    Smlald,
    Smlaldx,
    Smlsld,
    Smlsldx,
    Smmul,
    Smmulr,
    Smmla,
    Smmlar,
    Smmls,
    Smmlsr,
    // Extra loads and stores.
    Ldrh,
    Strh,
    Lda,
    Ldab,
    Ldah,
    Stl,
    Stlb,
    Stlh,
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
    // Integer divide.
    Udiv,
    Sdiv,
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
    Sevl,
    Dsb,
    Dmb,
    Isb,
    Clrex,
    Pld,
    // VFP: transfers.
    Vldr,
    Vstr,
    Vldm,
    Vstm,
    Vpush,
    Vpop,
    Vmov,
    VmovLane,
    VmovLaneS,
    VmovLaneU,
    Vdup,
    Vmrs,
    Vmsr,
    Mrc,
    Mcr,
    Mrrc,
    // VFP: arithmetic.
    Vmla,
    Vmls,
    Vnmla,
    Vnmls,
    Vmul,
    Vnmul,
    Vadd,
    Vsub,
    Vdiv,
    Vfnma,
    Vfnms,
    Vfma,
    Vfms,
    Vabs,
    Vneg,
    Vsqrt,
    Vcmp,
    Vcmpe,
    Vrintr,
    Vrintz,
    Vrintx,
    Vcvt,
    VcvtR,
    // Advanced SIMD.
    Vld1,
    Vst1,
    Vld2,
    Vst2,
    Vld3,
    Vst3,
    Vld4,
    Vst4,
    Vext,
    Vand,
    Vorr,
    Veor,
    Vbic,
    Vorn,
    VaddF,
    VsubF,
    VmulF,
    VmlaF,
    VmlsF,
    VmlaI,
    VmlsI,
    VmaxF,
    VminF,
    Vbsl,
    Vmax,
    VmaxU,
    Vmin,
    VminU,
    Vcgt,
    VcgtU,
    Vcge,
    VcgeU,
    Vceq,
    VceqF,
    VcgeF,
    Vtst,
    VabsU,
    VnegU,
    VabsF,
    VnegF,
    VsqrtF,
    Vrev16,
    Vrev32,
    Vrev64,
    Vmvn,
    VmovImm,
    VorrImm,
    VmvnImm,
    VbicImm,
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
        // Media: dual multiply and multiply-accumulate.
        Opcode::Smuad,
        Opcode::Smuadx,
        Opcode::Smusd,
        Opcode::Smusdx,
        Opcode::Smlad,
        Opcode::Smladx,
        Opcode::Smlsd,
        Opcode::Smlsdx,
        Opcode::Smlald,
        Opcode::Smlaldx,
        Opcode::Smlsld,
        Opcode::Smlsldx,
        Opcode::Smmul,
        Opcode::Smmulr,
        Opcode::Smmla,
        Opcode::Smmlar,
        Opcode::Smmls,
        Opcode::Smmlsr,
        // Extra loads and stores.
        Opcode::Ldrh,
        Opcode::Strh,
        Opcode::Lda,
        Opcode::Ldab,
        Opcode::Ldah,
        Opcode::Stl,
        Opcode::Stlb,
        Opcode::Stlh,
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
        Opcode::Udiv,
        Opcode::Sdiv,
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
        Opcode::Sevl,
        Opcode::Dsb,
        Opcode::Dmb,
        Opcode::Isb,
        Opcode::Clrex,
        Opcode::Pld,
        // VFP: transfers.
        Opcode::Vldr,
        Opcode::Vstr,
        Opcode::Vldm,
        Opcode::Vstm,
        Opcode::Vpush,
        Opcode::Vpop,
        Opcode::Vmov,
        Opcode::VmovLane,
        Opcode::VmovLaneS,
        Opcode::VmovLaneU,
        Opcode::Vdup,
        Opcode::Vmrs,
        Opcode::Vmsr,
        Opcode::Mrc,
        Opcode::Mcr,
        Opcode::Mrrc,
        // VFP: arithmetic.
        Opcode::Vmla,
        Opcode::Vmls,
        Opcode::Vnmla,
        Opcode::Vnmls,
        Opcode::Vmul,
        Opcode::Vnmul,
        Opcode::Vadd,
        Opcode::Vsub,
        Opcode::Vdiv,
        Opcode::Vfnma,
        Opcode::Vfnms,
        Opcode::Vfma,
        Opcode::Vfms,
        Opcode::Vabs,
        Opcode::Vneg,
        Opcode::Vsqrt,
        Opcode::Vcmp,
        Opcode::Vcmpe,
        Opcode::Vrintr,
        Opcode::Vrintz,
        Opcode::Vrintx,
        Opcode::Vcvt,
        Opcode::VcvtR,
        // Advanced SIMD.
        Opcode::Vld1,
        Opcode::Vst1,
        Opcode::Vld2,
        Opcode::Vst2,
        Opcode::Vld3,
        Opcode::Vst3,
        Opcode::Vld4,
        Opcode::Vst4,
        Opcode::Vext,
        Opcode::Vand,
        Opcode::Vorr,
        Opcode::Veor,
        Opcode::Vbic,
        Opcode::Vorn,
        Opcode::VaddF,
        Opcode::VsubF,
        Opcode::VmulF,
        Opcode::VmlaF,
        Opcode::VmlsF,
        Opcode::VmlaI,
        Opcode::VmlsI,
        Opcode::VmaxF,
        Opcode::VminF,
        Opcode::Vbsl,
        Opcode::Vmax,
        Opcode::VmaxU,
        Opcode::Vmin,
        Opcode::VminU,
        Opcode::Vcgt,
        Opcode::VcgtU,
        Opcode::Vcge,
        Opcode::VcgeU,
        Opcode::Vceq,
        Opcode::VceqF,
        Opcode::VcgeF,
        Opcode::Vtst,
        Opcode::VabsU,
        Opcode::VnegU,
        Opcode::VabsF,
        Opcode::VnegF,
        Opcode::VsqrtF,
        Opcode::Vrev16,
        Opcode::Vrev32,
        Opcode::Vrev64,
        Opcode::Vmvn,
        Opcode::VmovImm,
        Opcode::VorrImm,
        Opcode::VmvnImm,
        Opcode::VbicImm,
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
            Opcode::Smuad => "smuad",
            Opcode::Smuadx => "smuadx",
            Opcode::Smusd => "smusd",
            Opcode::Smusdx => "smusdx",
            Opcode::Smlad => "smlad",
            Opcode::Smladx => "smladx",
            Opcode::Smlsd => "smlsd",
            Opcode::Smlsdx => "smlsdx",
            Opcode::Smlald => "smlald",
            Opcode::Smlaldx => "smlaldx",
            Opcode::Smlsld => "smlsld",
            Opcode::Smlsldx => "smlsldx",
            Opcode::Smmul => "smmul",
            Opcode::Smmulr => "smmulr",
            Opcode::Smmla => "smmla",
            Opcode::Smmlar => "smmlar",
            Opcode::Smmls => "smmls",
            Opcode::Smmlsr => "smmlsr",
            Opcode::Ldrh => "ldrh",
            Opcode::Strh => "strh",
            Opcode::Lda => "lda",
            Opcode::Ldab => "ldab",
            Opcode::Ldah => "ldah",
            Opcode::Stl => "stl",
            Opcode::Stlb => "stlb",
            Opcode::Stlh => "stlh",
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
            Opcode::Udiv => "udiv",
            Opcode::Sdiv => "sdiv",
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
            Opcode::Sevl => "sevl",
            Opcode::Dsb => "dsb",
            Opcode::Dmb => "dmb",
            Opcode::Isb => "isb",
            Opcode::Clrex => "clrex",
            Opcode::Pld => "pld",
            Opcode::Vldr => "vldr",
            Opcode::Vstr => "vstr",
            Opcode::Vldm => "vldm",
            Opcode::Vstm => "vstm",
            Opcode::Vpush => "vpush",
            Opcode::Vpop => "vpop",
            Opcode::Vmov => "vmov",
            Opcode::VmovLane => "vmovlane",
            Opcode::VmovLaneS => "vmovlanes",
            Opcode::VmovLaneU => "vmovlaneu",
            Opcode::Vdup => "vdup",
            Opcode::Vmrs => "vmrs",
            Opcode::Vmsr => "vmsr",
            Opcode::Mrc => "mrc",
            Opcode::Mcr => "mcr",
            Opcode::Mrrc => "mrrc",
            Opcode::Vmla => "vmla",
            Opcode::Vmls => "vmls",
            Opcode::Vnmla => "vnmla",
            Opcode::Vnmls => "vnmls",
            Opcode::Vmul => "vmul",
            Opcode::Vnmul => "vnmul",
            Opcode::Vadd => "vadd",
            Opcode::Vsub => "vsub",
            Opcode::Vdiv => "vdiv",
            Opcode::Vfnma => "vfnma",
            Opcode::Vfnms => "vfnms",
            Opcode::Vfma => "vfma",
            Opcode::Vfms => "vfms",
            Opcode::Vabs => "vabs",
            Opcode::Vneg => "vneg",
            Opcode::Vsqrt => "vsqrt",
            Opcode::Vcmp => "vcmp",
            Opcode::Vcmpe => "vcmpe",
            Opcode::Vrintr => "vrintr",
            Opcode::Vrintz => "vrintz",
            Opcode::Vrintx => "vrintx",
            Opcode::Vcvt => "vcvt",
            Opcode::VcvtR => "vcvtr",
            Opcode::Vld1 => "vld1",
            Opcode::Vst1 => "vst1",
            Opcode::Vld2 => "vld2",
            Opcode::Vst2 => "vst2",
            Opcode::Vld3 => "vld3",
            Opcode::Vst3 => "vst3",
            Opcode::Vld4 => "vld4",
            Opcode::Vst4 => "vst4",
            Opcode::Vext => "vext",
            Opcode::Vand => "vand",
            Opcode::Vorr => "vorr",
            Opcode::Veor => "veor",
            Opcode::Vbic => "vbic",
            Opcode::Vorn => "vorn",
            Opcode::VaddF => "vaddf",
            Opcode::VsubF => "vsubf",
            Opcode::VmulF => "vmulf",
            Opcode::VmlaF => "vmlaf",
            Opcode::VmlsF => "vmlsf",
            Opcode::VmlaI => "vmlai",
            Opcode::VmlsI => "vmlsi",
            Opcode::VmaxF => "vmaxf",
            Opcode::VminF => "vminf",
            Opcode::Vbsl => "vbsl",
            Opcode::Vmax => "vmax",
            Opcode::VmaxU => "vmaxu",
            Opcode::Vmin => "vmin",
            Opcode::VminU => "vminu",
            Opcode::Vcgt => "vcgt",
            Opcode::VcgtU => "vcgtu",
            Opcode::Vcge => "vcge",
            Opcode::VcgeU => "vcgeu",
            Opcode::Vceq => "vceq",
            Opcode::VceqF => "vceqf",
            Opcode::VcgeF => "vcgef",
            Opcode::Vtst => "vtst",
            Opcode::VabsU => "vabsu",
            Opcode::VnegU => "vnegu",
            Opcode::VabsF => "vabsf",
            Opcode::VnegF => "vnegf",
            Opcode::VsqrtF => "vsqrtf",
            Opcode::Vrev16 => "vrev16",
            Opcode::Vrev32 => "vrev32",
            Opcode::Vrev64 => "vrev64",
            Opcode::Vmvn => "vmvn",
            Opcode::VmovImm => "vmovimm",
            Opcode::VorrImm => "vorrimm",
            Opcode::VmvnImm => "vmvnimm",
            Opcode::VbicImm => "vbicimm",
        }
    }

    /// The base mnemonic as printed.
    ///
    /// This differs from [`Opcode::name`] only for the variants whose tables share one
    /// assembly mnemonic with a different form: the modified immediate is a `vmov`,
    /// `vorr`, `vmvn` or `vbic`, and the element transfers are `vmov`s.
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Opcode::VmovImm => "vmov",
            Opcode::VmovLane | Opcode::VmovLaneS | Opcode::VmovLaneU => "vmov",
            Opcode::VorrImm => "vorr",
            Opcode::VmvnImm => "vmvn",
            Opcode::VbicImm => "vbic",
            Opcode::VaddF => "vadd",
            Opcode::VsubF => "vsub",
            Opcode::VmulF => "vmul",
            Opcode::VmlaF => "vmla",
            Opcode::VmlsF => "vmls",
            Opcode::VmlaI => "vmla",
            Opcode::VmlsI => "vmls",
            Opcode::VmaxF => "vmax",
            Opcode::VminF => "vmin",
            Opcode::VmaxU => "vmax",
            Opcode::VminU => "vmin",
            Opcode::VcgtU => "vcgt",
            Opcode::VcgeU => "vcge",
            Opcode::VceqF => "vceq",
            Opcode::VcgeF => "vcge",
            Opcode::VabsU => "vabs",
            Opcode::VnegU => "vneg",
            Opcode::VabsF => "vabs",
            Opcode::VnegF => "vneg",
            Opcode::VsqrtF => "vsqrt",
            other => other.name(),
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

    /// Whether the mnemonic carries the operand precision (`.f32`/`.f64`): the scalar
    /// VFP arithmetic, comparison and rounding instructions.
    ///
    /// `vmov` and `vcvt` add their suffix from the operand types instead, because the
    /// same mnemonic also names the core and element transfers and the conversions.
    pub const fn takes_precision_suffix(self) -> bool {
        matches!(
            self,
            Opcode::Vmla
                | Opcode::Vmls
                | Opcode::Vnmla
                | Opcode::Vnmls
                | Opcode::Vmul
                | Opcode::Vnmul
                | Opcode::Vadd
                | Opcode::Vsub
                | Opcode::Vdiv
                | Opcode::Vfnma
                | Opcode::Vfnms
                | Opcode::Vfma
                | Opcode::Vfms
                | Opcode::Vabs
                | Opcode::Vneg
                | Opcode::Vsqrt
                | Opcode::Vcmp
                | Opcode::Vcmpe
                | Opcode::Vrintr
                | Opcode::Vrintz
                | Opcode::Vrintx
        )
    }

    /// How the element size of this NEON instruction appears in its mnemonic.
    const fn elem_style(self) -> ElemStyle {
        match self {
            Opcode::Vld1
            | Opcode::Vst1
            | Opcode::Vld2
            | Opcode::Vst2
            | Opcode::Vld3
            | Opcode::Vst3
            | Opcode::Vld4
            | Opcode::Vst4
            | Opcode::Vext
            | Opcode::Vdup
            | Opcode::Vtst
            | Opcode::VmovLane
            | Opcode::Vrev16
            | Opcode::Vrev32
            | Opcode::Vrev64 => ElemStyle::Plain,
            Opcode::Vadd
            | Opcode::Vsub
            | Opcode::Vmul
            | Opcode::Vceq
            | Opcode::VmlaI
            | Opcode::VmlsI
            | Opcode::VmovImm
            | Opcode::VorrImm
            | Opcode::VmvnImm
            | Opcode::VbicImm => ElemStyle::Int,
            Opcode::Vcgt
            | Opcode::Vcge
            | Opcode::Vmax
            | Opcode::Vmin
            | Opcode::Vabs
            | Opcode::Vneg
            | Opcode::VmovLaneS => ElemStyle::Signed,
            Opcode::VcgtU
            | Opcode::VcgeU
            | Opcode::VmaxU
            | Opcode::VminU
            | Opcode::VabsU
            | Opcode::VnegU
            | Opcode::VmovLaneU => ElemStyle::Unsigned,
            Opcode::VaddF
            | Opcode::VsubF
            | Opcode::VmulF
            | Opcode::VmlaF
            | Opcode::VmlsF
            | Opcode::Vfma
            | Opcode::Vfms
            | Opcode::VmaxF
            | Opcode::VminF
            | Opcode::VceqF
            | Opcode::VcgeF
            | Opcode::VabsF
            | Opcode::VnegF
            | Opcode::VsqrtF => ElemStyle::Float,
            _ => ElemStyle::Hidden,
        }
    }

    /// Mnemonic suffix of an Advanced SIMD instruction, from its element size.
    fn vector_suffix(self, shape: super::ir::VecShape) -> Option<&'static str> {
        Some(match self.elem_style() {
            ElemStyle::Hidden => return None,
            ElemStyle::Float => "f32",
            ElemStyle::Plain => match shape.esize {
                8 => "8",
                16 => "16",
                32 => "32",
                _ => "64",
            },
            style => {
                let prefix = match style {
                    ElemStyle::Int => "i",
                    ElemStyle::Signed => "s",
                    _ => "u",
                };
                match (prefix, shape.esize) {
                    ("i", 8) => "i8",
                    ("i", 16) => "i16",
                    ("i", 32) => "i32",
                    ("i", _) => "i64",
                    ("s", 8) => "s8",
                    ("s", 16) => "s16",
                    ("s", 32) => "s32",
                    ("s", _) => "s64",
                    (_, 8) => "u8",
                    (_, 16) => "u16",
                    (_, 32) => "u32",
                    _ => "u64",
                }
            }
        })
    }

    /// Suffix [`super::Insn::mnemonic`] appends to the mnemonic.
    pub(crate) fn mnemonic_suffix(self, insn: &super::ir::Insn) -> Option<&'static str> {
        if let Some(shape) = insn.shape {
            return self.vector_suffix(shape);
        }
        if matches!(self, Opcode::Vcvt | Opcode::VcvtR)
            && let Some(super::ir::Operand::Convert(convert)) = insn.operands.get(2)
        {
            return Some(convert.suffix());
        }
        if matches!(self, Opcode::Vmov) {
            // The core and element transfers share the mnemonic with the register
            // moves, and only the latter print a precision.
            let fp = |operand: &super::ir::Operand| {
                matches!(
                    operand,
                    super::ir::Operand::S(_)
                        | super::ir::Operand::D(_)
                        | super::ir::Operand::ImmFloat32(_)
                        | super::ir::Operand::ImmFloat64(_)
                )
            };
            if !insn.operands.iter().all(fp) {
                return None;
            }
        } else if !self.takes_precision_suffix() {
            return None;
        }
        match insn.operands.first()? {
            super::ir::Operand::S(_) => Some("f32"),
            super::ir::Operand::D(_) => Some("f64"),
            _ => None,
        }
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
