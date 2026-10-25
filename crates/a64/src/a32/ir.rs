//! A32 (ARM mode) instruction IR.
//!
//! One [`Insn`] per 32-bit ARM word: condition field, mnemonic, flag side effect and a
//! flat operand list. The operand layout is fixed per [`Opcode`] (see its table), so a
//! later lowering pass can match on `(opcode, operands)` without reading the encoding
//! again.
//!
//! The IR stays close to the encoding on purpose: shifts keep their kind and amount,
//! memory operands keep base, offset and addressing mode, and block transfers keep the
//! register list. That mirrors how `crate::ir` keeps A64 `Stmt`s close to the encoding
//! and lets one lowering routine serve both the interpreter and a Cranelift backend.

use std::fmt;

use super::opcode::Opcode;

/// One of the sixteen A32 registers.
///
/// Reading `PC` inside an instruction yields `address + 8`, as on the architecture; the
/// IR stores the instruction address in [`Insn::pc`] and leaves that rule to the
/// consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reg(u8);

impl Reg {
    pub const R0: Reg = Reg(0);
    pub const R1: Reg = Reg(1);
    pub const R2: Reg = Reg(2);
    pub const R3: Reg = Reg(3);
    pub const R4: Reg = Reg(4);
    pub const R5: Reg = Reg(5);
    pub const R6: Reg = Reg(6);
    pub const R7: Reg = Reg(7);
    pub const R8: Reg = Reg(8);
    pub const R9: Reg = Reg(9);
    pub const R10: Reg = Reg(10);
    pub const R11: Reg = Reg(11);
    pub const R12: Reg = Reg(12);
    pub const SP: Reg = Reg(13);
    pub const LR: Reg = Reg(14);
    pub const PC: Reg = Reg(15);

    pub const ALL: &'static [Reg] = &[
        Reg::R0,
        Reg::R1,
        Reg::R2,
        Reg::R3,
        Reg::R4,
        Reg::R5,
        Reg::R6,
        Reg::R7,
        Reg::R8,
        Reg::R9,
        Reg::R10,
        Reg::R11,
        Reg::R12,
        Reg::SP,
        Reg::LR,
        Reg::PC,
    ];

    /// Returns `None` for `number > 15`.
    pub const fn new(number: u8) -> Option<Reg> {
        if number < 16 { Some(Reg(number)) } else { None }
    }

    pub const fn number(self) -> u8 {
        self.0
    }

    /// Builds a register from the low four bits of an encoding field.
    pub(super) const fn from_encoding(bits: u32) -> Reg {
        Reg((bits & 0xf) as u8)
    }

    pub fn name(self) -> &'static str {
        const NAMES: [&str; 16] = [
            "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "sp",
            "lr", "pc",
        ];
        NAMES[self.0 as usize]
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Condition field, bits 31:28. Field value `0b1111` is not a condition and is
/// reported by [`Cond::from_field`] as `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Cond {
    Eq = 0,
    Ne = 1,
    Cs = 2,
    Cc = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
    Al = 14,
}

impl Cond {
    pub const ALL: &'static [Cond] = &[
        Cond::Eq,
        Cond::Ne,
        Cond::Cs,
        Cond::Cc,
        Cond::Mi,
        Cond::Pl,
        Cond::Vs,
        Cond::Vc,
        Cond::Hi,
        Cond::Ls,
        Cond::Ge,
        Cond::Lt,
        Cond::Gt,
        Cond::Le,
        Cond::Al,
    ];

    pub const fn from_field(field: u32) -> Option<Cond> {
        match field & 0xf {
            0 => Some(Cond::Eq),
            1 => Some(Cond::Ne),
            2 => Some(Cond::Cs),
            3 => Some(Cond::Cc),
            4 => Some(Cond::Mi),
            5 => Some(Cond::Pl),
            6 => Some(Cond::Vs),
            7 => Some(Cond::Vc),
            8 => Some(Cond::Hi),
            9 => Some(Cond::Ls),
            10 => Some(Cond::Ge),
            11 => Some(Cond::Lt),
            12 => Some(Cond::Gt),
            13 => Some(Cond::Le),
            14 => Some(Cond::Al),
            _ => None,
        }
    }

    pub const fn field(self) -> u32 {
        self as u32
    }

    /// Suffix as written in assembly (`eq`, `ne`, …); empty for `al`.
    pub const fn suffix(self) -> &'static str {
        match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Cs => "cs",
            Cond::Cc => "cc",
            Cond::Mi => "mi",
            Cond::Pl => "pl",
            Cond::Vs => "vs",
            Cond::Vc => "vc",
            Cond::Hi => "hi",
            Cond::Ls => "ls",
            Cond::Ge => "ge",
            Cond::Lt => "lt",
            Cond::Gt => "gt",
            Cond::Le => "le",
            Cond::Al => "",
        }
    }

    /// The complementary condition. `Al` has no complement in the encoding spaces this
    /// slice decodes, so it maps to itself.
    pub const fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Cs => Cond::Cc,
            Cond::Cc => Cond::Cs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
            Cond::Al => Cond::Al,
        }
    }

    /// Evaluates the condition against NZCV packed as the A64 IR packs it: N in bit 31,
    /// Z in 30, C in 29, V in 28.
    pub const fn holds(self, nzcv: u32) -> bool {
        let n = nzcv >> 31 & 1 == 1;
        let z = nzcv >> 30 & 1 == 1;
        let c = nzcv >> 29 & 1 == 1;
        let v = nzcv >> 28 & 1 == 1;
        match self {
            Cond::Eq => z,
            Cond::Ne => !z,
            Cond::Cs => c,
            Cond::Cc => !c,
            Cond::Mi => n,
            Cond::Pl => !n,
            Cond::Vs => v,
            Cond::Vc => !v,
            Cond::Hi => c && !z,
            Cond::Ls => !c || z,
            Cond::Ge => n == v,
            Cond::Lt => n != v,
            Cond::Gt => n == v && !z,
            Cond::Le => n != v || z,
            Cond::Al => true,
        }
    }
}

/// Type pair of a `vcvt` instruction, printed as the mnemonic suffix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Convert {
    F64FromF32,
    F32FromF64,
    F32FromU32,
    F32FromS32,
    U32FromF32,
    S32FromF32,
    F64FromU32,
    F64FromS32,
    U32FromF64,
    S32FromF64,
}

impl Convert {
    pub const fn suffix(self) -> &'static str {
        match self {
            Convert::F64FromF32 => "f64.f32",
            Convert::F32FromF64 => "f32.f64",
            Convert::F32FromU32 => "f32.u32",
            Convert::F32FromS32 => "f32.s32",
            Convert::U32FromF32 => "u32.f32",
            Convert::S32FromF32 => "s32.f32",
            Convert::F64FromU32 => "f64.u32",
            Convert::F64FromS32 => "f64.s32",
            Convert::U32FromF64 => "u32.f64",
            Convert::S32FromF64 => "s32.f64",
        }
    }
}

impl fmt::Display for Convert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.suffix())
    }
}

/// One of the 32 64-bit VFP/Advanced SIMD registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DReg(u8);

impl DReg {
    /// Returns `None` for `number > 31`.
    pub const fn new(number: u8) -> Option<DReg> {
        if number < 32 {
            Some(DReg(number))
        } else {
            None
        }
    }

    /// Builds the register named by a four-bit encoding field and its extension bit,
    /// which is the high bit of the number.
    pub const fn from_parts(field: u8, high: bool) -> DReg {
        DReg((field & 0xf) | (high as u8) << 4)
    }

    pub const fn number(self) -> u8 {
        self.0
    }
}

impl fmt::Display for DReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "d{}", self.0)
    }
}

/// One of the 32 32-bit single-precision registers.
///
/// `S(2n)` is the low word of `D(n)` and `S(2n + 1)` its high word, so writing a
/// single register only touches half of the double register it aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SReg(u8);

impl SReg {
    /// Returns `None` for `number > 31`.
    pub const fn new(number: u8) -> Option<SReg> {
        if number < 32 {
            Some(SReg(number))
        } else {
            None
        }
    }

    /// Builds the register named by a four-bit encoding field and its extension bit,
    /// which is the low bit of the number.
    pub const fn from_parts(field: u8, high: bool) -> SReg {
        SReg(((field & 0xf) << 1) | high as u8)
    }

    pub const fn number(self) -> u8 {
        self.0
    }

    /// The double register this one aliases.
    pub const fn d(self) -> DReg {
        DReg(self.0 / 2)
    }

    /// Whether this is the high word of [`SReg::d`].
    pub const fn high(self) -> bool {
        self.0 & 1 == 1
    }
}

impl fmt::Display for SReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// Element size and register width of an Advanced SIMD data-processing instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VecShape {
    /// Element size in bits: 8, 16, 32 or 64.
    pub esize: u8,
    /// The `Q` bit of the encoding: the operands are 128-bit registers (`q`), which
    /// the IR names by the first of their two `d` halves.
    pub q: bool,
}

impl VecShape {
    /// Lanes per register.
    pub const fn lanes(self) -> u8 {
        (if self.q { 128 } else { 64 }) / self.esize
    }

    /// Bytes per register.
    pub const fn bytes(self) -> u8 {
        if self.q { 16 } else { 8 }
    }
}

/// System register of the `cp15` transfers the slice models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum A32SysReg {
    /// `TPIDRURW`: the thread pointer, user read/write.
    Tpidrurw,
    /// `TPIDRURO`: the thread pointer, user read-only.
    Tpidruro,
    /// `CNTPCT`: the 64-bit physical counter, `c14` with opcode 0.
    Cntpct,
    /// `CNTVCT`: the 64-bit virtual counter, `c14` with opcode 1.
    Cntvct,
}

impl A32SysReg {
    pub const fn name(self) -> &'static str {
        match self {
            A32SysReg::Tpidrurw => "tpidrurw",
            A32SysReg::Tpidruro => "tpidruro",
            A32SysReg::Cntpct => "cntpct",
            A32SysReg::Cntvct => "cntvct",
        }
    }
}

impl fmt::Display for A32SysReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// System register named by `vmrs`/`vmsr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VfpSys {
    /// `fpsid`: read-only identification register.
    Fpsid,
    /// `fpscr`: the status and control register modelled by the interpreter.
    Fpscr,
    /// `fpexc`: the exception-control register.
    Fpexc,
    Mvfr0,
    Mvfr1,
    Mvfr2,
}

impl VfpSys {
    /// Returns `None` for the media registers and the reserved encodings.
    pub const fn from_field(field: u32) -> Option<VfpSys> {
        match field & 0xf {
            0 => Some(VfpSys::Fpsid),
            1 => Some(VfpSys::Fpscr),
            2 => Some(VfpSys::Fpexc),
            5 => Some(VfpSys::Mvfr2),
            6 => Some(VfpSys::Mvfr1),
            7 => Some(VfpSys::Mvfr0),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            VfpSys::Fpsid => "fpsid",
            VfpSys::Fpscr => "fpscr",
            VfpSys::Fpexc => "fpexc",
            VfpSys::Mvfr0 => "mvfr0",
            VfpSys::Mvfr1 => "mvfr1",
            VfpSys::Mvfr2 => "mvfr2",
        }
    }
}

impl fmt::Display for VfpSys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Destination of `vmrs`: a core register, or the condition flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VfpDest {
    Core(Reg),
    /// `APSR_nzcv`: the four flags are written from FPSCR bits 31:28.
    Apsr,
}

impl fmt::Display for VfpDest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VfpDest::Core(reg) => write!(f, "{reg}"),
            VfpDest::Apsr => f.write_str("apsr_nzcv"),
        }
    }
}

/// A contiguous run of registers transferred by `vldm`/`vstm`/`vpush`/`vpop`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VBlock {
    /// `d` register the list starts at. A 32-bit list starts at `s(2 * first)`.
    pub first: DReg,
    /// `imm8`: registers in 32-bit words. A doubleword list covers `words / 2` `d`
    /// registers, which is why `vpush {d8-d15}` encodes `words = 16`.
    pub words: u8,
    /// `sz`: the list is written as `d` registers rather than `s` registers.
    pub double: bool,
    pub base: Reg,
    pub mode: BlockMode,
    pub writeback: bool,
    /// `vpush`/`vpop` name `sp` implicitly, so the base is not printed.
    pub implicit_base: bool,
}

impl VBlock {
    /// Bytes moved by the transfer.
    pub const fn bytes(self) -> u32 {
        self.words as u32 * 4
    }
}

impl fmt::Display for VBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.implicit_base {
            write!(f, "{}", self.base)?;
            if self.writeback {
                f.write_str("!")?;
            }
            f.write_str(", ")?;
        }
        let prefix = if self.double { 'd' } else { 's' };
        // `words` counts 32-bit registers, so a doubleword list covers half as many.
        let count = if self.double {
            self.words / 2
        } else {
            self.words
        };
        let first = if self.double {
            self.first.number()
        } else {
            self.first.number() * 2
        };
        let last = first + count - 1;
        match count {
            1 => write!(f, "{{{prefix}{first}}}"),
            2 => write!(f, "{{{prefix}{first}, {prefix}{last}}}"),
            _ => write!(f, "{{{prefix}{first}-{prefix}{last}}}"),
        }
    }
}

/// Update applied to the base register of an Advanced SIMD structure transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostUpdate {
    /// `[r0]`: the base is left alone.
    None,
    /// `[r0]!`: the base is advanced by the size of the transfer.
    Immediate,
    /// `[r0], r1`: the base is advanced by a register after the access.
    Register(Reg),
}

/// Address operand of `vld1`…`vld4` and `vst1`…`vst4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NeonMem {
    pub base: Reg,
    /// First register of the list.
    pub first: DReg,
    /// Registers in the list, 1 to 4.
    pub count: u8,
    /// Distance between the registers of the list: 1, or 2 for the encodings that
    /// name every other register.
    pub stride: u8,
    /// Element size in bits: 8, 16, 32 or 64.
    pub esize: u8,
    /// Interleaved streams in memory: 1 for `vld1`/`vst1`, otherwise the number of
    /// registers.
    pub interleave: u8,
    /// Alignment in bytes the encoding promises, or `None` for unaligned accesses.
    pub align: Option<u8>,
    pub update: PostUpdate,
}

impl NeonMem {
    /// Bytes moved by the transfer.
    pub const fn bytes(&self) -> u32 {
        self.count as u32 * 8
    }

    /// Register number of slot `slot` of the list.
    pub const fn register_number(&self, slot: u8) -> u8 {
        self.first.number() + slot * self.stride
    }
}

impl fmt::Display for NeonMem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for slot in 0..self.count {
            if slot != 0 {
                f.write_str(", ")?;
            }
            write!(f, "d{}", self.register_number(slot))?;
        }
        write!(f, "}}, [{}]", self.base)?;
        match self.update {
            PostUpdate::None => Ok(()),
            PostUpdate::Immediate => f.write_str("!"),
            PostUpdate::Register(reg) => write!(f, ", {reg}"),
        }
    }
}

/// Shift applied to the second operand of a data-processing instruction or to the index
/// register of a load/store.
///
/// Immediate forms keep the architectural normalisation: `LSR #0` and `ASR #0` mean 32,
/// and `ROR #0` is decoded as [`Shift::Rrx`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shift {
    Lsl(u8),
    Lsr(u8),
    Asr(u8),
    Ror(u8),
    Rrx,
    LslReg(Reg),
    LsrReg(Reg),
    AsrReg(Reg),
    RorReg(Reg),
}

impl Shift {
    pub const fn is_noop(self) -> bool {
        matches!(self, Shift::Lsl(0))
    }
}

impl fmt::Display for Shift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Shift::Lsl(n) => write!(f, "lsl #{n}"),
            Shift::Lsr(n) => write!(f, "lsr #{n}"),
            Shift::Asr(n) => write!(f, "asr #{n}"),
            Shift::Ror(n) => write!(f, "ror #{n}"),
            Shift::Rrx => f.write_str("rrx"),
            Shift::LslReg(r) => write!(f, "lsl {r}"),
            Shift::LsrReg(r) => write!(f, "lsr {r}"),
            Shift::AsrReg(r) => write!(f, "asr {r}"),
            Shift::RorReg(r) => write!(f, "ror {r}"),
        }
    }
}

/// A register list of a block transfer, bit *n* set means register *n*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegList(u16);

impl RegList {
    /// Block transfers with an empty list are unpredictable, so they are not
    /// representable.
    pub const fn new(bits: u16) -> Option<RegList> {
        if bits == 0 { None } else { Some(RegList(bits)) }
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub const fn contains(self, reg: Reg) -> bool {
        self.0 >> reg.number() & 1 == 1
    }

    /// Registers in ascending order.
    pub fn iter(self) -> impl Iterator<Item = Reg> {
        (0..16u8)
            .filter(move |n| self.0 >> n & 1 == 1)
            .filter_map(Reg::new)
    }
}

impl fmt::Display for RegList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut first = true;
        let mut n = 0u8;
        while n < 16 {
            if self.0 >> n & 1 == 0 {
                n += 1;
                continue;
            }
            let start = n;
            while n + 1 < 16 && self.0 >> (n + 1) & 1 == 1 {
                n += 1;
            }
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            f.write_str(Reg::new(start).map_or("?", Reg::name))?;
            if n >= start + 2 {
                write!(f, "-{}", Reg::new(n).map_or("?", Reg::name))?;
            } else if n == start + 1 {
                write!(f, ", {}", Reg::new(n).map_or("?", Reg::name))?;
            }
            n += 1;
        }
        f.write_str("}")
    }
}

/// Instruction set state of a branch target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecState {
    Arm,
    Thumb,
}

/// Absolute branch target. `addr` is the address the core jumps to; the low bit is
/// clear, and the interworking branch sets it when it writes `PC`, which the consumer
/// derives from `state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    pub addr: u32,
    pub state: ExecState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemSize {
    Byte,
    Halfword,
    Word,
    /// A register pair: eight bytes, low word first. Only `ldrd`/`strd` use it.
    Double,
}

/// Offset of a single-register load or store.
///
/// Register offsets only ever carry immediate shifts on A32; the register-shift variants
/// of [`Shift`] are never produced here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Offset {
    Imm(u16),
    Reg { reg: Reg, shift: Shift },
}

/// Addressing mode of a single-register load or store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mem {
    pub base: Reg,
    pub offset: Offset,
    /// `U` bit: add the offset to the base instead of subtracting it.
    pub add: bool,
    /// `P` bit: the offset is applied before the access.
    pub pre_indexed: bool,
    /// `W` bit. A post-indexed access writes the base back even though its `W` bit must
    /// be zero; use [`Mem::writes_back`] for the effective rule.
    pub writeback: bool,
    pub size: MemSize,
}

impl Mem {
    pub const fn writes_back(&self) -> bool {
        !self.pre_indexed || self.writeback
    }
}

fn write_offset(f: &mut fmt::Formatter<'_>, offset: Offset, add: bool) -> fmt::Result {
    let sign = if add { "" } else { "-" };
    match offset {
        Offset::Imm(imm) => write!(f, "#{sign}{imm:#x}"),
        Offset::Reg { reg, shift } => {
            write!(f, "{sign}{reg}")?;
            if shift.is_noop() {
                Ok(())
            } else {
                write!(f, ", {shift}")
            }
        }
    }
}

impl fmt::Display for Mem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}", self.base)?;
        if self.pre_indexed {
            if self.offset != Offset::Imm(0) {
                f.write_str(", ")?;
                write_offset(f, self.offset, self.add)?;
            }
            f.write_str("]")?;
            if self.writeback {
                f.write_str("!")?;
            }
        } else {
            f.write_str("], ")?;
            write_offset(f, self.offset, self.add)?;
        }
        Ok(())
    }
}

/// Addressing mode of a block transfer, from the `P` and `U` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockMode {
    IncrementAfter,
    IncrementBefore,
    DecrementAfter,
    DecrementBefore,
}

impl BlockMode {
    pub const fn suffix(self) -> &'static str {
        match self {
            BlockMode::IncrementAfter => "ia",
            BlockMode::IncrementBefore => "ib",
            BlockMode::DecrementAfter => "da",
            BlockMode::DecrementBefore => "db",
        }
    }
}

/// Operand of a block transfer: base register, register list and addressing mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Block {
    pub base: Reg,
    pub list: RegList,
    pub mode: BlockMode,
    pub writeback: bool,
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)?;
        if self.writeback {
            f.write_str("!")?;
        }
        write!(f, ", {}", self.list)
    }
}

/// Operand of a decoded instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    /// A `d` register, or the first half of the `q` register of a NEON instruction.
    D(DReg),
    /// One element of a `d` register, `d0[3]`.
    Lane {
        reg: DReg,
        index: u8,
    },
    /// An `s` register.
    S(SReg),
    /// Immediate. For data-processing instructions this is the rotated 32-bit value;
    /// for `movw`/`movt` a 16-bit value; for `svc`/`bkpt` the comment field; for the
    /// lane and shift amounts of a NEON instruction the encoded index.
    Imm(u32),
    /// A `vmov.f32` immediate, already expanded from its eight encoded bits.
    ImmFloat32(u32),
    /// A `vmov.f64` immediate, already expanded from its eight encoded bits.
    ImmFloat64(u64),
    /// A 64-bit integer immediate, the expansion of a NEON modified immediate.
    Imm64(u64),
    /// Register with a shift, the `<op2>` of a data-processing instruction.
    Shifted {
        reg: Reg,
        shift: Shift,
    },
    /// Two consecutive registers of a pair instruction. `lo` comes first in the
    /// encoding (`Rt`, `RdLo`); `hi` is its pair (`Rt + 1`, `RdHi`).
    Regs {
        lo: Reg,
        hi: Reg,
    },
    Mem(Mem),
    Block(Block),
    /// Register list of a VFP block transfer.
    VBlock(VBlock),
    /// Address operand of an Advanced SIMD structure transfer.
    NeonMem(NeonMem),
    /// System register of `vmrs`/`vmsr`.
    VfpSys(VfpSys),
    /// Destination of `vmrs`.
    VfpDest(VfpDest),
    /// Type pair of a `vcvt`.
    Convert(Convert),
    /// System register of an `mrc`/`mcr`.
    SysReg(A32SysReg),
    Target(Target),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Reg(reg) => write!(f, "{reg}"),
            Operand::D(reg) => write!(f, "{reg}"),
            Operand::Lane { reg, index } => write!(f, "{reg}[{index}]"),
            Operand::S(reg) => write!(f, "{reg}"),
            Operand::Imm(imm) => write!(f, "#{imm:#x}"),
            Operand::ImmFloat32(bits) => write!(f, "#{}", f32::from_bits(*bits)),
            Operand::ImmFloat64(bits) => write!(f, "#{}", f64::from_bits(*bits)),
            Operand::Imm64(value) => write!(f, "#{value:#x}"),
            Operand::Shifted { reg, shift } => {
                write!(f, "{reg}")?;
                if shift.is_noop() {
                    Ok(())
                } else {
                    write!(f, ", {shift}")
                }
            }
            Operand::Regs { lo, hi } => write!(f, "{lo}, {hi}"),
            Operand::Mem(mem) => write!(f, "{mem}"),
            Operand::Block(block) => write!(f, "{block}"),
            Operand::VBlock(block) => write!(f, "{block}"),
            Operand::NeonMem(mem) => write!(f, "{mem}"),
            Operand::VfpSys(sys) => write!(f, "{sys}"),
            Operand::VfpDest(dest) => write!(f, "{dest}"),
            Operand::Convert(convert) => write!(f, "{convert}"),
            Operand::SysReg(reg) => write!(f, "{reg}"),
            Operand::Target(target) => write!(f, "#{:#x}", target.addr),
        }
    }
}

/// One decoded A32 instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Insn {
    /// Address of the instruction.
    pub pc: u32,
    /// The original encoding.
    pub word: u32,
    pub cond: Cond,
    pub opcode: Opcode,
    /// `S` bit of a data-processing instruction, or an implicit flag update for
    /// `tst`/`teq`/`cmp`/`cmn`.
    pub sets_flags: bool,
    /// Element size and width of an Advanced SIMD data-processing instruction; `None`
    /// for the ARM and VFP classes.
    pub shape: Option<VecShape>,
    pub operands: Vec<Operand>,
}

impl Insn {
    /// Mnemonic including the width, flag and addressing-mode suffixes the assembly
    /// syntax uses (`ldrbne`, `adds`, `ldmiane`, `stmdb`).
    pub fn mnemonic(&self) -> String {
        let mut name = String::from(self.opcode.mnemonic());
        if matches!(self.opcode, Opcode::Ldr | Opcode::Str)
            && let Some(Operand::Mem(mem)) = self.operands.last()
            && mem.size == MemSize::Byte
        {
            name.push('b');
        }
        if self.sets_flags && self.opcode.takes_s_suffix() {
            name.push('s');
        }
        if let Some(Operand::Block(block)) = self.operands.first() {
            name.push_str(block.mode.suffix());
        }
        if matches!(self.opcode, Opcode::Vldm | Opcode::Vstm)
            && let Some(Operand::VBlock(block)) = self.operands.first()
        {
            name.push_str(block.mode.suffix());
        }
        if let Some(suffix) = self.opcode.mnemonic_suffix(self) {
            name.push('.');
            name.push_str(suffix);
        }
        name.push_str(self.cond.suffix());
        name
    }

    pub fn next(&self) -> u32 {
        self.pc.wrapping_add(4)
    }

    /// Branch target, if the instruction is a direct branch.
    pub fn target(&self) -> Option<Target> {
        self.operands.iter().find_map(|operand| match operand {
            Operand::Target(target) => Some(*target),
            _ => None,
        })
    }
}

impl fmt::Display for Insn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.mnemonic())?;
        let quad = self.shape.is_some_and(|shape| shape.q);
        let mut first = true;
        for operand in self.operands.iter() {
            // The conversion type of a `vcvt` is printed as the mnemonic suffix.
            if matches!(operand, Operand::Convert(_)) {
                continue;
            }
            f.write_str(if first { " " } else { ", " })?;
            first = false;
            match operand {
                // A 128-bit form names the quadword even though the IR stores the first
                // of its two `d` halves.
                Operand::D(reg) if quad => write!(f, "q{}", reg.number() / 2)?,
                // A modified immediate prints one lane; the IR keeps the value that
                // fills the register.
                Operand::Imm64(value)
                    if matches!(
                        self.opcode,
                        Opcode::VmovImm | Opcode::VorrImm | Opcode::VmvnImm | Opcode::VbicImm
                    ) && self.shape.is_some() =>
                {
                    let esize = u32::from(
                        self.shape
                            .unwrap_or(VecShape {
                                esize: 64,
                                q: false,
                            })
                            .esize,
                    );
                    let mask = if esize >= 64 {
                        u64::MAX
                    } else {
                        (1u64 << esize) - 1
                    };
                    write!(f, "#{:#x}", value & mask)?
                }
                other => write!(f, "{other}")?,
            }
        }
        Ok(())
    }
}
