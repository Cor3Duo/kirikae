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
    Word,
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
    /// Immediate. For data-processing instructions this is the rotated 32-bit value;
    /// for `movw`/`movt` a 16-bit value; for `svc`/`bkpt` the comment field.
    Imm(u32),
    /// Register with a shift, the `<op2>` of a data-processing instruction.
    Shifted {
        reg: Reg,
        shift: Shift,
    },
    Mem(Mem),
    Block(Block),
    Target(Target),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Reg(reg) => write!(f, "{reg}"),
            Operand::Imm(imm) => write!(f, "#{imm:#x}"),
            Operand::Shifted { reg, shift } => {
                write!(f, "{reg}")?;
                if shift.is_noop() {
                    Ok(())
                } else {
                    write!(f, ", {shift}")
                }
            }
            Operand::Mem(mem) => write!(f, "{mem}"),
            Operand::Block(block) => write!(f, "{block}"),
            Operand::Target(target) => write!(f, "#{:#x}", target.addr),
        }
    }
}

/// Decoded instruction mnemonic, without the condition suffix.
///
/// | mnemonic | operands |
/// |---|---|
/// | `and` … `bic`, except `mov`, `mvn` | `[Rd, Rn, op2]` |
/// | `mov`, `mvn` | `[Rd, op2]` |
/// | `tst`, `teq`, `cmp`, `cmn` | `[Rn, op2]` |
/// | `movw`, `movt` | `[Rd, Imm(imm16)]` |
/// | `b`, `bl` | `[Target]` |
/// | `bx` | `[Reg]` |
/// | `blx` | `[Reg]` (register form) or `[Target]` (immediate form, always Thumb) |
/// | `ldr`, `str` | `[Reg(rt), Mem]` |
/// | `ldm`, `stm` | `[Block]` |
/// | `mul` | `[Rd, Rm, Rs]` |
/// | `mla` | `[Rd, Rm, Rs, Ra]`, computing `Rd = Rm * Rs + Ra` (the assembler syntax
///   used by gas and LLVM, with the accumulator encoded in bits 15:12) |
/// | `svc` | `[Imm(imm24)]` |
/// | `bkpt` | `[Imm(imm16)]` |
///
/// `op2` is either `Operand::Imm` (already rotated) or `Operand::Shifted`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Opcode {
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
    B,
    Bl,
    Bx,
    Blx,
    Ldr,
    Str,
    Ldm,
    Stm,
    Mul,
    Mla,
    Svc,
    Bkpt,
}

impl Opcode {
    pub const ALL: &'static [Opcode] = &[
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
        Opcode::B,
        Opcode::Bl,
        Opcode::Bx,
        Opcode::Blx,
        Opcode::Ldr,
        Opcode::Str,
        Opcode::Ldm,
        Opcode::Stm,
        Opcode::Mul,
        Opcode::Mla,
        Opcode::Svc,
        Opcode::Bkpt,
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
            Opcode::Ldr => "ldr",
            Opcode::Str => "str",
            Opcode::Ldm => "ldm",
            Opcode::Stm => "stm",
            Opcode::Mul => "mul",
            Opcode::Mla => "mla",
            Opcode::Svc => "svc",
            Opcode::Bkpt => "bkpt",
        }
    }

    /// Data-processing instructions whose only result is NZCV.
    pub const fn is_flag_test(self) -> bool {
        matches!(self, Opcode::Tst | Opcode::Teq | Opcode::Cmp | Opcode::Cmn)
    }

    /// Data-processing instructions (`and` … `mvn`), including `movw`/`movt`.
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

    /// Whether the mnemonic takes the `S` suffix (`adds`, `movs`, `muls`).
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
    pub operands: Vec<Operand>,
}

impl Insn {
    /// Mnemonic including the width, flag and addressing-mode suffixes the assembly
    /// syntax uses (`ldrbne`, `adds`, `ldmiane`, `stmdb`).
    pub fn mnemonic(&self) -> String {
        let mut name = String::from(self.opcode.name());
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
        for (index, operand) in self.operands.iter().enumerate() {
            f.write_str(if index == 0 { " " } else { ", " })?;
            write!(f, "{operand}")?;
        }
        Ok(())
    }
}
