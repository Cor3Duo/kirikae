use std::fmt;

use super::opcode::Opcode;

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

    pub const fn new(number: u8) -> Option<Reg> {
        if number < 16 { Some(Reg(number)) } else { None }
    }

    pub const fn number(self) -> u8 {
        self.0
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DReg(u8);

impl DReg {
    pub const fn new(number: u8) -> Option<DReg> {
        if number < 32 {
            Some(DReg(number))
        } else {
            None
        }
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SReg(u8);

impl SReg {
    pub const fn new(number: u8) -> Option<SReg> {
        if number < 32 {
            Some(SReg(number))
        } else {
            None
        }
    }

    pub const fn from_parts(field: u8, high: bool) -> SReg {
        SReg(((field & 0xf) << 1) | high as u8)
    }

    pub const fn number(self) -> u8 {
        self.0
    }

    pub const fn d(self) -> DReg {
        DReg(self.0 / 2)
    }

    pub const fn high(self) -> bool {
        self.0 & 1 == 1
    }
}

impl fmt::Display for SReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VecShape {
    pub esize: u8,
    pub q: bool,
}

impl VecShape {
    pub const fn lanes(self) -> u8 {
        (if self.q { 128 } else { 64 }) / self.esize
    }

    pub const fn bytes(self) -> u8 {
        if self.q { 16 } else { 8 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum A32SysReg {
    Tpidrurw,
    Tpidruro,
    Cntpct,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VfpSys {
    Fpsid,
    Fpscr,
    Fpexc,
    Mvfr0,
    Mvfr1,
    Mvfr2,
}

impl VfpSys {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VfpDest {
    Core(Reg),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VBlock {
    pub first: DReg,
    pub words: u8,
    pub double: bool,
    pub base: Reg,
    pub mode: BlockMode,
    pub writeback: bool,
    pub implicit_base: bool,
}

impl VBlock {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostUpdate {
    None,
    Immediate,
    Register(Reg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NeonMem {
    pub base: Reg,
    pub first: DReg,
    pub count: u8,
    pub stride: u8,
    pub esize: u8,
    pub interleave: u8,
    pub align: Option<u8>,
    pub update: PostUpdate,
}

impl NeonMem {
    pub const fn bytes(&self) -> u32 {
        self.count as u32 * 8
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegList(u16);

impl RegList {
    pub const fn new(bits: u16) -> Option<RegList> {
        if bits == 0 { None } else { Some(RegList(bits)) }
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub const fn contains(self, reg: Reg) -> bool {
        self.0 >> reg.number() & 1 == 1
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecState {
    Arm,
    Thumb,
}

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
    Double,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Offset {
    Imm(u16),
    Reg { reg: Reg, shift: Shift },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mem {
    pub base: Reg,
    pub offset: Offset,
    pub add: bool,
    pub pre_indexed: bool,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    D(DReg),
    Lane {
        reg: DReg,
        index: u8,
    },
    S(SReg),
    Imm(u32),
    ImmFloat32(u32),
    ImmFloat64(u64),
    Imm64(u64),
    Shifted {
        reg: Reg,
        shift: Shift,
    },
    Regs {
        lo: Reg,
        hi: Reg,
    },
    Mem(Mem),
    Block(Block),
    VBlock(VBlock),
    NeonMem(NeonMem),
    VfpSys(VfpSys),
    VfpDest(VfpDest),
    Convert(Convert),
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Insn {
    pub pc: u32,
    pub word: u32,
    pub cond: Cond,
    pub opcode: Opcode,
    pub sets_flags: bool,
    pub shape: Option<VecShape>,
    pub operands: Vec<Operand>,
}

impl Insn {
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
            if matches!(operand, Operand::Convert(_)) {
                continue;
            }
            f.write_str(if first { " " } else { ", " })?;
            first = false;
            match operand {
                Operand::D(reg) if quad => write!(f, "q{}", reg.number() / 2)?,
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
