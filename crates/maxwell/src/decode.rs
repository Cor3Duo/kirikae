use crate::opcodes::{Form, OPCODES, Opcode};

pub const RZ: u8 = 255;
pub const PT: u8 = 7;

fn buckets() -> &'static [Vec<u16>; 1024] {
    static B: std::sync::OnceLock<[Vec<u16>; 1024]> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        let mut b: [Vec<u16>; 1024] = std::array::from_fn(|_| Vec::new());
        for (top, list) in b.iter_mut().enumerate() {
            let w = (top as u64) << 54;
            for (i, op) in OPCODES.iter().enumerate() {
                if (w ^ op.value) & op.mask & (0x3ff << 54) == 0 {
                    list.push(i as u16);
                }
            }
        }
        b
    })
}

pub fn lookup(w: u64) -> Option<&'static Opcode> {
    buckets()[(w >> 54) as usize].iter().map(|&i| &OPCODES[i as usize]).find(|op| w & op.mask == op.value)
}

#[derive(Clone, Copy, Debug)]
pub struct Insn {
    pub word: u64,
    pub op: &'static Opcode,
}

impl Insn {
    pub fn decode(word: u64) -> Option<Insn> {
        lookup(word).map(|op| Insn { word, op })
    }

    pub fn name(&self) -> &'static str {
        self.op.name
    }

    pub fn form(&self) -> Form {
        self.op.form
    }

    pub fn bits(&self, lo: u32, n: u32) -> u64 {
        (self.word >> lo) & ((1u64 << n) - 1)
    }

    pub fn bit(&self, b: u32) -> bool {
        self.word >> b & 1 == 1
    }

    pub fn rd(&self) -> u8 {
        self.bits(0, 8) as u8
    }
    pub fn ra(&self) -> u8 {
        self.bits(8, 8) as u8
    }
    pub fn rb(&self) -> u8 {
        self.bits(20, 8) as u8
    }
    pub fn rc(&self) -> u8 {
        self.bits(39, 8) as u8
    }
    pub fn rd2(&self) -> u8 {
        self.bits(28, 8) as u8
    }

    pub fn guard(&self) -> (u8, bool) {
        (self.bits(16, 3) as u8, self.bit(19))
    }

    pub fn pred(&self, lo: u32, not: u32) -> (u8, bool) {
        (self.bits(lo, 3) as u8, self.bit(not))
    }

    pub fn simm20(&self) -> i32 {
        let v = self.bits(20, 19) as u32 | (self.bits(56, 1) as u32) << 19;
        ((v << 12) as i32) >> 12
    }

    pub fn fimm20(&self) -> u32 {
        (self.bits(20, 19) as u32 | (self.bits(56, 1) as u32) << 19) << 12
    }

    pub fn imm32(&self) -> u32 {
        self.bits(20, 32) as u32
    }

    pub fn cbuf(&self) -> (u32, u32) {
        (self.bits(34, 5) as u32, (self.bits(20, 14) as u32) << 2)
    }

    pub fn branch_target(&self, pc: u32) -> u32 {
        let off = ((self.bits(20, 24) as u32) << 8) as i32 >> 8;
        let target = (pc as i32 + 8 + off) as u32;
        if target.is_multiple_of(32) { target + 8 } else { target }
    }

    pub fn half_operands(&self) -> HalfOperands {
        let top = self.word >> 48;
        let reg = |r: u64, swizzle: u64| HalfSrc::Reg { reg: r as u8, swizzle: swizzle as u8 };
        let op = |src: HalfSrc, neg: bool, abs: bool| HalfOperand { src, neg, abs };
        let (slot, offset) = self.cbuf();
        let cbuf = HalfSrc::Cbuf { slot, offset };
        let imm = HalfSrc::Imm((self.bits(20, 10) as u32) << 6 | (self.bits(30, 9) as u32 | (self.bit(56) as u32) << 9) << 22);
        let a_mods = op(reg(self.bits(8, 8), self.bits(47, 2)), self.bit(43), self.bit(44));
        let reg_b = op(reg(self.bits(20, 8), self.bits(28, 2)), self.bit(31), self.bit(30));
        let (a, b, c, sat) = match (self.name(), top & 0xfff8, top & 0xfe80) {
            ("hfma2", 0x5d00, _) => {
                let c = op(reg(self.bits(39, 8), self.bits(35, 2)), self.bit(30), false);
                (a_mods.plain(), op(reg_b.src, self.bit(31), false), Some(c), self.bit(32))
            }
            ("hfma2", ..) => {
                let r39 = reg(self.bits(39, 8), self.bits(53, 2));
                let (b, c) = match top & 0xf880 {
                    0x6080 => (op(r39, self.bit(56), false), op(cbuf, self.bit(51), false)),
                    0x7080 => (op(cbuf, self.bit(56), false), op(r39, self.bit(51), false)),
                    _ => (op(imm, false, false), op(r39, self.bit(51), false)),
                };
                (a_mods.plain(), b, Some(c), self.bit(52))
            }
            ("hmul2", 0x5d08, _) => (op(a_mods.src, false, a_mods.abs), reg_b, None, self.bit(32)),
            (_, 0x5d08 | 0x5d10 | 0x5d18 | 0x5d20, _) => (a_mods, reg_b, None, self.bit(32)),
            (_, _, 0x7a80 | 0x7c80 | 0x7e80) => (a_mods, op(cbuf, self.bit(56), self.bit(54)), None, self.bit(52)),
            (_, _, 0x7880) => (a_mods, op(cbuf, false, self.bit(54)), None, self.bit(52)),
            _ => (a_mods, op(imm, false, false), None, self.bit(52)),
        };
        HalfOperands { a, b, c, sat, mode: self.bits(49, 2) as u8 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfSrc {
    Reg { reg: u8, swizzle: u8 },
    Cbuf { slot: u32, offset: u32 },
    Imm(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalfOperand {
    pub src: HalfSrc,
    pub neg: bool,
    pub abs: bool,
}

impl HalfOperand {
    fn plain(self) -> HalfOperand {
        HalfOperand { neg: false, abs: false, ..self }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalfOperands {
    pub a: HalfOperand,
    pub b: HalfOperand,
    pub c: Option<HalfOperand>,
    pub sat: bool,
    pub mode: u8,
}

pub fn instructions(code: &[u8]) -> impl Iterator<Item = (u32, u64)> + '_ {
    code.as_chunks::<8>().0.iter().enumerate().filter(|(i, _)| i % 4 != 0).map(|(i, b)| ((i * 8) as u32, u64::from_le_bytes(*b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GUARD_PT: u64 = 7 << 16;

    #[test]
    fn table_lookup() {
        assert_eq!(lookup(0xe300_0000_0000_0000 | GUARD_PT | 0xf).map(|o| o.name), Some("exit"));
        let mov = Insn::decode(0x0100_0000_0000_0000 | (0x3f80_0000u64 << 20) | GUARD_PT).unwrap();
        assert_eq!(mov.name(), "mov32i");
        assert_eq!(mov.imm32(), 0x3f80_0000);
        let ffma = Insn::decode(0x5980_0000_0000_0000 | (3 << 39) | (2 << 20) | GUARD_PT | (1 << 8)).unwrap();
        assert_eq!((ffma.name(), ffma.rd(), ffma.ra(), ffma.rb(), ffma.rc()), ("ffma", 0, 1, 2, 3));
        assert_eq!(ffma.guard(), (PT, false));
        for op in OPCODES {
            assert!(lookup(op.value).is_some(), "{}", op.name);
        }
    }

    #[test]
    fn immediates() {
        let w = (0x7ffffu64 << 20) | (1 << 56) | 0x3810_0000_0000_0000;
        assert_eq!(Insn::decode(w).unwrap().simm20(), -1);
        let bra = Insn::decode(0xe240_0000_0000_0000 | (0x10 << 20) | GUARD_PT | 0xf).unwrap();
        assert_eq!(bra.name(), "bra");
        assert_eq!(bra.branch_target(0x30), 0x48);
        assert_eq!(bra.branch_target(0x28), 0x48);
    }
}
