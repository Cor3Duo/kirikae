#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

pub const SP: u32 = 31;
pub const XZR: u32 = 31;
pub const LR: u32 = 30;
pub const FP: u32 = 29;

pub mod cond {
    pub const EQ: u32 = 0;
    pub const NE: u32 = 1;
    pub const HS: u32 = 2;
    pub const LO: u32 = 3;
    pub const MI: u32 = 4;
    pub const PL: u32 = 5;
    pub const HI: u32 = 8;
    pub const LS: u32 = 9;
    pub const GE: u32 = 10;
    pub const LT: u32 = 11;
    pub const GT: u32 = 12;
    pub const LE: u32 = 13;
    pub const AL: u32 = 14;
}

fn sf(is64: bool) -> u32 {
    (is64 as u32) << 31
}

pub fn movz(is64: bool, rd: u32, imm16: u32, hw: u32) -> u32 {
    sf(is64) | 0b10 << 29 | 0b100101 << 23 | hw << 21 | (imm16 & 0xffff) << 5 | rd
}
pub fn movk(is64: bool, rd: u32, imm16: u32, hw: u32) -> u32 {
    sf(is64) | 0b11 << 29 | 0b100101 << 23 | hw << 21 | (imm16 & 0xffff) << 5 | rd
}
pub fn movn(is64: bool, rd: u32, imm16: u32, hw: u32) -> u32 {
    sf(is64) | 0b100101 << 23 | hw << 21 | (imm16 & 0xffff) << 5 | rd
}
pub fn addsub_imm(
    is64: bool,
    sub: bool,
    setflags: bool,
    rd: u32,
    rn: u32,
    imm12: u32,
    lsl12: bool,
) -> u32 {
    sf(is64)
        | (sub as u32) << 30
        | (setflags as u32) << 29
        | 0b100010 << 23
        | (lsl12 as u32) << 22
        | (imm12 & 0xfff) << 10
        | rn << 5
        | rd
}
pub fn add_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    addsub_imm(true, false, false, rd, rn, imm12, false)
}
pub fn sub_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    addsub_imm(true, true, false, rd, rn, imm12, false)
}
pub fn subs_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    addsub_imm(true, true, true, rd, rn, imm12, false)
}
pub fn cmp_imm(rn: u32, imm12: u32) -> u32 {
    subs_imm(XZR, rn, imm12)
}
pub fn addsub_reg(
    is64: bool,
    sub: bool,
    setflags: bool,
    rd: u32,
    rn: u32,
    rm: u32,
    shift: u32,
    amount: u32,
) -> u32 {
    sf(is64)
        | (sub as u32) << 30
        | (setflags as u32) << 29
        | 0b01011 << 24
        | shift << 22
        | rm << 16
        | amount << 10
        | rn << 5
        | rd
}
pub fn add_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    addsub_reg(true, false, false, rd, rn, rm, 0, 0)
}
pub fn sub_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    addsub_reg(true, true, false, rd, rn, rm, 0, 0)
}
pub fn addsub_ext(
    is64: bool,
    sub: bool,
    setflags: bool,
    rd: u32,
    rn: u32,
    rm: u32,
    option: u32,
    shift: u32,
) -> u32 {
    sf(is64)
        | (sub as u32) << 30
        | (setflags as u32) << 29
        | 0b01011 << 24
        | 1 << 21
        | rm << 16
        | option << 13
        | shift << 10
        | rn << 5
        | rd
}
pub fn logical_reg(
    is64: bool,
    opc: u32,
    invert: bool,
    rd: u32,
    rn: u32,
    rm: u32,
    shift: u32,
    amount: u32,
) -> u32 {
    sf(is64)
        | opc << 29
        | 0b01010 << 24
        | shift << 22
        | (invert as u32) << 21
        | rm << 16
        | amount << 10
        | rn << 5
        | rd
}
pub fn mov_reg(rd: u32, rm: u32) -> u32 {
    logical_reg(true, 1, false, rd, XZR, rm, 0, 0)
}
pub fn logical_imm(is64: bool, opc: u32, rd: u32, rn: u32, n: u32, immr: u32, imms: u32) -> u32 {
    sf(is64) | opc << 29 | 0b100100 << 23 | n << 22 | immr << 16 | imms << 10 | rn << 5 | rd
}
pub fn bitfield(is64: bool, opc: u32, rd: u32, rn: u32, immr: u32, imms: u32) -> u32 {
    sf(is64)
        | opc << 29
        | 0b100110 << 23
        | (is64 as u32) << 22
        | immr << 16
        | imms << 10
        | rn << 5
        | rd
}
pub fn lsl_imm(rd: u32, rn: u32, amount: u32) -> u32 {
    bitfield(true, 2, rd, rn, (64 - amount) % 64, 63 - amount)
}
pub fn lsr_imm(rd: u32, rn: u32, amount: u32) -> u32 {
    bitfield(true, 2, rd, rn, amount, 63)
}
pub fn dp2(is64: bool, opcode: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    sf(is64) | 0b0011010110 << 21 | rm << 16 | opcode << 10 | rn << 5 | rd
}
pub fn udiv(rd: u32, rn: u32, rm: u32) -> u32 {
    dp2(true, 2, rd, rn, rm)
}
pub fn dp3(is64: bool, op31: u32, o0: bool, rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    sf(is64) | 0b11011 << 24 | op31 << 21 | rm << 16 | (o0 as u32) << 15 | ra << 10 | rn << 5 | rd
}
pub fn madd(rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    dp3(true, 0, false, rd, rn, rm, ra)
}
pub fn msub(rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    dp3(true, 0, true, rd, rn, rm, ra)
}
pub fn mul(rd: u32, rn: u32, rm: u32) -> u32 {
    madd(rd, rn, rm, XZR)
}
pub fn csel_family(is64: bool, op: u32, op2: u32, rd: u32, rn: u32, rm: u32, cc: u32) -> u32 {
    sf(is64) | op << 30 | 0b11010100 << 21 | rm << 16 | cc << 12 | op2 << 10 | rn << 5 | rd
}
pub fn csel(rd: u32, rn: u32, rm: u32, cc: u32) -> u32 {
    csel_family(true, 0, 0, rd, rn, rm, cc)
}
pub fn cset(rd: u32, cc: u32) -> u32 {
    csel_family(true, 0, 1, rd, XZR, XZR, cc ^ 1)
}
pub fn ldst_uimm(size: u32, opc: u32, rt: u32, rn: u32, offset: u32) -> u32 {
    size << 30 | 0b111001 << 24 | opc << 22 | (offset >> size) << 10 | rn << 5 | rt
}
pub fn ldr(rt: u32, rn: u32, offset: u32) -> u32 {
    ldst_uimm(3, 1, rt, rn, offset)
}
pub fn str(rt: u32, rn: u32, offset: u32) -> u32 {
    ldst_uimm(3, 0, rt, rn, offset)
}
pub fn ldst_imm9(size: u32, opc: u32, rt: u32, rn: u32, imm9: i32, idx: u32) -> u32 {
    size << 30
        | 0b111000 << 24
        | opc << 22
        | ((imm9 as u32) & 0x1ff) << 12
        | idx << 10
        | rn << 5
        | rt
}
pub fn ldst_reg(size: u32, opc: u32, rt: u32, rn: u32, rm: u32, option: u32, scaled: bool) -> u32 {
    size << 30
        | 0b111000 << 24
        | opc << 22
        | 1 << 21
        | rm << 16
        | option << 13
        | (scaled as u32) << 12
        | 0b10 << 10
        | rn << 5
        | rt
}
pub fn ldstp(opc: u32, idx: u32, load: bool, rt: u32, rt2: u32, rn: u32, offset: i32) -> u32 {
    let scale = if opc == 2 { 8 } else { 4 };
    opc << 30
        | 0b101 << 27
        | idx << 23
        | (load as u32) << 22
        | (((offset / scale) as u32) & 0x7f) << 15
        | rt2 << 10
        | rn << 5
        | rt
}
pub fn ldaxr(is64: bool, rt: u32, rn: u32) -> u32 {
    (2 + is64 as u32) << 30
        | 0b001000 << 24
        | 1 << 22
        | 31 << 16
        | 1 << 15
        | 31 << 10
        | rn << 5
        | rt
}
pub fn stlxr(is64: bool, rs: u32, rt: u32, rn: u32) -> u32 {
    (2 + is64 as u32) << 30 | 0b001000 << 24 | rs << 16 | 1 << 15 | 31 << 10 | rn << 5 | rt
}
pub fn adr(rd: u32, offset: i64) -> u32 {
    let imm = offset as u32;
    (imm & 3) << 29 | 0b10000 << 24 | ((imm >> 2) & 0x7ffff) << 5 | rd
}
pub fn adrp(rd: u32, pages: i64) -> u32 {
    1 << 31 | adr(rd, pages)
}
pub fn b(offset: i64) -> u32 {
    0b000101 << 26 | ((offset >> 2) as u32 & 0x3ff_ffff)
}
pub fn bl(offset: i64) -> u32 {
    0b100101 << 26 | ((offset >> 2) as u32 & 0x3ff_ffff)
}
pub fn bcond(cc: u32, offset: i64) -> u32 {
    0x5400_0000 | ((offset >> 2) as u32 & 0x7ffff) << 5 | cc
}
pub fn cbz(is64: bool, nonzero: bool, rt: u32, offset: i64) -> u32 {
    sf(is64) | 0b011010 << 25 | (nonzero as u32) << 24 | ((offset >> 2) as u32 & 0x7ffff) << 5 | rt
}
pub fn tbz(nonzero: bool, rt: u32, bit: u32, offset: i64) -> u32 {
    (bit >> 5) << 31
        | 0b011011 << 25
        | (nonzero as u32) << 24
        | (bit & 31) << 19
        | ((offset >> 2) as u32 & 0x3fff) << 5
        | rt
}
pub fn br(rn: u32) -> u32 {
    0xD61F_0000 | rn << 5
}
pub fn blr(rn: u32) -> u32 {
    0xD63F_0000 | rn << 5
}
pub fn ret() -> u32 {
    0xD65F_0000 | LR << 5
}
pub fn svc(imm: u32) -> u32 {
    0xD400_0001 | (imm & 0xffff) << 5
}
pub fn nop() -> u32 {
    0xD503_201F
}

/// Advanced SIMD three-same bitwise: `size` 0/1/2 picks AND/BIC/ORR/ORN with
/// `u = false` and EOR/BSL/BIT/BIF with `u = true`.
pub fn neon_logic(q: bool, u: bool, size: u32, rm: u32, rn: u32, rd: u32) -> u32 {
    (q as u32) << 30
        | (u as u32) << 29
        | 0b01110 << 24
        | (size & 3) << 22
        | 1 << 21
        | rm << 16
        | 0b00011 << 11
        | 1 << 10
        | rn << 5
        | rd
}

/// Advanced SIMD three-same arithmetic: `opcode` `0b10000` is `ADD`/`SUB` (`u` picks
/// `SUB`), `0b10011` is `MUL` with `u = false`, `0b01100`/`0b01101` are `MAX`/`MIN` with
/// `u` picking the unsigned form.
pub fn neon_three_same(q: bool, u: bool, size: u32, opcode: u32, rm: u32, rn: u32, rd: u32) -> u32 {
    (q as u32) << 30
        | (u as u32) << 29
        | 0b01110 << 24
        | (size & 3) << 22
        | 1 << 21
        | rm << 16
        | (opcode & 0x1f) << 11
        | 1 << 10
        | rn << 5
        | rd
}

/// Advanced SIMD two-register miscellaneous: `opcode` `0b01011` is `ABS`/`NEG` (`u`
/// picks `NEG`).
pub fn neon_two_misc(q: bool, u: bool, size: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
    (q as u32) << 30
        | (u as u32) << 29
        | 0b01110 << 24
        | (size & 3) << 22
        | 0b10000 << 17
        | (opcode & 0x1f) << 12
        | 0b10 << 10
        | rn << 5
        | rd
}

/// Advanced SIMD across-vector reduction: `opcode` `0b11011` is `ADDV` with `u = false`,
/// `0b00011` is `SADDLV` with `u = false` and `UADDLV` with `u = true`.
pub fn neon_across(q: bool, u: bool, size: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
    (q as u32) << 30
        | (u as u32) << 29
        | 0b01110 << 24
        | (size & 3) << 22
        | 0b11000 << 17
        | (opcode & 0x1f) << 12
        | 0b10 << 10
        | rn << 5
        | rd
}

/// Advanced SIMD shift by immediate, vector form. `immhb` is `immh:immb`: 0 gives
/// `SHL #es`, `2*es - n` gives USHR/SSHR `#n`.
pub fn neon_shift_imm(q: bool, u: bool, immhb: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
    (q as u32) << 30
        | (u as u32) << 29
        | 0b1111 << 24
        | (immhb & 0x7f) << 16
        | (opcode & 0x1f) << 11
        | 1 << 10
        | rn << 5
        | rd
}

/// `imm5` for the copy group: lane width `es = 8 << (imm5.trailing_zeros())`.
pub fn neon_imm5(es: u32, index: u32) -> u32 {
    let size = (es / 8).trailing_zeros();
    (1 << size) | (index << (size + 1))
}

/// Advanced SIMD copy: `op = 0` is DUP (element), `op = 1` is INS (element).
pub fn neon_copy(q: bool, op: bool, imm5: u32, imm4: u32, rn: u32, rd: u32) -> u32 {
    (q as u32) << 30
        | (op as u32) << 29
        | 0b01110 << 24
        | (imm5 & 0x1f) << 16
        | (imm4 & 0xf) << 11
        | 1 << 10
        | rn << 5
        | rd
}

#[derive(Clone, Copy)]
enum Fix {
    B,
    Bl,
    Bcond(u32),
    Cbz(bool, bool, u32),
    Tbz(bool, u32, u32),
    Adr(u32),
    Adrp(u32),
}

pub struct Asm {
    base: u64,
    words: Vec<u32>,
    labels: HashMap<String, u64>,
    fixups: Vec<(usize, String, Fix)>,
}

impl Asm {
    pub fn new(base: u64) -> Self {
        Asm {
            base,
            words: Vec::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    pub fn here(&self) -> u64 {
        self.base + 4 * self.words.len() as u64
    }

    pub fn emit(&mut self, word: u32) -> &mut Self {
        self.words.push(word);
        self
    }

    pub fn label(&mut self, name: &str) -> &mut Self {
        let at = self.here();
        assert!(
            self.labels.insert(name.to_string(), at).is_none(),
            "duplicate label {name}"
        );
        self
    }

    pub fn label_addr(&self, name: &str) -> u64 {
        *self
            .labels
            .get(name)
            .unwrap_or_else(|| panic!("undefined label {name}"))
    }

    pub fn define(&mut self, name: &str, addr: u64) -> &mut Self {
        self.labels.insert(name.to_string(), addr);
        self
    }

    fn fix(&mut self, name: &str, f: Fix) -> &mut Self {
        self.fixups.push((self.words.len(), name.to_string(), f));
        self.words.push(0);
        self
    }

    pub fn b_to(&mut self, l: &str) -> &mut Self {
        self.fix(l, Fix::B)
    }
    pub fn bl_to(&mut self, l: &str) -> &mut Self {
        self.fix(l, Fix::Bl)
    }
    pub fn bcond_to(&mut self, cc: u32, l: &str) -> &mut Self {
        self.fix(l, Fix::Bcond(cc))
    }
    pub fn cbz_to(&mut self, rt: u32, l: &str) -> &mut Self {
        self.fix(l, Fix::Cbz(true, false, rt))
    }
    pub fn cbnz_to(&mut self, rt: u32, l: &str) -> &mut Self {
        self.fix(l, Fix::Cbz(true, true, rt))
    }
    pub fn tbnz_to(&mut self, rt: u32, bit: u32, l: &str) -> &mut Self {
        self.fix(l, Fix::Tbz(true, rt, bit))
    }
    pub fn tbz_to(&mut self, rt: u32, bit: u32, l: &str) -> &mut Self {
        self.fix(l, Fix::Tbz(false, rt, bit))
    }
    pub fn adr_to(&mut self, rd: u32, l: &str) -> &mut Self {
        self.fix(l, Fix::Adr(rd))
    }
    pub fn addr_of(&mut self, rd: u32, l: &str) -> &mut Self {
        self.fix(l, Fix::Adrp(rd));
        self.fixups
            .push((self.words.len(), l.to_string(), Fix::Adrp(rd | 0x100)));
        self.words.push(0);
        self
    }

    pub fn mov_imm(&mut self, rd: u32, value: u64) -> &mut Self {
        self.emit(movz(true, rd, (value & 0xffff) as u32, 0));
        for hw in 1..4 {
            let part = ((value >> (16 * hw)) & 0xffff) as u32;
            if part != 0 {
                self.emit(movk(true, rd, part, hw));
            }
        }
        self
    }

    pub fn finish(mut self) -> Vec<u32> {
        for (index, name, fix) in std::mem::take(&mut self.fixups) {
            let target = *self
                .labels
                .get(&name)
                .unwrap_or_else(|| panic!("undefined label {name}"));
            let pc = self.base + 4 * index as u64;
            let off = target.wrapping_sub(pc) as i64;
            self.words[index] = match fix {
                Fix::B => b(off),
                Fix::Bl => bl(off),
                Fix::Bcond(cc) => bcond(cc, off),
                Fix::Cbz(is64, nz, rt) => cbz(is64, nz, rt, off),
                Fix::Tbz(nz, rt, bit) => tbz(nz, rt, bit, off),
                Fix::Adr(rd) => adr(rd, off),
                Fix::Adrp(rd) if rd & 0x100 == 0 => {
                    let pages = ((target & !0xfff) as i64 - (pc & !0xfff) as i64) >> 12;
                    adrp(rd, pages)
                }
                Fix::Adrp(rd) => {
                    let rd = rd & 0x1f;
                    add_imm(rd, rd, (target & 0xfff) as u32)
                }
            };
        }
        self.words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoders_match_a_real_assembler() {
        assert_eq!(nop(), 0xd503201f);
        assert_eq!(ret(), 0xd65f03c0);
        assert_eq!(svc(0x27), 0xd40004e1);
        assert_eq!(add_imm(0, 0, 1), 0x91000400);
        assert_eq!(add_imm(FP, SP, 0), 0x910003fd);
        assert_eq!(subs_imm(1, 1, 1), 0xf1000421);
        assert_eq!(cmp_imm(0, 5), 0xf100141f);
        assert_eq!(mov_reg(0, 1), 0xaa0103e0);
        assert_eq!(sub_reg(1, 1, 2), 0xcb020021);
        assert_eq!(udiv(0, 1, 2), 0x9ac20820);
        assert_eq!(msub(0, 1, 2, 3), 0x9b028c20);
        assert_eq!(csel(0, 1, 2, cond::EQ), 0x9a820020);
        assert_eq!(lsl_imm(0, 1, 3), 0xd37df020);
        assert_eq!(ldr(0, 1, 8), 0xf9400420);
        assert_eq!(ldst_imm9(0, 0, 5, 2, -1, 3), 0x381ffc45);
        assert_eq!(ldst_reg(2, 1, 10, 9, 0, 3, true), 0xb860792a);
        assert_eq!(ldstp(2, 3, false, FP, LR, SP, -16), 0xa9bf7bfd);
        assert_eq!(ldstp(2, 1, true, FP, LR, SP, 16), 0xa8c17bfd);
        assert_eq!(logical_imm(true, 0, 0, 1, 1, 0, 7), 0x92401c20);
        assert_eq!(movz(true, 0, 0, 0), 0xd2800000);
        assert_eq!(cbz(true, false, 0, 0), 0xb4000000);
        assert_eq!(br(9), 0xd61f0120);
        assert_eq!(blr(9), 0xd63f0120);
        assert_eq!(adrp(0, 0), 0x90000000);
        assert_eq!(cset(0, cond::EQ), 0x9a9f17e0);
        assert_eq!(ldaxr(false, 9, 20), 0x885ffe89);
        assert_eq!(stlxr(false, 10, 19, 20), 0x880afe93);
        assert_eq!(ldaxr(true, 0, 1), 0xc85ffc20);
        assert_eq!(
            neon_three_same(true, false, 2, 0b10000, 17, 1, 1),
            0x4eb1_8421
        );
        assert_eq!(
            neon_three_same(true, false, 2, 0b10011, 2, 1, 0),
            0x4ea2_9c20
        );
        assert_eq!(
            neon_three_same(true, false, 0, 0b01100, 2, 1, 0),
            0x4e22_6420
        );
        assert_eq!(neon_two_misc(true, false, 0, 0b01011, 1, 0), 0x4e20_b820);
        assert_eq!(neon_two_misc(true, true, 0, 0b01011, 1, 0), 0x6e20_b820);
    }

    #[test]
    fn labels_resolve() {
        let mut a = Asm::new(0x1000);
        a.label("top")
            .emit(nop())
            .b_to("top")
            .bl_to("end")
            .label("end")
            .emit(ret());
        let words = a.finish();
        assert_eq!(words[1], b(-4));
        assert_eq!(words[2], bl(4));
    }
}
