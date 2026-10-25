//! Synthetic AArch32 programs run by `kirikae-rt`'s interpreter.
//!
//! Each program is a self-checking guest: it computes, compares each result
//! with an immediate built by `movw`/`movt`, and ends with `svc #7` (exit 0) or
//! `svc #0x26` (break, exit 1, with the failing check id in `r0`). The words are
//! assembled here from the ARM ARM encodings so the test does not depend on the
//! decoder's own tests for the input side.
//!
//! The images place code at `BASE` and a writable data page at `BASE + 0x1000`,
//! which is the same shape the loader produces for an A32 program.

use std::collections::HashMap;

use kirikae_loader::{Arch, Executable, Image, Module};
use kirikae_rt::{RunOptions, run_executable};

const BASE: u64 = 0x0800_0000;
const DATA: u64 = BASE + 0x1000;
const SVC_EXIT: u32 = 7;
const SVC_BREAK: u32 = 0x26;

const EQ: u32 = 0;
const NE: u32 = 1;
const CS: u32 = 2;
const CC: u32 = 3;
const MI: u32 = 4;
const PL: u32 = 5;
const VS: u32 = 6;
const VC: u32 = 7;
const HI: u32 = 8;
const LS: u32 = 9;
const GE: u32 = 10;
const LT: u32 = 11;
const GT: u32 = 12;
const LE: u32 = 13;

fn movw(rd: u32, imm: u16) -> u32 {
    0xE300_0000 | (rd << 12) | (u32::from(imm & 0xF000) << 4) | u32::from(imm & 0x0FFF)
}

fn movt(rd: u32, imm: u16) -> u32 {
    0xE340_0000 | (rd << 12) | (u32::from(imm & 0xF000) << 4) | u32::from(imm & 0x0FFF)
}

fn mov_reg(rd: u32, rm: u32) -> u32 {
    0xE1A0_0000 | (rd << 12) | rm
}

fn mvn_reg(rd: u32, rm: u32) -> u32 {
    0xE1E0_0000 | (rd << 12) | rm
}

fn and_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xE000_0000 | (rn << 16) | (rd << 12) | rm
}

fn eor_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xE020_0000 | (rn << 16) | (rd << 12) | rm
}

fn orr_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xE180_0000 | (rn << 16) | (rd << 12) | rm
}

fn bic_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xE1C0_0000 | (rn << 16) | (rd << 12) | rm
}

fn add_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0xE280_0000 | (rn << 16) | (rd << 12) | imm
}

fn adds_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0xE290_0000 | (rn << 16) | (rd << 12) | imm
}

fn adcs_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0xE2B0_0000 | (rn << 16) | (rd << 12) | imm
}

fn sub_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0xE240_0000 | (rn << 16) | (rd << 12) | imm
}

fn subs_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0xE250_0000 | (rn << 16) | (rd << 12) | imm
}

fn sbcs_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0xE2D0_0000 | (rn << 16) | (rd << 12) | imm
}

fn rsbs_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0xE270_0000 | (rn << 16) | (rd << 12) | imm
}

fn movs_imm(rd: u32, imm: u32) -> u32 {
    0xE3B0_0000 | (rd << 12) | imm
}

fn mvn_imm(rd: u32, imm: u32) -> u32 {
    0xE3E0_0000 | (rd << 12) | imm
}

fn cmp_imm(rn: u32, imm: u32) -> u32 {
    0xE350_0000 | (rn << 16) | imm
}

fn cmp_reg(rn: u32, rm: u32) -> u32 {
    0xE150_0000 | (rn << 16) | rm
}

fn muls(rd: u32, rm: u32, rs: u32) -> u32 {
    0xE010_0090 | (rd << 16) | (rs << 8) | rm
}

fn mla(rd: u32, rm: u32, rs: u32, ra: u32) -> u32 {
    0xE020_0090 | (rd << 16) | (ra << 12) | (rs << 8) | rm
}

fn ldr_imm(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE590_0000 | (rn << 16) | (rt << 12) | imm
}

fn str_imm(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE580_0000 | (rn << 16) | (rt << 12) | imm
}

fn ldrb_imm(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE5D0_0000 | (rn << 16) | (rt << 12) | imm
}

fn strb_imm(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE5C0_0000 | (rn << 16) | (rt << 12) | imm
}

fn str_imm_wb(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE5A0_0000 | (rn << 16) | (rt << 12) | imm
}

fn ldr_imm_post(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE490_0000 | (rn << 16) | (rt << 12) | imm
}

fn ldr_imm_neg(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE510_0000 | (rn << 16) | (rt << 12) | imm
}

fn ldr_reg_lsl(rt: u32, rn: u32, rm: u32, shift: u32) -> u32 {
    0xE790_0000 | (rn << 16) | (rt << 12) | (shift << 7) | rm
}

fn stmia_wb(rn: u32, list: u16) -> u32 {
    0xE8A0_0000 | (rn << 16) | u32::from(list)
}

fn ldmia_wb(rn: u32, list: u16) -> u32 {
    0xE8B0_0000 | (rn << 16) | u32::from(list)
}

fn stmdb(rn: u32, list: u16) -> u32 {
    0xE900_0000 | (rn << 16) | u32::from(list)
}

fn stmdb_wb(rn: u32, list: u16) -> u32 {
    0xE920_0000 | (rn << 16) | u32::from(list)
}

fn ldmia(rn: u32, list: u16) -> u32 {
    0xE8B0_0000 | (rn << 16) | u32::from(list)
}

fn bx(rm: u32) -> u32 {
    0xE12F_FF10 | rm
}

fn svc(imm: u32) -> u32 {
    0xEF00_0000 | imm
}

fn cond_branch(cond: u32, offset: u32) -> u32 {
    (cond << 28) | 0x0A00_0000 | (offset & 0x00FF_FFFF)
}

// --- Encodings of the media, extra load/store, multiply and system classes ---------
// The words below are assembled from the ARM ARM encodings, like the ones above.

/// `uxtb`/`sxth`/`uxtb16`/... with an optional `ror #rotate` (0, 8, 16 or 24).
fn extend(group: u32, rd: u32, rn: u32, rm: u32, rotate: u32) -> u32 {
    0xE600_0000
        | (group << 20)
        | (rn << 16)
        | (rd << 12)
        | ((rotate / 8) << 10)
        | (0b0111 << 4)
        | rm
}

fn rev(rd: u32, rm: u32) -> u32 {
    0xE6BF_0F30 | (rd << 12) | rm
}

fn rev16(rd: u32, rm: u32) -> u32 {
    0xE6BF_0FB0 | (rd << 12) | rm
}

fn revsh(rd: u32, rm: u32) -> u32 {
    0xE6FF_0FB0 | (rd << 12) | rm
}

fn rbit(rd: u32, rm: u32) -> u32 {
    0xE6FF_0F30 | (rd << 12) | rm
}

fn clz(rd: u32, rm: u32) -> u32 {
    0xE16F_0F10 | (rd << 12) | rm
}

fn pkhbt(rd: u32, rn: u32, rm: u32, imm5: u32) -> u32 {
    0xE680_0000 | (rn << 16) | (rd << 12) | (imm5 << 7) | 0x10 | rm
}

fn pkhtb(rd: u32, rn: u32, rm: u32, imm5: u32) -> u32 {
    0xE680_0000 | (rn << 16) | (rd << 12) | (imm5 << 7) | 0x50 | rm
}

fn sel(rd: u32, rn: u32, rm: u32) -> u32 {
    0xE680_0000 | (rn << 16) | (rd << 12) | (0xF << 8) | 0xB0 | rm
}

/// One of the six parallel groups with one of the six selectors.
fn parallel(group: u32, selector: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    0xE600_0000 | (group << 20) | (rn << 16) | (rd << 12) | (0xF << 8) | (selector << 4) | rm
}

fn qadd(kind: u32, rd: u32, rm: u32, rn: u32) -> u32 {
    // 0 = qadd, 1 = qsub, 2 = qdadd, 3 = qdsub.
    let double = kind >> 1;
    let subtract = kind & 1;
    (0xE100_0000 | (double << 22) | (subtract << 21) | (rn << 16) | (rd << 12) | 0x50) | rm
}

fn ssat(rd: u32, width: u32, rm: u32, imm5: u32, asr: bool) -> u32 {
    let sat = width - 1;
    0xE600_0000
        | ((0b01010 | (sat >> 4)) << 20)
        | ((sat & 0xF) << 16)
        | (rd << 12)
        | (imm5 << 7)
        | (u32::from(asr) << 6)
        | 0x10
        | rm
}

fn usat(rd: u32, width: u32, rm: u32) -> u32 {
    0xE600_0000 | ((0b01110 | (width >> 4)) << 20) | ((width & 0xF) << 16) | (rd << 12) | 0x10 | rm
}

fn ssat16(rd: u32, width: u32, rm: u32) -> u32 {
    let sat = width - 1;
    0xE600_0000
        | ((0b01010 | (sat >> 4)) << 20)
        | ((sat & 0xF) << 16)
        | (rd << 12)
        | (0xF << 8)
        | 0x30
        | rm
}

fn usat16(rd: u32, width: u32, rm: u32) -> u32 {
    0xE600_0000
        | ((0b01110 | (width >> 4)) << 20)
        | ((width & 0xF) << 16)
        | (rd << 12)
        | (0xF << 8)
        | 0x30
        | rm
}

fn bfc(rd: u32, lsb: u32, width: u32) -> u32 {
    0xE7C0_0000 | ((lsb + width - 1) << 16) | (rd << 12) | (lsb << 7) | 0x10 | 0xF
}

fn bfi(rd: u32, rn: u32, lsb: u32, width: u32) -> u32 {
    0xE7C0_0000 | ((lsb + width - 1) << 16) | (rd << 12) | (lsb << 7) | 0x10 | rn
}

fn sbfx(rd: u32, rn: u32, lsb: u32, width: u32) -> u32 {
    0xE7A0_0000 | ((width - 1) << 16) | (rd << 12) | (lsb << 7) | 0x50 | rn
}

fn ubfx(rd: u32, rn: u32, lsb: u32, width: u32) -> u32 {
    0xE7E0_0000 | ((width - 1) << 16) | (rd << 12) | (lsb << 7) | 0x50 | rn
}

fn ldrh(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE1D0_00B0 | (rn << 16) | (rt << 12) | ((imm >> 4) << 8) | (imm & 0xF)
}

fn strh(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE1C0_00B0 | (rn << 16) | (rt << 12) | ((imm >> 4) << 8) | (imm & 0xF)
}

fn ldrsb(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE1D0_00D0 | (rn << 16) | (rt << 12) | ((imm >> 4) << 8) | (imm & 0xF)
}

fn ldrsh(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE1D0_00F0 | (rn << 16) | (rt << 12) | ((imm >> 4) << 8) | (imm & 0xF)
}

fn ldrd(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE1C0_00D0 | (rn << 16) | (rt << 12) | ((imm >> 4) << 8) | (imm & 0xF)
}

fn strd(rt: u32, rn: u32, imm: u32) -> u32 {
    0xE1C0_00F0 | (rn << 16) | (rt << 12) | ((imm >> 4) << 8) | (imm & 0xF)
}

fn ldrex(rt: u32, rn: u32) -> u32 {
    0xE190_0F90 | (rn << 16) | (rt << 12) | 0xF
}

fn strex(rd: u32, rt: u32, rn: u32) -> u32 {
    0xE180_0F90 | (rn << 16) | (rd << 12) | rt
}

fn long_multiply(rdlo: u32, rdhi: u32, rm: u32, rs: u32, signed: bool, accumulate: bool) -> u32 {
    0xE080_0000
        | (u32::from(signed) << 22)
        | (u32::from(accumulate) << 21)
        | (rdhi << 16)
        | (rdlo << 12)
        | (rs << 8)
        | 0x90
        | rm
}

fn smulxy(rd: u32, rm: u32, rs: u32, x: bool, y: bool) -> u32 {
    0xE160_0000 | (rd << 16) | (u32::from(y) << 6) | (u32::from(x) << 5) | (rs << 8) | 0x80 | rm
}

fn smlaxy(rd: u32, rm: u32, rs: u32, ra: u32, x: bool, y: bool) -> u32 {
    0xE100_0000
        | (rd << 16)
        | (ra << 12)
        | (rs << 8)
        | (u32::from(y) << 6)
        | (u32::from(x) << 5)
        | 0x80
        | rm
}

fn smlalxy(rdlo: u32, rdhi: u32, rm: u32, rs: u32, x: bool, y: bool) -> u32 {
    0xE140_0000
        | (rdhi << 16)
        | (rdlo << 12)
        | (rs << 8)
        | (u32::from(y) << 6)
        | (u32::from(x) << 5)
        | 0x80
        | rm
}

fn smlawb(rd: u32, rm: u32, rs: u32, ra: u32) -> u32 {
    0xE120_0000 | (rd << 16) | (ra << 12) | (rs << 8) | 0x80 | rm
}

fn mrs(rd: u32) -> u32 {
    0xE10F_0000 | (rd << 12)
}

fn msr(rm: u32) -> u32 {
    0xE128_F000 | rm
}

const NOP: u32 = 0xE320_F000;
const YIELD: u32 = 0xE320_F001;
const WFE: u32 = 0xE320_F002;
const WFI: u32 = 0xE320_F003;
const SEV: u32 = 0xE320_F004;

struct Program {
    code: Vec<u32>,
    labels: HashMap<&'static str, usize>,
    fixups: Vec<(usize, &'static str)>,
}

impl Program {
    fn new() -> Program {
        Program {
            code: Vec::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    fn push(&mut self, word: u32) -> usize {
        self.code.push(word);
        self.code.len() - 1
    }

    fn label(&mut self, name: &'static str) {
        self.labels.insert(name, self.code.len());
    }

    fn branch(&mut self, base: u32, label: &'static str) {
        self.fixups.push((self.code.len(), label));
        self.push(base);
    }

    fn mov_imm(&mut self, rd: u32, value: u32) {
        self.push(movw(rd, value as u16));
        if value >> 16 != 0 {
            self.push(movt(rd, (value >> 16) as u16));
        }
    }

    /// Fails check `id` unless `reg == value`. `r12` is the scratch register, and the
    /// value that did not match travels to the report in `r1` (or `r2` when the checked
    /// register is `r1`).
    fn expect_eq(&mut self, reg: u32, value: u32, id: u16) {
        assert_ne!(reg, 12, "expect_eq needs r12 as scratch");
        self.mov_imm(12, value);
        self.push(cmp_reg(reg, 12));
        self.push(cond_branch(EQ, 2));
        self.push(mov_reg(if reg == 1 { 2 } else { 1 }, reg));
        self.push(movw(0, id));
        self.push(svc(SVC_BREAK));
    }

    /// Fails check `id` unless the condition holds on the current flags.
    fn expect_cond(&mut self, cond: u32, id: u16) {
        self.push(cond_branch(cond, 1));
        self.push(movw(0, id));
        self.push(svc(SVC_BREAK));
    }

    fn exit_ok(&mut self) {
        self.push(movw(0, 0));
        self.push(svc(SVC_EXIT));
    }

    fn finish(mut self, data: &[u8]) -> Executable {
        assert!(self.code.len() * 4 <= 0x1000, "the code needs its own page");
        for (index, label) in &self.fixups {
            let target = *self
                .labels
                .get(label)
                .unwrap_or_else(|| panic!("missing label {label}"));
            let offset = target as i64 - *index as i64 - 2;
            assert!(
                (-8_388_608..8_388_608).contains(&offset),
                "branch to {label} is out of range"
            );
            self.code[*index] |= offset as u32 & 0x00FF_FFFF;
        }
        let mut memory = vec![0u8; 0x2000];
        for (index, word) in self.code.iter().enumerate() {
            memory[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        memory[0x1000..0x1000 + data.len()].copy_from_slice(data);
        let text_end = BASE + (self.code.len() * 4) as u64;
        Executable {
            modules: vec![Module {
                name: "main".into(),
                module_id: [0; 32],
                image: Image {
                    base: BASE,
                    memory,
                    text: BASE..text_end,
                    ro: DATA..DATA,
                    // A real module always has a writable segment; the runtime
                    // maps the page and then writes the image in one pass, so an
                    // empty data range would leave the code unmapped.
                    data: DATA..DATA + 0x1000,
                    bss: DATA + 0x1000..DATA + 0x1000,
                    entry: BASE,
                },
            }],
            entry: BASE,
            arch: Arch::A32,
            title_id: None,
            main_stack_size: None,
            system_resource_size: 0,
            romfs: None,
        }
    }
}

fn run(exe: &Executable) -> i32 {
    run_executable(
        exe,
        &[],
        RunOptions {
            capture_output: true,
        },
    )
    .expect("the synthetic A32 program starts")
    .exit_code
}

#[test]
fn arithmetic_and_flags_are_exact() {
    let mut p = Program::new();
    p.push(mvn_imm(1, 0)); // r1 = 0xffffffff
    p.push(adds_imm(2, 1, 1)); // 0 with carry out
    p.expect_cond(EQ, 1);
    p.expect_cond(CS, 2);
    p.expect_cond(PL, 3);
    p.expect_cond(VC, 4);

    p.mov_imm(3, 0x8000_0000);
    p.push(subs_imm(4, 3, 1)); // 0x7fffffff, signed overflow
    p.expect_cond(VS, 5);
    p.expect_cond(CS, 6);
    p.expect_cond(PL, 7);
    p.expect_cond(NE, 8);

    p.mov_imm(5, 1);
    p.push(subs_imm(6, 5, 2)); // 0xffffffff, borrow
    p.expect_cond(CC, 9);
    p.expect_cond(MI, 10);
    p.expect_cond(VC, 11);

    // A zero immediate rotation (`movs r7, #0`) leaves C alone.
    p.push(movs_imm(7, 0));
    p.expect_cond(CC, 12);
    p.expect_cond(EQ, 13);
    p.push(adds_imm(8, 1, 1)); // carry out again
    p.push(movs_imm(9, 0)); // C stays set
    p.expect_cond(CS, 14);

    p.mov_imm(10, 5);
    p.push(adcs_imm(11, 10, 0)); // 5 + 0 + C = 6
    p.expect_cond(CC, 16);
    p.expect_eq(11, 6, 15);

    p.mov_imm(0, 3);
    p.push(rsbs_imm(1, 0, 10)); // 10 - 3
    p.expect_eq(1, 7, 17);

    p.mov_imm(2, 0xF0);
    p.mov_imm(3, 0x3C);
    p.push(and_reg(4, 2, 3));
    p.push(orr_reg(5, 2, 3));
    p.push(eor_reg(6, 2, 3));
    p.push(bic_reg(7, 2, 3));
    p.push(mvn_reg(8, 3));
    p.expect_eq(4, 0x30, 18);
    p.expect_eq(5, 0xFC, 19);
    p.expect_eq(6, 0xCC, 20);
    p.expect_eq(7, 0xC0, 21);
    p.expect_eq(8, 0xFFFF_FFC3, 22);

    p.mov_imm(10, 3);
    p.mov_imm(11, 5);
    p.push(muls(2, 10, 11));
    p.expect_cond(NE, 23);
    p.expect_eq(2, 15, 24);
    p.mov_imm(1, 7);
    p.push(mla(3, 10, 11, 1));
    p.expect_eq(3, 22, 25);

    p.mov_imm(4, 0x100);
    p.push(subs_imm(5, 4, 1)); // 0xff, borrow clear
    p.mov_imm(6, 0);
    p.push(sbcs_imm(7, 6, 0)); // 0 - 0 - 0
    p.expect_eq(7, 0, 26);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn loads_and_stores_cover_every_addressing_mode() {
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.mov_imm(1, 0xAABB_CCDD);
    p.push(str_imm(1, 0, 0));
    p.push(ldr_imm(2, 0, 0));
    p.expect_eq(2, 0xAABB_CCDD, 1);

    p.mov_imm(3, 0x5A);
    p.push(strb_imm(3, 0, 4));
    p.push(ldrb_imm(4, 0, 4));
    p.expect_eq(4, 0x5A, 2);

    // Pre-indexed writeback: the base becomes the effective address.
    p.push(mov_reg(5, 0));
    p.push(str_imm_wb(1, 5, 8));
    p.expect_eq(5, DATA as u32 + 8, 3);
    p.push(ldr_imm(6, 5, 0));
    p.expect_eq(6, 0xAABB_CCDD, 4);

    // Post-indexed: loads the old base and then adds the offset.
    p.push(ldr_imm_post(7, 5, 4));
    p.expect_eq(7, 0xAABB_CCDD, 5);
    p.expect_eq(5, DATA as u32 + 12, 6);

    // Negative immediate offset.
    p.push(ldr_imm_neg(8, 5, 4));
    p.expect_eq(8, 0xAABB_CCDD, 7);

    // Register offset with an immediate shift.
    p.mov_imm(9, DATA as u32);
    p.mov_imm(10, 2);
    p.push(ldr_reg_lsl(11, 9, 10, 2));
    p.expect_eq(11, 0xAABB_CCDD, 8);

    // Literal load: the base is Align(PC, 4) + 8.
    p.push(0xE59F_0004); // ldr r0, [pc, #4]
    p.push(cond_branch(14, 1)); // b over the filler and the literal
    p.push(0); // filler
    p.push(0x1234_5678); // literal at pc + 12
    p.expect_eq(0, 0x1234_5678, 9);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn byte_and_halfword_transfers_touch_only_their_width() {
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.mov_imm(1, 0xAABB_CCDD);
    p.push(str_imm(1, 0, 0));
    p.mov_imm(2, 0x5A);
    p.push(strb_imm(2, 0, 1));
    p.push(ldr_imm(3, 0, 0));
    p.expect_eq(3, 0xAABB_5ADD, 1);
    p.push(ldrb_imm(4, 0, 3));
    p.expect_eq(4, 0xAA, 2);

    p.mov_imm(5, 0x1122_3344);
    p.push(str_imm(5, 0, 4));
    p.mov_imm(6, 0x1234);
    p.push(strh(6, 0, 6));
    p.push(ldr_imm(7, 0, 4));
    p.expect_eq(7, 0x1234_3344, 3);
    p.push(ldrh(8, 0, 4));
    p.expect_eq(8, 0x3344, 4);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn block_transfers_order_and_write_back() {
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.mov_imm(1, 0x11);
    p.mov_imm(2, 0x22);
    p.mov_imm(3, 0x33);
    p.push(stmia_wb(0, 0x000E)); // stmia r0!, {r1, r2, r3}
    p.expect_eq(0, DATA as u32 + 12, 1);

    p.mov_imm(1, 0);
    p.mov_imm(2, 0);
    p.mov_imm(3, 0);
    p.push(sub_imm(0, 0, 12));
    p.push(ldmia_wb(0, 0x000E)); // ldmia r0!, {r1, r2, r3}
    p.expect_eq(1, 0x11, 2);
    p.expect_eq(2, 0x22, 3);
    p.expect_eq(3, 0x33, 4);
    p.expect_eq(0, DATA as u32 + 12, 5);

    // Decrement-before without writeback: the list still fills ascending
    // addresses, starting eight bytes below the base.
    p.mov_imm(4, DATA as u32 + 12);
    p.mov_imm(1, 0x44);
    p.mov_imm(2, 0x55);
    p.mov_imm(3, 0x66);
    p.push(stmdb(4, 0x000E));
    p.expect_eq(4, DATA as u32 + 12, 6);
    p.mov_imm(5, DATA as u32);
    p.push(ldr_imm(6, 5, 0));
    p.push(ldr_imm(7, 5, 4));
    p.push(ldr_imm(8, 5, 8));
    p.expect_eq(6, 0x44, 7);
    p.expect_eq(7, 0x55, 8);
    p.expect_eq(8, 0x66, 9);

    // `push {r9, lr}` / `pop {r9, lr}` leave sp and lr as they were.
    p.push(mov_reg(0, 13));
    p.mov_imm(9, 0x77);
    p.push(stmdb_wb(13, 0x4200));
    p.mov_imm(9, 0);
    p.push(ldmia(13, 0x4200));
    p.push(mov_reg(1, 13));
    p.expect_eq(9, 0x77, 10);
    p.expect_eq(0, 0x2010_0000, 11);
    p.push(cmp_reg(0, 1));
    p.expect_cond(EQ, 12);
    p.expect_eq(14, 0xFFFF_0000, 13);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn branches_link_and_loop() {
    let mut p = Program::new();
    p.mov_imm(1, 0);
    p.branch(0xEB00_0000, "func"); // bl func
    p.branch(0xEA00_0000, "after");
    p.label("func");
    p.mov_imm(1, 0x42);
    p.push(bx(14));
    p.label("after");
    p.expect_eq(1, 0x42, 1);

    // `bl` targets pc + 8, so the word after the call is not executed.
    p.mov_imm(2, 0);
    p.push(0xEB00_0000); // bl +0
    p.push(movs_imm(2, 9)); // skipped: would leave 9 in r2
    p.push(adds_imm(2, 2, 2)); // target: 0 + 2
    p.expect_eq(2, 2, 2);

    // A backward conditional branch.
    p.mov_imm(3, 1);
    p.label("loop");
    p.push(subs_imm(3, 3, 1));
    p.branch(cond_branch(NE, 0), "loop");
    p.expect_eq(3, 0, 3);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn false_conditions_do_nothing() {
    let mut p = Program::new();
    p.mov_imm(0, 0);
    p.mov_imm(1, 7);
    p.push(cmp_imm(0, 1)); // Z = 0
    p.push(0x0281_1001); // addeq r1, r1, #1
    p.push(0x03A0_2009); // moveq r2, #9
    p.expect_eq(1, 7, 1);
    p.expect_eq(2, 0, 2);

    p.push(cmp_imm(0, 0)); // Z = 1
    p.push(0x1281_1001); // addne r1, r1, #1
    p.push(0x13A0_2009); // movne r2, #9
    p.expect_eq(1, 7, 3);
    p.expect_eq(2, 0, 4);

    p.push(0x03A0_2009); // moveq r2, #9
    p.expect_eq(2, 9, 5);

    // A conditional store that does not run leaves memory alone.
    p.mov_imm(3, DATA as u32);
    p.mov_imm(4, 0x1234);
    p.push(str_imm(4, 3, 0));
    p.push(cmp_imm(0, 1)); // Z = 0, so the next store is skipped
    p.push(0x0584_5000); // streq r5, [r4]
    p.push(ldr_imm(6, 3, 0));
    p.expect_eq(6, 0x1234, 6);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn pc_reads_as_the_address_plus_eight() {
    let mut p = Program::new();
    let at = p.code.len() as u32;
    p.push(mov_reg(0, 15));
    p.expect_eq(0, BASE as u32 + at * 4 + 8, 1);

    let at = p.code.len() as u32;
    p.push(add_imm(1, 15, 4));
    p.expect_eq(1, BASE as u32 + at * 4 + 12, 2);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn rejected_instruction_stops_with_a_clean_report() {
    // The deprecated `swp` is rejected, and a rejection stops the interpreter with
    // the report the discovery loop reads.
    let mut p = Program::new();
    p.push(0xE102_0092);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 2, "a fatal interpreter stop exits 2");
}

#[test]
fn thumb_branch_stops_with_a_clean_report() {
    let mut p = Program::new();
    p.push(movs_imm(0, 3)); // bit 0 selects Thumb
    p.push(bx(0));
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 2);
}

#[test]
fn media_extend_reverse_pack_and_select() {
    let mut p = Program::new();
    p.mov_imm(0, 0x1234_5678);
    p.push(extend(0b01110, 1, 0xF, 0, 0)); // uxtb
    p.push(extend(0b01111, 2, 0xF, 0, 0)); // uxth
    p.push(extend(0b01010, 3, 0xF, 0, 0)); // sxtb
    p.push(extend(0b01011, 4, 0xF, 0, 0)); // sxth
    p.expect_eq(1, 0x78, 1);
    p.expect_eq(2, 0x5678, 2);
    p.expect_eq(3, 0x78, 3);
    p.expect_eq(4, 0x5678, 4);

    // A rotated operand and the signed 16-bit extraction.
    p.mov_imm(5, 0x0000_8081);
    p.push(extend(0b01110, 6, 0xF, 5, 8)); // uxtb r6, r5, ror #8 -> 0x80
    p.push(extend(0b01000, 7, 0xF, 5, 0)); // sxtb16 r7, r5
    p.push(extend(0b01100, 8, 0xF, 5, 0)); // uxtb16 r8, r5
    p.expect_eq(6, 0x80, 5);
    p.expect_eq(7, 0x0000_ff81, 6);
    p.expect_eq(8, 0x81, 7);

    // The accumulating form adds the extension to the first source.
    p.mov_imm(0, 0x1122_3344);
    p.mov_imm(1, 0x80);
    p.push(extend(0b01010, 2, 0, 1, 0)); // sxtab r2, r0, r1
    p.expect_eq(2, 0x1122_32c4, 8);

    p.mov_imm(9, 0x1234_5678);
    p.push(rev(10, 9));
    p.push(rev16(11, 9));
    p.expect_eq(10, 0x7856_3412, 9);
    p.expect_eq(11, 0x3412_7856, 10);
    p.push(revsh(1, 9));
    p.push(rbit(2, 9));
    p.expect_eq(1, 0x0000_7856, 11);
    p.expect_eq(2, 0x1e6a_2c48, 12);
    p.mov_imm(3, 1);
    p.push(clz(4, 3));
    p.expect_eq(4, 31, 13);
    p.mov_imm(3, 0x8000_0000);
    p.push(clz(5, 3));
    p.expect_eq(5, 0, 14);

    p.mov_imm(6, 0x1234_5678);
    p.mov_imm(7, 0xaabb_ccdd);
    p.push(pkhbt(8, 6, 7, 8));
    p.push(pkhtb(9, 6, 7, 8));
    p.expect_eq(8, 0xbbcc_5678, 15);
    p.expect_eq(9, 0x1234_bbcc, 16);

    // `asr #0` in `pkhtb` is a shift by 32.
    p.push(pkhtb(10, 6, 7, 0));
    p.expect_eq(10, 0x1234_ffff, 17);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn parallel_arithmetic_and_the_ge_flags() {
    let mut p = Program::new();
    p.mov_imm(0, 0x0001_7fff);
    p.mov_imm(1, 0x0002_0001);
    // The six signed forms: add16, asx, sax, sub16, add8, sub8.
    for (index, selector) in [1u32, 3, 5, 7, 9, 0xF].iter().enumerate() {
        p.push(parallel(0b00001, *selector, 2 + index as u32, 0, 1));
    }
    p.expect_eq(2, 0x0003_8000, 1);
    p.expect_eq(3, 0x0002_7ffd, 2);
    p.expect_eq(4, 0x0000_8001, 3);
    p.expect_eq(5, 0xffff_7ffe, 4);
    p.expect_eq(6, 0x0003_7f00, 5);
    p.expect_eq(7, 0x00ff_7ffe, 6);

    // Unsigned, saturating and halving 16-bit additions.
    p.push(parallel(0b00101, 1, 8, 0, 1));
    p.push(parallel(0b00110, 1, 9, 0, 1));
    p.push(parallel(0b00111, 1, 10, 0, 1));
    p.push(parallel(0b00011, 1, 11, 0, 1));
    p.expect_eq(8, 0x0003_8000, 7);
    p.expect_eq(9, 0x0003_8000, 8);
    p.expect_eq(10, 0x0001_4000, 9);
    p.expect_eq(11, 0x0001_4000, 10);

    // Only the byte-wide forms write GE; `sel` reads it per byte.
    p.mov_imm(0, 0x0000_00ff);
    p.mov_imm(1, 0x0000_0001);
    p.push(parallel(0b00101, 9, 2, 0, 1)); // uadd8: only byte 0 carries out
    p.expect_eq(2, 0, 11);
    p.mov_imm(3, 0x1122_3344);
    p.mov_imm(4, 0xaabb_ccdd);
    p.push(sel(5, 3, 4));
    p.expect_eq(5, 0xaabb_cc44, 12);

    // usub8 sets GE[i] when the unsigned difference is not negative.
    p.mov_imm(6, 0x0001_7fff);
    p.mov_imm(7, 0x0002_0001);
    p.push(parallel(0b00101, 0xF, 8, 6, 7));
    p.expect_eq(8, 0x00ff_7ffe, 13);
    p.push(sel(9, 3, 4));
    p.expect_eq(9, 0x11bb_3344, 14);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn saturate_bitfield_and_scalar_saturating_arithmetic() {
    let mut p = Program::new();
    p.mov_imm(0, 0x7fff_ffff);
    p.push(ssat(1, 8, 0, 0, false));
    p.push(usat(2, 8, 0));
    p.push(ssat(3, 8, 0, 8, true));
    p.expect_eq(1, 0x7f, 1);
    p.expect_eq(2, 0xff, 2);
    p.expect_eq(3, 0x7f, 3);

    // The Q flag is set by a saturating `ssat`; extract it with `ubfx`.
    p.push(mrs(4));
    p.push(ubfx(5, 4, 27, 1));
    p.expect_eq(5, 1, 4);

    p.mov_imm(6, 0x1234_5678);
    p.push(ssat16(7, 8, 6));
    p.push(usat16(8, 8, 6));
    p.expect_eq(7, 0x007f_007f, 5);
    p.expect_eq(8, 0x00ff_00ff, 6);

    p.mov_imm(9, 0xffff_ffff);
    p.push(bfc(9, 4, 8));
    p.expect_eq(9, 0xffff_f00f, 7);
    p.push(bfi(9, 6, 4, 8));
    p.expect_eq(9, 0xffff_f78f, 8);
    p.push(sbfx(10, 6, 4, 8));
    p.push(ubfx(11, 6, 8, 4));
    p.expect_eq(10, 0x67, 9);
    p.expect_eq(11, 0x6, 10);

    p.mov_imm(0, 0x7fff_ffff);
    p.mov_imm(1, 1);
    p.push(qadd(0, 2, 0, 1)); // qadd r2, r0, r1
    p.push(qadd(1, 3, 0, 1)); // qsub r3, r0, r1
    p.expect_eq(2, 0x7fff_ffff, 11);
    p.expect_eq(3, 0x7fff_fffe, 12);
    p.mov_imm(0, 0x4000_0000);
    p.push(qadd(2, 4, 0, 1)); // qdadd r4, r0, r1
    p.push(qadd(3, 5, 0, 1)); // qdsub r5, r0, r1
    p.expect_eq(4, 0x4000_0002, 13);
    p.expect_eq(5, 0x3fff_fffe, 14);

    // The parallel saturating forms saturate silently: Q is not affected.
    p.mov_imm(0, 0);
    p.push(msr(0));
    p.mov_imm(6, 0x0000_7fff);
    p.mov_imm(7, 0x0000_0002);
    p.push(parallel(0b00010, 1, 8, 6, 7)); // qadd16 r8, r6, r7
    p.expect_eq(8, 0x0000_7fff, 15);
    p.push(mrs(9));
    p.push(ubfx(10, 9, 27, 1));
    p.expect_eq(10, 0, 16);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn extra_loads_stores_pairs_and_exclusives() {
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.mov_imm(1, 0xaabb_ccdd);
    p.push(str_imm(1, 0, 0));
    p.push(ldrh(2, 0, 2));
    p.push(ldrsb(3, 0, 3));
    p.push(ldrsh(4, 0, 0));
    p.expect_eq(2, 0xaabb, 1);
    p.expect_eq(3, 0xffff_ffaa, 2);
    p.expect_eq(4, 0xffff_ccdd, 3);

    p.mov_imm(5, 0x1234);
    p.push(strh(5, 0, 4));
    p.push(ldr_imm(6, 0, 4));
    p.expect_eq(6, 0x1234, 4);

    p.push(ldrd(7, 0, 0));
    p.expect_eq(7, 0xaabb_ccdd, 5);
    p.expect_eq(8, 0x0000_1234, 6);

    p.mov_imm(9, 0x5566_7788);
    p.mov_imm(10, 0x99aa_bbcc);
    p.push(strd(9, 0, 8));
    p.push(ldrd(1, 0, 8));
    p.expect_eq(1, 0x5566_7788, 7);
    p.expect_eq(2, 0x99aa_bbcc, 8);

    // The exclusive monitor remembers the loaded value, so the first store succeeds
    // and the second one, without a new `ldrex`, reports failure.
    p.push(ldrex(3, 0));
    p.expect_eq(3, 0xaabb_ccdd, 9);
    p.mov_imm(4, 0x1122_3344);
    p.push(strex(5, 4, 0));
    p.expect_eq(5, 0, 10);
    p.push(ldr_imm(6, 0, 0));
    p.expect_eq(6, 0x1122_3344, 11);
    p.push(strex(7, 4, 0));
    p.expect_eq(7, 1, 12);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn long_halfword_multiplies_and_system_registers() {
    let mut p = Program::new();
    p.mov_imm(0, 0xffff_ffff);
    p.mov_imm(1, 0xffff_ffff);
    p.push(long_multiply(2, 3, 0, 1, false, false)); // umull r2, r3
    p.push(long_multiply(4, 5, 0, 1, true, false)); // smull r4, r5
    p.expect_eq(2, 1, 1);
    p.expect_eq(3, 0xffff_fffe, 2);
    p.expect_eq(4, 1, 3);
    p.expect_eq(5, 0, 4);

    p.mov_imm(6, 2);
    p.push(long_multiply(2, 3, 0, 1, false, true)); // umlal r2, r3
    p.push(long_multiply(4, 5, 0, 1, true, true)); // smlal r4, r5
    p.expect_eq(2, 2, 5);
    p.expect_eq(3, 0xffff_fffc, 6);
    p.expect_eq(4, 2, 7);
    p.expect_eq(5, 0, 8);

    // Halfword multiplies: Rm = 0x0002ffff, Rs = 0x00030004.
    p.mov_imm(7, 0x0002_ffff);
    p.mov_imm(8, 0x0003_0004);
    p.push(smulxy(9, 7, 8, false, false)); // bottom * bottom: -1 * 4
    p.push(smulxy(10, 7, 8, false, true)); // bottom * top: -1 * 3
    p.push(smulxy(11, 7, 8, true, true)); // top * top: 2 * 3
    p.expect_eq(9, 0xffff_fffc, 9);
    p.expect_eq(10, 0xffff_fffd, 10);
    p.expect_eq(11, 6, 11);
    p.push(smlaxy(9, 7, 8, 6, false, false)); // -4 + 2
    p.expect_eq(9, 0xffff_fffe, 12);

    // `smlawb` adds the accumulator before the 16-bit shift.
    p.mov_imm(0, 0xffff_ffff);
    p.mov_imm(1, 1);
    p.mov_imm(2, 0);
    p.push(smlawb(3, 0, 1, 2));
    p.expect_eq(3, 0xffff_ffff, 13);

    // `smlal<x><y>` accumulates the signed product into the pair.
    p.mov_imm(0, 0);
    p.mov_imm(1, 0);
    p.push(smlalxy(0, 1, 7, 8, false, false));
    p.expect_eq(0, 0xffff_fffc, 14);
    p.expect_eq(1, 0xffff_ffff, 15);

    // `smla<x><y>` sets Q when the 32-bit addition overflows.
    p.mov_imm(0, 0x7fff);
    p.mov_imm(1, 0x7fff);
    p.mov_imm(2, 0x7fff_ffff);
    p.push(smlaxy(3, 0, 1, 2, false, false));
    p.expect_eq(3, 0xbfff_0000, 22);
    p.push(mrs(4));
    p.push(ubfx(5, 4, 27, 1));
    p.expect_eq(5, 1, 23);

    // `smlaw<y>` shifts a 64-bit sum and sets Q when the shifted value does not fit.
    p.mov_imm(0, 0x7fff_ffff);
    p.mov_imm(1, 0x7fff);
    p.mov_imm(2, 0x7fff_ffff);
    p.push(smlawb(3, 0, 1, 2));
    p.expect_eq(3, 0xbfff_7ffe, 24);
    p.push(mrs(4));
    p.push(ubfx(5, 4, 27, 1));
    p.expect_eq(5, 1, 25);

    // `mrs` reports the modelled CPSR: user mode with the T bit clear.
    p.push(mrs(0));
    p.push(ubfx(1, 0, 0, 5));
    p.expect_eq(1, 0x10, 16);

    // `msr` of the flags byte writes NZCVQ ...
    p.mov_imm(4, 0x8000_0000);
    p.push(msr(4));
    p.expect_cond(MI, 17);
    p.expect_cond(NE, 18);
    p.push(mrs(5));
    p.push(ubfx(6, 5, 27, 1));
    p.expect_eq(6, 0, 19);

    // ... and leaves GE alone, so `sel` still sees the byte comparison.
    p.mov_imm(0, 0x0000_00ff);
    p.mov_imm(1, 0x0000_0001);
    p.push(parallel(0b00101, 9, 2, 0, 1));
    p.mov_imm(7, 0);
    p.push(msr(7));
    p.mov_imm(8, 0x1122_3344);
    p.mov_imm(9, 0xaabb_ccdd);
    p.push(sel(10, 8, 9));
    p.expect_eq(10, 0xaabb_cc44, 20);

    // The hints are no-ops the interpreter steps over.
    p.push(NOP);
    p.push(YIELD);
    p.push(WFE);
    p.push(WFI);
    p.push(SEV);
    p.expect_eq(10, 0xaabb_cc44, 21);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn deprecated_and_privileged_encodings_stop_with_a_clean_report() {
    // `swp r0, r1, [r2]` is architecturally deprecated and is rejected on purpose.
    let mut p = Program::new();
    p.push(0xE102_0092);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 2);

    // `msr cpsr_c, r0` writes a control field, which needs the banked modes.
    let mut p = Program::new();
    p.push(0xE121_F000);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 2);
}

// --- VFP and Advanced SIMD ----------------------------------------------------------
//
// The words below are built from the ARM ARM encodings, like the ones above. The
// arithmetic is also checked against the host's IEEE implementation, which is the
// reference for the default FPSCR (round to nearest even, no flush to zero).

/// The encoding fields of one VFP register in a data-processing instruction: the
/// four-bit field and the extension bit, which is the high bit for `d` and the low bit
/// for `s`.
fn scalar_fields(register: u32, double: bool) -> (u32, bool) {
    if double {
        (register & 0xf, register >> 4 & 1 == 1)
    } else {
        (register >> 1, register & 1 == 1)
    }
}

/// A VFP data-processing instruction. `opc` is the three-bit opcode (bits 23, 21, 20);
/// bit 22 carries the high bit of the destination.
fn vfp_bin(opc: u32, bit6: bool, double: bool, vd: u32, vn: u32, vm: u32) -> u32 {
    let (vd_field, vd_hi) = scalar_fields(vd, double);
    let (vn_field, vn_hi) = scalar_fields(vn, double);
    let (vm_field, vm_hi) = scalar_fields(vm, double);
    let cp = if double { 0b1011 } else { 0b1010 };
    0xEE00_0000
        | (((opc >> 2) & 1) << 23)
        | ((vd_hi as u32) << 22)
        | (((opc >> 1) & 1) << 21)
        | ((opc & 1) << 20)
        | (vn_field << 16)
        | (vd_field << 12)
        | (cp << 8)
        | ((vn_hi as u32) << 7)
        | ((bit6 as u32) << 6)
        | ((vm_hi as u32) << 5)
        | vm_field
}

/// `vabs`, `vneg`, `vsqrt`, `vrintr`, `vrintz`, `vrintx` and `vmov` between registers,
/// selected by `vn` and bit 7.
fn vfp_unary(vn: u32, bit7: bool, double: bool, vd: u32, vm: u32) -> u32 {
    let (vd_field, vd_hi) = scalar_fields(vd, double);
    let (vm_field, vm_hi) = scalar_fields(vm, double);
    let cp = if double { 0b1011 } else { 0b1010 };
    0xEEB0_0000
        | (vn << 16)
        | ((vd_hi as u32) << 22)
        | (vd_field << 12)
        | (cp << 8)
        | ((bit7 as u32) << 7)
        | (1 << 6)
        | ((vm_hi as u32) << 5)
        | vm_field
}

/// `vcmp`/`vcmpe`, against a register or against zero.
fn vfp_compare(double: bool, first: u32, second: u32, zero: bool, with_exception: bool) -> u32 {
    let (vd_field, vd_hi) = scalar_fields(first, double);
    let (vm_field, vm_hi) = scalar_fields(if zero { 0 } else { second }, double);
    let cp = if double { 0b1011 } else { 0b1010 };
    0xEEB0_0000
        | ((if zero { 5 } else { 4 }) << 16)
        | ((vd_hi as u32) << 22)
        | (vd_field << 12)
        | (cp << 8)
        | ((with_exception as u32) << 7)
        | (1 << 6)
        | ((vm_hi as u32) << 5)
        | vm_field
}

/// `vcvt` between the two precisions: `double` selects `f32.f64`, otherwise `f64.f32`.
fn vcvt_precision(double: bool, vd: u32, source: u32) -> u32 {
    if double {
        let (vd_field, vd_hi) = scalar_fields(vd, false);
        let (vm_field, vm_hi) = scalar_fields(source, true);
        0xEEB7_0BC0 | ((vd_hi as u32) << 22) | (vd_field << 12) | ((vm_hi as u32) << 5) | vm_field
    } else {
        let (vd_field, vd_hi) = scalar_fields(vd, true);
        let (vm_field, vm_hi) = scalar_fields(source, false);
        0xEEB7_0AC0 | ((vd_hi as u32) << 22) | (vd_field << 12) | ((vm_hi as u32) << 5) | vm_field
    }
}

/// `vcvt`/`vcvtr` between floating point and an integer single register. `vn` is 8 for
/// the integer to float form, `0xc`/`0xd` for the unsigned/signed read form. For the
/// read direction bit 7 selects the rounding (clear is round to nearest even) and `Vn`
/// the signedness.
fn vcvt_integer(
    vn: u32,
    signed: bool,
    to_nearest: bool,
    double: bool,
    dest: u32,
    source: u32,
) -> u32 {
    let (vn, bit7) = if vn == 8 {
        (8, signed)
    } else {
        (if signed { 0xd } else { 0xc }, !to_nearest)
    };
    let (dd_field, dd_hi) = scalar_fields(dest, false);
    let source_double = if vn == 8 { false } else { double };
    let (ds_field, ds_hi) = scalar_fields(source, source_double);
    let cp = if double { 0b1011 } else { 0b1010 };
    0xEEB0_0000
        | (vn << 16)
        | ((dd_hi as u32) << 22)
        | (dd_field << 12)
        | (cp << 8)
        | ((bit7 as u32) << 7)
        | (1 << 6)
        | ((ds_hi as u32) << 5)
        | ds_field
}

/// `vldr`/`vstr` of a `d` register, non-negative offset.
fn vldr_d(d: u32, rn: u32, offset: u32) -> u32 {
    0xED90_0B00 | ((d >> 4) << 22) | (rn << 16) | ((d & 0xf) << 12) | (offset / 4)
}

fn vstr_d(d: u32, rn: u32, offset: u32) -> u32 {
    vldr_d(d, rn, offset) & !0x0010_0000
}

/// `vldr`/`vstr` of an `s` register, non-negative offset.
fn vldr_s(s: u32, rn: u32, offset: u32) -> u32 {
    0xED90_0A00 | ((s & 1) << 22) | (rn << 16) | ((s >> 1) << 12) | (offset / 4)
}

fn vstr_s(s: u32, rn: u32, offset: u32) -> u32 {
    vldr_s(s, rn, offset) & !0x0010_0000
}

/// `vpush`/`vpop` of `words` 32-bit registers starting at `d(first)`.
fn vpush(words: u32, first: u32) -> u32 {
    0xED2D_0B00 | ((first >> 4) << 22) | ((first & 0xf) << 12) | words
}

fn vpop(words: u32, first: u32) -> u32 {
    0xECBD_0B00 | ((first >> 4) << 22) | ((first & 0xf) << 12) | words
}

/// `vldmia`/`vstmia` of a `d` register list.
fn vldmia(words: u32, first: u32, rn: u32, writeback: bool) -> u32 {
    0xEC90_0B00
        | ((writeback as u32) << 21)
        | ((first >> 4) << 22)
        | (rn << 16)
        | ((first & 0xf) << 12)
        | words
}

fn vstmia(words: u32, first: u32, rn: u32, writeback: bool) -> u32 {
    vldmia(words, first, rn, writeback) & !0x0010_0000
}

/// `vmov.f32/f64 <reg>, #imm8`.
fn vmov_imm(imm8: u32, double: bool, reg: u32) -> u32 {
    let (field, high) = scalar_fields(reg, double);
    0xEEB0_0000
        | (((imm8 >> 4) & 0xf) << 16)
        | ((high as u32) << 22)
        | (field << 12)
        | ((if double { 0b1011 } else { 0b1010 }) << 8)
        | (imm8 & 0xf)
}

/// `vmov Sn, Rt`.
fn vmov_s_core(s: u32, rt: u32) -> u32 {
    0xEE00_0A10 | ((s >> 1) << 16) | ((s & 1) << 7) | (rt << 12)
}

/// `vmov Rt, Sn`.
fn vmov_core_s(rt: u32, s: u32) -> u32 {
    0xEE10_0A10 | ((s >> 1) << 16) | ((s & 1) << 7) | (rt << 12)
}

/// `vmov Rt, Rt2, Dm`.
fn vmov_core_d(rt: u32, rt2: u32, d: u32) -> u32 {
    0xEC50_0B10 | (rt2 << 16) | (rt << 12) | ((d >> 4) << 5) | (d & 0xf)
}

/// `vmov Dm, Rt, Rt2`.
fn vmov_d_core(d: u32, rt: u32, rt2: u32) -> u32 {
    0xEC40_0B10 | (rt2 << 16) | (rt << 12) | ((d >> 4) << 5) | (d & 0xf)
}

fn vmrs_apsr() -> u32 {
    0xEEF1_FA10
}

fn vmrs_fpscr(rt: u32) -> u32 {
    0xEEF1_0A10 | (rt << 12)
}

fn vmsr_fpscr(rt: u32) -> u32 {
    0xEEE1_0A10 | (rt << 12)
}

/// The NEON three-register group.
fn neon3(bit4: bool, u: bool, size: u32, opc1: u32, vd: u32, vn: u32, vm: u32) -> u32 {
    0xF200_0000
        | ((u as u32) << 24)
        | (((vd >> 4) & 1) << 22)
        | ((size & 3) << 20)
        | ((vn & 0xf) << 16)
        | ((vd & 0xf) << 12)
        | ((opc1 & 0xf) << 8)
        | (((vn >> 4) & 1) << 7)
        | ((bit4 as u32) << 4)
        | (((vm >> 4) & 1) << 5)
        | (vm & 0xf)
}

/// `vdup.<size> Dd, Rt`.
fn vdup_core(esize: u32, d: u32, rt: u32) -> u32 {
    let size = match esize {
        8 => 0x0040_0000,
        16 => 0x0000_0020,
        _ => 0,
    };
    0xEE80_0B10 | size | (((d >> 4) & 1) << 7) | ((d & 0xf) << 16) | (rt << 12)
}

/// `vmov.<size> Dd[lane], Rt`.
fn vmov_lane_to(esize: u32, d: u32, lane: u32, rt: u32) -> u32 {
    let (size, lane_bits) = match esize {
        8 => (
            0x0040_0000 | (((lane >> 2) & 1) << 21),
            (((lane >> 1) & 1) << 6) | ((lane & 1) << 5),
        ),
        16 => ((((lane >> 1) & 1) << 21) | 0x20, (lane & 1) << 6),
        _ => ((lane & 1) << 21, 0),
    };
    0xEE00_0B10 | size | lane_bits | (((d >> 4) & 1) << 7) | ((d & 0xf) << 16) | (rt << 12)
}

/// `vmov.s/u8/16/32 Rt, Dd[lane]`.
fn vmov_lane_from(esize: u32, d: u32, lane: u32, rt: u32, unsigned: bool) -> u32 {
    vmov_lane_to(esize, d, lane, rt) | 0x0010_0000 | ((unsigned as u32) << 23)
}

/// A `vld1`…`vst1` structure transfer: `kind` is the type field, `rm` is 15 for no
/// writeback and 13 for the immediate one.
fn neon_struct(load: bool, kind: u32, esize: u32, register: u32, rn: u32, rm: u32) -> u32 {
    let size = match esize {
        8 => 0,
        16 => 1,
        32 => 2,
        _ => 3,
    };
    0xF400_0000
        | ((load as u32) << 21)
        | (((register >> 4) & 1) << 22)
        | (rn << 16)
        | ((register & 0xf) << 12)
        | (kind << 8)
        | (size << 6)
        | rm
}

/// `vext.8`.
fn vext(dst: u32, first: u32, second: u32, index: u32) -> u32 {
    0xF2B0_0000
        | (second & 0xf)
        | (((dst >> 4) & 1) << 22)
        | ((first & 0xf) << 16)
        | ((dst & 0xf) << 12)
        | (index << 8)
        | (((first >> 4) & 1) << 7)
        | (((second >> 4) & 1) << 5)
}

/// A NEON modified immediate; `cmode` selects the expansion.
fn neon_imm(cmode: u32, d: u32, q: bool, imm8: u32) -> u32 {
    0xF280_0010
        | (((imm8 >> 7) & 1) << 24)
        | (((d >> 4) & 1) << 22)
        | (((imm8 >> 4) & 0xf) << 16)
        | ((d & 0xf) << 12)
        | ((cmode & 0xf) << 8)
        | ((q as u32) << 6)
        | (imm8 & 0xf)
}

fn float_data(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn double_data(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

#[test]
fn vfp_single_arithmetic_matches_the_host() {
    let cases: &[(u32, bool, f32, f32)] = &[
        (0b011, false, 1.5, 2.25),
        (0b011, true, 1.5, 2.25),
        (0b010, false, 1.5, -2.25),
        (0b100, false, 1.0, 3.0),
        (0b010, false, 1e-30, 1e-30),
        (0b100, false, 1.0, 0.0),
        (0b011, false, 0.1, 0.2),
        (0b010, false, 3.402_823_5e38, 2.0),
    ];
    for (opc, bit6, a, b) in cases {
        let expected = match (*opc, *bit6) {
            (0b011, false) => a + b,
            (0b011, true) => a - b,
            (0b010, false) => a * b,
            (0b100, false) => a / b,
            other => panic!("{other:?}"),
        };
        let mut p = Program::new();
        p.mov_imm(0, DATA as u32);
        p.push(vldr_s(0, 0, 0));
        p.push(vldr_s(1, 0, 4));
        p.push(vfp_bin(*opc, *bit6, false, 2, 0, 1));
        p.push(vmov_core_s(1, 2));
        p.expect_eq(1, expected.to_bits(), 1);
        p.exit_ok();
        assert_eq!(
            run(&p.finish(&float_data(&[*a, *b]))),
            0,
            "{a} {b} {opc:03b}"
        );
    }
}

#[test]
fn vfp_double_arithmetic_matches_the_host() {
    let cases: &[(u32, bool, f64, f64)] = &[
        (0b011, false, 1.5, 2.25),
        (0b011, true, 1.5, 2.25),
        (0b010, false, 1.5, -2.25),
        (0b100, false, 1.0, 3.0),
        (0b011, false, 0.1, 0.2),
    ];
    for (opc, bit6, a, b) in cases {
        let expected = match (*opc, *bit6) {
            (0b011, false) => a + b,
            (0b011, true) => a - b,
            (0b010, false) => a * b,
            (0b100, false) => a / b,
            other => panic!("{other:?}"),
        };
        let mut p = Program::new();
        p.mov_imm(0, DATA as u32);
        p.push(vldr_d(0, 0, 0));
        p.push(vldr_d(1, 0, 8));
        p.push(vfp_bin(*opc, *bit6, true, 2, 0, 1));
        p.push(vstr_d(2, 0, 16));
        p.push(ldr_imm(1, 0, 16));
        p.expect_eq(1, expected.to_bits() as u32, 1);
        p.push(ldr_imm(1, 0, 20));
        p.expect_eq(1, (expected.to_bits() >> 32) as u32, 2);
        p.exit_ok();
        assert_eq!(
            run(&p.finish(&double_data(&[*a, *b]))),
            0,
            "{a} {b} {opc:03b}"
        );
    }
}

#[test]
fn vfp_fused_multiply_add_and_unary() {
    let (a, b, addend) = (1.5f32, -2.25f32, 0.75f32);
    let fused = a.mul_add(b, addend);
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.push(vldr_s(0, 0, 0));
    p.push(vldr_s(1, 0, 4));
    p.push(vldr_s(3, 0, 8));

    // vfma s3, s0, s1 and vmla s5, s0, s1 are both fused on VFPv4.
    p.push(vfp_bin(0b110, false, false, 3, 0, 1));
    p.push(vmov_core_s(4, 3));
    p.expect_eq(4, fused.to_bits(), 1);
    p.push(vldr_s(5, 0, 8));
    p.push(vfp_bin(0b000, false, false, 5, 0, 1));
    p.push(vmov_core_s(6, 5));
    p.expect_eq(6, fused.to_bits(), 2);

    // vnmul negates the product; vnmls computes `-d + n*m` and vnmla `-d - n*m`.
    p.push(vfp_bin(0b010, true, false, 7, 0, 1));
    p.push(vmov_core_s(8, 7));
    p.expect_eq(8, (-(a * b)).to_bits(), 3);
    p.push(vldr_s(9, 0, 8));
    p.push(vfp_bin(0b001, false, false, 9, 0, 1)); // vnmls s9, s0, s1
    p.push(vmov_core_s(10, 9));
    p.expect_eq(10, (-addend + a * b).to_bits(), 4);
    p.push(vldr_s(11, 0, 8));
    p.push(vfp_bin(0b001, true, false, 11, 0, 1)); // vnmla s11, s0, s1
    p.push(vmov_core_s(2, 11));
    p.expect_eq(2, (-addend - a * b).to_bits(), 5);

    // vabs, vneg and vsqrt of a fresh copy of the addend.
    p.push(vldr_s(20, 0, 8));
    p.push(vfp_unary(0, true, false, 21, 20));
    p.push(vmov_core_s(4, 21));
    p.expect_eq(4, addend.abs().to_bits(), 6);
    p.push(vfp_unary(1, false, false, 22, 20));
    p.push(vmov_core_s(5, 22));
    p.expect_eq(5, (-addend).to_bits(), 7);
    // The immediate form builds 4.0 without a memory access.
    p.push(vmov_imm(0x10, false, 23));
    p.push(vfp_unary(1, true, false, 24, 23));
    p.push(vmov_core_s(6, 24));
    p.expect_eq(6, 4.0f32.sqrt().to_bits(), 8);
    // vrintz rounds towards zero, vrintr with the default mode rounds to even.
    p.push(vldr_s(25, 0, 12));
    p.push(vfp_unary(6, false, false, 26, 25));
    p.push(vmov_core_s(7, 26));
    p.expect_eq(7, (-2.0f32).to_bits(), 9);
    p.push(vfp_unary(6, true, false, 27, 25));
    p.push(vmov_core_s(8, 27));
    p.expect_eq(8, (-2.0f32).to_bits(), 10);

    p.exit_ok();
    assert_eq!(run(&p.finish(&float_data(&[a, b, addend, -2.5]))), 0);
}

#[test]
fn vfp_compare_sets_the_fpscr_flags() {
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.push(vldr_s(0, 0, 0));
    p.push(vldr_s(1, 0, 4));

    // Equal: Z and C are set.
    p.push(vfp_compare(false, 0, 1, false, false));
    p.push(vmrs_apsr());
    p.expect_cond(EQ, 1);
    p.expect_cond(CS, 2);
    p.expect_cond(LS, 3);

    // Less than: N and C are clear.
    p.push(vldr_s(2, 0, 8));
    p.push(vfp_compare(false, 2, 0, false, false));
    p.push(vmrs_apsr());
    p.expect_cond(MI, 4);
    p.expect_cond(CC, 5);
    p.expect_cond(LT, 6);
    p.expect_cond(LE, 7);

    // Greater than.
    p.push(vfp_compare(false, 0, 2, false, false));
    p.push(vmrs_apsr());
    p.expect_cond(GT, 8);
    p.expect_cond(HI, 9);
    p.expect_cond(GE, 10);

    // Against zero.
    p.push(vfp_compare(false, 2, 0, true, false));
    p.push(vmrs_apsr());
    p.expect_cond(MI, 11);
    p.expect_cond(NE, 12);

    // Unordered: C and V are set, and `vcmpe` raises Invalid Operation.
    p.push(vldr_s(3, 0, 12));
    p.push(vfp_compare(false, 3, 0, false, true));
    p.push(vmrs_apsr());
    p.expect_cond(VS, 13);
    p.expect_cond(CS, 14);
    p.push(vmrs_fpscr(4));
    p.push(ubfx(5, 4, 0, 1));
    p.expect_eq(5, 1, 15);

    p.exit_ok();
    assert_eq!(run(&p.finish(&float_data(&[2.5, 2.5, -1.0, f32::NAN]))), 0);
}

#[test]
fn vfp_conversions_are_exact() {
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.push(vldr_s(0, 0, 0)); // 2.5
    p.push(vldr_s(3, 0, 4)); // -7.75
    p.push(vldr_s(20, 0, 8)); // 0.0

    // vcvt with the default mode rounds to nearest even, like vcvtr.
    p.push(vcvt_integer(0xc, false, true, false, 1, 0));
    p.push(vmov_core_s(2, 1));
    p.expect_eq(2, 2, 1);
    // Truncation needs the rounding mode in FPSCR.
    p.mov_imm(1, 0x00C0_0000); // RMode = 3 (towards zero)
    p.push(vmsr_fpscr(1));
    p.push(vcvt_integer(0xc, false, false, false, 4, 0));
    p.push(vmov_core_s(5, 4));
    p.expect_eq(5, 2, 2);
    p.push(vcvt_integer(0xd, true, false, false, 6, 3));
    p.push(vmov_core_s(7, 6));
    p.expect_eq(7, (-7i32) as u32, 3);
    p.mov_imm(1, 0);
    p.push(vmsr_fpscr(1));

    // Integer to float, signed, and back through the precision conversion.
    p.push(vmov_s_core(8, 7));
    p.push(vcvt_integer(8, true, true, false, 9, 8));
    p.push(vmov_core_s(10, 9));
    p.expect_eq(10, (-7.0f32).to_bits(), 4);
    p.push(vcvt_precision(false, 16, 9)); // f64.f32
    p.push(vcvt_precision(true, 11, 16)); // f32.f64
    p.push(vmov_core_s(6, 11));
    p.expect_eq(6, (-7.0f32).to_bits(), 5);

    // A double-precision value built from an integer: 2 -> 2.0f32 -> 2.0f64.
    p.mov_imm(10, 2);
    p.push(vmov_s_core(10, 10));
    p.push(vcvt_integer(8, true, true, false, 13, 10));
    p.push(vcvt_precision(false, 18, 13));
    p.push(vstr_d(18, 0, 16));
    p.push(ldr_imm(14, 0, 16));
    p.expect_eq(14, 2.0f64.to_bits() as u32, 6);
    p.push(ldr_imm(14, 0, 20));
    p.expect_eq(14, (2.0f64.to_bits() >> 32) as u32, 7);

    p.exit_ok();
    assert_eq!(run(&p.finish(&float_data(&[2.5, -7.75, 0.0, 0.0]))), 0);
}

#[test]
fn vfp_load_store_and_block_transfers() {
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.mov_imm(1, 0x1234_5678);
    p.push(vmov_s_core(0, 1));
    p.push(vstr_s(0, 0, 0));
    p.push(vldr_s(1, 0, 0));
    p.push(vmov_core_s(2, 1));
    p.expect_eq(2, 0x1234_5678, 1);

    // A double register is two words, low first.
    p.mov_imm(3, 0x9abc_def0);
    p.push(vmov_d_core(2, 1, 3));
    p.push(vstr_d(2, 0, 8));
    p.push(ldr_imm(4, 0, 8));
    p.push(ldr_imm(5, 0, 12));
    p.expect_eq(4, 0x1234_5678, 2);
    p.expect_eq(5, 0x9abc_def0, 3);
    p.push(vmov_core_d(6, 7, 2));
    p.expect_eq(6, 0x1234_5678, 4);
    p.expect_eq(7, 0x9abc_def0, 5);

    // `vpush`/`vpop` round-trip two double registers through the stack.
    p.push(vmov_d_core(8, 1, 3));
    p.push(vmov_d_core(9, 3, 1));
    p.push(vpush(8, 8));
    p.mov_imm(1, 0);
    p.mov_imm(3, 0);
    p.push(vmov_d_core(8, 1, 3));
    p.push(vmov_d_core(9, 3, 1));
    p.push(vpop(8, 8));
    p.push(vmov_core_d(10, 11, 8));
    p.expect_eq(10, 0x1234_5678, 6);
    p.expect_eq(11, 0x9abc_def0, 7);
    p.push(vmov_core_d(2, 4, 9));
    p.expect_eq(2, 0x9abc_def0, 8);

    // `vstmia`/`vldmia` of one double register, with and without writeback.
    p.mov_imm(0, DATA as u32 + 64);
    p.push(vstmia(2, 8, 0, true));
    p.expect_eq(0, DATA as u32 + 72, 9);
    p.mov_imm(1, DATA as u32 + 64);
    p.push(ldr_imm(2, 1, 0));
    p.expect_eq(2, 0x1234_5678, 10);
    p.push(ldr_imm(3, 1, 4));
    p.expect_eq(3, 0x9abc_def0, 11);
    p.push(vldmia(2, 20, 1, false));
    p.push(vmov_core_d(4, 5, 20));
    p.expect_eq(4, 0x1234_5678, 12);
    p.expect_eq(5, 0x9abc_def0, 13);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn fpscr_round_trips_and_controls_rounding() {
    let mut p = Program::new();
    // FZ and a non-default rounding mode survive a write and a read.
    p.mov_imm(0, 0x0140_0000); // FZ (bit 24) and RMode = 1 (+inf)
    p.push(vmsr_fpscr(0));
    p.push(vmrs_fpscr(1));
    p.expect_eq(1, 0x0140_0000, 1);

    // Round towards +inf makes 2.5 convert to 3, and the default mode to 2.
    p.mov_imm(2, DATA as u32);
    p.push(vldr_s(0, 2, 0));
    p.push(vcvt_integer(0xc, false, false, false, 1, 0));
    p.push(vmov_core_s(3, 1));
    p.expect_eq(3, 3, 2);
    p.mov_imm(0, 0);
    p.push(vmsr_fpscr(0));
    p.push(vcvt_integer(0xc, false, false, false, 1, 0));
    p.push(vmov_core_s(3, 1));
    p.expect_eq(3, 2, 3);

    // A divide by zero sets the cumulative bit, and the quotient is an infinity.
    p.push(vldr_s(1, 2, 4)); // 1.0
    p.push(vldr_s(2, 2, 8)); // 0.0
    p.push(vfp_bin(0b100, false, false, 3, 1, 2));
    p.push(vmov_core_s(4, 3));
    p.expect_eq(4, f32::INFINITY.to_bits(), 4);
    p.push(vmrs_fpscr(5));
    p.push(ubfx(6, 5, 1, 1));
    p.expect_eq(6, 1, 5);

    p.exit_ok();
    assert_eq!(run(&p.finish(&float_data(&[2.5, 1.0, 0.0]))), 0);
}

#[test]
fn neon_lanes_and_integer_arithmetic() {
    let mut p = Program::new();
    // d0 = {0x11223344, 0x11223344} and d1 = 0x01010101 in every lane.
    p.mov_imm(0, 0x1122_3344);
    p.push(vmov_lane_to(32, 0, 0, 0));
    p.push(vmov_lane_to(32, 0, 1, 0));
    p.mov_imm(1, 0x0101_0101);
    p.push(vdup_core(32, 1, 1));
    p.push(vmov_lane_from(32, 1, 0, 2, false));
    p.expect_eq(2, 0x0101_0101, 1);

    // vadd.i32, then the four bitwise operations.
    p.push(neon3(false, false, 2, 8, 2, 0, 1));
    p.push(vmov_lane_from(32, 2, 0, 3, false));
    p.expect_eq(3, 0x1223_3445, 2);
    p.push(neon3(true, false, 0, 1, 3, 0, 1)); // vand
    p.push(vmov_lane_from(32, 3, 0, 4, false));
    p.expect_eq(4, 0x0100_0100, 3);
    p.push(neon3(true, false, 2, 1, 3, 0, 1)); // vorr
    p.push(vmov_lane_from(32, 3, 0, 5, false));
    p.expect_eq(5, 0x1123_3345, 4);
    p.push(neon3(true, true, 0, 1, 3, 0, 1)); // veor
    p.push(vmov_lane_from(32, 3, 0, 6, false));
    p.expect_eq(6, 0x1023_3245, 5);
    p.push(neon3(true, false, 1, 1, 3, 0, 1)); // vbic
    p.push(vmov_lane_from(32, 3, 0, 7, false));
    p.expect_eq(7, 0x1022_3244, 6);

    // 16-bit subtraction and multiplication, read back signed and unsigned.
    p.mov_imm(8, 0x1122_3344);
    p.push(vdup_core(16, 4, 8));
    p.mov_imm(9, 0x0000_0101);
    p.push(vdup_core(16, 5, 9));
    p.push(neon3(false, true, 1, 8, 6, 4, 5)); // vsub.i16
    p.push(vmov_lane_from(16, 6, 0, 10, false));
    p.expect_eq(10, 0x3243, 7);
    // vmla.i16 reads its addend from the destination, which starts at zero, so the
    // result is the low half of the product.
    p.push(neon3(false, false, 1, 9, 7, 4, 5));
    p.push(vmov_lane_from(16, 7, 0, 11, false));
    p.expect_eq(11, 0x7744, 8);

    // vmax.s16 and vmin.u16.
    p.push(neon3(false, false, 1, 6, 8, 4, 5)); // vmax.s16
    p.push(vmov_lane_from(16, 8, 0, 2, false));
    p.expect_eq(2, 0x3344, 9);
    p.push(neon3(true, true, 1, 6, 9, 4, 5)); // vmin.u16
    p.push(vmov_lane_from(16, 9, 0, 2, false));
    p.expect_eq(2, 0x0101, 10);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn neon_byte_absolute_value_and_negate() {
    let mut p = Program::new();
    p.mov_imm(0, 0x0000_00f6); // -10 in the low byte
    p.push(vmov_lane_to(32, 0, 0, 0));
    // vabs.s8 (the two-register group) turns it into 10.
    p.push(0xF3B1_1300); // vabs.s8 d1, d0
    p.push(vmov_lane_from(8, 1, 0, 1, false));
    p.expect_eq(1, 0x0a, 1);
    p.push(0xF3B1_1380); // vneg.s8 d1, d0
    p.push(vmov_lane_from(8, 1, 0, 2, true));
    p.expect_eq(2, 0xf6, 2);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn neon_comparisons_and_extract() {
    let mut p = Program::new();
    // vceq.i16 / vcgt.s16 produce all-ones lanes.
    p.mov_imm(0, 0x0000_0002);
    p.push(vdup_core(16, 0, 0));
    p.mov_imm(1, 0x0000_0002);
    p.push(vdup_core(16, 1, 1));
    p.push(neon3(true, true, 1, 8, 2, 0, 1)); // vceq.i16 of equal values
    p.push(vmov_lane_from(16, 2, 0, 3, true));
    p.expect_eq(3, 0xffff, 1);
    p.mov_imm(4, 0x0000_0001);
    p.push(vdup_core(16, 3, 4));
    p.push(neon3(false, false, 1, 3, 4, 0, 3)); // vcgt.s16: 2 > 1
    p.push(vmov_lane_from(16, 4, 0, 5, true));
    p.expect_eq(5, 0xffff, 2);
    p.push(neon3(false, true, 1, 3, 6, 0, 3)); // vcgt.u16 too
    p.push(vmov_lane_from(16, 6, 0, 7, true));
    p.expect_eq(7, 0xffff, 3);

    // vext.8 takes bytes from the concatenation of the two registers.
    p.mov_imm(0, 0x0302_0100);
    p.push(vmov_lane_to(32, 8, 0, 0));
    p.mov_imm(1, 0x0706_0504);
    p.push(vmov_lane_to(32, 8, 1, 1));
    p.push(vmov_lane_to(32, 9, 0, 1));
    p.mov_imm(2, 0x0b0a_0908);
    p.push(vmov_lane_to(32, 9, 0, 2));
    p.mov_imm(3, 0x0f0e_0d0c);
    p.push(vmov_lane_to(32, 9, 1, 3));
    p.push(vext(10, 8, 9, 2));
    p.push(vmov_lane_from(32, 10, 0, 4, false));
    p.expect_eq(4, 0x0504_0302, 4);
    p.push(vmov_lane_from(32, 10, 1, 5, false));
    p.expect_eq(5, 0x0908_0706, 5);
    p.push(vext(11, 8, 9, 6));
    p.push(vmov_lane_from(32, 11, 0, 6, false));
    p.expect_eq(6, 0x0908_0706, 6);
    p.push(vmov_lane_from(32, 11, 1, 7, false));
    p.expect_eq(7, 0x0d0c_0b0a, 7);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn neon_structure_transfers_layout() {
    let mut p = Program::new();
    let mut data = vec![0u8; 64];
    // The interleaved pair of streams that `vld2.8` splits.
    data[8..16].copy_from_slice(&[0x11, 0x21, 0x12, 0x22, 0x13, 0x23, 0x14, 0x24]);

    p.mov_imm(0, DATA as u32);
    p.mov_imm(1, 0x0302_0100);
    p.push(vmov_lane_to(32, 0, 0, 1));
    p.mov_imm(1, 0x0706_0504);
    p.push(vmov_lane_to(32, 1, 0, 1));

    // vst1.32 {d0}, [r0]! stores eight bytes and advances the base.
    p.push(neon_struct(false, 0b0111, 32, 0, 0, 0b1101));
    p.expect_eq(0, DATA as u32 + 8, 1);
    p.mov_imm(2, DATA as u32);
    p.push(ldr_imm(3, 2, 0));
    p.expect_eq(3, 0x0302_0100, 2);

    // vld2.8 splits two interleaved streams: even bytes to d2, odd to d3.
    p.mov_imm(4, DATA as u32 + 8);
    p.push(neon_struct(true, 0b1000, 8, 2, 4, 0b1111));
    p.push(vmov_lane_from(8, 2, 0, 5, false));
    p.push(vmov_lane_from(8, 2, 1, 6, true));
    p.push(vmov_lane_from(8, 3, 0, 7, false));
    p.expect_eq(5, 0x11, 3);
    p.expect_eq(6, 0x12, 4);
    p.expect_eq(7, 0x21, 5);

    // vst2.8 writes them back interleaved.
    p.mov_imm(8, DATA as u32 + 16);
    p.push(neon_struct(false, 0b1000, 8, 2, 8, 0b1111));
    p.push(ldr_imm(9, 8, 0));
    p.expect_eq(9, 0x2212_2111, 6);

    // A register post-index is applied after the transfer.
    p.mov_imm(10, DATA as u32 + 24);
    p.mov_imm(11, 8);
    p.push(neon_struct(true, 0b0111, 32, 0, 10, 11));
    p.expect_eq(10, DATA as u32 + 32, 7);

    p.exit_ok();
    assert_eq!(run(&p.finish(&data)), 0);
}

#[test]
fn neon_float_lanes_match_the_scalar_path() {
    // The same operation through NEON `vadd.f32` and through the scalar VFP unit must
    // produce the same bits in every lane.
    let values = [1.5f32, -2.25, 3.5, 0.125];
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    for (index, _) in values.iter().enumerate() {
        p.push(vldr_s(index as u32, 0, (index * 4) as u32));
    }
    // d2 = {s4, s5} holds the first two values and d3 the other two, one operand at a
    // time from the core registers.
    for index in 0..4u32 {
        p.push(vmov_core_s(4 + index, index));
        p.push(vmov_lane_to(32, 2 + index / 2, index % 2, 4 + index));
    }
    p.push(neon3(false, false, 0, 0b1101, 4, 2, 2)); // vadd.f32 d4, d2, d2
    p.push(vmov_lane_from(32, 4, 0, 1, false));
    p.expect_eq(1, (values[0] + values[0]).to_bits(), 1);
    p.push(neon3(true, true, 0, 0b1101, 5, 2, 3)); // vmul.f32 d5, d2, d3
    p.push(vmov_lane_from(32, 5, 0, 2, false));
    p.expect_eq(2, (values[0] * values[2]).to_bits(), 2);
    p.push(vmov_lane_from(32, 5, 1, 3, false));
    p.expect_eq(3, (values[1] * values[3]).to_bits(), 3);
    p.push(neon3(false, false, 0, 0b1111, 6, 2, 3)); // vmax.f32
    p.push(vmov_lane_from(32, 6, 0, 4, false));
    p.expect_eq(4, values[0].max(values[2]).to_bits(), 4);
    p.push(neon3(false, false, 2, 0b1111, 7, 2, 3)); // vmin.f32
    p.push(vmov_lane_from(32, 7, 1, 5, false));
    p.expect_eq(5, values[1].min(values[3]).to_bits(), 5);
    p.push(neon3(false, false, 0, 0b1110, 8, 2, 2)); // vceq.f32 with itself
    p.push(vmov_lane_from(32, 8, 0, 6, false));
    p.expect_eq(6, 0xffff_ffff, 6);

    p.exit_ok();
    assert_eq!(run(&p.finish(&float_data(&values))), 0);
}

#[test]
fn neon_modified_immediate_and_wide_forms() {
    let mut p = Program::new();
    // vmov.i32 writes the immediate to every 32-bit lane.
    p.push(neon_imm(0b0000, 0, false, 0xff));
    p.push(vmov_lane_from(32, 0, 0, 1, false));
    p.push(vmov_lane_from(32, 0, 1, 2, false));
    p.expect_eq(1, 0xff, 1);
    p.expect_eq(2, 0xff, 2);

    // vorr.i32 keeps the previous contents of the lane it does not set.
    p.mov_imm(3, 0x0f00_0000);
    p.push(vmov_lane_to(32, 1, 1, 3));
    p.push(neon_imm(0b0001, 1, false, 0xff));
    p.push(vmov_lane_from(32, 1, 0, 4, false));
    p.expect_eq(4, 0xff, 3);
    p.push(vmov_lane_from(32, 1, 1, 5, false));
    p.expect_eq(5, 0x0f00_00ff, 4);

    // A quadword form covers both halves.
    p.push(neon_imm(0b1110, 0, true, 0x5f));
    p.push(vmov_lane_from(32, 0, 0, 6, false));
    p.push(vmov_lane_from(32, 1, 3, 7, false));
    p.expect_eq(6, 0x5f5f_5f5f, 5);
    p.expect_eq(7, 0x5f5f_5f5f, 6);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn vfp_nan_rules() {
    // A signalling operand wins over a quiet one, and is quieted in the result.
    let quiet = f32::from_bits(0x7fc0_1234);
    let signalling = f32::from_bits(0x7f80_0001);
    let mut p = Program::new();
    p.mov_imm(0, DATA as u32);
    p.push(vldr_s(0, 0, 0));
    p.push(vldr_s(1, 0, 4));
    p.push(vfp_bin(0b011, false, false, 2, 0, 1)); // vadd s2, quiet, signalling
    p.push(vmov_core_s(3, 2));
    p.expect_eq(3, signalling.to_bits() | 0x0040_0000, 1);

    // `vcmpe` of a NaN sets Invalid Operation and reports unordered.
    p.push(vfp_compare(false, 0, 1, false, true));
    p.push(vmrs_fpscr(4));
    p.push(ubfx(5, 4, 0, 1));
    p.expect_eq(5, 1, 2);
    p.push(vmrs_apsr());
    p.expect_cond(VS, 3);

    p.exit_ok();
    assert_eq!(run(&p.finish(&float_data(&[quiet, signalling]))), 0);
}

#[test]
fn acquire_release_and_cp15() {
    let mut p = Program::new();
    // `stlh` and `ldrh` move halfwords like a plain access.
    p.mov_imm(0, DATA as u32);
    p.mov_imm(1, 0xabcd);
    p.push(0xE1E0_FC91); // stlh r1, [r0]
    p.push(0xE1D0_00B0); // ldrh r0, [r0]
    p.expect_eq(0, 0xabcd, 1);

    // `mcr`/`mrc` of TPIDRURW round-trip through `tpidr_el0`, and TPIDRURO ignores
    // writes.
    p.mov_imm(2, 0x4000_1234);
    p.push(0xEE0D_2F50); // mcr p15, 0, r2, c13, c0, 2
    p.push(0xEE1D_3F50); // mrc p15, 0, r3, c13, c0, 2
    p.expect_eq(3, 0x4000_1234, 2);
    p.push(0xEE1D_4F70); // mrc p15, 0, r4, c13, c0, 3
    p.push(0xEE0D_2F70); // mcr p15, 0, r2, c13, c0, 3 (ignored)
    p.push(0xEE1D_5F70); // mrc p15, 0, r5, c13, c0, 3
    p.push(cmp_reg(4, 5));
    p.expect_cond(EQ, 3);

    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

// --- Division, barriers, the counters and the media dual multiplies ----------------

/// `udiv`/`sdiv`: `cond 0111 00x1 Rd 1111 Rm 0001 Rn`.
fn divide(unsigned: bool, rd: u32, rn: u32, rm: u32) -> u32 {
    0xE710_F010 | (u32::from(unsigned) << 21) | (rd << 16) | (rm << 8) | rn
}

/// One of the media dual multiplies: `group` is bits 27:20, `form` bits 7:6 and
/// `modifier` bit 5 (the `<x>` exchange or the rounding bit).
fn media_mac(group: u32, form: u32, modifier: bool, rd: u32, ra: u32, rm: u32, rn: u32) -> u32 {
    0xE000_0000
        | (group << 20)
        | (rd << 16)
        | (ra << 12)
        | (rm << 8)
        | (form << 6)
        | (u32::from(modifier) << 5)
        | 0x10
        | rn
}

/// `mrrc p15, <opc>, Rt, Rt2, c14`: the 64-bit counter read.
fn mrrc(rt: u32, rt2: u32, opc: u32) -> u32 {
    0xEC50_0F0E | (rt2 << 16) | (rt << 12) | (opc << 4)
}

const DMB: u32 = 0xF57F_F05B;
const DSB: u32 = 0xF57F_F04F;
const ISB: u32 = 0xF57F_F06F;
const CLREX: u32 = 0xF57F_F070;
const SEVL: u32 = 0xF57F_F035;

#[test]
fn division_follows_the_architecture() {
    // `(numerator, divisor, quotient)` for the unsigned form and for the signed one,
    // which truncates towards zero and cannot trap on `INT_MIN / -1`.
    let unsigned: &[(u32, u32, u32)] = &[
        (7, 2, 3),
        (0, 5, 0),
        (0xffff_ffff, 1, 0xffff_ffff),
        (0xffff_ffff, 2, 0x7fff_ffff),
        (0x8000_0000, 0x8000_0000, 1),
        (1, 0, 0),
        (0, 0, 0),
    ];
    let signed: &[(u32, u32, u32)] = &[
        (7, 2, 3),
        ((-7i32) as u32, 2, (-3i32) as u32),
        (7, (-2i32) as u32, (-3i32) as u32),
        ((-7i32) as u32, (-2i32) as u32, 3),
        (0x8000_0000, 0xffff_ffff, 0x8000_0000),
        (0x8000_0000, 1, 0x8000_0000),
        (1, 0, 0),
    ];
    for (dividend, divisor, quotient) in unsigned {
        let mut p = Program::new();
        p.mov_imm(1, *dividend);
        p.mov_imm(2, *divisor);
        p.push(divide(true, 0, 1, 2));
        p.expect_eq(0, *quotient, 1);
        p.exit_ok();
        assert_eq!(run(&p.finish(&[])), 0, "udiv {dividend:#x} / {divisor:#x}");
    }
    for (dividend, divisor, quotient) in signed {
        let mut p = Program::new();
        p.mov_imm(1, *dividend);
        p.mov_imm(2, *divisor);
        p.push(divide(false, 0, 1, 2));
        p.expect_eq(0, *quotient, 2);
        p.exit_ok();
        assert_eq!(run(&p.finish(&[])), 0, "sdiv {dividend:#x} / {divisor:#x}");
    }
}

#[test]
fn division_leaves_the_flags_alone() {
    let mut p = Program::new();
    // `cmp r0, r0` sets Z and C; the divide that follows must not touch either.
    p.push(cmp_reg(0, 0));
    p.push(divide(false, 1, 1, 2));
    p.expect_cond(EQ, 1);
    p.expect_cond(CS, 2);
    p.expect_cond(PL, 3);
    p.expect_cond(VC, 4);
    // Q is not a division result either.
    p.push(divide(true, 3, 1, 2));
    p.push(mrs(4));
    p.mov_imm(5, 1 << 27);
    p.push(and_reg(6, 4, 5));
    p.expect_eq(6, 0, 5);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn barriers_and_hints_do_not_change_state() {
    let mut p = Program::new();
    p.mov_imm(0, 1);
    p.push(cmp_reg(0, 0));
    for word in [DMB, DSB, ISB, SEVL, NOP] {
        p.push(word);
    }
    p.expect_cond(EQ, 1);
    p.expect_cond(CS, 2);
    p.expect_eq(0, 1, 3);

    // `clrex` drops the exclusive monitor, so the `strex` that follows reports failure.
    p.mov_imm(1, DATA as u32);
    p.mov_imm(2, 0x1234);
    p.push(str_imm(2, 1, 0));
    p.push(ldrex(3, 1));
    p.push(CLREX);
    p.mov_imm(4, 0x5678);
    p.push(strex(5, 4, 1));
    p.expect_eq(5, 1, 4);
    p.push(ldr_imm(6, 1, 0));
    p.expect_eq(6, 0x1234, 5);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn mrrc_reads_a_running_counter() {
    let mut p = Program::new();
    p.push(mrrc(0, 1, 0)); // cntpct
    p.push(mrrc(2, 3, 0));
    // The counter only moves forward, so the second read is not below the first.
    p.push(cmp_reg(2, 0));
    p.expect_cond(CS, 1);
    // One second after reset the high word of a 19.2 MHz counter is still zero.
    p.expect_eq(1, 0, 2);
    p.expect_eq(3, 0, 3);
    // `cntvct` reads the same process clock.
    p.push(mrrc(4, 5, 1));
    p.push(cmp_reg(4, 0));
    p.expect_cond(CS, 4);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0);
}

#[test]
fn media_dual_multiplies_accumulate_with_q() {
    // `smuad rd, rn, rm` with rn = 0x0002_0003 and rm = 0x0004_0005 sums the halfword
    // products 3 * 5 + 2 * 4 = 23; `smuadx` exchanges the halves of the *second*
    // operand, giving 3 * 4 + 2 * 5 = 22. `smusd` subtracts the products (15 - 8) and
    // `smusdx` the exchanged ones (12 - 10).
    let dual: &[(u32, bool, u32, &str)] = &[
        (0, false, 23, "smuad"),
        (0, true, 22, "smuadx"),
        (1, false, 7, "smusd"),
        (1, true, 2, "smusdx"),
    ];
    for (form, exchange, expected, name) in dual {
        let mut p = Program::new();
        p.mov_imm(0, 0x0002_0003);
        p.mov_imm(1, 0x0004_0005);
        p.push(media_mac(0x70, *form, *exchange, 2, 15, 1, 0));
        p.expect_eq(2, *expected, 1);
        p.exit_ok();
        assert_eq!(run(&p.finish(&[])), 0, "{name}");
    }

    // `smlad` adds the accumulator with wraparound and sets Q; the 64-bit `smlald` and
    // the `smm*` forms leave Q alone.
    // `smlad rd, rn, rn, ra`: with rn = 1 the products are 1 * 1 + 0 * 0 = 1, so the
    // result is the accumulator plus one.
    let accumulate: &[(u32, u32, u32, &str)] = &[
        (0x0000_0001, 1, 2, "smlad"),
        (0xffff_ffff, 1, 0, "smlad with a negative accumulator"),
        (
            0x0000_0000,
            0x8000_8000,
            0x8000_0000,
            "smlad at the signed edge",
        ),
    ];
    for (addend, rn, expected, name) in accumulate {
        let mut p = Program::new();
        p.mov_imm(0, *addend);
        p.mov_imm(1, *rn);
        p.push(media_mac(0x70, 0, false, 2, 0, 1, 1));
        p.expect_eq(2, *expected, 1);
        p.exit_ok();
        assert_eq!(run(&p.finish(&[])), 0, "{name}");
    }

    // Q is set only when a 32-bit dual form overflows.
    let mut p = Program::new();
    p.mov_imm(0, 0x7fff_ffff);
    p.mov_imm(1, 1);
    p.push(media_mac(0x70, 0, false, 2, 0, 1, 1)); // 1 + 1, the accumulator is 0x7fffffff
    p.push(mrs(3));
    p.mov_imm(4, 1 << 27);
    p.push(and_reg(5, 3, 4));
    p.expect_eq(5, 1 << 27, 1);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0, "Q after an overflowing smlad");

    let mut p = Program::new();
    p.mov_imm(0, 0x0002_0003);
    p.mov_imm(1, 0x0004_0005);
    p.push(media_mac(0x74, 0, true, 12, 13, 1, 0)); // smlaldx r12, r13
    p.push(mrs(3));
    p.mov_imm(4, 1 << 27);
    p.push(and_reg(5, 3, 4));
    p.expect_eq(5, 0, 1);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0, "Q after an smlaldx");

    // `smmul` keeps the top word; the rounding form adds 0x8000_0000 first.
    let mut p = Program::new();
    p.mov_imm(0, 0x4000_0000);
    p.push(media_mac(0x75, 0, false, 1, 15, 0, 0));
    p.expect_eq(1, 0x1000_0000, 1);
    // 1 * -2^31 is -2^31, whose top word is -1; adding the rounding constant brings
    // it to zero.
    p.mov_imm(2, 0x0000_0001);
    p.mov_imm(3, 0x8000_0000);
    p.push(media_mac(0x75, 0, false, 4, 15, 3, 2));
    p.expect_eq(4, 0xffff_ffff, 2);
    p.push(media_mac(0x75, 0, true, 5, 15, 3, 2));
    p.expect_eq(5, 0, 3);
    // `smmla` adds `Ra << 32` before the shift, `smmls` subtracts the product.
    p.mov_imm(6, 2);
    p.mov_imm(7, 3);
    p.mov_imm(8, 3);
    p.push(media_mac(0x75, 0, false, 9, 8, 7, 6));
    p.expect_eq(9, 3, 4);
    p.push(media_mac(0x75, 3, false, 10, 8, 7, 6));
    p.expect_eq(10, 2, 5);
    p.exit_ok();
    assert_eq!(run(&p.finish(&[])), 0, "smmul/smmla/smmls");
}
