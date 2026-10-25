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

    /// Fails check `id` unless `reg == value`. `r12` is the scratch register.
    fn expect_eq(&mut self, reg: u32, value: u32, id: u16) {
        assert_ne!(reg, 12, "expect_eq needs r12 as scratch");
        self.mov_imm(12, value);
        self.push(cmp_reg(reg, 12));
        self.push(cond_branch(EQ, 1));
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
    p.expect_eq(0, 0x4010_0000, 11);
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
    // `smlad r0, r1, r2, r3`: a media multiply-accumulate form outside the slice.
    let mut p = Program::new();
    p.push(0xE700_0211);
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
