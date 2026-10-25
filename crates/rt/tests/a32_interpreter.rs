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
    // `uxtb r1, r1`: allocated, media/parallel class, outside the slice.
    let mut p = Program::new();
    p.push(0xE6EF_1071);
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
