//! Behavioural tests for the A32 decoder.
//!
//! Every expected encoding below comes from the ARM ARM (A5.1 encoding tables and the
//! A8.8 per-instruction pages for ARMv7-A), not from this decoder; the expected text is
//! the assembly syntax those pages use.

use kirikae_a64::a32::*;

fn decode_at(pc: u32, word: u32) -> Insn {
    decode(pc, word).unwrap_or_else(|error| panic!("{word:#010x} at {pc:#x}: {error}"))
}

fn text_at(pc: u32, word: u32) -> String {
    decode_at(pc, word).to_string()
}

fn text(word: u32) -> String {
    text_at(0x1000, word)
}

fn block(word: u32) -> Block {
    match decode_at(0x1000, word).operands.as_slice() {
        [Operand::Block(block)] => *block,
        other => panic!("{word:#010x} is not a block transfer: {other:?}"),
    }
}

fn mem(word: u32) -> Mem {
    match decode_at(0x1000, word).operands.as_slice() {
        [Operand::Reg(_), Operand::Mem(mem)]
        | [Operand::Regs { .. }, Operand::Mem(mem)]
        | [Operand::S(_), Operand::Mem(mem)]
        | [Operand::D(_), Operand::Mem(mem)] => *mem,
        other => panic!("{word:#010x} is not a single transfer: {other:?}"),
    }
}

/// `uxtb`/`sxth`/... with an optional `ror #rotate` (0, 8, 16 or 24).
fn extend(group: u32, rd: u32, rn: u32, rm: u32, rotate: u32) -> u32 {
    0xE600_0000
        | (group << 20)
        | (rn << 16)
        | (rd << 12)
        | ((rotate / 8) << 10)
        | (0b0111 << 4)
        | rm
}

/// `ssat`/`usat`; `sat` is the encoded 5-bit field.
fn saturate(group: u32, rd: u32, sat: u32, rm: u32, imm5: u32, asr: bool) -> u32 {
    0xE600_0000
        | (group << 20)
        | (sat << 16)
        | (rd << 12)
        | (imm5 << 7)
        | (u32::from(asr) << 6)
        | 0x10
        | rm
}

/// `ssat16`/`usat16`; `sat` is the encoded 4-bit field.
fn saturate16(group: u32, rd: u32, sat: u32, rm: u32) -> u32 {
    0xE600_0000 | (group << 20) | (sat << 16) | (rd << 12) | (0xF << 8) | (0b0011 << 4) | rm
}

/// `bfi`/`bfc` (bits 6:4 = 001) or `sbfx`/`ubfx` (bits 6:4 = 101).
fn bitfield(group: u32, extract: bool, rd: u32, rn: u32, lsb: u32, field: u32) -> u32 {
    0xE600_0000
        | (group << 20)
        | (field << 16)
        | (rd << 12)
        | (lsb << 7)
        | ((if extract { 0b101 } else { 0b001 }) << 4)
        | rn
}

#[test]
fn data_processing_immediate() {
    assert_eq!(text(0xE3A00001), "mov r0, #0x1");
    assert_eq!(text(0xE20320FF), "and r2, r3, #0xff");
    assert_eq!(text(0xE22210FF), "eor r1, r2, #0xff");
    assert_eq!(text(0xE2421004), "sub r1, r2, #0x4");
    assert_eq!(text(0xE2610000), "rsb r0, r1, #0x0");
    assert_eq!(text(0xE2810002), "add r0, r1, #0x2");
    assert_eq!(text(0xE2A10002), "adc r0, r1, #0x2");
    assert_eq!(text(0xE2C10002), "sbc r0, r1, #0x2");
    assert_eq!(text(0xE2E10002), "rsc r0, r1, #0x2");
    assert_eq!(text(0xE3810002), "orr r0, r1, #0x2");
    assert_eq!(text(0xE3C10002), "bic r0, r1, #0x2");
    assert_eq!(text(0xE3E00002), "mvn r0, #0x2");

    // Examples from the real MK8D `main` disassembly in aarch32-confirmation.md.
    assert_eq!(text_at(0x0800_6028, 0xE28DB008), "add r11, sp, #0x8");
    assert_eq!(text_at(0x0800_602C, 0xE24DD048), "sub sp, sp, #0x48");
    assert_eq!(text_at(0x0800_6030, 0xE59F0164), "ldr r0, [pc, #0x164]");
    assert_eq!(text_at(0x0800_6038, 0xE3500000), "cmp r0, #0x0");
}

#[test]
fn data_processing_register() {
    assert_eq!(text(0xE1A00001), "mov r0, r1");
    assert_eq!(text(0xE1A02003), "mov r2, r3");
    assert_eq!(text(0xE0810002), "add r0, r1, r2");
    assert_eq!(text(0xE1B00001), "movs r0, r1");
    assert_eq!(text(0xE0900001), "adds r0, r0, r1");
    assert_eq!(text(0xE0B10002), "adcs r0, r1, r2");

    // Shifts, including the architectural normalisation of a zero amount.
    assert_eq!(text(0xE1A00121), "mov r0, r1, lsr #2");
    assert_eq!(text(0xE1A00041), "mov r0, r1, asr #32");
    assert_eq!(text(0xE1A00141), "mov r0, r1, asr #2");
    assert_eq!(text(0xE1A00261), "mov r0, r1, ror #4");
    assert_eq!(text(0xE1A00061), "mov r0, r1, rrx");
    assert_eq!(text(0xE1A00211), "mov r0, r1, lsl r2");
    assert_eq!(text(0xE1A00231), "mov r0, r1, lsr r2");
    assert_eq!(text(0xE1A00251), "mov r0, r1, asr r2");
    assert_eq!(text(0xE1A00271), "mov r0, r1, ror r2");
}

#[test]
fn flag_testing_forms() {
    for (word, expected) in [
        (0xE3100001u32, "tst r0, #0x1"),
        (0xE1300001, "teq r0, r1"),
        (0xE3510000, "cmp r1, #0x0"),
        (0xE1700001, "cmn r0, r1"),
        (0xE1100001, "tst r0, r1"),
        (0xE1500211, "cmp r0, r1, lsl r2"),
    ] {
        let insn = decode_at(0x1000, word);
        assert_eq!(insn.to_string(), expected);
        assert!(insn.opcode.is_flag_test(), "{expected}");
        assert!(insn.sets_flags, "{expected}");
        // No destination register: the first operand is Rn, not Rd.
        assert!(matches!(insn.operands[0], Operand::Reg(_)));
    }
}

#[test]
fn immediate_rotation() {
    // ROR(0xff, 8) = 0xff000000, encoded with rotate = 4.
    let insn = decode_at(0x1000, 0xE3A004FF);
    assert_eq!(insn.operands[1], Operand::Imm(0xff00_0000));
    assert_eq!(insn.to_string(), "mov r0, #0xff000000");

    // ROR(0x02, 2) = 0x80000000, encoded with rotate = 1.
    let insn = decode_at(0x1000, 0xE3A00102);
    assert_eq!(insn.operands[1], Operand::Imm(0x8000_0000));
    assert_eq!(insn.to_string(), "mov r0, #0x80000000");

    // ROR(0x3f, 24) = 0x3f00, encoded with rotate = 12.
    let insn = decode_at(0x1000, 0xE2010C3F);
    assert_eq!(insn.operands[2], Operand::Imm(0x3f00));
    assert_eq!(insn.to_string(), "and r0, r1, #0x3f00");
}

#[test]
fn condition_field() {
    let insn = decode_at(0x1000, 0x1284_000C);
    assert_eq!(insn.cond, Cond::Ne);
    assert_eq!(insn.mnemonic(), "addne");
    assert_eq!(insn.to_string(), "addne r0, r4, #0xc");

    assert_eq!(text(0x01A02003), "moveq r2, r3");
    assert_eq!(decode_at(0x1000, 0x41A02003).cond, Cond::Mi);
    assert_eq!(decode_at(0x1000, 0xE1A02003).cond, Cond::Al);

    // Suffix order is mnemonic, width, S, addressing mode, condition.
    assert_eq!(text(0x1090_0001), "addsne r0, r0, r1");
    assert_eq!(text(0x15D1_2004), "ldrbne r2, [r1, #0x4]");

    // Field 0b1111 is the unconditional space, which this slice only enters for `blx`.
    assert_eq!(decode(0x1000, 0xF000_0000), Err(DecodeError::Unsupported));
    assert_eq!(
        text(0xF5D1_F000),
        "pld [r1]",
        "pld is a hint in the unconditional space"
    );
}

#[test]
fn conditions_evaluate_against_nzcv() {
    const N: u32 = 0x8000_0000;
    const Z: u32 = 0x4000_0000;
    const C: u32 = 0x2000_0000;
    const V: u32 = 0x1000_0000;

    assert!(Cond::Eq.holds(Z));
    assert!(!Cond::Eq.holds(0));
    assert!(Cond::Ne.holds(0));
    assert!(!Cond::Ne.holds(Z));
    assert!(Cond::Cs.holds(C));
    assert!(Cond::Cc.holds(0));
    assert!(Cond::Mi.holds(N));
    assert!(Cond::Pl.holds(0));
    assert!(Cond::Vs.holds(V));
    assert!(Cond::Vc.holds(0));
    assert!(Cond::Hi.holds(C));
    assert!(!Cond::Hi.holds(C | Z));
    assert!(Cond::Ls.holds(C | Z));
    assert!(Cond::Ls.holds(0));
    assert!(Cond::Ge.holds(0));
    assert!(Cond::Ge.holds(N | V));
    assert!(!Cond::Ge.holds(N));
    assert!(Cond::Lt.holds(N));
    assert!(Cond::Gt.holds(C));
    assert!(!Cond::Gt.holds(Z));
    assert!(Cond::Le.holds(Z));
    assert!(Cond::Le.holds(N));
    assert!(Cond::Al.holds(0));

    for cond in Cond::ALL {
        assert_eq!(*cond, Cond::from_field(cond.field()).unwrap());
        assert_eq!(*cond, cond.invert().invert());
    }
    assert_eq!(Cond::from_field(0xf), None);
}

#[test]
fn branch_targets_are_pc_plus_eight() {
    let insn = decode_at(0x1000, 0xEB00_0000);
    assert_eq!(insn.opcode, Opcode::Bl);
    assert_eq!(insn.to_string(), "bl #0x1008");
    assert_eq!(
        insn.target(),
        Some(Target {
            addr: 0x1008,
            state: ExecState::Arm
        })
    );

    // imm24 = -2 -> branch back to the instruction itself.
    let insn = decode_at(0x1000, 0xEAFF_FFFE);
    assert_eq!(insn.opcode, Opcode::B);
    assert_eq!(insn.to_string(), "b #0x1000");

    let insn = decode_at(0x2000, 0x1A00_0001);
    assert_eq!(insn.cond, Cond::Ne);
    assert_eq!(insn.to_string(), "bne #0x200c");

    // The 32-bit target wraps around the address space.
    let insn = decode_at(0xFFFF_FFFC, 0xEA00_0000);
    assert_eq!(insn.target().map(|target| target.addr), Some(0x4));

    // `blx` immediate: Align(PC, 4) + sext(imm24) * 4 + H * 2, always Thumb.
    let insn = decode_at(0x1000, 0xFA00_0000);
    assert_eq!(insn.opcode, Opcode::Blx);
    assert_eq!(insn.to_string(), "blx #0x1008");
    assert_eq!(
        insn.target(),
        Some(Target {
            addr: 0x1008,
            state: ExecState::Thumb
        })
    );
    // H sets bit 1 of the target, so the branch can land halfword-aligned.
    assert_eq!(
        decode_at(0x1000, 0xFB00_0000)
            .target()
            .map(|target| target.addr),
        Some(0x100a)
    );
    // Negative offset.
    assert_eq!(
        decode_at(0x1000, 0xFAFF_FFFF)
            .target()
            .map(|target| target.addr),
        Some(0x1004)
    );
    // PC is aligned down before the offset is added.
    assert_eq!(
        decode_at(0x2004, 0xFB00_0000)
            .target()
            .map(|target| target.addr),
        Some(0x200e)
    );

    // Register branches carry no immediate target.
    assert_eq!(decode_at(0x1000, 0xE12F_FF1E).target(), None);
}

#[test]
fn block_transfer_lists_and_modes() {
    // `push {r11, lr}` and `push {r4-r11, lr}` as measured on the MK8D module.
    let pushed = block(0xE92D_4800);
    assert_eq!(pushed.base, Reg::SP);
    assert_eq!(pushed.list.bits(), 0x4800);
    assert_eq!(pushed.mode, BlockMode::DecrementBefore);
    assert!(pushed.writeback);
    assert_eq!(decode_at(0x1000, 0xE92D_4800).opcode, Opcode::Stm);
    assert_eq!(text(0xE92D_4800), "stmdb sp!, {r11, lr}");
    assert_eq!(text(0xE92D_4FF0), "stmdb sp!, {r4-r11, lr}");

    // `pop {r4, pc}`.
    let popped = block(0xE8BD_8010);
    assert_eq!(popped.mode, BlockMode::IncrementAfter);
    assert!(popped.writeback);
    assert!(popped.list.contains(Reg::PC));
    assert!(popped.list.contains(Reg::R4));
    assert_eq!(decode_at(0x1000, 0xE8BD_8010).opcode, Opcode::Ldm);
    assert_eq!(text(0xE8BD_8010), "ldmia sp!, {r4, pc}");

    // All four addressing modes.
    assert_eq!(block(0xE890_0001).mode, BlockMode::IncrementAfter);
    assert_eq!(block(0xE990_0001).mode, BlockMode::IncrementBefore);
    assert_eq!(block(0xE810_0001).mode, BlockMode::DecrementAfter);
    assert_eq!(block(0xE910_0001).mode, BlockMode::DecrementBefore);
    assert_eq!(text(0xE910_0001), "ldmdb r0, {r0}");
    assert_eq!(text(0xE900_0001), "stmdb r0, {r0}");

    // Register list order and range compression.
    assert_eq!(
        block(0xE92D_4FF0).list.iter().collect::<Vec<_>>(),
        [
            Reg::R4,
            Reg::R5,
            Reg::R6,
            Reg::R7,
            Reg::R8,
            Reg::R9,
            Reg::R10,
            Reg::R11,
            Reg::LR,
        ]
    );
    assert_eq!(RegList::new(0x0001).unwrap().to_string(), "{r0}");
    assert_eq!(RegList::new(0x0003).unwrap().to_string(), "{r0, r1}");
    assert_eq!(RegList::new(0x0007).unwrap().to_string(), "{r0-r2}");
    assert_eq!(RegList::new(0x8001).unwrap().to_string(), "{r0, pc}");
    assert_eq!(RegList::new(0).map(RegList::bits), None);
}

#[test]
fn load_store_addressing_modes() {
    // From the MK8D disassembly.
    assert_eq!(text_at(0x0800_7000, 0xE59D1004), "ldr r1, [sp, #0x4]");
    assert_eq!(
        mem(0xE59D_1004),
        Mem {
            base: Reg::SP,
            offset: Offset::Imm(4),
            add: true,
            pre_indexed: true,
            writeback: false,
            size: MemSize::Word,
        }
    );

    assert_eq!(text(0xE581_0004), "str r0, [r1, #0x4]");
    assert_eq!(text(0xE524_3008), "str r3, [r4, #-0x8]!");
    assert_eq!(text(0xE4D3_2001), "ldrb r2, [r3], #0x1");
    assert_eq!(text(0xE5C1_0004), "strb r0, [r1, #0x4]");
    assert_eq!(text(0xE5D1_2004), "ldrb r2, [r1, #0x4]");

    let post = mem(0xE4D3_2001);
    assert!(!post.pre_indexed);
    assert!(!post.writeback);
    assert!(post.writes_back());
    assert_eq!(post.size, MemSize::Byte);
    assert_eq!(post.offset, Offset::Imm(1));

    let pre = mem(0xE524_3008);
    assert!(!pre.add);
    assert!(pre.pre_indexed);
    assert!(pre.writeback);
    assert!(pre.writes_back());

    // Register offsets: bit 4 clear, imm5 in 11:7 and the shift type in 6:5.
    assert_eq!(text(0xE791_0102), "ldr r0, [r1, r2, lsl #2]");
    assert_eq!(text(0xE711_0002), "ldr r0, [r1, -r2]");
    assert_eq!(text(0xE79F_1002), "ldr r1, [pc, r2]");
    assert_eq!(text(0xE691_0102), "ldr r0, [r1], r2, lsl #2");
    assert_eq!(
        mem(0xE791_0102).offset,
        Offset::Reg {
            reg: Reg::R2,
            shift: Shift::Lsl(2)
        }
    );

    // A word load or store to PC is a branch or a deprecated store of the PC value.
    let load = decode_at(0x1000, 0xE591_F000);
    assert_eq!(load.opcode, Opcode::Ldr);
    assert_eq!(load.to_string(), "ldr pc, [r1]");
    let store = decode_at(0x1000, 0xE581_F000);
    assert_eq!(store.opcode, Opcode::Str);
    assert_eq!(store.to_string(), "str pc, [r1]");
}

#[test]
fn multiplies() {
    assert_eq!(text(0xE000_0291), "mul r0, r1, r2");
    assert_eq!(text(0xE010_0291), "muls r0, r1, r2");
    assert_eq!(text(0xE020_3291), "mla r0, r1, r2, r3");

    let mla = decode_at(0x1000, 0xE020_3291);
    assert_eq!(mla.opcode, Opcode::Mla);
    assert_eq!(
        mla.operands,
        [
            Operand::Reg(Reg::R0),
            Operand::Reg(Reg::R1),
            Operand::Reg(Reg::R2),
            Operand::Reg(Reg::R3),
        ]
    );
    assert_eq!(decode_at(0x1000, 0xE010_0291).opcode, Opcode::Mul);
    assert!(decode_at(0x1000, 0xE010_0291).sets_flags);

    assert_eq!(decode(0x1000, 0xE00F_0291), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE000_1291), Err(DecodeError::Unpredictable));
}

#[test]
fn long_multiplies() {
    for (word, expected, opcode) in [
        (0xE081_0392u32, "umull r0, r1, r2, r3", Opcode::Umull),
        (0xE0A1_0392, "umlal r0, r1, r2, r3", Opcode::Umlal),
        (0xE0C1_0392, "smull r0, r1, r2, r3", Opcode::Smull),
        (0xE0E1_0392, "smlal r0, r1, r2, r3", Opcode::Smlal),
        (0xE091_0392, "umulls r0, r1, r2, r3", Opcode::Umull),
    ] {
        let insn = decode_at(0x1000, word);
        assert_eq!(insn.to_string(), expected);
        assert_eq!(insn.opcode, opcode);
        // The pair is RdLo first, the order the assembly syntax uses.
        assert_eq!(
            insn.operands[0],
            Operand::Regs {
                lo: Reg::R0,
                hi: Reg::R1
            }
        );
    }
    // `umaal` is allocated but outside the slice.
    assert_eq!(decode(0x1000, 0xE041_1291), Err(DecodeError::Unsupported));
    // RdHi and RdLo are the same register, or `pc` is involved.
    assert_eq!(decode(0x1000, 0xE081_1392), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE08F_0392), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE081_F392), Err(DecodeError::Unpredictable));
}

#[test]
fn halfword_multiplies() {
    for (word, expected) in [
        (0xE100_3281u32, "smlabb r0, r1, r2, r3"),
        (0xE100_32C1, "smlabt r0, r1, r2, r3"),
        (0xE100_32A1, "smlatb r0, r1, r2, r3"),
        (0xE100_32E1, "smlatt r0, r1, r2, r3"),
        (0xE120_3281, "smlawb r0, r1, r2, r3"),
        (0xE120_32C1, "smlawt r0, r1, r2, r3"),
        (0xE160_0281, "smulbb r0, r1, r2"),
        (0xE160_02C1, "smulbt r0, r1, r2"),
        (0xE160_02A1, "smultb r0, r1, r2"),
        (0xE160_02E1, "smultt r0, r1, r2"),
        (0xE140_3281, "smlalbb r3, r0, r1, r2"),
        (0xE140_32E1, "smlaltt r3, r0, r1, r2"),
    ] {
        assert_eq!(text(word), expected);
    }
    // `smul<x><y>` has Ra SBZ, and `smlal<x><y>` cannot repeat its destination.
    assert_eq!(decode(0x1000, 0xE160_1281), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE141_1281), Err(DecodeError::Unpredictable));
    // `smlaw<y>` and `smulw<y>` only encode the two `y` selectors.
    assert_eq!(decode(0x1000, 0xE120_32A1), Err(DecodeError::Unsupported));
}

#[test]
fn extra_load_stores() {
    assert_eq!(text(0xE1D1_00B4), "ldrh r0, [r1, #0x4]");
    assert_eq!(text(0xE1C1_00B4), "strh r0, [r1, #0x4]");
    assert_eq!(text(0xE1D1_00D4), "ldrsb r0, [r1, #0x4]");
    assert_eq!(text(0xE1D1_00F4), "ldrsh r0, [r1, #0x4]");
    assert_eq!(text(0xE191_00B2), "ldrh r0, [r1, r2]");
    assert_eq!(text(0xE181_00B2), "strh r0, [r1, r2]");
    assert_eq!(text(0xE191_00D2), "ldrsb r0, [r1, r2]");
    assert_eq!(text(0xE191_00F2), "ldrsh r0, [r1, r2]");
    assert_eq!(text(0xE0D1_00B4), "ldrh r0, [r1], #0x4");
    assert_eq!(text(0xE1F1_00B4), "ldrh r0, [r1, #0x4]!");

    let half = mem(0xE1D1_00B4);
    assert_eq!(half.size, MemSize::Halfword);
    assert_eq!(half.offset, Offset::Imm(4));
    assert_eq!(mem(0xE1D1_00D4).size, MemSize::Byte);
    assert_eq!(
        mem(0xE191_00F2).offset,
        Offset::Reg {
            reg: Reg::R2,
            shift: Shift::Lsl(0)
        }
    );
    assert_eq!(decode_at(0x1000, 0xE1D1_00D4).opcode, Opcode::Ldrsb);

    // `pc` as the destination, a base that also writes back and a non-zero SBZ field
    // in the register form are all unpredictable.
    assert_eq!(decode(0x1000, 0xE1D1_F0B4), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE1B0_00B4), Err(DecodeError::Unpredictable));
    assert!(decode(0x1000, 0xE1F1_00B4).is_ok());
    assert_eq!(decode(0x1000, 0xE191_01B2), Err(DecodeError::Unpredictable));
    // The unprivileged `ldrht`/`strht` forms are outside the slice.
    assert_eq!(decode(0x1000, 0xE0F1_00B4), Err(DecodeError::Unsupported));
}

#[test]
fn doubleword_transfers() {
    assert_eq!(text(0xE1C1_00D8), "ldrd r0, r1, [r1, #0x8]");
    assert_eq!(text(0xE1C1_00F8), "strd r0, r1, [r1, #0x8]");
    assert_eq!(text(0xE0C1_20D8), "ldrd r2, r3, [r1], #0x8");
    assert_eq!(text(0xE1C1_04D2), "ldrd r0, r1, [r1, #0x42]");
    assert_eq!(text(0xE181_00D2), "ldrd r0, r1, [r1, r2]");
    assert_eq!(decode_at(0x1000, 0xE1C1_00D8).opcode, Opcode::Ldrd);
    assert_eq!(decode_at(0x1000, 0xE1C1_00F8).opcode, Opcode::Strd);

    let pair = decode_at(0x1000, 0xE1C1_00D8);
    assert_eq!(
        pair.operands[0],
        Operand::Regs {
            lo: Reg::R0,
            hi: Reg::R1
        }
    );
    assert_eq!(mem(0xE1C1_00D8).size, MemSize::Double);

    // `rt` cannot be `lr` or `pc`, and the writeback base cannot alias either half.
    assert_eq!(decode(0x1000, 0xE1C1_E0D8), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE1C1_F0D8), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE1E0_00D8), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE1E0_00D8), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE1E1_00D8), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE1E1_10D8), Err(DecodeError::Unpredictable));
}

#[test]
fn exclusive_access() {
    assert_eq!(text(0xE191_0F9F), "ldrex r0, [r1]");
    assert_eq!(text(0xE1D1_0F9F), "ldrexb r0, [r1]");
    assert_eq!(text(0xE1F1_0F9F), "ldrexh r0, [r1]");
    assert_eq!(text(0xE182_0F91), "strex r0, r1, [r2]");
    assert_eq!(text(0xE1C2_0F91), "strexb r0, r1, [r2]");
    assert_eq!(text(0xE1E2_0F91), "strexh r0, r1, [r2]");

    let ldrex = decode_at(0x1000, 0xE191_0F9F);
    assert_eq!(ldrex.opcode, Opcode::Ldrex);
    assert_eq!(mem(0xE191_0F9F).offset, Offset::Imm(0));
    assert_eq!(mem(0xE191_0F9F).size, MemSize::Word);
    assert_eq!(mem(0xE1D1_0F9F).size, MemSize::Byte);
    assert_eq!(mem(0xE1F1_0F9F).size, MemSize::Halfword);

    let strex = decode_at(0x1000, 0xE182_0F91);
    assert_eq!(strex.opcode, Opcode::Strex);
    assert_eq!(strex.operands.len(), 3, "status, value and address");

    // The encoding has no offset field: bits 11:8 are fixed at 0b1111.
    assert_eq!(decode(0x1000, 0xE191_009F), Err(DecodeError::Unpredictable));
    // The status register and the stored register cannot be the same.
    assert_eq!(decode(0x1000, 0xE181_2F92), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE1D1_0F9E), Err(DecodeError::Unpredictable));
}

#[test]
fn media_extend_and_reverse() {
    // The instruction the MK8D `rtld` used to stop at, and the builder that
    // reproduces its encoding.
    assert_eq!(extend(0b01110, 1, 0xF, 1, 0), 0xE6EF_1071);
    assert_eq!(text(0xE6EF_1071), "uxtb r1, r1");
    assert_eq!(text(0xE6FF_1071), "uxth r1, r1");
    assert_eq!(text(0xE6AF_1071), "sxtb r1, r1");
    assert_eq!(text(0xE6BF_1071), "sxth r1, r1");

    // Rotated forms: the rotate field is bits 11:10.
    assert_eq!(text(0xE6EF_1471), "uxtb r1, r1, ror #8");
    assert_eq!(text(0xE6EF_1871), "uxtb r1, r1, ror #16");
    assert_eq!(text(0xE6EF_1C71), "uxtb r1, r1, ror #24");

    // The 16-bit variants and the accumulating forms (Rn != 0b1111).
    assert_eq!(text(0xE68F_0071), "sxtb16 r0, r1");
    assert_eq!(text(0xE6CF_0071), "uxtb16 r0, r1");
    assert_eq!(text(0xE68F_0471), "sxtb16 r0, r1, ror #8");
    assert_eq!(text(0xE6A1_2071), "sxtab r2, r1, r1");
    assert_eq!(text(0xE6E1_2071), "uxtab r2, r1, r1");
    assert_eq!(text(0xE6C8_1071), "uxtab16 r1, r8, r1");
    assert_eq!(text(0xE6BF_1071), "sxth r1, r1");

    let uxtab = decode_at(0x1000, 0xE6A1_2071);
    assert_eq!(uxtab.opcode, Opcode::Sxtab);
    assert_eq!(
        uxtab.operands,
        [
            Operand::Reg(Reg::R2),
            Operand::Reg(Reg::R1),
            Operand::Shifted {
                reg: Reg::R1,
                shift: Shift::Lsl(0)
            },
        ]
    );

    assert_eq!(text(0xE6BF_0F31), "rev r0, r1");
    assert_eq!(text(0xE6BF_0FB1), "rev16 r0, r1");
    assert_eq!(text(0xE6FF_0FB1), "revsh r0, r1");
    assert_eq!(text(0xE6FF_0F31), "rbit r0, r1");
    assert_eq!(text(0xE16F_0F11), "clz r0, r1");

    // Bits 9:8 are SBZ in the rotated forms and bits 11:8 / 19:16 are SBO in the
    // reversals; a non-zero value is unpredictable.
    assert_eq!(decode(0x1000, 0xE6AF_0571), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE6BF_0031), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE6BF_0231), Err(DecodeError::Unpredictable));
    // Rn and Rd cannot be `pc`.
    assert_eq!(decode(0x1000, 0xE6EF_F071), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE6EF_107F), Err(DecodeError::Unpredictable));
}

#[test]
fn media_pack_and_select() {
    assert_eq!(text(0xE681_0192), "pkhbt r0, r1, r2, lsl #3");
    assert_eq!(text(0xE681_0012), "pkhbt r0, r1, r2");
    assert_eq!(text(0xE681_01D2), "pkhtb r0, r1, r2, asr #3");
    // `asr #0` encodes a shift by 32.
    assert_eq!(text(0xE681_0052), "pkhtb r0, r1, r2, asr #32");
    assert_eq!(text(0xE681_0FB2), "sel r0, r1, r2");

    assert_eq!(
        decode_at(0x1000, 0xE681_0192).operands[2],
        Operand::Shifted {
            reg: Reg::R2,
            shift: Shift::Lsl(3)
        }
    );
    assert_eq!(
        decode_at(0x1000, 0xE681_0052).operands[2],
        Operand::Shifted {
            reg: Reg::R2,
            shift: Shift::Asr(32)
        }
    );

    // `sel` needs bits 11:8 = 0b1111, and the pack instructions cannot write `pc`.
    assert_eq!(decode(0x1000, 0xE681_00B2), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE681_F192), Err(DecodeError::Unpredictable));
    // A selector that no packing instruction uses is unallocated.
    assert_eq!(decode(0x1000, 0xE681_0032), Err(DecodeError::Unallocated));
}

#[test]
fn media_saturate() {
    assert_eq!(text(0xE6A1_0011), "ssat r0, #0x2, r1");
    assert_eq!(text(0xE6A1_2091), "ssat r2, #0x2, r1, lsl #1");
    assert_eq!(text(0xE6A1_2051), "ssat r2, #0x2, r1, asr #32");
    assert_eq!(text(0xE6A1_0F31), "ssat16 r0, #0x2, r1");
    assert_eq!(text(0xE6E1_0011), "usat r0, #0x1, r1");
    assert_eq!(text(0xE6E1_0F31), "usat16 r0, #0x1, r1");
    // Bit 20 is the top bit of the saturation width, so `ssat #32` is op1 = 0b01011.
    assert_eq!(text(0xE6BF_0011), "ssat r0, #0x20, r1");
    // The builders reproduce the encodings used above.
    assert_eq!(saturate(0b01010, 0, 1, 1, 0, false), 0xE6A1_0011);
    assert_eq!(saturate16(0b01010, 0, 1, 1), 0xE6A1_0F31);

    // The IR carries the width the assembly syntax prints, not the encoded field.
    assert_eq!(decode_at(0x1000, 0xE6A1_0011).operands[1], Operand::Imm(2));
    assert_eq!(decode_at(0x1000, 0xE6E1_0011).operands[1], Operand::Imm(1));
    assert_eq!(decode_at(0x1000, 0xE6A1_0F31).operands[1], Operand::Imm(2));
    assert_eq!(decode_at(0x1000, 0xE6E1_0F31).operands[1], Operand::Imm(1));

    // Bits 11:8 are SBO in the 16-bit forms.
    assert_eq!(decode(0x1000, 0xE6A1_0031), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE6A1_F011), Err(DecodeError::Unpredictable));
}

#[test]
fn media_bitfield() {
    // `bfc Rd, #lsb, #width`: the field holds the most significant bit.
    assert_eq!(text(0xE7C1_201F), "bfc r2, #0x0, #0x2");
    assert_eq!(
        text(bitfield(0b11100, false, 2, 0xF, 30, 31)),
        "bfc r2, #0x1e, #0x2"
    );
    assert_eq!(
        text(bitfield(0b11100, false, 2, 1, 4, 5)),
        "bfi r2, r1, #0x4, #0x2"
    );
    // `sbfx`/`ubfx` carry the width minus one, so field 7 is a width of 8.
    assert_eq!(
        text(bitfield(0b11010, true, 0, 1, 4, 11)),
        "sbfx r0, r1, #0x4, #0xc"
    );
    assert_eq!(
        text(bitfield(0b11110, true, 0, 1, 4, 11)),
        "ubfx r0, r1, #0x4, #0xc"
    );
    assert_eq!(
        text(bitfield(0b11010, true, 0, 1, 0, 7)),
        "sbfx r0, r1, #0x0, #0x8"
    );

    // An extract whose field reaches past bit 31 is unpredictable, and so is a `bfi`
    // whose most significant bit is below its least significant bit.
    assert_eq!(
        decode(0x1000, bitfield(0b11110, true, 0, 1, 20, 15)),
        Err(DecodeError::Unpredictable)
    );
    assert_eq!(
        decode(0x1000, bitfield(0b11100, false, 2, 1, 30, 1)),
        Err(DecodeError::Unpredictable)
    );
    assert_eq!(
        decode(0x1000, bitfield(0b11100, false, 15, 0xF, 0, 31)),
        Err(DecodeError::Unpredictable),
        "bfc cannot write pc"
    );
    // A field that covers the whole word is a legal `bfc`/`bfi`.
    assert_eq!(
        text(bitfield(0b11100, false, 0, 0xF, 0, 31)),
        "bfc r0, #0x0, #0x20"
    );
}

#[test]
fn parallel_add_subtract() {
    const GROUPS: [(u32, [&str; 6]); 6] = [
        (
            0b00001,
            ["sadd16", "sasx", "ssax", "ssub16", "sadd8", "ssub8"],
        ),
        (
            0b00010,
            ["qadd16", "qasx", "qsax", "qsub16", "qadd8", "qsub8"],
        ),
        (
            0b00011,
            ["shadd16", "shasx", "shsax", "shsub16", "shadd8", "shsub8"],
        ),
        (
            0b00101,
            ["uadd16", "uasx", "usax", "usub16", "uadd8", "usub8"],
        ),
        (
            0b00110,
            ["uqadd16", "uqasx", "uqsax", "uqsub16", "uqadd8", "uqsub8"],
        ),
        (
            0b00111,
            ["uhadd16", "uhasx", "uhsax", "uhsub16", "uhadd8", "uhsub8"],
        ),
    ];
    const SELECTORS: [u32; 6] = [1, 3, 5, 7, 9, 0xF];
    for (group, names) in GROUPS {
        for (index, selector) in SELECTORS.iter().enumerate() {
            let word = 0xE600_0000 | (group << 20) | (1 << 16) | (0xF << 8) | (selector << 4) | 2;
            assert_eq!(
                text(word),
                format!("{} r0, r1, r2", names[index]),
                "{word:#010x}"
            );
            assert_eq!(
                decode_at(0x1000, word).opcode,
                Opcode::parallel(group, *selector).unwrap()
            );
        }
    }

    // Only the four byte-wide forms touch GE, and only the six selectors exist.
    assert_eq!(decode(0x1000, 0xE611_0FD2), Err(DecodeError::Unallocated));
    assert_eq!(decode(0x1000, 0xE611_0FB2), Err(DecodeError::Unallocated));
    assert_eq!(decode(0x1000, 0xE611_0F1F), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE611_F012), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE611_0F1F), Err(DecodeError::Unpredictable));
}

#[test]
fn saturating_scalar_arithmetic() {
    assert_eq!(text(0xE101_0052), "qadd r0, r2, r1");
    assert_eq!(text(0xE121_0052), "qsub r0, r2, r1");
    assert_eq!(text(0xE141_0052), "qdadd r0, r2, r1");
    assert_eq!(text(0xE161_0052), "qdsub r0, r2, r1");

    let qadd = decode_at(0x1000, 0xE101_0052);
    assert_eq!(
        qadd.operands,
        [
            Operand::Reg(Reg::R0),
            Operand::Reg(Reg::R2),
            Operand::Reg(Reg::R1),
        ],
        "the assembly syntax is Rd, Rm, Rn"
    );
    // Bits 11:8 are SBZ and `pc` is not an operand.
    assert_eq!(decode(0x1000, 0xE101_0152), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE101_005F), Err(DecodeError::Unpredictable));
}

#[test]
fn mrs_msr_and_hints() {
    assert_eq!(text(0xE10F_0000), "mrs r0");
    assert_eq!(text(0xE128_F000), "msr r0");
    assert_eq!(text(0xE328_F001), "msr #0x1");
    assert_eq!(text(0xE320_F000), "nop #0x0");
    assert_eq!(text(0xE320_F001), "yield #0x1");
    assert_eq!(text(0xE320_F002), "wfe #0x2");
    assert_eq!(text(0xE320_F003), "wfi #0x3");
    assert_eq!(text(0xE320_F004), "sev #0x4");

    // Only the flags byte is writable; the SPSR and the control fields are rejected
    // as needing state this runtime does not model.
    assert_eq!(decode(0x1000, 0xE128_F000).unwrap().opcode, Opcode::Msr);
    assert_eq!(decode(0x1000, 0xE129_F000), Err(DecodeError::Privileged));
    assert_eq!(decode(0x1000, 0xE12F_F000), Err(DecodeError::Privileged));
    assert_eq!(decode(0x1000, 0xE14F_0000), Err(DecodeError::Privileged));
    assert_eq!(decode(0x1000, 0xE169_F000), Err(DecodeError::Privileged));
    assert_eq!(decode(0x1000, 0xE321_F000), Err(DecodeError::Privileged));

    // `mrs` cannot write `pc` and its SBZ fields must be zero; a hint with a
    // condition or an unassigned hint number is rejected.
    assert_eq!(decode(0x1000, 0xE10F_F000), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE10F_000F), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0x1320_F001), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE320_F005), Err(DecodeError::Unsupported));
}

#[test]
fn movw_and_movt() {
    let movw = decode_at(0x1000, 0xE301_0234);
    assert_eq!(movw.opcode, Opcode::Movw);
    assert_eq!(movw.operands, [Operand::Reg(Reg::R0), Operand::Imm(0x1234)]);
    assert_eq!(movw.to_string(), "movw r0, #0x1234");

    let movt = decode_at(0x1000, 0xE34A_0BCD);
    assert_eq!(movt.opcode, Opcode::Movt);
    assert_eq!(movt.operands, [Operand::Reg(Reg::R0), Operand::Imm(0xabcd)]);
    assert_eq!(movt.to_string(), "movt r0, #0xabcd");

    assert_eq!(decode(0x1000, 0xE30F_F234), Err(DecodeError::Unpredictable));
}

#[test]
fn register_branches() {
    assert_eq!(text(0xE12F_FF1E), "bx lr");
    assert_eq!(decode_at(0x1000, 0xE12F_FF1E).opcode, Opcode::Bx);
    assert_eq!(text(0xE12F_FF30), "blx r0");
    assert_eq!(decode_at(0x1000, 0xE12F_FF3E).opcode, Opcode::Blx);
    assert_eq!(text(0x012F_FF11), "bxeq r1");
    assert_eq!(decode(0x1000, 0xE12F_FF1F), Err(DecodeError::Unpredictable));
}

#[test]
fn svc_and_bkpt() {
    let svc = decode_at(0x1000, 0xEF00_0000);
    assert_eq!(svc.opcode, Opcode::Svc);
    assert_eq!(svc.operands, [Operand::Imm(0)]);
    assert_eq!(svc.to_string(), "svc #0x0");

    assert_eq!(text(0xEF12_3456), "svc #0x123456");
    assert_eq!(decode_at(0x1000, 0x0F00_0001).cond, Cond::Eq);

    let bkpt = decode_at(0x1000, 0xE121_2374);
    assert_eq!(bkpt.opcode, Opcode::Bkpt);
    assert_eq!(bkpt.operands, [Operand::Imm(0x1234)]);
    assert_eq!(bkpt.to_string(), "bkpt #0x1234");

    // `bkpt` is only defined with cond = 0b1110.
    assert_eq!(decode(0x1000, 0x1121_2374), Err(DecodeError::Unpredictable));
}

#[test]
fn rejected_encodings() {
    // Coprocessor data operations and transfers that are not VFP.
    assert_eq!(decode(0x1000, 0xEE00_0E10), Err(DecodeError::Unsupported));
    assert_eq!(decode(0x1000, 0xED91_0E00), Err(DecodeError::Unsupported));
    // The media multiply-accumulate forms are modelled; an unallocated form of the
    // group is still rejected.
    assert_eq!(text(0xE700_0211), "smlad r0, r1, r2, r0");
    assert_eq!(
        text_at(0x1000, media_mac(0x75, 0, false, 0, 15, 1, 2)),
        "smmul r0, r2, r1"
    );
    // `Rm` may not be `pc`.
    assert_eq!(
        decode(0x1000, media_mac(0x75, 0, false, 0, 15, 15, 2)),
        Err(DecodeError::Unpredictable)
    );
    assert_eq!(
        decode(0x1000, media_mac(0x70, 2, false, 0, 3, 1, 2)),
        Err(DecodeError::Unallocated)
    );
    // `swp`/`swpb` are architecturally deprecated.
    assert_eq!(decode(0x1000, 0xE102_0092), Err(DecodeError::Deprecated));
    assert_eq!(decode(0x1000, 0xE142_0092), Err(DecodeError::Deprecated));
    // The `msr` immediate space with Rn != 0b1111 is unallocated.
    assert_eq!(decode(0x1000, 0xE320_0001), Err(DecodeError::Unallocated));
    assert_eq!(decode(0x1000, 0xE360_0001), Err(DecodeError::Unallocated));
    // `tst` with a non-zero Rd (SBZ) is unpredictable.
    assert_eq!(decode(0x1000, 0xE310_1001), Err(DecodeError::Unpredictable));
    // Rn is SBZ in `mov`/`mvn`.
    assert_eq!(decode(0x1000, 0xE1A3_0001), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE3AC_665D), Err(DecodeError::Unpredictable));
    assert!(decode(0x1000, 0xE3A0_665D).is_ok());
    // Block transfers: empty list, PC as base, writeback over the base, store of PC and
    // the S bit.
    assert_eq!(decode(0x1000, 0xE890_0000), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE89F_0001), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE8B0_0003), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE880_8001), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE9D0_0001), Err(DecodeError::Unsupported));
    // Unprivileged transfers (`strt`/`ldrt`, P = 0 and W = 1) are outside the slice.
    assert_eq!(decode(0x1000, 0xE4A1_0004), Err(DecodeError::Unsupported));
    assert_eq!(decode(0x1000, 0xE4B1_0004), Err(DecodeError::Unsupported));
    // PC as an index register, byte load into PC and writeback over the base or the
    // destination are unpredictable.
    assert_eq!(decode(0x1000, 0xE791_000F), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE5D1_F000), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE490_0004), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xE5B0_0004), Err(DecodeError::Unpredictable));
}

#[test]
fn opcode_table_is_consistent() {
    let mut names: Vec<&str> = Opcode::ALL.iter().map(|opcode| opcode.name()).collect();
    let count = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), count, "duplicate mnemonic");
    assert!(
        names.iter().all(|name| !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())),
        "{names:?}"
    );
    assert_eq!(
        Opcode::ALL
            .iter()
            .filter(|opcode| opcode.is_data_processing())
            .count(),
        18
    );
    // Every opcode the parallel table can produce is also in `ALL`.
    let mut parallel = Vec::new();
    for group in 0..32 {
        for selector in 0..16 {
            if let Some(opcode) = Opcode::parallel(group, selector) {
                parallel.push(opcode);
            }
        }
    }
    assert_eq!(parallel.len(), 36);
    for opcode in parallel {
        assert!(Opcode::ALL.contains(&opcode), "{opcode:?} is missing");
    }
}

#[test]
fn random_words_decode_or_reject_without_panicking() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut decoded = 0u32;
    let mut rejected = [0u32; 5];

    for index in 0..200_000u32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let word = (state >> 32) as u32;
        let pc = 0x0080_0000u32.wrapping_add(index.wrapping_mul(4));

        match decode(pc, word) {
            Ok(insn) => {
                decoded += 1;
                assert_eq!(insn.word, word);
                assert_eq!(insn.pc, pc);
                assert_eq!(insn.next(), pc.wrapping_add(4));
                assert!(!insn.mnemonic().is_empty());
                let rendered = insn.to_string();
                assert!(rendered.len() > 2);
                for operand in &insn.operands {
                    if let Operand::Target(target) = operand {
                        assert_eq!(
                            target.addr & 1,
                            0,
                            "odd branch target for {word:#010x}: {rendered}"
                        );
                    }
                }
            }
            Err(error) => {
                let slot = match error {
                    DecodeError::Unsupported => 0,
                    DecodeError::Unallocated => 1,
                    DecodeError::Unpredictable => 2,
                    DecodeError::Deprecated => 3,
                    DecodeError::Privileged => 4,
                };
                rejected[slot] += 1;
            }
        }
    }

    assert_eq!(rejected.iter().sum::<u32>() + decoded, 200_000);
    assert!(decoded > 10_000, "only {decoded} of 200000 words decoded");
    assert!(rejected.iter().all(|count| *count > 0), "{rejected:?}");
}

// --- VFP and Advanced SIMD ----------------------------------------------------------
//
// The expected text of every word below is the assembly syntax of the ARM ARM (A8.8,
// checked against an independent disassembler on a generated grid). The register numbers
// are chosen so the three field layouts — `D:Vd` for `d`, `Vd:D` for `s` and `Vm:M` for
// a source — are each exercised with a value that tells them apart.

#[test]
fn vfp_load_store() {
    assert_eq!(text(0xED90_0A00), "vldr s0, [r0]");
    assert_eq!(text(0xED90_0B00), "vldr d0, [r0]");
    assert_eq!(text(0xED91_0A04), "vldr s0, [r1, #0x10]");
    assert_eq!(text(0xED91_0B08), "vldr d0, [r1, #0x20]");
    assert_eq!(text(0xED1F_0B04), "vldr d0, [pc, #-0x10]");
    assert_eq!(text(0xED8D_0A10), "vstr s0, [sp, #0x40]");
    assert_eq!(text(0xED8D_0B10), "vstr d0, [sp, #0x40]");
    assert_eq!(text(0xED9D_0B04), "vldr d0, [sp, #0x10]");

    // The offset is `imm8 * 4` and the size follows the register kind.
    assert_eq!(mem(0xED91_0A04).size, MemSize::Word);
    assert_eq!(mem(0xED91_0B08).size, MemSize::Double);
    assert_eq!(mem(0xED91_0B08).offset, Offset::Imm(0x20));
    assert_eq!(decode_at(0x1000, 0xED91_0B08).opcode, Opcode::Vldr);

    // A different coprocessor number is not VFP.
    assert_eq!(decode(0x1000, 0xED91_0E08), Err(DecodeError::Unsupported));
}

#[test]
fn vfp_block_transfers() {
    // `vpush`/`vpop` are the `sp` forms of `vstmdb`/`vldmia`. `imm8` counts 32-bit
    // registers, so eight double registers encode `imm8 = 16`.
    assert_eq!(text(0xED2D_0B10), "vpush {d0-d7}");
    assert_eq!(text(0xED2D_8B10), "vpush {d8-d15}");
    assert_eq!(text(0xECBD_8B10), "vpop {d8-d15}");
    assert_eq!(decode_at(0x1000, 0xED2D_0B10).opcode, Opcode::Vpush);
    assert_eq!(decode_at(0x1000, 0xECBD_8B10).opcode, Opcode::Vpop);
    assert_eq!(text(0xED2D_0A10), "vpush {s0-s15}");

    assert_eq!(text(0xEC90_0B02), "vldmia r0, {d0}");
    assert_eq!(text(0xECB0_0B02), "vldmia r0!, {d0}");
    assert_eq!(text(0xECA0_0B04), "vstmia r0!, {d0, d1}");
    assert_eq!(text(0xEC80_0B02), "vstmia r0, {d0}");

    let block = match decode_at(0x1000, 0xED2D_8B10).operands.as_slice() {
        [Operand::VBlock(block)] => *block,
        other => panic!("{other:?}"),
    };
    assert_eq!(block.first, DReg::new(8).unwrap());
    assert_eq!(block.words, 16);
    assert!(block.double);
    assert_eq!(block.mode, BlockMode::DecrementBefore);
    assert_eq!(block.bytes(), 64);

    // An odd 32-bit register count cannot be a doubleword list, and the zero form is
    // unpredictable on a VFPv3-D32 implementation.
    assert_eq!(decode(0x1000, 0xED2D_0B01), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xED2D_0B00), Err(DecodeError::Unpredictable));
    // A list that runs past d31 is rejected.
    assert_eq!(decode(0x1000, 0xED6D_FB04), Err(DecodeError::Unpredictable));
}

#[test]
fn vfp_data_processing() {
    // Every arithmetic form, single and double. The name comes from opc1 and bit 6.
    for (word, expected) in [
        (0xEE00_0A00u32, "vmla.f32 s0, s0, s0"),
        (0xEE00_0A40, "vmls.f32 s0, s0, s0"),
        (0xEE10_0A00, "vnmls.f32 s0, s0, s0"),
        (0xEE10_0A40, "vnmla.f32 s0, s0, s0"),
        (0xEE20_0A00, "vmul.f32 s0, s0, s0"),
        (0xEE20_0A40, "vnmul.f32 s0, s0, s0"),
        (0xEE30_0A00, "vadd.f32 s0, s0, s0"),
        (0xEE30_0A40, "vsub.f32 s0, s0, s0"),
        (0xEE80_0A00, "vdiv.f32 s0, s0, s0"),
        (0xEE90_0A00, "vfnms.f32 s0, s0, s0"),
        (0xEE90_0A40, "vfnma.f32 s0, s0, s0"),
        (0xEEA0_0A00, "vfma.f32 s0, s0, s0"),
        (0xEEA0_0A40, "vfms.f32 s0, s0, s0"),
        (0xEE00_0B00, "vmla.f64 d0, d0, d0"),
        (0xEE30_0B00, "vadd.f64 d0, d0, d0"),
        (0xEE80_0B00, "vdiv.f64 d0, d0, d0"),
    ] {
        assert_eq!(text(word), expected);
    }
    assert_eq!(decode_at(0x1000, 0xEE80_0A00).opcode, Opcode::Vdiv);
    // `vdiv` has no second form.
    assert_eq!(decode(0x1000, 0xEE80_0A40), Err(DecodeError::Unallocated));

    // The register fields: `s(2n)` is `Vd:D` and a double is `D:Vd`.
    assert_eq!(text(0xEE00_1A00), "vmla.f32 s2, s0, s0");
    assert_eq!(text(0xEE49_0AA9), "vmla.f32 s1, s19, s19");
    assert_eq!(text(0xEE40_0B00), "vmla.f64 d16, d0, d0");
    assert_eq!(text(0xEE41_0B00), "vmla.f64 d16, d1, d0");
    assert_eq!(text(0xEE4A_0BA1), "vmla.f64 d16, d26, d17");

    let vmla = decode_at(0x1000, 0xEE4A_0BA1);
    assert_eq!(vmla.operands[0], Operand::D(DReg::new(16).unwrap()));
    assert_eq!(vmla.operands[1], Operand::D(DReg::new(26).unwrap()));
    assert_eq!(vmla.operands[2], Operand::D(DReg::new(17).unwrap()));
    assert_eq!(vmla.shape, None);
}

#[test]
fn vfp_move_compare_and_convert() {
    // `vmov` immediate: the eight encoded bits expand to a float through VFPExpandImm.
    assert_eq!(text(0xEEB0_0A00), "vmov.f32 s0, #2");
    assert_eq!(text(0xEEB0_0B00), "vmov.f64 d0, #2");
    // The register moves print the precision; the core transfers do not.
    assert_eq!(text(0xEEB0_0A40), "vmov.f32 s0, s0");
    assert_eq!(text(0xEEB0_0B40), "vmov.f64 d0, d0");
    assert_eq!(text(0xEE00_0A10), "vmov s0, r0");
    assert_eq!(text(0xEE10_0A10), "vmov r0, s0");
    assert_eq!(text(0xEC41_0B30), "vmov d16, r0, r1");
    assert_eq!(text(0xEC51_0B30), "vmov r0, r1, d16");
    // The two-single transfer names two consecutive `s` registers and is not modelled.
    assert_eq!(decode(0x1000, 0xEC40_0A30), Err(DecodeError::Unsupported));

    // Unary operations, comparisons and the two conversions between precisions.
    assert_eq!(text(0xEEB0_0AE0), "vabs.f32 s0, s1");
    assert_eq!(text(0xEEB1_0A41), "vneg.f32 s0, s2");
    assert_eq!(text(0xEEB1_0AC1), "vsqrt.f32 s0, s2");
    assert_eq!(text(0xEEB4_0A41), "vcmp.f32 s0, s2");
    assert_eq!(text(0xEEB4_0AC1), "vcmpe.f32 s0, s2");
    assert_eq!(text(0xEEB5_0A40), "vcmp.f32 s0, #0");
    assert_eq!(text(0xEEB5_0AC0), "vcmpe.f32 s0, #0");
    assert_eq!(text(0xEEB6_0A40), "vrintr.f32 s0, s0");
    assert_eq!(text(0xEEB6_0AC0), "vrintz.f32 s0, s0");
    assert_eq!(text(0xEEB7_0A40), "vrintx.f32 s0, s0");
    assert_eq!(text(0xEEB7_0AC0), "vcvt.f64.f32 d0, s0");
    assert_eq!(text(0xEEB7_0BC0), "vcvt.f32.f64 s0, d0");

    // Integer conversions name both types in the mnemonic, and `vcvtr` rounds to
    // nearest even regardless of FPSCR.
    assert_eq!(text(0xEEB8_0A40), "vcvt.f32.u32 s0, s0");
    assert_eq!(text(0xEEB8_0AC0), "vcvt.f32.s32 s0, s0");
    assert_eq!(text(0xEEB8_0B40), "vcvt.f64.u32 d0, s0");
    assert_eq!(text(0xEEBC_0A40), "vcvtr.u32.f32 s0, s0");
    assert_eq!(text(0xEEBC_0AC0), "vcvt.u32.f32 s0, s0");
    assert_eq!(text(0xEEBC_0BC0), "vcvt.u32.f64 s0, d0");
    assert_eq!(decode_at(0x1000, 0xEEBC_0A40).opcode, Opcode::VcvtR);
    assert_eq!(decode_at(0x1000, 0xEEBC_0AC0).opcode, Opcode::Vcvt);
    assert_eq!(
        decode_at(0x1000, 0xEEB8_0AC0).operands[2],
        Operand::Convert(Convert::F32FromS32)
    );

    // The fixed-point conversions are not modelled, and a comparison against zero
    // cannot name a register.
    assert_eq!(decode(0x1000, 0xEEBE_0A40), Err(DecodeError::Unsupported));
    assert_eq!(decode(0x1000, 0xEEB5_0A41), Err(DecodeError::Unpredictable));
}

#[test]
fn vfp_system_registers_and_elements() {
    assert_eq!(text(0xEEF1_0A10), "vmrs r0, fpscr");
    assert_eq!(text(0xEEF1_FA10), "vmrs apsr_nzcv, fpscr");
    assert_eq!(text(0xEEE1_0A10), "vmsr fpscr, r0");
    assert_eq!(
        decode_at(0x1000, 0xEEF1_FA10).operands[0],
        Operand::VfpDest(VfpDest::Apsr)
    );

    // `vmov.32 d0[0], r0` and the reading forms; the element index counts lanes of the
    // named size.
    assert_eq!(text(0xEE00_0B10), "vmov.32 d0[0], r0");
    assert_eq!(text(0xEE20_0B10), "vmov.32 d0[1], r0");
    assert_eq!(text(0xEE10_0B10), "vmov.32 r0, d0[0]");
    assert_eq!(text(0xEE00_0B30), "vmov.16 d0[0], r0");
    assert_eq!(text(0xEE20_0B70), "vmov.16 d0[3], r0");
    assert_eq!(text(0xEE40_0B10), "vmov.8 d0[0], r0");
    assert_eq!(text(0xEE60_0B70), "vmov.8 d0[7], r0");
    assert_eq!(text(0xEE10_0B30), "vmov.s16 r0, d0[0]");
    assert_eq!(text(0xEE90_0B30), "vmov.u16 r0, d0[0]");
    assert_eq!(text(0xEE50_0B10), "vmov.s8 r0, d0[0]");
    assert_eq!(text(0xEED0_0B10), "vmov.u8 r0, d0[0]");
    // The extension bit of the register is bit 7 in these forms.
    assert_eq!(text(0xEE00_0B90), "vmov.32 d16[0], r0");

    // `vdup` from a core register, `d` and `q` forms.
    assert_eq!(text(0xEE80_0B10), "vdup.32 d0, r0");
    assert_eq!(text(0xEEA0_0B10), "vdup.32 q0, r0");
    assert_eq!(text(0xEE80_0B30), "vdup.16 d0, r0");
    assert_eq!(text(0xEEC0_0B10), "vdup.8 d0, r0");

    // An element move cannot write or read `pc`.
    assert_eq!(decode(0x1000, 0xEE00_FB10), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xEE10_FB10), Err(DecodeError::Unpredictable));
}

#[test]
fn cp15_thread_pointers() {
    // The SDK reads the thread pointer through `mrc p15, 0, rt, c13, c0, 3`.
    assert_eq!(text(0xEE1D_0F70), "mrc r0, tpidruro");
    assert_eq!(decode_at(0x1000, 0xEE1D_0F70).opcode, Opcode::Mrc);
    assert_eq!(text(0xEE0D_0F50), "mcr r0, tpidrurw");
    assert_eq!(decode_at(0x1000, 0xEE0D_0F50).opcode, Opcode::Mcr);
    // Any other cp15 register is outside the slice.
    assert_eq!(decode(0x1000, 0xEE1D_0F30), Err(DecodeError::Unsupported));
}

#[test]
fn neon_structure_transfers() {
    assert_eq!(text(0xF420_070F), "vld1.8 {d0}, [r0]");
    assert_eq!(text(0xF420_078F), "vld1.32 {d0}, [r0]");
    assert_eq!(text(0xF420_0A8F), "vld1.32 {d0, d1}, [r0]");
    assert_eq!(text(0xF420_020F), "vld1.8 {d0, d1, d2, d3}, [r0]");
    assert_eq!(text(0xF400_020F), "vst1.8 {d0, d1, d2, d3}, [r0]");
    // The writeback forms: `!` advances by the transfer size, a register is a
    // post-index.
    assert_eq!(text(0xF420_078D), "vld1.32 {d0}, [r0]!");
    assert_eq!(text(0xF420_0781), "vld1.32 {d0}, [r0], r1");
    // The interleaved layouts.
    assert_eq!(text(0xF420_080F), "vld2.8 {d0, d1}, [r0]");
    assert_eq!(text(0xF400_080F), "vst2.8 {d0, d1}, [r0]");
    assert_eq!(text(0xF420_090F), "vld2.8 {d0, d2}, [r0]");
    assert_eq!(text(0xF420_030F), "vld2.8 {d0, d1, d2, d3}, [r0]");
    assert_eq!(text(0xF420_040F), "vld3.8 {d0, d1, d2}, [r0]");
    assert_eq!(text(0xF420_000F), "vld4.8 {d0, d1, d2, d3}, [r0]");
    assert_eq!(text(0xF420_079F), "vld1.32 {d0}, [r0]");
    assert_eq!(text(0xF420_07CF), "vld1.64 {d0}, [r0]");

    let mem = match decode_at(0x1000, 0xF420_090F).operands.as_slice() {
        [Operand::NeonMem(mem)] => *mem,
        other => panic!("{other:?}"),
    };
    assert_eq!(mem.first, DReg::new(0).unwrap());
    assert_eq!(mem.count, 2);
    assert_eq!(mem.stride, 2);
    assert_eq!(mem.interleave, 2);
    assert_eq!(mem.esize, 8);
    assert_eq!(mem.update, PostUpdate::None);
    // The alignment field is a promise about the address, not an encoding of the
    // transfer; `vld1.32` with align `01` is 8 bytes aligned.
    let aligned = match decode_at(0x1000, 0xF420_079F).operands.as_slice() {
        [Operand::NeonMem(mem)] => *mem,
        other => panic!("{other:?}"),
    };
    assert_eq!(aligned.align, Some(8));

    // The single-lane forms and the block transfer are outside the slice, and an
    // unallocated type is rejected.
    assert_eq!(decode(0x1000, 0xF4A0_000F), Err(DecodeError::Unsupported));
    assert_eq!(decode(0x1000, 0xF420_0F0F), Err(DecodeError::Unsupported));
}

#[test]
fn neon_data_processing() {
    // The three-register group: the logic operations use the size field as their own
    // selector and print no element suffix.
    assert_eq!(text(0xF201_0112), "vand d0, d1, d2");
    assert_eq!(text(0xF211_0112), "vbic d0, d1, d2");
    assert_eq!(text(0xF221_0112), "vorr d0, d1, d2");
    assert_eq!(text(0xF231_0112), "vorn d0, d1, d2");
    assert_eq!(text(0xF301_0112), "veor d0, d1, d2");
    assert_eq!(text(0xF311_0112), "vbsl d0, d1, d2");

    // Integer arithmetic, with the element suffix.
    assert_eq!(text(0xF201_0802), "vadd.i8 d0, d1, d2");
    assert_eq!(text(0xF211_0802), "vadd.i16 d0, d1, d2");
    assert_eq!(text(0xF221_0802), "vadd.i32 d0, d1, d2");
    assert_eq!(text(0xF231_0802), "vadd.i64 d0, d1, d2");
    assert_eq!(text(0xF301_0802), "vsub.i8 d0, d1, d2");
    assert_eq!(text(0xF221_0912), "vmul.i32 d0, d1, d2");
    assert_eq!(text(0xF201_0912), "vmul.i8 d0, d1, d2");

    // Comparisons, tests and the extrema, signed and unsigned.
    assert_eq!(text(0xF201_0312), "vcge.s8 d0, d1, d2");
    assert_eq!(text(0xF301_0312), "vcge.u8 d0, d1, d2");
    assert_eq!(text(0xF201_0302), "vcgt.s8 d0, d1, d2");
    assert_eq!(text(0xF301_0302), "vcgt.u8 d0, d1, d2");
    assert_eq!(text(0xF201_0602), "vmax.s8 d0, d1, d2");
    assert_eq!(text(0xF301_0602), "vmax.u8 d0, d1, d2");
    assert_eq!(text(0xF201_0612), "vmin.s8 d0, d1, d2");
    assert_eq!(text(0xF301_0612), "vmin.u8 d0, d1, d2");
    assert_eq!(text(0xF201_0812), "vtst.8 d0, d1, d2");
    assert_eq!(text(0xF301_0812), "vceq.i8 d0, d1, d2");

    // The float forms live at their own `(size, opc1)` positions.
    assert_eq!(text(0xF201_0D02), "vadd.f32 d0, d1, d2");
    assert_eq!(text(0xF221_0D02), "vsub.f32 d0, d1, d2");
    assert_eq!(text(0xF301_0D12), "vmul.f32 d0, d1, d2");
    assert_eq!(text(0xF201_0D12), "vmla.f32 d0, d1, d2");
    assert_eq!(text(0xF221_0D12), "vmls.f32 d0, d1, d2");
    assert_eq!(text(0xF201_0C12), "vfma.f32 d0, d1, d2");
    assert_eq!(text(0xF221_0C12), "vfms.f32 d0, d1, d2");
    assert_eq!(text(0xF201_0E02), "vceq.f32 d0, d1, d2");
    assert_eq!(text(0xF301_0E02), "vcge.f32 d0, d1, d2");
    assert_eq!(text(0xF201_0F02), "vmax.f32 d0, d1, d2");
    assert_eq!(text(0xF221_0F02), "vmin.f32 d0, d1, d2");

    // A quadword names an even `d` register and sets the shape.
    let q = decode_at(0x1000, 0xF242_0842);
    assert_eq!(q.to_string(), "vadd.i8 q8, q1, q1");
    assert_eq!(q.shape, Some(VecShape { esize: 8, q: true }));
    assert_eq!(q.operands[0], Operand::D(DReg::new(16).unwrap()));

    // The two-register group.
    assert_eq!(text(0xF3B0_0042), "vrev64.8 q0, q1");
    assert_eq!(text(0xF3B4_0082), "vrev32.16 d0, d2");
    assert_eq!(text(0xF3B4_0102), "vrev16.16 d0, d2");
    assert_eq!(text(0xF3B0_0582), "vmvn d0, d2");
    assert_eq!(text(0xF3B1_0300), "vabs.s8 d0, d0");
    assert_eq!(text(0xF3B1_0380), "vneg.s8 d0, d0");
    assert_eq!(text(0xF3B9_0700), "vabs.f32 d0, d0");
    assert_eq!(text(0xF3B9_0780), "vneg.f32 d0, d0");

    // `vext` shifts the concatenation by whole bytes.
    assert_eq!(text(0xF2B1_0302), "vext.8 d0, d1, d2, #0x3");
    assert_eq!(text(0xF2B1_0382), "vext.8 d0, d17, d2, #0x3");

    // The modified immediate: the lane size comes from `cmode` and the value repeats in
    // every lane.
    assert_eq!(text(0xF280_101F), "vmov.i32 d1, #0xf");
    assert_eq!(text(0xF280_121F), "vmov.i32 d1, #0xf00");
    assert_eq!(text(0xF280_181F), "vmov.i16 d1, #0xf");
    assert_eq!(text(0xF280_1E1F), "vmov.i8 d1, #0xf");
    assert_eq!(text(0xF280_1C1F), "vmov.i32 d1, #0xfff");
    assert_eq!(text(0xF280_111F), "vorr.i32 d1, #0xf");
    assert_eq!(text(0xF280_103F), "vmvn.i32 d1, #0xf");
    assert_eq!(text(0xF280_113F), "vbic.i32 d1, #0xf");
    assert_eq!(
        decode_at(0x1000, 0xF280_1E1F).operands[1],
        Operand::Imm64(0x0f0f_0f0f_0f0f_0f0f)
    );

    // The table lookups, the lane duplicate and the float immediate are not modelled.
    assert_eq!(decode(0x1000, 0xF3B0_0802), Err(DecodeError::Unsupported));
    assert_eq!(decode(0x1000, 0xF3B1_0C02), Err(DecodeError::Unsupported));
    assert_eq!(decode(0x1000, 0xF280_1F1F), Err(DecodeError::Unsupported));
}

#[test]
fn unsupported_families_are_named() {
    assert_eq!(
        describe_unsupported(0xEEBE_0A40),
        "vcvt between floating point and fixed point is outside the A32 slice"
    );
    assert_eq!(
        describe_unsupported(0xEE00_0E10),
        "coprocessor access other than VFP/NEON (MCR/MRC/CDP) is outside the A32 slice"
    );
    // The two-register group names what it is missing.
    assert_eq!(
        describe_unsupported(0xF3B0_0802),
        "an Advanced SIMD two-register encoding this slice does not model (the table lookups, `vswp`, `vdup` from a lane or a shift by immediate)"
    );
    // The three-register space keeps the generic name.
    assert_eq!(
        describe_unsupported(0xF213_0112),
        "an Advanced SIMD data-processing encoding this slice does not model"
    );
    assert_eq!(
        describe_unsupported(0xF420_0F0F),
        "an Advanced SIMD structure transfer this slice does not model"
    );
    assert_eq!(
        describe_unsupported(0xF4A0_000F),
        "the single-lane Advanced SIMD transfers (`vld1.single` and friends) are outside the A32 slice"
    );
}

/// `udiv`/`sdiv`: `cond 0111 00x1 Rd 1111 Rm 0001 Rn`.
fn divide(unsigned: bool, rd: u32, rn: u32, rm: u32) -> u32 {
    0xE710_F010 | (u32::from(unsigned) << 21) | (rd << 16) | (rm << 8) | rn
}

/// One of the media dual multiplies. `group` is bits 27:20 (`0b0111_0000`,
/// `0b0111_0100` or `0b0111_0101`), `form` bits 7:6 and `modifier` bit 5.
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

#[test]
fn integer_divide_decodes() {
    // The word MK8D stops on.
    assert_eq!(text(0xE735_F710), "udiv r5, r0, r7");
    assert_eq!(text_at(0x1000, divide(true, 0, 1, 2)), "udiv r0, r1, r2");
    assert_eq!(text_at(0x1000, divide(false, 3, 4, 5)), "sdiv r3, r4, r5");
    assert_eq!(decode_at(0x1000, 0xE735_F710).opcode, Opcode::Udiv);
    assert_eq!(
        decode_at(0x1000, divide(false, 0, 0, 0)).opcode,
        Opcode::Sdiv
    );
    // Bits 15:12 are SBZ and bits 7:4 are fixed at 0b0001.
    assert_eq!(
        decode(0x1000, divide(false, 0, 0, 7) & !0x0000_F000),
        Err(DecodeError::Unpredictable)
    );
    assert_eq!(
        decode(0x1000, divide(false, 0, 0, 7) | 0x20),
        Err(DecodeError::Unpredictable)
    );
    // `pc` is not a legal operand of either form.
    assert_eq!(
        decode(0x1000, divide(true, 15, 0, 7)),
        Err(DecodeError::Unpredictable)
    );
}

#[test]
fn media_dual_multiplies_decode() {
    // `smuad`/`smusd` name no accumulator, so bits 15:12 are 1111.
    assert_eq!(
        text_at(0x1000, media_mac(0x70, 0, false, 0, 15, 1, 2)),
        "smuad r0, r2, r1"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x70, 0, true, 0, 15, 1, 2)),
        "smuadx r0, r2, r1"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x70, 1, false, 0, 15, 1, 2)),
        "smusd r0, r2, r1"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x70, 1, true, 0, 15, 1, 2)),
        "smusdx r0, r2, r1"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x70, 0, false, 0, 3, 1, 2)),
        "smlad r0, r2, r1, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x70, 0, true, 0, 3, 1, 2)),
        "smladx r0, r2, r1, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x70, 1, false, 0, 3, 1, 2)),
        "smlsd r0, r2, r1, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x70, 1, true, 0, 3, 1, 2)),
        "smlsdx r0, r2, r1, r3"
    );
    // Bits 7:6 = 1x are not allocated in this group.
    assert_eq!(
        decode(0x1000, media_mac(0x70, 2, false, 0, 3, 1, 2)),
        Err(DecodeError::Unallocated)
    );

    // The 64-bit accumulators name `RdHi:RdLo` in the two register fields.
    assert_eq!(
        text_at(0x1000, media_mac(0x74, 0, false, 1, 2, 3, 4)),
        "smlald r1, r2, r4, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x74, 0, true, 1, 2, 3, 4)),
        "smlaldx r1, r2, r4, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x74, 1, false, 1, 2, 3, 4)),
        "smlsld r1, r2, r4, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x74, 1, true, 1, 2, 3, 4)),
        "smlsldx r1, r2, r4, r3"
    );
    assert_eq!(
        decode(0x1000, media_mac(0x74, 0, false, 1, 1, 3, 4)),
        Err(DecodeError::Unpredictable),
        "the two halves of the accumulator must differ"
    );

    // The top-word multiplies: bits 7:6 = 00 with `Ra = 1111` is `smmul`, otherwise
    // `smmla`; bits 7:6 = 11 is `smmls`.
    assert_eq!(
        text_at(0x1000, media_mac(0x75, 0, false, 0, 15, 1, 2)),
        "smmul r0, r2, r1"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x75, 0, true, 0, 15, 1, 2)),
        "smmulr r0, r2, r1"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x75, 0, false, 0, 3, 1, 2)),
        "smmla r0, r2, r1, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x75, 0, true, 0, 3, 1, 2)),
        "smmlar r0, r2, r1, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x75, 3, false, 0, 3, 1, 2)),
        "smmls r0, r2, r1, r3"
    );
    assert_eq!(
        text_at(0x1000, media_mac(0x75, 3, true, 0, 3, 1, 2)),
        "smmlsr r0, r2, r1, r3"
    );
    assert_eq!(
        decode(0x1000, media_mac(0x75, 1, false, 0, 3, 1, 2)),
        Err(DecodeError::Unallocated)
    );
}

#[test]
fn barriers_hints_and_prefetch_decode() {
    assert_eq!(text(0xF57F_F05B), "dmb #0xb");
    assert_eq!(text(0xF57F_F04F), "dsb #0xf");
    assert_eq!(text(0xF57F_F06F), "isb #0xf");
    assert_eq!(text(0xF57F_F070), "clrex #0x0");
    assert_eq!(text(0xF57F_F030), "nop #0x0");
    assert_eq!(text(0xF57F_F031), "yield #0x1");
    assert_eq!(text(0xF57F_F032), "wfe #0x2");
    assert_eq!(text(0xF57F_F033), "wfi #0x3");
    assert_eq!(text(0xF57F_F034), "sev #0x4");
    assert_eq!(text(0xF57F_F035), "sevl #0x5");
    // `pld [rn, #imm12]`, including the `pc`-relative literal form.
    assert_eq!(text(0xF5D0_F010), "pld [r0, #0x10]");
    assert_eq!(text(0xF550_F010), "pld [r0, #-0x10]");
    assert_eq!(text(0xF55F_F004), "pld [pc, #-0x4]");
    // `clrex` has a fixed zero option, and the hint numbers stop at `sevl`.
    assert_eq!(decode(0x1000, 0xF57F_F071), Err(DecodeError::Unallocated));
    assert_eq!(decode(0x1000, 0xF57F_F036), Err(DecodeError::Unallocated));
    // `cps`, `setend` and `pli` share the space and are not modelled.
    assert_eq!(decode(0x1000, 0xF102_0000), Err(DecodeError::Unsupported));
    assert_eq!(decode(0x1000, 0xF51F_F000), Err(DecodeError::Unsupported));
}

#[test]
fn mrrc_reads_the_counters() {
    // The word MK8D executes to read `CNTPCT` into `r2:r3`.
    assert_eq!(text(0xEC53_2F0E), "mrrc r2, r3, cntpct");
    assert_eq!(text(0xEC51_0F1E), "mrrc r0, r1, cntvct");
    assert_eq!(decode_at(0x1000, 0xEC53_2F0E).opcode, Opcode::Mrrc);
    assert_eq!(
        decode_at(0x1000, 0xEC53_2F0E).operands[0],
        Operand::Regs {
            lo: Reg::R2,
            hi: Reg::R3
        }
    );
    // `Rt` and `Rt2` must differ and must not be `pc`.
    assert_eq!(decode(0x1000, 0xEC52_2F0E), Err(DecodeError::Unpredictable));
    assert_eq!(decode(0x1000, 0xEC5F_2F0E), Err(DecodeError::Unpredictable));
    // Any other `cp15` pair transfer is outside the slice.
    assert_eq!(decode(0x1000, 0xEC53_2F03), Err(DecodeError::Unsupported));
}
