//! Jump-table shapes: the layouts a switch compiles to, the tables that must be rejected, and
//! the no-panic rule for arbitrary bytes.
//!
//! Each program is a real ARM64 image assembled here, with the table in constant data right after
//! the code (a byte-offset table must be close to its cases). The expected case labels come from
//! the assembler, the detector's answer from `jump_table`, and the independent answer from the
//! stepper in `tests/oracle`: a shape passes only when the detector and the stepper agree, so a
//! change to the detector cannot make the test agree with itself.

mod oracle;

use std::collections::BTreeSet;

use kirikae_a64::asm;
use kirikae_loader::{Image, LOAD_BASE};
use kirikae_recomp::discover::{CodeSpace, build_function, find_entries_measured};
use kirikae_recomp::jump::jump_table;
use oracle::{Cpu, Oracle};

/// An image whose `.rodata` starts at `ro` (a contiguous range of `memory`).
fn image(code: &[u32], ro: &[u8], entry: u64, ro_at: u64) -> Image {
    let text = LOAD_BASE + 4 * code.len() as u64;
    assert_eq!(text, ro_at, "constant data follows the code");
    let mut memory = vec![0u8; (ro_at - LOAD_BASE) as usize + ro.len()];
    for (i, word) in code.iter().enumerate() {
        memory[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
    }
    memory[(ro_at - LOAD_BASE) as usize..].copy_from_slice(ro);
    Image {
        base: LOAD_BASE,
        memory,
        text: LOAD_BASE..text,
        ro: ro_at..ro_at + ro.len() as u64,
        data: 0..0,
        bss: 0..0,
        entry,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `ldr x10, [x9, x0, lsl #3]`: an array of absolute 8-byte pointers.
    Absolute,
    /// `ldrsw x10, [x9, x0, lsl #2]` then `add x9, x9, x10`: 4-byte offsets from the table.
    Relative,
    /// `ldrsb w10, [x9, x0]` then `add x9, x9, w10, sxtw`: 1-byte offsets from the table.
    Byte,
}

#[derive(Debug)]
struct Shape {
    kind: Kind,
    cases: usize,
    /// Emit `cmp w0, #cases-1` + `b.hi default`.
    guarded: bool,
    /// Which register the bound check compares, when it is not the index.
    checked: u32,
    /// Overwrite entry `k` of the table with this raw value.
    corrupt: Option<(usize, u64)>,
}

impl Shape {
    fn new(kind: Kind, cases: usize) -> Shape {
        Shape {
            kind,
            cases,
            guarded: true,
            checked: 0,
            corrupt: None,
        }
    }

    fn unbounded(mut self) -> Shape {
        self.guarded = false;
        self
    }

    fn checked(mut self, reg: u32) -> Shape {
        self.checked = reg;
        self
    }

    fn corrupt(mut self, at: usize, value: u64) -> Shape {
        self.corrupt = Some((at, value));
        self
    }
}

struct Program {
    image: Image,
    br: u64,
    entry: u64,
    cases: Vec<u64>,
    default: u64,
    table: u64,
}

/// One dispatch shape. Case `k` is a `mov` + `ret` block, so every case is a distinct address and
/// no case falls through into the next.
fn program(shape: &Shape) -> Program {
    let mut a = asm::Asm::new(LOAD_BASE);
    // Patched after the code is emitted, so the table can sit right after it.
    a.define("table", 0);
    a.label("entry");
    if shape.guarded {
        a.emit(asm::cmp_imm(shape.checked, shape.cases as u32 - 1));
        a.bcond_to(asm::cond::HI, "default");
    }
    a.addr_of(9, "table");
    match shape.kind {
        Kind::Absolute => {
            a.emit(asm::ldst_reg(3, 1, 10, 9, 0, 3, true));
            a.emit(asm::br(10));
        }
        Kind::Relative => {
            a.emit(asm::ldst_reg(2, 2, 10, 9, 0, 3, true));
            a.emit(asm::add_reg(9, 9, 10));
            a.emit(asm::br(9));
        }
        Kind::Byte => {
            a.emit(asm::ldst_reg(0, 2, 10, 9, 0, 3, false));
            a.emit(asm::addsub_ext(true, false, false, 9, 9, 10, 6, 0));
            a.emit(asm::br(9));
        }
    }
    for i in 0..shape.cases {
        a.label(&format!("case{i}"));
        a.emit(asm::movz(true, 0, b'a' as u32 + i as u32, 0));
        a.emit(asm::ret());
    }
    a.label("default");
    a.emit(asm::movz(true, 0, b'?' as u32, 0));
    a.emit(asm::ret());

    let table = a.here();
    a.define("table", table);
    let br = a.label_addr("case0") - 4;
    let cases: Vec<u64> = (0..shape.cases)
        .map(|i| a.label_addr(&format!("case{i}")))
        .collect();
    let default = a.label_addr("default");
    let code = a.finish();

    let width = match shape.kind {
        Kind::Absolute => 8,
        Kind::Relative => 4,
        Kind::Byte => 1,
    };
    let mut ro = vec![0u8; width * (shape.cases + 1)];
    for (i, &addr) in cases.iter().enumerate() {
        let value = match shape.kind {
            Kind::Absolute => addr,
            Kind::Relative => (addr as i64 - table as i64) as i32 as i64 as u64,
            Kind::Byte => (addr as i64 - table as i64) as i8 as i64 as u64,
        };
        let bytes = value.to_le_bytes();
        ro[width * i..width * i + width].copy_from_slice(&bytes[..width]);
    }
    if let Some((at, value)) = shape.corrupt {
        let bytes = value.to_le_bytes();
        ro[width * at..width * at + width].copy_from_slice(&bytes[..width]);
    }
    let image = image(&code, &ro, LOAD_BASE, table);
    Program {
        image,
        br,
        entry: LOAD_BASE,
        cases,
        default,
        table,
    }
}

/// The detector and the stepper must agree: the stepper executes the dispatch with the case
/// index in `x0` and observes where the `br` goes.
fn check(shape: Shape, expected_bounded: bool) {
    let p = program(&shape);
    let space = CodeSpace::new(vec![&p.image]);
    let table = jump_table(&space, p.br).expect("the shape is a table");
    assert_eq!(table.bounded, expected_bounded, "{shape:?}");
    assert_eq!(
        table.targets, p.cases,
        "the detector reads every case once, in order"
    );

    let images = [&p.image];
    let oracle = Oracle::new(&space, &images);
    for (i, &case) in p.cases.iter().enumerate() {
        let cpu = Cpu::new().with(0, i as u64);
        let observed = oracle.targets(p.entry, &cpu, 64);
        assert_eq!(observed, BTreeSet::from([case]), "case {i} of {shape:?}");
    }
    if shape.guarded {
        // Out of range: only the default.
        let cpu = Cpu::new()
            .with(0, shape.cases as u64)
            .with(shape.checked as u8, shape.cases as u64);
        assert_eq!(
            oracle.targets(p.entry, &cpu, 64),
            BTreeSet::from([p.default]),
            "out of range takes the default of {shape:?}"
        );
    }
    let _ = p.table;
}

#[test]
fn absolute_eight_byte_table() {
    check(Shape::new(Kind::Absolute, 4), true);
}

#[test]
fn relative_four_byte_table() {
    check(Shape::new(Kind::Relative, 4), true);
}

#[test]
fn byte_offset_table() {
    check(Shape::new(Kind::Byte, 4), true);
}

#[test]
fn unbounded_absolute_table() {
    check(Shape::new(Kind::Absolute, 5).unbounded(), false);
}

#[test]
fn unbounded_relative_table() {
    check(Shape::new(Kind::Relative, 6).unbounded(), false);
}

#[test]
fn the_bound_may_be_a_register_constant() {
    // `mov w1, #3; cmp w0, w1; b.hi default` is the same bound with the constant in a register.
    let mut a = asm::Asm::new(LOAD_BASE);
    a.define("table", 0);
    a.label("entry");
    a.mov_imm(1, 3);
    a.emit(asm::addsub_reg(false, true, true, 31, 0, 1, 0, 0));
    a.bcond_to(asm::cond::HI, "default");
    a.addr_of(9, "table");
    a.emit(asm::ldst_reg(3, 1, 10, 9, 0, 3, true));
    a.emit(asm::br(10));
    for i in 0..4 {
        a.label(&format!("case{i}"));
        a.emit(asm::movz(true, 0, b'a' as u32 + i, 0));
        a.emit(asm::ret());
    }
    a.label("default");
    a.emit(asm::movz(true, 0, b'?' as u32, 0));
    a.emit(asm::ret());
    let table = a.here();
    a.define("table", table);
    let br = a.label_addr("case0") - 4;
    let cases: Vec<u64> = (0..4).map(|i| a.label_addr(&format!("case{i}"))).collect();
    let default = a.label_addr("default");
    let code = a.finish();
    let mut ro = vec![0u8; 8 * 5];
    for (i, &addr) in cases.iter().enumerate() {
        ro[8 * i..8 * i + 8].copy_from_slice(&addr.to_le_bytes());
    }
    let image = image(&code, &ro, LOAD_BASE, table);
    let space = CodeSpace::new(vec![&image]);
    let table = jump_table(&space, br).expect("register bound");
    assert!(table.bounded);
    assert_eq!(table.targets, cases);

    let images = [&image];
    let oracle = Oracle::new(&space, &images);
    let cpu = Cpu::new().with(0, 1).with(1, 0);
    assert_eq!(
        oracle.targets(LOAD_BASE, &cpu, 64),
        BTreeSet::from([cases[1]])
    );
    let _ = default;
}

#[test]
fn a_table_whose_entries_are_not_code_is_rejected() {
    // The bound says four entries, but entry 2 is text: reading past a real table into
    // neighbouring data must not produce targets.
    let shape = Shape::new(Kind::Absolute, 4).corrupt(2, 0x4141_4141);
    let p = program(&shape);
    let space = CodeSpace::new(vec![&p.image]);
    assert_eq!(jump_table(&space, p.br), None);

    // The stepper shows the computed destination is not code, so nothing is lost.
    let images = [&p.image];
    let oracle = Oracle::new(&space, &images);
    let cpu = Cpu::new().with(0, 2);
    for target in oracle.targets(p.entry, &cpu, 64) {
        if target != p.default {
            assert!(!space.is_code(target), "{target:#x} is not code");
        }
    }
}

#[test]
fn a_short_unguarded_run_is_not_a_table() {
    // Three code entries and then zero padding: below the minimum an unguarded scan needs,
    // because a single computed address that lands in constant data would otherwise pass.
    let shape = Shape::new(Kind::Relative, 3).unbounded();
    let p = program(&shape);
    let space = CodeSpace::new(vec![&p.image]);
    assert_eq!(jump_table(&space, p.br), None);
}

#[test]
fn a_guard_on_another_value_does_not_guess_the_length() {
    // `cmp w1, #3` bounds a tag, not the index in `x0`. The table is real, but its length is
    // unknown here; following it could run into the next code-pointer array in constant data.
    let shape = Shape::new(Kind::Absolute, 4).checked(1);
    let p = program(&shape);
    let space = CodeSpace::new(vec![&p.image]);
    assert_eq!(jump_table(&space, p.br), None);

    // The stepper shows the code does reach the cases, which is why this is a refusal to guess
    // rather than a claim that the shape is wrong.
    let images = [&p.image];
    let oracle = Oracle::new(&space, &images);
    let cpu = Cpu::new().with(0, 2).with(1, 0);
    assert!(oracle.targets(p.entry, &cpu, 64).contains(&p.cases[2]));
}

#[test]
fn a_constant_pointer_jump_is_a_single_target() {
    // `adrp x9, slot; ldr x9, [x9]; br x9` with a fixed code pointer: one edge, not a table.
    let mut a = asm::Asm::new(LOAD_BASE);
    a.define("slot", 0);
    a.label("entry");
    a.addr_of(9, "slot");
    a.emit(asm::ldr(9, 9, 0));
    a.emit(asm::br(9));
    a.label("target");
    a.emit(asm::movz(true, 0, b't' as u32, 0));
    a.emit(asm::ret());
    let slot = a.here();
    a.define("slot", slot);
    let br = a.label_addr("target") - 4;
    let target = a.label_addr("target");
    let code = a.finish();
    let ro = target.to_le_bytes().to_vec();
    let image = image(&code, &ro, LOAD_BASE, slot);
    let space = CodeSpace::new(vec![&image]);
    let table = jump_table(&space, br).expect("single pointer");
    assert_eq!(table.targets, vec![target]);
    assert!(!table.bounded);
}

#[test]
fn case_targets_are_lifted_exactly_once() {
    let shape = Shape::new(Kind::Relative, 6);
    let p = program(&shape);
    let space = CodeSpace::new(vec![&p.image]);
    let (entries, reachable, stats) = find_entries_measured(&space, &[p.entry]);
    assert!(stats.tables >= 1, "{stats:?}");
    let lifted: usize = entries
        .iter()
        .map(|&e| build_function(&space, e, &entries).insn_count())
        .sum();
    assert_eq!(
        lifted as u64, reachable,
        "every reachable word is lifted once"
    );
}

#[test]
fn arbitrary_bytes_never_panic() {
    // Fixed-seed noise: text and constant data both meaningless. The detector must answer for
    // every address without panicking, and any target it does report must be code.
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let code: Vec<u32> = (0..0x400).map(|_| next() as u32).collect();
    let ro: Vec<u8> = (0..0x800).map(|_| next() as u8).collect();
    let image = image(&code, &ro, LOAD_BASE, LOAD_BASE + 0x1000);
    let space = CodeSpace::new(vec![&image]);
    for addr in (LOAD_BASE..LOAD_BASE + 4 * code.len() as u64).step_by(4) {
        if let Some(table) = jump_table(&space, addr) {
            assert!(!table.targets.is_empty());
            for target in &table.targets {
                assert!(space.is_code(*target), "{target:#x} from {addr:#x}");
            }
        }
    }
    // Discovery walks the same bytes and must survive them too.
    let _ = kirikae_recomp::discover(&image);
}

#[test]
fn a_table_of_zeros_is_rejected() {
    // Every entry is readable and none is code.
    let shape = Shape::new(Kind::Absolute, 8).corrupt(0, 0);
    let mut p = program(&shape);
    p.image.memory[(p.table - LOAD_BASE) as usize..][..8 * 9].fill(0);
    let space = CodeSpace::new(vec![&p.image]);
    assert_eq!(jump_table(&space, p.br), None);
    // And an address in the middle of the run is not a dispatch at all.
    assert_eq!(jump_table(&space, p.entry + 8), None);
}
