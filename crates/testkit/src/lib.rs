use kirikae_a64::asm::{self, cond, Asm, FP, LR, SP, XZR};
use kirikae_loader::nro::{Header, Segment, HEADER_SIZE};
use kirikae_loader::LOAD_BASE;

const TEXT: u32 = 0x0000;
const RO: u32 = 0x1000;
const DATA: u32 = 0x2000;
const PAGE: u32 = 0x1000;
const BSS: u64 = LOAD_BASE + (DATA + PAGE) as u64;

pub const HELLO_EXPECTED: &str = "Hello from kirikae!\n\
sum(1..100) = 5050\n\
square(12) = 144\n\
switch: abcd\n\
fib(15) = 610\n";

pub const THREAD_ITERATIONS: u32 = 200;
pub const THREAD_ATOMIC_ITERATIONS: u32 = 20000;
pub const THREADS_EXPECTED: &str = "threads: 40400\n";

const SVC_EXIT_PROCESS: u32 = 0x07;
const SVC_CREATE_THREAD: u32 = 0x08;
const SVC_START_THREAD: u32 = 0x09;
const SVC_EXIT_THREAD: u32 = 0x0a;
const SVC_SLEEP_THREAD: u32 = 0x0b;
const SVC_WAIT_SYNCHRONIZATION: u32 = 0x18;
const SVC_ARBITRATE_LOCK: u32 = 0x1a;
const SVC_ARBITRATE_UNLOCK: u32 = 0x1b;
const SVC_OUTPUT_DEBUG_STRING: u32 = 0x27;
const SVC_WAIT_FOR_ADDRESS: u32 = 0x34;
const SVC_SIGNAL_TO_ADDRESS: u32 = 0x35;

struct Program {
    a: Asm,
    ro: Vec<u8>,
    mod0_at: usize,
}

impl Program {
    fn new() -> Program {
        let mut p = Program { a: Asm::new(LOAD_BASE + HEADER_SIZE as u64), ro: Vec::new(), mod0_at: 0 };
        p.ro.extend_from_slice(&[0u8; 0x1c]);
        p.string("msg_nl", b"\n");
        p.a.define("numbuf_end", BSS + 32);
        p
    }

    fn string(&mut self, label: &str, bytes: &[u8]) -> u64 {
        let addr = LOAD_BASE + RO as u64 + self.ro.len() as u64;
        self.a.define(label, addr);
        self.ro.extend_from_slice(bytes);
        while !self.ro.len().is_multiple_of(8) {
            self.ro.push(0);
        }
        bytes.len() as u64
    }

    fn print(&mut self, label: &str, len: u64) {
        self.a.addr_of(0, label).mov_imm(1, len).emit(asm::svc(SVC_OUTPUT_DEBUG_STRING));
    }

    fn emit_print_u64(&mut self) {
        let a = &mut self.a;
        a.label("print_u64");
        a.emit(asm::ldstp(2, 3, false, FP, LR, SP, -16));
        a.addr_of(1, "numbuf_end");
        a.emit(asm::mov_reg(2, 1)).emit(asm::movz(true, 3, 10, 0));
        a.label("digit");
        a.emit(asm::udiv(4, 0, 3));
        a.emit(asm::msub(5, 4, 3, 0));
        a.emit(asm::add_imm(5, 5, b'0' as u32));
        a.emit(asm::ldst_imm9(0, 0, 5, 2, -1, 3));
        a.emit(asm::mov_reg(0, 4)).cbnz_to(0, "digit");
        a.emit(asm::sub_reg(1, 1, 2)).emit(asm::mov_reg(0, 2)).emit(asm::svc(SVC_OUTPUT_DEBUG_STRING));
        self.print("msg_nl", 1);
        let a = &mut self.a;
        a.emit(asm::ldstp(2, 1, true, FP, LR, SP, 16));
        a.emit(asm::ret());
    }

    fn finish(mut self, bss_size: u32) -> Vec<u8> {
        let mod0_addr = LOAD_BASE + RO as u64 + self.mod0_at as u64;
        let rel = |addr: u64| (addr as i64 - mod0_addr as i64) as i32 as u32;
        let mod0 = [u32::from_le_bytes(*b"MOD0"), 0, rel(BSS), rel(BSS + bss_size as u64), 0, 0, 0];
        for (i, v) in mod0.iter().enumerate() {
            let at = self.mod0_at + 4 * i;
            self.ro[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        assert!(self.ro.len() <= PAGE as usize, "test program outgrew its rodata page");

        let words = self.a.finish();
        let text_size = HEADER_SIZE as u32 + 4 * words.len() as u32;
        assert!(text_size <= RO, "test program outgrew its text page");

        let mut file = vec![0u8; (DATA + PAGE) as usize];
        file[0..4].copy_from_slice(&asm::b(HEADER_SIZE as i64).to_le_bytes());
        for (i, w) in words.iter().enumerate() {
            let off = HEADER_SIZE + 4 * i;
            file[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        file[RO as usize..RO as usize + self.ro.len()].copy_from_slice(&self.ro);
        let header = Header {
            mod0_offset: RO + self.mod0_at as u32,
            version: 0,
            size: file.len() as u32,
            flags: 0,
            text: Segment { offset: TEXT, size: RO },
            ro: Segment { offset: RO, size: PAGE },
            data: Segment { offset: DATA, size: PAGE },
            bss_size,
            module_id: [0; 32],
            dso_handle_offset: 0,
            api_info: Segment::default(),
            dynstr: Segment::default(),
            dynsym: Segment::default(),
        };
        header.write(&mut file);
        file
    }
}

pub fn hello_nro() -> Vec<u8> {
    let mut p = Program::new();
    let hello = p.string("msg_hello", b"Hello from kirikae!\n");
    let sum = p.string("msg_sum", b"sum(1..100) = ");
    let square = p.string("msg_square", b"square(12) = ");
    let switch = p.string("msg_switch", b"switch: ");
    let fib = p.string("msg_fib", b"fib(15) = ");
    let jtable_off = p.ro.len();
    p.string("jtable", &[0u8; 16]);
    p.a.define("chbuf", LOAD_BASE + DATA as u64);

    p.a.label("start");
    p.a.emit(asm::ldstp(2, 3, false, FP, LR, SP, -16));
    p.a.emit(asm::add_imm(FP, SP, 0));
    p.print("msg_hello", hello);

    p.a.emit(asm::movz(true, 0, 0, 0)).emit(asm::movz(true, 1, 100, 0));
    p.a.label("sum_loop");
    p.a.emit(asm::add_reg(0, 0, 1)).emit(asm::subs_imm(1, 1, 1)).bcond_to(cond::NE, "sum_loop");
    p.a.emit(asm::mov_reg(19, 0));
    p.print("msg_sum", sum);
    p.a.emit(asm::mov_reg(0, 19)).bl_to("print_u64");

    p.a.adr_to(9, "square").emit(asm::movz(true, 0, 12, 0)).emit(asm::blr(9));
    p.a.emit(asm::mov_reg(19, 0));
    p.print("msg_square", square);
    p.a.emit(asm::mov_reg(0, 19)).bl_to("print_u64");

    p.print("msg_switch", switch);
    p.a.emit(asm::movz(true, 20, 0, 0));
    p.a.label("switch_loop");
    p.a.emit(asm::mov_reg(0, 20)).bl_to("classify");
    p.a.addr_of(1, "chbuf").emit(asm::ldst_uimm(0, 0, 0, 1, 0));
    p.a.emit(asm::mov_reg(0, 1)).emit(asm::movz(true, 1, 1, 0)).emit(asm::svc(SVC_OUTPUT_DEBUG_STRING));
    p.a.emit(asm::add_imm(20, 20, 1)).emit(asm::cmp_imm(20, 4)).bcond_to(cond::NE, "switch_loop");
    p.print("msg_nl", 1);

    p.print("msg_fib", fib);
    p.a.emit(asm::movz(true, 0, 15, 0)).bl_to("fib").bl_to("print_u64");

    p.a.emit(asm::ldstp(2, 1, true, FP, LR, SP, 16));
    p.a.emit(asm::svc(SVC_EXIT_PROCESS));

    p.emit_print_u64();
    let a = &mut p.a;

    a.label("square");
    a.emit(asm::mul(0, 0, 0)).emit(asm::ret());

    a.label("classify");
    a.emit(asm::cmp_imm(0, 3)).bcond_to(cond::HI, "case_other");
    a.addr_of(9, "jtable");
    a.emit(asm::ldst_reg(2, 2, 10, 9, 0, 3, true));
    a.emit(asm::add_reg(9, 9, 10)).emit(asm::br(9));
    for (i, ch) in b"abcd".iter().enumerate() {
        a.label(&format!("case{i}"));
        a.emit(asm::movz(true, 0, *ch as u32, 0)).emit(asm::ret());
    }
    a.label("case_other");
    a.emit(asm::movz(true, 0, b'?' as u32, 0)).emit(asm::ret());

    a.label("fib");
    a.emit(asm::ldstp(2, 3, false, FP, LR, SP, -32));
    a.emit(asm::str(19, SP, 16));
    a.emit(asm::cmp_imm(0, 2)).bcond_to(cond::LO, "fib_out");
    a.emit(asm::mov_reg(19, 0));
    a.emit(asm::sub_imm(0, 19, 1)).bl_to("fib").emit(asm::str(0, SP, 24));
    a.emit(asm::sub_imm(0, 19, 2)).bl_to("fib");
    a.emit(asm::ldr(1, SP, 24)).emit(asm::add_reg(0, 0, 1));
    a.label("fib_out");
    a.emit(asm::ldr(19, SP, 16));
    a.emit(asm::ldstp(2, 1, true, FP, LR, SP, 32));
    a.emit(asm::ret());

    let jtable = a.label_addr("jtable");
    for i in 0..4 {
        let off = (a.label_addr(&format!("case{i}")) as i64 - jtable as i64) as i32;
        p.ro[jtable_off + 4 * i..jtable_off + 4 * i + 4].copy_from_slice(&off.to_le_bytes());
    }
    p.finish(PAGE)
}

pub fn unguarded_jump_nro() -> Vec<u8> {
    let mut p = Program::new();
    let table_off = p.ro.len();
    p.string("table", &[0u8; 4]);
    p.a.label("start");
    p.a.addr_of(9, "table");
    p.a.emit(asm::ldst_reg(2, 2, 10, 9, 0, 3, true));
    p.a.emit(asm::add_reg(9, 9, 10)).emit(asm::br(9));
    p.a.label("target");
    p.a.emit(asm::ret());
    let rel = (p.a.label_addr("target") as i64 - p.a.label_addr("table") as i64) as i32;
    p.ro[table_off..table_off + 4].copy_from_slice(&rel.to_le_bytes());
    p.finish(PAGE)
}

pub fn threads_nro() -> Vec<u8> {
    threads_program(None)
}

pub fn parallel_threads_nro() -> Vec<u8> {
    threads_program(Some([1, 2]))
}

fn threads_program(cores: Option<[u32; 2]>) -> Vec<u8> {
    const HAS_WAITERS: (u32, u32) = (2, 0);
    const OWNER_MASK: (u32, u32) = (1, 30);
    let data = LOAD_BASE + DATA as u64;
    let mut p = Program::new();
    let label = p.string("msg_threads", b"threads: ");
    p.a.define("mutex", data);
    p.a.define("counter", data + 0x08);
    p.a.define("done", data + 0x10);
    p.a.define("slot0", data + 0x18);
    p.a.define("slot1", data + 0x20);
    p.a.define("hits", data + 0x28);
    p.a.define("stack0_top", BSS + 0x5000);
    p.a.define("stack1_top", BSS + 0x9000);

    let a = &mut p.a;
    a.label("start");
    a.emit(asm::ldstp(2, 3, false, FP, LR, SP, -16));
    for k in 0..2 {
        a.addr_of(1, "worker").addr_of(2, &format!("slot{k}")).addr_of(3, &format!("stack{k}_top"));
        a.emit(asm::movz(true, 4, 44, 0));
        match cores {
            Some(c) => a.emit(asm::movz(true, 5, c[k], 0)),
            None => a.emit(asm::movn(true, 5, 1, 0)),
        };
        a.emit(asm::svc(SVC_CREATE_THREAD));
        a.addr_of(9, &format!("slot{k}")).emit(asm::ldst_uimm(2, 0, 1, 9, 0));
        a.emit(asm::mov_reg(0, 1)).emit(asm::svc(SVC_START_THREAD));
    }
    a.addr_of(23, "done");
    a.label("wait_done");
    a.emit(asm::ldst_uimm(2, 1, 9, 23, 0));
    a.emit(asm::cmp_imm(9, 2)).bcond_to(cond::EQ, "all_done");
    a.emit(asm::mov_reg(0, 23)).emit(asm::movz(true, 1, 0, 0)).emit(asm::movz(true, 2, 2, 0)).emit(asm::movn(true, 3, 0, 0));
    a.emit(asm::svc(SVC_WAIT_FOR_ADDRESS));
    a.b_to("wait_done");
    a.label("all_done");
    for k in 0..2 {
        a.addr_of(1, &format!("slot{k}")).emit(asm::movz(true, 2, 1, 0)).emit(asm::movn(true, 3, 0, 0));
        a.emit(asm::svc(SVC_WAIT_SYNCHRONIZATION));
    }
    p.print("msg_threads", label);
    let a = &mut p.a;
    a.addr_of(9, "counter").emit(asm::ldst_uimm(2, 1, 0, 9, 0));
    a.addr_of(9, "hits").emit(asm::ldst_uimm(2, 1, 9, 9, 0)).emit(asm::add_reg(0, 0, 9)).bl_to("print_u64");
    a.emit(asm::ldstp(2, 1, true, FP, LR, SP, 16));
    a.emit(asm::svc(SVC_EXIT_PROCESS));
    p.emit_print_u64();

    let a = &mut p.a;
    a.label("worker");
    a.emit(asm::ldst_uimm(2, 1, 19, 0, 0));
    a.addr_of(20, "mutex").addr_of(21, "counter").addr_of(23, "done");
    a.addr_of(24, "hits").emit(asm::movz(true, 22, THREAD_ATOMIC_ITERATIONS, 0));
    a.label("hit");
    a.emit(asm::ldaxr(false, 9, 24)).emit(asm::addsub_imm(false, false, false, 9, 9, 1, false));
    a.emit(asm::stlxr(false, 10, 9, 24)).cbnz_to(10, "hit");
    a.emit(asm::subs_imm(22, 22, 1)).bcond_to(cond::NE, "hit");
    a.emit(asm::movz(true, 22, THREAD_ITERATIONS, 0));
    a.label("iteration");
    a.label("lock");
    a.emit(asm::ldaxr(false, 9, 20)).cbnz_to(9, "contended");
    a.emit(asm::stlxr(false, 10, 19, 20)).cbnz_to(10, "lock").b_to("locked");
    a.label("contended");
    a.emit(asm::logical_imm(false, 1, 11, 9, 0, HAS_WAITERS.0, HAS_WAITERS.1));
    a.emit(asm::stlxr(false, 10, 11, 20)).cbnz_to(10, "lock");
    a.emit(asm::logical_imm(false, 0, 0, 11, 0, OWNER_MASK.0, OWNER_MASK.1));
    a.emit(asm::mov_reg(1, 20)).emit(asm::mov_reg(2, 19)).emit(asm::svc(SVC_ARBITRATE_LOCK));
    a.emit(asm::ldst_uimm(2, 1, 9, 20, 0));
    a.emit(asm::logical_imm(false, 0, 9, 9, 0, OWNER_MASK.0, OWNER_MASK.1));
    a.emit(asm::addsub_reg(false, true, true, XZR, 9, 19, 0, 0)).bcond_to(cond::NE, "lock");
    a.label("locked");
    a.emit(asm::ldst_uimm(2, 1, 24, 21, 0));
    a.emit(asm::movz(true, 0, 0, 0)).emit(asm::svc(SVC_SLEEP_THREAD));
    a.emit(asm::add_imm(24, 24, 1)).emit(asm::ldst_uimm(2, 0, 24, 21, 0));
    a.label("unlock");
    a.emit(asm::ldaxr(false, 9, 20));
    a.emit(asm::addsub_reg(false, true, true, XZR, 9, 19, 0, 0)).bcond_to(cond::NE, "unlock_slow");
    a.emit(asm::stlxr(false, 10, XZR, 20)).cbnz_to(10, "unlock").b_to("unlocked");
    a.label("unlock_slow");
    a.emit(asm::mov_reg(0, 20)).emit(asm::svc(SVC_ARBITRATE_UNLOCK));
    a.label("unlocked");
    a.emit(asm::subs_imm(22, 22, 1)).bcond_to(cond::NE, "iteration");
    a.label("bump_done");
    a.emit(asm::ldaxr(false, 9, 23)).emit(asm::addsub_imm(false, false, false, 9, 9, 1, false));
    a.emit(asm::stlxr(false, 10, 9, 23)).cbnz_to(10, "bump_done");
    a.emit(asm::mov_reg(0, 23)).emit(asm::movz(true, 1, 0, 0)).emit(asm::movz(true, 2, 0, 0)).emit(asm::movn(true, 3, 0, 0));
    a.emit(asm::svc(SVC_SIGNAL_TO_ADDRESS));
    a.emit(asm::svc(SVC_EXIT_THREAD));

    p.finish(0x9000)
}

pub fn deadlock_nro() -> Vec<u8> {
    let mut p = Program::new();
    p.a.label("start");
    p.a.emit(asm::movz(true, 2, 0, 0)).emit(asm::movn(true, 3, 0, 0));
    p.a.emit(asm::svc(SVC_WAIT_SYNCHRONIZATION));
    p.a.emit(asm::svc(SVC_EXIT_PROCESS));
    p.finish(PAGE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_parses() {
        let file = hello_nro();
        let image = kirikae_loader::load(&file, LOAD_BASE).unwrap();
        assert_eq!(image.entry, LOAD_BASE);
        assert_eq!(image.word(LOAD_BASE), Some(asm::b(0x80)));
        assert_eq!(image.bss.start, LOAD_BASE + 0x3000);
    }

    #[test]
    fn mutex_masks() {
        use kirikae_a64::bits::decode_bit_masks;
        assert_eq!(decode_bit_masks(0, 0, 2, true, 32).unwrap().0, 0x4000_0000);
        assert_eq!(decode_bit_masks(0, 30, 1, true, 32).unwrap().0, 0xbfff_ffff);
    }
}
