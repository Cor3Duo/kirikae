use std::fmt::Write;

use kirikae_a64::{Expr, Insn, Stmt};

use crate::discover::{Block, Function, Program};

pub fn fn_name(addr: u64) -> String {
    format!("f_{addr:x}")
}

fn expr(e: &Expr) -> String {
    match e {
        Expr::Const(v) => format!("{v:#x}u64"),
        Expr::Reg(n) => format!("c.x[{n}]"),
        Expr::Sp => "c.sp".into(),
        Expr::Nzcv => "c.nzcv".into(),
        Expr::Tmp(n) => format!("t{n}"),
        Expr::Op(op, args) => {
            let args: Vec<String> = args.iter().map(expr).collect();
            format!("ops::{}({})", op.name(), args.join(", "))
        }
        Expr::Load { size, addr } => format!("c.ld{}({})", *size as u32 * 8, expr(addr)),
        Expr::SysReg(r) => format!("c.sysreg(SysReg::{r:?})"),
        Expr::V { reg, hi } => format!("c.v[{reg}][{}]", *hi as u8),
    }
}

struct Emitter<'a> {
    out: String,
    prog: &'a Program,
    func: &'a Function,
    block_start: u64,
}

impl Emitter<'_> {
    fn line(&mut self, indent: usize, text: &str) {
        for _ in 0..indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn goto(&mut self, indent: usize, target: u64) {
        if self.func.has_block(target) {
            if target <= self.block_start {
                self.line(indent, "if c.poll() { return; }");
            }
            self.line(indent, &format!("bb = {target:#x}; continue;"));
        } else if self.prog.entries.contains(&target) {
            self.line(indent, &format!("c.pc = {target:#x}; return;"));
        } else {
            self.line(indent, &format!("c.pc = {target:#x}; return;"));
        }
    }

    fn after_call(&mut self, indent: usize, ret: u64) {
        self.line(indent, &format!("if c.pc != {ret:#x} {{ kirikae_rt::call_indirect(c, {ret:#x}); if c.pc != {ret:#x} {{ return; }} }}"));
        self.goto(indent, ret);
    }

    fn stmt(&mut self, indent: usize, insn: &Insn, s: &Stmt) {
        let line = match s {
            Stmt::Let(n, e) => format!("let t{n}: u64 = {};", expr(e)),
            Stmt::SetReg(n, e) => format!("c.x[{n}] = {};", expr(e)),
            Stmt::SetSp(e) => format!("c.sp = {};", expr(e)),
            Stmt::SetNzcv(e) => format!("c.nzcv = {};", expr(e)),
            Stmt::SetSysReg(r, e) => format!("c.set_sysreg(SysReg::{r:?}, {});", expr(e)),
            Stmt::SetV { reg, hi, value } => format!("c.v[{reg}][{}] = {};", *hi as u8, expr(value)),
            Stmt::Store { size, addr, value } => format!("c.st{}({}, {});", *size as u32 * 8, expr(addr), expr(value)),
            Stmt::Monitor { addr, lo, hi } => format!("c.monitor({}, {}, {});", expr(addr), expr(lo), expr(hi)),
            Stmt::StoreExclusive { size, addr, lo, hi, status } => {
                let call = format!("c.store_exclusive({size}, {}, {}, {})", expr(addr), expr(lo), expr(hi));
                if *status == 31 { format!("{call};") } else { format!("c.x[{status}] = {call};") }
            }
            Stmt::Fence => "std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);".into(),
            Stmt::Simd(word) => format!("kirikae_rt::simd::exec(c, {word:#010x});"),
            Stmt::Svc(imm) => {
                let next = insn.next();
                format!("c.pc = {next:#x}; kirikae_rt::svc(c, {imm:#x}); if c.halted || c.pc != {next:#x} {{ return; }}")
            }
            Stmt::Undefined(word) => {
                format!("c.pc = {:#x}; kirikae_rt::undefined(c, {:#x}, {word:#010x}); return;", insn.addr, insn.addr)
            }
            Stmt::Return(e) => format!("c.pc = {}; return;", expr(e)),
            Stmt::Jump(t) => return self.goto(indent, *t),
            Stmt::JumpIf(cond, t) => {
                self.line(indent, &format!("if {} != 0 {{", expr(cond)));
                self.goto(indent + 1, *t);
                self.line(indent, "}");
                return self.goto(indent, insn.next());
            }
            Stmt::JumpIndirect(e) => format!("if c.poll() {{ return; }} bb = {}; continue;", expr(e)),
            Stmt::Call(t) => {
                if self.prog.entries.contains(t) {
                    self.line(indent, &format!("c.pc = {t:#x}; {}(c);", fn_name(*t)));
                } else {
                    self.line(indent, &format!("c.pc = {t:#x}; kirikae_rt::call_indirect(c, {:#x});", insn.next()));
                }
                return self.after_call(indent, insn.next());
            }
            Stmt::CallIndirect(e) => {
                self.line(indent, &format!("c.pc = {}; kirikae_rt::call_indirect(c, {:#x});", expr(e), insn.next()));
                return self.after_call(indent, insn.next());
            }
        };
        self.line(indent, &line);
    }

    fn block(&mut self, b: &Block) {
        self.block_start = b.start;
        self.line(3, &format!("{:#x} => {{", b.start));
        for insn in &b.insns {
            self.line(4, &format!("// {:#x}: {:08x}", insn.addr, insn.word));
            self.line(4, "{");
            for s in &insn.stmts {
                self.stmt(5, insn, s);
            }
            self.line(4, "}");
        }
        if b.last().control() == kirikae_a64::Control::Fallthrough {
            self.goto(4, b.next);
        }
        self.line(3, "}");
    }

    fn function(&mut self) {
        let f = self.func;
        self.line(0, &format!("pub extern \"C\" fn {}(c: &mut Cpu) {{", fn_name(f.entry)));
        self.line(1, &format!("let mut bb: u64 = {:#x};", f.entry));
        self.line(1, "loop {");
        self.line(2, "match bb {");
        for b in &f.blocks {
            self.block(b);
        }
        self.line(3, "_ => { c.pc = bb; return; }");
        self.line(2, "}");
        self.line(1, "}");
        self.line(0, "}");
        self.line(0, "");
    }
}

pub fn emit_function(prog: &Program, func: &Function) -> String {
    let mut e = Emitter { out: String::new(), prog, func, block_start: func.entry };
    e.function();
    e.out
}

pub fn emit_program(prog: &Program, image_file: &str) -> String {
    let mut out = String::new();
    writeln!(out, "// Generated by kirikae {}. Do not edit: rerun `kirikae build` instead.", env!("CARGO_PKG_VERSION")).unwrap();
    out.push_str(
        "#![allow(unused_mut, unused_variables, unused_braces, unused_imports, unused_parens, unreachable_code, non_snake_case, clippy::all)]\n\n",
    );
    out.push_str("use kirikae_rt::{ops, Cpu, SysReg};\n\n");
    writeln!(out, "static IMAGE: &[u8] = include_bytes!({image_file:?});\n").unwrap();
    writeln!(out, "static ENTRIES: [kirikae_rt::Entry; {}] = [", prog.functions.len()).unwrap();
    for f in &prog.functions {
        writeln!(out, "    kirikae_rt::Entry {{ addr: {:#x}, func: {} }},", f.entry, fn_name(f.entry)).unwrap();
    }
    out.push_str("];\n\n");
    out.push_str("fn main() {\n    kirikae_rt::program_main(IMAGE, &ENTRIES)\n}\n\n");
    for f in &prog.functions {
        out.push_str(&emit_function(prog, f));
    }
    out
}
