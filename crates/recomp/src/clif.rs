use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    self, AbiParam, BlockArg, InstBuilder, MemFlagsData, Signature, UserFuncName, Value, types,
};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use kirikae_a64::{Expr, Insn, Op, Stmt, SysReg};
use kirikae_rt::abi;
use kirikae_rt::mem::ADDR_MASK;

fn fp_double(op: Op) -> bool {
    matches!(
        op,
        Op::FAdd64
            | Op::FSub64
            | Op::FMul64
            | Op::FDiv64
            | Op::FMax64
            | Op::FMin64
            | Op::FSqrt64
            | Op::FMulAdd64
            | Op::FCmp64
    )
}

fn fp_inputs(op: Op) -> usize {
    match op {
        Op::FMulAdd32 | Op::FMulAdd64 => 3,
        Op::FSqrt32 | Op::FSqrt64 => 1,
        _ => 2,
    }
}

/// Lane widths the vector arithmetic below can fold: they divide the 64-bit halves of a
/// register, so no lane crosses a half boundary.
fn lane_width(esize: u64) -> bool {
    esize != 0 && esize <= 64 && 64 % esize == 0
}

use crate::discover::{CodeSpace, Function, build_function};

const I64: ir::Type = types::I64;
const I32: ir::Type = types::I32;

pub fn host_isa() -> Result<OwnedTargetIsa, String> {
    native_isa(false)
}

fn object_isa() -> Result<OwnedTargetIsa, String> {
    native_isa(!cfg!(windows))
}

fn native_isa(pic: bool) -> Result<OwnedTargetIsa, String> {
    let mut flags = settings::builder();
    let set = |f: &mut settings::Builder, k: &str, v: &str| {
        f.set(k, v)
            .map_err(|e| format!("cranelift setting {k}: {e}"))
    };
    set(&mut flags, "opt_level", "speed")?;
    set(
        &mut flags,
        "enable_verifier",
        if cfg!(debug_assertions) {
            "true"
        } else {
            "false"
        },
    )?;
    set(&mut flags, "is_pic", if pic { "true" } else { "false" })?;
    let builder =
        cranelift_native::builder().map_err(|e| format!("host not supported by Cranelift: {e}"))?;
    builder
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())
}

pub fn symbol(addr: u64) -> String {
    format!("kirikae_g_{addr:x}")
}

fn cpu_flags() -> MemFlagsData {
    MemFlagsData::trusted()
}

fn guest_flags() -> MemFlagsData {
    MemFlagsData::new()
}

struct HelperIds {
    svc: FuncId,
    undefined: FuncId,
    call_indirect: FuncId,
    poll_slow: FuncId,
    sysreg: FuncId,
    set_sysreg: FuncId,
    op: FuncId,
    simd: FuncId,
    store_exclusive: FuncId,
}

#[derive(Clone, Copy)]
struct HelperRefs {
    svc: ir::FuncRef,
    undefined: ir::FuncRef,
    call_indirect: ir::FuncRef,
    poll_slow: ir::FuncRef,
    sysreg: ir::FuncRef,
    set_sysreg: ir::FuncRef,
    op: ir::FuncRef,
    simd: ir::FuncRef,
    store_exclusive: ir::FuncRef,
}

pub struct Compiler<M: Module> {
    pub module: M,
    helpers: HelperIds,
    guest_sig: Signature,
    funcs: HashMap<u64, FuncId>,
    ctx: Context,
    fctx: FunctionBuilderContext,
    fast_regs: bool,
}

impl<M: Module> Compiler<M> {
    pub fn new(mut module: M) -> Result<Compiler<M>, String> {
        let ptr = module.target_config().pointer_type();
        fn sig<M: Module>(module: &M, params: &[ir::Type], ret: Option<ir::Type>) -> Signature {
            let mut s = module.make_signature();
            s.params.extend(params.iter().map(|t| AbiParam::new(*t)));
            s.returns.extend(ret.map(AbiParam::new));
            s
        }
        fn declare<M: Module>(
            module: &mut M,
            name: &str,
            params: &[ir::Type],
            ret: Option<ir::Type>,
        ) -> Result<FuncId, String> {
            let s = sig(module, params, ret);
            module
                .declare_function(name, Linkage::Import, &s)
                .map_err(|e| e.to_string())
        }
        let m = &mut module;
        let helpers = HelperIds {
            svc: declare(m, "kirikae_svc", &[ptr, I32], None)?,
            undefined: declare(m, "kirikae_undefined", &[ptr, I64, I32], None)?,
            call_indirect: declare(m, "kirikae_call_indirect", &[ptr, I64], None)?,
            poll_slow: declare(m, "kirikae_poll_slow", &[ptr], Some(I32))?,
            sysreg: declare(m, "kirikae_sysreg", &[ptr, I32], Some(I64))?,
            set_sysreg: declare(m, "kirikae_set_sysreg", &[ptr, I32, I64], None)?,
            op: declare(m, "kirikae_op", &[I32, I64, I64, I64, I64, I64], Some(I64))?,
            simd: declare(m, "kirikae_simd", &[ptr, I32], None)?,
            store_exclusive: declare(
                m,
                "kirikae_store_exclusive",
                &[ptr, I32, I64, I64, I64],
                Some(I64),
            )?,
        };
        let guest_sig = sig(&module, &[ptr], None);
        let ctx = module.make_context();
        Ok(Compiler {
            module,
            helpers,
            guest_sig,
            funcs: HashMap::new(),
            ctx,
            fctx: FunctionBuilderContext::new(),
            fast_regs: false,
        })
    }

    pub fn set_fast_regs(&mut self, on: bool) {
        self.fast_regs = on;
    }

    pub fn guest(&mut self, addr: u64, linkage: Linkage) -> Result<FuncId, String> {
        if let Some(id) = self.funcs.get(&addr)
            && linkage != Linkage::Export
        {
            return Ok(*id);
        }
        let id = self
            .module
            .declare_function(&symbol(addr), linkage, &self.guest_sig)
            .map_err(|e| e.to_string())?;
        self.funcs.insert(addr, id);
        Ok(id)
    }

    /// Compile one guest function into this object. `in_scope` tells whether a call target
    /// can be a link-time import: everything in a single-module build, only this object's own
    /// functions when the objects are linked into separate DLLs (see [`Calls`]).
    pub fn compile(
        &mut self,
        f: &Function,
        entries: &BTreeSet<u64>,
        in_scope: &dyn Fn(u64) -> bool,
    ) -> Result<FuncId, String> {
        let id = self.guest(f.entry, Linkage::Export)?;
        let mut callees = BTreeSet::new();
        for b in &f.blocks {
            for insn in &b.insns {
                for s in &insn.stmts {
                    if let Stmt::Call(t) = s {
                        callees.insert(*t);
                    }
                }
            }
        }
        let mut callee_ids = Vec::new();
        for t in callees
            .into_iter()
            .filter(|t| entries.contains(t) && in_scope(*t) && !f.has_block(*t))
        {
            callee_ids.push((t, self.guest(t, Linkage::Import)?));
        }

        self.ctx.func.signature = self.guest_sig.clone();
        self.ctx.func.name = UserFuncName::user(0, id.as_u32());
        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fctx);
            let h = &self.helpers;
            let m = &mut self.module;
            let helpers = HelperRefs {
                svc: m.declare_func_in_func(h.svc, b.func),
                undefined: m.declare_func_in_func(h.undefined, b.func),
                call_indirect: m.declare_func_in_func(h.call_indirect, b.func),
                poll_slow: m.declare_func_in_func(h.poll_slow, b.func),
                sysreg: m.declare_func_in_func(h.sysreg, b.func),
                set_sysreg: m.declare_func_in_func(h.set_sysreg, b.func),
                op: m.declare_func_in_func(h.op, b.func),
                simd: m.declare_func_in_func(h.simd, b.func),
                store_exclusive: m.declare_func_in_func(h.store_exclusive, b.func),
            };
            let callees: HashMap<u64, ir::FuncRef> = callee_ids
                .iter()
                .map(|(a, id)| (*a, m.declare_func_in_func(*id, b.func)))
                .collect();
            Lower::function(&mut b, f, entries, helpers, callees, self.fast_regs);
            b.seal_all_blocks();
            b.finalize(self.module.target_config());
        }
        let result = self
            .module
            .define_function(id, &mut self.ctx)
            .map_err(|e| format!("function {:#x}: {e:?}", f.entry));
        self.module.clear_context(&mut self.ctx);
        result.map(|_| id)
    }
}

struct Lower<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    cpu: Value,
    mem: Value,
    h: HelperRefs,
    blocks: HashMap<u64, ir::Block>,
    callees: HashMap<u64, ir::FuncRef>,
    ret: ir::Block,
    dispatch: Option<ir::Block>,
    block_start: u64,
    cache: Option<Cache>,
}

type Temps = [Option<Value>; 8];

type Regs = u128;
const SLOTS: u32 = 97;
const SP_SLOT: u32 = 31;
const NZCV_SLOT: u32 = 96;
const _: () = assert!(abi::OFF_SP == abi::OFF_X + 8 * 31 && abi::OFF_V == abi::OFF_X + 8 * 32);

fn x_slot(n: u8) -> u32 {
    n as u32
}

fn v_slot(reg: u8, hi: bool) -> u32 {
    32 + 2 * reg as u32 + hi as u32
}

fn slot_offset(s: u32) -> usize {
    if s == NZCV_SLOT {
        abi::OFF_NZCV
    } else {
        abi::OFF_X + 8 * s as usize
    }
}

fn slots(set: Regs) -> impl Iterator<Item = u32> {
    (0..SLOTS).filter(move |s| set & (1 << s) != 0)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RegState {
    valid: Regs,
    dirty: Regs,
}

#[derive(Clone, Copy)]
enum Event {
    Read(u32),
    Write(u32),
    Sync,
    Simd(Regs),
}

struct WriteRecorder {
    seed: u64,
    writes: Regs,
}

impl WriteRecorder {
    fn value(&self, salt: u64) -> u64 {
        let mut x = self.seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 31;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^ (x >> 29)
    }
}

impl kirikae_a64::simd::Env for WriteRecorder {
    fn x(&self, n: u32) -> u64 {
        if n == 31 { 0 } else { self.value(n as u64) }
    }
    fn set_x(&mut self, n: u32, _: u64) {
        if n != 31 {
            self.writes |= 1 << n;
        }
    }
    fn sp(&self) -> u64 {
        self.value(31)
    }
    fn set_sp(&mut self, _: u64) {
        self.writes |= 1 << SP_SLOT;
    }
    fn v(&self, n: u32) -> u128 {
        self.value(32 + n as u64) as u128 | (self.value(64 + n as u64) as u128) << 64
    }
    fn set_v(&mut self, n: u32, _: u128) {
        self.writes |= 3 << v_slot(n as u8, false);
    }
    fn nzcv(&self) -> u64 {
        self.value(96) & 0xf000_0000
    }
    fn set_nzcv(&mut self, _: u64) {
        self.writes |= 1 << NZCV_SLOT;
    }
    fn fpcr(&self) -> u64 {
        0
    }
    fn fpsr(&self) -> u64 {
        0
    }
    fn set_fpsr(&mut self, _: u64) {}
    fn load(&mut self, addr: u64, _: u8) -> u64 {
        self.value(addr)
    }
    fn store(&mut self, _: u64, _: u8, _: u64) {}
}

fn simd_writes(word: u32) -> Regs {
    let mut writes = 0;
    for seed in [
        0,
        u64::MAX,
        1,
        0x8000_0000_0000_0000,
        0x1234_5678_9abc_def0,
        0x0f0f_f0f0_3c3c_c3c3,
        7,
        0xdead_beef,
    ] {
        let mut r = WriteRecorder { seed, writes: 0 };
        kirikae_a64::simd::exec(&mut r, word);
        writes |= r.writes;
    }
    writes
}

fn expr_events(e: &Expr, out: &mut Vec<Event>) {
    match e {
        Expr::Reg(n) => out.push(Event::Read(x_slot(*n))),
        Expr::Sp => out.push(Event::Read(SP_SLOT)),
        Expr::Nzcv => out.push(Event::Read(NZCV_SLOT)),
        Expr::V { reg, hi } => out.push(Event::Read(v_slot(*reg, *hi))),
        Expr::Load { addr, .. } => expr_events(addr, out),
        Expr::Op(_, args) => args.iter().for_each(|a| expr_events(a, out)),
        Expr::Const(_) | Expr::Tmp(_) | Expr::SysReg(_) => {}
    }
}

fn insn_events(insn: &Insn, out: &mut Vec<Event>) {
    for s in &insn.stmts {
        match s {
            Stmt::Let(_, e) | Stmt::SetSysReg(_, e) => expr_events(e, out),
            Stmt::SetReg(n, e) => {
                expr_events(e, out);
                out.push(Event::Write(x_slot(*n)));
            }
            Stmt::SetSp(e) => {
                expr_events(e, out);
                out.push(Event::Write(SP_SLOT));
            }
            Stmt::SetNzcv(e) => {
                expr_events(e, out);
                out.push(Event::Write(NZCV_SLOT));
            }
            Stmt::SetV { reg, hi, value } => {
                expr_events(value, out);
                out.push(Event::Write(v_slot(*reg, *hi)));
            }
            Stmt::Store { addr, value, .. } => {
                expr_events(addr, out);
                expr_events(value, out);
            }
            Stmt::Monitor { addr, lo, hi } => {
                expr_events(addr, out);
                expr_events(lo, out);
                expr_events(hi, out);
            }
            Stmt::StoreExclusive {
                addr,
                lo,
                hi,
                status,
                ..
            } => {
                expr_events(addr, out);
                expr_events(lo, out);
                expr_events(hi, out);
                if *status != 31 {
                    out.push(Event::Write(x_slot(*status)));
                }
            }
            Stmt::Fence => {}
            Stmt::Simd(w) => out.push(Event::Simd(simd_writes(*w))),
            Stmt::Svc(_) => out.push(Event::Sync),
            Stmt::Undefined(_) | Stmt::Call(_) => {
                out.push(Event::Sync);
                return;
            }
            Stmt::Return(e) | Stmt::JumpIf(e, _) => {
                expr_events(e, out);
                return;
            }
            Stmt::JumpIndirect(e) | Stmt::CallIndirect(e) => {
                expr_events(e, out);
                out.push(Event::Sync);
                return;
            }
            Stmt::Jump(_) => return,
        }
    }
}

fn apply(mut st: RegState, ev: Event) -> RegState {
    match ev {
        Event::Read(s) => st.valid |= 1 << s,
        Event::Write(s) => {
            st.valid |= 1 << s;
            st.dirty |= 1 << s;
        }
        Event::Sync => st = RegState::default(),
        Event::Simd(w) => {
            st = RegState {
                valid: st.valid & !w,
                dirty: 0,
            }
        }
    }
    st
}

fn successors(f: &Function, g: &crate::discover::Block) -> Vec<u64> {
    let last = g.last();
    let mut out = match last.stmts.last() {
        Some(Stmt::Jump(t)) => vec![*t],
        Some(Stmt::JumpIf(_, t)) => vec![*t, last.next()],
        Some(Stmt::Call(_) | Stmt::CallIndirect(_)) => vec![last.next()],
        Some(Stmt::Return(_) | Stmt::Undefined(_) | Stmt::JumpIndirect(_)) => vec![],
        _ => vec![g.next],
    };
    out.retain(|t| f.has_block(*t));
    out
}

fn plan_registers(f: &Function) -> HashMap<u64, RegState> {
    let n = f.blocks.len();
    let index: HashMap<u64, usize> = f
        .blocks
        .iter()
        .enumerate()
        .map(|(i, g)| (g.start, i))
        .collect();
    let events: Vec<Vec<Event>> = f
        .blocks
        .iter()
        .map(|g| {
            let mut ev = Vec::new();
            for insn in &g.insns {
                insn_events(insn, &mut ev);
            }
            ev
        })
        .collect();
    let succ: Vec<Vec<usize>> = f
        .blocks
        .iter()
        .map(|g| successors(f, g).iter().map(|t| index[t]).collect())
        .collect();
    let mut preds = vec![Vec::new(); n];
    for (i, s) in succ.iter().enumerate() {
        for &t in s {
            preds[t].push(i);
        }
    }

    let mut live: Vec<Regs> = vec![0; n];
    loop {
        let mut changed = false;
        for i in (0..n).rev() {
            let mut l = succ[i].iter().fold(0, |a, &t| a | live[t]);
            for ev in events[i].iter().rev() {
                match *ev {
                    Event::Read(s) => l |= 1 << s,
                    Event::Write(s) => l &= !(1 << s),
                    Event::Sync => l = 0,
                    Event::Simd(w) => l &= !w,
                }
            }
            if l != live[i] {
                live[i] = l;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    const ALL: Regs = (1 << SLOTS) - 1;
    let first = index[&f.entry];
    let mut entry = vec![RegState::default(); n];
    let mut exit = vec![
        RegState {
            valid: ALL,
            dirty: 0
        };
        n
    ];
    loop {
        let mut changed = false;
        for i in 0..n {
            let start = if i == first || preds[i].is_empty() {
                0
            } else {
                ALL
            };
            let (valid, dirty) = preds[i].iter().fold((start, 0), |(v, d), &p| {
                (v & exit[p].valid, d | exit[p].dirty)
            });
            let valid = valid & live[i];
            let st = RegState {
                valid,
                dirty: dirty & valid,
            };
            let out = events[i].iter().fold(st, |s, ev| apply(s, *ev));
            if st != entry[i] || out != exit[i] {
                entry[i] = st;
                exit[i] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    f.blocks
        .iter()
        .zip(entry)
        .map(|(g, st)| (g.start, st))
        .collect()
}

struct Cache {
    vars: Vec<Variable>,
    state: RegState,
    entry: HashMap<u64, RegState>,
}

impl Lower<'_, '_> {
    fn function(
        b: &mut FunctionBuilder,
        f: &Function,
        entries: &BTreeSet<u64>,
        h: HelperRefs,
        callees: HashMap<u64, ir::FuncRef>,
        fast_regs: bool,
    ) {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let cpu = b.block_params(entry)[0];
        let mem = b
            .ins()
            .load(I64, cpu_flags().with_readonly(), cpu, abi::OFF_MEM as i32);
        let blocks: HashMap<u64, ir::Block> = f
            .blocks
            .iter()
            .map(|g| (g.start, b.create_block()))
            .collect();
        let ret = b.create_block();
        let has_indirect = f
            .blocks
            .iter()
            .any(|g| matches!(g.last().stmts.last(), Some(Stmt::JumpIndirect(_))));
        let dispatch = has_indirect.then(|| {
            let d = b.create_block();
            b.append_block_param(d, I64);
            d
        });
        let cache = fast_regs.then(|| Cache {
            vars: (0..SLOTS).map(|_| b.declare_var(I64)).collect(),
            state: RegState::default(),
            entry: plan_registers(f),
        });
        let mut l = Lower {
            b,
            cpu,
            mem,
            h,
            blocks,
            callees,
            ret,
            dispatch,
            block_start: f.entry,
            cache,
        };
        l.edge(f.entry);
        l.b.ins().jump(l.blocks[&f.entry], &[]);
        let _ = entries;

        for g in &f.blocks {
            l.block_start = g.start;
            l.b.switch_to_block(l.blocks[&g.start]);
            if let Some(c) = &mut l.cache {
                c.state = c.entry[&g.start];
            }
            let mut ended = false;
            for insn in &g.insns {
                ended = l.insn(insn);
            }
            if !ended {
                l.goto(g.next);
            }
        }

        l.b.switch_to_block(ret);
        l.b.ins().return_(&[]);

        if let Some(d) = dispatch {
            let mut targets: Vec<(u64, ir::Block)> =
                l.blocks.iter().map(|(a, b)| (*a, *b)).collect();
            for (addr, blk) in &mut targets {
                let load = l.cache.as_ref().map_or(0, |c| c.entry[addr].valid);
                if load != 0 {
                    let tramp = l.b.create_block();
                    l.b.switch_to_block(tramp);
                    l.load_vars(load);
                    l.b.ins().jump(*blk, &[]);
                    *blk = tramp;
                }
            }
            l.b.switch_to_block(d);
            let target = l.b.block_params(d)[0];
            l.set_pc(target);
            let fallback = l.b.create_block();
            let mut sw = Switch::new();
            for (addr, blk) in &targets {
                sw.set_entry(*addr as u128, *blk);
            }
            sw.emit(l.b, target, fallback);
            l.b.switch_to_block(fallback);
            l.b.ins().jump(ret, &[]);
        }
    }

    fn iconst(&mut self, v: u64) -> Value {
        self.b.ins().iconst(I64, v as i64)
    }

    fn load_cpu(&mut self, off: usize) -> Value {
        self.b.ins().load(I64, cpu_flags(), self.cpu, off as i32)
    }

    fn store_cpu(&mut self, off: usize, v: Value) {
        self.b.ins().store(cpu_flags(), v, self.cpu, off as i32);
    }

    fn read(&mut self, s: u32) -> Value {
        let Some(c) = &mut self.cache else {
            return self.load_cpu(slot_offset(s));
        };
        let var = c.vars[s as usize];
        if c.state.valid & (1 << s) != 0 {
            return self.b.use_var(var);
        }
        c.state.valid |= 1 << s;
        let v = self.load_cpu(slot_offset(s));
        self.b.def_var(var, v);
        v
    }

    fn write(&mut self, s: u32, v: Value) {
        let Some(c) = &mut self.cache else {
            return self.store_cpu(slot_offset(s), v);
        };
        c.state.valid |= 1 << s;
        c.state.dirty |= 1 << s;
        let var = c.vars[s as usize];
        self.b.def_var(var, v);
    }

    fn var(&self, s: u32) -> Variable {
        self.cache.as_ref().expect("fast registers").vars[s as usize]
    }

    fn store_vars(&mut self, set: Regs) {
        for s in slots(set) {
            let v = self.b.use_var(self.var(s));
            self.store_cpu(slot_offset(s), v);
        }
    }

    fn load_vars(&mut self, set: Regs) {
        for s in slots(set) {
            let v = self.load_cpu(slot_offset(s));
            self.b.def_var(self.var(s), v);
        }
    }

    fn flush(&mut self) {
        let Some(c) = &mut self.cache else { return };
        let dirty = std::mem::take(&mut c.state.dirty);
        self.store_vars(dirty);
    }

    fn sync(&mut self) {
        self.flush();
        if let Some(c) = &mut self.cache {
            c.state = RegState::default();
        }
    }

    fn edge(&mut self, target: u64) {
        let Some(c) = &mut self.cache else { return };
        let (cur, to) = (c.state, c.entry[&target]);
        c.state = RegState {
            valid: cur.valid | to.valid,
            dirty: cur.dirty & to.dirty,
        };
        self.store_vars(cur.dirty & !to.dirty);
        self.load_vars(to.valid & !cur.valid);
    }

    fn leave(&mut self) {
        self.flush();
        self.b.ins().jump(self.ret, &[]);
    }

    fn set_pc(&mut self, v: Value) {
        self.store_cpu(abi::OFF_PC, v);
    }

    fn set_pc_const(&mut self, pc: u64) {
        let v = self.iconst(pc);
        self.set_pc(v);
    }

    fn host(&mut self, addr: Value) -> Value {
        let masked = self.b.ins().band_imm_u(addr, ADDR_MASK as i64);
        self.b.ins().iadd(self.mem, masked)
    }

    fn call_op(&mut self, op: Op, args: &[Value]) -> Value {
        let mut v = vec![self.b.ins().iconst(I32, op.index() as i64)];
        v.extend_from_slice(args);
        while v.len() < 6 {
            v.push(self.b.ins().iconst(I64, 0));
        }
        let call = self.b.ins().call(self.h.op, &v);
        self.b.inst_results(call)[0]
    }

    fn expr(&mut self, e: &Expr, t: &Temps) -> Value {
        match e {
            Expr::Const(v) => self.iconst(*v),
            Expr::Reg(n) => self.read(x_slot(*n)),
            Expr::Sp => self.read(SP_SLOT),
            Expr::Nzcv => self.read(NZCV_SLOT),
            Expr::Tmp(n) => t[*n as usize].expect("temporary used before it is set"),
            Expr::V { reg, hi } => self.read(v_slot(*reg, *hi)),
            Expr::Load { size, addr } => {
                let a = self.expr(addr, t);
                let p = self.host(a);
                match size {
                    8 => self.b.ins().load(I64, guest_flags(), p, 0),
                    4 => self.b.ins().uload32(guest_flags(), p, 0),
                    2 => self.b.ins().uload16(I64, guest_flags(), p, 0),
                    _ => self.b.ins().uload8(I64, guest_flags(), p, 0),
                }
            }
            Expr::SysReg(SysReg::Fpcr) => self.load_cpu(abi::OFF_FPCR),
            Expr::SysReg(SysReg::Fpsr) => self.load_cpu(abi::OFF_FPSR),
            Expr::SysReg(r) => {
                let id = self.b.ins().iconst(I32, *r as i64);
                let call = self.b.ins().call(self.h.sysreg, &[self.cpu, id]);
                self.b.inst_results(call)[0]
            }
            Expr::Op(op, args) => self.op(*op, args, t),
        }
    }

    fn r32(&mut self, v: Value) -> Value {
        self.b.ins().ireduce(I32, v)
    }

    fn z64(&mut self, v: Value) -> Value {
        self.b.ins().uextend(I64, v)
    }

    fn bool64(&mut self, v: Value) -> Value {
        self.b.ins().uextend(I64, v)
    }

    fn op(&mut self, op: Op, args: &[Expr], t: &Temps) -> Value {
        let k = |i: usize| match args.get(i) {
            Some(Expr::Const(v)) => Some(*v),
            _ => None,
        };
        match op {
            Op::Ubfm64
            | Op::Ubfm32
            | Op::Sbfm64
            | Op::Sbfm32
            | Op::Bfm64
            | Op::Bfm32
            | Op::Extr64
            | Op::Extr32
            | Op::Cond => {
                if let Some(v) = self.op_with_immediates(op, args, t, &k) {
                    return v;
                }
            }
            _ => {}
        }
        let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
        match op {
            Op::Add => self.b.ins().iadd(v[0], v[1]),
            Op::Sub => self.b.ins().isub(v[0], v[1]),
            Op::And => self.b.ins().band(v[0], v[1]),
            Op::Or => self.b.ins().bor(v[0], v[1]),
            Op::Xor => self.b.ins().bxor(v[0], v[1]),
            Op::Not => self.b.ins().bnot(v[0]),
            Op::Mul => self.b.ins().imul(v[0], v[1]),
            Op::Zext8 => self.b.ins().band_imm_u(v[0], 0xff),
            Op::Zext16 => self.b.ins().band_imm_u(v[0], 0xffff),
            Op::Zext32 => self.b.ins().band_imm_u(v[0], 0xffff_ffff),
            Op::Sext8 | Op::Sext16 | Op::Sext32 => {
                let ty = match op {
                    Op::Sext8 => types::I8,
                    Op::Sext16 => types::I16,
                    _ => I32,
                };
                let narrow = self.b.ins().ireduce(ty, v[0]);
                self.b.ins().sextend(I64, narrow)
            }
            Op::Lsl64 => self.b.ins().ishl(v[0], v[1]),
            Op::Lsr64 => self.b.ins().ushr(v[0], v[1]),
            Op::Asr64 => self.b.ins().sshr(v[0], v[1]),
            Op::Ror64 => self.b.ins().rotr(v[0], v[1]),
            Op::Lsl32 | Op::Lsr32 | Op::Asr32 | Op::Ror32 => {
                let a = self.r32(v[0]);
                let s = self.r32(v[1]);
                let r = match op {
                    Op::Lsl32 => self.b.ins().ishl(a, s),
                    Op::Lsr32 => self.b.ins().ushr(a, s),
                    Op::Asr32 => self.b.ins().sshr(a, s),
                    _ => self.b.ins().rotr(a, s),
                };
                self.z64(r)
            }
            Op::Smulh => self.b.ins().smulhi(v[0], v[1]),
            Op::Umulh => self.b.ins().umulhi(v[0], v[1]),
            Op::Clz64 => self.b.ins().clz(v[0]),
            Op::Rbit64 => self.b.ins().bitrev(v[0]),
            Op::Rev64 => self.b.ins().bswap(v[0]),
            Op::Clz32 | Op::Rbit32 | Op::Rev32 => {
                let a = self.r32(v[0]);
                let r = match op {
                    Op::Clz32 => self.b.ins().clz(a),
                    Op::Rbit32 => self.b.ins().bitrev(a),
                    _ => self.b.ins().bswap(a),
                };
                self.z64(r)
            }
            Op::FlagsAdd64 => self.flags_add(v[0], v[1], v[2], false),
            Op::FlagsAdd32 => self.flags_add(v[0], v[1], v[2], true),
            Op::FlagsLogic64 | Op::FlagsLogic32 => {
                let narrow = op == Op::FlagsLogic32;
                let r = if narrow { self.r32(v[0]) } else { v[0] };
                let sign = self.b.ins().ushr_imm_u(r, if narrow { 31 } else { 63 });
                let n = if narrow { self.z64(sign) } else { sign };
                let z = self.b.ins().icmp_imm_u(IntCC::Equal, r, 0);
                let z = self.bool64(z);
                let n = self.b.ins().ishl_imm_u(n, 31);
                let z = self.b.ins().ishl_imm_u(z, 30);
                self.b.ins().bor(n, z)
            }
            Op::Carry => {
                let s = self.b.ins().ushr_imm_u(v[0], 29);
                self.b.ins().band_imm_u(s, 1)
            }
            Op::Select => self.b.ins().select(v[0], v[1], v[2]),
            Op::Eq | Op::Ne => {
                let cc = if op == Op::Eq {
                    IntCC::Equal
                } else {
                    IntCC::NotEqual
                };
                let c = self.b.ins().icmp(cc, v[0], v[1]);
                self.bool64(c)
            }
            Op::Bit => {
                let s = self.b.ins().ushr(v[0], v[1]);
                self.b.ins().band_imm_u(s, 1)
            }
            Op::Vshl | Op::Vushr | Op::Vsshr => self.vec_shift(op, args, t, &k),
            Op::Splat => self.vec_splat(args, t, &k),
            Op::Vmul => self.vec_mul(args, t, &k),
            Op::Vmax | Op::Vmin => self.vec_minmax(op, args, t, &k),
            Op::Vabs => self.vec_abs(args, t, &k),
            Op::Vneg => self.vec_neg(args, t, &k),
            Op::Vsat | Op::VsatFlag => self.vec_sat(op, args, t, &k),
            Op::Vmla | Op::Vmls => self.vec_mla(op, args, t, &k),
            Op::Vpmax | Op::Vpmin => self.vec_pair(op, args, t, &k),
            Op::FAdd32
            | Op::FSub32
            | Op::FMul32
            | Op::FDiv32
            | Op::FMax32
            | Op::FMin32
            | Op::FSqrt32
            | Op::FMulAdd32
            | Op::FCmp32
            | Op::FAdd64
            | Op::FSub64
            | Op::FMul64
            | Op::FDiv64
            | Op::FMax64
            | Op::FMin64
            | Op::FSqrt64
            | Op::FMulAdd64
            | Op::FCmp64 => {
                let fpcr = *v.last().unwrap();
                let (r, ok) = self.fp_scalar(op, &v);
                let double = fp_double(op);
                let n = fp_inputs(op);
                let no_fz = self.no_flush(fpcr);
                let result = ok.is_some().then_some(r);
                let fz_ok = self.flush_free(&v[..n], result, double);
                let fz_ok = self.b.ins().bor(no_fz, fz_ok);
                let ok = match ok {
                    Some(ok) => self.b.ins().band(ok, fz_ok),
                    None => fz_ok,
                };
                self.fp_or_call(op, &v, r, ok)
            }
            Op::FAdd32x2 | Op::FSub32x2 | Op::FMul32x2 | Op::FMulAdd32x2 => {
                let fpcr = *v.last().unwrap();
                let lanes: Vec<(Value, Value)> =
                    v[..v.len() - 1].iter().map(|&x| self.lanes32(x)).collect();
                let one = match op {
                    Op::FAdd32x2 => Op::FAdd32,
                    Op::FSub32x2 => Op::FSub32,
                    Op::FMul32x2 => Op::FMul32,
                    _ => Op::FMulAdd32,
                };
                let lo: Vec<Value> = lanes.iter().map(|l| l.0).collect();
                let hi: Vec<Value> = lanes.iter().map(|l| l.1).collect();
                let (r_lo, ok_lo) = self.fp_scalar(one, &lo);
                let (r_hi, ok_hi) = self.fp_scalar(one, &hi);
                let fz_lo = self.flush_free(&lo, Some(r_lo), false);
                let fz_hi = self.flush_free(&hi, Some(r_hi), false);
                let r_hi = self.b.ins().ishl_imm_u(r_hi, 32);
                let r = self.b.ins().bor(r_lo, r_hi);
                let no_fz = self.no_flush(fpcr);
                let fz_ok = self.b.ins().band(fz_lo, fz_hi);
                let fz_ok = self.b.ins().bor(no_fz, fz_ok);
                let ok = self
                    .b
                    .ins()
                    .band(ok_lo.expect("arithmetic"), ok_hi.expect("arithmetic"));
                let ok = self.b.ins().band(ok, fz_ok);
                self.fp_or_call(op, &v, r, ok)
            }
            Op::FCvtSD | Op::FCvtDS => {
                let widen = op == Op::FCvtSD;
                let f = self.bits_as_float(v[0], !widen);
                let r = if widen {
                    self.b.ins().fpromote(types::F64, f)
                } else {
                    self.b.ins().fdemote(types::F32, f)
                };
                let ordered = self.b.ins().fcmp(FloatCC::Ordered, r, r);
                let bits = self.float_as_bits(r, widen);
                let input_ok = self.flush_free(&[v[0]], None, !widen);
                let fz_ok = if widen {
                    input_ok
                } else {
                    let result_ok = self.flush_free(&[], Some(bits), false);
                    self.b.ins().band(input_ok, result_ok)
                };
                let no_fz = self.no_flush(v[1]);
                let fz_ok = self.b.ins().bor(no_fz, fz_ok);
                let ok = self.b.ins().band(ordered, fz_ok);
                self.fp_or_call(op, &v, bits, ok)
            }
            Op::FToInt32 | Op::FToInt64 | Op::FFromInt32 | Op::FFromInt64 => match k(1) {
                Some(mode) => match self.fp_convert(op, v[0], mode) {
                    Some(r) => r,
                    None => self.call_op(op, &v),
                },
                None => self.call_op(op, &v),
            },
            _ => self.call_op(op, &v),
        }
    }

    /// A lane shift on one 64-bit half. Lane widths divide 64, so this is a plain word
    /// shift plus a keep mask that clears the bits each lane must not receive from its
    /// neighbour; both are constants because the lifter always passes `esize` and
    /// `shift` literal. Falls back to the shared semantics for anything else.
    fn vec_shift(
        &mut self,
        op: Op,
        args: &[Expr],
        t: &Temps,
        k: &impl Fn(usize) -> Option<u64>,
    ) -> Value {
        let (Some(esize), Some(shift)) = (k(1), k(2)) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(op, &v);
        };
        let keep = if op == Op::Vshl {
            kirikae_a64::ops::lane_keep_left(esize, shift)
        } else {
            kirikae_a64::ops::lane_keep(esize, shift)
        };
        let a = self.expr(&args[0], t);
        if op == Op::Vsshr {
            return self.vec_sshr(a, esize, shift);
        }
        if keep == 0 {
            return self.iconst(0);
        }
        let s = shift as i64;
        let r = match op {
            Op::Vshl => self.b.ins().ishl_imm_u(a, s),
            _ => self.b.ins().ushr_imm_u(a, s),
        };
        self.b.ins().band_imm_u(r, keep as i64)
    }

    /// Same formula as `kirikae_a64::ops::vsshr`, with every constant folded.
    fn vec_sshr(&mut self, a: Value, esize: u64, shift: u64) -> Value {
        if esize == 0 || esize > 64 {
            return self.iconst(0);
        }
        let keep = kirikae_a64::ops::lane_keep(esize, shift);
        let shifted = if keep == 0 {
            self.iconst(0)
        } else {
            let r = self.b.ins().ushr_imm_u(a, shift as i64);
            self.b.ins().band_imm_u(r, keep as i64)
        };
        let unit = if shift == 0 {
            0
        } else if shift >= esize {
            kirikae_a64::ops::lane_mask(esize)
        } else {
            (kirikae_a64::ops::lane_mask(shift) << (esize - shift))
                & kirikae_a64::ops::lane_mask(esize)
        };
        if unit == 0 {
            return shifted;
        }
        let signmask = self.iconst(kirikae_a64::ops::splat(1 << (esize - 1), esize));
        let signs = self.b.ins().band(a, signmask);
        let signs = self.b.ins().ushr_imm_u(signs, esize as i64 - 1);
        let unit = self.iconst(unit);
        let fill = self.b.ins().imul(signs, unit);
        self.b.ins().bor(shifted, fill)
    }

    /// Broadcasts the low `esize` bits of a value to every lane. Every lane width that
    /// divides 64 is a mask plus a multiply by a repeated-ones constant.
    fn vec_splat(&mut self, args: &[Expr], t: &Temps, k: &impl Fn(usize) -> Option<u64>) -> Value {
        let Some(esize) = k(1) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(Op::Splat, &v);
        };
        let a = self.expr(&args[0], t);
        if esize >= 64 {
            return a;
        }
        let lane = self
            .b
            .ins()
            .band_imm_u(a, kirikae_a64::ops::lane_mask(esize) as i64);
        let rep = self.iconst(kirikae_a64::ops::lane_repeat_ones(esize));
        self.b.ins().imul(lane, rep)
    }

    /// Lane-wise wrapping multiply. A whole-half `imul` would mix lanes (the high half
    /// of one lane's product lands in the next), so every lane is isolated, multiplied
    /// and put back. `esize` is always a literal here.
    fn vec_mul(&mut self, args: &[Expr], t: &Temps, k: &impl Fn(usize) -> Option<u64>) -> Value {
        let Some(esize) = k(2) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(Op::Vmul, &v);
        };
        if !lane_width(esize) {
            return self.iconst(0);
        }
        let (a, b) = (self.expr(&args[0], t), self.expr(&args[1], t));
        if esize >= 64 {
            return self.b.ins().imul(a, b);
        }
        let m = kirikae_a64::ops::lane_mask(esize) as i64;
        let mut out = self.iconst(0);
        let mut i = 0i64;
        while i < 64 {
            let (ea, eb) = if i == 0 {
                (a, b)
            } else {
                (self.b.ins().ushr_imm_u(a, i), self.b.ins().ushr_imm_u(b, i))
            };
            let ea = self.b.ins().band_imm_u(ea, m);
            let eb = self.b.ins().band_imm_u(eb, m);
            let p = self.b.ins().imul(ea, eb);
            let p = self.b.ins().band_imm_u(p, m);
            let p = if i == 0 {
                p
            } else {
                self.b.ins().ishl_imm_u(p, i)
            };
            out = self.b.ins().bor(out, p);
            i += esize as i64;
        }
        out
    }

    /// Lane-wise signed/unsigned maximum or minimum, with the same constants folded as
    /// `kirikae_a64::ops::vmax`. `(x | H) - (y & !H)` cannot borrow between lanes and its
    /// top bit says whether the low bits of `x` are not below those of `y`; when the top
    /// bits differ they decide, and a signed compare is the unsigned one with the sign
    /// bits flipped.
    fn vec_minmax(
        &mut self,
        op: Op,
        args: &[Expr],
        t: &Temps,
        k: &impl Fn(usize) -> Option<u64>,
    ) -> Value {
        let (Some(esize), Some(signed)) = (k(2), k(3)) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(op, &v);
        };
        if !lane_width(esize) {
            return self.iconst(0);
        }
        let (a, b) = (self.expr(&args[0], t), self.expr(&args[1], t));
        let hrep = kirikae_a64::ops::splat(1 << (esize - 1), esize);
        let (hval, low) = (self.iconst(hrep), self.iconst(!hrep));
        let lane = self.iconst(kirikae_a64::ops::lane_mask(esize));
        let (x, y) = if signed != 0 {
            (self.b.ins().bxor(a, hval), self.b.ins().bxor(b, hval))
        } else {
            (a, b)
        };
        let xh = self.b.ins().bor(x, hval);
        let yl = self.b.ins().band(y, low);
        let diff = self.b.ins().isub(xh, yl);
        let tops = self.b.ins().bxor(x, y);
        let tops = self.b.ins().band(tops, hval);
        let differ = self.b.ins().ushr_imm_u(tops, esize as i64 - 1);
        let differ = self.b.ins().imul(differ, lane);
        let xh = self.b.ins().band(x, hval);
        let ge_x = self.b.ins().band(xh, differ);
        let dh = self.b.ins().band(diff, hval);
        let not_differ = self.b.ins().bnot(differ);
        let ge_t = self.b.ins().band(dh, not_differ);
        let ge = self.b.ins().bor(ge_x, ge_t);
        let ge = self.b.ins().ushr_imm_u(ge, esize as i64 - 1);
        let mask = self.b.ins().imul(ge, lane);
        let ab = self.b.ins().bxor(a, b);
        let sel = self.b.ins().band(ab, mask);
        if op == Op::Vmax {
            self.b.ins().bxor(b, sel)
        } else {
            self.b.ins().bxor(a, sel)
        }
    }

    /// Lane-wise non-saturating absolute value, the same formula as
    /// `kirikae_a64::ops::vabs`: negate the negative lanes with `(x ^ m) + sign`, where
    /// the sign bit added at each lane base cannot carry out of its lane.
    fn vec_abs(&mut self, args: &[Expr], t: &Temps, k: &impl Fn(usize) -> Option<u64>) -> Value {
        let Some(esize) = k(1) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(Op::Vabs, &v);
        };
        if !lane_width(esize) {
            return self.iconst(0);
        }
        let a = self.expr(&args[0], t);
        let ones = self.iconst(kirikae_a64::ops::splat(1, esize));
        let lane = self.iconst(kirikae_a64::ops::lane_mask(esize));
        let sign = self.b.ins().ushr_imm_u(a, esize as i64 - 1);
        let sign = self.b.ins().band(sign, ones);
        let negate = self.b.ins().imul(sign, lane);
        let x = self.b.ins().bxor(a, negate);
        self.b.ins().iadd(x, sign)
    }

    /// Lane-wise wrapping negation, mirroring `kirikae_a64::ops::vneg`. Adding one at
    /// every lane base would carry into the next lane when a lane is zero, so the low
    /// bits and the sign bits are combined separately.
    fn vec_neg(&mut self, args: &[Expr], t: &Temps, k: &impl Fn(usize) -> Option<u64>) -> Value {
        let Some(esize) = k(1) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(Op::Vneg, &v);
        };
        if !lane_width(esize) {
            return self.iconst(0);
        }
        let a = self.expr(&args[0], t);
        let hrep = kirikae_a64::ops::splat(1 << (esize - 1), esize);
        let (hval, low) = (self.iconst(hrep), self.iconst(!hrep));
        let ones = self.iconst(kirikae_a64::ops::splat(1, esize));
        let inverted = self.b.ins().bnot(a);
        let low_bits = self.b.ins().band(inverted, low);
        let v = self.b.ins().iadd(low_bits, ones);
        let low_part = self.b.ins().band(v, low);
        let sign = self.b.ins().band(inverted, hval);
        let vh = self.b.ins().band(v, hval);
        let top = self.b.ins().bxor(sign, vh);
        self.b.ins().bor(low_part, top)
    }

    /// Sign-extends a lane value held in the low bits of `v` to a full `i64`.
    fn sx_lane(&mut self, v: Value, esize: u64) -> Value {
        let sh = (64 - esize) as i64;
        let shifted = self.b.ins().ishl_imm_u(v, sh);
        self.b.ins().sshr_imm_u(shifted, sh)
    }

    /// One saturating lane, the same formula as `kirikae_a64::ops::sat_lane`: the clamped
    /// value and whether it clamped. The bounds and the sign extension are constants, so
    /// only the compares and the selects remain.
    fn sat_lane(&mut self, x: Value, y: Value, esize: u64, mode: u64) -> (Value, Value) {
        let mask = self.iconst(kirikae_a64::ops::lane_mask(esize));
        let zero = self.iconst(0);
        if mode == 1 || mode == 3 {
            return if mode == 1 {
                let sum = self.b.ins().iadd(x, y);
                let over = self.b.ins().icmp(IntCC::UnsignedGreaterThan, sum, mask);
                (self.b.ins().select(over, mask, sum), over)
            } else {
                let diff = self.b.ins().isub(x, y);
                let under = self.b.ins().icmp(IntCC::UnsignedLessThan, x, y);
                (self.b.ins().select(under, zero, diff), under)
            };
        }
        let exact = match mode {
            2 => {
                let (sa, sb) = (self.sx_lane(x, esize), self.sx_lane(y, esize));
                self.b.ins().isub(sa, sb)
            }
            4 => {
                let a = self.sx_lane(x, esize);
                let negated = self.b.ins().isub(zero, a);
                let negative = self.b.ins().icmp(IntCC::SignedLessThan, a, zero);
                self.b.ins().select(negative, negated, a)
            }
            5 => {
                let a = self.sx_lane(x, esize);
                self.b.ins().isub(zero, a)
            }
            _ => {
                let (sa, sb) = (self.sx_lane(x, esize), self.sx_lane(y, esize));
                self.b.ins().iadd(sa, sb)
            }
        };
        let bits = 1u64 << (esize - 1);
        let (hi, lo) = (self.iconst(bits - 1), self.iconst(bits.wrapping_neg()));
        let too_hi = self.b.ins().icmp(IntCC::SignedGreaterThan, exact, hi);
        let too_lo = self.b.ins().icmp(IntCC::SignedLessThan, exact, lo);
        let low = self.b.ins().select(too_lo, lo, exact);
        let clamped = self.b.ins().select(too_hi, hi, low);
        let value = self.b.ins().band(clamped, mask);
        let wrapped = self.b.ins().band(exact, mask);
        let changed = self.b.ins().icmp(IntCC::NotEqual, value, wrapped);
        (value, changed)
    }

    /// `SQADD`/`SQSUB`/`SQABS`/`SQNEG` (`Op::Vsat`) and their cumulative `QC` bit
    /// (`Op::VsatFlag`). `esize` and `mode` are literals from the lifter. A 64-bit lane
    /// cannot widen into `i64`, so it is clamped with explicit overflow rules.
    fn vec_sat(
        &mut self,
        op: Op,
        args: &[Expr],
        t: &Temps,
        k: &impl Fn(usize) -> Option<u64>,
    ) -> Value {
        let (Some(esize), Some(mode)) = (k(2), k(3)) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(op, &v);
        };
        if !lane_width(esize) {
            return self.iconst(0);
        }
        let (a, b) = (self.expr(&args[0], t), self.expr(&args[1], t));
        let flag = op == Op::VsatFlag;
        if esize == 64 {
            let (value, changed) = self.sat_lane64(a, b, mode);
            return if flag { self.bool64(changed) } else { value };
        }
        let mask = self.iconst(kirikae_a64::ops::lane_mask(esize));
        let mut out = self.iconst(0);
        let mut i = 0i64;
        while i < 64 {
            let lane = |me: &mut Self, base: Value| {
                let shifted = if i == 0 {
                    base
                } else {
                    me.b.ins().ushr_imm_u(base, i)
                };
                me.b.ins().band(shifted, mask)
            };
            let (x, y) = (lane(self, a), lane(self, b));
            let (value, changed) = self.sat_lane(x, y, esize, mode);
            out = if flag {
                let bit = self.bool64(changed);
                self.b.ins().bor(out, bit)
            } else {
                let placed = if i == 0 {
                    value
                } else {
                    self.b.ins().ishl_imm_u(value, i)
                };
                self.b.ins().bor(out, placed)
            };
            i += esize as i64;
        }
        out
    }

    /// The 64-bit lane cases of `vec_sat`: `sadd`/`ssub`/`sabs`/`sneg` overflow rules
    /// replace the widen-and-clamp of the narrower lanes.
    fn sat_lane64(&mut self, a: Value, b: Value, mode: u64) -> (Value, Value) {
        let zero = self.iconst(0);
        let max = self.iconst(u64::MAX);
        let smax = self.iconst(i64::MAX as u64);
        let smin = self.iconst(1u64 << 63);
        let b_negative = self.b.ins().icmp(IntCC::SignedLessThan, b, zero);
        let overflow = |me: &mut Self, left: Value, right: Value| {
            let t = me.b.ins().band(left, right);
            me.b.ins().icmp(IntCC::SignedLessThan, t, zero)
        };
        match mode {
            1 => {
                let r = self.b.ins().iadd(a, b);
                let over = self.b.ins().icmp(IntCC::UnsignedLessThan, r, a);
                (self.b.ins().select(over, max, r), over)
            }
            3 => {
                let r = self.b.ins().isub(a, b);
                let under = self.b.ins().icmp(IntCC::UnsignedLessThan, a, b);
                (self.b.ins().select(under, zero, r), under)
            }
            2 => {
                let r = self.b.ins().isub(a, b);
                let (x, y) = (self.b.ins().bxor(a, b), self.b.ins().bxor(a, r));
                let over = overflow(self, x, y);
                let sat = self.b.ins().select(b_negative, smax, smin);
                (self.b.ins().select(over, sat, r), over)
            }
            4 => {
                let negated = self.b.ins().isub(zero, a);
                let negative = self.b.ins().icmp(IntCC::SignedLessThan, a, zero);
                let abs = self.b.ins().select(negative, negated, a);
                let over = self.b.ins().icmp(IntCC::Equal, a, smin);
                (self.b.ins().select(over, smax, abs), over)
            }
            5 => {
                let negated = self.b.ins().isub(zero, a);
                let over = self.b.ins().icmp(IntCC::Equal, a, smin);
                (self.b.ins().select(over, smax, negated), over)
            }
            _ => {
                let r = self.b.ins().iadd(a, b);
                let (x, y) = (self.b.ins().bxor(a, r), self.b.ins().bxor(b, r));
                let over = overflow(self, x, y);
                let sat = self.b.ins().select(b_negative, smin, smax);
                (self.b.ins().select(over, sat, r), over)
            }
        }
    }

    /// Lane-wise `MLA`/`MLS`: `d + a * b` or `d - a * b`, isolated per lane and masked,
    /// exactly as `kirikae_a64::ops::vmla`.
    fn vec_mla(
        &mut self,
        op: Op,
        args: &[Expr],
        t: &Temps,
        k: &impl Fn(usize) -> Option<u64>,
    ) -> Value {
        let Some(esize) = k(3) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(op, &v);
        };
        if !lane_width(esize) {
            return self.iconst(0);
        }
        let (a, b, d) = (
            self.expr(&args[0], t),
            self.expr(&args[1], t),
            self.expr(&args[2], t),
        );
        let add = op == Op::Vmla;
        if esize == 64 {
            let product = self.b.ins().imul(a, b);
            return if add {
                self.b.ins().iadd(d, product)
            } else {
                self.b.ins().isub(d, product)
            };
        }
        let mask = self.iconst(kirikae_a64::ops::lane_mask(esize));
        let mut out = self.iconst(0);
        let mut i = 0i64;
        while i < 64 {
            let lane = |me: &mut Self, base: Value| {
                let shifted = if i == 0 {
                    base
                } else {
                    me.b.ins().ushr_imm_u(base, i)
                };
                me.b.ins().band(shifted, mask)
            };
            let (x, y) = (lane(self, a), lane(self, b));
            let product = self.b.ins().imul(x, y);
            let acc = lane(self, d);
            let r = if add {
                self.b.ins().iadd(acc, product)
            } else {
                self.b.ins().isub(acc, product)
            };
            let r = self.b.ins().band(r, mask);
            let placed = if i == 0 {
                r
            } else {
                self.b.ins().ishl_imm_u(r, i)
            };
            out = self.b.ins().bor(out, placed);
            i += esize as i64;
        }
        out
    }

    /// Pairwise maximum or minimum (`Op::Vpmax`/`Op::Vpmin`) over `a | b << 64`, one
    /// result lane per adjacent input pair, mirroring `kirikae_a64::ops::vpmax`.
    fn vec_pair(
        &mut self,
        op: Op,
        args: &[Expr],
        t: &Temps,
        k: &impl Fn(usize) -> Option<u64>,
    ) -> Value {
        let (Some(esize), Some(signed)) = (k(2), k(3)) else {
            let v: Vec<Value> = args.iter().map(|a| self.expr(a, t)).collect();
            return self.call_op(op, &v);
        };
        if !lane_width(esize) {
            return self.iconst(0);
        }
        let (a, b) = (self.expr(&args[0], t), self.expr(&args[1], t));
        let mask = self.iconst(kirikae_a64::ops::lane_mask(esize));
        let signed = signed != 0;
        let max = op == Op::Vpmax;
        let cc = match (max, signed) {
            (true, true) => IntCC::SignedGreaterThan,
            (true, false) => IntCC::UnsignedGreaterThan,
            (false, true) => IntCC::SignedLessThan,
            (false, false) => IntCC::UnsignedLessThan,
        };
        let mut out = self.iconst(0);
        let mut j = 0i64;
        while j < 64 {
            let input = |me: &mut Self, index: i64| {
                let bit = index * esize as i64;
                let word = if bit < 64 { a } else { b };
                let shift = bit % 64;
                let shifted = if shift == 0 {
                    word
                } else {
                    me.b.ins().ushr_imm_u(word, shift)
                };
                me.b.ins().band(shifted, mask)
            };
            let first = 2 * (j / esize as i64);
            let x = input(self, first);
            let y = input(self, first + 1);
            let (cx, cy) = if signed {
                (self.sx_lane(x, esize), self.sx_lane(y, esize))
            } else {
                (x, y)
            };
            let take_x = self.b.ins().icmp(cc, cx, cy);
            let r = self.b.ins().select(take_x, x, y);
            let placed = if j == 0 {
                r
            } else {
                self.b.ins().ishl_imm_u(r, j)
            };
            out = self.b.ins().bor(out, placed);
            j += esize as i64;
        }
        out
    }

    fn bits_as_float(&mut self, v: Value, double: bool) -> Value {
        if double {
            self.b.ins().bitcast(types::F64, MemFlagsData::new(), v)
        } else {
            let x = self.r32(v);
            self.b.ins().bitcast(types::F32, MemFlagsData::new(), x)
        }
    }

    fn float_as_bits(&mut self, f: Value, double: bool) -> Value {
        if double {
            self.b.ins().bitcast(I64, MemFlagsData::new(), f)
        } else {
            let x = self.b.ins().bitcast(I32, MemFlagsData::new(), f);
            self.z64(x)
        }
    }

    fn lanes32(&mut self, v: Value) -> (Value, Value) {
        let lo = self.b.ins().band_imm_u(v, 0xffff_ffff);
        let hi = self.b.ins().ushr_imm_u(v, 32);
        (lo, hi)
    }

    fn no_flush(&mut self, fpcr: Value) -> Value {
        let fz = self.b.ins().band_imm_u(fpcr, 1 << 24);
        self.b.ins().icmp_imm_u(IntCC::Equal, fz, 0)
    }

    fn flush_free(&mut self, inputs: &[Value], result: Option<Value>, double: bool) -> Value {
        let (abs, frac, min) = if double {
            (u64::MAX >> 1, (1u64 << 52) - 1, 1u64 << 52)
        } else {
            (0x7fff_ffff, 0x7f_ffff, 0x80_0000)
        };
        let within = |me: &mut Self, x: Value, limit: u64| {
            let a = me.b.ins().band_imm_u(x, abs as i64);
            let a = me.b.ins().iadd_imm_s(a, -1);
            me.b.ins()
                .icmp_imm_u(IntCC::UnsignedLessThan, a, limit as i64)
        };
        let mut bad = None;
        for &x in inputs.iter() {
            let d = within(self, x, frac);
            bad = Some(match bad {
                Some(b) => self.b.ins().bor(b, d),
                None => d,
            });
        }
        if let Some(r) = result {
            let t = within(self, r, min);
            bad = Some(match bad {
                Some(b) => self.b.ins().bor(b, t),
                None => t,
            });
        }
        let bad = bad.expect("a value to check");
        self.b.ins().icmp_imm_u(IntCC::Equal, bad, 0)
    }

    fn fp_scalar(&mut self, op: Op, v: &[Value]) -> (Value, Option<Value>) {
        let double = fp_double(op);
        let n = fp_inputs(op);
        let f: Vec<Value> = v[..n]
            .iter()
            .map(|&x| self.bits_as_float(x, double))
            .collect();
        let r = match op {
            Op::FAdd32 | Op::FAdd64 => self.b.ins().fadd(f[0], f[1]),
            Op::FSub32 | Op::FSub64 => self.b.ins().fsub(f[0], f[1]),
            Op::FMul32 | Op::FMul64 => self.b.ins().fmul(f[0], f[1]),
            Op::FDiv32 | Op::FDiv64 => self.b.ins().fdiv(f[0], f[1]),
            Op::FMax32 | Op::FMax64 => self.b.ins().fmax(f[0], f[1]),
            Op::FMin32 | Op::FMin64 => self.b.ins().fmin(f[0], f[1]),
            Op::FSqrt32 | Op::FSqrt64 => self.b.ins().sqrt(f[0]),
            Op::FMulAdd32 | Op::FMulAdd64 => self.b.ins().fma(f[1], f[2], f[0]),
            _ => {
                let uno = self.b.ins().fcmp(FloatCC::Unordered, f[0], f[1]);
                let eq = self.b.ins().fcmp(FloatCC::Equal, f[0], f[1]);
                let lt = self.b.ins().fcmp(FloatCC::LessThan, f[0], f[1]);
                let [a, b, less, greater] =
                    [0b0011u64, 0b0110, 0b1000, 0b0010].map(|x| self.iconst(x << 28));
                let r = self.b.ins().select(lt, less, greater);
                let r = self.b.ins().select(eq, b, r);
                return (self.b.ins().select(uno, a, r), None);
            }
        };
        let ordered = self.b.ins().fcmp(FloatCC::Ordered, r, r);
        (self.float_as_bits(r, double), Some(ordered))
    }

    fn fp_or_call(&mut self, op: Op, v: &[Value], fast: Value, ok: Value) -> Value {
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, I64);
        self.b
            .ins()
            .brif(ok, done, &[BlockArg::Value(fast)], slow, &[]);
        self.b.switch_to_block(slow);
        let r = self.call_op(op, v);
        self.b.ins().jump(done, &[BlockArg::Value(r)]);
        self.b.switch_to_block(done);
        self.b.block_params(done)[0]
    }

    fn fp_convert(&mut self, op: Op, x: Value, mode: u64) -> Option<Value> {
        let (signed, width) = (mode >> 3 & 1 == 1, mode >> 8 & 0xff);
        let int = if width == 64 { I64 } else { I32 };
        Some(match op {
            Op::FToInt32 | Op::FToInt64 => {
                if mode & 7 != 4 {
                    return None;
                }
                let f = self.bits_as_float(x, op == Op::FToInt64);
                let r = if signed {
                    self.b.ins().fcvt_to_sint_sat(int, f)
                } else {
                    self.b.ins().fcvt_to_uint_sat(int, f)
                };
                if width == 64 { r } else { self.z64(r) }
            }
            _ => {
                let double = op == Op::FFromInt64;
                let i = if width == 64 { x } else { self.r32(x) };
                let ty = if double { types::F64 } else { types::F32 };
                let f = if signed {
                    self.b.ins().fcvt_from_sint(ty, i)
                } else {
                    self.b.ins().fcvt_from_uint(ty, i)
                };
                self.float_as_bits(f, double)
            }
        })
    }

    fn flags_add(&mut self, a: Value, b: Value, carry: Value, narrow: bool) -> Value {
        let (a, b, cin) = if narrow {
            (self.r32(a), self.r32(b), self.r32(carry))
        } else {
            (a, b, carry)
        };
        let cin = self.b.ins().band_imm_u(cin, 1);
        let r1 = self.b.ins().iadd(a, b);
        let c1 = self.b.ins().icmp(IntCC::UnsignedLessThan, r1, a);
        let r = self.b.ins().iadd(r1, cin);
        let c2 = self.b.ins().icmp(IntCC::UnsignedLessThan, r, r1);
        let c = self.b.ins().bor(c1, c2);
        let ar = self.b.ins().bxor(a, r);
        let br = self.b.ins().bxor(b, r);
        let ov = self.b.ins().band(ar, br);
        let top = if narrow { 31 } else { 63 };
        let v = self.b.ins().ushr_imm_u(ov, top);
        let n = self.b.ins().ushr_imm_u(r, top);
        let z = self.b.ins().icmp_imm_u(IntCC::Equal, r, 0);
        let (n, v) = if narrow {
            (self.z64(n), self.z64(v))
        } else {
            (n, v)
        };
        let (z, c) = (self.bool64(z), self.bool64(c));
        let n = self.b.ins().ishl_imm_u(n, 31);
        let z = self.b.ins().ishl_imm_u(z, 30);
        let c = self.b.ins().ishl_imm_u(c, 29);
        let v = self.b.ins().ishl_imm_u(v, 28);
        let nz = self.b.ins().bor(n, z);
        let cv = self.b.ins().bor(c, v);
        self.b.ins().bor(nz, cv)
    }

    fn op_with_immediates(
        &mut self,
        op: Op,
        args: &[Expr],
        t: &Temps,
        k: &dyn Fn(usize) -> Option<u64>,
    ) -> Option<Value> {
        let i = |v: u64| v as i64;
        Some(match op {
            Op::Ubfm64 | Op::Ubfm32 => {
                let (r, wm, tm) = (k(1)?, k(2)?, k(3)?);
                let src = self.expr(&args[0], t);
                let rot = if op == Op::Ubfm64 {
                    self.b.ins().rotr_imm_u(src, i(r))
                } else {
                    let s = self.r32(src);
                    let r32 = self.b.ins().rotr_imm_u(s, i(r));
                    self.z64(r32)
                };
                self.b.ins().band_imm_u(rot, i(wm & tm))
            }
            Op::Sbfm64 | Op::Sbfm32 => {
                let (r, s, wm, tm) = (k(1)?, k(2)?, k(3)?, k(4)?);
                let src = self.expr(&args[0], t);
                let rot = if op == Op::Sbfm64 {
                    self.b.ins().rotr_imm_u(src, i(r))
                } else {
                    let s32 = self.r32(src);
                    let r32 = self.b.ins().rotr_imm_u(s32, i(r));
                    self.z64(r32)
                };
                let bot = self.b.ins().band_imm_u(rot, i(wm));
                let up = self.b.ins().ishl_imm_u(src, i(63 - s));
                let top = self.b.ins().sshr_imm_u(up, 63);
                let top = self.b.ins().band_imm_u(top, i(!tm));
                let bot = self.b.ins().band_imm_u(bot, i(tm));
                let r = self.b.ins().bor(top, bot);
                if op == Op::Sbfm32 {
                    self.b.ins().band_imm_u(r, 0xffff_ffff)
                } else {
                    r
                }
            }
            Op::Bfm64 | Op::Bfm32 => {
                let (r, wm, tm) = (k(2)?, k(3)?, k(4)?);
                let dst = self.expr(&args[0], t);
                let src = self.expr(&args[1], t);
                let rot = if op == Op::Bfm64 {
                    self.b.ins().rotr_imm_u(src, i(r))
                } else {
                    let s32 = self.r32(src);
                    let r32 = self.b.ins().rotr_imm_u(s32, i(r));
                    self.z64(r32)
                };
                let keep = self.b.ins().band_imm_u(dst, i(!wm));
                let ins = self.b.ins().band_imm_u(rot, i(wm));
                let bot = self.b.ins().bor(keep, ins);
                let keep = self.b.ins().band_imm_u(dst, i(!tm));
                let bot = self.b.ins().band_imm_u(bot, i(tm));
                let res = self.b.ins().bor(keep, bot);
                if op == Op::Bfm32 {
                    self.b.ins().band_imm_u(res, 0xffff_ffff)
                } else {
                    res
                }
            }
            Op::Extr64 | Op::Extr32 => {
                let lsb = k(2)?;
                let hi = self.expr(&args[0], t);
                let lo = self.expr(&args[1], t);
                let width = if op == Op::Extr64 { 64 } else { 32 };
                let (hi, lo) = if width == 32 {
                    (self.r32(hi), self.r32(lo))
                } else {
                    (hi, lo)
                };
                let r = if lsb == 0 {
                    lo
                } else {
                    let l = self.b.ins().ushr_imm_u(lo, i(lsb));
                    let h = self.b.ins().ishl_imm_u(hi, i(width - lsb));
                    self.b.ins().bor(l, h)
                };
                if width == 32 { self.z64(r) } else { r }
            }
            Op::Cond => {
                let cc = k(1)?;
                let nzcv = self.expr(&args[0], t);
                let flag = |l: &mut Self, bit: i64| {
                    let s = l.b.ins().ushr_imm_u(nzcv, bit);
                    l.b.ins().band_imm_u(s, 1)
                };
                let base = match (cc >> 1) & 7 {
                    0 => flag(self, 30),
                    1 => flag(self, 29),
                    2 => flag(self, 31),
                    3 => flag(self, 28),
                    4 => {
                        let c = flag(self, 29);
                        let z = flag(self, 30);
                        let nz = self.b.ins().bxor_imm_u(z, 1);
                        self.b.ins().band(c, nz)
                    }
                    5 | 6 => {
                        let n = flag(self, 31);
                        let v = flag(self, 28);
                        let x = self.b.ins().bxor(n, v);
                        let ge = self.b.ins().bxor_imm_u(x, 1);
                        if (cc >> 1) & 7 == 5 {
                            ge
                        } else {
                            let z = flag(self, 30);
                            let nz = self.b.ins().bxor_imm_u(z, 1);
                            self.b.ins().band(ge, nz)
                        }
                    }
                    _ => self.iconst(1),
                };
                if cc & 1 == 1 && cc != 15 {
                    self.b.ins().bxor_imm_u(base, 1)
                } else {
                    base
                }
            }
            _ => return None,
        })
    }

    fn store_exclusive(&mut self, size: u8, addr: Value, lo: Value, hi: Value) -> Value {
        if size == 16 {
            let size = self.b.ins().iconst(I32, 16);
            let call = self
                .b
                .ins()
                .call(self.h.store_exclusive, &[self.cpu, size, addr, lo, hi]);
            return self.b.inst_results(call)[0];
        }
        let armed = self.load_cpu(abi::OFF_EXCL_ADDR);
        let expected = self.load_cpu(abi::OFF_EXCL_LO);
        let none = self.iconst(kirikae_rt::cpu::NO_MONITOR);
        self.store_cpu(abi::OFF_EXCL_ADDR, none);
        let same = self.b.ins().icmp(IntCC::Equal, addr, armed);
        let (try_cas, done) = (self.b.create_block(), self.b.create_block());
        self.b.append_block_param(done, I64);
        let failed = self.iconst(1);
        self.b
            .ins()
            .brif(same, try_cas, &[], done, &[BlockArg::Value(failed)]);
        self.b.switch_to_block(try_cas);
        let ty = match size {
            1 => types::I8,
            2 => types::I16,
            4 => I32,
            _ => I64,
        };
        let (e, x) = if ty == I64 {
            (expected, lo)
        } else {
            (
                self.b.ins().ireduce(ty, expected),
                self.b.ins().ireduce(ty, lo),
            )
        };
        let p = self.host(addr);
        let old = self.b.ins().atomic_cas(guest_flags(), p, e, x);
        let changed = self.b.ins().icmp(IntCC::NotEqual, old, e);
        let status = self.bool64(changed);
        self.b.ins().jump(done, &[BlockArg::Value(status)]);
        self.b.switch_to_block(done);
        self.b.block_params(done)[0]
    }

    fn poll_then(&mut self, dest: ir::Block, args: &[BlockArg], reload: Regs) {
        let budget = self
            .b
            .ins()
            .load(I32, cpu_flags(), self.cpu, abi::OFF_BUDGET as i32);
        let left = self.b.ins().iadd_imm_s(budget, -1);
        self.b
            .ins()
            .store(cpu_flags(), left, self.cpu, abi::OFF_BUDGET as i32);
        let slow = self.b.create_block();
        self.b.ins().brif(left, dest, args, slow, &[]);
        self.b.switch_to_block(slow);
        let dirty = self.cache.as_ref().map(|c| c.state.dirty);
        if let Some(d) = dirty {
            self.store_vars(d);
        }
        let call = self.b.ins().call(self.h.poll_slow, &[self.cpu]);
        let stop = self.b.inst_results(call)[0];
        if dirty.is_some() {
            self.load_vars(reload);
        }
        self.b.ins().brif(stop, self.ret, &[], dest, args);
    }

    fn goto(&mut self, target: u64) {
        let saved = self.cache.as_ref().map(|c| c.state);
        if let Some(&blk) = self.blocks.get(&target) {
            self.edge(target);
            if target <= self.block_start {
                let reload = self.cache.as_ref().map_or(0, |c| c.entry[&target].valid);
                self.poll_then(blk, &[], reload);
            } else {
                self.b.ins().jump(blk, &[]);
            }
        } else {
            self.set_pc_const(target);
            self.leave();
        }
        if let (Some(c), Some(st)) = (&mut self.cache, saved) {
            c.state = st;
        }
    }

    fn after_call(&mut self, ret_addr: u64) {
        let pc = self.load_cpu(abi::OFF_PC);
        let wrong = self
            .b
            .ins()
            .icmp_imm_u(IntCC::NotEqual, pc, ret_addr as i64);
        let (resume, cont) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(wrong, resume, &[], cont, &[]);
        self.b.switch_to_block(resume);
        self.sync();
        let ret = self.iconst(ret_addr);
        self.b.ins().call(self.h.call_indirect, &[self.cpu, ret]);
        let pc = self.load_cpu(abi::OFF_PC);
        let still_wrong = self
            .b
            .ins()
            .icmp_imm_u(IntCC::NotEqual, pc, ret_addr as i64);
        self.b.ins().brif(still_wrong, self.ret, &[], cont, &[]);
        self.b.switch_to_block(cont);
        self.goto(ret_addr);
    }

    fn insn(&mut self, insn: &Insn) -> bool {
        let mut t: Temps = [None; 8];
        for s in &insn.stmts {
            match s {
                Stmt::Let(n, e) => t[*n as usize] = Some(self.expr(e, &t)),
                Stmt::SetReg(n, e) => {
                    let v = self.expr(e, &t);
                    self.write(x_slot(*n), v);
                }
                Stmt::SetSp(e) => {
                    let v = self.expr(e, &t);
                    self.write(SP_SLOT, v);
                }
                Stmt::SetNzcv(e) => {
                    let v = self.expr(e, &t);
                    self.write(NZCV_SLOT, v);
                }
                Stmt::SetV { reg, hi, value } => {
                    let v = self.expr(value, &t);
                    self.write(v_slot(*reg, *hi), v);
                }
                Stmt::SetSysReg(SysReg::Fpsr, e) => {
                    let v = self.expr(e, &t);
                    self.store_cpu(abi::OFF_FPSR, v);
                }
                Stmt::SetSysReg(r, e) => {
                    let v = self.expr(e, &t);
                    let id = self.b.ins().iconst(I32, *r as i64);
                    self.b.ins().call(self.h.set_sysreg, &[self.cpu, id, v]);
                }
                Stmt::Store { size, addr, value } => {
                    let a = self.expr(addr, &t);
                    let v = self.expr(value, &t);
                    let p = self.host(a);
                    match size {
                        8 => self.b.ins().store(guest_flags(), v, p, 0),
                        4 => self.b.ins().istore32(guest_flags(), v, p, 0),
                        2 => self.b.ins().istore16(guest_flags(), v, p, 0),
                        _ => self.b.ins().istore8(guest_flags(), v, p, 0),
                    };
                }
                Stmt::Monitor { addr, lo, hi } => {
                    let (a, l, h) = (self.expr(addr, &t), self.expr(lo, &t), self.expr(hi, &t));
                    self.store_cpu(abi::OFF_EXCL_ADDR, a);
                    self.store_cpu(abi::OFF_EXCL_LO, l);
                    self.store_cpu(abi::OFF_EXCL_HI, h);
                }
                Stmt::StoreExclusive {
                    size,
                    addr,
                    lo,
                    hi,
                    status,
                } => {
                    let (a, l, h) = (self.expr(addr, &t), self.expr(lo, &t), self.expr(hi, &t));
                    let r = self.store_exclusive(*size, a, l, h);
                    if *status != 31 {
                        self.write(x_slot(*status), r);
                    }
                }
                Stmt::Fence => {
                    self.b.ins().fence();
                }
                Stmt::Simd(word) => {
                    self.flush();
                    if let Some(c) = &mut self.cache {
                        c.state.valid &= !simd_writes(*word);
                    }
                    let word = self.b.ins().iconst(I32, *word as i64);
                    self.b.ins().call(self.h.simd, &[self.cpu, word]);
                }
                Stmt::Svc(imm) => {
                    let next = insn.next();
                    self.set_pc_const(next);
                    self.sync();
                    let imm = self.b.ins().iconst(I32, *imm as i64);
                    self.b.ins().call(self.h.svc, &[self.cpu, imm]);
                    let halted =
                        self.b
                            .ins()
                            .uload8(I64, cpu_flags(), self.cpu, abi::OFF_HALTED as i32);
                    let pc = self.load_cpu(abi::OFF_PC);
                    let moved = self.b.ins().icmp_imm_u(IntCC::NotEqual, pc, next as i64);
                    let moved = self.bool64(moved);
                    let leave = self.b.ins().bor(halted, moved);
                    let cont = self.b.create_block();
                    self.b.ins().brif(leave, self.ret, &[], cont, &[]);
                    self.b.switch_to_block(cont);
                }
                Stmt::Undefined(word) => {
                    self.set_pc_const(insn.addr);
                    self.sync();
                    let addr = self.iconst(insn.addr);
                    let word = self.b.ins().iconst(I32, *word as i64);
                    self.b.ins().call(self.h.undefined, &[self.cpu, addr, word]);
                    self.b.ins().jump(self.ret, &[]);
                    return true;
                }
                Stmt::Return(e) => {
                    let v = self.expr(e, &t);
                    self.set_pc(v);
                    self.leave();
                    return true;
                }
                Stmt::Jump(target) => {
                    self.goto(*target);
                    return true;
                }
                Stmt::JumpIf(cond, target) => {
                    let c = self.expr(cond, &t);
                    let (taken, not) = (self.b.create_block(), self.b.create_block());
                    self.b.ins().brif(c, taken, &[], not, &[]);
                    self.b.switch_to_block(taken);
                    self.goto(*target);
                    self.b.switch_to_block(not);
                    self.goto(insn.next());
                    return true;
                }
                Stmt::JumpIndirect(e) => {
                    let v = self.expr(e, &t);
                    let d = self
                        .dispatch
                        .expect("dispatch block exists for indirect jumps");
                    self.sync();
                    self.poll_then(d, &[BlockArg::Value(v)], 0);
                    return true;
                }
                Stmt::Call(target) => {
                    self.set_pc_const(*target);
                    self.sync();
                    match self.callees.get(target) {
                        Some(&f) => {
                            self.b.ins().call(f, &[self.cpu]);
                        }
                        None => {
                            let ret = self.iconst(insn.next());
                            self.b.ins().call(self.h.call_indirect, &[self.cpu, ret]);
                        }
                    }
                    self.after_call(insn.next());
                    return true;
                }
                Stmt::CallIndirect(e) => {
                    let v = self.expr(e, &t);
                    self.set_pc(v);
                    self.sync();
                    let ret = self.iconst(insn.next());
                    self.b.ins().call(self.h.call_indirect, &[self.cpu, ret]);
                    self.after_call(insn.next());
                    return true;
                }
            }
        }
        false
    }
}

fn object_compiler(isa: OwnedTargetIsa, name: &str) -> Result<Compiler<ObjectModule>, String> {
    let builder = ObjectBuilder::new(isa, name, cranelift_module::default_libcall_names())
        .map_err(|e| e.to_string())?;
    Compiler::new(ObjectModule::new(builder))
}

fn entries_object(isa: OwnedTargetIsa, entries: &BTreeSet<u64>) -> Result<Vec<u8>, String> {
    let mut c = object_compiler(isa, "kirikae_entries")?;
    let err = |e: cranelift_module::ModuleError| e.to_string();
    let table = c
        .module
        .declare_data("kirikae_entries", Linkage::Export, false, false)
        .map_err(err)?;
    let count = c
        .module
        .declare_data("kirikae_entry_count", Linkage::Export, false, false)
        .map_err(err)?;
    let mut desc = DataDescription::new();
    let mut bytes = vec![0u8; 16 * entries.len()];
    for (i, addr) in entries.iter().enumerate() {
        bytes[16 * i..16 * i + 8].copy_from_slice(&addr.to_le_bytes());
    }
    desc.define(bytes.into_boxed_slice());
    desc.set_align(8);
    for (i, addr) in entries.iter().enumerate() {
        let id = c.guest(*addr, Linkage::Import)?;
        let f = c.module.declare_func_in_data(id, &mut desc);
        desc.write_function_addr((16 * i + 8) as u32, f);
    }
    c.module.define_data(table, &desc).map_err(err)?;
    let mut desc = DataDescription::new();
    desc.define(
        (entries.len() as u64)
            .to_le_bytes()
            .to_vec()
            .into_boxed_slice(),
    );
    desc.set_align(8);
    c.module.define_data(count, &desc).map_err(err)?;
    c.module.finish().emit().map_err(|e| e.to_string())
}

/// How a call to a function that the current object does not hold is emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Calls {
    /// Declare it as an import: the linker resolves it in the single module (the default).
    Import,
    /// Leave it to the dispatcher. Required when the objects end up in separate DLLs, which
    /// cannot reference each other's symbols.
    Dispatch,
}

pub struct CompileReport {
    pub objects: Vec<PathBuf>,
    /// Size in bytes of each object in `objects`, measured after writing it.
    pub sizes: Vec<u64>,
    /// The address range of each chunk in `objects`, in the order the entry table sorts them.
    pub chunks: Vec<std::ops::Range<usize>>,
    pub functions: usize,
    pub instructions: u64,
}

/// Instructions the backend will lift, without compiling them. `build` uses it to decide up
/// front whether the image has to be split, which is what fixes each object's call scope.
pub fn estimate(space: &CodeSpace, entries: &BTreeSet<u64>) -> u64 {
    entries
        .iter()
        .map(|&e| build_function(space, e, entries).insn_count() as u64)
        .sum()
}

/// Write the `kirikae_entries` table for `addrs` as `<file>` inside `dir`.
pub fn write_entries(dir: &Path, file: &str, addrs: &BTreeSet<u64>) -> Result<PathBuf, String> {
    let isa = object_isa()?;
    let path = dir.join(file);
    std::fs::write(&path, entries_object(isa, addrs)?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// How `compile_program` lays the objects out and what it emits for calls.
pub struct CompileOptions {
    pub per_object: usize,
    pub jobs: usize,
    pub fast_regs: bool,
    pub calls: Calls,
}

pub fn compile_program(
    space: &CodeSpace,
    entries: &BTreeSet<u64>,
    dir: &Path,
    opts: &CompileOptions,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Result<CompileReport, String> {
    let CompileOptions {
        per_object,
        jobs,
        fast_regs,
        calls,
    } = *opts;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let isa = object_isa()?;
    let list: Vec<u64> = entries.iter().copied().collect();
    let chunks: Vec<&[u64]> = list.chunks(per_object.max(1)).collect();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let instructions = AtomicUsize::new(0);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let obj_path = |i: usize| dir.join(format!("chunk{i:05}.obj"));

    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= chunks.len() || failure.lock().unwrap().is_some() {
                        return;
                    }
                    let result = (|| -> Result<(), String> {
                        let mut c = object_compiler(isa.clone(), &format!("chunk{i}"))?;
                        c.set_fast_regs(fast_regs);
                        let in_scope = |t: u64| match calls {
                            Calls::Import => true,
                            Calls::Dispatch => chunks[i].binary_search(&t).is_ok(),
                        };
                        for &entry in chunks[i] {
                            let f = build_function(space, entry, entries);
                            instructions.fetch_add(f.insn_count(), Ordering::Relaxed);
                            c.compile(&f, entries, &in_scope)?;
                        }
                        let bytes = c.module.finish().emit().map_err(|e| e.to_string())?;
                        std::fs::write(obj_path(i), bytes).map_err(|e| e.to_string())
                    })();
                    if let Err(e) = result {
                        *failure.lock().unwrap() = Some(e);
                        return;
                    }
                    progress(done.fetch_add(1, Ordering::Relaxed) + 1, chunks.len());
                }
            });
        }
    });
    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }
    let mut objects: Vec<PathBuf> = (0..chunks.len()).map(obj_path).collect();
    let table = write_entries(dir, "entries.obj", entries)?;
    objects.push(table);
    let sizes = objects
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .collect();
    let step = per_object.max(1);
    let ranges = (0..chunks.len())
        .map(|i| i * step..i * step + chunks[i].len())
        .collect();
    Ok(CompileReport {
        objects,
        sizes,
        chunks: ranges,
        functions: entries.len(),
        instructions: instructions.into_inner() as u64,
    })
}
