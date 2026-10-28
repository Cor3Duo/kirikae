use std::collections::{BTreeMap, HashMap};

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, Dim, ExecutionMode, ExecutionModel, FunctionControl, ImageFormat, ImageOperands, LoopControl,
    MemoryModel, SelectionControl, StorageClass, Word,
};

use crate::cfg::{self, Program};
use crate::decode::{HalfOperand, HalfSrc, Insn, PT, RZ};
use crate::opcodes::Form;
use crate::sph::{self, Header, Interpolation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Kind {
    #[default]
    Float,
    Sint,
    Uint,
}

pub trait Env {
    fn cbuf(&self, _slot: u32, _offset: u32) -> Option<u32> {
        None
    }
}

pub struct NoEnv;
impl Env for NoEnv {}

pub const STORAGE_DESCRIPTORS: u32 = 0x110;
pub const STORAGE_SLOTS: u32 = 16;

pub fn storage_window(stage: Stage) -> std::ops::Range<u32> {
    match stage {
        Stage::Vertex => STORAGE_DESCRIPTORS..STORAGE_DESCRIPTORS + 16 * STORAGE_SLOTS,
        Stage::Fragment => STORAGE_DESCRIPTORS + 0x400..STORAGE_DESCRIPTORS + 0x400 + 16 * STORAGE_SLOTS,
        Stage::Compute => 0x260..0x410,
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub stage: Stage,
    pub cbuf_binding: u32,
    pub texture_binding: u32,
    pub storage_binding: u32,
    pub attribute_types: [Kind; 32],
    pub output_types: [Kind; 8],
    pub debug_color: Option<[f32; 4]>,
    pub debug_probe: Option<(u32, u8)>,
    pub debug_probe_signed: bool,
    pub local_size: [u32; 3],
    pub local_memory: u32,
    pub shared_memory: u32,
    pub image_binding: u32,
    pub depth_transform: Option<(f32, f32)>,
    pub depth_unorm16: bool,
    pub alpha_test: bool,
    pub exact_texel_rounding: bool,
}

impl Options {
    pub fn new(stage: Stage, base: u32) -> Options {
        Options {
            stage,
            cbuf_binding: base,
            texture_binding: base + 40,
            storage_binding: base + 18,
            attribute_types: [Kind::Float; 32],
            output_types: [Kind::Float; 8],
            debug_color: None,
            debug_probe: None,
            debug_probe_signed: false,
            local_size: [1; 3],
            local_memory: 0,
            shared_memory: 0,
            image_binding: base + 104,
            depth_transform: None,
            depth_unorm16: false,
            alpha_test: false,
            exact_texel_rounding: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexDim {
    D1,
    D2,
    D3,
    Cube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureUse {
    pub slot: Option<u32>,
    pub handle: u32,
    pub dim: TexDim,
    pub array: bool,
    pub shadow: bool,
    pub binding: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageUse {
    pub slot: Option<u32>,
    pub handle: u32,
    pub dim: TexDim,
    pub array: bool,
    pub binding: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Shader {
    pub spirv: Vec<u32>,
    pub cbufs: u32,
    pub cbuf_used: [u32; 32],
    pub storage: u32,
    pub storage_offsets: Vec<u32>,
    pub storage_written: u32,
    pub textures: Vec<TextureUse>,
    pub images: Vec<ImageUse>,
    pub inputs: u32,
    pub outputs: u32,
    pub color_outputs: u32,
    pub writes_depth: bool,
    pub unsupported: Vec<&'static str>,
    pub mnemonics: Vec<(&'static str, u32)>,
    pub brx: Vec<(u32, Vec<u32>)>,
    pub probe: bool,
}

#[derive(Debug)]
pub enum Error {
    NoHeader,
    Spirv(String),
}

impl From<rspirv::dr::Error> for Error {
    fn from(e: rspirv::dr::Error) -> Self {
        Error::Spirv(format!("{e:?}"))
    }
}

type R<T = ()> = Result<T, Error>;

const EXIT_BLOCK: u32 = u32::MAX;
const STACK_DEPTH: u32 = 16;
const CBUF_VEC4S: u32 = 4096;

mod attr {
    pub const POINT_SIZE: u32 = 0x6c;
    pub const POSITION: u32 = 0x70;
    pub const GENERIC: u32 = 0x80;
    pub const GENERIC_END: u32 = 0x280;
    pub const POINT_COORD: u32 = 0x2e0;
    pub const INSTANCE_ID: u32 = 0x2f8;
    pub const VERTEX_ID: u32 = 0x2fc;
    pub const FRONT_FACING: u32 = 0x3fc;
}

struct Types {
    void: Word,
    bool: Word,
    u32: Word,
    i32: Word,
    f32: Word,
    v2f: Word,
    v4f: Word,
    v2u: Word,
    v3u: Word,
    v4u: Word,
    v4i: Word,
    fn_u32: Word,
    fn_bool: Word,
}

struct Io {
    var: Word,
    kind: Kind,
}

struct Translator<'a> {
    b: Builder,
    opts: &'a Options,
    header: Header,
    glsl: Word,
    t: Types,
    consts: HashMap<(Word, u32), Word>,
    cc: [Word; 4],
    subgroups: bool,
    quad_ops: bool,
    derivative_control: bool,
    sampled_1d: bool,
    gather_extended: bool,
    regs: Vec<Word>,
    preds: Vec<Word>,
    inputs: BTreeMap<u32, Io>,
    outputs: BTreeMap<u32, Io>,
    probe: Option<[Word; 6]>,
    builtins: HashMap<u32, Word>,
    interface: Vec<Word>,
    cbufs: BTreeMap<u32, Word>,
    textures: Vec<(TextureUse, Word)>,
    images: Vec<(ImageUse, Word)>,
    local: Option<Word>,
    shared: Option<(Word, Word)>,
    storage: BTreeMap<u32, Word>,
    storage_written: std::collections::BTreeSet<u32>,
    storage_descs: Vec<u32>,
    flat_decorated: Vec<Word>,
    insns: BTreeMap<u32, Insn>,
    brx: BTreeMap<u32, Vec<u32>>,
    pc_var: Option<Word>,
    ssy: Option<(Word, Word)>,
    pbk: Option<(Word, Word)>,
    block_ids: HashMap<u32, u32>,
    dispatch_merge: Option<Word>,
    ended: bool,
    unsupported: Vec<&'static str>,
    shader: Shader,
    storage_images: bool,
}

pub fn translate(code_with_header: &[u8], opts: &Options, env: &dyn Env) -> Result<Shader, Error> {
    let header = Header::parse(code_with_header).ok_or(Error::NoHeader)?;
    let code = &code_with_header[sph::SIZE..];
    let program = cfg::scan_with(code, &|slot, offset| env.cbuf(slot, offset));
    let mut t = Translator::new(opts, header);
    t.insns = program.insns.clone();
    t.brx = program.brx.clone();
    t.program(&program)?;
    for &(_, w) in &program.unknown {
        t.unsupported.push(if w == 0 { "(zero word)" } else { "(unknown encoding)" });
    }
    let mut shader = std::mem::take(&mut t.shader);
    shader.textures = t.textures.iter().map(|(u, _)| *u).collect();
    shader.images = t.images.iter().map(|(u, _)| *u).collect();
    shader.cbufs = t.cbufs.keys().fold(0, |m, s| m | 1 << s);
    shader.storage = t.storage.keys().filter(|&&s| s < STORAGE_SLOTS).fold(0, |m, s| m | 1 << s);
    shader.storage_offsets = std::mem::take(&mut t.storage_descs);
    shader.storage_written = t.storage_written.iter().fold(0, |m, s| m | 1 << s);
    t.unsupported.sort_unstable();
    t.unsupported.dedup();
    shader.unsupported = std::mem::take(&mut t.unsupported);
    let mut counts: std::collections::BTreeMap<&'static str, u32> = Default::default();
    for i in program.insns.values() {
        *counts.entry(i.name()).or_default() += 1;
    }
    shader.mnemonics = counts.into_iter().collect();
    shader.brx = program.brx.iter().map(|(k, v)| (*k, v.clone())).collect();
    shader.spirv = t.b.module().assemble();
    Ok(shader)
}

impl<'a> Translator<'a> {
    fn kill(&mut self) -> R {
        if let Some(pc_var) = self.pc_var {
            let exit = self.u(EXIT_BLOCK);
            self.b.store(pc_var, exit, None, [])?;
        }
        self.b.kill()?;
        Ok(())
    }

    fn new(opts: &'a Options, header: Header) -> Self {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(Capability::Shader);
        b.capability(Capability::ImageQuery);
        let glsl = b.ext_inst_import("GLSL.std.450");
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void = b.type_void();
        let bool_ = b.type_bool();
        let u32_ = b.type_int(32, 0);
        let i32_ = b.type_int(32, 1);
        let f32_ = b.type_float(32, None);
        let v2f = b.type_vector(f32_, 2);
        let v4f = b.type_vector(f32_, 4);
        let v2u = b.type_vector(u32_, 2);
        let v3u = b.type_vector(u32_, 3);
        let v4u = b.type_vector(u32_, 4);
        let v4i = b.type_vector(i32_, 4);
        let fn_u32 = b.type_pointer(None, StorageClass::Function, u32_);
        let fn_bool = b.type_pointer(None, StorageClass::Function, bool_);
        let t = Types { void, bool: bool_, u32: u32_, i32: i32_, f32: f32_, v2f, v4f, v2u, v3u, v4u, v4i, fn_u32, fn_bool };
        Translator {
            b,
            opts,
            header,
            glsl,
            t,
            consts: HashMap::new(),
            cc: [0; 4],
            subgroups: false,
            quad_ops: false,
            derivative_control: false,
            sampled_1d: false,
            gather_extended: false,
            regs: Vec::new(),
            preds: Vec::new(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            probe: None,
            builtins: HashMap::new(),
            interface: Vec::new(),
            cbufs: BTreeMap::new(),
            textures: Vec::new(),
            images: Vec::new(),
            storage_images: false,
            local: None,
            shared: None,
            storage: BTreeMap::new(),
            storage_written: std::collections::BTreeSet::new(),
            storage_descs: Vec::new(),
            flat_decorated: Vec::new(),
            insns: BTreeMap::new(),
            brx: BTreeMap::new(),
            pc_var: None,
            ssy: None,
            pbk: None,
            block_ids: HashMap::new(),
            dispatch_merge: None,
            ended: false,
            unsupported: Vec::new(),
            shader: Shader::default(),
        }
    }

    fn constant(&mut self, ty: Word, bits: u32) -> Word {
        if let Some(&c) = self.consts.get(&(ty, bits)) {
            return c;
        }
        let c = self.b.constant_bit32(ty, bits);
        self.consts.insert((ty, bits), c);
        c
    }

    fn global(&mut self, ptr: Word, class: StorageClass) -> Word {
        let id = self.b.id();
        let inst = rspirv::dr::Instruction::new(rspirv::spirv::Op::Variable, Some(ptr), Some(id), vec![Operand::StorageClass(class)]);
        self.b.insert_types_global_values(rspirv::dr::InsertPoint::End, inst);
        id
    }

    fn u(&mut self, v: u32) -> Word {
        self.constant(self.t.u32, v)
    }

    fn f(&mut self, v: f32) -> Word {
        self.constant(self.t.f32, v.to_bits())
    }

    fn btrue(&mut self) -> Word {
        self.b.constant_true(self.t.bool)
    }

    fn r(&mut self, reg: u8) -> R<Word> {
        if reg == RZ {
            return Ok(self.u(0));
        }
        Ok(self.b.load(self.t.u32, None, self.regs[reg as usize], None, [])?)
    }

    fn rf(&mut self, reg: u8) -> R<Word> {
        let v = self.r(reg)?;
        Ok(self.b.bitcast(self.t.f32, None, v)?)
    }

    fn set(&mut self, reg: u8, v: Word) -> R {
        if reg != RZ {
            self.b.store(self.regs[reg as usize], v, None, [])?;
        }
        Ok(())
    }

    fn setf(&mut self, reg: u8, v: Word) -> R {
        let u = self.b.bitcast(self.t.u32, None, v)?;
        self.set(reg, u)
    }

    fn p(&mut self, idx: u8, not: bool) -> R<Word> {
        let v = if idx == PT { self.btrue() } else { self.b.load(self.t.bool, None, self.preds[idx as usize], None, [])? };
        Ok(if not { self.b.logical_not(self.t.bool, None, v)? } else { v })
    }

    fn setp(&mut self, idx: u8, v: Word) -> R {
        if idx != PT {
            self.b.store(self.preds[idx as usize], v, None, [])?;
        }
        Ok(())
    }

    fn ext(&mut self, ty: Word, inst: u32, args: &[Word]) -> R<Word> {
        let ops: Vec<Operand> = args.iter().map(|&a| Operand::IdRef(a)).collect();
        Ok(self.b.ext_inst(ty, None, self.glsl, inst, ops)?)
    }

    fn exact(&mut self, v: Word) {
        self.b.decorate(v, Decoration::NoContraction, []);
    }

    fn fneg_abs(&mut self, v: Word, neg: bool, abs: bool) -> R<Word> {
        let mut v = v;
        if abs {
            v = self.ext(self.t.f32, glsl::FABS, &[v])?;
        }
        if neg {
            v = self.b.f_negate(self.t.f32, None, v)?;
        }
        Ok(v)
    }

    fn saturate(&mut self, v: Word, sat: bool) -> R<Word> {
        if !sat {
            return Ok(v);
        }
        let (z, o) = (self.f(0.0), self.f(1.0));
        self.ext(self.t.f32, glsl::FCLAMP, &[v, z, o])
    }

    fn b2u(&mut self, cond: Word, t: u32, f: u32) -> R<Word> {
        let (a, b) = (self.u(t), self.u(f));
        Ok(self.b.select(self.t.u32, None, cond, a, b)?)
    }

    fn cbuf_at(&mut self, slot: u32, offset: Word) -> R<Word> {
        let var = self.cbuf_var(slot);
        let four = self.u(4);
        let vec = self.b.shift_right_logical(self.t.u32, None, offset, four)?;
        let two = self.u(2);
        let w = self.b.shift_right_logical(self.t.u32, None, offset, two)?;
        let three = self.u(3);
        let comp = self.b.bitwise_and(self.t.u32, None, w, three)?;
        let zero = self.u(0);
        let ptr_ty = self.b.type_pointer(None, StorageClass::Uniform, self.t.u32);
        let ptr = self.b.access_chain(ptr_ty, None, var, [zero, vec, comp])?;
        Ok(self.b.load(self.t.u32, None, ptr, None, [])?)
    }

    fn cbuf_imm(&mut self, slot: u32, offset: u32) -> R<Word> {
        if let Some(used) = self.shader.cbuf_used.get_mut(slot as usize) {
            *used = (*used).max((offset & !3) + 4);
        }
        let off = self.u(offset);
        self.cbuf_at(slot, off)
    }

    fn cbuf_var(&mut self, slot: u32) -> Word {
        if let Some(&v) = self.cbufs.get(&slot) {
            return v;
        }
        let len = self.u(CBUF_VEC4S);
        let arr = self.b.type_array(self.t.v4u, len);
        self.b.decorate(arr, Decoration::ArrayStride, [Operand::LiteralBit32(16)]);
        let st = self.b.type_struct([arr]);
        self.b.decorate(st, Decoration::Block, []);
        self.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let ptr = self.b.type_pointer(None, StorageClass::Uniform, st);
        let var = self.global(ptr, StorageClass::Uniform);
        self.b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(self.opts.cbuf_binding + slot)]);
        self.cbufs.insert(slot, var);
        var
    }

    fn src_b(&mut self, i: &Insn) -> R<Word> {
        match i.form() {
            Form::Reg => self.r(i.rb()),
            Form::Cbuf => {
                let (s, o) = i.cbuf();
                self.cbuf_imm(s, o)
            }
            Form::Imm => Ok(self.u(i.simm20() as u32)),
        }
    }

    fn src_bf(&mut self, i: &Insn) -> R<Word> {
        match i.form() {
            Form::Imm => {
                let bits = self.u(i.fimm20());
                Ok(self.b.bitcast(self.t.f32, None, bits)?)
            }
            _ => {
                let v = self.src_b(i)?;
                Ok(self.b.bitcast(self.t.f32, None, v)?)
            }
        }
    }

    fn src_bc(&mut self, i: &Insn, cbuf_in_c: bool) -> R<(Word, Word)> {
        if i.form() == Form::Cbuf && cbuf_in_c {
            let b = self.r(i.rc())?;
            let (s, o) = i.cbuf();
            let c = self.cbuf_imm(s, o)?;
            return Ok((b, c));
        }
        let b = self.src_b(i)?;
        let c = self.r(i.rc())?;
        Ok((b, c))
    }

    fn cast_f(&mut self, v: Word) -> R<Word> {
        Ok(self.b.bitcast(self.t.f32, None, v)?)
    }

    fn cast_u(&mut self, v: Word) -> R<Word> {
        Ok(self.b.bitcast(self.t.u32, None, v)?)
    }

    fn cast_i(&mut self, v: Word) -> R<Word> {
        Ok(self.b.bitcast(self.t.i32, None, v)?)
    }

    fn program(&mut self, program: &Program) -> R {
        let model = match self.opts.stage {
            Stage::Vertex => ExecutionModel::Vertex,
            Stage::Fragment => ExecutionModel::Fragment,
            Stage::Compute => ExecutionModel::GLCompute,
        };
        let fn_ty = self.b.type_function(self.t.void, []);
        let main = self.b.begin_function(self.t.void, None, FunctionControl::NONE, fn_ty)?;
        self.b.begin_block(None)?;
        for _ in 0..255 {
            let v = self.b.variable(self.t.fn_u32, None, StorageClass::Function, None);
            self.regs.push(v);
        }
        for _ in 0..7 {
            let v = self.b.variable(self.t.fn_bool, None, StorageClass::Function, None);
            self.preds.push(v);
        }
        for k in 0..4 {
            self.cc[k] = self.b.variable(self.t.fn_bool, None, StorageClass::Function, None);
        }
        if self.opts.debug_probe.is_some() {
            self.probe = Some([(); 6].map(|_| self.b.variable(self.t.fn_u32, None, StorageClass::Function, None)));
        }
        let local_size = if self.opts.stage == Stage::Compute { self.opts.local_memory } else { self.header.local_memory_size() };
        let local_words = local_size.div_ceil(4).min(0x4000);
        if local_words > 0 {
            let len = self.u(local_words);
            let arr = self.b.type_array(self.t.u32, len);
            let ptr = self.b.type_pointer(None, StorageClass::Function, arr);
            self.local = Some(self.b.variable(ptr, None, StorageClass::Function, None));
        }
        let shared_words = self.opts.shared_memory.div_ceil(4).min(0x4000);
        if self.opts.stage == Stage::Compute && shared_words > 0 {
            let len = self.u(shared_words);
            let arr = self.b.type_array(self.t.u32, len);
            let ptr = self.b.type_pointer(None, StorageClass::Workgroup, arr);
            let elem = self.b.type_pointer(None, StorageClass::Workgroup, self.t.u32);
            self.shared = Some((self.global(ptr, StorageClass::Workgroup), elem));
        }

        if program.branches() {
            self.dispatcher(program)?;
        } else {
            self.init_state()?;
            for (_, block) in program.blocks() {
                for (pc, i) in block {
                    self.insn(pc, &i)?;
                }
            }
        }
        if self.opts.stage == Stage::Fragment {
            self.fragment_outputs()?;
        }
        if self.opts.stage == Stage::Vertex && (self.opts.depth_transform.is_some() || self.opts.depth_unorm16) {
            let v4f = self.t.v4f;
            let position = self.builtin(BuiltIn::Position, v4f, true);
            let f32_ = self.t.f32;
            let ptr_ty = self.b.type_pointer(None, StorageClass::Output, f32_);
            let (two, three) = (self.u(2), self.u(3));
            let zp = self.b.access_chain(ptr_ty, None, position, [two])?;
            let wp = self.b.access_chain(ptr_ty, None, position, [three])?;
            let mut z = self.b.load(f32_, None, zp, None, [])?;
            let w = self.b.load(f32_, None, wp, None, [])?;
            if let Some((scale, offset)) = self.opts.depth_transform {
                let (scale, offset) = (self.f(scale), self.f(offset));
                let wo = self.b.f_mul(f32_, None, w, offset)?;
                self.exact(wo);
                z = self.ext(f32_, glsl::FMA, &[z, scale, wo])?;
            }
            if self.opts.depth_unorm16 {
                let window = self.b.f_div(f32_, None, z, w)?;
                let snapped = self.snap_unorm16(window)?;
                let snapped = self.b.f_mul(f32_, None, snapped, w)?;
                let zero = self.f(0.0);
                let bool_ = self.t.bool;
                let positive = self.b.f_ord_greater_than(bool_, None, w, zero)?;
                z = self.b.select(f32_, None, positive, snapped, z)?;
            }
            self.b.store(zp, z, None, [])?;
        }
        self.b.ret()?;
        self.b.end_function()?;
        let interface = self.interface.clone();
        self.b.entry_point(model, main, "main", interface);
        if self.opts.stage == Stage::Compute {
            let [x, y, z] = self.opts.local_size.map(|n| n.max(1));
            self.b.execution_mode(main, ExecutionMode::LocalSize, [x, y, z]);
        }
        if self.opts.stage == Stage::Fragment {
            self.b.execution_mode(main, ExecutionMode::OriginUpperLeft, []);
            if self.header.ps_writes_depth() {
                self.b.execution_mode(main, ExecutionMode::DepthReplacing, []);
            }
        }
        Ok(())
    }

    fn init_state(&mut self) -> R {
        let f = self.b.constant_false(self.t.bool);
        for v in self.cc.into_iter().chain(self.preds.clone()) {
            self.b.store(v, f, None, [])?;
        }
        let zero = self.u(0);
        for k in 0..255 {
            let v = self.regs[k];
            self.b.store(v, zero, None, [])?;
        }
        if self.opts.stage == Stage::Vertex {
            let (z, o) = (self.f(0.0), self.f(1.0));
            let v4f = self.t.v4f;
            let init = self.b.constant_composite(v4f, [z, z, z, o]);
            let position = self.builtin(BuiltIn::Position, v4f, true);
            self.b.store(position, init, None, [])?;
            for g in 0..32 {
                if (0..4).any(|c| self.header.vtg_output(g, c)) {
                    let (var, _) = self.io_var(g as u32, true);
                    self.b.store(var, init, None, [])?;
                }
            }
        }
        for var in self.probe.into_iter().flatten() {
            self.b.store(var, zero, None, [])?;
        }
        Ok(())
    }

    fn dispatcher(&mut self, program: &Program) -> R {
        let blocks = program.blocks();
        for (k, (leader, _)) in blocks.iter().enumerate() {
            self.block_ids.insert(*leader, k as u32);
        }
        let pc_var = self.b.variable(self.t.fn_u32, None, StorageClass::Function, None);
        self.pc_var = Some(pc_var);
        let len = self.u(STACK_DEPTH);
        let arr = self.b.type_array(self.t.u32, len);
        let arr_ptr = self.b.type_pointer(None, StorageClass::Function, arr);
        let ssy = (self.b.variable(arr_ptr, None, StorageClass::Function, None), self.b.variable(self.t.fn_u32, None, StorageClass::Function, None));
        let pbk = (self.b.variable(arr_ptr, None, StorageClass::Function, None), self.b.variable(self.t.fn_u32, None, StorageClass::Function, None));
        self.init_state()?;
        let zero = self.u(0);
        self.b.store(ssy.1, zero, None, [])?;
        self.b.store(pbk.1, zero, None, [])?;
        self.ssy = Some(ssy);
        self.pbk = Some(pbk);
        let entry = self.block_ids[&program.leaders.first().copied().unwrap_or(cfg::ENTRY)];
        let entry_c = self.u(entry);
        self.b.store(pc_var, entry_c, None, [])?;

        if program.forward_only(STACK_DEPTH as usize) {
            for (k, (_, insns)) in blocks.iter().enumerate() {
                let pc = self.b.load(self.t.u32, None, pc_var, None, [])?;
                let kc = self.u(k as u32);
                let here = self.b.i_equal(self.t.bool, None, pc, kc)?;
                let (body, skip) = (self.b.id(), self.b.id());
                self.b.selection_merge(skip, SelectionControl::NONE)?;
                self.b.branch_conditional(here, body, skip, [])?;
                self.b.begin_block(Some(body))?;
                self.dispatch_merge = Some(skip);
                self.block(pc_var, skip, insns)?;
                self.b.begin_block(Some(skip))?;
            }
            self.dispatch_merge = None;
            return Ok(());
        }

        let (header, body, cont, merge) = (self.b.id(), self.b.id(), self.b.id(), self.b.id());
        self.b.branch(header)?;
        self.b.begin_block(Some(header))?;
        self.b.loop_merge(merge, cont, LoopControl::NONE, [])?;
        self.b.branch(body)?;
        self.b.begin_block(Some(body))?;
        let pc = self.b.load(self.t.u32, None, pc_var, None, [])?;
        let sw_merge = self.b.id();
        let labels: Vec<Word> = blocks.iter().map(|_| self.b.id()).collect();
        self.b.selection_merge(sw_merge, SelectionControl::NONE)?;
        let targets: Vec<(Operand, Word)> = labels.iter().enumerate().map(|(k, &l)| (Operand::LiteralBit32(k as u32), l)).collect();
        self.b.switch(pc, sw_merge, targets)?;
        self.dispatch_merge = Some(sw_merge);
        for (k, (_, insns)) in blocks.iter().enumerate() {
            self.b.begin_block(Some(labels[k]))?;
            self.block(pc_var, sw_merge, insns)?;
        }
        self.b.begin_block(Some(sw_merge))?;
        self.b.branch(cont)?;
        self.b.begin_block(Some(cont))?;
        let pc = self.b.load(self.t.u32, None, pc_var, None, [])?;
        let exit = self.u(EXIT_BLOCK);
        let more = self.b.i_not_equal(self.t.bool, None, pc, exit)?;
        self.b.branch_conditional(more, header, merge, [])?;
        self.b.begin_block(Some(merge))?;
        self.dispatch_merge = None;
        Ok(())
    }

    fn block(&mut self, pc_var: Word, merge: Word, insns: &[(u32, Insn)]) -> R {
        self.ended = false;
        let mut last = 0;
        for (pc, i) in insns {
            last = *pc;
            self.insn(*pc, i)?;
            if self.ended {
                break;
            }
        }
        if !self.ended {
            let fall = self.block_number(cfg::next(last));
            let c = self.u(fall);
            self.b.store(pc_var, c, None, [])?;
            self.b.branch(merge)?;
        }
        Ok(())
    }

    fn block_number(&self, addr: u32) -> u32 {
        self.block_ids.get(&addr).copied().unwrap_or(EXIT_BLOCK)
    }

    fn guarded(&mut self, i: &Insn, body: impl FnOnce(&mut Self) -> R) -> R {
        let (p, not) = i.guard();
        if p == PT {
            return if not { Ok(()) } else { body(self) };
        }
        let cond = self.p(p, not)?;
        let (then, merge) = (self.b.id(), self.b.id());
        self.b.selection_merge(merge, SelectionControl::NONE)?;
        self.b.branch_conditional(cond, then, merge, [])?;
        self.b.begin_block(Some(then))?;
        body(self)?;
        self.b.branch(merge)?;
        self.b.begin_block(Some(merge))?;
        Ok(())
    }

    fn flow_condition(&mut self, i: &Insn) -> R<Option<Word>> {
        if cfg::unconditional(i) {
            return Ok(None);
        }
        let (p, not) = i.guard();
        let guard = self.p(p, not)?;
        if i.bits(0, 5) == 0xf {
            return Ok(Some(guard));
        }
        let test = self.cc_test(i.bits(0, 5))?;
        Ok(Some(self.b.logical_and(self.t.bool, None, guard, test)?))
    }

    fn cc_flag(&mut self, k: usize) -> R<Word> {
        Ok(self.b.load(self.t.bool, None, self.cc[k], None, [])?)
    }

    fn set_cc_float(&mut self, v: Word) -> R {
        let bt = self.t.bool;
        let zero = self.f(0.0);
        let z = self.b.f_ord_equal(bt, None, v, zero)?;
        let s = self.b.f_ord_less_than(bt, None, v, zero)?;
        let f = self.b.constant_false(bt);
        let (cz, cs, cc, co) = (self.cc[0], self.cc[1], self.cc[2], self.cc[3]);
        self.b.store(cz, z, None, [])?;
        self.b.store(cs, s, None, [])?;
        self.b.store(cc, f, None, [])?;
        self.b.store(co, f, None, [])?;
        Ok(())
    }

    fn set_cc(&mut self, v: Word, carry: Option<Word>, overflow: Option<Word>) -> R {
        let zero = self.u(0);
        let z = self.b.i_equal(self.t.bool, None, v, zero)?;
        let s = self.b.s_less_than(self.t.bool, None, v, zero)?;
        let f = self.b.constant_false(self.t.bool);
        let (cz, cs, cc, co) = (self.cc[0], self.cc[1], self.cc[2], self.cc[3]);
        self.b.store(cz, z, None, [])?;
        self.b.store(cs, s, None, [])?;
        self.b.store(cc, carry.unwrap_or(f), None, [])?;
        self.b.store(co, overflow.unwrap_or(f), None, [])?;
        Ok(())
    }

    fn cc_test(&mut self, test: u64) -> R<Word> {
        let bt = self.t.bool;
        let (z, s, c, o) = (self.cc_flag(0)?, self.cc_flag(1)?, self.cc_flag(2)?, self.cc_flag(3)?);
        let lt = self.b.logical_not_equal(bt, None, s, o)?;
        let not = |t: &mut Self, v: Word| -> R<Word> { Ok(t.b.logical_not(bt, None, v)?) };
        Ok(match test {
            0 | 0x10 => self.b.constant_false(bt),
            1 | 9 => lt,
            2 | 10 => z,
            3 | 11 => self.b.logical_or(bt, None, lt, z)?,
            4 | 12 => {
                let le = self.b.logical_or(bt, None, lt, z)?;
                not(self, le)?
            }
            5 | 13 => not(self, z)?,
            6 | 14 => not(self, lt)?,
            7 => not(self, o)?,
            8 => o,
            0x11 => not(self, c)?,
            0x12 => not(self, s)?,
            0x13 => {
                let nc = not(self, c)?;
                self.b.logical_or(bt, None, nc, z)?
            }
            0x14 => {
                let nz = not(self, z)?;
                self.b.logical_and(bt, None, c, nz)?
            }
            0x15 => s,
            0x16 => c,
            0x17 => o,
            _ => {
                self.unsupported.push("condition code test");
                self.btrue()
            }
        })
    }

    fn jump(&mut self, cond: Option<Word>, target: Word, fall_addr: u32) -> R {
        let pc_var = self.pc_var.expect("dispatcher");
        let value = match cond {
            None => target,
            Some(c) => {
                let fall = self.block_number(fall_addr);
                let f = self.u(fall);
                self.b.select(self.t.u32, None, c, target, f)?
            }
        };
        self.b.store(pc_var, value, None, [])?;
        self.b.branch(self.dispatch_merge.unwrap())?;
        self.ended = true;
        Ok(())
    }

    fn push(&mut self, stack: (Word, Word), value: u32) -> R {
        let sp = self.b.load(self.t.u32, None, stack.1, None, [])?;
        let ptr = self.b.access_chain(self.t.fn_u32, None, stack.0, [sp])?;
        let v = self.u(value);
        self.b.store(ptr, v, None, [])?;
        let one = self.u(1);
        let sp1 = self.b.i_add(self.t.u32, None, sp, one)?;
        self.b.store(stack.1, sp1, None, [])?;
        Ok(())
    }

    fn pop(&mut self, stack: (Word, Word)) -> R<Word> {
        let sp = self.b.load(self.t.u32, None, stack.1, None, [])?;
        let one = self.u(1);
        let sp1 = self.b.i_sub(self.t.u32, None, sp, one)?;
        let max = self.u(STACK_DEPTH - 1);
        let sp1 = self.ext(self.t.u32, glsl::UMIN, &[sp1, max])?;
        self.b.store(stack.1, sp1, None, [])?;
        let ptr = self.b.access_chain(self.t.fn_u32, None, stack.0, [sp1])?;
        Ok(self.b.load(self.t.u32, None, ptr, None, [])?)
    }

    fn insn(&mut self, pc: u32, i: &Insn) -> R {
        if let (Some((at, n)), Some(vars)) = (self.opts.debug_probe, self.probe)
            && at == pc
        {
            if self.opts.stage == Stage::Compute {
                self.compute_probe(n)?;
            } else {
                for (k, var) in vars.into_iter().enumerate() {
                    let v = self.r(n.saturating_add(k as u8))?;
                    self.b.store(var, v, None, [])?;
                }
            }
        }
        match i.name() {
            "nop" | "depbar" | "membar" | "bar" | "sync" | "brk" | "ssy" | "pbk" | "bra" | "exit" | "kil" => self.flow(pc, i),
            "brx" => self.brx(pc, i),
            _ => self.guarded(i, |t| t.op(pc, i)),
        }
    }

    fn flow(&mut self, pc: u32, i: &Insn) -> R {
        let fall = cfg::next(pc);
        match i.name() {
            "bar" if self.opts.stage == Stage::Compute => {
                let workgroup = self.u(2);
                let semantics = self.u(0x8 | 0x100);
                self.b.control_barrier(workgroup, workgroup, semantics)?;
                Ok(())
            }
            "nop" | "depbar" | "membar" | "bar" => Ok(()),
            "ssy" | "pbk" => {
                if i.bit(5) {
                    self.unsupported.push("ssy/pbk from constant buffer");
                    return Ok(());
                }
                let target = self.block_number(i.branch_target(pc));
                let stack = if i.name() == "ssy" { self.ssy } else { self.pbk };
                match stack {
                    Some(s) => self.push(s, target),
                    None => Ok(()),
                }
            }
            "bra" => {
                if i.bit(5) {
                    self.unsupported.push("bra to constant buffer address");
                    return Ok(());
                }
                if self.pc_var.is_none() {
                    return Ok(());
                }
                let cond = self.flow_condition(i)?;
                let target = self.block_number(i.branch_target(pc));
                let t = self.u(target);
                self.jump(cond, t, fall)
            }
            "sync" | "brk" => {
                let Some(stack) = (if i.name() == "sync" { self.ssy } else { self.pbk }) else { return Ok(()) };
                let cond = self.flow_condition(i)?;
                match cond {
                    None => {
                        let target = self.pop(stack)?;
                        self.jump(None, target, fall)
                    }
                    Some(c) => {
                        let pc_var = self.pc_var.unwrap();
                        let (then, other, merge) = (self.b.id(), self.b.id(), self.b.id());
                        self.b.selection_merge(merge, SelectionControl::NONE)?;
                        self.b.branch_conditional(c, then, other, [])?;
                        self.b.begin_block(Some(then))?;
                        let target = self.pop(stack)?;
                        self.b.store(pc_var, target, None, [])?;
                        self.b.branch(merge)?;
                        self.b.begin_block(Some(other))?;
                        let f = self.block_number(fall);
                        let fc = self.u(f);
                        self.b.store(pc_var, fc, None, [])?;
                        self.b.branch(merge)?;
                        self.b.begin_block(Some(merge))?;
                        self.b.branch(self.dispatch_merge.unwrap())?;
                        self.ended = true;
                        Ok(())
                    }
                }
            }
            "exit" => {
                if self.pc_var.is_none() {
                    return Ok(());
                }
                let cond = self.flow_condition(i)?;
                let e = self.u(EXIT_BLOCK);
                self.jump(cond, e, fall)
            }
            "kil" => {
                let cond = self.flow_condition(i)?;
                match cond {
                    None => {
                        self.kill()?;
                        self.b.begin_block(None)?;
                        Ok(())
                    }
                    Some(c) => {
                        let (then, merge) = (self.b.id(), self.b.id());
                        self.b.selection_merge(merge, SelectionControl::NONE)?;
                        self.b.branch_conditional(c, then, merge, [])?;
                        self.b.begin_block(Some(then))?;
                        self.kill()?;
                        self.b.begin_block(Some(merge))?;
                        Ok(())
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    fn op(&mut self, pc: u32, i: &Insn) -> R {
        let t = self.t.f32;
        let u = self.t.u32;
        match i.name() {
            "mov" if i.op.value == 0xf0c8000000000000 => {
                let v = match i.bits(20, 8) {
                    0x00 => self.lane_id()?,
                    0x12 => self.u(1.0f32.to_bits()),
                    0x13 => {
                        if self.opts.stage == Stage::Fragment {
                            let var = self.builtin(BuiltIn::HelperInvocation, self.t.bool, false);
                            let helper = self.b.load(self.t.bool, None, var, None, [])?;
                            self.b2u(helper, 1, 0)?
                        } else {
                            self.u(0)
                        }
                    }
                    0x20 => {
                        let id = self.compute_id(BuiltIn::LocalInvocationId)?;
                        let (x, y, z) = (self.b.composite_extract(u, None, id, [0])?, self.b.composite_extract(u, None, id, [1])?, self.b.composite_extract(u, None, id, [2])?);
                        let (s16, s26) = (self.u(16), self.u(26));
                        let y = self.b.shift_left_logical(u, None, y, s16)?;
                        let z = self.b.shift_left_logical(u, None, z, s26)?;
                        let xy = self.b.bitwise_or(u, None, x, y)?;
                        self.b.bitwise_or(u, None, xy, z)?
                    }
                    k @ (0x21..=0x23 | 0x25..=0x27) => {
                        let which = if k < 0x24 { BuiltIn::LocalInvocationId } else { BuiltIn::WorkgroupId };
                        let id = self.compute_id(which)?;
                        let c = if k < 0x24 { k - 0x21 } else { k - 0x25 };
                        self.b.composite_extract(u, None, id, [c as u32])?
                    }
                    0x38..=0x3c => {
                        let lane = self.lane_id()?;
                        let one = self.u(1);
                        let bit = self.b.shift_left_logical(u, None, one, lane)?;
                        let below = self.b.i_sub(u, None, bit, one)?;
                        match i.bits(20, 8) {
                            0x38 => bit,
                            0x39 => below,
                            0x3a => self.b.bitwise_or(u, None, below, bit)?,
                            0x3b => {
                                let le = self.b.bitwise_or(u, None, below, bit)?;
                                self.b.not(u, None, le)?
                            }
                            _ => self.b.not(u, None, below)?,
                        }
                    }
                    _ => {
                        self.unsupported.push("s2r (system value)");
                        self.u(0)
                    }
                };
                self.set(i.rd(), v)
            }
            "mov" => {
                let v = self.src_b(i)?;
                self.set(i.rd(), v)
            }
            "mov32i" => {
                let v = self.u(i.imm32());
                self.set(i.rd(), v)
            }
            "sel" => {
                let (p, n) = i.pred(39, 42);
                let c = self.p(p, n)?;
                let a = self.r(i.ra())?;
                let b = self.src_b(i)?;
                let v = self.b.select(u, None, c, a, b)?;
                self.set(i.rd(), v)
            }
            "fadd" | "fadd32i" => {
                let (neg_a, abs_a, neg_b, abs_b, sat) = if i.name() == "fadd" {
                    (i.bit(48), i.bit(46), i.bit(45), i.bit(49), i.bit(50))
                } else {
                    (i.bit(56), i.bit(54), i.bit(53), i.bit(57), false)
                };
                let a = self.rf(i.ra())?;
                let a = self.fneg_abs(a, neg_a, abs_a)?;
                let b = if i.name() == "fadd32i" { self.f(f32::from_bits(i.imm32())) } else { self.src_bf(i)? };
                let b = self.fneg_abs(b, neg_b, abs_b)?;
                let v = self.b.f_add(t, None, a, b)?;
                self.exact(v);
                let v = self.saturate(v, sat)?;
                if (i.name() == "fadd" && i.bit(47)) || (i.name() == "fadd32i" && i.bit(52)) {
                    self.set_cc_float(v)?;
                }
                self.setf(i.rd(), v)
            }
            "fmul" | "fmul32i" => {
                let a = self.rf(i.ra())?;
                let (b, neg, sat) = if i.name() == "fmul" {
                    (self.src_bf(i)?, i.bit(48), i.bit(50))
                } else {
                    (self.f(f32::from_bits(i.imm32())), false, i.bit(55))
                };
                let b = self.fneg_abs(b, neg, false)?;
                let mut v = self.b.f_mul(t, None, a, b)?;
                self.exact(v);
                if i.name() == "fmul" {
                    let scale = match i.bits(41, 3) {
                        1 => 0.5,
                        2 => 0.25,
                        3 => 0.125,
                        4 => 8.0,
                        5 => 4.0,
                        6 => 2.0,
                        _ => 1.0,
                    };
                    if scale != 1.0 {
                        let s = self.f(scale);
                        v = self.b.f_mul(t, None, v, s)?;
                        self.exact(v);
                    }
                }
                let v = self.saturate(v, sat)?;
                if i.name() == "fmul" && i.bit(47) {
                    self.set_cc_float(v)?;
                }
                self.setf(i.rd(), v)
            }
            "ffma" | "ffma32i" => {
                let a = self.rf(i.ra())?;
                let (b, c, neg_b, neg_c, sat) = if i.name() == "ffma" {
                    let (b, c) = self.src_bc(i, i.op.value == 0x5180000000000000)?;
                    let (b, c) = if i.form() == Form::Imm {
                        let bits = self.u(i.fimm20());
                        (bits, c)
                    } else {
                        (b, c)
                    };
                    (self.cast_f(b)?, self.cast_f(c)?, i.bit(48), i.bit(49), i.bit(50))
                } else {
                    let b = self.f(f32::from_bits(i.imm32()));
                    let c = self.rf(i.rd())?;
                    (b, c, i.bit(56), i.bit(57), i.bit(55))
                };
                let b = self.fneg_abs(b, neg_b, false)?;
                let c = self.fneg_abs(c, neg_c, false)?;
                let v = self.ext(t, glsl::FMA, &[a, b, c])?;
                self.exact(v);
                let v = self.saturate(v, sat)?;
                if i.name() == "ffma" && i.bit(47) {
                    self.set_cc_float(v)?;
                }
                self.setf(i.rd(), v)
            }
            "fmnmx" => {
                let a = self.rf(i.ra())?;
                let a = self.fneg_abs(a, i.bit(48), i.bit(46))?;
                let b = self.src_bf(i)?;
                let b = self.fneg_abs(b, i.bit(45), i.bit(49))?;
                let (p, n) = i.pred(39, 42);
                let c = self.p(p, n)?;
                let mn = self.ext(t, glsl::NMIN, &[a, b])?;
                let mx = self.ext(t, glsl::NMAX, &[a, b])?;
                let v = self.b.select(t, None, c, mn, mx)?;
                if i.bit(47) {
                    self.set_cc_float(v)?;
                }
                self.setf(i.rd(), v)
            }
            "mufu" => {
                let a = self.rf(i.ra())?;
                let a = self.fneg_abs(a, i.bit(48), i.bit(46))?;
                let v = match i.bits(20, 4) {
                    0 => self.ext(t, glsl::COS, &[a])?,
                    1 => self.ext(t, glsl::SIN, &[a])?,
                    2 => self.ext(t, glsl::EXP2, &[a])?,
                    3 => self.ext(t, glsl::LOG2, &[a])?,
                    4 => {
                        let one = self.f(1.0);
                        self.b.f_div(t, None, one, a)?
                    }
                    5 => self.ext(t, glsl::INVERSE_SQRT, &[a])?,
                    8 => self.ext(t, glsl::SQRT, &[a])?,
                    _ => {
                        self.unsupported.push("mufu (64-bit)");
                        a
                    }
                };
                let v = self.saturate(v, i.bit(50))?;
                self.setf(i.rd(), v)
            }
            "rro" => {
                let b = self.src_bf(i)?;
                let b = self.fneg_abs(b, i.bit(45), i.bit(49))?;
                self.setf(i.rd(), b)
            }
            "fswzadd" => {
                let a = self.rf(i.ra())?;
                let b = self.rf(i.rb())?;
                let lane = self.subgroup_invocation()?;
                let three = self.u(3);
                let lane = self.b.bitwise_and(u, None, lane, three)?;
                let one = self.u(1);
                let shift = self.b.shift_left_logical(u, None, lane, one)?;
                let pn = self.u(i.bits(28, 8) as u32);
                let sel = self.b.shift_right_logical(u, None, pn, shift)?;
                let sel = self.b.bitwise_and(u, None, sel, three)?;
                let mut x = self.f(0.0);
                let mut y = self.f(1.0);
                for (k, (xk, yk)) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)].into_iter().enumerate() {
                    let kc = self.u(k as u32);
                    let hit = self.b.i_equal(self.t.bool, None, sel, kc)?;
                    let (xc, yc) = (self.f(xk), self.f(yk));
                    x = self.b.select(t, None, hit, xc, x)?;
                    y = self.b.select(t, None, hit, yc, y)?;
                }
                let ax = self.b.f_mul(t, None, a, x)?;
                let v = self.ext(t, glsl::FMA, &[b, y, ax])?;
                self.setf(i.rd(), v)
            }
            "fset" | "fsetp" => self.fcompare(i),
            "csetp" => {
                let test = self.cc_test(i.bits(8, 5))?;
                let (p, n) = i.pred(39, 42);
                let pv = self.p(p, n)?;
                let op = i.bits(45, 2);
                let r = self.bool_op(op, test, pv)?;
                let nt = self.b.logical_not(self.t.bool, None, test)?;
                let r2 = self.bool_op(op, nt, pv)?;
                self.setp(i.bits(3, 3) as u8, r)?;
                self.setp(i.bits(0, 3) as u8, r2)
            }
            "vote" => self.vote(i),
            "shfl" => self.shfl(i),
            "red" | "atom" => self.atomic(pc, i),
            "isetp" | "iset" => self.icompare(i),
            "psetp" => {
                let (a, na) = i.pred(12, 15);
                let (b, nb) = i.pred(29, 32);
                let (c, nc) = i.pred(39, 42);
                let a = self.p(a, na)?;
                let b = self.p(b, nb)?;
                let c = self.p(c, nc)?;
                let ab = self.bool_op(i.bits(24, 2), a, b)?;
                let r = self.bool_op(i.bits(45, 2), ab, c)?;
                let nab = self.b.logical_not(self.t.bool, None, ab)?;
                let r2 = self.bool_op(i.bits(45, 2), nab, c)?;
                self.setp(i.bits(3, 3) as u8, r)?;
                self.setp(i.bits(0, 3) as u8, r2)
            }
            "pset" => {
                let (a, na) = i.pred(12, 15);
                let (b, nb) = i.pred(29, 32);
                let (c, nc) = i.pred(39, 42);
                let a = self.p(a, na)?;
                let b = self.p(b, nb)?;
                let c = self.p(c, nc)?;
                let ab = self.bool_op(i.bits(24, 2), a, b)?;
                let r = self.bool_op(i.bits(45, 2), ab, c)?;
                let v = if i.bit(44) { self.b2u(r, 1.0f32.to_bits(), 0)? } else { self.b2u(r, u32::MAX, 0)? };
                if i.bit(47) {
                    self.set_cc(v, None, None)?;
                }
                self.set(i.rd(), v)
            }
            "iadd" | "iadd32i" => {
                let a = self.r(i.ra())?;
                let (b, neg_a, neg_b, x, cc) = if i.name() == "iadd" {
                    (self.src_b(i)?, i.bit(49), i.bit(48), i.bit(43), i.bit(47))
                } else {
                    (self.u(i.imm32()), i.bit(56), false, i.bit(53), i.bit(52))
                };
                let a = self.ineg(a, neg_a)?;
                let b = self.ineg(b, neg_b)?;
                let v = self.add_with_flags(a, b, x, cc)?;
                self.set(i.rd(), v)
            }
            "iadd3" => {
                let a = self.r(i.ra())?;
                let (b, c) = self.src_bc(i, false)?;
                let (a, b, c, shift) = if i.form() == Form::Reg {
                    let a = self.half_select(a, i.bits(35, 2))?;
                    let b = self.half_select(b, i.bits(33, 2))?;
                    let c = self.half_select(c, i.bits(31, 2))?;
                    (a, b, c, i.bits(37, 2))
                } else {
                    (a, b, c, 0)
                };
                let (a, b, c) = (self.ineg(a, i.bit(51))?, self.ineg(b, i.bit(50))?, self.ineg(c, i.bit(49))?);
                let mut ab = self.b.i_add(u, None, a, b)?;
                let sixteen = self.u(16);
                match shift {
                    1 => ab = self.b.shift_right_logical(u, None, ab, sixteen)?,
                    2 => ab = self.b.shift_left_logical(u, None, ab, sixteen)?,
                    _ => {}
                }
                let v = self.add_with_flags(ab, c, i.bit(48), i.bit(47))?;
                self.set(i.rd(), v)
            }
            "iscadd" => {
                let a = self.r(i.ra())?;
                let a = self.ineg(a, i.bit(49))?;
                let b = self.src_b(i)?;
                let b = self.ineg(b, i.bit(48))?;
                let sh = self.u(i.bits(39, 5) as u32);
                let a = self.b.shift_left_logical(u, None, a, sh)?;
                let v = self.b.i_add(u, None, a, b)?;
                self.set(i.rd(), v)
            }
            "xmad" => self.xmad(i),
            "imnmx" => {
                let a = self.r(i.ra())?;
                let b = self.src_b(i)?;
                let (p, n) = i.pred(39, 42);
                let c = self.p(p, n)?;
                let signed = i.bit(48);
                let (mn, mx) = if signed { (glsl::SMIN, glsl::SMAX) } else { (glsl::UMIN, glsl::UMAX) };
                let lo = self.ext(u, mn, &[a, b])?;
                let hi = self.ext(u, mx, &[a, b])?;
                let v = self.b.select(u, None, c, lo, hi)?;
                self.set(i.rd(), v)
            }
            "fcmp" => {
                let a = self.r(i.ra())?;
                let (b, c) = match i.form() {
                    Form::Imm => (self.u(i.fimm20()), self.r(i.rc())?),
                    _ => self.src_bc(i, i.op.value == 0x53a0000000000000)?,
                };
                let cf = self.cast_f(c)?;
                let zero = self.f(0.0);
                let cond = self.fcmp(i.bits(48, 4), cf, zero)?;
                let v = self.b.select(u, None, cond, a, b)?;
                self.set(i.rd(), v)
            }
            "icmp" => {
                let a = self.r(i.ra())?;
                let (b, c) = self.src_bc(i, i.op.value == 0x5340000000000000)?;
                let zero = self.u(0);
                let cond = self.int_cmp(i.bits(49, 3), i.bit(48), c, zero)?;
                let v = self.b.select(u, None, cond, a, b)?;
                self.set(i.rd(), v)
            }
            "lop" | "lop32i" => {
                let (op, inv_a, inv_b, b) = if i.name() == "lop" {
                    (i.bits(41, 2), i.bit(39), i.bit(40), self.src_b(i)?)
                } else {
                    (i.bits(53, 2), i.bit(55), i.bit(56), self.u(i.imm32()))
                };
                let a = self.r(i.ra())?;
                let a = if inv_a { self.b.not(u, None, a)? } else { a };
                let b = if inv_b { self.b.not(u, None, b)? } else { b };
                let v = match op {
                    0 => self.b.bitwise_and(u, None, a, b)?,
                    1 => self.b.bitwise_or(u, None, a, b)?,
                    2 => self.b.bitwise_xor(u, None, a, b)?,
                    _ => b,
                };
                if (i.name() == "lop" && i.bit(47)) || (i.name() == "lop32i" && i.bit(52)) {
                    self.set_cc(v, None, None)?;
                }
                if i.name() == "lop" && i.bits(48, 3) != PT as u64 {
                    let zero = self.u(0);
                    let p = match i.bits(44, 2) {
                        2 => self.b.i_equal(self.t.bool, None, v, zero)?,
                        3 => self.b.i_not_equal(self.t.bool, None, v, zero)?,
                        1 => self.btrue(),
                        _ => self.b.constant_false(self.t.bool),
                    };
                    self.setp(i.bits(48, 3) as u8, p)?;
                }
                self.set(i.rd(), v)
            }
            "lop3" => {
                let a = self.r(i.ra())?;
                let (b, c, lut) = match i.form() {
                    Form::Reg => (self.r(i.rb())?, self.r(i.rc())?, i.bits(28, 8) as u32),
                    Form::Imm => (self.u(i.simm20() as u32), self.r(i.rc())?, i.bits(48, 8) as u32),
                    Form::Cbuf => {
                        let (s, o) = i.cbuf();
                        (self.cbuf_imm(s, o)?, self.r(i.rc())?, i.bits(48, 8) as u32)
                    }
                };
                let v = self.lop3(a, b, c, lut)?;
                self.set(i.rd(), v)
            }
            "shl" => {
                let a = self.r(i.ra())?;
                let b = self.src_b(i)?;
                let v = self.shift(a, b, true, false, i.bit(39))?;
                self.set(i.rd(), v)
            }
            "shr" => {
                let a = self.r(i.ra())?;
                let b = self.src_b(i)?;
                let a = if i.bit(40) { self.b.bit_reverse(u, None, a)? } else { a };
                let v = self.shift(a, b, false, i.bit(48), i.bit(39))?;
                self.set(i.rd(), v)
            }
            "bfe" => {
                let a = self.r(i.ra())?;
                let b = self.src_b(i)?;
                let a = if i.bit(40) { self.b.bit_reverse(u, None, a)? } else { a };
                let ff = self.u(0xff);
                let pos = self.b.bitwise_and(u, None, b, ff)?;
                let eight = self.u(8);
                let len = self.b.shift_right_logical(u, None, b, eight)?;
                let len = self.b.bitwise_and(u, None, len, ff)?;
                let c32 = self.u(32);
                let pos = self.ext(u, glsl::UMIN, &[pos, c32])?;
                let room = self.b.i_sub(u, None, c32, pos)?;
                let len = self.ext(u, glsl::UMIN, &[len, room])?;
                let v = if i.bit(48) {
                    let ai = self.cast_i(a)?;
                    let r = self.b.bit_field_s_extract(self.t.i32, None, ai, pos, len)?;
                    self.cast_u(r)?
                } else {
                    self.b.bit_field_u_extract(u, None, a, pos, len)?
                };
                self.set(i.rd(), v)
            }
            "bfi" => {
                let a = self.r(i.ra())?;
                let (b, c) = self.src_bc(i, i.op.value == 0x53f0000000000000)?;
                let ff = self.u(0xff);
                let pos = self.b.bitwise_and(u, None, b, ff)?;
                let eight = self.u(8);
                let len = self.b.shift_right_logical(u, None, b, eight)?;
                let len = self.b.bitwise_and(u, None, len, ff)?;
                let c32 = self.u(32);
                let pos = self.ext(u, glsl::UMIN, &[pos, c32])?;
                let room = self.b.i_sub(u, None, c32, pos)?;
                let len = self.ext(u, glsl::UMIN, &[len, room])?;
                let v = self.b.bit_field_insert(u, None, c, a, pos, len)?;
                self.set(i.rd(), v)
            }
            "vmnmx" => self.vmnmx(i),
            "i2f" => self.i2f(i),
            "f2i" => self.f2i(i),
            "f2f" => self.f2f(i),
            "i2i" => self.i2i(i),
            "flo" => {
                let b = self.src_b(i)?;
                let b = if i.bit(40) { self.b.not(u, None, b)? } else { b };
                let v = if i.bit(48) { self.ext(u, glsl::FIND_S_MSB, &[b])? } else { self.ext(u, glsl::FIND_U_MSB, &[b])? };
                let v = if i.bit(41) {
                    let c31 = self.u(31);
                    let s = self.b.i_sub(u, None, c31, v)?;
                    let none = self.u(u32::MAX);
                    let is_none = self.b.i_equal(self.t.bool, None, v, none)?;
                    self.b.select(u, None, is_none, none, s)?
                } else {
                    v
                };
                self.set(i.rd(), v)
            }
            "popc" => {
                let b = self.src_b(i)?;
                let b = if i.bit(40) { self.b.not(u, None, b)? } else { b };
                let v = self.b.bit_count(u, None, b)?;
                self.set(i.rd(), v)
            }
            "ld" => self.load(i),
            "st" => self.store(i),
            "ldg" | "stg" => self.global_memory(pc, i),
            "r2p" => {
                if i.bit(40) {
                    self.unsupported.push("r2p (condition codes)");
                    return Ok(());
                }
                let a = self.r(i.ra())?;
                let mask = match i.form() {
                    Form::Imm => i.simm20() as u32,
                    _ => {
                        self.unsupported.push("r2p (register mask)");
                        0x7f
                    }
                };
                let byte = i.bits(41, 2) as u32;
                for k in 0..7u8 {
                    if mask >> k & 1 == 0 {
                        continue;
                    }
                    let bit = self.u(1 << (byte * 8 + k as u32));
                    let v = self.b.bitwise_and(u, None, a, bit)?;
                    let zero = self.u(0);
                    let p = self.b.i_not_equal(self.t.bool, None, v, zero)?;
                    self.setp(k, p)?;
                }
                Ok(())
            }
            "p2r" => {
                if i.bit(40) {
                    self.unsupported.push("p2r (condition codes)");
                    return Ok(());
                }
                let a = self.r(i.ra())?;
                let mask = match i.form() {
                    Form::Imm => i.simm20() as u32 & 0xff,
                    _ => {
                        self.unsupported.push("p2r (register mask)");
                        0x7f
                    }
                };
                let byte = i.bits(41, 2) as u32;
                let keep = self.u(!(mask << (byte * 8)));
                let mut v = self.b.bitwise_and(u, None, a, keep)?;
                for k in 0..7u8 {
                    if mask >> k & 1 == 0 {
                        continue;
                    }
                    let p = self.p(k, false)?;
                    let bit = self.b2u(p, 1 << (byte * 8 + k as u32), 0)?;
                    v = self.b.bitwise_or(u, None, v, bit)?;
                }
                self.set(i.rd(), v)
            }
            "shf" => {
                let low = self.r(i.ra())?;
                let sh = self.src_b(i)?;
                let high = self.r(i.rc())?;
                let left = matches!(i.op.value, 0x5bf8000000000000 | 0x36f8000000000000);
                let c31 = self.u(31);
                let c32 = self.u(32);
                let amount = if i.bit(50) {
                    self.b.bitwise_and(u, None, sh, c31)?
                } else {
                    self.ext(u, glsl::UMIN, &[sh, c32])?
                };
                let inv = self.b.i_sub(u, None, c32, amount)?;
                let v = if left {
                    let a = self.b.shift_left_logical(u, None, high, amount)?;
                    let b = self.b.shift_right_logical(u, None, low, inv)?;
                    let r = self.b.bitwise_or(u, None, a, b)?;
                    let zero = self.u(0);
                    let is0 = self.b.i_equal(self.t.bool, None, amount, zero)?;
                    let is32 = self.b.i_equal(self.t.bool, None, amount, c32)?;
                    let r = self.b.select(u, None, is0, high, r)?;
                    self.b.select(u, None, is32, low, r)?
                } else {
                    let a = self.b.shift_right_logical(u, None, low, amount)?;
                    let b = self.b.shift_left_logical(u, None, high, inv)?;
                    let r = self.b.bitwise_or(u, None, a, b)?;
                    let zero = self.u(0);
                    let is0 = self.b.i_equal(self.t.bool, None, amount, zero)?;
                    let is32 = self.b.i_equal(self.t.bool, None, amount, c32)?;
                    let r = self.b.select(u, None, is0, low, r)?;
                    self.b.select(u, None, is32, high, r)?
                };
                if i.bits(37, 2) != 0 || i.bits(48, 2) != 0 {
                    self.unsupported.push("shf (64-bit modes)");
                }
                self.set(i.rd(), v)
            }
            "lea" => {
                let a = self.r(i.ra())?;
                let (b, sh, hi) = match i.op.value {
                    0x5bd8000000000000 => (self.r(i.rb())?, i.bits(28, 5) as u32, true),
                    0x1800000000000000 => {
                        let (s, o) = i.cbuf();
                        (self.cbuf_imm(s, o)?, i.bits(51, 5) as u32, true)
                    }
                    _ => (self.src_b(i)?, i.bits(39, 5) as u32, false),
                };
                let neg = if hi { i.bit(37) || i.bit(56) } else { i.bit(45) };
                let a = self.ineg(a, neg && !hi)?;
                let shifted = if hi {
                    let c = self.r(i.rc())?;
                    let s = self.u(sh);
                    let inv = self.u(32 - sh);
                    let x = self.b.shift_left_logical(u, None, c, s)?;
                    let y = self.b.shift_right_logical(u, None, a, inv)?;
                    if sh == 0 { c } else { self.b.bitwise_or(u, None, x, y)? }
                } else {
                    let s = self.u(sh);
                    self.b.shift_left_logical(u, None, a, s)?
                };
                let v = self.b.i_add(u, None, shifted, b)?;
                self.set(i.rd(), v)
            }
            "txq" => self.txq(i),
            "tmml" => self.tmml(pc, i),
            "ipa" => self.ipa(i),
            "texs" => self.texs(i),
            "tlds" => self.tlds(i),
            "sust" => self.sust(pc, i),
            "tex" => self.tex(i),
            "tld4" => self.tld4(i),
            "tld4s" => self.tld4s(i),
            "hadd2" | "hmul2" | "hfma2" => self.half(i),
            "hsetp2" => self.hsetp2(i),
            name => {
                self.unsupported.push(name);
                Ok(())
            }
        }
    }

    fn video_operand(&mut self, v: Word, sel: u64) -> R<Word> {
        let u = self.t.u32;
        let (pos, len) = match sel & 7 {
            k @ 0..=3 => (8 * k as u32, 8),
            k @ (4 | 5) => (16 * (k as u32 - 4), 16),
            _ => return Ok(v),
        };
        let (pos, len) = (self.u(pos), self.u(len));
        if sel & 8 != 0 {
            let vi = self.cast_i(v)?;
            let r = self.b.bit_field_s_extract(self.t.i32, None, vi, pos, len)?;
            self.cast_u(r)
        } else {
            Ok(self.b.bit_field_u_extract(u, None, v, pos, len)?)
        }
    }

    fn vmnmx(&mut self, i: &Insn) -> R {
        let u = self.t.u32;
        let bt = self.t.bool;
        let a_sel = (i.bits(48, 1) << 3) | i.bits(36, 3);
        let b_sel = (i.bits(49, 1) << 3) | i.bits(28, 3);
        let a = self.r(i.ra())?;
        let a = self.video_operand(a, a_sel)?;
        let b = if i.bit(50) {
            let b = self.r(i.rb())?;
            self.video_operand(b, b_sel)?
        } else {
            let imm = i.bits(20, 16) as u32;
            self.u(if b_sel & 8 != 0 { imm as u16 as i16 as i32 as u32 } else { imm })
        };
        let c = self.r(i.rc())?;
        let (a_signed, b_signed) = (a_sel & 8 != 0, b_sel & 8 != 0);
        let max = i.bit(56);
        let dformat = i.bit(54);
        let zero = self.u(0);
        let (mut v, signed) = if a_signed != b_signed {
            let v = if max { self.ext(u, glsl::UMAX, &[a, b])? } else { self.ext(u, glsl::SMIN, &[a, b])? };
            let (signed_op, other) = if a_signed { (a, b) } else { (b, a) };
            let neg = self.b.s_less_than(bt, None, signed_op, zero)?;
            let v = if max { self.b.select(u, None, neg, other, v)? } else { self.b.select(u, None, neg, signed_op, v)? };
            (v, !max)
        } else {
            let op = match (max, a_signed) {
                (true, true) => glsl::SMAX,
                (true, false) => glsl::UMAX,
                (false, true) => glsl::SMIN,
                (false, false) => glsl::UMIN,
            };
            (self.ext(u, op, &[a, b])?, a_signed)
        };
        if i.bit(55) {
            if dformat && !signed {
                let m = self.u(i32::MAX as u32);
                v = self.ext(u, glsl::UMIN, &[v, m])?;
            } else if !dformat && signed {
                v = self.ext(u, glsl::SMAX, &[v, zero])?;
            }
        }
        let merge = |t: &mut Self, v: Word, pos: u32, len: u32| -> R<Word> {
            let (p, l) = (t.u(pos), t.u(len));
            Ok(t.b.bit_field_insert(u, None, c, v, p, l)?)
        };
        let v = match i.bits(51, 3) {
            0 => merge(self, v, 16, 16)?,
            1 => merge(self, v, 0, 16)?,
            2 => merge(self, v, 0, 8)?,
            3 => merge(self, v, 16, 8)?,
            4 => self.b.i_add(u, None, v, c)?,
            5 => self.ext(u, if dformat { glsl::SMIN } else { glsl::UMIN }, &[v, c])?,
            6 => self.ext(u, if dformat { glsl::SMAX } else { glsl::UMAX }, &[v, c])?,
            _ => v,
        };
        self.set(i.rd(), v)
    }

    fn half_select(&mut self, v: Word, sel: u64) -> R<Word> {
        match sel {
            1 => self.half16(v, false, false),
            2 => self.half16(v, true, false),
            _ => Ok(v),
        }
    }

    fn add_with_flags(&mut self, a: Word, b: Word, x: bool, cc: bool) -> R<Word> {
        let u = self.t.u32;
        if !x && !cc {
            return Ok(self.b.i_add(u, None, a, b)?);
        }
        let mut v = self.b.i_add(u, None, a, b)?;
        let mut carry = self.b.u_less_than(self.t.bool, None, v, a)?;
        if x {
            let cin = self.cc_flag(2)?;
            let cin = self.b2u(cin, 1, 0)?;
            let v2 = self.b.i_add(u, None, v, cin)?;
            let c2 = self.b.u_less_than(self.t.bool, None, v2, v)?;
            carry = self.b.logical_or(self.t.bool, None, carry, c2)?;
            v = v2;
        }
        if cc {
            let zero = self.u(0);
            let c = carry;
            let xa = self.b.bitwise_xor(u, None, a, v)?;
            let xb = self.b.bitwise_xor(u, None, b, v)?;
            let both = self.b.bitwise_and(u, None, xa, xb)?;
            let o = self.b.s_less_than(self.t.bool, None, both, zero)?;
            self.set_cc(v, Some(c), Some(o))?;
        }
        Ok(v)
    }

    fn ineg(&mut self, v: Word, neg: bool) -> R<Word> {
        if !neg {
            return Ok(v);
        }
        let zero = self.u(0);
        Ok(self.b.i_sub(self.t.u32, None, zero, v)?)
    }

    fn bool_op(&mut self, op: u64, a: Word, b: Word) -> R<Word> {
        let bt = self.t.bool;
        Ok(match op {
            0 => self.b.logical_and(bt, None, a, b)?,
            1 => self.b.logical_or(bt, None, a, b)?,
            _ => self.b.logical_not_equal(bt, None, a, b)?,
        })
    }

    fn fcmp(&mut self, op: u64, a: Word, b: Word) -> R<Word> {
        let bt = self.t.bool;
        Ok(match op {
            0 => self.b.constant_false(bt),
            1 => self.b.f_ord_less_than(bt, None, a, b)?,
            2 => self.b.f_ord_equal(bt, None, a, b)?,
            3 => self.b.f_ord_less_than_equal(bt, None, a, b)?,
            4 => self.b.f_ord_greater_than(bt, None, a, b)?,
            5 => self.b.f_ord_not_equal(bt, None, a, b)?,
            6 => self.b.f_ord_greater_than_equal(bt, None, a, b)?,
            7 | 8 => {
                let na = self.b.is_nan(bt, None, a)?;
                let nb = self.b.is_nan(bt, None, b)?;
                let any = self.b.logical_or(bt, None, na, nb)?;
                if op == 8 { any } else { self.b.logical_not(bt, None, any)? }
            }
            9 => self.b.f_unord_less_than(bt, None, a, b)?,
            10 => self.b.f_unord_equal(bt, None, a, b)?,
            11 => self.b.f_unord_less_than_equal(bt, None, a, b)?,
            12 => self.b.f_unord_greater_than(bt, None, a, b)?,
            13 => self.b.f_unord_not_equal(bt, None, a, b)?,
            14 => self.b.f_unord_greater_than_equal(bt, None, a, b)?,
            _ => self.btrue(),
        })
    }

    fn int_cmp(&mut self, op: u64, signed: bool, a: Word, b: Word) -> R<Word> {
        let bt = self.t.bool;
        Ok(match (op, signed) {
            (0, _) => self.b.constant_false(bt),
            (1, true) => self.b.s_less_than(bt, None, a, b)?,
            (1, false) => self.b.u_less_than(bt, None, a, b)?,
            (2, _) => self.b.i_equal(bt, None, a, b)?,
            (3, true) => self.b.s_less_than_equal(bt, None, a, b)?,
            (3, false) => self.b.u_less_than_equal(bt, None, a, b)?,
            (4, true) => self.b.s_greater_than(bt, None, a, b)?,
            (4, false) => self.b.u_greater_than(bt, None, a, b)?,
            (5, _) => self.b.i_not_equal(bt, None, a, b)?,
            (6, true) => self.b.s_greater_than_equal(bt, None, a, b)?,
            (6, false) => self.b.u_greater_than_equal(bt, None, a, b)?,
            _ => self.btrue(),
        })
    }

    fn fcompare(&mut self, i: &Insn) -> R {
        let a = self.rf(i.ra())?;
        let b = self.src_bf(i)?;
        let (p, n) = i.pred(39, 42);
        let pv = self.p(p, n)?;
        if i.name() == "fsetp" {
            let a = self.fneg_abs(a, i.bit(43), i.bit(7))?;
            let b = self.fneg_abs(b, i.bit(6), i.bit(44))?;
            let c = self.fcmp(i.bits(48, 4), a, b)?;
            let op = i.bits(45, 2);
            let r = self.bool_op(op, c, pv)?;
            let nc = self.b.logical_not(self.t.bool, None, c)?;
            let r2 = self.bool_op(op, nc, pv)?;
            self.setp(i.bits(3, 3) as u8, r)?;
            self.setp(i.bits(0, 3) as u8, r2)
        } else {
            let a = self.fneg_abs(a, i.bit(43), i.bit(54))?;
            let b = self.fneg_abs(b, i.bit(53), i.bit(44))?;
            let c = self.fcmp(i.bits(48, 4), a, b)?;
            let r = self.bool_op(i.bits(45, 2), c, pv)?;
            let v = if i.bit(52) { self.b2u(r, 1.0f32.to_bits(), 0)? } else { self.b2u(r, u32::MAX, 0)? };
            if i.bit(47) {
                self.set_cc(v, None, None)?;
            }
            self.set(i.rd(), v)
        }
    }

    fn icompare(&mut self, i: &Insn) -> R {
        let a = self.r(i.ra())?;
        let b = self.src_b(i)?;
        let (p, n) = i.pred(39, 42);
        let pv = self.p(p, n)?;
        let c = self.int_cmp(i.bits(49, 3), i.bit(48), a, b)?;
        let op = i.bits(45, 2);
        let r = self.bool_op(op, c, pv)?;
        if i.name() == "isetp" {
            let nc = self.b.logical_not(self.t.bool, None, c)?;
            let r2 = self.bool_op(op, nc, pv)?;
            self.setp(i.bits(3, 3) as u8, r)?;
            self.setp(i.bits(0, 3) as u8, r2)
        } else {
            let v = if i.bit(44) { self.b2u(r, 1.0f32.to_bits(), 0)? } else { self.b2u(r, u32::MAX, 0)? };
            if i.bit(47) {
                self.set_cc(v, None, None)?;
            }
            self.set(i.rd(), v)
        }
    }

    fn shift(&mut self, a: Word, b: Word, left: bool, signed: bool, wrap: bool) -> R<Word> {
        let u = self.t.u32;
        let c31 = self.u(31);
        let amount = self.b.bitwise_and(u, None, b, c31)?;
        let shifted = if left {
            self.b.shift_left_logical(u, None, a, amount)?
        } else if signed {
            self.b.shift_right_arithmetic(u, None, a, amount)?
        } else {
            self.b.shift_right_logical(u, None, a, amount)?
        };
        if wrap {
            return Ok(shifted);
        }
        let c32 = self.u(32);
        let big = self.b.u_greater_than_equal(self.t.bool, None, b, c32)?;
        let fill = if signed && !left {
            let ai = self.cast_i(a)?;
            let c31i = self.u(31);
            let s = self.b.shift_right_arithmetic(self.t.i32, None, ai, c31i)?;
            self.cast_u(s)?
        } else {
            self.u(0)
        };
        Ok(self.b.select(u, None, big, fill, shifted)?)
    }

    fn lop3(&mut self, a: Word, b: Word, c: Word, lut: u32) -> R<Word> {
        let u = self.t.u32;
        let mut acc = self.u(0);
        let (na, nb, nc) = (self.b.not(u, None, a)?, self.b.not(u, None, b)?, self.b.not(u, None, c)?);
        for k in 0..8u32 {
            if lut >> k & 1 == 0 {
                continue;
            }
            let x = if k & 4 != 0 { a } else { na };
            let y = if k & 2 != 0 { b } else { nb };
            let z = if k & 1 != 0 { c } else { nc };
            let xy = self.b.bitwise_and(u, None, x, y)?;
            let xyz = self.b.bitwise_and(u, None, xy, z)?;
            acc = self.b.bitwise_or(u, None, acc, xyz)?;
        }
        Ok(acc)
    }

    fn xmad(&mut self, i: &Insn) -> R {
        let u = self.t.u32;
        let a = self.r(i.ra())?;
        let (b, c, hi_b, psl, mrg, mode, b_imm) = match (i.form(), i.op.value) {
            (Form::Reg, _) => (self.r(i.rb())?, self.r(i.rc())?, i.bit(35), i.bit(36), i.bit(37), i.bits(50, 3), false),
            (Form::Imm, _) => (self.u(i.bits(20, 16) as u32), self.r(i.rc())?, false, i.bit(36), i.bit(37), i.bits(50, 3), true),
            (Form::Cbuf, 0x5100000000000000) => {
                let b = self.r(i.rc())?;
                let (s, o) = i.cbuf();
                let c = self.cbuf_imm(s, o)?;
                (b, c, i.bit(52), false, false, i.bits(50, 2), false)
            }
            (Form::Cbuf, _) => {
                let (s, o) = i.cbuf();
                let b = self.cbuf_imm(s, o)?;
                let c = self.r(i.rc())?;
                (b, c, i.bit(52), i.bit(55), i.bit(56), i.bits(50, 2), false)
            }
        };
        let (signed_a, signed_b) = (i.bit(48), i.bit(49) && !b_imm);
        let a16 = self.half16(a, i.bit(53), signed_a)?;
        let b16 = self.half16(b, hi_b, signed_b)?;
        let mut prod = self.b.i_mul(u, None, a16, b16)?;
        if psl {
            let s = self.u(16);
            prod = self.b.shift_left_logical(u, None, prod, s)?;
        }
        let sixteen = self.u(16);
        let c = match mode {
            1 => {
                let m = self.u(0xffff);
                self.b.bitwise_and(u, None, c, m)?
            }
            2 => self.b.shift_right_logical(u, None, c, sixteen)?,
            4 => {
                let bs = self.b.shift_left_logical(u, None, b, sixteen)?;
                self.b.i_add(u, None, c, bs)?
            }
            3 => {
                self.unsupported.push("xmad.csfu");
                c
            }
            _ => c,
        };
        let mut v = self.b.i_add(u, None, prod, c)?;
        if mrg {
            let m = self.u(0xffff);
            let lo = self.b.bitwise_and(u, None, v, m)?;
            let bs = self.b.shift_left_logical(u, None, b, sixteen)?;
            v = self.b.bitwise_or(u, None, lo, bs)?;
        }
        self.set(i.rd(), v)
    }

    fn half16(&mut self, v: Word, high: bool, signed: bool) -> R<Word> {
        let u = self.t.u32;
        let off = self.u(if high { 16 } else { 0 });
        let len = self.u(16);
        if signed {
            let vi = self.cast_i(v)?;
            let r = self.b.bit_field_s_extract(self.t.i32, None, vi, off, len)?;
            self.cast_u(r)
        } else {
            Ok(self.b.bit_field_u_extract(u, None, v, off, len)?)
        }
    }

    fn int_source(&mut self, v: Word, size_code: u64, signed: bool, byte: u64) -> R<Word> {
        let width = match size_code {
            0 => 8,
            1 => 16,
            _ => 32,
        };
        if width == 32 {
            return Ok(v);
        }
        let off = self.u((byte * 8) as u32);
        let len = self.u(width);
        if signed {
            let vi = self.cast_i(v)?;
            let r = self.b.bit_field_s_extract(self.t.i32, None, vi, off, len)?;
            self.cast_u(r)
        } else {
            Ok(self.b.bit_field_u_extract(self.t.u32, None, v, off, len)?)
        }
    }

    fn i2f(&mut self, i: &Insn) -> R {
        let b = self.src_b(i)?;
        let signed = i.bit(13);
        let size = i.bits(10, 2);
        let v = self.int_source(b, size, signed, i.bits(41, 2))?;
        let mut f = if signed {
            let vi = self.cast_i(v)?;
            self.b.convert_s_to_f(self.t.f32, None, vi)?
        } else {
            self.b.convert_u_to_f(self.t.f32, None, v)?
        };
        f = self.fneg_abs(f, i.bit(45), i.bit(49))?;
        match i.bits(8, 2) {
            2 => self.setf(i.rd(), f),
            1 => {
                let h = self.pack_half(f)?;
                self.set(i.rd(), h)
            }
            _ => {
                self.unsupported.push("i2f (64-bit)");
                Ok(())
            }
        }
    }

    fn f2i(&mut self, i: &Insn) -> R {
        let b = self.src_bf(i)?;
        let src = match i.bits(10, 2) {
            1 => {
                let raw = self.src_b(i)?;
                self.unpack_half(raw, false)?
            }
            2 => b,
            _ => {
                self.unsupported.push("f2i (64-bit)");
                b
            }
        };
        let src = self.fneg_abs(src, i.bit(45), i.bit(49))?;
        let rounded = match i.bits(39, 2) {
            1 => self.ext(self.t.f32, glsl::FLOOR, &[src])?,
            2 => self.ext(self.t.f32, glsl::CEIL, &[src])?,
            3 => self.ext(self.t.f32, glsl::TRUNC, &[src])?,
            _ => self.ext(self.t.f32, glsl::ROUND_EVEN, &[src])?,
        };
        let signed = i.bit(12);
        let (lo, hi) = match (i.bits(8, 2), signed) {
            (0, false) => (0.0, 255.0),
            (1, false) => (0.0, 65535.0),
            (0, true) => (-128.0, 127.0),
            (1, true) => (-32768.0, 32767.0),
            (_, true) => (-2147483648.0, 2147483520.0),
            (_, false) => (0.0, 4294967040.0),
        };
        let (l, h) = (self.f(lo), self.f(hi));
        let clamped = self.ext(self.t.f32, glsl::FCLAMP, &[rounded, l, h])?;
        let v = if signed {
            let r = self.b.convert_f_to_s(self.t.i32, None, clamped)?;
            self.cast_u(r)?
        } else {
            self.b.convert_f_to_u(self.t.u32, None, clamped)?
        };
        self.set(i.rd(), v)
    }

    fn f2f(&mut self, i: &Insn) -> R {
        let src = match i.bits(10, 2) {
            1 => {
                let raw = self.src_b(i)?;
                self.unpack_half(raw, i.bit(41))?
            }
            2 => self.src_bf(i)?,
            _ => {
                self.unsupported.push("f2f (64-bit)");
                return Ok(());
            }
        };
        let v = self.fneg_abs(src, i.bit(45), i.bit(49))?;
        let v = match (i.bits(39, 2), i.bit(42)) {
            (0, true) => self.ext(self.t.f32, glsl::ROUND_EVEN, &[v])?,
            (1, true) => self.ext(self.t.f32, glsl::FLOOR, &[v])?,
            (2, true) => self.ext(self.t.f32, glsl::CEIL, &[v])?,
            (3, true) => self.ext(self.t.f32, glsl::TRUNC, &[v])?,
            _ => v,
        };
        let v = self.saturate(v, i.bit(50))?;
        match i.bits(8, 2) {
            1 => {
                let h = self.pack_half(v)?;
                self.set(i.rd(), h)
            }
            2 => self.setf(i.rd(), v),
            _ => {
                self.unsupported.push("f2f (64-bit)");
                Ok(())
            }
        }
    }

    fn i2i(&mut self, i: &Insn) -> R {
        let b = self.src_b(i)?;
        let v = self.int_source(b, i.bits(10, 2), i.bit(13), i.bits(41, 2))?;
        let v = if i.bit(49) && i.bit(13) {
            let vi = self.cast_i(v)?;
            let a = self.ext(self.t.i32, glsl::SABS, &[vi])?;
            self.cast_u(a)?
        } else {
            v
        };
        let v = self.ineg(v, i.bit(45))?;
        let v = match i.bits(8, 2) {
            0 => {
                let m = self.u(0xff);
                self.b.bitwise_and(self.t.u32, None, v, m)?
            }
            1 => {
                let m = self.u(0xffff);
                self.b.bitwise_and(self.t.u32, None, v, m)?
            }
            _ => v,
        };
        self.set(i.rd(), v)
    }

    fn pack_half(&mut self, f: Word) -> R<Word> {
        let zero = self.f(0.0);
        let v2 = self.b.composite_construct(self.t.v2f, None, [f, zero])?;
        self.ext(self.t.u32, glsl::PACK_HALF_2X16, &[v2])
    }

    fn unpack_h2(&mut self, raw: Word) -> R<Word> {
        let u = self.t.u32;
        let bt = self.t.bool;
        let mut out = [0; 2];
        for (k, o) in out.iter_mut().enumerate() {
            let h = if k == 0 {
                let m = self.u(0xffff);
                self.b.bitwise_and(u, None, raw, m)?
            } else {
                let sixteen = self.u(16);
                self.b.shift_right_logical(u, None, raw, sixteen)?
            };
            let (c8000, c16, c10, c1f, c3ff, c13, c23, c112, c7f8) =
                (self.u(0x8000), self.u(16), self.u(10), self.u(0x1f), self.u(0x3ff), self.u(13), self.u(23), self.u(112), self.u(0x7f80_0000));
            let sign = self.b.bitwise_and(u, None, h, c8000)?;
            let sign = self.b.shift_left_logical(u, None, sign, c16)?;
            let e = self.b.shift_right_logical(u, None, h, c10)?;
            let e = self.b.bitwise_and(u, None, e, c1f)?;
            let m = self.b.bitwise_and(u, None, h, c3ff)?;
            let mant = self.b.shift_left_logical(u, None, m, c13)?;
            let eb = self.b.i_add(u, None, e, c112)?;
            let eb = self.b.shift_left_logical(u, None, eb, c23)?;
            let normal = self.b.bitwise_or(u, None, sign, eb)?;
            let normal = self.b.bitwise_or(u, None, normal, mant)?;
            let special = self.b.bitwise_or(u, None, sign, c7f8)?;
            let special = self.b.bitwise_or(u, None, special, mant)?;
            let mf = self.b.convert_u_to_f(self.t.f32, None, m)?;
            let scale = self.f(5.960_464_5e-8);
            let d = self.b.f_mul(self.t.f32, None, mf, scale)?;
            let du = self.cast_u(d)?;
            let denormal = self.b.bitwise_or(u, None, du, sign)?;
            let zero = self.u(0);
            let is_den = self.b.i_equal(bt, None, e, zero)?;
            let is_special = self.b.i_equal(bt, None, e, c1f)?;
            let v = self.b.select(u, None, is_special, special, normal)?;
            let v = self.b.select(u, None, is_den, denormal, v)?;
            *o = self.cast_f(v)?;
        }
        Ok(self.b.composite_construct(self.t.v2f, None, out)?)
    }

    fn unpack_half(&mut self, raw: Word, high: bool) -> R<Word> {
        let v2 = self.unpack_h2(raw)?;
        Ok(self.b.composite_extract(self.t.f32, None, v2, [high as u32])?)
    }

    fn halves(&mut self, raw: Word, swizzle: u64) -> R<Word> {
        let v2 = self.unpack_h2(raw)?;
        Ok(match swizzle {
            1 => {
                let f = self.cast_f(raw)?;
                self.b.composite_construct(self.t.v2f, None, [f, f])?
            }
            2 => self.b.vector_shuffle(self.t.v2f, None, v2, v2, [0, 0])?,
            3 => self.b.vector_shuffle(self.t.v2f, None, v2, v2, [1, 1])?,
            _ => v2,
        })
    }

    fn half_neg_abs(&mut self, v: Word, neg: bool, abs: bool) -> R<Word> {
        let mut v = v;
        if abs {
            v = self.ext(self.t.v2f, glsl::FABS, &[v])?;
        }
        if neg {
            v = self.b.f_negate(self.t.v2f, None, v)?;
        }
        Ok(v)
    }

    fn store_halves(&mut self, rd: u8, v: Word, mode: u64, sat: bool) -> R {
        let v = if sat {
            let z = self.f(0.0);
            let o = self.f(1.0);
            let z2 = self.b.composite_construct(self.t.v2f, None, [z, z])?;
            let o2 = self.b.composite_construct(self.t.v2f, None, [o, o])?;
            self.ext(self.t.v2f, glsl::FCLAMP, &[v, z2, o2])?
        } else {
            v
        };
        match mode {
            1 => {
                let f = self.b.composite_extract(self.t.f32, None, v, [0])?;
                self.setf(rd, f)
            }
            2 | 3 => {
                let packed = self.ext(self.t.u32, glsl::PACK_HALF_2X16, &[v])?;
                let old = self.r(rd)?;
                let u = self.t.u32;
                let (keep, take) = if mode == 2 { (0xffff_0000u32, 0x0000_ffffu32) } else { (0x0000_ffff, 0xffff_0000) };
                let (k, t) = (self.u(keep), self.u(take));
                let o = self.b.bitwise_and(u, None, old, k)?;
                let n = self.b.bitwise_and(u, None, packed, t)?;
                let r = self.b.bitwise_or(u, None, o, n)?;
                self.set(rd, r)
            }
            _ => {
                let packed = self.ext(self.t.u32, glsl::PACK_HALF_2X16, &[v])?;
                self.set(rd, packed)
            }
        }
    }

    fn half_operand(&mut self, o: HalfOperand) -> R<Word> {
        let v = match o.src {
            HalfSrc::Reg { reg, swizzle } => {
                let raw = self.r(reg)?;
                self.halves(raw, swizzle as u64)?
            }
            HalfSrc::Cbuf { slot, offset } => {
                let raw = self.cbuf_imm(slot, offset)?;
                self.halves(raw, 1)?
            }
            HalfSrc::Imm(bits) => {
                let raw = self.u(bits);
                self.halves(raw, 0)?
            }
        };
        self.half_neg_abs(v, o.neg, o.abs)
    }

    fn half(&mut self, i: &Insn) -> R {
        let v2 = self.t.v2f;
        let ops = i.half_operands();
        let a = self.half_operand(ops.a)?;
        let b = self.half_operand(ops.b)?;
        let v = match (i.name(), ops.c) {
            ("hadd2", _) => self.b.f_add(v2, None, a, b)?,
            ("hmul2", _) | (_, None) => self.b.f_mul(v2, None, a, b)?,
            (_, Some(c)) => {
                let c = self.half_operand(c)?;
                self.ext(v2, glsl::FMA, &[a, b, c])?
            }
        };
        self.exact(v);
        self.store_halves(i.rd(), v, ops.mode as u64, ops.sat)
    }

    fn hsetp2(&mut self, i: &Insn) -> R {
        let ops = i.half_operands();
        let a = self.half_operand(ops.a)?;
        let b = self.half_operand(ops.b)?;
        let reg_form = i.op.value == 0x5d20000000000000;
        let (cmp, h_and) = if reg_form { (i.bits(35, 4), i.bit(49)) } else { (i.bits(49, 4), i.bit(53)) };
        let f = self.t.f32;
        let (a0, a1) = (self.b.composite_extract(f, None, a, [0])?, self.b.composite_extract(f, None, a, [1])?);
        let (b0, b1) = (self.b.composite_extract(f, None, b, [0])?, self.b.composite_extract(f, None, b, [1])?);
        let c0 = self.fcmp(cmp, a0, b0)?;
        let c1 = self.fcmp(cmp, a1, b1)?;
        let (p, n) = i.pred(39, 42);
        let pv = self.p(p, n)?;
        let op = i.bits(45, 2);
        let (r0, r1) = if h_and {
            let both = self.b.logical_and(self.t.bool, None, c0, c1)?;
            let r = self.bool_op(op, both, pv)?;
            let nb = self.b.logical_not(self.t.bool, None, both)?;
            (r, self.bool_op(op, nb, pv)?)
        } else {
            (self.bool_op(op, c0, pv)?, self.bool_op(op, c1, pv)?)
        };
        self.setp(i.bits(3, 3) as u8, r0)?;
        self.setp(i.bits(0, 3) as u8, r1)
    }

    fn load(&mut self, i: &Insn) -> R {
        match i.op.value {
            0xefd8000000000000 => {
                let base = i.bits(20, 10) as u32;
                let count = i.bits(47, 2) as u32 + 1;
                if i.ra() != RZ {
                    self.unsupported.push("ald (indexed)");
                }
                for k in 0..count {
                    let v = self.attribute_read(base + 4 * k, i.bit(32))?;
                    self.set(i.rd().wrapping_add(k as u8), v)?;
                }
                Ok(())
            }
            0xef90000000000000 => {
                let slot = i.bits(36, 5) as u32;
                let imm = ((i.bits(20, 16) as u32) << 16) as i32 >> 16;
                let base = self.r(i.ra())?;
                let off = self.u(imm as u32);
                let addr = self.b.i_add(self.t.u32, None, base, off)?;
                if i.bits(44, 2) != 0 {
                    self.unsupported.push("ldc (indexed slot)");
                }
                if let Some(used) = self.shader.cbuf_used.get_mut(slot as usize) {
                    *used = u32::MAX;
                }
                match i.bits(48, 3) {
                    size @ 4..=6 => {
                        let four = self.u(4);
                        let mut at = addr;
                        for k in 0..1u8 << (size - 4) {
                            if k > 0 {
                                at = self.b.i_add(self.t.u32, None, at, four)?;
                            }
                            let w = self.cbuf_at(slot, at)?;
                            self.set(i.rd().wrapping_add(k), w)?;
                        }
                        Ok(())
                    }
                    size => {
                        let three = self.u(3);
                        let not3 = self.b.not(self.t.u32, None, three)?;
                        let aligned = self.b.bitwise_and(self.t.u32, None, addr, not3)?;
                        let word = self.cbuf_at(slot, aligned)?;
                        let byte = self.b.bitwise_and(self.t.u32, None, addr, three)?;
                        let eight = self.u(8);
                        let off = self.b.i_mul(self.t.u32, None, byte, eight)?;
                        let len = self.u(if size < 2 { 8 } else { 16 });
                        let v = if size % 2 == 1 {
                            let wi = self.cast_i(word)?;
                            let r = self.b.bit_field_s_extract(self.t.i32, None, wi, off, len)?;
                            self.cast_u(r)?
                        } else {
                            self.b.bit_field_u_extract(self.t.u32, None, word, off, len)?
                        };
                        self.set(i.rd(), v)
                    }
                }
            }
            0xef40000000000000 => {
                let count = match i.bits(48, 3) {
                    5 => 2,
                    6 => 4,
                    _ => 1,
                };
                for k in 0..count {
                    let addr = self.local_address(i, 4 * k)?;
                    let v = match self.local {
                        Some(l) => {
                            let ptr = self.b.access_chain(self.t.fn_u32, None, l, [addr])?;
                            self.b.load(self.t.u32, None, ptr, None, [])?
                        }
                        None => self.u(0),
                    };
                    self.set(i.rd().wrapping_add(k as u8), v)?;
                }
                Ok(())
            }
            0xef48000000000000 => {
                let Some(count) = Self::word_count(i) else {
                    self.unsupported.push("lds (sub-word)");
                    return Ok(());
                };
                for k in 0..count {
                    let v = match self.shared_pointer(i, 4 * k)? {
                        Some(ptr) => self.b.load(self.t.u32, None, ptr, None, [])?,
                        None => self.u(0),
                    };
                    self.set(i.rd().wrapping_add(k as u8), v)?;
                }
                Ok(())
            }
            _ => {
                self.unsupported.push("ld (global/shared)");
                Ok(())
            }
        }
    }

    fn store(&mut self, i: &Insn) -> R {
        match i.op.value {
            0xeff0000000000000 => {
                let base = i.bits(20, 10) as u32;
                let count = i.bits(47, 2) as u32 + 1;
                if i.ra() != RZ {
                    self.unsupported.push("ast (indexed)");
                }
                for k in 0..count {
                    let v = self.r(i.rd().wrapping_add(k as u8))?;
                    self.attribute_write(base + 4 * k, v)?;
                }
                Ok(())
            }
            0xef50000000000000 => {
                let count = match i.bits(48, 3) {
                    5 => 2,
                    6 => 4,
                    _ => 1,
                };
                for k in 0..count {
                    let addr = self.local_address(i, 4 * k)?;
                    let v = self.r(i.rd().wrapping_add(k as u8))?;
                    if let Some(l) = self.local {
                        let ptr = self.b.access_chain(self.t.fn_u32, None, l, [addr])?;
                        self.b.store(ptr, v, None, [])?;
                    }
                }
                Ok(())
            }
            0xef58000000000000 => {
                let Some(count) = Self::word_count(i) else {
                    self.unsupported.push("sts (sub-word)");
                    return Ok(());
                };
                for k in 0..count {
                    let v = self.r(i.rd().wrapping_add(k as u8))?;
                    if let Some(ptr) = self.shared_pointer(i, 4 * k)? {
                        self.b.store(ptr, v, None, [])?;
                    }
                }
                Ok(())
            }
            _ => {
                self.unsupported.push("st (global/shared)");
                Ok(())
            }
        }
    }

    fn word_count(i: &Insn) -> Option<u32> {
        match i.bits(48, 3) {
            4 => Some(1),
            5 => Some(2),
            6 => Some(4),
            _ => None,
        }
    }

    fn shared_pointer(&mut self, i: &Insn, extra: u32) -> R<Option<Word>> {
        let Some((var, elem)) = self.shared else { return Ok(None) };
        let base = self.r(i.ra())?;
        let imm = ((i.bits(20, 24) as u32) << 8) as i32 >> 8;
        let off = self.u((imm as u32).wrapping_add(extra));
        let addr = self.b.i_add(self.t.u32, None, base, off)?;
        let two = self.u(2);
        let w = self.b.shift_right_logical(self.t.u32, None, addr, two)?;
        let max = self.u(self.opts.shared_memory.div_ceil(4).clamp(1, 0x4000) - 1);
        let w = self.ext(self.t.u32, glsl::UMIN, &[w, max])?;
        Ok(Some(self.b.access_chain(elem, None, var, [w])?))
    }

    fn local_address(&mut self, i: &Insn, extra: u32) -> R<Word> {
        let base = self.r(i.ra())?;
        let imm = ((i.bits(20, 24) as u32) << 8) as i32 >> 8;
        let off = self.u((imm as u32).wrapping_add(extra));
        let addr = self.b.i_add(self.t.u32, None, base, off)?;
        let two = self.u(2);
        let w = self.b.shift_right_logical(self.t.u32, None, addr, two)?;
        let max = self.u(self.header.local_memory_size().div_ceil(4).clamp(1, 0x4000) - 1);
        self.ext(self.t.u32, glsl::UMIN, &[w, max])
    }

    fn io_var(&mut self, generic: u32, output: bool) -> (Word, Kind) {
        let map = if output { &self.outputs } else { &self.inputs };
        if let Some(io) = map.get(&generic) {
            return (io.var, io.kind);
        }
        let kind = if !output && self.opts.stage == Stage::Vertex { self.opts.attribute_types[generic as usize] } else { Kind::Float };
        let vec = match kind {
            Kind::Float => self.t.v4f,
            Kind::Sint => self.t.v4i,
            Kind::Uint => self.t.v4u,
        };
        let class = if output { StorageClass::Output } else { StorageClass::Input };
        let ptr = self.b.type_pointer(None, class, vec);
        let var = self.global(ptr, class);
        self.b.decorate(var, Decoration::Location, [Operand::LiteralBit32(generic)]);
        if !output && self.opts.stage == Stage::Fragment {
            match self.header.ps_input(generic as usize, 0) {
                Interpolation::Constant => self.b.decorate(var, Decoration::Flat, []),
                Interpolation::ScreenLinear => self.b.decorate(var, Decoration::NoPerspective, []),
                _ => {}
            }
        }
        self.interface.push(var);
        let map = if output { &mut self.outputs } else { &mut self.inputs };
        map.insert(generic, Io { var, kind });
        if output {
            self.shader.outputs |= 1 << generic;
        } else {
            self.shader.inputs |= 1 << generic;
        }
        (var, kind)
    }

    fn builtin(&mut self, which: BuiltIn, ty: Word, output: bool) -> Word {
        let key = which as u32 | (output as u32) << 31;
        if let Some(&v) = self.builtins.get(&key) {
            return v;
        }
        let class = if output { StorageClass::Output } else { StorageClass::Input };
        let ptr = self.b.type_pointer(None, class, ty);
        let var = self.global(ptr, class);
        self.b.decorate(var, Decoration::BuiltIn, [Operand::BuiltIn(which)]);
        if output && matches!(which, BuiltIn::Position) {
            self.b.decorate(var, Decoration::Invariant, []);
        }
        self.interface.push(var);
        self.builtins.insert(key, var);
        var
    }

    fn component(&mut self, var: Word, kind: Kind, class: StorageClass, c: u32) -> R<Word> {
        let scalar = match kind {
            Kind::Float => self.t.f32,
            Kind::Sint => self.t.i32,
            Kind::Uint => self.t.u32,
        };
        let ptr_ty = self.b.type_pointer(None, class, scalar);
        let idx = self.u(c);
        let ptr = self.b.access_chain(ptr_ty, None, var, [idx])?;
        let v = self.b.load(scalar, None, ptr, None, [])?;
        if kind == Kind::Uint { Ok(v) } else { self.cast_u(v) }
    }

    fn attribute_read(&mut self, addr: u32, _output: bool) -> R<Word> {
        match addr {
            attr::GENERIC..attr::GENERIC_END => {
                let g = (addr - attr::GENERIC) / 16;
                let (var, kind) = self.io_var(g, false);
                self.component(var, kind, StorageClass::Input, (addr % 16) / 4)
            }
            attr::POSITION..=0x7c if self.opts.stage == Stage::Fragment => {
                let v4f = self.t.v4f;
                let var = self.builtin(BuiltIn::FragCoord, v4f, false);
                self.component(var, Kind::Float, StorageClass::Input, (addr - attr::POSITION) / 4)
            }
            attr::POINT_COORD | 0x2e4 if self.opts.stage == Stage::Fragment => {
                let v2f = self.t.v2f;
                let var = self.builtin(BuiltIn::PointCoord, v2f, false);
                self.component(var, Kind::Float, StorageClass::Input, (addr - attr::POINT_COORD) / 4)
            }
            attr::VERTEX_ID | attr::INSTANCE_ID if self.opts.stage == Stage::Vertex => {
                let which = if addr == attr::VERTEX_ID { BuiltIn::VertexIndex } else { BuiltIn::InstanceIndex };
                let i32_ = self.t.i32;
                let var = self.builtin(which, i32_, false);
                let v = self.b.load(self.t.i32, None, var, None, [])?;
                self.cast_u(v)
            }
            attr::FRONT_FACING if self.opts.stage == Stage::Fragment => {
                let bt = self.t.bool;
                let var = self.builtin(BuiltIn::FrontFacing, bt, false);
                let v = self.b.load(self.t.bool, None, var, None, [])?;
                self.b2u(v, u32::MAX, 0)
            }
            _ => {
                self.unsupported.push("attribute read (special)");
                Ok(self.u(0))
            }
        }
    }

    fn attribute_write(&mut self, addr: u32, v: Word) -> R {
        match addr {
            attr::POSITION..=0x7c => {
                let v4f = self.t.v4f;
                let var = self.builtin(BuiltIn::Position, v4f, true);
                let f = self.cast_f(v)?;
                let ptr_ty = self.b.type_pointer(None, StorageClass::Output, self.t.f32);
                let idx = self.u((addr - attr::POSITION) / 4);
                let ptr = self.b.access_chain(ptr_ty, None, var, [idx])?;
                self.b.store(ptr, f, None, [])?;
                Ok(())
            }
            attr::GENERIC..attr::GENERIC_END => {
                let g = (addr - attr::GENERIC) / 16;
                let (var, _) = self.io_var(g, true);
                let f = self.cast_f(v)?;
                let ptr_ty = self.b.type_pointer(None, StorageClass::Output, self.t.f32);
                let idx = self.u((addr % 16) / 4);
                let ptr = self.b.access_chain(ptr_ty, None, var, [idx])?;
                self.b.store(ptr, f, None, [])?;
                Ok(())
            }
            attr::POINT_SIZE | 0x64 | 0x68 | 0x2c0..=0x2dc => Ok(()),
            _ => {
                self.unsupported.push("attribute write (special)");
                Ok(())
            }
        }
    }

    fn ipa(&mut self, i: &Insn) -> R {
        if i.bit(38) {
            self.unsupported.push("ipa (indexed)");
            return Ok(());
        }
        let addr = i.bits(28, 10) as u32;
        let v = match addr {
            attr::POSITION..=0x7c => {
                let v4f = self.t.v4f;
                let var = self.builtin(BuiltIn::FragCoord, v4f, false);
                self.component(var, Kind::Float, StorageClass::Input, (addr - attr::POSITION) / 4)?
            }
            _ => self.attribute_read(addr, false)?,
        };
        let f = self.cast_f(v)?;
        let generic = (attr::GENERIC..attr::GENERIC_END).contains(&addr);
        let perspective = generic && self.header.ps_input(((addr - attr::GENERIC) / 16) as usize, ((addr % 16) / 4) as usize) == Interpolation::Perspective;
        let mode = i.bits(54, 2);
        let f = if mode <= 1 && perspective {
            let v4f = self.t.v4f;
            let var = self.builtin(BuiltIn::FragCoord, v4f, false);
            let w = self.component(var, Kind::Float, StorageClass::Input, 3)?;
            let w = self.cast_f(w)?;
            self.b.f_mul(self.t.f32, None, f, w)?
        } else {
            f
        };
        let f = if mode == 1 && generic {
            let rb = self.rf(i.rb())?;
            self.b.f_mul(self.t.f32, None, f, rb)?
        } else {
            f
        };
        let f = self.saturate(f, i.bit(51))?;
        self.setf(i.rd(), f)
    }

    fn fragment_outputs(&mut self) -> R {
        let mut reg = 0u8;
        for target in 0..8 {
            let comps: Vec<usize> = (0..4).filter(|&c| self.header.ps_output(target, c)).collect();
            if comps.is_empty() {
                continue;
            }
            self.shader.color_outputs |= comps.iter().fold(0, |m, &c| m | 1 << c) << (4 * target);
            let kind = self.opts.output_types[target];
            let (vec, scalar) = match kind {
                Kind::Float => (self.t.v4f, self.t.f32),
                Kind::Sint => (self.t.v4i, self.t.i32),
                Kind::Uint => (self.t.v4u, self.t.u32),
            };
            let ptr = self.b.type_pointer(None, StorageClass::Output, vec);
            let var = self.global(ptr, StorageClass::Output);
            self.b.decorate(var, Decoration::Location, [Operand::LiteralBit32(target as u32)]);
            self.interface.push(var);
            let mut values = Vec::new();
            for c in 0..4 {
                if let Some(color) = self.opts.debug_color {
                    let v = self.u(color[c].to_bits());
                    values.push(if kind == Kind::Uint { v } else { self.b.bitcast(scalar, None, v)? });
                    continue;
                }
                if let Some((pc, n)) = self.opts.debug_probe.filter(|_| target < 2) {
                    let signed = self.opts.debug_probe_signed;
                    let k = if signed { c } else { c + 3 * target };
                    let v = match (c, self.probe) {
                        (3, _) => self.u(1.0f32.to_bits()),
                        (_, Some(vars)) if pc != u32::MAX => self.b.load(self.t.u32, None, vars[k], None, [])?,
                        _ => self.r(n.saturating_add(k as u8))?,
                    };
                    let v = if signed && target == 1 && c < 3 {
                        let f = self.cast_f(v)?;
                        let f = self.b.f_negate(self.t.f32, None, f)?;
                        self.cast_u(f)?
                    } else {
                        v
                    };
                    values.push(if kind == Kind::Uint { v } else { self.b.bitcast(scalar, None, v)? });
                    continue;
                }
                let v = if comps.contains(&c) {
                    let r = self.r(reg)?;
                    reg += 1;
                    r
                } else if c == 3 {
                    self.u(1.0f32.to_bits())
                } else {
                    self.u(0)
                };
                values.push(if kind == Kind::Uint { v } else { self.b.bitcast(scalar, None, v)? });
            }
            if self.opts.alpha_test && target == 0 && kind == Kind::Float && comps.contains(&3) && self.opts.debug_color.is_none() && self.opts.debug_probe.is_none() {
                self.alpha_test(values[3])?;
            }
            let value = self.b.composite_construct(vec, None, values)?;
            self.b.store(var, value, None, [])?;
        }
        if self.header.ps_writes_depth() {
            self.shader.writes_depth = true;
            let f32_ = self.t.f32;
            let var = self.builtin(BuiltIn::FragDepth, f32_, true);
            let d = self.rf(reg + 1)?;
            let d = if self.opts.depth_unorm16 { self.snap_unorm16(d)? } else { d };
            self.b.store(var, d, None, [])?;
        }
        Ok(())
    }

    fn snap_unorm16(&mut self, v: Word) -> R<Word> {
        let f32_ = self.t.f32;
        let (scale, bias) = (self.f(65535.0), self.f(0.25));
        let scaled = self.ext(f32_, glsl::FMA, &[v, scale, bias])?;
        self.exact(scaled);
        let level = self.ext(f32_, glsl::FLOOR, &[scaled])?;
        let snapped = self.b.f_div(f32_, None, level, scale)?;
        self.exact(snapped);
        Ok(snapped)
    }

    fn alpha_test(&mut self, alpha: Word) -> R {
        let (u32_, f32_, bool_) = (self.t.u32, self.t.f32, self.t.bool);
        let mask = self.b.spec_constant_bit32(u32_, 7);
        self.b.decorate(mask, Decoration::SpecId, [Operand::LiteralBit32(0)]);
        let reference = self.b.spec_constant_bit32(f32_, 0);
        self.b.decorate(reference, Decoration::SpecId, [Operand::LiteralBit32(1)]);
        let zero = self.u(0);
        let mut pass = None;
        for bit in [1, 2, 4] {
            let cmp = match bit {
                1 => self.b.f_ord_less_than(bool_, None, alpha, reference)?,
                2 => self.b.f_ord_equal(bool_, None, alpha, reference)?,
                _ => self.b.f_ord_greater_than(bool_, None, alpha, reference)?,
            };
            let b = self.u(bit);
            let masked = self.b.bitwise_and(u32_, None, mask, b)?;
            let on = self.b.i_not_equal(bool_, None, masked, zero)?;
            let term = self.b.logical_and(bool_, None, on, cmp)?;
            pass = Some(match pass {
                Some(p) => self.b.logical_or(bool_, None, p, term)?,
                None => term,
            });
        }
        let fail = self.b.logical_not(bool_, None, pass.unwrap())?;
        let (then, merge) = (self.b.id(), self.b.id());
        self.b.selection_merge(merge, SelectionControl::NONE)?;
        self.b.branch_conditional(fail, then, merge, [])?;
        self.b.begin_block(Some(then))?;
        self.kill()?;
        self.b.begin_block(Some(merge))?;
        Ok(())
    }

    fn cbuf_source(&self, pc: u32, reg: u8) -> Option<(u32, u32)> {
        const NO_RD: &[&str] = &["st", "stg", "sts", "stl", "sust", "red", "exit", "bra", "brx", "sync", "ssy", "pbk", "brk", "kil", "depbar", "bar", "membar", "nop", "isetp", "fsetp", "psetp", "hsetp2", "csetp", "dsetp"];
        let (_, def) = self.insns.range(..pc).rev().take(512).find(|(_, i)| i.rd() == reg && !NO_RD.contains(&i.name()))?;
        (def.name() == "mov" && def.form() == Form::Cbuf).then(|| def.cbuf())
    }

    fn storage_slot(&mut self, pc: u32, reg: u8) -> Option<u32> {
        let off = self.storage_source(pc, reg, 8)?;
        if let Some(k) = self.storage_descs.iter().position(|&o| o == off) {
            return Some(k as u32);
        }
        if self.storage_descs.len() as u32 >= STORAGE_SLOTS {
            return None;
        }
        self.storage_descs.push(off);
        Some(self.storage_descs.len() as u32 - 1)
    }

    fn storage_source(&self, pc: u32, reg: u8, depth: u32) -> Option<u32> {
        const NO_RD: &[&str] = &["st", "stg", "sts", "stl", "sust", "red", "exit", "bra", "brx", "sync", "ssy", "pbk", "brk", "kil", "depbar", "bar", "membar", "nop", "isetp", "fsetp", "psetp", "hsetp2", "csetp"];
        if reg == 255 {
            return None;
        }
        let window = storage_window(self.opts.stage);
        let (&def_pc, def) = self.insns.range(..pc).rev().take(4096).find(|(_, i)| i.rd() == reg && !NO_RD.contains(&i.name()))?;
        if def.form() == Form::Cbuf {
            let (slot, off) = def.cbuf();
            if slot == 0 && window.contains(&off) && off % 16 == 0 {
                return Some(off);
            }
        }
        if depth == 0 {
            return None;
        }
        let mut sources = vec![def.ra()];
        if def.form() == Form::Reg {
            sources.push(def.rb());
        }
        sources.into_iter().find_map(|r| self.storage_source(def_pc, r, depth - 1))
    }

    fn storage_var(&mut self, slot: u32) -> Word {
        if let Some(&v) = self.storage.get(&slot) {
            return v;
        }
        let arr = self.b.type_runtime_array(self.t.u32);
        self.b.decorate(arr, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
        let st = self.b.type_struct([arr]);
        self.b.decorate(st, Decoration::Block, []);
        self.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let ptr = self.b.type_pointer(None, StorageClass::StorageBuffer, st);
        let var = self.global(ptr, StorageClass::StorageBuffer);
        self.b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(self.opts.storage_binding + slot)]);
        self.interface_storage(var);
        self.storage.insert(slot, var);
        var
    }

    fn interface_storage(&mut self, _var: Word) {
    }

    fn global_memory(&mut self, pc: u32, i: &Insn) -> R {
        let u = self.t.u32;
        let Some(slot) = self.storage_slot(pc, i.ra()) else {
            self.unsupported.push(if i.name() == "ldg" { "ldg (unknown buffer)" } else { "stg (unknown buffer)" });
            return Ok(());
        };
        let var = self.storage_var(slot);
        let addr = self.r(i.ra())?;
        let imm = ((i.bits(20, 24) as u32) << 8) as i32 >> 8;
        let off = self.u(imm as u32);
        let addr = self.b.i_add(u, None, addr, off)?;
        let base = self.cbuf_imm(0, self.storage_descs[slot as usize])?;
        let rel = self.b.i_sub(u, None, addr, base)?;
        let two = self.u(2);
        let word = self.b.shift_right_logical(u, None, rel, two)?;
        let count = match i.bits(48, 3) {
            5 => 2,
            6 | 7 => 4,
            _ => 1,
        };
        if i.bits(48, 3) < 4 {
            self.unsupported.push("ldg/stg (sub-word)");
        }
        let ptr_ty = self.b.type_pointer(None, StorageClass::StorageBuffer, u);
        let zero = self.u(0);
        for k in 0..count {
            let kc = self.u(k);
            let idx = self.b.i_add(u, None, word, kc)?;
            let ptr = self.b.access_chain(ptr_ty, None, var, [zero, idx])?;
            if i.name() == "ldg" {
                let v = self.b.load(u, None, ptr, None, [])?;
                self.set(i.rd().wrapping_add(k as u8), v)?;
            } else {
                let v = self.r(i.rd().wrapping_add(k as u8))?;
                self.b.store(ptr, v, None, [])?;
                self.storage_written.insert(slot);
            }
        }
        Ok(())
    }

    fn enable_subgroups(&mut self) {
        if !self.subgroups {
            self.subgroups = true;
            for c in [Capability::GroupNonUniform, Capability::GroupNonUniformVote, Capability::GroupNonUniformBallot, Capability::GroupNonUniformShuffle] {
                self.b.capability(c);
            }
        }
    }

    fn compute_probe(&mut self, n: u8) -> R {
        let u = self.t.u32;
        let var = self.storage_var(16);
        self.shader.probe = true;
        let index_var = self.builtin(BuiltIn::LocalInvocationIndex, u, false);
        let local = self.b.load(u, None, index_var, None, [])?;
        let wg = self.compute_id(BuiltIn::WorkgroupId)?;
        let count = self.compute_id(BuiltIn::NumWorkgroups)?;
        let (wx, wy) = (self.b.composite_extract(u, None, wg, [0])?, self.b.composite_extract(u, None, wg, [1])?);
        let nx = self.b.composite_extract(u, None, count, [0])?;
        let row = self.b.i_mul(u, None, wy, nx)?;
        let group = self.b.i_add(u, None, row, wx)?;
        let threads = self.u(self.opts.local_size.iter().map(|&n| n.max(1)).product());
        let base = self.b.i_mul(u, None, group, threads)?;
        let thread = self.b.i_add(u, None, base, local)?;
        let four = self.u(4);
        let entry = self.b.i_mul(u, None, thread, four)?;
        let ptr_ty = self.b.type_pointer(None, StorageClass::StorageBuffer, u);
        let zero = self.u(0);
        for k in 0..4u32 {
            let v = if k < 3 { self.r(n.saturating_add(k as u8))? } else { self.u(1.0f32.to_bits()) };
            let kc = self.u(k);
            let idx = self.b.i_add(u, None, entry, kc)?;
            let ptr = self.b.access_chain(ptr_ty, None, var, [zero, idx])?;
            self.b.store(ptr, v, None, [])?;
        }
        Ok(())
    }

    fn compute_id(&mut self, which: BuiltIn) -> R<Word> {
        if self.opts.stage != Stage::Compute {
            self.unsupported.push("s2r (compute value outside compute)");
            return Ok(self.b.constant_null(self.t.v3u));
        }
        let ty = self.t.v3u;
        let var = self.builtin(which, ty, false);
        Ok(self.b.load(ty, None, var, None, [])?)
    }

    fn subgroup_invocation(&mut self) -> R<Word> {
        self.enable_subgroups();
        let u32_ = self.t.u32;
        let var = self.builtin(BuiltIn::SubgroupLocalInvocationId, u32_, false);
        if self.opts.stage == Stage::Fragment && !self.flat_decorated.contains(&var) {
            self.flat_decorated.push(var);
            self.b.decorate(var, Decoration::Flat, []);
        }
        Ok(self.b.load(self.t.u32, None, var, None, [])?)
    }

    fn lane_id(&mut self) -> R<Word> {
        let id = self.subgroup_invocation()?;
        let m = self.u(31);
        Ok(self.b.bitwise_and(self.t.u32, None, id, m)?)
    }

    fn ballot32(&mut self, p: Word) -> R<Word> {
        let subgroup = self.u(3);
        let ballot = self.b.group_non_uniform_ballot(self.t.v4u, None, subgroup, p)?;
        let id = self.subgroup_invocation()?;
        let five = self.u(5);
        let word = self.b.shift_right_logical(self.t.u32, None, id, five)?;
        let word = self.ext(self.t.u32, glsl::UMIN, &[word, five])?;
        let three = self.u(3);
        let word = self.ext(self.t.u32, glsl::UMIN, &[word, three])?;
        Ok(self.b.vector_extract_dynamic(self.t.u32, None, ballot, word)?)
    }

    fn vote(&mut self, i: &Insn) -> R {
        if i.op.value != 0x50d8000000000000 {
            self.unsupported.push("vote (vtg)");
            return Ok(());
        }
        let (p, n) = i.pred(39, 42);
        let pv = self.p(p, n)?;
        let ballot = self.ballot32(pv)?;
        let zero = self.u(0);
        let result = match i.bits(48, 2) {
            0 => {
                let active = self.btrue();
                let active = self.ballot32(active)?;
                self.b.i_equal(self.t.bool, None, ballot, active)?
            }
            1 => self.b.i_not_equal(self.t.bool, None, ballot, zero)?,
            _ => {
                let active = self.btrue();
                let active = self.ballot32(active)?;
                let all = self.b.i_equal(self.t.bool, None, ballot, active)?;
                let none = self.b.i_equal(self.t.bool, None, ballot, zero)?;
                self.b.logical_or(self.t.bool, None, all, none)?
            }
        };
        self.set(i.rd(), ballot)?;
        self.setp(i.bits(45, 3) as u8, result)
    }

    fn shfl(&mut self, i: &Insn) -> R {
        let u = self.t.u32;
        let value = self.r(i.ra())?;
        if i.bits(30, 2) == 3 && i.bit(28) && i.bit(29) && matches!(i.bits(20, 5), 1..=3) && i.bits(34, 13) & 0x1f1f == 0x1c03 {
            let xor = i.bits(20, 5) as u32;
            let t = self.btrue();
            self.setp(i.bits(48, 3) as u8, t)?;
            if self.opts.stage == Stage::Fragment && xor != 3 {
                if !self.derivative_control {
                    self.derivative_control = true;
                    self.b.capability(Capability::DerivativeControl);
                }
                let f = self.cast_f(value)?;
                let ft = self.t.f32;
                let d = if xor == 1 { self.b.d_pdx_fine(ft, None, f)? } else { self.b.d_pdy_fine(ft, None, f)? };
                let lane = self.subgroup_invocation()?;
                let bit = self.u(xor);
                let side = self.b.bitwise_and(u, None, lane, bit)?;
                let zero = self.u(0);
                let first = self.b.i_equal(self.t.bool, None, side, zero)?;
                let plus = self.b.f_add(ft, None, f, d)?;
                let minus = self.b.f_sub(ft, None, f, d)?;
                let v = self.b.select(ft, None, first, plus, minus)?;
                return self.setf(i.rd(), v);
            }
            self.enable_subgroups();
            if !self.quad_ops {
                self.quad_ops = true;
                self.b.capability(Capability::GroupNonUniformQuad);
            }
            let subgroup = self.u(3);
            let direction = self.u(xor - 1);
            let v = self.b.group_non_uniform_quad_swap(u, None, subgroup, value, direction)?;
            return self.set(i.rd(), v);
        }
        let b = if i.bit(28) { self.u(i.bits(20, 5) as u32) } else { self.r(i.rb())? };
        let c = if i.bit(29) { self.u(i.bits(34, 13) as u32) } else { self.r(i.rc())? };
        let lane = self.lane_id()?;
        let m31 = self.u(31);
        let clamp = self.b.bitwise_and(u, None, c, m31)?;
        let eight = self.u(8);
        let seg = self.b.shift_right_logical(u, None, c, eight)?;
        let seg = self.b.bitwise_and(u, None, seg, m31)?;
        let nseg = self.b.not(u, None, seg)?;
        let min_lane = self.b.bitwise_and(u, None, lane, seg)?;
        let hi = self.b.bitwise_and(u, None, clamp, nseg)?;
        let max_lane = self.b.bitwise_or(u, None, min_lane, hi)?;
        let b5 = self.b.bitwise_and(u, None, b, m31)?;
        let bt = self.t.bool;
        let (j, valid) = match i.bits(30, 2) {
            0 => {
                let low = self.b.bitwise_and(u, None, b5, nseg)?;
                let j = self.b.bitwise_or(u, None, min_lane, low)?;
                let ok = self.b.u_less_than_equal(bt, None, j, max_lane)?;
                (j, ok)
            }
            1 => {
                let j = self.b.i_sub(u, None, lane, b5)?;
                let ji = self.cast_i(j)?;
                let mi = self.cast_i(max_lane)?;
                let ok = self.b.s_greater_than_equal(bt, None, ji, mi)?;
                (j, ok)
            }
            2 => {
                let j = self.b.i_add(u, None, lane, b5)?;
                let ok = self.b.u_less_than_equal(bt, None, j, max_lane)?;
                (j, ok)
            }
            _ => {
                let j = self.b.bitwise_xor(u, None, lane, b5)?;
                let ok = self.b.u_less_than_equal(bt, None, j, max_lane)?;
                (j, ok)
            }
        };
        let j = self.b.select(u, None, valid, j, lane)?;
        let id = self.subgroup_invocation()?;
        let n31 = self.u(!31);
        let group = self.b.bitwise_and(u, None, id, n31)?;
        let src = self.b.bitwise_or(u, None, group, j)?;
        let subgroup = self.u(3);
        let v = self.b.group_non_uniform_shuffle(u, None, subgroup, value, src)?;
        self.setp(i.bits(48, 3) as u8, valid)?;
        self.set(i.rd(), v)
    }

    fn atomic(&mut self, pc: u32, i: &Insn) -> R {
        let u = self.t.u32;
        let Some(slot) = self.storage_slot(pc, i.ra()) else {
            self.unsupported.push("red/atom (unknown buffer)");
            return Ok(());
        };
        let (op, size, data) = if i.name() == "red" {
            (i.bits(23, 3), i.bits(20, 3), self.r(i.rd())?)
        } else if i.op.value == 0xed00000000000000 {
            (i.bits(52, 4), i.bits(49, 3), self.r(i.rb())?)
        } else {
            self.unsupported.push("atom (cas)");
            return Ok(());
        };
        if size > 1 {
            self.unsupported.push("red/atom (64-bit or float)");
            return Ok(());
        }
        let var = self.storage_var(slot);
        let addr = self.r(i.ra())?;
        let imm = ((i.bits(28, 20) as u32) << 12) as i32 >> 12;
        let off = self.u(imm as u32);
        let addr = self.b.i_add(u, None, addr, off)?;
        let base = self.cbuf_imm(0, self.storage_descs[slot as usize])?;
        let rel = self.b.i_sub(u, None, addr, base)?;
        let two = self.u(2);
        let word = self.b.shift_right_logical(u, None, rel, two)?;
        let zero = self.u(0);
        let ptr_ty = self.b.type_pointer(None, StorageClass::StorageBuffer, u);
        let ptr = self.b.access_chain(ptr_ty, None, var, [zero, word])?;
        let scope = self.u(1);
        let semantics = self.u(0);
        let signed = size == 1;
        let old = match op {
            0 => self.b.atomic_i_add(u, None, ptr, scope, semantics, data)?,
            1 if signed => self.b.atomic_s_min(u, None, ptr, scope, semantics, data)?,
            1 => self.b.atomic_u_min(u, None, ptr, scope, semantics, data)?,
            2 if signed => self.b.atomic_s_max(u, None, ptr, scope, semantics, data)?,
            2 => self.b.atomic_u_max(u, None, ptr, scope, semantics, data)?,
            5 => self.b.atomic_and(u, None, ptr, scope, semantics, data)?,
            6 => self.b.atomic_or(u, None, ptr, scope, semantics, data)?,
            7 => self.b.atomic_xor(u, None, ptr, scope, semantics, data)?,
            8 => self.b.atomic_exchange(u, None, ptr, scope, semantics, data)?,
            _ => {
                self.unsupported.push("red/atom (inc/dec)");
                return Ok(());
            }
        };
        self.storage_written.insert(slot);
        if i.name() == "atom" {
            self.set(i.rd(), old)?;
        }
        Ok(())
    }

    fn brx(&mut self, pc: u32, i: &Insn) -> R {
        let Some(targets) = self.brx.get(&pc).cloned() else {
            self.unsupported.push("brx");
            return Ok(());
        };
        if self.pc_var.is_none() {
            return Ok(());
        }
        let u = self.t.u32;
        let ra = self.r(i.ra())?;
        let imm = ((i.bits(20, 24) as u32) << 8) as i32 >> 8;
        let base = self.u((pc as i32 + 8 + imm) as u32);
        let addr = self.b.i_add(u, None, ra, base)?;
        let mut value = self.u(EXIT_BLOCK);
        for &t in &targets {
            let block = self.block_number(t);
            let raw = if t % 32 == 8 { t - 8 } else { t };
            for candidate in [raw, t] {
                let c = self.u(candidate);
                let hit = self.b.i_equal(self.t.bool, None, addr, c)?;
                let bc = self.u(block);
                value = self.b.select(u, None, hit, bc, value)?;
            }
        }
        let pc_var = self.pc_var.unwrap();
        self.b.store(pc_var, value, None, [])?;
        self.b.branch(self.dispatch_merge.unwrap())?;
        self.ended = true;
        Ok(())
    }

    fn txq(&mut self, i: &Insn) -> R {
        if i.op.value != 0xdf48000000000000 {
            self.unsupported.push("txq (bindless)");
            return Ok(());
        }
        let handle = i.bits(36, 13) as u32;
        let query = i.bits(22, 6);
        if query != 1 {
            self.unsupported.push("txq (query type)");
            return Ok(());
        }
        let (image_var, _, image_ty, _) = self.texture(None, handle, TexDim::D2, false, false);
        let img = self.b.load(image_ty, None, image_var, None, [])?;
        let lod = self.r(i.ra())?;
        let lod_i = self.cast_i(lod)?;
        let v2i = self.b.type_vector(self.t.i32, 2);
        let size = self.b.image_query_size_lod(v2i, None, img, lod_i)?;
        let w = self.b.composite_extract(self.t.i32, None, size, [0])?;
        let h = self.b.composite_extract(self.t.i32, None, size, [1])?;
        let levels = self.b.image_query_levels(self.t.i32, None, img)?;
        let one = self.u(1);
        let values = [self.cast_u(w)?, self.cast_u(h)?, one, self.cast_u(levels)?];
        let mut reg = i.rd();
        for (c, v) in values.iter().enumerate() {
            if i.bit(31 + c as u32) && reg != RZ {
                self.set(reg, *v)?;
                reg += 1;
            }
        }
        Ok(())
    }

    fn tmml(&mut self, pc: u32, i: &Insn) -> R {
        let (slot, handle) = if i.op.value == 0xdf58000000000000 {
            (None, i.bits(36, 13) as u32)
        } else {
            match self.cbuf_source(pc, i.rb()) {
                Some((slot, offset)) => (Some(slot), offset / 4),
                None => {
                    self.unsupported.push("tmml (bindless)");
                    return Ok(());
                }
            }
        };
        let dim = match i.bits(29, 2) {
            0 => TexDim::D1,
            1 => TexDim::D2,
            2 => TexDim::D3,
            _ => TexDim::Cube,
        };
        let array = i.bit(28);
        let mut reg = i.ra();
        if array && reg != RZ {
            reg += 1;
        }
        let mut coords = Vec::new();
        for _ in 0..Self::coord_count(dim) {
            let v = self.r(reg)?;
            coords.push(self.cast_f(v)?);
            if reg != RZ {
                reg += 1;
            }
        }
        let zero = self.u(0);
        let (computed, accessed) = if self.opts.stage == Stage::Fragment {
            let (image_var, sampler_var, image_ty, sampled_ty) = self.texture(slot, handle, dim, array, false);
            let img = self.b.load(image_ty, None, image_var, None, [])?;
            let sampler_ty = self.b.type_sampler();
            let smp = self.b.load(sampler_ty, None, sampler_var, None, [])?;
            let image = self.b.sampled_image(sampled_ty, None, img, smp)?;
            let coord = if coords.len() == 1 {
                coords[0]
            } else {
                let vty = self.b.type_vector(self.t.f32, coords.len() as u32);
                self.b.composite_construct(vty, None, coords)?
            };
            let v2 = self.b.type_vector(self.t.f32, 2);
            let lod = self.b.image_query_lod(v2, None, image, coord)?;
            let scale = self.f(256.0);
            let mut fixed = [zero; 2];
            for (k, slot) in fixed.iter_mut().enumerate() {
                let v = self.b.composite_extract(self.t.f32, None, lod, [1 - k as u32])?;
                let v = self.b.f_mul(self.t.f32, None, v, scale)?;
                let v = self.b.convert_f_to_s(self.t.i32, None, v)?;
                *slot = self.cast_u(v)?;
            }
            (fixed[0], fixed[1])
        } else {
            (zero, zero)
        };
        let mut reg = i.rd();
        for (c, value) in [computed, accessed, zero, zero].into_iter().enumerate() {
            if i.bit(31 + c as u32) && reg != RZ {
                self.set(reg, value)?;
                reg += 1;
            }
        }
        Ok(())
    }

    fn texture(&mut self, slot: Option<u32>, handle: u32, dim: TexDim, array: bool, shadow: bool) -> (Word, Word, Word, Word) {
        let key = TextureUse { slot, handle, dim, array, shadow, binding: 0 };
        let image_ty = self.image_type(dim, array, shadow);
        let sampled_ty = self.b.type_sampled_image(image_ty);
        if let Some(&(_, var)) = self.textures.iter().find(|(u, _)| u.slot == slot && u.handle == handle && u.dim == dim && u.array == array && u.shadow == shadow) {
            return (var, var + 1, image_ty, sampled_ty);
        }
        let binding = self.opts.texture_binding + 2 * self.textures.len() as u32;
        let image_ptr = self.b.type_pointer(None, StorageClass::UniformConstant, image_ty);
        let sampler_ty = self.b.type_sampler();
        let sampler_ptr = self.b.type_pointer(None, StorageClass::UniformConstant, sampler_ty);
        let image = self.global(image_ptr, StorageClass::UniformConstant);
        let sampler = self.global(sampler_ptr, StorageClass::UniformConstant);
        debug_assert_eq!(sampler, image + 1);
        for (var, b) in [(image, binding), (sampler, binding + 1)] {
            self.b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            self.b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(b)]);
        }
        self.textures.push((TextureUse { binding, ..key }, image));
        (image, sampler, image_ty, sampled_ty)
    }

    fn storage_image(&mut self, slot: Option<u32>, handle: u32, dim: TexDim, array: bool) -> (Word, Word) {
        let d = match dim {
            TexDim::D1 => Dim::Dim1D,
            TexDim::D2 => Dim::Dim2D,
            TexDim::D3 => Dim::Dim3D,
            TexDim::Cube => Dim::DimCube,
        };
        if !self.storage_images {
            self.storage_images = true;
            self.b.capability(Capability::StorageImageWriteWithoutFormat);
            if dim == TexDim::D1 {
                self.b.capability(Capability::Image1D);
            }
        }
        let image_ty = self.b.type_image(self.t.f32, d, 0, array as u32, 0, 2, ImageFormat::Unknown, None);
        if let Some(&(_, var)) = self.images.iter().find(|(u, _)| u.slot == slot && u.handle == handle && u.dim == dim && u.array == array) {
            return (var, image_ty);
        }
        let binding = self.opts.image_binding + self.images.len() as u32;
        let ptr = self.b.type_pointer(None, StorageClass::UniformConstant, image_ty);
        let var = self.global(ptr, StorageClass::UniformConstant);
        self.b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(binding)]);
        self.b.decorate(var, Decoration::NonReadable, []);
        self.images.push((ImageUse { slot, handle, dim, array, binding }, var));
        (var, image_ty)
    }

    fn sust(&mut self, pc: u32, i: &Insn) -> R {
        if i.bit(52) {
            self.unsupported.push("sust (raw)");
            return Ok(());
        }
        let (slot, handle) = if i.bit(51) {
            (None, i.bits(36, 13) as u32)
        } else {
            match self.cbuf_source(pc, i.bits(39, 8) as u8) {
                Some((slot, offset)) => (Some(slot), offset / 4),
                None => {
                    self.unsupported.push("sust (bindless)");
                    return Ok(());
                }
            }
        };
        let (dim, array, coords) = match i.bits(33, 3) {
            0 => (TexDim::D1, false, 1),
            2 => (TexDim::D1, true, 2),
            3 => (TexDim::D2, false, 2),
            4 => (TexDim::D2, true, 3),
            5 => (TexDim::D3, false, 3),
            _ => {
                self.unsupported.push("sust (buffer)");
                return Ok(());
            }
        };
        let (var, image_ty) = self.storage_image(slot, handle, dim, array);
        let mut c = Vec::new();
        for k in 0..coords {
            let v = self.r(i.ra().saturating_add(k))?;
            c.push(v);
        }
        let coord = if c.len() == 1 {
            c[0]
        } else {
            let ty = match c.len() {
                2 => self.t.v2u,
                _ => self.t.v3u,
            };
            self.b.composite_construct(ty, None, c)?
        };
        let mask = i.bits(20, 4) as u32;
        let mut reg = i.rd();
        let mut values = Vec::new();
        for k in 0..4 {
            values.push(if mask >> k & 1 == 1 {
                let v = self.rf(reg)?;
                reg = reg.saturating_add(1);
                v
            } else {
                self.f(if k == 3 { 1.0 } else { 0.0 })
            });
        }
        let texel = self.b.composite_construct(self.t.v4f, None, values)?;
        let image = self.b.load(image_ty, None, var, None, [])?;
        self.b.image_write(image, coord, texel, None, [])?;
        Ok(())
    }

    fn image_type(&mut self, dim: TexDim, array: bool, shadow: bool) -> Word {
        let d = match dim {
            TexDim::D1 => Dim::Dim1D,
            TexDim::D2 => Dim::Dim2D,
            TexDim::D3 => Dim::Dim3D,
            TexDim::Cube => Dim::DimCube,
        };
        if dim == TexDim::D1 && !self.sampled_1d {
            self.sampled_1d = true;
            self.b.capability(Capability::Sampled1D);
        }
        self.b.type_image(self.t.f32, d, shadow as u32, array as u32, 0, 1, ImageFormat::Unknown, None)
    }

    fn coord_count(dim: TexDim) -> u32 {
        match dim {
            TexDim::D1 => 1,
            TexDim::D2 => 2,
            TexDim::D3 | TexDim::Cube => 3,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sample(&mut self, handle: u32, dim: TexDim, array: bool, shadow: bool, coords: Vec<Word>, layer: Option<Word>, lod: Lod, dref: Option<Word>, offset: Option<Word>) -> R<[Word; 4]> {
        let (image_var, sampler_var, image_ty, sampled_ty) = self.texture(None, handle, dim, array, shadow);
        let img = self.b.load(image_ty, None, image_var, None, [])?;
        let sampler_ty = self.b.type_sampler();
        let smp = self.b.load(sampler_ty, None, sampler_var, None, [])?;
        let image = self.b.sampled_image(sampled_ty, None, img, smp)?;
        let mut comps = coords;
        if let Some(packed) = offset
            && dim != TexDim::Cube
        {
            let n = Self::coord_count(dim);
            let level = match lod {
                Lod::Level(l) => {
                    let lf = self.cast_f(l)?;
                    self.b.convert_f_to_s(self.t.i32, None, lf)?
                }
                _ => self.u(0),
            };
            let dims = n + array as u32;
            let size_ty = if dims == 1 { self.t.i32 } else { self.b.type_vector(self.t.i32, dims) };
            let size = self.b.image_query_size_lod(size_ty, None, img, level)?;
            let pi = self.cast_i(packed)?;
            for (k, c) in comps.iter_mut().enumerate().take(n as usize) {
                let off = self.u(4 * k as u32);
                let four = self.u(4);
                let o = self.b.bit_field_s_extract(self.t.i32, None, pi, off, four)?;
                let of = self.b.convert_s_to_f(self.t.f32, None, o)?;
                let s = if dims == 1 { size } else { self.b.composite_extract(self.t.i32, None, size, [k as u32])? };
                let sf = self.b.convert_s_to_f(self.t.f32, None, s)?;
                let d = self.b.f_div(self.t.f32, None, of, sf)?;
                *c = self.b.f_add(self.t.f32, None, *c, d)?;
            }
        }
        self.nvidia_rounding(dim, &mut comps)?;
        if let Some(l) = layer {
            let li = self.cast_i(l)?;
            let lf = self.b.convert_s_to_f(self.t.f32, None, li)?;
            comps.push(lf);
        }
        let coord = if comps.len() == 1 {
            comps[0]
        } else {
            let vty = self.b.type_vector(self.t.f32, comps.len() as u32);
            self.b.composite_construct(vty, None, comps)?
        };
        let implicit_ok = self.opts.stage == Stage::Fragment;
        let (ops, extra): (Option<ImageOperands>, Vec<Operand>) = match lod {
            Lod::Zero => {
                let z = self.f(0.0);
                (Some(ImageOperands::LOD), vec![Operand::IdRef(z)])
            }
            Lod::Level(l) => (Some(ImageOperands::LOD), vec![Operand::IdRef(l)]),
            Lod::Bias(b) if implicit_ok => (Some(ImageOperands::BIAS), vec![Operand::IdRef(b)]),
            Lod::Bias(_) | Lod::Implicit if !implicit_ok => {
                let z = self.f(0.0);
                (Some(ImageOperands::LOD), vec![Operand::IdRef(z)])
            }
            _ => (None, vec![]),
        };
        let explicit = ops == Some(ImageOperands::LOD);
        let result: [Word; 4] = if let Some(d) = dref {
            let df = self.cast_f(d)?;
            let v = if explicit {
                self.b.image_sample_dref_explicit_lod(self.t.f32, None, image, coord, df, ops.unwrap(), extra)?
            } else {
                self.b.image_sample_dref_implicit_lod(self.t.f32, None, image, coord, df, ops, extra)?
            };
            let u = self.cast_u(v)?;
            let one = self.u(1.0f32.to_bits());
            [u, u, u, one]
        } else {
            let v = if explicit {
                self.b.image_sample_explicit_lod(self.t.v4f, None, image, coord, ops.unwrap(), extra)?
            } else {
                self.b.image_sample_implicit_lod(self.t.v4f, None, image, coord, ops, extra)?
            };
            let mut out = [0; 4];
            for (c, o) in out.iter_mut().enumerate() {
                let f = self.b.composite_extract(self.t.f32, None, v, [c as u32])?;
                *o = self.cast_u(f)?;
            }
            out
        };
        Ok(result)
    }

    fn nvidia_rounding(&mut self, dim: TexDim, comps: &mut [Word]) -> R {
        if dim == TexDim::Cube || self.opts.exact_texel_rounding {
            return Ok(());
        }
        let e = self.f(2e-6);
        for c in comps.iter_mut().take(Self::coord_count(dim) as usize) {
            *c = self.b.f_add(self.t.f32, None, *c, e)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gather(&mut self, handle: u32, dim: TexDim, array: bool, coords: Vec<Word>, layer: Option<Word>, component: u32, dref: Option<Word>, offsets: &[Word]) -> R<[Word; 4]> {
        let (image_var, sampler_var, image_ty, sampled_ty) = self.texture(None, handle, dim, array, dref.is_some());
        let img = self.b.load(image_ty, None, image_var, None, [])?;
        let sampler_ty = self.b.type_sampler();
        let smp = self.b.load(sampler_ty, None, sampler_var, None, [])?;
        let image = self.b.sampled_image(sampled_ty, None, img, smp)?;
        let mut comps = coords;
        self.nvidia_rounding(dim, &mut comps)?;
        if let Some(l) = layer {
            let li = self.cast_i(l)?;
            let lf = self.b.convert_s_to_f(self.t.f32, None, li)?;
            comps.push(lf);
        }
        let coord = if comps.len() == 1 {
            comps[0]
        } else {
            let vty = self.b.type_vector(self.t.f32, comps.len() as u32);
            self.b.composite_construct(vty, None, comps)?
        };
        let texel_offsets = match offsets {
            [] => vec![None],
            &[packed] => vec![Some(self.gather_offset(packed, 0)?)],
            _ => (0..4).map(|k| self.gather_offset(offsets[k / 2], 16 * (k as u32 % 2)).map(Some)).collect::<R<_>>()?,
        };
        let mut gathered = Vec::new();
        for offset in texel_offsets {
            let (ops, extra) = match offset {
                None => (None, vec![]),
                Some(v) => (Some(ImageOperands::OFFSET), vec![Operand::IdRef(v)]),
            };
            gathered.push(match dref {
                Some(d) => {
                    let df = self.cast_f(d)?;
                    self.b.image_dref_gather(self.t.v4f, None, image, coord, df, ops, extra)?
                }
                None => {
                    let c = self.u(component);
                    self.b.image_gather(self.t.v4f, None, image, coord, c, ops, extra)?
                }
            });
        }
        let mut out = [0; 4];
        for (c, o) in out.iter_mut().enumerate() {
            let v = gathered[c.min(gathered.len() - 1)];
            let f = self.b.composite_extract(self.t.f32, None, v, [c as u32])?;
            *o = self.cast_u(f)?;
        }
        Ok(out)
    }

    fn gather_offset(&mut self, packed: Word, shift: u32) -> R<Word> {
        let pi = self.cast_i(packed)?;
        let mut c = Vec::new();
        for k in 0..2 {
            let off = self.u(shift + 8 * k);
            let six = self.u(6);
            c.push(self.b.bit_field_s_extract(self.t.i32, None, pi, off, six)?);
        }
        let ty = self.b.type_vector(self.t.i32, 2);
        let v = self.b.composite_construct(ty, None, c)?;
        if !self.gather_extended {
            self.gather_extended = true;
            self.b.capability(Capability::ImageGatherExtended);
        }
        Ok(v)
    }

    fn tld4(&mut self, i: &Insn) -> R {
        if i.op.value != 0xc838000000000000 {
            self.unsupported.push("tld4 (bindless)");
            return Ok(());
        }
        let handle = i.bits(36, 13) as u32;
        let dim = match i.bits(29, 2) {
            0 => TexDim::D1,
            1 => TexDim::D2,
            2 => TexDim::D3,
            _ => TexDim::Cube,
        };
        let array = i.bit(28);
        let mut reg_a = i.ra();
        let mut take_a = |t: &mut Self| -> R<Word> {
            let r = reg_a;
            if reg_a != RZ {
                reg_a += 1;
            }
            t.r(r)
        };
        let layer = if array { Some(take_a(self)?) } else { None };
        let mut coords = Vec::new();
        for _ in 0..Self::coord_count(dim) {
            let v = take_a(self)?;
            coords.push(self.cast_f(v)?);
        }
        let mut reg_b = i.rb();
        let mut take_b = |t: &mut Self| -> R<Word> {
            let r = reg_b;
            if reg_b != RZ {
                reg_b += 1;
            }
            t.r(r)
        };
        let mut offsets = Vec::new();
        for _ in 0..i.bits(54, 2).min(2) {
            offsets.push(take_b(self)?);
        }
        let dref = if i.bit(50) { Some(take_b(self)?) } else { None };
        let result = self.gather(handle, dim, array, coords, layer, i.bits(56, 2) as u32, dref, &offsets)?;
        let mut reg = i.rd();
        for (c, &value) in result.iter().enumerate() {
            if i.bit(31 + c as u32) && reg != RZ {
                self.set(reg, value)?;
                reg += 1;
            }
        }
        Ok(())
    }

    fn tld4s(&mut self, i: &Insn) -> R {
        let handle = i.bits(36, 13) as u32;
        let (ra, rb) = (i.ra(), i.rb());
        let next = |r: u8| if r == RZ { RZ } else { r + 1 };
        let (aoffi, dc) = (i.bit(51), i.bit(50));
        let x = self.rf(ra)?;
        let (y, offset, dref) = if !aoffi && !dc {
            (self.rf(rb)?, None, None)
        } else {
            let y = self.rf(next(ra))?;
            let offset = if aoffi { Some(self.r(rb)?) } else { None };
            let dref = if dc { Some(self.r(if aoffi { next(rb) } else { rb })?) } else { None };
            (y, offset, dref)
        };
        let result = self.gather(handle, TexDim::D2, false, vec![x, y], None, i.bits(52, 2) as u32, dref, offset.as_slice())?;
        let (rd, rd2) = (i.rd(), i.rd2());
        if i.bit(55) {
            for (pair, reg) in result.chunks(2).zip([rd, rd2]) {
                if reg == RZ {
                    continue;
                }
                let lo = self.cast_f(pair[0])?;
                let hi = self.cast_f(pair[1])?;
                let v2 = self.b.composite_construct(self.t.v2f, None, [lo, hi])?;
                let packed = self.ext(self.t.u32, glsl::PACK_HALF_2X16, &[v2])?;
                self.set(reg, packed)?;
            }
            return Ok(());
        }
        for (pair, reg) in result.chunks(2).zip([rd, rd2]) {
            if reg == RZ {
                continue;
            }
            self.set(reg, pair[0])?;
            self.set(reg + 1, pair[1])?;
        }
        Ok(())
    }

    fn texs(&mut self, i: &Insn) -> R {
        let handle = i.bits(36, 13) as u32;
        let (ra, rb) = (i.ra(), i.rb());
        let next = |r: u8| if r == RZ { RZ } else { r + 1 };
        let (dim, array, shadow) = match i.bits(53, 4) {
            0 => (TexDim::D1, false, false),
            1..=3 => (TexDim::D2, false, false),
            4..=6 => (TexDim::D2, false, true),
            7 | 8 => (TexDim::D2, true, false),
            9 => (TexDim::D2, true, true),
            10 | 11 => (TexDim::D3, false, false),
            _ => (TexDim::Cube, false, false),
        };
        let code = i.bits(53, 4);
        let f = |t: &mut Self, r: u8| -> R<Word> { t.rf(r) };
        let mut layer = None;
        let mut dref = None;
        let mut lod = Lod::Implicit;
        let coords = match code {
            0 => {
                lod = Lod::Zero;
                vec![f(self, ra)?]
            }
            1 | 2 => {
                if code == 2 {
                    lod = Lod::Zero;
                }
                vec![f(self, ra)?, f(self, rb)?]
            }
            3 => {
                lod = Lod::Level(f(self, rb)?);
                vec![f(self, ra)?, f(self, next(ra))?]
            }
            4 | 6 => {
                if code == 6 {
                    lod = Lod::Zero;
                }
                dref = Some(self.r(rb)?);
                vec![f(self, ra)?, f(self, next(ra))?]
            }
            5 => {
                lod = Lod::Level(f(self, rb)?);
                dref = Some(self.r(next(rb))?);
                vec![f(self, ra)?, f(self, next(ra))?]
            }
            7 | 8 => {
                if code == 8 {
                    lod = Lod::Zero;
                }
                layer = Some(self.r(ra)?);
                vec![f(self, next(ra))?, f(self, rb)?]
            }
            9 => {
                lod = Lod::Zero;
                layer = Some(self.r(ra)?);
                dref = Some(self.r(next(rb))?);
                vec![f(self, next(ra))?, f(self, rb)?]
            }
            10 | 11 => {
                if code == 11 {
                    lod = Lod::Zero;
                }
                vec![f(self, ra)?, f(self, next(ra))?, f(self, rb)?]
            }
            12 => vec![f(self, ra)?, f(self, next(ra))?, f(self, rb)?],
            13 => {
                lod = Lod::Level(f(self, next(rb))?);
                vec![f(self, ra)?, f(self, next(ra))?, f(self, rb)?]
            }
            _ => {
                self.unsupported.push("texs (unknown target)");
                return Ok(());
            }
        };
        let result = self.sample(handle, dim, array, shadow, coords, layer, lod, dref, None)?;
        self.texs_results(i, result)
    }

    fn tlds(&mut self, i: &Insn) -> R {
        let handle = i.bits(36, 13) as u32;
        let (ra, rb) = (i.ra(), i.rb());
        let next = |r: u8| if r == RZ { RZ } else { r + 1 };
        let (mut dim, mut array, mut lod, mut offset, mut layer) = (TexDim::D2, false, None, None, None);
        let coords = match i.bits(53, 4) {
            0 => {
                dim = TexDim::D1;
                vec![self.r(ra)?]
            }
            1 => {
                dim = TexDim::D1;
                lod = Some(self.r(rb)?);
                vec![self.r(ra)?]
            }
            2 => vec![self.r(ra)?, self.r(rb)?],
            4 => {
                offset = Some(self.r(rb)?);
                vec![self.r(ra)?, self.r(next(ra))?]
            }
            5 => {
                lod = Some(self.r(rb)?);
                vec![self.r(ra)?, self.r(next(ra))?]
            }
            6 => vec![self.r(ra)?, self.r(next(ra))?],
            7 => {
                dim = TexDim::D3;
                vec![self.r(ra)?, self.r(next(ra))?, self.r(rb)?]
            }
            8 => {
                array = true;
                layer = Some(self.r(ra)?);
                vec![self.r(rb)?, self.r(next(rb))?]
            }
            12 => {
                lod = Some(self.r(rb)?);
                offset = Some(self.r(next(rb))?);
                vec![self.r(ra)?, self.r(next(ra))?]
            }
            _ => {
                self.unsupported.push("tlds (unknown target)");
                return Ok(());
            }
        };
        let result = self.fetch(handle, dim, array, coords, layer, lod, offset)?;
        self.texs_results(i, result)
    }

    #[allow(clippy::too_many_arguments)]
    fn fetch(&mut self, handle: u32, dim: TexDim, array: bool, coords: Vec<Word>, layer: Option<Word>, lod: Option<Word>, offset: Option<Word>) -> R<[Word; 4]> {
        let i32t = self.t.i32;
        let (image_var, _, image_ty, _) = self.texture(None, handle, dim, array, false);
        let img = self.b.load(image_ty, None, image_var, None, [])?;
        let mut comps = Vec::new();
        for (k, c) in coords.into_iter().enumerate() {
            let mut c = self.cast_i(c)?;
            if let Some(packed) = offset {
                let pi = self.cast_i(packed)?;
                let (off, four) = (self.u(4 * k as u32), self.u(4));
                let o = self.b.bit_field_s_extract(i32t, None, pi, off, four)?;
                c = self.b.i_add(i32t, None, c, o)?;
            }
            comps.push(c);
        }
        if let Some(l) = layer {
            comps.push(self.cast_i(l)?);
        }
        let coord = if comps.len() == 1 {
            comps[0]
        } else {
            let vty = self.b.type_vector(i32t, comps.len() as u32);
            self.b.composite_construct(vty, None, comps)?
        };
        let level = match lod {
            Some(l) => self.cast_i(l)?,
            None => self.u(0),
        };
        let v = self.b.image_fetch(self.t.v4f, None, img, coord, Some(ImageOperands::LOD), [Operand::IdRef(level)])?;
        let mut out = [0; 4];
        for (c, o) in out.iter_mut().enumerate() {
            let f = self.b.composite_extract(self.t.f32, None, v, [c as u32])?;
            *o = self.cast_u(f)?;
        }
        Ok(out)
    }

    fn texs_results(&mut self, i: &Insn, result: [Word; 4]) -> R {
        let rd2 = i.rd2();
        let comps: &[usize] = match (rd2 == RZ, i.bits(50, 3)) {
            (true, 0) => &[0],
            (true, 1) => &[1],
            (true, 2) => &[2],
            (true, 3) => &[3],
            (true, 4) => &[0, 1],
            (true, 5) => &[0, 3],
            (true, 6) => &[1, 3],
            (true, _) => &[2, 3],
            (false, 0) => &[0, 1, 2],
            (false, 1) => &[0, 1, 3],
            (false, 2) => &[0, 2, 3],
            (false, 3) => &[1, 2, 3],
            (false, _) => &[0, 1, 2, 3],
        };
        if !i.bit(59) {
            for (pair, reg) in comps.chunks(2).zip([i.rd(), rd2]) {
                if reg == RZ {
                    continue;
                }
                let lo = self.cast_f(result[pair[0]])?;
                let hi = match pair.get(1) {
                    Some(&c) => self.cast_f(result[c])?,
                    None => self.f(0.0),
                };
                let v2 = self.b.composite_construct(self.t.v2f, None, [lo, hi])?;
                let packed = self.ext(self.t.u32, glsl::PACK_HALF_2X16, &[v2])?;
                self.set(reg, packed)?;
            }
            return Ok(());
        }
        for (k, &c) in comps.iter().enumerate() {
            let reg = if k < 2 { i.rd().wrapping_add(k as u8) } else { rd2.wrapping_add(k as u8 - 2) };
            if (k < 2 && i.rd() == RZ) || (k >= 2 && rd2 == RZ) {
                continue;
            }
            self.set(reg, result[c])?;
        }
        Ok(())
    }

    fn tex(&mut self, i: &Insn) -> R {
        if i.op.value != 0xc038000000000000 {
            self.unsupported.push("tex (bindless)");
            return Ok(());
        }
        let handle = i.bits(36, 13) as u32;
        let dim = match i.bits(29, 2) {
            0 => TexDim::D1,
            1 => TexDim::D2,
            2 => TexDim::D3,
            _ => TexDim::Cube,
        };
        let array = i.bit(28);
        let shadow = i.bit(50);
        let mut reg_a = i.ra();
        let mut take_a = |t: &mut Self| -> R<Word> {
            let r = reg_a;
            if reg_a != RZ {
                reg_a += 1;
            }
            t.r(r)
        };
        let layer = if array { Some(take_a(self)?) } else { None };
        let mut coords = Vec::new();
        for _ in 0..Self::coord_count(dim) {
            let v = take_a(self)?;
            coords.push(self.cast_f(v)?);
        }
        let mut reg_b = i.rb();
        let mut take_b = |t: &mut Self| -> R<Word> {
            let r = reg_b;
            if reg_b != RZ {
                reg_b += 1;
            }
            t.r(r)
        };
        let lod = match i.bits(55, 3) {
            1 => Lod::Zero,
            2 => {
                let v = take_b(self)?;
                Lod::Bias(self.cast_f(v)?)
            }
            3 => {
                let v = take_b(self)?;
                Lod::Level(self.cast_f(v)?)
            }
            0 => Lod::Implicit,
            _ => {
                self.unsupported.push("tex (lod mode)");
                Lod::Implicit
            }
        };
        let offset = if i.bit(54) { Some(take_b(self)?) } else { None };
        let dref = if shadow { Some(take_b(self)?) } else { None };
        let result = self.sample(handle, dim, array, shadow, coords, layer, lod, dref, offset)?;
        let mut reg = i.rd();
        for (c, &value) in result.iter().enumerate() {
            if i.bit(31 + c as u32) && reg != RZ {
                self.set(reg, value)?;
                reg += 1;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Lod {
    Implicit,
    Zero,
    Level(Word),
    Bias(Word),
}

mod glsl {
    pub const FMA: u32 = 50;
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const FABS: u32 = 4;
    pub const SABS: u32 = 5;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;
    pub const UMIN: u32 = 38;
    pub const SMIN: u32 = 39;
    pub const UMAX: u32 = 41;
    pub const SMAX: u32 = 42;
    pub const FCLAMP: u32 = 43;
    pub const PACK_HALF_2X16: u32 = 58;
    pub const FIND_S_MSB: u32 = 74;
    pub const FIND_U_MSB: u32 = 75;
    pub const NMIN: u32 = 79;
    pub const NMAX: u32 = 80;
}
