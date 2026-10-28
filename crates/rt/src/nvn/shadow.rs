use std::collections::VecDeque;

use crate::gpu::fxhash::FastMap as HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::exec::{Step, Tracker};
use super::resources::{self, Resources};
use super::state::{self, Field};
use crate::cpu::Cpu;
use crate::hle::call_guest;

pub static ON: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, PartialEq)]
pub enum Work {
    Draw(Box<Draw>),
    Dispatch([u32; 3]),
    Clear(Clear),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Clear {
    pub what: u32,
    pub values: [u32; 4],
    pub target: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Draw {
    pub indexed: bool,
    pub count: u32,
    pub start: u64,
    pub base_vertex: i32,
    pub instances: u32,
    pub programs: Vec<u64>,
    pub targets: [u64; 2],
    pub uniforms: Vec<((u8, u8), (u64, u64))>,
    pub vertex: Vec<(u8, u64)>,
    pub textures: Vec<((u8, u16), u64)>,
    pub state: Vec<(Field, u64)>,
    pub texture_addresses: Vec<((u8, u16), u64)>,
    pub vertex_sizes: Vec<(u8, u64)>,
    pub index_type: u8,
    pub base_instance: u32,
    pub mode: u32,
    pub surfaces: Vec<(u8, resources::Surface)>,
    pub all_targets: [u64; 9],
    pub pools: [u64; 2],
    pub program: u64,
    pub program_id: u32,
}

impl Draw {
    fn shape(&self) -> (bool, u32, u64, i32, u32) {
        (self.indexed, self.count, self.start, self.base_vertex, self.instances)
    }
}

#[derive(Clone, Debug)]
pub(super) enum Cmd {
    Program(u64, u32, Arc<Vec<u64>>),
    Uniform { stage: u8, index: u8, range: (u64, u64) },
    Vertex { index: u8, address: u64, size: u64 },
    Texture { stage: u8, index: u16, handle: u64 },
    State(Vec<(Field, u64)>),
    Targets([u64; 9], [Option<resources::Surface>; 9]),
    Clear(Clear),
    Draw(Draw),
    Dispatch([u32; 3]),
    Call(Vec<u64>),
    UniformUpdate { address: u64, data: Arc<Vec<u8>> },
    Storage { stage: u8, index: u8, address: u64, size: u64 },
    Barrier(u64),
    Report { counter: u64, address: u64 },
    Pool { texture: bool, address: u64 },
    Multisample(u64),
    Scratch { address: u64, size: u64 },
}

#[derive(Default)]
struct Model {
    recording: HashMap<u64, Vec<(u64, Cmd)>>,
    lists: HashMap<u64, Arc<Vec<(u64, Cmd)>>>,
    programs: HashMap<u64, Arc<Vec<u64>>>,
    tracker: Tracker,
    exec: Tracker,
    objects: state::Objects,
    learned: state::Learned,
    res: Resources,
    expected: VecDeque<Work>,
    decoded: VecDeque<Work>,
    held: Option<Draw>,
    command_memory: HashMap<u64, u64>,
    missing: HashMap<u64, Vec<&'static str>>,
    pool_vas: HashMap<u64, u64>,
    list_ranges: HashMap<u64, (u64, u64)>,
    stage_constants: HashMap<u64, u64>,
    predicted: Option<Cached>,
    support: (Option<u64>, [Option<u64>; 5]),
    report_rules: HashMap<u64, (u64, u32, u32)>,
    reports_pending: VecDeque<(u64, u64)>,
    unsupported: std::collections::HashSet<u64>,
    unsupported_calls: HashMap<&'static str, u64>,
    stats: Stats,
}

#[derive(Default, Clone)]
struct Stats {
    matched: u64,
    shape: u64,
    programs: u64,
    targets: u64,
    uniforms: u64,
    vertex: u64,
    textures: u64,
    texture_addresses: u64,
    texture_addresses_checked: u64,
    clears: u64,
    clear_values: u64,
    clear_targets: u64,
    clear_layers: u64,
    extra: u64,
    missing: u64,
    unknown_lists: u64,
    uniform_updates: u64,
    executed: u64,
    topology_checked: u64,
    topology_differ: u64,
    constants_checked: u64,
    constants_differ: u64,
    words_checked: u64,
    words_missing: HashMap<String, u64>,
    words_extra: HashMap<&'static str, u64>,
    segments: u64,
    segments_off: u64,
    uniform_bytes: u64,
    uniform_last: u64,
    barriers: u64,
    reports: Vec<(u64, u64)>,
    multisample: u64,
    logged: HashMap<&'static str, u32>,
}

static MODEL: Mutex<Option<Model>> = Mutex::new(None);

type List = Arc<Vec<(u64, Cmd)>>;

pub type Bindings = [[Option<(u64, u32)>; crate::gpu::threed::CB_SLOTS]; 5];

type ConstantBindings = Vec<(usize, usize, (u64, u32))>;

type Cached = (PredictKey, Arc<super::encode::Prediction>, Arc<ConstantBindings>);

fn bind(steps: &mut Vec<Step>, bound: &mut Bindings, group: usize, slot: usize, binding: Option<(u64, u32)>) {
    if bound[group][slot] != binding {
        bound[group][slot] = binding;
        steps.push(Step::Bind { group, slot, binding });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PredictKey {
    tracker: u64,
    learned: u64,
    regs: (u32, u32, u32),
}

fn segment(list: &[(u64, Cmd)], start: u64, end: u64) -> &[(u64, Cmd)] {
    let from = list.partition_point(|c| c.0 < start);
    let to = from + list[from..].partition_point(|c| c.0 < end);
    &list[from..to]
}

struct Pending {
    start: u64,
    end: u64,
    handle: u64,
    list: List,
    supported: bool,
    batch: u64,
    consumed: u64,
}

struct Segments {
    submitted: VecDeque<Pending>,
    open: usize,
    batch: u64,
    words_supported: u64,
    words_in_lists: u64,
    words_elsewhere: u64,
    entries_in_lists: u64,
    entries_elsewhere: u64,
}

impl Segments {
    fn submit(&mut self, lists: Vec<(u64, u64, u64, List, bool)>) {
        self.batch += 1;
        for (start, end, handle, list, supported) in lists {
            self.submitted.push_back(Pending { start, end, handle, list, supported, batch: self.batch, consumed: start });
        }
        while self.submitted.len() > MAX_SUBMITTED {
            self.submitted.pop_front();
            self.open = self.open.saturating_sub(1);
        }
    }

    fn find(&mut self, va: u64, len: u64) -> Option<(u64, List, bool)> {
        let inside = |p: &Pending| va >= p.start && va + len <= p.end;
        let open = self.open.min(self.submitted.len());
        let i = self.submitted.range(open..).position(|p| inside(p) && va >= p.consumed).map(|i| i + open).or_else(|| self.submitted.iter().position(inside))?;
        let p = &mut self.submitted[i];
        p.consumed = p.consumed.max(va + len);
        let (handle, list, supported, batch) = (p.handle, p.list.clone(), p.supported, p.batch);
        while self.submitted.get(self.open).is_some_and(|p| p.consumed >= p.end) {
            self.open += 1;
        }
        while self.submitted.front().is_some_and(|p| p.batch < batch) {
            self.submitted.pop_front();
            self.open = self.open.saturating_sub(1);
        }
        Some((handle, list, supported))
    }
}

const MAX_SUBMITTED: usize = 100_000;

static SEGMENTS: Mutex<Segments> = Mutex::new(Segments { submitted: VecDeque::new(), open: 0, batch: 0, words_supported: 0, words_in_lists: 0, words_elsewhere: 0, entries_in_lists: 0, entries_elsewhere: 0 });

const NOT_RECORDED: [&str; 12] = [
    "nvnCommandBufferInitialize",
    "nvnCommandBufferFinalize",
    "nvnCommandBufferSetDebugLabel",
    "nvnCommandBufferSetMemoryCallback",
    "nvnCommandBufferSetMemoryCallbackData",
    "nvnCommandBufferAddControlMemory",
    "nvnCommandBufferGetCommandMemoryUsed",
    "nvnCommandBufferGetControlMemoryUsed",
    "nvnCommandBufferGetCommandMemoryFree",
    "nvnCommandBufferGetControlMemoryFree",
    "nvnCommandBufferGetCommandMemorySize",
    "nvnCommandBufferGetControlMemorySize",
];

pub fn recorded_elsewhere(name: &str) -> bool {
    name.starts_with("nvnCommandBuffer") && !NOT_RECORDED.contains(&name)
}

pub fn unsupported(name: &'static str, cmdbuf: u64) {
    with(|m| {
        *m.unsupported_calls.entry(name).or_default() += 1;
        if let Some(v) = m.missing.get_mut(&cmdbuf) {
            v.push(name);
        }
    });
}

pub struct Plan {
    handle: u64,
    list: List,
    end: u64,
    va: u64,
    draws: u64,
    clears: u64,
    words: Vec<(u64, u32, &'static str)>,
}

static UPLOADS: Mutex<Vec<(u64, u32)>> = Mutex::new(Vec::new());

pub fn cp_upload(at: u64, bytes: &[u8]) {
    let mut u = UPLOADS.lock().unwrap();
    u.extend(bytes.as_chunks::<4>().0.iter().enumerate().map(|(i, w)| (at + 4 * i as u64, u32::from_le_bytes(*w))));
}

pub fn cp_entry(va: u64, words: u64) -> Option<Plan> {
    let found = {
        let mut s = SEGMENTS.lock().unwrap();
        let found = s.find(va, words * 4);
        match found {
            Some((_, _, supported)) => {
                s.words_in_lists += words;
                if supported {
                    s.words_supported += words;
                }
                s.entries_in_lists += 1;
            }
            None => {
                s.words_elsewhere += words;
                s.entries_elsewhere += 1;
            }
        }
        found
    };
    let (handle, list, supported) = found?;
    if !supported {
        return None;
    }
    {
        let (mut draws, mut clears) = (0, 0);
        let support = with(|m| m.support);
        let mut predicted = Vec::new();
        for (_, cmd) in segment(&list, va, va + words * 4) {
            match cmd {
                Cmd::Draw(d) => {
                    draws += d.instances as u64;
                    if let Some(base) = support.0
                        && (d.base_vertex != 0 || d.base_instance != 0)
                    {
                        predicted.extend([(base + BASE_VERTEX, d.base_vertex as u32, "base vertex"), (base + BASE_VERTEX, 0, "base vertex")]);
                        predicted.extend([(base + BASE_VERTEX + 4, d.base_instance, "base instance"), (base + BASE_VERTEX + 4, 0, "base instance")]);
                    }
                }
                Cmd::Clear(_) => clears += 1,
                Cmd::Texture { stage, index, handle } => {
                    if let Some(base) = stage_group(*stage).and_then(|g| support.1[g as usize]) {
                        predicted.push((base + TEXTURE_HANDLES + 8 * *index as u64, *handle as u32, "texture handle"));
                    }
                }
                Cmd::Storage { index, address, size, .. } => {
                    if let Some(base) = support.0 {
                        let at = base + STORAGE_BUFFERS + 0x10 * *index as u64;
                        predicted.extend([(at, *address as u32, "storage buffer"), (at + 4, (*address >> 32) as u32, "storage buffer"), (at + 8, *size as u32, "storage buffer")]);
                    }
                }
                Cmd::Multisample(_) => {
                    if let Some(base) = support.0 {
                        predicted.extend((0..8).map(|i| (base + 0x620 + 4 * i, 0, "multisample")));
                    }
                }
                Cmd::UniformUpdate { address, data } => {
                    predicted.extend(data.as_chunks::<4>().0.iter().enumerate().map(|(i, w)| (address + 4 * i as u64, u32::from_le_bytes(*w), "uniform update")));
                }
                _ => {}
            }
        }
        UPLOADS.lock().unwrap().clear();
        Some(Plan { handle, list, end: va + words * 4, va, draws, clears, words: predicted })
    }
}

pub fn cp_entry_done(plan: Plan, draws: u64, clears: u64) {
    let uploads = std::mem::take(&mut *UPLOADS.lock().unwrap());
    with(|m| {
        m.reports_pending.clear();
        if super::exec::ON.load(Ordering::Relaxed) {
            m.segment_end();
        }
        let seen: std::collections::HashSet<(u64, u32)> = uploads.iter().copied().collect();
        let predicted: std::collections::HashSet<(u64, u32)> = plan.words.iter().map(|w| (w.0, w.1)).collect();
        m.stats.words_checked += uploads.len() as u64;
        let support = m.support;
        for &(at, value) in &uploads {
            if !predicted.contains(&(at, value)) {
                let key = match support.0 {
                    Some(s0) if at >= s0 && at < s0 + 0x5f00 => format!("driver constants +{:#x}", at - s0),
                    _ if support.1.iter().flatten().any(|&s2| at >= s2 && at < s2 + 0x200) => "texture handles".to_string(),
                    _ => format!("other {at:#x}"),
                };
                *m.stats.words_missing.entry(key).or_default() += 1;
            }
        }
        for &(at, value, kind) in &plan.words {
            if !seen.contains(&(at, value)) {
                *m.stats.words_extra.entry(kind).or_default() += 1;
            }
        }
        m.stats.segments += 1;
        if plan.draws != draws || plan.clears > clears {
            m.stats.segments_off += 1;
            m.log("segments", format!("segment at {:#x} of list {:#x}: model {} draws {} clears, command processor {draws} draws {clears} clears", plan.va, plan.handle, plan.draws, plan.clears));
        }
    });
}

const COMMAND_WRITE: u64 = 0x18;
const COMMAND_BASE: u64 = 0x50;
const COMMAND_GPU_BASE: u64 = 0x60;

fn command_position(c: &Cpu, cmdbuf: u64) -> u64 {
    let read = |at: u64| c.process().memory.read(cmdbuf + at, 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    read(COMMAND_GPU_BASE) + read(COMMAND_WRITE).wrapping_sub(read(COMMAND_BASE))
}

fn with<R>(f: impl FnOnce(&mut Model) -> R) -> R {
    let mut m = MODEL.lock().unwrap();
    f(m.get_or_insert_with(Model::default))
}

#[derive(Clone, Copy, Debug)]
pub enum Hook {
    BeginRecording,
    EndRecording,
    AddCommandMemory,
    CallCommands,
    QueueSubmitCommands,
    ProgramSetShaders,
    BindProgram,
    SetRenderTargets,
    DrawElements,
    DrawElementsBaseVertex,
    DrawElementsInstanced,
    DrawArrays,
    DrawArraysInstanced,
    DispatchCompute,
    BindUniformBuffer,
    BindVertexBuffer,
    BindTexture,
    Setter(state::Setter),
    BindChannelMaskState,
    BindColorState,
    BindBlendState,
    BindDepthStencilState,
    BindPolygonState,
    BindVertexAttribState,
    BindVertexStreamState,
    SetBlendColor,
    SetViewport,
    SetScissor,
    SetPolygonOffset,
    Stencil(state::Kind),
    Flag(state::Kind),
    Float(state::Kind),
    DepthBounds,
    BindMultisampleState,
    Barrier,
    UpdateUniformBuffer,
    BindStorageBuffer,
    ReportCounter,
    SetPool(bool),
    PoolInitialize(bool),
    SetShaderScratchMemory,
    ClearColor,
    ClearDepthStencil,
    Resource(resources::Call),
}

pub fn hook(name: &str) -> Option<Hook> {
    Some(match name {
        "nvnCommandBufferBeginRecording" => Hook::BeginRecording,
        "nvnCommandBufferEndRecording" => Hook::EndRecording,
        "nvnCommandBufferAddCommandMemory" => Hook::AddCommandMemory,
        "nvnCommandBufferCallCommands" | "nvnCommandBufferCopyCommands" => Hook::CallCommands,
        "nvnQueueSubmitCommands" => Hook::QueueSubmitCommands,
        "nvnProgramSetShaders" => Hook::ProgramSetShaders,
        "nvnCommandBufferBindProgram" => Hook::BindProgram,
        "nvnCommandBufferSetRenderTargets" => Hook::SetRenderTargets,
        "nvnCommandBufferDrawElements" => Hook::DrawElements,
        "nvnCommandBufferDrawElementsBaseVertex" => Hook::DrawElementsBaseVertex,
        "nvnCommandBufferDrawElementsInstanced" => Hook::DrawElementsInstanced,
        "nvnCommandBufferDrawArrays" => Hook::DrawArrays,
        "nvnCommandBufferDrawArraysInstanced" => Hook::DrawArraysInstanced,
        "nvnCommandBufferDispatchCompute" => Hook::DispatchCompute,
        "nvnCommandBufferBindUniformBuffer" => Hook::BindUniformBuffer,
        "nvnCommandBufferBindVertexBuffer" => Hook::BindVertexBuffer,
        "nvnCommandBufferBindTexture" => Hook::BindTexture,
        "nvnCommandBufferBindChannelMaskState" => Hook::BindChannelMaskState,
        "nvnCommandBufferBindColorState" => Hook::BindColorState,
        "nvnCommandBufferBindBlendState" => Hook::BindBlendState,
        "nvnCommandBufferBindDepthStencilState" => Hook::BindDepthStencilState,
        "nvnCommandBufferBindPolygonState" => Hook::BindPolygonState,
        "nvnCommandBufferBindVertexAttribState" => Hook::BindVertexAttribState,
        "nvnCommandBufferBindVertexStreamState" => Hook::BindVertexStreamState,
        "nvnCommandBufferSetBlendColor" => Hook::SetBlendColor,
        "nvnCommandBufferSetViewport" => Hook::SetViewport,
        "nvnCommandBufferSetScissor" => Hook::SetScissor,
        "nvnCommandBufferSetPolygonOffsetClamp" => Hook::SetPolygonOffset,
        "nvnCommandBufferSetStencilRef" => Hook::Stencil(state::Kind::StencilRef),
        "nvnCommandBufferSetStencilValueMask" => Hook::Stencil(state::Kind::StencilValueMask),
        "nvnCommandBufferSetStencilMask" => Hook::Stencil(state::Kind::StencilMask),
        "nvnCommandBufferSetDepthClamp" => Hook::Flag(state::Kind::DepthClamp),
        "nvnCommandBufferSetRasterizerDiscard" => Hook::Flag(state::Kind::RasterizerDiscard),
        "nvnCommandBufferSetPointSize" => Hook::Float(state::Kind::PointSize),
        "nvnCommandBufferSetLineWidth" => Hook::Float(state::Kind::LineWidth),
        "nvnCommandBufferSetDepthBounds" => Hook::DepthBounds,
        "nvnCommandBufferBindMultisampleState" => Hook::BindMultisampleState,
        "nvnCommandBufferBarrier" => Hook::Barrier,
        "nvnCommandBufferUpdateUniformBuffer" => Hook::UpdateUniformBuffer,
        "nvnCommandBufferBindStorageBuffer" => Hook::BindStorageBuffer,
        "nvnCommandBufferReportCounter" => Hook::ReportCounter,
        "nvnCommandBufferSetTexturePool" => Hook::SetPool(true),
        "nvnCommandBufferSetSamplerPool" => Hook::SetPool(false),
        "nvnCommandBufferSetShaderScratchMemory" => Hook::SetShaderScratchMemory,
        "nvnTexturePoolInitialize" => Hook::PoolInitialize(true),
        "nvnSamplerPoolInitialize" => Hook::PoolInitialize(false),
        "nvnCommandBufferClearColor" => Hook::ClearColor,
        "nvnCommandBufferClearDepthStencil" => Hook::ClearDepthStencil,
        _ => return state::setter(name).map(Hook::Setter).or_else(|| resources::call(name).map(Hook::Resource)),
    })
}

fn index_size(ty: u64) -> u64 {
    match ty {
        0 => 1,
        1 => 2,
        _ => 4,
    }
}

const TEXTURE_HANDLES: u64 = 0x20;

const BASE_VERTEX: u64 = 0x640;

const STORAGE_BUFFERS: u64 = 0x110;


const MAX_UNIFORM_UPDATE: u64 = 0x10000;

const PROGRAM_HEADER: u64 = 0x30;

const COMPUTE_STAGE: u8 = 5;

const CONTROL_CONSTANTS: u64 = 0x700;

const PROGRAM_ID: u64 = 8;

const CONSTANTS_SIZE: u32 = 0x10000;

fn read_u64s(c: &Cpu, addr: u64, n: u64) -> Vec<u64> {
    let n = n.min(4096);
    c.process().memory.read(addr, 8 * n).map(|b| b.chunks(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect()).unwrap_or_default()
}

fn query(c: &mut Cpu, real: u64, args: &[u64]) -> u64 {
    let (x, v, nzcv) = (c.x, c.v, c.nzcv);
    c.x[..args.len()].copy_from_slice(args);
    call_guest(c, real, false);
    let r = c.x[0];
    (c.x, c.v, c.nzcv) = (x, v, nzcv);
    r
}

pub fn run(hook: Hook, c: &mut Cpu, real: u64, real_of: &dyn Fn(&str) -> Option<u64>) {
    let a = c.x;
    let at = command_position(c, a[0]);
    let record = |cmd: Cmd| with(|m| m.recording.entry(a[0]).or_default().push((at, cmd)));
    match hook {
        Hook::BeginRecording => with(|m| {
            m.recording.insert(a[0], Vec::new());
            m.missing.insert(a[0], Vec::new());
        }),
        Hook::AddCommandMemory => {
            let mem = &c.process().memory;
            let read = |at: u64, len: u64| mem.read(at, len);
            with(|m| {
                if let Some(va) = m.res.pool_address(a[1], &read) {
                    m.command_memory.insert(a[0], va + a[2]);
                }
            });
        }
        Hook::EndRecording => {
            call_guest(c, real, false);
            let handle = c.x[0];
            let used = real_of("nvnCommandBufferGetCommandMemoryUsed").map(|f| query(c, f, &[a[0]]));
            with(|m| {
                let list = m.recording.remove(&a[0]).unwrap_or_default();
                let calls_unsupported = list.iter().any(|(_, c)| matches!(c, Cmd::Call(h) if h.iter().any(|h| m.unsupported.contains(h) || !m.lists.contains_key(h))));
                let unsorted = !list.is_sorted_by_key(|c| c.0);
                if m.missing.remove(&a[0]).is_some_and(|v| !v.is_empty()) || calls_unsupported || unsorted {
                    m.unsupported.insert(handle);
                } else {
                    m.unsupported.remove(&handle);
                }
                m.lists.insert(handle, Arc::new(list));
                if let (Some(&va), Some(used)) = (m.command_memory.get(&a[0]), used)
                    && used > 0
                {
                    m.list_ranges.insert(handle, (va, va + used));
                }
            });
            c.pc = c.x[30];
            return;
        }
        Hook::CallCommands => record(Cmd::Call(read_u64s(c, a[2], a[1]))),
        Hook::QueueSubmitCommands => {
            let handles = read_u64s(c, a[2], a[1]);
            with(|m| {
                let mut flat = Vec::new();
                for &h in &handles {
                    m.flatten(h, 0, &mut flat);
                }
                SEGMENTS.lock().unwrap().submit(flat);
                if !super::exec::ON.load(Ordering::Relaxed) {
                    for h in handles {
                        m.expand(h, 0);
                    }
                    m.compare();
                }
            });
        }
        Hook::ProgramSetShaders => {
            let stages = read_u64s(c, a[2], 2 * a[1]);
            let constants: Vec<(u64, u64)> = stages
                .chunks(2)
                .filter_map(|s| {
                    let delta = c.process().memory.read(s[1] + CONTROL_CONSTANTS, 4)?;
                    Some((s[0] + PROGRAM_HEADER, s[0] + u32::from_le_bytes(delta[..4].try_into().unwrap()) as u64))
                })
                .collect();
            let code: Vec<u64> = stages.chunks(2).map(|s| s[0] + PROGRAM_HEADER).collect();
            with(|m| {
                m.programs.insert(a[0], Arc::new(code));
                m.stage_constants.extend(constants);
            });
        }
        Hook::BindProgram => {
            let p = with(|m| m.programs.get(&a[1]).cloned().unwrap_or_default());
            let id = c.process().memory.read(a[1] + PROGRAM_ID, 4).map_or(u32::MAX, |b| u32::from_le_bytes(b[..4].try_into().unwrap()));
            record(Cmd::Program(a[1], id, p));
        }
        Hook::SetRenderTargets => {
            let n = a[1].min(8);
            let list = |at: u64| if at == 0 { vec![0; n as usize] } else { read_u64s(c, at, n) };
            let (mut textures, mut views) = (list(a[2]), list(a[3]));
            textures.resize(8, 0);
            views.resize(8, 0);
            textures.push(a[4]);
            views.push(a[5]);
            let mut targets = [0u64; 9];
            let mut surfaces = [None; 9];
            for i in 0..9 {
                if textures[i] != 0 {
                    (targets[i], surfaces[i]) = target_address(c, textures[i], views[i], real_of);
                }
            }
            record(Cmd::Targets(targets, surfaces));
        }
        Hook::ClearColor => {
            let words = c.process().memory.read(a[2], 16).unwrap_or_default();
            let mut values = [0u32; 4];
            for (v, w) in values.iter_mut().zip(words.as_chunks::<4>().0) {
                *v = u32::from_le_bytes(*w);
            }
            let what = (a[3] as u32 & 0xf) << 2 | (a[1] as u32 & 0xf) << 6;
            if what & 0x3c != 0 {
                record(Cmd::Clear(Clear { what, values, target: 0 }));
            }
        }
        Hook::ClearDepthStencil => {
            let (depth, stencil) = (a[1] as u8 != 0, a[3] as u8 != 0);
            let what = depth as u32 | (stencil as u32) << 1;
            let values = [if depth { c.v[0][0] as u32 } else { 0 }, if stencil { a[2] as u32 & 0xff } else { 0 }, 0, 0];
            if what != 0 {
                record(Cmd::Clear(Clear { what, values, target: 0 }));
            }
        }
        Hook::Resource(call) => {
            let mem = &c.process().memory;
            let read = |at: u64, len: u64| mem.read(at, len);
            if !resources::after(call) {
                with(|m| m.res.before(call, &a, &read));
            } else {
                let reserved = c.process().gpu.last_reserved();
                call_guest(c, real, false);
                let ret = c.x[0];
                let gpu = &c.process().gpu;
                let reserved = Some(gpu.last_reserved()).filter(|r| r.0 != reserved.0).map(|r| r.1);
                let mem = &c.process().memory;
                let read = |at: u64, len: u64| mem.read(at, len);
                with(|m| {
                    let mut logged = Vec::new();
                    m.res.after(call, &a, ret, &|cpu| gpu.gpu_addresses_of(cpu), &read, &mut |k, s| logged.push((k, s)));
                    if let resources::Call::PoolInitialize = call {
                        m.res.pool_mappings.insert(a[0], gpu.gpu_mappings_of(m.res.pool_memory(a[1])));
                        if let Some(va) = reserved {
                            m.res.reserved(a[0], a[1], va);
                        }
                    }
                    for (k, s) in logged {
                        m.log(k, s);
                    }
                });
                if let resources::Call::TextureInitialize = call
                    && let Some(f) = real_of("nvnTextureGetTextureAddress")
                {
                    let driver = query(c, f, &[a[0]]);
                    with(|m| {
                        if let Some(msg) = m.res.check_texture(a[0], a[1], driver) {
                            m.log("new textures", msg);
                        }
                    });
                }
                if let resources::Call::PoolInitialize = call
                    && let Some(f) = real_of("nvnMemoryPoolGetBufferAddress")
                {
                    let driver = query(c, f, &[a[0]]);
                    with(|m| {
                        if let Some(msg) = m.res.check_pool(a[0], a[1], driver) {
                            m.log("new pools", msg);
                        }
                    });
                }
                c.pc = c.x[30];
                return;
            }
        }
        Hook::DrawElements | Hook::DrawElementsBaseVertex | Hook::DrawElementsInstanced => {
            let (base_vertex, base_instance, instances) = match hook {
                Hook::DrawElements => (0, 0, 1),
                Hook::DrawElementsBaseVertex => (a[5] as i32, 0, 1),
                _ => (a[5] as i32, a[6] as u32, a[7] as u32),
            };
            let _ = index_size(a[2]);
            record(Cmd::Draw(Draw { indexed: true, count: a[3] as u32, start: a[4], base_vertex, base_instance, instances, index_type: a[2] as u8, mode: a[1] as u32, ..Default::default() }));
        }
        Hook::DrawArrays | Hook::DrawArraysInstanced => {
            let (base_instance, instances) = if matches!(hook, Hook::DrawArrays) { (0, 1) } else { (a[4] as u32, a[5] as u32) };
            record(Cmd::Draw(Draw { indexed: false, count: a[3] as u32, start: a[2], base_instance, instances, mode: a[1] as u32, ..Default::default() }));
        }
        Hook::DispatchCompute => record(Cmd::Dispatch([a[1] as u32, a[2] as u32, a[3] as u32])),
        Hook::BindUniformBuffer => record(Cmd::Uniform { stage: a[1] as u8, index: a[2] as u8, range: (a[3], a[4]) }),
        Hook::BindVertexBuffer => record(Cmd::Vertex { index: a[1] as u8, address: a[2], size: a[3] }),
        Hook::BindTexture => record(Cmd::Texture { stage: a[1] as u8, index: a[2] as u16, handle: a[3] }),
        Hook::Setter(s) => with(|m| m.objects.set(s, &a)),
        Hook::BindChannelMaskState => record(Cmd::State(with(|m| m.objects.channel_mask(a[1])))),
        Hook::BindColorState => record(Cmd::State(with(|m| m.objects.color(a[1])))),
        Hook::BindBlendState => record(Cmd::State(with(|m| m.objects.blend(a[1])))),
        Hook::BindDepthStencilState => record(Cmd::State(with(|m| m.objects.depth_stencil(a[1])))),
        Hook::BindPolygonState => record(Cmd::State(with(|m| m.objects.polygon(a[1])))),
        Hook::BindVertexAttribState => record(Cmd::State(with(|m| m.objects.attribs(a[2], a[1])))),
        Hook::BindVertexStreamState => record(Cmd::State(with(|m| m.objects.streams(a[2], a[1])))),
        Hook::SetBlendColor => {
            let words = c.process().memory.read(a[1], 16).unwrap_or_default();
            let color = words.chunks(4).enumerate().map(|(i, w)| (Field(state::Kind::BlendColor, i as u8), u32::from_le_bytes(w.try_into().unwrap()) as u64));
            record(Cmd::State(color.collect()));
        }
        Hook::Stencil(kind) => record(Cmd::State((0..2u8).filter(|f| a[1] >> f & 1 == 1).map(|f| (Field(kind, f), a[2] & 0xff)).collect())),
        Hook::Flag(kind) => record(Cmd::State(vec![(Field(kind, 0), a[1] & 0xff)])),
        Hook::Float(kind) => record(Cmd::State(vec![(Field(kind, 0), c.v[0][0] as u32 as u64)])),
        Hook::DepthBounds => {
            let on = a[1] & 0xff;
            let mut fields = vec![(Field(state::Kind::DepthBoundsEnable, 0), on)];
            if on != 0 {
                fields.push((Field(state::Kind::DepthBoundsMin, 0), c.v[0][0] as u32 as u64));
                fields.push((Field(state::Kind::DepthBoundsMax, 0), c.v[1][0] as u32 as u64));
            }
            record(Cmd::State(fields));
        }
        Hook::BindMultisampleState => record(Cmd::Multisample(a[1])),
        Hook::Barrier => record(Cmd::Barrier(a[1])),
        Hook::UpdateUniformBuffer => {
            let data = c.process().memory.read(a[5], a[4].min(MAX_UNIFORM_UPDATE)).unwrap_or_default();
            record(Cmd::UniformUpdate { address: a[1] + a[3], data: Arc::new(data) });
        }
        Hook::BindStorageBuffer => record(Cmd::Storage { stage: a[1] as u8, index: a[2] as u8, address: a[3], size: a[4] }),
        Hook::ReportCounter => record(Cmd::Report { counter: a[1], address: a[2] }),
        Hook::SetPool(texture) => {
            let address = with(|m| m.pool_vas.get(&a[1]).copied().unwrap_or(0));
            record(Cmd::Pool { texture, address });
        }
        Hook::SetShaderScratchMemory => {
            let mem = &c.process().memory;
            let read = |at: u64, len: u64| mem.read(at, len);
            let address = with(|m| m.res.pool_address(a[1], &read)).map_or(0, |va| va + a[2]);
            record(Cmd::Scratch { address, size: a[3] });
        }
        Hook::PoolInitialize(_) => {
            let mem = &c.process().memory;
            let read = |at: u64, len: u64| mem.read(at, len);
            with(|m| {
                if let Some(va) = m.res.pool_address(a[1], &read) {
                    m.pool_vas.insert(a[0], va + a[2]);
                }
            });
        }
        Hook::SetPolygonOffset => record(Cmd::State((0..3).map(|i| (Field(state::Kind::PolygonOffset, i as u8), c.v[i][0] as u32 as u64)).collect())),
        Hook::SetViewport | Hook::SetScissor => {
            let k = if matches!(hook, Hook::SetViewport) { state::Kind::Viewport } else { state::Kind::Scissor };
            record(Cmd::State((0..4).map(|i| (Field(k, i as u8), a[1 + i] as u32 as i32 as i64 as u64)).collect()));
        }
    }
    c.pc = real;
}

impl Model {
    fn flatten(&self, handle: u64, depth: u32, out: &mut Vec<(u64, u64, u64, List, bool)>) {
        let Some(list) = self.lists.get(&handle) else { return };
        if let Some(&(start, end)) = self.list_ranges.get(&handle) {
            out.push((start, end, handle, list.clone(), !self.unsupported.contains(&handle)));
        }
        for (_, cmd) in list.iter() {
            if let Cmd::Call(handles) = cmd
                && depth < 8
            {
                for &h in handles {
                    self.flatten(h, depth + 1, out);
                }
            }
        }
    }

    fn expand(&mut self, handle: u64, depth: u32) {
        let Some(list) = self.lists.get(&handle).cloned() else {
            self.stats.unknown_lists += 1;
            return;
        };
        for (_, cmd) in list.iter() {
            self.tracker.apply(cmd);
            match cmd {
                Cmd::Clear(cl) => self.expected.push_back(Work::Clear(self.tracker.clear(cl))),
                Cmd::Draw(d) => self.expected.push_back(Work::Draw(Box::new(self.tracker.draw(d)))),
                Cmd::Dispatch(g) => self.expected.push_back(Work::Dispatch(*g)),
                Cmd::Call(handles) if depth < 8 => {
                    for &h in handles {
                        self.expand(h, depth + 1);
                    }
                }
                Cmd::UniformUpdate { address, data } => {
                    self.stats.uniform_updates += 1;
                    self.stats.uniform_bytes += data.len() as u64;
                    self.stats.uniform_last = *address;
                }
                Cmd::Barrier(bits) => self.stats.barriers |= *bits,
                Cmd::Report { counter, address } => self.stats.reports.push((*counter, *address)),
                Cmd::Multisample(state) => self.stats.multisample = *state,
                _ => {}
            }
        }
    }

    fn pending_draw(&self, d: &Draw, instance: u32) -> Option<&Draw> {
        let continues = self.held.as_ref().is_some_and(|h| instance > 0 && instance == h.instances && (Draw { instances: h.instances, ..d.clone() }) == *h);
        let at = (self.held.is_some() && !continues) as usize;
        match self.expected.get(at) {
            Some(Work::Draw(e)) if (e.indexed, e.count, e.start, e.base_vertex) == (d.indexed, d.count, d.start, d.base_vertex) => Some(e),
            _ => None,
        }
    }

    fn decoded(&mut self, w: Work, instance: u32) {
        if let (Work::Draw(d), Some(held)) = (&w, &mut self.held)
            && instance > 0
            && instance == held.instances
            && (Draw { instances: held.instances, ..(**d).clone() }) == *held
        {
            held.instances += 1;
            return;
        }
        if let Some(h) = self.held.take() {
            self.decoded.push_back(Work::Draw(Box::new(h)));
        }
        match w {
            Work::Draw(d) => self.held = Some(*d),
            other => self.decoded.push_back(other),
        }
        self.compare();
    }

    fn constant_bindings(&self, programs: &[u64], read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> Option<ConstantBindings> {
        let mut out = Vec::new();
        for &code in programs {
            let ty = read(code, 4).map_or(99, |b| (u32::from_le_bytes(b[..4].try_into().unwrap()) >> 10 & 0xf) as usize);
            let Some(g) = [None, Some(0), Some(1), Some(2), Some(3), Some(4)].get(ty).copied().flatten() else { continue };
            out.push((g, 1, (*self.stage_constants.get(&code)?, CONSTANTS_SIZE)));
        }
        Some(out)
    }

    fn cached_prediction(&mut self, d: &Draw, regs: &[u32], read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> Result<Cached, String> {
        use crate::gpu::threed::reg;
        let key = PredictKey {
            tracker: self.exec.generation,
            learned: self.learned.generation,
            regs: (regs[reg::CT_SELECT], regs[reg::PROGRAM_REGION], regs[reg::PROGRAM_REGION + 1]),
        };
        if let Some((k, r, b)) = &self.predicted
            && *k == key
        {
            return Ok((key, r.clone(), b.clone()));
        }
        let full = self.exec.draw(&Draw { indexed: false, ..d.clone() });
        let predicted = Arc::new(super::encode::predict_complete(&full, &self.learned, regs, read).map_err(|e| format!("draw: {e}"))?);
        let bindings = Arc::new(self.constant_bindings(&full.programs, read).ok_or_else(|| "program constants unknown".to_string())?);
        self.predicted = Some((key, predicted.clone(), bindings.clone()));
        Ok((key, predicted, bindings))
    }

    fn segment_end(&mut self) {
        if let Some(h) = self.held.take() {
            self.decoded.push_back(Work::Draw(Box::new(h)));
        }
        self.compare();
        self.expected.clear();
        self.decoded.clear();
    }

    fn same(a: &Work, b: &Work) -> bool {
        match (a, b) {
            (Work::Draw(x), Work::Draw(y)) => x.shape() == y.shape(),
            (Work::Dispatch(x), Work::Dispatch(y)) => x == y,
            (Work::Clear(x), Work::Clear(y)) => x.what == y.what,
            _ => false,
        }
    }

    fn compare(&mut self) {
        const AHEAD: usize = 16;
        while let (Some(e), Some(d)) = (self.expected.front(), self.decoded.front()) {
            if Self::same(e, d) {
                let (e, d) = (self.expected.pop_front().unwrap(), self.decoded.pop_front().unwrap());
                self.stats.matched += 1;
                if let (Work::Clear(e), Work::Clear(d)) = (&e, &d) {
                    self.stats.clears += 1;
                    if e.values != d.values {
                        self.stats.clear_values += 1;
                        self.log("clear values", format!("clear values: model {e:x?}, command processor {d:x?}"));
                    }
                    if e.target != d.target {
                        self.stats.clear_targets += 1;
                        self.log("clear targets", format!("clear target: model {e:x?}, command processor {d:x?}"));
                    }
                }
                if let (Work::Draw(e), Work::Draw(d)) = (&e, &d)
                    && !e.state.is_empty()
                {
                    let programs_ok = d.programs.iter().all(|p| *p == 0 || e.programs.contains(p));
                    if !programs_ok {
                        self.stats.programs += 1;
                        self.log("program", format!("program: model {:x?}, command processor {:x?}", e.programs, d.programs));
                    }
                    if e.targets != d.targets {
                        self.stats.targets += 1;
                        self.log("targets", format!("targets: model {:x?}, command processor {:x?} (draw {d:x?})", e.targets, d.targets));
                    }
                    let cp_uniform = |k: &(u8, u8)| d.uniforms.iter().find(|u| u.0 == *k).map(|u| u.1);
                    if let Some(u) = e.uniforms.iter().find(|u| cp_uniform(&u.0).is_none_or(|v| v.0 != u.1.0)) {
                        self.stats.uniforms += 1;
                        let msg = format!("uniform {:?}: model {:x?}, command processor {:x?}", u.0, u.1, cp_uniform(&u.0));
                        self.log("uniforms", msg);
                    }
                    let cp_vertex = |i: u8| d.vertex.iter().find(|v| v.0 == i).map(|v| v.1);
                    if let Some(v) = e.vertex.iter().find(|v| cp_vertex(v.0) != Some(v.1)) {
                        self.stats.vertex += 1;
                        let msg = format!("vertex stream {}: model {:#x}, command processor {:x?}", v.0, v.1, cp_vertex(v.0));
                        self.log("vertex", msg);
                    }
                    for &(f, v) in &e.state {
                        if let Some(&(_, r)) = d.state.iter().find(|x| x.0 == f)
                            && !self.learned.examples.contains_key(&(f.0, v, r))
                        {
                            let blend: Vec<u64> = d.state.iter().filter(|x| x.0.0 == state::Kind::BlendRegister).map(|x| x.1).collect();
                            let format = d.state.iter().find(|x| x.0 == Field(state::Kind::TargetFormat, f.1)).map(|x| x.1);
                            let model: Vec<(Field, u64)> = e.state.iter().filter(|x| x.0.1 == f.1).copied().collect();
                            self.learned.examples.insert((f.0, v, r), format!("field {f:?}, target format {format:x?}, blend registers {blend:x?}, model {model:x?}"));
                        }
                    }
                    for (f, v, learned, now) in self.learned.add(&e.state, &d.state) {
                        let format = d.state.iter().find(|x| x.0 == Field(state::Kind::TargetFormat, f.1)).map(|x| x.1);
                        let blend: Vec<u64> = d.state.iter().filter(|x| x.0.0 == state::Kind::BlendRegister).map(|x| x.1).collect();
                        let model: Vec<(Field, u64)> = e.state.iter().filter(|x| x.0.1 == f.1 && format!("{:?}", x.0.0).starts_with("Blend")).copied().collect();
                        self.log("state", format!("state {f:?}: NVN {v:#x} gave {now:#x}, before {learned:#x} (target format {format:x?}; blend registers {blend:x?}; model {model:x?})"));
                    }
                    let cp_texture = |k: &(u8, u16)| d.textures.iter().find(|t| t.0 == *k).map(|t| t.1);
                    if let Some(t) = e.textures.iter().find(|t| cp_texture(&t.0).is_some_and(|h| h != t.1 & 0xffff_ffff)) {
                        self.stats.textures += 1;
                        let msg = format!(
                            "texture {:?}: model {:#x}, command processor {:x?}; model {:x?}, command processor group {}: {:x?}",
                            t.0,
                            t.1,
                            cp_texture(&t.0),
                            e.textures.iter().filter(|x| x.0.0 == t.0.0).collect::<Vec<_>>(),
                            t.0.0,
                            d.textures.iter().filter(|x| x.0.0 == t.0.0).map(|x| x.1).collect::<Vec<_>>()
                        );
                        self.log("textures", msg);
                    }
                    for &(k, handle) in &e.textures {
                        let Some(&(_, cp)) = d.texture_addresses.iter().find(|x| x.0 == k) else { continue };
                        let id = (handle & 0xfffff) as u32;
                        let Some(&(Some(t), view)) = self.res.registered.get(&id) else { continue };
                        self.stats.texture_addresses_checked += 1;
                        let model = t.address + t.view_offset(&view);
                        if model != cp {
                            self.stats.texture_addresses += 1;
                            self.log("texture addresses", format!("texture {k:?} id {id}: model {model:#x} ({t:x?}, view {view:?}), texture header {cp:#x}"));
                        }
                    }
                }
                continue;
            }
            if let Some(k) = self.expected.iter().take(AHEAD).position(|x| Self::same(x, d)) {
                self.stats.missing += k as u64;
                let skipped: Vec<Work> = self.expected.drain(..k).collect();
                self.log("missing", format!("not decoded: {skipped:?}"));
            } else if let Some(k) = self.decoded.iter().take(AHEAD).position(|x| Self::same(e, x)) {
                self.stats.extra += k as u64;
                let skipped: Vec<Work> = self.decoded.drain(..k).collect();
                self.log("extra", format!("not in the model: {skipped:?}"));
            } else {
                self.stats.shape += 1;
                let (e, d) = (self.expected.pop_front().unwrap(), self.decoded.pop_front().unwrap());
                self.log("differ", format!("differ: model {e:?}, command processor {d:?}"));
            }
        }
        while self.expected.len() > 100_000 {
            self.expected.pop_front();
            self.stats.missing += 1;
        }
        while self.decoded.len() > 100_000 {
            self.decoded.pop_front();
            self.stats.extra += 1;
        }
    }

    fn log(&mut self, kind: &'static str, what: String) {
        let n = self.stats.logged.entry(kind).or_default();
        if *n < 40 {
            *n += 1;
            eprintln!("kirikae: nvn shadow: {what}");
        }
    }
}

pub(super) fn stage_group(stage: u8) -> Option<u8> {
    [0, 4, 3, 1, 2].get(stage as usize).copied()
}

fn target_address(c: &mut Cpu, tex: u64, view: u64, real_of: &dyn Fn(&str) -> Option<u64>) -> (u64, Option<resources::Surface>) {
    let mem = &c.process().memory;
    let read = |at: u64, len: u64| mem.read(at, len);
    let (t, v) = with(|m| (m.res.texture(tex, &read), if view == 0 { resources::View::default() } else { m.res.view(view, &read) }));
    let model = t.map(|t| t.address + t.view_offset(&v));
    let (address, offset) = (real_of("nvnTextureGetTextureAddress"), real_of("nvnTextureGetViewOffset"));
    let driver = address.map(|f| query(c, f, &[tex]));
    let driver_offset = offset.filter(|_| view != 0).map_or(0, |g| query(c, g, &[tex, view]));
    with(|m| {
        m.res.checks.textures += 1;
        if view != 0 {
            m.res.checks.views += 1;
        }
        if let (Some(t), Some(d)) = (t, driver) {
            if t.address != d {
                m.res.checks.textures_differ += 1;
                m.log("texture objects", format!("texture {tex:#x}: model {:#x} ({t:x?}), driver {d:#x}", t.address));
            } else if view != 0 && t.view_offset(&v) != driver_offset {
                m.res.checks.views_differ += 1;
                m.log("views", format!("view of {tex:#x}: model +{:#x} ({t:x?}, {v:?}), driver +{driver_offset:#x}", t.view_offset(&v)));
            }
        }
    });
    (model.or(driver.map(|d| d + driver_offset)).unwrap_or(0), t.map(|t| t.surface(&v)))
}

pub fn cp_clear(regs: &[u32], v: u32) {
    use crate::gpu::threed::{address, reg};
    if v >> 10 & 0x7ff != 0 {
        with(|m| m.stats.clear_layers += 1);
        return;
    }
    let what = v & 0x3ff;
    let (values, target) = if what & 0x3c != 0 {
        let rt = reg::COLOR_TARGET + 16 * (v >> 6 & 0xf) as usize;
        ([regs[reg::CLEAR_COLOR], regs[reg::CLEAR_COLOR + 1], regs[reg::CLEAR_COLOR + 2], regs[reg::CLEAR_COLOR + 3]], address(regs, rt))
    } else {
        let depth = if v & 1 == 1 { regs[0x364] } else { 0 };
        let stencil = if v & 2 == 2 { regs[0x368] & 0xff } else { 0 };
        ([depth, stencil, 0, 0], if regs[0x54e] & 1 == 1 { address(regs, reg::ZT_A) } else { 0 })
    };
    with(|m| m.decoded(Work::Clear(Clear { what, values, target }), 0));
}

pub fn cp_draw(regs: &[u32], cbs: &[[Option<(u64, u32)>; crate::gpu::threed::CB_SLOTS]; 5], read: &dyn Fn(u64, u64) -> Option<Vec<u8>>, draw: &crate::gpu::threed::Draw) {
    use crate::gpu::threed::{address, reg, stage_enabled};
    if super::exec::ON.load(Ordering::Relaxed)
        && with(|m| {
            m.support = (cbs[0][0].map(|b| b.0), std::array::from_fn(|g| cbs[g][2].map(|b| b.0)));
            m.expected.is_empty() && m.held.is_none()
        })
    {
        return;
    }
    let base = address(regs, reg::PROGRAM_REGION);
    let programs: Vec<u64> = [1, 5].iter().filter(|&&s| stage_enabled(regs, s)).map(|&s| base + regs[reg::PIPELINE + 16 * s + 1] as u64).collect();
    let (start, base_vertex) = match draw.index_size {
        Some(size) => (address(regs, reg::INDEX_BUFFER) + draw.first as u64 * size as u64, regs[reg::VB_ELEMENT_BASE] as i32),
        None => (draw.first as u64, 0),
    };
    let depth = if regs[0x54e] & 1 == 1 { address(regs, 0x3f8) } else { 0 };
    let ct = regs[reg::CT_SELECT];
    let rt0 = reg::COLOR_TARGET + 16 * (ct >> 4 & 7) as usize;
    let color = if ct & 0xf > 0 && regs[rt0 + 4] != 0 { address(regs, rt0) } else { 0 };
    let mut uniforms = Vec::new();
    let mut textures = Vec::new();
    let mut texture_addresses = Vec::new();
    let pool = address(regs, reg::TEX_HEADER_POOL);
    let bindless = (regs[reg::BINDLESS_TEXTURE] & 0x1f) as usize;
    for g in 0..5u8 {
        for (slot, b) in cbs[g as usize].iter().enumerate() {
            if let Some((addr, size)) = b {
                uniforms.push(((g, slot as u8), (*addr, *size as u64)));
            }
        }
        if let Some((addr, size)) = cbs[g as usize].get(bindless).copied().flatten()
            && let Some(bytes) = read(addr + TEXTURE_HANDLES, (size as u64).saturating_sub(TEXTURE_HANDLES).min(8 * 32))
        {
            for (i, w) in bytes.as_chunks::<8>().0.iter().enumerate() {
                let handle = u32::from_le_bytes(w[..4].try_into().unwrap());
                textures.push(((g, i as u16), handle as u64));
                if handle != 0
                    && let Some(tic) = read(pool + 32 * (handle & 0xfffff) as u64, 32)
                {
                    let mut words = [0u32; 8];
                    for (w, b) in words.iter_mut().zip(tic.as_chunks::<4>().0) {
                        *w = u32::from_le_bytes(*b);
                    }
                    texture_addresses.push(((g, i as u16), crate::gpu::tic::texture_header(&words).address));
                }
            }
        }
    }
    let vertex: Vec<(u8, u64)> = (0..16u8).map(|s| (s, address(regs, reg::VERTEX_STREAM + 4 * s as usize + 1))).collect();
    let state = state::from_registers(regs);
    let d = Draw { indexed: draw.index_size.is_some(), count: draw.count, start, base_vertex, instances: 1, programs, targets: [color, depth], uniforms, vertex, textures, state, texture_addresses, ..Default::default() };
    with(|m| {
        m.support = (cbs[0][0].map(|b| b.0), std::array::from_fn(|g| cbs[g][2].map(|b| b.0)));
        if let Some(mode) = m.pending_draw(&d, draw.instance).map(|e| e.mode) {
            m.stats.topology_checked += 1;
            if mode != draw.topology {
                m.stats.topology_differ += 1;
                let msg = format!("draw mode {mode:#x}, command processor topology {:#x}", draw.topology);
                m.log("topology", msg);
            }
        }
        if let Some(e) = m.pending_draw(&d, draw.instance).filter(|e| e.program != 0) {
            let (program, codes) = (e.program, e.programs.clone());
            for code in codes {
                let Some(ty) = read(code, 4).map(|b| (u32::from_le_bytes(b[..4].try_into().unwrap()) >> 10 & 0xf) as usize) else { continue };
                let Some(g) = [None, Some(0), Some(1), Some(2), Some(3), Some(4)].get(ty).copied().flatten() else { continue };
                let seen = cbs[g][1];
                let model = m.stage_constants.get(&code).map(|&a| (a, CONSTANTS_SIZE));
                m.stats.constants_checked += 1;
                if model != seen {
                    m.stats.constants_differ += 1;
                    m.log("program constants", format!("program {program:#x} group {g}: model {model:x?}, command processor {seen:x?}"));
                }
            }
        }
        if super::encode::ON.load(Ordering::Relaxed) {
            let prediction = m.pending_draw(&d, draw.instance).map(|e| super::encode::predict(e, &m.learned, regs, read));
            super::encode::check(prediction, regs);
        }
        m.decoded(Work::Draw(Box::new(d)), draw.instance)
    });
}

pub fn cp_dispatch(grid: [u32; 3]) {
    with(|m| m.decoded(Work::Dispatch(grid), 0));
}

pub fn cp_report(address: u64, payload: u32, control: u32) {
    with(|m| {
        if let Some((counter, at)) = m.reports_pending.pop_front() {
            m.report_rules.entry(counter).or_insert((address.wrapping_sub(at), payload, control));
        }
    });
}

pub fn report() {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    with(|m| {
        let s = &m.stats;
        eprintln!(
            "kirikae:         nvn shadow: {} matched ({} with other programs, {} other targets, {} other constant buffers, {} other vertex buffers, {} other textures), {} differ, {} only decoded, {} only in the model, {} lists not seen recorded; {} waiting to be decoded, {} decoded and waiting",
            s.matched,
            s.programs,
            s.targets,
            s.uniforms,
            s.vertex,
            s.textures,
            s.shape,
            s.extra,
            s.missing,
            s.unknown_lists,
            m.expected.len(),
            m.decoded.len()
        );
        eprintln!(
            "kirikae:         nvn shadow state: {} fields checked, {} gave another register value than before, {} binds of objects no setter filled; NVN value -> register values:",
            m.learned.checked,
            m.learned.conflicts,
            m.objects.unknown
        );
        let s = &m.stats;
        eprintln!(
            "kirikae:         nvn shadow clears: {} matched, {} other values, {} other targets ({} of other layers not compared); texture addresses by handle: {} checked, {} differ",
            s.clears,
            s.clear_values,
            s.clear_targets,
            s.clear_layers,
            s.texture_addresses_checked,
            s.texture_addresses
        );
        eprintln!("kirikae:         nvn shadow resources: {:?}, {} texture pools", m.res.checks, m.res.texture_pools.len());
        eprintln!(
            "kirikae:         nvn model commands: {} uniform updates ({} bytes, last at {:#x}), barrier bits seen {:#x}, {} counter reports, {} storage buffers bound, multisample state {:#x}, shader scratch {:x?}",
            m.stats.uniform_updates,
            m.stats.uniform_bytes,
            m.stats.uniform_last,
            m.stats.barriers,
            m.stats.reports.len(),
            m.tracker.storage.len(),
            m.stats.multisample,
            m.tracker.scratch
        );
        m.stats.reports.clear();
        let seg = SEGMENTS.lock().unwrap();
        eprintln!(
            "kirikae:         nvn command memory: {} pushbuffer entries ({} words, {} from lists the host fully models) inside submitted lists, {} entries ({} words) elsewhere; {} submitted lists pending; {} segments of modeled lists checked, {} with other draw or clear counts",
            seg.entries_in_lists,
            seg.words_in_lists,
            seg.words_supported,
            seg.entries_elsewhere,
            seg.words_elsewhere,
            seg.submitted.len(),
            m.stats.segments,
            m.stats.segments_off
        );
        let mut missing: Vec<_> = m.stats.words_missing.iter().collect();
        missing.sort_by_key(|w| std::cmp::Reverse(*w.1));
        let missing: Vec<String> = missing.iter().take(12).map(|(k, n)| format!("{k} {n}")).collect();
        eprintln!(
            "kirikae:         nvn program constants: {} checked, {} differ; draw topology: {} checked, {} differ; driver uploads: {} words seen, not predicted: {}; predicted but not seen: {:?}",
            m.stats.constants_checked,
            m.stats.constants_differ,
            m.stats.topology_checked,
            m.stats.topology_differ,
            m.stats.words_checked,
            missing.join(", "),
            m.stats.words_extra
        );
        let mut calls: Vec<_> = m.unsupported_calls.iter().collect();
        calls.sort_by_key(|c| std::cmp::Reverse(*c.1));
        let calls: Vec<String> = calls.iter().take(20).map(|(n, k)| format!("{} {k}", n.trim_start_matches("nvnCommandBuffer"))).collect();
        eprintln!("kirikae:         nvn commands the model does not follow yet: {}", calls.join(", "));
        eprint!("{}", m.learned.report());
        eprint!("{}", m.learned.odd_examples());
    });
}

pub fn exec_steps(plan: &Plan, regs: &[u32], cbs: &Bindings, read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> Option<Vec<Step>> {
    with(|m| {
        let mut bound = *cbs;
        let cmds: Vec<&Cmd> = segment(&plan.list, plan.va, plan.end).iter().map(|(_, c)| c).collect();
        let (slot0, slot2) = m.support;
        let mut why = match () {
            _ if slot0.is_none() => Some("driver constants unknown".to_string()),
            _ if cmds.iter().any(|c| matches!(c, Cmd::Dispatch(_))) => Some("dispatch".to_string()),
            _ => None,
        };
        let cmds_reports: VecDeque<(u64, u64)> = cmds.iter().filter_map(|c| if let Cmd::Report { counter, address } = c { Some((*counter, *address)) } else { None }).collect();
        let mut steps = Vec::new();
        let mut works = Vec::new();
        let mut last = None;
        for cmd in cmds {
            m.exec.apply(cmd);
            match cmd {
                Cmd::Draw(d) if why.is_none() => works.push(Work::Draw(Box::new(d.clone()))),
                Cmd::Draw(d) => works.push(Work::Draw(Box::new(m.exec.draw(d)))),
                Cmd::Clear(cl) => works.push(Work::Clear(m.exec.clear(cl))),
                _ => {}
            }
            if why.is_some() {
                continue;
            }
            let base = slot0.unwrap_or(0);
            match cmd {
                Cmd::Draw(d) => {
                    let (key, regs_now, bindings) = match m.cached_prediction(d, regs, read) {
                        Ok(p) => p,
                        Err(e) => {
                            why = Some(e);
                            if let Some(w) = works.last_mut() {
                                *w = Work::Draw(Box::new(m.exec.draw(d)));
                            }
                            continue;
                        }
                    };
                    if last != Some(key) {
                        steps.push(Step::Regs(regs_now));
                        last = Some(key);
                    }
                    if d.indexed {
                        use crate::gpu::threed::reg;
                        let vertex = d.base_vertex as u32;
                        steps.push(Step::Regs(Arc::new(vec![
                            (reg::INDEX_BUFFER, 0xff, (d.start >> 32) as u32 & 0xff),
                            (reg::INDEX_BUFFER + 1, u32::MAX, d.start as u32),
                            (reg::INDEX_BUFFER + 4, 3, d.index_type as u32 & 3),
                            (reg::VB_ELEMENT_BASE, u32::MAX, vertex),
                            (0x446, u32::MAX, vertex),
                        ])));
                    }
                    for &(g, slot, b) in bindings.iter() {
                        bind(&mut steps, &mut bound, g, slot, Some(b));
                    }
                    for (&(g, slot), &(address, size)) in &m.exec.uniforms {
                        bind(&mut steps, &mut bound, g as usize, slot as usize, Some((address, size as u32)));
                    }
                    let based = d.base_vertex != 0 || d.base_instance != 0;
                    if based {
                        steps.push(Step::Write { at: base + BASE_VERTEX, words: vec![d.base_vertex as u32, d.base_instance] });
                    }
                    let index_size = d.indexed.then(|| 1 << (d.index_type & 3));
                    let first = if d.indexed { 0 } else { d.start as u32 };
                    for instance in 0..d.instances.max(1) {
                        steps.push(Step::Draw(crate::gpu::threed::Draw { topology: d.mode, first, count: d.count, index_size, instance }));
                    }
                    if based {
                        steps.push(Step::Write { at: base + BASE_VERTEX, words: vec![0, 0] });
                    }
                    if d.indexed && d.base_vertex != 0 {
                        steps.push(Step::Regs(Arc::new(vec![(crate::gpu::threed::reg::VB_ELEMENT_BASE, u32::MAX, 0), (0x446, u32::MAX, 0)])));
                    }
                }
                Cmd::Clear(cl) => {
                    let full = m.exec.draw(&Draw::default());
                    let mut regs_now = match super::encode::predict_complete(&full, &m.learned, regs, read) {
                        Ok(r) => r,
                        Err(e) => {
                            why = Some(format!("clear: {e}"));
                            continue;
                        }
                    };
                    if cl.what & 0x3c != 0 {
                        regs_now.extend((0..4).map(|i| (crate::gpu::threed::reg::CLEAR_COLOR + i, u32::MAX, cl.values[i])));
                    } else {
                        regs_now.push((0x364, u32::MAX, cl.values[0]));
                        regs_now.push((0x368, 0xff, cl.values[1]));
                    }
                    steps.push(Step::Regs(Arc::new(regs_now)));
                    last = None;
                    for layer in 0..m.exec.layers(cl) {
                        steps.push(Step::Clear(cl.what | layer << 10));
                    }
                }
                Cmd::Texture { stage: COMPUTE_STAGE, .. } => {}
                Cmd::Texture { stage, index, handle } => match stage_group(*stage).and_then(|g| slot2[g as usize]) {
                    Some(s2) => steps.push(Step::Write { at: s2 + TEXTURE_HANDLES + 8 * *index as u64, words: vec![*handle as u32] }),
                    None => why = Some(format!("texture handles unknown for stage {stage}")),
                },
                Cmd::UniformUpdate { address, data } => {
                    steps.push(Step::Write { at: *address, words: data.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect() });
                }
                Cmd::Storage { stage: COMPUTE_STAGE, .. } => why = Some("compute storage buffer".to_string()),
                Cmd::Storage { index, address, size, .. } => {
                    steps.push(Step::Write { at: base + STORAGE_BUFFERS + 0x10 * *index as u64, words: vec![*address as u32, (*address >> 32) as u32, *size as u32] });
                }
                Cmd::Multisample(_) => steps.push(Step::Write { at: base + 0x620, words: vec![0; 8] }),
                Cmd::Report { counter, address } => match m.report_rules.get(counter) {
                    Some(&(offset, payload, control)) => steps.push(Step::Report { at: address.wrapping_add(offset), payload, control }),
                    None => why = Some(format!("report counter {counter:#x} not learned")),
                },
                _ => {}
            }
        }
        if why.is_none() {
            match m.cached_prediction(&Draw::default(), regs, read) {
                Ok((key, regs_now, bindings)) => {
                    if last != Some(key) {
                        steps.push(Step::Regs(regs_now));
                    }
                    for &(g, slot, b) in bindings.iter() {
                        bind(&mut steps, &mut bound, g, slot, Some(b));
                    }
                    for (&(g, slot), &(address, size)) in &m.exec.uniforms {
                        bind(&mut steps, &mut bound, g as usize, slot as usize, Some((address, size as u32)));
                    }
                }
                Err(e) => why = Some(format!("segment end: {e}")),
            }
        }
        if why.is_none() {
            m.stats.executed += works.len() as u64;
        } else {
            if super::exec::ON.load(Ordering::Relaxed) && why.as_deref().is_some_and(|w| w.contains(": ")) {
                m.expected.extend(works);
            }
            m.reports_pending = cmds_reports;
        }
        super::exec::counted(why.as_deref());
        why.is_none().then_some(steps)
    })
}

pub fn executed_report() -> String {
    with(|m| format!(" ({} draws and clears)", m.stats.executed))
}
