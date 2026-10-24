use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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
    pub surfaces: Vec<(u8, resources::Surface)>,
    pub all_targets: [u64; 9],
    pub pools: [u64; 2],
}

impl Draw {
    fn shape(&self) -> (bool, u32, u64, i32, u32) {
        (self.indexed, self.count, self.start, self.base_vertex, self.instances)
    }
}

#[derive(Clone, Debug)]
enum Cmd {
    Program(Arc<Vec<u64>>),
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
    program: Arc<Vec<u64>>,
    targets: [u64; 9],
    surfaces: [Option<resources::Surface>; 9],
    uniforms: std::collections::BTreeMap<(u8, u8), (u64, u64)>,
    vertex: std::collections::BTreeMap<u8, u64>,
    vertex_sizes: std::collections::BTreeMap<u8, u64>,
    textures: std::collections::BTreeMap<(u8, u16), u64>,
    state: std::collections::BTreeMap<Field, u64>,
    objects: state::Objects,
    learned: state::Learned,
    res: Resources,
    expected: VecDeque<Work>,
    decoded: VecDeque<Work>,
    held: Option<Draw>,
    command_memory: HashMap<u64, u64>,
    missing: HashMap<u64, Vec<&'static str>>,
    pool_vas: HashMap<u64, u64>,
    pools: [u64; 2],
    storage: std::collections::BTreeMap<(u8, u8), (u64, u64)>,
    scratch: (u64, u64),
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

#[derive(Default)]
struct Segments {
    ranges: std::collections::BTreeMap<u64, (u64, u64, bool)>,
    words_supported: u64,
    words_in_lists: u64,
    words_elsewhere: u64,
    entries_in_lists: u64,
    entries_elsewhere: u64,
}

impl Segments {
    fn insert(&mut self, start: u64, end: u64, handle: u64, supported: bool) {
        let overlapping: Vec<u64> = self.ranges.range(..end).rev().take_while(|(_, r)| r.0 > start).map(|(s, _)| *s).collect();
        for s in overlapping {
            self.ranges.remove(&s);
        }
        self.ranges.insert(start, (end, handle, supported));
    }

    fn find(&self, va: u64, len: u64) -> Option<(u64, bool)> {
        self.ranges.range(..=va).next_back().filter(|(_, r)| va + len <= r.0).map(|(_, r)| (r.1, r.2))
    }
}

static SEGMENTS: Mutex<Segments> = Mutex::new(Segments { ranges: std::collections::BTreeMap::new(), words_supported: 0, words_in_lists: 0, words_elsewhere: 0, entries_in_lists: 0, entries_elsewhere: 0 });

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
    va: u64,
    draws: u64,
    clears: u64,
}

pub fn cp_entry(va: u64, words: u64) -> Option<Plan> {
    let found = {
        let mut s = SEGMENTS.lock().unwrap();
        let found = s.find(va, words * 4);
        match found {
            Some((_, supported)) => {
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
    let (handle, supported) = found?;
    if !supported {
        return None;
    }
    with(|m| {
        let list = m.lists.get(&handle)?.clone();
        let (mut draws, mut clears) = (0, 0);
        for (at, cmd) in list.iter() {
            if *at < va || *at >= va + words * 4 {
                continue;
            }
            match cmd {
                Cmd::Draw(d) => draws += d.instances as u64,
                Cmd::Clear(_) => clears += 1,
                _ => {}
            }
        }
        Some(Plan { handle, va, draws, clears })
    })
}

pub fn cp_entry_done(plan: Plan, draws: u64, clears: u64) {
    with(|m| {
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

const MAX_UNIFORM_UPDATE: u64 = 0x10000;

const PROGRAM_HEADER: u64 = 0x30;

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
                if m.missing.remove(&a[0]).is_some_and(|v| !v.is_empty()) || calls_unsupported {
                    m.unsupported.insert(handle);
                } else {
                    m.unsupported.remove(&handle);
                }
                m.lists.insert(handle, Arc::new(list));
                if let (Some(&va), Some(used)) = (m.command_memory.get(&a[0]), used)
                    && used > 0
                {
                    let supported = !m.unsupported.contains(&handle);
                    SEGMENTS.lock().unwrap().insert(va, va + used, handle, supported);
                }
            });
            c.pc = c.x[30];
            return;
        }
        Hook::CallCommands => record(Cmd::Call(read_u64s(c, a[2], a[1]))),
        Hook::QueueSubmitCommands => {
            let handles = read_u64s(c, a[2], a[1]);
            with(|m| {
                for h in handles {
                    m.expand(h, 0);
                }
                m.compare();
            });
        }
        Hook::ProgramSetShaders => {
            let stages = read_u64s(c, a[2], 2 * a[1]);
            let code: Vec<u64> = stages.chunks(2).map(|s| s[0] + PROGRAM_HEADER).collect();
            with(|m| {
                m.programs.insert(a[0], Arc::new(code));
            });
        }
        Hook::BindProgram => {
            let p = with(|m| m.programs.get(&a[1]).cloned().unwrap_or_default());
            record(Cmd::Program(p));
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
            let (base_vertex, instances) = match hook {
                Hook::DrawElements => (0, 1),
                Hook::DrawElementsBaseVertex => (a[5] as i32, 1),
                _ => (a[5] as i32, a[7] as u32),
            };
            let _ = index_size(a[2]);
            record(Cmd::Draw(Draw { indexed: true, count: a[3] as u32, start: a[4], base_vertex, instances, index_type: a[2] as u8, ..Default::default() }));
        }
        Hook::DrawArrays | Hook::DrawArraysInstanced => {
            let instances = if matches!(hook, Hook::DrawArrays) { 1 } else { a[5] as u32 };
            record(Cmd::Draw(Draw { indexed: false, count: a[3] as u32, start: a[2], instances, ..Default::default() }));
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
    fn expand(&mut self, handle: u64, depth: u32) {
        let Some(list) = self.lists.get(&handle).cloned() else {
            self.stats.unknown_lists += 1;
            return;
        };
        for (_, cmd) in list.iter() {
            match cmd {
                Cmd::Program(p) => self.program = p.clone(),
                Cmd::State(fields) => self.state.extend(fields.iter().copied()),
                Cmd::Uniform { stage, index, range } => {
                    if let Some(g) = stage_group(*stage) {
                        self.uniforms.insert((g, index + 3), *range);
                    }
                }
                Cmd::Vertex { index, address, size } => {
                    self.vertex.insert(*index, *address);
                    self.vertex_sizes.insert(*index, *size);
                }
                Cmd::Texture { stage, index, handle } => {
                    if let Some(g) = stage_group(*stage) {
                        self.textures.insert((g, *index), *handle);
                    }
                }
                Cmd::Targets(t, s) => (self.targets, self.surfaces) = (*t, *s),
                Cmd::Clear(cl) => {
                    let index = (cl.what >> 6 & 7) as usize;
                    let target = if cl.what & 0x3c != 0 { self.targets[index] } else { self.targets[8] };
                    self.expected.push_back(Work::Clear(Clear { target, ..cl.clone() }));
                }
                Cmd::Draw(d) => {
                    let d = Draw {
                        programs: self.program.to_vec(),
                        targets: [self.targets[0], self.targets[8]],
                        uniforms: self.uniforms.iter().map(|(k, v)| (*k, *v)).collect(),
                        vertex: self.vertex.iter().map(|(k, v)| (*k, *v)).collect(),
                        vertex_sizes: self.vertex_sizes.iter().map(|(k, v)| (*k, *v)).collect(),
                        textures: self.textures.iter().map(|(k, v)| (*k, *v)).collect(),
                        state: self.target_state(),
                        surfaces: self.surfaces.iter().enumerate().filter_map(|(k, s)| s.map(|s| (k as u8, s))).collect(),
                        all_targets: self.targets,
                        pools: self.pools,
                        ..d.clone()
                    };
                    self.expected.push_back(Work::Draw(Box::new(d)));
                }
                Cmd::Dispatch(g) => self.expected.push_back(Work::Dispatch(*g)),
                Cmd::Call(handles) if depth < 8 => {
                    for &h in handles {
                        self.expand(h, depth + 1);
                    }
                }
                Cmd::Call(_) => {}
                Cmd::Pool { texture, address } => self.pools[usize::from(!*texture)] = *address,
                Cmd::UniformUpdate { address, data } => {
                    self.stats.uniform_updates += 1;
                    self.stats.uniform_bytes += data.len() as u64;
                    self.stats.uniform_last = *address;
                }
                Cmd::Storage { stage, index, address, size } => {
                    if let Some(g) = stage_group(*stage) {
                        self.storage.insert((g, *index), (*address, *size));
                    }
                }
                Cmd::Barrier(bits) => self.stats.barriers |= *bits,
                Cmd::Report { counter, address } => self.stats.reports.push((*counter, *address)),
                Cmd::Multisample(state) => self.stats.multisample = *state,
                Cmd::Scratch { address, size } => self.scratch = (*address, *size),
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

    fn target_state(&self) -> Vec<(Field, u64)> {
        let mut state: Vec<(Field, u64)> = self.state.iter().map(|(k, v)| (*k, *v)).collect();
        for (k, s) in self.surfaces.iter().enumerate() {
            if let Some(s) = s {
                state.push(if k < 8 { (Field(state::Kind::TargetFormat, k as u8), s.format as u64) } else { (Field(state::Kind::DepthFormat, 0), s.format as u64) });
            }
        }
        state
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
                if let (Work::Draw(e), Work::Draw(d)) = (&e, &d) {
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

fn stage_group(stage: u8) -> Option<u8> {
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
            m.storage.len(),
            m.stats.multisample,
            m.scratch
        );
        m.stats.reports.clear();
        let seg = SEGMENTS.lock().unwrap();
        eprintln!(
            "kirikae:         nvn command memory: {} pushbuffer entries ({} words, {} from lists the host fully models) inside recorded lists, {} entries ({} words) elsewhere; {} list ranges known; {} segments of modeled lists checked, {} with other draw or clear counts",
            seg.entries_in_lists,
            seg.words_in_lists,
            seg.words_supported,
            seg.entries_elsewhere,
            seg.words_elsewhere,
            seg.ranges.len(),
            m.stats.segments,
            m.stats.segments_off
        );
        let mut calls: Vec<_> = m.unsupported_calls.iter().collect();
        calls.sort_by_key(|c| std::cmp::Reverse(*c.1));
        let calls: Vec<String> = calls.iter().take(20).map(|(n, k)| format!("{} {k}", n.trim_start_matches("nvnCommandBuffer"))).collect();
        eprintln!("kirikae:         nvn commands the model does not follow yet: {}", calls.join(", "));
        eprint!("{}", m.learned.report());
        eprint!("{}", m.learned.odd_examples());
    });
}
