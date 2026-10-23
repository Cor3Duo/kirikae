use std::collections::VecDeque;
use std::ffi::CStr;
use std::sync::Arc;

use ash::vk;
use kirikae_maxwell::{Kind, Options, Shader, Stage, TexDim};

use super::buffer_cache::{self, BufferCache};
use super::formats::{self, ColorFormat, TexFormat};
use super::fxhash::{FastMap as HashMap, FastSet};
use super::threed::{CB_SLOTS, Draw, address, reg};
use super::tic::{self, Layout, TextureHeader};
use super::{Gpu, block_linear_offset};
use crate::mem::GuestMemory;

const RING_SIZE: u64 = 192 << 20;
const IN_FLIGHT: usize = 3;
const BATCH_RING_LIMIT: u64 = RING_SIZE / 4;
const PIPELINE_CACHE_SAVE: std::time::Duration = std::time::Duration::from_secs(30);
const MAX_CBUF: u64 = 0x1_0000;
const MAX_STORAGE: u64 = 64 << 20;

const VERTEX_BASE: u32 = 0;
const FRAGMENT_BASE: u32 = 128;
const STORAGE_OFFSET: u32 = 18;
const TEXTURE_OFFSET: u32 = 40;
const IMAGE_OFFSET: u32 = 104;

pub struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    physical: vk::PhysicalDevice,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    limits: vk::PhysicalDeviceLimits,
    push: Option<ash::khr::push_descriptor::Device>,
    pool: vk::CommandPool,
    batches: Vec<Batch>,
    current: usize,
    in_flight: VecDeque<usize>,
    cmd: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    recording: bool,
    serial: u64,
    timestamps: Option<(vk::QueryPool, f64)>,
    rendering: Option<RenderTargets>,
    ring: Ring,
    pending: Vec<Pending>,
    pending_pages: PendingPages,
    draw_wrote_storage: bool,
    pub pending_flag: Arc<std::sync::atomic::AtomicBool>,
    garbage: Vec<Garbage>,
    submit_wait: Option<vk::Semaphore>,
    submit_signal: Option<vk::Semaphore>,
    surface_fn: Option<ash::khr::surface::Instance>,
    #[cfg(windows)]
    win32_fn: Option<ash::khr::win32_surface::Instance>,
    swapchain_fn: Option<ash::khr::swapchain::Device>,
    swapchain: Option<Swapchain>,
    family: u32,
    present_failed: bool,
    buffers: BufferCache,
    buffer_cache_on: bool,
    images: HashMap<ImageKey, Image>,
    by_cpu: std::collections::BTreeMap<u64, Vec<ImageKey>>,
    samplers: HashMap<tic::Sampler, vk::Sampler>,
    shaders: HashMap<ShaderKey, Arc<ShaderModule>>,
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    layouts: HashMap<LayoutKey, (vk::DescriptorSetLayout, vk::PipelineLayout)>,
    pair_layouts: HashMap<(vk::ShaderModule, vk::ShaderModule), (vk::DescriptorSetLayout, vk::PipelineLayout)>,
    pair_drawn: HashMap<(vk::ShaderModule, vk::ShaderModule), u64>,
    scratch_key: PipelineKey,
    scratch: DrawScratch,
    free_memory: HashMap<(u32, u64), Vec<vk::DeviceMemory>>,
    pooled_memory: HashMap<vk::DeviceMemory, (u32, u64)>,
    bound: Bound,
    compute_pipelines: HashMap<(vk::ShaderModule, vk::PipelineLayout), vk::Pipeline>,
    readback: Option<(vk::Buffer, vk::DeviceMemory, u64, *mut u8)>,
    readback_coherent: bool,
    anisotropy: bool,
    shader_writes: HashMap<u64, bool>,
    skip_noop: bool,
    depth_clamp: bool,
    depth_bias_clamp: bool,
    timing: bool,
    created: std::time::Instant,
    index_checked: Option<(u64, u64, buffer_cache::Kind)>,
    heads: HashMap<u64, (u64, u64)>,
    storage: HashMap<(u64, u64), u64>,
    checked: HashMap<(u64, u64, buffer_cache::Kind), (usize, vk::Buffer, u64)>,
    cp_writes: CpWrites,
    write_back: Vec<(u64, u64, u64)>,
    pitch_targets: HashMap<ImageKey, (u64, u32, u32)>,
    clock: u64,
    crops: HashMap<ImageKey, (ImageKey, u64)>,
    draw_targets: Vec<ImageKey>,
    feedbacks: HashMap<ImageKey, u64>,
    feedback_used: Vec<ImageKey>,
    twins: HashMap<ImageKey, (ImageKey, u64)>,
    gathers: HashMap<ImageKey, Vec<u64>>,
    gather_sources: HashMap<ImageKey, (u64, Vec<Option<ImageKey>>)>,
    tex_cache: HashMap<TexKey, TexEntry>,
    tex_at: std::collections::BTreeMap<u64, Vec<TexKey>>,
    newest: HashMap<u64, ImageKey>,
    target_memo: [((u64, vk::Format, u32, u32, u64), ImageKey); 4],
    generation: u64,
    volumes: HashMap<(u64, vk::Format), Volume>,
    retired: Vec<ImageKey>,
    pipeline_cache: vk::PipelineCache,
    compiler: Option<PipelineCompiler>,
    cache_file: Option<std::path::PathBuf>,
    cache_saved: (u64, std::time::Instant),
    cache_saver: Option<std::thread::JoinHandle<()>>,
    guest: *const GuestMemory,
    pub name: String,
    pub stats: RenderStats,
    trace_state: String,
}

pub const DRAW_PARTS: [&str; 19] = [
    "resolve", "targets", "shaders", "layout", "pipeline", "descriptors", "vertices", "indices", "recording", "  constant buffers", "  storage buffers", "  textures",
    "    tex headers", "    tex layered", "    tex image", "    tex view",
    "      gather", "      crop", "      view+sampler",
];

#[derive(Default, Debug, Clone)]
pub struct RenderStats {
    pub draw_parts: [std::time::Duration; 19],
    pub crop_copies: u64,
    pub crop_calls: u64,
    pub draws: u64,
    pub skipped: u64,
    pub noop_draws: u64,
    pub pushes_reused: u64,
    pub dispatches: u64,
    pub skipped_dispatches: u64,
    pub skip_reasons: std::collections::BTreeMap<String, u64>,
    pub placeholders: std::collections::BTreeMap<String, u64>,
    pub uploads: u64,
    pub refreshes: u64,
    pub pipelines: u64,
    pub batches: u64,
    pub waits: u64,
    pub resolves: u64,
    pub patches: u64,
    pub crop_images: u64,
    pub written_back: u64,
    pub ring_bytes: u64,
    pub cbuf_bytes: u64,
    pub vertex_bytes: u64,
    pub cached_bytes: u64,
    pub cache_memory: (u64, u64),
    pub presents: u64,
    pub draw_time: std::time::Duration,
    pub flush_time: std::time::Duration,
    pub readback_time: std::time::Duration,
    pub shader_time: std::time::Duration,
    pub gpu_time: std::time::Duration,
}

type LayoutKey = (Vec<(u32, vk::DescriptorType)>, u32);

unsafe impl Send for Renderer {}

impl Drop for Renderer {
    fn drop(&mut self) {
        if let Some(mut c) = self.compiler.take() {
            for (key, r) in c.stop() {
                if let Ok(p) = r {
                    self.pipelines.insert(key, p);
                }
            }
        }
        self.save_pipeline_cache(true);
        unsafe {
            let d = &self.device;
            let _ = d.device_wait_idle();
            d.destroy_pipeline_cache(self.pipeline_cache, None);
            for p in self.pipelines.values().chain(self.compute_pipelines.values()) {
                d.destroy_pipeline(*p, None);
            }
            for (set, layout) in self.layouts.values() {
                d.destroy_pipeline_layout(*layout, None);
                d.destroy_descriptor_set_layout(*set, None);
            }
            for s in self.shaders.values() {
                d.destroy_shader_module(s.module, None);
            }
            for s in self.samplers.values() {
                d.destroy_sampler(*s, None);
            }
            for img in self.images.values() {
                for v in img.views.values() {
                    d.destroy_image_view(*v, None);
                }
                d.destroy_image(img.image, None);
                d.free_memory(img.memory, None);
            }
            for v in self.volumes.values() {
                for view in v.views.values() {
                    d.destroy_image_view(*view, None);
                }
                d.destroy_image(v.image, None);
                d.free_memory(v.memory, None);
            }
            if let Some((buffer, memory, ..)) = self.readback.take() {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
            }
            d.destroy_buffer(self.ring.buffer, None);
            d.free_memory(self.ring.memory, None);
            let garbage = self.batches.iter_mut().flat_map(|b| std::mem::take(&mut b.garbage)).chain(std::mem::take(&mut self.garbage));
            for g in garbage.collect::<Vec<_>>() {
                if let Garbage::Image { image, memory, views } = g {
                    for v in views {
                        d.destroy_image_view(v, None);
                    }
                    d.destroy_image(image, None);
                    d.free_memory(memory, None);
                }
            }
            for m in self.free_memory.drain().flat_map(|(_, v)| v) {
                d.free_memory(m, None);
            }
            for st in self.batches.iter_mut().filter_map(|b| b.staging.take()) {
                d.destroy_buffer(st.buffer, None);
                d.free_memory(st.memory, None);
            }
            self.buffers.destroy(d);
            if let Some((q, _)) = self.timestamps {
                d.destroy_query_pool(q, None);
            }
            if let Some(sc) = self.swapchain.take() {
                for s in sc.done {
                    d.destroy_semaphore(s, None);
                }
                if let Some(f) = &self.swapchain_fn {
                    f.destroy_swapchain(sc.handle, None);
                }
                if let Some(f) = &self.surface_fn {
                    f.destroy_surface(sc.surface, None);
                }
            }
            for b in &self.batches {
                d.destroy_semaphore(b.acquired, None);
                d.destroy_descriptor_pool(b.descriptor_pool, None);
                d.destroy_fence(b.fence, None);
            }
            d.destroy_command_pool(self.pool, None);
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Batch {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
    descriptor_pool: vk::DescriptorPool,
    serial: u64,
    ring_end: u64,
    write_back: Vec<(u64, u64, u64)>,
    garbage: Vec<Garbage>,
    staging: Option<Staging>,
    staged: bool,
}

struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
    coherent: bool,
}

enum Garbage {
    Image { image: vk::Image, memory: vk::DeviceMemory, views: Vec<vk::ImageView> },
    Block(buffer_cache::Block),
}

struct Ring {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    head: u64,
    tail: Option<u64>,
    recorded: u64,
}

impl Ring {
    fn fit(&mut self, size: u64, align: u64) -> Option<u64> {
        let mut offset = self.head.next_multiple_of(align);
        let wrapped = offset + size > RING_SIZE;
        if wrapped {
            offset = 0;
        }
        let free = match self.tail {
            None => true,
            Some(t) if t <= self.head => !wrapped || offset + size < t,
            Some(t) => !wrapped && offset + size < t,
        };
        if !free {
            return None;
        }
        self.tail.get_or_insert(offset);
        self.head = offset + size;
        self.recorded += size;
        Some(offset)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey {
    pub cpu: u64,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

#[derive(Default)]
struct PendingPages {
    pages: HashMap<u64, Vec<(u64, Pending)>>,
    next: u64,
}

type Pending = (u64, u64, u64, u64);

fn pages_of(cpu: u64, len: u64) -> std::ops::RangeInclusive<u64> {
    (cpu >> 16)..=(cpu.saturating_add(len.max(1) - 1) >> 16)
}

impl PendingPages {
    fn add(&mut self, p: Pending) {
        let n = self.next;
        self.next += 1;
        for page in pages_of(p.0, p.1) {
            self.pages.entry(page).or_default().push((n, p));
        }
    }

    fn retire(&mut self, serial: u64) {
        self.pages.retain(|_, v| {
            v.retain(|e| e.1.2 != serial);
            !v.is_empty()
        });
    }

    fn near(&self, cpu: u64, len: u64) -> bool {
        !self.pages.is_empty() && pages_of(cpu, len).any(|p| self.pages.contains_key(&p))
    }

    fn overlapping(&self, cpu: u64, len: u64) -> impl Iterator<Item = &(u64, Pending)> {
        pages_of(cpu, len).filter_map(|p| self.pages.get(&p)).flatten().filter(move |(_, p)| p.0 < cpu + len && cpu < p.0 + p.1)
    }
}

#[derive(Default)]
struct CpWrites {
    all: Vec<(u64, u64)>,
    by_page: HashMap<u64, Vec<u32>>,
    wide: Vec<u32>,
}

const WIDE_WRITE: u64 = 64;

impl CpWrites {
    fn clear(&mut self) {
        self.all.clear();
        self.by_page.clear();
        self.wide.clear();
    }

    fn len(&self) -> usize {
        self.all.len()
    }

    fn push(&mut self, cpu: u64, len: u64) {
        let i = self.all.len() as u32;
        self.all.push((cpu, cpu + len));
        let pages = pages_of(cpu, len);
        if pages.end() - pages.start() >= WIDE_WRITE {
            self.wide.push(i);
        } else {
            for p in pages {
                self.by_page.entry(p).or_default().push(i);
            }
        }
    }

    fn since(&self, c: u64, size: u64, seen: usize) -> bool {
        let hits = |i: &u32| {
            let (s, e) = self.all[*i as usize];
            s < c + size && c < e
        };
        let pages = pages_of(c, size);
        if pages.end() - pages.start() >= (self.all.len() - seen) as u64 {
            return self.all[seen..].iter().any(|&(s, e)| s < c + size && c < e);
        }
        self.wide.iter().rev().take_while(|&&i| i as usize >= seen).any(hits)
            || pages.into_iter().any(|p| self.by_page.get(&p).is_some_and(|v| v.iter().rev().take_while(|&&i| i as usize >= seen).any(hits)))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TexKey {
    tic: [u32; 8],
    tsc: [u32; 8],
    dim: TexDim,
    array: bool,
    shadow: bool,
}

#[derive(Clone, Copy)]
struct TexEntry {
    image: ImageKey,
    view: vk::ImageView,
    sampler: vk::Sampler,
    refresh: Option<ImageKey>,
    gather: Option<vk::Format>,
    crop: Option<ImageKey>,
    twin: Option<(ImageKey, vk::Format)>,
}

struct Volume {
    image: vk::Image,
    memory: vk::DeviceMemory,
    views: HashMap<[vk::ComponentSwizzle; 4], vk::ImageView>,
    hash: u64,
    checked: u64,
    slices: Vec<u64>,
}

pub struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    views: HashMap<ViewKey, vk::ImageView>,
    levels: u32,
    layers: u32,
    pub gpu_dirty: bool,
    stamp: u64,
    source: Option<Source>,
    drawn: Option<u64>,
    cleared: Option<u64>,
    clear_streak: u32,
}

impl Image {
    fn note_clear(&mut self, frame: u64) {
        if self.cleared == Some(frame) {
            return;
        }
        self.clear_streak = if self.cleared == frame.checked_sub(1) { self.clear_streak + 1 } else { 1 };
        self.cleared = Some(frame);
    }

    fn transient(&self, frame: u64) -> bool {
        self.cleared == Some(frame) && self.clear_streak >= 3
    }
}

#[derive(Clone, Copy)]
struct Source {
    hash: u64,
    checked: u64,
    volatile: bool,
}

const SOURCE_RECHECK: u64 = 30;

type ViewKey = (vk::Format, [vk::ComponentSwizzle; 4], vk::ImageViewType, u32, u32, u32);

const CROP: u64 = 1 << 62;
const GATHER: u64 = 1 << 61;
const TWIN: u64 = 1 << 60;
const FEEDBACK: u64 = 1 << 59;
const DERIVED: u64 = CROP | GATHER | TWIN | FEEDBACK;
const CROP_SIZES: usize = 3;

#[derive(Clone, PartialEq, Eq)]
struct RenderTargets {
    views: Vec<(vk::ImageView, vk::Format)>,
    depth: Option<(vk::ImageView, vk::Format)>,
    extent: (u32, u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ShaderKey {
    va: u64,
    hash: u64,
    stage: u8,
    types: u64,
}

struct ShaderModule {
    module: vk::ShaderModule,
    shader: Shader,
    va: u64,
    vertex_index: bool,
}

fn reads_vertex_index(spirv: &[u32]) -> bool {
    let mut i = 5;
    while i < spirv.len() {
        let (count, op) = ((spirv[i] >> 16) as usize, spirv[i] & 0xffff);
        if op == 71 && spirv.get(i + 2) == Some(&11) && spirv.get(i + 3) == Some(&42) {
            return true;
        }
        i += count.max(1);
    }
    false
}

#[derive(Default)]
struct DrawScratch {
    targets: Vec<Option<(ImageKey, ColorFormat)>>,
    sizes: Vec<(u32, u32)>,
    streams: Vec<u32>,
    views: Vec<(vk::ImageView, vk::Format)>,
    descriptors: Vec<Descriptor>,
    vertex_buffers: Vec<(u32, vk::Buffer, u64)>,
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    image_infos: Vec<vk::DescriptorImageInfo>,
    writes: Vec<vk::WriteDescriptorSet<'static>>,
}

#[derive(Default)]
struct Bound {
    pipeline: Option<vk::Pipeline>,
    viewport: Option<[u32; 4]>,
    scissor: Option<(i32, i32, u32, u32)>,
    cull: Option<vk::CullModeFlags>,
    front: Option<vk::FrontFace>,
    blend: Option<[u32; 4]>,
    depth: Option<(bool, bool, vk::CompareOp)>,
    stencil_test: Option<bool>,
    stencil: Option<[[u32; 7]; 2]>,
    depth_bias: Option<(bool, [u32; 3])>,
    vertex: [Option<(vk::Buffer, u64)>; 32],
    index: Option<(vk::Buffer, u64, vk::IndexType)>,
    descriptors: Option<(vk::PipelineLayout, Vec<Descriptor>)>,
}

fn changed<T: PartialEq>(slot: &mut Option<T>, v: T) -> bool {
    let new = slot.as_ref() != Some(&v);
    if new {
        *slot = Some(v);
    }
    new
}

#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct PipelineKey {
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    layout: vk::PipelineLayout,
    topology: vk::PrimitiveTopology,
    attributes: Vec<(u32, u32, vk::Format, u32)>,
    bindings: Vec<(u32, u32)>,
    colors: Vec<vk::Format>,
    depth: vk::Format,
    blend: Vec<[u32; 8]>,
    depth_clamp: bool,
    alpha_test: [u32; 2],
}

fn alpha_test_func(regs: &[u32]) -> u32 {
    if regs[reg::ALPHA_TEST_ENABLE] & 1 == 0 {
        return 7;
    }
    let f = regs[reg::ALPHA_TEST_FUNC];
    (if f >= 0x200 { f - 0x200 } else { f.saturating_sub(1) }) & 7
}

fn build_pipeline(device: &ash::Device, cache: vk::PipelineCache, key: &PipelineKey) -> Result<vk::Pipeline, vk::Result> {
        let entries = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }, vk::SpecializationMapEntry { constant_id: 1, offset: 4, size: 4 }];
        let alpha_data: Vec<u8> = key.alpha_test.iter().flat_map(|w| w.to_ne_bytes()).collect();
        let alpha_test = vk::SpecializationInfo::default().map_entries(&entries).data(&alpha_data);
        let mut stages = vec![vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(key.vertex).name(c"main")];
        if key.fragment != vk::ShaderModule::null() {
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(key.fragment).name(c"main").specialization_info(&alpha_test));
        }
        let bindings: Vec<vk::VertexInputBindingDescription> =
            key.bindings.iter().map(|&(b, stride)| vk::VertexInputBindingDescription { binding: b, stride, input_rate: vk::VertexInputRate::VERTEX }).collect();
        let attributes: Vec<vk::VertexInputAttributeDescription> =
            key.attributes.iter().map(|&(location, binding, format, offset)| vk::VertexInputAttributeDescription { location, binding, format, offset }).collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&bindings).vertex_attribute_descriptions(&attributes);
        let restart = matches!(key.topology, vk::PrimitiveTopology::TRIANGLE_STRIP | vk::PrimitiveTopology::LINE_STRIP | vk::PrimitiveTopology::TRIANGLE_FAN);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(key.topology).primitive_restart_enable(restart && false);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).line_width(1.0).depth_clamp_enable(key.depth_clamp);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default().depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
        let attachments: Vec<vk::PipelineColorBlendAttachmentState> = key
            .blend
            .iter()
            .map(|b| {
                let mask = b[7];
                let write = vk::ColorComponentFlags::from_raw(
                    (mask & 1 != 0) as u32 | ((mask >> 4 & 1 != 0) as u32) << 1 | ((mask >> 8 & 1 != 0) as u32) << 2 | ((mask >> 12 & 1 != 0) as u32) << 3,
                );
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(b[0] == 1)
                    .color_blend_op(formats::blend_op(b[1]))
                    .src_color_blend_factor(formats::blend_factor(b[2]))
                    .dst_color_blend_factor(formats::blend_factor(b[3]))
                    .alpha_blend_op(formats::blend_op(b[4]))
                    .src_alpha_blend_factor(formats::blend_factor(b[5]))
                    .dst_alpha_blend_factor(formats::blend_factor(b[6]))
                    .color_write_mask(write)
            })
            .collect();
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::CULL_MODE,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::DEPTH_BIAS,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&key.colors).depth_attachment_format(key.depth);
        if aspect_of(key.depth).contains(vk::ImageAspectFlags::STENCIL) {
            rendering = rendering.stencil_attachment_format(key.depth);
        }
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(key.layout)
            .push_next(&mut rendering);
        unsafe { device.create_graphics_pipelines(cache, &[info], None) }.map(|p| p[0]).map_err(|(_, e)| e)
}

type Built = (PipelineKey, Result<vk::Pipeline, vk::Result>);

struct PipelineCompiler {
    queue: Option<std::sync::mpsc::Sender<PipelineKey>>,
    done: Arc<std::sync::Mutex<Vec<Built>>>,
    pending: FastSet<PipelineKey>,
    failed: HashMap<PipelineKey, String>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl PipelineCompiler {
    fn new(device: &ash::Device, cache: vk::PipelineCache) -> Option<PipelineCompiler> {
        if crate::env_var!("KIRIKAE_SYNC_PIPELINES").is_some() {
            return None;
        }
        let (tx, rx) = std::sync::mpsc::channel::<PipelineKey>();
        let rx = Arc::new(std::sync::Mutex::new(rx));
        let done = Arc::new(std::sync::Mutex::new(Vec::new()));
        let n = std::thread::available_parallelism().map_or(2, |n| n.get()).saturating_sub(2).clamp(1, 4);
        let workers: Vec<_> = (0..n)
            .filter_map(|i| {
                let (rx, done, device) = (rx.clone(), done.clone(), device.clone());
                std::thread::Builder::new()
                    .name(format!("kirikae-pipelines-{i}"))
                    .spawn(move || {
                        loop {
                            let next = rx.lock().unwrap().recv();
                            let Ok(key) = next else { return };
                            let r = build_pipeline(&device, cache, &key);
                            done.lock().unwrap().push((key, r));
                        }
                    })
                    .ok()
            })
            .collect();
        if workers.is_empty() {
            return None;
        }
        Some(PipelineCompiler { queue: Some(tx), done, pending: FastSet::default(), failed: HashMap::default(), workers })
    }

    fn queue(&mut self, key: &PipelineKey) {
        if self.pending.insert(key.clone())
            && let Some(q) = &self.queue
        {
            let _ = q.send(key.clone());
        }
    }

    fn finished(&mut self) -> Vec<Built> {
        let out: Vec<_> = self.done.lock().unwrap().drain(..).collect();
        for (k, _) in &out {
            self.pending.remove(k);
        }
        out
    }

    fn stop(&mut self) -> Vec<Built> {
        self.queue = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        self.finished()
    }
}

struct Swapchain {
    surface: vk::SurfaceKHR,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    srgb: bool,
    done: Vec<vk::Semaphore>,
    stale: bool,
    format: vk::Format,
    readable: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Presented {
    Shown,
    NoImage,
    NoSwapchain,
}

const PRESENT: u64 = 1 << 60;

#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub va: u64,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub block_height_log2: u32,
    pub pitch: Option<u32>,
}

pub fn color_surface(regs: &[u32], base: usize) -> Surface {
    let (format, memory) = (regs[base + 4], regs[base + 5]);
    let linear = memory >> 12 & 1 == 1;
    let bytes = formats::color_target(format).map_or(4, |f| f.bytes.max(1));
    Surface {
        va: address(regs, base),
        format,
        width: if linear { regs[base + 2] / bytes } else { regs[base + 2] },
        height: regs[base + 3],
        block_height_log2: (memory >> 4) & 7,
        pitch: linear.then_some(regs[base + 2]),
    }
}

fn find_memory(props: &vk::PhysicalDeviceMemoryProperties, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| bits >> i & 1 == 1 && props.memory_types[i as usize].property_flags.contains(flags))
}

fn hash64(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for c in data.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Renderer {
    pub fn new() -> Result<Renderer, String> {
        unsafe { Self::create().map_err(|e| format!("{e:?}")) }
    }

    unsafe fn create() -> Result<Renderer, Box<dyn std::error::Error>> {
        let entry = unsafe { ash::Entry::load()? };
        let app = vk::ApplicationInfo::default().application_name(c"kirikae").engine_name(c"kirikae").api_version(vk::API_VERSION_1_3);
        let mut layers = Vec::new();
        if crate::env_var!("KIRIKAE_VK_VALIDATION").is_some() {
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }
        let instance_available = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let instance_has = |n: &CStr| instance_available.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == n);
        #[cfg(windows)]
        let surfaces = !crate::display::headless() && instance_has(ash::khr::surface::NAME) && instance_has(ash::khr::win32_surface::NAME);
        #[cfg(not(windows))]
        let surfaces = { let _ = instance_has; false };
        let mut instance_extensions = Vec::new();
        if surfaces {
            instance_extensions.push(ash::khr::surface::NAME.as_ptr());
            #[cfg(windows)]
            instance_extensions.push(ash::khr::win32_surface::NAME.as_ptr());
        }
        let instance = unsafe {
            entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app).enabled_layer_names(&layers).enabled_extension_names(&instance_extensions), None)?
        };
        let devices = unsafe { instance.enumerate_physical_devices()? };
        let physical = devices
            .iter()
            .copied()
            .filter(|&d| unsafe { instance.get_physical_device_properties(d) }.api_version >= vk::API_VERSION_1_3)
            .max_by_key(|&d| match unsafe { instance.get_physical_device_properties(d) }.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                _ => 1,
            })
            .ok_or("no Vulkan 1.3 device")?;
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy().into_owned();
        let family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("no graphics queue")? as u32;
        let available = unsafe { instance.enumerate_device_extension_properties(physical)? };
        let has = |n: &CStr| available.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == n);
        let mut extensions = Vec::new();
        let push_supported = has(ash::khr::push_descriptor::NAME);
        if push_supported {
            extensions.push(ash::khr::push_descriptor::NAME.as_ptr());
        }
        let swapchains = surfaces && has(ash::khr::swapchain::NAME);
        if swapchains {
            extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }
        let supported = unsafe { instance.get_physical_device_features(physical) };
        let features = vk::PhysicalDeviceFeatures::default()
            .independent_blend(supported.independent_blend == vk::TRUE)
            .dual_src_blend(supported.dual_src_blend == vk::TRUE)
            .sampler_anisotropy(supported.sampler_anisotropy == vk::TRUE)
            .texture_compression_bc(supported.texture_compression_bc == vk::TRUE)
            .fill_mode_non_solid(supported.fill_mode_non_solid == vk::TRUE)
            .shader_clip_distance(supported.shader_clip_distance == vk::TRUE)
            .shader_image_gather_extended(supported.shader_image_gather_extended == vk::TRUE)
            .depth_clamp(supported.depth_clamp == vk::TRUE)
            .depth_bias_clamp(supported.depth_bias_clamp == vk::TRUE);
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&extensions).enabled_features(&features).push_next(&mut v13);
        let device = unsafe { instance.create_device(physical, &info, None)? };
        let queue = unsafe { device.get_device_queue(family, 0) };
        let push = push_supported.then(|| ash::khr::push_descriptor::Device::new(&instance, &device));
        let swapchain_fn = swapchains.then(|| ash::khr::swapchain::Device::new(&instance, &device));
        let surface_fn = surfaces.then(|| ash::khr::surface::Instance::new(&entry, &instance));
        #[cfg(windows)]
        let win32_fn = surfaces.then(|| ash::khr::win32_surface::Instance::new(&entry, &instance));
        let pool = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)? };
        let cmds = unsafe { device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(IN_FLIGHT as u32))? };
        let timestamps = if unsafe { instance.get_physical_device_queue_family_properties(physical) }[family as usize].timestamp_valid_bits > 0 {
            let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2 * IN_FLIGHT as u32);
            unsafe { device.create_query_pool(&info, None) }.ok().map(|q| (q, props.limits.timestamp_period as f64))
        } else {
            None
        };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let usage = vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = unsafe { device.create_buffer(&vk::BufferCreateInfo::default().size(RING_SIZE).usage(usage), None)? };
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let ty = find_memory(&mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).ok_or("no host memory")?;
        let memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
        let ptr = unsafe { device.map_memory(memory, 0, RING_SIZE, vk::MemoryMapFlags::empty())? } as *mut u8;

        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 65536 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 16384 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 65536 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 65536 },
        ];
        let mut batches = Vec::new();
        for cmd in cmds {
            batches.push(Batch {
                cmd,
                fence: unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? },
                acquired: unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? },
                descriptor_pool: unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(16384).pool_sizes(&sizes), None)? },
                serial: 0,
                ring_end: 0,
                write_back: Vec::new(),
                garbage: Vec::new(),
                staging: None,
                staged: false,
            });
        }

        let cache_file = pipeline_cache_file(&props);
        if let Some(f) = &cache_file {
            remove_stale_saves(f);
        }
        let initial = cache_file.as_ref().and_then(|f| std::fs::read(f).ok()).filter(|d| pipeline_cache_matches(d, &props)).unwrap_or_default();
        let pipeline_cache = unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(&initial), None) }
            .or_else(|_| unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) })?;
        let compiler = PipelineCompiler::new(&device, pipeline_cache);

        Ok(Renderer {
            _entry: entry,
            instance,
            device,
            queue,
            physical,
            mem_props,
            limits: props.limits,
            push,
            pool,
            cmd: batches[0].cmd,
            descriptor_pool: batches[0].descriptor_pool,
            batches,
            current: 0,
            in_flight: VecDeque::new(),
            recording: false,
            serial: 1,
            timestamps,
            rendering: None,
            ring: Ring { buffer, memory, ptr, head: 0, tail: None, recorded: 0 },
            pending: Vec::new(),
            pending_pages: PendingPages::default(),
            draw_wrote_storage: false,
            pending_flag: Arc::default(),
            garbage: Vec::new(),
            submit_wait: None,
            submit_signal: None,
            surface_fn,
            #[cfg(windows)]
            win32_fn,
            swapchain_fn,
            swapchain: None,
            family,
            present_failed: false,
            buffers: BufferCache::new(&mem_props, ty),
            buffer_cache_on: crate::env_var!("KIRIKAE_NO_BUFFER_CACHE").is_none(),
            images: HashMap::default(),
            by_cpu: std::collections::BTreeMap::new(),
            samplers: HashMap::default(),
            shaders: HashMap::default(),
            pipelines: HashMap::default(),
            layouts: HashMap::default(),
            pair_layouts: HashMap::default(),
            pair_drawn: HashMap::default(),
            scratch_key: PipelineKey::default(),
            scratch: DrawScratch::default(),
            free_memory: HashMap::default(),
            pooled_memory: HashMap::default(),
            bound: Bound::default(),
            compute_pipelines: HashMap::default(),
            readback: None,
            readback_coherent: true,
            anisotropy: supported.sampler_anisotropy == vk::TRUE,
            shader_writes: HashMap::default(),
            skip_noop: crate::env_var!("KIRIKAE_KEEP_NOOP_DRAWS").is_none(),
            depth_clamp: supported.depth_clamp == vk::TRUE,
            depth_bias_clamp: supported.depth_bias_clamp == vk::TRUE,
            timing: crate::env_var!("KIRIKAE_STATS").is_some_and(|v| v == "2"),
            created: std::time::Instant::now(),
            index_checked: None,
            heads: HashMap::default(),
            storage: HashMap::default(),
            checked: HashMap::default(),
            cp_writes: CpWrites::default(),
            write_back: Vec::new(),
            pitch_targets: HashMap::default(),
            clock: 0,
            crops: HashMap::default(),
            draw_targets: Vec::new(),
            feedbacks: HashMap::default(),
            feedback_used: Vec::new(),
            twins: HashMap::default(),
            gathers: HashMap::default(),
            gather_sources: HashMap::default(),
            tex_cache: HashMap::default(),
            tex_at: std::collections::BTreeMap::new(),
            newest: HashMap::default(),
            target_memo: [((u64::MAX, vk::Format::UNDEFINED, 0, 0, u64::MAX), ImageKey { cpu: 0, format: vk::Format::UNDEFINED, width: 0, height: 0 }); 4],
            generation: 0,
            volumes: HashMap::default(),
            retired: Vec::new(),
            pipeline_cache,
            compiler,
            cache_file,
            cache_saved: (0, std::time::Instant::now()),
            cache_saver: None,
            guest: std::ptr::null(),
            name,
            stats: RenderStats::default(),
            trace_state: String::new(),
        })
    }

    fn begin(&mut self) {
        if self.recording {
            return;
        }
        unsafe {
            self.device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        }
        self.recording = true;
        self.bound = Bound::default();
        if let Some((q, _)) = self.timestamps {
            let first = 2 * self.current as u32;
            unsafe {
                self.device.cmd_reset_query_pool(self.cmd, q, first, 2);
                self.device.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, first);
            }
        }
        self.barrier();
    }

    fn end_rendering(&mut self) {
        if self.rendering.take().is_some() {
            unsafe { self.device.cmd_end_rendering(self.cmd) };
            self.barrier();
        }
    }

    fn barrier(&mut self) {
        self.draw_wrote_storage = false;
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        unsafe {
            self.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &mb, &[], &[]);
        }
    }

    pub fn flush(&mut self) {
        if !self.recording {
            self.poll();
            return;
        }
        self.end_rendering();
        for key in std::mem::take(&mut self.retired) {
            self.destroy_image(key);
        }
        self.copy_pitch_targets();
        self.stage_write_back();
        let cmd = self.cmd;
        let batch = &mut self.batches[self.current];
        batch.serial = self.serial;
        batch.ring_end = self.ring.head;
        batch.write_back = std::mem::take(&mut self.write_back);
        batch.garbage = std::mem::take(&mut self.garbage);
        let fence = batch.fence;
        unsafe {
            if let Some((q, _)) = self.timestamps {
                self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 2 * self.current as u32 + 1);
            }
            self.device.end_command_buffer(cmd).unwrap();
            let cmds = [cmd];
            let waits: Vec<vk::Semaphore> = self.submit_wait.take().into_iter().collect();
            let stages = vec![vk::PipelineStageFlags::TRANSFER; waits.len()];
            let signals: Vec<vk::Semaphore> = self.submit_signal.take().into_iter().collect();
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds).wait_semaphores(&waits).wait_dst_stage_mask(&stages).signal_semaphores(&signals)];
            self.device.queue_submit(self.queue, &submit, fence).unwrap();
        }
        self.recording = false;
        self.in_flight.push_back(self.current);
        self.serial += 1;
        self.stats.batches += 1;
        self.storage.clear();
        self.checked.clear();
        self.cp_writes.clear();
        self.ring.recorded = 0;
        self.current = (self.current + 1) % IN_FLIGHT;
        while self.in_flight.contains(&self.current) {
            self.retire_oldest();
        }
        self.cmd = self.batches[self.current].cmd;
        self.descriptor_pool = self.batches[self.current].descriptor_pool;
        self.poll();
        if self.stats.pipelines > self.cache_saved.0 && self.cache_saved.1.elapsed() > PIPELINE_CACHE_SAVE {
            self.save_pipeline_cache(false);
        }
    }

    pub fn finish(&mut self) {
        self.flush();
        while self.retire_oldest() {}
    }

    fn poll(&mut self) {
        while let Some(&i) = self.in_flight.front() {
            if unsafe { self.device.get_fence_status(self.batches[i].fence) } != Ok(true) {
                return;
            }
            self.retire_oldest();
        }
    }

    fn retire_oldest(&mut self) -> bool {
        let Some(i) = self.in_flight.pop_front() else { return false };
        let fence = self.batches[i].fence;
        unsafe {
            if self.device.get_fence_status(fence) != Ok(true) {
                let start = std::time::Instant::now();
                self.device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
                self.stats.flush_time += start.elapsed();
                self.stats.waits += 1;
            }
            if let Some((q, period)) = self.timestamps {
                let mut ticks = [0u64; 2];
                if self.device.get_query_pool_results(q, 2 * i as u32, &mut ticks, vk::QueryResultFlags::TYPE_64).is_ok() {
                    self.stats.gpu_time += std::time::Duration::from_nanos((ticks[1].saturating_sub(ticks[0]) as f64 * period) as u64);
                }
            }
            self.device.reset_fences(&[fence]).unwrap();
            self.device.reset_command_buffer(self.batches[i].cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.reset_descriptor_pool(self.batches[i].descriptor_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
        }
        let batch = &mut self.batches[i];
        let (serial, ring_end) = (batch.serial, batch.ring_end);
        let write_back = std::mem::take(&mut batch.write_back);
        let garbage = std::mem::take(&mut batch.garbage);
        let from = match (&batch.staging, std::mem::take(&mut batch.staged)) {
            (Some(st), true) => {
                if !st.coherent {
                    let range = [vk::MappedMemoryRange::default().memory(st.memory).size(vk::WHOLE_SIZE)];
                    let _ = unsafe { self.device.invalidate_mapped_memory_ranges(&range) };
                }
                st.ptr
            }
            _ => self.ring.ptr,
        };
        if !self.guest.is_null() {
            // SAFETY: the guest memory outlives the renderer (both belong to the process).
            let mem = unsafe { &*self.guest };
            for &(cpu, offset, size) in &write_back {
                self.stats.written_back += size;
                let data = unsafe { std::slice::from_raw_parts(from.add(offset as usize), size as usize) };
                mem.write(cpu, data);
                self.note_cp_write(cpu, size);
                if let Some(dir) = crate::env_var!("KIRIKAE_SSBO_DUMP") {
                    let n = self.stats.dispatches + self.stats.draws;
                    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{n:08}_{cpu:x}.bin")), &data[..data.len().min(0x1000)]);
                }
            }
        }
        self.pending_pages.retire(serial);
        self.pending.retain(|p| p.2 != serial);
        self.pending_flag.store(!self.pending.is_empty(), std::sync::atomic::Ordering::Relaxed);
        for g in garbage {
            match g {
                Garbage::Image { image, memory, views } => {
                    unsafe {
                        for v in views {
                            self.device.destroy_image_view(v, None);
                        }
                        self.device.destroy_image(image, None);
                    }
                    self.free_image_memory(memory);
                }
                Garbage::Block(b) => self.buffers.release(b),
            }
        }
        self.ring.tail = if self.in_flight.is_empty() && self.ring.recorded == 0 { None } else { Some(ring_end) };
        true
    }

    fn stage_write_back(&mut self) {
        let total: u64 = self.write_back.iter().map(|w| w.2.next_multiple_of(16)).sum();
        if total == 0 {
            return;
        }
        let i = self.current;
        if self.batches[i].staging.as_ref().is_none_or(|st| st.size < total) {
            if let Some(st) = self.batches[i].staging.take() {
                unsafe {
                    self.device.destroy_buffer(st.buffer, None);
                    self.device.free_memory(st.memory, None);
                }
            }
            self.batches[i].staging = self.new_staging(total.next_power_of_two().max(1 << 20));
        }
        let Some(buffer) = self.batches[i].staging.as_ref().map(|st| st.buffer) else { return };
        let mut at = 0;
        let regions: Vec<vk::BufferCopy> = self
            .write_back
            .iter_mut()
            .map(|w| {
                let r = vk::BufferCopy { src_offset: w.1, dst_offset: at, size: w.2 };
                w.1 = at;
                at += w.2.next_multiple_of(16);
                r
            })
            .collect();
        self.end_rendering();
        self.barrier();
        let host = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)];
        unsafe {
            self.device.cmd_copy_buffer(self.cmd, self.ring.buffer, buffer, &regions);
            self.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &host, &[], &[]);
        }
        self.batches[i].staged = true;
    }

    fn new_staging(&self, size: u64) -> Option<Staging> {
        unsafe {
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::TRANSFER_DST), None).ok()?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let ty = find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED));
            let Some(ty) = ty else {
                self.device.destroy_buffer(buffer, None);
                return None;
            };
            let coherent = self.mem_props.memory_types[ty as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None).ok()?;
            self.device.bind_buffer_memory(buffer, memory, 0).ok()?;
            let ptr = self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).ok()? as *mut u8;
            Some(Staging { buffer, memory, ptr, size, coherent })
        }
    }

    fn copy_pitch_targets(&mut self) {
        if self.pitch_targets.is_empty() {
            return;
        }
        let targets: Vec<(ImageKey, (u64, u32, u32))> = self.pitch_targets.drain().collect();
        self.barrier();
        for (key, (cpu, pitch, rows)) in targets {
            if super::trace() {
                eprintln!("gpu: pitch target {cpu:#x} {:?} {}x{} pitch {pitch:#x} rows {rows}", key.format, key.width, key.height);
            }
            let Some(bpp) = bytes_per_pixel(key.format) else { continue };
            if matches!(key.format, vk::Format::R16G16_UNORM | vk::Format::R16G16_SNORM | vk::Format::R32_UINT | vk::Format::R32_SINT | vk::Format::R16G16B16A16_UNORM | vk::Format::R16G16B16A16_SNORM | vk::Format::R16G16B16A16_UINT | vk::Format::R16G16B16A16_SINT) {
                continue;
            }
            let (width, rows) = (key.width.min(pitch / bpp), rows.min(key.height));
            if !self.images.contains_key(&key) || width == 0 || rows == 0 || pitch % bpp != 0 {
                continue;
            }
            let size = pitch as u64 * (rows - 1) as u64 + (width * bpp) as u64;
            let Some((offset, _)) = self.ring_alloc(size, 16) else { continue };
            let region = [vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .buffer_row_length(pitch / bpp)
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width, height: rows, depth: 1 })];
            let image = self.images[&key].image;
            unsafe { self.device.cmd_copy_image_to_buffer(self.cmd, image, vk::ImageLayout::GENERAL, self.ring.buffer, &region) };
            self.write_back.push((cpu, offset, size));
        }
        self.barrier();
    }

    pub fn resolve(&mut self, cpu: u64, len: u64) {
        if !self.pending_pages.near(cpu, len) {
            return;
        }
        if self.pending_pages.overlapping(cpu, len).any(|(_, p)| p.2 == self.serial) {
            self.flush();
        }
        self.resolve_submitted(cpu, len);
    }

    fn resolve_submitted(&mut self, cpu: u64, len: u64) {
        if !self.pending_pages.near(cpu, len) {
            return;
        }
        let serial = self.serial;
        let Some(wait) = self.pending_pages.overlapping(cpu, len).map(|(_, p)| p.2).filter(|&s| s < serial).max() else { return };
        self.stats.resolves += 1;
        while self.pending.iter().any(|p| p.2 <= wait) && self.retire_oldest() {}
    }

    #[allow(clippy::too_many_arguments)]
    fn ring_guest(&mut self, gpu: &Gpu, mem: &GuestMemory, va: u64, size: u64, align: u64, zeros: bool) -> Result<u64, String> {
        let cpu = gpu.translate(va);
        if let Some(c) = cpu {
            self.resolve_submitted(c, size);
        }
        let offset = match gpu.bytes(mem, va, size) {
            Some(d) => self.ring_copy(d, align),
            None if zeros => self.ring_alloc(size, align).map(|(o, dst)| {
                dst.fill(0);
                o
            }),
            None => return Err(format!("guest memory at GPU address {va:#x} ({size:#x} bytes) unreadable")),
        };
        let offset = offset.ok_or("ring full")?;
        let serial = self.serial;
        if let Some(c) = cpu
            && self.pending_pages.near(c, size)
        {
            let mut found: Vec<(u64, Pending)> = self.pending_pages.overlapping(c, size).filter(|(_, p)| p.2 == serial && p.3 != offset).copied().collect();
            found.sort_unstable_by_key(|e| e.0);
            found.dedup_by_key(|e| e.0);
            let copies: Vec<vk::BufferCopy> = found
                .iter()
                .map(|(_, p)| {
                    let (start, end) = (p.0.max(c), (p.0 + p.1).min(c + size));
                    vk::BufferCopy { src_offset: p.3 + (start - p.0), dst_offset: offset + (start - c), size: end - start }
                })
                .collect();
            if !copies.is_empty() {
                self.stats.patches += 1;
                self.end_rendering();
                self.barrier();
                unsafe { self.device.cmd_copy_buffer(self.cmd, self.ring.buffer, self.ring.buffer, &copies) };
                self.barrier();
            }
        }
        Ok(offset)
    }

    fn pending_in_ring(&self, cpu: u64, size: u64, align: u64) -> Option<u64> {
        if self.draw_wrote_storage || !self.pending_pages.near(cpu, size) {
            return None;
        }
        let serial = self.serial;
        let (_, p) = self.pending_pages.overlapping(cpu, size).max_by_key(|e| e.0)?;
        (p.2 == serial && p.0 <= cpu && cpu + size <= p.0 + p.1).then(|| p.3 + (cpu - p.0)).filter(|o| o % align == 0)
    }

    fn split(&mut self) {
        if self.ring.recorded > BATCH_RING_LIMIT {
            self.flush();
        }
    }

    fn save_pipeline_cache(&mut self, wait: bool) {
        if !wait && self.cache_saver.as_ref().is_some_and(|t| !t.is_finished()) {
            return;
        }
        if let Some(t) = self.cache_saver.take() {
            let _ = t.join();
        }
        self.cache_saved = (self.stats.pipelines, std::time::Instant::now());
        let Some(file) = self.cache_file.clone() else { return };
        let (device, cache) = (self.device.clone(), self.pipeline_cache);
        let save = move || {
            let Ok(data) = (unsafe { device.get_pipeline_cache_data(cache) }) else { return };
            if data.is_empty() {
                return;
            }
            if let Some(dir) = file.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = file.with_extension(format!("{}.tmp", std::process::id()));
            if std::fs::write(&tmp, &data).is_err() || std::fs::rename(&tmp, &file).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        };
        if wait {
            save();
        } else {
            self.cache_saver = std::thread::Builder::new().name("kirikae-pipeline-cache".into()).spawn(save).ok();
        }
    }

    fn touch(&mut self, key: ImageKey) {
        self.clock += 1;
        let img = self.images.get_mut(&key).expect("image exists");
        img.gpu_dirty = true;
        img.stamp = self.clock;
        self.restamped(key);
    }

    fn restamped(&mut self, key: ImageKey) {
        if key.cpu & DERIVED == 0 && self.newest.insert(key.cpu, key) != Some(key) && self.at(key.cpu).len() > 1 {
            self.forget_textures(key.cpu, false);
        }
    }

    fn forget_textures(&mut self, cpu: u64, layered: bool) {
        let end = if layered { cpu.saturating_add(1 << 32) } else { cpu };
        let found: Vec<u64> = self.tex_at.range(cpu..=end).map(|(&a, _)| a).collect();
        for a in found {
            for k in self.tex_at.remove(&a).unwrap_or_default() {
                self.tex_cache.remove(&k);
            }
        }
    }

    fn ring_alloc(&mut self, size: u64, align: u64) -> Option<(u64, &mut [u8])> {
        let size = size.max(4);
        if size > RING_SIZE / 2 {
            return None;
        }
        let offset = loop {
            if let Some(o) = self.ring.fit(size, align.max(4)) {
                break o;
            }
            if !self.retire_oldest() {
                return None;
            }
        };
        self.stats.ring_bytes += size;
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ring.ptr.add(offset as usize), size as usize) };
        Some((offset, slice))
    }

    fn ring_copy(&mut self, data: &[u8], align: u64) -> Option<u64> {
        let (offset, dst) = self.ring_alloc(data.len() as u64, align)?;
        dst[..data.len()].copy_from_slice(data);
        Some(offset)
    }

    fn create_image(&mut self, key: ImageKey, levels: u32, layers: u32) -> Result<(), vk::Result> {
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let format_props = unsafe { self.instance.get_physical_device_format_properties(self.physical, key.format) };
        let usage = if format_props.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) { usage | vk::ImageUsageFlags::STORAGE } else { usage };
        let usage = if format_props.optimal_tiling_features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usage | vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else if format_props.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
            usage | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            usage
        };
        let flags = if layers >= 6 && layers.is_multiple_of(6) && key.width == key.height {
            vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::MUTABLE_FORMAT
        };
        let info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(key.format)
            .extent(vk::Extent3D { width: key.width.max(1), height: key.height.max(1), depth: 1 })
            .mip_levels(levels.max(1))
            .array_layers(layers.max(1))
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&info, None)? };
        let req = unsafe { self.device.get_image_memory_requirements(image) };
        let ty = find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::empty()))
            .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
        let memory = self.image_memory(req.size, ty, key.cpu & CROP != 0)?;
        unsafe { self.device.bind_image_memory(image, memory, 0)? };
        self.begin();
        let aspect = aspect_of(key.format);
        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        self.end_rendering();
        unsafe {
            self.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &barrier);
        }
        self.clock += 1;
        self.images.insert(key, Image { image, memory, views: HashMap::default(), levels: levels.max(1), layers: layers.max(1), gpu_dirty: false, stamp: self.clock, source: None, drawn: None, cleared: None, clear_streak: 0 });
        if key.cpu & DERIVED == 0 {
            self.generation += 1;
            self.forget_textures(key.cpu, layers > 1);
            self.newest.insert(key.cpu, key);
        }
        let at = self.by_cpu.entry(key.cpu).or_default();
        if !at.contains(&key) {
            at.push(key);
        }
        Ok(())
    }

    fn image_memory(&mut self, size: u64, ty: u32, pooled: bool) -> Result<vk::DeviceMemory, vk::Result> {
        let size = if pooled { size.next_power_of_two().max(1 << 16) } else { size };
        if pooled && let Some(m) = self.free_memory.get_mut(&(ty, size)).and_then(Vec::pop) {
            self.pooled_memory.insert(m, (ty, size));
            return Ok(m);
        }
        let m = unsafe { self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty), None)? };
        if pooled {
            self.pooled_memory.insert(m, (ty, size));
        }
        Ok(m)
    }

    fn free_image_memory(&mut self, memory: vk::DeviceMemory) {
        const POOLED: usize = 8;
        if let Some(class) = self.pooled_memory.remove(&memory) {
            let free = self.free_memory.entry(class).or_default();
            if free.len() < POOLED {
                free.push(memory);
                return;
            }
        }
        unsafe { self.device.free_memory(memory, None) };
    }

    fn view(&mut self, key: ImageKey, format: vk::Format, swizzle: [vk::ComponentSwizzle; 4]) -> vk::ImageView {
        self.view_of(key, format, swizzle, vk::ImageViewType::TYPE_2D)
    }

    fn view_of(&mut self, key: ImageKey, format: vk::Format, swizzle: [vk::ComponentSwizzle; 4], view_type: vk::ImageViewType) -> vk::ImageView {
        self.view_range(key, format, swizzle, view_type, (0, u32::MAX), 0)
    }

    fn view_range(&mut self, key: ImageKey, format: vk::Format, swizzle: [vk::ComponentSwizzle; 4], view_type: vk::ImageViewType, levels: (u32, u32), layer: u32) -> vk::ImageView {
        let device = self.device.clone();
        let img = self.images.get_mut(&key).expect("image exists");
        let base = levels.0.min(img.levels - 1);
        let count = levels.1.clamp(1, img.levels - base);
        let layer = layer.min(img.layers - 1);
        if let Some(&v) = img.views.get(&(format, swizzle, view_type, base, count, layer)) {
            return v;
        }
        let layers = match view_type {
            vk::ImageViewType::TYPE_2D => 1,
            vk::ImageViewType::CUBE => 6,
            vk::ImageViewType::CUBE_ARRAY => (img.layers - layer) / 6 * 6,
            _ => img.layers - layer,
        };
        let info = vk::ImageViewCreateInfo::default()
            .image(img.image)
            .view_type(view_type)
            .format(format)
            .components(vk::ComponentMapping { r: swizzle[0], g: swizzle[1], b: swizzle[2], a: swizzle[3] })
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect_of(format) & !vk::ImageAspectFlags::STENCIL, base_mip_level: base, level_count: count, base_array_layer: layer, layer_count: layers });
        let v = unsafe { device.create_image_view(&info, None).unwrap() };
        img.views.insert((format, swizzle, view_type, base, count, layer), v);
        v
    }

    fn clock_start(&self) -> std::time::Instant {
        if self.timing { std::time::Instant::now() } else { self.created }
    }

    fn lap(&mut self, part: usize, since: &mut std::time::Instant) {
        if !self.timing {
            return;
        }
        let now = std::time::Instant::now();
        self.stats.draw_parts[part] += now - *since;
        *since = now;
    }

    fn at(&self, cpu: u64) -> &[ImageKey] {
        self.by_cpu.get(&cpu).map_or(&[], |v| v.as_slice())
    }

    fn find_image(&self, cpu: u64, width: u32, height: u32, like: vk::Format) -> Option<ImageKey> {
        let class = aspect_of(like) == vk::ImageAspectFlags::COLOR;
        self.at(cpu).iter().copied().filter(|k| k.width >= width && k.height >= height).min_by_key(|k| {
            (
                (aspect_of(k.format) == vk::ImageAspectFlags::COLOR) != class,
                !same_class(k.format, like),
                k.width != width || k.height != height,
                std::cmp::Reverse(self.images[k].stamp),
                k.width * k.height,
            )
        })
    }

    fn target_image(&mut self, cpu: u64, format: vk::Format, width: u32, height: u32) -> Option<(ImageKey, bool)> {
        let asked = (cpu, format, width, height, self.generation);
        if let Some(&(_, k)) = self.target_memo.iter().find(|m| m.0 == asked) {
            return Some((k, false));
        }
        if let Some(k) = self.at(cpu).iter().copied().find(|k| k.format == format && k.width >= width && k.height >= height) {
            self.target_memo.rotate_right(1);
            self.target_memo[0] = (asked, k);
            return Some((k, false));
        }
        let color = aspect_of(format) == vk::ImageAspectFlags::COLOR;
        let same_format: Vec<ImageKey> = self.at(cpu).iter().copied().filter(|k| k.format == format).collect();
        let w = same_format.iter().map(|k| k.width).fold(width, u32::max);
        let h = same_format.iter().map(|k| k.height).fold(height, u32::max);
        let mut others: Vec<ImageKey> = self.at(cpu).iter().copied().filter(|k| k.format != format && (aspect_of(k.format) == vk::ImageAspectFlags::COLOR) == color).collect();
        others.sort_by_key(|k| std::cmp::Reverse(self.images[k].stamp));
        let evicted = others.split_off(others.len().min(3));
        for k in same_format.into_iter().chain(evicted) {
            if super::trace() {
                eprintln!("gpu: image {:#x} {:?} {}x{} replaced by {format:?} {width}x{height}", k.cpu, k.format, k.width, k.height);
            }
            self.destroy_image(k);
        }
        let key = ImageKey { cpu, format, width: w, height: h };
        self.create_image(key, 1, 1).ok()?;
        Some((key, true))
    }

    fn upload(&mut self, key: ImageKey, levels: &[(u32, u32, Vec<u8>)]) {
        self.upload_layers(key, levels.len(), levels);
    }

    fn upload_layers(&mut self, key: ImageKey, per_layer: usize, levels: &[(u32, u32, Vec<u8>)]) {
        self.begin();
        self.end_rendering();
        let mut regions = Vec::new();
        for (i, (w, h, data)) in levels.iter().enumerate() {
            let Some(offset) = self.ring_copy(data, 16) else { continue };
            let (layer, level) = (i / per_layer.max(1), i % per_layer.max(1));
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect_of(key.format) & !vk::ImageAspectFlags::STENCIL, mip_level: level as u32, base_array_layer: layer as u32, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: *w, height: *h, depth: 1 }),
            );
        }
        let image = self.images[&key].image;
        self.barrier();
        unsafe { self.device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, image, vk::ImageLayout::GENERAL, &regions) };
        self.barrier();
        self.stats.uploads += 1;
        self.clock += 1;
        self.images.get_mut(&key).unwrap().stamp = self.clock;
        self.restamped(key);
    }

    pub fn surface_image(&mut self, gpu: &Gpu, mem: &GuestMemory, s: &Surface, upload: bool) -> Option<(ImageKey, ColorFormat)> {
        let fmt = formats::color_target(s.format)?;
        let cpu = gpu.translate(s.va)?;
        let (key, created) = self.target_image(cpu, fmt.vk, s.width, s.height)?;
        if created && upload && key.width == s.width && key.height == s.height
            && let Some(data) = read_surface(gpu, mem, s, fmt.bytes) {
                self.upload(key, &[(s.width, s.height, data)]);
            }
        Some((key, fmt))
    }

    fn destroy_image(&mut self, key: ImageKey) {
        self.crops.remove(&key);
        if self.feedbacks.remove(&key).is_some() {
            self.destroy_image(ImageKey { cpu: key.cpu | FEEDBACK, ..key });
        }
        self.gathers.remove(&key);
        self.gather_sources.remove(&key);
        if key.cpu & DERIVED == 0 {
            self.generation += 1;
            let layered = self.images.get(&key).is_some_and(|i| i.layers > 1);
            self.forget_textures(key.cpu, layered);
            if self.newest.get(&key.cpu) == Some(&key) {
                self.newest.remove(&key.cpu);
            }
        } else {
            self.tex_cache.retain(|_, e| e.image != key);
        }
        let copies: Vec<ImageKey> = self.crops.iter().filter(|(_, (src, _))| *src == key).map(|(k, _)| *k).collect();
        for k in copies {
            self.destroy_image(k);
        }
        if let Some(at) = self.by_cpu.get_mut(&key.cpu) {
            at.retain(|k| *k != key);
            if at.is_empty() {
                self.by_cpu.remove(&key.cpu);
            }
        }
        if let Some(img) = self.images.remove(&key) {
            self.garbage.push(Garbage::Image { image: img.image, memory: img.memory, views: img.views.into_values().collect() });
        }
    }

    fn depth_format(&self, zt: u32) -> Option<vk::Format> {
        let supports = |f: vk::Format| unsafe { self.instance.get_physical_device_format_properties(self.physical, f) }.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
        let pick = |list: &[vk::Format]| list.iter().copied().find(|&f| supports(f));
        use vk::Format as F;
        match zt {
            0x13 => pick(&[F::D16_UNORM, F::D32_SFLOAT]),
            0x0a | 0x1e => pick(&[F::D32_SFLOAT]),
            0x19 | 0x1f => pick(&[F::D32_SFLOAT_S8_UINT, F::D24_UNORM_S8_UINT]),
            0x14 | 0x15 | 0x16 | 0x18 | 0x1d => pick(&[F::D24_UNORM_S8_UINT, F::D32_SFLOAT_S8_UINT]),
            _ => None,
        }
    }

    fn depth_target(&mut self, gpu: &Gpu, regs: &[u32]) -> Option<ImageKey> {
        if regs[0x54e] & 1 == 0 {
            return None;
        }
        let format = self.depth_format(regs[0x3fa])?;
        let cpu = gpu.translate(address(regs, 0x3f8))?;
        let (w, h) = (regs[0x48a], regs[0x48b] & 0x1ffff);
        if w == 0 || h == 0 {
            return None;
        }
        Some(self.target_image(cpu, format, w, h)?.0)
    }

    pub fn clear_depth(&mut self, gpu: &Gpu, regs: &[u32], depth: bool, stencil: bool) {
        let Some(key) = self.depth_target(gpu, regs) else { return };
        let aspects = aspect_of(key.format);
        let mut mask = vk::ImageAspectFlags::empty();
        if depth {
            mask |= vk::ImageAspectFlags::DEPTH;
        }
        if stencil && aspects.contains(vk::ImageAspectFlags::STENCIL) {
            mask |= vk::ImageAspectFlags::STENCIL;
        }
        if mask.is_empty() {
            return;
        }
        self.begin();
        self.end_rendering();
        let depth = f32::from_bits(regs[0x364]).clamp(0.0, 1.0);
        let depth = if key.format == vk::Format::D16_UNORM { (depth * 65535.0 + 0.25).floor() / 65535.0 } else { depth };
        let value = vk::ClearDepthStencilValue { depth, stencil: regs[0x368] & 0xff };
        let range = [vk::ImageSubresourceRange { aspect_mask: mask, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }];
        let image = self.images[&key].image;
        unsafe { self.device.cmd_clear_depth_stencil_image(self.cmd, image, vk::ImageLayout::GENERAL, &value, &range) };
        self.barrier();
        self.touch(key);
        self.images.get_mut(&key).unwrap().note_clear(crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
    }

    pub fn clear(&mut self, gpu: &Gpu, mem: &GuestMemory, s: &Surface, color: [u32; 4]) {
        let Some((key, fmt)) = self.surface_image(gpu, mem, s, false) else { return };
        self.begin();
        self.end_rendering();
        let value = match fmt.kind {
            Kind::Float => vk::ClearColorValue { float32: color.map(f32::from_bits) },
            Kind::Sint => vk::ClearColorValue { int32: color.map(|c| f32::from_bits(c) as i32) },
            Kind::Uint => vk::ClearColorValue { uint32: color.map(|c| f32::from_bits(c) as u32) },
        };
        let range = [vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }];
        let image = self.images[&key].image;
        unsafe { self.device.cmd_clear_color_image(self.cmd, image, vk::ImageLayout::GENERAL, &value, &range) };
        self.barrier();
        self.touch(key);
        self.images.get_mut(&key).unwrap().note_clear(crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn blit(&mut self, gpu: &Gpu, mem: &GuestMemory, src: &Surface, dst: &Surface, src_rect: [f32; 4], dst_rect: [i32; 4], linear: bool) -> bool {
        let depth_src = gpu.translate(src.va).and_then(|cpu| {
            self.at(cpu).iter().copied().find(|k| self.images[k].gpu_dirty && aspect_of(k.format) != vk::ImageAspectFlags::COLOR && k.width >= src.width && k.height >= src.height)
        });
        let (sk, dk, aspect) = match depth_src {
            Some(sk) if formats::color_target(src.format).is_some_and(|f| f.bytes == 4) => {
                let Some(dcpu) = gpu.translate(dst.va) else { return false };
                let Some((dk, _)) = self.target_image(dcpu, sk.format, dst.width, dst.height) else { return false };
                (sk, dk, aspect_of(sk.format))
            }
            _ => {
                let Some((sk, _)) = self.surface_image(gpu, mem, src, true) else { return false };
                let Some((dk, _)) = self.surface_image(gpu, mem, dst, true) else { return false };
                (sk, dk, vk::ImageAspectFlags::COLOR)
            }
        };
        let linear = linear && aspect == vk::ImageAspectFlags::COLOR;
        self.begin();
        self.end_rendering();
        let sub = vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let clamp = |v: f32, max: u32| (v.round() as i32).clamp(0, max as i32);
        let region = [vk::ImageBlit::default()
            .src_subresource(sub)
            .src_offsets([
                vk::Offset3D { x: clamp(src_rect[0], src.width), y: clamp(src_rect[1], src.height), z: 0 },
                vk::Offset3D { x: clamp(src_rect[2], src.width), y: clamp(src_rect[3], src.height), z: 1 },
            ])
            .dst_subresource(sub)
            .dst_offsets([
                vk::Offset3D { x: dst_rect[0].clamp(0, dst.width as i32), y: dst_rect[1].clamp(0, dst.height as i32), z: 0 },
                vk::Offset3D { x: dst_rect[2].clamp(0, dst.width as i32), y: dst_rect[3].clamp(0, dst.height as i32), z: 1 },
            ])];
        let (si, di) = (self.images[&sk].image, self.images[&dk].image);
        let filter = if linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        unsafe { self.device.cmd_blit_image(self.cmd, si, vk::ImageLayout::GENERAL, di, vk::ImageLayout::GENERAL, &region, filter) };
        self.barrier();
        self.touch(dk);
        true
    }

    pub fn note_write(&mut self, cpu: u64, len: u64) {
        if !self.checked.is_empty() {
            self.note_cp_write(cpu, len);
        }
    }

    fn note_cp_write(&mut self, cpu: u64, len: u64) {
        self.cp_writes.push(cpu, len);
    }

    pub fn forget_guest_range(&mut self, cpu: u64, len: u64) {
        let stale: Vec<ImageKey> = self.by_cpu.range(cpu..cpu + len).flat_map(|(_, keys)| keys.iter().copied()).filter(|k| k.cpu & CROP == 0).collect();
        for key in stale {
            self.destroy_image(key);
        }
    }

    pub fn present(&mut self, cpu: u64, width: u32, height: u32, flip_h: bool, flip_v: bool) -> Presented {
        if self.swapchain_fn.is_none() || self.present_failed {
            return Presented::NoSwapchain;
        }
        let Some(key) = self.find_image(cpu, width, height, vk::Format::R8G8B8A8_UNORM).filter(|k| self.images[k].gpu_dirty && aspect_of(k.format) == vk::ImageAspectFlags::COLOR) else {
            return Presented::NoImage;
        };
        self.present_image(key, width, height, flip_h, flip_v)
    }

    pub fn present_pixels(&mut self, width: u32, height: u32, pixels: &[u32]) -> Presented {
        if self.swapchain_fn.is_none() || self.present_failed || pixels.len() != (width * height) as usize {
            return Presented::NoSwapchain;
        }
        let key = ImageKey { cpu: PRESENT, format: vk::Format::B8G8R8A8_UNORM, width, height };
        if !self.images.contains_key(&key) {
            let old: Vec<ImageKey> = self.images.keys().copied().filter(|k| k.cpu == PRESENT).collect();
            for k in old {
                self.destroy_image(k);
            }
            if self.create_image(key, 1, 1).is_err() {
                return Presented::NoSwapchain;
            }
        }
        let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
        self.upload(key, &[(width, height, bytes)]);
        self.present_image(key, width, height, false, false)
    }

    fn present_image(&mut self, key: ImageKey, width: u32, height: u32, flip_h: bool, flip_v: bool) -> Presented {
        let srgb = matches!(key.format, vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32);
        let dump = crate::env_var!("KIRIKAE_PRESENT_DUMP").filter(|_| self.stats.presents % 60 == 0);
        let readback = match (&dump, &self.swapchain) {
            (Some(_), Some(sc)) if sc.extent.width > 0 => self.readback_buffer(sc.extent.width as u64 * sc.extent.height as u64 * 4).map(|(b, _)| b),
            _ => None,
        };
        let Some(index) = self.acquire(srgb) else {
            return if self.swapchain.is_some() { Presented::Shown } else { Presented::NoSwapchain };
        };
        let sc = self.swapchain.as_ref().unwrap();
        let (target, extent, done) = (sc.images[index as usize], sc.extent, sc.done[index as usize]);
        self.begin();
        self.end_rendering();
        self.barrier();
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let to_dst = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .image(target)
            .subresource_range(range)];
        let (w, h) = (width.min(key.width) as i32, height.min(key.height) as i32);
        let (x0, x1) = if flip_h { (extent.width as i32, 0) } else { (0, extent.width as i32) };
        let (y0, y1) = if flip_v { (extent.height as i32, 0) } else { (0, extent.height as i32) };
        let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let region = [vk::ImageBlit::default()
            .src_subresource(sub)
            .src_offsets([vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: w, y: h, z: 1 }])
            .dst_subresource(sub)
            .dst_offsets([vk::Offset3D { x: x0, y: y0, z: 0 }, vk::Offset3D { x: x1, y: y1, z: 1 }])];
        let to_present = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .image(target)
            .subresource_range(range)];
        let source = self.images[&key].image;
        let readback = readback.filter(|_| self.swapchain.as_ref().is_some_and(|sc| sc.readable && sc.extent == extent));
        unsafe {
            let d = &self.device;
            d.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_dst);
            d.cmd_blit_image(self.cmd, source, vk::ImageLayout::GENERAL, target, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &region, vk::Filter::LINEAR);
            if let Some(buffer) = readback {
                let to_src = [vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .image(target)
                    .subresource_range(range)];
                d.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_src);
                let copy = [vk::BufferImageCopy::default().image_subresource(sub).image_extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })];
                d.cmd_copy_image_to_buffer(self.cmd, target, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &copy);
            }
            let from = if readback.is_some() { vk::ImageLayout::TRANSFER_SRC_OPTIMAL } else { vk::ImageLayout::TRANSFER_DST_OPTIMAL };
            let to_present = [to_present[0].old_layout(from)];
            d.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &to_present);
        }
        self.submit_wait = Some(self.batches[self.current].acquired);
        self.submit_signal = Some(done);
        self.flush();
        let sc = self.swapchain.as_mut().unwrap();
        let (waits, swapchains, indices) = ([done], [sc.handle], [index]);
        let info = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&swapchains).image_indices(&indices);
        match unsafe { self.swapchain_fn.as_ref().unwrap().queue_present(self.queue, &info) } {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => sc.stale = true,
            Err(e) => {
                eprintln!("kirikae: presenting failed ({e:?}); the swapchain is made again");
                sc.stale = true;
            }
        }
        let format = sc.format;
        self.stats.presents += 1;
        if let (Some(dir), Some(_)) = (dump, readback) {
            self.finish();
            if let Some((_, memory, _, ptr)) = self.readback {
                if !self.readback_coherent {
                    let range = [vk::MappedMemoryRange::default().memory(memory).offset(0).size(vk::WHOLE_SIZE)];
                    unsafe { self.device.invalidate_mapped_memory_ranges(&range).ok() };
                }
                let bytes = unsafe { std::slice::from_raw_parts(ptr, (extent.width * extent.height * 4) as usize) };
                if let Some(pixels) = decode_pixels(format, bytes) {
                    let path = std::path::Path::new(&dir).join(format!("present{:06}.png", self.stats.presents));
                    let _ = crate::display::write_png(&path, extent.width, extent.height, &pixels);
                }
            }
        }
        Presented::Shown
    }

    fn acquire(&mut self, srgb: bool) -> Option<u32> {
        for _ in 0..2 {
            self.prepare_swapchain(srgb)?;
            let sc = self.swapchain.as_mut()?;
            let semaphore = self.batches[self.current].acquired;
            match unsafe { self.swapchain_fn.as_ref()?.acquire_next_image(sc.handle, 100_000_000, semaphore, vk::Fence::null()) } {
                Ok((index, suboptimal)) => {
                    sc.stale |= suboptimal;
                    return Some(index);
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => sc.stale = true,
                Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                    self.drop_swapchain(true);
                }
                Err(_) => return None,
            }
        }
        None
    }

    fn prepare_swapchain(&mut self, srgb: bool) -> Option<()> {
        let surface_fn = self.surface_fn.clone()?;
        let surface = match &self.swapchain {
            Some(sc) => sc.surface,
            None => {
                let surface = self.create_surface()?;
                let supported = unsafe { surface_fn.get_physical_device_surface_support(self.physical, self.family, surface) }.unwrap_or(false);
                if !supported {
                    eprintln!("kirikae: the graphics queue cannot present to the window; frames are read back instead");
                    unsafe { surface_fn.destroy_surface(surface, None) };
                    self.present_failed = true;
                    return None;
                }
                self.swapchain = Some(Swapchain {
                    surface,
                    handle: vk::SwapchainKHR::null(),
                    images: Vec::new(),
                    extent: vk::Extent2D::default(),
                    srgb,
                    done: Vec::new(),
                    stale: true,
                    format: vk::Format::UNDEFINED,
                    readable: false,
                });
                surface
            }
        };
        let caps = unsafe { surface_fn.get_physical_device_surface_capabilities(self.physical, surface) }.ok()?;
        let extent = caps.current_extent;
        if extent.width == 0 || extent.height == 0 || extent.width == u32::MAX {
            return None;
        }
        let sc = self.swapchain.as_ref()?;
        if !sc.stale && sc.extent == extent && sc.srgb == srgb {
            return Some(());
        }
        if !caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DST) {
            eprintln!("kirikae: the window's swapchain cannot be blitted to; frames are read back instead");
            self.drop_swapchain(true);
            self.present_failed = true;
            return None;
        }
        self.finish();
        unsafe { self.device.device_wait_idle().ok()? };
        let formats = unsafe { surface_fn.get_physical_device_surface_formats(self.physical, surface) }.ok()?;
        let wanted: &[vk::Format] = if srgb { &[vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB] } else { &[vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM] };
        let blit_dst = |f: vk::Format| unsafe { self.instance.get_physical_device_format_properties(self.physical, f) }.optimal_tiling_features.contains(vk::FormatFeatureFlags::BLIT_DST);
        let format = wanted
            .iter()
            .find_map(|w| formats.iter().find(|f| f.format == *w))
            .or_else(|| formats.iter().find(|f| blit_dst(f.format)))
            .copied()?;
        let modes = unsafe { surface_fn.get_physical_device_surface_present_modes(self.physical, surface) }.unwrap_or_default();
        let order: &[vk::PresentModeKHR] =
            if crate::env_var!("KIRIKAE_VSYNC").is_some() { &[vk::PresentModeKHR::FIFO] } else { &[vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::FIFO] };
        let mode = order.iter().copied().find(|m| modes.contains(m)).unwrap_or(vk::PresentModeKHR::FIFO);
        let count = if caps.max_image_count > 0 { (caps.min_image_count + 1).min(caps.max_image_count) } else { caps.min_image_count + 1 };
        let old = self.swapchain.as_ref().map_or(vk::SwapchainKHR::null(), |sc| sc.handle);
        let readable = crate::env_var!("KIRIKAE_PRESENT_DUMP").is_some() && caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_SRC);
        let usage = if readable { vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC } else { vk::ImageUsageFlags::TRANSFER_DST };
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(mode)
            .clipped(true)
            .old_swapchain(old);
        let swapchain_fn = self.swapchain_fn.clone()?;
        let handle = match unsafe { swapchain_fn.create_swapchain(&info, None) } {
            Ok(h) => h,
            Err(e) => {
                eprintln!("kirikae: cannot make the window's swapchain ({e:?}); frames are read back instead");
                self.drop_swapchain(true);
                self.present_failed = true;
                return None;
            }
        };
        self.drop_swapchain(false);
        let images = unsafe { swapchain_fn.get_swapchain_images(handle) }.ok()?;
        let done = images.iter().map(|_| unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() }).collect();
        let sc = self.swapchain.as_mut()?;
        *sc = Swapchain { surface, handle, images, extent, srgb, done, stale: false, format: format.format, readable };
        if super::trace() || self.stats.presents == 0 {
            eprintln!("kirikae: presenting to the window through a swapchain ({}x{}, {:?}, {mode:?})", extent.width, extent.height, format.format);
        }
        Some(())
    }

    fn drop_swapchain(&mut self, surface: bool) {
        let Some(sc) = self.swapchain.as_mut() else { return };
        unsafe {
            let _ = self.device.device_wait_idle();
            for s in sc.done.drain(..) {
                self.device.destroy_semaphore(s, None);
            }
            if sc.handle != vk::SwapchainKHR::null()
                && let Some(f) = &self.swapchain_fn
            {
                f.destroy_swapchain(sc.handle, None);
            }
            sc.handle = vk::SwapchainKHR::null();
            sc.images.clear();
            sc.stale = true;
        }
        if surface {
            let sc = self.swapchain.take().unwrap();
            if let Some(f) = &self.surface_fn {
                unsafe { f.destroy_surface(sc.surface, None) };
            }
        }
    }

    #[cfg(windows)]
    fn create_surface(&mut self) -> Option<vk::SurfaceKHR> {
        let (hwnd, hinstance) = crate::display::window_handle()?;
        let info = vk::Win32SurfaceCreateInfoKHR::default().hwnd(hwnd).hinstance(hinstance);
        match unsafe { self.win32_fn.as_ref()?.create_win32_surface(&info, None) } {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("kirikae: cannot make a Vulkan surface for the window ({e:?}); frames are read back instead");
                self.present_failed = true;
                None
            }
        }
    }

    #[cfg(not(windows))]
    fn create_surface(&mut self) -> Option<vk::SurfaceKHR> {
        None
    }

    pub fn rendered(&self, cpu: u64, width: u32, height: u32) -> bool {
        self.find_image(cpu, width, height, vk::Format::R8G8B8A8_UNORM).is_some_and(|k| self.images[&k].gpu_dirty)
    }

    pub fn read_pixels(&mut self, cpu: u64, width: u32, height: u32) -> Option<Vec<u32>> {
        let key = self.find_image(cpu, width, height, vk::Format::R8G8B8A8_UNORM)?;
        if !self.images[&key].gpu_dirty {
            return None;
        }
        let pixels = self.read_image(key)?;
        if key.width == width && key.height == height {
            return Some(pixels);
        }
        let mut out = Vec::with_capacity((width * height) as usize);
        for y in 0..height as usize {
            out.extend_from_slice(&pixels[y * key.width as usize..][..width as usize]);
        }
        Some(out)
    }

    fn read_image(&mut self, key: ImageKey) -> Option<Vec<u32>> {
        let start = std::time::Instant::now();
        let r = self.read_image_inner(key);
        self.stats.readback_time += start.elapsed();
        r
    }

    fn read_image_inner(&mut self, key: ImageKey) -> Option<Vec<u32>> {
        let bytes = self.read_raw(key, 1)?;
        decode_pixels(key.format, &bytes)
    }

    fn read_raw(&mut self, key: ImageKey, layers: u32) -> Option<Vec<u8>> {
        let bpp = bytes_per_pixel(key.format)?;
        let size = (key.width * key.height * bpp) as u64 * layers as u64;
        let (buffer, ptr) = self.readback_buffer(size)?;
        self.begin();
        self.end_rendering();
        let region = [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect_of(key.format) & !vk::ImageAspectFlags::STENCIL, mip_level: 0, base_array_layer: 0, layer_count: layers })
            .image_extent(vk::Extent3D { width: key.width, height: key.height, depth: 1 })];
        let image = self.images[&key].image;
        unsafe { self.device.cmd_copy_image_to_buffer(self.cmd, image, vk::ImageLayout::GENERAL, buffer, &region) };
        self.barrier();
        self.finish();
        if !self.readback_coherent
            && let Some((_, memory, _, _)) = self.readback
        {
            let range = [vk::MappedMemoryRange::default().memory(memory).offset(0).size(vk::WHOLE_SIZE)];
            unsafe { self.device.invalidate_mapped_memory_ranges(&range).ok()? };
        }
        Some(unsafe { std::slice::from_raw_parts(ptr, size as usize) }.to_vec())
    }

    pub fn dump_images(&mut self, dir: &std::path::Path, tag: u64) {
        let only: Option<Vec<u64>> = std::env::var("KIRIKAE_TARGET_DUMP_ONLY").ok().map(|v| v.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect());
        let derived = crate::env_var!("KIRIKAE_TARGET_DUMP_DERIVED").is_some();
        let keys: Vec<ImageKey> = self
            .images
            .iter()
            .filter(|(k, i)| (i.gpu_dirty || (derived && k.cpu & DERIVED != 0)) && only.as_ref().is_none_or(|a| a.contains(&(k.cpu & !DERIVED))))
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            let layers = if crate::env_var!("KIRIKAE_TARGET_DUMP_RAW").is_some() { self.images[&key].layers } else { 1 };
            let Some(raw) = self.read_raw(key, layers) else { continue };
            if crate::env_var!("KIRIKAE_TARGET_DUMP_RAW").is_some() {
                let name = format!("f{tag:06}_{:x}_{:?}_{}x{}.raw", key.cpu, key.format, key.width, key.height);
                let _ = std::fs::write(dir.join(name), &raw);
            }
            let Some(pixels) = decode_pixels(key.format, &raw) else { continue };
            let name = format!("f{tag:06}_{:x}_{:?}_{}x{}.png", key.cpu, key.format, key.width, key.height);
            let _ = crate::display::write_png(&dir.join(name), key.width, key.height, &pixels);
        }
    }

    fn readback_buffer(&mut self, size: u64) -> Option<(vk::Buffer, *mut u8)> {
        if let Some((b, _, s, p)) = self.readback
            && s >= size
        {
            return Some((b, p));
        }
        self.finish();
        unsafe {
            if let Some((b, m, _, _)) = self.readback.take() {
                self.device.destroy_buffer(b, None);
                self.device.free_memory(m, None);
            }
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::TRANSFER_DST), None).ok()?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let ty = find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED))
                .or_else(|| find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT))?;
            self.readback_coherent = self.mem_props.memory_types[ty as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None).ok()?;
            self.device.bind_buffer_memory(buffer, memory, 0).ok()?;
            let ptr = self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).ok()? as *mut u8;
            self.readback = Some((buffer, memory, size, ptr));
            Some((buffer, ptr))
        }
    }

    pub fn dispatch(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], qmd: &super::compute::Qmd, va: u64) -> bool {
        let start = std::time::Instant::now();
        let ok = self.try_dispatch(gpu, mem, regs, qmd, va);
        self.stats.draw_time += start.elapsed();
        match &ok {
            Err(why) => {
                self.stats.skipped_dispatches += 1;
                let why = format!("compute: {why}");
                if super::trace() {
                    eprintln!("gpu: dispatch skipped: {why}");
                }
                let reasons = &mut self.stats.skip_reasons;
                if reasons.len() < 64 || reasons.contains_key(&why) {
                    *reasons.entry(why).or_default() += 1;
                }
            }
            Ok(()) => self.stats.dispatches += 1,
        }
        ok.is_ok()
    }

    fn try_dispatch(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], qmd: &super::compute::Qmd, va: u64) -> Result<(), String> {
        self.guest = mem;
        self.split();
        let mut cbufs = [None; CB_SLOTS];
        cbufs[..super::compute::QMD_CBUFS].copy_from_slice(&qmd.cbufs);
        let cs = self.shader(gpu, mem, regs, va, Stage::Compute, cbufs, [Kind::Float; 32], [Kind::Float; 8], Some(qmd))?;
        if !cs.shader.unsupported.is_empty() && crate::env_var!("KIRIKAE_DRAW_PARTIAL").is_none() {
            return Err(format!("shader {:#x} uses {:?}", cs.va, cs.shader.unsupported));
        }
        let mut bindings = Vec::new();
        Self::shader_bindings(&cs, VERTEX_BASE, &mut bindings);
        if cs.shader.probe {
            bindings.push((VERTEX_BASE + STORAGE_OFFSET + 16, vk::DescriptorType::STORAGE_BUFFER));
        }
        let (set_layout, layout) = self.layout(&bindings, vk::ShaderStageFlags::COMPUTE);
        let pipeline = match self.compute_pipelines.get(&(cs.module, layout)) {
            Some(&p) => p,
            None => {
                let start = std::time::Instant::now();
                let info = [vk::ComputePipelineCreateInfo::default()
                    .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(cs.module).name(c"main"))
                    .layout(layout)];
                let p = unsafe { self.device.create_compute_pipelines(self.pipeline_cache, &info, None) }.map_err(|(_, e)| format!("compute pipeline: {e:?}"))?[0];
                self.stats.pipelines += 1;
                self.stats.shader_time += start.elapsed();
                self.compute_pipelines.insert((cs.module, layout), p);
                p
            }
        };
        self.begin();
        self.end_rendering();
        let mut descriptors = Vec::new();
        self.shader_descriptors(gpu, mem, regs, cbufs, &cs, VERTEX_BASE, Stage::Compute, &mut descriptors)?;
        let threads = qmd.threads() * qmd.grid.iter().map(|&n| n.max(1) as u64).product::<u64>();
        let probe = if cs.shader.probe {
            let size = (threads * 16).min(16 << 20);
            let offset = self.ring_copy(&vec![0; size as usize], 16).ok_or("ring full")?;
            descriptors.push(Descriptor::Buffer(VERTEX_BASE + STORAGE_OFFSET + 16, vk::DescriptorType::STORAGE_BUFFER, self.ring.buffer, offset, size));
            Some((offset, size))
        } else {
            None
        };
        self.barrier();
        unsafe { self.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
        let mut scratch = std::mem::take(&mut self.scratch);
        self.bind_descriptors(vk::PipelineBindPoint::COMPUTE, set_layout, layout, &descriptors, &mut scratch);
        self.scratch = scratch;
        let [x, y, z] = qmd.grid.map(|n| n.max(1));
        unsafe { self.device.cmd_dispatch(self.cmd, x, y, z) };
        self.barrier();
        if let (Some((offset, size)), Some(dir)) = (probe, crate::env_var!("KIRIKAE_PROBE_DUMP")) {
            self.finish();
            let data = unsafe { std::slice::from_raw_parts(self.ring.ptr.add(offset as usize), size as usize) };
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("probe_{va:x}_{:?}.bin", qmd.grid)), data);
        }
        Ok(())
    }

    pub fn draw(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], cbufs: &[[Option<(u64, u32)>; CB_SLOTS]; 5], draw: Draw) -> bool {
        let start = std::time::Instant::now();
        let ok = self.try_draw(gpu, mem, regs, cbufs, draw);
        self.stats.draw_time += start.elapsed();
        if let Err(why) = &ok {
            self.stats.skipped += 1;
            let reasons = &mut self.stats.skip_reasons;
            if reasons.len() < 64 || reasons.contains_key(why) {
                *reasons.entry(why.clone()).or_default() += 1;
            }
            if super::trace() {
                eprintln!("gpu: draw skipped: {why}");
            }
        } else {
            self.stats.draws += 1;
        }
        ok.is_ok()
    }

    fn writes_nothing(&self, regs: &[u32]) -> bool {
        let fragment = regs[reg::PIPELINE + 16 * 5] & 1 == 1;
        if fragment {
            let ct = regs[reg::CT_SELECT];
            for k in 0..(ct & 0xf).min(8) as usize {
                let rt = (ct >> (4 + 3 * k) & 7) as usize;
                if regs[reg::COLOR_TARGET + rt * 16 + 4] != 0 && regs[0x680 + k] & 0x1111 != 0 {
                    return false;
                }
            }
        }
        if regs[0x4ba] & 1 == 1 || regs[0x4e0] & 1 == 1 {
            return false;
        }
        let program = address(regs, reg::PROGRAM_REGION);
        [1, 5].iter().filter(|&&stage| stage == 1 || fragment).all(|&stage| self.shader_writes.get(&(program + regs[reg::PIPELINE + 16 * stage + 1] as u64)) == Some(&false))
    }

    fn try_draw(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], cbufs: &[[Option<(u64, u32)>; CB_SLOTS]; 5], draw: Draw) -> Result<(), String> {
        if self.skip_noop && self.writes_nothing(regs) {
            self.stats.noop_draws += 1;
            return Ok(());
        }
        self.guest = mem;
        self.split();
        let mut lap = self.clock_start();
        if let Some(size) = draw.index_size
            && !self.pending.is_empty()
            && let Some(cpu) = gpu.translate(address(regs, reg::INDEX_BUFFER) + draw.first as u64 * size as u64)
        {
            self.resolve(cpu, draw.count as u64 * size as u64);
        }
        self.lap(0, &mut lap);
        let ct = regs[reg::CT_SELECT];
        let count = (ct & 0xf).min(8);
        let mut scratch = std::mem::take(&mut self.scratch);
        let mut targets = std::mem::take(&mut scratch.targets);
        let mut sizes = std::mem::take(&mut scratch.sizes);
        targets.clear();
        sizes.clear();
        let mut output_types = [Kind::Float; 8];
        for k in 0..count {
            let rt = (ct >> (4 + 3 * k) & 7) as usize;
            let base = reg::COLOR_TARGET + rt * 16;
            let format = regs[base + 4];
            if format == 0 {
                targets.push(None);
                continue;
            }
            let s = color_surface(regs, base);
            let (key, fmt) = self.surface_image(gpu, mem, &s, true).ok_or_else(|| format!("render target format {format:#x}"))?;
            output_types[k as usize] = fmt.kind;
            if let (Some(pitch), Some(cpu)) = (s.pitch, gpu.translate(s.va)) {
                self.pitch_targets.insert(key, (cpu, pitch, s.height));
            }
            targets.push(Some((key, fmt)));
            sizes.push((s.width, s.height));
        }
        while targets.last().is_some_and(Option::is_none) {
            targets.pop();
        }
        let depth = self.depth_target(gpu, regs);
        self.lap(1, &mut lap);
        if targets.is_empty() && depth.is_none() {
            return Err("no color or depth target".into());
        }
        if depth.is_some() {
            sizes.push((regs[0x48a], regs[0x48b] & 0x1ffff));
        }
        let extent = sizes
            .iter()
            .copied()
            .chain(targets.iter().flatten().map(|(k, _)| (k.width, k.height)))
            .chain(depth.iter().map(|k| (k.width, k.height)))
            .fold((u32::MAX, u32::MAX), |e, (w, h)| (e.0.min(w), e.1.min(h)));

        let mut attribute_types = [Kind::Float; 32];
        let mut attr_formats = [None; 32];
        for (i, slot) in attr_formats.iter_mut().enumerate() {
            let a = regs[reg::VERTEX_ATTRIBUTE + i];
            if let Some((f, kind)) = formats::vertex_attribute(a >> 21 & 0x3f, a >> 27 & 7) {
                attribute_types[i] = kind;
                *slot = Some((f, a & 0x1f, a >> 7 & 0x3fff, a >> 6 & 1 == 1));
            }
        }

        let program = address(regs, reg::PROGRAM_REGION);
        let shader_va = |stage: usize| program + regs[reg::PIPELINE + 16 * stage + 1] as u64;
        let vs = self.shader(gpu, mem, regs, shader_va(1), Stage::Vertex, cbufs[0], attribute_types, output_types, None)?;
        let fs = if regs[reg::PIPELINE + 16 * 5] & 1 == 1 { Some(self.shader(gpu, mem, regs, shader_va(5), Stage::Fragment, cbufs[4], attribute_types, output_types, None)?) } else { None };
        self.lap(2, &mut lap);
        for s in std::iter::once(&vs).chain(fs.as_ref()) {
            if !s.shader.unsupported.is_empty() && crate::env_var!("KIRIKAE_DRAW_PARTIAL").is_none() {
                return Err(format!("shader {:#x} uses {:?}", s.va, s.shader.unsupported));
            }
            if let Some(spec) = crate::env_var!("KIRIKAE_SKIP_SHADER")
                && spec.to_string_lossy().split(',').any(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16) == Ok(s.va))
            {
                return Err(format!("shader {:#x} skipped (KIRIKAE_SKIP_SHADER)", s.va));
            }
        }

        let pair = (vs.module, fs.as_ref().map_or(vk::ShaderModule::null(), |f| f.module));
        let (set_layout, layout) = match self.pair_layouts.get(&pair) {
            Some(&l) => l,
            None => {
                let mut bindings: Vec<(u32, vk::DescriptorType)> = Vec::new();
                Self::shader_bindings(&vs, VERTEX_BASE, &mut bindings);
                if let Some(fs) = &fs {
                    Self::shader_bindings(fs, FRAGMENT_BASE, &mut bindings);
                }
                let l = self.layout(&bindings, vk::ShaderStageFlags::ALL_GRAPHICS);
                self.pair_layouts.insert(pair, l);
                l
            }
        };
        self.lap(3, &mut lap);

        let mut key = std::mem::take(&mut self.scratch_key);
        let (vk_attributes, vk_bindings) = (&mut key.attributes, &mut key.bindings);
        vk_attributes.clear();
        vk_bindings.clear();
        let mut streams = std::mem::take(&mut scratch.streams);
        streams.clear();
        for i in 0..32u32 {
            if vs.shader.inputs >> i & 1 == 0 {
                continue;
            }
            let Some((format, stream, offset, inactive)) = attr_formats[i as usize] else {
                return Err(format!("vertex attribute {i} format {:#x}", regs[reg::VERTEX_ATTRIBUTE + i as usize]));
            };
            let (binding, offset) = if inactive { (31, 0) } else { (stream, offset) };
            vk_attributes.push((i, binding, format, offset));
            if !vk_bindings.iter().any(|b| b.0 == binding) {
                let stride = if inactive || instanced_stream(regs, stream) { 0 } else { regs[reg::VERTEX_STREAM + 4 * stream as usize] & 0xfff };
                vk_bindings.push((binding, stride));
                streams.push(binding);
            }
        }
        vk_bindings.sort_unstable();

        let per_target = regs[0x4b9] & 1 == 1;
        key.blend.clear();
        key.blend.extend((0..targets.len()).map(|k| {
                let enable = regs[0x4d8 + k] & 1;
                let written = fs.as_ref().map_or(0, |f| f.shader.color_outputs >> (4 * k) & 0xf);
                let written = (written & 1) | (written >> 1 & 1) << 4 | (written >> 2 & 1) << 8 | (written >> 3 & 1) << 12;
                let mask = if targets[k].is_some() { regs[0x680 + k] & written } else { 0 };
                let r = if per_target {
                    let b = 0x780 + 8 * k;
                    [regs[b], regs[b + 1], regs[b + 2], regs[b + 3], regs[b + 4], regs[b + 5], regs[b + 6]]
                } else {
                    [regs[0x4cf], regs[0x4d0], regs[0x4d1], regs[0x4d2], regs[0x4d3], regs[0x4d4], regs[0x4d6]]
                };
                let (aop, asrc, adst) = if per_target || r[0] & 1 == 1 { (r[4], r[5], r[6]) } else { (r[1], r[2], r[3]) };
                [enable, r[1], r[2], r[3], aop, asrc, adst, mask]
            }));

        let quads = draw.topology == 7;
        let topology = formats::topology(draw.topology);
        key.vertex = vs.module;
        key.fragment = fs.as_ref().map_or(vk::ShaderModule::null(), |f| f.module);
        key.layout = layout;
        key.topology = topology;
        key.colors.clear();
        key.colors.extend(targets.iter().map(|t| t.map_or(vk::Format::UNDEFINED, |(_, f)| f.vk)));
        key.depth = depth.map_or(vk::Format::UNDEFINED, |k| k.format);
        key.depth_clamp = self.depth_clamp && regs[reg::VIEW_VOLUME_CLIP_CTRL] & 1 << 11 == 0;
        key.alpha_test = match alpha_test_func(regs) {
            7 => [7, 0],
            f => [f, regs[reg::ALPHA_TEST_REF]],
        };
        let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let recurring = self.pair_drawn.get(&pair).is_some_and(|&f| frame.saturating_sub(f) <= 2);
        let lasting = !recurring || targets.iter().flatten().map(|&(k, _)| k).chain(depth).any(|k| !self.images[&k].transient(frame));
        let pipeline = self.pipeline(&key, lasting)?;
        self.lap(4, &mut lap);
        if super::trace() {
            let f = |r: usize| f32::from_bits(regs[r]);
            let state = format!(
                "gpu: draw state: origin {:#x} cull {:#x} face {:#x} front {:#x} zt select {:#x} depth test {} write {} func {:#x} stencil {} ops {:#x?} two sided {} zt {:#x} at {:#x} range {}..{} clear {} blend {:x?} single_ct_write {:#x} viewport scale ({}, {}, {}) offset ({}, {}, {}) scissor {:#x} {:#x} {:#x} attrs {:x?} bindings {:?} depth image {:x?} size {}x{} clip ctrl {:#x} alpha test {:?} depth bias {:?} units {} factor {} clamp {} base vertex {:#x} instance {:#x}",
                regs[0x4eb],
                regs[0x646],
                regs[0x648],
                regs[0x647],
                regs[0x54e],
                regs[0x4b3] & 1,
                regs[0x4ba] & 1,
                regs[0x4c3],
                regs[0x4e0] & 1,
                &regs[0x4e1..0x4e8],
                regs[0x565] & 1,
                regs[0x3fa],
                address(regs, 0x3f8),
                f(reg::VIEWPORT_CLIP + 2),
                f(reg::VIEWPORT_CLIP + 3),
                f(0x364),
                key.blend,
                regs[0x3e4],
                f(reg::VIEWPORT_SCALE),
                f(reg::VIEWPORT_SCALE + 1),
                f(reg::VIEWPORT_SCALE + 2),
                f(reg::VIEWPORT_SCALE + 3),
                f(reg::VIEWPORT_SCALE + 4),
                f(reg::VIEWPORT_SCALE + 5),
                regs[reg::SCISSOR],
                regs[reg::SCISSOR + 1],
                regs[reg::SCISSOR + 2],
                key.attributes,
                key.bindings,
                depth.map(|k| (k.cpu, k.format, k.width, k.height)),
                regs[0x48a],
                regs[0x48b],
                regs[reg::VIEW_VOLUME_CLIP_CTRL],
                (regs[reg::ALPHA_TEST_ENABLE], regs[reg::ALPHA_TEST_FUNC], f(reg::ALPHA_TEST_REF)),
                &regs[0x370..0x373],
                f(0x55b),
                f(0x56f),
                f(0x61f),
                regs[0x50d],
                regs[0x50e]
            );
            if state != self.trace_state {
                eprintln!("{state}");
                self.trace_state = state;
            }
        }
        self.scratch_key = key;

        let mut views = std::mem::take(&mut scratch.views);
        views.clear();
        for t in &targets {
            views.push(match *t {
                Some((k, f)) => (self.view(k, f.vk, IDENTITY), f.vk),
                None => (vk::ImageView::null(), vk::Format::UNDEFINED),
            });
        }
        let depth_view = depth.map(|k| (self.view(k, k.format, IDENTITY), k.format));
        self.begin();
        let mut descriptors = std::mem::take(&mut scratch.descriptors);
        descriptors.clear();
        self.draw_targets.clear();
        self.draw_targets.extend(targets.iter().flatten().map(|&(k, _)| k).chain(depth));
        self.feedback_used.clear();
        let described = self.descriptors(gpu, mem, regs, cbufs, &vs, fs.as_deref(), &mut descriptors);
        self.draw_targets.clear();
        described?;
        self.lap(5, &mut lap);
        let index = self.index_buffer(gpu, mem, regs, draw, quads)?;
        let base = if draw.index_size.is_some() { regs[reg::VB_ELEMENT_BASE] as i32 } else { 0 };
        let range = if vs.vertex_index {
            None
        } else {
            self.vertex_range(gpu, mem, regs, draw).map(|(lo, hi)| ((lo as i64 + base as i64).max(0) as u32, (hi as i64 + base as i64).max(0) as u32))
        };
        let mut vertex_buffers = std::mem::take(&mut scratch.vertex_buffers);
        vertex_buffers.clear();
        self.vertex_buffers(gpu, mem, regs, &streams, draw, range, &mut vertex_buffers)?;
        self.lap(6, &mut lap);
        self.lap(7, &mut lap);
        let lowest = range.map_or(0, |r| r.0);
        if !self.rendering.as_ref().is_some_and(|r| r.views == views && r.depth == depth_view && r.extent == extent) {
            let rt = RenderTargets { views: views.clone(), depth: depth_view, extent };
            self.end_rendering();
            let attachments: Vec<vk::RenderingAttachmentInfo> = views
                .iter()
                .map(|&(v, _)| vk::RenderingAttachmentInfo::default().image_view(v).image_layout(vk::ImageLayout::GENERAL).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE))
                .collect();
            let depth_attachment = depth_view.map(|(v, _)| {
                vk::RenderingAttachmentInfo::default().image_view(v).image_layout(vk::ImageLayout::GENERAL).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE)
            });
            let mut info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: extent.0, height: extent.1 } })
                .layer_count(1)
                .color_attachments(&attachments);
            if let Some(d) = depth_attachment.as_ref() {
                info = info.depth_attachment(d);
                if aspect_of(depth_view.unwrap().1).contains(vk::ImageAspectFlags::STENCIL) {
                    info = info.stencil_attachment(d);
                }
            }
            unsafe { self.device.cmd_begin_rendering(self.cmd, &info) };
            self.rendering = Some(rt);
        }
        let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        for k in targets.iter().flatten().map(|&(k, _)| k).chain(depth) {
            self.touch(k);
            self.images.get_mut(&k).unwrap().drawn = Some(frame);
        }
        self.pair_drawn.insert(pair, frame);
        for k in std::mem::take(&mut self.feedback_used) {
            let stamp = self.images[&k].stamp;
            self.feedbacks.insert(k, stamp);
        }

        let cmd = self.cmd;
        let d = &self.device;
        let b = &mut self.bound;
        unsafe {
            if changed(&mut b.pipeline, pipeline) {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            }
            let sx = f32::from_bits(regs[reg::VIEWPORT_SCALE]);
            let sy = f32::from_bits(regs[reg::VIEWPORT_SCALE + 1]);
            let ox = f32::from_bits(regs[reg::VIEWPORT_SCALE + 3]);
            let oy = f32::from_bits(regs[reg::VIEWPORT_SCALE + 4]);
            let viewport = vk::Viewport { x: ox - sx, y: oy - sy, width: 2.0 * sx, height: 2.0 * sy, min_depth: 0.0, max_depth: 1.0 };
            let viewport = if viewport.width == 0.0 {
                vk::Viewport { x: 0.0, y: 0.0, width: extent.0 as f32, height: extent.1 as f32, min_depth: 0.0, max_depth: 1.0 }
            } else {
                viewport
            };
            if changed(&mut b.viewport, [viewport.x, viewport.y, viewport.width, viewport.height].map(f32::to_bits)) {
                d.cmd_set_viewport(cmd, 0, &[viewport]);
            }
            let scissor = if regs[reg::SCISSOR] & 1 == 1 {
                let (h, v) = (regs[reg::SCISSOR + 1], regs[reg::SCISSOR + 2]);
                let (x0, x1) = ((h & 0xffff).min(extent.0), (h >> 16).min(extent.0));
                let (y0, y1) = ((v & 0xffff).min(extent.1), (v >> 16).min(extent.1));
                vk::Rect2D { offset: vk::Offset2D { x: x0 as i32, y: y0 as i32 }, extent: vk::Extent2D { width: x1.saturating_sub(x0), height: y1.saturating_sub(y0) } }
            } else {
                vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: extent.0, height: extent.1 } }
            };
            if changed(&mut b.scissor, (scissor.offset.x, scissor.offset.y, scissor.extent.width, scissor.extent.height)) {
                d.cmd_set_scissor(cmd, 0, &[scissor]);
            }
            let cull = if regs[0x646] & 1 == 1 && crate::env_var!("KIRIKAE_NO_CULL").is_none() {
                match regs[0x648] {
                    0x404 => vk::CullModeFlags::FRONT,
                    0x408 => vk::CullModeFlags::FRONT_AND_BACK,
                    _ => vk::CullModeFlags::BACK,
                }
            } else {
                vk::CullModeFlags::NONE
            };
            if changed(&mut b.cull, cull) {
                d.cmd_set_cull_mode(cmd, cull);
            }
            let ccw = regs[0x647] != 0x900;
            let flip = regs[0x4eb] >> 4 & 1 == 1;
            let front = if ccw != flip { vk::FrontFace::COUNTER_CLOCKWISE } else { vk::FrontFace::CLOCKWISE };
            if changed(&mut b.front, front) {
                d.cmd_set_front_face(cmd, front);
            }
            let constants = [regs[0x4c7], regs[0x4c8], regs[0x4c9], regs[0x4ca]];
            if changed(&mut b.blend, constants) {
                d.cmd_set_blend_constants(cmd, &constants.map(f32::from_bits));
            }
            let has_depth = depth.is_some() && crate::env_var!("KIRIKAE_NO_DEPTH_TEST").is_none();
            let (test, write, op) = (has_depth && regs[0x4b3] & 1 == 1, has_depth && regs[0x4ba] & 1 == 1, formats::compare_op(regs[0x4c3]));
            let was = b.depth;
            if changed(&mut b.depth, (test, write, op)) {
                if was.is_none_or(|w| w.0 != test) {
                    d.cmd_set_depth_test_enable(cmd, test);
                }
                if was.is_none_or(|w| w.1 != write) {
                    d.cmd_set_depth_write_enable(cmd, write);
                }
                if was.is_none_or(|w| w.2 != op) {
                    d.cmd_set_depth_compare_op(cmd, op);
                }
            }
            let has_stencil = depth.is_some_and(|k| aspect_of(k.format).contains(vk::ImageAspectFlags::STENCIL));
            let stencil_test = has_stencil && regs[0x4e0] & 1 == 1;
            if changed(&mut b.stencil_test, stencil_test) {
                d.cmd_set_stencil_test_enable(cmd, stencil_test);
            }
            let front = [regs[0x4e1], regs[0x4e2], regs[0x4e3], regs[0x4e4], regs[0x4e5], regs[0x4e6], regs[0x4e7]];
            let back = if regs[0x565] & 1 == 1 { [regs[0x566], regs[0x567], regs[0x568], regs[0x569], regs[0x3d5], regs[0x3d7], regs[0x3d6]] } else { front };
            if stencil_test && changed(&mut b.stencil, [front, back]) {
                for (face, s) in [(vk::StencilFaceFlags::FRONT, front), (vk::StencilFaceFlags::BACK, back)] {
                    d.cmd_set_stencil_op(cmd, face, formats::stencil_op(s[0]), formats::stencil_op(s[2]), formats::stencil_op(s[1]), formats::compare_op(s[3]));
                    d.cmd_set_stencil_reference(cmd, face, s[4] & 0xff);
                    d.cmd_set_stencil_compare_mask(cmd, face, s[5] & 0xff);
                    d.cmd_set_stencil_write_mask(cmd, face, s[6] & 0xff);
                }
            }
            let kind = match topology {
                vk::PrimitiveTopology::POINT_LIST => 0,
                vk::PrimitiveTopology::LINE_LIST | vk::PrimitiveTopology::LINE_STRIP => 1,
                _ => 2,
            };
            let values = [regs[0x55b], regs[0x56f], regs[0x61f]];
            let bias = has_depth && regs[0x370 + kind] & 1 == 1 && values[..2].iter().any(|&v| f32::from_bits(v) != 0.0);
            let was = b.depth_bias;
            if changed(&mut b.depth_bias, (bias, if bias { values } else { [0; 3] })) {
                if was.is_none_or(|w| w.0 != bias) {
                    d.cmd_set_depth_bias_enable(cmd, bias);
                }
                if bias {
                    let [units, factor, clamp] = values.map(f32::from_bits);
                    d.cmd_set_depth_bias(cmd, units / 2.0, if self.depth_bias_clamp { clamp } else { 0.0 }, factor);
                }
            }
            for &(binding, buffer, offset) in &vertex_buffers {
                if changed(&mut b.vertex[binding as usize], (buffer, offset)) {
                    d.cmd_bind_vertex_buffers(cmd, binding, &[buffer], &[offset]);
                }
            }
        }
        self.bind_descriptors(vk::PipelineBindPoint::GRAPHICS, set_layout, layout, &descriptors, &mut scratch);
        unsafe {
            let d = &self.device;
            match index {
                Some((buffer, offset, ty, count)) => {
                    if changed(&mut self.bound.index, (buffer, offset, ty)) {
                        d.cmd_bind_index_buffer(cmd, buffer, offset, ty);
                    }
                    d.cmd_draw_indexed(cmd, count, 1, 0, base.wrapping_sub(lowest as i32), draw.instance);
                }
                None => d.cmd_draw(cmd, draw.count, 1, draw.first - lowest, draw.instance),
            }
        }
        scratch.targets = targets;
        scratch.sizes = sizes;
        scratch.streams = streams;
        scratch.views = views;
        scratch.descriptors = descriptors;
        scratch.vertex_buffers = vertex_buffers;
        self.scratch = scratch;
        self.lap(8, &mut lap);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn shader(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        regs: &[u32],
        va: u64,
        stage: Stage,
        cbufs: [Option<(u64, u32)>; CB_SLOTS],
        attribute_types: [Kind; 32],
        output_types: [Kind; 8],
        compute: Option<&super::compute::Qmd>,
    ) -> Result<Arc<ShaderModule>, String> {
        let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let head = match self.heads.get(&va) {
            Some(&(f, h)) if f == frame => h,
            _ => {
                let h = hash64(gpu.bytes(mem, va, 0x200).ok_or("shader not mapped")?);
                self.heads.insert(va, (frame, h));
                h
            }
        };
        let types = attribute_types.iter().chain(output_types.iter()).fold(0u64, |h, k| h.wrapping_mul(3).wrapping_add(*k as u64));
        let types = compute.map_or(types, |q| q.block.iter().chain([&q.local]).fold(types, |h, &n| h.wrapping_mul(0x1_0001).wrapping_add(n as u64)));
        let depth = (stage == Stage::Vertex).then(|| depth_transform(regs)).flatten();
        let types = depth.map_or(types, |(s, o)| types.wrapping_mul(0x1_0000_0001).wrapping_add((s.to_bits() as u64) << 32 | o.to_bits() as u64));
        let alpha_test = stage == Stage::Fragment && alpha_test_func(regs) != 7;
        let types = types.wrapping_mul(2).wrapping_add(alpha_test as u64);
        let depth_unorm16 = matches!(stage, Stage::Vertex | Stage::Fragment) && regs[0x54e] & 1 == 1 && regs[0x3fa] == 0x13;
        let types = types.wrapping_mul(2).wrapping_add(depth_unorm16 as u64);
        let key = ShaderKey { va, hash: head, stage: stage as u8, types };
        if let Some(s) = self.shaders.get(&key) {
            return Ok(s.clone());
        }
        let mut size = 0x10000u64;
        let code = loop {
            if let Some(c) = gpu.read(mem, va, size) {
                break c;
            }
            size /= 2;
            if size < 0x200 {
                return Err("shader not mapped".into());
            }
        };
        let code = if compute.is_some() { [vec![0; kirikae_maxwell::sph::SIZE], code].concat() } else { code };
        let base = if stage == Stage::Fragment { FRAGMENT_BASE } else { VERTEX_BASE };
        let opts = Options {
            texture_binding: base + TEXTURE_OFFSET,
            storage_binding: base + STORAGE_OFFSET,
            attribute_types,
            output_types,
            debug_color: crate::env_var!("KIRIKAE_DEBUG_SOLID").map(|_| [1.0, 0.0, 1.0, 1.0]).filter(|_| stage == Stage::Fragment),
            debug_probe: probe(va),
            debug_probe_signed: crate::env_var!("KIRIKAE_PROBE_SIGNED").is_some(),
            local_size: compute.map_or([1; 3], |q| q.block),
            local_memory: compute.map_or(0, |q| q.local),
            shared_memory: compute.map_or(0, |q| q.shared),
            image_binding: base + IMAGE_OFFSET,
            depth_transform: depth,
            depth_unorm16,
            alpha_test,
            exact_texel_rounding: crate::env_var!("KIRIKAE_NO_TEXEL_ROUNDING").is_some(),
            ..Options::new(stage, base)
        };
        struct Env<'a> {
            gpu: &'a Gpu,
            mem: &'a GuestMemory,
            cbufs: [Option<(u64, u32)>; CB_SLOTS],
        }
        impl kirikae_maxwell::Env for Env<'_> {
            fn cbuf(&self, slot: u32, offset: u32) -> Option<u32> {
                let (addr, size) = self.cbufs.get(slot as usize).copied().flatten()?;
                if offset + 4 > size {
                    return None;
                }
                let b = self.gpu.read(self.mem, addr + offset as u64, 4)?;
                Some(u32::from_le_bytes(b.try_into().unwrap()))
            }
        }
        let env = Env { gpu, mem, cbufs };
        let start = std::time::Instant::now();
        let shader = kirikae_maxwell::translate(&code, &opts, &env).map_err(|e| format!("shader at {va:#x}: {e:?}"))?;
        self.stats.shader_time += start.elapsed();
        crate::usage::shader(va, &shader.mnemonics, &shader.unsupported);
        if super::trace() {
            eprintln!(
                "gpu: shader {va:#x} ({stage:?}): {} words of SPIR-V, cbufs {:#x}, {} textures, inputs {:#x}, outputs {:#x}, missing {:?}",
                shader.spirv.len(),
                shader.cbufs,
                shader.textures.len(),
                shader.inputs,
                shader.outputs,
                shader.unsupported
            );
            for (pc, targets) in &shader.brx {
                eprintln!("gpu:   brx at {pc:#x} -> {targets:x?}");
            }
        }
        if let Some(dir) = crate::env_var!("KIRIKAE_SPIRV_DUMP") {
            let bytes: Vec<u8> = shader.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{va:010x}_{stage:?}.spv")), bytes);
        }
        let module = unsafe { self.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&shader.spirv), None) }.map_err(|e| format!("shader module: {e:?}"))?;
        let vertex_index = reads_vertex_index(&shader.spirv);
        let writes = shader.storage_written != 0 || !shader.images.is_empty();
        *self.shader_writes.entry(va).or_default() |= writes;
        let s = Arc::new(ShaderModule { module, shader, va, vertex_index });
        self.shaders.insert(key, s.clone());
        Ok(s)
    }

    fn layout(&mut self, bindings: &[(u32, vk::DescriptorType)], stages: vk::ShaderStageFlags) -> (vk::DescriptorSetLayout, vk::PipelineLayout) {
        let key = (bindings.to_vec(), stages.as_raw());
        if let Some(&l) = self.layouts.get(&key) {
            return l;
        }
        let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|&(b, ty)| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(ty).descriptor_count(1).stage_flags(stages))
            .collect();
        let flags = if self.push.is_some() { vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR } else { vk::DescriptorSetLayoutCreateFlags::empty() };
        let set = unsafe { self.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(flags).bindings(&vk_bindings), None).unwrap() };
        let sets = [set];
        let layout = unsafe { self.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets), None).unwrap() };
        self.layouts.insert(key, (set, layout));
        (set, layout)
    }

    fn pipeline(&mut self, key: &PipelineKey, now: bool) -> Result<vk::Pipeline, String> {
        if let Some(&p) = self.pipelines.get(key) {
            return Ok(p);
        }
        let Some(c) = self.compiler.as_mut() else {
            let pipeline = build_pipeline(&self.device, self.pipeline_cache, key).map_err(|e| format!("pipeline: {e:?}"))?;
            self.pipelines.insert(key.clone(), pipeline);
            self.stats.pipelines += 1;
            return Ok(pipeline);
        };
        for (k, r) in c.finished() {
            match r {
                Ok(p) if self.pipelines.contains_key(&k) => unsafe { self.device.destroy_pipeline(p, None) },
                Ok(p) => {
                    self.pipelines.insert(k, p);
                    self.stats.pipelines += 1;
                }
                Err(e) => {
                    c.failed.insert(k, format!("pipeline: {e:?}"));
                }
            }
        }
        if let Some(&p) = self.pipelines.get(key) {
            return Ok(p);
        }
        if let Some(why) = c.failed.get(key) {
            return Err(why.clone());
        }
        if now {
            let pipeline = build_pipeline(&self.device, self.pipeline_cache, key).map_err(|e| format!("pipeline: {e:?}"))?;
            self.pipelines.insert(key.clone(), pipeline);
            self.stats.pipelines += 1;
            return Ok(pipeline);
        }
        c.queue(key);
        Err("pipeline compiling".into())
    }

    #[allow(clippy::too_many_arguments)]
    fn descriptors(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        regs: &[u32],
        cbufs: &[[Option<(u64, u32)>; CB_SLOTS]; 5],
        vs: &ShaderModule,
        fs: Option<&ShaderModule>,
        out: &mut Vec<Descriptor>,
    ) -> Result<(), String> {
        self.shader_descriptors(gpu, mem, regs, cbufs[0], vs, VERTEX_BASE, Stage::Vertex, out)?;
        if let Some(fs) = fs {
            self.shader_descriptors(gpu, mem, regs, cbufs[4], fs, FRAGMENT_BASE, Stage::Fragment, out)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn shader_descriptors(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        regs: &[u32],
        cbufs: [Option<(u64, u32)>; CB_SLOTS],
        s: &ShaderModule,
        base: u32,
        stage: Stage,
        out: &mut Vec<Descriptor>,
    ) -> Result<(), String> {
        let ubo_align = self.limits.min_uniform_buffer_offset_alignment.max(16);
        let ssbo_align = self.limits.min_storage_buffer_offset_alignment.max(16);
        let mut lap = self.clock_start();
        for (slot, cbuf) in cbufs.iter().enumerate() {
            if s.shader.cbufs >> slot & 1 == 0 {
                continue;
            }
            let (addr, size) = cbuf.unwrap_or((0, 0));
            let used = s.shader.cbuf_used[slot] as u64;
            let size = (size as u64).min(used.next_multiple_of(16)).clamp(16, MAX_CBUF);
            self.stats.cbuf_bytes += size;
            if let Some(spec) = crate::env_var!("KIRIKAE_CBUF_DUMP") {
                let spec = spec.to_string_lossy().into_owned();
                let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
                let at = crate::env_var!("KIRIKAE_CBUF_DUMP_FRAME").and_then(|f| f.to_str()?.parse::<u64>().ok());
                if let Some((dir, va)) = spec.rsplit_once(':')
                    && va.split('+').any(|v| u64::from_str_radix(v, 16) == Ok(s.va))
                    && at.is_none_or(|f| f == frame)
                {
                    static DUMPED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let n = if slot as u32 == s.shader.cbufs.trailing_zeros() { DUMPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) } else { DUMPED.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(1) };
                    let data = gpu.read(mem, addr, size).unwrap_or_else(|| vec![0; size as usize]);
                    if super::trace() {
                        eprintln!("gpu: cbuf dump {:#x} #{n} slot {slot}", s.va);
                    }
                    let _ = std::fs::write(std::path::Path::new(dir).join(format!("{:x}_f{frame}_{n:06}_c{slot}.bin", s.va)), &data);
                }
            }
            let (buffer, offset) = self.guest_buffer(gpu, mem, addr, size, ubo_align, buffer_cache::Kind::Constant)?;
            out.push(Descriptor::Buffer(base + slot as u32, vk::DescriptorType::UNIFORM_BUFFER, buffer, offset, size));
        }
        self.lap(9, &mut lap);
        for slot in 0..16u32 {
            if s.shader.storage >> slot & 1 == 0 {
                continue;
            }
            let (c0, c0_size) = cbufs[0].unwrap_or((0, 0));
            let Some(&off) = s.shader.storage_offsets.get(slot as usize) else { continue };
            let desc = if off + 16 <= c0_size { gpu.bytes(mem, c0 + off as u64, 16) } else { None };
            let (addr, size) = desc.map_or((0, 16), |d| {
                (u64::from_le_bytes(d[..8].try_into().unwrap()), u32::from_le_bytes(d[8..12].try_into().unwrap()) as u64)
            });
            if super::trace() {
                eprintln!("gpu: storage buffer {slot} ({stage:?}): {addr:#x} size {size:#x}");
            }
            let size = size.clamp(16, MAX_STORAGE).next_multiple_of(4);
            let written = s.shader.storage_written >> slot & 1 == 1;
            let cpu = gpu.translate(addr);
            let offset = match cpu.and_then(|c| self.storage.get(&(c, size)).copied()) {
                Some(o) => o,
                None => {
                    let o = self.ring_guest(gpu, mem, addr, size, ssbo_align, true)?;
                    if let Some(c) = cpu {
                        self.storage.insert((c, size), o);
                    }
                    o
                }
            };
            if written && stage != Stage::Compute {
                self.draw_wrote_storage = true;
            }
            if written
                && let Some(c) = cpu
                && !self.write_back.contains(&(c, offset, size))
            {
                self.write_back.push((c, offset, size));
                self.pending.push((c, size, self.serial, offset));
                self.pending_pages.add((c, size, self.serial, offset));
                self.pending_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            out.push(Descriptor::Buffer(base + STORAGE_OFFSET + slot, vk::DescriptorType::STORAGE_BUFFER, self.ring.buffer, offset, size));
        }
        self.lap(10, &mut lap);
        for &t in &s.shader.textures {
            let (view, sampler) = match self.texture(gpu, mem, regs, cbufs, t.slot, t.handle, t.dim, t.array, t.shadow) {
                Ok(v) => v,
                Err(why) => {
                    self.note_placeholder(&why);
                    let view_type = match (t.dim, t.array) {
                        (TexDim::Cube, false) => vk::ImageViewType::CUBE,
                        (TexDim::Cube, true) => vk::ImageViewType::CUBE_ARRAY,
                        (_, true) => vk::ImageViewType::TYPE_2D_ARRAY,
                        _ => vk::ImageViewType::TYPE_2D,
                    };
                    if t.dim == TexDim::D1 || t.dim == TexDim::D3 || t.shadow {
                        return Err(format!("texture: {why}"));
                    }
                    (self.placeholder(view_type)?, self.sampler(&tic::Sampler::default()))
                }
            };
            out.push(Descriptor::Image(t.binding, view));
            out.push(Descriptor::Sampler(t.binding + 1, sampler));
        }
        for &i in &s.shader.images {
            let view = match self.storage_image(gpu, mem, regs, cbufs, &i) {
                Ok(v) => v,
                Err(why) => {
                    self.note_placeholder(&why);
                    self.placeholder(vk::ImageViewType::TYPE_2D)?
                }
            };
            out.push(Descriptor::StorageImage(i.binding, view));
        }
        self.lap(11, &mut lap);
        Ok(())
    }

    fn shader_bindings(s: &ShaderModule, base: u32, bindings: &mut Vec<(u32, vk::DescriptorType)>) {
        for slot in 0..CB_SLOTS as u32 {
            if s.shader.cbufs >> slot & 1 == 1 {
                bindings.push((base + slot, vk::DescriptorType::UNIFORM_BUFFER));
            }
        }
        for slot in 0..16 {
            if s.shader.storage >> slot & 1 == 1 {
                bindings.push((base + STORAGE_OFFSET + slot, vk::DescriptorType::STORAGE_BUFFER));
            }
        }
        for t in &s.shader.textures {
            bindings.push((t.binding, vk::DescriptorType::SAMPLED_IMAGE));
            bindings.push((t.binding + 1, vk::DescriptorType::SAMPLER));
        }
        for i in &s.shader.images {
            bindings.push((i.binding, vk::DescriptorType::STORAGE_IMAGE));
        }
    }

    fn bind_descriptors(&mut self, point: vk::PipelineBindPoint, set_layout: vk::DescriptorSetLayout, layout: vk::PipelineLayout, descriptors: &[Descriptor], scratch: &mut DrawScratch) {
        if descriptors.is_empty() {
            return;
        }
        if point == vk::PipelineBindPoint::GRAPHICS && self.push.is_some() {
            if self.bound.descriptors.as_ref().is_some_and(|(l, d)| *l == layout && d.as_slice() == descriptors) {
                self.stats.pushes_reused += 1;
                return;
            }
            let last = self.bound.descriptors.get_or_insert_with(|| (layout, Vec::new()));
            last.0 = layout;
            last.1.clear();
            last.1.extend_from_slice(descriptors);
        }
        let set = if self.push.is_some() {
            vk::DescriptorSet::null()
        } else {
            let layouts = [set_layout];
            match unsafe { self.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.descriptor_pool).set_layouts(&layouts)) } {
                Ok(s) => s[0],
                Err(_) => return,
            }
        };
        let (buffer_infos, image_infos, writes) = (&mut scratch.buffer_infos, &mut scratch.image_infos, &mut scratch.writes);
        buffer_infos.clear();
        image_infos.clear();
        writes.clear();
        for d in descriptors {
            buffer_infos.push(match *d {
                Descriptor::Buffer(_, _, buffer, offset, range) => vk::DescriptorBufferInfo { buffer, offset, range },
                _ => vk::DescriptorBufferInfo::default(),
            });
            image_infos.push(match *d {
                Descriptor::Image(_, view) | Descriptor::StorageImage(_, view) => vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL },
                Descriptor::Sampler(_, s) => vk::DescriptorImageInfo { sampler: s, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED },
                _ => vk::DescriptorImageInfo::default(),
            });
        }
        for (k, d) in descriptors.iter().enumerate() {
            let (binding, ty) = match *d {
                Descriptor::Buffer(b, ty, ..) => (b, ty),
                Descriptor::Image(b, _) => (b, vk::DescriptorType::SAMPLED_IMAGE),
                Descriptor::Sampler(b, _) => (b, vk::DescriptorType::SAMPLER),
                Descriptor::StorageImage(b, _) => (b, vk::DescriptorType::STORAGE_IMAGE),
            };
            let buffer = matches!(d, Descriptor::Buffer(..));
            writes.push(vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: binding,
                descriptor_count: 1,
                descriptor_type: ty,
                p_buffer_info: if buffer { &buffer_infos[k] } else { std::ptr::null() },
                p_image_info: if buffer { std::ptr::null() } else { &image_infos[k] },
                ..Default::default()
            });
        }
        unsafe {
            match &self.push {
                Some(p) => p.cmd_push_descriptor_set(self.cmd, point, layout, 0, writes),
                None => {
                    self.device.update_descriptor_sets(writes, &[]);
                    self.device.cmd_bind_descriptor_sets(self.cmd, point, layout, 0, &[set], &[]);
                }
            }
        }
        writes.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn texture(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], cbufs: [Option<(u64, u32)>; CB_SLOTS], slot: Option<u32>, handle: u32, dim: TexDim, array: bool, shadow: bool) -> Result<(vk::ImageView, vk::Sampler), String> {
        let mut lap = self.clock_start();
        let slot = slot.unwrap_or(regs[reg::BINDLESS_TEXTURE] & 0x1f);
        let (cb, cb_size) = cbufs[slot as usize].ok_or_else(|| format!("texture constant buffer {slot} not bound"))?;
        if handle * 4 + 4 > cb_size {
            return Err("texture handle outside its constant buffer".into());
        }
        let word = gpu.bytes(mem, cb + handle as u64 * 4, 4).ok_or("texture handle unreadable")?;
        let word = u32::from_le_bytes(word.try_into().unwrap());
        let (tic_index, tsc_index) = (word & 0xfffff, word >> 20);
        let tic_pool = address(regs, reg::TEX_HEADER_POOL);
        let tsc_pool = address(regs, reg::TEX_SAMPLER_POOL);
        let tic = tic::words(gpu.bytes(mem, tic_pool + tic_index as u64 * 32, 32).ok_or("texture header unreadable")?).unwrap();
        let tsc = tic::words(gpu.bytes(mem, tsc_pool + tsc_index as u64 * 32, 32).ok_or("sampler unreadable")?).unwrap();
        let header = tic::texture_header(&tic);
        let view_type = match (dim, array, header.texture_type) {
            (TexDim::D2, false, 1 | 3 | 5 | 7 | 8) => vk::ImageViewType::TYPE_2D,
            (TexDim::D2, true, 1 | 5 | 7) => vk::ImageViewType::TYPE_2D_ARRAY,
            (TexDim::Cube, false, 3 | 8) => vk::ImageViewType::CUBE,
            (TexDim::Cube, true, 3 | 8) => vk::ImageViewType::CUBE_ARRAY,
            (TexDim::D3, false, 2) => vk::ImageViewType::TYPE_3D,
            _ => return Err(format!("texture type {} (shader {dim:?}{})", header.texture_type, if array { " array" } else { "" })),
        };
        let layers = match header.texture_type {
            5 => header.depth.max(1),
            3 => 6,
            8 => header.depth.max(1) * 6,
            _ => 1,
        };
        let mut sampler_state = tic::sampler(&tsc);
        if !shadow {
            sampler_state.compare = None;
        }
        if sampler_state.compare.is_none() && crate::env_var!("KIRIKAE_WHITE_TEXTURES").is_some() {
            let key = ImageKey { cpu: u64::MAX, format: vk::Format::R8G8B8A8_UNORM, width: 1, height: 1 };
            if !self.images.contains_key(&key) {
                self.create_image(key, 1, 6).map_err(|e| format!("image: {e:?}"))?;
                self.upload_layers(key, 1, &vec![(1, 1, vec![255; 4]); 6]);
            }
            let view = self.view_of(key, key.format, [vk::ComponentSwizzle::IDENTITY; 4], view_type);
            return Ok((view, self.sampler(&sampler_state)));
        }
        if let Some(spec) = crate::env_var!("KIRIKAE_TEX_SOLID")
            && let Some(byte) = spec.to_string_lossy().split(',').find_map(|e| {
                let (a, v) = e.split_once(':')?;
                (u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()? == header.address).then(|| v.parse::<u8>().ok()).flatten()
            })
            && sampler_state.compare.is_none()
        {
            let key = ImageKey { cpu: u64::MAX - 8 - byte as u64, format: vk::Format::R8G8B8A8_UNORM, width: 1, height: 1 };
            if !self.images.contains_key(&key) {
                self.create_image(key, 1, 6).map_err(|e| format!("image: {e:?}"))?;
                self.upload_layers(key, 1, &vec![(1, 1, vec![byte; 4]); 6]);
            }
            let view = self.view_of(key, key.format, [vk::ComponentSwizzle::IDENTITY; 4], view_type);
            return Ok((view, self.sampler(&sampler_state)));
        }
        let format = formats::texture(header.components, header.data_type, header.srgb).ok_or_else(|| format!("texture format {:#x} type {}", header.components, header.data_type))?;
        let cpu = gpu.translate(header.address).ok_or("texture not mapped")?;
        if super::trace() {
            eprintln!("gpu: texture {handle} -> tic {tic_index:#x}: {:#x} {}x{}x{} type {} levels {} view {}..={} {:?} swizzle {:?} types {:?} mip {} filter {}{} reduction {} wrap {:?} lod {}..{} bias {}{}", header.address, header.width, header.height, header.depth, header.texture_type, header.levels, header.min_level, header.max_level, format.vk, header.swizzle, header.data_types, sampler_state.mip, if sampler_state.mag_linear { "L" } else { "N" }, if sampler_state.min_linear { "L" } else { "N" }, sampler_state.reduction, sampler_state.address, sampler_state.min_lod as f32 / 256.0, sampler_state.max_lod as f32 / 256.0, sampler_state.lod_bias as f32 / 256.0, sampler_state.compare.map_or(String::new(), |c| format!(" compare {c}")));
        }
        if view_type == vk::ImageViewType::TYPE_3D {
            let view = self.volume(gpu, mem, cpu, &header, &format)?;
            return Ok((view, self.sampler(&sampler_state)));
        }
        let tex_key = TexKey { tic, tsc, dim, array, shadow };
        if let Some(e) = self.tex_cache.get(&tex_key).copied().filter(|e| !self.draw_targets.contains(&e.image)) {
            if let Some(k) = e.refresh {
                self.refresh_texture(gpu, mem, k, &header, &format, layers);
            }
            if let Some(like) = e.gather {
                self.gather(gpu, mem, cpu, &header, &format, like, layers)?;
            }
            if let Some(src) = e.crop {
                self.crop(src, header.width, header.height)?;
            }
            if let Some((src, format)) = e.twin {
                self.depth_twin(src, format)?;
            }
            self.lap(12, &mut lap);
            return Ok((e.view, e.sampler));
        }
        let mut entry = TexEntry { image: ImageKey { cpu: 0, format: vk::Format::UNDEFINED, width: 0, height: 0 }, view: vk::ImageView::null(), sampler: vk::Sampler::null(), refresh: None, gather: None, crop: None, twin: None };
        self.lap(12, &mut lap);
        let layered = self.layered_image(gpu, mem, cpu, &header, &format)?;
        self.lap(13, &mut lap);
        let key = match layered.map(|(k, _)| k).or_else(|| self.find_image(cpu, header.width, header.height, format.vk)) {
            Some(k) => {
                if layered.is_none() {
                    self.refresh_texture(gpu, mem, k, &header, &format, layers);
                    entry.refresh = Some(k);
                }
                k
            }
            None => {
                let key = ImageKey { cpu, format: format.vk, width: header.width, height: header.height };
                let levels = if header.texture_type == 7 { 1 } else { header.levels };
                self.create_image(key, levels, layers).map_err(|e| format!("image: {e:?}"))?;
                self.upload_texture(gpu, mem, key, &header, &format, levels, layers);
                let hash = texture_hash(gpu, mem, &header, &format, levels, layers).unwrap_or(0);
                let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
                self.images.get_mut(&key).unwrap().source = Some(Source { hash, checked: frame, volatile: false });
                key
            }
        };
        self.lap(14, &mut lap);
        let swizzle = header.swizzle.map(|s| match s {
            0 => vk::ComponentSwizzle::ZERO,
            2 => vk::ComponentSwizzle::R,
            3 => vk::ComponentSwizzle::G,
            4 => vk::ComponentSwizzle::B,
            5 => vk::ComponentSwizzle::A,
            _ => vk::ComponentSwizzle::ONE,
        });
        let view_format = if same_class(key.format, format.vk) { format.vk } else { key.format };
        let swizzle = if aspect_of(view_format) == vk::ImageAspectFlags::COLOR {
            swizzle
        } else {
            swizzle.map(|c| match c {
                vk::ComponentSwizzle::G | vk::ComponentSwizzle::B => vk::ComponentSwizzle::R,
                c => c,
            })
        };
        let mut sub = self.clock_start();
        let key = if self.images[&key].layers < layers.min(6) && view_type != vk::ImageViewType::TYPE_2D {
            entry.gather = Some(key.format);
            entry.refresh = None;
            self.gather(gpu, mem, cpu, &header, &format, key.format, layers)?
        } else {
            key
        };
        self.lap(16, &mut sub);
        let img = &self.images[&key];
        let flat = view_type == vk::ImageViewType::TYPE_2D || (view_type == vk::ImageViewType::TYPE_2D_ARRAY && layers == 1);
        let key = if (key.width, key.height) != (header.width, header.height) && img.levels == 1 && img.layers == 1 && flat {
            entry.crop = Some(key);
            self.crop(key, header.width, header.height)?
        } else {
            key
        };
        self.lap(17, &mut sub);
        let levels = if header.texture_type == 7 { (0, 1) } else { (header.min_level, header.max_level.saturating_sub(header.min_level) + 1) };
        let (key, view_format, swizzle, levels) = match depth_twin_format(key.format).filter(|_| sampler_state.compare.is_some()) {
            Some(format) => {
                entry.twin = Some((key, format));
                let twin = self.depth_twin(key, format)?;
                let swizzle = swizzle.map(|c| match c {
                    vk::ComponentSwizzle::G | vk::ComponentSwizzle::B => vk::ComponentSwizzle::R,
                    c => c,
                });
                (twin, format, swizzle, (0, 1))
            }
            None => (key, view_format, swizzle, levels),
        };
        if self.draw_targets.contains(&key) {
            let copy = self.feedback(key)?;
            let view = self.view_range(copy, view_format, swizzle, view_type, levels, layered.map_or(0, |(_, layer)| layer));
            return Ok((view, self.sampler(&sampler_state)));
        }
        let view = self.view_range(key, view_format, swizzle, view_type, levels, layered.map_or(0, |(_, layer)| layer));
        let sampler = self.sampler(&sampler_state);
        if self.tex_cache.len() > 16384 || self.tex_at.len() > 16384 {
            self.tex_cache.clear();
            self.tex_at.clear();
        }
        self.tex_cache.insert(tex_key, TexEntry { image: key, view, sampler, ..entry });
        let at = self.tex_at.entry(cpu).or_default();
        if !at.contains(&tex_key) {
            at.push(tex_key);
        }
        self.lap(18, &mut sub);
        self.lap(15, &mut lap);
        Ok((view, sampler))
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_texture(&mut self, gpu: &Gpu, mem: &GuestMemory, key: ImageKey, header: &TextureHeader, format: &TexFormat, levels: u32, layers: u32) {
        if !self.pending.is_empty()
            && let (Some(cpu), Some(size)) = (gpu.translate(header.address), layer_size(header, format, levels))
        {
            self.resolve_submitted(cpu, size * layers as u64);
        }
        let mut data = read_texture(gpu, mem, header, format, levels, layers);
        if format.astc {
            for (w, h, level) in &mut data {
                *level = super::astc::decode(level, *w, *h, format.block_w, format.block_h);
            }
        }
        if let Some(spec) = crate::env_var!("KIRIKAE_TEX_DUMP")
            && let Some((dir, cpu)) = spec.to_string_lossy().rsplit_once(':')
            && cpu.split('+').any(|c| u64::from_str_radix(c, 16).is_ok_and(|a| a == key.cpu || a == header.address))
        {
            let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
            let bytes: Vec<u8> = data.iter().flat_map(|(_, _, l)| l.iter().copied()).collect();
            let name = format!("{:x}_{:?}_{}x{}_f{frame}.bin", header.address, key.format, key.width, key.height);
            let _ = std::fs::write(std::path::Path::new(dir).join(name), bytes);
        }
        self.upload_layers(key, levels as usize, &data);
    }

    fn refresh_texture(&mut self, gpu: &Gpu, mem: &GuestMemory, key: ImageKey, header: &TextureHeader, format: &TexFormat, layers: u32) {
        let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let img = &self.images[&key];
        let Some(src) = img.source.filter(|_| !img.gpu_dirty) else { return };
        let due = if src.volatile { src.checked != frame } else { frame >= src.checked + SOURCE_RECHECK };
        if !due {
            return;
        }
        let levels = img.levels;
        let hash = texture_hash(gpu, mem, header, format, levels, layers).unwrap_or(src.hash);
        let changed = hash != src.hash;
        if changed {
            self.upload_texture(gpu, mem, key, header, format, levels, layers);
            self.stats.refreshes += 1;
        }
        self.images.get_mut(&key).unwrap().source = Some(Source { hash, checked: frame, volatile: changed || (src.volatile && frame < src.checked + SOURCE_RECHECK) });
    }

    fn crop(&mut self, src: ImageKey, width: u32, height: u32) -> Result<ImageKey, String> {
        self.stats.crop_calls += 1;
        let key = ImageKey { cpu: src.cpu | CROP, format: src.format, width, height };
        let stamp = self.images[&src].stamp;
        if self.crops.get(&key) == Some(&(src, stamp)) && self.images.contains_key(&key) {
            return Ok(key);
        }
        if !self.images.contains_key(&key) {
            let mut others: Vec<(u64, ImageKey)> = self
                .crops
                .iter()
                .filter(|(k, (s, _))| *s == src && **k != key && !self.retired.contains(k))
                .map(|(k, _)| (self.images.get(k).map_or(0, |i| i.stamp), *k))
                .collect();
            if others.len() >= CROP_SIZES {
                others.sort_unstable_by_key(|o| o.0);
                for &(_, k) in &others[..others.len() + 1 - CROP_SIZES] {
                    self.retired.push(k);
                }
            }
            self.create_image(key, 1, 1).map_err(|e| format!("image: {e:?}"))?;
            self.stats.crop_images += 1;
        }
        self.stats.crop_copies += 1;
        self.begin();
        self.end_rendering();
        let sub = vk::ImageSubresourceLayers { aspect_mask: aspect_of(src.format), mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let region = [vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub).extent(vk::Extent3D { width, height, depth: 1 })];
        let (si, di) = (self.images[&src].image, self.images[&key].image);
        self.barrier();
        unsafe { self.device.cmd_copy_image(self.cmd, si, vk::ImageLayout::GENERAL, di, vk::ImageLayout::GENERAL, &region) };
        self.barrier();
        self.clock += 1;
        self.images.get_mut(&key).unwrap().stamp = self.clock;
        self.crops.insert(key, (src, stamp));
        Ok(key)
    }

    fn feedback(&mut self, src: ImageKey) -> Result<ImageKey, String> {
        let key = ImageKey { cpu: src.cpu | FEEDBACK, ..src };
        if !self.feedback_used.contains(&src) {
            self.feedback_used.push(src);
        }
        let (stamp, levels, layers) = { let i = &self.images[&src]; (i.stamp, i.levels, i.layers) };
        if self.feedbacks.get(&src) == Some(&stamp) && self.images.contains_key(&key) {
            return Ok(key);
        }
        if self.images.get(&key).is_some_and(|i| i.levels != levels || i.layers != layers) {
            self.destroy_image(key);
        }
        if !self.images.contains_key(&key) {
            self.create_image(key, levels, layers).map_err(|e| format!("image: {e:?}"))?;
        }
        self.stats.crop_copies += 1;
        self.begin();
        self.end_rendering();
        let aspect = aspect_of(src.format);
        let regions: Vec<vk::ImageCopy> = (0..levels)
            .map(|m| {
                let sub = vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: m, base_array_layer: 0, layer_count: layers };
                vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub).extent(vk::Extent3D { width: (src.width >> m).max(1), height: (src.height >> m).max(1), depth: 1 })
            })
            .collect();
        let (si, di) = (self.images[&src].image, self.images[&key].image);
        self.barrier();
        unsafe { self.device.cmd_copy_image(self.cmd, si, vk::ImageLayout::GENERAL, di, vk::ImageLayout::GENERAL, &regions) };
        self.barrier();
        self.clock += 1;
        self.images.get_mut(&key).unwrap().stamp = self.clock;
        self.feedbacks.insert(src, stamp);
        Ok(key)
    }

    fn depth_twin(&mut self, src: ImageKey, format: vk::Format) -> Result<ImageKey, String> {
        let key = ImageKey { cpu: src.cpu | TWIN, format, width: src.width, height: src.height };
        let (stamp, layers) = (self.images[&src].stamp, self.images[&src].layers);
        if self.twins.get(&key) == Some(&(src, stamp)) && self.images.contains_key(&key) {
            return Ok(key);
        }
        if self.images.get(&key).is_some_and(|i| i.layers != layers) {
            self.destroy_image(key);
        }
        if !self.images.contains_key(&key) {
            self.create_image(key, 1, layers).map_err(|e| format!("image: {e:?}"))?;
        }
        let bytes = bytes_per_pixel(format).ok_or("depth copy format")? as u64;
        let size = src.width as u64 * src.height as u64 * layers as u64 * bytes;
        self.begin();
        self.end_rendering();
        let offset = self.ring_alloc(size, 16).ok_or("depth copy too large")?.0;
        let region = |aspect_mask| {
            [vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask, mip_level: 0, base_array_layer: 0, layer_count: layers })
                .image_extent(vk::Extent3D { width: src.width, height: src.height, depth: 1 })]
        };
        let (si, di, buffer) = (self.images[&src].image, self.images[&key].image, self.ring.buffer);
        self.barrier();
        unsafe { self.device.cmd_copy_image_to_buffer(self.cmd, si, vk::ImageLayout::GENERAL, buffer, &region(vk::ImageAspectFlags::COLOR)) };
        self.barrier();
        unsafe { self.device.cmd_copy_buffer_to_image(self.cmd, buffer, di, vk::ImageLayout::GENERAL, &region(vk::ImageAspectFlags::DEPTH)) };
        self.barrier();
        self.clock += 1;
        self.images.get_mut(&key).unwrap().stamp = self.clock;
        self.twins.insert(key, (src, stamp));
        Ok(key)
    }

    fn volume(&mut self, gpu: &Gpu, mem: &GuestMemory, cpu: u64, header: &TextureHeader, format: &TexFormat) -> Result<vk::ImageView, String> {
        let Layout::BlockLinear { block_height_log2, block_depth_log2 } = header.layout else {
            return Err("3D texture not block linear".into());
        };
        let (w, h, d) = (header.width.max(1), header.height.max(1), header.depth.max(1));
        let (bw, bh) = (w.div_ceil(format.block_w), h.div_ceil(format.block_h));
        let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let key = (cpu, format.vk);
        let due = self.volumes.get(&key).is_none_or(|v| frame >= v.checked + SOURCE_RECHECK);
        if due {
            let size = volume_size(bw, bh, d, format.block_bytes, block_height_log2, block_depth_log2);
            let raw = gpu.read(mem, header.address, size).ok_or("3D texture unreadable")?;
            let hash = hash64(&raw);
            if self.volumes.get(&key).map(|v| v.hash) != Some(hash) {
                if !self.volumes.contains_key(&key) {
                    let v = self.create_volume(format.vk, w, h, d).map_err(|e| format!("3D image: {e:?}"))?;
                    self.volumes.insert(key, v);
                }
                let mut data = deswizzle_3d(&raw, bw, bh, d, format.block_bytes, block_height_log2, block_depth_log2);
                if format.astc {
                    let slice = (bw * bh * format.block_bytes) as usize;
                    data = data.chunks(slice).flat_map(|s| super::astc::decode(s, w, h, format.block_w, format.block_h)).collect();
                }
                let image = self.volumes[&key].image;
                self.begin();
                self.end_rendering();
                let offset = self.ring_copy(&data, 16).ok_or("3D texture too large for the upload ring")?;
                let region = [vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: w, height: h, depth: d })];
                self.barrier();
                unsafe { self.device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, image, vk::ImageLayout::GENERAL, &region) };
                self.barrier();
                self.stats.uploads += 1;
                self.volumes.get_mut(&key).unwrap().slices.clear();
            }
            let v = self.volumes.get_mut(&key).unwrap();
            v.hash = hash;
            v.checked = frame;
        }
        if format.block_w == 1 && format.block_h == 1 {
            let bh = fit_block(block_height_log2, (h as u64).div_ceil(8));
            let bd = fit_block(block_depth_log2, d as u64);
            let gobs_wide = (w as u64 * format.block_bytes as u64).div_ceil(64);
            let blocks_high = (h as u64).div_ceil(8).div_ceil(1 << bh);
            let block_size = 512u64 << (bh + bd);
            let sources: Vec<Option<ImageKey>> = (0..d as u64)
                .map(|z| {
                    let at = cpu + (z >> bd) * blocks_high * gobs_wide * block_size + ((z & ((1 << bd) - 1)) << bh) * 512;
                    self.at(at).iter().copied().filter(|k| k.format == format.vk && k.width >= w && k.height >= h && self.images[k].gpu_dirty).max_by_key(|k| self.images[k].stamp)
                })
                .collect();
            if sources.iter().any(Option::is_some) {
                let stamps: Vec<u64> = sources.iter().map(|s| s.map_or(0, |k| self.images[&k].stamp)).collect();
                if self.volumes[&key].slices != stamps {
                    self.begin();
                    self.end_rendering();
                    self.barrier();
                    let dst = self.volumes[&key].image;
                    let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
                    for (z, src) in sources.iter().enumerate() {
                        let Some(src) = src else { continue };
                        let region = [vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub).dst_offset(vk::Offset3D { x: 0, y: 0, z: z as i32 }).extent(vk::Extent3D { width: w, height: h, depth: 1 })];
                        let si = self.images[src].image;
                        unsafe { self.device.cmd_copy_image(self.cmd, si, vk::ImageLayout::GENERAL, dst, vk::ImageLayout::GENERAL, &region) };
                    }
                    self.barrier();
                    self.volumes.get_mut(&key).unwrap().slices = stamps;
                }
            }
        }
        let swizzle = header.swizzle.map(|s| match s {
            0 => vk::ComponentSwizzle::ZERO,
            2 => vk::ComponentSwizzle::R,
            3 => vk::ComponentSwizzle::G,
            4 => vk::ComponentSwizzle::B,
            5 => vk::ComponentSwizzle::A,
            _ => vk::ComponentSwizzle::ONE,
        });
        let v = self.volumes.get_mut(&key).unwrap();
        if let Some(&view) = v.views.get(&swizzle) {
            return Ok(view);
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(v.image)
            .view_type(vk::ImageViewType::TYPE_3D)
            .format(format.vk)
            .components(vk::ComponentMapping { r: swizzle[0], g: swizzle[1], b: swizzle[2], a: swizzle[3] })
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
        let view = unsafe { self.device.create_image_view(&info, None).map_err(|e| format!("3D view: {e:?}"))? };
        v.views.insert(swizzle, view);
        Ok(view)
    }

    fn create_volume(&mut self, format: vk::Format, width: u32, height: u32, depth: u32) -> Result<Volume, vk::Result> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&info, None)? };
        let req = unsafe { self.device.get_image_memory_requirements(image) };
        let ty = find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| find_memory(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::empty()))
            .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
        let memory = unsafe { self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None)? };
        unsafe { self.device.bind_image_memory(image, memory, 0)? };
        self.begin();
        self.end_rendering();
        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        unsafe {
            self.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &barrier);
        }
        Ok(Volume { image, memory, views: HashMap::default(), hash: 0, checked: 0, slices: Vec::new() })
    }

    #[allow(clippy::too_many_arguments)]
    fn gather(&mut self, gpu: &Gpu, mem: &GuestMemory, cpu: u64, header: &TextureHeader, format: &TexFormat, like: vk::Format, layers: u32) -> Result<ImageKey, String> {
        let levels = if header.texture_type == 7 { 1 } else { header.levels.max(1) };
        let stride = layer_size(header, format, levels).ok_or("layered texture is not block linear")?;
        let key = ImageKey { cpu: cpu | GATHER, format: like, width: header.width, height: header.height };
        if !self.images.contains_key(&key) {
            self.create_image(key, levels, layers).map_err(|e| format!("image: {e:?}"))?;
            let data = read_texture(gpu, mem, header, format, levels, layers);
            self.upload_layers(key, levels as usize, &data);
            self.gathers.insert(key, vec![0; (layers * levels) as usize]);
        }
        if let (Some((generation, sources)), Some(copied)) = (self.gather_sources.get(&key), self.gathers.get(&key))
            && *generation == self.generation
            && sources.len() == copied.len()
            && sources.iter().zip(copied).all(|(s, &c)| s.map_or(0, |k| self.images[&k].stamp) == c)
        {
            return Ok(key);
        }
        let sources = match self.gather_sources.get(&key) {
            Some((generation, sources)) if *generation == self.generation => sources.clone(),
            _ => {
                let mut sources: Vec<Option<ImageKey>> = Vec::new();
                for l in 0..layers as u64 {
                    for m in 0..levels {
                        let (w, h) = ((header.width >> m).max(1), (header.height >> m).max(1));
                        let at = cpu + l * stride + level_offset(header, format, m);
                        sources.push(self.find_image(at, w, h, like).filter(|k| k.format == like && self.images[k].layers == 1 && k.width >= w && k.height >= h));
                    }
                }
                if super::trace() {
                    eprintln!("gpu: gather {cpu:#x} {}x{}x{layers} {like:?} stride {stride:#x} sources {sources:x?}", header.width, header.height);
                }
                self.gather_sources.insert(key, (self.generation, sources.clone()));
                sources
            }
        };
        let stamps: Vec<u64> = sources.iter().map(|s| s.map_or(0, |k| self.images[&k].stamp)).collect();
        let copied = self.gathers.get(&key).cloned().unwrap_or_default();
        if copied == stamps {
            return Ok(key);
        }
        self.begin();
        self.end_rendering();
        self.barrier();
        let aspect = aspect_of(like) & !vk::ImageAspectFlags::STENCIL;
        let dst = self.images[&key].image;
        for (i, src) in sources.iter().enumerate() {
            let Some(src) = src else { continue };
            if copied.get(i) == Some(&stamps[i]) {
                continue;
            }
            let (layer, level) = (i as u32 / levels, i as u32 % levels);
            let from = vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 };
            let to = vk::ImageSubresourceLayers { mip_level: level, base_array_layer: layer, ..from };
            let extent = vk::Extent3D { width: (header.width >> level).max(1), height: (header.height >> level).max(1), depth: 1 };
            let region = [vk::ImageCopy::default().src_subresource(from).dst_subresource(to).extent(extent)];
            let si = self.images[src].image;
            unsafe { self.device.cmd_copy_image(self.cmd, si, vk::ImageLayout::GENERAL, dst, vk::ImageLayout::GENERAL, &region) };
        }
        self.barrier();
        self.clock += 1;
        self.images.get_mut(&key).unwrap().stamp = self.clock;
        self.gathers.insert(key, stamps);
        Ok(key)
    }

    fn placeholder(&mut self, view_type: vk::ImageViewType) -> Result<vk::ImageView, String> {
        let key = ImageKey { cpu: u64::MAX - 1, format: vk::Format::R8G8B8A8_UNORM, width: 1, height: 1 };
        if !self.images.contains_key(&key) {
            self.create_image(key, 1, 6).map_err(|e| format!("image: {e:?}"))?;
            self.upload_layers(key, 1, &vec![(1, 1, vec![0; 4]); 6]);
        }
        Ok(self.view_of(key, key.format, IDENTITY, view_type))
    }

    fn note_placeholder(&mut self, why: &str) {
        let why = format!("placeholder for {why}");
        if super::trace() {
            eprintln!("gpu: {why}");
        }
        let reasons = &mut self.stats.placeholders;
        if reasons.len() < 64 || reasons.contains_key(&why) {
            *reasons.entry(why).or_default() += 1;
        }
    }

    fn storage_image(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], cbufs: [Option<(u64, u32)>; CB_SLOTS], image: &kirikae_maxwell::ImageUse) -> Result<vk::ImageView, String> {
        let slot = image.slot.unwrap_or(regs[reg::BINDLESS_TEXTURE] & 0x1f);
        let (cb, cb_size) = cbufs.get(slot as usize).copied().flatten().ok_or_else(|| format!("image constant buffer {slot} not bound"))?;
        if image.handle * 4 + 4 > cb_size {
            return Err("image handle outside its constant buffer".into());
        }
        let word = gpu.read(mem, cb + image.handle as u64 * 4, 4).ok_or("image handle unreadable")?;
        let tic_index = u32::from_le_bytes(word.try_into().unwrap()) & 0xfffff;
        let tic_pool = address(regs, reg::TEX_HEADER_POOL);
        let tic = tic::words(&gpu.read(mem, tic_pool + tic_index as u64 * 32, 32).ok_or("image header unreadable")?).unwrap();
        let header = tic::texture_header(&tic);
        if image.dim != TexDim::D2 || image.array || !matches!(header.texture_type, 1 | 7) {
            return Err(format!("storage image type {} (shader {:?})", header.texture_type, image.dim));
        }
        let format = formats::texture(header.components, header.data_type, header.srgb).ok_or_else(|| format!("image format {:#x} type {}", header.components, header.data_type))?;
        let cpu = gpu.translate(header.address).ok_or("image not mapped")?;
        if super::trace() {
            eprintln!("gpu: storage image {:#x} {}x{}x{} type {} levels {} view {}..={} {:?} {:?}", header.address, header.width, header.height, header.depth, header.texture_type, header.levels, header.min_level, header.max_level, header.layout, format.vk);
        }
        let (key, layer) = match self.layered_image(gpu, mem, cpu, &header, &format)? {
            Some(found) => found,
            None => match self.find_image(cpu, header.width, header.height, format.vk) {
                Some(k) => (k, 0),
                None => {
                    let (key, created) = self.target_image(cpu, format.vk, header.width, header.height).ok_or("image")?;
                    if created {
                        let data = read_texture(gpu, mem, &header, &format, 1, 1);
                        self.upload_layers(key, 1, &data);
                    }
                    (key, 0)
                }
            },
        };
        let view_format = match key.format {
            vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
            f => f,
        };
        let level = if header.texture_type == 7 { 0 } else { header.min_level };
        if level >= self.images[&key].levels {
            return Err(format!("storage image level {level} beyond the image's {}", self.images[&key].levels));
        }
        if super::trace() {
            let img = &self.images[&key];
            eprintln!("gpu:   -> image {:#x} {}x{} ({} levels, {} layers): level {level} layer {layer}", key.cpu, key.width, key.height, img.levels, img.layers);
        }
        self.touch(key);
        Ok(self.view_range(key, view_format, IDENTITY, vk::ImageViewType::TYPE_2D, (level, 1), layer))
    }

    fn layered_image(&mut self, gpu: &Gpu, mem: &GuestMemory, cpu: u64, header: &TextureHeader, format: &TexFormat) -> Result<Option<(ImageKey, u32)>, String> {
        let levels = if header.texture_type == 7 { 1 } else { header.levels };
        let layers = header.depth.max(1);
        if !matches!(header.texture_type, 1 | 7) || (levels == 1 && layers == 1) || format.astc || !matches!(header.layout, Layout::BlockLinear { .. }) {
            return Ok(None);
        }
        if let Some(found) = self.containing_image(cpu, header, format, levels) {
            return Ok(Some(found));
        }
        let key = ImageKey { cpu, format: format.vk, width: header.width, height: header.height };
        if self.images.contains_key(&key) {
            return Ok(None);
        }
        self.create_image(key, levels, layers).map_err(|e| format!("image: {e:?}"))?;
        let data = read_texture(gpu, mem, header, format, levels, layers);
        self.upload_layers(key, levels as usize, &data);
        Ok(Some((key, 0)))
    }

    fn containing_image(&self, cpu: u64, header: &TextureHeader, format: &TexFormat, levels: u32) -> Option<(ImageKey, u32)> {
        let stride = layer_size(header, format, levels)?;
        self.by_cpu
            .range(cpu.saturating_sub(1 << 32)..=cpu)
            .rev()
            .flat_map(|(_, keys)| keys.iter().map(|k| (k, &self.images[k])))
            .filter(|(k, i)| {
                k.cpu & CROP == 0
                    && k.cpu <= cpu
                    && (k.width, k.height) == (header.width, header.height)
                    && same_class(k.format, format.vk)
                    && (i.levels > 1 || i.layers > 1)
                    && i.layers >= header.depth.max(1)
            })
            .find_map(|(k, i)| {
                let d = cpu - k.cpu;
                (d.is_multiple_of(stride) && d / stride < i.layers as u64).then_some((*k, (d / stride) as u32))
            })
    }

    fn sampler(&mut self, s: &tic::Sampler) -> vk::Sampler {
        if let Some(&v) = self.samplers.get(s) {
            return v;
        }
        let address = |m: u32| match m {
            0 => vk::SamplerAddressMode::REPEAT,
            1 => vk::SamplerAddressMode::MIRRORED_REPEAT,
            3 | 6 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            _ => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        };
        let filter = |l: bool| if l { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(filter(s.mag_linear))
            .min_filter(filter(s.min_linear))
            .mipmap_mode(if s.mip == 3 { vk::SamplerMipmapMode::LINEAR } else { vk::SamplerMipmapMode::NEAREST })
            .address_mode_u(address(s.address[0]))
            .address_mode_v(address(s.address[1]))
            .address_mode_w(address(s.address[2]))
            .compare_enable(s.compare.is_some())
            .compare_op(formats::compare_op(s.compare.unwrap_or(7) + 1))
            .mip_lod_bias(s.lod_bias as f32 / 256.0)
            .min_lod(s.min_lod as f32 / 256.0)
            .max_lod(if s.mip == 1 { 0.25 } else { (s.max_lod as f32 / 256.0).max(s.min_lod as f32 / 256.0) })
            .anisotropy_enable(s.max_anisotropy > 1 && self.anisotropy)
            .max_anisotropy(s.max_anisotropy.min(16) as f32)
            .border_color(border_color(&s.border));
        let v = unsafe { self.device.create_sampler(&info, None).unwrap() };
        self.samplers.insert(*s, v);
        v
    }

    #[allow(clippy::too_many_arguments)]
    fn vertex_buffers(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], streams: &[u32], draw: Draw, range: Option<(u32, u32)>, out: &mut Vec<(u32, vk::Buffer, u64)>) -> Result<(), String> {
        for &s in streams {
            if s == 31 {
                let offset = self.ring_copy(&[0u8; 64], 16).ok_or("ring full")?;
                out.push((31, self.ring.buffer, offset));
                continue;
            }
            let base = reg::VERTEX_STREAM + 4 * s as usize;
            if instanced_stream(regs, s) {
                let stride = (regs[base] & 0xfff) as u64;
                let element = (draw.instance / regs[base + 3].max(1)) as u64;
                let va = address(regs, base + 1) + element * stride;
                if let Some(offset) = gpu.translate(va).and_then(|c| self.pending_in_ring(c, stride.max(4), 1)) {
                    out.push((s, self.ring.buffer, offset));
                    continue;
                }
                let offset = self.ring_guest(gpu, mem, va, stride.max(4), 16, false).map_err(|e| format!("instanced vertex stream {s}: {e}"))?;
                out.push((s, self.ring.buffer, offset));
                continue;
            }
            let start = address(regs, base + 1);
            let limit = address(regs, reg::VERTEX_STREAM_LIMIT + 2 * s as usize);
            let size = (limit + 1).saturating_sub(start).min(64 << 20);
            if size == 0 {
                return Err(format!("vertex stream {s} is empty"));
            }
            let (start, size) = match range {
                Some((lo, hi)) => {
                    let stride = (regs[base] & 0xfff) as u64;
                    let from = start + lo as u64 * stride;
                    let end = (start + (hi as u64 + 1) * stride + 64).min(start + size);
                    (from, end.saturating_sub(from).max(16))
                }
                None => (start, size),
            };
            if super::trace() {
                let data = gpu.read(mem, start, size.min(32)).unwrap_or_default();
                let floats: Vec<f32> = data.as_chunks::<4>().0.iter().take(8).map(|b| f32::from_le_bytes(*b)).collect();
                eprintln!("gpu: vertex stream {s} at {start:#x} ({size:#x} bytes): {floats:?}");
            }
            let (buffer, offset) = self.guest_buffer(gpu, mem, start, size, 16, buffer_cache::Kind::Vertex).map_err(|e| format!("vertex stream {s}: {e}"))?;
            self.stats.vertex_bytes += size;
            out.push((s, buffer, offset));
        }
        Ok(())
    }

    fn vertex_range(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], draw: Draw) -> Option<(u32, u32)> {
        if draw.count == 0 {
            return None;
        }
        let Some(size) = draw.index_size else {
            return Some((draw.first, draw.first + draw.count - 1));
        };
        let va = address(regs, reg::INDEX_BUFFER) + draw.first as u64 * size as u64;
        let len = draw.count as u64 * size as u64;
        let restart = (regs[reg::PRIMITIVE_RESTART] & 1 == 1).then_some(regs[reg::PRIMITIVE_RESTART + 1]);
        if let Some(key) = self.index_checked
            && gpu.translate(va).is_some_and(|cpu| (cpu, len) == (key.0, key.1))
        {
            return self.buffers.index_range(key, restart, |b| vertex_range(b, size, restart));
        }
        vertex_range(gpu.bytes(mem, va, len)?, size, restart)
    }

    fn index_buffer(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], draw: Draw, quads: bool) -> Result<Option<(vk::Buffer, u64, vk::IndexType, u32)>, String> {
        self.index_checked = None;
        let triangles = move |indices: Vec<u32>| -> Vec<u8> {
            let indices = if quads { indices.as_chunks::<4>().0.iter().flat_map(|q| [q[0], q[1], q[2], q[0], q[2], q[3]]).collect() } else { indices };
            indices.iter().flat_map(|i| i.to_le_bytes()).collect()
        };
        let count = if quads { draw.count / 4 * 6 } else { draw.count };
        let bytes = match draw.index_size {
            Some(size) => {
                let va = address(regs, reg::INDEX_BUFFER) + draw.first as u64 * size as u64;
                let len = draw.count as u64 * size as u64;
                let source = gpu.bytes(mem, va, len).ok_or("index buffer unreadable")?;
                if super::trace() {
                    eprintln!("gpu: indices {:?}", &source[..source.len().min(16)]);
                }
                let widen = move |b: &[u8]| -> Vec<u8> {
                    triangles(match size {
                        1 => b.iter().map(|&b| b as u32).collect(),
                        2 => b.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b) as u32).collect(),
                        _ => b.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect(),
                    })
                };
                if let Some(cpu) = gpu.translate(va)
                    && self.buffer_cache_on
                {
                    let key = (cpu, len, buffer_cache::Kind::Index(size, quads));
                    self.index_checked = Some(key);
                    if let Some((buffer, offset)) = self.cached(cpu, len, key.2, source, widen) {
                        return Ok(Some((buffer, offset, vk::IndexType::UINT32, count)));
                    }
                    return self.ring_indices(widen(source), count);
                }
                widen(source)
            }
            None if quads => triangles((draw.first..draw.first + draw.count).collect()),
            None => return Ok(None),
        };
        self.ring_indices(bytes, count)
    }

    fn ring_indices(&mut self, bytes: Vec<u8>, count: u32) -> Result<Option<(vk::Buffer, u64, vk::IndexType, u32)>, String> {
        let offset = self.ring_copy(&bytes, 16).ok_or("ring full")?;
        Ok(Some((self.ring.buffer, offset, vk::IndexType::UINT32, count)))
    }

    #[allow(clippy::too_many_arguments)]
    fn guest_buffer(&mut self, gpu: &Gpu, mem: &GuestMemory, va: u64, size: u64, align: u64, kind: buffer_cache::Kind) -> Result<(vk::Buffer, u64), String> {
        let cpu = gpu.translate(va);
        let pending = cpu.is_none_or(|c| self.pending_pages.overlapping(c, size).next().is_some());
        if !pending && let Some(c) = cpu {
            let key = (c, size, kind);
            if crate::env_var!("KIRIKAE_NO_CHECKED").is_none()
                && let Some(e) = self.checked.get_mut(&key)
                && !self.cp_writes.since(c, size, e.0)
            {
                e.0 = self.cp_writes.len();
                return Ok((e.1, e.2));
            }
            if let Some(data) = mem.bytes(c, size) {
                let found = match self.cached(c, size, kind, data, |d| d.to_vec()) {
                    Some(found) => found,
                    None => (self.ring.buffer, self.ring_copy(data, align).ok_or("ring full")?),
                };
                self.checked.insert(key, (self.cp_writes.len(), found.0, found.1));
                return Ok(found);
            }
        }
        if let Some(offset) = cpu.and_then(|c| self.pending_in_ring(c, size, align)) {
            return Ok((self.ring.buffer, offset));
        }
        let offset = self.ring_guest(gpu, mem, va, size, align, kind == buffer_cache::Kind::Constant)?;
        Ok((self.ring.buffer, offset))
    }

    #[allow(clippy::too_many_arguments)]
    fn cached(&mut self, cpu: u64, size: u64, kind: buffer_cache::Kind, data: &[u8], convert: impl FnOnce(&[u8]) -> Vec<u8>) -> Option<(vk::Buffer, u64)> {
        if !self.buffer_cache_on {
            return None;
        }
        let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let mut retired = Vec::new();
        let found = self.buffers.lookup(&self.device, (cpu, size, kind), data, frame, convert, &mut retired);
        self.buffers.evict(frame, &mut retired);
        self.garbage.extend(retired.into_iter().map(Garbage::Block));
        self.stats.cached_bytes = self.buffers.hits;
        self.stats.cache_memory = self.buffers.memory();
        found
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Descriptor {
    Buffer(u32, vk::DescriptorType, vk::Buffer, u64, u64),
    Image(u32, vk::ImageView),
    Sampler(u32, vk::Sampler),
    StorageImage(u32, vk::ImageView),
}

const IDENTITY: [vk::ComponentSwizzle; 4] = [vk::ComponentSwizzle::IDENTITY; 4];

fn aspect_of(f: vk::Format) -> vk::ImageAspectFlags {
    match f {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => vk::ImageAspectFlags::DEPTH,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

fn same_class(a: vk::Format, b: vk::Format) -> bool {
    use vk::Format as F;
    let class = |f: F| match f {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB | F::R8G8B8A8_SNORM | F::R8G8B8A8_UINT | F::R8G8B8A8_SINT | F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => 1,
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGBA_SRGB_BLOCK => 2,
        F::BC2_UNORM_BLOCK | F::BC2_SRGB_BLOCK => 3,
        F::BC3_UNORM_BLOCK | F::BC3_SRGB_BLOCK => 4,
        F::BC7_UNORM_BLOCK | F::BC7_SRGB_BLOCK => 5,
        _ => 100 + f.as_raw() as u32,
    };
    class(a) == class(b)
}

fn depth_twin_format(f: vk::Format) -> Option<vk::Format> {
    match f {
        vk::Format::R16_UNORM => Some(vk::Format::D16_UNORM),
        vk::Format::R32_SFLOAT => Some(vk::Format::D32_SFLOAT),
        _ => None,
    }
}

fn bytes_per_pixel(f: vk::Format) -> Option<u32> {
    use vk::Format as F;
    Some(match f {
        F::R8_UNORM => 1,
        F::R8G8_UNORM | F::R16_SFLOAT | F::R16_UNORM => 2,
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB | F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB | F::A2B10G10R10_UNORM_PACK32 | F::B10G11R11_UFLOAT_PACK32 | F::R32_SFLOAT | F::R16G16_SFLOAT => 4,
        F::R16G16_UNORM | F::R16G16_SNORM | F::R32_UINT | F::R32_SINT => 4,
        F::R16G16B16A16_SFLOAT | F::R32G32_SFLOAT => 8,
        F::R16G16B16A16_UNORM | F::R16G16B16A16_SNORM | F::R16G16B16A16_UINT | F::R16G16B16A16_SINT => 8,
        F::R32G32B32A32_SFLOAT => 16,
        F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT | F::D24_UNORM_S8_UINT | F::X8_D24_UNORM_PACK32 => 4,
        F::D16_UNORM => 2,
        _ => return None,
    })
}

fn decode_pixels(f: vk::Format, bytes: &[u8]) -> Option<Vec<u32>> {
    use vk::Format as F;
    let to8 = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let rgb = |r: f32, g: f32, b: f32| to8(r) << 16 | to8(g) << 8 | to8(b);
    let px: Vec<u32> = match f {
        F::R8_UNORM => bytes.iter().map(|&r| (r as u32) * 0x010101).collect(),
        F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => bytes.as_chunks::<4>().0.iter().map(|p| to8(f32::from_le_bytes(*p)) * 0x010101).collect(),
        F::D24_UNORM_S8_UINT | F::X8_D24_UNORM_PACK32 => bytes.as_chunks::<4>().0.iter().map(|p| (u32::from_le_bytes(*p) >> 16 & 0xff) * 0x010101).collect(),
        F::R8G8_UNORM => bytes.as_chunks::<2>().0.iter().map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8).collect(),
        F::R16_SFLOAT => bytes.as_chunks::<2>().0.iter().map(|p| to8(half_to_f32(u16::from_le_bytes(*p))) * 0x010101).collect(),
        F::R16_UNORM => bytes.as_chunks::<2>().0.iter().map(|p| (u16::from_le_bytes(*p) >> 8) as u32 * 0x010101).collect(),
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB => bytes.as_chunks::<4>().0.iter().map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32).collect(),
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => bytes.as_chunks::<4>().0.iter().map(|p| (p[2] as u32) << 16 | (p[1] as u32) << 8 | p[0] as u32).collect(),
        F::A2B10G10R10_UNORM_PACK32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                let v = u32::from_le_bytes(*p);
                ((v >> 2) & 0xff) << 16 | ((v >> 12) & 0xff) << 8 | ((v >> 22) & 0xff)
            })
            .collect(),
        F::B10G11R11_UFLOAT_PACK32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                let v = u32::from_le_bytes(*p);
                rgb(small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22 & 0x3ff, 5))
            })
            .collect(),
        F::R32_SFLOAT => bytes.as_chunks::<4>().0.iter().map(|p| to8(f32::from_le_bytes(*p)) * 0x010101).collect(),
        F::R16G16_SFLOAT => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| rgb(half_to_f32(u16::from_le_bytes([p[0], p[1]])), half_to_f32(u16::from_le_bytes([p[2], p[3]])), 0.0))
            .collect(),
        F::R16G16B16A16_SFLOAT => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| {
                let c = |k: usize| half_to_f32(u16::from_le_bytes([p[2 * k], p[2 * k + 1]]));
                rgb(c(0), c(1), c(2))
            })
            .collect(),
        F::R32G32_SFLOAT => bytes.as_chunks::<8>().0.iter().map(|p| rgb(f32::from_le_bytes(p[..4].try_into().unwrap()), f32::from_le_bytes(p[4..].try_into().unwrap()), 0.0)).collect(),
        F::R32G32B32A32_SFLOAT => bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|p| {
                let c = |k: usize| f32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap());
                rgb(c(0), c(1), c(2))
            })
            .collect(),
        _ => return None,
    };
    Some(px)
}

fn small_float(v: u32, mantissa: u32) -> f32 {
    let e = v >> mantissa;
    let m = v & ((1 << mantissa) - 1);
    match e {
        0 => m as f32 / (1 << mantissa) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        _ => (1.0 + m as f32 / (1 << mantissa) as f32) * 2f32.powi(e as i32 - 15),
    }
}

fn half_to_f32(h: u16) -> f32 {
    let s = (h >> 15) as u32;
    let e = (h >> 10 & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => {
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s << 31 | (e as u32) << 23 | (m & 0x3ff) << 13
        }
        31 => s << 31 | 0xff << 23 | m << 13,
        _ => s << 31 | (e + 127 - 15) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

fn vertex_range(bytes: &[u8], size: u32, restart: Option<u32>) -> Option<(u32, u32)> {
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    let mut see = |i: u32| {
        if Some(i) != restart {
            lo = lo.min(i);
            hi = hi.max(i);
        }
    };
    match size {
        1 => bytes.iter().for_each(|&b| see(b as u32)),
        2 => bytes.as_chunks::<2>().0.iter().for_each(|b| see(u16::from_le_bytes(*b) as u32)),
        _ => bytes.as_chunks::<4>().0.iter().for_each(|b| see(u32::from_le_bytes(*b))),
    }
    (lo <= hi).then_some((lo, hi))
}

pub fn read_surface(gpu: &Gpu, mem: &GuestMemory, s: &Surface, bpp: u32) -> Option<Vec<u8>> {
    let row = s.width as u64 * bpp as u64;
    match s.pitch {
        Some(pitch) => {
            let raw = gpu.read(mem, s.va, pitch as u64 * s.height as u64)?;
            let mut out = Vec::with_capacity((row * s.height as u64) as usize);
            for y in 0..s.height as u64 {
                let line = &raw[(y * pitch as u64) as usize..];
                let n = line.len().min(row as usize);
                out.extend_from_slice(&line[..n]);
                out.resize(out.len() + row as usize - n, 0);
            }
            Some(out)
        }
        None => {
            let width_bytes = row.div_ceil(64) * 64;
            let gob_rows = 8u64 << s.block_height_log2;
            let size = width_bytes * (s.height as u64).div_ceil(gob_rows) * gob_rows;
            let raw = gpu.read(mem, s.va, size)?;
            Some(deswizzle(&raw, s.width, s.height, bpp, s.block_height_log2))
        }
    }
}

fn depth_transform(regs: &[u32]) -> Option<(f32, f32)> {
    let sz = f32::from_bits(regs[reg::VIEWPORT_SCALE + 2]);
    let oz = f32::from_bits(regs[reg::VIEWPORT_SCALE + 5]);
    (sz != 0.0 && (sz, oz) != (1.0, 0.0)).then_some((sz, oz))
}

fn instanced_stream(regs: &[u32], stream: u32) -> bool {
    stream < 16 && regs[0x620 + stream as usize] & 1 == 1
}

fn probe(va: u64) -> Option<(u32, u8)> {
    let spec = crate::env_var!("KIRIKAE_PROBE")?.to_str()?;
    let parts: Vec<&str> = spec.split(':').collect();
    let reg = parts.last()?.trim_start_matches('r').parse().ok()?;
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok();
    match parts.len() {
        1 => Some((u32::MAX, reg)),
        2 => Some((hex(parts[0])? as u32, reg)),
        _ => (hex(parts[0])? == va).then_some((hex(parts[1])? as u32, reg)),
    }
}

pub fn deswizzle(raw: &[u8], width: u32, height: u32, bpp: u32, block_height_log2: u32) -> Vec<u8> {
    let row = width as u64 * bpp as u64;
    let width_bytes = row.div_ceil(64) * 64;
    let mut out = vec![0u8; (row * height as u64) as usize];
    for y in 0..height as u64 {
        let dst = &mut out[(y * row) as usize..][..row as usize];
        let mut x = 0u64;
        while x < row {
            let o = block_linear_offset(x, y, width_bytes, block_height_log2) as usize;
            let n = (16 - (x % 16)).min(row - x) as usize;
            if let Some(src) = raw.get(o..o + n) {
                dst[x as usize..x as usize + n].copy_from_slice(src);
            }
            x += n as u64;
        }
    }
    out
}

fn fit_block(log2: u32, extent: u64) -> u32 {
    let mut l = log2;
    while l > 0 && (1u64 << (l - 1)) >= extent {
        l -= 1;
    }
    l
}

fn volume_size(width: u32, height: u32, depth: u32, bpp: u32, block_height_log2: u32, block_depth_log2: u32) -> u64 {
    let gobs_wide = (width as u64 * bpp as u64).div_ceil(64);
    let gobs_high = (height as u64).div_ceil(8);
    let bh = fit_block(block_height_log2, gobs_high);
    let bd = fit_block(block_depth_log2, depth as u64);
    let blocks_high = gobs_high.div_ceil(1 << bh);
    let blocks_deep = (depth as u64).div_ceil(1 << bd);
    gobs_wide * blocks_high * blocks_deep * (512 << (bh + bd))
}

fn deswizzle_3d(raw: &[u8], width: u32, height: u32, depth: u32, bpp: u32, block_height_log2: u32, block_depth_log2: u32) -> Vec<u8> {
    let row = width as u64 * bpp as u64;
    let gobs_wide = row.div_ceil(64);
    let gobs_high = (height as u64).div_ceil(8);
    let bh = fit_block(block_height_log2, gobs_high);
    let bd = fit_block(block_depth_log2, depth as u64);
    let blocks_high = gobs_high.div_ceil(1 << bh);
    let block_size = 512u64 << (bh + bd);
    let mut out = vec![0u8; (row * height as u64 * depth as u64) as usize];
    for z in 0..depth as u64 {
        for y in 0..height as u64 {
            let dst = ((z * height as u64 + y) * row) as usize;
            let mut x = 0u64;
            while x < row {
                let block = ((z >> bd) * blocks_high + ((y / 8) >> bh)) * gobs_wide + x / 64;
                let gob = ((z & ((1 << bd) - 1)) << bh) + ((y / 8) & ((1 << bh) - 1));
                let (gx, gy) = (x % 64, y % 8);
                let inner = (gx / 32) * 256 + (gy / 2) * 64 + ((gx % 32) / 16) * 32 + (gy % 2) * 16 + (gx % 16);
                let o = (block * block_size + gob * 512 + inner) as usize;
                let n = (16 - (x % 16)).min(row - x) as usize;
                if let Some(src) = raw.get(o..o + n) {
                    out[dst + x as usize..dst + x as usize + n].copy_from_slice(src);
                }
                x += n as u64;
            }
        }
    }
    out
}

fn read_texture(gpu: &Gpu, mem: &GuestMemory, h: &TextureHeader, f: &TexFormat, levels: u32, layers: u32) -> Vec<(u32, u32, Vec<u8>)> {
    let mut out = Vec::new();
    let mut offset = 0u64;
    for layer in 0..layers {
        if layer > 0 {
            let Some(align) = layer_alignment(h, f) else { break };
            offset = offset.div_ceil(align) * align;
        }
        read_levels(gpu, mem, h, f, levels, &mut offset, &mut out);
    }
    if h.components == 0x15 {
        for (_, _, level) in &mut out {
            for px in level.as_chunks_mut::<2>().0 {
                let v = u16::from_le_bytes(*px);
                *px = (v & 0x07e0 | v >> 11 | v << 11).to_le_bytes();
            }
        }
    }
    out
}

fn layer_alignment(h: &TextureHeader, f: &TexFormat) -> Option<u64> {
    let Layout::BlockLinear { block_height_log2, .. } = h.layout else { return None };
    let gobs_high = (h.height.div_ceil(f.block_h) as u64).div_ceil(8);
    let mut bhl = block_height_log2;
    while bhl > 0 && (1u64 << (bhl - 1)) >= gobs_high {
        bhl -= 1;
    }
    Some(512 << bhl)
}

fn layer_size(h: &TextureHeader, f: &TexFormat, levels: u32) -> Option<u64> {
    let Layout::BlockLinear { .. } = h.layout else { return None };
    let size = level_offset(h, f, levels);
    let align = layer_alignment(h, f)?;
    Some(size.div_ceil(align) * align)
}

fn level_offset(h: &TextureHeader, f: &TexFormat, level: u32) -> u64 {
    let Layout::BlockLinear { block_height_log2, .. } = h.layout else { return 0 };
    let mut size = 0u64;
    for level in 0..level {
        let (bw, bh) = ((h.width >> level).max(1).div_ceil(f.block_w), (h.height >> level).max(1).div_ceil(f.block_h));
        let gobs_high = (bh as u64).div_ceil(8);
        let mut bhl = block_height_log2;
        while bhl > 0 && (1u64 << (bhl - 1)) >= gobs_high {
            bhl -= 1;
        }
        let rows = 8u64 << bhl;
        size += (bw as u64 * f.block_bytes as u64).div_ceil(64) * 64 * (bh as u64).div_ceil(rows) * rows;
    }
    size
}

fn pipeline_cache_file(props: &vk::PhysicalDeviceProperties) -> Option<std::path::PathBuf> {
    if let Some(p) = crate::env_var!("KIRIKAE_PIPELINE_CACHE") {
        return (!p.is_empty()).then(|| p.into());
    }
    let base = crate::env_var!("LOCALAPPDATA")?;
    Some(std::path::Path::new(&base).join("kirikae").join(format!("pipelines-{:04x}-{:04x}.bin", props.vendor_id, props.device_id)))
}

fn remove_stale_saves(file: &std::path::Path) {
    let (Some(dir), Some(stem)) = (file.parent(), file.file_stem().and_then(|s| s.to_str())) else { return };
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let name = e.file_name();
        let Some(pid) = name.to_str().and_then(|n| n.strip_prefix(stem)).and_then(|n| n.strip_prefix('.')).and_then(|n| n.strip_suffix(".tmp")) else { continue };
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|age| age > std::time::Duration::from_secs(600));
        if !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit()) && old {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

fn pipeline_cache_matches(data: &[u8], props: &vk::PhysicalDeviceProperties) -> bool {
    let word = |i: usize| data.get(4 * i..4 * i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    data.len() >= 32
        && word(0).is_some_and(|n| n >= 32)
        && word(1) == Some(vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32)
        && word(2) == Some(props.vendor_id)
        && word(3) == Some(props.device_id)
        && data[16..32] == props.pipeline_cache_uuid
}

fn texture_hash(gpu: &Gpu, mem: &GuestMemory, h: &TextureHeader, f: &TexFormat, levels: u32, layers: u32) -> Option<u64> {
    let first = texture_source(gpu, mem, h, f)?;
    let mut hash = hash64(&first);
    if layers > 1
        && let Some(stride) = layer_size(h, f, levels)
    {
        for layer in 1..layers as u64 {
            let Some(bytes) = gpu.bytes(mem, h.address + layer * stride, first.len() as u64) else { break };
            for gob in bytes.chunks(512).step_by(16) {
                hash = (hash ^ hash64(gob)).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5) ^ layer;
            }
        }
    }
    Some(hash)
}

fn texture_source(gpu: &Gpu, mem: &GuestMemory, h: &TextureHeader, f: &TexFormat) -> Option<Vec<u8>> {
    let (bw, bh) = (h.width.div_ceil(f.block_w) as u64, h.height.div_ceil(f.block_h) as u64);
    let row = bw * f.block_bytes as u64;
    let size = match h.layout {
        Layout::BlockLinear { block_height_log2, .. } => {
            let rows = 8u64 << block_height_log2;
            row.div_ceil(64) * 64 * bh.div_ceil(rows) * rows
        }
        Layout::Pitch { pitch } => pitch as u64 * bh,
        Layout::Buffer => return None,
    };
    gpu.read(mem, h.address, size.min(64 << 20))
}

fn read_levels(gpu: &Gpu, mem: &GuestMemory, h: &TextureHeader, f: &TexFormat, levels: u32, offset: &mut u64, out: &mut Vec<(u32, u32, Vec<u8>)>) {
    for level in 0..levels {
        let w = (h.width >> level).max(1);
        let hh = (h.height >> level).max(1);
        let (bw, bh) = (w.div_ceil(f.block_w), hh.div_ceil(f.block_h));
        let row = bw as u64 * f.block_bytes as u64;
        match h.layout {
            Layout::BlockLinear { block_height_log2, .. } => {
                let gobs_high = (bh as u64).div_ceil(8);
                let mut bhl = block_height_log2;
                while bhl > 0 && (1u64 << (bhl - 1)) >= gobs_high {
                    bhl -= 1;
                }
                let width_bytes = row.div_ceil(64) * 64;
                let rows = 8u64 << bhl;
                let size = width_bytes * (bh as u64).div_ceil(rows) * rows;
                let raw = gpu.read(mem, h.address + *offset, size).unwrap_or_else(|| vec![0; size as usize]);
                out.push((w, hh, deswizzle(&raw, bw, bh, f.block_bytes, bhl)));
                *offset += size;
            }
            Layout::Pitch { pitch } => {
                let raw = gpu.read(mem, h.address, pitch as u64 * bh as u64).unwrap_or_default();
                let mut data = Vec::new();
                for y in 0..bh as u64 {
                    if let Some(r) = raw.get((y * pitch as u64) as usize..((y * pitch as u64) + row) as usize) {
                        data.extend_from_slice(r);
                    }
                }
                data.resize((row * bh as u64) as usize, 0);
                out.push((w, hh, data));
                break;
            }
            Layout::Buffer => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_index_detection() {
        let header = [0x0723_0203, 0x0001_0300, 0, 100, 0];
        let with: Vec<u32> = header.iter().copied().chain([4 << 16 | 71, 50, 11, 42]).collect();
        let without: Vec<u32> = header.iter().copied().chain([4 << 16 | 71, 50, 11, 43]).collect();
        assert!(reads_vertex_index(&with));
        assert!(!reads_vertex_index(&without));
    }

    #[test]
    fn volume_slices() {
        let (w, h, d, bpp) = (32u32, 24u32, 3u32, 4u32);
        let slice = volume_size(w, h, 1, bpp, 1, 0);
        assert_eq!(volume_size(w, h, d, bpp, 1, 0), slice * d as u64);
        let mut raw = vec![0u8; (slice * d as u64) as usize];
        for z in 0..d as u64 {
            for y in 0..h as u64 {
                for x in 0..(w * bpp) as u64 {
                    raw[(z * slice + block_linear_offset(x, y, (w * bpp) as u64, 1)) as usize] = (x + 7 * y + 31 * z) as u8;
                }
            }
        }
        let lin = deswizzle_3d(&raw, w, h, d, bpp, 1, 0);
        let row = (w * bpp) as usize;
        for z in 0..d as usize {
            for y in 0..h as usize {
                for x in 0..row {
                    assert_eq!(lin[(z * h as usize + y) * row + x], (x + 7 * y + 31 * z) as u8);
                }
            }
        }
        let raw: Vec<u8> = (0..1024).map(|i| (i / 512) as u8).collect();
        let lin = deswizzle_3d(&raw, 16, 8, 2, 4, 0, 1);
        assert_eq!((lin[0], lin[16 * 4 * 8]), (0, 1));
    }

    fn header(width: u32, height: u32, levels: u32, block_height_log2: u32) -> TextureHeader {
        TextureHeader {
            components: 8,
            data_type: 2,
            data_types: [2; 4],
            swizzle: [2, 3, 4, 5],
            address: 0,
            layout: Layout::BlockLinear { block_height_log2, block_depth_log2: 0 },
            width,
            height,
            depth: 6,
            texture_type: 1,
            levels,
            min_level: 0,
            max_level: levels - 1,
            srgb: false,
        }
    }

    #[test]
    fn cube_layer_stride() {
        let f = formats::texture(8, 2, false).unwrap();
        assert_eq!(layer_size(&header(64, 64, 7, 3), &f, 7), Some(0x6000));
        assert_eq!(layer_size(&header(64, 64, 1, 3), &f, 1), Some(0x4000));
    }

    #[test]
    fn ring_keeps_data_in_use() {
        let mut ring = Ring { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), ptr: std::ptr::null_mut(), head: 0, tail: None, recorded: 0 };
        let mb = 1 << 20;
        assert_eq!(ring.fit(100 * mb, 16), Some(0));
        assert_eq!(ring.fit(80 * mb, 16), Some(100 * mb));
        assert_eq!(ring.fit(20 * mb, 16), None);
        ring.tail = Some(100 * mb);
        assert_eq!(ring.fit(20 * mb, 16), Some(0));
        assert_eq!(ring.fit(70 * mb, 256), Some(20 * mb));
        assert_eq!(ring.fit(20 * mb, 16), None);
        ring.tail = None;
        assert_eq!(ring.fit(90 * mb, 16), Some(90 * mb));
        assert_eq!(ring.recorded, 360 * mb);
    }

    #[test]
    fn pipeline_cache_header() {
        let props = vk::PhysicalDeviceProperties { vendor_id: 0x1002, device_id: 0x699f, pipeline_cache_uuid: [7; 16], ..Default::default() };
        let mut data = vec![32, 0, 0, 0, 1, 0, 0, 0, 0x02, 0x10, 0, 0, 0x9f, 0x69, 0, 0];
        data.extend_from_slice(&[7; 16]);
        data.extend_from_slice(&[0; 64]);
        assert!(pipeline_cache_matches(&data, &props));
        data[16] = 8;
        assert!(!pipeline_cache_matches(&data, &props));
        assert!(!pipeline_cache_matches(&data[..20], &props));
    }
}

fn border_color(border: &[u32; 4]) -> vk::BorderColor {
    let [r, g, b, a] = border.map(f32::from_bits);
    let rgb = (r + g + b) / 3.0;
    match (rgb >= 0.5, a >= 0.5) {
        (true, _) => vk::BorderColor::FLOAT_OPAQUE_WHITE,
        (false, true) => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        (false, false) => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
    }
}
