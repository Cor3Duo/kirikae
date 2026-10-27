use std::collections::VecDeque;
use std::ffi::CStr;
use std::sync::Arc;

use ash::vk;

use crate::mem::GuestMemory;
use super::buffer_cache::{self, BufferCache};
use super::fxhash::FastMap as HashMap;
use super::tic;

mod buffers;
mod descriptors;
mod draw;
mod guest_writes;
mod images;
mod pipeline;
mod present;
mod readback;
mod submit;
mod textures;
mod tiling;
mod volumes;
mod lines;

pub use images::{Surface, color_surface};
pub use present::Presented;

use draw::{Bound, DrawScratch, RenderTargets, TargetAsk};
use guest_writes::{CpWrites, Pending, PendingPages};
use images::{Image, ImageKey};
use pipeline::{LayoutKey, PipelineCompiler, PipelineKey, ShaderKey, ShaderModule, pipeline_cache_file, pipeline_cache_matches, remove_stale_saves};
use present::Swapchain;
use submit::{Batch, Garbage, IN_FLIGHT, RING_SIZE, Ring};
use textures::{TexEntry, TexKey};
use volumes::Volume;
use lines::{Line, LineKey};

pub struct Renderer {
    entry: ash::Entry,
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
    platforms: Vec<&'static CStr>,
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
    target_memo: [(TargetAsk, ImageKey); 4],
    generation: u64,
    volumes: HashMap<(u64, vk::Format), Volume>,
    lines: HashMap<LineKey, Line>,
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
            for l in self.lines.values() {
                for view in l.views.values() {
                    d.destroy_image_view(*view, None);
                }
                d.destroy_image(l.image, None);
                d.free_memory(l.memory, None);
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
        let candidates: &[&'static CStr] = if cfg!(windows) {
            &[ash::khr::win32_surface::NAME]
        } else {
            &[ash::khr::xlib_surface::NAME, ash::khr::xcb_surface::NAME, ash::khr::wayland_surface::NAME]
        };
        let platforms: Vec<&'static CStr> = candidates.iter().copied().filter(|n| instance_has(n)).collect();
        let surfaces = !crate::display::headless() && instance_has(ash::khr::surface::NAME) && !platforms.is_empty();
        let mut instance_extensions = Vec::new();
        if surfaces {
            instance_extensions.push(ash::khr::surface::NAME.as_ptr());
            instance_extensions.extend(platforms.iter().map(|n| n.as_ptr()));
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
        let platforms = if surfaces { platforms } else { Vec::new() };
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
            entry,
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
            platforms,
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
            lines: HashMap::default(),
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
}
