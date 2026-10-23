use std::sync::Arc;

use ash::vk;
use kirikae_maxwell::{Kind, Options, Shader, Stage};

use crate::gpu::fxhash::FastMap as HashMap;
use crate::gpu::fxhash::FastSet;
use crate::gpu::threed::{CB_SLOTS, reg};
use crate::gpu::{Gpu, formats};
use crate::mem::GuestMemory;
use super::descriptors::{FRAGMENT_BASE, IMAGE_OFFSET, STORAGE_OFFSET, TEXTURE_OFFSET, VERTEX_BASE};
use super::images::aspect_of;
use super::readback::probe;
use super::{Renderer, hash64};

pub(super) const PIPELINE_CACHE_SAVE: std::time::Duration = std::time::Duration::from_secs(30);

pub(super) type LayoutKey = (Vec<(u32, vk::DescriptorType)>, u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ShaderKey {
    pub(super) va: u64,
    pub(super) hash: u64,
    pub(super) stage: u8,
    pub(super) types: u64,
}

pub(super) struct ShaderModule {
    pub(super) module: vk::ShaderModule,
    pub(super) shader: Shader,
    pub(super) va: u64,
    pub(super) vertex_index: bool,
}

pub(super) fn reads_vertex_index(spirv: &[u32]) -> bool {
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

#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub(super) struct PipelineKey {
    pub(super) vertex: vk::ShaderModule,
    pub(super) fragment: vk::ShaderModule,
    pub(super) layout: vk::PipelineLayout,
    pub(super) topology: vk::PrimitiveTopology,
    pub(super) attributes: Vec<(u32, u32, vk::Format, u32)>,
    pub(super) bindings: Vec<(u32, u32)>,
    pub(super) colors: Vec<vk::Format>,
    pub(super) depth: vk::Format,
    pub(super) blend: Vec<[u32; 8]>,
    pub(super) depth_clamp: bool,
    pub(super) alpha_test: [u32; 2],
}

pub(super) fn alpha_test_func(regs: &[u32]) -> u32 {
    if regs[reg::ALPHA_TEST_ENABLE] & 1 == 0 {
        return 7;
    }
    let f = regs[reg::ALPHA_TEST_FUNC];
    (if f >= 0x200 { f - 0x200 } else { f.saturating_sub(1) }) & 7
}

pub(super) fn build_pipeline(device: &ash::Device, cache: vk::PipelineCache, key: &PipelineKey) -> Result<vk::Pipeline, vk::Result> {
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

pub(super) type Built = (PipelineKey, Result<vk::Pipeline, vk::Result>);

pub(super) struct PipelineCompiler {
    pub(super) queue: Option<std::sync::mpsc::Sender<PipelineKey>>,
    pub(super) done: Arc<std::sync::Mutex<Vec<Built>>>,
    pub(super) pending: FastSet<PipelineKey>,
    pub(super) failed: HashMap<PipelineKey, String>,
    pub(super) workers: Vec<std::thread::JoinHandle<()>>,
}

impl PipelineCompiler {
    pub(super) fn new(device: &ash::Device, cache: vk::PipelineCache) -> Option<PipelineCompiler> {
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

    pub(super) fn queue(&mut self, key: &PipelineKey) {
        if self.pending.insert(key.clone())
            && let Some(q) = &self.queue
        {
            let _ = q.send(key.clone());
        }
    }

    pub(super) fn finished(&mut self) -> Vec<Built> {
        let out: Vec<_> = self.done.lock().unwrap().drain(..).collect();
        for (k, _) in &out {
            self.pending.remove(k);
        }
        out
    }

    pub(super) fn stop(&mut self) -> Vec<Built> {
        self.queue = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        self.finished()
    }
}

impl Renderer {
    pub(super) fn save_pipeline_cache(&mut self, wait: bool) {
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn shader(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        regs: &[u32],
        va: u64,
        stage: Stage,
        cbufs: [Option<(u64, u32)>; CB_SLOTS],
        attribute_types: [Kind; 32],
        output_types: [Kind; 8],
        compute: Option<&crate::gpu::compute::Qmd>,
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
        if crate::gpu::trace() {
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

    pub(super) fn layout(&mut self, bindings: &[(u32, vk::DescriptorType)], stages: vk::ShaderStageFlags) -> (vk::DescriptorSetLayout, vk::PipelineLayout) {
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

    pub(super) fn pipeline(&mut self, key: &PipelineKey, now: bool) -> Result<vk::Pipeline, String> {
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
}

pub(super) fn depth_transform(regs: &[u32]) -> Option<(f32, f32)> {
    let sz = f32::from_bits(regs[reg::VIEWPORT_SCALE + 2]);
    let oz = f32::from_bits(regs[reg::VIEWPORT_SCALE + 5]);
    (sz != 0.0 && (sz, oz) != (1.0, 0.0)).then_some((sz, oz))
}

pub(super) fn pipeline_cache_file(props: &vk::PhysicalDeviceProperties) -> Option<std::path::PathBuf> {
    if let Some(p) = crate::env_var!("KIRIKAE_PIPELINE_CACHE") {
        return (!p.is_empty()).then(|| p.into());
    }
    let base = crate::env_var!("LOCALAPPDATA")?;
    Some(std::path::Path::new(&base).join("kirikae").join(format!("pipelines-{:04x}-{:04x}.bin", props.vendor_id, props.device_id)))
}

pub(super) fn remove_stale_saves(file: &std::path::Path) {
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

pub(super) fn pipeline_cache_matches(data: &[u8], props: &vk::PhysicalDeviceProperties) -> bool {
    let word = |i: usize| data.get(4 * i..4 * i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    data.len() >= 32
        && word(0).is_some_and(|n| n >= 32)
        && word(1) == Some(vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32)
        && word(2) == Some(props.vendor_id)
        && word(3) == Some(props.device_id)
        && data[16..32] == props.pipeline_cache_uuid
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
