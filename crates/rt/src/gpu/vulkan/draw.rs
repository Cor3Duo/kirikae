use ash::vk;
use kirikae_maxwell::{Kind, Stage};

use crate::gpu::formats::ColorFormat;
use crate::gpu::threed::{CB_SLOTS, Draw, address, reg};
use crate::gpu::{Gpu, formats};
use crate::mem::GuestMemory;
use super::Renderer;
use super::buffers::instanced_stream;
use super::descriptors::{Descriptor, FRAGMENT_BASE, STORAGE_OFFSET, VERTEX_BASE};
use super::images::{IDENTITY, ImageKey, aspect_of, color_surface};
use super::pipeline::alpha_test_func;

pub(super) type TargetAsk = (u64, vk::Format, u32, u32, u64);

#[derive(Clone, PartialEq, Eq)]
pub(super) struct RenderTargets {
    pub(super) views: Vec<(vk::ImageView, vk::Format)>,
    pub(super) depth: Option<(vk::ImageView, vk::Format)>,
    pub(super) extent: (u32, u32),
}

#[derive(Default)]
pub(super) struct DrawScratch {
    pub(super) targets: Vec<Option<(ImageKey, ColorFormat)>>,
    pub(super) sizes: Vec<(u32, u32)>,
    pub(super) streams: Vec<u32>,
    pub(super) views: Vec<(vk::ImageView, vk::Format)>,
    pub(super) descriptors: Vec<Descriptor>,
    pub(super) vertex_buffers: Vec<(u32, vk::Buffer, u64)>,
    pub(super) buffer_infos: Vec<vk::DescriptorBufferInfo>,
    pub(super) image_infos: Vec<vk::DescriptorImageInfo>,
    pub(super) writes: Vec<vk::WriteDescriptorSet<'static>>,
}

#[derive(Default)]
pub(super) struct Bound {
    pub(super) pipeline: Option<vk::Pipeline>,
    pub(super) viewport: Option<[u32; 4]>,
    pub(super) scissor: Option<(i32, i32, u32, u32)>,
    pub(super) cull: Option<vk::CullModeFlags>,
    pub(super) front: Option<vk::FrontFace>,
    pub(super) blend: Option<[u32; 4]>,
    pub(super) depth: Option<(bool, bool, vk::CompareOp)>,
    pub(super) stencil_test: Option<bool>,
    pub(super) stencil: Option<[[u32; 7]; 2]>,
    pub(super) depth_bias: Option<(bool, [u32; 3])>,
    pub(super) vertex: [Option<(vk::Buffer, u64)>; 32],
    pub(super) index: Option<(vk::Buffer, u64, vk::IndexType)>,
    pub(super) descriptors: Option<(vk::PipelineLayout, Vec<Descriptor>)>,
}

pub(super) fn changed<T: PartialEq>(slot: &mut Option<T>, v: T) -> bool {
    let new = slot.as_ref() != Some(&v);
    if new {
        *slot = Some(v);
    }
    new
}

impl Renderer {
    pub fn dispatch(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], qmd: &crate::gpu::compute::Qmd, va: u64) -> bool {
        let start = std::time::Instant::now();
        let ok = self.try_dispatch(gpu, mem, regs, qmd, va);
        self.stats.draw_time += start.elapsed();
        match &ok {
            Err(why) => {
                self.stats.skipped_dispatches += 1;
                let why = format!("compute: {why}");
                if crate::gpu::trace() {
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

    pub(super) fn try_dispatch(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], qmd: &crate::gpu::compute::Qmd, va: u64) -> Result<(), String> {
        self.guest = mem;
        self.split();
        let mut cbufs = [None; CB_SLOTS];
        cbufs[..crate::gpu::compute::QMD_CBUFS].copy_from_slice(&qmd.cbufs);
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
            if crate::gpu::trace() {
                eprintln!("gpu: draw skipped: {why}");
            }
        } else {
            self.stats.draws += 1;
        }
        ok.is_ok()
    }

    pub(super) fn writes_nothing(&self, regs: &[u32]) -> bool {
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

    pub(super) fn try_draw(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], cbufs: &[[Option<(u64, u32)>; CB_SLOTS]; 5], draw: Draw) -> Result<(), String> {
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
        if crate::gpu::trace() {
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
            let front = if regs[0x647] != 0x900 { vk::FrontFace::COUNTER_CLOCKWISE } else { vk::FrontFace::CLOCKWISE };
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
}
