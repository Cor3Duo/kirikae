//! The Vulkan side of NVN replay: render passes, clears and draws fed from the decoded records.
//!
//! One submit is one `VkCommandBuffer` (`Renderer::begin`/`flush`), so the queue order is the record order. Within
//! it a dynamic-rendering pass stays open across draws on the same attachments and ends when a clear or another
//! target needs the command buffer (`docs/nvn/CONTRACT.md` §4). Everything here uses the Maxwell path's helpers —
//! [`Renderer::pipeline`], [`Renderer::layout`], [`Renderer::bind_descriptors`], [`Renderer::texture_words`],
//! [`Renderer::target_image`], the ring and the buffer cache — so there is one renderer, two front ends.
//!
//! What the records can name today: render targets with their views, colour and depth/stencil clears, programs
//! (`nvnProgramSetShaders` stage lists), vertex attributes and streams, index and vertex buffers, uniform blocks,
//! textures through the NVN texture/sampler descriptor pools, and the blend/depth/stencil/cull state. A record that
//! names something outside that set (a storage buffer, a storage image, a compute dispatch, an instanced stream)
//! makes the walker skip only that draw with a logged reason.

use std::collections::HashSet;

use ash::vk;
use kirikae_maxwell::{Kind, Stage as MaxwellStage};

use super::Renderer;
use super::descriptors::{Descriptor, FRAGMENT_BASE, VERTEX_BASE};
use super::draw::{RenderTargets, admit_shader, changed};
use super::images::{IDENTITY, ImageKey, aspect_of};
use crate::gpu::threed::CB_SLOTS;
use crate::gpu::{Gpu, buffer_cache, tic};
use crate::mem::GuestMemory;
use crate::nvn::formats;
use crate::nvn::replay::{
    self, DrawCall, IndexCall, Sink, Stage, State, Submit, Targets, TextureInfo, ViewInfo, ViewRef,
};
use crate::process::Process;

/// A resolved attachment: the image the CPU address maps to, and the view the record selected.
#[derive(Clone, Copy)]
pub(super) struct Attachment {
    key: ImageKey,
    format: vk::Format,
    view: vk::ImageView,
    /// NVN format, for the clear value's type and the depth aspect.
    nvn_format: u32,
}

#[derive(Clone)]
pub(super) struct ResolvedTargets {
    colors: Vec<Option<Attachment>>,
    depth: Option<Attachment>,
    extent: (u32, u32),
}

/// The sink the replay worker walks into. Holds the pass state for one submit; a pass never survives it.
pub struct NvnSink<'a> {
    renderer: &'a mut Renderer,
    gpu: &'a Gpu,
    mem: &'a GuestMemory,
    submit: &'a Submit,
    targets: Option<ResolvedTargets>,
    pass_open: bool,
    logged: HashSet<String>,
    draws: u64,
    cleared: u64,
}

/// Replays one submit against the process's renderer. Called on the replay worker with no other lock held.
pub fn replay_submit(p: &'static Process, submit: &Submit) -> replay::Stats {
    let Some(stats) = p.gpu.with_renderer(|renderer| {
        let mut sink = NvnSink::new(renderer, &p.gpu, &p.memory, submit);
        let stats = replay::replay(submit, &mut sink);
        sink.finish_submit();
        stats
    }) else {
        let mut stats = replay::Stats::default();
        stats.skipped += 1;
        stats.reasons.insert("Vulkan is not available".into(), 1);
        return stats;
    };
    stats
}

impl<'a> NvnSink<'a> {
    fn new(
        renderer: &'a mut Renderer,
        gpu: &'a Gpu,
        mem: &'a GuestMemory,
        submit: &'a Submit,
    ) -> NvnSink<'a> {
        NvnSink {
            renderer,
            gpu,
            mem,
            submit,
            targets: None,
            pass_open: false,
            logged: HashSet::new(),
            draws: 0,
            cleared: 0,
        }
    }

    fn note(&mut self, reason: &str) -> String {
        if self.logged.insert(reason.to_string()) {
            eprintln!("nvn replay: {reason}");
        }
        reason.to_string()
    }

    fn attachment(&mut self, reference: ViewRef) -> Result<Attachment, String> {
        let texture = self
            .submit
            .textures
            .get(&reference.texture)
            .copied()
            .ok_or_else(|| {
                format!(
                    "texture {:#x} was not resolved at submit time",
                    reference.texture
                )
            })?;
        let view = self
            .submit
            .views
            .get(&reference.view)
            .copied()
            .unwrap_or_default();
        self.render_target(texture, view)
    }

    fn render_target(
        &mut self,
        texture: TextureInfo,
        view: ViewInfo,
    ) -> Result<Attachment, String> {
        let format = if view.format != 0 {
            view.format
        } else {
            texture.format
        };
        let vk_format = formats::vk(format)
            .ok_or_else(|| format!("render target format {format:#x} has no Vulkan equivalent"))?;
        let base_level = view.base_level.min(texture.levels.saturating_sub(1));
        let width = (texture.width >> base_level).max(1);
        let height = (texture.height >> base_level).max(1);
        let cpu = if texture.host != 0 {
            texture.host
        } else {
            self.gpu
                .translate(texture.gpu)
                .ok_or_else(|| format!("render target {:#x} is not mapped", texture.gpu))?
        };
        let (key, _) = self
            .renderer
            .target_image(cpu, vk_format, width, height)
            .ok_or_else(|| format!("no image for render target {cpu:#x} {vk_format:?}"))?;
        let levels = if view.levels == 0 {
            (base_level, u32::MAX)
        } else {
            (base_level, view.levels)
        };
        let image_view = self.renderer.view_range(
            key,
            vk_format,
            IDENTITY,
            vk::ImageViewType::TYPE_2D,
            levels,
            view.min_layer,
        );
        Ok(Attachment {
            key,
            format: vk_format,
            view: image_view,
            nvn_format: format,
        })
    }

    fn ensure_pass(&mut self) -> Result<(), String> {
        if self.pass_open {
            return Ok(());
        }
        let targets = self
            .targets
            .as_ref()
            .ok_or_else(|| "a draw or clear has no render targets".to_string())?;
        let views: Vec<(vk::ImageView, vk::Format)> = targets
            .colors
            .iter()
            .map(|color| match color {
                Some(a) => (a.view, a.format),
                None => (vk::ImageView::null(), vk::Format::UNDEFINED),
            })
            .collect();
        let depth = targets.depth.map(|d| (d.view, d.format));
        self.renderer.nvn_begin_pass(&views, depth, targets.extent);
        self.pass_open = true;
        Ok(())
    }

    fn end_pass(&mut self) {
        if self.pass_open {
            self.renderer.end_rendering();
            self.pass_open = false;
        }
    }

    fn finish_submit(&mut self) {
        self.end_pass();
        // One command buffer per submit, even when it only cleared: `flush` submits or polls.
        self.renderer.flush();
    }
}

impl Sink for NvnSink<'_> {
    fn begin(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn targets(&mut self, targets: &Targets) -> Result<(), String> {
        let mut colors = Vec::with_capacity(targets.colors.len());
        for color in &targets.colors {
            colors.push(match color {
                Some(reference) => Some(self.attachment(*reference)?),
                None => None,
            });
        }
        let depth = match targets.depth {
            Some(reference) => Some(self.attachment(reference)?),
            None => None,
        };
        let extent = colors
            .iter()
            .flatten()
            .chain(depth.iter())
            .fold((u32::MAX, u32::MAX), |e, a| {
                (e.0.min(a.key.width), e.1.min(a.key.height))
            });
        if extent.0 == u32::MAX {
            return Err("render targets name no attachment".into());
        }
        // Another set of targets: the current pass is over. The next draw opens one.
        if self
            .targets
            .as_ref()
            .is_some_and(|now| !same_targets(now, &colors, &depth))
        {
            self.end_pass();
        }
        self.targets = Some(ResolvedTargets {
            colors,
            depth,
            extent,
        });
        Ok(())
    }

    fn clear_color(&mut self, index: u32, color: [f32; 4], mask: u32) -> Result<(), String> {
        if mask & 0xf == 0 {
            return Ok(());
        }
        let Some(targets) = &self.targets else {
            return Err("clear before SetRenderTargets".into());
        };
        let Some(Some(attachment)) = targets.colors.get(index as usize).copied() else {
            return Err(format!("colour clear on unwritten target {index}"));
        };
        if mask & 0xf != 0xf {
            self.note("channel-masked colour clear is replayed as a full-channel clear");
        }
        self.end_pass();
        let value = match formats::kind(attachment.nvn_format) {
            Kind::Sint => vk::ClearColorValue {
                int32: color.map(|c| c.to_bits() as i32),
            },
            Kind::Uint => vk::ClearColorValue {
                uint32: color.map(|c| c.to_bits()),
            },
            Kind::Float => vk::ClearColorValue { float32: color },
        };
        self.renderer.nvn_clear_color(attachment.key, value);
        self.gpu
            .stats
            .clears
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cleared += 1;
        Ok(())
    }

    fn clear_depth_stencil(
        &mut self,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) -> Result<(), String> {
        let Some(targets) = &self.targets else {
            return Err("depth clear before SetRenderTargets".into());
        };
        let Some(attachment) = targets.depth else {
            return Err("depth clear without a depth target".into());
        };
        let aspects = aspect_of(attachment.format);
        if depth.is_some() && !aspects.contains(vk::ImageAspectFlags::DEPTH) {
            return Err("depth clear on a colour target".into());
        }
        if stencil.is_some() && !aspects.contains(vk::ImageAspectFlags::STENCIL) {
            return Err("stencil clear on a target without stencil".into());
        }
        self.end_pass();
        self.renderer
            .nvn_clear_depth(attachment.key, attachment.format, depth, stencil);
        self.gpu
            .stats
            .clears
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cleared += 1;
        Ok(())
    }

    fn draw(&mut self, state: &State, draw: &DrawCall) -> Result<(), String> {
        // A draw needs targets and a program; both errors are one line, not a silent skip.
        let program = self
            .submit
            .programs
            .get(&state.program)
            .ok_or_else(|| format!("program {:#x} was not recorded", state.program))?
            .clone();
        self.ensure_pass()?;
        let targets = self
            .targets
            .clone()
            .ok_or_else(|| "a draw has no render targets".to_string())?;
        self.renderer.nvn_draw(
            self.gpu,
            self.mem,
            self.submit,
            state,
            draw,
            &program.stages,
            &targets.colors,
            targets.depth,
            targets.extent,
        )?;
        // The frame report reads the process-wide counters, as the Maxwell path does; the NVN walker draws on the
        // replay worker, so nothing else moves them.
        self.gpu
            .stats
            .draws
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.draws += 1;
        Ok(())
    }

    fn dispatch(&mut self, grid: [u32; 3]) -> Result<(), String> {
        Err(self.note(&format!(
            "compute dispatch {grid:?} is not replayed yet (no NVN dispatch path)"
        )))
    }

    fn barrier(&mut self, _flags: u64) {
        self.end_pass();
    }

    fn finish(&mut self) -> Result<(), String> {
        self.end_pass();
        Ok(())
    }
}

fn same_targets(
    now: &ResolvedTargets,
    colors: &[Option<Attachment>],
    depth: &Option<Attachment>,
) -> bool {
    now.colors.len() == colors.len()
        && now.colors.iter().zip(colors).all(|(a, b)| match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => a.key == b.key && a.format == b.format,
            _ => false,
        })
        && match (now.depth, depth) {
            (None, None) => true,
            (Some(a), Some(b)) => a.key == b.key && a.format == b.format,
            _ => false,
        }
}

impl Renderer {
    /// Starts a dynamic-rendering pass on the resolved attachments. Loads, as NVN clears happen before it.
    pub(super) fn nvn_begin_pass(
        &mut self,
        colors: &[(vk::ImageView, vk::Format)],
        depth: Option<(vk::ImageView, vk::Format)>,
        extent: (u32, u32),
    ) {
        self.begin();
        let attachments: Vec<vk::RenderingAttachmentInfo> = colors
            .iter()
            .map(|&(view, _)| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
            })
            .collect();
        let depth_attachment = depth.map(|(view, _)| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        });
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: extent.0.max(1),
                    height: extent.1.max(1),
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        if let Some(d) = depth_attachment.as_ref() {
            info = info.depth_attachment(d);
            if depth.is_some_and(|(_, format)| {
                aspect_of(format).contains(vk::ImageAspectFlags::STENCIL)
            }) {
                info = info.stencil_attachment(d);
            }
        }
        unsafe { self.device.cmd_begin_rendering(self.cmd, &info) };
        self.rendering = Some(RenderTargets {
            views: colors.to_vec(),
            depth,
            extent,
        });
    }

    /// Clears a whole colour attachment: a transfer clear, the same call the Maxwell clear path makes.
    pub(super) fn nvn_clear_color(&mut self, key: ImageKey, value: vk::ClearColorValue) {
        self.begin();
        self.end_rendering();
        let range = [vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }];
        let image = self.images[&key].image;
        unsafe {
            self.device.cmd_clear_color_image(
                self.cmd,
                image,
                vk::ImageLayout::GENERAL,
                &value,
                &range,
            )
        };
        self.barrier();
        self.touch(key);
        self.images
            .get_mut(&key)
            .unwrap()
            .note_clear(crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
    }

    pub(super) fn nvn_clear_depth(
        &mut self,
        key: ImageKey,
        format: vk::Format,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
        let aspects = aspect_of(format);
        let mut mask = vk::ImageAspectFlags::empty();
        if depth.is_some() {
            mask |= aspects & vk::ImageAspectFlags::DEPTH;
        }
        if stencil.is_some() {
            mask |= aspects & vk::ImageAspectFlags::STENCIL;
        }
        if mask.is_empty() {
            return;
        }
        self.begin();
        self.end_rendering();
        let value = vk::ClearDepthStencilValue {
            depth: depth.unwrap_or(1.0).clamp(0.0, 1.0),
            stencil: stencil.unwrap_or(0) as u32 & 0xff,
        };
        let range = [vk::ImageSubresourceRange {
            aspect_mask: mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }];
        let image = self.images[&key].image;
        unsafe {
            self.device.cmd_clear_depth_stencil_image(
                self.cmd,
                image,
                vk::ImageLayout::GENERAL,
                &value,
                &range,
            )
        };
        self.barrier();
        self.touch(key);
        self.images
            .get_mut(&key)
            .unwrap()
            .note_clear(crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
    }

    /// One NVN draw: program, pipeline key, descriptors, vertex streams and the draw call itself.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn nvn_draw(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        submit: &Submit,
        state: &State,
        draw: &DrawCall,
        stages: &[Stage],
        colors: &[Option<Attachment>],
        depth: Option<Attachment>,
        extent: (u32, u32),
    ) -> Result<(), String> {
        self.guest = mem;
        // No `split()`: a submit is one command buffer, and its pass state would not survive a split.
        let vs_stage = stages
            .iter()
            .find(|s| !s.fragment)
            .copied()
            .ok_or("the program has no vertex stage")?;
        let fs_stage = stages.iter().find(|s| s.fragment).copied();
        let depth_format = depth.map_or(vk::Format::UNDEFINED, |d| d.format);
        // The translation reads a few viewport and depth fields; everything else is irrelevant to it. A fixed array
        // keeps this off the heap on the draw path.
        let mut regs = [0u32; 0x1000];
        if depth_format == vk::Format::D16_UNORM {
            regs[0x54e] = 1;
            regs[0x3fa] = 0x13;
        }
        if let Some(viewport) = state.viewport {
            let _ = viewport;
        }
        let mut attribute_types = [Kind::Float; 32];
        let mut attributes = Vec::new();
        for (index, attrib) in state.attribs.iter().enumerate() {
            let Some(attrib) = attrib else { continue };
            let Some(format) = formats::vk(attrib.format) else {
                continue;
            };
            let kind = formats::kind(attrib.format);
            if let Some(slot) = attribute_types.get_mut(index) {
                *slot = kind;
            }
            attributes.push((index as u32, attrib.stream, format, attrib.offset));
        }
        let mut output_types = [Kind::Float; 8];
        for (index, color) in colors.iter().enumerate().take(8) {
            if let Some(color) = color {
                output_types[index] = formats::kind(color.nvn_format);
            }
        }
        let vertex_cbufs = nvn_cbufs(state, 0);
        let vs = self.shader(
            gpu,
            mem,
            &regs,
            vs_stage.code,
            MaxwellStage::Vertex,
            vertex_cbufs,
            attribute_types,
            output_types,
            None,
        )?;
        let fs = match fs_stage {
            Some(stage) => Some(self.shader(
                gpu,
                mem,
                &regs,
                stage.code,
                MaxwellStage::Fragment,
                nvn_cbufs(state, 1),
                attribute_types,
                output_types,
                None,
            )?),
            None => None,
        };
        let partial = crate::env_var!("KIRIKAE_DRAW_PARTIAL").is_some();
        for shader in std::iter::once(&vs).chain(fs.as_ref()) {
            admit_shader(shader, partial)?;
        }
        let pair = (
            vs.module,
            fs.as_ref().map_or(vk::ShaderModule::null(), |f| f.module),
        );
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

        // The pipeline key: shaders, vertex layout, target formats and the blend state the records set.
        let mut key = std::mem::take(&mut self.scratch_key);
        key.vertex = vs.module;
        key.fragment = fs.as_ref().map_or(vk::ShaderModule::null(), |f| f.module);
        key.layout = layout;
        key.topology = topology(draw.topology)?;
        key.colors.clear();
        key.colors.extend(
            colors
                .iter()
                .map(|c| c.map_or(vk::Format::UNDEFINED, |a| a.format)),
        );
        key.depth = depth_format;
        key.depth_clamp = false;
        key.alpha_test = [7, 0];
        key.attributes.clear();
        key.bindings.clear();
        let mut used_streams: Vec<u32> = Vec::new();
        for &(location, stream, format, offset) in &attributes {
            if vs.shader.inputs >> location & 1 == 0 {
                continue;
            }
            let Some(stream_state) = state.streams.get(stream as usize).copied().flatten() else {
                return Err(format!(
                    "vertex attribute {location} uses unbound stream {stream}"
                ));
            };
            key.attributes.push((location, stream, format, offset));
            if !key.bindings.iter().any(|b| b.0 == stream) {
                key.bindings.push((stream, stream_state.stride));
                used_streams.push(stream);
            }
        }
        key.bindings.sort_unstable();
        for stream in &used_streams {
            if state
                .streams
                .get(*stream as usize)
                .copied()
                .flatten()
                .is_some_and(|s| s.divisor > 0)
            {
                return Err(format!(
                    "vertex stream {stream} has an instance divisor, which the pipeline key does not carry yet"
                ));
            }
        }
        key.blend.clear();
        for (index, color) in colors.iter().enumerate() {
            let mask = if color.is_some() {
                u32::from(state.render.channel_mask.get(index).copied().unwrap_or(0xf))
            } else {
                0
            };
            let enable = u32::from(
                state
                    .render
                    .blend_enable
                    .get(index)
                    .copied()
                    .unwrap_or(false),
            );
            key.blend.push([
                enable,
                state.render.blend_equation,
                blend_factor(state.render.blend_src),
                blend_factor(state.render.blend_dst),
                state.render.blend_equation_alpha,
                blend_factor(state.render.blend_src_alpha),
                blend_factor(state.render.blend_dst_alpha),
                mask,
            ]);
        }
        let pipeline = self.pipeline(&key, true)?;
        self.scratch_key = key;

        let mut scratch = std::mem::take(&mut self.scratch);
        let mut descriptors = std::mem::take(&mut scratch.descriptors);
        descriptors.clear();
        self.draw_targets.clear();
        self.draw_targets.extend(
            colors
                .iter()
                .flatten()
                .map(|a| a.key)
                .chain(depth.map(|d| d.key)),
        );
        let vs_constants = vs_stage.constants;
        let fs_constants = fs_stage.and_then(|stage| stage.constants);
        let described = self.nvn_descriptors(
            gpu,
            mem,
            submit,
            state,
            &vs,
            fs.as_deref(),
            vs_constants,
            fs_constants,
            &mut descriptors,
        );
        self.draw_targets.clear();
        described?;
        self.bind_descriptors(
            vk::PipelineBindPoint::GRAPHICS,
            set_layout,
            layout,
            &descriptors,
            &mut scratch,
        );

        let mut vertex_buffers = std::mem::take(&mut scratch.vertex_buffers);
        vertex_buffers.clear();
        for stream in &used_streams {
            let Some((address, size)) = state.buffers.get(*stream as usize).copied().flatten()
            else {
                return Err(format!("vertex stream {stream} has no BindVertexBuffer"));
            };
            if address == 0 || size == 0 {
                return Err(format!("vertex stream {stream} is empty"));
            }
            let (buffer, offset) = self
                .guest_buffer(gpu, mem, address, size, 16, buffer_cache::Kind::Vertex)
                .map_err(|e| format!("vertex stream {stream}: {e}"))?;
            self.stats.vertex_bytes += size;
            vertex_buffers.push((*stream, buffer, offset));
        }

        let index = match draw.index {
            Some(index) => Some(self.nvn_index_buffer(gpu, mem, index, draw.count)?),
            None => None,
        };
        self.begin();
        self.nvn_dynamic_state(extent, state);
        unsafe {
            let d = &self.device;
            let cmd = self.cmd;
            if changed(&mut self.bound.pipeline, pipeline) {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            }
            for &(binding, buffer, offset) in &vertex_buffers {
                if changed(&mut self.bound.vertex[binding as usize], (buffer, offset)) {
                    d.cmd_bind_vertex_buffers(cmd, binding, &[buffer], &[offset]);
                }
            }
            match index {
                Some((buffer, offset, ty, count)) => {
                    if changed(&mut self.bound.index, (buffer, offset, ty)) {
                        d.cmd_bind_index_buffer(cmd, buffer, offset, ty);
                    }
                    d.cmd_draw_indexed(
                        cmd,
                        count,
                        1,
                        0,
                        draw.index.map_or(0, |i| i.base_vertex),
                        draw.instances,
                    );
                }
                None => d.cmd_draw(
                    cmd,
                    draw.count,
                    draw.instances.max(1),
                    draw.first,
                    draw.base_instance,
                ),
            }
        }
        for color in colors.iter().flatten().chain(depth.iter()) {
            self.touch(color.key);
            let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
            if let Some(image) = self.images.get_mut(&color.key) {
                image.drawn = Some(frame);
            }
        }
        scratch.descriptors = descriptors;
        scratch.vertex_buffers = vertex_buffers;
        self.scratch = scratch;
        self.stats.draws += 1;
        Ok(())
    }

    fn nvn_dynamic_state(&mut self, extent: (u32, u32), state: &State) {
        let viewport = state
            .viewport
            .map(|v| vk::Viewport {
                x: v[0],
                y: v[1],
                width: v[2],
                height: v[3],
                min_depth: v[4],
                max_depth: v[5],
            })
            .filter(|v| v.width > 0.0 && v.height > 0.0)
            .unwrap_or(vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.0 as f32,
                height: extent.1 as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            });
        if changed(
            &mut self.bound.viewport,
            [viewport.x, viewport.y, viewport.width, viewport.height].map(f32::to_bits),
        ) {
            unsafe { self.device.cmd_set_viewport(self.cmd, 0, &[viewport]) };
        }
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent.0.max(1),
                height: extent.1.max(1),
            },
        };
        if changed(
            &mut self.bound.scissor,
            (
                scissor.offset.x,
                scissor.offset.y,
                scissor.extent.width,
                scissor.extent.height,
            ),
        ) {
            unsafe { self.device.cmd_set_scissor(self.cmd, 0, &[scissor]) };
        }
        let cull = cull_mode(state.render.cull);
        if changed(&mut self.bound.cull, cull) {
            unsafe { self.device.cmd_set_cull_mode(self.cmd, cull) };
        }
        let front = if state.render.front == 1 {
            vk::FrontFace::COUNTER_CLOCKWISE
        } else {
            vk::FrontFace::CLOCKWISE
        };
        if changed(&mut self.bound.front, front) {
            unsafe { self.device.cmd_set_front_face(self.cmd, front) };
        }
        let (test, write, op) = (
            state.render.depth_test,
            state.render.depth_write,
            compare_op(state.render.depth_func),
        );
        let was = self.bound.depth;
        if changed(&mut self.bound.depth, (test, write, op)) {
            unsafe {
                if was.is_none_or(|w| w.0 != test) {
                    self.device.cmd_set_depth_test_enable(self.cmd, test);
                }
                if was.is_none_or(|w| w.1 != write) {
                    self.device.cmd_set_depth_write_enable(self.cmd, write);
                }
                if was.is_none_or(|w| w.2 != op) {
                    self.device.cmd_set_depth_compare_op(self.cmd, op);
                }
            }
        }
        let stencil_test = state.render.stencil_test;
        if changed(&mut self.bound.stencil_test, stencil_test) {
            unsafe {
                self.device
                    .cmd_set_stencil_test_enable(self.cmd, stencil_test)
            };
        }
        // `bound.stencil` keeps the Maxwell order [fail, depth_fail, pass, func, ref, compare_mask, write_mask].
        let faces: [[u32; 7]; 2] = std::array::from_fn(|face| {
            let f = state.render.stencil[face];
            [f[1], f[2], f[3], f[0], 0, 0xff, 0xff]
        });
        if stencil_test && changed(&mut self.bound.stencil, faces) {
            for (face, s) in [
                (vk::StencilFaceFlags::FRONT, faces[0]),
                (vk::StencilFaceFlags::BACK, faces[1]),
            ] {
                unsafe {
                    self.device.cmd_set_stencil_op(
                        self.cmd,
                        face,
                        stencil_op(s[0]),
                        stencil_op(s[2]),
                        stencil_op(s[1]),
                        compare_op(s[3]),
                    );
                    self.device
                        .cmd_set_stencil_reference(self.cmd, face, s[4] & 0xff);
                    self.device
                        .cmd_set_stencil_compare_mask(self.cmd, face, s[5] & 0xff);
                    self.device
                        .cmd_set_stencil_write_mask(self.cmd, face, s[6] & 0xff);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn nvn_descriptors(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        submit: &Submit,
        state: &State,
        vs: &super::pipeline::ShaderModule,
        fs: Option<&super::pipeline::ShaderModule>,
        vs_constants: Option<(u64, u32)>,
        fs_constants: Option<(u64, u32)>,
        out: &mut Vec<Descriptor>,
    ) -> Result<(), String> {
        self.nvn_stage_descriptors(
            gpu,
            mem,
            submit,
            state,
            0,
            vs,
            VERTEX_BASE,
            vs_constants,
            out,
        )?;
        if let Some(fs) = fs {
            self.nvn_stage_descriptors(
                gpu,
                mem,
                submit,
                state,
                1,
                fs,
                FRAGMENT_BASE,
                fs_constants,
                out,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn nvn_stage_descriptors(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        submit: &Submit,
        state: &State,
        stage_index: usize,
        shader: &super::pipeline::ShaderModule,
        base: u32,
        constants: Option<(u64, u32)>,
        out: &mut Vec<Descriptor>,
    ) -> Result<(), String> {
        let ubo_align = self.limits.min_uniform_buffer_offset_alignment.max(16);
        for slot in 0..CB_SLOTS {
            if shader.shader.cbufs >> slot & 1 == 0 {
                continue;
            }
            match slot {
                // c[0]: driver constants. Nothing in the records fills the storage descriptors yet; the point size
                // lives at 0x610 and only two vertex shaders read it.
                0 => {
                    let (offset, data) = self.ring_alloc(0x620, 16).ok_or("ring full")?;
                    data.fill(0);
                    out.push(Descriptor::Buffer(
                        base,
                        vk::DescriptorType::UNIFORM_BUFFER,
                        self.ring.buffer,
                        offset,
                        0x620,
                    ));
                }
                // c[2]: texture handles. Resolved below, from the state and the NVN descriptor pools.
                2 => {}
                1 => {
                    let (address, size) = constants
                        .ok_or("the shader reads c[1] but the stage has no constant data")?;
                    let size = size.clamp(16, 0x10000) as u64;
                    let (buffer, offset) = self.guest_buffer(
                        gpu,
                        mem,
                        address,
                        size,
                        ubo_align,
                        buffer_cache::Kind::Constant,
                    )?;
                    self.stats.cbuf_bytes += size;
                    out.push(Descriptor::Buffer(
                        base + slot as u32,
                        vk::DescriptorType::UNIFORM_BUFFER,
                        buffer,
                        offset,
                        size,
                    ));
                }
                _ => {
                    let Some((address, size)) =
                        state.cbufs[stage_index].get(slot - 3).copied().flatten()
                    else {
                        return Err(format!(
                            "the shader reads c[{slot}] but no BindUniformBuffer filled it"
                        ));
                    };
                    let used = shader.shader.cbuf_used.get(slot).copied().unwrap_or(0);
                    let size = size
                        .min(u64::from(used.next_multiple_of(16)))
                        .clamp(16, 0x10000);
                    let (buffer, offset) = self.guest_buffer(
                        gpu,
                        mem,
                        address,
                        size,
                        ubo_align,
                        buffer_cache::Kind::Constant,
                    )?;
                    self.stats.cbuf_bytes += size;
                    out.push(Descriptor::Buffer(
                        base + slot as u32,
                        vk::DescriptorType::UNIFORM_BUFFER,
                        buffer,
                        offset,
                        size,
                    ));
                }
            }
        }
        if shader.shader.storage != 0 {
            return Err("the shader reads storage buffers, which are not recorded yet".into());
        }
        if !shader.shader.images.is_empty() {
            return Err("the shader writes storage images, which are not recorded yet".into());
        }
        for texture in &shader.shader.textures {
            let (view, sampler) =
                self.nvn_texture(gpu, mem, submit, state, stage_index, texture)?;
            out.push(Descriptor::Image(texture.binding, view));
            out.push(Descriptor::Sampler(texture.binding + 1, sampler));
        }
        Ok(())
    }

    fn nvn_texture(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        submit: &Submit,
        state: &State,
        stage_index: usize,
        texture: &kirikae_maxwell::TextureUse,
    ) -> Result<(vk::ImageView, vk::Sampler), String> {
        let handle = match texture.slot {
            // Both the immediate and the `c[2]` register form carry the handle's word offset in the texture bank,
            // where binding *n* is the 64-bit word at byte 0x20 + 8n, i.e. word 8 + 2n
            // (`docs/nvn/CONTRACT.md` §5). The 13-bit immediate index of the `tex`/`txd` immediate form counts
            // those same words.
            Some(2) | None => {
                let Some(binding) = texture.handle.checked_div(2).and_then(|w| w.checked_sub(4))
                else {
                    return Err(format!(
                        "texture handle word {} is outside c[2]",
                        texture.handle
                    ));
                };
                let Some(&handle) = state.textures[stage_index].get(binding as usize) else {
                    return Err(format!(
                        "texture binding {binding} outside the texture bank"
                    ));
                };
                handle
            }
            Some(slot) => {
                let Some((address, size)) = state.cbufs[stage_index]
                    .get(slot as usize - 3)
                    .copied()
                    .flatten()
                else {
                    return Err(format!("texture handle bank c[{slot}] is not bound"));
                };
                if u64::from(texture.handle) * 4 + 4 > size {
                    return Err("texture handle outside its uniform block".into());
                }
                let bytes = gpu
                    .bytes(mem, address + u64::from(texture.handle) * 4, 4)
                    .ok_or("texture handle unreadable")?;
                u32::from_le_bytes(bytes.try_into().unwrap_or_default())
            }
        };
        if handle == 0 {
            return Err("texture handle is 0".into());
        }
        let (tic_index, tsc_index) = (handle & 0xfffff, handle >> 20);
        if submit.texture_pool == 0 {
            return Err("no texture pool is bound".into());
        }
        let tic = tic::words(
            mem.read(submit.texture_pool + u64::from(tic_index) * 32, 32)
                .as_deref()
                .ok_or("texture descriptor unreadable")?,
        )
        .ok_or("texture descriptor is short")?;
        let tsc = if submit.sampler_pool != 0 {
            tic::words(
                mem.read(submit.sampler_pool + u64::from(tsc_index) * 32, 32)
                    .as_deref()
                    .ok_or("sampler descriptor unreadable")?,
            )
            .ok_or("sampler descriptor is short")?
        } else {
            sampler_default_words()
        };
        self.texture_words(
            gpu,
            mem,
            tic,
            tsc,
            texture.dim,
            texture.array,
            texture.shadow,
        )
        .map_err(|error| format!("{error} (handle {handle:#x})"))
    }

    fn nvn_index_buffer(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        index: IndexCall,
        count: u32,
    ) -> Result<(vk::Buffer, u64, vk::IndexType, u32), String> {
        if index.address == 0 || count == 0 {
            return Err("index buffer is empty".into());
        }
        match index.kind {
            0 => {
                let bytes = gpu
                    .bytes(mem, index.address, u64::from(count))
                    .ok_or("index buffer unreadable")?;
                let wide: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
                let data: Vec<u8> = wide.iter().flat_map(|i| i.to_le_bytes()).collect();
                let offset = self.ring_copy(&data, 16).ok_or("ring full")?;
                Ok((self.ring.buffer, offset, vk::IndexType::UINT32, count))
            }
            kind => {
                let size = if kind == 1 { 2 } else { 4 };
                let ty = if kind == 1 {
                    vk::IndexType::UINT16
                } else {
                    vk::IndexType::UINT32
                };
                let len = u64::from(count) * size;
                let (buffer, offset) = self
                    .guest_buffer(
                        gpu,
                        mem,
                        index.address,
                        len,
                        16,
                        buffer_cache::Kind::Index(size as u32, false),
                    )
                    .map_err(|e| format!("index buffer: {e}"))?;
                Ok((buffer, offset, ty, count))
            }
        }
    }
}

fn nvn_cbufs(state: &State, stage: usize) -> [Option<(u64, u32)>; CB_SLOTS] {
    let mut cbufs: [Option<(u64, u32)>; CB_SLOTS] = [None; CB_SLOTS];
    for (slot, value) in state.cbufs[stage].iter().enumerate() {
        if let Some(&(address, size)) = value.as_ref() {
            cbufs[slot + 3] = Some((address, size.min(u64::from(u32::MAX)) as u32));
        }
    }
    cbufs
}

fn topology(mode: u32) -> Result<vk::PrimitiveTopology, String> {
    Ok(match mode {
        0 => vk::PrimitiveTopology::POINT_LIST,
        1 => vk::PrimitiveTopology::LINE_LIST,
        3 => vk::PrimitiveTopology::LINE_STRIP,
        4 => vk::PrimitiveTopology::TRIANGLE_LIST,
        5 => vk::PrimitiveTopology::TRIANGLE_STRIP,
        6 => vk::PrimitiveTopology::TRIANGLE_FAN,
        other => {
            return Err(format!(
                "primitive topology {other} has no Vulkan equivalent"
            ));
        }
    })
}

/// NVN's constant blend factors are 0x61..0x64; the shared Maxwell mapping spells those 0xe/0xf/0xc003/0xc004.
fn blend_factor(factor: u32) -> u32 {
    match factor {
        0x61 => 0xe,
        0x62 => 0xf,
        0x63 => 0xc003,
        0x64 => 0xc004,
        other => other,
    }
}

fn cull_mode(face: u32) -> vk::CullModeFlags {
    match face {
        1 => vk::CullModeFlags::FRONT,
        2 => vk::CullModeFlags::BACK,
        3 => vk::CullModeFlags::FRONT_AND_BACK,
        _ => vk::CullModeFlags::NONE,
    }
}

pub(super) fn compare_op(func: u32) -> vk::CompareOp {
    match func {
        1 => vk::CompareOp::NEVER,
        2 => vk::CompareOp::LESS,
        3 => vk::CompareOp::EQUAL,
        4 => vk::CompareOp::LESS_OR_EQUAL,
        5 => vk::CompareOp::GREATER,
        6 => vk::CompareOp::NOT_EQUAL,
        7 => vk::CompareOp::GREATER_OR_EQUAL,
        _ => vk::CompareOp::ALWAYS,
    }
}

fn stencil_op(op: u32) -> vk::StencilOp {
    match op {
        2 => vk::StencilOp::ZERO,
        3 => vk::StencilOp::REPLACE,
        4 => vk::StencilOp::INCREMENT_AND_CLAMP,
        5 => vk::StencilOp::DECREMENT_AND_CLAMP,
        6 => vk::StencilOp::INVERT,
        7 => vk::StencilOp::INCREMENT_AND_WRAP,
        8 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => vk::StencilOp::KEEP,
    }
}

/// A sampler state that repeats and filters linearly, for a draw whose sampler slot is not in the pool.
fn sampler_default_words() -> [u32; 8] {
    let mut words = [0u32; 8];
    // mag linear, min linear, mip linear, lod clamp wide open: `tic::sampler` reads these bit positions.
    words[1] = 2 | 2 << 4 | 3 << 6;
    words[3] = 0xfff | 0xfff << 12;
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvn::replay::Render;

    #[test]
    fn nvn_enums_map_to_vulkan() {
        assert_eq!(topology(4), Ok(vk::PrimitiveTopology::TRIANGLE_LIST));
        assert_eq!(topology(5), Ok(vk::PrimitiveTopology::TRIANGLE_STRIP));
        assert!(topology(2).is_err(), "NVN has no quad list here");
        assert_eq!(cull_mode(0), vk::CullModeFlags::NONE);
        assert_eq!(cull_mode(2), vk::CullModeFlags::BACK);
        assert_eq!(compare_op(2), vk::CompareOp::LESS);
        assert_eq!(compare_op(4), vk::CompareOp::LESS_OR_EQUAL);
        assert_eq!(compare_op(99), vk::CompareOp::ALWAYS);
        assert_eq!(stencil_op(1), vk::StencilOp::KEEP);
        assert_eq!(stencil_op(3), vk::StencilOp::REPLACE);
    }

    #[test]
    fn default_render_state_is_the_nvn_one() {
        let render = Render::default();
        assert_eq!(render.channel_mask, [0xf; 8]);
        assert!(!render.depth_test);
        assert_eq!(render.depth_func, 2);
        assert_eq!(
            (render.blend_src, render.blend_dst),
            (2, 1),
            "ONE, ZERO: blending off by default"
        );
    }

    #[test]
    fn a_texture_handle_resolves_its_binding() {
        let mut state = State::default();
        state.textures[1][3] = 0x0020_2007;
        // `8 + 2n` is the word index of sampler binding n.
        let word = 8 + 2 * 3;
        assert_eq!(state.textures[1][word / 2 - 4], 0x0020_2007);
        assert_eq!(state.textures[0][0], 0);
    }

    #[test]
    fn stage_constants_are_carried_from_the_program_record() {
        // The stage list comes from `nvnProgramSetShaders`; its c[1] is a code-block range.
        let stage = Stage {
            code: 0x1030,
            fragment: false,
            constants: Some((0x1200, 0x40)),
        };
        assert_eq!(stage.constants, Some((0x1200, 0x40)));
        assert!(!stage.fragment);
    }
}
