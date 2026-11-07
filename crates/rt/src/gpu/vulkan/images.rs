use ash::vk;
use kirikae_maxwell::{Kind, TexDim};

use crate::gpu::formats::ColorFormat;
use crate::gpu::fxhash::FastMap as HashMap;
use crate::gpu::threed::{CB_SLOTS, address, reg};
use crate::gpu::{Gpu, formats, tic};
use crate::mem::GuestMemory;
use super::readback::{bytes_per_pixel, read_surface};
use super::submit::Garbage;
use super::textures::{CROP, DERIVED, FEEDBACK, Source, read_texture};
use super::{Renderer, find_memory};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey {
    pub cpu: u64,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

pub struct Image {
    pub(super) image: vk::Image,
    pub(super) memory: vk::DeviceMemory,
    pub(super) views: HashMap<ViewKey, vk::ImageView>,
    pub(super) levels: u32,
    pub(super) layers: u32,
    pub gpu_dirty: bool,
    pub(super) stamp: u64,
    pub(super) source: Option<Source>,
    pub(super) drawn: Option<u64>,
    pub(super) cleared: Option<u64>,
    pub(super) clear_streak: u32,
}

impl Image {
    pub(super) fn note_clear(&mut self, frame: u64) {
        if self.cleared == Some(frame) {
            return;
        }
        self.clear_streak = if self.cleared == frame.checked_sub(1) { self.clear_streak + 1 } else { 1 };
        self.cleared = Some(frame);
    }

    pub(super) fn transient(&self, frame: u64) -> bool {
        self.cleared == Some(frame) && self.clear_streak >= 3
    }
}

pub(super) type ViewKey = (vk::Format, [vk::ComponentSwizzle; 4], vk::ImageViewType, u32, u32, u32);

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

impl Renderer {
    pub(super) fn create_image(&mut self, key: ImageKey, levels: u32, layers: u32) -> Result<(), vk::Result> {
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

    pub(super) fn image_memory(&mut self, size: u64, ty: u32, pooled: bool) -> Result<vk::DeviceMemory, vk::Result> {
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

    pub(super) fn free_image_memory(&mut self, memory: vk::DeviceMemory) {
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

    pub(super) fn view(&mut self, key: ImageKey, format: vk::Format, swizzle: [vk::ComponentSwizzle; 4]) -> vk::ImageView {
        self.view_of(key, format, swizzle, vk::ImageViewType::TYPE_2D)
    }

    pub(super) fn view_of(&mut self, key: ImageKey, format: vk::Format, swizzle: [vk::ComponentSwizzle; 4], view_type: vk::ImageViewType) -> vk::ImageView {
        self.view_range(key, format, swizzle, view_type, (0, u32::MAX), 0)
    }

    pub(super) fn view_range(&mut self, key: ImageKey, format: vk::Format, swizzle: [vk::ComponentSwizzle; 4], view_type: vk::ImageViewType, levels: (u32, u32), layer: u32) -> vk::ImageView {
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

    pub(super) fn find_image(&self, cpu: u64, width: u32, height: u32, like: vk::Format) -> Option<ImageKey> {
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

    pub(super) fn target_image(&mut self, cpu: u64, format: vk::Format, width: u32, height: u32) -> Option<(ImageKey, bool)> {
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
            if crate::gpu::trace() {
                eprintln!("gpu: image {:#x} {:?} {}x{} replaced by {format:?} {width}x{height}", k.cpu, k.format, k.width, k.height);
            }
            self.destroy_image(k);
        }
        let key = ImageKey { cpu, format, width: w, height: h };
        self.create_image(key, 1, 1).ok()?;
        Some((key, true))
    }

    pub(super) fn upload(&mut self, key: ImageKey, levels: &[(u32, u32, Vec<u8>)]) {
        self.upload_layers(key, levels.len(), levels);
    }

    pub(super) fn upload_layers(&mut self, key: ImageKey, per_layer: usize, levels: &[(u32, u32, Vec<u8>)]) {
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

    pub(super) fn destroy_image(&mut self, key: ImageKey) {
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

    pub(super) fn depth_format(&self, zt: u32) -> Option<vk::Format> {
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

    pub(super) fn depth_target(&mut self, gpu: &Gpu, regs: &[u32]) -> Option<ImageKey> {
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
        let Some(area) = clear_area(regs, key) else { return };
        self.begin();
        self.end_rendering();
        let depth = f32::from_bits(regs[0x364]).clamp(0.0, 1.0);
        let depth = if key.format == vk::Format::D16_UNORM { (depth * 65535.0 + 0.25).floor() / 65535.0 } else { depth };
        let value = vk::ClearDepthStencilValue { depth, stencil: regs[0x368] & 0xff };
        match area {
            Some(rect) => {
                let view = self.view(key, key.format, IDENTITY);
                let op = |on: bool| if on { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::LOAD };
                let attachment = |on: bool| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(op(on))
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { depth_stencil: value })
                };
                let (d, st) = (attachment(mask.contains(vk::ImageAspectFlags::DEPTH)), attachment(mask.contains(vk::ImageAspectFlags::STENCIL)));
                let mut info = vk::RenderingInfo::default().render_area(rect).layer_count(1).depth_attachment(&d);
                if aspects.contains(vk::ImageAspectFlags::STENCIL) {
                    info = info.stencil_attachment(&st);
                }
                unsafe {
                    self.device.cmd_begin_rendering(self.cmd, &info);
                    self.device.cmd_end_rendering(self.cmd);
                }
            }
            None => {
                let range = [vk::ImageSubresourceRange { aspect_mask: mask, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }];
                let image = self.images[&key].image;
                unsafe { self.device.cmd_clear_depth_stencil_image(self.cmd, image, vk::ImageLayout::GENERAL, &value, &range) };
            }
        }
        self.barrier();
        self.touch(key);
        if area.is_none() {
            self.images.get_mut(&key).unwrap().note_clear(crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
        }
    }

    pub fn clear(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], s: &Surface, color: [u32; 4]) {
        let Some((key, fmt)) = self.surface_image(gpu, mem, s, false) else { return };
        let Some(area) = clear_area(regs, key) else { return };
        self.begin();
        self.end_rendering();
        let value = match fmt.kind {
            Kind::Float => vk::ClearColorValue { float32: color.map(f32::from_bits) },
            Kind::Sint => vk::ClearColorValue { int32: color.map(|c| f32::from_bits(c) as i32) },
            Kind::Uint => vk::ClearColorValue { uint32: color.map(|c| f32::from_bits(c) as u32) },
        };
        match area {
            Some(rect) => {
                let view = self.view(key, fmt.vk, IDENTITY);
                let attachment = [vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: value })];
                let info = vk::RenderingInfo::default().render_area(rect).layer_count(1).color_attachments(&attachment);
                unsafe {
                    self.device.cmd_begin_rendering(self.cmd, &info);
                    self.device.cmd_end_rendering(self.cmd);
                }
            }
            None => {
                let range = [vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }];
                let image = self.images[&key].image;
                unsafe { self.device.cmd_clear_color_image(self.cmd, image, vk::ImageLayout::GENERAL, &value, &range) };
            }
        }
        self.barrier();
        self.touch(key);
        if area.is_none() {
            self.images.get_mut(&key).unwrap().note_clear(crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
        }
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

    pub(super) fn storage_image(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], cbufs: [Option<(u64, u32)>; CB_SLOTS], image: &kirikae_maxwell::ImageUse) -> Result<vk::ImageView, String> {
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
        if crate::gpu::trace() {
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
            _ if image.raw != 0 => {
                if aspect_of(key.format) != vk::ImageAspectFlags::COLOR || bytes_per_pixel(key.format) != Some(image.raw as u32) {
                    return Err(format!("raw {}-byte image access to {:?}", image.raw, key.format));
                }
                raw_format(image.raw)
            }
            vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
            f => f,
        };
        let level = if header.texture_type == 7 { 0 } else { header.min_level };
        if level >= self.images[&key].levels {
            return Err(format!("storage image level {level} beyond the image's {}", self.images[&key].levels));
        }
        if crate::gpu::trace() {
            let img = &self.images[&key];
            eprintln!("gpu:   -> image {:#x} {}x{} ({} levels, {} layers): level {level} layer {layer}", key.cpu, key.width, key.height, img.levels, img.layers);
        }
        self.touch(key);
        Ok(self.view_range(key, view_format, IDENTITY, vk::ImageViewType::TYPE_2D, (level, 1), layer))
    }
}

pub(super) const IDENTITY: [vk::ComponentSwizzle; 4] = [vk::ComponentSwizzle::IDENTITY; 4];

pub(super) fn raw_format(bytes: u8) -> vk::Format {
    match bytes {
        1 => vk::Format::R8_UINT,
        2 => vk::Format::R16_UINT,
        4 => vk::Format::R32_UINT,
        8 => vk::Format::R32G32_UINT,
        _ => vk::Format::R32G32B32A32_UINT,
    }
}

pub(super) fn aspect_of(f: vk::Format) -> vk::ImageAspectFlags {
    match f {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => vk::ImageAspectFlags::DEPTH,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

pub(super) fn same_class(a: vk::Format, b: vk::Format) -> bool {
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

const CLEAR_FLAGS: usize = 0x43e;
const CLEAR_SCISSOR: u32 = 0x100;

fn clear_area(regs: &[u32], key: ImageKey) -> Option<Option<vk::Rect2D>> {
    if regs[CLEAR_FLAGS] & CLEAR_SCISSOR == 0 || regs[reg::SCISSOR] & 1 == 0 {
        return Some(None);
    }
    let (h, v) = (regs[reg::SCISSOR + 1], regs[reg::SCISSOR + 2]);
    let (x0, x1) = ((h & 0xffff).min(key.width), (h >> 16).min(key.width));
    let (y0, y1) = ((v & 0xffff).min(key.height), (v >> 16).min(key.height));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    if (x0, y0, x1, y1) == (0, 0, key.width, key.height) {
        return Some(None);
    }
    Some(Some(vk::Rect2D { offset: vk::Offset2D { x: x0 as i32, y: y0 as i32 }, extent: vk::Extent2D { width: x1 - x0, height: y1 - y0 } }))
}
