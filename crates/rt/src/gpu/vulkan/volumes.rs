use ash::vk;

use crate::gpu::Gpu;
use crate::gpu::formats::TexFormat;
use crate::gpu::fxhash::FastMap as HashMap;
use crate::gpu::tic::{Layout, TextureHeader};
use crate::mem::GuestMemory;
use super::images::{ImageKey, same_class};
use super::textures::{ARRAY_VOLUME, CROP, SOURCE_RECHECK, read_texture, swizzle};
use super::tiling::{deswizzle_3d, fit_block, layer_size, volume_size};
use super::{Renderer, find_memory, hash64};

pub(super) struct Volume {
    pub(super) image: vk::Image,
    pub(super) memory: vk::DeviceMemory,
    pub(super) views: HashMap<[vk::ComponentSwizzle; 4], vk::ImageView>,
    pub(super) hash: u64,
    pub(super) checked: u64,
    pub(super) slices: Vec<u64>,
}

impl Renderer {
    pub(super) fn volume(&mut self, gpu: &Gpu, mem: &GuestMemory, cpu: u64, header: &TextureHeader, format: &TexFormat) -> Result<vk::ImageView, String> {
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
                let mut data = deswizzle_3d(&raw, bw, bh, d, format.block_bytes, block_height_log2, block_depth_log2);
                if format.astc {
                    let slice = (bw * bh * format.block_bytes) as usize;
                    data = data.chunks(slice).flat_map(|s| crate::gpu::astc::decode(s, w, h, format.block_w, format.block_h)).collect();
                }
                self.upload_volume(key, &data, w, h, d)?;
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
                .map(|z| self.rendered_slice(cpu + (z >> bd) * blocks_high * gobs_wide * block_size + ((z & ((1 << bd) - 1)) << bh) * 512, format.vk, w, h))
                .collect();
            self.copy_slices(key, &sources, w, h);
        }
        self.volume_view(key, header, format)
    }

    pub(super) fn array_volume(&mut self, gpu: &Gpu, mem: &GuestMemory, cpu: u64, header: &TextureHeader, format: &TexFormat) -> Result<vk::ImageView, String> {
        if format.astc || !matches!(header.layout, Layout::BlockLinear { .. }) {
            return Err("2D array sampled as 3D: only uncompressed block-linear arrays".into());
        }
        let (w, h, d) = (header.width.max(1), header.height.max(1), header.depth.max(1));
        let levels = header.levels.max(1);
        let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let key = (cpu | ARRAY_VOLUME, format.vk);
        if self.volumes.get(&key).is_none_or(|v| frame >= v.checked + SOURCE_RECHECK) {
            let read = read_texture(gpu, mem, header, format, levels, d);
            if read.len() != (levels * d) as usize {
                return Err("2D array sampled as 3D: layers unreadable".into());
            }
            let data: Vec<u8> = read.into_iter().step_by(levels as usize).flat_map(|(_, _, bytes)| bytes).collect();
            let hash = hash64(&data);
            if self.volumes.get(&key).map(|v| v.hash) != Some(hash) {
                self.upload_volume(key, &data, w, h, d)?;
            }
            let v = self.volumes.get_mut(&key).unwrap();
            v.hash = hash;
            v.checked = frame;
        }
        if let Some(stride) = layer_size(header, format, levels) {
            let sources: Vec<Option<ImageKey>> = (0..d as u64).map(|z| self.rendered_slice(cpu + z * stride, format.vk, w, h)).collect();
            self.copy_slices(key, &sources, w, h);
        }
        self.volume_view(key, header, format)
    }

    fn upload_volume(&mut self, key: (u64, vk::Format), data: &[u8], w: u32, h: u32, d: u32) -> Result<(), String> {
        if !self.volumes.contains_key(&key) {
            let v = self.create_volume(key.1, w, h, d).map_err(|e| format!("3D image: {e:?}"))?;
            self.volumes.insert(key, v);
        }
        let image = self.volumes[&key].image;
        self.begin();
        self.end_rendering();
        let offset = self.ring_copy(data, 16).ok_or("3D texture too large for the upload ring")?;
        let region = [vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_extent(vk::Extent3D { width: w, height: h, depth: d })];
        self.barrier();
        unsafe { self.device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, image, vk::ImageLayout::GENERAL, &region) };
        self.barrier();
        self.stats.uploads += 1;
        self.volumes.get_mut(&key).unwrap().slices.clear();
        Ok(())
    }

    fn rendered_slice(&self, at: u64, format: vk::Format, w: u32, h: u32) -> Option<ImageKey> {
        self.at(at).iter().copied().filter(|k| k.format == format && k.width >= w && k.height >= h && self.images[k].gpu_dirty).max_by_key(|k| self.images[k].stamp)
    }

    fn copy_slices(&mut self, key: (u64, vk::Format), sources: &[Option<ImageKey>], w: u32, h: u32) {
        if sources.iter().all(Option::is_none) {
            return;
        }
        let stamps: Vec<u64> = sources.iter().map(|s| s.map_or(0, |k| self.images[&k].stamp)).collect();
        if self.volumes[&key].slices == stamps {
            return;
        }
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

    fn volume_view(&mut self, key: (u64, vk::Format), header: &TextureHeader, format: &TexFormat) -> Result<vk::ImageView, String> {
        let swizzle = swizzle(header);
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

    pub(super) fn create_volume(&mut self, format: vk::Format, width: u32, height: u32, depth: u32) -> Result<Volume, vk::Result> {
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

    pub(super) fn layered_image(&mut self, gpu: &Gpu, mem: &GuestMemory, cpu: u64, header: &TextureHeader, format: &TexFormat) -> Result<Option<(ImageKey, u32)>, String> {
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

    pub(super) fn containing_image(&self, cpu: u64, header: &TextureHeader, format: &TexFormat, levels: u32) -> Option<(ImageKey, u32)> {
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
}
