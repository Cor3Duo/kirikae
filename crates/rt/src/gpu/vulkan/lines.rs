use ash::vk;

use crate::gpu::Gpu;
use crate::gpu::formats::TexFormat;
use crate::gpu::fxhash::FastMap as HashMap;
use crate::gpu::tic::TextureHeader;
use crate::mem::GuestMemory;
use super::textures::{SOURCE_RECHECK, read_texture, swizzle};
use super::{Renderer, find_memory, hash64};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct LineKey {
    cpu: u64,
    format: vk::Format,
    width: u32,
    levels: u32,
    layers: u32,
}

pub(super) struct Line {
    pub(super) image: vk::Image,
    pub(super) memory: vk::DeviceMemory,
    pub(super) views: HashMap<([vk::ComponentSwizzle; 4], vk::ImageViewType), vk::ImageView>,
    hash: u64,
    checked: u64,
}

pub(super) fn line_levels(width: u32, levels: u32) -> u32 {
    levels.clamp(1, 32 - width.max(1).leading_zeros())
}

impl Renderer {
    pub(super) fn line(&mut self, gpu: &Gpu, mem: &GuestMemory, cpu: u64, header: &TextureHeader, format: &TexFormat, array: bool) -> Result<vk::ImageView, String> {
        if format.astc || format.block_w != 1 || format.block_h != 1 {
            return Err("1D texture in a block-compressed format".into());
        }
        let width = header.width.max(1);
        let key = LineKey { cpu, format: format.vk, width, levels: line_levels(width, header.levels), layers: if array { header.depth.max(1) } else { 1 } };
        let frame = crate::display::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        if self.lines.get(&key).is_none_or(|l| frame >= l.checked + SOURCE_RECHECK) {
            let read = read_texture(gpu, mem, header, format, key.levels, key.layers);
            if read.len() != (key.levels * key.layers) as usize {
                return Err("1D texture unreadable".into());
            }
            let hash = read.iter().fold(0, |h: u64, (_, _, bytes)| h.rotate_left(5) ^ hash64(bytes));
            if self.lines.get(&key).map(|l| l.hash) != Some(hash) {
                if !self.lines.contains_key(&key) {
                    let line = self.create_line(key).map_err(|e| format!("1D image: {e:?}"))?;
                    self.lines.insert(key, line);
                }
                self.upload_line(key, &read)?;
            }
            let line = self.lines.get_mut(&key).unwrap();
            line.hash = hash;
            line.checked = frame;
        }
        let view_type = if array { vk::ImageViewType::TYPE_1D_ARRAY } else { vk::ImageViewType::TYPE_1D };
        self.line_view(key, swizzle(header), view_type)
    }

    pub(super) fn line_placeholder(&mut self, array: bool) -> Result<vk::ImageView, String> {
        let key = LineKey { cpu: u64::MAX, format: vk::Format::R8G8B8A8_UNORM, width: 1, levels: 1, layers: 1 };
        if !self.lines.contains_key(&key) {
            let line = self.create_line(key).map_err(|e| format!("1D image: {e:?}"))?;
            self.lines.insert(key, line);
            self.upload_line(key, &[(1, 1, vec![0; 4])])?;
        }
        let view_type = if array { vk::ImageViewType::TYPE_1D_ARRAY } else { vk::ImageViewType::TYPE_1D };
        self.line_view(key, [vk::ComponentSwizzle::IDENTITY; 4], view_type)
    }

    fn upload_line(&mut self, key: LineKey, read: &[(u32, u32, Vec<u8>)]) -> Result<(), String> {
        self.begin();
        self.end_rendering();
        let mut regions = Vec::with_capacity(read.len());
        for (i, (w, _, bytes)) in read.iter().enumerate() {
            let offset = self.ring_copy(bytes, 16).ok_or("1D texture too large for the upload ring")?;
            let (layer, level) = (i as u32 / key.levels, i as u32 % key.levels);
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: level, base_array_layer: layer, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: *w, height: 1, depth: 1 }),
            );
        }
        let image = self.lines[&key].image;
        self.barrier();
        unsafe { self.device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, image, vk::ImageLayout::GENERAL, &regions) };
        self.barrier();
        self.stats.uploads += 1;
        Ok(())
    }

    fn line_view(&mut self, key: LineKey, swizzle: [vk::ComponentSwizzle; 4], view_type: vk::ImageViewType) -> Result<vk::ImageView, String> {
        let line = self.lines.get_mut(&key).unwrap();
        if let Some(&view) = line.views.get(&(swizzle, view_type)) {
            return Ok(view);
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(line.image)
            .view_type(view_type)
            .format(key.format)
            .components(vk::ComponentMapping { r: swizzle[0], g: swizzle[1], b: swizzle[2], a: swizzle[3] })
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: key.levels, base_array_layer: 0, layer_count: key.layers });
        let view = unsafe { self.device.create_image_view(&info, None).map_err(|e| format!("1D view: {e:?}"))? };
        line.views.insert((swizzle, view_type), view);
        Ok(view)
    }

    fn create_line(&mut self, key: LineKey) -> Result<Line, vk::Result> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_1D)
            .format(key.format)
            .extent(vk::Extent3D { width: key.width, height: 1, depth: 1 })
            .mip_levels(key.levels)
            .array_layers(key.layers)
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
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: key.levels, base_array_layer: 0, layer_count: key.layers })
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        unsafe {
            self.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &barrier);
        }
        Ok(Line { image, memory, views: HashMap::default(), hash: 0, checked: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::line_levels;

    #[test]
    fn a_line_has_no_more_levels_than_its_width_allows() {
        assert_eq!(line_levels(256, 9), 9);
        assert_eq!(line_levels(256, 12), 9);
        assert_eq!(line_levels(1, 4), 1);
        assert_eq!(line_levels(0, 0), 1);
        assert_eq!(line_levels(300, 3), 3);
    }
}
