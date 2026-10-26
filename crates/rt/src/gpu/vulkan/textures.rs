use ash::vk;
use kirikae_maxwell::TexDim;

use crate::gpu::formats::TexFormat;
use crate::gpu::threed::{CB_SLOTS, address, reg};
use crate::gpu::tic::{Layout, TextureHeader};
use crate::gpu::{Gpu, formats, tic};
use crate::mem::GuestMemory;
use super::images::{IDENTITY, ImageKey, aspect_of, same_class};
use super::readback::bytes_per_pixel;
use super::tiling::{deswizzle, layer_alignment, layer_size, level_offset};
use super::{Renderer, hash64};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct TexKey {
    pub(super) tic: [u32; 8],
    pub(super) tsc: [u32; 8],
    pub(super) dim: TexDim,
    pub(super) array: bool,
    pub(super) shadow: bool,
}

#[derive(Clone, Copy)]
pub(super) struct TexEntry {
    pub(super) image: ImageKey,
    pub(super) view: vk::ImageView,
    pub(super) sampler: vk::Sampler,
    pub(super) refresh: Option<ImageKey>,
    pub(super) gather: Option<vk::Format>,
    pub(super) crop: Option<ImageKey>,
    pub(super) twin: Option<(ImageKey, vk::Format)>,
}

#[derive(Clone, Copy)]
pub(super) struct Source {
    pub(super) hash: u64,
    pub(super) checked: u64,
    pub(super) volatile: bool,
}

pub(super) const SOURCE_RECHECK: u64 = 30;

pub(super) const CROP: u64 = 1 << 62;

pub(super) const GATHER: u64 = 1 << 61;

pub(super) const TWIN: u64 = 1 << 60;

pub(super) const FEEDBACK: u64 = 1 << 59;

pub(super) const DERIVED: u64 = CROP | GATHER | TWIN | FEEDBACK;

pub(super) const CROP_SIZES: usize = 3;

impl Renderer {
    pub(super) fn forget_textures(&mut self, cpu: u64, layered: bool) {
        let end = if layered { cpu.saturating_add(1 << 32) } else { cpu };
        let found: Vec<u64> = self.tex_at.range(cpu..=end).map(|(&a, _)| a).collect();
        for a in found {
            for k in self.tex_at.remove(&a).unwrap_or_default() {
                self.tex_cache.remove(&k);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn texture(&mut self, gpu: &Gpu, mem: &GuestMemory, regs: &[u32], cbufs: [Option<(u64, u32)>; CB_SLOTS], slot: Option<u32>, handle: u32, dim: TexDim, array: bool, shadow: bool) -> Result<(vk::ImageView, vk::Sampler), String> {
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
        if crate::gpu::trace() {
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
                entry.refresh = Some(key);
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
    pub(super) fn upload_texture(&mut self, gpu: &Gpu, mem: &GuestMemory, key: ImageKey, header: &TextureHeader, format: &TexFormat, levels: u32, layers: u32) {
        if !self.pending.is_empty()
            && let (Some(cpu), Some(size)) = (gpu.translate(header.address), layer_size(header, format, levels))
        {
            self.resolve_submitted(cpu, size * layers as u64);
        }
        let mut data = read_texture(gpu, mem, header, format, levels, layers);
        if format.astc {
            for (w, h, level) in &mut data {
                *level = crate::gpu::astc::decode(level, *w, *h, format.block_w, format.block_h);
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

    pub(super) fn refresh_texture(&mut self, gpu: &Gpu, mem: &GuestMemory, key: ImageKey, header: &TextureHeader, format: &TexFormat, layers: u32) {
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

    pub(super) fn crop(&mut self, src: ImageKey, width: u32, height: u32) -> Result<ImageKey, String> {
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

    pub(super) fn feedback(&mut self, src: ImageKey) -> Result<ImageKey, String> {
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

    pub(super) fn depth_twin(&mut self, src: ImageKey, format: vk::Format) -> Result<ImageKey, String> {
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gather(&mut self, gpu: &Gpu, mem: &GuestMemory, cpu: u64, header: &TextureHeader, format: &TexFormat, like: vk::Format, layers: u32) -> Result<ImageKey, String> {
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
                if crate::gpu::trace() {
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

    pub(super) fn placeholder(&mut self, view_type: vk::ImageViewType) -> Result<vk::ImageView, String> {
        let key = ImageKey { cpu: u64::MAX - 1, format: vk::Format::R8G8B8A8_UNORM, width: 1, height: 1 };
        if !self.images.contains_key(&key) {
            self.create_image(key, 1, 6).map_err(|e| format!("image: {e:?}"))?;
            self.upload_layers(key, 1, &vec![(1, 1, vec![0; 4]); 6]);
        }
        Ok(self.view_of(key, key.format, IDENTITY, view_type))
    }

    pub(super) fn note_placeholder(&mut self, why: &str) {
        let why = format!("placeholder for {why}");
        if crate::gpu::trace() {
            eprintln!("gpu: {why}");
        }
        let reasons = &mut self.stats.placeholders;
        if reasons.len() < 64 || reasons.contains_key(&why) {
            *reasons.entry(why).or_default() += 1;
        }
    }

    pub(super) fn sampler(&mut self, s: &tic::Sampler) -> vk::Sampler {
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
}

pub(super) fn depth_twin_format(f: vk::Format) -> Option<vk::Format> {
    match f {
        vk::Format::R16_UNORM => Some(vk::Format::D16_UNORM),
        vk::Format::R32_SFLOAT => Some(vk::Format::D32_SFLOAT),
        _ => None,
    }
}

pub(super) fn read_texture(gpu: &Gpu, mem: &GuestMemory, h: &TextureHeader, f: &TexFormat, levels: u32, layers: u32) -> Vec<(u32, u32, Vec<u8>)> {
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

pub(super) fn texture_hash(gpu: &Gpu, mem: &GuestMemory, h: &TextureHeader, f: &TexFormat, levels: u32, layers: u32) -> Option<u64> {
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

pub(super) fn texture_source(gpu: &Gpu, mem: &GuestMemory, h: &TextureHeader, f: &TexFormat) -> Option<Vec<u8>> {
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

pub(super) fn read_levels(gpu: &Gpu, mem: &GuestMemory, h: &TextureHeader, f: &TexFormat, levels: u32, offset: &mut u64, out: &mut Vec<(u32, u32, Vec<u8>)>) {
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

pub(super) fn border_color(border: &[u32; 4]) -> vk::BorderColor {
    let [r, g, b, a] = border.map(f32::from_bits);
    let rgb = (r + g + b) / 3.0;
    match (rgb >= 0.5, a >= 0.5) {
        (true, _) => vk::BorderColor::FLOAT_OPAQUE_WHITE,
        (false, true) => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        (false, false) => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
    }
}
