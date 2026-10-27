use ash::vk;
use kirikae_maxwell::{Stage, TexDim};

use crate::gpu::threed::CB_SLOTS;
use crate::gpu::{Gpu, buffer_cache, tic};
use crate::mem::GuestMemory;
use super::Renderer;
use super::draw::DrawScratch;
use super::pipeline::ShaderModule;

pub(super) const MAX_CBUF: u64 = 0x1_0000;

pub(super) const MAX_STORAGE: u64 = 64 << 20;

pub(super) const VERTEX_BASE: u32 = 0;

pub(super) const FRAGMENT_BASE: u32 = 128;

pub(super) const STORAGE_OFFSET: u32 = 18;

pub(super) const TEXTURE_OFFSET: u32 = 40;

pub(super) const IMAGE_OFFSET: u32 = 104;

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn descriptors(
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
    pub(super) fn shader_descriptors(
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
                    if crate::gpu::trace() {
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
            if crate::gpu::trace() {
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
                    if t.dim == TexDim::D3 || t.shadow {
                        return Err(format!("texture: {why}"));
                    }
                    let view = if t.dim == TexDim::D1 { self.line_placeholder(t.array)? } else { self.placeholder(view_type)? };
                    (view, self.sampler(&tic::Sampler::default()))
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

    pub(super) fn shader_bindings(s: &ShaderModule, base: u32, bindings: &mut Vec<(u32, vk::DescriptorType)>) {
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

    pub(super) fn bind_descriptors(&mut self, point: vk::PipelineBindPoint, set_layout: vk::DescriptorSetLayout, layout: vk::PipelineLayout, descriptors: &[Descriptor], scratch: &mut DrawScratch) {
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
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Descriptor {
    Buffer(u32, vk::DescriptorType, vk::Buffer, u64, u64),
    Image(u32, vk::ImageView),
    Sampler(u32, vk::Sampler),
    StorageImage(u32, vk::ImageView),
}
