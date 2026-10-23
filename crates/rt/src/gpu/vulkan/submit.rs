use ash::vk;

use crate::gpu::{Gpu, buffer_cache};
use crate::mem::GuestMemory;
use super::draw::Bound;
use super::guest_writes::Pending;
use super::images::ImageKey;
use super::pipeline::PIPELINE_CACHE_SAVE;
use super::readback::bytes_per_pixel;
use super::{Renderer, find_memory};

pub(super) const RING_SIZE: u64 = 192 << 20;

pub(super) const IN_FLIGHT: usize = 3;

pub(super) const BATCH_RING_LIMIT: u64 = RING_SIZE / 4;

pub(super) struct Batch {
    pub(super) cmd: vk::CommandBuffer,
    pub(super) fence: vk::Fence,
    pub(super) acquired: vk::Semaphore,
    pub(super) descriptor_pool: vk::DescriptorPool,
    pub(super) serial: u64,
    pub(super) ring_end: u64,
    pub(super) write_back: Vec<(u64, u64, u64)>,
    pub(super) garbage: Vec<Garbage>,
    pub(super) staging: Option<Staging>,
    pub(super) staged: bool,
}

pub(super) struct Staging {
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) ptr: *mut u8,
    pub(super) size: u64,
    pub(super) coherent: bool,
}

pub(super) enum Garbage {
    Image { image: vk::Image, memory: vk::DeviceMemory, views: Vec<vk::ImageView> },
    Block(buffer_cache::Block),
}

pub(super) struct Ring {
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) ptr: *mut u8,
    pub(super) head: u64,
    pub(super) tail: Option<u64>,
    pub(super) recorded: u64,
}

impl Ring {
    pub(super) fn fit(&mut self, size: u64, align: u64) -> Option<u64> {
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

impl Renderer {
    pub(super) fn begin(&mut self) {
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

    pub(super) fn end_rendering(&mut self) {
        if self.rendering.take().is_some() {
            unsafe { self.device.cmd_end_rendering(self.cmd) };
            self.barrier();
        }
    }

    pub(super) fn barrier(&mut self) {
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

    pub(super) fn poll(&mut self) {
        while let Some(&i) = self.in_flight.front() {
            if unsafe { self.device.get_fence_status(self.batches[i].fence) } != Ok(true) {
                return;
            }
            self.retire_oldest();
        }
    }

    pub(super) fn retire_oldest(&mut self) -> bool {
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

    pub(super) fn stage_write_back(&mut self) {
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

    pub(super) fn new_staging(&self, size: u64) -> Option<Staging> {
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

    pub(super) fn copy_pitch_targets(&mut self) {
        if self.pitch_targets.is_empty() {
            return;
        }
        let targets: Vec<(ImageKey, (u64, u32, u32))> = self.pitch_targets.drain().collect();
        self.barrier();
        for (key, (cpu, pitch, rows)) in targets {
            if crate::gpu::trace() {
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

    pub(super) fn resolve_submitted(&mut self, cpu: u64, len: u64) {
        if !self.pending_pages.near(cpu, len) {
            return;
        }
        let serial = self.serial;
        let Some(wait) = self.pending_pages.overlapping(cpu, len).map(|(_, p)| p.2).filter(|&s| s < serial).max() else { return };
        self.stats.resolves += 1;
        while self.pending.iter().any(|p| p.2 <= wait) && self.retire_oldest() {}
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ring_guest(&mut self, gpu: &Gpu, mem: &GuestMemory, va: u64, size: u64, align: u64, zeros: bool) -> Result<u64, String> {
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

    pub(super) fn pending_in_ring(&self, cpu: u64, size: u64, align: u64) -> Option<u64> {
        if self.draw_wrote_storage || !self.pending_pages.near(cpu, size) {
            return None;
        }
        let serial = self.serial;
        let (_, p) = self.pending_pages.overlapping(cpu, size).max_by_key(|e| e.0)?;
        (p.2 == serial && p.0 <= cpu && cpu + size <= p.0 + p.1).then(|| p.3 + (cpu - p.0)).filter(|o| o % align == 0)
    }

    pub(super) fn split(&mut self) {
        if self.ring.recorded > BATCH_RING_LIMIT {
            self.flush();
        }
    }

    pub(super) fn ring_alloc(&mut self, size: u64, align: u64) -> Option<(u64, &mut [u8])> {
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

    pub(super) fn ring_copy(&mut self, data: &[u8], align: u64) -> Option<u64> {
        let (offset, dst) = self.ring_alloc(data.len() as u64, align)?;
        dst[..data.len()].copy_from_slice(data);
        Some(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
