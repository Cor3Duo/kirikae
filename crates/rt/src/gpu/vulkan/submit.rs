use ash::vk;
use ash::vk::Handle as _;

use super::draw::Bound;
use super::guest_writes::Pending;
use super::images::ImageKey;
use super::pipeline::PIPELINE_CACHE_SAVE;
use super::readback::bytes_per_pixel;
use super::{Renderer, find_memory};
use crate::gpu::{Gpu, buffer_cache};
use crate::mem::GuestMemory;

pub(super) const RING_SIZE: u64 = 192 << 20;

pub(super) const IN_FLIGHT: usize = 3;

pub(super) const BATCH_RING_LIMIT: u64 = RING_SIZE / 4;

/// Exit status used when rendering cannot continue (a lost device or a failed submission).
/// The discovery runner records a non-zero exit as a crash instead of waiting for its bound.
pub(super) const DEVICE_LOST_EXIT: i32 = 70;

/// The one fatal report of a renderer that cannot submit any more. The message is built
/// lazily (it walks the resource maps) and logged a single time.
#[derive(Default)]
pub(super) struct LossReport {
    report: Option<String>,
}

impl LossReport {
    pub(super) fn is_lost(&self) -> bool {
        self.report.is_some()
    }

    pub(super) fn report(&self) -> Option<&str> {
        self.report.as_deref()
    }

    pub(super) fn record(&mut self, build: impl FnOnce() -> String) -> bool {
        if self.report.is_some() {
            return false;
        }
        self.report = Some(build());
        true
    }
}

/// The last command buffer handed to the queue, kept for the fatal report.
#[derive(Clone, Copy)]
pub(super) struct LastSubmit {
    pub(super) serial: u64,
    pub(super) index: usize,
    pub(super) cmd: vk::CommandBuffer,
    pub(super) ring_end: u64,
    pub(super) write_backs: usize,
    pub(super) write_back_bytes: u64,
    pub(super) staging: u64,
    pub(super) staged: bool,
    pub(super) garbage: usize,
}

impl Default for LastSubmit {
    fn default() -> Self {
        LastSubmit {
            serial: 0,
            index: 0,
            cmd: vk::CommandBuffer::null(),
            ring_end: 0,
            write_backs: 0,
            write_back_bytes: 0,
            staging: 0,
            staged: false,
            garbage: 0,
        }
    }
}

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
    Image {
        image: vk::Image,
        memory: vk::DeviceMemory,
        views: Vec<vk::ImageView>,
    },
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
    pub(super) fn lost(&self) -> bool {
        self.lost.is_lost()
    }

    /// Records the first fatal Vulkan failure (the device lost, or a submission that cannot
    /// be recovered) with the last command buffer and the resource sizes, then stops the
    /// process with a non-zero code so the runner sees a crash instead of hanging on a fence
    /// that will never signal. Under `cfg(test)` it only records, so the stop path is
    /// testable without killing the test harness.
    fn fatal(&mut self, at: &str, e: vk::Result) -> bool {
        if self.lost.is_lost() {
            return true;
        }
        let what = if e == vk::Result::ERROR_DEVICE_LOST {
            "GPU device lost"
        } else {
            "Vulkan submission failed"
        };
        let report = self.loss_report(what, at, e);
        if self.lost.record(|| report)
            && let Some(report) = self.lost.report()
        {
            eprintln!("{report}");
        }
        if !cfg!(test) {
            std::process::exit(DEVICE_LOST_EXIT);
        }
        true
    }

    /// Handles an error that is only fatal when the device is gone; other errors keep the
    /// old behaviour of the caller.
    pub(super) fn note_lost(&mut self, at: &str, e: vk::Result) {
        if e == vk::Result::ERROR_DEVICE_LOST {
            self.fatal(at, e);
        }
    }

    fn loss_report(&self, what: &str, at: &str, e: vk::Result) -> String {
        let last = self.last_submit;
        let tail = self
            .ring
            .tail
            .map_or_else(|| "none".to_string(), |t| format!("{t:#x}"));
        let mut image_bytes = 0u64;
        let mut views = 0usize;
        for (key, image) in &self.images {
            let bpp = bytes_per_pixel(key.format).map_or(0, u64::from);
            image_bytes +=
                u64::from(key.width) * u64::from(key.height) * bpp * u64::from(image.layers);
            views += image.views.len();
        }
        let staging: u64 = self
            .batches
            .iter()
            .filter_map(|b| b.staging.as_ref())
            .map(|s| s.size)
            .sum();
        let readback = self.readback.map_or(0, |(_, _, size, _)| size);
        let (vram, host) = self.buffers.memory();
        let in_flight: Vec<usize> = self.in_flight.iter().copied().collect();
        let s = &self.stats;
        format!(
            "kirikae: {what} ({e:?}) at {at}; rendering stops and the process exits with code \
             {DEVICE_LOST_EXIT} so the run cannot hang\n\
             kirikae:   last submit: serial {} batch {} cmd {:#x} ring_end {:#x}; {} write-backs \
             ({} bytes), staging {} bytes (staged {}), {} blocks freed\n\
             kirikae:   state: ring {RING_SIZE:#x} head {:#x} tail {tail} recorded {}; in flight \
             {in_flight:?}; descriptor pools {}; images {} (~{} bytes, {} views); volumes {}; \
             buffer cache {vram}/{host} MiB; readback {} bytes; staging {} bytes\n\
             kirikae:   stats: {} batches, {} draws, {} dispatches, {} uploads, {} presents, {} \
             waits, {} resolves; {} ring bytes, {} bytes written back",
            last.serial,
            last.index,
            last.cmd.as_raw(),
            last.ring_end,
            last.write_backs,
            last.write_back_bytes,
            last.staging,
            last.staged,
            last.garbage,
            self.ring.head,
            self.ring.recorded,
            self.batches.len(),
            self.images.len(),
            image_bytes,
            views,
            self.volumes.len(),
            readback,
            staging,
            s.batches,
            s.draws,
            s.dispatches,
            s.uploads,
            s.presents,
            s.waits,
            s.resolves,
            s.ring_bytes,
            s.written_back,
        )
    }

    pub(super) fn begin(&mut self) {
        if self.lost.is_lost() {
            self.recording = false;
            return;
        }
        if self.recording {
            return;
        }
        let begin = unsafe {
            self.device.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        };
        if let Err(e) = begin {
            self.fatal("vkBeginCommandBuffer", e);
            return;
        }
        self.recording = true;
        self.bound = Bound::default();
        if let Some((q, _)) = self.timestamps {
            let first = 2 * self.current as u32;
            unsafe {
                self.device.cmd_reset_query_pool(self.cmd, q, first, 2);
                self.device.cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    q,
                    first,
                );
            }
        }
        self.barrier();
    }

    pub(super) fn end_rendering(&mut self) {
        if self.lost.is_lost() {
            self.rendering = None;
            return;
        }
        if self.rendering.take().is_some() {
            unsafe { self.device.cmd_end_rendering(self.cmd) };
            self.barrier();
        }
    }

    pub(super) fn barrier(&mut self) {
        self.draw_wrote_storage = false;
        if self.lost.is_lost() {
            return;
        }
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            );
        }
    }

    pub fn flush(&mut self) {
        if self.lost.is_lost() {
            self.recording = false;
            return;
        }
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
        if self.lost.is_lost() {
            self.recording = false;
            return;
        }
        let cmd = self.cmd;
        let index = self.current;
        let serial = self.serial;
        let ring_end = self.ring.head;
        let write_backs = self.write_back.len();
        let write_back_bytes: u64 = self.write_back.iter().map(|w| w.2).sum();
        let staging = self.batches[index].staging.as_ref().map_or(0, |s| s.size);
        let staged = self.batches[index].staged;
        let garbage = self.garbage.len();
        {
            let batch = &mut self.batches[index];
            batch.serial = serial;
            batch.ring_end = ring_end;
            batch.write_back = std::mem::take(&mut self.write_back);
            batch.garbage = std::mem::take(&mut self.garbage);
        }
        let fence = self.batches[index].fence;
        if let Some((q, _)) = self.timestamps {
            unsafe {
                self.device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    q,
                    2 * index as u32 + 1,
                );
            }
        }
        if let Err(e) = unsafe { self.device.end_command_buffer(cmd) } {
            self.recording = false;
            self.fatal("vkEndCommandBuffer", e);
            return;
        }
        let cmds = [cmd];
        let waits: Vec<vk::Semaphore> = self.submit_wait.take().into_iter().collect();
        let stages = vec![vk::PipelineStageFlags::TRANSFER; waits.len()];
        let signals: Vec<vk::Semaphore> = self.submit_signal.take().into_iter().collect();
        let submit = [vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .signal_semaphores(&signals)];
        if let Err(e) = unsafe { self.device.queue_submit(self.queue, &submit, fence) } {
            self.recording = false;
            self.fatal("vkQueueSubmit", e);
            return;
        }
        self.last_submit = LastSubmit {
            serial,
            index,
            cmd,
            ring_end,
            write_backs,
            write_back_bytes,
            staging,
            staged,
            garbage,
        };
        self.recording = false;
        self.in_flight.push_back(index);
        self.serial += 1;
        self.stats.batches += 1;
        self.storage.clear();
        self.checked.clear();
        self.cp_writes.clear();
        self.ring.recorded = 0;
        self.current = (self.current + 1) % IN_FLIGHT;
        while self.in_flight.contains(&self.current) {
            if !self.retire_oldest() {
                break;
            }
        }
        self.cmd = self.batches[self.current].cmd;
        self.descriptor_pool = self.batches[self.current].descriptor_pool;
        self.poll();
        if self.stats.pipelines > self.cache_saved.0
            && self.cache_saved.1.elapsed() > PIPELINE_CACHE_SAVE
        {
            self.save_pipeline_cache(false);
        }
    }

    pub fn finish(&mut self) {
        if self.lost.is_lost() {
            return;
        }
        self.flush();
        while self.retire_oldest() {}
    }

    pub(super) fn poll(&mut self) {
        if self.lost.is_lost() {
            return;
        }
        while let Some(&i) = self.in_flight.front() {
            let status = unsafe { self.device.get_fence_status(self.batches[i].fence) };
            match status {
                Ok(true) => {}
                Ok(false) => return,
                Err(e) => {
                    self.fatal("vkGetFenceStatus", e);
                    return;
                }
            }
            if !self.retire_oldest() {
                return;
            }
        }
    }

    pub(super) fn retire_oldest(&mut self) -> bool {
        if self.lost.is_lost() {
            self.in_flight.clear();
            return false;
        }
        let Some(i) = self.in_flight.pop_front() else {
            return false;
        };
        let fence = self.batches[i].fence;
        match unsafe { self.device.get_fence_status(fence) } {
            Ok(true) => {}
            Ok(false) => {
                let start = std::time::Instant::now();
                if let Err(e) = unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) } {
                    self.fatal("vkWaitForFences", e);
                    return false;
                }
                self.stats.flush_time += start.elapsed();
                self.stats.waits += 1;
            }
            Err(e) => {
                self.fatal("vkGetFenceStatus", e);
                return false;
            }
        }
        if let Some((q, period)) = self.timestamps {
            let mut ticks = [0u64; 2];
            if unsafe {
                self.device.get_query_pool_results(
                    q,
                    2 * i as u32,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
            }
            .is_ok()
            {
                self.stats.gpu_time += std::time::Duration::from_nanos(
                    (ticks[1].saturating_sub(ticks[0]) as f64 * period) as u64,
                );
            }
        }
        if let Err(e) = unsafe { self.device.reset_fences(&[fence]) } {
            self.fatal("vkResetFences", e);
            return false;
        }
        if let Err(e) = unsafe {
            self.device
                .reset_command_buffer(self.batches[i].cmd, vk::CommandBufferResetFlags::empty())
        } {
            self.fatal("vkResetCommandBuffer", e);
            return false;
        }
        if let Err(e) = unsafe {
            self.device.reset_descriptor_pool(
                self.batches[i].descriptor_pool,
                vk::DescriptorPoolResetFlags::empty(),
            )
        } {
            self.fatal("vkResetDescriptorPool", e);
            return false;
        }
        let batch = &mut self.batches[i];
        let (serial, ring_end) = (batch.serial, batch.ring_end);
        let write_back = std::mem::take(&mut batch.write_back);
        let garbage = std::mem::take(&mut batch.garbage);
        let from = match (&batch.staging, std::mem::take(&mut batch.staged)) {
            (Some(st), true) => {
                if !st.coherent {
                    let range = [vk::MappedMemoryRange::default()
                        .memory(st.memory)
                        .size(vk::WHOLE_SIZE)];
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
                let data =
                    unsafe { std::slice::from_raw_parts(from.add(offset as usize), size as usize) };
                mem.write(cpu, data);
                self.note_cp_write(cpu, size);
                if let Some(dir) = crate::env_var!("KIRIKAE_SSBO_DUMP") {
                    let n = self.stats.dispatches + self.stats.draws;
                    let _ = std::fs::write(
                        std::path::Path::new(&dir).join(format!("{n:08}_{cpu:x}.bin")),
                        &data[..data.len().min(0x1000)],
                    );
                }
            }
        }
        self.pending_pages.retire(serial);
        self.pending.retain(|p| p.2 != serial);
        self.pending_flag.store(
            !self.pending.is_empty(),
            std::sync::atomic::Ordering::Relaxed,
        );
        for g in garbage {
            match g {
                Garbage::Image {
                    image,
                    memory,
                    views,
                } => {
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
        self.ring.tail = if self.in_flight.is_empty() && self.ring.recorded == 0 {
            None
        } else {
            Some(ring_end)
        };
        true
    }

    pub(super) fn stage_write_back(&mut self) {
        let total: u64 = self
            .write_back
            .iter()
            .map(|w| w.2.next_multiple_of(16))
            .sum();
        if total == 0 {
            return;
        }
        let i = self.current;
        if self.batches[i]
            .staging
            .as_ref()
            .is_none_or(|st| st.size < total)
        {
            if let Some(st) = self.batches[i].staging.take() {
                unsafe {
                    self.device.destroy_buffer(st.buffer, None);
                    self.device.free_memory(st.memory, None);
                }
            }
            self.batches[i].staging = self.new_staging(total.next_power_of_two().max(1 << 20));
        }
        let Some(buffer) = self.batches[i].staging.as_ref().map(|st| st.buffer) else {
            return;
        };
        let mut at = 0;
        let regions: Vec<vk::BufferCopy> = self
            .write_back
            .iter_mut()
            .map(|w| {
                let r = vk::BufferCopy {
                    src_offset: w.1,
                    dst_offset: at,
                    size: w.2,
                };
                w.1 = at;
                at += w.2.next_multiple_of(16);
                r
            })
            .collect();
        self.end_rendering();
        self.barrier();
        let host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        unsafe {
            self.device
                .cmd_copy_buffer(self.cmd, self.ring.buffer, buffer, &regions);
            self.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &host,
                &[],
                &[],
            );
        }
        self.batches[i].staged = true;
    }

    pub(super) fn new_staging(&self, size: u64) -> Option<Staging> {
        unsafe {
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST),
                    None,
                )
                .ok()?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let ty = find_memory(
                &self.mem_props,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .or_else(|| {
                find_memory(
                    &self.mem_props,
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
                )
            });
            let Some(ty) = ty else {
                self.device.destroy_buffer(buffer, None);
                return None;
            };
            let coherent = self.mem_props.memory_types[ty as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .ok()?;
            self.device.bind_buffer_memory(buffer, memory, 0).ok()?;
            let ptr = self
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .ok()? as *mut u8;
            Some(Staging {
                buffer,
                memory,
                ptr,
                size,
                coherent,
            })
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
                eprintln!(
                    "gpu: pitch target {cpu:#x} {:?} {}x{} pitch {pitch:#x} rows {rows}",
                    key.format, key.width, key.height
                );
            }
            let Some(bpp) = bytes_per_pixel(key.format) else {
                continue;
            };
            if matches!(
                key.format,
                vk::Format::R16G16_UNORM
                    | vk::Format::R16G16_SNORM
                    | vk::Format::R32_UINT
                    | vk::Format::R32_SINT
                    | vk::Format::R16G16B16A16_UNORM
                    | vk::Format::R16G16B16A16_SNORM
                    | vk::Format::R16G16B16A16_UINT
                    | vk::Format::R16G16B16A16_SINT
            ) {
                continue;
            }
            let (width, rows) = (key.width.min(pitch / bpp), rows.min(key.height));
            if !self.images.contains_key(&key) || width == 0 || rows == 0 || pitch % bpp != 0 {
                continue;
            }
            let size = pitch as u64 * (rows - 1) as u64 + (width * bpp) as u64;
            let Some((offset, _)) = self.ring_alloc(size, 16) else {
                continue;
            };
            let region = [vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .buffer_row_length(pitch / bpp)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height: rows,
                    depth: 1,
                })];
            let image = self.images[&key].image;
            unsafe {
                self.device.cmd_copy_image_to_buffer(
                    self.cmd,
                    image,
                    vk::ImageLayout::GENERAL,
                    self.ring.buffer,
                    &region,
                )
            };
            self.write_back.push((cpu, offset, size));
        }
        self.barrier();
    }

    pub fn resolve(&mut self, cpu: u64, len: u64) {
        if !self.pending_pages.near(cpu, len) {
            return;
        }
        if self
            .pending_pages
            .overlapping(cpu, len)
            .any(|(_, p)| p.2 == self.serial)
        {
            self.flush();
        }
        self.resolve_submitted(cpu, len);
    }

    pub(super) fn resolve_submitted(&mut self, cpu: u64, len: u64) {
        if !self.pending_pages.near(cpu, len) {
            return;
        }
        let serial = self.serial;
        let Some(wait) = self
            .pending_pages
            .overlapping(cpu, len)
            .map(|(_, p)| p.2)
            .filter(|&s| s < serial)
            .max()
        else {
            return;
        };
        self.stats.resolves += 1;
        while self.pending.iter().any(|p| p.2 <= wait) && self.retire_oldest() {}
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ring_guest(
        &mut self,
        gpu: &Gpu,
        mem: &GuestMemory,
        va: u64,
        size: u64,
        align: u64,
        zeros: bool,
    ) -> Result<u64, String> {
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
            None => {
                return Err(format!(
                    "guest memory at GPU address {va:#x} ({size:#x} bytes) unreadable"
                ));
            }
        };
        let offset = offset.ok_or("ring full")?;
        let serial = self.serial;
        if let Some(c) = cpu
            && self.pending_pages.near(c, size)
        {
            let mut found: Vec<(u64, Pending)> = self
                .pending_pages
                .overlapping(c, size)
                .filter(|(_, p)| p.2 == serial && p.3 != offset)
                .copied()
                .collect();
            found.sort_unstable_by_key(|e| e.0);
            found.dedup_by_key(|e| e.0);
            let copies: Vec<vk::BufferCopy> = found
                .iter()
                .map(|(_, p)| {
                    let (start, end) = (p.0.max(c), (p.0 + p.1).min(c + size));
                    vk::BufferCopy {
                        src_offset: p.3 + (start - p.0),
                        dst_offset: offset + (start - c),
                        size: end - start,
                    }
                })
                .collect();
            if !copies.is_empty() {
                self.stats.patches += 1;
                self.end_rendering();
                self.barrier();
                unsafe {
                    self.device.cmd_copy_buffer(
                        self.cmd,
                        self.ring.buffer,
                        self.ring.buffer,
                        &copies,
                    )
                };
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
        let (_, p) = self
            .pending_pages
            .overlapping(cpu, size)
            .max_by_key(|e| e.0)?;
        (p.2 == serial && p.0 <= cpu && cpu + size <= p.0 + p.1)
            .then(|| p.3 + (cpu - p.0))
            .filter(|o| o % align == 0)
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
        let slice = unsafe {
            std::slice::from_raw_parts_mut(self.ring.ptr.add(offset as usize), size as usize)
        };
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
    use super::super::Presented;
    use super::*;

    #[test]
    fn ring_keeps_data_in_use() {
        let mut ring = Ring {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            ptr: std::ptr::null_mut(),
            head: 0,
            tail: None,
            recorded: 0,
        };
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

    #[test]
    fn the_loss_report_is_built_and_logged_once() {
        let mut lost = LossReport::default();
        let mut builds = 0;
        assert!(lost.record(|| {
            builds += 1;
            "first report".to_string()
        }));
        assert!(!lost.record(|| {
            builds += 1;
            "second report".to_string()
        }));
        assert_eq!(builds, 1);
        assert!(lost.is_lost());
        assert_eq!(lost.report(), Some("first report"));
    }

    /// Submits many batches through the real path, then injects a lost device and checks
    /// that the renderer stops submitting and presenting instead of waiting on a fence that
    /// cannot signal. Every batch must hand its fence back unsignaled (no leak and no reuse
    /// while still signalled). Needs a Vulkan 1.3 device; ignored by default.
    #[test]
    #[ignore = "needs a Vulkan 1.3 device; run with --ignored"]
    fn submission_soak_survives_a_lost_device() {
        let mut r = match Renderer::new() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: no Vulkan device ({e})");
                return;
            }
        };
        for frame in 0..300u64 {
            r.begin();
            let offset = r
                .ring_copy(&[frame as u8; 8192], 16)
                .expect("ring has space");
            r.write_back.push((0x1000, offset, 8192));
            let key = ImageKey {
                cpu: 0x1_0000 + frame,
                format: vk::Format::R8G8B8A8_UNORM,
                width: 4,
                height: 4,
            };
            r.create_image(key, 1, 1).expect("image");
            r.destroy_image(key);
            r.flush();
            if frame % 4 == 0 {
                r.poll();
            }
        }
        r.finish();
        assert!(r.in_flight.is_empty());
        assert!(!r.recording);
        assert!(!r.lost());
        assert!(r.images.is_empty(), "every image was destroyed");
        assert!(r.garbage.is_empty(), "every freed block was retired");
        for batch in &r.batches {
            assert_eq!(unsafe { r.device.get_fence_status(batch.fence) }, Ok(false));
        }
        let batches = r.stats.batches;
        assert!(r.fatal("test", vk::Result::ERROR_DEVICE_LOST));
        assert!(r.lost());
        r.flush();
        assert_eq!(r.stats.batches, batches, "a lost device cannot submit");
        assert!(matches!(
            r.present(0, 16, 16, false, false),
            Presented::NoSwapchain
        ));
        r.finish();
    }
}
