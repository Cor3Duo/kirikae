use ash::vk;

use super::Renderer;
use super::images::{ImageKey, aspect_of};
use super::readback::decode_pixels;

pub(super) struct Swapchain {
    pub(super) surface: vk::SurfaceKHR,
    pub(super) handle: vk::SwapchainKHR,
    pub(super) images: Vec<vk::Image>,
    pub(super) extent: vk::Extent2D,
    pub(super) srgb: bool,
    pub(super) done: Vec<vk::Semaphore>,
    pub(super) stale: bool,
    pub(super) format: vk::Format,
    pub(super) readable: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Presented {
    Shown,
    NoImage,
    NoSwapchain,
}

pub(super) const PRESENT: u64 = 1 << 60;

impl Renderer {
    pub fn present(
        &mut self,
        cpu: u64,
        width: u32,
        height: u32,
        flip_h: bool,
        flip_v: bool,
    ) -> Presented {
        if self.swapchain_fn.is_none() || self.present_failed {
            return Presented::NoSwapchain;
        }
        let Some(key) = self
            .find_image(cpu, width, height, vk::Format::R8G8B8A8_UNORM)
            .filter(|k| {
                self.images[k].gpu_dirty && aspect_of(k.format) == vk::ImageAspectFlags::COLOR
            })
        else {
            self.note_present_miss(cpu, width, height);
            return Presented::NoImage;
        };
        self.present_image(key, width, height, flip_h, flip_v)
    }

    /// Explains the first present of an address the GPU never wrote to. Without
    /// this the window just stays black and nothing in the log says why.
    fn note_present_miss(&mut self, cpu: u64, width: u32, height: u32) {
        if !self.present_misses.insert(cpu) {
            return;
        }
        match self.find_image(cpu, width, height, vk::Format::R8G8B8A8_UNORM) {
            Some(k) => eprintln!(
                "kirikae: present: the image for {cpu:#x} ({:?} {}x{}) was never drawn to; the frame falls back to guest memory and the window can be black",
                k.format, k.width, k.height
            ),
            None => eprintln!(
                "kirikae: present: no image for {cpu:#x} {width}x{height}; the frame falls back to guest memory and the window can be black"
            ),
        }
    }

    pub fn present_pixels(&mut self, width: u32, height: u32, pixels: &[u32]) -> Presented {
        if self.swapchain_fn.is_none()
            || self.present_failed
            || pixels.len() != (width * height) as usize
        {
            return Presented::NoSwapchain;
        }
        let key = ImageKey {
            cpu: PRESENT,
            format: vk::Format::B8G8R8A8_UNORM,
            width,
            height,
        };
        if !self.images.contains_key(&key) {
            let old: Vec<ImageKey> = self
                .images
                .keys()
                .copied()
                .filter(|k| k.cpu == PRESENT)
                .collect();
            for k in old {
                self.destroy_image(k);
            }
            if self.create_image(key, 1, 1).is_err() {
                return Presented::NoSwapchain;
            }
        }
        let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
        self.upload(key, &[(width, height, bytes)]);
        self.present_image(key, width, height, false, false)
    }

    pub(super) fn present_image(
        &mut self,
        key: ImageKey,
        width: u32,
        height: u32,
        flip_h: bool,
        flip_v: bool,
    ) -> Presented {
        let srgb = matches!(
            key.format,
            vk::Format::R8G8B8A8_SRGB
                | vk::Format::B8G8R8A8_SRGB
                | vk::Format::A8B8G8R8_SRGB_PACK32
        );
        let dump = crate::env_var!("KIRIKAE_PRESENT_DUMP")
            .filter(|_| self.stats.presents.is_multiple_of(60));
        let capture = crate::gpu::capture::due();
        let readback = match self.swapchain.as_ref().map(|sc| sc.extent) {
            Some(extent) if (dump.is_some() || capture.is_some()) && extent.width > 0 => self
                .readback_buffer(extent.width as u64 * extent.height as u64 * 4)
                .map(|(b, _)| b),
            _ => None,
        };
        let Some(index) = self.acquire(srgb) else {
            return if self.swapchain.is_some() {
                Presented::Shown
            } else {
                Presented::NoSwapchain
            };
        };
        let sc = self.swapchain.as_ref().unwrap();
        let (target, extent, done) = (
            sc.images[index as usize],
            sc.extent,
            sc.done[index as usize],
        );
        self.begin();
        self.end_rendering();
        self.barrier();
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_dst = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .image(target)
            .subresource_range(range)];
        let (w, h) = (width.min(key.width) as i32, height.min(key.height) as i32);
        let (x0, x1) = if flip_h {
            (extent.width as i32, 0)
        } else {
            (0, extent.width as i32)
        };
        let (y0, y1) = if flip_v {
            (extent.height as i32, 0)
        } else {
            (0, extent.height as i32)
        };
        let sub = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = [vk::ImageBlit::default()
            .src_subresource(sub)
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D { x: w, y: h, z: 1 },
            ])
            .dst_subresource(sub)
            .dst_offsets([
                vk::Offset3D { x: x0, y: y0, z: 0 },
                vk::Offset3D { x: x1, y: y1, z: 1 },
            ])];
        let to_present = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .image(target)
            .subresource_range(range)];
        let source = self.images[&key].image;
        let readback = readback.filter(|_| {
            self.swapchain
                .as_ref()
                .is_some_and(|sc| sc.readable && sc.extent == extent)
        });
        unsafe {
            let d = &self.device;
            d.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_dst,
            );
            d.cmd_blit_image(
                self.cmd,
                source,
                vk::ImageLayout::GENERAL,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
                vk::Filter::LINEAR,
            );
            if let Some(buffer) = readback {
                let to_src = [vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .image(target)
                    .subresource_range(range)];
                d.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_src,
                );
                let copy = [vk::BufferImageCopy::default()
                    .image_subresource(sub)
                    .image_extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })];
                d.cmd_copy_image_to_buffer(
                    self.cmd,
                    target,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer,
                    &copy,
                );
            }
            let from = if readback.is_some() {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            } else {
                vk::ImageLayout::TRANSFER_DST_OPTIMAL
            };
            let to_present = [to_present[0].old_layout(from)];
            d.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_present,
            );
        }
        self.submit_wait = Some(self.batches[self.current].acquired);
        self.submit_signal = Some(done);
        self.flush();
        let sc = self.swapchain.as_mut().unwrap();
        let (waits, swapchains, indices) = ([done], [sc.handle], [index]);
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        match unsafe {
            self.swapchain_fn
                .as_ref()
                .unwrap()
                .queue_present(self.queue, &info)
        } {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => sc.stale = true,
            Err(e) => {
                eprintln!("kirikae: presenting failed ({e:?}); the swapchain is made again");
                sc.stale = true;
            }
        }
        let format = sc.format;
        self.stats.presents += 1;
        if readback.is_some() {
            self.finish();
            if let Some((_, memory, _, ptr)) = self.readback {
                if !self.readback_coherent {
                    let range = [vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)];
                    unsafe { self.device.invalidate_mapped_memory_ranges(&range).ok() };
                }
                let bytes = unsafe {
                    std::slice::from_raw_parts(ptr, (extent.width * extent.height * 4) as usize)
                };
                if let Some(pixels) = decode_pixels(format, bytes) {
                    if let Some(c) = capture.filter(|c| c.claim()) {
                        let _ = c.write(c.frame(), extent.width, extent.height, &pixels);
                    }
                    if let Some(dir) = dump {
                        let path = std::path::Path::new(&dir)
                            .join(format!("present{:06}.png", self.stats.presents));
                        let _ =
                            crate::display::write_png(&path, extent.width, extent.height, &pixels);
                    }
                }
            }
        }
        if let Some(c) = capture.filter(|c| !c.done() && c.claim()) {
            eprintln!(
                "kirikae: capture: the swapchain image is not readable; capturing the image that was blitted"
            );
            self.capture_key(Some(key), width, height, c);
        }
        Presented::Shown
    }

    pub(super) fn acquire(&mut self, srgb: bool) -> Option<u32> {
        for _ in 0..2 {
            self.prepare_swapchain(srgb)?;
            let sc = self.swapchain.as_mut()?;
            let semaphore = self.batches[self.current].acquired;
            match unsafe {
                self.swapchain_fn.as_ref()?.acquire_next_image(
                    sc.handle,
                    100_000_000,
                    semaphore,
                    vk::Fence::null(),
                )
            } {
                Ok((index, suboptimal)) => {
                    sc.stale |= suboptimal;
                    return Some(index);
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => sc.stale = true,
                Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                    self.drop_swapchain(true);
                }
                Err(_) => return None,
            }
        }
        None
    }

    pub(super) fn prepare_swapchain(&mut self, srgb: bool) -> Option<()> {
        let surface_fn = self.surface_fn.clone()?;
        let surface = match &self.swapchain {
            Some(sc) => sc.surface,
            None => {
                let surface = self.create_surface()?;
                let supported = unsafe {
                    surface_fn.get_physical_device_surface_support(
                        self.physical,
                        self.family,
                        surface,
                    )
                }
                .unwrap_or(false);
                if !supported {
                    eprintln!(
                        "kirikae: the graphics queue cannot present to the window; frames are read back instead"
                    );
                    unsafe { surface_fn.destroy_surface(surface, None) };
                    self.present_failed = true;
                    return None;
                }
                self.swapchain = Some(Swapchain {
                    surface,
                    handle: vk::SwapchainKHR::null(),
                    images: Vec::new(),
                    extent: vk::Extent2D::default(),
                    srgb,
                    done: Vec::new(),
                    stale: true,
                    format: vk::Format::UNDEFINED,
                    readable: false,
                });
                surface
            }
        };
        let caps =
            unsafe { surface_fn.get_physical_device_surface_capabilities(self.physical, surface) }
                .ok()?;
        let mut extent = caps.current_extent;
        if extent.width == u32::MAX {
            let (w, h) = crate::display::window_size()?;
            extent = vk::Extent2D {
                width: w.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: h.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            };
        }
        if extent.width == 0 || extent.height == 0 {
            return None;
        }
        let sc = self.swapchain.as_ref()?;
        if !sc.stale && sc.extent == extent && sc.srgb == srgb {
            return Some(());
        }
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            eprintln!(
                "kirikae: the window's swapchain cannot be blitted to; frames are read back instead"
            );
            self.drop_swapchain(true);
            self.present_failed = true;
            return None;
        }
        self.finish();
        unsafe { self.device.device_wait_idle().ok()? };
        let formats =
            unsafe { surface_fn.get_physical_device_surface_formats(self.physical, surface) }
                .ok()?;
        let wanted: &[vk::Format] = if srgb {
            &[vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB]
        } else {
            &[vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM]
        };
        let blit_dst = |f: vk::Format| {
            unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical, f)
            }
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::BLIT_DST)
        };
        let format = wanted
            .iter()
            .find_map(|w| formats.iter().find(|f| f.format == *w))
            .or_else(|| formats.iter().find(|f| blit_dst(f.format)))
            .copied()?;
        let modes =
            unsafe { surface_fn.get_physical_device_surface_present_modes(self.physical, surface) }
                .unwrap_or_default();
        let order: &[vk::PresentModeKHR] = if crate::env_var!("KIRIKAE_VSYNC").is_some() {
            &[vk::PresentModeKHR::FIFO]
        } else {
            &[
                vk::PresentModeKHR::MAILBOX,
                vk::PresentModeKHR::IMMEDIATE,
                vk::PresentModeKHR::FIFO,
            ]
        };
        let mode = order
            .iter()
            .copied()
            .find(|m| modes.contains(m))
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let count = if caps.max_image_count > 0 {
            (caps.min_image_count + 1).min(caps.max_image_count)
        } else {
            caps.min_image_count + 1
        };
        let old = self
            .swapchain
            .as_ref()
            .map_or(vk::SwapchainKHR::null(), |sc| sc.handle);
        let readable = (crate::env_var!("KIRIKAE_PRESENT_DUMP").is_some()
            || crate::env_var!("KIRIKAE_CAPTURE").is_some())
            && caps
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::TRANSFER_SRC);
        let usage = if readable {
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::TRANSFER_DST
        };
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(mode)
            .clipped(true)
            .old_swapchain(old);
        let swapchain_fn = self.swapchain_fn.clone()?;
        let handle = match unsafe { swapchain_fn.create_swapchain(&info, None) } {
            Ok(h) => h,
            Err(e) => {
                eprintln!(
                    "kirikae: cannot make the window's swapchain ({e:?}); frames are read back instead"
                );
                self.drop_swapchain(true);
                self.present_failed = true;
                return None;
            }
        };
        self.drop_swapchain(false);
        let images = unsafe { swapchain_fn.get_swapchain_images(handle) }.ok()?;
        let done = images
            .iter()
            .map(|_| unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            })
            .collect();
        let sc = self.swapchain.as_mut()?;
        *sc = Swapchain {
            surface,
            handle,
            images,
            extent,
            srgb,
            done,
            stale: false,
            format: format.format,
            readable,
        };
        if crate::gpu::trace() || self.stats.presents == 0 {
            eprintln!(
                "kirikae: presenting to the window through a swapchain ({}x{}, {:?}, {mode:?})",
                extent.width, extent.height, format.format
            );
        }
        Some(())
    }

    pub(super) fn drop_swapchain(&mut self, surface: bool) {
        let Some(sc) = self.swapchain.as_mut() else {
            return;
        };
        unsafe {
            let _ = self.device.device_wait_idle();
            for s in sc.done.drain(..) {
                self.device.destroy_semaphore(s, None);
            }
            if sc.handle != vk::SwapchainKHR::null()
                && let Some(f) = &self.swapchain_fn
            {
                f.destroy_swapchain(sc.handle, None);
            }
            sc.handle = vk::SwapchainKHR::null();
            sc.images.clear();
            sc.stale = true;
        }
        if surface {
            let sc = self.swapchain.take().unwrap();
            if let Some(f) = &self.surface_fn {
                unsafe { f.destroy_surface(sc.surface, None) };
            }
        }
    }

    pub(super) fn create_surface(&mut self) -> Option<vk::SurfaceKHR> {
        use crate::display::WindowHandle as W;
        let handle = crate::display::window_handle()?;
        let needed = match handle {
            W::Win32 { .. } => ash::khr::win32_surface::NAME,
            W::Xlib { .. } => ash::khr::xlib_surface::NAME,
            W::Xcb { .. } => ash::khr::xcb_surface::NAME,
            W::Wayland { .. } => ash::khr::wayland_surface::NAME,
        };
        if !self.platforms.contains(&needed) {
            eprintln!(
                "kirikae: the Vulkan driver lacks {}; frames are read back instead",
                needed.to_string_lossy()
            );
            self.present_failed = true;
            return None;
        }
        let (entry, instance) = (&self.entry, &self.instance);
        // SAFETY: the handles come from the live winit window, which stays open for the whole run.
        let made = unsafe {
            match handle {
                W::Win32 { hwnd, hinstance } => {
                    ash::khr::win32_surface::Instance::new(entry, instance).create_win32_surface(
                        &vk::Win32SurfaceCreateInfoKHR::default()
                            .hwnd(hwnd)
                            .hinstance(hinstance),
                        None,
                    )
                }
                W::Xlib { display, window } => {
                    ash::khr::xlib_surface::Instance::new(entry, instance).create_xlib_surface(
                        &vk::XlibSurfaceCreateInfoKHR::default()
                            .dpy(display as *mut vk::Display)
                            .window(window as vk::Window),
                        None,
                    )
                }
                W::Xcb { connection, window } => {
                    ash::khr::xcb_surface::Instance::new(entry, instance).create_xcb_surface(
                        &vk::XcbSurfaceCreateInfoKHR::default()
                            .connection(connection as *mut vk::xcb_connection_t)
                            .window(window),
                        None,
                    )
                }
                W::Wayland { display, surface } => {
                    ash::khr::wayland_surface::Instance::new(entry, instance)
                        .create_wayland_surface(
                            &vk::WaylandSurfaceCreateInfoKHR::default()
                                .display(display as *mut vk::wl_display)
                                .surface(surface as *mut vk::wl_surface),
                            None,
                        )
                }
            }
        };
        match made {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!(
                    "kirikae: cannot make a Vulkan surface for the window ({e:?}); frames are read back instead"
                );
                self.present_failed = true;
                None
            }
        }
    }

    pub fn rendered(&self, cpu: u64, width: u32, height: u32) -> bool {
        self.find_image(cpu, width, height, vk::Format::R8G8B8A8_UNORM)
            .is_some_and(|k| self.images[&k].gpu_dirty)
    }
}
