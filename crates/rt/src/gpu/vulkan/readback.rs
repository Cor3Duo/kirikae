use ash::vk;

use super::images::{ImageKey, Surface, aspect_of};
use super::textures::DERIVED;
use super::tiling::deswizzle;
use super::{Renderer, find_memory};
use crate::gpu::Gpu;
use crate::mem::GuestMemory;

impl Renderer {
    pub fn read_pixels(&mut self, cpu: u64, width: u32, height: u32) -> Option<Vec<u32>> {
        let key = self.find_image(cpu, width, height, vk::Format::R8G8B8A8_UNORM)?;
        if !self.images[&key].gpu_dirty {
            return None;
        }
        let pixels = self.read_image(key)?;
        if key.width == width && key.height == height {
            return Some(pixels);
        }
        let mut out = Vec::with_capacity((width * height) as usize);
        for y in 0..height as usize {
            out.extend_from_slice(&pixels[y * key.width as usize..][..width as usize]);
        }
        Some(out)
    }

    pub(super) fn read_image(&mut self, key: ImageKey) -> Option<Vec<u32>> {
        let start = std::time::Instant::now();
        let r = self.read_image_inner(key);
        self.stats.readback_time += start.elapsed();
        r
    }

    pub(super) fn read_image_inner(&mut self, key: ImageKey) -> Option<Vec<u32>> {
        let bytes = self.read_raw(key, 1)?;
        decode_pixels(key.format, &bytes)
    }

    pub(super) fn read_raw(&mut self, key: ImageKey, layers: u32) -> Option<Vec<u8>> {
        let bpp = bytes_per_pixel(key.format)?;
        let size = (key.width * key.height * bpp) as u64 * layers as u64;
        let (buffer, ptr) = self.readback_buffer(size)?;
        self.begin();
        self.end_rendering();
        let region = [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: aspect_of(key.format) & !vk::ImageAspectFlags::STENCIL,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: layers,
            })
            .image_extent(vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            })];
        let image = self.images[&key].image;
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.cmd,
                image,
                vk::ImageLayout::GENERAL,
                buffer,
                &region,
            )
        };
        self.barrier();
        self.finish();
        if !self.readback_coherent
            && let Some((_, memory, _, _)) = self.readback
        {
            let range = [vk::MappedMemoryRange::default()
                .memory(memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            unsafe { self.device.invalidate_mapped_memory_ranges(&range).ok()? };
        }
        Some(unsafe { std::slice::from_raw_parts(ptr, size as usize) }.to_vec())
    }

    pub fn dump_images(&mut self, dir: &std::path::Path, tag: u64) {
        let only: Option<Vec<u64>> = std::env::var("KIRIKAE_TARGET_DUMP_ONLY").ok().map(|v| {
            v.split(',')
                .filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        });
        let derived = crate::env_var!("KIRIKAE_TARGET_DUMP_DERIVED").is_some();
        let keys: Vec<ImageKey> = self
            .images
            .iter()
            .filter(|(k, i)| {
                (i.gpu_dirty || (derived && k.cpu & DERIVED != 0))
                    && only
                        .as_ref()
                        .is_none_or(|a| a.contains(&(k.cpu & !DERIVED)))
            })
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            let layers = if crate::env_var!("KIRIKAE_TARGET_DUMP_RAW").is_some() {
                self.images[&key].layers
            } else {
                1
            };
            let Some(raw) = self.read_raw(key, layers) else {
                continue;
            };
            if crate::env_var!("KIRIKAE_TARGET_DUMP_RAW").is_some() {
                let name = format!(
                    "f{tag:06}_{:x}_{:?}_{}x{}.raw",
                    key.cpu, key.format, key.width, key.height
                );
                let _ = std::fs::write(dir.join(name), &raw);
            }
            let Some(pixels) = decode_pixels(key.format, &raw) else {
                continue;
            };
            let name = format!(
                "f{tag:06}_{:x}_{:?}_{}x{}.png",
                key.cpu, key.format, key.width, key.height
            );
            let _ = crate::display::write_png(&dir.join(name), key.width, key.height, &pixels);
        }
    }

    pub(super) fn readback_buffer(&mut self, size: u64) -> Option<(vk::Buffer, *mut u8)> {
        if let Some((b, _, s, p)) = self.readback
            && s >= size
        {
            return Some((b, p));
        }
        self.finish();
        unsafe {
            if let Some((b, m, _, _)) = self.readback.take() {
                self.device.destroy_buffer(b, None);
                self.device.free_memory(m, None);
            }
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
            })
            .or_else(|| {
                find_memory(
                    &self.mem_props,
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })?;
            self.readback_coherent = self.mem_props.memory_types[ty as usize]
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
            self.readback = Some((buffer, memory, size, ptr));
            Some((buffer, ptr))
        }
    }
}

pub(super) fn bytes_per_pixel(f: vk::Format) -> Option<u32> {
    use vk::Format as F;
    Some(match f {
        F::R8_UNORM => 1,
        F::R8G8_UNORM | F::R16_SFLOAT | F::R16_UNORM => 2,
        F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SRGB
        | F::B8G8R8A8_UNORM
        | F::B8G8R8A8_SRGB
        | F::A2B10G10R10_UNORM_PACK32
        | F::B10G11R11_UFLOAT_PACK32
        | F::R32_SFLOAT
        | F::R16G16_SFLOAT => 4,
        F::R16G16_UNORM | F::R16G16_SNORM | F::R32_UINT | F::R32_SINT => 4,
        F::R16G16B16A16_SFLOAT | F::R32G32_SFLOAT => 8,
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT => 8,
        F::R32G32B32A32_SFLOAT => 16,
        F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT | F::D24_UNORM_S8_UINT | F::X8_D24_UNORM_PACK32 => 4,
        F::D16_UNORM => 2,
        _ => return None,
    })
}

pub(super) fn decode_pixels(f: vk::Format, bytes: &[u8]) -> Option<Vec<u32>> {
    use vk::Format as F;
    let to8 = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let rgb = |r: f32, g: f32, b: f32| to8(r) << 16 | to8(g) << 8 | to8(b);
    let px: Vec<u32> = match f {
        F::R8_UNORM => bytes.iter().map(|&r| (r as u32) * 0x010101).collect(),
        F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| to8(f32::from_le_bytes(*p)) * 0x010101)
            .collect(),
        F::D24_UNORM_S8_UINT | F::X8_D24_UNORM_PACK32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| (u32::from_le_bytes(*p) >> 16 & 0xff) * 0x010101)
            .collect(),
        F::R8G8_UNORM => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8)
            .collect(),
        F::R16_SFLOAT => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| to8(half_to_f32(u16::from_le_bytes(*p))) * 0x010101)
            .collect(),
        F::R16_UNORM => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| (u16::from_le_bytes(*p) >> 8) as u32 * 0x010101)
            .collect(),
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32)
            .collect(),
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| (p[2] as u32) << 16 | (p[1] as u32) << 8 | p[0] as u32)
            .collect(),
        F::A2B10G10R10_UNORM_PACK32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                let v = u32::from_le_bytes(*p);
                ((v >> 2) & 0xff) << 16 | ((v >> 12) & 0xff) << 8 | ((v >> 22) & 0xff)
            })
            .collect(),
        F::B10G11R11_UFLOAT_PACK32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                let v = u32::from_le_bytes(*p);
                rgb(
                    small_float(v & 0x7ff, 6),
                    small_float(v >> 11 & 0x7ff, 6),
                    small_float(v >> 22 & 0x3ff, 5),
                )
            })
            .collect(),
        F::R32_SFLOAT => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| to8(f32::from_le_bytes(*p)) * 0x010101)
            .collect(),
        F::R16G16_SFLOAT => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                rgb(
                    half_to_f32(u16::from_le_bytes([p[0], p[1]])),
                    half_to_f32(u16::from_le_bytes([p[2], p[3]])),
                    0.0,
                )
            })
            .collect(),
        F::R16G16B16A16_SFLOAT => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| {
                let c = |k: usize| half_to_f32(u16::from_le_bytes([p[2 * k], p[2 * k + 1]]));
                rgb(c(0), c(1), c(2))
            })
            .collect(),
        F::R32G32_SFLOAT => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| {
                rgb(
                    f32::from_le_bytes(p[..4].try_into().unwrap()),
                    f32::from_le_bytes(p[4..].try_into().unwrap()),
                    0.0,
                )
            })
            .collect(),
        F::R32G32B32A32_SFLOAT => bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|p| {
                let c = |k: usize| f32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap());
                rgb(c(0), c(1), c(2))
            })
            .collect(),
        _ => return None,
    };
    Some(px)
}

pub(super) fn small_float(v: u32, mantissa: u32) -> f32 {
    let e = v >> mantissa;
    let m = v & ((1 << mantissa) - 1);
    match e {
        0 => m as f32 / (1 << mantissa) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        _ => (1.0 + m as f32 / (1 << mantissa) as f32) * 2f32.powi(e as i32 - 15),
    }
}

pub(super) fn half_to_f32(h: u16) -> f32 {
    let s = (h >> 15) as u32;
    let e = (h >> 10 & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => {
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s << 31 | (e as u32) << 23 | (m & 0x3ff) << 13
        }
        31 => s << 31 | 0xff << 23 | m << 13,
        _ => s << 31 | (e + 127 - 15) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

pub fn read_surface(gpu: &Gpu, mem: &GuestMemory, s: &Surface, bpp: u32) -> Option<Vec<u8>> {
    let row = s.width as u64 * bpp as u64;
    match s.pitch {
        Some(pitch) => {
            let raw = gpu.read(mem, s.va, pitch as u64 * s.height as u64)?;
            let mut out = Vec::with_capacity((row * s.height as u64) as usize);
            for y in 0..s.height as u64 {
                let line = &raw[(y * pitch as u64) as usize..];
                let n = line.len().min(row as usize);
                out.extend_from_slice(&line[..n]);
                out.resize(out.len() + row as usize - n, 0);
            }
            Some(out)
        }
        None => {
            let width_bytes = row.div_ceil(64) * 64;
            let gob_rows = 8u64 << s.block_height_log2;
            let size = width_bytes * (s.height as u64).div_ceil(gob_rows) * gob_rows;
            let raw = gpu.read(mem, s.va, size)?;
            Some(deswizzle(&raw, s.width, s.height, bpp, s.block_height_log2))
        }
    }
}

pub(super) fn probe(va: u64) -> Option<(u32, u8)> {
    let spec = crate::env_var!("KIRIKAE_PROBE")?.to_str()?;
    let parts: Vec<&str> = spec.split(':').collect();
    let reg = parts.last()?.trim_start_matches('r').parse().ok()?;
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok();
    match parts.len() {
        1 => Some((u32::MAX, reg)),
        2 => Some((hex(parts[0])? as u32, reg)),
        _ => (hex(parts[0])? == va).then_some((hex(parts[1])? as u32, reg)),
    }
}
