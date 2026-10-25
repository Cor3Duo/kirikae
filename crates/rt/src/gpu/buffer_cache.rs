use super::fxhash::FastMap as HashMap;

use ash::vk;

const ARENA: u64 = 16 << 20;
const MIN_BLOCK: u64 = 256;
const CLASSES: usize = 17;
const PROMOTE: u32 = 2;
const KEEP_FRAMES: u64 = 300;
const VRAM_BUDGET: u64 = 128 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    Constant,
    Vertex,
    Index(u32, bool),
}

#[derive(Clone, Copy, Debug)]
pub struct Block {
    arena: usize,
    offset: u64,
    class: usize,
}

struct Arena {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    used: u64,
}

type IndexRange = (Option<u32>, Option<(u32, u32)>);

struct Entry {
    shadow: Vec<u8>,
    stable: u32,
    resident: Option<Block>,
    used: u64,
    range: Option<IndexRange>,
}

#[derive(Default)]
pub struct BufferCache {
    arenas: Vec<Arena>,
    free: Vec<Vec<Block>>,
    entries: HashMap<(u64, u64, Kind), Entry>,
    vram: Option<u32>,
    host: u32,
    vram_used: u64,
    last_eviction: u64,
    pub hits: u64,
}

fn class_of(size: u64) -> Option<usize> {
    let c = size.max(MIN_BLOCK).next_power_of_two();
    let class = (c.trailing_zeros() - MIN_BLOCK.trailing_zeros()) as usize;
    (class < CLASSES).then_some(class)
}

fn find_memory(
    props: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        props.memory_types[i as usize]
            .property_flags
            .contains(flags)
    })
}

impl BufferCache {
    pub fn new(props: &vk::PhysicalDeviceMemoryProperties, host: u32) -> BufferCache {
        let host_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let vram = find_memory(props, host_flags | vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .filter(|_| std::env::var_os("KIRIKAE_NO_BAR").is_none());
        BufferCache {
            free: vec![Vec::new(); CLASSES],
            vram,
            host,
            ..Default::default()
        }
    }

    pub fn lookup(
        &mut self,
        device: &ash::Device,
        key: (u64, u64, Kind),
        source: &[u8],
        frame: u64,
        convert: impl FnOnce(&[u8]) -> Vec<u8>,
        retired: &mut Vec<Block>,
    ) -> Option<(vk::Buffer, u64)> {
        let entry = self.entries.entry(key).or_insert_with(|| Entry {
            shadow: Vec::new(),
            stable: 0,
            resident: None,
            used: frame,
            range: None,
        });
        entry.used = frame;
        if entry.shadow != source {
            entry.range = None;
            entry.shadow.clear();
            entry.shadow.extend_from_slice(source);
            entry.stable = 0;
            retired.extend(entry.resident.take());
            return None;
        }
        entry.stable = entry.stable.saturating_add(1);
        if let Some(b) = entry.resident {
            self.hits += source.len() as u64;
            let a = &self.arenas[b.arena];
            return Some((a.buffer, b.offset));
        }
        if entry.stable < PROMOTE {
            return None;
        }
        let data = convert(source);
        let block = self.alloc(device, data.len() as u64)?;
        let a = &self.arenas[block.arena];
        // SAFETY: the block is fresh (not used by any GPU work) and inside the mapped arena.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                a.ptr.add(block.offset as usize),
                data.len(),
            )
        };
        let buffer = a.buffer;
        self.entries.get_mut(&key).unwrap().resident = Some(block);
        Some((buffer, block.offset))
    }

    pub fn release(&mut self, b: Block) {
        self.free[b.class].push(b);
    }

    pub fn index_range(
        &mut self,
        key: (u64, u64, Kind),
        restart: Option<u32>,
        compute: impl FnOnce(&[u8]) -> Option<(u32, u32)>,
    ) -> Option<(u32, u32)> {
        let entry = self.entries.get_mut(&key)?;
        match entry.range {
            Some((r, range)) if r == restart => range,
            _ => {
                let range = compute(&entry.shadow);
                entry.range = Some((restart, range));
                range
            }
        }
    }

    pub fn evict(&mut self, frame: u64, retired: &mut Vec<Block>) {
        if frame < self.last_eviction + 60 {
            return;
        }
        self.last_eviction = frame;
        self.entries.retain(|_, e| {
            let keep = e.used + KEEP_FRAMES >= frame;
            if !keep {
                retired.extend(e.resident.take());
            }
            keep
        });
    }

    fn alloc(&mut self, device: &ash::Device, size: u64) -> Option<Block> {
        let class = class_of(size)?;
        if let Some(b) = self.free[class].pop() {
            return Some(b);
        }
        let bytes = MIN_BLOCK << class;
        if let Some(i) = self.arenas.iter().position(|a| a.used + bytes <= ARENA) {
            let a = &mut self.arenas[i];
            let offset = a.used.next_multiple_of(bytes);
            if offset + bytes <= ARENA {
                a.used = offset + bytes;
                return Some(Block {
                    arena: i,
                    offset,
                    class,
                });
            }
        }
        let arena = self.new_arena(device)?;
        self.arenas.push(arena);
        let a = self.arenas.last_mut().unwrap();
        a.used = bytes;
        Some(Block {
            arena: self.arenas.len() - 1,
            offset: 0,
            class,
        })
    }

    fn new_arena(&mut self, device: &ash::Device) -> Option<Arena> {
        let usage = vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::UNIFORM_BUFFER;
        unsafe {
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(ARENA).usage(usage),
                    None,
                )
                .ok()?;
            let req = device.get_buffer_memory_requirements(buffer);
            let vram = self.vram.filter(|&t| {
                self.vram_used + req.size <= VRAM_BUDGET && req.memory_type_bits >> t & 1 == 1
            });
            let mut memory = None;
            if let Some(t) = vram {
                memory = device
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(t),
                        None,
                    )
                    .ok();
                if memory.is_some() {
                    self.vram_used += req.size;
                } else {
                    self.vram = None;
                }
            }
            let memory = match memory {
                Some(m) => m,
                None => match device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(self.host),
                    None,
                ) {
                    Ok(m) => m,
                    Err(_) => {
                        device.destroy_buffer(buffer, None);
                        return None;
                    }
                },
            };
            device.bind_buffer_memory(buffer, memory, 0).ok()?;
            let ptr = device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .ok()? as *mut u8;
            Some(Arena {
                buffer,
                memory,
                ptr,
                used: 0,
            })
        }
    }

    pub fn memory(&self) -> (u64, u64) {
        (self.vram_used, self.arenas.len() as u64 * ARENA)
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for a in self.arenas.drain(..) {
            unsafe {
                device.destroy_buffer(a.buffer, None);
                device.free_memory(a.memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes() {
        assert_eq!(class_of(1), Some(0));
        assert_eq!(class_of(256), Some(0));
        assert_eq!(class_of(257), Some(1));
        assert_eq!(class_of(16 << 20), Some(16));
        assert_eq!(class_of((16 << 20) + 1), None);
    }
}
