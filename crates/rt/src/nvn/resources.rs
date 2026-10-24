use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub enum Call {
    PoolBuilderDefaults,
    PoolBuilderStorage,
    PoolBuilderFlags,
    PoolInitialize,
    PoolAddress,
    BufferBuilderDefaults,
    BufferBuilderStorage,
    BufferInitialize,
    BufferAddress,
    TextureBuilder(TexSet),
    TextureInitialize,
    View(ViewSet),
    RegisterTexture,
}

#[derive(Clone, Copy, Debug)]
pub enum TexSet {
    Defaults,
    Flags,
    Target,
    Width,
    Height,
    Depth,
    Size1D,
    Size2D,
    Size3D,
    Levels,
    Format,
    Samples,
    Stride,
    Storage,
    PackagedLayout,
}

#[derive(Clone, Copy, Debug)]
pub enum ViewSet {
    Defaults,
    Levels,
    Layers,
    Format,
    Target,
}

pub fn call(name: &str) -> Option<Call> {
    use Call::*;
    Some(match name.strip_prefix("nvn")? {
        "MemoryPoolBuilderSetDefaults" => PoolBuilderDefaults,
        "MemoryPoolBuilderSetStorage" => PoolBuilderStorage,
        "MemoryPoolBuilderSetFlags" => PoolBuilderFlags,
        "MemoryPoolInitialize" => PoolInitialize,
        "MemoryPoolGetBufferAddress" => PoolAddress,
        "BufferBuilderSetDefaults" => BufferBuilderDefaults,
        "BufferBuilderSetStorage" => BufferBuilderStorage,
        "BufferInitialize" => BufferInitialize,
        "BufferGetAddress" => BufferAddress,
        "TextureBuilderSetDefaults" => TextureBuilder(TexSet::Defaults),
        "TextureBuilderSetFlags" => TextureBuilder(TexSet::Flags),
        "TextureBuilderSetTarget" => TextureBuilder(TexSet::Target),
        "TextureBuilderSetWidth" => TextureBuilder(TexSet::Width),
        "TextureBuilderSetHeight" => TextureBuilder(TexSet::Height),
        "TextureBuilderSetDepth" => TextureBuilder(TexSet::Depth),
        "TextureBuilderSetSize1D" => TextureBuilder(TexSet::Size1D),
        "TextureBuilderSetSize2D" => TextureBuilder(TexSet::Size2D),
        "TextureBuilderSetSize3D" => TextureBuilder(TexSet::Size3D),
        "TextureBuilderSetLevels" => TextureBuilder(TexSet::Levels),
        "TextureBuilderSetFormat" => TextureBuilder(TexSet::Format),
        "TextureBuilderSetSamples" => TextureBuilder(TexSet::Samples),
        "TextureBuilderSetStride" => TextureBuilder(TexSet::Stride),
        "TextureBuilderSetStorage" => TextureBuilder(TexSet::Storage),
        "TextureBuilderSetPackagedTextureLayout" => TextureBuilder(TexSet::PackagedLayout),
        "TextureInitialize" => TextureInitialize,
        "TextureViewSetDefaults" => View(ViewSet::Defaults),
        "TextureViewSetLevels" => View(ViewSet::Levels),
        "TextureViewSetLayers" => View(ViewSet::Layers),
        "TextureViewSetFormat" => View(ViewSet::Format),
        "TextureViewSetTarget" => View(ViewSet::Target),
        "TexturePoolRegisterTexture" | "TexturePoolRegisterImage" => RegisterTexture,
        _ => return None,
    })
}

pub fn after(c: Call) -> bool {
    matches!(c, Call::PoolInitialize | Call::PoolAddress | Call::BufferInitialize | Call::BufferAddress | Call::TextureInitialize | Call::View(_))
}

pub const TEXTURE_BYTES: u64 = 0xc0;
pub const POOL_BYTES: u64 = 0x100;
pub const VIEW_BYTES: u64 = 0x28;

const VIRTUAL: u32 = 0x200;
pub const BUFFER_BYTES: u64 = 0x30;

fn format(f: u32) -> (u32, u32, u32) {
    let c = |bytes| (bytes, 1, 1);
    match f {
        0x01..=0x04 | 0x31 | 0x51 | 0x52 => c(1),
        0x05..=0x09 | 0x0d..=0x10 | 0x32 | 0x39..=0x3c | 0x53 | 0x54 | 0x57 | 0x58 | 0x95..=0x97 => c(2),
        0x19..=0x1c | 0x5d | 0x5e => c(3),
        0x0a..=0x0c | 0x11..=0x15 | 0x25..=0x28 | 0x33..=0x35 | 0x37 | 0x38 | 0x3d..=0x40 | 0x55 | 0x56 | 0x59 | 0x5a | 0x63 | 0x64 | 0x69..=0x70 | 0x98..=0x9b => c(4),
        0x1d..=0x21 | 0x5f | 0x60 => c(6),
        0x16..=0x18 | 0x29..=0x2d | 0x36 | 0x5b | 0x5c | 0x65 | 0x66 | 0x71..=0x75 => c(8),
        0x22..=0x24 | 0x61 | 0x62 => c(12),
        0x2e..=0x30 | 0x67 | 0x68 | 0x76..=0x78 => c(16),
        0x41 | 0x42 | 0x45 | 0x46 | 0x49 | 0x4a => (8, 4, 4),
        0x43 | 0x44 | 0x47 | 0x48 | 0x4b..=0x50 => (16, 4, 4),
        0x79..=0x94 => {
            const ASTC: [(u32, u32); 14] = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12)];
            let (w, h) = ASTC[((f - 0x79) % 14) as usize];
            (16, w, h)
        }
        _ => (0, 1, 1),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TexBuilder {
    flags: u32,
    target: u32,
    width: u32,
    height: u32,
    depth: u32,
    levels: u32,
    format: u32,
    samples: u32,
    stride: u64,
    pool: u64,
    offset: u64,
    packaged: u64,
}

impl TexBuilder {
    fn defaults() -> Self {
        TexBuilder { target: 1, width: 1, height: 1, depth: 1, levels: 1, ..Default::default() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texture {
    pub address: u64,
    pub format: u32,
    pub target: u32,
    width: u32,
    height: u32,
    depth: u32,
    layers: u32,
    levels: u32,
    pitch: bool,
    block_height_log2: u32,
    block_depth_log2: u32,
}

impl Texture {
    fn new(b: &TexBuilder, address: u64) -> Self {
        let height = if matches!(b.target, 0 | 3 | 10) { 1 } else { b.height.max(1) };
        let layers = match b.target {
            3 => b.height,
            4 | 6 | 9 => b.depth,
            8 => 6,
            _ => 1,
        }
        .max(1);
        let depth = if b.target == 2 { b.depth.max(1) } else { 1 };
        let (_, _, bh) = format(b.format);
        let gobs_high = height.div_ceil(bh).div_ceil(8);
        let mut block_height_log2 = 0;
        while (1 << block_height_log2) < gobs_high && block_height_log2 < 4 {
            block_height_log2 += 1;
        }
        let mut block_depth_log2 = 0;
        while depth > 1 && (1 << block_depth_log2) < depth && block_depth_log2 < 5 {
            block_depth_log2 += 1;
        }
        if block_height_log2 + block_depth_log2 > 4 {
            block_height_log2 = 4u32.saturating_sub(block_depth_log2);
        }
        if b.packaged != 0 && b.packaged as u32 <= 5 {
            block_height_log2 = b.packaged as u32;
        }
        Texture {
            address,
            format: b.format,
            target: b.target,
            width: b.width.max(1),
            height,
            depth,
            layers,
            levels: b.levels.max(1),
            pitch: b.flags & LINEAR != 0 || b.target == 10,
            block_height_log2,
            block_depth_log2,
        }
    }

    fn level_layout(&self, level: u32) -> (u64, u32, u32, u64, u64) {
        let (bytes, bw, bh) = format(self.format);
        let (w, h, d) = ((self.width >> level).max(1), (self.height >> level).max(1), (self.depth >> level).max(1));
        let (wb, hb) = (w.div_ceil(bw), h.div_ceil(bh));
        let mut bhl = self.block_height_log2;
        while bhl > 0 && (8 << (bhl - 1)) >= hb {
            bhl -= 1;
        }
        let mut bdl = self.block_depth_log2;
        while bdl > 0 && (1 << (bdl - 1)) >= d {
            bdl -= 1;
        }
        let width_gobs = (wb as u64 * bytes as u64).div_ceil(64);
        (width_gobs, bhl, bdl, (hb as u64).div_ceil(8 << bhl), (d as u64).div_ceil(1 << bdl))
    }

    fn level_size(&self, level: u32) -> u64 {
        let (width_gobs, bhl, bdl, high, deep) = self.level_layout(level);
        width_gobs * (512 << (bhl + bdl)) * high * deep
    }

    fn layer_size(&self) -> u64 {
        let size: u64 = (0..self.levels).map(|l| self.level_size(l)).sum();
        if self.layers > 1 {
            let align = 512 << self.block_height_log2;
            return size.div_ceil(align) * align;
        }
        size
    }

    pub fn surface(&self, view: &View) -> Surface {
        let level = view.base_level.min(self.levels.saturating_sub(1));
        let (_, block_height_log2, _, _, _) = self.level_layout(level);
        let (bytes, _, _) = format(self.format);
        let width = (self.width >> level).max(1);
        Surface {
            format: if view.format != 0 { view.format } else { self.format },
            width,
            height: (self.height >> level).max(1),
            block_height_log2,
            pitch: self.pitch.then_some(width * bytes),
            layers: if view.layers != 0 { view.layers } else { self.layers },
            bytes,
        }
    }

    pub fn view_offset(&self, view: &View) -> u64 {
        if self.pitch {
            return 0;
        }
        let base = view.base_level.min(self.levels);
        let level: u64 = (0..base).map(|l| self.level_size(l)).sum();
        if self.target == 2 {
            let (width_gobs, bhl, bdl, high, _) = self.level_layout(base);
            let z = view.min_layer as u64;
            return level + (z >> bdl) * width_gobs * (512 << (bhl + bdl)) * high + ((z & ((1 << bdl) - 1)) << bhl) * 512;
        }
        self.layer_size() * view.min_layer as u64 + level
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Surface {
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub block_height_log2: u32,
    pub pitch: Option<u32>,
    pub layers: u32,
    pub bytes: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct View {
    pub base_level: u32,
    pub levels: u32,
    pub min_layer: u32,
    pub layers: u32,
    pub format: u32,
    pub target: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Pool {
    size: u64,
    pitch: u64,
    block_linear: u64,
    compressed: u64,
}

const COMPRESSIBLE: u32 = 0x8;
const LINEAR: u32 = 0x10 | 0x40;

#[derive(Default)]
pub struct Resources {
    pool_builders: HashMap<u64, (u64, u64, u32)>,
    pools: HashMap<u64, Pool>,
    pools_by_bytes: HashMap<u64, Pool>,
    pool_hashes: HashMap<u64, u64>,
    pool_flags: HashMap<u64, u32>,
    pub pool_mappings: HashMap<u64, Vec<(u64, u64, u64, u64)>>,
    buffer_builders: HashMap<u64, (u64, u64)>,
    buffers: HashMap<u64, u64>,
    buffers_by_bytes: HashMap<u64, u64>,
    tex_builders: HashMap<u64, TexBuilder>,
    textures: HashMap<u64, (u64, Texture)>,
    textures_by_bytes: HashMap<u64, Texture>,
    views: HashMap<u64, View>,
    views_by_bytes: HashMap<u64, View>,
    pub registered: HashMap<u32, (Option<Texture>, View)>,
    pub texture_pools: Vec<u64>,
    pub checks: Checks,
}

#[derive(Default, Clone, Debug)]
pub struct Checks {
    pub pools: u64,
    pub pools_differ: u64,
    pub pools_unknown: u64,
    pub pools_copied: u64,
    pub pools_made: u64,
    pub pools_made_differ: u64,
    pub buffers: u64,
    pub buffers_differ: u64,
    pub buffers_unknown: u64,
    pub buffers_copied: u64,
    pub textures: u64,
    pub textures_differ: u64,
    pub textures_unknown: u64,
    pub textures_copied: u64,
    pub textures_changed: u64,
    pub views: u64,
    pub views_differ: u64,
    pub views_copied: u64,
    pub views_by_address: u64,
    pub views_unknown: u64,
    pub textures_made: u64,
    pub textures_made_differ: u64,
}

fn hash(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}

impl Resources {
    pub fn before(&mut self, call: Call, a: &[u64], read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) {
        match call {
            Call::PoolBuilderDefaults => {
                self.pool_builders.insert(a[0], (0, 0, 0));
            }
            Call::PoolBuilderStorage => {
                let b = self.pool_builders.entry(a[0]).or_default();
                (b.0, b.1) = (a[1], a[2]);
            }
            Call::PoolBuilderFlags => self.pool_builders.entry(a[0]).or_default().2 = a[1] as u32,
            Call::BufferBuilderDefaults => {
                self.buffer_builders.insert(a[0], (0, 0));
            }
            Call::BufferBuilderStorage => {
                self.buffer_builders.insert(a[0], (a[1], a[2]));
            }
            Call::TextureBuilder(s) => {
                if let TexSet::Defaults = s {
                    self.tex_builders.insert(a[0], TexBuilder::defaults());
                    return;
                }
                let b = self.tex_builders.entry(a[0]).or_insert_with(TexBuilder::defaults);
                let w = |i: usize| a[i] as u32;
                match s {
                    TexSet::Defaults => {}
                    TexSet::Flags => b.flags = w(1),
                    TexSet::Target => b.target = w(1),
                    TexSet::Width => b.width = w(1),
                    TexSet::Height => b.height = w(1),
                    TexSet::Depth => b.depth = w(1),
                    TexSet::Size1D => (b.width, b.height, b.depth) = (w(1), 1, 1),
                    TexSet::Size2D => (b.width, b.height, b.depth) = (w(1), w(2), 1),
                    TexSet::Size3D => (b.width, b.height, b.depth) = (w(1), w(2), w(3)),
                    TexSet::Levels => b.levels = w(1),
                    TexSet::Format => b.format = w(1),
                    TexSet::Samples => b.samples = w(1),
                    TexSet::Stride => b.stride = a[1],
                    TexSet::Storage => (b.pool, b.offset) = (a[1], a[2]),
                    TexSet::PackagedLayout => {
                        b.packaged = if a[1] == 0 { 0 } else { read(a[1], 8).map_or(0, |v| u64::from_le_bytes(v[..8].try_into().unwrap())) }
                    }
                }
            }
            Call::View(s) => {
                if let ViewSet::Defaults = s {
                    self.views.insert(a[0], View::default());
                    return;
                }
                let v = self.views.entry(a[0]).or_default();
                match s {
                    ViewSet::Defaults => {}
                    ViewSet::Levels => (v.base_level, v.levels) = (a[1] as u32, a[2] as u32),
                    ViewSet::Layers => (v.min_layer, v.layers) = (a[1] as u32, a[2] as u32),
                    ViewSet::Format => v.format = a[1] as u32,
                    ViewSet::Target => v.target = a[1] as u32,
                }
            }
            Call::RegisterTexture => {
                if !self.texture_pools.contains(&a[0]) {
                    self.texture_pools.push(a[0]);
                }
                let tex = self.texture(a[2], read);
                let view = if a[3] == 0 { View::default() } else { self.view(a[3], read) };
                self.registered.insert(a[1] as u32, (tex, view));
            }
            _ => {}
        }
    }

    pub fn after(&mut self, call: Call, a: &[u64], ret: u64, gpu_addresses_of: &dyn Fn(u64) -> Vec<u64>, read: &dyn Fn(u64, u64) -> Option<Vec<u8>>, log: &mut dyn FnMut(&'static str, String)) {
        match call {
            Call::View(_) => {
                self.before(call, a, read);
                let v = self.views.get(&a[0]).copied().unwrap_or_default();
                if let Some(bytes) = read(a[0], VIEW_BYTES) {
                    self.views_by_bytes.insert(hash(&bytes), v);
                }
            }
            Call::PoolInitialize => {
                let (memory, size, flags) = self.pool_builders.get(&a[1]).copied().unwrap_or_default();
                let vas = gpu_addresses_of(memory);
                let pitch = vas.first().copied().unwrap_or(0);
                let block_linear = vas.get(1).copied().unwrap_or(pitch);
                let p = Pool { size, pitch, block_linear, compressed: vas.get(2).copied().unwrap_or(block_linear) };
                self.pools.insert(a[0], p);
                self.pool_flags.insert(a[0], flags);
                if let Some(bytes) = read(a[0], POOL_BYTES) {
                    self.pools_by_bytes.insert(hash(&bytes), p);
                    self.pool_hashes.insert(a[0], hash(&bytes));
                }
                let _ = flags;
            }
            Call::PoolAddress => {
                self.checks.pools += 1;
                let model = self.pools.get(&a[0]).map(|p| p.pitch);
                if model != Some(ret) {
                    self.checks.pools_differ += 1;
                    log("pools", format!("pool {:#x}: model {model:x?}, driver {ret:#x}; mappings of its memory then {:x?}", a[0], self.pool_mappings.get(&a[0])));
                }
            }
            Call::BufferInitialize => {
                let (pool, offset) = self.buffer_builders.get(&a[1]).copied().unwrap_or_default();
                let address = self.pool(pool, read).map_or(0, |p| p.pitch + offset);
                self.buffers.insert(a[0], address);
                if let Some(bytes) = read(a[0], BUFFER_BYTES) {
                    self.buffers_by_bytes.insert(hash(&bytes), address);
                }
            }
            Call::BufferAddress => {
                self.checks.buffers += 1;
                let model = match self.buffers.get(&a[0]) {
                    Some(&m) => Some(m),
                    None => {
                        let copy = read(a[0], BUFFER_BYTES).and_then(|b| self.buffers_by_bytes.get(&hash(&b)).copied());
                        if copy.is_some() {
                            self.checks.buffers_copied += 1;
                        }
                        copy
                    }
                };
                match model {
                    None => self.checks.buffers_unknown += 1,
                    Some(m) if m != ret => {
                        self.checks.buffers_differ += 1;
                        log("buffers", format!("buffer {:#x}: model {m:#x}, driver {ret:#x}", a[0]));
                    }
                    Some(_) => {}
                }
            }
            Call::TextureInitialize => {
                let b = self.tex_builders.get(&a[1]).copied().unwrap_or_else(TexBuilder::defaults);
                let address = self.pool(b.pool, read).map_or(0, |p| {
                    let base = if b.flags & LINEAR != 0 || b.target == 10 {
                        p.pitch
                    } else if b.flags & COMPRESSIBLE != 0 {
                        p.compressed
                    } else {
                        p.block_linear
                    };
                    base + b.offset
                });
                let t = Texture::new(&b, address);
                let bytes = read(a[0], TEXTURE_BYTES).map(|b| hash(&b));
                if let Some(h) = bytes {
                    self.textures_by_bytes.insert(h, t);
                }
                self.textures.insert(a[0], (bytes.unwrap_or(0), t));
            }
            _ => {}
        }
    }

    pub fn texture(&mut self, obj: u64, read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> Option<Texture> {
        let bytes = read(obj, TEXTURE_BYTES).map(|b| hash(&b));
        let here = self.textures.get(&obj).copied();
        if let Some((h, t)) = here
            && Some(h) == bytes
        {
            return Some(t);
        }
        if let Some(t) = bytes.and_then(|h| self.textures_by_bytes.get(&h).copied()) {
            self.checks.textures_copied += 1;
            return Some(t);
        }
        if let Some((_, t)) = here {
            self.checks.textures_changed += 1;
            return Some(t);
        }
        self.checks.textures_unknown += 1;
        None
    }

    fn pool(&mut self, obj: u64, read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> Option<Pool> {
        if let Some(&p) = self.pools.get(&obj) {
            return Some(p);
        }
        let p = read(obj, POOL_BYTES).and_then(|b| self.pools_by_bytes.get(&hash(&b)).copied());
        match p {
            Some(_) => self.checks.pools_copied += 1,
            None => self.checks.pools_unknown += 1,
        }
        p
    }

    pub fn reserved(&mut self, pool: u64, builder: u64, va: u64) {
        let flags = self.pool_builders.get(&builder).map_or(0, |b| b.2);
        if flags & VIRTUAL != 0
            && let Some(p) = self.pools.get_mut(&pool)
        {
            *p = Pool { size: p.size, pitch: va, block_linear: va, compressed: va };
            let p = *p;
            if let Some(h) = self.pool_hashes.get(&pool) {
                self.pools_by_bytes.insert(*h, p);
            }
        }
    }

    pub fn check_pool(&mut self, pool: u64, builder: u64, driver: u64) -> Option<String> {
        self.checks.pools_made += 1;
        let model = self.pools.get(&pool).map_or(0, |p| p.pitch);
        if model == driver {
            return None;
        }
        self.checks.pools_made_differ += 1;
        let b = self.pool_builders.get(&builder).copied().unwrap_or_default();
        Some(format!(
            "new pool {pool:#x}: model {model:#x}, driver {driver:#x}; memory {:#x} size {:#x} flags {:#x}; mappings of the memory {:x?}",
            b.0,
            b.1,
            b.2,
            self.pool_mappings.get(&pool)
        ))
    }

    pub fn pool_memory(&self, builder: u64) -> u64 {
        self.pool_builders.get(&builder).map_or(0, |b| b.0)
    }

    pub fn view(&mut self, obj: u64, read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> View {
        let bytes = read(obj, VIEW_BYTES).map(|b| hash(&b));
        if let Some(&v) = bytes.and_then(|h| self.views_by_bytes.get(&h)) {
            if self.views.get(&obj) != Some(&v) {
                self.checks.views_copied += 1;
            }
            return v;
        }
        match self.views.get(&obj) {
            Some(&v) => {
                self.checks.views_by_address += 1;
                v
            }
            None => {
                self.checks.views_unknown += 1;
                View::default()
            }
        }
    }

    pub fn check_texture(&mut self, obj: u64, builder: u64, driver: u64) -> Option<String> {
        self.checks.textures_made += 1;
        let (_, t) = self.textures.get(&obj).copied()?;
        if t.address == driver {
            return None;
        }
        self.checks.textures_made_differ += 1;
        let b = self.tex_builders.get(&builder).copied();
        let pool = b.map_or(0, |b| b.pool);
        Some(format!(
            "new texture {obj:#x}: model {:#x}, driver {driver:#x}; builder {builder:#x} {b:x?}; pool flags {:x?}, mappings {:x?}",
            t.address,
            self.pool_flags.get(&pool),
            self.pool_mappings.get(&pool)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(target: u32, size: (u32, u32, u32), levels: u32, format: u32) -> Texture {
        let b = TexBuilder { target, width: size.0, height: size.1, depth: size.2, levels, format, ..TexBuilder::defaults() };
        Texture::new(&b, 0)
    }

    #[test]
    fn view_offsets() {
        let t = texture(1, (256, 256, 1), 3, 0x25);
        assert_eq!(t.view_offset(&View { base_level: 1, ..Default::default() }), 0x40000);
        assert_eq!(t.view_offset(&View { base_level: 2, ..Default::default() }), 0x50000);
        let t = texture(4, (64, 64, 4), 1, 0x25);
        assert_eq!(t.view_offset(&View { min_layer: 2, ..Default::default() }), 0x8000);
        let t = texture(1, (128, 128, 1), 2, 0x41);
        assert_eq!(t.view_offset(&View { base_level: 1, ..Default::default() }), 0x2000);
        let t = texture(2, (32, 32, 32), 6, 0x01);
        assert_eq!(t.view_offset(&View { base_level: 1, min_layer: 1, ..Default::default() }), 0x10200);
        let t = texture(2, (8, 8, 8), 1, 0x3f);
        assert_eq!(t.view_offset(&View { min_layer: 3, ..Default::default() }), 0x600);
    }
}
