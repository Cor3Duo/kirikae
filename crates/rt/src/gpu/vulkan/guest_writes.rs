use crate::gpu::fxhash::FastMap as HashMap;
use super::Renderer;
use super::images::ImageKey;
use super::textures::{CROP, DERIVED};

#[derive(Default)]
pub(super) struct PendingPages {
    pub(super) pages: HashMap<u64, Vec<(u64, Pending)>>,
    pub(super) next: u64,
}

pub(super) type Pending = (u64, u64, u64, u64);

pub(super) fn pages_of(cpu: u64, len: u64) -> std::ops::RangeInclusive<u64> {
    (cpu >> 16)..=(cpu.saturating_add(len.max(1) - 1) >> 16)
}

impl PendingPages {
    pub(super) fn add(&mut self, p: Pending) {
        let n = self.next;
        self.next += 1;
        for page in pages_of(p.0, p.1) {
            self.pages.entry(page).or_default().push((n, p));
        }
    }

    pub(super) fn retire(&mut self, serial: u64) {
        self.pages.retain(|_, v| {
            v.retain(|e| e.1.2 != serial);
            !v.is_empty()
        });
    }

    pub(super) fn near(&self, cpu: u64, len: u64) -> bool {
        !self.pages.is_empty() && pages_of(cpu, len).any(|p| self.pages.contains_key(&p))
    }

    pub(super) fn overlapping(&self, cpu: u64, len: u64) -> impl Iterator<Item = &(u64, Pending)> {
        pages_of(cpu, len).filter_map(|p| self.pages.get(&p)).flatten().filter(move |(_, p)| p.0 < cpu + len && cpu < p.0 + p.1)
    }
}

#[derive(Default)]
pub(super) struct CpWrites {
    pub(super) all: Vec<(u64, u64)>,
    pub(super) by_page: HashMap<u64, Vec<u32>>,
    pub(super) wide: Vec<u32>,
}

pub(super) const WIDE_WRITE: u64 = 64;

impl CpWrites {
    pub(super) fn clear(&mut self) {
        self.all.clear();
        self.by_page.clear();
        self.wide.clear();
    }

    pub(super) fn len(&self) -> usize {
        self.all.len()
    }

    pub(super) fn push(&mut self, cpu: u64, len: u64) {
        let i = self.all.len() as u32;
        self.all.push((cpu, cpu + len));
        let pages = pages_of(cpu, len);
        if pages.end() - pages.start() >= WIDE_WRITE {
            self.wide.push(i);
        } else {
            for p in pages {
                self.by_page.entry(p).or_default().push(i);
            }
        }
    }

    pub(super) fn since(&self, c: u64, size: u64, seen: usize) -> bool {
        let hits = |i: &u32| {
            let (s, e) = self.all[*i as usize];
            s < c + size && c < e
        };
        let pages = pages_of(c, size);
        if pages.end() - pages.start() >= (self.all.len() - seen) as u64 {
            return self.all[seen..].iter().any(|&(s, e)| s < c + size && c < e);
        }
        self.wide.iter().rev().take_while(|&&i| i as usize >= seen).any(hits)
            || pages.into_iter().any(|p| self.by_page.get(&p).is_some_and(|v| v.iter().rev().take_while(|&&i| i as usize >= seen).any(hits)))
    }
}

impl Renderer {
    pub(super) fn touch(&mut self, key: ImageKey) {
        self.clock += 1;
        let img = self.images.get_mut(&key).expect("image exists");
        img.gpu_dirty = true;
        img.stamp = self.clock;
        self.restamped(key);
    }

    pub(super) fn restamped(&mut self, key: ImageKey) {
        if key.cpu & DERIVED == 0 && self.newest.insert(key.cpu, key) != Some(key) && self.at(key.cpu).len() > 1 {
            self.forget_textures(key.cpu, false);
        }
    }

    pub fn note_write(&mut self, cpu: u64, len: u64) {
        if !self.checked.is_empty() {
            self.note_cp_write(cpu, len);
        }
    }

    pub(super) fn note_cp_write(&mut self, cpu: u64, len: u64) {
        self.cp_writes.push(cpu, len);
    }

    pub fn forget_guest_range(&mut self, cpu: u64, len: u64) {
        let stale: Vec<ImageKey> = self.by_cpu.range(cpu..cpu + len).flat_map(|(_, keys)| keys.iter().copied()).filter(|k| k.cpu & CROP == 0).collect();
        for key in stale {
            self.destroy_image(key);
        }
    }
}
