use std::collections::BTreeMap;
use std::io;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub const ADDR_BITS: u32 = 39;
pub const ADDR_MASK: u64 = (1 << ADDR_BITS) - 1;
pub const PAGE: u64 = 0x1000;
const GUARD: usize = 0x10000;
const RESERVE: usize = (1usize << ADDR_BITS) + GUARD;

pub mod state {
    pub const FREE: u32 = 0x00;
    pub const NORMAL: u32 = 0x02;
    pub const CODE_STATIC: u32 = 0x03;
    pub const CODE_MUTABLE: u32 = 0x04;
    pub const HEAP: u32 = 0x05;
    pub const SHARED: u32 = 0x06;
    pub const STACK: u32 = 0x0b;
    pub const THREAD_LOCAL: u32 = 0x0c;
}

pub mod perm {
    pub const NONE: u32 = 0;
    pub const R: u32 = 1;
    pub const RW: u32 = 3;
    pub const RX: u32 = 5;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub range: Range<u64>,
    pub state: u32,
    pub perm: u32,
}

#[derive(Debug)]
pub enum PhysicalError {
    State,
    Limit,
    Host(io::Error),
}

fn insert_merged(mapped: &mut BTreeMap<u64, Region>, mut r: Region) {
    let same = |o: &Region, r: &Region| o.state == r.state && o.perm == r.perm;
    if let Some((&key, prev)) = mapped.range(..r.range.start).next_back()
        && prev.range.end == r.range.start
        && same(prev, &r)
    {
        r.range.start = key;
        mapped.remove(&key);
    }
    if let Some(next) = mapped.get(&r.range.end)
        && same(next, &r)
    {
        let key = r.range.end;
        r.range.end = next.range.end;
        mapped.remove(&key);
    }
    mapped.insert(r.range.start, r);
}

pub struct GuestMemory {
    base: *mut u8,
    mapped: Mutex<BTreeMap<u64, Region>>,
    unmaps: AtomicU64,
}

type MappedHit = (usize, u64, Range<u64>);

thread_local! {
    static LAST_MAPPED: std::cell::RefCell<[Option<MappedHit>; 4]> = const { std::cell::RefCell::new([None, None, None, None]) };
}

unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

impl GuestMemory {
    pub fn new() -> io::Result<GuestMemory> {
        let base = unsafe { sys::reserve(RESERVE) };
        if base.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(GuestMemory { base, mapped: Mutex::new(BTreeMap::new()), unmaps: AtomicU64::new(0) })
    }

    pub fn base(&self) -> *mut u8 {
        self.base
    }

    pub fn map(&self, addr: u64, size: u64) -> io::Result<()> {
        self.map_as(addr, size, state::NORMAL, perm::RW)
    }

    pub fn map_as(&self, addr: u64, size: u64, state: u32, perm: u32) -> io::Result<()> {
        let start = addr & !(PAGE - 1);
        let end = (addr + size).div_ceil(PAGE) * PAGE;
        if end > ADDR_MASK + 1 || start >= end {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "mapping outside the guest address space"));
        }
        let mut mapped = self.mapped.lock().unwrap();
        if mapped.range(..end).next_back().is_some_and(|(_, r)| r.range.end > start) {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{start:#x}..{end:#x} overlaps a mapping")));
        }
        if !unsafe { sys::commit(self.base.add(start as usize), (end - start) as usize) } {
            return Err(io::Error::last_os_error());
        }
        mapped.insert(start, Region { range: start..end, state, perm });
        Ok(())
    }

    pub fn unmap(&self, addr: u64, size: u64) -> io::Result<()> {
        let start = addr & !(PAGE - 1);
        let end = (addr + size).div_ceil(PAGE) * PAGE;
        self.unmap_locked(&mut self.mapped.lock().unwrap(), start, end)
    }

    fn unmap_locked(&self, mapped: &mut BTreeMap<u64, Region>, start: u64, end: u64) -> io::Result<()> {
        self.unmaps.fetch_add(1, Ordering::AcqRel);
        let touched: Vec<u64> = mapped.range(..end).filter(|(_, r)| r.range.end > start).map(|(s, _)| *s).collect();
        for key in touched {
            let r = mapped.remove(&key).unwrap();
            if r.range.start < start {
                mapped.insert(r.range.start, Region { range: r.range.start..start, ..r.clone() });
            }
            if r.range.end > end {
                mapped.insert(end, Region { range: end..r.range.end, ..r.clone() });
            }
        }
        if !unsafe { sys::decommit(self.base.add(start as usize), (end - start) as usize) } {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn map_physical(&self, start: u64, end: u64, budget: u64) -> Result<u64, PhysicalError> {
        let mut mapped = self.mapped.lock().unwrap();
        let mut gaps = Vec::new();
        let mut at = start;
        for (_, r) in mapped.range(..end).filter(|(_, r)| r.range.end > start) {
            if r.state != state::HEAP {
                return Err(PhysicalError::State);
            }
            if r.range.start > at {
                gaps.push(at..r.range.start);
            }
            at = r.range.end;
        }
        if at < end {
            gaps.push(at..end);
        }
        let total: u64 = gaps.iter().map(|g| g.end - g.start).sum();
        if total > budget {
            return Err(PhysicalError::Limit);
        }
        for g in &gaps {
            if !unsafe { sys::commit(self.base.add(g.start as usize), (g.end - g.start) as usize) } {
                return Err(PhysicalError::Host(io::Error::last_os_error()));
            }
        }
        for g in gaps {
            insert_merged(&mut mapped, Region { range: g, state: state::HEAP, perm: perm::RW });
        }
        Ok(total)
    }

    pub fn unmap_physical(&self, start: u64, end: u64) -> Result<u64, PhysicalError> {
        let mut mapped = self.mapped.lock().unwrap();
        let mut held = 0;
        for (_, r) in mapped.range(..end).filter(|(_, r)| r.range.end > start) {
            if r.state != state::HEAP {
                return Err(PhysicalError::State);
            }
            held += r.range.end.min(end) - r.range.start.max(start);
        }
        self.unmap_locked(&mut mapped, start, end).map_err(PhysicalError::Host)?;
        Ok(held)
    }

    pub fn protect(&self, start: u64, end: u64, perm: u32, allowed: &[u32]) -> bool {
        let mut mapped = self.mapped.lock().unwrap();
        let mut at = start;
        let mut touched = Vec::new();
        while at < end {
            match mapped.range(..=at).next_back() {
                Some((&key, r)) if r.range.contains(&at) && allowed.contains(&r.state) => {
                    touched.push(key);
                    at = r.range.end;
                }
                _ => return false,
            }
        }
        for key in touched {
            let r = mapped.remove(&key).unwrap();
            if r.range.start < start {
                mapped.insert(r.range.start, Region { range: r.range.start..start, ..r.clone() });
            }
            if r.range.end > end {
                mapped.insert(end, Region { range: end..r.range.end, ..r.clone() });
            }
            let inner = r.range.start.max(start)..r.range.end.min(end);
            insert_merged(&mut mapped, Region { range: inner, state: r.state, perm });
        }
        true
    }

    pub fn copy(&self, dst: u64, src: u64, len: u64) -> bool {
        if !self.is_mapped(dst, len) || !self.is_mapped(src, len) {
            return false;
        }
        unsafe { std::ptr::copy(self.base.add(src as usize), self.base.add(dst as usize), len as usize) };
        true
    }

    pub fn query(&self, addr: u64) -> Region {
        let mapped = self.mapped.lock().unwrap();
        if let Some((_, r)) = mapped.range(..=addr).next_back()
            && r.range.contains(&addr)
        {
            return r.clone();
        }
        let start = mapped.range(..=addr).next_back().map_or(0, |(_, r)| r.range.end);
        let end = mapped.range(addr..).next().map_or(ADDR_MASK + 1, |(s, _)| *s);
        Region { range: start..end, state: state::FREE, perm: perm::NONE }
    }

    pub fn is_mapped(&self, addr: u64, len: u64) -> bool {
        let end = addr.saturating_add(len);
        let me = self as *const Self as usize;
        let unmaps = self.unmaps.load(Ordering::Acquire);
        let hit = LAST_MAPPED.with_borrow(|last| last.iter().flatten().any(|(m, n, r)| (*m, *n) == (me, unmaps) && r.start <= addr && end <= r.end));
        if hit {
            return true;
        }
        let regions = self.mapped.lock().unwrap();
        let mut at = addr;
        let mut first = addr;
        while at < end {
            match regions.range(..=at).next_back() {
                Some((_, r)) if r.range.contains(&at) => {
                    first = first.min(r.range.start);
                    at = r.range.end;
                }
                _ => return false,
            }
        }
        if at > addr {
            let entry = (me, self.unmaps.load(Ordering::Acquire), first..at);
            LAST_MAPPED.with_borrow_mut(|last| {
                last.rotate_right(1);
                last[0] = Some(entry);
            });
        }
        true
    }

    pub fn bytes(&self, addr: u64, len: u64) -> Option<&[u8]> {
        if !self.is_mapped(addr, len) {
            return None;
        }
        // SAFETY: the range is committed (checked above) inside the reservation.
        Some(unsafe { std::slice::from_raw_parts(self.base.add(addr as usize), len as usize) })
    }

    pub fn write(&self, addr: u64, bytes: &[u8]) -> bool {
        if !self.is_mapped(addr, bytes.len() as u64) {
            return false;
        }
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(addr as usize), bytes.len()) };
        true
    }

    pub fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
        if !self.is_mapped(addr, len) {
            return None;
        }
        let mut out = vec![0u8; len as usize];
        unsafe { std::ptr::copy_nonoverlapping(self.base.add(addr as usize), out.as_mut_ptr(), len as usize) };
        Some(out)
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;

    const MEM_DECOMMIT: u32 = 0x4000;

    unsafe extern "system" {
        fn VirtualAlloc(address: *const c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
    }

    pub unsafe fn decommit(at: *mut u8, size: usize) -> bool {
        unsafe { VirtualFree(at as *mut c_void, size, MEM_DECOMMIT) != 0 }
    }

    pub unsafe fn reserve(size: usize) -> *mut u8 {
        unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_NOACCESS) as *mut u8 }
    }

    pub unsafe fn commit(at: *mut u8, size: usize) -> bool {
        unsafe { !VirtualAlloc(at as *const c_void, size, MEM_COMMIT, PAGE_READWRITE).is_null() }
    }
}

#[cfg(unix)]
mod sys {
    pub unsafe fn reserve(size: usize) -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED { std::ptr::null_mut() } else { p as *mut u8 }
    }

    pub unsafe fn commit(at: *mut u8, size: usize) -> bool {
        unsafe { libc::mprotect(at as *mut libc::c_void, size, libc::PROT_READ | libc::PROT_WRITE) == 0 }
    }

    pub unsafe fn decommit(at: *mut u8, size: usize) -> bool {
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED;
        unsafe { libc::mmap(at as *mut libc::c_void, size, libc::PROT_NONE, flags, -1, 0) != libc::MAP_FAILED }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_read_write() {
        let m = GuestMemory::new().unwrap();
        m.map(0x1000_0000, 0x2000).unwrap();
        assert!(m.write(0x1000_0ff0, b"hello world, across a page"));
        assert_eq!(&m.read(0x1000_0ff0, 5).unwrap(), b"hello");
        assert!(!m.is_mapped(0x1000_2000, 1));
        assert!(m.read(0x2000_0000, 1).is_none());
        assert!(m.map(0x1000_1000, 0x1000).is_err(), "overlap");
        let gap = m.query(0x1800_0000);
        assert_eq!((gap.range.start, gap.state), (0x1000_2000, state::FREE));
        assert_eq!(m.query(0x1000_0010).range, 0x1000_0000..0x1000_2000);
        m.unmap(0x1000_0000, 0x1000).unwrap();
        assert!(!m.is_mapped(0x1000_0000, 1));
        assert_eq!(m.query(0x1000_1000).range, 0x1000_1000..0x1000_2000);
        assert_eq!(&m.read(0x1000_1000, 5).unwrap(), b"oss a");
    }

    #[test]
    fn physical_memory_fills_gaps_and_merges() {
        let m = GuestMemory::new().unwrap();
        let base = 0x10_0000_0000;
        assert_eq!(m.map_physical(base + 0x2000, base + 0x3000, u64::MAX).unwrap(), 0x1000);
        assert!(m.write(base + 0x2000, b"kept"));
        assert_eq!(m.map_physical(base, base + 0x5000, u64::MAX).unwrap(), 0x4000);
        assert_eq!(&m.read(base + 0x2000, 4).unwrap(), b"kept");
        assert_eq!(m.read(base + 0x4000, 4).unwrap(), vec![0; 4], "fresh pages are zeroed");
        let r = m.query(base + 0x1234);
        assert_eq!((r.range, r.state, r.perm), (base..base + 0x5000, state::HEAP, perm::RW), "one merged block");
        assert_eq!(m.map_physical(base, base + 0x5000, 0).unwrap(), 0);
        assert!(matches!(m.map_physical(base, base + 0x7000, 0x1000), Err(PhysicalError::Limit)));
        assert!(!m.is_mapped(base + 0x5000, 1), "a refused mapping changes nothing");

        assert_eq!(m.unmap_physical(base + 0x1000, base + 0x2000).unwrap(), 0x1000);
        assert!(!m.is_mapped(base + 0x1000, 1));
        assert_eq!(m.unmap_physical(base, base + 0x8000).unwrap(), 0x4000);
        assert_eq!(m.query(base).state, state::FREE);
        m.map_physical(base + 0x2000, base + 0x3000, u64::MAX).unwrap();
        assert_eq!(m.read(base + 0x2000, 4).unwrap(), vec![0; 4]);
    }

    #[test]
    fn physical_memory_rejects_other_mappings() {
        let m = GuestMemory::new().unwrap();
        let base = 0x10_0000_0000;
        m.map_as(base + 0x1000, 0x1000, state::STACK, perm::RW).unwrap();
        assert!(matches!(m.map_physical(base, base + 0x3000, u64::MAX), Err(PhysicalError::State)));
        assert!(!m.is_mapped(base, 1));
        assert!(matches!(m.unmap_physical(base, base + 0x3000), Err(PhysicalError::State)));
        assert!(m.is_mapped(base + 0x1000, 1));
    }

    #[test]
    fn protect_splits_and_rejoins() {
        let m = GuestMemory::new().unwrap();
        let base = 0x2_0000_0000;
        m.map_as(base, 0x4000, state::HEAP, perm::RW).unwrap();
        assert!(m.protect(base + 0x1000, base + 0x2000, perm::R, &[state::HEAP]));
        assert_eq!(m.query(base).range, base..base + 0x1000);
        let mid = m.query(base + 0x1000);
        assert_eq!((mid.range, mid.perm), (base + 0x1000..base + 0x2000, perm::R));
        assert_eq!(m.query(base + 0x3000).range, base + 0x2000..base + 0x4000);
        assert!(m.protect(base + 0x1000, base + 0x2000, perm::RW, &[state::HEAP]));
        assert_eq!(m.query(base).range, base..base + 0x4000, "equal neighbours merge back");
        assert!(!m.protect(base, base + 0x5000, perm::R, &[state::HEAP]), "past the end");
        assert!(!m.protect(base, base + 0x1000, perm::R, &[state::STACK]), "state not allowed");
        assert_eq!(m.query(base).perm, perm::RW, "a refused call changes nothing");
    }
}
