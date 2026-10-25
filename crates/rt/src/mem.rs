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
    /// `MemType_Stack`. On 2.0.0+ `svcMapMemory` only maps into the stack
    /// region and the kernel reports the destination as a stack, so the two
    /// names share the value.
    pub const STACK: u32 = 0x0b;
    pub const MAPPED: u32 = STACK;
    pub const THREAD_LOCAL: u32 = 0x0c;
    pub const TRANSFER_ISOLATED: u32 = 0x0d;
    pub const TRANSFER: u32 = 0x0e;
    pub const CODE_READONLY: u32 = 0x14;
    pub const CODE_WRITABLE: u32 = 0x15;
    /// `MemType_Insecure`, the pages `svcMapInsecurePhysicalMemory` allocates.
    pub const INSECURE: u32 = 0x17;
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

/// Why a mirrored range (`svcMapMemory`) could not be created or removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorError {
    /// The size is zero or not a whole number of pages.
    Size,
    /// An address is not page aligned or the range does not fit.
    Address,
    /// The source is not a single mapping that may be lent, the destination
    /// is not free, or the source is already lent somewhere else.
    State,
}

/// A range the guest mapped twice with `svcMapMemory`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mirror {
    pub src: u64,
    pub size: u64,
    /// The permission `svcUnmapMemory` restores on the source.
    pub perm: u32,
}

fn overlaps(a: u64, b: u64, c: u64, d: u64) -> bool {
    a < d && c < b
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

/// Splits and retags `start..end`, returning the slices it replaced. Every
/// page must be mapped with a state in `allowed`; on any other page nothing
/// changes.
fn reprotect_locked(
    mapped: &mut BTreeMap<u64, Region>,
    start: u64,
    end: u64,
    state: Option<u32>,
    perm: u32,
    allowed: &[u32],
) -> Option<Vec<Region>> {
    let mut at = start;
    let mut touched = Vec::new();
    while at < end {
        match mapped.range(..=at).next_back() {
            Some((&key, r)) if r.range.contains(&at) && allowed.contains(&r.state) => {
                touched.push(key);
                at = r.range.end;
            }
            _ => return None,
        }
    }
    let mut saved = Vec::with_capacity(touched.len());
    for key in touched {
        let r = mapped.remove(&key)?;
        if r.range.start < start {
            mapped.insert(
                r.range.start,
                Region {
                    range: r.range.start..start,
                    ..r.clone()
                },
            );
        }
        if r.range.end > end {
            mapped.insert(
                end,
                Region {
                    range: end..r.range.end,
                    ..r.clone()
                },
            );
        }
        let inner = r.range.start.max(start)..r.range.end.min(end);
        saved.push(Region {
            range: inner.clone(),
            state: r.state,
            perm: r.perm,
        });
        insert_merged(
            mapped,
            Region {
                range: inner,
                state: state.unwrap_or(r.state),
                perm,
            },
        );
    }
    Some(saved)
}

pub struct GuestMemory {
    base: *mut u8,
    mapped: Mutex<BTreeMap<u64, Region>>,
    /// Mirrored ranges (`svcMapMemory`) keyed by their destination.
    mirrors: Mutex<BTreeMap<u64, Mirror>>,
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
        Ok(GuestMemory {
            base,
            mapped: Mutex::new(BTreeMap::new()),
            mirrors: Mutex::new(BTreeMap::new()),
            unmaps: AtomicU64::new(0),
        })
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
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mapping outside the guest address space",
            ));
        }
        let mut mapped = self.mapped.lock().unwrap();
        if mapped
            .range(..end)
            .next_back()
            .is_some_and(|(_, r)| r.range.end > start)
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("{start:#x}..{end:#x} overlaps a mapping"),
            ));
        }
        if !unsafe { sys::commit(self.base.add(start as usize), (end - start) as usize) } {
            return Err(io::Error::last_os_error());
        }
        mapped.insert(
            start,
            Region {
                range: start..end,
                state,
                perm,
            },
        );
        Ok(())
    }

    pub fn unmap(&self, addr: u64, size: u64) -> io::Result<()> {
        let start = addr & !(PAGE - 1);
        let end = (addr + size).div_ceil(PAGE) * PAGE;
        self.unmap_locked(&mut self.mapped.lock().unwrap(), start, end)
    }

    fn unmap_locked(
        &self,
        mapped: &mut BTreeMap<u64, Region>,
        start: u64,
        end: u64,
    ) -> io::Result<()> {
        self.unmaps.fetch_add(1, Ordering::AcqRel);
        let touched: Vec<u64> = mapped
            .range(..end)
            .filter(|(_, r)| r.range.end > start)
            .map(|(s, _)| *s)
            .collect();
        for key in touched {
            let r = mapped.remove(&key).unwrap();
            if r.range.start < start {
                mapped.insert(
                    r.range.start,
                    Region {
                        range: r.range.start..start,
                        ..r.clone()
                    },
                );
            }
            if r.range.end > end {
                mapped.insert(
                    end,
                    Region {
                        range: end..r.range.end,
                        ..r.clone()
                    },
                );
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
            if !unsafe { sys::commit(self.base.add(g.start as usize), (g.end - g.start) as usize) }
            {
                return Err(PhysicalError::Host(io::Error::last_os_error()));
            }
        }
        for g in gaps {
            insert_merged(
                &mut mapped,
                Region {
                    range: g,
                    state: state::HEAP,
                    perm: perm::RW,
                },
            );
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
        self.unmap_locked(&mut mapped, start, end)
            .map_err(PhysicalError::Host)?;
        Ok(held)
    }

    /// `svcMapInsecurePhysicalMemory`: commits zero-filled pages for the whole
    /// range. Unlike `map_physical`, the range must be completely free: the
    /// insecure pool hands out new pages, it does not adopt existing heap.
    pub fn map_insecure(&self, start: u64, end: u64, budget: u64) -> Result<u64, PhysicalError> {
        if start >= end || !start.is_multiple_of(PAGE) || !end.is_multiple_of(PAGE) {
            return Err(PhysicalError::State);
        }
        let mut mapped = self.mapped.lock().unwrap();
        if mapped.range(..end).any(|(_, r)| r.range.end > start) {
            return Err(PhysicalError::State);
        }
        let total = end - start;
        if total > budget {
            return Err(PhysicalError::Limit);
        }
        if !unsafe { sys::commit(self.base.add(start as usize), total as usize) } {
            return Err(PhysicalError::Host(io::Error::last_os_error()));
        }
        insert_merged(
            &mut mapped,
            Region {
                range: start..end,
                state: state::INSECURE,
                perm: perm::RW,
            },
        );
        Ok(total)
    }

    /// `svcUnmapInsecurePhysicalMemory`: every page must belong to an insecure
    /// mapping; the range may be a part of one.
    pub fn unmap_insecure(&self, start: u64, end: u64) -> Result<u64, PhysicalError> {
        if start >= end || !start.is_multiple_of(PAGE) || !end.is_multiple_of(PAGE) {
            return Err(PhysicalError::State);
        }
        let mut mapped = self.mapped.lock().unwrap();
        let mut at = start;
        while at < end {
            match mapped.range(..=at).next_back() {
                Some((_, r)) if r.range.contains(&at) && r.state == state::INSECURE => {
                    at = r.range.end;
                }
                _ => return Err(PhysicalError::State),
            }
        }
        self.unmap_locked(&mut mapped, start, end)
            .map_err(PhysicalError::Host)?;
        Ok(end - start)
    }

    pub fn protect(&self, start: u64, end: u64, perm: u32, allowed: &[u32]) -> bool {
        self.reprotect(start, end, None, perm, allowed)
    }

    /// Reassigns the state (and permission) of a mapped range, splitting and
    /// merging the region map the way `protect` does. `false` when any page of
    /// the range is unmapped or its state is not in `allowed`.
    pub fn retag(&self, start: u64, end: u64, state: u32, perm: u32, allowed: &[u32]) -> bool {
        self.reprotect(start, end, Some(state), perm, allowed)
    }

    /// Retags a range and returns the tags it replaced, so the caller can put
    /// them back when its claim ends. Transfer memory uses this: the creator's
    /// pages are retagged and locked for as long as the object lives, and the
    /// kernel restores their state and permission when the last handle closes.
    pub fn retag_leased(
        &self,
        start: u64,
        end: u64,
        state: u32,
        perm: u32,
        allowed: &[u32],
    ) -> Option<Vec<Region>> {
        let mut mapped = self.mapped.lock().unwrap();
        reprotect_locked(&mut mapped, start, end, Some(state), perm, allowed)
    }

    /// Puts back the tags returned by `retag_leased`. Every saved slice must
    /// still be transfer memory: `false` means the pages were released or
    /// reused since and nothing is changed.
    pub fn restore(&self, saved: &[Region]) -> bool {
        if saved.is_empty() {
            return false;
        }
        let mut mapped = self.mapped.lock().unwrap();
        for r in saved {
            let mut at = r.range.start;
            while at < r.range.end {
                match mapped.range(..=at).next_back() {
                    Some((_, cur)) if cur.range.contains(&at) && cur.state == state::TRANSFER => {
                        at = cur.range.end;
                    }
                    _ => return false,
                }
            }
        }
        for r in saved {
            if reprotect_locked(
                &mut mapped,
                r.range.start,
                r.range.end,
                Some(r.state),
                r.perm,
                &[state::TRANSFER],
            )
            .is_none()
            {
                return false;
            }
        }
        true
    }

    fn reprotect(
        &self,
        start: u64,
        end: u64,
        state: Option<u32>,
        perm: u32,
        allowed: &[u32],
    ) -> bool {
        let mut mapped = self.mapped.lock().unwrap();
        reprotect_locked(&mut mapped, start, end, state, perm, allowed).is_some()
    }

    /// `svcMapMemory`: makes `dst` a view of `src`. The kernel borrows the
    /// source pages (they become read-only to the guest) and maps the same
    /// pages at `dst`. The runtime keeps one host view, so the destination gets
    /// a copy and the source's permission drops to `perm::NONE`; `unmap_mirror`
    /// copies the contents back before restoring the source. The guest cannot
    /// observe the difference because the source is unreachable while borrowed,
    /// and a second borrow of the same pages is refused.
    pub fn map_mirror(&self, dst: u64, src: u64, size: u64) -> Result<(), MirrorError> {
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Err(MirrorError::Size);
        }
        if !dst.is_multiple_of(PAGE) || !src.is_multiple_of(PAGE) {
            return Err(MirrorError::Address);
        }
        let (Some(dst_end), Some(src_end)) = (dst.checked_add(size), src.checked_add(size)) else {
            return Err(MirrorError::Address);
        };
        if dst_end > ADDR_MASK + 1 || src_end > ADDR_MASK + 1 {
            return Err(MirrorError::Address);
        }
        let source = self.query(src);
        // The source may sit inside a larger region (the stack allocator maps a
        // page of the thread stack) and may be a mutable code page, which the
        // loader maps into the stack for the thread entry; only an unmapped source
        // is refused.
        if !source.range.contains(&src) || source.range.end < src_end || source.state == state::FREE
        {
            return Err(MirrorError::State);
        }
        // The same physical pages may be mapped at several destinations at once (the
        // SDK does that for stacks and shared buffers), so only a destination that is
        // already mapped is a conflict. The copy on unmap keeps the aliases coherent.
        let conflicts = {
            let mirrors = self.mirrors.lock().unwrap();
            mirrors
                .iter()
                .any(|(at, m)| overlaps(*at, *at + m.size, dst, dst_end))
        };
        if conflicts {
            return Err(MirrorError::State);
        }
        // The view is writable even when the source page is a guard or was left
        // without permissions: the SDK maps stack pages and expects to write them.
        if self.map_as(dst, size, state::MAPPED, perm::RW).is_err() {
            return Err(MirrorError::State);
        }
        if !self.copy(dst, src, size) {
            let _ = self.unmap(dst, size);
            return Err(MirrorError::State);
        }
        self.mirrors.lock().unwrap().insert(
            dst,
            Mirror {
                src,
                size,
                perm: source.perm,
            },
        );
        Ok(())
    }

    /// `svcUnmapMemory`: copies the borrowed pages back and restores the
    /// source's permission. `dst`, `src` and `size` must match the pair that
    /// `map_mirror` created.
    pub fn unmap_mirror(&self, dst: u64, src: u64, size: u64) -> Result<(), MirrorError> {
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Err(MirrorError::Size);
        }
        if !dst.is_multiple_of(PAGE) || !src.is_multiple_of(PAGE) {
            return Err(MirrorError::Address);
        }
        let Some(mirror) = self.mirrors.lock().unwrap().get(&dst).cloned() else {
            return Err(MirrorError::State);
        };
        if mirror.src != src || mirror.size != size {
            return Err(MirrorError::State);
        }
        if !self.copy(src, dst, size) || self.unmap(dst, size).is_err() {
            return Err(MirrorError::State);
        }
        self.mirrors.lock().unwrap().remove(&dst);
        Ok(())
    }

    /// The mirror whose destination covers `addr`, with its destination base.
    pub fn mirror(&self, addr: u64) -> Option<(u64, Mirror)> {
        let mirrors = self.mirrors.lock().unwrap();
        mirrors
            .range(..=addr)
            .next_back()
            .filter(|(dst, m)| addr < **dst + m.size)
            .map(|(dst, m)| (*dst, m.clone()))
    }

    pub fn copy(&self, dst: u64, src: u64, len: u64) -> bool {
        if !self.is_mapped(dst, len) || !self.is_mapped(src, len) {
            return false;
        }
        unsafe {
            std::ptr::copy(
                self.base.add(src as usize),
                self.base.add(dst as usize),
                len as usize,
            )
        };
        true
    }

    pub fn query(&self, addr: u64) -> Region {
        let mapped = self.mapped.lock().unwrap();
        if let Some((_, r)) = mapped.range(..=addr).next_back()
            && r.range.contains(&addr)
        {
            return r.clone();
        }
        let start = mapped
            .range(..=addr)
            .next_back()
            .map_or(0, |(_, r)| r.range.end);
        let end = mapped
            .range(addr..)
            .next()
            .map_or(ADDR_MASK + 1, |(s, _)| *s);
        Region {
            range: start..end,
            state: state::FREE,
            perm: perm::NONE,
        }
    }

    pub fn is_mapped(&self, addr: u64, len: u64) -> bool {
        let end = addr.saturating_add(len);
        let me = self as *const Self as usize;
        let unmaps = self.unmaps.load(Ordering::Acquire);
        let hit = LAST_MAPPED.with_borrow(|last| {
            last.iter()
                .flatten()
                .any(|(m, n, r)| (*m, *n) == (me, unmaps) && r.start <= addr && end <= r.end)
        });
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
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(addr as usize), bytes.len())
        };
        true
    }

    pub fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
        if !self.is_mapped(addr, len) {
            return None;
        }
        let mut out = vec![0u8; len as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.base.add(addr as usize),
                out.as_mut_ptr(),
                len as usize,
            )
        };
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
        fn VirtualAlloc(
            address: *const c_void,
            size: usize,
            kind: u32,
            protect: u32,
        ) -> *mut c_void;
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
        if p == libc::MAP_FAILED {
            std::ptr::null_mut()
        } else {
            p as *mut u8
        }
    }

    pub unsafe fn commit(at: *mut u8, size: usize) -> bool {
        unsafe {
            libc::mprotect(
                at as *mut libc::c_void,
                size,
                libc::PROT_READ | libc::PROT_WRITE,
            ) == 0
        }
    }

    pub unsafe fn decommit(at: *mut u8, size: usize) -> bool {
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED;
        unsafe {
            libc::mmap(at as *mut libc::c_void, size, libc::PROT_NONE, flags, -1, 0)
                != libc::MAP_FAILED
        }
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
        assert_eq!(
            m.map_physical(base + 0x2000, base + 0x3000, u64::MAX)
                .unwrap(),
            0x1000
        );
        assert!(m.write(base + 0x2000, b"kept"));
        assert_eq!(
            m.map_physical(base, base + 0x5000, u64::MAX).unwrap(),
            0x4000
        );
        assert_eq!(&m.read(base + 0x2000, 4).unwrap(), b"kept");
        assert_eq!(
            m.read(base + 0x4000, 4).unwrap(),
            vec![0; 4],
            "fresh pages are zeroed"
        );
        let r = m.query(base + 0x1234);
        assert_eq!(
            (r.range, r.state, r.perm),
            (base..base + 0x5000, state::HEAP, perm::RW),
            "one merged block"
        );
        assert_eq!(m.map_physical(base, base + 0x5000, 0).unwrap(), 0);
        assert!(matches!(
            m.map_physical(base, base + 0x7000, 0x1000),
            Err(PhysicalError::Limit)
        ));
        assert!(
            !m.is_mapped(base + 0x5000, 1),
            "a refused mapping changes nothing"
        );

        assert_eq!(
            m.unmap_physical(base + 0x1000, base + 0x2000).unwrap(),
            0x1000
        );
        assert!(!m.is_mapped(base + 0x1000, 1));
        assert_eq!(m.unmap_physical(base, base + 0x8000).unwrap(), 0x4000);
        assert_eq!(m.query(base).state, state::FREE);
        m.map_physical(base + 0x2000, base + 0x3000, u64::MAX)
            .unwrap();
        assert_eq!(m.read(base + 0x2000, 4).unwrap(), vec![0; 4]);
    }

    #[test]
    fn physical_memory_rejects_other_mappings() {
        let m = GuestMemory::new().unwrap();
        let base = 0x10_0000_0000;
        m.map_as(base + 0x1000, 0x1000, state::STACK, perm::RW)
            .unwrap();
        assert!(matches!(
            m.map_physical(base, base + 0x3000, u64::MAX),
            Err(PhysicalError::State)
        ));
        assert!(!m.is_mapped(base, 1));
        assert!(matches!(
            m.unmap_physical(base, base + 0x3000),
            Err(PhysicalError::State)
        ));
        assert!(m.is_mapped(base + 0x1000, 1));
    }

    #[test]
    fn a_lease_restores_the_tags_it_replaced() {
        let m = GuestMemory::new().unwrap();
        let base = 0x10_0000_0000;
        m.map_as(base, 0x2000, state::HEAP, perm::RW).unwrap();
        m.map_as(base + 0x2000, 0x2000, state::NORMAL, perm::R)
            .unwrap();
        let saved = m
            .retag_leased(
                base,
                base + 0x4000,
                state::TRANSFER,
                perm::NONE,
                &[state::HEAP, state::NORMAL],
            )
            .expect("both kinds of page may be leased");
        assert_eq!(saved.len(), 2, "the split keeps both original tags");
        assert_eq!(m.query(base).state, state::TRANSFER);
        assert_eq!(m.query(base + 0x3000).state, state::TRANSFER);
        assert_eq!(m.query(base).perm, perm::NONE);
        assert!(
            !m.retag_leased(
                base,
                base + 0x4000,
                state::TRANSFER,
                perm::NONE,
                &[state::HEAP, state::NORMAL]
            )
            .is_some(),
            "a leased block cannot be leased twice"
        );
        assert!(m.restore(&saved));
        let first = m.query(base);
        assert_eq!(
            (first.range, first.state, first.perm),
            (base..base + 0x2000, state::HEAP, perm::RW)
        );
        let second = m.query(base + 0x3000);
        assert_eq!(
            (second.state, second.perm),
            (state::NORMAL, perm::R),
            "every slice gets its own tag back"
        );
        assert!(!m.restore(&saved), "restore is not repeatable");
    }

    #[test]
    fn physical_memory_adopts_a_released_transfer_range() {
        let m = GuestMemory::new().unwrap();
        let base = 0x10_0000_0000;
        m.map_as(base, 0x2000, state::HEAP, perm::RW).unwrap();
        let saved = m
            .retag_leased(
                base,
                base + 0x2000,
                state::TRANSFER,
                perm::NONE,
                &[state::HEAP],
            )
            .expect("the heap pages become transfer memory");

        // A live transfer memory is a genuine conflict: the documented 0xD401
        // path, not something to move behind the guest's back.
        assert!(matches!(
            m.map_physical(base, base + 0x4000, u64::MAX),
            Err(PhysicalError::State)
        ));
        assert!(
            !m.is_mapped(base + 0x2000, 1),
            "a refused mapping commits nothing"
        );
        assert_eq!(m.query(base).state, state::TRANSFER);

        assert!(m.restore(&saved));
        // The restored pages are heap again, so adopting the range only has to
        // fill the hole after them.
        assert_eq!(
            m.map_physical(base, base + 0x4000, u64::MAX).unwrap(),
            0x2000
        );
        assert_eq!(m.query(base).state, state::HEAP);
        assert_eq!(
            m.query(base).range,
            base..base + 0x4000,
            "the released pages merge with the fresh heap"
        );
    }

    #[test]
    fn unmap_physical_restores_the_pages_to_free() {
        let m = GuestMemory::new().unwrap();
        let base = 0x10_0000_0000;
        assert_eq!(
            m.map_physical(base, base + 0x2000, u64::MAX).unwrap(),
            0x2000
        );
        assert_eq!(m.unmap_physical(base, base + 0x2000).unwrap(), 0x2000);
        assert_eq!(m.query(base).state, state::FREE);
        assert_eq!(
            m.map_physical(base, base + 0x2000, u64::MAX).unwrap(),
            0x2000,
            "the unmapped pages can be adopted again"
        );
    }

    #[test]
    fn insecure_memory_maps_zeroed_pages_and_only_unmaps_its_own() {
        let m = GuestMemory::new().unwrap();
        let base = 0x0800_0000;
        m.map_as(base + 0x2000, 0x1000, state::HEAP, perm::RW)
            .unwrap();

        assert!(matches!(
            m.map_insecure(base, base + 0x3000, u64::MAX),
            Err(PhysicalError::State)
        ));
        assert_eq!(
            m.map_insecure(base, base + 0x2000, u64::MAX).unwrap(),
            0x2000
        );
        assert_eq!(m.query(base).state, state::INSECURE);
        assert_eq!(m.query(base + 0x1000).perm, perm::RW);
        assert_eq!(m.read(base, 4).unwrap(), vec![0; 4], "pages are zeroed");
        assert_eq!(
            m.query(base + 0x2000).state,
            state::HEAP,
            "the heap next door keeps its state"
        );
        assert!(matches!(
            m.map_insecure(base + 0x4000, base + 0x6000, 0),
            Err(PhysicalError::Limit)
        ));
        assert!(
            matches!(
                m.map_insecure(base, base + 0x3000, u64::MAX),
                Err(PhysicalError::State)
            ),
            "an insecure mapping never adopts the heap"
        );

        assert_eq!(
            m.unmap_insecure(base + 0x1000, base + 0x2000).unwrap(),
            0x1000
        );
        assert_eq!(m.query(base).state, state::INSECURE);
        assert_eq!(m.query(base + 0x1000).state, state::FREE);
        assert!(matches!(
            m.unmap_insecure(base, base + 0x3000),
            Err(PhysicalError::State)
        ));
        assert!(matches!(
            m.unmap_insecure(base + 0x1000, base + 0x2000),
            Err(PhysicalError::State)
        ));
        assert_eq!(m.unmap_insecure(base, base + 0x1000).unwrap(), 0x1000);
        assert_eq!(m.query(base).state, state::FREE);
        assert_eq!(
            m.query(base + 0x2000).state,
            state::HEAP,
            "unmapping insecure pages never touches the heap"
        );
    }

    #[test]
    fn protect_splits_and_rejoins() {
        let m = GuestMemory::new().unwrap();
        let base = 0x2_0000_0000;
        m.map_as(base, 0x4000, state::HEAP, perm::RW).unwrap();
        assert!(m.protect(base + 0x1000, base + 0x2000, perm::R, &[state::HEAP]));
        assert_eq!(m.query(base).range, base..base + 0x1000);
        let mid = m.query(base + 0x1000);
        assert_eq!(
            (mid.range, mid.perm),
            (base + 0x1000..base + 0x2000, perm::R)
        );
        assert_eq!(m.query(base + 0x3000).range, base + 0x2000..base + 0x4000);
        assert!(m.protect(base + 0x1000, base + 0x2000, perm::RW, &[state::HEAP]));
        assert_eq!(
            m.query(base).range,
            base..base + 0x4000,
            "equal neighbours merge back"
        );
        assert!(
            !m.protect(base, base + 0x5000, perm::R, &[state::HEAP]),
            "past the end"
        );
        assert!(
            !m.protect(base, base + 0x1000, perm::R, &[state::STACK]),
            "state not allowed"
        );
        assert_eq!(
            m.query(base).perm,
            perm::RW,
            "a refused call changes nothing"
        );
    }

    #[test]
    fn a_mirror_borrows_the_source_and_copies_back() {
        let m = GuestMemory::new().unwrap();
        let base = 0x2_0000_0000;
        m.map_as(base, 0x2000, state::HEAP, perm::RW).unwrap();
        assert!(m.write(base, b"guard"));
        m.map_mirror(base + 0x2000, base, 0x1000).unwrap();
        assert_eq!(&m.read(base + 0x2000, 5).unwrap(), b"guard");
        let dst = m.query(base + 0x2000);
        assert_eq!((dst.state, dst.perm), (state::MAPPED, perm::RW));
        assert_eq!(
            m.query(base).perm,
            perm::RW,
            "the source keeps its permission"
        );
        assert_eq!(
            m.mirror(base + 0x2000).map(|(at, _)| at),
            Some(base + 0x2000)
        );
        assert!(m.write(base + 0x2000, b"GUARD"));
        // The same source may be aliased at several destinations, but a destination
        // that already holds a mapping is refused.
        m.map_mirror(base + 0x4000, base, 0x1000)
            .expect("a second destination aliases the same source");
        assert!(matches!(
            m.map_mirror(base + 0x2000, base, 0x1000),
            Err(MirrorError::State)
        ));
        m.unmap_mirror(base + 0x2000, base, 0x1000).unwrap();
        assert_eq!(&m.read(base, 5).unwrap(), b"GUARD", "writes come back");
        assert_eq!(m.query(base).perm, perm::RW);
        assert!(!m.is_mapped(base + 0x2000, 1));
        assert!(m.mirror(base + 0x2000).is_none());
    }

    #[test]
    fn mirror_and_retag_validation() {
        let m = GuestMemory::new().unwrap();
        m.map_as(0x1000_0000, 0x1000, state::NORMAL, perm::RW)
            .unwrap();
        assert_eq!(
            m.map_mirror(0x1000_1000, 0x1000_0000, 0),
            Err(MirrorError::Size)
        );
        assert_eq!(
            m.map_mirror(0x1000_1001, 0x1000_0000, 0x1000),
            Err(MirrorError::Address)
        );
        assert_eq!(
            m.map_mirror(0x2000_0000, 0x1000_0000, 0x1001),
            Err(MirrorError::Size)
        );
        assert_eq!(
            m.map_mirror(0x2000_0000, 0x3000_0000, 0x1000),
            Err(MirrorError::State),
            "the source must be mapped"
        );
        assert_eq!(
            m.unmap_mirror(0x2000_0000, 0x1000_0000, 0x1000),
            Err(MirrorError::State),
            "nothing was mapped there"
        );
        m.map_mirror(0x2000_0000, 0x1000_0000, 0x1000).unwrap();
        assert_eq!(
            m.unmap_mirror(0x2000_0000, 0x1000_0000, 0x2000),
            Err(MirrorError::State),
            "the size must match the mirror"
        );
        assert_eq!(
            m.unmap_mirror(0x2000_0000, 0x3000_0000, 0x1000),
            Err(MirrorError::State),
            "the source must match the mirror"
        );

        assert!(m.retag(
            0x1000_0000,
            0x1000_1000,
            state::TRANSFER,
            perm::R,
            &[state::NORMAL, state::MAPPED]
        ));
        let r = m.query(0x1000_0000);
        assert_eq!((r.state, r.perm), (state::TRANSFER, perm::R));
        assert!(!m.retag(
            0x1000_0000,
            0x1000_1000,
            state::TRANSFER,
            perm::R,
            &[state::HEAP]
        ));
    }
}
