use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub const ADDR_BITS: u32 = 39;
pub const ADDR_MASK: u64 = (1 << ADDR_BITS) - 1;
pub const PAGE: u64 = 0x1000;
const GUARD: usize = 0x10000;
const RESERVE: usize = (1usize << ADDR_BITS) + GUARD;

pub mod state {
    pub const FREE: u32 = 0x00;
    /// `MemType_Io`: pages the kernel exposed with `svcMapIoRegion`. They are
    /// device memory (the pool's backing store), not guest RAM, so the mapping
    /// calls that lend guest pages do not accept them.
    pub const IO: u32 = 0x01;
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
    /// Write-only. Guest mappings never take it (`svcSetMemoryPermission`
    /// rejects it), but a device mapping may be write-only.
    pub const W: u32 = 2;
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

// ---------------------------------------------------------------------------
// Device physical memory and device address spaces
//
// The runtime has no second physical address space and no IOMMU hardware, so
// this section models the two things the kernel offers to a guest that talks to
// a device, and nothing else. What is modelled:
//
// * An [`IoPool`] is the physical aperture of an `IoPoolType_PcieA2` pool,
//   0x1200_0000..0x2000_0000. It owns no bytes by itself: a pool's host
//   allocation is the set of [`IoRegion`]s the guest claimed inside it, each
//   with its own zero-filled host buffer. Creating a pool therefore never
//   reserves 224 MiB, and every mapped region is real memory the guest can read
//   and write.
// * An [`IoRegion`] of kind `MemoryMapping_IoRegister` is refused: device
//   registers only exist on the console. The other two kinds describe memory
//   and are backed.
// * A [`DeviceAddressSpace`] is the device-side page table an IOMMU would use:
//   it maps device addresses to pages of the one process the runtime runs. The
//   alias is real in both directions (a device read sees the guest's bytes and
//   a device write lands in guest memory), and [`DeviceAddressSpace::read`],
//   [`DeviceAddressSpace::write`] and [`DeviceAddressSpace::copy`] are the
//   entry points a synthetic device uses.
//
// What is not modelled: MMIO registers, a DMA engine, interrupts from a device,
// per-device IOMMU contexts and cache coherence. `svcAttachDeviceAddressSpace`
// records which device the guest attached; no hardware device reacts to it.
// ---------------------------------------------------------------------------

/// The physical aperture of the only IO pool the console documents
/// (`IoPoolType_PcieA2`): 0x1200_0000..0x2000_0000.
pub const IO_POOL_BASE: u64 = 0x1200_0000;
pub const IO_POOL_SIZE: u64 = 0x0e00_0000;
pub const IO_POOL_END: u64 = IO_POOL_BASE + IO_POOL_SIZE;

/// The largest device address space this runtime models. A device address space
/// is an IOMMU window; the console normally creates one over `0..0xFFFF_FFFF`,
/// which this ceiling accepts, and nothing above 4 GiB is backed.
pub const DEVICE_SPACE_LIMIT: u64 = 0x1_0000_0000;

/// `MemoryMapping` of `svcCreateIoRegion` (libnx's enum of the same name).
pub mod io_mapping {
    /// Device registers. Refused: the runtime does not fabricate MMIO.
    pub const IO_REGISTER: u32 = 0;
    /// Normal memory without a cache.
    pub const UNCACHED: u32 = 1;
    /// Normal memory.
    pub const MEMORY: u32 = 2;
}

/// Why an IO pool, an IO region or a device address space refused a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceError {
    /// Zero size or a size that is not a whole number of pages (0xCA01).
    Size,
    /// An address is not page aligned (0xCC01).
    Address,
    /// A range leaves the pool or the device address space, or overflows
    /// (0xDC01).
    Range,
    /// A range overlaps a live claim, is already mapped, or is not mapped
    /// (0xD401).
    State,
    /// The permission is not one the object supports (0xD801).
    Permission,
    /// The region asks for device registers, which only real hardware has.
    Registers,
    /// The host refused the backing allocation.
    Host,
}

/// The permission an IO region or a device mapping may carry: read, write or
/// both, never execute and never nothing.
pub fn device_perm(perm: u32) -> bool {
    matches!(perm, perm::R | perm::W | perm::RW)
}

/// A PCIe A2 IO pool: the aperture and the page ranges the guest claimed in it.
///
/// The kernel gives the pool a physical allocator, so two live regions can never
/// overlap. The runtime keeps that rule with a per-pool claim table, which also
/// bounds a pool's host memory at the aperture size (224 MiB) without reserving
/// it up front.
pub struct IoPool {
    pub base: u64,
    pub size: u64,
    /// Claimed ranges keyed by their physical base, with their size.
    claims: Mutex<BTreeMap<u64, u64>>,
}

impl IoPool {
    pub fn new(base: u64, size: u64) -> Arc<IoPool> {
        Arc::new(IoPool {
            base,
            size,
            claims: Mutex::new(BTreeMap::new()),
        })
    }

    /// The pool `svcCreateIoPool` builds for `IoPoolType_PcieA2`.
    pub fn pcie_a2() -> Arc<IoPool> {
        IoPool::new(IO_POOL_BASE, IO_POOL_SIZE)
    }

    pub fn end(&self) -> u64 {
        self.base + self.size
    }

    /// Whether `at..at+size` fits the aperture.
    pub fn contains(&self, at: u64, size: u64) -> bool {
        at >= self.base && size != 0 && at.checked_add(size).is_some_and(|e| e <= self.end())
    }

    /// Reserves `at..at+size` for a new region. The caller must have checked
    /// [`IoPool::contains`]; an overlapping claim is the kernel's invalid state.
    fn claim(&self, at: u64, size: u64) -> Result<(), DeviceError> {
        let mut claims = self.claims.lock().unwrap();
        if claims
            .range(..at + size)
            .any(|(&start, &len)| start + len > at)
        {
            return Err(DeviceError::State);
        }
        claims.insert(at, size);
        Ok(())
    }

    fn release(&self, at: u64) {
        self.claims.lock().unwrap().remove(&at);
    }

    /// Bytes currently claimed by live regions (diagnostics and tests).
    pub fn claimed(&self) -> u64 {
        self.claims.lock().unwrap().values().sum()
    }
}

/// One `svcCreateIoRegion` claim: a physical range of the pool with a real host
/// buffer behind it.
///
/// `svcMapIoRegion` makes the region visible at a guest address. The runtime
/// keeps one host view (the guest mapping is the source of truth while it
/// exists) and copies back into the shadow when it is unmapped, so the contents
/// survive between mappings and a synthetic device that reads through
/// [`IoRegion::read`] sees exactly what the guest wrote.
pub struct IoRegion {
    pub physical: u64,
    pub size: u64,
    /// One of [`io_mapping`].
    pub mapping: u32,
    /// The permission the region was created with.
    pub perm: u32,
    pool: Arc<IoPool>,
    state: Mutex<IoRegionState>,
}

struct IoRegionState {
    shadow: Vec<u8>,
    mapped_at: Option<u64>,
}

impl IoRegion {
    /// `svcCreateIoRegion`. The caller validates the enum value and the handle;
    /// this checks the range, the kind, the permission and the backing store.
    pub fn new(
        pool: Arc<IoPool>,
        physical: u64,
        size: u64,
        mapping: u32,
        perm: u32,
    ) -> Result<Arc<IoRegion>, DeviceError> {
        if !physical.is_multiple_of(PAGE) {
            return Err(DeviceError::Address);
        }
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Err(DeviceError::Size);
        }
        if !pool.contains(physical, size) {
            return Err(DeviceError::Range);
        }
        if mapping == io_mapping::IO_REGISTER {
            return Err(DeviceError::Registers);
        }
        if !device_perm(perm) {
            return Err(DeviceError::Permission);
        }
        let len = usize::try_from(size).map_err(|_| DeviceError::Size)?;
        let mut shadow = Vec::new();
        shadow
            .try_reserve_exact(len)
            .map_err(|_| DeviceError::Host)?;
        shadow.resize(len, 0);
        pool.claim(physical, size)?;
        Ok(Arc::new(IoRegion {
            physical,
            size,
            mapping,
            perm,
            pool,
            state: Mutex::new(IoRegionState {
                shadow,
                mapped_at: None,
            }),
        }))
    }

    pub fn mapped_at(&self) -> Option<u64> {
        self.state.lock().unwrap().mapped_at
    }

    /// `svcMapIoRegion`: makes the region visible at `addr` with `perm`, which
    /// must be the permission the region was created with. Only one mapping
    /// exists per region at a time.
    pub fn map(
        &self,
        memory: &GuestMemory,
        addr: u64,
        size: u64,
        perm: u32,
    ) -> Result<(), DeviceError> {
        if size != self.size {
            return Err(DeviceError::Size);
        }
        if !addr.is_multiple_of(PAGE) {
            return Err(DeviceError::Address);
        }
        if addr.checked_add(size).is_none_or(|end| end > ADDR_MASK + 1) {
            return Err(DeviceError::Range);
        }
        if perm != self.perm {
            return Err(DeviceError::Permission);
        }
        let mut st = self.state.lock().unwrap();
        if st.mapped_at.is_some() {
            return Err(DeviceError::State);
        }
        if memory.map_as(addr, size, state::IO, self.perm).is_err() {
            return Err(DeviceError::State);
        }
        if !memory.write(addr, &st.shadow) {
            let _ = memory.unmap(addr, size);
            return Err(DeviceError::State);
        }
        st.mapped_at = Some(addr);
        Ok(())
    }

    /// `svcUnmapIoRegion`: copies the guest pages back into the region and
    /// releases them.
    pub fn unmap(&self, memory: &GuestMemory, addr: u64, size: u64) -> Result<(), DeviceError> {
        if size != self.size {
            return Err(DeviceError::Size);
        }
        if !addr.is_multiple_of(PAGE) {
            return Err(DeviceError::Address);
        }
        let mut st = self.state.lock().unwrap();
        if st.mapped_at != Some(addr) {
            return Err(DeviceError::State);
        }
        let Some(bytes) = memory.read(addr, size) else {
            return Err(DeviceError::State);
        };
        if memory.unmap(addr, size).is_err() {
            return Err(DeviceError::State);
        }
        st.shadow = bytes;
        st.mapped_at = None;
        Ok(())
    }

    /// Drops any guest mapping (writing its contents back). `svcCloseHandle`
    /// calls this on the last handle so the guest pages do not leak; the
    /// contents stay in the region's shadow.
    pub fn release(&self, memory: &GuestMemory) -> bool {
        let mut st = self.state.lock().unwrap();
        let Some(at) = st.mapped_at else {
            return true;
        };
        let Some(bytes) = memory.read(at, self.size) else {
            return false;
        };
        if memory.unmap(at, self.size).is_err() {
            return false;
        }
        st.shadow = bytes;
        st.mapped_at = None;
        true
    }

    /// Reads `len` bytes at `offset` from the device side: through the guest
    /// mapping while the region is mapped, otherwise from the shadow.
    pub fn read(&self, memory: &GuestMemory, offset: u64, len: u64) -> Option<Vec<u8>> {
        let st = self.state.lock().unwrap();
        let end = offset.checked_add(len)?;
        if end > self.size {
            return None;
        }
        match st.mapped_at {
            Some(at) => memory.read(at + offset, len),
            None => {
                let start = usize::try_from(offset).ok()?;
                let end = usize::try_from(end).ok()?;
                Some(st.shadow.get(start..end)?.to_vec())
            }
        }
    }

    /// Writes `bytes` at `offset` on the device side. `false` when the range
    /// leaves the region.
    pub fn write(&self, memory: &GuestMemory, offset: u64, bytes: &[u8]) -> bool {
        let mut st = self.state.lock().unwrap();
        let Ok(len) = u64::try_from(bytes.len()) else {
            return false;
        };
        let Some(end) = offset.checked_add(len) else {
            return false;
        };
        if end > self.size {
            return false;
        }
        match st.mapped_at {
            Some(at) => memory.write(at + offset, bytes),
            None => {
                let (Ok(start), Ok(end)) = (usize::try_from(offset), usize::try_from(end)) else {
                    return false;
                };
                let Some(dst) = st.shadow.get_mut(start..end) else {
                    return false;
                };
                dst.copy_from_slice(bytes);
                true
            }
        }
    }
}

impl Drop for IoRegion {
    fn drop(&mut self) {
        self.pool.release(self.physical);
    }
}

/// A device-side page mapping: a guest page exposed at a device address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceMapping {
    pub guest: u64,
    pub size: u64,
    pub perm: u32,
}

/// The page table of a modelled device address space (an IOMMU window).
///
/// `svcMapDeviceAddressSpace*` fills it with guest pages and
/// `svcUnmapDeviceAddressSpace` removes them. Synthetic devices reach guest
/// memory through [`DeviceAddressSpace::read`], [`DeviceAddressSpace::write`]
/// and [`DeviceAddressSpace::copy`], which enforce the mapped permission.
pub struct DeviceAddressSpace {
    pub base: u64,
    pub size: u64,
    state: Mutex<DeviceState>,
}

struct DeviceState {
    /// The `DeviceName`s the guest attached, in attach order.
    attached: BTreeSet<u32>,
    /// Mappings keyed by device address, contiguous and non-overlapping.
    pages: BTreeMap<u64, DeviceMapping>,
}

impl DeviceAddressSpace {
    pub fn new(base: u64, size: u64) -> Arc<DeviceAddressSpace> {
        Arc::new(DeviceAddressSpace {
            base,
            size,
            state: Mutex::new(DeviceState {
                attached: BTreeSet::new(),
                pages: BTreeMap::new(),
            }),
        })
    }

    pub fn end(&self) -> u64 {
        self.base + self.size
    }

    /// Whether `at..at+len` fits the device window.
    pub fn contains(&self, at: u64, len: u64) -> bool {
        at >= self.base && at.checked_add(len).is_some_and(|e| e <= self.end())
    }

    /// `svcAttachDeviceAddressSpace`. Recording the name is all the runtime can
    /// do: no device reacts to the attachment.
    pub fn attach(&self, device: u32) -> Result<(), DeviceError> {
        if !self.state.lock().unwrap().attached.insert(device) {
            return Err(DeviceError::State);
        }
        Ok(())
    }

    /// `svcDetachDeviceAddressSpace`.
    pub fn detach(&self, device: u32) -> Result<(), DeviceError> {
        if !self.state.lock().unwrap().attached.remove(&device) {
            return Err(DeviceError::State);
        }
        Ok(())
    }

    pub fn is_attached(&self, device: u32) -> bool {
        self.state.lock().unwrap().attached.contains(&device)
    }

    pub fn attached(&self) -> Vec<u32> {
        self.state
            .lock()
            .unwrap()
            .attached
            .iter()
            .copied()
            .collect()
    }

    /// Maps `size` bytes of the current process at `guest` to `device`. The
    /// guest range must be mapped; the mapping is an alias, not a copy.
    pub fn map(
        &self,
        memory: &GuestMemory,
        guest: u64,
        size: u64,
        device: u64,
        perm: u32,
    ) -> Result<(), DeviceError> {
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Err(DeviceError::Size);
        }
        if !guest.is_multiple_of(PAGE) || !device.is_multiple_of(PAGE) {
            return Err(DeviceError::Address);
        }
        let end = device.checked_add(size).ok_or(DeviceError::Range)?;
        if !self.contains(device, size) {
            return Err(DeviceError::Range);
        }
        if !device_perm(perm) {
            return Err(DeviceError::Permission);
        }
        if !memory.is_mapped(guest, size) {
            return Err(DeviceError::Range);
        }
        let mut st = self.state.lock().unwrap();
        if st.pages.range(..end).any(|(&at, m)| at + m.size > device) {
            return Err(DeviceError::State);
        }
        st.pages.insert(device, DeviceMapping { guest, size, perm });
        Ok(())
    }

    /// `svcUnmapDeviceAddressSpace`: every page of the range must be mapped.
    /// The range may be part of one or more mappings.
    pub fn unmap(&self, device: u64, size: u64) -> Result<(), DeviceError> {
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Err(DeviceError::Size);
        }
        if !device.is_multiple_of(PAGE) {
            return Err(DeviceError::Address);
        }
        let end = device.checked_add(size).ok_or(DeviceError::Range)?;
        if !self.contains(device, size) {
            return Err(DeviceError::Range);
        }
        let mut st = self.state.lock().unwrap();
        let mut at = device;
        while at < end {
            let Some((&key, m)) = st.pages.range(..=at).next_back() else {
                return Err(DeviceError::State);
            };
            if at >= key + m.size {
                return Err(DeviceError::State);
            }
            at = (key + m.size).min(end);
        }
        let touched: Vec<(u64, DeviceMapping)> = st
            .pages
            .range(..end)
            .map(|(&at, &m)| (at, m))
            .filter(|(at, m)| *at + m.size > device)
            .collect();
        for (key, m) in touched {
            st.pages.remove(&key);
            let m_end = key + m.size;
            if key < device {
                st.pages.insert(
                    key,
                    DeviceMapping {
                        size: device - key,
                        ..m
                    },
                );
            }
            if m_end > end {
                st.pages.insert(
                    end,
                    DeviceMapping {
                        guest: m.guest + (end - key),
                        size: m_end - end,
                        perm: m.perm,
                    },
                );
            }
        }
        Ok(())
    }

    /// The guest page behind `device`, with the permission of its mapping.
    fn resolve(&self, device: u64) -> Option<(u64, u32)> {
        let st = self.state.lock().unwrap();
        let (&at, m) = st.pages.range(..=device).next_back()?;
        if device >= at + m.size {
            return None;
        }
        Some((m.guest + (device - at), m.perm))
    }

    /// Reads `len` bytes of guest memory through the device window. `None` when
    /// the range is not mapped or any page forbids reading.
    pub fn read(&self, memory: &GuestMemory, device: u64, len: u64) -> Option<Vec<u8>> {
        let end = device.checked_add(len)?;
        let mut out = Vec::new();
        out.try_reserve_exact(usize::try_from(len).ok()?).ok()?;
        let mut at = device;
        while at < end {
            let (guest, perm) = self.resolve(at)?;
            if perm & perm::R == 0 {
                return None;
            }
            // A page boundary ends the chunk: the next device page may map a
            // different guest page.
            let chunk = (PAGE - (at & (PAGE - 1))).min(end - at);
            out.extend_from_slice(&memory.read(guest, chunk)?);
            at += chunk;
        }
        Some(out)
    }

    /// Writes `bytes` to guest memory through the device window. `false` when
    /// the range is not mapped, a page forbids writing, or the guest page is
    /// gone.
    pub fn write(&self, memory: &GuestMemory, device: u64, bytes: &[u8]) -> bool {
        let Ok(len) = u64::try_from(bytes.len()) else {
            return false;
        };
        let Some(end) = device.checked_add(len) else {
            return false;
        };
        let mut at = device;
        let mut done = 0u64;
        while at < end {
            let Some((guest, perm)) = self.resolve(at) else {
                return false;
            };
            if perm & perm::W == 0 {
                return false;
            }
            let chunk = (PAGE - (at & (PAGE - 1))).min(end - at);
            let (Ok(from), Ok(to)) = (usize::try_from(done), usize::try_from(done + chunk)) else {
                return false;
            };
            if !memory.write(guest, &bytes[from..to]) {
                return false;
            }
            at += chunk;
            done += chunk;
        }
        true
    }

    /// A device-to-device copy: reads `len` bytes from `src` and writes them to
    /// `dst`, both device addresses of this address space.
    pub fn copy(&self, memory: &GuestMemory, dst: u64, src: u64, len: u64) -> bool {
        let Some(bytes) = self.read(memory, src, len) else {
            return false;
        };
        self.write(memory, dst, &bytes)
    }

    /// The mapping that covers `device`, for diagnostics and tests.
    pub fn mapping_at(&self, device: u64) -> Option<DeviceMapping> {
        let st = self.state.lock().unwrap();
        let (&at, m) = st.pages.range(..=device).next_back()?;
        if device >= at + m.size {
            return None;
        }
        Some(*m)
    }

    /// Bytes currently mapped into the device window.
    pub fn mapped_bytes(&self) -> u64 {
        self.state
            .lock()
            .unwrap()
            .pages
            .values()
            .map(|m| m.size)
            .sum()
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

    #[test]
    fn an_io_pool_backs_its_regions_with_real_memory() {
        let m = GuestMemory::new().unwrap();
        let pool = IoPool::pcie_a2();
        let region = IoRegion::new(
            pool.clone(),
            IO_POOL_BASE,
            0x2000,
            io_mapping::MEMORY,
            perm::RW,
        )
        .expect("a page range inside the aperture is a valid region");
        assert_eq!(pool.claimed(), 0x2000);
        assert_eq!(
            IoRegion::new(
                pool.clone(),
                IO_POOL_BASE + 0x1000,
                0x1000,
                io_mapping::MEMORY,
                perm::RW
            )
            .err(),
            Some(DeviceError::State),
            "a claim inside a live one is refused"
        );
        assert_eq!(
            IoRegion::new(
                pool.clone(),
                IO_POOL_END - PAGE,
                PAGE * 2,
                io_mapping::MEMORY,
                perm::RW
            )
            .err(),
            Some(DeviceError::Range),
            "the aperture is the limit"
        );
        assert_eq!(
            IoRegion::new(
                pool.clone(),
                IO_POOL_BASE,
                PAGE,
                io_mapping::IO_REGISTER,
                perm::RW
            )
            .err(),
            Some(DeviceError::Registers),
            "registers are not fabricated"
        );
        assert_eq!(
            IoRegion::new(pool, IO_POOL_BASE, PAGE, io_mapping::MEMORY, perm::RX).err(),
            Some(DeviceError::Permission)
        );

        // Unmapped, the device side reads and writes the shadow directly.
        assert!(region.write(&m, 0, b"pool"));
        assert_eq!(region.read(&m, 0, 4).as_deref(), Some(&b"pool"[..]));
        // Mapped, the guest pages are the source of truth and both sides see
        // the same bytes.
        let at = 0x3_0000_0000;
        region.map(&m, at, 0x2000, perm::RW).expect("maps");
        assert!(m.write(at + 0x1000, b"device!"));
        assert_eq!(
            region.read(&m, 0x1000, 7).as_deref(),
            Some(&b"device!"[..]),
            "the device side reads what the guest wrote"
        );
        assert!(region.write(&m, 0, b"POOL"));
        assert_eq!(&m.read(at, 4).unwrap(), b"POOL");
        region.unmap(&m, at, 0x2000).expect("unmaps");
        assert!(!m.is_mapped(at, 1));
        assert_eq!(
            region.read(&m, 0, 4).as_deref(),
            Some(&b"POOL"[..]),
            "the contents survive the mapping"
        );
        region.map(&m, at + 0x2000, 0x2000, perm::RW).unwrap();
        assert_eq!(&m.read(at + 0x2000, 4).unwrap(), b"POOL");
        assert!(region.release(&m));
        assert!(!m.is_mapped(at + 0x2000, 1));
    }

    #[test]
    fn a_device_address_space_aliases_guest_pages_with_permissions() {
        let m = GuestMemory::new().unwrap();
        let guest = 0x2_0000_0000;
        m.map_as(guest, 0x3000, state::HEAP, perm::RW).unwrap();
        assert!(m.write(guest, b"device"));
        let das = DeviceAddressSpace::new(0, DEVICE_SPACE_LIMIT);

        assert!(das.map(&m, guest, PAGE, 0x1000, perm::RW).is_ok());
        assert_eq!(das.read(&m, 0x1000, 6).as_deref(), Some(&b"device"[..]));
        assert_eq!(das.mapping_at(0x1000).map(|p| p.guest), Some(guest));
        assert!(
            das.write(&m, 0x1000, b"DEVICE"),
            "a device write lands in the guest page"
        );
        assert_eq!(&m.read(guest, 6).unwrap(), b"DEVICE");

        assert!(das.map(&m, guest + PAGE, PAGE, 0x3000, perm::R).is_ok());
        assert!(das.read(&m, 0x3000, 1).is_some());
        assert!(
            !das.write(&m, 0x3000, b"x"),
            "a read-only device mapping refuses writes"
        );
        assert_eq!(
            das.map(&m, guest, PAGE, 0x1000, perm::RW),
            Err(DeviceError::State),
            "device mappings do not overlap"
        );
        assert_eq!(
            das.map(&m, guest + 0x2000, PAGE, 0x1000, perm::RX),
            Err(DeviceError::Permission)
        );
        assert_eq!(
            das.map(&m, guest + 0x2000, PAGE, DEVICE_SPACE_LIMIT, perm::RW),
            Err(DeviceError::Range),
            "outside the device window"
        );
        assert!(das.read(&m, 0x5000, 1).is_none(), "nothing is mapped there");

        assert!(m.write(guest + PAGE, b"copy me"));
        assert!(das.copy(&m, 0x1000, 0x3000, 7));
        assert_eq!(&m.read(guest, 7).unwrap(), b"copy me");

        assert!(das.attach(8).is_ok(), "NV");
        assert_eq!(das.attach(8), Err(DeviceError::State));
        assert_eq!(
            das.detach(29),
            Err(DeviceError::State),
            "SE was not attached"
        );
        assert!(das.detach(8).is_ok());

        assert_eq!(das.unmap(0x2000, PAGE), Err(DeviceError::State));
        assert!(das.unmap(0x1000, PAGE).is_ok());
        assert!(das.read(&m, 0x1000, 1).is_none());
        assert_eq!(
            das.mapped_bytes(),
            PAGE,
            "only the read-only mapping is left"
        );

        // A partial unmap splits the entry and keeps the rest.
        assert!(das.map(&m, guest, PAGE * 2, 0x4000, perm::RW).is_ok());
        assert!(das.unmap(0x5000, PAGE).is_ok());
        assert_eq!(das.mapping_at(0x4000).map(|p| p.size), Some(PAGE));
        assert!(das.mapping_at(0x5000).is_none());
        assert!(das.read(&m, 0x4000, PAGE).is_some());
    }
}
