//! GPU virtual addresses: allocation and the pool ranges they name.
//!
//! NVN GPU addresses are 40-bit and 64 KiB aligned, and address 0 and low values stay invalid
//! (`nvn_objects.cpp:25-34`). Every memory pool gets one range; a range is never reused while it is registered, and
//! ranges are separated by a guard so an off-by-one read cannot land in the next pool. Lookup maps an address back
//! to the pool that owns it and the byte offset inside it.

use std::collections::BTreeMap;

use super::error::Error;

pub const BASE: u64 = 0x1_0000_0000;
pub const LIMIT: u64 = 1 << 40;
pub const ALIGN: u64 = 0x1_0000;
/// Empty space between two allocations. A pool's last byte plus this is where the next range starts.
pub const GUARD: u64 = ALIGN;

/// Monotonic allocator; a released range is not handed out again.
#[derive(Debug)]
pub struct Allocator {
    next: u64,
}

impl Default for Allocator {
    fn default() -> Self {
        Allocator { next: BASE }
    }
}

impl Allocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves `size` bytes and returns the 64 KiB-aligned start. `size` is rounded up so the next range starts
    /// after the requested bytes plus a guard gap.
    pub fn allocate(&mut self, size: u64) -> Result<u64, Error> {
        if size == 0 {
            return Err(Error::ZeroSize);
        }
        let size = size
            .div_ceil(ALIGN)
            .checked_mul(ALIGN)
            .ok_or(Error::Overflow)?;
        let end = self
            .next
            .checked_add(size)
            .and_then(|end| end.checked_add(GUARD))
            .ok_or(Error::Overflow)?;
        if end > LIMIT {
            return Err(Error::Overflow);
        }
        let start = self.next;
        self.next = end;
        Ok(start)
    }

    /// First address the next allocation can use, guard gap included.
    pub fn next(&self) -> u64 {
        self.next
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub pool: u64,
    pub size: u64,
}

/// GPU address range to guest pool. Ranges must stay disjoint: the real driver never aliases two pools.
#[derive(Debug, Default)]
pub struct Ranges {
    live: BTreeMap<u64, Range>,
}

impl Ranges {
    pub fn register(&mut self, start: u64, size: u64, pool: u64) -> Result<(), Error> {
        if size == 0 {
            return Err(Error::ZeroSize);
        }
        let end = start.checked_add(size).ok_or(Error::Overflow)?;
        if start < BASE || !start.is_multiple_of(ALIGN) || end > LIMIT {
            return Err(Error::OutOfRange);
        }
        if let Some((&before, range)) = self.live.range(..=start).next_back()
            && before
                .checked_add(range.size)
                .is_none_or(|before_end| before_end > start)
        {
            return Err(Error::Overlap);
        }
        if let Some((&after, _)) = self.live.range(start..).next()
            && after < end
        {
            return Err(Error::Overlap);
        }
        self.live.insert(start, Range { pool, size });
        Ok(())
    }

    pub fn unregister(&mut self, start: u64) -> Option<Range> {
        self.live.remove(&start)
    }

    pub fn get(&self, start: u64) -> Option<Range> {
        self.live.get(&start).copied()
    }

    /// `(pool, offset)` when `[gpu, gpu + len)` lies inside one range.
    pub fn lookup(&self, gpu: u64, len: u64) -> Option<(u64, u64)> {
        let end = gpu.checked_add(len)?;
        let (&start, range) = self.live.range(..=gpu).next_back()?;
        if end > start.checked_add(range.size)? {
            return None;
        }
        Some((range.pool, gpu - start))
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_are_aligned_and_disjoint() {
        let mut va = Allocator::new();
        let a = va.allocate(0x1000).expect("first allocation");
        let b = va.allocate(0x1_0000).expect("second allocation");
        assert_eq!(a, BASE);
        assert_eq!(a % ALIGN, 0);
        assert_eq!(b % ALIGN, 0);
        assert_eq!(b, a + ALIGN + GUARD);
        assert!(a + ALIGN <= b);
    }

    #[test]
    fn overflow_stops_at_the_40_bit_limit() {
        let mut va = Allocator {
            next: LIMIT - ALIGN,
        };
        assert_eq!(va.allocate(ALIGN), Err(Error::Overflow));
        let mut va = Allocator {
            next: LIMIT - ALIGN - GUARD,
        };
        assert_eq!(
            va.allocate(ALIGN).expect("last range"),
            LIMIT - ALIGN - GUARD
        );
        assert_eq!(va.allocate(ALIGN), Err(Error::Overflow));
        assert_eq!(Allocator::new().allocate(0), Err(Error::ZeroSize));
    }

    #[test]
    fn ranges_map_back_to_their_pool() {
        let mut ranges = Ranges::default();
        ranges
            .register(0x1_0000_0000, 0x2_0000, 0x1000)
            .expect("first range");
        ranges
            .register(0x1_0004_0000, 0x1_0000, 0x2000)
            .expect("second range");
        assert_eq!(ranges.lookup(0x1_0000_0000, 0x10), Some((0x1000, 0)));
        assert_eq!(ranges.lookup(0x1_0001_2340, 0x10), Some((0x1000, 0x12340)));
        assert_eq!(ranges.lookup(0x1_0004_8000, 0x8000), Some((0x2000, 0x8000)));
        assert_eq!(ranges.lookup(0x1_0004_8000, 0x8001), None);
        assert_eq!(ranges.lookup(0x1_0003_0000, 0x10), None);
        assert_eq!(ranges.lookup(0x1_0000_0000, u64::MAX), None);
        assert_eq!(ranges.lookup(u64::MAX, 1), None);
        assert_eq!(ranges.len(), 2);
    }

    #[test]
    fn overlapping_ranges_are_rejected() {
        let mut ranges = Ranges::default();
        ranges
            .register(BASE, 0x2_0000, 0x1000)
            .expect("first range");
        assert_eq!(ranges.register(BASE, 0x1_0000, 0x2000), Err(Error::Overlap));
        assert_eq!(
            ranges.register(BASE + 0x1_0000, 0x1_0000, 0x2000),
            Err(Error::Overlap)
        );
        assert_eq!(
            ranges.register(BASE - ALIGN, 0x1_0000, 0x2000),
            Err(Error::OutOfRange)
        );
        assert_eq!(
            ranges
                .register(BASE + 0x4_0000, 0x1_0000, 0x2000)
                .expect("free range"),
            ()
        );
        assert_eq!(ranges.unregister(BASE).map(|r| r.pool), Some(0x1000));
        assert!(ranges.unregister(BASE).is_none());
    }
}
