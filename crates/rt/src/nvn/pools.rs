//! NVN memory pools: storage, GPU address, and virtual mappings.
//!
//! A pool owns either a span of guest memory (a physical pool) or nothing (a virtual pool). Virtual pools are
//! filled by `nvnMemoryPoolMapVirtual`, which maps `[virtualOffset, +size)` onto `(physical pool, physicalOffset)`
//! in 0x28-byte requests (`nvn_objects.cpp:84-93, 142-167`). Host resolution walks those mappings; a hole, an
//! overlap or a range past the end of either pool is rejected instead of guessed.

use std::collections::BTreeMap;

use super::error::Error;

/// `NVN_MEMORY_POOL_FLAGS_VIRTUAL` in the public NVN enum.
pub const VIRTUAL: u32 = 0x200;

/// One `nvnMemoryPoolMapVirtual` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping {
    pub physical: u64,
    pub physical_offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pool {
    /// Guest address of the storage the game gave; 0 for a virtual pool.
    pub storage: u64,
    pub size: u64,
    pub flags: u32,
    /// GPU virtual address of byte 0.
    pub gpu: u64,
    /// Virtual offset -> physical range. Empty for a physical pool.
    pub mappings: BTreeMap<u64, Mapping>,
}

impl Pool {
    pub fn is_virtual(&self) -> bool {
        self.storage == 0
    }

    /// Checks that `[offset, offset + size)` is inside the pool.
    pub fn contains(&self, offset: u64, size: u64) -> Result<(), Error> {
        if size == 0 {
            return Err(Error::ZeroSize);
        }
        let end = offset.checked_add(size).ok_or(Error::Overflow)?;
        if end > self.size {
            return Err(Error::OutOfRange);
        }
        Ok(())
    }

    /// Adds or replaces the mapping at `offset`.
    pub fn map_virtual(&mut self, offset: u64, mapping: Mapping) -> Result<(), Error> {
        if !self.is_virtual() {
            return Err(Error::NotVirtual);
        }
        if mapping.physical == 0 {
            return Err(Error::UnknownPool);
        }
        self.contains(offset, mapping.size)?;
        let end = offset.checked_add(mapping.size).ok_or(Error::Overflow)?;
        if let Some((&before, previous)) = self.mappings.range(..offset).next_back()
            && before
                .checked_add(previous.size)
                .is_none_or(|before_end| before_end > offset)
        {
            return Err(Error::Overlap);
        }
        if let Some((&after, _)) = self.mappings.range(offset.saturating_add(1)..).next()
            && after < end
        {
            return Err(Error::Overlap);
        }
        self.mappings.insert(offset, mapping);
        Ok(())
    }

    pub fn unmap_virtual(&mut self, offset: u64) -> bool {
        self.mappings.remove(&offset).is_some()
    }

    /// The mapping covering `[offset, offset + size)`, with its virtual start.
    pub fn resolve(&self, offset: u64, size: u64) -> Result<(u64, Mapping), Error> {
        self.contains(offset, size)?;
        let (&start, mapping) = self
            .mappings
            .range(..=offset)
            .next_back()
            .ok_or(Error::Unmapped)?;
        let end = offset.checked_add(size).ok_or(Error::Overflow)?;
        let mapped_end = start.checked_add(mapping.size).ok_or(Error::Overflow)?;
        if end > mapped_end {
            return Err(Error::Unmapped);
        }
        Ok((start, *mapping))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn virtual_pool(size: u64) -> Pool {
        Pool {
            size,
            flags: VIRTUAL,
            ..Pool::default()
        }
    }

    fn physical_pool(storage: u64, size: u64) -> Pool {
        Pool {
            storage,
            size,
            ..Pool::default()
        }
    }

    const PHYSICAL: u64 = 0x1_0000_0000;
    const STORAGE: u64 = 0x8_0000_0000;

    #[test]
    fn physical_pool_keeps_its_span() {
        let pool = physical_pool(STORAGE, 0x10_0000);
        assert!(!pool.is_virtual());
        assert_eq!(pool.contains(0, 0x10_0000), Ok(()));
        assert_eq!(pool.contains(0, 0x10_0001), Err(Error::OutOfRange));
        assert_eq!(pool.contains(0x10_0000, 0), Err(Error::ZeroSize));
        assert_eq!(pool.contains(u64::MAX, 2), Err(Error::Overflow));
        assert_eq!(pool.resolve(0, 0x1000), Err(Error::Unmapped));
    }

    #[test]
    fn virtual_mappings_resolve_by_offset() {
        let mut pool = virtual_pool(0x20_0000);
        assert!(pool.is_virtual());
        pool.map_virtual(
            0x1000,
            Mapping {
                physical: PHYSICAL,
                physical_offset: 0x2000,
                size: 0x4000,
            },
        )
        .expect("mapping");
        assert_eq!(
            pool.resolve(0x1000, 0x4000),
            Ok((
                0x1000,
                Mapping {
                    physical: PHYSICAL,
                    physical_offset: 0x2000,
                    size: 0x4000
                }
            ))
        );
        assert_eq!(
            pool.resolve(0x2fff, 0x10).map(|(_, m)| m.physical),
            Ok(PHYSICAL)
        );
        assert_eq!(pool.resolve(0x2000, 0x4000), Err(Error::Unmapped));
        assert_eq!(pool.resolve(0, 0x1000), Err(Error::Unmapped));
        assert_eq!(pool.resolve(0x4fff, 0x1000), Err(Error::Unmapped));
    }

    #[test]
    fn mappings_reject_holes_overlaps_and_overflow() {
        let mut pool = virtual_pool(0x20_0000);
        assert_eq!(
            pool.map_virtual(
                0x30_0000,
                Mapping {
                    physical: PHYSICAL,
                    physical_offset: 0,
                    size: 0x1000
                }
            ),
            Err(Error::OutOfRange)
        );
        assert_eq!(
            pool.map_virtual(
                u64::MAX,
                Mapping {
                    physical: PHYSICAL,
                    physical_offset: 0,
                    size: 2
                }
            ),
            Err(Error::Overflow)
        );
        assert_eq!(
            pool.map_virtual(
                0,
                Mapping {
                    physical: 0,
                    physical_offset: 0,
                    size: 0x1000
                }
            ),
            Err(Error::UnknownPool)
        );
        pool.map_virtual(
            0x1000,
            Mapping {
                physical: PHYSICAL,
                physical_offset: 0,
                size: 0x2000,
            },
        )
        .expect("first mapping");
        assert_eq!(
            pool.map_virtual(
                0x2000,
                Mapping {
                    physical: PHYSICAL,
                    physical_offset: 0,
                    size: 0x2000
                }
            ),
            Err(Error::Overlap)
        );
        assert_eq!(
            pool.map_virtual(
                0x0800,
                Mapping {
                    physical: PHYSICAL,
                    physical_offset: 0,
                    size: 0x1000
                }
            ),
            Err(Error::Overlap)
        );
        pool.map_virtual(
            0x1000,
            Mapping {
                physical: PHYSICAL,
                physical_offset: 0x1000,
                size: 0x1000,
            },
        )
        .expect("replace");
        assert_eq!(pool.mappings.len(), 1);
        assert!(pool.unmap_virtual(0x1000));
        assert!(!pool.unmap_virtual(0x1000));

        let mut physical = physical_pool(STORAGE, 0x10_0000);
        assert_eq!(
            physical.map_virtual(
                0,
                Mapping {
                    physical: PHYSICAL,
                    physical_offset: 0,
                    size: 0x1000
                }
            ),
            Err(Error::NotVirtual)
        );
    }
}
