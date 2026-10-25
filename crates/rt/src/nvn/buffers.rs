//! NVN buffers: a span of a memory pool seen as a GPU address.
//!
//! The buffer keeps the pool and offset the builder named, the guest address its bytes start at (host pointer) and
//! its GPU address. A range that leaves its pool is rejected: a buffer past the end of the storage would otherwise
//! make every later bind read unrelated memory (`nvn_objects.cpp:199-246`).

use super::error::Error;
use super::pools::Pool;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buffer {
    /// Guest address of the `NVNbuffer`'s `NVNmemoryPool`.
    pub pool: u64,
    pub offset: u64,
    pub size: u64,
    /// Guest (= host) address of the first byte.
    pub host: u64,
    /// GPU address of the first byte.
    pub gpu: u64,
}

impl Buffer {
    pub fn create(
        pool_address: u64,
        pool: &Pool,
        offset: u64,
        size: u64,
        host: u64,
    ) -> Result<Buffer, Error> {
        pool.contains(offset, size)?;
        if host == 0 {
            return Err(Error::Unmapped);
        }
        let gpu = pool.gpu.checked_add(offset).ok_or(Error::Overflow)?;
        Ok(Buffer {
            pool: pool_address,
            offset,
            size,
            host,
            gpu,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STORAGE: u64 = 0x8_0000_0000;

    fn pool() -> Pool {
        Pool {
            storage: STORAGE,
            size: 0x10_0000,
            gpu: 0x1_0000_0000,
            ..Pool::default()
        }
    }

    #[test]
    fn buffer_keeps_pool_offset_host_and_gpu() {
        let b = Buffer::create(0x1000, &pool(), 0x2000, 0x4000, STORAGE + 0x2000).expect("buffer");
        assert_eq!(
            (b.pool, b.offset, b.size, b.host, b.gpu),
            (0x1000, 0x2000, 0x4000, STORAGE + 0x2000, 0x1_0000_2000)
        );
    }

    #[test]
    fn buffer_ranges_outside_the_pool_are_rejected() {
        let pool = pool();
        assert_eq!(
            Buffer::create(0x1000, &pool, 0x10_0000, 1, STORAGE),
            Err(Error::OutOfRange)
        );
        assert_eq!(
            Buffer::create(0x1000, &pool, 0xf_ffff, 2, STORAGE),
            Err(Error::OutOfRange)
        );
        assert_eq!(
            Buffer::create(0x1000, &pool, u64::MAX, 2, STORAGE),
            Err(Error::Overflow)
        );
        assert_eq!(
            Buffer::create(0x1000, &pool, 0, 0, STORAGE),
            Err(Error::ZeroSize)
        );
        assert_eq!(
            Buffer::create(0x1000, &pool, 0x100, 0x100, 0),
            Err(Error::Unmapped)
        );
    }

    #[test]
    fn virtual_pool_buffer_uses_the_resolved_host_pointer() {
        use super::super::pools::Mapping;
        let mut pool = Pool {
            size: 0x10_0000,
            gpu: 0x1_0000_0000,
            ..Pool::default()
        };
        assert!(pool.is_virtual());
        pool.map_virtual(
            0x1000,
            Mapping {
                physical: 0x3000,
                physical_offset: 0x800,
                size: 0x1000,
            },
        )
        .expect("mapping");
        assert_eq!(
            Buffer::create(0x2000, &pool, 0x1000, 0x1000, STORAGE + 0x800).map(|b| b.gpu),
            Ok(0x1_0000_1000)
        );
    }
}
