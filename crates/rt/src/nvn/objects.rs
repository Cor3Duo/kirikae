//! Host objects keyed by their guest address.
//!
//! Every NVN object is guest memory the game allocated at the real driver's size; the host keeps what it decoded
//! from the builder next to the guest address. `make` publishes an object and `destroy` retires it, and neither
//! frees immediately: an old incarnation stays resolvable at a mark taken before the change until `drain`, because a
//! submit may still name it (`nvn_incarnation.cpp:1-14`, `docs/nvn/CONTRACT.md` §7).
//!
//! Lookups without a mark always see the current object. A lookup with mark `m` sees the object the first change
//! after `m` retired, or the current one when the address was empty at `m` and got its object later.

use std::collections::HashMap;
use std::sync::Arc;

use super::buffers::Buffer;
use super::error::Error;
use super::pools::Pool;
use super::samplers::Sampler;
use super::textures::{Texture, TextureView};

#[derive(Clone, Debug, PartialEq)]
pub enum Object {
    Pool(Pool),
    Buffer(Buffer),
    Texture(Texture),
    View(TextureView),
    Sampler(Sampler),
}

impl Object {
    pub fn as_pool(&self) -> Option<&Pool> {
        match self {
            Object::Pool(pool) => Some(pool),
            _ => None,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Object::Pool(_) => "pool",
            Object::Buffer(_) => "buffer",
            Object::Texture(_) => "texture",
            Object::View(_) => "view",
            Object::Sampler(_) => "sampler",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub made: u64,
    pub replaced: u64,
    pub destroyed: u64,
    pub drained: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub pools: u64,
    pub buffers: u64,
    pub textures: u64,
    pub views: u64,
    pub samplers: u64,
}

impl Counts {
    pub fn total(&self) -> u64 {
        self.pools + self.buffers + self.textures + self.views + self.samplers
    }
}

struct Entry {
    object: Arc<Object>,
}

struct Retired {
    epoch: u64,
    object: Option<Arc<Object>>,
}

#[derive(Default)]
pub struct Registry {
    live: HashMap<u64, Entry>,
    retired: HashMap<u64, Vec<Retired>>,
    deferred: Vec<Arc<Object>>,
    epoch: u64,
    stats: Stats,
}

impl Registry {
    /// Publishes `object` at `address` and returns the epoch of the change. Whatever lived there becomes an
    /// incarnation that a mark taken before this call still sees.
    pub fn make(&mut self, address: u64, object: Object) -> u64 {
        self.epoch += 1;
        let epoch = self.epoch;
        match self.live.insert(
            address,
            Entry {
                object: Arc::new(object),
            },
        ) {
            Some(old) => {
                self.retired.entry(address).or_default().push(Retired {
                    epoch,
                    object: Some(old.object.clone()),
                });
                self.deferred.push(old.object);
                self.stats.replaced += 1;
            }
            None => self.retired.entry(address).or_default().push(Retired {
                epoch,
                object: None,
            }),
        }
        self.stats.made += 1;
        epoch
    }

    /// Retires the object at `address`; it stays resolvable at older marks until `drain`.
    pub fn destroy(&mut self, address: u64) -> bool {
        let Some(old) = self.live.remove(&address) else {
            return false;
        };
        self.epoch += 1;
        let epoch = self.epoch;
        self.retired.entry(address).or_default().push(Retired {
            epoch,
            object: Some(old.object.clone()),
        });
        self.deferred.push(old.object);
        self.stats.destroyed += 1;
        true
    }

    fn find_at(&self, address: u64, mark: u64) -> Option<&Object> {
        let current = self.live.get(&address);
        if mark != 0
            && let Some(records) = self.retired.get(&address)
        {
            for record in records {
                if record.epoch > mark {
                    return match &record.object {
                        Some(object) => Some(object.as_ref()),
                        None => current.map(|entry| entry.object.as_ref()),
                    };
                }
            }
        }
        current.map(|entry| entry.object.as_ref())
    }

    pub fn get(&self, address: u64) -> Option<Object> {
        self.live
            .get(&address)
            .map(|entry| entry.object.as_ref().clone())
    }

    pub fn get_at(&self, address: u64, mark: u64) -> Option<Object> {
        self.find_at(address, mark).cloned()
    }

    pub fn with<R>(&self, address: u64, f: impl FnOnce(&Object) -> R) -> Option<R> {
        self.live.get(&address).map(|entry| f(&entry.object))
    }

    pub fn with_at<R>(&self, address: u64, mark: u64, f: impl FnOnce(&Object) -> R) -> Option<R> {
        self.find_at(address, mark).map(f)
    }

    pub fn contains(&self, address: u64) -> bool {
        self.live.contains_key(&address)
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// Epoch of the last change; a replay takes this as its mark.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn counts(&self) -> Counts {
        let mut counts = Counts::default();
        for entry in self.live.values() {
            match entry.object.as_ref() {
                Object::Pool(_) => counts.pools += 1,
                Object::Buffer(_) => counts.buffers += 1,
                Object::Texture(_) => counts.textures += 1,
                Object::View(_) => counts.views += 1,
                Object::Sampler(_) => counts.samplers += 1,
            }
        }
        counts
    }

    pub fn deferred(&self) -> usize {
        self.deferred.len()
    }

    /// Frees retired objects and forgets their incarnation records. Only safe when no replay can name them.
    pub fn drain(&mut self) -> usize {
        let drained = self.deferred.len();
        self.deferred.clear();
        self.retired.clear();
        self.stats.drained += drained as u64;
        drained
    }

    pub fn pool(&self, address: u64) -> Option<Pool> {
        self.with(address, |object| object.as_pool().cloned())
            .flatten()
    }

    /// Guest (= host) address of `[offset, offset + size)` in the pool at `address`, through the pool's virtual
    /// mappings when it has no storage of its own.
    pub fn pool_host(&self, address: u64, offset: u64, size: u64) -> Result<u64, Error> {
        let pool = self.pool(address).ok_or(Error::UnknownPool)?;
        if !pool.is_virtual() {
            pool.contains(offset, size)?;
            return pool.storage.checked_add(offset).ok_or(Error::Overflow);
        }
        let (start, mapping) = pool.resolve(offset, size)?;
        let physical = self.pool(mapping.physical).ok_or(Error::UnknownPool)?;
        if physical.is_virtual() {
            return Err(Error::Unmapped);
        }
        let offset = mapping
            .physical_offset
            .checked_add(offset.checked_sub(start).ok_or(Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        physical.contains(offset, size)?;
        physical.storage.checked_add(offset).ok_or(Error::Overflow)
    }
}

#[cfg(test)]
mod tests {
    use super::super::pools::Mapping;
    use super::*;

    fn pool(size: u64) -> Object {
        Object::Pool(Pool {
            size,
            ..Pool::default()
        })
    }

    fn size_of(object: &Object) -> u64 {
        match object {
            Object::Pool(pool) => pool.size,
            _ => 0,
        }
    }

    #[test]
    fn make_and_destroy_retire_incarnations() {
        let mut registry = Registry::default();
        assert!(registry.is_empty());

        let first = registry.make(0x1000, pool(0x1000));
        assert_eq!(first, 1);
        assert_eq!(registry.len(), 1);
        let second = registry.make(0x1000, pool(0x2000));
        assert_eq!(second, 2);
        assert_eq!(registry.len(), 1);

        assert_eq!(size_of(&registry.get(0x1000).expect("current")), 0x2000);
        assert_eq!(
            size_of(&registry.get_at(0x1000, first).expect("first incarnation")),
            0x1000
        );
        assert_eq!(
            size_of(&registry.get_at(0x1000, 0).expect("mark 0 is current")),
            0x2000
        );
        assert_eq!(
            size_of(
                &registry
                    .get_at(0x1000, second)
                    .expect("current at its epoch")
            ),
            0x2000
        );

        assert!(registry.destroy(0x1000));
        assert!(!registry.destroy(0x1000));
        assert!(registry.get(0x1000).is_none());
        assert_eq!(
            size_of(&registry.get_at(0x1000, second).expect("retired object")),
            0x2000
        );
        assert!(registry.get_at(0x1000, registry.epoch()).is_none());

        let stats = registry.stats();
        assert_eq!((stats.made, stats.replaced, stats.destroyed), (2, 1, 1));
        assert_eq!(registry.deferred(), 2);
        assert_eq!(registry.drain(), 2);
        assert_eq!(registry.stats().drained, 2);
        assert!(registry.get_at(0x1000, first).is_none());
    }

    #[test]
    fn object_created_after_a_mark_resolves_to_the_current_one() {
        let mut registry = Registry::default();
        let mark = registry.epoch();
        registry.make(0x2000, pool(0x1000));
        assert_eq!(
            size_of(
                &registry
                    .get_at(0x2000, mark)
                    .expect("created after the mark")
            ),
            0x1000
        );
        assert_eq!(registry.counts().pools, 1);
        assert_eq!(registry.counts().total(), 1);
    }

    #[test]
    fn virtual_pool_host_resolution_walks_mappings() {
        const STORAGE: u64 = 0x8_0000_0000;
        let mut registry = Registry::default();
        registry.make(
            0x1000,
            Object::Pool(Pool {
                storage: STORAGE,
                size: 0x10_0000,
                gpu: 0x1_0000_0000,
                ..Pool::default()
            }),
        );
        let mut virtual_pool = Pool {
            size: 0x10_0000,
            flags: super::super::pools::VIRTUAL,
            ..Pool::default()
        };
        virtual_pool
            .map_virtual(
                0x1000,
                Mapping {
                    physical: 0x1000,
                    physical_offset: 0x2000,
                    size: 0x1000,
                },
            )
            .expect("mapping");
        registry.make(0x2000, Object::Pool(virtual_pool));

        assert_eq!(
            registry.pool_host(0x1000, 0x3000, 0x100),
            Ok(STORAGE + 0x3000)
        );
        assert_eq!(
            registry.pool_host(0x2000, 0x1000, 0x1000),
            Ok(STORAGE + 0x2000)
        );
        assert_eq!(
            registry.pool_host(0x2000, 0x1200, 0x100),
            Ok(STORAGE + 0x2200)
        );
        assert_eq!(
            registry.pool_host(0x2000, 0x2000, 0x10),
            Err(Error::Unmapped)
        );
        assert_eq!(
            registry.pool_host(0x2000, 0x1f00, 0x200),
            Err(Error::Unmapped)
        );
        assert_eq!(registry.pool_host(0x2000, 0x0, 0x10), Err(Error::Unmapped));
        assert_eq!(
            registry.pool_host(0x3000, 0x0, 0x10),
            Err(Error::UnknownPool)
        );
        assert_eq!(
            registry.pool_host(0x1000, 0x10_0000, 0x10),
            Err(Error::OutOfRange)
        );
    }
}
