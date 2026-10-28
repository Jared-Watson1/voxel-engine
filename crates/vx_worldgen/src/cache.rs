//! Shared cache of height grid tiles (D21).
//!
//! A tile is the `TILE` x `TILE` block of node heights of one level and face starting at node
//! `(TILE * ti, TILE * tj)`. Node values are a pure function of (seed, level, face, i, j), so the
//! cache changes speed, never results: a lookup returns the same bits whether the tile was just
//! computed, computed by another thread, or evicted and recomputed. Tiles are computed outside the
//! lock; two threads that miss the same tile both compute it and keep the first insertion, which
//! is bit identical to the second.
//!
//! The cache is sharded by a hash of the key, each shard a small set associative array with least
//! recently used eviction. Lookups scan a shard's slots; nothing iterates in an order that could
//! reach a result.

use std::sync::{Arc, Mutex, PoisonError};

use crate::cubemap::CubeFace;
use crate::seed::splitmix64;

/// Nodes per tile edge.
pub(crate) const TILE: i32 = 8;
pub(crate) const TILE_NODES: usize = (TILE * TILE) as usize;

/// Node heights in metres, row major (`j` rows of `i`).
pub(crate) type Tile = [f64; TILE_NODES];

/// Default capacity in tiles: 16,384 tiles of 512 bytes, 8 MiB.
pub(crate) const DEFAULT_CAPACITY_TILES: usize = 16_384;
const SHARDS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileKey {
    pub level: u8,
    pub face: CubeFace,
    pub ti: i32,
    pub tj: i32,
}

impl TileKey {
    fn hash(self) -> u64 {
        let coords = u64::from(self.ti as u32) | (u64::from(self.tj as u32) << 32);
        let tag = (u64::from(self.level) << 8) | u64::from(self.face);
        splitmix64(coords ^ splitmix64(tag))
    }
}

/// Counters since the last [`TileCache::clear`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeightCacheStats {
    pub resident_tiles: usize,
    pub capacity_tiles: usize,
    pub hits: u64,
    pub misses: u64,
    pub resident_bytes: usize,
}

struct Slot {
    key: TileKey,
    tile: Arc<Tile>,
    last_used: u64,
}

#[derive(Default)]
struct Shard {
    slots: Vec<Slot>,
    clock: u64,
    hits: u64,
    misses: u64,
}

pub(crate) struct TileCache {
    shards: Box<[Mutex<Shard>]>,
    ways: usize,
}

impl TileCache {
    pub(crate) fn new(capacity_tiles: usize) -> Self {
        let shards = SHARDS.min(capacity_tiles.max(1));
        let ways = capacity_tiles.max(1).div_ceil(shards);
        Self {
            shards: (0..shards).map(|_| Mutex::new(Shard::default())).collect(),
            ways,
        }
    }

    fn shard(&self, key: TileKey) -> &Mutex<Shard> {
        &self.shards[(key.hash() % self.shards.len() as u64) as usize]
    }

    /// The tile for `key`, from the cache or from `compute` (called without any lock held).
    pub(crate) fn get_or_compute(&self, key: TileKey, compute: impl FnOnce() -> Tile) -> Arc<Tile> {
        let shard = self.shard(key);
        {
            let mut s = shard.lock().unwrap_or_else(PoisonError::into_inner);
            s.clock += 1;
            let now = s.clock;
            if let Some(slot) = s.slots.iter_mut().find(|slot| slot.key == key) {
                slot.last_used = now;
                let tile = Arc::clone(&slot.tile);
                s.hits += 1;
                return tile;
            }
            s.misses += 1;
        }
        let tile = Arc::new(compute());
        let mut s = shard.lock().unwrap_or_else(PoisonError::into_inner);
        s.clock += 1;
        let now = s.clock;
        if let Some(slot) = s.slots.iter_mut().find(|slot| slot.key == key) {
            slot.last_used = now;
            return Arc::clone(&slot.tile);
        }
        let slot = Slot {
            key,
            tile: Arc::clone(&tile),
            last_used: now,
        };
        if s.slots.len() < self.ways {
            s.slots.push(slot);
        } else if let Some(oldest) = s.slots.iter_mut().min_by_key(|slot| slot.last_used) {
            *oldest = slot;
        }
        tile
    }

    pub(crate) fn clear(&self) {
        for shard in self.shards.iter() {
            *shard.lock().unwrap_or_else(PoisonError::into_inner) = Shard::default();
        }
    }

    pub(crate) fn stats(&self) -> HeightCacheStats {
        let mut stats = HeightCacheStats {
            capacity_tiles: self.ways * self.shards.len(),
            ..HeightCacheStats::default()
        };
        for shard in self.shards.iter() {
            let s = shard.lock().unwrap_or_else(PoisonError::into_inner);
            stats.resident_tiles += s.slots.len();
            stats.hits += s.hits;
            stats.misses += s.misses;
        }
        stats.resident_bytes = stats.resident_tiles * std::mem::size_of::<Tile>();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(ti: i32) -> TileKey {
        TileKey {
            level: 3,
            face: 2,
            ti,
            tj: -ti,
        }
    }

    #[test]
    fn caches_evicts_and_recomputes_the_same_values() {
        let cache = TileCache::new(4);
        let compute = |ti: i32| [f64::from(ti); TILE_NODES];
        for round in 0..3 {
            for ti in -20..20 {
                let tile = cache.get_or_compute(key(ti), || compute(ti));
                assert_eq!(*tile, compute(ti), "round {round}");
            }
        }
        let stats = cache.stats();
        assert!(stats.resident_tiles <= stats.capacity_tiles);
        assert!(stats.misses >= 40);
        let hit = cache.get_or_compute(key(19), || unreachable!("resident"));
        assert_eq!(*hit, compute(19));
        cache.clear();
        assert_eq!(cache.stats().resident_tiles, 0);
    }
}
