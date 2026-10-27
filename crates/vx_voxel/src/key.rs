use glam::{DVec3, IVec3};
use vx_core::{CHUNK_SIZE, WorldConfig, WorldPos};

/// Coordinate of a chunk in the chunk grid: the chunk covers world sample coordinates
/// `32 * key` to `32 * key + 32` along each axis, and its stored block (D16) runs one sample further
/// on each side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkKey {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub const fn as_ivec3(self) -> IVec3 {
        IVec3::new(self.x, self.y, self.z)
    }

    /// Position of local sample (0, 0, 0) in the body fixed frame.
    pub fn origin(self, config: &WorldConfig) -> WorldPos {
        let extent = config.chunk_extent_m();
        WorldPos::body_fixed(DVec3::new(
            f64::from(self.x) * extent,
            f64::from(self.y) * extent,
            f64::from(self.z) * extent,
        ))
    }

    /// The chunk whose cell region contains `position` (floor division by the chunk extent).
    pub fn containing(position: WorldPos, config: &WorldConfig) -> Self {
        let extent = config.chunk_extent_m();
        let f = |v: f64| (v / extent).floor() as i32;
        Self::new(f(position.m.x), f(position.m.y), f(position.m.z))
    }

    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self::new(self.x + dx, self.y + dy, self.z + dz)
    }

    /// Squared distance between two keys in chunks, for streaming priorities.
    pub fn distance_sq(self, other: Self) -> u64 {
        let d = |a: i32, b: i32| {
            let v = i64::from(a) - i64::from(b);
            (v * v) as u64
        };
        d(self.x, other.x) + d(self.y, other.y) + d(self.z, other.z)
    }

    /// World sample coordinate (in voxels) of this chunk's local sample, where each local
    /// coordinate runs from -1 to `CHUNK_SIZE + 1` inclusive.
    pub fn world_sample(self, local: IVec3) -> IVec3 {
        self.as_ivec3() * CHUNK_SIZE as i32 + local
    }

    /// Local coordinate of a world sample in this chunk (may lie outside the stored block).
    pub fn local_sample(self, world: IVec3) -> IVec3 {
        world - self.as_ivec3() * CHUNK_SIZE as i32
    }

    /// Every chunk whose stored block (apron included) holds world sample `sample`: one or two
    /// per axis, so one to eight keys (D16).
    pub fn chunks_holding_sample(sample: IVec3) -> Vec<Self> {
        Self::chunks_intersecting_samples(sample, sample)
    }

    /// Every chunk whose stored block intersects the inclusive world sample box `min` to `max`.
    /// A brush's dirty set is this box for its padded bounds.
    pub fn chunks_intersecting_samples(min: IVec3, max: IVec3) -> Vec<Self> {
        debug_assert!(
            min.cmple(max).all(),
            "sample box min {min} exceeds max {max}"
        );
        let (x0, x1) = Self::chunk_range_along_axis(min.x, max.x);
        let (y0, y1) = Self::chunk_range_along_axis(min.y, max.y);
        let (z0, z1) = Self::chunk_range_along_axis(min.z, max.z);
        let count = |a: i32, b: i32| usize::try_from(i64::from(b) - i64::from(a) + 1).unwrap_or(0);
        let mut keys = Vec::with_capacity(count(x0, x1) * count(y0, y1) * count(z0, z1));
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    keys.push(Self::new(x, y, z));
                }
            }
        }
        keys
    }

    /// Inclusive range of chunk indices along one axis whose stored block (`32K - 1` to
    /// `32K + 33`, D16) intersects the inclusive sample range `min` to `max`. Euclidean division
    /// keeps the rounding direction correct for negative coordinates.
    fn chunk_range_along_axis(min: i32, max: i32) -> (i32, i32) {
        let size = CHUNK_SIZE as i32;
        let apron_end = size + 1;
        // ceil((min - apron_end) / size) and floor((max + 1) / size).
        let first = (min - apron_end + size - 1).div_euclid(size);
        let last = (max + 1).div_euclid(size);
        (first, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chunks along x holding `sample`; y and z sit deep inside chunk 0 so only x varies.
    fn axis_chunks(sample: i32) -> Vec<i32> {
        let mid = CHUNK_SIZE as i32 / 2;
        let probe = IVec3::new(sample, mid, mid);
        ChunkKey::chunks_intersecting_samples(probe, probe)
            .into_iter()
            .map(|k| k.x)
            .collect()
    }

    #[test]
    fn one_sample_lies_in_one_or_two_chunks_per_axis() {
        // Worked examples from D16, along x, in positive and negative coordinates.
        let cases: [(i32, &[i32]); 17] = [
            (0, &[-1, 0]),
            (1, &[-1, 0]),
            (2, &[0]),
            (30, &[0]),
            (31, &[0, 1]),
            (32, &[0, 1]),
            (33, &[0, 1]),
            (34, &[1]),
            (-1, &[-1, 0]),
            (-2, &[-1]),
            (-30, &[-1]),
            (-31, &[-2, -1]),
            (-32, &[-2, -1]),
            (-33, &[-2, -1]),
            (-34, &[-2]),
            (64, &[1, 2]),
            (-64, &[-3, -2]),
        ];
        for (sample, expected) in cases {
            assert_eq!(axis_chunks(sample), expected, "sample {sample}");
        }
    }

    #[test]
    fn every_chunk_returned_stores_the_sample_and_no_other_does() {
        let size = CHUNK_SIZE as i32;
        for sample in -100..100 {
            let found = axis_chunks(sample);
            for k in -5..5 {
                let holds = (size * k - 1..=size * k + size + 1).contains(&sample);
                assert_eq!(found.contains(&k), holds, "sample {sample} chunk {k}");
            }
        }
    }

    #[test]
    fn box_spanning_two_chunks_along_x_only() {
        let keys =
            ChunkKey::chunks_intersecting_samples(IVec3::new(20, 5, 5), IVec3::new(40, 10, 10));
        assert_eq!(keys, vec![ChunkKey::new(0, 0, 0), ChunkKey::new(1, 0, 0)]);
    }

    #[test]
    fn keys_come_out_x_fastest_then_y_then_z() {
        let keys = ChunkKey::chunks_intersecting_samples(IVec3::splat(31), IVec3::splat(32));
        assert_eq!(keys.len(), 8);
        assert_eq!(keys[0], ChunkKey::new(0, 0, 0));
        assert_eq!(keys[1], ChunkKey::new(1, 0, 0));
        assert_eq!(keys[2], ChunkKey::new(0, 1, 0));
        assert_eq!(keys[4], ChunkKey::new(0, 0, 1));
        assert_eq!(keys[7], ChunkKey::new(1, 1, 1));
        let mut sorted = keys.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "no duplicate keys");
    }

    #[test]
    fn holding_sample_counts_one_two_four_or_eight() {
        assert_eq!(
            ChunkKey::chunks_holding_sample(IVec3::new(16, 16, 16)).len(),
            1
        );
        assert_eq!(
            ChunkKey::chunks_holding_sample(IVec3::new(0, 16, 16)).len(),
            2
        );
        assert_eq!(
            ChunkKey::chunks_holding_sample(IVec3::new(0, 0, 16)).len(),
            4
        );
        assert_eq!(
            ChunkKey::chunks_holding_sample(IVec3::new(0, 0, 0)).len(),
            8
        );
        assert_eq!(
            ChunkKey::chunks_holding_sample(IVec3::new(-32, 33, 1)).len(),
            8
        );
    }
}
