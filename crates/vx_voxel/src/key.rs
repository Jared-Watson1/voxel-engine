use glam::{DVec3, IVec3};
use vx_core::{CHUNK_SIZE, WorldConfig, WorldPos};

/// Coordinate of a chunk in the octree (D20): at level `level` the chunk covers world sample
/// coordinates `32 * key` to `32 * key + 32` along each axis, in units of the level's voxel size
/// (`voxel_size_m * 2^level`), and its stored block (D16) runs one sample further on each side.
/// Level 0 is the finest. The derived order sorts by level, then x, y and z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkKey {
    pub level: u8,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// One of the six faces of a chunk. The discriminant is the bit index in a transition mask
/// (`1 << face as u8`): the bit is set when the neighbour across the face is drawn one level finer
/// (D22).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Face {
    LowX = 0,
    HighX = 1,
    LowY = 2,
    HighY = 3,
    LowZ = 4,
    HighZ = 5,
}

impl Face {
    pub const ALL: [Self; 6] = [
        Self::LowX,
        Self::HighX,
        Self::LowY,
        Self::HighY,
        Self::LowZ,
        Self::HighZ,
    ];

    /// 0 for x, 1 for y, 2 for z.
    pub const fn axis(self) -> usize {
        self as usize / 2
    }

    pub const fn is_high(self) -> bool {
        self as usize % 2 == 1
    }

    pub const fn bit(self) -> u8 {
        1 << self as u8
    }

    pub const fn opposite(self) -> Self {
        Self::ALL[self as usize ^ 1]
    }

    /// Unit step across the face in chunk coordinates.
    pub fn step(self) -> IVec3 {
        let mut v = IVec3::ZERO;
        v[self.axis()] = if self.is_high() { 1 } else { -1 };
        v
    }
}

impl ChunkKey {
    /// A level 0 key.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { level: 0, x, y, z }
    }

    pub const fn at_level(level: u8, x: i32, y: i32, z: i32) -> Self {
        Self { level, x, y, z }
    }

    pub const fn as_ivec3(self) -> IVec3 {
        IVec3::new(self.x, self.y, self.z)
    }

    fn with_coords(self, c: IVec3) -> Self {
        Self::at_level(self.level, c.x, c.y, c.z)
    }

    /// Sample spacing of this key's level in metres: `base_voxel_size_m * 2^level`, exact in f64.
    pub fn voxel_size_m(self, base_voxel_size_m: f64) -> f64 {
        base_voxel_size_m * f64::from(1u32 << self.level)
    }

    /// Edge length of this key's chunk in metres.
    pub fn extent_m(self, base_voxel_size_m: f64) -> f64 {
        CHUNK_SIZE as f64 * self.voxel_size_m(base_voxel_size_m)
    }

    /// Position of local sample (0, 0, 0) in the body fixed frame.
    pub fn origin(self, config: &WorldConfig) -> WorldPos {
        WorldPos::body_fixed(self.origin_m(config.voxel_size_m))
    }

    /// Position of local sample (0, 0, 0) in metres, from the integer sample index times the
    /// level's voxel size (one rounding, as `Chunk::sample_position`).
    pub fn origin_m(self, base_voxel_size_m: f64) -> DVec3 {
        (self.as_ivec3() * CHUNK_SIZE as i32).as_dvec3() * self.voxel_size_m(base_voxel_size_m)
    }

    /// The cell region (apron excluded) as minimum and maximum corners in metres.
    pub fn bounds_m(self, base_voxel_size_m: f64) -> (DVec3, DVec3) {
        let min = self.origin_m(base_voxel_size_m);
        (min, min + DVec3::splat(self.extent_m(base_voxel_size_m)))
    }

    /// The level 0 chunk whose cell region contains `position` (floor division by the extent).
    pub fn containing(position: WorldPos, config: &WorldConfig) -> Self {
        Self::containing_at(position, 0, config)
    }

    /// The chunk of `level` whose cell region contains `position`.
    pub fn containing_at(position: WorldPos, level: u8, config: &WorldConfig) -> Self {
        let extent = Self::at_level(level, 0, 0, 0).extent_m(config.voxel_size_m);
        let f = |v: f64| (v / extent).floor() as i32;
        Self::at_level(level, f(position.m.x), f(position.m.y), f(position.m.z))
    }

    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self::at_level(self.level, self.x + dx, self.y + dy, self.z + dz)
    }

    /// The same level neighbour across `face`.
    pub fn neighbour(self, face: Face) -> Self {
        self.with_coords(self.as_ivec3() + face.step())
    }

    /// The key one level coarser that contains this one.
    pub fn parent(self) -> Self {
        let c = self.as_ivec3().div_euclid(IVec3::splat(2));
        Self::at_level(self.level + 1, c.x, c.y, c.z)
    }

    /// The eight keys one level finer, x fastest, then y, then z. `None` at level 0.
    pub fn children(self) -> Option<[Self; 8]> {
        let level = self.level.checked_sub(1)?;
        let base = self.as_ivec3() * 2;
        Some(std::array::from_fn(|i| {
            let c = base + IVec3::new((i & 1) as i32, ((i >> 1) & 1) as i32, (i >> 2) as i32);
            Self::at_level(level, c.x, c.y, c.z)
        }))
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

    /// Every level 0 chunk whose stored block intersects the inclusive world sample box `min` to
    /// `max`. A brush's dirty set is this box for its padded bounds.
    pub fn chunks_intersecting_samples(min: IVec3, max: IVec3) -> Vec<Self> {
        Self::chunks_intersecting_samples_at(0, min, max)
    }

    /// Every chunk of `level` whose stored block intersects the inclusive box `min` to `max` of
    /// world sample coordinates in that level's units.
    pub fn chunks_intersecting_samples_at(level: u8, min: IVec3, max: IVec3) -> Vec<Self> {
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
                    keys.push(Self::at_level(level, x, y, z));
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
    fn levels_scale_the_extent_and_nest() {
        let k = ChunkKey::at_level(3, -1, 2, 5);
        assert_eq!(k.voxel_size_m(0.5), 4.0);
        assert_eq!(k.extent_m(0.5), 128.0);
        assert_eq!(k.origin_m(0.5), DVec3::new(-128.0, 256.0, 640.0));
        let children = k.children().unwrap();
        assert!(children.iter().all(|c| c.level == 2 && c.parent() == k));
        assert_eq!(children[0], ChunkKey::at_level(2, -2, 4, 10));
        assert_eq!(children[7], ChunkKey::at_level(2, -1, 5, 11));
        assert_eq!(ChunkKey::new(0, 0, 0).children(), None);
        let (min, max) = k.bounds_m(0.5);
        let (cmin, _) = children[0].bounds_m(0.5);
        let (_, cmax) = children[7].bounds_m(0.5);
        assert_eq!((min, max), (cmin, cmax));
        for face in Face::ALL {
            assert_eq!(k.neighbour(face).neighbour(face.opposite()), k);
            assert_eq!(face.opposite().opposite(), face);
        }
        let mask: u8 = Face::ALL.iter().map(|f| f.bit()).sum();
        assert_eq!(mask, 0b11_1111);
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
