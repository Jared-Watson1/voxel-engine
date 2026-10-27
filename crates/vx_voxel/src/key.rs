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
        let _ = (min, max);
        todo!("workstream B")
    }
}
