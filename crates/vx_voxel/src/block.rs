use glam::{DVec3, IVec3, UVec3};
use vx_core::CHUNK_SIZE;

use crate::chunk::Voxel;
use crate::key::{ChunkKey, Face};

/// A box of samples at one octree level outside any chunk: the fine resolution slabs that a
/// coarse chunk's transition cells read (D22). Sample positions use the same arithmetic as
/// `Chunk::sample_position` (integer world sample index times the level's voxel size, one
/// rounding), so a slab sample is bit identical to the fine chunk's sample at the same index.
#[derive(Clone)]
pub struct SampleBlock {
    level: u8,
    min: IVec3,
    dims: UVec3,
    voxel_size_m: f64,
    voxels: Box<[Voxel]>,
}

impl SampleBlock {
    /// An all air block of `dims` samples whose first sample is world sample `min` of `level`.
    pub fn new(level: u8, min: IVec3, dims: UVec3, base_voxel_size_m: f64) -> Self {
        assert!(
            dims.cmpgt(UVec3::ZERO).all(),
            "a sample block is never empty"
        );
        let voxel_size_m = ChunkKey::at_level(level, 0, 0, 0).voxel_size_m(base_voxel_size_m);
        let len = (dims.x * dims.y * dims.z) as usize;
        Self {
            level,
            min,
            dims,
            voxel_size_m,
            voxels: vec![Voxel::AIR; len].into_boxed_slice(),
        }
    }

    pub fn level(&self) -> u8 {
        self.level
    }

    /// World sample coordinate of the first sample.
    pub fn min(&self) -> IVec3 {
        self.min
    }

    /// World sample coordinate of the last sample (inclusive).
    pub fn max(&self) -> IVec3 {
        self.min + self.dims.as_ivec3() - IVec3::ONE
    }

    pub fn dims(&self) -> UVec3 {
        self.dims
    }

    pub fn voxel_size_m(&self) -> f64 {
        self.voxel_size_m
    }

    /// Frame position of world sample `world` in metres.
    pub fn sample_position(&self, world: IVec3) -> DVec3 {
        world.as_dvec3() * self.voxel_size_m
    }

    /// Index of world sample `world` in [`SampleBlock::voxels`], x fastest, then y, then z.
    pub fn index(&self, world: IVec3) -> Option<usize> {
        let local = world - self.min;
        if local.cmplt(IVec3::ZERO).any() || local.cmpge(self.dims.as_ivec3()).any() {
            return None;
        }
        let l = local.as_uvec3();
        Some((l.x + self.dims.x * (l.y + self.dims.y * l.z)) as usize)
    }

    pub fn get(&self, world: IVec3) -> Option<Voxel> {
        self.index(world).map(|i| self.voxels[i])
    }

    pub fn set(&mut self, world: IVec3, voxel: Voxel) {
        let i = self
            .index(world)
            .unwrap_or_else(|| panic!("sample {world} outside the block"));
        self.voxels[i] = voxel;
    }

    pub fn voxels(&self) -> &[Voxel] {
        &self.voxels
    }

    pub fn voxels_mut(&mut self) -> &mut [Voxel] {
        &mut self.voxels
    }
}

/// Samples across a slab: one on each side of the face plane, for central difference gradients.
pub const SLAB_DEPTH: u32 = 3;
/// Samples along each in plane axis of a slab: the fine face's 65 corner samples plus one apron
/// sample on each side.
pub const SLAB_WIDTH: u32 = 2 * CHUNK_SIZE as u32 + 3;

/// The empty fine resolution slab for `key`'s transition cells on `face` (D22), at level
/// `key.level - 1`; `None` at level 0. Along the face axis it holds the face plane and one sample
/// on each side; along the two in plane axes the face's samples plus one on each side. In fine
/// units the chunk spans `64 K` to `64 K + 64`.
pub fn transition_slab(key: ChunkKey, face: Face, base_voxel_size_m: f64) -> Option<SampleBlock> {
    let level = key.level.checked_sub(1)?;
    let fine = CHUNK_SIZE as i32 * 2;
    let axis = face.axis();
    let mut min = key.as_ivec3() * fine - IVec3::ONE;
    let mut dims = UVec3::splat(SLAB_WIDTH);
    let plane = key.as_ivec3()[axis] * fine + if face.is_high() { fine } else { 0 };
    min[axis] = plane - 1;
    dims[axis] = SLAB_DEPTH;
    Some(SampleBlock::new(level, min, dims, base_voxel_size_m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_sits_on_the_face_at_the_finer_level() {
        let key = ChunkKey::at_level(2, 1, -1, 0);
        assert!(transition_slab(ChunkKey::new(0, 0, 0), Face::LowX, 1.0).is_none());
        let slab = transition_slab(key, Face::HighX, 1.0).unwrap();
        assert_eq!(slab.level(), 1);
        assert_eq!(slab.dims(), UVec3::new(3, 67, 67));
        assert_eq!(slab.min(), IVec3::new(127, -65, -1));
        assert_eq!(slab.max(), IVec3::new(129, 1, 65));
        let low_y = transition_slab(key, Face::LowY, 1.0).unwrap();
        assert_eq!(low_y.dims(), UVec3::new(67, 3, 67));
        assert_eq!(low_y.min().y, -65);
        // The face plane of the slab coincides with the coarse chunk's face in metres.
        let (min_m, _) = key.bounds_m(1.0);
        assert_eq!(low_y.sample_position(IVec3::new(0, -64, 0)).y, min_m.y);
    }

    #[test]
    fn index_round_trips_and_rejects_outside() {
        let mut b = SampleBlock::new(0, IVec3::new(-2, 3, 4), UVec3::new(2, 3, 4), 1.0);
        let v = Voxel::solid(crate::Material::ROCK);
        b.set(IVec3::new(-1, 5, 7), v);
        assert_eq!(b.get(IVec3::new(-1, 5, 7)), Some(v));
        assert_eq!(b.index(IVec3::new(-1, 5, 7)), Some(b.voxels().len() - 1));
        assert_eq!(b.get(IVec3::new(0, 5, 7)), None);
        assert_eq!(b.get(IVec3::new(-3, 3, 4)), None);
    }
}
