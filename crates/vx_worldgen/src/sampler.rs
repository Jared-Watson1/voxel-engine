use glam::DVec3;
use vx_core::TerrainParams;
use vx_voxel::{Chunk, Material};

/// Samples the M1 terrain. Every method is a pure function of its arguments, the seed and the
/// parameters: no interior mutability that affects results, no hash map iteration, no
/// transcendental functions whose results differ between platforms (D13).
#[derive(Debug, Clone)]
pub struct TerrainSampler {
    seed: u64,
    params: TerrainParams,
}

impl TerrainSampler {
    pub fn new(seed: u64, params: TerrainParams) -> Self {
        Self { seed, params }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn params(&self) -> &TerrainParams {
        &self.params
    }

    /// Surface height in metres at a horizontal position (the heightfield without 3D detail).
    pub fn height_m(&self, x_m: f64, z_m: f64) -> f64 {
        let _ = (x_m, z_m);
        todo!("workstream C")
    }

    /// Signed distance to the terrain surface in metres (negative inside) and the material of the
    /// solid at `p`. The material is meaningful only where the distance is negative.
    pub fn sample(&self, p: DVec3) -> (f64, Material) {
        let _ = p;
        todo!("workstream C")
    }

    /// Fills every sample of `chunk`, apron included, with exactly the values [`Self::sample`]
    /// returns at each sample position (after the chunk's quantisation). Implementations may skip
    /// work for samples provably outside the distance band, but the result must be bit identical
    /// to the per sample path.
    pub fn fill_chunk(&self, chunk: &mut Chunk) {
        let _ = chunk;
        todo!("workstream C")
    }
}
