use glam::DVec3;
use vx_core::PlanetParams;
use vx_voxel::{Chunk, ChunkKey, Material, SampleBlock};

/// What a chunk can hold, decided from its bounds and the planet shell alone, before any noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkClass {
    /// Entirely above the shell: every sample is air.
    Air,
    /// Entirely below the shell: every sample is rock.
    Rock,
    /// Intersects the shell and may hold the surface; generate it.
    Surface,
}

/// Samples a planet's signed distance field in its body fixed frame (D3, D21): the surface height
/// `h(dir)` is seeded fBm of the unit direction times the radius, evaluated on a per level cube
/// sphere node grid and interpolated, so a chunk never evaluates the height noise per sample.
///
/// Every method is a pure function of its arguments, the seed, the parameters and the level 0
/// voxel size: the shared height cache changes speed, never results (engineering rule 2). No
/// transcendental function whose result differs between platforms is used in the sample path
/// (D13).
pub struct PlanetSampler {
    seed: u64,
    params: PlanetParams,
    base_voxel_size_m: f64,
}

impl PlanetSampler {
    pub fn new(seed: u64, params: &PlanetParams, base_voxel_size_m: f64) -> Self {
        Self {
            seed,
            params: *params,
            base_voxel_size_m,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn params(&self) -> &PlanetParams {
        &self.params
    }

    pub fn base_voxel_size_m(&self) -> f64 {
        self.base_voxel_size_m
    }

    pub fn radius_m(&self) -> f64 {
        self.params.radius_m()
    }

    /// Inner and outer radius between which the surface lies at every level, with margins for the
    /// detail noise, the interpolation and the representable distance band: outside them every
    /// sample is rock (inside) or air (outside).
    pub fn shell_m(&self) -> (f64, f64) {
        todo!("workstream B")
    }

    /// Classifies a chunk from its bounds and [`PlanetSampler::shell_m`]. Never returns `Air` or
    /// `Rock` for a chunk whose generated samples would hold a surface.
    pub fn classify(&self, key: ChunkKey) -> ChunkClass {
        let _ = key;
        todo!("workstream B")
    }

    /// Surface height above the mean radius in metres along `dir` (any length), at full detail
    /// and without the grid: for the camera's altitude and spawning, not for chunks.
    pub fn height_m(&self, dir: DVec3) -> f64 {
        let _ = dir;
        todo!("workstream B")
    }

    /// Signed distance in metres (negative inside) and the solid's material at `p` (body fixed
    /// metres) as level `level` defines the field: exactly the value `fill_chunk` and `fill_block`
    /// store at a sample position of that level, before quantisation.
    pub fn sample(&self, p: DVec3, level: u8) -> (f64, Material) {
        let _ = (p, level);
        todo!("workstream B")
    }

    /// Fills every sample of `chunk`, apron included, at the chunk's level. Bit identical to
    /// [`PlanetSampler::sample`] at every sample position, whatever the cache state or thread.
    pub fn fill_chunk(&self, chunk: &mut Chunk) {
        let _ = chunk;
        todo!("workstream B")
    }

    /// Fills every sample of `block` (a transition slab, D22) at the block's level, bit identical
    /// to the fine chunk's samples at the same world sample indices.
    pub fn fill_block(&self, block: &mut SampleBlock) {
        let _ = block;
        todo!("workstream B")
    }
}
