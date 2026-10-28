//! Terrain generation (D5, D13): a pure function of (position, seed, parameters), CPU
//! authoritative and bit identical on every machine (engineering rule 2).
//!
//! M2 samples a planet (D21): seeded fBm of the unit direction on a per level cube sphere grid,
//! plus 3D detail noise, with materials by altitude, depth and slope against the radial up. The
//! M1 flat `TerrainSampler` remains until the benchmark moves to the planet. The planet atlas
//! arrives in M3.

mod planet;
mod sampler;
mod seed;
mod textures;

pub use planet::{ChunkClass, PlanetSampler};
pub use sampler::TerrainSampler;
pub use textures::{MaterialTextureSet, material_textures};
