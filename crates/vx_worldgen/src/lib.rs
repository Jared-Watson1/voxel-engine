//! Terrain generation (D5, D13): a pure function of (position, seed, parameters), CPU
//! authoritative and bit identical on every machine (engineering rule 2).
//!
//! M1 is a flat world with +Y up: a fractal heightfield plus 3D detail noise, with rock, soil,
//! grass and sand assigned by height, depth and slope. The planet atlas arrives in M3.

mod sampler;
mod textures;

pub use sampler::TerrainSampler;
pub use textures::{MaterialTextureSet, material_textures};
