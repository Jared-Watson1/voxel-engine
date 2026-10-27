//! Chunk storage, signed distance sampling and Transvoxel meshing (D2).
//!
//! M0 holds one dense chunk and meshes regular cells only. Compression, edits and transition cells
//! arrive in M1 and M2.

mod chunk;
mod mesh;

pub use chunk::{Chunk, DISTANCE_UNITS_PER_VOXEL, Material, SAMPLES_PER_AXIS, Voxel};
pub use mesh::{ChunkMesh, mesh_regular};

#[cfg(test)]
mod tests;
