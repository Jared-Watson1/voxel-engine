//! Chunk storage and compression, signed distance brushes, the planet octree and Transvoxel
//! meshing with transition cells (D2, D20, D22).
//!
//! A world is an octree of 32^3 chunks keyed by [`ChunkKey`] (level, x, y, z). The dense working
//! form is [`Chunk`] (35^3 samples, apron included, D16); the resident form is [`ChunkStore`]
//! (D17). Brushes edit dense chunks and sample blocks and report which chunks must remesh. The
//! mesher emits [`PackedMesh`] with the shared packed vertex (D14) and per face transition ranges.

mod block;
mod brush;
mod chunk;
mod key;
pub mod lod;
mod mesh;
mod store;
mod transvoxel;

pub use block::{SLAB_DEPTH, SLAB_WIDTH, SampleBlock, transition_slab};
pub use brush::{Brush, BrushOp, BrushShape};
pub use chunk::{Chunk, DISTANCE_UNITS_PER_VOXEL, Material, SAMPLES_PER_AXIS, Voxel};
pub use key::{ChunkKey, Face};
pub use lod::{LodParams, LodRequest, LodStats, LodTree, LodUpdate};
pub use mesh::{PackedMesh, mesh_chunk, mesh_chunk_lod};
pub use store::ChunkStore;

#[cfg(test)]
mod tests;
