//! Chunk storage and compression, signed distance brushes and Transvoxel meshing (D2).
//!
//! A world is a grid of 32^3 chunks keyed by [`ChunkKey`]. The dense working form is [`Chunk`]
//! (35^3 samples, apron included, D16); the resident form is [`ChunkStore`] (D17). Brushes edit
//! dense chunks and report which chunks must remesh. The mesher emits [`PackedMesh`] with the
//! shared packed vertex (D14).

mod brush;
mod chunk;
mod key;
mod mesh;
mod store;

pub use brush::{Brush, BrushOp, BrushShape};
pub use chunk::{Chunk, DISTANCE_UNITS_PER_VOXEL, Material, SAMPLES_PER_AXIS, Voxel};
pub use key::ChunkKey;
pub use mesh::{PackedMesh, mesh_chunk};
pub use store::ChunkStore;

#[cfg(test)]
mod tests;
