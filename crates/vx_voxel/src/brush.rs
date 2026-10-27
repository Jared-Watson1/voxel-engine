use glam::{DVec3, IVec3};
use vx_core::{WorldConfig, WorldPos};

use crate::chunk::{Chunk, Material};
use crate::key::ChunkKey;

/// Shape of an edit in the body fixed frame, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BrushShape {
    Sphere {
        centre: WorldPos,
        radius_m: f64,
    },
    /// Axis aligned box from `min` to `max`.
    Box {
        min: WorldPos,
        max: WorldPos,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushOp {
    /// Removes the shape from the solid: `d = max(d, -shape)`; samples that leave the solid become
    /// air.
    Dig,
    /// Adds the shape to the solid with a material: `d = min(d, shape)`; samples the shape brought
    /// inside take the material.
    Place(Material),
}

/// A signed distance brush (design document, interaction). Applying it to every chunk in
/// [`Brush::affected_chunks`] keeps all apron copies of each sample equal (D16).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brush {
    pub shape: BrushShape,
    pub op: BrushOp,
}

impl Brush {
    /// Signed distance from the shape's surface in metres, negative inside.
    pub fn signed_distance_m(&self, p: DVec3) -> f64 {
        let _ = p;
        todo!("workstream B")
    }

    /// Inclusive world sample box (in voxels) outside which no sample can change: the shape's
    /// bounds padded by the representable distance band, because a dig also rewrites the
    /// distances of nearby solid samples.
    pub fn sample_bounds(&self, config: &WorldConfig) -> (IVec3, IVec3) {
        let _ = config;
        todo!("workstream B")
    }

    /// Every chunk whose stored block intersects [`Brush::sample_bounds`]: the set that must be
    /// edited and remeshed.
    pub fn affected_chunks(&self, config: &WorldConfig) -> Vec<ChunkKey> {
        let _ = config;
        todo!("workstream B")
    }

    /// Applies the brush to every sample of `chunk` inside the bounds, apron included. Returns
    /// true if any sample changed.
    pub fn apply(&self, chunk: &mut Chunk) -> bool {
        let _ = chunk;
        todo!("workstream B")
    }
}
