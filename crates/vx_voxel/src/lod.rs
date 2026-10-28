//! Octree clipmap over the whole planet (D20).
//!
//! The tree decides which chunk keys must exist and which are drawn; the caller owns the data,
//! the jobs and the meshes. Each [`LodTree::update`] returns the keys to make ready (generate and
//! mesh, or find empty), the keys to release, and the visibility and transition mask changes that
//! became safe to commit. A split commits only once all eight children are ready and a merge only
//! once the parent is ready, so the drawn set always covers the planet shell with no hole and no
//! overlap. Keys entirely outside the shell are never requested and count as ready and empty.

use glam::DVec3;
use vx_core::{Priority, WorldConfig};

use crate::key::ChunkKey;

/// Octree parameters, from [`WorldConfig`] and the planet's shell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodParams {
    /// Level 0 voxel size in metres.
    pub base_voxel_size_m: f64,
    /// Number of levels; the top level `levels - 1` holds the roots.
    pub levels: u8,
    /// A node splits when the camera is within this many of its own chunk extents of its box.
    pub radius_chunks: f64,
    /// A split node merges beyond `radius_chunks + hysteresis_chunks` extents.
    pub hysteresis_chunks: f64,
    /// Inner and outer radius of the shell around the body origin that can hold the surface at
    /// any level (from the sampler). Nodes whose box lies entirely inside the inner or outside the
    /// outer radius are never requested.
    pub shell_m: (f64, f64),
}

impl LodParams {
    pub fn new(config: &WorldConfig, shell_m: (f64, f64)) -> Self {
        Self {
            base_voxel_size_m: config.voxel_size_m,
            levels: config.lod_levels as u8,
            radius_chunks: config.lod_radius_chunks,
            hysteresis_chunks: config.lod_hysteresis_chunks,
            shell_m,
        }
    }
}

/// A key the caller should make ready: generate its data and mesh it (or find it empty), then
/// report with [`LodTree::mark_ready`]. Lower priorities are more urgent (`vx_core::Priority`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodRequest {
    pub key: ChunkKey,
    pub priority: Priority,
}

/// Changes from one [`LodTree::update`], to apply in this order: start the requests, apply the
/// visibility and mask changes (a handful of chunk table writes, D23), then release.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LodUpdate {
    /// Newly needed keys, most urgent first.
    pub requests: Vec<LodRequest>,
    /// Keys no longer needed: cancel their jobs, free their data and meshes. Never a drawn key.
    pub releases: Vec<ChunkKey>,
    /// Keys whose drawn state changed this update.
    pub visibility: Vec<(ChunkKey, bool)>,
    /// Drawn keys whose transition mask changed (bit `Face as u8` set when the neighbour across
    /// that face is drawn one level finer).
    pub masks: Vec<(ChunkKey, u8)>,
}

/// Counters for the overlay.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LodStats {
    /// Nodes held by the tree, requested or not.
    pub nodes: usize,
    /// Drawn keys per level, index 0 the finest.
    pub drawn_per_level: Vec<usize>,
    /// Requested keys not yet reported ready.
    pub pending: usize,
    pub splits_total: u64,
    pub merges_total: u64,
    /// Main thread time of the last update.
    pub update_ms: f64,
}

pub struct LodTree {
    params: LodParams,
}

impl LodTree {
    pub fn new(params: LodParams) -> Self {
        Self { params }
    }

    pub fn params(&self) -> &LodParams {
        &self.params
    }

    /// Advances the tree towards the structure the camera at `camera_m` (body fixed metres) wants.
    /// Does bounded work per call, so a large camera jump settles over several frames.
    pub fn update(&mut self, camera_m: DVec3) -> LodUpdate {
        let _ = camera_m;
        todo!("workstream C: octree split, merge, balance and readiness gating")
    }

    /// The caller finished a requested key: its mesh is uploaded (hidden) or it has none.
    pub fn mark_ready(&mut self, key: ChunkKey) {
        let _ = key;
        todo!("workstream C")
    }

    /// A requested key could not be made ready (for example a full GPU pool); the tree treats it
    /// as ready and empty so the octree never stalls, and counts it.
    pub fn mark_failed(&mut self, key: ChunkKey) {
        let _ = key;
        todo!("workstream C")
    }

    pub fn is_drawn(&self, key: ChunkKey) -> bool {
        let _ = key;
        todo!("workstream C")
    }

    /// Transition mask of a drawn key, 0 otherwise.
    pub fn mask(&self, key: ChunkKey) -> u8 {
        let _ = key;
        todo!("workstream C")
    }

    /// Every drawn key with its mask, sorted by key.
    pub fn drawn(&self) -> Vec<(ChunkKey, u8)> {
        todo!("workstream C")
    }

    pub fn stats(&self) -> LodStats {
        todo!("workstream C")
    }
}
