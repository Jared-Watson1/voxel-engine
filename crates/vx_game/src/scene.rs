//! The M0 scene: one chunk holding an analytic signed distance sphere, generated and meshed on a
//! worker thread so the main thread never waits on configuration I/O, generation or meshing (D9).

use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::time::Instant;

use glam::DVec3;
use serde::Deserialize;
use vx_core::{WorldConfig, WorldPos};
use vx_voxel::{Chunk, ChunkKey, ChunkMesh, mesh_regular};

/// Scene parameters for the M0 test sphere, loaded from `assets/config/m0_scene.ron`.
///
/// The sphere is defined in metres and does not change with the voxel size, so a smaller voxel
/// size samples the same shape more finely (D11).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SceneConfig {
    /// Radius of the test sphere. Default 6.0 m, which fits inside one chunk at voxel sizes of
    /// 0.5 m and above (the chunk is 16 m across at 0.5 m).
    pub sphere_radius_m: f64,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            sphere_radius_m: 6.0,
        }
    }
}

impl SceneConfig {
    /// The chunk holding the sphere; the sphere sits at its centre.
    pub const CHUNK_KEY: ChunkKey = ChunkKey::new(0, 0, 0);

    fn parse(text: &str) -> Result<Self, String> {
        let config: Self = ron::from_str(text).map_err(|e| e.to_string())?;
        if !(config.sphere_radius_m.is_finite() && config.sphere_radius_m > 0.0) {
            return Err(format!(
                "sphere_radius_m = {} must be positive",
                config.sphere_radius_m
            ));
        }
        Ok(config)
    }
}

pub struct ScenePaths {
    pub world: PathBuf,
    pub scene: PathBuf,
}

pub struct ChunkResult {
    pub job: u64,
    pub world: WorldConfig,
    pub scene: SceneConfig,
    pub mesh: ChunkMesh,
    /// Sphere centre in the body fixed frame.
    pub centre: WorldPos,
    pub generate_ms: f64,
    pub mesh_ms: f64,
    /// False when the sphere is larger than the chunk at this voxel size, so the mesh is clipped
    /// open at the chunk faces.
    pub sphere_fits: bool,
}

/// Loads both configs, generates the chunk and meshes it on a new thread, then sends the result.
pub fn spawn_chunk_job(
    job: u64,
    paths: &ScenePaths,
    voxel_size_override_m: Option<f64>,
    tx: Sender<Result<ChunkResult, String>>,
) {
    let world_path = paths.world.clone();
    let scene_path = paths.scene.clone();
    let spawned = std::thread::Builder::new()
        .name("vx-chunk".into())
        .spawn(move || {
            let result = build_chunk(job, &world_path, &scene_path, voxel_size_override_m);
            // A closed receiver means the app is shutting down.
            let _ = tx.send(result);
        });
    if let Err(e) = spawned {
        tracing::error!("could not spawn the chunk worker: {e}");
    }
}

fn build_chunk(
    job: u64,
    world_path: &std::path::Path,
    scene_path: &std::path::Path,
    voxel_size_override_m: Option<f64>,
) -> Result<ChunkResult, String> {
    profiling::scope!("chunk job");
    let mut world = WorldConfig::load(world_path).map_err(|e| e.to_string())?;
    if let Some(v) = voxel_size_override_m {
        world.voxel_size_m = v;
        world.validate().map_err(|e| e.to_string())?;
    }
    let scene_text = std::fs::read_to_string(scene_path)
        .map_err(|e| format!("reading {}: {e}", scene_path.display()))?;
    let scene = SceneConfig::parse(&scene_text)?;

    let extent_m = world.chunk_extent_m();
    let radius_m = scene.sphere_radius_m;

    let start = Instant::now();
    let mut chunk = Chunk::new(SceneConfig::CHUNK_KEY, world.voxel_size_m);
    let origin = chunk.origin();
    let centre = WorldPos::new(origin.frame, origin.m + DVec3::splat(0.5 * extent_m));
    chunk.fill_from(|p| p.distance(centre.m) - radius_m);
    let generated = Instant::now();
    let mesh = mesh_regular(&chunk);
    let meshed = Instant::now();

    Ok(ChunkResult {
        job,
        world,
        scene,
        mesh,
        centre,
        generate_ms: (generated - start).as_secs_f64() * 1.0e3,
        mesh_ms: (meshed - generated).as_secs_f64() * 1.0e3,
        sphere_fits: 2.0 * radius_m < extent_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_scene_file_holds_the_defaults() {
        let text = include_str!("../../../assets/config/m0_scene.ron");
        assert_eq!(SceneConfig::parse(text).unwrap(), SceneConfig::default());
    }

    #[test]
    fn default_sphere_fits_at_half_and_one_metre_voxels() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/config");
        for voxel_size_m in [0.5, 1.0] {
            let result = build_chunk(
                0,
                &root.join("world.ron"),
                &root.join("m0_scene.ron"),
                Some(voxel_size_m),
            )
            .unwrap();
            assert!(result.sphere_fits);
            assert!(result.mesh.triangle_count() > 0);
        }
    }

    #[test]
    fn rejects_bad_radius() {
        assert!(SceneConfig::parse("(sphere_radius_m: -1.0)").is_err());
        assert!(SceneConfig::parse("(radius: 1.0)").is_err());
    }
}
