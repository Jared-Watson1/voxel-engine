//! A chunk far from the frame origin, meshed and taken through the camera relative frame and the
//! reverse-Z projection exactly as the renderer receives it (D4).

use glam::{DVec3, Vec3};
use vx_core::{Projection, WorldConfig, WorldPos, camera_relative};
use vx_voxel::{Chunk, mesh_regular};

#[test]
fn chunk_at_a_billion_metres_projects_without_precision_loss() {
    let config = WorldConfig::default();
    let extent_m = config.chunk_extent_m();
    let origin = WorldPos::body_fixed(DVec3::new(1.0e9, 2.0e8, -7.0e8));
    let centre = origin.m + DVec3::splat(0.5 * extent_m);
    let radius_m = 0.3 * extent_m;
    let mut chunk = Chunk::new(origin, config.voxel_size_m);
    chunk.fill_from(|p| p.distance(centre) - radius_m);
    let mesh = mesh_regular(&chunk);
    assert!(!mesh.is_empty());

    // Camera on the +Z side looking down -Z at the sphere centre, so the view rotation is identity.
    let camera = WorldPos::body_fixed(centre + DVec3::new(0.0, 0.0, 2.0 * extent_m));
    let offset = camera_relative(mesh.origin, camera);
    let view_proj = Projection::reverse_z_infinite(1.0, 16.0 / 9.0, 0.05);

    for p in &mesh.positions {
        let local = Vec3::from(*p);
        let gpu = offset + local;
        let exact = origin.m + local.as_dvec3() - camera.m;
        assert!(
            (gpu.as_dvec3() - exact).abs().max_element() < 1.0e-3,
            "camera relative vertex off by more than 1 mm"
        );
        let clip = view_proj * gpu.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.z > 0.0 && ndc.z < 1.0, "depth {} outside (0, 1)", ndc.z);
        assert!(ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0, "vertex off screen");
    }
}
