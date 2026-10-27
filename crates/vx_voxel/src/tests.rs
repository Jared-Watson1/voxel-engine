//! Mesh invariants on an analytic sphere: closed, consistently wound, outward facing, on the
//! surface, and scale invariant across the supported voxel sizes (D11).

use std::collections::{HashMap, HashSet};

use glam::{DVec3, Vec3};
use vx_core::{CHUNK_SIZE, WorldConfig, WorldPos};

use crate::{Chunk, ChunkMesh, mesh_regular};

/// Radius as a fraction of the chunk extent; keeps the sphere inside the cell region (not touching
/// the apron) at every voxel size.
const SPHERE_RADIUS_CHUNK_FRACTION: f64 = 0.35;

struct SphereCase {
    centre_local_m: Vec3,
    radius_m: f32,
    voxel_size_m: f32,
    mesh: ChunkMesh,
}

fn sphere_case(voxel_size_m: f64) -> SphereCase {
    let extent_m = CHUNK_SIZE as f64 * voxel_size_m;
    // An origin far from the frame origin so the test also covers f64 sampling there.
    let origin = WorldPos::body_fixed(DVec3::new(1.0e6, -2.0e5, 3.0e4));
    let centre = origin.m + DVec3::splat(0.5 * extent_m);
    let radius_m = SPHERE_RADIUS_CHUNK_FRACTION * extent_m;
    let mut chunk = Chunk::new(origin, voxel_size_m);
    chunk.fill_from(|p| p.distance(centre) - radius_m);
    SphereCase {
        centre_local_m: (centre - origin.m).as_vec3(),
        radius_m: radius_m as f32,
        voxel_size_m: voxel_size_m as f32,
        mesh: mesh_regular(&chunk),
    }
}

fn voxel_sizes() -> [f64; 4] {
    let range = WorldConfig::VOXEL_SIZE_RANGE_M;
    [
        *range.start(),
        0.5,
        WorldConfig::default().voxel_size_m,
        *range.end(),
    ]
}

fn vertex(mesh: &ChunkMesh, i: u32) -> Vec3 {
    Vec3::from(mesh.positions[i as usize])
}

#[test]
fn sphere_mesh_is_closed_and_consistently_wound() {
    for voxel_size_m in voxel_sizes() {
        let case = sphere_case(voxel_size_m);
        let mesh = &case.mesh;
        assert!(
            mesh.triangle_count() > 100,
            "sphere produced {} triangles",
            mesh.triangle_count()
        );

        // Every directed edge appears exactly once and its reverse exactly once: each undirected
        // edge is shared by two triangles that agree on orientation.
        let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in mesh.indices.chunks_exact(3) {
            assert!(
                tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2],
                "degenerate indices"
            );
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                *directed.entry((a, b)).or_default() += 1;
            }
        }
        for (&(a, b), &count) in &directed {
            assert_eq!(
                count, 1,
                "directed edge {a}->{b} used {count} times at {voxel_size_m} m"
            );
            assert_eq!(
                directed.get(&(b, a)),
                Some(&1),
                "edge {a}-{b} is a boundary edge at {voxel_size_m} m"
            );
        }

        // A closed genus 0 surface has Euler characteristic 2.
        let used: HashSet<u32> = mesh.indices.iter().copied().collect();
        let v = used.len() as i64;
        let e = (directed.len() / 2) as i64;
        let f = mesh.triangle_count() as i64;
        assert_eq!(v - e + f, 2, "Euler characteristic at {voxel_size_m} m");
    }
}

#[test]
fn sphere_faces_and_normals_point_outward() {
    for voxel_size_m in voxel_sizes() {
        let case = sphere_case(voxel_size_m);
        let mesh = &case.mesh;
        for tri in mesh.indices.chunks_exact(3) {
            let (a, b, c) = (
                vertex(mesh, tri[0]),
                vertex(mesh, tri[1]),
                vertex(mesh, tri[2]),
            );
            let face = (b - a).cross(c - a);
            let outward = (a + b + c) / 3.0 - case.centre_local_m;
            assert!(
                face.dot(outward) > 0.0,
                "triangle {tri:?} winds inward at {voxel_size_m} m"
            );
        }
        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
            let (p, n) = (Vec3::from(*p), Vec3::from(*n));
            assert!(p.is_finite() && n.is_finite());
            assert!((n.length() - 1.0).abs() < 1.0e-4, "normal not unit length");
            let radial = (p - case.centre_local_m).normalize();
            assert!(n.dot(radial) > 0.95, "normal {n} not outward at {p}");
        }
    }
}

#[test]
fn vertices_lie_on_the_sphere() {
    for voxel_size_m in voxel_sizes() {
        let case = sphere_case(voxel_size_m);
        for p in &case.mesh.positions {
            let r = Vec3::from(*p).distance(case.centre_local_m);
            let error_voxels = (r - case.radius_m).abs() / case.voxel_size_m;
            assert!(
                error_voxels < 0.05,
                "vertex off the surface by {error_voxels} voxels"
            );
        }
    }
}

#[test]
fn voxel_size_only_scales_the_mesh() {
    let reference = sphere_case(WorldConfig::default().voxel_size_m);
    for voxel_size_m in voxel_sizes() {
        let case = sphere_case(voxel_size_m);
        assert_eq!(case.mesh.indices, reference.mesh.indices);
        let scale = case.voxel_size_m / reference.voxel_size_m;
        for (p, q) in case.mesh.positions.iter().zip(&reference.mesh.positions) {
            let expected = Vec3::from(*q) * scale;
            assert!(Vec3::from(*p).abs_diff_eq(expected, 1.0e-4 * case.voxel_size_m));
        }
    }
}

#[test]
fn uniform_chunks_produce_no_triangles() {
    let origin = WorldPos::default();
    let mut air = Chunk::new(origin, 1.0);
    air.fill_from(|_| 10.0);
    assert!(mesh_regular(&air).is_empty());
    let mut rock = Chunk::new(origin, 1.0);
    rock.fill_from(|_| -10.0);
    assert!(mesh_regular(&rock).is_empty());
}
