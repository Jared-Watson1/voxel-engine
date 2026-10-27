//! Mesh invariants on an analytic sphere: closed, consistently wound, outward facing, on the
//! surface, and bit identical across the supported voxel sizes (D11). Exact seams between chunks
//! sharing a face (D15, D16), the material rule, finite packed attributes and timing.

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::Instant;

use glam::{DVec3, IVec3, Vec3};
use vx_core::mesh::{MAX_MATERIAL_WEIGHT, POSITION_UNITS_PER_VOXEL};
use vx_core::{CHUNK_SIZE, PackedVertex, WorldConfig};

use crate::{Chunk, ChunkKey, Material, PackedMesh, Voxel, mesh_chunk};

/// Radius as a fraction of the chunk extent; keeps the sphere inside the cell region (not touching
/// the apron) at every voxel size.
const SPHERE_RADIUS_CHUNK_FRACTION: f64 = 0.35;

struct SphereCase {
    centre_local_m: Vec3,
    radius_m: f32,
    voxel_size_m: f32,
    mesh: PackedMesh,
}

/// A sphere centred in a chunk far from the frame origin, so the test also covers f64 sampling
/// there. Returns the chunk, the centre relative to the chunk origin and the radius, in metres.
fn sphere_chunk(voxel_size_m: f64) -> (Chunk, DVec3, f64) {
    let extent_m = CHUNK_SIZE as f64 * voxel_size_m;
    let mut chunk = Chunk::new(ChunkKey::new(31_250, -6_250, 937), voxel_size_m);
    let origin = chunk.origin();
    let centre = origin.m + DVec3::splat(0.5 * extent_m);
    let radius_m = SPHERE_RADIUS_CHUNK_FRACTION * extent_m;
    chunk.fill_from(|p| p.distance(centre) - radius_m);
    (chunk, centre - origin.m, radius_m)
}

fn sphere_case(voxel_size_m: f64) -> SphereCase {
    let (chunk, centre_local_m, radius_m) = sphere_chunk(voxel_size_m);
    SphereCase {
        centre_local_m: centre_local_m.as_vec3(),
        radius_m: radius_m as f32,
        voxel_size_m: voxel_size_m as f32,
        mesh: mesh_chunk(&chunk),
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

/// Unpacked position of vertex `i` in chunk local metres.
fn position_m(case: &SphereCase, i: u32) -> Vec3 {
    case.mesh.vertices[i as usize].unpack().position_voxels * case.voxel_size_m
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
                position_m(&case, tri[0]),
                position_m(&case, tri[1]),
                position_m(&case, tri[2]),
            );
            let face = (b - a).cross(c - a);
            let outward = (a + b + c) / 3.0 - case.centre_local_m;
            assert!(
                face.dot(outward) > 0.0,
                "triangle {tri:?} winds inward at {voxel_size_m} m"
            );
        }
        for vertex in &mesh.vertices {
            let attributes = vertex.unpack();
            let p = attributes.position_voxels * case.voxel_size_m;
            let n = attributes.normal;
            assert!(p.is_finite() && n.is_finite());
            assert!((n.length() - 1.0).abs() < 1.0e-4, "normal not unit length");
            let radial = (p - case.centre_local_m).normalize();
            assert!(n.dot(radial) > 0.95, "normal {n} not outward at {p}");
        }
    }
}

#[test]
fn vertices_lie_on_the_sphere() {
    // Interpolation on the i8 distance field dominates; position quantisation adds at most half
    // a unit per axis.
    let tolerance_voxels = 0.05 + 1.0 / POSITION_UNITS_PER_VOXEL;
    for voxel_size_m in voxel_sizes() {
        let case = sphere_case(voxel_size_m);
        for vertex in &case.mesh.vertices {
            let p = vertex.unpack().position_voxels * case.voxel_size_m;
            let r = p.distance(case.centre_local_m);
            let error_voxels = (r - case.radius_m).abs() / case.voxel_size_m;
            assert!(
                error_voxels < tolerance_voxels,
                "vertex off the surface by {error_voxels} voxels"
            );
        }
    }
}

#[test]
fn voxel_size_only_scales_the_mesh() {
    // Packed positions are in voxel units, so a field that scales with the chunk packs to the
    // same words at every voxel size.
    let reference = sphere_case(WorldConfig::default().voxel_size_m);
    for voxel_size_m in voxel_sizes() {
        let case = sphere_case(voxel_size_m);
        assert!(
            case.mesh == reference.mesh,
            "packed mesh at {voxel_size_m} m differs from the {} m reference",
            reference.voxel_size_m
        );
    }
}

#[test]
fn uniform_chunks_produce_no_triangles() {
    let key = ChunkKey::new(0, 0, 0);
    let mut air = Chunk::new(key, 1.0);
    air.fill_from(|_| 10.0);
    let mesh = mesh_chunk(&air);
    assert!(mesh.is_empty() && mesh.vertices.is_empty());
    let mut rock = Chunk::new(key, 1.0);
    rock.fill_from(|_| -10.0);
    let mesh = mesh_chunk(&rock);
    assert!(mesh.is_empty() && mesh.vertices.is_empty());
    assert_eq!(mesh.key, key);
}

/// Packed position along `axis`, in units of 1/1024 voxel.
fn packed_coordinate(vertex: PackedVertex, axis: usize) -> u32 {
    match axis {
        0 => vertex.words[0] & 0xFFFF,
        1 => vertex.words[0] >> 16,
        _ => vertex.words[1] & 0xFFFF,
    }
}

fn with_packed_coordinate(mut vertex: PackedVertex, axis: usize, value: u32) -> PackedVertex {
    match axis {
        0 => vertex.words[0] = (vertex.words[0] & 0xFFFF_0000) | value,
        1 => vertex.words[0] = (vertex.words[0] & 0xFFFF) | (value << 16),
        _ => vertex.words[1] = (vertex.words[1] & 0xFFFF_0000) | value,
    }
    vertex
}

/// Meshes the chunk at the origin and its neighbour along `axis`, both filled from `sample`, and
/// asserts that the vertices on the shared face agree word for word as multisets.
fn assert_shared_face_matches(axis: usize, sample: impl Fn(DVec3) -> (f64, Material)) {
    let far = CHUNK_SIZE as u32 * POSITION_UNITS_PER_VOXEL as u32;
    let mut offset = IVec3::ZERO;
    offset[axis] = 1;
    let mut a = Chunk::new(ChunkKey::new(0, 0, 0), 1.0);
    let mut b = Chunk::new(ChunkKey::new(offset.x, offset.y, offset.z), 1.0);
    a.fill_from_with_material(&sample);
    b.fill_from_with_material(&sample);

    let mut face_a: Vec<[u32; 4]> = mesh_chunk(&a)
        .vertices
        .iter()
        .filter(|v| packed_coordinate(**v, axis) == far)
        .map(|v| v.words)
        .collect();
    let mut face_b: Vec<[u32; 4]> = mesh_chunk(&b)
        .vertices
        .iter()
        .filter(|v| packed_coordinate(**v, axis) == 0)
        .map(|v| with_packed_coordinate(*v, axis, far).words)
        .collect();
    assert!(
        !face_a.is_empty() && !face_b.is_empty(),
        "no vertices on the shared face along axis {axis}"
    );
    assert_eq!(
        face_a.len(),
        face_b.len(),
        "vertex counts on the shared face along axis {axis}"
    );
    face_a.sort_unstable();
    face_b.sort_unstable();
    assert_eq!(
        face_a, face_b,
        "shared face vertices differ along axis {axis}"
    );
}

/// A plane tilted along `axis` with its height on the next axis, crossing the shared face of the
/// chunk pair, with three materials layered by height so neighbour mixing occurs near the
/// boundaries.
fn layered_plane(axis: usize) -> impl Fn(DVec3) -> (f64, Material) {
    let height_axis = (axis + 1) % 3;
    let third_axis = (axis + 2) % 3;
    move |p: DVec3| {
        let height = p[height_axis];
        let surface = 16.0 + 0.3 * (p[axis] - 16.0) - 0.2 * p[third_axis];
        let material = if height > 20.0 {
            Material::GRASS
        } else if height > 12.0 {
            Material::SOIL
        } else {
            Material::ROCK
        };
        (height - surface, material)
    }
}

/// A sphere centred on the shared face of the chunk pair along `axis`, materials layered by y.
fn two_chunk_sphere(axis: usize) -> impl Fn(DVec3) -> (f64, Material) {
    let mut centre = DVec3::splat(0.5 * CHUNK_SIZE as f64);
    centre[axis] = CHUNK_SIZE as f64;
    let radius_m = 0.375 * CHUNK_SIZE as f64;
    move |p: DVec3| {
        let material = if p.y > 22.0 {
            Material::GRASS
        } else if p.y < 10.0 {
            Material::SAND
        } else {
            Material::ROCK
        };
        (p.distance(centre) - radius_m, material)
    }
}

#[test]
fn shared_face_vertices_are_identical() {
    for axis in 0..3 {
        assert_shared_face_matches(axis, layered_plane(axis));
        assert_shared_face_matches(axis, two_chunk_sphere(axis));
    }
}

/// The grid point inside the solid at either end of the edge holding a vertex.
fn solid_endpoint(chunk: &Chunk, position_voxels: Vec3) -> (IVec3, Voxel) {
    let axis = (0..3)
        .find(|&i| position_voxels[i].fract() != 0.0)
        .expect("vertex lies strictly inside an edge");
    let low = position_voxels.floor().as_ivec3();
    let mut high = low;
    high[axis] += 1;
    let mut inside = [low, high].into_iter().filter_map(|p| {
        let voxel = chunk.get(p.x as isize, p.y as isize, p.z as isize);
        voxel.is_inside().then_some((p, voxel))
    });
    let found = inside.next().expect("one endpoint is inside");
    assert!(inside.next().is_none(), "both endpoints inside");
    found
}

#[test]
fn material_slots_carry_the_solid_material() {
    let mut chunk = Chunk::new(ChunkKey::new(0, 0, 0), 1.0);
    // GRASS above 20 m over ROCK, surface between 21.5 and 23.1 m: solid endpoints at 21 m see
    // one ROCK neighbour below them, higher ones see only GRASS.
    chunk.fill_from_with_material(|p| {
        let material = if p.y > 20.0 {
            Material::GRASS
        } else {
            Material::ROCK
        };
        (p.y - 21.5 - 0.05 * p.x, material)
    });
    let mesh = mesh_chunk(&chunk);
    assert!(!mesh.is_empty());
    let mut saw_secondary = false;
    for vertex in &mesh.vertices {
        let attributes = vertex.unpack();
        let (local, solid) = solid_endpoint(&chunk, attributes.position_voxels);
        let [dominant, second, third, fourth] = attributes.materials;
        assert_eq!(
            dominant,
            (solid.material.0, MAX_MATERIAL_WEIGHT),
            "slot 0 at {}",
            attributes.position_voxels
        );
        assert!(
            local.y >= 21,
            "solid endpoint {local} below the surface band"
        );
        if local.y == 21 {
            assert_eq!(second, (Material::ROCK.0, 3), "slot 1 at {local}");
            saw_secondary = true;
        } else {
            assert_eq!(second, (0, 0), "slot 1 at {local}");
        }
        assert_eq!((third, fourth), ((0, 0), (0, 0)));
    }
    assert!(saw_secondary);
}

fn set_material(chunk: &mut Chunk, local: IVec3, material: Material) {
    let (x, y, z) = (local.x as isize, local.y as isize, local.z as isize);
    let mut voxel = chunk.get(x, y, z);
    assert!(voxel.is_inside());
    voxel.material = material;
    chunk.set(x, y, z, voxel);
}

#[test]
fn material_slots_rank_neighbours_by_count_then_id() {
    let mut chunk = Chunk::new(ChunkKey::new(0, 0, 0), 1.0);
    chunk.fill_from(|p| p.y - 9.5);
    // The solid endpoint (5, 9, 5) of the vertical edge at x = z = 5: two SOIL and two GRASS
    // neighbours tie on count and order by id, one SAND neighbour follows, ROCK is dominant.
    let endpoint = IVec3::new(5, 9, 5);
    set_material(&mut chunk, IVec3::new(4, 9, 5), Material::SOIL);
    set_material(&mut chunk, IVec3::new(6, 9, 5), Material::GRASS);
    set_material(&mut chunk, IVec3::new(5, 9, 4), Material::GRASS);
    set_material(&mut chunk, IVec3::new(5, 9, 6), Material::SOIL);
    set_material(&mut chunk, IVec3::new(5, 8, 5), Material::SAND);
    let mesh = mesh_chunk(&chunk);
    let expected_position = Vec3::new(5.0, 9.5, 5.0);
    let vertex = mesh
        .vertices
        .iter()
        .find(|v| v.unpack().position_voxels == expected_position)
        .expect("vertex on the edge above the endpoint");
    assert_eq!(solid_endpoint(&chunk, expected_position).0, endpoint);
    assert_eq!(
        vertex.unpack().materials,
        [
            (Material::ROCK.0, MAX_MATERIAL_WEIGHT),
            (Material::SOIL.0, 5),
            (Material::GRASS.0, 5),
            (Material::SAND.0, 3),
        ]
    );
    // A vertex whose solid endpoint has six neighbours of one other material fills the weight.
    let endpoint = IVec3::new(20, 9, 20);
    for offset in [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ] {
        let n = endpoint + offset;
        if chunk
            .get(n.x as isize, n.y as isize, n.z as isize)
            .is_inside()
        {
            set_material(&mut chunk, n, Material::SAND);
        }
    }
    let mesh = mesh_chunk(&chunk);
    let vertex = mesh
        .vertices
        .iter()
        .find(|v| v.unpack().position_voxels == Vec3::new(20.0, 9.5, 20.0))
        .expect("vertex on the edge above the endpoint");
    // Five of the six neighbours are inside (the one above is air): round(15 * 5 / 6) = 13.
    assert_eq!(
        vertex.unpack().materials,
        [
            (Material::ROCK.0, MAX_MATERIAL_WEIGHT),
            (Material::SAND.0, 13),
            (0, 0),
            (0, 0)
        ]
    );
}

/// A rolling surface with three materials by height, for attribute checks and timing.
fn noisy_plane_chunk() -> Chunk {
    let mut chunk = Chunk::new(ChunkKey::new(0, 0, 0), 1.0);
    chunk.fill_from_with_material(|p| {
        let height = 16.0
            + 3.0 * ((p.x * 0.37).sin() + (p.z * 0.53).cos())
            + 1.5 * (p.x * 0.11 + p.z * 0.17).sin();
        let material = if p.y > 18.0 {
            Material::GRASS
        } else if p.y > 14.0 {
            Material::SOIL
        } else {
            Material::ROCK
        };
        (p.y - height, material)
    });
    chunk
}

#[test]
fn no_nan_and_slot_zero() {
    let sphere = sphere_case(WorldConfig::default().voxel_size_m).mesh;
    for mesh in [mesh_chunk(&noisy_plane_chunk()), sphere] {
        assert!(!mesh.is_empty());
        for vertex in &mesh.vertices {
            let attributes = vertex.unpack();
            let p = attributes.position_voxels;
            assert!(p.is_finite());
            assert!(
                p.min_element() >= 0.0 && p.max_element() <= CHUNK_SIZE as f32,
                "position {p} outside the chunk"
            );
            let n = attributes.normal;
            assert!(n.is_finite());
            assert!(
                (n.length() - 1.0).abs() < 1.0e-4,
                "normal {n} not unit length"
            );
            assert!(
                attributes
                    .materials
                    .iter()
                    .all(|&(_, weight)| weight <= MAX_MATERIAL_WEIGHT)
            );
            assert_eq!(vertex.words[3], 0, "chunk slot word must stay zero");
        }
        for &i in &mesh.indices {
            assert!((i as usize) < mesh.vertices.len(), "index {i} out of range");
        }
    }
}

/// Prints the median wall time of `mesh_chunk` at 1.0 m voxels on one thread. Read it with
/// `cargo test -p vx_voxel --release -- --nocapture mesh_chunk_timing`.
#[test]
fn mesh_chunk_timing() {
    let (runs, profile) = if cfg!(debug_assertions) {
        (5, "debug")
    } else {
        (100, "release")
    };
    let (sphere, _, _) = sphere_chunk(1.0);
    for (name, chunk) in [("sphere", sphere), ("noisy plane", noisy_plane_chunk())] {
        let mut times = Vec::with_capacity(runs);
        let mut mesh = None;
        for _ in 0..runs {
            let start = Instant::now();
            let result = mesh_chunk(black_box(&chunk));
            times.push(start.elapsed());
            mesh = Some(result);
        }
        times.sort_unstable();
        let median_ms = times[runs / 2].as_secs_f64() * 1.0e3;
        let mesh = mesh.expect("at least one run");
        println!(
            "mesh_chunk {name}: median {median_ms:.3} ms over {runs} runs ({profile}), {} vertices, {} triangles, {} m voxels, one thread",
            mesh.vertices.len(),
            mesh.triangle_count(),
            chunk.voxel_size_m()
        );
    }
}

/// Positions come from the integer world sample index, so the apron copies and the shared face
/// vertices agree bit for bit at voxel sizes that are not powers of two as well (D16).
#[test]
fn shared_face_vertices_are_identical_at_odd_voxel_sizes() {
    for voxel_size_m in [0.3, 0.7, 1.5, 2.5] {
        let sdf = |p: DVec3| {
            let d = p.y - 20.0 * voxel_size_m - 0.3 * p.x - 0.2 * p.z;
            let material = if p.y > 18.0 * voxel_size_m {
                crate::Material::GRASS
            } else {
                crate::Material::ROCK
            };
            (d, material)
        };
        let mut a = Chunk::new(ChunkKey::new(0, 0, 0), voxel_size_m);
        let mut b = Chunk::new(ChunkKey::new(1, 0, 0), voxel_size_m);
        a.fill_from_with_material(sdf);
        b.fill_from_with_material(sdf);
        for x in -1..=1 {
            for y in -1..=(CHUNK_SIZE as i32 + 1) {
                for z in -1..=(CHUNK_SIZE as i32 + 1) {
                    assert_eq!(
                        a.get((x + CHUNK_SIZE as i32) as isize, y as isize, z as isize),
                        b.get(x as isize, y as isize, z as isize),
                        "apron copy differs at local ({x}, {y}, {z}) of chunk B, {voxel_size_m} m"
                    );
                }
            }
        }
        let face_x = (CHUNK_SIZE as f32 * vx_core::mesh::POSITION_UNITS_PER_VOXEL) as u32;
        let mut on_a: Vec<[u32; 4]> = mesh_chunk(&a)
            .vertices
            .iter()
            .map(|v| v.words)
            .filter(|w| w[0] & 0xFFFF == face_x)
            .collect();
        let mut on_b: Vec<[u32; 4]> = mesh_chunk(&b)
            .vertices
            .iter()
            .map(|v| v.words)
            .filter(|w| w[0] & 0xFFFF == 0)
            .map(|mut w| {
                w[0] |= face_x;
                w
            })
            .collect();
        on_a.sort_unstable();
        on_b.sort_unstable();
        assert!(
            !on_a.is_empty(),
            "no vertices on the shared face at {voxel_size_m} m"
        );
        assert_eq!(on_a, on_b, "shared face differs at {voxel_size_m} m");
    }
}
