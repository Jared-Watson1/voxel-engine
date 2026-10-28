//! Transition seams (D22): a coarse level 1 chunk meshed with its six fine slabs, displaced by the
//! Rust mirror of the vertex shader (`secondary_position_voxels`) for its transition mask, together
//! with the level 0 chunks across the masked faces, forms a watertight surface. Vertices are
//! matched by bit equal positions in metres (f64): every directed edge between two distinct
//! positions must be balanced by its reverse, except edges lying on the outer boundary of the
//! region the chunks cover, where the surface is legitimately cut.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use glam::{DVec3, IVec3};
use vx_core::PackedVertex;
use vx_core::mesh::{FLAG_TRANSITION_FULL_RES, secondary_position_voxels};
use vx_voxel::{
    Chunk, ChunkKey, Face, Material, PackedMesh, SampleBlock, mesh_chunk, mesh_chunk_lod,
    transition_slab,
};

/// Signed distance in metres and material at a frame position, for chunks of `level`.
trait Field: Fn(u8, DVec3) -> (f64, Material) {}
impl<F: Fn(u8, DVec3) -> (f64, Material)> Field for F {}

struct Meshed {
    key: ChunkKey,
    mask: u8,
    chunk: Chunk,
    slabs: [Option<SampleBlock>; 6],
    mesh: PackedMesh,
    /// Vertex positions in metres after the displacement for `mask`.
    displaced_m: Vec<DVec3>,
    regular_vertex_count: usize,
}

impl Meshed {
    fn new(key: ChunkKey, mask: u8, base_m: f64, field: &impl Field) -> Self {
        let mut chunk = Chunk::new(key, base_m);
        chunk.fill_from_with_material(|p| field(key.level, p));
        let slabs = Face::ALL.map(|face| {
            transition_slab(key, face, base_m).map(|mut slab| {
                slab.fill_from_with_material(|p| field(key.level - 1, p));
                slab
            })
        });
        let mesh = mesh_chunk_lod(&chunk, &slabs);
        let plain = mesh_chunk(&chunk);
        let regular_vertex_count = plain.vertices.len();
        assert_eq!(
            mesh.vertices[..regular_vertex_count],
            plain.vertices[..],
            "slabs changed the regular vertices of {key:?}"
        );
        assert_eq!(
            mesh.regular_indices(),
            &plain.indices[..],
            "slabs changed the regular indices of {key:?}"
        );
        let voxel_size_m = key.voxel_size_m(base_m);
        let origin = (key.as_ivec3() * 32).as_dvec3();
        let displaced_m = mesh
            .vertices
            .iter()
            .map(|v| {
                let a = v.unpack();
                let local = secondary_position_voxels(a.position_voxels, a.normal, v.flags(), mask);
                (origin + local.as_dvec3()) * voxel_size_m
            })
            .collect();
        Self {
            key,
            mask,
            chunk,
            slabs,
            mesh,
            displaced_m,
            regular_vertex_count,
        }
    }

    /// The drawn triangles: the regular range and the masked transition ranges.
    fn triangles(&self) -> Vec<[u32; 3]> {
        let mut out: Vec<[u32; 3]> = self
            .mesh
            .regular_indices()
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        for face in Face::ALL {
            if self.mask & face.bit() != 0 {
                out.extend(
                    self.mesh
                        .transition_indices(face)
                        .chunks_exact(3)
                        .map(|t| [t[0], t[1], t[2]]),
                );
            }
        }
        out
    }

    fn bounds_m(&self, base_m: f64) -> (DVec3, DVec3) {
        self.key.bounds_m(base_m)
    }

    /// Coarse face points whose sample and the slab's fine sample at the same point lie on
    /// opposite sides of the surface, over the masked faces.
    fn flips(&self) -> usize {
        let mut flips = 0;
        for face in Face::ALL {
            let Some(slab) = &self.slabs[face as usize] else {
                continue;
            };
            if self.mask & face.bit() == 0 {
                continue;
            }
            let axis = face.axis();
            for i in 0..=32 {
                for j in 0..=32 {
                    let mut local = IVec3::ZERO;
                    local[axis] = if face.is_high() { 32 } else { 0 };
                    local[(axis + 1) % 3] = i;
                    local[(axis + 2) % 3] = j;
                    let coarse =
                        self.chunk
                            .get(local.x as isize, local.y as isize, local.z as isize);
                    let fine = slab
                        .get(self.key.world_sample(local) * 2)
                        .expect("face point in the slab");
                    flips += usize::from(coarse.is_inside() != fine.is_inside());
                }
            }
        }
        flips
    }
}

fn key_bits(p: DVec3) -> [u64; 3] {
    p.to_array().map(f64::to_bits)
}

/// Fine (level 0) chunks around the level 1 chunk at the origin for `mask`: every level 0 chunk
/// of the one chunk thick shell whose offsets outside the coarse chunk all point across masked
/// faces (faces, then the edges and corners where masked faces meet, as a 2:1 balanced octree
/// draws them).
fn fine_shell(mask: u8) -> Vec<ChunkKey> {
    let mut out = Vec::new();
    for z in -1..=2 {
        for y in -1..=2 {
            for x in -1..=2 {
                let c = IVec3::new(x, y, z);
                let mut ok = false;
                let mut all = true;
                for axis in 0..3 {
                    let face = match c[axis] {
                        -1 => Face::ALL[2 * axis],
                        2 => Face::ALL[2 * axis + 1],
                        _ => continue,
                    };
                    ok = true;
                    all &= mask & face.bit() != 0;
                }
                if ok && all {
                    out.push(ChunkKey::new(x, y, z));
                }
            }
        }
    }
    out
}

struct Report {
    triangles: usize,
    matched_edges: usize,
    boundary_edges: usize,
    unmatched: Vec<(DVec3, DVec3, i32)>,
}

/// Balances directed edges by bit equal endpoint positions across all chunks.
fn watertight(chunks: &[Meshed], base_m: f64) -> Report {
    let boxes: Vec<(DVec3, DVec3)> = chunks.iter().map(|c| c.bounds_m(base_m)).collect();
    let eps = 1.0e-6 * base_m;
    let covered = |p: DVec3| {
        boxes
            .iter()
            .any(|(lo, hi)| p.cmpge(*lo).all() && p.cmple(*hi).all())
    };
    let on_boundary = |p: DVec3| {
        let mut inside = 0;
        for corner in 0..8 {
            let s = DVec3::new(
                if corner & 1 == 0 { -eps } else { eps },
                if corner & 2 == 0 { -eps } else { eps },
                if corner & 4 == 0 { -eps } else { eps },
            );
            inside += usize::from(covered(p + s));
        }
        inside > 0 && inside < 8
    };

    let mut ids: HashMap<[u64; 3], usize> = HashMap::new();
    let mut positions: Vec<DVec3> = Vec::new();
    let mut directed: BTreeMap<(usize, usize), i32> = BTreeMap::new();
    let mut triangles = 0;
    for c in chunks {
        let id: Vec<usize> = c
            .displaced_m
            .iter()
            .map(|p| {
                *ids.entry(key_bits(*p)).or_insert_with(|| {
                    positions.push(*p);
                    positions.len() - 1
                })
            })
            .collect();
        for t in c.triangles() {
            triangles += 1;
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (a, b) = (id[a as usize], id[b as usize]);
                if a == b {
                    continue;
                }
                // Positive for a < b, negative for the reverse: a balanced edge sums to zero.
                let (key, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                *directed.entry(key).or_default() += sign;
            }
        }
    }
    let mut report = Report {
        triangles,
        matched_edges: 0,
        boundary_edges: 0,
        unmatched: Vec::new(),
    };
    for (&(a, b), &balance) in &directed {
        if balance == 0 {
            report.matched_edges += 1;
            continue;
        }
        let (pa, pb) = (positions[a], positions[b]);
        if on_boundary(pa) && on_boundary(pb) && on_boundary((pa + pb) * 0.5) {
            report.boundary_edges += 1;
        } else {
            report.unmatched.push((pa, pb, balance));
        }
    }
    report
}

fn assert_watertight(chunks: &[Meshed], base_m: f64, what: &str) -> Report {
    let report = watertight(chunks, base_m);
    assert!(report.triangles > 0, "{what}: no triangles");
    assert!(
        report.unmatched.is_empty(),
        "{what}: {} unmatched edges of {}, first {:?}",
        report.unmatched.len(),
        report.matched_edges,
        &report.unmatched[..report.unmatched.len().min(4)]
    );
    report
}

/// Signed volume enclosed by the drawn triangles (positive when wound outward).
fn volume_m3(chunks: &[Meshed]) -> f64 {
    let mut v = 0.0;
    for c in chunks {
        for [a, b, d] in c.triangles() {
            let (a, b, d) = (
                c.displaced_m[a as usize],
                c.displaced_m[b as usize],
                c.displaced_m[d as usize],
            );
            v += a.dot(b.cross(d)) / 6.0;
        }
    }
    v
}

/// Vertex identity across meshes: position in metres, normal and material words.
type VertexKey = ([u64; 3], u32, u32);

fn vertex_key(c: &Meshed, i: usize) -> VertexKey {
    let v: PackedVertex = c.mesh.vertices[i];
    (key_bits(c.displaced_m[i]), v.words[1] >> 16, v.words[2])
}

/// The full resolution vertices of each masked face equal the fine chunks' vertices on that face
/// bit for bit (position in metres, normal, materials), and every half resolution vertex is a
/// regular vertex where no coarse face sample disagrees with its fine sample.
/// Returns the number of masked faces the surface crosses.
fn assert_full_res_faces_match(coarse: &Meshed, fine: &[Meshed], base_m: f64, what: &str) -> usize {
    let mut crossed = 0;
    let (lo, hi) = coarse.bounds_m(base_m);
    let flips = coarse.flips();
    for face in Face::ALL {
        if coarse.mask & face.bit() == 0 {
            continue;
        }
        let axis = face.axis();
        let plane = if face.is_high() { hi[axis] } else { lo[axis] };
        let mut transition: BTreeSet<VertexKey> = BTreeSet::new();
        for &i in coarse.mesh.transition_indices(face) {
            let i = i as usize;
            let flagged = coarse.mesh.vertices[i].flags() & FLAG_TRANSITION_FULL_RES != 0;
            if flagged {
                assert_eq!(coarse.displaced_m[i][axis], plane, "{what}: off the face");
                transition.insert(vertex_key(coarse, i));
            } else if flips == 0 {
                assert!(
                    i < coarse.regular_vertex_count,
                    "{what}: half resolution vertex {i} is not a regular vertex"
                );
            }
        }
        let mut on_face: BTreeSet<VertexKey> = BTreeSet::new();
        for f in fine {
            let drawn: BTreeSet<u32> = f.triangles().into_iter().flatten().collect();
            for i in drawn.into_iter().map(|i| i as usize) {
                let p = f.displaced_m[i];
                let in_rect = (0..3).all(|a| a == axis || (p[a] >= lo[a] && p[a] <= hi[a]));
                if p[axis] == plane && in_rect {
                    on_face.insert(vertex_key(f, i));
                }
            }
        }
        if transition.is_empty() && on_face.is_empty() {
            // The surface does not cross this face.
            continue;
        }
        assert_eq!(
            transition.len(),
            on_face.len(),
            "{what}: {face:?} vertex counts differ"
        );
        assert!(
            transition == on_face,
            "{what}: {face:?} full resolution vertices differ from the fine face"
        );
        crossed += 1;
    }
    crossed
}

/// The level 1 chunk at the origin with `mask`, the fine shell for it, all filled from `field`.
fn coarse_scene(mask: u8, base_m: f64, field: &impl Field) -> (Meshed, Vec<Meshed>) {
    let coarse = Meshed::new(ChunkKey::at_level(1, 0, 0, 0), mask, base_m, field);
    let fine = fine_shell(mask)
        .into_iter()
        .map(|key| Meshed::new(key, 0, base_m, field))
        .collect();
    (coarse, fine)
}

fn masks() -> Vec<(u8, &'static str)> {
    let mut out: Vec<(u8, &'static str)> = vec![
        (Face::LowX.bit(), "LowX"),
        (Face::HighX.bit(), "HighX"),
        (Face::LowY.bit(), "LowY"),
        (Face::HighY.bit(), "HighY"),
        (Face::LowZ.bit(), "LowZ"),
        (Face::HighZ.bit(), "HighZ"),
    ];
    out.push((Face::LowX.bit() | Face::HighY.bit(), "LowX HighY"));
    out.push((Face::LowX.bit() | Face::HighX.bit(), "LowX HighX"));
    out.push((Face::HighZ.bit() | Face::LowY.bit(), "LowY HighZ"));
    out.push((
        Face::LowX.bit() | Face::LowY.bit() | Face::LowZ.bit(),
        "LowX LowY LowZ",
    ));
    out.push((
        Face::HighX.bit() | Face::HighY.bit() | Face::HighZ.bit(),
        "HighX HighY HighZ",
    ));
    out.push((0x3F, "all six"));
    out
}

/// Coarse units: the level 1 chunk at the origin spans 0 to 32 per axis.
fn coarse_units(p: DVec3, base_m: f64) -> DVec3 {
    p / (2.0 * base_m)
}

fn layered(height: f64) -> Material {
    if height > 20.0 {
        Material::GRASS
    } else if height > 12.0 {
        Material::SOIL
    } else {
        Material::ROCK
    }
}

/// Union of spheres (coarse units), materials layered by height.
fn spheres(base_m: f64, centres: Vec<DVec3>, radius: f64) -> impl Field {
    move |_, p| {
        let c = coarse_units(p, base_m);
        let d = centres
            .iter()
            .map(|centre| c.distance(*centre) - radius)
            .fold(f64::INFINITY, f64::min);
        (d * 2.0 * base_m, layered(c.y))
    }
}

/// Spheres closed inside the region of `mask`'s scene: one per masked face, at the meeting point
/// of masked faces where they share an edge or corner.
fn closed_spheres(mask: u8, base_m: f64) -> impl Field {
    let mut centre = DVec3::splat(16.0);
    let mut opposite = false;
    for axis in 0..3 {
        let low = mask & Face::ALL[2 * axis].bit() != 0;
        let high = mask & Face::ALL[2 * axis + 1].bit() != 0;
        match (low, high) {
            (true, false) => centre[axis] = 0.0,
            (false, true) => centre[axis] = 32.0,
            (true, true) => opposite = true,
            (false, false) => {}
        }
    }
    if mask == 0x3F {
        return spheres(base_m, vec![DVec3::splat(16.0)], 25.0);
    }
    if opposite {
        let centres = Face::ALL
            .iter()
            .filter(|f| mask & f.bit() != 0)
            .map(|f| {
                let mut c = DVec3::splat(16.0);
                c[f.axis()] = if f.is_high() { 32.0 } else { 0.0 };
                c
            })
            .collect();
        return spheres(base_m, centres, 10.0);
    }
    spheres(base_m, vec![centre], 10.0)
}

fn tilted_plane(base_m: f64, normal: DVec3) -> impl Field {
    let n = normal.normalize();
    move |_, p| {
        let c = coarse_units(p, base_m);
        ((c - DVec3::splat(16.3)).dot(n) * 2.0 * base_m, layered(c.y))
    }
}

fn sine_terrain(base_m: f64) -> impl Field {
    move |_, p| {
        let c = coarse_units(p, base_m);
        let h =
            16.0 + 7.0 * (c.x * 0.23).sin() * (c.z * 0.19 + 0.4).cos() + 2.0 * (c.x * 0.7).sin();
        let slope = 0.4 * (c.x - 16.0);
        ((c.y - h - slope) * 2.0 * base_m * 0.7, layered(c.y))
    }
}

/// A splitmix64 hash of the level 0 sample index to -1 to 1.
fn hash01(i: IVec3, salt: u64) -> f64 {
    let mut x = (i.x as u32 as u64)
        ^ ((i.y as u32 as u64) << 21)
        ^ ((i.z as u32 as u64) << 42)
        ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// A tilted plane plus white noise of up to two level 0 voxels, defined on the level 0 sample
/// index, so coarse samples are coincident points of the same field. Saddle face squares
/// (ambiguous faces) are frequent.
fn noisy_plane(base_m: f64, salt: u64) -> impl Field {
    move |_, p| {
        let c = coarse_units(p, base_m);
        let index = (p / base_m).round().as_ivec3();
        let d = (c - DVec3::splat(16.0)).dot(DVec3::new(0.3, 1.0, -0.2).normalize()) * 2.0;
        ((d + 2.0 * hash01(index, salt)) * base_m, layered(c.y))
    }
}

/// `field` at level 1, `field` plus a smooth perturbation at level 0, as per level height grids
/// produce.
fn perturbed(base_m: f64, field: impl Field, amplitude_voxels: f64) -> impl Field {
    move |level, p| {
        let (d, m) = field(level, p);
        if level == 0 {
            let c = coarse_units(p, base_m);
            let e = (c.x * 0.9 + 0.3).sin() * (c.y * 0.7).cos() * (c.z * 1.1 + 1.0).sin();
            (d + amplitude_voxels * base_m * e, m)
        } else {
            (d, m)
        }
    }
}

fn voxel_sizes() -> [f64; 4] {
    [0.25, 0.3, 1.0, 4.0]
}

#[test]
fn closed_spheres_through_every_mask_stay_closed() {
    for base_m in voxel_sizes() {
        for (mask, name) in masks() {
            let field = closed_spheres(mask, base_m);
            let (coarse, fine) = coarse_scene(mask, base_m, &field);
            let what = format!("sphere {name} at {base_m} m");
            let crossed = assert_full_res_faces_match(&coarse, &fine, base_m, &what);
            assert_eq!(
                crossed,
                mask.count_ones() as usize,
                "{what}: a face is not crossed"
            );
            let mut all = fine;
            all.push(coarse);
            let report = assert_watertight(&all, base_m, &what);
            assert_eq!(report.boundary_edges, 0, "{what}: the sphere is not closed");
            assert!(volume_m3(&all) > 0.0, "{what}: wound inward");
        }
    }
}

#[test]
fn planes_terrain_and_noise_stitch_for_every_mask() {
    let base_m = 1.0;
    for (mask, name) in masks() {
        let fields: Vec<(Box<dyn Field>, &str)> = vec![
            (
                Box::new(tilted_plane(base_m, DVec3::new(0.3, 1.0, 0.2))),
                "plane",
            ),
            (
                Box::new(tilted_plane(base_m, DVec3::new(1.0, -0.6, 0.45))),
                "steep plane",
            ),
            (
                Box::new(tilted_plane(base_m, DVec3::new(-0.5, 0.4, 1.0))),
                "plane z",
            ),
            (Box::new(sine_terrain(base_m)), "sine terrain"),
            (Box::new(noisy_plane(base_m, 1)), "noise"),
            (Box::new(noisy_plane(base_m, 2)), "noise 2"),
        ];
        for (field, field_name) in fields {
            let (coarse, fine) = coarse_scene(mask, base_m, &field);
            let what = format!("{field_name} {name}");
            assert_full_res_faces_match(&coarse, &fine, base_m, &what);
            let mut all = fine;
            all.push(coarse);
            assert_watertight(&all, base_m, &what);
        }
    }
}

/// A horizontal plane 0.02 m above a row of level 1 samples at 1 m: the level 1 copy of that row
/// quantises to the surface and is nudged outside, the level 0 copy stays inside, so every coarse
/// face corner on the row flips between the levels even though both sample one field.
#[test]
fn quantisation_flips_are_bridged() {
    let base_m = 1.0;
    let field = |_: u8, p: DVec3| (p.y - 20.02, Material::ROCK);
    for (mask, name) in masks() {
        let (coarse, fine) = coarse_scene(mask, base_m, &field);
        let what = format!("flip plane {name}");
        let vertical = (0..6).any(|f| mask & (1 << f) != 0 && f / 2 != 1);
        if vertical {
            assert!(coarse.flips() > 0, "{what}: no flips to bridge");
        }
        assert_full_res_faces_match(&coarse, &fine, base_m, &what);
        let mut all = fine;
        all.push(coarse);
        assert_watertight(&all, base_m, &what);
    }
}

#[test]
fn perturbed_fine_level_still_stitches() {
    let mut total_flips = 0;
    for base_m in [0.3, 1.0] {
        for (mask, name) in masks() {
            let fields: Vec<(Box<dyn Field>, &str)> = vec![
                (
                    Box::new(perturbed(base_m, closed_spheres(mask, base_m), 0.3)),
                    "sphere",
                ),
                (
                    Box::new(perturbed(base_m, sine_terrain(base_m), 0.3)),
                    "sine",
                ),
                (
                    Box::new(perturbed(base_m, noisy_plane(base_m, 3), 0.5)),
                    "noise",
                ),
            ];
            for (field, field_name) in fields {
                let (coarse, fine) = coarse_scene(mask, base_m, &field);
                total_flips += coarse.flips();
                let what = format!("perturbed {field_name} {name} at {base_m} m");
                assert_full_res_faces_match(&coarse, &fine, base_m, &what);
                let mut all = fine;
                all.push(coarse);
                assert_watertight(&all, base_m, &what);
            }
        }
    }
    println!("perturbed fields: {total_flips} coarse face samples flipped between the levels");
    assert!(
        total_flips > 0,
        "the perturbation never flipped a face sample"
    );
}

/// Displaced vertices of `c` lying on the plane `axis = plane_m`, as keys.
fn on_plane(c: &Meshed, axis: usize, plane_m: f64) -> BTreeSet<VertexKey> {
    let used: BTreeSet<u32> = c.triangles().into_iter().flatten().collect();
    used.into_iter()
        .map(|i| i as usize)
        .filter(|&i| c.displaced_m[i][axis] == plane_m)
        .map(|i| vertex_key(c, i))
        .collect()
}

/// Two level 1 chunks side by side along x, both above level 0 chunks (mask LowY): their shared
/// face stays bit identical after displacement and the union stays watertight.
#[test]
fn same_level_pair_over_a_finer_level_shares_its_face() {
    for base_m in voxel_sizes() {
        let fields: Vec<(Box<dyn Field>, &str)> = vec![
            (
                Box::new(spheres(base_m, vec![DVec3::new(32.0, 0.0, 16.0)], 10.0)),
                "sphere",
            ),
            (Box::new(sine_terrain(base_m)), "sine"),
            (Box::new(noisy_plane(base_m, 4)), "noise"),
            (
                Box::new(perturbed(base_m, noisy_plane(base_m, 5), 0.5)),
                "perturbed noise",
            ),
        ];
        for (field, name) in fields {
            let mask = Face::LowY.bit();
            let a = Meshed::new(ChunkKey::at_level(1, 0, 0, 0), mask, base_m, &field);
            let b = Meshed::new(ChunkKey::at_level(1, 1, 0, 0), mask, base_m, &field);
            let plane_m = a.bounds_m(base_m).1.x;
            let (on_a, on_b) = (on_plane(&a, 0, plane_m), on_plane(&b, 0, plane_m));
            let what = format!("pair {name} at {base_m} m");
            assert!(!on_a.is_empty(), "{what}: nothing on the shared face");
            assert!(
                on_a == on_b,
                "{what}: shared face differs after displacement"
            );
            let mut all = vec![a, b];
            for z in 0..2 {
                for x in 0..4 {
                    all.push(Meshed::new(ChunkKey::new(x, -1, z), 0, base_m, &field));
                }
            }
            let report = assert_watertight(&all, base_m, &what);
            if name == "sphere" {
                assert_eq!(report.boundary_edges, 0, "{what}: the sphere is not closed");
            }
        }
    }
}

/// The line where a transition face meets a same level face: level 1 chunk A with level 0 chunks
/// below it (mask LowY), level 1 chunk B beside A (no mask), and level 1 chunk D below B (mask
/// LowX, its low x neighbours are the level 0 chunks). A, B, D and the level 0 chunks meet on the
/// line x = 32, y = 0 (coarse units). Measures the gap along that line: the largest distance from
/// an unmatched edge endpoint to the nearest vertex of another chunk (zero when every edge
/// matches), and asserts the bound (zero).
#[test]
fn transition_meets_same_level_face_without_a_gap() {
    let mut worst_gap_m: f64 = 0.0;
    let mut sliver_edges = 0;
    for base_m in voxel_sizes() {
        let fields: Vec<(Box<dyn Field>, &str)> = vec![
            (
                Box::new(spheres(base_m, vec![DVec3::new(32.0, 0.0, 16.0)], 10.0)),
                "sphere",
            ),
            (
                Box::new(tilted_plane(base_m, DVec3::new(-1.0, 1.0, 0.3))),
                "plane",
            ),
            (Box::new(sine_terrain(base_m)), "sine"),
            (Box::new(noisy_plane(base_m, 6)), "noise"),
            (
                Box::new(perturbed(base_m, noisy_plane(base_m, 7), 0.5)),
                "perturbed noise",
            ),
            (
                // Coarse samples on the plane z = 20 (coarse units) flip against the fine ones on
                // both A's and D's transition faces, so bridges of two chunks on perpendicular
                // faces meet on the line.
                Box::new(move |_: u8, p: DVec3| (p.z - 40.02 * base_m, Material::ROCK)),
                "flip plane",
            ),
        ];
        for (field, name) in fields {
            let mut all = vec![
                Meshed::new(
                    ChunkKey::at_level(1, 0, 0, 0),
                    Face::LowY.bit(),
                    base_m,
                    &field,
                ),
                Meshed::new(ChunkKey::at_level(1, 1, 0, 0), 0, base_m, &field),
                Meshed::new(
                    ChunkKey::at_level(1, 1, -1, 0),
                    Face::LowX.bit(),
                    base_m,
                    &field,
                ),
            ];
            for z in 0..2 {
                for y in -2..0 {
                    for x in 0..2 {
                        all.push(Meshed::new(ChunkKey::new(x, y, z), 0, base_m, &field));
                    }
                }
            }
            let what = format!("line {name} at {base_m} m");
            if name == "flip plane" {
                assert!(all[0].flips() > 0 && all[2].flips() > 0, "{what}: no flips");
            }
            let report = watertight(&all, base_m);
            let line_x = 64.0 * base_m;
            for &(pa, pb, _) in &report.unmatched {
                for p in [pa, pb] {
                    let nearest = all
                        .iter()
                        .flat_map(|c| c.displaced_m.iter())
                        .filter(|q| **q != pa && **q != pb)
                        .map(|q| q.distance(p))
                        .fold(f64::INFINITY, f64::min);
                    worst_gap_m = worst_gap_m.max(nearest);
                }
            }
            // Edges lying on the line itself are the degenerate transition slivers; they must match.
            for c in &all {
                for t in c.triangles() {
                    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        let (pa, pb) = (c.displaced_m[a as usize], c.displaced_m[b as usize]);
                        if pa != pb && [pa, pb].iter().all(|p| p.x == line_x && p.y == 0.0) {
                            sliver_edges += 1;
                        }
                    }
                }
            }
            assert!(
                report.unmatched.is_empty(),
                "{what}: {} unmatched edges, first {:?}",
                report.unmatched.len(),
                &report.unmatched[..report.unmatched.len().min(4)]
            );
            if name == "sphere" {
                assert_eq!(report.boundary_edges, 0, "{what}: the sphere is not closed");
            }
        }
    }
    println!(
        "transition and same level line: gap {worst_gap_m} m, {sliver_edges} edges along the line"
    );
    assert_eq!(worst_gap_m, 0.0);
}

/// The closed sphere scenes at levels 3 and 2, about 23 km from the frame origin at 0.3 m voxels:
/// positions stay bit equal in metres at the levels and distances the planet uses.
#[test]
fn deeper_levels_far_from_the_origin_stay_closed() {
    let base_m = 0.3;
    let coarse_key = ChunkKey::at_level(3, 300, -301, 77);
    let voxel_m = coarse_key.voxel_size_m(base_m);
    let origin_m = coarse_key.bounds_m(base_m).0;
    for (mask, name) in masks() {
        let local = closed_spheres(mask, 1.0);
        // The scene's fields in coarse units of the level 1 scene at 1 m map onto this chunk.
        let field = move |level: u8, p: DVec3| {
            let c = (p - origin_m) / voxel_m * 2.0;
            let (d, m) = local(level, c);
            (d * voxel_m * 0.5, m)
        };
        let coarse = Meshed::new(coarse_key, mask, base_m, &field);
        let fine: Vec<Meshed> = fine_shell(mask)
            .into_iter()
            .map(|k| {
                let key = ChunkKey::at_level(
                    2,
                    2 * coarse_key.x + k.x,
                    2 * coarse_key.y + k.y,
                    2 * coarse_key.z + k.z,
                );
                Meshed::new(key, 0, base_m, &field)
            })
            .collect();
        let what = format!("level 3 sphere {name}");
        let crossed = assert_full_res_faces_match(&coarse, &fine, base_m, &what);
        assert_eq!(
            crossed,
            mask.count_ones() as usize,
            "{what}: a face is not crossed"
        );
        let mut all = fine;
        all.push(coarse);
        let report = assert_watertight(&all, base_m, &what);
        assert_eq!(report.boundary_edges, 0, "{what}: the sphere is not closed");
    }
}

/// Meshing time with and without all six transition faces on a level 1 surface chunk. Read it
/// with `cargo test -p vx_voxel --release --test transition_seams -- --ignored --nocapture`.
#[test]
#[ignore = "timing, run in release"]
fn transition_meshing_timing() {
    let runs = if cfg!(debug_assertions) { 5 } else { 200 };
    let base_m = 1.0;
    let key = ChunkKey::at_level(1, 0, 0, 0);
    for (name, field) in [
        (
            "sine terrain",
            Box::new(sine_terrain(base_m)) as Box<dyn Field>,
        ),
        ("noisy plane", Box::new(noisy_plane(base_m, 8))),
        (
            "perturbed sine",
            Box::new(perturbed(base_m, sine_terrain(base_m), 0.3)),
        ),
    ] {
        let mut chunk = Chunk::new(key, base_m);
        chunk.fill_from_with_material(|p| field(1, p));
        let slabs = Face::ALL.map(|face| {
            transition_slab(key, face, base_m).map(|mut slab| {
                slab.fill_from_with_material(|p| field(0, p));
                slab
            })
        });
        let time = |f: &dyn Fn() -> PackedMesh| {
            let mut times = Vec::with_capacity(runs);
            let mut mesh = None;
            for _ in 0..runs {
                let start = Instant::now();
                mesh = Some(std::hint::black_box(f()));
                times.push(start.elapsed().as_secs_f64() * 1.0e3);
            }
            times.sort_by(f64::total_cmp);
            (
                times[runs / 2],
                times[runs * 95 / 100],
                mesh.expect("one run"),
            )
        };
        let (regular_ms, regular_p95, regular) = time(&|| mesh_chunk(std::hint::black_box(&chunk)));
        let (lod_ms, lod_p95, lod) = time(&|| mesh_chunk_lod(std::hint::black_box(&chunk), &slabs));
        let transition_triangles = (lod.indices.len() - lod.regular_index_count as usize) / 3;
        println!(
            "mesh {name}: regular {regular_ms:.3} ms median ({regular_p95:.3} p95), with six transition faces {lod_ms:.3} ms median ({lod_p95:.3} p95); {} regular triangles, {transition_triangles} transition triangles, {} vertices; {runs} runs, one thread, {base_m} m, {}",
            regular.triangle_count(),
            lod.vertices.len(),
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
    }
}

#[test]
fn transition_vertices_are_finite_and_on_the_face() {
    let base_m = 1.0;
    let noise = noisy_plane(base_m, 9);
    let field = |level, p: DVec3| {
        let (d, m) = closed_spheres(0x3F, base_m)(level, p);
        (d + 0.5 * noise(level, p).0.signum() * base_m, m)
    };
    let (coarse, _) = coarse_scene(0x3F, base_m, &field);
    let mesh = &coarse.mesh;
    let regular = coarse.regular_vertex_count;
    for face in Face::ALL {
        let axis = face.axis();
        let plane = if face.is_high() { 32.0 } else { 0.0 };
        for &i in mesh.transition_indices(face) {
            let v = mesh.vertices[i as usize];
            let a = v.unpack();
            assert!(a.position_voxels.is_finite() && a.normal.is_finite());
            assert!((a.normal.length() - 1.0).abs() < 1.0e-4);
            assert_eq!(
                a.position_voxels[axis], plane,
                "{face:?} vertex off the face"
            );
            if (i as usize) >= regular {
                assert_eq!(v.chunk_slot(), 0);
            }
        }
    }
    let total: u32 = mesh.regular_index_count + mesh.transition_index_counts.iter().sum::<u32>();
    assert_eq!(total as usize, mesh.indices.len());
    assert!(mesh.transition_index_counts.iter().all(|&n| n > 0));
}
