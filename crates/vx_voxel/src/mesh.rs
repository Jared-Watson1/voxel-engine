use std::cmp::Reverse;

use glam::{IVec3, UVec3, Vec3};
use vx_core::mesh::{FLAG_TRANSITION_FULL_RES, MATERIAL_SLOTS, MAX_MATERIAL_WEIGHT};
use vx_core::{CHUNK_SIZE, PackedVertex, VertexAttributes};

use crate::block::SampleBlock;
use crate::chunk::{Chunk, Material, SAMPLES_PER_AXIS, Voxel};
use crate::key::{ChunkKey, Face};
use crate::transvoxel::{
    CellSink, FACE_FRAMES, FaceFrame, regular_cell, transition_case, transition_cell,
};

/// Packed, indexed triangle mesh of one chunk (D14): positions quantised in chunk local voxel
/// units, octahedral normals, material slots. Indices are local to the mesh (0 based). Counter
/// clockwise front faces seen from outside the solid.
///
/// Indices are laid out as the regular cells first (`regular_index_count`), then the transition
/// cells of each face in [`crate::Face`] order (`transition_index_counts`, D22); the renderer draws
/// a face's range only while that face's mask bit is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedMesh {
    pub key: ChunkKey,
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
    pub regular_index_count: u32,
    pub transition_index_counts: [u32; 6],
}

impl PackedMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn vertex_bytes(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<PackedVertex>()
    }

    pub fn index_bytes(&self) -> usize {
        self.indices.len() * std::mem::size_of::<u32>()
    }

    /// The regular cells' indices.
    pub fn regular_indices(&self) -> &[u32] {
        &self.indices[..self.regular_index_count as usize]
    }

    /// The transition cells' indices of `face` (empty when the face had no slab).
    pub fn transition_indices(&self, face: Face) -> &[u32] {
        let i = face as usize;
        let start = self.regular_index_count as usize
            + self.transition_index_counts[..i]
                .iter()
                .map(|&n| n as usize)
                .sum::<usize>();
        &self.indices[start..start + self.transition_index_counts[i] as usize]
    }
}

/// Meshes the chunk's `CHUNK_SIZE`^3 cells with Transvoxel regular cells into packed vertices;
/// equal to [`mesh_chunk_lod`] without slabs.
///
/// Vertex positions, normals and material slots depend only on the edge that produced the vertex
/// and the solid endpoint's face neighbours, never on the owning cell, so a vertex on a chunk face
/// is identical in both chunks that share the face (D14, D22, engineering rule 6). Each crossing
/// edge has one vertex. The chunk slot word is left zero for the renderer to patch.
///
/// Per vertex, with `a` and `b` the edge endpoints, the lower coordinate first, and `t` the
/// crossing of the quantised distance between them:
/// - position: `a + (b - a) * t` in voxel units, quantised by [`PackedVertex::pack`]. A vertex on
///   an edge lying in a face plane of the chunk is first rounded to 1/512 voxel (half the packed
///   resolution), so a coarser chunk's transition cells can hold the same position at its own
///   packed resolution (D22);
/// - normal: the central difference gradients of the two endpoints lerped by `t`, negated to point
///   out of the solid and normalised;
/// - material slot 0: the material of the solid endpoint with weight 15. Slots 1 to 3: the other
///   materials among the solid endpoint's six face neighbours that are inside the solid, each with
///   weight `round(15 * count / 6)` (3, 5, 8, 10, 13, 15 for one to six neighbours), in
///   descending count order with ties broken by ascending material id. Unused slots are `(0, 0)`
///   and neighbours carrying [`Material::AIR`] are ignored.
///
/// Two chunks sharing a face hold the same samples around it (D16), so `t`, the gradients and
/// the material rule see identical inputs on both sides.
pub fn mesh_chunk(chunk: &Chunk) -> PackedMesh {
    mesh_chunk_lod(chunk, &NO_SLABS)
}

const NO_SLABS: [Option<SampleBlock>; 6] = [None, None, None, None, None, None];

/// Meshes the chunk's regular cells unshrunk plus, for every face whose slab is given, that
/// face's transition cells from the slab's fine samples into their own index range (D22). Slabs
/// come from [`crate::transition_slab`] filled by the sampler and the edit log; a chunk at level 1
/// or above is meshed with all six so a mask change never needs a remesh.
///
/// The regular range (vertices and indices) equals [`mesh_chunk`] whatever the slabs; transition
/// vertices follow the regular ones. A face's range holds:
/// - Lengyel's transition cells between the slab's face samples (the full resolution face) and
///   the coarse face. Their full resolution vertices equal the finer chunk's face vertices bit
///   for bit (same endpoints, interpolation, gradients at fine spacing and material rule) and
///   carry [`vx_core::mesh::FLAG_TRANSITION_FULL_RES`]. Their half resolution vertices are the
///   regular vertices on the same coarse edges (shared indices), so the shader's displacement
///   moves both alike.
/// - Bridge cells where a coarse face sample and the fine sample at the same point lie on opposite
///   sides of the surface (the levels' fields differ, or quantisation rounds a near zero distance
///   differently at the two spacings). Lengyel's half resolution face takes the fine samples'
///   inside states; a zero thickness regular cell between those states and the coarse samples
///   joins it to the regular mesh. Where all four corners of a coarse face square agree, the
///   bridge is empty and Lengyel's half resolution vertices are the regular vertices.
///
/// Every transition vertex starts on the face plane; the vertex shader moves the unflagged ones
/// (`vx_core::mesh::secondary_position_voxels`).
///
/// # Panics
///
/// If a slab is given for a level 0 chunk, or a slab's level, position or size differs from
/// [`crate::transition_slab`] for that face.
pub fn mesh_chunk_lod(chunk: &Chunk, slabs: &[Option<SampleBlock>; 6]) -> PackedMesh {
    profiling::scope!("mesh_chunk_lod");
    let mut mesher = Mesher::new(chunk);
    mesher.regular_cells();
    let regular_index_count = mesher.indices.len() as u32;
    let mut transition_index_counts = [0u32; 6];
    for face in Face::ALL {
        if let Some(slab) = &slabs[face as usize] {
            check_slab(chunk, face, slab);
            let start = mesher.indices.len();
            mesher.transition_face(face, slab);
            transition_index_counts[face as usize] = (mesher.indices.len() - start) as u32;
        }
    }
    PackedMesh {
        key: chunk.key(),
        vertices: mesher.vertices,
        indices: mesher.indices,
        regular_index_count,
        transition_index_counts,
    }
}

fn check_slab(chunk: &Chunk, face: Face, slab: &SampleBlock) {
    let key = chunk.key();
    let expected = crate::block::transition_slab(key, face, 1.0)
        .unwrap_or_else(|| panic!("level 0 chunk {key:?} has no transition slabs"));
    assert!(
        slab.level() == expected.level()
            && slab.min() == expected.min()
            && slab.dims() == expected.dims()
            && slab.voxel_size_m() * 2.0 == chunk.voxel_size_m(),
        "slab for {face:?} of {key:?} does not match transition_slab"
    );
}

const N: i32 = CHUNK_SIZE as i32;
/// Fine samples per chunk edge on a slab's face.
const FINE: i32 = 2 * N;
const NO_VERTEX: u32 = u32::MAX;
/// Grid points per axis of the chunk's cell region.
const POINTS: usize = CHUNK_SIZE + 1;
const FINE_POINTS: usize = 2 * CHUNK_SIZE + 1;

/// One end of an edge as the vertex formula sees it.
#[derive(Clone, Copy)]
struct End {
    position: [f32; 3],
    density: f32,
    gradient: [f32; 3],
}

/// The crate's `Density::interpolate` for f32 at threshold 0.
fn crossing(a: f32, b: f32) -> f32 {
    if (b - a).abs() > f32::EPSILON {
        (0.0 - a) / (b - a)
    } else {
        0.5
    }
}

fn lerp(x: f32, y: f32, t: f32) -> f32 {
    x + (y - x) * t
}

/// Rounds to 1/512 voxel. `t` lies in [1/128, 127/128] for quantised distances (never zero, at
/// most 127 units), so a rounded position never reaches a grid point.
fn snap(v: f32) -> f32 {
    (v * 512.0).round() / 512.0
}

/// Position and normal of the vertex between `a` and `b` (the crate's `add_vertex_between` and
/// `gradients_to_normal` in f32, operation for operation).
fn edge_vertex(a: &End, b: &End, on_face: bool) -> (Vec3, Vec3) {
    let t = crossing(a.density, b.density);
    let mut p: [f32; 3] = std::array::from_fn(|i| lerp(a.position[i], b.position[i], t));
    if on_face {
        p = p.map(snap);
    }
    let [gx, gy, gz]: [f32; 3] = std::array::from_fn(|i| lerp(a.gradient[i], b.gradient[i], t));
    let norm = (gx * gx + gy * gy + gz * gz).sqrt();
    let normal = if norm > f32::EPSILON {
        Vec3::new(-gx / norm, -gy / norm, -gz / norm)
    } else {
        Vec3::ZERO
    };
    (Vec3::from_array(p), normal)
}

fn density(voxel: Voxel) -> f32 {
    -voxel.distance_voxels()
}

/// Edge between two grid points one step apart: its axis and lower point.
fn edge(a: IVec3, b: IVec3) -> (usize, IVec3) {
    let d = b - a;
    let axis = (0..3).find(|&i| d[i] != 0).unwrap_or(0);
    (axis, a.min(b))
}

struct Mesher<'a> {
    chunk: &'a Chunk,
    vertices: Vec<PackedVertex>,
    indices: Vec<u32>,
    /// Vertex per crossing edge of the cell region, by axis and lower point.
    regular_edges: Vec<u32>,
    /// Per face scratch, reset for each transition face.
    full_res_edges: Vec<u32>,
    virtual_edges: Vec<u32>,
    bridge_edges: Vec<u32>,
}

impl<'a> Mesher<'a> {
    fn new(chunk: &'a Chunk) -> Self {
        Self {
            chunk,
            vertices: Vec::new(),
            indices: Vec::new(),
            regular_edges: vec![NO_VERTEX; 3 * POINTS * POINTS * POINTS],
            full_res_edges: Vec::new(),
            virtual_edges: Vec::new(),
            bridge_edges: Vec::new(),
        }
    }

    /// Coarse sample at chunk local `p` (-1 to `CHUNK_SIZE + 1`).
    fn coarse(&self, p: IVec3) -> Voxel {
        let s = (p + IVec3::ONE).as_uvec3();
        debug_assert!(s.cmplt(UVec3::splat(SAMPLES_PER_AXIS as u32)).all());
        self.chunk.samples()[Chunk::linear(s.x as usize, s.y as usize, s.z as usize)]
    }

    fn coarse_end(&self, p: IVec3) -> End {
        End {
            position: p.as_vec3().to_array(),
            density: density(self.coarse(p)),
            gradient: gradient(|q| self.coarse(q), p),
        }
    }

    fn push(&mut self, position_voxels: Vec3, normal: Vec3, materials: Materials) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(PackedVertex::pack(&VertexAttributes {
            position_voxels,
            normal,
            materials,
        }));
        index
    }

    fn regular_cells(&mut self) {
        // Cell order of the crate's `extract_regular_cells` (x outermost), which with first come
        // vertex creation reproduces the vertex order of the M1 mesher.
        for x in 0..N {
            for y in 0..N {
                for z in 0..N {
                    let cell = IVec3::new(x, y, z);
                    let mut case = 0u8;
                    for corner in 0..8 {
                        if self.coarse(cell + corner_offset(corner)).is_inside() {
                            case |= 1 << corner;
                        }
                    }
                    if case != 0 && case != 0xFF {
                        regular_cell(case, &mut RegularSink { mesher: self, cell });
                    }
                }
            }
        }
    }

    fn regular_edge_slot(axis: usize, low: IVec3) -> usize {
        let l = low.as_uvec3();
        ((axis * POINTS + l.z as usize) * POINTS + l.y as usize) * POINTS + l.x as usize
    }

    /// The vertex on the crossing edge between grid points `a` and `b` of the cell region.
    fn regular_vertex(&mut self, a: IVec3, b: IVec3) -> u32 {
        let (axis, low) = edge(a, b);
        let slot = Self::regular_edge_slot(axis, low);
        if self.regular_edges[slot] != NO_VERTEX {
            return self.regular_edges[slot];
        }
        let high = low + unit(axis);
        let on_face = (0..3).any(|i| i != axis && (low[i] == 0 || low[i] == N));
        let (position, normal) =
            edge_vertex(&self.coarse_end(low), &self.coarse_end(high), on_face);
        let solid = if self.coarse(low).is_inside() {
            low
        } else {
            high
        };
        let materials = material_slots(|q| self.coarse(q), solid);
        let index = self.push(position, normal, materials);
        self.regular_edges[slot] = index;
        index
    }

    /// The regular vertex on a coarse face edge whose coarse samples straddle the surface: the
    /// regular cells made it, so the regular range does not grow during the transition pass.
    fn existing_regular_vertex(&mut self, a: IVec3, b: IVec3) -> u32 {
        let (axis, low) = edge(a, b);
        debug_assert!(
            self.regular_edges[Self::regular_edge_slot(axis, low)] != NO_VERTEX,
            "a transition needs a regular vertex the regular cells did not make"
        );
        self.regular_vertex(a, b)
    }

    fn transition_face(&mut self, face: Face, slab: &SampleBlock) {
        let fine = FineSlab::new(self.chunk.key(), face, slab);
        let frame = FACE_FRAMES[face as usize];
        reset(&mut self.full_res_edges, 2 * FINE_POINTS * FINE_POINTS);
        reset(&mut self.virtual_edges, 2 * POINTS * POINTS);
        reset(&mut self.bridge_edges, POINTS * POINTS);
        for cv in 0..N {
            for cu in 0..N {
                let inside: [bool; 9] = std::array::from_fn(|g| {
                    fine.voxel(full_res_point(&frame, cu, cv, g)).is_inside()
                });
                let case = transition_case(inside);
                if case != 0 && case != 0x1FF {
                    let mut sink = TransitionSink {
                        mesher: self,
                        fine: &fine,
                        frame,
                        cu,
                        cv,
                    };
                    transition_cell(case, &mut sink);
                }
                let mut bridge = 0u8;
                let mut disagree = false;
                for corner in 0..4 {
                    let p = coarse_point(&frame, cu, cv, corner);
                    let fine_inside = fine.voxel(p * 2).is_inside();
                    let coarse_inside = self.coarse(p).is_inside();
                    disagree |= fine_inside != coarse_inside;
                    bridge |= u8::from(fine_inside) << corner;
                    bridge |= u8::from(coarse_inside) << (corner + 4);
                }
                if disagree {
                    let mut sink = BridgeSink {
                        mesher: self,
                        fine: &fine,
                        frame,
                        cu,
                        cv,
                    };
                    regular_cell(bridge, &mut sink);
                }
            }
        }
    }

    /// The full resolution vertex on the slab's face between fine points `a` and `b` (chunk local
    /// in fine units), computed in the frame of the finer chunk that holds the edge so it matches
    /// that chunk's face vertex bit for bit, then halved into this chunk's units.
    fn full_res_vertex(&mut self, fine: &FineSlab, a: IVec3, b: IVec3) -> u32 {
        let (axis, low) = edge(a, b);
        let [p, q] = fine.plane_axes;
        let direction = usize::from(axis == q);
        let slot = (direction * FINE_POINTS + low[q] as usize) * FINE_POINTS + low[p] as usize;
        if self.full_res_edges[slot] != NO_VERTEX {
            return self.full_res_edges[slot];
        }
        let high = low + unit(axis);
        // Chunk local coordinate in the finer chunk along the edge axis; the other coordinates are
        // integers, exact in any frame.
        let offset = low[axis].div_euclid(N) * N;
        let end = |point: IVec3| {
            let mut position = point.as_vec3();
            position[axis] -= offset as f32;
            End {
                position: position.to_array(),
                density: density(fine.voxel(point)),
                gradient: gradient(|r| fine.voxel(r), point),
            }
        };
        let (mut position, normal) = edge_vertex(&end(low), &end(high), true);
        position[axis] += offset as f32;
        let position = position * 0.5;
        let solid = if fine.voxel(low).is_inside() {
            low
        } else {
            high
        };
        let materials = material_slots(|r| fine.voxel(r), solid);
        let index = self.push(position, normal, materials);
        let vertex = &mut self.vertices[index as usize];
        *vertex = vertex.with_flags(FLAG_TRANSITION_FULL_RES);
        self.full_res_edges[slot] = index;
        index
    }

    /// The vertex on coarse face edge `a`-`b` of Lengyel's half resolution face, whose samples take
    /// the inside state of the fine samples at the same points. Where both ends agree with the
    /// coarse samples it is the regular vertex on that edge; otherwise the fine distances place it.
    fn virtual_vertex(&mut self, fine: &FineSlab, a: IVec3, b: IVec3) -> u32 {
        let (axis, low) = edge(a, b);
        let high = low + unit(axis);
        let agrees = |p: IVec3| fine.voxel(p * 2).is_inside() == self.coarse(p).is_inside();
        if agrees(low) && agrees(high) {
            return self.existing_regular_vertex(low, high);
        }
        let [p, q] = fine.plane_axes;
        let direction = usize::from(axis == q);
        let slot = (direction * POINTS + low[q] as usize) * POINTS + low[p] as usize;
        if self.virtual_edges[slot] != NO_VERTEX {
            return self.virtual_edges[slot];
        }
        let end = |point: IVec3| End {
            density: density(fine.voxel(point * 2)),
            ..self.coarse_end(point)
        };
        let (position, normal) = edge_vertex(&end(low), &end(high), true);
        let solid = if fine.voxel(low * 2).is_inside() {
            low
        } else {
            high
        };
        let materials = material_slots(|r| fine.voxel(r), solid * 2);
        let index = self.push(position, normal, materials);
        self.virtual_edges[slot] = index;
        index
    }

    /// The vertex at coarse face point `p` between the fine sample there (outside the bridge) and
    /// the coarse sample (inside), which lie on opposite sides of the surface. Distances compare
    /// in coarse voxels.
    fn bridge_vertex(&mut self, fine: &FineSlab, p: IVec3) -> u32 {
        let [a, b] = fine.plane_axes;
        let slot = p[b] as usize * POINTS + p[a] as usize;
        if self.bridge_edges[slot] != NO_VERTEX {
            return self.bridge_edges[slot];
        }
        let fine_voxel = fine.voxel(p * 2);
        let outer = End {
            position: p.as_vec3().to_array(),
            density: density(fine_voxel) * 0.5,
            gradient: gradient(|r| fine.voxel(r), p * 2),
        };
        let (position, normal) = edge_vertex(&outer, &self.coarse_end(p), true);
        let materials = if fine_voxel.is_inside() {
            material_slots(|r| fine.voxel(r), p * 2)
        } else {
            material_slots(|r| self.coarse(r), p)
        };
        let index = self.push(position, normal, materials);
        self.bridge_edges[slot] = index;
        index
    }
}

fn reset(table: &mut Vec<u32>, len: usize) {
    table.clear();
    table.resize(len, NO_VERTEX);
}

fn unit(axis: usize) -> IVec3 {
    let mut v = IVec3::ZERO;
    v[axis] = 1;
    v
}

fn corner_offset(corner: usize) -> IVec3 {
    IVec3::new(
        (corner & 1) as i32,
        ((corner >> 1) & 1) as i32,
        ((corner >> 2) & 1) as i32,
    )
}

/// Central difference gradient of the density at `p` (the crate's `regular_voxel_gradient`).
fn gradient(voxel: impl Fn(IVec3) -> Voxel, p: IVec3) -> [f32; 3] {
    std::array::from_fn(|i| density(voxel(p + unit(i))) - density(voxel(p - unit(i))))
}

/// A transition slab addressed in the chunk's fine local coordinates: the chunk spans 0 to
/// `2 * CHUNK_SIZE` and the slab holds the face plane, one sample on each side of it and one
/// sample beyond the face's edges.
struct FineSlab<'a> {
    slab: &'a SampleBlock,
    /// World fine sample index of the chunk's fine local origin minus the slab's first sample.
    origin: IVec3,
    /// The two in plane axes in increasing order.
    plane_axes: [usize; 2],
}

impl<'a> FineSlab<'a> {
    fn new(key: ChunkKey, face: Face, slab: &'a SampleBlock) -> Self {
        Self {
            slab,
            origin: key.as_ivec3() * FINE - slab.min(),
            plane_axes: match face.axis() {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            },
        }
    }

    /// The slab sample at chunk local fine coordinates `p`; the extraction only reads points the
    /// slab holds (validated by `check_slab`).
    fn voxel(&self, p: IVec3) -> Voxel {
        let l = (p + self.origin).as_uvec3();
        let dims = self.slab.dims();
        debug_assert!(l.cmplt(dims).all(), "fine sample {p} outside the slab");
        self.slab.voxels()[(l.x + dims.x * (l.y + dims.y * l.z)) as usize]
    }
}

struct RegularSink<'m, 'a> {
    mesher: &'m mut Mesher<'a>,
    cell: IVec3,
}

impl CellSink for RegularSink<'_, '_> {
    fn vertex(&mut self, a: usize, b: usize) -> u32 {
        self.mesher
            .regular_vertex(self.cell + corner_offset(a), self.cell + corner_offset(b))
    }

    fn triangle(&mut self, indices: [u32; 3]) {
        self.mesher.indices.extend(indices);
    }
}

/// Transition cell `(cu, cv)` of one face: grid points 0 to 8 on the slab's face, 9 to 12 on the
/// chunk's face.
struct TransitionSink<'m, 'a, 's> {
    mesher: &'m mut Mesher<'a>,
    fine: &'s FineSlab<'s>,
    frame: FaceFrame,
    cu: i32,
    cv: i32,
}

/// Fine chunk local point of grid point `g` (0 to 8) of transition cell `(cu, cv)`.
fn full_res_point(frame: &FaceFrame, cu: i32, cv: i32, g: usize) -> IVec3 {
    let (du, dv) = ((g % 3) as i32, (g / 3) as i32);
    IVec3::from_array(frame.point(FINE, 2 * cu + du, 2 * cv + dv))
}

/// Chunk local point of corner `corner` (bit 0 along u, bit 1 along v) of face square `(cu, cv)`.
fn coarse_point(frame: &FaceFrame, cu: i32, cv: i32, corner: usize) -> IVec3 {
    let (du, dv) = ((corner & 1) as i32, ((corner >> 1) & 1) as i32);
    IVec3::from_array(frame.point(N, cu + du, cv + dv))
}

impl CellSink for TransitionSink<'_, '_, '_> {
    fn vertex(&mut self, a: usize, b: usize) -> u32 {
        if a < 9 {
            debug_assert!(b < 9);
            let (pa, pb) = (
                full_res_point(&self.frame, self.cu, self.cv, a),
                full_res_point(&self.frame, self.cu, self.cv, b),
            );
            self.mesher.full_res_vertex(self.fine, pa, pb)
        } else {
            debug_assert!(b >= 9);
            let (pa, pb) = (
                coarse_point(&self.frame, self.cu, self.cv, a - 9),
                coarse_point(&self.frame, self.cu, self.cv, b - 9),
            );
            self.mesher.virtual_vertex(self.fine, pa, pb)
        }
    }

    fn triangle(&mut self, indices: [u32; 3]) {
        self.mesher.indices.extend(indices);
    }
}

/// Bridge cell `(cu, cv)` of one face: a regular cell in the face's `(u, v, w)` frame (right
/// handed like x, y, z, so the regular winding holds) whose `w = 0` corners carry the fine
/// samples' inside states (Lengyel's half resolution face) and whose `w = 1` corners carry the
/// coarse samples at the same points (the regular mesh's face). It has zero thickness before the
/// shader's displacement.
struct BridgeSink<'m, 'a, 's> {
    mesher: &'m mut Mesher<'a>,
    fine: &'s FineSlab<'s>,
    frame: FaceFrame,
    cu: i32,
    cv: i32,
}

impl CellSink for BridgeSink<'_, '_, '_> {
    fn vertex(&mut self, a: usize, b: usize) -> u32 {
        let (pa, pb) = (
            coarse_point(&self.frame, self.cu, self.cv, a & 3),
            coarse_point(&self.frame, self.cu, self.cv, b & 3),
        );
        match (a >> 2, b >> 2) {
            (0, 0) => self.mesher.virtual_vertex(self.fine, pa, pb),
            (1, 1) => self.mesher.existing_regular_vertex(pa, pb),
            _ => self.mesher.bridge_vertex(self.fine, pa),
        }
    }

    fn triangle(&mut self, indices: [u32; 3]) {
        self.mesher.indices.extend(indices);
    }
}

type Materials = [(u8, u8); MATERIAL_SLOTS];

const FACE_NEIGHBOURS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// The material rule of [`mesh_chunk`] for the solid endpoint `solid` of an edge, reading samples
/// through `voxel` (the chunk, or a slab at its own spacing). The endpoint's face neighbours are
/// within the stored apron (D16) or the slab.
fn material_slots(voxel: impl Fn(IVec3) -> Voxel, solid: IVec3) -> Materials {
    let dominant = voxel(solid).material;
    let mut others = [(Material::AIR, 0u8); FACE_NEIGHBOURS.len()];
    let mut distinct = 0;
    for offset in FACE_NEIGHBOURS {
        let neighbour = voxel(solid + IVec3::from_array(offset));
        let material = neighbour.material;
        if !neighbour.is_inside() || material == dominant || material == Material::AIR {
            continue;
        }
        match others[..distinct].iter_mut().find(|(m, _)| *m == material) {
            Some(entry) => entry.1 += 1,
            None => {
                others[distinct] = (material, 1);
                distinct += 1;
            }
        }
    }
    let others = &mut others[..distinct];
    others.sort_unstable_by_key(|&(material, count)| (Reverse(count), material));

    let mut slots = [(0u8, 0u8); MATERIAL_SLOTS];
    slots[0] = (dominant.0, MAX_MATERIAL_WEIGHT);
    for (slot, &(material, count)) in slots[1..].iter_mut().zip(others.iter()) {
        *slot = (material.0, neighbour_weight(count));
    }
    slots
}

/// `round(15 * count / 6)` clamped to 1 to 15, in integers so the rounding is exact.
fn neighbour_weight(count: u8) -> u8 {
    let max = u16::from(MAX_MATERIAL_WEIGHT);
    let neighbours = FACE_NEIGHBOURS.len() as u16;
    ((max * u16::from(count) + neighbours / 2) / neighbours).clamp(1, max) as u8
}
