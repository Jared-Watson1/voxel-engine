use std::cmp::Reverse;

use glam::Vec3;
use transvoxel::prelude::{Block, BlockStarView, extract};
use transvoxel::structs::grid_point::GridPoint;
use transvoxel::structs::vertex_index::VertexIndex;
use transvoxel::structs::voxel_index::VoxelIndex;
use transvoxel::traits::mesh_builder::MeshBuilder;
use transvoxel::traits::voxel_block::VoxelBlock;
use transvoxel::traits::voxel_data::{Density, VoxelData};
use vx_core::mesh::{MATERIAL_SLOTS, MAX_MATERIAL_WEIGHT};
use vx_core::{CHUNK_SIZE, PackedVertex, VertexAttributes, WorldPos};

use crate::chunk::{Chunk, Material, Voxel};
use crate::key::ChunkKey;

/// Packed, indexed triangle mesh of one chunk (D14): positions quantised in chunk local voxel
/// units, octahedral normals, material slots. Indices are local to the mesh (0 based). Counter
/// clockwise front faces seen from outside the solid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedMesh {
    pub key: ChunkKey,
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
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
}

/// Meshes the chunk's `CHUNK_SIZE`^3 cells with Transvoxel regular cells into packed vertices.
///
/// Vertex positions, normals and material slots depend only on the edge that produced the vertex
/// and the solid endpoint's face neighbours, never on the owning cell, so a vertex on a chunk face
/// is identical in both chunks that share the face (D14, D15, engineering rule 6). The chunk slot
/// word is left zero for the renderer to patch.
///
/// Per vertex, with `a` and `b` the edge endpoints in the crate's fixed corner order (the lower
/// coordinate first) and `t` the crossing of the quantised distance between them:
/// - position: `a + t * (b - a)` in voxel units, quantised by [`PackedVertex::pack`];
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
    profiling::scope!("mesh_chunk");
    let view = ChunkBlock {
        chunk,
        block: Block::new([0.0; 3], CHUNK_SIZE as f32, CHUNK_SIZE),
    };
    let blocks: BlockStarView<f32, Sample, &ChunkBlock, &ChunkBlock> =
        BlockStarView::new_simple(&view);
    let builder = extract(&blocks, SURFACE_DENSITY, PackedBuilder::new(chunk));
    PackedMesh {
        key: chunk.key(),
        vertices: builder.vertices,
        indices: builder.indices,
    }
}

/// The crate treats density above the threshold as inside, so it receives the negated distance.
const SURFACE_DENSITY: f32 = 0.0;

/// One grid point as the crate sees it: the stored voxel (its negated distance is the density)
/// and its local coordinates, which the material rule needs to reach the face neighbours.
#[derive(Debug, Clone, Copy)]
struct Sample {
    voxel: Voxel,
    local: [i8; 3],
}

impl VoxelData for Sample {
    type Density = f32;

    fn density(&self) -> f32 {
        -self.voxel.distance_voxels()
    }
}

/// Exposes a chunk to the crate in voxel units: block coordinates equal sample indices, so grid
/// point positions are exact integers.
struct ChunkBlock<'a> {
    chunk: &'a Chunk,
    block: Block<f32>,
}

impl VoxelBlock<f32, Sample> for ChunkBlock<'_> {
    fn block(&self) -> &Block<f32> {
        &self.block
    }

    fn get(&self, index: VoxelIndex) -> Sample {
        Sample {
            voxel: self.chunk.get(index.x, index.y, index.z),
            local: [index.x as i8, index.y as i8, index.z as i8],
        }
    }
}

struct PackedBuilder<'a> {
    chunk: &'a Chunk,
    vertices: Vec<PackedVertex>,
    indices: Vec<u32>,
    /// Vertex index per edge in a low face of the block, `NO_VERTEX` until the edge produced one.
    low_face_vertices: Vec<u32>,
}

const NO_VERTEX: u32 = u32::MAX;

/// Edges in the three low faces: per face, two edge directions over `(CHUNK_SIZE + 1)`^2 points.
const LOW_FACE_EDGE_SLOTS: usize = 3 * 2 * (CHUNK_SIZE + 1) * (CHUNK_SIZE + 1);

impl<'a> PackedBuilder<'a> {
    fn new(chunk: &'a Chunk) -> Self {
        Self {
            chunk,
            vertices: Vec::new(),
            indices: Vec::new(),
            low_face_vertices: vec![NO_VERTEX; LOW_FACE_EDGE_SLOTS],
        }
    }
}

/// Slot of an edge lying in a low face of the block (x, y or z equal to 0), `None` otherwise.
///
/// The crate creates a fresh vertex every time a cell meets such an edge, because the cell that
/// would own the shared vertex under Lengyel's vertex reuse scheme (Lengyel 2010) lies outside the
/// block. The builder hands back the first one instead, so a chunk face holds one vertex per
/// crossing edge on both of its sides and the seam is exact (D15).
fn low_face_edge_slot(a: [i8; 3], b: [i8; 3]) -> Option<usize> {
    let axis = (0..3).find(|&i| a[i] != b[i])?;
    let low = [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])];
    let face = (0..3).find(|&i| i != axis && low[i] == 0)?;
    let [p, q] = match face {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let samples = CHUNK_SIZE + 1;
    let direction = usize::from(axis == q);
    Some(((face * 2 + direction) * samples + low[p] as usize) * samples + low[q] as usize)
}

impl MeshBuilder<Sample, f32> for PackedBuilder<'_> {
    fn add_vertex_between(
        &mut self,
        a: GridPoint<Sample, f32>,
        b: GridPoint<Sample, f32>,
        t: f32,
    ) -> VertexIndex {
        let slot = low_face_edge_slot(a.voxel_data.local, b.voxel_data.local);
        if let Some(slot) = slot
            && self.low_face_vertices[slot] != NO_VERTEX
        {
            return VertexIndex(self.low_face_vertices[slot] as usize);
        }
        let lerp = |x: f32, y: f32| x + (y - x) * t;
        let position_voxels = Vec3::new(
            lerp(a.position.x, b.position.x),
            lerp(a.position.y, b.position.y),
            lerp(a.position.z, b.position.z),
        );
        let normal = Vec3::from(f32::gradients_to_normal(
            lerp(a.gradient.0, b.gradient.0),
            lerp(a.gradient.1, b.gradient.1),
            lerp(a.gradient.2, b.gradient.2),
        ));
        let solid = if a.voxel_data.voxel.is_inside() {
            a.voxel_data
        } else {
            b.voxel_data
        };
        let vertex = PackedVertex::pack(&VertexAttributes {
            position_voxels,
            normal,
            materials: material_slots(self.chunk, solid),
        });
        let index = self.vertices.len();
        self.vertices.push(vertex);
        if let Some(slot) = slot {
            self.low_face_vertices[slot] = index as u32;
        }
        VertexIndex(index)
    }

    fn add_triangle(&mut self, a: VertexIndex, b: VertexIndex, c: VertexIndex) {
        self.indices.extend([a.0 as u32, b.0 as u32, c.0 as u32]);
    }
}

const FACE_NEIGHBOURS: [[isize; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// The material rule of [`mesh_chunk`] for the solid endpoint of an edge. The endpoint lies in
/// 0 to `CHUNK_SIZE`, so its face neighbours are within the stored apron (D16).
fn material_slots(chunk: &Chunk, solid: Sample) -> [(u8, u8); MATERIAL_SLOTS] {
    let dominant = solid.voxel.material;
    let [x, y, z] = solid.local.map(isize::from);
    let mut others = [(Material::AIR, 0u8); FACE_NEIGHBOURS.len()];
    let mut distinct = 0;
    for [dx, dy, dz] in FACE_NEIGHBOURS {
        let neighbour = chunk.get(x + dx, y + dy, z + dz);
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

/// Indexed triangle mesh of one chunk in chunk local metres (relative to [`ChunkMesh::origin`]),
/// counter clockwise front faces seen from outside the solid.
///
/// The M0 form: f32 positions and normals unpacked from [`PackedMesh`] for the renderer until it
/// consumes packed vertices directly (D14).
#[derive(Debug, Clone, Default)]
pub struct ChunkMesh {
    pub origin: WorldPos,
    pub positions: Vec<[f32; 3]>,
    /// Unit normals from the signed distance gradient, pointing out of the solid.
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// [`mesh_chunk`] unpacked to f32 metres relative to the chunk origin.
pub fn mesh_regular(chunk: &Chunk) -> ChunkMesh {
    profiling::scope!("mesh_regular");
    let packed = mesh_chunk(chunk);
    let voxel_size_m = chunk.voxel_size_m() as f32;
    let (positions, normals) = packed
        .vertices
        .iter()
        .map(|vertex| {
            let attributes = vertex.unpack();
            (
                (attributes.position_voxels * voxel_size_m).to_array(),
                attributes.normal.to_array(),
            )
        })
        .unzip();
    ChunkMesh {
        origin: chunk.origin(),
        positions,
        normals,
        indices: packed.indices,
    }
}
