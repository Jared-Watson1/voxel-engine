use transvoxel::prelude::{Block, BlockStarView, extract};
use transvoxel::structs::grid_point::GridPoint;
use transvoxel::structs::vertex_index::VertexIndex;
use transvoxel::structs::voxel_index::VoxelIndex;
use transvoxel::traits::mesh_builder::MeshBuilder;
use transvoxel::traits::voxel_block::VoxelBlock;
use transvoxel::traits::voxel_data::Density;
use vx_core::{CHUNK_SIZE, PackedVertex, WorldPos};

use crate::chunk::Chunk;
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
pub fn mesh_chunk(chunk: &Chunk) -> PackedMesh {
    let _ = chunk;
    todo!("workstream D")
}

/// Indexed triangle mesh of one chunk in chunk local metres (relative to [`ChunkMesh::origin`]),
/// counter clockwise front faces seen from outside the solid.
///
/// Positions and normals are f32 for M0; M1 quantises them (design document, meshing).
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

/// Meshes the chunk's `CHUNK_SIZE`^3 cells with Transvoxel regular cells (Lengyel 2010) through the
/// `transvoxel` crate. Transition cells are not requested until LOD exists (M2).
pub fn mesh_regular(chunk: &Chunk) -> ChunkMesh {
    profiling::scope!("mesh_regular");
    let view = ChunkBlock {
        chunk,
        block: Block::new([0.0; 3], CHUNK_SIZE as f32, CHUNK_SIZE),
    };
    let blocks: BlockStarView<f32, f32, &ChunkBlock, &ChunkBlock> =
        BlockStarView::new_simple(&view);
    let builder = extract(
        &blocks,
        SURFACE_DENSITY,
        Builder {
            voxel_size_m: chunk.voxel_size_m() as f32,
            mesh: ChunkMesh {
                origin: chunk.origin(),
                ..ChunkMesh::default()
            },
        },
    );
    builder.mesh
}

/// The crate treats density above the threshold as inside, so it receives the negated distance.
const SURFACE_DENSITY: f32 = 0.0;

/// Exposes a chunk to the crate in voxel units: block coordinates equal sample indices.
struct ChunkBlock<'a> {
    chunk: &'a Chunk,
    block: Block<f32>,
}

impl VoxelBlock<f32, f32> for ChunkBlock<'_> {
    fn block(&self) -> &Block<f32> {
        &self.block
    }

    fn get(&self, index: VoxelIndex) -> f32 {
        -self.chunk.get(index.x, index.y, index.z).distance_voxels()
    }
}

struct Builder {
    voxel_size_m: f32,
    mesh: ChunkMesh,
}

impl MeshBuilder<f32, f32> for Builder {
    fn add_vertex_between(
        &mut self,
        a: GridPoint<f32, f32>,
        b: GridPoint<f32, f32>,
        t: f32,
    ) -> VertexIndex {
        let lerp = |x: f32, y: f32| x + (y - x) * t;
        let s = self.voxel_size_m;
        self.mesh.positions.push([
            lerp(a.position.x, b.position.x) * s,
            lerp(a.position.y, b.position.y) * s,
            lerp(a.position.z, b.position.z) * s,
        ]);
        self.mesh.normals.push(f32::gradients_to_normal(
            lerp(a.gradient.0, b.gradient.0),
            lerp(a.gradient.1, b.gradient.1),
            lerp(a.gradient.2, b.gradient.2),
        ));
        VertexIndex(self.mesh.positions.len() - 1)
    }

    fn add_triangle(&mut self, a: VertexIndex, b: VertexIndex, c: VertexIndex) {
        self.mesh
            .indices
            .extend([a.0 as u32, b.0 as u32, c.0 as u32]);
    }
}
