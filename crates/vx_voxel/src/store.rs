use crate::chunk::{Chunk, Voxel};

/// Compressed resident form of a chunk (D17).
///
/// `Uniform` holds one value for a block whose samples are all equal (sky, deep rock). `Palette`
/// holds the distinct `(distance, material)` pairs of the block and one bit packed index per
/// sample, using the smallest width that addresses the palette. Both are lossless.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkStore {
    Uniform(Voxel),
    Palette(PaletteBlock),
}

/// Palette compressed samples. Fields are private so the layout can change without touching
/// callers; [`ChunkStore::get`] and [`ChunkStore::decompress_into`] are the read paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteBlock {
    palette: Vec<Voxel>,
    bits_per_sample: u8,
    words: Box<[u64]>,
}

impl ChunkStore {
    /// Compresses a dense chunk. Never loses information: `decompress_into` restores every sample.
    pub fn compress(chunk: &Chunk) -> Self {
        let _ = chunk;
        todo!("workstream B")
    }

    /// Writes every sample into `chunk`, which keeps its key and voxel size.
    pub fn decompress_into(&self, chunk: &mut Chunk) {
        let _ = chunk;
        todo!("workstream B")
    }

    /// Sample at local coordinates -1 to `CHUNK_SIZE + 1`, decoded in constant time.
    pub fn get(&self, x: isize, y: isize, z: isize) -> Voxel {
        let _ = (x, y, z);
        todo!("workstream B")
    }

    pub fn uniform(&self) -> Option<Voxel> {
        match self {
            Self::Uniform(v) => Some(*v),
            Self::Palette(_) => None,
        }
    }

    /// True when the block holds both inside and outside samples, so meshing can produce
    /// triangles. A uniform block never has a surface.
    pub fn has_surface(&self) -> bool {
        todo!("workstream B")
    }

    /// Bytes this store occupies in memory, inline and on the heap, for the resident voxel budget.
    pub fn resident_bytes(&self) -> usize {
        todo!("workstream B")
    }
}
