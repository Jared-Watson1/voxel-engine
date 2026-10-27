/// Procedural material albedo layers (D18): one RGBA8 sRGB image per material id, layer index
/// equal to the id (0 is air and gets a debug colour), all `size` by `size`, row major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialTextureSet {
    pub size: u32,
    pub layers: Vec<Vec<u8>>,
}

impl MaterialTextureSet {
    pub fn layer_count(&self) -> u32 {
        self.layers.len() as u32
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(Vec::len).sum()
    }
}

/// Generates [`vx_voxel::Material::COUNT`] tileable layers from seeded noise. Pure in
/// (seed, size); runs on a worker at startup.
pub fn material_textures(seed: u64, size: u32) -> MaterialTextureSet {
    let _ = (seed, size);
    todo!("workstream C")
}
