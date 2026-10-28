//! Terrain generation (D5, D13): a pure function of (position, seed, parameters), CPU
//! authoritative and bit identical on every machine (engineering rule 2).
//!
//! M2 samples a planet (D21): seeded fBm of the unit direction on a per level cube sphere grid,
//! plus 3D detail noise, with materials by altitude, depth and slope against the radial up. The
//! M1 flat `TerrainSampler` remains only while `vx_game` still streams the flat world. The planet
//! atlas arrives in M3.

mod cache;
mod cubemap;
mod height;
mod planet;
mod sampler;
mod seed;
mod textures;

pub use cache::HeightCacheStats;
pub use planet::{ChunkClass, PlanetSampler};
pub use sampler::TerrainSampler;
pub use textures::{MaterialTextureSet, material_textures};

#[cfg(test)]
mod tests {
    #[test]
    fn no_hash_maps_in_the_crate() {
        // Assembled from pieces so this file does not itself contain the forbidden text.
        let forbidden: Vec<String> = [
            ["Hash", "Map"],
            ["Hash", "Set"],
            [".si", "n("],
            [".co", "s("],
            [".ex", "p("],
            [".pow", "f("],
            [".l", "n("],
            [".ta", "n("],
            [".ata", "n2("],
        ]
        .iter()
        .map(|parts| parts.concat())
        .collect();
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<_> = std::fs::read_dir(&src)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
            .collect();
        files.sort();
        assert!(
            files.len() >= 4,
            "expected the crate's source files in {src:?}"
        );
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            for (line_number, line) in text.lines().enumerate() {
                for word in &forbidden {
                    assert!(
                        !line.contains(word.as_str()),
                        "{}:{}: contains {word}",
                        path.display(),
                        line_number + 1
                    );
                }
            }
        }
    }
}
