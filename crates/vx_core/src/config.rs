use std::ops::RangeInclusive;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Voxels along each edge of a chunk. The one compile time scale constant (D11): a fixed size
/// keeps meshing, compression and the GPU layout branch free.
pub const CHUNK_SIZE: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing world config: {0}")]
    Parse(#[from] ron::error::SpannedError),
    #[error("{field} = {value} is outside the supported range {min} to {max}")]
    OutOfRange {
        field: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
}

/// World scale parameters (D11), loaded from `assets/config/world.ron`.
///
/// Every field has a design default that the performance budgets are measured at and a supported
/// range that tests cover at both ends. Missing fields in the file take the design default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorldConfig {
    /// Edge length of one voxel in metres. Design default 1.0 (D11): 0.5 doubles memory and meshing
    /// cost per visible area, and 1 m matches godot_voxel, Veloren and Space Engineers class engines.
    /// Supported range [`WorldConfig::VOXEL_SIZE_RANGE_M`].
    pub voxel_size_m: f64,
    /// World seed (D5). Every generated value is a pure function of position and this seed.
    /// Design default 1.
    pub seed: u64,
    /// Radius in chunks of the sphere of chunk keys streamed around the camera. Design default 16
    /// (512 m at 1 m voxels), chosen so one LOD level stays within the terrain draw budget.
    /// Supported range [`WorldConfig::VIEW_RADIUS_RANGE_CHUNKS`].
    pub view_radius_chunks: u32,
    /// Chunks beyond `view_radius_chunks + stream_hysteresis_chunks` are evicted, so a camera
    /// oscillating at the edge does not thrash. Design default 2. Supported range
    /// [`WorldConfig::STREAM_HYSTERESIS_RANGE_CHUNKS`].
    pub stream_hysteresis_chunks: u32,
    /// The M1 flat world's terrain shape and materials.
    pub terrain: TerrainParams,
}

/// Shape and material parameters of the M1 noise terrain (a flat world, +Y up). All lengths in
/// metres. Each field carries its design default and supported range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TerrainParams {
    /// Mean surface height. Design default 0.0. Range [`TerrainParams::HEIGHT_RANGE_M`].
    pub base_height_m: f64,
    /// Largest deviation of the heightfield from the base height. Design default 96.0.
    /// Range [`TerrainParams::AMPLITUDE_RANGE_M`].
    pub amplitude_m: f64,
    /// Wavelength of the largest heightfield feature. Design default 1024.0.
    /// Range [`TerrainParams::WAVELENGTH_RANGE_M`].
    pub wavelength_m: f64,
    /// Heightfield fractal octaves. Design default 6. Range [`TerrainParams::OCTAVES_RANGE`].
    pub octaves: u32,
    /// Amplitude of the 3D detail noise added to the signed distance (roughness, small overhangs).
    /// Design default 3.0. Range [`TerrainParams::DETAIL_AMPLITUDE_RANGE_M`].
    pub detail_amplitude_m: f64,
    /// Wavelength of the 3D detail noise. Design default 40.0.
    /// Range [`TerrainParams::DETAIL_WAVELENGTH_RANGE_M`].
    pub detail_wavelength_m: f64,
    /// Surfaces below this height are sand (beaches). Design default -6.0.
    /// Range [`TerrainParams::HEIGHT_RANGE_M`].
    pub sand_below_m: f64,
    /// Thickness of the grass (or sand) layer at the surface; soil lies beneath it. Design default
    /// 1.5. Range [`TerrainParams::SOIL_DEPTH_RANGE_M`].
    pub grass_depth_m: f64,
    /// Depth below the surface at which soil gives way to rock. Design default 4.0.
    /// Range [`TerrainParams::SOIL_DEPTH_RANGE_M`].
    pub soil_depth_m: f64,
    /// Slopes steeper than this (rise over run) are bare rock. Design default 1.0 (45 degrees).
    /// Range [`TerrainParams::ROCK_SLOPE_RANGE`].
    pub rock_slope: f64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 1.0,
            seed: 1,
            view_radius_chunks: 16,
            stream_hysteresis_chunks: 2,
            terrain: TerrainParams::default(),
        }
    }
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            base_height_m: 0.0,
            amplitude_m: 96.0,
            wavelength_m: 1024.0,
            octaves: 6,
            detail_amplitude_m: 3.0,
            detail_wavelength_m: 40.0,
            sand_below_m: -6.0,
            grass_depth_m: 1.5,
            soil_depth_m: 4.0,
            rock_slope: 1.0,
        }
    }
}

impl TerrainParams {
    pub const HEIGHT_RANGE_M: RangeInclusive<f64> = -2000.0..=2000.0;
    pub const AMPLITUDE_RANGE_M: RangeInclusive<f64> = 0.0..=2000.0;
    pub const WAVELENGTH_RANGE_M: RangeInclusive<f64> = 16.0..=100_000.0;
    pub const OCTAVES_RANGE: RangeInclusive<u32> = 1..=12;
    pub const DETAIL_AMPLITUDE_RANGE_M: RangeInclusive<f64> = 0.0..=64.0;
    pub const DETAIL_WAVELENGTH_RANGE_M: RangeInclusive<f64> = 1.0..=1000.0;
    pub const SOIL_DEPTH_RANGE_M: RangeInclusive<f64> = 0.0..=64.0;
    pub const ROCK_SLOPE_RANGE: RangeInclusive<f64> = 0.1..=10.0;

    pub fn validate(&self) -> Result<(), ConfigError> {
        check(
            "terrain.base_height_m",
            self.base_height_m,
            &Self::HEIGHT_RANGE_M,
        )?;
        check(
            "terrain.amplitude_m",
            self.amplitude_m,
            &Self::AMPLITUDE_RANGE_M,
        )?;
        check(
            "terrain.wavelength_m",
            self.wavelength_m,
            &Self::WAVELENGTH_RANGE_M,
        )?;
        check(
            "terrain.octaves",
            f64::from(self.octaves),
            &(f64::from(*Self::OCTAVES_RANGE.start())..=f64::from(*Self::OCTAVES_RANGE.end())),
        )?;
        check(
            "terrain.detail_amplitude_m",
            self.detail_amplitude_m,
            &Self::DETAIL_AMPLITUDE_RANGE_M,
        )?;
        check(
            "terrain.detail_wavelength_m",
            self.detail_wavelength_m,
            &Self::DETAIL_WAVELENGTH_RANGE_M,
        )?;
        check(
            "terrain.sand_below_m",
            self.sand_below_m,
            &Self::HEIGHT_RANGE_M,
        )?;
        check(
            "terrain.grass_depth_m",
            self.grass_depth_m,
            &Self::SOIL_DEPTH_RANGE_M,
        )?;
        check(
            "terrain.soil_depth_m",
            self.soil_depth_m,
            &Self::SOIL_DEPTH_RANGE_M,
        )?;
        check(
            "terrain.rock_slope",
            self.rock_slope,
            &Self::ROCK_SLOPE_RANGE,
        )?;
        Ok(())
    }
}

impl WorldConfig {
    pub const VOXEL_SIZE_RANGE_M: RangeInclusive<f64> = 0.25..=4.0;
    pub const VIEW_RADIUS_RANGE_CHUNKS: RangeInclusive<u32> = 2..=48;
    pub const STREAM_HYSTERESIS_RANGE_CHUNKS: RangeInclusive<u32> = 0..=8;

    pub fn from_ron_str(text: &str) -> Result<Self, ConfigError> {
        let config: Self = ron::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Reads and validates the file. Blocking I/O: call from a worker, never the main thread (D9).
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_ron_str(&text)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        check("voxel_size_m", self.voxel_size_m, &Self::VOXEL_SIZE_RANGE_M)?;
        check(
            "view_radius_chunks",
            f64::from(self.view_radius_chunks),
            &(f64::from(*Self::VIEW_RADIUS_RANGE_CHUNKS.start())
                ..=f64::from(*Self::VIEW_RADIUS_RANGE_CHUNKS.end())),
        )?;
        check(
            "stream_hysteresis_chunks",
            f64::from(self.stream_hysteresis_chunks),
            &(f64::from(*Self::STREAM_HYSTERESIS_RANGE_CHUNKS.start())
                ..=f64::from(*Self::STREAM_HYSTERESIS_RANGE_CHUNKS.end())),
        )?;
        self.terrain.validate()
    }

    /// Edge length of one chunk in metres.
    pub fn chunk_extent_m(&self) -> f64 {
        CHUNK_SIZE as f64 * self.voxel_size_m
    }

    /// Chunks evicted beyond this radius (view radius plus hysteresis).
    pub fn evict_radius_chunks(&self) -> u32 {
        self.view_radius_chunks + self.stream_hysteresis_chunks
    }
}

fn check(field: &'static str, value: f64, range: &RangeInclusive<f64>) -> Result<(), ConfigError> {
    if value.is_finite() && range.contains(&value) {
        Ok(())
    } else {
        Err(ConfigError::OutOfRange {
            field,
            value,
            min: *range.start(),
            max: *range.end(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_defaults() {
        let c = WorldConfig::default();
        assert_eq!(c.voxel_size_m, 1.0);
        assert_eq!(c.chunk_extent_m(), 32.0);
        assert_eq!(c.seed, 1);
        assert_eq!(c.view_radius_chunks, 16);
        assert_eq!(c.evict_radius_chunks(), 18);
        assert_eq!(c.terrain.amplitude_m, 96.0);
        c.validate().unwrap();
    }

    #[test]
    fn shipped_world_ron_holds_the_design_defaults() {
        let text = include_str!("../../../assets/config/world.ron");
        assert_eq!(
            WorldConfig::from_ron_str(text).unwrap(),
            WorldConfig::default()
        );
    }

    #[test]
    fn missing_fields_take_defaults() {
        assert_eq!(
            WorldConfig::from_ron_str("()").unwrap(),
            WorldConfig::default()
        );
        assert_eq!(
            WorldConfig::from_ron_str("(terrain: (octaves: 3))")
                .unwrap()
                .terrain
                .octaves,
            3
        );
    }

    #[test]
    fn both_ends_of_every_range_are_accepted() {
        for v in [
            *WorldConfig::VOXEL_SIZE_RANGE_M.start(),
            *WorldConfig::VOXEL_SIZE_RANGE_M.end(),
        ] {
            let c = WorldConfig::from_ron_str(&format!("(voxel_size_m: {v})")).unwrap();
            assert_eq!(c.voxel_size_m, v);
        }
        for v in [
            *WorldConfig::VIEW_RADIUS_RANGE_CHUNKS.start(),
            *WorldConfig::VIEW_RADIUS_RANGE_CHUNKS.end(),
        ] {
            let c = WorldConfig::from_ron_str(&format!("(view_radius_chunks: {v})")).unwrap();
            assert_eq!(c.view_radius_chunks, v);
        }
        for (field, range) in [
            ("base_height_m", TerrainParams::HEIGHT_RANGE_M),
            ("amplitude_m", TerrainParams::AMPLITUDE_RANGE_M),
            ("wavelength_m", TerrainParams::WAVELENGTH_RANGE_M),
            (
                "detail_amplitude_m",
                TerrainParams::DETAIL_AMPLITUDE_RANGE_M,
            ),
            (
                "detail_wavelength_m",
                TerrainParams::DETAIL_WAVELENGTH_RANGE_M,
            ),
            ("sand_below_m", TerrainParams::HEIGHT_RANGE_M),
            ("grass_depth_m", TerrainParams::SOIL_DEPTH_RANGE_M),
            ("soil_depth_m", TerrainParams::SOIL_DEPTH_RANGE_M),
            ("rock_slope", TerrainParams::ROCK_SLOPE_RANGE),
        ] {
            for v in [*range.start(), *range.end()] {
                WorldConfig::from_ron_str(&format!("(terrain: ({field}: {v}))"))
                    .unwrap_or_else(|e| panic!("{field} = {v}: {e}"));
            }
        }
    }

    #[test]
    fn out_of_range_is_rejected() {
        for text in [
            "(voxel_size_m: 0.1)",
            "(voxel_size_m: 8.0)",
            "(voxel_size_m: 0.0)",
            "(voxel_size_m: -1.0)",
            "(view_radius_chunks: 1)",
            "(view_radius_chunks: 49)",
            "(stream_hysteresis_chunks: 9)",
            "(terrain: (octaves: 0))",
            "(terrain: (octaves: 13))",
            "(terrain: (wavelength_m: 1.0))",
            "(terrain: (rock_slope: 0.0))",
            "(terrain: (amplitude_m: -1.0))",
            "(terrain: (grass_depth_m: -5.0))",
            "(terrain: (grass_depth_m: 1e300))",
        ] {
            let r = WorldConfig::from_ron_str(text);
            assert!(matches!(r, Err(ConfigError::OutOfRange { .. })), "{text}");
        }
    }

    #[test]
    fn unknown_fields_are_rejected() {
        assert!(matches!(
            WorldConfig::from_ron_str("(voxel_size: 1.0)"),
            Err(ConfigError::Parse(_))
        ));
        assert!(matches!(
            WorldConfig::from_ron_str("(terrain: (height: 1.0))"),
            Err(ConfigError::Parse(_))
        ));
    }
}
