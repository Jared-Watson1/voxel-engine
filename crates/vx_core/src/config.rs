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
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self { voxel_size_m: 1.0 }
    }
}

impl WorldConfig {
    pub const VOXEL_SIZE_RANGE_M: RangeInclusive<f64> = 0.25..=4.0;

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
        let range = Self::VOXEL_SIZE_RANGE_M;
        if !range.contains(&self.voxel_size_m) {
            return Err(ConfigError::OutOfRange {
                field: "voxel_size_m",
                value: self.voxel_size_m,
                min: *range.start(),
                max: *range.end(),
            });
        }
        Ok(())
    }

    /// Edge length of one chunk in metres.
    pub fn chunk_extent_m(&self) -> f64 {
        CHUNK_SIZE as f64 * self.voxel_size_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_default_is_one_metre() {
        assert_eq!(WorldConfig::default().voxel_size_m, 1.0);
        assert_eq!(WorldConfig::default().chunk_extent_m(), 32.0);
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
    }

    #[test]
    fn both_ends_of_the_range_are_accepted() {
        for v in [
            *WorldConfig::VOXEL_SIZE_RANGE_M.start(),
            *WorldConfig::VOXEL_SIZE_RANGE_M.end(),
        ] {
            let c = WorldConfig::from_ron_str(&format!("(voxel_size_m: {v})")).unwrap();
            assert_eq!(c.voxel_size_m, v);
        }
    }

    #[test]
    fn out_of_range_is_rejected() {
        for v in ["0.1", "8.0", "0.0", "-1.0"] {
            let r = WorldConfig::from_ron_str(&format!("(voxel_size_m: {v})"));
            assert!(matches!(r, Err(ConfigError::OutOfRange { .. })), "{v}");
        }
    }

    #[test]
    fn unknown_fields_are_rejected() {
        assert!(matches!(
            WorldConfig::from_ron_str("(voxel_size: 1.0)"),
            Err(ConfigError::Parse(_))
        ));
    }
}
