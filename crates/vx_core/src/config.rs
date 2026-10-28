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
/// range that tests cover at both ends. Missing fields in the file take the design default. The
/// planet itself is described by [`PlanetParams`] in the system configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorldConfig {
    /// Edge length of one level 0 voxel in metres; level `L` voxels are `2^L` times larger.
    /// Design default 1.0 (D11): 0.5 doubles memory and meshing cost per visible area, and 1 m
    /// matches godot_voxel, Veloren and Space Engineers class engines. Supported range
    /// [`WorldConfig::VOXEL_SIZE_RANGE_M`].
    pub voxel_size_m: f64,
    /// World seed (D5). Every generated value is a pure function of position and this seed.
    /// Design default 1.
    pub seed: u64,
    /// Levels of the planet octree (D20); the top level's chunks are the roots. Design default 8,
    /// enough for a 10 km planet from the ground to 100 km (level 7 chunks are 4,096 m at 1 m
    /// voxels); 10 for the 50 km run. Supported range [`WorldConfig::LOD_LEVELS_RANGE`].
    pub lod_levels: u32,
    /// A node splits when the camera is within this many of its own chunk extents of the node's
    /// box. Design default 4.0 (the design document's level 0 radius of four chunks): at 1080p
    /// and a 70 degree field of view a voxel then spans about seven pixels at every level.
    /// Supported range [`WorldConfig::LOD_RADIUS_RANGE_CHUNKS`].
    pub lod_radius_chunks: f64,
    /// A split node merges only beyond `lod_radius_chunks + lod_hysteresis_chunks` extents, so a
    /// camera at the boundary does not thrash. Design default 1.0. Supported range
    /// [`WorldConfig::LOD_HYSTERESIS_RANGE_CHUNKS`].
    pub lod_hysteresis_chunks: f64,
    /// The render origin rebases when the camera is farther than this from it (D4, D23). Design
    /// default 4,000 m: f32 offsets within it keep a quarter millimetre of resolution. Supported
    /// range [`WorldConfig::ORIGIN_REBASE_RANGE_M`].
    pub origin_rebase_m: f64,
}

/// Physical parameters of one planet (D11, D24, D25), part of the system configuration. Lengths
/// in the names' units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlanetParams {
    /// Mean radius (the zero of `TerrainParams::base_height_m`). Design default 10 km for the M2
    /// slice, 600 km for generated worlds (M7). Range [`PlanetParams::RADIUS_RANGE_KM`].
    pub radius_km: f64,
    /// Gravity at the mean radius; with the radius it fixes the gravitational parameter
    /// `g R^2`. Design default 9.81. Range [`PlanetParams::SURFACE_GRAVITY_RANGE_M_S2`].
    pub surface_gravity_m_s2: f64,
    /// Sidereal rotation period. Design default 3,600 s, a visible day in a short session.
    /// Range [`PlanetParams::ROTATION_PERIOD_RANGE_S`].
    pub rotation_period_s: f64,
    /// Angle between the rotation axis and the orbit normal. Design default 15 degrees.
    /// Range [`PlanetParams::AXIAL_TILT_RANGE_DEG`].
    pub axial_tilt_deg: f64,
    /// Terrain shape and materials; heights are radial, relative to the mean radius.
    pub terrain: TerrainParams,
    pub atmosphere: AtmosphereParams,
}

/// Atmosphere of one planet for the Hillaire sky (D8, D25), given as zenith optical depths and
/// scale heights relative to the atmosphere's height, so a small planet keeps an Earth like sky.
/// Defaults are Earth's values from Hillaire 2020 (Rayleigh 5.802, 13.558, 33.1 per Mm over 8 km;
/// ozone 0.650, 1.881, 0.085 per Mm in a 30 km tent centred at 25 km; a 100 km atmosphere),
/// converted to optical depths, except the Mie layer, which is hazier than Earth's for the art
/// direction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AtmosphereParams {
    /// Height of the top of the atmosphere above the mean radius as a fraction of the radius.
    /// Design default 0.1 (D11, the M2 slice). Range [`AtmosphereParams::HEIGHT_FRACTION_RANGE`].
    pub height_fraction: f64,
    /// Rayleigh scattering optical depth at the zenith for red, green and blue (680, 550 and
    /// 440 nm). Design default 0.0464, 0.1085, 0.2648. Range [`AtmosphereParams::OPTICAL_DEPTH_RANGE`].
    pub rayleigh_optical_depth: [f64; 3],
    /// Rayleigh density scale height as a fraction of the atmosphere height. Design default 0.08.
    /// Range [`AtmosphereParams::SCALE_HEIGHT_FRACTION_RANGE`].
    pub rayleigh_scale_height_fraction: f64,
    /// Mie scattering optical depth at the zenith (grey). Design default 0.02, about four times
    /// Earth's 0.0048, for the bright warm haze of the art direction (Lay of the Land).
    pub mie_scattering_optical_depth: f64,
    /// Mie absorption optical depth at the zenith (grey). Design default 0.002 (Earth 0.00528),
    /// so the haze stays bright rather than grey.
    pub mie_absorption_optical_depth: f64,
    /// Mie density scale height as a fraction of the atmosphere height. Design default 0.012.
    pub mie_scale_height_fraction: f64,
    /// Mie phase asymmetry (Cornette-Shanks g). Design default 0.8. Range
    /// [`AtmosphereParams::MIE_ASYMMETRY_RANGE`].
    pub mie_asymmetry: f64,
    /// Ozone absorption optical depth at the zenith. Design default 0.00975, 0.0282, 0.00128.
    pub ozone_optical_depth: [f64; 3],
    /// Altitude of the ozone tent's peak as a fraction of the atmosphere height. Design default
    /// 0.25. Range [`AtmosphereParams::FRACTION_RANGE`].
    pub ozone_centre_fraction: f64,
    /// Full width of the ozone tent as a fraction of the atmosphere height. Design default 0.3.
    /// Range [`AtmosphereParams::FRACTION_RANGE`].
    pub ozone_width_fraction: f64,
    /// Ground albedo seen by the multiple scattering LUT. Design default 0.3.
    /// Range [`AtmosphereParams::FRACTION_RANGE`].
    pub ground_albedo: [f64; 3],
}

/// Shape and material parameters of the noise terrain. On a planet every height is radial and
/// relative to the mean radius. All lengths in metres. Each field carries its design default and
/// supported range.
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
    /// Surfaces above this height are snow. Design default 2000.0, above the default amplitude so
    /// the M1 terrain has none; the planet configuration sets its own. Range
    /// [`TerrainParams::HEIGHT_RANGE_M`].
    pub snow_above_m: f64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 1.0,
            seed: 1,
            lod_levels: 8,
            lod_radius_chunks: 4.0,
            lod_hysteresis_chunks: 1.0,
            origin_rebase_m: 4000.0,
        }
    }
}

impl Default for PlanetParams {
    fn default() -> Self {
        Self {
            radius_km: 10.0,
            surface_gravity_m_s2: 9.81,
            rotation_period_s: 3600.0,
            axial_tilt_deg: 15.0,
            terrain: TerrainParams::planet_default(),
            atmosphere: AtmosphereParams::default(),
        }
    }
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self {
            height_fraction: 0.1,
            rayleigh_optical_depth: [0.0464, 0.1085, 0.2648],
            rayleigh_scale_height_fraction: 0.08,
            mie_scattering_optical_depth: 0.02,
            mie_absorption_optical_depth: 0.002,
            mie_scale_height_fraction: 0.012,
            mie_asymmetry: 0.8,
            ozone_optical_depth: [0.00975, 0.0282, 0.00128],
            ozone_centre_fraction: 0.25,
            ozone_width_fraction: 0.3,
            ground_albedo: [0.3, 0.3, 0.3],
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
            snow_above_m: 2000.0,
        }
    }
}

impl TerrainParams {
    /// Design defaults for the 10 km M2 planet: rolling hills of up to 260 m over 1.5 km
    /// wavelengths with pale rock outcrops on slopes steeper than 0.6 (about 3 % of the surface,
    /// the Lay of the Land reference), sand in the lowest basins (about 5 %) and snow only on the
    /// highest peaks (about 2 %). Shares measured by `vx_worldgen` over 20,000 directions at seed 1.
    /// Heights are radial, relative to the mean radius.
    pub fn planet_default() -> Self {
        Self {
            amplitude_m: 260.0,
            wavelength_m: 1500.0,
            octaves: 7,
            sand_below_m: -65.0,
            rock_slope: 0.6,
            snow_above_m: 85.0,
            ..Self::default()
        }
    }

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
        check(
            "terrain.snow_above_m",
            self.snow_above_m,
            &Self::HEIGHT_RANGE_M,
        )?;
        Ok(())
    }
}

impl PlanetParams {
    pub const RADIUS_RANGE_KM: RangeInclusive<f64> = 1.0..=6371.0;
    pub const SURFACE_GRAVITY_RANGE_M_S2: RangeInclusive<f64> = 0.1..=30.0;
    pub const ROTATION_PERIOD_RANGE_S: RangeInclusive<f64> = 60.0..=1.0e8;
    pub const AXIAL_TILT_RANGE_DEG: RangeInclusive<f64> = 0.0..=180.0;

    pub fn radius_m(&self) -> f64 {
        self.radius_km * 1000.0
    }

    /// Gravitational parameter `G M = g R^2` in m^3/s^2.
    pub fn gm_m3_s2(&self) -> f64 {
        self.surface_gravity_m_s2 * self.radius_m() * self.radius_m()
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        check("planet.radius_km", self.radius_km, &Self::RADIUS_RANGE_KM)?;
        check(
            "planet.surface_gravity_m_s2",
            self.surface_gravity_m_s2,
            &Self::SURFACE_GRAVITY_RANGE_M_S2,
        )?;
        check(
            "planet.rotation_period_s",
            self.rotation_period_s,
            &Self::ROTATION_PERIOD_RANGE_S,
        )?;
        check(
            "planet.axial_tilt_deg",
            self.axial_tilt_deg,
            &Self::AXIAL_TILT_RANGE_DEG,
        )?;
        self.terrain.validate()?;
        self.atmosphere.validate()
    }
}

impl AtmosphereParams {
    pub const HEIGHT_FRACTION_RANGE: RangeInclusive<f64> = 0.01..=0.5;
    pub const OPTICAL_DEPTH_RANGE: RangeInclusive<f64> = 0.0..=10.0;
    pub const SCALE_HEIGHT_FRACTION_RANGE: RangeInclusive<f64> = 0.001..=0.5;
    pub const MIE_ASYMMETRY_RANGE: RangeInclusive<f64> = 0.0..=0.99;
    pub const FRACTION_RANGE: RangeInclusive<f64> = 0.0..=1.0;

    /// Height of the top of the atmosphere above the mean radius, in metres.
    pub fn top_m(&self, radius_m: f64) -> f64 {
        self.height_fraction * radius_m
    }

    /// Rayleigh scattering coefficient at the mean radius per metre, `tau / H` (the integral of
    /// an exponential density over a thick atmosphere is its scale height).
    pub fn rayleigh_scattering_per_m(&self, radius_m: f64) -> [f64; 3] {
        let h = self.rayleigh_scale_height_fraction * self.top_m(radius_m);
        self.rayleigh_optical_depth.map(|tau| tau / h)
    }

    /// Mie scattering and absorption coefficients at the mean radius per metre.
    pub fn mie_per_m(&self, radius_m: f64) -> (f64, f64) {
        let h = self.mie_scale_height_fraction * self.top_m(radius_m);
        (
            self.mie_scattering_optical_depth / h,
            self.mie_absorption_optical_depth / h,
        )
    }

    /// Ozone absorption coefficient at the tent's peak per metre: the tent integrates to half
    /// its width.
    pub fn ozone_absorption_per_m(&self, radius_m: f64) -> [f64; 3] {
        let half_width = 0.5 * self.ozone_width_fraction * self.top_m(radius_m);
        self.ozone_optical_depth.map(|tau| tau / half_width)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        check(
            "atmosphere.height_fraction",
            self.height_fraction,
            &Self::HEIGHT_FRACTION_RANGE,
        )?;
        for tau in self.rayleigh_optical_depth {
            check(
                "atmosphere.rayleigh_optical_depth",
                tau,
                &Self::OPTICAL_DEPTH_RANGE,
            )?;
        }
        for tau in self.ozone_optical_depth {
            check(
                "atmosphere.ozone_optical_depth",
                tau,
                &Self::OPTICAL_DEPTH_RANGE,
            )?;
        }
        check(
            "atmosphere.mie_scattering_optical_depth",
            self.mie_scattering_optical_depth,
            &Self::OPTICAL_DEPTH_RANGE,
        )?;
        check(
            "atmosphere.mie_absorption_optical_depth",
            self.mie_absorption_optical_depth,
            &Self::OPTICAL_DEPTH_RANGE,
        )?;
        check(
            "atmosphere.rayleigh_scale_height_fraction",
            self.rayleigh_scale_height_fraction,
            &Self::SCALE_HEIGHT_FRACTION_RANGE,
        )?;
        check(
            "atmosphere.mie_scale_height_fraction",
            self.mie_scale_height_fraction,
            &Self::SCALE_HEIGHT_FRACTION_RANGE,
        )?;
        check(
            "atmosphere.mie_asymmetry",
            self.mie_asymmetry,
            &Self::MIE_ASYMMETRY_RANGE,
        )?;
        check(
            "atmosphere.ozone_centre_fraction",
            self.ozone_centre_fraction,
            &Self::FRACTION_RANGE,
        )?;
        check(
            "atmosphere.ozone_width_fraction",
            self.ozone_width_fraction,
            &Self::FRACTION_RANGE,
        )?;
        if self.ozone_width_fraction <= 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "atmosphere.ozone_width_fraction",
                value: self.ozone_width_fraction,
                min: f64::MIN_POSITIVE,
                max: 1.0,
            });
        }
        for a in self.ground_albedo {
            check("atmosphere.ground_albedo", a, &Self::FRACTION_RANGE)?;
        }
        Ok(())
    }
}

impl WorldConfig {
    pub const VOXEL_SIZE_RANGE_M: RangeInclusive<f64> = 0.25..=4.0;
    pub const LOD_LEVELS_RANGE: RangeInclusive<u32> = 4..=14;
    pub const LOD_RADIUS_RANGE_CHUNKS: RangeInclusive<f64> = 2.0..=16.0;
    pub const LOD_HYSTERESIS_RANGE_CHUNKS: RangeInclusive<f64> = 0.0..=4.0;
    pub const ORIGIN_REBASE_RANGE_M: RangeInclusive<f64> = 1000.0..=16000.0;

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
            "lod_levels",
            f64::from(self.lod_levels),
            &(f64::from(*Self::LOD_LEVELS_RANGE.start())
                ..=f64::from(*Self::LOD_LEVELS_RANGE.end())),
        )?;
        check(
            "lod_radius_chunks",
            self.lod_radius_chunks,
            &Self::LOD_RADIUS_RANGE_CHUNKS,
        )?;
        check(
            "lod_hysteresis_chunks",
            self.lod_hysteresis_chunks,
            &Self::LOD_HYSTERESIS_RANGE_CHUNKS,
        )?;
        check(
            "origin_rebase_m",
            self.origin_rebase_m,
            &Self::ORIGIN_REBASE_RANGE_M,
        )
    }

    /// Edge length of one level 0 chunk in metres.
    pub fn chunk_extent_m(&self) -> f64 {
        CHUNK_SIZE as f64 * self.voxel_size_m
    }

    /// Voxel size of octree level `level` in metres.
    pub fn level_voxel_size_m(&self, level: u8) -> f64 {
        self.voxel_size_m * f64::from(1u32 << level)
    }

    /// The coarsest octree level, whose chunks are the roots.
    pub fn top_level(&self) -> u8 {
        (self.lod_levels - 1) as u8
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

    fn planet(text: &str) -> Result<PlanetParams, ConfigError> {
        let p: PlanetParams = ron::from_str(text)?;
        p.validate()?;
        Ok(p)
    }

    #[test]
    fn design_defaults() {
        let c = WorldConfig::default();
        assert_eq!(c.voxel_size_m, 1.0);
        assert_eq!(c.chunk_extent_m(), 32.0);
        assert_eq!(c.seed, 1);
        assert_eq!(c.lod_levels, 8);
        assert_eq!(c.top_level(), 7);
        assert_eq!(c.level_voxel_size_m(7), 128.0);
        assert_eq!(c.origin_rebase_m, 4000.0);
        c.validate().unwrap();
        let p = PlanetParams::default();
        assert_eq!(p.radius_m(), 10_000.0);
        assert!((p.gm_m3_s2() - 9.81e8).abs() < 1.0);
        p.validate().unwrap();
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
        assert_eq!(planet("()").unwrap(), PlanetParams::default());
        assert_eq!(
            planet("(terrain: (octaves: 3))").unwrap().terrain.octaves,
            3
        );
    }

    #[test]
    fn atmosphere_defaults_reproduce_earth_coefficients() {
        // Earth: 6360 km radius with a 100 km atmosphere (Hillaire 2020).
        let earth = AtmosphereParams {
            height_fraction: 100.0 / 6360.0,
            mie_scattering_optical_depth: 0.0048,
            mie_absorption_optical_depth: 0.00528,
            ..AtmosphereParams::default()
        };
        let r = 6_360_000.0;
        let rayleigh = earth.rayleigh_scattering_per_m(r);
        for (got, want) in rayleigh.iter().zip([5.802e-6, 13.558e-6, 33.1e-6]) {
            assert!((got / want - 1.0).abs() < 1.0e-3, "{got} vs {want}");
        }
        let (mie_s, mie_a) = earth.mie_per_m(r);
        assert!((mie_s / 3.996e-6 - 1.0).abs() < 2.0e-3);
        assert!((mie_a / 4.40e-6 - 1.0).abs() < 2.0e-3);
        let ozone = earth.ozone_absorption_per_m(r);
        assert!((ozone[1] / 1.881e-6 - 1.0).abs() < 2.0e-3);
        // A 10 km planet keeps the optical depths, so its coefficients are 636 times larger.
        let small = AtmosphereParams::default().rayleigh_scattering_per_m(10_000.0);
        assert!((small[2] * 1000.0 * 0.08 - 0.2648).abs() < 1.0e-9);
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
            *WorldConfig::LOD_LEVELS_RANGE.start(),
            *WorldConfig::LOD_LEVELS_RANGE.end(),
        ] {
            let c = WorldConfig::from_ron_str(&format!("(lod_levels: {v})")).unwrap();
            assert_eq!(c.lod_levels, v);
        }
        for (field, range) in [
            ("lod_radius_chunks", WorldConfig::LOD_RADIUS_RANGE_CHUNKS),
            (
                "lod_hysteresis_chunks",
                WorldConfig::LOD_HYSTERESIS_RANGE_CHUNKS,
            ),
            ("origin_rebase_m", WorldConfig::ORIGIN_REBASE_RANGE_M),
        ] {
            for v in [*range.start(), *range.end()] {
                WorldConfig::from_ron_str(&format!("({field}: {v:?})"))
                    .unwrap_or_else(|e| panic!("{field} = {v}: {e}"));
            }
        }
        for (field, range) in [
            ("radius_km", PlanetParams::RADIUS_RANGE_KM),
            (
                "surface_gravity_m_s2",
                PlanetParams::SURFACE_GRAVITY_RANGE_M_S2,
            ),
            ("rotation_period_s", PlanetParams::ROTATION_PERIOD_RANGE_S),
            ("axial_tilt_deg", PlanetParams::AXIAL_TILT_RANGE_DEG),
        ] {
            for v in [*range.start(), *range.end()] {
                planet(&format!("({field}: {v:?})"))
                    .unwrap_or_else(|e| panic!("{field} = {v}: {e}"));
            }
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
            ("snow_above_m", TerrainParams::HEIGHT_RANGE_M),
            ("grass_depth_m", TerrainParams::SOIL_DEPTH_RANGE_M),
            ("soil_depth_m", TerrainParams::SOIL_DEPTH_RANGE_M),
            ("rock_slope", TerrainParams::ROCK_SLOPE_RANGE),
        ] {
            for v in [*range.start(), *range.end()] {
                planet(&format!("(terrain: ({field}: {v:?}))"))
                    .unwrap_or_else(|e| panic!("{field} = {v}: {e}"));
            }
        }
        for (field, range) in [
            ("height_fraction", AtmosphereParams::HEIGHT_FRACTION_RANGE),
            (
                "rayleigh_scale_height_fraction",
                AtmosphereParams::SCALE_HEIGHT_FRACTION_RANGE,
            ),
            ("mie_asymmetry", AtmosphereParams::MIE_ASYMMETRY_RANGE),
            (
                "mie_scattering_optical_depth",
                AtmosphereParams::OPTICAL_DEPTH_RANGE,
            ),
            ("ozone_centre_fraction", AtmosphereParams::FRACTION_RANGE),
        ] {
            for v in [*range.start(), *range.end()] {
                planet(&format!("(atmosphere: ({field}: {v:?}))"))
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
            "(lod_levels: 3)",
            "(lod_levels: 15)",
            "(lod_radius_chunks: 1.0)",
            "(origin_rebase_m: 100.0)",
        ] {
            let r = WorldConfig::from_ron_str(text);
            assert!(matches!(r, Err(ConfigError::OutOfRange { .. })), "{text}");
        }
        for text in [
            "(radius_km: 0.5)",
            "(radius_km: 7000.0)",
            "(surface_gravity_m_s2: 0.0)",
            "(terrain: (octaves: 0))",
            "(terrain: (octaves: 13))",
            "(terrain: (wavelength_m: 1.0))",
            "(terrain: (rock_slope: 0.0))",
            "(terrain: (amplitude_m: -1.0))",
            "(terrain: (grass_depth_m: -5.0))",
            "(terrain: (grass_depth_m: 1e300))",
            "(atmosphere: (height_fraction: 0.0))",
            "(atmosphere: (ozone_width_fraction: 0.0))",
            "(atmosphere: (mie_asymmetry: 1.0))",
            "(atmosphere: (rayleigh_optical_depth: (0.1, -0.1, 0.1)))",
        ] {
            let r = planet(text);
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
            WorldConfig::from_ron_str("(terrain: (octaves: 3))"),
            Err(ConfigError::Parse(_))
        ));
        assert!(matches!(
            planet("(terrain: (height: 1.0))"),
            Err(ConfigError::Parse(_))
        ));
    }
}
