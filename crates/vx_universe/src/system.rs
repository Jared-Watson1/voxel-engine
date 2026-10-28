use std::path::Path;

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};
use vx_core::PlanetParams;

use crate::kepler::{OrbitElements, RotationState};

#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing system config: {0}")]
    Parse(#[from] ron::error::SpannedError),
    #[error("planet parameters: {0}")]
    Planet(#[from] vx_core::ConfigError),
    #[error("{0}")]
    Invalid(String),
}

/// The star (D24). Design defaults are the Sun's values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StarConfig {
    /// Gravitational parameter. Design default 1.32712440018e20 m^3/s^2.
    pub gm_m3_s2: f64,
    /// Radiated power. Design default 3.828e26 W.
    pub luminosity_w: f64,
    /// Photosphere radius, for the sun disc's angular size. Design default 696,000 km.
    pub radius_km: f64,
}

/// An orbit in configuration units: distances in km, angles in degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OrbitConfig {
    pub semi_major_axis_km: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub longitude_of_ascending_node_deg: f64,
    pub argument_of_periapsis_deg: f64,
    pub mean_anomaly_at_epoch_deg: f64,
}

/// The planet the voxel world belongs to: its physical parameters (radius, gravity, rotation,
/// terrain, atmosphere) and its orbit around the star.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlanetConfig {
    pub params: PlanetParams,
    /// Design default 1 AU, eccentricity 0.0167.
    pub orbit: OrbitConfig,
    /// Rotation angle about the body +Y at the epoch. Design default 0.
    pub prime_meridian_at_epoch_deg: f64,
}

/// A moon on rails around the planet, drawn by the sky pass (no voxels in M2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoonConfig {
    /// Design default 3 km.
    pub radius_km: f64,
    /// Bond albedo for its brightness. Design default 0.12.
    pub albedo: f64,
    /// Design default 200 km, near circular, a period near 5 h around the default planet.
    pub orbit: OrbitConfig,
}

/// Configuration of the star system, loaded from `assets/config/system.ron` (D11, D24).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SystemConfig {
    pub star: StarConfig,
    pub planet: PlanetConfig,
    pub moon: Option<MoonConfig>,
    /// Simulation time at startup, seconds since the epoch. Design default 0.
    pub start_time_s: f64,
}

impl SystemConfig {
    pub fn from_ron_str(text: &str) -> Result<Self, SystemError> {
        let config: Self = ron::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Reads and validates the file. Blocking I/O: call from a worker, never the main thread (D9).
    pub fn load(path: &Path) -> Result<Self, SystemError> {
        let text = std::fs::read_to_string(path).map_err(|source| SystemError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_ron_str(&text)
    }

    pub fn validate(&self) -> Result<(), SystemError> {
        self.planet.params.validate()?;
        Ok(())
    }
}

impl Default for StarConfig {
    fn default() -> Self {
        Self {
            gm_m3_s2: 1.327_124_400_18e20,
            luminosity_w: 3.828e26,
            radius_km: 696_000.0,
        }
    }
}

impl Default for OrbitConfig {
    fn default() -> Self {
        Self {
            semi_major_axis_km: 149_597_870.7,
            eccentricity: 0.0167,
            inclination_deg: 0.0,
            longitude_of_ascending_node_deg: 0.0,
            argument_of_periapsis_deg: 0.0,
            mean_anomaly_at_epoch_deg: 0.0,
        }
    }
}

impl Default for PlanetConfig {
    fn default() -> Self {
        Self {
            params: PlanetParams::default(),
            orbit: OrbitConfig::default(),
            prime_meridian_at_epoch_deg: 0.0,
        }
    }
}

impl Default for MoonConfig {
    fn default() -> Self {
        Self {
            radius_km: 3.0,
            albedo: 0.12,
            orbit: OrbitConfig {
                semi_major_axis_km: 200.0,
                eccentricity: 0.01,
                inclination_deg: 5.0,
                ..OrbitConfig::default()
            },
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            star: StarConfig::default(),
            planet: PlanetConfig::default(),
            moon: Some(MoonConfig::default()),
            start_time_s: 0.0,
        }
    }
}

/// The system evaluated at one simulation time, in the planet's body fixed axes where it matters
/// for rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemSnapshot {
    pub time_s: f64,
    /// Unit direction from the planet centre to the star, in body fixed axes.
    pub sun_direction: DVec3,
    pub sun_distance_m: f64,
    /// Irradiance of the star at the planet's distance above the atmosphere, W/m^2 (1361 at 1 AU
    /// for the Sun).
    pub sun_irradiance_w_m2: f64,
    /// Angular radius of the star's disc seen from the planet.
    pub sun_angular_radius_rad: f64,
    /// Moon centre relative to the planet centre in body fixed axes, and its radius.
    pub moon: Option<(DVec3, f64)>,
    /// Rotation from the planet's body fixed axes to system axes.
    pub planet_rotation: DQuat,
    /// Planet centre in the system frame.
    pub planet_position_m: DVec3,
    pub planet_velocity_m_s: DVec3,
}

/// The star, the planet and the moon on rails (D24).
pub struct StarSystem {
    config: SystemConfig,
    planet_orbit: OrbitElements,
    planet_rotation: RotationState,
    moon_orbit: Option<OrbitElements>,
}

impl StarSystem {
    pub fn new(config: SystemConfig) -> Self {
        let _ = config;
        todo!("workstream A")
    }

    pub fn config(&self) -> &SystemConfig {
        &self.config
    }

    pub fn planet_orbit(&self) -> &OrbitElements {
        &self.planet_orbit
    }

    pub fn planet_rotation(&self) -> &RotationState {
        &self.planet_rotation
    }

    pub fn moon_orbit(&self) -> Option<&OrbitElements> {
        self.moon_orbit.as_ref()
    }

    pub fn snapshot(&self, t_s: f64) -> SystemSnapshot {
        let _ = t_s;
        todo!("workstream A")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_system_ron_holds_the_design_defaults() {
        let text = include_str!("../../../assets/config/system.ron");
        assert_eq!(
            SystemConfig::from_ron_str(text).unwrap(),
            SystemConfig::default()
        );
    }
}
