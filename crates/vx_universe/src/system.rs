use std::f64::consts::PI;
use std::ops::RangeInclusive;
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
    /// Range [`StarConfig::GM_RANGE_M3_S2`].
    pub gm_m3_s2: f64,
    /// Radiated power. Design default 3.828e26 W. Range [`StarConfig::LUMINOSITY_RANGE_W`].
    pub luminosity_w: f64,
    /// Photosphere radius, for the sun disc's angular size. Design default 696,000 km.
    /// Range [`StarConfig::RADIUS_RANGE_KM`].
    pub radius_km: f64,
}

/// An orbit in configuration units: distances in km, angles in degrees, measured against the
/// system axes (see [`OrbitElements`] for the orientation convention).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OrbitConfig {
    /// Range [`OrbitConfig::SEMI_MAJOR_AXIS_RANGE_KM`].
    pub semi_major_axis_km: f64,
    /// Range [`OrbitConfig::ECCENTRICITY_RANGE`]; hyperbolic orbits wait for the ship in M6 (D24).
    pub eccentricity: f64,
    /// Range [`OrbitConfig::INCLINATION_RANGE_DEG`]; above 90 degrees the orbit is retrograde.
    pub inclination_deg: f64,
    /// Range [`OrbitConfig::ANGLE_RANGE_DEG`].
    pub longitude_of_ascending_node_deg: f64,
    /// Range [`OrbitConfig::ANGLE_RANGE_DEG`].
    pub argument_of_periapsis_deg: f64,
    /// Range [`OrbitConfig::ANGLE_RANGE_DEG`].
    pub mean_anomaly_at_epoch_deg: f64,
}

/// The planet the voxel world belongs to: its physical parameters (radius, gravity, rotation,
/// terrain, atmosphere) and its orbit around the star.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlanetConfig {
    pub params: PlanetParams,
    /// Design default 1 AU, eccentricity 0.0167. The periapsis must clear the star's surface.
    pub orbit: OrbitConfig,
    /// Rotation angle about the body +Y at the epoch. Design default 0, which with the default
    /// orbit puts the prime meridian at local midnight at time 0 (the star lies along body -X).
    /// Range [`OrbitConfig::ANGLE_RANGE_DEG`].
    pub prime_meridian_at_epoch_deg: f64,
}

/// A moon on rails around the planet, drawn by the sky pass (no voxels in M2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoonConfig {
    /// Design default 3 km. Range [`MoonConfig::RADIUS_RANGE_KM`].
    pub radius_km: f64,
    /// Bond albedo for its brightness. Design default 0.12. Range [`MoonConfig::ALBEDO_RANGE`].
    pub albedo: f64,
    /// Design default 200 km, near circular, a period near 5 h around the default planet. The
    /// periapsis must clear the planet's atmosphere and the apoapsis must stay inside the planet's
    /// Hill sphere at its own periapsis.
    pub orbit: OrbitConfig,
}

/// Configuration of the star system, loaded from `assets/config/system.ron` (D11, D24).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SystemConfig {
    pub star: StarConfig,
    pub planet: PlanetConfig,
    pub moon: Option<MoonConfig>,
    /// Simulation time at startup, seconds since the epoch. Design default 0. Any finite value.
    pub start_time_s: f64,
}

impl StarConfig {
    /// From a brown dwarf (about 1e-2 solar masses) to a massive star (about 100 solar masses).
    pub const GM_RANGE_M3_S2: RangeInclusive<f64> = 1.0e18..=1.0e22;
    pub const LUMINOSITY_RANGE_W: RangeInclusive<f64> = 1.0e20..=1.0e31;
    pub const RADIUS_RANGE_KM: RangeInclusive<f64> = 1.0e3..=1.0e7;

    pub fn validate(&self) -> Result<(), SystemError> {
        check("star.gm_m3_s2", self.gm_m3_s2, &Self::GM_RANGE_M3_S2)?;
        check(
            "star.luminosity_w",
            self.luminosity_w,
            &Self::LUMINOSITY_RANGE_W,
        )?;
        check("star.radius_km", self.radius_km, &Self::RADIUS_RANGE_KM)
    }
}

impl OrbitConfig {
    pub const SEMI_MAJOR_AXIS_RANGE_KM: RangeInclusive<f64> = 1.0..=1.0e10;
    pub const ECCENTRICITY_RANGE: RangeInclusive<f64> = 0.0..=0.99;
    pub const INCLINATION_RANGE_DEG: RangeInclusive<f64> = 0.0..=180.0;
    pub const ANGLE_RANGE_DEG: RangeInclusive<f64> = -360.0..=360.0;

    /// Checks the ranges of every field; `which` prefixes the field names in errors.
    pub fn validate(&self, which: &str) -> Result<(), SystemError> {
        let field = |name: &str| format!("{which}.{name}");
        check(
            &field("semi_major_axis_km"),
            self.semi_major_axis_km,
            &Self::SEMI_MAJOR_AXIS_RANGE_KM,
        )?;
        check(
            &field("eccentricity"),
            self.eccentricity,
            &Self::ECCENTRICITY_RANGE,
        )?;
        check(
            &field("inclination_deg"),
            self.inclination_deg,
            &Self::INCLINATION_RANGE_DEG,
        )?;
        check(
            &field("longitude_of_ascending_node_deg"),
            self.longitude_of_ascending_node_deg,
            &Self::ANGLE_RANGE_DEG,
        )?;
        check(
            &field("argument_of_periapsis_deg"),
            self.argument_of_periapsis_deg,
            &Self::ANGLE_RANGE_DEG,
        )?;
        check(
            &field("mean_anomaly_at_epoch_deg"),
            self.mean_anomaly_at_epoch_deg,
            &Self::ANGLE_RANGE_DEG,
        )
    }

    /// The same orbit in SI units (metres, radians).
    pub fn to_elements(&self) -> OrbitElements {
        OrbitElements {
            semi_major_axis_m: self.semi_major_axis_km * 1000.0,
            eccentricity: self.eccentricity,
            inclination_rad: self.inclination_deg.to_radians(),
            longitude_of_ascending_node_rad: self.longitude_of_ascending_node_deg.to_radians(),
            argument_of_periapsis_rad: self.argument_of_periapsis_deg.to_radians(),
            mean_anomaly_at_epoch_rad: self.mean_anomaly_at_epoch_deg.to_radians(),
        }
    }

    pub fn periapsis_km(&self) -> f64 {
        self.semi_major_axis_km * (1.0 - self.eccentricity)
    }

    pub fn apoapsis_km(&self) -> f64 {
        self.semi_major_axis_km * (1.0 + self.eccentricity)
    }
}

impl MoonConfig {
    pub const RADIUS_RANGE_KM: RangeInclusive<f64> = 0.1..=3000.0;
    pub const ALBEDO_RANGE: RangeInclusive<f64> = 0.0..=1.0;
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

    /// Checks every field's range, then the geometry: the planet's periapsis clears the star,
    /// the moon's periapsis clears the planet's atmosphere and the moon's apoapsis stays inside
    /// the planet's Hill sphere `r_p (m / 3 M)^(1/3)` evaluated at the planet's periapsis.
    pub fn validate(&self) -> Result<(), SystemError> {
        self.star.validate()?;
        self.planet.params.validate()?;
        self.planet.orbit.validate("planet.orbit")?;
        check(
            "planet.prime_meridian_at_epoch_deg",
            self.planet.prime_meridian_at_epoch_deg,
            &OrbitConfig::ANGLE_RANGE_DEG,
        )?;
        if !self.start_time_s.is_finite() {
            return Err(SystemError::Invalid(format!(
                "start_time_s = {} must be finite",
                self.start_time_s
            )));
        }

        let planet_radius_km = self.planet.params.radius_km;
        let planet_clearance_km = self.star.radius_km + planet_radius_km;
        if self.planet.orbit.periapsis_km() <= planet_clearance_km {
            return Err(SystemError::Invalid(format!(
                "planet.orbit periapsis {} km must exceed the star radius plus the planet radius \
                 ({planet_clearance_km} km)",
                self.planet.orbit.periapsis_km()
            )));
        }

        if let Some(moon) = &self.moon {
            check(
                "moon.radius_km",
                moon.radius_km,
                &MoonConfig::RADIUS_RANGE_KM,
            )?;
            check("moon.albedo", moon.albedo, &MoonConfig::ALBEDO_RANGE)?;
            moon.orbit.validate("moon.orbit")?;

            let params = &self.planet.params;
            let atmosphere_top_km = params.atmosphere.top_m(params.radius_m()) / 1000.0;
            let moon_clearance_km = planet_radius_km + atmosphere_top_km + moon.radius_km;
            if moon.orbit.periapsis_km() <= moon_clearance_km {
                return Err(SystemError::Invalid(format!(
                    "moon.orbit periapsis {} km must exceed the planet radius plus its atmosphere \
                     plus the moon radius ({moon_clearance_km} km)",
                    moon.orbit.periapsis_km()
                )));
            }
            let hill_km = self.planet.orbit.periapsis_km()
                * (params.gm_m3_s2() / (3.0 * self.star.gm_m3_s2)).cbrt();
            if moon.orbit.apoapsis_km() >= hill_km {
                return Err(SystemError::Invalid(format!(
                    "moon.orbit apoapsis {} km must stay inside the planet's Hill sphere \
                     ({hill_km} km)",
                    moon.orbit.apoapsis_km()
                )));
            }
        }
        Ok(())
    }
}

fn check(field: &str, value: f64, range: &RangeInclusive<f64>) -> Result<(), SystemError> {
    if value.is_finite() && range.contains(&value) {
        Ok(())
    } else {
        Err(SystemError::Invalid(format!(
            "{field} = {value} is outside the supported range {} to {}",
            range.start(),
            range.end()
        )))
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
///
/// The star sits at the system origin and the planet orbits it with the star's gravitational
/// parameter alone: the barycentre offset and the planet's own mass in the two body problem are
/// ignored in M2 (for the default planet they are below 1e-11 of the star's). The moon orbits the
/// planet with the planet's `g R^2` and its elements are measured against the system axes, the
/// planet's body centred inertial frame.
pub struct StarSystem {
    config: SystemConfig,
    planet_orbit: OrbitElements,
    planet_rotation: RotationState,
    moon_orbit: Option<OrbitElements>,
}

impl StarSystem {
    /// Builds the rails from a configuration that has passed [`SystemConfig::validate`] (an
    /// invalid one, such as an eccentricity of 1, produces non finite snapshots).
    pub fn new(config: SystemConfig) -> Self {
        debug_assert!(
            config.validate().is_ok(),
            "StarSystem::new needs a validated SystemConfig"
        );
        let params = &config.planet.params;
        let planet_rotation = RotationState {
            sidereal_period_s: params.rotation_period_s,
            axial_tilt_rad: params.axial_tilt_deg.to_radians(),
            prime_meridian_at_epoch_rad: config.planet.prime_meridian_at_epoch_deg.to_radians(),
        };
        Self {
            planet_orbit: config.planet.orbit.to_elements(),
            planet_rotation,
            moon_orbit: config.moon.map(|moon| moon.orbit.to_elements()),
            config,
        }
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

    /// Evaluates every body at `t_s`. A pure function of the configuration and `t_s`, so the same
    /// time gives bit identical snapshots.
    pub fn snapshot(&self, t_s: f64) -> SystemSnapshot {
        let (planet_position_m, planet_velocity_m_s) =
            self.planet_orbit.state_at(self.config.star.gm_m3_s2, t_s);
        let rotation = self.planet_rotation.body_to_inertial(t_s);
        let inertial_to_body = rotation.inverse();

        let sun_distance_m = planet_position_m.length();
        let sun_direction = inertial_to_body * (-planet_position_m / sun_distance_m);
        let sun_irradiance_w_m2 =
            self.config.star.luminosity_w / (4.0 * PI * sun_distance_m * sun_distance_m);
        let sun_radius_m = self.config.star.radius_km * 1000.0;
        let sun_angular_radius_rad = (sun_radius_m / sun_distance_m).min(1.0).asin();

        let moon =
            self.config
                .moon
                .as_ref()
                .zip(self.moon_orbit.as_ref())
                .map(|(moon_config, orbit)| {
                    let (moon_m, _) = orbit.state_at(self.config.planet.params.gm_m3_s2(), t_s);
                    (inertial_to_body * moon_m, moon_config.radius_km * 1000.0)
                });

        SystemSnapshot {
            time_s: t_s,
            sun_direction,
            sun_distance_m,
            sun_irradiance_w_m2,
            sun_angular_radius_rad,
            moon,
            planet_rotation: rotation,
            planet_position_m,
            planet_velocity_m_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;

    const AU_M: f64 = 149_597_870_700.0;

    #[test]
    fn shipped_system_ron_holds_the_design_defaults() {
        let text = include_str!("../../../assets/config/system.ron");
        assert_eq!(
            SystemConfig::from_ron_str(text).unwrap(),
            SystemConfig::default()
        );
    }

    fn system_with(edit: impl FnOnce(&mut SystemConfig)) -> StarSystem {
        let mut config = SystemConfig::default();
        edit(&mut config);
        config.validate().unwrap();
        StarSystem::new(config)
    }

    fn year_s(system: &StarSystem) -> f64 {
        system
            .planet_orbit()
            .period_s(system.config().star.gm_m3_s2)
    }

    fn subsolar_latitude_deg(snapshot: &SystemSnapshot) -> f64 {
        snapshot.sun_direction.y.asin().to_degrees()
    }

    /// East positive: body +Y is north and +X the prime meridian, so east at the prime meridian
    /// is -Z.
    fn longitude_rad(dir: DVec3) -> f64 {
        (-dir.z).atan2(dir.x)
    }

    #[test]
    fn irradiance_and_sun_size_at_one_au() {
        let circular = system_with(|c| c.planet.orbit.eccentricity = 0.0);
        let s = circular.snapshot(12_345.0);
        assert!((s.sun_distance_m - AU_M).abs() < 1.0);
        assert!(
            (s.sun_irradiance_w_m2 - 1361.0).abs() < 1.0,
            "{}",
            s.sun_irradiance_w_m2
        );
        assert!(
            (s.sun_angular_radius_rad - 0.004_65).abs() < 1.0e-5,
            "{}",
            s.sun_angular_radius_rad
        );

        let default = StarSystem::new(SystemConfig::default());
        let year_s = year_s(&default);
        assert!((year_s / 86_400.0 - 365.256).abs() < 0.01, "{year_s}");
        for k in 0..12 {
            let s = default.snapshot(f64::from(k) / 12.0 * year_s);
            let at_one_au = s.sun_irradiance_w_m2 * (s.sun_distance_m / AU_M).powi(2);
            assert!((at_one_au - 1361.17).abs() < 0.05, "{at_one_au}");
        }
        let perihelion = default.snapshot(0.0);
        assert!((perihelion.sun_distance_m - AU_M * (1.0 - 0.0167)).abs() < 10.0);
    }

    #[test]
    fn sun_crosses_the_equatorial_horizon_twice_per_sidereal_day() {
        let system = StarSystem::new(SystemConfig::default());
        let rotation = *system.planet_rotation();
        let day_s = rotation.sidereal_period_s;
        let t0_s = 0.137 * day_s;
        let fixed_sun = rotation.body_to_inertial(t0_s) * system.snapshot(t0_s).sun_direction;
        let samples = 3600;
        for longitude_deg in [0.0_f64, 37.0, 90.0, 200.0] {
            let lon = longitude_deg.to_radians();
            let up = DVec3::new(lon.cos(), 0.0, -lon.sin());
            let mut crossings_fixed = 0;
            let mut crossings_moving = 0;
            let mut previous: Option<(f64, f64)> = None;
            for k in 0..=samples {
                let t_s = t0_s + f64::from(k) / f64::from(samples) * day_s;
                let fixed = rotation.body_to_inertial(t_s).inverse() * fixed_sun;
                let moving = system.snapshot(t_s).sun_direction;
                let now = (up.dot(fixed), up.dot(moving));
                if let Some(before) = previous {
                    crossings_fixed += usize::from((before.0 > 0.0) != (now.0 > 0.0));
                    crossings_moving += usize::from((before.1 > 0.0) != (now.1 > 0.0));
                }
                previous = Some(now);
            }
            assert_eq!(crossings_fixed, 2, "longitude {longitude_deg}");
            assert_eq!(crossings_moving, 2, "longitude {longitude_deg}");
        }
    }

    #[test]
    fn subsolar_point_moves_west() {
        let system = StarSystem::new(SystemConfig::default());
        let day_s = system.planet_rotation().sidereal_period_s;
        let mut previous = longitude_rad(system.snapshot(0.0).sun_direction);
        let mut total_rad = 0.0;
        for k in 1..=360 {
            let lon = longitude_rad(system.snapshot(f64::from(k) / 360.0 * day_s).sun_direction);
            let step = (lon - previous + PI).rem_euclid(TAU) - PI;
            assert!(step < 0.0, "step {k} moved east by {step}");
            total_rad += step;
            previous = lon;
        }
        // One full turn west per sidereal day, less the planet's orbital motion (about 0.04 deg).
        assert!((total_rad + TAU).abs() < 1.0e-3, "{total_rad}");
    }

    #[test]
    fn default_epoch_puts_the_sun_over_longitude_180() {
        let s = StarSystem::new(SystemConfig::default()).snapshot(0.0);
        assert!((s.sun_direction - DVec3::NEG_X).length() < 1.0e-12);
    }

    fn solstices(system: &StarSystem) -> ((f64, f64), (f64, f64)) {
        let year_s = year_s(system);
        let samples = 4000;
        let mut max = (f64::MIN, 0.0);
        let mut min = (f64::MAX, 0.0);
        for k in 0..samples {
            let t_s = f64::from(k) / f64::from(samples) * year_s;
            let lat = subsolar_latitude_deg(&system.snapshot(t_s));
            if lat > max.0 {
                max = (lat, t_s);
            }
            if lat < min.0 {
                min = (lat, t_s);
            }
        }
        (max, min)
    }

    #[test]
    fn seasons_follow_the_tilt() {
        for tilt_deg in [15.0, 23.44, 60.0] {
            let system = system_with(|c| c.planet.params.axial_tilt_deg = tilt_deg);
            let year_s = year_s(&system);
            let ((lat_max, t_max), (lat_min, t_min)) = solstices(&system);
            assert!(
                (lat_max - tilt_deg).abs() < 0.05,
                "tilt {tilt_deg}: {lat_max}"
            );
            assert!(
                (lat_min + tilt_deg).abs() < 0.05,
                "tilt {tilt_deg}: {lat_min}"
            );
            let apart = (t_max - t_min).abs() / year_s;
            assert!((apart - 0.5).abs() < 0.03, "tilt {tilt_deg}: {apart}");
            let equinox = subsolar_latitude_deg(&system.snapshot(0.0));
            assert!(equinox.abs() < 1.0e-9, "time 0 is an equinox");
        }
    }

    #[test]
    fn zero_tilt_has_no_seasons() {
        let system = system_with(|c| c.planet.params.axial_tilt_deg = 0.0);
        let year_s = year_s(&system);
        for k in 0..400 {
            let s = system.snapshot(f64::from(k) / 400.0 * year_s + 17.0);
            assert!(subsolar_latitude_deg(&s).abs() < 1.0e-9);
        }
    }

    #[test]
    fn default_moon_period_is_near_five_hours() {
        let system = StarSystem::new(SystemConfig::default());
        let gm = system.config().planet.params.gm_m3_s2();
        assert_eq!(gm, 9.81e8);
        let period_s = system.moon_orbit().unwrap().period_s(gm);
        assert!((period_s - 17_942.8).abs() < 1.0, "{period_s}");
        assert!((period_s / 3600.0 - 5.0).abs() < 0.1);
    }

    #[test]
    fn moon_is_reported_in_body_fixed_axes() {
        let system = StarSystem::new(SystemConfig::default());
        let gm = system.config().planet.params.gm_m3_s2();
        let orbit = *system.moon_orbit().unwrap();
        for t_s in [0.0, 1000.0, 7777.7, 1.0e7] {
            let s = system.snapshot(t_s);
            let (moon_body, radius_m) = s.moon.unwrap();
            assert_eq!(radius_m, 3000.0);
            let (moon_inertial, _) = orbit.state_at(gm, t_s);
            assert!((s.planet_rotation * moon_body - moon_inertial).length() < 1.0e-6);
            let r = moon_body.length();
            assert!(
                (198_000.0 - 1.0e-6..=202_000.0 + 1.0e-6).contains(&r),
                "{r}"
            );
        }
        let no_moon = system_with(|c| c.moon = None);
        assert!(no_moon.snapshot(0.0).moon.is_none());
    }

    #[test]
    fn snapshot_frames_are_consistent() {
        let system = StarSystem::new(SystemConfig::default());
        for t_s in [0.0, 5.0e6, 2.0e7, 3.0e9] {
            let s = system.snapshot(t_s);
            assert!((s.sun_direction.length() - 1.0).abs() < 1.0e-12);
            assert!((s.planet_position_m.length() - s.sun_distance_m).abs() < 1.0e-3);
            let sun_system = s.planet_rotation * s.sun_direction;
            assert!((sun_system + s.planet_position_m.normalize()).length() < 1.0e-12);
            let orbital_speed = s.planet_velocity_m_s.length();
            assert!(
                (29_000.0..31_000.0).contains(&orbital_speed),
                "{orbital_speed}"
            );
            assert!(
                s.planet_position_m
                    .cross(s.planet_velocity_m_s)
                    .normalize()
                    .dot(DVec3::Y)
                    > 0.999_999
            );
        }
    }

    #[test]
    fn snapshots_are_deterministic() {
        let a = StarSystem::new(SystemConfig::default());
        let b = StarSystem::new(SystemConfig::default());
        for t_s in [0.0, 0.1, 1234.567, 1.0e8, -5.0e5] {
            let (sa, sb) = (a.snapshot(t_s), b.snapshot(t_s));
            assert_eq!(sa, sb);
            assert_eq!(
                sa.sun_direction.to_array().map(f64::to_bits),
                sb.sun_direction.to_array().map(f64::to_bits)
            );
            assert_eq!(
                sa.moon.unwrap().0.to_array().map(f64::to_bits),
                sb.moon.unwrap().0.to_array().map(f64::to_bits)
            );
            assert_eq!(
                sa.sun_irradiance_w_m2.to_bits(),
                a.snapshot(t_s).sun_irradiance_w_m2.to_bits()
            );
        }
    }

    type Edit = Box<dyn Fn(&mut SystemConfig)>;

    #[test]
    fn validation_accepts_the_ends_of_each_range() {
        let ends: Vec<Edit> = vec![
            Box::new(|c| c.planet.orbit.eccentricity = 0.0),
            Box::new(|c| {
                c.planet.orbit.eccentricity = 0.99;
                c.moon = None;
            }),
            Box::new(|c| c.planet.orbit.inclination_deg = 0.0),
            Box::new(|c| c.planet.orbit.inclination_deg = 180.0),
            Box::new(|c| c.planet.orbit.longitude_of_ascending_node_deg = -360.0),
            Box::new(|c| c.planet.orbit.argument_of_periapsis_deg = 360.0),
            Box::new(|c| c.planet.orbit.semi_major_axis_km = 1.0e10),
            Box::new(|c| c.planet.prime_meridian_at_epoch_deg = -360.0),
            Box::new(|c| c.star.gm_m3_s2 = 1.0e18),
            Box::new(|c| c.star.gm_m3_s2 = 1.0e22),
            Box::new(|c| c.star.luminosity_w = 1.0e20),
            Box::new(|c| c.star.luminosity_w = 1.0e31),
            Box::new(|c| c.star.radius_km = 1.0e3),
            Box::new(|c| c.star.radius_km = 1.0e7),
            Box::new(|c| c.planet.params.axial_tilt_deg = 180.0),
            Box::new(|c| c.moon = None),
            Box::new(|c| c.moon.as_mut().unwrap().albedo = 0.0),
            Box::new(|c| c.moon.as_mut().unwrap().albedo = 1.0),
            Box::new(|c| c.moon.as_mut().unwrap().radius_km = 0.1),
            Box::new(|c| c.moon.as_mut().unwrap().orbit.eccentricity = 0.0),
            Box::new(|c| {
                let moon = c.moon.as_mut().unwrap();
                moon.orbit.semi_major_axis_km = 2000.0;
                moon.orbit.eccentricity = 0.99;
            }),
            Box::new(|c| {
                c.planet.params.radius_km = 6371.0;
                let moon = c.moon.as_mut().unwrap();
                moon.radius_km = 3000.0;
                moon.orbit.semi_major_axis_km = 384_400.0;
                moon.orbit.inclination_deg = 180.0;
            }),
        ];
        for (k, edit) in ends.iter().enumerate() {
            let mut config = SystemConfig::default();
            edit(&mut config);
            if let Err(error) = config.validate() {
                panic!("case {k}: {error}");
            }
            let s = StarSystem::new(config).snapshot(4321.0);
            assert!(s.sun_direction.is_finite() && s.sun_irradiance_w_m2.is_finite());
        }
    }

    #[test]
    fn validation_rejects_bad_values() {
        let bad: Vec<(&str, Edit)> = vec![
            (
                "planet.orbit.eccentricity",
                Box::new(|c| c.planet.orbit.eccentricity = 1.0),
            ),
            (
                "planet.orbit.eccentricity",
                Box::new(|c| c.planet.orbit.eccentricity = -0.1),
            ),
            (
                "planet.orbit.semi_major_axis_km",
                Box::new(|c| c.planet.orbit.semi_major_axis_km = 0.0),
            ),
            (
                "planet.orbit.semi_major_axis_km",
                Box::new(|c| c.planet.orbit.semi_major_axis_km = -5.0),
            ),
            (
                "planet.orbit.inclination_deg",
                Box::new(|c| c.planet.orbit.inclination_deg = f64::NAN),
            ),
            (
                "planet.orbit.mean_anomaly_at_epoch_deg",
                Box::new(|c| c.planet.orbit.mean_anomaly_at_epoch_deg = f64::INFINITY),
            ),
            (
                "planet.prime_meridian_at_epoch_deg",
                Box::new(|c| c.planet.prime_meridian_at_epoch_deg = 400.0),
            ),
            ("star.gm_m3_s2", Box::new(|c| c.star.gm_m3_s2 = 0.0)),
            (
                "star.luminosity_w",
                Box::new(|c| c.star.luminosity_w = -1.0),
            ),
            ("star.radius_km", Box::new(|c| c.star.radius_km = 0.0)),
            (
                "moon.albedo",
                Box::new(|c| c.moon.as_mut().unwrap().albedo = 1.1),
            ),
            (
                "moon.albedo",
                Box::new(|c| c.moon.as_mut().unwrap().albedo = -0.01),
            ),
            (
                "moon.radius_km",
                Box::new(|c| c.moon.as_mut().unwrap().radius_km = 0.0),
            ),
            (
                "moon.orbit.eccentricity",
                Box::new(|c| c.moon.as_mut().unwrap().orbit.eccentricity = 0.995),
            ),
            ("start_time_s", Box::new(|c| c.start_time_s = f64::NAN)),
            (
                "moon.orbit periapsis",
                Box::new(|c| c.moon.as_mut().unwrap().orbit.semi_major_axis_km = 12.0),
            ),
            (
                "moon.orbit apoapsis",
                Box::new(|c| c.moon.as_mut().unwrap().orbit.semi_major_axis_km = 50_000.0),
            ),
            (
                "planet.orbit periapsis",
                Box::new(|c| c.planet.orbit.semi_major_axis_km = 500_000.0),
            ),
            (
                "planet.radius_km",
                Box::new(|c| c.planet.params.radius_km = 0.0),
            ),
        ];
        for (field, edit) in bad {
            let mut config = SystemConfig::default();
            edit(&mut config);
            let error = config.validate().expect_err(field).to_string();
            assert!(error.contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn ron_errors_are_reported() {
        let error = SystemConfig::from_ron_str("(moon: Some((albedo: 2.0)))")
            .unwrap_err()
            .to_string();
        assert!(error.contains("moon.albedo"), "{error}");
        assert!(SystemConfig::from_ron_str("(planet: (bogus: 1.0))").is_err());
        assert_eq!(
            SystemConfig::from_ron_str("()").unwrap(),
            SystemConfig::default()
        );
    }
}
