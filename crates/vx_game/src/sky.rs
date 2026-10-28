//! The star system on rails (D24) as the game uses it: the simulation clock with time warp, the
//! sun and the moon in the planet's body fixed axes for the renderer, and local solar time.

use glam::{DVec3, Vec3};
use vx_render::{MoonLight, SunLight};
use vx_universe::{SimClock, StarSystem, SystemConfig, SystemSnapshot};

/// Irradiance that the renderer's sun illuminance of 1.0 stands for: the Sun at 1 AU, W/m^2.
const REFERENCE_IRRADIANCE_W_M2: f64 = 1361.0;
/// Warp steps for the time keys: real time up to a day per 36 s at the 1 h design default day.
pub const WARP_STEPS: [f64; 7] = [0.0, 1.0, 10.0, 60.0, 300.0, 1000.0, 10_000.0];

pub struct Sky {
    clock: SimClock,
    system: StarSystem,
    snapshot: SystemSnapshot,
    warp_index: usize,
}

impl Sky {
    pub fn new(config: SystemConfig) -> Self {
        let clock = SimClock::new(config.start_time_s);
        let system = StarSystem::new(config);
        let snapshot = system.snapshot(clock.time_s());
        Self {
            clock,
            system,
            snapshot,
            warp_index: 1,
        }
    }

    pub fn config(&self) -> &SystemConfig {
        self.system.config()
    }

    pub fn snapshot(&self) -> &SystemSnapshot {
        &self.snapshot
    }

    pub fn time_s(&self) -> f64 {
        self.clock.time_s()
    }

    pub fn warp(&self) -> f64 {
        self.clock.warp()
    }

    /// Steps the warp up (`+1`) or down (`-1`) through [`WARP_STEPS`]; the first step pauses.
    pub fn step_warp(&mut self, delta: i32) {
        let last = WARP_STEPS.len() as i32 - 1;
        self.warp_index = (self.warp_index as i32 + delta).clamp(0, last) as usize;
        self.clock.set_warp(WARP_STEPS[self.warp_index]);
    }

    /// Jumps the clock to `time_s` seconds since the epoch.
    pub fn set_time_s(&mut self, time_s: f64) {
        self.clock.set_time_s(time_s);
        self.snapshot = self.system.snapshot(time_s);
    }

    /// Sets an arbitrary warp (for automation); the warp keys then step from the nearest step.
    pub fn set_warp(&mut self, warp: f64) {
        self.clock.set_warp(warp);
        self.warp_index = WARP_STEPS
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - warp).abs().total_cmp(&(b.1 - warp).abs()))
            .map_or(1, |(i, _)| i);
    }

    pub fn advance(&mut self, real_dt_s: f64) {
        self.clock.advance(real_dt_s);
        self.snapshot = self.system.snapshot(self.clock.time_s());
    }

    /// Jumps the clock so the local solar time at body fixed `position` is `time_of_day` (0 is
    /// midnight, 0.25 sunrise at the equinox, 0.5 noon), searching forward within one day.
    pub fn set_local_time(&mut self, position: DVec3, time_of_day: f64) {
        let period = self.config().planet.params.rotation_period_s.abs();
        let start = self.clock.time_s();
        let steps = 720;
        let mut best = (f64::INFINITY, start);
        for i in 0..steps {
            let t = start + period * f64::from(i) / f64::from(steps);
            let snapshot = self.system.snapshot(t);
            let d = wrap_half(local_solar_time(&snapshot, position) - time_of_day).abs();
            if d < best.0 {
                best = (d, t);
            }
        }
        self.clock.set_time_s(best.1);
        self.snapshot = self.system.snapshot(best.1);
    }

    pub fn sun_light(&self) -> SunLight {
        let s = &self.snapshot;
        let relative = (s.sun_irradiance_w_m2 / REFERENCE_IRRADIANCE_W_M2) as f32;
        SunLight {
            direction: s.sun_direction.as_vec3(),
            illuminance: Vec3::splat(relative),
            angular_radius_rad: s.sun_angular_radius_rad as f32,
        }
    }

    /// The moon relative to the camera at body fixed `camera_m`.
    pub fn moon_light(&self, camera_m: DVec3) -> Option<MoonLight> {
        let (centre, radius_m) = self.snapshot.moon?;
        let albedo = self.config().moon.map_or(0.12, |m| m.albedo);
        Some(MoonLight {
            centre_m: (centre - camera_m).as_vec3(),
            radius_m: radius_m as f32,
            albedo: albedo as f32,
        })
    }

    pub fn local_time(&self, position: DVec3) -> f64 {
        local_solar_time(&self.snapshot, position)
    }
}

/// Local solar time at body fixed `position` as a fraction of a day (0 midnight, 0.5 noon): the
/// angle about the rotation axis (+Y) from the antisolar meridian, in the sense the surface turns
/// (positive spin about +Y carries a point towards the sun in the morning).
pub fn local_solar_time(snapshot: &SystemSnapshot, position: DVec3) -> f64 {
    let sun = snapshot.sun_direction;
    let s = DVec3::new(sun.x, 0.0, sun.z);
    let p = DVec3::new(position.x, 0.0, position.z);
    if s.length_squared() < 1.0e-18 || p.length_squared() < 1.0e-18 {
        return 0.5;
    }
    // Angle from the point to the sun about +Y, positive when the sun lies ahead in the spin.
    let ahead = p.cross(s).y.atan2(p.dot(s));
    (0.5 - ahead / std::f64::consts::TAU).rem_euclid(1.0)
}

fn wrap_half(x: f64) -> f64 {
    (x + 0.5).rem_euclid(1.0) - 0.5
}

/// Formats a fraction of a day as hours and minutes.
pub fn clock_text(time_of_day: f64) -> String {
    let minutes = (time_of_day.rem_euclid(1.0) * 1440.0).round() as u32 % 1440;
    format!("{:02}:{:02}", minutes / 60, minutes % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DQuat;

    fn snapshot_with_sun(sun: DVec3) -> SystemSnapshot {
        SystemSnapshot {
            time_s: 0.0,
            sun_direction: sun.normalize(),
            sun_distance_m: 1.5e11,
            sun_irradiance_w_m2: 1361.0,
            sun_angular_radius_rad: 0.00465,
            moon: None,
            planet_rotation: DQuat::IDENTITY,
            planet_position_m: DVec3::ZERO,
            planet_velocity_m_s: DVec3::ZERO,
        }
    }

    #[test]
    fn noon_midnight_and_the_morning_side() {
        let s = snapshot_with_sun(DVec3::X);
        assert!((local_solar_time(&s, DVec3::X * 10.0) - 0.5).abs() < 1.0e-12);
        assert!((local_solar_time(&s, DVec3::NEG_X * 10.0) - 0.0).abs() < 1.0e-12);
        // Spin about +Y carries +Z towards +X (a rotation by +90 degrees about Y maps Z to X),
        // so +Z sees the sun rise: morning.
        let rot = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
        assert!((rot * DVec3::Z - DVec3::X).length() < 1.0e-12);
        assert!((local_solar_time(&s, DVec3::Z * 10.0) - 0.25).abs() < 1.0e-12);
        assert!((local_solar_time(&s, DVec3::NEG_Z * 10.0) - 0.75).abs() < 1.0e-12);
    }

    #[test]
    fn clock_text_formats_hours_and_minutes() {
        assert_eq!(clock_text(0.25), "06:00");
        assert_eq!(clock_text(0.5 + 1.0 / 1440.0), "12:01");
        assert_eq!(clock_text(1.0), "00:00");
    }
}
