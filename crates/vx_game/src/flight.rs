//! Scripted flights for measurements, so the exit runs need nobody at the keyboard: a square loop
//! over the planet's surface at a fixed height (the M1 run, `--flight-km`), and the M2 descent:
//! standing on the surface, a climb straight up to orbit, a hold, and the way back down
//! (`--descent`).
//!
//! Heights come from a function of the body fixed direction (the sampler's full detail height
//! above the mean radius), so the flights run before any chunk is generated.

use glam::DVec3;
use vx_core::WorldPos;

/// Height of the loop above the terrain, as a multiple of the level 0 chunk extent.
const LOOP_HEIGHT_CHUNK_EXTENTS: f64 = 2.0;
/// Downward pitch of the loop's view in radians (about 18 degrees).
const LOOP_PITCH_RAD: f64 = -0.32;
/// Time constant in seconds of the loop's height smoothing, so ridges do not throw the camera.
const HEIGHT_SMOOTHING_S: f64 = 1.5;
/// Eye height above the terrain when standing.
pub const EYE_HEIGHT_M: f64 = 1.8;
/// Seconds standing on the surface before the climb and after the landing.
const DESCENT_GROUND_HOLD_S: f64 = 3.0;
/// Seconds at the top of the climb.
const DESCENT_TOP_HOLD_S: f64 = 4.0;
/// Pitch of the view while standing (slightly below the horizon) and at the top (nearly
/// straight down, so the whole planet and its limb are in view).
const DESCENT_GROUND_PITCH_RAD: f64 = -0.08;
const DESCENT_TOP_PITCH_RAD: f64 = -1.45;

pub enum Flight {
    Loop(SurfaceLoop),
    Descent(Descent),
}

impl Flight {
    pub fn duration_s(&self) -> f64 {
        match self {
            Self::Loop(f) => f.total_m() / f.speed_m_s,
            Self::Descent(d) => d.duration_s(),
        }
    }

    pub fn done(&self) -> bool {
        self.progress() >= 1.0
    }

    pub fn progress(&self) -> f64 {
        match self {
            Self::Loop(f) => (f.distance_m / f.total_m()).clamp(0.0, 1.0),
            Self::Descent(d) => (d.time_s / d.duration_s()).clamp(0.0, 1.0),
        }
    }

    /// Advances the flight and returns the camera position and a point to look at.
    pub fn advance(&mut self, dt_s: f64, height_m: impl Fn(DVec3) -> f64) -> (WorldPos, DVec3) {
        match self {
            Self::Loop(f) => f.advance(dt_s, height_m),
            Self::Descent(d) => d.advance(dt_s, height_m),
        }
    }
}

/// A square of side `side_m` in the tangent plane at `start_dir`, projected onto the planet and
/// flown at a fixed height above the terrain.
pub struct SurfaceLoop {
    radius_m: f64,
    start_dir: DVec3,
    east: DVec3,
    north: DVec3,
    side_m: f64,
    speed_m_s: f64,
    height_above_m: f64,
    distance_m: f64,
    height_m: Option<f64>,
}

impl SurfaceLoop {
    pub fn new(
        radius_m: f64,
        start_dir: DVec3,
        side_m: f64,
        speed_m_s: f64,
        chunk_extent_m: f64,
    ) -> Self {
        let start_dir = start_dir.normalize();
        let (east, north) = tangent_axes(start_dir);
        Self {
            radius_m,
            start_dir,
            east,
            north,
            side_m,
            speed_m_s,
            height_above_m: LOOP_HEIGHT_CHUNK_EXTENTS * chunk_extent_m,
            distance_m: 0.0,
            height_m: None,
        }
    }

    fn total_m(&self) -> f64 {
        4.0 * self.side_m
    }

    /// Tangent plane offset and direction of travel at a distance along the loop.
    fn along(&self, distance_m: f64) -> (DVec3, DVec3) {
        let l = self.side_m;
        let d = distance_m.rem_euclid(self.total_m());
        let (e, n) = (self.east, self.north);
        let (start, dir) = if d < l {
            (DVec3::ZERO, e)
        } else if d < 2.0 * l {
            (e * l, n)
        } else if d < 3.0 * l {
            ((e + n) * l, -e)
        } else {
            (n * l, -n)
        };
        (start + dir * d.rem_euclid(l), dir)
    }

    fn advance(&mut self, dt_s: f64, height_m: impl Fn(DVec3) -> f64) -> (WorldPos, DVec3) {
        self.distance_m = (self.distance_m + self.speed_m_s * dt_s).min(self.total_m());
        let (offset, travel) = self.along(self.distance_m);
        let dir = (self.start_dir * self.radius_m + offset).normalize();
        let target = height_m(dir) + self.height_above_m;
        let height = match self.height_m {
            None => target,
            Some(h) => h + (target - h) * (1.0 - (-dt_s / HEIGHT_SMOOTHING_S).exp()),
        };
        self.height_m = Some(height);
        let position = dir * (self.radius_m + height);
        let horizontal = (travel - dir * travel.dot(dir)).normalize();
        let forward = horizontal * LOOP_PITCH_RAD.cos() + dir * LOOP_PITCH_RAD.sin();
        (WorldPos::body_fixed(position), position + forward * 100.0)
    }
}

/// Standing on the surface at `start_dir`, a climb straight up to `top_altitude_m` above the
/// terrain over `climb_s` with altitude growing geometrically (equal time per decade), a hold, the
/// same way back down, and a hold on the ground. The view pitches from just below the horizon to
/// nearly straight down as the planet shrinks below.
pub struct Descent {
    radius_m: f64,
    start_dir: DVec3,
    heading: DVec3,
    top_altitude_m: f64,
    climb_s: f64,
    time_s: f64,
}

impl Descent {
    pub fn new(
        radius_m: f64,
        start_dir: DVec3,
        heading: DVec3,
        top_altitude_m: f64,
        climb_s: f64,
    ) -> Self {
        let start_dir = start_dir.normalize();
        let heading =
            (heading - start_dir * heading.dot(start_dir)).normalize_or(tangent_axes(start_dir).0);
        Self {
            radius_m,
            start_dir,
            heading,
            top_altitude_m,
            climb_s,
            time_s: 0.0,
        }
    }

    pub fn duration_s(&self) -> f64 {
        2.0 * (DESCENT_GROUND_HOLD_S + self.climb_s) + DESCENT_TOP_HOLD_S
    }

    /// Climb fraction 0 (standing) to 1 (top) at time `t_s`.
    fn climb_fraction(&self, t_s: f64) -> f64 {
        let up_end = DESCENT_GROUND_HOLD_S + self.climb_s;
        let down_start = up_end + DESCENT_TOP_HOLD_S;
        let u = if t_s < DESCENT_GROUND_HOLD_S {
            0.0
        } else if t_s < up_end {
            (t_s - DESCENT_GROUND_HOLD_S) / self.climb_s
        } else if t_s < down_start {
            1.0
        } else {
            1.0 - (t_s - down_start) / self.climb_s
        };
        let u = u.clamp(0.0, 1.0);
        u * u * (3.0 - 2.0 * u)
    }

    fn advance(&mut self, dt_s: f64, height_m: impl Fn(DVec3) -> f64) -> (WorldPos, DVec3) {
        self.time_s = (self.time_s + dt_s).min(self.duration_s());
        let s = self.climb_fraction(self.time_s);
        let altitude = EYE_HEIGHT_M * (self.top_altitude_m / EYE_HEIGHT_M).powf(s);
        let up = self.start_dir;
        let position = up * (self.radius_m + height_m(up) + altitude);
        let pitch =
            DESCENT_GROUND_PITCH_RAD + (DESCENT_TOP_PITCH_RAD - DESCENT_GROUND_PITCH_RAD) * s;
        let forward = self.heading * pitch.cos() + up * pitch.sin();
        (WorldPos::body_fixed(position), position + forward * 100.0)
    }
}

/// East and north unit vectors of the tangent plane at `up` (north towards body +Y; at the poles
/// any perpendicular pair).
pub fn tangent_axes(up: DVec3) -> (DVec3, DVec3) {
    let east = DVec3::Y.cross(up);
    let east = if east.length_squared() < 1.0e-12 {
        up.any_orthonormal_vector()
    } else {
        east.normalize()
    };
    (east, up.cross(east))
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = 10_000.0;

    fn flat(_: DVec3) -> f64 {
        5.0
    }

    #[test]
    fn loop_returns_to_the_start_and_keeps_its_height() {
        let start = DVec3::new(0.3, 0.8, 0.5);
        let mut flight = Flight::Loop(SurfaceLoop::new(R, start, 2000.0, 100.0, 32.0));
        assert_eq!(flight.duration_s(), 80.0);
        let (first, _) = flight.advance(0.0, flat);
        for _ in 0..800 {
            let (p, target) = flight.advance(0.1, flat);
            assert!((p.m.length() - (R + 5.0 + 64.0)).abs() < 1.0e-6);
            assert!(
                (target - p.m).dot(p.m) < 0.0,
                "the view points below the horizon"
            );
        }
        assert!(flight.done());
        let (last, _) = flight.advance(0.1, flat);
        assert!((last.m - first.m).length() < 1.0e-6);
    }

    #[test]
    fn descent_climbs_to_the_top_and_lands_where_it_started() {
        let start = DVec3::new(-0.2, 0.4, 0.9);
        let mut flight = Flight::Descent(Descent::new(R, start, DVec3::X, 100_000.0, 30.0));
        let (ground, _) = flight.advance(0.0, flat);
        assert!((ground.m.length() - (R + 5.0 + EYE_HEIGHT_M)).abs() < 1.0e-9);
        let mut highest: f64 = 0.0;
        while !flight.done() {
            let (p, target) = flight.advance(0.05, flat);
            highest = highest.max(p.m.length() - R - 5.0);
            assert!((target - p.m).normalize().dot(p.m.normalize()) < 0.0);
        }
        assert!((highest - 100_000.0).abs() < 1.0, "{highest}");
        let (end, _) = flight.advance(0.05, flat);
        assert!((end.m - ground.m).length() < 1.0e-6);
    }

    #[test]
    fn tangent_axes_are_orthonormal_even_at_the_poles() {
        for up in [DVec3::Y, DVec3::NEG_Y, DVec3::new(0.6, 0.0, 0.8)] {
            let (e, n) = tangent_axes(up);
            assert!(e.dot(up).abs() < 1.0e-12 && n.dot(up).abs() < 1.0e-12);
            assert!(e.dot(n).abs() < 1.0e-12);
            assert!((e.length() - 1.0).abs() < 1.0e-12 && (n.length() - 1.0).abs() < 1.0e-12);
        }
    }
}
