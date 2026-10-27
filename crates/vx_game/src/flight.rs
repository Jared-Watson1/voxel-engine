//! Scripted flight for measurements: a square loop over the terrain at a fixed speed and height
//! above the surface, so the M1 exit run (2 km by 2 km at 60 fps) needs nobody at the keyboard.

use glam::{DVec2, DVec3};
use vx_core::WorldPos;
use vx_worldgen::TerrainSampler;

/// Height above the terrain the camera keeps, as a multiple of the chunk extent so the view
/// scales with the voxel size.
const HEIGHT_CHUNK_EXTENTS: f64 = 2.0;
/// Downward pitch of the view in radians (about 18 degrees).
const PITCH_RAD: f64 = -0.32;
/// Time constant in seconds of the height smoothing, so ridges do not throw the camera.
const HEIGHT_SMOOTHING_S: f64 = 1.5;

pub struct Flight {
    side_m: f64,
    speed_m_s: f64,
    chunk_extent_m: f64,
    corner: DVec2,
    distance_m: f64,
    height_m: Option<f64>,
}

impl Flight {
    /// A square of side `side_m` whose first corner is at the frame origin, flown at `speed_m_s`.
    pub fn new(side_m: f64, speed_m_s: f64, chunk_extent_m: f64) -> Self {
        Self {
            side_m,
            speed_m_s,
            chunk_extent_m,
            corner: DVec2::ZERO,
            distance_m: 0.0,
            height_m: None,
        }
    }

    pub fn total_m(&self) -> f64 {
        4.0 * self.side_m
    }

    pub fn duration_s(&self) -> f64 {
        self.total_m() / self.speed_m_s
    }

    pub fn done(&self) -> bool {
        self.distance_m >= self.total_m()
    }

    pub fn progress(&self) -> f64 {
        (self.distance_m / self.total_m()).clamp(0.0, 1.0)
    }

    /// Horizontal position and direction of travel at a distance along the loop.
    fn along(&self, distance_m: f64) -> (DVec2, DVec2) {
        let l = self.side_m;
        let d = distance_m.rem_euclid(self.total_m());
        let (start, dir) = if d < l {
            (DVec2::ZERO, DVec2::X)
        } else if d < 2.0 * l {
            (DVec2::new(l, 0.0), DVec2::Y)
        } else if d < 3.0 * l {
            (DVec2::new(l, l), DVec2::NEG_X)
        } else {
            (DVec2::new(0.0, l), DVec2::NEG_Y)
        };
        (self.corner + start + dir * d.rem_euclid(l), dir)
    }

    /// Advances the flight and returns the camera position and the point it looks at.
    pub fn advance(&mut self, dt_s: f64, sampler: &TerrainSampler) -> (WorldPos, DVec3) {
        self.distance_m = (self.distance_m + self.speed_m_s * dt_s).min(self.total_m());
        let (xz, dir) = self.along(self.distance_m);
        let target_height =
            sampler.height_m(xz.x, xz.y) + HEIGHT_CHUNK_EXTENTS * self.chunk_extent_m;
        let height = match self.height_m {
            None => target_height,
            Some(h) => {
                let alpha = 1.0 - (-dt_s / HEIGHT_SMOOTHING_S).exp();
                h + (target_height - h) * alpha
            }
        };
        self.height_m = Some(height);
        let position = DVec3::new(xz.x, height, xz.y);
        let forward = DVec3::new(dir.x, 0.0, dir.y) * PITCH_RAD.cos() + DVec3::Y * PITCH_RAD.sin();
        (WorldPos::body_fixed(position), position + forward * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vx_core::TerrainParams;

    #[test]
    fn loop_returns_to_the_start_and_covers_four_sides() {
        let sampler = TerrainSampler::new(1, TerrainParams::default());
        let mut flight = Flight::new(2000.0, 100.0, 32.0);
        assert_eq!(flight.duration_s(), 80.0);
        let (start, _) = flight.advance(0.0, &sampler);
        let mut max_x: f64 = 0.0;
        let mut max_z: f64 = 0.0;
        for _ in 0..800 {
            let (p, _) = flight.advance(0.1, &sampler);
            max_x = max_x.max(p.m.x);
            max_z = max_z.max(p.m.z);
        }
        assert!(flight.done());
        let (end, _) = flight.advance(0.1, &sampler);
        assert!((end.m.x - start.m.x).abs() < 1.0e-6 && (end.m.z - start.m.z).abs() < 1.0e-6);
        assert!((max_x - 2000.0).abs() < 1.0e-6 && (max_z - 2000.0).abs() < 1.0e-6);
    }
}
