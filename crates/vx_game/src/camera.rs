use glam::{DQuat, DVec3, Mat4};
use vx_core::{Projection, WorldPos};

/// Slowest and fastest fly speeds; scrolling moves between them logarithmically.
pub const MIN_SPEED_M_S: f64 = 1.0;
pub const MAX_SPEED_M_S: f64 = 10_000.0;
/// Speed factor per scroll line.
const SCROLL_SPEED_FACTOR: f64 = 1.25;
/// Mouse look sensitivity in radians per pixel of mouse motion.
const LOOK_RAD_PER_PIXEL: f64 = 0.0025;
/// Pitch stops this far short of straight up or down so yaw stays defined.
const PITCH_LIMIT_RAD: f64 = 0.01;

/// Movement keys held this frame, in camera local axes.
#[derive(Debug, Default, Clone, Copy)]
pub struct MoveInput {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

/// Free fly camera with an f64 position (D4).
///
/// View space is right handed with the camera looking down -Z and +Y up. Orientation is the
/// quaternion `yaw about frame +Y * pitch about camera +X`, rebuilt from the two angles so pitch
/// can be clamped exactly and no roll accumulates.
#[derive(Debug, Clone)]
pub struct FlyCamera {
    pub position: WorldPos,
    yaw_rad: f64,
    pitch_rad: f64,
    pub speed_m_s: f64,
    pub projection: Projection,
}

impl FlyCamera {
    pub fn looking_at(
        position: WorldPos,
        target: DVec3,
        speed_m_s: f64,
        projection: Projection,
    ) -> Self {
        let forward = (target - position.m).normalize();
        let mut camera = Self {
            position,
            yaw_rad: (-forward.x).atan2(-forward.z),
            pitch_rad: forward.y.clamp(-1.0, 1.0).asin(),
            speed_m_s: speed_m_s.clamp(MIN_SPEED_M_S, MAX_SPEED_M_S),
            projection,
        };
        camera.clamp_pitch();
        camera
    }

    pub fn orientation(&self) -> DQuat {
        DQuat::from_rotation_y(self.yaw_rad) * DQuat::from_rotation_x(self.pitch_rad)
    }

    #[cfg(test)]
    pub fn forward(&self) -> DVec3 {
        self.orientation() * DVec3::NEG_Z
    }

    pub fn look(&mut self, dx_px: f64, dy_px: f64) {
        self.yaw_rad = (self.yaw_rad - dx_px * LOOK_RAD_PER_PIXEL) % std::f64::consts::TAU;
        self.pitch_rad -= dy_px * LOOK_RAD_PER_PIXEL;
        self.clamp_pitch();
    }

    fn clamp_pitch(&mut self) {
        let limit = std::f64::consts::FRAC_PI_2 - PITCH_LIMIT_RAD;
        self.pitch_rad = self.pitch_rad.clamp(-limit, limit);
    }

    /// Scales speed by a fixed factor per scroll line, clamped to the speed range.
    pub fn scroll(&mut self, lines: f64) {
        self.speed_m_s =
            (self.speed_m_s * SCROLL_SPEED_FACTOR.powf(lines)).clamp(MIN_SPEED_M_S, MAX_SPEED_M_S);
    }

    pub fn update(&mut self, input: MoveInput, dt_s: f64) {
        let axis = |pos: bool, neg: bool| f64::from(u8::from(pos)) - f64::from(u8::from(neg));
        let local = DVec3::new(
            axis(input.right, input.left),
            0.0,
            axis(input.back, input.forward),
        );
        let vertical = DVec3::Y * axis(input.up, input.down);
        let direction = self.orientation() * local + vertical;
        if direction != DVec3::ZERO {
            self.position.m += direction.normalize() * self.speed_m_s * dt_s;
        }
    }

    /// Rotation only view matrix: the camera sits at the origin of the render frame (D4).
    pub fn view_rotation(&self) -> Mat4 {
        Mat4::from_quat(self.orientation().inverse().as_quat())
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection.matrix() * self.view_rotation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use vx_core::camera_relative;

    fn projection() -> Projection {
        Projection {
            fov_y_rad: 1.0,
            aspect: 1.5,
            near_m: 0.05,
        }
    }

    #[test]
    fn looking_at_centres_the_target() {
        let eye = WorldPos::body_fixed(DVec3::new(1.0e9, 20.0, 35.0));
        let target = DVec3::new(1.0e9 + 3.0, 0.0, 0.0);
        let camera = FlyCamera::looking_at(eye, target, 10.0, projection());
        let rel = camera_relative(WorldPos::body_fixed(target), eye);
        let clip = camera.view_proj() * rel.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1.0e-4 && ndc.y.abs() < 1.0e-4, "{ndc}");
        assert!(ndc.z > 0.0 && ndc.z < 1.0);
    }

    #[test]
    fn forward_moves_along_the_view_direction() {
        let mut camera = FlyCamera::looking_at(
            WorldPos::default(),
            DVec3::new(0.0, 0.0, -1.0),
            4.0,
            projection(),
        );
        camera.update(
            MoveInput {
                forward: true,
                ..Default::default()
            },
            0.5,
        );
        assert!(
            camera
                .position
                .m
                .abs_diff_eq(DVec3::new(0.0, 0.0, -2.0), 1.0e-12)
        );
    }

    #[test]
    fn pitch_is_limited_short_of_the_poles() {
        let mut camera = FlyCamera::looking_at(
            WorldPos::default(),
            DVec3::new(0.0, 0.0, -1.0),
            4.0,
            projection(),
        );
        for _ in 0..1000 {
            camera.look(0.0, -50.0);
        }
        let f = camera.forward();
        assert!(f.y < 1.0 && f.y > 0.99, "{f}");
        for _ in 0..1000 {
            camera.look(37.0, 50.0);
        }
        let f = camera.forward();
        assert!(f.y > -1.0 && f.y < -0.99, "{f}");
        let right = camera.orientation() * DVec3::X;
        assert!(right.y.abs() < 1.0e-9, "no roll accumulates: {right}");
    }

    #[test]
    fn scroll_clamps_to_the_speed_range() {
        let mut camera =
            FlyCamera::looking_at(WorldPos::default(), DVec3::NEG_Z, 4.0, projection());
        camera.scroll(1000.0);
        assert_eq!(camera.speed_m_s, MAX_SPEED_M_S);
        camera.scroll(-1000.0);
        assert_eq!(camera.speed_m_s, MIN_SPEED_M_S);
    }

    #[test]
    fn view_rotation_has_no_translation() {
        let camera = FlyCamera::looking_at(
            WorldPos::body_fixed(DVec3::splat(5.0e8)),
            DVec3::splat(5.0e8) + DVec3::X,
            4.0,
            projection(),
        );
        assert_eq!(camera.view_rotation().w_axis.truncate(), Vec3::ZERO);
    }
}
