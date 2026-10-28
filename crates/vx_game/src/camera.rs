use glam::{DMat3, DQuat, DVec3, Mat4};
use vx_core::{Projection, WorldPos};

/// Slowest and fastest speed factors; scrolling moves between them logarithmically.
pub const MIN_SPEED_SCALE: f64 = 0.01;
pub const MAX_SPEED_SCALE: f64 = 100.0;
/// Speed factor per scroll line.
const SCROLL_SPEED_FACTOR: f64 = 1.25;
/// Mouse look sensitivity in radians per pixel of mouse motion.
const LOOK_RAD_PER_PIXEL: f64 = 0.0025;
/// Pitch stops this far short of straight up or down so the heading stays defined.
const PITCH_LIMIT_RAD: f64 = 0.01;
/// Fly speed per metre of altitude at a speed scale of 1, so a dive from orbit and a stroll over
/// the grass both take seconds: 20 m/s near the ground, 1 km/s at 1 km, 100 km/s at 100 km.
const SPEED_PER_ALTITUDE_PER_S: f64 = 1.0;
/// Altitudes below this fly as if at this altitude, so the camera never stalls on the ground.
const SPEED_FLOOR_ALTITUDE_M: f64 = 20.0;

/// Movement keys held this frame, in camera local axes; up and down follow the local vertical.
#[derive(Debug, Default, Clone, Copy)]
pub struct MoveInput {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

/// Free fly camera with an f64 position in the planet's body fixed frame (D4) whose horizon stays
/// level: up is the radial direction from the planet centre, so the camera can fly around the
/// whole planet and out to orbit without a pole singularity or accumulated roll.
///
/// View space is right handed with the camera looking down -Z and +Y up.
#[derive(Debug, Clone)]
pub struct FlyCamera {
    pub position: WorldPos,
    orientation: DQuat,
    /// Multiplies the altitude based speed; scrolling changes it.
    pub speed_scale: f64,
    /// Speed of the last update in metres per second, for the overlay.
    pub speed_m_s: f64,
    pub projection: Projection,
}

impl FlyCamera {
    pub fn looking_at(position: WorldPos, target: DVec3, projection: Projection) -> Self {
        let mut camera = Self {
            position,
            orientation: DQuat::IDENTITY,
            speed_scale: 1.0,
            speed_m_s: 0.0,
            projection,
        };
        camera.face(target - position.m);
        camera
    }

    /// Radial up at the camera; +Y at the planet centre itself.
    pub fn up(&self) -> DVec3 {
        self.position.m.normalize_or(DVec3::Y)
    }

    pub fn orientation(&self) -> DQuat {
        self.orientation
    }

    pub fn forward(&self) -> DVec3 {
        self.orientation * DVec3::NEG_Z
    }

    /// Turns the camera to look along `direction`, level with the local horizon.
    pub fn face(&mut self, direction: DVec3) {
        let up = self.up();
        let forward = direction.normalize_or(-up);
        self.orientation = level_orientation(forward, up, self.forward());
    }

    pub fn look(&mut self, dx_px: f64, dy_px: f64) {
        let up = self.up();
        let yaw = DQuat::from_axis_angle(up, -dx_px * LOOK_RAD_PER_PIXEL);
        let right = self.orientation * DVec3::X;
        let pitch = DQuat::from_axis_angle(right, -dy_px * LOOK_RAD_PER_PIXEL);
        let forward = yaw * pitch * self.forward();
        self.orientation = level_orientation(forward, up, self.forward());
    }

    /// Scales the speed factor per scroll line, clamped to the range.
    pub fn scroll(&mut self, lines: f64) {
        self.speed_scale = (self.speed_scale * SCROLL_SPEED_FACTOR.powf(lines))
            .clamp(MIN_SPEED_SCALE, MAX_SPEED_SCALE);
    }

    /// Moves by the held keys for `dt_s` at a speed proportional to `altitude_m`, then keeps the
    /// horizon level against the new radial up.
    pub fn update(&mut self, input: MoveInput, dt_s: f64, altitude_m: f64) {
        let axis = |pos: bool, neg: bool| f64::from(u8::from(pos)) - f64::from(u8::from(neg));
        let local = DVec3::new(
            axis(input.right, input.left),
            0.0,
            axis(input.back, input.forward),
        );
        let up = self.up();
        let direction = self.orientation * local + up * axis(input.up, input.down);
        self.speed_m_s =
            self.speed_scale * SPEED_PER_ALTITUDE_PER_S * altitude_m.max(SPEED_FLOOR_ALTITUDE_M);
        if direction != DVec3::ZERO {
            self.position.m += direction.normalize() * self.speed_m_s * dt_s;
        }
        // Carry the view with the turn of the local vertical, so flying around the planet keeps
        // the pitch relative to the horizon instead of a fixed direction in space.
        let transport = DQuat::from_rotation_arc(up, self.up());
        let forward = transport * self.forward();
        self.orientation = level_orientation(forward, self.up(), forward);
    }

    /// Rotation only view matrix: the camera sits at the origin of the render frame (D4).
    pub fn view_rotation(&self) -> Mat4 {
        Mat4::from_quat(self.orientation.inverse().as_quat())
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection.matrix() * self.view_rotation()
    }
}

/// The orientation looking along `forward` with no roll against `up`, the pitch limited short of
/// straight up or down. When `forward` is too close to vertical, the heading of `previous` is
/// kept so the limit does not spin the camera.
fn level_orientation(forward: DVec3, up: DVec3, previous: DVec3) -> DQuat {
    let max_sin = (std::f64::consts::FRAC_PI_2 - PITCH_LIMIT_RAD).sin();
    let mut horizontal = forward - up * forward.dot(up);
    if horizontal.length_squared() < 1.0e-12 {
        horizontal = previous - up * previous.dot(up);
    }
    let horizontal = horizontal.normalize_or(up.any_orthonormal_vector());
    let sin_pitch = forward.dot(up).clamp(-max_sin, max_sin);
    let cos_pitch = (1.0 - sin_pitch * sin_pitch).sqrt();
    let f = horizontal * cos_pitch + up * sin_pitch;
    let right = f.cross(up).normalize();
    let cam_up = right.cross(f);
    DQuat::from_mat3(&DMat3::from_cols(right, cam_up, -f))
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
        let camera = FlyCamera::looking_at(eye, target, projection());
        let rel = camera_relative(WorldPos::body_fixed(target), eye);
        let clip = camera.view_proj() * rel.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1.0e-4 && ndc.y.abs() < 1.0e-4, "{ndc}");
        assert!(ndc.z > 0.0 && ndc.z < 1.0);
    }

    #[test]
    fn horizon_stays_level_everywhere_on_the_planet() {
        for p in [
            DVec3::new(0.0, 10_000.0, 0.0),
            DVec3::new(10_000.0, 0.0, 0.0),
            DVec3::new(-3000.0, -7000.0, 6000.0),
            DVec3::new(0.0, -10_000.0, 1.0e-3),
        ] {
            let mut camera = FlyCamera::looking_at(
                WorldPos::body_fixed(p),
                p + p.any_orthonormal_vector() * 100.0,
                projection(),
            );
            for i in 0..200 {
                camera.look(
                    13.0 * f64::from(i % 7) - 30.0,
                    5.0 * f64::from(i % 5) - 10.0,
                );
                let right = camera.orientation() * DVec3::X;
                assert!(right.dot(camera.up()).abs() < 1.0e-9, "no roll at {p}");
            }
        }
    }

    #[test]
    fn flying_around_the_planet_keeps_the_view_level() {
        let start = DVec3::new(0.0, 10_100.0, 0.0);
        let mut camera =
            FlyCamera::looking_at(WorldPos::body_fixed(start), start + DVec3::X, projection());
        let input = MoveInput {
            forward: true,
            ..Default::default()
        };
        for _ in 0..2000 {
            camera.update(input, 0.1, 100.0);
            let up = camera.up();
            assert!(
                camera.forward().dot(up).abs() < 0.05,
                "forward stays near the horizon"
            );
            assert!((camera.orientation() * DVec3::X).dot(up).abs() < 1.0e-9);
        }
        assert!(
            camera.position.m.y < 10_000.0,
            "the camera passed over the pole"
        );
    }

    #[test]
    fn pitch_is_limited_short_of_the_vertical() {
        let p = DVec3::new(10_000.0, 0.0, 0.0);
        let mut camera = FlyCamera::looking_at(WorldPos::body_fixed(p), p + DVec3::Z, projection());
        for _ in 0..1000 {
            camera.look(0.0, -50.0);
        }
        let f = camera.forward().dot(camera.up());
        assert!(f < 1.0 && f > 0.99, "{f}");
        for _ in 0..1000 {
            camera.look(37.0, 50.0);
        }
        let f = camera.forward().dot(camera.up());
        assert!(f > -1.0 && f < -0.99, "{f}");
    }

    #[test]
    fn speed_follows_altitude_and_scroll_clamps() {
        let p = DVec3::new(0.0, 10_000.0, 0.0);
        let mut camera = FlyCamera::looking_at(WorldPos::body_fixed(p), p + DVec3::X, projection());
        camera.update(MoveInput::default(), 0.0, 1000.0);
        assert_eq!(camera.speed_m_s, 1000.0);
        camera.update(MoveInput::default(), 0.0, 0.5);
        assert_eq!(camera.speed_m_s, SPEED_FLOOR_ALTITUDE_M);
        camera.scroll(1000.0);
        assert_eq!(camera.speed_scale, MAX_SPEED_SCALE);
        camera.scroll(-1000.0);
        assert_eq!(camera.speed_scale, MIN_SPEED_SCALE);
    }

    #[test]
    fn view_rotation_has_no_translation() {
        let camera = FlyCamera::looking_at(
            WorldPos::body_fixed(DVec3::splat(5.0e8)),
            DVec3::splat(5.0e8) + DVec3::X,
            projection(),
        );
        assert_eq!(camera.view_rotation().w_axis.truncate(), Vec3::ZERO);
    }
}
