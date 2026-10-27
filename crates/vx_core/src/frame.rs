use glam::{DVec3, Vec3};

/// Reference frame a position is expressed in (D4).
///
/// Only the body fixed frame exists in M0; the system and body centred inertial frames arrive
/// with the universe crate in M6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Frame {
    /// Origin at the body centre, axes rotating with the body.
    #[default]
    BodyFixed,
}

/// An f64 position in metres in an explicit frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WorldPos {
    pub frame: Frame,
    pub m: DVec3,
}

impl WorldPos {
    pub const fn new(frame: Frame, m: DVec3) -> Self {
        Self { frame, m }
    }

    pub const fn body_fixed(m: DVec3) -> Self {
        Self::new(Frame::BodyFixed, m)
    }
}

/// Offset of `world` from `origin` as f32, the only form in which a position may reach the GPU.
///
/// The subtraction happens in f64, so precision depends on the distance between the two points
/// rather than on their distance from the frame origin.
pub fn camera_relative(world: WorldPos, origin: WorldPos) -> Vec3 {
    debug_assert_eq!(
        world.frame, origin.frame,
        "positions in different frames must be converted before subtraction"
    );
    (world.m - origin.m).as_vec3()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_relative_is_exact_far_from_the_origin() {
        // A camera 10^9 m from the frame origin (D4 precision target) looking at a point 1.25 m away.
        let camera = WorldPos::body_fixed(DVec3::new(1.0e9, -3.0e8, 7.0e8));
        let point = WorldPos::body_fixed(camera.m + DVec3::new(1.25, -0.5, 0.001));
        let rel = camera_relative(point, camera);
        assert!((rel - Vec3::new(1.25, -0.5, 0.001)).abs().max_element() < 1.0e-6);
    }

    #[test]
    fn f32_alone_loses_millimetres_at_distance() {
        // Documents why the subtraction must stay in f64: the same offset computed in f32 is wrong.
        let camera = 1.0e9_f32;
        let point = camera + 0.001;
        assert_ne!(point - camera, 0.001);
    }
}
