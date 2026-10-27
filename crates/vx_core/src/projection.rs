use glam::{Mat4, Vec4};

/// Perspective projection with reverse-Z and an infinite far plane (D4).
///
/// Depth is 1.0 at the near plane and tends to 0.0 at infinity, so the depth buffer clears to 0.0
/// and the compare function is `Greater`. With a float depth buffer this spreads precision evenly
/// in log space (Reed 2015, "Depth Precision Visualized").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    pub fov_y_rad: f32,
    pub aspect: f32,
    pub near_m: f32,
}

impl Projection {
    pub fn matrix(&self) -> Mat4 {
        Self::reverse_z_infinite(self.fov_y_rad, self.aspect, self.near_m)
    }

    /// Right handed view space (the camera looks down -Z), clip depth in [0, 1].
    ///
    /// For a view space depth `z < 0` the result is `clip.z = near`, `clip.w = -z`, so
    /// `ndc.z = near / -z`.
    pub fn reverse_z_infinite(fov_y_rad: f32, aspect: f32, near_m: f32) -> Mat4 {
        debug_assert!(near_m > 0.0 && aspect > 0.0 && fov_y_rad > 0.0);
        let f = 1.0 / (0.5 * fov_y_rad).tan();
        Mat4::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, near_m, 0.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    const FOV: f32 = std::f32::consts::FRAC_PI_3;
    const ASPECT: f32 = 16.0 / 9.0;
    const NEAR: f32 = 0.05;

    fn ndc(m: &Mat4, view: Vec3) -> Vec3 {
        let clip = *m * view.extend(1.0);
        clip.truncate() / clip.w
    }

    #[test]
    fn depth_is_one_at_the_near_plane() {
        let m = Projection::reverse_z_infinite(FOV, ASPECT, NEAR);
        let d = ndc(&m, Vec3::new(0.0, 0.0, -NEAR)).z;
        assert!((d - 1.0).abs() < 1.0e-6, "depth at near = {d}");
    }

    #[test]
    fn depth_tends_to_zero_at_infinity() {
        let m = Projection::reverse_z_infinite(FOV, ASPECT, NEAR);
        let mut previous = 1.0;
        for distance in [1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12] {
            let d = ndc(&m, Vec3::new(0.0, 0.0, -distance)).z;
            assert!(
                d > 0.0 && d < previous,
                "depth must fall monotonically, got {d}"
            );
            previous = d;
        }
        assert!(previous < 1.0e-12);
        let clip_at_infinity = m * Vec4::new(0.0, 0.0, -1.0, 0.0);
        assert_eq!(clip_at_infinity.z, 0.0);
    }

    #[test]
    fn behind_the_near_plane_is_clipped() {
        let m = Projection::reverse_z_infinite(FOV, ASPECT, NEAR);
        let d = ndc(&m, Vec3::new(0.0, 0.0, -0.5 * NEAR)).z;
        assert!(d > 1.0);
    }

    #[test]
    fn field_of_view_maps_to_the_viewport_edge() {
        let m = Projection::reverse_z_infinite(FOV, ASPECT, NEAR);
        let z = -10.0;
        let half_height = -z * (0.5 * FOV).tan();
        let p = ndc(&m, Vec3::new(half_height * ASPECT, half_height, z));
        assert!((p.x - 1.0).abs() < 1.0e-5 && (p.y - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn matches_glam_webgpu_reference() {
        let ours = Projection::reverse_z_infinite(FOV, ASPECT, NEAR);
        let reference =
            glam::camera::rh::proj::directx::perspective_infinite_reverse(FOV, ASPECT, NEAR);
        assert!(ours.abs_diff_eq(reference, 1.0e-6));
    }
}
