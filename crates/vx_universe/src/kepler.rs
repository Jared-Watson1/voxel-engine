use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Classical elements of an elliptic orbit relative to the parent body, in the parent's inertial
/// frame (system axes). Angles in radians, lengths in metres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrbitElements {
    pub semi_major_axis_m: f64,
    /// 0 for a circle, below 1 for an ellipse.
    pub eccentricity: f64,
    /// Tilt of the orbit plane from the reference plane (the system XZ plane).
    pub inclination_rad: f64,
    pub longitude_of_ascending_node_rad: f64,
    pub argument_of_periapsis_rad: f64,
    /// Mean anomaly at the system epoch (time 0).
    pub mean_anomaly_at_epoch_rad: f64,
}

impl OrbitElements {
    /// Orbital period for the parent's gravitational parameter `mu_m3_s2`.
    pub fn period_s(&self, mu_m3_s2: f64) -> f64 {
        let _ = mu_m3_s2;
        todo!("workstream A")
    }

    /// Position and velocity relative to the parent at simulation time `t_s`, in the parent's
    /// inertial frame.
    pub fn state_at(&self, mu_m3_s2: f64, t_s: f64) -> (DVec3, DVec3) {
        let _ = (mu_m3_s2, t_s);
        todo!("workstream A")
    }
}

/// Solves Kepler's equation `M = E - e sin E` for the eccentric anomaly `E` by Newton iteration,
/// for `0 <= e < 1`, to a residual under 1e-12.
pub fn solve_kepler(mean_anomaly_rad: f64, eccentricity: f64) -> f64 {
    let _ = (mean_anomaly_rad, eccentricity);
    todo!("workstream A")
}

/// Rotation of a body about its own +Y axis, tilted from the system's +Y.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotationState {
    /// Sidereal rotation period; negative for retrograde rotation.
    pub sidereal_period_s: f64,
    /// Angle between the rotation axis and the system +Y, tilted about the system +X.
    pub axial_tilt_rad: f64,
    /// Rotation angle about the body +Y at time 0.
    pub prime_meridian_at_epoch_rad: f64,
}

impl RotationState {
    /// Rotation taking body fixed axes to the body centred inertial (system) axes at `t_s`: first
    /// the spin about body +Y, then the tilt about system +X.
    pub fn body_to_inertial(&self, t_s: f64) -> DQuat {
        let _ = t_s;
        todo!("workstream A")
    }
}
