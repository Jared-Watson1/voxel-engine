use std::f64::consts::{PI, TAU};

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Classical elements of an elliptic orbit relative to the parent body, in the parent's inertial
/// frame (system axes). Angles in radians, lengths in metres.
///
/// Orientation (Y up, right handed): the perifocal frame has the periapsis along +X and the
/// direction of motion at periapsis along -Z, so an orbit with zero inclination is prograde
/// (angular momentum along +Y, counterclockwise seen from +Y). The perifocal frame is carried to
/// system axes by `R_Y(node) * R_X(inclination) * R_Y(argument_of_periapsis)`: a rotation about
/// the orbit normal by the argument of periapsis, a tilt about the node line (+X before the node
/// rotation) by the inclination, then a rotation about the ecliptic north (+Y) by the longitude of
/// the ascending node. A positive rotation about +Y takes +X toward -Z, so the node longitude is
/// measured in the XZ plane from +X toward -Z, the direction of prograde motion. This is the
/// textbook `R3(node) R1(i) R3(w)` of a Z up frame with Z mapped to +Y and Y mapped to -Z.
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
        TAU * (self.semi_major_axis_m.powi(3) / mu_m3_s2).sqrt()
    }

    /// Rotation from the perifocal frame to the parent's inertial frame (see the type docs).
    pub fn orientation(&self) -> DQuat {
        DQuat::from_rotation_y(self.longitude_of_ascending_node_rad)
            * DQuat::from_rotation_x(self.inclination_rad)
            * DQuat::from_rotation_y(self.argument_of_periapsis_rad)
    }

    /// Position and velocity relative to the parent at simulation time `t_s`, in the parent's
    /// inertial frame.
    ///
    /// Time is reduced modulo the period before the mean anomaly is formed, so precision does not
    /// degrade with large `t_s` and `state_at(mu, t + period_s(mu))` equals `state_at(mu, t)` to
    /// rounding (exactly at `t = 0`).
    pub fn state_at(&self, mu_m3_s2: f64, t_s: f64) -> (DVec3, DVec3) {
        let a = self.semi_major_axis_m;
        let e = self.eccentricity;
        let period_s = self.period_s(mu_m3_s2);
        let mean_motion_rad_s = TAU / period_s;
        let mean_anomaly_rad =
            self.mean_anomaly_at_epoch_rad + mean_motion_rad_s * t_s.rem_euclid(period_s);
        let ecc_anomaly_rad = solve_kepler(mean_anomaly_rad, e);
        let (sin_e, cos_e) = ecc_anomaly_rad.sin_cos();
        let one_minus_e2_sqrt = (1.0 - e * e).sqrt();

        let x_pf = a * (cos_e - e);
        let y_pf = a * one_minus_e2_sqrt * sin_e;
        let r_m = a * (1.0 - e * cos_e);
        let speed_scale = (mu_m3_s2 * a).sqrt() / r_m;
        let vx_pf = -speed_scale * sin_e;
        let vy_pf = speed_scale * one_minus_e2_sqrt * cos_e;

        let rotation = self.orientation();
        let position = rotation * DVec3::new(x_pf, 0.0, -y_pf);
        let velocity = rotation * DVec3::new(vx_pf, 0.0, -vy_pf);
        (position, velocity)
    }
}

/// Newton steps allowed before the safeguarded solve stops refining. The tests measure at most
/// 8 over e in 0 to 0.99 (`kepler_converges_in_few_steps`), so the cap is never reached.
const KEPLER_MAX_ITERATIONS: usize = 64;

/// Solves Kepler's equation `M = E - e sin E` for the eccentric anomaly `E` by Newton iteration,
/// for `0 <= e < 1`, to a residual under 1e-12.
///
/// `M` is reduced to `[-pi, pi]` and, since the equation is odd, solved for `|M|` on `[0, pi]`,
/// where `f(E) = E - e sin E - |M|` is increasing with `f(0) <= 0 <= f(pi)`. Each Newton step is
/// clamped to the shrinking bracket and falls back to bisection when it makes no progress, so
/// the solve converges for any `e` below 1 however poor the start. The start `|M| + 0.85 e` lies
/// right of the root for high eccentricity, where `f` is convex and Newton then descends
/// monotonically. The sign and the removed whole turns are restored at the end, so
/// `E - e sin E` returns the caller's `M` for any finite input and `E` is continuous in `M`.
pub fn solve_kepler(mean_anomaly_rad: f64, eccentricity: f64) -> f64 {
    solve_kepler_counted(mean_anomaly_rad, eccentricity).0
}

/// [`solve_kepler`] and the number of Newton steps taken.
fn solve_kepler_counted(mean_anomaly_rad: f64, eccentricity: f64) -> (f64, usize) {
    debug_assert!(
        (0.0..1.0).contains(&eccentricity),
        "solve_kepler needs 0 <= e < 1, got {eccentricity}"
    );
    let e = eccentricity;
    let turns = ((mean_anomaly_rad + PI) / TAU).floor();
    let m_wrapped = mean_anomaly_rad - turns * TAU;
    let m = m_wrapped.abs().min(PI);

    let (mut lo, mut hi) = (0.0_f64, PI);
    let mut ecc = (m + 0.85 * e).min(hi);
    let mut steps = 0;
    while steps < KEPLER_MAX_ITERATIONS {
        let f = ecc - e * ecc.sin() - m;
        if f.abs() < 1.0e-15 {
            break;
        }
        steps += 1;
        if f > 0.0 {
            hi = ecc;
        } else {
            lo = ecc;
        }
        let derivative = 1.0 - e * ecc.cos();
        let mut next = (ecc - f / derivative).clamp(lo, hi);
        if next == ecc {
            next = 0.5 * (lo + hi);
        }
        if next == ecc {
            break;
        }
        ecc = next;
    }
    (ecc.copysign(m_wrapped) + turns * TAU, steps)
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
    ///
    /// The result is `R_X(tilt) * R_Y(prime_meridian + 2 pi t / period)`. A positive spin takes
    /// the body +X toward -Z, the same sense as a prograde orbit, so for a positive period the
    /// surface moves east (east at the prime meridian is body -Z, longitude `atan2(-z, x)`) and
    /// the star's subsolar point moves west. With a positive tilt the north pole leans toward
    /// system +Z. The spin angle is formed from `t_s` reduced modulo the period, so it stays
    /// precise at large times.
    pub fn body_to_inertial(&self, t_s: f64) -> DQuat {
        let period_s = self.sidereal_period_s;
        let spin_rad =
            self.prime_meridian_at_epoch_rad + TAU * (t_s.rem_euclid(period_s) / period_s);
        DQuat::from_rotation_x(self.axial_tilt_rad) * DQuat::from_rotation_y(spin_rad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH_MU: f64 = 3.986_004_418e14;

    fn kepler_residual(m: f64, e: f64) -> f64 {
        let ecc = solve_kepler(m, e);
        (ecc - e * ecc.sin() - m).abs()
    }

    #[test]
    fn kepler_residual_is_tiny_across_eccentricities_and_the_circle() {
        let mut worst = 0.0_f64;
        for ei in 0..=99 {
            let e = f64::from(ei) * 0.01;
            for mi in 0..=720 {
                let m = f64::from(mi) / 720.0 * TAU;
                worst = worst.max(kepler_residual(m, e));
            }
            for m in [
                1.0e-12,
                1.0e-6,
                -1.0e-9,
                PI - 1.0e-9,
                PI + 1.0e-9,
                TAU - 1.0e-12,
            ] {
                worst = worst.max(kepler_residual(m, e));
            }
        }
        assert!(worst < 1.0e-12, "worst residual {worst:e}");
    }

    #[test]
    fn kepler_converges_in_few_steps() {
        let mut worst = 0;
        for ei in 0..=99 {
            let e = f64::from(ei) * 0.01;
            for mi in 0..=720 {
                let m = f64::from(mi) / 720.0 * TAU;
                worst = worst.max(solve_kepler_counted(m, e).1);
            }
            for m in [0.0, 1.0e-300, 1.0e-15, 1.0e-10, 1.0e-4, PI - 1.0e-12, PI] {
                worst = worst.max(solve_kepler_counted(m, e).1);
            }
        }
        assert!(worst <= 8, "worst {worst} steps");
    }

    #[test]
    fn kepler_handles_extreme_eccentricity_near_periapsis() {
        for e in [0.95, 0.98, 0.99, 0.999] {
            for m in [0.0, 1.0e-300, 1.0e-15, 1.0e-10, 1.0e-4, 0.01, 0.1, 6.0] {
                assert!(kepler_residual(m, e) < 1.0e-12, "e {e} m {m}");
            }
        }
    }

    #[test]
    fn kepler_is_continuous_for_any_mean_anomaly() {
        for m in [-50.0, -7.0, -PI, 0.3, 7.0, 123.456] {
            assert!(kepler_residual(m, 0.6) < 1.0e-12, "m {m}");
        }
        let a = solve_kepler(TAU + 0.2, 0.3);
        let b = solve_kepler(0.2, 0.3);
        assert!((a - TAU - b).abs() < 1.0e-12);
    }

    #[test]
    fn kepler_known_values() {
        assert!(solve_kepler(0.0, 0.7).abs() < 1.0e-15);
        assert!((solve_kepler(PI, 0.9) - PI).abs() < 1.0e-15);
        assert_eq!(solve_kepler(1.234, 0.0), 1.234);
        // Meeus, Astronomical Algorithms, example 30.a: e 0.1, M 5 degrees, E 5.554589 degrees.
        let ecc_deg = solve_kepler(5.0_f64.to_radians(), 0.1).to_degrees();
        assert!((ecc_deg - 5.554_589_253_872).abs() < 1.0e-9, "{ecc_deg}");
        // Vallado, Fundamentals of Astrodynamics, example 2-1: e 0.4, M 235.4 degrees.
        let ecc = solve_kepler(235.4_f64.to_radians(), 0.4);
        assert!((ecc - 3.848_661_745).abs() < 1.0e-9, "{ecc}");
    }

    fn orbit(a: f64, e: f64, i_deg: f64, node_deg: f64, w_deg: f64, m0_deg: f64) -> OrbitElements {
        OrbitElements {
            semi_major_axis_m: a,
            eccentricity: e,
            inclination_rad: i_deg.to_radians(),
            longitude_of_ascending_node_rad: node_deg.to_radians(),
            argument_of_periapsis_rad: w_deg.to_radians(),
            mean_anomaly_at_epoch_rad: m0_deg.to_radians(),
        }
    }

    #[test]
    fn circular_orbit_returns_to_its_start_after_one_period() {
        let o = orbit(7.0e6, 0.0, 28.5, 40.0, 0.0, 10.0);
        let period_s = o.period_s(EARTH_MU);
        for t0 in [0.0, 1234.5, -800.0, 1.0e8] {
            let (p0, v0) = o.state_at(EARTH_MU, t0);
            let (p1, v1) = o.state_at(EARTH_MU, t0 + period_s);
            assert!((p1 - p0).length() < 1.0e-6 * p0.length(), "t0 {t0}");
            assert!((v1 - v0).length() < 1.0e-6 * v0.length(), "t0 {t0}");
            assert!((p0.length() - 7.0e6).abs() < 1.0e-6);
        }
        let (p0, _) = o.state_at(EARTH_MU, 0.0);
        let (p1, _) = o.state_at(EARTH_MU, period_s);
        assert_eq!(p0, p1);
        let (half, _) = o.state_at(EARTH_MU, 0.5 * period_s);
        assert!((half + p0).length() < 1.0e-6);
    }

    #[test]
    fn energy_angular_momentum_and_vis_viva_hold_over_many_orbits() {
        let o = orbit(2.4e7, 0.7, 35.0, 120.0, 250.0, 33.0);
        let a = o.semi_major_axis_m;
        let e = o.eccentricity;
        let period_s = o.period_s(EARTH_MU);
        let energy = -EARTH_MU / (2.0 * a);
        let h = (EARTH_MU * a * (1.0 - e * e)).sqrt();
        let (p_ref, v_ref) = o.state_at(EARTH_MU, 0.0);
        let h_dir = p_ref.cross(v_ref).normalize();
        for k in 0..2000 {
            let t_s = f64::from(k) * 0.618_034 * period_s;
            let (p, v) = o.state_at(EARTH_MU, t_s);
            let r = p.length();
            let eps = 0.5 * v.length_squared() - EARTH_MU / r;
            assert!(((eps - energy) / energy).abs() < 1.0e-10, "energy at {k}");
            let hv = p.cross(v);
            assert!(((hv.length() - h) / h).abs() < 1.0e-10, "h at {k}");
            assert!((hv.normalize() - h_dir).length() < 1.0e-10);
            let vis_viva = EARTH_MU * (2.0 / r - 1.0 / a);
            assert!(((v.length_squared() - vis_viva) / vis_viva).abs() < 1.0e-10);
            assert!(r >= a * (1.0 - e) * (1.0 - 1.0e-12) && r <= a * (1.0 + e) * (1.0 + 1.0e-12));
        }
        let (p_late, _) = o.state_at(EARTH_MU, 1000.0 * period_s);
        assert!((p_late - p_ref).length() < 1.0e-6 * a);
    }

    #[test]
    fn velocity_is_the_derivative_of_position() {
        let o = orbit(1.0e7, 0.4, 60.0, 15.0, 80.0, 200.0);
        let dt_s = 0.01;
        for t_s in [0.0, 500.0, 3333.0, 9000.0] {
            let (p0, _) = o.state_at(EARTH_MU, t_s - dt_s);
            let (p1, _) = o.state_at(EARTH_MU, t_s + dt_s);
            let (_, v) = o.state_at(EARTH_MU, t_s);
            let fd = (p1 - p0) / (2.0 * dt_s);
            assert!((fd - v).length() < 1.0e-5 * v.length(), "t {t_s}");
        }
    }

    #[test]
    fn orientation_follows_the_y_up_convention() {
        let flat = orbit(1.0e7, 0.2, 0.0, 0.0, 0.0, 0.0);
        let (p, v) = flat.state_at(EARTH_MU, 0.0);
        assert!(p.x > 0.0 && p.y.abs() < 1.0e-9 && p.z.abs() < 1.0e-9);
        assert!(v.z < 0.0, "prograde motion goes from +X toward -Z");
        assert!((p.cross(v).normalize() - DVec3::Y).length() < 1.0e-12);

        let (i, node) = (30.0_f64.to_radians(), 50.0_f64.to_radians());
        let tilted = orbit(1.0e7, 0.2, 30.0, 50.0, 0.0, 0.0);
        let (p, v) = tilted.state_at(EARTH_MU, 0.0);
        let node_dir = DVec3::new(node.cos(), 0.0, -node.sin());
        assert!(
            (p.normalize() - node_dir).length() < 1.0e-12,
            "periapsis at the node"
        );
        assert!(
            v.y > 0.0,
            "the ascending node goes from below to above the ecliptic"
        );
        let normal = DVec3::new(i.sin() * node.sin(), i.cos(), i.sin() * node.cos());
        assert!((p.cross(v).normalize() - normal).length() < 1.0e-12);
    }

    #[test]
    fn rotation_spins_about_the_tilted_axis() {
        let rot = RotationState {
            sidereal_period_s: 3600.0,
            axial_tilt_rad: 15.0_f64.to_radians(),
            prime_meridian_at_epoch_rad: 0.0,
        };
        let tilt = rot.axial_tilt_rad;
        for t_s in [0.0, 100.0, 1800.0, 1.0e9] {
            let pole = rot.body_to_inertial(t_s) * DVec3::Y;
            assert!((pole - DVec3::new(0.0, tilt.cos(), tilt.sin())).length() < 1.0e-12);
        }
        let quarter = rot.body_to_inertial(900.0) * DVec3::X;
        let expected = DQuat::from_rotation_x(tilt) * DVec3::NEG_Z;
        assert!((quarter - expected).length() < 1.0e-12);
        let day = rot.body_to_inertial(3600.0 * 1.0e5) * DVec3::X;
        assert!((day - DVec3::X).length() < 1.0e-12);

        let retro = RotationState {
            sidereal_period_s: -3600.0,
            ..rot
        };
        let quarter = retro.body_to_inertial(900.0) * DVec3::X;
        let expected = DQuat::from_rotation_x(tilt) * DVec3::Z;
        assert!((quarter - expected).length() < 1.0e-12);
    }
}
