//! Cube sphere parameterisation of directions (D3, D21).
//!
//! A direction `d` belongs to the face of its largest absolute component (ties go to x, then y),
//! and its gnomonic coordinates on that face are the two minor components divided by the major
//! one: `d` is parallel to `sign * e_major + x * e_u + y * e_v`. The gnomonic coordinate `x` is
//! then mapped to a grid coordinate `a` in -1 to 1 by the rational map
//!
//! `a = F(x) = (1 + c) x / (1 + c |x|)`, inverse `x = G(a) = a / ((1 + c) - c |a|)`, `c = 1/4`,
//!
//! an algebraic stand in for the equal angle map `a = (4 / pi) atan(x)`: both pass through 0 and
//! +-1, `F` has the equal angle map's slope at the face centre, and `F` and `G` are exact inverses
//! with only `abs`, multiplication and division (D13: no `atan` or `tan` in the sample path). Over
//! a face, the arc between neighbouring grid nodes lies within 0.72 and 1.083 times the mean (the
//! equal angle map gives 0.71 to 1.0; the plain gnomonic grid 0.60 to 1.27). Both maps extend
//! smoothly past +-1 (to `|a| < 5`), so the ghost nodes just outside a face are ordinary
//! directions.

use glam::DVec3;

/// Largest arc between neighbouring nodes of a face grid over the mean arc `pi / (2 n)` (measured
/// over a 2,000 cell grid by `spacing_extremes_match_the_documented_bounds`, rounded up).
pub(crate) const SPACING_MAX_OVER_MEAN: f64 = 1.0825;

/// `F(x) = 5 x / (4 + |x|)`, the `c = 1/4` map with exact binary constants.
const MAP_NUMERATOR: f64 = 5.0;
const MAP_OFFSET: f64 = 4.0;
/// `F'(x) = 20 / (4 + |x|)^2`.
const MAP_DERIVATIVE_NUMERATOR: f64 = 20.0;

/// One of the six cube faces: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z. The face's gnomonic axes are
/// (y, z) on the X faces, (z, x) on the Y faces and (x, y) on the Z faces.
pub(crate) type CubeFace = u8;

/// Major axis index, sign and the two minor axis indices of a face.
#[inline(always)]
pub(crate) const fn face_axes(face: CubeFace) -> (usize, f64, usize, usize) {
    let major = (face / 2) as usize;
    let sign = if face.is_multiple_of(2) { 1.0 } else { -1.0 };
    (major, sign, (major + 1) % 3, (major + 2) % 3)
}

/// The face of `p` and the components along its major, u and v axes, with the major component
/// as an absolute value.
#[inline(always)]
pub(crate) fn face_of(p: DVec3) -> (CubeFace, f64, f64, f64) {
    let (ax, ay, az) = (p.x.abs(), p.y.abs(), p.z.abs());
    if ax >= ay && ax >= az {
        (u8::from(p.x < 0.0), ax, p.y, p.z)
    } else if ay >= az {
        (2 + u8::from(p.y < 0.0), ay, p.z, p.x)
    } else {
        (4 + u8::from(p.z < 0.0), az, p.x, p.y)
    }
}

/// Gnomonic coordinates of `p` on `face`, or `None` when `p` is not in front of the face's plane.
#[inline(always)]
pub(crate) fn gnomonic_on(face: CubeFace, p: DVec3) -> Option<(f64, f64)> {
    let (major, sign, u, v) = face_axes(face);
    let m = p[major] * sign;
    (m > 0.0).then(|| {
        let inv = 1.0 / m;
        (p[u] * inv, p[v] * inv)
    })
}

/// `1 / (4 + |x|)`, shared by [`grid_from_gnomonic`] and the slope's chain rule.
#[inline(always)]
pub(crate) fn map_inv_denominator(x: f64) -> f64 {
    1.0 / (MAP_OFFSET + x.abs())
}

/// `F(x)` given `inv_d = map_inv_denominator(x)`.
#[inline(always)]
pub(crate) fn grid_from_gnomonic(x: f64, inv_d: f64) -> f64 {
    MAP_NUMERATOR * x * inv_d
}

/// `F'(x)` given `inv_d = map_inv_denominator(x)`.
#[inline(always)]
pub(crate) fn grid_derivative(inv_d: f64) -> f64 {
    MAP_DERIVATIVE_NUMERATOR * inv_d * inv_d
}

/// `G(a)`, the inverse of `F`.
#[inline(always)]
pub(crate) fn gnomonic_from_grid(a: f64) -> f64 {
    MAP_OFFSET * a / (MAP_NUMERATOR - a.abs())
}

/// Grid coordinate `a` in -1 to 1 of node index `i` on a face of `n` cells (exact for `i` in
/// range; ghost nodes give `|a|` slightly above 1).
#[inline(always)]
pub(crate) fn node_grid_coordinate(i: i32, n: i32) -> f64 {
    f64::from(2 * i - n) / f64::from(n)
}

/// Unit direction of node `(i, j)` of `face` on a grid of `n` cells per edge.
#[cfg(test)]
pub(crate) fn node_direction(face: CubeFace, n: i32, i: i32, j: i32) -> DVec3 {
    node_direction_from(
        face,
        gnomonic_from_grid(node_grid_coordinate(i, n)),
        gnomonic_from_grid(node_grid_coordinate(j, n)),
    )
}

/// Unit direction of gnomonic coordinates `(x, y)` on `face`.
#[inline]
pub(crate) fn node_direction_from(face: CubeFace, x: f64, y: f64) -> DVec3 {
    let (major, sign, u, v) = face_axes(face);
    let mut q = DVec3::ZERO;
    q[major] = sign;
    q[u] = x;
    q[v] = y;
    q * (1.0 / (1.0 + x * x + y * y).sqrt())
}

/// Cells per face edge for a grid whose mean node spacing at `radius_m` is at most `spacing_m`
/// (the quarter great circle over the cell count), at least 2.
pub(crate) fn cells_per_edge(radius_m: f64, spacing_m: f64) -> i32 {
    let cells = (std::f64::consts::FRAC_PI_2 * radius_m / spacing_m).ceil();
    // Saturating float to int conversion; i32::MAX / 4 keeps `2 * i - n` and the tile arithmetic
    // in range for any configuration.
    (cells as i32).clamp(2, i32::MAX / 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_and_inverse_round_trip() {
        for k in -1000..=1000 {
            let x = f64::from(k) / 1000.0 * 1.2;
            let a = grid_from_gnomonic(x, map_inv_denominator(x));
            let back = gnomonic_from_grid(a);
            assert!((back - x).abs() <= 4.0 * f64::EPSILON, "{x}: {back}");
        }
        assert_eq!(grid_from_gnomonic(1.0, map_inv_denominator(1.0)), 1.0);
        assert_eq!(grid_from_gnomonic(-1.0, map_inv_denominator(-1.0)), -1.0);
        assert_eq!(gnomonic_from_grid(1.0), 1.0);
        assert_eq!(gnomonic_from_grid(0.0), 0.0);
        // Monotone, and its slope at the centre is the equal angle map's 4 / pi to within 2.5 %.
        let slope_at_zero = grid_derivative(map_inv_denominator(0.0));
        assert!(
            (slope_at_zero - 4.0 / std::f64::consts::PI).abs() < 0.025 * 4.0 / std::f64::consts::PI
        );
    }

    #[test]
    fn face_selection_breaks_ties_towards_x_then_y() {
        assert_eq!(face_of(DVec3::new(1.0, 1.0, 1.0)).0, 0);
        assert_eq!(face_of(DVec3::new(-1.0, 1.0, 1.0)).0, 1);
        assert_eq!(face_of(DVec3::new(0.5, -1.0, 1.0)).0, 3);
        assert_eq!(face_of(DVec3::new(0.5, 0.2, -1.0)).0, 5);
        assert_eq!(face_of(DVec3::ZERO).0, 0);
        for face in 0..6u8 {
            let d = node_direction(face, 64, 32, 32);
            assert_eq!(face_of(d).0, face);
            let (x, y) = gnomonic_on(face, d).unwrap();
            assert!(x.abs() < 1e-15 && y.abs() < 1e-15);
            assert!(gnomonic_on(face ^ 1, d).is_none());
        }
    }

    #[test]
    fn nodes_round_trip_through_the_sample_mapping() {
        let n = 157;
        for face in 0..6u8 {
            for (i, j) in [(0, 0), (n, n), (3, 150), (78, 79), (n, 0)] {
                let d = node_direction(face, n, i, j);
                assert!((d.length() - 1.0).abs() < 1e-15);
                let (x, y) = gnomonic_on(face, d).unwrap();
                let u = (grid_from_gnomonic(x, map_inv_denominator(x)) + 1.0) * f64::from(n) * 0.5;
                let v = (grid_from_gnomonic(y, map_inv_denominator(y)) + 1.0) * f64::from(n) * 0.5;
                assert!((u - f64::from(i)).abs() < 1e-9, "{face} {i}: {u}");
                assert!((v - f64::from(j)).abs() < 1e-9, "{face} {j}: {v}");
            }
        }
    }

    #[test]
    fn spacing_extremes_match_the_documented_bounds() {
        let n = 2000;
        let mean = std::f64::consts::FRAC_PI_2 / f64::from(n);
        let mut lo = f64::MAX;
        let mut hi = 0.0f64;
        // One face suffices (all six are congruent); every second row keeps the test quick.
        for j in (0..=n).step_by(2) {
            for i in 0..n {
                let a = node_direction(0, n, i, j);
                let b = node_direction(0, n, i + 1, j);
                // Chord to arc: 2 asin(c / 2); the chord is within 1e-7 of the arc here.
                let ratio = (a - b).length() / mean;
                lo = lo.min(ratio);
                hi = hi.max(ratio);
            }
        }
        println!("node spacing over the mean: {lo:.4} to {hi:.4}");
        assert!(hi <= SPACING_MAX_OVER_MEAN, "{hi}");
        assert!(hi > SPACING_MAX_OVER_MEAN - 0.002, "bound is loose: {hi}");
        assert!(lo > 0.70, "{lo}");
    }

    #[test]
    fn cells_per_edge_keeps_the_mean_spacing_under_the_target() {
        for (radius_m, spacing_m) in [(10_000.0, 1.0), (10_000.0, 128.0), (1000.0, 4096.0)] {
            let n = cells_per_edge(radius_m, spacing_m);
            let mean = std::f64::consts::FRAC_PI_2 * radius_m / f64::from(n);
            assert!(
                mean <= spacing_m * (1.0 + 1e-12),
                "{radius_m} {spacing_m}: {mean}"
            );
            assert!(n >= 2);
        }
    }
}
