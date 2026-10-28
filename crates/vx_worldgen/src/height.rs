//! The planet's height function `h(dir)` (D21): seeded fBm of the unit direction scaled by the
//! radius, shaped like the M1 heightfield, plus the 3D detail noise.
//!
//! The fractal is summed here octave by octave (one noise evaluation per octave, each with its own
//! seed) and normalised by the amplitude sum of all octaves, so dropping the short octaves at a
//! coarse level removes them and changes nothing else (the noise crate's own fBm renormalises when
//! the octave count changes).
//!
//! The height octaves use OpenSimplex2S, the smooth variant. The fast OpenSimplex2 3D kernel of
//! fastnoise-lite 1.1.1 adds only one of two equally distant second lattice points where two
//! offsets tie, so it jumps by up to 0.5 % of its amplitude across the tie planes (measured by
//! `noise_kernels_are_continuous_where_offsets_tie`). Its rotation is symmetric in x, y and z, so
//! six of the twelve cube edges (where two components are equal) lie inside such planes at every
//! octave: about 0.5 m steps of the default height along those edges. OpenSimplex2S is continuous
//! there and costs about 10 % more per evaluation. The detail noise keeps the M1 kernel (its
//! steps are about 1.5 cm at the default 3 m amplitude).

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use glam::DVec3;
use vx_core::TerrainParams;

use crate::seed::{STREAM_DETAIL, STREAM_HEIGHT, derive_seed};

/// Lacunarity and gain of the height fractal (as M1), set here so a change of the noise crate's
/// defaults cannot change a world (D5).
pub(crate) const LACUNARITY: f64 = 2.0;
pub(crate) const GAIN: f64 = 0.5;

/// Weight `w` of the signed square term in the shaping `(1 - w) n + w n |n|` (as M1): lowlands
/// flatten towards the base height while hills keep the full amplitude.
pub(crate) const HILL_SHARPNESS: f64 = 0.5;

/// Most octaves [`TerrainParams::OCTAVES_RANGE`] allows.
const MAX_OCTAVES: usize = 12;

pub(crate) struct HeightNoise {
    octaves: Vec<Octave>,
    inv_amplitude_sum: f64,
    base_height_m: f64,
    amplitude_m: f64,
    detail: FastNoiseLite,
    inv_detail_wavelength: f64,
    detail_wavelength_m: f64,
    detail_amplitude_m: f64,
}

struct Octave {
    noise: FastNoiseLite,
    amplitude: f64,
    /// Noise coordinate per unit of direction: radius over the octave's wavelength.
    scale: f64,
    wavelength_m: f64,
}

impl HeightNoise {
    pub(crate) fn new(seed: u64, terrain: &TerrainParams, radius_m: f64) -> Self {
        let height_seed = derive_seed(seed, STREAM_HEIGHT);
        let count = (terrain.octaves as usize).min(MAX_OCTAVES);
        let mut octaves = Vec::with_capacity(count);
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut amplitude_sum = 0.0;
        for k in 0..count {
            octaves.push(Octave {
                // `derive_seed` stays below 2^30, so adding the octave index cannot overflow.
                noise: single_noise(NoiseType::OpenSimplex2S, height_seed + k as i32),
                amplitude,
                scale: radius_m / terrain.wavelength_m * frequency,
                wavelength_m: terrain.wavelength_m / frequency,
            });
            amplitude_sum += amplitude;
            amplitude *= GAIN;
            frequency *= LACUNARITY;
        }
        Self {
            octaves,
            inv_amplitude_sum: if amplitude_sum > 0.0 {
                1.0 / amplitude_sum
            } else {
                0.0
            },
            base_height_m: terrain.base_height_m,
            amplitude_m: terrain.amplitude_m,
            detail: single_noise(NoiseType::OpenSimplex2, derive_seed(seed, STREAM_DETAIL)),
            inv_detail_wavelength: 1.0 / terrain.detail_wavelength_m,
            detail_wavelength_m: terrain.detail_wavelength_m,
            detail_amplitude_m: terrain.detail_amplitude_m,
        }
    }

    pub(crate) fn octave_count(&self) -> usize {
        self.octaves.len()
    }

    /// Octaves whose wavelength is at least `min_wavelength_m`: a prefix, since wavelengths
    /// shrink with the octave index.
    pub(crate) fn octaves_at_least(&self, min_wavelength_m: f64) -> usize {
        self.octaves
            .iter()
            .take_while(|o| o.wavelength_m >= min_wavelength_m)
            .count()
    }

    pub(crate) fn amplitude_m(&self) -> f64 {
        self.amplitude_m
    }

    pub(crate) fn base_height_m(&self) -> f64 {
        self.base_height_m
    }

    /// Height above the mean radius along the unit direction `dir`, from the first `octaves`
    /// octaves. Always within `base_height_m +- amplitude_m`.
    #[inline]
    pub(crate) fn height(&self, dir: DVec3, octaves: usize) -> f64 {
        let mut sum = 0.0;
        for o in &self.octaves[..octaves] {
            let n = o
                .noise
                .get_noise_3d(dir.x * o.scale, dir.y * o.scale, dir.z * o.scale);
            sum += o.amplitude * f64::from(n);
        }
        let n = sum * self.inv_amplitude_sum;
        let shaped = (1.0 - HILL_SHARPNESS) * n + HILL_SHARPNESS * n * n.abs();
        self.base_height_m + self.amplitude_m * shaped.clamp(-1.0, 1.0)
    }

    pub(crate) fn detail_amplitude_m(&self) -> f64 {
        self.detail_amplitude_m
    }

    pub(crate) fn detail_wavelength_m(&self) -> f64 {
        self.detail_wavelength_m
    }

    /// The M1 detail noise at `p` (body fixed metres), within `+-detail_amplitude_m`.
    #[inline]
    pub(crate) fn detail(&self, p: DVec3) -> f64 {
        let n = self.detail.get_noise_3d(
            p.x * self.inv_detail_wavelength,
            p.y * self.inv_detail_wavelength,
            p.z * self.inv_detail_wavelength,
        );
        self.detail_amplitude_m * f64::from(n).clamp(-1.0, 1.0)
    }
}

fn single_noise(noise_type: NoiseType, seed: i32) -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(seed);
    noise.set_noise_type(Some(noise_type));
    noise.set_frequency(Some(1.0));
    noise.set_fractal_type(Some(FractalType::None));
    noise
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;

    /// Largest change of the noise between points on the plane `y = z` (two equal components,
    /// as on six cube edges) and points 1e-9 off it.
    fn tie_plane_jump(noise_type: NoiseType) -> f32 {
        let noise = single_noise(noise_type, 1234);
        let mut worst = 0.0f32;
        for k in 0..100_000 {
            let t = f64::from(k) * 0.00731;
            let p = DVec3::new(t * 0.37 + 0.1, t, t);
            let a = noise.get_noise_3d(p.x, p.y, p.z);
            let b = noise.get_noise_3d(p.x, p.y + 1e-9, p.z - 1e-9);
            worst = worst.max((a - b).abs());
        }
        worst
    }

    #[test]
    fn noise_kernels_are_continuous_where_offsets_tie() {
        let fast = tie_plane_jump(NoiseType::OpenSimplex2);
        let smooth = tie_plane_jump(NoiseType::OpenSimplex2S);
        println!("tie plane jump: OpenSimplex2 {fast:e}, OpenSimplex2S {smooth:e}");
        assert!(
            fast > 1e-3,
            "the fast kernel's jump is the reason for OpenSimplex2S: {fast}"
        );
        assert!(smooth < 1e-5, "{smooth}");
    }
}
