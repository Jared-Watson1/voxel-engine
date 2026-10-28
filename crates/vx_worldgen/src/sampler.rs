use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use glam::{DVec3, IVec3};
use vx_core::TerrainParams;
use vx_voxel::{Chunk, ChunkKey, DISTANCE_UNITS_PER_VOXEL, Material, SAMPLES_PER_AXIS, Voxel};

use crate::seed::{STREAM_DETAIL, STREAM_HEIGHT, derive_seed};

/// Step of the central differences that estimate the heightfield gradient, in metres. A numerical
/// differentiation step rather than a world scale (D11): it stays fixed as the voxel size changes,
/// so the slope factor and the rock rule describe the same terrain at every voxel size.
const GRADIENT_STEP_M: f64 = 1.0;

/// Lacunarity and gain of the heightfield fractal, set explicitly so a change of the noise crate's
/// defaults cannot change a world (D5).
const LACUNARITY: f32 = 2.0;
const GAIN: f32 = 0.5;

/// Weight `w` of the signed square term in the heightfield shaping `(1 - w) n + w n |n|`:
/// lowlands flatten towards the base height while hills keep the full amplitude. Dimensionless.
const HILL_SHARPNESS: f64 = 0.5;

/// Upper bound of the gradient magnitude of one OpenSimplex2 2D octave per unit of input
/// coordinate, derived from the kernel in fastnoise-lite 1.1.1. Each of the three corner terms is
/// `a^4 (g . d)` with a unit gradient `g` and `a = 0.5 - |d|^2`, whose gradient magnitude is at
/// most `4 a^3 - 7 a^4 <= 189 / 2401` (reached at `a = 3 / 7`). Three corners and the output scale
/// 99.83685 (rounded up) give the bound. The skew in the coordinate transform and the unskew in
/// the kernel compose to the identity, so `d` is an offset in input coordinates.
const SIMPLEX_2D_SLOPE_BOUND: f64 = 3.0 * (189.0 / 2401.0) * 99.84;

/// Safety factor and absolute slack on the heightfield slope bound. They cover what the exact
/// analysis ignores: the f32 arithmetic inside the noise crate, its f32 fractal bounding and the
/// finite differences of rounded values.
const SLOPE_BOUND_SAFETY: f64 = 1.125;
const SLOPE_BOUND_SLACK: f64 = 1e-2;

/// Largest gradient step, in voxels, for which `fill_chunk` shares column heights between
/// neighbouring columns instead of evaluating the four gradient taps per column.
const MAX_SHARED_STEP_VOXELS: f64 = 16.0;

const N: usize = SAMPLES_PER_AXIS;

/// Samples the M1 terrain. Every method is a pure function of its arguments, the seed and the
/// parameters: no interior mutability that affects results, no hash map iteration, no
/// transcendental functions whose results differ between platforms (D13).
#[derive(Debug, Clone)]
pub struct TerrainSampler {
    seed: u64,
    params: TerrainParams,
    height_seed: i32,
    detail_seed: i32,
    inv_wavelength: f64,
    inv_detail_wavelength: f64,
    /// Lower bound of the slope factor anywhere on the heightfield, from
    /// [`slope_bound_per_axis`]. Lets `fill_chunk` classify whole chunks and whole columns
    /// without evaluating their gradients.
    slope_factor_min: f64,
    /// Depth below which the solid is rock whatever the slope and height.
    rock_depth_m: f64,
}

impl TerrainSampler {
    pub fn new(seed: u64, params: TerrainParams) -> Self {
        let slope_bound = slope_bound_per_axis(&params);
        Self {
            seed,
            params,
            height_seed: derive_seed(seed, STREAM_HEIGHT),
            detail_seed: derive_seed(seed, STREAM_DETAIL),
            inv_wavelength: 1.0 / params.wavelength_m,
            inv_detail_wavelength: 1.0 / params.detail_wavelength_m,
            slope_factor_min: 1.0 / (1.0 + 2.0 * slope_bound * slope_bound).sqrt(),
            rock_depth_m: params.grass_depth_m.max(params.soil_depth_m),
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn params(&self) -> &TerrainParams {
        &self.params
    }

    /// Surface height in metres at a horizontal position (the heightfield without 3D detail).
    pub fn height_m(&self, x_m: f64, z_m: f64) -> f64 {
        self.height_with(&self.height_noise(), x_m, z_m)
    }

    /// Signed distance to the terrain surface in metres (negative inside) and the material of the
    /// solid at `p`. The material is meaningful only where the distance is negative.
    pub fn sample(&self, p: DVec3) -> (f64, Material) {
        let height_noise = self.height_noise();
        let detail_noise = self.detail_noise();
        let h = self.height_with(&height_noise, p.x, p.z);
        let (slope, slope_factor) = self.slopes(&height_noise, p.x, p.z);
        let distance_m = (p.y - h) * slope_factor + self.detail_with(&detail_noise, p);
        (distance_m, self.material_at(h, slope, h - p.y))
    }

    /// Fills every sample of `chunk`, apron included, with exactly the values [`Self::sample`]
    /// returns at each sample position (after the chunk's quantisation). Implementations may skip
    /// work for samples provably outside the distance band, but the result must be bit identical
    /// to the per sample path.
    ///
    /// Three levels of early out, all with a margin of one quantisation unit so rounding cannot
    /// move a skipped sample into the band: the whole chunk against the amplitude bounds, each
    /// column against its own height using the slope factor lower bound, and each sample against
    /// the column's exact slope factor. Only samples that survive all three evaluate the 3D detail
    /// noise.
    pub fn fill_chunk(&self, chunk: &mut Chunk) {
        profiling::scope!("TerrainSampler::fill_chunk");
        let voxel_size_m = chunk.voxel_size_m();
        let plan = FillPlan::new(chunk.key(), voxel_size_m);
        let detail_amplitude_m = self.params.detail_amplitude_m;
        let sf_min = self.slope_factor_min;

        let h_hi = self.params.base_height_m + self.params.amplitude_m;
        let h_lo = self.params.base_height_m - self.params.amplitude_m;
        if (plan.y_lo - h_hi) * sf_min - detail_amplitude_m > plan.coarse_skip_m {
            chunk.fill_uniform(Voxel::AIR);
            return;
        }
        if (plan.y_hi - h_lo) * sf_min + detail_amplitude_m < -plan.coarse_skip_m
            && h_lo - plan.y_hi > self.rock_depth_m + plan.unit_m
        {
            chunk.fill_uniform(Voxel::solid(Material::ROCK));
            return;
        }

        let height_noise = self.height_noise();
        let detail_noise = self.detail_noise();
        let mut heights = [0.0f64; N * N];
        for (iz, &z) in plan.zs.iter().enumerate() {
            for (ix, &x) in plan.xs.iter().enumerate() {
                heights[iz * N + ix] = self.height_with(&height_noise, x, z);
            }
        }

        let share_step = SharedHeights::step_for(voxel_size_m);
        let mut shared: Option<SharedHeights> = None;
        let samples = chunk.samples_mut();
        for (iz, &z) in plan.zs.iter().enumerate() {
            for (ix, &x) in plan.xs.iter().enumerate() {
                let h = heights[iz * N + ix];
                let column = |j: usize| Chunk::linear(ix, j, iz);

                if (plan.y_lo - h) * sf_min - detail_amplitude_m > plan.coarse_skip_m {
                    for j in 0..N {
                        samples[column(j)] = Voxel::AIR;
                    }
                    continue;
                }

                let all_solid = (plan.y_hi - h) * sf_min + detail_amplitude_m < -plan.coarse_skip_m;
                let slope_matters =
                    h >= self.params.sand_below_m && h - plan.y_hi < self.params.grass_depth_m;
                if all_solid && !slope_matters {
                    // Slope unused: every sample is at least `grass_depth_m` deep or the column
                    // is a beach, so `material_at` never reaches the grass or rock branch.
                    for (j, &y) in plan.ys.iter().enumerate() {
                        samples[column(j)] = Voxel::solid(self.material_at(h, 0.0, h - y));
                    }
                    continue;
                }

                let from_shared = share_step.and_then(|step| {
                    shared
                        .get_or_insert_with(|| {
                            SharedHeights::build(self, &height_noise, &plan, &heights, step)
                        })
                        .slopes(ix, iz, x, z)
                });
                let (slope, slope_factor) =
                    from_shared.unwrap_or_else(|| self.slopes(&height_noise, x, z));
                // `scaled` never decreases with `y` and rounding is monotone, so the samples
                // saturated to solid form a prefix of the column and those saturated to air a
                // suffix. Binary searches on the same predicates find both; only the samples
                // between them evaluate the detail noise.
                let scaled = |y: f64| (y - h) * slope_factor;
                let solid_end = plan
                    .ys
                    .partition_point(|&y| scaled(y) + detail_amplitude_m < -plan.skip_m);
                let air_start = plan
                    .ys
                    .partition_point(|&y| scaled(y) - detail_amplitude_m <= plan.skip_m);
                for (j, &y) in plan.ys.iter().enumerate().take(solid_end) {
                    samples[column(j)] = Voxel::solid(self.material_at(h, slope, h - y));
                }
                for (j, &y) in plan.ys.iter().enumerate().take(air_start).skip(solid_end) {
                    let distance_m =
                        scaled(y) + self.detail_with(&detail_noise, DVec3::new(x, y, z));
                    let distance = Voxel::quantise_distance(distance_m * plan.inv_voxel);
                    let material = if distance < 0 {
                        self.material_at(h, slope, h - y)
                    } else {
                        Material::AIR
                    };
                    samples[column(j)] = Voxel { distance, material };
                }
                for j in air_start..N {
                    samples[column(j)] = Voxel::AIR;
                }
            }
        }
    }

    /// Lower bound of the slope factor over the whole heightfield.
    #[cfg(test)]
    pub(crate) fn slope_factor_min(&self) -> f64 {
        self.slope_factor_min
    }

    /// Slope magnitude (rise over run) and slope factor at a horizontal position.
    #[cfg(test)]
    pub(crate) fn slope_and_factor(&self, x_m: f64, z_m: f64) -> (f64, f64) {
        self.slopes(&self.height_noise(), x_m, z_m)
    }

    fn height_noise(&self) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(self.height_seed);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(1.0));
        noise.set_fractal_type(Some(FractalType::FBm));
        noise.set_fractal_lacunarity(Some(LACUNARITY));
        noise.set_fractal_gain(Some(GAIN));
        // The octave setter recomputes the fractal bounding, so it comes after the gain.
        noise.set_fractal_octaves(Some(i32::try_from(self.params.octaves).unwrap_or(i32::MAX)));
        noise
    }

    fn detail_noise(&self) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(self.detail_seed);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(1.0));
        noise.set_fractal_type(Some(FractalType::None));
        noise
    }

    fn height_with(&self, noise: &FastNoiseLite, x_m: f64, z_m: f64) -> f64 {
        let n = f64::from(noise.get_noise_2d(x_m * self.inv_wavelength, z_m * self.inv_wavelength));
        let shaped = (1.0 - HILL_SHARPNESS) * n + HILL_SHARPNESS * n * n.abs();
        self.params.base_height_m + self.params.amplitude_m * shaped.clamp(-1.0, 1.0)
    }

    fn slopes(&self, noise: &FastNoiseLite, x_m: f64, z_m: f64) -> (f64, f64) {
        slopes_from_taps(
            self.height_with(noise, x_m - GRADIENT_STEP_M, z_m),
            self.height_with(noise, x_m + GRADIENT_STEP_M, z_m),
            self.height_with(noise, x_m, z_m - GRADIENT_STEP_M),
            self.height_with(noise, x_m, z_m + GRADIENT_STEP_M),
        )
    }

    fn detail_with(&self, noise: &FastNoiseLite, p: DVec3) -> f64 {
        let n = noise.get_noise_3d(
            p.x * self.inv_detail_wavelength,
            p.y * self.inv_detail_wavelength,
            p.z * self.inv_detail_wavelength,
        );
        self.params.detail_amplitude_m * f64::from(n).clamp(-1.0, 1.0)
    }

    /// Material of the solid at `depth_m` below a surface of height `height_m` and slope `slope`.
    fn material_at(&self, height_m: f64, slope: f64, depth_m: f64) -> Material {
        let p = &self.params;
        if depth_m < p.grass_depth_m {
            if height_m < p.sand_below_m {
                Material::SAND
            } else if slope > p.rock_slope {
                Material::ROCK
            } else {
                Material::GRASS
            }
        } else if depth_m < p.soil_depth_m {
            Material::SOIL
        } else {
            Material::ROCK
        }
    }
}

/// Slope magnitude and slope factor from the four central difference taps of a column.
fn slopes_from_taps(hx0: f64, hx1: f64, hz0: f64, hz1: f64) -> (f64, f64) {
    let gx = (hx1 - hx0) / (2.0 * GRADIENT_STEP_M);
    let gz = (hz1 - hz0) / (2.0 * GRADIENT_STEP_M);
    let g2 = gx * gx + gz * gz;
    (g2.sqrt(), 1.0 / (1.0 + g2).sqrt())
}

/// Upper bound of the magnitude of either central difference of the heightfield, in metres per
/// metre. Octave `k` of the fractal has amplitude `gain^k / sum(gain^i)` and frequency
/// `lacunarity^k`, the shaping multiplies slopes by at most `1 + HILL_SHARPNESS`, and the mean
/// value theorem carries the derivative bound over to the finite differences.
fn slope_bound_per_axis(params: &TerrainParams) -> f64 {
    let gain = f64::from(GAIN);
    let lacunarity = f64::from(LACUNARITY);
    let mut amplitude_sum = 0.0;
    let mut slope_sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    for _ in 0..params.octaves {
        amplitude_sum += amplitude;
        slope_sum += amplitude * frequency;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    let fractal_slope = slope_sum / amplitude_sum.max(1.0) * SIMPLEX_2D_SLOPE_BOUND;
    let height_slope =
        params.amplitude_m * (1.0 + HILL_SHARPNESS) * fractal_slope / params.wavelength_m;
    height_slope * SLOPE_BOUND_SAFETY + SLOPE_BOUND_SLACK
}

/// Sample positions and distance thresholds of one chunk fill. Positions are computed with the
/// same expressions as `Chunk::fill_from_with_material` so the per sample path sees identical
/// inputs.
struct FillPlan {
    xs: [f64; N],
    ys: [f64; N],
    zs: [f64; N],
    y_lo: f64,
    y_hi: f64,
    inv_voxel: f64,
    voxel_size_m: f64,
    /// World sample index of local sample (0, 0, 0), the base of every position: positions are
    /// `(index) as f64 * voxel_size_m` with one rounding, as `Chunk::sample_position` computes
    /// them, so apron copies agree bit for bit (D16).
    first_sample: IVec3,
    /// One quantisation unit in metres.
    unit_m: f64,
    /// A sample whose distance lies this far outside the surface quantises to the saturated value
    /// whatever the rounding: the band plus one unit.
    skip_m: f64,
    /// Threshold for tests that use the slope factor lower bound instead of the exact one, with
    /// one more unit for the rounding of that bound.
    coarse_skip_m: f64,
}

impl FillPlan {
    fn new(key: ChunkKey, voxel_size_m: f64) -> Self {
        let unit_m = voxel_size_m / f64::from(DISTANCE_UNITS_PER_VOXEL);
        let band_m = f64::from(i8::MAX) * unit_m;
        let skip_m = band_m + unit_m;
        let first_sample = key.world_sample(IVec3::ZERO);
        let axis = |first: i32| -> [f64; N] {
            std::array::from_fn(|i| f64::from(first + i as i32 - 1) * voxel_size_m)
        };
        let ys = axis(first_sample.y);
        Self {
            xs: axis(first_sample.x),
            ys,
            zs: axis(first_sample.z),
            y_lo: ys[0],
            y_hi: ys[N - 1],
            inv_voxel: 1.0 / voxel_size_m,
            voxel_size_m,
            first_sample,
            unit_m,
            skip_m,
            coarse_skip_m: skip_m + unit_m,
        }
    }
}

/// Column heights on a lattice extended by `step` samples on each side, so a column's gradient
/// taps at `GRADIENT_STEP_M` are other lattice points when the step is a whole number of voxels.
/// Every tap is checked for bit equality with the position the per sample path would use before
/// its height is reused.
struct SharedHeights {
    step: usize,
    width: usize,
    xs: Vec<f64>,
    zs: Vec<f64>,
    heights: Vec<f64>,
}

impl SharedHeights {
    /// The gradient step in samples when it is a whole number of voxels, else `None`.
    fn step_for(voxel_size_m: f64) -> Option<usize> {
        let step_voxels = GRADIENT_STEP_M / voxel_size_m;
        ((1.0..=MAX_SHARED_STEP_VOXELS).contains(&step_voxels) && step_voxels.fract() == 0.0)
            .then_some(step_voxels as usize)
    }

    fn build(
        sampler: &TerrainSampler,
        noise: &FastNoiseLite,
        plan: &FillPlan,
        inner: &[f64; N * N],
        step: usize,
    ) -> Self {
        let width = N + 2 * step;
        let axis = |first: i32| -> Vec<f64> {
            (0..width)
                .map(|g| {
                    let local = g as i32 - step as i32;
                    f64::from(first + local - 1) * plan.voxel_size_m
                })
                .collect()
        };
        let xs = axis(plan.first_sample.x);
        let zs = axis(plan.first_sample.z);
        let mut heights = vec![0.0f64; width * width];
        for gz in 0..width {
            for gx in 0..width {
                let inside = (step..step + N).contains(&gx) && (step..step + N).contains(&gz);
                heights[gz * width + gx] = if inside {
                    inner[(gz - step) * N + (gx - step)]
                } else {
                    sampler.height_with(noise, xs[gx], zs[gz])
                };
            }
        }
        Self {
            step,
            width,
            xs,
            zs,
            heights,
        }
    }

    /// Slope and slope factor of column `(ix, iz)` at position `(x, z)` from shared heights, or
    /// `None` when a tap position differs from the per sample path's by rounding.
    fn slopes(&self, ix: usize, iz: usize, x: f64, z: f64) -> Option<(f64, f64)> {
        let (gx, gz) = (ix + self.step, iz + self.step);
        let (gx0, gx1) = (gx - self.step, gx + self.step);
        let (gz0, gz1) = (gz - self.step, gz + self.step);
        let taps_match = self.xs[gx0] == x - GRADIENT_STEP_M
            && self.xs[gx1] == x + GRADIENT_STEP_M
            && self.zs[gz0] == z - GRADIENT_STEP_M
            && self.zs[gz1] == z + GRADIENT_STEP_M;
        if !taps_match {
            return None;
        }
        let at = |cx: usize, cz: usize| self.heights[cz * self.width + cx];
        Some(slopes_from_taps(
            at(gx0, gz),
            at(gx1, gz),
            at(gx, gz0),
            at(gx, gz1),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use vx_core::{CHUNK_SIZE, WorldConfig};
    use vx_voxel::ChunkKey;

    use super::*;

    const WORLD_SEED: u64 = 1;

    fn sampler() -> TerrainSampler {
        TerrainSampler::new(WORLD_SEED, TerrainParams::default())
    }

    /// A reference fill through the public per sample path.
    fn slow_fill(sampler: &TerrainSampler, key: ChunkKey, voxel_size_m: f64) -> Chunk {
        let mut chunk = Chunk::new(key, voxel_size_m);
        chunk.fill_from_with_material(|p| sampler.sample(p));
        chunk
    }

    fn fast_fill(sampler: &TerrainSampler, key: ChunkKey, voxel_size_m: f64) -> Chunk {
        let mut chunk = Chunk::new(key, voxel_size_m);
        sampler.fill_chunk(&mut chunk);
        chunk
    }

    fn assert_same_samples(fast: &Chunk, slow: &Chunk, label: &str) {
        let first_difference = fast
            .samples()
            .iter()
            .zip(slow.samples())
            .position(|(a, b)| a != b);
        if let Some(i) = first_difference {
            let x = i % SAMPLES_PER_AXIS;
            let y = (i / SAMPLES_PER_AXIS) % SAMPLES_PER_AXIS;
            let z = i / (SAMPLES_PER_AXIS * SAMPLES_PER_AXIS);
            panic!(
                "{label}: first difference at stored ({x}, {y}, {z}): fast {:?}, slow {:?}",
                fast.samples()[i],
                slow.samples()[i]
            );
        }
    }

    /// Horizontal positions worth testing, found by scanning the heightfield: the highest and
    /// lowest points, the steepest column, a beach below `sand_below_m` and a flat grass column.
    fn interesting_columns(sampler: &TerrainSampler) -> Vec<(&'static str, f64, f64)> {
        let p = sampler.params();
        let mut highest = (f64::MIN, 0.0, 0.0);
        let mut lowest = (f64::MAX, 0.0, 0.0);
        let mut steepest = (f64::MIN, 0.0, 0.0);
        let mut flattest_grass = (f64::MAX, 0.0, 0.0);
        let mut beach = None;
        let span_m = 4.0 * p.wavelength_m;
        let steps = 96;
        for iz in 0..steps {
            for ix in 0..steps {
                let x = -span_m / 2.0 + span_m * f64::from(ix) / f64::from(steps);
                let z = -span_m / 2.0 + span_m * f64::from(iz) / f64::from(steps);
                let h = sampler.height_m(x, z);
                let (slope, _) = sampler.slope_and_factor(x, z);
                if h > highest.0 {
                    highest = (h, x, z);
                }
                if h < lowest.0 {
                    lowest = (h, x, z);
                }
                if slope > steepest.0 {
                    steepest = (slope, x, z);
                }
                if h >= p.sand_below_m && slope < flattest_grass.0 {
                    flattest_grass = (slope, x, z);
                }
                if beach.is_none() && h < p.sand_below_m - 1.0 {
                    beach = Some((x, z));
                }
            }
        }
        let beach = beach.expect("a beach exists in the scanned region at the defaults");
        vec![
            ("highest", highest.1, highest.2),
            ("lowest", lowest.1, lowest.2),
            ("steepest", steepest.1, steepest.2),
            ("flat grass", flattest_grass.1, flattest_grass.2),
            ("beach", beach.0, beach.1),
        ]
    }

    fn key_containing(x: f64, y: f64, z: f64, voxel_size_m: f64) -> ChunkKey {
        let extent = CHUNK_SIZE as f64 * voxel_size_m;
        let f = |v: f64| (v / extent).floor() as i32;
        ChunkKey::new(f(x), f(y), f(z))
    }

    /// Chunk keys through the column at `(x, z)`: the chunk holding the surface, its neighbours
    /// above and below, two more at a distance where the column level early outs apply, and two
    /// far enough for the chunk level early outs.
    fn keys_through_column(
        sampler: &TerrainSampler,
        x: f64,
        z: f64,
        voxel_size_m: f64,
    ) -> Vec<ChunkKey> {
        let h = sampler.height_m(x, z);
        let surface = key_containing(x, h, z, voxel_size_m);
        [0, 1, -1, 3, -3, 48, -48]
            .into_iter()
            .map(|dy| surface.offset(0, dy, 0))
            .collect()
    }

    #[test]
    fn fill_chunk_matches_per_sample_path() {
        let sampler = sampler();
        let columns = interesting_columns(&sampler);
        for voxel_size_m in [0.25, 1.0, 4.0] {
            for (label, x, z) in &columns {
                for key in keys_through_column(&sampler, *x, *z, voxel_size_m) {
                    let fast = fast_fill(&sampler, key, voxel_size_m);
                    let slow = slow_fill(&sampler, key, voxel_size_m);
                    assert_same_samples(
                        &fast,
                        &slow,
                        &format!("{label} column, key {key:?}, voxel {voxel_size_m} m"),
                    );
                }
            }
        }
    }

    #[test]
    fn fill_chunk_matches_per_sample_path_at_range_ends() {
        let d = TerrainParams::default();
        let cases = [
            TerrainParams {
                amplitude_m: *TerrainParams::AMPLITUDE_RANGE_M.end(),
                wavelength_m: *TerrainParams::WAVELENGTH_RANGE_M.start(),
                octaves: *TerrainParams::OCTAVES_RANGE.end(),
                detail_amplitude_m: *TerrainParams::DETAIL_AMPLITUDE_RANGE_M.end(),
                detail_wavelength_m: *TerrainParams::DETAIL_WAVELENGTH_RANGE_M.start(),
                ..d
            },
            TerrainParams {
                amplitude_m: *TerrainParams::AMPLITUDE_RANGE_M.start(),
                wavelength_m: *TerrainParams::WAVELENGTH_RANGE_M.end(),
                octaves: *TerrainParams::OCTAVES_RANGE.start(),
                detail_amplitude_m: *TerrainParams::DETAIL_AMPLITUDE_RANGE_M.start(),
                detail_wavelength_m: *TerrainParams::DETAIL_WAVELENGTH_RANGE_M.end(),
                ..d
            },
            TerrainParams {
                base_height_m: *TerrainParams::HEIGHT_RANGE_M.end(),
                sand_below_m: *TerrainParams::HEIGHT_RANGE_M.end(),
                grass_depth_m: *TerrainParams::SOIL_DEPTH_RANGE_M.end(),
                soil_depth_m: *TerrainParams::SOIL_DEPTH_RANGE_M.start(),
                rock_slope: *TerrainParams::ROCK_SLOPE_RANGE.start(),
                ..d
            },
            TerrainParams {
                base_height_m: *TerrainParams::HEIGHT_RANGE_M.start(),
                sand_below_m: *TerrainParams::HEIGHT_RANGE_M.start(),
                grass_depth_m: *TerrainParams::SOIL_DEPTH_RANGE_M.start(),
                soil_depth_m: *TerrainParams::SOIL_DEPTH_RANGE_M.end(),
                rock_slope: *TerrainParams::ROCK_SLOPE_RANGE.end(),
                ..d
            },
        ];
        for (i, params) in cases.into_iter().enumerate() {
            params.validate().unwrap();
            let sampler = TerrainSampler::new(WORLD_SEED + i as u64, params);
            let voxel_size_m = if i % 2 == 0 { 1.0 } else { 0.5 };
            for key in keys_through_column(&sampler, 37.0, -91.0, voxel_size_m) {
                let fast = fast_fill(&sampler, key, voxel_size_m);
                let slow = slow_fill(&sampler, key, voxel_size_m);
                assert_same_samples(&fast, &slow, &format!("case {i}, key {key:?}"));
            }
        }
    }

    #[test]
    fn same_seed_is_bit_identical_across_runs_and_threads() {
        let sampler = sampler();
        let voxel_size_m = 1.0;
        let keys: Vec<ChunkKey> = interesting_columns(&sampler)
            .iter()
            .flat_map(|(_, x, z)| {
                let surface = key_containing(*x, sampler.height_m(*x, *z), *z, voxel_size_m);
                [surface, surface.offset(0, 1, 0)]
            })
            .collect();
        let reference: Vec<Chunk> = keys
            .iter()
            .map(|&key| fast_fill(&sampler, key, voxel_size_m))
            .collect();

        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        let sampler = TerrainSampler::new(WORLD_SEED, TerrainParams::default());
                        keys.iter()
                            .map(|&key| fast_fill(&sampler, key, voxel_size_m))
                            .collect::<Vec<Chunk>>()
                    })
                })
                .collect();
            for worker in workers {
                let chunks = worker.join().unwrap();
                for (a, b) in chunks.iter().zip(&reference) {
                    assert!(a.samples() == b.samples(), "thread result differs");
                }
            }
        });

        let other = TerrainSampler::new(WORLD_SEED + 1, TerrainParams::default());
        let differs = keys
            .iter()
            .zip(&reference)
            .any(|(&key, chunk)| fast_fill(&other, key, voxel_size_m).samples() != chunk.samples());
        assert!(differs, "a different seed must change the world");
    }

    #[test]
    fn all_four_materials_appear() {
        let sampler = sampler();
        let voxel_size_m = 1.0;
        let mut seen = [false; Material::COUNT];
        for (_, x, z) in interesting_columns(&sampler) {
            let surface = key_containing(x, sampler.height_m(x, z), z, voxel_size_m);
            for dy in [0, -1] {
                let chunk = fast_fill(&sampler, surface.offset(0, dy, 0), voxel_size_m);
                for v in chunk.samples() {
                    if v.is_inside() {
                        seen[usize::from(v.material.0)] = true;
                    }
                }
            }
        }
        for material in [
            Material::ROCK,
            Material::SOIL,
            Material::GRASS,
            Material::SAND,
        ] {
            assert!(
                seen[usize::from(material.0)],
                "{} never appears",
                material.name()
            );
        }
        assert!(!seen[usize::from(Material::AIR.0)], "air inside the solid");
    }

    #[test]
    fn height_is_within_amplitude() {
        let sampler = sampler();
        let p = sampler.params();
        let span_m = 3.0 * p.wavelength_m;
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for iz in 0..200 {
            for ix in 0..200 {
                let x = -span_m / 2.0 + span_m * f64::from(ix) / 200.0;
                let z = -span_m / 2.0 + span_m * f64::from(iz) / 200.0;
                let h = sampler.height_m(x, z);
                assert!(h.is_finite());
                assert!(
                    (h - p.base_height_m).abs() <= p.amplitude_m + 1e-9,
                    "{x}, {z}: {h}"
                );
                min = min.min(h);
                max = max.max(h);
            }
        }
        assert!(
            max - min > 0.25 * p.amplitude_m,
            "terrain is nearly flat: {min} to {max}"
        );

        let coarse = WorldConfig {
            voxel_size_m: 4.0,
            ..WorldConfig::default()
        };
        let fine = WorldConfig {
            voxel_size_m: 0.25,
            ..WorldConfig::default()
        };
        let a = TerrainSampler::new(coarse.seed, TerrainParams::default());
        let b = TerrainSampler::new(fine.seed, TerrainParams::default());
        for (x, z) in [(0.0, 0.0), (123.5, -77.25), (-4096.0, 999.0)] {
            assert_eq!(a.height_m(x, z).to_bits(), b.height_m(x, z).to_bits());
        }
    }

    #[test]
    fn slope_factor_lower_bound_holds_empirically() {
        for (seed, params) in [
            (WORLD_SEED, TerrainParams::default()),
            (
                WORLD_SEED + 9,
                TerrainParams {
                    amplitude_m: 400.0,
                    wavelength_m: 64.0,
                    octaves: 12,
                    ..TerrainParams::default()
                },
            ),
        ] {
            let sampler = TerrainSampler::new(seed, params);
            let sf_min = sampler.slope_factor_min();
            let mut worst = 1.0f64;
            let mut state = crate::seed::splitmix64(seed);
            for _ in 0..20_000 {
                state = crate::seed::splitmix64(state);
                let x = (state >> 11) as f64 / (1u64 << 53) as f64 * 8.0 * params.wavelength_m;
                let z = (state & 0xFFFF_FFFF) as f64 / 4_294_967_296.0 * 8.0 * params.wavelength_m;
                let (_, sf) = sampler.slope_and_factor(x - 4.0 * params.wavelength_m, z);
                worst = worst.min(sf);
            }
            println!("slope factor: empirical minimum {worst}, provable bound {sf_min}");
            assert!(
                worst >= sf_min,
                "slope factor {worst} below the bound {sf_min}"
            );
            // Sound with headroom, and at the defaults still useful for the early outs.
            assert!(
                worst > 1.5 * sf_min,
                "bound {sf_min} is unexpectedly tight against {worst}"
            );
            if params == TerrainParams::default() {
                assert!(
                    worst < 50.0 * sf_min,
                    "bound {sf_min} is too loose against {worst}"
                );
            }
        }
    }

    fn median_duration(mut samples: Vec<Duration>) -> Duration {
        samples.sort();
        samples[samples.len() / 2]
    }

    #[test]
    fn fill_chunk_timing() {
        let sampler = sampler();
        let runs = if cfg!(debug_assertions) { 5 } else { 50 };
        let (x, z) = (37.0, -91.0);

        // Cost of the two noise kernels on their own, to attribute the chunk times.
        let calls = if cfg!(debug_assertions) {
            2_000
        } else {
            200_000
        };
        let height_noise = sampler.height_noise();
        let detail_noise = sampler.detail_noise();
        let start = Instant::now();
        let mut sink = 0.0f32;
        for i in 0..calls {
            sink += height_noise.get_noise_2d(f64::from(i) * 0.37, f64::from(i) * 0.11);
        }
        let per_2d = start.elapsed().as_secs_f64() * 1e9 / f64::from(calls);
        let start = Instant::now();
        for i in 0..calls {
            let t = f64::from(i) * 0.013;
            sink += detail_noise.get_noise_3d(t, t * 0.7, t * 0.3);
        }
        let per_3d = start.elapsed().as_secs_f64() * 1e9 / f64::from(calls);
        println!(
            "noise: heightfield fBm ({} octaves) {per_2d:.1} ns per call, 3D detail {per_3d:.1} ns per call (checksum {sink})",
            sampler.params().octaves
        );
        for voxel_size_m in [1.0, 0.5] {
            let surface = key_containing(x, sampler.height_m(x, z), z, voxel_size_m);
            for (label, key) in [
                ("surface", surface),
                ("air", surface.offset(0, 3, 0)),
                ("rock", surface.offset(0, -3, 0)),
                ("far air", surface.offset(0, 64, 0)),
            ] {
                let mut chunk = Chunk::new(key, voxel_size_m);
                let mut times = Vec::with_capacity(runs);
                for _ in 0..runs {
                    let start = Instant::now();
                    sampler.fill_chunk(&mut chunk);
                    times.push(start.elapsed());
                }
                let median = median_duration(times);
                println!(
                    "fill_chunk {label} chunk {key:?} at {voxel_size_m} m voxels: median {:.1} us over {runs} runs (has_surface {})",
                    median.as_secs_f64() * 1e6,
                    chunk.has_surface()
                );
            }
        }
    }
}
