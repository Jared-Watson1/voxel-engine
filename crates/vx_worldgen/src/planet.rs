use std::sync::Arc;

use glam::{DVec3, IVec3, UVec3};
use vx_core::{PlanetParams, WorldConfig};
use vx_voxel::{
    Chunk, ChunkKey, DISTANCE_UNITS_PER_VOXEL, Material, SAMPLES_PER_AXIS, SampleBlock, Voxel,
};

use crate::cache::{DEFAULT_CAPACITY_TILES, HeightCacheStats, TILE, Tile, TileCache, TileKey};
use crate::cubemap::{
    CubeFace, cells_per_edge, face_of, gnomonic_on, grid_derivative, grid_from_gnomonic,
    map_inv_denominator, node_direction,
};
use crate::height::HeightNoise;

/// Lower limit of the slope factor `1 / sqrt(1 + slope^2)`, reached at slopes of sqrt(3) (60
/// degrees). Steeper terrain keeps the factor at this floor, which bounds the factor from below
/// everywhere, so the early outs and the shell need no derivative bound of the noise. The floor
/// moves the zero set only through the detail term, by at most a factor of two of the detail
/// amplitude, and only on slopes steeper than 60 degrees (none at the design defaults).
pub(crate) const SLOPE_FACTOR_MIN: f64 = 0.5;

/// A Catmull-Rom patch exceeds the range of its 16 nodes by at most this fraction of the range:
/// the 1D weights' negative part sums to `t (1 - t) / 2 <= 1/8`, and the tensor product's
/// negative part to `N_s (1 + N_t) + N_t (1 + N_s) <= 9/32`.
const CATMULL_ROM_OVERSHOOT: f64 = 9.0 / 32.0;

/// Bound of a Catmull-Rom patch's derivative along one grid axis per grid unit over the largest
/// difference of neighbouring nodes along that axis: the 1D derivative is a combination of the
/// three differences with coefficients whose magnitudes sum to at most 3/2 (at `t = 1/2`), and
/// the weights across the other axis have magnitudes summing to at most 5/4.
const CATMULL_ROM_SLOPE_PER_DIFFERENCE: f64 = 1.5 * 1.25;

/// Samples per brick edge: the unit of the local early outs inside a chunk or block.
const BRICK: u32 = 8;

/// Node indices added on each side of a footprint, covering rounding differences between the
/// corner projections and the per sample projection.
const FOOTPRINT_PAD: i32 = 2;

/// Largest node window gathered per fill; larger footprints (coarse chunks spanning a large part
/// of a face) read nodes through the cache per sample instead.
const MAX_WINDOW_NODES: i64 = 1 << 18;

/// The octree's deepest possible level (`WorldConfig::LOD_LEVELS_RANGE`), for [`PlanetSampler::shell_m`].
const MAX_LEVEL: u8 = (*WorldConfig::LOD_LEVELS_RANGE.end() - 1) as u8;

const N: u32 = SAMPLES_PER_AXIS as u32;

/// What a chunk can hold, decided from its bounds and the planet shell alone, before any noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkClass {
    /// Entirely above the shell: every sample is air.
    Air,
    /// Entirely below the shell: every sample is rock.
    Rock,
    /// Intersects the shell and may hold the surface; generate it.
    Surface,
}

/// Samples a planet's signed distance field in its body fixed frame (D3, D21): the surface height
/// `h(dir)` is seeded fBm of the unit direction times the radius, evaluated on a per level cube
/// sphere node grid and interpolated, so a chunk never evaluates the height noise per sample.
///
/// At level `L` the grid's node spacing at the mean radius is at most the level's voxel size
/// (`base_voxel_size_m * 2^L`), the nodes are the fBm without the octaves shorter than two voxel
/// sizes, the height between nodes is the Catmull-Rom (bicubic) interpolant of the 4 x 4 nodes
/// around it, and the detail noise is dropped once its wavelength is shorter than two voxel
/// sizes. The signed distance is `(|p| - R - h) * slope_factor + detail(p)`, where the slope is
/// the interpolant's gradient against the radial up. The cube sphere mapping is documented in the
/// `cubemap` module.
///
/// Every method is a pure function of its arguments, the seed, the parameters and the level 0
/// voxel size: the shared height cache changes speed, never results (engineering rule 2). No
/// transcendental function whose result differs between platforms is used in the sample path
/// (D13).
pub struct PlanetSampler {
    seed: u64,
    params: PlanetParams,
    base_voxel_size_m: f64,
    radius_m: f64,
    noise: HeightNoise,
    cache: TileCache,
    /// Bounds of the interpolated height at every level, overshoot and rounding slack included.
    h_lo_m: f64,
    h_hi_m: f64,
    /// Absolute slack for floating point rounding in the bounds, in metres.
    slack_m: f64,
    rock_depth_m: f64,
    rock_slope_sq: f64,
}

/// Per level constants.
#[derive(Debug, Clone, Copy)]
struct Level {
    level: u8,
    voxel_size_m: f64,
    inv_voxel: f64,
    unit_m: f64,
    /// A distance this far outside the surface quantises to the saturated value: the band plus one
    /// unit.
    skip_m: f64,
    /// `skip_m` plus one more unit, for tests on bounds rather than on exact values.
    coarse_skip_m: f64,
    cells: i32,
    half_cells: f64,
    octaves: usize,
    detail: bool,
    detail_amplitude_m: f64,
}

/// A sample's place on the cube sphere grid.
#[derive(Debug, Clone, Copy)]
struct Located {
    face: CubeFace,
    x: f64,
    y: f64,
    inv_dx: f64,
    inv_dy: f64,
    i: i32,
    j: i32,
    s: f64,
    t: f64,
}

/// The interpolated surface below or above a sample.
#[derive(Debug, Clone, Copy)]
struct Evaluated {
    /// `|p| - R`.
    altitude_m: f64,
    h_m: f64,
    slope_sq: f64,
    /// `(altitude - h) * slope_factor`, the distance before the detail noise.
    base_m: f64,
}

/// Radii outside which samples of a box are saturated air (`air_m`) or saturated deep rock
/// (`rock_m`).
#[derive(Debug, Clone, Copy)]
struct Thresholds {
    air_m: f64,
    rock_m: f64,
}

/// Dense node heights of one face over an index range, gathered once per fill.
struct Window {
    face: CubeFace,
    i0: i32,
    j0: i32,
    width: i32,
    height: i32,
    nodes: Vec<f64>,
}

impl Window {
    fn holds(&self, i_lo: i32, i_hi: i32, j_lo: i32, j_hi: i32) -> bool {
        i_lo >= self.i0
            && j_lo >= self.j0
            && i_hi < self.i0 + self.width
            && j_hi < self.j0 + self.height
    }

    #[inline(always)]
    fn stencil(&self, i: i32, j: i32) -> [f64; 16] {
        let w = self.width as usize;
        let base = (j - 1 - self.j0) as usize * w + (i - 1 - self.i0) as usize;
        let mut out = [0.0; 16];
        for jj in 0..4 {
            out[jj * 4..jj * 4 + 4].copy_from_slice(&self.nodes[base + jj * w..base + jj * w + 4]);
        }
        out
    }

    /// Smallest and largest node and the largest neighbour differences along i and along j over
    /// the inclusive index range, which [`Window::holds`].
    fn range_stats(&self, i_lo: i32, i_hi: i32, j_lo: i32, j_hi: i32) -> (f64, f64, f64, f64) {
        let w = self.width as usize;
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        let mut du = 0.0f64;
        let mut dv = 0.0f64;
        for j in j_lo..=j_hi {
            let row = (j - self.j0) as usize * w;
            let next = row + w;
            for i in i_lo..=i_hi {
                let k = row + (i - self.i0) as usize;
                let v = self.nodes[k];
                lo = lo.min(v);
                hi = hi.max(v);
                if i < i_hi {
                    du = du.max((self.nodes[k + 1] - v).abs());
                }
                if j < j_hi {
                    dv = dv.max((self.nodes[next + (i - self.i0) as usize] - v).abs());
                }
            }
        }
        (lo, hi, du, dv)
    }
}

/// Catmull-Rom weights and their derivatives at `t` in 0 to 1.
#[inline(always)]
fn catmull_rom_weights(t: f64) -> ([f64; 4], [f64; 4]) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        [
            0.5 * (-t3 + 2.0 * t2 - t),
            0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
            0.5 * (-3.0 * t3 + 4.0 * t2 + t),
            0.5 * (t3 - t2),
        ],
        [
            0.5 * (-3.0 * t2 + 4.0 * t - 1.0),
            0.5 * (9.0 * t2 - 10.0 * t),
            0.5 * (-9.0 * t2 + 8.0 * t + 1.0),
            0.5 * (3.0 * t2 - 2.0 * t),
        ],
    )
}

/// Value and derivatives per grid unit along i and j of the Catmull-Rom patch over `nodes` (row
/// `jj` holds nodes `i - 1 + ii`, `j - 1 + jj`) at fractions `s` along i and `t` along j.
#[inline(always)]
fn catmull_rom(nodes: &[f64; 16], s: f64, t: f64) -> (f64, f64, f64) {
    let (ws, dws) = catmull_rom_weights(s);
    let (wt, dwt) = catmull_rom_weights(t);
    let mut h = 0.0;
    let mut hs = 0.0;
    let mut ht = 0.0;
    for jj in 0..4 {
        let r = &nodes[jj * 4..jj * 4 + 4];
        let value = ws[0] * r[0] + ws[1] * r[1] + ws[2] * r[2] + ws[3] * r[3];
        let along = dws[0] * r[0] + dws[1] * r[1] + dws[2] * r[2] + dws[3] * r[3];
        h += wt[jj] * value;
        hs += wt[jj] * along;
        ht += dwt[jj] * value;
    }
    (h, hs, ht)
}

#[inline(always)]
fn length_sq(p: DVec3) -> f64 {
    p.x * p.x + p.y * p.y + p.z * p.z
}

/// Smallest and largest distance from the origin over the axis aligned box `lo` to `hi`.
fn radial_range(lo: DVec3, hi: DVec3) -> (f64, f64) {
    let nearest = DVec3::ZERO.clamp(lo, hi);
    let farthest = DVec3::new(
        lo.x.abs().max(hi.x.abs()),
        lo.y.abs().max(hi.y.abs()),
        lo.z.abs().max(hi.z.abs()),
    );
    (length_sq(nearest).sqrt(), length_sq(farthest).sqrt())
}

fn box_corners(lo: DVec3, hi: DVec3) -> [DVec3; 8] {
    std::array::from_fn(|c| {
        DVec3::new(
            if c & 1 == 0 { lo.x } else { hi.x },
            if c & 2 == 0 { lo.y } else { hi.y },
            if c & 4 == 0 { lo.z } else { hi.z },
        )
    })
}

/// Footprint of a box on one face: node index ranges (inclusive) holding the 4 x 4 stencil of
/// every sample of the box that lies on the face, with the gnomonic coordinate ranges.
#[derive(Debug, Clone, Copy)]
struct Footprint {
    i_lo: i32,
    i_hi: i32,
    j_lo: i32,
    j_hi: i32,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
}

impl PlanetSampler {
    pub fn new(seed: u64, params: &PlanetParams, base_voxel_size_m: f64) -> Self {
        Self::with_cache_capacity(seed, params, base_voxel_size_m, DEFAULT_CAPACITY_TILES)
    }

    /// A sampler whose height cache holds at most `capacity_tiles` tiles of 8 x 8 nodes
    /// (512 bytes each); the default is 16,384 tiles (8 MiB). The capacity changes speed only.
    pub fn with_cache_capacity(
        seed: u64,
        params: &PlanetParams,
        base_voxel_size_m: f64,
        capacity_tiles: usize,
    ) -> Self {
        assert!(base_voxel_size_m > 0.0, "voxel size must be positive");
        let terrain = &params.terrain;
        let radius_m = params.radius_m();
        let noise = HeightNoise::new(seed, terrain, radius_m);
        let amplitude = noise.amplitude_m();
        let base = noise.base_height_m();
        let slack_m = 1e-9 * (radius_m + base.abs() + amplitude) + 1e-9;
        let overshoot = 2.0 * amplitude * CATMULL_ROM_OVERSHOOT;
        Self {
            seed,
            params: *params,
            base_voxel_size_m,
            radius_m,
            noise,
            cache: TileCache::new(capacity_tiles),
            h_lo_m: base - amplitude - overshoot - slack_m,
            h_hi_m: base + amplitude + overshoot + slack_m,
            slack_m,
            rock_depth_m: terrain.grass_depth_m.max(terrain.soil_depth_m),
            rock_slope_sq: terrain.rock_slope * terrain.rock_slope,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn params(&self) -> &PlanetParams {
        &self.params
    }

    pub fn base_voxel_size_m(&self) -> f64 {
        self.base_voxel_size_m
    }

    pub fn radius_m(&self) -> f64 {
        self.radius_m
    }

    /// Inner and outer radius between which the surface lies at every level: outside the outer
    /// radius every sample of every level is outside the solid (positive distance), inside the
    /// inner radius every sample is inside. Margins: the amplitude, the Catmull-Rom overshoot
    /// (9/32 of the node range on each side), the detail noise divided by the slope factor floor,
    /// quantisation (a distance within half a unit of zero is stored as outside, which can move
    /// the inner side by half a unit of the level over the slope factor floor) and floating point
    /// slack. Levels up to 13, the deepest `WorldConfig::lod_levels` allows, are covered.
    ///
    /// The saturated band (about four voxels on each side of the surface) is level dependent and
    /// enters [`PlanetSampler::classify`], which uses the key's level.
    pub fn shell_m(&self) -> (f64, f64) {
        (0..=MAX_LEVEL).fold((f64::MAX, f64::MIN), |(lo, hi), level| {
            let (a, b) = self.shell_at_m(level);
            (lo.min(a), hi.max(b))
        })
    }

    /// [`PlanetSampler::shell_m`] for one level.
    pub fn shell_at_m(&self, level: u8) -> (f64, f64) {
        let lv = self.level(level);
        let detail = lv.detail_amplitude_m / SLOPE_FACTOR_MIN;
        let quantisation = 0.5 * lv.unit_m / SLOPE_FACTOR_MIN;
        let inner = self.radius_m + self.h_lo_m - detail - quantisation - self.slack_m;
        let outer = self.radius_m + self.h_hi_m + detail + self.slack_m;
        (inner.max(0.0), outer)
    }

    /// Classifies a chunk from its bounds and [`PlanetSampler::shell_m`] widened by the saturated
    /// band of the key's level. `Air` and `Rock` are exact: [`PlanetSampler::fill_chunk`] would
    /// fill the chunk with `Voxel::AIR` or `Voxel::solid(Material::ROCK)` only. A `Surface` chunk
    /// may still turn out uniform.
    pub fn classify(&self, key: ChunkKey) -> ChunkClass {
        let lv = self.level(key.level);
        let min = key.world_sample(IVec3::NEG_ONE);
        let max = key.world_sample(IVec3::splat(N as i32 - 2));
        self.classify_box(&lv, min, max)
    }

    /// Surface height above the mean radius in metres along `dir` (any nonzero length), at full
    /// detail and without the grid: for the camera's altitude and spawning, not for chunks. The
    /// zero vector gives the base height.
    pub fn height_m(&self, dir: DVec3) -> f64 {
        let len_sq = length_sq(dir);
        if len_sq == 0.0 {
            return self.noise.base_height_m();
        }
        let unit = dir * (1.0 / len_sq.sqrt());
        self.noise.height(unit, self.noise.octave_count())
    }

    /// Signed distance in metres (negative inside) and the solid's material at `p` (body fixed
    /// metres) as level `level` defines the field: exactly the value `fill_chunk` and `fill_block`
    /// store at a sample position of that level, before quantisation.
    pub fn sample(&self, p: DVec3, level: u8) -> (f64, Material) {
        let lv = self.level(level);
        let e = self.evaluate(&lv, p, &mut |face, i, j| {
            self.stencil_from_cache(&lv, face, i, j)
        });
        let distance_m = if lv.detail {
            e.base_m + self.noise.detail(p)
        } else {
            e.base_m
        };
        (distance_m, self.material(&e))
    }

    /// Fills every sample of `chunk`, apron included, at the chunk's level. Bit identical to
    /// [`PlanetSampler::sample`] at every sample position, whatever the cache state or thread.
    pub fn fill_chunk(&self, chunk: &mut Chunk) {
        profiling::scope!("PlanetSampler::fill_chunk");
        let key = chunk.key();
        let lv = self.level(key.level);
        debug_assert_eq!(
            chunk.voxel_size_m(),
            lv.voxel_size_m,
            "chunk built for another base voxel size"
        );
        let min = key.world_sample(IVec3::NEG_ONE);
        self.fill_box(&lv, min, UVec3::splat(N), chunk.samples_mut());
    }

    /// Fills every sample of `block` (a transition slab, D22) at the block's level, bit identical
    /// to the fine chunk's samples at the same world sample indices.
    pub fn fill_block(&self, block: &mut SampleBlock) {
        profiling::scope!("PlanetSampler::fill_block");
        let lv = self.level(block.level());
        debug_assert_eq!(
            block.voxel_size_m(),
            lv.voxel_size_m,
            "block built for another base voxel size"
        );
        let (min, dims) = (block.min(), block.dims());
        self.fill_box(&lv, min, dims, block.voxels_mut());
    }

    /// Drops every cached height tile (for cold cache measurements). Results do not change.
    pub fn clear_height_cache(&self) {
        self.cache.clear();
    }

    pub fn height_cache_stats(&self) -> HeightCacheStats {
        self.cache.stats()
    }

    fn level(&self, level: u8) -> Level {
        let voxel_size_m = ChunkKey::at_level(level, 0, 0, 0).voxel_size_m(self.base_voxel_size_m);
        let unit_m = voxel_size_m / f64::from(DISTANCE_UNITS_PER_VOXEL);
        let skip_m = f64::from(i8::MAX) * unit_m + unit_m;
        let cells = cells_per_edge(self.radius_m, voxel_size_m);
        let min_wavelength_m = 2.0 * voxel_size_m;
        let detail = self.noise.detail_wavelength_m() >= min_wavelength_m;
        Level {
            level,
            voxel_size_m,
            inv_voxel: 1.0 / voxel_size_m,
            unit_m,
            skip_m,
            coarse_skip_m: skip_m + unit_m,
            cells,
            half_cells: 0.5 * f64::from(cells),
            octaves: self.noise.octaves_at_least(min_wavelength_m),
            detail,
            detail_amplitude_m: if detail {
                self.noise.detail_amplitude_m()
            } else {
                0.0
            },
        }
    }

    fn thresholds(&self, lv: &Level, h_lo: f64, h_hi: f64, slope_factor_min: f64) -> Thresholds {
        let margin = (lv.coarse_skip_m + lv.detail_amplitude_m) / slope_factor_min;
        Thresholds {
            air_m: self.radius_m + h_hi + margin + self.slack_m,
            rock_m: self.radius_m + h_lo - margin.max(self.rock_depth_m + lv.unit_m) - self.slack_m,
        }
    }

    fn classify_box(&self, lv: &Level, min: IVec3, max: IVec3) -> ChunkClass {
        let lo = min.as_dvec3() * lv.voxel_size_m;
        let hi = max.as_dvec3() * lv.voxel_size_m;
        let (r_min, r_max) = radial_range(lo, hi);
        let t = self.thresholds(lv, self.h_lo_m, self.h_hi_m, SLOPE_FACTOR_MIN);
        if r_min >= t.air_m {
            ChunkClass::Air
        } else if r_max <= t.rock_m {
            ChunkClass::Rock
        } else {
            ChunkClass::Surface
        }
    }

    #[inline(always)]
    fn locate(&self, lv: &Level, p: DVec3) -> Located {
        let (face, major, pu, pv) = face_of(p);
        let inv_major = if major > 0.0 { 1.0 / major } else { 0.0 };
        Self::locate_on(lv, face, pu * inv_major, pv * inv_major)
    }

    /// Grid place of gnomonic coordinates `(x, y)` on `face`.
    #[inline(always)]
    fn locate_on(lv: &Level, face: CubeFace, x: f64, y: f64) -> Located {
        let inv_dx = map_inv_denominator(x);
        let inv_dy = map_inv_denominator(y);
        let u = (grid_from_gnomonic(x, inv_dx) + 1.0) * lv.half_cells;
        let v = (grid_from_gnomonic(y, inv_dy) + 1.0) * lv.half_cells;
        let i = (u.floor() as i32).clamp(0, lv.cells - 1);
        let j = (v.floor() as i32).clamp(0, lv.cells - 1);
        Located {
            face,
            x,
            y,
            inv_dx,
            inv_dy,
            i,
            j,
            s: u - f64::from(i),
            t: v - f64::from(j),
        }
    }

    /// The per sample kernel shared by every path: locate, interpolate, slope, base distance.
    #[inline(always)]
    fn evaluate(
        &self,
        lv: &Level,
        p: DVec3,
        stencil: &mut impl FnMut(CubeFace, i32, i32) -> [f64; 16],
    ) -> Evaluated {
        let at = self.locate(lv, p);
        self.evaluate_at(lv, p, &at, stencil)
    }

    #[inline(always)]
    fn evaluate_at(
        &self,
        lv: &Level,
        p: DVec3,
        at: &Located,
        stencil: &mut impl FnMut(CubeFace, i32, i32) -> [f64; 16],
    ) -> Evaluated {
        let r_sq = length_sq(p);
        let nodes = stencil(at.face, at.i, at.j);
        let (h_m, hs, ht) = catmull_rom(&nodes, at.s, at.t);
        let hx = hs * lv.half_cells * grid_derivative(at.inv_dx);
        let hy = ht * lv.half_cells * grid_derivative(at.inv_dy);
        let (x, y) = (at.x, at.y);
        // Squared gradient on the unit sphere through the inverse metric of the gnomonic chart,
        // `(1 + x^2 + y^2) [[1 + x^2, x y], [x y, 1 + y^2]]`, then per metre at the sample's radius.
        let gradient_sq = (1.0 + x * x + y * y)
            * ((1.0 + x * x) * hx * hx + 2.0 * x * y * hx * hy + (1.0 + y * y) * hy * hy);
        let slope_sq = gradient_sq / r_sq.max(1.0);
        let slope_factor = (1.0 / (1.0 + slope_sq).sqrt()).max(SLOPE_FACTOR_MIN);
        let altitude_m = r_sq.sqrt() - self.radius_m;
        Evaluated {
            altitude_m,
            h_m,
            slope_sq,
            base_m: (altitude_m - h_m) * slope_factor,
        }
    }

    /// Material of the solid at an evaluated sample: by depth below the interpolated surface,
    /// then by the surface's height and slope.
    #[inline(always)]
    fn material(&self, e: &Evaluated) -> Material {
        let t = &self.params.terrain;
        let depth_m = e.h_m - e.altitude_m;
        if depth_m < t.grass_depth_m {
            if e.h_m < t.sand_below_m {
                Material::SAND
            } else if e.slope_sq > self.rock_slope_sq {
                Material::ROCK
            } else if e.h_m > t.snow_above_m {
                Material::SNOW
            } else {
                Material::GRASS
            }
        } else if depth_m < t.soil_depth_m {
            Material::SOIL
        } else {
            Material::ROCK
        }
    }

    fn tile(&self, lv: &Level, key: TileKey) -> Arc<Tile> {
        self.cache.get_or_compute(key, || {
            let mut tile = [0.0; TILE as usize * TILE as usize];
            for jj in 0..TILE {
                let j = key.tj * TILE + jj;
                for ii in 0..TILE {
                    let i = key.ti * TILE + ii;
                    // Nodes beyond the ghost ring are never read; leaving them zero keeps the
                    // inverse map away from its pole at |a| = 5.
                    if (-1..=lv.cells + 1).contains(&i) && (-1..=lv.cells + 1).contains(&j) {
                        let dir = node_direction(key.face, lv.cells, i, j);
                        tile[(jj * TILE + ii) as usize] = self.noise.height(dir, lv.octaves);
                    }
                }
            }
            tile
        })
    }

    fn tile_key(lv: &Level, face: CubeFace, i: i32, j: i32) -> TileKey {
        TileKey {
            level: lv.level,
            face,
            ti: i.div_euclid(TILE),
            tj: j.div_euclid(TILE),
        }
    }

    fn stencil_from_cache(&self, lv: &Level, face: CubeFace, i: i32, j: i32) -> [f64; 16] {
        let mut out = [0.0; 16];
        let mut last: Option<(TileKey, Arc<Tile>)> = None;
        for jj in 0..4 {
            for ii in 0..4 {
                let (ni, nj) = (i - 1 + ii, j - 1 + jj);
                let key = Self::tile_key(lv, face, ni, nj);
                let tile = match &last {
                    Some((k, t)) if *k == key => Arc::clone(t),
                    _ => {
                        let t = self.tile(lv, key);
                        last = Some((key, Arc::clone(&t)));
                        t
                    }
                };
                out[(jj * 4 + ii) as usize] =
                    tile[(nj.rem_euclid(TILE) * TILE + ni.rem_euclid(TILE)) as usize];
            }
        }
        out
    }

    fn footprint(&self, lv: &Level, face: CubeFace, corners: &[DVec3; 8]) -> Option<Footprint> {
        let mut x_lo = f64::MAX;
        let mut x_hi = f64::MIN;
        let mut y_lo = f64::MAX;
        let mut y_hi = f64::MIN;
        for &c in corners {
            // The gnomonic projection maps segments to segments, so the box's image is the
            // convex hull of its corners' images when every corner is in front of the plane.
            let (x, y) = gnomonic_on(face, c)?;
            x_lo = x_lo.min(x);
            x_hi = x_hi.max(x);
            y_lo = y_lo.min(y);
            y_hi = y_hi.max(y);
        }
        if x_lo > 1.0 || x_hi < -1.0 || y_lo > 1.0 || y_hi < -1.0 {
            return None;
        }
        let (x_lo, x_hi) = (x_lo.max(-1.0), x_hi.min(1.0));
        let (y_lo, y_hi) = (y_lo.max(-1.0), y_hi.min(1.0));
        let index = |g: f64| {
            let u = (grid_from_gnomonic(g, map_inv_denominator(g)) + 1.0) * lv.half_cells;
            (u.floor() as i32).clamp(0, lv.cells - 1)
        };
        let ghost = |i: i32| i.clamp(-1, lv.cells + 1);
        Some(Footprint {
            i_lo: ghost(index(x_lo) - 1 - FOOTPRINT_PAD),
            i_hi: ghost(index(x_hi) + 2 + FOOTPRINT_PAD),
            j_lo: ghost(index(y_lo) - 1 - FOOTPRINT_PAD),
            j_hi: ghost(index(y_hi) + 2 + FOOTPRINT_PAD),
            x_lo,
            x_hi,
            y_lo,
            y_hi,
        })
    }

    fn gather_window(&self, lv: &Level, face: CubeFace, f: &Footprint) -> Option<Window> {
        let width = f.i_hi - f.i_lo + 1;
        let height = f.j_hi - f.j_lo + 1;
        if i64::from(width) * i64::from(height) > MAX_WINDOW_NODES {
            return None;
        }
        let mut nodes = vec![0.0; (width * height) as usize];
        for tj in f.j_lo.div_euclid(TILE)..=f.j_hi.div_euclid(TILE) {
            for ti in f.i_lo.div_euclid(TILE)..=f.i_hi.div_euclid(TILE) {
                let tile = self.tile(
                    lv,
                    TileKey {
                        level: lv.level,
                        face,
                        ti,
                        tj,
                    },
                );
                let i_from = (ti * TILE).max(f.i_lo);
                let i_to = (ti * TILE + TILE - 1).min(f.i_hi);
                for j in (tj * TILE).max(f.j_lo)..=(tj * TILE + TILE - 1).min(f.j_hi) {
                    let src = ((j - tj * TILE) * TILE + (i_from - ti * TILE)) as usize;
                    let dst = ((j - f.j_lo) * width + (i_from - f.i_lo)) as usize;
                    let len = (i_to - i_from + 1) as usize;
                    nodes[dst..dst + len].copy_from_slice(&tile[src..src + len]);
                }
            }
        }
        Some(Window {
            face,
            i0: f.i_lo,
            j0: f.j_lo,
            width,
            height,
            nodes,
        })
    }

    /// Thresholds for a brick from its own footprint when it lies on one face inside a window:
    /// the interpolant's range from the nodes it can reach and a slope factor bound from their
    /// differences. Otherwise the global bounds.
    fn brick_thresholds(&self, lv: &Level, windows: &[Window], lo: DVec3, hi: DVec3) -> Thresholds {
        let global = self.thresholds(lv, self.h_lo_m, self.h_hi_m, SLOPE_FACTOR_MIN);
        let corners = box_corners(lo, hi);
        let face = face_of(corners[0]).0;
        if corners.iter().any(|&c| face_of(c).0 != face) {
            return global;
        }
        let Some(window) = windows.iter().find(|w| w.face == face) else {
            return global;
        };
        let Some(f) = self.footprint(lv, face, &corners) else {
            return global;
        };
        if !window.holds(f.i_lo, f.i_hi, f.j_lo, f.j_hi) {
            return global;
        }
        let (lo_n, hi_n, du, dv) = window.range_stats(f.i_lo, f.i_hi, f.j_lo, f.j_hi);
        let overshoot = CATMULL_ROM_OVERSHOOT * (hi_n - lo_n);
        let h_lo = lo_n - overshoot - self.slack_m;
        let h_hi = hi_n + overshoot + self.slack_m;

        let abs_max = |a: f64, b: f64| a.abs().max(b.abs());
        let abs_min = |a: f64, b: f64| {
            if a <= 0.0 && b >= 0.0 {
                0.0
            } else {
                a.abs().min(b.abs())
            }
        };
        let (x_max, y_max) = (abs_max(f.x_lo, f.x_hi), abs_max(f.y_lo, f.y_hi));
        let hx = CATMULL_ROM_SLOPE_PER_DIFFERENCE
            * du
            * lv.half_cells
            * grid_derivative(map_inv_denominator(abs_min(f.x_lo, f.x_hi)));
        let hy = CATMULL_ROM_SLOPE_PER_DIFFERENCE
            * dv
            * lv.half_cells
            * grid_derivative(map_inv_denominator(abs_min(f.y_lo, f.y_hi)));
        let gradient_sq = (1.0 + x_max * x_max + y_max * y_max)
            * ((1.0 + x_max * x_max) * hx * hx
                + 2.0 * x_max * y_max * hx * hy
                + (1.0 + y_max * y_max) * hy * hy);
        let (r_min, _) = radial_range(lo, hi);
        let slope_sq = gradient_sq * (1.0 + 1e-9) / (r_min * r_min).max(1.0);
        let slope_factor_min = (1.0 / (1.0 + slope_sq).sqrt()).max(SLOPE_FACTOR_MIN);
        self.thresholds(lv, h_lo, h_hi, slope_factor_min)
    }

    /// Fills `out` (x fastest, then y, then z) with the samples of the box of `dims` world samples
    /// of `lv` starting at `min`.
    fn fill_box(&self, lv: &Level, min: IVec3, dims: UVec3, out: &mut [Voxel]) {
        debug_assert_eq!(out.len(), (dims.x * dims.y * dims.z) as usize);
        let max = min + dims.as_ivec3() - IVec3::ONE;
        match self.classify_box(lv, min, max) {
            ChunkClass::Air => {
                out.fill(Voxel::AIR);
                return;
            }
            ChunkClass::Rock => {
                out.fill(Voxel::solid(Material::ROCK));
                return;
            }
            ChunkClass::Surface => {}
        }

        let axis = |first: i32, len: u32| -> Vec<f64> {
            (0..len as i32)
                .map(|k| f64::from(first + k) * lv.voxel_size_m)
                .collect()
        };
        let xs = axis(min.x, dims.x);
        let ys = axis(min.y, dims.y);
        let zs = axis(min.z, dims.z);

        let lo = DVec3::new(xs[0], ys[0], zs[0]);
        let hi = DVec3::new(
            xs[dims.x as usize - 1],
            ys[dims.y as usize - 1],
            zs[dims.z as usize - 1],
        );
        let corners = box_corners(lo, hi);
        let mut windows: Vec<Window> = Vec::new();
        for c in corners {
            let face = face_of(c).0;
            if windows.iter().any(|w| w.face == face) {
                continue;
            }
            if let Some(w) = self
                .footprint(lv, face, &corners)
                .and_then(|f| self.gather_window(lv, face, &f))
            {
                windows.push(w);
            }
        }
        let mut stencil = |face: CubeFace, i: i32, j: i32| -> [f64; 16] {
            if let Some(w) = windows.iter().find(|w| w.face == face)
                && w.holds(i - 1, i + 2, j - 1, j + 2)
            {
                return w.stencil(i, j);
            }
            self.stencil_from_cache(lv, face, i, j)
        };

        let (nx, ny) = (dims.x as usize, dims.y as usize);
        for bz in (0..dims.z).step_by(BRICK as usize) {
            for by in (0..dims.y).step_by(BRICK as usize) {
                for bx in (0..dims.x).step_by(BRICK as usize) {
                    let end = (UVec3::new(bx, by, bz) + UVec3::splat(BRICK)).min(dims);
                    let (x0, y0, z0) = (bx as usize, by as usize, bz as usize);
                    let (x1, y1, z1) = (end.x as usize, end.y as usize, end.z as usize);
                    let b_lo = DVec3::new(xs[x0], ys[y0], zs[z0]);
                    let b_hi = DVec3::new(xs[x1 - 1], ys[y1 - 1], zs[z1 - 1]);
                    let t = self.brick_thresholds(lv, &windows, b_lo, b_hi);
                    let (r_min, r_max) = radial_range(b_lo, b_hi);
                    let uniform = if r_min >= t.air_m {
                        Some(Voxel::AIR)
                    } else if r_max <= t.rock_m {
                        Some(Voxel::solid(Material::ROCK))
                    } else {
                        None
                    };
                    let air_sq = t.air_m * t.air_m;
                    let rock_sq = if t.rock_m > 0.0 {
                        t.rock_m * t.rock_m
                    } else {
                        -1.0
                    };
                    for (z, &pz) in zs.iter().enumerate().take(z1).skip(z0) {
                        for (y, &py) in ys.iter().enumerate().take(y1).skip(y0) {
                            let row = (z * ny + y) * nx;
                            let dst = &mut out[row + x0..row + x1];
                            if let Some(v) = uniform {
                                dst.fill(v);
                                continue;
                            }
                            for (voxel, &px) in dst.iter_mut().zip(&xs[x0..x1]) {
                                let p = DVec3::new(px, py, pz);
                                let r_sq = length_sq(p);
                                *voxel = if r_sq >= air_sq {
                                    Voxel::AIR
                                } else if r_sq <= rock_sq {
                                    Voxel::solid(Material::ROCK)
                                } else {
                                    self.sample_voxel(lv, p, &mut stencil)
                                };
                            }
                        }
                    }
                }
            }
        }
    }

    /// The quantised sample at `p`, equal to quantising [`PlanetSampler::sample`]; skips the
    /// detail noise where its amplitude cannot reach the band.
    #[inline(always)]
    fn sample_voxel(
        &self,
        lv: &Level,
        p: DVec3,
        stencil: &mut impl FnMut(CubeFace, i32, i32) -> [f64; 16],
    ) -> Voxel {
        let e = self.evaluate(lv, p, stencil);
        let distance_m = if lv.detail {
            if e.base_m - lv.detail_amplitude_m > lv.skip_m {
                return Voxel::AIR;
            }
            if e.base_m + lv.detail_amplitude_m < -lv.skip_m {
                return Voxel::solid(self.material(&e));
            }
            e.base_m + self.noise.detail(p)
        } else {
            e.base_m
        };
        let distance = Voxel::quantise_distance(distance_m * lv.inv_voxel);
        let material = if distance < 0 {
            self.material(&e)
        } else {
            Material::AIR
        };
        Voxel { distance, material }
    }

    /// Interpolated height and distance before the detail noise at `p` as `face`'s grid defines
    /// them, which may be a neighbouring face for points near a cube edge (ghost nodes extend
    /// every grid).
    #[cfg(test)]
    pub(crate) fn on_face(&self, level: u8, face: CubeFace, p: DVec3) -> Option<(f64, f64)> {
        let lv = self.level(level);
        let (x, y) = gnomonic_on(face, p)?;
        let at = Self::locate_on(&lv, face, x, y);
        let e = self.evaluate_at(&lv, p, &at, &mut |face, i, j| {
            self.stencil_from_cache(&lv, face, i, j)
        });
        Some((e.h_m, e.base_m))
    }

    /// The level's interpolated height along `dir` as the sample path sees it.
    #[cfg(test)]
    pub(crate) fn grid_height_m(&self, level: u8, dir: DVec3) -> f64 {
        let lv = self.level(level);
        let e = self.evaluate(&lv, dir * self.radius_m, &mut |face, i, j| {
            self.stencil_from_cache(&lv, face, i, j)
        });
        e.h_m
    }

    #[cfg(test)]
    pub(crate) fn level_octaves(&self, level: u8) -> (usize, bool, i32) {
        let lv = self.level(level);
        (lv.octaves, lv.detail, lv.cells)
    }

    /// Height and slope of the level's grid surface along `dir`, for material statistics.
    #[cfg(test)]
    pub(crate) fn grid_height_and_slope(&self, level: u8, dir: DVec3) -> (f64, f64) {
        let lv = self.level(level);
        let e = self.evaluate(&lv, dir * self.radius_m, &mut |face, i, j| {
            self.stencil_from_cache(&lv, face, i, j)
        });
        (e.h_m, e.slope_sq.sqrt())
    }
}

#[cfg(test)]
#[path = "planet_tests.rs"]
mod tests;
