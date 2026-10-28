use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use vx_core::{CHUNK_SIZE, TerrainParams, WorldConfig, WorldPos};
use vx_voxel::{Face, transition_slab};

use super::*;
use crate::seed::splitmix64;

const SEED: u64 = 1;
const BASE_VOXEL_M: f64 = 1.0;

fn sampler() -> PlanetSampler {
    PlanetSampler::new(SEED, &PlanetParams::default(), BASE_VOXEL_M)
}

fn config() -> WorldConfig {
    WorldConfig {
        voxel_size_m: BASE_VOXEL_M,
        ..WorldConfig::default()
    }
}

/// A uniformly spread unit direction from a hash.
fn direction(hash: u64) -> DVec3 {
    let a = splitmix64(hash);
    let b = splitmix64(a);
    let c = splitmix64(b);
    let unit = |h: u64| (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
    let v = DVec3::new(unit(a), unit(b), unit(c));
    if v.length_squared() < 1e-6 {
        DVec3::Y
    } else {
        v.normalize()
    }
}

/// Directions worth testing: two random ones, a point on a cube edge, a cube corner, both poles,
/// the prime meridian and a point near (but not on) an edge.
fn special_directions() -> Vec<(&'static str, DVec3)> {
    vec![
        ("random a", direction(11)),
        ("random b", direction(12)),
        ("edge +X+Y", DVec3::new(1.0, 1.0, 0.31).normalize()),
        ("corner", DVec3::new(-1.0, 1.0, 1.0).normalize()),
        ("north pole", DVec3::Y),
        ("south pole", DVec3::NEG_Y),
        ("prime meridian", DVec3::X),
        ("near edge -Z+X", DVec3::new(0.999, -0.2, -1.0).normalize()),
    ]
}

fn surface_key(s: &PlanetSampler, dir: DVec3, level: u8) -> ChunkKey {
    let p = dir * (s.radius_m() + s.grid_height_m(level, dir));
    ChunkKey::containing_at(WorldPos::body_fixed(p), level, &config())
}

fn fast_fill(s: &PlanetSampler, key: ChunkKey) -> Chunk {
    let mut chunk = Chunk::new(key, BASE_VOXEL_M);
    s.fill_chunk(&mut chunk);
    chunk
}

fn slow_fill(s: &PlanetSampler, key: ChunkKey) -> Chunk {
    let mut chunk = Chunk::new(key, BASE_VOXEL_M);
    chunk.fill_from_with_material(|p| s.sample(p, key.level));
    chunk
}

fn assert_same(fast: &[Voxel], slow: &[Voxel], label: &str) {
    if let Some(i) = fast.iter().zip(slow).position(|(a, b)| a != b) {
        panic!(
            "{label}: first difference at index {i}: fast {:?}, slow {:?}",
            fast[i], slow[i]
        );
    }
}

/// FNV-1a over the voxel bytes, for compact equality checks and the golden value.
fn checksum(voxels: &[Voxel]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in voxels {
        for b in v.to_u16().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

#[test]
fn fill_chunk_matches_sample_at_every_position() {
    let s = sampler();
    let mut surfaces = 0;
    for level in [0u8, 2, 5] {
        for (label, dir) in special_directions() {
            let key = surface_key(&s, dir, level);
            for key in [key, key.offset(0, 1, 0), key.offset(1, 0, -1)] {
                let fast = fast_fill(&s, key);
                let slow = slow_fill(&s, key);
                assert_same(
                    fast.samples(),
                    slow.samples(),
                    &format!("{label}, level {level}, key {key:?}"),
                );
                surfaces += usize::from(fast.has_surface());
            }
        }
    }
    assert!(surfaces >= 20, "only {surfaces} surface chunks exercised");
}

#[test]
fn fill_block_matches_sample_and_the_fine_chunks() {
    let s = sampler();
    for level in [1u8, 4] {
        for (label, dir) in [special_directions()[0], special_directions()[2]] {
            let key = surface_key(&s, dir, level);
            let mut fine: BTreeMap<ChunkKey, Chunk> = BTreeMap::new();
            let mut crossings = 0;
            for face in Face::ALL {
                let mut slab = transition_slab(key, face, BASE_VOXEL_M).unwrap();
                s.fill_block(&mut slab);
                let mut reference = slab.clone();
                let (min, dims) = (reference.min(), reference.dims());
                for z in 0..dims.z as i32 {
                    for y in 0..dims.y as i32 {
                        for x in 0..dims.x as i32 {
                            let w = min + IVec3::new(x, y, z);
                            let (d, m) = s.sample(reference.sample_position(w), level - 1);
                            let distance = Voxel::quantise_distance(d / reference.voxel_size_m());
                            let material = if distance < 0 { m } else { Material::AIR };
                            reference.set(w, Voxel { distance, material });
                        }
                    }
                }
                assert_same(
                    slab.voxels(),
                    reference.voxels(),
                    &format!("{label} level {level} {face:?} slab against sample"),
                );

                // Every slab sample equals the fine chunk's sample at the same world index.
                for z in 0..dims.z as i32 {
                    for y in 0..dims.y as i32 {
                        for x in 0..dims.x as i32 {
                            let w = min + IVec3::new(x, y, z);
                            let holder = ChunkKey::chunks_intersecting_samples_at(level - 1, w, w)
                                .into_iter()
                                .next()
                                .unwrap();
                            let chunk = fine.entry(holder).or_insert_with(|| fast_fill(&s, holder));
                            let local = holder.local_sample(w);
                            let v = chunk.get(local.x as isize, local.y as isize, local.z as isize);
                            assert_eq!(slab.get(w), Some(v), "{label} {face:?} sample {w}");
                        }
                    }
                }
                crossings += usize::from(
                    slab.voxels().iter().any(|v| v.is_inside())
                        && slab.voxels().iter().any(|v| !v.is_inside()),
                );
            }
            assert!(
                crossings >= 2,
                "{label} level {level}: surface crosses {crossings} slabs"
            );
        }
    }
}

#[test]
fn chunks_are_identical_across_samplers_threads_orders_and_cache_states() {
    let reference_sampler = sampler();
    let keys: Vec<ChunkKey> = special_directions()
        .iter()
        .flat_map(|&(_, dir)| [0u8, 3].map(|level| surface_key(&reference_sampler, dir, level)))
        .collect();
    let reference: Vec<u64> = keys
        .iter()
        .map(|&k| checksum(fast_fill(&reference_sampler, k).samples()))
        .collect();

    // Another sampler (another run), cold, then warm, then with the cache cleared.
    let other = sampler();
    for round in 0..3 {
        if round == 2 {
            other.clear_height_cache();
        }
        for (k, &expected) in keys.iter().zip(&reference) {
            assert_eq!(
                checksum(fast_fill(&other, *k).samples()),
                expected,
                "round {round}"
            );
        }
    }

    // A cache of four tiles evicts constantly inside every fill.
    let tiny = PlanetSampler::with_cache_capacity(SEED, &PlanetParams::default(), BASE_VOXEL_M, 4);
    for (k, &expected) in keys.iter().zip(&reference).rev() {
        assert_eq!(
            checksum(fast_fill(&tiny, *k).samples()),
            expected,
            "tiny cache"
        );
    }
    assert!(tiny.height_cache_stats().resident_tiles <= 4);

    // One shared sampler, four threads, each generating the keys in a different order.
    let shared = Arc::new(sampler());
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|t| {
                let shared = Arc::clone(&shared);
                let keys = &keys;
                scope.spawn(move || {
                    let n = keys.len();
                    let mut out = vec![0u64; n];
                    for step in 0..n {
                        let i = if t % 2 == 0 {
                            (step + 3 * t) % n
                        } else {
                            n - 1 - step
                        };
                        out[i] = checksum(fast_fill(&shared, keys[i]).samples());
                    }
                    out
                })
            })
            .collect();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), reference);
        }
    });

    let reseeded = PlanetSampler::new(SEED + 1, &PlanetParams::default(), BASE_VOXEL_M);
    assert!(
        keys.iter()
            .zip(&reference)
            .any(|(&k, &c)| checksum(fast_fill(&reseeded, k).samples()) != c),
        "a different seed must change the world"
    );
}

/// Bytes of a few chunks at the design defaults, fixed so a change of the generator, the noise
/// crate or the platform shows up as a failure. Update only with a deliberate change of the world.
#[test]
fn golden_checksum_at_the_defaults() {
    let s = sampler();
    let mut h = 0u64;
    for (_, dir) in special_directions().into_iter().take(4) {
        for level in [0u8, 4] {
            h = splitmix64(h ^ checksum(fast_fill(&s, surface_key(&s, dir, level)).samples()));
        }
    }
    println!("golden checksum {h:#018x}");
    assert_eq!(
        h, GOLDEN_CHECKSUM,
        "the world at the design defaults changed"
    );
}

const GOLDEN_CHECKSUM: u64 = 0x6143_1a27_dfc9_053c;

#[test]
fn classify_is_exact_and_every_surface_lies_in_the_shell() {
    let s = sampler();
    let (shell_lo, shell_hi) = s.shell_m();
    let mut counts = [0usize; 3];
    let mut surfaces = 0;
    let mut dirs: Vec<DVec3> = special_directions().into_iter().map(|(_, d)| d).collect();
    dirs.extend((0..4).map(|k| direction(1000 + k)));
    for level in [0u8, 2, 4, 6, 7] {
        let (inner, outer) = s.shell_at_m(level);
        assert!(inner >= shell_lo && outer <= shell_hi);
        let extent = ChunkKey::at_level(level, 0, 0, 0).extent_m(BASE_VOXEL_M);
        let band = 4.0 * ChunkKey::at_level(level, 0, 0, 0).voxel_size_m(BASE_VOXEL_M);
        let start = inner - 2.0 * band / SLOPE_FACTOR_MIN - 2.0 * extent;
        let stop = outer + 2.0 * band / SLOPE_FACTOR_MIN + 2.0 * extent;
        for (d, &dir) in dirs.iter().enumerate() {
            // Jitter the radial line per direction so keys straddle the thresholds differently.
            let mut r = start + extent * (d as f64 * 0.37).fract();
            while r < stop {
                let key = ChunkKey::containing_at(WorldPos::body_fixed(dir * r), level, &config());
                r += 0.5 * extent;
                let class = s.classify(key);
                let chunk = fast_fill(&s, key);
                match class {
                    ChunkClass::Air => {
                        counts[0] += 1;
                        assert_eq!(chunk.uniform_value(), Some(Voxel::AIR), "{key:?} said air");
                    }
                    ChunkClass::Rock => {
                        counts[1] += 1;
                        assert_eq!(
                            chunk.uniform_value(),
                            Some(Voxel::solid(Material::ROCK)),
                            "{key:?} said rock"
                        );
                    }
                    ChunkClass::Surface => counts[2] += 1,
                }
                if !chunk.has_surface() {
                    continue;
                }
                surfaces += 1;
                // Every stored sign agrees with the level's shell.
                for z in -1..=CHUNK_SIZE as i32 + 1 {
                    for y in -1..=CHUNK_SIZE as i32 + 1 {
                        for x in -1..=CHUNK_SIZE as i32 + 1 {
                            let local = IVec3::new(x, y, z);
                            let r = chunk.sample_position(local).length();
                            let v = chunk.get(x as isize, y as isize, z as isize);
                            if v.is_inside() {
                                assert!(r <= outer, "{key:?}: inside sample at {r} above {outer}");
                            } else {
                                assert!(r >= inner, "{key:?}: outside sample at {r} below {inner}");
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "classify air {} rock {} surface {}; {surfaces} chunks hold a surface",
        counts[0], counts[1], counts[2]
    );
    assert!(counts.iter().all(|&c| c > 10), "{counts:?}");
    assert!(surfaces > 30, "{surfaces}");
}

#[test]
fn classify_on_random_keys_never_hides_a_surface() {
    let s = sampler();
    let (inner, outer) = s.shell_m();
    let mut state = 77u64;
    let mut checked = 0;
    for level in [0u8, 1, 3, 5, 7] {
        let extent = ChunkKey::at_level(level, 0, 0, 0).extent_m(BASE_VOXEL_M);
        for _ in 0..40 {
            state = splitmix64(state);
            let dir = direction(state);
            let t = (state >> 11) as f64 / (1u64 << 53) as f64;
            let r = inner - 3.0 * extent + t * (outer - inner + 6.0 * extent);
            let key = ChunkKey::containing_at(WorldPos::body_fixed(dir * r), level, &config());
            let class = s.classify(key);
            if class != ChunkClass::Surface {
                let chunk = fast_fill(&s, key);
                assert!(
                    !chunk.has_surface(),
                    "{key:?} classified {class:?} holds a surface"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 20, "{checked}");
}

#[test]
fn shell_is_ordered_and_honest() {
    let s = sampler();
    let (inner, outer) = s.shell_m();
    let r = s.radius_m();
    let a = s.params().terrain.amplitude_m;
    assert!(inner < r - a && outer > r + a, "{inner} {outer}");
    // Not absurdly loose at level 0: amplitude, overshoot, detail over the floor, slack.
    let (inner0, outer0) = s.shell_at_m(0);
    assert!(outer0 - r < 1.6 * a + 10.0, "{outer0}");
    assert!(r - inner0 < 1.6 * a + 10.0, "{inner0}");
    // The full detail height and every level's grid height stay inside.
    for k in 0..2000 {
        let dir = direction(5000 + k);
        let h = s.height_m(dir);
        assert!(r + h > inner && r + h < outer);
        for level in [0u8, 3, 7] {
            let g = s.grid_height_m(level, dir);
            let (lo, hi) = s.shell_at_m(level);
            assert!(r + g > lo && r + g < hi, "level {level}: {g}");
        }
    }
}

#[test]
fn octaves_and_detail_drop_with_the_level() {
    let s = sampler();
    let t = &s.params().terrain;
    let mut previous = usize::MAX;
    for level in 0..=10u8 {
        let (octaves, detail, cells) = s.level_octaves(level);
        let voxel = ChunkKey::at_level(level, 0, 0, 0).voxel_size_m(BASE_VOXEL_M);
        assert!(octaves <= previous);
        previous = octaves;
        // The shortest kept octave spans at least two of the largest node spacings, the next one
        // does not.
        let largest_spacing = crate::cubemap::SPACING_MAX_OVER_MEAN * voxel;
        let shortest = t.wavelength_m / f64::from(1u32 << octaves.saturating_sub(1));
        if octaves > 0 {
            assert!(shortest >= 2.0 * largest_spacing);
        }
        if octaves < t.octaves as usize {
            assert!(shortest / 2.0 < 2.0 * largest_spacing || octaves == 0);
        }
        assert_eq!(detail, t.detail_wavelength_m >= 2.0 * voxel);
        let mean_spacing = std::f64::consts::FRAC_PI_2 * s.radius_m() / f64::from(cells);
        assert!(
            mean_spacing <= voxel && mean_spacing > 0.85 * voxel,
            "level {level}"
        );
    }
}

#[test]
fn grid_height_follows_the_full_detail_height() {
    let s = sampler();
    let mut worst = 0.0f64;
    for k in 0..500 {
        let dir = direction(9000 + k);
        worst = worst.max((s.grid_height_m(0, dir) - s.height_m(dir)).abs());
    }
    println!("level 0 grid against full detail height: max difference {worst:.4} m");
    assert!(worst < 0.25, "{worst}");
    assert_eq!(s.height_m(DVec3::ZERO), s.params().terrain.base_height_m);
    assert_eq!(s.height_m(DVec3::X * 5.0), s.height_m(DVec3::X));
}

/// The twelve cube edges as (face a, face b, direction at parameter -1 to 1 along the edge).
fn edge_direction(edge: usize, t: f64) -> (CubeFace, CubeFace, DVec3) {
    let axes = [(0usize, 1usize), (1, 2), (2, 0)];
    let (a, b) = axes[edge / 4];
    let c = 3 - a - b;
    let sa = if edge & 1 == 0 { 1.0 } else { -1.0 };
    let sb = if edge & 2 == 0 { 1.0 } else { -1.0 };
    let mut v = DVec3::ZERO;
    v[a] = sa;
    v[b] = sb;
    v[c] = t;
    let face = |axis: usize, sign: f64| (2 * axis) as u8 + u8::from(sign < 0.0);
    (face(a, sa), face(b, sb), v.normalize())
}

/// Along an edge the two faces' edge nodes are the same directions, so the interpolated height is
/// continuous across the edge (the step is rounding only). Its derivative across the edge comes
/// from each face's own nodes, so the slope factor and with it the distance away from the surface
/// step by the difference of the two interpolation errors (D21 expects this step). Measured over
/// the 12 edges and the 8 corners: within one voxel of the surface and over the saturated band
/// (four voxels each side).
#[test]
fn cube_edge_step_is_small() {
    let s = sampler();
    let mut report = Vec::new();
    for level in 0..=7u8 {
        let voxel = ChunkKey::at_level(level, 0, 0, 0).voxel_size_m(BASE_VOXEL_M);
        let mut height_step = 0.0f64;
        let mut near_step = 0.0f64;
        let mut band_step = 0.0f64;
        let mut compare = |faces: &[CubeFace], dir: DVec3| {
            let surface = s.radius_m() + s.grid_height_m(level, dir);
            for k in -4i32..=4 {
                let p = dir * (surface + f64::from(k) * voxel);
                let values: Vec<(f64, f64)> = faces
                    .iter()
                    .map(|&f| s.on_face(level, f, p).unwrap())
                    .collect();
                for a in &values {
                    for b in &values {
                        height_step = height_step.max((a.0 - b.0).abs());
                        band_step = band_step.max((a.1 - b.1).abs());
                        if k.abs() <= 1 {
                            near_step = near_step.max((a.1 - b.1).abs());
                        }
                    }
                }
            }
        };
        for edge in 0..12 {
            for k in 0..=100 {
                let t = -1.0 + 2.0 * f64::from(k) / 100.0;
                let (fa, fb, dir) = edge_direction(edge, t);
                compare(&[fa, fb], dir);
            }
        }
        for corner in 0..8 {
            let sign = |bit: usize| if corner & bit == 0 { 1.0 } else { -1.0 };
            let dir = DVec3::new(sign(1), sign(2), sign(4)).normalize();
            let faces: Vec<CubeFace> = (0..3)
                .map(|axis| (2 * axis) as u8 + u8::from(dir[axis] < 0.0))
                .collect();
            compare(&faces, dir);
        }
        report.push(format!(
            "level {level}: height {height_step:.1e} m, distance within 1 voxel {:.4} voxels, over the band {:.4} voxels",
            near_step / voxel,
            band_step / voxel
        ));
        assert!(
            height_step < 1e-9 * s.radius_m(),
            "level {level}: height step {height_step}"
        );
        assert!(
            near_step < 0.08 * voxel,
            "level {level}: {near_step} m near the surface"
        );
        assert!(
            band_step < 0.3 * voxel,
            "level {level}: {band_step} m over the band"
        );
    }
    println!(
        "cube edge step, max over the 12 edges and 8 corners:\n  {}",
        report.join("\n  ")
    );
}

#[test]
fn all_six_materials_appear_at_the_defaults() {
    let s = sampler();
    let t = s.params().terrain;
    let mut highest = (f64::MIN, DVec3::Y);
    let mut lowest = (f64::MAX, DVec3::Y);
    let mut steepest = (f64::MIN, DVec3::Y);
    let mut flat = (f64::MAX, DVec3::Y);
    let mut hs = Vec::new();
    let mut slopes = Vec::new();
    for k in 0..20_000 {
        let dir = direction(70_000 + k);
        let (h, slope) = s.grid_height_and_slope(0, dir);
        hs.push(h);
        slopes.push(slope);
        if h > highest.0 {
            highest = (h, dir);
        }
        if h < lowest.0 {
            lowest = (h, dir);
        }
        if slope > steepest.0 && h > t.sand_below_m && h < t.snow_above_m {
            steepest = (slope, dir);
        }
        if slope < flat.0 && h > t.sand_below_m + 5.0 && h < t.snow_above_m - 5.0 {
            flat = (slope, dir);
        }
    }
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * p) as usize]
    };
    println!(
        "height percentiles 1/5/50/95/99: {:.1} {:.1} {:.1} {:.1} {:.1} m; slope 50/90/99/max: {:.3} {:.3} {:.3} {:.3}",
        pct(&mut hs, 0.01),
        pct(&mut hs, 0.05),
        pct(&mut hs, 0.5),
        pct(&mut hs, 0.95),
        pct(&mut hs, 0.99),
        pct(&mut slopes, 0.5),
        pct(&mut slopes, 0.9),
        pct(&mut slopes, 0.99),
        pct(&mut slopes, 1.0)
    );
    let rock_share =
        slopes.iter().filter(|&&x| x > t.rock_slope).count() as f64 / slopes.len() as f64;
    let snow_share = hs.iter().filter(|&&h| h > t.snow_above_m).count() as f64 / hs.len() as f64;
    let sand_share = hs.iter().filter(|&&h| h < t.sand_below_m).count() as f64 / hs.len() as f64;
    println!(
        "surface shares: rock slopes {:.1} %, snow {:.1} %, sand {:.1} %",
        100.0 * rock_share,
        100.0 * snow_share,
        100.0 * sand_share
    );

    let mut seen = [false; Material::COUNT];
    for (_, dir) in [highest, lowest, steepest, flat] {
        let key = surface_key(&s, dir, 0);
        for dy in [-1, 0, 1] {
            for v in fast_fill(&s, key.offset(0, dy, 0)).samples() {
                if v.is_inside() {
                    seen[usize::from(v.material.0)] = true;
                }
            }
        }
    }
    for m in [
        Material::ROCK,
        Material::SOIL,
        Material::GRASS,
        Material::SAND,
        Material::SNOW,
    ] {
        assert!(seen[usize::from(m.0)], "{} never appears", m.name());
    }
    assert!(!seen[usize::from(Material::AIR.0)], "air inside the solid");
}

#[test]
fn sampling_the_centre_is_finite_rock() {
    let s = sampler();
    for level in [0u8, 7] {
        let (d, m) = s.sample(DVec3::ZERO, level);
        assert!(d.is_finite() && d < 0.0);
        assert_eq!(m, Material::ROCK);
    }
    let key = ChunkKey::at_level(6, -1, -1, -1);
    let chunk = fast_fill(&s, key);
    assert!(chunk.samples().iter().all(|v| v.is_inside()));
    assert_eq!(s.classify(ChunkKey::new(0, 0, 0)), ChunkClass::Rock);
}

#[test]
fn range_ends_fill_like_sample() {
    let d = PlanetParams::default();
    let rough = PlanetParams {
        radius_km: *PlanetParams::RADIUS_RANGE_KM.start(),
        terrain: TerrainParams {
            amplitude_m: 400.0,
            wavelength_m: *TerrainParams::WAVELENGTH_RANGE_M.start() * 8.0,
            octaves: *TerrainParams::OCTAVES_RANGE.end(),
            detail_amplitude_m: *TerrainParams::DETAIL_AMPLITUDE_RANGE_M.end(),
            detail_wavelength_m: *TerrainParams::DETAIL_WAVELENGTH_RANGE_M.start(),
            rock_slope: *TerrainParams::ROCK_SLOPE_RANGE.start(),
            ..d.terrain
        },
        ..d
    };
    let smooth = PlanetParams {
        radius_km: 50.0,
        terrain: TerrainParams {
            amplitude_m: *TerrainParams::AMPLITUDE_RANGE_M.start(),
            octaves: *TerrainParams::OCTAVES_RANGE.start(),
            detail_amplitude_m: *TerrainParams::DETAIL_AMPLITUDE_RANGE_M.start(),
            detail_wavelength_m: *TerrainParams::DETAIL_WAVELENGTH_RANGE_M.end(),
            ..d.terrain
        },
        ..d
    };
    for (i, (params, base)) in [(rough, 0.25), (smooth, 4.0), (d, 0.5)]
        .into_iter()
        .enumerate()
    {
        params.validate().unwrap();
        let s = PlanetSampler::new(SEED + i as u64, &params, base);
        for (label, dir) in special_directions().into_iter().take(3) {
            for level in [0u8, 3] {
                let p = dir * (s.radius_m() + s.grid_height_m(level, dir));
                let size = ChunkKey::at_level(level, 0, 0, 0).extent_m(base);
                let f = |v: f64| (v / size).floor() as i32;
                let key = ChunkKey::at_level(level, f(p.x), f(p.y), f(p.z));
                let mut fast = Chunk::new(key, base);
                s.fill_chunk(&mut fast);
                let mut slow = Chunk::new(key, base);
                slow.fill_from_with_material(|q| s.sample(q, level));
                assert_same(
                    fast.samples(),
                    slow.samples(),
                    &format!("case {i} {label} level {level}"),
                );
            }
        }
    }
}

#[test]
fn planet_fill_timing() {
    let s = sampler();
    let runs = if cfg!(debug_assertions) { 2 } else { 20 };
    for level in [0u8, 3, 6] {
        let key = surface_key(&s, special_directions()[0].1, level);
        let mut chunk = Chunk::new(key, BASE_VOXEL_M);
        let mut cold = Vec::new();
        let mut warm = Vec::new();
        for _ in 0..runs {
            s.clear_height_cache();
            let t0 = Instant::now();
            s.fill_chunk(&mut chunk);
            cold.push(t0.elapsed().as_secs_f64() * 1e3);
            let t0 = Instant::now();
            s.fill_chunk(&mut chunk);
            warm.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        cold.sort_by(f64::total_cmp);
        warm.sort_by(f64::total_cmp);
        println!(
            "fill_chunk level {level} {key:?}: cold {:.3} ms, warm {:.3} ms (median of {runs}, has_surface {})",
            cold[runs / 2],
            warm[runs / 2],
            chunk.has_surface()
        );
    }
}

/// The constants every early out rests on, checked over a fine grid of fractions: the 1D weights'
/// magnitudes sum to at most 5/4, the tensor product's negative weights to at most 9/32, and the
/// 1D derivative, written over the three node differences, has coefficients whose magnitudes sum
/// to at most 3/2.
#[test]
fn catmull_rom_bounds_hold() {
    let steps = 1000;
    let mut weight_sum = 0.0f64;
    let mut negative_2d = 0.0f64;
    let mut difference_sum = 0.0f64;
    for a in 0..=steps {
        let s = f64::from(a) / f64::from(steps);
        let (ws, dws) = catmull_rom_weights(s);
        assert!((ws.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(dws.iter().sum::<f64>().abs() < 1e-12);
        weight_sum = weight_sum.max(ws.iter().map(|w| w.abs()).sum());
        // f' = sum dws_i n_i = sum_k c_k (n_{k+1} - n_k) with c_k = -(dws_0 + ... + dws_k).
        let c0 = -dws[0];
        let c1 = -(dws[0] + dws[1]);
        let c2 = -(dws[0] + dws[1] + dws[2]);
        difference_sum = difference_sum.max(c0.abs() + c1.abs() + c2.abs());
        for b in (0..=steps).step_by(10) {
            let t = f64::from(b) / f64::from(steps);
            let (wt, _) = catmull_rom_weights(t);
            let negative: f64 = ws
                .iter()
                .flat_map(|x| wt.iter().map(move |y| x * y))
                .filter(|w| *w < 0.0)
                .map(f64::abs)
                .sum();
            negative_2d = negative_2d.max(negative);
        }
    }
    println!(
        "Catmull-Rom: weight magnitude sum {weight_sum:.6}, 2D negative sum {negative_2d:.6}, derivative coefficients {difference_sum:.6}"
    );
    assert!(weight_sum <= 1.25 + 1e-12);
    assert!(negative_2d <= CATMULL_ROM_OVERSHOOT + 1e-12);
    assert!(difference_sum <= 1.5 + 1e-12);
    assert!(CATMULL_ROM_SLOPE_PER_DIFFERENCE >= difference_sum * weight_sum - 1e-12);
    // The bounds are reached (at s = t = 1/2), so they are tight as well as sound.
    assert!(negative_2d > CATMULL_ROM_OVERSHOOT - 1e-9 && difference_sum > 1.5 - 1e-9);
}
