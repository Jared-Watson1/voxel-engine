use glam::{DVec3, IVec3};
use vx_core::{CHUNK_SIZE, WorldConfig, WorldPos};

use crate::chunk::{Chunk, DISTANCE_UNITS_PER_VOXEL, Material, Voxel};
use crate::key::ChunkKey;

/// Shape of an edit in the body fixed frame, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BrushShape {
    Sphere {
        centre: WorldPos,
        radius_m: f64,
    },
    /// Axis aligned box from `min` to `max`.
    Box {
        min: WorldPos,
        max: WorldPos,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushOp {
    /// Removes the shape from the solid: `d = max(d, -shape)`; samples that leave the solid become
    /// air.
    Dig,
    /// Adds the shape to the solid with a material: `d = min(d, shape)`; samples the shape brought
    /// inside take the material.
    Place(Material),
}

/// A signed distance brush (design document, interaction). Applying it to every chunk in
/// [`Brush::affected_chunks`] keeps all apron copies of each sample equal (D16).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brush {
    pub shape: BrushShape,
    pub op: BrushOp,
}

impl Brush {
    /// Signed distance from the shape's surface in metres, negative inside.
    pub fn signed_distance_m(&self, p: DVec3) -> f64 {
        match self.shape {
            BrushShape::Sphere { centre, radius_m } => (p - centre.m).length() - radius_m,
            BrushShape::Box { min, max } => {
                // Exact box distance, https://iquilezles.org/articles/distfunctions/
                let centre = 0.5 * (min.m + max.m);
                let half_extent = 0.5 * (max.m - min.m);
                let q = (p - centre).abs() - half_extent;
                q.max(DVec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
            }
        }
    }

    /// Inclusive world sample box (in voxels) outside which no sample can change: the shape's
    /// bounds padded by the representable distance band, because a dig also rewrites the
    /// distances of nearby solid samples.
    pub fn sample_bounds(&self, config: &WorldConfig) -> (IVec3, IVec3) {
        self.sample_bounds_at(config.voxel_size_m)
    }

    /// Every chunk whose stored block intersects [`Brush::sample_bounds`]: the set that must be
    /// edited and remeshed.
    pub fn affected_chunks(&self, config: &WorldConfig) -> Vec<ChunkKey> {
        let (min, max) = self.sample_bounds(config);
        ChunkKey::chunks_intersecting_samples(min, max)
    }

    /// Applies the brush to every sample of `chunk` inside the bounds, apron included. Returns
    /// true if any sample changed.
    pub fn apply(&self, chunk: &mut Chunk) -> bool {
        profiling::scope!("Brush::apply");
        let voxel_size_m = chunk.voxel_size_m();
        let (min, max) = self.sample_bounds_at(voxel_size_m);
        let key = chunk.key();
        let min = key.local_sample(min).max(IVec3::splat(-1));
        let max = key
            .local_sample(max)
            .min(IVec3::splat(CHUNK_SIZE as i32 + 1));
        if min.cmpgt(max).any() {
            return false;
        }

        // Same arithmetic as Chunk::sample_position (integer world sample index times the voxel
        // size, one rounding, D16), hoisted out of the loop.
        let first_sample = chunk.key().world_sample(IVec3::ZERO);
        let samples = chunk.samples_mut();
        let mut changed = false;
        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let local = IVec3::new(x, y, z);
                    let p = (first_sample + local).as_dvec3() * voxel_size_m;
                    let brush_voxels = self.signed_distance_m(p) / voxel_size_m;
                    let i = Chunk::linear(stored(x), stored(y), stored(z));
                    let old = samples[i];
                    let new = self.op.combine(old, brush_voxels);
                    if new != old {
                        samples[i] = new;
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    /// Metre bounds of the shape itself, before padding.
    fn bounds_m(&self) -> (DVec3, DVec3) {
        match self.shape {
            BrushShape::Sphere { centre, radius_m } => (
                centre.m - DVec3::splat(radius_m),
                centre.m + DVec3::splat(radius_m),
            ),
            BrushShape::Box { min, max } => (min.m, max.m),
        }
    }

    fn sample_bounds_at(&self, voxel_size_m: f64) -> (IVec3, IVec3) {
        let (min_m, max_m) = self.bounds_m();
        let pad = IVec3::splat(band_padding_samples());
        let min = (min_m / voxel_size_m).floor().as_ivec3() - pad;
        let max = (max_m / voxel_size_m).ceil().as_ivec3() + pad;
        (min, max)
    }
}

impl BrushOp {
    /// Combines a stored sample with the brush distance (both in voxels) in f64, then requantises.
    /// A sample the brush does not reach requantises to the same i8, so untouched samples stay bit
    /// identical.
    fn combine(self, old: Voxel, brush_voxels: f64) -> Voxel {
        let old_voxels = f64::from(old.distance_voxels());
        match self {
            Self::Dig => {
                let distance = Voxel::quantise_distance(old_voxels.max(-brush_voxels));
                let material = if distance < 0 {
                    old.material
                } else {
                    Material::AIR
                };
                Voxel { distance, material }
            }
            Self::Place(material) => {
                let distance = Voxel::quantise_distance(old_voxels.min(brush_voxels));
                let material = if old.is_inside() {
                    old.material
                } else if distance < 0 {
                    material
                } else {
                    Material::AIR
                };
                Voxel { distance, material }
            }
        }
    }
}

/// Samples beyond the shape's bounds that an edit can still rewrite: the representable distance
/// band rounded up, plus one for the floor and ceil of the bounds.
fn band_padding_samples() -> i32 {
    (f64::from(i8::MAX) / f64::from(DISTANCE_UNITS_PER_VOXEL)).ceil() as i32 + 1
}

/// Stored coordinate of a local coordinate in -1 to `CHUNK_SIZE + 1`.
fn stored(local: i32) -> usize {
    (local + 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> WorldConfig {
        WorldConfig::default()
    }

    fn sphere(centre_m: DVec3, radius_m: f64, op: BrushOp) -> Brush {
        Brush {
            shape: BrushShape::Sphere {
                centre: WorldPos::body_fixed(centre_m),
                radius_m,
            },
            op,
        }
    }

    fn boxed(min_m: DVec3, max_m: DVec3, op: BrushOp) -> Brush {
        Brush {
            shape: BrushShape::Box {
                min: WorldPos::body_fixed(min_m),
                max: WorldPos::body_fixed(max_m),
            },
            op,
        }
    }

    /// Flat ground: rock below `ground_m`, air above.
    fn ground_chunk(key: ChunkKey, voxel_size_m: f64, ground_m: f64) -> Chunk {
        let mut chunk = Chunk::new(key, voxel_size_m);
        chunk.fill_from(|p| p.y - ground_m);
        chunk
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn sphere_and_box_distances() {
        let s = sphere(DVec3::new(1.0, 2.0, 3.0), 2.0, BrushOp::Dig);
        approx(s.signed_distance_m(DVec3::new(1.0, 2.0, 3.0)), -2.0);
        approx(s.signed_distance_m(DVec3::new(4.0, 2.0, 3.0)), 1.0);
        approx(s.signed_distance_m(DVec3::new(1.0, 0.0, 3.0)), 0.0);

        let b = boxed(DVec3::splat(-1.0), DVec3::new(3.0, 1.0, 1.0), BrushOp::Dig);
        approx(b.signed_distance_m(DVec3::new(1.0, 0.0, 0.0)), -1.0);
        approx(b.signed_distance_m(DVec3::new(2.5, 0.0, 0.0)), -0.5);
        approx(b.signed_distance_m(DVec3::new(5.0, 0.0, 0.0)), 2.0);
        approx(b.signed_distance_m(DVec3::new(-1.0, 1.0, 0.0)), 0.0);
        approx(b.signed_distance_m(DVec3::new(4.0, 2.0, 2.0)), 3f64.sqrt());
    }

    #[test]
    fn sample_bounds_pad_by_the_distance_band() {
        let pad = band_padding_samples();
        assert_eq!(pad, 5);
        let s = sphere(DVec3::splat(16.0), 2.0, BrushOp::Dig);
        assert_eq!(
            s.sample_bounds(&config()),
            (IVec3::splat(14 - pad), IVec3::splat(18 + pad))
        );
        let half = WorldConfig {
            voxel_size_m: 0.5,
            ..config()
        };
        assert_eq!(
            s.sample_bounds(&half),
            (IVec3::splat(28 - pad), IVec3::splat(36 + pad))
        );
        let b = boxed(
            DVec3::new(0.3, -0.3, 2.0),
            DVec3::new(1.2, 1.0, 2.0),
            BrushOp::Dig,
        );
        assert_eq!(
            b.sample_bounds(&config()),
            (
                IVec3::new(-pad, -1 - pad, 2 - pad),
                IVec3::new(2 + pad, 1 + pad, 2 + pad)
            )
        );
    }

    #[test]
    fn dirty_set_grows_at_chunk_faces_edges_and_corners() {
        let cfg = config();
        let extent = cfg.chunk_extent_m();
        let r = 2.0 * cfg.voxel_size_m;
        let mid = 0.5 * extent;
        let cases = [
            (DVec3::new(mid, mid, mid), 1),
            (DVec3::new(0.0, mid, mid), 2),
            (DVec3::new(0.0, 0.0, mid), 4),
            (DVec3::new(0.0, 0.0, 0.0), 8),
            (DVec3::new(extent, mid, mid), 2),
            (DVec3::new(-extent, -extent, -extent), 8),
        ];
        for (centre, count) in cases {
            let keys = sphere(centre, r, BrushOp::Dig).affected_chunks(&cfg);
            assert_eq!(keys.len(), count, "sphere at {centre}");
            let mut sorted = keys.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), keys.len(), "duplicate keys at {centre}");
        }
        assert_eq!(
            sphere(DVec3::new(mid, mid, mid), r, BrushOp::Dig).affected_chunks(&cfg),
            vec![ChunkKey::new(0, 0, 0)]
        );
    }

    #[test]
    fn dirty_set_of_a_large_sphere_follows_the_count_formula() {
        let cfg = config();
        let brush = sphere(DVec3::splat(0.5 * cfg.chunk_extent_m()), 40.0, BrushOp::Dig);
        let (min, max) = brush.sample_bounds(&cfg);
        let keys = brush.affected_chunks(&cfg);
        let size = CHUNK_SIZE as i32;
        let per_axis = |lo: i32, hi: i32| {
            let first = (lo - (size + 1) + size - 1).div_euclid(size);
            let last = (hi + 1).div_euclid(size);
            (last - first + 1) as usize
        };
        let expected = per_axis(min.x, max.x) * per_axis(min.y, max.y) * per_axis(min.z, max.z);
        assert_eq!(keys.len(), expected);
        assert!(keys.len() >= 27, "{} keys", keys.len());
        for key in &keys {
            assert!(keys.iter().filter(|k| *k == key).count() == 1);
        }
    }

    #[test]
    fn dirty_set_holds_at_both_ends_of_the_voxel_size_range() {
        for voxel_size_m in [
            *WorldConfig::VOXEL_SIZE_RANGE_M.start(),
            *WorldConfig::VOXEL_SIZE_RANGE_M.end(),
        ] {
            let cfg = WorldConfig {
                voxel_size_m,
                ..config()
            };
            let mid = 0.5 * cfg.chunk_extent_m();
            let r = 2.0 * voxel_size_m;
            assert_eq!(
                sphere(DVec3::splat(mid), r, BrushOp::Dig)
                    .affected_chunks(&cfg)
                    .len(),
                1
            );
            assert_eq!(
                sphere(DVec3::new(0.0, mid, mid), r, BrushOp::Dig)
                    .affected_chunks(&cfg)
                    .len(),
                2
            );
            assert_eq!(
                sphere(DVec3::ZERO, r, BrushOp::Dig)
                    .affected_chunks(&cfg)
                    .len(),
                8
            );
        }
    }

    /// Applies `brush` to every affected chunk, each first filled with flat ground, and checks
    /// that every pair of chunks agrees on every shared world sample (D16).
    fn assert_apron_consistent(brush: Brush, ground_m: f64) -> Vec<Chunk> {
        let cfg = config();
        let keys = brush.affected_chunks(&cfg);
        assert!(keys.len() >= 2, "brush must span chunks");
        let chunks: Vec<Chunk> = keys
            .iter()
            .map(|&key| {
                let mut chunk = ground_chunk(key, cfg.voxel_size_m, ground_m);
                assert!(brush.apply(&mut chunk), "brush left {key:?} unchanged");
                chunk
            })
            .collect();
        let last = CHUNK_SIZE as i32 + 1;
        let mut shared = 0usize;
        for a in &chunks {
            for b in &chunks {
                if a.key() == b.key() {
                    continue;
                }
                for z in -1..=last {
                    for y in -1..=last {
                        for x in -1..=last {
                            let world = a.key().world_sample(IVec3::new(x, y, z));
                            let lb = b.key().local_sample(world);
                            if lb.cmplt(IVec3::splat(-1)).any()
                                || lb.cmpgt(IVec3::splat(last)).any()
                            {
                                continue;
                            }
                            let va = a.get(x as isize, y as isize, z as isize);
                            let vb = b.get(lb.x as isize, lb.y as isize, lb.z as isize);
                            assert_eq!(
                                va,
                                vb,
                                "world sample {world} in {:?} and {:?}",
                                a.key(),
                                b.key()
                            );
                            shared += 1;
                        }
                    }
                }
            }
        }
        assert!(shared > 0);
        chunks
    }

    #[test]
    fn dig_at_a_chunk_corner_keeps_aprons_consistent() {
        let chunks = assert_apron_consistent(sphere(DVec3::ZERO, 3.0, BrushOp::Dig), 10.0);
        assert_eq!(chunks.len(), 8);
    }

    #[test]
    fn place_box_across_a_chunk_corner_keeps_aprons_consistent() {
        let brush = boxed(
            DVec3::new(-2.0, 8.0, -2.0),
            DVec3::new(3.0, 14.0, 3.0),
            BrushOp::Place(Material::SAND),
        );
        let chunks = assert_apron_consistent(brush, 10.0);
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn dig_makes_air_and_keeps_remaining_rock() {
        let cfg = config();
        let ground_m = 10.0;
        let mut chunk = ground_chunk(ChunkKey::new(0, 0, 0), cfg.voxel_size_m, ground_m);
        let centre = DVec3::new(16.0, 8.0, 16.0);
        let brush = sphere(centre, 3.0, BrushOp::Dig);
        assert!(brush.apply(&mut chunk));

        let mut dug = 0;
        for z in -1..=33 {
            for y in -1..=33 {
                for x in -1..=33 {
                    let v = chunk.get(x, y, z);
                    let p = chunk.sample_position(IVec3::new(x as i32, y as i32, z as i32));
                    let d = brush.signed_distance_m(p);
                    if d < -0.5 {
                        assert!(
                            v.distance > 0 && v.material == Material::AIR,
                            "({x}, {y}, {z})"
                        );
                        dug += 1;
                    } else if d > 4.5 && p.y < ground_m - 0.5 {
                        assert_eq!(v.material, Material::ROCK, "({x}, {y}, {z})");
                        assert!(v.is_inside());
                    }
                }
            }
        }
        assert!(dug > 50, "{dug} samples dug");
        assert!(!brush.apply(&mut chunk), "a second dig changes nothing");
    }

    #[test]
    fn place_sets_the_material_and_keeps_existing_solid() {
        let cfg = config();
        let ground_m = 10.0;
        let mut chunk = ground_chunk(ChunkKey::new(0, 0, 0), cfg.voxel_size_m, ground_m);
        let (min, max) = (DVec3::new(10.0, 6.0, 10.0), DVec3::new(20.0, 14.0, 20.0));
        let brush = boxed(min, max, BrushOp::Place(Material::SAND));
        assert!(brush.apply(&mut chunk));

        let mut placed = 0;
        for z in -1..=33 {
            for y in -1..=33 {
                for x in -1..=33 {
                    let v = chunk.get(x, y, z);
                    let p = chunk.sample_position(IVec3::new(x as i32, y as i32, z as i32));
                    let d = brush.signed_distance_m(p);
                    if p.y < ground_m - 0.5 {
                        assert_eq!(v.material, Material::ROCK, "({x}, {y}, {z})");
                        assert!(v.is_inside());
                    } else if d < -0.5 {
                        assert_eq!(v.material, Material::SAND, "({x}, {y}, {z})");
                        assert!(v.is_inside());
                        placed += 1;
                    } else if d > 0.5 && p.y > ground_m + 0.5 {
                        assert_eq!(v.material, Material::AIR, "({x}, {y}, {z})");
                        assert!(!v.is_inside(), "({x}, {y}, {z})");
                    }
                }
            }
        }
        assert!(placed > 50, "{placed} samples placed");
        assert!(!brush.apply(&mut chunk), "a second place changes nothing");
    }

    #[test]
    fn brush_outside_the_padded_range_changes_nothing() {
        let cfg = config();
        let base = ground_chunk(ChunkKey::new(0, 0, 0), cfg.voxel_size_m, 10.0);
        let far = [
            sphere(DVec3::splat(200.0), 2.0, BrushOp::Dig),
            sphere(
                DVec3::new(16.0, -40.0, 16.0),
                2.0,
                BrushOp::Place(Material::SAND),
            ),
            boxed(DVec3::splat(-100.0), DVec3::splat(-60.0), BrushOp::Dig),
        ];
        for brush in far {
            let mut chunk = base.clone();
            assert!(!brush.apply(&mut chunk), "{brush:?}");
            assert!(chunk.samples() == base.samples(), "{brush:?}");
        }
    }

    #[test]
    fn untouched_samples_stay_bit_identical() {
        let cfg = config();
        let base = ground_chunk(ChunkKey::new(0, 0, 0), cfg.voxel_size_m, 10.0);
        // Inside the padded range but farther than the band from every sample: nothing changes.
        let near_air = sphere(DVec3::new(16.0, 38.0, 16.0), 2.0, BrushOp::Dig);
        let mut chunk = base.clone();
        assert!(!near_air.apply(&mut chunk));
        assert!(chunk.samples() == base.samples());

        // A dig in the middle leaves every sample beyond the band equal to the base.
        let brush = sphere(DVec3::new(16.0, 8.0, 16.0), 3.0, BrushOp::Dig);
        let mut chunk = base.clone();
        assert!(brush.apply(&mut chunk));
        let band_m = f64::from(i8::MAX) / f64::from(DISTANCE_UNITS_PER_VOXEL) * cfg.voxel_size_m;
        let mut untouched = 0;
        for z in -1..=33 {
            for y in -1..=33 {
                for x in -1..=33 {
                    let p = chunk.sample_position(IVec3::new(x as i32, y as i32, z as i32));
                    if brush.signed_distance_m(p) > band_m {
                        assert_eq!(chunk.get(x, y, z), base.get(x, y, z), "({x}, {y}, {z})");
                        untouched += 1;
                    }
                }
            }
        }
        assert!(untouched > 1000);

        // Requantising a stored distance gives the same i8 for every representable value.
        for d in -i8::MAX..=i8::MAX {
            if d == 0 {
                continue;
            }
            let material = if d < 0 { Material::ROCK } else { Material::AIR };
            let v = Voxel {
                distance: d,
                material,
            };
            assert_eq!(Voxel::quantise_distance(f64::from(v.distance_voxels())), d);
            assert_eq!(BrushOp::Dig.combine(v, 100.0), v);
            assert_eq!(BrushOp::Place(Material::SAND).combine(v, 100.0), v);
        }
    }

    #[test]
    fn dig_rewrites_deep_rock_near_the_hole() {
        let cfg = config();
        let mut chunk = ground_chunk(ChunkKey::new(0, 0, 0), cfg.voxel_size_m, 30.0);
        assert_eq!(chunk.get(16, 8, 16).distance, -i8::MAX);
        let brush = sphere(DVec3::new(16.0, 8.0, 16.0), 2.0, BrushOp::Dig);
        assert!(brush.apply(&mut chunk));
        // One voxel outside the sphere: distance becomes -1 voxel, material stays rock.
        let v = chunk.get(16, 11, 16);
        assert_eq!(v.distance, -32);
        assert_eq!(v.material, Material::ROCK);
        // Three voxels outside: -3 voxels.
        assert_eq!(chunk.get(16, 13, 16).distance, -96);
        // Beyond the band: untouched.
        assert_eq!(chunk.get(16, 14, 16).distance, -i8::MAX);
    }
}
