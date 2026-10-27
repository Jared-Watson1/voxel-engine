use glam::DVec3;
use vx_core::{CHUNK_SIZE, WorldPos};

/// Samples along each axis of a chunk: `CHUNK_SIZE` cells need `CHUNK_SIZE + 1` corner samples,
/// and Transvoxel's central difference gradients reach one sample further on each side, so the
/// stored range is -1 to `CHUNK_SIZE + 1` inclusive.
pub const SAMPLES_PER_AXIS: usize = CHUNK_SIZE + 3;

/// Quantisation scale of the stored signed distance: one voxel edge equals this many i8 units, so
/// the representable band is about +-4 voxels around the surface (D2).
pub const DISTANCE_UNITS_PER_VOXEL: f32 = 32.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Material(pub u8);

impl Material {
    pub const AIR: Self = Self(0);
    pub const ROCK: Self = Self(1);
}

/// One voxel sample: quantised signed distance (negative inside) and a material id (D2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Voxel {
    pub distance: i8,
    pub material: Material,
}

impl Voxel {
    pub const AIR: Self = Self {
        distance: i8::MAX,
        material: Material::AIR,
    };

    /// Quantises a signed distance given in voxels.
    ///
    /// Zero is never stored: a sample exactly on the surface is nudged one unit outward, so every
    /// mesh vertex lies strictly inside a cell edge and no triangle collapses to zero area.
    pub fn quantise_distance(distance_voxels: f64) -> i8 {
        let q = (distance_voxels * f64::from(DISTANCE_UNITS_PER_VOXEL))
            .round()
            .clamp(-f64::from(i8::MAX), f64::from(i8::MAX)) as i8;
        if q == 0 { 1 } else { q }
    }

    /// Signed distance in voxels.
    pub fn distance_voxels(self) -> f32 {
        f32::from(self.distance) / DISTANCE_UNITS_PER_VOXEL
    }

    pub fn is_inside(self) -> bool {
        self.distance < 0
    }
}

/// A dense `SAMPLES_PER_AXIS`^3 block of voxels whose sample (0, 0, 0) sits at `origin`.
///
/// Sample spacing is `voxel_size_m` from the world config (D11), never a literal.
pub struct Chunk {
    origin: WorldPos,
    voxel_size_m: f64,
    voxels: Box<[Voxel]>,
}

impl Chunk {
    pub fn new(origin: WorldPos, voxel_size_m: f64) -> Self {
        assert!(voxel_size_m > 0.0, "voxel size must be positive");
        Self {
            origin,
            voxel_size_m,
            voxels: vec![Voxel::AIR; SAMPLES_PER_AXIS.pow(3)].into_boxed_slice(),
        }
    }

    pub fn origin(&self) -> WorldPos {
        self.origin
    }

    pub fn voxel_size_m(&self) -> f64 {
        self.voxel_size_m
    }

    /// Fills every sample, apron included, from a signed distance function of the frame position in
    /// metres (negative inside). Material is rock inside and air outside.
    pub fn fill_from(&mut self, sdf: impl Fn(DVec3) -> f64) {
        profiling::scope!("Chunk::fill_from");
        let base = self.origin.m;
        let inv_voxel = 1.0 / self.voxel_size_m;
        for z in 0..SAMPLES_PER_AXIS {
            for y in 0..SAMPLES_PER_AXIS {
                for x in 0..SAMPLES_PER_AXIS {
                    let local = DVec3::new(x as f64, y as f64, z as f64) - DVec3::ONE;
                    let distance_m = sdf(base + local * self.voxel_size_m);
                    let distance = Voxel::quantise_distance(distance_m * inv_voxel);
                    let material = if distance < 0 {
                        Material::ROCK
                    } else {
                        Material::AIR
                    };
                    self.voxels[Self::linear(x, y, z)] = Voxel { distance, material };
                }
            }
        }
    }

    /// Sample at grid coordinates in -1 to `CHUNK_SIZE + 1` inclusive.
    pub fn get(&self, x: isize, y: isize, z: isize) -> Voxel {
        let offset = |v: isize| {
            let i = v + 1;
            debug_assert!(
                (0..SAMPLES_PER_AXIS as isize).contains(&i),
                "sample {v} outside apron"
            );
            i as usize
        };
        self.voxels[Self::linear(offset(x), offset(y), offset(z))]
    }

    fn linear(x: usize, y: usize, z: usize) -> usize {
        (z * SAMPLES_PER_AXIS + y) * SAMPLES_PER_AXIS + x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantisation_round_trips_within_half_a_unit() {
        for d in [-3.9, -1.0, -0.26, 0.03, 0.5, 2.0, 3.9] {
            let q = Voxel::quantise_distance(d);
            let back = f64::from(
                Voxel {
                    distance: q,
                    material: Material::AIR,
                }
                .distance_voxels(),
            );
            assert!(
                (back - d).abs() <= 0.5 / f64::from(DISTANCE_UNITS_PER_VOXEL) + 1e-9,
                "{d}"
            );
        }
    }

    #[test]
    fn quantisation_clamps_and_never_stores_zero() {
        assert_eq!(Voxel::quantise_distance(100.0), i8::MAX);
        assert_eq!(Voxel::quantise_distance(-100.0), -i8::MAX);
        assert_eq!(Voxel::quantise_distance(0.0), 1);
        assert_eq!(Voxel::quantise_distance(0.001), 1);
        assert_eq!(Voxel::quantise_distance(-0.001), 1);
    }

    #[test]
    fn samples_are_spaced_by_the_voxel_size() {
        for voxel_size_m in [0.25, 1.0, 4.0] {
            let origin = WorldPos::body_fixed(DVec3::new(100.0, -50.0, 3.0));
            let mut chunk = Chunk::new(origin, voxel_size_m);
            // A plane whose signed distance is the height above the chunk origin in metres.
            chunk.fill_from(|p| p.y - origin.m.y - 2.0 * voxel_size_m);
            assert!(chunk.get(5, 1, 5).is_inside());
            assert!(!chunk.get(5, 3, 5).is_inside());
            assert_eq!(chunk.get(0, -1, 0).distance_voxels(), -3.0);
            assert_eq!(chunk.get(0, CHUNK_SIZE as isize + 1, 0).distance, i8::MAX);
        }
    }
}
