use glam::{DVec3, IVec3};
use vx_core::{CHUNK_SIZE, WorldConfig, WorldPos};

use crate::key::ChunkKey;

/// Samples along each axis of a chunk (D16): `CHUNK_SIZE` cells need `CHUNK_SIZE + 1` corner
/// samples, and Transvoxel's central difference gradients reach one sample further on each side,
/// so the stored range is -1 to `CHUNK_SIZE + 1` inclusive.
pub const SAMPLES_PER_AXIS: usize = CHUNK_SIZE + 3;

/// Quantisation scale of the stored signed distance: one voxel edge equals this many i8 units, so
/// the representable band is about +-4 voxels around the surface (D2).
pub const DISTANCE_UNITS_PER_VOXEL: f32 = 32.0;

/// Material id. Ids 1 to 15 fit the packed vertex (D14); 0 is air.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Material(pub u8);

impl Material {
    pub const AIR: Self = Self(0);
    pub const ROCK: Self = Self(1);
    pub const SOIL: Self = Self(2);
    pub const GRASS: Self = Self(3);
    pub const SAND: Self = Self(4);
    /// Snow above the snow line (M2).
    pub const SNOW: Self = Self(5);
    /// Largest id the vertex format carries.
    pub const MAX_ID: u8 = vx_core::mesh::MAX_MATERIAL_ID;
    /// Number of material ids with a texture layer, air included; texture arrays have this many
    /// layers.
    pub const COUNT: usize = 6;

    pub fn name(self) -> &'static str {
        match self {
            Self::AIR => "air",
            Self::ROCK => "rock",
            Self::SOIL => "soil",
            Self::GRASS => "grass",
            Self::SAND => "sand",
            Self::SNOW => "snow",
            _ => "unknown",
        }
    }
}

/// One voxel sample: quantised signed distance (negative inside) and a material id (D2).
///
/// The material of a sample outside the solid is always [`Material::AIR`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Voxel {
    pub distance: i8,
    pub material: Material,
}

impl Voxel {
    pub const AIR: Self = Self {
        distance: i8::MAX,
        material: Material::AIR,
    };

    pub const fn solid(material: Material) -> Self {
        Self {
            distance: -i8::MAX,
            material,
        }
    }

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

    /// Packs the pair into one u16 (distance in the low byte), the palette key in [`crate::ChunkStore`].
    pub fn to_u16(self) -> u16 {
        u16::from(self.distance as u8) | (u16::from(self.material.0) << 8)
    }

    pub fn from_u16(v: u16) -> Self {
        Self {
            distance: (v & 0xFF) as u8 as i8,
            material: Material((v >> 8) as u8),
        }
    }
}

/// A dense `SAMPLES_PER_AXIS`^3 block of voxels for chunk `key`: the working form for
/// generation, brushes and meshing. The resident form is [`crate::ChunkStore`].
///
/// Sample spacing is the level's voxel size, `voxel_size_m * 2^level` from the world config (D11,
/// D20), never a literal.
#[derive(Clone)]
pub struct Chunk {
    key: ChunkKey,
    voxel_size_m: f64,
    voxels: Box<[Voxel]>,
}

impl Chunk {
    /// An all air chunk for `key`, whose sample spacing is `base_voxel_size_m * 2^key.level`.
    pub fn new(key: ChunkKey, base_voxel_size_m: f64) -> Self {
        assert!(base_voxel_size_m > 0.0, "voxel size must be positive");
        Self {
            key,
            voxel_size_m: key.voxel_size_m(base_voxel_size_m),
            voxels: vec![Voxel::AIR; SAMPLES_PER_AXIS.pow(3)].into_boxed_slice(),
        }
    }

    pub fn for_key(key: ChunkKey, config: &WorldConfig) -> Self {
        Self::new(key, config.voxel_size_m)
    }

    pub fn key(&self) -> ChunkKey {
        self.key
    }

    /// Position of local sample (0, 0, 0) in the body fixed frame.
    pub fn origin(&self) -> WorldPos {
        let extent = CHUNK_SIZE as f64 * self.voxel_size_m;
        WorldPos::body_fixed(self.key.as_ivec3().as_dvec3() * extent)
    }

    /// Sample spacing of this chunk's level in metres.
    pub fn voxel_size_m(&self) -> f64 {
        self.voxel_size_m
    }

    /// Frame position of local sample `local` (each coordinate -1 to `CHUNK_SIZE + 1`).
    ///
    /// Computed from the integer world sample index with one rounding, so every chunk that stores
    /// the sample evaluates it at the same f64 position and the apron copies stay bit identical at
    /// any voxel size (D16).
    pub fn sample_position(&self, local: IVec3) -> DVec3 {
        self.key.world_sample(local).as_dvec3() * self.voxel_size_m
    }

    /// Fills every sample, apron included, from a signed distance function of the frame position in
    /// metres (negative inside). Material is rock inside and air outside.
    pub fn fill_from(&mut self, sdf: impl Fn(DVec3) -> f64) {
        self.fill_from_with_material(|p| (sdf(p), Material::ROCK));
    }

    /// Fills every sample from a function returning the signed distance in metres and the material
    /// of the solid; samples outside the solid store [`Material::AIR`].
    pub fn fill_from_with_material(&mut self, sample: impl Fn(DVec3) -> (f64, Material)) {
        profiling::scope!("Chunk::fill_from");
        let inv_voxel = 1.0 / self.voxel_size_m;
        for z in 0..SAMPLES_PER_AXIS {
            for y in 0..SAMPLES_PER_AXIS {
                for x in 0..SAMPLES_PER_AXIS {
                    let local = IVec3::new(x as i32, y as i32, z as i32) - IVec3::ONE;
                    let (distance_m, material) = sample(self.sample_position(local));
                    let distance = Voxel::quantise_distance(distance_m * inv_voxel);
                    let material = if distance < 0 {
                        material
                    } else {
                        Material::AIR
                    };
                    self.voxels[Self::linear(x, y, z)] = Voxel { distance, material };
                }
            }
        }
    }

    /// True when every sample on `face`'s plane (apron rows included) sits at the distance clamp
    /// (about four voxels) with one sign. When the next finer level's surface is known to lie
    /// well within that distance of this level's (the sampler's `level_gap_m` under two voxels),
    /// the finer surface cannot cross such a face and its transition slab would hold no geometry
    /// (D22, D26). Skipping those slabs saves their height nodes, the dominant cost of a coarse
    /// chunk, because the faces parallel to the ground cover the whole chunk area.
    pub fn face_is_saturated(&self, face: crate::Face) -> bool {
        let plane = if face.is_high() {
            CHUNK_SIZE as isize
        } else {
            0
        };
        let at = |u: isize, v: isize| match face.axis() {
            0 => self.get(plane, u, v),
            1 => self.get(u, plane, v),
            _ => self.get(u, v, plane),
        };
        let first = at(-1, -1).distance;
        if first.unsigned_abs() < i8::MAX as u8 {
            return false;
        }
        let last = CHUNK_SIZE as isize + 1;
        (-1..=last).all(|v| (-1..=last).all(|u| at(u, v).distance == first))
    }

    /// Sets every sample to `voxel`.
    pub fn fill_uniform(&mut self, voxel: Voxel) {
        self.voxels.fill(voxel);
    }

    /// Sample at grid coordinates in -1 to `CHUNK_SIZE + 1` inclusive.
    pub fn get(&self, x: isize, y: isize, z: isize) -> Voxel {
        self.voxels[Self::linear(Self::offset(x), Self::offset(y), Self::offset(z))]
    }

    pub fn set(&mut self, x: isize, y: isize, z: isize, voxel: Voxel) {
        self.voxels[Self::linear(Self::offset(x), Self::offset(y), Self::offset(z))] = voxel;
    }

    /// All samples in x fastest, then y, then z order; index with [`Chunk::linear`].
    pub fn samples(&self) -> &[Voxel] {
        &self.voxels
    }

    pub fn samples_mut(&mut self) -> &mut [Voxel] {
        &mut self.voxels
    }

    /// The single value every sample holds, if the chunk is uniform.
    pub fn uniform_value(&self) -> Option<Voxel> {
        let first = self.voxels[0];
        self.voxels.iter().all(|v| *v == first).then_some(first)
    }

    /// True when the block holds both inside and outside samples, so meshing can produce triangles.
    pub fn has_surface(&self) -> bool {
        let mut inside = false;
        let mut outside = false;
        for v in self.voxels.iter() {
            if v.is_inside() {
                inside = true;
            } else {
                outside = true;
            }
            if inside && outside {
                return true;
            }
        }
        false
    }

    /// Index into [`Chunk::samples`] of stored coordinates 0 to `SAMPLES_PER_AXIS - 1` (stored
    /// coordinate = local coordinate + 1).
    pub fn linear(x: usize, y: usize, z: usize) -> usize {
        (z * SAMPLES_PER_AXIS + y) * SAMPLES_PER_AXIS + x
    }

    fn offset(v: isize) -> usize {
        let i = v + 1;
        debug_assert!(
            (0..SAMPLES_PER_AXIS as isize).contains(&i),
            "sample {v} outside apron"
        );
        i as usize
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
    fn voxel_u16_round_trips() {
        for v in [
            Voxel::AIR,
            Voxel::solid(Material::SAND),
            Voxel {
                distance: -3,
                material: Material::GRASS,
            },
        ] {
            assert_eq!(Voxel::from_u16(v.to_u16()), v);
        }
    }

    #[test]
    fn samples_are_spaced_by_the_voxel_size() {
        for voxel_size_m in [0.25, 1.0, 4.0] {
            let key = ChunkKey::new(3, -2, 7);
            let mut chunk = Chunk::new(key, voxel_size_m);
            let origin_y = chunk.origin().m.y;
            assert_eq!(origin_y, -2.0 * 32.0 * voxel_size_m);
            // A plane whose signed distance is the height above the chunk origin in metres.
            chunk.fill_from(|p| p.y - origin_y - 2.0 * voxel_size_m);
            assert!(chunk.get(5, 1, 5).is_inside());
            assert!(!chunk.get(5, 3, 5).is_inside());
            assert_eq!(chunk.get(0, -1, 0).distance_voxels(), -3.0);
            assert_eq!(chunk.get(0, CHUNK_SIZE as isize + 1, 0).distance, i8::MAX);
            assert!(chunk.has_surface());
            assert!(chunk.uniform_value().is_none());
        }
    }

    #[test]
    fn uniform_and_surface_detection() {
        let mut chunk = Chunk::new(ChunkKey::new(0, 0, 0), 1.0);
        assert_eq!(chunk.uniform_value(), Some(Voxel::AIR));
        assert!(!chunk.has_surface());
        chunk.fill_uniform(Voxel::solid(Material::ROCK));
        assert_eq!(chunk.uniform_value(), Some(Voxel::solid(Material::ROCK)));
        assert!(!chunk.has_surface());
        chunk.set(4, 4, 4, Voxel::AIR);
        assert!(chunk.has_surface());
    }
}

#[cfg(test)]
mod face_saturation_tests {
    use super::*;
    use crate::{ChunkKey, Face};

    #[test]
    fn only_faces_entirely_at_the_clamp_are_saturated() {
        let mut chunk = Chunk::new(ChunkKey::new(0, 0, 0), 1.0);
        // A plane at y = 10 m: the top face (y = 32) is far above, the bottom face (y = 0) far below.
        chunk.fill_from(|p| p.y - 10.0);
        assert!(chunk.face_is_saturated(Face::HighY));
        assert!(chunk.face_is_saturated(Face::LowY));
        assert!(
            !chunk.face_is_saturated(Face::LowX),
            "the surface crosses the side faces"
        );
        chunk.fill_from(|p| p.y - 1.5);
        assert!(
            !chunk.face_is_saturated(Face::LowY),
            "within the band of the low face"
        );
    }
}
