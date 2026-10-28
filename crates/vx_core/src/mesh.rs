//! Packed terrain vertex (D14) shared by the mesher (`vx_voxel`) and the renderer (`vx_render`).
//!
//! Sixteen bytes in four u32 words, matching `struct PackedVertex` in
//! `assets/shaders/terrain.wgsl` (engineering rule 9):
//!
//! - word 0: position x in bits 0 to 15, position y in bits 16 to 31, both u16 in units of
//!   1/[`POSITION_UNITS_PER_VOXEL`] voxel, chunk local (0 at the chunk origin, 32768 at the far face).
//! - word 1: position z in bits 0 to 15, octahedral normal x as snorm8 in bits 16 to 23 and y in
//!   bits 24 to 31.
//! - word 2: four material slots. Slot `i` has its material id in bits `4i` to `4i + 3` and its
//!   weight in bits `16 + 4i` to `16 + 4i + 3`. Ids are 0 to 15, weights 0 to 15; the shader
//!   normalises the weights. Slot 0 is the dominant material.
//! - word 3: chunk slot (index into the GPU chunk table, patched by the renderer at upload) in bits
//!   0 to 15; flags in bits 16 to 31 ([`FLAG_TRANSITION_FULL_RES`] in bit 16, the rest zero).
//!
//! Positions are quantised in voxel units so the layout is independent of the voxel size; the
//! vertex shader multiplies by the chunk's own voxel size from the chunk table (D23).
//!
//! Transition faces (D22): a chunk's regular cells are meshed unshrunk, and the transition cells
//! of each face are separate index ranges. The vertex shader moves vertices to Lengyel's
//! secondary position for the faces in the chunk's transition mask; [`secondary_position_voxels`]
//! is the reference the WGSL mirrors operation for operation.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

/// Position quantisation: 1024 units per voxel, so a 32 voxel chunk spans 0 to 32768 of the
/// u16 range and the resolution is just under 1 mm at the 1 m design default.
pub const POSITION_UNITS_PER_VOXEL: f32 = 1024.0;
/// Largest material id the vertex can carry (4 bits).
pub const MAX_MATERIAL_ID: u8 = 15;
/// Largest material weight the vertex can carry (4 bits).
pub const MAX_MATERIAL_WEIGHT: u8 = 15;
/// Material slots per vertex.
pub const MATERIAL_SLOTS: usize = 4;
/// Word 3 flag: a transition cell vertex on the full resolution face, which never moves (D22).
pub const FLAG_TRANSITION_FULL_RES: u16 = 1;
/// Width of a transition cell as a fraction of the coarse cell (Lengyel's `w`, D22).
pub const TRANSITION_WIDTH_CELLS: f32 = 0.5;
/// Chunk extent in voxels as the float the displacement compares against.
const CHUNK_VOXELS: f32 = crate::config::CHUNK_SIZE as f32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct PackedVertex {
    pub words: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<PackedVertex>() == 16);
const _: () = assert!(std::mem::align_of::<PackedVertex>() == 4);

/// Unpacked form of a vertex, used by the mesher to build one and by tests to read one back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexAttributes {
    /// Chunk local position in voxels, 0.0 to 32.0 along each axis.
    pub position_voxels: Vec3,
    /// Unit normal pointing out of the solid.
    pub normal: Vec3,
    /// `(material id, weight)` per slot; slot 0 is dominant. Unused slots are `(0, 0)`.
    pub materials: [(u8, u8); MATERIAL_SLOTS],
}

impl PackedVertex {
    /// Quantises the attributes. Positions outside 0 to 63.99 voxels and ids or weights above 15
    /// saturate; the mesher never produces them.
    pub fn pack(attributes: &VertexAttributes) -> Self {
        let q = |v: f32| (v * POSITION_UNITS_PER_VOXEL).round().clamp(0.0, 65535.0) as u32;
        let p = attributes.position_voxels;
        let [nx, ny] = encode_octahedral(attributes.normal);
        let mut materials = 0u32;
        for (i, &(id, weight)) in attributes.materials.iter().enumerate() {
            let id = u32::from(id.min(MAX_MATERIAL_ID));
            let weight = u32::from(weight.min(MAX_MATERIAL_WEIGHT));
            materials |= id << (4 * i);
            materials |= weight << (16 + 4 * i);
        }
        Self {
            words: [
                q(p.x) | (q(p.y) << 16),
                q(p.z) | ((nx as u8 as u32) << 16) | ((ny as u8 as u32) << 24),
                materials,
                0,
            ],
        }
    }

    pub fn unpack(self) -> VertexAttributes {
        let [w0, w1, w2, _] = self.words;
        let d = |u: u32| (u & 0xFFFF) as f32 / POSITION_UNITS_PER_VOXEL;
        let nx = ((w1 >> 16) & 0xFF) as u8 as i8;
        let ny = ((w1 >> 24) & 0xFF) as u8 as i8;
        let mut materials = [(0u8, 0u8); MATERIAL_SLOTS];
        for (i, slot) in materials.iter_mut().enumerate() {
            *slot = (
                ((w2 >> (4 * i)) & 0xF) as u8,
                ((w2 >> (16 + 4 * i)) & 0xF) as u8,
            );
        }
        VertexAttributes {
            position_voxels: Vec3::new(d(w0), d(w0 >> 16), d(w1)),
            normal: decode_octahedral([nx, ny]),
            materials,
        }
    }

    pub fn chunk_slot(self) -> u16 {
        (self.words[3] & 0xFFFF) as u16
    }

    /// Returns the vertex with its chunk slot set and the flags untouched.
    pub fn with_chunk_slot(mut self, slot: u16) -> Self {
        self.words[3] = (self.words[3] & 0xFFFF_0000) | u32::from(slot);
        self
    }

    pub fn flags(self) -> u16 {
        (self.words[3] >> 16) as u16
    }

    /// Returns the vertex with its flags set and the chunk slot untouched.
    pub fn with_flags(mut self, flags: u16) -> Self {
        self.words[3] = (self.words[3] & 0xFFFF) | (u32::from(flags) << 16);
        self
    }
}

/// Lengyel's secondary vertex position (dissertation section 4.3) for a chunk whose transition
/// `mask` has bit `2 * axis` set when the neighbour across the low face of `axis` is drawn one level
/// finer and bit `2 * axis + 1` for the high face (D22). `position_voxels` is chunk local (0 to 32)
/// and `normal` the unit vertex normal.
///
/// Within one cell of a masked face the vertex moves inward by `(1 - d) * w` cells, `d` its
/// distance from that face in cells, and the move is projected onto the tangent plane of the
/// normal. A vertex lying exactly on a face whose bit is clear stays put (godot_voxel's rule, so a
/// same level neighbour keeps a bit identical shared face), and so does a vertex flagged
/// [`FLAG_TRANSITION_FULL_RES`]. Positions compare exactly against 0 and 32, which the packed
/// format represents exactly.
///
/// The WGSL in `assets/shaders/terrain.wgsl` must perform the same operations in the same order
/// so the coarse chunk's half resolution face and its transition cells coincide bit for bit.
pub fn secondary_position_voxels(
    position_voxels: Vec3,
    normal: Vec3,
    flags: u16,
    mask: u8,
) -> Vec3 {
    if mask == 0 || flags & FLAG_TRANSITION_FULL_RES != 0 {
        return position_voxels;
    }
    let p = position_voxels.to_array();
    let mut delta = [0.0f32; 3];
    for axis in 0..3 {
        let low = mask & (1 << (2 * axis)) != 0;
        let high = mask & (1 << (2 * axis + 1)) != 0;
        if (p[axis] == 0.0 && !low) || (p[axis] == CHUNK_VOXELS && !high) {
            return position_voxels;
        }
        if low && p[axis] < 1.0 {
            delta[axis] = (1.0 - p[axis]) * TRANSITION_WIDTH_CELLS;
        } else if high && p[axis] > CHUNK_VOXELS - 1.0 {
            delta[axis] = (CHUNK_VOXELS - 1.0 - p[axis]) * TRANSITION_WIDTH_CELLS;
        }
    }
    let delta = Vec3::from_array(delta);
    if delta == Vec3::ZERO {
        return position_voxels;
    }
    position_voxels + (delta - normal * normal.dot(delta))
}

/// Octahedral normal encoding (Cigolle et al. 2014, "A Survey of Efficient Representations for
/// Independent Unit Vectors"): project onto the octahedron, fold the lower hemisphere outward,
/// quantise each coordinate to snorm8. The decoder in the shader must use the same formulas.
pub fn encode_octahedral(n: Vec3) -> [i8; 2] {
    let n = n.normalize_or(Vec3::Y);
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    let mut p = Vec2::new(n.x, n.y) / l1;
    if n.z < 0.0 {
        p = octahedral_wrap(p);
    }
    let q = |v: f32| (v.clamp(-1.0, 1.0) * 127.0).round() as i8;
    [q(p.x), q(p.y)]
}

pub fn decode_octahedral(e: [i8; 2]) -> Vec3 {
    let p = Vec2::new(f32::from(e[0]), f32::from(e[1])) / 127.0;
    let z = 1.0 - p.x.abs() - p.y.abs();
    let mut xy = p;
    if z < 0.0 {
        xy = octahedral_wrap(p);
    }
    Vec3::new(xy.x, xy.y, z).normalize_or(Vec3::Y)
}

fn octahedral_wrap(v: Vec2) -> Vec2 {
    let sign = |x: f32| if x >= 0.0 { 1.0 } else { -1.0 };
    Vec2::new((1.0 - v.y.abs()) * sign(v.x), (1.0 - v.x.abs()) * sign(v.y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_round_trips_within_half_a_unit() {
        for p in [
            Vec3::ZERO,
            Vec3::splat(32.0),
            Vec3::new(0.001, 15.4999, 31.9999),
            Vec3::new(12.3456, 0.5, 20.0),
        ] {
            let v = PackedVertex::pack(&VertexAttributes {
                position_voxels: p,
                normal: Vec3::Y,
                materials: [(0, 0); 4],
            });
            let back = v.unpack().position_voxels;
            assert!(
                (back - p).abs().max_element() <= 0.5 / POSITION_UNITS_PER_VOXEL + 1.0e-6,
                "{p} became {back}"
            );
        }
    }

    #[test]
    fn normal_round_trips_within_two_degrees() {
        let mut worst = 0.0f32;
        for i in 0..200 {
            for j in 0..100 {
                let theta = i as f32 / 200.0 * std::f32::consts::TAU;
                let phi = (j as f32 / 100.0 - 0.5) * std::f32::consts::PI;
                let n = Vec3::new(phi.cos() * theta.cos(), phi.sin(), phi.cos() * theta.sin());
                let back = decode_octahedral(encode_octahedral(n));
                worst = worst.max(n.dot(back).clamp(-1.0, 1.0).acos().to_degrees());
            }
        }
        assert!(worst < 2.0, "worst normal error {worst} degrees");
    }

    #[test]
    fn materials_and_slot_are_exact() {
        let attributes = VertexAttributes {
            position_voxels: Vec3::splat(1.0),
            normal: Vec3::X,
            materials: [(1, 15), (4, 7), (15, 1), (3, 0)],
        };
        let v = PackedVertex::pack(&attributes).with_chunk_slot(0xBEEF);
        assert_eq!(v.unpack().materials, attributes.materials);
        assert_eq!(v.chunk_slot(), 0xBEEF);
        assert_eq!(v.flags(), 0, "flags start zero");
        assert_eq!(v.with_chunk_slot(7).chunk_slot(), 7);
        let flagged = v.with_flags(FLAG_TRANSITION_FULL_RES);
        assert_eq!(flagged.flags(), FLAG_TRANSITION_FULL_RES);
        assert_eq!(flagged.chunk_slot(), 0xBEEF);
        assert_eq!(flagged.with_chunk_slot(3).flags(), FLAG_TRANSITION_FULL_RES);
        let saturated = PackedVertex::pack(&VertexAttributes {
            materials: [(200, 200); 4],
            ..attributes
        });
        assert_eq!(saturated.unpack().materials, [(15, 15); 4]);
    }

    #[test]
    fn secondary_position_moves_only_near_masked_faces() {
        let up = Vec3::Y;
        let low_x = 1u8;
        // Not near any masked face, or no mask: unchanged.
        let p = Vec3::new(5.0, 7.0, 9.0);
        assert_eq!(secondary_position_voxels(p, up, 0, low_x), p);
        assert_eq!(
            secondary_position_voxels(Vec3::new(0.25, 7.0, 9.0), up, 0, 0),
            Vec3::new(0.25, 7.0, 9.0)
        );
        // On the masked face with a normal in the tangent plane: moves the full width inward.
        let moved = secondary_position_voxels(Vec3::new(0.0, 7.0, 9.0), up, 0, low_x);
        assert_eq!(moved, Vec3::new(TRANSITION_WIDTH_CELLS, 7.0, 9.0));
        // Half a cell in: half the width.
        let moved = secondary_position_voxels(Vec3::new(0.5, 7.0, 9.0), up, 0, low_x);
        assert_eq!(moved.x, 0.5 + 0.5 * TRANSITION_WIDTH_CELLS);
        // A normal along the axis projects the move away entirely.
        let moved = secondary_position_voxels(Vec3::new(0.0, 7.0, 9.0), Vec3::X, 0, low_x);
        assert_eq!(moved, Vec3::new(0.0, 7.0, 9.0));
        // The full resolution flag pins the vertex.
        let pinned = secondary_position_voxels(
            Vec3::new(0.0, 7.0, 9.0),
            up,
            FLAG_TRANSITION_FULL_RES,
            low_x,
        );
        assert_eq!(pinned, Vec3::new(0.0, 7.0, 9.0));
        // A vertex on a face without a transition stays put even near a masked face.
        let corner = Vec3::new(0.25, 0.0, 9.0);
        assert_eq!(secondary_position_voxels(corner, Vec3::Z, 0, low_x), corner);
        // High face moves inward (negative).
        let high_x = 2u8;
        let moved = secondary_position_voxels(Vec3::new(32.0, 7.0, 9.0), up, 0, high_x);
        assert_eq!(moved.x, 32.0 - TRANSITION_WIDTH_CELLS);
    }

    #[test]
    fn is_sixteen_bytes_of_plain_data() {
        let v = PackedVertex {
            words: [1, 2, 3, 4],
        };
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 16);
        assert_eq!(bytes[0..4], 1u32.to_ne_bytes());
    }
}
