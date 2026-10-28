use transvoxel_data::transition_cell_data::{
    TRANSITION_CELL_CLASS, TRANSITION_CELL_DATA, TRANSITION_VERTEX_DATA,
};

use super::{CellSink, emit};

/// A face's transition cell frame (the crate's `ROTATIONS`): `u` and `v` span the face and
/// `u x v` points into the chunk for every face (the crate's `w`), so one winding rule serves all
/// six. `base` is the corner the frame starts from, in units of the chunk extent.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FaceFrame {
    pub base: [i32; 3],
    pub u: [i32; 3],
    pub v: [i32; 3],
}

const fn frame(base: [i32; 3], u: [i32; 3], v: [i32; 3]) -> FaceFrame {
    FaceFrame { base, u, v }
}

/// In [`crate::Face`] order: LowX, HighX, LowY, HighY, LowZ, HighZ.
#[rustfmt::skip]
pub(crate) const FACE_FRAMES: [FaceFrame; 6] = [
    frame([0, 0, 1], [0, 0, -1], [0, 1, 0]),
    frame([1, 0, 0], [0, 0, 1], [0, 1, 0]),
    frame([0, 0, 1], [1, 0, 0], [0, 0, -1]),
    frame([0, 1, 0], [1, 0, 0], [0, 0, 1]),
    frame([0, 0, 0], [1, 0, 0], [0, 1, 0]),
    frame([1, 0, 1], [-1, 0, 0], [0, 1, 0]),
];

impl FaceFrame {
    /// Chunk local grid coordinates of the point `(u, v)` on the face, for a grid of `extent`
    /// steps per chunk edge.
    pub fn point(&self, extent: i32, u: i32, v: i32) -> [i32; 3] {
        std::array::from_fn(|i| self.base[i] * extent + self.u[i] * u + self.v[i] * v)
    }
}

/// Lengyel's case bit of each full resolution sample, indexed `u + 3 v`.
const CASE_BITS: [u16; 9] = [0x01, 0x02, 0x04, 0x80, 0x100, 0x08, 0x40, 0x20, 0x10];

/// The 9 bit transition case from the inside flags of the full resolution samples (`u + 3 v`).
pub(crate) fn transition_case(inside: [bool; 9]) -> u16 {
    inside
        .iter()
        .zip(CASE_BITS)
        .filter(|(inside, _)| **inside)
        .map(|(_, bit)| bit)
        .sum()
}

/// Polygonises one transition cell. The half resolution grid points (9 to 12) carry the inside
/// state of the full resolution corners (0, 2, 6, 8) in Lengyel's construction, so only the nine
/// full resolution samples choose the case.
pub(crate) fn transition_cell(case: u16, sink: &mut impl CellSink) {
    let case = usize::from(case);
    let raw_class = TRANSITION_CELL_CLASS[case];
    let data = &TRANSITION_CELL_DATA[usize::from(raw_class & 0x7F)];
    let vertex_count = usize::from(data.get_vertex_count());
    let mut vertices = [0u32; 12];
    for (slot, &code) in vertices
        .iter_mut()
        .zip(&TRANSITION_VERTEX_DATA[case])
        .take(vertex_count)
    {
        *slot = sink.vertex(usize::from((code >> 4) & 0xF), usize::from(code & 0xF));
    }
    // The tables wind for Lengyel's frame; the crate's frames (w inward) flip it, and the high bit
    // of the class marks cases reached through the inverse state, which flips it back.
    let flip = raw_class & 0x80 != 0;
    let triangles = usize::from(data.get_triangle_count());
    for t in data.vertex_index[..3 * triangles].chunks_exact(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| vertices[usize::from(i)]);
        emit(sink, if flip { [c, b, a] } else { [a, b, c] });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_right_handed_with_w_inward() {
        for (face, f) in FACE_FRAMES.iter().enumerate() {
            let [u, v] = [f.u, f.v].map(glam::IVec3::from_array);
            let axis = face / 2;
            let mut inward = glam::IVec3::ZERO;
            inward[axis] = if face % 2 == 0 { 1 } else { -1 };
            assert_eq!(u.cross(v), inward, "face {face}");
            // The frame covers the face: its four corners lie on the face plane, in range.
            for (cu, cv) in [(0, 0), (32, 0), (0, 32), (32, 32)] {
                let p = f.point(32, cu, cv);
                assert_eq!(p[axis], if face % 2 == 0 { 0 } else { 32 }, "face {face}");
                assert!(p.iter().all(|c| (0..=32).contains(c)), "face {face}: {p:?}");
            }
        }
    }
}
