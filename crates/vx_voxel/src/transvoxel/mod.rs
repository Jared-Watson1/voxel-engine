//! Transvoxel cell polygonisation (Lengyel 2010, "Voxel-Based Terrain for Real-Time Virtual
//! Simulations", chapter 4), vendored for D22.
//!
//! Derived from the `transvoxel` crate 2.0.0 by Seb E. (<https://github.com/Gnurfos/transvoxel_rs>),
//! licensed MIT OR Apache-2.0: the regular and transition cell walks of
//! `src/implementation/algorithm.rs`, the case bit layout and grid point numbering of
//! `src/implementation/aux_tables.rs` and the per face frames of `ROTATIONS`. The cell tables come
//! from the `transvoxel-data` 0.2.1 dependency (MIT, Lengyel's tables). Changed from the crate:
//! regular cells are never shrunk, so a mesh does not depend on its neighbours' levels; positions,
//! gradients, materials, vertex sharing and output ranges belong to the caller through
//! [`CellSink`] (`crate::mesh`); the crate's vertex reuse bits are replaced by edge keyed tables in
//! the caller.
//!
//! Corner numbering of a regular cell: bit 0 is +x, bit 1 is +y, bit 2 is +z. Grid points of a
//! transition cell: 0 to 8 on the full resolution face at `u + 3 v` (u and v in half cells, 0 to
//! 2), 9 to 12 on the half resolution face at `(0, 0), (1, 0), (0, 1), (1, 1)` in cells.

mod regular;
mod transition;

pub(crate) use regular::regular_cell;
pub(crate) use transition::{FACE_FRAMES, FaceFrame, transition_case, transition_cell};

/// Receives the vertices and triangles of one cell.
pub(crate) trait CellSink {
    /// Index of the vertex on the edge between corners (or grid points) `a` and `b`. The two lie
    /// on opposite sides of the surface.
    fn vertex(&mut self, a: usize, b: usize) -> u32;

    /// A triangle wound counter clockwise seen from outside the solid.
    fn triangle(&mut self, indices: [u32; 3]);
}

/// Emits a triangle unless two of its indices coincide. Bridge cells (`crate::mesh`) map several
/// edges to one vertex; regular and transition cells never do.
fn emit(sink: &mut impl CellSink, [a, b, c]: [u32; 3]) {
    if a != b && b != c && a != c {
        sink.triangle([a, b, c]);
    }
}

#[cfg(test)]
mod tests {
    use transvoxel_data::regular_cell_data::{
        REGULAR_CELL_CLASS, REGULAR_CELL_DATA, REGULAR_VERTEX_DATA,
    };
    use transvoxel_data::transition_cell_data::{
        TRANSITION_CELL_CLASS, TRANSITION_CELL_DATA, TRANSITION_VERTEX_DATA,
    };

    #[test]
    fn regular_edges_list_the_lower_corner_first() {
        for case in 0..256 {
            let class = REGULAR_CELL_CLASS[case] as usize;
            let count = REGULAR_CELL_DATA[class].get_vertex_count() as usize;
            for &code in &REGULAR_VERTEX_DATA[case][..count] {
                let (a, b) = (((code >> 4) & 0xF) as usize, (code & 0xF) as usize);
                let differ = a ^ b;
                assert!(
                    differ.is_power_of_two(),
                    "case {case}: {a}-{b} is not an edge"
                );
                assert_eq!(
                    a & differ,
                    0,
                    "case {case}: {a}-{b} lists the upper corner first"
                );
            }
        }
    }

    /// Grid point position in half cells: (u, v, face) with face 0 the full resolution face.
    fn grid_point(i: usize) -> (i32, i32, i32) {
        if i < 9 {
            ((i % 3) as i32, (i / 3) as i32, 0)
        } else {
            let j = i as i32 - 9;
            (2 * (j % 2), 2 * (j / 2), 1)
        }
    }

    /// Every transition vertex joins two samples of one face, one step apart on that face's grid:
    /// no vertex lies between the full and the half resolution faces, whose corner samples are
    /// equal in Lengyel's construction (D22).
    #[test]
    fn transition_edges_stay_on_one_face() {
        for case in 0..512 {
            let class = (TRANSITION_CELL_CLASS[case] & 0x7F) as usize;
            let count = TRANSITION_CELL_DATA[class].get_vertex_count() as usize;
            for &code in &TRANSITION_VERTEX_DATA[case][..count] {
                let (a, b) = (((code >> 4) & 0xF) as usize, (code & 0xF) as usize);
                let (pa, pb) = (grid_point(a), grid_point(b));
                assert_eq!(pa.2, pb.2, "case {case}: {a}-{b} crosses between the faces");
                let step = if pa.2 == 0 { 1 } else { 2 };
                let distance = (pa.0 - pb.0).abs() + (pa.1 - pb.1).abs();
                assert_eq!(distance, step, "case {case}: {a}-{b} is not a grid edge");
            }
        }
    }
}
