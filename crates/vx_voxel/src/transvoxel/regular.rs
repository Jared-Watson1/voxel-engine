use transvoxel_data::regular_cell_data::{
    REGULAR_CELL_CLASS, REGULAR_CELL_DATA, REGULAR_VERTEX_DATA,
};

use super::{CellSink, emit};

/// Polygonises one regular cell whose case has bit `i` set when corner `i` is inside the solid.
/// Vertices are requested in table order with the lower corner first (the order the crate's
/// `extract_regular_cell` creates them, which fixes the vertex order of a chunk mesh), then the
/// triangles in table order.
pub(crate) fn regular_cell(case: u8, sink: &mut impl CellSink) {
    let case = usize::from(case);
    let data = &REGULAR_CELL_DATA[usize::from(REGULAR_CELL_CLASS[case])];
    let vertex_count = usize::from(data.get_vertex_count());
    let mut vertices = [0u32; 12];
    for (slot, &code) in vertices
        .iter_mut()
        .zip(&REGULAR_VERTEX_DATA[case])
        .take(vertex_count)
    {
        *slot = sink.vertex(usize::from((code >> 4) & 0xF), usize::from(code & 0xF));
    }
    let triangles = usize::from(data.get_triangle_count());
    for t in data.vertex_index[..3 * triangles].chunks_exact(3) {
        emit(
            sink,
            [
                vertices[usize::from(t[0])],
                vertices[usize::from(t[1])],
                vertices[usize::from(t[2])],
            ],
        );
    }
}
