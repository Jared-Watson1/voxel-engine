// Frustum culling and index compaction for the pooled terrain (D10, D19). Pass `cull` tests one
// chunk per thread and appends the visible ones to a list; pass `compact` copies each visible
// chunk's indices, rebased by its vertex base, into one contiguous index buffer that a single
// indexed indirect draw consumes. Units are metres.
//
// `Frame` and `ChunkEntry` repeat the definitions in `terrain.wgsl`; `Counters` matches
// `Counters` in `crates/vx_render/src/cull.rs` (engineering rule 9).

struct Frame {
    view_proj: mat4x4<f32>,
    planes: array<vec4<f32>, 6>,
    sun_dir: vec4<f32>,
    params: vec4<f32>,
    slot_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct ChunkEntry {
    offset_m: vec3<f32>,
    vertex_base: u32,
    texture_phase_m: vec3<f32>,
    index_base: u32,
    index_count: u32,
    flags: u32,
    pad0: u32,
    pad1: u32,
};

// Indexed indirect draw arguments followed by the indirect dispatch arguments of the compact
// pass, so one buffer feeds both and one copy reads both back.
struct Counters {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
    group_count_x: atomic<u32>,
    group_count_y: u32,
    group_count_z: u32,
};

struct Visible {
    slot: u32,
    // First index of the chunk in the compacted buffer.
    dst: u32,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> chunks: array<ChunkEntry>;
@group(0) @binding(2) var<storage, read_write> visible: array<Visible>;
@group(1) @binding(0) var<storage, read_write> counters: Counters;
@group(2) @binding(0) var<storage, read> index_pool: array<u32>;
@group(2) @binding(1) var<storage, read_write> compacted: array<u32>;

const CULL_WORKGROUP_SIZE: u32 = 64u;
const COMPACT_WORKGROUP_SIZE: u32 = 256u;
const FLAG_RESIDENT: u32 = 1u;

// Positive vertex test: the corner furthest along the plane normal is outside only when the
// whole box is. A box straddling a plane passes.
fn aabb_inside(min_m: vec3<f32>, max_m: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = frame.planes[i];
        let corner = select(min_m, max_m, plane.xyz > vec3<f32>(0.0));
        if dot(plane.xyz, corner) + plane.w < 0.0 {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(CULL_WORKGROUP_SIZE)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= frame.slot_count {
        return;
    }
    let chunk = chunks[slot];
    if (chunk.flags & FLAG_RESIDENT) == 0u || chunk.index_count == 0u {
        return;
    }
    if !aabb_inside(chunk.offset_m, chunk.offset_m + vec3<f32>(frame.params.y)) {
        return;
    }
    let dst = atomicAdd(&counters.index_count, chunk.index_count);
    let v = atomicAdd(&counters.group_count_x, 1u);
    visible[v] = Visible(slot, dst);
}

// One workgroup per visible chunk; the threads stride over its indices.
@compute @workgroup_size(COMPACT_WORKGROUP_SIZE)
fn compact(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let entry = visible[group.x];
    let chunk = chunks[entry.slot];
    for (var i = lane; i < chunk.index_count; i += COMPACT_WORKGROUP_SIZE) {
        compacted[entry.dst + i] = index_pool[chunk.index_base + i] + chunk.vertex_base;
    }
}
