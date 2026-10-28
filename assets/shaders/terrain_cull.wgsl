// Culling and index compaction for the pooled terrain (D10, D19, D22, D23). Pass `cull` tests one
// chunk per thread against the frustum and the planet's horizon and appends the visible ones to
// a list; pass `compact` copies each visible chunk's regular indices and the transition ranges of
// its masked faces, rebased by its vertex base, into one contiguous index buffer that a single
// indexed indirect draw consumes. Units are metres; positions are relative to the camera.
//
// `Frame` and `ChunkEntry` repeat the definitions in `terrain.wgsl`; `Counters` matches
// `Counters` in `crates/vx_render/src/cull.rs` (engineering rule 9).

struct Frame {
    view_proj: mat4x4<f32>,
    planes: array<vec4<f32>, 6>,
    // xyz: camera position relative to the render origin. w: material tile length.
    camera_offset_m: vec4<f32>,
    // xyz: planet centre relative to the camera. w: occluder radius (0 disables the horizon test).
    planet: vec4<f32>,
    slot_count: u32,
    zero: u32,
    pad0: u32,
    pad1: u32,
};

struct ChunkEntry {
    offset_m: vec3<f32>,
    vertex_base: u32,
    texture_phase_m: vec3<f32>,
    index_base: u32,
    regular_count: u32,
    // Face 2k in the low 16 bits of element k, face 2k + 1 in the high 16 bits.
    transition_counts: array<u32, 3>,
    voxel_size_m: f32,
    // Bit 0: visible. Bits 8 to 13: transition mask, bit `8 + face`.
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
const FLAG_VISIBLE: u32 = 1u;
const CHUNK_MASK_SHIFT: u32 = 8u;
const CHUNK_VOXELS: f32 = 32.0;
// Relative slack on both horizon inequalities so f32 rounding never culls a chunk on the limb.
const HORIZON_MARGIN: f32 = 1.0e-4;

fn transition_count(chunk: ChunkEntry, face: u32) -> u32 {
    return (chunk.transition_counts[face >> 1u] >> (16u * (face & 1u))) & 0xffffu;
}

fn chunk_mask(chunk: ChunkEntry) -> u32 {
    return (chunk.flags >> CHUNK_MASK_SHIFT) & 0x3fu;
}

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

// Horizon culling against the occluder sphere (centre c, radius r) seen from the camera at the
// origin (Cozzi, "Horizon culling", cesium.com 2013). With h^2 = |c|^2 - r^2 a point p is behind
// the sphere when it lies past the horizon plane, p.c > h^2, and inside the tangent cone,
// (p.c)^2 > h^2 |p|^2. The hidden region is the intersection of a convex cone and a half space,
// so a box is hidden when all eight corners are. Vectors are scaled by 1/|c| so the squares stay
// near 1 at any distance.
fn horizon_hidden(min_m: vec3<f32>, max_m: vec3<f32>) -> bool {
    let r = frame.planet.w;
    let d = length(frame.planet.xyz);
    if r <= 0.0 || d <= r {
        return false;
    }
    let inv_d = 1.0 / d;
    let c = frame.planet.xyz * inv_d;
    let rs = r * inv_d;
    let h2 = (1.0 - rs) * (1.0 + rs);
    let plane = h2 * (1.0 + HORIZON_MARGIN);
    for (var i = 0u; i < 8u; i++) {
        let pick = vec3<bool>((i & 1u) != 0u, (i & 2u) != 0u, (i & 4u) != 0u);
        let p = select(min_m, max_m, pick) * inv_d;
        let pc = dot(p, c);
        if pc <= plane || pc * pc <= plane * dot(p, p) {
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
    if (chunk.flags & FLAG_VISIBLE) == 0u {
        return;
    }
    let mask = chunk_mask(chunk);
    var count = chunk.regular_count;
    for (var face = 0u; face < 6u; face++) {
        if (mask & (1u << face)) != 0u {
            count += transition_count(chunk, face);
        }
    }
    if count == 0u {
        return;
    }
    let min_m = chunk.offset_m - frame.camera_offset_m.xyz;
    let max_m = min_m + vec3<f32>(CHUNK_VOXELS * chunk.voxel_size_m);
    if !aabb_inside(min_m, max_m) || horizon_hidden(min_m, max_m) {
        return;
    }
    let dst = atomicAdd(&counters.index_count, count);
    let v = atomicAdd(&counters.group_count_x, 1u);
    visible[v] = Visible(slot, dst);
}

fn copy_range(src: u32, dst: u32, count: u32, vertex_base: u32, lane: u32) {
    for (var i = lane; i < count; i += COMPACT_WORKGROUP_SIZE) {
        compacted[dst + i] = index_pool[src + i] + vertex_base;
    }
}

// One workgroup per visible chunk; the threads stride over the regular range, then over each
// masked face's transition range. In the pool the ranges follow each other in face order.
@compute @workgroup_size(COMPACT_WORKGROUP_SIZE)
fn compact(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let entry = visible[group.x];
    let chunk = chunks[entry.slot];
    let mask = chunk_mask(chunk);
    copy_range(chunk.index_base, entry.dst, chunk.regular_count, chunk.vertex_base, lane);
    var src = chunk.index_base + chunk.regular_count;
    var dst = entry.dst + chunk.regular_count;
    for (var face = 0u; face < 6u; face++) {
        let count = transition_count(chunk, face);
        if (mask & (1u << face)) != 0u {
            copy_range(src, dst, count, chunk.vertex_base, lane);
            dst += count;
        }
        src += count;
    }
}
