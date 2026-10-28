// Pooled terrain with vertex pulling and triplanar materials (D10, D14, D19) into the G-buffer
// (D25), with per chunk scale, placements relative to the render origin (D23) and Lengyel's
// secondary positions on faces next to a finer level (D22). Vertices arrive packed (vx_core::mesh)
// and chunk local; the chunk table supplies each chunk's offset from the render origin and the
// frame uniform the camera's, so no world position ever reaches this shader (D4). Units are
// metres unless a name says voxels. Lighting happens in lighting.wgsl.
//
// The struct layouts match `FrameUniform`, `PackedVertex` and `ChunkEntry` in
// `crates/vx_render/src/terrain.rs` (engineering rule 9) and are repeated in `terrain_cull.wgsl`.

struct Frame {
    // Projection times the rotation only view matrix (reverse-Z, infinite far plane).
    view_proj: mat4x4<f32>,
    // Frustum planes in the camera relative frame as (normal, distance); inside is positive.
    planes: array<vec4<f32>, 6>,
    // xyz: camera position relative to the render origin. w: material tile length.
    camera_offset_m: vec4<f32>,
    // xyz: planet centre relative to the camera. w: occluder radius (0 disables the horizon test).
    planet: vec4<f32>,
    // Entries in the chunk table this frame.
    slot_count: u32,
    // Always 0; see `rounded`.
    zero: u32,
    pad0: u32,
    pad1: u32,
};

// Four u32 words, bit layout documented in vx_core::mesh.
struct PackedVertex {
    words: vec4<u32>,
};

// 64 bytes (D23).
struct ChunkEntry {
    // Chunk origin relative to the render origin.
    offset_m: vec3<f32>,
    vertex_base: u32,
    // Chunk origin modulo the material tile length.
    texture_phase_m: vec3<f32>,
    index_base: u32,
    regular_count: u32,
    // Index counts of the six transition ranges as u16 pairs: face 2k in the low half of
    // element k, face 2k + 1 in the high half.
    transition_counts: array<u32, 3>,
    voxel_size_m: f32,
    // Bit 0: visible. Bits 8 to 13: transition mask, bit `8 + face`.
    flags: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> vertices: array<PackedVertex>;
@group(0) @binding(2) var<storage, read> chunks: array<ChunkEntry>;
@group(1) @binding(0) var material_albedo: texture_2d_array<f32>;
@group(1) @binding(1) var material_sampler: sampler;

const TRIPLANAR_SHARPNESS: f32 = 4.0;
const CHUNK_MASK_SHIFT: u32 = 8u;

// BEGIN vertex decoding and secondary position
// Everything between the BEGIN and END markers is also compiled on its own by the parity test in
// `crates/vx_render/src/terrain.rs`, which checks it bit for bit against
// `vx_core::mesh::secondary_position_voxels`.

const POSITION_UNITS_PER_VOXEL: f32 = 1024.0;
// vx_core::mesh::FLAG_TRANSITION_FULL_RES, in the high half of word 3.
const FLAG_TRANSITION_FULL_RES: u32 = 1u;
// vx_core::mesh::TRANSITION_WIDTH_CELLS.
const TRANSITION_WIDTH_CELLS: f32 = 0.5;
// CHUNK_SIZE in voxels.
const CHUNK_VOXELS: f32 = 32.0;

// Chunk local position in voxels. Exact: a u16 divided by a power of two.
fn unpack_position_voxels(v: vec4<u32>) -> vec3<f32> {
    let units = vec3<f32>(f32(v.x & 0xffffu), f32(v.x >> 16u), f32(v.y & 0xffffu));
    return units / POSITION_UNITS_PER_VOXEL;
}

// Byte `shift` to `shift + 7` of `word` as a two's complement value in -127 to 127, scaled to
// -1 to 1. The arithmetic right shift sign extends, which is what `i8 as f32` does on the CPU.
fn snorm8(word: u32, shift: u32) -> f32 {
    return f32(bitcast<i32>(word << (24u - shift)) >> 24u) / 127.0;
}

// Mirrors vx_core::mesh::decode_octahedral, including the wrap of the lower hemisphere.
fn octahedral_wrap(v: vec2<f32>) -> vec2<f32> {
    let sign = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
    return (1.0 - abs(v.yx)) * sign;
}

fn unpack_normal(v: vec4<u32>) -> vec3<f32> {
    let e = vec2<f32>(snorm8(v.y, 16u), snorm8(v.y, 24u));
    let z = 1.0 - abs(e.x) - abs(e.y);
    var xy = e;
    if z < 0.0 {
        xy = octahedral_wrap(e);
    }
    return normalize(vec3<f32>(xy, z));
}

fn vertex_flags(v: vec4<u32>) -> u32 {
    return v.w >> 16u;
}

// Rounds `x` to f32 where it stands. Metal compiles with fast math, which fuses a multiply and
// an add into one rounding and reorders sums; OR-ing in `zero`, a uniform that is always 0 and
// that the compiler cannot fold, turns the value opaque so neither happens across it.
fn rounded(x: f32, zero: u32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | zero);
}

// Mirror of vx_core::mesh::secondary_position_voxels: the same operations in the same order, f32
// throughout, exact comparisons against 0 and 32 (D22). `mask` holds bit `face` for each face
// whose neighbour is drawn one level finer; `zero` must be 0 (see `rounded`). The dot product is
// spelled out in glam's order and every rounding of the reference happens here too, so the result
// matches the reference bit for bit.
fn secondary_position_voxels(
    p: vec3<f32>,
    n: vec3<f32>,
    flags: u32,
    mask: u32,
    zero: u32,
) -> vec3<f32> {
    if mask == 0u || (flags & FLAG_TRANSITION_FULL_RES) != 0u {
        return p;
    }
    var delta = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        let low = (mask & (1u << (2u * axis))) != 0u;
        let high = (mask & (1u << (2u * axis + 1u))) != 0u;
        let c = p[axis];
        if (c == 0.0 && !low) || (c == CHUNK_VOXELS && !high) {
            return p;
        }
        if low && c < 1.0 {
            delta[axis] = (1.0 - c) * TRANSITION_WIDTH_CELLS;
        } else if high && c > CHUNK_VOXELS - 1.0 {
            delta[axis] = (CHUNK_VOXELS - 1.0 - c) * TRANSITION_WIDTH_CELLS;
        }
    }
    if all(delta == vec3<f32>(0.0)) {
        return p;
    }
    let xy = rounded(rounded(n.x * delta.x, zero) + rounded(n.y * delta.y, zero), zero);
    let along = rounded(xy + rounded(n.z * delta.z, zero), zero);
    var shift = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        shift[axis] = rounded(delta[axis] - rounded(n[axis] * along, zero), zero);
    }
    return p + shift;
}
// END vertex decoding and secondary position

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    // Displaced chunk local position plus the chunk's texture phase: periodic in the tile length
    // and independent of the camera and the render origin.
    @location(1) triplanar_m: vec3<f32>,
    // One weight per material id (0 to 15), so interpolation across a triangle blends
    // materials, never slots: a slot that is empty on one vertex and soil on another would
    // otherwise interpolate a weight onto whichever id the provoking vertex holds.
    @location(2) weights0: vec4<f32>,
    @location(3) weights1: vec4<f32>,
    @location(4) weights2: vec4<f32>,
    @location(5) weights3: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    let v = vertices[vi].words;
    let chunk = chunks[v.w & 0xffffu];
    let normal = unpack_normal(v);
    let mask = (chunk.flags >> CHUNK_MASK_SHIFT) & 0x3fu;
    let local_voxels =
        secondary_position_voxels(unpack_position_voxels(v), normal, vertex_flags(v), mask, frame.zero);
    let local_m = local_voxels * chunk.voxel_size_m;
    // Both terms stay within the rebase distance plus the chunk's distance from the camera (D23).
    // The position relative to the render origin is rounded where it stands: with the origin on
    // the voxel grid that sum is exact, so a vertex shared by two chunks of different levels gets
    // the same bits from both, and fast math must not fold the camera offset into it first.
    let z = frame.zero;
    let world_m = local_m + chunk.offset_m;
    let anchored_m = vec3<f32>(rounded(world_m.x, z), rounded(world_m.y, z), rounded(world_m.z, z));
    let relative_m = anchored_m - frame.camera_offset_m.xyz;

    var out: VertexOut;
    out.clip = frame.view_proj * vec4<f32>(relative_m, 1.0);
    out.normal = normal;
    out.triplanar_m = local_m + chunk.texture_phase_m;
    var w = array<f32, 16>();
    for (var i = 0u; i < 4u; i++) {
        let id = (v.z >> (4u * i)) & 0xfu;
        w[id] += f32((v.z >> (16u + 4u * i)) & 0xfu);
    }
    out.weights0 = vec4<f32>(w[0], w[1], w[2], w[3]);
    out.weights1 = vec4<f32>(w[4], w[5], w[6], w[7]);
    out.weights2 = vec4<f32>(w[8], w[9], w[10], w[11]);
    out.weights3 = vec4<f32>(w[12], w[13], w[14], w[15]);
    return out;
}

// Explicit gradients let the samples sit inside per slot branches, which textureSample forbids
// because it needs uniform control flow for its implicit derivatives.
fn triplanar(layer: u32, p: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>, blend: vec3<f32>) -> vec3<f32> {
    let l = min(layer, textureNumLayers(material_albedo) - 1u);
    let x = textureSampleGrad(material_albedo, material_sampler, p.zy, l, dx.zy, dy.zy).rgb;
    let y = textureSampleGrad(material_albedo, material_sampler, p.xz, l, dx.xz, dy.xz).rgb;
    let z = textureSampleGrad(material_albedo, material_sampler, p.xy, l, dx.xy, dy.xy).rgb;
    return x * blend.x + y * blend.y + z * blend.z;
}

struct GBuffer {
    // Linear albedo (the target stores sRGB), alpha 1 where terrain was drawn.
    @location(0) albedo: vec4<f32>,
    // Unit normal in body fixed axes, w reserved (0).
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOut) -> GBuffer {
    let n = normalize(in.normal);
    var blend = pow(abs(n), vec3<f32>(TRIPLANAR_SHARPNESS));
    blend = blend / (blend.x + blend.y + blend.z);

    let p = in.triplanar_m / frame.camera_offset_m.w;
    let dx = dpdx(p);
    let dy = dpdy(p);
    var weights = array<vec4<f32>, 4>(in.weights0, in.weights1, in.weights2, in.weights3);
    let ones = vec4<f32>(1.0);
    let total = dot(in.weights0, ones) + dot(in.weights1, ones) + dot(in.weights2, ones)
        + dot(in.weights3, ones);
    var albedo = vec3<f32>(0.0);
    if total > 0.0 {
        for (var i = 0u; i < 16u; i++) {
            let weight = weights[i / 4u][i % 4u];
            if weight > 0.0 {
                albedo += triplanar(i, p, dx, dy, blend) * (weight / total);
            }
        }
    } else {
        // A vertex without materials renders layer 0, the debug colour.
        albedo = triplanar(0u, p, dx, dy, blend);
    }

    var out: GBuffer;
    out.albedo = vec4<f32>(albedo, 1.0);
    out.normal = vec4<f32>(n, 0.0);
    return out;
}
