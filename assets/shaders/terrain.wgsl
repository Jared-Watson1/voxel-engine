// Pooled terrain with vertex pulling and triplanar materials (D10, D14, D19). Vertices arrive
// packed (vx_core::mesh) and chunk local; the chunk table supplies the camera relative offset,
// so no world position ever reaches this shader (D4). Units are metres.
//
// The struct layouts match `FrameUniform`, `PackedVertex` and `ChunkEntry` in
// `crates/vx_render/src/terrain.rs` (engineering rule 9) and are repeated in `terrain_cull.wgsl`.

struct Frame {
    // Projection times the rotation only view matrix (reverse-Z, infinite far plane).
    view_proj: mat4x4<f32>,
    // Frustum planes in the camera relative frame as (normal, distance); inside is positive.
    planes: array<vec4<f32>, 6>,
    // xyz: unit direction towards the sun in frame axes. w: ambient strength.
    sun_dir: vec4<f32>,
    // x: voxel size. y: chunk extent (32 voxels). z: material tile length. w: unused.
    params: vec4<f32>,
    // Entries in the chunk table this frame.
    slot_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Four u32 words, bit layout documented in vx_core::mesh.
struct PackedVertex {
    words: vec4<u32>,
};

struct ChunkEntry {
    offset_m: vec3<f32>,
    vertex_base: u32,
    texture_phase_m: vec3<f32>,
    index_base: u32,
    index_count: u32,
    // Bit 0: resident this frame.
    flags: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> vertices: array<PackedVertex>;
@group(0) @binding(2) var<storage, read> chunks: array<ChunkEntry>;
@group(1) @binding(0) var material_albedo: texture_2d_array<f32>;
@group(1) @binding(1) var material_sampler: sampler;

const POSITION_UNITS_PER_VOXEL: f32 = 1024.0;
const TRIPLANAR_SHARPNESS: f32 = 4.0;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    // Chunk local position plus the chunk's texture phase: periodic in the tile length and
    // independent of the camera.
    @location(1) triplanar_m: vec3<f32>,
    @location(2) @interpolate(flat) material_ids: vec4<u32>,
    @location(3) material_weights: vec4<f32>,
};

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

fn decode_octahedral(e: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(e.x) - abs(e.y);
    var xy = e;
    if z < 0.0 {
        xy = octahedral_wrap(e);
    }
    return normalize(vec3<f32>(xy, z));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    let v = vertices[vi].words;
    let units = vec3<f32>(f32(v.x & 0xffffu), f32(v.x >> 16u), f32(v.y & 0xffffu));
    let local_m = units * (frame.params.x / POSITION_UNITS_PER_VOXEL);
    let normal = decode_octahedral(vec2<f32>(snorm8(v.y, 16u), snorm8(v.y, 24u)));
    let chunk = chunks[v.w & 0xffffu];

    var out: VertexOut;
    out.clip = frame.view_proj * vec4<f32>(local_m + chunk.offset_m, 1.0);
    out.normal = normal;
    out.triplanar_m = local_m + chunk.texture_phase_m;
    out.material_ids = vec4<u32>(
        v.z & 0xfu,
        (v.z >> 4u) & 0xfu,
        (v.z >> 8u) & 0xfu,
        (v.z >> 12u) & 0xfu,
    );
    out.material_weights = vec4<f32>(
        f32((v.z >> 16u) & 0xfu),
        f32((v.z >> 20u) & 0xfu),
        f32((v.z >> 24u) & 0xfu),
        f32((v.z >> 28u) & 0xfu),
    );
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

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    var blend = pow(abs(n), vec3<f32>(TRIPLANAR_SHARPNESS));
    blend = blend / (blend.x + blend.y + blend.z);

    let p = in.triplanar_m / frame.params.z;
    let dx = dpdx(p);
    let dy = dpdy(p);
    let total = in.material_weights.x + in.material_weights.y + in.material_weights.z
        + in.material_weights.w;
    var albedo = vec3<f32>(0.0);
    if total > 0.0 {
        for (var i = 0u; i < 4u; i++) {
            let weight = in.material_weights[i];
            if weight > 0.0 {
                albedo += triplanar(in.material_ids[i], p, dx, dy, blend) * (weight / total);
            }
        }
    } else {
        // A vertex without materials renders layer 0, the debug colour.
        albedo = triplanar(0u, p, dx, dy, blend);
    }

    let sun = max(dot(n, frame.sun_dir.xyz), 0.0);
    let sky = 0.5 + 0.5 * n.y;
    let lit = albedo * (sun + frame.sun_dir.w * mix(0.35, 1.0, sky));
    return vec4<f32>(lit, 1.0);
}
