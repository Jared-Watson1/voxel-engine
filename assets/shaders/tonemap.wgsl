// HDR to display (D25): exposure, then Khronos PBR Neutral
// (https://github.com/KhronosGroup/ToneMapping, Apache-2.0), chosen in `crates/vx_render/src/post.rs`.
// The target is an sRGB view, so the output is linear. Layout matches `TonemapUniform` in
// `crates/vx_render/src/post.rs`, whose tests mirror `pbr_neutral` on the CPU.

struct Tonemap {
    // x: exposure scale (calibration times 2^EV). yzw: reserved.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tonemap: Tonemap;
@group(0) @binding(1) var hdr: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

const START_COMPRESSION: f32 = 0.76;
const DESATURATION: f32 = 0.15;

fn pbr_neutral(colour: vec3<f32>) -> vec3<f32> {
    let x = min(colour.r, min(colour.g, colour.b));
    var offset = 0.04;
    if x < 0.08 {
        offset = x - 6.25 * x * x;
    }
    var c = colour - offset;
    let peak = max(c.r, max(c.g, c.b));
    if peak < START_COMPRESSION {
        return c;
    }
    let d = 1.0 - START_COMPRESSION;
    let new_peak = 1.0 - d * d / (peak + d - START_COMPRESSION);
    c *= new_peak / peak;
    let g = 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0);
    return mix(c, vec3<f32>(new_peak), g);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let radiance = textureLoad(hdr, vec2<i32>(pos.xy), 0).rgb;
    // Non finite or negative radiance would poison the curve; clamp to the representable range.
    let exposed = clamp(radiance * tonemap.params.x, vec3<f32>(0.0), vec3<f32>(65504.0));
    return vec4<f32>(pbr_neutral(exposed), 1.0);
}
