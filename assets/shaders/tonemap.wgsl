// HDR to display (D25). Placeholder from the M2 interface commit: exposure and the ACES fit of
// Narkowicz 2016; workstream F replaces the curve. The target is an sRGB view, so the output is
// linear. Layout matches `TonemapUniform` in `crates/vx_render/src/post.rs`.

struct Tonemap {
    // x: exposure scale 2^EV. yzw: reserved.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tonemap: Tonemap;
@group(0) @binding(1) var hdr: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let radiance = textureLoad(hdr, vec2<i32>(pos.xy), 0).rgb;
    return vec4<f32>(aces(radiance * tonemap.params.x), 1.0);
}
