// Deferred lighting and the sky (D25). Placeholder from the M2 interface commit: a Lambert sun and
// a gradient sky; workstream F replaces it with the Hillaire LUTs (sun transmittance, sky ambient,
// aerial perspective). Positions are relative to the camera (D4). Layout matches
// `LightingUniform` in `crates/vx_render/src/atmosphere.rs`.

struct Lighting {
    inv_view_proj: mat4x4<f32>,
    // xyz: unit direction towards the sun. w: angular radius.
    sun_direction: vec4<f32>,
    // rgb: sun illuminance above the atmosphere.
    sun_illuminance: vec4<f32>,
    // xyz: planet centre relative to the camera. w: mean radius.
    planet: vec4<f32>,
    // x: atmosphere top above the mean radius.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Lighting;
@group(0) @binding(1) var albedo_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    let ndc = uv * 2.0 - 1.0;
    var out: VertexOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.ndc = ndc;
    return out;
}

fn sky(dir: vec3<f32>, up: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let day = smoothstep(-0.1, 0.2, dot(up, sun));
    let t = clamp(dot(dir, up), 0.0, 1.0);
    let colour = mix(vec3<f32>(0.62, 0.74, 0.9), vec3<f32>(0.16, 0.34, 0.78), sqrt(t));
    return colour * (0.02 + day);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.clip.xy);
    let depth = textureLoad(depth_tex, coord, 0);
    let sun = u.sun_direction.xyz;
    let up = normalize(-u.planet.xyz);
    if depth == 0.0 {
        // Reverse-Z: NDC depth 1 is the near plane, so this point is on the view ray.
        let near = u.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
        let dir = normalize(near.xyz / near.w);
        var colour = sky(dir, up, sun);
        if dot(dir, sun) > cos(u.sun_direction.w) {
            colour += u.sun_illuminance.rgb * 20.0;
        }
        return vec4<f32>(colour, 1.0);
    }
    let albedo = textureLoad(albedo_tex, coord, 0).rgb;
    let n = normalize(textureLoad(normal_tex, coord, 0).xyz);
    let direct = u.sun_illuminance.rgb * max(dot(n, sun), 0.0) * smoothstep(-0.05, 0.1, dot(up, sun));
    let ambient = sky(n, up, sun) * (0.6 + 0.4 * dot(n, up));
    return vec4<f32>(albedo * (direct + ambient), 1.0);
}
