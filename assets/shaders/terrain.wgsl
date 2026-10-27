// Flat shaded terrain. Positions arrive chunk local and are offset into the camera relative frame
// on the GPU, so no world position ever reaches this shader (D4). Units are metres.

struct Frame {
    // Projection times the rotation only view matrix (reverse-Z, infinite far plane).
    view_proj: mat4x4<f32>,
    // xyz: camera relative position of the mesh origin.
    mesh_offset_m: vec4<f32>,
    // xyz: unit direction towards the sun in frame axes. w: ambient strength.
    sun_dir: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;

struct VertexIn {
    @location(0) position_m: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) camera_rel_m: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    let camera_rel = v.position_m + frame.mesh_offset_m.xyz;
    var out: VertexOut;
    out.clip = frame.view_proj * vec4<f32>(camera_rel, 1.0);
    out.camera_rel_m = camera_rel;
    out.normal = v.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Face normal from screen space derivatives gives flat shading on an indexed mesh; its sign
    // depends on the raster y direction, so it is oriented by the interpolated vertex normal.
    var n = normalize(cross(dpdx(in.camera_rel_m), dpdy(in.camera_rel_m)));
    if dot(n, in.normal) < 0.0 {
        n = -n;
    }
    let sun = max(dot(n, frame.sun_dir.xyz), 0.0);
    let sky = 0.5 + 0.5 * n.y;
    let albedo = vec3<f32>(0.50, 0.45, 0.38);
    let lit = albedo * (sun + frame.sun_dir.w * mix(0.35, 1.0, sky));
    return vec4<f32>(lit, 1.0);
}
