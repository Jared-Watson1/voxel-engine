// Hillaire 2020 atmosphere (D8, D25): "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique", https://sebh.github.io/publications/egsr2020.pdf. Ported from the
// reference implementation sebh/UnrealEngineSkyAtmosphere (MIT) with ideas from Bevy's
// `bevy_pbr` atmosphere (MIT or Apache-2.0).
//
// Four compute entry points, each dispatched in its own pass: `transmittance` (256 x 64) and
// `multiple_scattering` (32 x 32, plus the sky irradiance LUT) when the parameters change,
// `sky_view` (192 x 108) and `aerial_perspective` (32 x 32 x 32 froxels) every frame.
//
// Units are metres throughout (engineering rule 9). Positions inside the atmosphere functions are
// relative to the planet centre; the only such position is the camera's, the negated planet
// centre relative to the camera (D4). Every LUT is radiance or irradiance per unit of sun
// illuminance above the atmosphere; the lighting pass multiplies by the sun's illuminance.
//
// `lighting.wgsl` is compiled appended to this file, so the structs, bindings of group 0 and the
// LUT parameterisations below are shared by both modules and cannot drift apart. Layouts match
// `AtmosphereUniform` and `ViewUniform` in `crates/vx_render/src/atmosphere.rs`.

struct Atmosphere {
    // x: ground (bottom) radius. y: top radius. z: Mie asymmetry g. w: aerial perspective range.
    radii: vec4<f32>,
    // rgb: Rayleigh scattering per metre at the ground. w: 1 / Rayleigh scale height.
    rayleigh: vec4<f32>,
    // x: Mie scattering per metre at the ground. y: Mie absorption. z: Mie extinction.
    // w: 1 / Mie scale height.
    mie: vec4<f32>,
    // rgb: ozone absorption per metre at the tent's peak. w: altitude of the peak.
    ozone: vec4<f32>,
    // rgb: ground albedo. w: half width of the ozone tent.
    ground: vec4<f32>,
};

struct View {
    // Clip space to camera relative metres (the rotation only view projection, inverted).
    inv_view_proj: mat4x4<f32>,
    // xyz: camera position relative to the planet centre. w: radius the terrain never dips below,
    // 0 for the mean radius.
    camera: vec4<f32>,
    // xyz: unit direction towards the sun. w: angular radius of the disc.
    sun_direction: vec4<f32>,
    // rgb: sun illuminance above the atmosphere. w: unused.
    sun_illuminance: vec4<f32>,
    // xyz: moon centre relative to the camera. w: moon radius, 0 when there is no moon.
    moon: vec4<f32>,
    // rgb: night sky radiance floor. w: moon albedo.
    night: vec4<f32>,
    // x, y: render target size in pixels. z: sky ambient scale. w: angular size of a froxel.
    params: vec4<f32>,
};

const PI: f32 = 3.14159265358979;
const TRANSMITTANCE_SIZE: vec2<f32> = vec2<f32>(256.0, 64.0);
const MULTI_SCATTERING_SIZE: f32 = 32.0;
const SKY_VIEW_SIZE: vec2<f32> = vec2<f32>(192.0, 108.0);
const AERIAL_SIZE: f32 = 32.0;
// Midpoint quadrature in a quadratic parameter (dense near the start of the ray), so the thin Mie
// layer at the ground is resolved. Mirrored by `cpu::optical_depth_to_top` in atmosphere.rs.
const TRANSMITTANCE_STEPS: u32 = 128u;
const MULTI_SCATTERING_STEPS: f32 = 20.0;
// Sky view steps grow with the ray's length through the atmosphere (the reference's variable
// sample count): the fewest for a short zenith ray, the most for the longest chord.
const SKY_VIEW_MIN_STEPS: f32 = 12.0;
const SKY_VIEW_MAX_STEPS: f32 = 32.0;
// Steps per aerial perspective slice; each column is one march written slice by slice.
const AERIAL_SUBSTEPS: u32 = 4u;
// Where in each march segment the medium is sampled (the reference implementation's 0.3).
const SEGMENT_SAMPLE: f32 = 0.3;

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> frame: View;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(4) var multi_scattering_lut: texture_2d<f32>;

@group(1) @binding(0) var transmittance_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var multi_scattering_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var irradiance_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var sky_view_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var aerial_out: texture_storage_3d<rgba16float, write>;

fn bottom_radius() -> f32 {
    return atmosphere.radii.x;
}

fn top_radius() -> f32 {
    return atmosphere.radii.y;
}

// Small lift above the ground sphere, relative to the atmosphere height (the reference uses 10 m
// on Earth's 100 km).
fn ground_offset() -> f32 {
    return 1.0e-4 * (top_radius() - bottom_radius());
}

// sqrt(a^2 - b^2) without the cancellation of the squares at planetary radii.
fn sqrt_diff_sq(a: f32, b: f32) -> f32 {
    return sqrt(max((a - b) * (a + b), 0.0));
}

// Both distances along the ray to the sphere, (-1, -1) on a miss. `ro` relative to the centre.
fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> vec2<f32> {
    let r = length(ro);
    let b = dot(ro, rd);
    let c = (r - radius) * (r + radius);
    let disc = b * b - c;
    if disc < 0.0 {
        return vec2<f32>(-1.0);
    }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

// Distance to the first intersection in front of the origin, or -1.
fn ray_sphere_nearest(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
    let t = ray_sphere(ro, rd, radius);
    if t.y < 0.0 {
        return -1.0;
    }
    if t.x > 0.0 {
        return t.x;
    }
    return t.y;
}

// Texel centres at the ends of the range map to exactly 0 and 1 (the reference's sub UV mapping).
fn unit_from_sub_uv(u: f32, size: f32) -> f32 {
    return (u - 0.5 / size) * (size / (size - 1.0));
}

fn sub_uv_from_unit(u: f32, size: f32) -> f32 {
    return (u + 0.5 / (size - 1.0)) * ((size - 1.0) / size);
}

struct Medium {
    scattering: vec3<f32>,
    extinction: vec3<f32>,
    rayleigh: vec3<f32>,
    mie: f32,
};

fn sample_medium(r: f32) -> Medium {
    let h = max(r - bottom_radius(), 0.0);
    let rayleigh_density = exp(-h * atmosphere.rayleigh.w);
    let mie_density = exp(-h * atmosphere.mie.w);
    let ozone_density = max(0.0, 1.0 - abs(h - atmosphere.ozone.w) / atmosphere.ground.w);
    var m: Medium;
    m.rayleigh = atmosphere.rayleigh.rgb * rayleigh_density;
    m.mie = atmosphere.mie.x * mie_density;
    m.scattering = m.rayleigh + vec3<f32>(m.mie);
    m.extinction = m.rayleigh + vec3<f32>(atmosphere.mie.z * mie_density)
        + atmosphere.ozone.rgb * ozone_density;
    return m;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

// Cornette and Shanks 1992, as in the reference implementation.
fn mie_phase(cos_theta: f32) -> f32 {
    let g = atmosphere.radii.z;
    let k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    let denom = max(1.0 + g * g - 2.0 * g * cos_theta, 1.0e-6);
    return k * (1.0 + cos_theta * cos_theta) / (denom * sqrt(denom));
}

// Transmittance LUT parameterisation from Bruneton 2017 (x: view zenith, y: altitude).
fn transmittance_uv(r: f32, mu: f32) -> vec2<f32> {
    let bottom = bottom_radius();
    let top = top_radius();
    let h = sqrt_diff_sq(top, bottom);
    let rho = sqrt_diff_sq(r, bottom);
    let disc = (top - r) * (top + r) + r * r * mu * mu;
    let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    let d_min = top - r;
    let d_max = rho + h;
    let x_mu = (d - d_min) / max(d_max - d_min, 1.0e-6);
    let x_r = rho / h;
    return vec2<f32>(
        sub_uv_from_unit(clamp(x_mu, 0.0, 1.0), TRANSMITTANCE_SIZE.x),
        sub_uv_from_unit(clamp(x_r, 0.0, 1.0), TRANSMITTANCE_SIZE.y),
    );
}

// Inverse of `transmittance_uv`: (r, mu) of a texel.
fn transmittance_r_mu(uv: vec2<f32>) -> vec2<f32> {
    let bottom = bottom_radius();
    let top = top_radius();
    let x_mu = unit_from_sub_uv(uv.x, TRANSMITTANCE_SIZE.x);
    let x_r = unit_from_sub_uv(uv.y, TRANSMITTANCE_SIZE.y);
    let h = sqrt_diff_sq(top, bottom);
    let rho = h * x_r;
    let r = sqrt(rho * rho + bottom * bottom);
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + x_mu * (d_max - d_min);
    var mu = 1.0;
    if d > 0.0 {
        mu = (h * h - rho * rho - d * d) / (2.0 * r * d);
    }
    return vec2<f32>(r, clamp(mu, -1.0, 1.0));
}

// Transmittance from radius `r` to the top of the atmosphere along a ray of zenith cosine `mu`.
// Only meaningful for rays that miss the ground; callers apply `planet_shadow`.
fn transmittance_to_top(r: f32, mu: f32) -> vec3<f32> {
    let rr = clamp(r, bottom_radius(), top_radius());
    return textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(rr, mu), 0.0).rgb;
}

// Visibility of the sun's centre from radius `r`, soft across the disc so the terminator does not
// alias.
fn planet_shadow(r: f32, mu_sun: f32) -> f32 {
    let sin_horizon = bottom_radius() / max(r, bottom_radius());
    let mu_horizon = -sqrt(max(1.0 - sin_horizon * sin_horizon, 0.0));
    let width = max(frame.sun_direction.w, 1.0e-3);
    return smoothstep(mu_horizon - width, mu_horizon + width, mu_sun);
}

fn multi_scattering_uv(r: f32, mu_sun: f32) -> vec2<f32> {
    let h = (r - bottom_radius()) / (top_radius() - bottom_radius());
    return vec2<f32>(
        sub_uv_from_unit(clamp(mu_sun * 0.5 + 0.5, 0.0, 1.0), MULTI_SCATTERING_SIZE),
        sub_uv_from_unit(clamp(h, 0.0, 1.0), MULTI_SCATTERING_SIZE),
    );
}

fn multi_scattering(r: f32, mu_sun: f32) -> vec3<f32> {
    return textureSampleLevel(multi_scattering_lut, lut_sampler, multi_scattering_uv(r, mu_sun), 0.0).rgb;
}

struct ScatteringResult {
    // In-scattered radiance per unit sun illuminance.
    luminance: vec3<f32>,
    transmittance: vec3<f32>,
    // Scattering integrated as if the in-scattered radiance were 1 (f_ms, equation 7).
    multi_scat_as_1: vec3<f32>,
};

// One march segment of length `dt` sampled at `p` (relative to the planet centre): the
// in-scattered radiance integrated analytically over the segment (equation 11 of the reference
// code), the same for a unit source (f_ms), and the segment's transmittance.
fn scatter_segment(
    p: vec3<f32>,
    dt: f32,
    sun: vec3<f32>,
    phase_r: f32,
    phase_m: f32,
    uniform_phase: bool,
    with_multi_scattering: bool,
) -> ScatteringResult {
    let r = length(p);
    let medium = sample_medium(r);
    let sample_transmittance = exp(-medium.extinction * dt);
    let mu_sun = dot(sun, p / r);
    let sun_transmittance = transmittance_to_top(r, mu_sun) * planet_shadow(r, mu_sun);
    var phase_scattering = medium.scattering / (4.0 * PI);
    if !uniform_phase {
        phase_scattering = medium.rayleigh * phase_r + medium.mie * phase_m;
    }
    var ms = vec3<f32>(0.0);
    if with_multi_scattering {
        ms = multi_scattering(r, mu_sun) * medium.scattering;
    }
    let s = sun_transmittance * phase_scattering + ms;
    let ext = max(medium.extinction, vec3<f32>(1.0e-12));
    var segment: ScatteringResult;
    segment.luminance = (s - s * sample_transmittance) / ext;
    segment.multi_scat_as_1 = (medium.scattering - medium.scattering * sample_transmittance) / ext;
    segment.transmittance = sample_transmittance;
    return segment;
}

// Single scattering along a ray plus the multiple scattering term (equations 1 to 11). `origin`
// is relative to the planet centre and inside the atmosphere. The march ends at the ground, the
// top or `t_limit`. Quadratic steps are dense near the origin, where a ray starting at the
// ground meets the thin Mie layer.
fn integrate_scattering(
    origin: vec3<f32>,
    dir: vec3<f32>,
    sun: vec3<f32>,
    t_limit: f32,
    steps: f32,
    quadratic: bool,
    uniform_phase: bool,
    with_multi_scattering: bool,
    with_ground: bool,
) -> ScatteringResult {
    var result: ScatteringResult;
    result.luminance = vec3<f32>(0.0);
    result.transmittance = vec3<f32>(1.0);
    result.multi_scat_as_1 = vec3<f32>(0.0);

    let t_bottom = ray_sphere_nearest(origin, dir, bottom_radius());
    var t_top = ray_sphere_nearest(origin, dir, top_radius());
    // An origin moved onto the top sphere has a near intersection at zero up to rounding; the
    // march always runs to the far one, where the ray leaves the atmosphere.
    if length(origin) >= top_radius() - ground_offset() {
        t_top = ray_sphere(origin, dir, top_radius()).y;
    }
    var t_max = 0.0;
    if t_bottom < 0.0 {
        if t_top < 0.0 {
            return result;
        }
        t_max = t_top;
    } else if t_top > 0.0 {
        t_max = min(t_top, t_bottom);
    } else {
        t_max = t_bottom;
    }
    let hits_ground = t_bottom > 0.0 && t_bottom <= t_max && t_bottom <= t_limit;
    t_max = min(t_max, t_limit);
    if t_max <= 0.0 {
        return result;
    }

    let cos_theta = dot(sun, dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta);

    var throughput = vec3<f32>(1.0);
    let n = max(steps, 1.0);
    let count = u32(n);
    for (var i = 0u; i < count; i++) {
        var t0 = f32(i) / n;
        var t1 = f32(i + 1u) / n;
        if quadratic {
            t0 = t0 * t0;
            t1 = t1 * t1;
        }
        t0 *= t_max;
        t1 *= t_max;
        let dt = t1 - t0;
        let p = origin + (t0 + dt * SEGMENT_SAMPLE) * dir;
        let segment = scatter_segment(p, dt, sun, phase_r, phase_m, uniform_phase, with_multi_scattering);
        result.luminance += throughput * segment.luminance;
        result.multi_scat_as_1 += throughput * segment.multi_scat_as_1;
        throughput *= segment.transmittance;
    }

    if with_ground && hits_ground {
        let p = origin + t_bottom * dir;
        let r = length(p);
        let up = p / r;
        let mu_sun = dot(sun, up);
        let sun_transmittance = transmittance_to_top(r, mu_sun) * planet_shadow(r, mu_sun);
        result.luminance += throughput * sun_transmittance * max(mu_sun, 0.0)
            * atmosphere.ground.rgb / PI;
    }
    result.transmittance = throughput;
    return result;
}

// Moves an origin outside the atmosphere to where the ray enters it. Returns the distance moved,
// or -1 when the ray misses the atmosphere.
fn distance_to_atmosphere(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    if length(origin) <= top_radius() {
        return 0.0;
    }
    let t = ray_sphere(origin, dir, top_radius());
    if t.y < 0.0 {
        return -1.0;
    }
    return max(t.x, 0.0);
}

// Origin on or above the ground sphere: a camera in a valley below the mean radius is lifted to
// it, since the atmosphere's ground is the mean radius.
fn lift_above_ground(p: vec3<f32>) -> vec3<f32> {
    let r = length(p);
    let floor_r = bottom_radius() + ground_offset();
    if r < floor_r {
        return p * (floor_r / max(r, 1.0e-6));
    }
    return p;
}

// Sky view LUT parameterisation (section 5.3): longitude relative to the sun on x, latitude on y
// with resolution concentrated at the horizon.
fn sky_view_uv(intersects_ground: bool, view_zenith_cos: f32, light_view_cos: f32, view_height: f32) -> vec2<f32> {
    let v_horizon = sqrt_diff_sq(view_height, bottom_radius());
    let beta = acos(clamp(v_horizon / view_height, -1.0, 1.0));
    let zenith_horizon_angle = PI - beta;
    let view_zenith = acos(clamp(view_zenith_cos, -1.0, 1.0));
    var v = 0.0;
    if !intersects_ground {
        var coord = view_zenith / zenith_horizon_angle;
        coord = 1.0 - sqrt(max(1.0 - coord, 0.0));
        v = coord * 0.5;
    } else {
        let coord = (view_zenith - zenith_horizon_angle) / max(beta, 1.0e-6);
        v = sqrt(clamp(coord, 0.0, 1.0)) * 0.5 + 0.5;
    }
    let u = sqrt(clamp(-light_view_cos * 0.5 + 0.5, 0.0, 1.0));
    return vec2<f32>(
        sub_uv_from_unit(u, SKY_VIEW_SIZE.x),
        sub_uv_from_unit(clamp(v, 0.0, 1.0), SKY_VIEW_SIZE.y),
    );
}

// View direction (z up, sun in the xz plane at positive x) of a sky view texel.
fn sky_view_direction(uv: vec2<f32>, view_height: f32) -> vec3<f32> {
    let u = unit_from_sub_uv(uv.x, SKY_VIEW_SIZE.x);
    let v = unit_from_sub_uv(uv.y, SKY_VIEW_SIZE.y);
    let v_horizon = sqrt_diff_sq(view_height, bottom_radius());
    let beta = acos(clamp(v_horizon / view_height, -1.0, 1.0));
    let zenith_horizon_angle = PI - beta;
    var view_zenith = 0.0;
    if v < 0.5 {
        var coord = 1.0 - 2.0 * v;
        coord = 1.0 - coord * coord;
        view_zenith = zenith_horizon_angle * coord;
    } else {
        var coord = v * 2.0 - 1.0;
        coord = coord * coord;
        view_zenith = zenith_horizon_angle + beta * coord;
    }
    let light_view_cos = 1.0 - 2.0 * u * u;
    let sin_zenith = sin(view_zenith);
    return vec3<f32>(
        sin_zenith * light_view_cos,
        sin_zenith * sqrt(max(1.0 - light_view_cos * light_view_cos, 0.0)),
        cos(view_zenith),
    );
}

// Camera relative point on the view ray through a normalised device coordinate.
fn unproject(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let p = frame.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return p.xyz / p.w;
}

// Aerial perspective slices are quadratic in distance: slice coordinate w maps to w^2 * range.
fn aerial_slice_distance(w: f32) -> f32 {
    return w * w * atmosphere.radii.w;
}

@compute @workgroup_size(8, 8, 1)
fn transmittance(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(TRANSMITTANCE_SIZE.x) || id.y >= u32(TRANSMITTANCE_SIZE.y) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / TRANSMITTANCE_SIZE;
    let r_mu = transmittance_r_mu(uv);
    let r = r_mu.x;
    let mu = r_mu.y;
    let t_max = max(0.0, -r * mu + sqrt(max((top_radius() - r) * (top_radius() + r) + r * r * mu * mu, 0.0)));
    let dir = vec3<f32>(sqrt(max(1.0 - mu * mu, 0.0)), 0.0, mu);
    let origin = vec3<f32>(0.0, 0.0, r);
    var depth = vec3<f32>(0.0);
    let n = f32(TRANSMITTANCE_STEPS);
    for (var i = 0u; i < TRANSMITTANCE_STEPS; i++) {
        let s = (f32(i) + 0.5) / n;
        let t = s * s * t_max;
        let dt = 2.0 * s * t_max / n;
        let p = origin + t * dir;
        depth += sample_medium(length(p)).extinction * dt;
    }
    textureStore(transmittance_out, id.xy, vec4<f32>(exp(-depth), 1.0));
}

const MS_DIRECTIONS: u32 = 64u;
var<workgroup> shared_luminance: array<vec3<f32>, 64>;
var<workgroup> shared_multi_scat: array<vec3<f32>, 64>;
var<workgroup> shared_irradiance: array<vec3<f32>, 64>;

// One workgroup per texel and one invocation per direction of an 8 x 8 uniform sphere grid
// (section 5.5). Also writes the sky irradiance on a horizontal surface per unit sun illuminance,
// the cosine weighted single scattering over the upper hemisphere scaled by the same geometric
// series as the multiple scattering (an approximation used for the sky ambient).
@compute @workgroup_size(1, 1, 64)
fn multiple_scattering(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    let uv = (vec2<f32>(id.xy) + 0.5) / MULTI_SCATTERING_SIZE;
    let x = unit_from_sub_uv(uv.x, MULTI_SCATTERING_SIZE);
    let y = unit_from_sub_uv(uv.y, MULTI_SCATTERING_SIZE);
    let mu_sun = clamp(x * 2.0 - 1.0, -1.0, 1.0);
    let offset = ground_offset();
    let r = bottom_radius() + offset + clamp(y, 0.0, 1.0) * (top_radius() - bottom_radius() - 2.0 * offset);
    let origin = vec3<f32>(0.0, 0.0, r);
    let sun = vec3<f32>(0.0, sqrt(max(1.0 - mu_sun * mu_sun, 0.0)), mu_sun);

    let i = f32(index / 8u) + 0.5;
    let j = f32(index % 8u) + 0.5;
    let theta = 2.0 * PI * i / 8.0;
    let cos_phi = 1.0 - 2.0 * j / 8.0;
    let sin_phi = sqrt(max(1.0 - cos_phi * cos_phi, 0.0));
    let dir = vec3<f32>(cos(theta) * sin_phi, sin(theta) * sin_phi, cos_phi);

    let result = integrate_scattering(origin, dir, sun, 1.0e30, MULTI_SCATTERING_STEPS, true, true, false, true);
    let weight = 4.0 * PI / f32(MS_DIRECTIONS);
    shared_luminance[index] = result.luminance * weight;
    shared_multi_scat[index] = result.multi_scat_as_1 * weight;
    shared_irradiance[index] = result.luminance * max(cos_phi, 0.0) * weight;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if index < stride {
            shared_luminance[index] += shared_luminance[index + stride];
            shared_multi_scat[index] += shared_multi_scat[index + stride];
            shared_irradiance[index] += shared_irradiance[index + stride];
        }
        workgroupBarrier();
    }
    if index == 0u {
        let isotropic = 1.0 / (4.0 * PI);
        let f_ms = shared_multi_scat[0] * isotropic;
        let series = 1.0 / (vec3<f32>(1.0) - min(f_ms, vec3<f32>(0.99)));
        let psi_ms = shared_luminance[0] * isotropic * series;
        textureStore(multi_scattering_out, id.xy, vec4<f32>(psi_ms, 1.0));
        textureStore(irradiance_out, id.xy, vec4<f32>(shared_irradiance[0] * series, 1.0));
    }
}

@compute @workgroup_size(8, 8, 1)
fn sky_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(SKY_VIEW_SIZE.x) || id.y >= u32(SKY_VIEW_SIZE.y) {
        return;
    }
    let camera = lift_above_ground(frame.camera.xyz);
    let view_height = length(camera);
    let up = camera / view_height;
    let mu_sun = dot(up, frame.sun_direction.xyz);
    let sun = vec3<f32>(sqrt(max(1.0 - mu_sun * mu_sun, 0.0)), 0.0, mu_sun);
    let uv = (vec2<f32>(id.xy) + 0.5) / SKY_VIEW_SIZE;
    let dir = sky_view_direction(uv, view_height);
    var origin = vec3<f32>(0.0, 0.0, view_height);
    let entry = distance_to_atmosphere(origin, dir);
    if entry < 0.0 {
        textureStore(sky_view_out, id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    origin += entry * dir;
    var length_m = ray_sphere(origin, dir, top_radius()).y;
    let t_ground = ray_sphere_nearest(origin, dir, bottom_radius());
    if t_ground > 0.0 {
        length_m = min(length_m, t_ground);
    }
    let steps = mix(SKY_VIEW_MIN_STEPS, SKY_VIEW_MAX_STEPS, clamp(length_m / atmosphere.radii.w, 0.0, 1.0));
    // From outside the atmosphere the densest air is mid ray, so the steps are uniform.
    let result = integrate_scattering(origin, dir, sun, 1.0e30, floor(steps), entry == 0.0, false, true, false);
    textureStore(sky_view_out, id.xy, vec4<f32>(result.luminance, 1.0));
}

// Froxels over the view frustum (section 5.4): x and y follow the screen (y down), slice w covers
// distance w^2 * range along each pixel's ray from where the ray enters the atmosphere (the camera
// itself when inside). rgb: in-scattered radiance per unit sun illuminance. a: 1 - mean
// transmittance. One invocation marches a whole column and writes each slice as it passes it
// (as Bevy's aerial view LUT does), so the column costs 32 slices of steps rather than the sum of
// 32 separate marches. Slices beyond the ground keep the value at the ground, as the reference
// moves such froxels onto it.
@compute @workgroup_size(8, 8, 1)
fn aerial_perspective(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = u32(AERIAL_SIZE);
    if id.x >= size || id.y >= size {
        return;
    }
    let camera = lift_above_ground(frame.camera.xyz);
    let uv = (vec2<f32>(id.xy) + 0.5) / AERIAL_SIZE;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let dir = normalize(unproject(ndc, 1.0));
    let entry = distance_to_atmosphere(camera, dir);
    if entry < 0.0 {
        for (var z = 0u; z < size; z++) {
            textureStore(aerial_out, vec3<u32>(id.xy, z), vec4<f32>(0.0));
        }
        return;
    }
    let origin = camera + entry * dir;
    let sun = frame.sun_direction.xyz;
    var t_end = ray_sphere(origin, dir, top_radius()).y;
    let t_ground = ray_sphere_nearest(origin, dir, bottom_radius());
    if t_ground > 0.0 {
        t_end = min(t_end, t_ground);
    }
    t_end = max(t_end, 0.0);
    let cos_theta = dot(sun, dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta);
    var luminance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var t = 0.0;
    for (var z = 0u; z < size; z++) {
        let t_slice = min(aerial_slice_distance((f32(z) + 0.5) / AERIAL_SIZE), t_end);
        let dt = (t_slice - t) / f32(AERIAL_SUBSTEPS);
        if dt > 0.0 {
            for (var k = 0u; k < AERIAL_SUBSTEPS; k++) {
                let p = origin + (t + (f32(k) + SEGMENT_SAMPLE) * dt) * dir;
                let segment = scatter_segment(p, dt, sun, phase_r, phase_m, false, true);
                luminance += throughput * segment.luminance;
                throughput *= segment.transmittance;
            }
            t = t_slice;
        }
        let mean_transmittance = dot(throughput, vec3<f32>(1.0 / 3.0));
        textureStore(aerial_out, vec3<u32>(id.xy, z), vec4<f32>(luminance, 1.0 - mean_transmittance));
    }
}
