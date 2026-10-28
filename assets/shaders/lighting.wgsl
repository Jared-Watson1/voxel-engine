// Deferred lighting and the sky (D25). Compiled appended to `atmosphere.wgsl`, whose structs,
// group 0 bindings and LUT functions it uses. Writes radiance per pixel into the HDR target in
// the units of the sun's illuminance (a white Lambert surface facing a sun of illuminance 1 above
// the atmosphere has radiance 1 / pi); the tonemap pass applies the exposure.
//
// Per pixel: depth 0 (reverse-Z, infinite far plane) is the sky from the sky view LUT, with the
// sun disc, the moon and the night floor. Anything else is terrain: the sun through the
// transmittance at the pixel's altitude and sun angle, sky ambient from the irradiance LUT, ground
// bounce and the night floor, then aerial perspective from the froxel volume. A per pixel march
// replaces the LUTs where they fall short: terrain beyond the froxel range, and from outside the
// atmosphere a planet small on screen (a sharp limb over few pixels) and the rim of the froxels.
// Positions are camera relative (D4); the camera's position relative to the planet centre comes
// from the uniform.

@group(2) @binding(0) var irradiance_lut: texture_2d<f32>;
@group(2) @binding(1) var sky_view_lut: texture_2d<f32>;
@group(2) @binding(2) var aerial_lut: texture_3d<f32>;
@group(2) @binding(3) var albedo_tex: texture_2d<f32>;
@group(2) @binding(4) var normal_tex: texture_2d<f32>;
@group(2) @binding(5) var depth_tex: texture_depth_2d;

// Steps of the per pixel march; uniform from outside the atmosphere, quadratic from inside.
const VIEW_STEPS: f32 = 24.0;
// Linear limb darkening coefficient of the sun disc (about 0.6 for the Sun in the visible).
const LIMB_DARKENING: f32 = 0.6;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VertexOut;
    out.clip = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn sky_irradiance(r: f32, mu_sun: f32) -> vec3<f32> {
    return textureSampleLevel(irradiance_lut, lut_sampler, multi_scattering_uv(r, mu_sun), 0.0).rgb;
}

// Sky radiance per unit sun illuminance from the sky view LUT; the camera is inside the
// atmosphere.
fn sky_view_radiance(camera: vec3<f32>, dir: vec3<f32>, hits_ground: bool) -> vec3<f32> {
    let view_height = length(camera);
    let up = camera / view_height;
    let sun = frame.sun_direction.xyz;
    let view_zenith_cos = dot(dir, up);
    let sun_zenith_cos = dot(sun, up);
    let side_sun = sun - up * sun_zenith_cos;
    let side_view = dir - up * view_zenith_cos;
    let ls = length(side_sun);
    let lv = length(side_view);
    // With the sun or the view at the zenith every azimuth is the same texel row.
    var light_view_cos = 1.0;
    if ls > 1.0e-5 && lv > 1.0e-5 {
        light_view_cos = dot(side_sun, side_view) / (ls * lv);
    }
    let uv = sky_view_uv(hits_ground, view_zenith_cos, light_view_cos, view_height);
    return textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb;
}

// In-scattering and transmittance from the camera along `dir` for `t_limit` metres, entering the
// atmosphere first when the camera is above it.
fn march_view(camera: vec3<f32>, dir: vec3<f32>, t_limit: f32) -> ScatteringResult {
    let entry = distance_to_atmosphere(camera, dir);
    if entry < 0.0 || entry >= t_limit {
        var none: ScatteringResult;
        none.luminance = vec3<f32>(0.0);
        none.transmittance = vec3<f32>(1.0);
        none.multi_scat_as_1 = vec3<f32>(0.0);
        return none;
    }
    let origin = camera + entry * dir;
    return integrate_scattering(origin, dir, frame.sun_direction.xyz, t_limit - entry, VIEW_STEPS, entry == 0.0, false, true, false);
}

// Outgoing radiance of a Lambert surface at `p` (relative to the planet centre): the sun through
// the atmosphere, the sky ambient on the upper hemisphere of the normal, the ground bounce on the
// lower one and the night floor.
fn surface_radiance(p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let r = max(length(p), bottom_radius() + ground_offset());
    let up = normalize(p);
    let sun = frame.sun_direction.xyz;
    let mu_sun = dot(up, sun);
    let e_sun = frame.sun_illuminance.rgb;
    let sun_transmittance = transmittance_to_top(r, mu_sun) * planet_shadow(r, mu_sun);
    let direct = e_sun * sun_transmittance * max(dot(n, sun), 0.0);
    let sky = sky_irradiance(r, mu_sun) * e_sun * frame.params.z;
    let hemisphere = 0.5 + 0.5 * dot(n, up);
    let ground_irradiance = sky + e_sun * sun_transmittance * max(mu_sun, 0.0);
    let bounce = atmosphere.ground.rgb * ground_irradiance * (1.0 - hemisphere);
    let irradiance = direct + sky * hemisphere + bounce;
    return albedo / PI * irradiance + albedo * frame.night.rgb * hemisphere;
}

// Radiance of the sun disc along `dir` above the atmosphere, limb darkened, normalised so the
// disc integrates to the sun's illuminance. `sin_angle` is the sine of the angle to the centre
// and `aa` its screen space derivative.
fn sun_disc(sin_angle: f32, aa: f32, facing: bool) -> vec3<f32> {
    let radius = frame.sun_direction.w;
    if !facing || radius <= 0.0 {
        return vec3<f32>(0.0);
    }
    let edge = clamp((radius - sin_angle) / max(aa, 1.0e-6) + 0.5, 0.0, 1.0);
    if edge <= 0.0 {
        return vec3<f32>(0.0);
    }
    let x = min(sin_angle / radius, 1.0);
    let mu_limb = sqrt(max(1.0 - x * x, 0.0));
    let limb = 1.0 - LIMB_DARKENING * (1.0 - mu_limb);
    let centre = 1.0 / (PI * radius * radius * (1.0 - LIMB_DARKENING / 3.0));
    return frame.sun_illuminance.rgb * centre * limb * edge;
}

// From outside the atmosphere the per pixel march serves a small planet disc (a sharp limb, few
// pixels) and the LUTs a large one, crossfaded between these angular radii of the atmosphere.
const MARCH_BELOW_RAD: f32 = 0.1047;
const LUTS_ABOVE_RAD: f32 = 0.1745;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.clip.xy);
    let uv = in.clip.xy / frame.params.xy;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray = normalize(unproject(ndc, 1.0));
    let sun = frame.sun_direction.xyz;
    // Derivatives need uniform control flow, so the sun's angular distance is taken here.
    let sin_sun = length(cross(ray, sun));
    let sun_aa = fwidth(sin_sun);

    let depth = textureLoad(depth_tex, coord, 0);
    let e_sun = frame.sun_illuminance.rgb;
    let camera_raw = frame.camera.xyz;
    let camera = lift_above_ground(camera_raw);
    let camera_r = length(camera);
    let camera_up = camera / camera_r;
    let inside = camera_r < top_radius();
    var lut_weight = 1.0;
    var atmosphere_radius_rad = PI;
    if !inside {
        atmosphere_radius_rad = asin(min(top_radius() / camera_r, 1.0));
        lut_weight = smoothstep(MARCH_BELOW_RAD, LUTS_ABOVE_RAD, atmosphere_radius_rad);
    }

    // A surface is terrain, or the ground sphere where a ray meets it with no terrain drawn (far
    // from orbit, or a streaming hole), lit with the ground albedo.
    var surface = false;
    var distance = 0.0;
    var dir = ray;
    var colour = vec3<f32>(0.0);
    if depth == 0.0 {
        let t_ground = ray_sphere_nearest(camera, ray, bottom_radius());
        if t_ground >= 0.0 {
            surface = true;
            distance = t_ground;
            let p = camera + t_ground * ray;
            colour = surface_radiance(p, normalize(p), atmosphere.ground.rgb);
        }
    } else {
        let pos = unproject(ndc, depth);
        surface = true;
        distance = length(pos);
        dir = pos / max(distance, 1.0e-6);
        let albedo = textureLoad(albedo_tex, coord, 0).rgb;
        let normal = normalize(textureLoad(normal_tex, coord, 0).xyz);
        colour = surface_radiance(camera_raw + pos, normal, albedo);
    }

    let entry_or_miss = distance_to_atmosphere(camera, dir);
    let entry = max(entry_or_miss, 0.0);
    var march_weight = 0.0;
    var lut_luminance = vec3<f32>(0.0);
    var lut_transmittance = vec3<f32>(1.0);
    var froxel = vec4<f32>(0.0);
    if surface {
        let range = atmosphere.radii.w;
        let along = distance - entry;
        var froxel_weight = lut_weight;
        if !inside {
            // Near the atmosphere's silhouette a froxel's centre ray can miss the atmosphere
            // while the pixel's ray does not, so the outer two froxels march per pixel.
            let from_centre = acos(clamp(dot(dir, -camera_up), -1.0, 1.0));
            let margin = (atmosphere_radius_rad - from_centre) / max(frame.params.w, 1.0e-4);
            froxel_weight *= smoothstep(1.0, 2.0, margin);
        }
        if along >= range {
            froxel_weight = 0.0;
        }
        march_weight = 1.0 - froxel_weight;
        if froxel_weight > 0.0 {
            let w = sqrt(max(along, 0.0) / range);
            let slice = w * AERIAL_SIZE;
            // The first half slice fades in from its start, as in the reference implementation.
            var fade = 1.0;
            var w_lookup = w;
            if slice < 0.5 {
                fade = clamp(slice * 2.0, 0.0, 1.0);
                w_lookup = 0.5 / AERIAL_SIZE;
            }
            froxel = textureSampleLevel(aerial_lut, lut_sampler, vec3<f32>(uv, w_lookup), 0.0) * fade;
        }
    } else {
        march_weight = 1.0 - lut_weight;
        if lut_weight > 0.0 && entry_or_miss >= 0.0 {
            lut_luminance = sky_view_radiance(camera, ray, false);
            let origin = camera + entry * ray;
            let origin_r = length(origin);
            lut_transmittance = transmittance_to_top(origin_r, dot(ray, origin / origin_r));
        }
    }

    var march: ScatteringResult;
    march.luminance = vec3<f32>(0.0);
    march.transmittance = vec3<f32>(1.0);
    if march_weight > 0.0 {
        var t_limit = 1.0e30;
        if surface {
            t_limit = distance;
        }
        march = march_view(camera, dir, t_limit);
    }

    if surface {
        let with_froxels = colour * (1.0 - froxel.a) + froxel.rgb * e_sun;
        let with_march = colour * march.transmittance + march.luminance * e_sun;
        return vec4<f32>(mix(with_froxels, with_march, march_weight), 1.0);
    }

    let sky = mix(lut_luminance, march.luminance, march_weight) * e_sun;
    let view_transmittance = mix(lut_transmittance, march.transmittance, march_weight);
    var behind = vec3<f32>(0.0);
    var sun_visible = true;
    if frame.moon.w > 0.0 {
        let moon_centre = frame.moon.xyz;
        let t_moon = ray_sphere(-moon_centre, ray, frame.moon.w);
        if t_moon.x > 0.0 {
            let n = normalize(t_moon.x * ray - moon_centre);
            behind += frame.night.w / PI * e_sun * max(dot(n, sun), 0.0);
            sun_visible = false;
        }
    }
    behind += sun_disc(sin_sun, sun_aa, sun_visible && dot(ray, sun) > 0.0);
    // The night floor glows from the air like airglow: full along a column as long as the
    // zenith column from the ground, none in space.
    let shell = ray_sphere(camera, ray, top_radius());
    let path = max(shell.y - max(shell.x, 0.0), 0.0);
    let glow = clamp(path / (top_radius() - bottom_radius()), 0.0, 1.0);
    return vec4<f32>(sky + view_transmittance * behind + frame.night.rgb * glow, 1.0);
}
