//! The atmosphere LUTs and the lighting pass (D8, D25), after Hillaire 2020 ("A Scalable and
//! Production Ready Sky and Atmosphere Rendering Technique"; reference code
//! sebh/UnrealEngineSkyAtmosphere, MIT; Bevy's `bevy_pbr` atmosphere, MIT or Apache-2.0).
//!
//! Four LUTs, each written by its own compute pass so it can be timed at a pass boundary:
//! transmittance (256 x 64) and multiple scattering (32 x 32, with a sky irradiance LUT of the
//! same shape written by the same pass) when the parameters change, sky view (192 x 108) and
//! aerial perspective (32 x 32 x 32) every frame. The lighting pass then lights the G-buffer and
//! draws the sky into the HDR target.
//!
//! Units: metres for every length and per metre coefficients (engineering rule 9). LUT values
//! are per unit of sun illuminance above the atmosphere. The only position given to the shaders
//! is the camera relative to the planet centre, the negated [`crate::PlanetView::centre_m`] (D4).
//!
//! The atmosphere's ground is the planet's mean radius from [`crate::PlanetView::radius_m`];
//! [`AtmospherePass::set_params`] supplies the optical depths and scale heights, which scale with
//! that radius ([`AtmosphereParams`]).

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use vx_core::AtmosphereParams;

use crate::post::{fullscreen_pipeline, uniform_entry};
use crate::timing::GpuTimer;
use crate::{FrameInput, FrameTargets, HDR_FORMAT};

pub(crate) const ATMOSPHERE_WGSL: &str = include_str!("../../../assets/shaders/atmosphere.wgsl");
pub(crate) const LIGHTING_WGSL: &str = include_str!("../../../assets/shaders/lighting.wgsl");

pub(crate) const TRANSMITTANCE_SIZE: [u32; 2] = [256, 64];
pub(crate) const MULTI_SCATTERING_SIZE: u32 = 32;
pub(crate) const SKY_VIEW_SIZE: [u32; 2] = [192, 108];
pub(crate) const AERIAL_SIZE: u32 = 32;
/// Storage writable and filterable without optional features.
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Steps of the transmittance integral, mirrored from `TRANSMITTANCE_STEPS` in atmosphere.wgsl.
#[cfg(test)]
pub(crate) const TRANSMITTANCE_STEPS: u32 = 128;

/// Multiplier on the physical sky irradiance. 1.0 keeps the ambient physical; the strong blue
/// ambient of the art direction comes from the irradiance LUT itself (about a fifth of the sun's
/// irradiance at noon on the default sky, more at low sun) and from the absence of shadows in M2.
pub(crate) const SKY_AMBIENT_SCALE: f32 = 1.0;

/// Night floor: radiance of a uniform dim blue sky added to the sky and, as irradiance, to the
/// terrain, per unit of sun illuminance. Starlight and airglow are about 1e-8 of daylight and
/// moonlight about 1e-6, invisible at a fixed exposure, so this floor stands in for the adapted
/// eye (with the bluish cast of scotopic vision) until auto exposure exists. Artistic, not
/// physical: about a sixth of the noon zenith's blue, the least that survives the toe of the
/// tonemap as a dark navy sky over near black terrain at `exposure_ev` 0.
pub(crate) const NIGHT_SKY_RADIANCE: [f32; 3] = [0.0012, 0.0020, 0.0045];

/// Matches `struct Atmosphere` in `assets/shaders/atmosphere.wgsl` (engineering rule 9).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct AtmosphereUniform {
    /// x: ground radius. y: top radius. z: Mie asymmetry. w: aerial perspective range. Metres.
    pub radii: [f32; 4],
    /// rgb: Rayleigh scattering per metre at the ground. w: inverse scale height.
    pub rayleigh: [f32; 4],
    /// x: Mie scattering, y: absorption, z: extinction per metre at the ground. w: inverse scale
    /// height.
    pub mie: [f32; 4],
    /// rgb: ozone absorption per metre at the tent's peak. w: altitude of the peak.
    pub ozone: [f32; 4],
    /// rgb: ground albedo. w: half width of the ozone tent.
    pub ground: [f32; 4],
}

const _: () = assert!(size_of::<AtmosphereUniform>() == 80);
const _: () = assert!(std::mem::offset_of!(AtmosphereUniform, rayleigh) == 16);
const _: () = assert!(std::mem::offset_of!(AtmosphereUniform, mie) == 32);
const _: () = assert!(std::mem::offset_of!(AtmosphereUniform, ozone) == 48);
const _: () = assert!(std::mem::offset_of!(AtmosphereUniform, ground) == 64);

/// Matches `struct View` in `assets/shaders/atmosphere.wgsl` (engineering rule 9).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct ViewUniform {
    /// Clip space to camera relative metres.
    pub inv_view_proj: [[f32; 4]; 4],
    /// xyz: camera relative to the planet centre in metres. w: unused.
    pub camera: [f32; 4],
    /// xyz: unit direction towards the sun. w: angular radius.
    pub sun_direction: [f32; 4],
    /// rgb: sun illuminance above the atmosphere. w: unused.
    pub sun_illuminance: [f32; 4],
    /// xyz: moon centre relative to the camera in metres. w: radius, 0 without a moon.
    pub moon: [f32; 4],
    /// rgb: night sky radiance floor. w: moon albedo.
    pub night: [f32; 4],
    /// x, y: target size in pixels. z: sky ambient scale. w: angular size of a froxel in
    /// radians (the larger of its width and height).
    pub params: [f32; 4],
}

const _: () = assert!(size_of::<ViewUniform>() == 160);
const _: () = assert!(std::mem::offset_of!(ViewUniform, camera) == 64);
const _: () = assert!(std::mem::offset_of!(ViewUniform, sun_direction) == 80);
const _: () = assert!(std::mem::offset_of!(ViewUniform, sun_illuminance) == 96);
const _: () = assert!(std::mem::offset_of!(ViewUniform, moon) == 112);
const _: () = assert!(std::mem::offset_of!(ViewUniform, night) == 128);
const _: () = assert!(std::mem::offset_of!(ViewUniform, params) == 144);

impl ViewUniform {
    pub(crate) fn new(frame: &FrameInput, width: u32, height: u32) -> Self {
        let (moon, albedo) = match frame.moon {
            Some(m) => (m.centre_m.extend(m.radius_m).to_array(), m.albedo),
            None => ([0.0; 4], 0.0),
        };
        let [nr, ng, nb] = NIGHT_SKY_RADIANCE;
        Self {
            inv_view_proj: frame.view_proj.inverse().to_cols_array_2d(),
            camera: (-frame.planet.centre_m).extend(0.0).to_array(),
            sun_direction: frame
                .sun
                .direction
                .normalize_or(Vec3::Y)
                .extend(frame.sun.angular_radius_rad)
                .to_array(),
            sun_illuminance: frame.sun.illuminance.extend(0.0).to_array(),
            moon,
            night: [nr, ng, nb, albedo],
            params: [
                width as f32,
                height as f32,
                SKY_AMBIENT_SCALE,
                froxel_angle_rad(frame.view_proj),
            ],
        }
    }
}

/// Angular size of a froxel: the field of view over the froxel count, along the wider axis. The
/// view is rotation only, so a row of the projection's upper 3 x 3 block keeps its focal scale.
fn froxel_angle_rad(view_proj: glam::Mat4) -> f32 {
    let focal = view_proj
        .row(0)
        .truncate()
        .length()
        .min(view_proj.row(1).truncate().length())
        .max(1.0e-6);
    2.0 * (1.0 / focal).atan() / AERIAL_SIZE as f32
}

/// Per metre coefficients and radii derived from [`AtmosphereParams`] and the planet radius, in
/// f64. The single derivation behind the uniform and the CPU reference of the transmittance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Coefficients {
    pub bottom_m: f64,
    pub top_m: f64,
    pub rayleigh_per_m: [f64; 3],
    pub inv_rayleigh_height_per_m: f64,
    pub mie_scattering_per_m: f64,
    pub mie_absorption_per_m: f64,
    pub inv_mie_height_per_m: f64,
    pub mie_asymmetry: f64,
    pub ozone_per_m: [f64; 3],
    pub ozone_centre_m: f64,
    pub ozone_half_width_m: f64,
    pub ground_albedo: [f64; 3],
}

impl Coefficients {
    pub(crate) fn new(params: &AtmosphereParams, radius_m: f64) -> Self {
        let height = params.top_m(radius_m);
        let (mie_scattering_per_m, mie_absorption_per_m) = params.mie_per_m(radius_m);
        Self {
            bottom_m: radius_m,
            top_m: radius_m + height,
            rayleigh_per_m: params.rayleigh_scattering_per_m(radius_m),
            inv_rayleigh_height_per_m: 1.0 / (params.rayleigh_scale_height_fraction * height),
            mie_scattering_per_m,
            mie_absorption_per_m,
            inv_mie_height_per_m: 1.0 / (params.mie_scale_height_fraction * height),
            mie_asymmetry: params.mie_asymmetry,
            ozone_per_m: params.ozone_absorption_per_m(radius_m),
            ozone_centre_m: params.ozone_centre_fraction * height,
            ozone_half_width_m: 0.5 * params.ozone_width_fraction * height,
            ground_albedo: params.ground_albedo,
        }
    }

    /// Longest segment inside the atmosphere that misses the ground: the chord through the top
    /// sphere tangent to the ground. Every visible ground point is closer than this to the camera
    /// (inside the atmosphere) or to where the view ray enters it (outside), so the froxels,
    /// whose slices start at that entry, cover all terrain but valleys below the mean radius.
    pub(crate) fn aerial_range_m(&self) -> f64 {
        2.0 * ((self.top_m - self.bottom_m) * (self.top_m + self.bottom_m)).sqrt()
    }

    #[cfg(test)]
    /// Extinction per metre at radius `r_m`, as `sample_medium` in atmosphere.wgsl.
    pub(crate) fn extinction_per_m(&self, r_m: f64) -> [f64; 3] {
        let h = (r_m - self.bottom_m).max(0.0);
        let rayleigh = (-h * self.inv_rayleigh_height_per_m).exp();
        let mie = (-h * self.inv_mie_height_per_m).exp();
        let ozone = (1.0 - (h - self.ozone_centre_m).abs() / self.ozone_half_width_m).max(0.0);
        let mie_ext = (self.mie_scattering_per_m + self.mie_absorption_per_m) * mie;
        std::array::from_fn(|c| {
            self.rayleigh_per_m[c] * rayleigh + mie_ext + self.ozone_per_m[c] * ozone
        })
    }

    #[cfg(test)]
    /// Optical depth from radius `r_m` to the top along a ray of zenith cosine `mu`, by the
    /// quadrature of the transmittance pass: midpoints of a quadratic parameter.
    pub(crate) fn optical_depth_to_top(&self, r_m: f64, mu: f64, steps: u32) -> [f64; 3] {
        let disc = (self.top_m - r_m) * (self.top_m + r_m) + r_m * r_m * mu * mu;
        let t_max = (-r_m * mu + disc.max(0.0).sqrt()).max(0.0);
        let sin = (1.0 - mu * mu).max(0.0).sqrt();
        let n = f64::from(steps);
        let mut depth = [0.0; 3];
        for i in 0..steps {
            let s = (f64::from(i) + 0.5) / n;
            let t = s * s * t_max;
            let dt = 2.0 * s * t_max / n;
            let (x, z) = (t * sin, r_m + t * mu);
            let e = self.extinction_per_m((x * x + z * z).sqrt());
            for c in 0..3 {
                depth[c] += e[c] * dt;
            }
        }
        depth
    }

    #[cfg(test)]
    /// (r, mu) of a transmittance LUT texture coordinate, as `transmittance_r_mu`.
    pub(crate) fn transmittance_r_mu(&self, u: f64, v: f64) -> (f64, f64) {
        let unit = |u: f64, size: u32| {
            let size = f64::from(size);
            (u - 0.5 / size) * (size / (size - 1.0))
        };
        let x_mu = unit(u, TRANSMITTANCE_SIZE[0]);
        let x_r = unit(v, TRANSMITTANCE_SIZE[1]);
        let h = ((self.top_m - self.bottom_m) * (self.top_m + self.bottom_m)).sqrt();
        let rho = h * x_r;
        let r = (rho * rho + self.bottom_m * self.bottom_m).sqrt();
        let d_min = self.top_m - r;
        let d_max = rho + h;
        let d = d_min + x_mu * (d_max - d_min);
        let mu = if d > 0.0 {
            (h * h - rho * rho - d * d) / (2.0 * r * d)
        } else {
            1.0
        };
        (r, mu.clamp(-1.0, 1.0))
    }

    pub(crate) fn uniform(&self) -> AtmosphereUniform {
        let f = |x: f64| x as f32;
        let [rr, rg, rb] = self.rayleigh_per_m.map(f);
        let [or, og, ob] = self.ozone_per_m.map(f);
        let [gr, gg, gb] = self.ground_albedo.map(f);
        AtmosphereUniform {
            radii: [
                f(self.bottom_m),
                f(self.top_m),
                f(self.mie_asymmetry),
                f(self.aerial_range_m()),
            ],
            rayleigh: [rr, rg, rb, f(self.inv_rayleigh_height_per_m)],
            mie: [
                f(self.mie_scattering_per_m),
                f(self.mie_absorption_per_m),
                f(self.mie_scattering_per_m + self.mie_absorption_per_m),
                f(self.inv_mie_height_per_m),
            ],
            ozone: [or, og, ob, f(self.ozone_centre_m)],
            ground: [gr, gg, gb, f(self.ozone_half_width_m)],
        }
    }
}

/// Distinguishes [`FrameTargets`] instances so cached bind groups follow a resize.
static TARGETS_GENERATION: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_targets_generation() -> u64 {
    TARGETS_GENERATION.fetch_add(1, Ordering::Relaxed)
}

struct Lut {
    view: wgpu::TextureView,
}

impl Lut {
    fn new(device: &wgpu::Device, label: &str, size: [u32; 3], storage: bool) -> Self {
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC;
        if storage {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: size[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: if size[2] > 1 {
                wgpu::TextureDimension::D3
            } else {
                wgpu::TextureDimension::D2
            },
            format: LUT_FORMAT,
            usage,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
}

struct Luts {
    transmittance: Lut,
    multi_scattering: Lut,
    irradiance: Lut,
    sky_view: Lut,
    aerial: Lut,
}

/// The static LUTs are recomputed over two frames, transmittance then multiple scattering (which
/// reads it), so a parameter change never puts both passes in one frame's sky budget. The frame
/// between lights with the new transmittance and the previous multiple scattering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StaticLuts {
    Current,
    MultiScattering,
    Both,
}

struct LutPipelines {
    transmittance: wgpu::ComputePipeline,
    multi_scattering: wgpu::ComputePipeline,
    sky_view: wgpu::ComputePipeline,
    aerial: wgpu::ComputePipeline,
}

/// Bind group layouts and pipeline layouts, fixed for the pass's lifetime.
struct Layouts {
    shared: wgpu::BindGroupLayout,
    out_transmittance: wgpu::BindGroupLayout,
    out_multi_scattering: wgpu::BindGroupLayout,
    out_sky_view: wgpu::BindGroupLayout,
    out_aerial: wgpu::BindGroupLayout,
    transmittance: wgpu::PipelineLayout,
    multi_scattering: wgpu::PipelineLayout,
    sky_view: wgpu::PipelineLayout,
    aerial: wgpu::PipelineLayout,
    lighting_inputs: wgpu::BindGroupLayout,
    lighting: wgpu::PipelineLayout,
}

pub(crate) struct AtmospherePass {
    params: AtmosphereParams,
    radius_m: f64,
    /// Set by [`AtmospherePass::set_params`]; a frame whose planet radius differs is then logged.
    explicit_radius: bool,
    warned_radius: bool,
    /// Static LUTs still to recompute after a parameter change.
    dirty: StaticLuts,
    atmosphere_uniform: wgpu::Buffer,
    view_uniform: wgpu::Buffer,
    luts: Luts,
    layouts: Layouts,
    /// Group 0 for the transmittance pass (both sampled LUTs are placeholders), the multiple
    /// scattering pass (the multiple scattering LUT is a placeholder) and every later pass. A
    /// texture cannot be bound for sampling and storage in one dispatch.
    shared_transmittance: wgpu::BindGroup,
    shared_multi_scattering: wgpu::BindGroup,
    shared_full: wgpu::BindGroup,
    out_transmittance: wgpu::BindGroup,
    out_multi_scattering: wgpu::BindGroup,
    out_sky_view: wgpu::BindGroup,
    out_aerial: wgpu::BindGroup,
    pipelines: LutPipelines,
    lighting: wgpu::RenderPipeline,
    lighting_inputs: Option<(u64, wgpu::BindGroup)>,
    atmosphere_source: String,
    lighting_source: String,
}

fn sampled_entry(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: LUT_FORMAT,
            view_dimension: dimension,
        },
        count: None,
    }
}

fn gbuffer_entry(binding: u32, depth: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: if depth {
                wgpu::TextureSampleType::Depth
            } else {
                wgpu::TextureSampleType::Float { filterable: false }
            },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn layouts(device: &wgpu::Device) -> Layouts {
    use wgpu::TextureViewDimension::{D2, D3};
    let shared = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("atmosphere shared"),
        entries: &[
            uniform_entry(0),
            uniform_entry(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            sampled_entry(3, D2),
            sampled_entry(4, D2),
        ],
    });
    let output = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        })
    };
    let out_transmittance = output("transmittance out", &[storage_entry(0, D2)]);
    let out_multi_scattering = output(
        "multiple scattering out",
        &[storage_entry(1, D2), storage_entry(2, D2)],
    );
    let out_sky_view = output("sky view out", &[storage_entry(3, D2)]);
    let out_aerial = output("aerial perspective out", &[storage_entry(4, D3)]);
    let pipeline = |label, out: &wgpu::BindGroupLayout| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&shared), Some(out)],
            immediate_size: 0,
        })
    };
    let transmittance = pipeline("transmittance lut", &out_transmittance);
    let multi_scattering = pipeline("multiple scattering lut", &out_multi_scattering);
    let sky_view = pipeline("sky view lut", &out_sky_view);
    let aerial = pipeline("aerial perspective lut", &out_aerial);
    let lighting_inputs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lighting inputs"),
        entries: &[
            sampled_entry(0, D2),
            sampled_entry(1, D2),
            sampled_entry(2, D3),
            gbuffer_entry(3, false),
            gbuffer_entry(4, false),
            gbuffer_entry(5, true),
        ],
    });
    let lighting = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("lighting"),
        bind_group_layouts: &[Some(&shared), None, Some(&lighting_inputs)],
        immediate_size: 0,
    });
    Layouts {
        shared,
        out_transmittance,
        out_multi_scattering,
        out_sky_view,
        out_aerial,
        transmittance,
        multi_scattering,
        sky_view,
        aerial,
        lighting_inputs,
        lighting,
    }
}

/// Builds the four LUT pipelines from one atmosphere.wgsl source inside a validation scope.
fn lut_pipelines(
    device: &wgpu::Device,
    layouts: &Layouts,
    source: &str,
) -> Result<LutPipelines, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("atmosphere.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let build = |label, layout, entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipelines = LutPipelines {
        transmittance: build("transmittance lut", &layouts.transmittance, "transmittance"),
        multi_scattering: build(
            "multiple scattering lut",
            &layouts.multi_scattering,
            "multiple_scattering",
        ),
        sky_view: build("sky view lut", &layouts.sky_view, "sky_view"),
        aerial: build(
            "aerial perspective lut",
            &layouts.aerial,
            "aerial_perspective",
        ),
    };
    match pollster::block_on(scope.pop()) {
        None => Ok(pipelines),
        Some(error) => Err(error.to_string()),
    }
}

fn lighting_pipeline(
    device: &wgpu::Device,
    layouts: &Layouts,
    atmosphere_source: &str,
    lighting_source: &str,
) -> Result<wgpu::RenderPipeline, String> {
    let source = format!("{atmosphere_source}\n{lighting_source}");
    fullscreen_pipeline(
        device,
        &layouts.lighting,
        "lighting.wgsl",
        &source,
        HDR_FORMAT,
    )
}

impl AtmospherePass {
    pub(crate) fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        let atmosphere_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmosphere uniform"),
            size: size_of::<AtmosphereUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmosphere view uniform"),
            size: size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmosphere lut sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let [tw, th] = TRANSMITTANCE_SIZE;
        let [sw, sh] = SKY_VIEW_SIZE;
        let ms = MULTI_SCATTERING_SIZE;
        let luts = Luts {
            transmittance: Lut::new(device, "transmittance lut", [tw, th, 1], true),
            multi_scattering: Lut::new(device, "multiple scattering lut", [ms, ms, 1], true),
            irradiance: Lut::new(device, "sky irradiance lut", [ms, ms, 1], true),
            sky_view: Lut::new(device, "sky view lut", [sw, sh, 1], true),
            aerial: Lut::new(
                device,
                "aerial perspective lut",
                [AERIAL_SIZE, AERIAL_SIZE, AERIAL_SIZE],
                true,
            ),
        };
        let placeholder = Lut::new(device, "lut placeholder", [1, 1, 1], false);
        let layouts = layouts(device);

        let shared = |label, transmittance: &wgpu::TextureView, multi: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layouts.shared,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atmosphere_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: view_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(transmittance),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(multi),
                    },
                ],
            })
        };
        let shared_transmittance = shared(
            "atmosphere shared (transmittance pass)",
            &placeholder.view,
            &placeholder.view,
        );
        let shared_multi_scattering = shared(
            "atmosphere shared (multiple scattering pass)",
            &luts.transmittance.view,
            &placeholder.view,
        );
        let shared_full = shared(
            "atmosphere shared",
            &luts.transmittance.view,
            &luts.multi_scattering.view,
        );
        let out = |label, layout: &wgpu::BindGroupLayout, entries: &[(u32, &Lut)]| {
            let entries: Vec<_> = entries
                .iter()
                .map(|(binding, lut)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::TextureView(&lut.view),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
        };
        let out_transmittance = out(
            "transmittance out",
            &layouts.out_transmittance,
            &[(0, &luts.transmittance)],
        );
        let out_multi_scattering = out(
            "multiple scattering out",
            &layouts.out_multi_scattering,
            &[(1, &luts.multi_scattering), (2, &luts.irradiance)],
        );
        let out_sky_view = out(
            "sky view out",
            &layouts.out_sky_view,
            &[(3, &luts.sky_view)],
        );
        let out_aerial = out(
            "aerial perspective out",
            &layouts.out_aerial,
            &[(4, &luts.aerial)],
        );

        let pipelines = lut_pipelines(device, &layouts, ATMOSPHERE_WGSL)
            .expect("the built in atmosphere shader must compile");
        let lighting = lighting_pipeline(device, &layouts, ATMOSPHERE_WGSL, LIGHTING_WGSL)
            .expect("the built in lighting shader must compile");
        Self {
            params: AtmosphereParams::default(),
            radius_m: 10_000.0,
            explicit_radius: false,
            warned_radius: false,
            dirty: StaticLuts::Both,
            atmosphere_uniform,
            view_uniform,
            luts,
            layouts,
            shared_transmittance,
            shared_multi_scattering,
            shared_full,
            out_transmittance,
            out_multi_scattering,
            out_sky_view,
            out_aerial,
            pipelines,
            lighting,
            lighting_inputs: None,
            atmosphere_source: ATMOSPHERE_WGSL.to_owned(),
            lighting_source: LIGHTING_WGSL.to_owned(),
        }
    }
}

impl AtmospherePass {
    /// New optical depths and scale heights; the static LUTs are recomputed before the next
    /// frame. The radius sets the atmosphere's ground until a frame brings a different
    /// [`crate::PlanetView::radius_m`].
    pub(crate) fn set_params(&mut self, params: &AtmosphereParams, planet_radius_m: f64) {
        if let Err(error) = params.validate() {
            tracing::warn!("atmosphere parameters out of range, used as given: {error}");
        }
        self.params = *params;
        self.radius_m = planet_radius_m;
        self.explicit_radius = true;
        self.warned_radius = false;
        self.dirty = StaticLuts::Both;
    }

    pub(crate) fn coefficients(&self) -> Coefficients {
        Coefficients::new(&self.params, self.radius_m)
    }

    /// Writes the frame's uniforms. The frame's planet radius is the atmosphere's ground: when it
    /// differs from the one given to [`AtmospherePass::set_params`] the coefficients are derived
    /// again from it (logged once), so the LUTs always match the geometry the camera is in.
    pub(crate) fn prepare_frame(
        &mut self,
        queue: &wgpu::Queue,
        frame: &FrameInput,
        width: u32,
        height: u32,
    ) {
        let frame_radius_m = f64::from(frame.planet.radius_m);
        if frame_radius_m > 0.0 && ((frame_radius_m - self.radius_m) / self.radius_m).abs() > 1.0e-5
        {
            if self.explicit_radius && !self.warned_radius {
                tracing::warn!(
                    "frame planet radius {frame_radius_m} m differs from the atmosphere's {} m; \
                     the atmosphere follows the frame",
                    self.radius_m
                );
                self.warned_radius = true;
            }
            self.radius_m = frame_radius_m;
            self.dirty = StaticLuts::Both;
        }
        if self.dirty == StaticLuts::Both {
            let uniform = self.coefficients().uniform();
            queue.write_buffer(&self.atmosphere_uniform, 0, bytemuck::bytes_of(&uniform));
        }
        let view = ViewUniform::new(frame, width, height);
        queue.write_buffer(&self.view_uniform, 0, bytemuck::bytes_of(&view));
    }

    /// Encodes the LUT passes this frame needs: transmittance or multiple scattering in the two
    /// frames after a parameter change, sky view and aerial perspective always.
    pub(crate) fn encode_luts(&mut self, encoder: &mut wgpu::CommandEncoder, timer: &mut GpuTimer) {
        profiling::scope!("atmosphere luts");
        let groups = |size: u32, workgroup: u32| size.div_ceil(workgroup);
        if self.dirty == StaticLuts::Both {
            let [tw, th] = TRANSMITTANCE_SIZE;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transmittance lut"),
                timestamp_writes: timer.compute_writes("transmittance lut"),
            });
            pass.set_pipeline(&self.pipelines.transmittance);
            pass.set_bind_group(0, &self.shared_transmittance, &[]);
            pass.set_bind_group(1, &self.out_transmittance, &[]);
            pass.dispatch_workgroups(groups(tw, 8), groups(th, 8), 1);
            drop(pass);
            self.dirty = StaticLuts::MultiScattering;
        } else if self.dirty == StaticLuts::MultiScattering {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("multiple scattering lut"),
                timestamp_writes: timer.compute_writes("multiple scattering lut"),
            });
            pass.set_pipeline(&self.pipelines.multi_scattering);
            pass.set_bind_group(0, &self.shared_multi_scattering, &[]);
            pass.set_bind_group(1, &self.out_multi_scattering, &[]);
            pass.dispatch_workgroups(MULTI_SCATTERING_SIZE, MULTI_SCATTERING_SIZE, 1);
            drop(pass);
            self.dirty = StaticLuts::Current;
        }

        let [sw, sh] = SKY_VIEW_SIZE;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sky view lut"),
            timestamp_writes: timer.compute_writes("sky view lut"),
        });
        pass.set_pipeline(&self.pipelines.sky_view);
        pass.set_bind_group(0, &self.shared_full, &[]);
        pass.set_bind_group(1, &self.out_sky_view, &[]);
        pass.dispatch_workgroups(groups(sw, 8), groups(sh, 8), 1);
        drop(pass);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aerial perspective lut"),
            timestamp_writes: timer.compute_writes("aerial perspective lut"),
        });
        pass.set_pipeline(&self.pipelines.aerial);
        pass.set_bind_group(0, &self.shared_full, &[]);
        pass.set_bind_group(1, &self.out_aerial, &[]);
        pass.dispatch_workgroups(groups(AERIAL_SIZE, 8), groups(AERIAL_SIZE, 8), 1);
    }

    pub(crate) fn encode_lighting(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        targets: &FrameTargets,
        timer: &mut GpuTimer,
    ) {
        profiling::scope!("lighting pass");
        if self
            .lighting_inputs
            .as_ref()
            .is_none_or(|(generation, _)| *generation != targets.generation)
        {
            let view = |binding, view| wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting inputs"),
                layout: &self.layouts.lighting_inputs,
                entries: &[
                    view(0, &self.luts.irradiance.view),
                    view(1, &self.luts.sky_view.view),
                    view(2, &self.luts.aerial.view),
                    view(3, &targets.albedo),
                    view(4, &targets.normal),
                    view(5, &targets.depth),
                ],
            });
            self.lighting_inputs = Some((targets.generation, group));
        }
        let Some((_, inputs)) = &self.lighting_inputs else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.hdr,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timer.render_writes("lighting"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.lighting);
        pass.set_bind_group(0, &self.shared_full, &[]);
        pass.set_bind_group(2, inputs, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Rebuilds the LUT pipelines and the lighting pipeline (which includes atmosphere.wgsl) from
    /// new source. All or nothing: on any error the previous pipelines stay.
    pub(crate) fn reload_luts(
        &mut self,
        device: &wgpu::Device,
        source: &str,
    ) -> Result<(), String> {
        let pipelines = lut_pipelines(device, &self.layouts, source)?;
        let lighting = lighting_pipeline(device, &self.layouts, source, &self.lighting_source)?;
        self.pipelines = pipelines;
        self.lighting = lighting;
        self.atmosphere_source = source.to_owned();
        self.dirty = StaticLuts::Both;
        Ok(())
    }

    pub(crate) fn reload_lighting(
        &mut self,
        device: &wgpu::Device,
        source: &str,
    ) -> Result<(), String> {
        self.lighting = lighting_pipeline(device, &self.layouts, &self.atmosphere_source, source)?;
        self.lighting_source = source.to_owned();
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {

    use glam::{Mat4, Vec3};
    use vx_core::{Budgets, Projection};

    use super::*;
    use crate::post::PostPass;
    use crate::{GpuTimings, MoonLight, PlanetView, SunLight};

    /// Size in bytes and member offsets of a struct declared in WGSL, from naga's front end.
    pub(crate) fn wgsl_struct_layout(source: &str, name: &str) -> (u32, Vec<u32>) {
        let module = wgpu::naga::front::wgsl::parse_str(source).expect("WGSL parses");
        let ty = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("struct {name} not found"))
            .1;
        match &ty.inner {
            wgpu::naga::TypeInner::Struct { members, span } => {
                (*span, members.iter().map(|m| m.offset).collect())
            }
            other => panic!("{name} is not a struct: {other:?}"),
        }
    }

    #[test]
    fn uniform_layouts_match_the_wgsl() {
        let lighting = format!("{ATMOSPHERE_WGSL}\n{LIGHTING_WGSL}");
        for source in [ATMOSPHERE_WGSL, lighting.as_str()] {
            let (size, offsets) = wgsl_struct_layout(source, "Atmosphere");
            assert_eq!(size as usize, size_of::<AtmosphereUniform>());
            assert_eq!(
                offsets,
                [
                    std::mem::offset_of!(AtmosphereUniform, radii),
                    std::mem::offset_of!(AtmosphereUniform, rayleigh),
                    std::mem::offset_of!(AtmosphereUniform, mie),
                    std::mem::offset_of!(AtmosphereUniform, ozone),
                    std::mem::offset_of!(AtmosphereUniform, ground),
                ]
                .map(|o| o as u32)
            );
            let (size, offsets) = wgsl_struct_layout(source, "View");
            assert_eq!(size as usize, size_of::<ViewUniform>());
            assert_eq!(
                offsets,
                [
                    std::mem::offset_of!(ViewUniform, inv_view_proj),
                    std::mem::offset_of!(ViewUniform, camera),
                    std::mem::offset_of!(ViewUniform, sun_direction),
                    std::mem::offset_of!(ViewUniform, sun_illuminance),
                    std::mem::offset_of!(ViewUniform, moon),
                    std::mem::offset_of!(ViewUniform, night),
                    std::mem::offset_of!(ViewUniform, params),
                ]
                .map(|o| o as u32)
            );
        }
    }

    fn sum3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|c| a[c] + b[c])
    }

    #[test]
    fn coefficients_give_the_configured_zenith_optical_depths() {
        let params = AtmosphereParams::default();
        for radius_m in [10_000.0, 50_000.0] {
            let k = Coefficients::new(&params, radius_m);
            let depth = k.optical_depth_to_top(k.bottom_m, 1.0, TRANSMITTANCE_STEPS);
            let mie = params.mie_scattering_optical_depth + params.mie_absorption_optical_depth;
            let expected = sum3(params.rayleigh_optical_depth, params.ozone_optical_depth)
                .map(|tau| tau + mie);
            for c in 0..3 {
                let error = (depth[c] - expected[c]).abs() / expected[c];
                assert!(
                    error < 0.002,
                    "radius {radius_m}: channel {c} zenith depth {} vs {}",
                    depth[c],
                    expected[c]
                );
            }
            // The Mie layer alone, whose scale height is 1.2 % of the column, is the hardest
            // part of the quadrature; the total would hide its error.
            let mie_only = Coefficients {
                rayleigh_per_m: [0.0; 3],
                ozone_per_m: [0.0; 3],
                ..k
            };
            let depth = mie_only.optical_depth_to_top(k.bottom_m, 1.0, TRANSMITTANCE_STEPS)[0];
            assert!(
                (depth - mie).abs() / mie < 0.005,
                "Mie column {depth} vs {mie}"
            );
            // Earth's sea level coefficients (Hillaire 2020) times the shrink of the scale height.
            let shrink = 8_000.0 / (params.rayleigh_scale_height_fraction * params.top_m(radius_m));
            let earth_blue_per_m = 33.1e-6 * shrink;
            assert!((k.rayleigh_per_m[2] - earth_blue_per_m).abs() / earth_blue_per_m < 0.001);
        }
    }

    #[test]
    fn transmittance_quadrature_matches_a_thin_homogeneous_layer() {
        let tau = 0.3;
        let thickness_m = 100.0;
        let k = Coefficients {
            bottom_m: 1.0e9,
            top_m: 1.0e9 + thickness_m,
            rayleigh_per_m: [tau / thickness_m; 3],
            inv_rayleigh_height_per_m: 0.0,
            mie_scattering_per_m: 0.0,
            mie_absorption_per_m: 0.0,
            inv_mie_height_per_m: 0.0,
            mie_asymmetry: 0.8,
            ozone_per_m: [0.0; 3],
            ozone_centre_m: 0.0,
            ozone_half_width_m: 1.0,
            ground_albedo: [0.3; 3],
        };
        for zenith_deg in [0.0_f64, 30.0, 60.0, 75.0] {
            let mu = zenith_deg.to_radians().cos();
            let depth = k.optical_depth_to_top(k.bottom_m, mu, TRANSMITTANCE_STEPS)[0];
            let analytic = (-tau / mu).exp();
            let transmittance = (-depth).exp();
            assert!(
                (transmittance - analytic).abs() < 1.0e-4,
                "zenith {zenith_deg}: {transmittance} vs {analytic}"
            );
        }
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let mantissa = f32::from(bits & 0x3ff);
        sign * match exponent {
            0 => mantissa * 2.0_f32.powi(-24),
            31 => {
                if mantissa == 0.0 {
                    f32::INFINITY
                } else {
                    f32::NAN
                }
            }
            e => (1.0 + mantissa / 1024.0) * 2.0_f32.powi(e - 15),
        }
    }

    pub(crate) fn headless_device(features: wgpu::Features) -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let required_features = adapter.features() & features;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vx_render atmosphere test device"),
            required_features,
            ..Default::default()
        }))
        .ok()
    }

    /// Texels of an Rgba16Float texture (2D or 3D), row major, slices last.
    fn read_lut(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) -> Vec<[f32; 4]> {
        let texture = view.texture();
        let (w, h, d) = (
            texture.width(),
            texture.height(),
            texture.depth_or_array_layers(),
        );
        let bytes_per_row = (w * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lut readback"),
            size: u64::from(bytes_per_row * h * d),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            texture.size(),
        );
        queue.submit([encoder.finish()]);
        buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = buffer.get_mapped_range(..).expect("mapped").to_vec();
        let mut texels = Vec::with_capacity((w * h * d) as usize);
        for row in data.chunks_exact(bytes_per_row as usize) {
            for px in row[..(w * 8) as usize].chunks_exact(8) {
                texels.push(std::array::from_fn(|c| {
                    f16_to_f32(u16::from_le_bytes([px[2 * c], px[2 * c + 1]]))
                }));
            }
        }
        texels
    }

    /// A planet of `radius_m` with the camera `altitude_m` above its north pole (+Y up), the
    /// camera looking along `forward`.
    pub(crate) fn frame_input(
        radius_m: f32,
        altitude_m: f32,
        sun: Vec3,
        forward: Vec3,
        width: u32,
        height: u32,
    ) -> FrameInput<'static> {
        let projection = Projection {
            fov_y_rad: 60.0_f32.to_radians(),
            aspect: width as f32 / height as f32,
            near_m: 0.1,
        };
        let up = if forward.cross(Vec3::Y).length() < 1.0e-3 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let f = forward.normalize();
        let right = f.cross(up).normalize();
        let view = Mat4::from_mat3(glam::Mat3::from_cols(right, right.cross(f), -f).transpose());
        FrameInput {
            view_proj: projection.matrix() * view,
            camera_offset_m: Vec3::ZERO,
            planet: PlanetView {
                centre_m: Vec3::new(0.0, -(radius_m + altitude_m), 0.0),
                radius_m,
                occluder_radius_m: radius_m * 0.9,
            },
            sun: SunLight {
                direction: sun.normalize(),
                ..Default::default()
            },
            moon: None,
            exposure_ev: 0.0,
            egui: None,
        }
    }

    fn sun_at_elevation(elevation_deg: f32) -> Vec3 {
        let e = elevation_deg.to_radians();
        Vec3::new(e.cos(), e.sin(), 0.0)
    }

    fn run_luts(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut AtmospherePass,
        frame: &FrameInput,
    ) {
        let mut timer = GpuTimer::new(device, queue);
        // Two frames, so static LUTs pending after a parameter change are both done.
        for _ in 0..2 {
            pass.prepare_frame(queue, frame, 64, 64);
            let mut encoder = device.create_command_encoder(&Default::default());
            pass.encode_luts(&mut encoder, &mut timer);
            queue.submit([encoder.finish()]);
        }
    }

    /// Row of the sky view LUT whose texel centre is the last one above the horizon.
    fn last_row_above_horizon() -> usize {
        let h = SKY_VIEW_SIZE[1] as f64;
        (0..SKY_VIEW_SIZE[1] as usize)
            .rev()
            .find(|&row| {
                let v = (row as f64 + 0.5) / h;
                (v - 0.5 / h) * (h / (h - 1.0)) < 0.5
            })
            .expect("a row above the horizon")
    }

    fn luminance(c: [f32; 4]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    #[test]
    fn atmosphere_luts_on_the_gpu() {
        let Some((device, queue)) = headless_device(wgpu::Features::empty()) else {
            eprintln!("no GPU adapter available, skipping atmosphere_luts_on_the_gpu");
            return;
        };
        let radius_m = 10_000.0_f32;
        let mut pass = AtmospherePass::new(&device, &queue);
        pass.set_params(&AtmosphereParams::default(), f64::from(radius_m));

        let noon = frame_input(radius_m, 2.0, Vec3::Y, Vec3::X, 64, 64);
        run_luts(&device, &queue, &mut pass, &noon);
        let transmittance = read_lut(&device, &queue, &pass.luts.transmittance.view);
        let [tw, th] = TRANSMITTANCE_SIZE.map(|x| x as usize);
        let k = pass.coefficients();

        // Column 0 looks straight up: 1 at the top row, decreasing towards the ground.
        let column: Vec<f32> = (0..th).map(|row| transmittance[row * tw][2]).collect();
        assert!(
            column[th - 1] > 0.999,
            "top of the atmosphere: {}",
            column[th - 1]
        );
        assert!(
            column.windows(2).all(|w| w[0] <= w[1] + 1.0e-3),
            "transmittance must not decrease with altitude: {column:?}"
        );
        // The GPU table against the CPU reference at a spread of texels (f16 storage).
        for (x, y) in [(0, 0), (40, 3), (128, 20), (200, 40), (250, 63), (255, 0)] {
            let (r, mu) =
                k.transmittance_r_mu((x as f64 + 0.5) / tw as f64, (y as f64 + 0.5) / th as f64);
            let depth = k.optical_depth_to_top(r, mu, TRANSMITTANCE_STEPS);
            let gpu = transmittance[y * tw + x];
            for c in 0..3 {
                let cpu = (-depth[c]).exp() as f32;
                assert!(
                    (gpu[c] - cpu).abs() < 2.0e-3 + 0.01 * cpu,
                    "texel ({x}, {y}) channel {c}: gpu {} cpu {cpu}",
                    gpu[c]
                );
            }
        }

        let multi = read_lut(&device, &queue, &pass.luts.multi_scattering.view);
        let irradiance = read_lut(&device, &queue, &pass.luts.irradiance.view);
        let ms = MULTI_SCATTERING_SIZE as usize;
        // Sun at the zenith (last column) at the ground (row 0).
        let noon_ms = multi[ms - 1];
        let noon_irradiance = irradiance[ms - 1];
        assert!(noon_ms.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(
            noon_ms[2] > noon_ms[0],
            "multiple scattering is blue: {noon_ms:?}"
        );
        // Sky irradiance on a horizontal surface at noon is about a tenth to a quarter of the
        // sun's (clear sky, Earth like optical depths), bluer than the sun.
        assert!(
            noon_irradiance[2] > 0.08 && noon_irradiance[2] < 0.4,
            "sky irradiance at noon {noon_irradiance:?}"
        );
        assert!(noon_irradiance[2] > noon_irradiance[1] && noon_irradiance[1] > noon_irradiance[0]);
        // Night: sun far below the horizon (first column).
        assert!(irradiance[0].iter().take(3).all(|v| *v < 1.0e-3));

        let sky_noon = read_lut(&device, &queue, &pass.luts.sky_view.view);
        let [sw, _] = SKY_VIEW_SIZE.map(|x| x as usize);
        let zenith = sky_noon[0];
        assert!(
            zenith[2] > zenith[1] && zenith[1] > zenith[0] && zenith[0] > 0.0,
            "zenith sky is blue: {zenith:?}"
        );
        let horizon_row = last_row_above_horizon();
        let noon_horizon = sky_noon[horizon_row * sw];

        let low = frame_input(radius_m, 2.0, sun_at_elevation(2.0), Vec3::X, 64, 64);
        run_luts(&device, &queue, &mut pass, &low);
        let sky_low = read_lut(&device, &queue, &pass.luts.sky_view.view);
        let low_horizon = sky_low[horizon_row * sw];
        let redness = |c: [f32; 4]| c[0] / c[2];
        eprintln!(
            "horizon towards the sun, noon {noon_horizon:?} r/b {:.3}, sun 2 deg {low_horizon:?} r/b {:.3}",
            redness(noon_horizon),
            redness(low_horizon)
        );
        assert!(
            redness(low_horizon) > 1.3 * redness(noon_horizon),
            "a low sun reddens the horizon"
        );

        // From 100 km the planet (10 km) is a disc of about 5 degrees; the atmosphere is a thin
        // blue ring just above the planet's horizon, black space above it.
        let tilted = Vec3::new(1.0, 1.0, 0.0).normalize();
        let orbit = frame_input(radius_m, 100_000.0, tilted, Vec3::X, 64, 64);
        run_luts(&device, &queue, &mut pass, &orbit);
        let sky_orbit = read_lut(&device, &queue, &pass.luts.sky_view.view);
        let column: Vec<[f32; 4]> = (0..=horizon_row).map(|row| sky_orbit[row * sw]).collect();
        let peak = column
            .iter()
            .copied()
            .max_by(|a, b| luminance(*a).total_cmp(&luminance(*b)))
            .expect("rows");
        let peak_row = column.iter().position(|c| *c == peak).expect("peak row");
        eprintln!("orbit limb: peak {peak:?} at row {peak_row} of {horizon_row}");
        assert!(luminance(peak) > 1.0e-3, "the limb is lit: {peak:?}");
        assert!(peak[2] > peak[0], "the limb is blue: {peak:?}");
        assert!(peak_row + 6 >= horizon_row, "the rim hugs the horizon");
        assert!(
            column[..horizon_row / 2]
                .iter()
                .all(|c| luminance(*c) < 1.0e-5),
            "space above the limb is black"
        );

        // Aerial perspective from the ground at noon: more haze with distance.
        let aerial = read_lut(&device, &queue, &pass.luts.aerial.view);
        assert!(aerial.iter().all(|c| c.iter().all(|v| v.is_finite())));
        run_luts(&device, &queue, &mut pass, &noon);
        let aerial = read_lut(&device, &queue, &pass.luts.aerial.view);
        let a = AERIAL_SIZE as usize;
        let centre = |slice: usize| aerial[slice * a * a + (a / 2) * a + a / 2];
        let near = centre(1);
        let far = centre(a - 1);
        assert!(
            far[3] > near[3] && far[3] <= 1.0,
            "opacity grows: {near:?} {far:?}"
        );
        assert!(far[2] > near[2], "in-scattering grows: {near:?} {far:?}");
    }

    /// Fills the G-buffer for the full frame tests: a sphere planet with a rolling normal
    /// pattern (grass, rock on steep normals), a constant distance surface, or only sky.
    const SCENE_WGSL: &str = r"
struct Scene {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    // xyz: planet centre relative to the camera. w: radius.
    planet: vec4<f32>,
    // x, y: size. z: mode (0 sphere, 1 constant distance, 2 sky, 3 plain sphere). w: distance
    // for mode 1.
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> scene: Scene;

struct GBuffer {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn depth_of(p: vec3<f32>) -> f32 {
    let clip = scene.view_proj * vec4<f32>(p, 1.0);
    return clip.z / clip.w;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> GBuffer {
    let uv = pos.xy / scene.params.xy;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = scene.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let dir = normalize(near.xyz / near.w);
    let centre = scene.planet.xyz;
    let up = normalize(-centre);
    var out: GBuffer;
    out.albedo = vec4<f32>(0.0);
    out.normal = vec4<f32>(0.0);
    out.depth = 0.0;
    let grass = vec3<f32>(0.12, 0.25, 0.03);
    let mode = u32(scene.params.z + 0.5);
    if mode == 2u {
        return out;
    }
    if mode == 1u {
        let p = dir * scene.params.w;
        out.albedo = vec4<f32>(grass, 1.0);
        out.normal = vec4<f32>(up, 0.0);
        out.depth = depth_of(p);
        return out;
    }
    let oc = -centre;
    let r = length(oc);
    let radius = scene.planet.w;
    let b = dot(oc, dir);
    let c = (r - radius) * (r + radius);
    let disc = b * b - c;
    if disc < 0.0 || -b + sqrt(disc) < 0.0 {
        return out;
    }
    var t = -b - sqrt(disc);
    if t < 0.0 {
        t = -b + sqrt(disc);
    }
    let p = t * dir;
    let local = p - centre;
    let n0 = normalize(local);
    let q = local / 120.0;
    let bump = vec3<f32>(cos(q.x * 1.7 + sin(q.z)), cos(q.y * 1.3 + sin(q.x * 0.7)), cos(q.z * 1.1 + sin(q.y))) * 0.6;
    var n = normalize(n0 + bump - n0 * dot(bump, n0));
    var albedo = grass;
    if dot(n, n0) < 0.8 {
        albedo = vec3<f32>(0.42, 0.40, 0.34);
    }
    if mode == 3u {
        n = n0;
        albedo = grass;
    }
    out.albedo = vec4<f32>(albedo, 1.0);
    out.normal = vec4<f32>(n, 0.0);
    out.depth = depth_of(p);
    return out;
}
";

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Pod, Zeroable)]
    struct SceneUniform {
        inv_view_proj: [[f32; 4]; 4],
        view_proj: [[f32; 4]; 4],
        planet: [f32; 4],
        params: [f32; 4],
    }

    #[derive(Debug, Clone, Copy)]
    pub(crate) enum Surface {
        Sphere,
        PlainSphere,
        Distance(f32),
        Sky,
    }

    pub(crate) struct Harness {
        pub device: wgpu::Device,
        pub queue: wgpu::Queue,
        pub atmosphere: AtmospherePass,
        post: PostPass,
        pub timer: GpuTimer,
        targets: FrameTargets,
        display: wgpu::TextureView,
        scene_uniform: wgpu::Buffer,
        scene_group: wgpu::BindGroup,
        scene_pipeline: wgpu::RenderPipeline,
        pub width: u32,
        pub height: u32,
    }

    impl Harness {
        pub(crate) fn new(width: u32, height: u32) -> Option<Self> {
            let (device, queue) = headless_device(wgpu::Features::TIMESTAMP_QUERY)?;
            let atmosphere = AtmospherePass::new(&device, &queue);
            let post = PostPass::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
            let timer = GpuTimer::new(&device, &queue);
            let targets = FrameTargets::new(&device, width, height);
            let display = device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("display"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&Default::default());
            let scene_uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scene uniform"),
                size: size_of::<SceneUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene"),
                entries: &[uniform_entry(0)],
            });
            let scene_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_uniform.as_entire_binding(),
                }],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SCENE_WGSL)),
            });
            let target = |format| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            };
            let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("scene"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[
                        target(crate::GBUFFER_ALBEDO_FORMAT),
                        target(crate::GBUFFER_NORMAL_FORMAT),
                    ],
                }),
                multiview_mask: None,
                cache: None,
            });
            Some(Self {
                device,
                queue,
                atmosphere,
                post,
                timer,
                targets,
                display,
                scene_uniform,
                scene_group,
                scene_pipeline,
                width,
                height,
            })
        }

        /// Renders one frame and waits for it; returns that frame's GPU timings.
        pub(crate) fn render(
            &mut self,
            frame: &FrameInput,
            surface: Surface,
        ) -> Option<GpuTimings> {
            let (mode, distance) = match surface {
                Surface::Sphere => (0.0, 0.0),
                Surface::PlainSphere => (3.0, 0.0),
                Surface::Distance(d) => (1.0, d),
                Surface::Sky => (2.0, 0.0),
            };
            let scene = SceneUniform {
                inv_view_proj: frame.view_proj.inverse().to_cols_array_2d(),
                view_proj: frame.view_proj.to_cols_array_2d(),
                planet: frame
                    .planet
                    .centre_m
                    .extend(frame.planet.radius_m)
                    .to_array(),
                params: [self.width as f32, self.height as f32, mode, distance],
            };
            self.queue
                .write_buffer(&self.scene_uniform, 0, bytemuck::bytes_of(&scene));
            self.timer.begin_frame(&self.device);
            self.atmosphere
                .prepare_frame(&self.queue, frame, self.width, self.height);
            self.post.prepare_frame(&self.queue, frame);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let attachment = |view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene"),
                    color_attachments: &[
                        attachment(&self.targets.albedo),
                        attachment(&self.targets.normal),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.targets.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(crate::DEPTH_CLEAR),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: self.timer.render_writes("gbuffer (test scene)"),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.scene_pipeline);
                pass.set_bind_group(0, &self.scene_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.atmosphere.encode_luts(&mut encoder, &mut self.timer);
            self.atmosphere.encode_lighting(
                &self.device,
                &mut encoder,
                &self.targets,
                &mut self.timer,
            );
            self.post.encode_tonemap(
                &self.device,
                &mut encoder,
                &self.targets,
                &self.display,
                &mut self.timer,
            );
            self.timer.end_frame(&mut encoder);
            self.queue.submit([encoder.finish()]);
            self.timer.after_submit();
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            self.timer.poll(&self.device);
            self.timer.timings().filter(|t| t.frames_old == 0)
        }

        /// The tonemapped frame as sRGB encoded RGBA8.
        pub(crate) fn read_display(&self) -> Vec<u8> {
            let texture = self.display.texture();
            let bytes_per_row = (self.width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("display readback"),
                size: u64::from(bytes_per_row * self.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(self.height),
                    },
                },
                texture.size(),
            );
            self.queue.submit([encoder.finish()]);
            buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            let data = buffer.get_mapped_range(..).expect("mapped");
            data.chunks_exact(bytes_per_row as usize)
                .flat_map(|row| row[..(self.width * 4) as usize].to_vec())
                .collect()
        }

        pub(crate) fn read_hdr(&self) -> Vec<[f32; 4]> {
            read_lut(&self.device, &self.queue, &self.targets.hdr)
        }
    }

    /// Ground at 30 m, sky and terrain, sun 50 degrees up behind the right shoulder.
    fn ground_view(
        width: u32,
        height: u32,
        sun_elevation_deg: f32,
        forward: Vec3,
    ) -> FrameInput<'static> {
        let e = sun_elevation_deg.to_radians();
        let sun = Vec3::new(0.5 * e.cos(), e.sin(), 0.85 * e.cos());
        frame_input(10_000.0, 30.0, sun, forward, width, height)
    }

    #[test]
    fn lighting_and_tonemap_produce_a_blue_sky_and_lit_terrain() {
        let Some(mut h) = Harness::new(320, 180) else {
            eprintln!(
                "no GPU adapter available, skipping lighting_and_tonemap_produce_a_blue_sky_and_lit_terrain"
            );
            return;
        };
        let forward = Vec3::new(1.0, -0.05, 0.0);
        let noon = ground_view(h.width, h.height, 60.0, forward);
        h.render(&noon, Surface::Sphere);
        let hdr = h.read_hdr();
        assert!(
            hdr.iter()
                .all(|c| c.iter().all(|v| v.is_finite() && *v >= 0.0))
        );
        let display = h.read_display();
        let px = |x: u32, y: u32| {
            let i = ((y * h.width + x) * 4) as usize;
            [display[i], display[i + 1], display[i + 2]]
        };
        let sky = px(h.width / 2, 2);
        assert!(sky[2] > sky[1] && sky[1] > sky[0], "sky is blue: {sky:?}");
        let ground = px(h.width / 2, h.height - 2);
        assert!(
            ground[1] > ground[0] && ground[1] > ground[2],
            "grass is green: {ground:?}"
        );
        assert!(ground[1] > 40, "grass is lit: {ground:?}");

        let night = ground_view(h.width, h.height, -30.0, forward);
        h.render(&night, Surface::Sphere);
        let display = h.read_display();
        let sky = [display[8], display[9], display[10]];
        assert!(
            sky[2] > 0 && sky[2] >= sky[0],
            "night sky is dark blue, not black: {sky:?}"
        );
    }

    struct Measurement {
        /// Median of each pass over the frames it ran in.
        passes: Vec<(&'static str, f64)>,
        /// Median and largest per frame sky and lighting time.
        sky_median_ms: f64,
        sky_max_ms: f64,
    }

    fn measure(
        h: &mut Harness,
        frame: &FrameInput,
        surface: Surface,
        frames: usize,
        static_luts: bool,
    ) -> Measurement {
        let mut samples: Vec<(&'static str, Vec<f64>)> = Vec::new();
        let mut sky = Vec::new();
        for i in 0..frames + 5 {
            if static_luts && h.atmosphere.dirty == StaticLuts::Current {
                h.atmosphere.dirty = StaticLuts::Both;
            }
            let Some(timings) = h.render(frame, surface) else {
                continue;
            };
            if i < 5 {
                continue;
            }
            sky.push(timings.sky_and_lighting_ms());
            for (name, ms) in timings.passes {
                match samples.iter_mut().find(|(n, _)| *n == name) {
                    Some((_, v)) => v.push(ms),
                    None => samples.push((name, vec![ms])),
                }
            }
        }
        sky.sort_by(f64::total_cmp);
        Measurement {
            passes: samples
                .into_iter()
                .map(|(name, mut v)| {
                    v.sort_by(f64::total_cmp);
                    (name, v[v.len() / 2])
                })
                .collect(),
            sky_median_ms: sky.get(sky.len() / 2).copied().unwrap_or(f64::NAN),
            sky_max_ms: sky.last().copied().unwrap_or(f64::NAN),
        }
    }

    #[test]
    fn gpu_pass_times_at_1080p() {
        let Some(mut h) = Harness::new(1920, 1080) else {
            eprintln!("no GPU adapter available, skipping gpu_pass_times_at_1080p");
            return;
        };
        if !h.timer.supported() {
            eprintln!("no timestamp queries on this adapter, skipping gpu_pass_times_at_1080p");
            return;
        }
        let (w, hh) = (h.width, h.height);
        let level = Vec3::new(1.0, -0.05, 0.0);
        let scenes = [
            (
                "ground, half sky",
                ground_view(w, hh, 50.0, level),
                Surface::Sphere,
                false,
            ),
            (
                "ground, static LUTs alternating every frame",
                ground_view(w, hh, 50.0, level),
                Surface::Sphere,
                true,
            ),
            (
                "all sky",
                ground_view(w, hh, 50.0, Vec3::new(1.0, 1.0, 0.0)),
                Surface::Sky,
                false,
            ),
            (
                "all terrain at 2 km (froxels)",
                ground_view(w, hh, 50.0, level),
                Surface::Distance(2_000.0),
                false,
            ),
            (
                "1.5 km, above the 1 km atmosphere, planet fills the view (march)",
                frame_input(
                    10_000.0,
                    1_500.0,
                    Vec3::new(0.6, 0.8, 0.0),
                    Vec3::new(0.3, -1.0, 0.0),
                    w,
                    hh,
                ),
                Surface::Sphere,
                false,
            ),
            (
                "30 km, froxels with a per pixel rim",
                frame_input(
                    10_000.0,
                    30_000.0,
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(0.0, -1.0, 0.05),
                    w,
                    hh,
                ),
                Surface::Sphere,
                false,
            ),
            (
                "69 km, LUT to march crossfade (atmosphere radius 8 deg)",
                frame_input(
                    10_000.0,
                    69_000.0,
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(0.0, -1.0, 0.05),
                    w,
                    hh,
                ),
                Surface::Sphere,
                false,
            ),
            (
                "100 km orbit, planet and limb",
                frame_input(
                    10_000.0,
                    100_000.0,
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(0.0, -1.0, 0.05),
                    w,
                    hh,
                ),
                Surface::Sphere,
                false,
            ),
        ];
        let mut worst: f64 = 0.0;
        eprintln!(
            "GPU pass medians at {w}x{hh}, 10 km planet, default atmosphere \
             (render passes exclusive, compute passes raw):"
        );
        for (name, frame, surface, static_luts) in &scenes {
            let m = measure(&mut h, frame, *surface, 30, *static_luts);
            assert!(!m.passes.is_empty(), "timings for {name}");
            assert!(
                GpuTimings::SKY_AND_LIGHTING_PASSES
                    .iter()
                    .skip(2)
                    .all(|p| m.passes.iter().any(|(n, _)| n == p)),
                "every per frame sky pass is timed: {:?}",
                m.passes
            );
            worst = worst.max(m.sky_median_ms);
            let list: Vec<String> = m
                .passes
                .iter()
                .map(|(n, ms)| format!("{n} {ms:.3}"))
                .collect();
            eprintln!(
                "  {name}: {}; sky and lighting per frame median {:.3} ms, max {:.3} ms",
                list.join(", "),
                m.sky_median_ms,
                m.sky_max_ms
            );
        }
        eprintln!(
            "worst median sky and lighting {worst:.3} ms against the {} ms budget{}",
            Budgets::SKY_AND_LIGHTING_MS,
            if worst > Budgets::SKY_AND_LIGHTING_MS {
                " (OVER)"
            } else {
                ""
            }
        );
    }

    #[test]
    fn broken_shaders_keep_the_previous_pipelines() {
        let Some(mut h) = Harness::new(64, 64) else {
            eprintln!(
                "no GPU adapter available, skipping broken_shaders_keep_the_previous_pipelines"
            );
            return;
        };
        assert!(h.atmosphere.reload_luts(&h.device, "fn broken(").is_err());
        let missing_entry = ATMOSPHERE_WGSL.replace("fn sky_view(", "fn sky_view_renamed(");
        assert!(h.atmosphere.reload_luts(&h.device, &missing_entry).is_err());
        assert!(
            h.atmosphere
                .reload_lighting(&h.device, "@fragment fn fs_main() -> nope {}")
                .is_err()
        );
        assert!(h.post.reload(&h.device, "garbage").is_err());
        assert!(h.atmosphere.reload_luts(&h.device, ATMOSPHERE_WGSL).is_ok());
        assert!(
            h.atmosphere
                .reload_lighting(&h.device, LIGHTING_WGSL)
                .is_ok()
        );
        assert!(h.post.reload(&h.device, crate::post::TONEMAP_WGSL).is_ok());
        let frame = ground_view(h.width, h.height, 40.0, Vec3::new(1.0, 0.3, 0.0));
        h.render(&frame, Surface::Sphere);
        let display = h.read_display();
        let sky = &display[..3];
        assert!(
            sky[2] > sky[0],
            "the frame after failed reloads still renders: {sky:?}"
        );
    }

    /// Writes preview images (binary PPM) of a few views to `VX_PREVIEW_DIR`, for judging the look
    /// against the reference images. `cargo test -p vx_render preview -- --ignored`.
    #[test]
    #[ignore]
    fn preview_images() {
        let Ok(dir) = std::env::var("VX_PREVIEW_DIR") else {
            eprintln!("set VX_PREVIEW_DIR to write preview images");
            return;
        };
        let Some(mut h) = Harness::new(1280, 720) else {
            return;
        };
        let (w, hh) = (h.width, h.height);
        let level = Vec3::new(1.0, -0.08, 0.3);
        let moon = MoonLight {
            centre_m: Vec3::new(3.0e8, 1.2e8, 1.0e8),
            radius_m: 1.7e6,
            albedo: 0.12,
        };
        let mut night = ground_view(w, hh, -25.0, Vec3::new(1.0, 0.25, 0.4));
        night.moon = Some(moon);
        let mut dusk_moon = ground_view(w, hh, 4.0, Vec3::new(1.0, 0.25, 0.4));
        dusk_moon.moon = Some(moon);
        let ground = |sun: f32, forward: Vec3| (ground_view(w, hh, sun, forward), Surface::Sphere);
        let at = |altitude: f32, sun: Vec3, forward: Vec3| {
            (
                frame_input(10_000.0, altitude, sun, forward, w, hh),
                Surface::PlainSphere,
            )
        };
        let sunset = frame_input(
            10_000.0,
            30.0,
            sun_at_elevation(3.0),
            Vec3::new(1.0, -0.02, 0.15),
            w,
            hh,
        );
        let views = [
            ("noon", ground(60.0, level)),
            ("afternoon", ground(25.0, level)),
            ("sunset", (sunset, Surface::Sphere)),
            ("dusk-moon", (dusk_moon, Surface::Sphere)),
            ("night", (night, Surface::Sphere)),
            (
                "high",
                at(600.0, Vec3::new(0.5, 0.8, 0.3), Vec3::new(1.0, -0.35, 0.0)),
            ),
            (
                "orbit-2km",
                at(2_000.0, Vec3::new(1.0, 0.6, 0.2), Vec3::new(1.0, -0.6, 0.0)),
            ),
            (
                "orbit-30km",
                at(
                    30_000.0,
                    Vec3::new(1.0, 0.6, 0.2),
                    Vec3::new(0.0, -1.0, 0.1),
                ),
            ),
            (
                "orbit-69km",
                at(
                    69_000.0,
                    Vec3::new(1.0, 0.6, 0.2),
                    Vec3::new(0.0, -1.0, 0.1),
                ),
            ),
            (
                "orbit-100km",
                at(
                    100_000.0,
                    Vec3::new(1.0, 0.3, 0.4),
                    Vec3::new(0.0, -1.0, 0.0),
                ),
            ),
            ("noon-hazy", ground(60.0, level)),
            ("afternoon-hazy", ground(25.0, level)),
        ];
        let hazy = AtmosphereParams {
            mie_scattering_optical_depth: 0.04,
            mie_absorption_optical_depth: 0.004,
            ..AtmosphereParams::default()
        };
        let only = std::env::var("VX_PREVIEW_ONLY").ok();
        for (name, (frame, surface)) in views {
            if only.as_deref().is_some_and(|o| !name.contains(o)) {
                continue;
            }
            if name.ends_with("hazy") {
                h.atmosphere.set_params(&hazy, 10_000.0);
            }
            // Two frames, so static LUTs pending after a parameter change are both done.
            h.render(&frame, surface);
            h.render(&frame, surface);
            let rgba = h.read_display();
            let mut ppm = format!("P6\n{w} {hh}\n255\n").into_bytes();
            for px in rgba.chunks_exact(4) {
                ppm.extend_from_slice(&px[..3]);
            }
            let path = std::path::Path::new(&dir).join(format!("{name}.ppm"));
            std::fs::write(&path, ppm).expect("write preview");
            eprintln!("wrote {}", path.display());
        }
    }
}
