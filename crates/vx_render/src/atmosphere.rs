//! The atmosphere LUTs and the lighting pass (D8, D25).
//!
//! Placeholder from the M2 interface commit: no LUTs yet, and the lighting pass lights the
//! G-buffer with a Lambert sun and a gradient sky so every workstream starts from a running frame.
//! Workstream F replaces both with Hillaire 2020.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use vx_core::AtmosphereParams;

use crate::post::{fullscreen_pipeline, texture_entry, uniform_entry};
use crate::{FrameInput, FrameTargets, HDR_FORMAT};

pub(crate) const ATMOSPHERE_WGSL: &str = include_str!("../../../assets/shaders/atmosphere.wgsl");
pub(crate) const LIGHTING_WGSL: &str = include_str!("../../../assets/shaders/lighting.wgsl");

/// Matches `struct Lighting` in `assets/shaders/lighting.wgsl` (engineering rule 9).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct LightingUniform {
    /// Inverse of the rotation only view projection: clip space to camera relative metres.
    pub inv_view_proj: [[f32; 4]; 4],
    /// xyz: unit direction towards the sun. w: angular radius of the disc.
    pub sun_direction: [f32; 4],
    /// rgb: sun illuminance above the atmosphere. w: reserved.
    pub sun_illuminance: [f32; 4],
    /// xyz: planet centre relative to the camera in metres. w: mean radius.
    pub planet: [f32; 4],
    /// x: atmosphere top above the mean radius in metres. yzw: reserved.
    pub params: [f32; 4],
}

const _: () = assert!(size_of::<LightingUniform>() == 128);

pub(crate) struct AtmospherePass {
    params: AtmosphereParams,
    planet_radius_m: f64,
    uniform: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    lighting: wgpu::RenderPipeline,
}

impl AtmospherePass {
    pub(crate) fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lighting uniform"),
            size: size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting"),
            entries: &[
                uniform_entry(0),
                texture_entry(1, false),
                texture_entry(2, false),
                texture_entry(3, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lighting"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let lighting = fullscreen_pipeline(
            device,
            &pipeline_layout,
            "lighting.wgsl",
            LIGHTING_WGSL,
            HDR_FORMAT,
        )
        .expect("the built in lighting shader must compile");
        compile_check(device, ATMOSPHERE_WGSL)
            .expect("the built in atmosphere shader must compile");
        Self {
            params: AtmosphereParams::default(),
            planet_radius_m: 10_000.0,
            uniform,
            layout,
            pipeline_layout,
            lighting,
        }
    }

    pub(crate) fn set_params(&mut self, params: &AtmosphereParams, planet_radius_m: f64) {
        self.params = *params;
        self.planet_radius_m = planet_radius_m;
    }

    pub(crate) fn reload_luts(
        &mut self,
        device: &wgpu::Device,
        source: &str,
    ) -> Result<(), String> {
        compile_check(device, source)
    }

    pub(crate) fn reload_lighting(
        &mut self,
        device: &wgpu::Device,
        source: &str,
    ) -> Result<(), String> {
        self.lighting = fullscreen_pipeline(
            device,
            &self.pipeline_layout,
            "lighting.wgsl",
            source,
            HDR_FORMAT,
        )?;
        Ok(())
    }

    pub(crate) fn prepare_frame(&self, queue: &wgpu::Queue, frame: &FrameInput) {
        let uniform = LightingUniform {
            inv_view_proj: frame.view_proj.inverse().to_cols_array_2d(),
            sun_direction: frame
                .sun
                .direction
                .normalize_or_zero()
                .extend(frame.sun.angular_radius_rad)
                .to_array(),
            sun_illuminance: frame.sun.illuminance.extend(0.0).to_array(),
            planet: frame
                .planet
                .centre_m
                .extend(frame.planet.radius_m)
                .to_array(),
            params: [
                self.params.top_m(self.planet_radius_m) as f32,
                0.0,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Recomputes the LUTs that change this frame. Placeholder: none yet.
    pub(crate) fn encode_luts(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

    pub(crate) fn encode_lighting(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        targets: &FrameTargets,
    ) {
        profiling::scope!("lighting pass");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&targets.albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&targets.normal),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&targets.depth),
                },
            ],
        });
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
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.lighting);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Compiles a WGSL module inside a validation error scope.
fn compile_check(device: &wgpu::Device, source: &str) -> Result<(), String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("atmosphere.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    match pollster::block_on(scope.pop()) {
        None => Ok(()),
        Some(error) => Err(error.to_string()),
    }
}
