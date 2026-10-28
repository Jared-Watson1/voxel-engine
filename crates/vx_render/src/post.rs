//! Tonemapping from the HDR target into the swapchain (D25), and the fullscreen pipeline helper
//! the lighting pass shares.
//!
//! The curve is Khronos PBR Neutral (Khronos 3D Commerce, 2024,
//! https://github.com/KhronosGroup/ToneMapping): an identity with a small toe below the shoulder,
//! so a lit albedo reaches the screen with its authored hue and saturation, and a hue preserving
//! shoulder that desaturates only highlights towards white. Against the reference images (Lay of
//! the Land) it keeps the saturated greens and the blue sky that the Narkowicz ACES fit shifts and
//! darkens and that AgX washes out; `tests::pbr_neutral_keeps_reference_colours` records the
//! comparison.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

use crate::timing::GpuTimer;
use crate::{FrameInput, FrameTargets};

/// Scale from HDR radiance (in units of the sun's illuminance, see lighting.wgsl) to the tonemap
/// input at `exposure_ev` 0: pi, so a white Lambert surface facing a sun of illuminance 1 above
/// the atmosphere maps to 1.0, the white of the curve, and a sunlit albedo reads close to its
/// authored value after the atmosphere's transmittance.
pub(crate) const EXPOSURE_CALIBRATION: f32 = std::f32::consts::PI;

pub(crate) const TONEMAP_WGSL: &str = include_str!("../../../assets/shaders/tonemap.wgsl");

/// Matches `struct Tonemap` in `assets/shaders/tonemap.wgsl` (engineering rule 9).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct TonemapUniform {
    /// x: exposure scale `EXPOSURE_CALIBRATION * 2^exposure_ev`. yzw: reserved.
    pub params: [f32; 4],
}

const _: () = assert!(size_of::<TonemapUniform>() == 16);

pub(crate) struct PostPass {
    uniform: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
    target_format: wgpu::TextureFormat,
    /// Bind group of the HDR target, rebuilt when the targets are recreated.
    bind_group: Option<(u64, wgpu::BindGroup)>,
}

impl PostPass {
    pub(crate) fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tonemap uniform"),
            size: size_of::<TonemapUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap"),
            entries: &[uniform_entry(0), texture_entry(1, false)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = fullscreen_pipeline(
            device,
            &pipeline_layout,
            "tonemap.wgsl",
            TONEMAP_WGSL,
            target_format,
        )
        .expect("the built in tonemap shader must compile");
        Self {
            uniform,
            layout,
            pipeline_layout,
            pipeline,
            target_format,
            bind_group: None,
        }
    }

    pub(crate) fn reload(&mut self, device: &wgpu::Device, source: &str) -> Result<(), String> {
        self.pipeline = fullscreen_pipeline(
            device,
            &self.pipeline_layout,
            "tonemap.wgsl",
            source,
            self.target_format,
        )?;
        Ok(())
    }

    pub(crate) fn prepare_frame(&self, queue: &wgpu::Queue, frame: &FrameInput) {
        let uniform = TonemapUniform {
            params: [
                EXPOSURE_CALIBRATION * frame.exposure_ev.exp2(),
                0.0,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    pub(crate) fn encode_tonemap(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        targets: &FrameTargets,
        target: &wgpu::TextureView,
        timer: &mut GpuTimer,
    ) {
        profiling::scope!("tonemap pass");
        if self
            .bind_group
            .as_ref()
            .is_none_or(|(generation, _)| *generation != targets.generation)
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tonemap"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&targets.hdr),
                    },
                ],
            });
            self.bind_group = Some((targets.generation, group));
        }
        let Some((_, bind_group)) = &self.bind_group else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timer.render_writes("tonemap"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A texture read with `textureLoad` (no sampler), float or depth.
pub(crate) fn texture_entry(binding: u32, depth: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
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

/// A fullscreen triangle pipeline (`vs_main` and `fs_main`, no vertex buffers, no depth) built
/// inside a validation error scope so a bad hot reload returns an error instead of aborting.
pub(crate) fn fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    label: &str,
    source: &str,
    format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    match pollster::block_on(scope.pop()) {
        None => Ok(pipeline),
        Some(error) => Err(error.to_string()),
    }
}
