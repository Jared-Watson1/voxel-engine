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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tonemap_uniform_matches_the_wgsl() {
        let (size, offsets) = crate::atmosphere::tests::wgsl_struct_layout(TONEMAP_WGSL, "Tonemap");
        assert_eq!(size as usize, size_of::<TonemapUniform>());
        assert_eq!(offsets, [0]);
    }

    /// Mirror of `pbr_neutral` in tonemap.wgsl.
    fn pbr_neutral(c: [f64; 3]) -> [f64; 3] {
        const START: f64 = 0.76;
        const DESATURATION: f64 = 0.15;
        let x = c[0].min(c[1]).min(c[2]);
        let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
        let mut c = c.map(|v| v - offset);
        let peak = c[0].max(c[1]).max(c[2]);
        if peak < START {
            return c;
        }
        let d = 1.0 - START;
        let new_peak = 1.0 - d * d / (peak + d - START);
        c = c.map(|v| v * new_peak / peak);
        let g = 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0);
        c.map(|v| v + (new_peak - v) * g)
    }

    /// The placeholder curve: Narkowicz 2016's fit of the ACES reference rendering transform.
    fn aces_fit(c: [f64; 3]) -> [f64; 3] {
        c.map(|x| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0))
    }

    /// AgX with the base look (Troy Sobotka; the minimal fit by bwrensch), returning linear sRGB.
    fn agx(c: [f64; 3]) -> [f64; 3] {
        const INSET: [[f64; 3]; 3] = [
            [
                0.842_479_062_253_094,
                0.078_433_599_999_999_2,
                0.079_223_745_147_764_3,
            ],
            [
                0.042_328_242_261_012_3,
                0.878_468_636_469_772,
                0.079_166_127_460_543_4,
            ],
            [0.042_375_654_905_705_1, 0.078_433_6, 0.879_142_973_793_104],
        ];
        const OUTSET: [[f64; 3]; 3] = [
            [
                1.196_879_005_120_17,
                -0.098_020_881_140_136_8,
                -0.099_029_744_079_720_5,
            ],
            [
                -0.052_896_851_757_456_2,
                1.151_903_129_904_17,
                -0.098_961_176_844_843_3,
            ],
            [
                -0.052_971_635_514_443_8,
                -0.098_043_450_117_124_1,
                1.151_073_672_641_16,
            ],
        ];
        let mul = |m: [[f64; 3]; 3], v: [f64; 3]| {
            std::array::from_fn::<f64, 3, _>(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
        };
        let (min_ev, max_ev) = (-12.473_931_188, 4.026_068_812);
        let v = mul(INSET, c).map(|x| {
            let x = (x.max(1.0e-10).log2().clamp(min_ev, max_ev) - min_ev) / (max_ev - min_ev);
            let x2 = x * x;
            let x4 = x2 * x2;
            15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
                - 0.00232
        });
        mul(OUTSET, v).map(|x| x.clamp(0.0, 1.0).powf(2.2))
    }

    fn srgb_encode(x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        if x <= 0.003_130_8 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    }

    fn srgb_decode(x: f64) -> f64 {
        if x <= 0.040_45 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Hue in degrees and HSV saturation of a display (sRGB encoded) colour.
    fn hue_saturation(c: [f64; 3]) -> (f64, f64) {
        let max = c[0].max(c[1]).max(c[2]);
        let min = c[0].min(c[1]).min(c[2]);
        let chroma = max - min;
        let hue = if chroma <= 0.0 {
            0.0
        } else if max == c[0] {
            60.0 * ((c[1] - c[2]) / chroma).rem_euclid(6.0)
        } else if max == c[1] {
            60.0 * ((c[2] - c[0]) / chroma + 2.0)
        } else {
            60.0 * ((c[0] - c[1]) / chroma + 4.0)
        };
        (hue, if max > 0.0 { chroma / max } else { 0.0 })
    }

    /// Colours sampled from the Lay of the Land references (sRGB, 8 bit): canopy and grass from
    /// the vista and the ground capture, the zenith and the mid sky, the haze on the far ridge.
    const REFERENCE: [(&str, [u8; 3]); 5] = [
        ("canopy", [82, 102, 27]),
        ("grass", [96, 119, 41]),
        ("zenith sky", [124, 159, 192]),
        ("mid sky", [156, 188, 210]),
        ("ridge haze", [171, 164, 143]),
    ];

    /// Treats each reference colour as the scene value the curve should reproduce (its linear
    /// value at exposure 1, then twice as bright for a sunlit highlight) and measures the hue
    /// shift and the saturation kept on the display.
    #[test]
    fn pbr_neutral_keeps_reference_colours() {
        type Curve = fn([f64; 3]) -> [f64; 3];
        let curves: [(&str, Curve); 3] = [
            ("PBR Neutral", pbr_neutral),
            ("ACES fit", aces_fit),
            ("AgX", agx),
        ];
        let mut worst = Vec::new();
        for (curve_name, curve) in curves {
            let mut worst_hue: f64 = 0.0;
            let mut worst_saturation: f64 = f64::MAX;
            for (name, rgb) in REFERENCE {
                let display = rgb.map(|v| f64::from(v) / 255.0);
                let linear = display.map(srgb_decode);
                let (hue, saturation) = hue_saturation(display);
                for scale in [1.0, 2.0] {
                    let out = curve(linear.map(|v| v * scale)).map(srgb_encode);
                    let (h, s) = hue_saturation(out);
                    let shift = (h - hue).abs().min(360.0 - (h - hue).abs());
                    let kept = s / saturation;
                    eprintln!(
                        "{curve_name:12} {name:11} x{scale}: display {:?} hue shift {shift:5.1} deg, saturation kept {kept:4.2}",
                        out.map(|v| (v * 255.0).round() as u8)
                    );
                    worst_hue = worst_hue.max(shift);
                    worst_saturation = worst_saturation.min(kept);
                }
            }
            worst.push((curve_name, worst_hue, worst_saturation));
        }
        eprintln!("worst hue shift and saturation kept: {worst:?}");
        let (_, neutral_hue, neutral_saturation) = worst[0];
        // PBR Neutral keeps every reference colour within a few degrees of hue and keeps most
        // of its saturation, also at twice the brightness; it does better on both than the
        // others.
        assert!(neutral_hue < 4.0, "{worst:?}");
        assert!(neutral_saturation > 0.85, "{worst:?}");
        for (name, hue, saturation) in &worst[1..] {
            assert!(
                *hue > neutral_hue || *saturation < neutral_saturation,
                "{name} does not lose to PBR Neutral: {worst:?}"
            );
        }
    }
}
