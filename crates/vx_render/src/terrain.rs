use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::{DEPTH_COMPARE, DEPTH_FORMAT, MeshData};

/// Built in copy of `assets/shaders/terrain.wgsl`, so the first frame needs no file I/O. The file
/// watcher replaces it when the file changes.
pub(crate) const TERRAIN_WGSL: &str = include_str!("../../../assets/shaders/terrain.wgsl");

/// Matches `struct Frame` in `assets/shaders/terrain.wgsl` (engineering rule 9).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FrameUniform {
    pub view_proj: [[f32; 4]; 4],
    pub mesh_offset_m: [f32; 4],
    pub sun_dir: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<FrameUniform>() == 96);

impl FrameUniform {
    pub fn new(view_proj: Mat4, mesh_offset_m: Vec3, sun_dir: Vec3, ambient: f32) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            mesh_offset_m: mesh_offset_m.extend(0.0).to_array(),
            sun_dir: sun_dir.normalize_or_zero().extend(ambient).to_array(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position_m: [f32; 3],
    normal: [f32; 3],
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

pub(crate) struct TerrainPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::PipelineLayout,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    mesh: Option<GpuMesh>,
}

impl TerrainPass {
    pub(crate) fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<FrameUniform>() as u64
                    ),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain frame uniform"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain frame"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let pipeline = build_pipeline(device, &layout, color_format, TERRAIN_WGSL)
            .expect("the built in terrain shader must compile");
        Self {
            pipeline,
            layout,
            uniform,
            bind_group,
            mesh: None,
        }
    }

    pub(crate) fn reload(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        source: &str,
    ) -> Result<(), String> {
        self.pipeline = build_pipeline(device, &self.layout, color_format, source)?;
        Ok(())
    }

    pub(crate) fn set_mesh(&mut self, device: &wgpu::Device, mesh: &MeshData) {
        debug_assert_eq!(mesh.positions.len(), mesh.normals.len());
        if mesh.indices.is_empty() {
            self.mesh = None;
            return;
        }
        let vertices: Vec<Vertex> = mesh
            .positions
            .iter()
            .zip(mesh.normals)
            .map(|(&position_m, &normal)| Vertex { position_m, normal })
            .collect();
        self.mesh = Some(GpuMesh {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain indices"),
                contents: bytemuck::cast_slice(mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: mesh.indices.len() as u32,
        });
    }

    pub(crate) fn write_uniform(&self, queue: &wgpu::Queue, uniform: &FrameUniform) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(uniform));
    }

    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass) {
        let Some(mesh) = &self.mesh else { return };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
    }
}

/// Builds the pipeline inside a validation error scope, because WGSL errors surface at module
/// creation and entry point or binding mismatches only at pipeline creation.
fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    source: &str,
) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("terrain"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &VERTEX_ATTRIBUTES,
            })],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(DEPTH_COMPARE),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    // wgpu-core resolves error scopes synchronously on native, so this returns immediately.
    match pollster::block_on(scope.pop()) {
        None => Ok(pipeline),
        Some(error) => Err(error.to_string()),
    }
}
