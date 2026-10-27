//! The terrain pass (D10, D14, D19): pooled vertex and index buffers with a range allocator each,
//! a slot table of resident meshes, the per frame chunk table and uniform, and the render
//! pipeline that pulls packed vertices from the pool through the compacted index buffer. Culling
//! lives in [`crate::cull`], the material array in [`crate::materials`].

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use vx_core::{CHUNK_SIZE, PackedVertex};

use crate::cull::{CullPass, FLAG_RESIDENT, MAX_SLOTS, frustum_planes};
use crate::materials::MaterialArray;
use crate::pool::{Range, RangeAllocator};
use crate::{
    CullStats, DEPTH_COMPARE, DEPTH_FORMAT, FrameInput, MATERIAL_TILE_M, MaterialTextures,
    MeshHandle, PoolError, PoolStats, TerrainPoolConfig,
};

/// Built in copy of `assets/shaders/terrain.wgsl`, so the first frame needs no file I/O. The file
/// watcher replaces it when the file changes.
pub(crate) const TERRAIN_WGSL: &str = include_str!("../../../assets/shaders/terrain.wgsl");

/// Matches `struct Frame` in `assets/shaders/terrain.wgsl` and `terrain_cull.wgsl` (engineering
/// rule 9). Everything is camera relative (D4).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FrameUniform {
    pub view_proj: [[f32; 4]; 4],
    /// Frustum planes as (normal, distance), inside positive; the sixth always passes because the
    /// far plane is at infinity.
    pub planes: [[f32; 4]; 6],
    /// xyz: unit direction towards the sun. w: ambient strength.
    pub sun_dir: [f32; 4],
    /// Voxel size, chunk extent and material tile length in metres; the fourth is unused.
    pub params: [f32; 4],
    pub slot_count: u32,
    pub padding: [u32; 3],
}

const _: () = assert!(size_of::<FrameUniform>() == 208);
const _: () = assert!(std::mem::offset_of!(FrameUniform, planes) == 64);
const _: () = assert!(std::mem::offset_of!(FrameUniform, sun_dir) == 160);
const _: () = assert!(std::mem::offset_of!(FrameUniform, params) == 176);
const _: () = assert!(std::mem::offset_of!(FrameUniform, slot_count) == 192);

impl FrameUniform {
    pub fn new(
        view_proj: Mat4,
        sun_dir: Vec3,
        ambient: f32,
        voxel_size_m: f32,
        slot_count: u32,
    ) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            planes: frustum_planes(view_proj),
            sun_dir: sun_dir.normalize_or_zero().extend(ambient).to_array(),
            params: [
                voxel_size_m,
                CHUNK_SIZE as f32 * voxel_size_m,
                MATERIAL_TILE_M as f32,
                0.0,
            ],
            slot_count,
            padding: [0; 3],
        }
    }
}

/// Matches `struct ChunkEntry` in both shaders: 48 bytes, a multiple of the 16 byte alignment
/// the vec3 members give the WGSL struct, so the array stride agrees.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
pub(crate) struct ChunkEntry {
    pub offset_m: [f32; 3],
    pub vertex_base: u32,
    pub texture_phase_m: [f32; 3],
    pub index_base: u32,
    pub index_count: u32,
    pub flags: u32,
    pub padding: [u32; 2],
}

const _: () = assert!(size_of::<ChunkEntry>() == 48);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, vertex_base) == 12);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, texture_phase_m) == 16);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, index_base) == 28);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, index_count) == 32);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, flags) == 36);

const VERTEX_BYTES: usize = size_of::<PackedVertex>();
const INDEX_BYTES: usize = size_of::<u32>();
/// Entries in the chunk table buffer at creation; it doubles when the slot count passes it.
const INITIAL_TABLE_ENTRIES: usize = 1024;

struct ResidentMesh {
    vertex_range: Range,
    index_range: Range,
    vertex_base: u32,
    index_base: u32,
    index_count: u32,
}

pub(crate) struct TerrainPass {
    vertex_pool: wgpu::Buffer,
    index_pool: wgpu::Buffer,
    vertex_alloc: RangeAllocator,
    index_alloc: RangeAllocator,
    /// Indexed by chunk slot; `None` is a free slot. The length is the highest slot handed out
    /// plus one, which bounds the chunk table.
    slots: Vec<Option<ResidentMesh>>,
    /// Lowest freed slot first, so the table stays short after churn.
    free_slots: BinaryHeap<Reverse<u16>>,
    mesh_count: usize,
    staging: Vec<PackedVertex>,
    table: Vec<ChunkEntry>,
    table_buffer: wgpu::Buffer,
    uniform: wgpu::Buffer,
    frame_layout: wgpu::BindGroupLayout,
    frame_bind_group: wgpu::BindGroup,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
    materials: MaterialArray,
    cull: CullPass,
    frame_index: u64,
    slot_count_this_frame: u32,
}

impl TerrainPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        pools: TerrainPoolConfig,
    ) -> Self {
        let vertex_pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain vertex pool"),
            size: pools.vertex_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain index pool"),
            size: pools.index_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let table_buffer = create_table_buffer(device, INITIAL_TABLE_ENTRIES);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain frame uniform"),
            size: size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain frame"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<FrameUniform>() as u64),
                    },
                    count: None,
                },
                read_only_storage(1),
                read_only_storage(2),
            ],
        });
        let materials = MaterialArray::new(device, queue);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain"),
            bind_group_layouts: &[Some(&frame_layout), Some(materials.layout())],
            immediate_size: 0,
        });
        let pipeline = build_pipeline(device, &pipeline_layout, color_format, TERRAIN_WGSL)
            .expect("the built in terrain shader must compile");
        let frame_bind_group =
            frame_bind_group(device, &frame_layout, &uniform, &vertex_pool, &table_buffer);
        let cull = CullPass::new(
            device,
            &uniform,
            size_of::<FrameUniform>() as u64,
            &table_buffer,
            &index_pool,
            pools.index_bytes,
        );

        Self {
            vertex_pool,
            index_pool,
            vertex_alloc: RangeAllocator::new(pools.vertex_bytes, VERTEX_BYTES),
            index_alloc: RangeAllocator::new(pools.index_bytes, INDEX_BYTES),
            slots: Vec::new(),
            free_slots: BinaryHeap::new(),
            mesh_count: 0,
            staging: Vec::new(),
            table: Vec::new(),
            table_buffer,
            uniform,
            frame_layout,
            frame_bind_group,
            pipeline_layout,
            pipeline,
            materials,
            cull,
            frame_index: 0,
            slot_count_this_frame: 0,
        }
    }

    pub(crate) fn reload(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        source: &str,
    ) -> Result<(), String> {
        self.pipeline = build_pipeline(device, &self.pipeline_layout, color_format, source)?;
        Ok(())
    }

    pub(crate) fn reload_cull(
        &mut self,
        device: &wgpu::Device,
        source: &str,
    ) -> Result<(), String> {
        self.cull.reload(device, source)
    }

    pub(crate) fn upload_mesh(
        &mut self,
        queue: &wgpu::Queue,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> Result<MeshHandle, PoolError> {
        profiling::scope!("terrain upload");
        if vertices.is_empty() || indices.is_empty() {
            return Err(PoolError::EmptyMesh);
        }
        let slot = self.take_slot()?;
        let vertex_range = match self.vertex_alloc.allocate(vertices.len() * VERTEX_BYTES) {
            Ok(range) => range,
            Err(e) => {
                self.free_slots.push(Reverse(slot));
                return Err(PoolError::VertexPoolFull {
                    needed: e.needed,
                    largest_free: e.largest_free,
                });
            }
        };
        let index_range = match self.index_alloc.allocate(indices.len() * INDEX_BYTES) {
            Ok(range) => range,
            Err(e) => {
                self.vertex_alloc.free(vertex_range);
                self.free_slots.push(Reverse(slot));
                return Err(PoolError::IndexPoolFull {
                    needed: e.needed,
                    largest_free: e.largest_free,
                });
            }
        };
        self.staging.clear();
        self.staging
            .extend(vertices.iter().map(|v| v.with_chunk_slot(slot)));
        queue.write_buffer(
            &self.vertex_pool,
            vertex_range.offset as u64,
            bytemuck::cast_slice(&self.staging),
        );
        queue.write_buffer(
            &self.index_pool,
            index_range.offset as u64,
            bytemuck::cast_slice(indices),
        );
        self.slots[usize::from(slot)] = Some(ResidentMesh {
            vertex_range,
            index_range,
            vertex_base: (vertex_range.offset / VERTEX_BYTES) as u32,
            index_base: (index_range.offset / INDEX_BYTES) as u32,
            index_count: indices.len() as u32,
        });
        self.mesh_count += 1;
        Ok(MeshHandle(slot))
    }

    pub(crate) fn free_mesh(&mut self, handle: MeshHandle) {
        let slot = handle.0;
        match self.slots.get_mut(usize::from(slot)).and_then(Option::take) {
            Some(mesh) => {
                self.vertex_alloc.free(mesh.vertex_range);
                self.index_alloc.free(mesh.index_range);
                self.mesh_count -= 1;
                self.free_slots.push(Reverse(slot));
            }
            None => tracing::warn!("free_mesh: slot {slot} holds no mesh"),
        }
    }

    fn take_slot(&mut self) -> Result<u16, PoolError> {
        if let Some(Reverse(slot)) = self.free_slots.pop() {
            return Ok(slot);
        }
        if self.slots.len() >= MAX_SLOTS {
            return Err(PoolError::NoFreeSlot);
        }
        self.slots.push(None);
        Ok((self.slots.len() - 1) as u16)
    }

    pub(crate) fn set_material_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &MaterialTextures,
    ) {
        profiling::scope!("terrain upload materials");
        if let Err(e) = self.materials.set(device, queue, textures) {
            tracing::error!("material textures rejected, keeping the previous array: {e}");
        }
    }

    pub(crate) fn pool_stats(&self) -> PoolStats {
        PoolStats {
            vertex_bytes_used: self.vertex_alloc.used(),
            vertex_bytes_capacity: self.vertex_alloc.capacity(),
            index_bytes_used: self.index_alloc.used(),
            index_bytes_capacity: self.index_alloc.capacity(),
            largest_free_vertex_bytes: self.vertex_alloc.largest_free(),
            largest_free_index_bytes: self.index_alloc.largest_free(),
            meshes: self.mesh_count,
            texture_bytes: self.materials.bytes(),
        }
    }

    pub(crate) fn cull_stats(&self) -> Option<CullStats> {
        self.cull.cull_stats(self.frame_index)
    }

    /// Collects finished readbacks, builds the chunk table and writes the frame's uniform and
    /// counters. Queue writes land before the frame's submission.
    pub(crate) fn prepare_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameInput,
    ) {
        profiling::scope!("terrain prepare");
        self.frame_index += 1;
        self.cull.poll_readback(device);

        let slot_count = self.slots.len().max(1);
        self.table.clear();
        self.table.resize(slot_count, ChunkEntry::zeroed());
        for draw in frame.chunks {
            let slot = usize::from(draw.handle.0);
            // A handle freed earlier this frame has no mesh and is skipped, never drawn.
            let Some(Some(mesh)) = self.slots.get(slot) else {
                continue;
            };
            self.table[slot] = ChunkEntry {
                offset_m: draw.offset_m.to_array(),
                vertex_base: mesh.vertex_base,
                texture_phase_m: draw.texture_phase_m.to_array(),
                index_base: mesh.index_base,
                index_count: mesh.index_count,
                flags: FLAG_RESIDENT,
                padding: [0; 2],
            };
        }

        let capacity = self.table_buffer.size() as usize / size_of::<ChunkEntry>();
        if slot_count > capacity {
            let mut entries = capacity.max(1);
            while entries < slot_count {
                entries *= 2;
            }
            self.table_buffer = create_table_buffer(device, entries.min(MAX_SLOTS));
            self.frame_bind_group = frame_bind_group(
                device,
                &self.frame_layout,
                &self.uniform,
                &self.vertex_pool,
                &self.table_buffer,
            );
            self.cull
                .rebind_chunk_table(device, &self.uniform, &self.table_buffer);
        }
        queue.write_buffer(&self.table_buffer, 0, bytemuck::cast_slice(&self.table));

        let uniform = FrameUniform::new(
            frame.view_proj,
            frame.sun_dir,
            frame.ambient,
            frame.voxel_size_m,
            slot_count as u32,
        );
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        self.cull.reset_counters(queue);
        self.slot_count_this_frame = slot_count as u32;
    }

    pub(crate) fn encode_cull(&self, encoder: &mut wgpu::CommandEncoder) {
        self.cull.encode(encoder, self.slot_count_this_frame);
    }

    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass) {
        profiling::scope!("terrain draw");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.frame_bind_group, &[]);
        pass.set_bind_group(1, self.materials.bind_group(), &[]);
        pass.set_index_buffer(self.cull.compacted().slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(self.cull.counters(), 0);
    }

    pub(crate) fn encode_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.cull.encode_readback(encoder, self.frame_index);
    }

    pub(crate) fn after_submit(&mut self) {
        self.cull.after_submit();
    }

    #[cfg(test)]
    fn cull_pass(&self) -> &CullPass {
        &self.cull
    }

    #[cfg(test)]
    fn poll_readback(&mut self, device: &wgpu::Device) {
        self.cull.poll_readback(device);
    }
}

fn read_only_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_table_buffer(device: &wgpu::Device, entries: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain chunk table"),
        size: (entries * size_of::<ChunkEntry>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn frame_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    vertex_pool: &wgpu::Buffer,
    table: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain frame"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_pool.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: table.as_entire_binding(),
            },
        ],
    })
}

/// Builds the pipeline inside a validation error scope, because WGSL errors surface at module
/// creation and entry point or binding mismatches only at pipeline creation. No vertex buffers:
/// the vertex shader pulls from the pool by `vertex_index`.
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
            buffers: &[],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChunkDraw;
    use crate::cull::Counters;
    use vx_core::{Projection, VertexAttributes};

    /// A device without a surface, or `None` when the machine has no adapter (CI).
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vx_render test device"),
            ..Default::default()
        }))
        .ok()?;
        Some((device, queue))
    }

    fn quad() -> (Vec<PackedVertex>, Vec<u32>) {
        let vertex = |x: f32, y: f32| {
            PackedVertex::pack(&VertexAttributes {
                position_voxels: Vec3::new(x, y, 16.0),
                normal: Vec3::Z,
                materials: [(2, 15), (0, 0), (0, 0), (0, 0)],
            })
        };
        (
            vec![
                vertex(0.0, 0.0),
                vertex(32.0, 0.0),
                vertex(32.0, 32.0),
                vertex(0.0, 32.0),
            ],
            vec![0, 1, 2, 0, 2, 3],
        )
    }

    fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
        buffer.map_async(wgpu::MapMode::Read, .., |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        let bytes = buffer.get_mapped_range(..).expect("mapped").to_vec();
        buffer.unmap();
        bytes
    }

    const TEST_POOLS: TerrainPoolConfig = TerrainPoolConfig {
        vertex_bytes: 1 << 16,
        index_bytes: 1 << 16,
    };

    #[test]
    fn cull_and_compact_on_the_gpu() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter available, skipping cull_and_compact_on_the_gpu");
            return;
        };
        let mut pass = TerrainPass::new(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            TEST_POOLS,
        );
        let (vertices, indices) = quad();
        // Uploaded first so its vertex base is 0, which makes the rebase of the visible mesh
        // observable (vertex base 4).
        let hidden = pass.upload_mesh(&queue, &vertices, &indices).unwrap();
        let seen = pass.upload_mesh(&queue, &vertices, &indices).unwrap();
        assert_eq!((hidden, seen), (MeshHandle(0), MeshHandle(1)));

        let view_proj = Projection::reverse_z_infinite(std::f32::consts::FRAC_PI_3, 1.0, 0.05);
        // The seen chunk spans z from -40 to -8 in front of the camera; the hidden one spans
        // z from 20 to 52 behind it.
        let chunks = [
            ChunkDraw {
                handle: seen,
                offset_m: Vec3::new(-16.0, -16.0, -40.0),
                texture_phase_m: Vec3::ZERO,
            },
            ChunkDraw {
                handle: hidden,
                offset_m: Vec3::new(-16.0, -16.0, 20.0),
                texture_phase_m: Vec3::ZERO,
            },
        ];
        let frame = FrameInput {
            view_proj,
            chunks: &chunks,
            voxel_size_m: 1.0,
            sun_dir: Vec3::Y,
            ambient: 0.2,
            egui: None,
        };
        pass.prepare_frame(&device, &queue, &frame);
        assert_eq!(pass.slot_count_this_frame, 2);

        let compacted_copy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test compacted copy"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let counters_copy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test counters copy"),
            size: size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        pass.encode_cull(&mut encoder);
        encoder.copy_buffer_to_buffer(pass.cull_pass().compacted(), 0, &compacted_copy, 0, 64);
        encoder.copy_buffer_to_buffer(
            pass.cull_pass().counters(),
            0,
            &counters_copy,
            0,
            size_of::<Counters>() as u64,
        );
        pass.encode_readback(&mut encoder);
        queue.submit([encoder.finish()]);
        pass.after_submit();

        let counters: Counters = *bytemuck::from_bytes(&read_buffer(&device, &counters_copy));
        assert_eq!(counters.index_count, 6, "{counters:?}");
        assert_eq!(counters.group_count_x, 1, "one visible chunk");
        assert_eq!((counters.instance_count, counters.group_count_y), (1, 1));
        let compacted: Vec<u32> =
            bytemuck::cast_slice(&read_buffer(&device, &compacted_copy)).to_vec();
        assert_eq!(
            &compacted[..6],
            &[4, 5, 6, 4, 6, 7],
            "indices rebased by vertex base 4"
        );

        pass.poll_readback(&device);
        let stats = pass.cull_stats().expect("readback mapped after the wait");
        assert_eq!(stats.visible_chunks, 1);
        assert_eq!(stats.triangles, 2);
        assert_eq!(stats.frames_old, 0);

        // A handle freed before the frame is skipped even though it is still listed.
        pass.free_mesh(seen);
        pass.prepare_frame(&device, &queue, &frame);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test 2"),
        });
        pass.encode_cull(&mut encoder);
        encoder.copy_buffer_to_buffer(
            pass.cull_pass().counters(),
            0,
            &counters_copy,
            0,
            size_of::<Counters>() as u64,
        );
        queue.submit([encoder.finish()]);
        let counters: Counters = *bytemuck::from_bytes(&read_buffer(&device, &counters_copy));
        assert_eq!((counters.index_count, counters.group_count_x), (0, 0));
        assert_eq!(pass.pool_stats().meshes, 1);
    }

    #[test]
    fn upload_fails_without_partial_state() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter available, skipping upload_fails_without_partial_state");
            return;
        };
        let mut pass = TerrainPass::new(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            TerrainPoolConfig {
                vertex_bytes: 4 * VERTEX_BYTES * 4,
                index_bytes: 8 * INDEX_BYTES,
            },
        );
        let (vertices, indices) = quad();
        assert!(matches!(
            pass.upload_mesh(&queue, &[], &indices),
            Err(PoolError::EmptyMesh)
        ));
        let first = pass.upload_mesh(&queue, &vertices, &indices).unwrap();
        let before = pass.pool_stats();
        let err = pass.upload_mesh(&queue, &vertices, &indices).unwrap_err();
        assert!(
            matches!(
                err,
                PoolError::IndexPoolFull {
                    needed: 24,
                    largest_free: 8
                }
            ),
            "{err}"
        );
        assert_eq!(
            pass.pool_stats(),
            before,
            "a failed upload leaves no partial state"
        );
        assert_eq!(
            pass.slots.len(),
            2,
            "the slot taken for the failed upload is free again"
        );
        assert_eq!(pass.free_slots.peek(), Some(&Reverse(1)));

        pass.free_mesh(first);
        let stats = pass.pool_stats();
        assert_eq!(
            (
                stats.meshes,
                stats.vertex_bytes_used,
                stats.index_bytes_used
            ),
            (0, 0, 0)
        );
        assert_eq!(stats.largest_free_vertex_bytes, stats.vertex_bytes_capacity);
        pass.free_mesh(first);
        assert_eq!(
            pass.pool_stats().meshes,
            0,
            "a double free is a warning, not a panic"
        );
        let again = pass.upload_mesh(&queue, &vertices, &indices).unwrap();
        assert_eq!(again, MeshHandle(0), "the lowest free slot is reused");
    }

    #[test]
    fn frame_uniform_carries_the_scale_parameters() {
        let u = FrameUniform::new(Mat4::IDENTITY, Vec3::new(0.0, 2.0, 0.0), 0.3, 0.5, 7);
        assert_eq!(u.params, [0.5, 16.0, MATERIAL_TILE_M as f32, 0.0]);
        assert_eq!(u.sun_dir, [0.0, 1.0, 0.0, 0.3]);
        assert_eq!(u.slot_count, 7);
        assert_eq!(u.planes[5], [0.0, 0.0, 0.0, 1.0]);
    }
}
