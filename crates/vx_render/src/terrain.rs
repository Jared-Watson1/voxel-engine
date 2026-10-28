//! The terrain pass (D10, D14, D19, D22, D23): pooled vertex and index buffers with a range
//! allocator each, a slot table of resident meshes with their placements, the persistent chunk
//! table and the frame uniform, and the render pipeline that pulls packed vertices from the pool
//! through the compacted index buffer into the G-buffer. Culling lives in [`crate::cull`], the
//! material array in [`crate::materials`].
//!
//! The chunk table (D23) holds one 64 byte [`ChunkEntry`] per slot with the chunk's placement
//! relative to the render origin, its own voxel size and its transition mask. A CPU mirror is kept
//! in step with the slots; an entry is marked dirty when an upload, a placement change or a free
//! touches it, and only the dirty runs are written before the next frame. The frame uniform
//! carries the camera's offset from the render origin, so a camera move writes no entry.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use vx_core::PackedVertex;

use crate::cull::{CHUNK_MASK_SHIFT, CullPass, FLAG_VISIBLE, MAX_SLOTS, frustum_planes};
use crate::materials::MaterialArray;
use crate::pool::{Range, RangeAllocator};
use crate::{
    ChunkPlacement, CullStats, DEPTH_COMPARE, DEPTH_FORMAT, FrameInput, GBUFFER_ALBEDO_FORMAT,
    GBUFFER_NORMAL_FORMAT, MATERIAL_TILE_M, MeshHandle, MeshUpload, PlanetView, PoolError,
    PoolStats, PreparedMaterialTextures, TerrainPoolConfig,
};

/// Built in copy of `assets/shaders/terrain.wgsl`, so the first frame needs no file I/O. The file
/// watcher replaces it when the file changes.
pub(crate) const TERRAIN_WGSL: &str = include_str!("../../../assets/shaders/terrain.wgsl");

/// Matches `struct Frame` in `assets/shaders/terrain.wgsl` and `terrain_cull.wgsl` (engineering
/// rule 9). Everything is relative to the camera or the render origin (D4, D23).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FrameUniform {
    /// Projection times the rotation only view matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Frustum planes as (normal, distance), inside positive; the sixth always passes because the
    /// far plane is at infinity.
    pub planes: [[f32; 4]; 6],
    /// xyz: camera position relative to the render origin. w: material tile length.
    pub camera_offset_m: [f32; 4],
    /// xyz: planet centre relative to the camera. w: occluder radius, 0 to disable the horizon
    /// test.
    pub planet: [f32; 4],
    pub slot_count: u32,
    /// Always 0. The vertex shader ORs it into the bits of every intermediate of the secondary
    /// position so Metal's fast math cannot fuse or reorder that arithmetic: a uniform is a value
    /// the shader compiler cannot fold (D22).
    pub zero: u32,
    pub padding: [u32; 2],
}

const _: () = assert!(size_of::<FrameUniform>() == 208);
const _: () = assert!(std::mem::offset_of!(FrameUniform, planes) == 64);
const _: () = assert!(std::mem::offset_of!(FrameUniform, camera_offset_m) == 160);
const _: () = assert!(std::mem::offset_of!(FrameUniform, planet) == 176);
const _: () = assert!(std::mem::offset_of!(FrameUniform, slot_count) == 192);
const _: () = assert!(std::mem::offset_of!(FrameUniform, zero) == 196);

impl FrameUniform {
    pub fn new(
        view_proj: Mat4,
        camera_offset_m: Vec3,
        planet: &PlanetView,
        slot_count: u32,
    ) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            planes: frustum_planes(view_proj),
            camera_offset_m: camera_offset_m.extend(MATERIAL_TILE_M as f32).to_array(),
            planet: planet
                .centre_m
                .extend(planet.occluder_radius_m.max(0.0))
                .to_array(),
            slot_count,
            zero: 0,
            padding: [0; 2],
        }
    }
}

/// One chunk table entry (D23), matching `struct ChunkEntry` in both terrain shaders. 64 bytes:
/// the vec3 members give the WGSL struct a 16 byte alignment and 64 is a multiple of it, so the
/// array strides agree.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
pub(crate) struct ChunkEntry {
    /// Chunk origin relative to the render origin.
    pub offset_m: [f32; 3],
    /// First vertex of the mesh in the vertex pool, added to its mesh local indices.
    pub vertex_base: u32,
    pub texture_phase_m: [f32; 3],
    /// First index of the mesh in the index pool: the regular range, then the six transition
    /// ranges in face order.
    pub index_base: u32,
    pub regular_count: u32,
    /// Transition range index counts as u16 pairs: face `2k` in the low half of element `k`,
    /// face `2k + 1` in the high half.
    pub transition_counts: [u32; 3],
    pub voxel_size_m: f32,
    /// [`FLAG_VISIBLE`] and the transition mask at [`CHUNK_MASK_SHIFT`].
    pub flags: u32,
    pub padding: [u32; 2],
}

const _: () = assert!(size_of::<ChunkEntry>() == 64);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, vertex_base) == 12);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, texture_phase_m) == 16);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, index_base) == 28);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, regular_count) == 32);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, transition_counts) == 36);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, voxel_size_m) == 48);
const _: () = assert!(std::mem::offset_of!(ChunkEntry, flags) == 52);

impl ChunkEntry {
    fn new(mesh: &ResidentMesh) -> Self {
        let t = mesh.transition_counts.map(u32::from);
        let placement = &mesh.placement;
        let visible = if placement.visible { FLAG_VISIBLE } else { 0 };
        Self {
            offset_m: placement.offset_m.to_array(),
            vertex_base: mesh.vertex_base,
            texture_phase_m: placement.texture_phase_m.to_array(),
            index_base: mesh.index_base,
            regular_count: mesh.regular_count,
            transition_counts: [t[0] | t[1] << 16, t[2] | t[3] << 16, t[4] | t[5] << 16],
            voxel_size_m: placement.voxel_size_m,
            flags: visible | u32::from(placement.transition_mask & 0x3f) << CHUNK_MASK_SHIFT,
            padding: [0; 2],
        }
    }
}

const VERTEX_BYTES: usize = size_of::<PackedVertex>();
const INDEX_BYTES: usize = size_of::<u32>();
const ENTRY_BYTES: usize = size_of::<ChunkEntry>();
/// Entries in the chunk table buffer at creation; it doubles when the slot count passes it.
const INITIAL_TABLE_ENTRIES: usize = 1024;
/// Clean entries between two dirty runs that are rewritten anyway to save a write call; a write
/// costs more than a few hundred bytes of copy.
const DIRTY_RUN_GAP: usize = 4;

struct ResidentMesh {
    vertex_range: Range,
    index_range: Range,
    vertex_base: u32,
    index_base: u32,
    regular_count: u32,
    transition_counts: [u16; 6],
    placement: ChunkPlacement,
}

/// Chunk table writes of one [`TerrainPass::prepare_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct TableWrites {
    pub entries: usize,
    pub calls: usize,
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
    /// CPU mirror of the chunk table, one entry per slot (zeroed for a free slot).
    table: Vec<ChunkEntry>,
    /// Slots whose entry changed since the last frame, unordered and without repeats.
    dirty: Vec<u16>,
    is_dirty: Vec<bool>,
    /// Set when the table buffer was replaced, which loses every entry.
    rewrite_table: bool,
    #[cfg(test)]
    last_writes: TableWrites,
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
        let pipeline = build_pipeline(device, &pipeline_layout, TERRAIN_WGSL)
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
            dirty: Vec::new(),
            is_dirty: Vec::new(),
            rewrite_table: false,
            #[cfg(test)]
            last_writes: TableWrites::default(),
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

    pub(crate) fn reload(&mut self, device: &wgpu::Device, source: &str) -> Result<(), String> {
        self.pipeline = build_pipeline(device, &self.pipeline_layout, source)?;
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
        mesh: &MeshUpload,
        placement: ChunkPlacement,
    ) -> Result<MeshHandle, PoolError> {
        profiling::scope!("terrain upload");
        let (vertices, indices) = (mesh.vertices, mesh.indices);
        if vertices.is_empty() || indices.is_empty() {
            return Err(PoolError::EmptyMesh);
        }
        let Some(transition_counts) = checked_counts(mesh) else {
            // PoolError has no variant for inconsistent counts; the mesher never produces them.
            tracing::error!(
                "upload_mesh: index counts (regular {}, transitions {:?}) do not describe {} \
                 indices with u16 transition ranges; mesh rejected",
                mesh.regular_index_count,
                mesh.transition_index_counts,
                indices.len()
            );
            return Err(PoolError::EmptyMesh);
        };
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
        let resident = ResidentMesh {
            vertex_range,
            index_range,
            vertex_base: (vertex_range.offset / VERTEX_BYTES) as u32,
            index_base: (index_range.offset / INDEX_BYTES) as u32,
            regular_count: mesh.regular_index_count,
            transition_counts,
            placement,
        };
        self.write_entry(slot, ChunkEntry::new(&resident));
        self.slots[usize::from(slot)] = Some(resident);
        self.mesh_count += 1;
        Ok(MeshHandle(slot))
    }

    pub(crate) fn set_placement(&mut self, handle: MeshHandle, placement: ChunkPlacement) {
        let slot = handle.0;
        let entry = match self.slots.get_mut(usize::from(slot)) {
            Some(Some(mesh)) if mesh.placement != placement => {
                mesh.placement = placement;
                ChunkEntry::new(mesh)
            }
            Some(Some(_)) => return,
            _ => {
                tracing::warn!("set_placement: slot {slot} holds no mesh");
                return;
            }
        };
        self.write_entry(slot, entry);
    }

    pub(crate) fn free_mesh(&mut self, handle: MeshHandle) {
        let slot = handle.0;
        match self.slots.get_mut(usize::from(slot)).and_then(Option::take) {
            Some(mesh) => {
                self.vertex_alloc.free(mesh.vertex_range);
                self.index_alloc.free(mesh.index_range);
                self.mesh_count -= 1;
                self.free_slots.push(Reverse(slot));
                self.write_entry(slot, ChunkEntry::zeroed());
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
        self.table.push(ChunkEntry::zeroed());
        self.is_dirty.push(false);
        Ok((self.slots.len() - 1) as u16)
    }

    /// Updates the CPU mirror and queues the entry for the next frame's table write.
    fn write_entry(&mut self, slot: u16, entry: ChunkEntry) {
        let i = usize::from(slot);
        self.table[i] = entry;
        if !self.is_dirty[i] {
            self.is_dirty[i] = true;
            self.dirty.push(slot);
        }
    }

    pub(crate) fn set_material_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &PreparedMaterialTextures,
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

    /// Collects finished readbacks, writes the changed chunk table entries and the frame's
    /// uniform and counters. Queue writes land before the frame's submission.
    pub(crate) fn prepare_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameInput,
    ) {
        profiling::scope!("terrain prepare");
        self.frame_index += 1;
        self.cull.poll_readback(device);

        let slot_count = self.slots.len();
        let capacity = self.table_buffer.size() as usize / ENTRY_BYTES;
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
            self.rewrite_table = true;
        }
        let _writes = self.write_table(queue);
        #[cfg(test)]
        {
            self.last_writes = _writes;
        }

        let uniform = FrameUniform::new(
            frame.view_proj,
            frame.camera_offset_m,
            &frame.planet,
            slot_count as u32,
        );
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        self.cull.reset_counters(queue);
        self.slot_count_this_frame = slot_count as u32;
    }

    fn write_table(&mut self, queue: &wgpu::Queue) -> TableWrites {
        profiling::scope!("terrain chunk table");
        let runs = if std::mem::take(&mut self.rewrite_table) {
            self.dirty.clear();
            vec![std::ops::Range {
                start: 0,
                end: self.table.len(),
            }]
        } else {
            dirty_runs(&mut self.dirty, DIRTY_RUN_GAP)
        };
        let mut writes = TableWrites::default();
        for run in runs.into_iter().filter(|run| !run.is_empty()) {
            queue.write_buffer(
                &self.table_buffer,
                (run.start * ENTRY_BYTES) as u64,
                bytemuck::cast_slice(&self.table[run.clone()]),
            );
            self.is_dirty[run.clone()].fill(false);
            writes.entries += run.len();
            writes.calls += 1;
        }
        writes
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

/// The six transition counts as u16 when they and the regular count sum to the index count.
fn checked_counts(mesh: &MeshUpload) -> Option<[u16; 6]> {
    let total = mesh
        .transition_index_counts
        .iter()
        .try_fold(mesh.regular_index_count, |sum, &c| sum.checked_add(c))?;
    if total as usize != mesh.indices.len() {
        return None;
    }
    let mut counts = [0u16; 6];
    for (out, &c) in counts.iter_mut().zip(&mesh.transition_index_counts) {
        *out = u16::try_from(c).ok()?;
    }
    Some(counts)
}

/// Sorts and drains the dirty slots into runs of entries to write, joining runs separated by at
/// most `gap` clean entries.
fn dirty_runs(dirty: &mut Vec<u16>, gap: usize) -> Vec<std::ops::Range<usize>> {
    dirty.sort_unstable();
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    for slot in dirty.drain(..).map(usize::from) {
        match runs.last_mut() {
            Some(run) if slot <= run.end + gap => run.end = slot + 1,
            _ => runs.push(slot..slot + 1),
        }
    }
    runs
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

/// COPY_SRC lets tests read the table back.
fn create_table_buffer(device: &wgpu::Device, entries: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain chunk table"),
        size: (entries * ENTRY_BYTES) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
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
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: GBUFFER_ALBEDO_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: GBUFFER_NORMAL_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
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
mod tests;
