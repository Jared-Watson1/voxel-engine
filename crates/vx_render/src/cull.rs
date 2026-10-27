//! GPU frustum culling and index compaction (D10, D19) and the non blocking readback of the
//! culling counters. Two compute entry points in `assets/shaders/terrain_cull.wgsl`: `cull` runs
//! one thread per chunk slot and appends visible chunks to a list while reserving their index
//! range with an atomic; `compact` runs one workgroup per visible chunk and copies its indices,
//! rebased by the chunk's vertex base, into the compacted index buffer.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use bytemuck::{Pod, Zeroable};
#[cfg(test)]
use glam::Vec3;
use glam::{Mat4, Vec4};

use crate::CullStats;

/// Built in copy of `assets/shaders/terrain_cull.wgsl`; the game replaces it on hot reload.
pub(crate) const CULL_WGSL: &str = include_str!("../../../assets/shaders/terrain_cull.wgsl");

/// Threads per workgroup of the `cull` entry point; matches `CULL_WORKGROUP_SIZE` in the shader.
pub(crate) const CULL_WORKGROUP_SIZE: u32 = 64;
/// Chunk slots addressable by the 16 bits in a packed vertex (D14).
/// One below the u16 range: the compaction pass dispatches one workgroup per visible chunk and
/// `max_compute_workgroups_per_dimension` is 65535.
pub(crate) const MAX_SLOTS: usize = (1 << 16) - 1;
/// `ChunkEntry::flags` bit 0: the slot is resident and listed this frame.
pub(crate) const FLAG_RESIDENT: u32 = 1;

/// Matches `struct Counters` in `terrain_cull.wgsl`: indexed indirect draw arguments
/// (`wgpu::util::DrawIndexedIndirectArgs` layout) followed by the indirect dispatch arguments of
/// the compact pass (`wgpu::util::DispatchIndirectArgs` layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct Counters {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
}

const _: () = assert!(size_of::<Counters>() == 32);
const _: () = assert!(size_of::<Counters>().is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize));

/// Byte offset of the dispatch arguments inside [`Counters`].
pub(crate) const DISPATCH_ARGS_OFFSET: u64 = std::mem::offset_of!(Counters, group_count_x) as u64;
const _: () = assert!(DISPATCH_ARGS_OFFSET == 20);
const _: () = assert!(DISPATCH_ARGS_OFFSET.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT));

impl Counters {
    /// State at the start of a frame: nothing visible, one instance, one workgroup row.
    pub(crate) const RESET: Self = Self {
        index_count: 0,
        instance_count: 1,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
        group_count_x: 0,
        group_count_y: 1,
        group_count_z: 1,
    };
}

/// Matches `struct Visible` in `terrain_cull.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct VisibleEntry {
    pub slot: u32,
    pub dst: u32,
}

const _: () = assert!(size_of::<VisibleEntry>() == 8);

/// Frustum planes as (normal, distance) with the inside positive, extracted from the clip matrix
/// (Gribb and Hartmann 2001, "Fast Extraction of Viewing Frustum Planes from the World-View-
/// Projection Matrix"). With reverse-Z and an infinite far plane (D4) the clip condition
/// `0 <= z` always holds, so the near plane is `z <= w` and the sixth plane always passes.
pub(crate) fn frustum_planes(view_proj: Mat4) -> [[f32; 4]; 6] {
    let r0 = view_proj.row(0);
    let r1 = view_proj.row(1);
    let r2 = view_proj.row(2);
    let r3 = view_proj.row(3);
    let normalize = |p: Vec4| {
        let length = p.truncate().length();
        if length > 0.0 { p / length } else { p }
    };
    [
        normalize(r3 + r0).to_array(),
        normalize(r3 - r0).to_array(),
        normalize(r3 + r1).to_array(),
        normalize(r3 - r1).to_array(),
        normalize(r3 - r2).to_array(),
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// CPU mirror of `aabb_inside` in `terrain_cull.wgsl` (positive vertex test), for tests.
#[cfg(test)]
pub(crate) fn aabb_visible(planes: &[[f32; 4]; 6], min_m: Vec3, max_m: Vec3) -> bool {
    planes.iter().all(|plane| {
        let n = Vec3::new(plane[0], plane[1], plane[2]);
        let corner = Vec3::select(n.cmpgt(Vec3::ZERO), max_m, min_m);
        n.dot(corner) + plane[3] >= 0.0
    })
}

const MAP_PENDING: u8 = 0;
const MAP_OK: u8 = 1;
const MAP_FAILED: u8 = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Copied,
    Mapping,
}

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    mapped: Arc<AtomicU8>,
    frame: u64,
    state: SlotState,
}

/// Three readback buffers so a frame's copy never waits for an earlier map to finish; a frame
/// with no free buffer skips its readback and the statistics simply age.
struct ReadbackRing {
    slots: Vec<ReadbackSlot>,
    latest: Option<(Counters, u64)>,
}

const READBACK_RING: usize = 3;

impl ReadbackRing {
    fn new(device: &wgpu::Device) -> Self {
        let slots = (0..READBACK_RING)
            .map(|i| ReadbackSlot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("terrain counters readback {i}")),
                    size: size_of::<Counters>() as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                mapped: Arc::new(AtomicU8::new(MAP_PENDING)),
                frame: 0,
                state: SlotState::Free,
            })
            .collect();
        Self {
            slots,
            latest: None,
        }
    }

    fn encode_copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        counters: &wgpu::Buffer,
        frame: u64,
    ) {
        let Some(slot) = self.slots.iter_mut().find(|s| s.state == SlotState::Free) else {
            return;
        };
        encoder.copy_buffer_to_buffer(counters, 0, &slot.buffer, 0, size_of::<Counters>() as u64);
        slot.frame = frame;
        slot.state = SlotState::Copied;
    }

    fn after_submit(&mut self) {
        for slot in self
            .slots
            .iter_mut()
            .filter(|s| s.state == SlotState::Copied)
        {
            slot.mapped.store(MAP_PENDING, Ordering::Release);
            let flag = slot.mapped.clone();
            slot.buffer
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    flag.store(
                        if result.is_ok() { MAP_OK } else { MAP_FAILED },
                        Ordering::Release,
                    );
                });
            slot.state = SlotState::Mapping;
        }
    }

    /// Collects finished maps. Never waits: `PollType::Poll` only fires callbacks of completed
    /// submissions.
    fn poll(&mut self, device: &wgpu::Device) {
        if !self.slots.iter().any(|s| s.state == SlotState::Mapping) {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in self
            .slots
            .iter_mut()
            .filter(|s| s.state == SlotState::Mapping)
        {
            match slot.mapped.load(Ordering::Acquire) {
                MAP_PENDING => {}
                MAP_OK => {
                    let counters = {
                        match slot.buffer.get_mapped_range(..) {
                            Ok(view) => Some(*bytemuck::from_bytes::<Counters>(&view)),
                            Err(_) => None,
                        }
                    };
                    slot.buffer.unmap();
                    slot.state = SlotState::Free;
                    if let Some(counters) = counters
                        && self.latest.is_none_or(|(_, frame)| frame < slot.frame)
                    {
                        self.latest = Some((counters, slot.frame));
                    }
                }
                _ => {
                    tracing::warn!("terrain counters readback failed to map");
                    // A failed map leaves the buffer unmapped, so no unmap here.
                    slot.state = SlotState::Free;
                }
            }
        }
    }
}

/// Owns the culling pipelines, their buffers and bind groups. The chunk table and frame uniform
/// belong to the terrain pass and are rebound here when the table buffer grows.
pub(crate) struct CullPass {
    shared_layout: wgpu::BindGroupLayout,
    cull_pipeline_layout: wgpu::PipelineLayout,
    compact_pipeline_layout: wgpu::PipelineLayout,
    cull_pipeline: wgpu::ComputePipeline,
    compact_pipeline: wgpu::ComputePipeline,
    counters: wgpu::Buffer,
    visible: wgpu::Buffer,
    compacted: wgpu::Buffer,
    shared_bind_group: wgpu::BindGroup,
    counters_bind_group: wgpu::BindGroup,
    compact_bind_group: wgpu::BindGroup,
    readback: ReadbackRing,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl CullPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        uniform: &wgpu::Buffer,
        uniform_size: u64,
        chunk_table: &wgpu::Buffer,
        index_pool: &wgpu::Buffer,
        index_bytes: usize,
    ) -> Self {
        let shared_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain cull shared"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(uniform_size),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let counters_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain cull counters"),
            entries: &[storage_entry(0, false)],
        });
        let compact_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain compact"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain cull"),
            bind_group_layouts: &[Some(&shared_layout), Some(&counters_layout)],
            immediate_size: 0,
        });
        // The compact pass dispatches indirectly from the counters buffer, which a pass cannot
        // also bind as storage, so its layout leaves group 1 empty.
        let compact_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain compact"),
                bind_group_layouts: &[Some(&shared_layout), None, Some(&compact_layout)],
                immediate_size: 0,
            });
        let (cull_pipeline, compact_pipeline) = build_pipelines(
            device,
            &cull_pipeline_layout,
            &compact_pipeline_layout,
            CULL_WGSL,
        )
        .expect("the built in terrain cull shader must compile");

        let counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain counters"),
            size: size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let visible = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain visible list"),
            size: (MAX_SLOTS * size_of::<VisibleEntry>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // The sum of visible index counts never exceeds the index pool, so the compacted copy is
        // the pool's size. COPY_SRC lets tests and tools read it back.
        let compacted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain compacted indices"),
            size: index_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shared_bind_group =
            shared_bind_group(device, &shared_layout, uniform, chunk_table, &visible);
        let counters_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain cull counters"),
            layout: &counters_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counters.as_entire_binding(),
            }],
        });
        let compact_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain compact"),
            layout: &compact_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: index_pool.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compacted.as_entire_binding(),
                },
            ],
        });
        Self {
            shared_layout,
            cull_pipeline_layout,
            compact_pipeline_layout,
            cull_pipeline,
            compact_pipeline,
            counters,
            visible,
            compacted,
            shared_bind_group,
            counters_bind_group,
            compact_bind_group,
            readback: ReadbackRing::new(device),
        }
    }

    /// Rebuilds the bind group that references the chunk table after the table buffer grew.
    pub(crate) fn rebind_chunk_table(
        &mut self,
        device: &wgpu::Device,
        uniform: &wgpu::Buffer,
        chunk_table: &wgpu::Buffer,
    ) {
        self.shared_bind_group = shared_bind_group(
            device,
            &self.shared_layout,
            uniform,
            chunk_table,
            &self.visible,
        );
    }

    pub(crate) fn reload(&mut self, device: &wgpu::Device, source: &str) -> Result<(), String> {
        let (cull, compact) = build_pipelines(
            device,
            &self.cull_pipeline_layout,
            &self.compact_pipeline_layout,
            source,
        )?;
        self.cull_pipeline = cull;
        self.compact_pipeline = compact;
        Ok(())
    }

    pub(crate) fn counters(&self) -> &wgpu::Buffer {
        &self.counters
    }

    pub(crate) fn compacted(&self) -> &wgpu::Buffer {
        &self.compacted
    }

    /// Queued before the frame's commands, so the GPU sees zeroed counters when `cull` runs.
    pub(crate) fn reset_counters(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.counters, 0, bytemuck::bytes_of(&Counters::RESET));
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, slot_count: u32) {
        profiling::scope!("terrain cull");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_pipeline);
            pass.set_bind_group(0, &self.shared_bind_group, &[]);
            pass.set_bind_group(1, &self.counters_bind_group, &[]);
            pass.dispatch_workgroups(slot_count.div_ceil(CULL_WORKGROUP_SIZE), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain compact"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compact_pipeline);
            pass.set_bind_group(0, &self.shared_bind_group, &[]);
            pass.set_bind_group(2, &self.compact_bind_group, &[]);
            pass.dispatch_workgroups_indirect(&self.counters, DISPATCH_ARGS_OFFSET);
        }
    }

    /// Encoded after the draw so the copy sees the frame's final counters.
    pub(crate) fn encode_readback(&mut self, encoder: &mut wgpu::CommandEncoder, frame: u64) {
        profiling::scope!("terrain cull readback");
        self.readback.encode_copy(encoder, &self.counters, frame);
    }

    pub(crate) fn after_submit(&mut self) {
        self.readback.after_submit();
    }

    pub(crate) fn poll_readback(&mut self, device: &wgpu::Device) {
        profiling::scope!("terrain cull readback");
        self.readback.poll(device);
    }

    pub(crate) fn cull_stats(&self, frame_now: u64) -> Option<CullStats> {
        self.readback.latest.map(|(counters, frame)| CullStats {
            visible_chunks: counters.group_count_x,
            triangles: counters.index_count / 3,
            frames_old: frame_now.saturating_sub(frame).min(u64::from(u32::MAX)) as u32,
        })
    }
}

fn shared_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    chunk_table: &wgpu::Buffer,
    visible: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain cull shared"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: chunk_table.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: visible.as_entire_binding(),
            },
        ],
    })
}

/// Builds both compute pipelines inside one validation error scope so a bad edit to either
/// entry point keeps both previous pipelines (engineering rule 9).
fn build_pipelines(
    device: &wgpu::Device,
    cull_layout: &wgpu::PipelineLayout,
    compact_layout: &wgpu::PipelineLayout,
    source: &str,
) -> Result<(wgpu::ComputePipeline, wgpu::ComputePipeline), String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain_cull.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let cull = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain cull"),
        layout: Some(cull_layout),
        module: &module,
        entry_point: Some("cull"),
        compilation_options: Default::default(),
        cache: None,
    });
    let compact = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain compact"),
        layout: Some(compact_layout),
        module: &module,
        entry_point: Some("compact"),
        compilation_options: Default::default(),
        cache: None,
    });
    // wgpu-core resolves error scopes synchronously on native, so this returns immediately.
    match pollster::block_on(scope.pop()) {
        None => Ok((cull, compact)),
        Some(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vx_core::Projection;

    const FOV: f32 = std::f32::consts::FRAC_PI_3;
    const NEAR: f32 = 0.05;

    fn planes() -> [[f32; 4]; 6] {
        frustum_planes(Projection::reverse_z_infinite(FOV, 16.0 / 9.0, NEAR))
    }

    fn point_passes(planes: &[[f32; 4]; 6], p: Vec3) -> [bool; 6] {
        std::array::from_fn(|i| {
            let plane = planes[i];
            Vec3::new(plane[0], plane[1], plane[2]).dot(p) + plane[3] >= 0.0
        })
    }

    #[test]
    fn a_point_ahead_passes_every_plane() {
        assert_eq!(
            point_passes(&planes(), Vec3::new(0.0, 0.0, -10.0)),
            [true; 6]
        );
    }

    #[test]
    fn a_point_behind_the_camera_fails_the_near_plane() {
        let result = point_passes(&planes(), Vec3::new(0.0, 0.0, 5.0));
        assert!(!result[4], "near plane must reject {result:?}");
        assert!(result[5], "the far plane always passes");
        let just_inside = point_passes(&planes(), Vec3::new(0.0, 0.0, -2.0 * NEAR));
        assert!(just_inside[4]);
        let just_outside = point_passes(&planes(), Vec3::new(0.0, 0.0, -0.5 * NEAR));
        assert!(!just_outside[4]);
    }

    #[test]
    fn a_point_far_to_the_side_fails_a_side_plane() {
        let planes = planes();
        let right = point_passes(&planes, Vec3::new(1000.0, 0.0, -10.0));
        assert!(right[0] && !right[1], "{right:?}");
        let left = point_passes(&planes, Vec3::new(-1000.0, 0.0, -10.0));
        assert!(!left[0] && left[1], "{left:?}");
        let above = point_passes(&planes, Vec3::new(0.0, 1000.0, -10.0));
        assert!(above[2] && !above[3], "{above:?}");
        let below = point_passes(&planes, Vec3::new(0.0, -1000.0, -10.0));
        assert!(!below[2] && below[3], "{below:?}");
    }

    #[test]
    fn a_box_straddling_a_plane_is_visible() {
        let planes = planes();
        assert!(aabb_visible(
            &planes,
            Vec3::new(-5.0, -5.0, -20.0),
            Vec3::new(5.0, 5.0, 10.0)
        ));
        assert!(aabb_visible(
            &planes,
            Vec3::new(5.0, -5.0, -20.0),
            Vec3::new(2000.0, 5.0, -10.0)
        ));
        assert!(!aabb_visible(
            &planes,
            Vec3::new(-5.0, -5.0, 1.0),
            Vec3::new(5.0, 5.0, 10.0)
        ));
        assert!(!aabb_visible(
            &planes,
            Vec3::new(1000.0, -5.0, -20.0),
            Vec3::new(2000.0, 5.0, -10.0)
        ));
    }

    #[test]
    fn planes_are_unit_length_except_the_far_plane() {
        for (i, plane) in planes().iter().enumerate().take(5) {
            let n = Vec3::new(plane[0], plane[1], plane[2]).length();
            assert!((n - 1.0).abs() < 1.0e-5, "plane {i} has length {n}");
        }
        assert_eq!(planes()[5], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn counters_reset_matches_the_indirect_argument_layouts() {
        let draw: wgpu::util::DrawIndexedIndirectArgs = wgpu::util::DrawIndexedIndirectArgs {
            index_count: 0,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };
        let dispatch = wgpu::util::DispatchIndirectArgs { x: 0, y: 1, z: 1 };
        let bytes = bytemuck::bytes_of(&Counters::RESET);
        assert_eq!(&bytes[..20], draw.as_bytes());
        assert_eq!(&bytes[20..], dispatch.as_bytes());
    }
}
