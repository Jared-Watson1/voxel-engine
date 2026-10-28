use super::*;
use crate::cull::{Counters, VisibleEntry, aabb_visible, horizon_hidden};
use crate::{PlanetView, SunLight};
use vx_core::mesh::{FLAG_TRANSITION_FULL_RES, decode_octahedral, secondary_position_voxels};
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

fn adapter_backend() -> String {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .map(|a| format!("{} on {:?}", a.get_info().name, a.get_info().backend))
        .unwrap_or_default()
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

fn readback_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
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

fn placement(offset_m: Vec3) -> ChunkPlacement {
    ChunkPlacement {
        offset_m,
        texture_phase_m: Vec3::ZERO,
        voxel_size_m: 1.0,
        transition_mask: 0,
        visible: true,
    }
}

fn upload(
    pass: &mut TerrainPass,
    queue: &wgpu::Queue,
    vertices: &[PackedVertex],
    indices: &[u32],
    offset_m: Vec3,
) -> Result<MeshHandle, PoolError> {
    pass.upload_mesh(
        queue,
        &MeshUpload {
            vertices,
            indices,
            regular_index_count: indices.len() as u32,
            transition_index_counts: [0; 6],
        },
        placement(offset_m),
    )
}

fn frame_input(view_proj: Mat4, camera_offset_m: Vec3, planet: PlanetView) -> FrameInput<'static> {
    FrameInput {
        view_proj,
        camera_offset_m,
        planet,
        sun: SunLight::default(),
        moon: None,
        exposure_ev: 0.0,
        egui: None,
    }
}

/// No horizon culling: the occluder radius is 0.
const NO_PLANET: PlanetView = PlanetView {
    centre_m: Vec3::new(0.0, -1.0e6, 0.0),
    radius_m: 1.0e6,
    occluder_radius_m: 0.0,
};

const TEST_POOLS: TerrainPoolConfig = TerrainPoolConfig {
    vertex_bytes: 1 << 16,
    index_bytes: 1 << 16,
};

/// What one cull and compact produced.
struct CullResult {
    counters: Counters,
    visible: Vec<VisibleEntry>,
    compacted: Vec<u32>,
}

impl CullResult {
    fn visible_slots(&self) -> Vec<u32> {
        let mut slots: Vec<u32> = self.visible.iter().map(|v| v.slot).collect();
        slots.sort_unstable();
        slots
    }
}

fn run_cull(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &mut TerrainPass,
    frame: &FrameInput,
) -> CullResult {
    pass.prepare_frame(device, queue, frame);
    let compacted_bytes = 4096;
    let visible_bytes = (64 * size_of::<VisibleEntry>()) as u64;
    let counters_bytes = size_of::<Counters>() as u64;
    let compacted_copy = readback_buffer(device, compacted_bytes);
    let visible_copy = readback_buffer(device, visible_bytes);
    let counters_copy = readback_buffer(device, counters_bytes);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test cull"),
    });
    pass.encode_cull(&mut encoder);
    let cull = pass.cull_pass();
    encoder.copy_buffer_to_buffer(cull.compacted(), 0, &compacted_copy, 0, compacted_bytes);
    encoder.copy_buffer_to_buffer(cull.visible(), 0, &visible_copy, 0, visible_bytes);
    encoder.copy_buffer_to_buffer(cull.counters(), 0, &counters_copy, 0, counters_bytes);
    pass.encode_readback(&mut encoder);
    queue.submit([encoder.finish()]);
    pass.after_submit();
    let counters: Counters = *bytemuck::from_bytes(&read_buffer(device, &counters_copy));
    let visible: Vec<VisibleEntry> = bytemuck::cast_slice(&read_buffer(device, &visible_copy))
        [..counters.group_count_x as usize]
        .to_vec();
    let compacted = bytemuck::cast_slice(&read_buffer(device, &compacted_copy)).to_vec();
    CullResult {
        counters,
        visible,
        compacted,
    }
}

fn read_table(device: &wgpu::Device, queue: &wgpu::Queue, pass: &TerrainPass) -> Vec<ChunkEntry> {
    let bytes = (pass.table.len() * ENTRY_BYTES) as u64;
    let copy = readback_buffer(device, bytes);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test table"),
    });
    encoder.copy_buffer_to_buffer(&pass.table_buffer, 0, &copy, 0, bytes);
    queue.submit([encoder.finish()]);
    bytemuck::cast_slice(&read_buffer(device, &copy)).to_vec()
}

#[test]
fn chunk_entry_packs_counts_mask_and_visibility() {
    let mesh = ResidentMesh {
        vertex_range: Range {
            offset: 0,
            size: 16,
        },
        index_range: Range { offset: 0, size: 4 },
        vertex_base: 7,
        index_base: 9,
        regular_count: 300,
        transition_counts: [1, 2, 3, 4, 5, 65535],
        placement: ChunkPlacement {
            offset_m: Vec3::new(1.0, 2.0, 3.0),
            texture_phase_m: Vec3::new(0.5, 1.5, 2.5),
            voxel_size_m: 8.0,
            transition_mask: 0b10_0101,
            visible: true,
        },
    };
    let entry = ChunkEntry::new(&mesh);
    assert_eq!(entry.offset_m, [1.0, 2.0, 3.0]);
    assert_eq!(entry.texture_phase_m, [0.5, 1.5, 2.5]);
    assert_eq!((entry.vertex_base, entry.index_base), (7, 9));
    assert_eq!(entry.regular_count, 300);
    assert_eq!(
        entry.transition_counts,
        [1 | 2 << 16, 3 | 4 << 16, 5 | 65535 << 16]
    );
    assert_eq!(entry.voxel_size_m, 8.0);
    assert_eq!(entry.flags, FLAG_VISIBLE | 0b10_0101 << CHUNK_MASK_SHIFT);

    let hidden = ResidentMesh {
        placement: ChunkPlacement {
            visible: false,
            transition_mask: 0xff,
            ..mesh.placement
        },
        ..mesh
    };
    assert_eq!(
        ChunkEntry::new(&hidden).flags,
        0x3f << CHUNK_MASK_SHIFT,
        "hidden, and only six mask bits"
    );
    let bytes = bytemuck::bytes_of(&entry);
    assert_eq!(bytes.len(), 64);
    assert_eq!(&bytes[48..52], &8.0f32.to_ne_bytes());
}

#[test]
fn frame_uniform_carries_the_camera_offset_and_planet() {
    let planet = PlanetView {
        centre_m: Vec3::new(0.0, -10_100.0, 0.0),
        radius_m: 10_000.0,
        occluder_radius_m: 9_000.0,
    };
    let u = FrameUniform::new(Mat4::IDENTITY, Vec3::new(10.0, 20.0, 30.0), &planet, 7);
    assert_eq!(
        u.camera_offset_m,
        [10.0, 20.0, 30.0, MATERIAL_TILE_M as f32]
    );
    assert_eq!(u.planet, [0.0, -10_100.0, 0.0, 9_000.0]);
    assert_eq!(u.slot_count, 7);
    assert_eq!(u.planes[5], [0.0, 0.0, 0.0, 1.0]);
    let negative = PlanetView {
        occluder_radius_m: -1.0,
        ..planet
    };
    assert_eq!(
        FrameUniform::new(Mat4::IDENTITY, Vec3::ZERO, &negative, 0).planet[3],
        0.0,
        "a negative occluder disables the horizon test"
    );
}

#[test]
fn dirty_runs_join_small_gaps() {
    let mut dirty = vec![9, 3, 0, 1, 20, 4];
    assert_eq!(dirty_runs(&mut dirty, 0), [0..2, 3..5, 9..10, 20..21]);
    assert!(dirty.is_empty(), "drained");
    let mut dirty = vec![9, 3, 0, 1, 20, 4];
    assert_eq!(dirty_runs(&mut dirty, 4), [0..10, 20..21]);
    assert_eq!(dirty_runs(&mut Vec::new(), 4), []);
}

#[test]
fn checked_counts_rejects_inconsistent_meshes() {
    let indices = vec![0u32; 12];
    let mesh = |regular, transitions| MeshUpload {
        vertices: &[],
        indices: &indices,
        regular_index_count: regular,
        transition_index_counts: transitions,
    };
    assert_eq!(
        checked_counts(&mesh(6, [3, 0, 0, 0, 0, 3])),
        Some([3, 0, 0, 0, 0, 3])
    );
    assert_eq!(checked_counts(&mesh(6, [3, 0, 0, 0, 0, 0])), None);
    assert_eq!(checked_counts(&mesh(12, [u32::MAX, 1, 0, 0, 0, 0])), None);
    let many = vec![0u32; 70_000];
    let big = MeshUpload {
        vertices: &[],
        indices: &many,
        regular_index_count: 0,
        transition_index_counts: [70_000, 0, 0, 0, 0, 0],
    };
    assert_eq!(checked_counts(&big), None, "a transition range over u16");
}

const SHARED_BEGIN: &str = "// BEGIN vertex decoding and secondary position";
const SHARED_END: &str = "// END vertex decoding and secondary position";

/// The functions the vertex shader uses to decode a vertex and displace it, cut from the shipped
/// shader text so the test compiles exactly that code.
fn shared_wgsl() -> &'static str {
    let begin = TERRAIN_WGSL.find(SHARED_BEGIN).expect("BEGIN marker");
    let end = TERRAIN_WGSL.find(SHARED_END).expect("END marker");
    &TERRAIN_WGSL[begin..end]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ParityCase {
    words: [u32; 4],
    position_voxels: [f32; 3],
    flags: u32,
    normal: [f32; 3],
    mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ParityOut {
    displaced: [f32; 3],
    pad0: u32,
    normal: [f32; 3],
    pad1: u32,
    position_voxels: [f32; 3],
    pad2: u32,
    displaced_decoded: [f32; 3],
    pad3: u32,
}

const _: () = assert!(size_of::<ParityCase>() == 48);
const _: () = assert!(size_of::<ParityOut>() == 64);

const PARITY_KERNEL: &str = r#"
struct Case {
    words: vec4<u32>,
    position_voxels: vec3<f32>,
    flags: u32,
    normal: vec3<f32>,
    mask: u32,
};

struct Out {
    displaced: vec3<f32>,
    normal: vec3<f32>,
    position_voxels: vec3<f32>,
    displaced_decoded: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> results: array<Out>;
// x is 0, like `Frame::zero` in the vertex shader.
@group(0) @binding(2) var<uniform> zero: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&cases) {
        return;
    }
    let c = cases[i];
    var out: Out;
    out.displaced =
        secondary_position_voxels(c.position_voxels, c.normal, c.flags, c.mask, zero.x);
    let p = unpack_position_voxels(c.words);
    let n = unpack_normal(c.words);
    out.normal = n;
    out.position_voxels = p;
    out.displaced_decoded =
        secondary_position_voxels(p, n, vertex_flags(c.words), c.mask, zero.x);
    results[i] = out;
}
"#;

fn bits(v: Vec3) -> [u32; 3] {
    v.to_array().map(f32::to_bits)
}

/// The vertex shader's displacement against `vx_core::mesh::secondary_position_voxels`, bit for
/// bit (D22). Part one feeds both the same f32 position and normal; part two lets the GPU decode
/// the packed vertex and feeds its decoded normal to the Rust reference, which is the vertex
/// shader's path. Normals are decoded from real snorm8 pairs so the products are not exact.
#[test]
fn secondary_position_matches_the_rust_reference_bit_for_bit() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter available, skipping the displacement parity test");
        return;
    };
    eprintln!("displacement parity on {}", adapter_backend());

    let positions_voxels = [
        0.0f32,
        0.25,
        0.5,
        1.0,
        31.0,
        32.0,
        341.0 / 1024.0,
        32_601.0 / 1024.0,
    ];
    let normals: [[i8; 2]; 10] = [
        [0, 0],
        [127, 0],
        [0, 127],
        [-127, 0],
        [37, -81],
        [-100, 20],
        [90, 90],
        [5, -120],
        [-64, -64],
        [113, -7],
    ];
    let mut cases = Vec::new();
    for (pi, &x) in positions_voxels.iter().enumerate() {
        for (pj, &y) in positions_voxels.iter().enumerate() {
            for (pk, &z) in positions_voxels.iter().enumerate() {
                for (ni, &[nx, ny]) in normals.iter().enumerate() {
                    for mask in 0..64u32 {
                        let flags = if (pi + pj + pk + ni + mask as usize).is_multiple_of(7) {
                            u32::from(FLAG_TRANSITION_FULL_RES)
                        } else {
                            0
                        };
                        let q = |v: f32| (v * 1024.0) as u32;
                        let words = [
                            q(x) | q(y) << 16,
                            q(z) | u32::from(nx as u8) << 16 | u32::from(ny as u8) << 24,
                            0,
                            flags << 16,
                        ];
                        let unpacked = PackedVertex { words }.unpack();
                        assert_eq!(unpacked.position_voxels, Vec3::new(x, y, z));
                        cases.push(ParityCase {
                            words,
                            position_voxels: [x, y, z],
                            flags,
                            normal: decode_octahedral([nx, ny]).to_array(),
                            mask,
                        });
                    }
                }
            }
        }
    }

    let source = format!("{}\n{}", shared_wgsl(), PARITY_KERNEL);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("displacement parity"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("displacement parity"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("parity kernel: {error}");
    }
    let input = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity cases"),
        size: (cases.len() * size_of::<ParityCase>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&cases));
    let output_bytes = (cases.len() * size_of::<ParityOut>()) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity results"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let copy = readback_buffer(&device, output_bytes);
    let zero = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity zero"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&zero, 0, &[0; 16]);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: zero.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parity"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("parity"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &copy, 0, output_bytes);
    queue.submit([encoder.finish()]);
    let results: Vec<ParityOut> = bytemuck::cast_slice(&read_buffer(&device, &copy)).to_vec();

    let mut moved = 0usize;
    let mut mismatches = Vec::new();
    let mut signed_zero_only = 0usize;
    let mut decode_mismatches = 0usize;
    let mut normal_ulps_max = 0u32;
    for (case, out) in cases.iter().zip(&results) {
        let flags = case.flags as u16;
        let mask = case.mask as u8;
        let p = Vec3::from_array(case.position_voxels);
        let expected = secondary_position_voxels(p, Vec3::from_array(case.normal), flags, mask);
        if expected != p {
            moved += 1;
        }
        let gpu_normal = Vec3::from_array(out.normal);
        let gpu_p = Vec3::from_array(out.position_voxels);
        if bits(gpu_p) != bits(p) {
            decode_mismatches += 1;
        }
        for (a, b) in out.normal.iter().zip(&case.normal) {
            normal_ulps_max = normal_ulps_max.max(a.to_bits().abs_diff(b.to_bits()));
        }
        let expected_decoded = secondary_position_voxels(gpu_p, gpu_normal, flags, mask);
        for (gpu, cpu, part) in [
            (Vec3::from_array(out.displaced), expected, "shared inputs"),
            (
                Vec3::from_array(out.displaced_decoded),
                expected_decoded,
                "GPU decoded inputs",
            ),
        ] {
            if bits(gpu) != bits(cpu) {
                if gpu == cpu {
                    signed_zero_only += 1;
                }
                mismatches.push(format!(
                    "{part}: p {p:?} n {:?} flags {flags} mask {mask:#08b}: GPU {gpu:?} CPU {cpu:?}",
                    case.normal
                ));
            }
        }
    }
    eprintln!(
        "{} cases, {moved} displaced, GPU octahedral decode differs from the CPU by at most \
         {normal_ulps_max} ulp per component",
        cases.len()
    );
    assert!(
        moved > cases.len() / 10,
        "the cases must exercise the displacement"
    );
    assert_eq!(decode_mismatches, 0, "positions decode exactly");
    assert!(
        mismatches.is_empty(),
        "{} mismatches ({signed_zero_only} differ only in the sign of zero), first: {:#?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(8)]
    );
}

/// Index layout of a test mesh: distinct values per range so order and rebasing are checked.
fn ranged_mesh(regular: u32, transitions: [u32; 6]) -> (Vec<PackedVertex>, Vec<u32>) {
    let total = regular + transitions.iter().sum::<u32>();
    let vertices = vec![PackedVertex::default(); 64];
    let indices = (0..total).map(|i| (i * 7 + 3) % 64).collect();
    (vertices, indices)
}

/// The compacted indices a visible chunk must produce: the regular range, then each masked
/// face's range in face order, all plus the vertex base.
fn expected_indices(pass: &TerrainPass, slot: u16, indices: &[u32]) -> Vec<u32> {
    let mesh = pass.slots[usize::from(slot)].as_ref().expect("resident");
    let mut out: Vec<u32> = indices[..mesh.regular_count as usize].to_vec();
    let mut start = mesh.regular_count as usize;
    for face in 0..6 {
        let count = usize::from(mesh.transition_counts[face]);
        if mesh.placement.transition_mask & (1 << face) != 0 {
            out.extend_from_slice(&indices[start..start + count]);
        }
        start += count;
    }
    out.iter().map(|i| i + mesh.vertex_base).collect()
}

fn check_compacted(pass: &TerrainPass, result: &CullResult, meshes: &[(MeshHandle, Vec<u32>)]) {
    let mut total = 0;
    for visible in &result.visible {
        let (handle, indices) = meshes
            .iter()
            .find(|(h, _)| u32::from(h.0) == visible.slot)
            .expect("a visible slot belongs to a test mesh");
        let expected = expected_indices(pass, handle.0, indices);
        let dst = visible.dst as usize;
        assert_eq!(
            &result.compacted[dst..dst + expected.len()],
            &expected[..],
            "slot {}",
            visible.slot
        );
        total += expected.len();
    }
    assert_eq!(result.counters.index_count as usize, total);
}

/// Mixed voxel sizes, masks and hidden chunks: each visible chunk reserves its regular range plus
/// its masked faces' ranges, and the compacted copy holds exactly those, rebased.
#[test]
fn cull_and_compact_mixed_levels_masks_and_hidden_chunks() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter available, skipping the cull and compact test");
        return;
    };
    let mut pass = TerrainPass::new(&device, &queue, TEST_POOLS);
    let camera = Vec3::new(1000.0, -200.0, 500.0);
    let view_proj = Projection::reverse_z_infinite(std::f32::consts::FRAC_PI_3, 1.0, 0.05);
    let planes = frustum_planes(view_proj);

    // (camera relative min corner, voxel size, mask, visible, regular, transitions)
    let chunks: [(Vec3, f32, u8, bool, u32, [u32; 6]); 6] = [
        // Two masked faces of four with ranges.
        (
            Vec3::new(-16.0, -16.0, -60.0),
            1.0,
            0b00_0101,
            true,
            6,
            [3, 6, 3, 0, 3, 0],
        ),
        // Inside the frustum only through its own 128 m extent.
        (Vec3::new(-250.0, -16.0, -230.0), 4.0, 0, true, 6, [0; 6]),
        // Hidden by its placement.
        (Vec3::new(-16.0, -16.0, -60.0), 1.0, 0, false, 6, [0; 6]),
        // Every face masked at half the voxel size.
        (
            Vec3::new(-8.0, -8.0, -30.0),
            0.5,
            0b11_1111,
            true,
            3,
            [3, 3, 3, 6, 3, 3],
        ),
        // Behind the camera.
        (Vec3::new(-16.0, -16.0, 20.0), 1.0, 0, true, 6, [0; 6]),
        // Masked faces without ranges and ranges on unmasked faces.
        (
            Vec3::new(-32.0, -32.0, -100.0),
            2.0,
            0b10_1010,
            true,
            6,
            [3, 0, 3, 0, 3, 0],
        ),
    ];
    let wide = chunks[1].0;
    assert!(!aabb_visible(&planes, wide, wide + Vec3::splat(32.0)));
    assert!(aabb_visible(&planes, wide, wide + Vec3::splat(128.0)));

    let mut meshes = Vec::new();
    for (min, voxel_size_m, mask, visible, regular, transitions) in chunks {
        let (vertices, indices) = ranged_mesh(regular, transitions);
        let handle = pass
            .upload_mesh(
                &queue,
                &MeshUpload {
                    vertices: &vertices,
                    indices: &indices,
                    regular_index_count: regular,
                    transition_index_counts: transitions,
                },
                ChunkPlacement {
                    offset_m: min + camera,
                    texture_phase_m: Vec3::ZERO,
                    voxel_size_m,
                    transition_mask: mask,
                    visible,
                },
            )
            .unwrap();
        meshes.push((handle, indices));
    }

    let frame = frame_input(view_proj, camera, NO_PLANET);
    let result = run_cull(&device, &queue, &mut pass, &frame);
    assert_eq!(result.visible_slots(), [0, 1, 3, 5]);
    assert_eq!(result.counters.index_count, 12 + 6 + 24 + 6);
    assert_eq!(result.counters.group_count_x, 4);
    check_compacted(&pass, &result, &meshes);
    pass.poll_readback(&device);
    let stats = pass.cull_stats().expect("readback mapped after the wait");
    assert_eq!((stats.visible_chunks, stats.triangles), (4, 16));

    // A mask change, a visibility change and a free are entry writes only.
    let (a, c, d) = (meshes[0].0, meshes[2].0, meshes[3].0);
    let mut moved = pass.slots[0].as_ref().unwrap().placement;
    moved.transition_mask = 0b00_0010;
    pass.set_placement(a, moved);
    let mut shown = pass.slots[2].as_ref().unwrap().placement;
    shown.visible = true;
    pass.set_placement(c, shown);
    pass.free_mesh(d);
    let result = run_cull(&device, &queue, &mut pass, &frame);
    assert_eq!(result.visible_slots(), [0, 1, 2, 5]);
    assert_eq!(result.counters.index_count, 12 + 6 + 6 + 6);
    check_compacted(&pass, &result, &meshes);
}

/// The horizon test against an occluder 10 km in radius, 20 km ahead of the camera.
#[test]
fn horizon_culls_behind_the_planet_and_keeps_the_limb() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter available, skipping the horizon test");
        return;
    };
    let mut pass = TerrainPass::new(&device, &queue, TEST_POOLS);
    let view_proj = Projection::reverse_z_infinite(std::f32::consts::FRAC_PI_2, 1.0, 0.05);
    let planet = PlanetView {
        centre_m: Vec3::new(0.0, 0.0, -20_000.0),
        radius_m: 10_000.0,
        occluder_radius_m: 10_000.0,
    };
    let at_angle = |degrees: f32, distance_m: f32| {
        let a = degrees.to_radians();
        Vec3::new(a.sin(), 0.0, -a.cos()) * distance_m
    };
    // The tangent cone's half angle is 30 degrees and the tangent points are 17,320 m away.
    let tangent_m = (20_000.0f32 * 20_000.0 - 10_000.0 * 10_000.0).sqrt();
    let centres = [
        ("in front", Vec3::new(0.0, 0.0, -9_000.0), true),
        ("behind", Vec3::new(0.0, 0.0, -30_000.0), false),
        ("on the limb", at_angle(30.0, tangent_m), true),
        ("past the limb", at_angle(31.0, 30_000.0), true),
        ("just inside the limb", at_angle(29.0, 30_000.0), false),
    ];
    let (vertices, indices) = quad();
    for (name, centre, kept) in centres {
        let min = centre - Vec3::splat(16.0);
        assert_eq!(
            horizon_hidden(planet.centre_m, 10_000.0, min, min + 32.0),
            !kept,
            "CPU mirror: {name}"
        );
        upload(&mut pass, &queue, &vertices, &indices, min).unwrap();
    }

    let result = run_cull(
        &device,
        &queue,
        &mut pass,
        &frame_input(view_proj, Vec3::ZERO, planet),
    );
    let expected: Vec<u32> = centres
        .iter()
        .enumerate()
        .filter(|(_, (_, _, kept))| *kept)
        .map(|(slot, _)| slot as u32)
        .collect();
    assert_eq!(
        result.visible_slots(),
        expected,
        "{:?}",
        centres.map(|c| c.0)
    );

    let result = run_cull(
        &device,
        &queue,
        &mut pass,
        &frame_input(view_proj, Vec3::ZERO, NO_PLANET),
    );
    assert_eq!(
        result.visible_slots(),
        [0, 1, 2, 3, 4],
        "no occluder, no horizon culling"
    );
}

#[test]
fn only_changed_entries_are_written() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter available, skipping the table write test");
        return;
    };
    let mut pass = TerrainPass::new(&device, &queue, TEST_POOLS);
    let frame = frame_input(Mat4::IDENTITY, Vec3::ZERO, NO_PLANET);
    let (vertices, indices) = quad();
    let handles: Vec<MeshHandle> = (0..12)
        .map(|i| {
            upload(
                &mut pass,
                &queue,
                &vertices,
                &indices,
                Vec3::splat(i as f32),
            )
            .unwrap()
        })
        .collect();
    let writes = |pass: &mut TerrainPass| {
        pass.prepare_frame(&device, &queue, &frame);
        (pass.last_writes.entries, pass.last_writes.calls)
    };
    assert_eq!(writes(&mut pass), (12, 1), "uploads write their entries");
    assert_eq!(
        writes(&mut pass),
        (0, 0),
        "nothing changed, nothing written"
    );

    let mut p = pass.slots[5].as_ref().unwrap().placement;
    p.transition_mask = 0b1;
    pass.set_placement(handles[5], p);
    pass.set_placement(handles[5], p);
    assert_eq!(writes(&mut pass), (1, 1), "one placement, one entry");
    pass.set_placement(handles[5], p);
    assert_eq!(
        writes(&mut pass),
        (0, 0),
        "an unchanged placement writes nothing"
    );

    for &h in &[handles[0], handles[11]] {
        let mut p = pass.slots[usize::from(h.0)].as_ref().unwrap().placement;
        p.visible = false;
        pass.set_placement(h, p);
    }
    assert_eq!(
        writes(&mut pass),
        (2, 2),
        "distant slots are separate writes"
    );

    // A rebase moves every placement.
    for &h in &handles {
        let mut p = pass.slots[usize::from(h.0)].as_ref().unwrap().placement;
        p.offset_m -= Vec3::new(4000.0, 0.0, 0.0);
        pass.set_placement(h, p);
    }
    assert_eq!(
        writes(&mut pass),
        (12, 1),
        "a rebase writes every entry once"
    );

    pass.free_mesh(handles[7]);
    assert_eq!(writes(&mut pass), (1, 1), "a free clears its entry");
    let gpu = read_table(&device, &queue, &pass);
    assert_eq!(gpu, pass.table, "the GPU table matches the CPU mirror");
    assert_eq!(gpu[7], ChunkEntry::zeroed());
    assert_eq!(gpu[0].offset_m, [-4000.0, 0.0, 0.0]);
    assert_eq!(gpu[0].flags & FLAG_VISIBLE, 0);
    assert_eq!(gpu[5].flags, FLAG_VISIBLE | 1 << CHUNK_MASK_SHIFT);
}

#[test]
fn table_grows_and_rewrites_every_entry() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter available, skipping the table growth test");
        return;
    };
    let mut pass = TerrainPass::new(&device, &queue, TEST_POOLS);
    let frame = frame_input(Mat4::IDENTITY, Vec3::ZERO, NO_PLANET);
    let vertex = [PackedVertex::default()];
    let count = INITIAL_TABLE_ENTRIES + 1;
    for i in 0..count {
        upload(
            &mut pass,
            &queue,
            &vertex,
            &[0, 0, 0],
            Vec3::splat(i as f32),
        )
        .unwrap();
    }
    pass.prepare_frame(&device, &queue, &frame);
    assert_eq!(
        pass.table_buffer.size() as usize,
        2 * INITIAL_TABLE_ENTRIES * ENTRY_BYTES
    );
    assert_eq!(
        (pass.last_writes.entries, pass.last_writes.calls),
        (count, 1)
    );
    let gpu = read_table(&device, &queue, &pass);
    assert_eq!(gpu, pass.table);
    assert_eq!(gpu[count - 1].offset_m, [(count - 1) as f32; 3]);

    let mut p = pass.slots[3].as_ref().unwrap().placement;
    p.voxel_size_m = 2.0;
    pass.set_placement(MeshHandle(3), p);
    pass.prepare_frame(&device, &queue, &frame);
    assert_eq!((pass.last_writes.entries, pass.last_writes.calls), (1, 1));
    assert_eq!(read_table(&device, &queue, &pass)[3].voxel_size_m, 2.0);
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
        TerrainPoolConfig {
            vertex_bytes: 4 * VERTEX_BYTES * 4,
            index_bytes: 8 * INDEX_BYTES,
        },
    );
    let frame = frame_input(Mat4::IDENTITY, Vec3::ZERO, NO_PLANET);
    let (vertices, indices) = quad();
    assert!(matches!(
        upload(&mut pass, &queue, &[], &indices, Vec3::ZERO),
        Err(PoolError::EmptyMesh)
    ));
    let inconsistent = MeshUpload {
        vertices: &vertices,
        indices: &indices,
        regular_index_count: 3,
        transition_index_counts: [0; 6],
    };
    assert!(matches!(
        pass.upload_mesh(&queue, &inconsistent, placement(Vec3::ZERO)),
        Err(PoolError::EmptyMesh)
    ));
    assert!(pass.slots.is_empty(), "a rejected mesh takes no slot");

    let first = upload(&mut pass, &queue, &vertices, &indices, Vec3::ZERO).unwrap();
    pass.prepare_frame(&device, &queue, &frame);
    let before = pass.pool_stats();
    let err = upload(&mut pass, &queue, &vertices, &indices, Vec3::ZERO).unwrap_err();
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
    assert_eq!(
        pass.table[1],
        ChunkEntry::zeroed(),
        "no entry for the failure"
    );
    assert!(pass.dirty.is_empty(), "and no entry write");

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
    let again = upload(&mut pass, &queue, &vertices, &indices, Vec3::ZERO).unwrap();
    assert_eq!(again, MeshHandle(0), "the lowest free slot is reused");
}

#[test]
fn shader_reload_keeps_the_pipeline_on_errors() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter available, skipping the shader reload test");
        return;
    };
    let mut pass = TerrainPass::new(&device, &queue, TEST_POOLS);
    assert_eq!(pass.reload(&device, TERRAIN_WGSL), Ok(()));
    assert_eq!(pass.reload_cull(&device, crate::cull::CULL_WGSL), Ok(()));
    assert!(pass.reload(&device, "fn broken(").is_err());
    assert!(pass.reload_cull(&device, "fn broken(").is_err());
    let no_entry_points = shared_wgsl();
    assert!(
        pass.reload(&device, no_entry_points).is_err(),
        "missing entry points fail at pipeline creation"
    );
    // The previous pipelines still cull.
    let (vertices, indices) = quad();
    upload(
        &mut pass,
        &queue,
        &vertices,
        &indices,
        Vec3::new(-16.0, -16.0, -40.0),
    )
    .unwrap();
    let view_proj = Projection::reverse_z_infinite(std::f32::consts::FRAC_PI_3, 1.0, 0.05);
    let result = run_cull(
        &device,
        &queue,
        &mut pass,
        &frame_input(view_proj, Vec3::ZERO, NO_PLANET),
    );
    assert_eq!(result.counters.index_count, 6);
}

/// Draws one quad into a small G-buffer: alpha 1 and the unit normal where the terrain is, the
/// clear value elsewhere, at the place the camera offset and the chunk offset put it.
#[test]
fn draw_writes_albedo_and_normal_to_the_gbuffer() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter available, skipping the G-buffer test");
        return;
    };
    const SIZE: u32 = 32;
    const ROW_BYTES: u32 = 256;
    let mut pass = TerrainPass::new(&device, &queue, TEST_POOLS);
    let (vertices, indices) = quad();
    let camera = Vec3::new(-3000.0, 120.0, 2500.0);
    // The quad lies at z = -24 in front of the camera and covers x from 0 to 32: the right half
    // of the view.
    upload(
        &mut pass,
        &queue,
        &vertices,
        &indices,
        camera + Vec3::new(0.0, -16.0, -40.0),
    )
    .unwrap();
    let view_proj = Projection::reverse_z_infinite(std::f32::consts::FRAC_PI_3, 1.0, 0.05);
    pass.prepare_frame(&device, &queue, &frame_input(view_proj, camera, NO_PLANET));

    let target = |format, label| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let albedo = target(GBUFFER_ALBEDO_FORMAT, "test albedo");
    let normal = target(GBUFFER_NORMAL_FORMAT, "test normal");
    let depth = target(DEPTH_FORMAT, "test depth");
    let views = [&albedo, &normal, &depth].map(|t| t.create_view(&Default::default()));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test draw"),
    });
    pass.encode_cull(&mut encoder);
    {
        let clear = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };
        let attachment = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: clear,
            })
        };
        let mut render = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test terrain"),
            color_attachments: &[attachment(&views[0]), attachment(&views[1])],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &views[2],
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.draw(&mut render);
    }
    let copies = [&albedo, &normal].map(|texture| {
        let copy = readback_buffer(&device, u64::from(ROW_BYTES * SIZE));
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &copy,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW_BYTES),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );
        copy
    });
    queue.submit([encoder.finish()]);
    let albedo = read_buffer(&device, &copies[0]);
    let normal = read_buffer(&device, &copies[1]);
    let row = (SIZE / 2) as usize * ROW_BYTES as usize;
    let texel = |bytes: &[u8], x: usize, size: usize| bytes[row + x * size..][..size].to_vec();

    let left = texel(&albedo, 4, 4);
    let right = texel(&albedo, 28, 4);
    assert_eq!(left[3], 0, "nothing drawn on the left: {left:?}");
    assert_eq!(right[3], 255, "terrain alpha is 1: {right:?}");
    assert!(right[..3] != [0, 0, 0], "albedo written: {right:?}");
    let half = |bytes: Vec<u8>| -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_ne_bytes([b[0], b[1]]))
            .collect()
    };
    // f16 bits: 0x3c00 is 1.0.
    assert_eq!(
        half(texel(&normal, 28, 8)),
        [0, 0, 0x3c00, 0],
        "unit +Z normal, w 0"
    );
    assert_eq!(half(texel(&normal, 4, 8)), [0, 0, 0, 0]);
}
