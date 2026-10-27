//! Device, surface, passes and pipelines.
//!
//! The renderer receives packed meshes in chunk local coordinates and camera relative parameters
//! only; no world position ever reaches it (D4). It never depends on the voxel, generation or
//! physics crates.
//!
//! Terrain (D10, D19): meshes live in pooled vertex and index buffers ([`Renderer::upload_mesh`]),
//! a compute pass culls every resident chunk against the frustum and compacts the visible indices,
//! and one indexed indirect draw renders them with vertex pulling and triplanar materials.

mod capture;
mod terrain;

use std::sync::Arc;

use glam::{Mat4, Vec3};
use vx_core::{Budgets, PackedVertex};
use winit::window::Window;

pub use capture::CapturedImage;
pub use terrain::FrameUniform;

/// Sizes of the pooled terrain buffers. The sum, plus material textures, stays under
/// [`Budgets::GPU_TERRAIN_BYTES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainPoolConfig {
    pub vertex_bytes: usize,
    pub index_bytes: usize,
}

impl Default for TerrainPoolConfig {
    /// 256 MiB of vertices (16 M packed vertices) and 128 MiB of indices: about 5,000 surface
    /// chunks of 3,000 vertices at the design defaults, with room for the compacted index copy.
    fn default() -> Self {
        Self {
            vertex_bytes: 256 << 20,
            index_bytes: 128 << 20,
        }
    }
}

impl TerrainPoolConfig {
    /// Bytes the pools and the compacted index copy occupy on the GPU.
    pub fn gpu_bytes(&self) -> usize {
        self.vertex_bytes + 2 * self.index_bytes
    }
}

const _: () = assert!(
    (256 << 20) + 2 * (128 << 20) <= Budgets::GPU_TERRAIN_BYTES,
    "default pools exceed the GPU terrain budget"
);

/// A resident chunk mesh: the slot in the GPU chunk table, which the packed vertices carry (D14).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u16);

#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("vertex pool full: {needed} bytes needed, largest free range {largest_free} bytes")]
    VertexPoolFull { needed: usize, largest_free: usize },
    #[error("index pool full: {needed} bytes needed, largest free range {largest_free} bytes")]
    IndexPoolFull { needed: usize, largest_free: usize },
    #[error("no free chunk slot (65536 resident meshes)")]
    NoFreeSlot,
    #[error("an empty mesh cannot be uploaded")]
    EmptyMesh,
}

/// Material albedo layers for the texture array: RGBA8 sRGB, `size` by `size`, layer index equal
/// to the material id.
#[derive(Debug, Clone)]
pub struct MaterialTextures {
    pub size: u32,
    pub layers: Vec<Vec<u8>>,
}

/// Repeat length of the material textures in metres. Triplanar coordinates are chunk local
/// positions plus [`ChunkDraw::texture_phase_m`], so they are stable as the camera moves without
/// any world position reaching the shader (D4).
pub const MATERIAL_TILE_M: f64 = 4.0;

/// One resident chunk to draw this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkDraw {
    pub handle: MeshHandle,
    /// Camera relative position of the chunk origin in metres, computed in f64 by the caller.
    pub offset_m: Vec3,
    /// Chunk origin modulo [`MATERIAL_TILE_M`] along each axis (0 to the tile length), computed in
    /// f64 by the caller, so texture coordinates do not depend on the camera.
    pub texture_phase_m: Vec3,
}

/// Pool occupancy for the overlay and the GPU budget check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    pub vertex_bytes_used: usize,
    pub vertex_bytes_capacity: usize,
    pub index_bytes_used: usize,
    pub index_bytes_capacity: usize,
    /// Largest allocatable range in each pool, which shows fragmentation.
    pub largest_free_vertex_bytes: usize,
    pub largest_free_index_bytes: usize,
    pub meshes: usize,
    /// Material textures and other fixed terrain resources.
    pub texture_bytes: usize,
}

impl PoolStats {
    pub fn gpu_bytes(&self) -> usize {
        self.vertex_bytes_capacity + 2 * self.index_bytes_capacity + self.texture_bytes
    }
}

/// Culling results read back from the GPU; `frames_old` says how stale they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CullStats {
    pub visible_chunks: u32,
    pub triangles: u32,
    pub frames_old: u32,
}

/// Depth format for every 3D pass: 32 bit float with reverse-Z (D4).
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Depth clear value for reverse-Z: 0.0 is infinitely far.
pub const DEPTH_CLEAR: f32 = 0.0;
/// Depth test for reverse-Z: nearer fragments have greater depth.
pub const DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::Greater;

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("creating the surface: {0}")]
    Surface(#[from] wgpu::CreateSurfaceError),
    #[error("no compatible GPU adapter: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error("requesting the device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("the surface is not supported by the adapter")]
    UnsupportedSurface,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderSettings {
    /// Wait for vertical blank. Off gives uncapped frame times for measurements.
    pub vsync: bool,
    pub pools: TerrainPoolConfig,
}

/// A triangle mesh in metres relative to its own origin, counter clockwise front faces.
pub struct MeshData<'a> {
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
    pub indices: &'a [u32],
}

/// egui output for one frame, already tessellated.
pub struct EguiFrame<'a> {
    pub primitives: &'a [egui::ClippedPrimitive],
    /// Consumed by [`Renderer::render`]; egui asserts in debug builds that every delta is applied.
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// Everything the renderer needs for one frame, all camera relative.
pub struct FrameInput<'a> {
    /// Projection times the rotation only view matrix.
    pub view_proj: Mat4,
    /// Camera relative position of the mesh origin, computed in f64 by the caller (M0 path,
    /// removed when the pooled path lands).
    pub mesh_offset_m: Vec3,
    /// Every resident chunk with its camera relative offset; the compute pass culls them.
    pub chunks: &'a [ChunkDraw],
    /// Scales packed vertex positions (1/1024 voxel units) to metres.
    pub voxel_size_m: f32,
    /// Unit direction towards the sun in frame axes.
    pub sun_dir: Vec3,
    pub ambient: f32,
    pub egui: Option<EguiFrame<'a>>,
}

/// Outcome of one [`Renderer::render`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameResult {
    pub status: FrameStatus,
    /// Time spent waiting for a swapchain image. With vsync this is where the frame waits for the
    /// display, so it is excluded from CPU work time.
    pub acquire_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameStatus {
    Presented,
    /// The surface was outdated, lost, occluded or timed out; nothing was drawn this frame.
    Skipped(&'static str),
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub adapter: String,
    pub backend: String,
    pub present_mode: String,
    pub surface_format: String,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// sRGB view format the 3D passes render through; egui draws through the surface's own format.
    scene_format: wgpu::TextureFormat,
    /// Set when the last image was suboptimal; the surface cannot be reconfigured while an image
    /// is held, so it happens before the next acquire.
    reconfigure: bool,
    depth_view: wgpu::TextureView,
    terrain: terrain::TerrainPass,
    egui: egui_wgpu::Renderer,
    capture: capture::Capture,
    info: GpuInfo,
}

impl Renderer {
    /// Creates the device and surface. Blocks on adapter and device requests once at startup.
    pub fn new(window: Arc<Window>, settings: RenderSettings) -> Result<Self, RenderError> {
        let size = window.inner_size();
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("vx device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            }))?;
        // Validation errors outside an error scope are logged rather than aborting, so a bad
        // hot reloaded shader can never take the process down.
        device.on_uncaptured_error(Arc::new(|error| {
            tracing::error!("wgpu: {error}");
        }));
        device.set_device_lost_callback(|reason, message| {
            tracing::error!("wgpu device lost ({reason:?}): {message}");
        });

        // Lighting is computed in linear space and written through an sRGB view. egui expects a
        // non sRGB target, so the surface itself is the linear format when the platform has it.
        let caps = surface.get_capabilities(&adapter);
        let scene_format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .ok_or(RenderError::UnsupportedSurface)?;
        let linear = scene_format.remove_srgb_suffix();
        let format = if caps.formats.contains(&linear) {
            linear
        } else {
            scene_format
        };
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(RenderError::UnsupportedSurface)?;
        config.format = format;
        config.view_formats = if format == scene_format {
            vec![]
        } else {
            vec![scene_format]
        };
        config.usage = usage;
        config.present_mode = if settings.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface.configure(&device, &config);

        let adapter_info = adapter.get_info();
        let info = GpuInfo {
            adapter: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            present_mode: format!("{:?}", config.present_mode),
            surface_format: format!("{format:?} (scene {scene_format:?})"),
        };
        tracing::info!(
            "GPU {} on {}, surface {} {}x{}, {}",
            info.adapter,
            info.backend,
            info.surface_format,
            config.width,
            config.height,
            info.present_mode
        );

        let depth_view = create_depth_view(&device, config.width, config.height);
        let terrain = terrain::TerrainPass::new(&device, scene_format);
        let egui = egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
        let capture = capture::Capture::new(usage.contains(wgpu::TextureUsages::COPY_SRC));

        Ok(Self {
            surface,
            device,
            queue,
            config,
            scene_format,
            reconfigure: false,
            depth_view,
            terrain,
            egui,
            capture,
            info,
        })
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    pub fn surface_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, width, height);
    }

    /// Replaces the terrain mesh. An empty mesh clears it. M0 path, removed by workstream E.
    pub fn set_mesh(&mut self, mesh: &MeshData) {
        self.terrain.set_mesh(&self.device, mesh);
    }

    /// Copies a packed mesh into the pools and returns its chunk slot. The slot is written into
    /// every vertex's fourth word before the copy (D14). Indices are mesh local. Fails without
    /// side effects when a pool or the slot table is full.
    pub fn upload_mesh(
        &mut self,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> Result<MeshHandle, PoolError> {
        let _ = (vertices, indices);
        todo!("workstream E")
    }

    /// Releases the mesh's pool ranges and slot. Safe to call in the same frame as a draw that
    /// listed the handle; the renderer drops it from that frame's chunk table.
    pub fn free_mesh(&mut self, handle: MeshHandle) {
        let _ = handle;
        todo!("workstream E")
    }

    /// Uploads the material texture array. Until called, a built in debug pattern is used.
    pub fn set_material_textures(&mut self, textures: &MaterialTextures) {
        let _ = textures;
        todo!("workstream E")
    }

    pub fn pool_stats(&self) -> PoolStats {
        todo!("workstream E")
    }

    /// Latest culling results, read back without blocking; `None` until the first readback.
    pub fn cull_stats(&self) -> Option<CullStats> {
        todo!("workstream E")
    }

    /// Compiles new terrain shader source. On any error the previous pipeline stays in use and the
    /// error is returned for logging (engineering rule 9).
    pub fn reload_terrain_shader(&mut self, source: &str) -> Result<(), String> {
        self.terrain.reload(&self.device, self.scene_format, source)
    }

    /// Copies the next presented frame, egui included, for [`Renderer::poll_capture`].
    pub fn request_capture(&mut self) -> Result<(), String> {
        self.capture.request()
    }

    /// Returns a finished capture once the GPU readback completes. Never waits.
    pub fn poll_capture(&mut self) -> Option<CapturedImage> {
        if !self.capture.is_pending() {
            return None;
        }
        // Non blocking: only advances completed submissions and fires map callbacks.
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.capture.try_take(self.config.format)
    }

    pub fn render(&mut self, mut frame: FrameInput) -> FrameResult {
        profiling::scope!("Renderer::render");
        // egui texture deltas arrive once; they must be applied and freed even when the surface
        // skips the frame, or later frames reference textures that were never uploaded.
        if let Some(ui) = &mut frame.egui {
            for (id, deltas) in ui.textures_delta.set.drain() {
                for delta in &deltas {
                    self.egui
                        .update_texture(&self.device, &self.queue, id, delta);
                }
            }
        }
        if std::mem::take(&mut self.reconfigure) {
            self.surface.configure(&self.device, &self.config);
        }
        let acquire_start = std::time::Instant::now();
        let acquired = {
            profiling::scope!("acquire surface");
            self.surface.get_current_texture()
        };
        let acquire_ms = acquire_start.elapsed().as_secs_f64() * 1.0e3;
        let status = self.draw(&frame, acquired);
        if let Some(ui) = &mut frame.egui {
            for id in ui.textures_delta.free.drain() {
                self.egui.free_texture(&id);
            }
        }
        FrameResult { status, acquire_ms }
    }

    fn draw(&mut self, frame: &FrameInput, acquired: wgpu::CurrentSurfaceTexture) -> FrameStatus {
        let surface_texture = match acquired {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.reconfigure = true;
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return FrameStatus::Skipped("surface outdated");
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return FrameStatus::Skipped("surface lost");
            }
            wgpu::CurrentSurfaceTexture::Timeout => return FrameStatus::Skipped("surface timeout"),
            wgpu::CurrentSurfaceTexture::Occluded => {
                return FrameStatus::Skipped("window occluded");
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return FrameStatus::Skipped("surface validation error");
            }
        };
        let scene_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.scene_format),
                ..Default::default()
            });
        let ui_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.terrain.write_uniform(
            &self.queue,
            &FrameUniform::new(
                frame.view_proj,
                frame.mesh_offset_m,
                frame.sun_dir,
                frame.ambient,
            ),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            profiling::scope!("terrain pass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scene_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.018,
                            g: 0.022,
                            b: 0.032,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.terrain.draw(&mut pass);
        }

        let mut extra_buffers = Vec::new();
        if let Some(ui) = &frame.egui {
            profiling::scope!("egui pass");
            let screen = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: ui.pixels_per_point,
            };
            extra_buffers = self.egui.update_buffers(
                &self.device,
                &self.queue,
                &mut encoder,
                ui.primitives,
                &screen,
            );
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &ui_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui.render(&mut pass, ui.primitives, &screen);
        }

        self.capture
            .encode_copy(&self.device, &mut encoder, &surface_texture.texture);

        {
            profiling::scope!("submit and present");
            self.queue
                .submit(extra_buffers.into_iter().chain([encoder.finish()]));
            self.capture.after_submit();
            self.queue.present(surface_texture);
        }

        FrameStatus::Presented
    }
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}
