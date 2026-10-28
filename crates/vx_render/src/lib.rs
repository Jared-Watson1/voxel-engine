//! Device, surface, passes and pipelines.
//!
//! The renderer receives packed meshes in chunk local coordinates and positions relative to a
//! floating render origin or to the camera only; no world position ever reaches it (D4, D23). It
//! never depends on the voxel, generation or physics crates.
//!
//! Terrain (D10, D19, D22, D23): meshes live in pooled vertex and index buffers
//! ([`Renderer::upload_mesh`]) with a persistent chunk table of placements relative to the render
//! origin; a compute pass culls every resident chunk and compacts the visible indices, including
//! the transition ranges of masked faces, and one indexed indirect draw renders them into a
//! G-buffer. Lighting and the sky (D25): the atmosphere LUTs, one fullscreen lighting pass into an
//! HDR target and a tonemap pass into the swapchain, with egui on top.
//!
//! Frame order: cull, terrain G-buffer, atmosphere LUTs, lighting, tonemap, egui. Each pass is
//! timed with timestamp queries at its boundaries when the adapter has them ([`GpuTimings`]).

mod atmosphere;
mod capture;
mod cull;
mod materials;
mod pool;
mod post;
mod terrain;
mod timing;

use std::sync::Arc;

use glam::{Mat4, Vec3};
use vx_core::{AtmosphereParams, Budgets, PackedVertex};
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

impl MaterialTextures {
    /// Checks the layer shapes and builds every mip level on the CPU (a box filter in linear
    /// light). Pure CPU work with no device, so it belongs on a worker thread (D9); the renderer
    /// only copies the result.
    pub fn prepare(&self) -> Result<PreparedMaterialTextures, String> {
        materials::prepare(self)
    }
}

/// Material layers with their mip chains, ready for [`Renderer::set_material_textures`].
#[derive(Debug, Clone)]
pub struct PreparedMaterialTextures {
    pub(crate) size: u32,
    /// Per layer, every level as (width, height, RGBA8 sRGB pixels).
    pub(crate) layers: Vec<Vec<(u32, u32, Vec<u8>)>>,
}

impl PreparedMaterialTextures {
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|levels| levels.iter().map(|(_, _, p)| p.len()))
            .sum()
    }
}

/// Repeat length of the material textures in metres. Triplanar coordinates are chunk local
/// positions plus [`ChunkPlacement::texture_phase_m`], so they are stable as the camera moves
/// without any world position reaching the shader (D4).
pub const MATERIAL_TILE_M: f64 = 4.0;

/// A packed chunk mesh to upload. `indices` holds the regular range followed by the six
/// transition ranges in face order (low x, high x, low y, high y, low z, high z), with these
/// counts (D22); the counts sum to `indices.len()`.
#[derive(Debug, Clone, Copy)]
pub struct MeshUpload<'a> {
    pub vertices: &'a [PackedVertex],
    pub indices: &'a [u32],
    pub regular_index_count: u32,
    pub transition_index_counts: [u32; 6],
}

/// Where and how a resident chunk draws (D23). Written into the chunk table at upload and again
/// only when it changes: on a rebase of the render origin, a transition mask change or a
/// visibility change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkPlacement {
    /// Chunk origin relative to the render origin in metres, computed in f64 by the caller.
    pub offset_m: Vec3,
    /// Chunk origin modulo [`MATERIAL_TILE_M`] along each axis (0 to the tile length), computed in
    /// f64 by the caller, so texture coordinates do not depend on the camera or the origin.
    pub texture_phase_m: Vec3,
    /// Sample spacing of the chunk's level; scales the packed positions.
    pub voxel_size_m: f32,
    /// Bit `face` set when the neighbour across that face is drawn one level finer: the vertex
    /// shader moves vertices to their secondary positions and the face's transition range draws.
    pub transition_mask: u8,
    /// Hidden chunks stay resident but are skipped by the cull pass.
    pub visible: bool,
}

/// The planet as the GPU needs it, relative to the camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetView {
    /// Planet centre relative to the camera in metres (body fixed axes).
    pub centre_m: Vec3,
    /// Mean radius in metres; the atmosphere's ground.
    pub radius_m: f32,
    /// Radius of a sphere the terrain never dips below, for the horizon test in the cull pass.
    pub occluder_radius_m: f32,
}

/// The star's light at the planet, above the atmosphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunLight {
    /// Unit direction towards the sun in body fixed axes.
    pub direction: Vec3,
    /// Illuminance above the atmosphere in the renderer's linear units (1.0 is the Sun at 1 AU),
    /// per colour channel.
    pub illuminance: Vec3,
    /// Angular radius of the disc.
    pub angular_radius_rad: f32,
}

/// A moon drawn by the sky pass as a lit sphere (stretch in M2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonLight {
    /// Moon centre relative to the camera in metres (body fixed axes).
    pub centre_m: Vec3,
    pub radius_m: f32,
    pub albedo: f32,
}

impl Default for PlanetView {
    fn default() -> Self {
        Self {
            centre_m: Vec3::new(0.0, -10_000.0, 0.0),
            radius_m: 10_000.0,
            occluder_radius_m: 9_000.0,
        }
    }
}

impl Default for SunLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.3, 0.8, 0.5).normalize(),
            illuminance: Vec3::ONE,
            angular_radius_rad: 0.004_65,
        }
    }
}

/// GPU time per pass from timestamp queries, read back a few frames late; `None` from
/// [`Renderer::gpu_timings`] when the adapter has no timestamp queries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuTimings {
    /// Pass name and milliseconds, in frame order.
    pub passes: Vec<(&'static str, f64)>,
    pub frames_old: u32,
}

impl GpuTimings {
    /// Names of the passes that count against [`Budgets::SKY_AND_LIGHTING_MS`]: the four LUT
    /// passes (the static two only on frames after a parameter change) and the lighting pass.
    pub const SKY_AND_LIGHTING_PASSES: [&'static str; 5] = [
        "transmittance lut",
        "multiple scattering lut",
        "sky view lut",
        "aerial perspective lut",
        "lighting",
    ];

    pub fn total_ms(&self) -> f64 {
        self.passes.iter().map(|(_, ms)| ms).sum()
    }

    /// Milliseconds of the named pass, `None` when it was not timed in this frame.
    pub fn pass_ms(&self, name: &str) -> Option<f64> {
        self.passes
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, ms)| *ms)
    }

    /// GPU time of the atmosphere LUTs and the lighting pass, the sky and lighting budget.
    pub fn sky_and_lighting_ms(&self) -> f64 {
        self.passes
            .iter()
            .filter(|(n, _)| Self::SKY_AND_LIGHTING_PASSES.contains(n))
            .map(|(_, ms)| ms)
            .sum()
    }
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
/// G-buffer albedo: linear albedo stored as sRGB, alpha 1 where terrain was drawn (D25).
pub const GBUFFER_ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// G-buffer normal: unit normal in body fixed axes in xyz, w reserved (0).
pub const GBUFFER_NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Lit scene radiance before tonemapping.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Every shader file the renderer compiles, for hot reload through [`Renderer::reload_shader`].
pub const SHADER_FILES: [&str; 5] = [
    "terrain.wgsl",
    "terrain_cull.wgsl",
    "atmosphere.wgsl",
    "lighting.wgsl",
    "tonemap.wgsl",
];
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
    #[error("terrain pools need {requested} bytes, over the {budget} byte GPU terrain budget")]
    PoolBudget { requested: usize, budget: usize },
    #[error("terrain pools need a {needed} byte storage binding, the adapter allows {supported}")]
    PoolBindingTooLarge { needed: u64, supported: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderSettings {
    /// Wait for vertical blank. Off gives uncapped frame times for measurements.
    pub vsync: bool,
    pub pools: TerrainPoolConfig,
}

/// egui output for one frame, already tessellated.
pub struct EguiFrame<'a> {
    pub primitives: &'a [egui::ClippedPrimitive],
    /// Consumed by [`Renderer::render`]; egui asserts in debug builds that every delta is applied.
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// Everything the renderer needs for one frame besides the persistent chunk table (D23).
pub struct FrameInput<'a> {
    /// Projection times the rotation only view matrix (body fixed axes to clip space).
    pub view_proj: Mat4,
    /// Camera position relative to the render origin in metres, within the rebase distance.
    pub camera_offset_m: Vec3,
    pub planet: PlanetView,
    pub sun: SunLight,
    pub moon: Option<MoonLight>,
    /// Exposure in stops applied before the tonemap curve.
    pub exposure_ev: f32,
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
    targets: FrameTargets,
    terrain: terrain::TerrainPass,
    atmosphere: atmosphere::AtmospherePass,
    post: post::PostPass,
    egui: egui_wgpu::Renderer,
    capture: capture::Capture,
    timer: timing::GpuTimer,
    info: GpuInfo,
}

impl Renderer {
    /// Creates the device and surface. Blocks on adapter and device requests once at startup.
    pub fn new(window: Arc<Window>, settings: RenderSettings) -> Result<Self, RenderError> {
        if settings.pools.gpu_bytes() > Budgets::GPU_TERRAIN_BYTES {
            return Err(RenderError::PoolBudget {
                requested: settings.pools.gpu_bytes(),
                budget: Budgets::GPU_TERRAIN_BYTES,
            });
        }
        let size = window.inner_size();
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))?;
        // Each pool is one storage binding; the WebGPU default binding limit (128 MiB) is below
        // the default vertex pool, so the limit is raised to the pool size when the adapter can.
        let pool_bytes = settings.pools.vertex_bytes.max(settings.pools.index_bytes) as u64;
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffer_binding_size = required_limits
            .max_storage_buffer_binding_size
            .max(pool_bytes);
        required_limits.max_buffer_size = required_limits.max_buffer_size.max(pool_bytes);
        if required_limits.max_storage_buffer_binding_size
            > adapter_limits.max_storage_buffer_binding_size
            || required_limits.max_buffer_size > adapter_limits.max_buffer_size
        {
            return Err(RenderError::PoolBindingTooLarge {
                needed: pool_bytes,
                supported: adapter_limits
                    .max_storage_buffer_binding_size
                    .min(adapter_limits.max_buffer_size),
            });
        }
        // Timestamps at pass boundaries are all Apple GPUs offer; requested when present so the
        // overlay can show GPU time per pass (engineering rule 4).
        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("vx device"),
                required_features,
                required_limits,
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

        let targets = FrameTargets::new(&device, config.width, config.height);
        let terrain = terrain::TerrainPass::new(&device, &queue, settings.pools);
        let atmosphere = atmosphere::AtmospherePass::new(&device, &queue);
        let post = post::PostPass::new(&device, scene_format);
        let egui = egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
        let capture = capture::Capture::new(usage.contains(wgpu::TextureUsages::COPY_SRC));
        let timer = timing::GpuTimer::new(&device, &queue);
        if !timer.supported() {
            tracing::info!("no timestamp queries on this adapter; GPU pass timings disabled");
        }

        Ok(Self {
            surface,
            device,
            queue,
            config,
            scene_format,
            reconfigure: false,
            targets,
            terrain,
            atmosphere,
            post,
            egui,
            capture,
            timer,
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
        self.targets = FrameTargets::new(&self.device, width, height);
    }

    /// Copies a packed mesh into the pools, writes its chunk table entry and returns its chunk
    /// slot. The slot is written into every vertex's fourth word before the copy (D14). Indices
    /// are mesh local. Fails without side effects when a pool or the slot table is full.
    pub fn upload_mesh(
        &mut self,
        mesh: &MeshUpload,
        placement: ChunkPlacement,
    ) -> Result<MeshHandle, PoolError> {
        self.terrain.upload_mesh(&self.queue, mesh, placement)
    }

    /// Replaces a resident chunk's placement; the table entry is rewritten before the next frame.
    pub fn set_placement(&mut self, handle: MeshHandle, placement: ChunkPlacement) {
        self.terrain.set_placement(handle, placement);
    }

    /// Releases the mesh's pool ranges and slot; it is not drawn from the next frame on. The slot
    /// may be handed to the next upload.
    pub fn free_mesh(&mut self, handle: MeshHandle) {
        self.terrain.free_mesh(handle);
    }

    /// Sets the planet's atmosphere (D25); the static LUTs are recomputed over the next two
    /// frames. The atmosphere's ground is [`PlanetView::radius_m`] of each frame; a different
    /// `planet_radius_m` here is logged once and the frame's radius wins.
    pub fn set_atmosphere(&mut self, atmosphere: &AtmosphereParams, planet_radius_m: f64) {
        self.atmosphere.set_params(atmosphere, planet_radius_m);
    }

    /// GPU time per pass, or `None` when the adapter has no timestamp queries or no readback has
    /// completed yet.
    pub fn gpu_timings(&self) -> Option<GpuTimings> {
        self.timer.timings()
    }

    /// Uploads a prepared material texture array (mip chains built by
    /// [`MaterialTextures::prepare`] off the main thread). Until called, a built in debug pattern
    /// is used (magenta for layer 0, then grey, brown, green, yellow). A set the device cannot
    /// hold is logged and the previous array stays bound.
    pub fn set_material_textures(&mut self, textures: &PreparedMaterialTextures) {
        self.terrain
            .set_material_textures(&self.device, &self.queue, textures);
        let stats = self.terrain.pool_stats();
        if stats.gpu_bytes() > Budgets::GPU_TERRAIN_BYTES {
            tracing::warn!(
                "GPU terrain memory {} bytes exceeds the {} byte budget after the material upload",
                stats.gpu_bytes(),
                Budgets::GPU_TERRAIN_BYTES
            );
        }
    }

    pub fn pool_stats(&self) -> PoolStats {
        self.terrain.pool_stats()
    }

    /// Latest culling results, read back without blocking; `None` until the first readback.
    /// Readbacks are polled at the start of [`Renderer::render`], so the values describe a frame
    /// one or two renders ago (`frames_old`).
    pub fn cull_stats(&self) -> Option<CullStats> {
        self.terrain.cull_stats()
    }

    /// Compiles new source for one of [`SHADER_FILES`] (a file name under `assets/shaders/`). On
    /// any error every previous pipeline built from that file stays in use and the error is
    /// returned for logging (engineering rule 9).
    pub fn reload_shader(&mut self, file: &str, source: &str) -> Result<(), String> {
        match file {
            "terrain.wgsl" => self.terrain.reload(&self.device, source),
            "terrain_cull.wgsl" => self.terrain.reload_cull(&self.device, source),
            "atmosphere.wgsl" => self.atmosphere.reload_luts(&self.device, source),
            "lighting.wgsl" => self.atmosphere.reload_lighting(&self.device, source),
            "tonemap.wgsl" => self.post.reload(&self.device, source),
            other => Err(format!("{other} is not a renderer shader")),
        }
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

        self.timer.begin_frame(&self.device);
        self.terrain.prepare_frame(&self.device, &self.queue, frame);
        self.atmosphere
            .prepare_frame(&self.queue, frame, self.config.width, self.config.height);
        self.post.prepare_frame(&self.queue, frame);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        // The cull pass opens its own untimed compute passes; it is timed from an empty marker
        // pass to the start of the terrain pass.
        let cull_start = self.timer.marker(&mut encoder);
        self.terrain.encode_cull(&mut encoder);
        self.timer.span_to_next("cull", cull_start);

        {
            profiling::scope!("terrain pass");
            let clear = wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.albedo,
                        depth_slice: None,
                        resolve_target: None,
                        ops: clear,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.normal,
                        depth_slice: None,
                        resolve_target: None,
                        ops: clear,
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: self.timer.render_writes("terrain"),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.terrain.draw(&mut pass);
        }
        self.terrain.encode_readback(&mut encoder);
        self.atmosphere.encode_luts(&mut encoder, &mut self.timer);
        self.atmosphere
            .encode_lighting(&self.device, &mut encoder, &self.targets, &mut self.timer);
        self.post.encode_tonemap(
            &self.device,
            &mut encoder,
            &self.targets,
            &scene_view,
            &mut self.timer,
        );

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
                    timestamp_writes: self.timer.render_writes("egui"),
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui.render(&mut pass, ui.primitives, &screen);
        }

        self.capture
            .encode_copy(&self.device, &mut encoder, &surface_texture.texture);
        self.timer.end_frame(&mut encoder);

        {
            profiling::scope!("submit and present");
            self.queue
                .submit(extra_buffers.into_iter().chain([encoder.finish()]));
            self.capture.after_submit();
            self.terrain.after_submit();
            self.timer.after_submit();
            self.queue.present(surface_texture);
        }

        FrameStatus::Presented
    }
}

/// The screen sized textures every frame renders through: the G-buffer, depth and the HDR target.
/// Recreated on resize.
pub(crate) struct FrameTargets {
    pub(crate) albedo: wgpu::TextureView,
    pub(crate) normal: wgpu::TextureView,
    pub(crate) depth: wgpu::TextureView,
    pub(crate) hdr: wgpu::TextureView,
    /// Unique per instance, so passes caching bind groups of these views rebuild them on resize.
    pub(crate) generation: u64,
}

impl FrameTargets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let view = |label, format, extra: wgpu::TextureUsages| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: usage | extra,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let none = wgpu::TextureUsages::empty();
        Self {
            albedo: view("gbuffer albedo", GBUFFER_ALBEDO_FORMAT, none),
            normal: view("gbuffer normal", GBUFFER_NORMAL_FORMAT, none),
            depth: view("depth", DEPTH_FORMAT, none),
            // Copyable for readbacks of the lit frame in tests and debugging.
            hdr: view("hdr", HDR_FORMAT, wgpu::TextureUsages::COPY_SRC),
            generation: atmosphere::next_targets_generation(),
        }
    }
}
