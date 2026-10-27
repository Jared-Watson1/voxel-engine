use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use glam::{DVec3, Vec3};
use vx_core::{FileWatcher, Projection, WorldConfig, WorldPos, camera_relative};
use vx_render::{EguiFrame, FrameInput, FrameStatus, MeshData, RenderSettings, Renderer};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::args::Args;
use crate::camera::{FlyCamera, MoveInput};
use crate::capture::{self, CaptureOutcome};
use crate::overlay::{self, OverlayData};
use crate::scene::{self, ChunkResult, SceneConfig, ScenePaths};
use crate::stats::{FrameSamples, Summary};

const FOV_Y_RAD: f32 = std::f32::consts::FRAC_PI_3;
const NEAR_PLANE_M: f32 = 0.05;
const SUN_DIR: Vec3 = Vec3::new(-0.6, 0.7, 0.4);
const AMBIENT: f32 = 0.18;
/// Seconds of rendering after the mesh arrives before measurements or captures start, so shader
/// compilation and the first uploads are not counted.
const WARMUP_S: f64 = 2.0;
/// Pixel scroll deltas (trackpads) per scroll line.
const PIXELS_PER_SCROLL_LINE: f64 = 40.0;

pub struct App {
    args: Args,
    running: Option<Running>,
    exit_code: i32,
}

impl App {
    pub fn new(args: Args) -> Self {
        Self {
            args,
            running: None,
            exit_code: 0,
        }
    }

    pub fn exit_code(&self) -> i32 {
        self.exit_code
    }
}

struct LoadedChunk {
    world: WorldConfig,
    scene: SceneConfig,
    origin: WorldPos,
    triangles: usize,
    vertices: usize,
    generate_ms: f64,
    mesh_ms: f64,
    loaded_at: Instant,
}

/// Automation state for `--bench-seconds` and `--screenshot`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Automation {
    WaitingForMesh,
    WarmingUp,
    Measuring { since: Instant },
    Capturing,
    Writing,
    Done,
}

struct Running {
    window: Arc<Window>,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    camera: FlyCamera,
    camera_placed: bool,
    input: MoveInput,
    looking: bool,
    paths: ScenePaths,
    shader_path: PathBuf,
    watcher: Option<FileWatcher>,
    chunk_tx: Sender<Result<ChunkResult, String>>,
    chunk_rx: Receiver<Result<ChunkResult, String>>,
    latest_job: u64,
    chunk: Option<LoadedChunk>,
    chunk_error: Option<String>,
    shader_error: Option<String>,
    last_frame: Instant,
    samples: FrameSamples,
    bench_samples: FrameSamples,
    automation: Automation,
    capture_tx: Sender<CaptureOutcome>,
    capture_rx: Receiver<CaptureOutcome>,
    capture_path: Option<PathBuf>,
    last_capture: Option<String>,
    last_status: FrameStatus,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        match Running::new(event_loop, &self.args) {
            Ok(running) => self.running = Some(running),
            Err(e) => {
                tracing::error!("startup failed: {e:#}");
                self.exit_code = 1;
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(running) = &mut self.running else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => running.renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                if let Some(code) = running.frame(&self.args) {
                    self.exit_code = code;
                    event_loop.exit();
                }
            }
            // Measurement and capture runs ignore input so the camera stays where it was placed.
            _ if self.args.automated() => {}
            event => running.input_event(event_loop, event),
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let (Some(running), DeviceEvent::MouseMotion { delta }) = (&mut self.running, event)
            && running.looking
            && !self.args.automated()
        {
            running.camera.look(delta.0, delta.1);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(running) = &self.running {
            running.window.request_redraw();
        }
    }
}

impl Running {
    fn input_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let running = self;
        let response = running.egui_state.on_window_event(&running.window, &event);
        match event {
            WindowEvent::KeyboardInput { event, .. } if !response.consumed => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape if pressed => event_loop.exit(),
                        KeyCode::F12 if pressed && !event.repeat => running.request_capture(None),
                        _ => running.set_key(code, pressed),
                    }
                }
            }
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                running.looking = state == ElementState::Pressed && !response.consumed;
            }
            WindowEvent::MouseWheel { delta, .. } if !response.consumed => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(p) => p.y / PIXELS_PER_SCROLL_LINE,
                };
                running.camera.scroll(lines);
            }
            WindowEvent::Focused(false) => {
                running.input = MoveInput::default();
                running.looking = false;
            }
            _ => {}
        }
    }

    fn new(event_loop: &ActiveEventLoop, args: &Args) -> anyhow::Result<Self> {
        let attributes = Window::default_attributes()
            .with_title("Voxel Engine")
            .with_inner_size(PhysicalSize::new(args.width, args.height));
        let window = Arc::new(event_loop.create_window(attributes)?);
        let renderer = Renderer::new(
            window.clone(),
            RenderSettings {
                vsync: args.vsync,
                pools: Default::default(),
            },
        )?;

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let assets = args.assets_dir.clone();
        let paths = ScenePaths {
            world: assets.join("config/world.ron"),
            scene: assets.join("config/m0_scene.ron"),
        };
        let shader_path = assets.join("shaders/terrain.wgsl");
        let watcher = match FileWatcher::new(&[
            shader_path.clone(),
            paths.world.clone(),
            paths.scene.clone(),
        ]) {
            Ok(w) => Some(w),
            Err(e) => {
                tracing::warn!("hot reload disabled, file watcher failed: {e}");
                None
            }
        };

        let (chunk_tx, chunk_rx) = channel();
        scene::spawn_chunk_job(1, &paths, args.voxel_size_m, chunk_tx.clone());
        let (capture_tx, capture_rx) = channel();

        let (w, h) = renderer.surface_size();
        let camera = FlyCamera::looking_at(
            WorldPos::body_fixed(DVec3::new(0.0, 0.0, 1.0)),
            DVec3::ZERO,
            1.0,
            Projection {
                fov_y_rad: FOV_Y_RAD,
                aspect: w as f32 / h as f32,
                near_m: NEAR_PLANE_M,
            },
        );

        Ok(Self {
            window,
            renderer,
            egui_ctx,
            egui_state,
            camera,
            camera_placed: false,
            input: MoveInput::default(),
            looking: false,
            paths,
            shader_path,
            watcher,
            chunk_tx,
            chunk_rx,
            latest_job: 1,
            chunk: None,
            chunk_error: None,
            shader_error: None,
            last_frame: Instant::now(),
            samples: FrameSamples::with_capacity(240),
            bench_samples: FrameSamples::with_capacity(100_000),
            automation: Automation::WaitingForMesh,
            capture_tx,
            capture_rx,
            capture_path: None,
            last_capture: None,
            last_status: FrameStatus::Presented,
        })
    }

    fn set_key(&mut self, code: KeyCode, pressed: bool) {
        let input = &mut self.input;
        match code {
            KeyCode::KeyW => input.forward = pressed,
            KeyCode::KeyS => input.back = pressed,
            KeyCode::KeyA => input.left = pressed,
            KeyCode::KeyD => input.right = pressed,
            KeyCode::Space | KeyCode::KeyE => input.up = pressed,
            KeyCode::ShiftLeft | KeyCode::KeyQ => input.down = pressed,
            _ => {}
        }
    }

    fn request_capture(&mut self, path: Option<PathBuf>) {
        let path = path.unwrap_or_else(capture::default_path);
        match self.renderer.request_capture() {
            Ok(()) => self.capture_path = Some(path),
            Err(e) => tracing::error!("screenshot unavailable: {e}"),
        }
    }

    /// Runs one frame. Returns an exit code when automation has finished.
    fn frame(&mut self, args: &Args) -> Option<i32> {
        profiling::scope!("frame");
        let frame_start = Instant::now();
        let dt_s = (frame_start - self.last_frame).as_secs_f64();
        self.last_frame = frame_start;

        self.poll_files(args);
        self.poll_chunks();

        {
            profiling::scope!("update");
            self.camera.update(self.input, dt_s);
            let (w, h) = self.renderer.surface_size();
            self.camera.projection.aspect = w as f32 / h as f32;
        }

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let overlay_data = self.overlay_data();
        let full_output = {
            profiling::scope!("egui");
            self.egui_ctx
                .run_pass(raw_input, |ctx| overlay::draw(ctx, &overlay_data))
        };
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        let pixels_per_point = full_output.pixels_per_point;
        let primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, pixels_per_point);

        let mesh_offset_m = self.chunk.as_ref().map_or(Vec3::ZERO, |c| {
            camera_relative(c.origin, self.camera.position)
        });
        let rendered = {
            profiling::scope!("render");
            self.renderer.render(FrameInput {
                view_proj: self.camera.view_proj(),
                mesh_offset_m,
                chunks: &[],
                voxel_size_m: self
                    .chunk
                    .as_ref()
                    .map_or(1.0, |c| c.world.voxel_size_m as f32),
                sun_dir: SUN_DIR.normalize(),
                ambient: AMBIENT,
                egui: Some(EguiFrame {
                    primitives: &primitives,
                    textures_delta: full_output.textures_delta,
                    pixels_per_point,
                }),
            })
        };
        let status = rendered.status;
        let cpu_ms = frame_start.elapsed().as_secs_f64() * 1.0e3 - rendered.acquire_ms;
        if status != self.last_status {
            match status {
                FrameStatus::Presented => tracing::info!("presenting frames again"),
                FrameStatus::Skipped(reason) => tracing::info!("skipping frames: {reason}"),
            }
            self.last_status = status;
        }

        if status == FrameStatus::Presented {
            self.samples.push(dt_s * 1.0e3, cpu_ms);
            if matches!(self.automation, Automation::Measuring { .. }) {
                self.bench_samples.push(dt_s * 1.0e3, cpu_ms);
            }
        }

        if let Some(image) = self.renderer.poll_capture() {
            let path = self
                .capture_path
                .take()
                .unwrap_or_else(capture::default_path);
            capture::spawn_write(image, path, self.capture_tx.clone());
        }
        let exit = self.poll_captures_and_automation(args);
        profiling::finish_frame!();
        exit
    }

    fn poll_files(&mut self, args: &Args) {
        let Some(watcher) = &self.watcher else { return };
        let mut regenerate = false;
        for event in watcher.poll() {
            if event.path == self.shader_path {
                match event.contents {
                    Ok(source) => match self.renderer.reload_terrain_shader(&source) {
                        Ok(()) => {
                            tracing::info!("terrain shader reloaded");
                            self.shader_error = None;
                        }
                        Err(e) => {
                            tracing::error!(
                                "terrain shader rejected, keeping the previous pipeline: {e}"
                            );
                            self.shader_error = Some(e);
                        }
                    },
                    Err(e) => tracing::error!("reading {}: {e}", self.shader_path.display()),
                }
            } else {
                regenerate = true;
            }
        }
        if regenerate {
            self.latest_job += 1;
            tracing::info!("config changed, regenerating the chunk");
            scene::spawn_chunk_job(
                self.latest_job,
                &self.paths,
                args.voxel_size_m,
                self.chunk_tx.clone(),
            );
        }
    }

    fn poll_chunks(&mut self) {
        while let Ok(result) = self.chunk_rx.try_recv() {
            match result {
                Ok(result) if result.job == self.latest_job => self.load_chunk(result),
                Ok(_) => {}
                Err(e) => {
                    tracing::error!("chunk job failed: {e}");
                    self.chunk_error = Some(e);
                }
            }
        }
    }

    fn load_chunk(&mut self, result: ChunkResult) {
        let mesh = &result.mesh;
        self.renderer.set_mesh(&MeshData {
            positions: &mesh.positions,
            normals: &mesh.normals,
            indices: &mesh.indices,
        });
        tracing::info!(
            "chunk: voxel {} m, {} triangles, {} vertices, generate {:.2} ms, mesh {:.2} ms",
            result.world.voxel_size_m,
            mesh.triangle_count(),
            mesh.positions.len(),
            result.generate_ms,
            result.mesh_ms
        );
        if !result.sphere_fits {
            tracing::warn!(
                "the {} m sphere is larger than the {} m chunk at this voxel size; the mesh is clipped",
                result.scene.sphere_radius_m,
                result.world.chunk_extent_m()
            );
        }
        if !self.camera_placed {
            self.place_camera(&result.scene, result.centre);
        }
        self.chunk = Some(LoadedChunk {
            world: result.world,
            scene: result.scene,
            origin: mesh.origin,
            triangles: mesh.triangle_count(),
            vertices: mesh.positions.len(),
            generate_ms: result.generate_ms,
            mesh_ms: result.mesh_ms,
            loaded_at: Instant::now(),
        });
        self.chunk_error = None;
        if self.automation == Automation::WaitingForMesh {
            self.automation = Automation::WarmingUp;
        }
    }

    /// Frames the sphere: a three quarter view from above at a few radii, speed scaled to its size.
    fn place_camera(&mut self, scene: &SceneConfig, centre: WorldPos) {
        let r = scene.sphere_radius_m;
        let eye = WorldPos::new(centre.frame, centre.m + DVec3::new(1.4, 1.1, 2.2) * r);
        self.camera = FlyCamera::looking_at(eye, centre.m, r, self.camera.projection);
        self.camera_placed = true;
    }

    fn overlay_data(&self) -> OverlayData {
        let info = self.renderer.info();
        let (w, h) = self.renderer.surface_size();
        OverlayData {
            frame_ms: self.samples.mean_frame_ms(),
            cpu_ms: self.samples.mean_cpu_ms(),
            triangles: self.chunk.as_ref().map(|c| c.triangles),
            vertices: self.chunk.as_ref().map(|c| c.vertices),
            camera_m: self.camera.position.m,
            speed_m_s: self.camera.speed_m_s,
            voxel_size_m: self.chunk.as_ref().map(|c| c.world.voxel_size_m),
            chunk_extent_m: self.chunk.as_ref().map(|c| c.world.chunk_extent_m()),
            generate_ms: self.chunk.as_ref().map(|c| c.generate_ms),
            mesh_ms: self.chunk.as_ref().map(|c| c.mesh_ms),
            adapter: format!("{} ({})", info.adapter, info.backend),
            present_mode: info.present_mode.clone(),
            resolution: (w, h),
            shader_error: self.shader_error.clone(),
            chunk_error: self.chunk_error.clone(),
            last_capture: self.last_capture.clone(),
            profiler: crate::profiler::status(),
        }
    }

    fn poll_captures_and_automation(&mut self, args: &Args) -> Option<i32> {
        while let Ok(outcome) = self.capture_rx.try_recv() {
            match outcome {
                Ok(path) => {
                    tracing::info!("screenshot saved to {}", path.display());
                    self.last_capture = Some(path.display().to_string());
                }
                Err(e) => {
                    tracing::error!("screenshot failed: {e}");
                    self.last_capture = Some(format!("failed: {e}"));
                }
            }
            if self.automation == Automation::Writing {
                self.automation = Automation::Done;
            }
        }

        if !args.automated() {
            return None;
        }
        match self.automation {
            Automation::WaitingForMesh | Automation::Writing => None,
            Automation::WarmingUp => {
                let ready = self
                    .chunk
                    .as_ref()
                    .is_some_and(|c| c.loaded_at.elapsed().as_secs_f64() >= WARMUP_S);
                if ready {
                    self.automation = if args.bench_seconds.is_some() {
                        self.bench_samples.clear();
                        Automation::Measuring {
                            since: Instant::now(),
                        }
                    } else {
                        Automation::Capturing
                    };
                }
                None
            }
            Automation::Measuring { since } => {
                if since.elapsed().as_secs_f64() >= args.bench_seconds.unwrap_or_default() {
                    self.report_bench();
                    self.automation = Automation::Capturing;
                }
                None
            }
            Automation::Capturing => {
                if let Some(path) = &args.screenshot {
                    self.request_capture(Some(path.clone()));
                    self.automation = Automation::Writing;
                    None
                } else {
                    self.automation = Automation::Done;
                    Some(0)
                }
            }
            Automation::Done => Some(0),
        }
    }

    fn report_bench(&self) {
        let Some(summary) = Summary::from_samples(&self.bench_samples) else {
            tracing::error!("benchmark collected no frames");
            return;
        };
        let (w, h) = self.renderer.surface_size();
        let info = self.renderer.info();
        let chunk = self.chunk.as_ref();
        let line = format!(
            "BENCH voxel_size_m={} sphere_radius_m={} resolution={}x{} present_mode={} adapter=\"{}\" \
             frames={} seconds={:.1} frame_ms_mean={:.3} frame_ms_p50={:.3} frame_ms_p99={:.3} \
             frame_ms_max={:.3} fps={:.1} cpu_ms_mean={:.3} triangles={} vertices={} \
             generate_ms={:.3} mesh_ms={:.3}",
            chunk.map_or(f64::NAN, |c| c.world.voxel_size_m),
            chunk.map_or(f64::NAN, |c| c.scene.sphere_radius_m),
            w,
            h,
            info.present_mode,
            info.adapter,
            summary.frames,
            summary.seconds,
            summary.frame_ms_mean,
            summary.frame_ms_p50,
            summary.frame_ms_p99,
            summary.frame_ms_max,
            summary.fps,
            summary.cpu_ms_mean,
            chunk.map_or(0, |c| c.triangles),
            chunk.map_or(0, |c| c.vertices),
            chunk.map_or(f64::NAN, |c| c.generate_ms),
            chunk.map_or(f64::NAN, |c| c.mesh_ms),
        );
        println!("{line}");
    }
}
