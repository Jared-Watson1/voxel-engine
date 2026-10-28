use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use glam::{DVec3, Vec3};
use vx_core::{
    BudgetMonitor, BudgetUse, Budgets, FileWatcher, JobSystem, Projection, WorldConfig, WorldPos,
};
use vx_render::{
    EguiFrame, FrameInput, FrameStatus, MaterialTextures, PlanetView, PreparedMaterialTextures,
    RenderSettings, Renderer, SHADER_FILES,
};
use vx_universe::SystemConfig;
use vx_voxel::{Brush, BrushOp, BrushShape, Material};
use vx_worldgen::material_textures;
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
use crate::flight::{Descent, EYE_HEIGHT_M, Flight, SurfaceLoop, tangent_axes};
use crate::overlay::{self, OverlayData};
use crate::sky::{Sky, clock_text};
use crate::stats::{FrameSamples, Summary};
use crate::world::{STAGE_GENERATE, STAGE_MESH, STAGE_NAMES, World};

const FOV_Y_RAD: f32 = std::f32::consts::FRAC_PI_3;
const NEAR_PLANE_M: f32 = 0.05;
/// Seconds of rendering after the world settles before measurements or captures start.
const WARMUP_S: f64 = 2.0;
/// Longest wait for the initial streaming to settle before an automated run proceeds anyway.
const SETTLE_TIMEOUT_S: f64 = 60.0;
/// Pixel scroll deltas (trackpads) per scroll line.
const PIXELS_PER_SCROLL_LINE: f64 = 40.0;
/// Edge length of the generated material textures in texels.
const MATERIAL_TEXTURE_SIZE: u32 = 512;
/// Brush radius in voxels at startup; `[` and `]` change it.
const BRUSH_RADIUS_VOXELS: f64 = 3.0;
const BRUSH_RADIUS_MIN_VOXELS: f64 = 1.0;
const BRUSH_RADIUS_MAX_VOXELS: f64 = 16.0;
/// Reach of the brush pick ray in chunk extents.
const BRUSH_REACH_CHUNKS: f64 = 8.0;
/// Frames between scripted brushes in `--edit-test`.
const EDIT_TEST_INTERVAL_FRAMES: u64 = 30;
/// Start position: latitude in degrees, local solar time when none is given (a low morning sun)
/// and the eye height above the terrain.
const START_LATITUDE_DEG: f64 = 25.0;
const START_TIME_OF_DAY: f64 = 0.3;
/// Exposure change per key press in stops.
const EXPOSURE_STEP_EV: f32 = 0.5;

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

/// Everything loaded off the main thread at startup (D9).
struct Startup {
    config: WorldConfig,
    system: SystemConfig,
    textures: PreparedMaterialTextures,
    load_ms: f64,
}

/// Automation state for `--bench-seconds`, `--flight-km` and `--screenshot`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Automation {
    WaitingForWorld,
    WarmingUp { since: Instant },
    Measuring { since: Instant },
    Capturing,
    Writing,
    Done,
}

#[derive(Debug, Clone, Copy)]
struct BrushSettings {
    op: BrushOp,
    radius_voxels: f64,
}

/// Main thread time of each part of the last frame, in milliseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameSections {
    pub polls_ms: f64,
    pub update_ms: f64,
    pub egui_ms: f64,
    pub draw_list_ms: f64,
    pub render_ms: f64,
}

#[derive(Debug, Default, Clone, Copy)]
struct BenchPeaks {
    cpu_ms_max: f64,
    triangles_max: u32,
    visible_max: u32,
    edits: u64,
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
    world_path: PathBuf,
    system_path: PathBuf,
    shader_dir: PathBuf,
    watcher: Option<FileWatcher>,
    jobs: JobSystem,
    startup_rx: Receiver<Result<Startup, String>>,
    world: Option<World>,
    sky: Option<Sky>,
    exposure_ev: f32,
    start_time_of_day: Option<f64>,
    world_error: Option<String>,
    shader_error: Option<String>,
    brush: BrushSettings,
    flight: Option<Flight>,
    budget_monitor: BudgetMonitor,
    last_frame: Instant,
    started: Instant,
    frame_index: u64,
    samples: FrameSamples,
    bench_samples: FrameSamples,
    peaks: BenchPeaks,
    sections: FrameSections,
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
        let response = self.egui_state.on_window_event(&self.window, &event);
        match event {
            WindowEvent::KeyboardInput { event, .. } if !response.consumed => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape if pressed => event_loop.exit(),
                        KeyCode::F12 if pressed && !event.repeat => self.request_capture(None),
                        KeyCode::KeyB if pressed && !event.repeat => {
                            self.brush.op = match self.brush.op {
                                BrushOp::Dig => BrushOp::Place(Material::ROCK),
                                BrushOp::Place(_) => BrushOp::Dig,
                            };
                        }
                        KeyCode::BracketLeft if pressed => {
                            self.brush.radius_voxels =
                                (self.brush.radius_voxels - 1.0).max(BRUSH_RADIUS_MIN_VOXELS);
                        }
                        KeyCode::BracketRight if pressed => {
                            self.brush.radius_voxels =
                                (self.brush.radius_voxels + 1.0).min(BRUSH_RADIUS_MAX_VOXELS);
                        }
                        KeyCode::Comma if pressed && !event.repeat => {
                            if let Some(sky) = &mut self.sky {
                                sky.step_warp(-1);
                            }
                        }
                        KeyCode::Period if pressed && !event.repeat => {
                            if let Some(sky) = &mut self.sky {
                                sky.step_warp(1);
                            }
                        }
                        KeyCode::Minus if pressed => self.exposure_ev -= EXPOSURE_STEP_EV,
                        KeyCode::Equal if pressed => self.exposure_ev += EXPOSURE_STEP_EV,
                        _ => self.set_key(code, pressed),
                    }
                }
            }
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                self.looking = state == ElementState::Pressed && !response.consumed;
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: ElementState::Pressed,
                ..
            } if !response.consumed => {
                let op = self.brush.op;
                self.apply_brush_along_view(op, 0.0);
            }
            WindowEvent::MouseWheel { delta, .. } if !response.consumed => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(p) => p.y / PIXELS_PER_SCROLL_LINE,
                };
                self.camera.scroll(lines);
            }
            WindowEvent::Focused(false) => {
                self.input = MoveInput::default();
                self.looking = false;
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
        let world_path = assets.join("config/world.ron");
        let system_path = assets.join("config/system.ron");
        let shader_dir = assets.join("shaders");
        let mut watched: Vec<PathBuf> = SHADER_FILES.iter().map(|f| shader_dir.join(f)).collect();
        watched.push(world_path.clone());
        watched.push(system_path.clone());
        let watcher = match FileWatcher::new(&watched) {
            Ok(w) => Some(w),
            Err(e) => {
                tracing::warn!("hot reload disabled, file watcher failed: {e}");
                None
            }
        };

        let workers = args
            .workers
            .unwrap_or_else(JobSystem::default_worker_count)
            .max(1);
        let jobs = JobSystem::new(workers, &STAGE_NAMES);
        tracing::info!("job system with {workers} workers");

        let (startup_tx, startup_rx) = channel();
        spawn_startup(args, &world_path, &system_path, startup_tx);
        let (capture_tx, capture_rx) = channel();

        let (w, h) = renderer.surface_size();
        let camera = FlyCamera::looking_at(
            WorldPos::body_fixed(DVec3::new(0.0, 20_000.0, 0.0)),
            DVec3::new(100.0, 20_000.0, 0.0),
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
            world_path,
            system_path,
            shader_dir,
            watcher,
            jobs,
            startup_rx,
            world: None,
            sky: None,
            exposure_ev: args.exposure_ev,
            start_time_of_day: args.time_of_day,
            world_error: None,
            shader_error: None,
            brush: BrushSettings {
                op: BrushOp::Dig,
                radius_voxels: args
                    .brush_radius_voxels
                    .unwrap_or(BRUSH_RADIUS_VOXELS)
                    .clamp(BRUSH_RADIUS_MIN_VOXELS, BRUSH_RADIUS_MAX_VOXELS),
            },
            flight: None,
            budget_monitor: BudgetMonitor::new(),
            last_frame: Instant::now(),
            started: Instant::now(),
            frame_index: 0,
            samples: FrameSamples::with_capacity(240),
            bench_samples: FrameSamples::with_capacity(100_000),
            peaks: BenchPeaks::default(),
            sections: FrameSections::default(),
            automation: Automation::WaitingForWorld,
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

    /// Picks the surface along the view direction and applies the brush there. A dig is centred
    /// on the hit; a place is pulled back by half a radius so the blob sits on the surface.
    /// `side_offset_radii` shifts the centre along the camera's right axis in brush radii.
    fn apply_brush_along_view(&mut self, op: BrushOp, side_offset_radii: f64) -> usize {
        let Some(world) = &mut self.world else {
            return 0;
        };
        let config = *world.config();
        let direction = self.camera.orientation() * DVec3::NEG_Z;
        let right = self.camera.orientation() * DVec3::X;
        let reach = BRUSH_REACH_CHUNKS * config.chunk_extent_m();
        let Some(hit) = world.raycast(self.camera.position, direction, reach) else {
            return 0;
        };
        let radius_m = self.brush.radius_voxels * config.voxel_size_m;
        let centre = match op {
            BrushOp::Dig => hit,
            BrushOp::Place(_) => hit - direction * (0.5 * radius_m),
        } + right * (side_offset_radii * radius_m);
        let brush = Brush {
            shape: BrushShape::Sphere {
                centre: WorldPos::body_fixed(centre),
                radius_m,
            },
            op,
        };
        world.apply_brush(&brush)
    }

    /// Runs one frame. Returns an exit code when automation has finished.
    fn frame(&mut self, args: &Args) -> Option<i32> {
        profiling::scope!("frame");
        let frame_start = Instant::now();
        let dt_s = (frame_start - self.last_frame).as_secs_f64();
        self.last_frame = frame_start;
        self.frame_index += 1;

        let ms_since = |t: Instant| t.elapsed().as_secs_f64() * 1.0e3;
        self.poll_files(args);
        self.poll_startup(args);
        let mut sections = FrameSections {
            polls_ms: ms_since(frame_start),
            ..Default::default()
        };

        let update_start = Instant::now();
        {
            profiling::scope!("update");
            let (w, h) = self.renderer.surface_size();
            self.camera.projection.aspect = w as f32 / h as f32;
            if let Some(sky) = &mut self.sky {
                sky.advance(dt_s);
            }
            self.update_camera(dt_s);
            if let Some(world) = &mut self.world {
                world.update(self.camera.position, &mut self.renderer);
            }
            self.scripted_edit(args);
        }
        sections.update_ms = ms_since(update_start);

        let egui_start = Instant::now();
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let overlay_data = self.overlay_data(args);
        let show_overlay = !args.no_overlay;
        let full_output = {
            profiling::scope!("egui");
            self.egui_ctx.run_pass(raw_input, |ctx| {
                if show_overlay {
                    overlay::draw(ctx, &overlay_data);
                }
            })
        };
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        let pixels_per_point = full_output.pixels_per_point;
        let primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, pixels_per_point);
        sections.egui_ms = ms_since(egui_start);

        let draw_list_start = Instant::now();
        let camera_m = self.camera.position.m;
        let (camera_offset_m, planet) = match &self.world {
            Some(world) => (
                (camera_m - world.origin()).as_vec3(),
                PlanetView {
                    centre_m: (-camera_m).as_vec3(),
                    radius_m: world.planet().radius_m() as f32,
                    occluder_radius_m: world.sampler().shell_m().0 as f32,
                },
            ),
            None => (Vec3::ZERO, PlanetView::default()),
        };
        let sun = self
            .sky
            .as_ref()
            .map_or_else(vx_render::SunLight::default, Sky::sun_light);
        let moon = self.sky.as_ref().and_then(|sky| sky.moon_light(camera_m));
        sections.draw_list_ms = ms_since(draw_list_start);
        let render_start = Instant::now();
        let rendered = {
            profiling::scope!("render");
            self.renderer.render(FrameInput {
                view_proj: self.camera.view_proj(),
                camera_offset_m,
                planet,
                sun,
                moon,
                exposure_ev: self.exposure_ev,
                egui: Some(EguiFrame {
                    primitives: &primitives,
                    textures_delta: full_output.textures_delta,
                    pixels_per_point,
                }),
            })
        };
        sections.render_ms = ms_since(render_start) - rendered.acquire_ms;
        self.sections = sections;
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
                self.peaks.cpu_ms_max = self.peaks.cpu_ms_max.max(cpu_ms);
                if let Some(c) = self.renderer.cull_stats() {
                    self.peaks.triangles_max = self.peaks.triangles_max.max(c.triangles);
                    self.peaks.visible_max = self.peaks.visible_max.max(c.visible_chunks);
                }
            }
        }
        let uses = self.budget_uses(cpu_ms);
        let _ = self.budget_monitor.check(&uses);

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

    fn update_camera(&mut self, dt_s: f64) {
        let measuring = matches!(self.automation, Automation::Measuring { .. });
        let Some(world) = &self.world else {
            return;
        };
        if let Some(flight) = &mut self.flight {
            // The flight holds its start pose until measuring begins, so streaming settles there.
            let sampler = world.sampler();
            let (position, target) = flight.advance(if measuring { dt_s } else { 0.0 }, |dir| {
                sampler.height_m(dir)
            });
            let speed_scale = self.camera.speed_scale;
            self.camera = FlyCamera::looking_at(position, target, self.camera.projection);
            self.camera.speed_scale = speed_scale;
        } else {
            let altitude = world.altitude_m(self.camera.position.m);
            self.camera.update(self.input, dt_s, altitude);
        }
    }

    fn scripted_edit(&mut self, args: &Args) {
        if !args.edit_test
            || !matches!(self.automation, Automation::Measuring { .. })
            || !self.frame_index.is_multiple_of(EDIT_TEST_INTERVAL_FRAMES)
        {
            return;
        }
        // Three digs deepen a crater, then one place builds a rock pile two radii to the right.
        let (op, side) = if (self.frame_index / EDIT_TEST_INTERVAL_FRAMES).is_multiple_of(4) {
            (BrushOp::Place(Material::ROCK), 2.0)
        } else {
            (BrushOp::Dig, 0.0)
        };
        if self.apply_brush_along_view(op, side) > 0 {
            self.peaks.edits += 1;
        }
    }

    fn budget_uses(&self, cpu_ms: f64) -> Vec<BudgetUse> {
        let mut uses = vec![
            BudgetUse::ms("frame", self.samples.mean_frame_ms(), Budgets::FRAME_MS),
            BudgetUse::ms("main thread", cpu_ms, Budgets::MAIN_THREAD_MS),
        ];
        if let Some(world) = &self.world {
            let s = world.stats();
            let generate = self.jobs.stats(STAGE_GENERATE).mean_ms;
            let mesh = self.jobs.stats(STAGE_MESH).mean_ms;
            uses.push(BudgetUse::ms(
                "generate + mesh",
                generate + mesh,
                Budgets::CHUNK_GENERATE_MESH_MS,
            ));
            uses.push(BudgetUse::bytes(
                "voxel memory",
                s.voxel_bytes,
                Budgets::RESIDENT_VOXEL_BYTES,
            ));
            uses.push(BudgetUse::bytes(
                "upload per frame",
                s.upload_bytes_frame,
                Budgets::UPLOAD_BYTES_PER_FRAME,
            ));
        }
        let pool = self.renderer.pool_stats();
        uses.push(BudgetUse::bytes(
            "gpu terrain",
            pool.gpu_bytes(),
            Budgets::GPU_TERRAIN_BYTES,
        ));
        if let Some(c) = self.renderer.cull_stats() {
            uses.push(BudgetUse::count(
                "triangles on screen",
                u64::from(c.triangles),
                Budgets::TERRAIN_TRIANGLES_ON_SCREEN,
            ));
        }
        uses
    }

    fn poll_files(&mut self, args: &Args) {
        let Some(watcher) = &self.watcher else { return };
        for event in watcher.poll() {
            let contents = match event.contents {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("reading {}: {e}", event.path.display());
                    continue;
                }
            };
            if let Some(file) = SHADER_FILES
                .iter()
                .find(|f| event.path == self.shader_dir.join(f))
            {
                let result = self.renderer.reload_shader(file, &contents);
                self.reload_shader(file, result);
            } else if event.path == self.world_path {
                let system = self.sky.as_ref().map(|s| *s.config());
                match (parse_config(&contents, args), system) {
                    (Ok(config), Some(system)) => self.replace_world(config, system),
                    (Err(e), _) => {
                        tracing::error!("world config rejected, keeping the current world: {e}");
                        self.world_error = Some(e);
                    }
                    (Ok(_), None) => {}
                }
            } else if event.path == self.system_path {
                let config = self.world.as_ref().map(|w| *w.config());
                match (parse_system(&contents, args), config) {
                    (Ok(system), Some(config)) => self.replace_world(config, system),
                    (Err(e), _) => {
                        tracing::error!("system config rejected, keeping the current world: {e}");
                        self.world_error = Some(e);
                    }
                    (Ok(_), None) => {}
                }
            }
        }
    }

    fn reload_shader(&mut self, name: &str, result: Result<(), String>) {
        match result {
            Ok(()) => {
                tracing::info!("{name} shader reloaded");
                self.shader_error = None;
            }
            Err(e) => {
                tracing::error!("{name} shader rejected, keeping the previous pipeline: {e}");
                self.shader_error = Some(e);
            }
        }
    }

    fn poll_startup(&mut self, args: &Args) {
        while let Ok(result) = self.startup_rx.try_recv() {
            match result {
                Ok(startup) => {
                    tracing::info!(
                        "config and {} material layers ({:.1} MB with mips) loaded in {:.1} ms",
                        startup.textures.layer_count(),
                        startup.textures.bytes() as f64 / 1.0e6,
                        startup.load_ms
                    );
                    self.renderer.set_material_textures(&startup.textures);
                    self.replace_world(startup.config, startup.system);
                    if let Some(warp) = args.warp
                        && let Some(sky) = &mut self.sky
                    {
                        sky.set_warp(warp);
                    }
                    let radius_m = startup.system.planet.params.radius_m();
                    let up = self.camera.up();
                    let flight = if args.descent {
                        Some(Flight::Descent(Descent::new(
                            radius_m,
                            up,
                            self.camera.forward(),
                            args.descent_top_km * 1000.0,
                            args.descent_climb_s,
                        )))
                    } else {
                        args.flight_km.map(|km| {
                            Flight::Loop(SurfaceLoop::new(
                                radius_m,
                                up,
                                km * 1000.0,
                                args.flight_speed_m_s,
                                startup.config.chunk_extent_m(),
                            ))
                        })
                    };
                    if let Some(flight) = &flight {
                        tracing::info!("scripted flight of {:.0} s", flight.duration_s());
                    }
                    self.flight = flight;
                }
                Err(e) => {
                    tracing::error!("startup failed: {e}");
                    self.world_error = Some(e);
                }
            }
        }
    }

    fn replace_world(&mut self, config: WorldConfig, system: SystemConfig) {
        let planet = system.planet.params;
        if let Some(mut old) = self.world.take() {
            if *old.config() == config && *old.planet() == planet {
                self.world = Some(old);
                return;
            }
            old.release(&mut self.renderer);
            tracing::info!("world or planet config changed, rebuilding the world");
        }
        self.renderer
            .set_atmosphere(&planet.atmosphere, planet.radius_m());
        let sky_changed = self.sky.as_ref().is_none_or(|s| *s.config() != system);
        if sky_changed {
            self.sky = Some(Sky::new(system));
        }
        let mut world = World::new(config, planet, self.jobs.clone(), self.camera.position.m);
        if !self.camera_placed {
            self.place_camera(&mut world);
        }
        self.world = Some(world);
        self.world_error = None;
    }

    /// Stands on the surface at the start latitude, at the requested local time, facing the sun
    /// along the horizon (in the morning, the sunrise).
    fn place_camera(&mut self, world: &mut World) {
        let lat = START_LATITUDE_DEG.to_radians();
        let mut dir = DVec3::new(lat.cos(), lat.sin(), 0.0);
        if let Some(sky) = &mut self.sky {
            let time_of_day = self.start_time_of_day.unwrap_or(START_TIME_OF_DAY);
            sky.set_local_time(dir, time_of_day);
            dir = dir.normalize();
        }
        let radius = world.planet().radius_m();
        let eye = dir * (radius + world.sampler().height_m(dir) + EYE_HEIGHT_M);
        let sun = self
            .sky
            .as_ref()
            .map_or(tangent_axes(dir).0, |s| s.snapshot().sun_direction);
        let heading = sun - dir * sun.dot(dir);
        let heading = heading.normalize_or(tangent_axes(dir).0);
        self.camera = FlyCamera::looking_at(
            WorldPos::body_fixed(eye),
            eye + heading * 100.0 - dir * 3.0,
            self.camera.projection,
        );
        world.set_origin(eye);
        self.camera_placed = true;
    }

    fn overlay_data(&self, args: &Args) -> OverlayData {
        let info = self.renderer.info();
        let (w, h) = self.renderer.surface_size();
        let config = self.world.as_ref().map(|w| *w.config());
        let brush = match self.brush.op {
            BrushOp::Dig => format!("dig, {:.0} voxels", self.brush.radius_voxels),
            BrushOp::Place(m) => {
                format!("place {}, {:.0} voxels", m.name(), self.brush.radius_voxels)
            }
        };
        let cpu_ms_max = if args.automated() {
            self.peaks.cpu_ms_max
        } else {
            self.samples.max_cpu_ms()
        };
        OverlayData {
            frame_ms: self.samples.mean_frame_ms(),
            cpu_ms: self.samples.mean_cpu_ms(),
            cpu_ms_max,
            sections: self.sections,
            camera_m: self.camera.position.m,
            speed_m_s: self.camera.speed_m_s,
            altitude_m: self
                .world
                .as_ref()
                .map_or(0.0, |w| w.altitude_m(self.camera.position.m)),
            voxel_size_m: config.map(|c| c.voxel_size_m),
            lod: config.map(|c| (c.lod_levels, c.lod_radius_chunks)),
            planet_radius_km: self.world.as_ref().map(|w| w.planet().radius_km),
            seed: config.map(|c| c.seed),
            time: self.sky.as_ref().map_or_else(String::new, |sky| {
                format!(
                    "{} local, warp {}x, t = {:.0} s",
                    clock_text(sky.local_time(self.camera.position.m)),
                    sky.warp(),
                    sky.time_s()
                )
            }),
            exposure_ev: self.exposure_ev,
            gpu_timings: self.renderer.gpu_timings(),
            workers: self.jobs.worker_count(),
            world: self.world.as_ref().map(|w| w.stats().clone()),
            pool: Some(self.renderer.pool_stats()),
            cull: self.renderer.cull_stats(),
            budgets: self.budget_uses(self.samples.mean_cpu_ms()),
            brush,
            flight_progress: self.flight.as_ref().map(Flight::progress),
            adapter: format!("{} ({})", info.adapter, info.backend),
            present_mode: info.present_mode.clone(),
            resolution: (w, h),
            shader_error: self.shader_error.clone(),
            world_error: self.world_error.clone(),
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
            Automation::WaitingForWorld => {
                let settled = self.world.as_ref().is_some_and(World::is_settled);
                let timed_out = self.started.elapsed().as_secs_f64() > SETTLE_TIMEOUT_S;
                if settled || timed_out {
                    if timed_out {
                        tracing::warn!("initial streaming did not settle; measuring anyway");
                    }
                    tracing::info!(
                        "world settled after {:.1} s; warming up",
                        self.started.elapsed().as_secs_f64()
                    );
                    self.automation = Automation::WarmingUp {
                        since: Instant::now(),
                    };
                }
                None
            }
            Automation::WarmingUp { since } => {
                if since.elapsed().as_secs_f64() >= WARMUP_S {
                    self.automation = if args.bench_seconds.is_some() || self.flight.is_some() {
                        self.bench_samples.clear();
                        self.peaks = BenchPeaks::default();
                        if let Some(world) = &mut self.world {
                            world.reset_peaks();
                        }
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
                let elapsed = since.elapsed().as_secs_f64();
                let time_up = args.bench_seconds.is_some_and(|s| elapsed >= s);
                let flight_done = self.flight.as_ref().is_some_and(Flight::done);
                if time_up || flight_done {
                    self.report_bench(args);
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
            Automation::Writing => None,
            Automation::Done => Some(0),
        }
    }

    fn report_bench(&self, args: &Args) {
        let Some(summary) = Summary::from_samples(&self.bench_samples) else {
            tracing::error!("benchmark collected no frames");
            return;
        };
        let (w, h) = self.renderer.surface_size();
        let info = self.renderer.info();
        let config = self.world.as_ref().map(|w| *w.config());
        let stats = self.world.as_ref().map(|w| w.stats().clone());
        let pool = self.renderer.pool_stats();
        let cull = self.renderer.cull_stats().unwrap_or_default();
        let generate = self.jobs.stats(STAGE_GENERATE);
        let mesh = self.jobs.stats(STAGE_MESH);
        let planet = self.world.as_ref().map(|w| *w.planet());
        let drawn = stats.as_ref().map_or_else(String::new, |s| {
            s.lod
                .drawn_per_level
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("/")
        });
        let gpu = self.renderer.gpu_timings();
        let line = format!(
            "BENCH voxel_size_m={} radius_km={} lod_levels={} lod_radius_chunks={} seed={} \
             resolution={}x{} present_mode={} adapter=\"{}\" workers={} descent={} \
             descent_top_km={} flight_km={} flight_speed_m_s={} edit_test={} edits={} frames={} \
             seconds={:.1} frame_ms_mean={:.3} frame_ms_p50={:.3} frame_ms_p99={:.3} \
             frame_ms_max={:.3} fps={:.1} cpu_ms_mean={:.3} cpu_ms_max={:.3} \
             triangles_visible={} triangles_visible_max={} chunks_visible={} chunks_visible_max={} \
             nodes={} meshes_resident={} drawn_per_level={} voxel_mb={:.1} voxel_mb_peak={:.1} \
             gpu_vertex_mb={:.1} gpu_index_mb={:.1} gpu_pool_mb={:.1} upload_mb_frame_peak={:.2} \
             queue_peak={} generate_ms_mean={:.3} generate_ms_max={:.3} mesh_ms_mean={:.3} \
             mesh_ms_max={:.3} generated={} meshed={} uploaded={} upload_failures={} splits={} \
             merges={} lod_ms_max={:.3} rebases={} rebase_ms_max={:.3} world_update_ms_max={:.3} \
             gpu_ms={:.3}",
            config.map_or(f64::NAN, |c| c.voxel_size_m),
            planet.map_or(f64::NAN, |p| p.radius_km),
            config.map_or(0, |c| c.lod_levels),
            config.map_or(f64::NAN, |c| c.lod_radius_chunks),
            config.map_or(0, |c| c.seed),
            w,
            h,
            info.present_mode,
            info.adapter,
            self.jobs.worker_count(),
            args.descent,
            args.descent_top_km,
            args.flight_km.unwrap_or(0.0),
            args.flight_speed_m_s,
            args.edit_test,
            self.peaks.edits,
            summary.frames,
            summary.seconds,
            summary.frame_ms_mean,
            summary.frame_ms_p50,
            summary.frame_ms_p99,
            summary.frame_ms_max,
            summary.fps,
            summary.cpu_ms_mean,
            self.peaks.cpu_ms_max,
            cull.triangles,
            self.peaks.triangles_max,
            cull.visible_chunks,
            self.peaks.visible_max,
            stats.as_ref().map_or(0, |s| s.nodes),
            stats.as_ref().map_or(0, |s| s.meshed),
            drawn,
            stats.as_ref().map_or(0.0, |s| s.voxel_bytes as f64 / 1.0e6),
            stats
                .as_ref()
                .map_or(0.0, |s| s.voxel_bytes_peak as f64 / 1.0e6),
            pool.vertex_bytes_used as f64 / 1.0e6,
            pool.index_bytes_used as f64 / 1.0e6,
            pool.gpu_bytes() as f64 / 1.0e6,
            stats
                .as_ref()
                .map_or(0.0, |s| s.upload_bytes_peak as f64 / 1.0e6),
            stats.as_ref().map_or(0, |s| s.queued_peak),
            generate.mean_ms,
            generate.max_ms,
            mesh.mean_ms,
            mesh.max_ms,
            stats.as_ref().map_or(0, |s| s.generated_total),
            stats.as_ref().map_or(0, |s| s.meshed_total),
            stats.as_ref().map_or(0, |s| s.uploaded_total),
            stats.as_ref().map_or(0, |s| s.upload_failures),
            stats.as_ref().map_or(0, |s| s.lod.splits_total),
            stats.as_ref().map_or(0, |s| s.lod.merges_total),
            stats.as_ref().map_or(0.0, |s| s.lod_ms_max),
            stats.as_ref().map_or(0, |s| s.rebases),
            stats.as_ref().map_or(0.0, |s| s.rebase_ms_max),
            stats.as_ref().map_or(0.0, |s| s.update_ms_max),
            gpu.map_or(f64::NAN, |g| g.total_ms()),
        );
        println!("{line}");
    }
}

/// Loads the world and system configs and generates the material textures on a thread, so the
/// main thread never reads a file or runs generation (D9).
fn spawn_startup(
    args: &Args,
    world_path: &std::path::Path,
    system_path: &std::path::Path,
    tx: Sender<Result<Startup, String>>,
) {
    let args = args.clone();
    let world_path = world_path.to_path_buf();
    let system_path = system_path.to_path_buf();
    let spawned = std::thread::Builder::new()
        .name("vx-startup".into())
        .spawn(move || {
            let start = Instant::now();
            let result = (|| {
                let read = |path: &std::path::Path| {
                    std::fs::read_to_string(path)
                        .map_err(|e| format!("reading {}: {e}", path.display()))
                };
                let config = parse_config(&read(&world_path)?, &args)?;
                let system = parse_system(&read(&system_path)?, &args)?;
                let generated = material_textures(config.seed, MATERIAL_TEXTURE_SIZE);
                let textures = MaterialTextures {
                    size: generated.size,
                    layers: generated.layers,
                }
                .prepare()?;
                Ok(Startup {
                    config,
                    system,
                    textures,
                    load_ms: start.elapsed().as_secs_f64() * 1.0e3,
                })
            })();
            // A closed receiver means the app is shutting down.
            let _ = tx.send(result);
        });
    if let Err(e) = spawned {
        tracing::error!("could not spawn the startup thread: {e}");
    }
}

/// Parses the world config text and applies the one run overrides, validating the result.
fn parse_config(text: &str, args: &Args) -> Result<WorldConfig, String> {
    let mut config = WorldConfig::from_ron_str(text).map_err(|e| e.to_string())?;
    if let Some(v) = args.voxel_size_m {
        config.voxel_size_m = v;
    }
    if let Some(s) = args.seed {
        config.seed = s;
    }
    if let Some(r) = args.lod_radius_chunks {
        config.lod_radius_chunks = r;
    }
    if let Some(l) = args.lod_levels {
        config.lod_levels = l;
    }
    config.validate().map_err(|e| e.to_string())?;
    Ok(config)
}

/// Parses the system config text and applies the planet radius override, validating the result.
fn parse_system(text: &str, args: &Args) -> Result<SystemConfig, String> {
    let mut system = SystemConfig::from_ron_str(text).map_err(|e| e.to_string())?;
    if let Some(r) = args.radius_km {
        system.planet.params.radius_km = r;
    }
    system.validate().map_err(|e| e.to_string())?;
    Ok(system)
}
