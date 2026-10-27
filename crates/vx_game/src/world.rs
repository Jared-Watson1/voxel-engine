//! The streamed chunk world: generate, mesh and upload as pipeline stages on the job system
//! (D9, D12), streaming around the camera within the configured view radius, and brush edits that
//! remesh at priority across chunk borders (D16).
//!
//! The main thread only schedules, drains result channels, uploads within the per frame byte
//! budget and answers picking rays; every generate and mesh runs on a worker.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use glam::{DVec3, IVec3, Vec3};
use vx_core::{
    Budgets, CHUNK_SIZE, CancelToken, JobSystem, Priority, Stage, StageStats, WorldConfig,
    WorldPos, camera_relative,
};
use vx_render::{ChunkDraw, MATERIAL_TILE_M, MeshHandle, Renderer};
use vx_voxel::{Brush, Chunk, ChunkKey, ChunkStore, PackedMesh, Voxel, mesh_chunk};
use vx_worldgen::TerrainSampler;

pub const STAGE_GENERATE: Stage = Stage(0);
pub const STAGE_MESH: Stage = Stage(1);
pub const STAGE_NAMES: [&str; 2] = ["generate", "mesh"];

/// Seconds between repeated pool full log lines.
const POOL_FULL_LOG_INTERVAL_S: f64 = 2.0;
/// Generate jobs queued per frame while filling the view radius, so a recentre spreads its
/// scheduling over frames instead of spiking the main thread; nearest chunks go first.
const SPAWNS_PER_FRAME: usize = 1024;
/// Frames between full scans of the slot map for the overlay counts.
const COUNT_REFRESH_FRAMES: u64 = 30;

enum ChunkData {
    /// Generation in flight; nothing to read or edit yet.
    Pending,
    Compressed(ChunkStore),
    /// Edited chunks stay dense so brushes never pay a decompress and recompress per stroke (D17).
    Dense(Box<Chunk>),
}

struct Slot {
    /// Bumped on every regenerate or edit; results tagged with an older generation are dropped.
    generation: u64,
    data: ChunkData,
    mesh: Option<MeshHandle>,
    has_surface: bool,
    /// Cancels the job in flight for the current generation.
    token: CancelToken,
    bytes: usize,
    distance_sq: u64,
}

enum Message {
    Generated {
        key: ChunkKey,
        generation: u64,
        store: ChunkStore,
        generate_ms: f64,
    },
    Meshed {
        key: ChunkKey,
        generation: u64,
        priority: Priority,
        mesh: PackedMesh,
        mesh_ms: f64,
    },
}

struct Upload {
    priority: Priority,
    seq: u64,
    key: ChunkKey,
    generation: u64,
    mesh: PackedMesh,
}

impl PartialEq for Upload {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl Eq for Upload {}

impl PartialOrd for Upload {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Upload {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.priority, self.seq).cmp(&(other.priority, other.seq))
    }
}

/// Counters shown in the overlay and the BENCH line.
#[derive(Debug, Clone, Default)]
pub struct WorldStats {
    pub resident: usize,
    pub meshed: usize,
    pub dense: usize,
    pub pending: usize,
    pub upload_queued: usize,
    pub upload_bytes_frame: usize,
    pub upload_bytes_peak: usize,
    pub upload_ms_frame: f64,
    pub voxel_bytes: usize,
    pub voxel_bytes_peak: usize,
    pub generated_total: u64,
    pub meshed_total: u64,
    pub uploaded_total: u64,
    pub upload_failures: u64,
    pub last_generate_ms: f64,
    pub last_mesh_ms: f64,
    pub stream_ms_last: f64,
    pub stream_ms_max: f64,
    /// Main thread time of the whole update this frame and its peak.
    pub update_ms: f64,
    pub update_ms_max: f64,
    pub draw_list_ms: f64,
    pub edits: u64,
    pub queued_peak: usize,
    pub stages: Vec<StageStats>,
}

pub struct World {
    config: WorldConfig,
    sampler: Arc<TerrainSampler>,
    jobs: JobSystem,
    slots: HashMap<ChunkKey, Slot>,
    /// Chunks with a GPU mesh, so the draw list never scans the mostly uniform slot map.
    meshed: HashMap<ChunkKey, MeshHandle>,
    /// Chunk offsets within the view radius, nearest first, computed once.
    offsets: Vec<(IVec3, u64)>,
    centre: Option<ChunkKey>,
    /// Next entry of `offsets` to check for a missing chunk; reset to 0 on every recentre.
    scan_cursor: usize,
    frame: u64,
    tx: Sender<Message>,
    rx: Receiver<Message>,
    uploads: BinaryHeap<Reverse<Upload>>,
    upload_seq: u64,
    last_pool_full_log: Option<Instant>,
    stats: WorldStats,
}

impl World {
    pub fn new(config: WorldConfig, jobs: JobSystem) -> Self {
        let sampler = Arc::new(TerrainSampler::new(config.seed, config.terrain));
        let r = i32::try_from(config.view_radius_chunks).unwrap_or(i32::MAX);
        let r_sq = u64::from(config.view_radius_chunks).pow(2);
        let mut offsets = Vec::new();
        for z in -r..=r {
            for y in -r..=r {
                for x in -r..=r {
                    let d =
                        (i64::from(x).pow(2) + i64::from(y).pow(2) + i64::from(z).pow(2)) as u64;
                    if d <= r_sq {
                        offsets.push((IVec3::new(x, y, z), d));
                    }
                }
            }
        }
        offsets.sort_by_key(|&(o, d)| (d, o.x, o.y, o.z));
        let (tx, rx) = channel();
        Self {
            config,
            sampler,
            jobs,
            slots: HashMap::new(),
            meshed: HashMap::new(),
            offsets,
            centre: None,
            scan_cursor: 0,
            frame: 0,
            tx,
            rx,
            uploads: BinaryHeap::new(),
            upload_seq: 0,
            last_pool_full_log: None,
            stats: WorldStats::default(),
        }
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn sampler(&self) -> &TerrainSampler {
        &self.sampler
    }

    pub fn stats(&self) -> &WorldStats {
        &self.stats
    }

    /// True once every chunk in the view radius has been generated and uploaded or found empty,
    /// so measurements can start from a settled state.
    pub fn is_settled(&self) -> bool {
        self.centre.is_some()
            && self.scan_cursor >= self.offsets.len()
            && self.uploads.is_empty()
            && self.jobs.queued_total() == 0
            && self.jobs.all_stats().iter().all(|s| s.running == 0)
            && self
                .slots
                .values()
                .all(|s| !matches!(s.data, ChunkData::Pending))
    }

    /// Runs the main thread part of the pipeline for one frame: streaming around `camera`,
    /// draining worker results and uploading within the byte budget.
    pub fn update(&mut self, camera: WorldPos, renderer: &mut Renderer) {
        profiling::scope!("World::update");
        let start = Instant::now();
        self.frame += 1;
        let key = ChunkKey::containing(camera, &self.config);
        if self.centre != Some(key) {
            let recentre_start = Instant::now();
            self.recentre(key, renderer);
            let ms = recentre_start.elapsed().as_secs_f64() * 1.0e3;
            self.stats.stream_ms_last = ms;
            self.stats.stream_ms_max = self.stats.stream_ms_max.max(ms);
        }
        self.scan();
        self.poll_results(renderer);
        self.upload(renderer);
        if self.frame.is_multiple_of(COUNT_REFRESH_FRAMES) {
            self.refresh_counts();
        }
        let ms = start.elapsed().as_secs_f64() * 1.0e3;
        self.stats.update_ms = ms;
        self.stats.update_ms_max = self.stats.update_ms_max.max(ms);
    }

    /// Evicts chunks beyond the evict radius and restarts the nearest first scan for missing ones.
    fn recentre(&mut self, centre: ChunkKey, renderer: &mut Renderer) {
        profiling::scope!("World::recentre");
        self.centre = Some(centre);
        self.scan_cursor = 0;
        let evict_sq = u64::from(self.config.evict_radius_chunks()).pow(2);
        let mut gone = Vec::new();
        for (key, slot) in &mut self.slots {
            slot.distance_sq = key.distance_sq(centre);
            if slot.distance_sq > evict_sq {
                gone.push(*key);
            }
        }
        for key in gone {
            self.remove(key, renderer);
        }
        self.refresh_counts();
    }

    /// Queues generation for up to `SPAWNS_PER_FRAME` missing chunks, nearest first.
    fn scan(&mut self) {
        let Some(centre) = self.centre else { return };
        let mut spawned = 0;
        while self.scan_cursor < self.offsets.len() && spawned < SPAWNS_PER_FRAME {
            let (offset, distance_sq) = self.offsets[self.scan_cursor];
            self.scan_cursor += 1;
            let key = centre.offset(offset.x, offset.y, offset.z);
            if !self.slots.contains_key(&key) {
                self.spawn_generate(key, distance_sq);
                spawned += 1;
            }
        }
    }

    fn spawn_generate(&mut self, key: ChunkKey, distance_sq: u64) {
        let token = CancelToken::new();
        let generation = 1;
        self.slots.insert(
            key,
            Slot {
                generation,
                data: ChunkData::Pending,
                mesh: None,
                has_surface: false,
                token: token.clone(),
                bytes: 0,
                distance_sq,
            },
        );
        let priority = Priority::streaming(distance_sq);
        let sampler = self.sampler.clone();
        let config = self.config;
        let tx = self.tx.clone();
        let jobs = self.jobs.clone();
        self.jobs
            .spawn(STAGE_GENERATE, priority, token, move |token| {
                let start = Instant::now();
                let mut chunk = Chunk::for_key(key, &config);
                sampler.fill_chunk(&mut chunk);
                let store = ChunkStore::compress(&chunk);
                let generate_ms = start.elapsed().as_secs_f64() * 1.0e3;
                if token.is_cancelled() {
                    return;
                }
                let has_surface = store.has_surface();
                // A closed receiver means the world was replaced or the app is shutting down.
                let _ = tx.send(Message::Generated {
                    key,
                    generation,
                    store,
                    generate_ms,
                });
                if has_surface {
                    spawn_mesh(&jobs, priority, token.clone(), key, generation, chunk, tx);
                }
            });
    }

    fn poll_results(&mut self, renderer: &mut Renderer) {
        profiling::scope!("World::poll_results");
        while let Ok(message) = self.rx.try_recv() {
            match message {
                Message::Generated {
                    key,
                    generation,
                    store,
                    generate_ms,
                } => {
                    let Some(slot) = self.slots.get_mut(&key) else {
                        continue;
                    };
                    if slot.generation != generation {
                        continue;
                    }
                    let bytes = store.resident_bytes();
                    slot.has_surface = store.has_surface();
                    slot.data = ChunkData::Compressed(store);
                    self.stats.voxel_bytes = self.stats.voxel_bytes - slot.bytes + bytes;
                    slot.bytes = bytes;
                    self.stats.generated_total += 1;
                    self.stats.last_generate_ms = generate_ms;
                    if !slot.has_surface
                        && let Some(handle) = slot.mesh.take()
                    {
                        self.meshed.remove(&key);
                        renderer.free_mesh(handle);
                    }
                }
                Message::Meshed {
                    key,
                    generation,
                    priority,
                    mesh,
                    mesh_ms,
                } => {
                    let Some(slot) = self.slots.get(&key) else {
                        continue;
                    };
                    if slot.generation != generation {
                        continue;
                    }
                    self.stats.meshed_total += 1;
                    self.stats.last_mesh_ms = mesh_ms;
                    self.upload_seq += 1;
                    self.uploads.push(Reverse(Upload {
                        priority,
                        seq: self.upload_seq,
                        key,
                        generation,
                        mesh,
                    }));
                }
            }
        }
        self.stats.voxel_bytes_peak = self.stats.voxel_bytes_peak.max(self.stats.voxel_bytes);
    }

    /// Copies queued meshes to the GPU, nearest and edited first, until the per frame byte budget
    /// is reached. At least one mesh is uploaded per frame so a single large mesh cannot stall.
    fn upload(&mut self, renderer: &mut Renderer) {
        profiling::scope!("World::upload");
        let start = Instant::now();
        let mut bytes = 0usize;
        while let Some(Reverse(next)) = self.uploads.peek() {
            let size = next.mesh.vertex_bytes() + next.mesh.index_bytes();
            if bytes > 0 && bytes + size > Budgets::UPLOAD_BYTES_PER_FRAME {
                break;
            }
            let Some(Reverse(upload)) = self.uploads.pop() else {
                break;
            };
            let Some(slot) = self.slots.get_mut(&upload.key) else {
                continue;
            };
            if slot.generation != upload.generation {
                continue;
            }
            bytes += size;
            if upload.mesh.is_empty() {
                if let Some(old) = slot.mesh.take() {
                    self.meshed.remove(&upload.key);
                    renderer.free_mesh(old);
                }
                continue;
            }
            match renderer.upload_mesh(&upload.mesh.vertices, &upload.mesh.indices) {
                Ok(handle) => {
                    if let Some(old) = slot.mesh.replace(handle) {
                        renderer.free_mesh(old);
                    }
                    self.meshed.insert(upload.key, handle);
                    self.stats.uploaded_total += 1;
                }
                Err(e) => {
                    self.stats.upload_failures += 1;
                    let due = self
                        .last_pool_full_log
                        .is_none_or(|t| t.elapsed().as_secs_f64() > POOL_FULL_LOG_INTERVAL_S);
                    if due {
                        tracing::error!("mesh upload failed for {:?}: {e}", upload.key);
                        self.last_pool_full_log = Some(Instant::now());
                    }
                }
            }
        }
        self.stats.upload_bytes_frame = bytes;
        self.stats.upload_bytes_peak = self.stats.upload_bytes_peak.max(bytes);
        self.stats.upload_ms_frame = start.elapsed().as_secs_f64() * 1.0e3;
    }

    fn refresh_counts(&mut self) {
        let mut dense = 0;
        let mut pending = 0;
        for slot in self.slots.values() {
            match slot.data {
                ChunkData::Pending => pending += 1,
                ChunkData::Dense(_) => dense += 1,
                ChunkData::Compressed(_) => {}
            }
        }
        self.stats.resident = self.slots.len();
        self.stats.meshed = self.meshed.len();
        self.stats.dense = dense;
        self.stats.pending = pending;
        self.stats.upload_queued = self.uploads.len();
        self.stats.stages = self.jobs.all_stats();
        let queued = self.jobs.queued_total() + self.uploads.len();
        self.stats.queued_peak = self.stats.queued_peak.max(queued);
    }

    fn remove(&mut self, key: ChunkKey, renderer: &mut Renderer) {
        if let Some(slot) = self.slots.remove(&key) {
            slot.token.cancel();
            if let Some(handle) = slot.mesh {
                self.meshed.remove(&key);
                renderer.free_mesh(handle);
            }
            self.stats.voxel_bytes -= slot.bytes;
        }
    }

    /// Every resident mesh with its camera relative offset, computed in f64 (D4), nearest first
    /// is not required because the GPU culls.
    pub fn fill_draw_list(&mut self, camera: WorldPos, out: &mut Vec<ChunkDraw>) {
        profiling::scope!("World::fill_draw_list");
        let start = Instant::now();
        out.clear();
        for (key, &handle) in &self.meshed {
            let origin = key.origin(&self.config);
            let phase = origin.m.map(|v| v.rem_euclid(MATERIAL_TILE_M));
            out.push(ChunkDraw {
                handle,
                offset_m: camera_relative(origin, camera),
                texture_phase_m: Vec3::new(phase.x as f32, phase.y as f32, phase.z as f32),
            });
        }
        self.stats.draw_list_ms = start.elapsed().as_secs_f64() * 1.0e3;
    }

    /// Reads the stored sample containing `p`, or `None` where no generated chunk holds it.
    pub fn sample_at(&self, p: DVec3) -> Option<Voxel> {
        let ws = (p / self.config.voxel_size_m).floor().as_ivec3();
        let n = CHUNK_SIZE as i32;
        let key = ChunkKey::new(ws.x.div_euclid(n), ws.y.div_euclid(n), ws.z.div_euclid(n));
        let local = ws - key.as_ivec3() * n;
        let slot = self.slots.get(&key)?;
        let (x, y, z) = (local.x as isize, local.y as isize, local.z as isize);
        match &slot.data {
            ChunkData::Pending => None,
            ChunkData::Compressed(store) => Some(store.get(x, y, z)),
            ChunkData::Dense(chunk) => Some(chunk.get(x, y, z)),
        }
    }

    /// Marches a ray through the stored samples in half voxel steps and returns the first point
    /// inside the solid, refined by bisection to a few millimetres.
    pub fn raycast(&self, origin: WorldPos, direction: DVec3, max_m: f64) -> Option<DVec3> {
        profiling::scope!("World::raycast");
        let direction = direction.normalize_or_zero();
        if direction == DVec3::ZERO {
            return None;
        }
        let step = 0.5 * self.config.voxel_size_m;
        let inside = |t: f64| {
            self.sample_at(origin.m + direction * t)
                .is_some_and(Voxel::is_inside)
        };
        let mut t = 0.0;
        if inside(0.0) {
            return Some(origin.m);
        }
        while t < max_m {
            let next = t + step;
            if inside(next) {
                let (mut lo, mut hi) = (t, next);
                for _ in 0..8 {
                    let mid = 0.5 * (lo + hi);
                    if inside(mid) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(origin.m + direction * hi);
            }
            t = next;
        }
        None
    }

    /// Applies a brush to every generated chunk it touches, apron copies included, and queues a
    /// priority remesh of each changed chunk (D9). Returns how many chunks changed.
    pub fn apply_brush(&mut self, brush: &Brush, renderer: &mut Renderer) -> usize {
        profiling::scope!("World::apply_brush");
        let mut changed = 0;
        for key in brush.affected_chunks(&self.config) {
            let Some(slot) = self.slots.get_mut(&key) else {
                continue;
            };
            if let ChunkData::Compressed(store) = &slot.data {
                let mut chunk = Chunk::for_key(key, &self.config);
                store.decompress_into(&mut chunk);
                let bytes = std::mem::size_of::<Chunk>() + chunk.samples().len() * 2;
                self.stats.voxel_bytes = self.stats.voxel_bytes - slot.bytes + bytes;
                slot.bytes = bytes;
                slot.data = ChunkData::Dense(Box::new(chunk));
            }
            let ChunkData::Dense(chunk) = &mut slot.data else {
                continue;
            };
            if !brush.apply(chunk) {
                continue;
            }
            changed += 1;
            slot.generation += 1;
            slot.token.cancel();
            slot.token = CancelToken::new();
            slot.has_surface = chunk.has_surface();
            if slot.has_surface {
                spawn_mesh(
                    &self.jobs,
                    Priority::EDIT,
                    slot.token.clone(),
                    key,
                    slot.generation,
                    (**chunk).clone(),
                    self.tx.clone(),
                );
            } else if let Some(handle) = slot.mesh.take() {
                self.meshed.remove(&key);
                renderer.free_mesh(handle);
            }
        }
        if changed > 0 {
            self.stats.edits += 1;
        }
        changed
    }

    /// Frees every GPU mesh; called before the world is dropped so the pools are returned.
    pub fn release(&mut self, renderer: &mut Renderer) {
        let keys: Vec<ChunkKey> = self.slots.keys().copied().collect();
        for key in keys {
            self.remove(key, renderer);
        }
    }
}

fn spawn_mesh(
    jobs: &JobSystem,
    priority: Priority,
    token: CancelToken,
    key: ChunkKey,
    generation: u64,
    chunk: Chunk,
    tx: Sender<Message>,
) {
    jobs.spawn(STAGE_MESH, priority, token, move |token| {
        let start = Instant::now();
        let mesh = mesh_chunk(&chunk);
        let mesh_ms = start.elapsed().as_secs_f64() * 1.0e3;
        if token.is_cancelled() {
            return;
        }
        let _ = tx.send(Message::Meshed {
            key,
            generation,
            priority,
            mesh,
            mesh_ms,
        });
    });
}
