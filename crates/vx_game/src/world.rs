//! The planet world: the octree clipmap (D20) decides which chunk keys exist and which are drawn;
//! this module generates and meshes them on the job system (D9, D12), uploads them within the
//! per frame byte budget, keeps their chunk table placements relative to a floating render origin
//! (D23), and replays the edit log at every level so an edit shows from the ground to orbit.
//!
//! The main thread only schedules, drains result channels, uploads and answers picking rays. The
//! octree's own update does bounded work per frame.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use glam::{DVec3, IVec3, Vec3};
use vx_core::{
    Budgets, CHUNK_SIZE, CancelToken, JobSystem, PlanetParams, Priority, Stage, StageStats,
    WorldConfig, WorldPos,
};
use vx_render::{ChunkPlacement, MATERIAL_TILE_M, MeshHandle, MeshUpload, Renderer};
use vx_voxel::{
    Brush, Chunk, ChunkKey, ChunkStore, Face, LodParams, LodStats, LodTree, PackedMesh, Voxel,
    mesh_chunk_lod, transition_slab,
};
use vx_worldgen::{ChunkClass, PlanetSampler};

pub const STAGE_GENERATE: Stage = Stage(0);
pub const STAGE_MESH: Stage = Stage(1);
pub const STAGE_NAMES: [&str; 2] = ["generate", "mesh"];

/// Seconds between repeated pool full log lines.
const POOL_FULL_LOG_INTERVAL_S: f64 = 2.0;
/// Frames an upload is retried when a pool is full before the key is reported failed.
const UPLOAD_RETRY_FRAMES: u32 = 120;
/// Coarsest level whose resident samples a picking ray reads before falling back to the sampler.
const PICK_MAX_LEVEL: u8 = 3;

enum Message {
    Generated {
        key: ChunkKey,
        generation: u64,
        store: ChunkStore,
        has_surface: bool,
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
    retries: u32,
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

/// One key the octree asked for.
struct Node {
    /// Taken from a world wide counter on every generate or edit, so a result from an earlier
    /// generation of this key, or from a previous residency of it, is dropped.
    generation: u64,
    token: CancelToken,
    /// Compressed samples once generated; `None` while pending or when the key was classified
    /// as all air or all rock from its bounds.
    store: Option<ChunkStore>,
    mesh: Option<MeshHandle>,
    /// A job for the current generation is in flight.
    pending: bool,
    /// The octree has been told this key is ready; later remeshes swap the mesh in place.
    reported: bool,
    visible: bool,
    mask: u8,
    bytes: usize,
}

/// Counters shown in the overlay and the BENCH line.
#[derive(Debug, Clone, Default)]
pub struct WorldStats {
    pub nodes: usize,
    pub meshed: usize,
    pub pending: usize,
    pub upload_queued: usize,
    pub upload_bytes_frame: usize,
    pub upload_bytes_peak: usize,
    pub upload_ms_frame: f64,
    pub voxel_bytes: usize,
    pub voxel_bytes_peak: usize,
    pub generated_total: u64,
    pub classified_empty_total: u64,
    pub meshed_total: u64,
    pub uploaded_total: u64,
    pub upload_failures: u64,
    pub last_generate_ms: f64,
    pub last_mesh_ms: f64,
    /// Main thread time of the whole update this frame and its peak.
    pub update_ms: f64,
    pub update_ms_max: f64,
    pub lod_ms_max: f64,
    pub rebases: u64,
    pub rebase_ms_max: f64,
    pub edits: u64,
    pub edit_log_entries: usize,
    pub queued_peak: usize,
    pub stages: Vec<StageStats>,
    pub lod: LodStats,
}

pub struct World {
    config: WorldConfig,
    planet: PlanetParams,
    sampler: Arc<PlanetSampler>,
    jobs: JobSystem,
    lod: LodTree,
    nodes: HashMap<ChunkKey, Node>,
    /// Every brush applied this session, in order. A generate or mesh job replays the brushes
    /// whose bounds reach its samples, at its own level, so edits survive eviction and show at
    /// every level (D16, D20).
    edits: Arc<Vec<Brush>>,
    /// The render origin in body fixed metres (D23).
    origin: DVec3,
    next_generation: u64,
    tx: Sender<Message>,
    rx: Receiver<Message>,
    uploads: BinaryHeap<Reverse<Upload>>,
    upload_seq: u64,
    /// Meshes of chunks an edit emptied, freed at the next upload stage.
    removed_meshes: Vec<MeshHandle>,
    last_pool_full_log: Option<Instant>,
    stats: WorldStats,
}

impl World {
    pub fn new(config: WorldConfig, planet: PlanetParams, jobs: JobSystem, camera: DVec3) -> Self {
        let sampler = Arc::new(PlanetSampler::new(
            config.seed,
            &planet,
            config.voxel_size_m,
        ));
        let lod = LodTree::new(LodParams::new(&config, sampler.shell_m()));
        let (tx, rx) = channel();
        Self {
            config,
            planet,
            sampler,
            jobs,
            lod,
            nodes: HashMap::new(),
            edits: Arc::new(Vec::new()),
            origin: camera,
            next_generation: 1,
            tx,
            rx,
            uploads: BinaryHeap::new(),
            upload_seq: 0,
            removed_meshes: Vec::new(),
            last_pool_full_log: None,
            stats: WorldStats::default(),
        }
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn planet(&self) -> &PlanetParams {
        &self.planet
    }

    pub fn sampler(&self) -> &PlanetSampler {
        &self.sampler
    }

    pub fn stats(&self) -> &WorldStats {
        &self.stats
    }

    /// The render origin in body fixed metres; the camera offset sent to the GPU is the camera
    /// position minus this.
    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// Moves the render origin without the rebase threshold, for placing the camera before any
    /// mesh exists; resident meshes keep their old placements until the next rebase.
    pub fn set_origin(&mut self, origin: DVec3) {
        self.origin = origin;
    }

    /// Height of `position` above the terrain under it (full detail height, no edits).
    pub fn altitude_m(&self, position: DVec3) -> f64 {
        position.length() - self.planet.radius_m() - self.sampler.height_m(position)
    }

    /// True once nothing is requested, queued or uploading, so measurements can start from a
    /// settled state.
    pub fn is_settled(&self) -> bool {
        self.stats.lod.pending == 0
            && self.uploads.is_empty()
            && self.jobs.queued_total() == 0
            && self.jobs.all_stats().iter().all(|s| s.running == 0)
            && self.nodes.values().all(|n| !n.pending)
    }

    /// Runs the main thread part of the pipeline for one frame: the render origin, the octree,
    /// worker results and uploads within the byte budget.
    pub fn update(&mut self, camera: WorldPos, renderer: &mut Renderer) {
        profiling::scope!("World::update");
        let start = Instant::now();
        self.rebase_if_needed(camera.m, renderer);
        let lod_start = Instant::now();
        let update = self.lod.update(camera.m);
        self.stats.lod_ms_max = self
            .stats
            .lod_ms_max
            .max(lod_start.elapsed().as_secs_f64() * 1.0e3);
        for request in &update.requests {
            self.request(request.key, request.priority);
        }
        for &(key, visible) in &update.visibility {
            if let Some(node) = self.nodes.get_mut(&key) {
                node.visible = visible;
            }
            self.write_placement(key, renderer);
        }
        for &(key, mask) in &update.masks {
            if let Some(node) = self.nodes.get_mut(&key) {
                node.mask = mask;
            }
            self.write_placement(key, renderer);
        }
        for &key in &update.releases {
            self.remove(key, renderer);
        }
        self.poll_results();
        self.upload(renderer);
        self.refresh_stats();
        let ms = start.elapsed().as_secs_f64() * 1.0e3;
        self.stats.update_ms = ms;
        self.stats.update_ms_max = self.stats.update_ms_max.max(ms);
    }

    fn write_placement(&self, key: ChunkKey, renderer: &mut Renderer) {
        if let Some(node) = self.nodes.get(&key)
            && let Some(handle) = node.mesh
        {
            renderer.set_placement(handle, self.placement(key, node));
        }
    }

    /// Moves the render origin to the camera when it has drifted beyond the rebase distance and
    /// rewrites every resident chunk's placement once (D23).
    fn rebase_if_needed(&mut self, camera: DVec3, renderer: &mut Renderer) {
        if (camera - self.origin).length() <= self.config.origin_rebase_m {
            return;
        }
        let start = Instant::now();
        self.origin = camera;
        for (&key, node) in &self.nodes {
            if let Some(handle) = node.mesh {
                renderer.set_placement(handle, self.placement(key, node));
            }
        }
        self.stats.rebases += 1;
        self.stats.rebase_ms_max = self
            .stats
            .rebase_ms_max
            .max(start.elapsed().as_secs_f64() * 1.0e3);
    }

    fn placement(&self, key: ChunkKey, node: &Node) -> ChunkPlacement {
        let origin = key.origin_m(self.config.voxel_size_m);
        let phase = origin.map(|v| v.rem_euclid(MATERIAL_TILE_M));
        ChunkPlacement {
            offset_m: (origin - self.origin).as_vec3(),
            texture_phase_m: Vec3::new(phase.x as f32, phase.y as f32, phase.z as f32),
            voxel_size_m: key.voxel_size_m(self.config.voxel_size_m) as f32,
            transition_mask: node.mask,
            visible: node.visible,
        }
    }

    fn take_generation(&mut self) -> u64 {
        let g = self.next_generation;
        self.next_generation += 1;
        g
    }

    /// Starts making `key` ready: keys whose bounds lie outside the planet shell are ready at once
    /// with no data; the rest get a generate job.
    fn request(&mut self, key: ChunkKey, priority: Priority) {
        if self.nodes.contains_key(&key) {
            return;
        }
        let generation = self.take_generation();
        let mut node = Node {
            generation,
            token: CancelToken::new(),
            store: None,
            mesh: None,
            pending: false,
            reported: false,
            visible: false,
            mask: 0,
            bytes: 0,
        };
        if self.sampler.classify(key) == ChunkClass::Surface {
            node.pending = true;
            self.spawn_generate(key, generation, node.token.clone(), priority);
        } else {
            node.reported = true;
            self.stats.classified_empty_total += 1;
            self.lod.mark_ready(key);
        }
        self.nodes.insert(key, node);
    }

    /// Queues a generate job for `key`: it samples the planet at the key's level, replays the
    /// edits that reach it and, when the chunk has a surface, queues the mesh job with the six
    /// transition slabs itself.
    fn spawn_generate(
        &mut self,
        key: ChunkKey,
        generation: u64,
        token: CancelToken,
        priority: Priority,
    ) {
        let edits = self.edits.clone();
        let sampler = self.sampler.clone();
        let base_voxel_size_m = self.config.voxel_size_m;
        let tx = self.tx.clone();
        let jobs = self.jobs.clone();
        self.jobs
            .spawn(STAGE_GENERATE, priority, token, move |token| {
                let start = Instant::now();
                let mut chunk = Chunk::new(key, base_voxel_size_m);
                sampler.fill_chunk(&mut chunk);
                for brush in edits.iter() {
                    brush.apply(&mut chunk);
                }
                let has_surface = chunk.has_surface();
                let store = ChunkStore::compress(&chunk);
                if token.is_cancelled() {
                    return;
                }
                let _ = tx.send(Message::Generated {
                    key,
                    generation,
                    store,
                    has_surface,
                    generate_ms: start.elapsed().as_secs_f64() * 1.0e3,
                });
                if has_surface {
                    let job = MeshJob {
                        key,
                        generation,
                        priority,
                        chunk,
                        sampler,
                        edits,
                        tx,
                    };
                    job.spawn(&jobs, token.clone());
                }
            });
    }

    fn poll_results(&mut self) {
        profiling::scope!("World::poll_results");
        while let Ok(message) = self.rx.try_recv() {
            match message {
                Message::Generated {
                    key,
                    generation,
                    store,
                    has_surface,
                    generate_ms,
                } => {
                    let Some(node) = self.nodes.get_mut(&key) else {
                        continue;
                    };
                    if node.generation != generation {
                        continue;
                    }
                    self.stats.generated_total += 1;
                    self.stats.last_generate_ms = generate_ms;
                    let bytes = store.resident_bytes();
                    self.stats.voxel_bytes = self.stats.voxel_bytes - node.bytes + bytes;
                    node.bytes = bytes;
                    node.store = Some(store);
                    if !has_surface {
                        node.pending = false;
                        if let Some(handle) = node.mesh.take() {
                            self.removed_meshes.push(handle);
                        }
                        if !node.reported {
                            node.reported = true;
                            self.lod.mark_ready(key);
                        }
                    }
                }
                Message::Meshed {
                    key,
                    generation,
                    priority,
                    mesh,
                    mesh_ms,
                } => {
                    let Some(node) = self.nodes.get(&key) else {
                        continue;
                    };
                    if node.generation != generation {
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
                        retries: 0,
                        mesh,
                    }));
                }
            }
        }
    }

    /// Uploads queued meshes in priority order until the frame's byte budget is spent (at least
    /// one mesh per frame so a mesh larger than the budget still lands).
    fn upload(&mut self, renderer: &mut Renderer) {
        profiling::scope!("World::upload");
        for handle in self.removed_meshes.drain(..) {
            renderer.free_mesh(handle);
        }
        let start = Instant::now();
        let mut bytes = 0;
        let mut retry = Vec::new();
        while let Some(Reverse(mut upload)) = self.uploads.pop() {
            let size = upload.mesh.vertex_bytes() + upload.mesh.index_bytes();
            if bytes > 0 && bytes + size > Budgets::UPLOAD_BYTES_PER_FRAME {
                self.uploads.push(Reverse(upload));
                break;
            }
            let key = upload.key;
            let Some(node) = self.nodes.get(&key) else {
                continue;
            };
            if node.generation != upload.generation {
                continue;
            }
            bytes += size;
            let result = if upload.mesh.is_empty() {
                None
            } else {
                let mesh = MeshUpload {
                    vertices: &upload.mesh.vertices,
                    indices: &upload.mesh.indices,
                    regular_index_count: upload.mesh.regular_index_count,
                    transition_index_counts: upload.mesh.transition_index_counts,
                };
                Some(renderer.upload_mesh(&mesh, self.placement(key, node)))
            };
            let Some(node) = self.nodes.get_mut(&key) else {
                continue;
            };
            match result {
                Some(Err(e)) => {
                    let due = self
                        .last_pool_full_log
                        .is_none_or(|t| t.elapsed().as_secs_f64() > POOL_FULL_LOG_INTERVAL_S);
                    if due {
                        tracing::error!("mesh upload failed for {key:?}: {e}");
                        self.last_pool_full_log = Some(Instant::now());
                    }
                    upload.retries += 1;
                    if upload.retries < UPLOAD_RETRY_FRAMES {
                        retry.push(upload);
                    } else {
                        self.stats.upload_failures += 1;
                        node.pending = false;
                        if !node.reported {
                            node.reported = true;
                            self.lod.mark_failed(key);
                        }
                    }
                    continue;
                }
                Some(Ok(handle)) => {
                    if let Some(old) = node.mesh.replace(handle) {
                        renderer.free_mesh(old);
                    }
                    self.stats.uploaded_total += 1;
                }
                None => {
                    if let Some(old) = node.mesh.take() {
                        renderer.free_mesh(old);
                    }
                }
            }
            node.pending = false;
            if !node.reported {
                node.reported = true;
                self.lod.mark_ready(key);
            }
        }
        for upload in retry {
            self.uploads.push(Reverse(upload));
        }
        self.stats.upload_bytes_frame = bytes;
        self.stats.upload_bytes_peak = self.stats.upload_bytes_peak.max(bytes);
        self.stats.upload_ms_frame = start.elapsed().as_secs_f64() * 1.0e3;
    }

    fn remove(&mut self, key: ChunkKey, renderer: &mut Renderer) {
        if let Some(node) = self.nodes.remove(&key) {
            node.token.cancel();
            if let Some(handle) = node.mesh {
                renderer.free_mesh(handle);
            }
            self.stats.voxel_bytes -= node.bytes;
        }
    }

    fn refresh_stats(&mut self) {
        let s = &mut self.stats;
        s.nodes = self.nodes.len();
        s.meshed = self.nodes.values().filter(|n| n.mesh.is_some()).count();
        s.pending = self.nodes.values().filter(|n| n.pending).count();
        s.upload_queued = self.uploads.len();
        s.voxel_bytes_peak = s.voxel_bytes_peak.max(s.voxel_bytes);
        s.edit_log_entries = self.edits.len();
        s.stages = self.jobs.all_stats();
        s.queued_peak = s.queued_peak.max(self.jobs.queued_total());
        s.lod = self.lod.stats();
    }

    /// The stored sample at world sample `ws` of `level`, or `None` where no generated chunk
    /// of that level holds it.
    fn sample_at_index(&self, level: u8, ws: IVec3) -> Option<Voxel> {
        let n = CHUNK_SIZE as i32;
        let key = ChunkKey::at_level(
            level,
            ws.x.div_euclid(n),
            ws.y.div_euclid(n),
            ws.z.div_euclid(n),
        );
        let local = ws - key.as_ivec3() * n;
        let store = self.nodes.get(&key)?.store.as_ref()?;
        Some(store.get(local.x as isize, local.y as isize, local.z as isize))
    }

    /// Signed distance in metres at `p`, trilinear between the eight surrounding samples of the
    /// finest level resident there (the surface the mesher draws), or from the sampler beyond
    /// the resident levels (edits ignored there).
    pub fn distance_m_at(&self, p: DVec3) -> f64 {
        for level in 0..=PICK_MAX_LEVEL {
            let voxel = self.config.level_voxel_size_m(level);
            let f = p / voxel;
            let base = f.floor();
            let t = f - base;
            let b = base.as_ivec3();
            let mut d = [0.0f64; 8];
            let mut complete = true;
            for (i, slot) in d.iter_mut().enumerate() {
                let o = IVec3::new((i & 1) as i32, ((i >> 1) & 1) as i32, (i >> 2) as i32);
                match self.sample_at_index(level, b + o) {
                    Some(v) => *slot = f64::from(v.distance_voxels()),
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                continue;
            }
            let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
            let x00 = lerp(d[0], d[1], t.x);
            let x10 = lerp(d[2], d[3], t.x);
            let x01 = lerp(d[4], d[5], t.x);
            let x11 = lerp(d[6], d[7], t.x);
            return lerp(lerp(x00, x10, t.y), lerp(x01, x11, t.y), t.z) * voxel;
        }
        self.sampler.sample(p, 0).0
    }

    /// Marches a ray through the field in steps of half the distance (never under half a voxel)
    /// and returns the first point inside the surface, refined by bisection.
    pub fn raycast(&self, origin: WorldPos, direction: DVec3, max_m: f64) -> Option<DVec3> {
        profiling::scope!("World::raycast");
        let direction = direction.normalize_or_zero();
        if direction == DVec3::ZERO {
            return None;
        }
        let min_step = 0.5 * self.config.voxel_size_m;
        let at = |t: f64| self.distance_m_at(origin.m + direction * t);
        let mut t = 0.0;
        let mut d = at(0.0);
        if d < 0.0 {
            return Some(origin.m);
        }
        while t < max_m {
            let next = t + (0.5 * d).max(min_step);
            let dn = at(next);
            if dn < 0.0 {
                let (mut lo, mut hi) = (t, next);
                for _ in 0..12 {
                    let mid = 0.5 * (lo + hi);
                    if at(mid) < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(origin.m + direction * hi);
            }
            t = next;
            d = dn;
        }
        None
    }

    /// Appends the brush to the edit log and regenerates every resident chunk it reaches at any
    /// level, plus the coarse chunks whose transition slabs it reaches (D22: a slab sample of key
    /// `K` lies in the stored block of a child of `K` or of a face neighbour of `K`), at edit
    /// priority. The old meshes stay drawn until the new ones upload. Returns how many chunks
    /// regenerate.
    pub fn apply_brush(&mut self, brush: &Brush) -> usize {
        profiling::scope!("World::apply_brush");
        Arc::make_mut(&mut self.edits).push(*brush);
        let mut keys = Vec::new();
        for level in 0..self.config.lod_levels as u8 {
            for key in brush.affected_chunks_at(&self.config, level) {
                keys.push(key);
                let parent = key.parent();
                keys.push(parent);
                keys.extend(Face::ALL.map(|face| parent.neighbour(face)));
            }
        }
        keys.sort();
        keys.dedup();
        let mut changed = 0;
        for key in keys {
            let Some(node) = self.nodes.get(&key) else {
                continue;
            };
            if node.store.is_none() && !node.pending {
                continue;
            }
            let generation = self.take_generation();
            let token = CancelToken::new();
            if let Some(node) = self.nodes.get_mut(&key) {
                node.token.cancel();
                node.token = token.clone();
                node.generation = generation;
                node.pending = true;
            }
            self.spawn_generate(key, generation, token, Priority::EDIT);
            changed += 1;
        }
        self.stats.edits += 1;
        changed
    }

    /// Clears the peak values so a measurement window reports only its own peaks.
    pub fn reset_peaks(&mut self) {
        let s = &mut self.stats;
        s.upload_bytes_peak = 0;
        s.voxel_bytes_peak = s.voxel_bytes;
        s.update_ms_max = 0.0;
        s.lod_ms_max = 0.0;
        s.rebase_ms_max = 0.0;
        s.queued_peak = 0;
    }

    /// Frees every GPU mesh; called before the world is dropped so the pools are returned.
    pub fn release(&mut self, renderer: &mut Renderer) {
        let keys: Vec<ChunkKey> = self.nodes.keys().copied().collect();
        for key in keys {
            self.remove(key, renderer);
        }
        for handle in self.removed_meshes.drain(..) {
            renderer.free_mesh(handle);
        }
    }
}

/// The mesh stage of one key: fills the six transition slabs from the sampler and the edits
/// (D22) and meshes the chunk with them.
struct MeshJob {
    key: ChunkKey,
    generation: u64,
    priority: Priority,
    chunk: Chunk,
    sampler: Arc<PlanetSampler>,
    edits: Arc<Vec<Brush>>,
    tx: Sender<Message>,
}

impl MeshJob {
    fn spawn(self, jobs: &JobSystem, token: CancelToken) {
        let priority = self.priority;
        jobs.spawn(STAGE_MESH, priority, token, move |token| {
            let start = Instant::now();
            let base_voxel_size_m = self.sampler.base_voxel_size_m();
            let slabs = Face::ALL.map(|face| {
                let mut slab = transition_slab(self.key, face, base_voxel_size_m)?;
                self.sampler.fill_block(&mut slab);
                for brush in self.edits.iter() {
                    brush.apply_block(&mut slab);
                }
                Some(slab)
            });
            if token.is_cancelled() {
                return;
            }
            let mesh = mesh_chunk_lod(&self.chunk, &slabs);
            if token.is_cancelled() {
                return;
            }
            let _ = self.tx.send(Message::Meshed {
                key: self.key,
                generation: self.generation,
                priority,
                mesh,
                mesh_ms: start.elapsed().as_secs_f64() * 1.0e3,
            });
        });
    }
}
