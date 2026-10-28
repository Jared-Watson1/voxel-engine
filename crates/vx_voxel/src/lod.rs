//! Octree clipmap over the whole planet (D20).
//!
//! The tree decides which chunk keys must exist and which are drawn; the caller owns the data,
//! the jobs and the meshes. Each [`LodTree::update`] returns the keys to make ready (generate and
//! mesh, or find empty), the keys to release, and the visibility and transition mask changes that
//! became safe to commit. A split commits only once all eight children are ready and a merge only
//! once the parent is ready, so the drawn set always covers the planet shell with no hole and no
//! overlap. Keys entirely outside the shell are never requested and count as ready and empty.
//!
//! # Structure
//!
//! Nodes live in a flat arena indexed by `u32` slots, with a hash index from [`ChunkKey`] to slot
//! used only for lookups (never iterated, so hash order never reaches an output, engineering rule
//! 2). Only keys whose box intersects the shell are stored; a child outside the shell is absent
//! from its parent's child slots and counts as ready and empty. Every stored node has a role:
//!
//! - staged: a child of a pending split (or a root before the first commit), requested, not drawn;
//! - leaf: committed and drawn;
//! - split: committed, hidden, its children are committed.
//!
//! The drawn set is the committed leaves, which partition the part of the shell under the roots.
//!
//! # Update
//!
//! One linear scan over the arena measures the distance from the camera to each committed node's
//! box in units of the node's own extent and collects candidates, then four phases act on them:
//!
//! 1. split commits, coarse first: a pending split whose children are all ready commits when no
//!    leaf two or more levels coarser would touch the new children (2:1 balance over the 26
//!    neighbourhood). A coarser leaf in the way is forced to split (its children are requested
//!    at once) and held while the finer split waits for it;
//! 2. cancels: a pending split whose node moved beyond `radius + hysteresis` and is not held by a
//!    forced split drops its staged children;
//! 3. merges, fine first: a split node beyond `radius + hysteresis` whose children are all leaves
//!    with no pending split merges when it is ready and no leaf two or more levels finer (or a
//!    pending split that would create one) touches it;
//! 4. split starts, coarse first: a leaf within `radius` extents requests its children.
//!
//! Residency: drawn leaves, staged children and split nodes whose children are all drawn leaves
//! (the only nodes that can merge, so a merge back is instant) are resident. A split node with a
//! split child is a grandparent and is released; it is requested again when its last split child
//! merges. Children of a merged node are released with it.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use glam::{DVec3, IVec3};
use vx_core::{Priority, WorldConfig};

use crate::key::{ChunkKey, Face};

/// Most new splits started in one update; each requests up to eight children. Bounds the work and
/// the request burst of a large camera jump, which then settles over several updates.
pub const MAX_SPLIT_STARTS_PER_UPDATE: usize = 64;

/// Most merges committed in one update; each hides and releases up to eight children. At 64 a
/// teleport across the 10 km planet stays near 0.5 ms per update while the old side unwinds.
pub const MAX_MERGES_PER_UPDATE: usize = 64;

/// Octree parameters, from [`WorldConfig`] and the planet's shell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodParams {
    /// Level 0 voxel size in metres.
    pub base_voxel_size_m: f64,
    /// Number of levels; the top level `levels - 1` holds the roots.
    pub levels: u8,
    /// A node splits when the camera is within this many of its own chunk extents of its box.
    pub radius_chunks: f64,
    /// A split node merges beyond `radius_chunks + hysteresis_chunks` extents.
    pub hysteresis_chunks: f64,
    /// Inner and outer radius of the shell around the body origin that can hold the surface at
    /// any level (from the sampler). Nodes whose box lies entirely inside the inner or outside the
    /// outer radius are never requested.
    pub shell_m: (f64, f64),
}

impl LodParams {
    pub fn new(config: &WorldConfig, shell_m: (f64, f64)) -> Self {
        Self {
            base_voxel_size_m: config.voxel_size_m,
            levels: config.lod_levels as u8,
            radius_chunks: config.lod_radius_chunks,
            hysteresis_chunks: config.lod_hysteresis_chunks,
            shell_m,
        }
    }

    /// True when the box of `key` intersects the shell: its nearest point to the body origin is
    /// within the outer radius and its farthest point beyond the inner radius.
    pub fn intersects_shell(&self, key: ChunkKey) -> bool {
        let (min, max) = key.bounds_m(self.base_voxel_size_m);
        let nearest = DVec3::ZERO.clamp(min, max).length();
        let farthest = min.abs().max(max.abs()).length();
        nearest <= self.shell_m.1 && farthest >= self.shell_m.0
    }
}

/// A key the caller should make ready: generate its data and mesh it (or find it empty), then
/// report with [`LodTree::mark_ready`]. Lower priorities are more urgent (`vx_core::Priority`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodRequest {
    pub key: ChunkKey,
    pub priority: Priority,
}

/// Changes from one [`LodTree::update`], to apply in this order: start the requests, apply the
/// visibility and mask changes (a handful of chunk table writes, D23), then release.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LodUpdate {
    /// Newly needed keys, most urgent first.
    pub requests: Vec<LodRequest>,
    /// Keys no longer needed: cancel their jobs, free their data and meshes. Never a drawn key.
    pub releases: Vec<ChunkKey>,
    /// Keys whose drawn state changed this update.
    pub visibility: Vec<(ChunkKey, bool)>,
    /// Drawn keys whose transition mask changed (bit `Face as u8` set when the neighbour across
    /// that face is drawn one level finer). Every key that became visible in this update is
    /// listed too, even with a mask of 0, so a key shown again never keeps a stale mask.
    pub masks: Vec<(ChunkKey, u8)>,
}

impl LodUpdate {
    /// True when the update changes nothing.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
            && self.releases.is_empty()
            && self.visibility.is_empty()
            && self.masks.is_empty()
    }
}

/// Counters for the overlay.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LodStats {
    /// Nodes held by the tree, requested or not.
    pub nodes: usize,
    /// Drawn keys per level, index 0 the finest.
    pub drawn_per_level: Vec<usize>,
    /// Requested keys not yet reported ready.
    pub pending: usize,
    pub splits_total: u64,
    pub merges_total: u64,
    /// Main thread time of the last update.
    pub update_ms: f64,
    /// Keys requested and not released, ready or not.
    pub resident: usize,
    /// Keys reported through [`LodTree::mark_failed`].
    pub failed_total: u64,
}

const NONE: u32 = u32::MAX;
const NEVER: u64 = 0;

/// Request priority bands: `1 + standby * STANDBY_TIER + (top - level) * DISTANCE_BAND +
/// min(floor(distance_extents * DISTANCE_STEPS_PER_EXTENT), DISTANCE_BAND - 1)`.
const DISTANCE_STEPS_PER_EXTENT: f64 = 64.0;
const DISTANCE_BAND: u32 = 1 << 16;
const STANDBY_TIER: u32 = 1 << 24;

/// The 26 offsets of the face, edge and corner neighbours.
const NEIGHBOUR_OFFSETS: [IVec3; 26] = {
    let mut out = [IVec3::ZERO; 26];
    let mut n = 0;
    let mut i = 0;
    while i < 27 {
        if i != 13 {
            out[n] = IVec3::new(i % 3 - 1, (i / 3) % 3 - 1, i / 9 - 1);
            n += 1;
        }
        i += 1;
    }
    out
};

/// True when child `child` (x fastest, as [`ChunkKey::children`]) of the neighbour at `offset`
/// touches the node the offset is taken from.
fn child_touches(offset: IVec3, child: usize) -> bool {
    (0..3).all(|axis| {
        let low_half = (child >> axis) & 1 == 0;
        match offset[axis] {
            1 => low_half,
            -1 => !low_half,
            _ => true,
        }
    })
}

/// Fx style hasher for chunk keys: the index is only looked up, never iterated.
#[derive(Default)]
struct KeyHasher(u64);

impl KeyHasher {
    const SEED: u64 = 0x517c_c1b7_2722_0a95;

    fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(Self::SEED);
    }
}

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        // The multiply leaves the entropy in the high bits; fold it into the bucket bits.
        self.0 ^ (self.0 >> 32)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(u64::from(b));
        }
    }

    fn write_u8(&mut self, v: u8) {
        self.add(u64::from(v));
    }

    fn write_i32(&mut self, v: i32) {
        self.add(u64::from(v as u32));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Staged,
    Leaf,
    Split,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Unrequested,
    Pending,
    Ready,
}

#[derive(Debug, Clone)]
struct Node {
    key: ChunkKey,
    parent: u32,
    /// Slots of the in shell children, `NONE` where the child lies outside the shell. Valid for a
    /// split node and for a leaf with a pending split.
    children: [u32; 8],
    role: Role,
    status: Status,
    pending_split: bool,
    child_count: u8,
    ready_children: u8,
    drawn_children: u8,
    /// Last reported transition mask while a leaf.
    mask: u8,
    /// Update in which a finer split last needed this leaf to split (balance).
    forced_at: u64,
    requested_at: u64,
}

impl Node {
    fn new(key: ChunkKey, parent: u32) -> Self {
        Self {
            key,
            parent,
            children: [NONE; 8],
            role: Role::Staged,
            status: Status::Unrequested,
            pending_split: false,
            child_count: 0,
            ready_children: 0,
            drawn_children: 0,
            mask: 0,
            forced_at: NEVER,
            requested_at: NEVER,
        }
    }

    fn children(&self) -> impl Iterator<Item = u32> + '_ {
        self.children.iter().copied().filter(|&c| c != NONE)
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    slot: u32,
    key: ChunkKey,
    distance_extents: f64,
}

fn coarse_near_first(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.key
        .level
        .cmp(&a.key.level)
        .then(a.distance_extents.total_cmp(&b.distance_extents))
        .then(a.key.cmp(&b.key))
}

fn fine_far_first(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    a.key
        .level
        .cmp(&b.key.level)
        .then(b.distance_extents.total_cmp(&a.distance_extents))
        .then(a.key.cmp(&b.key))
}

pub struct LodTree {
    params: LodParams,
    top_level: u8,
    nodes: Vec<Option<Node>>,
    free: Vec<u32>,
    index: HashMap<ChunkKey, u32, BuildHasherDefault<KeyHasher>>,
    roots: Vec<u32>,
    roots_requested: bool,
    roots_ready: usize,
    roots_committed: bool,
    /// Update counter, starting at 1 so `NEVER` (0) is never a real update.
    now: u64,
    camera_m: DVec3,
    pending: usize,
    resident: usize,
    drawn_per_level: Vec<usize>,
    splits_total: u64,
    merges_total: u64,
    failed_total: u64,
    /// Splits started only to keep 2:1 balance.
    forced_total: u64,
    update_ms: f64,
    settled: bool,
    out: LodUpdate,
    mask_dirty: Vec<ChunkKey>,
    residency_dirty: Vec<ChunkKey>,
}

impl LodTree {
    pub fn new(params: LodParams) -> Self {
        let levels = params.levels.clamp(1, 31);
        let top_level = levels - 1;
        let mut tree = Self {
            params,
            top_level,
            nodes: Vec::new(),
            free: Vec::new(),
            index: HashMap::default(),
            roots: Vec::new(),
            roots_requested: false,
            roots_ready: 0,
            roots_committed: false,
            now: 1,
            camera_m: DVec3::ZERO,
            pending: 0,
            resident: 0,
            drawn_per_level: vec![0; usize::from(levels)],
            splits_total: 0,
            merges_total: 0,
            failed_total: 0,
            forced_total: 0,
            update_ms: 0.0,
            settled: false,
            out: LodUpdate::default(),
            mask_dirty: Vec::new(),
            residency_dirty: Vec::new(),
        };
        let extent = ChunkKey::at_level(top_level, 0, 0, 0).extent_m(params.base_voxel_size_m);
        let outer = params.shell_m.1;
        if outer.is_finite() && extent > 0.0 {
            let lo = (-outer / extent).floor() as i32;
            let hi = (outer / extent).floor() as i32;
            for x in lo..=hi {
                for y in lo..=hi {
                    for z in lo..=hi {
                        let key = ChunkKey::at_level(top_level, x, y, z);
                        if params.intersects_shell(key) {
                            let slot = tree.alloc(Node::new(key, NONE));
                            tree.roots.push(slot);
                        }
                    }
                }
            }
        }
        tree
    }

    pub fn params(&self) -> &LodParams {
        &self.params
    }

    /// Advances the tree towards the structure the camera at `camera_m` (body fixed metres) wants.
    /// Does bounded work per call, so a large camera jump settles over several frames.
    ///
    /// Requests are sorted most urgent first. The priority of a key at `level`, whose box lies
    /// `d` of its own extents from the camera, is
    /// `1 + standby * 2^24 + (top_level - level) * 2^16 + min(floor(64 d), 2^16 - 1)`: always
    /// after `Priority::EDIT`, coarse before fine (a coarse split unblocks the finer ones and a
    /// root unblocks the first frame), then nearest first in the key's own extents. `standby` is
    /// 1 for a split parent requested again only so that a later merge is instant, 0 otherwise.
    pub fn update(&mut self, camera_m: DVec3) -> LodUpdate {
        profiling::scope!("LodTree::update");
        let start = Instant::now();
        self.now += 1;
        self.camera_m = camera_m;
        if !self.roots_requested {
            self.roots_requested = true;
            for i in 0..self.roots.len() {
                self.request(self.roots[i], false);
            }
        }
        if !self.roots_committed && self.roots_ready == self.roots.len() {
            self.roots_committed = true;
            for i in 0..self.roots.len() {
                self.show(self.roots[i]);
            }
        }
        if self.roots_committed {
            self.advance();
        }
        self.settle_residency();
        let out = self.finish_output();
        self.settled = out.is_empty() && self.pending == 0;
        self.update_ms = start.elapsed().as_secs_f64() * 1.0e3;
        out
    }

    /// The caller finished a requested key: its mesh is uploaded (hidden) or it has none. A key
    /// that is not pending (released, never requested, or already reported) is ignored, so a job
    /// that completes after its release is harmless. The caller reports only the result of the
    /// latest request of a key.
    pub fn mark_ready(&mut self, key: ChunkKey) {
        let Some(&slot) = self.index.get(&key) else {
            return;
        };
        let node = self.node_mut(slot);
        if node.status != Status::Pending {
            return;
        }
        node.status = Status::Ready;
        let (role, parent) = (node.role, node.parent);
        self.pending -= 1;
        if role == Role::Staged {
            if parent == NONE {
                self.roots_ready += 1;
            } else {
                self.node_mut(parent).ready_children += 1;
            }
        }
    }

    /// A requested key could not be made ready (for example a full GPU pool); the tree treats it
    /// as ready and empty so the octree never stalls, and counts it.
    pub fn mark_failed(&mut self, key: ChunkKey) {
        let pending = self
            .index
            .get(&key)
            .is_some_and(|&s| self.node(s).status == Status::Pending);
        if pending {
            self.failed_total += 1;
            self.mark_ready(key);
        }
    }

    pub fn is_drawn(&self, key: ChunkKey) -> bool {
        self.index
            .get(&key)
            .is_some_and(|&s| self.node(s).role == Role::Leaf)
    }

    /// Transition mask of a drawn key, 0 otherwise.
    pub fn mask(&self, key: ChunkKey) -> u8 {
        match self.index.get(&key) {
            Some(&s) if self.node(s).role == Role::Leaf => self.node(s).mask,
            _ => 0,
        }
    }

    /// Every drawn key with its mask, sorted by key.
    pub fn drawn(&self) -> Vec<(ChunkKey, u8)> {
        let mut drawn: Vec<_> = self
            .nodes
            .iter()
            .flatten()
            .filter(|n| n.role == Role::Leaf)
            .map(|n| (n.key, n.mask))
            .collect();
        drawn.sort_unstable_by_key(|d| d.0);
        drawn
    }

    pub fn stats(&self) -> LodStats {
        LodStats {
            nodes: self.index.len(),
            drawn_per_level: self.drawn_per_level.clone(),
            pending: self.pending,
            splits_total: self.splits_total,
            merges_total: self.merges_total,
            update_ms: self.update_ms,
            resident: self.resident,
            failed_total: self.failed_total,
        }
    }

    /// True when the last update changed nothing and no request is outstanding: the tree has
    /// reached the structure the camera wants (until the camera moves).
    pub fn is_settled(&self) -> bool {
        self.settled
    }

    /// Slot indices held by the tree (parents, children, roots, the index) always point at live
    /// nodes: a node is freed only after its parent's child slot is cleared or the parent merged.
    fn node(&self, slot: u32) -> &Node {
        self.nodes[slot as usize]
            .as_ref()
            .expect("slot indices held by the tree point at live nodes")
    }

    fn node_mut(&mut self, slot: u32) -> &mut Node {
        self.nodes[slot as usize]
            .as_mut()
            .expect("slot indices held by the tree point at live nodes")
    }

    /// The node at `slot` if it is still the one recorded for `key`.
    fn live(&self, slot: u32, key: ChunkKey) -> Option<&Node> {
        self.nodes[slot as usize].as_ref().filter(|n| n.key == key)
    }

    fn alloc(&mut self, node: Node) -> u32 {
        let key = node.key;
        let slot = match self.free.pop() {
            Some(slot) => {
                self.nodes[slot as usize] = Some(node);
                slot
            }
            None => {
                self.nodes.push(Some(node));
                (self.nodes.len() - 1) as u32
            }
        };
        let previous = self.index.insert(key, slot);
        debug_assert!(previous.is_none(), "{key:?} stored twice");
        slot
    }

    fn remove(&mut self, slot: u32) {
        self.release(slot);
        let key = self.node(slot).key;
        self.index.remove(&key);
        self.nodes[slot as usize] = None;
        self.free.push(slot);
    }

    fn extent_m(&self, level: u8) -> f64 {
        ChunkKey::at_level(level, 0, 0, 0).extent_m(self.params.base_voxel_size_m)
    }

    /// Distance from the camera to the box of `key`, in extents of the key's level.
    fn distance_extents(&self, key: ChunkKey) -> f64 {
        let (min, max) = key.bounds_m(self.params.base_voxel_size_m);
        let p = self.camera_m;
        let d = (min - p).max(p - max).max(DVec3::ZERO);
        d.length() / self.extent_m(key.level)
    }

    fn merge_radius(&self) -> f64 {
        self.params.radius_chunks + self.params.hysteresis_chunks
    }

    fn priority(&self, key: ChunkKey, standby: bool) -> Priority {
        let rank = u32::from(self.top_level - key.level);
        let steps = (self.distance_extents(key) * DISTANCE_STEPS_PER_EXTENT)
            .min(f64::from(DISTANCE_BAND - 1)) as u32;
        Priority(1 + u32::from(standby) * STANDBY_TIER + rank * DISTANCE_BAND + steps)
    }

    fn request(&mut self, slot: u32, standby: bool) {
        let key = self.node(slot).key;
        let priority = self.priority(key, standby);
        let now = self.now;
        let node = self.node_mut(slot);
        debug_assert_eq!(node.status, Status::Unrequested);
        node.status = Status::Pending;
        node.requested_at = now;
        self.pending += 1;
        self.resident += 1;
        self.out.requests.push(LodRequest { key, priority });
    }

    fn release(&mut self, slot: u32) {
        let now = self.now;
        let node = self.node_mut(slot);
        let (key, status, requested_at) = (node.key, node.status, node.requested_at);
        if status == Status::Unrequested {
            return;
        }
        node.status = Status::Unrequested;
        if status == Status::Pending {
            self.pending -= 1;
        }
        self.resident -= 1;
        if requested_at == now {
            self.out.requests.retain(|r| r.key != key);
        } else {
            self.out.releases.push(key);
        }
    }

    fn show(&mut self, slot: u32) {
        let node = self.node_mut(slot);
        debug_assert_ne!(node.role, Role::Leaf);
        node.role = Role::Leaf;
        let key = node.key;
        self.drawn_per_level[usize::from(key.level)] += 1;
        self.out.visibility.push((key, true));
        self.mask_dirty.push(key);
    }

    /// Hides a leaf; the caller sets the new role or removes the node.
    fn hide(&mut self, slot: u32) {
        let node = self.node_mut(slot);
        debug_assert_eq!(node.role, Role::Leaf);
        node.mask = 0;
        let key = node.key;
        self.drawn_per_level[usize::from(key.level)] -= 1;
        self.out.visibility.push((key, false));
    }

    fn mark_face_neighbours_dirty(&mut self, key: ChunkKey) {
        for face in Face::ALL {
            self.mask_dirty.push(key.neighbour(face));
        }
    }

    fn forced_recently(&self, node: &Node) -> bool {
        node.forced_at != NEVER && node.forced_at + 1 >= self.now
    }

    fn advance(&mut self) {
        let radius = self.params.radius_chunks;
        let merge_radius = self.merge_radius();
        let mut commits = Vec::new();
        let mut cancels = Vec::new();
        let mut merges = Vec::new();
        let mut starts = Vec::new();
        for (slot, node) in self.nodes.iter().enumerate() {
            let Some(node) = node else { continue };
            if node.role == Role::Staged {
                continue;
            }
            let candidate = Candidate {
                slot: slot as u32,
                key: node.key,
                distance_extents: self.distance_extents(node.key),
            };
            let d = candidate.distance_extents;
            match node.role {
                Role::Leaf if node.pending_split => {
                    if node.ready_children == node.child_count {
                        commits.push(candidate);
                    }
                    if d > merge_radius {
                        cancels.push(candidate);
                    }
                }
                Role::Leaf => {
                    if node.key.level > 0 && d < radius {
                        starts.push(candidate);
                    }
                }
                Role::Split => {
                    if d > merge_radius && node.drawn_children == node.child_count {
                        merges.push(candidate);
                    }
                }
                Role::Staged => {}
            }
        }

        commits.sort_unstable_by(coarse_near_first);
        for c in &commits {
            self.try_commit_split(c);
        }

        for c in &cancels {
            let Some(node) = self.live(c.slot, c.key) else {
                continue;
            };
            if node.role == Role::Leaf && node.pending_split && !self.forced_recently(node) {
                self.cancel_split(c.slot);
            }
        }

        merges.sort_unstable_by(fine_far_first);
        let mut merged = 0;
        for c in &merges {
            if merged == MAX_MERGES_PER_UPDATE {
                break;
            }
            if self.try_merge(c) {
                merged += 1;
            }
        }

        starts.sort_unstable_by(coarse_near_first);
        let mut started = 0;
        for c in &starts {
            if started == MAX_SPLIT_STARTS_PER_UPDATE {
                break;
            }
            let ok = self
                .live(c.slot, c.key)
                .is_some_and(|n| n.role == Role::Leaf && !n.pending_split);
            if ok {
                self.start_split(c.slot);
                started += 1;
            }
        }
    }

    fn start_split(&mut self, slot: u32) {
        let key = self.node(slot).key;
        let Some(children) = key.children() else {
            return;
        };
        let mut slots = [NONE; 8];
        let mut count = 0;
        for (i, child) in children.into_iter().enumerate() {
            if !self.params.intersects_shell(child) {
                continue;
            }
            let s = self.alloc(Node::new(child, slot));
            slots[i] = s;
            count += 1;
            self.request(s, false);
        }
        let node = self.node_mut(slot);
        node.children = slots;
        node.child_count = count;
        node.ready_children = 0;
        node.pending_split = true;
    }

    fn cancel_split(&mut self, slot: u32) {
        let children = self.node(slot).children;
        for c in children.into_iter().filter(|&c| c != NONE) {
            self.remove(c);
        }
        let node = self.node_mut(slot);
        node.children = [NONE; 8];
        node.child_count = 0;
        node.ready_children = 0;
        node.pending_split = false;
        node.forced_at = NEVER;
    }

    fn try_commit_split(&mut self, c: &Candidate) {
        let Some(node) = self.live(c.slot, c.key) else {
            return;
        };
        if node.role != Role::Leaf || !node.pending_split || node.ready_children != node.child_count
        {
            return;
        }
        if c.distance_extents > self.merge_radius() && !self.forced_recently(node) {
            return;
        }
        let blockers = self.split_blockers(c.key);
        if blockers.is_empty() {
            self.commit_split(c.slot);
            return;
        }
        let now = self.now;
        for b in blockers {
            self.node_mut(b).forced_at = now;
            if !self.node(b).pending_split {
                self.forced_total += 1;
                self.start_split(b);
            }
        }
    }

    /// Committed leaves two or more levels coarser than the children of `key` that would touch
    /// them: each must split before `key` may.
    fn split_blockers(&self, key: ChunkKey) -> Vec<u32> {
        let mut blockers = Vec::new();
        for o in NEIGHBOUR_OFFSETS {
            if let Some(b) = self.covering_coarser_leaf(key.offset(o.x, o.y, o.z))
                && !blockers.contains(&b)
            {
                blockers.push(b);
            }
        }
        blockers
    }

    /// The committed leaf coarser than `q` that covers the box of `q`, if any. A region under a
    /// split node with no stored descendant lies outside the shell and holds nothing.
    fn covering_coarser_leaf(&self, q: ChunkKey) -> Option<u32> {
        let mut a = q;
        loop {
            if let Some(&s) = self.index.get(&a) {
                let n = self.node(s);
                return match n.role {
                    Role::Split => None,
                    Role::Leaf => (a.level > q.level).then_some(s),
                    Role::Staged => (n.parent != NONE).then_some(n.parent),
                };
            }
            if a.level >= self.top_level {
                return None;
            }
            a = a.parent();
        }
    }

    fn commit_split(&mut self, slot: u32) {
        self.hide(slot);
        let node = self.node_mut(slot);
        node.role = Role::Split;
        node.pending_split = false;
        node.forced_at = NEVER;
        node.drawn_children = node.child_count;
        let (key, parent, children) = (node.key, node.parent, node.children);
        for c in children.into_iter().filter(|&c| c != NONE) {
            self.show(c);
        }
        self.mark_face_neighbours_dirty(key);
        if parent != NONE {
            let p = self.node_mut(parent);
            p.drawn_children -= 1;
            let pkey = p.key;
            self.residency_dirty.push(pkey);
        }
        self.splits_total += 1;
    }

    /// Merges the split node of `c` when it is allowed; true when it merged.
    fn try_merge(&mut self, c: &Candidate) -> bool {
        let Some(node) = self.live(c.slot, c.key) else {
            return false;
        };
        if node.role != Role::Split
            || node.status != Status::Ready
            || node.drawn_children != node.child_count
        {
            return false;
        }
        let children_settled = node.children().all(|ch| {
            let n = self.node(ch);
            n.role == Role::Leaf && !n.pending_split
        });
        if !children_settled || self.merge_refused(c.key) {
            return false;
        }
        self.commit_merge(c.slot);
        true
    }

    /// True when merging `key` would put it next to a leaf two levels finer: a same level
    /// neighbour has a child touching `key` that is split or has a pending split.
    fn merge_refused(&self, key: ChunkKey) -> bool {
        NEIGHBOUR_OFFSETS.iter().any(|o| {
            let Some(&s) = self.index.get(&key.offset(o.x, o.y, o.z)) else {
                return false;
            };
            let q = self.node(s);
            q.role == Role::Split
                && (0..8).any(|i| {
                    let c = q.children[i];
                    c != NONE && child_touches(*o, i) && {
                        let cn = self.node(c);
                        cn.role == Role::Split || cn.pending_split
                    }
                })
        })
    }

    fn commit_merge(&mut self, slot: u32) {
        let children = self.node(slot).children;
        for c in children.into_iter().filter(|&c| c != NONE) {
            self.hide(c);
            self.remove(c);
        }
        let node = self.node_mut(slot);
        node.children = [NONE; 8];
        node.child_count = 0;
        node.ready_children = 0;
        node.drawn_children = 0;
        let (key, parent) = (node.key, node.parent);
        self.show(slot);
        self.mark_face_neighbours_dirty(key);
        if parent != NONE {
            let p = self.node_mut(parent);
            p.drawn_children += 1;
            let pkey = p.key;
            self.residency_dirty.push(pkey);
        }
        self.merges_total += 1;
    }

    /// Requests split nodes that became parents of drawn leaves only and releases those that
    /// became grandparents.
    fn settle_residency(&mut self) {
        let mut dirty = std::mem::take(&mut self.residency_dirty);
        dirty.sort_unstable();
        dirty.dedup();
        let merge_radius = self.merge_radius();
        for &key in &dirty {
            let Some(&slot) = self.index.get(&key) else {
                continue;
            };
            let node = self.node(slot);
            if node.role != Role::Split {
                continue;
            }
            let wanted = node.drawn_children == node.child_count;
            match (wanted, node.status) {
                (true, Status::Unrequested) => {
                    let standby = self.distance_extents(key) <= merge_radius;
                    self.request(slot, standby);
                }
                (false, Status::Pending | Status::Ready) => self.release(slot),
                _ => {}
            }
        }
        dirty.clear();
        self.residency_dirty = dirty;
    }

    fn compute_mask(&self, key: ChunkKey) -> u8 {
        let mut mask = 0;
        for face in Face::ALL {
            let Some(&s) = self.index.get(&key.neighbour(face)) else {
                continue;
            };
            let q = self.node(s);
            if q.role != Role::Split {
                continue;
            }
            // Children of the neighbour on the side facing `key`.
            let axis = face.axis();
            let facing_bit = usize::from(!face.is_high());
            let finer = (0..8).any(|i| (i >> axis) & 1 == facing_bit && q.children[i] != NONE);
            if finer {
                mask |= face.bit();
            }
        }
        mask
    }

    fn finish_output(&mut self) -> LodUpdate {
        let mut out = std::mem::take(&mut self.out);

        // Net visibility: a key that flipped twice within the update is not reported.
        let mut changes = std::mem::take(&mut out.visibility);
        changes.sort_by_key(|c| c.0);
        let mut i = 0;
        while i < changes.len() {
            let (key, first) = changes[i];
            let mut j = i;
            while j + 1 < changes.len() && changes[j + 1].0 == key {
                j += 1;
            }
            // The state before the update is `!first`; it changed when the last state differs.
            if changes[j].1 == first {
                out.visibility.push((key, first));
            }
            i = j + 1;
        }

        let mut dirty = std::mem::take(&mut self.mask_dirty);
        dirty.sort_unstable();
        dirty.dedup();
        for &key in &dirty {
            let Some(&slot) = self.index.get(&key) else {
                continue;
            };
            if self.node(slot).role != Role::Leaf {
                continue;
            }
            let mask = self.compute_mask(key);
            let shown = out
                .visibility
                .binary_search_by_key(&key, |v| v.0)
                .is_ok_and(|i| out.visibility[i].1);
            let node = self.node_mut(slot);
            if shown || node.mask != mask {
                node.mask = mask;
                out.masks.push((key, mask));
            }
        }
        dirty.clear();
        self.mask_dirty = dirty;

        out.requests
            .sort_unstable_by(|a, b| a.priority.cmp(&b.priority).then(a.key.cmp(&b.key)));
        out.releases.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap, HashSet};

    /// Design defaults: 1 m voxels, 8 levels, radius 4, hysteresis 1, 10 km planet.
    fn defaults_10km() -> LodParams {
        LodParams {
            base_voxel_size_m: 1.0,
            levels: 8,
            radius_chunks: 4.0,
            hysteresis_chunks: 1.0,
            shell_m: (9_800.0, 10_200.0),
        }
    }

    /// The 50 km parameter run: 0.5 m voxels, 10 levels.
    fn params_50km() -> LodParams {
        LodParams {
            base_voxel_size_m: 0.5,
            levels: 10,
            radius_chunks: 4.0,
            hysteresis_chunks: 1.0,
            shell_m: (49_500.0, 50_500.0),
        }
    }

    /// Splitmix64.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn direction(&mut self) -> DVec3 {
            loop {
                let v = DVec3::new(self.unit(), self.unit(), self.unit()) * 2.0 - 1.0;
                let l = v.length();
                if l > 1.0e-3 && l <= 1.0 {
                    return v / l;
                }
            }
        }
    }

    fn top_keys_in_shell(p: &LodParams) -> Vec<ChunkKey> {
        let top = p.levels - 1;
        let e = ChunkKey::at_level(top, 0, 0, 0).extent_m(p.base_voxel_size_m);
        let n = (p.shell_m.1 / e).ceil() as i32 + 1;
        let mut keys = Vec::new();
        for x in -n..=n {
            for y in -n..=n {
                for z in -n..=n {
                    let k = ChunkKey::at_level(top, x, y, z);
                    if p.intersects_shell(k) {
                        keys.push(k);
                    }
                }
            }
        }
        keys
    }

    fn containing(p: &LodParams, point: DVec3, level: u8) -> ChunkKey {
        let e = ChunkKey::at_level(level, 0, 0, 0).extent_m(p.base_voxel_size_m);
        let f = |v: f64| (v / e).floor() as i32;
        ChunkKey::at_level(level, f(point.x), f(point.y), f(point.z))
    }

    /// A caller with a job queue: keys complete in priority order, a few updates after they were
    /// requested, a bounded number per update; some fail and some released jobs complete late.
    struct Sim {
        tree: LodTree,
        params: LodParams,
        roots: Vec<ChunkKey>,
        rng: Rng,
        step: u64,
        max_latency: u64,
        throughput: usize,
        fail_one_in: u64,
        queue: BTreeMap<(Priority, u64), (ChunkKey, u64)>,
        queued: HashMap<ChunkKey, (Priority, u64)>,
        seq: u64,
        resident: HashSet<ChunkKey>,
        ready: HashSet<ChunkKey>,
        drawn: HashMap<ChunkKey, u8>,
        late: Vec<ChunkKey>,
        ever_drawn: bool,
        failed: u64,
        checks: bool,
        check_every: u64,
        sample_points: usize,
        update_ms: Vec<f64>,
    }

    impl Sim {
        fn new(params: LodParams, checks: bool) -> Self {
            Self {
                tree: LodTree::new(params),
                params,
                roots: top_keys_in_shell(&params),
                rng: Rng(7),
                step: 0,
                max_latency: 4,
                throughput: 400,
                fail_one_in: 29,
                queue: BTreeMap::new(),
                queued: HashMap::new(),
                seq: 0,
                resident: HashSet::new(),
                ready: HashSet::new(),
                drawn: HashMap::new(),
                late: Vec::new(),
                ever_drawn: false,
                failed: 0,
                checks,
                check_every: 1,
                sample_points: 400,
                update_ms: Vec::new(),
            }
        }

        fn complete_jobs(&mut self) {
            let step = self.step;
            let due: Vec<_> = self
                .queue
                .iter()
                .filter(|(_, (_, at))| *at <= step)
                .take(self.throughput)
                .map(|(k, v)| (*k, v.0))
                .collect();
            for (qk, key) in due {
                self.queue.remove(&qk);
                self.queued.remove(&key);
                self.ready.insert(key);
                if self.rng.next().is_multiple_of(self.fail_one_in) {
                    self.failed += 1;
                    self.tree.mark_failed(key);
                } else {
                    self.tree.mark_ready(key);
                }
            }
            // Jobs released while in flight finish anyway; the tree ignores them.
            for key in std::mem::take(&mut self.late) {
                if !self.resident.contains(&key) {
                    let before = self.tree.stats();
                    self.tree.mark_ready(key);
                    self.tree.mark_failed(key);
                    assert_eq!(
                        self.tree.stats(),
                        before,
                        "late result of {key:?} changed the tree"
                    );
                }
            }
        }

        fn step(&mut self, camera: DVec3) -> LodUpdate {
            self.step += 1;
            self.complete_jobs();
            let update = self.tree.update(camera);
            self.update_ms.push(self.tree.stats().update_ms);
            self.apply(&update);
            if self.checks && self.step.is_multiple_of(self.check_every) {
                self.check_structure(camera);
            }
            update
        }

        fn apply(&mut self, u: &LodUpdate) {
            let requested: HashSet<ChunkKey> = u.requests.iter().map(|r| r.key).collect();
            assert_eq!(requested.len(), u.requests.len(), "duplicate request");
            for w in u.requests.windows(2) {
                assert!(
                    w[0].priority <= w[1].priority,
                    "requests not most urgent first"
                );
            }
            for r in &u.requests {
                assert!(r.priority > Priority::EDIT);
                assert!(
                    self.params.intersects_shell(r.key),
                    "request outside the shell: {:?}",
                    r.key
                );
                assert!(self.resident.insert(r.key), "{:?} requested twice", r.key);
                let latency = 1 + self.rng.next() % self.max_latency;
                self.seq += 1;
                let qk = (r.priority, self.seq);
                self.queue.insert(qk, (r.key, self.step + latency));
                self.queued.insert(r.key, qk);
            }
            for &(key, visible) in &u.visibility {
                if visible {
                    assert!(
                        self.ready.contains(&key),
                        "{key:?} shown before it is ready"
                    );
                    assert!(self.resident.contains(&key));
                    assert!(self.drawn.insert(key, 0).is_none(), "{key:?} shown twice");
                    assert!(
                        u.masks.iter().any(|m| m.0 == key),
                        "{key:?} shown without its mask"
                    );
                } else {
                    assert!(self.drawn.remove(&key).is_some(), "{key:?} hidden twice");
                }
            }
            for &(key, mask) in &u.masks {
                let m = self
                    .drawn
                    .get_mut(&key)
                    .expect("mask of a key that is not drawn");
                *m = mask;
            }
            for &key in &u.releases {
                assert!(!requested.contains(&key), "{key:?} requested and released");
                assert!(!self.drawn.contains_key(&key), "drawn key {key:?} released");
                assert!(
                    self.resident.remove(&key),
                    "{key:?} released but not resident"
                );
                self.ready.remove(&key);
                if let Some(qk) = self.queued.remove(&key) {
                    self.queue.remove(&qk);
                    if self.rng.next().is_multiple_of(2) {
                        self.late.push(key);
                    }
                }
            }
            if !self.drawn.is_empty() {
                self.ever_drawn = true;
            }
            for &(key, _) in u.visibility.iter().filter(|v| v.1) {
                assert!(self.ready.contains(&key) && self.resident.contains(&key));
            }

            let s = self.tree.stats();
            assert_eq!(s.drawn_per_level.iter().sum::<usize>(), self.drawn.len());
            if self.checks && self.step.is_multiple_of(self.check_every) {
                let mut replay: Vec<_> = self.drawn.iter().map(|(&k, &m)| (k, m)).collect();
                replay.sort_unstable();
                assert_eq!(
                    replay,
                    self.tree.drawn(),
                    "replayed deltas differ from the tree"
                );
                let mut per_level = vec![0; s.drawn_per_level.len()];
                for k in self.drawn.keys() {
                    per_level[usize::from(k.level)] += 1;
                }
                assert_eq!(per_level, s.drawn_per_level);
            }
            assert_eq!(s.resident, self.resident.len());
            assert_eq!(s.pending, self.resident.len() - self.ready.len());
            assert_eq!(s.failed_total, self.failed);
            assert!(s.nodes >= s.resident);
        }

        fn check_structure(&mut self, camera: DVec3) {
            let p = self.params;
            let top = p.levels - 1;
            if self.drawn.is_empty() {
                assert!(
                    !self.ever_drawn,
                    "the drawn set emptied after the first commit"
                );
                return;
            }

            // Exact partition: no drawn key has a drawn ancestor, and every in shell root is
            // covered by drawn descendants.
            let mut ancestors = HashSet::new();
            for &k in self.drawn.keys() {
                let mut a = k;
                while a.level < top {
                    a = a.parent();
                    assert!(!self.drawn.contains_key(&a), "{k:?} overlaps {a:?}");
                    if !ancestors.insert(a) {
                        break;
                    }
                }
            }
            fn covered(
                k: ChunkKey,
                p: &LodParams,
                drawn: &HashMap<ChunkKey, u8>,
                ancestors: &HashSet<ChunkKey>,
            ) -> bool {
                drawn.contains_key(&k)
                    || !p.intersects_shell(k)
                    || (ancestors.contains(&k)
                        && k.children()
                            .is_some_and(|c| c.iter().all(|&c| covered(c, p, drawn, ancestors))))
            }
            for &root in &self.roots {
                assert!(
                    covered(root, &p, &self.drawn, &ancestors),
                    "hole under {root:?}"
                );
            }

            // Sampled points: half anywhere in the shell, half near the camera on log scales.
            let (inner, outer) = p.shell_m;
            for i in 0..self.sample_points {
                let point = if i % 2 == 0 {
                    self.rng.direction() * (inner + (outer - inner) * self.rng.unit())
                } else {
                    let near = camera.normalize_or(DVec3::Y) * camera.length().clamp(inner, outer);
                    let dist = 10f64.powf(self.rng.unit() * 4.5);
                    let q = near + self.rng.direction() * dist;
                    let r = q.length();
                    if (inner..=outer).contains(&r) {
                        q
                    } else {
                        q / r * (inner + (outer - inner) * self.rng.unit())
                    }
                };
                let hits = (0..p.levels)
                    .filter(|&l| self.drawn.contains_key(&containing(&p, point, l)))
                    .count();
                assert_eq!(hits, 1, "point {point} covered {hits} times");
            }

            // 2:1 balance over faces, edges and corners.
            for &k in self.drawn.keys() {
                for o in NEIGHBOUR_OFFSETS {
                    let q = k.offset(o.x, o.y, o.z);
                    if self.drawn.contains_key(&q) || q.level >= top {
                        continue;
                    }
                    let mut a = q.parent();
                    if self.drawn.contains_key(&a) {
                        continue;
                    }
                    while a.level < top {
                        a = a.parent();
                        assert!(
                            !self.drawn.contains_key(&a),
                            "{k:?} touches {a:?}, two or more levels coarser"
                        );
                    }
                }
            }

            // Masks by brute force: a face bit is set when a drawn key one level finer lies
            // across that face.
            for (&k, &mask) in &self.drawn {
                let mut expected = 0;
                for face in Face::ALL {
                    let finer = k.neighbour(face).children().is_some_and(|c| {
                        c.iter().enumerate().any(|(i, c)| {
                            (i >> face.axis()) & 1 == usize::from(!face.is_high())
                                && self.drawn.contains_key(c)
                        })
                    });
                    if finer {
                        expected |= face.bit();
                    }
                }
                assert_eq!(mask, expected, "mask of {k:?}");
                assert_eq!(self.tree.mask(k), expected);
                assert!(self.tree.is_drawn(k));
            }
        }

        /// Steps at `camera` until the tree settles; returns the number of updates.
        fn settle(&mut self, camera: DVec3, max_updates: usize) -> usize {
            for n in 1..=max_updates {
                self.step(camera);
                if self.tree.is_settled() {
                    return n;
                }
            }
            panic!("not settled after {max_updates} updates");
        }
    }

    fn descent_dir() -> DVec3 {
        DVec3::new(0.36, 0.8, -0.48).normalize()
    }

    fn altitude(i: usize, steps: usize) -> f64 {
        100_000.0 * (2.0f64 / 100_000.0).powf(i as f64 / steps as f64)
    }

    fn describe(label: &str, tree: &LodTree) {
        let s = tree.stats();
        println!(
            "{label}: nodes {} resident {} drawn {} per level {:?}",
            s.nodes,
            s.resident,
            s.drawn_per_level.iter().sum::<usize>(),
            s.drawn_per_level
        );
    }

    fn run_descent(
        params: LodParams,
        surface_m: f64,
        steps: usize,
        check_every: u64,
        label: &str,
    ) -> u64 {
        let dir = descent_dir();
        let mut sim = Sim::new(params, true);
        sim.check_every = check_every;
        let mut first_drawn = None;
        for i in 0..=steps {
            sim.step(dir * (surface_m + altitude(i, steps)));
            if first_drawn.is_none() && sim.ever_drawn {
                first_drawn = Some(i);
            }
        }
        let roots_bound = 6 + sim.roots.len() / sim.throughput;
        assert!(
            first_drawn.is_some_and(|i| i < roots_bound),
            "roots took {first_drawn:?} updates"
        );
        let n = sim.settle(dir * (surface_m + 2.0), 400);
        describe(
            &format!("{label} surface (settled in {n} more updates)"),
            &sim.tree,
        );
        for i in (0..=steps).rev() {
            sim.step(dir * (surface_m + altitude(i, steps)));
        }
        let n = sim.settle(dir * (surface_m + 100_000.0), 400);
        describe(
            &format!("{label} 100 km (settled in {n} more updates)"),
            &sim.tree,
        );
        let s = sim.tree.stats();
        assert!(s.splits_total > 0 && s.merges_total > 0);
        assert_eq!(s.drawn_per_level[0], 0, "level 0 drawn from 100 km");
        assert_eq!(s.pending, 0);
        println!(
            "{label}: splits {} merges {} failed {} forced {}",
            s.splits_total, s.merges_total, s.failed_total, sim.tree.forced_total
        );
        sim.tree.forced_total
    }

    #[test]
    fn descent_from_100_km_partitions_the_shell_at_every_step() {
        let steps = if cfg!(debug_assertions) { 60 } else { 200 };
        run_descent(defaults_10km(), 10_000.0, steps, 1, "10 km");
    }

    #[test]
    fn descent_at_50_km_with_10_levels() {
        let steps = if cfg!(debug_assertions) { 40 } else { 200 };
        let check_every = if cfg!(debug_assertions) { 3 } else { 1 };
        run_descent(params_50km(), 50_000.0, steps, check_every, "50 km");
    }

    #[test]
    fn descent_at_parameter_range_ends_keeps_balance() {
        // The distance rule alone keeps 2:1 balance at a split when radius >= sqrt 3 and at a
        // merge when radius >= hysteresis + sqrt 3. Radius 2 with hysteresis 4 relies on refused
        // merges; radius 1 (below the supported range) relies on forced splits. Radius 16 runs on a 1 km planet with 4 m voxels so the brute force checks stay
        // quick (about 97,000 drawn keys at the 10 km defaults).
        let small = LodParams {
            base_voxel_size_m: 4.0,
            shell_m: (950.0, 1_050.0),
            ..defaults_10km()
        };
        for (radius, hysteresis, levels, base, surface_m) in [
            (2.0, 4.0, 6, defaults_10km(), 10_000.0),
            (2.0, 0.0, 8, defaults_10km(), 10_000.0),
            (1.0, 0.0, 8, defaults_10km(), 10_000.0),
            (16.0, 0.0, 4, small, 1_000.0),
            (16.0, 4.0, 5, small, 1_000.0),
        ] {
            let params = LodParams {
                radius_chunks: radius,
                hysteresis_chunks: hysteresis,
                levels,
                ..base
            };
            let forced = run_descent(
                params,
                surface_m,
                30,
                1,
                &format!("radius {radius} hysteresis {hysteresis} levels {levels}"),
            );
            if radius < 3f64.sqrt() {
                assert!(forced > 0, "balance never needed a forced split");
            }
        }
    }

    #[test]
    fn nothing_is_drawn_until_every_root_is_ready() {
        let params = defaults_10km();
        let mut tree = LodTree::new(params);
        let u = tree.update(DVec3::new(0.0, 110_000.0, 0.0));
        let roots = top_keys_in_shell(&params);
        assert_eq!(u.requests.len(), roots.len());
        assert!(u.visibility.is_empty());
        for r in &u.requests[1..] {
            tree.mark_ready(r.key);
        }
        assert!(tree.update(DVec3::new(0.0, 110_000.0, 0.0)).is_empty());
        assert!(tree.drawn().is_empty());
        tree.mark_failed(u.requests[0].key);
        let u2 = tree.update(DVec3::new(0.0, 110_000.0, 0.0));
        assert_eq!(u2.visibility.len(), roots.len());
        assert_eq!(u2.masks.len(), roots.len());
        assert!(u2.masks.iter().all(|m| m.1 == 0));
        assert_eq!(tree.stats().failed_total, 1);
        assert_eq!(tree.drawn().len(), roots.len());
    }

    #[test]
    fn unknown_and_repeated_results_are_ignored() {
        let mut tree = LodTree::new(defaults_10km());
        let before = tree.stats();
        tree.mark_ready(ChunkKey::at_level(3, 1, 2, 3));
        tree.mark_failed(ChunkKey::new(0, 0, 0));
        assert_eq!(tree.stats(), before);
        let u = tree.update(DVec3::new(0.0, 20_000.0, 0.0));
        let key = u.requests[0].key;
        tree.mark_ready(key);
        let pending = tree.stats().pending;
        tree.mark_ready(key);
        tree.mark_failed(key);
        assert_eq!(tree.stats().pending, pending);
        assert_eq!(tree.stats().failed_total, 0);
    }

    fn surface_sim(params: LodParams, surface: DVec3) -> Sim {
        let mut sim = Sim::new(params, true);
        sim.settle(surface, 600);
        sim
    }

    #[test]
    fn wobbling_camera_does_not_oscillate() {
        let dir = descent_dir();
        let tangent = dir.cross(DVec3::Y).normalize();
        let surface = dir * 10_002.0;
        let mut sim = surface_sim(defaults_10km(), surface);
        sim.sample_points = 100;
        sim.check_every = 5;
        // 40 m peak to peak, under the 64 m hysteresis band of level 1.
        let cycle = 16;
        let mut counts = Vec::new();
        for c in 0..6 {
            for i in 0..cycle {
                let phase = std::f64::consts::TAU * i as f64 / cycle as f64;
                let p = surface + tangent * 20.0 * phase.sin();
                sim.step(p.normalize() * 10_002.0);
            }
            let s = sim.tree.stats();
            counts.push((c, s.splits_total, s.merges_total));
        }
        println!("wobble (cycle, splits, merges): {counts:?}");
        assert_eq!(counts[1].1, counts[5].1, "splits keep happening");
        assert_eq!(counts[1].2, counts[5].2, "merges keep happening");

        // A radial wobble of 400 m around 4.1 km altitude crosses split radii of levels 5 and 6.
        let mut radial = Vec::new();
        for c in 0..6 {
            for i in 0..cycle {
                let phase = std::f64::consts::TAU * i as f64 / cycle as f64;
                sim.step(dir * (10_000.0 + 4_100.0 + 200.0 * phase.sin()));
            }
            let s = sim.tree.stats();
            radial.push((c, s.splits_total, s.merges_total));
        }
        println!("radial wobble (cycle, splits, merges): {radial:?}");
        assert_eq!(radial[2].1, radial[5].1, "splits keep happening");
        assert_eq!(radial[2].2, radial[5].2, "merges keep happening");
    }

    #[test]
    fn teleport_to_the_far_side_settles_with_coverage_intact() {
        let dir = descent_dir();
        let mut sim = surface_sim(defaults_10km(), dir * 10_002.0);
        sim.sample_points = 100;
        let n = sim.settle(-dir * 10_002.0, 300);
        describe(&format!("teleport settled in {n} updates"), &sim.tree);
        let s = sim.tree.stats();
        let near = sim
            .tree
            .drawn()
            .iter()
            .filter(|(k, _)| k.level == 0)
            .all(|(k, _)| k.origin_m(1.0).dot(dir) < 0.0);
        assert!(near, "level 0 chunks left on the old side");
        assert!(s.drawn_per_level[0] > 0);
    }

    #[test]
    fn identical_runs_give_identical_updates() {
        let run = || {
            let mut sim = Sim::new(defaults_10km(), false);
            let dir = descent_dir();
            let mut log = Vec::new();
            for i in 0..=80 {
                log.push(sim.step(dir * (10_000.0 + altitude(i, 80))));
            }
            for _ in 0..40 {
                log.push(sim.step(-dir * 10_002.0));
            }
            log
        };
        assert!(run() == run());
    }

    /// Update time at the design defaults; meaningful with `cargo test --release`.
    #[test]
    fn update_time_at_the_defaults() {
        measure_update_time(defaults_10km(), 10_000.0, "10 km");
    }

    /// Update time of the 50 km, 0.5 m run.
    #[test]
    fn update_time_at_50_km() {
        measure_update_time(params_50km(), 50_000.0, "50 km");
    }

    fn measure_update_time(params: LodParams, radius_m: f64, label: &str) {
        let r = radius_m + 2.0;
        let dir = descent_dir();
        let tangent = dir.cross(DVec3::Y).normalize();
        let mut sim = Sim::new(params, false);
        let quick = cfg!(debug_assertions);
        let steps = if quick { 60 } else { 300 };
        for i in 0..=steps {
            sim.step(dir * (radius_m + altitude(i, steps)));
        }
        sim.settle(dir * r, 600);
        let descent: Vec<f64> = std::mem::take(&mut sim.update_ms);
        for _ in 0..if quick { 10 } else { 120 } {
            sim.step(dir * r);
        }
        let standing: Vec<f64> = std::mem::take(&mut sim.update_ms);
        // 100 m/s at 60 updates per second along the surface.
        for i in 0..if quick { 60 } else { 1200 } {
            let p = dir * r + tangent * (i as f64 * 100.0 / 60.0);
            sim.step(p.normalize() * r);
        }
        let walking: Vec<f64> = std::mem::take(&mut sim.update_ms);
        sim.settle(-dir * r, 600);
        let teleport: Vec<f64> = std::mem::take(&mut sim.update_ms);
        let summary = |v: &[f64]| {
            let mut sorted = v.to_vec();
            sorted.sort_by(f64::total_cmp);
            let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            (mean, at(0.5), at(0.99), at(1.0))
        };
        let build = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        for (phase, v) in [
            ("descent", &descent),
            ("standing", &standing),
            ("walking 100 m/s", &walking),
            ("teleport", &teleport),
        ] {
            let (mean, median, p99, max) = summary(v);
            println!(
                "LOD update {build} {label} {phase} ({} updates): mean {mean:.4} median {median:.4} \
                 p99 {p99:.4} max {max:.4} ms",
                v.len()
            );
        }
        describe(&format!("{label} after teleport"), &sim.tree);
        // Medians, because other processes preempting the test show up as rare spikes.
        if !cfg!(debug_assertions) {
            assert!(summary(&standing).1 < 0.5);
            assert!(summary(&walking).1 < 0.5);
            assert!(summary(&descent).1 < 0.5);
        }
    }
}
