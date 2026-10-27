//! Job system (D9, D12): a fixed pool of worker threads draining one priority queue.
//!
//! Jobs carry a [`Priority`] (lower runs first, FIFO within a priority), a [`Stage`] label for
//! the per stage counters shown in the debug overlay, and a [`CancelToken`]. A queued job whose
//! token is cancelled before a worker picks it up never runs; a running job polls its own token.
//! Results travel on channels owned by the caller. Workers may spawn jobs, so one pipeline stage
//! can queue the next without a round trip through the main thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Index of a stage in the names passed to [`JobSystem::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stage(pub usize);

/// Scheduling priority: lower values run first. Equal priorities run in submission order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u32);

impl Priority {
    /// Player edits run before any streaming work (D9).
    pub const EDIT: Priority = Priority(0);

    /// Streaming priority from the squared distance to the camera in chunks: nearer chunks first,
    /// always after edits.
    pub fn streaming(distance_sq_chunks: u64) -> Self {
        Self(u32::try_from(distance_sq_chunks.saturating_add(1)).unwrap_or(u32::MAX))
    }
}

/// Shared flag that tells a queued or running job to stop.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Counters for one stage, read by the overlay every frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StageStats {
    pub name: &'static str,
    /// Jobs waiting in the queue.
    pub queued: usize,
    /// Jobs currently executing on a worker.
    pub running: usize,
    pub completed: u64,
    /// Jobs dropped because their token was cancelled before they started.
    pub cancelled: u64,
    /// Duration of the most recent completed job in milliseconds.
    pub last_ms: f64,
    /// Mean duration over every completed job.
    pub mean_ms: f64,
    pub max_ms: f64,
}

/// Handle to the worker pool. Cloning shares the pool; the workers stop when the last clone drops.
#[derive(Clone)]
pub struct JobSystem {
    inner: Arc<Inner>,
}

struct Inner {}

impl JobSystem {
    /// Starts `workers` threads named `vx-worker-<n>`. `stage_names` label the per stage counters;
    /// [`Stage`] values index into this slice.
    pub fn new(workers: usize, stage_names: &[&'static str]) -> Self {
        let _ = (workers, stage_names);
        todo!("workstream A")
    }

    /// `available_parallelism() - 1`, clamped to 1 to 16, so one core stays with the main thread.
    pub fn default_worker_count() -> usize {
        todo!("workstream A")
    }

    pub fn worker_count(&self) -> usize {
        todo!("workstream A")
    }

    /// Queues a job. It runs on a worker unless `cancel` is set before it starts.
    pub fn spawn(
        &self,
        stage: Stage,
        priority: Priority,
        cancel: CancelToken,
        job: impl FnOnce(&CancelToken) + Send + 'static,
    ) {
        let _ = (stage, priority, cancel, job, &self.inner);
        todo!("workstream A")
    }

    pub fn stats(&self, stage: Stage) -> StageStats {
        let _ = stage;
        todo!("workstream A")
    }

    pub fn all_stats(&self) -> Vec<StageStats> {
        todo!("workstream A")
    }

    /// Jobs waiting across every stage.
    pub fn queued_total(&self) -> usize {
        todo!("workstream A")
    }
}
