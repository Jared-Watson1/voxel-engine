//! Job system (D9, D12): a fixed pool of worker threads draining one priority queue.
//!
//! Jobs carry a [`Priority`] (lower runs first, FIFO within a priority), a [`Stage`] label for
//! the per stage counters shown in the debug overlay, and a [`CancelToken`]. A queued job whose
//! token is cancelled before a worker picks it up never runs; a running job polls its own token.
//! Results travel on channels owned by the caller. Workers may spawn jobs, so one pipeline stage
//! can queue the next without a round trip through the main thread.

use std::any::Any;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Instant;

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

/// Owner of the worker threads, shared by every [`JobSystem`] clone. Workers hold only an
/// `Arc<Shared>`, never an `Arc<Inner>`, so the pool shuts down as soon as the last handle drops,
/// including when that last handle is one a job captured and the drop happens on a worker.
struct Inner {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    stages: Vec<Mutex<StageSlot>>,
}

/// The shutdown flag lives under the queue lock so a worker cannot miss the notification between
/// checking the flag and waiting on the condvar.
struct Queue {
    heap: BinaryHeap<QueuedJob>,
    next_seq: u64,
    shutdown: bool,
}

#[derive(Default)]
struct StageSlot {
    stats: StageStats,
    total_ms: f64,
}

type Job = Box<dyn FnOnce(&CancelToken) + Send>;

/// Heap key: priority first, then the submission sequence number, so equal priorities are FIFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct JobKey {
    priority: Priority,
    seq: u64,
}

struct QueuedJob {
    /// Reversed because `BinaryHeap` pops the greatest element and the lowest key runs first.
    key: Reverse<JobKey>,
    stage: Stage,
    cancel: CancelToken,
    job: Job,
}

impl PartialEq for QueuedJob {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for QueuedJob {}

impl PartialOrd for QueuedJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

/// A poisoned lock only means another thread panicked while holding it; the data inside is plain
/// counters and a heap, both left consistent, so the guard is recovered instead of propagated.
fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non string panic payload")
}

impl JobSystem {
    /// Starts `workers` threads named `vx-worker-<n>`. `stage_names` label the per stage counters;
    /// [`Stage`] values index into this slice.
    ///
    /// Panics if `workers` is zero, `stage_names` is empty, or the OS refuses to create a thread.
    pub fn new(workers: usize, stage_names: &[&'static str]) -> Self {
        assert!(
            workers > 0,
            "JobSystem::new: at least one worker is required"
        );
        assert!(
            !stage_names.is_empty(),
            "JobSystem::new: at least one stage name is required"
        );
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                heap: BinaryHeap::new(),
                next_seq: 0,
                shutdown: false,
            }),
            wake: Condvar::new(),
            stages: stage_names
                .iter()
                .map(|&name| {
                    Mutex::new(StageSlot {
                        stats: StageStats {
                            name,
                            ..StageStats::default()
                        },
                        total_ms: 0.0,
                    })
                })
                .collect(),
        });
        let workers = (0..workers)
            .map(|n| {
                let shared = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("vx-worker-{n}"))
                    .spawn(move || worker_loop(&shared))
                    // A pool with fewer workers than asked for would silently starve the
                    // pipeline; a thread the OS cannot create at startup is fatal.
                    .unwrap_or_else(|error| {
                        panic!("JobSystem::new: cannot spawn worker thread {n}: {error}")
                    })
            })
            .collect();
        Self {
            inner: Arc::new(Inner { shared, workers }),
        }
    }

    /// `available_parallelism() - 1`, clamped to 1 to 16, so one core stays with the main thread.
    pub fn default_worker_count() -> usize {
        thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .saturating_sub(1)
            .clamp(1, 16)
    }

    pub fn worker_count(&self) -> usize {
        self.inner.workers.len()
    }

    /// Queues a job. It runs on a worker unless `cancel` is set before it starts.
    ///
    /// Panics if `stage` is outside the names given to [`JobSystem::new`].
    pub fn spawn(
        &self,
        stage: Stage,
        priority: Priority,
        cancel: CancelToken,
        job: impl FnOnce(&CancelToken) + Send + 'static,
    ) {
        let shared = &self.inner.shared;
        // The queued counter rises before the push so the worker's decrement after the pop can
        // never observe it at zero.
        lock_recover(self.stage_slot(stage)).stats.queued += 1;
        {
            let mut queue = lock_recover(&shared.queue);
            let seq = queue.next_seq;
            queue.next_seq += 1;
            queue.heap.push(QueuedJob {
                key: Reverse(JobKey { priority, seq }),
                stage,
                cancel,
                job: Box::new(job),
            });
        }
        shared.wake.notify_one();
    }

    pub fn stats(&self, stage: Stage) -> StageStats {
        lock_recover(self.stage_slot(stage)).stats
    }

    pub fn all_stats(&self) -> Vec<StageStats> {
        self.inner
            .shared
            .stages
            .iter()
            .map(|slot| lock_recover(slot).stats)
            .collect()
    }

    /// Jobs waiting across every stage.
    pub fn queued_total(&self) -> usize {
        lock_recover(&self.inner.shared.queue).heap.len()
    }

    fn stage_slot(&self, stage: Stage) -> &Mutex<StageSlot> {
        let stages = &self.inner.shared.stages;
        assert!(
            stage.0 < stages.len(),
            "Stage({}) is out of range: the job system has {} stages",
            stage.0,
            stages.len()
        );
        &stages[stage.0]
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        lock_recover(&self.shared.queue).shutdown = true;
        self.shared.wake.notify_all();
        let current = thread::current().id();
        for handle in self.workers.drain(..) {
            // The last handle can drop inside a job, on a worker. That worker cannot join itself;
            // it leaves its loop on its own once it sees the shutdown flag.
            if handle.thread().id() == current {
                continue;
            }
            if handle.join().is_err() {
                tracing::error!("job system worker thread panicked outside a job");
            }
        }
    }
}

fn worker_loop(shared: &Shared) {
    while let Some(queued) = shared.next_job() {
        shared.execute(queued);
    }
}

impl Shared {
    /// Blocks until a job is available. Returns `None` once shutdown is requested; queued jobs are
    /// then left in the heap and dropped with it.
    fn next_job(&self) -> Option<QueuedJob> {
        let mut queue = lock_recover(&self.queue);
        loop {
            if queue.shutdown {
                return None;
            }
            if let Some(queued) = queue.heap.pop() {
                return Some(queued);
            }
            queue = self
                .wake
                .wait(queue)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Runs one popped job with no lock held, so the job may spawn and may drop the last
    /// [`JobSystem`] handle.
    fn execute(&self, queued: QueuedJob) {
        let QueuedJob {
            stage, cancel, job, ..
        } = queued;
        let slot = &self.stages[stage.0];
        if cancel.is_cancelled() {
            let mut slot = lock_recover(slot);
            slot.stats.queued = slot.stats.queued.saturating_sub(1);
            slot.stats.cancelled += 1;
            return;
        }
        {
            let mut slot = lock_recover(slot);
            slot.stats.queued = slot.stats.queued.saturating_sub(1);
            slot.stats.running += 1;
        }
        let start = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(|| job(&cancel)));
        let elapsed_ms = start.elapsed().as_secs_f64() * 1.0e3;
        if let Err(payload) = outcome {
            let worker = thread::current();
            tracing::error!(
                stage = lock_recover(slot).stats.name,
                worker = worker.name().unwrap_or("unnamed"),
                "job panicked: {}",
                panic_message(payload.as_ref())
            );
        }
        let mut slot = lock_recover(slot);
        let stats = &mut slot.stats;
        stats.running = stats.running.saturating_sub(1);
        stats.completed += 1;
        stats.last_ms = elapsed_ms;
        stats.max_ms = stats.max_ms.max(elapsed_ms);
        slot.total_ms += elapsed_ms;
        slot.stats.mean_ms = slot.total_ms / slot.stats.completed as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::sync::mpsc;
    use std::time::Duration;

    const TIMEOUT: Duration = Duration::from_secs(10);

    fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + TIMEOUT;
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Blocks a worker inside a job until the test opens it.
    struct Gate {
        open: Mutex<bool>,
        changed: Condvar,
    }

    impl Gate {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                open: Mutex::new(false),
                changed: Condvar::new(),
            })
        }

        fn open(&self) {
            *self.open.lock().unwrap() = true;
            self.changed.notify_all();
        }

        fn wait(&self) {
            let mut open = self.open.lock().unwrap();
            while !*open {
                open = self.changed.wait(open).unwrap();
            }
        }
    }

    #[test]
    fn jobs_run_in_priority_order_then_fifo() {
        let js = JobSystem::new(1, &["order"]);
        let order = Arc::new(Mutex::new(Vec::new()));
        let gate = Gate::new();
        {
            let gate = gate.clone();
            let order = order.clone();
            js.spawn(Stage(0), Priority::EDIT, CancelToken::new(), move |_| {
                gate.wait();
                order.lock().unwrap().push("gate");
            });
        }
        wait_until("gate job to start", || js.stats(Stage(0)).running == 1);
        for (label, priority) in [("p5", 5), ("p1a", 1), ("p3", 3), ("p1b", 1)] {
            let order = order.clone();
            js.spawn(
                Stage(0),
                Priority(priority),
                CancelToken::new(),
                move |_| {
                    order.lock().unwrap().push(label);
                },
            );
        }
        assert_eq!(js.queued_total(), 4);
        assert_eq!(js.stats(Stage(0)).queued, 4);
        gate.open();
        wait_until("all jobs to complete", || js.stats(Stage(0)).completed == 5);
        assert_eq!(*order.lock().unwrap(), ["gate", "p1a", "p1b", "p3", "p5"]);
    }

    #[test]
    fn cancelled_queued_job_never_runs() {
        let js = JobSystem::new(1, &["cancel"]);
        let gate = Gate::new();
        {
            let gate = gate.clone();
            js.spawn(Stage(0), Priority::EDIT, CancelToken::new(), move |_| {
                gate.wait()
            });
        }
        wait_until("gate job to start", || js.stats(Stage(0)).running == 1);
        let ran = Arc::new(AtomicBool::new(false));
        let token = CancelToken::new();
        {
            let ran = ran.clone();
            js.spawn(Stage(0), Priority(1), token.clone(), move |_| {
                ran.store(true, SeqCst);
            });
        }
        token.cancel();
        gate.open();
        wait_until("cancelled job to be dropped", || {
            js.stats(Stage(0)).cancelled == 1
        });
        assert!(!ran.load(SeqCst), "a cancelled queued job ran");
        let stats = js.stats(Stage(0));
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.queued, 0);
        assert_eq!(stats.running, 0);
    }

    #[test]
    fn running_job_observes_cancellation() {
        let js = JobSystem::new(1, &["cancel"]);
        let token = CancelToken::new();
        let (tx, rx) = mpsc::channel();
        js.spawn(Stage(0), Priority::EDIT, token.clone(), move |cancel| {
            while !cancel.is_cancelled() {
                thread::sleep(Duration::from_micros(100));
            }
            tx.send("observed").unwrap();
        });
        wait_until("job to start", || js.stats(Stage(0)).running == 1);
        assert!(rx.try_recv().is_err(), "job finished before cancellation");
        token.cancel();
        assert_eq!(rx.recv_timeout(TIMEOUT), Ok("observed"));
    }

    #[test]
    fn results_arrive_on_channels() {
        let js = JobSystem::new(4, &["generate", "mesh"]);
        let (tx, rx) = mpsc::channel();
        for i in 0..1000u64 {
            let tx = tx.clone();
            js.spawn(
                Stage(1),
                Priority::streaming(i),
                CancelToken::new(),
                move |_| {
                    tx.send(i * i).unwrap();
                },
            );
        }
        drop(tx);
        let sum: u64 = rx.iter().sum();
        assert_eq!(sum, (0..1000u64).map(|i| i * i).sum::<u64>());
        wait_until("completed to reach 1000", || {
            js.stats(Stage(1)).completed == 1000
        });
        let stats = js.stats(Stage(1));
        assert_eq!(stats.queued, 0);
        assert_eq!(stats.running, 0);
        assert_eq!(stats.cancelled, 0);
        assert_eq!(js.queued_total(), 0);
        assert_eq!(js.stats(Stage(0)).completed, 0);
    }

    #[test]
    fn job_can_spawn_a_child_job() {
        let js = JobSystem::new(1, &["pipeline"]);
        let (tx, rx) = mpsc::channel();
        let from_job = js.clone();
        js.spawn(Stage(0), Priority::EDIT, CancelToken::new(), move |_| {
            from_job.spawn(Stage(0), Priority(1), CancelToken::new(), move |_| {
                tx.send("child").unwrap();
            });
        });
        assert_eq!(rx.recv_timeout(TIMEOUT), Ok("child"));
        wait_until("both jobs to complete", || {
            js.stats(Stage(0)).completed == 2
        });
    }

    #[test]
    fn last_handle_dropped_inside_a_job_shuts_the_pool_down() {
        let js = JobSystem::new(2, &["pipeline"]);
        let shared = Arc::downgrade(&js.inner.shared);
        let gate = Gate::new();
        {
            let held = js.clone();
            let gate = gate.clone();
            js.spawn(Stage(0), Priority::EDIT, CancelToken::new(), move |_| {
                gate.wait();
                assert_eq!(held.worker_count(), 2);
            });
        }
        wait_until("job to start", || js.stats(Stage(0)).running == 1);
        drop(js);
        gate.open();
        wait_until("workers to release the shared state", || {
            shared.upgrade().is_none()
        });
    }

    #[test]
    fn stats_record_durations_and_names() {
        let js = JobSystem::new(1, &["generate", "mesh"]);
        js.spawn(Stage(1), Priority::EDIT, CancelToken::new(), |_| {
            thread::sleep(Duration::from_millis(5));
        });
        wait_until("first job to complete", || {
            js.stats(Stage(1)).completed == 1
        });
        let first = js.stats(Stage(1));
        assert!(first.last_ms >= 5.0, "last_ms {}", first.last_ms);
        assert_eq!(first.mean_ms, first.last_ms);
        assert_eq!(first.max_ms, first.last_ms);

        js.spawn(Stage(1), Priority::EDIT, CancelToken::new(), |_| {
            thread::sleep(Duration::from_millis(1));
        });
        wait_until("second job to complete", || {
            js.stats(Stage(1)).completed == 2
        });
        let second = js.stats(Stage(1));
        assert!(second.last_ms >= 1.0);
        assert!(second.max_ms >= first.max_ms);
        assert!(second.mean_ms <= second.max_ms);
        assert!(second.mean_ms >= second.last_ms.min(second.max_ms));
        let expected_mean = (first.last_ms + second.last_ms) / 2.0;
        assert!((second.mean_ms - expected_mean).abs() < 1.0e-9);

        let names: Vec<_> = js.all_stats().iter().map(|s| s.name).collect();
        assert_eq!(names, ["generate", "mesh"]);
        assert_eq!(js.all_stats()[0], js.stats(Stage(0)));
        assert_eq!(js.all_stats()[1], second);
    }

    #[test]
    fn dropping_with_queued_jobs_returns_and_stops_workers() {
        let js = JobSystem::new(1, &["drop"]);
        let shared = Arc::clone(&js.inner.shared);
        let gate = Gate::new();
        {
            let gate = gate.clone();
            js.spawn(Stage(0), Priority::EDIT, CancelToken::new(), move |_| {
                gate.wait()
            });
        }
        wait_until("gate job to start", || js.stats(Stage(0)).running == 1);
        let ran = Arc::new(AtomicUsize::new(0));
        for _ in 0..100 {
            let ran = ran.clone();
            js.spawn(Stage(0), Priority(1), CancelToken::new(), move |_| {
                ran.fetch_add(1, SeqCst);
            });
        }
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(js);
            done_tx.send(()).unwrap();
        });
        wait_until("shutdown flag", || lock_recover(&shared.queue).shutdown);
        gate.open();
        done_rx.recv_timeout(TIMEOUT).expect("shutdown deadlocked");
        assert_eq!(ran.load(SeqCst), 0, "queued jobs ran after shutdown");
        assert_eq!(lock_recover(&shared.queue).heap.len(), 100);
        let weak = Arc::downgrade(&shared);
        drop(shared);
        assert!(
            weak.upgrade().is_none(),
            "a worker still holds the shared state"
        );
    }

    #[test]
    fn panicking_job_does_not_kill_the_worker() {
        let js = JobSystem::new(1, &["panic"]);
        js.spawn(Stage(0), Priority::EDIT, CancelToken::new(), |_| {
            panic!("job panic for the test");
        });
        let (tx, rx) = mpsc::channel();
        js.spawn(Stage(0), Priority(1), CancelToken::new(), move |_| {
            tx.send("alive").unwrap();
        });
        assert_eq!(rx.recv_timeout(TIMEOUT), Ok("alive"));
        wait_until("both jobs to count as completed", || {
            js.stats(Stage(0)).completed == 2
        });
        assert_eq!(js.stats(Stage(0)).running, 0);
    }

    #[test]
    fn workers_are_named() {
        let js = JobSystem::new(3, &["name"]);
        assert_eq!(js.worker_count(), 3);
        let (tx, rx) = mpsc::channel();
        js.spawn(Stage(0), Priority::EDIT, CancelToken::new(), move |_| {
            tx.send(thread::current().name().map(str::to_owned))
                .unwrap();
        });
        let name = rx
            .recv_timeout(TIMEOUT)
            .unwrap()
            .expect("worker has no name");
        assert!(name.starts_with("vx-worker-"), "{name}");
    }

    #[test]
    fn priorities_and_default_worker_count() {
        assert!(Priority::streaming(0) > Priority::EDIT);
        assert!(Priority::streaming(1) > Priority::streaming(0));
        assert_eq!(Priority::streaming(u64::MAX), Priority(u32::MAX));
        let workers = JobSystem::default_worker_count();
        assert!((1..=16).contains(&workers), "{workers}");
    }

    #[test]
    #[should_panic(expected = "at least one worker")]
    fn zero_workers_panics() {
        let _ = JobSystem::new(0, &["stage"]);
    }

    #[test]
    #[should_panic(expected = "at least one stage")]
    fn no_stages_panics() {
        let _ = JobSystem::new(1, &[]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn unknown_stage_panics() {
        let js = JobSystem::new(1, &["stage"]);
        js.spawn(Stage(1), Priority::EDIT, CancelToken::new(), |_| {});
    }

    /// Measurement, not a check (engineering rule 5). Prints a BENCH line with the spawn cost.
    #[test]
    #[ignore = "run with: cargo test -p vx_core --release -- --ignored --nocapture spawn_overhead"]
    fn spawn_overhead() {
        const JOBS: u64 = 200_000;
        let workers = JobSystem::default_worker_count();
        let js = JobSystem::new(workers, &["bench"]);
        let (tx, rx) = mpsc::channel::<()>();
        let start = Instant::now();
        for _ in 0..JOBS {
            let tx = tx.clone();
            js.spawn(Stage(0), Priority(1), CancelToken::new(), move |_| drop(tx));
        }
        let spawn_elapsed = start.elapsed();
        drop(tx);
        // No job sends; the channel closes once every job has run and dropped its sender.
        assert!(rx.recv().is_err());
        let total_elapsed = start.elapsed();
        let stats = js.stats(Stage(0));
        assert_eq!(stats.completed, JOBS);
        println!(
            "BENCH spawn: {JOBS} empty jobs on {workers} workers: {:.0} ns per spawn on the caller, \
             {:.1} ms until all ran ({:.2} M jobs/s), mean job {:.4} ms",
            spawn_elapsed.as_nanos() as f64 / JOBS as f64,
            total_elapsed.as_secs_f64() * 1.0e3,
            JOBS as f64 / total_elapsed.as_secs_f64() / 1.0e6,
            stats.mean_ms,
        );
    }
}
