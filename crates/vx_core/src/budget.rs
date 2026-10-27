//! Performance budgets as named constants (engineering rule 4). The values come from the design
//! document's budgets table and change only with measurements recorded in the roadmap.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Budgets for the Apple M1 baseline at the design defaults (D11).
pub struct Budgets;

impl Budgets {
    /// Whole frame at 60 fps.
    pub const FRAME_MS: f64 = 16.6;
    /// Main thread work per frame: scheduling, streaming bookkeeping, uploads and egui.
    pub const MAIN_THREAD_MS: f64 = 3.0;
    /// GPU time for the terrain pass.
    pub const TERRAIN_DRAW_MS: f64 = 6.0;
    /// Generate plus mesh of one 32^3 chunk on one worker.
    pub const CHUNK_GENERATE_MESH_MS: f64 = 4.0;
    /// Compressed voxel data resident in RAM, apron included.
    pub const RESIDENT_VOXEL_BYTES: usize = 1536 << 20;
    /// Terrain vertex and index pools plus material textures in VRAM.
    pub const GPU_TERRAIN_BYTES: usize = 1536 << 20;
    /// Mesh bytes copied to the GPU per frame; the upload stage stops for the frame once reached.
    pub const UPLOAD_BYTES_PER_FRAME: usize = 4 << 20;
    /// Terrain triangles on screen after culling.
    pub const TERRAIN_TRIANGLES_ON_SCREEN: u64 = 3_000_000;
}

/// One measured value against its budget, for the overlay and the debug checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetUse {
    pub name: &'static str,
    pub used: f64,
    pub budget: f64,
    /// Unit shown after the numbers, for example "ms" or "MB".
    pub unit: &'static str,
}

impl BudgetUse {
    pub fn ms(name: &'static str, used_ms: f64, budget_ms: f64) -> Self {
        Self {
            name,
            used: used_ms,
            budget: budget_ms,
            unit: "ms",
        }
    }

    /// Bytes are displayed in megabytes (10^6).
    pub fn bytes(name: &'static str, used: usize, budget: usize) -> Self {
        Self {
            name,
            used: used as f64 / 1.0e6,
            budget: budget as f64 / 1.0e6,
            unit: "MB",
        }
    }

    pub fn count(name: &'static str, used: u64, budget: u64) -> Self {
        Self {
            name,
            used: used as f64,
            budget: budget as f64,
            unit: "",
        }
    }

    pub fn fraction(&self) -> f64 {
        if self.budget > 0.0 {
            self.used / self.budget
        } else {
            0.0
        }
    }

    pub fn exceeded(&self) -> bool {
        self.used > self.budget
    }
}

/// Minimum time between two warnings about the same budget, so a sustained overrun logs once
/// instead of once per frame.
const LOG_INTERVAL: Duration = Duration::from_secs(5);

/// Runtime budget check for debug builds: logs each exceeded budget, rate limited per budget, and
/// does nothing in release builds (the overlay still shows the values there).
#[derive(Debug, Default)]
pub struct BudgetMonitor {
    last_logged: HashMap<&'static str, Instant>,
}

impl BudgetMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the names of the budgets exceeded in `uses`; in debug builds also logs them.
    pub fn check(&mut self, uses: &[BudgetUse]) -> Vec<&'static str> {
        self.check_at(Instant::now(), uses)
    }

    fn check_at(&mut self, now: Instant, uses: &[BudgetUse]) -> Vec<&'static str> {
        let mut exceeded = Vec::new();
        for over in uses.iter().filter(|budget_use| budget_use.exceeded()) {
            exceeded.push(over.name);
            if cfg!(debug_assertions) {
                self.log_rate_limited(now, over);
            }
        }
        exceeded
    }

    fn log_rate_limited(&mut self, now: Instant, over: &BudgetUse) {
        let recently_logged = self
            .last_logged
            .get(over.name)
            .is_some_and(|last| now.duration_since(*last) < LOG_INTERVAL);
        if recently_logged {
            return;
        }
        self.last_logged.insert(over.name, now);
        tracing::warn!(
            budget = over.name,
            used = over.used,
            limit = over.budget,
            unit = over.unit,
            "budget exceeded: {} used {:.2}{} of {:.2}{} ({:.0}%)",
            over.name,
            over.used,
            over.unit,
            over.budget,
            over.unit,
            over.fraction() * 100.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_returns_exactly_the_exceeded_names() {
        let mut monitor = BudgetMonitor::new();
        let uses = [
            BudgetUse::ms("frame", 20.0, Budgets::FRAME_MS),
            BudgetUse::ms("main_thread", 1.0, Budgets::MAIN_THREAD_MS),
            BudgetUse::bytes("resident_voxels", 2048 << 20, Budgets::RESIDENT_VOXEL_BYTES),
            BudgetUse::count("triangles", 100, Budgets::TERRAIN_TRIANGLES_ON_SCREEN),
            BudgetUse::count("uploads", 5, 4),
        ];
        assert_eq!(
            monitor.check(&uses),
            ["frame", "resident_voxels", "uploads"]
        );
        assert_eq!(
            monitor.check(&uses),
            ["frame", "resident_voxels", "uploads"]
        );
    }

    #[test]
    fn under_budget_returns_empty() {
        let mut monitor = BudgetMonitor::new();
        let uses = [
            BudgetUse::ms("frame", 16.6, Budgets::FRAME_MS),
            BudgetUse::ms("terrain_draw", 0.0, Budgets::TERRAIN_DRAW_MS),
            BudgetUse::bytes("gpu_terrain", 0, Budgets::GPU_TERRAIN_BYTES),
            BudgetUse::count("triangles", 0, Budgets::TERRAIN_TRIANGLES_ON_SCREEN),
        ];
        assert!(monitor.check(&uses).is_empty());
        assert!(monitor.check(&[]).is_empty());
    }

    #[test]
    fn exactly_at_budget_is_not_exceeded() {
        let at_limit = BudgetUse::ms("frame", Budgets::FRAME_MS, Budgets::FRAME_MS);
        assert!(!at_limit.exceeded());
        assert_eq!(at_limit.fraction(), 1.0);
        assert_eq!(BudgetUse::count("zero", 1, 0).fraction(), 0.0);
    }

    #[test]
    fn bytes_display_in_megabytes() {
        let use_bytes = BudgetUse::bytes("upload", 4_000_000, Budgets::UPLOAD_BYTES_PER_FRAME);
        assert_eq!(use_bytes.used, 4.0);
        assert_eq!(use_bytes.unit, "MB");
        assert!(!use_bytes.exceeded());
        assert!(BudgetUse::bytes("upload", (4 << 20) + 1, 4 << 20).exceeded());
    }

    #[test]
    fn logging_is_rate_limited_per_budget_in_debug_builds() {
        let mut monitor = BudgetMonitor::new();
        let over = [
            BudgetUse::ms("frame", 20.0, Budgets::FRAME_MS),
            BudgetUse::ms("main_thread", 4.0, Budgets::MAIN_THREAD_MS),
        ];
        let start = Instant::now();
        assert_eq!(monitor.check_at(start, &over), ["frame", "main_thread"]);
        let within = start + Duration::from_secs(4);
        assert_eq!(monitor.check_at(within, &over), ["frame", "main_thread"]);
        let after = start + LOG_INTERVAL;
        assert_eq!(monitor.check_at(after, &over[..1]), ["frame"]);

        if cfg!(debug_assertions) {
            assert_eq!(monitor.last_logged.get("frame"), Some(&after));
            assert_eq!(monitor.last_logged.get("main_thread"), Some(&start));
        } else {
            assert!(monitor.last_logged.is_empty());
        }
    }
}
