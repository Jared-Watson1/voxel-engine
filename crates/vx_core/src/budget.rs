//! Performance budgets as named constants (engineering rule 4). The values come from the design
//! document's budgets table and change only with measurements recorded in the roadmap.

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

/// Runtime budget check for debug builds: logs each exceeded budget, rate limited per budget, and
/// does nothing in release builds (the overlay still shows the values there).
#[derive(Debug, Default)]
pub struct BudgetMonitor {}

impl BudgetMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the names of the budgets exceeded in `uses`; in debug builds also logs them.
    pub fn check(&mut self, uses: &[BudgetUse]) -> Vec<&'static str> {
        let _ = uses;
        todo!("workstream A")
    }
}
