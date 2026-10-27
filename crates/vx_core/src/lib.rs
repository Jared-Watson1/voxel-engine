//! Math and frames, time, configuration, the job system, budgets and the constants every other
//! crate shares.
//!
//! Positions are f64 in an explicit frame; the GPU only ever sees camera relative f32 (D4).

pub mod budget;
pub mod config;
pub mod frame;
pub mod job;
pub mod mesh;
pub mod projection;
pub mod time;
pub mod watch;

pub use budget::{BudgetMonitor, BudgetUse, Budgets};
pub use config::{CHUNK_SIZE, ConfigError, TerrainParams, WorldConfig};
pub use frame::{Frame, WorldPos, camera_relative};
pub use job::{CancelToken, JobSystem, Priority, Stage, StageStats};
pub use mesh::{PackedVertex, VertexAttributes};
pub use projection::Projection;
pub use time::FixedStep;
pub use watch::{FileEvent, FileWatcher};
