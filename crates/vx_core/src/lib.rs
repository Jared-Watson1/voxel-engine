//! Math and frames, time, configuration and the constants every other crate shares.
//!
//! Positions are f64 in an explicit frame; the GPU only ever sees camera relative f32 (D4).

pub mod config;
pub mod frame;
pub mod projection;
pub mod time;
pub mod watch;

pub use config::{CHUNK_SIZE, ConfigError, WorldConfig};
pub use frame::{Frame, WorldPos, camera_relative};
pub use projection::Projection;
pub use time::FixedStep;
pub use watch::{FileEvent, FileWatcher};
