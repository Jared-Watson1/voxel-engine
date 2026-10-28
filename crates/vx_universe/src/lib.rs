//! Star system model, Keplerian propagation, rotation states, frames and simulation time (D6,
//! D24).
//!
//! Bodies move on Keplerian rails relative to their parent and are evaluated analytically at the
//! simulation time, so time warp scales the time they are evaluated at (engineering rule 8). All
//! positions are f64 metres. Axes: the system frame has +Y at the ecliptic north, the ecliptic in
//! the XZ plane and +X as the reference direction; a body fixed frame has +Y at the north pole
//! (the rotation axis) and +X through the prime meridian. M2 builds one star, one planet and an
//! optional moon; spheres of influence and the ship's N-body integration arrive in M6.

mod clock;
mod kepler;
mod system;

pub use clock::SimClock;
pub use kepler::{OrbitElements, RotationState, solve_kepler};
pub use system::{
    MoonConfig, OrbitConfig, PlanetConfig, StarConfig, StarSystem, SystemConfig, SystemError,
    SystemSnapshot,
};
