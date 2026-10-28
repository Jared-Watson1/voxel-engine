/// Simulation time in f64 seconds since the system epoch, advanced from real frame time times a
/// warp factor. Bodies on rails are analytic, so any warp is exact for them (engineering rule 8);
/// stepped simulation (the ship in M5 and M6) multiplies its step count instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimClock {
    time_s: f64,
    warp: f64,
    paused: bool,
}

impl SimClock {
    pub fn new(start_s: f64) -> Self {
        Self {
            time_s: start_s,
            warp: 1.0,
            paused: false,
        }
    }

    pub fn time_s(&self) -> f64 {
        self.time_s
    }

    pub fn warp(&self) -> f64 {
        self.warp
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Sets the warp factor (simulated seconds per real second, at least 0).
    pub fn set_warp(&mut self, warp: f64) {
        let _ = warp;
        todo!("workstream A")
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Jumps to an absolute simulation time, for automation and screenshots.
    pub fn set_time_s(&mut self, time_s: f64) {
        self.time_s = time_s;
    }

    /// Advances by `real_dt_s` of real time and returns the simulated seconds that passed.
    pub fn advance(&mut self, real_dt_s: f64) -> f64 {
        let _ = real_dt_s;
        todo!("workstream A")
    }
}
