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

    /// Sets the warp factor (simulated seconds per real second, at least 0). Negative values clamp
    /// to 0; a non finite value (NaN or infinity) sets 0 so the clock can never reach a non finite
    /// time.
    pub fn set_warp(&mut self, warp: f64) {
        self.warp = if warp.is_finite() { warp.max(0.0) } else { 0.0 };
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Jumps to an absolute simulation time, for automation and screenshots.
    pub fn set_time_s(&mut self, time_s: f64) {
        self.time_s = time_s;
    }

    /// Advances by `real_dt_s` of real time and returns the simulated seconds that passed: 0 while
    /// paused, and 0 for a negative or non finite `real_dt_s` (simulation time never runs
    /// backwards through `advance`; `set_time_s` is the way to jump).
    pub fn advance(&mut self, real_dt_s: f64) -> f64 {
        if self.paused || !real_dt_s.is_finite() || real_dt_s <= 0.0 {
            return 0.0;
        }
        let dt_s = real_dt_s * self.warp;
        self.time_s += dt_s;
        dt_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_scales_real_time_by_the_warp() {
        let mut clock = SimClock::new(10.0);
        assert_eq!(clock.advance(0.5), 0.5);
        assert_eq!(clock.time_s(), 10.5);
        clock.set_warp(1000.0);
        assert_eq!(clock.advance(0.25), 250.0);
        assert_eq!(clock.time_s(), 260.5);
    }

    #[test]
    fn paused_clock_does_not_move() {
        let mut clock = SimClock::new(0.0);
        clock.set_warp(100.0);
        clock.set_paused(true);
        assert_eq!(clock.advance(1.0), 0.0);
        assert_eq!(clock.time_s(), 0.0);
        clock.set_paused(false);
        assert_eq!(clock.advance(1.0), 100.0);
    }

    #[test]
    fn warp_is_clamped_and_finite() {
        let mut clock = SimClock::new(0.0);
        clock.set_warp(-3.0);
        assert_eq!(clock.warp(), 0.0);
        clock.set_warp(f64::NAN);
        assert_eq!(clock.warp(), 0.0);
        clock.set_warp(f64::INFINITY);
        assert_eq!(clock.warp(), 0.0);
        clock.set_warp(0.0);
        assert_eq!(clock.advance(1.0), 0.0);
    }

    #[test]
    fn bad_real_time_steps_are_ignored() {
        let mut clock = SimClock::new(5.0);
        assert_eq!(clock.advance(-1.0), 0.0);
        assert_eq!(clock.advance(f64::NAN), 0.0);
        assert_eq!(clock.advance(f64::INFINITY), 0.0);
        assert_eq!(clock.time_s(), 5.0);
    }

    #[test]
    fn set_time_jumps() {
        let mut clock = SimClock::new(0.0);
        clock.set_time_s(-42.0);
        assert_eq!(clock.time_s(), -42.0);
    }
}
