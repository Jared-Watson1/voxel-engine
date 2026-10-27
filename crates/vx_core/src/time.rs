/// Fixed time step accumulator: simulation advances in whole steps of `step_s`, rendering
/// interpolates with [`FixedStep::alpha`] (engineering rule 8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedStep {
    step_s: f64,
    max_steps_per_frame: u32,
    accumulator_s: f64,
}

impl FixedStep {
    pub fn new(step_s: f64, max_steps_per_frame: u32) -> Self {
        assert!(step_s > 0.0 && max_steps_per_frame > 0);
        Self {
            step_s,
            max_steps_per_frame,
            accumulator_s: 0.0,
        }
    }

    pub fn step_s(&self) -> f64 {
        self.step_s
    }

    /// Adds real elapsed time and returns how many fixed steps to run this frame.
    ///
    /// Time beyond `max_steps_per_frame` steps is dropped so a long stall cannot snowball into
    /// ever longer frames.
    pub fn advance(&mut self, dt_s: f64) -> u32 {
        self.accumulator_s += dt_s.max(0.0);
        let steps = (self.accumulator_s / self.step_s).floor();
        let taken = steps.min(f64::from(self.max_steps_per_frame));
        self.accumulator_s -= taken * self.step_s;
        if steps > taken {
            self.accumulator_s = 0.0;
        }
        taken as u32
    }

    /// Fraction of a step left in the accumulator, for interpolating between simulation states.
    pub fn alpha(&self) -> f64 {
        (self.accumulator_s / self.step_s).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_partial_steps() {
        let mut t = FixedStep::new(0.01, 8);
        assert_eq!(t.advance(0.004), 0);
        assert_eq!(t.advance(0.004), 0);
        assert_eq!(t.advance(0.004), 1);
        assert!((t.alpha() - 0.2).abs() < 1.0e-9);
    }

    #[test]
    fn caps_steps_after_a_stall() {
        let mut t = FixedStep::new(0.01, 4);
        assert_eq!(t.advance(1.0), 4);
        assert_eq!(t.alpha(), 0.0);
        assert_eq!(t.advance(0.0), 0);
    }
}
