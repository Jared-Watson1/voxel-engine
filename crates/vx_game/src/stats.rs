use std::collections::VecDeque;

/// Rolling frame interval and CPU time samples in milliseconds.
pub struct FrameSamples {
    capacity: usize,
    frame_ms: VecDeque<f64>,
    cpu_ms: VecDeque<f64>,
}

impl FrameSamples {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            frame_ms: VecDeque::with_capacity(capacity),
            cpu_ms: VecDeque::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, frame_ms: f64, cpu_ms: f64) {
        if self.frame_ms.len() == self.capacity {
            self.frame_ms.pop_front();
            self.cpu_ms.pop_front();
        }
        self.frame_ms.push_back(frame_ms);
        self.cpu_ms.push_back(cpu_ms);
    }

    pub fn clear(&mut self) {
        self.frame_ms.clear();
        self.cpu_ms.clear();
    }

    pub fn mean_frame_ms(&self) -> f64 {
        mean(&self.frame_ms)
    }

    pub fn mean_cpu_ms(&self) -> f64 {
        mean(&self.cpu_ms)
    }
}

fn mean(v: &VecDeque<f64>) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub frames: usize,
    pub seconds: f64,
    pub frame_ms_mean: f64,
    pub frame_ms_p50: f64,
    pub frame_ms_p99: f64,
    pub frame_ms_max: f64,
    pub fps: f64,
    pub cpu_ms_mean: f64,
}

impl Summary {
    pub fn from_samples(samples: &FrameSamples) -> Option<Self> {
        if samples.frame_ms.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = samples.frame_ms.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let percentile = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
        let total_ms: f64 = sorted.iter().sum();
        Some(Self {
            frames: sorted.len(),
            seconds: total_ms / 1.0e3,
            frame_ms_mean: total_ms / sorted.len() as f64,
            frame_ms_p50: percentile(0.5),
            frame_ms_p99: percentile(0.99),
            frame_ms_max: sorted[sorted.len() - 1],
            fps: sorted.len() as f64 / (total_ms / 1.0e3),
            cpu_ms_mean: samples.mean_cpu_ms(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_of_known_samples() {
        let mut s = FrameSamples::with_capacity(1000);
        for i in 1..=100 {
            s.push(f64::from(i) * 0.1, 1.0);
        }
        let sum = Summary::from_samples(&s).unwrap();
        assert_eq!(sum.frames, 100);
        assert!((sum.frame_ms_mean - 5.05).abs() < 1e-9);
        assert!((sum.frame_ms_max - 10.0).abs() < 1e-9);
        assert!((sum.fps - 100.0 / 0.505).abs() < 1e-6);
        assert_eq!(sum.cpu_ms_mean, 1.0);
    }

    #[test]
    fn rolling_window_drops_oldest() {
        let mut s = FrameSamples::with_capacity(2);
        s.push(10.0, 0.0);
        s.push(20.0, 0.0);
        s.push(30.0, 0.0);
        assert_eq!(s.mean_frame_ms(), 25.0);
        s.clear();
        assert!(Summary::from_samples(&s).is_none());
    }
}
