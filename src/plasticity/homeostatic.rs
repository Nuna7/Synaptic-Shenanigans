//! Homeostatic intrinsic plasticity.
//!
//! Adapts each neuron's firing threshold to maintain a target firing rate:
//!   τ_h · dθ/dt = r_actual(t) - r_target
//!
//! Works with any [`NeuronPopulation`] via the `set_threshold` trait method.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HomeostaticConfig {
    pub target_rate_hz: f32,
    pub tau_h: f32,
    pub theta_min: f32,
    pub theta_max: f32,
    pub rate_window_ms: f32,
    pub update_interval_ms: f32,
    pub enabled: bool,
}

impl Default for HomeostaticConfig {
    fn default() -> Self {
        Self {
            target_rate_hz: 5.0,
            tau_h: 10_000.0,
            theta_min: -70.0,
            theta_max: -40.0,
            rate_window_ms: 1_000.0,
            update_interval_ms: 100.0,
            enabled: true,
        }
    }
}

impl HomeostaticConfig {
    pub fn fast() -> Self {
        Self {
            tau_h: 500.0,
            update_interval_ms: 50.0,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HomeostaticState {
    pub config: HomeostaticConfig,
    pub theta: Vec<f32>,
    pub spike_count_window: Vec<u32>,
    spike_times: Vec<Vec<f32>>,
    pub update_count: u64,
    last_update_time: f32,
    pub history: Vec<(f32, usize, f32, f32, f32)>,
}

impl HomeostaticState {
    pub fn new(n_neurons: usize, initial_theta: f32, config: HomeostaticConfig) -> Self {
        Self {
            theta: vec![initial_theta; n_neurons],
            spike_count_window: vec![0; n_neurons],
            spike_times: vec![Vec::new(); n_neurons],
            update_count: 0,
            last_update_time: 0.0,
            history: Vec::new(),
            config,
        }
    }

    pub fn record_spike(&mut self, nid: usize, t: f32) {
        if nid >= self.spike_times.len() {
            return;
        }
        self.spike_times[nid].push(t);
        self.spike_count_window[nid] += 1;
    }

    fn prune_window(&mut self, t_now: f32) {
        let cutoff = t_now - self.config.rate_window_ms;
        for nid in 0..self.spike_times.len() {
            let before = self.spike_times[nid].len();
            self.spike_times[nid].retain(|&st| st >= cutoff);
            let removed = before.saturating_sub(self.spike_times[nid].len());
            self.spike_count_window[nid] =
                self.spike_count_window[nid].saturating_sub(removed as u32);
        }
    }

    pub fn estimated_rate(&self, nid: usize) -> f32 {
        self.spike_times[nid].len() as f32 / (self.config.rate_window_ms / 1000.0)
    }

    pub fn update(&mut self, t_now: f32) -> usize {
        if !self.config.enabled {
            return 0;
        }
        if t_now - self.last_update_time < self.config.update_interval_ms {
            return 0;
        }
        let dt = t_now - self.last_update_time;
        self.last_update_time = t_now;
        self.prune_window(t_now);
        let mut changed = 0usize;
        for nid in 0..self.theta.len() {
            let r_actual = self.estimated_rate(nid);
            let old = self.theta[nid];
            let delta = (dt / self.config.tau_h) * (r_actual - self.config.target_rate_hz);
            let new_theta = (old + delta).clamp(self.config.theta_min, self.config.theta_max);
            if (new_theta - old).abs() > 1e-5 {
                self.theta[nid] = new_theta;
                self.update_count += 1;
                changed += 1;
                if (new_theta - old).abs() > 0.1 {
                    self.history.push((t_now, nid, old, new_theta, r_actual));
                }
            }
        }
        changed
    }

    /// Apply adapted thresholds to any NeuronPopulation via the trait method.
    pub fn apply_thresholds(&self, neurons: &dyn crate::neurons::NeuronPopulation) {
        for (i, &theta) in self.theta.iter().enumerate() {
            neurons.set_threshold(i, theta);
        }
    }

    /// Kept for backward compatibility — delegates to `apply_thresholds`.
    pub fn apply_thresholds_to_lif(&self, neurons: &dyn crate::neurons::NeuronPopulation) {
        self.apply_thresholds(neurons);
    }

    pub fn threshold_stats(&self) -> ThresholdStats {
        let n = self.theta.len();
        if n == 0 {
            return ThresholdStats::default();
        }
        let min = self.theta.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self.theta.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = self.theta.iter().sum::<f32>() / n as f32;
        let var = self.theta.iter().map(|&t| (t - mean).powi(2)).sum::<f32>() / n as f32;
        ThresholdStats {
            min,
            max,
            mean,
            std: var.sqrt(),
            n,
        }
    }

    pub fn rate_distribution(&self) -> Vec<f32> {
        (0..self.theta.len())
            .map(|i| self.estimated_rate(i))
            .collect()
    }

    pub fn fraction_at_target(&self) -> f32 {
        let n = self.theta.len();
        if n == 0 {
            return 0.0;
        }
        let target = self.config.target_rate_hz;
        (0..n)
            .filter(|&i| (self.estimated_rate(i) - target).abs() <= 1.0)
            .count() as f32
            / n as f32
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ThresholdStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub n: usize,
}

impl std::fmt::Display for ThresholdStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "θ  min={:.2} mean={:.2} max={:.2} std={:.3} mV (n={})",
            self.min, self.mean, self.max, self.std, self.n
        )
    }
}
