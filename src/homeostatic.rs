//! Homeostatic intrinsic plasticity.
//!
//! **The problem:** Hebbian/STDP learning rules are inherently unstable —
//! strong synapses drive more spikes, which strengthen synapses further,
//! leading to runaway excitation or silence (Bienenstock-Cooper-Munro problem).
//!
//! **The solution:** Homeostatic plasticity regulates neuron intrinsic
//! excitability so that each neuron maintains a target firing rate over a slow
//! timescale, independent of (and complementary to) Hebbian weight changes.
//!
//! **Mechanism:** We adapt each neuron's firing threshold V_thresh:
//!
//!   τ_h · dθ/dt = r_actual(t) - r_target
//!
//! where:
//!   - θ = V_thresh (adapted threshold, mV)
//!   - r_actual = sliding average of recent firing rate (Hz)
//!   - r_target = desired firing rate (Hz)
//!   - τ_h = homeostatic time constant (ms, typically very slow, ~10 000 ms)
//!
//! If the neuron fires **too fast** → threshold rises → harder to fire.
//! If the neuron fires **too slowly** → threshold drops → easier to fire.
//!
//! This implements a slow negative feedback loop that stabilises the network
//! after Hebbian plasticity changes, synaptic rewiring, or input statistics
//! changes.
//!
//! References:
//!   Turrigiano et al. (1998). Nature, 391, 892–896.
//!   Turrigiano (2011). Cold Spring Harb. Perspect. Biol.

use serde::{Deserialize, Serialize};

/// Configuration for homeostatic plasticity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HomeostaticConfig {
    /// Target firing rate per neuron (Hz).
    pub target_rate_hz: f32,
    /// Homeostatic time constant (ms). Larger → slower adaptation.
    /// Biologically ~days, but we use ~10 000 ms for simulation.
    pub tau_h: f32,
    /// Minimum threshold (mV). Hard floor to prevent pathological behaviour.
    pub theta_min: f32,
    /// Maximum threshold (mV). Hard ceiling.
    pub theta_max: f32,
    /// Rate estimation window (ms). Sliding window for average rate.
    pub rate_window_ms: f32,
    /// How often to apply homeostatic updates (ms). Decoupled from sim dt.
    pub update_interval_ms: f32,
    /// Whether homeostatic adaptation is currently enabled.
    pub enabled: bool,
}

impl Default for HomeostaticConfig {
    fn default() -> Self {
        Self {
            target_rate_hz:   5.0,
            tau_h:         10_000.0,
            theta_min:       -70.0,
            theta_max:       -40.0,
            rate_window_ms: 1_000.0,
            update_interval_ms: 100.0,
            enabled: true,
        }
    }
}

impl HomeostaticConfig {
    /// Faster adaptation for testing (τ_h = 500 ms).
    pub fn fast() -> Self {
        Self { tau_h: 500.0, update_interval_ms: 50.0, ..Default::default() }
    }

    /// Slow biological timescale (τ_h = 100 000 ms ≈ 100 s simulated).
    pub fn biological() -> Self {
        Self { tau_h: 100_000.0, update_interval_ms: 500.0, ..Default::default() }
    }
}

/// Per-neuron homeostatic state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HomeostaticState {
    pub config: HomeostaticConfig,

    /// Current adapted threshold for each neuron (mV).
    pub theta: Vec<f32>,

    /// Sliding spike count in the rate estimation window.
    pub spike_count_window: Vec<u32>,

    /// Timestamp of each neuron's spikes in the current window (ring buffer).
    spike_times: Vec<Vec<f32>>,

    /// Number of homeostatic updates applied (for logging).
    pub update_count: u64,

    /// Time of last homeostatic update.
    last_update_time: f32,

    /// Log: (time_ms, neuron_id, old_theta, new_theta, actual_rate)
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

    /// Record a spike for neuron `nid` at time `t`.
    pub fn record_spike(&mut self, nid: usize, t: f32) {
        if nid >= self.spike_times.len() { return; }
        self.spike_times[nid].push(t);
        self.spike_count_window[nid] += 1;
    }

    /// Prune spikes older than the rate estimation window.
    fn prune_window(&mut self, t_now: f32) {
        let cutoff = t_now - self.config.rate_window_ms;
        for nid in 0..self.spike_times.len() {
            let before = self.spike_times[nid].len();
            self.spike_times[nid].retain(|&st| st >= cutoff);
            let after = self.spike_times[nid].len();
            let removed = before.saturating_sub(after);
            self.spike_count_window[nid] = self.spike_count_window[nid].saturating_sub(removed as u32);
        }
    }

    /// Compute the estimated firing rate for neuron `nid` (Hz).
    pub fn estimated_rate(&self, nid: usize) -> f32 {
        self.spike_times[nid].len() as f32 / (self.config.rate_window_ms / 1000.0)
    }

    /// Apply homeostatic threshold updates for all neurons.
    ///
    /// Should be called periodically (every `config.update_interval_ms`).
    /// Modifies `self.theta` and returns the number of neurons whose threshold changed.
    pub fn update(&mut self, t_now: f32) -> usize {
        if !self.config.enabled { return 0; }
        if t_now - self.last_update_time < self.config.update_interval_ms { return 0; }

        let dt = t_now - self.last_update_time;
        self.last_update_time = t_now;
        self.prune_window(t_now);

        let mut changed = 0usize;
        let n = self.theta.len();

        for nid in 0..n {
            let r_actual = self.estimated_rate(nid);
            let old_theta = self.theta[nid];

            // Euler update: Δθ = (dt / τ_h) · (r_actual - r_target)
            let delta = (dt / self.config.tau_h) * (r_actual - self.config.target_rate_hz);
            let new_theta = (old_theta + delta).clamp(self.config.theta_min, self.config.theta_max);

            if (new_theta - old_theta).abs() > 1e-5 {
                self.theta[nid] = new_theta;
                self.update_count += 1;
                changed += 1;

                // Log significant changes (> 0.1 mV)
                if (new_theta - old_theta).abs() > 0.1 {
                    self.history.push((t_now, nid, old_theta, new_theta, r_actual));
                }
            }
        }
        changed
    }

    /// Apply adapted thresholds back into the LIF neuron population.
    ///
    /// This writes `self.theta[i]` into `neurons.v_thresh[i]` for all neurons,
    /// effectively updating each neuron's firing threshold.
    pub fn apply_thresholds_to_lif(&self, neurons: &mut crate::lif::LifNeuron) {
        for (i, &theta) in self.theta.iter().enumerate() {
            if i < neurons.v_thresh.len() {
                neurons.v_thresh[i] = theta;
            }
        }
    }

    /// Summary statistics for current threshold distribution.
    pub fn threshold_stats(&self) -> ThresholdStats {
        let n = self.theta.len();
        if n == 0 { return ThresholdStats::default(); }
        let min  = self.theta.iter().cloned().fold(f32::INFINITY, f32::min);
        let max  = self.theta.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = self.theta.iter().sum::<f32>() / n as f32;
        let var  = self.theta.iter().map(|&t| (t - mean).powi(2)).sum::<f32>() / n as f32;
        ThresholdStats { min, max, mean, std: var.sqrt(), n }
    }

    /// Rate distribution across neurons.
    pub fn rate_distribution(&self) -> Vec<f32> {
        (0..self.theta.len()).map(|i| self.estimated_rate(i)).collect()
    }

    /// How many neurons are within ±1 Hz of the target rate.
    pub fn fraction_at_target(&self) -> f32 {
        let n = self.theta.len();
        if n == 0 { return 0.0; }
        let target = self.config.target_rate_hz;
        let at_target = (0..n).filter(|&i| {
            (self.estimated_rate(i) - target).abs() <= 1.0
        }).count();
        at_target as f32 / n as f32
    }
}

/// Summary statistics for the threshold distribution.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ThresholdStats {
    pub min:  f32,
    pub max:  f32,
    pub mean: f32,
    pub std:  f32,
    pub n:    usize,
}

impl std::fmt::Display for ThresholdStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "θ  min={:.2} mean={:.2} max={:.2} std={:.3} mV (n={})",
            self.min, self.mean, self.max, self.std, self.n)
    }
}
