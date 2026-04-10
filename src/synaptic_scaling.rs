//! Synaptic scaling — global homeostatic weight normalisation.
//!
//! While `homeostatic.rs` adjusts *intrinsic excitability* (threshold) of
//! individual neurons, synaptic scaling adjusts *all incoming weights* of a
//! neuron multiplicatively to maintain a target average synaptic strength.
//!
//! This is the network-level complement to intrinsic homeostasis:
//!   - Hebbian STDP can strengthen individual synapses without bound.
//!   - Intrinsic homeostasis can only compensate via threshold shifts.
//!   - Synaptic scaling renormalises the entire weight distribution while
//!     preserving the *relative* differences STDP created.
//!
//! Rule (Turrigiano 1998):
//!   W_i(t+1) = W_i(t) · (r_target / r_actual)^α
//!
//! where α controls the strength of the correction (default 1.0).
//!
//! References:
//!   Turrigiano et al. (1998). Nature, 391, 892-896.
//!   Turrigiano (2008). Neuron, 60, 477-490.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapticScalingConfig {
    /// Target firing rate each neuron should achieve (Hz).
    pub target_rate_hz: f32,
    /// Scaling exponent α. 1.0 = linear correction. Default 1.0.
    pub alpha: f32,
    /// Minimum allowed weight after scaling.
    pub w_min: f32,
    /// Maximum allowed weight after scaling.
    pub w_max: f32,
    /// How often to apply scaling (ms).
    pub update_interval_ms: f32,
    /// Rate estimation window (ms).
    pub rate_window_ms: f32,
    /// Whether scaling is currently enabled.
    pub enabled: bool,
}

impl Default for SynapticScalingConfig {
    fn default() -> Self {
        Self {
            target_rate_hz:    5.0,
            alpha:             1.0,
            w_min:             0.0,
            w_max:            20.0,
            update_interval_ms: 500.0,
            rate_window_ms:   1_000.0,
            enabled:          true,
        }
    }
}

/// Tracks per-neuron firing history and applies multiplicative weight scaling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapticScaling {
    pub config: SynapticScalingConfig,

    /// Recent spike times per neuron (rolling window).
    spike_history: Vec<Vec<f32>>,

    /// Number of scaling updates applied.
    pub update_count: u64,

    /// Last time scaling was applied.
    last_update_time: f32,

    /// History of scale factors: (time_ms, neuron_id, scale_factor).
    pub log: Vec<(f32, usize, f32)>,

    n_neurons: usize,
}

impl SynapticScaling {
    pub fn new(n_neurons: usize, config: SynapticScalingConfig) -> Self {
        Self {
            spike_history: vec![Vec::new(); n_neurons],
            update_count: 0,
            last_update_time: 0.0,
            log: Vec::new(),
            n_neurons,
            config,
        }
    }

    /// Record that neuron `nid` fired at time `t`.
    pub fn record_spike(&mut self, nid: usize, t: f32) {
        if nid < self.n_neurons {
            self.spike_history[nid].push(t);
        }
    }

    /// Prune spikes older than the rate window.
    fn prune(&mut self, t_now: f32) {
        let cutoff = t_now - self.config.rate_window_ms;
        for h in self.spike_history.iter_mut() {
            h.retain(|&t| t >= cutoff);
        }
    }

    /// Estimated current firing rate (Hz) for neuron `nid`.
    pub fn estimated_rate(&self, nid: usize) -> f32 {
        self.spike_history[nid].len() as f32 / (self.config.rate_window_ms / 1000.0)
    }

    /// Apply multiplicative scaling to all incoming synapses of each neuron.
    ///
    /// `post_vec`: `synapse.post[i]` — the post-synaptic neuron of synapse i.
    /// `weights`:  `synapse.weight`  — mutable weight vector.
    ///
    /// Returns the number of synapses whose weights were modified.
    pub fn scale_weights(
        &mut self,
        t_now: f32,
        post_vec: &[usize],
        weights: &mut [f32],
    ) -> usize {
        if !self.config.enabled { return 0; }
        if t_now - self.last_update_time < self.config.update_interval_ms { return 0; }

        self.last_update_time = t_now;
        self.prune(t_now);

        let target = self.config.target_rate_hz;
        let alpha  = self.config.alpha;

        // Compute per-neuron scale factors
        let scale_factors: Vec<f32> = (0..self.n_neurons).map(|nid| {
            let r = self.estimated_rate(nid);
            if r < 1e-3 {
                // Under-active → scale up, but cap to avoid explosion
                (1.0f32 + alpha * 0.1).min(1.5)
            } else {
                (target / r).powf(alpha)
            }
        }).collect();

        // Apply to all synapses based on their post-synaptic neuron
        let mut changed = 0;
        for (i, w) in weights.iter_mut().enumerate() {
            if i >= post_vec.len() { break; }
            let post = post_vec[i];
            if post >= self.n_neurons { continue; }
            let s = scale_factors[post];
            if (s - 1.0).abs() > 1e-4 {
                let new_w = (*w * s).clamp(self.config.w_min, self.config.w_max);
                *w = new_w;
                changed += 1;
                self.update_count += 1;
            }
        }

        // Log non-trivial scaling events
        for (nid, &s) in scale_factors.iter().enumerate() {
            if (s - 1.0).abs() > 0.05 {
                self.log.push((t_now, nid, s));
            }
        }

        changed
    }

    /// Current mean scale factor across neurons (diagnostic).
    pub fn mean_estimated_rate(&self) -> f32 {
        let sum: f32 = (0..self.n_neurons).map(|i| self.estimated_rate(i)).sum();
        sum / self.n_neurons as f32
    }

    /// Fraction of neurons within ±1 Hz of target.
    pub fn fraction_at_target(&self) -> f32 {
        let at: usize = (0..self.n_neurons)
            .filter(|&i| (self.estimated_rate(i) - self.config.target_rate_hz).abs() <= 1.0)
            .count();
        at as f32 / self.n_neurons as f32
    }

    /// Weight distribution stats after scaling.
    pub fn weight_stats(weights: &[f32]) -> (f32, f32, f32) {
        if weights.is_empty() { return (0.0, 0.0, 0.0); }
        let min  = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max  = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = weights.iter().sum::<f32>() / weights.len() as f32;
        (min, mean, max)
    }
}