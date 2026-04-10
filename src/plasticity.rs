//! Spike-Timing Dependent Plasticity (STDP)
//!
//! STDP is a Hebbian synaptic learning rule where synaptic strength changes
//! based on the relative timing of pre- and post-synaptic spikes:
//!
//!   - Pre before Post (Δt > 0): LTP — weight increases (causally connected)
//!   - Post before Pre (Δt < 0): LTD — weight decreases (non-causal)
//!
//! Update rule (nearest-neighbour pair-based):
//!   ΔW_+ = A_+ · exp(-Δt / τ_+)    if pre fires before post
//!   ΔW_- = -A_- · exp( Δt / τ_-)   if post fires before pre
//!
//! This module tracks eligibility traces per neuron and integrates with the
//! main Simulation event loop through `StdpState::apply`.

use serde::{Deserialize, Serialize};

/// Configuration for the STDP learning rule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StdpConfig {
    /// LTP amplitude (positive).
    pub a_plus: f32,
    /// LTD amplitude (positive — sign is applied internally).
    pub a_minus: f32,
    /// LTP time constant (ms).
    pub tau_plus: f32,
    /// LTD time constant (ms).
    pub tau_minus: f32,
    /// Minimum synaptic weight.
    pub w_min: f32,
    /// Maximum synaptic weight.
    pub w_max: f32,
    /// Whether STDP is currently enabled.
    pub enabled: bool,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            a_plus:    0.005,
            a_minus:   0.005,
            tau_plus:  20.0,
            tau_minus: 20.0,
            w_min:     0.0,
            w_max:     10.0,
            enabled:   true,
        }
    }
}

impl StdpConfig {
    /// Standard symmetric STDP (equal LTP and LTD windows).
    pub fn symmetric() -> Self { Self::default() }

    /// Asymmetric (anti-Hebbian) STDP — LTD window wider than LTP.
    pub fn asymmetric_ltd() -> Self {
        Self {
            a_plus:    0.004,
            a_minus:   0.006,
            tau_plus:  15.0,
            tau_minus: 30.0,
            ..Default::default()
        }
    }

    /// Compute potentiation magnitude for a pre→post pair with delay Δt (ms).
    #[inline]
    pub fn ltp(&self, dt_ms: f32) -> f32 {
        self.a_plus * (-dt_ms / self.tau_plus).exp()
    }

    /// Compute depression magnitude for a post→pre pair with delay |Δt| (ms).
    #[inline]
    pub fn ltd(&self, dt_ms: f32) -> f32 {
        -self.a_minus * (-dt_ms / self.tau_minus).exp()
    }
}

/// Per-neuron STDP eligibility traces.
///
/// Two exponentially-decaying traces are maintained for each neuron:
///   - `x`: pre-synaptic trace, bumped on pre-spike
///   - `y`: post-synaptic trace, bumped on post-spike
///
/// Weight updates are computed when a spike occurs on either end.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StdpState {
    /// Last spike time for each neuron (f32::NEG_INFINITY if never fired).
    pub last_pre_spike_time:  Vec<f32>,
    pub last_post_spike_time: Vec<f32>,

    /// Eligibility trace for each neuron (decays exponentially between spikes).
    pub x_trace: Vec<f32>, // pre-synaptic trace
    pub y_trace: Vec<f32>, // post-synaptic trace

    pub config: StdpConfig,

    /// Accumulated weight deltas, indexed by synapse index.
    /// Flushed into the Synapse weight vector at each learning step.
    pub delta_w: Vec<f32>,

    /// Total number of weight updates applied (for logging).
    pub update_count: u64,
}

impl StdpState {
    pub fn new(n_neurons: usize, n_synapses: usize, config: StdpConfig) -> Self {
        Self {
            last_pre_spike_time:  vec![f32::NEG_INFINITY; n_neurons],
            last_post_spike_time: vec![f32::NEG_INFINITY; n_neurons],
            x_trace: vec![0.0; n_neurons],
            y_trace: vec![0.0; n_neurons],
            config,
            delta_w: vec![0.0; n_synapses],
            update_count: 0,
        }
    }

    /// Decay all traces by one timestep `dt`.
    pub fn decay_traces(&mut self, dt: f32) {
        let inv_tau_plus  = 1.0 / self.config.tau_plus;
        let inv_tau_minus = 1.0 / self.config.tau_minus;
        for x in self.x_trace.iter_mut() {
            *x *= (-dt * inv_tau_plus).exp();
        }
        for y in self.y_trace.iter_mut() {
            *y *= (-dt * inv_tau_minus).exp();
        }
    }

    /// Record a pre-synaptic spike at neuron `nid` at time `t`.
    ///
    /// Bumps the x-trace and returns the current y-trace value (used for LTD).
    pub fn record_pre_spike(&mut self, nid: usize, t: f32) -> f32 {
        self.last_pre_spike_time[nid] = t;
        self.x_trace[nid] += 1.0;
        self.y_trace[nid] // LTD contribution: how strong was post-synaptic recent activity?
    }

    /// Record a post-synaptic spike at neuron `nid` at time `t`.
    ///
    /// Bumps the y-trace and returns the current x-trace value (used for LTP).
    pub fn record_post_spike(&mut self, nid: usize, t: f32) -> f32 {
        self.last_post_spike_time[nid] = t;
        self.y_trace[nid] += 1.0;
        self.x_trace[nid] // LTP contribution: how strong was pre-synaptic recent activity?
    }

    /// Apply pending weight changes to synapse weight vectors.
    ///
    /// Should be called once per simulation timestep (or batch of timesteps).
    /// Returns the number of synapses whose weights changed.
    pub fn flush_weight_updates(
        &mut self,
        weights: &mut [f32],
    ) -> usize {
        if !self.config.enabled {
            return 0;
        }

        let mut changed = 0usize;
        for (w, dw) in weights.iter_mut().zip(self.delta_w.iter_mut()) {
            if dw.abs() > 1e-8 {
                *w = (*w + *dw).clamp(self.config.w_min, self.config.w_max);
                *dw = 0.0;
                changed += 1;
                self.update_count += 1;
            }
        }
        changed
    }

    /// Compute and accumulate STDP updates for all synapses involving `spiking_neuron`.
    ///
    /// Call once per neuron per spike event. Takes:
    ///   - `spiking_neuron`: global index of the neuron that just fired
    ///   - `t`: current simulation time
    ///   - `pre_vec` / `post_vec`: synapse pre/post neuron index arrays
    ///   - `model_mask`: only update synapses marked as plastic (bit 0 of model_type)
    pub fn accumulate_for_spike(
        &mut self,
        nid: usize,
        t: f32,
        syn_pre: &[usize],
        syn_post: &[usize],
        pre_index: &[Vec<usize>],
    ) {
        if !self.config.enabled {
            return;
        }

        // -------- PRE spike (causes LTD on outgoing synapses)
        if nid < pre_index.len() {
            self.last_pre_spike_time[nid] = t;
            self.x_trace[nid] += 1.0;

            for &syn_idx in &pre_index[nid] {
                let post = syn_post[syn_idx];
                let dt = t - self.last_post_spike_time[post];

                if dt <= 0.0 {   // pre before post
                    self.delta_w[syn_idx] += self.config.ltp((-dt).max(0.0));
                } else {
                    self.delta_w[syn_idx] += self.config.ltd(dt);
                }
            }
        }

        // -------- POST spike (causes LTP on incoming synapses)
        for (syn_idx, &post) in syn_post.iter().enumerate() {
            if post == nid {
                let pre = syn_pre[syn_idx];

                self.last_post_spike_time[nid] = t;
                self.y_trace[nid] += 1.0;

                let dt = t - self.last_pre_spike_time[pre];

                if dt >= 0.0 {
                    self.delta_w[syn_idx] += self.config.ltp(dt.max(0.0));   // ← changed to ltp
                } else {
                    self.delta_w[syn_idx] += self.config.ltd(dt.abs());      // optional: handle anti-causal
                }
                
            }
        }
    }

    
    /// Return a summary of the current weight distribution.
    pub fn weight_stats(weights: &[f32]) -> WeightStats {
        if weights.is_empty() {
            return WeightStats::default();
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &w in weights {
            if w < min { min = w; }
            if w > max { max = w; }
            sum += w as f64;
        }
        let mean = sum / weights.len() as f64;
        let variance = weights.iter().map(|&w| {
            let d = w as f64 - mean;
            d * d
        }).sum::<f64>() / weights.len() as f64;

        WeightStats {
            min,
            max,
            mean: mean as f32,
            std: variance.sqrt() as f32,
            n: weights.len(),
        }
    }
}

/// Summary statistics for synaptic weight distribution.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WeightStats {
    pub min:  f32,
    pub max:  f32,
    pub mean: f32,
    pub std:  f32,
    pub n:    usize,
}

impl std::fmt::Display for WeightStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n={} min={:.4} mean={:.4} max={:.4} std={:.4}",
            self.n, self.min, self.mean, self.max, self.std)
    }
}
