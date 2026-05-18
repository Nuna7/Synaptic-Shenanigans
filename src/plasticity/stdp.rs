use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StdpConfig {
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub w_min: f32,
    pub w_max: f32,
    pub enabled: bool,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.005,
            a_minus: 0.005,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_min: 0.0,
            w_max: 10.0,
            enabled: true,
        }
    }
}

impl StdpConfig {
    pub fn symmetric() -> Self {
        Self::default()
    }
    pub fn asymmetric_ltd() -> Self {
        Self {
            a_plus: 0.004,
            a_minus: 0.006,
            tau_plus: 15.0,
            tau_minus: 30.0,
            ..Default::default()
        }
    }
    #[inline]
    pub fn ltp(&self, dt_ms: f32) -> f32 {
        self.a_plus * (-dt_ms / self.tau_plus).exp()
    }
    #[inline]
    pub fn ltd(&self, dt_ms: f32) -> f32 {
        -self.a_minus * (-dt_ms / self.tau_minus).exp()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StdpState {
    pub last_pre_spike_time: Vec<f32>,
    pub last_post_spike_time: Vec<f32>,
    pub x_trace: Vec<f32>,
    pub y_trace: Vec<f32>,
    pub config: StdpConfig,
    pub delta_w: Vec<f32>,
    pub update_count: u64,
}

impl StdpState {
    pub fn new(n_neurons: usize, n_synapses: usize, config: StdpConfig) -> Self {
        Self {
            last_pre_spike_time: vec![f32::NEG_INFINITY; n_neurons],
            last_post_spike_time: vec![f32::NEG_INFINITY; n_neurons],
            x_trace: vec![0.0; n_neurons],
            y_trace: vec![0.0; n_neurons],
            config,
            delta_w: vec![0.0; n_synapses],
            update_count: 0,
        }
    }

    pub fn decay_traces(&mut self, dt: f32) {
        let ip = 1.0 / self.config.tau_plus;
        let im = 1.0 / self.config.tau_minus;
        for x in &mut self.x_trace {
            *x *= (-dt * ip).exp();
        }
        for y in &mut self.y_trace {
            *y *= (-dt * im).exp();
        }
    }

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
        if nid < pre_index.len() {
            self.last_pre_spike_time[nid] = t;
            self.x_trace[nid] += 1.0;
            for &si in &pre_index[nid] {
                let post = syn_post[si];
                let dt = t - self.last_post_spike_time[post];
                self.delta_w[si] += if dt <= 0.0 {
                    self.config.ltp((-dt).max(0.0))
                } else {
                    self.config.ltd(dt)
                };
            }
        }
        for (si, &post) in syn_post.iter().enumerate() {
            if post == nid {
                let pre = syn_pre[si];
                self.last_post_spike_time[nid] = t;
                self.y_trace[nid] += 1.0;
                let dt = t - self.last_pre_spike_time[pre];
                self.delta_w[si] += if dt >= 0.0 {
                    self.config.ltp(dt.max(0.0))
                } else {
                    self.config.ltd(dt.abs())
                };
            }
        }
    }

    pub fn flush_weight_updates(&mut self, weights: &mut [f32]) -> usize {
        if !self.config.enabled {
            return 0;
        }
        let mut changed = 0;
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

    pub fn weight_stats(weights: &[f32]) -> WeightStats {
        if weights.is_empty() {
            return WeightStats::default();
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &w in weights {
            if w < min {
                min = w;
            }
            if w > max {
                max = w;
            }
            sum += w as f64;
        }
        let mean = sum / weights.len() as f64;
        let var = weights
            .iter()
            .map(|&w| {
                let d = w as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / weights.len() as f64;
        WeightStats {
            min,
            max,
            mean: mean as f32,
            std: var.sqrt() as f32,
            n: weights.len(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WeightStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub n: usize,
}

impl std::fmt::Display for WeightStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} min={:.4} mean={:.4} max={:.4} std={:.4}",
            self.n, self.min, self.mean, self.max, self.std
        )
    }
}
