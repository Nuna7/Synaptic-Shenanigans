use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapticScalingConfig {
    pub target_rate_hz: f32,
    pub alpha: f32,
    pub w_min: f32,
    pub w_max: f32,
    pub update_interval_ms: f32,
    pub rate_window_ms: f32,
    pub enabled: bool,
}

impl Default for SynapticScalingConfig {
    fn default() -> Self {
        Self {
            target_rate_hz: 5.0,
            alpha: 1.0,
            w_min: 0.0,
            w_max: 20.0,
            update_interval_ms: 500.0,
            rate_window_ms: 1_000.0,
            enabled: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapticScaling {
    pub config: SynapticScalingConfig,
    spike_history: Vec<Vec<f32>>,
    pub update_count: u64,
    last_update_time: f32,
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

    pub fn record_spike(&mut self, nid: usize, t: f32) {
        if nid < self.n_neurons {
            self.spike_history[nid].push(t);
        }
    }

    fn prune(&mut self, t_now: f32) {
        let c = t_now - self.config.rate_window_ms;
        for h in &mut self.spike_history {
            h.retain(|&t| t >= c);
        }
    }

    pub fn estimated_rate(&self, nid: usize) -> f32 {
        self.spike_history[nid].len() as f32 / (self.config.rate_window_ms / 1000.0)
    }

    pub fn scale_weights(&mut self, t_now: f32, post_vec: &[usize], weights: &mut [f32]) -> usize {
        if !self.config.enabled {
            return 0;
        }
        if t_now - self.last_update_time < self.config.update_interval_ms {
            return 0;
        }
        self.last_update_time = t_now;
        self.prune(t_now);
        let target = self.config.target_rate_hz;
        let alpha = self.config.alpha;
        let scales: Vec<f32> = (0..self.n_neurons)
            .map(|nid| {
                let r = self.estimated_rate(nid);
                if r < 1e-3 {
                    (1.0f32 + alpha * 0.1).min(1.5)
                } else {
                    (target / r).powf(alpha)
                }
            })
            .collect();
        let mut changed = 0;
        for (i, w) in weights.iter_mut().enumerate() {
            if i >= post_vec.len() {
                break;
            }
            let post = post_vec[i];
            if post >= self.n_neurons {
                continue;
            }
            let s = scales[post];
            if (s - 1.0).abs() > 1e-4 {
                *w = (*w * s).clamp(self.config.w_min, self.config.w_max);
                changed += 1;
                self.update_count += 1;
            }
        }
        for (nid, &s) in scales.iter().enumerate() {
            if (s - 1.0).abs() > 0.05 {
                self.log.push((t_now, nid, s));
            }
        }
        changed
    }

    pub fn fraction_at_target(&self) -> f32 {
        if self.n_neurons == 0 {
            return 0.0;
        }
        let at = (0..self.n_neurons)
            .filter(|&i| (self.estimated_rate(i) - self.config.target_rate_hz).abs() <= 1.0)
            .count();
        at as f32 / self.n_neurons as f32
    }

    pub fn weight_stats(weights: &[f32]) -> (f32, f32, f32) {
        if weights.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = weights.iter().sum::<f32>() / weights.len() as f32;
        (min, mean, max)
    }
}
