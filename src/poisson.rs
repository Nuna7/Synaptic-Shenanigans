//! Poisson spike generator — deterministic, reproducible input spike trains.
//!
//! Cortical neurons receive thousands of synaptic inputs, many of which fire
//! approximately independently at low rates. Poisson processes are the standard
//! statistical model for this background activity.
//!
//! This module provides:
//!
//!   1. **`PoissonSource`** — a single homogeneous Poisson process (constant rate).
//!      Generates spike times via inverse-transform sampling: t_{i+1} = t_i - ln(U)/λ.
//!
//!   2. **`PoissonPopulation`** — N independent Poisson sources, all driven by the
//!      same seeded RNG for full reproducibility.
//!
//!   3. **`StimulusPattern`** — inhomogeneous Poisson with a time-varying rate
//!      λ(t). Supports step functions and sinusoidal modulation.
//!
//! All generators are deterministic: given the same seed and parameters they
//! produce bit-identical spike trains across machines and thread counts.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::simulation::Simulation;

/// A single homogeneous Poisson process.
///
/// Spike times are generated on demand by drawing exponential inter-spike
/// intervals: ISI ~ Exp(λ), so P(ISI > t) = e^{-λt}.
#[derive(Clone, Debug, Serialize)]
pub struct PoissonSource {
    /// Firing rate (Hz = spikes/second = spikes/1000 ms).
    pub rate_hz: f32,
    /// Current simulation time cursor (ms). Advanced by `generate`.
    pub t_cursor: f32,
    seed: u64,

    /// Seeded RNG state — fully reproducible.
    #[serde(skip_serializing, skip_deserializing)]
    rng_state: ChaCha20Rng,
}

impl PoissonSource {
    pub fn new(rate_hz: f32, seed: u64) -> Self {
        Self {
            rate_hz,
            t_cursor: 0.0,
            seed,
            rng_state: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    pub fn restore_rng(&mut self) {
        self.rng_state = ChaCha20Rng::seed_from_u64(self.seed);
    }

    /// Generate all spikes in `[t_start, t_end)` ms.
    ///
    /// Advances the internal cursor to `t_end` after generation.
    /// Returns a `Vec<f32>` of spike times (ms).
    pub fn generate(&mut self, t_start: f32, t_end: f32) -> Vec<f32> {
        if self.rate_hz <= 0.0 || t_end <= t_start {
            return Vec::new();
        }

        let lambda_per_ms = self.rate_hz / 1000.0;
        let mut t = t_start.max(self.t_cursor);
        let mut spikes = Vec::new();

        loop {
            let u: f32 = self.rng_state.r#gen();
            let isi = -(1.0f32 - u).ln() / lambda_per_ms;
            t += isi;
            if t >= t_end { break; }
            spikes.push(t);
        }

        self.t_cursor = t_end;
        spikes
    }


    /// Theoretical statistics.
    pub fn mean_isi_ms(&self) -> f32 { 1000.0 / self.rate_hz }
    pub fn cv(&self) -> f32 { 1.0 } // Poisson is always CV = 1
}

impl<'de> Deserialize<'de> for PoissonSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            rate_hz: f32,
            t_cursor: f32,
            seed: u64,
        }

        let h = Helper::deserialize(deserializer)?;
        Ok(Self {
            rate_hz: h.rate_hz,
            t_cursor: h.t_cursor,
            seed: h.seed,
            rng_state: ChaCha20Rng::seed_from_u64(h.seed),
        })
    }
}


/// N independent Poisson sources driving N target neurons.
///
/// Each source has its own independent RNG stream (seeded with `seed + neuron_id`),
/// ensuring statistical independence while remaining fully reproducible.
pub struct PoissonPopulation {
    pub sources: Vec<PoissonSource>,
    pub n_neurons: usize,
    pub target_neurons: Vec<usize>,   // which simulation neurons receive input
    pub weight: f32,                   // synaptic weight per input spike
    pub model_type: u8,               // 0 = current-based
    pub e_rev: f32,                   // reversal potential (conductance-based only)
}

impl PoissonPopulation {
    /// Drive neurons `0..n_neurons` with independent Poisson inputs.
    pub fn new(n_neurons: usize, rate_hz: f32, weight: f32, seed: u64) -> Self {
        let sources = (0..n_neurons)
            .map(|i| PoissonSource::new(rate_hz, seed.wrapping_add((i as u64).wrapping_mul(6364136223846793005))))
            .collect();
        Self {
            sources,
            n_neurons,
            target_neurons: (0..n_neurons).collect(),
            weight,
            model_type: 0,
            e_rev: 0.0,
        }
    }

    /// Drive a specific subset of neurons.
    pub fn targeting(targets: Vec<usize>, rate_hz: f32, weight: f32, seed: u64) -> Self {
        let n = targets.len();
        let sources = (0..n)
            .map(|i| PoissonSource::new(rate_hz, seed.wrapping_add((i as u64).wrapping_mul(6364136223846793005))))
            .collect();
        Self {
            sources,
            n_neurons: n,
            target_neurons: targets,
            weight,
            model_type: 0,
            e_rev: 0.0,
        }
    }

    /// Inject all Poisson spikes in `[t_start, t_end)` into `sim` as events.
    ///
    /// Returns the total number of events injected.
    pub fn inject_into(&mut self, sim: &mut Simulation, t_start: f32, t_end: f32) -> usize {
        let mut total = 0usize;
        for (src_idx, source) in self.sources.iter_mut().enumerate() {
            let target = self.target_neurons[src_idx];
            let spikes = source.generate(t_start, t_end);
            for spike_t in spikes {
                sim.push_event(spike_t, target, self.weight, self.model_type, self.e_rev);
                total += 1;
            }
        }
        total
    }

    /// Pre-generate all input events for `[0, t_end)` in one batch.
    ///
    /// More efficient than calling `inject_into` at each timestep.
    pub fn prebuild(&mut self, sim: &mut Simulation, t_end: f32) -> usize {
        self.inject_into(sim, 0.0, t_end)
    }

    /// Estimate actual firing rates from generated spikes (statistical check).
    pub fn rate_stats(&mut self, t_end: f32) -> (f32, f32) {
        let mut clone = self.sources.clone();
        let counts: Vec<usize> = clone.iter_mut()
            .map(|s| s.generate(0.0, t_end).len())
            .collect();
        let mean = counts.iter().sum::<usize>() as f32 / counts.len() as f32;
        let var  = counts.iter().map(|&c| (c as f32 - mean).powi(2)).sum::<f32>() / counts.len() as f32;
        (mean / (t_end / 1000.0), var.sqrt())
    }
}

/// Inhomogeneous Poisson process with time-varying rate λ(t).
///
/// Uses thinning (rejection sampling) against a maximum rate bound.
pub struct StimulusPattern {
    max_rate_hz: f32,
    rate_fn: Box<dyn Fn(f32) -> f32 + Send + Sync>,
    rng: ChaCha20Rng,
}

impl StimulusPattern {
    /// Step function: low background → high stimulus at `t_on`, returns to `bg` at `t_off`.
    pub fn step(bg_rate: f32, stim_rate: f32, t_on: f32, t_off: f32, seed: u64) -> Self {
        Self {
            max_rate_hz: stim_rate,
            rate_fn: Box::new(move |t| {
                if t >= t_on && t < t_off { stim_rate } else { bg_rate }
            }),
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Sinusoidal modulation: rate = base + amplitude·sin(2π·freq·t/1000).
    pub fn sinusoidal(base_hz: f32, amplitude: f32, freq_hz: f32, seed: u64) -> Self {
        use std::f32::consts::PI;
        Self {
            max_rate_hz: base_hz + amplitude,
            rate_fn: Box::new(move |t| {
                let r = base_hz + amplitude * (2.0 * PI * freq_hz * t / 1000.0).sin();
                r.max(0.0)
            }),
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Generate spike times in `[t_start, t_end)` ms using thinning.
    pub fn generate(&mut self, t_start: f32, t_end: f32) -> Vec<f32> {
        let lambda_max = self.max_rate_hz / 1000.0;
        if lambda_max <= 0.0 { return vec![]; }

        let mut t = t_start;
        let mut spikes = Vec::new();

        loop {
            let u: f32 = self.rng.gen_range(0.0..1.0);
            t += -(1.0 - u).ln() / lambda_max;
            if t >= t_end { break; }

            // Thinning: accept with probability λ(t) / λ_max
            let accept: f32 = self.rng.gen_range(0.0..1.0);
            let lambda_t = (self.rate_fn)(t) / 1000.0;
            if accept < lambda_t / lambda_max {
                spikes.push(t);
            }
        }
        spikes
    }
}

// ── Convenience builder ──────────────────────────────────────────────────

/// Generate a Poisson background drive for an entire simulation and inject
/// it as pre-built events. Returns event count.
///
/// # Example
/// ```rust,no_run
/// use synaptic_shenanigans::poisson::drive_background;
///
/// # let mut sim = todo!(); // provided by simulation setup
/// let n_events = drive_background(&mut sim, 100, 10.0, 50.0, 42, 500.0);
/// ```
pub fn drive_background(
    sim: &mut Simulation,
    n_neurons: usize,
    rate_hz: f32,
    weight: f32,
    seed: u64,
    t_end: f32,
) -> usize {
    let mut pop = PoissonPopulation::new(n_neurons, rate_hz, weight, seed);
    pop.prebuild(sim, t_end)
}
