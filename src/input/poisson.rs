//! Poisson spike generators — deterministic, reproducible input spike trains.

use crate::simulation::Simulation;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize)]
pub struct PoissonSource {
    pub rate_hz: f32,
    pub t_cursor: f32,
    seed: u64,
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

    pub fn generate(&mut self, t_start: f32, t_end: f32) -> Vec<f32> {
        if self.rate_hz <= 0.0 || t_end <= t_start {
            return Vec::new();
        }
        let lambda = self.rate_hz / 1000.0;
        let mut t = t_start.max(self.t_cursor);
        let mut spikes = Vec::new();
        loop {
            let u: f32 = self.rng_state.r#gen();
            t += -(1.0f32 - u).ln() / lambda;
            if t >= t_end {
                break;
            }
            spikes.push(t);
        }
        self.t_cursor = t_end;
        spikes
    }

    pub fn mean_isi_ms(&self) -> f32 {
        1000.0 / self.rate_hz
    }
    pub fn cv(&self) -> f32 {
        1.0
    }
}

impl<'de> Deserialize<'de> for PoissonSource {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
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

pub struct PoissonPopulation {
    pub sources: Vec<PoissonSource>,
    pub n_neurons: usize,
    pub target_neurons: Vec<usize>,
    pub weight: f32,
    pub model_type: u8,
    pub e_rev: f32,
}

impl PoissonPopulation {
    pub fn new(n_neurons: usize, rate_hz: f32, weight: f32, seed: u64) -> Self {
        let sources = (0..n_neurons)
            .map(|i| {
                PoissonSource::new(
                    rate_hz,
                    seed.wrapping_add((i as u64).wrapping_mul(6364136223846793005)),
                )
            })
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

    pub fn targeting(targets: Vec<usize>, rate_hz: f32, weight: f32, seed: u64) -> Self {
        let n = targets.len();
        let sources = (0..n)
            .map(|i| {
                PoissonSource::new(
                    rate_hz,
                    seed.wrapping_add((i as u64).wrapping_mul(6364136223846793005)),
                )
            })
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

    pub fn inject_into(&mut self, sim: &mut Simulation, t_start: f32, t_end: f32) -> usize {
        let mut total = 0;
        for (i, src) in self.sources.iter_mut().enumerate() {
            let target = self.target_neurons[i];
            for t in src.generate(t_start, t_end) {
                sim.push_event(t, target, self.weight, self.model_type, self.e_rev);
                total += 1;
            }
        }
        total
    }

    pub fn prebuild(&mut self, sim: &mut Simulation, t_end: f32) -> usize {
        self.inject_into(sim, 0.0, t_end)
    }

    pub fn rate_stats(&mut self, t_end: f32) -> (f32, f32) {
        let mut clone = self.sources.clone();
        let counts: Vec<usize> = clone
            .iter_mut()
            .map(|s| s.generate(0.0, t_end).len())
            .collect();
        let mean = counts.iter().sum::<usize>() as f32 / counts.len() as f32;
        let var = counts
            .iter()
            .map(|&c| (c as f32 - mean).powi(2))
            .sum::<f32>()
            / counts.len() as f32;
        (mean / (t_end / 1000.0), var.sqrt())
    }
}

pub struct StimulusPattern {
    max_rate_hz: f32,
    rate_fn: Box<dyn Fn(f32) -> f32 + Send + Sync>,
    rng: ChaCha20Rng,
}

impl StimulusPattern {
    pub fn step(bg_rate: f32, stim_rate: f32, t_on: f32, t_off: f32, seed: u64) -> Self {
        Self {
            max_rate_hz: stim_rate,
            rate_fn: Box::new(move |t| {
                if t >= t_on && t < t_off {
                    stim_rate
                } else {
                    bg_rate
                }
            }),
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    pub fn sinusoidal(base_hz: f32, amplitude: f32, freq_hz: f32, seed: u64) -> Self {
        use std::f32::consts::PI;
        Self {
            max_rate_hz: base_hz + amplitude,
            rate_fn: Box::new(move |t| {
                (base_hz + amplitude * (2.0 * PI * freq_hz * t / 1000.0).sin()).max(0.0)
            }),
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    pub fn generate(&mut self, t_start: f32, t_end: f32) -> Vec<f32> {
        let lm = self.max_rate_hz / 1000.0;
        if lm <= 0.0 {
            return vec![];
        }
        let mut t = t_start;
        let mut spikes = Vec::new();
        loop {
            let u: f32 = self.rng.gen_range(0.0..1.0);
            t += -(1.0 - u).ln() / lm;
            if t >= t_end {
                break;
            }
            let accept: f32 = self.rng.gen_range(0.0..1.0);
            if accept < (self.rate_fn)(t) / 1000.0 / lm {
                spikes.push(t);
            }
        }
        spikes
    }
}

pub fn drive_background(
    sim: &mut Simulation,
    n_neurons: usize,
    rate_hz: f32,
    weight: f32,
    seed: u64,
    t_end: f32,
) -> usize {
    PoissonPopulation::new(n_neurons, rate_hz, weight, seed).prebuild(sim, t_end)
}
