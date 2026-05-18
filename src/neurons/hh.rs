//! Hodgkin-Huxley neuron model — biophysically accurate Na⁺/K⁺/leak channels.
//!
//!   Cm·dV/dt = -I_Na - I_K - I_L + I_ext
//!   dx/dt    = α_x(V)·(1-x) - β_x(V)·x   for x ∈ {m, h, n}
//!
//! Spike detection: upward crossing of `v_spike_thresh` (default 0 mV).
//! Refractory period emerges from channel kinetics — no hard timer needed.

use super::{NeuronPartition, NeuronPopulation};
use crossbeam::atomic::AtomicCell;
use std::any::Any;

#[derive(Clone, Debug)]
pub struct HHParams {
    pub c_m: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_spike_thresh: f64,
}

impl Default for HHParams {
    fn default() -> Self {
        Self {
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.387,
            dt: 0.01,
            v_spike_thresh: 0.0,
        }
    }
}

pub struct HHNeuronState {
    pub v: Vec<AtomicCell<f64>>,
    pub m: Vec<AtomicCell<f64>>,
    pub h: Vec<AtomicCell<f64>>,
    pub n: Vec<AtomicCell<f64>>,
    pub spiked: Vec<AtomicCell<bool>>,
    pub above_thresh: Vec<AtomicCell<bool>>,
}

pub struct HHPopulation {
    pub state: HHNeuronState,
    pub params: Vec<HHParams>,
    n: usize,
    // spike detection threshold stored for trait compat (matches params.v_spike_thresh)
    spike_thresh: Vec<AtomicCell<f32>>,
}

impl HHPopulation {
    pub fn homogeneous(n: usize, params: HHParams) -> Self {
        let v_rest = -65.0f64;
        let (m0, h0, n0) = steady_state(v_rest);
        let thresh = params.v_spike_thresh as f32;
        Self {
            state: HHNeuronState {
                v: (0..n).map(|_| AtomicCell::new(v_rest)).collect(),
                m: (0..n).map(|_| AtomicCell::new(m0)).collect(),
                h: (0..n).map(|_| AtomicCell::new(h0)).collect(),
                n: (0..n).map(|_| AtomicCell::new(n0)).collect(),
                spiked: (0..n).map(|_| AtomicCell::new(false)).collect(),
                above_thresh: (0..n).map(|_| AtomicCell::new(false)).collect(),
            },
            spike_thresh: (0..n).map(|_| AtomicCell::new(thresh)).collect(),
            params: vec![params; n],
            n,
        }
    }

    pub fn heterogeneous(n: usize, base: HHParams, noise_frac: f64, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let v_rest = -65.0f64;
        let (m0, h0, n0) = steady_state(v_rest);
        let thresh = base.v_spike_thresh as f32;

        let state = HHNeuronState {
            v: (0..n)
                .map(|_| AtomicCell::new(v_rest + rng.gen_range(-2.0..2.0)))
                .collect(),
            m: (0..n).map(|_| AtomicCell::new(m0)).collect(),
            h: (0..n).map(|_| AtomicCell::new(h0)).collect(),
            n: (0..n).map(|_| AtomicCell::new(n0)).collect(),
            spiked: (0..n).map(|_| AtomicCell::new(false)).collect(),
            above_thresh: (0..n).map(|_| AtomicCell::new(false)).collect(),
        };
        let params: Vec<HHParams> = (0..n)
            .map(|_| {
                let mut p = base.clone();
                p.g_na *= 1.0 + rng.gen_range(-noise_frac..noise_frac);
                p.g_k *= (1.0 + rng.gen_range(-noise_frac..noise_frac)).max(0.5);
                p
            })
            .collect();

        Self {
            state,
            spike_thresh: (0..n).map(|_| AtomicCell::new(thresh)).collect(),
            params,
            n,
        }
    }

    pub fn read_v(&self, idx: usize) -> f64 {
        self.state.v[idx].load()
    }
    pub fn local_spiked(&self, idx: usize) -> bool {
        self.state.spiked[idx].load()
    }
    pub fn snapshot_v(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.read_v(i)).collect()
    }
}

impl NeuronPopulation for HHPopulation {
    fn len(&self) -> usize {
        self.n
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition> {
        (0..self.n.div_ceil(chunk))
            .map(|p| {
                let start = p * chunk;
                NeuronPartition {
                    start_index: start,
                    len: (start + chunk).min(self.n) - start,
                }
            })
            .collect()
    }

    fn step_range(&self, input_current: &[f32], start: usize) {
        for (local_i, &i_coarse) in input_current.iter().enumerate() {
            let idx = start + local_i;
            let p = &self.params[idx];
            let sub = (1.0 / p.dt).ceil() as usize;
            let dt = 1.0 / sub as f64;
            let i_ext = i_coarse as f64;

            self.state.spiked[idx].store(false);
            let was_above = self.state.above_thresh[idx].load();

            let mut v = self.state.v[idx].load();
            let mut m = self.state.m[idx].load();
            let mut h = self.state.h[idx].load();
            let mut n = self.state.n[idx].load();
            let mut crossed = false;

            for _ in 0..sub {
                let i_na = p.g_na * m * m * m * h * (v - p.e_na);
                let i_k = p.g_k * n * n * n * n * (v - p.e_k);
                let i_l = p.g_l * (v - p.e_l);
                v += dt * (i_ext - i_na - i_k - i_l) / p.c_m;

                let (am, bm) = alpha_beta_m(v);
                let (ah, bh) = alpha_beta_h(v);
                let (an, bn) = alpha_beta_n(v);
                m = (m + dt * (am * (1.0 - m) - bm * m)).clamp(0.0, 1.0);
                h = (h + dt * (ah * (1.0 - h) - bh * h)).clamp(0.0, 1.0);
                n = (n + dt * (an * (1.0 - n) - bn * n)).clamp(0.0, 1.0);

                if !was_above && v >= p.v_spike_thresh {
                    crossed = true;
                }
            }

            self.state.v[idx].store(v.clamp(-100.0, 60.0));
            self.state.m[idx].store(m);
            self.state.h[idx].store(h);
            self.state.n[idx].store(n);
            self.state.above_thresh[idx].store(v >= p.v_spike_thresh);
            self.state.spiked[idx].store(crossed);
        }
    }

    fn local_spiked(&self, idx: usize) -> bool {
        self.state.spiked[idx].load()
    }
    fn read_v(&self, idx: usize) -> f32 {
        self.state.v[idx].load() as f32
    }
    fn get_threshold(&self, idx: usize) -> f32 {
        self.spike_thresh[idx].load()
    }
    fn set_threshold(&self, idx: usize, v: f32) {
        self.spike_thresh[idx].store(v);
    }

    fn reset_neuron(&self, idx: usize, v_rest: f32) {
        let vr = v_rest as f64;
        let (m0, h0, n0) = steady_state(vr);
        self.state.v[idx].store(vr);
        self.state.m[idx].store(m0);
        self.state.h[idx].store(h0);
        self.state.n[idx].store(n0);
        self.state.spiked[idx].store(false);
        self.state.above_thresh[idx].store(false);
    }
}

// ── HH rate functions ─────────────────────────────────────────────────────────

fn alpha_beta_m(v: f64) -> (f64, f64) {
    let dv = v + 40.0;
    let alpha = if dv.abs() < 1e-7 {
        1.0
    } else {
        0.1 * dv / (1.0 - (-dv / 10.0).exp())
    };
    (alpha, 4.0 * (-(v + 65.0) / 18.0).exp())
}
fn alpha_beta_h(v: f64) -> (f64, f64) {
    (
        0.07 * (-(v + 65.0) / 20.0).exp(),
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp()),
    )
}
fn alpha_beta_n(v: f64) -> (f64, f64) {
    let dv = v + 55.0;
    let alpha = if dv.abs() < 1e-7 {
        0.1
    } else {
        0.01 * dv / (1.0 - (-dv / 10.0).exp())
    };
    (alpha, 0.125 * (-(v + 65.0) / 80.0).exp())
}

/// Steady-state gating variables at voltage `v` (mV).
pub fn steady_state(v: f64) -> (f64, f64, f64) {
    let (am, bm) = alpha_beta_m(v);
    let (ah, bh) = alpha_beta_h(v);
    let (an, bn) = alpha_beta_n(v);
    (am / (am + bm), ah / (ah + bh), an / (an + bn))
}
