//! Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
//!
//! Bridges LIF simplicity and Hodgkin-Huxley realism. Two features LIF
//! cannot produce:
//!   1. Exponential spike initiation — AP upswing matches cortical recordings.
//!   2. Spike-frequency adaptation — firing rate decreases over sustained input.
//!
//! Equations:
//!   C dV/dt = -g_L(V-E_L) + g_L·Δ_T·exp((V-V_T)/Δ_T) - w + I
//!   τ_w dw/dt = a(V-E_L) - w
//!   on spike: V ← V_r,  w ← w + b
//!
//! Reference: Brette & Gerstner (2005). J. Neurophysiol., 94, 3637–3642.

use crossbeam::atomic::AtomicCell;
use crate::lif::{NeuronPartition, NeuronPopulation};
use serde::{Deserialize, Serialize};

/// AdEx parameter set (all values are per-neuron for heterogeneous pops).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdExParams {
    pub c_m:     f32,  // membrane capacitance (pF)   — default 200
    pub g_l:     f32,  // leak conductance (nS)        — default 10
    pub e_l:     f32,  // leak/rest potential (mV)     — default -70
    pub v_t:     f32,  // spike threshold (mV)         — default -50
    pub delta_t: f32,  // slope factor (mV)            — default 2
    pub a:       f32,  // sub-threshold adaptation (nS)— default 4
    pub b:       f32,  // spike adaptation increment (pA)— default 80
    pub tau_w:   f32,  // adaptation time constant (ms)— default 100
    pub v_r:     f32,  // reset voltage (mV)           — default -58
    pub v_peak:  f32,  // spike clip (mV)              — default 0
    pub dt:      f32,  // sub-step (ms)                — default 0.1
}

impl Default for AdExParams {
    fn default() -> Self {
        Self {
            c_m: 200.0, g_l: 10.0, e_l: -70.0,
            v_t: -50.0, delta_t: 2.0,
            a: 4.0, b: 80.0, tau_w: 100.0,
            v_r: -58.0, v_peak: 0.0, dt: 0.1,
        }
    }
}

/// Named parameter presets calibrated to published recordings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdExProfile {
    AdaptingRS,     // most cortical pyramidal cells
    Bursting,       // burst followed by adaptation
    TonicRS,        // no adaptation, tonic firing
    FastSpiking,    // interneuron, minimal adaptation
    TransientBurst, // initial burst then silence
}

impl AdExProfile {
    pub fn params(self) -> AdExParams {
        let base = AdExParams::default();
        match self {
            Self::AdaptingRS     => AdExParams { a:  4.0, b:  80.0, tau_w: 150.0, ..base },
            Self::Bursting       => AdExParams { a: -0.5, b:   7.0, tau_w: 300.0, v_r: -47.0, ..base },
            Self::TonicRS        => AdExParams { a:  0.0, b:   0.0, tau_w: 100.0, ..base },
            Self::FastSpiking    => AdExParams { a:  0.0, b:   0.0, tau_w: 10.0,
                                                 delta_t: 0.5, c_m: 100.0, ..base },
            Self::TransientBurst => AdExParams { a:  4.0, b: 200.0, tau_w: 500.0, ..base },
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::AdaptingRS     => "Adapting-RS",
            Self::Bursting       => "Bursting",
            Self::TonicRS        => "Tonic-RS",
            Self::FastSpiking    => "Fast-Spiking",
            Self::TransientBurst => "Transient-Burst",
        }
    }
}

pub struct AdExPopulation {
    pub c_m: Vec<f32>, pub g_l: Vec<f32>, pub e_l: Vec<f32>,
    pub v_t: Vec<f32>, pub delta_t: Vec<f32>,
    pub a: Vec<f32>,   pub b: Vec<f32>,   pub tau_w: Vec<f32>,
    pub v_r: Vec<f32>, pub v_peak: Vec<f32>, pub dt: Vec<f32>,

    pub v:      Vec<AtomicCell<f32>>,
    pub w:      Vec<AtomicCell<f32>>,  // adaptation current
    pub spiked: Vec<AtomicCell<bool>>,
    n: usize,
}

impl AdExPopulation {
    pub fn homogeneous(n: usize, p: AdExParams) -> Self {
        Self {
            c_m: vec![p.c_m; n], g_l: vec![p.g_l; n], e_l: vec![p.e_l; n],
            v_t: vec![p.v_t; n], delta_t: vec![p.delta_t; n],
            a: vec![p.a; n], b: vec![p.b; n], tau_w: vec![p.tau_w; n],
            v_r: vec![p.v_r; n], v_peak: vec![p.v_peak; n], dt: vec![p.dt; n],
            v:      (0..n).map(|_| AtomicCell::new(p.e_l)).collect(),
            w:      (0..n).map(|_| AtomicCell::new(0.0f32)).collect(),
            spiked: (0..n).map(|_| AtomicCell::new(false)).collect(),
            n,
        }
    }

    pub fn from_profile(n: usize, profile: AdExProfile) -> Self {
        Self::homogeneous(n, profile.params())
    }

    pub fn heterogeneous(n: usize, profile: AdExProfile, noise: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let base = profile.params();
        let mut pop = Self::homogeneous(n, base.clone());
        
        for i in 0..n {
            pop.v_r[i] = base.v_r + rng.gen_range(-2.0..2.0);

            pop.b[i] = (base.b * (1.0 + rng.gen_range(-noise..noise))).max(0.0);

            pop.tau_w[i] =
                (base.tau_w * (1.0 + rng.gen_range(-noise..noise))).max(1.0);

            pop.v[i] = AtomicCell::new(
                base.e_l + rng.gen_range(-1.0..1.0)
            );
        }
        pop
    }

    pub fn read_v(&self, idx: usize) -> f32      { self.v[idx].load() }
    pub fn read_w(&self, idx: usize) -> f32      { self.w[idx].load() }
    pub fn local_spiked(&self, idx: usize) -> bool { self.spiked[idx].load() }
    pub fn snapshot_v(&self) -> Vec<f32> { (0..self.n).map(|i| self.read_v(i)).collect() }
    pub fn snapshot_w(&self) -> Vec<f32> { (0..self.n).map(|i| self.read_w(i)).collect() }
}

impl NeuronPopulation for AdExPopulation {
    fn len(&self) -> usize { self.n }

    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition> {
        (0..self.n.div_ceil(chunk)).map(|p| {
            let start = p * chunk;
            NeuronPartition { start_index: start, len: (start + chunk).min(self.n) - start }
        }).collect()
    }

    fn step_range(&self, input_current: &[f32], start: usize) {
        for (local_i, &i_ext) in input_current.iter().enumerate() {
            let idx     = start + local_i;
            let sub_n   = (1.0 / self.dt[idx]).ceil() as usize;
            let dt_s    = 1.0 / sub_n as f32;
            let c_m     = self.c_m[idx];
            let g_l     = self.g_l[idx];
            let e_l     = self.e_l[idx];
            let v_t     = self.v_t[idx];
            let delta_t = self.delta_t[idx];
            let a_s     = self.a[idx];
            let b_inc   = self.b[idx];
            let tau_w   = self.tau_w[idx];
            let v_r     = self.v_r[idx];
            let v_peak  = self.v_peak[idx];

            self.spiked[idx].store(false);
            let mut v = self.v[idx].load();
            let mut w = self.w[idx].load();
            let mut fired = false;

            for _ in 0..sub_n {
                let exp_term = if delta_t > 1e-6 {
                    g_l * delta_t * (((v - v_t) / delta_t).min(20.0)).exp()
                } else { 0.0 };
                v += dt_s * (-g_l * (v - e_l) + exp_term - w + i_ext) / c_m;
                w += dt_s * (a_s * (v - e_l) - w) / tau_w;
                if v >= v_peak { fired = true; v = v_r; w += b_inc; break; }
            }

            self.v[idx].store(v.max(-100.0).min(v_peak + 5.0));
            self.w[idx].store(w);
            self.spiked[idx].store(fired);
        }
    }
}