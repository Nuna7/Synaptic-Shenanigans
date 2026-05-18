//! Izhikevich neuron model — 6 firing patterns, low computational cost.
//!
//!   dv/dt = 0.04v² + 5v + 140 - u + I
//!   du/dt = a(bv - u)
//!   if v ≥ 30: v ← c,  u ← u + d

use super::{NeuronPartition, NeuronPopulation};
use crossbeam::atomic::AtomicCell;
use std::any::Any;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronType {
    RegularSpiking,
    IntrinsicallyBursting,
    Chattering,
    FastSpiking,
    LowThresholdSpiking,
    Resonator,
    Custom { a: f32, b: f32, c: f32, d: f32 },
}

impl NeuronType {
    pub fn params(self) -> (f32, f32, f32, f32) {
        match self {
            Self::RegularSpiking => (0.02, 0.20, -65.0, 8.0),
            Self::IntrinsicallyBursting => (0.02, 0.20, -55.0, 4.0),
            Self::Chattering => (0.02, 0.20, -50.0, 2.0),
            Self::FastSpiking => (0.10, 0.20, -65.0, 2.0),
            Self::LowThresholdSpiking => (0.02, 0.25, -65.0, 2.0),
            Self::Resonator => (0.10, 0.26, -65.0, 2.0),
            Self::Custom { a, b, c, d } => (a, b, c, d),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::RegularSpiking => "RS",
            Self::IntrinsicallyBursting => "IB",
            Self::Chattering => "CH",
            Self::FastSpiking => "FS",
            Self::LowThresholdSpiking => "LTS",
            Self::Resonator => "RZ",
            Self::Custom { .. } => "Custom",
        }
    }
}

pub struct IzhikevichPop {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d: Vec<f32>,
    pub dt: Vec<f32>,
    pub v: Vec<AtomicCell<f32>>,
    pub u: Vec<AtomicCell<f32>>,
    pub spiked: Vec<AtomicCell<bool>>,
    pub neuron_types: Vec<NeuronType>,
    // spike threshold (fixed at 30 mV for Izhikevich, stored for trait compat)
    spike_thresh: Vec<AtomicCell<f32>>,
}

impl IzhikevichPop {
    pub fn homogeneous(n: usize, neuron_type: NeuronType, dt: f32) -> Self {
        let (_a, _b, c, d) = neuron_type.params();
        let spec: Vec<_> = (0..n).map(|_| (neuron_type, dt, c, d)).collect();
        Self::heterogeneous(&spec)
    }

    pub fn heterogeneous(spec: &[(NeuronType, f32, f32, f32)]) -> Self {
        let n = spec.len();
        let mut a_v = Vec::with_capacity(n);
        let mut b_v = Vec::with_capacity(n);
        let mut c_v = Vec::with_capacity(n);
        let mut d_v = Vec::with_capacity(n);
        let mut dt_v = Vec::with_capacity(n);
        let mut nt_v = Vec::with_capacity(n);

        for &(nt, dt, c_override, d_override) in spec {
            let (a, b, _, _) = nt.params();
            a_v.push(a);
            b_v.push(b);
            c_v.push(c_override);
            d_v.push(d_override);
            dt_v.push(dt);
            nt_v.push(nt);
        }

        let v_init: Vec<AtomicCell<f32>> = c_v.iter().map(|&c| AtomicCell::new(c)).collect();
        let u_init: Vec<AtomicCell<f32>> = b_v
            .iter()
            .zip(c_v.iter())
            .map(|(&b, &c)| AtomicCell::new(b * c))
            .collect();

        Self {
            spike_thresh: (0..n).map(|_| AtomicCell::new(30.0)).collect(),
            a: a_v,
            b: b_v,
            c: c_v,
            d: d_v,
            dt: dt_v,
            v: v_init,
            u: u_init,
            spiked: (0..n).map(|_| AtomicCell::new(false)).collect(),
            neuron_types: nt_v,
        }
    }

    pub fn mixed_cortical(n_exc: usize, n_inh: usize, dt: f32, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut spec = Vec::with_capacity(n_exc + n_inh);
        for _ in 0..n_exc {
            let r: f32 = rng.gen_range(0.0..1.0);
            spec.push((
                NeuronType::RegularSpiking,
                dt,
                -65.0 + 15.0 * r * r,
                8.0 - 6.0 * r * r,
            ));
        }
        for _ in 0..n_inh {
            let r: f32 = rng.gen_range(0.0..1.0);
            spec.push((
                NeuronType::Custom {
                    a: 0.02 + 0.08 * r,
                    b: 0.25 - 0.05 * r,
                    c: -65.0,
                    d: 2.0,
                },
                dt,
                -65.0,
                2.0,
            ));
        }
        Self::heterogeneous(&spec)
    }

    pub fn read_v(&self, idx: usize) -> f32 {
        self.v[idx].load()
    }
    pub fn read_u(&self, idx: usize) -> f32 {
        self.u[idx].load()
    }
    pub fn local_spiked(&self, idx: usize) -> bool {
        self.spiked[idx].load()
    }
    pub fn snapshot_v(&self) -> Vec<f32> {
        (0..self.len()).map(|i| self.read_v(i)).collect()
    }
}

impl NeuronPopulation for IzhikevichPop {
    fn len(&self) -> usize {
        self.v.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition> {
        let n = self.len();
        (0..n.div_ceil(chunk))
            .map(|p| {
                let start = p * chunk;
                NeuronPartition {
                    start_index: start,
                    len: (start + chunk).min(n) - start,
                }
            })
            .collect()
    }

    fn step_range(&self, input_current: &[f32], start: usize) {
        let sub_steps = 4usize;
        for (local_i, &i_ext) in input_current.iter().enumerate() {
            let idx = start + local_i;
            let a = self.a[idx];
            let b = self.b[idx];
            let c = self.c[idx];
            let d = self.d[idx];
            let dt_sub = self.dt[idx] / sub_steps as f32;

            self.spiked[idx].store(false);
            let mut v = self.v[idx].load();
            let mut u = self.u[idx].load();
            let mut fired = false;

            for _ in 0..sub_steps {
                v += dt_sub * (0.04 * v * v + 5.0 * v + 140.0 - u + i_ext);
                u += dt_sub * a * (b * v - u);
                if v >= 30.0 {
                    fired = true;
                    v = c;
                    u += d;
                    break;
                }
            }

            self.v[idx].store(v.clamp(-90.0, 35.0));
            self.u[idx].store(u);
            if fired {
                self.spiked[idx].store(true);
            }
        }
    }

    fn local_spiked(&self, idx: usize) -> bool {
        self.spiked[idx].load()
    }
    fn read_v(&self, idx: usize) -> f32 {
        self.v[idx].load()
    }
    fn get_threshold(&self, idx: usize) -> f32 {
        self.spike_thresh[idx].load()
    }
    fn set_threshold(&self, idx: usize, v: f32) {
        self.spike_thresh[idx].store(v);
    }

    fn reset_neuron(&self, idx: usize, v_rest: f32) {
        if idx < self.v.len() {
            self.v[idx].store(v_rest);
            self.u[idx].store(self.b[idx] * v_rest);
            self.spiked[idx].store(false);
        }
    }
}
