//! Izhikevich neuron model — a computationally cheap model that can
//! reproduce a rich variety of cortical firing patterns.
//!
//! Model equations (Euler integration, 1 ms step):
//!   dv/dt = 0.04v² + 5v + 140 - u + I
//!   du/dt = a(bv - u)
//!   if v >= 30 mV: v ← c,  u ← u + d
//!
//! References: Izhikevich (2003), *IEEE Trans. Neural Netw.*

use crossbeam::atomic::AtomicCell;
use crate::lif::{NeuronPartition, NeuronPopulation};

/// Firing patterns produced by the Izhikevich model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronType {
    /// Regular spiking — most common excitatory cortical cells.
    RegularSpiking,
    /// Intrinsically bursting — burst on first stimulus, then RS.
    IntrinsicallyBursting,
    /// Chattering — high-frequency bursts (cat visual cortex).
    Chattering,
    /// Fast spiking — GABAergic interneurons.
    FastSpiking,
    /// Low-threshold spiking — another interneuron class.
    LowThresholdSpiking,
    /// Resonator — responds best to inputs at its natural frequency.
    Resonator,
    /// Custom: pass (a, b, c, d) directly.
    Custom { a: f32, b: f32, c: f32, d: f32 },
}

impl NeuronType {
    /// Return the four Izhikevich parameters for this type.
    pub fn params(self) -> (f32, f32, f32, f32) {
        match self {
            Self::RegularSpiking        => (0.02, 0.20, -65.0, 8.0),
            Self::IntrinsicallyBursting => (0.02, 0.20, -55.0, 4.0),
            Self::Chattering            => (0.02, 0.20, -50.0, 2.0),
            Self::FastSpiking           => (0.10, 0.20, -65.0, 2.0),
            Self::LowThresholdSpiking   => (0.02, 0.25, -65.0, 2.0),
            Self::Resonator             => (0.10, 0.26, -65.0, 2.0),
            Self::Custom { a, b, c, d } => (a, b, c, d),
        }
    }

    /// Human-readable name for logging / CSV output.
    pub fn name(self) -> &'static str {
        match self {
            Self::RegularSpiking        => "RS",
            Self::IntrinsicallyBursting => "IB",
            Self::Chattering            => "CH",
            Self::FastSpiking           => "FS",
            Self::LowThresholdSpiking   => "LTS",
            Self::Resonator             => "RZ",
            Self::Custom { .. }         => "Custom",
        }
    }
}

/// Population of Izhikevich neurons. Fields are per-neuron parameter vectors;
/// mutable state is stored in `AtomicCell` for thread-safe concurrent reads/writes.
pub struct IzhikevichPop {
    // Parameters
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,     // reset voltage (mV)
    pub d: Vec<f32>,     // reset recovery variable increment
    pub dt: Vec<f32>,

    // Mutable state (AtomicCell — no locks needed for single-writer per neuron)
    pub v: Vec<AtomicCell<f32>>,   // membrane potential (mV)
    pub u: Vec<AtomicCell<f32>>,   // recovery variable
    pub spiked: Vec<AtomicCell<bool>>,

    // Neuron type labels (kept for introspection / export)
    pub neuron_types: Vec<NeuronType>,
}

impl IzhikevichPop {
    /// Create a homogeneous population of `n` neurons of the same type.
    pub fn homogeneous(n: usize, neuron_type: NeuronType, dt: f32) -> Self {
        let (_a, _b, c, d) = neuron_type.params();
        Self::heterogeneous(vec![(n, neuron_type, dt, c, d)].into_iter().flat_map(|(count, nt, dt_val, c_val, d_val)| {
            (0..count).map(move |_| (nt, dt_val, c_val, d_val))
        }).collect::<Vec<_>>().as_slice())
    }

    /// Create from an explicit per-neuron specification slice.
    ///
    /// Each entry is `(neuron_type, dt, c_override, d_override)` where the
    /// c/d overrides let you add Gaussian noise to break symmetry.
    pub fn heterogeneous(spec: &[(NeuronType, f32, f32, f32)]) -> Self {
        let n = spec.len();
        let mut a_vec = Vec::with_capacity(n);
        let mut b_vec = Vec::with_capacity(n);
        let mut c_vec = Vec::with_capacity(n);
        let mut d_vec = Vec::with_capacity(n);
        let mut dt_vec = Vec::with_capacity(n);
        let mut nt_vec = Vec::with_capacity(n);

        for &(nt, dt_val, c_override, d_override) in spec {
            let (a, b, _c_base, _d_base) = nt.params();
            a_vec.push(a);
            b_vec.push(b);
            c_vec.push(c_override);
            d_vec.push(d_override);
            dt_vec.push(dt_val);
            nt_vec.push(nt);
        }

        let v_init: Vec<AtomicCell<f32>> = c_vec.iter().map(|&c| AtomicCell::new(c)).collect();
        let u_init: Vec<AtomicCell<f32>> = b_vec.iter().zip(c_vec.iter())
            .map(|(&b, &c)| AtomicCell::new(b * c))
            .collect();

        Self {
            a: a_vec,
            b: b_vec,
            c: c_vec,
            d: d_vec,
            dt: dt_vec,
            v: v_init,
            u: u_init,
            spiked: (0..n).map(|_| AtomicCell::new(false)).collect(),
            neuron_types: nt_vec,
        }
    }

    /// Mixed excitatory / inhibitory cortical population.
    ///
    /// `n_exc` RS neurons, `n_inh` FS neurons, with small parameter noise
    /// to produce biologically realistic asynchronous irregular activity.
    pub fn mixed_cortical(n_exc: usize, n_inh: usize, dt: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut spec = Vec::with_capacity(n_exc + n_inh);

        // Excitatory: RS with noise on c and d
        for _ in 0..n_exc {
            let r: f32 = rng.gen_range(0.0..1.0);
            let c = -65.0 + 15.0 * r * r;
            let d = 8.0 - 6.0 * r * r;
            spec.push((NeuronType::RegularSpiking, dt, c, d));
        }
        // Inhibitory: FS with noise on a and b (encoded via Custom)
        for _ in 0..n_inh {
            let r: f32 = rng.gen_range(0.0..1.0);
            let a = 0.02 + 0.08 * r;
            let b = 0.25 - 0.05 * r;
            spec.push((NeuronType::Custom { a, b, c: -65.0, d: 2.0 }, dt, -65.0, 2.0));
        }

        Self::heterogeneous(&spec)
    }

    pub fn read_v(&self, idx: usize) -> f32 { self.v[idx].load() }
    pub fn read_u(&self, idx: usize) -> f32 { self.u[idx].load() }
    pub fn local_spiked(&self, idx: usize) -> bool { self.spiked[idx].load() }

    /// Export full membrane-potential snapshot (useful for probes).
    pub fn snapshot_v(&self) -> Vec<f32> {
        (0..self.len()).map(|i| self.read_v(i)).collect()
    }
}

impl NeuronPopulation for IzhikevichPop {
    fn len(&self) -> usize { self.v.len() }

    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition> {
        let n = self.len();
        (0..n.div_ceil(chunk)).map(|p| {
            let start = p * chunk;
            let end = (start + chunk).min(n);
            NeuronPartition { start_index: start, len: end - start }
        }).collect()
    }

    /// Integrate one timestep for neurons `[start, start+input_current.len())`.
    ///
    /// Uses **sub-step integration** (4 sub-steps of dt/4) for numerical
    /// stability at large input currents — important for bursting patterns.
    fn step_range(&self, input_current: &[f32], start: usize) {
        let sub_steps = 4usize; // sub-divide each ms for stability

        for (local_i, &i_ext) in input_current.iter().enumerate() {
            let idx = start + local_i;
            let a = self.a[idx];
            let b = self.b[idx];
            let c = self.c[idx];
            let d = self.d[idx];
            let dt_full = self.dt[idx];
            let dt_sub = dt_full / sub_steps as f32;

            self.spiked[idx].store(false);

            let mut v = self.v[idx].load();
            let mut u = self.u[idx].load();

            let mut fired = false;
            for _ in 0..sub_steps {
                // Izhikevich update (mV scale)
                v += dt_sub * (0.04 * v * v + 5.0 * v + 140.0 - u + i_ext);
                u += dt_sub * a * (b * v - u);

                if v >= 30.0 {
                    fired = true;
                    v = c;
                    u += d;
                    break; // spike and reset this ms
                }
            }

            self.v[idx].store(v.clamp(-90.0, 35.0)); // soft clamp for stability
            self.u[idx].store(u);

            if fired {
                self.spiked[idx].store(true);
            }
        }
    }
}
