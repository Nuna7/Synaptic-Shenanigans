//! Network topology generators for building structured spiking neural networks.
//!
//! All generators are deterministic (seeded) and return a `Synapse` ready to
//! attach to a `Simulation`. Three classical topologies are provided:
//!
//! | Topology      | Description                                        | Biological analogue             |
//! |---------------|----------------------------------------------------|---------------------------------|
//! | Erdős-Rényi   | Each edge exists with probability `p`              | Random cortical connectivity    |
//! | Watts-Strogatz| Ring lattice + random rewiring                     | Small-world cortical networks   |
//! | Barabási-Albert| Preferential attachment / rich-get-richer          | Scale-free hub neurons          |
//!
//! # Example
//! ```rust
//! use synaptic_shenanigans::network::{NetworkBuilder, EdgeParams};
//!
//! let ep = EdgeParams {
//!     weight: 1.0,
//!     delay: 1.5,
//!     ..Default::default()
//! };
//!
//! let syn = NetworkBuilder::small_world(100, 4, 0.1, ep, 42);
//! ```

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha20Rng;

use crate::synapse::Synapse;

/// Common parameters shared by all topology builders.
#[derive(Clone, Debug)]
pub struct EdgeParams {
    /// Synaptic weight (pA for current-based, nS for conductance-based).
    pub weight: f32,
    /// Synaptic transmission delay (ms).
    pub delay: f32,
    /// Fraction of synapses that are inhibitory.
    pub inhibitory_fraction: f32,
    /// Weight multiplier for inhibitory synapses.
    pub inh_weight_scale: f32,
    /// Time constant for synapse dynamics.
    pub tau_syn: f32,
    /// Reversal potential for inhibitory conductances (mV, ignored for current-based).
    pub e_inh: f32,
}

impl EdgeParams {
    pub fn simple(weight: f32, delay: f32) -> Self {
        Self { weight, delay, ..Default::default() }
    }
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            weight: 5.0,
            delay: 1.5,
            inhibitory_fraction: 0.2,
            inh_weight_scale: 4.0,
            tau_syn: 5.0,
            e_inh: -70.0,
        }
    }
}

/// Entry point for all topology generators.
pub struct NetworkBuilder;

impl NetworkBuilder {
    /// **Erdős-Rényi G(n,p)** random graph.
    ///
    /// Each directed edge (i→j, i≠j) is included independently with probability `p`.
    /// Expected degree = (n-1)·p per neuron.
    ///
    /// `ep` — edge parameters (weight, delay, inhibitory fraction).
    /// `seed` — deterministic seed.
    pub fn erdos_renyi(n: usize, p: f64, ep: EdgeParams, seed: u64) -> Synapse {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut syn = Synapse::new();
        let n_inh_cutoff = (n as f32 * ep.inhibitory_fraction) as usize;

        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                if rng.gen_range(0.0..1.0) < p {
                    let is_inh = i < n_inh_cutoff;
                    add_edge(&mut syn, i, j, &ep, is_inh, &mut rng);
                }
            }
        }
        syn
    }

    /// **Watts-Strogatz small-world** network.
    ///
    /// 1. Start with a ring lattice where each neuron connects to `k` nearest neighbours.
    /// 2. Rewire each edge independently with probability `beta`.
    ///
    /// Low `beta` → ordered lattice (high clustering, high path length).  
    /// High `beta` → random graph (low clustering, low path length).  
    /// Sweet-spot around `beta` ≈ 0.1–0.2 gives high clustering AND short paths.
    pub fn small_world(n: usize, k: usize, beta: f64, ep: EdgeParams, seed: u64) -> Synapse {
        assert!(k.is_multiple_of(2), "k must be even for the ring lattice");
        assert!(k < n, "k must be < n");

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut syn = Synapse::new();
        let n_inh_cutoff = (n as f32 * ep.inhibitory_fraction) as usize;

        // 1. Build ring lattice
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            for j in 1..=(k / 2) {
                let target = (i + j) % n;
                edges.push((i, target));
            }
        }

        // 2. Rewire
        for edge in edges.iter_mut() {
            if rng.gen_range(0.0..1.0) < beta {
                let new_target = loop {
                    let t: usize = rng.gen_range(0..n);
                    if t != edge.0 { break t; }
                };
                edge.1 = new_target;
            }
        }

        // 3. Convert to synapses
        for (i, j) in edges {
            let is_inh = i < n_inh_cutoff;
            add_edge(&mut syn, i, j, &ep, is_inh, &mut rng);
        }
        syn
    }

    /// **Barabási-Albert scale-free** network via preferential attachment.
    ///
    /// Neurons are added one at a time; each new neuron forms `m` edges,
    /// preferring to connect to already well-connected neurons (rich-get-richer).
    /// This produces a power-law degree distribution with hub neurons.
    pub fn scale_free(n: usize, m: usize, ep: EdgeParams, seed: u64) -> Synapse {
        assert!(m >= 1, "m must be >= 1");
        assert!(n > m, "n must be > m");

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut syn = Synapse::new();
        let n_inh_cutoff = (n as f32 * ep.inhibitory_fraction) as usize;

        // Track degree for preferential attachment
        let mut degree = vec![0usize; n];
        let mut edges: Vec<(usize, usize)> = Vec::new();

        // Seed with a complete graph of the first m+1 nodes
        for (i, degree_i) in degree.iter_mut().enumerate().take(m + 1) {
            for j in 0..=m {
                if i != j {
                    edges.push((i, j));
                    *degree_i += 1;
                }
            }
        }

        // Preferential attachment for remaining neurons
        for new_node in (m + 1)..n {
            let total_degree: usize = degree[..new_node].iter().sum();
            if total_degree == 0 { continue; }

            let mut targets_chosen: Vec<usize> = Vec::new();
            let mut attempts = 0;

            while targets_chosen.len() < m && attempts < 10_000 {
                attempts += 1;
                let pick: usize = rng.gen_range(0..total_degree);
                let mut cum = 0usize;
                for (candidate, &deg) in degree[..new_node].iter().enumerate() {
                    cum += deg;
                    if cum > pick && !targets_chosen.contains(&candidate) {
                        targets_chosen.push(candidate);
                        break;
                    }
                }
            }

            for target in targets_chosen {
                edges.push((new_node, target));
                edges.push((target, new_node));
                degree[new_node] += 1;
                degree[target] += 1;
            }
        }

        for (i, j) in edges {
            if i < n && j < n {
                let is_inh = i < n_inh_cutoff;
                add_edge(&mut syn, i, j, &ep, is_inh, &mut rng);
            }
        }
        syn
    }

    /// **Ring lattice** (no rewiring). Useful as a baseline.
    pub fn ring(n: usize, k: usize, ep: EdgeParams, seed: u64) -> Synapse {
        Self::small_world(n, k, 0.0, ep, seed)
    }

    /// **Fully connected** (all-to-all). Expensive but useful for small populations.
    pub fn all_to_all(n: usize, ep: EdgeParams, seed: u64) -> Synapse {
        Self::erdos_renyi(n, 1.0, ep, seed)
    }

    /// **Layered feedforward** network.
    ///
    /// Connects neurons in `layers[i]` to neurons in `layers[i+1]` with
    /// probability `p_connect`. Useful for modelling sensory hierarchies.
    pub fn layered_feedforward(
        layers: &[usize],
        p_connect: f64,
        ep: EdgeParams,
        seed: u64,
    ) -> (Synapse, Vec<(usize, usize)>) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut syn = Synapse::new();

        // Build layer start indices
        let mut starts = vec![0usize];
        for &layer_n in layers {
            starts.push(starts.last().unwrap() + layer_n);
        }

        let mut layer_ranges: Vec<(usize, usize)> = Vec::new();
        for i in 0..layers.len() {
            layer_ranges.push((starts[i], starts[i + 1]));
        }

        for layer_idx in 0..layers.len().saturating_sub(1) {
            let (a_start, a_end) = layer_ranges[layer_idx];
            let (b_start, b_end) = layer_ranges[layer_idx + 1];

            for i in a_start..a_end {
                for j in b_start..b_end {
                    if rng.gen_range(0.0..1.0) < p_connect {
                        add_edge(&mut syn, i, j, &ep, false, &mut rng);
                    }
                }
            }
        }

        (syn, layer_ranges)
    }
}

/// Network analysis metrics.
pub struct NetworkMetrics {
    pub n_neurons: usize,
    pub n_synapses: usize,
    pub mean_degree_in: f32,
    pub mean_degree_out: f32,
    pub max_degree_out: usize,
    pub clustering_approx: f32,
}

impl NetworkMetrics {
    /// Compute basic metrics from a built synapse structure.
    pub fn compute(syn: &Synapse, n_neurons: usize) -> Self {
        let mut degree_out = vec![0usize; n_neurons];
        let mut degree_in  = vec![0usize; n_neurons];

        for (&pre, &post) in syn.pre.iter().zip(syn.post.iter()) {
            if pre < n_neurons { degree_out[pre] += 1; }
            if post < n_neurons { degree_in[post] += 1; }
        }

        let mean_out = degree_out.iter().sum::<usize>() as f32 / n_neurons as f32;
        let mean_in  = degree_in.iter().sum::<usize>()  as f32 / n_neurons as f32;
        let max_out  = degree_out.iter().cloned().max().unwrap_or(0);

        Self {
            n_neurons,
            n_synapses: syn.len(),
            mean_degree_in:  mean_in,
            mean_degree_out: mean_out,
            max_degree_out:  max_out,
            clustering_approx: 0.0, // placeholder; full clustering requires triangle counting
        }
    }
}

impl std::fmt::Display for NetworkMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "neurons={} synapses={} mean_in={:.1} mean_out={:.1} max_out={}",
            self.n_neurons, self.n_synapses,
            self.mean_degree_in, self.mean_degree_out, self.max_degree_out
        )
    }
}

// ---- helpers ---------------------------------------------------------------

fn add_edge(
    syn: &mut Synapse,
    pre: usize,
    post: usize,
    ep: &EdgeParams,
    is_inh: bool,
    rng: &mut ChaCha20Rng,
) {
    // Add small jitter to delay for more realistic asynchronous dynamics
    let delay_jitter: f32 = rng.gen_range(-0.2f32..0.2f32);
    let delay = (ep.delay + delay_jitter).max(0.5);

    if is_inh {
        // Inhibitory: conductance-based with reversal potential
        syn.add_conductance_based(
            pre, post,
            ep.weight * ep.inh_weight_scale,
            delay,
            ep.tau_syn,
            ep.e_inh,
            delay.ceil() as usize,
        );
    } else {
        // Excitatory: current-based
        syn.add_current_based(
            pre, post,
            ep.weight,
            delay,
            ep.tau_syn,
            delay.ceil() as usize,
        );
    }
}
