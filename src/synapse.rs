#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Synapse {
    pub pre: Vec<usize>,
    pub post: Vec<usize>,
    pub weight: Vec<f32>,
    pub delay: Vec<f32>,
    pub tau_syn: Vec<f32>,
    pub model_type: Vec<u8>, // 0 = CurrentBased, 1 = ConductanceBased
    pub e_rev: Vec<f32>,     // used only for conductance-based
    pub spike_queue: Vec<VecDeque<bool>>,
}

impl Default for Synapse {
    fn default() -> Self {
        Self::new()
    }
}

impl Synapse {

    pub fn new() -> Self {
        Self {
            pre: Vec::new(),
            post: Vec::new(),
            weight: Vec::new(),
            delay: Vec::new(),
            tau_syn: Vec::new(),
            model_type: Vec::new(),
            e_rev: Vec::new(),
            spike_queue: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add_current_based(
        &mut self,
        pre: usize,
        post: usize,
        weight: f32,
        delay: f32,
        tau_syn: f32,
        delay_steps: usize,
    ) {
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(tau_syn);
        self.model_type.push(0);
        self.e_rev.push(0.0);
        self.spike_queue.push(VecDeque::from(vec![false; delay_steps]));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_conductance_based(
        &mut self,
        pre: usize,
        post: usize,
        weight: f32,
        delay: f32,
        tau_syn: f32,
        e_rev: f32,
        delay_steps: usize,
    ) {
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(tau_syn);
        self.model_type.push(1);
        self.e_rev.push(e_rev);
        self.spike_queue.push(VecDeque::from(vec![false; delay_steps]));
    }

    pub fn len(&self) -> usize {
        self.pre.len()
    }

    pub fn build_pre_index(&self, n_neurons: usize) -> Vec<Vec<usize>> {
        let mut index = vec![Vec::new(); n_neurons];
        for (idx, &pre) in self.pre.iter().enumerate() {
            if pre < n_neurons {
                index[pre].push(idx);
            }
        }
        index
    }

    // ── Synapse model type constants ─────────────────────────────────────────────
    // These are used in `model_type` field of each synapse.
    // 0 = current-based (original)
    // 1 = conductance-based (original)
    // 2 = AMPA  (fast excitatory, τ ≈ 2 ms)
    // 3 = NMDA  (slow excitatory + Mg block, τ ≈ 100 ms)
    // 4 = GABA_A (fast inhibitory, τ ≈ 6 ms, E_rev = -70 mV)
    // 5 = GABA_B (slow inhibitory, τ ≈ 150 ms, E_rev = -90 mV)

    pub fn add_ampa(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(2.0);   // fast AMPA decay
        self.model_type.push(2);  // AMPA
        self.e_rev.push(0.0);     // excitatory reversal
        self.spike_queue.push(VecDeque::from(vec![false; delay.ceil() as usize]));
    }

    pub fn add_nmda(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(100.0); // slow NMDA decay
        self.model_type.push(3);  // NMDA
        self.e_rev.push(0.0);     // excitatory reversal
        self.spike_queue.push(VecDeque::from(vec![false; delay.ceil() as usize]));
    }

    pub fn add_gaba_a(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(6.0);    // fast GABA_A decay
        self.model_type.push(4);   // GABA_A
        self.e_rev.push(-70.0);    // inhibitory reversal
        self.spike_queue.push(VecDeque::from(vec![false; delay.ceil() as usize]));
    }

    pub fn add_gaba_b(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(150.0);  // slow GABA_B decay
        self.model_type.push(5);   // GABA_B
        self.e_rev.push(-90.0);    // hyperpolarising reversal
        self.spike_queue.push(VecDeque::from(vec![false; delay.ceil() as usize]));
    }
}


/// Compute synaptic conductance contribution for a synapse event.
///
/// AMPA/NMDA/GABA_A/GABA_B are all conductance-based; the input current
/// to the post-synaptic neuron is:  I = g_syn · (V_post - E_rev)
///
/// `model_type`: 2 = AMPA, 3 = NMDA, 4 = GABA_A, 5 = GABA_B
/// Returns (current_contribution, is_inhibitory)
pub fn synapse_current(model_type: u8, weight: f32, v_post: f32, e_rev: f32) -> f32 {
    match model_type {
        0 => weight,                           // current-based: I = weight
        1 | 2 | 3 | 4 | 5 => weight * (e_rev - v_post), // conductance-based
        _ => 0.0,
    }
    
}


