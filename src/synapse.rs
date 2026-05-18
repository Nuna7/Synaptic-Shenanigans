#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Full synapse struct used by the Simulation engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Synapse {
    pub pre: Vec<usize>,
    pub post: Vec<usize>,
    pub weight: Vec<f32>,
    pub delay: Vec<f32>,
    pub tau_syn: Vec<f32>,
    pub model_type: Vec<u8>, // 0=current, 1=conductance, 2=AMPA, 3=NMDA, 4=GABA_A, 5=GABA_B
    pub e_rev: Vec<f32>,
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
            pre: vec![],
            post: vec![],
            weight: vec![],
            delay: vec![],
            tau_syn: vec![],
            model_type: vec![],
            e_rev: vec![],
            spike_queue: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.pre.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pre.is_empty()
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
        self.spike_queue
            .push(VecDeque::from(vec![false; delay_steps]));
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
        self.spike_queue
            .push(VecDeque::from(vec![false; delay_steps]));
    }

    pub fn add_ampa(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        let ds = delay.ceil() as usize;
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(2.0);
        self.model_type.push(2);
        self.e_rev.push(0.0);
        self.spike_queue.push(VecDeque::from(vec![false; ds]));
    }
    pub fn add_nmda(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        let ds = delay.ceil() as usize;
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(100.0);
        self.model_type.push(3);
        self.e_rev.push(0.0);
        self.spike_queue.push(VecDeque::from(vec![false; ds]));
    }
    pub fn add_gaba_a(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        let ds = delay.ceil() as usize;
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(6.0);
        self.model_type.push(4);
        self.e_rev.push(-70.0);
        self.spike_queue.push(VecDeque::from(vec![false; ds]));
    }
    pub fn add_gaba_b(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        let ds = delay.ceil() as usize;
        self.pre.push(pre);
        self.post.push(post);
        self.weight.push(weight);
        self.delay.push(delay);
        self.tau_syn.push(150.0);
        self.model_type.push(5);
        self.e_rev.push(-90.0);
        self.spike_queue.push(VecDeque::from(vec![false; ds]));
    }

    pub fn build_pre_index(&self, n_neurons: usize) -> Vec<Vec<usize>> {
        let mut idx = vec![Vec::new(); n_neurons];
        for (i, &pre) in self.pre.iter().enumerate() {
            if pre < n_neurons {
                idx[pre].push(i);
            }
        }
        idx
    }
}

pub fn synapse_current(model_type: u8, weight: f32, v_post: f32, e_rev: f32) -> f32 {
    match model_type {
        0 => weight,
        1..=5 => weight * (e_rev - v_post),
        _ => 0.0,
    }
}

/// Lightweight matrix produced by topology generators.
/// Convert to [`Synapse`] via [`SynapseMatrix::into_synapse`].
#[derive(Clone, Debug, Default)]
pub struct SynapseMatrix {
    pub pre_neurons: Vec<u32>,
    pub post_neurons: Vec<u32>,
    pub weights: Vec<f32>,
    pub delays: Vec<f32>,
}

impl SynapseMatrix {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.pre_neurons.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pre_neurons.is_empty()
    }

    pub fn into_synapse(self, tau_syn: f32) -> Synapse {
        let mut s = Synapse::new();
        for i in 0..self.pre_neurons.len() {
            let d = self.delays[i];
            s.add_current_based(
                self.pre_neurons[i] as usize,
                self.post_neurons[i] as usize,
                self.weights[i],
                d,
                tau_syn,
                d.ceil() as usize,
            );
        }
        s
    }
}
