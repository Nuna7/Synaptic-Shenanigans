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
}
