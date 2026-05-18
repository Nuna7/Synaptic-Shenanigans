//! Leaky Integrate-and-Fire (LIF) neuron population.
//!
//! The simplest spiking neuron model. Best choice for large-scale network
//! simulations where speed matters and spike shape is unimportant.
//!
//!   τ_m · dV/dt = -(V - V_rest) + R_m · I_ext
//!   if V ≥ V_thresh: spike, V ← V_rest, enter refractory for T_ref ms

use crossbeam::atomic::AtomicCell;
use super::{NeuronPartition, NeuronPopulation};
use std::any::Any;

pub struct LifNeuron {
    pub v:                 Vec<AtomicCell<f32>>,
    pub v_rest:            Vec<f32>,
    pub tau_m:             Vec<f32>,
    pub v_thresh:          Vec<AtomicCell<f32>>,   // mutable for homeostasis
    pub r_m:               Vec<f32>,
    pub dt:                Vec<f32>,
    pub spiked:            Vec<AtomicCell<bool>>,
    pub refractory:        Vec<AtomicCell<bool>>,
    pub refractory_timer:  Vec<AtomicCell<f32>>,
    pub refractory_period: Vec<f32>,
}

impl LifNeuron {
    pub fn new(
        n: usize,
        v_rest: f32,
        v_thresh: f32,
        tau_m: f32,
        r_m: f32,
        dt: f32,
        refract_period: f32,
    ) -> Self {
        Self {
            v:                 (0..n).map(|_| AtomicCell::new(v_rest)).collect(),
            v_rest:            vec![v_rest; n],
            tau_m:             vec![tau_m; n],
            v_thresh:          (0..n).map(|_| AtomicCell::new(v_thresh)).collect(),
            r_m:               vec![r_m; n],
            dt:                vec![dt; n],
            spiked:            (0..n).map(|_| AtomicCell::new(false)).collect(),
            refractory:        (0..n).map(|_| AtomicCell::new(false)).collect(),
            refractory_timer:  (0..n).map(|_| AtomicCell::new(0.0)).collect(),
            refractory_period: vec![refract_period; n],
        }
    }

    pub fn read_v(&self, idx: usize) -> f32      { self.v[idx].load() }
    pub fn local_spiked(&self, idx: usize) -> bool { self.spiked[idx].load() }
}

impl NeuronPopulation for LifNeuron {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn len(&self) -> usize { self.v.len() }

    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition> {
        let n = self.v.len();
        (0..n.div_ceil(chunk)).map(|p| {
            let start = p * chunk;
            NeuronPartition { start_index: start, len: (start + chunk).min(n) - start }
        }).collect()
    }

    fn step_range(&self, input_current: &[f32], start: usize) {
        for (local_i, &i_ext) in input_current.iter().enumerate() {
            let i   = start + local_i;
            let dt  = self.dt[i];

            self.spiked[i].store(false);

            if self.refractory[i].load() {
                let new_timer = self.refractory_timer[i].load() - dt;
                if new_timer <= 0.0 {
                    self.refractory[i].store(false);
                    self.refractory_timer[i].store(0.0);
                } else {
                    self.refractory_timer[i].store(new_timer);
                    self.v[i].store(self.v_rest[i]);
                    continue;
                }
            }

            let v_old   = self.v[i].load();
            let v_thresh = self.v_thresh[i].load();
            let dv = (-(v_old - self.v_rest[i]) + self.r_m[i] * i_ext)
                     * (dt / self.tau_m[i]);
            let v_new = v_old + dv;
            self.v[i].store(v_new);

            if v_new >= v_thresh {
                self.v[i].store(self.v_rest[i]);
                self.spiked[i].store(true);
                self.refractory[i].store(true);
                self.refractory_timer[i].store(self.refractory_period[i]);
            }
        }
    }

    fn local_spiked(&self, idx: usize) -> bool { self.spiked[idx].load() }
    fn read_v(&self, idx: usize) -> f32        { self.v[idx].load() }
    fn get_threshold(&self, idx: usize) -> f32 { self.v_thresh[idx].load() }

    fn set_threshold(&self, idx: usize, v_thresh: f32) {
        if idx < self.v_thresh.len() {
            self.v_thresh[idx].store(v_thresh);
        }
    }

    fn reset_neuron(&self, idx: usize, v_rest: f32) {
        if idx < self.v.len() {
            self.v[idx].store(v_rest);
            self.spiked[idx].store(false);
            self.refractory[idx].store(false);
            self.refractory_timer[idx].store(0.0);
        }
    }
}