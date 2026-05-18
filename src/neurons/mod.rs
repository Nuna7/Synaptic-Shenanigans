//! Neuron Models subsystem.
//!
//! This module defines the single [`NeuronPopulation`] trait that **all**
//! neuron models implement, and re-exports every concrete type.
//!
//! [`Simulation`] stores `Arc<dyn NeuronPopulation>`, so any model can be
//! dropped in without changing the engine.
//!
//! # Adding a new neuron model
//! 1. Create `src/neurons/my_model.rs` and implement [`NeuronPopulation`].
//! 2. Add `pub mod my_model;` and a re-export below.
//! 3. Done — `Simulation::new_with_neurons` accepts it immediately.

pub mod lif;
pub mod izhikevich;
pub mod hh;
pub mod adex;

pub use lif::LifNeuron;
pub use izhikevich::{IzhikevichPop, NeuronType};
pub use hh::{HHPopulation, HHParams};
pub use adex::{AdExPopulation, AdExProfile, AdExParams};

// ── Shared helper ─────────────────────────────────────────────────────────────

/// A contiguous range of global neuron indices assigned to one worker thread.
#[derive(Clone, Copy, Debug)]
pub struct NeuronPartition {
    pub start_index: usize,
    pub len: usize,
}

use std::any::Any;

// ── The unified trait ─────────────────────────────────────────────────────────

/// Unified interface implemented by every neuron population.
///
/// All methods work on **global** neuron indices or index-relative input slices
/// so that the simulation engine can assign disjoint ranges to threads without
/// knowing the concrete model type.
///
/// The trait is **object-safe** (no generics, no `Self` returns) which lets
/// `Simulation` store `Arc<dyn NeuronPopulation>`.
pub trait NeuronPopulation: Any + Send + Sync {
    // ── Size ──────────────────────────────────────────────────────────────────
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn as_any(&self) -> &dyn Any;

    // ── Partitioning ──────────────────────────────────────────────────────────
    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition>;

    // ── Integration ───────────────────────────────────────────────────────────
    /// Integrate one timestep for neurons `[start, start+input_current.len())`.
    fn step_range(&self, input_current: &[f32], start: usize);

    // ── State queries ─────────────────────────────────────────────────────────
    fn local_spiked(&self, idx: usize) -> bool;
    fn read_v(&self, idx: usize) -> f32;
    fn snapshot_v(&self) -> Vec<f32> {
        (0..self.len()).map(|i| self.read_v(i)).collect()
    }
    /// Firing threshold of neuron `idx` (mV). Returns spike-detect threshold
    /// for models without an explicit threshold.
    fn get_threshold(&self, idx: usize) -> f32;
    fn get_thresholds(&self) -> Vec<f32> {
        (0..self.len()).map(|i| self.get_threshold(i)).collect()
    }

    // ── State mutation ────────────────────────────────────────────────────────
    /// Set firing threshold of neuron `idx` (mV). Used by homeostatic plasticity.
    fn set_threshold(&self, idx: usize, v_thresh: f32);

    /// Reset neuron `idx` to resting state at potential `v_rest`.
    /// Clears refractory, recovery variables, etc.
    fn reset_neuron(&self, idx: usize, v_rest: f32);
}