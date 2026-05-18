//! Core simulation engine.
//!
//! [`Simulation`] is the single runtime object. It holds:
//!   - Any [`NeuronPopulation`] behind `Arc<dyn NeuronPopulation>` — fully modular.
//!   - A [`Synapse`] connection matrix behind `Arc<Synapse>`.
//!   - A min-heap event queue ordered by `(tick, seq)` for determinism.
//!   - A flat spike log `Vec<(time_ms, neuron_id)>`.
//!
//! # Modularity
//! Pass any type that implements [`NeuronPopulation`] to
//! [`Simulation::new_with_neurons`]:
//! ```rust,no_run
//! use synaptic_shenanigans::{Simulation, Synapse};
//! use synaptic_shenanigans::neurons::{IzhikevichPop, NeuronType};
//!
//! let pop = IzhikevichPop::homogeneous(100, NeuronType::FastSpiking, 1.0);
//! let mut sim = Simulation::new_with_neurons(pop, Synapse::new(), 1.0, 42, 1);
//! ```

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use crossbeam::atomic::AtomicCell;

use crate::event::Event;
use crate::neurons::{LifNeuron, NeuronPopulation};
use crate::synapse::Synapse;

const EPS: f32 = 1e-6;

// ── Scheduler mode ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerMode {
    /// Sequential. Fully deterministic. Use for debugging.
    SingleThreaded,
    /// Parallel with post-sort. Bit-identical to `SingleThreaded`.
    Deterministic { n_threads: usize },
    /// Parallel without post-sort. Fastest but non-reproducible.
    /// Requires `--features performance`.
    Performance { n_threads: usize },
}

// ── Thread-local buffers ──────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ThreadLocal {
    pub local_queue:     Vec<Event>,
    pub local_spike_log: Vec<(f32, usize, usize)>, // (time, neuron_id, thread_id)
}

impl Default for ThreadLocal {
    fn default() -> Self { Self { local_queue: Vec::new(), local_spike_log: Vec::new() } }
}

impl ThreadLocal {
    pub fn clear(&mut self) { self.local_queue.clear(); self.local_spike_log.clear(); }
}

// ── SimConfig (C ABI construction) ────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub n_neurons: usize,
    pub n_threads: usize,
    pub seed: u64,
    pub scheduler: u32, // 0=ST, 1=Det, 2=Perf
}

// ── Simulation ────────────────────────────────────────────────────────────────

pub struct Simulation {
    /// Neuron population — any model, fully generic via trait object.
    pub neurons:      Arc<dyn NeuronPopulation>,
    pub synapses:     Arc<Synapse>,
    pub event_queue:  BinaryHeap<Event>,
    pub dt:           f32,
    pub time:         f32,
    pub seed:         u64,
    rng:              ChaCha20Rng,
    pub next_seq:     AtomicU64,
    pub spike_log:    Vec<(f32, usize)>,
    pub thread_locals: Vec<ThreadLocal>,
    pub num_threads:  usize,
    pub pre_index:    Vec<Vec<usize>>,
    pub verbose:      bool,
    pub scheduler_mode: SchedulerMode,
    pub probes:       Vec<Vec<f32>>,
}

impl Simulation {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a simulation with any neuron population type.
    pub fn new_with_neurons<N: NeuronPopulation + 'static>(
        neurons: N,
        synapses: Synapse,
        dt: f32,
        seed: u64,
        num_threads: usize,
    ) -> Self {
        Self::from_arc(Arc::new(neurons), synapses, dt, seed, num_threads)
    }

    /// Convenience constructor for LIF neurons (backward-compatible name).
    pub fn new_with_seed(
        neurons: LifNeuron,
        synapses: Synapse,
        dt: f32,
        seed: u64,
        num_threads: usize,
    ) -> Self {
        Self::new_with_neurons(neurons, synapses, dt, seed, num_threads)
    }

    /// Construct from a pre-built `Arc<dyn NeuronPopulation>`.
    pub fn from_arc(
        neurons: Arc<dyn NeuronPopulation>,
        synapses: Synapse,
        dt: f32,
        seed: u64,
        num_threads: usize,
    ) -> Self {
        let locals: Vec<ThreadLocal> = (0..num_threads).map(|_| ThreadLocal::default()).collect();
        let rng    = ChaCha20Rng::seed_from_u64(seed);
        let arc_syn = Arc::new(synapses);
        let pre_index = arc_syn.build_pre_index(neurons.len());
        Self {
            neurons,
            synapses: arc_syn,
            event_queue: BinaryHeap::new(),
            dt, time: 0.0, seed, rng,
            next_seq: AtomicU64::new(0),
            spike_log: Vec::new(),
            thread_locals: locals,
            num_threads,
            pre_index,
            verbose: false,
            scheduler_mode: SchedulerMode::SingleThreaded,
            probes: Vec::new(),
        }
    }

    /// Create from [`SimConfig`] (used by C ABI).
    pub fn new(cfg: SimConfig) -> Self {
        let sched = match cfg.scheduler {
            1 => SchedulerMode::Deterministic { n_threads: cfg.n_threads },
            2 => SchedulerMode::Performance   { n_threads: cfg.n_threads },
            _ => SchedulerMode::SingleThreaded,
        };
        let neurons = LifNeuron::new(cfg.n_neurons, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
        let mut sim = Self::new_with_seed(neurons, Synapse::new(), 1.0, cfg.seed, cfg.n_threads);
        sim.scheduler_mode = sched;
        sim
    }

    // ── Scheduling ────────────────────────────────────────────────────────────

    fn time_to_tick(&self, t: f32) -> u64 {
        ((t / self.dt) as f64 + 1e-9).floor() as u64
    }

    fn owner_of(&self, neuron_id: usize) -> usize {
        let chunk = self.neurons.len().div_ceil(self.num_threads);
        (neuron_id / chunk).min(self.num_threads - 1)
    }

    /// Advance to `end_time`, choosing the scheduler set in `scheduler_mode`.
    pub fn run_auto(&mut self, end_time: f32) {
        match self.scheduler_mode {
            SchedulerMode::SingleThreaded        => self.run_until(end_time),
            SchedulerMode::Deterministic { .. }  => self.run_deterministic_multithreaded(end_time),
            SchedulerMode::Performance { .. } => {
                #[cfg(feature = "performance")]
                { self.run_performance_multithreaded(end_time); return; }
                #[cfg(not(feature = "performance"))]
                {
                    eprintln!("[synaptic-shenanigans] WARNING: Performance mode \
                               requested but `performance` feature is not enabled. \
                               Falling back to SingleThreaded.");
                    self.run_until(end_time);
                }
            }
        }
    }

    // ── Single-threaded loop ──────────────────────────────────────────────────

    pub fn run_until(&mut self, end_time: f32) {
        let n = self.neurons.len();
        if n == 0 { return; }

        let mut inputs = vec![0.0f32; n];

        while let Some(next_ev) = self.event_queue.pop() {
            if next_ev.time > end_time {
                self.event_queue.push(next_ev);
                break;
            }
            if next_ev.time > self.time + 1e-9 { self.time = next_ev.time; }

            let mut events_at_t = vec![next_ev];
            while let Some(peek) = self.event_queue.peek() {
                if (peek.time - self.time).abs() < 1e-6 {
                    events_at_t.push(self.event_queue.pop().unwrap());
                } else { break; }
            }

            for tl in &mut self.thread_locals { tl.local_queue.clear(); }
            for ev in events_at_t {
                let owner = self.owner_of(ev.target);
                self.thread_locals[owner].local_queue.push(ev);
            }

            let t_threads = self.num_threads;
            let chunk = if t_threads <= 1 { n } else { n.div_ceil(t_threads) };

            for tid in 0..self.num_threads {
                let start = tid * chunk;
                let end   = ((tid+1)*chunk).min(n);
                if start >= end { continue; }

                let mut to_process = Vec::new();
                std::mem::swap(&mut to_process, &mut self.thread_locals[tid].local_queue);
                to_process.sort_by(|a,b| a.tick.cmp(&b.tick).then(a.seq.cmp(&b.seq)).then(a.target.cmp(&b.target)));

                for inp in inputs[start..end].iter_mut() { *inp = 0.0; }
                for ev in &to_process {
                    if ev.target >= start && ev.target < end {
                        let current = inputs[ev.target];
                        inputs[ev.target] = if ev.model_type == 0 {
                            current + ev.weight
                        } else {
                            current + ev.weight * (ev.e_rev - self.neurons.read_v(ev.target))
                        };
                    } else {
                        let owner = self.owner_of(ev.target);
                        if owner != tid { self.thread_locals[owner].local_queue.push(ev.clone()); }
                    }
                }

                self.neurons.step_range(&inputs[start..end], start);

                for nid in start..end {
                    if self.neurons.local_spiked(nid) {
                        self.thread_locals[tid].local_spike_log.push((self.time, nid, tid));
                        self.emit_synaptic_events(nid, end_time);
                    }
                }
            }

            self.merge_queues_into_global();
            self.merge_spike_logs();
        }
    }

    fn emit_synaptic_events(&mut self, nid: usize, end_time: f32) {
        for &s_idx in &self.pre_index[nid] {
            if self.synapses.pre[s_idx] == nid {
                let post       = self.synapses.post[s_idx];
                let weight     = self.synapses.weight[s_idx];
                let delay      = self.synapses.delay[s_idx];
                let model_type = self.synapses.model_type[s_idx];
                let e_rev      = self.synapses.e_rev[s_idx];
                let arrival    = self.time + delay;
                if arrival <= end_time && arrival > self.time + EPS {
                    let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
                    let owner = self.owner_of(post);
                    let tick = self.time_to_tick(arrival);
                    self.thread_locals[owner].local_queue.push(Event {
                        tick,
                        time: arrival, target: post, weight, seq, model_type, e_rev,
                    });
                }
            }
        }
    }

    // ── Deterministic multi-threaded loop ─────────────────────────────────────

    pub fn run_deterministic_multithreaded(&mut self, end_time: f32) {
        if self.num_threads <= 1 { return self.run_until(end_time); }
        let n = self.neurons.len();
        if n == 0 { return; }

        let t       = self.num_threads;
        let chunk   = n.div_ceil(t);
        let pre_idx = Arc::new(self.pre_index.clone());
        let syn     = Arc::clone(&self.synapses);
        let neurons = Arc::clone(&self.neurons);

        while let Some(next_ev) = self.event_queue.pop() {
            if next_ev.time > end_time { self.event_queue.push(next_ev); break; }
            self.time = next_ev.time;

            let mut events_at_t = vec![next_ev];
            while let Some(p) = self.event_queue.peek() {
                if (p.time - self.time).abs() < 1e-6 { events_at_t.push(self.event_queue.pop().unwrap()); }
                else { break; }
            }

            let mut in_queues: Vec<Vec<Event>> = vec![Vec::new(); t];
            for ev in events_at_t { in_queues[self.owner_of(ev.target)].push(ev); }

            let mut partitions = neurons.split_indices(chunk);
            partitions.retain(|p| p.len > 0);
            let real_t = partitions.len();
            let dt     = self.dt;
            let cur    = self.time;

            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(real_t);
                for tid in 0..real_t {
                    let syn       = Arc::clone(&syn);
                    let pre_idx   = Arc::clone(&pre_idx);
                    let neurons   = Arc::clone(&neurons);
                    let in_q      = std::mem::take(&mut in_queues[tid]);
                    let part      = partitions[tid];

                    handles.push(scope.spawn(move || {
                        let mut inputs = vec![0.0f32; part.len];
                        for ev in &in_q {
                            if ev.target >= part.start_index && ev.target < part.start_index + part.len {
                                let li = ev.target - part.start_index;
                                inputs[li] += if ev.model_type == 0 { ev.weight }
                                              else { ev.weight * (ev.e_rev - neurons.read_v(ev.target)) };
                            }
                        }
                        neurons.step_range(&inputs, part.start_index);

                        let mut new_evs: Vec<Event> = Vec::new();
                        let mut logs: Vec<(f32, usize, usize)> = Vec::new();
                        for i in 0..part.len {
                            let g = part.start_index + i;
                            if neurons.local_spiked(g) {
                                logs.push((cur, g, tid));
                                if g < pre_idx.len() {
                                    for &s in &pre_idx[g] {
                                        if syn.pre[s] == g {
                                            let arr = cur + syn.delay[s];
                                            if arr <= cur + 2000.0 && arr > cur + EPS {
                                                new_evs.push(Event {
                                                    tick: (arr / dt) as u64,
                                                    time: arr, target: syn.post[s],
                                                    weight: syn.weight[s], seq: 0,
                                                    model_type: syn.model_type[s],
                                                    e_rev: syn.e_rev[s],
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        (tid, new_evs, logs)
                    }));
                }
                for h in handles {
                    let (tid, new_evs, logs) = h.join().expect("thread panicked");
                    for (t, nid, _) in logs { self.thread_locals[tid].local_spike_log.push((t,nid,tid)); }
                    for mut ev in new_evs {
                        ev.seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
                        let owner = self.owner_of(ev.target);
                        self.thread_locals[owner].local_queue.push(ev);
                    }
                }
            });

            self.merge_queues_into_global();
            self.merge_spike_logs();
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn merge_queues_into_global(&mut self) {
        let mut all: Vec<Event> = self.thread_locals.iter_mut()
            .flat_map(|tl| tl.local_queue.drain(..)).collect();
        all.sort_by(|a,b| a.tick.cmp(&b.tick).then(a.seq.cmp(&b.seq)).then(a.target.cmp(&b.target)));
        for ev in all { self.event_queue.push(ev); }
    }

    fn merge_spike_logs(&mut self) {
        let mut all: Vec<(f32,usize,usize)> = self.thread_locals.iter_mut().enumerate()
            .flat_map(|(tid,tl)| tl.local_spike_log.drain(..).map(move|(t,n,_)|(t,n,tid)))
            .collect();
        all.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for (t,n,_) in all { self.spike_log.push((t,n)); }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn push_event(&mut self, time: f32, target: usize, weight: f32, model_type: u8, e_rev: f32) {
        let tick = self.time_to_tick(time);
        let seq  = self.next_seq.fetch_add(1, Ordering::Relaxed);
        self.event_queue.push(Event { tick, time, target, weight, seq, model_type, e_rev });
    }

    pub fn inject_spike(&mut self, neuron: u32, _weight: f32) {
        let nid = neuron as usize;
        if nid < self.pre_index.len() {
            for &s in &self.pre_index[nid] {
                if self.synapses.pre[s] == nid {
                    let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
                    let arr = self.time + self.synapses.delay[s];
                    self.event_queue.push(Event {
                        tick: self.time_to_tick(arr), time: arr,
                        target: self.synapses.post[s], weight: self.synapses.weight[s],
                        seq, model_type: self.synapses.model_type[s], e_rev: self.synapses.e_rev[s],
                    });
                }
            }
        }
    }

    pub fn get_all_voltages(&self) -> Vec<f32> { self.neurons.snapshot_v() }
    pub fn current_time(&self) -> f32           { self.time }
    pub fn spike_count(&self) -> usize          { self.spike_log.len() }
    pub fn clear_spikes(&mut self)              { self.spike_log.clear(); }

    pub fn step_until(&mut self, until_ms: f32) -> Vec<(f32, usize)> {
        let before = self.spike_log.len();
        self.run_auto(until_ms);
        self.spike_log[before..].to_vec()
    }

    pub fn run_and_log_until(&mut self, end_time: f32) -> Vec<(f32, usize)> {
        self.run_until(end_time);
        self.spike_log.clone()
    }

    pub fn record_probes(&mut self) {
        self.probes.push(self.neurons.snapshot_v());
    }

    pub fn advance_step(&mut self)             { let t = self.time + self.dt; self.run_auto(t); }
    pub fn advance_steps(&mut self, n: usize)  { for _ in 0..n { self.advance_step(); } }

    // ── Checkpoint ────────────────────────────────────────────────────────────

    /// Serialize LIF-specific state to disk (with SHA-256 hash).
    /// For non-LIF populations, voltage snapshot is saved; model-specific
    /// gating variables are not persisted (use the population's own serializer
    /// if needed).
    pub fn save_state(&self, path: &str, hash_path: &str) -> std::io::Result<()> {
        use sha2::{Digest, Sha256};

        // Only LIF neurons are fully serializeable today.
        // We save a common header (time, seq, synapse, pre_index, dt)
        // plus the voltage snapshot that works for all models.
        #[derive(serde::Serialize)]
        struct Snapshot {
            time: f32,
            next_seq: u64,
            v: Vec<f32>,
            synapses: Synapse,
            pre_index: Vec<Vec<usize>>,
            dt: f32,
            // LIF-specific (may be empty for other models)
            v_rest: Vec<f32>,
            tau_m: Vec<f32>,
            v_thresh: Vec<f32>,
            r_m: Vec<f32>,
            dt_vec: Vec<f32>,
            refractory_period: Vec<f32>,
            spiked: Vec<bool>,
            refractory: Vec<bool>,
            refractory_timer: Vec<f32>,
        }

        let n = self.neurons.len();
        // Try to downcast to LifNeuron for full fidelity.
        // (Other models get voltage-only snapshots.)
        let (v_rest, tau_m, v_thresh_v, r_m, dt_vec, refract_period, spiked, refract, refract_t)
            = if let Some(lif) = self.neurons.as_ref().as_any().downcast_ref::<LifNeuron>() {
                (
                    lif.v_rest.clone(),
                    lif.tau_m.clone(),
                    (0..n).map(|i| lif.v_thresh[i].load()).collect(),
                    lif.r_m.clone(),
                    lif.dt.clone(),
                    lif.refractory_period.clone(),
                    (0..n).map(|i| lif.spiked[i].load()).collect(),
                    (0..n).map(|i| lif.refractory[i].load()).collect(),
                    (0..n).map(|i| lif.refractory_timer[i].load()).collect(),
                )
            } else {
                let thresholds = self.neurons.get_thresholds();
                (vec![-65.0; n], vec![20.0; n], thresholds, vec![1.0; n],
                 vec![1.0; n], vec![5.0; n],
                 vec![false; n], vec![false; n], vec![0.0f32; n])
            };

        let snap = Snapshot {
            time: self.time,
            next_seq: self.next_seq.load(Ordering::Relaxed),
            v: self.neurons.snapshot_v(),
            synapses: (*self.synapses).clone(),
            pre_index: self.pre_index.clone(),
            dt: self.dt,
            v_rest, tau_m, v_thresh: v_thresh_v, r_m, dt_vec,
            refractory_period: refract_period,
            spiked, refractory: refract, refractory_timer: refract_t,
        };

        let encoded = bincode::serialize(&snap)
            .map_err(|e| std::io::Error::other(format!("bincode: {e}")))?;
        std::fs::write(path, &encoded)?;
        let digest = Sha256::digest(&encoded);
        std::fs::write(hash_path, hex::encode(digest))?;
        Ok(())
    }

    /// Load a checkpoint. Always restores as a LIF population.
    pub fn load_state(path: &str, seed: u64, num_threads: usize) -> std::io::Result<Simulation> {
        #[derive(serde::Deserialize)]
        struct Snapshot {
            time: f32,
            next_seq: u64,
            v: Vec<f32>,
            synapses: Synapse,
            pre_index: Vec<Vec<usize>>,
            dt: f32,
            v_rest: Vec<f32>,
            tau_m: Vec<f32>,
            v_thresh: Vec<f32>,
            r_m: Vec<f32>,
            dt_vec: Vec<f32>,
            refractory_period: Vec<f32>,
            spiked: Vec<bool>,
            refractory: Vec<bool>,
            refractory_timer: Vec<f32>,
        }

        let bytes = std::fs::read(path)?;
        let snap: Snapshot = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::other(format!("bincode: {e}")))?;

        let neurons = LifNeuron {
            v:                 snap.v.into_iter().map(AtomicCell::new).collect(),
            v_rest:            snap.v_rest,
            tau_m:             snap.tau_m,
            v_thresh:          snap.v_thresh.into_iter().map(AtomicCell::new).collect(),
            r_m:               snap.r_m,
            dt:                snap.dt_vec,
            spiked:            snap.spiked.into_iter().map(AtomicCell::new).collect(),
            refractory:        snap.refractory.into_iter().map(AtomicCell::new).collect(),
            refractory_timer:  snap.refractory_timer.into_iter().map(AtomicCell::new).collect(),
            refractory_period: snap.refractory_period,
        };

        let mut sim = Simulation::new_with_neurons(neurons, snap.synapses, snap.dt, seed, num_threads);
        sim.time = snap.time;
        sim.next_seq.store(snap.next_seq, Ordering::Relaxed);
        sim.pre_index = snap.pre_index;
        Ok(sim)
    }
}

// ── Determinism utility ───────────────────────────────────────────────────────

pub fn replay_equal(build: impl Fn(u64) -> Simulation, end_time: f32, seed: u64) -> bool {
    let mut a = build(seed);
    let mut b = build(seed);
    a.run_and_log_until(end_time) == b.run_and_log_until(end_time)
}