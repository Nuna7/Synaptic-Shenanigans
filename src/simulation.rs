#![allow(dead_code)]
use crate::event::Event;
use crate::lif::{LifNeuron, NeuronPopulation};
use crate::synapse::Synapse;
use std::collections::BinaryHeap;
use std::thread;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use crossbeam::atomic::AtomicCell;

const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
pub struct ThreadLocal {
    pub local_queue: Vec<Event>,
    // (time, neuron_id, thread_id)
    pub local_spike_log: Vec<(f32, usize, usize)>,
}

impl Default for ThreadLocal {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadLocal {

    pub fn new() -> Self {
        Self {
            local_queue: Vec::new(),
            local_spike_log: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.local_queue.clear();
        self.local_spike_log.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerMode {
    /// Single thread, fully deterministic.
    SingleThreaded,
    /// Deterministic multithreaded with static partition + barriers.
    Deterministic { n_threads: usize },
    /// Performance mode: rayon / dynamic work-stealing, not deterministic.
    Performance { n_threads: usize },
}

pub struct Simulation {
    pub time: f32,
    pub neurons: Arc<LifNeuron>,
    pub synapses: Arc<Synapse>,
    pub event_queue: BinaryHeap<Event>,
    pub dt: f32,

    // Determinism
    pub seed: u64,
    rng: ChaCha20Rng,
    // next_seq: u64,
    next_seq: AtomicU64,

    // Logging
    pub spike_log: Vec<(f32, usize)>, // (time, neuron_id)

    // Per-thread local buffers (single-threaded execution uses these sequentially).
    pub thread_locals: Vec<ThreadLocal>,
    pub num_threads: usize,

    pub pre_index: Vec<Vec<usize>>,
    pub verbose: bool,

    pub scheduler_mode: SchedulerMode,

    // Probes
    pub probes: Vec<Vec<f32>>,
}

impl Simulation {
    pub fn new_with_seed(
        neurons: LifNeuron,
        synapses: Synapse,
        dt: f32,
        seed: u64,
        num_threads: usize,
    ) -> Self {
        let locals = (0..num_threads).map(|_| ThreadLocal::new()).collect();
        let rng = ChaCha20Rng::seed_from_u64(seed);
        let arc_neurons = Arc::new(neurons);
        let arc_syn = Arc::new(synapses);

        let mut sim = Self {
            time: 0.0,
            neurons: arc_neurons,
            synapses: arc_syn,
            event_queue: BinaryHeap::new(),
            dt,
            seed,
            rng,
            next_seq: AtomicU64::new(0),
            spike_log: Vec::new(),
            thread_locals: locals,
            num_threads,
            pre_index: Vec::new(),
            verbose: false,
            scheduler_mode: SchedulerMode::SingleThreaded,
            probes: Vec::new(),
        };
        sim.pre_index = sim.synapses.build_pre_index(sim.neurons.len());
        sim
    }

    pub fn new(neurons: LifNeuron, synapses: Synapse, dt: f32, num_threads: usize) -> Self {
        Self::new_with_seed(neurons, synapses, dt, 42, num_threads)
    }

    fn time_to_tick(&self, t: f32) -> u64 {
        ((t / self.dt) as f64 + 1e-9).floor() as u64
    }

    fn owner_of(&self, neuron_id: usize) -> usize {
        let chunk = self.neurons.len().div_ceil(self.num_threads);
        let owner = neuron_id / chunk;

        // clamp to last *real* partition
        owner.min(self.num_threads - 1)
    }

    /// Deterministic multi-threaded using index ranges and Arc + AtomicCell neurons.
    pub fn run_deterministic_multithreaded(&mut self, end_time: f32) {
        if self.num_threads <= 1 {
            // Deterministic fallback to single-threaded execution
            self.run_until(end_time);
            return;
        }

        let n = self.neurons.len();
        if n == 0 {
            return;
        }
        let t = self.num_threads;
        let chunk = n.div_ceil(t);
        let pre_index = Arc::new(self.pre_index.clone());
        let syn = Arc::clone(&self.synapses);
        let neurons = Arc::clone(&self.neurons);

        while let Some(next_ev) = self.event_queue.pop() {
            if next_ev.time > end_time {
                self.event_queue.push(next_ev);
                break;
            }
            self.time = next_ev.time;

            // gather epoch events
            let mut events_at_t = vec![next_ev];
            while let Some(peek) = self.event_queue.peek() {
                if (peek.time - self.time).abs() < 1e-6 {
                    events_at_t.push(self.event_queue.pop().unwrap());
                } else {
                    break;
                }
            }

            // distribute by owner
            let mut in_queues: Vec<Vec<Event>> = vec![Vec::new(); t];
            for ev in events_at_t {
                let owner = self.owner_of(ev.target);
                in_queues[owner].push(ev);
            }

            // build partitions as index ranges
            let mut partitions = self.neurons.split_indices(chunk);
            partitions.retain(|p| p.len > 0);
            let real_t = partitions.len();

            let dt = self.dt;
            let cur_time = self.time;
            let eps = EPS;

            // results per thread will be collected here (inside scope)
            thread::scope(|scope| {
                let mut handles = Vec::with_capacity(real_t);

                for tid in 0..real_t {
                    // clone arcs for the thread
                    let syn = Arc::clone(&syn);
                    let pre_index = Arc::clone(&pre_index);
                    let neurons = Arc::clone(&neurons);

                    // take this owner's events
                    let in_q = std::mem::take(&mut in_queues[tid]);

                    // partition indices
                    let part = partitions[tid];

                    handles.push(scope.spawn(move || {
                        // Build local input buffer for this partition
                        let mut inputs_local = vec![0.0f32; part.len];
                        for ev in in_q.iter() {
                            if ev.target >= part.start_index && ev.target < part.start_index + part.len {
                                let local_i = ev.target - part.start_index;
                                if ev.model_type == 0 {
                                    inputs_local[local_i] += ev.weight;
                                } else {
                                    let v_post = neurons.read_v(ev.target);
                                    inputs_local[local_i] += ev.weight * (ev.e_rev - v_post);
                                }
                            }
                        }

                        // Step the partition (mutates atomics inside neurons)
                        
                        neurons.step_range(&inputs_local, part.start_index);

                        // collect events & local logs
                        let mut new_events: Vec<Event> = Vec::new();
                        let mut local_logs: Vec<(f32, usize, usize)> = Vec::new();

                        for i in 0..part.len {
                            let global_idx = part.start_index + i;
                            if neurons.local_spiked(global_idx) {
                                local_logs.push((cur_time, global_idx, tid));

                                if global_idx < pre_index.len() {
                                    for &s_idx in &pre_index[global_idx] {
                                        if syn.pre[s_idx] == global_idx {
                                            let post = syn.post[s_idx];
                                            let w = syn.weight[s_idx];
                                            let d = syn.delay[s_idx];
                                            let model = syn.model_type[s_idx];
                                            let e_rev = syn.e_rev[s_idx];
                                            let arrival = cur_time + d;
                                            if arrival <= end_time && arrival > cur_time + eps {
                                                new_events.push(Event {
                                                    tick: (arrival / dt) as u64,
                                                    time: arrival,
                                                    target: post,
                                                    weight: w,
                                                    seq: 0,
                                                    model_type: model,
                                                    e_rev,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        (tid, new_events, local_logs)
                    }));
                }

                // join threads deterministically and merge results into thread-local buffers
                for handle in handles {
                    let (tid, new_events, local_logs) = handle.join().expect("thread panic");

                    // append spike logs deterministically
                    for (tstamp, nid, _tid2) in local_logs {
                        self.thread_locals[tid].local_spike_log.push((tstamp, nid, tid));
                    }

                    // assign seqs deterministically and push events into owning local queues
                    for mut ev in new_events {
                        ev.seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
                        let owner_post = self.owner_of(ev.target);
                        self.thread_locals[owner_post].local_queue.push(ev);
                    }
                }
            });

            // merge local queues into global heap (deterministic)
            self.merge_queues_into_global();
            self.merge_spike_logs();
        }
    }

    /// Non-deterministic performance-mode (Rayon). Kept optional behind feature flag.
    /// Non-deterministic performance mode (Rayon).
    /// No owner routing — events are processed globally and
    /// new events are pushed into a global mutex queue.
    #[cfg(feature = "performance")]
    pub fn run_performance_multithreaded(&mut self, end_time: f32) {
        use parking_lot::Mutex;
        use rayon::prelude::*;

        let n = self.neurons.len();
        if n == 0 {
            return;
        }

        let pre_index = Arc::new(self.pre_index.clone());
        let syn = Arc::clone(&self.synapses);
        let neurons = Arc::clone(&self.neurons);
        let dt = self.dt;
        let eps = EPS;

        let global_new_events = Arc::new(Mutex::new(Vec::<Event>::new()));

        while let Some(next_ev) = self.event_queue.pop() {
            if next_ev.time > end_time {
                self.event_queue.push(next_ev);
                break;
            }

            self.time = next_ev.time;

            // Gather epoch events
            let mut events_at_t = vec![next_ev];
            while let Some(peek) = self.event_queue.peek() {
                if (peek.time - self.time).abs() < 1e-6 {
                    events_at_t.push(self.event_queue.pop().unwrap());
                } else {
                    break;
                }
            }

            // Each neuron accumulates input current independently.
            let mut inputs = vec![0.0f32; n];
            for ev in &events_at_t {
                if ev.target < n {
                    if ev.model_type == 0 {
                        inputs[ev.target] += ev.weight;
                    } else {
                        let v_post = neurons.read_v(ev.target);
                        inputs[ev.target] += ev.weight * (ev.e_rev - v_post);
                    }
                }
            }

            // Step all neuron partitions in parallel
            let chunk = (n + self.num_threads - 1) / self.num_threads;
            let partitions = neurons.split_indices(chunk);

            partitions.par_iter().for_each(|part| {
                let start = part.start_index;
                let len = part.len;

                // Slice of inputs for this region
                let local_inputs = &inputs[start..start + len];

                neurons.step_range(local_inputs, start);

                // Collect new spikes and produce outgoing events
                for local_i in 0..len {
                    let nid = start + local_i;
                    if neurons.local_spiked(nid) {
                        // produce outgoing synaptic events
                        if nid < pre_index.len() {
                            for &s_idx in &pre_index[nid] {
                                if syn.pre[s_idx] == nid {
                                    let post = syn.post[s_idx];
                                    let w = syn.weight[s_idx];
                                    let d = syn.delay[s_idx];
                                    let model = syn.model_type[s_idx];
                                    let e_rev = syn.e_rev[s_idx];

                                    let arrival = self.time + d;
                                    if arrival <= end_time && arrival > self.time + eps {
                                        let ev = Event {
                                            tick: (arrival / dt) as u64,
                                            time: arrival,
                                            target: post,
                                            weight: w,
                                            seq: 0, // seq assigned later
                                            model_type: model,
                                            e_rev,
                                        };
                                        global_new_events.lock().push(ev);
                                    }
                                }
                            }
                        }
                    }
                }
            });

            // Drain new events into thread locals (nondeterministic order)
            let mut new_events = global_new_events.lock();
            for mut ev in new_events.drain(..) {
                ev.seq = self.next_seq.fetch_add(1, Ordering::Relaxed);

                let owner = self.owner_of(ev.target);
                self.thread_locals[owner].local_queue.push(ev);
            }

            self.merge_queues_into_global();
            self.merge_spike_logs();
        }
    }


    pub fn run_auto(&mut self, end_time: f32) {
        match self.scheduler_mode {
            SchedulerMode::SingleThreaded => self.run_until(end_time),
            SchedulerMode::Deterministic { .. } => self.run_deterministic_multithreaded(end_time),
            SchedulerMode::Performance { .. } => {
                #[cfg(feature = "performance")]
                {
                    self.run_performance_multithreaded(end_time)
                }
            }
        }
    }

    /// Push an event into the global queue (used for external inputs)
    pub fn push_event(
        &mut self,
        time: f32,
        target: usize,
        weight: f32,
        model_type: u8,
        e_rev: f32,
    ) {
        let tick = self.time_to_tick(time);
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let ev = Event {
            tick,
            time,
            target,
            weight,
            seq,
            model_type,
            e_rev,
        };
        self.event_queue.push(ev);
    }

    fn merge_queues_into_global(&mut self) {
        let mut estimated_total = 0usize;
        for tl in &self.thread_locals {
            estimated_total += tl.local_queue.len();
        }
        let mut all: Vec<Event> = Vec::with_capacity(estimated_total);

        for tl in self.thread_locals.iter_mut() {
            for ev in tl.local_queue.drain(..) {
                all.push(ev);
            }
        }

        all.sort_by(|a, b| {
            a.tick
                .cmp(&b.tick)
                .then(a.seq.cmp(&b.seq))
                .then(a.target.cmp(&b.target))
        });

        for ev in all {
            self.event_queue.push(ev);
        }
    }

    fn merge_spike_logs(&mut self) {
        let mut estimated_total = 0usize;
        for tl in &self.thread_locals {
            estimated_total += tl.local_spike_log.len();
        }
        let mut all: Vec<(f32, usize, usize)> = Vec::with_capacity(estimated_total);

        for (tid, tl) in self.thread_locals.iter_mut().enumerate() {
            for &(t, nid, _) in &tl.local_spike_log {
                all.push((t, nid, tid));
            }
            tl.local_spike_log.clear();
        }

        all.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });

        for (t, nid, _) in all {
            self.spike_log.push((t, nid));
        }
    }

    pub fn record_probes(&mut self) {
        self.probes.push((0..self.neurons.len()).map(|i| self.neurons.read_v(i)).collect());
    }

    pub fn advance_step(&mut self) {
        let target = self.time + self.dt;
        self.run_auto(target);
    }

    pub fn advance_steps(&mut self, n: usize) {
        for _ in 0..n {
            self.advance_step();
        }
    }

    pub fn run_until(&mut self, end_time: f32) {
        let n = self.neurons.len();
        if n == 0 {
            return;
        }

        let mut inputs = vec![0.0f32; n];
        let mut processed_events = 0usize;

        while let Some(next_ev) = self.event_queue.pop() {
            if next_ev.time > end_time {
                self.event_queue.push(next_ev);
                break;
            }

            processed_events += 1;
            if processed_events.is_multiple_of(100) && self.verbose {
                eprintln!("Processed {} events @ t={:.2}", processed_events, self.time);
            }

            if next_ev.time < self.time - 1e-9 {
                panic!("Time went backward: {} -> {}", self.time, next_ev.time);
            }

            if next_ev.time > self.time + 1e-9 {
                self.time = next_ev.time;
            }

            let mut events_at_t: Vec<Event> = vec![next_ev];

            while let Some(peek) = self.event_queue.peek() {
                if (peek.time - self.time).abs() < 1e-6 {
                    events_at_t.push(self.event_queue.pop().unwrap());
                } else {
                    break;
                }
            }

            for tl in self.thread_locals.iter_mut() {
                tl.local_queue.clear();
            }

            for ev in events_at_t {
                let owner = self.owner_of(ev.target);
                self.thread_locals[owner].local_queue.push(ev);
            }

            let t = self.num_threads;
            let chunk = if t <= 1 { n } else { n.div_ceil(t) };

            for tid in 0..self.num_threads {
                let start = tid * chunk;
                let end = ((tid + 1) * chunk).min(n);
                if start >= end {
                    continue;
                }

                let mut to_process: Vec<Event> = Vec::new();
                std::mem::swap(&mut to_process, &mut self.thread_locals[tid].local_queue);

                to_process.sort_by(|a, b| {
                    a.tick
                        .cmp(&b.tick)
                        .then(a.seq.cmp(&b.seq))
                        .then(a.target.cmp(&b.target))
                });

                for input in inputs.iter_mut().take(end).skip(start){
                    *input = 0.0;
                }

                for ev in &to_process {
                    if ev.target >= start && ev.target < end {
                        if ev.model_type == 0 {
                            inputs[ev.target] += ev.weight;
                        } else {
                            let v_post = self.neurons.read_v(ev.target);
                            inputs[ev.target] += ev.weight * (ev.e_rev - v_post);
                        }
                    } else {
                        let owner = self.owner_of(ev.target);
                        if owner != tid {
                            self.thread_locals[owner].local_queue.push(ev.clone());
                        }
                    }
                }
                

                self.neurons.step_range(&inputs, start);

                for nid in start..end {
                    if self.neurons.local_spiked(nid) {
                        self.thread_locals[tid].local_spike_log.push((self.time, nid, tid));

                        for &s_idx in &self.pre_index[nid] {
                            if self.synapses.pre[s_idx] == nid {
                                let post = self.synapses.post[s_idx];
                                let weight = self.synapses.weight[s_idx];
                                let delay = self.synapses.delay[s_idx];
                                let model_type = self.synapses.model_type[s_idx];
                                let e_rev = self.synapses.e_rev[s_idx];
                                let arrival_time = self.time + delay;
                                let arrival_tick = self.time_to_tick(arrival_time);

                                if arrival_time <= end_time && arrival_time > self.time + EPS {
                                    let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);

                                    let ev = Event {
                                        tick: arrival_tick,
                                        time: arrival_time,
                                        target: post,
                                        weight,
                                        seq,
                                        model_type,
                                        e_rev,
                                    };

                                    let owner_post = self.owner_of(post);
                                    self.thread_locals[owner_post].local_queue.push(ev);
                                }
                            }
                        }
                    }
                }
            }

            self.merge_queues_into_global();

            if let Some(peek) = self.event_queue.peek()
                && self.verbose {
                    eprintln!("Next event in heap: tick={} t={}", peek.tick, peek.time);
                }

            self.merge_spike_logs();
        }
    }

    pub fn run_and_log_until(&mut self, end_time: f32) -> Vec<(f32, usize)> {
        self.run_until(end_time);
        self.spike_log.clone()
    }

    /// Save checkpoint WITHOUT serializing AtomicCell — only plain data.
    pub fn save_state(&self, path: &str, hash_path: &str) -> std::io::Result<()> {
        use sha2::{Digest, Sha256};
        use std::fs;

        #[derive(serde::Serialize)]
        struct Snapshot {
            time: f32,
            next_seq: u64,
            // neuron state values (plain)
            v: Vec<f32>,
            spiked: Vec<bool>,
            refractory: Vec<bool>,
            refractory_timer: Vec<f32>,
            // neuron parameters
            v_rest: Vec<f32>,
            tau_m: Vec<f32>,
            v_thresh: Vec<f32>,
            r_m: Vec<f32>,
            dt_vec: Vec<f32>,
            refractory_period: Vec<f32>,

            synapses: Synapse, // Synapse *is* serializable
            pre_index: Vec<Vec<usize>>,
            dt: f32,
        }

        // collect complete snapshot
        let n = self.neurons.len();
        let snap = Snapshot {
            time: self.time,
            next_seq: self.next_seq.load(Ordering::Relaxed),
            v: (0..n).map(|i| self.neurons.read_v(i)).collect(),
            spiked: (0..n).map(|i| self.neurons.local_spiked(i)).collect(),
            refractory: (0..n).map(|i| self.neurons.refractory[i].load()).collect(),
            refractory_timer: (0..n).map(|i| self.neurons.refractory_timer[i].load()).collect(),
            v_rest: self.neurons.v_rest.clone(),
            tau_m: self.neurons.tau_m.clone(),
            v_thresh: self.neurons.v_thresh.clone(),
            r_m: self.neurons.r_m.clone(),
            dt_vec: self.neurons.dt.clone(),
            refractory_period: self.neurons.refractory_period.clone(),
            synapses: (*self.synapses).clone(),
            pre_index: self.pre_index.clone(),
            dt: self.dt,
        };

        let encoded = bincode::serialize(&snap).map_err(|e| {
            std::io::Error::other(format!("bincode serialize: {}", e))
        })?;
        fs::write(path, &encoded)?;

        let digest = Sha256::digest(&encoded);
        fs::write(hash_path, hex::encode(digest))?;

        Ok(())
    }

    pub fn load_state(path: &str, seed: u64, num_threads: usize) -> std::io::Result<Simulation> {
        use std::fs;

        #[derive(serde::Deserialize)]
        struct Snapshot {
            time: f32,
            next_seq: u64,
            v: Vec<f32>,
            spiked: Vec<bool>,
            refractory: Vec<bool>,
            refractory_timer: Vec<f32>,
            v_rest: Vec<f32>,
            tau_m: Vec<f32>,
            v_thresh: Vec<f32>,
            r_m: Vec<f32>,
            dt_vec: Vec<f32>,
            refractory_period: Vec<f32>,
            synapses: Synapse,
            pre_index: Vec<Vec<usize>>,
            dt: f32,
        }

        let bytes = fs::read(path)?;
        let snap: Snapshot = bincode::deserialize(&bytes).map_err(|e| {
            std::io::Error::other(format!("bincode deserialize: {}", e))
        })?;

        // build a LifNeuron using the saved parameter vectors and values
        let neurons = LifNeuron {
            v: snap.v.into_iter().map(AtomicCell::new).collect(),
            v_rest: snap.v_rest,
            tau_m: snap.tau_m,
            v_thresh: snap.v_thresh,
            r_m: snap.r_m,
            dt: snap.dt_vec,
            spiked: snap.spiked.into_iter().map(AtomicCell::new).collect(),
            refractory: snap.refractory.into_iter().map(AtomicCell::new).collect(),
            refractory_timer: snap.refractory_timer.into_iter().map(AtomicCell::new).collect(),
            refractory_period: snap.refractory_period,
        };

        let mut sim = Simulation::new_with_seed(neurons, snap.synapses, snap.dt, seed, num_threads);
        sim.time = snap.time;
        sim.next_seq.store(snap.next_seq, Ordering::Relaxed);
        sim.pre_index = snap.pre_index;
        
        Ok(sim)
    }


}

/// Determinism utility
pub fn replay_equal(build: impl Fn(u64) -> Simulation, end_time: f32, seed: u64) -> bool {
    let mut a = build(seed);
    let mut b = build(seed);

    let log_a = a.run_and_log_until(end_time);
    let log_b = b.run_and_log_until(end_time);

    log_a == log_b
}
