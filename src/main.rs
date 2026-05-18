fn build_sim(seed: u64, n_threads: usize) -> synaptic_shenanigans::Simulation {
    use synaptic_shenanigans::{LifNeuron, Synapse, Simulation, SchedulerMode};
    let neurons = LifNeuron::new(2, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut syn = Synapse::new();
    syn.add_current_based(0, 1, 1000.0, 2.0, 10.0, 3);
    syn.add_conductance_based(1, 0, 1000.0, 3.0, 8.0, 0.0, 2);
    let mut sim = Simulation::new_with_neurons(neurons, syn, 1.0, seed, n_threads);
    sim.scheduler_mode = SchedulerMode::Deterministic { n_threads };
    sim
}

fn main() {
    use synaptic_shenanigans::replay_equal;
    let threads = std::env::var("SIM_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let seed    = std::env::var("SIM_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42);
    let mut sim = build_sim(seed, threads);
    sim.verbose = false;
    for step in (0..100usize).step_by(10) { sim.push_event(step as f32, 0, 400.0, 0, 0.0); }
    sim.record_probes();
    sim.run_auto(400.0);
    for (t, nid) in &sim.spike_log { println!("spike t={:.3} nid={}", t, nid); }
    println!("probes: {}  replay_equal: {}", sim.probes.len(), replay_equal(|s| build_sim(s, threads), 400.0, seed));
}