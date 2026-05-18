mod utils;
use crate::utils::build_sim;
use synaptic_shenanigans::simulation::SchedulerMode;

#[test]
fn deterministic_replay() {
    assert_eq!(build_sim(42).spike_log, build_sim(42).spike_log);
}

#[test]
fn deterministic_mt_matches_single_thread() {
    let a = build_sim(42);
    let mut b = build_sim(42);
    b.scheduler_mode = SchedulerMode::Deterministic { n_threads: 4 };
    // b was already run in build_sim; rebuild properly
    use synaptic_shenanigans::{LifNeuron, Simulation, Synapse};
    let neurons = LifNeuron::new(100, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut syn = Synapse::new();
    for i in 0..1000usize {
        syn.add_current_based(i % 100, (i * 7) % 100, 1.0, 1.0, 5.0, 0);
    }
    let mut sim_mt = Simulation::new_with_neurons(neurons, syn, 1.0, 42, 4);
    sim_mt.scheduler_mode = SchedulerMode::Deterministic { n_threads: 4 };
    sim_mt.run_auto(500.0);
    assert_eq!(a.spike_log, sim_mt.spike_log);
}

#[test]
fn many_same_time_events_deterministic() {
    use synaptic_shenanigans::{LifNeuron, SchedulerMode, Simulation, Synapse};
    let neurons = LifNeuron::new(10, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_neurons(neurons, Synapse::new(), 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;
    for _ in 0..1000 {
        sim.push_event(10.0, 0, 1.0, 0, 0.0);
    }
    sim.run_auto(100.0);
    let ref_log = sim.spike_log.clone();
    let neurons2 = LifNeuron::new(10, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim2 = Simulation::new_with_neurons(neurons2, Synapse::new(), 1.0, 42, 1);
    sim2.scheduler_mode = SchedulerMode::SingleThreaded;
    for _ in 0..1000 {
        sim2.push_event(10.0, 0, 1.0, 0, 0.0);
    }
    sim2.run_auto(100.0);
    assert_eq!(ref_log, sim2.spike_log);
}
