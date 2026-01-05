use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::synapse::Synapse;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};

mod utils;

use crate::utils::build_sim;

#[test]
fn deterministic_replay() {
    let a = build_sim(42);
    let b = build_sim(42);

    assert_eq!(a.spike_log, b.spike_log);
}

#[test]
fn deterministic_mt_matches_single_thread() {
    let mut a = build_sim(42);

    let mut b = build_sim(42);
    b.scheduler_mode = SchedulerMode::Deterministic { n_threads: 4 };

    a.run_auto(500.0);
    b.run_auto(500.0);

    assert_eq!(a.spike_log, b.spike_log);
}

#[test]
fn many_same_time_events_deterministic() {
    let mut sim = build_sim(42);
    for _ in 0..1000 {
        sim.push_event(10.0, 0, 1.0, 0, 0.0);
    }
    sim.run_auto(100.0);

    let ref_log = sim.spike_log.clone();

    let mut sim2 = build_sim(42);
    for _ in 0..1000 {
        sim2.push_event(10.0, 0, 1.0, 0, 0.0);
    }
    sim2.run_auto(100.0);

    assert_eq!(ref_log, sim2.spike_log);
}
