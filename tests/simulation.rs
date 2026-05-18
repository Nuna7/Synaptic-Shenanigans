mod utils;
use crate::utils::build_sim;
use synaptic_shenanigans::simulation::Simulation;

#[test]
fn spike_times_monotonic() {
    let sim = build_sim(42);
    for w in sim.spike_log.windows(2) { assert!(w[0].0 <= w[1].0); }
}

#[test]
fn voltages_are_finite() {
    let sim = build_sim(42);
    for i in 0..sim.neurons.len() { assert!(sim.neurons.read_v(i).is_finite()); }
}

#[test]
fn checkpoint_roundtrip() {
    let mut sim = build_sim(42);
    sim.run_auto(200.0);
    sim.save_state("tmp_test.bin", "tmp_test.bin.sha256").unwrap();
    let mut sim2 = Simulation::load_state("tmp_test.bin", 42, 1).unwrap();
    sim2.run_auto(300.0);
    let mut sim_ref = build_sim(42);
    sim_ref.run_auto(500.0);
    assert_eq!(sim2.spike_log, sim_ref.spike_log);
    let _ = std::fs::remove_file("tmp_test.bin");
    let _ = std::fs::remove_file("tmp_test.bin.sha256");
}