use synaptic_shenanigans::{LifNeuron, SchedulerMode, Simulation, Synapse};

#[test]
fn lif_passive_decay() {
    let neurons = LifNeuron::new(1, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_neurons(neurons, Synapse::new(), 1.0, 0, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;
    // set voltage above rest
    sim.neurons.set_threshold(0, -50.0); // ensure threshold unchanged
    // push zero current and run — voltage should decay toward rest
    for _ in 0..100 {
        sim.push_event(sim.time + 0.001, 0, 0.0, 0, 0.0);
    }
    // just run passively
    sim.run_auto(100.0);
    let v_end = sim.neurons.read_v(0);
    assert!(v_end.is_finite());
    assert!(v_end >= -65.0);
}

#[test]
fn lif_spike_and_reset() {
    let neurons = LifNeuron::new(1, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_neurons(neurons, Synapse::new(), 1.0, 0, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;
    sim.push_event(1.0, 0, 1000.0, 0, 0.0);
    sim.run_auto(10.0);
    assert!(!sim.spike_log.is_empty());
}

#[test]
fn lif_set_threshold_via_trait() {
    let neurons = LifNeuron::new(3, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let sim = Simulation::new_with_neurons(neurons, Synapse::new(), 1.0, 0, 1);
    sim.neurons.set_threshold(0, -45.0);
    assert!((sim.neurons.get_threshold(0) - (-45.0)).abs() < 1e-5);
}

#[test]
fn lif_reset_neuron_via_trait() {
    let neurons = LifNeuron::new(1, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_neurons(neurons, Synapse::new(), 1.0, 0, 1);
    sim.push_event(1.0, 0, 1000.0, 0, 0.0);
    sim.run_auto(10.0);
    sim.neurons.reset_neuron(0, -65.0);
    assert!((sim.neurons.read_v(0) - (-65.0)).abs() < 1e-5);
}
