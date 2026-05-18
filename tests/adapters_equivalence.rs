use synaptic_shenanigans::network::{EdgeParams, NetworkBuilder};
use synaptic_shenanigans::{LifNeuron, Simulation};

#[test]
fn topology_execution_preserves_time_monotonicity() {
    let n = 200;
    let syn = NetworkBuilder::erdos_renyi(n, 0.05, EdgeParams::default(), 42);
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_neurons(neurons, syn, 1.0, 42, 1);
    sim.run_auto(500.0);
    for w in sim.spike_log.windows(2) {
        assert!(w[0].0 <= w[1].0);
    }
}
