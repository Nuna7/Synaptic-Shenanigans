use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::simulation::Simulation;
use synaptic_shenanigans::network::{NetworkBuilder, EdgeParams};

#[test]
fn topology_execution_preserves_time_monotonicity() {
    let n = 200;
    let ep = EdgeParams::default();

    let syn = NetworkBuilder::erdos_renyi(n, 0.05, ep, 42);
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);

    let mut sim = Simulation::new_with_seed(neurons, syn, 1.0, 42, 1);
    sim.run_auto(500.0);
}
