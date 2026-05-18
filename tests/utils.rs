use synaptic_shenanigans::{LifNeuron, SchedulerMode, Simulation, Synapse};

pub fn build_sim(seed: u64) -> Simulation {
    let neurons = LifNeuron::new(100, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut syn = Synapse::new();
    for i in 0..1000 {
        syn.add_current_based(i % 100, (i * 7) % 100, 1.0, 1.0, 5.0, 0);
    }
    let mut sim = Simulation::new_with_neurons(neurons, syn, 1.0, seed, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;
    sim.run_auto(500.0);
    sim
}
