mod lif;
mod simulation;
mod synapse;
mod event;

use lif::LifNeuron;
use simulation::{replay_equal, Simulation, SchedulerMode};
use synapse::Synapse;
use std::env;

fn build_sim(seed: u64, n_threads: usize) -> Simulation {
    let neurons = LifNeuron::new(2, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);

    let mut synapses = Synapse::new();
    synapses.add_current_based(0, 1, 1000.0, 2.0, 10.0, 3);
    synapses.add_conductance_based(1, 0, 1000.0, 3.0, 8.0, 0.0, 2);

    Simulation::new_with_seed(neurons, synapses, 1.0, seed, n_threads)
}

fn main() {
    // choose number of threads from env or default
    let threads: usize = env::var("SIM_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let seed: u64 = env::var("SIM_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42);

    // build and run demo
    let mut sim = build_sim(seed, threads);
    sim.scheduler_mode = SchedulerMode::Deterministic { n_threads: threads };
    sim.verbose = false;

    // push external pulses to neuron 0 only
    for step in (0..100).step_by(10) {
        sim.push_event(step as f32, 0, 400.0, 0, 0.0);
    }

    sim.record_probes();
    sim.run_auto(400.0);

    println!("Spike log:");
    for (t, nid) in &sim.spike_log {
        println!("spike t={:.3} nid={}", t, nid);
    }
    println!("probes recorded: {}", sim.probes.len());

    // quick determinism check: build two sims with the same seed and compare
    let equal = replay_equal(|s| build_sim(s, threads), 400.0, seed);
    println!("replay_equal (same seed) => {}", equal);
}

// mod rpc;
// mod simulation;
// mod lif;
// mod synapse;
// mod event;

// use std::sync::Arc;
// use tonic::transport::Server;

// use rpc::{RpcService, SimStore};
// use rpc::pb::neuro_sim_server::NeuroSimServer;

// #[tokio::main]
// async fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let addr = "127.0.0.1:50051".parse()?;
//     let store = Arc::new(SimStore::new());
//     let service = RpcService::new(store);

//     println!("RPC server listening on {}", addr);

//     Server::builder()
//         .add_service(NeuroSimServer::new(service))
//         .serve(addr)
//         .await?;

//     Ok(())
// }
