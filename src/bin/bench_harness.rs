use std::time::Instant;
use std::fs::File;
use std::io::Write;

use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::synapse::Synapse;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};

fn main() {
    // -----------------------------
    // Configuration (deterministic)
    // -----------------------------
    let n_neurons = 10_000;
    let n_synapses = 100_000;
    let dt = 1.0;
    let sim_time = 1_000.0;
    let seed = 42;
    let threads = 4;

    println!("Benchmark harness");
    println!("neurons={} synapses={} threads={}", n_neurons, n_synapses, threads);

    // -----------------------------
    // Build network
    // -----------------------------
    let neurons = LifNeuron::new(
        n_neurons,
        -65.0, -50.0, 20.0,
        1.0, 1.0, 5.0,
    );

    let mut syn = Synapse::new();
    for i in 0..n_synapses {
        let pre = i % n_neurons;
        let post = (i * 31) % n_neurons;
        syn.add_current_based(pre, post, 1.0, 2.0, 10.0, 0);
    }

    let mut sim = Simulation::new_with_seed(
        neurons,
        syn,
        dt,
        seed,
        threads,
    );

    sim.scheduler_mode = SchedulerMode::SingleThreaded; // deterministic
    sim.verbose = false;

    // -----------------------------
    // Warm-up
    // -----------------------------
    println!("Warm-up...");
    sim.run_auto(100.0);

    // -----------------------------
    // Timed run
    // -----------------------------
    let mut latencies = Vec::new();
    let mut sim_t = 0.0;

    let wall_start = Instant::now();

    while sim_t < sim_time {
        let t0 = Instant::now();
        sim.run_auto(sim_t + dt);
        let elapsed = t0.elapsed();
        latencies.push(elapsed.as_secs_f64());
        sim_t += dt;
    }

    let wall_elapsed = wall_start.elapsed().as_secs_f64();

    let throughput = sim_time as f64 / wall_elapsed;

    // -----------------------------
    // Stats
    // -----------------------------
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() * 99) / 100];
    let max = *latencies.last().unwrap();

    println!("Throughput: {:.2} sim-ms / wall-s", throughput);
    println!("Latency p50={:.6}s p99={:.6}s max={:.6}s", p50, p99, max);

    // -----------------------------
    // Write CSV
    // -----------------------------
    std::fs::create_dir_all("bench/results").unwrap();
    let mut f = File::create("bench/results/harness.csv").unwrap();

    writeln!(f, "metric,value").unwrap();
    writeln!(f, "throughput,{:.6}", throughput).unwrap();
    writeln!(f, "latency_p50,{:.6}", p50).unwrap();
    writeln!(f, "latency_p99,{:.6}", p99).unwrap();
    writeln!(f, "latency_max,{:.6}", max).unwrap();

    println!("Results written to bench/results/harness.csv");
}
