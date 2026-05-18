//! Benchmark harness. Run: cargo run --release --bin bench_harness
use std::time::Instant;
use std::io::Write;
use synaptic_shenanigans::{LifNeuron, Synapse, Simulation, SchedulerMode};

fn main() {
    let n = 10_000; let n_syn = 100_000; let dt = 1.0; let sim_time = 1_000.0; let seed = 42;
    println!("Benchmark harness\nneurons={} synapses={}", n, n_syn);
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut syn = Synapse::new();
    for i in 0..n_syn { syn.add_current_based(i%n, (i*31)%n, 1.0, 2.0, 10.0, 0); }
    let mut sim = Simulation::new_with_neurons(neurons, syn, dt, seed, 4);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;
    println!("Warm-up...");
    sim.run_auto(100.0);
    let mut latencies = Vec::new();
    let mut sim_t = 0.0f64;
    let wall = Instant::now();
    while sim_t < sim_time {
        let t0 = Instant::now();
        sim.run_auto(sim_t as f32 + dt as f32);
        latencies.push(t0.elapsed().as_secs_f64());
        sim_t += dt as f64;
    }
    let elapsed = wall.elapsed().as_secs_f64();
    latencies.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len()/2];
    let p99 = latencies[latencies.len()*99/100];
    let max = *latencies.last().unwrap();
    println!("Throughput: {:.2} sim-ms/wall-s", sim_time/elapsed);
    println!("Latency p50={:.6}s p99={:.6}s max={:.6}s", p50, p99, max);
    std::fs::create_dir_all("bench/results").unwrap();
    let mut f = std::fs::File::create("bench/results/harness.csv").unwrap();
    writeln!(f, "metric,value").unwrap();
    writeln!(f, "throughput,{:.6}", sim_time/elapsed).unwrap();
    writeln!(f, "latency_p50,{:.6}", p50).unwrap();
    writeln!(f, "latency_p99,{:.6}", p99).unwrap();
    writeln!(f, "latency_max,{:.6}", max).unwrap();
}