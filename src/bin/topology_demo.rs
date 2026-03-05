//! Network topology demo — compares Erdős-Rényi, Watts-Strogatz, and Barabási-Albert
//! networks on the same LIF population, reporting structure and activity metrics.
//!
//! Run:
//!   cargo run --release --bin topology_demo

use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};
use synaptic_shenanigans::network::{NetworkBuilder, EdgeParams, NetworkMetrics};
use std::time::Instant;

struct Result {
    name: &'static str,
    metrics: NetworkMetrics,
    n_spikes: usize,
    mean_rate: f32,
    wall_ms: f64,
}

fn run_topology(name: &'static str, n: usize, syn: synaptic_shenanigans::synapse::Synapse) -> Result {
    let metrics = NetworkMetrics::compute(&syn, n);

    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_seed(neurons, syn, 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;

    // Seed with Poisson-like external inputs
    let mut t = 0.0f32;
    let rate = 0.1f32; // mean events per neuron per ms
    let mut _rng_seed = 0u64;
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(99);

    while t < 500.0 {
        for neuron_id in 0..n {
            if rng.gen_range(0.0..1.0) < rate {
                sim.push_event(t, neuron_id, 80.0, 0, 0.0);
            }
        }
        t += 1.0;
    }

    let wall_start = Instant::now();
    sim.run_auto(500.0);
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

    let n_spikes = sim.spike_log.len();
    let mean_rate = n_spikes as f32 / (n as f32 * 0.5); // spikes/neuron/s

    Result { name, metrics, n_spikes, mean_rate, wall_ms }
}

fn main() {
    let n = 200usize;
    let sim_time = 500.0f32;
    let ep = EdgeParams::default();

    println!("=== Network Topology Comparison ===");
    println!("Neurons: {}  Sim time: {} ms\n", n, sim_time);

    let topologies: Vec<(&'static str, synaptic_shenanigans::synapse::Synapse)> = vec![
        ("Erdos-Renyi (p=0.05)",  NetworkBuilder::erdos_renyi(n, 0.05, ep.clone(), 42)),
        ("Small-World (k=6,b=0.1)", NetworkBuilder::small_world(n, 6, 0.1, ep.clone(), 42)),
        ("Scale-Free (m=3)",      NetworkBuilder::scale_free(n, 3, ep.clone(), 42)),
    ];

    let mut results = Vec::new();
    for (name, syn) in topologies {
        print!("Running {}... ", name);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let r = run_topology(name, n, syn);
        println!("done ({:.1} ms wall)", r.wall_ms);
        results.push(r);
    }

    println!("\n{:<30} {:>8} {:>8} {:>12} {:>12} {:>12}",
        "Topology", "Synapses", "MaxDeg", "Spikes", "Rate(Hz)", "Wall(ms)");
    println!("{}", "-".repeat(90));

    for r in &results {
        println!("{:<30} {:>8} {:>8} {:>12} {:>12.1} {:>12.1}",
            r.name,
            r.metrics.n_synapses,
            r.metrics.max_degree_out,
            r.n_spikes,
            r.mean_rate,
            r.wall_ms,
        );
    }

    // Write CSV
    std::fs::create_dir_all("bench/results").unwrap();
    let mut f = std::fs::File::create("bench/results/topology_comparison.csv").unwrap();
    use std::io::Write;
    writeln!(f, "topology,n_synapses,max_degree_out,n_spikes,mean_rate_hz,wall_ms").unwrap();
    for r in &results {
        writeln!(f, "{},{},{},{},{:.4},{:.4}",
            r.name, r.metrics.n_synapses, r.metrics.max_degree_out,
            r.n_spikes, r.mean_rate, r.wall_ms).unwrap();
    }
    println!("\nResults → bench/results/topology_comparison.csv");
}
