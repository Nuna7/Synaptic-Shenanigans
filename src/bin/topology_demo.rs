//! Network topology demo. Run: cargo run --release --bin topology_demo
use std::io::Write;
use std::time::Instant;
use synaptic_shenanigans::network::{EdgeParams, NetworkBuilder, NetworkMetrics};
use synaptic_shenanigans::simulation::SchedulerMode;
use synaptic_shenanigans::synapse::Synapse;
use synaptic_shenanigans::{LifNeuron, Simulation};

fn run_topology(name: &'static str, n: usize, syn: Synapse) -> (NetworkMetrics, usize, f32, f64) {
    let metrics = NetworkMetrics::compute(&syn, n);
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_neurons(neurons, syn, 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(99);
    let mut t = 0.0f32;
    while t < 500.0 {
        for nid in 0..n {
            if rng.gen_range(0.0f32..1.0) < 0.1 {
                sim.push_event(t, nid, 80.0, 0, 0.0);
            }
        }
        t += 1.0;
    }
    let wall = Instant::now();
    sim.run_auto(500.0);
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let rate = sim.spike_log.len() as f32 / (n as f32 * 0.5);
    println!("  {} done ({:.1} ms)", name, wall_ms);
    (metrics, sim.spike_log.len(), rate, wall_ms)
}

fn main() {
    let n = 200usize;
    let ep = EdgeParams::default();
    println!(
        "=== Network Topology Comparison ===\nNeurons: {}  Sim time: 500 ms\n",
        n
    );
    let topologies: Vec<(&str, Synapse)> = vec![
        (
            "Erdos-Renyi (p=0.05)",
            NetworkBuilder::erdos_renyi(n, 0.05, ep.clone(), 42),
        ),
        (
            "Small-World (k=6,b=0.1)",
            NetworkBuilder::small_world(n, 6, 0.1, ep.clone(), 42),
        ),
        (
            "Scale-Free (m=3)",
            NetworkBuilder::scale_free(n, 3, ep.clone(), 42),
        ),
    ];
    let mut results = Vec::new();
    for (name, syn) in topologies {
        results.push((name, run_topology(name, n, syn)));
    }
    println!(
        "\n{:<30} {:>8} {:>8} {:>12} {:>12} {:>12}",
        "Topology", "Synapses", "MaxDeg", "Spikes", "Rate(Hz)", "Wall(ms)"
    );
    println!("{}", "-".repeat(90));
    std::fs::create_dir_all("bench/results").unwrap();
    let mut f = std::fs::File::create("bench/results/topology_comparison.csv").unwrap();
    writeln!(
        f,
        "topology,n_synapses,max_degree_out,n_spikes,mean_rate_hz,wall_ms"
    )
    .unwrap();
    for (name, (m, spikes, rate, wall)) in &results {
        println!(
            "{:<30} {:>8} {:>8} {:>12} {:>12.1} {:>12.1}",
            name, m.n_synapses, m.max_degree_out, spikes, rate, wall
        );
        writeln!(
            f,
            "{},{},{},{},{:.4},{:.4}",
            name, m.n_synapses, m.max_degree_out, spikes, rate, wall
        )
        .unwrap();
    }
    println!("\nResults → bench/results/topology_comparison.csv");
}
