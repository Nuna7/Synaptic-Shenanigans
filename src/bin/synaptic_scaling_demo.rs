//! Synaptic scaling demo: weights stabilise around a target firing rate
//! even after the network is perturbed by a sudden change in input strength.
//!
//! Run: cargo run --release --bin synaptic_scaling_demo

use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::network::{NetworkBuilder, EdgeParams};
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};
use synaptic_shenanigans::poisson::PoissonPopulation;
use synaptic_shenanigans::synaptic_scaling::{SynapticScaling, SynapticScalingConfig};
use std::io::Write;

fn main() {
    println!("=== Synaptic Scaling Demo ===\n");
    println!("Target: ~5 Hz despite two abrupt input changes.\n");
    std::fs::create_dir_all("bench/results").unwrap();

    let n = 80usize;
    let ep = EdgeParams { weight: 4.0, inhibitory_fraction: 0.2, ..EdgeParams::default() };
    let syn = NetworkBuilder::erdos_renyi(n, 0.08, ep, 42);

    let neurons  = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim  = Simulation::new_with_seed(neurons, syn, 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;

    let cfg = SynapticScalingConfig {
        target_rate_hz: 5.0,
        alpha: 1.0,
        w_min: 0.0, w_max: 15.0,
        update_interval_ms: 200.0,
        rate_window_ms: 500.0,
        enabled: true,
    };
    let mut scaler = SynapticScaling::new(n, cfg);

    let mut csv = std::fs::File::create("bench/results/synaptic_scaling.csv").unwrap();
    writeln!(csv, "t_ms,phase,input_hz,mean_rate_hz,mean_weight,fraction_at_target").unwrap();

    let phases = [(1000.0f32, 10.0f32, "Baseline"),
                  (1000.0f32, 50.0f32, "Perturbation"),
                  (2000.0f32, 50.0f32, "Scaling")];

    println!("{:>8}  {:>14}  {:>10}  {:>12}  {:>12}  {:>10}",
        "t(ms)", "Phase", "Input(Hz)", "Rate(Hz)", "Mean W", "AtTarget");
    println!("{}", "-".repeat(75));

    let mut t_now = 0.0f32;
    let report_step = 200.0f32;

    for (phase_dur, input_hz, phase_name) in phases {
        let phase_end = t_now + phase_dur;
        let mut poisson = PoissonPopulation::new(n, input_hz, 60.0, 42);
        poisson.inject_into(&mut sim, t_now, phase_end);

        while t_now < phase_end {
            let step_end = (t_now + report_step).min(phase_end);
            sim.run_auto(step_end);

            // Record spikes for scaling
            for &(t, nid) in sim.spike_log.iter().filter(|&&(t,_)| t >= t_now && t < step_end) {
                scaler.record_spike(nid, t);
            }

            // Apply scaling
            if let Some(syn_ref) = std::sync::Arc::get_mut(&mut sim.synapses) {
                scaler.scale_weights(step_end, &syn_ref.post.clone(), &mut syn_ref.weight);
            }

            // Report
            let recent: Vec<_> = sim.spike_log.iter()
                .filter(|&&(t,_)| t >= step_end - report_step && t < step_end)
                .collect();
            let rate    = recent.len() as f32 / (n as f32 * report_step / 1000.0);
            let weights = &sim.synapses.weight;
            let mean_w  = weights.iter().sum::<f32>() / weights.len().max(1) as f32;
            let frac    = scaler.fraction_at_target();

            println!("{:>8.0}  {:>14}  {:>10.0}  {:>12.2}  {:>12.4}  {:>9.1}%",
                step_end, phase_name, input_hz, rate, mean_w, frac * 100.0);
            writeln!(csv, "{:.0},{},{},{:.4},{:.4},{:.4}",
                step_end, phase_name, input_hz, rate, mean_w, frac).unwrap();

            t_now = step_end;
        }
    }

    println!("\nTotal scaling updates: {}", scaler.update_count);
    println!("Results → bench/results/synaptic_scaling.csv");
}