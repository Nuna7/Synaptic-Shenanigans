//! Homeostatic intrinsic plasticity demo.
//!
//! Demonstrates how neurons regulate their own excitability to maintain a
//! target firing rate despite changes in input statistics.
//!
//! Run:
//!   cargo run --release --bin homeostatic_demo

use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::synapse::Synapse;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};
use synaptic_shenanigans::homeostatic::{HomeostaticState, HomeostaticConfig};
use synaptic_shenanigans::poisson::PoissonPopulation;
use std::io::Write;

fn main() {
    println!("=== Homeostatic Plasticity Demo ===\n");
    println!("Goal: neurons maintain ~5 Hz despite abrupt changes in input rate.\n");

    let n = 50usize;
    let target_rate = 5.0f32;

    // ── Phase 1: baseline (low input, ~2 Hz spontaneous) ────────────────────
    // ── Phase 2: strong drive (high input, overshoots to ~20 Hz) ────────────
    // ── Phase 3: homeostasis kicks in, rate returns to ~target ──────────────

    let phase_ms = [1000.0f32, 1000.0f32, 3000.0f32];
    let phase_rates = [5.0f32, 50.0f32, 50.0f32]; // input Poisson rates (Hz)

    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let syn = Synapse::new();
    let mut sim = Simulation::new_with_seed(neurons, syn, 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;

    // Homeostatic config (fast for demo: τ_h = 2000 ms)
    let cfg = HomeostaticConfig {
        target_rate_hz:   target_rate,
        tau_h:            2_000.0,
        theta_min:        -72.0,
        theta_max:        -42.0,
        rate_window_ms:   500.0,
        update_interval_ms: 50.0,
        enabled:          true,
    };
    let mut homeo = HomeostaticState::new(n, -50.0, cfg);

    // Output CSV
    std::fs::create_dir_all("bench/results").unwrap();
    let mut csv = std::fs::File::create("bench/results/homeostatic.csv").unwrap();
    writeln!(csv, "t_ms,phase,input_rate_hz,mean_firing_hz,mean_theta_mV,fraction_at_target").unwrap();

    let mut t_sim = 0.0f32;
    let report_interval = 100.0f32;
    let mut last_report = 0.0f32;

    println!("{:>8}  {:>8}  {:>14}  {:>12}  {:>12}  {:>10}",
        "t (ms)", "Phase", "Input (Hz)", "Rate (Hz)", "θ_mean (mV)", "At target");
    println!("{}", "-".repeat(70));

    for (phase_idx, (&phase_dur, &input_rate)) in phase_ms.iter().zip(phase_rates.iter()).enumerate() {
        let phase_end = t_sim + phase_dur;
        let phase_name = ["Baseline", "Overstimulation", "Homeostasis"][phase_idx];
        println!("\n--- Phase {}: {} (input = {} Hz) ---", phase_idx+1, phase_name, input_rate);

        // Inject Poisson input for this phase (pre-built for efficiency)
        let mut poisson = PoissonPopulation::new(n, input_rate, 80.0, 42 + phase_idx as u64);
        poisson.prebuild(&mut sim, phase_end);

        while t_sim < phase_end {
            let step_end = (t_sim + report_interval).min(phase_end);
            sim.run_auto(step_end);

            // Update homeostatic thresholds
            for &(t, nid) in sim.spike_log.iter().filter(|&&(t,_)| t >= t_sim && t < step_end) {
                homeo.record_spike(nid, t);
            }
            let _ = homeo.update(step_end);

            // Apply adapted thresholds back to neuron population
            if let Some(neurons) = std::sync::Arc::get_mut(&mut sim.neurons) {
                homeo.apply_thresholds_to_lif(neurons);
            }

            if step_end - last_report >= report_interval {
                // Compute recent firing rate
                let recent_spikes: Vec<(f32,usize)> = sim.spike_log.iter()
                    .filter(|&&(t,_)| t >= step_end - report_interval && t < step_end)
                    .cloned().collect();

                let rate_hz = recent_spikes.len() as f32 / (n as f32 * report_interval / 1000.0);
                let theta_stats = homeo.threshold_stats();
                let frac = homeo.fraction_at_target();

                println!("{:>8.0}  {:>8}  {:>14.0}  {:>12.2}  {:>12.3}  {:>10.1}%",
                    step_end, phase_name, input_rate, rate_hz, theta_stats.mean, frac * 100.0);
                writeln!(csv, "{:.0},{},{},{:.4},{:.4},{:.4}",
                    step_end, phase_idx+1, input_rate, rate_hz, theta_stats.mean, frac).unwrap();
                last_report = step_end;
            }

            t_sim = step_end;
        }
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    let final_stats = homeo.threshold_stats();
    println!("Final threshold distribution: {}", final_stats);
    println!("Fraction at target rate: {:.1}%", homeo.fraction_at_target() * 100.0);
    println!("Total homeostatic updates applied: {}", homeo.update_count);
    println!("Significant threshold events logged: {}", homeo.history.len());
    println!("Total spikes recorded: {}", sim.spike_log.len());
    println!("\nData → bench/results/homeostatic.csv");
}
