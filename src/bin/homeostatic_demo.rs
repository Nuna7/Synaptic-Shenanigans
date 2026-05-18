//! Homeostatic intrinsic plasticity demo.
//! Run: cargo run --release --bin homeostatic_demo

use synaptic_shenanigans::LifNeuron;
use synaptic_shenanigans::synapse::Synapse;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};
use synaptic_shenanigans::plasticity::{HomeostaticState, HomeostaticConfig};
use synaptic_shenanigans::input::poisson::PoissonPopulation;
use std::io::Write;

fn main() {
    println!("=== Homeostatic Plasticity Demo ===\n");
    let n = 50usize;
    let target_rate = 5.0f32;
    let phase_ms    = [1000.0f32, 1000.0f32, 3000.0f32];
    let phase_rates = [5.0f32, 50.0f32, 50.0f32];

    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let syn = Synapse::new();
    let mut sim = Simulation::new_with_neurons(neurons, syn, 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;

    let cfg = HomeostaticConfig {
        target_rate_hz: target_rate, tau_h: 2_000.0, theta_min: -72.0, theta_max: -42.0,
        rate_window_ms: 500.0, update_interval_ms: 50.0, enabled: true,
    };
    let mut homeo = HomeostaticState::new(n, -50.0, cfg);

    std::fs::create_dir_all("bench/results").unwrap();
    let mut csv = std::fs::File::create("bench/results/homeostatic.csv").unwrap();
    writeln!(csv, "t_ms,phase,input_rate_hz,mean_firing_hz,mean_theta_mV,fraction_at_target").unwrap();

    let mut t_sim = 0.0f32;
    let report_interval = 100.0f32;
    let mut last_report = 0.0f32;

    for (phase_idx, (&phase_dur, &input_rate)) in phase_ms.iter().zip(phase_rates.iter()).enumerate() {
        let phase_end = t_sim + phase_dur;
        let phase_name = ["Baseline", "Overstimulation", "Homeostasis"][phase_idx];
        println!("\n--- Phase {}: {} (input = {} Hz) ---", phase_idx+1, phase_name, input_rate);
        let mut poisson = PoissonPopulation::new(n, input_rate, 80.0, 42 + phase_idx as u64);
        poisson.prebuild(&mut sim, phase_end);

        while t_sim < phase_end {
            let step_end = (t_sim + report_interval).min(phase_end);
            sim.run_auto(step_end);
            for &(t, nid) in sim.spike_log.iter().filter(|&&(t,_)| t >= t_sim && t < step_end) {
                homeo.record_spike(nid, t);
            }
            let _ = homeo.update(step_end);
            // KEY CHANGE: apply_thresholds works on Arc<dyn NeuronPopulation>
            homeo.apply_thresholds(sim.neurons.as_ref());

            if step_end - last_report >= report_interval {
                let recent = sim.spike_log.iter()
                    .filter(|&&(t,_)| t >= step_end - report_interval && t < step_end).count();
                let rate_hz = recent as f32 / (n as f32 * report_interval / 1000.0);
                let ts   = homeo.threshold_stats();
                let frac = homeo.fraction_at_target();
                println!("{:>8.0}  {:>8}  {:>14.0}  {:>12.2}  {:>12.3}  {:>10.1}%",
                    step_end, phase_name, input_rate, rate_hz, ts.mean, frac*100.0);
                writeln!(csv, "{:.0},{},{},{:.4},{:.4},{:.4}",
                    step_end, phase_idx+1, input_rate, rate_hz, ts.mean, frac).unwrap();
                last_report = step_end;
            }
            t_sim = step_end;
        }
    }
    println!("\nFinal: {}  fraction@target={:.1}%  updates={}  total_spikes={}",
        homeo.threshold_stats(), homeo.fraction_at_target()*100.0, homeo.update_count, sim.spike_log.len());
    println!("Data → bench/results/homeostatic.csv");
}