//! STDP learning demo.
//! Run: cargo run --release --bin stdp_demo

use std::io::Write;
use synaptic_shenanigans::LifNeuron;
use synaptic_shenanigans::plasticity::{StdpConfig, StdpState};
use synaptic_shenanigans::simulation::{SchedulerMode, Simulation};
use synaptic_shenanigans::synapse::Synapse;

fn main() {
    println!("=== STDP Learning Demo ===\n");
    let n_pre = 10usize;
    let n_post = 10usize;
    let n_total = n_pre + n_post;
    let neurons = LifNeuron::new(n_total, -65.0, -54.0, 30.0, 40.0, 1.0, 5.0);
    let mut syn = Synapse::new();
    let initial_weight = 3.0;
    for i in 0..n_pre {
        for j in 0..n_post {
            syn.add_current_based(i, n_pre + j, initial_weight, 1.0, 2.0, 1);
        }
    }
    let n_synapses = syn.len();
    println!(
        "Network: {} pre, {} post, {} synapses",
        n_pre, n_post, n_synapses
    );

    let mut sim = Simulation::new_with_neurons(neurons, syn, 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;

    let stdp_cfg = StdpConfig {
        a_plus: 0.01,
        a_minus: 0.0085,
        tau_plus: 16.8,
        tau_minus: 33.7,
        w_min: 0.0,
        w_max: 15.0,
        enabled: true,
    };
    let mut stdp = StdpState::new(n_total, n_synapses, stdp_cfg);

    std::fs::create_dir_all("bench/results").unwrap();
    let mut csv = std::fs::File::create("bench/results/stdp_weights.csv").unwrap();
    writeln!(csv, "trial,syn_idx,pre,post,weight").unwrap();

    let syn_pre_snap = sim.synapses.pre.clone();
    let syn_post_snap = sim.synapses.post.clone();

    let n_trials = 50;
    let trial_dur = 150.0f32;
    let stim_w = 140.0f32;
    println!(
        "\n{:>6}  {:>12}  {:>12}  {:>12}",
        "Trial", "Mean W", "Max W", "STDP Updates"
    );

    for trial in 0..n_trials {
        let t_start = trial as f32 * trial_dur;
        for i in 0..n_pre {
            sim.push_event(t_start + 2.0, i, stim_w, 0, 0.0);
        }
        sim.run_auto(t_start + trial_dur);

        // KEY CHANGE: reset neurons via trait method reset_neuron — no direct field access
        for i in 0..n_total {
            sim.neurons.reset_neuron(i, -65.0);
        }

        let mut trial_spikes: Vec<(f32, usize)> = sim
            .spike_log
            .iter()
            .filter(|&&(t, _)| t >= t_start && t < t_start + trial_dur)
            .cloned()
            .collect();
        trial_spikes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut last_t = t_start;
        for &(t, nid) in &trial_spikes {
            stdp.decay_traces(t - last_t);
            last_t = t;
            stdp.accumulate_for_spike(
                nid,
                t,
                &sim.synapses.pre,
                &sim.synapses.post,
                &sim.pre_index,
            );
        }

        let updated = std::sync::Arc::get_mut(&mut sim.synapses)
            .map(|s| stdp.flush_weight_updates(&mut s.weight))
            .unwrap_or(0);

        let stats = StdpState::weight_stats(&sim.synapses.weight);
        println!(
            "{:>6}  {:>12.4}  {:>12.4}  {:>12}",
            trial + 1,
            stats.mean,
            stats.max,
            updated
        );

        for (si, &w) in sim.synapses.weight.iter().enumerate() {
            if si < syn_pre_snap.len() {
                writeln!(
                    csv,
                    "{},{},{},{},{:.6}",
                    trial + 1,
                    si,
                    syn_pre_snap[si],
                    syn_post_snap[si],
                    w
                )
                .unwrap();
            }
        }
    }

    let final_stats = StdpState::weight_stats(&sim.synapses.weight);
    println!("\n=== Weight Evolution Summary ===");
    println!("Initial mean weight: {:.4}", initial_weight);
    println!("Final   mean weight: {:.4}", final_stats.mean);
    println!(
        "Change:              {:.4} ({:+.1}%)",
        final_stats.mean - initial_weight,
        100.0 * (final_stats.mean - initial_weight) / initial_weight
    );
    println!("Total weight updates applied: {}", stdp.update_count);
    println!("Weight time-series → bench/results/stdp_weights.csv");
    println!("Total spikes: {}", sim.spike_log.len());
}
