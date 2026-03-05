//! STDP demo — learning via spike-timing dependent plasticity.
//!
//! Sets up two small populations (pre → post), repeatedly presents stimuli,
//! and shows that synaptic weights increase for causally connected pairs
//! and decrease for acausal pairs.
//!
//! Run:
//!   cargo run --release --bin stdp_demo

use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::synapse::Synapse;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};
use synaptic_shenanigans::plasticity::{StdpState, StdpConfig};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("=== STDP Learning Demo ===\n");

    // ---- Setup -------------------------------------------------------------
    // 10 pre-synaptic + 10 post-synaptic neurons.
    // Neurons 0-9: pre-population (stimulus-driven)
    // Neurons 10-19: post-population (receives synaptic input from pre)
    let n_pre  = 10usize;
    let n_post = 10usize;
    let n_total = n_pre + n_post;

    let neurons = LifNeuron::new(
        n_total,
        -65.0,
        -54.0,   // ← easier to reach
        30.0,    // ← slower leak, better summation
        40.0,    // ← if r_m is resistance (MΩ), much less leaky
        1.0,
        5.0,
    );

    let mut syn = Synapse::new();

    // All-to-all pre→post connections, initial weight = 2.0
    let initial_weight = 3.0;
    for i in 0..n_pre {
        for j in 0..n_post {
            syn.add_current_based(i, n_pre + j, initial_weight, 1.0, 2.0, 1);
        }
    }


    let n_synapses = syn.len();
    println!("Network: {} pre, {} post, {} synapses", n_pre, n_post, n_synapses);

    let mut sim = Simulation::new_with_seed(neurons, syn, 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;

    // ---- STDP state --------------------------------------------------------
    let stdp_cfg = StdpConfig {
        a_plus:    0.01,
        a_minus:   0.0085,
        tau_plus:  16.8,
        tau_minus: 33.7,
        w_min:     0.0,
        w_max:     15.0,
        enabled:   true,
    };

    let mut stdp = StdpState::new(n_total, n_synapses, stdp_cfg);

    // ---- CSV output --------------------------------------------------------
    std::fs::create_dir_all("bench/results").unwrap();
    let mut csv = File::create("bench/results/stdp_weights.csv").unwrap();
    writeln!(csv, "trial,syn_idx,pre,post,weight").unwrap();

    // ---- Training loop -----------------------------------------------------
    // 50 trials: present pre-population stimulus, let post-population respond.
    // STDP rule updates weights after each trial based on spike timing.
    let n_trials = 50;
    let trial_duration = 150.0f32; // ms per trial
    let stim_weight   = 140.0f32; // strong enough to drive pre-neurons to fire

    let syn_pre_snapshot  = std::sync::Arc::clone(&sim.synapses).pre.clone();
    let syn_post_snapshot = std::sync::Arc::clone(&sim.synapses).post.clone();
    let mut weight_history: Vec<Vec<f32>> = Vec::new();

        // ---- Training loop (single continuous simulation) ----------------------
    let total_duration = n_trials as f32 * trial_duration;

    println!("\n{:>6}  {:>12}  {:>12}  {:>12}",
        "Trial", "Mean W", "Max W", "STDP Updates");

    for trial in 0..n_trials {
        let t_start = trial as f32 * trial_duration;

        // Inject stimuli for this trial
        for i in 0..n_pre {
            sim.push_event(t_start + 2.0, i, stim_weight, 0, 0.0);
        }
  

        // Advance simulation exactly 50 ms
        sim.run_auto(t_start + trial_duration);

        for i in 0..n_total {
            sim.neurons.v[i].store(-65.0);
            sim.neurons.refractory[i].store(false);
            sim.neurons.refractory_timer[i].store(0.0);
            sim.neurons.spiked[i].store(false);
        }

        // Collect spikes that occurred during this trial
        let trial_spikes: Vec<(f32, usize)> = sim.spike_log
            .iter()
            .filter(|&&(t, _)| t >= t_start && t < t_start + trial_duration)
            .cloned()
            .collect();

        let mut trial_spikes = trial_spikes;
        trial_spikes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Apply STDP using sorted spikes
        let mut last_t = t_start;
        for &(t, nid) in &trial_spikes {
            stdp.decay_traces(t - last_t);
            last_t = t;

            stdp.accumulate_for_spike(
                nid, t,
                &sim.synapses.pre,
                &sim.synapses.post,
                &sim.pre_index,
            );
        }

        // Flush updates to weights
        let updated = if let Some(w) = std::sync::Arc::get_mut(&mut sim.synapses)
            .map(|s| &mut s.weight) {
            stdp.flush_weight_updates(w)
        } else {
            0
        };

        // Report
        let current_weights = sim.synapses.weight.clone();
        let stats = StdpState::weight_stats(&current_weights);

        println!("{:>6}  {:>12.4}  {:>12.4}  {:>12}",
            trial + 1, stats.mean, stats.max, updated);

        // CSV logging (optional — can be moved outside loop if too slow)
        for (syn_idx, &w) in current_weights.iter().enumerate() {
            if syn_idx < syn_pre_snapshot.len() {
                writeln!(csv, "{},{},{},{},{:.6}",
                    trial + 1, syn_idx,
                    syn_pre_snapshot[syn_idx],
                    syn_post_snapshot[syn_idx],
                    w).unwrap();
            }
        }
    }

    // ---- Summary (now using the last weights) ------------------------------
    println!("\n=== Weight Evolution Summary ===");
    let final_weights = sim.synapses.weight.clone();
    let final_stats = StdpState::weight_stats(&final_weights);
    let initial_weights = vec![3.0f32; n_synapses]; // or load from first snapshot
    let initial_mean = 3.0;
    let final_mean = final_stats.mean;

    println!("Initial mean weight: {:.4}", initial_mean);
    println!("Final   mean weight: {:.4}", final_mean);
    println!("Change:              {:.4} ({:+.1}%)",
        final_mean - initial_mean,
        100.0 * (final_mean - initial_mean) / initial_mean);
    println!("Total weight updates applied: {}", stdp.update_count);
    println!("\nWeight time-series → bench/results/stdp_weights.csv");
    println!("Total spikes in simulation:   {}", sim.spike_log.len());

    // for trial in 0..n_trials {
    //     let t_start = trial as f32 * trial_duration;

    //     // Stimulate pre-neurons (cause)
    //     for i in 0..n_pre {
    //         sim.push_event(t_start + 2.0, i, stim_weight, 0, 0.0);
    //     }

    //     for j in 0..n_post {
    //         sim.push_event(t_start + 5.0, n_pre + j, 18.0, 0, 0.0);   
    //     }

    //     // Run the trial
    //     sim.run_auto(t_start + trial_duration);

    //     // Extract spikes from this trial
    //     let trial_spikes: Vec<(f32, usize)> = sim.spike_log
    //         .iter()
    //         .filter(|&&(t, _)| t >= t_start && t < t_start + trial_duration)
    //         .cloned()
    //         .collect();

    //     let mut trial_spikes = trial_spikes;
    //     trial_spikes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    //     // Apply STDP based on spike timing
    //     let mut last_t = t_start;
    //     for &(t, nid) in &trial_spikes {      // ← sorted by time
    //         stdp.decay_traces(t - last_t);
    //         last_t = t;

    //         stdp.accumulate_for_spike(
    //             nid, t,
    //             &sim.synapses.pre,    
    //             &sim.synapses.post,
    //             &sim.pre_index,
    //         );
    //     }


    //     // Flush weight updates into synapse structure
    //     let weights = std::sync::Arc::get_mut(&mut sim.synapses)
    //         .map(|s| &mut s.weight);

    //     let updated = if let Some(w) = weights {
    //         stdp.flush_weight_updates(w)
    //     } else {
    //         0
    //     };

    //     // Sample current weights
    //     let current_weights = sim.synapses.weight.clone();
    //     let stats = StdpState::weight_stats(&current_weights);
    //     weight_history.push(current_weights.clone());

    //     println!("{:>6}  {:>12.4}  {:>12.4}  {:>12}",
    //         trial + 1, stats.mean, stats.max, updated);
 
    //     // Write CSV
    //     for (syn_idx, &w) in current_weights.iter().enumerate() {
    //         if syn_idx < syn_pre_snapshot.len() {
    //             writeln!(csv, "{},{},{},{},{:.6}",
    //                 trial + 1, syn_idx,
    //                 syn_pre_snapshot[syn_idx],
    //                 syn_post_snapshot[syn_idx],
    //                 w).unwrap();
    //         }
    //     }
    // }

    // ---- Summary -----------------------------------------------------------
    println!("\n=== Weight Evolution Summary ===");
    if let (Some(first), Some(last)) = (weight_history.first(), weight_history.last()) {
        let first_mean = first.iter().sum::<f32>() / first.len() as f32;
        let last_mean  = last.iter().sum::<f32>() / last.len() as f32;
        println!("Initial mean weight: {:.4}", first_mean);
        println!("Final   mean weight: {:.4}", last_mean);
        println!("Change:              {:.4} ({:+.1}%)",
            last_mean - first_mean,
            100.0 * (last_mean - first_mean) / first_mean);
    }
    println!("Total weight updates applied: {}", stdp.update_count);
    println!("\nWeight time-series → bench/results/stdp_weights.csv");
    println!("Total spikes in simulation:   {}", sim.spike_log.len());
}
