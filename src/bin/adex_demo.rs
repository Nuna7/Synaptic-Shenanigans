//! AdEx neuron showcase: all 5 profiles, F-I curve, adaptation traces.
//!
//! Run: cargo run --release --bin adex_demo

use synaptic_shenanigans::adex::{AdExPopulation, AdExProfile};
use synaptic_shenanigans::lif::NeuronPopulation;
use std::io::Write;

fn main() {
    println!("=== AdEx Neuron Model Demo ===\n");
    std::fs::create_dir_all("bench/results").unwrap();

    let profiles = [
        AdExProfile::AdaptingRS,
        AdExProfile::Bursting,
        AdExProfile::TonicRS,
        AdExProfile::FastSpiking,
        AdExProfile::TransientBurst,
    ];

    // ── 1. Profile comparison at fixed current ───────────────────────────────
    println!("{:<20}  {:>10}  {:>12}  {:>12}",
        "Profile", "Spikes/s", "Mean ISI(ms)", "CV(ISI)");
    println!("{}", "-".repeat(60));

    let mut trace_csv = std::fs::File::create("bench/results/adex_traces.csv").unwrap();
    writeln!(trace_csv, "t_ms,profile,v_mV,w_pA").unwrap();

    for profile in profiles {
        let pop = AdExPopulation::from_profile(1, profile);
        let sim_ms = 1000usize;
        let i_ext  = 500.0f32;

        let mut spike_times: Vec<f32> = Vec::new();
        for step in 0..sim_ms {
            pop.step_range(&[i_ext], 0);
            if pop.local_spiked(0) { spike_times.push(step as f32); }
            if step < 200 {
                writeln!(trace_csv, "{},{},{:.4},{:.4}",
                    step, profile.name(), pop.read_v(0), pop.read_w(0)).unwrap();
            }
        }

        let isis: Vec<f32> = spike_times.windows(2).map(|w| w[1]-w[0]).collect();
        let mean_isi = if isis.is_empty() { f32::NAN }
                       else { isis.iter().sum::<f32>() / isis.len() as f32 };
        let cv = if isis.len() < 2 { f32::NAN } else {
            let m = mean_isi;
            let s = (isis.iter().map(|&v|(v-m).powi(2)).sum::<f32>()/isis.len() as f32).sqrt();
            s / m
        };
        println!("{:<20}  {:>10}  {:>12.1}  {:>12.3}",
            profile.name(), spike_times.len(), mean_isi, cv);
    }

    // ── 2. AdEx F-I curve ────────────────────────────────────────────────────
    println!("\n=== F-I Curve (Adapting-RS) ===");
    let mut fi = std::fs::File::create("bench/results/adex_fi_curve.csv").unwrap();
    writeln!(fi, "i_ext_pA,rate_initial_hz,rate_steady_hz").unwrap();

    for &i_level in &[100.0f32, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0] {
        let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
        let mut spikes_early = 0usize;
        let mut spikes_late  = 0usize;
        for step in 0..2000 {
            pop.step_range(&[i_level], 0);
            if pop.local_spiked(0) {
                if step <  500 { spikes_early += 1; }
                if step > 1500 { spikes_late  += 1; }
            }
        }
        let rate_init   = spikes_early as f32 * 2.0; // 500 ms → Hz
        let rate_steady = spikes_late  as f32 * 2.0;
        println!("  I={:>6.0} pA  initial={:>5.1} Hz  steady={:>5.1} Hz",
            i_level, rate_init, rate_steady);
        writeln!(fi, "{},{:.2},{:.2}", i_level, rate_init, rate_steady).unwrap();
    }

    // ── 3. Heterogeneous population ──────────────────────────────────────────
    println!("\n=== Heterogeneous Population (50 Adapting-RS neurons) ===");
    let pop = AdExPopulation::heterogeneous(50, AdExProfile::AdaptingRS, 0.15, 42);
    let mut total_spikes = 0usize;
    for _ in 0..1000 {
        pop.step_range(&vec![500.0f32; 50], 0);
        total_spikes += (0..50).filter(|&i| pop.local_spiked(i)).count();
    }
    println!("  Total spikes: {}   Mean rate: {:.1} Hz",
        total_spikes, total_spikes as f32 / 50.0);

    println!("\nOutputs:");
    println!("  Voltage traces → bench/results/adex_traces.csv");
    println!("  F-I curve      → bench/results/adex_fi_curve.csv");
}