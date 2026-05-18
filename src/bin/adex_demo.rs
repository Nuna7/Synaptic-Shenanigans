//! AdEx neuron showcase. Run: cargo run --release --bin adex_demo
use synaptic_shenanigans::neurons::{AdExPopulation, AdExProfile, NeuronPopulation};
use std::io::Write;

fn main() {
    println!("=== AdEx Neuron Model Demo ===\n");
    std::fs::create_dir_all("bench/results").unwrap();
    let profiles = [AdExProfile::AdaptingRS, AdExProfile::Bursting, AdExProfile::TonicRS,
                    AdExProfile::FastSpiking, AdExProfile::TransientBurst];

    let mut trace_csv = std::fs::File::create("bench/results/adex_traces.csv").unwrap();
    writeln!(trace_csv, "t_ms,profile,v_mV,w_pA").unwrap();

    println!("{:<20}  {:>10}  {:>12}  {:>12}", "Profile", "Spikes/s", "Mean ISI(ms)", "CV(ISI)");
    println!("{}", "-".repeat(60));

    for profile in profiles {
        let pop = AdExPopulation::from_profile(1, profile);
        let mut spike_times: Vec<f32> = Vec::new();
        for step in 0..1000usize {
            pop.step_range(&[500.0f32], 0);
            if pop.local_spiked(0) { spike_times.push(step as f32); }
            if step < 200 {
                writeln!(trace_csv, "{},{},{:.4},{:.4}", step, profile.name(), pop.read_v(0), pop.read_w(0)).unwrap();
            }
        }
        let isis: Vec<f32> = spike_times.windows(2).map(|w| w[1]-w[0]).collect();
        let mean_isi = if isis.is_empty() { f32::NAN } else { isis.iter().sum::<f32>() / isis.len() as f32 };
        let cv = if isis.len() < 2 { f32::NAN } else {
            let m = mean_isi;
            (isis.iter().map(|&v|(v-m).powi(2)).sum::<f32>()/isis.len() as f32).sqrt() / m
        };
        println!("{:<20}  {:>10}  {:>12.1}  {:>12.3}", profile.name(), spike_times.len(), mean_isi, cv);
    }

    println!("\n=== F-I Curve (Adapting-RS) ===");
    let mut fi = std::fs::File::create("bench/results/adex_fi_curve.csv").unwrap();
    writeln!(fi, "i_ext_pA,rate_initial_hz,rate_steady_hz").unwrap();
    for &i_level in &[100.0f32, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0] {
        let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
        let (mut early, mut late) = (0, 0);
        for step in 0..2000 {
            pop.step_range(&[i_level], 0);
            if pop.local_spiked(0) { if step < 500 { early += 1; } if step > 1500 { late += 1; } }
        }
        println!("  I={:>6.0} pA  initial={:>5.1} Hz  steady={:>5.1} Hz", i_level, early as f32*2.0, late as f32*2.0);
        writeln!(fi, "{},{:.2},{:.2}", i_level, early as f32*2.0, late as f32*2.0).unwrap();
    }

    println!("\n=== Heterogeneous Population (50 Adapting-RS neurons) ===");
    let pop = AdExPopulation::heterogeneous(50, AdExProfile::AdaptingRS, 0.15, 42);
    let mut total = 0usize;
    for _ in 0..1000 { pop.step_range(&[500.0f32; 50], 0); total += (0..50).filter(|&i| pop.local_spiked(i)).count(); }
    println!("  Total spikes: {}   Mean rate: {:.1} Hz", total, total as f32/50.0);
    println!("\nOutputs: bench/results/adex_traces.csv  bench/results/adex_fi_curve.csv");
}