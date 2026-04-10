//! Population metrics demo — synchrony, bursts, oscillations, and avalanches.
//!
//! Three distinct network regimes are compared:
//!   1. Asynchronous irregular (healthy cortex baseline)
//!   2. Synchronous bursting (low-input / strong coupling)
//!   3. Oscillatory (rhythmic external drive)
//!
//! Run:
//!   cargo run --release --bin metrics_demo

use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};
use synaptic_shenanigans::network::{NetworkBuilder, EdgeParams};
use synaptic_shenanigans::poisson::{PoissonPopulation, StimulusPattern};
use synaptic_shenanigans::metrics::{SynchronyIndex, BurstDetector, AvalancheResult, ISIStats, dominant_frequency};
use std::io::Write;

struct RegimeResult {
    name: &'static str,
    spikes: Vec<(f32, usize)>,
    n: usize,
    dur: f32,
}

fn run_regime(name: &'static str, n: usize, input_rate: f32, weight: f32, seed: u64) -> RegimeResult {
    // Small-world connectivity
    let ep = EdgeParams {
        weight,
        delay: 1.5,
        inhibitory_fraction: 0.2,
        inh_weight_scale: 3.0,
        tau_syn: 5.0,
        e_inh: -70.0,
    };
    let syn = NetworkBuilder::small_world(n, 6, 0.1, ep, seed);
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_seed(neurons, syn, 1.0, seed, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;

    let dur = 2000.0f32;
    let mut pop = PoissonPopulation::new(n, input_rate, 60.0, seed + 99);
    pop.prebuild(&mut sim, dur);
    sim.run_auto(dur);

    RegimeResult { name, spikes: sim.spike_log.clone(), n, dur }
}

fn run_oscillatory(n: usize, seed: u64) -> RegimeResult {
    let ep = EdgeParams { weight: 3.0, ..EdgeParams::default() };
    let syn = NetworkBuilder::small_world(n, 4, 0.05, ep, seed);
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_seed(neurons, syn, 1.0, seed, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;

    let dur = 2000.0f32;
    // Sinusoidal drive at gamma frequency (40 Hz)
    let mut pattern = StimulusPattern::sinusoidal(10.0, 15.0, 40.0, seed);
    for nid in 0..n {
        let spikes = pattern.generate(0.0, dur);
        for t in spikes {
            sim.push_event(t, nid, 80.0, 0, 0.0);
        }
    }
    sim.run_auto(dur);

    RegimeResult { name: "Oscillatory (40 Hz drive)", spikes: sim.spike_log.clone(), n, dur }
}

fn main() {
    println!("=== Population Metrics Demo ===\n");
    std::fs::create_dir_all("bench/results").unwrap();

    let n = 200usize;

    let regimes = vec![
        run_regime("Async-Irregular (low drive)",   n, 8.0,  3.0, 42),
        run_regime("Synchronous-Bursting (strong)", n, 3.0, 12.0, 43),
        run_oscillatory(n, 44),
    ];

    // ── Print comparison table ───────────────────────────────────────────────
    println!("{:<35} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "Regime", "Spikes", "Rate(Hz)", "χ sync", "CV", "Fano", "Dom.Freq");
    println!("{}", "─".repeat(90));

    let mut csv = std::fs::File::create("bench/results/metrics_comparison.csv").unwrap();
    writeln!(csv, "regime,n_spikes,rate_hz,chi,cv,fano,dominant_freq_hz,n_bursts").unwrap();

    for r in &regimes {
        let spikes = &r.spikes;
        let n = r.n;
        let dur = r.dur;

        let sync  = SynchronyIndex::compute(spikes, n, dur, 5.0);
        let isi   = ISIStats::compute(spikes, n, dur, 5.0);
        let rate  = spikes.len() as f32 / (n as f32 * dur / 1000.0);

        let detector = BurstDetector::new(n, 15.0, 5.0);
        let bursts = detector.detect(spikes, dur);

        let dom_freq = dominant_frequency(spikes, n, dur, 1.0, 1.0, 100.0)
            .unwrap_or(0.0);

        println!("{:<35} {:>8} {:>8.1} {:>8.4} {:>8.3} {:>10.3} {:>8.1}",
            r.name, spikes.len(), rate, sync.chi, isi.cv, isi.fano_factor, dom_freq);
        println!("  State: {}  |  Bursts detected: {}", sync.state(), bursts.len());

        writeln!(csv, "{},{},{:.3},{:.4},{:.3},{:.3},{:.1},{}", 
            r.name, spikes.len(), rate, sync.chi, isi.cv,
            isi.fano_factor, dom_freq, bursts.len()).unwrap();

        // Print burst details
        if !bursts.is_empty() {
            println!("  First 3 bursts:");
            for b in bursts.iter().take(3) {
                println!("    t={:.0}–{:.0} ms  peak={:.1} Hz  neurons={} ({:.0}%)",
                    b.t_start, b.t_end, b.peak_rate_hz, b.n_neurons,
                    b.recruitment * 100.0);
            }
        }
        println!();
    }

    // ── Avalanche analysis ───────────────────────────────────────────────────
    println!("=== Neural Avalanche Analysis (AI regime) ===");
    let ai = &regimes[0];
    let av = AvalancheResult::detect(&ai.spikes, ai.dur, 1.0);
    println!("{}", av.summary());
    if !av.sizes.is_empty() {
        let size_mean = av.sizes.iter().sum::<usize>() as f32 / av.sizes.len() as f32;
        println!("  Avalanche count: {}   Mean size: {:.1}", av.sizes.len(), size_mean);
    }

    // ── Power spectrum ───────────────────────────────────────────────────────
    println!("\n=== Dominant Frequencies by Regime ===");
    let bands = [
        ("Theta (4-8 Hz)",   4.0f32,   8.0),
        ("Alpha (8-13 Hz)",  8.0, 13.0),
        ("Beta (13-30 Hz)", 13.0, 30.0),
        ("Gamma (30-80 Hz)",30.0, 80.0),
    ];
    for r in &regimes {
        print!("  {:<35}", r.name);
        for &(band_name, lo, hi) in &bands {
            let f = dominant_frequency(&r.spikes, r.n, r.dur, 1.0, lo, hi);
            if let Some(freq) = f {
                print!("  {}: {:.1} Hz", band_name.split_whitespace().next().unwrap_or(""), freq);
            }
        }
        println!();
    }

    println!("\nResults → bench/results/metrics_comparison.csv");
}
