//! Population metrics demo. Run: cargo run --release --bin metrics_demo
use std::io::Write;
use synaptic_shenanigans::input::poisson::{PoissonPopulation, StimulusPattern};
use synaptic_shenanigans::metrics::{BurstDetector, ISIStats, SynchronyIndex, dominant_frequency};
use synaptic_shenanigans::network::{EdgeParams, NetworkBuilder};
use synaptic_shenanigans::simulation::SchedulerMode;
use synaptic_shenanigans::{LifNeuron, Simulation};

fn run_regime(
    name: &'static str,
    n: usize,
    input_rate: f32,
    weight: f32,
    seed: u64,
) -> (Vec<(f32, usize)>, usize, f32) {
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
    let mut sim = Simulation::new_with_neurons(neurons, syn, 1.0, seed, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;
    let dur = 2000.0f32;
    let mut pop = PoissonPopulation::new(n, input_rate, 60.0, seed + 99);
    pop.prebuild(&mut sim, dur);
    sim.run_auto(dur);
    println!("  {} done ({} spikes)", name, sim.spike_log.len());
    (sim.spike_log, n, dur)
}

fn main() {
    println!("=== Population Metrics Demo ===\n");
    std::fs::create_dir_all("bench/results").unwrap();
    let n = 200usize;

    let regimes = [
        run_regime("Async-Irregular", n, 8.0, 3.0, 42),
        run_regime("Synchronous-Bursting", n, 3.0, 12.0, 43),
        {
            let ep = EdgeParams {
                weight: 3.0,
                ..EdgeParams::default()
            };
            let syn = NetworkBuilder::small_world(n, 4, 0.05, ep, 44);
            let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
            let mut sim = Simulation::new_with_neurons(neurons, syn, 1.0, 44, 1);
            sim.scheduler_mode = SchedulerMode::SingleThreaded;
            let dur = 2000.0f32;
            let mut pat = StimulusPattern::sinusoidal(10.0, 15.0, 40.0, 44);
            for nid in 0..n {
                for t in pat.generate(0.0, dur) {
                    sim.push_event(t, nid, 80.0, 0, 0.0);
                }
            }
            sim.run_auto(dur);
            println!("  Oscillatory done ({} spikes)", sim.spike_log.len());
            (sim.spike_log, n, dur)
        },
    ];
    let names = [
        "Async-Irregular",
        "Synchronous-Bursting",
        "Oscillatory(40Hz)",
    ];

    let mut csv = std::fs::File::create("bench/results/metrics_comparison.csv").unwrap();
    writeln!(
        csv,
        "regime,n_spikes,rate_hz,chi,cv,fano,dominant_freq_hz,n_bursts"
    )
    .unwrap();

    for ((spikes, n, dur), name) in regimes.iter().zip(names.iter()) {
        let sync = SynchronyIndex::compute(spikes, *n, *dur, 5.0);
        let isi = ISIStats::compute(spikes, *n, *dur, 5.0);
        let rate = spikes.len() as f32 / (*n as f32 * dur / 1000.0);
        let bursts = BurstDetector::new(*n, 15.0, 5.0).detect(spikes, *dur);
        let dom = dominant_frequency(spikes, *n, *dur, 1.0, 1.0, 100.0).unwrap_or(0.0);
        println!(
            "{:<35} spikes={} rate={:.1}Hz χ={:.4} CV={:.3} Fano={:.3} dom={:.1}Hz bursts={}",
            name,
            spikes.len(),
            rate,
            sync.chi,
            isi.cv,
            isi.fano_factor,
            dom,
            bursts.len()
        );
        writeln!(
            csv,
            "{},{},{:.3},{:.4},{:.3},{:.3},{:.1},{}",
            name,
            spikes.len(),
            rate,
            sync.chi,
            isi.cv,
            isi.fano_factor,
            dom,
            bursts.len()
        )
        .unwrap();
    }

    println!("\nResults → bench/results/metrics_comparison.csv");
}
