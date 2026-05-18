//! Hodgkin-Huxley showcase. Run: cargo run --release --bin hh_demo
use synaptic_shenanigans::neurons::{HHPopulation, HHParams, NeuronPopulation};
use synaptic_shenanigans::neurons::hh::steady_state;
use synaptic_shenanigans::LifNeuron;
use synaptic_shenanigans::simulation::Simulation;
use synaptic_shenanigans::synapse::Synapse;
use std::io::Write;

fn main() {
    println!("=== Hodgkin-Huxley Model Showcase ===\n");
    std::fs::create_dir_all("bench/results").unwrap();

    let (m0,h0,n0) = steady_state(-65.0);
    println!("Steady-state at V = -65.0 mV:  m={:.6}  h={:.6}  n={:.6}", m0, h0, n0);

    println!("\n=== F-I Curve ===");
    let mut fi_csv = std::fs::File::create("bench/results/hh_fi_curve.csv").unwrap();
    writeln!(fi_csv, "i_ext,hh_rate_hz,lif_rate_hz").unwrap();

    for &i_level in &[0.0f32,1.0,2.0,5.0,7.5,10.0,15.0,20.0,30.0,50.0] {
        let hh = HHPopulation::homogeneous(1, HHParams::default());
        let mut hh_sp = 0usize;
        for _ in 0..1000 { hh.step_range(&[i_level],0); if hh.local_spiked(0) { hh_sp+=1; } }

        let lif = LifNeuron::new(1,-65.0,-50.0,20.0,1.0,1.0,5.0);
        let lif_sim_neurons = lif;
        let mut lif_sim = Simulation::new_with_neurons(lif_sim_neurons, Synapse::new(), 1.0, 0, 1);
        for step in 0..1000usize { lif_sim.push_event(step as f32, 0, i_level*10.0, 0, 0.0); }
        lif_sim.run_auto(1000.0);
        let lif_sp = lif_sim.spike_log.len();

        println!("{:>10.1}  {:>12.1}  {:>12.1}", i_level, hh_sp as f32, lif_sp as f32);
        writeln!(fi_csv, "{},{:.2},{:.2}", i_level, hh_sp, lif_sp).unwrap();
    }

    println!("\n=== Heterogeneous HH Population (100 neurons, 5% noise) ===");
    let het = HHPopulation::heterogeneous(100, HHParams::default(), 0.05, 42);
    let mut total = 0usize;
    let i_stim = 10.0f32;
    for _ in 0..1000 {
        let i: Vec<f32> = (0..100).map(|i| if i < 50 { i_stim } else { 0.0 }).collect();
        het.step_range(&i,0);
        for n in 0..100 { if het.local_spiked(n) { total+=1; } }
    }
    println!("  Total spikes: {}   Mean rate: {:.1} Hz", total, total as f32/100.0);
    println!("\nResults: bench/results/hh_fi_curve.csv");
}