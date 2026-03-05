//! Hodgkin-Huxley showcase — compare LIF / Izhikevich / HH on the same stimulus.
//!
//! Run:
//!   cargo run --release --bin hh_demo

use synaptic_shenanigans::hodgkin_huxley::{HHPopulation, HHParams, steady_state};
use synaptic_shenanigans::lif::{LifNeuron, NeuronPopulation};
use synaptic_shenanigans::izhikevich::{IzhikevichPop, NeuronType};
use std::io::Write;

fn main() {
    println!("=== Hodgkin-Huxley Model Showcase ===\n");

    std::fs::create_dir_all("bench/results").unwrap();

    // ── 1. Resting-state gating variables ───────────────────────────────────
    let v_rest = -65.0f64;
    let (m0, h0, n0) = steady_state(v_rest);
    println!("Steady-state at V = {:.1} mV:", v_rest);
    println!("  m (Na⁺ activation):     {:.6}", m0);
    println!("  h (Na⁺ inactivation):   {:.6}", h0);
    println!("  n (K⁺  activation):     {:.6}", n0);
    println!();

    // ── 2. Single-neuron F-I curve (firing rate vs input current) ──────────
    println!("=== F-I Curve (firing rate vs. injected current) ===");
    println!("{:>10}  {:>12}  {:>12}", "I (µA/cm²)", "HH rate (Hz)", "LIF rate (Hz)");

    let dt_coarse = 1.0f32;  // 1 ms per step_range call
    let sim_ms    = 1000usize;
    let params    = HHParams::default();

    let mut fi_csv = std::fs::File::create("bench/results/hh_fi_curve.csv").unwrap();
    writeln!(fi_csv, "i_ext,hh_rate_hz,lif_rate_hz").unwrap();

    for &i_level in &[0.0f32, 1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0] {
        // HH
        let hh = HHPopulation::homogeneous(1, params.clone());
        let mut hh_spikes = 0usize;
        for _ in 0..sim_ms {
            hh.step_range(&[i_level], 0);
            if hh.local_spiked(0) { hh_spikes += 1; }
        }
        let hh_rate = hh_spikes as f32;  // spikes/s since sim_ms = 1000

        // LIF (comparable parameter regime)
        let lif = LifNeuron::new(1, -65.0, -50.0, 20.0, 1.0, dt_coarse, 5.0);
        let mut lif_spikes = 0usize;
        for _ in 0..sim_ms {
            lif.step_range(&[i_level * 10.0], 0); // scale for comparable rates
            if lif.local_spiked(0) { lif_spikes += 1; }
        }
        let lif_rate = lif_spikes as f32;

        println!("{:>10.1}  {:>12.1}  {:>12.1}", i_level, hh_rate, lif_rate);
        writeln!(fi_csv, "{},{:.2},{:.2}", i_level, hh_rate, lif_rate).unwrap();
    }

    // ── 3. Voltage trace comparison ─────────────────────────────────────────
    println!("\n=== Voltage Trace — HH vs Izhikevich vs LIF ===");

    let trace_ms = 200usize;
    let i_stim   = 10.0f32;  // µA/cm²
    let i_lif    = 200.0f32; // scaled for LIF

    // HH trace
    let hh = HHPopulation::homogeneous(1, params.clone());
    let mut hh_trace: Vec<f64> = Vec::with_capacity(trace_ms);
    let mut hh_spike_times: Vec<usize> = Vec::new();
    for step in 0..trace_ms {
        hh.step_range(&[i_stim], 0);
        hh_trace.push(hh.read_v(0));
        if hh.local_spiked(0) { hh_spike_times.push(step); }
    }

    // Izhikevich (RS) trace
    let izh = IzhikevichPop::homogeneous(1, NeuronType::RegularSpiking, 0.25);
    let mut izh_trace: Vec<f32> = Vec::with_capacity(trace_ms * 4);
    let mut izh_spike_steps: Vec<usize> = Vec::new();
    for step in 0..(trace_ms * 4) {
        izh.step_range(&[i_stim], 0);
        izh_trace.push(izh.read_v(0) as f32);
        if izh.local_spiked(0) { izh_spike_steps.push(step); }
    }

    // LIF trace
    let lif = LifNeuron::new(1, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut lif_trace: Vec<f32> = Vec::with_capacity(trace_ms);
    let mut lif_spike_times: Vec<usize> = Vec::new();
    for step in 0..trace_ms {
        lif.step_range(&[i_lif], 0);
        lif_trace.push(lif.read_v(0));
        if lif.local_spiked(0) { lif_spike_times.push(step); }
    }

    println!("  HH spikes in {} ms: {}  (rate = {} Hz)", trace_ms, hh_spike_times.len(),
             hh_spike_times.len() * 1000 / trace_ms);
    println!("  Izhikevich spikes in {} ms: {}  (rate = {} Hz)", trace_ms, izh_spike_steps.len() / 4,
             izh_spike_steps.len() * 1000 / (trace_ms * 4));
    println!("  LIF spikes in {} ms: {}  (rate = {} Hz)", trace_ms, lif_spike_times.len(),
             lif_spike_times.len() * 1000 / trace_ms);

    // Write unified trace CSV
    let mut trace_csv = std::fs::File::create("bench/results/hh_traces.csv").unwrap();
    writeln!(trace_csv, "t_ms,hh_v_mV,izh_v_mV,lif_v_mV").unwrap();
    for step in 0..trace_ms {
        let izh_v = izh_trace.get(step * 4).cloned().unwrap_or(f32::NAN);
        writeln!(trace_csv, "{},{:.4},{:.4},{:.4}",
            step, hh_trace[step], izh_v, lif_trace[step]).unwrap();
    }

    // ── 4. Heterogeneous population ─────────────────────────────────────────
    println!("\n=== Heterogeneous HH Population (100 neurons, 5% parameter noise) ===");
    let het = HHPopulation::heterogeneous(100, params.clone(), 0.05, 42);
    let mut total_spikes = 0usize;
    for step in 0..1000 {
        let i_ext: Vec<f32> = (0..100).map(|i| {
            if i < 50 { i_stim } else { 0.0 }  // drive first 50, silence rest
        }).collect();
        het.step_range(&i_ext, 0);
        for nid in 0..100 { if het.local_spiked(nid) { total_spikes += 1; } }
    }
    let mean_rate = total_spikes as f32 / 100.0; // spikes/neuron/s
    println!("  Total spikes: {}   Mean rate: {:.1} Hz", total_spikes, mean_rate);
    println!("  (First 50 neurons driven, last 50 silent)");

    // ── 5. Gating variable traces (the unique HH story) ────────────────────
    println!("\n=== Channel Gating Variables During a Spike ===");
    let single = HHPopulation::homogeneous(1, params);
    let mut gate_csv = std::fs::File::create("bench/results/hh_gating.csv").unwrap();
    writeln!(gate_csv, "t_ms,v_mV,m,h,n,i_na,i_k").unwrap();

    for step in 0..100 {
        let stim = if step < 50 { i_stim } else { 0.0 };
        single.step_range(&[stim], 0);
        let v = single.read_v(0);
        let m = single.state.m[0].load();
        let h = single.state.h[0].load();
        let n = single.state.n[0].load();
        // compute currents from state
        let params2 = HHParams::default();
        let i_na = params2.g_na * m*m*m * h * (v - params2.e_na);
        let i_k  = params2.g_k  * n*n*n*n   * (v - params2.e_k);
        writeln!(gate_csv, "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            step, v, m, h, n, i_na, i_k).unwrap();
    }

    println!("Results written:");
    println!("  F-I curve       → bench/results/hh_fi_curve.csv");
    println!("  Voltage traces  → bench/results/hh_traces.csv");
    println!("  Gating vars     → bench/results/hh_gating.csv");
}
