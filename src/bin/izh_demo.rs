//! Izhikevich neuron type showcase. Run: cargo run --release --bin izh_demo
use synaptic_shenanigans::neurons::{IzhikevichPop, NeuronType, NeuronPopulation};
use std::io::Write;

fn main() {
    println!("=== Izhikevich Neuron Type Demo ===\n");
    let types = [NeuronType::RegularSpiking, NeuronType::IntrinsicallyBursting,
                 NeuronType::Chattering, NeuronType::FastSpiking,
                 NeuronType::LowThresholdSpiking, NeuronType::Resonator];
    let dt = 0.1f32; let t_end = 200.0f32; let i_ext = 10.0f32;
    std::fs::create_dir_all("bench/results").unwrap();

    for nt in &types {
        let name = nt.name();
        let (a,b,c,d) = nt.params();
        println!("Type: {:>4}  a={:.2} b={:.2} c={:.1} d={:.1}", name, a, b, c, d);
        let pop = IzhikevichPop::heterogeneous(&[(*nt, dt, c, d)]);
        let steps = (t_end / dt) as usize;
        let mut spike_times = Vec::new();
        let mut trace = Vec::new();
        for step in 0..steps {
            pop.step_range(&[i_ext], 0);
            let v = pop.read_v(0);
            trace.push((step as f32 * dt, v));
            if pop.local_spiked(0) { spike_times.push(step as f32 * dt); }
        }
        let isis: Vec<f32> = spike_times.windows(2).map(|w| w[1]-w[0]).collect();
        let mean_isi = if isis.is_empty() { f32::NAN } else { isis.iter().sum::<f32>() / isis.len() as f32 };
        let mean_rate = if mean_isi.is_nan() { 0.0 } else { 1000.0 / mean_isi };
        println!("  Spikes: {}   Mean ISI: {:.1} ms   Mean rate: {:.1} Hz", spike_times.len(), mean_isi, mean_rate);
        let path = format!("bench/results/izh_{}.csv", name.to_lowercase());
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "t_ms,v_mV,spiked").unwrap();
        for (t, v) in &trace {
            let spiked = spike_times.iter().any(|&st| (st-t).abs() < dt*0.5);
            writeln!(f, "{:.2},{:.4},{}", t, v, if spiked { 1 } else { 0 }).unwrap();
        }
    }

    println!("\n=== Mixed Cortical Population (800 RS + 200 FS) ===");
    let pop = IzhikevichPop::mixed_cortical(800, 200, 0.1, 42);
    println!("Population size: {}", pop.len());
    use rand::{SeedableRng, Rng};
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let n = pop.len();
    let mut total_spikes = 0usize;
    for _ in 0..(1000usize) {
        let inputs: Vec<f32> = (0..n).map(|_| if rng.gen_range(0.0..1.0f32) < 0.05 { 15.0 } else { 0.0 }).collect();
        pop.step_range(&inputs, 0);
        total_spikes += (0..n).filter(|&i| pop.local_spiked(i)).count();
    }
    println!("Total spikes: {}   Mean population rate: {:.2} Hz", total_spikes, total_spikes as f32/(n as f32*0.1));
    println!("\nVoltage traces → bench/results/izh_*.csv");
}