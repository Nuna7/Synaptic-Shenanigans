//! Tests for the AdEx neuron model.

use synaptic_shenanigans::adex::{AdExPopulation, AdExProfile, AdExParams};
use synaptic_shenanigans::lif::NeuronPopulation;

fn run_n(pop: &AdExPopulation, i_ext: f32, ms: usize) -> (Vec<bool>, Vec<f32>) {
    let mut spikes = Vec::new();
    let mut vs     = Vec::new();
    for _ in 0..ms {
        pop.step_range(&[i_ext], 0);
        spikes.push(pop.local_spiked(0));
        vs.push(pop.read_v(0));
    }
    (spikes, vs)
}

fn count_spikes(profile: AdExProfile, i_ext: f32, ms: usize) -> usize {
    let pop = AdExPopulation::from_profile(1, profile);
    run_n(&pop, i_ext, ms).0.into_iter().filter(|&s| s).count()
}

#[test]
fn adex_fires_above_threshold() {
    let n = count_spikes(AdExProfile::AdaptingRS, 500.0, 500);
    assert!(n > 0, "Should fire above rheobase");
}

#[test]
fn adex_silent_at_zero_current() {
    let n = count_spikes(AdExProfile::AdaptingRS, 0.0, 500);
    assert_eq!(n, 0, "Should be silent at zero input");
}

#[test]
fn adex_adapting_rs_rate_decreases_over_time() {
    let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    let (spikes, _) = run_n(&pop, 600.0, 2000);
    let first_500:  usize = spikes[..500].iter().filter(|&&s| s).count();
    let second_500: usize = spikes[500..1000].iter().filter(|&&s| s).count();
    assert!(second_500 <= first_500,
        "Adapting-RS should show rate decrease: {} -> {}", first_500, second_500);
}

#[test]
fn adex_tonic_rs_has_stable_rate() {
    let pop = AdExPopulation::from_profile(1, AdExProfile::TonicRS);
    let (spikes, _) = run_n(&pop, 500.0, 2000);
    let first:  usize = spikes[200..700].iter().filter(|&&s| s).count();
    let second: usize = spikes[700..1200].iter().filter(|&&s| s).count();
    let diff = (first as i32 - second as i32).abs();
    assert!(diff <= 3, "Tonic-RS should be stable: {} vs {}", first, second);
}

#[test]
fn adex_v_stays_in_range() {
    let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    let (_, vs) = run_n(&pop, 500.0, 1000);
    for v in vs {
        assert!((-100.0..=5.0).contains(&v), "V out of range: {}", v);
    }
}

#[test]
fn adex_w_is_non_negative_for_positive_b() {
    let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    for _ in 0..500 {
        pop.step_range(&[600.0], 0);
    }
    assert!(pop.read_w(0) >= 0.0, "w should be non-negative for positive b");
}

#[test]
fn adex_w_increases_after_spike() {
    let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    let w_before = pop.read_w(0);
    // Apply strong pulse to force a spike
    pop.step_range(&[5000.0], 0);
    if pop.local_spiked(0) {
        let w_after = pop.read_w(0);
        let b = pop.b[0];
        assert!(w_after >= w_before + b * 0.5,
            "w should jump by ~b={} after spike: {} -> {}", b, w_before, w_after);
    }
}

#[test]
fn adex_all_profiles_fire_at_high_current() {
    use AdExProfile::*;
    for profile in [AdaptingRS, Bursting, TonicRS, FastSpiking, TransientBurst] {
        let n = count_spikes(profile, 1000.0, 500);
        assert!(n > 0, "Profile {:?} should fire at high current", profile);
    }
}

#[test]
fn adex_fast_spiking_has_higher_rate_than_adapting_rs() {
    let fs_rate = count_spikes(AdExProfile::FastSpiking,  500.0, 1000);
    let rs_rate = count_spikes(AdExProfile::AdaptingRS,   500.0, 1000);
    // FS has weaker adaptation (b=0) so should sustain higher rate
    assert!(fs_rate >= rs_rate,
        "FS should fire at least as fast as adapting-RS: {} vs {}", fs_rate, rs_rate);
}

#[test]
fn adex_deterministic_across_runs() {
    let p1 = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    let p2 = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    for _ in 0..500 {
        p1.step_range(&[500.0], 0);
        p2.step_range(&[500.0], 0);
        assert_eq!(p1.local_spiked(0), p2.local_spiked(0));
        assert!((p1.read_v(0) - p2.read_v(0)).abs() < 1e-5);
    }
}

#[test]
fn adex_heterogeneous_produces_different_rates() {
    let pop = AdExPopulation::heterogeneous(10, AdExProfile::AdaptingRS, 0.15, 42);
    let mut counts = vec![0usize; 10];
    for _ in 0..1000 {
        pop.step_range(&[500.0f32; 10], 0);
        for (i, spike_count) in counts.iter_mut().enumerate().take(10) {
            if pop.local_spiked(i) { *spike_count += 1; }
        }
    }
    let unique: std::collections::HashSet<usize> = counts.into_iter().collect();
    assert!(unique.len() > 1, "Heterogeneous population should have different rates");
}

#[test]
fn adex_split_indices_covers_all_neurons() {
    let pop = AdExPopulation::from_profile(100, AdExProfile::AdaptingRS);
    let parts = pop.split_indices(32);
    assert_eq!(parts.iter().map(|p| p.len).sum::<usize>(), 100);
    assert_eq!(parts[0].start_index, 0);
}

#[test]
fn adex_custom_params_respected() {
    let p = AdExParams { a: 0.0, b: 0.0, ..AdExParams::default() };
    let pop = AdExPopulation::homogeneous(1, p);
    // With a=0, b=0: w should stay near 0 regardless of activity
    for _ in 0..500 { pop.step_range(&[500.0], 0); }
    assert!(pop.read_w(0).abs() < 5.0, "w should stay small with a=0, b=0");
}