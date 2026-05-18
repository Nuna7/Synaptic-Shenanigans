use synaptic_shenanigans::neurons::{AdExPopulation, AdExProfile, NeuronPopulation};

fn count_spikes(profile: AdExProfile, i_ext: f32, ms: usize) -> usize {
    let pop = AdExPopulation::from_profile(1, profile);
    let mut n = 0;
    for _ in 0..ms {
        pop.step_range(&[i_ext], 0);
        if pop.local_spiked(0) {
            n += 1;
        }
    }
    n
}

#[test]
fn adex_fires_above_threshold() {
    assert!(count_spikes(AdExProfile::AdaptingRS, 500.0, 500) > 0);
}
#[test]
fn adex_silent_at_zero_current() {
    assert_eq!(count_spikes(AdExProfile::AdaptingRS, 0.0, 500), 0);
}
#[test]
fn adex_adapting_rs_rate_decreases() {
    let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    let mut sp = vec![];
    for step in 0..2000usize {
        pop.step_range(&[600.0], 0);
        if pop.local_spiked(0) {
            sp.push(step);
        }
    }
    let first = sp.iter().filter(|&&s| s < 500).count();
    let second = sp.iter().filter(|&&s| (500..1000).contains(&s)).count();
    assert!(second <= first, "first={first} second={second}");
}
#[test]
fn adex_all_profiles_fire_at_high_current() {
    for p in [
        AdExProfile::AdaptingRS,
        AdExProfile::Bursting,
        AdExProfile::TonicRS,
        AdExProfile::FastSpiking,
        AdExProfile::TransientBurst,
    ] {
        assert!(count_spikes(p, 1000.0, 500) > 0, "{:?} should fire", p);
    }
}
#[test]
fn adex_set_threshold_via_trait() {
    let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    pop.set_threshold(0, -45.0);
    assert!((pop.get_threshold(0) - (-45.0)).abs() < 1e-4);
}
#[test]
fn adex_reset_neuron_via_trait() {
    let pop = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    for _ in 0..200 {
        pop.step_range(&[600.0], 0);
    }
    pop.reset_neuron(0, -70.0);
    assert!((pop.read_v(0) - (-70.0)).abs() < 1e-3);
    assert!(!pop.local_spiked(0));
}
#[test]
fn adex_deterministic_across_runs() {
    let p1 = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    let p2 = AdExPopulation::from_profile(1, AdExProfile::AdaptingRS);
    for _ in 0..500 {
        p1.step_range(&[500.0], 0);
        p2.step_range(&[500.0], 0);
        assert_eq!(p1.local_spiked(0), p2.local_spiked(0));
        assert!((p1.read_v(0) - p2.read_v(0)).abs() < 1e-4);
    }
}
