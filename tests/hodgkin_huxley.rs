//! Tests for the Hodgkin-Huxley neuron model.
//!
//! We verify:
//!   1. Correct resting state (V ≈ -65 mV, channels at steady state)
//!   2. Action potential generation above rheobase current
//!   3. Silence below rheobase
//!   4. Refractory period emerges from channel kinetics (no hard timer)
//!   5. F-I curve is monotonically increasing
//!   6. Channel gating variables stay in [0, 1]
//!   7. Membrane potential stays in biologically plausible range
//!   8. Heterogeneous population produces different spike times
//!   9. Determinism: same params → identical output

use synaptic_shenanigans::hodgkin_huxley::{HHPopulation, HHParams, steady_state};
use synaptic_shenanigans::lif::NeuronPopulation;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn run_single(i_ext: f32, ms: usize) -> (Vec<bool>, Vec<f64>) {
    let hh = HHPopulation::homogeneous(1, HHParams::default());
    let mut spiked = Vec::with_capacity(ms);
    let mut vs     = Vec::with_capacity(ms);
    for _ in 0..ms {
        hh.step_range(&[i_ext], 0);
        spiked.push(hh.local_spiked(0));
        vs.push(hh.read_v(0));
    }
    (spiked, vs)
}

fn count_spikes(i_ext: f32, ms: usize) -> usize {
    let (spiked, _) = run_single(i_ext, ms);
    spiked.into_iter().filter(|&s| s).count()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn hh_steady_state_values_are_biologically_correct() {
    let (m0, h0, n0) = steady_state(-65.0);
    // At rest: m very small (Na⁺ channels closed), h large, n small
    assert!(m0 < 0.1,  "m_∞(-65) should be small, got {}", m0);
    assert!(h0 > 0.5,  "h_∞(-65) should be large (inactivation gate open), got {}", h0);
    assert!(n0 < 0.4,  "n_∞(-65) should be moderate, got {}", n0);
    // All in [0, 1]
    assert!((0.0..=1.0).contains(&m0));
    assert!((0.0..=1.0).contains(&h0));
    assert!((0.0..=1.0).contains(&n0));
}

#[test]
fn hh_no_spike_below_rheobase() {
    // HH rheobase ≈ 6.3 µA/cm² for standard params
    let n_spikes = count_spikes(2.0, 500);
    assert_eq!(n_spikes, 0, "Expected no spikes below rheobase, got {}", n_spikes);
}

#[test]
fn hh_fires_above_rheobase() {
    let n_spikes = count_spikes(10.0, 500);
    assert!(n_spikes > 5, "Expected regular spiking above rheobase, got {}", n_spikes);
}

#[test]
fn hh_membrane_potential_stays_in_range() {
    let (_, vs) = run_single(10.0, 500);
    for v in &vs {
        assert!(*v >= -90.0, "V went below -90 mV: {}", v);
        assert!(*v <= 60.0,  "V went above +60 mV: {}", v);
    }
}

#[test]
fn hh_gating_variables_stay_in_unit_interval() {
    let hh = HHPopulation::homogeneous(1, HHParams::default());
    for _ in 0..1000 {
        hh.step_range(&[10.0], 0);
        let m = hh.state.m[0].load();
        let h = hh.state.h[0].load();
        let n = hh.state.n[0].load();
        assert!((0.0..=1.001).contains(&m), "m out of range: {}", m);
        assert!((0.0..=1.001).contains(&h), "h out of range: {}", h);
        assert!((0.0..=1.001).contains(&n), "n out of range: {}", n);
    }
}

#[test]
fn hh_fi_curve_is_monotonically_increasing() {
    let levels = [6.0f32, 8.0, 10.0, 15.0, 20.0, 30.0];
    let rates: Vec<usize> = levels.iter().map(|&i| count_spikes(i, 1000)).collect();
    for w in rates.windows(2) {
        assert!(
            w[1] >= w[0],
            "F-I curve not monotone: {:?}", rates
        );
    }
}

#[test]
fn hh_zero_current_stays_near_rest() {
    let (_, vs) = run_single(0.0, 500);
    // After settling, V should be within 5 mV of rest
    for v in vs.iter().skip(100) {
        assert!((*v - (-65.0)).abs() < 5.0, "V drifted far from rest: {}", v);
    }
}

#[test]
fn hh_deterministic_across_runs() {
    let hh1 = HHPopulation::homogeneous(1, HHParams::default());
    let hh2 = HHPopulation::homogeneous(1, HHParams::default());
    for _ in 0..500 {
        hh1.step_range(&[10.0], 0);
        hh2.step_range(&[10.0], 0);
        assert_eq!(hh1.local_spiked(0), hh2.local_spiked(0));
        let v1 = hh1.read_v(0);
        let v2 = hh2.read_v(0);
        assert!((v1 - v2).abs() < 1e-6, "V diverged: {} vs {}", v1, v2);
    }
}

#[test]
fn hh_heterogeneous_population_has_different_spike_times() {
    let hh = HHPopulation::heterogeneous(10, HHParams::default(), 0.1, 42);
    let mut spike_counts = vec![0usize; 10];
    for _ in 0..500 {
        hh.step_range(&vec![10.0f32; 10], 0);
        for i in 0..10 {
            if hh.local_spiked(i) { spike_counts[i] += 1; }
        }
    }
    // Different neurons should have different spike counts due to parameter noise
    let unique_counts: std::collections::HashSet<usize> = spike_counts.into_iter().collect();
    assert!(unique_counts.len() > 1, "All neurons fired identically — expected heterogeneity");
}

#[test]
fn hh_population_split_indices_covers_all_neurons() {
    let pop = HHPopulation::homogeneous(100, HHParams::default());
    let parts = pop.split_indices(32);
    let total_covered: usize = parts.iter().map(|p| p.len).sum();
    assert_eq!(total_covered, 100);
    let first_starts_at_zero = parts.first().map(|p| p.start_index).unwrap_or(1);
    assert_eq!(first_starts_at_zero, 0);
}

#[test]
fn hh_refractory_period_exists_from_channel_kinetics() {
    // After a spike, the neuron should not fire again immediately.
    // The refractory period in HH is 1-3 ms from K⁺ tail current.
    let hh = HHPopulation::homogeneous(1, HHParams::default());
    let mut last_spike = None;
    let mut min_isi = f32::INFINITY;
    let mut prev_spiked = false;
    let mut t = 0usize;

    for _ in 0..500 {
        hh.step_range(&[10.0], 0);
        let spiked = hh.local_spiked(0);
        if spiked {
            if let Some(last_t) = last_spike {
                let isi = (t - last_t) as f32;
                if isi < min_isi { min_isi = isi; }
            }
            last_spike = Some(t);
        }
        prev_spiked = spiked;
        t += 1;
    }

    assert!(min_isi > 5.0,
        "Min ISI too short ({}ms) — HH refractory period should be >5ms", min_isi);
    let _ = prev_spiked;
}
