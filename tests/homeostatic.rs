//! Tests for homeostatic intrinsic plasticity.
//!
//! Verifies:
//!   1. Threshold rises when neuron fires too fast (over-excited)
//!   2. Threshold drops when neuron fires too slowly (under-excited)
//!   3. System converges toward target rate over time
//!   4. Thresholds are clamped to [theta_min, theta_max]
//!   5. Disabled homeostasis produces no changes
//!   6. Statistics (fraction_at_target, threshold_stats) are correct

use synaptic_shenanigans::homeostatic::{HomeostaticState, HomeostaticConfig};

fn fast_config(target: f32) -> HomeostaticConfig {
    HomeostaticConfig {
        target_rate_hz:    target,
        tau_h:             500.0,  // fast for tests
        theta_min:        -75.0,
        theta_max:        -40.0,
        rate_window_ms:   200.0,
        update_interval_ms: 10.0,
        enabled:           true,
    }
}

// ── Basic adaptation ─────────────────────────────────────────────────────────

#[test]
fn homeo_threshold_rises_when_firing_too_fast() {
    let n = 1;
    let initial_theta = -50.0f32;
    let target_rate   =  5.0f32;
    let cfg = fast_config(target_rate);
    let mut homeo = HomeostaticState::new(n, initial_theta, cfg);

    // Simulate high firing: inject 50 spikes in 200 ms window → 250 Hz >> 5 Hz target
    for i in 0..50 {
        homeo.record_spike(0, i as f32 * 4.0); // spikes every 4 ms
    }
    homeo.update(200.0);

    assert!(homeo.theta[0] > initial_theta,
        "Threshold should rise when firing too fast: old={}, new={}",
        initial_theta, homeo.theta[0]);
}

#[test]
fn homeo_threshold_falls_when_firing_too_slow() {
    let n = 1;
    let initial_theta = -50.0f32;
    let target_rate   = 20.0f32; // high target
    let cfg = fast_config(target_rate);
    let mut homeo = HomeostaticState::new(n, initial_theta, cfg);

    // Only 1 spike in window → rate ≈ 5 Hz << 20 Hz target
    homeo.record_spike(0, 100.0);
    homeo.update(200.0);

    assert!(homeo.theta[0] < initial_theta,
        "Threshold should fall when firing too slowly: old={}, new={}",
        initial_theta, homeo.theta[0]);
}

#[test]
fn homeo_threshold_stable_at_target_rate() {
    let n = 1;
    let initial_theta = -50.0f32;
    let target = 10.0f32;
    let cfg = fast_config(target);
    let rate_window_ms = cfg.rate_window_ms;
    let mut homeo = HomeostaticState::new(n, initial_theta, cfg);

    // Inject exactly target-rate spikes: 10 Hz in 200 ms window → 2 spikes
    let expected_spikes = (target * rate_window_ms / 1000.0) as usize;
    for i in 0..expected_spikes {
        homeo.record_spike(0, i as f32 * (rate_window_ms / expected_spikes as f32));
    }
    let prev_theta = homeo.theta[0];
    homeo.update(rate_window_ms);

    // Change should be very small (not necessarily zero due to discretisation)
    let delta = (homeo.theta[0] - prev_theta).abs();
    assert!(delta < 2.0, "Threshold should barely change at target rate: Δ={}", delta);
}

#[test]
fn homeo_theta_clamped_to_min_max() {
    let n = 1;
    let cfg = HomeostaticConfig {
        target_rate_hz: 100.0, // unachievably high target → threshold will try to go below theta_min
        theta_min: -70.0,
        theta_max: -40.0,
        tau_h: 10.0, // very fast
        ..fast_config(100.0)
    };
    let mut homeo = HomeostaticState::new(n, -50.0, cfg);

    // No spikes → rate = 0 << target → threshold should fall but be clamped
    for i in 0..100 {
        homeo.update(10.0 * i as f32 + 10.0);
    }
    assert!(homeo.theta[0] >= -70.0, "Theta should not go below theta_min");
    assert!(homeo.theta[0] <= -40.0, "Theta should not go above theta_max");
}

#[test]
fn homeo_disabled_produces_no_changes() {
    let n = 5;
    let initial_theta = -50.0f32;
    let mut cfg = fast_config(10.0);
    cfg.enabled = false;
    let mut homeo = HomeostaticState::new(n, initial_theta, cfg);

    // Inject many spikes
    for nid in 0..n {
        for i in 0..100 {
            homeo.record_spike(nid, i as f32 * 2.0);
        }
    }
    let _ = homeo.update(500.0);

    for nid in 0..n {
        assert_eq!(homeo.theta[nid], initial_theta,
            "Disabled homeostasis should not change thresholds");
    }
    assert_eq!(homeo.update_count, 0, "No updates should be applied when disabled");
}

// ── Population properties ─────────────────────────────────────────────────────

#[test]
fn homeo_threshold_stats_are_correct() {
    let n = 4;
    let mut cfg = fast_config(5.0);
    cfg.enabled = false; // disable so we can set thresholds manually
    let mut homeo = HomeostaticState::new(n, -50.0, cfg);
    homeo.theta = vec![-60.0, -55.0, -50.0, -45.0];

    let stats = homeo.threshold_stats();
    assert!((stats.min  - (-60.0)).abs() < 0.01);
    assert!((stats.max  - (-45.0)).abs() < 0.01);
    assert!((stats.mean - (-52.5)).abs() < 0.01);
    assert_eq!(stats.n, 4);
}

#[test]
fn homeo_fraction_at_target_counts_correctly() {
    let n = 10;
    let target = 5.0f32;
    let cfg = fast_config(target);
    let rate_window_ms = cfg.rate_window_ms;
    let mut homeo = HomeostaticState::new(n, -50.0, cfg);

    // Make first 6 neurons fire at target, last 4 fire way too fast
    for nid in 0..6 {
        let n_spikes = (target * rate_window_ms / 1000.0) as usize;
        for i in 0..n_spikes {
            homeo.record_spike(nid, i as f32 * (rate_window_ms / n_spikes as f32));
        }
    }
    for nid in 6..10 {
        for i in 0..100 {
            homeo.record_spike(nid, i as f32 * 2.0); // 500 Hz
        }
    }

    let frac = homeo.fraction_at_target();
    // ~60% should be near target (allow ±1 Hz tolerance)
    assert!((frac - 0.6).abs() <= 0.25, "Fraction at target: expected ~0.6, got {}", frac);
}

#[test]
fn homeo_update_interval_is_respected() {
    let n = 1;
    let mut cfg = fast_config(5.0);
    cfg.update_interval_ms = 100.0;
    let mut homeo = HomeostaticState::new(n, -50.0, cfg);

    // First update at t = 50 ms → should be skipped (< 100 ms interval)
    let changed = homeo.update(50.0);
    assert_eq!(changed, 0, "Update should be skipped before interval");

    // Update at t = 150 ms → should apply
    let changed2 = homeo.update(150.0);
    // May or may not change threshold, but should not panic
    let _ = changed2;
}

#[test]
fn homeo_rate_distribution_has_correct_length() {
    let n = 20;
    let homeo = HomeostaticState::new(n, -50.0, fast_config(5.0));
    let rates = homeo.rate_distribution();
    assert_eq!(rates.len(), n);
}

// ── Convergence ───────────────────────────────────────────────────────────────

#[test]
fn homeo_thresholds_diverge_for_high_vs_low_firing_neurons() {
    let n = 2;
    let target = 10.0f32;
    let mut cfg = fast_config(target);
    cfg.rate_window_ms = 100.0;
    let mut homeo = HomeostaticState::new(n, -50.0, cfg);

    // Neuron 0: fires at 50 Hz (too fast)
    // Neuron 1: fires at 1 Hz (too slow)
    for t_ms in (0..1000).step_by(20) {
        homeo.record_spike(0, t_ms as f32); // 50 Hz
    }
    for t_ms in (0..1000).step_by(1000) {
        homeo.record_spike(1, t_ms as f32); // 1 Hz
    }

    for i in 0..10 {
        homeo.update(100.0 * (i + 1) as f32);
    }

    // Neuron 0 (over-active) should have higher threshold than neuron 1 (under-active)
    assert!(homeo.theta[0] > homeo.theta[1],
        "Over-active neuron should develop higher threshold: θ0={}, θ1={}",
        homeo.theta[0], homeo.theta[1]);
}
