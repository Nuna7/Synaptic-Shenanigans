//! Tests for synaptic scaling.

use synaptic_shenanigans::synaptic_scaling::{SynapticScaling, SynapticScalingConfig};

fn default_config() -> SynapticScalingConfig {
    SynapticScalingConfig {
        target_rate_hz:    5.0,
        alpha:             1.0,
        w_min:             0.0,
        w_max:            20.0,
        update_interval_ms: 10.0,
        rate_window_ms:   200.0,
        enabled:          true,
    }
}

fn make_weights(n: usize, val: f32) -> Vec<f32> { vec![val; n] }
fn make_post(n: usize) -> Vec<usize> { (0..n).map(|i| i % 5).collect() }

#[test]
fn scaling_weights_increase_for_silent_neurons() {
    let n_neurons = 5;
    let mut scaler = SynapticScaling::new(n_neurons, default_config());
    // No spikes → rate = 0 Hz < target 5 Hz → weights should scale UP
    let mut weights = make_weights(10, 2.0);
    let post = make_post(10);
    scaler.scale_weights(500.0, &post, &mut weights);
    assert!(weights.iter().any(|&w| w > 2.0), "Weights should increase for silent neurons");
}

#[test]
fn scaling_weights_decrease_for_overactive_neurons() {
    let n_neurons = 5;
    let mut scaler = SynapticScaling::new(n_neurons, default_config());
    // Inject 100 Hz worth of spikes
    for nid in 0..n_neurons {
        for i in 0..20 { scaler.record_spike(nid, i as f32 * 10.0); }
    }
    let mut weights = make_weights(10, 5.0);
    let post = make_post(10);
    scaler.scale_weights(200.0, &post, &mut weights);
    assert!(weights.iter().any(|&w| w < 5.0), "Weights should decrease for overactive neurons");
}

#[test]
fn scaling_weights_clamped_to_w_max() {
    let mut cfg = default_config();
    cfg.w_max = 3.0;
    let n = 3;
    let mut scaler = SynapticScaling::new(n, cfg);
    // No spikes → tries to scale way up, but must clamp
    let mut weights = make_weights(6, 2.5);
    let post: Vec<usize> = vec![0, 1, 2, 0, 1, 2];
    scaler.scale_weights(200.0, &post, &mut weights);
    for &w in &weights {
        assert!(w <= 3.0, "Weight exceeded w_max: {}", w);
    }
}

#[test]
fn scaling_weights_clamped_to_w_min() {
    let mut cfg = default_config();
    cfg.w_min = 0.5;
    let n = 3;
    let mut scaler = SynapticScaling::new(n, cfg);
    // Fire at 1000 Hz → rate >> target → scales down aggressively
    for nid in 0..n {
        for i in 0..200 { scaler.record_spike(nid, i as f32); }
    }
    let mut weights = make_weights(6, 0.6);
    let post: Vec<usize> = vec![0, 1, 2, 0, 1, 2];
    scaler.scale_weights(200.0, &post, &mut weights);
    for &w in &weights {
        assert!(w >= 0.5, "Weight below w_min: {}", w);
    }
}

#[test]
fn scaling_disabled_produces_no_changes() {
    let mut cfg = default_config();
    cfg.enabled = false;
    let n = 4;
    let mut scaler = SynapticScaling::new(n, cfg);
    let mut weights = make_weights(8, 3.0);
    let post = make_post(8);
    scaler.scale_weights(500.0, &post, &mut weights);
    assert!(weights.iter().all(|&w| w == 3.0), "Disabled scaling should not change weights");
    assert_eq!(scaler.update_count, 0);
}

#[test]
fn scaling_respects_update_interval() {
    let mut cfg = default_config();
    cfg.update_interval_ms = 500.0;
    let mut scaler = SynapticScaling::new(3, cfg);
    let mut weights = make_weights(6, 2.0);
    let post: Vec<usize> = vec![0, 1, 2, 0, 1, 2];
    // Call at t=100 ms — should skip (< 500 ms interval)
    let changed = scaler.scale_weights(100.0, &post, &mut weights);
    assert_eq!(changed, 0, "Should skip before update interval");
}

#[test]
fn scaling_fraction_at_target_is_accurate() {
    let n = 6;
    let cfg = default_config(); // target = 5 Hz, window = 200 ms
    let mut scaler = SynapticScaling::new(n, cfg);
    // Neurons 0-3: fire at 5 Hz (target) → 1 spike per 200 ms
    for nid in 0..4 { scaler.record_spike(nid, 100.0); }
    // Neurons 4-5: fire at 50 Hz → 10 spikes per 200 ms
    for nid in 4..6 {
        for i in 0..10 { scaler.record_spike(nid, i as f32 * 20.0); }
    }
    let frac = scaler.fraction_at_target();
    // ~4/6 neurons near target
    assert!((frac - 0.667).abs() < 0.2, "Expected ~4/6 at target, got {:.3}", frac);
}

#[test]
fn scaling_weight_stats_are_correct() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let (min, mean, max) = SynapticScaling::weight_stats(&weights);
    assert!((min  - 1.0).abs() < 1e-5);
    assert!((mean - 2.5).abs() < 1e-5);
    assert!((max  - 4.0).abs() < 1e-5);
}

#[test]
fn scaling_weight_stats_empty_is_safe() {
    let (min, mean, max) = SynapticScaling::weight_stats(&[]);
    assert_eq!(min, 0.0); assert_eq!(mean, 0.0); assert_eq!(max, 0.0);
}

#[test]
fn scaling_preserves_relative_weight_ratios() {
    // If all neurons have same rate, scale factor is identical → ratios preserved
    let n = 4;
    let cfg = default_config();
    let mut scaler = SynapticScaling::new(n, cfg);
    // All neurons at 50 Hz (above target 5 Hz)
    for nid in 0..n {
        for i in 0..10 { scaler.record_spike(nid, i as f32 * 20.0); }
    }
    // Different initial weights
    let mut weights = vec![1.0f32, 2.0, 4.0, 8.0];
    let post: Vec<usize> = vec![0, 1, 2, 3];
    scaler.scale_weights(200.0, &post, &mut weights);
    // All should be scaled by same factor → ratios preserved
    let ratio_01 = weights[0] / weights[1];
    assert!((ratio_01 - 0.5).abs() < 0.01,
        "Ratio should be preserved (0.5): got {:.4}", ratio_01);
}