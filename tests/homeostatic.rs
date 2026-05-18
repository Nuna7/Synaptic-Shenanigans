use synaptic_shenanigans::plasticity::{HomeostaticConfig, HomeostaticState};

fn fast_cfg(target: f32) -> HomeostaticConfig {
    HomeostaticConfig {
        target_rate_hz: target,
        tau_h: 500.0,
        theta_min: -75.0,
        theta_max: -40.0,
        rate_window_ms: 200.0,
        update_interval_ms: 10.0,
        enabled: true,
    }
}

#[test]
fn homeo_threshold_rises_when_firing_too_fast() {
    let mut h = HomeostaticState::new(1, -50.0, fast_cfg(5.0));
    for i in 0..50 {
        h.record_spike(0, i as f32 * 4.0);
    }
    h.update(200.0);
    assert!(h.theta[0] > -50.0, "theta should rise: {}", h.theta[0]);
}
#[test]
fn homeo_threshold_falls_when_firing_too_slow() {
    let mut h = HomeostaticState::new(1, -50.0, fast_cfg(20.0));
    h.record_spike(0, 100.0);
    h.update(200.0);
    assert!(h.theta[0] < -50.0, "theta should fall: {}", h.theta[0]);
}
#[test]
fn homeo_disabled_produces_no_changes() {
    let mut cfg = fast_cfg(5.0);
    cfg.enabled = false;
    let mut h = HomeostaticState::new(3, -50.0, cfg);
    for nid in 0..3 {
        for i in 0..100 {
            h.record_spike(nid, i as f32 * 2.0);
        }
    }
    h.update(500.0);
    for nid in 0..3 {
        assert_eq!(h.theta[nid], -50.0);
    }
}
#[test]
fn homeo_apply_thresholds_uses_trait() {
    use synaptic_shenanigans::{LifNeuron, Simulation, Synapse};
    let neurons = LifNeuron::new(3, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let sim = Simulation::new_with_neurons(neurons, Synapse::new(), 1.0, 0, 1);
    let mut h = HomeostaticState::new(3, -50.0, fast_cfg(5.0));
    for i in 0..3 {
        h.theta[i] = -48.0 - i as f32;
    }
    h.apply_thresholds(sim.neurons.as_ref());
    for i in 0..3 {
        assert!((sim.neurons.get_threshold(i) - (-48.0 - i as f32)).abs() < 1e-4);
    }
}
