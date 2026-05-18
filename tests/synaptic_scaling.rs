use synaptic_shenanigans::plasticity::{SynapticScaling, SynapticScalingConfig};

fn cfg() -> SynapticScalingConfig {
    SynapticScalingConfig { target_rate_hz: 5.0, alpha: 1.0, w_min: 0.0, w_max: 20.0,
                            update_interval_ms: 10.0, rate_window_ms: 200.0, enabled: true }
}

#[test]
fn scaling_weights_increase_for_silent_neurons() {
    let mut s = SynapticScaling::new(5, cfg());
    let mut w = vec![2.0f32; 10]; let post: Vec<usize> = (0..10).map(|i| i%5).collect();
    s.scale_weights(500.0, &post, &mut w);
    assert!(w.iter().any(|&x| x > 2.0));
}
#[test]
fn scaling_weights_decrease_for_overactive() {
    let mut s = SynapticScaling::new(5, cfg());
    for nid in 0..5 { for i in 0..20 { s.record_spike(nid, i as f32 * 10.0); } }
    let mut w = vec![5.0f32; 10]; let post: Vec<usize> = (0..10).map(|i| i%5).collect();
    s.scale_weights(200.0, &post, &mut w);
    assert!(w.iter().any(|&x| x < 5.0));
}
#[test]
fn scaling_disabled_no_changes() {
    let mut cfg = cfg(); cfg.enabled = false;
    let mut s = SynapticScaling::new(3, cfg);
    let mut w = vec![3.0f32; 6]; let post = vec![0,1,2,0,1,2];
    s.scale_weights(500.0, &post, &mut w);
    assert!(w.iter().all(|&x| x == 3.0));
}