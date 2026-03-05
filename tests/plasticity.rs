#[test]
fn stdp_produces_weight_change() {
    use synaptic_shenanigans::plasticity::*;

    let mut stdp = StdpState::new(2, 1, StdpConfig::default());
    let mut weights = vec![1.0];

    let pre = vec![0];
    let post = vec![1];
    let pre_index = vec![vec![0], vec![]];

    // Pre fires before post
    stdp.accumulate_for_spike(0, 1.0, &pre, &post, &pre_index);
    stdp.accumulate_for_spike(1, 2.0, &pre, &post, &pre_index);

    let changed = stdp.flush_weight_updates(&mut weights);

    assert!(changed > 0);
    assert!(weights[0] != 1.0);
}
