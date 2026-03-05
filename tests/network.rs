#[test]
fn network_delays_are_non_negative_and_stable() {
    use synaptic_shenanigans::network::*;
    let syn = NetworkBuilder::erdos_renyi(
        100,
        0.05,
        EdgeParams::default(),
        42,
    );

    for &delay in syn.delay.iter() {
        assert!(delay >= 0.5, "illegal synaptic delay: {}", delay);
    }
}
