#[test]
fn network_delays_are_non_negative() {
    use synaptic_shenanigans::network::{NetworkBuilder, EdgeParams};
    let syn = NetworkBuilder::erdos_renyi(100, 0.05, EdgeParams::default(), 42);
    for &d in &syn.delay { assert!(d >= 0.5, "illegal delay: {}", d); }
}