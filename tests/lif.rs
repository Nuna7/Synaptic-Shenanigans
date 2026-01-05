use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::lif::NeuronPopulation;

#[test]
fn lif_passive_decay() {
    let n = 1;
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);

    neurons.v[0].store(-40.0); // start above rest
    let input = vec![0.0];

    for _ in 0..100 {
        neurons.step_range(&input, 0);
    }

    let v = neurons.read_v(0);
    assert!(v < -40.0);
    assert!(v >= -65.0);
}


#[test]
fn lif_spike_and_reset() {
    let neurons = LifNeuron::new(1, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let input = vec![1000.0];

    neurons.step_range(&input, 0);

    assert!(neurons.local_spiked(0));
    assert_eq!(neurons.read_v(0), -65.0);
}


#[test]
fn lif_refractory_blocks_spikes() {
    let neurons = LifNeuron::new(1, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let input = vec![1000.0];

    neurons.step_range(&input, 0);
    assert!(neurons.local_spiked(0));

    for _ in 0..3 {
        neurons.step_range(&input, 0);
        assert!(!neurons.local_spiked(0));
    }
}
