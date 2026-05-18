#[test]
fn izhikevich_deterministic_single_neuron() {
    use synaptic_shenanigans::neurons::{IzhikevichPop, NeuronPopulation, NeuronType};
    let p1 = IzhikevichPop::homogeneous(1, NeuronType::RegularSpiking, 1.0);
    let p2 = IzhikevichPop::homogeneous(1, NeuronType::RegularSpiking, 1.0);
    for _ in 0..100 {
        p1.step_range(&[10.0], 0);
        p2.step_range(&[10.0], 0);
    }
    assert_eq!(p1.read_v(0), p2.read_v(0));
    assert_eq!(p1.local_spiked(0), p2.local_spiked(0));
}

#[test]
fn izhikevich_all_types_fire_at_high_current() {
    use synaptic_shenanigans::neurons::{IzhikevichPop, NeuronPopulation, NeuronType};
    for nt in [
        NeuronType::RegularSpiking,
        NeuronType::FastSpiking,
        NeuronType::Chattering,
    ] {
        let pop = IzhikevichPop::homogeneous(1, nt, 1.0);
        let mut spiked = false;
        for _ in 0..500 {
            pop.step_range(&[15.0], 0);
            if pop.local_spiked(0) {
                spiked = true;
            }
        }
        assert!(spiked, "{:?} should fire at high current", nt);
    }
}

#[test]
fn izhikevich_reset_neuron() {
    use synaptic_shenanigans::neurons::{IzhikevichPop, NeuronPopulation, NeuronType};
    let pop = IzhikevichPop::homogeneous(1, NeuronType::RegularSpiking, 1.0);
    for _ in 0..100 {
        pop.step_range(&[15.0], 0);
    }
    pop.reset_neuron(0, -65.0);
    assert!((pop.read_v(0) - (-65.0)).abs() < 1e-4);
    assert!(!pop.local_spiked(0));
}
