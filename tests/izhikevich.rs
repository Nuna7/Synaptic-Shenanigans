#[test]
fn izhikevich_deterministic_single_neuron() {
    use synaptic_shenanigans::izhikevich::*;
    use synaptic_shenanigans::lif::NeuronPopulation;

    let pop1 = IzhikevichPop::homogeneous(1, NeuronType::RegularSpiking, 1.0);
    let pop2 = IzhikevichPop::homogeneous(1, NeuronType::RegularSpiking, 1.0);

    let input = vec![10.0];

    for _ in 0..100 {
        pop1.step_range(&input, 0);
        pop2.step_range(&input, 0);
    }

    assert_eq!(pop1.read_v(0), pop2.read_v(0));
    assert_eq!(pop1.local_spiked(0), pop2.local_spiked(0));
}
