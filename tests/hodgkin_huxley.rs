use synaptic_shenanigans::neurons::{HHPopulation, HHParams, NeuronPopulation};
use synaptic_shenanigans::neurons::hh::steady_state;

fn run_single(i_ext: f32, ms: usize) -> (Vec<bool>, Vec<f32>) {
    let hh = HHPopulation::homogeneous(1, HHParams::default());
    let mut spiked = Vec::new(); let mut vs = Vec::new();
    for _ in 0..ms { hh.step_range(&[i_ext], 0); spiked.push(hh.local_spiked(0)); vs.push(hh.read_v(0) as f32); }
    (spiked, vs)
}
fn count_spikes(i_ext: f32, ms: usize) -> usize {
    run_single(i_ext, ms).0.into_iter().filter(|&s| s).count()
}

#[test]
fn hh_steady_state_values_are_biologically_correct() {
    let (m0,h0,n0) = steady_state(-65.0);
    assert!(m0 < 0.1); assert!(h0 > 0.5); assert!(n0 < 0.4);
    assert!((0.0..=1.0).contains(&m0)); assert!((0.0..=1.0).contains(&h0)); assert!((0.0..=1.0).contains(&n0));
}
#[test] fn hh_no_spike_below_rheobase() { assert_eq!(count_spikes(2.0, 500), 0); }
#[test] fn hh_fires_above_rheobase()    { assert!(count_spikes(10.0, 500) > 5); }
#[test]
fn hh_membrane_potential_stays_in_range() {
    let (_, vs) = run_single(10.0, 500);
    for v in &vs { assert!(*v >= -90.0 && *v <= 60.0); }
}
#[test]
fn hh_fi_curve_is_monotonically_increasing() {
    let rates: Vec<usize> = [6.0f32,8.0,10.0,15.0,20.0].iter().map(|&i| count_spikes(i, 1000)).collect();
    for w in rates.windows(2) { assert!(w[1] >= w[0]); }
}
#[test]
fn hh_deterministic_across_runs() {
    let h1 = HHPopulation::homogeneous(1, HHParams::default());
    let h2 = HHPopulation::homogeneous(1, HHParams::default());
    for _ in 0..500 {
        h1.step_range(&[10.0],0); h2.step_range(&[10.0],0);
        assert_eq!(h1.local_spiked(0), h2.local_spiked(0));
        assert!((h1.read_v(0) - h2.read_v(0)).abs() < 1e-4);
    }
}
#[test]
fn hh_reset_neuron() {
    let hh = HHPopulation::homogeneous(1, HHParams::default());
    for _ in 0..200 { hh.step_range(&[10.0], 0); }
    hh.reset_neuron(0, -65.0);
    assert!((hh.read_v(0) - (-65.0)).abs() < 1.0);
    assert!(!hh.local_spiked(0));
}