use synaptic_shenanigans::{LifNeuron, Synapse, Simulation, SchedulerMode};
use synaptic_shenanigans::input::poisson::{PoissonSource, PoissonPopulation, StimulusPattern, drive_background};

const RATE: f32 = 20.0; const DUR: f32 = 10_000.0;

#[test]
fn poisson_mean_spike_count_correct() {
    let mut src = PoissonSource::new(RATE, 42);
    let n = src.generate(0.0, DUR).len() as f32;
    let exp = RATE * DUR / 1000.0;
    assert!((n/exp - 1.0).abs() < 0.15, "expected ~{exp} got {n}");
}
#[test]
fn poisson_cv_near_one() {
    let mut src = PoissonSource::new(RATE, 99);
    let spikes = src.generate(0.0, DUR);
    let mut s = spikes.clone(); s.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let isis: Vec<f32> = s.windows(2).map(|w| w[1]-w[0]).collect();
    let mean = isis.iter().sum::<f32>() / isis.len() as f32;
    let var  = isis.iter().map(|&v|(v-mean).powi(2)).sum::<f32>() / isis.len() as f32;
    let cv = var.sqrt() / mean;
    assert!((cv - 1.0).abs() < 0.25, "CV={cv:.3}");
}
#[test]
fn poisson_reproducible_same_seed() {
    let spikes_a = PoissonSource::new(RATE, 42).generate(0.0, 500.0);
    let spikes_b = PoissonSource::new(RATE, 42).generate(0.0, 500.0);
    assert_eq!(spikes_a, spikes_b);
}
#[test]
fn poisson_zero_rate_no_spikes() {
    assert!(PoissonSource::new(0.0, 42).generate(0.0, 1000.0).is_empty());
}
#[test]
fn poisson_population_injects_into_simulation() {
    let n = 10;
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_neurons(neurons, Synapse::new(), 1.0, 42, 1);
    sim.scheduler_mode = SchedulerMode::SingleThreaded;
    let n_ev = PoissonPopulation::new(n, RATE, 500.0, 42).prebuild(&mut sim, 500.0);
    assert!(n_ev > 0);
    sim.run_auto(500.0);
    assert!(!sim.spike_log.is_empty());
}
#[test]
fn drive_background_helper_works() {
    let n = 5;
    let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Simulation::new_with_neurons(neurons, Synapse::new(), 1.0, 42, 1);
    assert!(drive_background(&mut sim, n, 20.0, 60.0, 42, 1000.0) > 0);
}
#[test]
fn stimulus_step_respects_on_off() {
    let mut p = StimulusPattern::step(0.0, 100.0, 200.0, 400.0, 42);
    let sp = p.generate(0.0, 600.0);
    for &t in &sp { assert!(t >= 200.0 && t < 400.0, "t={t} outside window"); }
    assert!(!sp.is_empty());
}