use criterion::{criterion_group, criterion_main, Criterion};
use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::lif::NeuronPopulation;

fn bench_neuron_step(c: &mut Criterion) {
    let n = 10_000;
    let neurons = LifNeuron::new(
        n,
        -65.0, -50.0, 20.0,
        1.0, 1.0, 5.0,
    );

    let input = vec![1.0f32; n];

    c.bench_function("lif_step_10k", |b| {
        b.iter(|| {
            neurons.step_range(&input, 0);
        })
    });
}

criterion_group!(benches, bench_neuron_step);
criterion_main!(benches);
