use super::TopologyGenerator;
use crate::synapse::SynapseMatrix;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct WattsStrogatzParams {
    pub k: usize,
    pub beta: f64,
    pub weight: f32,
    pub delay_ms: f32,
}
impl Default for WattsStrogatzParams {
    fn default() -> Self {
        Self {
            k: 4,
            beta: 0.1,
            weight: 1.0,
            delay_ms: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WattsStrogatz {
    pub params: WattsStrogatzParams,
}
impl WattsStrogatz {
    pub fn new(params: WattsStrogatzParams) -> Self {
        Self { params }
    }
}

impl TopologyGenerator for WattsStrogatz {
    fn generate(&self, n: usize, seed: u64) -> SynapseMatrix {
        let mut rng = SmallRng::seed_from_u64(seed);
        let k = self.params.k;
        let mut edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (1..=(k / 2)).map(move |j| (i, (i + j) % n)))
            .collect();
        for e in &mut edges {
            if rng.r#gen::<f64>() < self.params.beta {
                e.1 = rng.gen_range(0..n);
            }
        }
        let edges: Vec<_> = edges.into_iter().filter(|&(a, b)| a != b).collect();
        let n_e = edges.len();
        SynapseMatrix {
            pre_neurons: edges.iter().map(|&(a, _)| a as u32).collect(),
            post_neurons: edges.iter().map(|&(_, b)| b as u32).collect(),
            weights: vec![self.params.weight; n_e],
            delays: vec![self.params.delay_ms; n_e],
        }
    }
}
