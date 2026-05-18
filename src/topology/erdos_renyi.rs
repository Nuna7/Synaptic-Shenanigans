use super::TopologyGenerator;
use crate::synapse::SynapseMatrix;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct ErdosRenyiParams {
    pub p: f64,
    pub allow_self_loops: bool,
    pub weight: f32,
    pub delay_ms: f32,
}
impl Default for ErdosRenyiParams {
    fn default() -> Self {
        Self {
            p: 0.1,
            allow_self_loops: false,
            weight: 1.0,
            delay_ms: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ErdosRenyi {
    pub params: ErdosRenyiParams,
}
impl ErdosRenyi {
    pub fn new(params: ErdosRenyiParams) -> Self {
        Self { params }
    }
}

impl TopologyGenerator for ErdosRenyi {
    fn generate(&self, n: usize, seed: u64) -> SynapseMatrix {
        let mut rng = SmallRng::seed_from_u64(seed);
        let (mut pre, mut post, mut weights, mut delays) = (vec![], vec![], vec![], vec![]);
        for i in 0..n {
            for j in 0..n {
                if !self.params.allow_self_loops && i == j {
                    continue;
                }
                if rng.r#gen::<f64>() < self.params.p {
                    pre.push(i as u32);
                    post.push(j as u32);
                    weights.push(self.params.weight);
                    delays.push(self.params.delay_ms);
                }
            }
        }
        SynapseMatrix {
            pre_neurons: pre,
            post_neurons: post,
            weights,
            delays,
        }
    }
}
