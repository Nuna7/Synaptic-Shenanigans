use super::TopologyGenerator;
use crate::synapse::SynapseMatrix;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct BarabasiAlbertParams {
    pub m: usize,
    pub weight: f32,
    pub delay_ms: f32,
}
impl Default for BarabasiAlbertParams {
    fn default() -> Self {
        Self {
            m: 2,
            weight: 1.0,
            delay_ms: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BarabasiAlbert {
    pub params: BarabasiAlbertParams,
}
impl BarabasiAlbert {
    pub fn new(params: BarabasiAlbertParams) -> Self {
        Self { params }
    }
}

impl TopologyGenerator for BarabasiAlbert {
    fn generate(&self, n: usize, seed: u64) -> SynapseMatrix {
        let m = self.params.m.min(n.saturating_sub(1));
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut degree = vec![0usize; n];
        let mut pre = Vec::new();
        let mut post = Vec::new();
        for i in 0..m.min(n) {
            for j in (i + 1)..m.min(n) {
                pre.push(i as u32);
                post.push(j as u32);
                pre.push(j as u32);
                post.push(i as u32);
                degree[i] += 1;
                degree[j] += 1;
            }
        }
        for new_node in m..n {
            let total: usize = degree.iter().take(new_node).sum();
            let mut chosen = std::collections::HashSet::new();
            while chosen.len() < m {
                let r = rng.gen_range(0..total.max(1));
                let mut cum = 0;
                for (ex, &d) in degree.iter().enumerate().take(new_node) {
                    cum += d;
                    if cum > r {
                        chosen.insert(ex);
                        break;
                    }
                }
                if chosen.len() < m {
                    chosen.insert(rng.gen_range(0..new_node));
                }
            }
            for ex in chosen {
                pre.push(new_node as u32);
                post.push(ex as u32);
                degree[new_node] += 1;
                degree[ex] += 1;
            }
        }
        let n_e = pre.len();
        SynapseMatrix {
            pre_neurons: pre,
            post_neurons: post,
            weights: vec![self.params.weight; n_e],
            delays: vec![self.params.delay_ms; n_e],
        }
    }
}
