use crate::synapse::SynapseMatrix;
use super::TopologyGenerator;

#[derive(Clone, Debug)]
pub struct LayeredParams { pub layer_sizes: Vec<usize>, pub weight: f32, pub delay_ms: f32 }
impl Default for LayeredParams {
    fn default() -> Self { Self { layer_sizes: vec![10,10,10], weight: 1.0, delay_ms: 1.0 } }
}

#[derive(Clone, Debug, Default)]
pub struct Layered { pub params: LayeredParams }
impl Layered { pub fn new(params: LayeredParams) -> Self { Self { params } } }

impl TopologyGenerator for Layered {
    fn generate(&self, _n: usize, _seed: u64) -> SynapseMatrix {
        let mut offsets = vec![0usize];
        for &sz in &self.params.layer_sizes { offsets.push(offsets.last().unwrap() + sz); }
        let mut pre = Vec::new(); let mut post = Vec::new();
        for l in 0..(self.params.layer_sizes.len().saturating_sub(1)) {
            for i in offsets[l]..offsets[l+1] { for j in offsets[l+1]..offsets[l+2] {
                pre.push(i as u32); post.push(j as u32);
            }}
        }
        let n_e = pre.len();
        SynapseMatrix { pre_neurons: pre, post_neurons: post, weights: vec![self.params.weight; n_e], delays: vec![self.params.delay_ms; n_e] }
    }
}