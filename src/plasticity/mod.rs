pub mod stdp;
pub mod homeostatic;
pub mod synaptic_scaling;

pub use stdp::{StdpState, StdpConfig, WeightStats};
pub use homeostatic::{HomeostaticState, HomeostaticConfig, ThresholdStats};
pub use synaptic_scaling::{SynapticScaling, SynapticScalingConfig};

pub struct PlasticityContext<'a> {
    pub new_spikes:  &'a [(f32, usize)],
    pub spike_log:   &'a [(f32, usize)],
    pub weights:     &'a mut Vec<f32>,
    pub pre_neurons: &'a [usize],
    pub post_neurons: &'a [usize],
    pub thresholds:  Option<&'a mut Vec<f32>>,
    pub current_time: f32,
    pub n_neurons:   usize,
}

pub trait PlasticityRule: Send + Sync {
    fn apply(&mut self, ctx: &mut PlasticityContext<'_>);
    fn is_enabled(&self) -> bool;
    fn set_enabled(&mut self, enabled: bool);
}