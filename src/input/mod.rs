pub mod poisson;
pub use poisson::{PoissonSource, PoissonPopulation, StimulusPattern, drive_background};

pub trait StimulusSource {
    fn generate(&mut self, t_start: f32, t_end: f32) -> Vec<f32>;
    fn rate_hz(&self) -> f32;
}