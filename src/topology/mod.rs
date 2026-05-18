pub mod erdos_renyi;
pub mod watts_strogatz;
pub mod barabasi_albert;
pub mod layered;

pub use erdos_renyi::{ErdosRenyi, ErdosRenyiParams};
pub use watts_strogatz::{WattsStrogatz, WattsStrogatzParams};
pub use barabasi_albert::{BarabasiAlbert, BarabasiAlbertParams};
pub use layered::{Layered, LayeredParams};

use crate::synapse::SynapseMatrix;

pub trait TopologyGenerator {
    fn generate(&self, n_neurons: usize, seed: u64) -> SynapseMatrix;
}

pub fn build(name: &str, n_neurons: usize, seed: u64) -> SynapseMatrix {
    match name {
        "erdos_renyi"     => ErdosRenyi::default().generate(n_neurons, seed),
        "watts_strogatz"  => WattsStrogatz::default().generate(n_neurons, seed),
        "barabasi_albert" => BarabasiAlbert::default().generate(n_neurons, seed),
        "layered"         => Layered::default().generate(n_neurons, seed),
        other => panic!("Unknown topology: {other}"),
    }
}