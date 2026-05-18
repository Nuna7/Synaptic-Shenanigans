//! gRPC server binary.
//!
//! Run: cargo run --release --features rpc --bin rpc_server
//!
//! Listens on 127.0.0.1:50051 by default.
//! Override with env var: NEUROSIM_ADDR=0.0.0.0:50051

#[cfg(feature = "rpc")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use tonic::transport::Server;
    use synaptic_shenanigans::rpc::{RpcService, SimStore};
    use synaptic_shenanigans::rpc::pb::neuro_sim_server::NeuroSimServer;

    let addr = std::env::var("NEUROSIM_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:50051".to_string())
        .parse()?;

    let store   = Arc::new(SimStore::new());
    let service = RpcService::new(store);

    println!("Synaptic-Shenanigans gRPC server listening on {}", addr);

    Server::builder()
        .add_service(NeuroSimServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

#[cfg(not(feature = "rpc"))]
fn main() {
    eprintln!("ERROR: This binary requires the `rpc` feature.");
    eprintln!("Build with:  cargo run --release --features rpc --bin rpc_server");
    std::process::exit(1);
}