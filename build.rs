fn main() {
    println!("cargo:rerun-if-changed=rpc/neurosim.proto");

    tonic_build::configure()
        .compile(&["rpc/neurosim.proto"], &["rpc"])
        .unwrap();
}
