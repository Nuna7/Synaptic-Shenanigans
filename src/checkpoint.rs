use crate::simulation::Simulation;
use std::io;

pub struct Checkpoint;

impl Checkpoint {
    pub fn save(sim: &Simulation, path: &str) -> io::Result<()> {
        sim.save_state(path, &format!("{path}.sha256"))
    }

    pub fn load(path: &str) -> io::Result<Simulation> {
        let hash_path = format!("{path}.sha256");
        if std::path::Path::new(&hash_path).exists() {
            use sha2::{Digest, Sha256};
            let expected = std::fs::read_to_string(&hash_path).unwrap_or_default();
            let bytes = std::fs::read(path)?;
            let actual = format!("{:x}", Sha256::digest(&bytes));
            if !expected.trim().is_empty() && actual.trim() != expected.trim() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "SHA-256 mismatch: expected {}, got {}",
                        expected.trim(),
                        actual.trim()
                    ),
                ));
            }
        }
        Simulation::load_state(path, 42, 1)
    }
}
