mod utils;

use crate::utils::build_sim;

#[test]
fn determinism_many_seeds() {
    for seed in 0..50 {
        let a = build_sim(seed);
        let b = build_sim(seed);
        assert_eq!(a.spike_log, b.spike_log);
    }
}

