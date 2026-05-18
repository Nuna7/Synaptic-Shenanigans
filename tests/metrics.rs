use rand::Rng;
use synaptic_shenanigans::metrics::{SynchronyIndex, BurstDetector, AvalancheResult,
                                    ISIStats, power_spectrum, dominant_frequency};

fn sync_spikes(n: usize, period: f32, dur: f32) -> Vec<(f32,usize)> {
    let mut s = Vec::new(); let mut t = period;
    while t < dur { for nid in 0..n { s.push((t, nid)); } t += period; } s
}
fn async_spikes(n: usize, rate: f32, dur: f32, seed: u64) -> Vec<(f32,usize)> {
    use rand::SeedableRng; use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let lam = rate / 1000.0;
    let mut s = Vec::new();
    for nid in 0..n { let mut t = 0.0f32;
        loop { let u: f32 = rng.r#gen(); t += -(1.0-u).ln()/lam; if t >= dur { break; } s.push((t,nid)); }
    } s
}

#[test]
fn synchrony_async_low_chi() {
    let sp = async_spikes(50, 20.0, 2000.0, 42);
    assert!(SynchronyIndex::compute(&sp, 50, 2000.0, 5.0).chi < 0.4);
}
#[test]
fn synchrony_sync_high_chi() {
    let sp = sync_spikes(50, 20.0, 2000.0);
    assert!(SynchronyIndex::compute(&sp, 50, 2000.0, 5.0).chi > 0.5);
}
#[test]
fn synchrony_empty_returns_zero() {
    assert_eq!(SynchronyIndex::compute(&[], 10, 1000.0, 5.0).chi, 0.0);
}
#[test]
fn burst_finds_bursts_in_sync_data() {
    let sp = sync_spikes(50, 200.0, 2000.0);
    assert!(!BurstDetector::new(50, 5.0, 5.0).detect(&sp, 2000.0).is_empty());
}
#[test]
fn burst_nothing_in_sparse_data() {
    assert!(BurstDetector::new(100, 20.0, 5.0).detect(&[(100.0, 0)], 2000.0).is_empty());
}
#[test]
fn dominant_frequency_detects_40hz() {
    let sp = sync_spikes(40, 25.0, 2000.0);
    let f = dominant_frequency(&sp, 40, 2000.0, 1.0, 30.0, 60.0);
    assert!(f.is_some());
    assert!((f.unwrap() - 40.0).abs() < 8.0, "f={:.1}", f.unwrap());
}
#[test]
fn avalanche_no_panic_on_empty() {
    let r = AvalancheResult::detect(&[], 1000.0, 1.0);
    assert!(r.sizes.is_empty());
}
#[test]
fn isi_regular_train_low_cv() {
    let sp: Vec<(f32,usize)> = (0..40).map(|i| (i as f32 * 50.0 + 50.0, 0)).collect();
    let s = ISIStats::compute(&sp, 1, 2000.0, 5.0);
    assert!(s.cv < 0.1, "CV={:.3}", s.cv);
}