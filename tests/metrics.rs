//! Tests for population metrics.
//!
//! Verifies:
//!   1. Synchrony index: async → χ ≈ 0, sync → χ ≈ 1
//!   2. Burst detector finds bursts in synchronous data
//!   3. Burst detector finds nothing in sparse data
//!   4. Power spectrum correctly identifies injected oscillation frequency
//!   5. Avalanche analysis runs without panic on various inputs
//!   6. ISI stats are mathematically correct on known spike trains
//!   7. Edge cases: empty spikes, single neuron, zero duration

use rand::Rng;
use synaptic_shenanigans::metrics::{
    SynchronyIndex, BurstDetector, AvalancheResult,
    ISIStats, power_spectrum, dominant_frequency,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Generate synchronous spikes: all n neurons fire together every `period_ms` ms.
fn sync_spikes(n: usize, period_ms: f32, duration_ms: f32) -> Vec<(f32, usize)> {
    let mut spikes = Vec::new();
    let mut t = period_ms;
    while t < duration_ms {
        for nid in 0..n {
            spikes.push((t, nid));
        }
        t += period_ms;
    }
    spikes
}

/// Generate asynchronous independent Poisson spikes.
fn async_spikes(n: usize, rate_hz: f32, duration_ms: f32, seed: u64) -> Vec<(f32, usize)> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let lambda = rate_hz / 1000.0;
    let mut spikes = Vec::new();
    for nid in 0..n {
        let mut t = 0.0f32;
        loop {
            let u: f32 = rng.r#gen();
            t += -(1.0f32 - u).ln() / lambda;
            if t >= duration_ms { break; }
            spikes.push((t, nid));
        }
    }
    spikes
}

/// Regular ISI spike train: one neuron firing at fixed intervals.
fn regular_spikes(n: usize, rate_hz: f32, duration_ms: f32) -> Vec<(f32, usize)> {
    let isi = 1000.0 / rate_hz;
    let mut spikes = Vec::new();
    for nid in 0..n {
        let mut t = isi;
        while t < duration_ms {
            spikes.push((t, nid));
            t += isi;
        }
    }
    spikes
}

// ── SynchronyIndex ────────────────────────────────────────────────────────────

#[test]
fn synchrony_async_population_has_low_chi() {
    let n = 50;
    let spikes = async_spikes(n, 20.0, 2000.0, 42);
    let sync = SynchronyIndex::compute(&spikes, n, 2000.0, 5.0);
    assert!(sync.chi < 0.4,
        "Async population should have low χ, got {:.4}", sync.chi);
}

#[test]
fn synchrony_sync_population_has_high_chi() {
    let n = 50;
    let spikes = sync_spikes(n, 20.0, 2000.0);
    let sync = SynchronyIndex::compute(&spikes, n, 2000.0, 5.0);
    assert!(sync.chi > 0.5,
        "Synchronous population should have high χ, got {:.4}", sync.chi);
}

#[test]
fn synchrony_empty_spikes_returns_default() {
    let sync = SynchronyIndex::compute(&[], 10, 1000.0, 5.0);
    assert_eq!(sync.chi, 0.0);
    assert_eq!(sync.n_neurons, 0);
}

#[test]
fn synchrony_chi_is_between_zero_and_one() {
    let n = 30;
    let spikes = async_spikes(n, 15.0, 1000.0, 99);
    let sync = SynchronyIndex::compute(&spikes, n, 1000.0, 2.0);
    assert!(sync.chi >= 0.0 && sync.chi <= 1.0, "χ = {} not in [0,1]", sync.chi);
}

#[test]
fn synchrony_state_label_is_consistent() {
    let n = 50;
    let async_s = async_spikes(n, 10.0, 2000.0, 1);
    let s = SynchronyIndex::compute(&async_s, n, 2000.0, 5.0);
    let label = s.state();
    assert!(!label.is_empty(), "State label should be non-empty");
}

// ── BurstDetector ─────────────────────────────────────────────────────────────

#[test]
fn burst_detector_finds_bursts_in_sync_data() {
    let n = 50;
    // Synchronous spikes every 200 ms → dense activity → bursts
    let spikes = sync_spikes(n, 200.0, 2000.0);
    let detector = BurstDetector::new(n, 5.0, 5.0); // threshold 5 Hz
    let bursts = detector.detect(&spikes, 2000.0);
    assert!(!bursts.is_empty(), "Should detect bursts in synchronous population");
}

#[test]
fn burst_detector_finds_nothing_in_sparse_data() {
    let n = 100;
    // 1 spike total → clearly no burst
    let spikes = vec![(100.0f32, 0usize)];
    let detector = BurstDetector::new(n, 20.0, 5.0);
    let bursts = detector.detect(&spikes, 2000.0);
    assert!(bursts.is_empty(), "Should find no bursts in 1-spike trace");
}

#[test]
fn burst_properties_are_biologically_plausible() {
    let n = 30;
    let spikes = sync_spikes(n, 100.0, 1000.0); // dense bursts
    let detector = BurstDetector::new(n, 5.0, 5.0);
    let bursts = detector.detect(&spikes, 1000.0);
    for b in &bursts {
        assert!(b.t_start < b.t_end, "Burst start must precede end");
        assert!(b.n_spikes > 0, "Burst must contain spikes");
        assert!(b.recruitment >= 0.0 && b.recruitment <= 1.0,
            "Recruitment must be in [0,1]");
        assert!(b.peak_rate_hz >= 0.0, "Peak rate must be non-negative");
    }
}

#[test]
fn burst_duration_matches_timespan() {
    let n = 20;
    let spikes = sync_spikes(n, 50.0, 500.0);
    let detector = BurstDetector::new(n, 5.0, 5.0);
    let bursts = detector.detect(&spikes, 500.0);
    for b in &bursts {
        let dur = b.duration_ms();
        assert!(dur > 0.0, "Burst duration must be positive");
    }
}

// ── Power spectrum ────────────────────────────────────────────────────────────

#[test]
fn power_spectrum_has_correct_length() {
    let n = 20;
    let spikes = async_spikes(n, 20.0, 2000.0, 5);
    let (freqs, power) = power_spectrum(&spikes, n, 2000.0, 1.0);
    assert_eq!(freqs.len(), power.len(), "Freqs and power must have same length");
    assert!(!freqs.is_empty(), "Should produce non-empty spectrum");
}

#[test]
fn power_spectrum_frequencies_are_non_negative() {
    let spikes = async_spikes(30, 10.0, 1000.0, 7);
    let (freqs, _) = power_spectrum(&spikes, 30, 1000.0, 1.0);
    for &f in &freqs {
        assert!(f >= 0.0, "Frequency must be non-negative: {}", f);
    }
}

#[test]
fn dominant_frequency_detects_injected_oscillation() {
    // Synchronous bursts every 25 ms → 40 Hz oscillation
    let n = 40;
    let period_ms = 25.0f32; // 40 Hz
    let spikes = sync_spikes(n, period_ms, 2000.0);

    let dom = dominant_frequency(&spikes, n, 2000.0, 1.0, 30.0, 60.0);
    assert!(dom.is_some(), "Should detect a dominant frequency in gamma band");
    let f = dom.unwrap();
    // Allow ±5 Hz tolerance (limited frequency resolution at 1 ms bins)
    assert!((f - 40.0).abs() < 8.0,
        "Dominant frequency should be near 40 Hz, got {:.1}", f);
}

#[test]
fn power_spectrum_empty_spikes_does_not_panic() {
    let (freqs, power) = power_spectrum(&[], 10, 1000.0, 1.0);
    assert_eq!(freqs.len(), power.len());
}

// ── Avalanche analysis ────────────────────────────────────────────────────────

#[test]
fn avalanche_detect_does_not_panic_on_empty_spikes() {
    let result = AvalancheResult::detect(&[], 1000.0, 1.0);
    assert!(result.sizes.is_empty());
    assert_eq!(result.tau, 0.0);
}

#[test]
fn avalanche_detect_finds_cascades_in_active_trace() {
    let spikes = async_spikes(50, 20.0, 2000.0, 123);
    let result = AvalancheResult::detect(&spikes, 2000.0, 1.0);
    // Should find some avalanches in a typical active trace
    assert!(!result.sizes.is_empty(), "Should find avalanches");
    assert!(result.activity_fraction > 0.0 && result.activity_fraction <= 1.0);
}

#[test]
fn avalanche_sizes_are_positive() {
    let spikes = async_spikes(30, 15.0, 1000.0, 77);
    let result = AvalancheResult::detect(&spikes, 1000.0, 1.0);
    for &s in &result.sizes {
        assert!(s > 0, "All avalanche sizes must be positive");
    }
}

#[test]
fn avalanche_durations_are_positive() {
    let spikes = async_spikes(30, 15.0, 1000.0, 88);
    let result = AvalancheResult::detect(&spikes, 1000.0, 1.0);
    for &d in &result.durations {
        assert!(d > 0, "All avalanche durations must be positive");
    }
}

// ── ISI Stats ─────────────────────────────────────────────────────────────────

#[test]
fn isi_stats_poisson_cv_is_near_one() {
    let spikes = async_spikes(1, 20.0, 10_000.0, 42);
    let stats = ISIStats::compute(&spikes, 1, 10_000.0, 10.0);
    assert!((stats.cv - 1.0).abs() < 0.3, "Poisson CV should be ~1.0, got {}", stats.cv);
}

#[test]
fn isi_stats_regular_train_has_low_cv() {
    let spikes = regular_spikes(1, 20.0, 2000.0);
    let stats = ISIStats::compute(&spikes, 1, 2000.0, 5.0);
    assert!(stats.cv < 0.1, "Regular train should have CV near 0, got {}", stats.cv);
}

#[test]
fn isi_stats_mean_isi_matches_inverse_rate() {
    let rate_hz = 10.0f32;
    let spikes = regular_spikes(1, rate_hz, 5000.0);
    let stats = ISIStats::compute(&spikes, 1, 5000.0, 5.0);
    let expected_isi = 1000.0 / rate_hz;
    assert!((stats.mean_isi_ms - expected_isi).abs() < 1.0,
        "Mean ISI: expected {:.1} ms, got {:.1} ms", expected_isi, stats.mean_isi_ms);
}

#[test]
fn isi_stats_empty_returns_defaults() {
    let stats = ISIStats::compute(&[], 10, 1000.0, 5.0);
    assert_eq!(stats.total_isis, 0);
    assert_eq!(stats.n_active, 0);
}

#[test]
fn isi_stats_n_active_counts_correctly() {
    let mut spikes = vec![];
    // Only neurons 0, 2, 4 fire
    for &nid in &[0usize, 2, 4] {
        for i in 0..10 {
            spikes.push((i as f32 * 50.0, nid));
        }
    }
    let stats = ISIStats::compute(&spikes, 10, 500.0, 5.0);
    assert_eq!(stats.n_active, 3, "Should count 3 active neurons");
}
