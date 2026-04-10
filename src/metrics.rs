//! Population-level neural metrics.
//!
//! These metrics characterise collective dynamics that emerge from the
//! activity of many neurons — phenomena that are invisible at the single-cell
//! level but are of central importance in neuroscience.
//!
//! ## Provided metrics
//!
//! | Metric | What it measures |
//! |---|---|
//! | `SynchronyIndex` | How correlated is population firing? (0 = async, 1 = lockstep) |
//! | `BurstDetector` | Detect and characterise multi-neuron bursts |
//! | `PowerSpectrum` | Oscillation frequencies in the population firing rate |
//! | `AvalancheAnalysis` | Scale-free activity cascades (criticality) |
//! | `ISIStats` | Population-level ISI statistics (CV, Fano factor) |
//!
//! References:
//!   Golomb & Hansel (2000). Neural Comput. — synchrony index.
//!   Plenz & Thiagarajan (2007). Trends Neurosci. — neural avalanches.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Synchrony index ──────────────────────────────────────────────────────────

/// Population synchrony (also called "chi" or "population synchrony measure").
///
/// Based on van Vreeswijk (1996):
///   χ = Var(population_firing_rate) / mean(Var(individual_rates))
///
/// χ ≈ 0 → asynchronous irregular (AI state — typical in healthy cortex)
/// χ ≈ 1 → fully synchronous (as in epilepsy or deep sleep)
/// χ ∈ (0.1, 0.3) → weakly synchronised (normal active cortex)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SynchronyIndex {
    pub chi:         f32,
    pub pop_var:     f32,
    pub mean_indiv:  f32,
    pub n_neurons:   usize,
    pub n_bins:      usize,
}

impl SynchronyIndex {
    /// Compute synchrony from a spike log.
    ///
    /// `bin_ms` controls the time binning for rate estimation.
    /// Values of 1–5 ms are typical.
    pub fn compute(spikes: &[(f32, usize)], n_neurons: usize, duration_ms: f32, bin_ms: f32) -> Self {
        if spikes.is_empty() || n_neurons == 0 { return Self::default(); }

        let n_bins = (duration_ms / bin_ms).ceil() as usize;

        // Population-level PSTH: total spikes per bin
        let mut pop_counts = vec![0.0f64; n_bins];
        // Per-neuron counts: neuron_id → counts per bin
        let mut per_neuron: HashMap<usize, Vec<f64>> = HashMap::new();

        for &(t, nid) in spikes {
            let bin = ((t / bin_ms) as usize).min(n_bins - 1);
            pop_counts[bin] += 1.0;
            per_neuron.entry(nid).or_insert_with(|| vec![0.0; n_bins])[bin] += 1.0;
        }

        // Population rate: normalise by n_neurons
        let pop_rate: Vec<f64> = pop_counts.iter().map(|&c| c / n_neurons as f64).collect();

        // Variance of population rate
        let pop_mean = pop_rate.iter().sum::<f64>() / pop_rate.len() as f64;
        let pop_var  = pop_rate.iter().map(|&r| (r - pop_mean).powi(2)).sum::<f64>()
                       / pop_rate.len() as f64;

        // Mean of individual neuron variances
        let mut sum_indiv_var = 0.0f64;
        let mut n_active = 0usize;
        for counts in per_neuron.values() {
            let m = counts.iter().sum::<f64>() / counts.len() as f64;
            let v = counts.iter().map(|&c| (c - m).powi(2)).sum::<f64>() / counts.len() as f64;
            sum_indiv_var += v;
            n_active += 1;
        }
        // Silent neurons contribute zero variance
        let mean_indiv = if n_active > 0 { sum_indiv_var / n_neurons as f64 } else { 1e-10 };

        let chi = if mean_indiv > 1e-10 { (pop_var / mean_indiv).sqrt() as f32 } else { 0.0 };

        Self {
            chi: chi.clamp(0.0, 1.0),
            pop_var: pop_var as f32,
            mean_indiv: mean_indiv as f32,
            n_neurons,
            n_bins,
        }
    }

    pub fn state(&self) -> &'static str {
        if self.chi < 0.05  { "Asynchronous Irregular (AI)" }
        else if self.chi < 0.2  { "Weakly Synchronous" }
        else if self.chi < 0.5  { "Moderately Synchronous" }
        else                     { "Strongly Synchronous" }
    }
}

impl std::fmt::Display for SynchronyIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "χ = {:.4}  [{}]  (pop_var={:.4}, mean_indiv={:.4})",
            self.chi, self.state(), self.pop_var, self.mean_indiv)
    }
}

// ── Burst detection ──────────────────────────────────────────────────────────

/// A detected population burst.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Burst {
    /// Burst onset (ms).
    pub t_start: f32,
    /// Burst offset (ms).
    pub t_end:   f32,
    /// Peak firing rate within the burst (Hz, population-level).
    pub peak_rate_hz: f32,
    /// Number of spikes in the burst.
    pub n_spikes: usize,
    /// Number of distinct neurons that participated.
    pub n_neurons: usize,
    /// Recruitment fraction (n_neurons / total_population).
    pub recruitment: f32,
}

impl Burst {
    pub fn duration_ms(&self) -> f32 { self.t_end - self.t_start }
}

/// Detect population bursts using a sliding-window threshold.
pub struct BurstDetector {
    pub bin_ms:        f32,
    pub threshold_hz:  f32, // rate threshold for burst onset
    pub min_duration_ms: f32,
    pub n_neurons:     usize,
}

impl BurstDetector {
    pub fn new(n_neurons: usize, threshold_hz: f32, bin_ms: f32) -> Self {
        Self { bin_ms, threshold_hz, min_duration_ms: bin_ms, n_neurons }
    }

    pub fn detect(&self, spikes: &[(f32, usize)], duration_ms: f32) -> Vec<Burst> {
        let n_bins = (duration_ms / self.bin_ms).ceil() as usize;
        let mut counts = vec![0u32; n_bins];
        let mut spike_sets: Vec<Vec<usize>> = vec![Vec::new(); n_bins];

        for &(t, nid) in spikes {
            let bin = ((t / self.bin_ms) as usize).min(n_bins.saturating_sub(1));
            counts[bin] += 1;
            spike_sets[bin].push(nid);
        }

        // Convert counts to Hz
        let bin_s = self.bin_ms / 1000.0;
        let rates: Vec<f32> = counts.iter()
            .map(|&c| c as f32 / (self.n_neurons as f32 * bin_s))
            .collect();

        // Threshold-crossing detection
        let mut bursts = Vec::new();
        let mut in_burst = false;
        let mut burst_start = 0usize;
        let mut burst_spikes = Vec::new();

        for (bin, &rate) in rates.iter().enumerate() {
            if !in_burst && rate >= self.threshold_hz {
                in_burst = true;
                burst_start = bin;
                burst_spikes.clear();
            }

            if in_burst {
                // collect spikes in this bin
                burst_spikes.extend_from_slice(&spike_sets[bin]);

                // burst ends
                if rate < self.threshold_hz {
                    in_burst = false;
                    let t_start = burst_start as f32 * self.bin_ms;
                    let t_end   = bin as f32 * self.bin_ms;

                    if t_end - t_start >= self.min_duration_ms && !burst_spikes.is_empty() {
                        let n_spikes = burst_spikes.len();
                        let participating: std::collections::HashSet<usize> =
                            burst_spikes.iter().cloned().collect();
                        let n_part = participating.len();
                        let peak_rate = rates[burst_start..bin]
                            .iter()
                            .cloned()
                            .fold(0.0f32, f32::max);

                        bursts.push(Burst {
                            t_start,
                            t_end,
                            peak_rate_hz: peak_rate,
                            n_spikes,
                            n_neurons: n_part,
                            recruitment: n_part as f32 / self.n_neurons as f32,
                        });
                    }
                }
            }
        }
        
        if in_burst && !burst_spikes.is_empty() {
            let t_start = burst_start as f32 * self.bin_ms;
            let t_end   = n_bins as f32 * self.bin_ms;

            if t_end - t_start >= self.min_duration_ms {
                let n_spikes = burst_spikes.len();
                let participating: std::collections::HashSet<usize> =
                    burst_spikes.iter().cloned().collect();
                let n_part = participating.len();
                let peak_rate = rates[burst_start..]
                    .iter()
                    .cloned()
                    .fold(0.0f32, f32::max);

                bursts.push(Burst {
                    t_start,
                    t_end,
                    peak_rate_hz: peak_rate,
                    n_spikes,
                    n_neurons: n_part,
                    recruitment: n_part as f32 / self.n_neurons as f32,
                });
            }
        }


        bursts
    }
}

// ── Power spectrum ────────────────────────────────────────────────────────────

/// Estimate the power spectrum of the population firing rate.
///
/// Uses a simplified DFT on binned spike counts.
/// Returns `(frequencies_hz, power)` pairs.
///
/// Note: For accurate spectra, use a proper windowed FFT (e.g. via `rustfft`).
/// This implementation is lightweight and dependency-free.
pub fn power_spectrum(
    spikes: &[(f32, usize)],
    n_neurons: usize,
    duration_ms: f32,
    bin_ms: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n_bins = (duration_ms / bin_ms).ceil() as usize;
    if n_bins < 4 { return (vec![], vec![]); }

    let mut counts = vec![0.0f32; n_bins];
    for &(t, _) in spikes {
        let bin = ((t / bin_ms) as usize).min(n_bins - 1);
        counts[bin] += 1.0 / (n_neurons as f32 * bin_ms / 1000.0); // Hz
    }

    // Simple DFT (O(n²) — for moderate n_bins this is fine)
    let n = n_bins;
    let dt_s = bin_ms / 1000.0;
    let freq_res = 1.0 / (n as f32 * dt_s); // Hz per bin

    let n_freq = n / 2;
    let mut freqs = Vec::with_capacity(n_freq);
    let mut power = Vec::with_capacity(n_freq);

    use std::f32::consts::PI;

    for k in 0..n_freq {
        let freq = k as f32 * freq_res;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (j, &x) in counts.iter().enumerate() {
            let angle = 2.0 * PI * k as f32 * j as f32 / n as f32;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        freqs.push(freq);
        power.push((re * re + im * im) / n as f32);
    }

    (freqs, power)
}

/// Find the dominant oscillation frequency (Hz) in the range `[min_hz, max_hz]`.
pub fn dominant_frequency(
    spikes: &[(f32, usize)],
    n_neurons: usize,
    duration_ms: f32,
    bin_ms: f32,
    min_hz: f32,
    max_hz: f32,
) -> Option<f32> {
    let (freqs, power) = power_spectrum(spikes, n_neurons, duration_ms, bin_ms);
    freqs.iter().zip(power.iter())
        .filter(|&(f, _)| *f >= min_hz && *f <= max_hz)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(f, _)| *f)
}

// ── Neural avalanche analysis ─────────────────────────────────────────────────

/// Result of an avalanche analysis.
///
/// At the critical point between order and chaos, neural activity organises
/// into scale-free (power-law) avalanches. This is a signature of criticality
/// and is observed in cortical slice recordings.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AvalancheResult {
    /// Sizes of detected avalanches (number of spikes per avalanche).
    pub sizes: Vec<usize>,
    /// Durations of detected avalanches (number of bins).
    pub durations: Vec<usize>,
    /// Estimated power-law exponent for size distribution (negative; ≈ -1.5 at criticality).
    pub tau: f32,
    /// R² of the power-law fit.
    pub r_squared: f32,
    /// Fraction of total time spent in avalanches.
    pub activity_fraction: f32,
}

impl AvalancheResult {
    /// Detect avalanches using the standard discrete-time method.
    ///
    /// An avalanche starts when a bin is non-empty after an empty bin,
    /// and ends at the next empty bin.
    pub fn detect(spikes: &[(f32, usize)], duration_ms: f32, bin_ms: f32) -> Self {
        let n_bins = (duration_ms / bin_ms).ceil() as usize;
        let mut counts = vec![0u32; n_bins];
        for &(t, _) in spikes {
            let bin = ((t / bin_ms) as usize).min(n_bins.saturating_sub(1));
            counts[bin] += 1;
        }

        // Detect avalanches
        let mut sizes = Vec::new();
        let mut durations = Vec::new();
        let mut in_av = false;
        let mut cur_size = 0usize;
        let mut cur_dur  = 0usize;
        let mut active_bins = 0usize;

        for &c in &counts {
            if c > 0 {
                if !in_av { in_av = true; cur_size = 0; cur_dur = 0; }
                cur_size += c as usize;
                cur_dur  += 1;
                active_bins += 1;
            } else if in_av {
                in_av = false;
                sizes.push(cur_size);
                durations.push(cur_dur);
            }
        }
        if in_av { sizes.push(cur_size); durations.push(cur_dur); }

        // Estimate power-law exponent via log-log linear regression
        let tau = fit_power_law(&sizes);
        let r_squared = power_law_r2(&sizes, tau);

        Self {
            sizes,
            durations,
            tau,
            r_squared,
            activity_fraction: active_bins as f32 / n_bins as f32,
        }
    }

    pub fn is_critical(&self) -> bool {
        self.tau.abs() > 0.5 && (self.tau + 1.5).abs() < 0.3 && self.r_squared > 0.85
    }

    pub fn summary(&self) -> String {
        format!(
            "avalanches={} τ={:.3} R²={:.3} activity={:.1}% {}",
            self.sizes.len(), self.tau, self.r_squared,
            self.activity_fraction * 100.0,
            if self.is_critical() { "⚡ CRITICAL" } else { "" }
        )
    }
}

fn fit_power_law(sizes: &[usize]) -> f32 {
    if sizes.len() < 4 { return 0.0; }
    // Log-log linear regression: log(count) ~ τ·log(size)
    let mut log_s = Vec::new();
    let mut counts_map: HashMap<usize, usize> = HashMap::new();
    for &s in sizes { *counts_map.entry(s).or_insert(0) += 1; }
    let mut log_c = Vec::new();
    for (&s, &c) in &counts_map {
        if s > 0 && c > 0 {
            log_s.push((s as f32).ln());
            log_c.push((c as f32).ln());
        }
    }
    if log_s.len() < 3 { return 0.0; }
    // OLS slope
    let n = log_s.len() as f32;
    let sx: f32 = log_s.iter().sum();
    let sy: f32 = log_c.iter().sum();
    let sxx: f32 = log_s.iter().map(|&x| x*x).sum();
    let sxy: f32 = log_s.iter().zip(log_c.iter()).map(|(&x,&y)| x*y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-8 { return 0.0; }
    (n * sxy - sx * sy) / denom
}

fn power_law_r2(sizes: &[usize], tau: f32) -> f32 {
    if sizes.len() < 4 || tau == 0.0 { return 0.0; }
    let mut counts_map: HashMap<usize, usize> = HashMap::new();
    for &s in sizes { *counts_map.entry(s).or_insert(0) += 1; }
    let log_c: Vec<f32> = counts_map.iter()
        .filter(|&(s, _)| *s > 0)
        .map(|(&_, &c)| (c as f32).ln())
        .collect();
    if log_c.is_empty() { return 0.0; }
    let mean_y = log_c.iter().sum::<f32>() / log_c.len() as f32;
    let predicted: Vec<f32> = counts_map.iter()
        .filter(|&(s, _)| *s > 0)
        .map(|(&s, _)| tau * (s as f32).ln())
        .collect();
    let ss_res: f32 = log_c.iter().zip(predicted.iter()).map(|(&y,&yp)| (y-yp).powi(2)).sum();
    let ss_tot: f32 = log_c.iter().map(|&y| (y-mean_y).powi(2)).sum();
    if ss_tot < 1e-10 { return 1.0; }
    1.0 - ss_res / ss_tot
}

// ── ISI population statistics ─────────────────────────────────────────────────

/// Population-level ISI statistics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ISIStats {
    pub mean_isi_ms:  f32,
    pub cv:           f32,   // coefficient of variation
    pub fano_factor:  f32,   // variance-to-mean ratio of spike counts
    pub total_isis:   usize,
    pub n_active:     usize,
}

impl ISIStats {
    pub fn compute(spikes: &[(f32, usize)], _n_neurons: usize, duration_ms: f32, bin_ms: f32) -> Self {
        let mut per_neuron: HashMap<usize, Vec<f32>> = HashMap::new();
        for &(t, nid) in spikes {
            per_neuron.entry(nid).or_default().push(t);
        }

        let mut all_isis = Vec::new();
        for times in per_neuron.values_mut() {
            times.sort_by(|a,b| a.partial_cmp(b).unwrap());
            for w in times.windows(2) {
                all_isis.push(w[1] - w[0]);
            }
        }

        let total = all_isis.len();
        if total == 0 { return Self::default(); }

        let mean = all_isis.iter().sum::<f32>() / total as f32;
        let var  = all_isis.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / total as f32;
        let cv   = var.sqrt() / mean.max(1e-6);

        // Fano factor: variance / mean of spike counts in bins
        let n_bins = (duration_ms / bin_ms).ceil() as usize;
        let mut counts = vec![0u32; n_bins];
        for &(t, _) in spikes {
            let bin = ((t / bin_ms) as usize).min(n_bins.saturating_sub(1));
            counts[bin] += 1;
        }
        let mean_count = counts.iter().sum::<u32>() as f32 / n_bins as f32;
        let var_count  = counts.iter().map(|&c| (c as f32 - mean_count).powi(2)).sum::<f32>() / n_bins as f32;
        let fano = if mean_count > 1e-6 { var_count / mean_count } else { 0.0 };

        Self {
            mean_isi_ms: mean,
            cv,
            fano_factor: fano,
            total_isis: total,
            n_active: per_neuron.len(),
        }
    }
}

impl std::fmt::Display for ISIStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mean_ISI={:.1}ms  CV={:.3}  Fano={:.3}  active={}/total  n_isi={}",
            self.mean_isi_ms, self.cv, self.fano_factor, self.n_active, self.total_isis)
    }
}
