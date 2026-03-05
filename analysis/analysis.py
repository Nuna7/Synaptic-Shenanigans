"""
Spike train analysis toolkit for Synaptic-Shenanigans.

Provides:
  - Spike raster plots
  - Population firing rate (PSTH)
  - Inter-spike interval (ISI) distribution
  - Fano factor & coefficient of variation
  - Cross-correlation between neuron pairs
  - Voltage trace plots
  - Weight evolution plots (for STDP)
  - Network connectivity matrix

Usage:
    from analysis import SpikeAnalyzer, plot_raster, plot_psth
    from neurosim_ffi import NeuroSim

    with NeuroSim.basic(n_neurons=100, n_threads=1, seed=42) as sim:
        for t in range(0, 500, 10):
            sim.push_current(float(t), 0, 200.0)
        sim.run_until(500.0)
        spikes = sim.get_spikes()

    sa = SpikeAnalyzer(spikes, n_neurons=100, sim_duration=500.0)
    sa.plot_raster(save_path="raster.png")
    sa.plot_psth(bin_ms=10.0, save_path="psth.png")
    sa.print_summary()
"""

from __future__ import annotations
import math
import csv
import os
from typing import List, Tuple, Optional, Dict


# ────────────────────────────────────────────────────────────────────────────
# Core data structure
# ────────────────────────────────────────────────────────────────────────────

class SpikeAnalyzer:
    """
    Wraps a spike log and exposes analysis + plotting.

    Parameters
    ----------
    spikes : list of (time_ms, neuron_id)
    n_neurons : total neuron count in the simulation
    sim_duration : total simulation time (ms)
    """

    def __init__(
        self,
        spikes: List[Tuple[float, int]],
        n_neurons: int,
        sim_duration: float,
    ) -> None:
        self.spikes = sorted(spikes, key=lambda x: x[0])
        self.n_neurons = n_neurons
        self.sim_duration = sim_duration

        # Pre-compute per-neuron spike trains
        self._per_neuron: Dict[int, List[float]] = {i: [] for i in range(n_neurons)}
        for t, nid in self.spikes:
            if 0 <= nid < n_neurons:
                self._per_neuron[nid].append(t)

    # ── Firing rate ──────────────────────────────────────────────────────────

    def mean_firing_rate(self, neuron: Optional[int] = None) -> float:
        """Mean firing rate in Hz for one neuron or the whole population."""
        duration_s = self.sim_duration / 1000.0
        if duration_s == 0:
            return 0.0
        if neuron is not None:
            return len(self._per_neuron[neuron]) / duration_s
        total = sum(len(v) for v in self._per_neuron.values())
        return total / (self.n_neurons * duration_s)

    def firing_rates(self) -> List[float]:
        """Per-neuron firing rates in Hz."""
        dur = self.sim_duration / 1000.0
        return [len(self._per_neuron[i]) / dur if dur > 0 else 0.0
                for i in range(self.n_neurons)]

    # ── ISI ─────────────────────────────────────────────────────────────────

    def isis(self, neuron: Optional[int] = None) -> List[float]:
        """Inter-spike intervals (ms) for one neuron or all neurons."""
        if neuron is not None:
            st = sorted(self._per_neuron[neuron])
            return [st[i+1] - st[i] for i in range(len(st) - 1)]
        result = []
        for nid in range(self.n_neurons):
            result.extend(self.isis(neuron=nid))
        return result

    def cv(self, neuron: Optional[int] = None) -> float:
        """
        Coefficient of Variation of ISIs.
        CV ≈ 1 → Poisson-like, CV < 1 → regular, CV > 1 → bursting.
        """
        vals = self.isis(neuron)
        if len(vals) < 2:
            return float('nan')
        mean = sum(vals) / len(vals)
        if mean == 0:
            return float('nan')
        var = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
        return math.sqrt(var) / mean

    def fano_factor(self, bin_ms: float = 50.0) -> float:
        """
        Fano factor of spike counts across bins.
        FF ≈ 1 → Poisson, FF < 1 → sub-Poisson (regular), FF > 1 → super-Poisson.
        """
        n_bins = max(1, int(self.sim_duration / bin_ms))
        counts = [0] * n_bins
        for t, _ in self.spikes:
            b = min(int(t / bin_ms), n_bins - 1)
            counts[b] += 1
        if not counts:
            return float('nan')
        mean = sum(counts) / len(counts)
        if mean == 0:
            return float('nan')
        var = sum((c - mean)**2 for c in counts) / len(counts)
        return var / mean

    # ── PSTH ────────────────────────────────────────────────────────────────

    def psth(self, bin_ms: float = 10.0) -> Tuple[List[float], List[float]]:
        """
        Peri-stimulus time histogram.
        Returns (bin_centers_ms, rate_Hz).
        """
        n_bins = max(1, int(self.sim_duration / bin_ms))
        counts = [0] * n_bins
        for t, _ in self.spikes:
            b = min(int(t / bin_ms), n_bins - 1)
            counts[b] += 1
        dur_s = bin_ms / 1000.0
        centers = [(b + 0.5) * bin_ms for b in range(n_bins)]
        rates   = [c / (self.n_neurons * dur_s) for c in counts]
        return centers, rates

    # ── Cross-correlation ────────────────────────────────────────────────────

    def cross_correlation(
        self,
        nid_a: int,
        nid_b: int,
        max_lag_ms: float = 50.0,
        bin_ms: float = 1.0,
    ) -> Tuple[List[float], List[int]]:
        """
        Spike cross-correlogram between two neurons.
        Returns (lag_bins_ms, counts).
        """
        spikes_a = self._per_neuron[nid_a]
        spikes_b = self._per_neuron[nid_b]
        n_bins = int(max_lag_ms / bin_ms)
        counts = [0] * (2 * n_bins + 1)

        for ta in spikes_a:
            for tb in spikes_b:
                lag = tb - ta
                if -max_lag_ms <= lag <= max_lag_ms:
                    idx = int(lag / bin_ms) + n_bins
                    if 0 <= idx < len(counts):
                        counts[idx] += 1

        lags = [(i - n_bins) * bin_ms for i in range(len(counts))]
        return lags, counts

    # ── Summary ──────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a text summary of population activity."""
        rates = self.firing_rates()
        active = sum(1 for r in rates if r > 0)
        mean_rate = sum(rates) / len(rates) if rates else 0.0
        all_isis = self.isis()
        cv = self.cv()
        ff = self.fano_factor()

        print("=" * 50)
        print("  Spike Train Summary")
        print("=" * 50)
        print(f"  Neurons:           {self.n_neurons}")
        print(f"  Active neurons:    {active} ({100*active//self.n_neurons}%)")
        print(f"  Total spikes:      {len(self.spikes)}")
        print(f"  Sim duration:      {self.sim_duration:.1f} ms")
        print(f"  Mean firing rate:  {mean_rate:.2f} Hz")
        print(f"  ISI count:         {len(all_isis)}")
        if all_isis:
            mean_isi = sum(all_isis) / len(all_isis)
            print(f"  Mean ISI:          {mean_isi:.2f} ms")
        print(f"  CV (ISI):          {cv:.3f}")
        print(f"  Fano factor:       {ff:.3f}")
        print("=" * 50)

    # ── Plot (matplotlib if available, else ASCII fallback) ──────────────────

    def plot_raster(
        self,
        t_start: float = 0.0,
        t_end: Optional[float] = None,
        save_path: Optional[str] = None,
        title: str = "Spike Raster",
    ) -> None:
        t_end = t_end or self.sim_duration
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            times = [t for t, _ in self.spikes if t_start <= t <= t_end]
            nids  = [n for t, n in self.spikes if t_start <= t <= t_end]
            ax.scatter(times, nids, s=0.8, c='black', alpha=0.7, linewidths=0)
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.set_ylabel("Neuron ID", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_xlim(t_start, t_end)
            ax.set_ylim(-0.5, self.n_neurons - 0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150)
                print(f"Raster saved → {save_path}")
            else:
                plt.show()
            plt.close(fig)
        except ImportError:
            self._ascii_raster(t_start, t_end)

    def plot_psth(
        self,
        bin_ms: float = 10.0,
        save_path: Optional[str] = None,
        title: str = "Population Firing Rate (PSTH)",
    ) -> None:
        centers, rates = self.psth(bin_ms)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(centers, rates, width=bin_ms * 0.9, color='steelblue', alpha=0.8)
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.set_ylabel("Firing rate (Hz)", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150)
                print(f"PSTH saved → {save_path}")
            else:
                plt.show()
            plt.close(fig)
        except ImportError:
            print("matplotlib not available. PSTH (text):")
            for c, r in zip(centers, rates):
                bar = "#" * int(r / max(rates) * 40) if rates else ""
                print(f"  t={c:6.1f} ms  {r:6.2f} Hz  {bar}")

    def plot_isi_distribution(
        self,
        bins: int = 50,
        save_path: Optional[str] = None,
        log_scale: bool = True,
    ) -> None:
        all_isis = self.isis()
        if not all_isis:
            print("No ISIs to plot.")
            return
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(all_isis, bins=bins, color='coral', edgecolor='white', alpha=0.85)
            ax.set_xlabel("ISI (ms)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"ISI Distribution  (CV={self.cv():.3f})", fontsize=14)
            if log_scale:
                ax.set_yscale('log')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150)
                print(f"ISI histogram saved → {save_path}")
            else:
                plt.show()
            plt.close(fig)
        except ImportError:
            print(f"ISI range: {min(all_isis):.2f}–{max(all_isis):.2f} ms  (n={len(all_isis)})")

    def _ascii_raster(self, t_start: float, t_end: float, width: int = 80, height: int = 20) -> None:
        """Minimal ASCII raster for environments without matplotlib."""
        grid = [[" "] * width for _ in range(height)]
        for t, nid in self.spikes:
            if t_start <= t <= t_end and 0 <= nid < self.n_neurons:
                col = int((t - t_start) / (t_end - t_start) * (width - 1))
                row = int(nid / self.n_neurons * height)
                grid[min(row, height - 1)][col] = "·"
        print("┌" + "─" * width + "┐")
        for row in grid:
            print("│" + "".join(row) + "│")
        print("└" + "─" * width + "┘")
        print(f"  t={t_start:.0f}ms {'':>{width-20}}t={t_end:.0f}ms")

    # ── Export ───────────────────────────────────────────────────────────────

    def export_csv(self, path: str) -> None:
        """Export spike log to CSV."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_ms", "neuron_id"])
            for t, nid in self.spikes:
                w.writerow([f"{t:.4f}", nid])
        print(f"Spike log → {path}  ({len(self.spikes)} spikes)")

    def export_rate_csv(self, path: str) -> None:
        """Export per-neuron firing rates to CSV."""
        rates = self.firing_rates()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["neuron_id", "rate_hz", "n_spikes"])
            for nid, rate in enumerate(rates):
                w.writerow([nid, f"{rate:.4f}", len(self._per_neuron[nid])])
        print(f"Rates → {path}")


# ────────────────────────────────────────────────────────────────────────────
# Voltage trace helper
# ────────────────────────────────────────────────────────────────────────────

class VoltageTracer:
    """Records and plots membrane potential traces."""

    def __init__(self) -> None:
        self.traces: Dict[int, List[Tuple[float, float]]] = {}

    def record(self, t: float, neuron: int, v: float) -> None:
        if neuron not in self.traces:
            self.traces[neuron] = []
        self.traces[neuron].append((t, v))

    def plot(
        self,
        neurons: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        title: str = "Voltage Traces",
    ) -> None:
        to_plot = neurons or list(self.traces.keys())[:6]
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(to_plot), 1, figsize=(12, 2 * len(to_plot)), sharex=True)
            if len(to_plot) == 1:
                axes = [axes]
            colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
            for ax, nid, col in zip(axes, to_plot, colors):
                if nid in self.traces:
                    ts, vs = zip(*self.traces[nid])
                    ax.plot(ts, vs, lw=0.8, color=col)
                    ax.set_ylabel(f"N{nid}\n(mV)", fontsize=9)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
            axes[-1].set_xlabel("Time (ms)", fontsize=11)
            fig.suptitle(title, fontsize=13, y=1.01)
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Voltage trace → {save_path}")
            else:
                plt.show()
            plt.close(fig)
        except ImportError:
            for nid in to_plot:
                if nid in self.traces:
                    vs = [v for _, v in self.traces[nid]]
                    print(f"Neuron {nid}: min={min(vs):.1f}  max={max(vs):.1f} mV")


# ────────────────────────────────────────────────────────────────────────────
# STDP weight evolution plotter
# ────────────────────────────────────────────────────────────────────────────

def plot_weight_evolution(
    csv_path: str,
    save_path: Optional[str] = None,
    n_sample: int = 20,
) -> None:
    """
    Read the STDP weight CSV produced by stdp_demo and plot evolution curves.

    CSV format: trial,syn_idx,pre,post,weight
    """
    from collections import defaultdict
    data: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            syn_idx = int(row["syn_idx"])
            trial   = int(row["trial"])
            weight  = float(row["weight"])
            data[syn_idx].append((trial, weight))

    sample_ids = sorted(data.keys())[:n_sample]

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.cm.viridis
        for i, syn_id in enumerate(sample_ids):
            pairs = sorted(data[syn_id])
            trials  = [p[0] for p in pairs]
            weights = [p[1] for p in pairs]
            ax.plot(trials, weights, alpha=0.7, lw=1.2,
                    color=cmap(i / max(len(sample_ids) - 1, 1)))
        ax.set_xlabel("Trial", fontsize=12)
        ax.set_ylabel("Synaptic weight", fontsize=12)
        ax.set_title(f"STDP Weight Evolution (first {n_sample} synapses)", fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Weight evolution → {save_path}")
        else:
            plt.show()
        plt.close(fig)
    except ImportError:
        print(f"Loaded {len(data)} synapse weight traces. Install matplotlib to plot.")


# ────────────────────────────────────────────────────────────────────────────
# Quick convenience functions
# ────────────────────────────────────────────────────────────────────────────

def plot_raster(
    spikes: List[Tuple[float, int]],
    n_neurons: int,
    sim_duration: float,
    **kwargs,
) -> None:
    """One-liner raster plot."""
    SpikeAnalyzer(spikes, n_neurons, sim_duration).plot_raster(**kwargs)


def plot_psth(
    spikes: List[Tuple[float, int]],
    n_neurons: int,
    sim_duration: float,
    bin_ms: float = 10.0,
    **kwargs,
) -> None:
    """One-liner PSTH."""
    SpikeAnalyzer(spikes, n_neurons, sim_duration).plot_psth(bin_ms=bin_ms, **kwargs)


# ────────────────────────────────────────────────────────────────────────────
# Demo (run standalone to verify without a live simulation)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    rng = random.Random(42)

    # Synthetic Poisson spike trains
    n, dur = 100, 500.0
    rate   = 20.0  # Hz
    prob   = rate / 1000.0  # per ms

    synth = []
    for nid in range(n):
        t = 0.0
        while t < dur:
            t += 1.0
            if rng.random() < prob:
                synth.append((t, nid))

    print(f"Generated {len(synth)} synthetic spikes over {dur} ms ({n} neurons).")

    sa = SpikeAnalyzer(synth, n, dur)
    sa.print_summary()

    os.makedirs("bench/results", exist_ok=True)
    sa.export_csv("bench/results/synthetic_spikes.csv")
    sa.plot_raster(save_path="bench/results/raster_demo.png",
                   title="Synthetic Poisson Spike Raster")
    sa.plot_psth(bin_ms=20.0, save_path="bench/results/psth_demo.png")
    sa.plot_isi_distribution(save_path="bench/results/isi_demo.png")
