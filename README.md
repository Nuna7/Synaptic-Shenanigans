# Synaptic-Shenanigans

**A deterministic, reproducible, multi-threaded spiking neural network simulation engine in Rust.**

Synaptic-Shenanigans is a research-grade event-driven simulator designed for **correctness first**, while providing a rich library of neuron models, learning rules, network topologies, and analysis tools.

```
 ╔═══════════════════════════════════════════════════════════╗
 ║  Same seed → identical spike trains. Always. Guaranteed.  ║
 ╚═══════════════════════════════════════════════════════════╝
```

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Map](#feature-map)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Neuron Models](#neuron-models)
6. [Synaptic Plasticity](#synaptic-plasticity)
7. [Network Topologies](#network-topologies)
8. [Input Generation](#input-generation)
9. [Population Metrics](#population-metrics)
10. [Scheduler Modes](#scheduler-modes)
11. [Checkpointing](#checkpointing)
12. [Benchmarking](#benchmarking)
13. [Python Interface](#python-interface)
14. [Test Suite](#test-suite)
15. [Architecture](#architecture)
16. [Status & Roadmap](#status--roadmap)

---

## Overview

Synaptic-Shenanigans is built around four core principles:

| Principle | Mechanism |
|---|---|
| **Determinism** | Seeded RNG, explicit event sequencing, deterministic heap ordering |
| **Reproducibility** | Single-thread ≡ multi-thread ≡ checkpoint-resume |
| **Safety** | `Arc + AtomicCell`, no data races, no global mutable state |
| **Extensibility** | `NeuronPopulation` trait allows custom neuron models |

---

## Feature Map

```
synaptic-shenanigans/
├── Neuron Models
│   ├── LIF          — Leaky Integrate-and-Fire (fast; used in Simulation)
│   ├── Izhikevich   — 6 firing types: RS, IB, CH, FS, LTS, Resonator (standalone)
│   └── Hodgkin-HH   — Biophysically accurate Na⁺/K⁺/leak channels (standalone)
│
├── Plasticity
│   ├── STDP         — Spike-timing dependent (Hebbian)
│   └── Homeostatic  — Intrinsic excitability regulation (firing rate control)
│   └── Synaptic Scaling — Multiplicative weight homeostasis (network-level)
│
├── Network Topologies
│   ├── Erdős-Rényi  — Random G(n,p)
│   ├── Watts-Strogatz — Small-world (high clustering, short paths)
│   ├── Barabási-Albert — Scale-free (power-law degree distribution)
│   └── Layered Feedforward — Hierarchical sensory networks
│
├── Input Generation
│   ├── PoissonSource     — Homogeneous, seeded, reproducible
│   ├── PoissonPopulation — N independent channels → N neurons
│   └── StimulusPattern   — Inhomogeneous (step, sinusoidal)
│
├── Population Metrics
│   ├── SynchronyIndex  — χ: asynchronous vs. synchronous state
│   ├── BurstDetector   — Threshold-crossing burst finding
│   ├── PowerSpectrum   — Oscillation frequency analysis
│   ├── AvalancheResult — Scale-free activity (criticality)
│   └── ISIStats        — CV, Fano factor, mean ISI
│
├── Infrastructure
│   ├── Scheduler      — SingleThreaded / Deterministic-MT / Performance-MT
│   ├── Checkpointing  — Save → Load → Resume with SHA256 verification
│   ├── FFI            — C ABI for Python (ctypes)
│   └── gRPC           — Remote simulation + streaming spikes
│
└── Analysis
    └── analysis/      — Spike raster, PSTH, ISI distribution, voltage traces
```

---

## Installation

### Requirements

- Rust ≥ 1.72
- Linux / macOS (flamegraph profiling is Linux-only)

```bash
git clone https://github.com/yourname/Synaptic-Shenanigans.git
cd Synaptic-Shenanigans
cargo build --release
```

### Optional Python dependencies

```bash
pip install matplotlib numpy grpcio grpcio-tools
```

### Regenerating gRPC stubs

After editing `rpc/neurosim.proto`, regenerate the Python stubs:

```bash
python -m grpc_tools.protoc \
    -I rpc \
    --python_out=. \
    --grpc_python_out=. \
    rpc/neurosim.proto
```

---

## Quick Start

### Minimal deterministic simulation

```rust
use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::synapse::Synapse;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};

fn main() {
    let neurons  = LifNeuron::new(100, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let synapses = Synapse::new();

    let mut sim = Simulation::new_with_seed(neurons, synapses, 1.0, 42, 1);
    sim.push_event(10.0, 0, 100.0, 0, 0.0);
    sim.run_auto(100.0);

    for (t, neuron) in &sim.spike_log {
        println!("Neuron {} spiked at t={}", neuron, t);
    }
}
```

### Small-world network with Poisson drive

```rust
use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::simulation::Simulation;
use synaptic_shenanigans::network::{NetworkBuilder, EdgeParams};
use synaptic_shenanigans::poisson::drive_background;
use synaptic_shenanigans::metrics::SynchronyIndex;

let n   = 200;
let syn = NetworkBuilder::small_world(n, 6, 0.1, EdgeParams::default(), 42);
let neurons = LifNeuron::new(n, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
let mut sim = Simulation::new_with_seed(neurons, syn, 1.0, 42, 1);

drive_background(&mut sim, n, 10.0, 60.0, 42, 1000.0);
sim.run_auto(1000.0);

let sync = SynchronyIndex::compute(&sim.spike_log, n, 1000.0, 5.0);
println!("{}", sync); // χ = 0.0321  [Asynchronous Irregular (AI)]
```

---

## Neuron Models

All three models implement the `NeuronPopulation` trait, which provides a
shared interface for stepping, partitioning, and reading state.

> **Note on `Simulation` compatibility.**
> **Note on `Simulation` compatibility.**
> The event-driven `Simulation` struct currently uses `Arc<LifNeuron>` directly.
> `IzhikevichPop`, `HHPopulation`, and `AdExPopulation` are **standalone** — they
> integrate independently via their own `step_range` call and are not yet pluggable
> into `Simulation`. Connecting them to the scheduler is on the roadmap.

### LIF — Leaky Integrate-and-Fire

The simplest spiking neuron. Best choice for large-scale network simulations.

```
τ_m · dV/dt = -(V - V_rest) + R_m · I_ext
if V ≥ V_thresh: spike, V ← V_rest, enter refractory
```

```rust
let neurons = LifNeuron::new(
    n,      // population size
    -65.0,  // V_rest (mV)
    -50.0,  // V_thresh (mV)
    20.0,   // τ_m (ms)
    1.0,    // R_m (MΩ)
    1.0,    // dt (ms)
    5.0,    // refractory period (ms)
);
```

### Izhikevich — Multi-type spiking (standalone)

A two-variable model that reproduces a wide range of cortical firing patterns
at low computational cost.

```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)
if v ≥ 30: v ← c, u ← u + d
```

| Type | a | b | c | d | Firing pattern |
|---|---|---|---|---|---|
| `RegularSpiking` | 0.02 | 0.20 | -65 | 8 | Most cortical excitatory cells |
| `IntrinsicallyBursting` | 0.02 | 0.20 | -55 | 4 | Burst on first input, then RS |
| `Chattering` | 0.02 | 0.20 | -50 | 2 | High-frequency burst trains |
| `FastSpiking` | 0.10 | 0.20 | -65 | 2 | GABAergic interneurons |
| `LowThresholdSpiking` | 0.02 | 0.25 | -65 | 2 | Another interneuron class |
| `Resonator` | 0.10 | 0.26 | -65 | 2 | Responds at natural frequency |

```rust
use synaptic_shenanigans::izhikevich::{IzhikevichPop, NeuronType};

// Homogeneous population
let pop = IzhikevichPop::homogeneous(100, NeuronType::FastSpiking, 0.25);

// Mixed cortical (800 RS + 200 FS with parameter noise)
let pop = IzhikevichPop::mixed_cortical(800, 200, 0.25, seed);
```

**Run demo:** `cargo run --release --bin izh_demo`

### Hodgkin-Huxley — Biophysically accurate (standalone)

The original conductance-based model with explicit voltage-gated ion channels.

```
Cm · dV/dt = -g_Na·m³·h·(V - E_Na) - g_K·n⁴·(V - E_K) - g_L·(V - E_L) + I
dm/dt = α_m(V)·(1-m) - β_m(V)·m
dh/dt = α_h(V)·(1-h) - β_h(V)·h
dn/dt = α_n(V)·(1-n) - β_n(V)·n
```

Key differences from LIF/Izhikevich:
- Refractory period emerges from channel kinetics (no hard timer)
- Correct spike shape and afterhyperpolarization
- Subthreshold oscillations and resonance

```rust
use synaptic_shenanigans::hodgkin_huxley::{HHPopulation, HHParams};

let pop = HHPopulation::homogeneous(50, HHParams::default());

// With heterogeneity (5% parameter noise)
let pop = HHPopulation::heterogeneous(50, HHParams::default(), 0.05, seed);
```

**Run demo:** `cargo run --release --bin hh_demo`

### AdEx — Adaptive Exponential Integrate-and-Fire (standalone)

Bridges LIF simplicity and Hodgkin-Huxley realism. Two features LIF cannot produce:
exponential spike initiation that matches cortical recordings, and spike-frequency
adaptation where firing rate decreases under sustained input.

```
C dV/dt = -g_L(V-E_L) + g_L·Δ_T·exp((V-V_T)/Δ_T) - w + I
τ_w dw/dt = a(V-E_L) - w
on spike: V ← V_r,  w ← w + b
```

Five built-in profiles calibrated to published recordings:

| Profile | `a` | `b` | `tau_w` | Firing pattern |
|---|---|---|---|---|
| `AdaptingRS`     | 4.0  | 80   | 150 ms | Most cortical pyramidal cells |
| `Bursting`       | -0.5 | 7    | 300 ms | Burst followed by adaptation |
| `TonicRS`        | 0.0  | 0    | 100 ms | No adaptation, tonic firing |
| `FastSpiking`    | 0.0  | 0    | 10 ms  | Interneuron, minimal adaptation |
| `TransientBurst` | 4.0  | 200  | 500 ms | Initial burst then silence |

```rust
use synaptic_shenanigans::adex::{AdExPopulation, AdExProfile};

// Homogeneous population
let pop = AdExPopulation::from_profile(100, AdExProfile::AdaptingRS);

// Heterogeneous (15% parameter noise)
let pop = AdExPopulation::heterogeneous(100, AdExProfile::AdaptingRS, 0.15, seed);
```

**Run demo:** `cargo run --release --bin adex_demo`

---

## Synaptic Plasticity

### STDP — Spike-Timing Dependent Plasticity

```
ΔW_+ = A_+ · exp(-Δt / τ_+)   if pre before post (LTP)
ΔW_- = -A_- · exp( Δt / τ_-) if post before pre (LTD)
```

```rust
use synaptic_shenanigans::plasticity::{StdpState, StdpConfig};

let mut stdp = StdpState::new(n_neurons, n_synapses, StdpConfig::default());

// Per timestep:
stdp.decay_traces(dt);
stdp.accumulate_for_spike(nid, t, &syn.pre, &syn.post, &pre_index);
stdp.flush_weight_updates(&mut syn.weight);

println!("{}", StdpState::weight_stats(&syn.weight));
```

**Run demo:** `cargo run --release --bin stdp_demo`

### Homeostatic Intrinsic Plasticity

```
τ_h · dθ/dt = r_actual(t) - r_target
```

```rust
use synaptic_shenanigans::homeostatic::{HomeostaticState, HomeostaticConfig};

let mut homeo = HomeostaticState::new(n_neurons, -50.0, HomeostaticConfig::default());

// Per timestep:
for &(t, nid) in &sim.spike_log { homeo.record_spike(nid, t); }
homeo.update(current_time);
homeo.apply_thresholds_to_lif(&mut neurons);
```

**Run demo:** `cargo run --release --bin homeostatic_demo`

### Synaptic Scaling — Multiplicative weight homeostasis

The network-level complement to intrinsic homeostasis. Rather than adjusting firing
thresholds, synaptic scaling renormalises all incoming weights of a neuron
multiplicatively, preserving the relative differences that STDP created while
correcting overall rate.

```
W_i(t+1) = W_i(t) · (r_target / r_actual)^α
```

```rust
use synaptic_shenanigans::synaptic_scaling::{SynapticScaling, SynapticScalingConfig};

let cfg = SynapticScalingConfig {
    target_rate_hz: 5.0,
    alpha: 1.0,
    w_min: 0.0, w_max: 15.0,
    update_interval_ms: 200.0,
    rate_window_ms: 500.0,
    enabled: true,
};
let mut scaler = SynapticScaling::new(n_neurons, cfg);

// Per reporting window:
for &(t, nid) in &sim.spike_log { scaler.record_spike(nid, t); }
scaler.scale_weights(current_time, &sim.synapses.post, &mut sim.synapses.weight);
```

**Run demo:** `cargo run --release --bin synaptic_scaling_demo`


---

## Network Topologies

All generators are seeded and return a `Synapse` ready to attach to `Simulation`.

```rust
use synaptic_shenanigans::network::{NetworkBuilder, EdgeParams};

let ep = EdgeParams::default();

// Erdős-Rényi G(n, p)
let syn = NetworkBuilder::erdos_renyi(200, 0.05, ep.clone(), 42);

// Watts-Strogatz small-world
let syn = NetworkBuilder::small_world(200, 6, 0.1, ep.clone(), 42);

// Barabási-Albert scale-free
let syn = NetworkBuilder::scale_free(200, 3, ep.clone(), 42);

// Layered feedforward
let (syn, ranges) = NetworkBuilder::layered_feedforward(
    &[40, 80, 80, 40], 0.25, ep, 42,
);
```

```rust
use synaptic_shenanigans::network::NetworkMetrics;
println!("{}", NetworkMetrics::compute(&syn, n));
// neurons=200 synapses=1153 mean_in=5.8 mean_out=5.8 max_out=28
```

**Run demo:** `cargo run --release --bin topology_demo`

---

## Input Generation

```rust
use synaptic_shenanigans::poisson::{PoissonSource, PoissonPopulation, StimulusPattern, drive_background};

// Single homogeneous source
let mut src = PoissonSource::new(20.0, 42);
let spikes  = src.generate(0.0, 500.0);

// N independent channels → N neurons (pre-built)
let mut pop = PoissonPopulation::new(100, 20.0, 60.0, 42);
pop.prebuild(&mut sim, 1000.0);

// One-liner convenience helper
drive_background(&mut sim, 100, 20.0, 60.0, 42, 1000.0);

// Inhomogeneous: step stimulus
let mut pat = StimulusPattern::step(5.0, 80.0, 200.0, 400.0, seed);

// Inhomogeneous: sinusoidal gamma modulation
let mut pat = StimulusPattern::sinusoidal(10.0, 15.0, 40.0, seed);
```

---

## Population Metrics

```rust
use synaptic_shenanigans::metrics::{
    SynchronyIndex, BurstDetector, AvalancheResult,
    ISIStats, power_spectrum, dominant_frequency,
};

// Synchrony
let sync = SynchronyIndex::compute(&sim.spike_log, n, 1000.0, 5.0);
println!("{}", sync); // χ = 0.0241  [Asynchronous Irregular (AI)]

// Bursts
let detector = BurstDetector::new(n, 20.0, 5.0);
let bursts   = detector.detect(&sim.spike_log, 1000.0);

// Power spectrum & dominant frequency
let (freqs, power) = power_spectrum(&sim.spike_log, n, 1000.0, 1.0);
let gamma = dominant_frequency(&sim.spike_log, n, 1000.0, 1.0, 30.0, 80.0);

// Neural avalanches
let av = AvalancheResult::detect(&sim.spike_log, 1000.0, 1.0);
println!("{}", av.summary()); // avalanches=47 τ=-1.473 R²=0.912 ⚡ CRITICAL

// ISI statistics
let stats = ISIStats::compute(&sim.spike_log, n, 1000.0, 5.0);
println!("{}", stats); // mean_ISI=48.3ms  CV=0.923  Fano=1.041
```

**Run demo:** `cargo run --release --bin metrics_demo`

---

## Scheduler Modes

```rust
use synaptic_shenanigans::simulation::SchedulerMode;

sim.scheduler_mode = SchedulerMode::SingleThreaded;            // reference / debug
sim.scheduler_mode = SchedulerMode::Deterministic { n_threads: 4 }; // deterministic MT
sim.scheduler_mode = SchedulerMode::Performance  { n_threads: 4 }; // requires --features performance
```

| Mode | Deterministic | Notes |
|---|---|---|
| `SingleThreaded` | ✅ | Reference, debugging |
| `Deterministic { n_threads }` | ✅ | Production parallel simulation |
| `Performance { n_threads }` | ❌ | Rayon-based; requires `--features performance`; falls back to `SingleThreaded` if feature is absent |

**Determinism Contract:**

| Scenario | Guarantee |
|---|---|
| Same seed, same config | Identical spike logs |
| Single vs. multithreaded | Identical spike logs |
| Save → load → resume | Identical spike logs |

---

## Checkpointing

```rust
// Save (writes path and path + ".sha256")
sim.save_state("checkpoint.bin", "checkpoint.bin.sha256").unwrap();

// Load and continue — produces identical results to a continuous run
let mut sim2 = Simulation::load_state("checkpoint.bin", seed, n_threads).unwrap();
sim2.run_auto(500.0);

// Verify reproducibility
use synaptic_shenanigans::simulation::replay_equal;
assert!(replay_equal(build_sim, 500.0, 42));
```

---

## Benchmarking

```bash
# Criterion microbenchmark: 10 000 LIF neurons, one step
cargo bench

# Full-system harness: 10 000 neurons, 100 000 synapses, 1 000 ms
cargo run --release --bin bench_harness

# Flamegraph (Linux only)
cargo install flamegraph
cargo flamegraph --release --bin bench_harness
```

---

## Python Interface

### FFI (ctypes) — low-latency local control

```python
from neurosim_ffi import NeuroSim

with NeuroSim.basic(n_neurons=100, n_threads=1, seed=42) as sim:
    for t in range(0, 100, 10):
        sim.push_current(time=float(t), neuron=0, weight=400.0)

    sim.run_until(500.0)

    spikes   = sim.get_spikes()       # full log: [(time_ms, neuron_id), ...]
    voltages = sim.get_all_voltages() # [v0, v1, ..., vN]

    # Closed-loop control: clear the log between epochs
    sim.clear_spikes()
    sim.inject_spike(neuron=0, weight=300.0)
    sim.run_for(100.0)
    new_spikes = sim.get_spikes()

    sim.save_checkpoint("checkpoint.bin")
```

### gRPC — distributed / remote simulation

```bash
# Start the server (requires --features rpc)
cargo run --release --features rpc
```

```python
from neurosim_rpc import RemoteSim

with RemoteSim("127.0.0.1:50051", n_neurons=100, seed=42) as sim:
    sim.push(neuron=0, time=0.0, weight=400.0)

    spikes   = sim.step(until_time=500.0) # returns full spike log
    voltages = sim.voltages()             # separate GetVoltages call

    sim.clear_spikes()                    # reset spike log between epochs

    for t, nid in sim.stream_spikes():
        print(f"  spike: t={t:.1f} ms  neuron={nid}")
```

> **Note:** `StepReply` contains only spikes. Membrane voltages are always
> fetched via the separate `GetVoltages` RPC.

---

## Test Suite

```bash
cargo test
```

| Test file | What it proves |
|---|---|
| `tests/lif.rs` | LIF decay, spike, refractory, numerical stability |
| `tests/izhikevich.rs` | Determinism across neuron types |
| `tests/hodgkin_huxley.rs` | F-I curve, gating variables, refractory from kinetics |
| `tests/determinism.rs` | Same seed = same result; MT ≡ ST |
| `tests/simulation.rs` | Monotonic times, finite voltages, checkpoint roundtrip |
| `tests/network.rs` | Topology connectivity, delay bounds |
| `tests/plasticity.rs` | STDP LTP/LTD direction, weight bounds |
| `tests/homeostatic.rs` | Convergence, threshold clamping, disabled mode |
| `tests/poisson.rs` | Rate accuracy, ISI CV ≈ 1, reproducibility |
| `tests/metrics.rs` | Synchrony, burst detection, power spectrum, avalanches |
| `tests/topology_simulation.rs` | End-to-end simulation per topology |
| `tests/adapters_equivalence.rs` | Determinism across 50 seeds |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Simulation                               │
│  ┌────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │ EventQueue │  │  Arc<LifNeuron>│  │   ThreadLocals         │ │
│  │ BinaryHeap │  │ (concrete type)│  │ (per-thread buffers)   │ │
│  └────────────┘  └────────────────┘  └────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Schedulers: SingleThreaded | Deterministic | Performance │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
         ↕ FFI (C ABI)           ↕ gRPC (tonic)
    neurosim_ffi.py            neurosim_rpc.py

┌───────────────────────────────────────────────────────────────┐
│              Standalone NeuronPopulation models               │
│   IzhikevichPop   ·   HHPopulation   (not in Simulation yet) │
└───────────────────────────────────────────────────────────────┘
```

### Thread-safety model

- Neuron state: `AtomicCell<f32>` — lock-free per-cell reads and writes
- Shared neuron population: `Arc<LifNeuron>`
- No global mutable state; no interior mutability outside of atomics
- Deterministic merge barriers between every epoch

### Event ordering

Each event is tagged with `(tick, seq)`:
- `tick = floor(time / dt)` — discrete time bin
- `seq` — monotonically increasing per-simulation counter

The priority queue orders by `(tick ASC, seq ASC)`, guaranteeing deterministic
processing of simultaneous events regardless of thread scheduling.

---

## Demo Binaries

| Binary | Command | What it shows |
|---|---|---|
| `bench_harness`          | `cargo run --release --bin bench_harness`          | 10k neuron throughput benchmark |
| `hh_demo`                | `cargo run --release --bin hh_demo`                | HH F-I curve, gating variables |
| `izh_demo`               | `cargo run --release --bin izh_demo`               | All 6 Izhikevich types |
| `adex_demo`              | `cargo run --release --bin adex_demo`              | AdEx 5 profiles, F-I curve, adaptation |
| `stdp_demo`              | `cargo run --release --bin stdp_demo`              | STDP weight evolution |
| `topology_demo`          | `cargo run --release --bin topology_demo`          | ER vs WS vs BA comparison |
| `homeostatic_demo`       | `cargo run --release --bin homeostatic_demo`       | Rate regulation over 3 phases |
| `synaptic_scaling_demo`  | `cargo run --release --bin synaptic_scaling_demo`  | Weight homeostasis under input changes |
| `metrics_demo`           | `cargo run --release --bin metrics_demo`           | Sync, bursts, spectrum, avalanches |

---

## Status & Roadmap

## Status

All major components are implemented, tested, and documented.

### Complete 

| Component | Module | Tests | Demo binary |
|---|---|---|---|
| LIF neuron | `lif.rs` | `tests/lif.rs` | — |
| Izhikevich (6 types) | `izhikevich.rs` | `tests/izhikevich.rs` | `izh_demo` |
| Hodgkin-Huxley | `hodgkin_huxley.rs` | `tests/hodgkin_huxley.rs` | `hh_demo` |
| AdEx (5 profiles) | `adex.rs` | `tests/adex.rs` | `adex_demo` |
| STDP | `plasticity.rs` | `tests/plasticity.rs` | `stdp_demo` |
| Homeostatic (intrinsic) | `homeostatic.rs` | `tests/homeostatic.rs` | `homeostatic_demo` |
| Synaptic scaling | `synaptic_scaling.rs` | `tests/synaptic_scaling.rs` | `synaptic_scaling_demo` |
| AMPA/NMDA/GABA_A/GABA_B | `synapse.rs` | `tests/simulation.rs` | — |
| Erdős-Rényi, WS, BA, Layered | `network.rs` | `tests/network.rs` | `topology_demo` |
| Poisson (homo + inhomo) | `poisson.rs` | `tests/poisson.rs` | — |
| Synchrony, Bursts, Spectrum | `metrics.rs` | `tests/metrics.rs` | `metrics_demo` |
| Deterministic MT scheduler | `simulation.rs` | `tests/determinism.rs` | — |
| Checkpointing (SHA256) | `simulation.rs` | `tests/simulation.rs` | — |
| C FFI | `lib.rs` | — | — |
| gRPC server | `rpc.rs` | — | — |
| Python FFI wrapper | `neurosim_ffi.py` | — | — |
| Python gRPC wrapper | `neurosim_rpc.py` | — | — |
| Browser dashboard | `neural_dashboard.html` | — | — |

### Planned
- GPU offload (wgpu, still deterministic)
- NEST / Brian2 import compatibility
- Online closed-loop gRPC streaming at sub-millisecond latency

