# Synaptic-Shenanigans

**A deterministic, reproducible, multi-threaded spiking neural network simulation engine in Rust.**

```
 ╔═══════════════════════════════════════════════════════════╗
 ║  Same seed → identical spike trains. Always. Guaranteed.  ║
 ╚═══════════════════════════════════════════════════════════╝
```

---

## Module Layout

```
src/
├── Core flat modules (engine + all demos/tests depend on these)
│   ├── event.rs          — Event type for the BinaryHeap priority queue
│   ├── lif.rs            — LifNeuron + NeuronPopulation trait (canonical)
│   ├── simulation.rs     — Simulation struct, SchedulerMode, event loop
│   ├── synapse.rs        — Synapse (engine type) + SynapseMatrix (topology type)
│   ├── network.rs        — NetworkBuilder → Synapse (used by demos/tests)
│   ├── metrics.rs        — SynchronyIndex, BurstDetector, ISIStats, etc.
│   └── checkpoint.rs     — Checkpoint save/load with SHA-256
│
├── Flat shims (make `use synaptic_shenanigans::X::Y` work for all old imports)
│   ├── izhikevich.rs     — pub use neurons::izhikevich::*
│   ├── hodgkin_huxley.rs — pub use neurons::hh::*
│   ├── adex.rs           — pub use neurons::adex::*
│   ├── poisson.rs        — pub use input::poisson::*
│   ├── homeostatic.rs    — pub use plasticity::homeostatic::*
│   └── synaptic_scaling.rs — pub use plasticity::synaptic_scaling::*
│
├── neurons/              — Canonical neuron model implementations
│   ├── mod.rs            — re-exports + NeuronPopulation delegation
│   ├── lif.rs            — shim → crate::lif (Simulation owns Arc<LifNeuron>)
│   ├── izhikevich.rs     — IzhikevichPop, NeuronType (6 firing patterns)
│   ├── hh.rs             — HHPopulation, HHParams, steady_state
│   └── adex.rs           — AdExPopulation, AdExProfile (5 profiles)
│
├── plasticity/           — Canonical plasticity implementations
│   ├── mod.rs            — re-exports + PlasticityRule trait
│   ├── stdp.rs           — StdpState, StdpConfig (Hebbian spike-timing)
│   ├── homeostatic.rs    — HomeostaticState, HomeostaticConfig
│   └── synaptic_scaling.rs — SynapticScaling, SynapticScalingConfig
│
├── topology/             — TopologyGenerator trait + generators → SynapseMatrix
│   ├── mod.rs
│   ├── erdos_renyi.rs
│   ├── watts_strogatz.rs
│   ├── barabasi_albert.rs
│   └── layered.rs
│
└── input/                — StimulusSource trait + Poisson generators
    ├── mod.rs
    └── poisson.rs        — PoissonSource, PoissonPopulation, StimulusPattern
```

---

## Installation

```bash
git clone https://github.com/yourname/Synaptic-Shenanigans.git
cd Synaptic-Shenanigans
cargo build --release
```

### Optional Python dependencies

```bash
pip install matplotlib numpy grpcio grpcio-tools
```

---

## Quick Start

### Minimal deterministic simulation

```rust
use synaptic_shenanigans::{LifNeuron, Synapse, Simulation, SchedulerMode};

fn main() {
    let neurons  = LifNeuron::new(100, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let synapses = Synapse::new();

    let mut sim = Simulation::new_with_seed(neurons, synapses, 1.0, 42, 1);
    sim.push_event(10.0, 0, 100.0, 0, 0.0);
    sim.run_auto(100.0);

    for (t, nid) in &sim.spike_log {
        println!("Neuron {} spiked at t={:.1} ms", nid, t);
    }
}
```

### Small-world network with Poisson drive

```rust
use synaptic_shenanigans::{LifNeuron, Simulation};
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
println!("{}", sync);
```

---

## Neuron Models

All standalone neuron models implement the `NeuronPopulation` trait from `lif.rs`.
They integrate via their own `step_range` call and are **not yet** plugged into
`Simulation` (which uses `Arc<LifNeuron>` directly — generalisation is on the roadmap).

| Model | Module | Demo |
|---|---|---|
| LIF | `lif` / `neurons::lif` | — (used by Simulation) |
| Izhikevich (6 types) | `izhikevich` / `neurons::izhikevich` | `cargo run --bin izh_demo` |
| Hodgkin-Huxley | `hodgkin_huxley` / `neurons::hh` | `cargo run --bin hh_demo` |
| AdEx (5 profiles) | `adex` / `neurons::adex` | `cargo run --bin adex_demo` |

```rust
// Izhikevich
use synaptic_shenanigans::izhikevich::{IzhikevichPop, NeuronType};
let pop = IzhikevichPop::homogeneous(100, NeuronType::FastSpiking, 0.25);

// Hodgkin-Huxley
use synaptic_shenanigans::hodgkin_huxley::{HHPopulation, HHParams};
let pop = HHPopulation::homogeneous(50, HHParams::default());

// AdEx
use synaptic_shenanigans::adex::{AdExPopulation, AdExProfile};
let pop = AdExPopulation::from_profile(100, AdExProfile::AdaptingRS);
```

## Adding a New Neuron Model (3 steps)
 
```rust
// Step 1: src/neurons/my_model.rs
use crate::neurons::{NeuronPartition, NeuronPopulation};
pub struct MyPop { /* ... */ }
impl NeuronPopulation for MyPop {
    fn len(&self) -> usize { /* ... */ }
    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition> { /* ... */ }
    fn step_range(&self, input_current: &[f32], start: usize) { /* ... */ }
    fn local_spiked(&self, idx: usize) -> bool { /* ... */ }
    fn read_v(&self, idx: usize) -> f32 { /* ... */ }
    fn get_threshold(&self, idx: usize) -> f32 { /* ... */ }
    fn set_threshold(&self, idx: usize, v_thresh: f32) { /* ... */ }
    fn reset_neuron(&self, idx: usize, v_rest: f32) { /* ... */ }
}
 
// Step 2: add to src/neurons/mod.rs
pub mod my_model;
pub use my_model::MyPop;
 
// Step 3: use it
let sim = Simulation::new_with_neurons(MyPop::new(100), Synapse::new(), 1.0, 42, 1);
// No other files need changing.
```
 

---

## Synaptic Plasticity

```rust
// STDP
use synaptic_shenanigans::plasticity::{StdpState, StdpConfig};
let mut stdp = StdpState::new(n_neurons, n_synapses, StdpConfig::default());
stdp.decay_traces(dt);
stdp.accumulate_for_spike(nid, t, &syn.pre, &syn.post, &pre_index);
stdp.flush_weight_updates(&mut syn.weight);

// Homeostatic
use synaptic_shenanigans::homeostatic::{HomeostaticState, HomeostaticConfig};
let mut homeo = HomeostaticState::new(n, -50.0, HomeostaticConfig::default());
homeo.record_spike(nid, t);
homeo.update(current_time);
homeo.apply_thresholds_to_lif(&mut neurons);

// Synaptic Scaling
use synaptic_shenanigans::synaptic_scaling::{SynapticScaling, SynapticScalingConfig};
let mut scaler = SynapticScaling::new(n, SynapticScalingConfig::default());
scaler.record_spike(nid, t);
scaler.scale_weights(t_now, &syn.post, &mut syn.weight);
```

---

## Network Topologies


### New API — returns `SynapseMatrix` (via `topology/` subsystem)

```rust
use synaptic_shenanigans::topology::{TopologyGenerator, WattsStrogatz, WattsStrogatzParams};

let gen = WattsStrogatz::new(WattsStrogatzParams { k: 6, beta: 0.1, ..Default::default() });
let matrix = gen.generate(200, 42);
// Convert to Synapse for use with Simulation:
let syn = matrix.into_synapse(5.0 /* tau_syn_ms */);
```

---

## Scheduler Modes

```rust
use synaptic_shenanigans::SchedulerMode;

sim.scheduler_mode = SchedulerMode::SingleThreaded;              // deterministic, 1 thread
sim.scheduler_mode = SchedulerMode::Deterministic { n_threads: 4 }; // deterministic, 4 threads
sim.scheduler_mode = SchedulerMode::Performance { n_threads: 4 };   // fast, non-deterministic
                                                                      // (needs --features performance)
```

| Mode | Deterministic | Notes |
|---|---|---|
| `SingleThreaded` | Yes | Reference / debugging |
| `Deterministic { n_threads }` | Yes | Production parallel simulation |
| `Performance { n_threads }` | Partial | Requires `--features performance` |

---

## Checkpointing

```rust
// Save
sim.save_state("checkpoint.bin", "checkpoint.bin.sha256").unwrap();

// Load and resume — produces identical results to a continuous run
let mut sim2 = Simulation::load_state("checkpoint.bin", seed, n_threads).unwrap();
sim2.run_auto(500.0);

// Verify determinism
use synaptic_shenanigans::replay_equal;
assert!(replay_equal(|s| build_sim(s), 500.0, 42));
```

---

## Python Interface

### FFI (ctypes)

## Python FFI Setup
 
No extra install step beyond `cargo build --release`. The shared library is at:
 
| Platform | Path |
|---|---|
| Linux   | `target/release/libsynaptic_shenanigans.so` |
| macOS   | `target/release/libsynaptic_shenanigans.dylib` |
| Windows | `target/release/synaptic_shenanigans.dll` |
 
`neurosim_ffi.py` auto-discovers it. Override with:
 
```bash
export NEUROSIM_LIB=/path/to/libsynaptic_shenanigans.so
```
 
 ### Run Python FFI tests
 
```bash
pip install pytest
pytest tests/test_ffi.py -v
```

```python
from neurosim_ffi import NeuroSim
 
with NeuroSim.basic(n_neurons=100, n_threads=4, seed=42) as sim:
    sim.push_current(time=0.0, neuron=0, weight=400.0)
    sim.run_until(500.0)
    spikes   = sim.get_spikes()        # [(time_ms, neuron_id), ...]
    voltages = sim.get_all_voltages()  # [v0, v1, ..., vN]
    sim.save_checkpoint("ckpt.bin")    # also writes ckpt.bin.sha256
```

### gRPC

```bash
cargo run --release --features rpc
```

## gRPC Setup (optional feature)
 
### Step 1 — Install protoc
 
```bash
# macOS
brew install protobuf
 
# Ubuntu / Debian
sudo apt-get install -y protobuf-compiler
 
# Windows (Chocolatey)
choco install protoc
```

### Step 2 — Build the gRPC server binary
 
```bash
cargo build --release --features rpc
```
 
### Step 3 — Regenerate Python stubs (only needed after editing the .proto file)
 
```bash
pip install grpcio grpcio-tools
 
python -m grpc_tools.protoc \
    -I rpc \
    --python_out=. \
    --grpc_python_out=. \
    rpc/neurosim.proto
```

This overwrites `neurosim_pb2.py` and `neurosim_pb2_grpc.py`.
 
### Step 4 — Start the server
 
```bash
# Default: listens on 127.0.0.1:50051
cargo run --release --features rpc --bin rpc_server
 
# Custom address
NEUROSIM_ADDR=0.0.0.0:50051 cargo run --release --features rpc --bin rpc_server
```

### Step 5 — Run Python gRPC tests (server must be running)
 
```bash
pip install grpcio pytest
 
# In a second terminal (server already running):
NEUROSIM_SERVER_RUNNING=1 pytest tests/test_rpc.py tests/test_ffi_rpc.py -v
```

```python
from neurosim_rpc import RemoteSim

with RemoteSim("127.0.0.1:50051", n_neurons=100, seed=42) as sim:
    sim.push(neuron=0, time=0.0, weight=400.0)
    spikes = sim.step(until_time=500.0)
    voltages = sim.voltages()
```

---

## Demo Binaries

```bash
cargo run --release --bin izh_demo            # Izhikevich 6-type showcase
cargo run --release --bin hh_demo             # Hodgkin-Huxley F-I curve + gating vars
cargo run --release --bin adex_demo           # AdEx 5 profiles + F-I curve
cargo run --release --bin stdp_demo           # STDP weight evolution
cargo run --release --bin homeostatic_demo    # Rate regulation over 3 phases
cargo run --release --bin synaptic_scaling_demo # Weight homeostasis
cargo run --release --bin metrics_demo        # Synchrony, bursts, spectrum, avalanches
cargo run --release --bin topology_demo       # ER vs WS vs BA network comparison
cargo run --release --bin bench_harness       # 10k neuron throughput benchmark
```

---

## Test Suite

```bash
cargo test
cargo test --features rpc
```

| Test file | Coverage |
|---|---|
| `tests/lif.rs` | LIF decay, spike, refractory |
| `tests/izhikevich.rs` | Determinism across neuron types |
| `tests/hodgkin_huxley.rs` | F-I curve, gating variables, channel refractory |
| `tests/adex.rs` | All 5 profiles, adaptation, heterogeneity |
| `tests/determinism.rs` | Same seed = same result; MT ≡ ST |
| `tests/simulation.rs` | Monotonic times, finite voltages, checkpoint roundtrip |
| `tests/network.rs` | Topology connectivity, delay bounds |
| `tests/plasticity.rs` | STDP LTP/LTD direction, weight bounds |
| `tests/homeostatic.rs` | Convergence, threshold clamping, disabled mode |
| `tests/synaptic_scaling.rs` | Weight increase/decrease, clamp, ratio preservation |
| `tests/poisson.rs` | Rate accuracy, ISI CV≈1, reproducibility |
| `tests/metrics.rs` | Synchrony, burst detection, power spectrum, avalanches |
| `tests/topology_simulation.rs` | End-to-end simulation per topology |
| `tests/adapters_equivalence.rs` | Determinism across 50 seeds |

---

## Status

### Complete 

| Component | Location | Tests | Demo |
|---|---|---|---|
| LIF neuron | `lif.rs` | `tests/lif.rs` | — |
| Izhikevich (6 types) | `neurons/izhikevich.rs` | `tests/izhikevich.rs` | `izh_demo` |
| Hodgkin-Huxley | `neurons/hh.rs` | `tests/hodgkin_huxley.rs` | `hh_demo` |
| AdEx (5 profiles) | `neurons/adex.rs` | `tests/adex.rs` | `adex_demo` |
| STDP | `plasticity/stdp.rs` | `tests/plasticity.rs` | `stdp_demo` |
| Homeostatic (intrinsic) | `plasticity/homeostatic.rs` | `tests/homeostatic.rs` | `homeostatic_demo` |
| Synaptic scaling | `plasticity/synaptic_scaling.rs` | `tests/synaptic_scaling.rs` | `synaptic_scaling_demo` |
| ER, WS, BA, Layered | `network.rs` / `topology/` | `tests/network.rs` | `topology_demo` |
| Poisson generators | `input/poisson.rs` | `tests/poisson.rs` | — |
| Population metrics | `metrics.rs` | `tests/metrics.rs` | `metrics_demo` |
| Deterministic MT scheduler | `simulation.rs` | `tests/determinism.rs` | — |
| Checkpointing (SHA-256) | `checkpoint.rs` | `tests/simulation.rs` | — |
| C FFI | `lib.rs` | `tests/test_ffi.py` | — |
| gRPC server | `rpc.rs` | `tests/test_rpc.py` | — |
| Python FFI wrapper | `neurosim_ffi.py` | `tests/test_ffi.py` | — |
| Python gRPC wrapper | `neurosim_rpc.py` | `tests/test_rpc.py` | — |

### Planned
- GPU offload via `wgpu` (deterministic)
- NEST / Brian2 import compatibility