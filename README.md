# Synaptic-Shenanigans

**A deterministic, reproducible, multi-threaded spiking neural network simulation engine in Rust.**

Synaptic-Shenanigans is a research-grade event-driven simulator designed for **correctness first**:
- bit-for-bit determinism
- reproducible multithreading
- checkpointable state
- safe concurrency
- clean FFI + RPC boundaries

This project is intended for **computational neuroscience**, **neuromorphic systems**, and **closed-loop experiments** where *reproducibility and determinism matter with requirement for high speed simulation*.

---

## Key Features

- **Deterministic execution**
  - Same seed → identical spike trains
  - Deterministic multithreading = single-thread equivalence
- **Event-driven simulation**
  - Sparse synaptic activity scales efficiently
- **Reproducible checkpoints**
  - Save → load → resume gives identical results
- **Thread-safe neuron updates**
  - `Arc + AtomicCell`, no data races
- **Multiple scheduler modes**
  - Single-threaded
  - Deterministic multithreaded
  - (Optional) performance mode
- **FFI-ready & RPC-ready**
  - C / Python bindings
  - gRPC streaming of spikes
- **Extensive test coverage**
  - Unit, integration, determinism, serialization

---

## Installation

### Requirements
- Rust ≥ 1.72
- Linux / macOS (profiling tools are Linux-only)

```bash
git clone https://github.com/yourname/Synaptic-Shenanigans.git
cd Synaptic-Shenanigans
cargo build --release
```

# Quick Start Example

## Minimal deterministic simulation

```rust
use synaptic_shenanigans::lif::LifNeuron;
use synaptic_shenanigans::synapse::Synapse;
use synaptic_shenanigans::simulation::{Simulation, SchedulerMode};

fn main() {
    // Create 100 LIF neurons
    let neurons = LifNeuron::new(
        100,
        -65.0, // v_rest
        -50.0, // v_thresh
        20.0,  // tau_m
        1.0,   // r_m
        1.0,   // dt
        5.0,   // refractory period
    );

    // Empty synapse set (external inputs only)
    let synapses = Synapse::new();

    // Deterministic simulation with seed = 42
    let mut sim = Simulation::new_with_seed(
        neurons,
        synapses,
        1.0,   // dt
        42,    // seed
        1      // num_threads
    );

    // Push an external current pulse
    sim.push_event(10.0, 0, 100.0, 0, 0.0);

    // Run simulation
    sim.run_auto(100.0);

    // Inspect spikes
    for (t, neuron) in sim.spike_log {
        println!("Neuron {} spiked at t={}", neuron, t);
    }
}

```

## Reproducibility Guarantees

### Determinism Contract

Synaptic-Shenanigans guarantees:

| Scenario | Guarantee |
| --- | --- |
| Same seed, same config | Identical spike logs |
| Single vs multithreaded | Identical spike logs |
| Save → load → resume | Identical spike logs |
| Different machines | Identical results (within IEEE float rules) |

This is enforced by:
- explicit event sequencing (`seq`)
- deterministic heap ordering
- static thread partitioning
- atomic state updates
- sorted merge barriers

## Reproducibility Example

Replay equality

```rust
use synaptic_shenanigans::simulation::replay_equal;

assert!(
    replay_equal(build_sim, 500.0, 42),
    "Simulation is not reproducible!"
);
```

## Checkpointing & Resume

Save simulation state

```rust
sim.save_state("checkpoint.bin", "checkpoint.bin.sha256").unwrap();
```

Load and resume

```rust
let mut sim2 = Simulation::load_state("checkpoint.bin", 42, 1).unwrap();
sim2.run_auto(500.0);
```

This will produce identical results to a continuous run.

## Scheduler Modes

```rust
use synaptic_shenanigans::simulation::SchedulerMode;

sim.scheduler_mode = SchedulerMode::SingleThreaded;
// or
sim.scheduler_mode = SchedulerMode::Deterministic { n_threads: 4 };

```

| Mode             | Deterministic | Purpose
| --- | --- | --- |
| SingleThreaded   | ✅             | Reference behavior
| Deterministic MT | ✅ | Parallel + reproducible
| Performance MT   | ❌ | Benchmarking only

## Neuron Model

Currently implemented:

- Leaky Integrate-and-Fire (LIF)
    - Euler integration
    - Absolute refractory period
    - Current-based & conductance-based synapses

The design intentionally separates:
- neuron dynamics
- synaptic propagation
- scheduling / execution

## Test Suite Overview

Run all tests:
```rust
cargo test
```

### What is tested?
| Test category | What it proves |
| --- | --- |
| Neuron dynamics |	Correct LIF behavior |
| Determinism |	Same seed = same result |
| Multithreading | MT ≡ single-thread |
| Serialization	Checkpoint | correctness |
| Numerical stability |	No NaNs / infs |
| Ordering | Spike times monotonic |

This is engine-level correctness, not biological validation

## Thread Safety Model

- Neuron state: AtomicCell
- Shared structures: Arc
- No interior mutability without atomics
- No global mutable state
- No implicit RNG usage

This ensures:
- no data races
- no heisenbugs
- no scheduler-dependent behavior

## RPC / FFI Support

The engine exposes:
- C ABI for Python / C++ bindings
- gRPC server for remote simulation & streaming spikes

Example use cases:
- closed-loop robotics
- hardware-in-the-loop experiments
- distributed neuromorphic control

---

## Benchmarking & Flamegraph Profiling

This project includes **both microbenchmarks** (Criterion) and **full-system benchmark harnesses** designed for realistic workloads.

---

### 1. Microbenchmarks (Criterion)

Located in:

`benches/neuron_step.rs`


Example benchmark:
- 10,000 LIF neurons
- single integration step
- measures per-step cost

Run:

```bash
cargo bench
```


Example benchmark:
- 10,000 LIF neurons
- single integration step
- measures per-step cost

Run:
```bash
cargo bench
```

Results are printed to stdout and stored under:

`target/criterion/`

Use this for:
- regression detection
- comparing neuron model changes
- verifying performance stability

### 2. Full-System Benchmark Harness

Located at:

`src/bin/bench_harness.rs`

This benchmark simulates:
- 10,000 neurons
- 100,000 synapses
- event-driven execution
- deterministic scheduling

Run:
```bash
cargo run --release --bin bench_harness
```

Output:
- throughput (simulated ms / wall second)
- latency percentiles (p50, p99, max)
- CSV written to:
    ```bash
    bench/results/harness.csv
    ```

This benchmark is deterministic and suitable for:
- hardware comparisons
- CI performance smoke tests
- reproducible performance reporting

## Performance Notes
- Event-driven → sparse activity scales well
- Deterministic MT trades speed for correctness
- Performance mode available but explicitly non-deterministic

This project prioritizes scientific correctness over peak throughput.

### 3. Flamegraph Profiling (Linux-only)
    ⚠️ Flamegraphs require Linux and perf.
    macOS users should use a Linux VM or remote machine

Install tools
```bash
sudo apt install linux-tools-common linux-tools-generic
cargo install flamegraph
```

Run flamegraph
```bash
cargo flamegraph --release --bin bench_harness
```

This produces:
    
    flamegraph.svg


Open it in a browser:

    firefox flamegraph.svg

### What to Look For in Flamegraphs

Typical hotspots:
- Simulation::run_until
- synaptic event dispatch
- neuron integration (step_range)
- event heap operations

This helps identify:
- memory bottlenecks
- excessive cloning
- ordering overhead
- cache misses

## Python Usage Examples

Synaptic-Shenanigans supports Python via:
- C FFI (ctypes)
- gRPC (recommended for distributed setups)

### 1. Python via FFI (ctypes)

File:

    neurosim_ffi.py

Example usage:
```python
from neurosim_ffi import NeuroSim

with NeuroSim.basic(n_neurons=2, n_threads=1, seed=42) as sim:
    # Inject external currents
    for t in range(0, 100, 10):
        sim.push_current(time=float(t), neuron=0, weight=400.0)

    # Run simulation
    sim.run_until(400.0)

    # Read spikes
    spikes = sim.get_spikes()
    print("Spikes:", spikes)

    # Read voltage
    v0 = sim.get_voltage(0)
    print("Neuron 0 voltage:", v0)

    # Save checkpoint
    sim.save_checkpoint("checkpoint.bin")
```

- ✔ Deterministic
- ✔ Zero-copy FFI
- ✔ Suitable for tight control loops

### 2. Python Closed-Loop Control Example
```bash
from neurosim_ffi import NeuroSim

with NeuroSim.basic(2, 1, seed=7) as sim:
    for step in range(50):
        if step % 5 == 0:
            sim.inject_spike(0, weight=300.0)

        spikes = sim.step(1.0)
        voltages = sim.get_all_voltages()

        print(f"t={step}, V={voltages}, spikes={spikes}")
```

This pattern is ideal for:
- robotics
- neuromorphic controllers
- adaptive stimulation

### 3. Python via gRPC (Remote Simulation)

Start the server (Rust):

    cargo run --release --features rpc


Python client:

```python
from neurosim_rpc import RemoteSim

with RemoteSim("127.0.0.1:50051", n_neurons=2, n_threads=1, seed=42) as sim:
    for t in range(0, 100, 10):
        sim.push(neuron=0, time=float(t), weight=400.0)

    spikes = sim.step(400.0)
    print("Spikes:", spikes)

    print("Voltages:", sim.voltages())

    for s in sim.stream_spikes():
        print("Spike:", s)
```

- ✔ Network-transparent
- ✔ Streaming spikes
- ✔ Suitable for distributed systems

### When to Use Which Python Interface
| Interface |	Best for |
| --- | --- |
| FFI |	Low latency, local control |
| gRPC | Distributed, remote control |
| Checkpoint | Offline replay, audits |

## Status
- Correctness-complete ✔
- Determinism-proven ✔
- Serialization-safe ✔

Next milestones:
- additional neuron models
- synaptic plasticity
- GPU-offload (still deterministic)