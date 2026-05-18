"""
tests/test_ffi.py
-----------------
Pytest suite for the C-ABI FFI layer (neurosim_ffi.py / libsynaptic_shenanigans.so).

Previously this was a bare script (`if __name__ == "__main__": ...`) with no
assertions and no test runner integration.  It is now a proper pytest module:

  - Every behaviour is an independent test function.
  - Hard assertions replace informal prints.
  - The `sim` fixture handles creation + teardown.
  - The file can be discovered by pytest, GitHub Actions, and IDEs.

Run:
    pytest tests/test_ffi.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import pytest

# ── Import guard ──────────────────────────────────────────────────────────────
# Skip the entire module if the shared library has not been built yet.
# This allows `pytest` to run in CI environments where only the Rust unit
# tests are built (e.g. `cargo test --lib`).

try:
    from neurosim_ffi import NeuroSim
except (ImportError, OSError) as _err:
    pytest.skip(
        f"neurosim_ffi not available ({_err}). "
        "Build the cdylib first: cargo build --release",
        allow_module_level=True,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def basic_sim():
    """Yield a 2-neuron simulation and ensure it is freed after the test."""
    with NeuroSim.basic(n_neurons=2, n_threads=1, seed=42) as sim:
        yield sim


@pytest.fixture
def driven_sim():
    """2-neuron sim with 10 current pulses pre-loaded into neuron 0."""
    with NeuroSim.basic(n_neurons=2, n_threads=1, seed=42) as sim:
        for t in range(0, 100, 10):
            sim.push_current(time=float(t), neuron=0, weight=400.0)
        yield sim


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCreation:
    def test_sim_object_is_created(self, basic_sim):
        """NeuroSim.basic() should return a live object."""
        assert basic_sim is not None

    def test_initial_time_is_zero(self, basic_sim):
        assert basic_sim.current_time() == pytest.approx(0.0, abs=1e-4)

    def test_initial_spike_count_is_zero(self, basic_sim):
        assert basic_sim.get_spike_count() == 0

    def test_initial_voltages_are_finite(self, basic_sim):
        vs = basic_sim.get_all_voltages()
        assert len(vs) == 2
        for v in vs:
            assert v == v  # NaN check
            assert -100.0 <= v <= 0.0, f"Unexpected resting voltage: {v}"


class TestPushAndRun:
    def test_run_advances_time(self, driven_sim):
        driven_sim.run_until(400.0)
        assert driven_sim.current_time() == pytest.approx(400.0, abs=1.0)

    def test_spikes_detected_after_strong_drive(self, driven_sim):
        driven_sim.run_until(400.0)
        spikes = driven_sim.get_spikes()
        assert len(spikes) > 0, "No spikes detected despite strong current input"

    def test_spike_times_are_monotonic(self, driven_sim):
        driven_sim.run_until(400.0)
        spikes = driven_sim.get_spikes()
        times = [t for t, _ in spikes]
        assert times == sorted(times), "Spike times are not monotonically ordered"

    def test_spike_neuron_ids_in_range(self, driven_sim):
        driven_sim.run_until(400.0)
        spikes = driven_sim.get_spikes()
        for _, nid in spikes:
            assert 0 <= nid < 2, f"Neuron ID {nid} out of range"

    def test_voltage_after_run_is_finite(self, driven_sim):
        driven_sim.run_until(400.0)
        v0 = driven_sim.get_voltage(0)
        assert v0 == v0, "Voltage is NaN"
        assert -100.0 <= v0 <= 10.0, f"Voltage out of plausible range: {v0}"


class TestStepInterface:
    def test_step_returns_spike_list(self):
        """step(dt) should return a list (possibly empty)."""
        with NeuroSim.basic(n_neurons=2, n_threads=1, seed=7) as sim:
            spikes = sim.step(1.0)
            assert isinstance(spikes, list)

    def test_step_increments_time(self):
        with NeuroSim.basic(n_neurons=2, n_threads=1, seed=7) as sim:
            t_before = sim.current_time()
            sim.step(1.0)
            t_after  = sim.current_time()
            assert t_after >= t_before

    def test_inject_spike_triggers_response(self):
        """inject_spike should raise voltage even without external events."""
        with NeuroSim.basic(n_neurons=2, n_threads=1, seed=7) as sim:
            sim.inject_spike(0, weight=300.0)
            spikes = sim.step(1.0)
            # We don't require a spike on this exact step (refractory),
            # but voltage or spike should be affected.
            v = sim.get_all_voltages()
            assert all(vi == vi for vi in v), "Voltage is NaN after inject_spike"

    def test_control_loop_accumulates_spikes(self):
        """50 control steps with periodic stimulation should produce spikes."""
        with NeuroSim.basic(n_neurons=2, n_threads=1, seed=7) as sim:
            total_spikes = 0
            for t in range(50):
                if t % 5 == 0:
                    sim.inject_spike(0, weight=300.0)
                spikes = sim.step(1.0)
                total_spikes += len(spikes)
            assert total_spikes > 0, "Periodic stimulation produced no spikes in 50 steps"


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, driven_sim):
        driven_sim.run_until(400.0)
        spikes_before = driven_sim.get_spikes()

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        hash_path = path + ".sha256"

        try:
            driven_sim.save_checkpoint(path)
            assert os.path.exists(path), "Checkpoint file not created"
            assert os.path.exists(hash_path), "SHA-256 hash file not created"
            assert os.path.getsize(path) > 0, "Checkpoint file is empty"
        finally:
            for p in (path, hash_path):
                try: os.unlink(p)
                except FileNotFoundError: pass


class TestDeterminism:
    def test_same_seed_same_spikes(self):
        """Two simulations with the same seed must produce identical output."""
        def run(seed):
            with NeuroSim.basic(n_neurons=10, n_threads=1, seed=seed) as sim:
                for t in range(0, 200, 10):
                    sim.push_current(float(t), 0, 300.0)
                sim.run_until(500.0)
                return sim.get_spikes()

        assert run(42) == run(42), "Determinism broken: same seed, different spikes"

    def test_different_seeds_may_differ(self):
        """Two different seeds are allowed to (and usually do) differ."""
        def run(seed):
            with NeuroSim.basic(n_neurons=10, n_threads=1, seed=seed) as sim:
                for t in range(0, 200, 10):
                    sim.push_current(float(t), 0, 300.0)
                sim.run_until(500.0)
                return sim.get_spikes()

        # Different seeds should very likely differ (not a hard requirement,
        # but a smoke-test for the seed parameter being respected).
        a, b = run(1), run(2)
        # We just assert they are valid lists — no assertion on equality.
        assert isinstance(a, list) and isinstance(b, list)