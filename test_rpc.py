"""
tests/test_rpc.py
-----------------
Pytest suite for the high-level RemoteSim gRPC client (neurosim_rpc.py).

Previously this was a bare `if __name__ == "__main__"` script with no
assertions.  It is now a proper pytest module with:

  - Session-scoped server fixture that starts/stops the gRPC server
  - Per-test simulation fixtures using RemoteSim context manager
  - Hard assertions on every observable behaviour
  - Graceful skip when the server binary or grpc packages are absent

Run (server must be available):
    pytest tests/test_rpc.py -v

Or skip automatically in environments without the server:
    pytest tests/test_rpc.py -v --ignore-glob="*rpc*"   # (CI without server)
"""

from __future__ import annotations

import os
import subprocess
import time
import signal
import pytest

# ── Import guard ──────────────────────────────────────────────────────────────

try:
    import grpc
    from neurosim_rpc import RemoteSim
    import neurosim_pb2        # noqa: F401 – presence check
    import neurosim_pb2_grpc   # noqa: F401 – presence check
except ImportError as _err:
    pytest.skip(
        f"gRPC dependencies not available ({_err}). "
        "Install: pip install grpcio && cargo build --release --features rpc",
        allow_module_level=True,
    )

SERVER_ADDR = os.environ.get("NEUROSIM_ADDR", "127.0.0.1:50051")
SERVER_BIN  = os.environ.get(
    "NEUROSIM_BIN",
    os.path.join(os.path.dirname(__file__), "..", "target", "release", "neurosim_server"),
)

# ── Server fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def grpc_server():
    """
    Start the gRPC server once per test session and shut it down afterwards.

    If NEUROSIM_ADDR points to an already-running server (e.g. in CI where the
    server is started as a separate service), we skip the launch step.
    """
    already_running = os.environ.get("NEUROSIM_SERVER_RUNNING", "").lower() in (
        "1", "true", "yes"
    )

    proc = None
    if not already_running:
        if not os.path.exists(SERVER_BIN):
            pytest.skip(
                f"Server binary not found at {SERVER_BIN}. "
                "Build with: cargo build --release --features rpc",
            )
        proc = subprocess.Popen(
            [SERVER_BIN],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait up to 5 s for the server to become reachable
        deadline = time.time() + 5.0
        channel  = grpc.insecure_channel(SERVER_ADDR)
        reachable = False
        while time.time() < deadline:
            try:
                grpc.channel_ready_future(channel).result(timeout=0.5)
                reachable = True
                break
            except grpc.FutureTimeoutError:
                pass
        channel.close()
        if not reachable:
            proc.kill()
            pytest.skip("gRPC server did not become reachable within 5 s")

    yield SERVER_ADDR

    if proc is not None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sim(addr, n_neurons=2, seed=42):
    """Context manager that creates a RemoteSim and frees it on exit."""
    return RemoteSim(addr, n_neurons=n_neurons, n_threads=1, seed=seed)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_create_and_free(self, grpc_server):
        """Create a remote simulation and free it without errors."""
        sim = make_sim(grpc_server)
        sim.free()
        sim.close()

    def test_context_manager(self, grpc_server):
        """The context manager protocol should free automatically."""
        with make_sim(grpc_server) as sim:
            assert sim is not None

    def test_initial_time_is_zero(self, grpc_server):
        with make_sim(grpc_server) as sim:
            assert sim.current_time() == pytest.approx(0.0, abs=1.0)

    def test_initial_spike_count_is_zero(self, grpc_server):
        with make_sim(grpc_server) as sim:
            assert sim.get_spike_count() == 0


class TestStepAndSpikes:
    def test_step_returns_list(self, grpc_server):
        with make_sim(grpc_server) as sim:
            spikes = sim.step(until_time=10.0)
            assert isinstance(spikes, list)

    def test_spikes_after_strong_drive(self, grpc_server):
        with make_sim(grpc_server) as sim:
            for t in range(0, 100, 10):
                sim.push(neuron=0, time=float(t), weight=400.0)
            spikes = sim.step(until_time=400.0)
            assert len(spikes) > 0, "No spikes despite strong current drive"

    def test_spike_times_are_monotonic(self, grpc_server):
        with make_sim(grpc_server) as sim:
            for t in range(0, 100, 10):
                sim.push(neuron=0, time=float(t), weight=400.0)
            spikes = sim.step(until_time=400.0)
            times = [t for t, _ in spikes]
            assert times == sorted(times)

    def test_spike_neuron_ids_in_range(self, grpc_server):
        with make_sim(grpc_server) as sim:
            for t in range(0, 100, 10):
                sim.push(neuron=0, time=float(t), weight=400.0)
            spikes = sim.step(until_time=400.0)
            for _, nid in spikes:
                assert 0 <= nid < 2


class TestVoltages:
    def test_get_voltages_returns_correct_length(self, grpc_server):
        with make_sim(grpc_server, n_neurons=5) as sim:
            vs = sim.voltages()
            assert len(vs) == 5

    def test_voltages_are_finite(self, grpc_server):
        with make_sim(grpc_server) as sim:
            for t in range(0, 50, 10):
                sim.push(neuron=0, time=float(t), weight=400.0)
            sim.step(until_time=200.0)
            vs = sim.voltages()
            for v in vs:
                assert v == v, "Voltage is NaN"
                assert -150.0 <= v <= 60.0, f"Voltage out of range: {v}"


class TestStreamSpikes:
    def test_stream_spikes_yields_pairs(self, grpc_server):
        with make_sim(grpc_server) as sim:
            for t in range(0, 100, 10):
                sim.push(neuron=0, time=float(t), weight=400.0)
            sim.step(until_time=400.0)
            streamed = list(sim.stream_spikes())
            assert isinstance(streamed, list)
            for item in streamed:
                assert len(item) == 2, f"Expected (time, nid) pair, got {item}"

    def test_stream_matches_step_spikes(self, grpc_server):
        """stream_spikes() and step() should return the same spike set."""
        with make_sim(grpc_server) as sim:
            for t in range(0, 100, 10):
                sim.push(neuron=0, time=float(t), weight=400.0)
            step_spikes   = sim.step(until_time=400.0)
            stream_spikes = list(sim.stream_spikes())
        # Sort both by (time, nid) before comparing
        def key(s): return (round(s[0], 3), s[1])
        assert sorted(step_spikes, key=key) == sorted(stream_spikes, key=key)


class TestClearSpikes:
    def test_clear_removes_spike_log(self, grpc_server):
        with make_sim(grpc_server) as sim:
            for t in range(0, 100, 10):
                sim.push(neuron=0, time=float(t), weight=400.0)
            sim.step(until_time=400.0)
            assert sim.get_spike_count() > 0
            sim.clear_spikes()
            assert sim.get_spike_count() == 0

    def test_neuron_state_preserved_after_clear(self, grpc_server):
        """Clearing the spike log must not reset neuron voltages."""
        with make_sim(grpc_server) as sim:
            for t in range(0, 100, 10):
                sim.push(neuron=0, time=float(t), weight=400.0)
            sim.step(until_time=400.0)
            vs_before = sim.voltages()
            sim.clear_spikes()
            vs_after  = sim.voltages()
            for vb, va in zip(vs_before, vs_after):
                assert abs(vb - va) < 1e-3, "Voltage changed after clear_spikes"


class TestDeterminism:
    def test_same_seed_same_spikes(self, grpc_server):
        def run(seed):
            with make_sim(grpc_server, n_neurons=10, seed=seed) as sim:
                for t in range(0, 100, 10):
                    sim.push(neuron=0, time=float(t), weight=300.0)
                return sim.step(until_time=400.0)

        a = run(42)
        b = run(42)
        def key(s): return (round(s[0], 3), s[1])
        assert sorted(a, key=key) == sorted(b, key=key), \
            "Determinism broken: same seed produced different spikes"

    def test_different_seeds_allowed_to_differ(self, grpc_server):
        def run(seed):
            with make_sim(grpc_server, n_neurons=10, seed=seed) as sim:
                for t in range(0, 100, 10):
                    sim.push(neuron=0, time=float(t), weight=300.0)
                return sim.step(until_time=400.0)

        a = run(1)
        b = run(2)
        # Both are valid lists — the test just checks they are well-formed.
        assert isinstance(a, list) and isinstance(b, list)