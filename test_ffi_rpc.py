"""
tests/test_ffi_rpc.py
---------------------
End-to-end smoke tests for the raw gRPC stubs (neurosim_pb2 / neurosim_pb2_grpc).

Previously this was a bare script that printed results with no assertions.
It is now a pytest module that:

  - Uses the same session-scoped server fixture pattern as test_rpc.py
  - Asserts on every observable result (spike count, voltage range, etc.)
  - Skips cleanly when the server or protobuf packages are absent

Run:
    pytest tests/test_ffi_rpc.py -v

Environment variables:
  NEUROSIM_ADDR            – server address (default 127.0.0.1:50051)
  NEUROSIM_SERVER_RUNNING  – set to 1 to skip server launch
  NEUROSIM_BIN             – path to the server binary
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
    import neurosim_pb2
    import neurosim_pb2_grpc
except ImportError as _err:
    pytest.skip(
        f"gRPC protobuf stubs not available ({_err}). "
        "Build with: cargo build --release --features rpc && "
        "pip install grpcio grpcio-tools",
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
    already_running = os.environ.get("NEUROSIM_SERVER_RUNNING", "").lower() in (
        "1", "true", "yes",
    )
    proc = None

    if not already_running:
        if not os.path.exists(SERVER_BIN):
            pytest.skip(f"Server binary not found at {SERVER_BIN}.")
        proc = subprocess.Popen(
            [SERVER_BIN],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline  = time.time() + 5.0
        channel   = grpc.insecure_channel(SERVER_ADDR)
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

def make_stub(addr):
    channel = grpc.insecure_channel(addr)
    stub    = neurosim_pb2_grpc.NeuroSimStub(channel)
    return channel, stub


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCreateAndFree:
    def test_create_returns_positive_id(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))
        assert h.id > 0, f"Expected positive handle id, got {h.id}"
        stub.Free(neurosim_pb2.Handle(id=h.id))
        channel.close()

    def test_double_create_returns_distinct_ids(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h1 = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=1))
        h2 = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=2))
        assert h1.id != h2.id, "Two Create calls returned the same handle"
        stub.Free(neurosim_pb2.Handle(id=h1.id))
        stub.Free(neurosim_pb2.Handle(id=h2.id))
        channel.close()


class TestPushAndStep:
    def test_push_and_step_returns_spikes(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))

        for t in range(0, 100, 10):
            stub.Push(neurosim_pb2.InputEvent(
                sim_id=h.id, time=float(t), neuron=0, weight=400.0,
            ))

        reply = stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=400.0))
        assert len(reply.spikes) > 0, "No spikes despite strong current drive"

        stub.Free(neurosim_pb2.Handle(id=h.id))
        channel.close()

    def test_spike_times_monotonic(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))

        for t in range(0, 100, 10):
            stub.Push(neurosim_pb2.InputEvent(
                sim_id=h.id, time=float(t), neuron=0, weight=400.0,
            ))

        reply = stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=400.0))
        times = [s.time for s in reply.spikes]
        assert times == sorted(times), "Spike times are not monotonically ordered"

        stub.Free(neurosim_pb2.Handle(id=h.id))
        channel.close()

    def test_spike_neuron_ids_in_range(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))

        for t in range(0, 100, 10):
            stub.Push(neurosim_pb2.InputEvent(
                sim_id=h.id, time=float(t), neuron=0, weight=400.0,
            ))

        reply = stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=400.0))
        for spike in reply.spikes:
            assert 0 <= spike.neuron < 2, f"Spike neuron {spike.neuron} out of range"

        stub.Free(neurosim_pb2.Handle(id=h.id))
        channel.close()


class TestGetVoltages:
    def test_get_voltages_returns_all_neurons(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h = stub.Create(neurosim_pb2.SimConfig(n_neurons=5, n_threads=1, seed=42))

        stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=10.0))
        reply = stub.GetVoltages(neurosim_pb2.Handle(id=h.id))

        assert len(reply.volts) == 5, f"Expected 5 voltages, got {len(reply.volts)}"

        stub.Free(neurosim_pb2.Handle(id=h.id))
        channel.close()

    def test_voltages_are_finite_and_in_range(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))

        for t in range(0, 100, 10):
            stub.Push(neurosim_pb2.InputEvent(
                sim_id=h.id, time=float(t), neuron=0, weight=400.0,
            ))
        stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=400.0))
        reply = stub.GetVoltages(neurosim_pb2.Handle(id=h.id))

        for v in reply.volts:
            assert v == v, "Voltage is NaN"
            assert -150.0 <= v <= 60.0, f"Voltage {v} out of plausible range"

        stub.Free(neurosim_pb2.Handle(id=h.id))
        channel.close()


class TestClearSpikes:
    def test_clear_spikes_empties_log(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))

        for t in range(0, 100, 10):
            stub.Push(neurosim_pb2.InputEvent(
                sim_id=h.id, time=float(t), neuron=0, weight=400.0,
            ))
        reply = stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=400.0))
        assert len(reply.spikes) > 0, "Precondition: need spikes to clear"

        stub.ClearSpikes(neurosim_pb2.Handle(id=h.id))

        # After clearing, streaming should yield nothing
        stream = list(stub.StreamSpikes(neurosim_pb2.Handle(id=h.id)))
        assert len(stream) == 0, f"Expected empty spike log after clear, got {len(stream)}"

        stub.Free(neurosim_pb2.Handle(id=h.id))
        channel.close()


class TestStreamSpikes:
    def test_stream_yields_same_as_step(self, grpc_server):
        channel, stub = make_stub(grpc_server)
        h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))

        for t in range(0, 100, 10):
            stub.Push(neurosim_pb2.InputEvent(
                sim_id=h.id, time=float(t), neuron=0, weight=400.0,
            ))

        reply  = stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=400.0))
        step_spikes   = sorted((s.time, s.neuron) for s in reply.spikes)
        stream_spikes = sorted((s.time, s.neuron) for s in
                               stub.StreamSpikes(neurosim_pb2.Handle(id=h.id)))

        assert step_spikes == stream_spikes, \
            "StreamSpikes and Step returned different spike logs"

        stub.Free(neurosim_pb2.Handle(id=h.id))
        channel.close()


class TestDeterminism:
    def test_same_seed_same_spikes(self, grpc_server):
        def run():
            channel, stub = make_stub(grpc_server)
            h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))
            for t in range(0, 100, 10):
                stub.Push(neurosim_pb2.InputEvent(
                    sim_id=h.id, time=float(t), neuron=0, weight=400.0,
                ))
            reply = stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=400.0))
            spikes = sorted((s.time, s.neuron) for s in reply.spikes)
            stub.Free(neurosim_pb2.Handle(id=h.id))
            channel.close()
            return spikes

        assert run() == run(), "Determinism broken: same seed, different spikes"