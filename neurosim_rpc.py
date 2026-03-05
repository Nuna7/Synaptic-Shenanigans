import grpc
from neurosim_pb2 import (
    SimConfig,
    InputEvent,
    StepRequest,
    Handle,
)
from neurosim_pb2_grpc import NeuroSimStub


class RemoteSim:
    """
    High-level gRPC client that mirrors the NeuroSim FFI interface.

    Usage:
        with RemoteSim("localhost:50051", n_neurons=100, seed=42) as sim:
            sim.push(neuron=0, time=0.0, weight=400.0)
            spikes   = sim.step(until_time=500.0)
            voltages = sim.voltages()
            sim.clear_spikes()
    """

    def __init__(self, address: str, n_neurons: int, n_threads: int = 1, seed: int = 42):
        self.channel = grpc.insecure_channel(address)
        self.stub    = NeuroSimStub(self.channel)

        res = self.stub.Create(SimConfig(
            n_neurons=n_neurons,
            n_threads=n_threads,
            seed=seed,
        ))
        self.sim_id = res.id

    # ---- inputs -------------------------------------------------------

    def push(self, neuron: int, time: float, weight: float) -> None:
        """Inject a current pulse into *neuron* at *time* ms."""
        self.stub.Push(InputEvent(
            sim_id=self.sim_id,
            neuron=neuron,
            time=time,
            weight=weight,
        ))

    # ---- stepping -----------------------------------------------------

    def step(self, until_time: float):
        """
        Advance the simulation to *until_time* ms.

        Returns the **full** spike log as ``[(time_ms, neuron_id), ...]``.
        Call :meth:`clear_spikes` before each step if you only want new spikes.
        """
        reply = self.stub.Step(StepRequest(
            sim_id=self.sim_id,
            until_time=until_time,
        ))
        return [(s.time, s.neuron) for s in reply.spikes]

    # ---- queries ------------------------------------------------------

    def voltages(self):
        """Return a list of membrane potentials (mV) for all neurons."""
        reply = self.stub.GetVoltages(Handle(id=self.sim_id))
        return list(reply.volts)

    def stream_spikes(self):
        """Yield all recorded spikes as ``(time_ms, neuron_id)`` pairs."""
        for msg in self.stub.StreamSpikes(Handle(id=self.sim_id)):
            yield (msg.time, msg.neuron)

    # ---- spike log management -----------------------------------------

    def clear_spikes(self) -> None:
        """Clear the spike log on the server.  Neuron states are preserved."""
        self.stub.ClearSpikes(Handle(id=self.sim_id))

    # ---- lifecycle ----------------------------------------------------

    def free(self) -> None:
        """Release the simulation on the server side."""
        self.stub.Free(Handle(id=self.sim_id))

    def close(self) -> None:
        """Close the gRPC channel."""
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.free()
        finally:
            self.close()

    def get_spike_count(self) -> int:
        reply = self.stub.GetSpikeCount(Handle(id=self.sim_id))
        return reply.count

    def current_time(self) -> float:
        reply = self.stub.GetTime(Handle(id=self.sim_id))
        return reply.time

    def save_checkpoint(self, path: str) -> None:
        from neurosim_pb2 import CheckpointRequest
        self.stub.SaveCheckpoint(CheckpointRequest(sim_id=self.sim_id, path=path))