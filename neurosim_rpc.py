# neurosim_rpc.py
import grpc
from neurosim_pb2 import (
    SimConfig,
    InputEvent,
    StepRequest,
    Handle
)
from neurosim_pb2_grpc import NeuroSimStub


class RemoteSim:

    def __init__(self, address, n_neurons, n_threads=1, seed=42):
        self.channel = grpc.insecure_channel(address)
        self.stub = NeuroSimStub(self.channel)

        # NOTE: Use Create() not create()
        res = self.stub.Create(SimConfig(
            n_neurons=n_neurons,
            n_threads=n_threads,
            seed=seed
        ))
        self.sim_id = res.id

    def push(self, neuron, time, weight):
        # Push()
        self.stub.Push(InputEvent(
            sim_id=self.sim_id,
            neuron=neuron,
            time=time,
            weight=weight
        ))

    def step(self, until_time):
        # Step()
        reply = self.stub.Step(StepRequest(
            sim_id=self.sim_id,
            until_time=until_time
        ))
        return [(s.time, s.neuron) for s in reply.spikes]

    def voltages(self):
        # GetVoltages()
        reply = self.stub.GetVoltages(Handle(id=self.sim_id))
        return list(reply.volts)

    def free(self):
        # Free()
        self.stub.Free(Handle(id=self.sim_id))

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.free()
        finally:
            self.close()

    def stream_spikes(self):
        for msg in self.stub.StreamSpikes(Handle(id=self.sim_id)):
            yield (msg.time, msg.neuron)
