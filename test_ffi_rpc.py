# test_ffi_rpc.py
#
# End-to-end smoke test for the gRPC interface.
# Requires the server to be running:  cargo run --release --features rpc
#
import grpc
import neurosim_pb2
import neurosim_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub    = neurosim_pb2_grpc.NeuroSimStub(channel)

# Create a 2-neuron simulation.
h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2, n_threads=1, seed=42))

# Inject ten pulses into neuron 0, spaced 10 ms apart.
for t in range(0, 100, 10):
    stub.Push(neurosim_pb2.InputEvent(
        sim_id=h.id, time=t, neuron=0, weight=400,
    ))

# Advance to 400 ms.  StepReply only contains spikes.
reply = stub.Step(neurosim_pb2.StepRequest(sim_id=h.id, until_time=400))
print("spikes:", reply.spikes)

# Voltages are obtained via a separate GetVoltages call.
v_reply = stub.GetVoltages(neurosim_pb2.Handle(id=h.id))
print("voltages:", list(v_reply.volts))

# Clear the spike log between epochs.
stub.ClearSpikes(neurosim_pb2.Handle(id=h.id))
print("spike log cleared")

# Free the simulation.
stub.Free(neurosim_pb2.Handle(id=h.id))
channel.close()