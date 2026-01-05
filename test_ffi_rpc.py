import grpc
import neurosim_pb2
import neurosim_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = neurosim_pb2_grpc.NeuroSimStub(channel)

h = stub.Create(neurosim_pb2.SimConfig(n_neurons=2,n_threads=1,seed=42))

for t in range(0,100,10):
    stub.Push(neurosim_pb2.InputEvent(sim_id=h.id,time=t,neuron=0,weight=400))

reply = stub.Step(neurosim_pb2.StepRequest(sim_id=h.id,until_time=400))
print(reply.spikes, reply.voltages)
