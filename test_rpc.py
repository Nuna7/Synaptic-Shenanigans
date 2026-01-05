from neurosim_rpc import RemoteSim

if __name__ == "__main__":
    with RemoteSim("127.0.0.1:50051", n_neurons=2, n_threads=1, seed=42) as sim:

        for t in range(0, 100, 10):
            sim.push(neuron=0, time=float(t), weight=400.0)

        spikes = sim.step(400.0)
        print("Spikes:", spikes)

        print("Voltages:", sim.voltages())

        for s in sim.stream_spikes():
            print("Spike:", s)