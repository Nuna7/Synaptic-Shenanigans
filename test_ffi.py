# test_ffi.py
from neurosim_ffi import NeuroSim

if __name__ == "__main__":
    with NeuroSim.basic(n_neurons=2, n_threads=1, seed=42) as sim:
        # external pulses to neuron 0
        for t in range(0, 100, 10):
            sim.push_current(time=float(t), neuron=0, weight=400.0)

        sim.run_until(400.0)

        spikes = sim.get_spikes()
        print("spikes:", spikes)

        v0 = sim.get_voltage(0)
        print("V0 =", v0)

        print("Hello")

        sim.save_checkpoint("checkpoint.bin")

    with NeuroSim.basic(2, 1, seed=7) as sim:

        for t in range(50):  # 50 control steps
            if t % 5 == 0:
                sim.inject_spike(0, weight=300.0)  # Stimulate neuron 0

            spikes = sim.step(1.0)
            v = sim.get_all_voltages()

            print(f"t={t}, v={v}, spikes={spikes}")

