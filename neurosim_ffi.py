"""
High-level Python wrapper for the synaptic-shenanigans FFI.

Assumes the Rust library exports:

    SimHandle (opaque)
    sim_create_basic(n_neurons: c_int, n_threads: c_int, seed: c_ulong) -> *mut SimHandle
    sim_free(*mut SimHandle)
    sim_push_current(handle, time: c_float, target: c_int, weight: c_float) -> c_int
    sim_step_and_query(handle, end_time: c_float) -> c_int
    sim_spike_count(handle) -> c_int
    sim_get_spikes(handle, out: *mut FfiSpike, max_len: c_int) -> c_int
    sim_get_voltage(handle, neuron: c_int, out_v: *mut c_float) -> c_int
    sim_save_checkpoint(handle, path: *const c_char) -> c_int
"""

import ctypes as ct
import os
import sys
from typing import List, Tuple, Optional

# -------------------------------------------------------------------
# Dynamic library loading
# -------------------------------------------------------------------

def _default_lib_name() -> str:
    if sys.platform.startswith("linux"):
        return "libsynaptic_shenanigans.so"
    elif sys.platform == "darwin":
        return "libsynaptic_shenanigans.dylib"
    elif os.name == "nt":
        return "synaptic_shenanigans.dll"
    else:
        # fallback; user can override via NEUROSIM_LIB
        return "libsynaptic_shenanigans.so"


def _load_lib() -> ct.CDLL:
    # Allow override via env var
    lib_path = os.environ.get("NEUROSIM_LIB")
    if lib_path is None:
        # default to target/release/...
        base = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base, "target", "release", _default_lib_name())
        lib_path = candidate
    return ct.CDLL(lib_path)


_lib = _load_lib()

# -------------------------------------------------------------------
# C types / ABI
# -------------------------------------------------------------------

class SimHandle(ct.Structure):
    _fields_ = [("sim", ct.c_void_p)]


class FfiSpike(ct.Structure):
    _fields_ = [
        ("time", ct.c_float),
        ("neuron_id", ct.c_int),
    ]


# Return types
_lib.sim_create_basic.restype = ct.POINTER(SimHandle)

# Argument types
_lib.sim_create_basic.argtypes = [ct.c_int, ct.c_int, ct.c_ulong]
_lib.sim_free.argtypes = [ct.POINTER(SimHandle)]

_lib.sim_push_current.argtypes = [
    ct.POINTER(SimHandle),
    ct.c_float,
    ct.c_int,
    ct.c_float,
]
_lib.sim_push_current.restype = ct.c_int

_lib.sim_step_and_query.argtypes = [
    ct.POINTER(SimHandle),
    ct.c_float,
]
_lib.sim_step_and_query.restype = ct.c_int

_lib.sim_spike_count.argtypes = [ct.POINTER(SimHandle)]
_lib.sim_spike_count.restype = ct.c_int

_lib.sim_get_spikes.argtypes = [
    ct.POINTER(SimHandle),
    ct.POINTER(FfiSpike),
    ct.c_int,
]
_lib.sim_get_spikes.restype = ct.c_int

_lib.sim_get_voltage.argtypes = [
    ct.POINTER(SimHandle),
    ct.c_int,
    ct.POINTER(ct.c_float),
]
_lib.sim_get_voltage.restype = ct.c_int

_lib.sim_save_checkpoint.argtypes = [
    ct.POINTER(SimHandle),
    ct.c_char_p,
]
_lib.sim_save_checkpoint.restype = ct.c_int


# -------------------------------------------------------------------
# High-level Python wrapper
# -------------------------------------------------------------------

class NeuroSim:
    """
    High-level object-oriented wrapper around the C ABI.

    Usage:
        sim = NeuroSim.basic(n_neurons=2, n_threads=1, seed=42)
        sim.push_current(time=0.0, neuron=0, weight=400.0)
        sim.run_until(400.0)
        spikes = sim.get_spikes()
        v0 = sim.get_voltage(0)
        sim.close()
    """

    def __init__(self, handle: ct.POINTER(SimHandle), n_neurons: int, n_threads: int, seed: int):
        self._handle = handle
        self.n_neurons = n_neurons
        self.n_threads = n_threads
        self.seed = seed
        self._closed = False

    # ---- constructors ------------------------------------------------

    @classmethod
    def basic(cls, n_neurons: int, n_threads: int = 1, seed: int = 42) -> "NeuroSim":
        h = _lib.sim_create_basic(
            ct.c_int(n_neurons),
            ct.c_int(n_threads),
            ct.c_ulong(seed),
        )
        if not h:
            raise RuntimeError("sim_create_basic returned NULL")
        return cls(h, n_neurons=n_neurons, n_threads=n_threads, seed=seed)

    # ---- low-level lifetime management -------------------------------

    def close(self) -> None:
        if not self._closed and self._handle:
            _lib.sim_free(self._handle)
            self._closed = True
            self._handle = None

    def __del__(self) -> None:
        # best-effort; avoid raising in destructor
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "NeuroSim":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- simulation control ------------------------------------------

    def push_current(self, time: float, neuron: int, weight: float) -> None:
        """Inject a current-based event: I(t) into neuron."""
        if self._closed:
            raise RuntimeError("Simulation is closed")
        ret = _lib.sim_push_current(
            self._handle,
            ct.c_float(time),
            ct.c_int(neuron),
            ct.c_float(weight),
        )
        if ret != 0:
            raise RuntimeError(f"sim_push_current failed with code {ret}")

    def run_until(self, end_time: float) -> None:
        """Advance simulation until (inclusive) `end_time`."""
        if self._closed:
            raise RuntimeError("Simulation is closed")
        ret = _lib.sim_step_and_query(self._handle, ct.c_float(end_time))
        if ret != 0:
            raise RuntimeError(f"sim_step_and_query failed with code {ret}")

    # ---- observables -------------------------------------------------

    def spike_count(self) -> int:
        if self._closed:
            raise RuntimeError("Simulation is closed")
        return int(_lib.sim_spike_count(self._handle))

    def get_spikes(self) -> List[Tuple[float, int]]:
        """
        Snapshot of current spike log as list of (time, neuron_id).

        (We allocate a buffer of size spike_count, call sim_get_spikes,
         then slice to the actual number copied.)
        """
        n = self.spike_count()
        if n <= 0:
            return []

        buf = (FfiSpike * n)()
        copied = _lib.sim_get_spikes(self._handle, buf, ct.c_int(n))
        if copied < 0:
            raise RuntimeError(f"sim_get_spikes failed with code {copied}")

        out = []
        for i in range(copied):
            out.append((float(buf[i].time), int(buf[i].neuron_id)))
        return out

    def get_voltage(self, neuron: int) -> float:
        """Read membrane potential v of one neuron."""
        if neuron < 0 or neuron >= self.n_neurons:
            raise IndexError(f"neuron index {neuron} out of range [0, {self.n_neurons})")
        v = ct.c_float()
        ret = _lib.sim_get_voltage(self._handle, ct.c_int(neuron), ct.byref(v))
        if ret != 0:
            raise RuntimeError(f"sim_get_voltage failed with code {ret}")
        return float(v.value)

    # ---- checkpointing -----------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save deterministic checkpoint + .sha256 hash (Rust side handles hash file).

        The Rust side writes:
            path
            path + ".sha256"
        """
        if self._closed:
            raise RuntimeError("Simulation is closed")
        bpath = path.encode("utf-8")
        ret = _lib.sim_save_checkpoint(self._handle, ct.c_char_p(bpath))
        if ret != 0:
            raise RuntimeError(f"sim_save_checkpoint failed with code {ret}")
        
        # ---- session / lifecycle API -----------------------------------------

    def reset(self):
        """
        Reset spike log but keep neuron states intact for closed-loop work.
        If a full state reset is needed, recreate NeuroSim.basic().
        """
        # Easiest: free + recreate, but let's keep ABI cheap:
        self.run_until(self.time)  # no-op ensures sim is synced
        self.spikes_last_read = 0

    # ---- time-stepping helpers -------------------------------------------

    def run_for(self, duration: float):
        """
        Run for relative time (t -> t + duration).
        """
        if duration < 0:
            raise ValueError("Duration must be >= 0")
        target = self.current_time() + duration
        self.run_until(target)

    def step(self, dt: float = 1.0):
        """
        Single control-loop step: run_for(dt), return latest spikes.
        """
        self.run_for(dt)
        return self.get_spikes()

    # ---- querying helpers -------------------------------------------------

    def current_time(self) -> float:
        """
        Read time by probing spike timestamps if ABI doesn't expose time.
        For your ABI this isn't exported, so we approximate via last event.
        Extension: add sim_get_time in Rust to expose it properly.
        """
        # This is placeholder logic and can be replaced once ABI exposes time
        spikes = self.get_spikes()
        if spikes:
            return spikes[-1][0]
        return 0.0

    def get_all_voltages(self):
        """
        Pull full state vector v(t).
        """
        return [self.get_voltage(i) for i in range(self.n_neurons)]

    # ---- optional stimulation abstraction --------------------------------

    def inject_spike(self, neuron: int, weight: float = 400.0, at_time: Optional[float] = None):
        """
        Handy abstraction: inject event "now" by default or delayed.
        """
        if at_time is None:
            at_time = self.current_time()
        self.push_current(at_time, neuron, weight)



# Convenience alias
__all__ = ["NeuroSim", "FfiSpike"]
