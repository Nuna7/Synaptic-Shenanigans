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
        return "libsynaptic_shenanigans.so"


def _load_lib() -> ct.CDLL:
    lib_path = os.environ.get("NEUROSIM_LIB")
    if lib_path is None:
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


# sim_create_basic
_lib.sim_create_basic.restype  = ct.POINTER(SimHandle)
_lib.sim_create_basic.argtypes = [ct.c_int, ct.c_int, ct.c_ulong]

# sim_free
_lib.sim_free.argtypes = [ct.POINTER(SimHandle)]
_lib.sim_free.restype  = None

# sim_push_current
_lib.sim_push_current.argtypes = [
    ct.POINTER(SimHandle),
    ct.c_float,   # time
    ct.c_int,     # target neuron
    ct.c_float,   # weight
]
_lib.sim_push_current.restype = ct.c_int

# sim_step_and_query
_lib.sim_step_and_query.argtypes = [ct.POINTER(SimHandle), ct.c_float]
_lib.sim_step_and_query.restype  = ct.c_int

# sim_spike_count
_lib.sim_spike_count.argtypes = [ct.POINTER(SimHandle)]
_lib.sim_spike_count.restype  = ct.c_int

# sim_clear_spikes  — clears the spike log in-place
_lib.sim_clear_spikes.argtypes = [ct.POINTER(SimHandle)]
_lib.sim_clear_spikes.restype  = ct.c_int

# sim_get_spikes
_lib.sim_get_spikes.argtypes = [
    ct.POINTER(SimHandle),
    ct.POINTER(FfiSpike),
    ct.c_int,
]
_lib.sim_get_spikes.restype = ct.c_int

# sim_get_voltage
_lib.sim_get_voltage.argtypes = [
    ct.POINTER(SimHandle),
    ct.c_int,
    ct.POINTER(ct.c_float),
]
_lib.sim_get_voltage.restype = ct.c_int

# sim_save_checkpoint
_lib.sim_save_checkpoint.argtypes = [ct.POINTER(SimHandle), ct.c_char_p]
_lib.sim_save_checkpoint.restype  = ct.c_int

_lib.sim_get_time.argtypes       = [ct.POINTER(SimHandle), ct.POINTER(ct.c_float)]
_lib.sim_get_time.restype        = ct.c_int

_lib.sim_set_scheduler.argtypes  = [ct.POINTER(SimHandle), ct.c_int, ct.c_int]
_lib.sim_set_scheduler.restype   = ct.c_int

_lib.sim_load_checkpoint.argtypes = [ct.c_char_p, ct.c_ulong, ct.c_int]
_lib.sim_load_checkpoint.restype  = ct.POINTER(SimHandle)

# -------------------------------------------------------------------
# High-level Python wrapper
# -------------------------------------------------------------------

class NeuroSim:
    """
    High-level object-oriented wrapper around the Rust C ABI.

    Usage:
        sim = NeuroSim.basic(n_neurons=2, n_threads=1, seed=42)
        sim.push_current(time=0.0, neuron=0, weight=400.0)
        sim.run_until(400.0)
        spikes = sim.get_spikes()
        v0     = sim.get_voltage(0)
        sim.close()

    Context-manager protocol is supported:
        with NeuroSim.basic(2) as sim:
            ...
    """

    def __init__(
        self,
        handle: ct.POINTER(SimHandle),
        n_neurons: int,
        n_threads: int,
        seed: int,
    ):
        self._handle   = handle
        self.n_neurons = n_neurons
        self.n_threads = n_threads
        self.seed      = seed
        self._closed   = False

    # ---- constructors ------------------------------------------------

    @classmethod
    def basic(
        cls,
        n_neurons: int,
        n_threads: int = 1,
        seed: int = 42,
    ) -> "NeuroSim":
        h = _lib.sim_create_basic(
            ct.c_int(n_neurons),
            ct.c_int(n_threads),
            ct.c_ulong(seed),
        )
        if not h:
            raise RuntimeError("sim_create_basic returned NULL")
        return cls(h, n_neurons=n_neurons, n_threads=n_threads, seed=seed)

    # ---- lifetime management -----------------------------------------

    def close(self) -> None:
        if not self._closed and self._handle:
            _lib.sim_free(self._handle)
            self._closed = True
            self._handle = None

    def __del__(self) -> None:
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
        """Inject a current-based event into *neuron* at *time*."""
        self._require_open()
        ret = _lib.sim_push_current(
            self._handle,
            ct.c_float(time),
            ct.c_int(neuron),
            ct.c_float(weight),
        )
        if ret != 0:
            raise RuntimeError(f"sim_push_current failed (code {ret})")

    # Alias used by inject_spike and other helpers
    inject_current = push_current

    def run_until(self, end_time: float) -> None:
        """Advance simulation up to and including *end_time* ms."""
        self._require_open()
        ret = _lib.sim_step_and_query(self._handle, ct.c_float(end_time))
        if ret != 0:
            raise RuntimeError(f"sim_step_and_query failed (code {ret})")

    # ---- spike log ---------------------------------------------------

    def spike_count(self) -> int:
        """Return the total number of spikes recorded so far."""
        self._require_open()
        return int(_lib.sim_spike_count(self._handle))

    def get_spikes(self) -> List[Tuple[float, int]]:
        """
        Return the full spike log as ``[(time_ms, neuron_id), ...]``.

        The list grows monotonically; use :meth:`clear_spikes` to reset it.
        """
        n = self.spike_count()
        if n <= 0:
            return []
        buf    = (FfiSpike * n)()
        copied = _lib.sim_get_spikes(self._handle, buf, ct.c_int(n))
        if copied < 0:
            raise RuntimeError(f"sim_get_spikes failed (code {copied})")
        return [(float(buf[i].time), int(buf[i].neuron_id)) for i in range(copied)]

    def clear_spikes(self) -> None:
        """Clear the spike log.  Neuron membrane states are preserved."""
        self._require_open()
        ret = _lib.sim_clear_spikes(self._handle)
        if ret != 0:
            raise RuntimeError(f"sim_clear_spikes failed (code {ret})")

    # ---- voltage queries ---------------------------------------------

    def get_voltage(self, neuron: int) -> float:
        """Read the membrane potential (mV) of *neuron*."""
        if neuron < 0 or neuron >= self.n_neurons:
            raise IndexError(
                f"neuron index {neuron} out of range [0, {self.n_neurons})"
            )
        v   = ct.c_float()
        ret = _lib.sim_get_voltage(
            self._handle, ct.c_int(neuron), ct.byref(v)
        )
        if ret != 0:
            raise RuntimeError(f"sim_get_voltage failed (code {ret})")
        return float(v.value)

    def get_all_voltages(self) -> List[float]:
        """Read membrane potentials for all neurons."""
        return [self.get_voltage(i) for i in range(self.n_neurons)]

    # ---- checkpointing -----------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Persist simulation state to *path*.

        The Rust side also writes ``path + ".sha256"`` for integrity checking.
        """
        self._require_open()
        ret = _lib.sim_save_checkpoint(
            self._handle, ct.c_char_p(path.encode("utf-8"))
        )
        if ret != 0:
            raise RuntimeError(f"sim_save_checkpoint failed (code {ret})")

    # ---- time-stepping helpers ---------------------------------------

    def current_time(self) -> float:
        """Return current simulation time (ms) via FFI — exact, not approximate."""
        if self._closed:
            raise RuntimeError("Simulation is closed")
        t = ct.c_float()
        ret = _lib.sim_get_time(self._handle, ct.byref(t))
        if ret != 0:
            raise RuntimeError(f"sim_get_time failed with code {ret}")
        return float(t.value)

    def run_for(self, duration: float) -> None:
        """Advance the simulation by *duration* ms relative to the current time."""
        if duration < 0:
            raise ValueError("duration must be >= 0")
        self.run_until(self.current_time() + duration)

    def step(self, dt: float = 1.0) -> List[Tuple[float, int]]:
        """
        Single closed-loop step: advance by *dt* ms then return all spikes.

        This does **not** auto-clear the log; callers that only want new spikes
        should call :meth:`clear_spikes` before :meth:`step`, or use the
        timestamp to filter.
        """
        self.run_for(dt)
        return self.get_spikes()

    def reset(self) -> None:
        """
        Clear the spike log while keeping neuron membrane states intact.

        Useful for closed-loop control loops where you want per-step spike
        counts without recreating the simulation.  For a full state reset,
        recreate with :meth:`basic`.
        """
        self.clear_spikes()

    # ---- stimulation helpers ----------------------------------------

    def inject_spike(
        self,
        neuron: int,
        weight: float = 400.0,
        at_time: Optional[float] = None,
    ) -> None:
        """
        Inject a current pulse into *neuron*.

        If *at_time* is ``None``, the event is scheduled at the current
        estimated simulation time.
        """
        if at_time is None:
            at_time = self.current_time()
        self.push_current(at_time, neuron, weight)

    def set_scheduler(self, mode: int, n_threads: int = 1) -> None:
        """Set scheduler mode. mode=0: single-threaded, mode=1: deterministic-MT."""
        if self._closed:
            raise RuntimeError("Simulation is closed")
        ret = _lib.sim_set_scheduler(self._handle, ct.c_int(mode), ct.c_int(n_threads))
        if ret != 0:
            raise RuntimeError(f"sim_set_scheduler failed: {ret}")
        
    @classmethod
    def from_checkpoint(cls, path: str, seed: int = 42, n_threads: int = 1) -> "NeuroSim":
        """Load a simulation from a checkpoint file."""
        bpath = path.encode("utf-8")
        h = _lib.sim_load_checkpoint(ct.c_char_p(bpath), ct.c_ulong(seed), ct.c_int(n_threads))
        if not h:
            raise RuntimeError(f"sim_load_checkpoint returned NULL for path: {path}")
        # n_neurons unknown from checkpoint alone — set to 0, user can override
        return cls(h, n_neurons=0, n_threads=n_threads, seed=seed)

    # ---- internals ---------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("NeuroSim instance is closed")


__all__ = ["NeuroSim", "FfiSpike"]