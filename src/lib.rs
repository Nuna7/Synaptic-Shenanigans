#![cfg_attr(not(feature = "ffi"), allow(dead_code))]

// ── Core modules ──────────────────────────────────────────────────────────────
pub mod event;
pub mod lif;
pub mod simulation;
pub mod synapse;

// ── Neuron models ─────────────────────────────────────────────────────────────
pub mod izhikevich;     // multi-type spiking (RS, FS, IB, CH, LTS, RZ)
pub mod hodgkin_huxley; // biophysical Na+/K+/leak conductances
pub mod adex;           // Adaptive Exponential Integrate-and-Fire

// ── Plasticity ────────────────────────────────────────────────────────────────
pub mod plasticity;     // STDP (Hebbian, nearest-neighbour)
pub mod homeostatic;    // intrinsic excitability regulation
pub mod synaptic_scaling; // multiplicative weight homeostasis

// ── Network topology ──────────────────────────────────────────────────────────
pub mod network;        // Erdős-Rényi, Watts-Strogatz, Barabási-Albert, Layered

// ── Input generation ──────────────────────────────────────────────────────────
pub mod poisson;        // homogeneous + inhomogeneous Poisson processes

// ── Analysis ──────────────────────────────────────────────────────────────────
pub mod metrics;        // synchrony, bursts, power spectrum, avalanches

// ── FFI ───────────────────────────────────────────────────────────────────────

use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int, c_ulong};

use crate::simulation::{Simulation, SchedulerMode};
use crate::lif::{LifNeuron, NeuronPopulation};
use crate::synapse::Synapse;

#[repr(C)]
pub struct SimHandle { sim: *mut Simulation }

#[repr(C)]
pub struct FfiSpike { pub time: c_float, pub neuron_id: c_int }

// ── Construction ──────────────────────────────────────────────────────────────

/// Create a simulation with LIF neurons and empty synapse set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_create_basic(
    n_neurons: c_int, n_threads: c_int, seed: c_ulong,
) -> *mut SimHandle {
    let neurons = LifNeuron::new(n_neurons as usize, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let sim     = Simulation::new_with_seed(neurons, Synapse::new(), 1.0, seed, n_threads as usize);
    Box::into_raw(Box::new(SimHandle { sim: Box::into_raw(Box::new(sim)) }))
}

/// Free a simulation handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_free(handle: *mut SimHandle) {
    if handle.is_null() { return; }
    unsafe {
        let h = Box::from_raw(handle);
        if !h.sim.is_null() { let _ = Box::from_raw(h.sim); }
    }
}

// ── Stepping ──────────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_step_and_query(handle: *mut SimHandle, end_time: c_float) -> c_int {
    if handle.is_null() { return -1; }
    unsafe {
        let sim = &mut *(*handle).sim;
        sim.scheduler_mode = SchedulerMode::SingleThreaded;
        sim.run_auto(end_time);
    }
    0
}

// ── Scheduler control — NEW ───────────────────────────────────────────────────

/// Set scheduler mode. mode=0 → SingleThreaded, mode=1 → Deterministic(n_threads).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_set_scheduler(
    handle: *mut SimHandle, mode: c_int, n_threads: c_int,
) -> c_int {
    if handle.is_null() { return -1; }
    unsafe {
        let sim = &mut *(*handle).sim;
        sim.scheduler_mode = match mode {
            0 => SchedulerMode::SingleThreaded,
            1 => SchedulerMode::Deterministic { n_threads: n_threads as usize },
            _ => return -2,
        };
    }
    0
}

// ── Time query — NEW ──────────────────────────────────────────────────────────

/// Return current simulation time (ms).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_get_time(handle: *mut SimHandle, out_t: *mut c_float) -> c_int {
    if handle.is_null() || out_t.is_null() { return -1; }
    unsafe {
        let sim = &*(*handle).sim;
        *out_t = sim.time as c_float;
    }
    0
}

// ── Input ─────────────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_push_current(
    handle: *mut SimHandle, time: c_float, target: c_int, weight: c_float,
) -> c_int {
    if handle.is_null() { return -1; }
    if target < 0 { return -2; }
    unsafe {
        let sim = &mut *(*handle).sim;
        if target as usize >= sim.neurons.as_ref().len() { return -2; }
        sim.push_event(time, target as usize, weight, 0u8, 0.0);
    }
    0
}

// ── Spike log ────────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_spike_count(handle: *mut SimHandle) -> c_int {
    if handle.is_null() { return -1; }
    unsafe { (*(*handle).sim).spike_log.len() as c_int }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_clear_spikes(handle: *mut SimHandle) -> c_int {
    if handle.is_null() { return -1; }
    unsafe { (*(*handle).sim).spike_log.clear(); }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_get_spikes(
    handle: *mut SimHandle, out: *mut FfiSpike, max: c_int,
) -> c_int {
    if handle.is_null() || out.is_null() || max <= 0 { return -1; }
    unsafe {
        let log = &(*(*handle).sim).spike_log;
        let n = (log.len() as c_int).min(max);
        let slice = std::slice::from_raw_parts_mut(out, n as usize);
        for (i, (t, nid)) in log.iter().take(n as usize).enumerate() {
            slice[i] = FfiSpike { time: *t as c_float, neuron_id: *nid as c_int };
        }
        n
    }
}

// ── Voltage ──────────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_get_voltage(
    handle: *mut SimHandle, neuron: c_int, out_v: *mut c_float,
) -> c_int {
    if handle.is_null() || out_v.is_null() { return -1; }
    if neuron < 0 { return -2; }
    unsafe {
        let sim = &*(*handle).sim;
        if neuron as usize >= sim.neurons.as_ref().len() { return -2; }
        *out_v = sim.neurons.read_v(neuron as usize) as c_float;
    }
    0
}

// ── Checkpointing ────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_save_checkpoint(handle: *mut SimHandle, path: *const c_char) -> c_int {
    if handle.is_null() || path.is_null() { return -1; }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s, Err(_) => return -2,
    };
    unsafe {
        let sim = &*(*handle).sim;
        let hash = format!("{}.sha256", path_str);
        if sim.save_state(path_str, &hash).is_err() { return -3; }
    }
    0
}

/// Load a checkpoint and return a new handle. Caller owns the returned pointer.
///
/// # Safety
/// `path` must be a valid null-terminated C string.
/// The returned handle must be freed with `sim_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_load_checkpoint(
    path: *const c_char, seed: c_ulong, n_threads: c_int,
) -> *mut SimHandle {
    if path.is_null() { return std::ptr::null_mut(); }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s, Err(_) => return std::ptr::null_mut(),
    };
    match Simulation::load_state(path_str, seed, n_threads as usize) {
        Ok(sim) => Box::into_raw(Box::new(SimHandle { sim: Box::into_raw(Box::new(sim)) })),
        Err(e)  => { eprintln!("sim_load_checkpoint: {}", e); std::ptr::null_mut() }
    }
}