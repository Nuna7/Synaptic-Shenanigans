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

/// # Safety
/// 
/// - `handle` must be a valid, non-null pointer to a `SimHandle` created by `sim_create_basic`.
/// - `handle` must not have been freed by `sim_free` yet.
/// - This function must not be called concurrently with other functions using the same `handle` unless the handle is behind a mutex.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_create_basic(
    n_neurons: c_int, n_threads: c_int, seed: c_ulong,
) -> *mut SimHandle {
    let neurons = LifNeuron::new(n_neurons as usize, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let sim     = Simulation::new_with_seed(neurons, Synapse::new(), 1.0, seed, n_threads as usize);
    Box::into_raw(Box::new(SimHandle { sim: Box::into_raw(Box::new(sim)) }))
}


/// # Safety
///
/// `handle` must be a valid, non-null pointer to a `SimHandle` previously returned by
/// `sim_create_basic`. The handle must not have been freed. This function is not thread-safe:
/// do not call concurrently with other `sim_*` functions on the same handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_free(handle: *mut SimHandle) {
    if handle.is_null() { return; }
    unsafe {
        let h = Box::from_raw(handle);
        if !h.sim.is_null() { let _ = Box::from_raw(h.sim); }
    }
}

/// # Safety
///
/// - `handle` must be valid and non-null, from `sim_create_basic`.
/// - `end_time` must be finite, not NaN.
/// - The caller must ensure no other thread mutates the simulation while this runs.
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

/// # Safety
///
/// - `handle` must be valid and non-null.
/// - `mode` must be 0, 1, or 2 corresponding to valid scheduler variants.
/// - `n_threads` must be > 0 if mode requires threading.
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

/// # Safety
///
/// - `handle` must be valid and non-null.
/// - `out_t` must be either null or a valid, writable pointer to `c_float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_get_time(handle: *mut SimHandle, out_t: *mut c_float) -> c_int {
    if handle.is_null() || out_t.is_null() { return -1; }
    unsafe {
        let sim = &*(*handle).sim;
        *out_t = sim.time as c_float;
    }
    0
}

/// # Safety
///
/// - `handle` must be valid and non-null.
/// - `time` and `weight` must be finite.
/// - `target` must be a valid neuron index < population size.
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

/// # Safety
///
/// `handle` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_spike_count(handle: *mut SimHandle) -> c_int {
    if handle.is_null() { return -1; }
    unsafe { (*(*handle).sim).spike_log.len() as c_int }
}

/// # Safety
///
/// `handle` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_clear_spikes(handle: *mut SimHandle) -> c_int {
    if handle.is_null() { return -1; }
    unsafe { (*(*handle).sim).spike_log.clear(); }
    0
}

/// # Safety
///
/// - `handle` must be valid and non-null.
/// - `out` must be valid for writes of `max * size_of::<FfiSpike>()` bytes if `max > 0`.
/// - `max` must not be negative.
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


/// # Safety
///
/// - `handle` must be valid and non-null.
/// - `neuron` must be a valid index < population size.
/// - `out_v` must be null or valid for writing one `c_float`.
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

/// # Safety
///
/// - `handle` must be valid and non-null.
/// - `path` must be a valid pointer to a null-terminated C string.
/// - The filesystem path must be writable by this process.
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