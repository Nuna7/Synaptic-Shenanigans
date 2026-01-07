#![cfg_attr(not(feature = "ffi"), allow(dead_code))]

pub mod event;
pub mod lif;
pub mod simulation;
pub mod synapse;

use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int, c_ulong};

use crate::simulation::{Simulation, SchedulerMode};
use crate::lif::{LifNeuron, NeuronPopulation};
use crate::synapse::Synapse;

/// Opaque handle exposed to C / Python.
///
/// From C you only see a pointer to this struct.
#[repr(C)]
pub struct SimHandle {
    sim: *mut Simulation,
}

/// Simple C-side representation of a spike: (time, neuron_id)
#[repr(C)]
pub struct FfiSpike {
    pub time: c_float,
    pub neuron_id: c_int,
}

// ----------------------
// Construction / teardown
// ----------------------

/// Create a new simulation and return an opaque handle.
///
/// # Safety
/// - The returned pointer must eventually be passed to `sim_free` exactly once.
/// - The returned handle must not be accessed after being freed.
/// - The caller must not concurrently use the same handle from multiple threads
///   unless externally synchronized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_create_basic(
    n_neurons: c_int,
    n_threads: c_int,
    seed: c_ulong,
) -> *mut SimHandle {
    let neurons = LifNeuron::new(
        n_neurons as usize,
        -65.0, // v_rest
        -50.0, // v_thresh
        20.0,  // tau_m
        1.0,   // r_m
        1.0,   // dt
        5.0,   // refractory_period
    );

    let syn = Synapse::new();
    let sim = Simulation::new_with_seed(neurons, syn, 1.0, seed, n_threads as usize);

    let boxed_sim = Box::new(sim);
    let handle = Box::new(SimHandle {
        sim: Box::into_raw(boxed_sim),
    });
    Box::into_raw(handle)
}

/// Free a simulation handle.
///
/// # Safety
/// - `handle` must be a pointer previously returned by `sim_create_basic`.
/// - `handle` must not have been freed already.
/// - No other thread may be using the handle while this function is called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_free(handle: *mut SimHandle) {
    if handle.is_null() {
        return;
    }

    // SAFETY: caller must provide a valid pointer previously returned by sim_create_basic,
    // and must ensure no other references remain.
    unsafe {
        let handle_box = Box::from_raw(handle);
        if !handle_box.sim.is_null() {
            // Reconstruct Box<Simulation> and drop it.
            let _ = Box::from_raw(handle_box.sim);
        }
        // `handle_box` (Box<SimHandle>) drops here; it contains only a raw pointer field.
    }
}

// ----------------------
// Core stepping / control
// ----------------------

/// Advance the simulation until `end_time`.
///
/// # Safety
/// - `handle` must be a valid, non-null pointer returned by `sim_create_basic`.
/// - The handle must not have been freed.
/// - No concurrent access to the same handle is allowed during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_step_and_query(
    handle: *mut SimHandle,
    end_time: c_float,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    // SAFETY: pointer validity / exclusivity guaranteed by caller contract.
    unsafe {
        let sim = &mut *(*handle).sim;
        sim.scheduler_mode = SchedulerMode::SingleThreaded;
        sim.run_auto(end_time);
    }

    0
}

// ----------------------
// External input / events
// ----------------------

/// Inject a current-based external event.
///
/// # Safety
/// - `handle` must be a valid, non-null pointer returned by `sim_create_basic`.
/// - The handle must not have been freed.
/// - The simulation must not be concurrently accessed from another thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_push_current(
    handle: *mut SimHandle,
    time: c_float,
    target: c_int,
    weight: c_float,
) -> c_int {
    if handle.is_null() {
        return -1;
    }
    if target < 0 {
        return -2;
    }

    unsafe {
        let sim = &mut *(*handle).sim;
        let n = sim.neurons.as_ref().len();
        let idx = target as usize;
        if idx >= n {
            return -2;
        }
        // model_type = 0 => current-based; e_rev is ignored
        sim.push_event(time, idx, weight, 0u8, 0.0);
    }
    0
}

// ----------------------
// Spike log access
// ----------------------

/// Return the number of spikes recorded.
///
/// # Safety
/// - `handle` must be a valid, non-null pointer returned by `sim_create_basic`.
/// - The handle must not have been freed.
/// - No concurrent mutation of the simulation may occur during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_spike_count(handle: *mut SimHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    unsafe {
        let sim = &*(*handle).sim;
        sim.spike_log.len() as c_int
    }
}

/// Clear the spike log.
///
/// # Safety
/// - `handle` must be a valid, non-null pointer returned by `sim_create_basic`.
/// - The handle must not have been freed.
/// - No concurrent access to the simulation is allowed during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_clear_spikes(handle: *mut SimHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    unsafe {
        let sim = &mut *(*handle).sim;
        sim.spike_log.clear();
    }
    0
}

/// Copy spike log entries into a user-provided buffer.
///
/// # Safety
/// - `handle` must be a valid, non-null pointer returned by `sim_create_basic`.
/// - `out_spikes` must point to writable memory for at least `max_spikes` entries.
/// - The handle must not have been freed.
/// - No concurrent mutation of the spike log may occur during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_get_spikes(
    handle: *mut SimHandle,
    out_spikes: *mut FfiSpike,
    max_spikes: c_int,
) -> c_int {
    if handle.is_null() || out_spikes.is_null() || max_spikes <= 0 {
        return -1;
    }

    unsafe {
        let sim = &*(*handle).sim;
        let available = sim.spike_log.len() as c_int;
        let to_copy = available.min(max_spikes);

        // SAFETY: caller must ensure out_spikes points to valid writable memory for `to_copy` elements.
        let slice = std::slice::from_raw_parts_mut(out_spikes, to_copy as usize);
        for (i, (t, nid)) in sim.spike_log.iter().take(to_copy as usize).enumerate() {
            slice[i] = FfiSpike {
                time: *t as c_float,
                neuron_id: *nid as c_int,
            };
        }
        to_copy
    }
}

// ----------------------
// Voltage / probe access
// ----------------------

/// Read the membrane potential of a neuron.
///
/// # Safety
/// - `handle` must be a valid, non-null pointer returned by `sim_create_basic`.
/// - `out_v` must point to valid writable memory.
/// - The handle must not have been freed.
/// - No concurrent mutation of neuron state may occur during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_get_voltage(
    handle: *mut SimHandle,
    neuron: c_int,
    out_v: *mut c_float,
) -> c_int {
    if handle.is_null() || out_v.is_null() {
        return -1;
    }
    if neuron < 0 {
        return -2;
    }

    unsafe {
        let sim = &*(*handle).sim;
        let n = sim.neurons.as_ref().len();
        let idx = neuron as usize;
        if idx >= n {
            return -2;
        }
        let v = sim.neurons.read_v(idx);
        *out_v = v as c_float;
    }
    0
}

// ----------------------
// Checkpointing
// ----------------------

/// Save the simulation state to disk.
///
/// # Safety
/// - `handle` must be a valid, non-null pointer returned by `sim_create_basic`.
/// - `path` must be a valid, null-terminated C string.
/// - The handle must not have been freed.
/// - No concurrent mutation of the simulation may occur during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_save_checkpoint(
    handle: *mut SimHandle,
    path: *const c_char,
) -> c_int {
    if handle.is_null() || path.is_null() {
        return -1;
    }

    // SAFETY: `path` is a C string pointer; caller must ensure it's valid.
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    unsafe {
        let sim = &*(*handle).sim;
        let hash_path = format!("{}.sha256", path_str);
        if let Err(e) = sim.save_state(path_str, &hash_path) {
            eprintln!("save_state error: {}", e);
            return -3;
        }
    }
    0
}
