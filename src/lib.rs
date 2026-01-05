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

#[unsafe(no_mangle)]
pub extern "C" fn sim_create_basic(
    n_neurons: c_int,
    n_threads: c_int,
    seed: c_ulong,
) -> *mut SimHandle {
    // Basic LIF population with uniform parameters
    let neurons = LifNeuron::new(
        n_neurons as usize,
        -65.0, // v_rest
        -50.0, // v_thresh
        20.0,  // tau_m
        1.0,   // r_m
        1.0,   // dt
        5.0,   // refractory_period
    );

    // Start with empty synapse set; caller can drive via external events
    let syn = Synapse::new();
    let sim = Simulation::new_with_seed(neurons, syn, 1.0, seed, n_threads as usize);

    let boxed_sim = Box::new(sim);
    let handle = Box::new(SimHandle {
        sim: Box::into_raw(boxed_sim),
    });
    Box::into_raw(handle)
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_free(handle: *mut SimHandle) {
    if handle.is_null() {
        return;
    }
    unsafe {
        // Reconstruct box and drop it
        let handle_box = Box::from_raw(handle);
        if !handle_box.sim.is_null() {
            let _ = Box::from_raw(handle_box.sim);
        }
    }
}

// ----------------------
// Core stepping / control
// ----------------------

/// Step the simulation until `end_time` (absolute time, in the same units as `dt`).
/// Returns 0 on success, negative on error.
///
/// This uses the current `scheduler_mode` internally; we force SingleThreaded
/// here just to keep demo behavior simple.
#[unsafe(no_mangle)]
pub extern "C" fn sim_step_and_query(handle: *mut SimHandle, end_time: c_float) -> c_int {
    if handle.is_null() {
        return -1;
    }
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

/// Push a *current-based* external event into the simulation.
///
/// - `time`   : when the event should arrive
/// - `target` : neuron index (0-based)
/// - `weight` : injected current
///
/// Returns:
///   0  on success
///  -1  invalid handle
///  -2  target out of range
#[unsafe(no_mangle)]
pub extern "C" fn sim_push_current(
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

/// Return total spike count in the log (for convenience).
#[unsafe(no_mangle)]
pub extern "C" fn sim_spike_count(handle: *mut SimHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    unsafe {
        let sim = &*(*handle).sim;
        sim.spike_log.len() as c_int
    }
}

/// Clear the spike log (does *not* affect neuron state).
#[unsafe(no_mangle)]
pub extern "C" fn sim_clear_spikes(handle: *mut SimHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    unsafe {
        let sim = &mut *(*handle).sim;
        sim.spike_log.clear();
    }
    0
}

/// Copy spike log into a user-provided buffer of `FfiSpike`.
///
/// - `out_spikes` : pointer to an array of at least `max_spikes` elements
/// - `max_spikes` : capacity of the array
///
/// Returns:
///   >=0 : number of spikes actually written
///   -1  : null handle or buffer
#[unsafe(no_mangle)]
pub extern "C" fn sim_get_spikes(
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
        let to_copy = if available < max_spikes {
            available
        } else {
            max_spikes
        };

        let slice = std::slice::from_raw_parts_mut(out_spikes, to_copy as usize);
        for (i, (t, nid)) in sim
            .spike_log
            .iter()
            .take(to_copy as usize)
            .enumerate()
        {
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

/// Read the membrane potential of a single neuron.
///
/// - `neuron` : index (0-based)
/// - `out_v`  : pointer to a single `float` to store the result
///
/// Returns:
///   0  on success
///  -1  invalid handle or out_v
///  -2  neuron index out of range
#[unsafe(no_mangle)]
pub extern "C" fn sim_get_voltage(
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

/// Save a checkpoint plus a `*.sha256` hash next to it.
///
/// Returns:
///   0  on success
///  -1  invalid handle or path
///  -2  invalid UTF-8 in path
///  -3  I/O or serialization error
#[unsafe(no_mangle)]
pub extern "C" fn sim_save_checkpoint(
    handle: *mut SimHandle,
    path: *const c_char,
) -> c_int {
    if handle.is_null() || path.is_null() {
        return -1;
    }
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
