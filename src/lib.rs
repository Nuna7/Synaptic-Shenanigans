// ─────────────────────────────────────────────────────────────────────────────
// Synaptic-Shenanigans — crate root
// ─────────────────────────────────────────────────────────────────────────────

// Core flat modules
pub mod checkpoint;
pub mod event;
pub mod metrics;
pub mod network;
pub mod simulation;
pub mod synapse;

// Subsystem modules (canonical implementations live here)
pub mod input;
pub mod neurons;
pub mod plasticity;
pub mod topology;

// gRPC (optional)
#[cfg(feature = "rpc")]
pub mod rpc;

// ── Crate-root re-exports ─────────────────────────────────────────────────────

// Neuron models
pub use neurons::{
    AdExParams, AdExPopulation, AdExProfile, HHParams, HHPopulation, IzhikevichPop, LifNeuron,
    NeuronType,
};
pub use neurons::{NeuronPartition, NeuronPopulation};

// Plasticity
pub use plasticity::{
    HomeostaticConfig, HomeostaticState, PlasticityContext, PlasticityRule, StdpConfig, StdpState,
    SynapticScaling, SynapticScalingConfig, ThresholdStats, WeightStats,
};

// Network topology (old Synapse-returning NetworkBuilder, used by demos/tests)
pub use network::{EdgeParams, NetworkBuilder, NetworkMetrics};

// Input generation
pub use input::poisson::{PoissonPopulation, PoissonSource, StimulusPattern, drive_background};

// Core
pub use checkpoint::Checkpoint;
pub use input::StimulusSource;
pub use metrics::{
    AvalancheResult, Burst, BurstDetector, ISIStats, SynchronyIndex, dominant_frequency,
    power_spectrum,
};
pub use simulation::{SchedulerMode, SimConfig, Simulation, replay_equal};
pub use synapse::{Synapse, SynapseMatrix, synapse_current};
pub use topology::TopologyGenerator;

// ─────────────────────────────────────────────────────────────────────────────
// C ABI — neurosim_* handle-based API (used by Python ctypes via any language)
// ─────────────────────────────────────────────────────────────────────────────

use simulation::Simulation as Sim;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
lazy_static::lazy_static! {
    static ref SIMS: Mutex<HashMap<u64, Arc<Mutex<Sim>>>> = Mutex::new(HashMap::new());
    static ref NEXT_ID: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(1);
}

fn insert_sim(s: Sim) -> u64 {
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    SIMS.lock().unwrap().insert(id, Arc::new(Mutex::new(s)));
    id
}

fn with_sim<F, T>(id: u64, f: F) -> Option<T>
where
    F: FnOnce(&mut Sim) -> T,
{
    let arc = SIMS.lock().unwrap().get(&id)?.clone();
    Some(f(&mut arc.lock().unwrap()))
}

#[repr(C)]
pub struct FfiSpike {
    pub time: f32,
    pub neuron_id: std::os::raw::c_int,
}

#[unsafe(no_mangle)]
pub extern "C" fn neurosim_create(
    n_neurons: u32,
    n_threads: u32,
    seed: u64,
    scheduler: u32,
) -> u64 {
    let sched = match scheduler {
        1 => SchedulerMode::Deterministic {
            n_threads: n_threads as usize,
        },
        _ => SchedulerMode::SingleThreaded,
    };
    let neurons = LifNeuron::new(n_neurons as usize, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let mut sim = Sim::new_with_neurons(neurons, Synapse::new(), 1.0, seed, n_threads as usize);
    sim.scheduler_mode = sched;
    insert_sim(sim)
}

/// # Safety
///
/// - `handle` must be a valid pointer returned by `sim_create_basic`.
/// - `out_t` must be non-null and writable.
/// - The caller must ensure exclusive access.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn neurosim_load(path_ptr: *const std::os::raw::c_char) -> u64 {
    let path = unsafe { std::ffi::CStr::from_ptr(path_ptr).to_str().unwrap_or("") };
    match checkpoint::Checkpoint::load(path) {
        Ok(sim) => insert_sim(sim),
        Err(_) => 0,
    }
}

/// # Safety
///
/// - `handle` must be a valid pointer returned by `sim_create_basic`.
/// - `out_t` must be non-null and writable.
/// - The caller must ensure exclusive access.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn neurosim_step(
    id: u64,
    until_time: f32,
    out_buf: *mut FfiSpike,
    buf_len: u32,
) -> u32 {
    with_sim(id, |sim| {
        let spikes = sim.step_until(until_time);
        let n = spikes.len().min(buf_len as usize);
        for (i, &(t, nid)) in spikes.iter().take(n).enumerate() {
            unsafe {
                *out_buf.add(i) = FfiSpike {
                    time: t,
                    neuron_id: nid as _,
                };
            }
        }
        n as u32
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn neurosim_push(id: u64, time: f32, neuron: u32, weight: f32) {
    with_sim(id, |sim| {
        sim.push_event(time, neuron as usize, weight, 0, 0.0)
    });
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn neurosim_get_voltages(id: u64, out_buf: *mut f32, buf_len: u32) -> u32 {
    with_sim(id, |sim| {
        let v = sim.get_all_voltages();
        let n = v.len().min(buf_len as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), out_buf, n);
        }
        n as u32
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn neurosim_spike_count(id: u64) -> u32 {
    with_sim(id, |s| s.spike_count() as u32).unwrap_or(0)
}
#[unsafe(no_mangle)]
pub extern "C" fn neurosim_current_time(id: u64) -> f32 {
    with_sim(id, |s| s.current_time()).unwrap_or(0.0)
}
#[unsafe(no_mangle)]
pub extern "C" fn neurosim_clear_spikes(id: u64) {
    with_sim(id, |s| s.clear_spikes());
}
#[unsafe(no_mangle)]
pub extern "C" fn neurosim_inject_spike(id: u64, neuron: u32, weight: f32) {
    with_sim(id, |s| s.inject_spike(neuron, weight));
}
#[unsafe(no_mangle)]
pub extern "C" fn neurosim_free(id: u64) {
    SIMS.lock().unwrap().remove(&id);
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn neurosim_checkpoint(
    id: u64,
    path_ptr: *const std::os::raw::c_char,
) -> u32 {
    let path = unsafe { std::ffi::CStr::from_ptr(path_ptr).to_str().unwrap_or("") };
    with_sim(id, |s| checkpoint::Checkpoint::save(s, path).is_ok() as u32).unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// C ABI — sim_* pointer-based API (for Python neurosim_ffi.py ctypes)
// ─────────────────────────────────────────────────────────────────────────────

use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int, c_ulong};

#[repr(C)]
pub struct SimHandle {
    sim: *mut Sim,
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_create_basic(
    n_neurons: c_int,
    n_threads: c_int,
    seed: c_ulong,
) -> *mut SimHandle {
    let neurons = LifNeuron::new(n_neurons as usize, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
    let sim = Sim::new_with_neurons(neurons, Synapse::new(), 1.0, seed, n_threads as usize);
    Box::into_raw(Box::new(SimHandle {
        sim: Box::into_raw(Box::new(sim)),
    }))
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_free(handle: *mut SimHandle) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let h = Box::from_raw(handle);
        if !h.sim.is_null() {
            drop(Box::from_raw(h.sim));
        }
    }
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_step_and_query(handle: *mut SimHandle, end_time: c_float) -> c_int {
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

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_set_scheduler(
    handle: *mut SimHandle,
    mode: c_int,
    n_threads: c_int,
) -> c_int {
    if handle.is_null() {
        return -1;
    }
    unsafe {
        (*(*handle).sim).scheduler_mode = match mode {
            0 => SchedulerMode::SingleThreaded,
            1 => SchedulerMode::Deterministic {
                n_threads: n_threads as usize,
            },
            _ => return -2,
        };
    }
    0
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_get_time(handle: *mut SimHandle, out_t: *mut c_float) -> c_int {
    if handle.is_null() || out_t.is_null() {
        return -1;
    }
    unsafe {
        *out_t = (*(*handle).sim).time;
    }
    0
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
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
        if target as usize >= sim.neurons.len() {
            return -2;
        }
        sim.push_event(time, target as usize, weight, 0, 0.0);
    }
    0
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_spike_count(handle: *mut SimHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    unsafe { (*(*handle).sim).spike_log.len() as c_int }
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_clear_spikes(handle: *mut SimHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    unsafe {
        (*(*handle).sim).spike_log.clear();
    }
    0
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_get_spikes(
    handle: *mut SimHandle,
    out: *mut FfiSpike,
    max: c_int,
) -> c_int {
    if handle.is_null() || out.is_null() || max <= 0 {
        return -1;
    }
    unsafe {
        let log = &(*(*handle).sim).spike_log;
        let n = (log.len() as c_int).min(max);
        let slice = std::slice::from_raw_parts_mut(out, n as usize);
        for (i, (t, nid)) in log.iter().take(n as usize).enumerate() {
            slice[i] = FfiSpike {
                time: *t,
                neuron_id: *nid as c_int,
            };
        }
        n
    }
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
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
        if neuron as usize >= sim.neurons.len() {
            return -2;
        }
        *out_v = sim.neurons.read_v(neuron as usize);
    }
    0
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_save_checkpoint(handle: *mut SimHandle, path: *const c_char) -> c_int {
    if handle.is_null() || path.is_null() {
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    unsafe {
        let sim = &*(*handle).sim;
        if sim
            .save_state(path_str, &format!("{path_str}.sha256"))
            .is_err()
        {
            return -3;
        }
    }
    0
}

/// # Safety
///
/// The caller must ensure all raw pointers are valid,
/// non-null where required, and properly aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_load_checkpoint(
    path: *const c_char,
    seed: c_ulong,
    n_threads: c_int,
) -> *mut SimHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match Sim::load_state(path_str, seed, n_threads as usize) {
        Ok(sim) => Box::into_raw(Box::new(SimHandle {
            sim: Box::into_raw(Box::new(sim)),
        })),
        Err(e) => {
            eprintln!("sim_load_checkpoint: {e}");
            std::ptr::null_mut()
        }
    }
}
