#ifndef NEUROSIM_H
#define NEUROSIM_H

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct Simulation Simulation;

/**
 * Opaque handle exposed to C / Python.
 *
 * From C you only see a pointer to this struct.
 */
typedef struct SimHandle {
  struct Simulation *sim;
} SimHandle;

/**
 * Simple C-side representation of a spike: (time, neuron_id)
 */
typedef struct FfiSpike {
  float time;
  int neuron_id;
} FfiSpike;

struct SimHandle *sim_create_basic(int n_neurons, int n_threads, unsigned long seed);

void sim_free(struct SimHandle *handle);

/**
 * Step the simulation until `end_time` (absolute time, in the same units as `dt`).
 * Returns 0 on success, negative on error.
 *
 * This uses the current `scheduler_mode` internally; we force SingleThreaded
 * here just to keep demo behavior simple.
 */
int sim_step_and_query(struct SimHandle *handle, float end_time);

/**
 * Push a *current-based* external event into the simulation.
 *
 * - `time`   : when the event should arrive
 * - `target` : neuron index (0-based)
 * - `weight` : injected current
 *
 * Returns:
 *   0  on success
 *  -1  invalid handle
 *  -2  target out of range
 */
int sim_push_current(struct SimHandle *handle, float time, int target, float weight);

/**
 * Return total spike count in the log (for convenience).
 */
int sim_spike_count(struct SimHandle *handle);

/**
 * Clear the spike log (does *not* affect neuron state).
 */
int sim_clear_spikes(struct SimHandle *handle);

/**
 * Copy spike log into a user-provided buffer of `FfiSpike`.
 *
 * - `out_spikes` : pointer to an array of at least `max_spikes` elements
 * - `max_spikes` : capacity of the array
 *
 * Returns:
 *   >=0 : number of spikes actually written
 *   -1  : null handle or buffer
 */
int sim_get_spikes(struct SimHandle *handle, struct FfiSpike *out_spikes, int max_spikes);

/**
 * Read the membrane potential of a single neuron.
 *
 * - `neuron` : index (0-based)
 * - `out_v`  : pointer to a single `float` to store the result
 *
 * Returns:
 *   0  on success
 *  -1  invalid handle or out_v
 *  -2  neuron index out of range
 */
int sim_get_voltage(struct SimHandle *handle, int neuron, float *out_v);

/**
 * Save a checkpoint plus a `*.sha256` hash next to it.
 *
 * Returns:
 *   0  on success
 *  -1  invalid handle or path
 *  -2  invalid UTF-8 in path
 *  -3  I/O or serialization error
 */
int sim_save_checkpoint(struct SimHandle *handle, const char *path);

#endif  /* NEUROSIM_H */
