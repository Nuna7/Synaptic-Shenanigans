use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event{
    pub tick: u64,     // discrete time index = floor(time / dt)
    pub time: f32,      // When the event should be processed
    pub target: usize,  // ID of target neuron
    pub weight: f32,    // Synaptic weight or current
    pub seq: u64,       // Monotonic sequence for deterministic tie-breaking 
    pub model_type: u8, // 0=current, 1=conductance
    pub e_rev: f32,     // used when model_type == 1
}

// BinaryHeap is a max-heap so we reverse order: smaller tick -> greater ordering result
impl Ord for Event {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order by tick (earlier ticks are "greater" so pop() gives smallest tick),
        // tie-break by seq (smaller seq -> earlier)
        other.tick.cmp(&self.tick).then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        // We intentionally ignore floats because floats cannot implement Eq safely.
        // Determinism is guaranteed by (tick, seq) ordering.
        self.tick == other.tick && self.seq == other.seq
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}



