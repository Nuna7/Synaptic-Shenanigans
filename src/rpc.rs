use tonic::{Request, Response, Status};
use std::sync::{Arc, Mutex};
use std::pin::Pin;
use futures::{Stream, stream};

use crate::simulation::{Simulation, SchedulerMode};
use crate::lif::{LifNeuron, NeuronPopulation};
use crate::synapse::Synapse;

pub mod pb {
    tonic::include_proto!("neurosim");
}

use pb::*;
use pb::neuro_sim_server::{NeuroSim, NeuroSimServer};

#[derive(Default)]
pub struct SimStore {
    sims: Mutex<Vec<Arc<Mutex<Simulation>>>>,
}

impl SimStore {
    pub fn new() -> Self { Self::default() }

    pub fn create(&self, sim: Simulation) -> u64 {
        let mut sims = self.sims.lock().unwrap();
        sims.push(Arc::new(Mutex::new(sim)));
        (sims.len() - 1) as u64
    }

    pub fn get(&self, id: u64) -> Result<Arc<Mutex<Simulation>>, Status> {
        let sims = self.sims.lock().unwrap();
        sims.get(id as usize)
            .cloned()
            .ok_or_else(|| Status::not_found("invalid sim id"))
    }
}

pub struct RpcService { store: Arc<SimStore> }
impl RpcService {
    pub fn new(store: Arc<SimStore>) -> Self { Self { store } }
}

#[tonic::async_trait]
impl NeuroSim for RpcService {
    type StreamSpikesStream = Pin<Box<dyn Stream<Item = Result<Spike, Status>> + Send>>;

    async fn create(&self, req: Request<SimConfig>) -> Result<Response<Handle>, Status> {
        let cfg = req.into_inner();
        let neurons = LifNeuron::new(cfg.n_neurons as usize, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
        let mut sim = Simulation::new_with_seed(neurons, Synapse::new(), 1.0, cfg.seed, cfg.n_threads as usize);
        sim.scheduler_mode = match cfg.scheduler {
            1 => SchedulerMode::Deterministic { n_threads: cfg.n_threads as usize },
            _ => SchedulerMode::SingleThreaded,
        };
        Ok(Response::new(Handle { id: self.store.create(sim) }))
    }

    async fn free(&self, req: Request<Handle>) -> Result<Response<Empty>, Status> {
        let id = req.into_inner().id;
        let mut sims = self.store.sims.lock().unwrap();
        if let Some(slot) = sims.get_mut(id as usize) {
            let neurons = LifNeuron::new(0, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
            *slot = Arc::new(Mutex::new(
                Simulation::new_with_seed(neurons, Synapse::new(), 1.0, 0, 1)
            ));
        }
        Ok(Response::new(Empty {}))
    }

    async fn push(&self, req: Request<InputEvent>) -> Result<Response<Empty>, Status> {
        let ev = req.into_inner();
        let entry = self.store.get(ev.sim_id)?;
        let mut sim = entry.lock().unwrap();
        sim.push_event(ev.time, ev.neuron as usize, ev.weight, 0, 0.0);
        Ok(Response::new(Empty {}))
    }

    async fn step(&self, req: Request<StepRequest>) -> Result<Response<StepReply>, Status> {
        let r = req.into_inner();
        let entry = self.store.get(r.sim_id)?;
        let mut sim = entry.lock().unwrap();
        sim.scheduler_mode = SchedulerMode::SingleThreaded;
        sim.run_auto(r.until_time);
        let spikes = sim.spike_log.iter()
            .map(|&(t, nid)| Spike { time: t, neuron: nid as u32 })
            .collect();
        Ok(Response::new(StepReply { spikes }))
    }

    async fn get_voltages(&self, req: Request<Handle>) -> Result<Response<VoltageReply>, Status> {
        let entry = self.store.get(req.into_inner().id)?;
        let sim = entry.lock().unwrap();
        let volts = (0..sim.neurons.len()).map(|i| sim.neurons.read_v(i)).collect();
        Ok(Response::new(VoltageReply { volts }))
    }

    // ── NEW RPCS ─────────────────────────────────────────────────────────────

    async fn get_spike_count(&self, req: Request<Handle>) -> Result<Response<CountReply>, Status> {
        let entry = self.store.get(req.into_inner().id)?;
        let sim   = entry.lock().unwrap();
        Ok(Response::new(CountReply { count: sim.spike_log.len() as i32 }))
    }

    async fn clear_spikes(&self, req: Request<Handle>) -> Result<Response<Empty>, Status> {
        let entry = self.store.get(req.into_inner().id)?;
        entry.lock().unwrap().spike_log.clear();
        Ok(Response::new(Empty {}))
    }

    async fn get_time(&self, req: Request<Handle>) -> Result<Response<TimeReply>, Status> {
        let entry = self.store.get(req.into_inner().id)?;
        let t = entry.lock().unwrap().time;
        Ok(Response::new(TimeReply { time: t }))
    }

    async fn save_checkpoint(&self, req: Request<CheckpointRequest>) -> Result<Response<Empty>, Status> {
        let r     = req.into_inner();
        let entry = self.store.get(r.sim_id)?;
        let sim   = entry.lock().unwrap();
        let hash  = format!("{}.sha256", r.path);
        sim.save_state(&r.path, &hash)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(Empty {}))
    }

    async fn stream_spikes(&self, req: Request<Handle>) -> Result<Response<Self::StreamSpikesStream>, Status> {
        let entry    = self.store.get(req.into_inner().id)?;
        let snapshot = entry.lock().unwrap().spike_log.clone();
        let stream   = stream::iter(snapshot.into_iter().map(|(t, nid)| {
            Ok(Spike { time: t, neuron: nid as u32 })
        }));
        Ok(Response::new(Box::pin(stream)))
    }
}