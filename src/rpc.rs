use futures::{Stream, stream};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status};

use crate::neurons::LifNeuron;
use crate::simulation::{SchedulerMode, Simulation};
use crate::synapse::Synapse;

pub mod pb {
    tonic::include_proto!("neurosim");
}
use pb::neuro_sim_server::NeuroSim;
use pb::*;

#[derive(Default)]
pub struct SimStore {
    sims: Mutex<Vec<Arc<Mutex<Simulation>>>>,
}

impl SimStore {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn create(&self, sim: Simulation) -> u64 {
        let mut sims = self.sims.lock().unwrap();
        sims.push(Arc::new(Mutex::new(sim)));
        (sims.len() - 1) as u64
    }

    #[allow(clippy::result_large_err)]
    pub fn get(&self, id: u64) -> Result<Arc<Mutex<Simulation>>, Status> {
        self.sims
            .lock()
            .unwrap()
            .get(id as usize)
            .cloned()
            .ok_or_else(|| Status::not_found("invalid sim id"))
    }
}

pub struct RpcService {
    store: Arc<SimStore>,
}
impl RpcService {
    pub fn new(store: Arc<SimStore>) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl NeuroSim for RpcService {
    type StreamSpikesStream = Pin<Box<dyn Stream<Item = Result<Spike, Status>> + Send>>;

    async fn create(&self, req: Request<SimConfig>) -> Result<Response<Handle>, Status> {
        let cfg = req.into_inner();
        let neurons = LifNeuron::new(cfg.n_neurons as usize, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
        let mut sim = Simulation::new_with_neurons(
            neurons,
            Synapse::new(),
            1.0,
            cfg.seed,
            cfg.n_threads as usize,
        );
        sim.scheduler_mode = match cfg.scheduler {
            1 => SchedulerMode::Deterministic {
                n_threads: cfg.n_threads as usize,
            },
            _ => SchedulerMode::SingleThreaded,
        };
        Ok(Response::new(Handle {
            id: self.store.create(sim),
        }))
    }

    async fn free(&self, req: Request<Handle>) -> Result<Response<Empty>, Status> {
        let id = req.into_inner().id;
        if let Some(slot) = self.store.sims.lock().unwrap().get_mut(id as usize) {
            let n = LifNeuron::new(0, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
            *slot = Arc::new(Mutex::new(Simulation::new_with_neurons(
                n,
                Synapse::new(),
                1.0,
                0,
                1,
            )));
        }
        Ok(Response::new(Empty {}))
    }

    async fn push(&self, req: Request<InputEvent>) -> Result<Response<Empty>, Status> {
        let ev = req.into_inner();
        self.store.get(ev.sim_id)?.lock().unwrap().push_event(
            ev.time,
            ev.neuron as usize,
            ev.weight,
            0,
            0.0,
        );
        Ok(Response::new(Empty {}))
    }

    async fn step(&self, req: Request<StepRequest>) -> Result<Response<StepReply>, Status> {
        let r = req.into_inner();
        let entry = self.store.get(r.sim_id)?;
        let mut sim = entry.lock().unwrap();
        sim.scheduler_mode = SchedulerMode::SingleThreaded;
        sim.run_auto(r.until_time);
        let spikes = sim
            .spike_log
            .iter()
            .map(|&(t, n)| Spike {
                time: t,
                neuron: n as u32,
            })
            .collect();
        Ok(Response::new(StepReply { spikes }))
    }

    async fn get_voltages(&self, req: Request<Handle>) -> Result<Response<VoltageReply>, Status> {
        let req = req.into_inner();
        let sim_arc = self.store.get(req.id)?;
        let sim = sim_arc.lock().unwrap();
        Ok(Response::new(VoltageReply {
            volts: sim.neurons.snapshot_v(),
        }))
    }

    async fn get_spike_count(&self, req: Request<Handle>) -> Result<Response<CountReply>, Status> {
        let req = req.into_inner();
        let sim_arc = self.store.get(req.id)?;
        let sim = sim_arc.lock().unwrap();
        Ok(Response::new(CountReply {
            count: sim.spike_log.len() as i32,
        }))
    }

    async fn clear_spikes(&self, req: Request<Handle>) -> Result<Response<Empty>, Status> {
        self.store
            .get(req.into_inner().id)?
            .lock()
            .unwrap()
            .spike_log
            .clear();
        Ok(Response::new(Empty {}))
    }

    async fn get_time(&self, req: Request<Handle>) -> Result<Response<TimeReply>, Status> {
        let t = self.store.get(req.into_inner().id)?.lock().unwrap().time;
        Ok(Response::new(TimeReply { time: t }))
    }

    async fn save_checkpoint(
        &self,
        req: Request<CheckpointRequest>,
    ) -> Result<Response<Empty>, Status> {
        let r = req.into_inner();
        let sim_arc = self.store.get(r.sim_id)?;
        let sim = sim_arc.lock().unwrap();
        sim.save_state(&r.path, &format!("{}.sha256", r.path))
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(Empty {}))
    }

    async fn stream_spikes(
        &self,
        req: Request<Handle>,
    ) -> Result<Response<Self::StreamSpikesStream>, Status> {
        let snapshot = self
            .store
            .get(req.into_inner().id)?
            .lock()
            .unwrap()
            .spike_log
            .clone();

        #[allow(clippy::result_large_err)]
        let s = stream::iter(snapshot.into_iter().map(|(t, n)| {
            Ok(Spike {
                time: t,
                neuron: n as u32,
            })
        }));

        Ok(Response::new(Box::pin(s)))
    }
}
