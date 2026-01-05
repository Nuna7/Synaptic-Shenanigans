use tonic::{Request, Response, Status};
use std::sync::{Arc, Mutex};

use crate::simulation::{Simulation, SchedulerMode};
use crate::lif::{LifNeuron, NeuronPopulation};
use crate::synapse::Synapse;

use std::pin::Pin;
use futures::Stream;
use futures::stream;

// bring generated proto module
pub mod pb {
    tonic::include_proto!("neurosim");
}

use pb::*;
use pb::neuro_sim_server::{NeuroSim, NeuroSimServer};
use pb::Empty;

/// Simple simulation store
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

pub struct RpcService {
    store: Arc<SimStore>,
}

impl RpcService {
    pub fn new(store: Arc<SimStore>) -> Self { Self { store } }
}


#[tonic::async_trait]
impl NeuroSim for RpcService {

    type StreamSpikesStream = Pin<Box<dyn Stream<Item = Result<Spike, Status>> + Send>>;

    async fn create(
        &self,
        request: Request<SimConfig>,
    ) -> Result<Response<Handle>, Status> {

        let cfg = request.into_inner();

        let neurons = LifNeuron::new(
            cfg.n_neurons as usize,
            -65.0, -50.0, 20.0, 1.0, 1.0, 5.0,
        );

        let syn = Synapse::new();

        let sim = Simulation::new_with_seed(
            neurons,
            syn,
            1.0,
            cfg.seed,
            cfg.n_threads as usize,
        );

        let id = self.store.create(sim);

        Ok(Response::new(Handle { id }))
    }

    async fn push(
        &self,
        request: Request<InputEvent>,
    ) -> Result<Response<Empty>, Status> {

        let ev = request.into_inner();

        let entry = self.store.get(ev.sim_id)?;
        let mut sim = entry.lock().unwrap();

        sim.push_event(ev.time, ev.neuron as usize, ev.weight, 0, 0.0);

        Ok(Response::new(Empty {}))
    }

    async fn step(
        &self,
        request: Request<StepRequest>,
    ) -> Result<Response<StepReply>, Status> {

        let req = request.into_inner();

        let entry = self.store.get(req.sim_id)?;
        let mut sim = entry.lock().unwrap();

        sim.scheduler_mode = SchedulerMode::SingleThreaded;
        sim.run_auto(req.until_time);

        let mut reply = StepReply::default();

        for (t, nid) in &sim.spike_log {
            reply.spikes.push(Spike {
                time: *t,
                neuron: *nid as u32,
            });
        }

        Ok(Response::new(reply))
    }

    async fn get_voltages(
        &self,
        request: Request<Handle>,
    ) -> Result<Response<VoltageReply>, Status> {

        let req = request.into_inner();

        let entry = self.store.get(req.id)?;
        let sim = entry.lock().unwrap();

        let mut reply = VoltageReply::default();

        for i in 0..sim.neurons.len() {
            reply.volts.push(sim.neurons.read_v(i));
        }

        Ok(Response::new(reply))
    }

        async fn free(
        &self,
        request: Request<Handle>,
    ) -> Result<Response<Empty>, Status> {

        let req = request.into_inner();

        let mut sims = self.store.sims.lock().unwrap();
        if let Some(slot) = sims.get_mut(req.id as usize) {
            // Replace with a zero-neuron dummy sim to logically free it.
            let neurons = LifNeuron::new(0, -65.0, -50.0, 20.0, 1.0, 1.0, 5.0);
            let syn = Synapse::new();
            let sim = Simulation::new_with_seed(neurons, syn, 1.0, 0, 1);
            *slot = Arc::new(Mutex::new(sim));
        }

        Ok(Response::new(Empty {}))
    }

    async fn stream_spikes(
        &self,
        request: Request<Handle>,
    ) -> Result<Response<Self::StreamSpikesStream>, Status> {

        let id = request.into_inner().id;

        let entry = self.store.get(id)?;
        let sim = entry.lock().unwrap();

        // clone historical spike log
        let snapshot: Vec<(f32, usize)> = sim.spike_log.clone();
        drop(sim); // release lock early

        let out = stream::iter(
            snapshot.into_iter().map(|(t, nid)| {
                Ok(Spike {
                    time: t,
                    neuron: nid as u32,
                })
            })
        );

        Ok(Response::new(Box::pin(out)))
    }
}
