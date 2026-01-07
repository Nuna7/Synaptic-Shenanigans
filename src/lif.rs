use crossbeam::atomic::AtomicCell;

#[derive(Clone, Copy)]
pub struct NeuronPartition {
    pub start_index: usize,
    pub len: usize,
}

pub trait NeuronPopulation {
    fn len(&self) -> usize;
    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition>;
    fn step_range(&self, input_current: &[f32], start: usize);
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct LifNeuron {
    pub v: Vec<AtomicCell<f32>>,
    pub v_rest: Vec<f32>,
    pub tau_m: Vec<f32>,
    pub v_thresh: Vec<f32>,
    pub r_m: Vec<f32>,
    pub dt: Vec<f32>,
    pub spiked: Vec<AtomicCell<bool>>,
    pub refractory: Vec<AtomicCell<bool>>,
    pub refractory_timer: Vec<AtomicCell<f32>>,
    pub refractory_period: Vec<f32>,
}

impl LifNeuron {
    pub fn new(
        n: usize,
        v_rest: f32,
        v_thresh: f32,
        tau_m: f32,
        r_m: f32,
        dt: f32,
        refract_period: f32,
    ) -> Self {
        Self {
            v: (0..n).map(|_| AtomicCell::new(v_rest)).collect(),
            v_rest: vec![v_rest; n],
            tau_m: vec![tau_m; n],
            v_thresh: vec![v_thresh; n],
            r_m: vec![r_m; n],
            dt: vec![dt; n],
            spiked: (0..n).map(|_| AtomicCell::new(false)).collect(),
            refractory: (0..n).map(|_| AtomicCell::new(false)).collect(),
            refractory_timer: (0..n).map(|_| AtomicCell::new(0.0)).collect(),
            refractory_period: vec![refract_period; n],
        }
    }

    pub fn read_v(&self, idx: usize) -> f32 {
        self.v[idx].load()
    }

    pub fn local_spiked(&self, idx: usize) -> bool {
        self.spiked[idx].load()
    }
}

impl NeuronPopulation for LifNeuron {
    fn len(&self) -> usize {
        self.v.len()
    }

    fn split_indices(&self, chunk: usize) -> Vec<NeuronPartition> {
        let n = self.v.len();
        let num_parts = n.div_ceil(chunk);

        (0..num_parts)
            .map(|p| {
                let start = p * chunk;
                let end = (start + chunk).min(n);
                NeuronPartition {
                    start_index: start,
                    len: end - start,
                }
            })
            .collect()
    }

    fn step_range(&self, input_current: &[f32], start: usize) {
        let len = input_current.len();

        //for local_i in 0..len {
        for (local_i, input_cur) in input_current.iter().enumerate().take(len) {
            let i = start + local_i; // map back to global index            
            let dt_i = self.dt[i];

            self.spiked[i].store(false);

            if self.refractory[i].load() {
                let new_timer = self.refractory_timer[i].load() - dt_i;
                if new_timer <= 0.0 {
                    self.refractory[i].store(false);
                    self.refractory_timer[i].store(0.0);
                } else {
                    self.refractory_timer[i].store(new_timer);
                    self.v[i].store(self.v_rest[i]);
                    continue;
                }
            }


            let v_old = self.v[i].load();
            let dv =
                (-(v_old - self.v_rest[i]) + self.r_m[i] * input_cur)//input_current[local_i]) 
                * (dt_i / self.tau_m[i]);

            let v_new = v_old + dv;
            self.v[i].store(v_new);

            if v_new >= self.v_thresh[i] {
                self.v[i].store(self.v_rest[i]);
                self.spiked[i].store(true);
                self.refractory[i].store(true);
                self.refractory_timer[i].store(self.refractory_period[i]);
            }

        }
    }

}
