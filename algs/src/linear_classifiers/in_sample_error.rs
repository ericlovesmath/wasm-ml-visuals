use super::linear_classifier::LinearClassifier;
use crate::linear_classifiers::utils::get_random_sample;
use crate::utils::set_panic_hook;
use wasm_bindgen::prelude::*;

/// Designed to be easily used from JS.
/// Runs a Linear Classifier against the boundary `M` * x + `B` with `n` sample points.
/// Trains with `n` sample points `RUNS` times, calculating the in-sample error.
/// Evaluates Mean and Standard deviation of in-sample errors
#[wasm_bindgen]
pub struct LCInSampleError {
    pub mean: f64,
    pub std: f64,
    lc: LinearClassifier,
}

const M: f64 = 1.0;
const B: f64 = 0.3;
const RUNS: usize = 300;

#[wasm_bindgen]
impl LCInSampleError {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        set_panic_hook();
        LCInSampleError {
            mean: 0.0,
            std: 0.0,
            lc: LinearClassifier::new(),
        }
    }

    pub fn run(&mut self, n: usize) {
        let errors: Vec<f64> = (0..RUNS)
            .map(|_| {
                let (sample, labels) = get_random_sample(n, M, B);
                self.lc.train(&sample, &labels);
                let errors = self.lc.predict(&sample);
                (0..n).filter(|i| errors[*i] != labels[*i]).count() as f64 / n as f64
            })
            .collect();

        self.mean = errors.iter().sum::<f64>() / RUNS as f64;
        self.std = (errors
            .iter()
            .map(|n| (n - self.mean) * (n - self.mean))
            .sum::<f64>()
            / RUNS as f64)
            .sqrt();
    }
}
