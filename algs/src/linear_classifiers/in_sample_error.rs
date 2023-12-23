use super::linear_classifier::LinearClassifier;
use crate::linear_classifiers::types::*;
use crate::linear_classifiers::utils::get_random_sample;
use crate::utils::set_panic_hook;
use wasm_bindgen::prelude::*;

/// Designed to be easily used from JS.
/// Runs a Linear Classifier against a preset linear boundary with `n` sample points.
/// Trains with `n` sample points `RUNS` times, calculating the in-sample error.
/// Evaluates Mean and Standard deviation of in-sample errors
#[wasm_bindgen]
pub struct LCInSampleError {
    pub mean: f64,
    pub std: f64,
    lc: LinearClassifier,
}

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
        let f = Weights::from_vec(vec![-0.05, 0.2, 0.2]);
        let errors: Vec<f64> = (0..RUNS)
            .map(|_| {
                let (sample, labels) =
                    get_random_sample(n, &f, |x, y| Weights::from_vec(vec![1.0, x, y]));
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
