use super::linear_classifier::LinearClassifier;
use crate::linear_classifiers::utils::*;
use crate::utils::set_panic_hook;
use wasm_bindgen::prelude::*;

/// Designed to be easily used from JS.
/// Runs a Linear Classifier against the boundary `mx + b` with `n` random sample points.
/// Saves f(-2) and f(2), where `f` is the target function
/// Saves g(-2) and g(2), where `g` is the trained prediction function
#[wasm_bindgen]
pub struct LCBiasVariance {
    pub f_neg: f64,
    pub f_pos: f64,
    pub g_neg: f64,
    pub g_pos: f64,
    line: (f64, f64),
    lc: LinearClassifier,
}

#[wasm_bindgen]
impl LCBiasVariance {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        set_panic_hook();

        let (m, b) = get_random_line();
        LCBiasVariance {
            f_neg: -2.0 * m + b,
            f_pos: 2.0 * m + b,
            g_neg: 0.0,
            g_pos: 0.0,
            line: (m, b),
            lc: LinearClassifier::new(),
        }
    }

    pub fn run(&mut self, n: usize) {
        let (sample, labels) = get_random_sample(n, self.line.0, self.line.1);
        let weights = self.lc.train(&sample, &labels).unwrap();

        // Solves 1 * w[0] + x * w[1] + y * w[2] = 0 for y
        self.g_neg = (-weights[0] + 2.0 * weights[1]) / weights[2];
        self.g_pos = (-weights[0] - 2.0 * weights[1]) / weights[2];
    }
}
