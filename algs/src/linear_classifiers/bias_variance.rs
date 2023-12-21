use super::linear_classifier::{get_random_sample, random_line, LinearClassifier};
use crate::utils::set_panic_hook;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LCBiasVariance {
    pub m: f64,
    pub b: f64,
    pub y_neg_two: f64,
    pub y_pos_two: f64,
    lc: LinearClassifier,
}

#[wasm_bindgen]
impl LCBiasVariance {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        set_panic_hook();

        let (m, b) = random_line();
        LCBiasVariance {
            m,
            b,
            y_neg_two: 0.0,
            y_pos_two: 0.0,
            lc: LinearClassifier::new(),
        }
    }

    pub fn run(&mut self, n: usize, m: f64, b: f64) {
        let (sample, labels) = get_random_sample(n, m, b);
        let weights = self.lc.train(&sample, &labels).unwrap();
        self.y_neg_two = (weights[0] + 2.0 * weights[1]) / weights[2];
        self.y_pos_two = (weights[0] - 2.0 * weights[1]) / weights[2];
    }
}
