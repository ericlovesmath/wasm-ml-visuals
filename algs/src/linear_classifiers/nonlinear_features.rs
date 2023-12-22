use super::linear_classifier::LinearClassifier;
use super::types::*;
use crate::linear_classifiers::utils::*;
use crate::utils::set_panic_hook;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub enum LCFeatures {
    Linear,    // [1, x, y]
    Quadratic, // [1, x, y, xy, x^2, y^2]
}

#[wasm_bindgen]
pub struct LCNonlinear {
    f: [f64; 961],
    g: [f64; 961],
    weights: Weights,
    features: fn(f64, f64) -> Weights,
    lc: LinearClassifier,
}

#[wasm_bindgen]
impl LCNonlinear {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        set_panic_hook();

        LCNonlinear {
            f: [0.0; 961],
            g: [0.0; 961],
            weights: Weights::zeros(1),
            features: |_, _| Weights::zeros(1),
            lc: LinearClassifier::new(),
        }
    }

    pub fn set_features(&mut self, mode: LCFeatures) {
        match mode {
            LCFeatures::Linear => {
                self.weights = get_random_line();
                self.features = |x, y| Weights::from_vec(vec![1.0, x, y]);
            }
            LCFeatures::Quadratic => {
                self.weights = get_random_quadratic();
                self.features = |x, y| Weights::from_vec(vec![1.0, x, y, x * y, x * x, y * y]);
            }
            _ => panic!("Unexpected mode in LCNonlinear::set_features()"),
        };
        for i in 0..=30 {
            for j in 0..=30 {
                let (x, y) = ((i as f64) / 10.0 - 1.5, (j as f64) / 10.0 - 1.5);
                self.f[i * 31 + j] = ((self.features)(x, y)).dot(&self.weights);
            }
        }
    }

    pub fn get_prediction(&mut self, n: usize) {
        let (sample, labels) = get_random_sample_test(n, &self.weights, self.features);
        let weights = self.lc.train(&sample, &labels).unwrap();
        for i in 0..=30 {
            for j in 0..=30 {
                let (x, y) = ((i as f64) / 10.0 - 1.5, (j as f64) / 10.0 - 1.5);
                self.g[i * 31 + j] = ((self.features)(x, y)).dot(weights);
            }
        }
    }

    pub fn get_f(&self) -> *const f64 {
        self.f.as_ptr()
    }

    pub fn get_g(&self) -> *const f64 {
        self.g.as_ptr()
    }
}
