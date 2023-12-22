use super::linear_classifier::LinearClassifier;
use crate::linear_classifiers::utils::*;
use crate::utils::set_panic_hook;
use wasm_bindgen::prelude::*;

pub enum LCFeatures {
    Linear([f64; 2]),    // [1, x, y]
    Quadratic([f64; 5]), // [1, x, y, xy, x^2, y^2]
}

#[wasm_bindgen]
pub struct LCNonlinear {
    f: [f64; 201],
    g: [f64; 201],
    features: LCFeatures,
    lc: LinearClassifier,
}

#[wasm_bindgen]
impl LCNonlinear {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        set_panic_hook();

        let (m, b) = get_random_line();
        LCNonlinear {
            f: [0.0; 201],
            g: [0.0; 201],
            features: LCFeatures::Linear([m, b]),
            lc: LinearClassifier::new(),
        }
    }

    pub fn set_features(&mut self, mode: usize) -> *const f64 {
        match mode {
            0 => {
                let (m, b) = get_random_line();
                self.features = LCFeatures::Linear([m, b]);
                for i in 0..=200 {
                    let x = (i as f64) * 0.01 - 1.0;
                    self.f[i] = m * x + b;
                }
            }
            _ => panic!("Unexpected mode {mode} in LCNonlinear::set_features()"),
        }
        self.f.as_ptr()
    }

    pub fn get_prediction(&mut self, n: usize) -> *const f64 {
        let g = match self.features {
            LCFeatures::Linear([m, b]) => {
                let (sample, labels) = get_random_sample(n, m, b);
                let weights = self.lc.train(&sample, &labels).unwrap();
                |x: f64| (-weights[0] - x * weights[1]) / weights[2]
            }
            _ => unreachable!(),
        };
        for i in 0..=200 {
            let x = (i as f64) * 0.01 - 1.0;
            self.g[i] = g(x);
        }
        self.g.as_ptr()
    }
}
