extern crate nalgebra as na;

use crate::utils::set_panic_hook;
use js_sys::Array;
use na::{Dyn, OMatrix, U1, U3};
use wasm_bindgen::prelude::*;

type Labels = OMatrix<f64, U1, Dyn>;
type Perceptron = OMatrix<f64, U3, U1>;
type Points = OMatrix<f64, U3, Dyn>;

/// Linear Classifier for labeled 2D vectors
/// Intended to be called from `JS`, compiled to `WASM`
#[wasm_bindgen]
pub struct LinearClassifier {
    sample: Points,
    perceptron: Option<Perceptron>,
    prediction: Labels,
}

#[wasm_bindgen]
impl LinearClassifier {
    pub fn new() -> LinearClassifier {
        set_panic_hook();
        LinearClassifier {
            sample: Points::zeros(1),
            perceptron: None,
            prediction: Labels::zeros(1),
        }
    }

    pub fn train(&mut self, xs: Array, ys: Array, target: Array) {
        let eps = 1E-6;

        let (n, sample) = LinearClassifier::parse_sample(xs, ys);
        let target = Labels::from_fn(n, |_, c| target.get(c as u32).as_f64().unwrap());

        self.sample = sample;
        self.perceptron =
            Some(self.sample.transpose().pseudo_inverse(eps).unwrap() * target.transpose());
    }

    pub fn predict(&mut self, xs: Array, ys: Array) -> *const f64 {
        let (n, sample) = LinearClassifier::parse_sample(xs, ys);

        self.prediction = match self.perceptron {
            None => panic!("LinearClassifier::predict() called before train()"),
            Some(perceptron) => Labels::from_fn(n, |_, i| {
                if sample.column(i).dot(&perceptron) > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }),
        };
        self.prediction.as_ptr()
    }

    pub fn get_weights(&self) -> *const f64 {
        match self.perceptron {
            None => panic!("LinearClassifier::get_weights() called before train()"),
            Some(perceptron) => perceptron.as_ptr(),
        }
    }
}

impl LinearClassifier {
    fn parse_sample(xs: Array, ys: Array) -> (usize, Points) {
        let n = xs.length() as usize;
        let sample = Points::from_fn(n, |r, c| match r {
            0 => 1.0,
            1 => xs.get(c as u32).as_f64().unwrap(),
            2 => ys.get(c as u32).as_f64().unwrap(),
            _ => unreachable!(),
        });
        (n, sample)
    }
}

impl Default for LinearClassifier {
    fn default() -> Self {
        Self::new()
    }
}
