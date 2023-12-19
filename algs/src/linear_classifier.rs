extern crate nalgebra as na;

use crate::utils::set_panic_hook;
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
    perceptron: Perceptron,
    prediction: Labels,
}

#[wasm_bindgen]
impl LinearClassifier {
    pub fn new() -> LinearClassifier {
        set_panic_hook();
        LinearClassifier {
            sample: Points::zeros(1),
            perceptron: Perceptron::zeros(),
            prediction: Labels::zeros(1),
        }
    }

    /// Trains `n` labeled `sample` points.
    /// * `n` - Number of `sample` points or length of `labels`
    /// * `sample` - Length `2n`, Flattened training data, not including bias
    /// * `labels` - Length `n`, labels for training data (-1.0 or 1.0)
    pub fn train(&mut self, n: usize, sample: &[f64], labels: &[f64]) {
        let eps = 1E-6;
        let sample = LinearClassifier::parse_sample(n, sample);
        let labels = Labels::from_row_slice(labels);

        self.perceptron = sample.transpose().pseudo_inverse(eps).unwrap() * labels.transpose();
        self.sample = sample;
    }

    /// Labels `n` `sample` points with the trained model.
    /// Returns prediction of `0`s if not model is not trained
    /// * `n` - Number of `sample` points
    /// * `sample` - Length `2n`, Flattened testing data, not including bias
    pub fn predict(&mut self, n: usize, sample: &[f64]) -> *const f64 {
        let sample = LinearClassifier::parse_sample(n, sample);
        self.prediction = Labels::from_fn(n, |_, i| {
            if sample.column(i).dot(&self.perceptron) > 0.0 {
                1.0
            } else {
                -1.0
            }
        });
        self.prediction.as_ptr()
    }

    /// Returns pointer to perceptron, includes bias.
    /// Returns perceptron of `0`s if model not trained.
    /// Call from `js` by indexing `wasm_memory()`.
    pub fn get_weights(&self) -> *const f64 {
        self.perceptron.as_ptr()
    }
}

impl LinearClassifier {
    /// Unflattens array, converting to `DMatrix`
    fn parse_sample(n: usize, sample: &[f64]) -> Points {
        Points::from_fn(n, |r, c| match r {
            0 => 1.0,
            r => sample[n * (r - 1) + c],
        })
    }
}

impl Default for LinearClassifier {
    fn default() -> Self {
        Self::new()
    }
}
