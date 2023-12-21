extern crate nalgebra as na;

use crate::utils::set_panic_hook;
use js_sys::Math::random;
use na::{Dyn, OMatrix, U1};

type Labels = OMatrix<f64, U1, Dyn>;
type Perceptron = OMatrix<f64, Dyn, U1>;
type Points = OMatrix<f64, Dyn, Dyn>;

const EPSILON: f64 = 1E-6;

/// Linear Classifier for labeled 2D vectors
pub struct LinearClassifier {
    perceptron: Option<Perceptron>,
}

impl LinearClassifier {
    pub fn new() -> Self {
        set_panic_hook();
        LinearClassifier { perceptron: None }
    }

    /// Trains labeled sample points with Linear Classifier.
    /// Returns trained perceptron / weights.
    /// * `sample` - Flattened training data
    /// * `labels` - Labels for training data (-1.0 or 1.0)
    pub fn train(&mut self, sample: &Points, labels: &Labels) -> Option<&Perceptron> {
        self.perceptron =
            Some(sample.transpose().pseudo_inverse(EPSILON).unwrap() * labels.transpose());
        self.perceptron.as_ref()
    }

    /// Returns labels for sample using the trained model.
    /// Panics if `LinearClassifier::train()` was not run prior>
    /// * `sample` - Testing data
    pub fn predict(&mut self, sample: &Points) -> Labels {
        match &self.perceptron {
            None => panic!("LinearClassifier::predict() ran before training"),
            Some(weights) => Labels::from_iterator(
                sample.ncols(),
                sample.column_iter().map(|col| col.dot(weights).signum()),
            ),
        }
    }
}

pub fn get_random_sample(n: usize, m: f64, b: f64) -> (Points, Labels) {
    let mut sample: Vec<f64> = vec![1.0; n];
    for _ in 0..(2 * n) {
        sample.push(random() * 2.0 - 1.0);
    }
    let labels: Vec<f64> = (0..n)
        .map(|i| (sample[2 * n + i] - sample[n + i] * m + b).signum())
        .collect();

    (
        Points::from_row_slice(3, n, sample.as_slice()),
        Labels::from_row_slice(labels.as_slice()),
    )
}

/// Returns m, b for some random mx + b
pub fn random_line() -> (f64, f64) {
    let x_1 = random() * 2.0 - 1.0;
    let x_2 = random() * 2.0 - 1.0;
    let y_1 = random() * 2.0 - 1.0;
    let y_2 = random() * 2.0 - 1.0;

    let m = (y_2 - y_1) / (x_2 - x_1);
    let b = y_1 - m * x_1;
    (m, b)
}
