extern crate nalgebra as na;

use crate::linear_classifiers::types::*;
use crate::utils::set_panic_hook;

const EPSILON: f64 = 1E-6;

/// Linear Classifier for labeled 2D vectors
pub struct LinearClassifier {
    perceptron: Option<Weights>,
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
    pub fn train(&mut self, sample: &Points, labels: &Labels) -> Option<&Weights> {
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
