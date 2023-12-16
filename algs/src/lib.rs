extern crate nalgebra as na;

use js_sys::Array;
use na::{Dyn, OMatrix, U1, U3};
use wasm_bindgen::prelude::*;

type Labels = OMatrix<f64, U1, Dyn>;
type Perceptron = OMatrix<f64, U3, U1>;
type Points = OMatrix<f64, U3, Dyn>;

macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

fn hypothesis(x: f64, y: f64) -> f64 {
    let (m, b) = (1.0, -0.5);
    if y > m * x + b {
        1.0
    } else {
        -1.0
    }
}

#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[wasm_bindgen]
pub struct LinearClassifier {
    n: usize,
    sample: Points,
    target: Labels,
    perceptron: Option<Perceptron>,
    prediction: Option<Labels>,
}

#[wasm_bindgen]
impl LinearClassifier {
    pub fn new() -> LinearClassifier {
        LinearClassifier {
            n: 1,
            sample: Points::from_fn(1, |_, _| 1.0),
            target: Labels::from_fn(1, |_, _| 1.0),
            perceptron: None,
            prediction: None,
        }
    }

    pub fn init(&mut self, n: usize, xs: Array, ys: Array) {
        let sample = Points::from_fn(n, |r, c| match r {
            0 => 1.0,
            1 => xs.get(c as u32).as_f64().unwrap(),
            2 => ys.get(c as u32).as_f64().unwrap(),
            _ => unreachable!(),
        });

        let target = Labels::from_fn(sample.ncols(), |_, i| {
            hypothesis(sample[(1, i)], sample[(2, i)])
        });

        self.n = n;
        self.sample = sample;
        self.target = target;
    }

    fn test_sample(&self) -> Labels {
        match self.perceptron {
            None => panic!("LinearClassifier::test_sample called before training"),
            Some(perceptron) => Labels::from_fn(self.n, |_, i| {
                if self.sample.column(i).dot(&perceptron) > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }),
        }
    }

    pub fn train(&mut self) {
        let eps = 1E-6;

        self.perceptron =
            Some(self.sample.transpose().pseudo_inverse(eps).unwrap() * self.target.transpose());
        self.prediction = Some(self.test_sample());
    }

    pub fn in_sample_error(&self) -> f64 {
        self.target
            .iter()
            .zip(self.prediction.as_ref().unwrap().iter())
            .filter(|(&a, &b)| a != b)
            .count() as f64
            / self.n as f64
    }

    pub fn get_target(&mut self) -> *const f64 {
        self.target.as_ptr()
    }
}

impl Default for LinearClassifier {
    fn default() -> Self {
        Self::new()
    }
}
