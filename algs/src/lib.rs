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

fn hypothesis(points: &Points) -> Labels {
    let (m, b) = (1.0, -0.5);
    Labels::from_fn(points.ncols(), |_, i| {
        if points[(2, i)] > m * points[(1, i)] + b {
            1.0
        } else {
            -1.0
        }
    })
}

fn apply_perceptron(points: &Points, perc: &Perceptron) -> Labels {
    Labels::from_fn(points.ncols(), |_, i| {
        if points.column(i).dot(perc) > 0.0 {
            1.0
        } else {
            -1.0
        }
    })
}

#[wasm_bindgen]
pub struct SimResult {
    pub error_in: f64, // In Sample Error
}

#[wasm_bindgen]
pub fn linear_classification(n: usize, xs: Array, ys: Array) -> f64 {
    let eps = 1E-6;

    // let sample: Points = random_points(n);
    let sample = Points::from_fn(n, |r, c| match r {
        0 => 1.0,
        1 => xs.get(c as u32).as_f64().unwrap(),
        2 => ys.get(c as u32).as_f64().unwrap(),
        _ => unreachable!(),
    });

    let target: Labels = hypothesis(&sample);
    let weights: Perceptron = sample.transpose().pseudo_inverse(eps).unwrap() * &target.transpose();
    let pred: Labels = apply_perceptron(&sample, &weights);

    // In Sample Error
    target
        .iter()
        .zip(pred.iter())
        .filter(|(&a, &b)| a != b)
        .count() as f64
        / n as f64
}
