extern crate nalgebra as na;

use js_sys::Math::random;
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

fn random_points(n: usize) -> Points {
    Points::from_fn(n, |r, _| match r {
        0 => 1.0,
        1 => random() * 2.0 - 1.0,
        2 => random() * 2.0 - 1.0,
        _ => unreachable!(),
    })
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
pub fn linear_classification(n: usize) {
    let eps = 1E-6;

    let sample: Points = random_points(n);
    let target: Labels = hypothesis(&sample);
    let weights: Perceptron = sample.transpose().pseudo_inverse(eps).unwrap() * &target.transpose();
    let pred: Labels = apply_perceptron(&sample, &weights);

    let in_sample_mislabel: usize = target
        .iter()
        .zip(pred.iter())
        .filter(|(&a, &b)| a != b)
        .count();

    log!("{}", in_sample_mislabel as f64 / n as f64);
}
