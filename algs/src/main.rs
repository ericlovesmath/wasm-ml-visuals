extern crate nalgebra as na;

use na::{DVector, Dyn, OMatrix, U1, U3};
use rand::{thread_rng, Rng};

type Labels = OMatrix<f64, U1, Dyn>;
type Perceptron = OMatrix<f64, U3, U1>;
type Points = OMatrix<f64, U3, Dyn>;

macro_rules! print_matrix {
    ($matrix:expr) => {
        for i in 0..$matrix.nrows() {
            for j in 0..$matrix.ncols() {
                print!("{} ", $matrix[(i, j)]);
            }
            println!();
        }
    };
}

fn random_points(n: usize) -> Points {
    let mut rng = thread_rng();
    Points::from_fn(n, |r, _| match r {
        0 => 1.0,
        1 => rng.gen::<f64>() * 2.0 - 1.0,
        2 => rng.gen::<f64>() * 2.0 - 1.0,
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

fn main() {
    let eps = 1E-6;
    let n = 1000;
    let sample: Points = random_points(n);
    let target = hypothesis(&sample).transpose();

    let weights = sample.transpose().pseudo_inverse(eps).unwrap() * &target;

    let pred = apply_perceptron(&sample, &weights).transpose();

    let in_sample_mislabel: usize = target
        .iter()
        .zip(pred.iter())
        .filter(|(&a, &b)| a != b)
        .count();

    println!("{}", in_sample_mislabel as f64 / n as f64);
}
