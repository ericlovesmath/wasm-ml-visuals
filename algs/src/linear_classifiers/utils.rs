use crate::linear_classifiers::types::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Generates `n` sample points using boundary `mx + b`.
/// Uses the standard linear features `(1, x, y)`.
pub fn get_random_sample(n: usize, m: f64, b: f64) -> (Points, Labels) {
    let mut rng = SmallRng::from_entropy();
    let sample = Points::from_fn(3, n, |r, _| match r {
        0 => 1.0,
        _ => rng.gen_range(-1.0..1.0),
    });
    let labels = Labels::from_fn(n, |_, c| (sample[(2, c)] - sample[(1, c)] * m - b).signum());
    (sample, labels)
}

pub fn get_random_line() -> Weights {
    let mut rng = SmallRng::from_entropy();
    Weights::from_vec(vec![
        rng.gen_range(-0.2..0.2),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    ])
}

pub fn get_random_quadratic() -> Weights {
    let mut rng = SmallRng::from_entropy();
    Weights::from_vec(vec![
        rng.gen_range(-0.1..0.1),
        rng.gen_range(-0.1..0.1),
        rng.gen_range(-0.1..0.1),
        rng.gen_range(-0.1..0.1),
        rng.gen_range(-0.1..0.1),
        rng.gen_range(-0.1..0.1),
    ])
}

pub fn get_random_sample_test(
    n: usize,
    f: &Weights,
    features: fn(f64, f64) -> Weights,
) -> (Points, Labels) {
    let mut rng = SmallRng::from_entropy();
    let x = Labels::from_fn(n, |_, _| rng.gen_range(-1.0..1.0));
    let y = Labels::from_fn(n, |_, _| rng.gen_range(-1.0..1.0));

    let sample = Points::from_fn(f.len(), n, |r, c| features(x[c], y[c])[r]);
    let labels = Labels::from_iterator(n, sample.column_iter().map(|col| col.dot(f).signum()));

    (sample, labels)
}
