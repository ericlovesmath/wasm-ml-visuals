use crate::linear_classifiers::types::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;

thread_local! {
    pub static RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_entropy());
}

/// Generates `n` sample points using boundary `mx + b`.
/// Uses the standard linear features `(1, x, y)`.
pub fn get_random_sample(n: usize, m: f64, b: f64) -> (Points, Labels) {
    let sample = RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        Points::from_fn(3, n, |r, _| match r {
            0 => 1.0,
            _ => rng.gen_range(-1.0..1.0),
        })
    });
    let labels = Labels::from_fn(n, |_, c| (sample[(2, c)] - sample[(1, c)] * m - b).signum());
    (sample, labels)
}

/// Returns m, b for some random mx + b
pub fn get_random_line() -> (f64, f64) {
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let x_1 = rng.gen_range(-1.0..1.0);
        let x_2 = rng.gen_range(-1.0..1.0);
        let y_1 = rng.gen_range(-1.0..1.0);
        let y_2 = rng.gen_range(-1.0..1.0);

        let m = (y_2 - y_1) / (x_2 - x_1);
        let b = y_1 - m * x_1;
        (m, b)
    })
}
