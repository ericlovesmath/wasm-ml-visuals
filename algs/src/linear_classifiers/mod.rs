pub mod in_sample_error;
pub mod linear_classifier;
pub mod nonlinear_features;
pub mod utils;

pub mod types {
    extern crate nalgebra as na;

    use na::{Dyn, OMatrix, U1};

    pub type Labels = OMatrix<f64, U1, Dyn>;
    pub type Weights = OMatrix<f64, Dyn, U1>;
    pub type Points = OMatrix<f64, Dyn, Dyn>;
}
