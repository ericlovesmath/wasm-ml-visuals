use rand::{thread_rng, Rng};
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![1.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = thread_rng();

        let mut res = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }

        res
    }
}

// impl Debug for Matrix {
//     fn fmt(&self, f: &mut Formatter) -> Result {
//         write!(
//             f,
//             "Matrix {{\n{}\n}}",
//             (&self.data)
//                 .into_iter()
//                 .map(|row| "  ".to_string()
//                     + &row
//                         .into_iter()
//                         .map(|value| value.to_string())
//                         .collect::<Vec<String>>()
//                         .join(" "))
//                 .collect::<Vec<String>>()
//                 .join("\n")
//         )
//     }
// }
