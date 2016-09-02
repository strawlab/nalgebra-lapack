# Change Log

## [0.3.0] - 2016-09-06

* Documentation is hosted at https://docs.rs/nalgebra-lapack/
* Updated `nalgebra` to 0.10.
* Rename traits `HasSVD` to `SVD` and `HasEigensystem` to `Eigensystem`.
* Added `Solve` trait for solving a linear matrix equation.
* Added `Inverse` for computing the multiplicative inverse of a matrix.
* Added `Cholesky` for decomposing a positive-definite matrix.
* The `Eigensystem` and `SVD` traits are now generic over types. The
  associated types have been removed.
