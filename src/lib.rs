/*!
linear algebra operations for [nalgebra][1] matrices using [LAPACK][2].

Functions to compute the singular value decomposition (SVD) and eigensystem are
implemented.

# Examples

```rust
extern crate nalgebra_lapack;
extern crate nalgebra as na;

use nalgebra_lapack::{SVD, Eigensystem};

fn main() {
    // Create an input matrix
    let m = na::DMatrix::from_row_vector(3,5,&[
        -1.01,   0.86,  -4.60,   3.31,  -4.81,
         3.98,   0.53,  -7.04,   5.29,   3.55,
         3.30,   8.26,  -3.89,   8.20,  -1.51]);

    // Now perform SVD
    let (u,s,vt) = m.svd().unwrap();
    println!("u {:?}",u);
    println!("s {:?}",s);
    println!("vt {:?}",vt);

    // Create an input matrix
    let m = na::DMatrix::from_row_vector(2,2,&[
        2.0, 1.0,
        1.0, 2.0]);

    // Now get the eigensystem
    let (vals, vecs) = m.eigensystem().unwrap();
    println!("eigenvalues {:?}",vals);
    println!("eigenvectors {:?}",vecs);

}
```

[1]: https://crates.io/crates/nalgebra
[2]: https://crates.io/crates/lapack
*/

extern crate nalgebra;
extern crate lapack;
extern crate num;

use std::error::Error;
use std::fmt::{self, Display};
use num::complex::Complex;
use num::{Zero, One};

use nalgebra::{DMatrix, DVector, Iterable, Eye, Row};

/// A type for which eigenvalues and eigenvectors can be computed.
pub trait Eigensystem<N> {

    /// `eigensystem` computes eigenvalues and right eigenvectors
    ///
    /// Because the input matrix may be overwritten or destroyed, it is consumed.
    ///
    /// # Returns
    ///
    /// * `eigen_values` - The eigenvalues, normalized to have Euclidean norm equal to 1 and largest component real.
    /// * `right_eigen_vectors` - The right eigenvectors. They are contained as columns of this matrix.
    fn eigensystem(mut self) -> NalgebraLapackResult<(DVector<Complex<N>>, DMatrix<Complex<N>>)>;
}

/// A type for which a singular value decomposition can be computed.
pub trait SVD<V,M> {

    /// `svd` computes the singular value decomposition (SVD). Returns full
    /// matrices.
    ///
    /// Because the input matrix may be overwritten or destroyed, it is consumed.
    ///
    /// # Returns
    ///
    /// * `u` - The left-singular vectors.
    /// * `s` - The singular values.
    /// * `vt` - The right-singular vectors.
    fn svd(mut self) -> NalgebraLapackResult<(DMatrix<M>, DVector<V>, DMatrix<M>)>;
}

pub trait Solve<N> : Row<DVector<N>> + Sized where N: Copy + Clone + Zero + One {
    fn solve(self, b: DMatrix<N>) -> NalgebraLapackResult<DMatrix<N>>;
}

/// A type for which the inverse can be computed.
pub trait Inverse<N> : HasSolve<N> where N: Copy + Clone + Zero + One {
    /// `inv` computes the (multiplicative) inverse.
    fn inv(self) -> NalgebraLapackResult<DMatrix<N>> {
        let n = self.nrows();
        let b = DMatrix::new_identity(n);
        self.solve(b)
    }
}

/// A type for which the Cholesky decomposition can be computed.
pub trait Cholesky<N> where N: Copy {
    /// `cholesky` computes the cholesky decomposition of hermitian positive-definite matrices.
    fn cholesky(self) -> NalgebraLapackResult<DMatrix<N>>;
}

#[derive(Debug)]
pub struct NalgebraLapackError {
  pub desc: String,
}

pub type NalgebraLapackResult<T> = Result<T, NalgebraLapackError>;

impl Error for NalgebraLapackError {
  fn description(&self) -> &str { &self.desc }
}

impl Display for NalgebraLapackError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    Display::fmt(&self.desc, f)
  }
}

impl From<String> for NalgebraLapackError {
  fn from(err: String) -> NalgebraLapackError {
    NalgebraLapackError { desc: format!(
                  "String ({})",
                  err
                  ),
    }
  }
}

macro_rules! eigensystem_impl(
    ($t: ty, $lapack_func: path) => (
        impl Eigensystem<$t> for DMatrix<$t> {
            fn eigensystem(mut self) -> NalgebraLapackResult<(DVector<Complex<$t>>, DMatrix<Complex<$t>>)> {
                let jobvl = b'N';
                let jobvr = b'V';

                if self.ncols() != self.nrows() {
                    return Err(NalgebraLapackError { desc: "argument to eigen must be square.".to_owned() } );
                }
                let n = self.ncols();

                let lda = n as i32;
                let ldvl = 1 as i32;
                let ldvr = n;

                let mut wr: DVector<$t> = DVector::from_element( n, 0.0);
                let mut wi: DVector<$t> = DVector::from_element( n, 0.0);

                let mut vl: DVector<$t> = DVector::from_element( (ldvl*ldvl) as usize, 0.0);
                let mut vr: DVector<$t> = DVector::from_element( n*n, 0.0);

                let mut work = vec![0.0];
                let mut lwork = -1 as i32;
                let mut info = 0;

                $lapack_func(jobvl, jobvr, n as i32, self.as_mut_vector(), lda, wr.as_mut(),
                    wi.as_mut(), vl.as_mut(), ldvl, vr.as_mut(), ldvr as i32,
                    &mut work, lwork, &mut info);

                if info < 0 {
                  return Err(NalgebraLapackError { desc: "illegal argument in eigensystem.".to_owned() } );
                }

                lwork = work[0] as i32;
                let mut work = vec![0.0; lwork as usize];

                $lapack_func(jobvl, jobvr, n as i32, self.as_mut_vector(), lda, wr.as_mut(),
                    wi.as_mut(), vl.as_mut(), ldvl, vr.as_mut(), ldvr as i32,
                    &mut work, lwork, &mut info);

                if info < 0 {
                    return Err(NalgebraLapackError { desc: format!(
                                  "illegal argument {} to eigensystem()",
                                  -info
                                  ) } );
                }
                if info > 0 {
                    // TODO: should figure out how to return the correct eigenvalues.
                    return Err(NalgebraLapackError { desc: format!(
                                  "The QR algorithm failed to compute all the eigenvalues. {} were computed.",
                                  -info
                                  ) } );
                }

                let x: Vec<Complex<$t>> = wr.iter().zip(wi.iter()).map( |(r,i)| {Complex{re:*r,im:*i}} ).collect();
                let eigen_values = DVector{at:x};
                let mut result: Vec<Complex<$t>> = Vec::with_capacity(n*n);
                for i in 0..n {
                    let mut j = 0;
                    while j < n {
                        if eigen_values[j].im == 0.0 {
                            result.push( Complex{ re: vr[i+j*ldvr], im: 0.0 });
                            j += 1;
                        } else {
                            result.push( Complex{ re: vr[i+j*ldvr], im: vr[i+(j+1)*ldvr] });
                            result.push( Complex{ re: vr[i+j*ldvr], im: -vr[i+(j+1)*ldvr] });
                            j += 2;
                        }
                    }
                }
                let right_eigen_vectors = DMatrix::from_row_vector(n,n,&result);
                Ok((eigen_values,right_eigen_vectors))
            }
        }
    );
);

macro_rules! eigensystem_complex_impl(
    ($t: ty, $lapack_func: path) => (
        impl Eigensystem<$t> for DMatrix<Complex<$t>> {
            fn eigensystem(mut self) -> NalgebraLapackResult<(DVector<Complex<$t>>, DMatrix<Complex<$t>>)> {
                let jobvl = b'N';
                let jobvr = b'V';

                if self.ncols() != self.nrows() {
                    return Err(NalgebraLapackError { desc: "argument to eigen must be square.".to_owned() } );
                }
                let n = self.ncols();

                let lda = n as i32;
                let ldvl = 1 as i32;
                let ldvr = n as i32;

                let mut w: DVector<Complex<$t>> = DVector::from_element( n, Complex{re:0.0, im:0.0});

                let mut vl: DVector<Complex<$t>> = DVector::from_element( (ldvl*ldvl) as usize, Complex{re:0.0, im:0.0});
                let mut vr: DVector<Complex<$t>> = DVector::from_element( n*n, Complex{re:0.0, im:0.0});

                let mut work = vec![Complex{re:0.0, im:0.0}];
                let mut lwork = -1 as i32;
                let mut rwork: Vec<$t> = vec![0.0; (2*n) as usize];

                let mut info = 0;

                $lapack_func(jobvl, jobvr, n as i32, self.as_mut_vector(), lda, w.as_mut(),
                    vl.as_mut(), ldvl, vr.as_mut(), ldvr,
                    & mut work, lwork, & mut rwork, &mut info);

                if info < 0 {
                  return Err(NalgebraLapackError { desc: "illegal argument in eigensystem.".to_owned() } );
                }

                lwork = work[0].re as i32;
                let mut work = vec![Complex{re:0.0, im:0.0}; lwork as usize];

                $lapack_func(jobvl, jobvr, n as i32, self.as_mut_vector(), lda, w.as_mut(),
                    vl.as_mut(), ldvl, vr.as_mut(), ldvr,
                    & mut work, lwork, & mut rwork, &mut info);

                if info < 0 {
                    return Err(NalgebraLapackError { desc: format!(
                                  "illegal argument {} to eigensystem()",
                                  -info
                                  ) } );
                }
                if info > 0 {
                    // TODO: should figure out how to return the correct eigenvalues.
                    return Err(NalgebraLapackError { desc: format!(
                                  "The QR algorithm failed to compute all the eigenvalues. {} were computed.",
                                  -info
                                  ) } );
                }

                let eigen_values = w;
                let right_eigen_vectors = DMatrix::from_row_vector(n,n,&vr.at);
                Ok((eigen_values,right_eigen_vectors))
            }
        }
    );
);

macro_rules! svd_impl(
    ($t: ty, $lapack_func: path) => (
        impl SVD<$t,$t> for DMatrix<$t> {
            fn svd(mut self) -> NalgebraLapackResult<(DMatrix<$t>, DVector<$t>, DMatrix<$t>)> {
                let m = self.nrows();
                let n = self.ncols();

                let jobu = b'A';
                let jobvt = b'A';

                let lda = m as i32;
                let min_mn = if m <= n { m } else {n};
                let mut s: DVector<$t> = DVector::from_element( min_mn, 0.0);
                let ldu = m;
                let mut u: DMatrix<$t> = DMatrix::new_zeros(ldu, m);
                let ldvt = n;
                let mut vt: DMatrix<$t> = DMatrix::new_zeros(ldvt, n);
                let mut work: Vec<$t> = vec![0.0];
                let mut lwork = -1 as i32;
                let mut info = 0;

                $lapack_func(jobu, jobvt, m as i32, n as i32, self.as_mut_vector(), lda, &mut s.as_mut(),
                               u.as_mut_vector(), ldu as i32, vt.as_mut_vector(),
                               ldvt as i32, &mut work, lwork, &mut info);
                if info < 0 {
                  return Err(NalgebraLapackError { desc: "illegal argument to svd.".to_owned() } );
                }

                lwork = work[0] as i32;
                work = vec![0.0; lwork as usize];

                $lapack_func(jobu, jobvt, m as i32, n as i32, self.as_mut_vector(), lda, &mut s.as_mut(),
                               u.as_mut_vector(), ldu as i32, vt.as_mut_vector(), ldvt as i32, &mut work,
                               lwork, &mut info);

                if info < 0 {
                    return Err(NalgebraLapackError { desc: format!(
                                  "illegal argument {} in svd",
                                  -info
                                  ) } );
                }
                if info > 0 {
                    return Err(NalgebraLapackError { desc: format!(
                                  "{} superdiagonals did not converge.",
                                  info
                                  ) } );
                }

                Ok((u, s, vt))
            }
        }
    );
);

macro_rules! svd_complex_impl(
    ($t: ty, $lapack_func: path) => (
        impl SVD<$t,Complex<$t>> for DMatrix<Complex<$t>> {
            fn svd(mut self) -> NalgebraLapackResult<(DMatrix<Complex<$t>>, DVector<$t>, DMatrix<Complex<$t>>)> {
                let m = self.nrows();
                let n = self.ncols();

                let jobu = b'A';
                let jobvt = b'A';

                let lda = m as i32;
                let min_mn = if m <= n { m } else {n};
                let mut s: DVector<$t> = DVector::from_element( min_mn, 0.0);
                let ldu = m;
                let mut u: DMatrix<Complex<$t>> = DMatrix::new_zeros(ldu, m);
                let ldvt = n;
                let mut vt: DMatrix<Complex<$t>> = DMatrix::new_zeros(ldvt, n);
                let mut work: Vec<Complex<$t>> = vec![Complex{re:0.0, im:0.0}];
                let mut lwork = -1 as i32;
                let mut rwork: Vec<$t> = vec![0.0; (5*min_mn as usize)];
                let mut info = 0;

                $lapack_func(jobu, jobvt, m as i32, n as i32, self.as_mut_vector(), lda, &mut s.as_mut(),
                             u.as_mut_vector(), ldu as i32, vt.as_mut_vector(), ldvt as i32, &mut work,
                             lwork, &mut rwork, &mut info);

                if info < 0 {
                  return Err(NalgebraLapackError { desc: "illegal argument to svd.".to_owned() } );
                }

                lwork = work[0].re as i32;
                let mut work: Vec<Complex<$t>> = vec![Complex{re:0.0, im:0.0}; lwork as usize];

                $lapack_func(jobu, jobvt, m as i32, n as i32, self.as_mut_vector(), lda, &mut s.as_mut(),
                             u.as_mut_vector(), ldu as i32, vt.as_mut_vector(), ldvt as i32, &mut work,
                             lwork, &mut rwork, &mut info);

                if info < 0 {
                    return Err(NalgebraLapackError { desc: format!(
                                  "illegal argument {} in svd",
                                  -info
                                  ) } );
                }
                if info > 0 {
                    return Err(NalgebraLapackError { desc: format!(
                                  "{} superdiagonals did not converge.",
                                  info
                                  ) } );
                }

                Ok((u, s, vt))
            }
        }
    );
);

macro_rules! solve_impl(
    ($t: ty, $lapack_func: path) => (
        impl Solve<$t> for DMatrix<$t> {
            fn solve(self, mut b: DMatrix<$t>) -> NalgebraLapackResult<DMatrix<$t>> {
                let mut a = self;
                let n = a.nrows();
                let m = a.ncols();
                if m != n {
                    return Err(NalgebraLapackError { desc: format!(
                                  "Square matrix required."
                                  ) } );
                }

                let nrhs = n as i32;
                let lda = n as i32;
                let ldb = n as i32;

                let mut ipiv: DVector<i32> = DVector::from_element(n, 0);
                let mut info = 0;

                $lapack_func(n as i32, nrhs, a.as_mut_vector(), lda, ipiv.as_mut(), b.as_mut_vector(), ldb, &mut info);

                if info < 0 {
                    return Err(NalgebraLapackError { desc: "illegal argument to solve.".to_owned() } );
                }
                if info > 0 {
                    return Err(NalgebraLapackError { desc: "cannot solve singular matrix.".to_owned() } );
                }
                Ok(b)

            }
        }
    );
);

macro_rules! cholesky_impl(
    ($t: ty, $lapack_func: path) => (
        impl Cholesky<$t> for DMatrix<$t> {
            fn cholesky(self) -> NalgebraLapackResult<DMatrix<$t>> {
                let uplo = b'L';
                let mut a = self;
                let n = a.nrows() as i32;
                let lda = n;
                let mut info = 0;

                $lapack_func(uplo, n, a.as_mut_vector(), lda, &mut info);

                if info < 0 {
                    return Err(NalgebraLapackError { desc: "illegal argument to cholesky.".to_owned() } );
                }
                if info > 0 {
                    return Err(NalgebraLapackError {
                        desc: "factorization could not be completed (matrix not positive definite?).".to_owned()
                    });
                }

                // zero the upper-triangular part
                for i in 0..a.nrows() {
                    for j in 0..a.ncols() {
                        if j>i {
                            a[(i,j)] = Zero::zero();
                        }
                    }
                }

                Ok(a)
            }
        }
    );
);

use lapack::fortran as interface;

eigensystem_impl!(f32, interface::sgeev);
eigensystem_impl!(f64, interface::dgeev);
eigensystem_complex_impl!(f32, interface::cgeev);
eigensystem_complex_impl!(f64, interface::zgeev);

svd_impl!(f32, interface::sgesvd);
svd_impl!(f64, interface::dgesvd);
svd_complex_impl!(f32, interface::cgesvd);
svd_complex_impl!(f64, interface::zgesvd);

solve_impl!(f32, interface::sgesv);
solve_impl!(f64, interface::dgesv);
solve_impl!(Complex<f32>, interface::cgesv);
solve_impl!(Complex<f64>, interface::zgesv);

impl Inverse<f32> for DMatrix<f32> {}
impl Inverse<f64> for DMatrix<f64> {}
impl Inverse<Complex<f32>> for DMatrix<Complex<f32>> {}
impl Inverse<Complex<f64>> for DMatrix<Complex<f64>> {}

cholesky_impl!(f32, interface::spotrf);
cholesky_impl!(f64, interface::dpotrf);
cholesky_impl!(Complex<f32>, interface::cpotrf);
cholesky_impl!(Complex<f64>, interface::zpotrf);
