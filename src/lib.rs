/*!
linear algebra operations for [nalgebra][1] matrices using [LAPACK][2].

# Examples

```rust
extern crate nalgebra_lapack;
extern crate nalgebra as na;

use nalgebra_lapack::{SVD, Eigensystem, Inverse, Solve};
use na::Eye;

fn main() {
    // SVD -----------------------------------
    let m = na::DMatrix::from_row_vector(3, 5,
        &[-1.01,   0.86,  -4.60,   3.31,  -4.81,
           3.98,   0.53,  -7.04,   5.29,   3.55,
           3.30,   8.26,  -3.89,   8.20,  -1.51]);
    let (u,s,vt) = m.svd().unwrap();
    println!("u {:?}", u);
    println!("s {:?}", s);
    println!("vt {:?}", vt);

    // Eigensystem ---------------------------
    let m = na::DMatrix::from_row_vector(2, 2,
        &[2.0, 1.0,
          1.0, 2.0]);
    let (vals, vecs) = m.eigensystem().unwrap();
    println!("eigenvalues {:?}", vals);
    println!("eigenvectors {:?}", vecs);

    // Invert matrix -------------------------
    let a = na::DMatrix::from_row_vector(2, 2,
        &[1.0, 2.0,
          3.0, 4.0]);
    let a_inv = a.inv().unwrap();
    println!("a_inv {:?}", a_inv);

    // Solve ---------------------------------
    // invert matrix `a` by solving solution to `ax=1`
    let a = na::DMatrix::from_row_vector(2, 2,
        &[1.0, 2.0,
          3.0, 4.0]);
    let n = a.nrows();
    let b = na::DMatrix::new_identity(n);
    let a_inv = a.solve(b);
    println!("a_inv {:?}", a_inv);
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
use num::Zero;

use nalgebra::{DMatrix, DVector, Iterable, Eye};

/// A type for which eigenvalues and eigenvectors can be computed.
pub trait Eigensystem {
    type N;
    /// compute eigenvalues and right eigenvectors
    ///
    /// Because the input matrix may be overwritten or destroyed, it is consumed.
    ///
    /// # Returns
    ///
    /// * `eigen_values` - The eigenvalues, normalized to have Euclidean norm equal to 1 and
    ///   largest component real.
    /// * `right_eigen_vectors` - The right eigenvectors. They are contained as columns of this
    ///   matrix.
    fn eigensystem(mut self) -> NalgebraLapackResult<(DVector<Complex<Self::N>>, DMatrix<Complex<Self::N>>)>;
}

/// A type for which a singular value decomposition can be computed.
pub trait SVD {
    type V;
    type M;
    /// compute the singular value decomposition (SVD). Returns full matrices.
    ///
    /// Because the input matrix may be overwritten or destroyed, it is consumed.
    ///
    /// # Returns
    ///
    /// * `u` - The left-singular vectors.
    /// * `s` - The singular values.
    /// * `vt` - The right-singular vectors.
    fn svd(mut self) -> NalgebraLapackResult<(DMatrix<Self::M>, DVector<Self::V>, DMatrix<Self::M>)>;
}

/// A type for solving a linear matrix equation.
pub trait Solve
{
    type N;
    /// solve a linear matrix equation.
    ///
    /// Given the equation `ax=b` where `a` and `b` are known, find  matrix `x`.
    ///
    /// Because the input matrix may be overwritten or destroyed, it is consumed.
    ///
    /// # Arguments
    ///
    /// * `b` - The known matrix.
    ///
    /// # Returns
    ///
    /// * `x` - The solution to the linear equation `ax=b`.
    fn solve(self, b: DMatrix<Self::N>) -> NalgebraLapackResult<DMatrix<Self::N>>;
}

/// A type for which the inverse can be computed.
pub trait Inverse : Solve
{
    /// compute the (multiplicative) inverse.
    ///
    /// Because the input matrix may be overwritten or destroyed, it is consumed.
    ///
    /// # Returns
    ///
    /// * `inverse` - The inverted matrix.
    fn inv(self) -> NalgebraLapackResult<DMatrix<<Self as Solve>::N>> where Self: std::marker::Sized;
}

/// A type for which the Cholesky decomposition can be computed.
pub trait Cholesky
{
    type N;

    /// computes the cholesky decomposition of a Hermitian positive-definite matrix.
    ///
    /// Because the input matrix may be overwritten or destroyed, it is consumed.
    ///
    /// # Returns
    ///
    /// * `lower_triangular` - The lower triangular part of the decomposed matrix.
    fn cholesky(self) -> NalgebraLapackResult<DMatrix<Self::N>>;
}

#[derive(Debug)]
pub struct NalgebraLapackError {
    pub desc: String,
}

pub type NalgebraLapackResult<T> = Result<T, NalgebraLapackError>;

impl Error for NalgebraLapackError {
    fn description(&self) -> &str {
        &self.desc
    }
}

impl Display for NalgebraLapackError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.desc, f)
    }
}

impl From<String> for NalgebraLapackError {
    fn from(err: String) -> NalgebraLapackError {
        NalgebraLapackError { desc: format!("String ({})", err) }
    }
}

macro_rules! check_info(
    ($info: expr) => (
        if $info < 0 {
            return Err(NalgebraLapackError { desc: format!(
                          "illegal argument to lapack {}",-$info)});
        } else if $info > 0 {
            return Err(NalgebraLapackError { desc: format!(
                          "lapack failure {}", $info)});
        }
    );
);

macro_rules! eigensystem_impl(
    ($t: ty, $lapack_func: path) => (
        impl Eigensystem for DMatrix<$t> {
            type N = $t;
            fn eigensystem(mut self) ->
            NalgebraLapackResult<(DVector<Complex<$t>>, DMatrix<Complex<$t>>)> {
                let jobvl = b'N';
                let jobvr = b'V';

                if self.ncols() != self.nrows() {
                    return Err(NalgebraLapackError {
                         desc: "argument to eigen must be square.".to_owned()
                     } );
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

                check_info!(info);

                lwork = work[0] as i32;
                let mut work = vec![0.0; lwork as usize];

                $lapack_func(jobvl, jobvr, n as i32, self.as_mut_vector(), lda, wr.as_mut(),
                    wi.as_mut(), vl.as_mut(), ldvl, vr.as_mut(), ldvr as i32,
                    &mut work, lwork, &mut info);
                check_info!(info);

                let x: Vec<Complex<$t>> = wr
                    .iter()
                    .zip(wi.iter())
                    .map( |(r,i)| {Complex{re:*r,im:*i}} )
                    .collect();
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
        impl Eigensystem for DMatrix<Complex<$t>> {
            type N = $t;

            fn eigensystem(mut self)
                -> NalgebraLapackResult<(DVector<Complex<$t>>, DMatrix<Complex<$t>>)> {
                let jobvl = b'N';
                let jobvr = b'V';

                if self.ncols() != self.nrows() {
                    return Err(NalgebraLapackError {
                        desc: "argument to eigen must be square.".to_owned()
                    } );
                }
                let n = self.ncols();

                let lda = n as i32;
                let ldvl = 1 as i32;
                let ldvr = n as i32;

                let mut w: DVector<Complex<$t>> = DVector::from_element(
                    n, Complex{re:0.0, im:0.0});

                let mut vl: DVector<Complex<$t>> = DVector::from_element(
                    (ldvl*ldvl) as usize, Complex{re:0.0, im:0.0});
                let mut vr: DVector<Complex<$t>> = DVector::from_element(
                    n*n, Complex{re:0.0, im:0.0});

                let mut work = vec![Complex{re:0.0, im:0.0}];
                let mut lwork = -1 as i32;
                let mut rwork: Vec<$t> = vec![0.0; (2*n) as usize];

                let mut info = 0;

                $lapack_func(jobvl, jobvr, n as i32, self.as_mut_vector(), lda, w.as_mut(),
                    vl.as_mut(), ldvl, vr.as_mut(), ldvr,
                    & mut work, lwork, & mut rwork, &mut info);
                check_info!(info);

                lwork = work[0].re as i32;
                let mut work = vec![Complex{re:0.0, im:0.0}; lwork as usize];

                $lapack_func(jobvl, jobvr, n as i32, self.as_mut_vector(), lda, w.as_mut(),
                    vl.as_mut(), ldvl, vr.as_mut(), ldvr,
                    & mut work, lwork, & mut rwork, &mut info);
                check_info!(info);

                let eigen_values = w;
                let right_eigen_vectors = DMatrix::from_row_vector(n,n,&vr.at);
                Ok((eigen_values,right_eigen_vectors))
            }
        }
    );
);

macro_rules! svd_impl(
    ($t: ty, $lapack_func: path) => (
        impl SVD for DMatrix<$t> {
            type V = $t;
            type M = $t;
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

                $lapack_func(jobu, jobvt, m as i32, n as i32, self.as_mut_vector(), lda,
                    &mut s.as_mut(), u.as_mut_vector(), ldu as i32, vt.as_mut_vector(),
                    ldvt as i32, &mut work, lwork, &mut info);
                check_info!(info);

                lwork = work[0] as i32;
                work = vec![0.0; lwork as usize];

                $lapack_func(jobu, jobvt, m as i32, n as i32, self.as_mut_vector(), lda,
                    &mut s.as_mut(), u.as_mut_vector(), ldu as i32, vt.as_mut_vector(),
                    ldvt as i32, &mut work, lwork, &mut info);
                check_info!(info);

                Ok((u, s, vt))
            }
        }
    );
);

macro_rules! svd_complex_impl(
    ($t: ty, $lapack_func: path) => (
        impl SVD for DMatrix<Complex<$t>> {
            type V=$t;
            type M=Complex<$t>;
            fn svd(mut self)
                -> NalgebraLapackResult<(DMatrix<Complex<$t>>, DVector<$t>, DMatrix<Complex<$t>>)> {
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

                $lapack_func(jobu, jobvt, m as i32, n as i32, self.as_mut_vector(), lda,
                             &mut s.as_mut(),
                             u.as_mut_vector(), ldu as i32, vt.as_mut_vector(), ldvt as i32,
                             &mut work,
                             lwork, &mut rwork, &mut info);
                check_info!(info);

                lwork = work[0].re as i32;
                let mut work: Vec<Complex<$t>> = vec![Complex{re:0.0, im:0.0}; lwork as usize];

                $lapack_func(jobu, jobvt, m as i32, n as i32, self.as_mut_vector(), lda,
                             &mut s.as_mut(),
                             u.as_mut_vector(), ldu as i32, vt.as_mut_vector(), ldvt as i32,
                             &mut work,
                             lwork, &mut rwork, &mut info);
                check_info!(info);

                Ok((u, s, vt))
            }
        }
    );
);

macro_rules! solve_impl(
    ($t: ty, $lapack_func: path) => (
        impl Solve for DMatrix<$t> {
            type N=$t;
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

                $lapack_func(n as i32, nrhs, a.as_mut_vector(), lda, ipiv.as_mut(),
                    b.as_mut_vector(), ldb, &mut info);
                check_info!(info);
                Ok(b)

            }
        }
    );
);

macro_rules! inverse_impl(
    ($t: ty) => (
        impl Inverse for DMatrix<$t> {
            fn inv(self) -> NalgebraLapackResult<DMatrix<<Self as Solve>::N>>
                where Self: std::marker::Sized
            {
                let n = self.nrows();
                let b = DMatrix::new_identity(n);
                self.solve(b)
            }
        }
    );
);

macro_rules! cholesky_impl(
    ($t: ty, $lapack_func: path) => (
        impl Cholesky for DMatrix<$t> {
            type N = $t;
            fn cholesky(self) -> NalgebraLapackResult<DMatrix<$t>> {
                let uplo = b'L';
                let mut a = self;
                let n = a.nrows() as i32;
                let lda = n;
                let mut info = 0;

                $lapack_func(uplo, n, a.as_mut_vector(), lda, &mut info);
                check_info!(info);

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

inverse_impl!(f32);
inverse_impl!(f64);
inverse_impl!(Complex<f32>);
inverse_impl!(Complex<f64>);

cholesky_impl!(f32, interface::spotrf);
cholesky_impl!(f64, interface::dpotrf);
cholesky_impl!(Complex<f32>, interface::cpotrf);
cholesky_impl!(Complex<f64>, interface::zpotrf);
