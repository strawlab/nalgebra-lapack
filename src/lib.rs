/*!
linear algebra operations for [nalgebra][1] matrices using [LAPACK][2].

Functions to compute the singular value decomposition (SVD) and eigensystem are
implemented.

# Examples

```rust
extern crate nalgebra_lapack;
extern crate nalgebra as na;

use nalgebra_lapack::{HasSVD, HasEigensystem};

fn main() {
    // Create an input matrix
    let m = na::DMat::from_row_vec(3,5,&[
        -1.01,   0.86,  -4.60,   3.31,  -4.81,
         3.98,   0.53,  -7.04,   5.29,   3.55,
         3.30,   8.26,  -3.89,   8.20,  -1.51]);

    // Now perform SVD
    let (u,s,vt) = m.svd().unwrap();
    println!("u {:?}",u);
    println!("s {:?}",s);
    println!("vt {:?}",vt);

    // Create an input matrix
    let m = na::DMat::from_row_vec(2,2,&[
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

extern crate nalgebra as na;
extern crate lapack;
extern crate num;

use std::error::Error;
use std::fmt::{self, Display};
use num::complex::Complex;

use na::{DMat, DVec, Iterable};

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
        impl HasEigensystem<Complex<$t>> for DMat<$t> {
            fn eigensystem(mut self) -> NalgebraLapackResult<(DVec<Complex<$t>>, DMat<Complex<$t>>)> {
                let jobvl = b'N';
                let jobvr = b'V';

                if self.ncols() != self.nrows() {
                    return Err(NalgebraLapackError { desc: "argument to eigen must be square.".to_owned() } );
                }
                let n = self.ncols();

                let lda = n;
                let ldvl = 1;
                let ldvr = n;

                let mut wr: DVec<$t> = DVec::from_elem( n, 0.0);
                let mut wi: DVec<$t> = DVec::from_elem( n, 0.0);

                let mut vl: DVec<$t> = DVec::from_elem( ldvl*ldvl, 0.0);
                let mut vr: DVec<$t> = DVec::from_elem( n*n, 0.0);

                let mut work = vec![0.0];
                let mut lwork = -1;
                let mut info = 0;

                $lapack_func(jobvl, jobvr, n, self.as_mut_vec(), lda, wr.as_mut_slice(),
                    wi.as_mut_slice(), vl.as_mut_slice(), ldvl, vr.as_mut_slice(), ldvr,
                    &mut work, lwork, &mut info);

                if info < 0 {
                  return Err(NalgebraLapackError { desc: "illegal argument in eigensystem.".to_owned() } );
                }

                lwork = work[0] as isize;
                let mut work = vec![0.0; lwork as usize];

                $lapack_func(jobvl, jobvr, n, self.as_mut_vec(), lda, wr.as_mut_slice(),
                    wi.as_mut_slice(), vl.as_mut_slice(), ldvl, vr.as_mut_slice(), ldvr,
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
                let eigen_values = DVec{at:x};
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
                let right_eigen_vectors = DMat::from_row_vec(n,n,&result);
                Ok((eigen_values,right_eigen_vectors))
            }
        }
    );
);

pub trait HasEigensystem<T> {
    /// `eigensystem` computes eigenvalues and right eigenvectors
    ///
    /// Because the input matrix may be overwritten or destroyed, it is consumed.
    ///
    /// # Returns
    ///
    /// * `eigen_values` - The eigenvalues, normalized to have Euclidean norm equal to 1 and largest component real.
    /// * `right_eigen_vectors` - The right eigenvectors. They are contained as columns of this matrix.
    fn eigensystem(mut self) -> NalgebraLapackResult<(DVec<T>, DMat<T>)>;
}

macro_rules! eigensystem_complex_impl(
    ($t: ty, $lapack_func: path) => (
        impl HasEigensystem<Complex<$t>> for DMat<Complex<$t>> {
            fn eigensystem(mut self) -> NalgebraLapackResult<(DVec<Complex<$t>>, DMat<Complex<$t>>)> {
                let jobvl = b'N';
                let jobvr = b'V';

                if self.ncols() != self.nrows() {
                    return Err(NalgebraLapackError { desc: "argument to eigen must be square.".to_owned() } );
                }
                let n = self.ncols();

                let lda = n;
                let ldvl = 1;
                let ldvr = n;

                let mut w: DVec<Complex<$t>> = DVec::from_elem( n, Complex{re:0.0, im:0.0});

                let mut vl: DVec<Complex<$t>> = DVec::from_elem( ldvl*ldvl, Complex{re:0.0, im:0.0});
                let mut vr: DVec<Complex<$t>> = DVec::from_elem( n*n, Complex{re:0.0, im:0.0});

                let mut work = vec![Complex{re:0.0, im:0.0}];
                let mut lwork = -1;
                let mut rwork: Vec<$t> = vec![0.0; (2*n) as usize];

                let mut info = 0;

                $lapack_func(jobvl, jobvr, n, self.as_mut_vec(), lda, w.as_mut_slice(),
                    vl.as_mut_slice(), ldvl, vr.as_mut_slice(), ldvr,
                    & mut work, lwork, & mut rwork, &mut info);

                if info < 0 {
                  return Err(NalgebraLapackError { desc: "illegal argument in eigensystem.".to_owned() } );
                }

                lwork = work[0].re as isize;
                let mut work = vec![Complex{re:0.0, im:0.0}; lwork as usize];

                $lapack_func(jobvl, jobvr, n, self.as_mut_vec(), lda, w.as_mut_slice(),
                    vl.as_mut_slice(), ldvl, vr.as_mut_slice(), ldvr,
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
                let right_eigen_vectors = DMat::from_row_vec(n,n,&vr.at);
                Ok((eigen_values,right_eigen_vectors))
            }
        }
    );
);

pub trait HasSVD<T,U> {
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
    fn svd(mut self) -> NalgebraLapackResult<(DMat<T>, DVec<U>, DMat<T>)>;
}

macro_rules! svd_impl(
    ($t: ty, $lapack_func: path) => (
        impl HasSVD<$t,$t> for DMat<$t> {
            fn svd(mut self) -> NalgebraLapackResult<(DMat<$t>, DVec<$t>, DMat<$t>)> {
                let m = self.nrows();
                let n = self.ncols();

                let jobu = b'A';
                let jobvt = b'A';

                let lda = m;
                let min_mn = if m <= n { m } else {n};
                let mut s: DVec<$t> = DVec::from_elem( min_mn, 0.0);
                let ldu = m;
                let mut u: DMat<$t> = DMat::new_zeros(ldu, m);
                let ldvt = n;
                let mut vt: DMat<$t> = DMat::new_zeros(ldvt, n);
                let mut work = vec![0.0];
                let mut lwork = -1;
                let mut info = 0;

                $lapack_func(jobu, jobvt, m, n, self.as_mut_vec(), lda, &mut s.as_mut_slice(),
                               u.as_mut_vec(), ldu, vt.as_mut_vec(),
                               ldvt, &mut work, lwork, &mut info);
                if info < 0 {
                  return Err(NalgebraLapackError { desc: "illegal argument to svd.".to_owned() } );
                }

                lwork = work[0] as isize;
                work = vec![0.0; lwork as usize];

                $lapack_func(jobu, jobvt, m, n, self.as_mut_vec(), lda, &mut s.as_mut_slice(),
                               u.as_mut_vec(), ldu, vt.as_mut_vec(), ldvt, &mut work,
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
        impl HasSVD<Complex<$t>,$t> for DMat<Complex<$t>> {
            fn svd(mut self) -> NalgebraLapackResult<(DMat<Complex<$t>>, DVec<$t>, DMat<Complex<$t>>)> {
                let m = self.nrows();
                let n = self.ncols();

                let jobu = b'A';
                let jobvt = b'A';

                let lda = m;
                let min_mn = if m <= n { m } else {n};
                let mut s: DVec<$t> = DVec::from_elem( min_mn, 0.0);
                let ldu = m;
                let mut u: DMat<Complex<$t>> = DMat::new_zeros(ldu, m);
                let ldvt = n;
                let mut vt: DMat<Complex<$t>> = DMat::new_zeros(ldvt, n);
                let mut work: Vec<Complex<$t>> = vec![Complex{re:0.0, im:0.0}];
                let mut lwork = -1;
                let mut rwork: Vec<$t> = vec![0.0; (5*min_mn as usize)];
                let mut info = 0;

                $lapack_func(jobu, jobvt, m, n, self.as_mut_vec(), lda, &mut s.as_mut_slice(),
                             u.as_mut_vec(), ldu, vt.as_mut_vec(), ldvt, &mut work,
                             lwork, &mut rwork, &mut info);

                if info < 0 {
                  return Err(NalgebraLapackError { desc: "illegal argument to svd.".to_owned() } );
                }

                lwork = work[0].re as isize;
                let mut work: Vec<Complex<$t>> = vec![Complex{re:0.0, im:0.0}; lwork as usize];

                $lapack_func(jobu, jobvt, m, n, self.as_mut_vec(), lda, &mut s.as_mut_slice(),
                             u.as_mut_vec(), ldu, vt.as_mut_vec(), ldvt, &mut work,
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

eigensystem_impl!(f32, lapack::sgeev);
eigensystem_impl!(f64, lapack::dgeev);
eigensystem_complex_impl!(f32, lapack::cgeev);
eigensystem_complex_impl!(f64, lapack::zgeev);

svd_impl!(f32, lapack::sgesvd);
svd_impl!(f64, lapack::dgesvd);
svd_complex_impl!(f32, lapack::cgesvd);
svd_complex_impl!(f64, lapack::zgesvd);
