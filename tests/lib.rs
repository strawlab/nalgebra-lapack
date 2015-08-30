extern crate nalgebra_lapack;
extern crate nalgebra as na;
extern crate num;

use nalgebra_lapack::{HasSVD, HasEigensystem};

use na::{DMat,DVec,Norm,ColSlice,Iterable};
use num::complex::Complex;

#[test]
fn test_svd_wikipedia() {

  // Example from https://en.wikipedia.org/wiki/Singular_value_decomposition#Example

  let m = DMat::from_row_vec(4,5,&[
      1.0,   0.0,  0.0,   0.0,  2.0,
      0.0,   0.0,  3.0,   0.0,  0.0,
      0.0,   0.0,  0.0,   0.0,  0.0,
      0.0,   4.0,  0.0,   0.0,  0.0]);

  let (u,s,vt) = m.svd().unwrap();

  let mut expected_u: DMat<f64> = DMat::new_zeros( 4, 4);
  expected_u[(0,2)] =  1.0;
  expected_u[(1,1)] =  1.0;
  expected_u[(2,3)] = -1.0;
  expected_u[(3,0)] =  1.0;

  assert_eq!(u.nrows(),  expected_u.nrows());
  assert_eq!(u.ncols(),  expected_u.ncols());
  assert_eq!(u.as_vec(), expected_u.as_vec());

  let mut expected_s: DVec<f64> = DVec::from_elem( 4, 0.0);
  expected_s[0] = 4.0;
  expected_s[1] = 3.0;
  expected_s[2] = 5.0_f64.sqrt();

  assert_eq!(s.len(),  expected_s.len());
  assert_eq!(s.as_slice(), expected_s.as_slice());

  let mut expected_vt: DMat<f64> = DMat::new_zeros( 5, 5);
  expected_vt[(0,1)] =   1.0;
  expected_vt[(1,2)] =   1.0;
  expected_vt[(1,2)] =   1.0;
  expected_vt[(2,0)] =   0.2_f64.sqrt();
  expected_vt[(2,4)] =   0.8_f64.sqrt();
  expected_vt[(3,3)] =   1.0;
  expected_vt[(4,0)] = -(0.8_f64.sqrt());
  expected_vt[(4,4)] =   0.2_f64.sqrt();

  assert_eq!(vt.nrows(),  expected_vt.nrows());
  assert_eq!(vt.ncols(),  expected_vt.ncols());
  assert!(na::approx_eq(&vt, &expected_vt));
}

#[test]
fn test_svd_recomposition() {
  // The actual matrix contents and size should not matter.
  let m = DMat::from_row_vec(3,5,&[
      -1.01,   0.86,  -4.60,   3.31,  -4.81,
       3.98,   0.53,  -7.04,   5.29,   3.55,
       3.30,   8.26,  -3.89,   8.20,  -1.51]);
  let expected_m = m.clone(); // copy since original is moved into svd()
  let (u,s,vt) = m.svd().unwrap();

  let mut full_s = DMat::new_zeros( u.nrows(), vt.nrows() );
  for i in 0..s.len() {
    full_s[(i,i)] = s[i];
  }
  let actual_m = u*(full_s*vt);

  assert!(na::approx_eq(&actual_m, &expected_m));
}

#[test]
fn test_svd_recomposition_complex() {

  fn real_only_mat(u: DMat<Complex<f64>>) -> DMat<f64> {
      DMat::from_col_vec(u.nrows(),u.ncols(),&u.as_vec().iter().map( |c| {assert!(c.im==0.0); c.re}).collect::<Vec<_>>())
  }

  // The actual matrix contents and size should not matter.
  let mr = DMat::from_row_vec(3,5,&[
          -1.01,   0.86,  -4.60,   3.31,  -4.81,
           3.98,   0.53,  -7.04,   5.29,   3.55,
           3.30,   8.26,  -3.89,   8.20,  -1.51]);
  let expected_m = mr.clone(); // copy since original is moved into svd()
  let m: DMat<Complex<f64>> = DMat::from_col_vec(mr.nrows(),mr.ncols(),
    &mr.as_vec().iter().map( |re| {Complex{ re: *re, im: 0.0}}).collect::<Vec<_>>());

  let (uc,s,vtc) = m.svd().unwrap();
  let u = real_only_mat(uc);
  let vt = real_only_mat(vtc);

  let mut full_s = DMat::new_zeros( u.nrows(), vt.nrows() );
  for i in 0..s.len() {
    full_s[(i,i)] = s[i];
  }
  let actual_m = u*(full_s*vt);

  assert!(na::approx_eq(&actual_m, &expected_m));
}

fn real_only(a: &DVec<Complex<f64>> ) -> DVec<f64> {
    let v: Vec<_> = a.iter().map(|cmplx| {assert!(cmplx.im==0.0); cmplx.re}).collect();
    DVec{at:v}
}

fn na_position(a: &DVec<f64>, val: f64, eps: f64) -> Option<usize> {
    a.at.iter().position(|x| (*x-val).abs() < eps )
}

#[test]
fn test_eigenvalues_wikipedia_triangular() {

    // Based on example at
    // https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Triangular_matrices

    let n = 3;
    let mat = DMat::from_row_vec(n,n,&[
           1.0, 0.0, 0.0,
           1.0, 2.0, 0.0,
           2.0, 3.0, 3.0]);
    let (eigen_values, eigen_vectors) = mat.eigensystem().unwrap();

    assert!( eigen_values.len() == n);
    let ev = real_only(&eigen_values);
    let eps = 1e-16;

    // Check for eigvenvalue of 1.0 and corresponding eigenvector.
    if let Some(i) = na_position(&ev,1.0,eps) {
        let expected = DVec{at:vec![1.0, -1.0, 0.5]}.normalize();
        let actual = real_only(&eigen_vectors.col_slice(i,0,n));
        assert!(na::approx_eq( &expected, &actual ));
    } else {
        panic!("expected value not found");
    }

    // Check for eigvenvalue of 2.0 and corresponding eigenvector.
    if let Some(i) = na_position(&ev,2.0,eps) {
        let expected = DVec{at:vec![0.0, 1.0, -3.0]}.normalize();
        let actual = real_only(&eigen_vectors.col_slice(i,0,n)).normalize();
        assert!(na::approx_eq( &expected, &actual ));
    } else {
        panic!("expected value not found");
    }

    // Check for eigvenvalue of 3.0 and corresponding eigenvector.
    if let Some(i) = na_position(&ev,3.0,eps) {
        let expected = DVec{at:vec![0.0, 0.0, 1.0]}.normalize();
        let actual = real_only(&eigen_vectors.col_slice(i,0,n)).normalize();
        assert!(na::approx_eq( &expected, &actual ));
    } else {
        panic!("expected value not found");
    }

}
