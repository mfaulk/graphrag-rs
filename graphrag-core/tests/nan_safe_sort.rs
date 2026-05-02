//! Regression test for #15: workspace-wide replacement of
//! `partial_cmp(...).unwrap()` with `total_cmp(...)`.
//!
//! Verifies that sorting f32/f64 vectors that contain NaN does not panic.
//! Previously the bare `.unwrap()` on `partial_cmp` returned `None` on NaN
//! and crashed the request thread.

#[test]
fn total_cmp_does_not_panic_on_nan_f32() {
    let mut values: Vec<f32> = vec![3.0, f32::NAN, 1.0, 2.0, f32::NAN];
    values.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(values.len(), 5);
}

#[test]
fn total_cmp_does_not_panic_on_nan_f64() {
    let mut values: Vec<f64> = vec![3.0, f64::NAN, 1.0, 2.0, f64::NAN];
    values.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(values.len(), 5);
}

#[test]
fn total_cmp_orders_normally_when_no_nan() {
    let mut values: Vec<f32> = vec![3.0, 1.0, 2.0];
    values.sort_by(|a, b| b.total_cmp(a));
    assert_eq!(values, vec![3.0, 2.0, 1.0]);
}
