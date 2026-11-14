//! The maximum every test comparison takes its worst difference with.

/// The largest of `values`, NaN when any of them is, and 0 of none. `fold(0.0, f64::max)`
/// gives the largest of the others instead, which a tolerance then passes.
pub fn worst_of(values: impl IntoIterator<Item = f64>) -> f64 {
    let worse = |w: f64, v: f64| if w >= v || w.is_nan() { w } else { v };
    values.into_iter().fold(0.0, worse)
}
