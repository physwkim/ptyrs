//! ptyrs host library (DESIGN §4): one module per former crate. `base` is the data model, a name
//! other than `core`, which would shadow the built-in crate.

pub mod base;
pub mod dist;
pub mod engine;
mod fault;
pub mod fft;
pub mod gpu;
pub mod io;
pub mod live;
// burn builds only with the `nn` feature.
#[cfg(feature = "nn")]
pub mod nn;
pub mod refgen;
pub mod tile;

/// The comparisons' maximum, the one `tests/` shares.
#[cfg(test)]
#[path = "../tests/common/worst.rs"]
mod worst;

#[cfg(test)]
mod tests {
    use crate::worst::worst_of;

    #[test]
    fn worst_of_is_nan_of_any_nan() {
        assert_eq!(worst_of([1e-7, 2.0, 3e-7]), 2.0);
        assert_eq!(worst_of([]), 0.0);
        for at in 0..3 {
            let mut v = [1e-7, 2.0, 3e-7];
            v[at] = f64::NAN;
            assert!(worst_of(v).is_nan(), "NaN at {at}");
        }
    }
}
