//! The M6 initial-value measure (DESIGN, M6): iterations a reconstruction takes to bring its
//! LLK down to a target, from the predicted
//! ([`Predictor::initial_object`](crate::predictor::Predictor::initial_object)) and from the
//! flat ([`flat_object`](crate::predictor::flat_object)) initial object. The engine is reached
//! only through `iterate`, so the measure runs on any reconstruction loop.

/// Iterations until the LLK is at most `target`, from a start whose LLK is `start`: 0 when the
/// start already is, `None` when `max` iterations do not get there. `iterate` runs one
/// iteration and returns the LLK after it; it is not called once the target is met.
pub fn iterations_to_llk<E>(
    start: f64,
    target: f64,
    max: usize,
    mut iterate: impl FnMut() -> Result<f64, E>,
) -> Result<Option<usize>, E> {
    if start <= target {
        return Ok(Some(0));
    }
    for k in 1..=max {
        if iterate()? <= target {
            return Ok(Some(k));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts the calls of an `iterate` that walks down `llk`.
    fn walk(llk: &[f64], start: f64, target: f64, max: usize) -> (Option<usize>, usize) {
        let mut calls = 0;
        let n = iterations_to_llk(start, target, max, || {
            calls += 1;
            Ok::<_, ()>(llk[calls - 1])
        })
        .unwrap();
        (n, calls)
    }

    #[test]
    fn a_start_at_the_target_takes_no_iteration() {
        assert_eq!(walk(&[], 3.0, 3.0, 10), (Some(0), 0));
    }

    #[test]
    fn counts_iterations_up_to_the_first_at_the_target() {
        assert_eq!(walk(&[5.0, 3.0, 2.0], 6.0, 3.0, 10), (Some(2), 2));
    }

    #[test]
    fn stops_after_max_iterations_short_of_the_target() {
        assert_eq!(walk(&[5.0, 4.0, 3.0], 6.0, 3.0, 2), (None, 2));
    }

    #[test]
    fn passes_on_the_first_iteration_error() {
        let mut calls = 0;
        let r = iterations_to_llk(6.0, 3.0, 10, || {
            calls += 1;
            if calls == 2 { Err("diverged") } else { Ok(5.0) }
        });
        assert_eq!((r, calls), (Err("diverged"), 2));
    }
}
