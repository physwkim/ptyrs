//! The M6 initial-value measure (DESIGN, M6): iterations a reconstruction takes to bring its
//! LLK down to a target, from the predicted
//! ([`Predictor::initial_object`](crate::predictor::Predictor::initial_object)) and from the
//! flat ([`flat_object`](crate::predictor::flat_object)) initial object. The engine is reached
//! only through the LLK values it yields, so the measure runs on any reconstruction loop.

/// Iterations until the LLK is at most `target`, from a start whose LLK is `start`: 0 when the
/// start already is, `None` when `llk` ends first. `llk` yields the LLK after each iteration in
/// order, a recorded history or a lazy loop; it is not advanced past the iteration that meets
/// the target.
pub fn iterations_to_llk(
    start: f64,
    target: f64,
    llk: impl IntoIterator<Item = f64>,
) -> Option<usize> {
    if start <= target {
        return Some(0);
    }
    llk.into_iter().position(|l| l <= target).map(|k| k + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count, and how many values of `llk` it consumed.
    fn walk(llk: &[f64], start: f64, target: f64) -> (Option<usize>, usize) {
        let mut taken = 0;
        let lazy = std::iter::from_fn(|| {
            taken += 1;
            llk.get(taken - 1).copied()
        });
        let n = iterations_to_llk(start, target, lazy);
        (n, taken)
    }

    #[test]
    fn a_start_at_the_target_takes_no_iteration() {
        assert_eq!(walk(&[2.0], 3.0, 3.0), (Some(0), 0));
    }

    #[test]
    fn counts_iterations_up_to_the_first_at_the_target() {
        assert_eq!(walk(&[5.0, 3.0, 2.0], 6.0, 3.0), (Some(2), 2));
    }

    #[test]
    fn a_history_that_ends_short_of_the_target_gives_none() {
        assert_eq!(walk(&[5.0, 4.0], 6.0, 3.0), (None, 3));
    }
}
