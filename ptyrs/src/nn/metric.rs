//! The M6 initial-value measure (DESIGN, M6): iterations a reconstruction takes to bring its
//! LLK down to a target for good, from the predicted
//! ([`Predictor::initial_object`](crate::nn::predictor::Predictor::initial_object)) and from the
//! flat ([`flat_object`](crate::nn::predictor::flat_object)) initial object. The engine is reached
//! only through the LLK values it yields, so the measure runs on any reconstruction loop.

/// The measure's margin: the band an LLK has to stay in reaches 1% above the target, the margin
/// of M3's solve.
pub const LLK_MARGIN: f64 = 0.01;

/// The first iteration from which the LLK stays at most `target · (1 + margin)` through the last
/// of `llk`, from a start whose LLK is `start`: 0 when the start and every value are, `None` when
/// the last is above it (or NaN). `llk` holds the LLK after each iteration in order. A curve that
/// dips into the band and climbs back out has not reached the target at the dip: DM with the
/// probe updated does so in its first epochs, and a first crossing would count that dip.
pub fn iterations_to_llk(start: f64, target: f64, margin: f64, llk: &[f64]) -> Option<usize> {
    let bound = target * (1.0 + margin);
    llk.iter()
        .enumerate()
        .fold((start <= bound).then_some(0), |from, (i, &l)| {
            if l <= bound {
                from.or(Some(i + 1))
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_start_at_the_target_that_stays_takes_no_iteration() {
        assert_eq!(iterations_to_llk(3.0, 3.0, 0.0, &[2.0]), Some(0));
        assert_eq!(iterations_to_llk(3.0, 3.0, 0.0, &[]), Some(0));
    }

    #[test]
    fn counts_iterations_up_to_the_first_at_the_target() {
        assert_eq!(iterations_to_llk(6.0, 3.0, 0.0, &[5.0, 3.0, 2.0]), Some(2));
    }

    #[test]
    fn the_margin_admits_values_up_to_that_fraction_above_the_target() {
        assert_eq!(
            iterations_to_llk(6.0, 3.0, 0.01, &[5.0, 3.04, 3.02, 2.0]),
            Some(3)
        );
        assert_eq!(
            iterations_to_llk(6.0, 3.0, 0.0, &[5.0, 3.04, 3.02, 2.0]),
            Some(4)
        );
        assert_eq!(iterations_to_llk(3.02, 3.0, 0.01, &[]), Some(0));
    }

    #[test]
    fn a_dip_into_the_band_that_climbs_back_out_does_not_count() {
        // In at 2, out at 3, back in at 5 for good.
        let llk = [5.0, 1.0, 4.0, 3.5, 2.9, 3.0];
        assert_eq!(iterations_to_llk(6.0, 3.0, 0.0, &llk), Some(5));
        // In at 2 and out to the end: never reached.
        assert_eq!(iterations_to_llk(6.0, 3.0, 0.0, &llk[..4]), None);
        // A start in the band that climbs out counts from its return.
        assert_eq!(iterations_to_llk(2.0, 3.0, 0.0, &[4.0, 3.0]), Some(2));
    }

    #[test]
    fn a_history_that_ends_short_of_the_target_gives_none() {
        assert_eq!(iterations_to_llk(6.0, 3.0, 0.0, &[5.0, 4.0]), None);
        assert_eq!(iterations_to_llk(6.0, 3.0, 0.0, &[2.0, f64::NAN]), None);
    }
}
