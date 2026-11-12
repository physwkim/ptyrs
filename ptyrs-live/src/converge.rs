//! The M5 clock (DESIGN.md §7 M5, §0 condition 4): from the scan's end, the
//! arrival of its last frame ([`SourceStats::scan_end`]), to the first live
//! object that matches the offline run of the same algorithm on the same
//! frames in Fourier ring correlation.
//!
//! [`SourceStats::scan_end`]: crate::pipeline::SourceStats::scan_end

use std::time::Duration;

use ptyrs_core::frc::Frc;
use ptyrs_core::{C64, CoreError};

/// Every ring of a converged live object correlates with the offline one at
/// least this much. 0.99 holds the difference under 2% of the offline power
/// in each ring (FRC = 1/√(1 + ε²) for an uncorrelated difference of
/// relative power ε²); two rPIE runs from different starts on the same
/// simulated frames hold their outer rings only at 0.983–0.9994, so 0.999
/// is out of reach.
pub const MATCH: f64 = 0.99;

/// One scan's clock, started at its end.
pub struct Clock {
    offline: Vec<C64>,
    shape: [usize; 2],
    scan_end: i64,
}

impl Clock {
    /// Against `offline`, the offline run's object, `[ny, nx]` row-major.
    /// `earlier`, a snapshot of that run some iterations before, must match
    /// it to [`MATCH`] in every ring: a run still moving is no reference.
    /// `scan_end` is wall ns, as frame arrivals are.
    pub fn new(
        offline: Vec<C64>,
        earlier: &[C64],
        shape: [usize; 2],
        scan_end: i64,
    ) -> Result<Self, CoreError> {
        let f = Frc::compute(&offline, earlier, shape)?;
        if let Some(r) = below(&f) {
            return Err(CoreError::Invalid(format!(
                "the offline run is still moving: ring {r} of {} at FRC {:.4} against its \
                 earlier snapshot, under {MATCH}",
                f.rings.len(),
                f.rings[r].frc
            )));
        }
        Ok(Self {
            offline,
            shape,
            scan_end,
        })
    }

    /// Time from the scan's end to `taken` (wall ns) if the live object
    /// taken then matches the offline one to [`MATCH`] in every ring; `None`
    /// below that or before the end, when frames are still missing. The
    /// first snapshot with a time is the converged one.
    pub fn check(&self, live: &[C64], taken: i64) -> Result<Option<Duration>, CoreError> {
        let Ok(after) = u64::try_from(taken - self.scan_end) else {
            return Ok(None);
        };
        let f = Frc::compute(&self.offline, live, self.shape)?;
        Ok(below(&f).is_none().then(|| Duration::from_nanos(after)))
    }
}

/// The first ring under [`MATCH`].
fn below(f: &Frc) -> Option<usize> {
    f.rings.iter().position(|r| r.frc < MATCH)
}

#[cfg(test)]
mod tests {
    use crate::synth::{hash3, unit};

    use super::*;

    const SHAPE: [usize; 2] = [64, 64];

    /// White complex noise: every ring holds the same power.
    fn noise(seed: u64) -> Vec<C64> {
        (0..(SHAPE[0] * SHAPE[1]) as u64)
            .map(|i| C64::new(unit(hash3(seed, i, 0)), unit(hash3(seed, i, 1))))
            .collect()
    }

    /// `a` plus independent noise `eps` times its amplitude: an FRC near
    /// 1/√(1 + eps²) in every ring.
    fn plus(a: &[C64], eps: f64, seed: u64) -> Vec<C64> {
        a.iter()
            .zip(noise(seed))
            .map(|(a, n)| a + n * eps)
            .collect()
    }

    /// A live object within 0.05 of the offline one (FRC near 0.9988), in
    /// another global phase, converges at the scan's end or after, never
    /// before; one 0.5 away (near 0.89) does not.
    #[test]
    fn a_live_object_converges_once_every_ring_matches_after_the_scan_end() {
        let offline = noise(1);
        let earlier = plus(&offline, 0.02, 2);
        let clock = Clock::new(offline.clone(), &earlier, SHAPE, 1_000).unwrap();
        let rot = C64::from_polar(1.0, 0.7);
        let close: Vec<C64> = plus(&offline, 0.05, 3).iter().map(|z| z * rot).collect();
        let far = plus(&offline, 0.5, 4);
        assert_eq!(clock.check(&close, 999).unwrap(), None);
        assert_eq!(clock.check(&close, 1_000).unwrap(), Some(Duration::ZERO));
        assert_eq!(
            clock.check(&close, 1_250).unwrap(),
            Some(Duration::from_nanos(250))
        );
        assert_eq!(clock.check(&far, 2_000).unwrap(), None);
        let mut bad = close;
        bad[5] = C64::new(f64::NAN, 0.0);
        assert!(clock.check(&bad, 2_000).is_err());
    }

    /// An offline run whose earlier snapshot is 0.5 away is still moving and
    /// starts no clock.
    #[test]
    fn an_offline_run_still_moving_is_no_reference() {
        let offline = noise(1);
        let earlier = plus(&offline, 0.5, 5);
        let err = Clock::new(offline, &earlier, SHAPE, 0).err().unwrap();
        assert!(
            matches!(&err, CoreError::Invalid(m) if m.contains("still moving")),
            "{err}"
        );
    }
}
