//! The M5 clock (DESIGN.md §7 M5, §0 condition 4): from the scan's end, the
//! arrival of its last frame ([`SourceStats::scan_end`]), to the first live
//! object that matches the offline run of the same algorithm on the same
//! frames in Fourier ring correlation over the scanned region.
//!
//! The region is the bounding box of the window centres, from the scan
//! positions alone. Outside it pixels are lit only by the probe's edges,
//! which constrain them in neither run: on a 6×6 raster landed in batches,
//! the live object holds the whole array's outer rings at 0.95 against the
//! offline one, the scanned region's at 0.992. [`Clock::whole`] gives the
//! whole array's figure, to report beside the clock.
//!
//! [`SourceStats::scan_end`]: crate::pipeline::SourceStats::scan_end

use std::fmt;
use std::ops::Range;
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

/// The offline reference has settled once its object at epoch `E`
/// correlates with its own at `E/2` at least this much in every ring,
/// checked as `E` doubles. AP's tail is slow: over a fixed lag a run can
/// pass 0.999 and still drift by the whole [`MATCH`] tolerance later, while
/// doubling bounds what is left to change by the last change.
pub const SETTLED: f64 = 0.999;

/// Why [`Clock::new`] refused its reference.
#[derive(Debug)]
pub enum ClockError {
    /// Ring `ring` of `rings` of the offline object correlates with its
    /// object at half the epochs only at `frc`, under [`SETTLED`].
    Moving {
        ring: usize,
        rings: usize,
        frc: f64,
    },
    Invalid(CoreError),
}

impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Moving { ring, rings, frc } => write!(
                f,
                "the offline run is still moving: ring {ring} of {rings} at FRC {frc:.4} \
                 against its object at half the epochs, under {SETTLED}"
            ),
            Self::Invalid(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for ClockError {}

impl From<CoreError> for ClockError {
    fn from(e: CoreError) -> Self {
        Self::Invalid(e)
    }
}

/// One scan's clock, started at its end.
pub struct Clock {
    offline: Vec<C64>,
    /// `offline` over the region.
    cropped: Vec<C64>,
    shape: [usize; 2],
    region: [Range<usize>; 2],
    scan_end: i64,
    settled: f64,
}

impl Clock {
    /// Against `offline`, the offline run's object, `[ny, nx]` = `shape`
    /// row-major, compared over the rows and columns of `region`.
    /// `half`, that run's object at half its epochs, must match it to
    /// [`SETTLED`] in every ring: a run still moving is no reference.
    /// `scan_end` is wall ns, as frame arrivals are.
    pub fn new(
        offline: &[C64],
        half: &[C64],
        shape: [usize; 2],
        region: [Range<usize>; 2],
        scan_end: i64,
    ) -> Result<Self, ClockError> {
        if region
            .iter()
            .zip(shape)
            .any(|(r, n)| r.is_empty() || r.end > n)
        {
            return Err(CoreError::Invalid(format!(
                "region {region:?} is empty or leaves the {shape:?} object"
            ))
            .into());
        }
        let cropped = crop(offline, shape, &region)?;
        let f = Frc::compute(
            &cropped,
            &crop(half, shape, &region)?,
            region_shape(&region),
        )?;
        if let Some(ring) = below(&f) {
            return Err(ClockError::Moving {
                ring,
                rings: f.rings.len(),
                frc: f.rings[ring].frc,
            });
        }
        Ok(Self {
            offline: offline.to_vec(),
            cropped,
            shape,
            region,
            scan_end,
            settled: lowest(&f),
        })
    }

    /// The lowest ring's FRC of the offline object against its own at half
    /// the epochs, over the region: how far the reference still moved.
    pub fn settled(&self) -> f64 {
        self.settled
    }

    /// Time from the scan's end to `taken` (wall ns) if the live object
    /// taken then, of the offline one's shape, matches it to [`MATCH`] in
    /// every ring; `None` below that or before the end, when frames are
    /// still missing. The first snapshot with a time is the converged one.
    pub fn check(&self, live: &[C64], taken: i64) -> Result<Option<Duration>, CoreError> {
        let Ok(after) = u64::try_from(taken - self.scan_end) else {
            return Ok(None);
        };
        Ok((self.region_frc(live)? >= MATCH).then(|| Duration::from_nanos(after)))
    }

    /// The lowest ring's FRC of `live` against the offline object over the
    /// region, the figure [`Self::check`] holds to [`MATCH`].
    pub fn region_frc(&self, live: &[C64]) -> Result<f64, CoreError> {
        let live = crop(live, self.shape, &self.region)?;
        let f = Frc::compute(&self.cropped, &live, region_shape(&self.region))?;
        Ok(lowest(&f))
    }

    /// The lowest ring's FRC of `live` against the offline object over the
    /// whole array, region or not.
    pub fn whole(&self, live: &[C64]) -> Result<f64, CoreError> {
        Ok(lowest(&Frc::compute(&self.offline, live, self.shape)?))
    }
}

/// `region`'s pixels of an object of `shape`.
fn crop(obj: &[C64], shape: [usize; 2], region: &[Range<usize>; 2]) -> Result<Vec<C64>, CoreError> {
    let [ny, nx] = shape;
    if obj.len() != ny * nx {
        return Err(CoreError::Invalid(format!(
            "{} pixels for a {shape:?} object",
            obj.len()
        )));
    }
    let [rows, cols] = region;
    Ok(rows
        .clone()
        .flat_map(|y| &obj[y * nx..][cols.clone()])
        .copied()
        .collect())
}

fn region_shape(region: &[Range<usize>; 2]) -> [usize; 2] {
    region.clone().map(|r| r.len())
}

/// The first ring under [`SETTLED`].
fn below(f: &Frc) -> Option<usize> {
    f.rings.iter().position(|r| r.frc < SETTLED)
}

fn lowest(f: &Frc) -> f64 {
    f.rings.iter().map(|r| r.frc).fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use crate::synth::{hash3, unit};

    use super::*;

    const SHAPE: [usize; 2] = [64, 64];
    const ALL: [Range<usize>; 2] = [0..64, 0..64];

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
        let half = plus(&offline, 0.02, 2);
        let clock = Clock::new(&offline, &half, SHAPE, ALL, 1_000).unwrap();
        assert!((SETTLED..1.0).contains(&clock.settled()));
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

    /// An offline run whose object at half the epochs is 0.1 away (FRC
    /// near 0.995, over [`MATCH`] but under [`SETTLED`]) or 0.5 away is still
    /// moving and starts no clock.
    #[test]
    fn an_offline_run_still_moving_is_no_reference() {
        let offline = noise(1);
        for (eps, seed) in [(0.1, 5), (0.5, 6)] {
            let half = plus(&offline, eps, seed);
            let err = Clock::new(&offline, &half, SHAPE, ALL, 0).err().unwrap();
            assert!(
                matches!(err, ClockError::Moving { frc, .. } if frc < SETTLED),
                "{err}"
            );
        }
    }

    /// `b` inside rows and columns 16..48, `a` outside.
    fn inside(a: &[C64], b: &[C64]) -> Vec<C64> {
        let core = 16..48;
        (0..SHAPE[0] * SHAPE[1])
            .map(|i| {
                let (y, x) = (i / SHAPE[1], i % SHAPE[1]);
                if core.contains(&y) && core.contains(&x) {
                    b[i]
                } else {
                    a[i]
                }
            })
            .collect()
    }

    /// Over the region 16..48 the clock sees only its pixels: an offline run
    /// still moving outside it is settled, and a live object 0.5 away
    /// outside it converges, while [`Clock::whole`] shows the gap. A region
    /// that is empty or leaves the object, or an object of another size, is
    /// refused.
    #[test]
    fn only_the_region_counts_and_the_whole_array_is_reported_beside_it() {
        let offline = noise(1);
        let region = [16..48, 16..48];
        let far = plus(&offline, 0.5, 6);
        let half = inside(&far, &offline);
        let clock = Clock::new(&offline, &half, SHAPE, region.clone(), 0).unwrap();
        let live = inside(&far, &plus(&offline, 0.05, 7));
        assert_eq!(
            clock.check(&live, 10).unwrap(),
            Some(Duration::from_nanos(10))
        );
        assert!(clock.region_frc(&live).unwrap() >= MATCH);
        assert!(clock.whole(&live).unwrap() < MATCH);
        assert!(clock.whole(&offline).unwrap() > 0.999_999);
        assert_eq!(clock.check(&inside(&live, &far), 10).unwrap(), None);
        assert!(clock.region_frc(&inside(&live, &far)).unwrap() < MATCH);
        assert!(clock.check(&live[1..], 10).is_err());
        for bad in [[16..48, 16..65], [16..16, 16..48]] {
            assert!(Clock::new(&offline, &offline, SHAPE, bad, 0).is_err());
        }
    }
}
