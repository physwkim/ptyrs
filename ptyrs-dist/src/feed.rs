//! How evenly a live scan feeds the regions (DESIGN §7 M4): frames arrive in frame order and
//! each goes to the region that computes it, so the new frames a GPU gets in a stretch of the
//! measurement are the counts per region over consecutive arrivals.

use crate::Partition;

/// New frames per region in every window of `len` consecutive arrivals, one window per start
/// `0..=nframe - len`; empty when the scan is shorter than `len`. Windows start at every
/// arrival because live batches are not aligned to scan rows or spiral turns.
///
/// # Panics
/// If `len` is 0.
pub fn windows(part: &Partition, len: usize) -> Vec<Vec<usize>> {
    assert!(len > 0, "window of 0 arrivals");
    let n = part.nframe();
    if n < len {
        return Vec::new();
    }
    let mut counts = vec![0; part.nregion()];
    for f in 0..len {
        counts[part.computed_by(f)] += 1;
    }
    let mut out = Vec::with_capacity(n - len + 1);
    out.push(counts.clone());
    for f in len..n {
        counts[part.computed_by(f - len)] -= 1;
        counts[part.computed_by(f)] += 1;
        out.push(counts.clone());
    }
    out
}

/// Most minus fewest new frames over the regions of one window.
pub fn spread(counts: &[usize]) -> usize {
    counts.iter().max().unwrap_or(&0) - counts.iter().min().unwrap_or(&0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::tests::addr;
    use crate::{DEFAULT_TILE, Scheme};
    use ptyrs_core::sim;

    #[test]
    fn windows_count_each_stretch_of_arrivals() {
        let p = Partition::from_addr(
            &addr(sim::raster(12, 20, 8e-8), 32),
            DEFAULT_TILE,
            Scheme::Strips,
            4,
        )
        .unwrap();
        let n = p.nframe();
        for len in [1, 7, 20, n] {
            let w = windows(&p, len);
            assert_eq!(w.len(), n - len + 1, "len {len}");
            for (start, counts) in w.iter().enumerate() {
                let mut want = vec![0; 4];
                for f in start..start + len {
                    want[p.computed_by(f)] += 1;
                }
                assert_eq!(counts, &want, "len {len} start {start}");
            }
        }
        assert!(windows(&p, n + 1).is_empty());
        assert_eq!(spread(&[3, 7, 5]), 4);
    }

    /// Every raster row gives each strip the same frames, so a window is whole rows plus one
    /// arc of a row: a row-long window gets exactly the row shares, and any window's spread is
    /// at most the largest share plus the share difference for every whole row. A slow-axis cut
    /// would give a whole row to one GPU.
    #[test]
    fn strips_on_a_raster_stay_within_one_row_share() {
        let (ny, nx) = (12, 20);
        let a = addr(sim::raster(ny, nx, 8e-8), 32);
        for nregion in [2, 4] {
            let p = Partition::from_addr(&a, DEFAULT_TILE, Scheme::Strips, nregion).unwrap();
            for f in nx..ny * nx {
                assert_eq!(p.computed_by(f), p.computed_by(f % nx), "frame {f}");
            }
            let mut share = vec![0; nregion];
            for f in 0..nx {
                share[p.computed_by(f)] += 1;
            }
            assert!(!share.contains(&0), "{share:?}");
            let (hi, lo) = (*share.iter().max().unwrap(), *share.iter().min().unwrap());
            assert!(windows(&p, nx).iter().all(|c| *c == share));
            for len in [8, 16, 2 * nx, 64, 100] {
                for (start, c) in windows(&p, len).iter().enumerate() {
                    assert!(
                        spread(c) <= hi + len / nx * (hi - lo),
                        "{nregion} regions, len {len} start {start}: {c:?}, row shares {share:?}"
                    );
                }
            }
        }
    }

    /// Golden-angle steps put consecutive spiral points in different sectors, so with a step of
    /// five tiles every GPU gets new frames in every 8 arrivals, and the spread stays at most 4
    /// (as measured) however long the window.
    #[test]
    fn sectors_on_a_spiral_feed_every_gpu_in_8_arrivals() {
        let a = addr(sim::fermat(2000, 8e-7), 256);
        for nregion in [2, 4] {
            let p =
                Partition::from_addr(&a, DEFAULT_TILE, Scheme::Sectors { centre: None }, nregion)
                    .unwrap();
            assert!(windows(&p, 8).iter().all(|c| !c.contains(&0)));
            for len in [8, 16, 64, 256, 1000] {
                for (start, c) in windows(&p, len).iter().enumerate() {
                    assert!(
                        spread(c) <= 4,
                        "{nregion} regions, len {len} start {start}: {c:?}"
                    );
                }
            }
        }
    }
}
