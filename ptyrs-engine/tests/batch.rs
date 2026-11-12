//! [`random_batches`]: a partition of the frames into sorted batches, reproducible from the seed
//! and the epoch and different across them, each frame equally likely in every batch.

use ptyrs_engine::random_batches;

#[test]
fn batches_partition_frames_in_scan_order() {
    for (nframe, size, lens) in [
        (40, 16, vec![16, 16, 8]),
        (40, 20, vec![20, 20]),
        (5, 8, vec![5]),
        (1, 1, vec![1]),
        (0, 4, vec![]),
    ] {
        for epoch in 0..4 {
            let b = random_batches(nframe, size, 3, epoch);
            assert_eq!(b.iter().map(Vec::len).collect::<Vec<_>>(), lens);
            assert!(b.iter().all(|c| c.is_sorted()));
            let mut all: Vec<usize> = b.concat();
            all.sort_unstable();
            assert_eq!(all, (0..nframe).collect::<Vec<_>>());
        }
    }
}

#[test]
fn batches_follow_seed_and_epoch() {
    let b = random_batches(40, 16, 3, 5);
    assert_eq!(random_batches(40, 16, 3, 5), b);
    assert_ne!(random_batches(40, 16, 3, 6), b);
    assert_ne!(random_batches(40, 16, 4, 5), b);
}

/// Over 4000 epochs each of 10 frames lands in the first of two batches `2000 ± 160` times
/// (5σ of the binomial count).
#[test]
fn every_frame_is_equally_likely_in_a_batch() {
    let mut count = [0usize; 10];
    for epoch in 0..4000 {
        random_batches(10, 5, 11, epoch)[0]
            .iter()
            .for_each(|&f| count[f] += 1);
    }
    assert!(
        count.iter().all(|&c| c.abs_diff(2000) < 160),
        "first-batch counts {count:?}"
    );
}
