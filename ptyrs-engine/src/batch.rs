//! Seeded random mini-batches, pty-chi's default `BatchingModes.RANDOM` (a torch `DataLoader`
//! with `shuffle=True`: a fresh permutation of the frames every epoch, the last batch shorter)
//! from a generator any backend can reproduce bit for bit. Each batch is in scan order, so its
//! sums run in one order whatever the permutation; consecutive scan-order batches would be
//! spatially clustered and bias the mini-batch steps.

/// SplitMix64's output function (Steele, Lea, Flood 2014), a bijection on `u64`.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The frames `0..nframe` of epoch `epoch` in batches of `size`, the last one shorter: a
/// Fisher–Yates permutation from the end, swapping `i` with `⌊r·(i + 1)/2⁶⁴⌋` for the next
/// SplitMix64 output `r` from state `seed ^ mix(epoch)`, cut into consecutive chunks, each
/// sorted ascending.
pub fn random_batches(nframe: usize, size: usize, seed: u64, epoch: u64) -> Vec<Vec<usize>> {
    assert!(size > 0, "batch size must be positive");
    let mut state = seed ^ mix(epoch);
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix(state)
    };
    let mut order: Vec<usize> = (0..nframe).collect();
    for i in (1..nframe).rev() {
        let j = ((next() as u128 * (i as u128 + 1)) >> 64) as usize;
        order.swap(i, j);
    }
    order
        .chunks(size)
        .map(|c| {
            let mut c = c.to_vec();
            c.sort_unstable();
            c
        })
        .collect()
}
