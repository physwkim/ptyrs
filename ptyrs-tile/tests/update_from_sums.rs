//! `ptyrs_tile::update_from_sums` against the SIMT `ptyrs_gpu::Kernels::update_from_sums`
//! and an f64 reference, on the data of `ptyrs-gpu`'s `update_from_sums` test.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs_gpu::{C32, Kernels, UpdateSums};

/// Deterministic values in `[-scale, scale)`.
fn noise(len: usize, seed: u64, scale: f32) -> Vec<C32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
    };
    (0..len)
        .map(|_| C32 {
            re: next(),
            im: next(),
        })
        .collect()
}

fn up<T: DeviceCopy>(stream: &CudaStream, host: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(stream, host).unwrap()
}

fn flat(v: &[C32]) -> Vec<f32> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
}

/// `sqrt(sum (got - want)^2 / sum want^2)`.
fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (e, r) = got.iter().zip(want).fold((0.0, 0.0), |(e, r), (&g, &w)| {
        (e + (g as f64 - w).powi(2), r + w * w)
    });
    (e / r).sqrt()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn update_from_sums_matches_simt() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    let inertia = 0.01f32;
    // 300_001 values span several max programs; their maximum is the last value. The 1500
    // have no positive `den` (negatives, -0, NaN), so `reg` is 0 and the update is exact.
    for (len, seed, positive) in [
        (6000usize, 20u64, true),
        (300_001, 30, true),
        (1500, 40, false),
    ] {
        let x_host = noise(len, seed, 1.0);
        let num_host = noise(len, seed + 1, 3.0);
        let mut den_host: Vec<f32> = noise(len, seed + 2, 2.0)
            .iter()
            .map(|c| c.re.abs())
            .collect();
        if positive {
            den_host[len - 1] = 7.5;
        } else {
            den_host.iter_mut().for_each(|v| *v = -*v);
            den_host[100] = f32::NAN;
            den_host[200] = -0.0;
        }
        let reg = inertia as f64 * den_host.iter().fold(0.0f64, |m, &v| m.max(v as f64));
        let want: Vec<f64> = (0..len)
            .flat_map(|i| {
                let (a, b) = (num_host[i], x_host[i]);
                let d = (den_host[i] as f64 + reg).max(1e-12);
                [
                    (a.re as f64 + reg * b.re as f64) / d,
                    (a.im as f64 + reg * b.im as f64) / d,
                ]
            })
            .collect();
        let mut sums =
            UpdateSums::new(up(&stream, &num_host), up(&stream, &den_host), &stream).unwrap();
        let mut simt = up(&stream, &x_host);
        k.update_from_sums(&ex, &mut simt, &mut sums, len as u32, inertia)
            .unwrap();
        let simt = flat(&simt.to_host_vec(&stream).unwrap());
        // Stale scratch above every `den`: the op must zero it itself.
        sums.max_bits = up(&stream, &[1e30f32.to_bits()]);
        let mut run = || {
            let mut x = up(&stream, &x_host);
            ptyrs_tile::update_from_sums(&ex, &mut x, &mut sums, len as u32, inertia).unwrap();
            flat(&x.to_host_vec(&stream).unwrap())
        };
        let tile = run();

        let simt64: Vec<f64> = simt.iter().map(|&v| v as f64).collect();
        let (vs_simt, vs_f64) = (rel_l2(&tile, &simt64), rel_l2(&tile, &want));
        let same = tile
            .iter()
            .zip(&simt)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!(
            "update_from_sums n={len}: rel L2 Tile-SIMT {vs_simt:.2e}, Tile-f64 {vs_f64:.2e}; \
             {same} of {} values bit-equal to SIMT",
            tile.len()
        );
        assert!(vs_simt < 1e-6, "n={len}: Tile vs SIMT rel L2 {vs_simt:.2e}");
        assert!(vs_f64 < 1e-6, "n={len}: Tile vs f64 rel L2 {vs_f64:.2e}");
        if !positive {
            assert_eq!(bits(&tile), bits(&simt), "n={len}: reg 0 is not exact");
        }
        assert_eq!(bits(&run()), bits(&tile), "n={len}: differs run to run");
    }
}
