//! The CG kernels against f64 references of their documented formulas.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs_gpu::{C32, CG_BLOCKS, GradKernels, Kernels, UpdateSums};

/// A stream, an eager execution context on it and the CG kernels.
fn setup() -> (Arc<CudaStream>, ExecutionContext, GradKernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let grad = GradKernels::load(&kernels).expect("bind the CG kernels");
    let stream = ctx.default_stream();
    (stream.clone(), ExecutionContext::new(stream), grad)
}

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

fn neg(v: &[C32]) -> Vec<C32> {
    v.iter()
        .map(|c| C32 {
            re: -c.re,
            im: -c.im,
        })
        .collect()
}

/// Two arrays, the larger past `CG_BLOCKS * PIXEL_THREADS` so threads stride, into one parts
/// buffer: the summed pair is the f64 Polak–Ribière sums of both, and `h` ends as `g`.
#[test]
fn beta_sums_both_arrays_and_keeps_the_gradient() {
    let (stream, ctx, k) = setup();
    let mut parts = DeviceBuffer::<f32>::zeroed(&stream, 4 * CG_BLOCKS as usize).unwrap();
    let mut pair = DeviceBuffer::<f32>::zeroed(&stream, 2).unwrap();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (slot, n) in [1000usize, 300_000].into_iter().enumerate() {
        let (g, h) = (
            noise(n, 3 + slot as u64, 1.0),
            noise(n, 5 + slot as u64, 1.0),
        );
        for (g, h) in g.iter().zip(&h) {
            let (g, h) = ((g.re as f64, g.im as f64), (h.re as f64, h.im as f64));
            num += g.0 * (g.0 - h.0) + g.1 * (g.1 - h.1);
            den += h.0 * h.0 + h.1 * h.1;
        }
        let dg = up(&stream, &g);
        let mut dh = up(&stream, &h);
        let first = slot as u32 * CG_BLOCKS;
        k.beta_parts(&ctx, &dg, &mut dh, n as u32, &mut parts, first)
            .unwrap();
        assert_eq!(dh.to_host_vec(&stream).unwrap(), g);
    }
    k.sum_parts(&ctx, &parts, 2 * CG_BLOCKS, &mut pair).unwrap();
    let got = pair.to_host_vec(&stream).unwrap();
    for (g, w) in got.iter().zip([num, den]) {
        assert!(
            ((*g as f64) - w).abs() < 1e-5 * w.abs(),
            "{got:?} vs {num} {den}"
        );
    }
}

/// `d = β d - g` for a positive `β`, and the reset `d = -g` for a negative, infinite and
/// NaN quotient; the gradient sums are zero after.
#[test]
fn direction_combines_or_resets_and_consumes_the_gradient() {
    let (stream, ctx, k) = setup();
    let n = 1000usize;
    let (d0, g) = (noise(n, 7, 2.0), noise(n, 11, 3.0));
    let den: Vec<f32> = (0..n).map(|i| i as f32).collect();
    for (b_num, b_den) in [(3.0f32, 2.0f32), (-1.0, 2.0), (1.0, 0.0), (0.0, 0.0)] {
        let mut d = up(&stream, &d0);
        let mut sums = UpdateSums::new(up(&stream, &g), up(&stream, &den), &stream).unwrap();
        let beta = up(&stream, &[b_num, b_den]);
        k.direction(&ctx, &mut d, &mut sums, n as u32, &beta)
            .unwrap();
        let got = d.to_host_vec(&stream).unwrap();
        let b = b_num / b_den;
        if b.is_finite() && b >= 0.0 {
            let b = b as f64;
            for ((r, d), g) in got.iter().zip(&d0).zip(&g) {
                let want = (b * d.re as f64 - g.re as f64, b * d.im as f64 - g.im as f64);
                let err = (r.re as f64 - want.0).hypot(r.im as f64 - want.1);
                assert!(
                    err < 1e-6 * want.0.hypot(want.1).max(1.0),
                    "{r:?} vs {want:?}"
                );
            }
        } else {
            assert_eq!(got, neg(&g), "beta pair ({b_num}, {b_den})");
        }
        let num = sums.num.to_host_vec(&stream).unwrap();
        assert!(num.iter().all(|c| *c == C32::default()));
        assert!(
            sums.den
                .to_host_vec(&stream)
                .unwrap()
                .iter()
                .all(|v| *v == 0.0)
        );
    }
}

/// `x += γ d` for a finite `γ`; for an infinite or NaN quotient `x` stays and `d = -h`.
#[test]
fn step_moves_or_resets_the_direction() {
    let (stream, ctx, k) = setup();
    let n = 1000usize;
    let (x0, d0, h) = (noise(n, 13, 2.0), noise(n, 17, 3.0), noise(n, 19, 1.0));
    let dh = up(&stream, &h);
    for (g_num, g_den) in [(1.0f32, 4.0f32), (1.0, 0.0), (0.0, 0.0)] {
        let (mut x, mut d) = (up(&stream, &x0), up(&stream, &d0));
        let gamma = up(&stream, &[g_num, g_den]);
        k.step(&ctx, &mut x, &mut d, &dh, n as u32, &gamma).unwrap();
        let (gx, gd) = (
            x.to_host_vec(&stream).unwrap(),
            d.to_host_vec(&stream).unwrap(),
        );
        let c = g_num / g_den;
        if c.is_finite() {
            let c = c as f64;
            for ((r, x), d) in gx.iter().zip(&x0).zip(&d0) {
                let want = (x.re as f64 + c * d.re as f64, x.im as f64 + c * d.im as f64);
                let err = (r.re as f64 - want.0).hypot(r.im as f64 - want.1);
                assert!(
                    err < 1e-6 * want.0.hypot(want.1).max(1.0),
                    "{r:?} vs {want:?}"
                );
            }
            assert_eq!(gd, d0);
        } else {
            assert_eq!(gx, x0, "gamma pair ({g_num}, {g_den})");
            assert_eq!(gd, neg(&h), "gamma pair ({g_num}, {g_den})");
        }
    }
}
