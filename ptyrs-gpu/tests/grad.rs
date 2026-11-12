//! The CG kernels against f64 references of their documented formulas, and the line-search
//! exit waves against `Kernels::obj_probe_to_psi`.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs_gpu::{
    C32, CG_BLOCKS, GammaStacks, GradKernels, Kernels, MASKED, StackTable, UpdateSums, ViewAddr,
    ViewDims, Views,
};

/// A stream, an eager execution context on it, the M1 kernels and the CG kernels.
fn setup() -> (Arc<CudaStream>, ExecutionContext, Kernels, GradKernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let grad = GradKernels::load(&kernels).expect("bind the CG kernels");
    let stream = ctx.default_stream();
    (stream.clone(), ExecutionContext::new(stream), kernels, grad)
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
    let (stream, ctx, _, k) = setup();
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
    let (stream, ctx, _, k) = setup();
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
    let (stream, ctx, _, k) = setup();
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

/// Five 8x8 frames of two probe modes on a 24x24 object, at fractional offsets with a
/// far-field phase, cut into stacks `0..3` and `3..5` (`nf_max = 3`).
fn scan(stream: &CudaStream) -> (Views, StackTable) {
    let dims = ViewDims {
        nframe: 5,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [8, 8],
        obj: [24, 24],
        quad_phase: 0.01,
    };
    let offsets = [
        (0, 0, 0.0, 0.0),
        (3, 5, 0.25, 0.0),
        (7, 2, 0.5, 0.75),
        (10, 12, 0.0, 0.3),
        (15, 15, 0.9, 0.1),
    ];
    let addr: Vec<ViewAddr> = offsets
        .iter()
        .flat_map(|&(oy, ox, fy, fx)| {
            (0..2).map(move |m| ViewAddr {
                oy,
                ox,
                fy,
                fx,
                probe: m,
                obj: 0,
            })
        })
        .collect();
    let views = Views::new(stream, dims, &addr).unwrap();
    let table = StackTable::new(stream, &dims, &[0..3, 3..5]).unwrap();
    (views, table)
}

/// `po` and `b` are `obj_probe_to_psi` of `(O, P)` and `(dO, dP)` bit for bit, `a` the sum
/// of those of `(dO, P)` and `(O, dP)` to rounding, on a full and a short stack.
#[test]
fn gamma_fields_are_the_exit_waves_of_the_directions() {
    let (stream, ctx, m1, k) = setup();
    let (views, table) = scan(&stream);
    let len = 3 * 2 * 64;
    let (obj, d_obj) = (noise(24 * 24, 1, 1.0), noise(24 * 24, 2, 0.5));
    let (probe, d_probe) = (noise(2 * 64, 3, 2.0), noise(2 * 64, 4, 0.7));
    let (obj, d_obj, probe, d_probe) = (
        up(&stream, &obj),
        up(&stream, &d_obj),
        up(&stream, &probe),
        up(&stream, &d_probe),
    );
    let psi = |o: &DeviceBuffer<C32>, p: &DeviceBuffer<C32>, s: usize| {
        let mut psi = DeviceBuffer::<C32>::zeroed(&stream, len).unwrap();
        m1.obj_probe_to_psi(&ctx, &views, table.at(s), o, p, &mut psi)
            .unwrap();
        psi.to_host_vec(&stream).unwrap()
    };
    for s in 0..2 {
        let mut out = GammaStacks::zeroed(&stream, len).unwrap();
        k.gamma_fields(
            &ctx,
            &views,
            table.at(s),
            [&obj, &probe],
            [&d_obj, &d_probe],
            &mut out,
        )
        .unwrap();
        assert_eq!(out.po.to_host_vec(&stream).unwrap(), psi(&obj, &probe, s));
        assert_eq!(
            out.b.to_host_vec(&stream).unwrap(),
            psi(&d_obj, &d_probe, s)
        );
        let (a0, a1) = (psi(&d_obj, &probe, s), psi(&obj, &d_probe, s));
        for (i, got) in out.a.to_host_vec(&stream).unwrap().iter().enumerate() {
            let want = (
                a0[i].re as f64 + a1[i].re as f64,
                a0[i].im as f64 + a1[i].im as f64,
            );
            let err = (got.re as f64 - want.0).hypot(got.im as f64 - want.1);
            assert!(
                err < 1e-5 * want.0.hypot(want.1).max(1.0),
                "a[{i}] {got:?} vs {want:?}"
            );
        }
    }
}

/// Each frame's pair is the f64 `CG_Poisson_Gamma` sum of its unmasked pixels, frames outside
/// the stack keep their pair, and `sum_parts` of all frames' pairs is their total.
#[test]
fn gamma_frames_match_f64_and_sum_to_gamma() {
    let (stream, ctx, _, k) = setup();
    let (views, table) = scan(&stream);
    let (ny, nx, nmode, n) = (8usize, 8usize, 2usize, 64usize);
    let counts: Vec<u16> = (0..5 * n)
        .map(|i| {
            if i % 13 == 5 {
                MASKED
            } else {
                (i * 37 % 23) as u16
            }
        })
        .collect();
    let dcounts = up(&stream, &counts);
    let mut pairs = up(&stream, &[7.0f32; 10]);
    let mut total = (0.0f64, 0.0f64);
    for (s, f0, nf) in [(0usize, 0usize, 3usize), (1, 3, 2)] {
        let len = 3 * nmode * n;
        let x = [
            noise(len, 10 + s as u64, 20.0),
            noise(len, 20 + s as u64, 5.0),
            noise(len, 30 + s as u64, 2.0),
        ];
        let stacks = GammaStacks {
            po: up(&stream, &x[0]),
            a: up(&stream, &x[1]),
            b: up(&stream, &x[2]),
        };
        let before = pairs.to_host_vec(&stream).unwrap();
        k.gamma_frames(
            &ctx,
            views.dims(),
            table.at(s),
            &dcounts,
            &stacks,
            &mut pairs,
        )
        .unwrap();
        let got = pairs.to_host_vec(&stream).unwrap();
        for f in 0..5 {
            let (lo, hi) = (2 * f, 2 * f + 2);
            if !(f0..f0 + nf).contains(&f) {
                assert_eq!(got[lo..hi], before[lo..hi], "frame {f} outside stack {s}");
                continue;
            }
            let fl = f - f0;
            let (mut num, mut den, mut scale) = (0.0f64, 0.0f64, [0.0f64; 2]);
            for d in 0..n {
                let obs = counts[f * n + d];
                if obs == MASKED {
                    continue;
                }
                let at = ((d / nx + ny / 2) % ny) * nx + (d % nx + nx / 2) % nx;
                let (mut calc, mut r, mut sq) = (0.0f64, 0.0f64, 0.0f64);
                for m in 0..nmode {
                    let j = (fl * nmode + m) * n + at;
                    let c = |v: &Vec<C32>| (v[j].re as f64, v[j].im as f64);
                    let (p, a, b) = (c(&x[0]), c(&x[1]), c(&x[2]));
                    calc += p.0 * p.0 + p.1 * p.1;
                    r += p.0 * a.0 + p.1 * a.1;
                    sq += a.0 * a.0 + a.1 * a.1 + 2.0 * (p.0 * b.0 + p.1 * b.1);
                }
                let calc = (calc / n as f64).max(1e-12);
                let (r, sq, obs) = (r / n as f64, sq / n as f64, obs as f64);
                let f = 1.0 - obs / calc;
                let (tn, td) = (-2.0 * r * f, 2.0 * sq * f + obs * (2.0 * r / calc).powi(2));
                num += tn;
                den += td;
                scale[0] += tn.abs();
                scale[1] += td.abs();
            }
            for (i, w) in [num, den].into_iter().enumerate() {
                let e = (got[lo + i] as f64 - w).abs();
                assert!(
                    e < 1e-5 * scale[i],
                    "frame {f} pair {i}: {} vs {w}",
                    got[lo + i]
                );
            }
            total.0 += got[lo] as f64;
            total.1 += got[lo + 1] as f64;
        }
    }
    let mut gamma = DeviceBuffer::<f32>::zeroed(&stream, 2).unwrap();
    k.sum_parts(&ctx, &pairs, 5, &mut gamma).unwrap();
    let got = gamma.to_host_vec(&stream).unwrap();
    for (g, w) in got.iter().zip([total.0, total.1]) {
        assert!(
            ((*g as f64) - w).abs() < 1e-6 * w.abs(),
            "{got:?} vs {total:?}"
        );
    }
}
