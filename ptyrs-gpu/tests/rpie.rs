//! The rPIE kernels against f64 references of their documented formulas.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs_gpu::{C32, Kernels, MASKED, RpieKernels, ViewAddr, ViewDims, Views, Weight};

/// A stream, an eager execution context on it and the rPIE kernels.
fn setup() -> (Arc<CudaStream>, ExecutionContext, RpieKernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let rpie = RpieKernels::load(&kernels).expect("bind the rPIE kernels");
    let stream = ctx.default_stream();
    (stream.clone(), ExecutionContext::new(stream), rpie)
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

fn norm2(c: C32) -> f64 {
    (c.re as f64).powi(2) + (c.im as f64).powi(2)
}

/// Largest `|got - want| / max(|want|, floor)` over complex elements.
fn max_rel(got: &[C32], want: &[(f64, f64)], floor: f64) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let e = ((g.re as f64 - w.0).powi(2) + (g.im as f64 - w.1).powi(2)).sqrt();
            e / (w.0.hypot(w.1)).max(floor)
        })
        .fold(0.0, f64::max)
}

#[test]
fn amplitude_matches_ptychi_formula() {
    let (stream, ctx, k) = setup();
    let (ny, nx, nmode, nframe) = (8usize, 8usize, 2usize, 3usize);
    let n = ny * nx;
    let dims = ViewDims {
        nframe: nframe as u32,
        nmode: nmode as u32,
        nprobe: nmode as u32,
        nobj: 1,
        frame: [ny as u32, nx as u32],
        obj: [16, 16],
        quad_phase: 0.0,
    };
    let addr: Vec<ViewAddr> = (0..nframe * nmode)
        .map(|v| ViewAddr {
            oy: (v / nmode) as u32,
            ox: 2,
            probe: (v % nmode) as u32,
            ..Default::default()
        })
        .collect();
    let views = Views::new(&stream, dims, &addr).unwrap();
    let counts: Vec<u16> = (0..nframe * n)
        .map(|i| {
            if i % 11 == 3 {
                MASKED
            } else {
                (i * 37 % 200) as u16
            }
        })
        .collect();
    let (f0, f1) = (1usize, 3usize);
    let psi = noise((f1 - f0) * nmode * n, 7, 2.0);
    let mut want = vec![(0.0, 0.0); psi.len()];
    for fl in 0..f1 - f0 {
        for d in 0..n {
            let at = ((d / nx + ny / 2) % ny) * nx + (d % nx + nx / 2) % nx;
            let obs = counts[(f0 + fl) * n + d];
            let s = if obs == MASKED {
                1.0 / n as f64
            } else {
                let calc = ((0..nmode)
                    .map(|m| norm2(psi[(fl * nmode + m) * n + at]))
                    .sum::<f64>()
                    / n as f64)
                    .sqrt();
                (obs as f64 + 1e-7).sqrt() / (calc + 1e-7) / n as f64
            };
            for m in 0..nmode {
                let j = (fl * nmode + m) * n + at;
                want[j] = (psi[j].re as f64 * s, psi[j].im as f64 * s);
            }
        }
    }
    let c = up(&stream, &counts);
    let mut p = up(&stream, &psi);
    k.amplitude(&ctx, &views, f0 as u32..f1 as u32, &c, &mut p)
        .unwrap();
    let got = p.to_host_vec(&stream).unwrap();
    let err = max_rel(&got, &want, 1e-6);
    assert!(err < 1e-5, "amplitude max rel err {err:e}");
}

#[test]
fn power_max_and_weights_match_f64() {
    let (stream, ctx, k) = setup();
    let (nk, n) = (3usize, 300usize);
    let x = noise(nk * n, 3, 1.0);
    let w = noise(nk * n, 5, 1.5);
    // Summed over all planes, and per plane.
    let pw = |p: &[C32], k0: usize, nsum: usize| {
        (0..n)
            .map(|i| (0..nsum).map(|m| norm2(p[(k0 + m) * n + i])).sum::<f64>())
            .fold(0.0, f64::max)
    };
    let all = pw(&w, 0, nk);
    let per: Vec<f64> = (0..nk).map(|b| pw(&w, b, 1)).collect();
    let dw = up(&stream, &w);
    let mut m_all = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
    let mut m_per = DeviceBuffer::<f32>::zeroed(&stream, nk).unwrap();
    k.power_max(&ctx, &dw, 1, nk as u32, n as u32, &mut m_all)
        .unwrap();
    k.power_max(&ctx, &dw, nk as u32, 1, n as u32, &mut m_per)
        .unwrap();
    let got_all = m_all.to_host_vec(&stream).unwrap()[0] as f64;
    assert!((got_all - all).abs() < 1e-6 * all, "{got_all} vs {all}");
    for (g, w) in m_per.to_host_vec(&stream).unwrap().iter().zip(&per) {
        assert!((*g as f64 - w).abs() < 1e-6 * w, "{g} vs {w}");
    }

    let alpha = 0.3;
    for (per_plane, mx) in [(false, &m_all), (true, &m_per)] {
        let want: Vec<(f64, f64)> = (0..nk * n)
            .map(|i| {
                let m = if per_plane { per[i / n] } else { all };
                let d = (1.0 - alpha) * norm2(w[i]) + alpha * m;
                (x[i].re as f64 / d, x[i].im as f64 / d)
            })
            .collect();
        let spec = Weight {
            nk: nk as u32,
            n: n as u32,
            per_plane,
            alpha: alpha as f32,
        };
        let dx = up(&stream, &x);
        let mut out = DeviceBuffer::<C32>::zeroed(&stream, nk * n).unwrap();
        k.weight(&ctx, &dx, &dw, mx, &mut out, spec).unwrap();
        let got = out.to_host_vec(&stream).unwrap();
        let err = max_rel(&got, &want, 0.0);
        assert!(err < 1e-5, "weight per_plane={per_plane}: {err:e}");
        let mut inplace = up(&stream, &x);
        k.weight_inplace(&ctx, &mut inplace, &dw, mx, spec).unwrap();
        assert_eq!(inplace.to_host_vec(&stream).unwrap(), got);
    }
}

#[test]
fn rpie_shapes_are_checked() {
    let (stream, ctx, k) = setup();
    let w = up(&stream, &noise(8, 1, 1.0));
    let mx = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
    let mut out = DeviceBuffer::<C32>::zeroed(&stream, 8).unwrap();
    let spec = Weight {
        nk: 2,
        n: 4,
        per_plane: true,
        alpha: 0.1,
    };
    // `per_plane` needs one max per plane.
    assert!(k.weight(&ctx, &w, &w, &mx, &mut out, spec).is_err());
    let mut short = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
    assert!(k.power_max(&ctx, &w, 2, 1, 8, &mut short).is_err());
}
