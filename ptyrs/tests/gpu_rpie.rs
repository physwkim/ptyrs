//! The rPIE kernels against f64 references of their documented formulas.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs::gpu::{
    C32, Kernels, MASKED, POWER_CHUNK, PowerMax, RpieKernels, StackTable, UpdateSums, ViewAddr,
    ViewDims, Views, Weight,
};

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
    let counts: Vec<u16> = (0..nframe * n)
        .map(|i| {
            if i % 11 == 3 {
                MASKED
            } else {
                (i * 37 % 200) as u16
            }
        })
        .collect();
    // Entry 0 fills the `nf_max = 2` frames of `psi`; entry 1 only its first.
    let table = StackTable::new(&stream, &dims, &[1..3, 0..1]).unwrap();
    let c = up(&stream, &counts);
    let psi = noise(2 * nmode * n, 7, 2.0);
    for (s, f0, nf) in [(0, 1usize, 2usize), (1, 0, 1)] {
        let mut want: Vec<(f64, f64)> = psi.iter().map(|v| (v.re as f64, v.im as f64)).collect();
        for fl in 0..nf {
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
        let mut p = up(&stream, &psi);
        k.amplitude(&ctx, &dims, table.at(s), &c, &mut p).unwrap();
        let got = p.to_host_vec(&stream).unwrap();
        let err = max_rel(&got, &want, 1e-6);
        assert!(err < 1e-5, "amplitude entry {s} max rel err {err:e}");
        // Planes past the entry's frames are untouched.
        assert_eq!(got[nf * nmode * n..], psi[nf * nmode * n..], "entry {s}");
    }
}

#[test]
fn power_max_and_weights_match_f64() {
    let (stream, ctx, k) = setup();
    // Two chunks per plane, the last one short; plane 0's max in the short chunk, plane 2's
    // in the first.
    let (nk, n) = (3usize, 3072usize);
    assert_eq!(n.div_ceil(POWER_CHUNK as usize), 2);
    let x = noise(nk * n, 3, 1.0);
    let mut w = noise(nk * n, 5, 1.5);
    w[n - 1] = C32 { re: 4.0, im: 3.0 };
    w[2 * n + 7] = C32 { re: -3.0, im: 4.5 };
    // Summed over all planes, and per plane.
    let pw = |p: &[C32], k0: usize, nsum: usize| {
        (0..n)
            .map(|i| (0..nsum).map(|m| norm2(p[(k0 + m) * n + i])).sum::<f64>())
            .fold(0.0, f64::max)
    };
    let all = pw(&w, 0, nk);
    let per: Vec<f64> = (0..nk).map(|b| pw(&w, b, 1)).collect();
    let dw = up(&stream, &w);
    let (all_groups, per_groups) = (
        PowerMax {
            ngroup: 1,
            nsum: nk as u32,
            n: n as u32,
        },
        PowerMax {
            ngroup: nk as u32,
            nsum: 1,
            n: n as u32,
        },
    );
    assert_eq!(per_groups.chunks(), 2 * nk);
    let mut chunks = DeviceBuffer::<f32>::zeroed(&stream, per_groups.chunks()).unwrap();
    let mut m_all = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
    let mut m_per = DeviceBuffer::<f32>::zeroed(&stream, nk).unwrap();
    k.power_max(&ctx, &dw, all_groups, &mut chunks, &mut m_all)
        .unwrap();
    k.power_max(&ctx, &dw, per_groups, &mut chunks, &mut m_per)
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

/// The f64 formula on two stacks of a split, planes past each stack untouched, and every
/// frame's waves the same bits in the split as in one stack of the whole scan.
#[test]
fn obj_wave_to_psi_matches_f64() {
    let (stream, ctx, k) = setup();
    let (ny, nx, nprobe, nframe) = (8usize, 8usize, 2usize, 3usize);
    let (nyo, nxo) = (16usize, 16usize);
    let n = ny * nx;
    let dims = ViewDims {
        nframe: nframe as u32,
        nmode: nprobe as u32,
        nprobe: nprobe as u32,
        nobj: 1,
        frame: [ny as u32, nx as u32],
        obj: [nyo as u32, nxo as u32],
        quad_phase: 0.0,
    };
    // Frame 1 on whole pixels, so its zero-weight corners are not read.
    let corners = [(3, 5, 0.25f32, 0.5f32), (0, 0, 0.0, 0.0), (7, 2, 0.75, 0.0)];
    let addr: Vec<ViewAddr> = corners
        .iter()
        .flat_map(|&(oy, ox, fy, fx)| {
            (0..nprobe as u32).map(move |p| ViewAddr {
                oy,
                ox,
                fy,
                fx,
                probe: p,
                obj: 0,
            })
        })
        .collect();
    let views = Views::new(&stream, dims, &addr).unwrap();
    let obj = noise(nyo * nxo, 3, 1.0);
    let dobj = up(&stream, &obj);
    let fill = noise(2 * nprobe * n, 9, 1.0);
    let whole_w = noise(nframe * nprobe * n, 5, 1.5);
    let sample = |v: &ViewAddr, y: usize, x: usize| {
        let (fy, fx) = (v.fy as f64, v.fx as f64);
        let at =
            |dy: usize, dx: usize| obj[(v.oy as usize + y + dy) * nxo + v.ox as usize + x + dx];
        [
            ((1.0 - fy) * (1.0 - fx), 0, 0),
            ((1.0 - fy) * fx, 0, 1),
            (fy * (1.0 - fx), 1, 0),
            (fy * fx, 1, 1),
        ]
        .iter()
        .filter(|(w, ..)| *w != 0.0)
        .fold((0.0, 0.0), |(re, im), &(w, dy, dx)| {
            let o = at(dy, dx);
            (re + w * o.re as f64, im + w * o.im as f64)
        })
    };

    let table = StackTable::new(&stream, &dims, &[1..3, 0..1]).unwrap();
    let mut split = vec![C32::default(); nframe * nprobe * n];
    for (s, f0, nf) in [(0, 1usize, 2usize), (1, 0, 1)] {
        let w = &whole_w[f0 * nprobe * n..(f0 + nf) * nprobe * n];
        let mut want: Vec<(f64, f64)> = fill.iter().map(|v| (v.re as f64, v.im as f64)).collect();
        for kl in 0..nf * nprobe {
            let v = &addr[f0 * nprobe + kl];
            let plane = kl / nprobe * nprobe + v.probe as usize;
            for j in 0..n {
                let (y, x) = ((j / nx + ny / 2) % ny, (j % nx + nx / 2) % nx);
                let o = sample(v, y, x);
                let a = w[plane * n + j];
                let (ar, ai) = (a.re as f64, a.im as f64);
                want[kl * n + j] = (o.0 * ar - o.1 * ai, o.0 * ai + o.1 * ar);
            }
        }
        let pad = vec![C32::default(); (2 - nf) * nprobe * n];
        let dw = up(&stream, &[w, &pad].concat());
        let mut psi = up(&stream, &fill);
        k.obj_wave_to_psi(&ctx, &views, table.at(s), &dobj, &dw, &mut psi)
            .unwrap();
        let got = psi.to_host_vec(&stream).unwrap();
        let scale = want.iter().map(|w| w.0.hypot(w.1)).fold(0.0, f64::max);
        let err = max_rel(&got, &want, scale);
        assert!(err < 1e-6, "entry {s}: max err / max |psi| {err:e}");
        assert_eq!(got[nf * nprobe * n..], fill[nf * nprobe * n..], "entry {s}");
        split[f0 * nprobe * n..(f0 + nf) * nprobe * n].copy_from_slice(&got[..nf * nprobe * n]);
    }

    let whole = 0..nframe as u32;
    let one = StackTable::new(&stream, &dims, std::slice::from_ref(&whole)).unwrap();
    let mut psi = DeviceBuffer::<C32>::zeroed(&stream, nframe * nprobe * n).unwrap();
    k.obj_wave_to_psi(
        &ctx,
        &views,
        one.at(0),
        &dobj,
        &up(&stream, &whole_w),
        &mut psi,
    )
    .unwrap();
    assert_eq!(psi.to_host_vec(&stream).unwrap(), split);

    // `w` one plane short.
    let short = DeviceBuffer::<C32>::zeroed(&stream, (nframe * nprobe - 1) * n).unwrap();
    assert!(
        k.obj_wave_to_psi(&ctx, &views, one.at(0), &dobj, &short, &mut psi)
            .is_err()
    );
}

/// Bit for bit `x + s * num` with the product and the sum rounded separately, and the sums
/// left zero.
#[test]
fn step_adds_and_consumes_sums() {
    let (stream, ctx, k) = setup();
    let n = 1000usize;
    let (x, num) = (noise(n, 7, 2.0), noise(n, 11, 3.0));
    let den: Vec<f32> = noise(n, 13, 1.0).iter().map(|c| c.re).collect();
    let s = 0.37f32;
    let want: Vec<C32> = x
        .iter()
        .zip(&num)
        .map(|(a, d)| C32 {
            re: a.re + d.re * s,
            im: a.im + d.im * s,
        })
        .collect();
    let mut dx = up(&stream, &x);
    let mut sums = UpdateSums::new(up(&stream, &num), up(&stream, &den), &stream).unwrap();
    k.step(&ctx, &mut dx, &mut sums, n as u32, s).unwrap();
    assert_eq!(dx.to_host_vec(&stream).unwrap(), want);
    assert!(
        sums.num
            .to_host_vec(&stream)
            .unwrap()
            .iter()
            .all(|c| *c == C32::default())
    );
    assert!(
        sums.den
            .to_host_vec(&stream)
            .unwrap()
            .iter()
            .all(|d| *d == 0.0)
    );
    assert!(k.step(&ctx, &mut dx, &mut sums, n as u32 + 1, s).is_err());
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
    let groups = PowerMax {
        ngroup: 2,
        nsum: 1,
        n: 4,
    };
    let mut chunks = DeviceBuffer::<f32>::zeroed(&stream, groups.chunks()).unwrap();
    let mut short = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
    let mut two = DeviceBuffer::<f32>::zeroed(&stream, 2).unwrap();
    // One max per group, one chunk max per group.
    assert!(
        k.power_max(&ctx, &w, groups, &mut chunks, &mut short)
            .is_err()
    );
    assert!(k.power_max(&ctx, &w, groups, &mut short, &mut two).is_err());
    k.power_max(&ctx, &w, groups, &mut chunks, &mut two)
        .unwrap();
}
