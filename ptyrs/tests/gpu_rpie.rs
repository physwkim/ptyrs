//! The rPIE kernels against f64 references of their documented formulas.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs::gpu::{
    BhScan, C32, EpochOrder, Kernels, MASKED, POWER_CHUNK, PowerMax, RpieKernels, StackTable,
    ViewAddr, ViewDims, Views, Weight,
};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

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
    worst_of(got.iter().zip(want).map(|(g, w)| {
        let e = ((g.re as f64 - w.0).powi(2) + (g.im as f64 - w.1).powi(2)).sqrt();
        e / (w.0.hypot(w.1)).max(floor)
    }))
}

#[test]
fn amplitude_matches_ptychi_formula() {
    let cuda = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&cuda).expect("load ptyrs-gpu kernels");
    let k = RpieKernels::load(&kernels).expect("bind the rPIE kernels");
    let stream = cuda.default_stream();
    let ctx = ExecutionContext::new(stream.clone());
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
    // Entry 0 fills the `nf_max = 2` frames of `psi`; entry 1 only its first. Their frames
    // are epoch positions, which hold the scan frames `ids`.
    let table = StackTable::new(&stream, &dims, &[1..3, 0..1]).unwrap();
    let ids = [2u32, 0, 1];
    let addr: Vec<ViewAddr> = (0..nframe * nmode)
        .map(|i| ViewAddr {
            probe: (i % nmode) as u32,
            ..ViewAddr::default()
        })
        .collect();
    let views = Views::new(&stream, dims, &addr).unwrap();
    let mut order = EpochOrder::new(&kernels, &stream, &views).unwrap();
    order.set(&ctx, &kernels, &views, &ids).unwrap();
    let scan = BhScan {
        order: &order,
        counts: &up(&stream, &counts),
    };
    let psi = noise(2 * nmode * n, 7, 2.0);
    for (s, f0, nf) in [(0, 1usize, 2usize), (1, 0, 1)] {
        let mut want: Vec<(f64, f64)> = psi.iter().map(|v| (v.re as f64, v.im as f64)).collect();
        for fl in 0..nf {
            for d in 0..n {
                let at = ((d / nx + ny / 2) % ny) * nx + (d % nx + nx / 2) % nx;
                let obs = counts[ids[f0 + fl] as usize * n + d];
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
        k.amplitude(&ctx, scan, table.at(s), &mut p).unwrap();
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

    // The same maxes of the squared moduli as plain floats; `out` starts non-zero.
    let w2: Vec<f32> = w.iter().map(|c| norm2(*c) as f32).collect();
    let dw2 = up(&stream, &w2);
    for (groups, want) in [
        (all_groups, all),
        (per_groups, per.iter().copied().fold(0.0, f64::max)),
    ] {
        let mut out = up(&stream, &[1e30f32]);
        k.sum_max(&ctx, &dw2, groups, &mut out).unwrap();
        let got = out.to_host_vec(&stream).unwrap()[0] as f64;
        assert!(
            (got - want).abs() < 1e-6 * want,
            "sum_max {groups:?}: {got} vs {want}"
        );
    }

    let alpha = 0.3;
    let spec = Weight {
        nk: nk as u32,
        n: n as u32,
        alpha: alpha as f32,
    };
    let weighted = |m: &dyn Fn(usize) -> f64| -> Vec<(f64, f64)> {
        (0..nk * n)
            .map(|i| {
                let d = (1.0 - alpha) * norm2(w[i]) + alpha * m(i);
                (x[i].re as f64 / d, x[i].im as f64 / d)
            })
            .collect()
    };
    let mut got = up(&stream, &x);
    k.weight_inplace(&ctx, &mut got, &dw, &m_per, spec).unwrap();
    let err = max_rel(
        &got.to_host_vec(&stream).unwrap(),
        &weighted(&|i| per[i / n]),
        0.0,
    );
    assert!(err < 1e-5, "weight_inplace: {err:e}");
    let mut out = DeviceBuffer::<C32>::zeroed(&stream, nk * n).unwrap();
    let inputs = (&up(&stream, &x), &dw2, &m_all);
    k.overlap_weight(&ctx, inputs, &mut out, spec).unwrap();
    let err = max_rel(&out.to_host_vec(&stream).unwrap(), &weighted(&|_| all), 0.0);
    assert!(err < 1e-5, "overlap_weight: {err:e}");
}

/// `|probe[mode]|^2` repeated over every plane of the output.
#[test]
fn mode_power_matches_f64() {
    let (stream, ctx, k) = setup();
    let (n, nprobe, nout) = (64usize, 2usize, 3usize);
    let probe = noise(nprobe * n, 13, 1.5);
    let dp = up(&stream, &probe);
    let mut out = DeviceBuffer::<C32>::zeroed(&stream, nout * n).unwrap();
    for mode in 0..nprobe {
        k.mode_power(&ctx, &dp, mode as u32, n as u32, &mut out)
            .unwrap();
        let want: Vec<(f64, f64)> = (0..nout * n)
            .map(|t| (norm2(probe[mode * n + t % n]), 0.0))
            .collect();
        let err = max_rel(&out.to_host_vec(&stream).unwrap(), &want, 0.0);
        assert!(err < 1e-6, "mode {mode}: {err:e}");
    }
    // A mode past the probe, and an output not a whole number of planes.
    assert!(k.mode_power(&ctx, &dp, 2, n as u32, &mut out).is_err());
    let mut ragged = DeviceBuffer::<C32>::zeroed(&stream, nout * n + 1).unwrap();
    assert!(k.mode_power(&ctx, &dp, 0, n as u32, &mut ragged).is_err());
}

/// Three frames of two probe modes on one object mode at fractional corners, as
/// `obj_wave_to_psi_matches_f64` places them.
fn overlap_views(stream: &CudaStream) -> (ViewDims, Vec<ViewAddr>, Views) {
    let dims = ViewDims {
        nframe: 3,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [8, 8],
        obj: [16, 16],
        quad_phase: 0.0,
    };
    let corners = [(3, 5, 0.25f32, 0.5f32), (0, 0, 0.0, 0.0), (7, 2, 0.75, 0.0)];
    let addr: Vec<ViewAddr> = corners
        .iter()
        .flat_map(|&(oy, ox, fy, fx)| {
            (0..2).map(move |p| ViewAddr {
                oy,
                ox,
                fy,
                fx,
                probe: p,
                obj: 0,
            })
        })
        .collect();
    let views = Views::new(stream, dims, &addr).unwrap();
    (dims, addr, views)
}

/// The bilinear taps `(weight, dy, dx)` a view reads its object pixel `(y, x)` from.
fn taps(v: &ViewAddr) -> impl Iterator<Item = (f64, usize, usize)> {
    let (fy, fx) = (v.fy as f64, v.fx as f64);
    [
        ((1.0 - fy) * (1.0 - fx), 0, 0),
        ((1.0 - fy) * fx, 0, 1),
        (fy * (1.0 - fx), 1, 0),
        (fy * fx, 1, 1),
    ]
    .into_iter()
    .filter(|(w, ..)| *w != 0.0)
}

/// Each mode's `|P|^2` of the shared probe and of per-frame probes, placed over the views of
/// each stack of a split as the object step places them, against the f64 scatter; the pixels
/// no view of the stack reads keep their value or are zeroed.
#[test]
fn overlap_matches_f64() {
    let (stream, ctx, k) = setup();
    let (dims, addr, views) = overlap_views(&stream);
    let ([ny, nx], [nyo, nxo]) = (dims.frame.map(|v| v as usize), dims.obj.map(|v| v as usize));
    let (n, nprobe) = (ny * nx, dims.nprobe as usize);
    let natural = |j: usize| ((j / nx + ny / 2) % ny, (j % nx + nx / 2) % nx);
    let probe = noise(nprobe * n, 5, 1.5);
    let dprobe = up(&stream, &probe);
    let fill = noise(nyo * nxo, 9, 1.0);
    let table = StackTable::new(&stream, &dims, &[1..3, 0..1]).unwrap();
    let whole_w = noise(dims.nframe as usize * nprobe * n, 7, 1.5);
    for (s, f0, nf) in [(0, 1usize, 2usize), (1, 0, 1)] {
        let w = &whole_w[f0 * nprobe * n..(f0 + nf) * nprobe * n];
        let pad = vec![C32::default(); (2 - nf) * nprobe * n];
        let dw = up(&stream, &[w, &pad].concat());
        let stack_views = &addr[f0 * nprobe..(f0 + nf) * nprobe];
        let mut read = vec![false; nyo * nxo];
        for v in stack_views {
            for j in 0..n {
                let (y, x) = natural(j);
                for (_, dy, dx) in taps(v) {
                    read[(v.oy as usize + y + dy) * nxo + v.ox as usize + x + dx] = true;
                }
            }
        }
        for mode in 0..nprobe {
            // `|p|^2` of view `kl`'s probe at its exit-wave pixel `j`.
            let shared = |_: usize, j: usize| {
                let (y, x) = natural(j);
                norm2(probe[mode * n + y * nx + x])
            };
            let own = |kl: usize, j: usize| norm2(w[(kl / nprobe * nprobe + mode) * n + j]);
            for (name, p2) in [
                ("shared", &shared as &dyn Fn(usize, usize) -> f64),
                ("own", &own),
            ] {
                let mut want = vec![0.0f64; nyo * nxo];
                for (kl, v) in stack_views.iter().enumerate() {
                    if v.probe as usize != mode {
                        continue;
                    }
                    for j in 0..n {
                        let (y, x) = natural(j);
                        for (t, dy, dx) in taps(v) {
                            let at = (v.oy as usize + y + dy) * nxo + v.ox as usize + x + dx;
                            want[at] += t * p2(kl, j);
                        }
                    }
                }
                let mut d = up(&stream, &fill);
                if name == "shared" {
                    k.overlap(&ctx, &views, table.at(s), &dprobe, mode as u32, &mut d)
                } else {
                    k.overlap_moved(&ctx, &views, table.at(s), &dw, mode as u32, &mut d)
                }
                .unwrap();
                let got = d.to_host_vec(&stream).unwrap();
                let scale = want.iter().copied().fold(0.0, f64::max);
                for (i, g) in got.iter().enumerate() {
                    if read[i] {
                        let e = ((g.re as f64 - want[i]).powi(2) + (g.im as f64).powi(2)).sqrt();
                        assert!(
                            e < 1e-6 * scale,
                            "{name} entry {s} mode {mode} pixel {i}: {g:?} vs {}",
                            want[i]
                        );
                    } else {
                        assert!(
                            *g == fill[i] || *g == C32::default(),
                            "{name} entry {s} pixel {i}"
                        );
                    }
                }
            }
        }
    }
    // A mode past the probe, and per-frame probes one plane short.
    let mut d = up(&stream, &fill);
    assert!(
        k.overlap(&ctx, &views, table.at(0), &dprobe, 2, &mut d)
            .is_err()
    );
    let short = DeviceBuffer::<C32>::zeroed(&stream, (2 * nprobe - 1) * n).unwrap();
    assert!(
        k.overlap_moved(&ctx, &views, table.at(0), &short, 0, &mut d)
            .is_err()
    );
}

/// `max(0, Re O)` of an object with negative real parts, sampled at each view of one mode of
/// each stack of a split; the other entries are left.
#[test]
fn overlap_read_matches_f64() {
    let (stream, ctx, k) = setup();
    let (dims, addr, views) = overlap_views(&stream);
    let ([ny, nx], [nyo, nxo]) = (dims.frame.map(|v| v as usize), dims.obj.map(|v| v as usize));
    let (n, nprobe) = (ny * nx, dims.nprobe as usize);
    let obj = noise(nyo * nxo, 3, 1.0);
    let dobj = up(&stream, &obj);
    let fill: Vec<f32> = (0..2 * nprobe * n).map(|i| -1.0 - i as f32).collect();
    let table = StackTable::new(&stream, &dims, &[1..3, 0..1]).unwrap();
    for (s, f0, nf) in [(0, 1usize, 2usize), (1, 0, 1)] {
        for mode in 0..nprobe {
            let mut want: Vec<f64> = fill.iter().map(|&v| v as f64).collect();
            for kl in 0..nf * nprobe {
                let v = &addr[f0 * nprobe + kl];
                if v.probe as usize != mode {
                    continue;
                }
                for j in 0..n {
                    let (y, x) = ((j / nx + ny / 2) % ny, (j % nx + nx / 2) % nx);
                    let re: f64 = taps(v)
                        .map(|(t, dy, dx)| {
                            let at = (v.oy as usize + y + dy) * nxo + v.ox as usize + x + dx;
                            t * obj[at].re as f64
                        })
                        .sum();
                    want[kl * n + j] = re.max(0.0);
                }
            }
            let mut dens = up(&stream, &fill);
            k.overlap_read(&ctx, &views, table.at(s), &dobj, mode as u32, &mut dens)
                .unwrap();
            let got = dens.to_host_vec(&stream).unwrap();
            let err = worst_of(got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()));
            assert!(err < 1e-6, "entry {s} mode {mode}: max err {err:e}");
        }
    }
    let mut short = DeviceBuffer::<f32>::zeroed(&stream, 2 * nprobe * n - 1).unwrap();
    assert!(
        k.overlap_read(&ctx, &views, table.at(0), &dobj, 0, &mut short)
            .is_err()
    );
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

/// Bit for bit `x + s * num` with the product and the sum rounded separately, and `num`
/// left zero.
#[test]
fn step_adds_and_consumes_sums() {
    let (stream, ctx, k) = setup();
    let n = 1000usize;
    let (x, num) = (noise(n, 7, 2.0), noise(n, 11, 3.0));
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
    let mut num = up(&stream, &num);
    k.step(&ctx, &mut dx, &mut num, n as u32, s).unwrap();
    assert_eq!(dx.to_host_vec(&stream).unwrap(), want);
    assert!(
        num.to_host_vec(&stream)
            .unwrap()
            .iter()
            .all(|c| *c == C32::default())
    );
    assert!(k.step(&ctx, &mut dx, &mut num, n as u32 + 1, s).is_err());
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
        alpha: 0.1,
    };
    // One max per plane.
    assert!(k.weight_inplace(&ctx, &mut out, &w, &mx, spec).is_err());
    // As many floats in `dens` as values.
    let dens = DeviceBuffer::<f32>::zeroed(&stream, 7).unwrap();
    assert!(
        k.overlap_weight(&ctx, (&w, &dens, &mx), &mut out, spec)
            .is_err()
    );
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
