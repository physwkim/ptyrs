//! The propagator stages against the CPU's `f64` [`Propagator`], on the fused passes'
//! smallest side: around a host Fourier transform an exit stage gives the propagator's
//! intensities and, back through its adjoint, `Uᴴ·w·U` for real detector weights; the
//! in-object propagation gives angular spectrum and its adjoint. The slice launches give a
//! multislice object's products, their adjoint, the displaced plane phase of the CPU's
//! `Optics::moved_quad` and the row transforms. Angular spectrum with an evanescent band
//! carries the band from the forward stage into its adjoint, which then gives the CPU's
//! `Uᴴ w U - E`. Frames of two modes in stacks of 3 and 2, the shorter leaving the last
//! frame's planes, and their carry, as they were.

use std::f64::consts::PI;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs::base::C64;
use ptyrs::engine::{Optics, PlaneStage, Propagator, Wavefield, shift};
use ptyrs::gpu::{
    BandCarry, BandedTransfer, C32, ExitStage, FrameWaves, Kernels, OpticsKernels, SlicePhase,
    SliceSource, StackTable, ViewAddr, ViewDims, Views,
};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const NMODE: usize = 2;
/// Planes of the longest stack.
const NPLANE: usize = 3 * NMODE;

fn setup() -> (Arc<CudaStream>, ExecutionContext, OpticsKernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let optics = OpticsKernels::load(&kernels).expect("bind the stage kernels");
    let stream = ctx.default_stream();
    (stream.clone(), ExecutionContext::new(stream), optics)
}

fn dims(shape: [usize; 2]) -> ViewDims {
    ViewDims {
        nframe: 5,
        nmode: NMODE as u32,
        nprobe: NMODE as u32,
        nobj: 1,
        frame: shape.map(|n| n as u32),
        obj: shape.map(|n| n as u32 + 32),
        quad_phase: 0.0,
    }
}

/// Deterministic values in `[-1, 1)` per part.
fn noise(len: usize, seed: u64) -> Vec<C64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    (0..len).map(|_| C64::new(next(), next())).collect()
}

/// The unnormalized DFT of every row (`rows`) or column of `[ny, nx]` planes, forward
/// `exp(-2πi jk/n)`.
fn dft(planes: &[C64], [ny, nx]: [usize; 2], rows: bool, inverse: bool) -> Vec<C64> {
    let (n, stride, lines) = if rows { (nx, 1, ny) } else { (ny, nx, nx) };
    let sign = if inverse { 2.0 } else { -2.0 };
    let w: Vec<C64> = (0..n)
        .map(|k| C64::from_polar(1.0, sign * PI * k as f64 / n as f64))
        .collect();
    let mut out = vec![C64::default(); planes.len()];
    for (p, plane) in planes.chunks_exact(ny * nx).enumerate() {
        for l in 0..lines {
            let base = p * ny * nx + if rows { l * nx } else { l };
            for k in 0..n {
                out[base + k * stride] = (0..n)
                    .map(|j| plane[base - p * ny * nx + j * stride] * w[j * k % n])
                    .sum();
            }
        }
    }
    out
}

fn c32(v: &[C64]) -> Vec<C32> {
    v.iter()
        .map(|c| C32 {
            re: c.re as f32,
            im: c.im as f32,
        })
        .collect()
}

fn up(stream: &CudaStream, v: &[C64]) -> DeviceBuffer<C32> {
    DeviceBuffer::from_host(stream, &c32(v)).unwrap()
}

fn down(stream: &CudaStream, b: &DeviceBuffer<C32>) -> Vec<C64> {
    b.to_host_vec(stream)
        .unwrap()
        .iter()
        .map(|c| C64::new(c.re as f64, c.im as f64))
        .collect()
}

/// `max |a − b| / max |b|`.
fn max_rel(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let d = worst_of(a.iter().zip(b).map(|(a, b)| (a - b).norm()));
    d / b.iter().map(|b| b.norm()).fold(0.0, f64::max)
}

fn scaled(v: &[C64], s: f64) -> Vec<C64> {
    v.iter().map(|c| c * s).collect()
}

/// One propagator per kind and distance sign, and angular spectrum on a frame wider than
/// tall.
fn propagators() -> Vec<(&'static str, Propagator)> {
    let w = |shape, z| Wavefield::new(shape, 1e-10, [50e-9, 50e-9], z).unwrap();
    vec![
        (
            "far field",
            Propagator::free_space(&w([64, 64], f64::INFINITY)),
        ),
        (
            "angular spectrum",
            Propagator::free_space(&w([64, 64], 1e-4)),
        ),
        (
            "angular spectrum back",
            Propagator::free_space(&w([64, 64], -2e-4)),
        ),
        (
            "angular spectrum 64x128",
            Propagator::free_space(&w([64, 128], 1e-4)),
        ),
        ("Fresnel", Propagator::free_space(&w([64, 64], 5e-3))),
        ("Fresnel back", Propagator::free_space(&w([64, 64], -8e-3))),
    ]
}

#[test]
fn exit_stages_give_the_propagators_intensities_and_gram() {
    let (stream, ctx, k) = setup();
    let kinds: Vec<_> = propagators()
        .iter()
        .map(|(_, p)| match p {
            Propagator::Fourier(_) => 'F',
            Propagator::AngularSpectrum(_) => 'A',
            Propagator::Fresnel(_) => 'T',
        })
        .collect();
    assert_eq!(kinds, ['F', 'A', 'A', 'A', 'T', 'T']);
    for (name, p) in &propagators() {
        let shape = p.shape();
        let [ny, nx] = shape;
        let n = ny * nx;
        let d = dims(shape);
        let table = StackTable::new(&stream, &d, &[0..3, 3..5]).unwrap();
        let form = p.fourier_form().unwrap();
        let stage = match form.stage {
            PlaneStage::None => ExitStage::None,
            PlaneStage::Reverse => ExitStage::Reverse,
            PlaneStage::Transfer(tf) => ExitStage::Transfer(up(&stream, tf)),
        };
        // The plane phase, shifted layout.
        let q: Vec<C64> = (0..n)
            .map(|j| {
                let i = shift(j, ny, nx);
                let (x, y) = (
                    (i % nx) as f64 - (nx / 2) as f64,
                    (i / nx) as f64 - (ny / 2) as f64,
                );
                C64::from_polar(1.0, form.quad * (x * x + y * y))
            })
            .collect();
        let x = noise(NPLANE * n, 3);
        let rows = dft(&x, shape, true, false);

        // CPU: U (conj(q) x), and q Uᴴ w U (conj(q) x) for weights in 1/2..3/2.
        let weight: Vec<f64> = noise(n, 5).iter().map(|c| 1.0 + 0.5 * c.re).collect();
        let mut det: Vec<C64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v * q[i % n].conj())
            .collect();
        p.forward(&mut det);
        let mut gram: Vec<C64> = det
            .iter()
            .enumerate()
            .map(|(i, v)| v * weight[i % n])
            .collect();
        p.adjoint(&mut gram);
        gram.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v *= q[i % n]);

        for (s, live) in [(0, NPLANE), (1, 2 * NMODE)] {
            let mut work = up(&stream, &rows);
            k.exit(&ctx, &d, table.at(s), &stage, &mut work, false)
                .unwrap();
            let staged = down(&stream, &work);
            // The planes past the stack are as uploaded, f32 of `rows`.
            assert_eq!(
                staged[live * n..],
                down(&stream, &up(&stream, &rows))[live * n..],
                "{name}"
            );
            let far = scaled(&dft(&staged, shape, false, false), 1.0 / (n as f64).sqrt());
            let power = |v: &[C64]| -> Vec<C64> {
                v[..live * n]
                    .iter()
                    .map(|c| C64::new(c.norm_sqr(), 0.0))
                    .collect()
            };
            let err = max_rel(&power(&far), &power(&det));
            assert!(err < 2e-5, "{name}, stack {s}: intensity {err:.3e}");

            // Row transforms of Fᴴ w F V x, then the adjoint stage and the rows' inverse.
            let back: Vec<C64> = far
                .iter()
                .enumerate()
                .map(|(i, v)| v * weight[i % n])
                .collect();
            let back = scaled(
                &dft(&back, shape, false, true),
                (nx as f64 / ny as f64).sqrt(),
            );
            let mut work = up(&stream, &back);
            k.exit(&ctx, &d, table.at(s), &stage, &mut work, true)
                .unwrap();
            let got = scaled(
                &dft(&down(&stream, &work), shape, true, true),
                1.0 / nx as f64,
            );
            let err = max_rel(&got[..live * n], &gram[..live * n]);
            assert!(err < 2e-5, "{name}, stack {s}: gram {err:.3e}");
        }
    }
}

#[test]
fn banded_stage_carries_the_band_into_the_adjoint() {
    let (stream, ctx, k) = setup();
    for (shape, z) in [
        ([64, 64], 1.5e-6),
        ([64, 128], -1.5e-6),
        ([128, 64], 1.5e-6),
    ] {
        let [ny, nx] = shape;
        let n = ny * nx;
        let d = dims(shape);
        let table = StackTable::new(&stream, &d, &[0..3, 3..5]).unwrap();
        let w = Wavefield::new(shape, 5e-7, [3e-7, 3e-7], z).unwrap();
        let p = Propagator::angular_spectrum(&w).unwrap();
        assert!(!p.adjoint_inverts(), "{shape:?}: no band");
        let tf = p.transfer().unwrap();
        let banded = BandedTransfer::new(&stream, &c32(tf)).unwrap();
        let plain = ExitStage::Transfer(up(&stream, tf));
        let x = noise(NPLANE * n, 11);
        let rows = dft(&x, shape, true, false);

        // CPU: U x, and Uᴴ w U x - E x with `E = 1 - Uᴴ U` the projection onto the band.
        let weight: Vec<f64> = noise(n, 13).iter().map(|c| 1.0 + 0.5 * c.re).collect();
        let mut det = x.clone();
        p.forward(&mut det);
        let mut gram: Vec<C64> = det
            .iter()
            .enumerate()
            .map(|(i, v)| v * weight[i % n])
            .collect();
        p.adjoint(&mut gram);
        let mut kept = det.clone();
        p.adjoint(&mut kept);
        let want: Vec<C64> = (gram.iter().zip(&x).zip(&kept))
            .map(|((g, x), k)| g - (x - k))
            .collect();
        let band = max_rel(&gram, &want);
        assert!(band > 1e-2, "{shape:?}: band {band:.3e} too weak to test");

        let carry0 = noise(banded.carry_len(&d, table.nf_max()), 17);
        for (s, live) in [(0, NPLANE), (1, 2 * NMODE)] {
            let mut carry = up(&stream, &carry0);
            let mut work = up(&stream, &rows);
            let save = BandCarry::Save(&mut carry);
            k.exit_banded(&ctx, &d, table.at(s), &banded, &mut work, save)
                .unwrap();
            let staged = down(&stream, &work);
            let mut alone = up(&stream, &rows);
            k.exit(&ctx, &d, table.at(s), &plain, &mut alone, false)
                .unwrap();
            assert_eq!(staged, down(&stream, &alone), "{shape:?}, stack {s}");
            let past = banded.carry_len(&d, (live / NMODE) as u32);
            assert_eq!(
                down(&stream, &carry)[past..],
                down(&stream, &up(&stream, &carry0))[past..],
                "{shape:?}, stack {s}: carry past the stack"
            );
            let far = scaled(&dft(&staged, shape, false, false), 1.0 / (n as f64).sqrt());
            let power = |v: &[C64]| -> Vec<C64> {
                v[..live * n]
                    .iter()
                    .map(|c| C64::new(c.norm_sqr(), 0.0))
                    .collect()
            };
            let err = max_rel(&power(&far), &power(&det));
            assert!(err < 2e-5, "{shape:?}, stack {s}: intensity {err:.3e}");

            let back: Vec<C64> = far
                .iter()
                .enumerate()
                .map(|(i, v)| v * weight[i % n])
                .collect();
            let back = scaled(
                &dft(&back, shape, false, true),
                (nx as f64 / ny as f64).sqrt(),
            );
            let mut work = up(&stream, &back);
            let restore = BandCarry::Restore(&carry);
            k.exit_banded(&ctx, &d, table.at(s), &banded, &mut work, restore)
                .unwrap();
            let out = down(&stream, &work);
            assert_eq!(
                out[live * n..],
                down(&stream, &up(&stream, &back))[live * n..],
                "{shape:?}, stack {s}: planes past the stack"
            );
            let got = scaled(&dft(&out, shape, true, true), 1.0 / nx as f64);
            let err = max_rel(&got[..live * n], &want[..live * n]);
            assert!(err < 2e-5, "{shape:?}, stack {s}: adjoint {err:.3e}");
        }
    }
}

#[test]
fn gap_is_angular_spectrum_and_its_adjoint() {
    let (stream, ctx, k) = setup();
    for shape in [[64, 64], [64, 128]] {
        let n = shape[0] * shape[1];
        let d = dims(shape);
        let table = StackTable::new(&stream, &d, &[0..3, 3..5]).unwrap();
        let w = Wavefield::new(shape, 1e-10, [50e-9, 50e-9], 3e-5).unwrap();
        let p = Propagator::angular_spectrum(&w).unwrap();
        let tf = up(&stream, p.transfer().unwrap());
        let x = noise(NPLANE * n, 7);
        for adjoint in [false, true] {
            let mut want = x.clone();
            if adjoint {
                p.adjoint(&mut want);
            } else {
                p.forward(&mut want);
            }
            for (s, live) in [(0, NPLANE), (1, 2 * NMODE)] {
                let mut waves = up(&stream, &x);
                k.gap(&ctx, &d, table.at(s), &tf, &mut waves, adjoint)
                    .unwrap();
                let got = down(&stream, &waves);
                assert_eq!(got[live * n..], down(&stream, &up(&stream, &x))[live * n..]);
                let err = max_rel(&got[..live * n], &want[..live * n]);
                assert!(
                    err < 2e-5,
                    "{shape:?}, adjoint {adjoint}, stack {s}: {err:.3e}"
                );
            }
        }
    }
}

/// Views of every frame and mode on the object of [`dims`], windows at whole and fractional
/// pixels.
fn views() -> Vec<ViewAddr> {
    (0..5u32)
        .flat_map(|f| {
            (0..NMODE as u32).map(move |m| ViewAddr {
                oy: 3 + 5 * f,
                ox: 27 - 4 * f,
                fy: [0.0, 0.25, 0.5, 0.0, 0.875][f as usize],
                fx: [0.0, 0.625, 0.0, 0.375, 0.125][f as usize],
                probe: m,
                obj: 0,
            })
        })
        .collect()
}

/// The object's bilinear window of view `v` at natural pixel `j`.
fn window(obj: &[C64], v: &ViewAddr, j: usize, nx: usize, nxo: usize) -> C64 {
    let (fy, fx) = (v.fy as f64, v.fx as f64);
    let at = |dy: usize, dx: usize| {
        obj[(v.oy as usize + j / nx + dy) * nxo + v.ox as usize + j % nx + dx]
    };
    let corner = |w: f64, dy, dx| {
        if w == 0.0 {
            C64::default()
        } else {
            at(dy, dx) * w
        }
    };
    corner((1.0 - fy) * (1.0 - fx), 0, 0)
        + corner((1.0 - fy) * fx, 0, 1)
        + corner(fy * (1.0 - fx), 1, 0)
        + corner(fy * fx, 1, 1)
}

#[test]
fn slice_products_are_the_windows_times_probe_or_waves() {
    let (stream, ctx, k) = setup();
    for shape in [[64, 64], [64, 128]] {
        let [ny, nx] = shape;
        let n = ny * nx;
        let d = dims(shape);
        let nxo = d.obj[1] as usize;
        let table = StackTable::new(&stream, &d, &[0..3, 3..5]).unwrap();
        let addr = views();
        let on_device = Views::new(&stream, d, &addr).unwrap();
        let obj = noise(d.obj[0] as usize * nxo, 11);
        let probe = noise(NMODE * n, 13);
        let waves = noise(NPLANE * n, 17);
        let before = noise(NPLANE * n, 19);
        let (dev_obj, dev_probe, dev_waves) =
            (up(&stream, &obj), up(&stream, &probe), up(&stream, &waves));
        for (s, f0, live) in [(0, 0, NPLANE), (1, 3, 2 * NMODE)] {
            // Plane `kl`'s value at natural pixel `j`, placed at its stored pixel.
            let want = |f: &dyn Fn(usize, usize, C64) -> C64| {
                let mut out = before.clone();
                for kl in 0..live {
                    let v = &addr[f0 * NMODE + kl];
                    for j in 0..n {
                        out[kl * n + shift(j, ny, nx)] = f(kl, j, window(&obj, v, j, nx, nxo));
                    }
                }
                out
            };
            let stored = |kl: usize, j: usize| waves[kl * n + shift(j, ny, nx)];
            let cases: [(&str, Vec<C64>, Option<SliceSource<'_>>); 3] = [
                (
                    "probe",
                    want(&|kl, j, o| o * probe[kl % NMODE * n + j]),
                    Some(SliceSource::Probe(&dev_probe)),
                ),
                (
                    "waves",
                    want(&|kl, j, o| o * stored(kl, j)),
                    Some(SliceSource::Waves(FrameWaves::Views(&dev_waves))),
                ),
                ("adjoint", want(&|kl, j, o| o.conj() * stored(kl, j)), None),
            ];
            for (name, want, src) in cases {
                let mut dst = up(&stream, &before);
                match src {
                    Some(src) => {
                        k.slice_mul(&ctx, &on_device, table.at(s), &dev_obj, src, &mut dst)
                    }
                    None => k.slice_adjoint(
                        &ctx,
                        &on_device,
                        table.at(s),
                        &dev_obj,
                        &dev_waves,
                        &mut dst,
                    ),
                }
                .unwrap();
                let got = down(&stream, &dst);
                assert_eq!(
                    got[live * n..],
                    down(&stream, &up(&stream, &before))[live * n..],
                    "{name}"
                );
                let err = max_rel(&got[..live * n], &want[..live * n]);
                assert!(err < 2e-6, "{shape:?}, {name}, stack {s}: {err:.3e}");
            }
        }
    }
}

#[test]
fn slice_phase_is_the_cpu_moved_quad_but_for_a_frame_constant() {
    let (stream, ctx, k) = setup();
    let shape = [64, 64];
    let [ny, nx] = shape;
    let n = ny * nx;
    let d = dims(shape);
    let table = StackTable::new(&stream, &d, &[0..3, 3..5]).unwrap();
    let w = |z| Wavefield::new(shape, 1e-10, [50e-9, 50e-9], z).unwrap();
    let offsets: [[f64; 2]; 5] = [
        [0.0, 0.0],
        [1.75, -0.5],
        [-2.25, 3.0],
        [0.375, 0.625],
        [-1.0, -1.5],
    ];
    let flat: Vec<f32> = offsets.iter().flatten().map(|&v| v as f32).collect();
    let dev_offsets = DeviceBuffer::from_host(&stream, &flat).unwrap();
    let x = noise(NPLANE * n, 23);
    for z in [5e-3, -8e-3] {
        let optics = Optics::free_space(&w(z)).with_slices(&[w(2e-5)]).unwrap();
        let quad = optics.fourier_form().unwrap().quad;
        assert_ne!(quad, 0.0);
        for moved in [false, true] {
            for (s, f0, live) in [(0, 0, NPLANE), (1, 3, 2 * NMODE)] {
                // q(r − d) of each frame, less its constant exp(i quad |d|²).
                let q: Vec<Vec<C64>> = (0..live / NMODE)
                    .map(|fl| {
                        if !moved {
                            return optics.quad().to_vec();
                        }
                        let [dy, dx] = offsets[f0 + fl];
                        let c = C64::from_polar(1.0, -quad * (dy * dy + dx * dx));
                        let m = optics.moved_quad([dy, dx]).unwrap();
                        m.quad.iter().map(|q| q * c).collect()
                    })
                    .collect();
                for adjoint in [false, true] {
                    let mut want = x.clone();
                    for kl in 0..live {
                        for j in 0..n {
                            let q = q[kl / NMODE][j];
                            want[kl * n + shift(j, ny, nx)] *= if adjoint { q.conj() } else { q };
                        }
                    }
                    let phase = SlicePhase {
                        quad: quad as f32,
                        offsets: moved.then_some(&dev_offsets),
                    };
                    let mut waves = up(&stream, &x);
                    k.slice_phase(&ctx, &d, table.at(s), phase, &mut waves, adjoint)
                        .unwrap();
                    let got = down(&stream, &waves);
                    assert_eq!(got[live * n..], down(&stream, &up(&stream, &x))[live * n..]);
                    let err = max_rel(&got[..live * n], &want[..live * n]);
                    assert!(
                        err < 5e-5,
                        "z {z}, moved {moved}, adjoint {adjoint}, stack {s}: {err:.3e}"
                    );
                }
            }
        }
    }
}

#[test]
fn rows_and_pair_are_the_row_transforms_and_the_conjugate_product() {
    let (stream, ctx, k) = setup();
    for shape in [[64, 64], [64, 128]] {
        let n = shape[0] * shape[1];
        let d = dims(shape);
        let table = StackTable::new(&stream, &d, &[0..3, 3..5]).unwrap();
        let (a, b) = (noise(NPLANE * n, 29), noise(NPLANE * n, 31));
        let kept = down(&stream, &up(&stream, &a));
        for (s, live) in [(0, NPLANE), (1, 2 * NMODE)] {
            for inverse in [false, true] {
                let mut waves = up(&stream, &a);
                k.rows(&ctx, &d, table.at(s), &mut waves, inverse).unwrap();
                let got = down(&stream, &waves);
                assert_eq!(got[live * n..], kept[live * n..]);
                let want = dft(&a, shape, true, inverse);
                let err = max_rel(&got[..live * n], &want[..live * n]);
                assert!(
                    err < 2e-6,
                    "{shape:?}, inverse {inverse}, stack {s}: {err:.3e}"
                );
            }
            let mut dst = up(&stream, &a);
            let pair = [&up(&stream, &a), &up(&stream, &b)];
            k.slice_pair(&ctx, &d, table.at(s), pair, &mut dst).unwrap();
            let got = down(&stream, &dst);
            assert_eq!(got[live * n..], kept[live * n..]);
            let want: Vec<C64> = a.iter().zip(&b).map(|(a, b)| a.conj() * b).collect();
            let err = max_rel(&got[..live * n], &want[..live * n]);
            assert!(err < 2e-6, "{shape:?}, pair, stack {s}: {err:.3e}");
        }
    }
}
