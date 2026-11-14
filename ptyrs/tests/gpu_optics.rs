//! The propagator stages against the CPU's `f64` [`Propagator`], on the fused passes'
//! smallest side: around a host Fourier transform an exit stage gives the propagator's
//! intensities and, back through its adjoint, `Uᴴ·w·U` for real detector weights; the
//! in-object propagation gives angular spectrum and its adjoint. Frames of two modes in
//! stacks of 3 and 2, the shorter leaving the last frame's planes as they were.

use std::f64::consts::PI;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs::base::C64;
use ptyrs::engine::{PlaneStage, Propagator, Wavefield, shift};
use ptyrs::gpu::{C32, ExitStage, Kernels, OpticsKernels, StackTable, ViewDims};

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

fn up(stream: &CudaStream, v: &[C64]) -> DeviceBuffer<C32> {
    let host: Vec<C32> = v
        .iter()
        .map(|c| C32 {
            re: c.re as f32,
            im: c.im as f32,
        })
        .collect();
    DeviceBuffer::from_host(stream, &host).unwrap()
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
