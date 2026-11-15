//! The position sums against an `f64` host reference, read through one batch's step:
//! [`PositionKernels::sums`] of the periodic slopes of each frame's window, and
//! [`PositionKernels::derivative_sums`] of given derivative windows against the stack's
//! waves or the unmoved probe under a quadratic exit phase. Frames of two modes in stacks
//! of 3 and 2, the shorter leaving its last frame's planes unread.

use std::f64::consts::PI;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs::base::C64;
use ptyrs::gpu::{
    C32, EpochOrder, FrameWaves, Kernels, PositionKernels, PositionRule, Positions, SliceSource,
    Solve, StackTable, ViewAddr, ViewDims, Views,
};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const NFRAME: usize = 5;
const NMODE: usize = 2;
const NF_MAX: usize = 3;
const STACKS: [std::ops::Range<u32>; 2] = [0..3, 3..5];
const SHAPE: [usize; 2] = [64, 64];
const N: usize = SHAPE[0] * SHAPE[1];
/// Large enough that only the spread of the batch's shifts clips them.
const RULE: PositionRule = PositionRule {
    step: 0.3,
    limit: 1e3,
};

fn setup() -> (Arc<CudaStream>, ExecutionContext, Kernels, PositionKernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let position = PositionKernels::load(&kernels).expect("bind the position kernels");
    let stream = ctx.default_stream();
    (
        stream.clone(),
        ExecutionContext::new(stream),
        kernels,
        position,
    )
}

fn dims(quad_phase: f32) -> ViewDims {
    ViewDims {
        nframe: NFRAME as u32,
        nmode: NMODE as u32,
        nprobe: NMODE as u32,
        nobj: 1,
        frame: SHAPE.map(|n| n as u32),
        obj: SHAPE.map(|n| n as u32 + 32),
        quad_phase,
    }
}

fn views() -> Vec<ViewAddr> {
    (0..NFRAME as u32)
        .flat_map(|f| {
            (0..NMODE as u32).map(move |m| ViewAddr {
                oy: 3 + 5 * f,
                ox: 27 - 4 * f,
                fy: 0.0,
                fx: 0.0,
                probe: m,
                obj: 0,
            })
        })
        .collect()
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

/// The exit wave's stored index of natural pixel `i`.
fn stored(i: usize) -> usize {
    let [ny, nx] = SHAPE;
    let (y, x) = (i / nx, i % nx);
    (y + ny / 2) % ny * nx + (x + nx / 2) % nx
}

/// `exp(i quad r²)`, `r` from the frame's centre.
fn quad(quad: f64, i: usize) -> C64 {
    let [ny, nx] = SHAPE;
    let (dy, dx) = (
        (i / nx) as f64 - (ny / 2) as f64,
        (i % nx) as f64 - (nx / 2) as f64,
    );
    C64::from_polar(1.0, quad * (dy * dy + dx * dx))
}

/// `fftfreq`'s bin `k` of `n`.
fn freq(k: usize, n: usize) -> f64 {
    let k = if k < n.div_ceil(2) {
        k as f64
    } else {
        k as f64 - n as f64
    };
    k / n as f64
}

/// The periodic Fourier derivative along `axis` of a natural-layout `[ny, nx]` patch.
fn slope(patch: &[C64], axis: usize) -> Vec<C64> {
    let [ny, nx] = SHAPE;
    let (n, stride, lines) = if axis == 0 { (ny, nx, nx) } else { (nx, 1, ny) };
    let mut out = vec![C64::default(); N];
    for l in 0..lines {
        let base = if axis == 0 { l } else { l * nx };
        let line: Vec<C64> = (0..n).map(|j| patch[base + j * stride]).collect();
        let spec: Vec<C64> = (0..n)
            .map(|k| {
                let d = C64::new(0.0, 2.0 * PI * freq(k, n));
                d * (0..n)
                    .map(|j| {
                        line[j] * C64::from_polar(1.0, -2.0 * PI * (j * k % n) as f64 / n as f64)
                    })
                    .sum::<C64>()
            })
            .collect();
        for j in 0..n {
            out[base + j * stride] = (0..n)
                .map(|k| spec[k] * C64::from_polar(1.0, 2.0 * PI * (j * k % n) as f64 / n as f64))
                .sum::<C64>()
                / n as f64;
        }
    }
    out
}

/// pty-chi's batch step over every frame from per-frame `[num, den]` sums, as
/// `position_step` takes it: offsets from zero.
fn step(sums: &[[[f64; 2]; 2]]) -> Vec<f64> {
    let nf = sums.len() as f64;
    let max = [0, 1].map(|ax| sums.iter().map(|s| s[1][ax]).fold(1e-6, f64::max));
    let dx: Vec<[f64; 2]> = sums
        .iter()
        .map(|[num, den]| [0, 1].map(|ax| RULE.step as f64 * num[ax] / (den[ax] + max[ax])))
        .collect();
    let mean = [0, 1].map(|ax| dx.iter().map(|d| d[ax]).sum::<f64>() / nf);
    let lim = [0, 1].map(|ax| {
        let mad = dx.iter().map(|d| (d[ax] - mean[ax]).abs()).sum::<f64>() / nf;
        (10.0 * mad).min(RULE.limit as f64)
    });
    dx.iter()
        .flat_map(|d| [0, 1].map(|ax| d[ax].clamp(-lim[ax], lim[ax])))
        .collect()
}

/// Frame `f`'s `[num, den]`: `Σ Re conj(g w) chi`, `Σ |g w|²` over natural pixels `i` per
/// axis, `g` its two derivatives, `w` and `chi` functions of `i`.
fn frame_sums(
    g: [&[C64]; 2],
    w: impl Fn(usize) -> C64,
    chi: impl Fn(usize) -> C64,
) -> [[f64; 2]; 2] {
    let mut s = [[0.0; 2]; 2];
    for (i, gy) in g[0].iter().enumerate() {
        for (ax, g) in [gy, &g[1][i]].into_iter().enumerate() {
            let u = g * w(i);
            s[0][ax] += (u.conj() * chi(i)).re;
            s[1][ax] += u.norm_sqr();
        }
    }
    s
}

/// What a stack's sums run against.
enum Source {
    /// The stack's mode-0 view's slopes of the object, against the probes `[nf_max *
    /// nprobe]` of [`FrameWaves::Probes`].
    Slopes,
    /// Given derivative windows against the stack's view waves.
    Waves,
    /// Given derivative windows against the unmoved probe.
    Probe,
}

/// The step from zero offsets of one batch over every frame, after each stack's sums of
/// `src`; the device's and the host's.
fn run(src: Source, quad_phase: f32) -> (Vec<f64>, Vec<f64>) {
    let (stream, ctx, kernels, k) = setup();
    let d = dims(quad_phase);
    let nxo = d.obj[1] as usize;
    let addr = views();
    let views = Views::new(&stream, d, &addr).unwrap();
    let table = StackTable::new(&stream, &d, &STACKS).unwrap();
    let order = EpochOrder::new(&kernels, &stream, &views).unwrap();
    let mut pos = Positions::zeroed(&stream, &d, NF_MAX as u32).unwrap();
    let obj = noise(d.obj[0] as usize * nxo, 3);
    let probe = noise(NMODE * N, 5);
    let dev_obj = up(&stream, &obj);
    let dev_probe = up(&stream, &probe);
    let mut want = Vec::new();
    for (s, frames) in STACKS.iter().enumerate() {
        let seed = 10 * s as u64;
        let chi = noise(NF_MAX * NMODE * N, seed + 7);
        let waves = noise(NF_MAX * NMODE * N, seed + 9);
        let deriv = noise(2 * NF_MAX * N, seed + 11);
        let (dev_chi, dev_waves, dev_deriv) =
            (up(&stream, &chi), up(&stream, &waves), up(&stream, &deriv));
        let at = table.at(s);
        match src {
            Source::Slopes => {
                let w = FrameWaves::Probes(&dev_waves);
                k.sums(&ctx, &views, at, &dev_obj, (&dev_chi, w), &mut pos)
            }
            Source::Waves => {
                let w = SliceSource::Waves(FrameWaves::Views(&dev_waves));
                k.derivative_sums(&ctx, &views, at, &dev_deriv, (&dev_chi, w), &mut pos)
            }
            Source::Probe => {
                let w = SliceSource::Probe(&dev_probe);
                k.derivative_sums(&ctx, &views, at, &dev_deriv, (&dev_chi, w), &mut pos)
            }
        }
        .unwrap();
        for fl in 0..frames.len() {
            let chi = |i| chi[fl * NMODE * N + stored(i)];
            let sums = match src {
                Source::Slopes => {
                    let v = addr[(frames.start as usize + fl) * NMODE];
                    let patch: Vec<C64> = (0..N)
                        .map(|i| {
                            obj[(v.oy as usize + i / SHAPE[1]) * nxo + v.ox as usize + i % SHAPE[1]]
                        })
                        .collect();
                    let g = [slope(&patch, 0), slope(&patch, 1)];
                    let w = |i| waves[fl * NMODE * N + stored(i)];
                    frame_sums([&g[0], &g[1]], w, chi)
                }
                Source::Waves | Source::Probe => {
                    let g = [0, 1].map(|ax| &deriv[(ax * NF_MAX + fl) * N..][..N]);
                    match src {
                        Source::Waves => frame_sums(g, |i| waves[fl * NMODE * N + stored(i)], chi),
                        _ => frame_sums(g, |i| probe[i] * quad(quad_phase as f64, i), chi),
                    }
                }
            };
            want.push(sums);
        }
    }
    let mut offsets = DeviceBuffer::<f32>::zeroed(&stream, 2 * NFRAME).unwrap();
    let solve = Solve {
        rule: RULE,
        scale: 1.0,
    };
    let batch = 0..NFRAME as u32;
    k.step(&ctx, batch, &order, &mut offsets, solve, &mut pos)
        .unwrap();
    let got = offsets.to_host_vec(&stream).unwrap();
    (got.iter().map(|&v| v as f64).collect(), step(&want))
}

fn check(name: &str, (got, want): (Vec<f64>, Vec<f64>)) {
    let d = worst_of(got.iter().zip(&want).map(|(g, w)| (g - w).abs()));
    let rel = d / want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    assert!(
        rel < 1e-4,
        "{name}: offsets {got:?} against {want:?}, max rel {rel:e}"
    );
}

#[test]
fn slope_sums_step_as_the_host_does() {
    check("slopes", run(Source::Slopes, 0.0));
}

#[test]
fn derivative_sums_of_view_waves_step_as_the_host_does() {
    check("view waves", run(Source::Waves, 0.0));
}

#[test]
fn derivative_sums_of_the_probe_take_the_exit_phase() {
    check("probe", run(Source::Probe, 0.013));
}
