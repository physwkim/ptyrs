//! The LSQML preconditioner of a sampled illumination map against an `f64` host reference,
//! read through the preconditioned direction of a unit object direction: a first and a
//! second update, and at whole-pixel views without offsets the sampler's map against
//! [`LsqmlKernels::precondition`] of the views.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs::base::C64;
use ptyrs::gpu::sampling::ObjectSampler;
use ptyrs::gpu::{C32, Kernels, LsqmlKernels, LsqmlScan, MASKED, ViewAddr, ViewDims, Views};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const NFRAME: usize = 3;
const NPROBE: usize = 2;
const NOBJ: usize = 2;
const SHAPE: [usize; 2] = [16, 16];
const N: usize = SHAPE[0] * SHAPE[1];
const OBJ: [usize; 2] = [40, 36];
const LEN: usize = NOBJ * OBJ[0] * OBJ[1];
const SCALE: f32 = 256.0;

fn setup() -> (Arc<CudaStream>, ExecutionContext, Kernels, LsqmlKernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let lsqml = LsqmlKernels::load(&kernels).expect("bind the LSQML kernels");
    let stream = ctx.default_stream();
    (
        stream.clone(),
        ExecutionContext::new(stream),
        kernels,
        lsqml,
    )
}

fn dims() -> ViewDims {
    ViewDims {
        nframe: NFRAME as u32,
        nmode: (NPROBE * NOBJ) as u32,
        nprobe: NPROBE as u32,
        nobj: NOBJ as u32,
        frame: SHAPE.map(|n| n as u32),
        obj: OBJ.map(|n| n as u32),
        quad_phase: 0.0,
    }
}

fn views() -> Vec<ViewAddr> {
    (0..NFRAME as u32)
        .flat_map(|f| {
            (0..(NPROBE * NOBJ) as u32).map(move |m| ViewAddr {
                oy: 2 + 7 * f,
                ox: 3 + 5 * f,
                fy: 0.0,
                fx: 0.0,
                probe: m % NPROBE as u32,
                obj: m / NPROBE as u32,
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

/// Counts with every seventh pixel masked, and the brightest frame's unmasked photons.
fn counts() -> (Vec<u16>, f64) {
    let c: Vec<u16> = noise(NFRAME * N, 2)
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i % 7 == 3 {
                MASKED
            } else {
                (500.0 * (v.re + 1.0)) as u16
            }
        })
        .collect();
    let power = (0..NFRAME)
        .map(|f| {
            let frame = &c[f * N..][..N];
            frame
                .iter()
                .filter(|&&v| v != MASKED)
                .map(|&v| v as f64)
                .sum()
        })
        .fold(0.0, f64::max);
    (c, power)
}

/// A started run's state on `counts`.
fn started(stream: &CudaStream, ctx: &ExecutionContext, k: &LsqmlKernels) -> LsqmlScan {
    let mut st = LsqmlScan::zeroed(stream, &dims()).unwrap();
    let counts = DeviceBuffer::from_host(stream, &counts().0).unwrap();
    k.start(ctx, &counts, &mut st).unwrap();
    st
}

/// The preconditioned unit object direction at `st`'s preconditioner, its real parts.
fn direction(
    stream: &CudaStream,
    ctx: &ExecutionContext,
    k: &LsqmlKernels,
    st: &LsqmlScan,
) -> Vec<f64> {
    let mut num = up(stream, &vec![C64::new(1.0, 0.0); LEN]);
    k.dir_obj(ctx, st, &mut num).unwrap();
    let got = num.to_host_vec(stream).unwrap();
    got.iter().map(|c| c.re as f64).collect()
}

fn check(name: &str, got: &[f64], want: &[f64]) {
    let d = worst_of(got.iter().zip(want).map(|(g, w)| (g - w).abs()));
    let rel = d / want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    assert!(rel < 1e-5, "{name}: max rel {rel:e}");
}

#[test]
fn maps_update_the_preconditioner_as_the_host_does() {
    let (stream, ctx, _, k) = setup();
    let mut st = started(&stream, &ctx, &k);
    let start = counts().1 / (4.0 * N as f64 * SCALE as f64);
    let mut pre: Option<Vec<f64>> = None;
    for (seed, init) in [(5, true), (6, false)] {
        let illum: Vec<f64> = noise(LEN, seed).iter().map(|c| c.re + 1.0).collect();
        let dev = DeviceBuffer::from_host(
            &stream,
            &illum.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        )
        .unwrap();
        k.precondition_placed(&ctx, &dev, &mut st, init, SCALE)
            .unwrap();
        let next: Vec<f64> = (0..LEN)
            .map(|i| (pre.as_ref().map_or(illum[i] * start, |p| p[i]) + illum[i]) / 2.0)
            .collect();
        let floor = 0.1 * next.iter().copied().fold(0.0, f64::max);
        let want: Vec<f64> = next
            .iter()
            .map(|p| 1.0 / (p * p + floor * floor).sqrt())
            .collect();
        let name = if init { "first" } else { "second" };
        check(name, &direction(&stream, &ctx, &k, &st), &want);
        pre = Some(next);
    }
}

#[test]
fn sampled_map_at_whole_pixels_is_the_view_illumination() {
    let (stream, ctx, kernels, k) = setup();
    let probe = up(&stream, &noise(NPROBE * N, 9));
    let views = Views::new(&stream, dims(), &views()).unwrap();

    let mut sampler = ObjectSampler::new(&kernels, &stream, dims(), 1, 4).unwrap();
    let offsets = DeviceBuffer::<f32>::zeroed(&stream, 2 * NFRAME).unwrap();
    let mut illum = DeviceBuffer::<f32>::zeroed(&stream, LEN).unwrap();
    sampler
        .illumination(&ctx, &views, &offsets, &probe, &mut illum)
        .unwrap();
    let mut placed = started(&stream, &ctx, &k);
    k.precondition_placed(&ctx, &illum, &mut placed, true, SCALE)
        .unwrap();

    let mut by_views = started(&stream, &ctx, &k);
    k.precondition(&ctx, &views, &probe, &mut by_views, true, SCALE)
        .unwrap();
    check(
        "sampled",
        &direction(&stream, &ctx, &k, &placed),
        &direction(&stream, &ctx, &k, &by_views),
    );
}
