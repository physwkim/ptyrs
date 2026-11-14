//! The step-size sums on [`window_views`] over each stack's windows, cut from the object at
//! whole-pixel views in the sampler's layout, against the same sums on the scan's views,
//! through the step records. Frames of two probe and two object modes in stacks of 3 and 2,
//! the shorter leaving noise in its last window rows.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs::base::C64;
use ptyrs::gpu::sampling::window_views;
use ptyrs::gpu::{
    C32, Kernels, LsqmlAt, LsqmlChi, LsqmlKernels, LsqmlScan, StackTable, ViewAddr, ViewDims, Views,
};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const NFRAME: usize = 5;
const NPROBE: usize = 2;
const NOBJ: usize = 2;
const NMODE: usize = NPROBE * NOBJ;
const NF_MAX: usize = 3;
const STACKS: [Range<u32>; 2] = [0..3, 3..5];
const SHAPE: [usize; 2] = [64, 64];
const N: usize = SHAPE[0] * SHAPE[1];
const OBJ: [usize; 2] = [96, 100];
const PLANE: usize = OBJ[0] * OBJ[1];
/// A stack's windows, `[nobj * nf_max, ny, nx]`.
const WIN: usize = NOBJ * NF_MAX * N;
/// A stack's waves, `[nf_max * nmode, ny, nx]`.
const WAVES: usize = NF_MAX * NMODE * N;

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
        nmode: NMODE as u32,
        nprobe: NPROBE as u32,
        nobj: NOBJ as u32,
        frame: SHAPE.map(|n| n as u32),
        obj: OBJ.map(|n| n as u32),
        quad_phase: 0.0,
    }
}

/// Frame `f`'s corner.
fn corner(f: usize) -> [usize; 2] {
    [3 + 5 * f, 30 - 4 * f]
}

fn views() -> Vec<ViewAddr> {
    (0..NFRAME)
        .flat_map(|f| {
            let [oy, ox] = corner(f).map(|c| c as u32);
            (0..NMODE as u32).map(move |m| ViewAddr {
                oy,
                ox,
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

/// Object `[nobj, nyo, nxo]` pixel `(y, x)` of frame `f`'s window on object mode `o`.
fn at(o: usize, f: usize, y: usize, x: usize) -> usize {
    let [oy, ox] = corner(f);
    o * PLANE + (oy + y) * OBJ[1] + ox + x
}

/// Stack `s`'s windows of `obj` in the sampler's layout, plane `o * nf_max + fl`; the rows
/// of frames past the stack's hold noise.
fn windows(obj: &[C64], s: usize) -> Vec<C64> {
    let r = &STACKS[s];
    let mut win = noise(WIN, 100 + s as u64);
    for o in 0..NOBJ {
        for fl in 0..r.len() {
            for i in 0..N {
                let f = r.start as usize + fl;
                win[(o * NF_MAX + fl) * N + i] = obj[at(o, f, i / SHAPE[1], i % SHAPE[1])];
            }
        }
    }
    win
}

/// The largest difference over the largest magnitude of `want`.
fn rel(got: &[C64], want: &[C64]) -> f64 {
    let d = worst_of(got.iter().zip(want).map(|(g, w)| (g - w).norm()));
    d / want.iter().fold(0.0f64, |m, w| m.max(w.norm()))
}

#[test]
fn dots_on_windows_step_as_on_the_object() {
    let (stream, ctx, _, k) = setup();
    let d = dims();
    let (views, wviews) = (
        Views::new(&stream, d, &views()).unwrap(),
        window_views(&stream, &d, &STACKS).unwrap(),
    );
    let table = StackTable::new(&stream, &d, &STACKS).unwrap();
    let (obj, d_obj) = (noise(NOBJ * PLANE, 3), noise(NOBJ * PLANE, 4));
    let (probe, d_probe) = (
        up(&stream, &noise(NPROBE * N, 5)),
        up(&stream, &noise(NPROBE * N, 6)),
    );
    let (dev_obj, dev_d_obj) = (up(&stream, &obj), up(&stream, &d_obj));
    let mut records = Vec::new();
    for on_windows in [false, true] {
        let mut st = LsqmlScan::zeroed(&stream, &d).unwrap();
        for s in 0..STACKS.len() {
            let chi = up(&stream, &noise(WAVES, 20 + s as u64));
            let (win, d_win) = (
                up(&stream, &windows(&obj, s)),
                up(&stream, &windows(&d_obj, s)),
            );
            let (v, x, dx) = match on_windows {
                false => (&views, &dev_obj, &dev_d_obj),
                true => (&wviews, &win, &d_win),
            };
            let at = LsqmlAt {
                x: [x, &probe],
                d: [dx, &d_probe],
            };
            k.dots(&ctx, v, table.at(s), at, LsqmlChi::Unmoved(&chi), &mut st)
                .unwrap();
        }
        k.step(&ctx, 0..NFRAME as u32, &mut st, 1.0).unwrap();
        let rec = st.records().to_host_vec(&stream).unwrap();
        records.push(
            rec.iter()
                .map(|&v| C64::new(v as f64, 0.0))
                .collect::<Vec<_>>(),
        );
    }
    let e = rel(&records[1], &records[0]);
    assert!(
        e < 1e-5,
        "records {:?} against {:?}, max rel {e:e}",
        records[1],
        records[0]
    );
}
