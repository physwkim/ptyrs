//! The LSQML launchers on each frame's own probe mode 0, [`OprKernels::unique`]'s planes as
//! [`OwnModes`], against the same launchers on the shared-probe path one frame at a time, each
//! frame on its own probe as `ProbeVariation::unique` gives it: both read the same values in
//! the same order, so they agree bit for bit. The batch starts past position 0 of an epoch
//! order that is not the scan's, with two probe and two object modes in stacks of 3 and 2.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use super::{LsqmlAt, LsqmlChi, LsqmlKernels, LsqmlScan};
use crate::base::{C64, Feature, Probe, Stack, View};
use crate::engine::ProbeVariation;
use crate::gpu::opr::OwnModes;
use crate::gpu::{
    C32, EpochOrder, Kernels, Opr, OprKernels, StackTable, ViewAddr, ViewDims, Views,
};
use crate::worst::worst_of;

const NFRAME: usize = 7;
/// The scan frame at each epoch position.
const ORDER: [u32; NFRAME] = [5, 2, 6, 0, 3, 1, 4];
const NP: usize = 2;
const NOBJ: usize = 2;
const NMODE: usize = NP * NOBJ;
/// Eigenmodes.
const K: usize = 2;
/// The batch's epoch positions, and its stacks.
const BATCH: Range<u32> = 2..7;
const NB: usize = 5;
const STACKS: [Range<u32>; 2] = [2..5, 5..7];
const NF_MAX: usize = 3;
const SHAPE: [usize; 2] = [64, 64];
const N: usize = SHAPE[0] * SHAPE[1];
const OBJ: [usize; 2] = [96, 100];

struct Gpu {
    stream: Arc<CudaStream>,
    ctx: ExecutionContext,
    lsqml: LsqmlKernels,
    opr: OprKernels,
    /// The scan's views in [`ORDER`].
    order: EpochOrder,
    /// The views by frame `order` was set from.
    _views: Views,
}

fn gpu() -> Gpu {
    let cx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&cx).unwrap();
    let stream = cx.default_stream();
    let ctx = ExecutionContext::new(stream.clone());
    let views = Views::new(&stream, dims(), &scan_views()).unwrap();
    let mut order = EpochOrder::new(&kernels, &stream, &views).unwrap();
    order.set(&ctx, &kernels, &views, &ORDER).unwrap();
    Gpu {
        lsqml: LsqmlKernels::load(&kernels).unwrap(),
        opr: OprKernels::load(&kernels).unwrap(),
        stream,
        ctx,
        order,
        _views: views,
    }
}

fn dims() -> ViewDims {
    ViewDims {
        nframe: NFRAME as u32,
        nmode: NMODE as u32,
        nprobe: NP as u32,
        nobj: NOBJ as u32,
        frame: SHAPE.map(|n| n as u32),
        obj: OBJ.map(|n| n as u32),
        quad_phase: 2e-3,
    }
}

/// Frame `f`'s corner.
fn corner(f: u32) -> [u32; 2] {
    [2 + 4 * f, 30 - 4 * f]
}

fn scan_views() -> Vec<ViewAddr> {
    (0..NFRAME as u32)
        .flat_map(|f| {
            let [oy, ox] = corner(f);
            (0..NMODE as u32).map(move |m| ViewAddr {
                oy,
                ox,
                fy: 0.0,
                fx: 0.0,
                probe: m % NP as u32,
                obj: m / NP as u32,
            })
        })
        .collect()
}

/// Deterministic values in `[-1, 1)` per part, each rounded to f32.
fn noise(len: usize, seed: u64) -> Vec<C64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 52) as f64 - 1.0) as f32 as f64
    };
    (0..len).map(|_| C64::new(next(), next())).collect()
}

fn c32(z: &[C64]) -> Vec<C32> {
    z.iter()
        .map(|z| C32 {
            re: z.re as f32,
            im: z.im as f32,
        })
        .collect()
}

fn up(stream: &CudaStream, z: &[C64]) -> DeviceBuffer<C32> {
    DeviceBuffer::from_host(stream, &c32(z)).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The scan's variation and shared probe, the variation's main-mode weights near 1 and its
/// eigenmode weights near ±0.1.
fn variation() -> (ProbeVariation, Probe) {
    let eigen = Stack::from_vec([K, SHAPE[0], SHAPE[1]], noise(K * N, 7)).unwrap();
    let w = noise(NFRAME * K, 8).iter().map(|z| 0.1 * z.re).collect();
    let c = noise(NFRAME, 9).iter().map(|z| 1.0 + 0.1 * z.re).collect();
    let var = ProbeVariation::new(eigen, Feature::refined(w)).with_intensity(Feature::refined(c));
    let probe = Stack::from_vec([NP, SHAPE[0], SHAPE[1]], noise(NP * N, 10)).unwrap();
    (var, Probe::new(probe))
}

/// [`OprKernels::unique`]'s planes of the batch, checked against the CPU's: the views in epoch
/// order give the `b`-th frame, at position `BATCH.start + b`, the CPU's planes `b·NP + m`.
fn own_planes(g: &Gpu, var: &ProbeVariation, probe: &Probe) -> DeviceBuffer<C32> {
    let f32s = |x: &[f64]| x.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let (w, c) = (
        f32s(&var.weights().values),
        f32s(&var.intensity().unwrap().values),
    );
    let opr = Opr::new(
        &g.stream,
        N,
        &c32(var.eigenmodes().as_slice()),
        Feature::refined(w.as_slice()),
        Some(Feature::refined(c.as_slice())),
        NB as u32,
    )
    .unwrap();
    let mut planes = DeviceBuffer::<C32>::zeroed(&g.stream, NB * N).unwrap();
    let shared = up(&g.stream, probe.as_slice());
    g.opr
        .unique(&g.ctx, &opr, &g.order, BATCH, &shared, &mut planes)
        .unwrap();
    let cpu_views: Vec<View> = BATCH
        .flat_map(|p| {
            let f = ORDER[p as usize];
            let [oy, ox] = corner(f);
            (0..NMODE as u32).map(move |m| View {
                frame: f,
                mode: m,
                probe: m % NP as u32,
                obj: m / NP as u32,
                oy,
                ox,
            })
        })
        .collect();
    let (cpu_views, cpu_probe) = var.unique(&cpu_views, probe);
    for (k, v) in cpu_views.iter().enumerate() {
        let plane = k / NMODE * NP + k % NMODE % NP;
        assert_eq!(v.probe, plane as u32, "CPU view {k}'s probe plane");
    }
    let got = planes.to_host_vec(&g.stream).unwrap();
    let want: Vec<C64> = (0..NB)
        .flat_map(|b| cpu_probe.plane(b * NP).iter().copied())
        .collect();
    let scale = worst_of(want.iter().map(|w| w.norm()));
    let diff = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (C64::new(g.re as f64, g.im as f64) - w).norm());
    let err = worst_of(diff) / scale;
    assert!(err <= 5e-7, "own planes against the CPU's: {err:e}");
    planes
}

/// What a step reads: the object and its direction, the shared probe's direction and each
/// stack's `chi`, `[NF_MAX NMODE, n]`.
struct Inputs {
    obj: DeviceBuffer<C32>,
    d_obj: DeviceBuffer<C32>,
    d_probe: DeviceBuffer<C32>,
    chi: Vec<Vec<C64>>,
}

fn inputs(stream: &CudaStream) -> Inputs {
    let plane = OBJ[0] * OBJ[1];
    Inputs {
        obj: up(stream, &noise(NOBJ * plane, 1)),
        d_obj: up(stream, &noise(NOBJ * plane, 2)),
        d_probe: up(stream, &noise(NP * N, 3)),
        chi: (0..STACKS.len())
            .map(|s| noise(NF_MAX * NMODE * N, 20 + s as u64))
            .collect(),
    }
}

/// [`LsqmlKernels::step`] of the batch after `run` has taken its [`LsqmlKernels::dots`] into
/// a fresh scan state: the dots' sums and the records.
fn dots(g: &Gpu, run: impl FnOnce(&mut LsqmlScan)) -> [Vec<u32>; 2] {
    let mut st = LsqmlScan::zeroed(&g.stream, &dims()).unwrap();
    run(&mut st);
    g.lsqml.step(&g.ctx, BATCH, &mut st, 1.0).unwrap();
    [&st.dots, st.records()].map(|b| bits(&b.to_host_vec(&g.stream).unwrap()))
}

/// The dots of [`STACKS`] on the shared probe, each frame's own mode 0 from `own`.
fn stacks(
    g: &Gpu,
    x: &Inputs,
    probe: &DeviceBuffer<C32>,
    own: Option<&DeviceBuffer<C32>>,
) -> [Vec<u32>; 2] {
    let table = StackTable::new(&g.stream, &dims(), &STACKS).unwrap();
    dots(g, |st| {
        for (s, chi) in x.chi.iter().enumerate() {
            let chi = up(&g.stream, chi);
            let chi = match own {
                Some(planes) => LsqmlChi::Own {
                    chi: &chi,
                    own: OwnModes {
                        planes,
                        positions: BATCH,
                    },
                },
                None => LsqmlChi::Unmoved(&chi),
            };
            let at = LsqmlAt {
                x: [&x.obj, probe],
                d: [&x.d_obj, &x.d_probe],
            };
            g.lsqml
                .dots(&g.ctx, g.order.views(), table.at(s), at, chi, st)
                .unwrap();
        }
    })
}

/// The dots of each batch frame alone, on the shared probe with its own mode 0 from `own`.
fn frames(g: &Gpu, x: &Inputs, probe: &Probe, own: &DeviceBuffer<C32>) -> [Vec<u32>; 2] {
    let own = own.to_host_vec(&g.stream).unwrap();
    let shared = c32(probe.as_slice());
    let single: Vec<Range<u32>> = BATCH.map(|p| p..p + 1).collect();
    let table = StackTable::new(&g.stream, &dims(), &single).unwrap();
    dots(g, |st| {
        for (b, p) in BATCH.enumerate() {
            let s = STACKS.iter().position(|r| r.contains(&p)).unwrap();
            let fl = (p - STACKS[s].start) as usize;
            let chi = up(&g.stream, &x.chi[s][fl * NMODE * N..(fl + 1) * NMODE * N]);
            let mode0 = &own[b * N..(b + 1) * N];
            let probe: Vec<C32> = mode0.iter().chain(&shared[N..]).copied().collect();
            let probe = DeviceBuffer::from_host(&g.stream, &probe).unwrap();
            let at = LsqmlAt {
                x: [&x.obj, &probe],
                d: [&x.d_obj, &x.d_probe],
            };
            let chi = LsqmlChi::Unmoved(&chi);
            g.lsqml
                .dots(&g.ctx, g.order.views(), table.at(b), at, chi, st)
                .unwrap();
        }
    })
}

#[test]
fn dots_on_own_modes_match_each_frame_alone() {
    let g = gpu();
    let (var, probe) = variation();
    let planes = own_planes(&g, &var, &probe);
    let x = inputs(&g.stream);
    let shared = up(&g.stream, probe.as_slice());
    let own = stacks(&g, &x, &shared, Some(&planes));
    let want = frames(&g, &x, &probe, &planes);
    assert_eq!(own[0], want[0], "dots' sums");
    assert_eq!(own[1], want[1], "records");
    let plain = stacks(&g, &x, &shared, None);
    assert_ne!(
        plain[0], own[0],
        "the own modes left the sums as on the shared probe"
    );
}
