//! The LSQML launchers under OPR. The forward row pass, the object sums and the dots on each
//! frame's own probe mode 0, [`OprKernels::unique`]'s planes as [`OwnModes`], against the same launchers on the
//! shared-probe path one frame at a time, each frame on its own probe as
//! `ProbeVariation::unique` gives it: both read the same values in the same order, so they
//! agree bit for bit. Each frame's probe direction against the CPU's over those views, and bit
//! for bit against [`Kernels::psi_to_probe`] one frame at a time. The shifted probe on each
//! frame's own spectra against the same launcher one frame at a time on its own probe, and
//! against the CPU's move of that probe. The batch starts past position 0 of an epoch order
//! that is not the scan's, with two probe and two object modes in stacks of 3 and 2.

use std::f64::consts::PI;
use std::ops::Range;
use std::slice;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use super::{FrameDirs, LsqmlAt, LsqmlChi, LsqmlKernels, LsqmlScan};
use crate::base::{C64, Counts, Feature, Geometry, Object, Probe, Stack, View};
use crate::engine::{
    Cost, Features, Lsqml, LsqmlBatch, LsqmlParams, Observed, Optics, ProbeVariation, Sampling,
    accumulate_probe,
};
use crate::gpu::fused::{FrameMoves, OwnSpectrumBuf};
use crate::gpu::opr::{FrameProbe, OwnModes};
use crate::gpu::{
    C32, EpochOrder, Exit, FarSums, FusedBufs, FusedInputs, FusedSums, Kernels, Model, Opr,
    OprKernels, ProbeSpectrum, PsiSums, Shift, ShiftProbe, ShiftSums, ShiftedProbeSums,
    SpectrumBuf, StackTable, UpdateSums, ViewAddr, ViewDims, Views,
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
const QUAD: f32 = 2e-3;
/// Each epoch position's offsets `(dy, dx)` in pixels, one at zero.
const OFFSETS: [[f32; 2]; NFRAME] = [
    [0.3, -1.7],
    [2.25, 0.5],
    [-0.6, 3.4],
    [0.0, 0.0],
    [-2.1, -0.35],
    [1.15, 2.6],
    [-3.3, 0.8],
];

struct Gpu {
    stream: Arc<CudaStream>,
    ctx: ExecutionContext,
    kernels: Kernels,
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
        kernels,
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
        quad_phase: QUAD,
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

fn c_bits(v: &[C32]) -> Vec<u32> {
    v.iter()
        .flat_map(|c| [c.re, c.im])
        .map(f32::to_bits)
        .collect()
}

/// The largest difference over the largest magnitude of `want`.
fn rel(got: impl Iterator<Item = C64>, want: &[C64]) -> f64 {
    let scale = worst_of(want.iter().map(|w| w.norm()));
    worst_of(got.zip(want).map(|(g, w)| (g - w).norm())) / scale
}

/// The views of the batch's frames in epoch order.
fn batch_views() -> Vec<View> {
    BATCH
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
        .collect()
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
    let (cpu_views, cpu_probe) = var.unique(&batch_views(), probe);
    for (k, v) in cpu_views.iter().enumerate() {
        let plane = k / NMODE * NP + k % NMODE % NP;
        assert_eq!(v.probe, plane as u32, "CPU view {k}'s probe plane");
    }
    let got = planes.to_host_vec(&g.stream).unwrap();
    let want: Vec<C64> = (0..NB)
        .flat_map(|b| cpu_probe.plane(b * NP).iter().copied())
        .collect();
    let err = rel(
        got.iter().map(|g| C64::new(g.re as f64, g.im as f64)),
        &want,
    );
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

/// The object, `[NOBJ, OBJ]`.
fn obj() -> Vec<C64> {
    noise(NOBJ * OBJ[0] * OBJ[1], 1)
}

fn inputs(stream: &CudaStream) -> Inputs {
    let plane = OBJ[0] * OBJ[1];
    Inputs {
        obj: up(stream, &obj()),
        d_obj: up(stream, &noise(NOBJ * plane, 2)),
        d_probe: up(stream, &noise(NP * N, 3)),
        chi: (0..STACKS.len())
            .map(|s| noise(NF_MAX * NMODE * N, 20 + s as u64))
            .collect(),
    }
}

impl Inputs {
    /// The `chi` of the frame at epoch position `p` of the batch, `[NMODE, n]`.
    fn frame_chi(&self, p: u32) -> &[C64] {
        let s = STACKS.iter().position(|r| r.contains(&p)).unwrap();
        let fl = (p - STACKS[s].start) as usize;
        &self.chi[s][fl * NMODE * N..(fl + 1) * NMODE * N]
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

/// Each batch frame's probe alone: the shared probe with its own mode 0 from `own`.
fn frame_probes(g: &Gpu, probe: &Probe, own: &DeviceBuffer<C32>) -> Vec<DeviceBuffer<C32>> {
    let own = own.to_host_vec(&g.stream).unwrap();
    let shared = c32(probe.as_slice());
    own.as_chunks::<N>()
        .0
        .iter()
        .map(|mode0| {
            let probe: Vec<C32> = mode0.iter().chain(&shared[N..]).copied().collect();
            DeviceBuffer::from_host(&g.stream, &probe).unwrap()
        })
        .collect()
}

/// The batch's frames one stack each.
fn single() -> Vec<Range<u32>> {
    BATCH.map(|p| p..p + 1).collect()
}

/// The dots of each batch frame alone, on the shared probe with its own mode 0 from `own`.
fn frames(g: &Gpu, x: &Inputs, probe: &Probe, own: &DeviceBuffer<C32>) -> [Vec<u32>; 2] {
    let probes = frame_probes(g, probe, own);
    let table = StackTable::new(&g.stream, &dims(), &single()).unwrap();
    dots(g, |st| {
        for ((b, p), probe) in BATCH.enumerate().zip(&probes) {
            let chi = up(&g.stream, x.frame_chi(p));
            let at = LsqmlAt {
                x: [&x.obj, probe],
                d: [&x.d_obj, &x.d_probe],
            };
            let chi = LsqmlChi::Unmoved(&chi);
            g.lsqml
                .dots(&g.ctx, g.order.views(), table.at(b), at, chi, st)
                .unwrap();
        }
    })
}

/// The batch's row-transformed exit waves, `[NB NMODE, n]`: [`LsqmlKernels::forward_frames`]
/// of stack `s` of `stacks` on `probes[s]`.
fn exit_waves(g: &Gpu, x: &Inputs, stacks: &[Range<u32>], probes: &[FrameProbe<'_>]) -> Vec<C32> {
    let table = StackTable::new(&g.stream, &dims(), stacks).unwrap();
    let nf_max = stacks.iter().map(|r| r.len()).max().unwrap();
    let mut work = DeviceBuffer::<C32>::zeroed(&g.stream, nf_max * NMODE * N).unwrap();
    let mut out = Vec::new();
    for (s, (r, &probe)) in stacks.iter().zip(probes).enumerate() {
        let views = g.order.views();
        g.lsqml
            .forward_frames(&g.ctx, views, table.at(s), &x.obj, probe, &mut work)
            .unwrap();
        out.extend_from_slice(&work.to_host_vec(&g.stream).unwrap()[..r.len() * NMODE * N]);
    }
    out
}

#[test]
fn forward_on_own_modes_matches_each_frame_alone() {
    let g = gpu();
    let (var, probe) = variation();
    let planes = own_planes(&g, &var, &probe);
    let x = inputs(&g.stream);
    let shared = up(&g.stream, probe.as_slice());
    let own = OwnModes {
        planes: &planes,
        positions: BATCH,
    };
    let got = exit_waves(&g, &x, &STACKS, &[FrameProbe::Own(&shared, &own); 2]);
    let probes = frame_probes(&g, &probe, &planes);
    let alone: Vec<_> = probes.iter().map(FrameProbe::Shared).collect();
    let want = exit_waves(&g, &x, &single(), &alone);
    assert_eq!(c_bits(&got), c_bits(&want));
    let plain = exit_waves(&g, &x, &STACKS, &[FrameProbe::Shared(&shared); 2]);
    assert_ne!(
        c_bits(&plain),
        c_bits(&got),
        "the own modes left the waves as on the shared probe"
    );
}

/// [`Kernels::psi_to_obj_frames`] of `chi` as the waves, stack `s` of `stacks` on
/// `probes[s]`, into sums that start at zero.
fn obj_sums(
    g: &Gpu,
    chi: &[Vec<C64>],
    stacks: &[Range<u32>],
    probes: &[FrameProbe<'_>],
) -> [Vec<u32>; 2] {
    let table = StackTable::new(&g.stream, &dims(), stacks).unwrap();
    let mut sums = UpdateSums::zeroed(&g.stream, NOBJ * OBJ[0] * OBJ[1]).unwrap();
    for (s, (chi, &probe)) in chi.iter().zip(probes).enumerate() {
        let psi = up(&g.stream, chi);
        let views = g.order.views();
        g.kernels
            .psi_to_obj_frames(&g.ctx, views, table.at(s), &psi, probe, &mut sums)
            .unwrap();
    }
    [
        c_bits(&sums.num.to_host_vec(&g.stream).unwrap()),
        bits(&sums.den.to_host_vec(&g.stream).unwrap()),
    ]
}

#[test]
fn psi_to_obj_on_own_modes_matches_each_frame_alone() {
    let g = gpu();
    let (var, probe) = variation();
    let planes = own_planes(&g, &var, &probe);
    let x = inputs(&g.stream);
    let shared = up(&g.stream, probe.as_slice());
    let own = OwnModes {
        planes: &planes,
        positions: BATCH,
    };
    let got = obj_sums(&g, &x.chi, &STACKS, &[FrameProbe::Own(&shared, &own); 2]);
    let probes = frame_probes(&g, &probe, &planes);
    let alone: Vec<_> = probes.iter().map(FrameProbe::Shared).collect();
    let chi: Vec<_> = BATCH.map(|p| x.frame_chi(p).to_vec()).collect();
    let want = obj_sums(&g, &chi, &single(), &alone);
    assert_eq!(got[0], want[0], "num");
    assert_eq!(got[1], want[1], "den");
    let plain = obj_sums(&g, &x.chi, &STACKS, &[FrameProbe::Shared(&shared); 2]);
    assert_ne!(
        plain[1], got[1],
        "the own modes left the sums as on the shared probe"
    );
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

/// [`LsqmlKernels::frame_dirs`] of the batch, `[NB, n]` each, over a full and a short batch
/// stack and then a stack straddling the batch start and one before it, whose frames outside
/// the batch must write nothing: launched last so no batch stack can overwrite a stray
/// plane. Every plane starts as NaN.
fn frame_dirs(g: &Gpu, x: &Inputs) -> (Vec<C32>, Vec<f32>) {
    let nan = C32 {
        re: f32::NAN,
        im: f32::NAN,
    };
    let mut d = DeviceBuffer::from_host(&g.stream, &vec![nan; NB * N]).unwrap();
    let mut vv = DeviceBuffer::from_host(&g.stream, &vec![f32::NAN; NB * N]).unwrap();
    let stacks = [3..6, 6..7, 1..3, 0..BATCH.start];
    let table = StackTable::new(&g.stream, &dims(), &stacks).unwrap();
    let m = NMODE * N;
    for (s, r) in stacks.iter().enumerate() {
        let mut chi = noise(NF_MAX * m, 30 + s as u64);
        for (fl, p) in r.clone().enumerate().filter(|(_, p)| BATCH.contains(p)) {
            chi[fl * m..(fl + 1) * m].copy_from_slice(x.frame_chi(p));
        }
        let out = FrameDirs {
            d: &mut d,
            vv: &mut vv,
            positions: BATCH,
        };
        let chi = up(&g.stream, &chi);
        let views = g.order.views();
        g.lsqml
            .frame_dirs(&g.ctx, views, table.at(s), &x.obj, &chi, out)
            .unwrap();
    }
    (
        d.to_host_vec(&g.stream).unwrap(),
        vv.to_host_vec(&g.stream).unwrap(),
    )
}

#[test]
fn frame_dirs_match_the_cpu_and_psi_to_probe_frame_by_frame() {
    let g = gpu();
    let (var, probe) = variation();
    let x = inputs(&g.stream);
    let (d, vv) = frame_dirs(&g, &x);
    let (views, _) = var.unique(&batch_views(), &probe);
    let object = Object::new(Stack::from_vec([NOBJ, OBJ[0], OBJ[1]], obj()).unwrap());
    let half = SHAPE.map(|n| (n / 2) as f64);
    let quad: Vec<C64> = (0..N)
        .map(|i| {
            let (y, x) = (
                (i / SHAPE[1]) as f64 - half[0],
                (i % SHAPE[1]) as f64 - half[1],
            );
            C64::from_polar(1.0, QUAD as f64 * (y * y + x * x))
        })
        .collect();
    let psi: Vec<C64> = BATCH.flat_map(|p| x.frame_chi(p).to_vec()).collect();
    let (num, norm) = accumulate_probe(&views, &object, [NB * NP, SHAPE[0], SHAPE[1]], &quad, &psi);
    let want: Vec<C64> = (0..NB).flat_map(|b| num.plane(b * NP).to_vec()).collect();
    let err = rel(d.iter().map(|d| C64::new(d.re as f64, d.im as f64)), &want);
    assert!(err <= 1e-6, "d against the CPU's: {err:e}");
    let want: Vec<C64> = (0..NB)
        .flat_map(|b| norm.plane(b * NP).iter().map(|&v| C64::new(v, 0.0)))
        .collect();
    let err = rel(vv.iter().map(|&v| C64::new(v as f64, 0.0)), &want);
    assert!(err <= 2e-7, "vv against the CPU's: {err:e}");
    let table = StackTable::new(&g.stream, &dims(), &single()).unwrap();
    for (b, p) in BATCH.enumerate() {
        let mut sums = UpdateSums::zeroed(&g.stream, NP * N).unwrap();
        let chi = up(&g.stream, x.frame_chi(p));
        g.kernels
            .psi_to_probe(
                &g.ctx,
                g.order.views(),
                table.at(b),
                &chi,
                &x.obj,
                &mut sums,
            )
            .unwrap();
        let num = sums.num.to_host_vec(&g.stream).unwrap();
        let den = sums.den.to_host_vec(&g.stream).unwrap();
        let plane = b * N..(b + 1) * N;
        assert_eq!(
            c_bits(&d[plane.clone()]),
            c_bits(&num[..N]),
            "d at position {p}"
        );
        assert_eq!(bits(&vv[plane]), bits(&den[..N]), "vv at position {p}");
    }
}

/// The far field whose plane phase is the GPU's `exp(i QUAD r²)`: `π/(λz)·px²` with `px =
/// λz/(ny d)` is `QUAD`.
fn optics() -> Optics {
    let (wavelength, distance) = (1e-10, 1.0);
    let d = PI * wavelength * distance / (QUAD as f64 * (SHAPE[0] * SHAPE[0]) as f64);
    Optics::far_field(&Geometry::new(wavelength, distance, d.sqrt(), SHAPE).unwrap())
}

/// [`LsqmlBatch::adjoint_move`] of `x`, `[NB NP, n]` by batch frame, under a Patch batch whose
/// frame at epoch position `p` sits at `offsets[p]`: each frame's planes `move_planes` by
/// `−offsets[p]`.
fn cpu_move_back(x: &Stack<C64>, offsets: [[f64; 2]; NFRAME]) -> Stack<C64> {
    let (var, probe) = variation();
    let mut by_frame = vec![[0.0; 2]; NFRAME];
    for (&f, &d) in ORDER.iter().zip(&offsets) {
        by_frame[f as usize] = d;
    }
    let params = LsqmlParams {
        batch_size: NB,
        cost: Cost::Amplitude,
        seed: 0,
    };
    let features = Features {
        position: Some(Feature::refined(by_frame)),
        ..Features::default()
    };
    let run = (Lsqml::new(params).unwrap().with_sampling(Sampling::Patch))
        .with_features(features)
        .unwrap();
    let counts = Counts::from_raw([NFRAME, SHAPE[0], SHAPE[1]], vec![0; NFRAME * N]).unwrap();
    let obs = Observed::from_counts(&counts);
    let object = Object::new(Stack::from_vec([NOBJ, OBJ[0], OBJ[1]], obj()).unwrap());
    let (views, optics) = (batch_views(), optics());
    let objects = slice::from_ref(&object);
    LsqmlBatch::new(&run, &views, &obs, &optics, objects, &probe, &var).adjoint_move(x)
}

/// `q·w` of natural-layout planes `w` in the exit waves' layout, origin at `[0, 0]`.
fn exit_layout(w: &[C64], q: &[C64]) -> Vec<C64> {
    let [ny, nx] = SHAPE;
    let mut out = vec![C64::default(); w.len()];
    for (o, w) in out
        .as_chunks_mut::<N>()
        .0
        .iter_mut()
        .zip(w.as_chunks::<N>().0)
    {
        for (i, (&w, &q)) in w.iter().zip(q).enumerate() {
            o[(i / nx + ny / 2) % ny * nx + (i % nx + nx / 2) % nx] = q * w;
        }
    }
    out
}

/// The unnormalized inverse DFT of each row of `t`: [`Shift`]'s `W` of the column transforms
/// `T` [`Kernels::shifted_probe`] leaves.
fn rows_inverse(t: &[C32]) -> Vec<C64> {
    let nx = SHAPE[1];
    let e: Vec<C64> = (0..nx)
        .map(|k| C64::from_polar(1.0, 2.0 * PI * k as f64 / nx as f64))
        .collect();
    t.chunks_exact(nx)
        .flat_map(|row| {
            let e = &e;
            (0..nx).map(move |x| {
                (row.iter().enumerate())
                    .map(|(k, t)| C64::new(t.re as f64, t.im as f64) * e[k * x % nx])
                    .sum::<C64>()
            })
        })
        .collect()
}

/// The batch's `T` of [`Kernels::shifted_probe`] at [`OFFSETS`], `[NB NP, n]`: stack `s` of
/// `stacks` on `probes[s]` and its spectra.
fn shifted(
    g: &Gpu,
    stacks: &[Range<u32>],
    probes: &[(&DeviceBuffer<C32>, ProbeSpectrum<'_>)],
) -> Vec<C32> {
    let table = StackTable::new(&g.stream, &dims(), stacks).unwrap();
    let nf_max = stacks.iter().map(|r| r.len()).max().unwrap();
    let mut t = DeviceBuffer::<C32>::zeroed(&g.stream, nf_max * NP * N).unwrap();
    let counts = DeviceBuffer::<u16>::zeroed(&g.stream, NFRAME * N).unwrap();
    let obj = up(&g.stream, &obj());
    let offsets = DeviceBuffer::from_host(&g.stream, OFFSETS.as_flattened()).unwrap();
    let mut out = Vec::new();
    for (s, (r, &(probe, spectrum))) in stacks.iter().zip(probes).enumerate() {
        let inputs = FusedInputs {
            views: g.order.views(),
            counts: &counts,
            obj: &obj,
            probe,
            model: Model {
                shift: Some(Shift {
                    offsets: &offsets,
                    spectrum,
                }),
                ..Model::default()
            },
        };
        (g.kernels)
            .shifted_probe(&g.ctx, &inputs, table.at(s), &mut t)
            .unwrap();
        out.extend_from_slice(&t.to_host_vec(&g.stream).unwrap()[..r.len() * NP * N]);
    }
    out
}

/// The shifted probe on [`Kernels::own_spectrum`]'s spectra of each frame's own mode 0: the
/// CPU's move of the frame's own probe, `move_planes` at its offsets
/// ([`LsqmlBatch::adjoint_move`] at their negatives).
#[test]
fn shifted_probe_on_own_spectra_matches_the_cpu_and_each_frame_alone() {
    let g = gpu();
    let d = dims();
    let (var, probe) = variation();
    let planes = own_planes(&g, &var, &probe);
    let shared = up(&g.stream, probe.as_slice());
    let own = OwnModes {
        planes: &planes,
        positions: BATCH,
    };
    let mut buf = SpectrumBuf::zeroed(&g.stream, &d).unwrap();
    let spectrum = (g.kernels)
        .probe_spectrum(&g.ctx, &d, &shared, &mut buf)
        .unwrap();
    let mut own_buf = OwnSpectrumBuf::zeroed(&g.stream, &d, NB as u32).unwrap();
    let own_spectrum = (g.kernels)
        .own_spectrum(&g.ctx, &d, spectrum, &own, &mut own_buf)
        .unwrap();
    let got = shifted(&g, &STACKS, &[(&shared, own_spectrum); 2]);
    let probes = frame_probes(&g, &probe, &planes);
    let mut bufs: Vec<_> = (probes.iter())
        .map(|_| SpectrumBuf::zeroed(&g.stream, &d).unwrap())
        .collect();
    let alone: Vec<_> = (probes.iter().zip(&mut bufs))
        .map(|(p, b)| (p, g.kernels.probe_spectrum(&g.ctx, &d, p, b).unwrap()))
        .collect();
    assert_eq!(c_bits(&got), c_bits(&shifted(&g, &single(), &alone)));
    let (_, uprobe) = var.unique(&batch_views(), &probe);
    let moved = cpu_move_back(&uprobe, OFFSETS.map(|o| o.map(|o| -f64::from(o))));
    let want = exit_layout(moved.as_slice(), optics().quad());
    let err = rel(rows_inverse(&got).into_iter(), &want);
    eprintln!("W against the CPU's: {err:.3e}");
    assert!(err <= 2e-6, "W against the CPU's: {err:e}");
    let plain = shifted(&g, &STACKS, &[(&shared, spectrum); 2]);
    assert_ne!(
        c_bits(&plain),
        c_bits(&got),
        "the own spectra left T as on the shared spectra"
    );
}

/// The batch's [`FrameMoves`] and the probe sums [`Kernels::shifted_probe_sums`] takes from
/// what `stacks` added at [`OFFSETS`] on the shared probe, under [`ShiftProbe::Frames`] when
/// `frames`, else under [`ShiftProbe::Sums`] with no moves: with `project` the AP projection
/// of [`Kernels::fused_project`], else [`Kernels::shifted_psi_sums`] of each stack's waves,
/// the batch frames' `chi` from `x`. Every plane of the moves starts as NaN.
fn moves(
    g: &Gpu,
    x: &Inputs,
    stacks: &[Range<u32>],
    frames: bool,
    project: bool,
) -> (Option<Vec<C32>>, UpdateSums) {
    let d = dims();
    let (_, probe) = variation();
    let probe = up(&g.stream, probe.as_slice());
    let mut buf = SpectrumBuf::zeroed(&g.stream, &d).unwrap();
    let spectrum = (g.kernels)
        .probe_spectrum(&g.ctx, &d, &probe, &mut buf)
        .unwrap();
    let counts: Vec<u16> = (noise(NFRAME * N, 40).iter())
        .map(|z| ((z.re + 1.0) * 40.0) as u16)
        .collect();
    let counts = DeviceBuffer::from_host(&g.stream, &counts).unwrap();
    let offsets = DeviceBuffer::from_host(&g.stream, OFFSETS.as_flattened()).unwrap();
    let inputs = FusedInputs {
        views: g.order.views(),
        counts: &counts,
        obj: &x.obj,
        probe: &probe,
        model: Model {
            shift: Some(Shift {
                offsets: &offsets,
                spectrum,
            }),
            ..Model::default()
        },
    };
    let table = StackTable::new(&g.stream, &d, stacks).unwrap();
    let mut bufs = FusedBufs::with_shift(&g.stream, &d, table.nf_max()).unwrap();
    let mut sums = ShiftedProbeSums::zeroed(&g.stream, &d).unwrap();
    let nan = C32 {
        re: f32::NAN,
        im: f32::NAN,
    };
    let mut moved = DeviceBuffer::from_host(&g.stream, &vec![nan; NB * N]).unwrap();
    let m = NMODE * N;
    for (s, r) in stacks.iter().enumerate() {
        let probe = match frames {
            true => ShiftProbe::Frames {
                sums: &mut sums,
                moved: FrameMoves {
                    planes: &mut moved,
                    positions: BATCH,
                },
            },
            false => ShiftProbe::Sums(&mut sums),
        };
        if project {
            let sums = FusedSums {
                shift: ShiftSums {
                    probe: Some(probe),
                    ..ShiftSums::default()
                },
                ..FusedSums::default()
            };
            let exit = Exit::Ap {
                far: FarSums::default(),
            };
            (g.kernels)
                .fused_project(&g.ctx, &inputs, table.at(s), exit, &mut bufs, sums)
                .unwrap();
            continue;
        }
        let mut chi = noise(NF_MAX * m, 30 + s as u64);
        for (fl, p) in r.clone().enumerate().filter(|(_, p)| BATCH.contains(p)) {
            chi[fl * m..(fl + 1) * m].copy_from_slice(x.frame_chi(p));
        }
        let psi = up(&g.stream, &chi);
        let sums = PsiSums {
            obj: None,
            probe: Some(probe),
        };
        (g.kernels)
            .shifted_psi_sums(&g.ctx, &inputs, table.at(s), &psi, &mut bufs, sums)
            .unwrap();
    }
    let mut out = UpdateSums::zeroed(&g.stream, NP * N).unwrap();
    (g.kernels)
        .shifted_probe_sums(&g.ctx, &d, &mut sums, &mut out)
        .unwrap();
    (frames.then(|| moved.to_host_vec(&g.stream).unwrap()), out)
}

/// The bits of probe sums.
fn sum_bits(g: &Gpu, sums: &UpdateSums) -> [Vec<u32>; 2] {
    [
        c_bits(&sums.num.to_host_vec(&g.stream).unwrap()),
        bits(&sums.den.to_host_vec(&g.stream).unwrap()),
    ]
}

/// Each batch frame's own addend of probe mode 0 to the shifted probe sums, taken back alone:
/// the CPU's move of the frame's direction back by its offsets ([`LsqmlBatch::adjoint_move`]),
/// over stacks straddling the batch whose frames outside it must write nothing, launched
/// last; the sums as under [`ShiftProbe::Sums`].
#[test]
fn frame_moves_match_the_cpu_and_leave_the_sums() {
    let g = gpu();
    let x = inputs(&g.stream);
    let stacks = [3..6, 6..7, 1..3, 0..BATCH.start];
    let (moved, sums) = moves(&g, &x, &stacks, true, false);
    let (_, plain) = moves(&g, &x, &stacks, false, false);
    assert_eq!(sum_bits(&g, &sums), sum_bits(&g, &plain), "the probe sums");
    let (var, probe) = variation();
    let (views, _) = var.unique(&batch_views(), &probe);
    let object = Object::new(Stack::from_vec([NOBJ, OBJ[0], OBJ[1]], obj()).unwrap());
    let optics = optics();
    let psi: Vec<C64> = BATCH.flat_map(|p| x.frame_chi(p).to_vec()).collect();
    let shape = [NB * NP, SHAPE[0], SHAPE[1]];
    let (d, _) = accumulate_probe(&views, &object, shape, optics.quad(), &psi);
    let back = cpu_move_back(&d, OFFSETS.map(|o| o.map(f64::from)));
    let want: Vec<C64> = (0..NB).flat_map(|b| back.plane(b * NP).to_vec()).collect();
    let got = moved.unwrap().into_iter();
    let err = rel(got.map(|m| C64::new(m.re.into(), m.im.into())), &want);
    eprintln!("moves against the CPU's: {err:.3e}");
    assert!(err <= 2e-6, "moves against the CPU's: {err:e}");
}

/// [`Kernels::fused_project`]'s moves under [`ShiftProbe::Frames`]: each frame's addend of the
/// probe sums it leaves, which they sum to, and those sums as under [`ShiftProbe::Sums`].
#[test]
fn projected_frame_moves_sum_to_the_probe_sums() {
    let g = gpu();
    let x = inputs(&g.stream);
    let (moved, sums) = moves(&g, &x, &STACKS, true, true);
    let (_, plain) = moves(&g, &x, &STACKS, false, true);
    assert_eq!(sum_bits(&g, &sums), sum_bits(&g, &plain), "the probe sums");
    let moved = moved.unwrap();
    let got = (0..N).map(|i| {
        (0..NB)
            .map(|b| C64::new(moved[b * N + i].re.into(), moved[b * N + i].im.into()))
            .sum::<C64>()
    });
    let num = sums.num.to_host_vec(&g.stream).unwrap();
    let want: Vec<C64> = (num[..N].iter())
        .map(|z| C64::new(z.re.into(), z.im.into()))
        .collect();
    let err = rel(got, &want);
    eprintln!("summed moves against the probe sums: {err:.3e}");
    assert!(err <= 2e-6, "summed moves against the probe sums: {err:e}");
}
