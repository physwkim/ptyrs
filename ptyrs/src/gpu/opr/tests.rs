//! The OPR kernels against the CPU `ProbeVariation` on the same inputs, rounded to f32 for
//! both: `unique`'s mode-0 planes and `step`'s eigenmodes, weights and main-mode weights, on
//! even and odd plane sides with one and two eigenmodes, a one-frame batch, and the weights
//! or the main-mode weights held. The CPU runs two probe modes, of which the device takes
//! mode 0's planes. Every worst difference goes through `worst_of`, so a NaN fails its bound.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use crate::base::Feature;
use crate::base::{C64, Probe, Stack, View};
use crate::engine::ProbeVariation;
use crate::gpu::{C32, EpochOrder, Kernels, Opr, OprKernels, OprPlanes, ViewAddr, ViewDims, Views};
use crate::worst::worst_of;
use ptyrs_kernels::opr::OPR_SUMS;

use super::{Shared, SumsOf};

const NFRAME: usize = 9;
/// The scan frame at each epoch position.
const ORDER: [u32; NFRAME] = [4, 7, 0, 2, 8, 1, 5, 3, 6];
/// The CPU's probe modes.
const NP: usize = 2;

struct Gpu {
    stream: Arc<CudaStream>,
    ctx: ExecutionContext,
    opr: OprKernels,
    order: EpochOrder,
    /// The views `order` was set from.
    _views: Views,
}

fn gpu() -> Gpu {
    let cx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&cx).unwrap();
    let stream = cx.default_stream();
    let ctx = ExecutionContext::new(stream.clone());
    // Only the order's frame count matters; its views are the smallest valid ones.
    let dims = ViewDims {
        nframe: NFRAME as u32,
        nmode: 1,
        nprobe: 1,
        nobj: 1,
        frame: [2, 2],
        obj: [2, 2],
        quad_phase: 0.0,
    };
    let addr = [ViewAddr {
        oy: 0,
        ox: 0,
        fy: 0.0,
        fx: 0.0,
        probe: 0,
        obj: 0,
    }; NFRAME];
    let views = Views::new(&stream, dims, &addr).unwrap();
    let mut order = EpochOrder::new(&kernels, &stream, &views).unwrap();
    order.set(&ctx, &kernels, &views, &ORDER).unwrap();
    Gpu {
        opr: OprKernels::load(&kernels).unwrap(),
        stream,
        ctx,
        order,
        _views: views,
    }
}

/// Uniform values in `[-1, 1)`, each rounded to f32.
struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let u = 2.0 * ((self.0 >> 11) as f64 / (1u64 << 53) as f64) - 1.0;
        u as f32 as f64
    }

    fn c(&mut self, scale: f64) -> C64 {
        C64::new(
            (scale * self.next()) as f32 as f64,
            (scale * self.next()) as f32 as f64,
        )
    }

    fn stack(&mut self, dims: [usize; 3], scale: f64) -> Stack<C64> {
        let data = (0..dims.iter().product()).map(|_| self.c(scale)).collect();
        Stack::from_vec(dims, data).unwrap()
    }

    /// Values in `[lo, hi)`.
    fn range(&mut self, len: usize, lo: f64, hi: f64) -> Vec<f64> {
        let mid = (lo + hi) / 2.0;
        (0..len)
            .map(|_| (mid + (hi - mid) * self.next()) as f32 as f64)
            .collect()
    }
}

fn c32(z: &[C64]) -> Vec<C32> {
    z.iter()
        .map(|z| C32 {
            re: z.re as f32,
            im: z.im as f32,
        })
        .collect()
}

fn f32s(x: &[f64]) -> Vec<f32> {
    x.iter().map(|&x| x as f32).collect()
}

/// The device's `[nb, n]` planes of mode 0 from a CPU stack `[nb NP, ny, nx]`.
fn mode0<T: Copy>(s: &Stack<T>) -> Vec<T> {
    (0..s.len() / NP)
        .flat_map(|b| s.plane(b * NP).iter().copied())
        .collect()
}

/// `worst |got − want| / max |want|`.
fn rel(got: impl IntoIterator<Item = f64>, want: &[f64]) -> f64 {
    let scale = worst_of(want.iter().map(|w| w.abs()));
    worst_of(got.into_iter().zip(want).map(|(g, w)| (g - w).abs())) / scale
}

fn rel_c(got: &[C32], want: &[C64]) -> f64 {
    let scale = worst_of(want.iter().map(|w| w.norm()));
    let diff = got
        .iter()
        .zip(want)
        .map(|(g, w)| (C64::new(g.re as f64, g.im as f64) - w).norm());
    worst_of(diff) / scale
}

/// A case's main-mode weights.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Main {
    Absent,
    Refined,
    Held,
}

/// A scan of [`NFRAME`] frames with `k` eigenmodes of `ny × nx` planes, their weights refined
/// unless `held_weights`, and main-mode weights as `intensity`.
#[derive(Clone, Copy)]
struct Case {
    ny: usize,
    nx: usize,
    k: usize,
    held_weights: bool,
    intensity: Main,
}

struct Start {
    n: usize,
    probe: Probe,
    var: ProbeVariation,
}

impl Case {
    fn start(&self, noise: &mut Noise) -> Start {
        let n = self.ny * self.nx;
        // pty-chi's eigenmodes have mnorm √n, as the step leaves them.
        let eigen = noise.stack([self.k, self.ny, self.nx], (n as f64).sqrt());
        let signs = noise.range(NFRAME * self.k, -1.0, 1.0);
        let weights = noise
            .range(NFRAME * self.k, 0.05, 0.15)
            .iter()
            .zip(signs)
            .map(|(w, s)| w.copysign(s))
            .collect();
        let weights = Feature {
            values: weights,
            refine: !self.held_weights,
        };
        let mut var = ProbeVariation::new(eigen, weights);
        if self.intensity != Main::Absent {
            var = var.with_intensity(Feature {
                values: noise.range(NFRAME, 0.9, 1.1),
                refine: self.intensity == Main::Refined,
            });
        }
        Start {
            n,
            probe: Probe::new(noise.stack([NP, self.ny, self.nx], 1.0)),
            var,
        }
    }

    fn upload(&self, g: &Gpu, s: &Start) -> (Opr, DeviceBuffer<C32>) {
        let v = &s.var;
        let w = v.weights().as_ref().map(|w| f32s(w));
        let c = v.intensity().map(|c| c.as_ref().map(|c| f32s(c)));
        let opr = Opr::new(
            &g.stream,
            s.n,
            &c32(v.eigenmodes().as_slice()),
            w.as_ref().map(Vec::as_slice),
            c.as_ref().map(|c| c.as_ref().map(Vec::as_slice)),
            NFRAME as u32,
        )
        .unwrap();
        let probe = DeviceBuffer::from_host(&g.stream, &c32(s.probe.as_slice())).unwrap();
        (opr, probe)
    }
}

/// The scan frames at the epoch positions `p`.
fn frames(p: &Range<u32>) -> Vec<usize> {
    ORDER[p.start as usize..p.end as usize]
        .iter()
        .map(|&f| f as usize)
        .collect()
}

/// The device state after `step`: eigenmodes, weights and main-mode weights.
type After = (Vec<C32>, Vec<f32>, Option<Vec<f32>>);

fn download(g: &Gpu, opr: &Opr) -> After {
    let s = &g.stream;
    (
        opr.eigenmodes().to_host_vec(s).unwrap(),
        opr.weights().values.to_host_vec(s).unwrap(),
        opr.intensity().map(|c| c.values.to_host_vec(s).unwrap()),
    )
}

/// The batch `p` of `case` stepped on the CPU and on the device: the CPU's state, the
/// device's twice from the same start, and the start.
fn step(g: &Gpu, case: &Case, p: Range<u32>, seed: u64) -> (ProbeVariation, [After; 2], Start) {
    let mut noise = Noise::new(seed);
    let s = case.start(&mut noise);
    let nb = (p.end - p.start) as usize;
    let dims = [nb * NP, case.ny, case.nx];
    let n = s.n as f64;
    let d = noise.stack(dims, 0.1 * n.sqrt());
    let moved = noise.stack(dims, 0.1 * n.sqrt());
    let vv = Stack::from_vec(dims, noise.range(dims.iter().product(), 0.5, 1.5)).unwrap();
    let mut cpu = s.var.clone();
    cpu.step(&frames(&p), s.probe.plane(0), &moved, &d, &vv, NP);
    let gpu = [0, 1].map(|_| {
        let (mut opr, probe) = case.upload(g, &s);
        let up = |x: &[C64]| DeviceBuffer::from_host(&g.stream, &c32(x)).unwrap();
        let (dd, mut dm) = (up(&mode0(&d)), up(&mode0(&moved)));
        let dvv = DeviceBuffer::from_host(&g.stream, &f32s(&mode0(&vv))).unwrap();
        let planes = OprPlanes {
            d: &dd,
            vv: &dvv,
            moved: &mut dm,
        };
        g.opr
            .step(&g.ctx, &mut opr, &g.order, p.clone(), &probe, planes)
            .unwrap();
        download(g, &opr)
    });
    (cpu, gpu, s)
}

/// The relative worst differences of the device's eigenmodes, its weights' change and its
/// main-mode weights' change from the CPU's.
fn errors(cpu: &ProbeVariation, gpu: &After, start: &ProbeVariation) -> [f64; 3] {
    let delta = |now: &[f64], was: &[f64]| -> Vec<f64> {
        now.iter().zip(was).map(|(a, b)| a - b).collect()
    };
    let delta32 = |now: &[f32], was: &[f64]| -> Vec<f64> {
        now.iter().zip(was).map(|(&a, b)| a as f64 - b).collect()
    };
    let (u, w, c) = gpu;
    let dw = rel(
        delta32(w, &start.weights().values),
        &delta(&cpu.weights().values, &start.weights().values),
    );
    let dc = match (c, cpu.intensity(), start.intensity()) {
        (Some(c), Some(cpu), Some(was)) => {
            rel(delta32(c, &was.values), &delta(&cpu.values, &was.values))
        }
        (None, None, None) => 0.0,
        _ => panic!("main-mode weights on one side only"),
    };
    [rel_c(u, cpu.eigenmodes().as_slice()), dw, dc]
}

#[test]
fn opr_step_matches_cpu() {
    let g = gpu();
    let mut seed = 1;
    for (ny, nx) in [(8, 8), (7, 9)] {
        for k in [1, 2] {
            for intensity in [Main::Absent, Main::Refined] {
                let case = Case {
                    ny,
                    nx,
                    k,
                    held_weights: false,
                    intensity,
                };
                // Five frames from position 2: the order's frames 0, 2, 8, 1, 5.
                let (cpu, gpu, start) = step(&g, &case, 2..7, seed);
                seed += 1;
                let what = format!("{ny}x{nx}, k = {k}, intensity {intensity:?}");
                let [u, w, c] = errors(&cpu, &gpu[0], &start.var);
                println!("{what}: eigenmodes {u:.3e}, weight steps {w:.3e}, main {c:.3e}");
                assert!(u <= 1e-6, "{what}: eigenmodes {u:e}");
                assert!(w <= 5e-6, "{what}: weight steps {w:e}");
                assert!(c <= 5e-5, "{what}: main-mode weight steps {c:e}");
                assert_eq!(gpu[0], gpu[1], "{what}: two runs differ");
            }
        }
    }
}

#[test]
fn opr_one_frame_batch_moves_only_the_main_mode() {
    let g = gpu();
    for intensity in [Main::Absent, Main::Refined] {
        let case = Case {
            ny: 7,
            nx: 9,
            k: 2,
            held_weights: false,
            intensity,
        };
        let (cpu, gpu, start) = step(&g, &case, 4..5, 11);
        let v = &start.var;
        assert_eq!(cpu.eigenmodes(), v.eigenmodes());
        assert_eq!(cpu.weights(), v.weights());
        let (u, w, c) = &gpu[0];
        assert_eq!(*u, c32(v.eigenmodes().as_slice()), "eigenmodes moved");
        assert_eq!(*w, f32s(&v.weights().values), "weights moved");
        let [_, _, dc] = errors(&cpu, &gpu[0], v);
        println!("intensity {intensity:?}: main {dc:.3e}");
        assert!(dc <= 5e-5, "main-mode weight step {dc:e}");
        if let (Some(c), Some(was)) = (c, v.intensity()) {
            assert_ne!(*c, f32s(&was.values), "main-mode weights did not move");
        }
    }
}

/// Held main-mode weights keep their values on the device as on the CPU, and the eigenmodes
/// and weights step as with them refined.
#[test]
fn opr_held_main_mode_matches_cpu() {
    let g = gpu();
    for k in [1, 2] {
        let held = Case {
            ny: 7,
            nx: 9,
            k,
            held_weights: false,
            intensity: Main::Held,
        };
        let refined = Case {
            intensity: Main::Refined,
            ..held
        };
        let (cpu, gpu, start) = step(&g, &held, 2..7, 31);
        let (cpu_r, gpu_r, _) = step(&g, &refined, 2..7, 31);
        let was = start.var.intensity().unwrap();
        assert!(!was.refine);
        assert_eq!(
            cpu.intensity(),
            Some(was),
            "k = {k}: CPU main-mode weights moved"
        );
        let (u, w, c) = &gpu[0];
        assert_eq!(
            *c,
            Some(f32s(&was.values)),
            "k = {k}: device main-mode weights moved"
        );
        assert_ne!(
            gpu_r[0].2, *c,
            "k = {k}: refined main-mode weights did not move"
        );
        assert_eq!(
            (cpu.eigenmodes(), cpu.weights().values.as_slice()),
            (cpu_r.eigenmodes(), cpu_r.weights().values.as_slice()),
            "k = {k}: CPU"
        );
        assert_eq!((u, w), (&gpu_r[0].0, &gpu_r[0].1), "k = {k}: device");
        let [du, dw, _] = errors(&cpu, &gpu[0], &start.var);
        println!("k = {k} held: eigenmodes {du:.3e}, weight steps {dw:.3e}");
        assert!(du <= 1e-6, "k = {k}: eigenmodes {du:e}");
        assert!(dw <= 5e-6, "k = {k}: weight steps {dw:e}");
        assert_eq!(gpu[0], gpu[1], "k = {k}: two runs differ");
    }
}

/// Held weights keep their values on the device as on the CPU, and the eigenmodes step bit
/// for bit as with them refined: the weights' step comes after each eigenmode's move and the
/// deflation sums alike either way.
#[test]
fn opr_held_weights_match_cpu() {
    let g = gpu();
    for k in [1, 2] {
        for intensity in [Main::Absent, Main::Refined] {
            let held = Case {
                ny: 7,
                nx: 9,
                k,
                held_weights: true,
                intensity,
            };
            let refined = Case {
                held_weights: false,
                ..held
            };
            let what = format!("k = {k}, intensity {intensity:?}");
            let (cpu, gpu, start) = step(&g, &held, 2..7, 41);
            let (_, gpu_r, _) = step(&g, &refined, 2..7, 41);
            let was = start.var.weights();
            assert!(!was.refine);
            assert_eq!(cpu.weights(), was, "{what}: CPU weights moved");
            let (u, w, c) = &gpu[0];
            assert_eq!(*w, f32s(&was.values), "{what}: device weights moved");
            assert_ne!(gpu_r[0].1, *w, "{what}: refined weights did not move");
            assert_eq!((u, c), (&gpu_r[0].0, &gpu_r[0].2), "{what}: device");
            let [du, _, dc] = errors(&cpu, &gpu[0], &start.var);
            println!("{what} held weights: eigenmodes {du:.3e}, main {dc:.3e}");
            assert!(du <= 1e-6, "{what}: eigenmodes {du:e}");
            assert!(dc <= 5e-5, "{what}: main-mode weight steps {dc:e}");
            assert_eq!(gpu[0], gpu[1], "{what}: two runs differ");
        }
    }
}

#[test]
fn opr_unique_matches_cpu() {
    let g = gpu();
    let mut seed = 21;
    for (ny, nx) in [(8, 8), (7, 9)] {
        for k in [1, 2] {
            for intensity in [Main::Absent, Main::Refined] {
                let case = Case {
                    ny,
                    nx,
                    k,
                    held_weights: false,
                    intensity,
                };
                let s = case.start(&mut Noise::new(seed));
                seed += 1;
                let p = 3..8;
                let views: Vec<View> = frames(&p)
                    .iter()
                    .flat_map(|&f| {
                        (0..NP as u32).map(move |m| View {
                            frame: f as u32,
                            mode: m,
                            probe: m,
                            obj: 0,
                            oy: 0,
                            ox: 0,
                        })
                    })
                    .collect();
                let (_, planes) = s.var.unique(&views, &s.probe);
                let want = mode0(&planes);
                let (opr, probe) = case.upload(&g, &s);
                let mut out = DeviceBuffer::<C32>::zeroed(&g.stream, want.len()).unwrap();
                g.opr
                    .unique(&g.ctx, &opr, &g.order, p, &probe, &mut out)
                    .unwrap();
                let got = out.to_host_vec(&g.stream).unwrap();
                let err = rel_c(&got, &want);
                let what = format!("{ny}x{nx}, k = {k}, intensity {intensity:?}");
                println!("{what}: unique {err:.3e}");
                assert!(err <= 5e-7, "{what}: unique {err:e}");
            }
        }
    }
}

/// The deflation's coefficients are the residue sums of `opr_frame_sums` totalled in chunk
/// order and over `n²`, bit for bit, on planes of three chunks.
#[test]
fn opr_deflate_coef_totals_chunks_in_order() {
    let g = gpu();
    let case = Case {
        ny: 45,
        nx: 101,
        k: 2,
        held_weights: false,
        intensity: Main::Absent,
    };
    let mut noise = Noise::new(31);
    let s = case.start(&mut noise);
    let (mut opr, _) = case.upload(&g, &s);
    let p = 2..7;
    let span = opr.span(&g.order, &p).unwrap();
    let (n, nb, nchunk) = (s.n, span.nb as usize, span.nchunk as usize);
    assert_eq!(nchunk, 3);
    let sums = f32s(&noise.range(opr.sums.len(), -1.0, 1.0));
    opr.sums = DeviceBuffer::from_host(&g.stream, &sums).unwrap();
    let moved = noise.stack([nb, case.ny, case.nx], 1.0);
    let mut moved = DeviceBuffer::from_host(&g.stream, &c32(moved.as_slice())).unwrap();
    g.opr.deflate(&g.ctx, &opr, span, 0, &mut moved).unwrap();
    let coef = opr.deflate_coef.to_host_vec(&g.stream).unwrap();
    let nn = n as f32 * n as f32;
    for b in 0..nb {
        for (h, j) in [(0, 2), (1, 3)] {
            let k = OPR_SUMS as usize;
            let total = (0..nchunk).fold(0.0f32, |t, c| t + sums[k * (b * nchunk + c) + j]);
            assert_eq!(coef[2 * b + h], total / nn, "frame {b}, part {h}");
        }
    }
}

/// The deflation's residue sums (slots 2 and 3) and coefficients, bit for bit, whether the
/// weights' `d` and `vv` are summed with them (refined weights) or left out (held).
#[test]
fn opr_deflation_is_bit_identical_held_and_refined() {
    let g = gpu();
    for (ny, nx) in [(45, 101), (64, 64), (7, 9)] {
        let case = Case {
            ny,
            nx,
            k: 2,
            held_weights: false,
            intensity: Main::Absent,
        };
        let mut noise = Noise::new(77);
        let s = case.start(&mut noise);
        let (opr, _) = case.upload(&g, &s);
        let span = opr.span(&g.order, &(2..7)).unwrap();
        let (n, nb) = (s.n, span.nb as usize);
        let live = OPR_SUMS as usize * nb * span.nchunk as usize;
        let r = noise.stack([nb, ny, nx], 1.0);
        let r = DeviceBuffer::from_host(&g.stream, &c32(r.as_slice())).unwrap();
        let d = noise.stack([nb, ny, nx], 1.0);
        let d = DeviceBuffer::from_host(&g.stream, &c32(d.as_slice())).unwrap();
        let vv = f32s(&noise.range(nb * n, 0.0, 2.0));
        let vv = DeviceBuffer::from_host(&g.stream, &vv).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let [refined, held] = [Some((&d, &vv)), None].map(|dv| {
            let of = SumsOf {
                a: Shared::Eigenmode(0),
                dv,
                r: Some(&r),
            };
            g.opr.frame_sums(&g.ctx, &opr, span, of).unwrap();
            let sums = opr.sums.to_host_vec(&g.stream).unwrap();
            let mut moved = DeviceBuffer::<C32>::zeroed(&g.stream, nb * n).unwrap();
            g.opr.deflate(&g.ctx, &opr, span, 0, &mut moved).unwrap();
            let coef = opr.deflate_coef.to_host_vec(&g.stream).unwrap();
            let (per_chunk, _) = sums[..live].as_chunks::<{ OPR_SUMS as usize }>();
            let weights = per_chunk.iter().any(|c| c[0] != 0.0 || c[1] != 0.0);
            let slots: Vec<f32> = per_chunk.iter().flat_map(|c| [c[2], c[3]]).collect();
            (weights, bits(&slots), bits(&coef[..2 * nb]))
        });
        let what = format!("{ny}x{nx}");
        assert!(refined.0 && !held.0, "{what}: the weights' sums");
        assert_eq!(refined.1, held.1, "{what}: slots 2 and 3");
        assert_eq!(refined.2, held.2, "{what}: deflation coefficients");
    }
}
