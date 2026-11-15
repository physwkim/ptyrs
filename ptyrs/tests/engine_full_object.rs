//! AP's, DM's and RAAR's object update under `Sampling::FullObject`, which solves the normal
//! equations of the object's windows, on counts made through that model at drawn fractional
//! offsets; at zero and whole-pixel offsets it is the blend of the update without offsets.
//! One propagator: the solve is in the object plane.

use std::slice;
use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{
    AddrTable, C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack, required_obj_shape,
    sim,
};
use ptyrs::engine::{
    ApParams, DmParams, Features, GpuOptions, GpuPtycho, Optics, Ptycho, RaarParams, Sampling,
    SliceWaves, Wavefield, shift,
};
use ptyrs::gpu::FUSED_SIDES;

const NPOS: usize = 30;
const NPROBE: usize = 2;
const WAVELENGTH: f64 = 1e-10;
const PIXEL: f64 = 50e-9;
/// Photons the probe sends through a frame.
const PHOTONS: f64 = 2e5;
/// Fresnel.
const DISTANCE: f64 = 5e-3;

const AP: ApParams = ApParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    calc_llk: 1,
    zero_phase_ramp: false,
};
const DM: DmParams = DmParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    alpha: 0.02,
    loop_obj_probe: 1,
    calc_llk: 1,
    zero_phase_ramp: false,
};
const RAAR: RaarParams = RaarParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    beta: 0.9,
    loop_obj_probe: 1,
    calc_llk: 1,
    zero_phase_ramp: false,
};

fn frame() -> usize {
    FUSED_SIDES[0] as usize
}

fn free_space() -> Optics {
    let field = Wavefield::new([frame(); 2], WAVELENGTH, [PIXEL; 2], DISTANCE).unwrap();
    Optics::free_space(&field)
}

fn fermat() -> Positions {
    sim::fermat(NPOS, frame() as f64 / 8.0 * PIXEL)
}

/// Offsets within ±0.4 pixel, of mean 0.
fn drawn_offsets() -> Vec<[f64; 2]> {
    let mut d: Vec<[f64; 2]> = (0..NPOS)
        .map(|f| {
            let f = f as f64;
            [0.4 * (1.7 * f + 0.3).sin(), 0.4 * (2.3 * f).cos()]
        })
        .collect();
    for axis in 0..2 {
        let mean = d.iter().map(|d| d[axis]).sum::<f64>() / NPOS as f64;
        d.iter_mut().for_each(|d| d[axis] -= mean);
    }
    d
}

/// Whole-pixel offsets of mean 0, so that positions moved by them keep their mean.
fn whole_offsets() -> Vec<[f64; 2]> {
    let k = [[2.0, -1.0], [-1.0, 2.0], [-1.0, -1.0]];
    (0..NPOS).map(|f| k[f % 3]).collect()
}

fn positioned(offsets: &[[f64; 2]]) -> Features {
    Features {
        position: Some(offsets.to_vec()),
        ..Features::default()
    }
}

/// The smallest object holding every window of each of `raws`.
fn object_shape(raws: &[&Positions]) -> [usize; 2] {
    let pixel = [PIXEL; 2];
    raws.iter()
        .map(|r| required_obj_shape(&r.pynx_preprocess(pixel), pixel, [frame(); 2], 16))
        .fold([0, 0], |a, s| [a[0].max(s[0]), a[1].max(s[1])])
}

fn addr(raw: &Positions, shape: [usize; 2], nobj: usize) -> AddrTable {
    let (pixel, positions) = ([PIXEL; 2], raw.pynx_preprocess([PIXEL; 2]));
    AddrTable::new(&positions, pixel, [frame(); 2], shape, NPROBE, nobj).unwrap()
}

/// Object mode `o`'s amplitude.
fn mode_amplitude(o: usize) -> f64 {
    0.5f64.powi(o as i32)
}

struct Case {
    scan: Scan,
    /// Object mode `o` is the test object times [`mode_amplitude`], conjugated when `o` is odd.
    truth: (Object, Probe),
    /// Each object mode at [`mode_amplitude`] with a random phase in -1/4..1/4 rad, the truth
    /// probe's defocus cut by a third.
    start: (Object, Probe),
}

impl Case {
    /// Counts rounded from the truth's intensities under `Sampling::FullObject`, the windows
    /// at `offsets` from the corners of `raw`, in an object of `nobj` modes of `shape`.
    fn new(raw: Positions, shape: [usize; 2], offsets: &[[f64; 2]], nobj: usize) -> Self {
        let frame = frame();
        let optics = free_space();
        let addr = addr(&raw, shape, nobj);
        let first = sim::object(shape);
        let data = (0..nobj)
            .flat_map(|o| {
                let (a, conj) = (mode_amplitude(o), o % 2 == 1);
                (first.as_slice().iter()).map(move |&c| a * if conj { c.conj() } else { c })
            })
            .collect();
        let truth = Object::new(Stack::from_vec([nobj, shape[0], shape[1]], data).unwrap());
        let mut probe = sim::probe([frame; 2], NPROBE).unwrap();
        (probe.as_mut_slice().iter_mut()).for_each(|v| *v *= PHOTONS.sqrt());
        let (objects, full) = (slice::from_ref(&truth), Sampling::FullObject);
        let waves =
            SliceWaves::with_sampling(&optics, addr.views(), objects, &probe, Some(offsets), full);
        let mut psi = waves.psi().to_vec();
        optics.exit().forward(&mut psi);
        let (n, nmode, nframe) = (frame * frame, addr.nmode(), addr.nframe());
        let mut counts = vec![0.0; nframe * n];
        for (f, block) in psi.chunks_exact(nmode * n).enumerate() {
            for j in 0..n {
                let icalc: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                counts[f * n + shift(j, frame, frame)] = icalc.round();
            }
        }
        let counts = Counts::from_f64([nframe, frame, frame], &counts, None).unwrap();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let data = (0..truth.as_slice().len())
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                let a = mode_amplitude(i / (shape[0] * shape[1]));
                C64::from_polar(a, 0.5 * (u - 0.5))
            })
            .collect();
        let object = Object::new(Stack::from_vec(truth.dims(), data).unwrap());
        let mut start = probe.clone();
        let sigma = frame as f64 / 12.0;
        for m in 0..start.nmode() {
            for (i, v) in start.plane_mut(m).iter_mut().enumerate() {
                let u = ((i % frame) as f64 - frame as f64 / 2.0) / sigma;
                let w = ((i / frame) as f64 - frame as f64 / 2.0) / sigma;
                *v *= C64::from_polar(1.0, -0.5 * (u * u + w * w));
            }
        }
        let geometry = Geometry::new(WAVELENGTH, 1.0, PIXEL, [frame; 2]).unwrap();
        Self {
            scan: Scan::new(geometry, raw, counts).unwrap(),
            truth: (truth, probe),
            start: (object, start),
        }
    }

    /// A CPU engine on `scan` from `from`, under `Sampling::FullObject` and `features`.
    fn cpu(scan: &Scan, from: &(Object, Probe), features: &Features) -> Ptycho {
        let (o, p) = from.clone();
        let mut cpu = Ptycho::with_optics(scan, free_space(), o, p).unwrap();
        cpu.set_sampling(Sampling::FullObject);
        cpu.set_features(features.clone()).unwrap();
        cpu
    }

    /// [`Self::cpu`]'s on the GPU, `graph` capturing its cycles.
    fn gpu(
        ctx: &Arc<CudaContext>,
        scan: &Scan,
        from: &(Object, Probe),
        features: &Features,
        graph: bool,
    ) -> GpuPtycho {
        let (o, p) = from.clone();
        let options = GpuOptions {
            stack_size: Some(8),
            graph,
            llk_slots: 3,
            fused: true,
            psi_f16: false,
        };
        let mut gpu = GpuPtycho::with_optics(ctx, scan, &free_space(), o, p, options).unwrap();
        gpu.set_sampling(Sampling::FullObject).unwrap();
        gpu.set_features(features).unwrap();
        gpu
    }
}

/// What a run ends on: its LLK records, the LLK of the arrays it leaves, those arrays and
/// its offsets.
#[derive(PartialEq)]
struct Ended {
    records: Vec<f64>,
    end: f64,
    object: Object,
    probe: Probe,
    offsets: Option<Vec<[f64; 2]>>,
}

/// AP, DM and RAAR on the CPU or the GPU.
trait Engine {
    fn run(&mut self, algorithm: &str, ncycle: usize);
    fn hold(&mut self, features: &Features);
    fn ended(&mut self) -> Ended;

    /// `ncycle` cycles of `algorithm`, or with `held` that many calls of one cycle, each
    /// after setting `held`: every call starts from the held offsets, and DM and RAAR from the
    /// estimate's exit waves.
    fn cycles(&mut self, algorithm: &str, ncycle: usize, held: Option<&Features>) -> Ended {
        match held {
            None => self.run(algorithm, ncycle),
            Some(f) => (0..ncycle).for_each(|_| {
                self.hold(f);
                self.run(algorithm, 1)
            }),
        }
        self.ended()
    }
}

impl Engine for Ptycho {
    fn run(&mut self, algorithm: &str, ncycle: usize) {
        match algorithm {
            "AP" => self.ap(&AP, ncycle),
            "DM" => self.dm(&DM, ncycle),
            "RAAR" => self.raar(&RAAR, ncycle),
            a => panic!("{a}"),
        }
    }

    fn hold(&mut self, features: &Features) {
        self.set_features(features.clone()).unwrap();
    }

    fn ended(&mut self) -> Ended {
        Ended {
            records: self.history().iter().map(|r| r.llk.poisson).collect(),
            end: self.llk().poisson,
            object: self.object().clone(),
            probe: self.probe().clone(),
            offsets: self.features().position.clone(),
        }
    }
}

impl Engine for GpuPtycho {
    fn run(&mut self, algorithm: &str, ncycle: usize) {
        match algorithm {
            "AP" => self.ap(&AP, ncycle),
            "DM" => self.dm(&DM, ncycle),
            "RAAR" => self.raar(&RAAR, ncycle),
            a => panic!("{a}"),
        }
        .unwrap()
    }

    fn hold(&mut self, features: &Features) {
        self.set_features(features).unwrap();
    }

    fn ended(&mut self) -> Ended {
        Ended {
            records: self.history().iter().map(|r| r.llk.poisson).collect(),
            end: self.llk().unwrap().poisson,
            object: self.object().unwrap(),
            probe: self.probe().unwrap(),
            offsets: self.features().unwrap().position,
        }
    }
}

fn rel_l2(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        e += (x - y).norm_sqr();
        t += y.norm_sqr();
    }
    (e / t).sqrt()
}

impl Ended {
    /// From the truth (AP, RAAR), every LLK within 2% of the truth's `floor`; from the start
    /// (DM), every record below the last and the end a tenth of the first.
    fn check(&self, what: &str, from_truth: bool, floor: f64) {
        eprintln!(
            "{what}: floor {floor:.4}, records {:.4?}, end {:.4}",
            self.records, self.end
        );
        let all = || self.records.iter().chain([&self.end]);
        if from_truth {
            let worst = all().fold(0.0f64, |m, &v| m.max(v));
            assert!(worst < 1.02 * floor, "{what}: {worst} over {floor}");
        } else {
            for (i, w) in all().collect::<Vec<_>>().windows(2).enumerate() {
                assert!(
                    w[1] < w[0],
                    "{what} after cycle {i}: {} from {}",
                    w[1],
                    w[0]
                );
            }
            let first = self.records[0];
            assert!(self.end < first / 10.0, "{what}: {} from {first}", self.end);
        }
    }

    /// The worst relative LLK record, the relative end LLK, and the object's, the probe's and
    /// the offsets' (over the largest) distances from `want`'s, as named by [`ERRORS`].
    fn errors(&self, want: &Ended) -> [f64; 5] {
        assert_eq!(self.records.len(), want.records.len());
        let rel = |g: f64, w: f64| (g - w).abs() / w;
        let llk = (self.records.iter())
            .zip(&want.records)
            .fold(0.0f64, |m, (&g, &w)| m.max(rel(g, w)));
        let offsets = match (&self.offsets, &want.offsets) {
            (Some(g), Some(w)) => {
                let pairs = || g.iter().flatten().zip(w.iter().flatten());
                let worst = pairs().fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
                worst / w.iter().flatten().fold(0.0f64, |m, w| m.max(w.abs()))
            }
            _ => 0.0,
        };
        [
            llk,
            rel(self.end, want.end),
            rel_l2(&self.object, &want.object),
            rel_l2(&self.probe, &want.probe),
            offsets,
        ]
    }
}

const ERRORS: [&str; 5] = ["llk", "end", "object", "probe", "offsets"];

/// Each of [`Ended::errors`]'s `errs` within its `tol`; `None` leaves it unchecked.
fn assert_close(what: &str, errs: [f64; 5], tol: [Option<f64>; 5]) {
    let shown: Vec<String> = (ERRORS.iter().zip(errs))
        .map(|(n, e)| format!("{n} {e:.2e}"))
        .collect();
    eprintln!("{what}: {}", shown.join(", "));
    for ((name, e), tol) in ERRORS.iter().zip(errs).zip(tol) {
        assert!(tol.is_none_or(|t| e < t), "{what} {name}: {e:e}");
    }
}

/// [`solve_keeps_the_truth_and_descends`] in one object mode.
#[test]
fn solve_keeps_the_truth_and_descends_in_one_object_mode() {
    solve_keeps_the_truth_and_descends(1);
}

/// [`solve_keeps_the_truth_and_descends`] in two object modes.
#[test]
fn solve_keeps_the_truth_and_descends_in_two_object_modes() {
    solve_keeps_the_truth_and_descends(2);
}

/// On counts of the model at drawn offsets, in `nobj` object modes: AP from the truth with the
/// offsets held and RAAR from the truth stay at the truth's LLK, and DM from the start
/// descends, on the CPU and the GPU; the GPU's graphs replay its eager run bit for bit and
/// agree with the CPU. Each frame's views cover every probe mode on every object mode, where
/// the GPU's weight of a frame, `Σ_p |P_p|²`, is the CPU's `|W|²` of each of its views. A test
/// per `nobj`, so the two run side by side: the CPU's object solve is most of each.
fn solve_keeps_the_truth_and_descends(nobj: usize) {
    const NCYCLE: usize = 8;
    const TOL: [Option<f64>; 5] = [Some(1e-5), Some(1e-5), Some(1e-4), Some(1e-5), Some(3e-6)];
    let (raw, offsets) = (fermat(), drawn_offsets());
    let drawn = positioned(&offsets);
    let ctx = CudaContext::new(0).unwrap();
    let shape = object_shape(&[&raw]);
    let c = Case::new(raw, shape, &offsets, nobj);
    let floor = Case::cpu(&c.scan, &c.truth, &drawn).llk().poisson;
    // DM and RAAR carry their exit waves across cycles, which a held call restarts.
    for (algorithm, from_truth, held) in [
        ("AP", true, true),
        ("RAAR", true, false),
        ("DM", false, false),
    ] {
        let what = format!("{algorithm} nobj {nobj}");
        let from = [&c.start, &c.truth][usize::from(from_truth)];
        let held = held.then_some(&drawn);
        let want = Case::cpu(&c.scan, from, &drawn).cycles(algorithm, NCYCLE, held);
        want.check(&format!("{what} CPU"), from_truth, floor);
        let [eager, graphs] = [false, true].map(|graph| {
            let mut gpu = Case::gpu(&ctx, &c.scan, from, &drawn, graph);
            gpu.cycles(algorithm, NCYCLE, held)
        });
        assert!(graphs == eager, "{what}: graphs replay the eager run");
        eager.check(&format!("{what} GPU"), from_truth, floor);
        assert_close(&format!("{what} GPU against CPU"), eager.errors(&want), TOL);
    }
}

/// At zero offsets and at whole-pixel offsets from corners that many pixels back, held, the
/// update is the blend of the same counts without offsets at the moved corners: every LLK
/// record, the object and the probe, on the CPU and the GPU. The end LLK is not: the last
/// call's position step moves the offsets.
#[test]
fn whole_pixel_offsets_are_the_blend() {
    const NCYCLE: usize = 3;
    const TOL_CPU: [Option<f64>; 5] = [Some(1e-12), None, Some(1e-12), Some(1e-12), None];
    const TOL_GPU: [Option<f64>; 5] = [Some(1e-5), None, Some(1e-4), Some(1e-5), None];
    let (raw, whole) = (fermat(), whole_offsets());
    let moved = |a: usize, v: &[f64]| -> Vec<f64> {
        (v.iter().zip(&whole))
            .map(|(v, k)| v + k[a] * PIXEL)
            .collect()
    };
    let at = Positions::new(moved(0, raw.y()), moved(1, raw.x())).unwrap();
    let shape = object_shape(&[&raw, &at]);
    for (v, w) in addr(&raw, shape, 1)
        .views()
        .iter()
        .zip(addr(&at, shape, 1).views())
    {
        let k = whole[v.frame as usize];
        let corner = [v.oy as f64 + k[0], v.ox as f64 + k[1]];
        assert_eq!(corner, [w.oy as f64, w.ox as f64], "view {v:?}");
    }
    let c = Case::new(raw, shape, &whole, 1);
    let at = Scan::new(c.scan.geometry, at, c.scan.counts.clone()).unwrap();
    let none = Features::default();
    let runs = [
        ("zero", &at, positioned(&[[0.0; 2]; NPOS])),
        ("whole", &c.scan, positioned(&whole)),
    ];
    let ctx = CudaContext::new(0).unwrap();
    for algorithm in ["AP", "DM", "RAAR"] {
        let blend = Case::cpu(&at, &c.start, &none).cycles(algorithm, NCYCLE, Some(&none));
        let blend_gpu =
            Case::gpu(&ctx, &at, &c.start, &none, true).cycles(algorithm, NCYCLE, Some(&none));
        for (label, scan, features) in &runs {
            let got = Case::cpu(scan, &c.start, features).cycles(algorithm, NCYCLE, Some(features));
            let what = format!("{algorithm} CPU {label}");
            assert_close(&what, got.errors(&blend), TOL_CPU);
            let got = Case::gpu(&ctx, scan, &c.start, features, true).cycles(
                algorithm,
                NCYCLE,
                Some(features),
            );
            let what = format!("{algorithm} GPU {label}");
            assert_close(&what, got.errors(&blend_gpu), TOL_GPU);
        }
    }
}
