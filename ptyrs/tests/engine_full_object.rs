//! AP's, DM's and RAAR's object update under `Sampling::FullObject`, which solves the normal
//! equations of the object's windows, and rPIE's, on counts made through that model at drawn
//! fractional offsets, held or refined; at zero and whole-pixel offsets, held, it is the blend
//! of the update without offsets. One propagator: the solve is in the object plane.

use std::slice;
use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{
    AddrTable, C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack, required_obj_shape,
    sim,
};
use ptyrs::engine::{
    ApParams, DmParams, Feature, Features, GpuOptions, GpuPtycho, Optics, Ptycho, RaarParams,
    RpieParams, Sampling, SliceWaves, Wavefield, shift,
};
use ptyrs::gpu::FUSED_SIDES;

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

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
/// Four frames a step: with eight or more, rPIE diverges on these counts, with or without
/// offsets.
const RPIE: RpieParams = RpieParams {
    obj_step: Some(1.0),
    probe_step: Some(1.0),
    obj_alpha: 0.5,
    probe_alpha: 0.5,
    batch_size: 4,
    calc_llk: 1,
    seed: 7,
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

fn positioned(offsets: &[[f64; 2]], refine: bool) -> Features {
    Features {
        position: Some(Feature {
            values: offsets.to_vec(),
            refine,
        }),
        ..Features::default()
    }
}

/// `offsets` as the GPU keeps them, in f32.
fn in_f32(offsets: &[[f64; 2]]) -> Vec<[f64; 2]> {
    (offsets.iter())
        .map(|o| o.map(|v| v as f32 as f64))
        .collect()
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

/// AP, DM, RAAR and rPIE on the CPU or the GPU.
trait Engine {
    fn run(&mut self, algorithm: &str, ncycle: usize);
    fn ended(&mut self) -> Ended;

    /// `ncycle` cycles of `algorithm`, epochs of rPIE, in one call.
    fn cycles(&mut self, algorithm: &str, ncycle: usize) -> Ended {
        self.run(algorithm, ncycle);
        self.ended()
    }
}

impl Engine for Ptycho {
    fn run(&mut self, algorithm: &str, ncycle: usize) {
        match algorithm {
            "AP" => self.ap(&AP, ncycle),
            "DM" => self.dm(&DM, ncycle),
            "RAAR" => self.raar(&RAAR, ncycle),
            "rPIE" => self.rpie(&RPIE, ncycle).unwrap(),
            a => panic!("{a}"),
        }
    }

    fn ended(&mut self) -> Ended {
        Ended {
            records: self.history().iter().map(|r| r.llk.poisson).collect(),
            end: self.llk().poisson,
            object: self.object().clone(),
            probe: self.probe().clone(),
            offsets: self.features().position.clone().map(|p| p.values),
        }
    }
}

impl Engine for GpuPtycho {
    fn run(&mut self, algorithm: &str, ncycle: usize) {
        match algorithm {
            "AP" => self.ap(&AP, ncycle),
            "DM" => self.dm(&DM, ncycle),
            "RAAR" => self.raar(&RAAR, ncycle),
            "rPIE" => self.rpie(&RPIE, ncycle),
            a => panic!("{a}"),
        }
        .unwrap()
    }

    fn ended(&mut self) -> Ended {
        Ended {
            records: self.history().iter().map(|r| r.llk.poisson).collect(),
            end: self.llk().unwrap().poisson,
            object: self.object().unwrap(),
            probe: self.probe().unwrap(),
            offsets: self.features().unwrap().position.map(|p| p.values),
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

/// [`rel_l2`] with each pixel's squares weighted by `weight`'s.
fn weighted_l2(a: &Stack<C64>, b: &Stack<C64>, weight: &Stack<f64>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for ((x, y), w) in (a.as_slice().iter().zip(b.as_slice())).zip(weight.as_slice()) {
        e += w * (x - y).norm_sqr();
        t += w * y.norm_sqr();
    }
    (e / t).sqrt()
}

impl Ended {
    /// From the truth, whose LLK is `floor`, every LLK within 2% of it; from the start, every
    /// record below the last, the end not above the last record, which for rPIE's epochs is
    /// the end's, and a tenth of the first.
    fn check(&self, what: &str, floor: Option<f64>) {
        eprintln!(
            "{what}: floor {floor:.4?}, records {:.4?}, end {:.4}",
            self.records, self.end
        );
        if let Some(floor) = floor {
            let worst = worst_of(self.records.iter().chain([&self.end]).copied());
            assert!(worst < 1.02 * floor, "{what}: {worst} over {floor}");
        } else {
            for (i, w) in self.records.windows(2).enumerate() {
                assert!(
                    w[1] < w[0],
                    "{what} after cycle {i}: {} from {}",
                    w[1],
                    w[0]
                );
            }
            let (first, last) = (self.records[0], self.records[self.records.len() - 1]);
            assert!(self.end <= last, "{what}: {} from {last}", self.end);
            assert!(self.end < first / 10.0, "{what}: {} from {first}", self.end);
        }
    }

    /// The worst relative LLK record, the relative end LLK, and the object's, the probe's and
    /// the offsets' (over the largest) distances from `want`'s, as named by [`ERRORS`].
    fn errors(&self, want: &Ended) -> [f64; 5] {
        assert_eq!(self.records.len(), want.records.len());
        let rel = |g: f64, w: f64| (g - w).abs() / w;
        let llk = worst_of((self.records.iter().zip(&want.records)).map(|(&g, &w)| rel(g, w)));
        let offsets = match (&self.offsets, &want.offsets) {
            (Some(g), Some(w)) => {
                let pairs = || g.iter().flatten().zip(w.iter().flatten());
                worst_of(pairs().map(|(g, w)| (g - w).abs()))
                    / worst_of(pairs().map(|(_, w)| w.abs()))
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

    /// `Σ |P|²` of the probe this run ends on, over `scan`'s views at the offsets it ends on:
    /// the weight the counts put on each object pixel. Away from the views only the inertia
    /// holds the object, where the solve's `f32` errors grow a hundredfold.
    fn illumination(&self, scan: &Scan) -> Stack<f64> {
        let dims @ [nobj, ny, nx] = self.object.dims();
        let addr = addr(&scan.positions, [ny, nx], nobj);
        Sampling::FullObject.illumination(addr.views(), dims, &self.probe, self.offsets.as_deref())
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

/// `got`'s object within `tol` of `want`'s, weighted by `lit`, an [`Ended::illumination`]: the
/// band the counts see, which [`Ended::errors`]' whole plane leaves to the dark band's errors.
fn assert_lit(what: &str, got: &Ended, want: &Ended, lit: &Stack<f64>, tol: f64) {
    let e = weighted_l2(&got.object, &want.object, lit);
    eprintln!("{what}: lit object {e:.2e}");
    assert!(e < tol, "{what} lit object: {e:e}");
}

/// `ncycle` cycles of `algorithm` on `c` under `features`, from the truth, whose LLK is
/// `floor`, or without from the start, on the CPU and on the GPU eagerly and with graphs: each
/// passes [`Ended::check`], held offsets keep their values, the graphs replay the eager run
/// bit for bit and the GPU agrees with the CPU, the object over the whole plane and weighted by
/// [`Ended::illumination`].
fn on_cpu_and_gpu(
    c: &Case,
    ctx: &Arc<CudaContext>,
    (algorithm, ncycle): (&str, usize),
    features: &Features,
    floor: Option<f64>,
) {
    // Five-run worst: LLK 1.68e-6, end 7.28e-7, object 1.27e-5 over the whole plane (AP's in
    // two object modes; most of it the dark band's) and 1.46e-6 weighted by the CPU's
    // illumination (RAAR's refined in two object modes), probe 2.42e-6, offsets 2.44e-7.
    const TOL: [Option<f64>; 5] = [
        Some(5.1e-6),
        Some(2.2e-6),
        Some(3.9e-5),
        Some(7.3e-6),
        Some(7.4e-7),
    ];
    const TOL_LIT: f64 = 4.4e-6;
    let position = features.position.as_ref().unwrap();
    let nobj = c.truth.0.dims()[0];
    let what = format!("{algorithm} nobj {nobj} refine {}", position.refine);
    let from = if floor.is_some() { &c.truth } else { &c.start };
    let want = Case::cpu(&c.scan, from, features).cycles(algorithm, ncycle);
    let [eager, graphs] = [false, true]
        .map(|graph| Case::gpu(ctx, &c.scan, from, features, graph).cycles(algorithm, ncycle));
    if !position.refine {
        let kept = Some(&position.values);
        assert_eq!(
            want.offsets.as_ref(),
            kept,
            "{what}: the CPU keeps the offsets"
        );
        let kept = Some(in_f32(&position.values));
        assert_eq!(eager.offsets, kept, "{what}: the GPU keeps the offsets");
    }
    want.check(&format!("{what} CPU"), floor);
    assert!(graphs == eager, "{what}: graphs replay the eager run");
    eager.check(&format!("{what} GPU"), floor);
    let what = format!("{what} GPU against CPU");
    assert_close(&what, eager.errors(&want), TOL);
    assert_lit(&what, &eager, &want, &want.illumination(&c.scan), TOL_LIT);
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

/// On counts of the model at drawn offsets, in `nobj` object modes: AP and RAAR from the
/// truth stay at the truth's LLK and DM from the start descends, with the offsets held and
/// RAAR's and DM's also refined, by [`on_cpu_and_gpu`]. Each frame's views cover every probe
/// mode on every object mode, where the GPU's weight of a frame, `Σ_p |P_p|²`, is the CPU's
/// `|W|²` of each of its views. A test per `nobj`, so the two run side by side: the CPU's
/// object solve is most of each.
fn solve_keeps_the_truth_and_descends(nobj: usize) {
    const NCYCLE: usize = 8;
    let (raw, offsets) = (fermat(), drawn_offsets());
    let ctx = CudaContext::new(0).unwrap();
    let shape = object_shape(&[&raw]);
    let c = Case::new(raw, shape, &offsets, nobj);
    let floor = Case::cpu(&c.scan, &c.truth, &positioned(&offsets, false))
        .llk()
        .poisson;
    for (algorithm, from_truth, refine) in [
        ("AP", true, false),
        ("RAAR", true, false),
        ("RAAR", true, true),
        ("DM", false, false),
        ("DM", false, true),
    ] {
        let features = positioned(&offsets, refine);
        on_cpu_and_gpu(
            &c,
            &ctx,
            (algorithm, NCYCLE),
            &features,
            from_truth.then_some(floor),
        );
    }
}

/// On the counts of [`solve_keeps_the_truth_and_descends`] in one object mode, the only one
/// rPIE models: rPIE from the start descends with the offsets held, by [`on_cpu_and_gpu`],
/// in twice DM's cycles.
#[test]
fn rpie_descends_with_the_offsets_held() {
    const NEPOCH: usize = 16;
    let (raw, offsets) = (fermat(), drawn_offsets());
    let ctx = CudaContext::new(0).unwrap();
    let shape = object_shape(&[&raw]);
    let c = Case::new(raw, shape, &offsets, 1);
    on_cpu_and_gpu(
        &c,
        &ctx,
        ("rPIE", NEPOCH),
        &positioned(&offsets, false),
        None,
    );
}

/// At zero offsets and at whole-pixel offsets from corners that many pixels back, held, the
/// update is the blend of the same counts without offsets at the moved corners: every LLK
/// record, the end LLK, the object and the probe, on the CPU and the GPU, which keep the
/// offsets; the GPU's object also weighted by [`Ended::illumination`].
#[test]
fn whole_pixel_offsets_are_the_blend() {
    const NCYCLE: usize = 3;
    const TOL_CPU: [Option<f64>; 5] = [Some(1e-12), Some(1e-12), Some(1e-12), Some(1e-12), None];
    // The GPU's five-run worst: LLK 9.67e-7, end 8.65e-7, object 7.08e-6 over the whole plane
    // (RAAR's) and 1.02e-6 weighted by the blend's illumination (DM's), probe 9.43e-7.
    const TOL_GPU: [Option<f64>; 5] = [Some(3e-6), Some(2.6e-6), Some(2.2e-5), Some(2.9e-6), None];
    const TOL_GPU_LIT: f64 = 3.1e-6;
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
    let zero = [[0.0; 2]; NPOS];
    let runs = [("zero", &at, &zero[..]), ("whole", &c.scan, &whole[..])];
    let ctx = CudaContext::new(0).unwrap();
    for algorithm in ["AP", "DM", "RAAR", "rPIE"] {
        let blend = Case::cpu(&at, &c.start, &none).cycles(algorithm, NCYCLE);
        let blend_gpu = Case::gpu(&ctx, &at, &c.start, &none, true).cycles(algorithm, NCYCLE);
        let lit = blend_gpu.illumination(&at);
        for (label, scan, offsets) in runs {
            let held = positioned(offsets, false);
            let got = Case::cpu(scan, &c.start, &held).cycles(algorithm, NCYCLE);
            let what = format!("{algorithm} CPU {label}");
            assert_eq!(
                got.offsets.as_deref(),
                Some(offsets),
                "{what} keeps the offsets"
            );
            assert_close(&what, got.errors(&blend), TOL_CPU);
            let got = Case::gpu(&ctx, scan, &c.start, &held, true).cycles(algorithm, NCYCLE);
            let what = format!("{algorithm} GPU {label}");
            assert_eq!(
                got.offsets.as_deref(),
                Some(offsets),
                "{what} keeps the offsets"
            );
            assert_close(&what, got.errors(&blend_gpu), TOL_GPU);
            assert_lit(&what, &got, &blend_gpu, &lit, TOL_GPU_LIT);
        }
    }
}

/// With no object inertia and the offsets held, AP, DM and RAAR on the CPU and the GPU keep
/// the offsets; at zero and whole-pixel offsets they leave every pixel outside the windows at
/// its start, the GPU's at the start rounded to `f32`; at drawn offsets, whose windows sample
/// every pixel, every pixel stays finite. No LLK is due, so DM and RAAR balance nothing.
#[test]
fn zero_object_inertia_holds_the_unseen_pixels() {
    const NCYCLE: usize = 2;
    const AP0: ApParams = ApParams {
        obj_inertia: Some(0.0),
        calc_llk: 0,
        ..AP
    };
    const DM0: DmParams = DmParams {
        obj_inertia: Some(0.0),
        calc_llk: 0,
        ..DM
    };
    const RAAR0: RaarParams = RaarParams {
        obj_inertia: Some(0.0),
        calc_llk: 0,
        ..RAAR
    };
    let raw = fermat();
    let shape = object_shape(&[&raw]);
    let views = addr(&raw, shape, 1).views().to_vec();
    let ctx = CudaContext::new(0).unwrap();
    for (label, offsets, whole) in [
        ("zero", vec![[0.0; 2]; NPOS], true),
        ("whole", whole_offsets(), true),
        ("drawn", drawn_offsets(), false),
    ] {
        let mut seen = vec![false; shape[0] * shape[1]];
        for v in views.iter().filter(|_| whole) {
            let [dy, dx] = offsets[v.frame as usize].map(|d| d as i64);
            let (y0, x0) = (v.oy as i64 + dy, v.ox as i64 + dx);
            assert!(y0 >= 0 && x0 >= 0, "{label} view {v:?}");
            let (y0, x0) = (y0 as usize, x0 as usize);
            for y in y0..y0 + frame() {
                let row = &mut seen[y * shape[1]..(y + 1) * shape[1]];
                row[x0..x0 + frame()].fill(true);
            }
        }
        let c = Case::new(fermat(), shape, &offsets, 1);
        let held = positioned(&offsets, false);
        let check = |what: &str, got: &Object, start: &Object| {
            let it = got.as_slice().iter().zip(start.as_slice()).zip(&seen);
            for (j, ((g, s), &seen)) in it.enumerate() {
                assert!(
                    g.re.is_finite() && g.im.is_finite(),
                    "{what} pixel {j}: {g}"
                );
                assert!(!whole || seen || g == s, "{what} pixel {j}: {g} from {s}");
            }
        };
        let start = &c.start.0;
        let start32 = start.as_slice().iter().map(|v| {
            let [re, im] = [v.re, v.im].map(|x| f64::from(x as f32));
            C64::new(re, im)
        });
        let start32 = Object::new(Stack::from_vec(start.dims(), start32.collect()).unwrap());
        for algorithm in ["AP", "DM", "RAAR"] {
            let mut cpu = Case::cpu(&c.scan, &c.start, &held);
            let mut gpu = Case::gpu(&ctx, &c.scan, &c.start, &held, true);
            match algorithm {
                "AP" => (cpu.ap(&AP0, NCYCLE), gpu.ap(&AP0, NCYCLE)),
                "DM" => (cpu.dm(&DM0, NCYCLE), gpu.dm(&DM0, NCYCLE)),
                _ => (cpu.raar(&RAAR0, NCYCLE), gpu.raar(&RAAR0, NCYCLE)),
            }
            .1
            .unwrap();
            let what = format!("{algorithm} CPU {label}");
            let kept = cpu.features().position.as_ref().map(|p| &p.values);
            assert_eq!(kept, Some(&offsets), "{what} keeps the offsets");
            check(&what, cpu.object(), start);
            let what = format!("{algorithm} GPU {label}");
            let kept = gpu.features().unwrap().position.map(|p| p.values);
            assert_eq!(kept, Some(in_f32(&offsets)), "{what} keeps the offsets");
            check(&what, &gpu.object().unwrap(), &start32);
        }
    }
}
