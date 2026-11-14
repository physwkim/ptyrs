//! GPU LSQML through a multislice object against the CPU's `f64` [`Lsqml`] from the same
//! start: two and three slices under each of the far field, the angular spectrum and the
//! Fresnel transform forward and back, without and with a modelled position, for both costs;
//! and the LLK pass through the slices against the CPU's LLK of the same arrays.
//! Frames are on the fused passes' smallest side, two incoherent probe modes, 30 Fermat
//! positions in stacks of at most 8, so a batch of 12 ends in a stack of 4.

use std::f64::consts::PI;
use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{
    AddrTable, C64, Counts, Geometry, Object, Probe, Scan, Stack, required_obj_shape,
};
use ptyrs::engine::{
    ApParams, BhParams, Cost, Features, Fft2, GpuOptions, GpuPtycho, Llk, Lsqml, LsqmlParams,
    LsqmlStep, Observed, Optics, ProbeVariation, SliceWaves, Wavefield, llk, shift,
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
const FRAME: usize = FUSED_SIDES[0] as usize;
const PROPAGATORS: [&str; 4] = ["far field", "angular spectrum", "Fresnel", "Fresnel back"];

const OPTIONS: GpuOptions = GpuOptions {
    stack_size: Some(8),
    graph: true,
    llk_slots: 3,
    fused: true,
};

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA device 0")
}

/// A geometry whose far field has the object pixel [`PIXEL`].
fn geometry() -> Geometry {
    let detector = WAVELENGTH / (FRAME as f64 * PIXEL);
    Geometry::new(WAVELENGTH, 1.0, detector, [FRAME, FRAME]).unwrap()
}

fn wavefield(distance: f64) -> Wavefield {
    Wavefield::new([FRAME, FRAME], WAVELENGTH, [PIXEL, PIXEL], distance).unwrap()
}

/// `propagator` after `nslice` slices 20 µm, then 30 µm apart.
fn optics(propagator: &str, nslice: usize) -> Optics {
    let exit = match propagator {
        "far field" => Optics::far_field(&geometry()),
        "angular spectrum" => Optics::free_space(&wavefield(1e-4)),
        "Fresnel" => Optics::free_space(&wavefield(5e-3)),
        "Fresnel back" => Optics::free_space(&wavefield(-5e-3)),
        other => panic!("no propagator {other}"),
    };
    let gaps = [wavefield(2e-5), wavefield(3e-5)];
    exit.with_slices(&gaps[..nslice - 1]).unwrap()
}

struct Case {
    /// The frames and positions; of its geometry only the frame is the case's.
    scan: Scan,
    addr: AddrTable,
    obs: Observed,
    optics: Optics,
    /// The start, one object per slice.
    objects: Vec<Object>,
    probe: Probe,
}

/// Counts of the test object split over the slices of `optics`, and a start off it: unit
/// modulus and a random phase in -1/4..1/4 rad per slice, the probe's defocus cut by a third.
fn case(optics: Optics) -> Case {
    let (frame, pixel) = ([FRAME, FRAME], optics.pixel());
    let raw = ptyrs::base::sim::fermat(NPOS, FRAME as f64 / 8.0 * PIXEL);
    let positions = raw.pynx_preprocess(pixel);
    let shape = required_obj_shape(&positions, pixel, frame, 16);
    let addr = AddrTable::new(&positions, pixel, frame, shape, NPROBE, 1).unwrap();
    let whole = ptyrs::base::sim::object(shape);
    let weights: &[f64] = match optics.nslice() {
        2 => &[0.6, 0.4],
        _ => &[0.5, 0.3, 0.2],
    };
    assert_eq!(weights.len(), optics.nslice());
    let truth: Vec<Object> = weights
        .iter()
        .map(|&w| {
            let data = whole.as_slice().iter().map(|c| c.powf(w)).collect();
            Object::new(Stack::from_vec(whole.dims(), data).unwrap())
        })
        .collect();
    let mut probe = ptyrs::base::sim::probe(frame, NPROBE).unwrap();
    probe
        .as_mut_slice()
        .iter_mut()
        .for_each(|v| *v *= PHOTONS.sqrt());

    let mut psi = SliceWaves::new(&optics, addr.views(), &truth, &probe, None)
        .psi()
        .to_vec();
    optics.exit().forward(&mut psi);
    let (n, nmode, nframe) = (FRAME * FRAME, addr.nmode(), addr.nframe());
    let mut counts = vec![0.0; nframe * n];
    for (f, block) in psi.chunks_exact(nmode * n).enumerate() {
        for j in 0..n {
            let icalc: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
            counts[f * n + shift(j, FRAME, FRAME)] = icalc.round();
        }
    }
    let counts = Counts::from_f64([nframe, FRAME, FRAME], &counts, None).unwrap();

    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let objects = truth
        .iter()
        .map(|t| {
            let data = t
                .as_slice()
                .iter()
                .map(|_| C64::from_polar(1.0, 0.5 * (uniform() - 0.5)))
                .collect();
            Object::new(Stack::from_vec(t.dims(), data).unwrap())
        })
        .collect();
    let sigma = FRAME as f64 / 12.0;
    for m in 0..probe.nmode() {
        for (i, v) in probe.plane_mut(m).iter_mut().enumerate() {
            let u = ((i % FRAME) as f64 - FRAME as f64 / 2.0) / sigma;
            let w = ((i / FRAME) as f64 - FRAME as f64 / 2.0) / sigma;
            *v *= C64::from_polar(1.0, -0.5 * (u * u + w * w));
        }
    }
    Case {
        obs: Observed::from_counts(&counts),
        scan: Scan::new(geometry(), raw, counts).unwrap(),
        addr,
        optics,
        objects,
        probe,
    }
}

impl Case {
    fn gpu(&self, options: GpuOptions) -> GpuPtycho {
        let probe = self.probe.clone();
        GpuPtycho::with_slices(
            &ctx(),
            &self.scan,
            &self.optics,
            &self.objects,
            probe,
            options,
        )
        .unwrap()
    }
}

impl Case {
    /// The CPU's normalised LLK of `objects` and `probe`, the probe of each frame moved by
    /// its entry of `offsets` as LSQML's position model moves it: `conj(q) T(q p)`, `T` the
    /// Fourier translation and `q` the first slice's plane phase.
    fn llk(&self, objects: &[Object], probe: &Probe, offsets: Option<&[[f64; 2]]>) -> Llk {
        let (nframe, nmode) = (self.addr.nframe(), self.addr.nmode());
        let psi = match offsets {
            None => SliceWaves::new(&self.optics, self.addr.views(), objects, probe, None)
                .psi()
                .to_vec(),
            Some(offsets) => {
                let (fft, q) = (Fft2::new([FRAME, FRAME]), self.optics.slice_quad(0));
                let freq = |i: usize| {
                    (if i < FRAME / 2 {
                        i as f64
                    } else {
                        i as f64 - FRAME as f64
                    }) / FRAME as f64
                };
                let mut planes = Vec::with_capacity(nframe * probe.as_slice().len());
                for d in offsets {
                    let mut w = probe.as_slice().to_vec();
                    for plane in w.as_chunks_mut::<{ FRAME * FRAME }>().0 {
                        if let Some(q) = q {
                            plane.iter_mut().zip(q).for_each(|(p, q)| *p *= q);
                        }
                    }
                    fft.forward(&mut w);
                    for (j, w) in w.iter_mut().enumerate() {
                        let (fy, fx) = (freq(j % (FRAME * FRAME) / FRAME), freq(j % FRAME));
                        *w *= C64::from_polar(1.0, -2.0 * PI * (fy * d[0] + fx * d[1]));
                    }
                    fft.inverse(&mut w);
                    for plane in w.as_chunks_mut::<{ FRAME * FRAME }>().0 {
                        if let Some(q) = q {
                            plane.iter_mut().zip(q).for_each(|(p, q)| *p *= q.conj());
                        }
                    }
                    planes.extend(w);
                }
                let dims = [nframe * NPROBE, FRAME, FRAME];
                let moved = Probe::new(Stack::from_vec(dims, planes).unwrap());
                let views: Vec<_> = (self.addr.views().iter())
                    .map(|v| ptyrs::base::View {
                        probe: v.frame * NPROBE as u32 + v.probe,
                        ..*v
                    })
                    .collect();
                SliceWaves::new(&self.optics, &views, objects, &moved, Some(offsets))
                    .psi()
                    .to_vec()
            }
        };
        let mut psi = psi;
        self.optics.exit().forward(&mut psi);
        llk(&self.obs, 0..nframe, nmode, &psi).normalised(self.obs.nb_obs())
    }
}

/// The worst of the four LLK values' relative errors.
fn llk_rel(got: &Llk, want: &Llk) -> f64 {
    worst_of(
        [
            (got.poisson, want.poisson),
            (got.gaussian, want.gaussian),
            (got.euclidian, want.euclidian),
            (got.nb_photons_calc, want.nb_photons_calc),
        ]
        .map(|(g, w)| (g - w).abs() / w.abs()),
    )
}

fn llk_bits(l: &Llk) -> [u64; 4] {
    [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc].map(f64::to_bits)
}

/// `|a − b| / |b|` over the whole array.
fn rel_l2(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        e += (x - y).norm_sqr();
        t += y.norm_sqr();
    }
    (e / t).sqrt()
}

/// The worst `max |a − b| / max |b|` of a batch's sizes of `a_far`, `a_probe` and, over the
/// slices, `a_obj`; the frames equal.
fn steps_rel(got: &[LsqmlStep], want: &[LsqmlStep]) -> [f64; 3] {
    assert_eq!(got.len(), want.len());
    let rel = |a: &[f64], b: &[f64]| {
        assert_eq!(a.len(), b.len());
        let d = a.iter().zip(b).map(|(a, b)| (a - b).abs());
        worst_of(d) / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
    };
    let mut worst = [0.0f64; 3];
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.frames, w.frames);
        assert_eq!(g.a_obj.len(), w.a_obj.len());
        let slices = g.a_obj.iter().zip(&w.a_obj).map(|(a, b)| rel(a, b));
        let now = [
            rel(&g.a_far, &w.a_far),
            rel(&g.a_probe, &w.a_probe),
            worst_of(slices),
        ];
        for (w, n) in worst.iter_mut().zip(now) {
            *w = worst_of([*w, n]);
        }
    }
    worst
}

/// Offsets drawn about their mean, up to 0.4 pixel per axis.
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

fn moved(position: bool) -> Features {
    Features {
        position: position.then(drawn_offsets),
        ..Features::default()
    }
}

/// `max |Δa − Δb| / max |Δb|` of the offsets' changes from `start`'s; 0 without a position.
fn offsets_rel(a: &Features, b: &Features, start: &Features) -> f64 {
    let (Some(a), Some(b), Some(s)) = (&a.position, &b.position, &start.position) else {
        assert_eq!(a.position.is_some(), b.position.is_some());
        return 0.0;
    };
    let delta = |d: &[[f64; 2]]| -> Vec<f64> {
        d.iter()
            .zip(s)
            .flat_map(|(d, s)| [d[0] - s[0], d[1] - s[1]])
            .collect()
    };
    let (a, b) = (delta(a), delta(b));
    let d = worst_of(a.iter().zip(&b).map(|(a, b)| (a - b).abs()));
    d / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
}

/// After [`NEPOCH`] epochs (the preconditioner built, then updated) in batches of 12, the
/// last epoch's steps, every slice, the probe and the offsets' changes are the CPU's, for
/// `propagator` after two and three slices, without and with a position, for both costs.
fn matches_cpu(propagator: &str) {
    const NEPOCH: usize = 2;
    for nslice in [2, 3] {
        let c = case(optics(propagator, nslice));
        for (position, cost) in [false, true]
            .into_iter()
            .flat_map(|p| [Cost::Amplitude, Cost::Poisson].map(move |c| (p, c)))
        {
            let params = LsqmlParams {
                batch_size: 12,
                cost,
                seed: 7,
            };
            let features = moved(position);
            let (mut objects, mut probe) = (c.objects.clone(), c.probe.clone());
            let mut var = ProbeVariation::none(probe.plane_shape());
            let mut cpu = Lsqml::new(params)
                .unwrap()
                .with_features(features.clone())
                .unwrap();
            let mut want = Vec::new();
            for _ in 0..NEPOCH {
                want = cpu.epoch(
                    &c.addr,
                    &c.obs,
                    &c.optics,
                    &mut objects,
                    &mut probe,
                    &mut var,
                );
            }
            let mut gpu = c.gpu(OPTIONS);
            gpu.set_features(&features).unwrap();
            gpu.lsqml(&params, NEPOCH, 0).unwrap();
            let got = gpu.objects().unwrap();
            assert_eq!(got.len(), nslice);
            let slices = got.iter().zip(&objects).map(|(g, w)| rel_l2(g, w));
            let [a_far, a_probe, a_obj] = steps_rel(&gpu.lsqml_steps().unwrap(), &want);
            let errs = [
                a_far,
                a_probe,
                a_obj,
                worst_of(slices),
                rel_l2(&gpu.probe().unwrap(), &probe),
                offsets_rel(&gpu.features().unwrap(), cpu.features(), &features),
            ];
            let what = format!("{propagator} {nslice} slices {cost:?} position {position}");
            eprintln!(
                "{what}: a_far {:.2e} a_probe {:.2e} a_obj {:.2e} objects {:.2e} probe {:.2e} \
                 offsets {:.2e}",
                errs[0], errs[1], errs[2], errs[3], errs[4], errs[5]
            );
            // Each bound is 5 times the worst of the 32 runs or more: `a_far` 2.8e-5 and the
            // offsets 1.9e-5 under the Fresnel transform back, three slices, Poisson, a position.
            let bounds = [
                ("a_far", 2e-4),
                ("a_probe", 5e-5),
                ("a_obj", 5e-5),
                ("objects", 2e-6),
                ("probe", 5e-5),
                ("offsets", 2e-4),
            ];
            for ((name, bound), e) in bounds.into_iter().zip(errs) {
                assert!(e < bound, "{what} {name}: {e:e}");
            }
        }
    }
}

#[test]
fn sliced_lsqml_matches_cpu_in_the_far_field() {
    matches_cpu(PROPAGATORS[0]);
}

#[test]
fn sliced_lsqml_matches_cpu_under_angular_spectrum() {
    matches_cpu(PROPAGATORS[1]);
}

#[test]
fn sliced_lsqml_matches_cpu_under_fresnel() {
    matches_cpu(PROPAGATORS[2]);
}

#[test]
fn sliced_lsqml_matches_cpu_under_fresnel_back() {
    matches_cpu(PROPAGATORS[3]);
}

/// The LLK pass through two and three slices gives the CPU's LLK of the same arrays under
/// every propagator, without and with a position: of the start, and of what an LSQML epoch
/// left, whose record is the pass's own. Graphs and eager launches leave the same bits.
#[test]
fn sliced_llk_matches_cpu() {
    let params = LsqmlParams {
        batch_size: 12,
        cost: Cost::Poisson,
        seed: 7,
    };
    for (propagator, nslice, position) in PROPAGATORS.into_iter().flat_map(|p| {
        [2, 3]
            .into_iter()
            .flat_map(move |n| [false, true].map(move |m| (p, n, m)))
    }) {
        let c = case(optics(propagator, nslice));
        let features = moved(position);
        let run = |graph| {
            let mut gpu = c.gpu(GpuOptions { graph, ..OPTIONS });
            gpu.set_features(&features).unwrap();
            let start = gpu.llk().unwrap();
            gpu.lsqml(&params, 2, 1).unwrap();
            let records: Vec<Llk> = gpu.history().iter().map(|r| r.llk).collect();
            assert_eq!(records.len(), 2);
            let end = gpu.llk().unwrap();
            let (objects, probe) = (gpu.objects().unwrap(), gpu.probe().unwrap());
            let offsets = gpu.features().unwrap().position;
            let want = c.llk(&objects, &probe, offsets.as_deref());
            (start, records, end, want)
        };
        let (start, records, end, want) = run(true);
        let what = format!("{propagator} {nslice} slices position {position}");
        assert!(
            llk_bits(&records[1]) == llk_bits(&end),
            "{what}: the epoch's record is not the pass's"
        );
        let first = c.llk(&c.objects, &c.probe, features.position.as_deref());
        let errs = [llk_rel(&start, &first), llk_rel(&end, &want)];
        eprintln!(
            "{what}: start {:.2e} end {:.2e}, poisson {:.4e} -> {:.4e}",
            errs[0], errs[1], start.poisson, end.poisson
        );
        // 5 times the worst of the 16 or more: 4.3e-6 at the start under angular spectrum,
        // three slices, a position.
        assert!(worst_of(errs) < 5e-5, "{what}: {errs:?}");
        let eager = run(false);
        let all = |r: &(Llk, Vec<Llk>, Llk, Llk)| {
            [vec![r.0], r.1.clone(), vec![r.2]]
                .concat()
                .iter()
                .map(llk_bits)
                .collect::<Vec<_>>()
        };
        assert!(
            all(&(start, records, end, want)) == all(&eager),
            "{what}: graphs and eager launches differ"
        );
    }
}

/// M1's launches take the LLK through the slices where the propagator has no plane stage:
/// the far field and a Fresnel transform forward.
#[test]
fn sliced_llk_matches_cpu_without_the_fused_passes() {
    for propagator in ["far field", "Fresnel"] {
        let c = case(optics(propagator, 3));
        let want = c.llk(&c.objects, &c.probe, None);
        for graph in [true, false] {
            let options = GpuOptions {
                graph,
                fused: false,
                ..OPTIONS
            };
            let err = llk_rel(&c.gpu(options).llk().unwrap(), &want);
            eprintln!("{propagator} graph {graph}: {err:.2e}");
            // The worst is 1.9e-6, under the Fresnel transform.
            assert!(err < 2e-5, "{propagator} graph {graph}: {err:e}");
        }
    }
}

fn bits(s: &Stack<C64>) -> Vec<[u64; 2]> {
    s.as_slice()
        .iter()
        .map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect()
}

/// Three slices with a position leave the same bits from graphs and eagerly, in batches
/// whose last stack is short (12 in stacks of 8) and whose every stack is (4 in stacks of 8).
#[test]
fn sliced_lsqml_is_bit_identical_with_graphs_and_without() {
    let c = case(optics("angular spectrum", 3));
    let features = moved(true);
    for batch_size in [12, 4] {
        let params = LsqmlParams {
            batch_size,
            cost: Cost::Poisson,
            seed: 7,
        };
        let run = |graph| {
            let mut g = c.gpu(GpuOptions { graph, ..OPTIONS });
            g.set_features(&features).unwrap();
            g.lsqml(&params, 3, 0).unwrap();
            let f64s = |v: &[f64]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let steps: Vec<_> = g
                .lsqml_steps()
                .unwrap()
                .into_iter()
                .map(|s| {
                    let sizes = [s.a_far, s.a_obj.concat(), s.a_probe].concat();
                    (s.frames, f64s(&sizes))
                })
                .collect();
            let objects: Vec<_> = g.objects().unwrap().iter().map(|o| bits(o)).collect();
            let offsets = f64s(&g.features().unwrap().position.unwrap().concat());
            (steps, objects, bits(&g.probe().unwrap()), offsets)
        };
        let (graphs, eager) = (run(true), run(false));
        assert!(graphs.0 == eager.0, "batch {batch_size}: steps differ");
        assert!(graphs.1 == eager.1, "batch {batch_size}: objects differ");
        assert!(graphs.2 == eager.2, "batch {batch_size}: probe differs");
        assert!(graphs.3 == eager.3, "batch {batch_size}: offsets differ");
    }
}

/// What the GPU does not run on a multislice object fails when the engine is built or before
/// the pass launches anything, naming the cause and changing nothing.
#[test]
fn gpu_refuses_what_it_cannot_run_on_slices() {
    let c = case(optics("angular spectrum", 3));
    let build = |optics: &Optics, objects: &[Object]| {
        GpuPtycho::with_slices(&ctx(), &c.scan, optics, objects, c.probe.clone(), OPTIONS)
            .err()
            .unwrap()
            .to_string()
    };

    let e = build(&c.optics, &c.objects[..2]);
    assert!(e.contains("2 object slices for optics of 3"), "{e}");
    let e = build(&optics("angular spectrum", 2), &c.objects);
    assert!(e.contains("3 object slices for optics of 2"), "{e}");

    let [nmode, ny, nx] = c.objects[0].dims();
    let ones = |dims: [usize; 3]| {
        let n = dims.iter().product();
        Object::new(Stack::from_vec(dims, vec![C64::new(1.0, 0.0); n]).unwrap())
    };
    let mut other = c.objects.clone();
    other[2] = ones([nmode, ny + 2, nx]);
    let e = build(&c.optics, &other);
    assert!(e.contains("differs from the first's"), "{e}");
    let two = vec![ones([2, ny, nx]); 3];
    let e = build(&c.optics, &two);
    assert!(e.contains("a multislice object of one mode, got 2"), "{e}");

    let mut gpu = case(optics("far field", 3)).gpu(OPTIONS);
    let before: Vec<_> = gpu.objects().unwrap().iter().map(|o| bits(o)).collect();
    let probe = bits(&gpu.probe().unwrap());
    let bh = BhParams {
        batch_size: 12,
        probe_rho: 0.01,
        position_rho: 0.1,
        seed: 7,
    };
    let only = "only LSQML and the LLK pass run a multislice object";
    let e = gpu.ap(&ApParams::default(), 1).unwrap_err().to_string();
    assert!(
        e.contains("Ap runs one object slice, got 3") && e.contains(only),
        "{e}"
    );
    let e = gpu.dm(&Default::default(), 1).unwrap_err().to_string();
    assert!(e.contains(only), "{e}");
    let e = gpu.raar(&Default::default(), 1).unwrap_err().to_string();
    assert!(e.contains(only), "{e}");
    let e = gpu.rpie(&Default::default(), 1).unwrap_err().to_string();
    assert!(e.contains(only), "{e}");
    let e = gpu.bh(&bh, 1, 0).unwrap_err().to_string();
    assert!(e.contains(only), "{e}");
    let e = gpu.ml(&Default::default(), 1).unwrap_err().to_string();
    assert!(e.contains(only), "{e}");
    let e = gpu.ml_grads().unwrap_err().to_string();
    assert!(e.contains(only), "{e}");

    let after: Vec<_> = gpu.objects().unwrap().iter().map(|o| bits(o)).collect();
    assert!(after == before, "a refused pass changed the object");
    assert!(
        bits(&gpu.probe().unwrap()) == probe,
        "a refused pass changed the probe"
    );
    assert!(gpu.history().is_empty());
    assert_eq!(gpu.cycle(), 0);
}
