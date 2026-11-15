//! GPU BH through a multislice object against the CPU's `f64` [`Bh`] from the same start: two
//! and three slices under each of the far field, the angular spectrum and the Fresnel
//! transform forward and back, and three slices of two object modes, without a position and
//! with one that samples the object; and its refusal of a position that moves the probe.
//! Frames are on the fused passes' smallest side, two incoherent probe modes, 30 Fermat
//! positions in stacks of at most 8, so a batch of 12 ends in a stack of 4.

use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{
    AddrTable, C64, Counts, Geometry, Object, Probe, Scan, Stack, required_obj_shape,
};
use ptyrs::engine::{
    Bh, BhParams, BhStep, Feature, Features, GpuOptions, GpuPtycho, Observed, Optics, Sampling,
    SliceWaves, Wavefield, llk, shift,
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
    psi_f16: false,
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

/// Counts of the test object split over the slices of `optics`, in `nobj` incoherent modes,
/// each past the first the conjugate of the one before at half its amplitude; and a start off
/// it: unit modulus and a random phase in -1/4..1/4 rad per slice and mode, the probe's
/// defocus cut by a third.
fn case(optics: Optics, npos: usize, nobj: usize) -> Case {
    let (frame, pixel) = ([FRAME, FRAME], optics.pixel());
    let raw = ptyrs::base::sim::fermat(npos, FRAME as f64 / 8.0 * PIXEL);
    let positions = raw.pynx_preprocess(pixel);
    let shape = required_obj_shape(&positions, pixel, frame, 16);
    let addr = AddrTable::new(&positions, pixel, frame, shape, NPROBE, nobj).unwrap();
    let whole = ptyrs::base::sim::object(shape);
    let weights: &[f64] = match optics.nslice() {
        2 => &[0.6, 0.4],
        _ => &[0.5, 0.3, 0.2],
    };
    assert_eq!(weights.len(), optics.nslice());
    let truth: Vec<Object> = weights
        .iter()
        .map(|&w| {
            let first: Vec<C64> = whole.as_slice().iter().map(|c| c.powf(w)).collect();
            let data = (0..nobj)
                .flat_map(|o| {
                    let (a, conj) = (0.5f64.powi(o as i32), o % 2 == 1);
                    first
                        .iter()
                        .map(move |&c| a * if conj { c.conj() } else { c })
                })
                .collect();
            Object::new(Stack::from_vec([nobj, shape[0], shape[1]], data).unwrap())
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

    /// The CPU's normalised Poisson LLK of `objects` and `probe`, with `offsets` under
    /// [`Sampling::FullObject`].
    fn llk(&self, objects: &[Object], probe: &Probe, offsets: Option<&[[f64; 2]]>) -> f64 {
        let (nframe, nmode, views) = (self.addr.nframe(), self.addr.nmode(), self.addr.views());
        let sampling = Sampling::FullObject;
        let waves =
            SliceWaves::with_sampling(&self.optics, views, objects, probe, offsets, sampling);
        let mut psi = waves.psi().to_vec();
        self.optics.exit().forward(&mut psi);
        let llk = llk(&self.obs, 0..nframe, nmode, &psi).normalised(self.obs.nb_obs());
        llk.poisson
    }
}

fn params(batch_size: usize) -> BhParams {
    BhParams {
        batch_size,
        probe_rho: 0.01,
        position_rho: 0.1,
        seed: 7,
    }
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

/// The worst relative error of the GPU's steps; a CPU 0 (`β` in steepest descent) must be an
/// exact GPU 0.
fn step_err(got: &[BhStep], want: &[BhStep]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.frames, w.frames);
        for (a, b) in [
            (g.beta, w.beta),
            (g.alpha, w.alpha),
            (g.top, w.top),
            (g.bottom, w.bottom),
        ] {
            let e = if b == 0.0 {
                a.abs()
            } else {
                (a - b).abs() / b.abs()
            };
            worst = worst_of([worst, e]);
        }
    }
    worst
}

fn bits(s: &Stack<C64>) -> Vec<[u64; 2]> {
    s.as_slice()
        .iter()
        .map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect()
}

/// Offsets drawn about their mean, up to 0.4 pixel per axis.
fn drawn_offsets(npos: usize) -> Vec<[f64; 2]> {
    let mut d: Vec<[f64; 2]> = (0..npos)
        .map(|f| {
            let f = f as f64;
            [0.4 * (1.7 * f + 0.3).sin(), 0.4 * (2.3 * f).cos()]
        })
        .collect();
    for axis in 0..2 {
        let mean = d.iter().map(|d| d[axis]).sum::<f64>() / npos as f64;
        d.iter_mut().for_each(|d| d[axis] -= mean);
    }
    d
}

/// `max |Δa − Δb| / max |Δb|` of the offsets' changes from `start`; 0 without a position.
fn offsets_rel(a: &Features, b: &Features, start: &Features) -> f64 {
    let (Some(a), Some(b), Some(s)) = (&a.position, &b.position, &start.position) else {
        assert_eq!(a.position.is_some(), b.position.is_some());
        return 0.0;
    };
    let delta = |d: &[[f64; 2]]| -> Vec<f64> {
        d.iter()
            .zip(&s.values)
            .flat_map(|(d, s)| [d[0] - s[0], d[1] - s[1]])
            .collect()
    };
    let (a, b) = (delta(&a.values), delta(&b.values));
    let d = worst_of(a.iter().zip(&b).map(|(a, b)| (a - b).abs()));
    d / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
}

/// After [`NEPOCH`] epochs from a fresh GPU run, eagerly and from graphs, the last epoch's
/// steps, every slice, the probe, the LLK the GPU records after it and, with `position`, the
/// offsets' changes are the CPU's, for `propagator` after each of `nslices` slices of `nobj`
/// modes: in one batch of every frame (CG from the second epoch) and in batches of 12, the
/// last of 6. A `position` is drawn offsets refined under [`Sampling::FullObject`].
fn matches_cpu(propagator: &str, nslices: &[usize], nobj: usize, position: bool) {
    const NEPOCH: usize = 3;
    let features = Features {
        position: position.then(|| Feature::refined(drawn_offsets(NPOS))),
        ..Features::default()
    };
    for &nslice in nslices {
        let c = case(optics(propagator, nslice), NPOS, nobj);
        for batch_size in [NPOS, 12] {
            let p = params(batch_size);
            let mut bh = Bh::new(p)
                .unwrap()
                .with_sampling(Sampling::FullObject)
                .with_features(features.clone())
                .unwrap();
            let (mut objects, mut probe) = (c.objects.clone(), c.probe.clone());
            let start = c.llk(
                &objects,
                &probe,
                features.position.as_ref().map(|d| &d.values[..]),
            );
            let mut steps = Vec::new();
            for _ in 0..NEPOCH {
                steps = bh.epoch(&c.addr, &c.obs, &c.optics, &mut objects, &mut probe);
            }
            let llk = c.llk(
                &objects,
                &probe,
                bh.features().position.as_ref().map(|d| &d.values[..]),
            );
            for graph in [false, true] {
                let mut gpu = c.gpu(GpuOptions { graph, ..OPTIONS });
                if position {
                    gpu.set_features(&features).unwrap();
                    gpu.set_sampling(Sampling::FullObject).unwrap();
                }
                gpu.bh(&p, NEPOCH, 1).unwrap();
                let got = gpu.objects().unwrap();
                assert_eq!(got.len(), nslice);
                let recorded = gpu.history().last().unwrap().llk.poisson;
                let errs = [
                    step_err(&gpu.bh_steps().unwrap(), &steps),
                    worst_of(got.iter().zip(&objects).map(|(g, w)| rel_l2(g, w))),
                    rel_l2(&gpu.probe().unwrap(), &probe),
                    (recorded - llk).abs() / llk,
                    offsets_rel(&gpu.features().unwrap(), bh.features(), &features),
                ];
                let what = format!(
                    "{propagator} {nslice} slices {nobj} modes position {position} \
                     batch {batch_size} graph {graph}"
                );
                eprintln!(
                    "{what}: step {:.2e} objects {:.2e} probe {:.2e} llk {:.2e} offsets {:.2e}, \
                     llk {start:.4e} -> {llk:.4e}",
                    errs[0], errs[1], errs[2], errs[3], errs[4]
                );
                // The worst are 7.1e-6 (Fresnel back, two slices), 1.8e-7 (the same in batches
                // of 12), 8.5e-8 (far field, two object modes) and 2.3e-6 (Fresnel back, three
                // slices). With a position: 1.3e-5 (far field, three slices), 3.0e-7 (Fresnel
                // back, two slices, batches of 12), 9.4e-8, 1.6e-6 (far field, three slices of
                // two modes) and offsets 1.9e-5 (Fresnel, three slices).
                let bounds = [
                    ("step", 5e-5),
                    ("objects", 1e-6),
                    ("probe", 5e-7),
                    ("llk", 2e-5),
                    ("offsets", 2e-4),
                ];
                for ((name, bound), e) in bounds.into_iter().zip(errs) {
                    assert!(e < bound, "{what} {name}: {e:e}");
                }
            }
        }
    }
}

/// Three slices in one batch end above their start, LLK 22.28 → 22.60, by BH's own step rule:
/// the first CG step (second epoch) takes pty-chi's `α = top / bottom` (`bh.py`) with no line
/// search, and along it the amplitude cost goes 6.9e5 → 1.9e6 where the quadratic model says
/// 1.3e5. The gradient and Hessian match finite differences along that step; one slice falls
/// every epoch, and three reach 5.6 by the sixth.
#[test]
fn sliced_bh_matches_cpu_in_the_far_field() {
    matches_cpu(PROPAGATORS[0], &[2, 3], 1, false);
}

#[test]
fn sliced_bh_matches_cpu_under_angular_spectrum() {
    matches_cpu(PROPAGATORS[1], &[2, 3], 1, false);
}

#[test]
fn sliced_bh_matches_cpu_under_fresnel() {
    matches_cpu(PROPAGATORS[2], &[2, 3], 1, false);
}

#[test]
fn sliced_bh_matches_cpu_under_fresnel_back() {
    matches_cpu(PROPAGATORS[3], &[2, 3], 1, false);
}

/// Two object modes through three slices, each view's wave its own through every slice.
#[test]
fn sliced_bh_with_object_modes_matches_cpu_in_the_far_field() {
    matches_cpu(PROPAGATORS[0], &[3], 2, false);
}

#[test]
fn sliced_bh_with_object_modes_matches_cpu_under_fresnel() {
    matches_cpu(PROPAGATORS[2], &[3], 2, false);
}

/// A position that samples the object: every slice's windows.
#[test]
fn sliced_full_object_bh_matches_cpu_in_the_far_field() {
    matches_cpu(PROPAGATORS[0], &[2, 3], 1, true);
}

#[test]
fn sliced_full_object_bh_matches_cpu_under_angular_spectrum() {
    matches_cpu(PROPAGATORS[1], &[2, 3], 1, true);
}

#[test]
fn sliced_full_object_bh_matches_cpu_under_fresnel() {
    matches_cpu(PROPAGATORS[2], &[2, 3], 1, true);
}

#[test]
fn sliced_full_object_bh_matches_cpu_under_fresnel_back() {
    matches_cpu(PROPAGATORS[3], &[2, 3], 1, true);
}

#[test]
fn sliced_full_object_bh_with_object_modes_matches_cpu_in_the_far_field() {
    matches_cpu(PROPAGATORS[0], &[3], 2, true);
}

#[test]
fn sliced_full_object_bh_with_object_modes_matches_cpu_under_fresnel() {
    matches_cpu(PROPAGATORS[2], &[3], 2, true);
}

/// BH with a position that moves the probe through more than one slice fails before it
/// launches anything, naming the cause and changing nothing.
#[test]
fn sliced_bh_refuses_the_position_model() {
    let mut gpu = case(optics("far field", 3), NPOS, 1).gpu(OPTIONS);
    let before: Vec<_> = gpu.objects().unwrap().iter().map(|o| bits(o)).collect();
    let probe = bits(&gpu.probe().unwrap());
    let moved = Features {
        position: Some(Feature::refined(drawn_offsets(NPOS))),
        ..Features::default()
    };
    gpu.set_features(&moved).unwrap();
    let p = params(12);
    let e = gpu.bh(&p, 1, 1).unwrap_err().to_string();
    let patch = "BH with the position model under Sampling::Patch runs one object slice, got 3";
    assert!(e.contains(patch), "{e}");

    let after: Vec<_> = gpu.objects().unwrap().iter().map(|o| bits(o)).collect();
    assert!(after == before, "a refused pass changed the object");
    assert!(
        bits(&gpu.probe().unwrap()) == probe,
        "a refused pass changed the probe"
    );
    assert!(gpu.history().is_empty());
    assert_eq!(gpu.cycle(), 0);
}
