//! The near-field and multislice cases GPU LSQML is held to, on the fused passes' smallest
//! side: an angular-spectrum propagator with and without an evanescent band, a Fresnel
//! transform forward and back, and three object slices under angular spectrum, each run by the
//! CPU's `f64` [`Lsqml`] from a start off the truth its counts come from. Two incoherent probe
//! modes, 30 Fermat positions, counts the truth's intensities rounded.

use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{Geometry, Scan};
use ptyrs::engine::{
    ApParams, DmParams, GpuOptions, GpuPtycho, MlParams, PlaneStage, RaarParams, RpieParams,
    Sampling,
};

use ptyrs::base::{AddrTable, C64, Counts, Object, Probe, Stack, required_obj_shape};
use ptyrs::engine::{
    Cost, Feature, Features, Lsqml, LsqmlParams, LsqmlStep, Observed, Optics, ProbeVariation,
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
/// A wavelength whose angular spectrum over [`PIXEL`] drops the corners of the frequency
/// grid, `λ / pixel` the CPU's near case's 5/3.
const BAND_WAVELENGTH: f64 = PIXEL / 0.6;
/// Photons the probe sends through a frame.
const PHOTONS: f64 = 2e5;

struct Case {
    name: &'static str,
    /// The frames and positions; of its geometry only the frame is the case's.
    scan: Scan,
    addr: AddrTable,
    obs: Observed,
    optics: Optics,
    /// The start, one object per slice.
    objects: Vec<Object>,
    probe: Probe,
}

impl Case {
    /// The normalised Poisson LLK of `objects` and `probe` over every frame.
    fn llk(&self, objects: &[Object], probe: &Probe) -> f64 {
        let psi = propagated(&self.addr, &self.optics, objects, probe);
        let (nframe, nmode) = (self.addr.nframe(), self.addr.nmode());
        llk(&self.obs, 0..nframe, nmode, &psi)
            .normalised(self.obs.nb_obs())
            .poisson
    }

    /// `nepoch` epochs of the CPU's LSQML from the start: the last epoch's steps, the
    /// arrays, and the LLK after each epoch.
    fn cpu(
        &self,
        params: LsqmlParams,
        nepoch: usize,
    ) -> (Vec<LsqmlStep>, Vec<Object>, Probe, Vec<f64>) {
        let (mut objects, mut probe) = (self.objects.clone(), self.probe.clone());
        let mut var = ProbeVariation::none(probe.plane_shape());
        let mut lsqml = Lsqml::new(params).unwrap();
        let (mut steps, mut trace) = (Vec::new(), Vec::new());
        for _ in 0..nepoch {
            steps = lsqml.epoch(
                &self.addr,
                &self.obs,
                &self.optics,
                &mut objects,
                &mut probe,
                &mut var,
            );
            trace.push(self.llk(&objects, &probe));
        }
        (steps, objects, probe, trace)
    }
}

/// Detector-plane exit waves of every view.
fn propagated(addr: &AddrTable, optics: &Optics, objects: &[Object], probe: &Probe) -> Vec<C64> {
    let mut psi = SliceWaves::new(optics, addr.views(), objects, probe, None)
        .psi()
        .to_vec();
    optics.exit().forward(&mut psi);
    psi
}

fn wavefield(frame: usize, distance: f64) -> Wavefield {
    Wavefield::new([frame, frame], WAVELENGTH, [PIXEL, PIXEL], distance).unwrap()
}

/// `optics` over `weights.len()` slices, slice `s` the test object to the power `weights[s]`.
fn case(name: &'static str, frame: usize, optics: Optics, weights: &[f64]) -> Case {
    assert_eq!(optics.nslice(), weights.len());
    let pixel = [PIXEL, PIXEL];
    let raw = ptyrs::base::sim::fermat(NPOS, frame as f64 / 8.0 * PIXEL);
    let positions = raw.pynx_preprocess(pixel);
    let shape = required_obj_shape(&positions, pixel, [frame, frame], 16);
    let addr = AddrTable::new(&positions, pixel, [frame, frame], shape, NPROBE, 1).unwrap();
    let whole = ptyrs::base::sim::object(shape);
    let truth: Vec<Object> = weights
        .iter()
        .map(|&w| {
            let data = whole.as_slice().iter().map(|c| c.powf(w)).collect();
            Object::new(Stack::from_vec(whole.dims(), data).unwrap())
        })
        .collect();
    let mut probe = ptyrs::base::sim::probe([frame, frame], NPROBE).unwrap();
    probe
        .as_mut_slice()
        .iter_mut()
        .for_each(|v| *v *= PHOTONS.sqrt());

    let psi = propagated(&addr, &optics, &truth, &probe);
    let (n, nmode, nframe) = (frame * frame, addr.nmode(), addr.nframe());
    let mut counts = vec![0.0; nframe * n];
    for (f, block) in psi.chunks_exact(nmode * n).enumerate() {
        for j in 0..n {
            let icalc: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
            counts[f * n + shift(j, frame, frame)] = icalc.round();
        }
    }
    let counts = Counts::from_f64([nframe, frame, frame], &counts, None).unwrap();

    // The start: unit modulus and a random phase in -1/4..1/4 rad per slice, the probe's
    // defocus cut by a third.
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
    let sigma = frame as f64 / 12.0;
    for m in 0..probe.nmode() {
        for (i, v) in probe.plane_mut(m).iter_mut().enumerate() {
            let u = ((i % frame) as f64 - frame as f64 / 2.0) / sigma;
            let w = ((i / frame) as f64 - frame as f64 / 2.0) / sigma;
            *v *= C64::from_polar(1.0, -0.5 * (u * u + w * w));
        }
    }
    let geometry = Geometry::new(WAVELENGTH, 1.0, PIXEL, [frame, frame]).unwrap();
    Case {
        name,
        addr,
        obs: Observed::from_counts(&counts),
        scan: Scan::new(geometry, raw, counts).unwrap(),
        optics,
        objects,
        probe,
    }
}

fn cases() -> Vec<Case> {
    let frame = FUSED_SIDES[0] as usize;
    let gaps = [wavefield(frame, 2e-5), wavefield(frame, 3e-5)];
    let band = Wavefield::new(
        [frame, frame],
        BAND_WAVELENGTH,
        [PIXEL, PIXEL],
        6.0 * BAND_WAVELENGTH,
    )
    .unwrap();
    vec![
        case(
            "angular spectrum",
            frame,
            Optics::free_space(&wavefield(frame, 1e-4)),
            &[1.0],
        ),
        case("evanescent band", frame, Optics::free_space(&band), &[1.0]),
        case(
            "Fresnel",
            frame,
            Optics::free_space(&wavefield(frame, 5e-3)),
            &[1.0],
        ),
        case(
            "Fresnel back",
            frame,
            Optics::free_space(&wavefield(frame, -5e-3)),
            &[1.0],
        ),
        case(
            "three slices",
            frame,
            Optics::free_space(&wavefield(frame, 1e-4))
                .with_slices(&gaps)
                .unwrap(),
            &[0.5, 0.3, 0.2],
        ),
    ]
}

/// Every case has the one-transform form the GPU takes, and the CPU's LSQML lowers its LLK
/// from the start in each of the first epochs, for both costs.
#[test]
fn cpu_lsqml_descends_on_every_case() {
    for c in cases() {
        c.optics.fourier_form().unwrap();
        for cost in [Cost::Amplitude, Cost::Poisson] {
            let params = LsqmlParams {
                batch_size: 12,
                cost,
                seed: 7,
            };
            let start = c.llk(&c.objects, &c.probe);
            let (_, _, _, trace) = c.cpu(params, 6);
            eprintln!("{} {cost:?}: start {start:.4e}, epochs {trace:.4?}", c.name);
            let mut last = start;
            for (e, &v) in trace.iter().enumerate() {
                assert!(
                    v < last,
                    "{} {cost:?} epoch {e}: {v:e} after {last:e}",
                    c.name
                );
                last = v;
            }
        }
    }
}

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

impl Case {
    fn gpu(&self, optics: &Optics, options: GpuOptions) -> Result<GpuPtycho, String> {
        let (object, probe) = (self.objects[0].clone(), self.probe.clone());
        GpuPtycho::with_optics(&ctx(), &self.scan, optics, object, probe, options)
            .map_err(|e| e.to_string())
    }
}

/// The GPU's LLK of each one-slice start is the CPU's, with graphs and without.
#[test]
fn gpu_llk_is_the_cpus_under_every_propagator() {
    for c in cases().iter().filter(|c| c.optics.nslice() == 1) {
        let want = c.llk(&c.objects, &c.probe);
        for graph in [true, false] {
            let mut gpu = c.gpu(&c.optics, GpuOptions { graph, ..OPTIONS }).unwrap();
            let got = gpu.llk().unwrap().poisson;
            let err = (got - want).abs() / want;
            eprintln!(
                "{} graph {graph}: LLK {got:.6e}, CPU {want:.6e}, {err:.2e}",
                c.name
            );
            assert!(err < 1e-5, "{} graph {graph}: {got:e} vs {want:e}", c.name);
        }
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

/// The worst `max |a − b| / max |b|` of a batch's sizes of one kind; the frames equal.
// Per element a probe size of 5.7e-3 in a batch of median 0.69 read 8.9e-5 for an absolute
// error of 5e-7, the other sizes' own, where the CPU from an f32 start moves it 1.3e-5.
fn steps_rel(got: &[LsqmlStep], want: &[LsqmlStep]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.frames, w.frames);
        for (a, b) in [
            (&g.a_far, &w.a_far),
            (&g.a_obj[0], &w.a_obj[0]),
            (&g.a_probe, &w.a_probe),
        ] {
            assert_eq!(a.len(), b.len());
            let d = a.iter().zip(b).map(|(a, b)| (a - b).abs());
            let scale = b.iter().map(|b| b.abs()).fold(0.0, f64::max);
            worst = worst_of([worst, worst_of(d) / scale]);
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

/// GPU LSQML under every one-slice propagator against the CPU's from the same start, as
/// `engine_gpu_lsqml` holds the far field: the steps of each of the first epochs from a fresh
/// run without graphs, and with them the LLK after every epoch, the arrays and, modelling
/// the position, the offsets' changes if refined, or that both keep them if held. A position,
/// refined or held, enters as [`Sampling::Patch`], and once as [`Sampling::FullObject`].
#[test]
fn gpu_lsqml_matches_cpu_under_every_propagator() {
    const NEPOCH: usize = 12;
    const NSTEPS: usize = 3;
    for c in cases().iter().filter(|c| c.optics.nslice() == 1) {
        let refined = || Some(Feature::refined(drawn_offsets()));
        let held = || Some(Feature::held(drawn_offsets()));
        for (cost, batch_size, position, sampling) in [
            (Cost::Amplitude, 12, None, Sampling::Patch),
            (Cost::Poisson, 12, None, Sampling::Patch),
            (Cost::Poisson, NPOS, None, Sampling::Patch),
            (Cost::Amplitude, 4, refined(), Sampling::Patch),
            (Cost::Poisson, 12, refined(), Sampling::Patch),
            (Cost::Poisson, 12, refined(), Sampling::FullObject),
            (Cost::Amplitude, 4, held(), Sampling::Patch),
            (Cost::Poisson, 12, held(), Sampling::Patch),
            (Cost::Poisson, 12, held(), Sampling::FullObject),
        ] {
            let params = LsqmlParams {
                batch_size,
                cost,
                seed: 7,
            };
            let model = match &position {
                None => "none",
                Some(p) if p.refine => "refined",
                Some(_) => "held",
            };
            let features = Features {
                position,
                ..Features::default()
            };
            let (mut objects, mut probe) = (c.objects.clone(), c.probe.clone());
            let none = ProbeVariation::none(probe.plane_shape());
            let mut var = none.clone();
            let mut cpu = Lsqml::new(params)
                .unwrap()
                .with_features(features.clone())
                .unwrap()
                .with_sampling(sampling);
            let (mut steps, mut trace) = (0.0f64, Vec::new());
            for epoch in 0..NEPOCH {
                let want = cpu.epoch(
                    &c.addr,
                    &c.obs,
                    &c.optics,
                    &mut objects,
                    &mut probe,
                    &mut var,
                );
                trace.push(c.llk(&objects, &probe));
                if epoch < NSTEPS {
                    let options = GpuOptions {
                        graph: false,
                        ..OPTIONS
                    };
                    let mut gpu = c.gpu(&c.optics, options).unwrap();
                    gpu.set_features(&features).unwrap();
                    gpu.set_sampling(sampling).unwrap();
                    gpu.lsqml(&params, &mut none.clone(), epoch + 1, 0).unwrap();
                    steps = worst_of([steps, steps_rel(&gpu.lsqml_steps().unwrap(), &want)]);
                }
            }

            let mut gpu = c.gpu(&c.optics, OPTIONS).unwrap();
            gpu.set_features(&features).unwrap();
            gpu.set_sampling(sampling).unwrap();
            gpu.lsqml(&params, &mut none.clone(), NEPOCH, 1).unwrap();
            // The CPU's LLK here is the unmoved probe's.
            let llk = if features.position.is_some() {
                0.0
            } else {
                let got = gpu.history().iter().map(|r| r.llk.poisson);
                assert_eq!(got.len(), NEPOCH);
                worst_of(got.zip(&trace).map(|(g, w)| (g - w).abs() / w))
            };
            let label = format!(
                "{} {cost:?} batch {batch_size} position {model} {sampling:?}",
                c.name
            );
            let offsets = match (&features.position, cpu.features().position.as_ref()) {
                (Some(start), Some(want)) if !start.refine => {
                    // The GPU's are the start's in `f32`.
                    let kept = start
                        .as_ref()
                        .map(|d| d.iter().map(|d| d.map(|x| f64::from(x as f32))).collect());
                    assert_eq!(want, start, "{label}: the CPU moved held offsets");
                    let got = gpu.features().unwrap().position;
                    assert_eq!(got, Some(kept), "{label}: the GPU moved held offsets");
                    0.0
                }
                (Some(start), Some(want)) => {
                    let got = gpu.features().unwrap().position.unwrap().values;
                    let delta = |d: &[[f64; 2]]| -> Vec<f64> {
                        d.iter()
                            .zip(&start.values)
                            .flat_map(|(d, s)| [d[0] - s[0], d[1] - s[1]])
                            .collect()
                    };
                    let (a, b) = (delta(&got), delta(&want.values));
                    let d = a.iter().zip(&b).map(|(a, b)| (a - b).abs());
                    worst_of(d) / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
                }
                _ => 0.0,
            };
            let errs = [
                llk,
                rel_l2(&gpu.object().unwrap(), &objects[0]),
                rel_l2(&gpu.probe().unwrap(), &probe),
                steps,
                offsets,
            ];
            eprintln!(
                "{label}: llk {:.2e} object {:.2e} probe {:.2e} steps {:.2e} offsets {:.2e}, \
                 poisson llk {:.4e} -> {:.4e}",
                errs[0],
                errs[1],
                errs[2],
                errs[3],
                errs[4],
                trace[0],
                trace[NEPOCH - 1]
            );
            for (what, e) in ["llk", "object", "probe", "steps", "offsets"]
                .into_iter()
                .zip(errs)
            {
                assert!(e < 1e-4, "{label} {what}: {e:e}");
            }
        }
    }
}

/// Optics the GPU has no form for fail when the engine is built, naming the cause.
#[test]
fn gpu_refuses_optics_it_cannot_run() {
    let all = cases();
    let [c, slices] = ["angular spectrum", "three slices"]
        .map(|name| all.iter().find(|c| c.name == name).unwrap());
    let frame = c.probe.plane_shape();
    let free = |pixel, z| Optics::free_space(&Wavefield::new(frame, WAVELENGTH, pixel, z).unwrap());
    let refused = |optics: &Optics, options| c.gpu(optics, options).err().unwrap();

    let e = refused(&free([PIXEL, 1.2 * PIXEL], 5e-3), OPTIONS);
    assert!(e.contains("not square") && e.contains("6e-8"), "{e}");
    let e = refused(&slices.optics, OPTIONS);
    assert!(e.contains("1 object slices for optics of 3"), "{e}");
    let m1 = GpuOptions {
        fused: false,
        ..OPTIONS
    };
    let e = refused(&c.optics, m1);
    assert!(e.contains("fused: true"), "{e}");
    // A Fresnel transform forward in distance has no stage, and M1 runs it.
    let fresnel = all.iter().find(|c| c.name == "Fresnel").unwrap();
    let want = fresnel.llk(&fresnel.objects, &fresnel.probe);
    let got = fresnel.gpu(&fresnel.optics, m1).unwrap().llk().unwrap();
    assert!(
        (got.poisson - want).abs() < 1e-5 * want,
        "{got:?} vs {want:e}"
    );
}

/// Under a propagator that drops an evanescent band, every algorithm that takes its adjoint
/// for its inverse fails before it changes the arrays, as the CPU's `Ptycho` does; ML before
/// its photon scaling.
#[test]
fn gpu_refuses_what_an_evanescent_band_breaks() {
    let all = cases();
    let c = all.iter().find(|c| c.name == "evanescent band").unwrap();
    let form = c.optics.fourier_form().unwrap();
    assert!(matches!(form.stage, PlaneStage::Banded(_)));
    let mut gpu = c.gpu(&c.optics, OPTIONS).unwrap();
    let (object, probe) = (gpu.object().unwrap(), gpu.probe().unwrap());
    type Run = fn(&mut GpuPtycho) -> Option<String>;
    let runs: [(&str, Run); 6] = [
        ("AP", |g| {
            g.ap(&ApParams::default(), 1).err().map(|e| e.to_string())
        }),
        ("DM", |g| {
            g.dm(&DmParams::default(), 1).err().map(|e| e.to_string())
        }),
        ("RAAR", |g| {
            g.raar(&RaarParams::default(), 1)
                .err()
                .map(|e| e.to_string())
        }),
        ("rPIE", |g| {
            g.rpie(&RpieParams::default(), 1)
                .err()
                .map(|e| e.to_string())
        }),
        ("ML", |g| {
            g.ml(&MlParams::default(), 1).err().map(|e| e.to_string())
        }),
        ("ML's gradients", |g| {
            g.ml_grads().err().map(|e| e.to_string())
        }),
    ];
    for (name, run) in runs {
        let e = run(&mut gpu).unwrap_or_else(|| panic!("{name} ran"));
        assert!(e.contains("needs a unitary propagator"), "{name}: {e}");
    }
    assert_eq!(rel_l2(&gpu.object().unwrap(), &object), 0.0);
    assert_eq!(rel_l2(&gpu.probe().unwrap(), &probe), 0.0);
}
