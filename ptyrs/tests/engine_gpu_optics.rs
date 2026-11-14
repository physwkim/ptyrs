//! The near-field and multislice cases GPU LSQML is held to, on the fused passes' smallest
//! side: an angular-spectrum propagator, a Fresnel transform forward and back, and three
//! object slices under angular spectrum, each run by the CPU's `f64` [`Lsqml`] from a start
//! off the truth its counts come from. Two incoherent probe modes, 30 Fermat positions,
//! counts the truth's intensities rounded.

use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{Geometry, Scan};
use ptyrs::engine::{ApParams, GpuOptions, GpuPtycho};

use ptyrs::base::{AddrTable, C64, Counts, Object, Probe, Stack, required_obj_shape};
use ptyrs::engine::{
    Cost, Lsqml, LsqmlParams, LsqmlStep, Observed, Optics, ProbeVariation, SliceWaves, Wavefield,
    llk, shift,
};
use ptyrs::gpu::FUSED_SIDES;

const NPOS: usize = 30;
const NPROBE: usize = 2;
const WAVELENGTH: f64 = 1e-10;
const PIXEL: f64 = 50e-9;
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
    let mut psi = SliceWaves::new(optics, addr.views(), objects, probe)
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
    vec![
        case(
            "angular spectrum",
            frame,
            Optics::free_space(&wavefield(frame, 1e-4)),
            &[1.0],
        ),
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

/// Optics the GPU has no form for fail when the engine is built, naming the cause; a
/// propagator with a plane stage fails every pass that does not run it, changing nothing.
#[test]
fn gpu_refuses_optics_and_passes_it_cannot_run() {
    let all = cases();
    let [c, slices] = ["angular spectrum", "three slices"]
        .map(|name| all.iter().find(|c| c.name == name).unwrap());
    let frame = c.probe.plane_shape();
    let free = |pixel, z| Optics::free_space(&Wavefield::new(frame, WAVELENGTH, pixel, z).unwrap());
    let refused = |optics: &Optics, options| c.gpu(optics, options).err().unwrap();

    let e = refused(&free([PIXEL, 1.2 * PIXEL], 5e-3), OPTIONS);
    assert!(e.contains("not square") && e.contains("6e-8"), "{e}");
    let e = refused(&free([0.4 * WAVELENGTH; 2], 1e-9), OPTIONS);
    assert!(e.contains("evanescent band"), "{e}");
    let e = refused(&slices.optics, OPTIONS);
    assert!(e.contains("one object slice"), "{e}");
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

    let mut gpu = c.gpu(&c.optics, OPTIONS).unwrap();
    let before = gpu.object().unwrap();
    let e = gpu.ap(&ApParams::default(), 1).unwrap_err().to_string();
    assert!(
        e.contains("Ap runs no propagator with a plane stage"),
        "{e}"
    );
    assert!(gpu.ml(&Default::default(), 1).is_err());
    assert!(gpu.ml_grads().is_err());
    assert_eq!(gpu.object().unwrap().as_slice(), before.as_slice());
    assert!(gpu.history().is_empty());
}
