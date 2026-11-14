//! GPU BH under the propagators with a plane stage against the CPU's `f64` [`Bh`] from the
//! same start, on the fused passes' smallest side: an angular-spectrum propagator and a
//! Fresnel transform forward and back. Two incoherent probe modes, 30 Fermat positions,
//! counts the truth's intensities rounded.

use std::slice;
use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{
    AddrTable, C64, Counts, Geometry, Object, Probe, Scan, Stack, required_obj_shape,
};
use ptyrs::engine::{
    Bh, BhParams, BhStep, Features, GpuOptions, GpuPtycho, Observed, Optics, SliceWaves, Wavefield,
    llk, shift,
};
use ptyrs::gpu::FUSED_SIDES;

const NPOS: usize = 30;
const NPROBE: usize = 2;
const WAVELENGTH: f64 = 1e-10;
const PIXEL: f64 = 50e-9;
/// Photons the probe sends through a frame.
const PHOTONS: f64 = 2e5;
/// The far-field GPU-against-CPU BH tests' bound on steps, arrays and LLK.
const TOL: f64 = 1e-4;

const OPTIONS: GpuOptions = GpuOptions {
    stack_size: Some(8),
    graph: true,
    llk_slots: 3,
    fused: true,
};

struct Case {
    name: &'static str,
    scan: Scan,
    addr: AddrTable,
    obs: Observed,
    optics: Optics,
    object: Object,
    probe: Probe,
}

/// Detector-plane exit waves of every view.
fn propagated(addr: &AddrTable, optics: &Optics, object: &Object, probe: &Probe) -> Vec<C64> {
    let mut psi = SliceWaves::new(optics, addr.views(), slice::from_ref(object), probe)
        .psi()
        .to_vec();
    optics.exit().forward(&mut psi);
    psi
}

fn case(name: &'static str, distance: f64) -> Case {
    let frame = FUSED_SIDES[0] as usize;
    let pixel = [PIXEL, PIXEL];
    let optics =
        Optics::free_space(&Wavefield::new([frame, frame], WAVELENGTH, pixel, distance).unwrap());
    let raw = ptyrs::base::sim::fermat(NPOS, frame as f64 / 8.0 * PIXEL);
    let positions = raw.pynx_preprocess(pixel);
    let shape = required_obj_shape(&positions, pixel, [frame, frame], 16);
    let addr = AddrTable::new(&positions, pixel, [frame, frame], shape, NPROBE, 1).unwrap();
    let truth = ptyrs::base::sim::object(shape);
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

    // The start: unit modulus and a random phase in -1/4..1/4 rad, the probe's defocus cut
    // by a third.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let data = (truth.as_slice().iter())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            C64::from_polar(1.0, 0.5 * (u - 0.5))
        })
        .collect();
    let object = Object::new(Stack::from_vec(truth.dims(), data).unwrap());
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
        object,
        probe,
    }
}

fn cases() -> [Case; 3] {
    [
        case("angular spectrum", 1e-4),
        case("Fresnel", 5e-3),
        // Not nearer: at -5e-3 and -2e-3 the CPU's second CG step has H(D, D) < 0, after
        // which its own steps from an f32 start are 2e-6 to 2e-5 off and the GPU's 15x that.
        case("Fresnel back", -1e-2),
    ]
}

impl Case {
    /// The normalised Poisson LLK of `object` and `probe` over every frame.
    fn llk(&self, object: &Object, probe: &Probe) -> f64 {
        let psi = propagated(&self.addr, &self.optics, object, probe);
        llk(&self.obs, 0..self.addr.nframe(), self.addr.nmode(), &psi)
            .normalised(self.obs.nb_obs())
            .poisson
    }

    fn gpu(&self, graph: bool) -> GpuPtycho {
        let ctx: Arc<CudaContext> = CudaContext::new(0).expect("CUDA device 0");
        let options = GpuOptions { graph, ..OPTIONS };
        let (object, probe) = (self.object.clone(), self.probe.clone());
        GpuPtycho::with_optics(&ctx, &self.scan, &self.optics, object, probe, options).unwrap()
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
            worst = worst.max(if b == 0.0 {
                a.abs()
            } else {
                (a - b).abs() / b.abs()
            });
        }
    }
    worst
}

fn check(what: &str, names: &[&str], errs: &[f64]) {
    for (name, e) in names.iter().zip(errs) {
        assert!(*e < TOL, "{what} {name}: {e:e}");
    }
}

/// Under each propagator, eagerly and from graphs, after a fresh GPU run of each of the
/// first epochs: the last epoch's steps, the arrays and the LLK the GPU records after it.
/// One batch of every frame (CG from the second epoch) and batches of 4, the last of 2.
#[test]
fn bh_steps_match_cpu_under_every_propagator() {
    const NEPOCH: usize = 3;
    for c in cases() {
        for batch_size in [NPOS, 4] {
            let p = params(batch_size);
            let mut bh = Bh::new(p).unwrap();
            let (mut o, mut pr) = (c.object.clone(), c.probe.clone());
            let start = c.llk(&o, &pr);
            let mut worst = [[0.0f64; 4]; 2];
            let mut last = start;
            for nepoch in 1..=NEPOCH {
                let want = bh.epoch(&c.addr, &c.obs, &c.optics, slice::from_mut(&mut o), &mut pr);
                let llk = c.llk(&o, &pr);
                last = llk;
                for (graph, worst) in [false, true].into_iter().zip(&mut worst) {
                    let mut gpu = c.gpu(graph);
                    gpu.bh(&p, nepoch, 1).unwrap();
                    let got = gpu.history().last().unwrap().llk.poisson;
                    let errs = [
                        step_err(&gpu.bh_steps().unwrap(), &want),
                        rel_l2(&gpu.object().unwrap(), &o),
                        rel_l2(&gpu.probe().unwrap(), &pr),
                        (got - llk).abs() / llk,
                    ];
                    for (w, e) in worst.iter_mut().zip(errs) {
                        *w = w.max(e);
                    }
                }
            }
            for (graph, w) in [false, true].into_iter().zip(worst) {
                let what = format!("{} batch {batch_size} graph {graph}", c.name);
                eprintln!(
                    "{what}: step {:.2e} object {:.2e} probe {:.2e} llk {:.2e}, llk {start:.4e} -> {last:.4e}",
                    w[0], w[1], w[2], w[3]
                );
                check(&what, &["step", "object", "probe", "llk"], &w);
            }
        }
    }
}

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

/// BH refining positions from drawn offsets under each propagator, eagerly and from graphs:
/// after four epochs the last epoch's steps, the arrays and the offsets.
#[test]
fn bh_position_matches_cpu_under_every_propagator() {
    const NEPOCH: usize = 4;
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    for c in cases() {
        for batch_size in [NPOS, 4] {
            let p = params(batch_size);
            let mut bh = Bh::new(p).unwrap().with_features(moved.clone()).unwrap();
            let (mut o, mut pr) = (c.object.clone(), c.probe.clone());
            let mut steps = Vec::new();
            for _ in 0..NEPOCH {
                steps = bh.epoch(&c.addr, &c.obs, &c.optics, slice::from_mut(&mut o), &mut pr);
            }
            let want = bh.features().position.clone().unwrap();
            let scale = want.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            for graph in [false, true] {
                let mut gpu = c.gpu(graph);
                gpu.set_features(&moved).unwrap();
                gpu.bh(&p, NEPOCH, 0).unwrap();
                let got = gpu.features().unwrap().position.unwrap();
                let offset = (got.iter().flatten().zip(want.iter().flatten()))
                    .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()))
                    / scale;
                let errs = [
                    step_err(&gpu.bh_steps().unwrap(), &steps),
                    rel_l2(&gpu.object().unwrap(), &o),
                    rel_l2(&gpu.probe().unwrap(), &pr),
                    offset,
                ];
                let what = format!("{} positions batch {batch_size} graph {graph}", c.name);
                eprintln!(
                    "{what}: step {:.2e} object {:.2e} probe {:.2e} offset {:.2e}",
                    errs[0], errs[1], errs[2], errs[3]
                );
                check(&what, &["step", "object", "probe", "offset"], &errs);
            }
        }
    }
}
