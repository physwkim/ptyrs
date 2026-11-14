//! [`Ptycho`]'s AP, DM, RAAR, rPIE and ML under each propagator kind of [`Optics`]: on counts
//! of the truth's own model, drawn through the same optics without noise, the truth stays put
//! and a start away from it comes closer. [`Counts`] are whole photons, so "without noise" is
//! the model intensity rounded, at a peak of 4e4 a pixel; the truth is a fixed point up to
//! that rounding.

use ptyrs::base::{C64, Counts, Geometry, Object, Probe, Scan, Stack, required_obj_shape, sim};
use ptyrs::engine::{
    ApParams, DmParams, Features, MlParams, Optics, Propagator, Ptycho, RaarParams, RpieParams,
    Wavefield, obj_probe_shift_to_psi, shift,
};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const FRAME: [usize; 2] = [16, 16];
const N: usize = FRAME[0] * FRAME[1];
const WAVELENGTH: f64 = 1.5e-10;
const PEAK: f64 = 4e4;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Far,
    Angular,
    Fresnel,
    FresnelBack,
}

const NEAR: [Kind; 3] = [Kind::Angular, Kind::Fresnel, Kind::FresnelBack];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Algo {
    Ap,
    Dm,
    Raar,
    Rpie,
    Ml,
}

const ALGOS: [Algo; 5] = [Algo::Ap, Algo::Dm, Algo::Raar, Algo::Rpie, Algo::Ml];

/// What the counts' model holds beside the object and probe.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Extra {
    None,
    /// Offsets modelled, the true ones zero.
    Still,
    Position,
    ScaleBackground,
}

fn geometry() -> Geometry {
    Geometry::new(WAVELENGTH, 1.0, 75e-6, FRAME).unwrap()
}

fn optics(kind: Kind) -> Optics {
    let near = |distance: f64, want: char| {
        let w = Wavefield::new(FRAME, WAVELENGTH, [50e-9, 50e-9], distance).unwrap();
        let got = match Propagator::free_space(&w) {
            Propagator::Fourier(_) => 'F',
            Propagator::AngularSpectrum(_) => 'A',
            Propagator::Fresnel(_) => 'T',
        };
        assert_eq!(got, want);
        Optics::free_space(&w)
    };
    match kind {
        Kind::Far => Optics::far_field(&geometry()),
        Kind::Angular => near(2e-4, 'A'),
        Kind::Fresnel => near(1.25e-3, 'T'),
        Kind::FresnelBack => near(-1.25e-3, 'T'),
    }
}

struct Case {
    kind: Kind,
    scan: Scan,
    object: Object,
    probe: Probe,
    /// The features the counts were drawn under.
    truth: Features,
    /// The same features away from the truth.
    start: Features,
}

impl Case {
    fn at(&self, object: Object, features: &Features) -> Ptycho {
        let mut p =
            Ptycho::with_optics(&self.scan, optics(self.kind), object, self.probe.clone()).unwrap();
        p.set_features(features.clone()).unwrap();
        p
    }

    fn at_truth(&self) -> Ptycho {
        self.at(self.object.clone(), &self.truth)
    }

    /// The object's modulus and phase off by up to 10% and 0.4 rad, under [`Case::start`].
    fn away(&self) -> Ptycho {
        let o = (self.object.as_slice().iter().enumerate())
            .map(|(i, v)| v * C64::from_polar(0.9, 0.4 * (0.37 * i as f64).sin()))
            .collect();
        let o = Object::new(Stack::from_vec(self.object.dims(), o).unwrap());
        self.at(o, &self.start)
    }
}

/// The detector intensity of `p`'s object and probe moved by `offsets`, `[nframe · N]` in
/// detector order.
fn intensity(p: &Ptycho, offsets: &[[f64; 2]]) -> Vec<f64> {
    let views = p.addr().views();
    let nmode = p.addr().nmode();
    let mut psi = vec![C64::default(); views.len() * N];
    obj_probe_shift_to_psi(p, views, offsets, &mut psi);
    p.optics().exit().forward(&mut psi);
    let mut out = vec![0.0; p.addr().nframe() * N];
    for (out, block) in (out.as_chunks_mut::<N>().0.iter_mut()).zip(psi.chunks_exact(nmode * N)) {
        for (i, out) in out.iter_mut().enumerate() {
            let j = shift(i, FRAME[0], FRAME[1]);
            *out = (0..nmode).map(|m| block[m * N + j].norm_sqr()).sum();
        }
    }
    out
}

/// One object mode and two probe modes with `Σ|O| = Σ|P|`, a masked detector pixel, and the
/// rounded counts of their model under `kind` and `extra`.
fn case(kind: Kind, extra: Extra) -> Case {
    let g = geometry();
    let pixel = optics(kind).pixel();
    let positions = sim::fermat(20, 3.0 * pixel[1]);
    let nframe = positions.len();
    let shape = required_obj_shape(&positions.pynx_preprocess(pixel), pixel, FRAME, 0);
    let (mut object, mut probe) = (sim::object(shape), sim::probe(FRAME, 2).unwrap());
    let offsets: Vec<[f64; 2]> = match extra {
        Extra::Position => (0..nframe)
            .map(|f| {
                [
                    0.3 * (1.3 * f as f64 + 0.2).sin(),
                    -0.25 * (0.7 * f as f64).cos(),
                ]
            })
            .collect(),
        _ => vec![[0.0; 2]; nframe],
    };
    let dims = [nframe, FRAME[0], FRAME[1]];
    let blank = Counts::from_raw(dims, vec![0; nframe * N]).unwrap();
    let blank = Scan::new(g, positions.clone(), blank).unwrap();
    let sum = |a: &[C64]| a.iter().map(|c| c.norm()).sum::<f64>();
    let balance = (sum(object.as_slice()) / sum(probe.as_slice())).sqrt();
    probe.as_mut_slice().iter_mut().for_each(|c| *c *= balance);
    object.as_mut_slice().iter_mut().for_each(|c| *c /= balance);
    let unit = Ptycho::with_optics(&blank, optics(kind), object.clone(), probe.clone()).unwrap();
    let peak = intensity(&unit, &offsets).into_iter().fold(0.0, f64::max);
    let s = (PEAK / peak).powf(0.25);
    probe.as_mut_slice().iter_mut().for_each(|c| *c *= s);
    object.as_mut_slice().iter_mut().for_each(|c| *c *= s);
    let scaled = Ptycho::with_optics(&blank, optics(kind), object.clone(), probe.clone()).unwrap();
    let mut model = intensity(&scaled, &offsets);
    let (truth, start) = match extra {
        Extra::None => (Features::default(), Features::default()),
        Extra::Still => {
            let still = Features {
                position: Some(offsets),
                ..Features::default()
            };
            (still.clone(), still)
        }
        Extra::Position => (
            Features {
                position: Some(offsets),
                ..Features::default()
            },
            Features {
                position: Some(vec![[0.0; 2]; nframe]),
                ..Features::default()
            },
        ),
        Extra::ScaleBackground => {
            let scale: Vec<f64> = (0..nframe)
                .map(|f| 1.0 + 0.2 * (1.9 * f as f64).sin())
                .collect();
            let mean = scale.iter().sum::<f64>() / nframe as f64;
            let scale: Vec<f64> = scale.iter().map(|s| s / mean).collect();
            let background: Vec<f64> = (0..N)
                .map(|i| 200.0 + 100.0 * (0.3 * i as f64).cos())
                .collect();
            for (f, frame) in model.as_chunks_mut::<N>().0.iter_mut().enumerate() {
                for (v, b) in frame.iter_mut().zip(&background) {
                    *v = scale[f] * *v + b;
                }
            }
            (
                Features {
                    scale: Some(scale),
                    background: Some(background),
                    ..Features::default()
                },
                Features {
                    scale: Some(vec![1.0; nframe]),
                    background: Some(vec![150.0; N]),
                    ..Features::default()
                },
            )
        }
    };
    let rounded: Vec<f64> = model.iter().map(|v| v.round()).collect();
    let mut mask = vec![false; N];
    mask[3 * FRAME[1] + 5] = true;
    let counts = Counts::from_f64(dims, &rounded, Some(&mask)).unwrap();
    Case {
        kind,
        scan: Scan::new(g, positions, counts).unwrap(),
        object,
        probe,
        truth,
        start,
    }
}

fn run(p: &mut Ptycho, algo: Algo, n: usize) {
    match algo {
        Algo::Ap => p.ap(
            &ApParams {
                obj_inertia: Some(0.01),
                probe_inertia: Some(0.001),
                calc_llk: 0,
                zero_phase_ramp: false,
            },
            n,
        ),
        Algo::Dm => p.dm(
            &DmParams {
                zero_phase_ramp: false,
                ..DmParams::default()
            },
            n,
        ),
        Algo::Raar => p.raar(
            &RaarParams {
                zero_phase_ramp: false,
                ..RaarParams::default()
            },
            n,
        ),
        Algo::Rpie => p
            .rpie(
                &RpieParams {
                    batch_size: 4,
                    ..RpieParams::default()
                },
                n,
            )
            .unwrap(),
        Algo::Ml => p.ml(
            &MlParams {
                update_object: true,
                update_probe: true,
                calc_llk: 0,
                zero_phase_ramp: false,
            },
            n,
        ),
    }
}

/// `max |a − b| / max |b|`.
fn moved(a: &[C64], b: &[C64]) -> f64 {
    let d = worst_of(a.iter().zip(b).map(|(a, b)| (a - b).norm()));
    d / b.iter().map(|b| b.norm()).fold(0.0, f64::max)
}

/// The largest change of the object, the probe (each over its largest modulus), the offsets
/// (pixels), the scale and the background (each over its largest value) after `n` cycles of
/// `algo` from the truth.
fn drift(c: &Case, algo: Algo, n: usize) -> f64 {
    let mut p = c.at_truth();
    run(&mut p, algo, n);
    let of = |f: &Features| -> [Vec<f64>; 3] {
        [
            f.position.iter().flatten().flatten().copied().collect(),
            f.scale.clone().unwrap_or_default(),
            f.background.clone().unwrap_or_default(),
        ]
    };
    let (got, want) = (of(p.features()), of(&c.truth));
    let features = (got.iter().zip(&want)).enumerate().map(|(k, (g, w))| {
        let d = worst_of(g.iter().zip(w).map(|(g, w)| (g - w).abs()));
        // The offsets are compared in pixels.
        let over = if k == 0 {
            1.0
        } else {
            worst_of(w.iter().copied())
        };
        if w.is_empty() { 0.0 } else { d / over }
    });
    let arrays = [
        moved(p.object().as_slice(), c.object.as_slice()),
        moved(p.probe().as_slice(), c.probe.as_slice()),
    ];
    worst_of(arrays.into_iter().chain(features))
}

/// The Poisson LLK after `n` cycles of `algo` from [`Case::away`] over the LLK there, both of
/// the scaled estimate.
fn fall(c: &Case, algo: Algo, n: usize) -> f64 {
    let mut p = c.away();
    p.scale_obj_probe();
    let before = p.llk().poisson;
    run(&mut p, algo, n);
    p.llk().poisson / before
}

#[test]
fn the_truth_is_a_fixed_point_under_every_propagator() {
    for kind in [Kind::Far].into_iter().chain(NEAR) {
        let c = case(kind, Extra::None);
        for algo in ALGOS {
            let d = drift(&c, algo, 3);
            eprintln!("{kind:?} {algo:?}: drift {d:.3e}");
            assert!(d < 1e-3, "{kind:?} {algo:?}: drift {d:.3e}");
        }
    }
}

#[test]
fn the_llk_falls_from_a_start_away_from_the_truth() {
    for kind in [Kind::Far].into_iter().chain(NEAR) {
        let c = case(kind, Extra::None);
        for algo in ALGOS {
            let r = fall(&c, algo, 10);
            eprintln!("{kind:?} {algo:?}: llk ratio {r:.3e}");
            assert!(r < 0.6, "{kind:?} {algo:?}: llk ratio {r:.3e}");
        }
    }
}

/// With offsets modelled the truth stays put under all five, whether the counts hold none or
/// offsets of both signs, and the LLK falls from a start without them.
#[test]
fn positions_hold_and_the_llk_falls_under_every_propagator() {
    for kind in [Kind::Far].into_iter().chain(NEAR) {
        let still = case(kind, Extra::Still);
        let c = case(kind, Extra::Position);
        for algo in ALGOS {
            let d = drift(&still, algo, 3).max(drift(&c, algo, 3));
            let r = fall(&c, algo, 40);
            eprintln!("{kind:?} {algo:?}: drift {d:.3e}, llk ratio {r:.3e}");
            assert!(
                d < 2e-3 && r < 0.9,
                "{kind:?} {algo:?}: drift {d:.3e}, llk ratio {r:.3e}"
            );
        }
    }
}

/// rPIE models neither feature.
#[test]
fn scale_and_background_hold_and_the_llk_falls_under_angular_spectrum() {
    let c = case(Kind::Angular, Extra::ScaleBackground);
    for algo in [Algo::Ap, Algo::Dm, Algo::Raar, Algo::Ml] {
        let (d, r) = (drift(&c, algo, 3), fall(&c, algo, 10));
        eprintln!("{algo:?}: drift {d:.3e}, llk ratio {r:.3e}");
        assert!(
            d < 2e-3 && r < 0.4,
            "{algo:?}: drift {d:.3e}, llk ratio {r:.3e}"
        );
    }
}

#[test]
fn with_optics_refuses_what_ptycho_cannot_run() {
    let c = case(Kind::Angular, Extra::None);
    let refused = |optics: Optics| {
        Ptycho::with_optics(&c.scan, optics, c.object.clone(), c.probe.clone())
            .err()
            .map(|e| e.to_string())
            .expect("refused")
    };
    let evanescent = Wavefield::new(FRAME, 5e-7, [3e-7, 3e-7], 1.5e-6).unwrap();
    assert!(Propagator::free_space(&evanescent).fourier_form().is_err());
    assert!(refused(Optics::free_space(&evanescent)).contains("unitary"));
    let gap = Wavefield::new(FRAME, WAVELENGTH, [50e-9, 50e-9], 1e-6).unwrap();
    let sliced = optics(Kind::Angular).with_slices(&[gap]).unwrap();
    assert!(refused(sliced).contains("one object slice"));
    let other = Wavefield::new([16, 12], WAVELENGTH, [50e-9, 50e-9], 1e-4).unwrap();
    assert!(refused(Optics::free_space(&other)).contains("differs from frame"));
}
