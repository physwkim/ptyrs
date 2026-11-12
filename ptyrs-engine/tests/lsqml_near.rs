//! Poisson LSQML in pty-chi's planar near-field model against `refs/f64/lsqml_near.py`, which
//! generates the state and runs pty-chi on it, writing `lsqml_near_*_f64.txt`, saved to
//! `tests/data`. Its cases share the start object, probe and corners and differ in geometry
//! and counts: 0 an angular-spectrum propagator, 1 a Fresnel transform, 2 an angular spectrum
//! with an evanescent band. Every frame sits at a whole-pixel corner, so pty-chi's PATCH
//! extraction and the engine's views read the same pixels.

use ptyrs_core::{AddrTable, C64, Counts, Object, Positions, Probe, Stack, View};
use ptyrs_engine::{
    Cost, Lsqml, LsqmlBatch, LsqmlParams, LsqmlStep, Observed, Optics, ProbeVariation, Propagator,
    Wavefield, illumination, llk, obj_probe_to_psi, shift,
};

const SHAPE: [usize; 2] = [24, 24];
const OBJ: [usize; 2] = [40, 40];
const NPROBE: usize = 2;
const BATCH: usize = 8;

fn table(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

/// `max |a − b| / max |b|`.
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let d = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    d / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

fn stack(dims: [usize; 3], rows: &[Vec<f64>], lead: usize) -> Stack<C64> {
    let data = rows
        .iter()
        .map(|r| C64::new(r[lead], r[lead + 1]))
        .collect();
    Stack::from_vec(dims, data).unwrap()
}

struct Case {
    optics: Optics,
    obs: Observed,
    truth_llk: f64,
}

struct State {
    addr: AddrTable,
    object: Object,
    probe: Probe,
    cases: Vec<Case>,
}

fn state() -> State {
    // corner = (OBJ − SHAPE)/2 + position at a pixel of 1
    let corners = table(include_str!("data/lsqml_near_corners_f64.txt"));
    let at = |axis: usize| -> Vec<f64> {
        let centre = ((OBJ[axis] - SHAPE[axis]) / 2) as f64;
        corners.iter().map(|r| r[1 + axis] - centre).collect()
    };
    let positions = Positions::new(at(0), at(1)).unwrap();
    let addr = AddrTable::new(&positions, [1.0, 1.0], SHAPE, OBJ, NPROBE, 1).unwrap();
    for (f, r) in corners.iter().enumerate() {
        let v = addr.frame_views(f)[0];
        assert_eq!([v.oy as f64, v.ox as f64], [r[1], r[2]], "frame {f}");
    }
    let object = Object::new(stack(
        [1, OBJ[0], OBJ[1]],
        &table(include_str!("data/lsqml_near_obj_f64.txt")),
        2,
    ));
    let probe = Probe::new(stack(
        [NPROBE, SHAPE[0], SHAPE[1]],
        &table(include_str!("data/lsqml_near_probe_f64.txt")),
        3,
    ));
    let counts = table(include_str!("data/lsqml_near_counts_f64.txt"));
    let nframe = corners.len();
    let cases = table(include_str!("data/lsqml_near_cases_f64.txt"))
        .iter()
        .map(|c| {
            let w = Wavefield::new(SHAPE, c[1], [c[2], c[2]], c[3]).unwrap();
            let optics = Optics::free_space(&w);
            let p = optics.propagator();
            let kind = match p {
                Propagator::Fourier(_) => 0,
                Propagator::AngularSpectrum(_) => 1,
                Propagator::Fresnel(_) => 2,
            };
            assert_eq!(kind, c[4] as usize, "kind of case {}", c[0]);
            assert!(
                rel(p.intensity_scale(), c[5]) <= 1e-15,
                "scale of case {}",
                c[0]
            );
            let values: Vec<f64> = counts
                .iter()
                .filter(|r| r[0] == c[0])
                .flat_map(|r| r[2..].iter().copied())
                .collect();
            let dims = [nframe, SHAPE[0], SHAPE[1]];
            let obs = Observed::from_counts(&Counts::from_f64(dims, &values, None).unwrap());
            Case {
                optics,
                obs,
                truth_llk: c[6],
            }
        })
        .collect();
    State {
        addr,
        object,
        probe,
        cases,
    }
}

/// Detector-plane exit waves of `views`.
fn propagated(views: &[View], c: &Case, object: &Object, probe: &Probe) -> Vec<C64> {
    let mut psi = vec![C64::default(); views.len() * probe.plane_len()];
    obj_probe_to_psi(views, object, probe, c.optics.quad(), &mut psi);
    c.optics.propagator().forward(&mut psi);
    psi
}

/// The normalised Poisson LLK over every frame (`lsqml_near.llk`).
fn poisson_llk(s: &State, c: &Case, object: &Object, probe: &Probe) -> f64 {
    let psi = propagated(s.addr.views(), c, object, probe);
    let (nframe, nmode) = (s.addr.nframe(), s.addr.nmode());
    llk(&c.obs, 0..nframe, nmode, &psi)
        .normalised(c.obs.nb_obs())
        .poisson
}

/// pty-chi's forward model output `y_pred` (natural layout) at the start, frames 0..4.
#[test]
fn forward_matches_ptychi() {
    let s = state();
    let n = s.probe.plane_len();
    let nmode = s.addr.nmode();
    for r in table(include_str!("data/lsqml_near_icalc_f64.txt")) {
        let (c, f) = (&s.cases[r[0] as usize], r[1] as usize);
        let psi = propagated(s.addr.frame_views(f), c, &s.object, &s.probe);
        let got: Vec<f64> = (0..n)
            .map(|j| (0..nmode).map(|m| psi[m * n + j].norm_sqr()).sum())
            .collect();
        let want: Vec<f64> = (0..n)
            .map(|j| r[2 + shift(j, SHAPE[0], SHAPE[1])])
            .collect();
        let err = max_rel(&got, &want);
        assert!(err < 1e-14, "case {} frame {f}: {err:.3e}", r[0]);
    }
}

/// `Σ_f w_f Σ_pixels (icalc + 1e-6 − iobs·ln(icalc + 1e-6))`, the cost whose `icalc`
/// derivative is the step's Poisson residual, over frames `0..BATCH`.
fn weighted_cost(s: &State, c: &Case, w: &[f64], object: &Object, probe: &Probe) -> f64 {
    let (n, nmode) = (probe.plane_len(), s.addr.nmode());
    let psi = propagated(&s.addr.views()[..BATCH * nmode], c, object, probe);
    psi.chunks_exact(nmode * n)
        .enumerate()
        .zip(w)
        .map(|((f, block), w)| {
            let iobs = c.obs.frames(f..f + 1);
            let sum: f64 = (0..n)
                .filter_map(|j| {
                    let icalc: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                    iobs[j].map(|iobs| icalc + 1e-6 - iobs * (icalc + 1e-6).ln())
                })
                .sum();
            w * sum
        })
        .sum()
}

/// A complex array shaped as `like`, uniform in −1/2..1/2 per part, scaled to `like`'s largest
/// modulus.
fn direction(like: &Stack<C64>, seed: &mut u64) -> Stack<C64> {
    let mut uniform = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let scale = like.as_slice().iter().map(|c| c.norm()).fold(0.0, f64::max);
    let data = (0..like.as_slice().len())
        .map(|_| C64::new(uniform(), uniform()) * scale)
        .collect();
    Stack::from_vec(like.dims(), data).unwrap()
}

fn dot(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(a, b)| (a.conj() * b).re)
        .sum()
}

/// `x + t·d`.
fn moved(x: &Stack<C64>, t: f64, d: &Stack<C64>) -> Stack<C64> {
    let data = x
        .as_slice()
        .iter()
        .zip(d.as_slice())
        .map(|(x, d)| x + t * d)
        .collect();
    Stack::from_vec(x.dims(), data).unwrap()
}

/// `−2·obj_direction` and `−2·BATCH·probe_direction` against central differences of
/// [`weighted_cost`] with `w` the batch's `a_far`, cases 0 and 1. Case 2's directions are not
/// a gradient: its `UᴴU` is the projection off the evanescent band, so `chi` keeps
/// `−(1 − UᴴU)·psi` (pty-chi Eq. 19), which the step test checks against pty-chi instead.
#[test]
fn directions_match_central_differences() {
    let s = state();
    let (o, pr) = (&s.object, &s.probe);
    let none = ProbeVariation::none(pr.plane_shape());
    let run = Lsqml::new(LsqmlParams {
        batch_size: BATCH,
        cost: Cost::Poisson,
        seed: 0,
    });
    let views = &s.addr.views()[..BATCH * s.addr.nmode()];
    for (k, c) in s.cases.iter().enumerate().take(2) {
        let b = LsqmlBatch::new(&run, views, &c.obs, &c.optics, o, pr, &none);
        let w = b.a_far().to_vec();
        let mut g_o = b.obj_direction();
        let mut g_p = b.probe_direction();
        g_o.as_mut_slice().iter_mut().for_each(|v| *v *= -2.0);
        let count = BATCH as f64;
        g_p.as_mut_slice()
            .iter_mut()
            .for_each(|v| *v *= -2.0 * count);
        let mut seed = 0x2545_f491_4f6c_dd1d;
        let (d_o, d_p) = (direction(o, &mut seed), direction(pr, &mut seed));
        let f = |o: &Object, pr: &Probe| weighted_cost(&s, c, &w, o, pr);
        let h = 1e-5;
        let fo = (f(&Object::new(moved(o, h, &d_o)), pr) - f(&Object::new(moved(o, -h, &d_o)), pr))
            / (2.0 * h);
        let fp = (f(o, &Probe::new(moved(pr, h, &d_p))) - f(o, &Probe::new(moved(pr, -h, &d_p))))
            / (2.0 * h);
        let (eo, ep) = (rel(dot(&g_o, &d_o), fo), rel(dot(&g_p, &d_p), fp));
        eprintln!("case {k}: object {eo:.3e} probe {ep:.3e}");
        assert!(
            eo < 1e-7 && ep < 1e-7,
            "case {k}: object {eo:.3e} probe {ep:.3e}"
        );
    }
}

/// `steps` rows (`case [epoch] frame a_far a_o a_p`, `frame_col` the frame's column) against a
/// step's frames and step sizes.
fn assert_step(what: &str, s: &LsqmlStep, rows: &[&Vec<f64>], frame_col: usize) {
    let frames: Vec<usize> = rows.iter().map(|r| r[frame_col] as usize).collect();
    assert_eq!(s.frames, frames, "{what}");
    for (name, got, col, tol) in [
        ("a_far", &s.a_far, frame_col + 1, 1e-13),
        ("a_o", &s.a_obj, frame_col + 2, 1e-12),
        ("a_p", &s.a_probe, frame_col + 3, 1e-12),
    ] {
        let want: Vec<f64> = rows.iter().map(|r| r[col]).collect();
        let err = max_rel(got, &want);
        assert!(err < tol, "{what} {name}: max rel {err:.3e}");
    }
}

/// The changes of `object` and `probe` from the start against case `case`'s object
/// (`case iy ix re im`) and probe (`case m iy ix re im`) rows, each to `tol` of its largest.
fn assert_changes(s: &State, case: usize, object: &Object, probe: &Probe, obj: &str, pr: &str) {
    for (name, text, lead, got, before) in [
        ("object", obj, 3, &**object, &*s.object),
        ("probe", pr, 4, &**probe, &*s.probe),
    ] {
        let got: Vec<f64> = moved(got, -1.0, before)
            .as_slice()
            .iter()
            .flat_map(|c| [c.re, c.im])
            .collect();
        let want: Vec<f64> = table(text)
            .iter()
            .filter(|r| r[0] == case as f64)
            .flat_map(|r| [r[lead], r[lead + 1]])
            .collect();
        let err = max_rel(&got, &want);
        eprintln!("case {case} {name}: max rel {err:.3e}");
        assert!(err < 1e-12, "case {case} {name}: max rel {err:.3e}");
    }
}

/// One [`Lsqml::step`] on frames `0..BATCH` with every view's illumination as preconditioner,
/// every case, against pty-chi's `run_minibatch`.
#[test]
fn step_matches_ptychi() {
    let s = state();
    let rows = table(include_str!("data/lsqml_near_step_steps_f64.txt"));
    let views = &s.addr.views()[..BATCH * s.addr.nmode()];
    for (k, c) in s.cases.iter().enumerate() {
        let (mut object, mut probe) = (s.object.clone(), s.probe.clone());
        let pre = illumination(s.addr.views(), object.dims(), &probe);
        let mut var = ProbeVariation::none(probe.plane_shape());
        let step = Lsqml::new(LsqmlParams {
            batch_size: BATCH,
            cost: Cost::Poisson,
            seed: 0,
        })
        .with_preconditioner(pre)
        .step(views, &c.obs, &c.optics, &mut object, &mut probe, &mut var);
        let want: Vec<_> = rows.iter().filter(|r| r[0] == k as f64).collect();
        assert_step(&format!("case {k}"), &step, &want, 1);
        assert_changes(
            &s,
            k,
            &object,
            &probe,
            include_str!("data/lsqml_near_step_obj_f64.txt"),
            include_str!("data/lsqml_near_step_probe_f64.txt"),
        );
    }
}

/// The reconstruction: [`Lsqml::epoch`]s of batches of 8 (seed 7) from the start against
/// pty-chi's run of them with its own preconditioner schedule, cases 0 and 1: every step,
/// the normalised Poisson LLK after every epoch, the changes over the run, and the epoch
/// whose LLK first reaches the truth's.
#[test]
fn reconstruction_matches_ptychi() {
    let s = state();
    let steps = table(include_str!("data/lsqml_near_epochs_steps_f64.txt"));
    let trace = table(include_str!("data/lsqml_near_llk_f64.txt"));
    for k in [0, 1] {
        let c = &s.cases[k];
        let (mut object, mut probe) = (s.object.clone(), s.probe.clone());
        let mut var = ProbeVariation::none(probe.plane_shape());
        let mut lsqml = Lsqml::new(LsqmlParams {
            batch_size: BATCH,
            cost: Cost::Poisson,
            seed: 7,
        });
        let want: Vec<_> = trace.iter().filter(|r| r[0] == k as f64).collect();
        let epochs = want.len() - 1;
        let mut llks = vec![poisson_llk(&s, c, &object, &probe)];
        let rows: Vec<_> = steps.iter().filter(|r| r[0] == k as f64).collect();
        let mut next = 0;
        for e in 0..epochs {
            for st in lsqml.epoch(
                &s.addr,
                &c.obs,
                &c.optics,
                &mut object,
                &mut probe,
                &mut var,
            ) {
                let want = &rows[next..next + st.frames.len()];
                next += st.frames.len();
                assert!(want.iter().all(|r| r[1] == e as f64), "case {k} epoch {e}");
                assert_step(&format!("case {k} epoch {e}"), &st, want, 2);
            }
            llks.push(poisson_llk(&s, c, &object, &probe));
        }
        assert_eq!(next, rows.len());
        let want_llk: Vec<f64> = want.iter().map(|r| r[2]).collect();
        let err = max_rel(&llks, &want_llk);
        eprintln!("case {k} LLK trace: max rel {err:.3e}");
        assert!(err < 1e-12, "case {k} LLK trace: max rel {err:.3e}");
        // the number of epochs after which the LLK is first at or below the truth's
        let reached = |t: &[f64]| t.iter().position(|&v| v <= c.truth_llk);
        let (got, ptychi) = (reached(&llks), reached(&want_llk));
        eprintln!(
            "case {k}: LLK {:.4} -> {:.4}, truth {:.4} reached in {got:?} epochs (pty-chi \
             {ptychi:?})",
            llks[0], llks[epochs], c.truth_llk
        );
        assert_eq!(got, ptychi, "case {k}");
        assert!(got.is_some(), "case {k}");
        assert_changes(
            &s,
            k,
            &object,
            &probe,
            include_str!("data/lsqml_near_epochs_obj_f64.txt"),
            include_str!("data/lsqml_near_epochs_probe_f64.txt"),
        );
    }
}
