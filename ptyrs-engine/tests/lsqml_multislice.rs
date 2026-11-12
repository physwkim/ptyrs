//! Poisson LSQML through pty-chi's multislice object against `refs/f64/lsqml_multislice.py`,
//! which generates the state and runs pty-chi on it, writing `lsqml_multislice_*_f64.txt`,
//! saved to `tests/data`. Three slices with unequal gaps; the cases share the start and the
//! corners and differ in the exit propagator (0 angular spectrum, 1 Fresnel transform, 2 the
//! far field of [`Optics::far_field`], whose `q` multiplies the last slice) and the counts.
//! pty-chi runs with `multislice_solve_obj_prb_step_size_jointly`, the step-size rule of
//! [`LsqmlBatch::step_sizes`]. Every frame sits at a whole-pixel corner.

use ptyrs_core::{AddrTable, C64, Counts, Geometry, Object, Positions, Probe, Stack, View};
use ptyrs_engine::{
    Cost, Lsqml, LsqmlBatch, LsqmlParams, LsqmlStep, Observed, Optics, ProbeVariation, Propagator,
    SliceWaves, Wavefield, illumination, llk, shift,
};

const SHAPE: [usize; 2] = [24, 24];
const OBJ: [usize; 2] = [40, 40];
const NPROBE: usize = 2;
const NSLICE: usize = 3;
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

fn stack(dims: [usize; 3], rows: &[&Vec<f64>], lead: usize) -> Stack<C64> {
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
    objects: Vec<Object>,
    probe: Probe,
    cases: Vec<Case>,
}

fn state() -> State {
    // corner = (OBJ − SHAPE)/2 + position at a pixel of 1
    let corners = table(include_str!("data/lsqml_multislice_corners_f64.txt"));
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
    let obj = table(include_str!("data/lsqml_multislice_obj_f64.txt"));
    let objects = (0..NSLICE)
        .map(|s| {
            let rows: Vec<_> = obj.iter().filter(|r| r[0] == s as f64).collect();
            Object::new(stack([1, OBJ[0], OBJ[1]], &rows, 3))
        })
        .collect();
    let probe = table(include_str!("data/lsqml_multislice_probe_f64.txt"));
    let probe = Probe::new(stack(
        [NPROBE, SHAPE[0], SHAPE[1]],
        &probe.iter().collect::<Vec<_>>(),
        3,
    ));
    let counts = table(include_str!("data/lsqml_multislice_counts_f64.txt"));
    let nframe = corners.len();
    let cases = table(include_str!("data/lsqml_multislice_cases_f64.txt"))
        .iter()
        .map(|c| {
            // kind 0 is the far field at detector distance c[3], detector pixel λz/(ny·c[2])
            let (optics, pixel) = if c[4] == 0.0 {
                let lz = c[1] * c[3];
                let g = Geometry::new(c[1], c[3], lz / (SHAPE[0] as f64 * c[2]), SHAPE).unwrap();
                (Optics::far_field(&g), g.pixel_size_object())
            } else {
                let pixel = [c[2], c[2]];
                let w = Wavefield::new(SHAPE, c[1], pixel, c[3]).unwrap();
                (Optics::free_space(&w), pixel)
            };
            let gaps: Vec<_> = c[7..]
                .iter()
                .map(|&d| Wavefield::new(SHAPE, c[1], pixel, d).unwrap())
                .collect();
            assert_eq!(gaps.len(), NSLICE - 1);
            let optics = optics.with_slices(&gaps).unwrap();
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
        objects,
        probe,
        cases,
    }
}

/// Detector-plane exit waves of `views` through the slices.
fn propagated(views: &[View], c: &Case, objects: &[Object], probe: &Probe) -> Vec<C64> {
    let mut psi = SliceWaves::new(&c.optics, views, objects, probe)
        .psi()
        .to_vec();
    c.optics.propagator().forward(&mut psi);
    psi
}

/// The normalised Poisson LLK over every frame (`lsqml_multislice.llk`).
fn poisson_llk(s: &State, c: &Case, objects: &[Object], probe: &Probe) -> f64 {
    let psi = propagated(s.addr.views(), c, objects, probe);
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
    for r in table(include_str!("data/lsqml_multislice_icalc_f64.txt")) {
        let (c, f) = (&s.cases[r[0] as usize], r[1] as usize);
        let psi = propagated(s.addr.frame_views(f), c, &s.objects, &s.probe);
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
fn weighted_cost(s: &State, c: &Case, w: &[f64], objects: &[Object], probe: &Probe) -> f64 {
    let (n, nmode) = (probe.plane_len(), s.addr.nmode());
    let psi = propagated(&s.addr.views()[..BATCH * nmode], c, objects, probe);
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

/// `f'(0)` by the five-point central stencil. The far-field case's slice-0 cost has a curvature
/// that keeps the three-point stencil above 1e-7 at every step.
fn slope(f: impl Fn(f64) -> f64, h: f64) -> f64 {
    (8.0 * (f(h) - f(-h)) - (f(2.0 * h) - f(-2.0 * h))) / (12.0 * h)
}

/// Every slice's `−2·obj_directions` and `−2·BATCH·probe_direction` against central
/// differences of [`weighted_cost`] with `w` the batch's `a_far`, every case.
#[test]
fn directions_match_central_differences() {
    let s = state();
    let (os, pr) = (&s.objects, &s.probe);
    let none = ProbeVariation::none(pr.plane_shape());
    let run = Lsqml::new(LsqmlParams {
        batch_size: BATCH,
        cost: Cost::Poisson,
        seed: 0,
    });
    let views = &s.addr.views()[..BATCH * s.addr.nmode()];
    let h = 1e-3;
    for (k, c) in s.cases.iter().enumerate() {
        let b = LsqmlBatch::new(&run, views, &c.obs, &c.optics, os, pr, &none);
        let w = b.a_far().to_vec();
        let f = |os: &[Object], pr: &Probe| weighted_cost(&s, c, &w, os, pr);
        let mut seed = 0x2545_f491_4f6c_dd1d;
        for (i, mut g_o) in b.obj_directions().into_iter().enumerate() {
            g_o.as_mut_slice().iter_mut().for_each(|v| *v *= -2.0);
            let d_o = direction(&os[i], &mut seed);
            let at = |t: f64| {
                let mut os = os.clone();
                os[i] = Object::new(moved(&os[i], t, &d_o));
                f(&os, pr)
            };
            let e = rel(dot(&g_o, &d_o), slope(at, h));
            eprintln!("case {k} slice {i}: {e:.3e}");
            assert!(e < 1e-7, "case {k} slice {i}: {e:.3e}");
        }
        let mut g_p = b.probe_direction();
        let count = BATCH as f64;
        g_p.as_mut_slice()
            .iter_mut()
            .for_each(|v| *v *= -2.0 * count);
        let d_p = direction(pr, &mut seed);
        let fp = slope(|t| f(os, &Probe::new(moved(pr, t, &d_p))), h);
        let e = rel(dot(&g_p, &d_p), fp);
        eprintln!("case {k} probe: {e:.3e}");
        assert!(e < 1e-7, "case {k} probe: {e:.3e}");
    }
}

/// `steps` rows (`case [epoch] frame a_far a_o(slice 0..) a_p`, `frame_col` the frame's
/// column) against a step's frames and step sizes.
fn assert_step(what: &str, s: &LsqmlStep, rows: &[&Vec<f64>], frame_col: usize) {
    let frames: Vec<usize> = rows.iter().map(|r| r[frame_col] as usize).collect();
    assert_eq!(s.frames, frames, "{what}");
    let mut checks = vec![("a_far".to_string(), &s.a_far, frame_col + 1, 1e-13)];
    for (i, a) in s.a_obj.iter().enumerate() {
        checks.push((format!("a_o slice {i}"), a, frame_col + 2 + i, 1e-12));
    }
    checks.push(("a_p".to_string(), &s.a_probe, frame_col + 2 + NSLICE, 1e-12));
    for (name, got, col, tol) in checks {
        let want: Vec<f64> = rows.iter().map(|r| r[col]).collect();
        let err = max_rel(got, &want);
        assert!(err < tol, "{what} {name}: max rel {err:.3e}");
    }
}

/// The changes of `objects` and `probe` from the start against case `case`'s object
/// (`case slice iy ix re im`) and probe (`case m iy ix re im`) rows, each to `tol` of its
/// largest.
fn assert_changes(s: &State, case: usize, objects: &[Object], probe: &Probe, obj: &str, pr: &str) {
    let change = |got: &Stack<C64>, before: &Stack<C64>| -> Vec<f64> {
        moved(got, -1.0, before)
            .as_slice()
            .iter()
            .flat_map(|c| [c.re, c.im])
            .collect()
    };
    let obj_change: Vec<f64> = objects
        .iter()
        .zip(&s.objects)
        .flat_map(|(o, before)| change(o, before))
        .collect();
    for (name, text, got) in [
        ("object", obj, obj_change),
        ("probe", pr, change(probe, &s.probe)),
    ] {
        let want: Vec<f64> = table(text)
            .iter()
            .filter(|r| r[0] == case as f64)
            .flat_map(|r| [r[4], r[5]])
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
    let rows = table(include_str!("data/lsqml_multislice_step_steps_f64.txt"));
    let views = &s.addr.views()[..BATCH * s.addr.nmode()];
    for (k, c) in s.cases.iter().enumerate() {
        let (mut objects, mut probe) = (s.objects.clone(), s.probe.clone());
        let pre = illumination(s.addr.views(), objects[0].dims(), &probe);
        let mut var = ProbeVariation::none(probe.plane_shape());
        let step = Lsqml::new(LsqmlParams {
            batch_size: BATCH,
            cost: Cost::Poisson,
            seed: 0,
        })
        .with_preconditioner(pre)
        .step(views, &c.obs, &c.optics, &mut objects, &mut probe, &mut var);
        let want: Vec<_> = rows.iter().filter(|r| r[0] == k as f64).collect();
        assert_step(&format!("case {k}"), &step, &want, 1);
        assert_changes(
            &s,
            k,
            &objects,
            &probe,
            include_str!("data/lsqml_multislice_step_obj_f64.txt"),
            include_str!("data/lsqml_multislice_step_probe_f64.txt"),
        );
    }
}

/// The reconstruction: [`Lsqml::epoch`]s of batches of 8 (seed 7) from the start against
/// pty-chi's run of them with its own preconditioner schedule, every case: every step, the
/// normalised Poisson LLK after every epoch, the changes over the run, and the epoch whose
/// LLK first reaches the truth's.
#[test]
fn reconstruction_matches_ptychi() {
    let s = state();
    let steps = table(include_str!("data/lsqml_multislice_epochs_steps_f64.txt"));
    let trace = table(include_str!("data/lsqml_multislice_llk_f64.txt"));
    for (k, c) in s.cases.iter().enumerate() {
        let (mut objects, mut probe) = (s.objects.clone(), s.probe.clone());
        let mut var = ProbeVariation::none(probe.plane_shape());
        let mut lsqml = Lsqml::new(LsqmlParams {
            batch_size: BATCH,
            cost: Cost::Poisson,
            seed: 7,
        });
        let want: Vec<_> = trace.iter().filter(|r| r[0] == k as f64).collect();
        let epochs = want.len() - 1;
        let mut llks = vec![poisson_llk(&s, c, &objects, &probe)];
        let rows: Vec<_> = steps.iter().filter(|r| r[0] == k as f64).collect();
        let mut next = 0;
        for e in 0..epochs {
            for st in lsqml.epoch(
                &s.addr,
                &c.obs,
                &c.optics,
                &mut objects,
                &mut probe,
                &mut var,
            ) {
                let want = &rows[next..next + st.frames.len()];
                next += st.frames.len();
                assert!(want.iter().all(|r| r[1] == e as f64), "case {k} epoch {e}");
                assert_step(&format!("case {k} epoch {e}"), &st, want, 2);
            }
            llks.push(poisson_llk(&s, c, &objects, &probe));
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
            &objects,
            &probe,
            include_str!("data/lsqml_multislice_epochs_obj_f64.txt"),
            include_str!("data/lsqml_multislice_epochs_probe_f64.txt"),
        );
    }
}
