//! [`Ptycho::rpie`] under an angular-spectrum propagator and a Fresnel transform forward and
//! back against pty-chi's `RPIEReconstructor` in f64, `refs/f64/rpie_near.py`, which generates
//! the state and writes `rpie_near_*_f64.txt`, saved to `tests/engine/data`. Every frame sits
//! at a whole-pixel corner and the corners' mean is the object's centre, so the engine's
//! position preprocessing keeps them and both sides read the same pixels.

use ptyrs::base::{C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack};
use ptyrs::engine::{Optics, Propagator, Ptycho, RpieParams, Wavefield};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const SHAPE: [usize; 2] = [24, 24];
const OBJ: [usize; 2] = [40, 40];
const NPROBE: usize = 2;

fn table(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

/// `max |a − b| / max |b|`.
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let d = worst_of(a.iter().zip(b).map(|(a, b)| (a - b).abs()));
    d / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
}

fn stack(dims: [usize; 3], rows: &[Vec<f64>], lead: usize) -> Stack<C64> {
    let data = rows
        .iter()
        .map(|r| C64::new(r[lead], r[lead + 1]))
        .collect();
    Stack::from_vec(dims, data).unwrap()
}

/// `re_im(a − b)`.
fn change(a: &Stack<C64>, b: &Stack<C64>) -> Vec<f64> {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .flat_map(|(a, b)| [a.re - b.re, a.im - b.im])
        .collect()
}

/// The `re im` columns from `lead` of the rows of run `run`.
fn of_run(text: &str, run: f64, lead: usize) -> Vec<f64> {
    table(text)
        .iter()
        .filter(|r| r[0] == run)
        .flat_map(|r| [r[lead], r[lead + 1]])
        .collect()
}

/// Case row `c` (`case wavelength pixel distance kind scale truth_llk`) at the reference's
/// start.
fn start(c: &[f64]) -> Ptycho {
    let corners = table(include_str!("engine/data/rpie_near_corners_f64.txt"));
    let pixel = c[2];
    // corner = (OBJ − SHAPE)/2 + position in object pixels
    let at = |axis: usize| -> Vec<f64> {
        let centre = ((OBJ[axis] - SHAPE[axis]) / 2) as f64;
        corners
            .iter()
            .map(|r| (r[1 + axis] - centre) * pixel)
            .collect()
    };
    let positions = Positions::new(at(0), at(1)).unwrap();
    let counts: Vec<f64> = table(include_str!("engine/data/rpie_near_counts_f64.txt"))
        .iter()
        .filter(|r| r[0] == c[0])
        .flat_map(|r| r[2..].to_vec())
        .collect();
    let dims = [corners.len(), SHAPE[0], SHAPE[1]];
    let counts = Counts::from_f64(dims, &counts, None).unwrap();
    // Under `with_optics` only the frame of the geometry is read.
    let geometry = Geometry::new(c[1], 1.0, 75e-6, SHAPE).unwrap();
    let scan = Scan::new(geometry, positions, counts).unwrap();

    let w = Wavefield::new(SHAPE, c[1], [pixel, pixel], c[3]).unwrap();
    let kind = match Propagator::free_space(&w) {
        Propagator::Fourier(_) => 0,
        Propagator::AngularSpectrum(_) => 1,
        Propagator::Fresnel(_) => 2,
    };
    assert_eq!(kind, c[4] as usize, "kind of case {}", c[0]);
    let object = Object::new(stack(
        [1, OBJ[0], OBJ[1]],
        &table(include_str!("engine/data/rpie_near_obj_f64.txt")),
        2,
    ));
    let probe = Probe::new(stack(
        [NPROBE, SHAPE[0], SHAPE[1]],
        &table(include_str!("engine/data/rpie_near_probe_f64.txt")),
        3,
    ));
    let p = Ptycho::with_optics(&scan, Optics::free_space(&w), object, probe).unwrap();
    for (f, r) in corners.iter().enumerate() {
        let v = p.addr().frame_views(f)[0];
        assert_eq!([v.oy as f64, v.ox as f64], [r[1], r[2]], "frame {f}");
    }
    p
}

/// Every saved run (`run case batch epochs obj_step probe_step obj_alpha probe_alpha`: batches
/// of 1, 4 and all 16 frames under each propagator) against pty-chi's: the normalised Poisson
/// LLK after every epoch, the epoch it first reaches the truth's, and the object and probe
/// changes over the run.
#[test]
fn rpie_matches_ptychi() {
    let cases = table(include_str!("engine/data/rpie_near_cases_f64.txt"));
    let trace = table(include_str!("engine/data/rpie_near_llk_f64.txt"));
    let runs = table(include_str!("engine/data/rpie_near_runs_f64.txt"));
    assert_eq!(runs.len(), 3 * cases.len());
    for r in &runs {
        let c = &cases[r[1] as usize];
        let what = format!("run {} (case {}, batch {})", r[0], r[1], r[2]);
        let mut p = start(c);
        let (o0, p0) = (p.object().clone(), p.probe().clone());
        let params = RpieParams {
            obj_step: Some(r[4]),
            probe_step: Some(r[5]),
            obj_alpha: r[6],
            probe_alpha: r[7],
            batch_size: r[2] as usize,
            calc_llk: 1,
        };
        let mut llks = vec![p.llk().poisson];
        p.rpie(&params, r[3] as usize).unwrap();
        llks.extend(p.history().iter().map(|h| h.llk.poisson));
        let want: Vec<f64> = (trace.iter().filter(|t| t[0] == r[0]))
            .map(|t| t[2])
            .collect();
        let err = max_rel(&llks, &want);
        let reached = |t: &[f64]| t.iter().position(|&v| v <= c[6]);
        eprintln!(
            "{what}: LLK {:.4} -> {:.4}, truth {:.4} after {:?} epochs, trace max rel {err:.3e}",
            llks[0],
            llks[llks.len() - 1],
            c[6],
            reached(&llks)
        );
        assert!(err < 1e-12, "{what} LLK trace: max rel {err:.3e}");
        assert_eq!(reached(&llks), reached(&want), "{what}");
        assert!(llks[llks.len() - 1] < 0.1 * llks[0], "{what}");
        for (name, text, lead, got) in [
            (
                "object",
                include_str!("engine/data/rpie_near_run_obj_f64.txt"),
                3,
                change(p.object(), &o0),
            ),
            (
                "probe",
                include_str!("engine/data/rpie_near_run_probe_f64.txt"),
                4,
                change(p.probe(), &p0),
            ),
        ] {
            let err = max_rel(&got, &of_run(text, r[0], lead));
            eprintln!("{what} {name}: max rel {err:.3e}");
            assert!(err < 1e-12, "{what} {name}: max rel {err:.3e}");
        }
    }
}
