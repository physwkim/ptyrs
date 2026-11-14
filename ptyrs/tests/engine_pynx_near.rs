//! [`Ptycho`] AP, DM, RAAR and ML under an angular-spectrum propagator against PyNX's
//! `cpu_operator` promoted to f64 in its near-field model, `refs/f64/pynx_near.py`, which
//! generates the state and writes `pynx_near_*_f64.txt`, saved to `tests/engine/data`. PyNX
//! runs under its own paraxial kernel and under the engine's angular spectrum: the first runs
//! differ from the engine by what the kernels' phase difference (the case's `kernel`, in
//! radians) produces, the second by rounding.

use ptyrs::base::{C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack};
use ptyrs::engine::{
    ApParams, DmParams, Llk, MlParams, Optics, Propagator, Ptycho, RaarParams, Wavefield,
};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const SHAPE: [usize; 2] = [24, 24];
const OBJ: [usize; 2] = [40, 40];
const NPROBE: usize = 2;
/// PyNX's AP cycles before its ML (the script's `AP_BEFORE_ML`).
const AP_BEFORE_ML: usize = 10;

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

/// `re im` of `a / Σ|a|`: DM and RAAR here balance `Σ|O|` and `Σ|P|` after each recorded
/// cycle, which PyNX's `cpu_operator` does not, and their updates commute with that balance.
fn unit(a: &[C64]) -> Vec<f64> {
    let s: f64 = a.iter().map(|c| c.norm()).sum();
    a.iter().flat_map(|c| [c.re / s, c.im / s]).collect()
}

/// The case row `wavelength pixel distance kind scale truth_llk kernel` and the reference's
/// start.
fn start() -> (Vec<f64>, Ptycho) {
    let c = table(include_str!("engine/data/pynx_near_case_f64.txt")).remove(0);
    let corners = table(include_str!("engine/data/pynx_near_corners_f64.txt"));
    let pixel = c[1];
    // corner = (OBJ − SHAPE)/2 + position in object pixels
    let at = |axis: usize| -> Vec<f64> {
        let centre = ((OBJ[axis] - SHAPE[axis]) / 2) as f64;
        corners
            .iter()
            .map(|r| (r[1 + axis] - centre) * pixel)
            .collect()
    };
    let positions = Positions::new(at(0), at(1)).unwrap();
    let counts: Vec<f64> = table(include_str!("engine/data/pynx_near_counts_f64.txt"))
        .iter()
        .flat_map(|r| r[1..].to_vec())
        .collect();
    let dims = [corners.len(), SHAPE[0], SHAPE[1]];
    let counts = Counts::from_f64(dims, &counts, None).unwrap();
    // Under `with_optics` only the frame of the geometry is read.
    let geometry = Geometry::new(c[0], 1.0, 75e-6, SHAPE).unwrap();
    let scan = Scan::new(geometry, positions, counts).unwrap();

    let w = Wavefield::new(SHAPE, c[0], [pixel, pixel], c[2]).unwrap();
    assert!(matches!(
        Propagator::free_space(&w),
        Propagator::AngularSpectrum(_)
    ));
    let object = Object::new(stack(
        [1, OBJ[0], OBJ[1]],
        &table(include_str!("engine/data/pynx_near_obj_f64.txt")),
        2,
    ));
    let probe = Probe::new(stack(
        [NPROBE, SHAPE[0], SHAPE[1]],
        &table(include_str!("engine/data/pynx_near_probe_f64.txt")),
        3,
    ));
    let p = Ptycho::with_optics(&scan, Optics::free_space(&w), object, probe).unwrap();
    for (f, r) in corners.iter().enumerate() {
        let v = p.addr().frame_views(f)[0];
        assert_eq!([v.oy as f64, v.ox as f64], [r[1], r[2]], "frame {f}");
    }
    (c, p)
}

fn columns(l: &Llk) -> [f64; 4] {
    [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc]
}

/// Run `algorithm` (0 AP, 1 DM, 2 RAAR, 3 ML) of `cycles` cycles as PyNX's run: its LLK
/// records in PyNX's order, the final object and probe. PyNX's `cpu_operator` DM, RAAR and ML
/// record the estimate after each cycle's update, the engine's the one before, so those take
/// the engine's records from the second on and the final LLK.
fn engine_run(algorithm: usize, cycles: usize) -> (Vec<[f64; 4]>, Ptycho) {
    let (_, mut p) = start();
    let ap = ApParams {
        probe_inertia: Some(0.001),
        calc_llk: 1,
        zero_phase_ramp: false,
        ..ApParams::default()
    };
    match algorithm {
        0 => p.ap(&ap, cycles),
        1 => p.dm(
            &DmParams {
                calc_llk: 1,
                zero_phase_ramp: false,
                ..DmParams::default()
            },
            cycles,
        ),
        2 => p.raar(
            &RaarParams {
                calc_llk: 1,
                zero_phase_ramp: false,
                ..RaarParams::default()
            },
            cycles,
        ),
        _ => {
            p.ap(&ApParams { calc_llk: 0, ..ap }, AP_BEFORE_ML);
            p.ml(
                &MlParams {
                    update_probe: true,
                    calc_llk: 1,
                    zero_phase_ramp: false,
                    ..MlParams::default()
                },
                cycles,
            );
        }
    }
    let mut llk: Vec<[f64; 4]> = p.history().iter().map(|r| columns(&r.llk)).collect();
    if algorithm > 0 {
        llk.remove(0);
        llk.push(columns(&p.llk()));
    }
    assert_eq!(llk.len(), cycles);
    (llk, p)
}

/// Every saved run (`run exact algorithm cycles`) against PyNX's: the four LLK columns of
/// every record, and the final object and probe (DM's and RAAR's each over its `Σ|·|`, see
/// [`unit`]). The exact-kernel runs hold at rounding level; the paraxial ones at what the
/// kernel difference, printed beside them, produces.
#[test]
fn ap_dm_raar_ml_match_pynx() {
    let (case, _) = start();
    let runs = table(include_str!("engine/data/pynx_near_runs_f64.txt"));
    let trace = table(include_str!("engine/data/pynx_near_llk_f64.txt"));
    let objs = table(include_str!("engine/data/pynx_near_run_obj_f64.txt"));
    let probes = table(include_str!("engine/data/pynx_near_run_probe_f64.txt"));
    assert_eq!(runs.len(), 8);
    let names = ["AP", "DM", "RAAR", "ML"];
    // At least 5x the worst measured LLK, object or probe figure of any algorithm: exact ML's
    // LLK 4.1e-13, paraxial DM's probe 5.3e-6.
    let bounds = [5e-12, 3e-5];
    for r in &runs {
        let (exact, a, cycles) = (r[1] == 1.0, r[2] as usize, r[3] as usize);
        let what = format!("{} {}", if exact { "exact" } else { "paraxial" }, names[a]);
        let (llk, p) = engine_run(a, cycles);
        let want: Vec<f64> = (trace.iter().filter(|t| t[0] == r[0]))
            .flat_map(|t| t[2..6].to_vec())
            .collect();
        let got: Vec<f64> = llk.concat();
        // relative per column, so the photon count does not swamp the LLKs
        let llk_err = worst_of((0..4).map(|k| {
            let col = |v: &[f64]| v.iter().skip(k).step_by(4).copied().collect::<Vec<_>>();
            let (g, w) = (col(&got), col(&want));
            worst_of(g.iter().zip(&w).map(|(g, w)| (g - w).abs() / w.abs()))
        }));
        let of_run = |rows: &[Vec<f64>], lead: usize| -> Vec<C64> {
            (rows.iter().filter(|t| t[0] == r[0]))
                .map(|t| C64::new(t[lead], t[lead + 1]))
                .collect()
        };
        let (wo, wp) = (of_run(&objs, 3), of_run(&probes, 4));
        let (go, gp) = (p.object().as_slice(), p.probe().as_slice());
        let [obj_err, probe_err] = if a == 1 || a == 2 {
            [(go, &wo), (gp, &wp)].map(|(g, w)| max_rel(&unit(g), &unit(w)))
        } else {
            [(go, &wo), (gp, &wp)].map(|(g, w)| {
                let flat = |v: &[C64]| v.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<_>>();
                max_rel(&flat(g), &flat(w))
            })
        };
        eprintln!(
            "{what}: LLK {:.4} -> {:.4} (truth {:.4}); max rel LLK {llk_err:.2e}, object \
             {obj_err:.2e}, probe {probe_err:.2e}; kernel difference {:.2e} rad",
            got[0],
            got[got.len() - 4],
            case[5],
            case[6]
        );
        let bound = bounds[usize::from(!exact)];
        for (name, e) in [("LLK", llk_err), ("object", obj_err), ("probe", probe_err)] {
            assert!(e < bound, "{what} {name}: max rel {e:.3e} over {bound:.0e}");
        }
    }
}
