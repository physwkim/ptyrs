//! [`Ptycho`] under a Fresnel transform back (a negative distance) by reduction to the far field.
//! The back transform is `x ↦ a·Fft2⁻¹(b·x)` with `a`, `b` the conjugates of the same distance
//! forward's `a₊`, `b₊`, and the unitary `Fft2⁻¹` is `conj·Fft2·conj`, so it is
//! `conj·(a₊·Fft2(b₊·conj(x)))`: the far field over the object-plane phase `b₊`
//! ([`Optics::far_field`] at the forward distance), its detector factor `a₊` unseen by every
//! intensity and by the modulus projection, between two conjugations. Every update here is built
//! from products with one factor conjugated and real weights, so a run back from `(O, P)` ends
//! at the conjugate of the far-field run from `(conj O, conj P)` on the same counts. Written with
//! `Fft2⁻¹` as `Fft2` and a reversed frame index instead, the far field would need the phase
//! `conj(b₊)`, a negative distance, which [`Geometry`] refuses.
//!
//! The state is `refs/f64/rpie_near.py`'s Fresnel-back case: its counts, whole-pixel corners and
//! start, saved as `rpie_near_*_f64.txt` in `tests/engine/data`.

use ptyrs::base::{C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack};
use ptyrs::engine::{
    ApParams, DmParams, Llk, MlParams, Optics, Propagator, Ptycho, RaarParams, RpieParams,
    Wavefield,
};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const SHAPE: [usize; 2] = [24, 24];
const OBJ: [usize; 2] = [40, 40];
const NPROBE: usize = 2;
/// `rpie_near_cases_f64.txt`'s Fresnel transform back.
const CASE: usize = 2;
const CYCLES: usize = 20;
const AP_BEFORE_ML: usize = 10;

fn table(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

fn stack(dims: [usize; 3], rows: &[Vec<f64>], lead: usize) -> Stack<C64> {
    let data = rows
        .iter()
        .map(|r| C64::new(r[lead], r[lead + 1]))
        .collect();
    Stack::from_vec(dims, data).unwrap()
}

fn conj(s: &Stack<C64>) -> Stack<C64> {
    let data = s.as_slice().iter().map(|c| c.conj()).collect();
    Stack::from_vec(s.dims(), data).unwrap()
}

/// `max |a − b| / max |b|` over complex values.
fn max_rel(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let d = worst_of(a.iter().zip(b).map(|(a, b)| (a - b).norm()));
    d / b.iter().map(|b| b.norm()).fold(0.0, f64::max)
}

/// The case's start under the Fresnel transform back (`far` false) or the far field over its
/// forward's plane phase, conjugated (`far` true).
fn start(far: bool) -> Ptycho {
    let c = table(include_str!("engine/data/rpie_near_cases_f64.txt")).remove(CASE);
    let corners = table(include_str!("engine/data/rpie_near_corners_f64.txt"));
    let (wavelength, pixel, distance) = (c[1], c[2], c[3]);
    assert!(distance < 0.0);
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
    // The far field whose object pixel is the plane's: λ·|z| / (n·detector pixel) = pixel.
    let lz = wavelength * -distance;
    let geometry =
        Geometry::new(wavelength, -distance, lz / (SHAPE[1] as f64 * pixel), SHAPE).unwrap();
    let scan = Scan::new(geometry, positions, counts).unwrap();
    let object = stack(
        [1, OBJ[0], OBJ[1]],
        &table(include_str!("engine/data/rpie_near_obj_f64.txt")),
        2,
    );
    let probe = stack(
        [NPROBE, SHAPE[0], SHAPE[1]],
        &table(include_str!("engine/data/rpie_near_probe_f64.txt")),
        3,
    );
    let p = if far {
        let optics = Optics::far_field(&geometry);
        Ptycho::with_optics(
            &scan,
            optics,
            Object::new(conj(&object)),
            Probe::new(conj(&probe)),
        )
    } else {
        let w = Wavefield::new(SHAPE, wavelength, [pixel, pixel], distance).unwrap();
        assert!(matches!(Propagator::free_space(&w), Propagator::Fresnel(_)));
        Ptycho::with_optics(
            &scan,
            Optics::free_space(&w),
            Object::new(object),
            Probe::new(probe),
        )
    }
    .unwrap();
    for (f, r) in corners.iter().enumerate() {
        let v = p.addr().frame_views(f)[0];
        assert_eq!([v.oy as f64, v.ox as f64], [r[1], r[2]], "frame {f}");
    }
    p
}

fn columns(l: &Llk) -> [f64; 4] {
    [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc]
}

/// Run `algorithm` (0 AP, 1 DM, 2 RAAR, 3 ML after AP, 4 rPIE as `rpie_near.py`'s batches of
/// 4) from [`start`]: its LLK records, final object and probe. No zero phase ramp: it centres
/// the probe's spectrum, whose Nyquist row the conjugation does not mirror.
fn run(algorithm: usize, far: bool) -> (Vec<[f64; 4]>, Ptycho) {
    let mut p = start(far);
    let ap = ApParams {
        probe_inertia: Some(0.001),
        calc_llk: 1,
        zero_phase_ramp: false,
        ..ApParams::default()
    };
    match algorithm {
        0 => p.ap(&ap, CYCLES),
        1 => p.dm(
            &DmParams {
                calc_llk: 1,
                zero_phase_ramp: false,
                ..DmParams::default()
            },
            CYCLES,
        ),
        2 => p.raar(
            &RaarParams {
                calc_llk: 1,
                zero_phase_ramp: false,
                ..RaarParams::default()
            },
            CYCLES,
        ),
        3 => {
            p.ap(&ApParams { calc_llk: 0, ..ap }, AP_BEFORE_ML);
            p.ml(
                &MlParams {
                    update_probe: true,
                    calc_llk: 1,
                    zero_phase_ramp: false,
                    ..MlParams::default()
                },
                CYCLES,
            );
        }
        _ => p
            .rpie(
                &RpieParams {
                    obj_step: Some(0.25),
                    batch_size: 4,
                    calc_llk: 1,
                    ..RpieParams::default()
                },
                CYCLES,
            )
            .unwrap(),
    }
    let mut llk: Vec<[f64; 4]> = p.history().iter().map(|r| columns(&r.llk)).collect();
    llk.push(columns(&p.llk()));
    (llk, p)
}

/// Every algorithm back against the conjugate of its far-field run: the four LLK columns of
/// every record and the final, and the final object and probe.
#[test]
fn fresnel_back_is_the_conjugate_far_field() {
    let names = ["AP", "DM", "RAAR", "ML", "rPIE"];
    // At least 5x the worst measured LLK, object or probe figure of any algorithm, ML's object
    // 6.2e-14.
    let bound = 1e-12;
    for (a, name) in names.iter().enumerate() {
        let (back_llk, back) = run(a, false);
        let (far_llk, far) = run(a, true);
        assert_eq!(back_llk.len(), far_llk.len(), "{name}");
        let llk_err = worst_of(
            back_llk
                .iter()
                .flatten()
                .zip(far_llk.iter().flatten())
                .map(|(b, f)| (b - f).abs() / f.abs()),
        );
        let obj_err = max_rel(back.object().as_slice(), conj(far.object()).as_slice());
        let probe_err = max_rel(back.probe().as_slice(), conj(far.probe()).as_slice());
        eprintln!(
            "{name}: LLK {:.4} -> {:.4}; back vs conjugate far field, max rel LLK {llk_err:.2e}, \
             object {obj_err:.2e}, probe {probe_err:.2e}",
            back_llk[0][0],
            back_llk[back_llk.len() - 1][0]
        );
        for (what, e) in [("LLK", llk_err), ("object", obj_err), ("probe", probe_err)] {
            assert!(e < bound, "{name} {what}: max rel {e:.3e} over {bound:.0e}");
        }
    }
}
