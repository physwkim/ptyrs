use ptyrs::base::{C64, Geometry, Object, Probe, Scan, Stack, required_obj_shape};
use ptyrs::engine::{Algorithm, ApParams, DmParams, LlkRecord, MlParams, Ptycho, RpieParams};

#[path = "common/engine/mod.rs"]
mod common;
#[path = "common/ptychi/mod.rs"]
mod ptychi;

const PHOTONS: f64 = 1e5;

struct Case {
    scan: Scan,
    object: Object,
    probe: Probe,
    /// Poisson LLK of the truth at the photon scale of the counts: the noise floor.
    truth: f64,
}

/// 40 Fermat positions, 32x32 frames, `PHOTONS` per frame.
fn case(nmode: usize, masked: bool) -> Case {
    case_with(32, nmode, masked)
}

fn case_with(frame: usize, nmode: usize, masked: bool) -> Case {
    let g = Geometry::new(1.5e-10, 1.0, 75e-6, [frame, frame]).unwrap();
    let pixel = g.pixel_size_object();
    let pos = ptyrs::base::sim::fermat(40, 2.5 * pixel[1]);
    let shape = required_obj_shape(&pos.pynx_preprocess(pixel), pixel, g.frame, 4);
    let object = ptyrs::base::sim::object(shape);
    let probe = ptyrs::base::sim::probe(g.frame, nmode).unwrap();
    let mask: Option<Vec<bool>> = masked.then(|| {
        (0..frame * frame)
            .map(|i| i % frame == 20 || i / frame == 3)
            .collect()
    });
    let sim =
        ptyrs::engine::sim::simulate_truth(g, pos, &object, &probe, PHOTONS, mask.as_deref(), 7)
            .unwrap();
    Case {
        scan: sim.scan,
        object,
        probe,
        truth: sim.truth_llk.poisson,
    }
}

fn flat_object(o: &Object) -> Object {
    Object::new(Stack::filled(o.dims(), C64::new(1.0, 0.0)))
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// With and without masked pixels, the scaled calculated photons over the unmasked pixels
/// are the observed ones there.
#[test]
fn scale_obj_probe_matches_observed_photons() {
    for masked in [true, false] {
        let c = case(1, masked);
        let mut p = Ptycho::new(&c.scan, flat_object(&c.object), c.probe.clone()).unwrap();
        p.scale_obj_probe();
        let got = p.llk().nb_photons_calc;
        assert!(
            rel(got, p.observed().total()) < 1e-12,
            "masked {masked}: {got}"
        );
        let os: f64 = p.object().as_slice().iter().map(|c| c.norm()).sum();
        let ps: f64 = p.probe().as_slice().iter().map(|c| c.norm()).sum();
        assert!(rel(os, ps) < 1e-12);
    }
}

#[test]
fn zero_phase_ramp_keeps_llk_and_removes_a_probe_tilt() {
    // 64 pixels: the defocused probe's far field stays inside the frame (on 32 it wraps and
    // biases the centre of mass by ~0.01 pixel)
    let c = case_with(64, 2, false);
    let mut reference = Ptycho::new(&c.scan, c.object.clone(), c.probe.clone()).unwrap();
    reference.zero_phase_ramp();
    // tilt the probe by whole far-field pixels (-2, 1), an exact circular shift of its far field
    let mut probe = c.probe.clone();
    let nx = probe.plane_shape()[1];
    for m in 0..2 {
        for (i, v) in probe.plane_mut(m).iter_mut().enumerate() {
            let (y, x) = ((i / nx) as f64, (i % nx) as f64);
            *v *= C64::from_polar(1.0, 2.0 * std::f64::consts::PI * (x - 2.0 * y) / nx as f64);
        }
    }
    let mut p = Ptycho::new(&c.scan, c.object.clone(), probe).unwrap();
    let before = p.llk();
    p.zero_phase_ramp();
    let after = p.llk();
    assert!(rel(after.poisson, before.poisson) < 1e-10);
    assert!(rel(after.nb_photons_calc, before.nb_photons_calc) < 1e-12);
    // the tilt is removed: the same probe as the untilted truth after the same operator, up
    // to a constant phase per mode
    for m in 0..2 {
        let (a, b) = (p.probe().plane(m), reference.probe().plane(m));
        let tot: f64 = b.iter().map(|v| v.norm_sqr()).sum();
        let k = a.iter().zip(b).map(|(u, v)| u * v.conj()).sum::<C64>() / tot;
        let err: f64 = a.iter().zip(b).map(|(u, v)| (u - v * k).norm_sqr()).sum();
        assert!(err / tot < 1e-6, "mode {m}: {}", err / tot);
    }
}

#[test]
fn dm_converges_and_records_on_schedule() {
    let c = case(1, false);
    let truth = c.truth;
    let mut p = Ptycho::new(&c.scan, flat_object(&c.object), c.probe.clone()).unwrap();
    p.scale_obj_probe();
    let start = p.llk().poisson;
    let params = DmParams {
        calc_llk: 10,
        ..DmParams::default()
    };
    p.dm(&params, 101);
    let cycles: Vec<usize> = p.history().iter().map(|r| r.cycle).collect();
    assert_eq!(cycles, (0..=100).step_by(10).collect::<Vec<_>>());
    assert!(p.history().iter().all(|r| r.algorithm == Algorithm::Dm));
    let end = p.history().last().unwrap().llk.poisson;
    assert_eq!(p.cycle(), 101);
    // DM is not an LLK minimiser and settles above the noise floor, as PyNX DM does; only
    // the progress from the start is asserted
    assert!(end < start / 10.0, "start {start} end {end} truth {truth}");
}

/// The parity scenario at 32 pixels and 40 positions, scaled.
fn parity_start() -> Ptycho {
    let (scan, object, probe) = common::dm_scenario(32, 40, 1, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    p
}

/// `table` rows: cycle, then the normalised Poisson, Gaussian, Euclidian LLK and photon count;
/// record `k` of `history` against row `k`, at the row's cycle + `lag`.
fn assert_history_matches(history: &[LlkRecord], lag: usize, table: &str) {
    let rows: Vec<Vec<f64>> = table
        .lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(history.len(), rows.len());
    for (r, row) in history.iter().zip(&rows) {
        assert_eq!(r.cycle as f64, row[0] + lag as f64);
        let l = r.llk;
        for (a, b) in [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc]
            .into_iter()
            .zip(&row[1..])
        {
            assert!(
                rel(a, *b) < 1e-9,
                "cycle {}: rust {a} reference {b}",
                r.cycle
            );
        }
    }
}

/// DM against PyNX `cpu_operator` promoted to f64 (`scripts/compare_pynx.py <dir> dm`),
/// cycles 0-19 of `cargo run --release -p ptyrs --example engine_llk_trace -- <dir> dm 20 1 32
/// 40`, saved from `<dir>/llk_pynx_dm_f64.txt` to `tests/data/dm_pynx_f64.txt`. That DM
/// records the estimate after the cycle's update, so its cycle `c` is the record of `c + 1`
/// here, and record 0 is the start.
#[test]
fn dm_llk_matches_pynx() {
    let mut p = parity_start();
    let start = p.llk();
    let params = DmParams {
        calc_llk: 1,
        ..DmParams::default()
    };
    p.dm(&params, 21);
    let (first, rest) = p.history().split_first().unwrap();
    assert_eq!((first.cycle, first.llk), (0, start));
    assert_history_matches(rest, 1, include_str!("engine/data/dm_pynx_f64.txt"));
}

/// rPIE against pty-chi `RPIEReconstructor` in f64 (`scripts/compare_ptychi_rpie.py <dir>`),
/// epochs 0-19 of `cargo run --release -p ptyrs --example engine_llk_trace -- <dir> rpie 20 1
/// 32 40`, saved from `<dir>/llk_ptychi.txt` to `tests/data/rpie_ptychi_f64.txt`.
#[test]
fn rpie_llk_matches_ptychi() {
    let mut p = parity_start();
    p.rpie(&ptychi::RPIE, 20).unwrap();
    assert_history_matches(
        p.history(),
        0,
        include_str!("engine/data/rpie_ptychi_f64.txt"),
    );
}

/// [`Ptycho::obj_grad`] and the LLK against the float64 model written independently in
/// m3/solve's `refs/f64/ml_grad.py <dir>` (checked there by central differences, PyNX
/// `cpu_operator` and pty-chi autograd) on the scaled start of `engine_llk_trace -- <dir> dm 1 2 32 40
/// 4`, saved from `<dir>/ml_grad_f64.txt` (`iy ix re im` per object pixel) and
/// `<dir>/ml_grad_values_f64.txt` (`nb_obs` and the unnormalised Poisson sum).
#[test]
fn obj_grad_matches_ml_grad_reference() {
    let (scan, object, probe) = common::dm_scenario(32, 40, 2, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let values: Vec<f64> = include_str!("engine/data/ml_grad_values_f64.txt")
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let [nb_obs, sum] = values[..] else {
        panic!("values: {values:?}")
    };
    let llk = p.llk().poisson * nb_obs;
    assert!(rel(llk, sum) < 1e-12, "LLK sum: rust {llk} reference {sum}");

    let g = p.obj_grad();
    let [_, ny, nx] = g.dims();
    let mut rows = 0;
    let (mut err, mut max) = (0f64, 0f64);
    for l in include_str!("engine/data/ml_grad_f64.txt").lines() {
        let v: Vec<f64> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
        let want = C64::new(v[2], v[3]);
        let got = g.as_slice()[v[0] as usize * nx + v[1] as usize];
        err = err.max((got - want).norm());
        max = max.max(want.norm());
        rows += 1;
    }
    assert_eq!((g.dims()[0], rows), (1, ny * nx));
    eprintln!(
        "obj_grad: max |rust - reference| / max |reference| {:.2e}",
        err / max
    );
    assert!(err < 1e-12 * max, "{err:e} of {max:e}");
}

#[test]
fn ap_object_only_converges() {
    let c = case(2, true);
    let truth = c.truth;
    let mut p = Ptycho::new(&c.scan, flat_object(&c.object), c.probe.clone()).unwrap();
    p.scale_obj_probe();
    let params = ApParams {
        calc_llk: 50,
        ..ApParams::default()
    };
    p.ap(&params, 100);
    let h = p.history();
    assert_eq!(h.iter().map(|r| r.cycle).collect::<Vec<_>>(), [0, 50, 99]);
    // AP records the LLK before the update, so it decreases along the history
    assert!(h[0].llk.poisson > h[1].llk.poisson && h[1].llk.poisson > h[2].llk.poisson);
    let end = p.llk().poisson;
    assert!(end < 1.2 * truth, "end {end} truth {truth}");
}

#[test]
fn ml_after_ap_descends_below_the_truth() {
    let c = case(2, true);
    let truth = c.truth;
    let mut p = Ptycho::new(&c.scan, flat_object(&c.object), c.probe.clone()).unwrap();
    p.scale_obj_probe();
    p.ap(&ApParams::default(), 20);
    let params = MlParams {
        update_probe: true,
        calc_llk: 1,
        ..MlParams::default()
    };
    p.ml(&params, 40).unwrap();
    let h = p.history();
    assert_eq!(
        h.iter().map(|r| r.cycle).collect::<Vec<_>>(),
        (20..60).collect::<Vec<_>>()
    );
    assert!(h.iter().all(|r| r.algorithm == Algorithm::Ml));
    // ML records the LLK before each update
    for w in h.windows(2) {
        assert!(w[1].llk.poisson < w[0].llk.poisson, "{w:?}");
    }
    // the noise is fitted too, so ML ends below the LLK of the truth
    let end = p.llk().poisson;
    assert!(
        end < truth,
        "start {} end {end} truth {truth}",
        h[0].llk.poisson
    );
}

#[test]
fn rpie_converges() {
    let c = case(1, true);
    let truth = c.truth;
    let mut p = Ptycho::new(&c.scan, flat_object(&c.object), c.probe.clone()).unwrap();
    p.scale_obj_probe();
    // pty-chi sums overlapping updates in a batch, so the default step 1 / alpha 0.1 diverges
    // with several overlapping frames per batch; alpha 1 is the setting pty-chi tests use
    let params = RpieParams {
        batch_size: 4,
        obj_alpha: 1.0,
        probe_alpha: 1.0,
        calc_llk: 1,
        ..RpieParams::default()
    };
    p.rpie(&params, 60).unwrap();
    let h = p.history();
    assert_eq!(h.len(), 60);
    assert!(h.iter().all(|r| r.algorithm == Algorithm::Rpie));
    let end = h.last().unwrap().llk.poisson;
    assert!(
        end < 1.2 * truth,
        "start {} end {end} truth {truth}",
        h[0].llk.poisson
    );
}

#[test]
fn rpie_rejects_object_modes() {
    let c = case(1, false);
    let o = Object::new(Stack::filled(
        [2, c.object.dims()[1], c.object.dims()[2]],
        C64::new(1.0, 0.0),
    ));
    let mut p = Ptycho::new(&c.scan, o, c.probe.clone()).unwrap();
    assert!(p.rpie(&RpieParams::default(), 1).is_err());
}
