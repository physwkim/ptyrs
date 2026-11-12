//! The M6 measure on the engine's rPIE (CPU), with the true object standing in for a perfect
//! prediction.

use ptyrs_core::{AddrTable, Geometry, Object, Probe, Scan, required_obj_shape, sim};
use ptyrs_engine::{EngineError, Ptycho, RpieParams};
use ptyrs_nn::data::random_object;
use ptyrs_nn::metric::iterations_to_llk;
use ptyrs_nn::predictor::flat_object;

const FRAME: usize = 64;

/// A simulated scan, its truth and address table, and the probe scaled to its counts.
fn scan() -> (Scan, Object, AddrTable, Probe) {
    let g = Geometry::new(1.5e-10, 1.0, 75e-6, [FRAME, FRAME]).unwrap();
    let pixel = g.pixel_size_object();
    let positions = sim::fermat(40, 4.0 * pixel[1]);
    let pre = positions.pynx_preprocess(pixel);
    let shape = required_obj_shape(&pre, pixel, g.frame, 8);
    let object = random_object(shape, 11);
    let addr = AddrTable::new(&pre, pixel, g.frame, shape, 1, 1).unwrap();
    let mut probe = sim::probe([FRAME, FRAME], 1).unwrap();
    let scan = ptyrs_engine::sim::simulate(g, positions, &object, &probe, 1e5, None, 11).unwrap();
    // The simulator scales the intensity to the photon count; the probe has to follow.
    let p = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    let s = (p.observed().total() / p.llk().nb_photons_calc).sqrt();
    probe.as_mut_slice().iter_mut().for_each(|c| *c *= s);
    (scan, object, addr, probe)
}

/// rPIE epochs, probe fixed, from `start` to the Poisson LLK `target`. Single-frame batches: the
/// summed updates of 10 overlapping frames diverge on this scan.
fn epochs(
    scan: &Scan,
    start: Object,
    probe: &Probe,
    target: f64,
    max: usize,
) -> Result<Option<usize>, EngineError> {
    let params = RpieParams {
        probe_step: None,
        batch_size: 1,
        ..RpieParams::default()
    };
    let mut p = Ptycho::new(scan, start, probe.clone())?;
    iterations_to_llk(p.llk().poisson, target, max, || {
        p.rpie(&params, 1)?;
        Ok(p.llk().poisson)
    })
}

#[test]
fn flat_start_needs_epochs_the_true_object_does_not() {
    let (scan, truth, addr, probe) = scan();
    let llk = |o: Object| Ptycho::new(&scan, o, probe.clone()).unwrap().llk().poisson;
    let (at_truth, at_flat) = (llk(truth.clone()), llk(flat_object(&addr)));
    // From flat, rPIE stalls near 1.45 times the truth's LLK and passes 1.75 times it in a few
    // epochs.
    let target = 1.75 * at_truth;
    let flat = epochs(&scan, flat_object(&addr), &probe, target, 50).unwrap();
    let k = flat.unwrap_or_else(|| panic!("LLK truth {at_truth}, flat {at_flat}: never {target}"));
    assert!(k >= 2, "flat start reached the target in {k} epoch");
    assert_eq!(epochs(&scan, truth, &probe, target, 50).unwrap(), Some(0));
    assert_eq!(
        epochs(&scan, flat_object(&addr), &probe, target, k - 1).unwrap(),
        None
    );
}
