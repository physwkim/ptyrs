//! The M6 measure on the LLK history of the engine's rPIE (CPU).

use ptyrs::base::{AddrTable, Geometry, Object, Probe, Scan, required_obj_shape, sim};
use ptyrs::engine::{Ptycho, RpieParams};
use ptyrs::nn::data::random_object;
use ptyrs::nn::metric::iterations_to_llk;
use ptyrs::nn::predictor::flat_object;

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
    let scan = ptyrs::engine::sim::simulate(g, positions, &object, &probe, 1e5, None, 11).unwrap();
    // The simulator scales the intensity to the photon count; the probe has to follow.
    let p = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    let s = (p.observed().total() / p.llk().nb_photons_calc).sqrt();
    probe.as_mut_slice().iter_mut().for_each(|c| *c *= s);
    (scan, object, addr, probe)
}

/// The LLK of `start` and after each of `epochs` rPIE epochs, probe fixed. Single-frame
/// batches: the summed updates of 10 overlapping frames diverge on this scan.
fn rpie(scan: &Scan, start: Object, probe: &Probe, epochs: usize) -> (f64, Vec<f64>) {
    let params = RpieParams {
        probe_step: None,
        batch_size: 1,
        calc_llk: 1,
        ..RpieParams::default()
    };
    let mut p = Ptycho::new(scan, start, probe.clone()).unwrap();
    let start = p.llk().poisson;
    p.rpie(&params, epochs).unwrap();
    let history: Vec<f64> = p.history().iter().map(|r| r.llk.poisson).collect();
    assert_eq!(history.len(), epochs);
    (start, history)
}

#[test]
fn the_flat_start_reaches_a_target_near_the_truth_after_epochs() {
    let (scan, truth, addr, probe) = scan();
    let at_truth = Ptycho::new(&scan, truth, probe.clone())
        .unwrap()
        .llk()
        .poisson;
    let (at_flat, flat) = rpie(&scan, flat_object(&addr), &probe, 50);
    // From flat, rPIE stalls near 1.45 times the truth's LLK and passes 1.75 times it in a few
    // epochs.
    let target = 1.75 * at_truth;
    let k = iterations_to_llk(at_flat, target, 0.0, flat.iter().copied())
        .unwrap_or_else(|| panic!("LLK truth {at_truth}, flat {at_flat} then {flat:?}"));
    assert!(k >= 2, "flat start reached the target in {k} epoch");
    assert_eq!(
        iterations_to_llk(at_flat, target, 0.0, flat[..k - 1].iter().copied()),
        None
    );
}
