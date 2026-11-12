//! The scans of `nn_train_ptychonn`, shared so that other examples measure on the same held-out
//! scans.

use ptyrs::base::{AddrTable, Geometry, Object, Positions, Probe, Scan, required_obj_shape, sim};
use ptyrs::engine::Ptycho;
use ptyrs::nn::data::random_object;

/// The engine's parity geometry (`ptyrs/tests/engine/common`).
fn geometry(frame: usize) -> Geometry {
    Geometry::new(1.5e-10, 1.0, 75e-6, [frame, frame]).unwrap()
}

/// The scan positions: 100 on a Fermat spiral, 4 object pixels apart.
fn positions(g: &Geometry) -> Positions {
    sim::fermat(100, 4.0 * g.pixel_size_object()[1])
}

/// The object shape [`scan`] needs.
pub fn object_shape(frame: usize) -> [usize; 2] {
    let g = geometry(frame);
    let pixel = g.pixel_size_object();
    required_obj_shape(&positions(&g).pynx_preprocess(pixel), pixel, g.frame, 8)
}

/// A 100-point Fermat scan, 4 object pixels apart, of `object` (random when `None`).
pub fn scan(
    frame: usize,
    probe: &Probe,
    seed: u64,
    object: Option<Object>,
) -> (Scan, Object, AddrTable) {
    let g = geometry(frame);
    let pixel = g.pixel_size_object();
    let positions = positions(&g);
    let pre = positions.pynx_preprocess(pixel);
    let object = object.unwrap_or_else(|| random_object(object_shape(frame), seed));
    let addr = AddrTable::new(&pre, pixel, g.frame, object.plane_shape(), 1, 1).unwrap();
    let scan = ptyrs::engine::sim::simulate(g, positions, &object, probe, 1e6, None, seed).unwrap();
    (scan, object, addr)
}

/// The held-out scans as `(name, seed, scan, truth, addresses, probe)`: two of random objects
/// and one of the engine's test object, each with the probe that reproduces its counts from the
/// truth, since the simulator scales the intensity to its photon count.
pub fn held_out(
    frame: usize,
    probe: &Probe,
) -> Vec<(&'static str, u64, Scan, Object, AddrTable, Probe)> {
    let shape = {
        let (_, o, _) = scan(frame, probe, 0, None);
        o.plane_shape()
    };
    [
        ("random", None, 2_000_000),
        ("random", None, 2_000_001),
        ("sim::object", Some(sim::object(shape)), 2_000_002),
    ]
    .into_iter()
    .map(|(name, object, seed)| {
        let (scan, truth, addr) = scan(frame, probe, seed, object);
        let p = Ptycho::new(&scan, truth.clone(), probe.clone()).unwrap();
        let s = (p.observed().total() / p.llk().nb_photons_calc).sqrt();
        let mut known = probe.clone();
        known.as_mut_slice().iter_mut().for_each(|c| *c *= s);
        (name, seed, scan, truth, addr, known)
    })
    .collect()
}
