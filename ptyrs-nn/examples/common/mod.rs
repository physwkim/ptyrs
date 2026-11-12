//! The scans of `train_ptychonn`, shared so that other examples measure on the same held-out
//! scans.

use ptyrs_core::{AddrTable, Geometry, Object, Probe, Scan, required_obj_shape, sim};
use ptyrs_nn::data::random_object;

/// The engine's parity geometry (`ptyrs-engine/tests/common`).
fn geometry(frame: usize) -> Geometry {
    Geometry::new(1.5e-10, 1.0, 75e-6, [frame, frame]).unwrap()
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
    let positions = sim::fermat(100, 4.0 * pixel[1]);
    let pre = positions.pynx_preprocess(pixel);
    let shape = required_obj_shape(&pre, pixel, g.frame, 8);
    let object = object.unwrap_or_else(|| random_object(shape, seed));
    let addr = AddrTable::new(&pre, pixel, g.frame, object.plane_shape(), 1, 1).unwrap();
    let scan = ptyrs_engine::sim::simulate(g, positions, &object, probe, 1e6, None, seed).unwrap();
    (scan, object, addr)
}

/// Held-out scans as `(name, object, seed)` for [`scan`]: two random objects and the engine's
/// test object.
pub fn held_out(frame: usize, probe: &Probe) -> Vec<(&'static str, Option<Object>, u64)> {
    let shape = {
        let (_, o, _) = scan(frame, probe, 0, None);
        o.plane_shape()
    };
    vec![
        ("random", None, 2_000_000),
        ("random", None, 2_000_001),
        ("sim::object", Some(sim::object(shape)), 2_000_002),
    ]
}
