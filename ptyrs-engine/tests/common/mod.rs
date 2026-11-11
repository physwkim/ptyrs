//! The parity scenario, shared by `tests/reconstruct.rs` and `examples/llk_trace.rs` so the
//! PyNX values in `tests/data` stay reproducible from the example.

use ptyrs_core::{C64, Geometry, Object, Positions, Probe, Scan, Stack, required_obj_shape};

/// A simulated scan (`npos` Fermat positions 4 object pixels apart, `photons` per frame,
/// seed 1, no mask) and its starting object and probe, before `scale_obj_probe`.
///
/// The object starts at unit modulus with a uniform random phase in [-0.5, 0.5) rad, the
/// probe as the truth with its defocus phase 1.5 r²/σ² cut to 1.0 r²/σ². A flat object would
/// leave the far field of the Gaussian probe below the 1e-12 floor of `apply_amplitude` in
/// pixels where the data has counts; there it is FFT rounding noise and no two
/// implementations agree. The random phase scatters into every pixel.
pub fn dm_scenario(
    frame: usize,
    npos: usize,
    nprobe: usize,
    photons: f64,
) -> (Scan, Object, Probe) {
    let g = Geometry::new(1.5e-10, 1.0, 75e-6, [frame, frame]).unwrap();
    let pixel = g.pixel_size_object();
    // CXI stores translations as f32: round first so a CXI round trip keeps them exactly
    let f = ptyrs_core::sim::fermat(npos, 4.0 * pixel[1]);
    let round = |v: Vec<f64>| v.into_iter().map(|x| x as f32 as f64).collect();
    let positions = Positions::new(round(f.y), round(f.x)).unwrap();
    let shape = required_obj_shape(&positions.pynx_preprocess(pixel), pixel, g.frame, 16);
    let truth_obj = ptyrs_core::sim::object(shape);
    let truth_probe = ptyrs_core::sim::probe(g.frame, nprobe).unwrap();
    let scan =
        ptyrs_engine::sim::simulate(g, positions, &truth_obj, &truth_probe, photons, None, 1)
            .unwrap();

    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let phase: Vec<C64> = (0..truth_obj.as_slice().len())
        .map(|_| C64::from_polar(1.0, uniform() - 0.5))
        .collect();
    let object = Object::new(Stack::from_vec(truth_obj.dims(), phase).unwrap());
    let sigma = frame as f64 / 12.0;
    let mut probe = truth_probe;
    for m in 0..probe.nmode() {
        for (i, v) in probe.plane_mut(m).iter_mut().enumerate() {
            let u = ((i % frame) as f64 - frame as f64 / 2.0) / sigma;
            let w = ((i / frame) as f64 - frame as f64 / 2.0) / sigma;
            *v *= C64::from_polar(1.0, -0.5 * (u * u + w * w));
        }
    }
    (scan, object, probe)
}
