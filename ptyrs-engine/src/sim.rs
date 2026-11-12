//! Simulated scans from a known object and probe through the same far-field model the
//! reconstruction uses (PyNX `Calc2Obs`).

use ptyrs_core::{AddrTable, C64, Counts, Geometry, Object, Positions, Probe, Scan};

use crate::{EngineError, Fft2, shift};

/// PyNX `Calc2Obs(nb_photons_per_frame, poisson_noise=True)`: `icalc = Σ_modes |FT(P·O·q)|²`
/// per frame, scaled so the mean frame holds `photons_per_frame`, then Poisson-sampled with
/// `seed` ([`ptyrs_core::sim::poisson`]). `positions` are the raw positions the [`Scan`]
/// keeps; they are preprocessed as [`crate::Ptycho::new`] does. `detector_mask` (`[ny, nx]`,
/// true = masked) masks pixels after the scaling.
pub fn simulate(
    geometry: Geometry,
    positions: Positions,
    object: &Object,
    probe: &Probe,
    photons_per_frame: f64,
    detector_mask: Option<&[bool]>,
    seed: u64,
) -> Result<Scan, EngineError> {
    if probe.plane_shape() != geometry.frame {
        return Err(EngineError::Invalid(format!(
            "probe plane {:?} differs from frame {:?}",
            probe.plane_shape(),
            geometry.frame
        )));
    }
    let pixel = geometry.pixel_size_object();
    let addr = AddrTable::new(
        &positions.pynx_preprocess(pixel),
        pixel,
        geometry.frame,
        object.plane_shape(),
        probe.nmode(),
        object.nmode(),
    )?;
    let [ny, nx] = geometry.frame;
    let n = ny * nx;
    let mut psi = vec![C64::default(); addr.views().len() * n];
    crate::obj_probe_to_psi(
        addr.views(),
        object,
        probe,
        &crate::quad_phase(&geometry),
        &mut psi,
    );
    Fft2::new(geometry.frame).forward(&mut psi);
    let (nframe, nmode) = (addr.nframe(), addr.nmode());
    // detector order: natural pixel i is shifted pixel shift(i)
    let mut intensity = vec![0.0; nframe * n];
    for (out, block) in intensity
        .chunks_exact_mut(n)
        .zip(psi.chunks_exact(nmode * n))
    {
        for (i, out) in out.iter_mut().enumerate() {
            let j = shift(i, ny, nx);
            *out = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
        }
    }
    let scale = photons_per_frame / (intensity.iter().sum::<f64>() / nframe as f64);
    intensity.iter_mut().for_each(|v| *v *= scale);
    let counts: Counts =
        ptyrs_core::sim::poisson([nframe, ny, nx], &intensity, detector_mask, seed)?;
    Ok(Scan::new(geometry, positions, counts)?)
}
