//! Simulated scans from a known object and probe through the same far-field model the
//! reconstruction uses (PyNX `Calc2Obs`).

use ptyrs_core::{AddrTable, C64, Counts, Geometry, Object, Positions, Probe, Scan};

use crate::{EngineError, Fft2, Llk, Observed, llk, shift};

/// A [`simulate_truth`] scan and the LLK of the truth it was drawn from.
#[derive(Clone, Debug)]
pub struct Simulation {
    pub scan: Scan,
    /// Normalised [`llk`] of the mean intensity the counts were drawn from, which is
    /// [`crate::Ptycho::llk`] of the object and the probe scaled to the counts.
    pub truth_llk: Llk,
}

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
    let (scan, ..) = draw(
        geometry,
        positions,
        object,
        probe,
        photons_per_frame,
        detector_mask,
        seed,
    )?;
    Ok(scan)
}

/// [`simulate`] and the truth's LLK, taken from the intensity the counts are drawn from.
pub fn simulate_truth(
    geometry: Geometry,
    positions: Positions,
    object: &Object,
    probe: &Probe,
    photons_per_frame: f64,
    detector_mask: Option<&[bool]>,
    seed: u64,
) -> Result<Simulation, EngineError> {
    let (scan, mut psi, nmode, scale) = draw(
        geometry,
        positions,
        object,
        probe,
        photons_per_frame,
        detector_mask,
        seed,
    )?;
    let obs = Observed::from_counts(&scan.counts);
    let r = scale.sqrt();
    psi.iter_mut().for_each(|v| *v *= r);
    let truth_llk = llk(&obs, 0..obs.nframe(), nmode, &psi).normalised(obs.nb_obs());
    Ok(Simulation { scan, truth_llk })
}

/// [`simulate`]'s scan, the far field of the unscaled truth it was drawn from, the modes per
/// frame of that field and the intensity scale.
fn draw(
    geometry: Geometry,
    positions: Positions,
    object: &Object,
    probe: &Probe,
    photons_per_frame: f64,
    detector_mask: Option<&[bool]>,
    seed: u64,
) -> Result<(Scan, Vec<C64>, usize, f64), EngineError> {
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
    Ok((Scan::new(geometry, positions, counts)?, psi, nmode, scale))
}

#[cfg(test)]
mod tests {
    use ptyrs_core::required_obj_shape;

    use super::*;
    use crate::Ptycho;

    /// The truth LLK is the CPU LLK of the probe scaled by the flux factor, which is taken
    /// over every pixel while the LLK skips masked ones.
    #[test]
    fn truth_llk_is_the_llk_of_the_scaled_truth() {
        let g = Geometry::new(1.5e-10, 1.0, 75e-6, [16, 16]).unwrap();
        let pixel = g.pixel_size_object();
        let positions = ptyrs_core::sim::fermat(5, 3.0 * pixel[1]);
        let shape = required_obj_shape(&positions.pynx_preprocess(pixel), pixel, g.frame, 0);
        let object = ptyrs_core::sim::object(shape);
        let probe = ptyrs_core::sim::probe(g.frame, 2).unwrap();
        let photons = 2e3;
        let sim =
            |mask| simulate_truth(g, positions.clone(), &object, &probe, photons, mask, 3).unwrap();
        let unmasked = sim(None).scan;
        let calc = Ptycho::new(&unmasked, object.clone(), probe.clone()).unwrap();
        let r = (photons * 5.0 / calc.llk().nb_photons_calc).sqrt();
        let mut scaled = probe.clone();
        scaled.as_mut_slice().iter_mut().for_each(|v| *v *= r);
        let mut mask = vec![false; 256];
        mask[8 * 16 + 9] = true;
        mask[3 * 16 + 5] = true;
        for mask in [None, Some(&mask[..])] {
            let s = sim(mask);
            let want = Ptycho::new(&s.scan, object.clone(), scaled.clone())
                .unwrap()
                .llk();
            let got = s.truth_llk;
            for (g, w) in [
                (got.poisson, want.poisson),
                (got.gaussian, want.gaussian),
                (got.euclidian, want.euclidian),
                (got.nb_photons_calc, want.nb_photons_calc),
            ] {
                assert!(
                    (g - w).abs() <= 1e-12 * w.abs(),
                    "mask {}: {g} vs {w}",
                    mask.is_some()
                );
            }
        }
    }
}
