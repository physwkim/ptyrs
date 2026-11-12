use crate::base::CoreError;

/// PyNX/CXI photon energy constant: `E[keV] = 12.3984 / λ[Å]`.
const KEV_ANGSTROM: f64 = 12.3984;
/// PyNX/CXI joules per keV (`runner/cxi.py` divides the CXI energy by this).
const JOULE_PER_KEV: f64 = 1.60218e-16;

/// Far-field geometry with square detector pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// Wavelength in metres.
    pub wavelength: f64,
    /// Sample-detector distance in metres.
    pub distance: f64,
    /// Detector pixel size in metres (square pixels, as PyNX assumes).
    pub detector_pixel: f64,
    /// Frame shape `[ny, nx]` in detector pixels; both even.
    pub frame: [usize; 2],
}

impl Geometry {
    pub fn new(
        wavelength: f64,
        distance: f64,
        detector_pixel: f64,
        frame: [usize; 2],
    ) -> Result<Self, CoreError> {
        for (name, v) in [
            ("wavelength", wavelength),
            ("distance", distance),
            ("detector_pixel", detector_pixel),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(CoreError::Invalid(format!(
                    "{name} must be finite and > 0, got {v}"
                )));
            }
        }
        // The fftshift convention (shift == inverse shift) needs even sizes.
        if frame.iter().any(|&n| n == 0 || n % 2 != 0) {
            return Err(CoreError::Invalid(format!(
                "frame shape {frame:?} must be even and non-zero"
            )));
        }
        Ok(Self {
            wavelength,
            distance,
            detector_pixel,
            frame,
        })
    }

    /// Object-plane pixel size `[py, px] = λz / (n · detector_pixel)` per axis
    /// (PyNX `PtychoData.pixel_size_object`, which returns `(px, py)`).
    pub fn pixel_size_object(&self) -> [f64; 2] {
        let lz = self.wavelength * self.distance;
        [
            lz / (self.frame[0] as f64 * self.detector_pixel),
            lz / (self.frame[1] as f64 * self.detector_pixel),
        ]
    }

    /// Photon energy in joules as PyNX writes it to CXI `source_1/energy`.
    pub fn energy_joule(&self) -> f64 {
        KEV_ANGSTROM * 1e-10 / self.wavelength * JOULE_PER_KEV
    }

    /// Wavelength in metres from a CXI `source_1/energy` in joules (PyNX `runner/cxi.py`).
    pub fn wavelength_from_energy_joule(energy: f64) -> f64 {
        KEV_ANGSTROM * 1e-10 / (energy / JOULE_PER_KEV)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_round_trip_and_pixel_size() {
        let g = Geometry::new(1.5e-10, 1.0, 75e-6, [256, 128]).unwrap();
        let w = Geometry::wavelength_from_energy_joule(g.energy_joule());
        assert!((w - g.wavelength).abs() < 1e-24);
        let [py, px] = g.pixel_size_object();
        assert!((py - 1.5e-10 / (256.0 * 75e-6)).abs() < 1e-20);
        assert!((px - 2.0 * py).abs() < 1e-20);
    }

    #[test]
    fn rejects_odd_frame() {
        assert!(Geometry::new(1e-10, 1.0, 1e-4, [255, 256]).is_err());
    }
}
