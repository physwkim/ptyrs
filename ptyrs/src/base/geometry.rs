use std::fmt;
use std::ops::Index;

use crate::base::CoreError;

/// PyNX/CXI photon energy constant: `E[keV] = 12.3984 / λ[Å]`.
const KEV_ANGSTROM: f64 = 12.3984;
/// PyNX/CXI joules per keV (`runner/cxi.py` divides the CXI energy by this).
const JOULE_PER_KEV: f64 = 1.60218e-16;

/// Detector frame shape `[ny, nx]`: both sides even and non-zero, and the area `ny · nx` within
/// `usize`, so every product of the two sides downstream is in range.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Frame([usize; 2]);

impl Frame {
    pub fn new(sides: [usize; 2]) -> Result<Self, CoreError> {
        // The fftshift convention (shift == inverse shift) needs even sizes.
        if sides.iter().any(|&n| n == 0 || n % 2 != 0) {
            return Err(CoreError::Invalid(format!(
                "frame shape {sides:?} must be even and non-zero"
            )));
        }
        if sides[0].checked_mul(sides[1]).is_none() {
            return Err(CoreError::Invalid(format!(
                "frame shape {sides:?} has an area past usize"
            )));
        }
        Ok(Self(sides))
    }

    /// `[ny, nx]`.
    pub fn sides(self) -> [usize; 2] {
        self.0
    }

    /// `ny · nx`, in range by construction.
    pub fn area(self) -> usize {
        self.0[0] * self.0[1]
    }
}

/// `[ny, nx]`, as messages print a frame shape.
impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Index<usize> for Frame {
    type Output = usize;

    fn index(&self, axis: usize) -> &usize {
        &self.0[axis]
    }
}

impl PartialEq<[usize; 2]> for Frame {
    fn eq(&self, other: &[usize; 2]) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Frame> for [usize; 2] {
    fn eq(&self, other: &Frame) -> bool {
        *self == other.0
    }
}

/// Far-field geometry with square detector pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// Wavelength in metres.
    pub wavelength: f64,
    /// Sample-detector distance in metres.
    pub distance: f64,
    /// Detector pixel size in metres (square pixels, as PyNX assumes).
    pub detector_pixel: f64,
    /// Frame shape `[ny, nx]` in detector pixels.
    pub frame: Frame,
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
        let g = Self {
            wavelength,
            distance,
            detector_pixel,
            frame: Frame::new(frame)?,
        };
        // Each input can be in range while λz / (n · detector_pixel) underflows, overflows or
        // is inf / inf.
        let pixel = g.pixel_size_object();
        if !pixel.iter().all(|&p| p.is_finite() && p > 0.0) {
            return Err(CoreError::Invalid(format!(
                "object pixel size {pixel:?} must be finite and > 0"
            )));
        }
        Ok(g)
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
        assert!(Frame::new([256, 3]).is_err());
    }

    #[test]
    fn rejects_zero_frame() {
        assert!(Frame::new([0, 256]).is_err());
        assert!(Frame::new([256, 0]).is_err());
    }

    /// Each side is even and fits, their product does not.
    #[test]
    fn rejects_frame_area_past_usize() {
        let side = 1 << (usize::BITS / 2);
        let e = Frame::new([side, side]).unwrap_err();
        assert!(e.to_string().contains("area past usize"), "{e}");
        assert_eq!(
            Frame::new([side, side / 2]).unwrap().area(),
            usize::MAX / 2 + 1
        );
    }

    fn rejects_pixel(wavelength: f64, distance: f64, detector_pixel: f64) {
        let e = Geometry::new(wavelength, distance, detector_pixel, [256, 256]).unwrap_err();
        assert!(e.to_string().contains("object pixel size"), "{e}");
    }

    /// λz underflows to 0.
    #[test]
    fn rejects_zero_object_pixel() {
        rejects_pixel(1e-200, 1e-200, 75e-6);
    }

    /// λz overflows to inf.
    #[test]
    fn rejects_infinite_object_pixel() {
        rejects_pixel(1e200, 1e200, 75e-6);
    }

    /// λz and n · detector_pixel both overflow: inf / inf.
    #[test]
    fn rejects_nan_object_pixel() {
        rejects_pixel(1e200, 1e200, 1e308);
    }
}
