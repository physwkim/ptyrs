use crate::CoreError;

/// Mask sentinel in [`Counts`] (DESIGN §5 invariant 3). It is also the bad-pixel value 16-bit
/// Eiger/Jungfrau streams use, so raw detector `u16` data maps onto it unchanged.
pub const MASKED: u16 = u16::MAX;
/// Largest photon count the `u16` path holds; larger data needs the f32 path (DESIGN B5).
pub const MAX_COUNT: u16 = u16::MAX - 1;

/// Measured photon counts `[nframe, ny, nx]`, detector order (zero frequency at `[ny/2, nx/2]`).
///
/// Immutable after construction: there is no mutable accessor, so the measured data cannot
/// change once a scan is built (invariant 3). A pixel is masked iff it holds [`MASKED`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counts {
    dims: [usize; 3],
    data: Vec<u16>,
}

impl Counts {
    /// Takes raw `u16` counts; every [`MASKED`] value is a masked pixel.
    pub fn from_raw(dims: [usize; 3], data: Vec<u16>) -> Result<Self, CoreError> {
        let want: usize = dims.iter().product();
        if data.len() != want {
            return Err(CoreError::Shape(format!(
                "counts {dims:?} need {want} values, got {}",
                data.len()
            )));
        }
        Ok(Self { dims, data })
    }

    /// Converts PyNX-style intensities: a negative value is masked (PyNX `iobs < 0`), every
    /// other value must be an integer in `0..=MAX_COUNT`. `detector_mask` (`[ny, nx]`, true =
    /// masked) masks a pixel in every frame, like CXI `detector_1/mask`.
    pub fn from_f64(
        dims: [usize; 3],
        values: &[f64],
        detector_mask: Option<&[bool]>,
    ) -> Result<Self, CoreError> {
        let want: usize = dims.iter().product();
        let plane = dims[1] * dims[2];
        if values.len() != want {
            return Err(CoreError::Shape(format!(
                "counts {dims:?} need {want} values, got {}",
                values.len()
            )));
        }
        if let Some(m) = detector_mask
            && m.len() != plane
        {
            return Err(CoreError::Shape(format!(
                "detector mask needs {plane} values, got {}",
                m.len()
            )));
        }
        let mut data = Vec::with_capacity(want);
        for (index, &value) in values.iter().enumerate() {
            let masked = detector_mask.is_some_and(|m| m[index % plane]);
            data.push(if masked || value < 0.0 {
                MASKED
            } else if !value.is_finite() || value > f64::from(MAX_COUNT) {
                return Err(CoreError::CountOutOfRange {
                    index,
                    value,
                    max: MAX_COUNT,
                });
            } else if value.fract() != 0.0 {
                return Err(CoreError::NonInteger { index, value });
            } else {
                value as u16
            });
        }
        Ok(Self { dims, data })
    }

    /// Observed intensity of one stored value, `None` when masked.
    #[inline]
    pub fn intensity(value: u16) -> Option<f64> {
        (value != MASKED).then_some(f64::from(value))
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn nframe(&self) -> usize {
        self.dims[0]
    }

    pub fn frame_shape(&self) -> [usize; 2] {
        [self.dims[1], self.dims[2]]
    }

    pub fn frame(&self, i: usize) -> &[u16] {
        let n = self.dims[1] * self.dims[2];
        &self.data[i * n..(i + 1) * n]
    }

    pub fn as_slice(&self) -> &[u16] {
        &self.data
    }

    /// Number of unmasked pixels over all frames (PyNX `nb_obs = (iobs >= 0).sum()`).
    pub fn nb_obs(&self) -> usize {
        self.data.iter().filter(|&&v| v != MASKED).count()
    }

    /// `[ny, nx]` pixels masked in every frame (the part a CXI `detector_1/mask` can carry).
    pub fn detector_mask(&self) -> Vec<bool> {
        let plane = self.dims[1] * self.dims[2];
        (0..plane)
            .map(|p| (0..self.dims[0]).all(|f| self.data[f * plane + p] == MASKED))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_masks_negative_and_detector_mask() {
        let v = [0.0, 3.0, -1.0, 7.0, 1.0, 2.0, 3.0, 4.0];
        let m = [false, false, false, true];
        let c = Counts::from_f64([2, 2, 2], &v, Some(&m)).unwrap();
        assert_eq!(c.as_slice(), &[0, 3, MASKED, MASKED, 1, 2, 3, MASKED]);
        assert_eq!(c.nb_obs(), 5);
        assert_eq!(c.detector_mask(), vec![false, false, false, true]);
        assert_eq!(Counts::intensity(c.frame(1)[2]), Some(3.0));
        assert_eq!(Counts::intensity(c.frame(0)[2]), None);
    }

    #[test]
    fn from_f64_rejects_values_outside_u16_path() {
        assert!(matches!(
            Counts::from_f64([1, 1, 1], &[65535.0], None),
            Err(CoreError::CountOutOfRange { .. })
        ));
        assert!(matches!(
            Counts::from_f64([1, 1, 1], &[1.5], None),
            Err(CoreError::NonInteger { .. })
        ));
        assert!(Counts::from_f64([1, 1, 1], &[f64::NAN], None).is_err());
        assert!(Counts::from_f64([1, 1, 1], &[65534.0], None).is_ok());
    }
}
