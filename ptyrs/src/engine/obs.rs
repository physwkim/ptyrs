use crate::base::Counts;

/// Natural (centred) index of shifted index `j` in an `[ny, nx]` plane, and the reverse:
/// `fftshift` on both axes. Frames are even ([`crate::base::Geometry`]), so the map is its
/// own inverse.
#[inline]
pub fn shift(j: usize, ny: usize, nx: usize) -> usize {
    let (y, x) = (j / nx, j % nx);
    ((y + ny / 2) % ny) * nx + (x + nx / 2) % nx
}

/// Measured intensities in the FFT-shifted layout the operators read (PyNX
/// `fftshift(iobs)`), built once from [`Counts`]: `None` is a masked pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Observed {
    dims: [usize; 3],
    data: Vec<Option<f64>>,
    nb_obs: usize,
}

impl Observed {
    pub fn from_counts(counts: &Counts) -> Self {
        let dims = counts.dims();
        let [_, ny, nx] = dims;
        let n = ny * nx;
        let mut data = Vec::with_capacity(counts.as_slice().len());
        for f in 0..dims[0] {
            let frame = counts.frame(f);
            data.extend((0..n).map(|j| Counts::intensity(frame[shift(j, ny, nx)])));
        }
        Self {
            dims,
            data,
            nb_obs: counts.nb_obs(),
        }
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn nframe(&self) -> usize {
        self.dims[0]
    }

    pub fn plane_len(&self) -> usize {
        self.dims[1] * self.dims[2]
    }

    /// Shifted frames `frames`, concatenated.
    pub fn frames(&self, frames: std::ops::Range<usize>) -> &[Option<f64>] {
        let n = self.plane_len();
        &self.data[frames.start * n..frames.end * n]
    }

    /// Unmasked pixels over all frames (PyNX `nb_obs`), the LLK normalisation.
    pub fn nb_obs(&self) -> usize {
        self.nb_obs
    }

    /// Sum of the unmasked intensities.
    pub fn total(&self) -> f64 {
        self.data.iter().flatten().sum()
    }
}

#[cfg(test)]
mod tests {
    use crate::base::MASKED;

    use super::*;

    #[test]
    fn shift_moves_centre_to_origin_and_masks() {
        assert_eq!(shift(0, 4, 6), 2 * 6 + 3);
        assert_eq!(shift(shift(7, 4, 6), 4, 6), 7);
        let mut raw: Vec<u16> = (0..24).collect();
        raw[2 * 6 + 3] = MASKED;
        let o = Observed::from_counts(&Counts::from_raw([1, 4, 6], raw).unwrap());
        assert_eq!(o.frames(0..1)[0], None);
        assert_eq!(o.frames(0..1)[1], Some(16.0));
        assert_eq!(o.nb_obs(), 23);
    }
}
