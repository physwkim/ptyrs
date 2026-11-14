use crate::base::CoreError;

/// Scan positions in metres (sample translation; x = columns, y = rows), each coordinate
/// [`Positions::valid`] from [`Positions::new`] on.
#[derive(Clone, Debug, PartialEq)]
pub struct Positions {
    y: Vec<f64>,
    x: Vec<f64>,
}

impl Positions {
    pub fn new(y: Vec<f64>, x: Vec<f64>) -> Result<Self, CoreError> {
        if y.len() != x.len() {
            return Err(CoreError::Shape(format!(
                "{} y positions vs {} x positions",
                y.len(),
                x.len()
            )));
        }
        if !y.iter().chain(&x).all(|&v| Self::valid(v)) {
            return Err(CoreError::Invalid(
                "positions must be finite with |pos| <= 1e10 (direct-beam frames unsupported)"
                    .into(),
            ));
        }
        Ok(Self { y, x })
    }

    /// Whether `v`, metres, is a coordinate a scan position may have: finite, `|v| <= 1e10`.
    /// PyNX treats `|pos| > 1e10` as a direct-beam frame; the oracle does not model those.
    pub fn valid(v: f64) -> bool {
        v.is_finite() && v.abs() <= 1e10
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// PyNX `PtychoData.__init__` position preprocessing: subtract the mean, then if every
    /// position shares one sub-pixel offset (within 0.02 px), remove it and snap to whole
    /// object pixels. `pixel` is `[py, px]` from [`crate::base::Geometry::pixel_size_object`].
    ///
    /// # Panics
    ///
    /// For an infinite `pixel`, whose snap gives NaN, or a position more than 1e10 m from
    /// the scan's mean: either leaves a coordinate outside [`Positions::valid`].
    pub fn pynx_preprocess(&self, pixel: [f64; 2]) -> Positions {
        Positions::new(pynx_axis(&self.y, pixel[0]), pynx_axis(&self.x, pixel[1]))
            .expect("preprocessed positions")
    }

    /// Positions in object pixels, `[y, x]` per frame.
    pub fn in_pixels(&self, pixel: [f64; 2]) -> Vec<[f64; 2]> {
        self.y
            .iter()
            .zip(&self.x)
            .map(|(y, x)| [y / pixel[0], x / pixel[1]])
            .collect()
    }
}

fn pynx_axis(v: &[f64], px: f64) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let mut pos: Vec<f64> = v.iter().map(|p| p - mean).collect();
    let mut frac: Vec<f64> = pos.iter().map(|p| (p / px).rem_euclid(1.0)).collect();
    if frac.iter().all(|d| d.min(1.0 - d).abs() < 0.1) {
        // PyNX recentres the fraction with the position in metres, not pixels; kept for parity.
        frac = pos
            .iter()
            .map(|p| (p + 0.5).rem_euclid(1.0) - 0.5)
            .collect();
    }
    let fmean = frac.iter().sum::<f64>() / n;
    // numpy.allclose(frac, fmean, rtol=1e-5, atol=0.02)
    if frac
        .iter()
        .all(|d| (d - fmean).abs() <= 0.02 + 1e-5 * fmean.abs())
    {
        for p in &mut pos {
            *p = ((*p - fmean * px) / px).round_ties_even() * px;
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapped(pixels: &[f64], px: f64) -> Vec<f64> {
        let x: Vec<f64> = pixels.iter().map(|v| v * px).collect();
        let p = Positions::new(x.clone(), x)
            .unwrap()
            .pynx_preprocess([px, px]);
        p.in_pixels([px, px]).iter().map(|v| v[1]).collect()
    }

    #[test]
    fn near_integer_positions_snap_to_pixels() {
        // centred fractions all within 0.1 of a whole pixel
        let v = snapped(&[0.0, 2.02, 4.0, 6.0], 1e-8);
        for (got, want) in v.iter().zip([-3.0, -1.0, 1.0, 3.0]) {
            assert!((got - want).abs() < 1e-9, "{v:?}");
        }
    }

    #[test]
    fn shared_subpixel_offset_is_removed_then_snapped() {
        // centred fractions 0.50, 0.51, 0.50, 0.49: one shared offset of 0.5 px
        let v = snapped(&[0.0, 3.01, 6.0, 8.99], 1e-8);
        for (got, want) in v.iter().zip([-5.0, -2.0, 1.0, 4.0]) {
            assert!((got - want).abs() < 1e-9, "{v:?}");
        }
    }

    #[test]
    fn scattered_fractions_are_only_centred() {
        let px = 1.0;
        let x = vec![0.0, 1.3, 2.71, 4.05];
        let p = Positions::new(x.clone(), x.clone())
            .unwrap()
            .pynx_preprocess([px, px]);
        let mean = x.iter().sum::<f64>() / 4.0;
        for (a, b) in p.x.iter().zip(&x) {
            assert!((a - (b - mean)).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_direct_beam_marker() {
        assert!(Positions::new(vec![2e10], vec![0.0]).is_err());
    }

    #[test]
    #[should_panic(expected = "preprocessed positions")]
    fn preprocess_panics_for_an_infinite_pixel() {
        let x = vec![0.0, 1e-8];
        Positions::new(x.clone(), x)
            .unwrap()
            .pynx_preprocess([f64::INFINITY; 2]);
    }

    #[test]
    #[should_panic(expected = "preprocessed positions")]
    fn preprocess_panics_beyond_1e10_m_from_the_mean() {
        // mean 1e10 / 3: the last position lands 4e10 / 3 m from it
        let x = vec![1e10, 1e10, -1e10];
        Positions::new(x.clone(), x)
            .unwrap()
            .pynx_preprocess([1e-8; 2]);
    }
}
