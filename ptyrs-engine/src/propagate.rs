//! Free-space propagators from the exit-wave plane to the detector plane, after pty-chi
//! `propagate.py`, each made unitary.
//!
//! A pty-chi propagator is `√k·U`: `U` unitary (for angular spectrum a partial isometry, which
//! drops the evanescent band) and `k` its `get_intensity_scaling_factor`; its
//! `propagate_backward` is `Uᴴ/√k`. A [`Propagator`] applies `U` alone, in the engine's
//! FFT-shifted layout on both sides, so `Σ|psi|²` is kept for every propagator and the
//! adjoint is the inverse. The pty-chi probe for the same intensities is the engine's divided
//! by `√k` ([`Propagator::intensity_scale`]).

use std::f64::consts::PI;

use ptyrs_core::{C64, Geometry};

use crate::{EngineError, Fft2, quad_phase, shift};

/// The exit-wave model of the pty-chi algorithms: `psi = U(P·O·q)` per view, with `q` the
/// object-plane phase that [`crate::obj_probe_to_psi`] applies and `U` the [`Propagator`].
pub struct Optics {
    quad: Vec<C64>,
    propagator: Propagator,
}

impl Optics {
    /// PyNX's far field: [`quad_phase`] and [`Propagator::Fourier`].
    pub fn far_field(g: &Geometry) -> Self {
        Self {
            quad: quad_phase(g),
            propagator: Propagator::Fourier(Fft2::new(g.frame)),
        }
    }

    /// pty-chi's planar model: no object-plane phase and [`Propagator::free_space`].
    pub fn free_space(w: &Wavefield) -> Self {
        let [ny, nx] = w.shape;
        Self {
            quad: vec![C64::new(1.0, 0.0); ny * nx],
            propagator: Propagator::free_space(w),
        }
    }

    pub fn quad(&self) -> &[C64] {
        &self.quad
    }

    pub fn propagator(&self) -> &Propagator {
        &self.propagator
    }
}

/// One free-space propagation: pty-chi `WavefieldPropagatorParameters` in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wavefield {
    shape: [usize; 2],
    wavelength: f64,
    pixel: [f64; 2],
    distance: f64,
}

impl Wavefield {
    /// `shape` `[ny, nx]` (both even), source-plane `pixel` `[py, px]`, signed `distance`
    /// (infinite for the far field).
    pub fn new(
        shape: [usize; 2],
        wavelength: f64,
        pixel: [f64; 2],
        distance: f64,
    ) -> Result<Self, EngineError> {
        if shape.iter().any(|&n| n == 0 || n % 2 != 0) {
            return Err(EngineError::Invalid(format!(
                "plane shape {shape:?} must be even and non-zero"
            )));
        }
        for (name, v) in [
            ("wavelength", wavelength),
            ("pixel height", pixel[0]),
            ("pixel width", pixel[1]),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(EngineError::Invalid(format!(
                    "{name} must be finite and > 0, got {v}"
                )));
            }
        }
        // pty-chi divides by the distance in fresnel_number and the selection rule, and only
        // +inf selects the far field.
        if distance.is_nan() || distance == 0.0 || distance == f64::NEG_INFINITY {
            return Err(EngineError::Invalid(format!(
                "distance must be non-zero and not -inf, got {distance}"
            )));
        }
        Ok(Self {
            shape,
            wavelength,
            pixel,
            distance,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    fn pixel_width_wlu(&self) -> f64 {
        self.pixel[1] / self.wavelength
    }

    fn aspect_ratio(&self) -> f64 {
        self.pixel[1] / self.pixel[0]
    }

    fn distance_wlu(&self) -> f64 {
        self.distance / self.wavelength
    }

    /// Signed per-pixel Fresnel number `px²/(λz)`, zero in the far field.
    pub fn fresnel_number(&self) -> f64 {
        let w = self.pixel_width_wlu();
        w * w / self.distance_wlu()
    }

    /// pty-chi `is_fresnel_transform_preferrable`: the Fresnel transform's output pitch
    /// `λ|z|/(N p)` is finer than the source pitch `p` on either axis, where angular spectrum
    /// would be undersampled.
    pub fn prefers_fresnel(&self) -> bool {
        let [ny, nx] = self.shape;
        let d = self.distance_wlu().abs();
        let w = self.pixel_width_wlu();
        let h = w / self.aspect_ratio();
        w * w / d < 1.0 / nx as f64 || h * h / d < 1.0 / ny as f64
    }
}

/// A unitary free-space propagator over `[ny, nx]` planes in the FFT-shifted layout.
pub enum Propagator {
    /// Far field, pty-chi `FourierPropagator`: [`Fft2`], `k = ny·nx`.
    Fourier(Fft2),
    AngularSpectrum(AngularSpectrum),
    Fresnel(Fresnel),
}

/// pty-chi `AngularSpectrumPropagator`: `Fft2⁻¹·T·Fft2`, `k = 1`.
pub struct AngularSpectrum {
    fft: Fft2,
    /// Transfer function on the unshifted `fftfreq` grid, the order [`Fft2`] produces;
    /// zero on the evanescent band.
    tf: Vec<C64>,
}

/// pty-chi `FresnelTransformPropagator`: `x ↦ a·Fft2(b·x)`, `Fft2⁻¹` for a negative distance.
pub struct Fresnel {
    fft: Fft2,
    /// pty-chi `A = C1·C2·C0` over `|C0|`, shifted layout.
    a: Vec<C64>,
    /// pty-chi `B`, shifted layout.
    b: Vec<C64>,
    reverse: bool,
    /// `ny·nx·|C0|²`.
    scale: f64,
}

impl Propagator {
    /// pty-chi `build_free_space_propagator`: Fourier for an infinite distance, else the
    /// Fresnel transform where [`Wavefield::prefers_fresnel`], else angular spectrum.
    pub fn free_space(w: &Wavefield) -> Self {
        if w.distance.is_infinite() {
            Self::Fourier(Fft2::new(w.shape))
        } else if w.prefers_fresnel() {
            Self::Fresnel(Fresnel::new(w))
        } else {
            Self::AngularSpectrum(AngularSpectrum::new(w))
        }
    }

    /// Angular spectrum whatever the sampling, as pty-chi's in-object (multislice)
    /// propagator.
    pub fn angular_spectrum(w: &Wavefield) -> Result<Self, EngineError> {
        if w.distance.is_infinite() {
            return Err(EngineError::Invalid(
                "angular spectrum needs a finite distance".into(),
            ));
        }
        Ok(Self::AngularSpectrum(AngularSpectrum::new(w)))
    }

    pub fn shape(&self) -> [usize; 2] {
        self.fft().shape()
    }

    /// pty-chi `get_intensity_scaling_factor` of the same propagator: its output intensity
    /// over this one's.
    pub fn intensity_scale(&self) -> f64 {
        match self {
            Self::Fourier(fft) => {
                let [ny, nx] = fft.shape();
                (ny * nx) as f64
            }
            Self::AngularSpectrum(_) => 1.0,
            Self::Fresnel(f) => f.scale,
        }
    }

    /// Propagates every plane in `planes` (length a multiple of `ny * nx`).
    pub fn forward(&self, planes: &mut [C64]) {
        match self {
            Self::Fourier(fft) => fft.forward(planes),
            Self::AngularSpectrum(p) => {
                p.fft.forward(planes);
                mul_planes(planes, &p.tf, false);
                p.fft.inverse(planes);
            }
            Self::Fresnel(p) => {
                mul_planes(planes, &p.b, false);
                if p.reverse {
                    p.fft.inverse(planes)
                } else {
                    p.fft.forward(planes)
                }
                mul_planes(planes, &p.a, false);
            }
        }
    }

    /// The adjoint of [`Self::forward`], also its inverse except on angular spectrum's
    /// evanescent band.
    pub fn adjoint(&self, planes: &mut [C64]) {
        match self {
            Self::Fourier(fft) => fft.inverse(planes),
            Self::AngularSpectrum(p) => {
                p.fft.forward(planes);
                mul_planes(planes, &p.tf, true);
                p.fft.inverse(planes);
            }
            Self::Fresnel(p) => {
                mul_planes(planes, &p.a, true);
                if p.reverse {
                    p.fft.forward(planes)
                } else {
                    p.fft.inverse(planes)
                }
                mul_planes(planes, &p.b, true);
            }
        }
    }

    fn fft(&self) -> &Fft2 {
        match self {
            Self::Fourier(fft) => fft,
            Self::AngularSpectrum(p) => &p.fft,
            Self::Fresnel(p) => &p.fft,
        }
    }
}

impl AngularSpectrum {
    /// `T = exp(2πi z √(1 − r))` where `r = (fx² + (a fy)²)/px² < 1`, else 0, with `a` the
    /// aspect ratio and lengths in wavelengths.
    fn new(w: &Wavefield) -> Self {
        let [ny, nx] = w.shape;
        let ar = w.aspect_ratio();
        let width = w.pixel_width_wlu();
        let i2piz = 2.0 * PI * w.distance_wlu();
        let mut tf = Vec::with_capacity(ny * nx);
        for y in 0..ny {
            let fy = fftfreq(y, ny);
            for x in 0..nx {
                let fx = fftfreq(x, nx);
                let ratio = (fx * fx + (ar * fy) * (ar * fy)) / (width * width);
                tf.push(if ratio < 1.0 {
                    C64::from_polar(1.0, i2piz * (1.0 - ratio).sqrt())
                } else {
                    C64::default()
                });
            }
        }
        Self {
            fft: Fft2::new(w.shape),
            tf,
        }
    }
}

impl Fresnel {
    /// With `Fr` the Fresnel number, `a` the aspect ratio and centred `X`, `Y`:
    /// `C0 = Fr/(i a)`, `C1 = exp(2πi z)`, `C2 = exp(iπ/Fr ((X/nx)² + (a Y/ny)²))`,
    /// `B = exp(iπ Fr (X² + (Y/a)²))`.
    fn new(w: &Wavefield) -> Self {
        let [ny, nx] = w.shape;
        let fr = w.fresnel_number();
        let ar = w.aspect_ratio();
        let c1 = C64::from_polar(1.0, 2.0 * PI * w.distance_wlu());
        // C0/|C0| = −i·sign(Fr).
        let c1c0 = c1 * C64::new(0.0, -fr.signum());
        let (n, m) = (nx as f64, ny as f64);
        let mut a = Vec::with_capacity(ny * nx);
        let mut b = Vec::with_capacity(ny * nx);
        for j in 0..ny * nx {
            let i = shift(j, ny, nx);
            let xx = (i % nx) as f64 - (nx / 2) as f64;
            let yy = (i / nx) as f64 - (ny / 2) as f64;
            let s2 = (xx / n) * (xx / n) + (ar * yy / m) * (ar * yy / m);
            a.push(c1c0 * C64::from_polar(1.0, s2 * PI / fr));
            let sb = xx * xx + (yy / ar) * (yy / ar);
            b.push(C64::from_polar(1.0, PI * fr * sb));
        }
        let c0 = fr / ar;
        Self {
            fft: Fft2::new(w.shape),
            a,
            b,
            reverse: w.distance < 0.0,
            scale: (ny * nx) as f64 * (c0 * c0),
        }
    }
}

/// torch `fftfreq(n)[k]`, computed as torch does: `k·(1/n)` with `k` wrapped to `−n/2..`.
fn fftfreq(k: usize, n: usize) -> f64 {
    let k = if k < n.div_ceil(2) {
        k as f64
    } else {
        k as f64 - n as f64
    };
    k * (1.0 / n as f64)
}

/// Multiplies every plane of `planes` by `d` (or `conj(d)`) pointwise.
fn mul_planes(planes: &mut [C64], d: &[C64], conj: bool) {
    for plane in planes.chunks_exact_mut(d.len()) {
        for (v, d) in plane.iter_mut().zip(d) {
            *v *= if conj { d.conj() } else { *d };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::{Propagation, dot_error};

    fn wavefield(pixel: [f64; 2], distance: f64) -> Wavefield {
        Wavefield::new([12, 16], 1e-10, pixel, distance).unwrap()
    }

    /// One geometry per kind and distance sign, anisotropic where the kind allows it.
    fn propagators() -> Vec<Propagator> {
        vec![
            Propagator::free_space(&wavefield([50e-9, 50e-9], f64::INFINITY)),
            Propagator::free_space(&wavefield([50e-9, 50e-9], 1e-4)),
            Propagator::free_space(&wavefield([60e-9, 45e-9], -2e-4)),
            Propagator::free_space(&wavefield([50e-9, 50e-9], 1.25e-3)),
            Propagator::free_space(&wavefield([40e-9, 50e-9], -1.6e-3)),
            Propagator::free_space(&Wavefield::new([12, 16], 5e-7, [3e-7, 3e-7], 1.5e-6).unwrap()),
        ]
    }

    #[test]
    fn every_propagator_passes_the_dot_test() {
        let kinds: Vec<_> = propagators()
            .iter()
            .map(|p| match p {
                Propagator::Fourier(_) => 'F',
                Propagator::AngularSpectrum(_) => 'A',
                Propagator::Fresnel(_) => 'T',
            })
            .collect();
        assert_eq!(kinds, ['F', 'A', 'A', 'T', 'T', 'A']);
        for p in &propagators() {
            let op = Propagation {
                propagator: p,
                nplane: 3,
            };
            let err = dot_error(&op);
            assert!(err < 1e-14, "dot test error {err:.3e}");
            crate::register(op).unwrap();
        }
    }

    #[test]
    fn propagation_keeps_power_and_adjoint_inverts_it() {
        for p in &propagators() {
            let [ny, nx] = p.shape();
            let x = crate::linop::noise(2 * ny * nx, 3);
            let mut y = x.clone();
            p.forward(&mut y);
            let power = |v: &[C64]| v.iter().map(|u| u.norm_sqr()).sum::<f64>();
            let mut back = y.clone();
            p.adjoint(&mut back);
            // Angular spectrum drops the evanescent band; only the 5e-7 m wavelength one has
            // one at these pixels.
            if let Propagator::AngularSpectrum(a) = p
                && a.tf.iter().any(|t| *t == C64::default())
            {
                assert!(power(&y) < power(&x));
                continue;
            }
            assert!((power(&y) / power(&x) - 1.0).abs() < 1e-14);
            let d = back
                .iter()
                .zip(&x)
                .map(|(u, v)| (u - v).norm())
                .fold(0.0, f64::max);
            assert!(d < 1e-14, "adjoint does not invert: {d:.3e}");
        }
    }

    #[test]
    fn wavefield_rejects_what_ptychi_cannot_propagate() {
        assert!(Wavefield::new([12, 15], 1e-10, [1e-8, 1e-8], 1.0).is_err());
        assert!(Wavefield::new([12, 16], 1e-10, [0.0, 1e-8], 1.0).is_err());
        assert!(Wavefield::new([12, 16], 1e-10, [1e-8, 1e-8], 0.0).is_err());
        assert!(Wavefield::new([12, 16], 1e-10, [1e-8, 1e-8], f64::NEG_INFINITY).is_err());
        let far = wavefield([1e-8, 1e-8], f64::INFINITY);
        assert!(Propagator::angular_spectrum(&far).is_err());
    }
}
