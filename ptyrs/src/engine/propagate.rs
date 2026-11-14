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

use crate::base::{C64, Geometry};

use crate::engine::{EngineError, Fft2, quad_phase, shift};

/// The exit-wave model of the pty-chi algorithms: `psi = E(P·O·q)` per view, with `q` the
/// plane phase that [`crate::engine::obj_probe_to_psi`] applies and `E` the [`Exit`] after it;
/// through several object slices, [`crate::engine::SliceWaves`].
///
/// `q` is every phase the exit transform has in front of it, the far field's [`quad_phase`]
/// or the Fresnel transform's `B`: fixed in the lab frame as the probe is, so a probe moved
/// by a frame's offset moves with all of it.
pub struct Optics {
    /// The object-plane pixel `[py, px]` in metres.
    pixel: [f64; 2],
    quad: Vec<C64>,
    /// `quad` is `exp(i·(c[0]·Y² + c[1]·X²))` over the centred pixel.
    plane_quad: [f64; 2],
    propagator: Propagator,
    /// The in-object propagator from each slice to the next.
    gaps: Vec<Propagator>,
}

/// [`Optics::quad`] displaced by a frame's offset `d`, `q(r − d)`, with its derivatives over
/// `d`: `∂_a q = i·slope[a]·q` and `∂_a ∂_c q = (i·curve[a]·δ_ac − slope[a]·slope[c])·q`.
pub struct MovedQuad {
    /// Natural layout, as [`Optics::quad`].
    pub quad: Vec<C64>,
    pub slope: [Vec<f64>; 2],
    pub curve: [f64; 2],
}

impl Optics {
    /// PyNX's far field: [`quad_phase`] and [`Propagator::Fourier`].
    pub fn far_field(g: &Geometry) -> Self {
        let pixel = g.pixel_size_object();
        Self {
            pixel,
            quad: quad_phase(g),
            plane_quad: pixel.map(|p| PI * p * p / (g.wavelength * g.distance)),
            propagator: Propagator::Fourier(Fft2::new(g.frame)),
            gaps: Vec::new(),
        }
    }

    /// pty-chi's planar model, [`Propagator::free_space`]: its plane phase is the Fresnel
    /// transform's, none under the other two.
    pub fn free_space(w: &Wavefield) -> Self {
        let [ny, nx] = w.shape;
        let propagator = Propagator::free_space(w);
        let mut quad = vec![C64::new(1.0, 0.0); ny * nx];
        let mut plane_quad = [0.0; 2];
        if let Propagator::Fresnel(p) = &propagator {
            // `B` is in the shifted layout, the plane phase in the natural one.
            for (j, b) in p.b.iter().enumerate() {
                quad[shift(j, ny, nx)] = *b;
            }
            plane_quad = p.plane_quad;
        }
        Self {
            pixel: w.pixel,
            quad,
            plane_quad,
            propagator,
            gaps: Vec::new(),
        }
    }

    /// This model through `gaps.len() + 1` object slices, pty-chi's multislice object: the
    /// wave crosses gap `s`, from slice `s` to `s + 1`, by [`Propagator::angular_spectrum`] of
    /// `gaps[s]` (pty-chi `slice_spacings_m[s]` at the object pixel).
    pub fn with_slices(self, gaps: &[Wavefield]) -> Result<Self, EngineError> {
        let shape = self.propagator.shape();
        let gaps = gaps
            .iter()
            .map(|w| {
                if w.shape != shape {
                    return Err(EngineError::Invalid(format!(
                        "slice gap plane {:?} is not the exit plane {shape:?}",
                        w.shape
                    )));
                }
                Propagator::angular_spectrum(w)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { gaps, ..self })
    }

    pub fn quad(&self) -> &[C64] {
        &self.quad
    }

    /// The object-plane pixel `[py, px]` in metres, the unit of the views' corners.
    pub fn pixel(&self) -> [f64; 2] {
        self.pixel
    }

    /// [`Self::exit`] after [`Self::quad`] through one [`Fft2`]; fails as
    /// [`Propagator::fourier_form`] does, and where `quad` differs between the axes.
    pub fn fourier_form(&self) -> Result<FourierForm<'_>, EngineError> {
        let form = self.propagator.fourier_form()?;
        let [cy, cx] = self.plane_quad;
        if cy != cx {
            return Err(EngineError::Invalid(
                "an object-plane phase over pixels that are not square has no \
                 one-transform form"
                    .into(),
            ));
        }
        Ok(FourierForm { quad: cx, ..form })
    }

    /// The plane phase slice `s` applies: [`Self::quad`] on the last slice, none on the
    /// others.
    pub fn slice_quad(&self, s: usize) -> Option<&[C64]> {
        (s + 1 == self.nslice()).then_some(&self.quad[..])
    }

    /// The last slice's plane phase for a frame whose probe was moved by `offset` (`[dy, dx]`
    /// pixels) ahead of an earlier slice: [`Self::quad`] displaced as the probe is. `None` at
    /// one slice, where the probe's move carries the phase ([`Self::slice_quad`]).
    pub fn moved_quad(&self, offset: [f64; 2]) -> Option<MovedQuad> {
        if self.nslice() == 1 {
            return None;
        }
        let [ny, nx] = self.propagator.shape();
        let c = self.plane_quad;
        let at = |i: usize| {
            [
                (i / nx) as f64 - (ny / 2) as f64,
                (i % nx) as f64 - (nx / 2) as f64,
            ]
        };
        let slope = [0, 1].map(|a| {
            (0..ny * nx)
                .map(|i| -2.0 * c[a] * (at(i)[a] - offset[a]))
                .collect()
        });
        let quad = (self.quad.iter().enumerate())
            .map(|(i, q)| {
                let r = at(i);
                let phase = |a: usize| c[a] * offset[a] * (offset[a] - 2.0 * r[a]);
                q * C64::from_polar(1.0, phase(0) + phase(1))
            })
            .collect();
        Some(MovedQuad {
            quad,
            slope,
            curve: c.map(|c| 2.0 * c),
        })
    }

    /// What takes the last slice's exit waves, [`Self::quad`] on them, to the detector.
    pub fn exit(&self) -> Exit<'_> {
        Exit(&self.propagator)
    }

    /// The in-object propagators, gap `s` from slice `s` to `s + 1`.
    pub fn gaps(&self) -> &[Propagator] {
        &self.gaps
    }

    /// The object slices the wave crosses, `gaps().len() + 1`.
    pub fn nslice(&self) -> usize {
        self.gaps.len() + 1
    }
}

/// An [`Optics`]' propagator less the plane phase it starts with, which [`Optics::quad`]
/// holds: the whole of it but for the Fresnel transform, there `x ↦ a·Fft2(x)`.
#[derive(Clone, Copy)]
pub struct Exit<'a>(&'a Propagator);

impl Exit<'_> {
    pub fn shape(&self) -> [usize; 2] {
        self.0.shape()
    }

    /// [`Propagator::intensity_scale`] of the whole propagator.
    pub fn intensity_scale(&self) -> f64 {
        self.0.intensity_scale()
    }

    /// Every plane in `planes` (length a multiple of `ny * nx`) to the detector.
    pub fn forward(&self, planes: &mut [C64]) {
        self.0.after_plane(planes);
    }

    /// The adjoint of [`Self::forward`], as [`Propagator::adjoint`] its inverse.
    pub fn adjoint(&self, planes: &mut [C64]) {
        self.0.before_plane(planes);
    }

    /// [`Propagator::adjoint_inverts`] of the whole propagator.
    pub fn adjoint_inverts(&self) -> bool {
        self.0.adjoint_inverts()
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
    /// `B = exp(i·(c[0]·Y² + c[1]·X²))` over the centred pixel.
    plane_quad: [f64; 2],
}

/// A [`Propagator`] as `D·Fft2·V·q`: `q = exp(i·quad·(X² + Y²))` over the centred pixel of the
/// exit-wave plane, `V` its [`PlaneStage`], and `D` a unit-modulus detector factor, which
/// drops out of every intensity and of `Uᴴ·w·U` for real detector weights `w`.
pub struct FourierForm<'a> {
    pub quad: f64,
    pub stage: PlaneStage<'a>,
}

/// What a [`FourierForm`] applies between the plane's phase and [`Fft2`].
pub enum PlaneStage<'a> {
    None,
    /// The index reversal on both axes, `Fft2²`.
    Reverse,
    /// `Fft2⁻¹·Fft2⁻¹·T·Fft2` of this transfer function `T`, on the unshifted `fftfreq`
    /// grid and even in both frequencies.
    Transfer(&'a [C64]),
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
        if let Self::Fresnel(p) = self {
            mul_planes(planes, &p.b, false);
        }
        self.after_plane(planes);
    }

    /// [`Self::forward`] of planes that carry the Fresnel transform's `B` already.
    fn after_plane(&self, planes: &mut [C64]) {
        match self {
            Self::Fourier(fft) => fft.forward(planes),
            Self::AngularSpectrum(p) => {
                p.fft.forward(planes);
                mul_planes(planes, &p.tf, false);
                p.fft.inverse(planes);
            }
            Self::Fresnel(p) => {
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
        self.before_plane(planes);
        if let Self::Fresnel(p) = self {
            mul_planes(planes, &p.b, true);
        }
    }

    /// The adjoint of [`Self::after_plane`].
    fn before_plane(&self, planes: &mut [C64]) {
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
            }
        }
    }

    /// Whether [`Self::adjoint`] undoes [`Self::forward`]: not under angular spectrum with
    /// an evanescent band, whose transfer function is 0 there.
    pub fn adjoint_inverts(&self) -> bool {
        match self {
            Self::AngularSpectrum(p) => !p.tf.contains(&C64::default()),
            Self::Fourier(_) | Self::Fresnel(_) => true,
        }
    }

    /// This propagator through one [`Fft2`]. A Fresnel transform over pixels that are not
    /// square has no such form, its plane phase being no `quad`, nor angular spectrum with an
    /// evanescent band, where `Vᴴ·V` is not the identity the form's users take it for.
    pub fn fourier_form(&self) -> Result<FourierForm<'_>, EngineError> {
        let (quad, stage) = match self {
            Self::Fourier(_) => (0.0, PlaneStage::None),
            Self::AngularSpectrum(p) => {
                if !self.adjoint_inverts() {
                    return Err(EngineError::Invalid(
                        "angular spectrum with an evanescent band has no one-transform form".into(),
                    ));
                }
                (0.0, PlaneStage::Transfer(&p.tf))
            }
            Self::Fresnel(p) => {
                let [cy, cx] = p.plane_quad;
                let quad = (cy == cx).then_some(cx).ok_or_else(|| {
                    EngineError::Invalid(
                        "a Fresnel transform over pixels that are not square has no \
                         one-transform form"
                            .into(),
                    )
                })?;
                let stage = if p.reverse {
                    PlaneStage::Reverse
                } else {
                    PlaneStage::None
                };
                (quad, stage)
            }
        };
        Ok(FourierForm { quad, stage })
    }

    /// Angular spectrum's transfer function, on the unshifted `fftfreq` grid.
    pub fn transfer(&self) -> Option<&[C64]> {
        match self {
            Self::AngularSpectrum(p) => Some(&p.tf),
            _ => None,
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
            plane_quad: [PI * fr / (ar * ar), PI * fr],
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
    use crate::engine::linop::{Propagation, dot_error};
    use crate::worst::worst_of;

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
            crate::engine::register(op).unwrap();
        }
    }

    #[test]
    fn propagation_keeps_power_and_adjoint_inverts_it() {
        let inverts: Vec<_> = propagators().iter().map(|p| p.adjoint_inverts()).collect();
        assert_eq!(inverts, [true, true, true, true, true, false]);
        for p in &propagators() {
            let [ny, nx] = p.shape();
            let x = crate::engine::linop::noise(2 * ny * nx, 3);
            let mut y = x.clone();
            p.forward(&mut y);
            let power = |v: &[C64]| v.iter().map(|u| u.norm_sqr()).sum::<f64>();
            let mut back = y.clone();
            p.adjoint(&mut back);
            // Angular spectrum drops the evanescent band; only the 5e-7 m wavelength one has
            // one at these pixels.
            if !p.adjoint_inverts() {
                assert!(power(&y) < power(&x));
                continue;
            }
            assert!((power(&y) / power(&x) - 1.0).abs() < 1e-14);
            let d = worst_of(back.iter().zip(&x).map(|(u, v)| (u - v).norm()));
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

    /// A sample displaced by a fraction of a pixel under a fixed probe and Fresnel plane
    /// phase is what an offset stands for, and the probe moved with [`Optics::quad`] gives
    /// its intensities, forward and back in distance; moved without the phase (pty-chi
    /// PATCH) it does not.
    #[test]
    fn a_moved_probe_is_the_displaced_sample_under_a_fresnel_transform() {
        use crate::engine::features::move_planes;
        use crate::engine::linop::noise;

        const N: usize = 64;
        const OFFSET: [f64; 2] = [0.3, -0.45];
        // A band-limited object of period 256, so it has a value between its pixels.
        let terms: Vec<(C64, [f64; 2])> = noise(24, 3)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[c, k]| (0.15 * c, [k.re, k.im].map(|k| (12.0 * k).round())))
            .collect();
        let object = |y: f64, x: f64| {
            let wave = |&(c, k): &(C64, [f64; 2])| {
                c * C64::from_polar(1.0, 2.0 * PI * (k[0] * y + k[1] * x) / 256.0)
            };
            C64::new(1.0, 0.0) + terms.iter().map(wave).sum::<C64>()
        };
        let centred = |i: usize| [i / N, i % N].map(|v| v as f64 - (N / 2) as f64);
        let probe: Vec<C64> = (0..N * N)
            .map(|i| {
                let [y, x] = centred(i);
                let r2 = y * y + x * x;
                C64::from_polar(
                    (-r2 / 49.0).exp(),
                    0.04 * r2 + 0.5 * (y / 3.0).sin() - 0.3 * (x / 4.0).cos(),
                )
            })
            .collect();
        let fft = Fft2::new([N, N]);
        for distance in [5e-3, -8e-3] {
            let w = Wavefield::new([N, N], 1e-10, [50e-9, 50e-9], distance).unwrap();
            assert!(matches!(Propagator::free_space(&w), Propagator::Fresnel(_)));
            let optics = Optics::free_space(&w);
            // Intensities of `probe · O(· + at) · q` in the window.
            let intensity = |probe: &[C64], at: [f64; 2]| -> Vec<f64> {
                let mut psi: Vec<C64> = (0..N * N)
                    .map(|j| {
                        let i = shift(j, N, N);
                        let [y, x] = centred(i);
                        probe[i] * object(y + at[0], x + at[1]) * optics.quad()[i]
                    })
                    .collect();
                optics.exit().forward(&mut psi);
                psi.iter().map(|c| c.norm_sqr()).collect()
            };
            let truth = intensity(&probe, OFFSET);
            let err = |quad: Option<&[C64]>| {
                let mut moved = probe.clone();
                move_planes(&mut moved, quad, OFFSET, &fft);
                let model = intensity(&moved, [0.0; 2]);
                let d = model.iter().zip(&truth).map(|(a, b)| (a - b).abs());
                worst_of(d) / truth.iter().copied().fold(0.0, f64::max)
            };
            let (with, without) = (err(optics.slice_quad(0)), err(None));
            assert!(with < 1e-6, "distance {distance}: {with:.3e}");
            assert!(without > 1e-3, "distance {distance}: {without:.3e}");
        }
    }
}
