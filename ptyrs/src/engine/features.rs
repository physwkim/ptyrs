//! The scan features refined beside the object and probe: [`Features`], which [`Ptycho`]'s
//! AP, DM and ML loops model and update, and the CPU f64 operators LSQML builds on, over the
//! arrays of [`crate::engine::ops`] and its layout contract: each a forward map and the gradient of a
//! [`Cost`] with respect to the feature.

use std::f64::consts::PI;
use std::ops::Range;

use crate::base::{C64, Object, Probe, Stack, View};

use crate::engine::ops::{blend, natural};
use crate::engine::{EngineError, Fft2, Observed, Ptycho, shift};

/// The features of a reconstruction's model intensity, each modelled and refined while set;
/// `Default` sets none and leaves every loop's arithmetic as it is without features.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Features {
    /// Per-frame intensity scale `s`, `[nframe]`: frame `f`'s model intensity is
    /// `s[f]·Σ_m |psi_m|²`. Each AP, DM and ML cycle starts with Poisson's fixed-point step
    /// `s[f] *= Σ iobs·I / max(icalc, 1e-12) / Σ I` over the frame's unmasked pixels, with
    /// `I = Σ_m |psi_m|²` and `icalc = s[f]·I` of the estimate the cycle starts from (kept
    /// where `Σ I` is 0), then `s /= mean(s)`: the object and probe carry the mean flux. The
    /// cycle's LLK record is under the scale before the step, the rest of it models the
    /// stepped scale.
    pub scale: Option<Vec<f64>>,
    /// Detector background `b`, `[ny, nx]` in the detector order of the counts, the same for
    /// every frame: the model intensity is `s[f]·I + b`. Each cycle's step, from the estimate
    /// and model the scale's takes, is also the EM step `b[j] *= Σ_f iobs / max(icalc, 1e-12)
    /// / nobs[j]` over the `nobs[j]` frames where pixel `j` is unmasked (kept where there is
    /// none); a pixel at 0 stays at 0. ML's [`Ptycho::scale_obj_probe`] leaves the
    /// background's photons out of both sums.
    pub background: Option<Vec<f64>>,
    /// Per-frame position offsets `d`, `[nframe]` of `[dy, dx]` object pixels from each frame's
    /// view corner: the frame's probe modes are those of [`obj_probe_shift_to_psi`], `P·q`
    /// Fourier-moved by `d[f]` on the frame's own periodic grid (an offset past the frame
    /// wraps), the object window staying at the corner. After each AP and DM cycle's object
    /// and probe update, pty-chi's gradient position step from the exit-wave correction
    /// `c = x − O·W` of the cycle's projection (`x` the new exit waves, `W` the moved probe
    /// and `O` the object it was made from): per frame and axis, over the frame's views and
    /// pixels, `Δ = 0.3·Re Σ conj(c)·O·∂W / (Σ |O·∂W|² + max_f Σ |O·∂W|²)` (0 where that is
    /// 0) clipped to ±0.1, then `d += Δ − mean_f Δ`, which keeps the mean offset. pty-chi's
    /// clip at 10 MADs of `Δ` is left out. ML refuses it.
    ///
    /// [`crate::engine::Lsqml`] models it on every slice through the moved probe and steps it
    /// as pty-chi's LSQML does, after each batch's object and probe update, and
    /// [`Ptycho::rpie`](crate::engine::Ptycho::rpie) as pty-chi's PIE does:
    /// [`crate::engine::LsqmlBatch::position_directions`] times 0.3 at the batch's frames, 0
    /// at the others, each axis clipped to `min(10·mean_f |dx − mean dx|, 0.1)` over all
    /// frames, with no mean subtracted. Both models are pty-chi's while every `|d| ≤ 1/2`,
    /// past which pty-chi moves the window to the nearest pixel.
    pub position: Option<Vec<[f64; 2]>>,
}

impl Features {
    /// Checks the features against a scan of `nframe` frames of `frame` (`[ny, nx]`) pixels.
    pub(crate) fn check(&self, nframe: usize, frame: [usize; 2]) -> Result<(), EngineError> {
        if let Some(s) = &self.scale {
            if s.len() != nframe {
                return Err(EngineError::Invalid(format!(
                    "{} intensity scales for {nframe} frames",
                    s.len()
                )));
            }
            if !s.iter().all(|s| s.is_finite() && *s > 0.0) {
                return Err(EngineError::Invalid(
                    "intensity scales must be finite and positive".into(),
                ));
            }
        }
        if let Some(b) = &self.background {
            if b.len() != frame[0] * frame[1] {
                return Err(EngineError::Invalid(format!(
                    "a background of {} pixels for {frame:?} frames",
                    b.len()
                )));
            }
            if !b.iter().all(|b| b.is_finite() && *b >= 0.0) {
                return Err(EngineError::Invalid(
                    "the background must be finite and not negative".into(),
                ));
            }
        }
        if let Some(d) = &self.position {
            if d.len() != nframe {
                return Err(EngineError::Invalid(format!(
                    "{} position offsets for {nframe} frames",
                    d.len()
                )));
            }
            if !d.iter().flatten().all(|d| d.is_finite()) {
                return Err(EngineError::Invalid(
                    "position offsets must be finite".into(),
                ));
            }
        }
        Ok(())
    }

    /// Whether some feature takes a step before each cycle.
    pub(crate) fn stepped(&self) -> bool {
        self.scale.is_some() || self.background.is_some()
    }

    /// Every feature's step before a cycle models it, from the far field `psi` of the
    /// estimate the cycle starts from (all frames, as [`scale_sums`] takes it) under the
    /// current model: [`scale_step`] of the scale and [`background_step`] of the background,
    /// both from sums taken before either steps.
    pub(crate) fn step(&mut self, obs: &Observed, nmode: usize, psi: &[C64]) {
        let model = Model::of(self, obs);
        let scale = (self.scale.as_ref()).map(|s| scale_sums(obs, 0..s.len(), nmode, psi, model));
        let background =
            (self.background.as_ref()).map(|_| background_sums(obs, nmode, psi, model));
        if let (Some(scale), Some(sums)) = (&mut self.scale, scale) {
            scale_step(scale, &sums);
        }
        if let (Some(b), Some(sums)) = (&mut self.background, background) {
            background_step(b, &sums);
        }
    }

    /// The background's photons over the unmasked pixels of `obs`, `Σ_j b[j]·nobs[j]`; 0
    /// without one.
    pub(crate) fn background_photons(&self, obs: &Observed) -> f64 {
        let Some(b) = &self.background else {
            return 0.0;
        };
        let [_, ny, nx] = obs.dims();
        let nobs = unmasked_frames(obs);
        (nobs.iter().enumerate())
            .map(|(j, &k)| b[shift(j, ny, nx)] * k as f64)
            .sum()
    }
}

/// The model intensity `icalc` of a frame's `I = Σ_m |psi_m|²` under [`Features`]:
/// `Default` is `icalc = I`, bit for bit.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Model<'a> {
    scale: Option<&'a [f64]>,
    /// The background in detector order, and the detector's `[ny, nx]`.
    background: Option<(&'a [f64], [usize; 2])>,
}

impl<'a> Model<'a> {
    /// The model of `features` over the frames of `obs`.
    pub(crate) fn of(features: &'a Features, obs: &Observed) -> Self {
        let [_, ny, nx] = obs.dims();
        Self {
            scale: features.scale.as_deref(),
            background: features.background.as_deref().map(|b| (b, [ny, nx])),
        }
    }

    /// `x` times frame `f`'s intensity scale, `x` itself without one.
    #[inline]
    pub(crate) fn scaled(&self, f: usize, x: f64) -> f64 {
        match self.scale {
            Some(s) => s[f] * x,
            None => x,
        }
    }

    /// `icalc` of frame `f` at its shifted pixel `j` of intensity `i`.
    #[inline]
    pub(crate) fn calc(&self, f: usize, j: usize, i: f64) -> f64 {
        let calc = self.scaled(f, i);
        match self.background {
            Some((b, [ny, nx])) => calc + b[shift(j, ny, nx)],
            None => calc,
        }
    }
}

/// Per frame of `frames`, `[Σ I, Σ iobs·I / max(icalc, 1e-12)]` over its unmasked pixels of the
/// far field `psi` (`frames.len() * nmode` planes, frame-major), `I = Σ_m |psi_m|²` and `icalc`
/// of `model`: the sums of [`scale_step`].
pub(crate) fn scale_sums(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    psi: &[C64],
    model: Model<'_>,
) -> Vec<[f64; 2]> {
    let n = obs.plane_len();
    assert_eq!(psi.len(), frames.len() * nmode * n);
    frames
        .clone()
        .zip(obs.frames(frames).chunks_exact(n))
        .zip(psi.chunks_exact(nmode * n))
        .map(|((f, o), block)| {
            let mut sums = [0.0; 2];
            for (j, iobs) in o.iter().enumerate() {
                if let Some(iobs) = *iobs {
                    let i: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                    sums[0] += i;
                    sums[1] += iobs * i / model.calc(f, j, i).max(1e-12);
                }
            }
            sums
        })
        .collect()
}

/// Poisson's fixed-point step of the intensity scale, `scale[f] *= ratio / intensity` for each
/// frame's `[intensity, ratio]` of [`scale_sums`] where the intensity is above 0, then
/// `scale /= mean(scale)`.
pub(crate) fn scale_step(scale: &mut [f64], sums: &[[f64; 2]]) {
    assert_eq!(scale.len(), sums.len());
    for (s, [intensity, ratio]) in scale.iter_mut().zip(sums) {
        if *intensity > 0.0 {
            *s *= ratio / intensity;
        }
    }
    let mean = scale.iter().sum::<f64>() / scale.len() as f64;
    for s in scale {
        *s /= mean;
    }
}

/// Each shifted pixel's unmasked frames in `obs`.
fn unmasked_frames(obs: &Observed) -> Vec<usize> {
    let n = obs.plane_len();
    let mut nobs = vec![0; n];
    for o in obs.frames(0..obs.nframe()).chunks_exact(n) {
        for (k, iobs) in nobs.iter_mut().zip(o) {
            *k += usize::from(iobs.is_some());
        }
    }
    nobs
}

/// Per detector pixel `j` (detector order), `[Σ_f iobs / max(icalc, 1e-12), nobs]` over the
/// `nobs` frames of `obs` where `j` is unmasked, from the far field `psi` of every frame
/// (`nframe * nmode` planes, frame-major) and `icalc` of `model`: the sums of
/// [`background_step`].
pub(crate) fn background_sums(
    obs: &Observed,
    nmode: usize,
    psi: &[C64],
    model: Model<'_>,
) -> Vec<[f64; 2]> {
    let [nframe, ny, nx] = obs.dims();
    let n = ny * nx;
    assert_eq!(psi.len(), nframe * nmode * n);
    let mut sums = vec![[0.0; 2]; n];
    for (f, (o, block)) in (obs.frames(0..nframe).chunks_exact(n))
        .zip(psi.chunks_exact(nmode * n))
        .enumerate()
    {
        for (j, iobs) in o.iter().enumerate() {
            if let Some(iobs) = *iobs {
                let i: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                let s = &mut sums[shift(j, ny, nx)];
                s[0] += iobs / model.calc(f, j, i).max(1e-12);
                s[1] += 1.0;
            }
        }
    }
    sums
}

/// The background's EM (Richardson–Lucy) step, `b[j] *= ratio / nobs` for each pixel's
/// `[ratio, nobs]` of [`background_sums`] where `nobs` is above 0.
pub(crate) fn background_step(b: &mut [f64], sums: &[[f64; 2]]) {
    assert_eq!(b.len(), sums.len());
    for (b, [ratio, nobs]) in b.iter_mut().zip(sums) {
        if *nobs > 0.0 {
            *b *= ratio / nobs;
        }
    }
}

/// A cost of the modelled intensity `icalc` against `iobs`, summed over unmasked pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cost {
    /// `icalc − iobs + iobs·ln(iobs/icalc)` (`icalc` where `iobs = 0`): [`crate::engine::llk`]'s
    /// Poisson sum.
    Poisson,
    /// `(√icalc − √iobs)²`: a quarter of [`crate::engine::llk`]'s Euclidian sum.
    Amplitude,
}

impl Cost {
    /// The cost at one pixel.
    pub fn value(self, iobs: f64, icalc: f64) -> f64 {
        match self {
            Cost::Poisson if iobs > 0.0 => icalc - iobs + iobs * (iobs / icalc).ln(),
            Cost::Poisson => icalc,
            Cost::Amplitude => (icalc.sqrt() - iobs.sqrt()).powi(2),
        }
    }

    /// `∂cost/∂icalc` at one pixel, `icalc` floored at 1e-12 as in [`crate::engine::poisson_grad`].
    pub fn slope(self, iobs: f64, icalc: f64) -> f64 {
        let icalc = icalc.max(1e-12);
        match self {
            Cost::Poisson => 1.0 - iobs / icalc,
            Cost::Amplitude => 1.0 - (iobs / icalc).sqrt(),
        }
    }
}

/// numpy `fftfreq`: cycles per pixel of DFT index `k` on an axis of `n`, Nyquist at −1/2.
pub(crate) fn freq(k: usize, n: usize) -> f64 {
    let k = if k < n / 2 {
        k as f64
    } else {
        k as f64 - n as f64
    };
    k / n as f64
}

/// `DFT(fftshift(P[m]·q))` of every probe mode, one plane per mode.
fn probe_spectra(probe: &Probe, quad: &[C64], fft: &Fft2) -> Vec<C64> {
    let [ny, nx] = probe.plane_shape();
    let n = ny * nx;
    let mut a = vec![C64::default(); probe.as_slice().len()];
    for (m, out) in a.chunks_exact_mut(n).enumerate() {
        let p = probe.plane(m);
        for (j, out) in out.iter_mut().enumerate() {
            let (i, _, _) = natural(j, ny, nx);
            *out = p[i] * quad[i];
        }
    }
    fft.forward(&mut a);
    a
}

/// `[fy, fx]` of DFT index `j` of an `[ny, nx]` plane and the translation phase
/// `exp(−2πi f·offset)` there.
fn ramp(j: usize, [ny, nx]: [usize; 2], offset: [f64; 2]) -> ([f64; 2], C64) {
    let f = [freq(j / nx, ny), freq(j % nx, nx)];
    let e = C64::from_polar(1.0, -2.0 * PI * (f[0] * offset[0] + f[1] * offset[1]));
    (f, e)
}

/// One probe spectrum `a` translated by `offset` (`[dy, dx]` pixels) back in the object plane,
/// `DFT⁻¹(a·exp(−2πi f·offset))`, or with `axis` its derivative along that offset component.
fn translate(a: &[C64], offset: [f64; 2], axis: Option<usize>, fft: &Fft2) -> Vec<C64> {
    let shape = fft.shape();
    let mut w: Vec<C64> = a
        .iter()
        .enumerate()
        .map(|(j, &a)| {
            let (f, e) = ramp(j, shape, offset);
            match axis {
                Some(ax) => a * e * C64::new(0.0, -2.0 * PI * f[ax]),
                None => a * e,
            }
        })
        .collect();
    fft.inverse(&mut w);
    w
}

/// Natural-layout probe planes `p` moved by `offset` as pty-chi LSQML moves its probe,
/// `conj(q)·T(q·p)` per plane, with `q` the object-plane phase the slice the planes enter
/// applies after them (`None` for none) and `T` [`translate`]'s move on the plane's periodic
/// grid, which commutes with fftshift. The adjoint is the move by `−offset`.
pub(crate) fn move_planes(p: &mut [C64], quad: Option<&[C64]>, offset: [f64; 2], fft: &Fft2) {
    let shape = fft.shape();
    let n = shape[0] * shape[1];
    let phase = |p: &mut [C64], conj: bool| {
        if let Some(q) = quad {
            for plane in p.chunks_exact_mut(n) {
                for (p, q) in plane.iter_mut().zip(q) {
                    *p *= if conj { q.conj() } else { *q };
                }
            }
        }
    };
    phase(p, false);
    fft.forward(p);
    for plane in p.chunks_exact_mut(n) {
        plane
            .iter_mut()
            .enumerate()
            .for_each(|(j, p)| *p *= ramp(j, shape, offset).1);
    }
    fft.inverse(p);
    phase(p, true);
}

/// [`crate::engine::obj_probe_to_psi`] with frame `f` at its view corner plus `offsets[f]` (`[dy, dx]`
/// object pixels), pty-chi LSQML's sub-pixel position model: the window stays on whole object
/// pixels and every probe mode `P·q` is Fourier-translated by the offset on its own periodic
/// grid, `W = DFT⁻¹(DFT(P·q)·exp(−2πi f·offset))` with `f` numpy's `fftfreq`:
/// `psi[k] = fftshift(W[probe_k, frame_k] · O[obj_k, win_k])`.
///
/// `views` are those of `p` to model, `p`'s object, probe and quadratic phase the state; `psi`
/// holds one object-plane plane per view and is overwritten; with zero offsets it is
/// [`crate::engine::obj_probe_to_psi`] up to FFT rounding.
pub fn obj_probe_shift_to_psi(p: &Ptycho, views: &[View], offsets: &[[f64; 2]], psi: &mut [C64]) {
    let (object, probe) = (p.object(), p.probe());
    let fft = &Fft2::new(probe.plane_shape());
    Shifted::new(views, probe, p.quad(), offsets, fft).to_psi(views, object, psi);
}

/// pty-chi's position step size and per-axis update limit, in object pixels.
pub(crate) const POSITION_STEP: f64 = 0.3;
pub(crate) const POSITION_LIMIT: f64 = 0.1;
/// pty-chi's floor of the position directions' denominator maximum, on its probe of
/// `P·q/sqrt(intensity scale)`.
pub(crate) const EPS_POSITION: f64 = 1e-6;

/// pty-chi `PositionCorrection.get_gradient_update`'s sums over one frame (Fourier
/// differentiation of the PATCH window): per axis `[y, x]`, `num = Σ Re(conj(∂O·w)·chi)` and
/// `den = Σ |∂O·w|²`, with `∂O` the derivative of the natural-layout window `patch` taken as
/// periodic, and `w` the wave it multiplies and `chi` its correction, both in exit-wave layout.
pub(crate) fn gradient_sums(patch: &[C64], w: &[C64], chi: &[C64], fft: &Fft2) -> [[f64; 2]; 2] {
    let [ny, nx] = fft.shape();
    let grad = [0, 1].map(|axis| {
        let mut g = patch.to_vec();
        fft.forward(&mut g);
        for (j, g) in g.iter_mut().enumerate() {
            let f = [j / nx, j % nx];
            *g *= C64::new(0.0, 2.0 * PI * freq(f[axis], [ny, nx][axis]));
        }
        fft.inverse(&mut g);
        g
    });
    let mut sums = [[0.0; 2]; 2];
    for (j, (w, c)) in w.iter().zip(chi).enumerate() {
        let (i, _, _) = natural(j, ny, nx);
        for (axis, g) in grad.iter().enumerate() {
            let u = g[i] * w;
            sums[0][axis] += (u.conj() * c).re;
            sums[1][axis] += u.norm_sqr();
        }
    }
    sums
}

/// pty-chi's gradient position direction `[Δy, Δx]` of every frame of a batch from its
/// [`gradient_sums`]: `num / (den + max(max_batch den, eps))` per axis.
pub(crate) fn gradient_directions(sums: &[[[f64; 2]; 2]], eps: f64) -> Vec<[f64; 2]> {
    let max = [0, 1].map(|axis| sums.iter().map(|s| s[1][axis]).fold(eps, f64::max));
    sums.iter()
        .map(|[num, den]| [0, 1].map(|axis| num[axis] / (den[axis] + max[axis])))
        .collect()
}

/// pty-chi's position step (`ProbePositions.step_optimizer`) from the batch `frames`'
/// [`gradient_directions`] `delta` on the offsets `d` of every frame: its SGD step
/// `dx = 0.3·delta` at the batch's frames, 0 at the others, then each axis clipped to
/// `min(10·mean |dx − mean dx|, 0.1)` over every frame.
pub(crate) fn batch_position_step(d: &mut [[f64; 2]], frames: &[usize], delta: &[[f64; 2]]) {
    let mut dx = vec![[0.0; 2]; d.len()];
    for (&f, delta) in frames.iter().zip(delta) {
        dx[f] = delta.map(|v| POSITION_STEP * v);
    }
    let n = d.len() as f64;
    for axis in 0..2 {
        let mean = dx.iter().map(|x| x[axis]).sum::<f64>() / n;
        let mad = dx.iter().map(|x| (x[axis] - mean).abs()).sum::<f64>() / n;
        let limit = (10.0 * mad).min(POSITION_LIMIT);
        for (d, x) in d.iter_mut().zip(&dx) {
            d[axis] += x[axis].clamp(-limit, limit);
        }
    }
}

/// A probe under [`Features::position`]: its spectra and every view's moved probe `W`, which
/// take the place of `q·P` in the operators of [`crate::engine::ops`].
pub(crate) struct Shifted {
    shape: [usize; 2],
    offsets: Vec<[f64; 2]>,
    /// The quadratic phase `q` the probe was moved with.
    quad: Vec<C64>,
    /// [`probe_spectra`] of the probe.
    spectra: Vec<C64>,
    /// `W` of every view, one shifted plane per view.
    waves: Vec<C64>,
}

impl Shifted {
    /// `probe` (with the quadratic phase `quad`) moved by `offsets` for every view of `views`.
    pub(crate) fn new(
        views: &[View],
        probe: &Probe,
        quad: &[C64],
        offsets: &[[f64; 2]],
        fft: &Fft2,
    ) -> Self {
        let shape = probe.plane_shape();
        let n = shape[0] * shape[1];
        let spectra = probe_spectra(probe, quad, fft);
        let mut waves = Vec::with_capacity(views.len() * n);
        for v in views {
            let a = &spectra[v.probe as usize * n..][..n];
            waves.extend(translate(a, offsets[v.frame as usize], None, fft));
        }
        Self {
            shape,
            offsets: offsets.to_vec(),
            quad: quad.to_vec(),
            spectra,
            waves,
        }
    }

    /// [`crate::engine::psi_to_obj_probe`] with the sums of [`Self::accumulate_obj`] and
    /// [`Self::accumulate_probe`], both from the arrays on entry; `self` is of `probe`.
    pub(crate) fn psi_to_obj_probe(
        &self,
        views: &[View],
        psi: &[C64],
        object: &mut Object,
        probe: &mut Probe,
        [obj_inertia, probe_inertia]: [Option<f64>; 2],
        fft: &Fft2,
    ) {
        let obj_acc = obj_inertia.map(|_| self.accumulate_obj(views, object.dims(), psi));
        if let Some(inertia) = probe_inertia {
            let (num, norm) = self.accumulate_probe(views, object, probe.dims(), psi, fft);
            blend(probe, &num, &norm, inertia);
        }
        if let (Some(inertia), Some((num, norm))) = (obj_inertia, obj_acc) {
            blend(object, &num, &norm, inertia);
        }
    }

    /// `W` of the `k`th view of the views `self` was made for, in exit-wave layout.
    pub(crate) fn wave(&self, k: usize) -> &[C64] {
        let n = self.shape[0] * self.shape[1];
        &self.waves[k * n..(k + 1) * n]
    }

    /// The object window of view `v` at shifted index `j`.
    fn window(&self, object: &Object, v: &View, j: usize) -> C64 {
        let [ny, nx] = self.shape;
        let (_, y, x) = natural(j, ny, nx);
        let nxo = object.plane_shape()[1];
        object.plane(v.obj as usize)[(v.oy as usize + y) * nxo + v.ox as usize + x]
    }

    /// [`crate::engine::obj_probe_to_psi`] with `W` in place of `q·P`: `psi[k] = W[k]·O[win_k]`.
    pub(crate) fn to_psi(&self, views: &[View], object: &Object, psi: &mut [C64]) {
        let n = self.shape[0] * self.shape[1];
        assert_eq!(
            psi.len(),
            views.len() * n,
            "psi must hold one plane per view"
        );
        for ((v, out), w) in views
            .iter()
            .zip(psi.chunks_exact_mut(n))
            .zip(self.waves.chunks_exact(n))
        {
            for (j, (out, w)) in out.iter_mut().zip(w).enumerate() {
                *out = w * self.window(object, v, j);
            }
        }
    }

    /// [`crate::engine::accumulate_obj`] with `W` in place of `q·P`: `num += psi·conj(W)`,
    /// `norm += |W|²`.
    pub(crate) fn accumulate_obj(
        &self,
        views: &[View],
        obj_dims: [usize; 3],
        psi: &[C64],
    ) -> (Stack<C64>, Stack<f64>) {
        let [ny, nx] = self.shape;
        let n = ny * nx;
        let nxo = obj_dims[2];
        assert_eq!(psi.len(), views.len() * n);
        let mut num = Stack::filled(obj_dims, C64::default());
        let mut norm = Stack::filled(obj_dims, 0.0);
        for ((v, x), w) in views
            .iter()
            .zip(psi.chunks_exact(n))
            .zip(self.waves.chunks_exact(n))
        {
            let (oy, ox) = (v.oy as usize, v.ox as usize);
            let (num, norm) = (
                num.plane_mut(v.obj as usize),
                norm.plane_mut(v.obj as usize),
            );
            for (j, (x, w)) in x.iter().zip(w).enumerate() {
                let (_, y, xx) = natural(j, ny, nx);
                let o = (oy + y) * nxo + ox + xx;
                num[o] += x * w.conj();
                norm[o] += w.norm_sqr();
            }
        }
        (num, norm)
    }

    /// [`crate::engine::accumulate_probe`] through the move's adjoint: per probe mode,
    /// `num = conj(q)·Σ_f DFT⁻¹(DFT(Y_f)·conj(exp(−2πi f·d[f])))` with
    /// `Y_f = Σ conj(O[win])·psi` over the frame's views of that mode, summed in the spectrum
    /// frame by frame, and `norm = Σ |O[win]|²` as the unmoved one. The probe does not enter.
    pub(crate) fn accumulate_probe(
        &self,
        views: &[View],
        object: &Object,
        probe_dims: [usize; 3],
        psi: &[C64],
        fft: &Fft2,
    ) -> (Stack<C64>, Stack<f64>) {
        let [_, ny, nx] = probe_dims;
        let n = ny * nx;
        assert_eq!(psi.len(), views.len() * n);
        let mut spectrum = Stack::filled(probe_dims, C64::default());
        let mut norm = Stack::filled(probe_dims, 0.0);
        let mut y = Stack::filled(probe_dims, C64::default());
        let mut start = 0;
        for frame_views in views.chunk_by(|a, b| a.frame == b.frame) {
            let offset = self.offsets[frame_views[0].frame as usize];
            y.as_mut_slice().fill(C64::default());
            for (v, x) in frame_views.iter().zip(psi[start * n..].chunks_exact(n)) {
                let (y, norm) = (
                    y.plane_mut(v.probe as usize),
                    norm.plane_mut(v.probe as usize),
                );
                for (j, x) in x.iter().enumerate() {
                    let o = self.window(object, v, j);
                    y[j] += o.conj() * x;
                    norm[shift(j, ny, nx)] += o.norm_sqr();
                }
            }
            start += frame_views.len();
            fft.forward(y.as_mut_slice());
            for (s, y) in
                (spectrum.as_mut_slice().chunks_exact_mut(n)).zip(y.as_slice().chunks_exact(n))
            {
                for (j, (s, y)) in s.iter_mut().zip(y).enumerate() {
                    *s += y * ramp(j, [ny, nx], offset).1.conj();
                }
            }
        }
        fft.inverse(spectrum.as_mut_slice());
        let mut num = Stack::filled(probe_dims, C64::default());
        for (num, s) in
            (num.as_mut_slice().chunks_exact_mut(n)).zip(spectrum.as_slice().chunks_exact(n))
        {
            for (j, s) in s.iter().enumerate() {
                let i = shift(j, ny, nx);
                num[i] = self.quad[i].conj() * s;
            }
        }
        (num, norm)
    }

    /// Per frame of the offsets, `[num, den]` of the position step of [`Features::position`]
    /// over the frame's views of `views` and pixels: `num[a] = Re Σ conj(c)·O·∂W/∂d[a]` with
    /// `c = x − O·W`, `x` the new exit waves `psi` and `O` the object the projection read, and
    /// `den[a] = Σ |O·∂W/∂d[a]|²`.
    pub(crate) fn position_sums(
        &self,
        views: &[View],
        object: &Object,
        psi: &[C64],
        fft: &Fft2,
    ) -> Vec<[[f64; 2]; 2]> {
        let n = self.shape[0] * self.shape[1];
        assert_eq!(psi.len(), views.len() * n);
        let mut sums = vec![[[0.0; 2]; 2]; self.offsets.len()];
        for ((v, x), w) in views
            .iter()
            .zip(psi.chunks_exact(n))
            .zip(self.waves.chunks_exact(n))
        {
            let f = v.frame as usize;
            let a = &self.spectra[v.probe as usize * n..][..n];
            let [num, den] = &mut sums[f];
            for (axis, (num, den)) in num.iter_mut().zip(den).enumerate() {
                let dw = translate(a, self.offsets[f], Some(axis), fft);
                for (j, ((x, w), dw)) in x.iter().zip(w).zip(&dw).enumerate() {
                    let o = self.window(object, v, j);
                    let odw = o * dw;
                    *num += ((x - w * o).conj() * odw).re;
                    *den += odw.norm_sqr();
                }
            }
        }
        sums
    }
}

/// The position step of [`Features::position`] on the offsets `d` from each frame's
/// `[num, den]` of [`Shifted::position_sums`].
pub(crate) fn position_step(d: &mut [[f64; 2]], sums: &[[[f64; 2]; 2]]) {
    assert_eq!(d.len(), sums.len());
    for axis in 0..2 {
        let max = sums.iter().map(|s| s[1][axis]).fold(0.0, f64::max);
        let step: Vec<f64> = (sums.iter())
            .map(|[num, den]| {
                let den = den[axis] + max;
                if den > 0.0 {
                    (POSITION_STEP * num[axis] / den).clamp(-POSITION_LIMIT, POSITION_LIMIT)
                } else {
                    0.0
                }
            })
            .collect();
        let mean = step.iter().sum::<f64>() / step.len() as f64;
        for (d, s) in d.iter_mut().zip(&step) {
            d[axis] += s - mean;
        }
    }
}

/// Gradient of `cost` over the offsets of [`obj_probe_shift_to_psi`]: `[∂C/∂dy, ∂C/∂dx]` for
/// every frame of `offsets` (zero for a frame without views),
/// `Σ_views Σ Re(conj(g)·V·∂W/∂offset)` with `V` the object window and
/// `g = 2·∂C/∂exit* = 2·DFT⁻¹(slope·psi)` from the far field `psi` of the frame's modes against
/// `p`'s observed intensities.
pub fn position_grad(
    p: &Ptycho,
    views: &[View],
    offsets: &[[f64; 2]],
    cost: Cost,
) -> Vec<[f64; 2]> {
    let (object, probe, obs) = (p.object(), p.probe(), p.observed());
    let [ny, nx] = probe.plane_shape();
    let n = ny * nx;
    let nxo = object.plane_shape()[1];
    let fft = &Fft2::new([ny, nx]);
    let mut g = vec![C64::default(); views.len() * n];
    obj_probe_shift_to_psi(p, views, offsets, &mut g);
    fft.forward(&mut g);
    let a = probe_spectra(probe, p.quad(), fft);
    let mut grad = vec![[0.0; 2]; offsets.len()];
    let mut start = 0;
    for frame_views in views.chunk_by(|a, b| a.frame == b.frame) {
        let f = frame_views[0].frame as usize;
        let nmode = frame_views.len();
        let block = &mut g[start * n..(start + nmode) * n];
        start += nmode;
        for (j, iobs) in obs.frames(f..f + 1).iter().enumerate() {
            let r = iobs.map_or(0.0, |iobs| {
                let icalc: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                2.0 * cost.slope(iobs, icalc)
            });
            for m in 0..nmode {
                block[m * n + j] *= r;
            }
        }
        fft.inverse(block);
        for (v, g) in frame_views.iter().zip(block.chunks_exact(n)) {
            let o = object.plane(v.obj as usize);
            for (axis, grad) in grad[f].iter_mut().enumerate() {
                let dw = translate(&a[v.probe as usize * n..][..n], offsets[f], Some(axis), fft);
                *grad += g
                    .iter()
                    .zip(&dw)
                    .enumerate()
                    .map(|(j, (g, dw))| {
                        let (_, y, x) = natural(j, ny, nx);
                        let v = o[(v.oy as usize + y) * nxo + v.ox as usize + x];
                        (g.conj() * v * dw).re
                    })
                    .sum::<f64>();
            }
        }
    }
    grad
}

/// Per-frame intensity scale on every mode, as ptyrs::refgen's flux scale: the far field `psi`
/// covering the frames `frames` of `obs` (`frames.len() * nmode` planes, frame-major) is
/// multiplied by `√scale[f]` for frame `f`, so its `icalc` is `scale[f]·Σ_m |psi|²`.
pub fn scale_psi(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    scale: &[f64],
    psi: &mut [C64],
) {
    let block = nmode * obs.plane_len();
    assert_eq!(psi.len(), frames.len() * block);
    for (f, block) in frames.zip(psi.chunks_exact_mut(block)) {
        let r = scale[f].sqrt();
        block.iter_mut().for_each(|c| *c *= r);
    }
}

/// Gradient of `cost` over `scale` in [`scale_psi`] for every frame of `scale` (zero outside
/// `frames`), from the far field `psi` of `frames` before scaling:
/// `∂C/∂scale[f] = Σ slope(iobs, scale[f]·I)·I` with `I = Σ_m |psi|²`.
pub fn intensity_grad(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    scale: &[f64],
    psi: &[C64],
    cost: Cost,
) -> Vec<f64> {
    let n = obs.plane_len();
    assert_eq!(psi.len(), frames.len() * nmode * n);
    let mut grad = vec![0.0; scale.len()];
    for ((f, o), block) in frames
        .clone()
        .zip(obs.frames(frames).chunks_exact(n))
        .zip(psi.chunks_exact(nmode * n))
    {
        grad[f] = o
            .iter()
            .enumerate()
            .filter_map(|(j, iobs)| {
                iobs.map(|iobs| {
                    let i: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                    cost.slope(iobs, scale[f] * i) * i
                })
            })
            .sum();
    }
    grad
}

/// Calls `f(iobs, icalc, j)` at every unmasked pixel of the far field `psi` covering `frames`
/// of `obs`, with `icalc = Σ_m |psi|² + background[j]` and `j` the pixel's shifted index.
fn with_background(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    psi: &[C64],
    background: &[f64],
    mut f: impl FnMut(f64, f64, usize),
) {
    let n = obs.plane_len();
    assert_eq!(psi.len(), frames.len() * nmode * n);
    assert_eq!(background.len(), n);
    for (o, block) in obs
        .frames(frames)
        .chunks_exact(n)
        .zip(psi.chunks_exact(nmode * n))
    {
        for (j, iobs) in o.iter().enumerate() {
            if let Some(iobs) = *iobs {
                let i: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                f(iobs, i + background[j], j);
            }
        }
    }
}

/// `cost` summed over the frames `frames` with an incoherent background, one detector plane
/// added to every frame (as ptyrs::refgen adds it): `icalc = Σ_m |psi|² + background`, the far
/// field `psi` covering `frames` of `obs` and `background` shifted as [`Observed`].
pub fn background_cost(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    psi: &[C64],
    background: &[f64],
    cost: Cost,
) -> f64 {
    let mut sum = 0.0;
    with_background(obs, frames, nmode, psi, background, |iobs, icalc, _| {
        sum += cost.value(iobs, icalc)
    });
    sum
}

/// Gradient of [`background_cost`] over `background`, per pixel of the shifted plane:
/// `∂C/∂b = Σ_frames slope(iobs, icalc)`.
pub fn background_grad(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    psi: &[C64],
    background: &[f64],
    cost: Cost,
) -> Vec<f64> {
    let mut grad = vec![0.0; background.len()];
    with_background(obs, frames, nmode, psi, background, |iobs, icalc, j| {
        grad[j] += cost.slope(iobs, icalc)
    });
    grad
}
