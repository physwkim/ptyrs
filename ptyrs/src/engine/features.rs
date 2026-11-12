//! The scan features refined beside the object and probe: [`Features`], which [`Ptycho`]'s
//! AP, DM and ML loops model and update, and the CPU f64 operators LSQML builds on, over the
//! arrays of [`crate::engine::ops`] and its layout contract: each a forward map and the gradient of a
//! [`Cost`] with respect to the feature.

use std::f64::consts::PI;
use std::ops::Range;

use crate::base::{C64, Probe, View};

use crate::engine::ops::natural;
use crate::engine::{EngineError, Fft2, Observed, Ptycho};

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
}

impl Features {
    /// Checks the features against a scan of `nframe` frames.
    pub(crate) fn check(&self, nframe: usize) -> Result<(), EngineError> {
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
        Ok(())
    }

    /// Whether some feature takes a step before each cycle.
    pub(crate) fn stepped(&self) -> bool {
        self.scale.is_some()
    }

    /// Every feature's step before a cycle models it, from the far field `psi` of the
    /// estimate the cycle starts from (all frames, as [`scale_sums`] takes it) under the
    /// current model: [`scale_step`] of the scale.
    pub(crate) fn step(&mut self, obs: &Observed, nmode: usize, psi: &[C64]) {
        let model = Model::of(self);
        let sums = (self.scale.as_ref()).map(|s| scale_sums(obs, 0..s.len(), nmode, psi, model));
        if let (Some(scale), Some(sums)) = (&mut self.scale, sums) {
            scale_step(scale, &sums);
        }
    }
}

/// The model intensity `icalc` of a frame's `I = Σ_m |psi_m|²` under [`Features`]:
/// `Default` is `icalc = I`, bit for bit.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Model<'a> {
    scale: Option<&'a [f64]>,
}

impl<'a> Model<'a> {
    pub(crate) fn of(features: &'a Features) -> Self {
        Self {
            scale: features.scale.as_deref(),
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

    /// `icalc` of frame `f` at a pixel of intensity `i`.
    #[inline]
    pub(crate) fn calc(&self, f: usize, i: f64) -> f64 {
        self.scaled(f, i)
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
                    sums[1] += iobs * i / model.calc(f, i).max(1e-12);
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
fn freq(k: usize, n: usize) -> f64 {
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

/// One probe spectrum `a` translated by `offset` (`[dy, dx]` pixels) back in the object plane,
/// `DFT⁻¹(a·exp(−2πi f·offset))`, or with `axis` its derivative along that offset component.
fn translate(a: &[C64], offset: [f64; 2], axis: Option<usize>, fft: &Fft2) -> Vec<C64> {
    let [ny, nx] = fft.shape();
    let mut w: Vec<C64> = a
        .iter()
        .enumerate()
        .map(|(j, &a)| {
            let f = [freq(j / nx, ny), freq(j % nx, nx)];
            let e = C64::from_polar(1.0, -2.0 * PI * (f[0] * offset[0] + f[1] * offset[1]));
            match axis {
                Some(ax) => a * e * C64::new(0.0, -2.0 * PI * f[ax]),
                None => a * e,
            }
        })
        .collect();
    fft.inverse(&mut w);
    w
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
    let [ny, nx] = probe.plane_shape();
    let n = ny * nx;
    assert_eq!(
        psi.len(),
        views.len() * n,
        "psi must hold one plane per view"
    );
    let fft = &Fft2::new([ny, nx]);
    let a = probe_spectra(probe, p.quad(), fft);
    for (v, out) in views.iter().zip(psi.chunks_exact_mut(n)) {
        let w = translate(
            &a[v.probe as usize * n..][..n],
            offsets[v.frame as usize],
            None,
            fft,
        );
        let o = object.plane(v.obj as usize);
        let nxo = object.plane_shape()[1];
        for (j, (out, w)) in out.iter_mut().zip(&w).enumerate() {
            let (_, y, x) = natural(j, ny, nx);
            *out = w * o[(v.oy as usize + y) * nxo + v.ox as usize + x];
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
