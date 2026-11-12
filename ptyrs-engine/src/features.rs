//! The scan features LSQML refines beside the object and probe, as CPU f64 operators over the
//! arrays of [`crate::ops`] and its layout contract: each a forward map and the gradient of a
//! [`Cost`] with respect to the feature.

use std::f64::consts::PI;

use ptyrs_core::{C64, Object, Probe, View};

use crate::ops::natural;
use crate::{Fft2, Observed};

/// A cost of the modelled intensity `icalc` against `iobs`, summed over unmasked pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cost {
    /// `icalc − iobs + iobs·ln(iobs/icalc)` (`icalc` where `iobs = 0`): [`crate::llk`]'s
    /// Poisson sum.
    Poisson,
    /// `(√icalc − √iobs)²`: a quarter of [`crate::llk`]'s Euclidian sum.
    Amplitude,
}

impl Cost {
    /// `∂cost/∂icalc` at one pixel, `icalc` floored at 1e-12 as in [`crate::poisson_grad`].
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

/// [`crate::obj_probe_to_psi`] with frame `f` at its view corner plus `offsets[f]` (`[dy, dx]`
/// object pixels), pty-chi LSQML's sub-pixel position model: the window stays on whole object
/// pixels and every probe mode `P·q` is Fourier-translated by the offset on its own periodic
/// grid, `W = DFT⁻¹(DFT(P·q)·exp(−2πi f·offset))` with `f` numpy's `fftfreq`:
/// `psi[k] = fftshift(W[probe_k, frame_k] · O[obj_k, win_k])`.
///
/// `psi` holds one object-plane plane per view and is overwritten; with zero offsets it is
/// [`crate::obj_probe_to_psi`] up to FFT rounding.
pub fn obj_probe_shift_to_psi(
    views: &[View],
    object: &Object,
    probe: &Probe,
    quad: &[C64],
    offsets: &[[f64; 2]],
    psi: &mut [C64],
) {
    let [ny, nx] = probe.plane_shape();
    let n = ny * nx;
    assert_eq!(
        psi.len(),
        views.len() * n,
        "psi must hold one plane per view"
    );
    let fft = &Fft2::new([ny, nx]);
    let a = probe_spectra(probe, quad, fft);
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
/// `g = 2·∂C/∂exit* = 2·DFT⁻¹(slope·psi)` from the far field `psi` of the frame's modes.
pub fn position_grad(
    views: &[View],
    object: &Object,
    probe: &Probe,
    quad: &[C64],
    obs: &Observed,
    offsets: &[[f64; 2]],
    cost: Cost,
) -> Vec<[f64; 2]> {
    let [ny, nx] = probe.plane_shape();
    let n = ny * nx;
    let nxo = object.plane_shape()[1];
    let fft = &Fft2::new([ny, nx]);
    let mut g = vec![C64::default(); views.len() * n];
    obj_probe_shift_to_psi(views, object, probe, quad, offsets, &mut g);
    fft.forward(&mut g);
    let a = probe_spectra(probe, quad, fft);
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
