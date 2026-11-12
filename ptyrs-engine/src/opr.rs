//! pty-chi's orthogonal probe relaxation (OPR, Odstrčil et al. 2016) with one eigenmode of probe
//! mode 0 as CPU f64 operators over the arrays and layout of [`crate::ops`]: frame `k` sees
//! `P_k0 = P_0 + w_k·U` and `P_km = P_m` for `m ≥ 1`, the main mode's weight held at 1
//! (pty-chi `optimize_intensity_variation` off).
//!
//! [`ProbeVariation::unique`] gives every frame of a batch its own probe planes, so the
//! operators of [`crate::ops`] and [`LsqmlBatch`] run on per-frame probes unchanged.

use ptyrs_core::{C64, Object, Probe, Stack, View};

use crate::ops::{accumulate_probe, obj_probe_to_psi};
use crate::{Cost, Fft2, LsqmlBatch, Observed};

/// One eigenmode `U` of probe mode 0, shaped as one probe plane, and its weight `w_k` for every
/// frame of the scan.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeVariation {
    eigenmode: Stack<C64>,
    weights: Vec<f64>,
}

impl ProbeVariation {
    /// `eigenmode` holds one plane.
    pub fn new(eigenmode: Stack<C64>, weights: Vec<f64>) -> Self {
        assert_eq!(eigenmode.len(), 1, "one eigenmode plane");
        Self { eigenmode, weights }
    }

    pub fn eigenmode(&self) -> &Stack<C64> {
        &self.eigenmode
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// `views` (whole frames, in order) with probe plane `b·nprobe + m` for the `b`-th frame of
    /// `views` and probe mode `m`, and those planes: `P_0 + w_frame·U`, then `P_1 ..`.
    pub fn unique(&self, views: &[View], probe: &Probe) -> (Vec<View>, Probe) {
        assert_eq!(probe.plane_shape(), self.eigenmode.plane_shape());
        let np = probe.len();
        let mut frames: Vec<u32> = Vec::new();
        let views = views
            .iter()
            .map(|v| {
                if frames.last() != Some(&v.frame) {
                    frames.push(v.frame);
                }
                View {
                    probe: ((frames.len() - 1) * np) as u32 + v.probe,
                    ..*v
                }
            })
            .collect();
        let mut data = Vec::with_capacity(frames.len() * probe.as_slice().len());
        for &f in &frames {
            let w = self.weights[f as usize];
            let u = self.eigenmode.as_slice();
            data.extend(probe.plane(0).iter().zip(u).map(|(p, u)| p + w * u));
            (1..np).for_each(|m| data.extend_from_slice(probe.plane(m)));
        }
        let [_, ny, nx] = probe.dims();
        let planes = Stack::from_vec([frames.len() * np, ny, nx], data).unwrap();
        (views, Probe::new(planes))
    }
}

/// The frames of `views` (whole frames) in order.
fn frames(views: &[View]) -> Vec<usize> {
    let mut f: Vec<usize> = views.iter().map(|v| v.frame as usize).collect();
    f.dedup();
    f
}

/// `Σ Re(conj(a)·b)`.
fn dot(a: &[C64], b: &[C64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a.conj() * b).re).sum()
}

/// `√(mean |x|²)`.
fn mnorm(x: &[C64]) -> f64 {
    (x.iter().map(|c| c.norm_sqr()).sum::<f64>() / x.len() as f64).sqrt()
}

/// `∂cost/∂w_k` for every frame `k` of `views` (whole frames, in order), `cost` summed over
/// the unmasked pixels of those frames with `icalc = Σ_m |psi_km|²` from the per-frame probes:
/// `Σ_pixels slope·2 Re(conj(psi_k0)·F(U·q·V_k))`, taken as `⟨U, g_k0⟩` with `g` the gradient
/// over each frame's own probe planes, [`accumulate_probe`] of `F⁻¹(2·slope·psi)`.
pub fn opr_weight_grad(
    cost: Cost,
    views: &[View],
    obs: &Observed,
    quad: &[C64],
    object: &Object,
    probe: &Probe,
    var: &ProbeVariation,
) -> Vec<f64> {
    let np = probe.len();
    let (views, probe) = var.unique(views, probe);
    let n = probe.plane_len();
    let frames = frames(&views);
    let nmode = views.len() / frames.len();
    let fft = Fft2::new(probe.plane_shape());
    let mut psi = vec![C64::default(); views.len() * n];
    obj_probe_to_psi(&views, object, &probe, quad, &mut psi);
    fft.forward(&mut psi);
    for (&f, block) in frames.iter().zip(psi.chunks_exact_mut(nmode * n)) {
        for (j, iobs) in obs.frames(f..f + 1).iter().enumerate() {
            let s = iobs.map_or(0.0, |iobs| {
                let icalc = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                2.0 * cost.slope(iobs, icalc)
            });
            (0..nmode).for_each(|m| block[m * n + j] *= s);
        }
    }
    fft.inverse(&mut psi);
    let (g, _) = accumulate_probe(&views, object, probe.dims(), quad, &psi);
    (0..frames.len())
        .map(|b| dot(var.eigenmode.as_slice(), g.plane(b * np)))
        .collect()
}

/// One batch's OPR step.
#[derive(Clone, Debug, PartialEq)]
pub struct OprStep {
    pub first: usize,
    pub count: usize,
    /// pty-chi `alpha_psi_far` per frame of the batch, as [`LsqmlBatch::a_far`].
    pub a_far: Vec<f64>,
}

/// pty-chi `OPRModeWeights.update_opr_probe_modes_and_weights` (from PtychoShelves'
/// `update_variable_probe.m`, relaxations at their default 1) on the frames of `views` (whole
/// frames, in order), moving `var`'s eigenmode and those frames' weights. With `chi` the
/// [`LsqmlBatch`] exit-wave update at the per-frame probes and over the batch's `B` frames:
///
/// `d_k = conj(q·V_k)·chi_k0`, `r_k = d_k − mean d`,
/// `p_k = (Σ Re(conj(r_k)·U)/N² + w_k) / Σ w²`, `e = mean(r_k·p_k)`,
/// `U′ = U + min(0.1, B/nframe)·√N·e/mnorm(e)`, `U″ = √N·U′/mnorm(U′)`,
/// `w_k += Σ Re(conj(U″)·d_k) / (Σ |U″·V_k|² + 0.1·mean_batch Σ |U″·V_k|²)`
///
/// with `N` the probe plane's pixel count and `mnorm(x) = √(mean |x|²)`, so `mnorm(U″) = √N`
/// (pty-chi's eigenmode, `U·q/√N`, has mnorm 1). A one-frame batch moves nothing, as in
/// pty-chi, where its residues `r` vanish.
pub fn opr_step(
    cost: Cost,
    views: &[View],
    obs: &Observed,
    quad: &[C64],
    object: &Object,
    probe: &Probe,
    var: &mut ProbeVariation,
) -> OprStep {
    let (uviews, uprobe) = var.unique(views, probe);
    let b = LsqmlBatch::new(cost, &uviews, obs, quad, object, &uprobe);
    let frames = frames(views);
    let step = OprStep {
        first: frames[0],
        count: frames.len(),
        a_far: b.a_far().to_vec(),
    };
    if frames.len() == 1 {
        return step;
    }
    let (d, vv) = accumulate_probe(&uviews, object, uprobe.dims(), quad, b.chi());
    let (np, n, count) = (probe.len(), probe.plane_len(), frames.len() as f64);
    let d0 = |b: usize| d.plane(b * np);
    let mut mean = vec![C64::default(); n];
    for b in 0..frames.len() {
        mean.iter_mut().zip(d0(b)).for_each(|(m, d)| *m += d);
    }
    mean.iter_mut().for_each(|m| *m /= count);
    let u = var.eigenmode.as_slice();
    let w2: f64 = frames.iter().map(|&f| var.weights[f].powi(2)).sum();
    let nn = n as f64;
    let mut e = vec![C64::default(); n];
    for (b, &f) in frames.iter().enumerate() {
        let r: Vec<C64> = d0(b).iter().zip(&mean).map(|(d, m)| d - m).collect();
        let p = (dot(&r, u) / (nn * nn) + var.weights[f]) / w2;
        e.iter_mut().zip(&r).for_each(|(e, r)| *e += r * p);
    }
    e.iter_mut().for_each(|e| *e /= count);
    let a_u = 0.1f64.min(count / var.weights.len() as f64);
    let s = a_u * nn.sqrt() / mnorm(&e);
    let mut u: Vec<C64> = u.iter().zip(&e).map(|(u, e)| u + s * e).collect();
    let s = nn.sqrt() / mnorm(&u);
    u.iter_mut().for_each(|u| *u *= s);
    let den: Vec<f64> = (0..frames.len())
        .map(|b| {
            let vv = vv.plane(b * np);
            u.iter().zip(vv).map(|(u, v)| u.norm_sqr() * v).sum()
        })
        .collect();
    let reg = 0.1 * den.iter().sum::<f64>() / count;
    for (b, &f) in frames.iter().enumerate() {
        var.weights[f] += dot(&u, d0(b)) / (den[b] + reg);
    }
    var.eigenmode.as_mut_slice().copy_from_slice(&u);
    step
}
