//! pty-chi's orthogonal probe relaxation (OPR, Odstrčil et al. 2016) of probe mode 0 as CPU f64
//! operators over the arrays and layout of [`crate::engine::ops`]: with `k` eigenmodes `U_i`, frame `f`
//! sees `P_f0 = c_f·P_0 + Σ_i w_fi·U_i` and `P_fm = P_m` for `m ≥ 1`. The main mode's weight
//! `c_f` is 1 unless refined ([`ProbeVariation::with_intensity`], pty-chi
//! `optimize_intensity_variation`); its `primary_mode_weight_floor`, off by default, is not
//! modelled.
//!
//! [`ProbeVariation::unique`] gives every frame of a batch its own probe planes, so the
//! operators of [`crate::engine::ops`] and [`crate::engine::LsqmlBatch`] run on per-frame probes unchanged;
//! [`crate::engine::Lsqml::step`] moves the eigenmodes and weights with the object and probe.

use crate::base::{C64, Object, Probe, Stack, View};

use crate::engine::ops::{accumulate_probe, obj_probe_to_psi};
use crate::engine::{Cost, Fft2, Observed};

/// `k` eigenmodes `U_i` of probe mode 0, shaped as probe planes, their weights for every
/// frame of the scan and, when refined, the main mode's weight `c_f` of every frame; `k = 0` is
/// the shared probe.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeVariation {
    eigenmodes: Stack<C64>,
    /// Frame-major: `w_fi` at `f·k + i`.
    weights: Vec<f64>,
    /// `c_f`, 1 when `None`.
    intensity: Option<Vec<f64>>,
}

impl ProbeVariation {
    /// `eigenmodes` `[k, ny, nx]`; `weights[f·k + i]` the weight of eigenmode `i` at frame `f`,
    /// for every frame.
    pub fn new(eigenmodes: Stack<C64>, weights: Vec<f64>) -> Self {
        let k = eigenmodes.len();
        assert!(
            if k == 0 {
                weights.is_empty()
            } else {
                weights.len().is_multiple_of(k)
            },
            "{} weights for {k} eigenmodes",
            weights.len()
        );
        Self {
            eigenmodes,
            weights,
            intensity: None,
        }
    }

    /// With the main mode's weights refined from `c[f]` at frame `f`. Needs an eigenmode: pty-chi
    /// weights the main mode only with one.
    pub fn with_intensity(self, c: Vec<f64>) -> Self {
        let k = self.eigenmodes.len();
        assert!(k > 0, "main-mode weights without eigenmodes");
        assert_eq!(
            c.len(),
            self.weights.len() / k,
            "one main-mode weight per frame"
        );
        Self {
            intensity: Some(c),
            ..self
        }
    }

    /// No eigenmode, for probe planes shaped `plane_shape`: every frame sees the shared probe.
    pub fn none(plane_shape: [usize; 2]) -> Self {
        let [ny, nx] = plane_shape;
        Self::new(
            Stack::from_vec([0, ny, nx], Vec::new()).unwrap(),
            Vec::new(),
        )
    }

    pub fn eigenmodes(&self) -> &Stack<C64> {
        &self.eigenmodes
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// `c_f` of every frame when refined.
    pub fn intensity(&self) -> Option<&[f64]> {
        self.intensity.as_deref()
    }

    /// `views` (whole frames, in order) with probe plane `b·nprobe + m` for the `b`-th frame of
    /// `views` and probe mode `m`, and those planes: `c_f·P_0 + w_f0·U_0 + w_f1·U_1 + …` in
    /// that order, then `P_1 ..`.
    pub fn unique(&self, views: &[View], probe: &Probe) -> (Vec<View>, Probe) {
        assert_eq!(probe.plane_shape(), self.eigenmodes.plane_shape());
        let (np, k) = (probe.len(), self.eigenmodes.len());
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
            let start = data.len();
            let c = self.intensity.as_ref().map_or(1.0, |c| c[f as usize]);
            data.extend(probe.plane(0).iter().map(|p| c * p));
            for i in 0..k {
                let w = self.weights[f as usize * k + i];
                data[start..]
                    .iter_mut()
                    .zip(self.eigenmodes.plane(i))
                    .for_each(|(p, u)| *p += w * u);
            }
            (1..np).for_each(|m| data.extend_from_slice(probe.plane(m)));
        }
        let [_, ny, nx] = probe.dims();
        let planes = Stack::from_vec([frames.len() * np, ny, nx], data).unwrap();
        (views, Probe::new(planes))
    }

    /// pty-chi `OPRModeWeights.update_opr_probe_modes_and_weights` (from PtychoShelves'
    /// `update_variable_probe.m`, relaxations at their default 1) on the batch `frames`, from
    /// [`accumulate_probe`] of its exit-wave update `chi` over the [`Self::unique`] views:
    /// `d` and `vv`, plane `b·nprobe` holding `d_b = conj(q·V_b)·chi_b0` and `|V_b|²` of its
    /// `b`-th frame, `probe0` the shared probe's mode 0 `P_0`, and `moved`, `d` taken back
    /// through the frames' probe moves of
    /// [`crate::engine::Features::position`] (`d` itself without). With `B` frames, `N` pixels
    /// per plane and `mnorm(x) = √(mean |x|²)`, from `r_b = m_b − mean m`, `m_b` plane `b·nprobe`
    /// of `moved`, for each eigenmode `i` in order, from its weights on entry:
    ///
    /// `p_b = (Σ Re(conj(r_b)·U_i)/N² + w_bi) / Σ_b w_bi²`, `e = mean(r_b·p_b)`,
    /// `U′ = U_i + min(0.1, B/nframe)·√N·e/mnorm(e)`, `U_i ← U″ = √N·U′/mnorm(U′)`,
    /// `w_bi += Σ Re(conj(U″)·d_b) / (Σ |U″|²·vv_b + 0.1·mean_b Σ |U″|²·vv_b)`,
    /// then, before the next eigenmode, `r_b −= (Σ r_b·conj(U″) / Σ |U″|²)·U″`.
    ///
    /// pty-chi's eigenmode, `U·q/√N`, has mnorm 1. A one-frame batch moves no eigenmode or
    /// `w`, as in pty-chi, where its residues `r` vanish.
    ///
    /// Refined main-mode weights take pty-chi's `_calculate_intensity_variation_update_direction`
    /// and its SGD step, from `d` without the moves, with unit-modulus `q`:
    /// `c_b += 0.1·Σ Re(conj(P_0)·d_b) / Σ |P_0|²·vv_b`.
    pub(crate) fn step(
        &mut self,
        frames: &[usize],
        probe0: &[C64],
        moved: &Stack<C64>,
        d: &Stack<C64>,
        vv: &Stack<f64>,
        np: usize,
    ) {
        if let Some(c) = &mut self.intensity {
            let pp: Vec<f64> = probe0.iter().map(|p| p.norm_sqr()).collect();
            for (b, &f) in frames.iter().enumerate() {
                let den: f64 = pp.iter().zip(vv.plane(b * np)).map(|(p, v)| p * v).sum();
                c[f] += 0.1 * dot(probe0, d.plane(b * np)) / den;
            }
        }
        let (k, count) = (self.eigenmodes.len(), frames.len());
        if k == 0 || count < 2 {
            return;
        }
        let n = d.plane_len();
        let (nn, cnt) = (n as f64, count as f64);
        let (m0, d0) = (|b: usize| moved.plane(b * np), |b: usize| d.plane(b * np));
        let mut mean = vec![C64::default(); n];
        for b in 0..count {
            mean.iter_mut().zip(m0(b)).for_each(|(m, d)| *m += d);
        }
        mean.iter_mut().for_each(|m| *m /= cnt);
        let mut r: Vec<Vec<C64>> = (0..count)
            .map(|b| m0(b).iter().zip(&mean).map(|(d, m)| d - m).collect())
            .collect();
        let a_u = 0.1f64.min(cnt / (self.weights.len() / k) as f64);
        for i in 0..k {
            let w: Vec<f64> = frames.iter().map(|&f| self.weights[f * k + i]).collect();
            let w2: f64 = w.iter().map(|w| w * w).sum();
            let u = self.eigenmodes.plane(i);
            let mut e = vec![C64::default(); n];
            for (r, w) in r.iter().zip(&w) {
                let p = (dot(r, u) / (nn * nn) + w) / w2;
                e.iter_mut().zip(r).for_each(|(e, r)| *e += r * p);
            }
            e.iter_mut().for_each(|e| *e /= cnt);
            let s = a_u * nn.sqrt() / mnorm(&e);
            let mut u: Vec<C64> = u.iter().zip(&e).map(|(u, e)| u + s * e).collect();
            let s = nn.sqrt() / mnorm(&u);
            u.iter_mut().for_each(|u| *u *= s);
            let den: Vec<f64> = (0..count)
                .map(|b| {
                    let vv = vv.plane(b * np);
                    u.iter().zip(vv).map(|(u, v)| u.norm_sqr() * v).sum()
                })
                .collect();
            let reg = 0.1 * den.iter().sum::<f64>() / cnt;
            for (b, &f) in frames.iter().enumerate() {
                self.weights[f * k + i] = w[b] + dot(&u, d0(b)) / (den[b] + reg);
            }
            if i + 1 < k {
                let uu: f64 = u.iter().map(|u| u.norm_sqr()).sum();
                for r in &mut r {
                    let c = r.iter().zip(&u).map(|(r, u)| r * u.conj()).sum::<C64>() / uu;
                    r.iter_mut().zip(&u).for_each(|(r, u)| *r -= c * u);
                }
            }
            self.eigenmodes.plane_mut(i).copy_from_slice(&u);
        }
    }
}

/// `Σ Re(conj(a)·b)`.
fn dot(a: &[C64], b: &[C64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a.conj() * b).re).sum()
}

/// `√(mean |x|²)`.
fn mnorm(x: &[C64]) -> f64 {
    (x.iter().map(|c| c.norm_sqr()).sum::<f64>() / x.len() as f64).sqrt()
}

/// `∂cost/∂w_bi` for every frame `b` of `views` (whole frames, in order) and eigenmode `i`,
/// frame-major as [`ProbeVariation::weights`], `cost` summed over the unmasked pixels of those
/// frames with `icalc = Σ_m |psi_bm|²` from the per-frame probes:
/// `Σ_pixels slope·2 Re(conj(psi_b0)·F(U_i·q·V_b))`, taken as `⟨U_i, g_b0⟩` with `g` the
/// gradient over each frame's own probe planes, [`accumulate_probe`] of `F⁻¹(2·slope·psi)`.
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
    let mut frames: Vec<usize> = views.iter().map(|v| v.frame as usize).collect();
    frames.dedup();
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
    let k = var.eigenmodes.len();
    (0..frames.len() * k)
        .map(|bi| dot(var.eigenmodes.plane(bi % k), g.plane(bi / k * np)))
        .collect()
}
