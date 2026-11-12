//! pty-chi LSQML (Odstrčil's least-squares maximum likelihood, `reconstructors/lsqml.py`) as
//! CPU f64 operators over the arrays and layout of [`crate::ops`]: [`LsqmlBatch`] holds one
//! batch's update directions at one state, [`lsqml_step`] takes the step with pty-chi's per-frame
//! 2×2 step sizes, and [`Lsqml`] runs the epochs with pty-chi's object preconditioner.
//!
//! The noise model is a [`Cost`]: [`Cost::Amplitude`] is pty-chi's Gaussian model with weight
//! `1/(2σ)² = 1`.

use ptyrs_core::{AddrTable, C64, Object, Probe, Stack, View};

use crate::ops::{accumulate_obj, accumulate_probe, obj_probe_to_psi};
use crate::{Cost, Fft2, Observed, random_batches};

/// pty-chi's offsets: on `√icalc` in the Gaussian residual, on `icalc` in the Poisson
/// residual and in the Poisson `xi`.
const EPS_AMPLITUDE: f64 = 1e-9;
const EPS_POISSON: f64 = 1e-6;
const EPS_XI: f64 = 1e-5;
/// pty-chi's `λ0` per pixel and mode in the step-size system, in the engine's units.
const LAMBDA0: f64 = 1.2e-7;

/// LSQML parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsqmlParams {
    /// Frames per batch, capped at the scan's frame count.
    pub batch_size: usize,
    pub cost: Cost,
    /// The seed of every epoch's [`random_batches`].
    pub seed: u64,
}

/// One batch's step sizes, per frame of the batch in its order.
#[derive(Clone, Debug, PartialEq)]
pub struct LsqmlStep {
    pub frames: Vec<usize>,
    /// pty-chi `alpha_psi_far`: the far-field step of the exit-wave update.
    pub a_far: Vec<f64>,
    /// Object and probe step sizes from the 2×2 systems; the object moves by their trimmed
    /// mean, the probe by their mean.
    pub a_obj: Vec<f64>,
    pub a_probe: Vec<f64>,
}

/// The update directions of a batch of views at one object and probe. With `psi` the far
/// field of [`obj_probe_to_psi`], `icalc = Σ_m |psi_m|²` and `c` the cost's residual
/// (Gaussian `2(1 − √iobs/(√icalc + 1e-9))`, Poisson `1 − iobs/(icalc + 1e-6)`, 0 where
/// masked), the exit-wave update is `chi_m = F⁻¹(−a_far·c·psi_m)`. `views` are whole frames of
/// an [`AddrTable`], each frame's views in its order.
pub struct LsqmlBatch<'a> {
    views: &'a [View],
    quad: &'a [C64],
    object: &'a Object,
    probe: &'a Probe,
    nmode: usize,
    a_far: Vec<f64>,
    /// `chi`, one shifted object-plane plane per view.
    chi: Vec<C64>,
}

impl<'a> LsqmlBatch<'a> {
    /// `a_far` is 1/2 for the Gaussian model and PtychoShelves' `xi` solver for the Poisson
    /// one: two passes from 1/2 of `a = clip(a′/2 + Σ xi·(icalc − iobs/(1 − a′·xi)) /
    /// Σ xi²·icalc / 2, 0, 1)` per frame, with `xi = 1 − iobs/(icalc + 1e-5)` and `a′` the batch
    /// mean of the last pass (pty-chi's per-frame jitter of `1e-2·randn` left out).
    pub fn new(
        cost: Cost,
        views: &'a [View],
        obs: &Observed,
        quad: &'a [C64],
        object: &'a Object,
        probe: &'a Probe,
    ) -> Self {
        let n = probe.plane_len();
        let nmode = views
            .iter()
            .take_while(|v| v.frame == views[0].frame)
            .count();
        assert_eq!(views.len() % nmode, 0, "views must be whole frames");
        let fft = Fft2::new(probe.plane_shape());
        let mut chi = vec![C64::default(); views.len() * n];
        obj_probe_to_psi(views, object, probe, quad, &mut chi);
        fft.forward(&mut chi);
        let frames: Vec<&[Option<f64>]> = views
            .chunks_exact(nmode)
            .map(|fv| {
                let f = fv[0].frame as usize;
                assert!(fv.iter().all(|v| v.frame == fv[0].frame));
                obs.frames(f..f + 1)
            })
            .collect();
        let icalc: Vec<f64> = chi
            .chunks_exact(nmode * n)
            .flat_map(|block| {
                (0..n).map(move |j| (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum())
            })
            .collect();
        let mut a_far = vec![0.5; frames.len()];
        if cost == Cost::Poisson {
            for _ in 0..2 {
                let prev = a_far.iter().sum::<f64>() / a_far.len() as f64;
                for (a, (iobs, icalc)) in a_far
                    .iter_mut()
                    .zip(frames.iter().zip(icalc.chunks_exact(n)))
                {
                    let (mut num, mut den) = (0.0, 0.0);
                    for (&yt, &yp) in iobs.iter().zip(icalc) {
                        let Some(yt) = yt else { continue };
                        let xi = 1.0 - yt / (yp + EPS_XI);
                        num += xi * (yp - yt / (1.0 - prev * xi));
                        den += xi * xi * yp;
                    }
                    let ratio = if den > 0.0 { num / den } else { 0.0 };
                    *a = (0.5 * prev + 0.5 * ratio).clamp(0.0, 1.0);
                }
            }
        }
        for (k, block) in chi.chunks_exact_mut(nmode * n).enumerate() {
            let (iobs, icalc) = (frames[k], &icalc[k * n..(k + 1) * n]);
            for (j, (&yt, &yp)) in iobs.iter().zip(icalc).enumerate() {
                let c = match (cost, yt) {
                    (_, None) => 0.0,
                    (Cost::Amplitude, Some(yt)) => {
                        2.0 * (1.0 - yt.max(0.0).sqrt() / (yp.sqrt() + EPS_AMPLITUDE))
                    }
                    (Cost::Poisson, Some(yt)) => 1.0 - yt / (yp + EPS_POISSON),
                };
                (0..nmode).for_each(|m| block[m * n + j] *= -a_far[k] * c);
            }
        }
        fft.inverse(&mut chi);
        Self {
            views,
            quad,
            object,
            probe,
            nmode,
            a_far,
            chi,
        }
    }

    /// `a_far` per frame.
    pub fn a_far(&self) -> &[f64] {
        &self.a_far
    }

    /// `chi`, one shifted object-plane plane per view.
    pub(crate) fn chi(&self) -> &[C64] {
        &self.chi
    }

    /// pty-chi `delta_o_comb`, the object direction before preconditioning:
    /// `Σ_views conj(P_m·q)·chi_m` scattered onto the object ([`accumulate_obj`] of `chi`).
    pub fn obj_direction(&self) -> Stack<C64> {
        accumulate_obj(
            self.views,
            self.object.dims(),
            self.probe,
            self.quad,
            &self.chi,
        )
        .0
    }

    /// pty-chi `delta_p_hat`: `conj(q)·mean_frames(conj(O_view)·chi_m)` per mode
    /// ([`accumulate_probe`] of `chi` over the frame count).
    pub fn probe_direction(&self) -> Stack<C64> {
        let count = (self.views.len() / self.nmode) as f64;
        let mut d = accumulate_probe(
            self.views,
            self.object,
            self.probe.dims(),
            self.quad,
            &self.chi,
        )
        .0;
        d.as_mut_slice().iter_mut().for_each(|c| *c /= count);
        d
    }

    /// Per-frame `[a_obj, a_probe]` along the object and probe directions `d_obj` and
    /// `d_probe`: with `dop` and `dpo` the exit-wave changes of [`obj_probe_to_psi`] along
    /// each, the frame's system `[[a11, a12], [conj a12, a22]]·x = b` over its modes and
    /// pixels with `a11 = Σ(|dop|² + 1.2e-7)` and `a22 = Σ(|dpo|² + 1.2e-7)`, each then plus
    /// 0.1 of its batch mean, `a12 = Σ dop·conj(dpo)`, `b = [Σ Re(conj(dop)·chi),
    /// Σ Re(conj(dpo)·chi)]`; `a_obj = 0.9·max(Re x0, 0)/nmode`, `a_probe = 0.9·max(Re x1, 0)`.
    pub fn step_sizes(&self, d_obj: &Object, d_probe: &Probe) -> [Vec<f64>; 2] {
        let mut dop = vec![C64::default(); self.chi.len()];
        let mut dpo = vec![C64::default(); self.chi.len()];
        obj_probe_to_psi(self.views, d_obj, self.probe, self.quad, &mut dop);
        obj_probe_to_psi(self.views, self.object, d_probe, self.quad, &mut dpo);
        let block = self.nmode * self.probe.plane_len();
        let sums: Vec<[f64; 6]> = dop
            .chunks_exact(block)
            .zip(dpo.chunks_exact(block))
            .zip(self.chi.chunks_exact(block))
            .map(|((dop, dpo), chi)| {
                let mut s = [0.0; 6];
                for ((u, v), c) in dop.iter().zip(dpo).zip(chi) {
                    let a12 = u * v.conj();
                    s[0] += u.norm_sqr() + LAMBDA0;
                    s[1] += v.norm_sqr() + LAMBDA0;
                    s[2] += a12.re;
                    s[3] += a12.im;
                    s[4] += (u.conj() * c).re;
                    s[5] += (v.conj() * c).re;
                }
                s
            })
            .collect();
        let mean = |k: usize| sums.iter().map(|s| s[k]).sum::<f64>() / sums.len() as f64;
        let (reg11, reg22) = (0.1 * mean(0), 0.1 * mean(1));
        let nmode = self.nmode as f64;
        let (mut a_obj, mut a_probe) = (Vec::new(), Vec::new());
        for s in &sums {
            let (a11, a22, a12) = (s[0] + reg11, s[1] + reg22, C64::new(s[2], s[3]));
            // the real part of the Hermitian system's solution
            let det = a11 * a22 - a12.norm_sqr();
            let x0 = (a22 * s[4] - a12.re * s[5]) / det;
            let x1 = (a11 * s[5] - a12.re * s[4]) / det;
            a_obj.push(0.9 * x0.max(0.0) / nmode);
            a_probe.push(0.9 * x1.max(0.0));
        }
        [a_obj, a_probe]
    }
}

/// pty-chi's illumination map over `views`: `Σ |P[probe_k]|²` on each view's window, shaped
/// `obj_dims`.
pub fn illumination(views: &[View], obj_dims: [usize; 3], probe: &Probe) -> Stack<f64> {
    let [ny, nx] = probe.plane_shape();
    let nxo = obj_dims[2];
    let mut out = Stack::filled(obj_dims, 0.0);
    for v in views {
        let (p, out) = (probe.plane(v.probe as usize), out.plane_mut(v.obj as usize));
        for y in 0..ny {
            let row = (v.oy as usize + y) * nxo + v.ox as usize;
            for x in 0..nx {
                out[row + x] += p[y * nx + x].norm_sqr();
            }
        }
    }
    out
}

/// MATLAB `trimmean(x, 10)` as pty-chi's `trim_mean`: the mean without the
/// `ceil(n·0.05 − 0.5)` smallest and largest values.
fn trim_mean(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let k = (s.len() as f64 * 0.1 / 2.0 - 0.5).ceil() as usize;
    let kept = &s[k..s.len() - k];
    kept.iter().sum::<f64>() / kept.len() as f64
}

/// pty-chi `LSQMLReconstructor.run_minibatch` on `views` (object and probe only): with
/// [`LsqmlBatch`]'s directions, `dO = obj_direction / √(pre² + (0.1·max pre)²)` and
/// `dP = probe_direction`, `O += trim_mean(a_obj)·dO` and `P += mean(a_probe)·dP`, both from
/// the state on entry. `preconditioner` is shaped as the object.
pub fn lsqml_step(
    cost: Cost,
    views: &[View],
    obs: &Observed,
    quad: &[C64],
    preconditioner: &Stack<f64>,
    object: &mut Object,
    probe: &mut Probe,
) -> LsqmlStep {
    let b = LsqmlBatch::new(cost, views, obs, quad, object, probe);
    let floor = (0.1
        * preconditioner
            .as_slice()
            .iter()
            .fold(0.0, |a: f64, &b| a.max(b)))
    .powi(2);
    let mut d_obj = b.obj_direction();
    for (d, pre) in d_obj
        .as_mut_slice()
        .iter_mut()
        .zip(preconditioner.as_slice())
    {
        *d /= (pre * pre + floor).sqrt();
    }
    let (d_obj, d_probe) = (Object::new(d_obj), Probe::new(b.probe_direction()));
    let [a_obj, a_probe] = b.step_sizes(&d_obj, &d_probe);
    let a_far = b.a_far().to_vec();
    let frames = views
        .chunks_exact(b.nmode)
        .map(|v| v[0].frame as usize)
        .collect();
    let (so, sp) = (
        trim_mean(&a_obj),
        a_probe.iter().sum::<f64>() / a_probe.len() as f64,
    );
    for (x, d, s) in [
        (object.as_mut_slice(), d_obj.as_slice(), so),
        (probe.as_mut_slice(), d_probe.as_slice(), sp),
    ] {
        x.iter_mut().zip(d).for_each(|(x, d)| *x += s * d);
    }
    LsqmlStep {
        frames,
        a_far,
        a_obj,
        a_probe,
    }
}

/// LSQML state across epochs: the parameters, the epoch count and pty-chi's object
/// preconditioner.
pub struct Lsqml {
    params: LsqmlParams,
    epochs: usize,
    preconditioner: Option<Stack<f64>>,
}

impl Lsqml {
    pub fn new(params: LsqmlParams) -> Self {
        assert!(params.batch_size > 0, "batch_size must be positive");
        Self {
            params,
            epochs: 0,
            preconditioner: None,
        }
    }

    /// The preconditioner of the last epoch, `None` before the first.
    pub fn preconditioner(&self) -> Option<&Stack<f64>> {
        self.preconditioner.as_ref()
    }

    /// One epoch of [`lsqml_step`] over [`random_batches`] of `addr`'s frames, from the
    /// params' seed and this epoch's index. The preconditioner is updated first, on pty-chi's
    /// schedule with the probe optimised from epoch 0 (`Object.update_preconditioner`, epochs
    /// 0–9 and every tenth): `pre = (pre + I)/2` with `I` the [`illumination`] of every view,
    /// starting from `I` of the probe PtychoShelves renormalises from the data,
    /// `I·max_f Σ iobs_f / (4·(ny·nx)²)` in the engine's units.
    ///
    /// The start is the caller's object and probe, rescaled as for every algorithm by
    /// [`crate::Ptycho::scale_obj_probe`]. Departures from a pty-chi epoch: the batches come from
    /// [`random_batches`]' generator, not torch's, each in scan order; the first epoch updates
    /// the object and probe (pty-chi's default only rescales the probe); no Poisson `a_far`
    /// jitter; no post-epoch hook.
    pub fn epoch(
        &mut self,
        addr: &AddrTable,
        obs: &Observed,
        quad: &[C64],
        object: &mut Object,
        probe: &mut Probe,
    ) -> Vec<LsqmlStep> {
        let (nframe, nmode) = (addr.nframe(), addr.nmode());
        if self.epochs < 10 || self.epochs.is_multiple_of(10) {
            let illum = illumination(addr.views(), object.dims(), probe);
            let pre = self.preconditioner.take().unwrap_or_else(|| {
                let n = probe.plane_len() as f64;
                let power = (0..nframe)
                    .map(|f| obs.frames(f..f + 1).iter().flatten().sum::<f64>())
                    .fold(0.0, f64::max);
                let mut init = illum.clone();
                init.as_mut_slice()
                    .iter_mut()
                    .for_each(|v| *v *= power / (4.0 * n * n));
                init
            });
            let data = pre
                .as_slice()
                .iter()
                .zip(illum.as_slice())
                .map(|(p, i)| (p + i) / 2.0)
                .collect();
            self.preconditioner = Some(Stack::from_vec(illum.dims(), data).unwrap());
        }
        let epoch = self.epochs as u64;
        self.epochs += 1;
        let pre = self.preconditioner.as_ref().unwrap();
        let size = self.params.batch_size.min(nframe);
        random_batches(nframe, size, self.params.seed, epoch)
            .iter()
            .map(|frames| {
                let views: Vec<View> = frames
                    .iter()
                    .flat_map(|&f| &addr.views()[f * nmode..(f + 1) * nmode])
                    .copied()
                    .collect();
                lsqml_step(self.params.cost, &views, obs, quad, pre, object, probe)
            })
            .collect()
    }
}
