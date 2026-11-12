//! pty-chi LSQML (Odstrčil's least-squares maximum likelihood, `reconstructors/lsqml.py`) as
//! CPU f64 operators over the arrays and layout of [`crate::ops`]: [`LsqmlBatch`] holds one
//! batch's update directions at one state, and [`Lsqml`], holding pty-chi's object
//! preconditioner, takes the step with pty-chi's per-frame 2×2 step sizes and the
//! [`ProbeVariation`] step ([`Lsqml::step`]) and runs the epochs.
//!
//! The noise model is a [`Cost`]: [`Cost::Amplitude`] is pty-chi's Gaussian model with weight
//! `1/(2σ)² = 1`. The exit waves reach the detector through an [`Optics`]; pty-chi's probe is
//! the engine's over `√k`, `k` the propagator's [`crate::Propagator::intensity_scale`].

use ptyrs_core::{AddrTable, C64, Object, Probe, Stack, View};

use crate::ops::obj_probe_to_psi;
use crate::{Cost, Observed, Optics, ProbeVariation, SliceWaves, random_batches};

/// pty-chi's offsets: on `√icalc` in the Gaussian residual, on `icalc` in the Poisson
/// residual and in the Poisson `xi`.
const EPS_AMPLITUDE: f64 = 1e-9;
const EPS_POISSON: f64 = 1e-6;
const EPS_XI: f64 = 1e-5;
/// pty-chi's `λ0` per pixel and mode in the step-size system times the probe's pixel count:
/// `λ0 = 1.2e-7/(ny·nx)` on pty-chi's probe, `1.2e-7·k/(ny·nx)` on the engine's.
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
    /// Per object slice, the object step sizes; each slice moves by their trimmed mean.
    pub a_obj: Vec<Vec<f64>>,
    /// The probe step sizes; the probe moves by their mean.
    pub a_probe: Vec<f64>,
}

/// The update directions of a batch of views at one object (one per slice of the [`Optics`]),
/// probe and [`ProbeVariation`]. With `psi` the exit waves of [`SliceWaves`] at each frame's
/// own probe ([`ProbeVariation::unique`]), `U` the propagator, `icalc = Σ_m |U psi_m|²` and `c`
/// the cost's residual (Gaussian `2(1 − √iobs/(√icalc + 1e-9))`, Poisson
/// `1 − iobs/(icalc + 1e-6)`, 0 where masked), the exit-wave update is pty-chi's Eq. 19
/// `chi_m = Uᴴ((1 − a_far·c)·U psi_m) − psi_m`, which also drops the part of `psi` on angular
/// spectrum's evanescent band, taken back to every slice by [`SliceWaves::back_propagate`].
/// `views` are whole frames of an [`AddrTable`], each frame's views in its order.
pub struct LsqmlBatch<'a> {
    /// The shared views and probe, which the probe direction moves.
    views: &'a [View],
    probe: &'a Probe,
    /// The waves over the views and planes of [`ProbeVariation::unique`].
    waves: SliceWaves<'a>,
    objects: &'a [Object],
    optics: &'a Optics,
    frames: Vec<usize>,
    nmode: usize,
    a_far: Vec<f64>,
    /// Per slice, its `chi`: one shifted plane per view.
    chi: Vec<Vec<C64>>,
}

impl<'a> LsqmlBatch<'a> {
    /// The batch of `run`, with its cost. `a_far` is 1/2 for the Gaussian model and
    /// PtychoShelves' `xi` solver for the Poisson one: two passes of `a = clip(a′/2 +
    /// Σ xi·(icalc − iobs/(1 − a′·xi)) / Σ xi²·icalc / 2, 0, 1)` per frame, with
    /// `xi = 1 − iobs/(icalc + 1e-5)` and `a′` the batch mean of the last pass, the first pass
    /// from [`Lsqml::a_far`] of the batch's frames (pty-chi's per-frame jitter of `1e-2·randn`
    /// left out).
    pub fn new(
        run: &Lsqml,
        views: &'a [View],
        obs: &Observed,
        optics: &'a Optics,
        objects: &'a [Object],
        probe: &'a Probe,
        var: &ProbeVariation,
    ) -> Self {
        let cost = run.params.cost;
        let n = probe.plane_len();
        let nmode = views
            .iter()
            .take_while(|v| v.frame == views[0].frame)
            .count();
        assert_eq!(views.len() % nmode, 0, "views must be whole frames");
        let frames: Vec<usize> = views
            .chunks_exact(nmode)
            .map(|fv| {
                assert!(fv.iter().all(|v| v.frame == fv[0].frame));
                fv[0].frame as usize
            })
            .collect();
        let (uviews, uprobe) = var.unique(views, probe);
        let waves = SliceWaves::new(optics, &uviews, objects, &uprobe);
        let prop = optics.propagator();
        let mut chi = waves.psi().to_vec();
        prop.forward(&mut chi);
        let iobs: Vec<&[Option<f64>]> = frames.iter().map(|&f| obs.frames(f..f + 1)).collect();
        let icalc: Vec<f64> = chi
            .chunks_exact(nmode * n)
            .flat_map(|block| {
                (0..n).map(move |j| (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum())
            })
            .collect();
        let mut a_far: Vec<f64> = match cost {
            Cost::Poisson => frames.iter().map(|&f| run.a_far(f)).collect(),
            _ => vec![0.5; frames.len()],
        };
        if cost == Cost::Poisson {
            for _ in 0..2 {
                let prev = a_far.iter().sum::<f64>() / a_far.len() as f64;
                for (a, (iobs, icalc)) in
                    a_far.iter_mut().zip(iobs.iter().zip(icalc.chunks_exact(n)))
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
            let (iobs, icalc) = (iobs[k], &icalc[k * n..(k + 1) * n]);
            for (j, (&yt, &yp)) in iobs.iter().zip(icalc).enumerate() {
                let c = match (cost, yt) {
                    (_, None) => 0.0,
                    (Cost::Amplitude, Some(yt)) => {
                        2.0 * (1.0 - yt.max(0.0).sqrt() / (yp.sqrt() + EPS_AMPLITUDE))
                    }
                    (Cost::Poisson, Some(yt)) => 1.0 - yt / (yp + EPS_POISSON),
                };
                (0..nmode).for_each(|m| block[m * n + j] *= 1.0 - a_far[k] * c);
            }
        }
        prop.adjoint(&mut chi);
        chi.iter_mut().zip(waves.psi()).for_each(|(c, p)| *c -= p);
        let chi = waves.back_propagate(&chi);
        Self {
            views,
            probe,
            waves,
            objects,
            optics,
            frames,
            nmode,
            a_far,
            chi,
        }
    }

    /// The batch's frames in order.
    pub fn frames(&self) -> &[usize] {
        &self.frames
    }

    /// `a_far` per frame.
    pub fn a_far(&self) -> &[f64] {
        &self.a_far
    }

    /// pty-chi `delta_o_comb` per slice, the object directions before preconditioning:
    /// [`SliceWaves::obj_adjoint`] of each slice's `chi`, at a single slice
    /// `Σ_views conj(P_fm·q)·chi_m` at each frame's own probe.
    pub fn obj_directions(&self) -> Vec<Stack<C64>> {
        (0..self.chi.len())
            .map(|s| self.waves.obj_adjoint(s, &self.chi[s]))
            .collect()
    }

    /// pty-chi `delta_p_i` and `|O_view|²` on slice 0: plane `b·nprobe + m` holds
    /// `conj(q·O_view)·chi_m` of the batch's `b`-th frame and probe mode `m`
    /// ([`SliceWaves::probe_adjoint`] over the [`ProbeVariation::unique`] views).
    pub fn frame_probe_directions(&self) -> (Stack<C64>, Stack<f64>) {
        self.waves.probe_adjoint(&self.chi[0])
    }

    /// pty-chi `delta_p_hat`: the frame mean of [`Self::frame_probe_directions`] per mode.
    pub fn probe_direction(&self) -> Stack<C64> {
        frame_mean(&self.frame_probe_directions().0, self.probe.len())
    }

    /// Per slice and frame `a_obj`, and per frame `a_probe`, along the object directions
    /// `d_obj` (one per slice) and the probe direction `d_probe`, from each slice's `chi_s`
    /// and `dop_s`, its [`SliceWaves::obj_tangent`] along `d_obj[s]`.
    ///
    /// Slice 0 solves both jointly, pty-chi's Eq. 22 with `dpo` the change of `psi_0` along
    /// `d_probe` from the shared views: the frame's system `[[a11, a12], [conj a12, a22]]·x = b`
    /// over its modes and pixels with `a11 = Σ(|dop|² + λ0)` and `a22 = Σ(|dpo|² + λ0)`,
    /// `λ0 = 1.2e-7·k/(ny·nx)`, each then plus 0.1 of its batch mean, `a12 = Σ dop·conj(dpo)`,
    /// `b = [Σ Re(conj(dop)·chi), Σ Re(conj(dpo)·chi)]`; `a_obj = 0.9·max(Re x0, 0)/(n·nmode)`
    /// over `n` slices, `a_probe = 0.9·max(Re x1, 0)`. Every other slice takes Eq. 23b,
    /// `a_obj = max(0.9·Σ Re(conj(dop)·chi)/(2·Σ |dop|²), 0)/(n·nmode)`. This is pty-chi with
    /// `multislice_solve_obj_prb_step_size_jointly`; its default solves slice 0 by Eq. 23a/b
    /// as well whenever there is more than one slice.
    pub fn step_sizes(&self, d_obj: &[Object], d_probe: &Probe) -> (Vec<Vec<f64>>, Vec<f64>) {
        assert_eq!(d_obj.len(), self.objects.len());
        let n = self.probe.plane_len();
        let block = self.nmode * n;
        let scale = (self.objects.len() * self.nmode) as f64;
        let mut dpo = vec![C64::default(); self.views.len() * n];
        obj_probe_to_psi(
            self.views,
            &self.objects[0],
            d_probe,
            self.waves.quad(0),
            &mut dpo,
        );
        let dop = self.waves.obj_tangent(0, &d_obj[0]);
        let lambda0 = LAMBDA0 * self.optics.propagator().intensity_scale() / n as f64;
        let sums: Vec<[f64; 6]> = dop
            .chunks_exact(block)
            .zip(dpo.chunks_exact(block))
            .zip(self.chi[0].chunks_exact(block))
            .map(|((dop, dpo), chi)| {
                let mut s = [0.0; 6];
                for ((u, v), c) in dop.iter().zip(dpo).zip(chi) {
                    let a12 = u * v.conj();
                    s[0] += u.norm_sqr() + lambda0;
                    s[1] += v.norm_sqr() + lambda0;
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
        let (mut a_obj0, mut a_probe) = (Vec::new(), Vec::new());
        for s in &sums {
            let (a11, a22, a12) = (s[0] + reg11, s[1] + reg22, C64::new(s[2], s[3]));
            // the real part of the Hermitian system's solution
            let det = a11 * a22 - a12.norm_sqr();
            let x0 = (a22 * s[4] - a12.re * s[5]) / det;
            let x1 = (a11 * s[5] - a12.re * s[4]) / det;
            a_obj0.push(0.9 * x0.max(0.0) / scale);
            a_probe.push(0.9 * x1.max(0.0));
        }
        let mut a_obj = vec![a_obj0];
        for (s, d) in d_obj.iter().enumerate().skip(1) {
            let dop = self.waves.obj_tangent(s, d);
            let a = dop
                .chunks_exact(block)
                .zip(self.chi[s].chunks_exact(block))
                .map(|(dop, chi)| {
                    let (mut num, mut den) = (0.0, 0.0);
                    for (u, c) in dop.iter().zip(chi) {
                        num += (u.conj() * c).re;
                        den += u.norm_sqr();
                    }
                    (0.5 * num / den * 0.9 / scale).max(0.0)
                })
                .collect();
            a_obj.push(a);
        }
        (a_obj, a_probe)
    }
}

/// The mean over frames of `d`'s planes `b·np + m` per mode `m`, summed in frame order.
fn frame_mean(d: &Stack<C64>, np: usize) -> Stack<C64> {
    let [_, ny, nx] = d.dims();
    let count = d.len() / np;
    let mut out = Stack::filled([np, ny, nx], C64::default());
    for b in 0..count {
        for m in 0..np {
            out.plane_mut(m)
                .iter_mut()
                .zip(d.plane(b * np + m))
                .for_each(|(o, d)| *o += d);
        }
    }
    out.as_mut_slice()
        .iter_mut()
        .for_each(|c| *c /= count as f64);
    out
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

/// LSQML state across epochs: the parameters, the epoch count, pty-chi's object
/// preconditioner and every frame's `a_far` from its last step.
#[derive(Clone)]
pub struct Lsqml {
    params: LsqmlParams,
    epochs: usize,
    preconditioner: Option<Stack<f64>>,
    /// pty-chi `alpha_psi_far_all_pos`, by frame; frames past its end have not been stepped.
    a_far: Vec<f64>,
}

impl Lsqml {
    pub fn new(params: LsqmlParams) -> Self {
        assert!(params.batch_size > 0, "batch_size must be positive");
        Self {
            params,
            epochs: 0,
            preconditioner: None,
            a_far: Vec::new(),
        }
    }

    /// Frame `f`'s `a_far` from its last [`Self::step`], 1/2 before its first.
    pub fn a_far(&self, f: usize) -> f64 {
        self.a_far.get(f).copied().unwrap_or(0.5)
    }

    /// This run with its preconditioner set to `preconditioner` (shaped as the object), which on
    /// a new run replaces the renormalised probe's illumination, as on resuming one.
    pub fn with_preconditioner(self, preconditioner: Stack<f64>) -> Self {
        Self {
            preconditioner: Some(preconditioner),
            ..self
        }
    }

    /// pty-chi `LSQMLReconstructor.run_minibatch` on `views` (object slices, probe and probe
    /// variation) with the params' cost and the current preconditioner `pre`: with
    /// [`LsqmlBatch`]'s directions, every slice's `dO_s = obj_direction_s /
    /// √(pre² + (0.1·max pre)²)` and `dP = probe_direction`, `O_s += trim_mean(a_obj_s)·dO_s`,
    /// `P += mean(a_probe)·dP` and `var`'s eigenmodes and batch weights by [`ProbeVariation`]'s
    /// OPR step on slice 0, all from the state on entry; the frames' [`Self::a_far`] become the
    /// batch's.
    pub fn step(
        &mut self,
        views: &[View],
        obs: &Observed,
        optics: &Optics,
        objects: &mut [Object],
        probe: &mut Probe,
        var: &mut ProbeVariation,
    ) -> LsqmlStep {
        let pre = self.preconditioner.as_ref();
        let pre = pre
            .expect("no preconditioner before the first epoch")
            .as_slice();
        let floor = (0.1 * pre.iter().fold(0.0, |a: f64, &b| a.max(b))).powi(2);
        let b = LsqmlBatch::new(self, views, obs, optics, objects, probe, var);
        let d_obj: Vec<Object> = b
            .obj_directions()
            .into_iter()
            .map(|mut d| {
                for (d, p) in d.as_mut_slice().iter_mut().zip(pre) {
                    *d /= (p * p + floor).sqrt();
                }
                Object::new(d)
            })
            .collect();
        let np = probe.len();
        let (d, vv) = b.frame_probe_directions();
        let d_probe = Probe::new(frame_mean(&d, np));
        let (a_obj, a_probe) = b.step_sizes(&d_obj, &d_probe);
        let (a_far, frames) = (b.a_far().to_vec(), b.frames().to_vec());
        for (&f, &a) in frames.iter().zip(&a_far) {
            if self.a_far.len() <= f {
                self.a_far.resize(f + 1, 0.5);
            }
            self.a_far[f] = a;
        }
        var.step(&frames, &d, &vv, np);
        for ((o, d), a) in objects.iter_mut().zip(&d_obj).zip(&a_obj) {
            let so = trim_mean(a);
            o.as_mut_slice()
                .iter_mut()
                .zip(d.as_slice())
                .for_each(|(x, d)| *x += so * d);
        }
        let sp = a_probe.iter().sum::<f64>() / a_probe.len() as f64;
        probe
            .as_mut_slice()
            .iter_mut()
            .zip(d_probe.as_slice())
            .for_each(|(x, d)| *x += sp * d);
        LsqmlStep {
            frames,
            a_far,
            a_obj,
            a_probe,
        }
    }

    /// The preconditioner of the last epoch, `None` before the first.
    pub fn preconditioner(&self) -> Option<&Stack<f64>> {
        self.preconditioner.as_ref()
    }

    /// One epoch of [`Self::step`] over [`random_batches`] of `addr`'s frames, from the
    /// params' seed and this epoch's index. The preconditioner, one for every slice, is updated
    /// first, on pty-chi's schedule with the probe optimised from epoch 0
    /// (`Object.update_preconditioner`, epochs 0–9 and every tenth): `pre = (pre + I)/2` with
    /// `I` the [`illumination`] of every view at the shared probe (pty-chi's OPR mode 0),
    /// starting from `I` of the probe PtychoShelves renormalises from the data,
    /// `I·max_f Σ iobs_f / (4·ny·nx·k)` in the engine's units.
    ///
    /// The start is the caller's object and probe as given: an algorithm rescales its start
    /// only where its reference does, as ML does by [`crate::Ptycho::scale_obj_probe`].
    /// Departures from a pty-chi epoch: the batches come from [`random_batches`]' generator, not
    /// torch's, each in scan order; the first epoch updates the object and probe (pty-chi's
    /// default only rescales the probe); no Poisson `a_far` jitter; no post-epoch hook.
    pub fn epoch(
        &mut self,
        addr: &AddrTable,
        obs: &Observed,
        optics: &Optics,
        objects: &mut [Object],
        probe: &mut Probe,
        var: &mut ProbeVariation,
    ) -> Vec<LsqmlStep> {
        let (nframe, nmode) = (addr.nframe(), addr.nmode());
        if self.epochs < 10 || self.epochs.is_multiple_of(10) {
            let illum = illumination(addr.views(), objects[0].dims(), probe);
            let pre = self.preconditioner.take().unwrap_or_else(|| {
                let n = probe.plane_len() as f64;
                let k = optics.propagator().intensity_scale();
                let power = (0..nframe)
                    .map(|f| obs.frames(f..f + 1).iter().flatten().sum::<f64>())
                    .fold(0.0, f64::max);
                let mut init = illum.clone();
                init.as_mut_slice()
                    .iter_mut()
                    .for_each(|v| *v *= power / (4.0 * n * k));
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
        let size = self.params.batch_size.min(nframe);
        random_batches(nframe, size, self.params.seed, epoch)
            .iter()
            .map(|frames| {
                let views: Vec<View> = frames
                    .iter()
                    .flat_map(|&f| &addr.views()[f * nmode..(f + 1) * nmode])
                    .copied()
                    .collect();
                self.step(&views, obs, optics, objects, probe, var)
            })
            .collect()
    }
}
