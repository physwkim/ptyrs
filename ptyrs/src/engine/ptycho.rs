//! [`Ptycho`]: the reconstruction state and the AP, DM and rPIE loops built on the operators.

use std::f64::consts::PI;
use std::ops::Range;

use crate::base::{AddrTable, C64, Object, Probe, Scan, Stack};

use crate::engine::features::{Model, Shifted, position_step};
use crate::engine::ops::{
    Llk, accumulate_obj, accumulate_probe, apply_amplitude_with, cg_poisson_gamma_with, llk_with,
    natural, obj_probe_to_psi, poisson_grad_with, psi_to_obj_probe, quad_phase,
};
use crate::engine::{EngineError, Features, Fft2, Observed, shift};

/// Which loop recorded an [`LlkRecord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    Ap,
    Dm,
    Rpie,
    Ml,
    Bh,
}

/// One LLK measurement (PyNX `history`): normalised LLK ([`Llk::normalised`]) taken at
/// `cycle`, the cycle counter before it is incremented.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlkRecord {
    pub cycle: usize,
    pub algorithm: Algorithm,
    pub llk: Llk,
}

/// Every `calc_llk`-th cycle and the last one record an LLK; 0 never does (PyNX `calc_llk`).
pub(crate) fn llk_due(calc_llk: usize, ic: usize, ncycle: usize) -> bool {
    calc_llk > 0 && (ic.is_multiple_of(calc_llk) || ic + 1 == ncycle)
}

/// PyNX `AP` parameters. An inertia of `None` leaves that array fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApParams {
    /// Object update with this inertia (PyNX default `update_object=True`, 0.01).
    pub obj_inertia: Option<f64>,
    /// Probe update with this inertia (PyNX default `update_probe=False`; inertia 0.001).
    pub probe_inertia: Option<f64>,
    pub calc_llk: usize,
    /// [`Ptycho::zero_phase_ramp`] after the last cycle.
    pub zero_phase_ramp: bool,
}

impl Default for ApParams {
    fn default() -> Self {
        Self {
            obj_inertia: Some(0.01),
            probe_inertia: None,
            calc_llk: 0,
            zero_phase_ramp: true,
        }
    }
}

/// PyNX `DM` parameters. An inertia of `None` leaves that array fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DmParams {
    pub obj_inertia: Option<f64>,
    pub probe_inertia: Option<f64>,
    /// AP/DM mix: 0 is plain DM (PyNX default 0.02).
    pub alpha: f64,
    /// Object-then-probe updates per cycle (PyNX `loop_obj_probe`).
    pub loop_obj_probe: usize,
    pub calc_llk: usize,
    pub zero_phase_ramp: bool,
}

impl Default for DmParams {
    fn default() -> Self {
        Self {
            obj_inertia: Some(0.01),
            probe_inertia: Some(0.001),
            alpha: 0.02,
            loop_obj_probe: 1,
            calc_llk: 0,
            zero_phase_ramp: true,
        }
    }
}

/// PyNX `ML` parameters, Poisson noise without regularisation, background or position
/// updates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlParams {
    pub update_object: bool,
    /// PyNX default `update_probe=False`.
    pub update_probe: bool,
    pub calc_llk: usize,
    pub zero_phase_ramp: bool,
}

impl Default for MlParams {
    fn default() -> Self {
        Self {
            update_object: true,
            update_probe: false,
            calc_llk: 0,
            zero_phase_ramp: true,
        }
    }
}

/// pty-chi `RPIEReconstructor` parameters (SGD, sequential batches). A step of `None` leaves
/// that array fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RpieParams {
    pub obj_step: Option<f64>,
    pub probe_step: Option<f64>,
    /// Regularisation weights `α` of the object and probe step denominators.
    pub obj_alpha: f64,
    pub probe_alpha: f64,
    /// Frames per update; batches are consecutive frames in scan order.
    pub batch_size: usize,
    /// Epochs between LLK records, as `calc_llk` for AP and DM.
    pub calc_llk: usize,
}

impl Default for RpieParams {
    fn default() -> Self {
        Self {
            obj_step: Some(1.0),
            probe_step: Some(1.0),
            obj_alpha: 0.1,
            probe_alpha: 0.1,
            batch_size: 100,
            calc_llk: 0,
        }
    }
}

/// Reconstruction state for one far-field scan: object, probe, the persistent exit-wave stack
/// and the LLK history.
///
/// Positions go through [`crate::base::Positions::pynx_preprocess`] and become an
/// [`AddrTable`] once; the probe frame must match the detector frame. Object mode `m / nprobe`
/// and probe mode `m % nprobe` make view mode `m` (PyNX `_obj_psidx`, `_probe_psidx`).
/// The model intensity, every LLK and gradient included, is that of [`Ptycho::features`].
pub struct Ptycho {
    addr: AddrTable,
    quad: Vec<C64>,
    obs: Observed,
    fft: Fft2,
    object: Object,
    probe: Probe,
    /// Object-plane exit waves, one shifted plane per view; DM carries it across cycles.
    psi: Vec<C64>,
    features: Features,
    cycle: usize,
    history: Vec<LlkRecord>,
}

impl Ptycho {
    pub fn new(scan: &Scan, object: Object, probe: Probe) -> Result<Self, EngineError> {
        let g = scan.geometry;
        if probe.plane_shape() != g.frame {
            return Err(EngineError::Invalid(format!(
                "probe plane {:?} differs from frame {:?}",
                probe.plane_shape(),
                g.frame
            )));
        }
        let pixel = g.pixel_size_object();
        let positions = scan.positions.pynx_preprocess(pixel);
        let addr = AddrTable::new(
            &positions,
            pixel,
            g.frame,
            object.plane_shape(),
            probe.nmode(),
            object.nmode(),
        )?;
        let psi = vec![C64::default(); addr.views().len() * probe.plane_len()];
        Ok(Self {
            quad: quad_phase(&g),
            obs: Observed::from_counts(&scan.counts),
            fft: Fft2::new(g.frame),
            addr,
            object,
            probe,
            psi,
            features: Features::default(),
            cycle: 0,
            history: Vec::new(),
        })
    }

    pub fn features(&self) -> &Features {
        &self.features
    }

    /// Models and refines `features` from here on; an error leaves the current ones.
    pub fn set_features(&mut self, features: Features) -> Result<(), EngineError> {
        let [nframe, ny, nx] = self.obs.dims();
        features.check(nframe, [ny, nx])?;
        self.features = features;
        Ok(())
    }

    pub fn object(&self) -> &Object {
        &self.object
    }

    pub fn probe(&self) -> &Probe {
        &self.probe
    }

    pub fn into_modes(self) -> (Object, Probe) {
        (self.object, self.probe)
    }

    pub fn addr(&self) -> &AddrTable {
        &self.addr
    }

    pub fn observed(&self) -> &Observed {
        &self.obs
    }

    /// The far-field quadratic phase [`quad_phase`] of the scan's geometry, which every exit
    /// wave carries.
    pub fn quad(&self) -> &[C64] {
        &self.quad
    }

    /// LLK records in the order they were taken.
    pub fn history(&self) -> &[LlkRecord] {
        &self.history
    }

    /// Cycles (AP, DM) and epochs (rPIE) run so far.
    pub fn cycle(&self) -> usize {
        self.cycle
    }

    /// The probe moved by [`Features::position`] for every view, if one is set.
    fn shifted(&self) -> Option<Shifted> {
        let views = self.addr.views();
        (self.features.position.as_deref())
            .map(|d| Shifted::new(views, &self.probe, &self.quad, d, &self.fft))
    }

    /// Exit waves `P·O·q` of every view, `P·q` moved by [`Features::position`], into `psi`.
    fn to_psi(&self, shifted: Option<&Shifted>, psi: &mut [C64]) {
        let views = self.addr.views();
        match shifted {
            Some(s) => s.to_psi(views, &self.object, psi),
            None => obj_probe_to_psi(views, &self.object, &self.probe, &self.quad, psi),
        }
    }

    /// [`Ptycho::to_psi`] of the current object and probe, in a new buffer.
    fn calc_psi(&self) -> Vec<C64> {
        let mut psi = vec![C64::default(); self.psi.len()];
        self.to_psi(self.shifted().as_ref(), &mut psi);
        psi
    }

    /// [`psi_to_obj_probe`] of the exit waves `self.psi`, `P·q` moved by
    /// [`Features::position`].
    fn update_from_psi(&mut self, obj_inertia: Option<f64>, probe_inertia: Option<f64>) {
        let views = self.addr.views();
        match self.shifted() {
            Some(s) => s.psi_to_obj_probe(
                views,
                &self.psi,
                &mut self.object,
                &mut self.probe,
                [obj_inertia, probe_inertia],
                &self.fft,
            ),
            None => psi_to_obj_probe(
                views,
                &self.quad,
                &self.psi,
                &mut self.object,
                &mut self.probe,
                obj_inertia,
                probe_inertia,
            ),
        }
    }

    /// The numerators of the object's and the probe's back-projections of the object-plane
    /// `psi`, [`accumulate_obj`] and [`accumulate_probe`] with `P·q` moved by
    /// [`Features::position`]; an array not asked for gets zeros.
    fn back_project(&self, psi: &[C64], object: bool, probe: bool) -> [Stack<C64>; 2] {
        let views = self.addr.views();
        let (od, pd) = (self.object.dims(), self.probe.dims());
        let shifted = self.shifted();
        [
            match (object, &shifted) {
                (false, _) => Stack::filled(od, C64::default()),
                (true, Some(s)) => s.accumulate_obj(views, od, psi).0,
                (true, None) => accumulate_obj(views, od, &self.probe, &self.quad, psi).0,
            },
            match (probe, &shifted) {
                (false, _) => Stack::filled(pd, C64::default()),
                (true, Some(s)) => {
                    s.accumulate_probe(views, &self.object, pd, psi, &self.fft)
                        .0
                }
                (true, None) => accumulate_probe(views, &self.object, pd, &self.quad, psi).0,
            },
        ]
    }

    /// Normalised LLK of the current object and probe (PyNX `LLK() * FT() * ObjProbe2Psi()`
    /// then `get_llk(norm=True)`); the exit-wave stack is left as it is.
    pub fn llk(&self) -> Llk {
        let mut psi = self.calc_psi();
        self.fft.forward(&mut psi);
        let (nframe, nmode) = (self.addr.nframe(), self.addr.nmode());
        llk_with(
            &self.obs,
            0..nframe,
            nmode,
            &psi,
            Model::of(&self.features, &self.obs),
        )
        .normalised(self.obs.nb_obs())
    }

    /// [`Features::step`] from the far field of the current object and probe, if some
    /// feature takes one.
    fn step_features(&mut self) {
        if !self.features.stepped() {
            return;
        }
        let mut psi = self.calc_psi();
        self.fft.forward(&mut psi);
        self.features.step(&self.obs, self.addr.nmode(), &psi);
    }

    /// Gradient of the Poisson LLK sum ([`crate::engine::llk`], not normalised) of the current object
    /// and probe over the object's real and imaginary parts, packed as
    /// `∂L/∂Re O + i·∂L/∂Im O = 2·∂L/∂O*`: [`crate::engine::poisson_grad`] of the far field, inverse
    /// FFT, then [`accumulate_obj`]'s numerator (`P·q` moved by [`Features::position`]). PyNX
    /// `ML`'s `obj_grad` is half of it.
    pub fn obj_grad(&self) -> Stack<C64> {
        let [g, _] = self.grads(true, false);
        g
    }

    /// As [`Ptycho::obj_grad`] over the probe: `2·∂L/∂P*`, [`accumulate_probe`]'s numerator.
    /// PyNX `ML`'s `probe_grad` is half of it.
    pub fn probe_grad(&self) -> Stack<C64> {
        let [_, g] = self.grads(false, true);
        g
    }

    /// [`Ptycho::obj_grad`] and [`Ptycho::probe_grad`] from one far field; an array not asked
    /// for gets zeros.
    fn grads(&self, object: bool, probe: bool) -> [Stack<C64>; 2] {
        self.back_project(&self.grad_psi(), object, probe)
    }

    /// [`crate::engine::cg_poisson_gamma`] of the current object and probe along `(d_obj, d_probe)`,
    /// under the features' model.
    pub fn cg_gamma(&self, d_obj: &Stack<C64>, d_probe: &Stack<C64>) -> (f64, f64) {
        let (views, quad) = (self.addr.views(), &self.quad[..]);
        let (d_obj, d_probe) = (Object::new(d_obj.clone()), Probe::new(d_probe.clone()));
        let mut po = self.calc_psi();
        let mut a = vec![C64::default(); po.len()];
        let mut b = a.clone();
        match self.features.position.as_deref() {
            Some(d) => {
                let probe = Shifted::new(views, &self.probe, quad, d, &self.fft);
                let d_probe = Shifted::new(views, &d_probe, quad, d, &self.fft);
                probe.to_psi(views, &d_obj, &mut a);
                d_probe.to_psi(views, &self.object, &mut b);
                a.iter_mut().zip(&b).for_each(|(a, b)| *a += b);
                d_probe.to_psi(views, &d_obj, &mut b);
            }
            None => {
                obj_probe_to_psi(views, &d_obj, &self.probe, quad, &mut a);
                obj_probe_to_psi(views, &self.object, &d_probe, quad, &mut b);
                a.iter_mut().zip(&b).for_each(|(a, b)| *a += b);
                obj_probe_to_psi(views, &d_obj, &d_probe, quad, &mut b);
            }
        }
        for x in [&mut po, &mut a, &mut b] {
            self.fft.forward(x);
        }
        let (nframe, nmode) = (self.addr.nframe(), self.addr.nmode());
        let model = Model::of(&self.features, &self.obs);
        cg_poisson_gamma_with(&self.obs, 0..nframe, nmode, [&po, &a, &b], model)
    }

    /// The object-plane `2·∂L/∂psi*` both gradients back-project.
    fn grad_psi(&self) -> Vec<C64> {
        let mut psi = self.calc_psi();
        self.fft.forward(&mut psi);
        let (nframe, nmode) = (self.addr.nframe(), self.addr.nmode());
        poisson_grad_with(
            &self.obs,
            0..nframe,
            nmode,
            &mut psi,
            Model::of(&self.features, &self.obs),
        );
        self.fft.inverse(&mut psi);
        psi
    }

    fn record(&mut self, algorithm: Algorithm, llk: Llk) {
        self.history.push(LlkRecord {
            cycle: self.cycle,
            algorithm,
            llk,
        });
    }

    /// PyNX `ScaleObjProbe(absolute=True)`: scale so the calculated intensity over the
    /// unmasked pixels of every frame ([`Llk::nb_photons_calc`]) equals the observed one
    /// there, then balance `Σ|O|` and `Σ|P|`: `s = sqrt((Σ iobs − B) / (Σ icalc − B))` with
    /// `B` the photons of [`Features::background`] there (PyNX takes `iobs − background`),
    /// `P *= sqrt(Σ|O| / Σ|P| · s)`, `O *= sqrt(Σ|P| / Σ|O| · s)`. Masked pixels are in
    /// neither sum, as in PyNX CUDA's `scale_intensity`.
    pub fn scale_obj_probe(&mut self) {
        let b = self.features.background_photons(&self.obs);
        let calc = self.llk().nb_photons_calc - b;
        self.split_scale(((self.obs.total() - b) / calc).sqrt());
    }

    /// Scales `O P` by `s` and balances `Σ|O|` and `Σ|P|`: `P *= sqrt(Σ|O| / Σ|P| · s)`,
    /// `O *= sqrt(Σ|P| / Σ|O| · s)`. With `s = 1` it is PyNX `ScaleObjProbe(absolute=False)`.
    fn split_scale(&mut self, s: f64) {
        let os: f64 = self.object.as_slice().iter().map(|c| c.norm()).sum();
        let ps: f64 = self.probe.as_slice().iter().map(|c| c.norm()).sum();
        let (fp, fo) = ((os / ps * s).sqrt(), (ps / os * s).sqrt());
        self.probe.as_mut_slice().iter_mut().for_each(|c| *c *= fp);
        self.object.as_mut_slice().iter_mut().for_each(|c| *c *= fo);
    }

    /// PyNX `ZeroPhaseRamp`: centre the probe's far field. With `(cy, cx)` the centre of mass
    /// of `Σ_modes |fftshift(FT(fftshift(P)))|²`, `dx = cx − nx/2`, `dy = cy − ny/2`:
    /// `P *= exp(−2πi(dx·(x − nx/2)/nx + dy·(y − ny/2)/ny))` and
    /// `O *= exp(+2πi(dx·(x − nxo/2)/nx + dy·(y − nyo/2)/ny))` (integer halves). The two
    /// ramps cancel in every view up to a constant phase, so the LLK is unchanged, except that
    /// [`Features::position`] moves a ramp of non-integer `dx` or `dy` as a periodic one.
    pub fn zero_phase_ramp(&mut self) {
        zero_phase_ramp(&self.fft, &mut self.object, &mut self.probe);
    }

    /// PyNX `AP(...)**ncycle`: per cycle, exit waves from the current estimate, Fourier
    /// magnitude projection (the recorded LLK is of the estimate before the update), then
    /// [`psi_to_obj_probe`] with both updates from the old arrays, then the step of
    /// [`Features::position`].
    pub fn ap(&mut self, params: &ApParams, ncycle: usize) {
        let (nframe, nmode, nb_obs) = (self.addr.nframe(), self.addr.nmode(), self.obs.nb_obs());
        for ic in 0..ncycle {
            let shifted = self.shifted();
            let mut psi = std::mem::take(&mut self.psi);
            self.to_psi(shifted.as_ref(), &mut psi);
            self.psi = psi;
            self.fft.forward(&mut self.psi);
            let l = llk_due(params.calc_llk, ic, ncycle).then(|| {
                let model = Model::of(&self.features, &self.obs);
                llk_with(&self.obs, 0..nframe, nmode, &self.psi, model).normalised(nb_obs)
            });
            self.features.step(&self.obs, nmode, &self.psi);
            let model = Model::of(&self.features, &self.obs);
            apply_amplitude_with(&self.obs, 0..nframe, nmode, &mut self.psi, model);
            self.fft.inverse(&mut self.psi);
            let sums = (shifted.as_ref())
                .map(|s| s.position_sums(self.addr.views(), &self.object, &self.psi, &self.fft));
            self.update_from_psi(params.obj_inertia, params.probe_inertia);
            if let (Some(d), Some(sums)) = (&mut self.features.position, sums) {
                position_step(d, &sums);
            }
            if let Some(l) = l {
                self.record(Algorithm::Ap, l);
            }
            self.cycle += 1;
        }
        if params.zero_phase_ramp {
            self.zero_phase_ramp();
        }
    }

    /// PyNX `DM(...)**ncycle` with `a = alpha`. Cycle 0 starts from `psi = PO` (the current
    /// estimate's exit waves); later cycles reflect the carried `psi`:
    /// `psi ← (2 − a)·PO − (1 − a)·psi`, then magnitude projection `Π_F`, then
    /// `psi ← Π_F(psi) + (1 − a)·(PO − refl)` with `refl` the reflected `psi`. Then
    /// `loop_obj_probe` times an object-only and a probe-only [`psi_to_obj_probe`] (the probe
    /// update sees the new object), then the step of [`Features::position`]. The recorded LLK
    /// is of the estimate the cycle starts from, as PyNX's CUDA DM takes it; its `cpu_operator`
    /// DM takes the updated one. After the update of each cycle that records one, PyNX's CUDA
    /// DM balances `Σ|O|` and `Σ|P|` with `O P` unchanged (`ScaleObjProbe(absolute=False)`),
    /// and so does this.
    pub fn dm(&mut self, params: &DmParams, ncycle: usize) {
        let (nframe, nmode) = (self.addr.nframe(), self.addr.nmode());
        let a = params.alpha;
        let mut po = vec![C64::default(); self.psi.len()];
        let mut refl = vec![C64::default(); self.psi.len()];
        for ic in 0..ncycle {
            let l = llk_due(params.calc_llk, ic, ncycle).then(|| self.llk());
            self.step_features();
            let shifted = self.shifted();
            self.to_psi(shifted.as_ref(), &mut po);
            if ic == 0 {
                self.psi.copy_from_slice(&po);
            } else {
                for ((s, r), p) in self.psi.iter_mut().zip(&mut refl).zip(&po) {
                    *s = p * (2.0 - a) - *s * (1.0 - a);
                    *r = *s;
                }
            }
            self.fft.forward(&mut self.psi);
            let model = Model::of(&self.features, &self.obs);
            apply_amplitude_with(&self.obs, 0..nframe, nmode, &mut self.psi, model);
            self.fft.inverse(&mut self.psi);
            if ic > 0 {
                for ((s, r), p) in self.psi.iter_mut().zip(&refl).zip(&po) {
                    *s += (p - r) * (1.0 - a);
                }
            }
            let sums = (shifted.as_ref())
                .map(|s| s.position_sums(self.addr.views(), &self.object, &self.psi, &self.fft));
            for _ in 0..params.loop_obj_probe {
                for (o, p) in [(params.obj_inertia, None), (None, params.probe_inertia)] {
                    if o.is_some() || p.is_some() {
                        self.update_from_psi(o, p);
                    }
                }
            }
            if let (Some(d), Some(sums)) = (&mut self.features.position, sums) {
                position_step(d, &sums);
            }
            if let Some(l) = l {
                self.record(Algorithm::Dm, l);
                self.split_scale(1.0);
            }
            self.cycle += 1;
        }
        if params.zero_phase_ramp {
            self.zero_phase_ramp();
        }
    }

    /// PyNX `ML(...)**ncycle`, Poisson conjugate gradient, after [`Ptycho::scale_obj_probe`]
    /// (PyNX `ScaleObjProbe(absolute=True)`). Per cycle, with `g` the gradients
    /// ([`Ptycho::obj_grad`], [`Ptycho::probe_grad`]) of the arrays it updates (the recorded
    /// LLK is of the estimate before the update): the direction `d = −g` on the first cycle and
    /// `d = β·d − g` after, Polak–Ribière `β = Σ Re(conj(g)·(g − g_last)) / Σ|g_last|²` over
    /// both arrays (0 if negative or not finite); then `γ = num / den` of
    /// [`Ptycho::cg_gamma`] and `O += γ·d_O`, `P += γ·d_P`, or, for a `γ` that is not
    /// finite, no update and `d = −g` (PyNX `cu_linear_comb_fcfc_gamma`). PyNX CUDA keeps
    /// `−g / 2`, which changes neither `β` nor the step, and Hann-filters it in the far field;
    /// this does not filter. As in PyNX, a negative curvature `den` is not guarded: far from
    /// the solution (a flat object) the step can climb, so ML follows DM or AP. ML takes no
    /// position step, so it refuses [`Features::position`].
    pub fn ml(&mut self, params: &MlParams, ncycle: usize) -> Result<(), EngineError> {
        if self.features.position.is_some() {
            return Err(EngineError::Invalid("ML refines no positions".into()));
        }
        let (uo, up) = (params.update_object, params.update_probe);
        self.scale_obj_probe();
        let mut last: Option<[Stack<C64>; 2]> = None;
        let mut dir: [Stack<C64>; 2] = [
            Stack::filled(self.object.dims(), C64::default()),
            Stack::filled(self.probe.dims(), C64::default()),
        ];
        for ic in 0..ncycle {
            let l = llk_due(params.calc_llk, ic, ncycle).then(|| self.llk());
            self.step_features();
            let g = self.grads(uo, up);
            let beta = match &last {
                None => 0.0,
                Some(last) => {
                    let (mut num, mut den) = (0.0, 0.0);
                    for (g, h) in g.iter().zip(last) {
                        for (g, h) in g.as_slice().iter().zip(h.as_slice()) {
                            num += (g.conj() * (g - h)).re;
                            den += h.norm_sqr();
                        }
                    }
                    let b = num / den;
                    if b.is_finite() && b >= 0.0 { b } else { 0.0 }
                }
            };
            for (d, g) in dir.iter_mut().zip(&g) {
                for (d, g) in d.as_mut_slice().iter_mut().zip(g.as_slice()) {
                    *d = *d * beta - g;
                }
            }
            let (num, den) = self.cg_gamma(&dir[0], &dir[1]);
            // PyNX's γ has a wrong cross term in `den` (see `cg_poisson_gamma`); ptyrs follows
            // the math on both backends, so a γ gap to PyNX is not to be fixed toward it.
            let gamma = num / den;
            if gamma.is_finite() {
                let [d_obj, d_probe] = &dir;
                for (x, d) in [
                    (self.object.as_mut_slice(), d_obj),
                    (self.probe.as_mut_slice(), d_probe),
                ] {
                    x.iter_mut()
                        .zip(d.as_slice())
                        .for_each(|(x, d)| *x += d * gamma);
                }
            } else {
                for (d, g) in dir.iter_mut().zip(&g) {
                    d.as_mut_slice()
                        .iter_mut()
                        .zip(g.as_slice())
                        .for_each(|(d, g)| *d = -g);
                }
            }
            last = Some(g);
            if let Some(l) = l {
                self.record(Algorithm::Ml, l);
            }
            self.cycle += 1;
        }
        if params.zero_phase_ramp {
            self.zero_phase_ramp();
        }
        Ok(())
    }

    /// rPIE after pty-chi `RPIEReconstructor` with SGD, on consecutive batches of
    /// `batch_size` frames (pty-chi shuffles; this order is fixed). Needs one object mode.
    /// Per batch, from the batch-start object and probe, with `Δ = IFT(ψ') − ψ` and
    /// `ψ' = FT(ψ) / (sqrt(Σ_modes |FT ψ|²) + 1e-7) · sqrt(iobs + 1e-7)` (masked pixels keep
    /// `FT ψ`, so `Δ` is 0 there):
    /// - object: `O[win] += obj_step · Σ_modes conj(P_m q) Δ_m / ((1 − α_o)|P_m|² + α_o·max Σ_m|P_m|²)`,
    ///   summed over the batch;
    /// - probe: `P_m += probe_step · mean_batch conj(O_win q) Δ_m / ((1 − α_p)|O_win|² + α_p·max|O_win|²)`.
    ///
    /// pty-chi has no `q` and an unnormalised FFT: its probe `P·q / sqrt(ny·nx)` gives the same
    /// far-field moduli, so the `1e-7` guards act alike and the iterates correspond exactly
    /// (`scripts/compare_ptychi_rpie.py`). One LLK record (of the estimate after the epoch)
    /// every `calc_llk` epochs and after the last.
    pub fn rpie(&mut self, params: &RpieParams, nepoch: usize) -> Result<(), EngineError> {
        if self.addr.nobj() != 1 {
            return Err(EngineError::Invalid(
                "rPIE needs exactly one object mode".into(),
            ));
        }
        if params.batch_size == 0 {
            return Err(EngineError::Invalid("rPIE batch_size must be >= 1".into()));
        }
        if self.features != Features::default() {
            return Err(EngineError::Invalid("rPIE models no features".into()));
        }
        let nframe = self.addr.nframe();
        for ic in 0..nepoch {
            let mut b0 = 0;
            while b0 < nframe {
                let b1 = (b0 + params.batch_size).min(nframe);
                self.rpie_batch(params, b0..b1);
                b0 = b1;
            }
            if llk_due(params.calc_llk, ic, nepoch) {
                let l = self.llk();
                self.record(Algorithm::Rpie, l);
            }
            self.cycle += 1;
        }
        Ok(())
    }

    fn rpie_batch(&mut self, params: &RpieParams, frames: Range<usize>) {
        let [ny, nx] = self.probe.plane_shape();
        let n = ny * nx;
        let nmode = self.addr.nmode();
        let nxo = self.object.plane_shape()[1];
        let nbatch = frames.len();
        let views = &self.addr.views()[frames.start * nmode..frames.end * nmode];
        let mut psi = vec![C64::default(); views.len() * n];
        obj_probe_to_psi(views, &self.object, &self.probe, &self.quad, &mut psi);
        let mut delta = psi.clone();
        self.fft.forward(&mut delta);
        for (o, block) in self
            .obs
            .frames(frames)
            .chunks_exact(n)
            .zip(delta.chunks_exact_mut(nmode * n))
        {
            for (j, iobs) in o.iter().enumerate() {
                let Some(iobs) = *iobs else { continue };
                let calc = (0..nmode)
                    .map(|m| block[m * n + j].norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let r = (iobs + 1e-7).sqrt() / (calc + 1e-7);
                for m in 0..nmode {
                    block[m * n + j] *= r;
                }
            }
        }
        self.fft.inverse(&mut delta);
        delta.iter_mut().zip(&psi).for_each(|(d, p)| *d -= p);

        let mut d_obj = params
            .obj_step
            .map(|_| Stack::filled(self.object.dims(), C64::default()));
        if let Some(d_obj) = &mut d_obj {
            let pmax = (0..n)
                .map(|i| {
                    (0..nmode)
                        .map(|m| self.probe.plane(m)[i].norm_sqr())
                        .sum::<f64>()
                })
                .fold(0.0, f64::max);
            let a = params.obj_alpha;
            let d = d_obj.plane_mut(0);
            for (v, dv) in views.iter().zip(delta.chunks_exact(n)) {
                let p = self.probe.plane(v.probe as usize);
                let (oy, ox) = (v.oy as usize, v.ox as usize);
                for (j, dv) in dv.iter().enumerate() {
                    let (i, y, x) = natural(j, ny, nx);
                    let w = (p[i] * self.quad[i]).conj() / ((1.0 - a) * p[i].norm_sqr() + a * pmax);
                    d[(oy + y) * nxo + ox + x] += w * dv;
                }
            }
        }
        let mut d_probe = params
            .probe_step
            .map(|_| Stack::filled(self.probe.dims(), C64::default()));
        if let Some(d_probe) = &mut d_probe {
            let a = params.probe_alpha;
            let ob = self.object.plane(0);
            for (fv, dv) in views.chunks_exact(nmode).zip(delta.chunks_exact(nmode * n)) {
                let (oy, ox) = (fv[0].oy as usize, fv[0].ox as usize);
                let win = |i: usize| ob[(oy + i / nx) * nxo + ox + i % nx];
                let omax = (0..n).map(|i| win(i).norm_sqr()).fold(0.0, f64::max);
                for (v, dv) in fv.iter().zip(dv.chunks_exact(n)) {
                    let d = d_probe.plane_mut(v.probe as usize);
                    for (j, dv) in dv.iter().enumerate() {
                        let i = shift(j, ny, nx);
                        let o = win(i);
                        let w = (o * self.quad[i]).conj() / ((1.0 - a) * o.norm_sqr() + a * omax);
                        d[i] += w * dv;
                    }
                }
            }
        }
        if let (Some(step), Some(d)) = (params.obj_step, d_obj) {
            for (o, d) in self.object.as_mut_slice().iter_mut().zip(d.as_slice()) {
                *o += d * step;
            }
        }
        if let (Some(step), Some(d)) = (params.probe_step, d_probe) {
            let s = step / nbatch as f64;
            for (p, d) in self.probe.as_mut_slice().iter_mut().zip(d.as_slice()) {
                *p += d * s;
            }
        }
    }
}

/// [`Ptycho::zero_phase_ramp`] of the given arrays, which the GPU backend also runs on its
/// downloads.
pub(crate) fn zero_phase_ramp(fft: &Fft2, object: &mut Object, probe: &mut Probe) {
    let [ny, nx] = probe.plane_shape();
    let n = ny * nx;
    let mut far = vec![C64::default(); probe.as_slice().len()];
    for (m, out) in far.chunks_exact_mut(n).enumerate() {
        let p = probe.plane(m);
        for (j, out) in out.iter_mut().enumerate() {
            *out = p[shift(j, ny, nx)];
        }
    }
    fft.forward(&mut far);
    let (mut w, mut wy, mut wx) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let j = shift(i, ny, nx);
        let v: f64 = far.chunks_exact(n).map(|f| f[j].norm_sqr()).sum();
        w += v;
        wy += v * (i / nx) as f64;
        wx += v * (i % nx) as f64;
    }
    let dy = wy / w - (ny / 2) as f64;
    let dx = wx / w - (nx / 2) as f64;
    let ramp = |len: usize, half: usize, scale: usize, d: f64, sign: f64| -> Vec<f64> {
        (0..len)
            .map(|i| sign * 2.0 * PI * d * (i as f64 - half as f64) / scale as f64)
            .collect()
    };
    let (ry, rx) = (
        ramp(ny, ny / 2, ny, dy, -1.0),
        ramp(nx, nx / 2, nx, dx, -1.0),
    );
    apply_ramp(probe, &ry, &rx);
    let [nyo, nxo] = object.plane_shape();
    let (ry, rx) = (
        ramp(nyo, nyo / 2, ny, dy, 1.0),
        ramp(nxo, nxo / 2, nx, dx, 1.0),
    );
    apply_ramp(object, &ry, &rx);
}

/// `a[k, y, x] *= exp(i·(ry[y] + rx[x]))` for every mode `k`.
fn apply_ramp(a: &mut Stack<C64>, ry: &[f64], rx: &[f64]) {
    let nx = rx.len();
    for c in a.as_mut_slice().chunks_exact_mut(ry.len() * nx) {
        for (i, c) in c.iter_mut().enumerate() {
            *c *= C64::from_polar(1.0, ry[i / nx] + rx[i % nx]);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::base::{Geometry, required_obj_shape};

    use super::*;
    use crate::engine::llk;

    fn poisson_sum(p: &Ptycho) -> f64 {
        let mut psi = p.calc_psi();
        p.fft.forward(&mut psi);
        llk(&p.obs, 0..p.addr.nframe(), p.addr.nmode(), &psi).poisson
    }

    /// Two object and two probe modes, masked detector pixels and pixels with zero counts, the
    /// object away from the truth, scaled.
    fn grad_start() -> Ptycho {
        let g = Geometry::new(1.5e-10, 1.0, 75e-6, [16, 16]).unwrap();
        let pixel = g.pixel_size_object();
        let positions = crate::base::sim::fermat(5, 3.0 * pixel[1]);
        let shape = required_obj_shape(&positions.pynx_preprocess(pixel), pixel, g.frame, 0);
        let one = crate::base::sim::object(shape);
        let second = one.as_slice().iter().map(|v| v.conj() * 0.3);
        let truth: Vec<C64> = one.as_slice().iter().copied().chain(second).collect();
        let truth = Object::new(Stack::from_vec([2, shape[0], shape[1]], truth).unwrap());
        let probe = crate::base::sim::probe(g.frame, 2).unwrap();
        let mut mask = vec![false; 256];
        mask[8 * 16 + 9] = true;
        mask[3 * 16 + 5] = true;
        let scan = crate::engine::sim::simulate(g, positions, &truth, &probe, 2e3, Some(&mask), 3)
            .unwrap();
        let start = truth
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, v)| v * C64::from_polar(0.9, 0.4 * (0.37 * i as f64).sin()))
            .collect();
        let start = Object::new(Stack::from_vec(truth.dims(), start).unwrap());
        let mut p = Ptycho::new(&scan, start, probe).unwrap();
        p.scale_obj_probe();
        let obs = p.obs.frames(0..p.addr.nframe());
        assert!(obs.contains(&None) && obs.contains(&Some(0.0)));
        p
    }

    /// [`grad_start`] and [`grad_start`] with its probe moved by sub-pixel offsets of both signs.
    fn grad_starts() -> [Ptycho; 2] {
        let mut moved = grad_start();
        let d = (0..moved.addr.nframe())
            .map(|f| {
                [
                    0.6 * (1.3 * f as f64 + 0.2).sin(),
                    -0.45 * (0.7 * f as f64).cos(),
                ]
            })
            .collect();
        moved
            .set_features(Features {
                position: Some(d),
                ..Features::default()
            })
            .unwrap();
        [grad_start(), moved]
    }

    /// `grad` against central differences of the Poisson LLK sum in the real and imaginary
    /// part of every element of the array `arr` picks, steps 1e-6 of its rms.
    fn assert_central(p: &mut Ptycho, grad: &Stack<C64>, arr: fn(&mut Ptycho) -> &mut [C64]) {
        let len = arr(p).len();
        assert_eq!(grad.as_slice().len(), len);
        let gmax = grad.as_slice().iter().map(|v| v.norm()).fold(0.0, f64::max);
        let rms = (arr(p).iter().map(|v| v.norm_sqr()).sum::<f64>() / len as f64).sqrt();
        let h = 1e-6 * rms;
        let mut err = 0.0f64;
        for k in 0..len {
            let o = arr(p)[k];
            for (dir, want) in [
                (C64::new(h, 0.0), grad.as_slice()[k].re),
                (C64::new(0.0, h), grad.as_slice()[k].im),
            ] {
                arr(p)[k] = o + dir;
                let lp = poisson_sum(p);
                arr(p)[k] = o - dir;
                let lm = poisson_sum(p);
                arr(p)[k] = o;
                err = err.max(((lp - lm) / (2.0 * h) - want).abs());
            }
        }
        assert!(
            err < 1e-6 * gmax,
            "max |fd - grad| {err:e}, max |grad| {gmax:e}"
        );
    }

    #[test]
    fn obj_grad_matches_central_differences() {
        for mut p in grad_starts() {
            let grad = p.obj_grad();
            assert_central(&mut p, &grad, |p| p.object.as_mut_slice());
        }
    }

    #[test]
    fn probe_grad_matches_central_differences() {
        for mut p in grad_starts() {
            let grad = p.probe_grad();
            assert_central(&mut p, &grad, |p| p.probe.as_mut_slice());
        }
    }

    /// `cg_gamma` along a direction that is not the gradient's: its numerator is `−Re⟨g, d⟩`,
    /// and `num / den` is the Newton step `−φ′(0) / φ″(0)` of `φ(t)`, the Poisson sum at
    /// `O + t·d_O`, `P + t·d_P`, by central differences at 1e-4 of the step.
    #[test]
    fn cg_gamma_is_the_newton_step_of_the_llk() {
        for p in grad_starts() {
            cg_gamma_newton(p);
        }
    }

    fn cg_gamma_newton(mut p: Ptycho) {
        let g = p.grads(true, true);
        let d = g.clone().map(|mut d| {
            let w = d.as_mut_slice();
            w.iter_mut().enumerate().for_each(|(k, v)| {
                *v *= -C64::from_polar(
                    1.0 + 0.5 * (0.7 * k as f64).sin(),
                    0.3 * (1.3 * k as f64).cos(),
                );
            });
            d
        });
        let (num, den) = p.cg_gamma(&d[0], &d[1]);
        let dot: f64 = g
            .iter()
            .zip(&d)
            .flat_map(|(g, d)| g.as_slice().iter().zip(d.as_slice()))
            .map(|(g, d)| (g.conj() * d).re)
            .sum();
        assert!(
            (num + dot).abs() < 1e-12 * dot.abs(),
            "num {num} −Re⟨g, d⟩ {}",
            -dot
        );
        let gamma = num / den;
        let h = 1e-4 * gamma;
        let (o0, p0) = (p.object.clone(), p.probe.clone());
        let mut phi = |t: f64| {
            for (x, (x0, d)) in [
                (p.object.as_mut_slice(), (o0.as_slice(), &d[0])),
                (p.probe.as_mut_slice(), (p0.as_slice(), &d[1])),
            ] {
                for ((x, x0), d) in x.iter_mut().zip(x0).zip(d.as_slice()) {
                    *x = x0 + d * t;
                }
            }
            poisson_sum(&p)
        };
        let (lm, l0, lp) = (phi(-h), phi(0.0), phi(h));
        let newton = -((lp - lm) / (2.0 * h)) / ((lp - 2.0 * l0 + lm) / (h * h));
        assert!(
            (gamma - newton).abs() < 1e-5 * gamma.abs(),
            "gamma {gamma} newton {newton}"
        );
    }
}
