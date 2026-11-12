//! [`Ptycho`]: the reconstruction state and the AP, DM and rPIE loops built on the operators.

use std::f64::consts::PI;
use std::ops::Range;

use ptyrs_core::{AddrTable, C64, Object, Probe, Scan, Stack};

use crate::ops::{
    Llk, apply_amplitude, llk, natural, obj_probe_to_psi, psi_to_obj_probe, quad_phase,
};
use crate::{EngineError, Fft2, Observed, shift};

/// Which loop recorded an [`LlkRecord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    Ap,
    Dm,
    Rpie,
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
fn llk_due(calc_llk: usize, ic: usize, ncycle: usize) -> bool {
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
/// Positions go through [`ptyrs_core::Positions::pynx_preprocess`] and become an
/// [`AddrTable`] once; the probe frame must match the detector frame. Object mode `m / nprobe`
/// and probe mode `m % nprobe` make view mode `m` (PyNX `_obj_psidx`, `_probe_psidx`).
pub struct Ptycho {
    addr: AddrTable,
    quad: Vec<C64>,
    obs: Observed,
    fft: Fft2,
    object: Object,
    probe: Probe,
    /// Object-plane exit waves, one shifted plane per view; DM carries it across cycles.
    psi: Vec<C64>,
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
            cycle: 0,
            history: Vec::new(),
        })
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

    /// LLK records in the order they were taken.
    pub fn history(&self) -> &[LlkRecord] {
        &self.history
    }

    /// Cycles (AP, DM) and epochs (rPIE) run so far.
    pub fn cycle(&self) -> usize {
        self.cycle
    }

    /// Exit waves `P·O·q` of every view, in a new buffer.
    fn calc_psi(&self) -> Vec<C64> {
        let mut psi = vec![C64::default(); self.psi.len()];
        obj_probe_to_psi(
            self.addr.views(),
            &self.object,
            &self.probe,
            &self.quad,
            &mut psi,
        );
        psi
    }

    /// Normalised LLK of the current object and probe (PyNX `LLK() * FT() * ObjProbe2Psi()`
    /// then `get_llk(norm=True)`); the exit-wave stack is left as it is.
    pub fn llk(&self) -> Llk {
        let mut psi = self.calc_psi();
        self.fft.forward(&mut psi);
        llk(&self.obs, 0..self.addr.nframe(), self.addr.nmode(), &psi).normalised(self.obs.nb_obs())
    }

    fn record(&mut self, algorithm: Algorithm, llk: Llk) {
        self.history.push(LlkRecord {
            cycle: self.cycle,
            algorithm,
            llk,
        });
    }

    /// PyNX `ScaleObjProbe(absolute=True)`: scale so `Σ|P·O·q|²` over all views equals the
    /// total unmasked observed intensity, then balance `Σ|O|` and `Σ|P|`:
    /// `s = sqrt(Σ iobs / Σ|psi|²)`, `P *= sqrt(Σ|O| / Σ|P| · s)`, `O *= sqrt(Σ|P| / Σ|O| · s)`.
    /// PyNX sums `iobs` including its negative masked values; here masked pixels are left out.
    pub fn scale_obj_probe(&mut self) {
        let calc: f64 = self.calc_psi().iter().map(|c| c.norm_sqr()).sum();
        let s = (self.obs.total() / calc).sqrt();
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
    /// ramps cancel in every view up to a constant phase, so the LLK is unchanged.
    pub fn zero_phase_ramp(&mut self) {
        let [ny, nx] = self.probe.plane_shape();
        let n = ny * nx;
        let mut far = vec![C64::default(); self.probe.as_slice().len()];
        for (m, out) in far.chunks_exact_mut(n).enumerate() {
            let p = self.probe.plane(m);
            for (j, out) in out.iter_mut().enumerate() {
                *out = p[shift(j, ny, nx)];
            }
        }
        self.fft.forward(&mut far);
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
        apply_ramp(&mut self.probe, &ry, &rx);
        let [nyo, nxo] = self.object.plane_shape();
        let (ry, rx) = (
            ramp(nyo, nyo / 2, ny, dy, 1.0),
            ramp(nxo, nxo / 2, nx, dx, 1.0),
        );
        apply_ramp(&mut self.object, &ry, &rx);
    }

    /// PyNX `AP(...)**ncycle`: per cycle, exit waves from the current estimate, Fourier
    /// magnitude projection (the recorded LLK is of the estimate before the update), then
    /// [`psi_to_obj_probe`] with both updates from the old arrays.
    pub fn ap(&mut self, params: &ApParams, ncycle: usize) {
        let (nframe, nmode, nb_obs) = (self.addr.nframe(), self.addr.nmode(), self.obs.nb_obs());
        for ic in 0..ncycle {
            obj_probe_to_psi(
                self.addr.views(),
                &self.object,
                &self.probe,
                &self.quad,
                &mut self.psi,
            );
            self.fft.forward(&mut self.psi);
            let l = llk_due(params.calc_llk, ic, ncycle)
                .then(|| llk(&self.obs, 0..nframe, nmode, &self.psi).normalised(nb_obs));
            apply_amplitude(&self.obs, 0..nframe, nmode, &mut self.psi);
            self.fft.inverse(&mut self.psi);
            psi_to_obj_probe(
                self.addr.views(),
                &self.quad,
                &self.psi,
                &mut self.object,
                &mut self.probe,
                params.obj_inertia,
                params.probe_inertia,
            );
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
    /// update sees the new object). The recorded LLK is of the updated estimate.
    pub fn dm(&mut self, params: &DmParams, ncycle: usize) {
        let (nframe, nmode) = (self.addr.nframe(), self.addr.nmode());
        let a = params.alpha;
        let mut po = vec![C64::default(); self.psi.len()];
        let mut refl = vec![C64::default(); self.psi.len()];
        for ic in 0..ncycle {
            obj_probe_to_psi(
                self.addr.views(),
                &self.object,
                &self.probe,
                &self.quad,
                &mut po,
            );
            if ic == 0 {
                self.psi.copy_from_slice(&po);
            } else {
                for ((s, r), p) in self.psi.iter_mut().zip(&mut refl).zip(&po) {
                    *s = p * (2.0 - a) - *s * (1.0 - a);
                    *r = *s;
                }
            }
            self.fft.forward(&mut self.psi);
            apply_amplitude(&self.obs, 0..nframe, nmode, &mut self.psi);
            self.fft.inverse(&mut self.psi);
            if ic > 0 {
                for ((s, r), p) in self.psi.iter_mut().zip(&refl).zip(&po) {
                    *s += (p - r) * (1.0 - a);
                }
            }
            for _ in 0..params.loop_obj_probe {
                for (o, p) in [(params.obj_inertia, None), (None, params.probe_inertia)] {
                    if o.is_some() || p.is_some() {
                        psi_to_obj_probe(
                            self.addr.views(),
                            &self.quad,
                            &self.psi,
                            &mut self.object,
                            &mut self.probe,
                            o,
                            p,
                        );
                    }
                }
            }
            if llk_due(params.calc_llk, ic, ncycle) {
                let l = self.llk();
                self.record(Algorithm::Dm, l);
            }
            self.cycle += 1;
        }
        if params.zero_phase_ramp {
            self.zero_phase_ramp();
        }
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
    /// The FFT is unitary where pty-chi's is unnormalised; the iterates agree with the probe
    /// scaled by `sqrt(ny·nx)` apart from the `1e-7` guards. One LLK record (of the estimate
    /// after the epoch) every `calc_llk` epochs and after the last.
    pub fn rpie(&mut self, params: &RpieParams, nepoch: usize) -> Result<(), EngineError> {
        if self.addr.nobj() != 1 {
            return Err(EngineError::Invalid(
                "rPIE needs exactly one object mode".into(),
            ));
        }
        if params.batch_size == 0 {
            return Err(EngineError::Invalid("rPIE batch_size must be >= 1".into()));
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

/// `a[k, y, x] *= exp(i·(ry[y] + rx[x]))` for every mode `k`.
fn apply_ramp(a: &mut Stack<C64>, ry: &[f64], rx: &[f64]) {
    let nx = rx.len();
    for c in a.as_mut_slice().chunks_exact_mut(ry.len() * nx) {
        for (i, c) in c.iter_mut().enumerate() {
            *c *= C64::from_polar(1.0, ry[i / nx] + rx[i % nx]);
        }
    }
}
