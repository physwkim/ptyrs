//! pty-chi BH (Carlsson, Nikitin and Wendt's bilinear-Hessian method, `reconstructors/bh.py`)
//! on the amplitude cost, as CPU f64 operators over the arrays and layout of [`crate::ops`]
//! (DESIGN B7): [`BhBatch`] holds the cost's gradient and Hessian at one state over a batch of
//! views, [`Bh`] runs the epochs.
//!
//! Directions and displacements are pairs `[object, probe]` shaped as the object and probe,
//! with `⟨a, b⟩ = Σ Re(conj(a)·b)` over both arrays.

use ptyrs_core::{AddrTable, C64, Object, Probe, Stack, View};

use crate::ops::{accumulate_obj, accumulate_probe, natural, obj_probe_to_psi};
use crate::{Fft2, Observed};

/// pty-chi's offset on the modelled amplitude `A` in the residual and the Hessian.
const EPS: f64 = 1e-8;

/// BH parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BhParams {
    /// Frames per batch, consecutive in scan order, capped at the scan's frame count. One
    /// batch of every frame runs CG across epochs; smaller batches take steepest descent along
    /// each batch's gradient (DESIGN B7).
    pub batch_size: usize,
    /// pty-chi probe `rho`: the probe variable is the probe over `ρ = rho·√(ny·nx)`, so in
    /// probe displacements its gradient is scaled by `ρ²`.
    pub probe_rho: f64,
}

/// One batch's step, as pty-chi computes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BhStep {
    pub first: usize,
    pub count: usize,
    /// CG weight of the previous direction; 0 in steepest descent and the first epoch.
    pub beta: f64,
    /// Step along the direction, after the factor `count / batch_size`.
    pub alpha: f64,
    /// `−⟨g, D⟩` with `g` the gradient and `D` the direction.
    pub top: f64,
    /// `H(D, D)`.
    pub bottom: f64,
}

/// The amplitude cost of a batch of views at one object and probe and its derivatives:
/// `F = Σ_frames Σ_unmasked (A − √iobs)²` with `A = √(Σ_m |psi_m|²)` and `psi` the far field of
/// [`obj_probe_to_psi`]. `views` are whole frames of an [`AddrTable`] in its order.
pub struct BhBatch<'a> {
    views: &'a [View],
    quad: &'a [C64],
    object: &'a Object,
    probe: &'a Probe,
    fft: Fft2,
    nmode: usize,
    /// Far field, one shifted plane per view.
    psi: Vec<C64>,
    /// `A + EPS` and `√iobs` (`None` where masked) per frame and shifted pixel.
    amp: Vec<f64>,
    root: Vec<Option<f64>>,
    /// `F⁻¹(psi·(1 − √iobs / (A + EPS)))`, zero where masked: half the gradient of `F` over
    /// the exit waves, one shifted object-plane plane per view.
    r: Vec<C64>,
}

impl<'a> BhBatch<'a> {
    pub fn new(
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
        let mut psi = vec![C64::default(); views.len() * n];
        obj_probe_to_psi(views, object, probe, quad, &mut psi);
        fft.forward(&mut psi);
        let mut r = psi.clone();
        let (mut amp, mut root) = (Vec::new(), Vec::new());
        for (fv, block) in views.chunks_exact(nmode).zip(r.chunks_exact_mut(nmode * n)) {
            let f = fv[0].frame as usize;
            assert!(fv.iter().all(|v| v.frame == fv[0].frame));
            for (j, iobs) in obs.frames(f..f + 1).iter().enumerate() {
                let a = (0..nmode)
                    .map(|m| block[m * n + j].norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    + EPS;
                let d = iobs.map(f64::sqrt);
                let w = d.map_or(0.0, |d| 1.0 - d / a);
                (0..nmode).for_each(|m| block[m * n + j] *= w);
                amp.push(a);
                root.push(d);
            }
        }
        fft.inverse(&mut r);
        Self {
            views,
            quad,
            object,
            probe,
            fft,
            nmode,
            psi,
            amp,
            root,
            r,
        }
    }

    /// `[2·∂F/∂O*, 2·∂F/∂P*]` ([`accumulate_obj`] and [`accumulate_probe`] of `2r`), so
    /// `⟨g, u⟩` is the derivative of `F` along the displacement `u`.
    pub fn grads(&self) -> [Stack<C64>; 2] {
        let o = accumulate_obj(
            self.views,
            self.object.dims(),
            self.probe,
            self.quad,
            &self.r,
        );
        let p = accumulate_probe(
            self.views,
            self.object,
            self.probe.dims(),
            self.quad,
            &self.r,
        );
        [o.0, p.0].map(|mut g| {
            g.as_mut_slice().iter_mut().for_each(|c| *c *= 2.0);
            g
        })
    }

    /// The far field's derivative along the displacement `u`, one shifted plane per view:
    /// `F(q·(u_P·O_view + P·u_O,view))`.
    fn far_derivative(&self, u: [&Stack<C64>; 2]) -> Vec<C64> {
        let [ny, nx] = self.probe.plane_shape();
        let (n, nxo) = (ny * nx, self.object.plane_shape()[1]);
        let mut out = vec![C64::default(); self.views.len() * n];
        for (v, out) in self.views.iter().zip(out.chunks_exact_mut(n)) {
            let (p, up) = (
                self.probe.plane(v.probe as usize),
                u[1].plane(v.probe as usize),
            );
            let (o, uo) = (
                self.object.plane(v.obj as usize),
                u[0].plane(v.obj as usize),
            );
            for (j, out) in out.iter_mut().enumerate() {
                let (i, y, x) = natural(j, ny, nx);
                let w = (v.oy as usize + y) * nxo + v.ox as usize + x;
                *out = self.quad[i] * (up[i] * o[w] + p[i] * uo[w]);
            }
        }
        self.fft.forward(&mut out);
        out
    }

    /// The second derivative of `F` along the displacements `u` and `v`, pty-chi's bilinear
    /// Hessian with `A + EPS` in place of `A`:
    /// `Σ Re(conj(2r)·q·(u_P·v_O,view + v_P·u_O,view))
    ///  + 2 Σ_unmasked [(1 − d/A)·Σ_m Re(conj(L_u)·L_v) + (d/A)·c_u·c_v]`
    /// with `L_u` the far field's derivative along `u`, `c_u = Σ_m Re(conj(psi/A)·L_u)` and
    /// `d = √iobs`.
    pub fn hessian(&self, u: [&Stack<C64>; 2], v: [&Stack<C64>; 2]) -> f64 {
        let [ny, nx] = self.probe.plane_shape();
        let (n, nxo) = (ny * nx, self.object.plane_shape()[1]);
        let mut h = 0.0;
        for (view, r) in self.views.iter().zip(self.r.chunks_exact(n)) {
            let (up, vp) = (
                u[1].plane(view.probe as usize),
                v[1].plane(view.probe as usize),
            );
            let (uo, vo) = (u[0].plane(view.obj as usize), v[0].plane(view.obj as usize));
            for (j, r) in r.iter().enumerate() {
                let (i, y, x) = natural(j, ny, nx);
                let w = (view.oy as usize + y) * nxo + view.ox as usize + x;
                h += 2.0 * (r.conj() * self.quad[i] * (up[i] * vo[w] + vp[i] * uo[w])).re;
            }
        }
        let (lu, lv) = (self.far_derivative(u), self.far_derivative(v));
        let block = self.nmode * n;
        for (k, (&a, d)) in self.amp.iter().zip(&self.root).enumerate() {
            let Some(d) = *d else { continue };
            let (mut uv, mut cu, mut cv) = (0.0, 0.0, 0.0);
            for m in 0..self.nmode {
                let e = (k / n) * block + m * n + k % n;
                uv += (lu[e].conj() * lv[e]).re;
                cu += (self.psi[e].conj() * lu[e]).re;
                cv += (self.psi[e].conj() * lv[e]).re;
            }
            h += 2.0 * ((1.0 - d / a) * uv + d * cu * cv / (a * a * a));
        }
        h
    }
}

/// `⟨a, b⟩` of a pair.
fn dot(a: &[Stack<C64>; 2], b: &[Stack<C64>; 2]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.as_slice().iter().zip(b.as_slice()))
        .map(|(a, b)| (a.conj() * b).re)
        .sum()
}

fn refs(a: &[Stack<C64>; 2]) -> [&Stack<C64>; 2] {
    [&a[0], &a[1]]
}

/// BH state across epochs: the parameters and, with one batch per epoch, the last direction.
pub struct Bh {
    params: BhParams,
    prev: Option<[Stack<C64>; 2]>,
}

impl Bh {
    pub fn new(params: BhParams) -> Self {
        assert!(params.batch_size > 0, "batch_size must be positive");
        Self { params, prev: None }
    }

    /// One epoch over consecutive batches of `addr`'s frames in scan order, each moving
    /// `object` and `probe` by `α·D`. With `g` the batch's [`BhBatch::grads`] and
    /// `Pg = [g_O, ρ²·g_P]`: `D = −Pg`, or with one batch per epoch, after the first epoch,
    /// `D = −Pg + β·D′` with `β = H(Pg, D′) / H(D′, D′)` and `D′` the previous epoch's
    /// direction; `α = −⟨g, D⟩ / H(D, D) · count / batch_size`.
    pub fn epoch(
        &mut self,
        addr: &AddrTable,
        obs: &Observed,
        quad: &[C64],
        object: &mut Object,
        probe: &mut Probe,
    ) -> Vec<BhStep> {
        let (nframe, nmode) = (addr.nframe(), addr.nmode());
        let size = self.params.batch_size.min(nframe);
        let full = size == nframe;
        let rho2 = self.params.probe_rho.powi(2) * probe.plane_len() as f64;
        let mut steps = Vec::new();
        for first in (0..nframe).step_by(size) {
            let count = size.min(nframe - first);
            let views = &addr.views()[first * nmode..(first + count) * nmode];
            let b = BhBatch::new(views, obs, quad, object, probe);
            let g = b.grads();
            let mut pg = g.clone();
            pg[1].as_mut_slice().iter_mut().for_each(|c| *c *= rho2);
            let (beta, dir) = match self.prev.take().filter(|_| full) {
                Some(prev) => {
                    let beta =
                        b.hessian(refs(&pg), refs(&prev)) / b.hessian(refs(&prev), refs(&prev));
                    let mut dir = prev;
                    for (d, pg) in dir.iter_mut().zip(&pg) {
                        for (d, pg) in d.as_mut_slice().iter_mut().zip(pg.as_slice()) {
                            *d = beta * *d - pg;
                        }
                    }
                    (beta, dir)
                }
                None => {
                    pg.iter_mut()
                        .for_each(|a| a.as_mut_slice().iter_mut().for_each(|c| *c = -*c));
                    (0.0, pg)
                }
            };
            let top = -dot(&g, &dir);
            let bottom = b.hessian(refs(&dir), refs(&dir));
            let alpha = top / bottom * count as f64 / size as f64;
            for (x, d) in [object.as_mut_slice(), probe.as_mut_slice()]
                .into_iter()
                .zip(&dir)
            {
                x.iter_mut()
                    .zip(d.as_slice())
                    .for_each(|(x, d)| *x += alpha * d);
            }
            if full {
                self.prev = Some(dir);
            }
            steps.push(BhStep {
                first,
                count,
                beta,
                alpha,
                top,
                bottom,
            });
        }
        steps
    }
}
