//! pty-chi BH (Carlsson, Nikitin and Wendt's bilinear-Hessian method, `reconstructors/bh.py`)
//! on the amplitude cost, as CPU f64 operators through an [`Optics`] (DESIGN B7): [`BhBatch`]
//! holds the cost's gradient and Hessian at one state over a batch of views, [`Bh`] runs the
//! epochs. Through several slices, which pty-chi's BH rejects, the object variable is every
//! slice of [`SliceWaves`].
//!
//! Directions and displacements are flat: every slice's object, then the probe (the domain of
//! [`SliceJacobian`]), with `⟨a, b⟩ = Σ Re(conj(a)·b)`.

use ptyrs_core::{AddrTable, C64, Object, Probe, View};

use crate::linop::FarJacobian;
use crate::{LinearOp, Observed, Optics, SliceJacobian, SliceWaves, random_batches};

/// pty-chi's offset on the modelled amplitude `A` in the residual and the Hessian.
const EPS: f64 = 1e-8;

/// BH parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BhParams {
    /// Frames per batch, capped at the scan's frame count. One batch of every frame runs CG
    /// across epochs; smaller batches take steepest descent along each batch's gradient
    /// (DESIGN B7).
    pub batch_size: usize,
    /// pty-chi probe `rho`: the probe variable is the probe over `ρ = rho·√k`, `k` the
    /// propagator's [`crate::Propagator::intensity_scale`] (`ny·nx` in the far field), so in
    /// probe displacements its gradient is scaled by `ρ²`.
    pub probe_rho: f64,
    /// The seed of every epoch's [`random_batches`].
    pub seed: u64,
}

/// One batch's step, as pty-chi computes it.
#[derive(Clone, Debug, PartialEq)]
pub struct BhStep {
    pub frames: Vec<usize>,
    /// CG weight of the previous direction; 0 in steepest descent and the first epoch.
    pub beta: f64,
    /// Step along the direction, after the factor `frames.len() / batch_size`.
    pub alpha: f64,
    /// `−⟨g, D⟩` with `g` the gradient and `D` the direction.
    pub top: f64,
    /// `H(D, D)`.
    pub bottom: f64,
}

/// The amplitude cost of a batch of views at one state of the slices and probe and its
/// derivatives: `F = Σ_frames Σ_unmasked (A − √iobs)²` with `A = √(Σ_m |U psi_m|²)`, `psi`
/// the exit waves of [`SliceWaves`] and `U` the [`Optics::propagator`]. `views` are whole
/// frames of an [`AddrTable`], each frame's views in its order.
pub struct BhBatch<'a> {
    optics: &'a Optics,
    waves: SliceWaves<'a>,
    obj_dims: [usize; 3],
    probe_dims: [usize; 3],
    nmode: usize,
    /// `U psi`, one shifted plane per view.
    psi: Vec<C64>,
    /// `A + EPS` and `√iobs` (`None` where masked) per frame and shifted pixel.
    amp: Vec<f64>,
    root: Vec<Option<f64>>,
    /// `U psi·(1 − √iobs / (A + EPS))`, zero where masked: half the gradient of `F` over
    /// `U psi`.
    td: Vec<C64>,
}

impl<'a> BhBatch<'a> {
    /// `objects` holds one object per slice of `optics`.
    pub fn new(
        views: &[View],
        obs: &Observed,
        optics: &'a Optics,
        objects: &'a [Object],
        probe: &Probe,
    ) -> Self {
        let n = probe.plane_len();
        let nmode = views
            .iter()
            .take_while(|v| v.frame == views[0].frame)
            .count();
        assert_eq!(views.len() % nmode, 0, "views must be whole frames");
        let waves = SliceWaves::new(optics, views, objects, probe);
        let mut psi = waves.psi().to_vec();
        optics.propagator().forward(&mut psi);
        let mut td = psi.clone();
        let (mut amp, mut root) = (Vec::new(), Vec::new());
        for (fv, block) in views
            .chunks_exact(nmode)
            .zip(td.chunks_exact_mut(nmode * n))
        {
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
        Self {
            optics,
            waves,
            obj_dims: objects[0].dims(),
            probe_dims: probe.dims(),
            nmode,
            psi,
            amp,
            root,
            td,
        }
    }

    /// `x ↦ U·J x` with `J` the [`SliceJacobian`] at this state, the path of every derivative.
    fn jacobian(&self) -> FarJacobian<'_> {
        FarJacobian {
            slices: SliceJacobian {
                waves: &self.waves,
                obj_dims: self.obj_dims,
                nslice: self.optics.nslice(),
                probe_dims: self.probe_dims,
            },
            propagator: self.optics.propagator(),
        }
    }

    /// `2·∂F/∂x*` over every slice and the probe, `Jᴴ·Uᴴ` of `2·td`, so `⟨g, u⟩` is the
    /// derivative of `F` along the displacement `u`.
    pub fn grads(&self) -> Vec<C64> {
        let op = self.jacobian();
        let y: Vec<C64> = self.td.iter().map(|t| 2.0 * t).collect();
        let mut x = vec![C64::default(); op.domain_len()];
        op.adjoint(&y, &mut x);
        x
    }

    /// The second derivative of `F` along the displacements `u` and `v`, pty-chi's bilinear
    /// Hessian with `A + EPS` in place of `A`:
    /// `Σ Re(conj(2·td)·L_uv) + 2 Σ_unmasked [(1 − d/A)·Σ_m Re(conj(L_u)·L_v) + (d/A)·c_u·c_v]`
    /// with `L_u = U·J u`, `L_uv = U·J″(u, v)` ([`SliceWaves::second_order`]),
    /// `c_u = Σ_m Re(conj(U psi/A)·L_u)` and `d = √iobs`.
    pub fn hessian(&self, u: &[C64], v: &[C64]) -> f64 {
        let n = self.probe_dims[1] * self.probe_dims[2];
        let op = self.jacobian();
        let (uo, up) = op.slices.split(u);
        let (vo, vp) = op.slices.split(v);
        let [mut lu, mut lv, mut luv] = self.waves.second_order((&uo, &up), (&vo, &vp));
        for l in [&mut lu, &mut lv, &mut luv] {
            op.propagator.forward(l);
        }
        let mut h: f64 = self
            .td
            .iter()
            .zip(&luv)
            .map(|(t, l)| 2.0 * (t.conj() * l).re)
            .sum();
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

/// `⟨a, b⟩`.
fn dot(a: &[C64], b: &[C64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a.conj() * b).re).sum()
}

/// BH state across epochs: the parameters, the epoch count and, with one batch per epoch, the
/// last direction.
pub struct Bh {
    params: BhParams,
    epochs: usize,
    prev: Option<Vec<C64>>,
}

impl Bh {
    pub fn new(params: BhParams) -> Self {
        assert!(params.batch_size > 0, "batch_size must be positive");
        Self {
            params,
            epochs: 0,
            prev: None,
        }
    }

    /// One epoch over [`random_batches`] of `addr`'s frames, from the params' seed and this
    /// epoch's index, each batch moving every slice of `objects` (one per slice of `optics`)
    /// and `probe` by `α·D`. With `g` the batch's [`BhBatch::grads`] and `Pg` it with the
    /// probe part times `ρ²`: `D = −Pg`, or with one batch per epoch, after the first epoch,
    /// `D = −Pg + β·D′` with `β = H(Pg, D′) / H(D′, D′)` and `D′` the previous epoch's
    /// direction; `α = −⟨g, D⟩ / H(D, D) · count / batch_size` with `count` the batch's frames.
    pub fn epoch(
        &mut self,
        addr: &AddrTable,
        obs: &Observed,
        optics: &Optics,
        objects: &mut [Object],
        probe: &mut Probe,
    ) -> Vec<BhStep> {
        let (nframe, nmode) = (addr.nframe(), addr.nmode());
        let size = self.params.batch_size.min(nframe);
        let full = size == nframe;
        let rho2 = self.params.probe_rho.powi(2) * optics.propagator().intensity_scale();
        let len = objects[0].as_slice().len();
        let split = objects.len() * len;
        let epoch = self.epochs as u64;
        self.epochs += 1;
        let mut steps = Vec::new();
        for frames in random_batches(nframe, size, self.params.seed, epoch) {
            let views: Vec<View> = frames
                .iter()
                .flat_map(|&f| &addr.views()[f * nmode..(f + 1) * nmode])
                .copied()
                .collect();
            let count = frames.len();
            let b = BhBatch::new(&views, obs, optics, objects, probe);
            let g = b.grads();
            let mut pg = g.clone();
            pg[split..].iter_mut().for_each(|c| *c *= rho2);
            let (beta, dir) = match self.prev.take().filter(|_| full) {
                Some(mut dir) => {
                    let beta = b.hessian(&pg, &dir) / b.hessian(&dir, &dir);
                    dir.iter_mut()
                        .zip(&pg)
                        .for_each(|(d, pg)| *d = beta * *d - pg);
                    (beta, dir)
                }
                None => (0.0, pg.iter().map(|c| -c).collect()),
            };
            let top = -dot(&g, &dir);
            let bottom = b.hessian(&dir, &dir);
            let alpha = top / bottom * count as f64 / size as f64;
            let (d_obj, d_probe) = dir.split_at(split);
            let targets = objects
                .iter_mut()
                .map(|o| o.as_mut_slice())
                .chain([probe.as_mut_slice()]);
            for (x, d) in targets.zip(d_obj.chunks_exact(len).chain([d_probe])) {
                x.iter_mut().zip(d).for_each(|(x, d)| *x += alpha * d);
            }
            if full {
                self.prev = Some(dir);
            }
            steps.push(BhStep {
                frames,
                beta,
                alpha,
                top,
                bottom,
            });
        }
        steps
    }
}

#[cfg(test)]
mod tests {
    use ptyrs_core::{Counts, Geometry, Positions, Stack};

    use super::*;
    use crate::linop::{dot_error, noise};
    use crate::{Propagator, Wavefield, register, shift};

    const SHAPE: [usize; 2] = [8, 6];
    const OBJ: [usize; 3] = [2, 14, 12];
    const PROBE: [usize; 3] = [2, 8, 6];
    const OBJ_LEN: usize = OBJ[0] * OBJ[1] * OBJ[2];
    const PROBE_LEN: usize = PROBE[0] * PROBE[1] * PROBE[2];

    /// Three frames of two probe and two object modes, overlapping 8x6 views in 14x12 slices.
    fn addr() -> AddrTable {
        let pos = Positions::new(vec![0.0, 2.0, -3.0], vec![1.0, -2.0, 2.0]).unwrap();
        AddrTable::new(&pos, [1.0, 1.0], SHAPE, [OBJ[1], OBJ[2]], 2, 2).unwrap()
    }

    /// Angular spectrum and a Fresnel transform after one slice, then near and far field
    /// after three slices with unequal gaps of Fresnel numbers of order 1.
    fn optics() -> Vec<Optics> {
        let near = |pixel: [f64; 2], d| Wavefield::new(SHAPE, 1e-10, pixel, d).unwrap();
        let gaps = |pixel: [f64; 2], d: [f64; 2]| d.map(|d| near(pixel, d));
        let g = Geometry::new(1e-10, 1.0, 1e-4, SHAPE).unwrap();
        let far = Optics::far_field(&g).with_slices(&gaps(g.pixel_size_object(), [5e-5, 1e-4]));
        let ms =
            Optics::free_space(&near([5e-8; 2], 1e-4)).with_slices(&gaps([5e-8; 2], [2e-5, 3e-5]));
        let all = vec![
            Optics::free_space(&near([5e-8; 2], 1e-4)),
            Optics::free_space(&near([5e-8; 2], 1e-3)),
            ms.unwrap(),
            far.unwrap(),
        ];
        let kinds: Vec<_> = all
            .iter()
            .map(|o| match o.propagator() {
                Propagator::Fourier(_) => 'F',
                Propagator::AngularSpectrum(_) => 'A',
                Propagator::Fresnel(_) => 'T',
            })
            .collect();
        assert_eq!(kinds, ['A', 'T', 'A', 'F']);
        all
    }

    /// Slices near 1 and a probe of a few photons per pixel, each slice and probe mode its own
    /// noise.
    fn state(nslice: usize) -> (Vec<Object>, Probe) {
        let objects = (0..nslice as u64)
            .map(|s| {
                let v = noise(OBJ_LEN, 10 + s)
                    .iter()
                    .map(|c| 1.0 + 0.2 * c)
                    .collect();
                Object::new(Stack::from_vec(OBJ, v).unwrap())
            })
            .collect();
        let v = noise(PROBE_LEN, 3).iter().map(|c| 4.0 * c).collect();
        (objects, Probe::new(Stack::from_vec(PROBE, v).unwrap()))
    }

    /// `U psi` of every view.
    fn far(optics: &Optics, addr: &AddrTable, objects: &[Object], probe: &Probe) -> Vec<C64> {
        let mut psi = SliceWaves::new(optics, addr.views(), objects, probe)
            .psi()
            .to_vec();
        optics.propagator().forward(&mut psi);
        psi
    }

    /// Counts of the state's intensities times a factor in `[0.7, 1.3)`, one pixel masked.
    fn observed(optics: &Optics, addr: &AddrTable, objects: &[Object], probe: &Probe) -> Observed {
        let (n, nmode) = (probe.plane_len(), addr.nmode());
        let psi = far(optics, addr, objects, probe);
        let jitter = noise(addr.nframe() * n, 77);
        let mut values = vec![0.0; addr.nframe() * n];
        for (f, block) in psi.chunks_exact(nmode * n).enumerate() {
            for j in 0..n {
                let icalc: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                let k = f * n + shift(j, SHAPE[0], SHAPE[1]);
                values[k] = (icalc * (1.0 + 0.3 * jitter[k].re)).round();
            }
        }
        values[5] = -1.0;
        let dims = [addr.nframe(), SHAPE[0], SHAPE[1]];
        Observed::from_counts(&Counts::from_f64(dims, &values, None).unwrap())
    }

    /// The function [`BhBatch::grads`] is the gradient of: `F` plus `2·EPS·√iobs·ln(A + EPS)`
    /// per unmasked pixel, whose amplitude derivative turns `1 − √iobs/A` into pty-chi's
    /// `1 − √iobs/(A + EPS)`.
    fn cost(optics: &Optics, addr: &AddrTable, obs: &Observed, x: &[C64]) -> f64 {
        let (objects, probe) = split(x);
        let (n, nmode) = (probe.plane_len(), addr.nmode());
        let psi = far(optics, addr, &objects, &probe);
        let mut f = 0.0;
        for (frame, block) in psi.chunks_exact(nmode * n).enumerate() {
            for (j, iobs) in obs.frames(frame..frame + 1).iter().enumerate() {
                let a = (0..nmode)
                    .map(|m| block[m * n + j].norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                if let Some(d) = iobs.map(f64::sqrt) {
                    f += (a - d).powi(2) + 2.0 * EPS * d * (a + EPS).ln();
                }
            }
        }
        f
    }

    fn flat(objects: &[Object], probe: &Probe) -> Vec<C64> {
        let mut x: Vec<C64> = objects.iter().flat_map(|o| o.as_slice()).copied().collect();
        x.extend_from_slice(probe.as_slice());
        x
    }

    fn split(x: &[C64]) -> (Vec<Object>, Probe) {
        let (o, p) = x.split_at(x.len() - PROBE_LEN);
        let objects = o
            .as_chunks::<OBJ_LEN>()
            .0
            .iter()
            .map(|o| Object::new(Stack::from_vec(OBJ, o.to_vec()).unwrap()))
            .collect();
        (
            objects,
            Probe::new(Stack::from_vec(PROBE, p.to_vec()).unwrap()),
        )
    }

    /// `x + t·d`.
    fn moved(x: &[C64], t: f64, d: &[C64]) -> Vec<C64> {
        x.iter().zip(d).map(|(x, d)| x + t * d).collect()
    }

    /// The five-point central difference of `f` at 0.
    fn slope(f: impl Fn(f64) -> f64, h: f64) -> f64 {
        (8.0 * (f(h) - f(-h)) - (f(2.0 * h) - f(-2.0 * h))) / (12.0 * h)
    }

    #[test]
    fn jacobian_registers_as_the_adjoint_of_its_tangent() {
        let addr = addr();
        for optics in &optics() {
            let (objects, probe) = state(optics.nslice());
            let obs = observed(optics, &addr, &objects, &probe);
            let b = BhBatch::new(addr.views(), &obs, optics, &objects, &probe);
            let err = dot_error(&b.jacobian());
            assert!(err < 1e-14, "dot test error {err:.3e}");
            register(b.jacobian()).unwrap();
        }
    }

    /// `⟨g, d⟩` and `H(u, v)` against central differences of `F` and of `⟨g(x + t·v), u⟩`
    /// along directions over every slice and the probe.
    #[test]
    fn grads_and_hessian_match_central_differences() {
        let addr = addr();
        for (k, optics) in optics().iter().enumerate() {
            let (objects, probe) = state(optics.nslice());
            let obs = observed(optics, &addr, &objects, &probe);
            let x = flat(&objects, &probe);
            let [d, u, v] = [21, 22, 23].map(|seed| noise(x.len(), seed));
            let b = BhBatch::new(addr.views(), &obs, optics, &objects, &probe);
            let at = |t: f64, d: &[C64]| moved(&x, t, d);
            let fd = slope(|t| cost(optics, &addr, &obs, &at(t, &d)), 1e-3);
            let g = dot(&b.grads(), &d);
            let err = (g - fd).abs() / fd.abs();
            assert!(
                err < 1e-8,
                "optics {k}: gradient {g} differences {fd}, {err:.3e}"
            );
            let (h_uv, h_vu) = (b.hessian(&u, &v), b.hessian(&v, &u));
            assert!(
                (h_uv - h_vu).abs() < 1e-13 * h_uv.abs(),
                "optics {k}: H(u, v) {h_uv} H(v, u) {h_vu}"
            );
            let du = |t: f64| {
                let (o, p) = split(&at(t, &v));
                dot(
                    &BhBatch::new(addr.views(), &obs, optics, &o, &p).grads(),
                    &u,
                )
            };
            let fd = slope(du, 1e-3);
            let err = (h_uv - fd).abs() / fd.abs();
            assert!(
                err < 1e-8,
                "optics {k}: H {h_uv} differences {fd}, {err:.3e}"
            );
        }
    }
}
