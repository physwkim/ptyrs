//! pty-chi BH (Carlsson, Nikitin and Wendt's bilinear-Hessian method, `reconstructors/bh.py`)
//! on the amplitude cost, as CPU f64 operators through an [`Optics`] (DESIGN B7): [`BhBatch`]
//! holds the cost's gradient and Hessian at one state over a batch of views, [`Bh`] runs the
//! epochs. Through several slices, which pty-chi's BH rejects, the object variable is every
//! slice of [`SliceWaves`].
//!
//! With [`Features::position`] the offsets join the variables on the engine's position model:
//! frame `b`'s probe is moved by its offset `d_b` as [`crate::engine::Lsqml`] moves it, where
//! pty-chi's BH translates the object patch. A Fourier translation or derivative factors over
//! a product only when the product is unaliased, which a view's probe times object window in
//! general is not, so the two costs, gradients and steps differ (by 30–80% on
//! `refs/f64/bh_pos.py`'s scan); they agree at zero offsets on a band-limited state, where the
//! model checks its signs and units against pty-chi.

use crate::base::{AddrTable, C64, Object, Probe, Stack, View};

use crate::engine::features::{POSITION_STEP, move_derivative, move_planes};
use crate::engine::linop::FarJacobian;
use crate::engine::{
    EngineError, Features, Fft2, LinearOp, Observed, Optics, ProbeVariation, SliceJacobian,
    SliceWaves, random_batches,
};

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
    /// propagator's [`crate::engine::Propagator::intensity_scale`] (`ny·nx` in the far field), so in
    /// probe displacements its gradient is scaled by `ρ²`.
    pub probe_rho: f64,
    /// pty-chi's `BHProbePositionOptions.rho` (0.1 there): under [`Features::position`] the
    /// offset variable is the offsets over it, so in displacements their gradient is scaled by
    /// `position_rho²`.
    pub position_rho: f64,
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

/// A gradient, direction or displacement of a [`BhBatch`]: `x` every slice's object, then the
/// probe, and `d` a `[dy, dx]` offset change per frame of the batch in its order, empty
/// without [`Features::position`]. `⟨a, b⟩ = Σ Re(conj(a.x)·b.x) + Σ a.d·b.d`.
#[derive(Clone, Debug, PartialEq)]
pub struct BhVector {
    pub x: Vec<C64>,
    pub d: Vec<[f64; 2]>,
}

impl BhVector {
    /// `⟨self, other⟩`.
    pub fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.d.len(), other.d.len());
        let d: f64 = (self.d.iter().zip(&other.d))
            .map(|(a, b)| a[0] * b[0] + a[1] * b[1])
            .sum();
        dot(&self.x, &other.x) + d
    }
}

/// The batch frames' probe moves: frame `b`'s copy of the probe is `M_b(P)`, [`move_planes`]
/// by its offset `d_b`, with these first and second derivatives over `d_b`.
struct Moves {
    offsets: Vec<[f64; 2]>,
    fft: Fft2,
    /// `∂_a M_b(P)` per axis `a`, laid out as the copies.
    first: [Vec<C64>; 2],
    /// `∂_a ∂_c M_b(P)` at index `a + c`.
    second: [Vec<C64>; 3],
}

/// The amplitude cost of a batch of views at one state of the slices, probe and offsets, and
/// its derivatives: `F = Σ_frames Σ_unmasked (A − √iobs)²` with `A = √(Σ_m |U psi_m|²)`,
/// `psi` the exit waves of [`SliceWaves`] at frame `b`'s probe `M_b(P)` and `U` the
/// [`Optics::propagator`]. `views` are whole frames of an [`AddrTable`], each frame's views in
/// its order. Every frame reads its own copy of the probe, the probe moved by the frame's
/// offset (`M_b` the identity without offsets), and the derivatives over the probe and the
/// offsets go through those copies.
pub struct BhBatch<'a> {
    optics: &'a Optics,
    waves: SliceWaves<'a>,
    obj_dims: [usize; 3],
    probe_dims: [usize; 3],
    /// The copies' dims, `nprobe` planes per frame.
    copy_dims: [usize; 3],
    nmode: usize,
    moves: Option<Moves>,
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
    /// `objects` holds one object per slice of `optics`; `offsets`, one `[dy, dx]` per frame of
    /// `views` in its order, the frames' [`Features::position`] (`None` for none).
    pub fn new(
        views: &[View],
        obs: &Observed,
        optics: &'a Optics,
        objects: &'a [Object],
        probe: &Probe,
        offsets: Option<&[[f64; 2]]>,
    ) -> Self {
        let n = probe.plane_len();
        let nmode = views
            .iter()
            .take_while(|v| v.frame == views[0].frame)
            .count();
        assert_eq!(views.len() % nmode, 0, "views must be whole frames");
        let nb = views.len() / nmode;
        let (uviews, mut copies) = ProbeVariation::none(probe.plane_shape()).unique(views, probe);
        let moves = offsets.map(|offsets| {
            assert_eq!(offsets.len(), nb, "one offset per frame of the batch");
            let fft = Fft2::new(probe.plane_shape());
            let q = optics.slice_quad(0);
            let block = probe.as_slice().len();
            let each = |axes: &[usize]| {
                let mut out = copies.as_slice().to_vec();
                for (p, &d) in out.chunks_exact_mut(block).zip(offsets) {
                    move_derivative(p, q, d, axes, &fft);
                }
                out
            };
            let first = [each(&[0]), each(&[1])];
            let second = [each(&[0, 0]), each(&[0, 1]), each(&[1, 1])];
            for (p, &d) in copies.as_mut_slice().chunks_exact_mut(block).zip(offsets) {
                move_planes(p, q, d, &fft);
            }
            Moves {
                offsets: offsets.to_vec(),
                fft,
                first,
                second,
            }
        });
        let waves = SliceWaves::new(optics, &uviews, objects, &copies);
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
            copy_dims: copies.dims(),
            nmode,
            moves,
            psi,
            amp,
            root,
            td,
        }
    }

    /// `x ↦ U·J x` with `J` the [`SliceJacobian`] over the slices and the frames' probe copies
    /// at this state, the path of every derivative.
    fn jacobian(&self) -> FarJacobian<'_> {
        FarJacobian {
            slices: SliceJacobian {
                waves: &self.waves,
                obj_dims: self.obj_dims,
                nslice: self.optics.nslice(),
                probe_dims: self.copy_dims,
            },
            propagator: self.optics.propagator(),
        }
    }

    /// The length of the slices' part of `x`.
    fn split(&self) -> usize {
        self.optics.nslice() * self.obj_dims.iter().product::<usize>()
    }

    /// Frame `b`'s map `M_b` of shared-probe planes `p`, or with `axes` its derivative over
    /// those offset components, per frame, laid out as the copies; `sign` −1 takes the
    /// offsets negated, `M_bᴴ`.
    fn moved(&self, p: &[C64], axes: &[usize], sign: f64, b: usize) -> Vec<C64> {
        let mut out = p.to_vec();
        if let Some(m) = &self.moves {
            let q = self.optics.slice_quad(0);
            let d = m.offsets[b].map(|d| sign * d);
            move_derivative(&mut out, q, d, axes, &m.fft);
        } else {
            assert!(axes.is_empty(), "no offsets to differentiate over");
        }
        out
    }

    /// `2·∂F/∂x*` over every slice and the probe and `∂F/∂d` per frame, so `⟨g, u⟩` is the
    /// derivative of `F` along the displacement `u`: `Jᴴ·Uᴴ` of `2·td` gives `g_b` per frame
    /// copy, the probe's is `Σ_b M_bᴴ g_b` and frame `b`'s offsets' `Re⟨g_b, ∂_a M_b(P)⟩`.
    pub fn grads(&self) -> BhVector {
        let op = self.jacobian();
        let y: Vec<C64> = self.td.iter().map(|t| 2.0 * t).collect();
        let mut x = vec![C64::default(); op.domain_len()];
        op.adjoint(&y, &mut x);
        let copies = x.split_off(self.split());
        let block = self.probe_dims.iter().product::<usize>();
        let d = match &self.moves {
            Some(m) => (copies.chunks_exact(block).enumerate())
                .map(|(b, g)| [0, 1].map(|a| dot(g, &m.first[a][b * block..][..block])))
                .collect(),
            None => Vec::new(),
        };
        let mut probe = vec![C64::default(); block];
        for (b, g) in copies.chunks_exact(block).enumerate() {
            let g = self.moved(g, &[], -1.0, b);
            probe.iter_mut().zip(&g).for_each(|(p, g)| *p += g);
        }
        x.extend(probe);
        BhVector { x, d }
    }

    /// `u` on the slices and the frames' probe copies: the copies' first-order change
    /// `M_b(u_P) + Σ_a u_da·∂_a M_b(P)`.
    fn lift(&self, u: &BhVector) -> (Vec<Object>, Probe) {
        let split = self.split();
        let block = self.probe_dims.iter().product::<usize>();
        let len = self.obj_dims.iter().product::<usize>();
        let objects = (u.x[..split].chunks_exact(len))
            .map(|o| Object::new(Stack::from_vec(self.obj_dims, o.to_vec()).unwrap()))
            .collect();
        let mut copies = Vec::with_capacity(self.copy_dims.iter().product());
        for b in 0..self.copy_dims[0] / self.probe_dims[0] {
            let mut p = self.moved(&u.x[split..], &[], 1.0, b);
            if let Some(m) = &self.moves {
                for (a, first) in m.first.iter().enumerate() {
                    let t = u.d[b][a];
                    (p.iter_mut().zip(&first[b * block..][..block])).for_each(|(p, f)| *p += t * f);
                }
            }
            copies.extend(p);
        }
        let copies = Probe::new(Stack::from_vec(self.copy_dims, copies).unwrap());
        (objects, copies)
    }

    /// The copies' second-order change along `u` and `v`:
    /// `Σ_a (v_da·∂_a M_b(u_P) + u_da·∂_a M_b(v_P)) + Σ_ac u_da·v_dc·∂_a ∂_c M_b(P)`.
    fn second(&self, m: &Moves, u: &BhVector, v: &BhVector) -> Probe {
        let split = self.split();
        let block = self.probe_dims.iter().product::<usize>();
        let mut copies = Vec::with_capacity(self.copy_dims.iter().product());
        for b in 0..m.offsets.len() {
            let mut p = vec![C64::default(); block];
            for a in 0..2 {
                for (s, w) in [(v.d[b][a], &u.x[split..]), (u.d[b][a], &v.x[split..])] {
                    let t = self.moved(w, &[a], 1.0, b);
                    p.iter_mut().zip(&t).for_each(|(p, t)| *p += s * t);
                }
                for c in 0..2 {
                    let s = u.d[b][a] * v.d[b][c];
                    let t = &m.second[a + c][b * block..][..block];
                    p.iter_mut().zip(t).for_each(|(p, t)| *p += s * t);
                }
            }
            copies.extend(p);
        }
        Probe::new(Stack::from_vec(self.copy_dims, copies).unwrap())
    }

    /// The second derivative of `F` along the displacements `u` and `v`, pty-chi's bilinear
    /// Hessian with `A + EPS` in place of `A`:
    /// `Σ Re(conj(2·td)·L_uv) + 2 Σ_unmasked [(1 − d/A)·Σ_m Re(conj(L_u)·L_v) + (d/A)·c_u·c_v]`
    /// with `L_u = U·J u′`, `u′` the slices' and the copies' first-order change along `u`,
    /// `L_uv = U·(J″(u′, v′) + J(0, p″))` ([`SliceWaves::second_order`]) with `p″` the copies'
    /// second-order change, `c_u = Σ_m Re(conj(U psi/A)·L_u)` and `d = √iobs`.
    pub fn hessian(&self, u: &BhVector, v: &BhVector) -> f64 {
        let n = self.probe_dims[1] * self.probe_dims[2];
        let (uo, up) = self.lift(u);
        let (vo, vp) = self.lift(v);
        let [mut lu, mut lv, mut luv] = self.waves.second_order((&uo, &up), (&vo, &vp));
        if let Some(m) = &self.moves {
            let len = self.obj_dims.iter().product::<usize>();
            let zero: Vec<Object> = (0..uo.len())
                .map(|_| {
                    Object::new(Stack::from_vec(self.obj_dims, vec![C64::default(); len]).unwrap())
                })
                .collect();
            let t = self.waves.tangent(&zero, &self.second(m, u, v));
            luv.iter_mut().zip(&t).for_each(|(l, t)| *l += t);
        }
        let propagator = self.optics.propagator();
        for l in [&mut lu, &mut lv, &mut luv] {
            propagator.forward(l);
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

/// BH state across epochs: the parameters, the features it refines, the epoch count and, with
/// one batch per epoch, the last direction.
pub struct Bh {
    params: BhParams,
    features: Features,
    epochs: usize,
    /// The last direction's `x` and its offsets' part by frame of the scan, empty without
    /// positions.
    prev: Option<(Vec<C64>, Vec<[f64; 2]>)>,
}

impl Bh {
    pub fn new(params: BhParams) -> Self {
        assert!(params.batch_size > 0, "batch_size must be positive");
        Self {
            params,
            features: Features::default(),
            epochs: 0,
            prev: None,
        }
    }

    /// This run modelling and refining `features`, which hold one entry per frame of the scan
    /// it runs on, the offsets weighted by [`BhParams::position_rho`]. pty-chi's BH refines the
    /// positions alone; it has neither a per-frame intensity scale nor a detector background,
    /// and those are refused.
    pub fn with_features(self, features: Features) -> Result<Self, EngineError> {
        if features.scale.is_some() || features.background.is_some() {
            return Err(EngineError::Invalid(
                "BH refines no intensity scale or background, as pty-chi's".into(),
            ));
        }
        if let Some(d) = &features.position
            && !d.iter().flatten().all(|d| d.is_finite())
        {
            return Err(EngineError::Invalid(
                "position offsets must be finite".into(),
            ));
        }
        Ok(Self { features, ..self })
    }

    /// The features as the last step left them.
    pub fn features(&self) -> &Features {
        &self.features
    }

    /// One epoch over [`random_batches`] of `addr`'s frames, from the params' seed and this
    /// epoch's index, each batch moving every slice of `objects` (one per slice of `optics`)
    /// and `probe` by `α·D` and, with [`Features::position`], its frames' offsets by
    /// `0.3·α·D_d` (pty-chi's position step size, unclipped). With `g` the batch's
    /// [`BhBatch::grads`] and `Pg` it with the probe part times `ρ²` and the offsets' times
    /// [`BhParams::position_rho`]`²`: `D = −Pg`, or with one batch per epoch, after the first
    /// epoch, `D = −Pg + β·D′` with `β = H(Pg, D′) / H(D′, D′)` and `D′` the previous epoch's
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
        if let Some(d) = &self.features.position {
            assert_eq!(d.len(), nframe, "one position offset per frame");
        }
        let size = self.params.batch_size.min(nframe);
        let full = size == nframe;
        let rho2 = self.params.probe_rho.powi(2) * optics.propagator().intensity_scale();
        let pos_rho2 = self.params.position_rho.powi(2);
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
            let offsets: Option<Vec<[f64; 2]>> =
                (self.features.position.as_ref()).map(|d| frames.iter().map(|&f| d[f]).collect());
            let b = BhBatch::new(&views, obs, optics, objects, probe, offsets.as_deref());
            let g = b.grads();
            let mut pg = g.clone();
            pg.x[split..].iter_mut().for_each(|c| *c *= rho2);
            pg.d.iter_mut().flatten().for_each(|d| *d *= pos_rho2);
            let (beta, dir) = match self.prev.take().filter(|_| full) {
                Some((x, by_frame)) => {
                    let d = if by_frame.is_empty() {
                        Vec::new()
                    } else {
                        frames.iter().map(|&f| by_frame[f]).collect()
                    };
                    let mut dir = BhVector { x, d };
                    let beta = b.hessian(&pg, &dir) / b.hessian(&dir, &dir);
                    dir.x
                        .iter_mut()
                        .zip(&pg.x)
                        .for_each(|(d, pg)| *d = beta * *d - pg);
                    (dir.d.iter_mut().flatten())
                        .zip(pg.d.iter().flatten())
                        .for_each(|(d, pg)| *d = beta * *d - pg);
                    (beta, dir)
                }
                None => {
                    let x = pg.x.iter().map(|c| -c).collect();
                    let d = pg.d.iter().map(|d| d.map(|d| -d)).collect();
                    (0.0, BhVector { x, d })
                }
            };
            let top = -g.dot(&dir);
            let bottom = b.hessian(&dir, &dir);
            let alpha = top / bottom * count as f64 / size as f64;
            let (d_obj, d_probe) = dir.x.split_at(split);
            let targets = objects
                .iter_mut()
                .map(|o| o.as_mut_slice())
                .chain([probe.as_mut_slice()]);
            for (x, d) in targets.zip(d_obj.chunks_exact(len).chain([d_probe])) {
                x.iter_mut().zip(d).for_each(|(x, d)| *x += alpha * d);
            }
            let mut by_frame = Vec::new();
            if let Some(offsets) = &mut self.features.position {
                by_frame = vec![[0.0; 2]; nframe];
                for (&f, d) in frames.iter().zip(&dir.d) {
                    by_frame[f] = *d;
                    for a in 0..2 {
                        offsets[f][a] += POSITION_STEP * alpha * d[a];
                    }
                }
            }
            if full {
                self.prev = Some((dir.x, by_frame));
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
    use crate::base::{Counts, Geometry, Positions, Stack};

    use super::*;
    use crate::engine::linop::{dot_error, noise};
    use crate::engine::{Propagator, Wavefield, register, shift};

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

    /// Offsets of the three frames.
    const OFFSETS: [[f64; 2]; 3] = [[0.3, -0.2], [-0.45, 0.1], [0.2, 0.4]];

    /// `U psi` of every view, frame `f`'s probe moved by `offsets[f]` ([`move_planes`]).
    fn far(
        optics: &Optics,
        addr: &AddrTable,
        objects: &[Object],
        probe: &Probe,
        offsets: Option<&[[f64; 2]]>,
    ) -> Vec<C64> {
        let shape = probe.plane_shape();
        let (views, mut copies) = ProbeVariation::none(shape).unique(addr.views(), probe);
        if let Some(offsets) = offsets {
            let fft = Fft2::new(shape);
            let block = probe.as_slice().len();
            for (p, &d) in copies.as_mut_slice().chunks_exact_mut(block).zip(offsets) {
                move_planes(p, optics.slice_quad(0), d, &fft);
            }
        }
        let mut psi = SliceWaves::new(optics, &views, objects, &copies)
            .psi()
            .to_vec();
        optics.propagator().forward(&mut psi);
        psi
    }

    /// Counts of the state's intensities times a factor in `[0.7, 1.3)`, one pixel masked.
    fn observed(optics: &Optics, addr: &AddrTable, objects: &[Object], probe: &Probe) -> Observed {
        let (n, nmode) = (probe.plane_len(), addr.nmode());
        let psi = far(optics, addr, objects, probe, None);
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
    fn cost(
        optics: &Optics,
        addr: &AddrTable,
        obs: &Observed,
        (x, offsets): &(Vec<C64>, Option<Vec<[f64; 2]>>),
    ) -> f64 {
        let (objects, probe) = split(x);
        let (n, nmode) = (probe.plane_len(), addr.nmode());
        let psi = far(optics, addr, &objects, &probe, offsets.as_deref());
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

    /// The state `(x, offsets)` plus `t·w`.
    fn at(
        (x, offsets): &(Vec<C64>, Option<Vec<[f64; 2]>>),
        t: f64,
        w: &BhVector,
    ) -> (Vec<C64>, Option<Vec<[f64; 2]>>) {
        let d = offsets.as_ref().map(|d| {
            (d.iter().zip(&w.d))
                .map(|(d, w)| [d[0] + t * w[0], d[1] + t * w[1]])
                .collect()
        });
        (moved(x, t, &w.x), d)
    }

    /// [`BhBatch::grads`] at the state `(x, offsets)`.
    fn grads_at(
        optics: &Optics,
        addr: &AddrTable,
        obs: &Observed,
        (x, offsets): &(Vec<C64>, Option<Vec<[f64; 2]>>),
    ) -> BhVector {
        let (objects, probe) = split(x);
        BhBatch::new(
            addr.views(),
            obs,
            optics,
            &objects,
            &probe,
            offsets.as_deref(),
        )
        .grads()
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
            for offsets in [None, Some(&OFFSETS[..])] {
                let b = BhBatch::new(addr.views(), &obs, optics, &objects, &probe, offsets);
                let err = dot_error(&b.jacobian());
                assert!(err < 1e-14, "dot test error {err:.3e}");
                register(b.jacobian()).unwrap();
            }
        }
    }

    /// `⟨g, w⟩` and `H(u, v)` against central differences of `F` and of `⟨g(x + t·v), u⟩`
    /// along directions over every slice and the probe, then also the offsets.
    #[test]
    fn grads_and_hessian_match_central_differences() {
        let addr = addr();
        for (k, optics) in optics().iter().enumerate() {
            let (objects, probe) = state(optics.nslice());
            let obs = observed(optics, &addr, &objects, &probe);
            for offsets in [None, Some(OFFSETS.to_vec())] {
                let has = offsets.is_some();
                let x0 = (flat(&objects, &probe), offsets);
                let b = BhBatch::new(
                    addr.views(),
                    &obs,
                    optics,
                    &objects,
                    &probe,
                    x0.1.as_deref(),
                );
                let [w, u, v] = [21, 22, 23].map(|seed| BhVector {
                    x: noise(x0.0.len(), seed),
                    d: (x0.1.as_ref())
                        .map_or(Vec::new(), |d| noise(d.len(), seed + 10))
                        .iter()
                        .map(|c| [c.re, c.im])
                        .collect(),
                });
                let fd = slope(|t| cost(optics, &addr, &obs, &at(&x0, t, &w)), 1e-3);
                let g = b.grads().dot(&w);
                let err = (g - fd).abs() / fd.abs();
                assert!(
                    err < 1e-8,
                    "optics {k} offsets {has}: gradient {g} differences {fd}, {err:.3e}"
                );
                let (h_uv, h_vu) = (b.hessian(&u, &v), b.hessian(&v, &u));
                assert!(
                    (h_uv - h_vu).abs() < 1e-13 * h_uv.abs(),
                    "optics {k} offsets {has}: H(u, v) {h_uv} H(v, u) {h_vu}"
                );
                let du = |t: f64| grads_at(optics, &addr, &obs, &at(&x0, t, &v)).dot(&u);
                let fd = slope(du, 1e-3);
                let err = (h_uv - fd).abs() / fd.abs();
                assert!(
                    err < 1e-8,
                    "optics {k} offsets {has}: H {h_uv} differences {fd}, {err:.3e}"
                );
            }
        }
    }
}
