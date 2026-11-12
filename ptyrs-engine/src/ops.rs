//! The PyNX CPU operators as free functions over plain arrays, each with its contract.

use std::f64::consts::PI;
use std::ops::Range;

use ptyrs_core::{C64, Geometry, Object, Probe, Stack, View};

use crate::features::Model;
use crate::{Observed, shift};

/// Far-field quadratic phase `q`, natural (centred) layout `[ny, nx]`:
/// `q[y, x] = exp(i·π/(λz)·((x − nx/2)²·px² + (y − ny/2)²·py²))` with `[py, px]` the object
/// pixel ([`Geometry::pixel_size_object`]).
///
/// PyNX `calc_quadratic_phase_factor` uses `px` on both axes and rounds `π/(λz)` and `px` to
/// f32; the two agree on square frames.
pub fn quad_phase(geometry: &Geometry) -> Vec<C64> {
    let [ny, nx] = geometry.frame;
    let [py, px] = geometry.pixel_size_object();
    let f = PI / (geometry.wavelength * geometry.distance);
    let mut q = Vec::with_capacity(ny * nx);
    for y in 0..ny {
        let dy = y as f64 - (ny / 2) as f64;
        for x in 0..nx {
            let dx = x as f64 - (nx / 2) as f64;
            q.push(C64::from_polar(
                1.0,
                f * (dx * dx * px * px + dy * dy * py * py),
            ));
        }
    }
    q
}

/// Natural `(y, x)` of shifted index `j` in an `[ny, nx]` plane.
#[inline]
pub(crate) fn natural(j: usize, ny: usize, nx: usize) -> (usize, usize, usize) {
    let i = shift(j, ny, nx);
    (i, i / nx, i % nx)
}

/// ObjProbe2Psi: for every view `k`,
/// `psi[k] = fftshift(P[probe_k] · O[obj_k, oy_k .. oy_k + ny, ox_k .. ox_k + nx] · q)`.
///
/// `psi` holds `views.len()` planes of the probe shape `[ny, nx]` and is overwritten; `quad` is
/// [`quad_phase`].
pub fn obj_probe_to_psi(
    views: &[View],
    object: &Object,
    probe: &Probe,
    quad: &[C64],
    psi: &mut [C64],
) {
    let [ny, nx] = probe.plane_shape();
    let n = ny * nx;
    let nxo = object.plane_shape()[1];
    assert_eq!(
        psi.len(),
        views.len() * n,
        "psi must hold one plane per view"
    );
    for (v, out) in views.iter().zip(psi.chunks_exact_mut(n)) {
        let p = probe.plane(v.probe as usize);
        let o = object.plane(v.obj as usize);
        let (oy, ox) = (v.oy as usize, v.ox as usize);
        for (j, out) in out.iter_mut().enumerate() {
            let (i, y, x) = natural(j, ny, nx);
            *out = p[i] * o[(oy + y) * nxo + ox + x] * quad[i];
        }
    }
}

/// ApplyAmplitude on detector-plane `psi` covering the shifted frames `frames` of `obs`
/// (`frames.len() * nmode` planes, frame-major): at every unmasked pixel
/// `psi *= sqrt(iobs) / max(1e-12, sqrt(Σ_modes |psi|²))`; masked pixels keep their value
/// (PyNX sets `r = 1` where `iobs < 0`).
pub fn apply_amplitude(obs: &Observed, frames: Range<usize>, nmode: usize, psi: &mut [C64]) {
    apply_amplitude_with(obs, frames, nmode, psi, Model::default());
}

/// [`apply_amplitude`] against `model`'s intensity: `sqrt(icalc)` in place of
/// `sqrt(Σ_modes |psi|²)`.
pub(crate) fn apply_amplitude_with(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    psi: &mut [C64],
    model: Model<'_>,
) {
    let n = obs.plane_len();
    assert_eq!(psi.len(), frames.len() * nmode * n);
    for ((f, o), block) in frames
        .clone()
        .zip(obs.frames(frames).chunks_exact(n))
        .zip(psi.chunks_exact_mut(nmode * n))
    {
        for (j, iobs) in o.iter().enumerate() {
            let Some(iobs) = *iobs else { continue };
            let i = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
            let calc = model.calc(f, j, i).sqrt();
            let r = iobs.sqrt() / calc.max(1e-12);
            for m in 0..nmode {
                block[m * n + j] *= r;
            }
        }
    }
}

/// Log-likelihood sums of detector-plane `psi` against `obs` (PyNX `LLK`), with
/// `icalc = Σ_modes |psi|²` per pixel:
/// - Poisson: `icalc − iobs + iobs·ln(iobs/icalc)` where `iobs > 0`, `icalc` where `iobs = 0`;
/// - Gaussian: `(iobs − icalc)² / (iobs + 1)`;
/// - Euclidian: `4·(√iobs − √icalc)²`;
/// - `nb_photons_calc`: `Σ icalc`.
///
/// Masked pixels contribute to none of the four (PyNX CUDA `LLKAll`; the PyNX CPU operator
/// adds them to the Gaussian, Euclidian and photon sums). [`Llk::normalised`] gives the values
/// PyNX records.
pub fn llk(obs: &Observed, frames: Range<usize>, nmode: usize, psi: &[C64]) -> Llk {
    llk_with(obs, frames, nmode, psi, Model::default())
}

/// [`llk`] of `model`'s intensity.
pub(crate) fn llk_with(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    psi: &[C64],
    model: Model<'_>,
) -> Llk {
    let n = obs.plane_len();
    assert_eq!(psi.len(), frames.len() * nmode * n);
    let mut s = Llk::default();
    for ((f, o), block) in frames
        .clone()
        .zip(obs.frames(frames).chunks_exact(n))
        .zip(psi.chunks_exact(nmode * n))
    {
        for (j, iobs) in o.iter().enumerate() {
            let Some(iobs) = *iobs else { continue };
            let i = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
            let icalc = model.calc(f, j, i);
            s.poisson += if iobs > 0.0 {
                icalc - iobs + iobs * (iobs / icalc).ln()
            } else {
                icalc
            };
            s.gaussian += (iobs - icalc).powi(2) / (iobs + 1.0);
            s.euclidian += 4.0 * (iobs.sqrt() - icalc.sqrt()).powi(2);
            s.nb_photons_calc += icalc;
        }
    }
    s
}

/// PyNX `GradPoissonFourier` (no background, and without the CUDA operator's Hann filter) on
/// detector-plane `psi`: at every unmasked pixel `psi *= 2·(1 − iobs / max(1e-12, icalc))`
/// with `icalc = Σ_modes |psi|²`; masked pixels become 0. The result is the gradient of
/// [`llk`]'s Poisson sum over `(Re psi, Im psi)`, packed as `∂L/∂Re + i·∂L/∂Im = 2·∂L/∂psi*`;
/// PyNX keeps `∂L/∂psi*`, half of it.
pub fn poisson_grad(obs: &Observed, frames: Range<usize>, nmode: usize, psi: &mut [C64]) {
    poisson_grad_with(obs, frames, nmode, psi, Model::default());
}

/// [`poisson_grad`] of [`llk_with`]'s Poisson sum: `psi *= 2·s·(1 − iobs / max(1e-12, icalc))`
/// with `s` the frame's intensity scale and `icalc` `model`'s.
pub(crate) fn poisson_grad_with(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    psi: &mut [C64],
    model: Model<'_>,
) {
    let n = obs.plane_len();
    assert_eq!(psi.len(), frames.len() * nmode * n);
    for ((f, o), block) in frames
        .clone()
        .zip(obs.frames(frames).chunks_exact(n))
        .zip(psi.chunks_exact_mut(nmode * n))
    {
        for (j, iobs) in o.iter().enumerate() {
            let r = match *iobs {
                Some(iobs) => {
                    let i = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                    let icalc = model.calc(f, j, i);
                    model.scaled(f, 2.0 * (1.0 - iobs / icalc.max(1e-12)))
                }
                None => 0.0,
            };
            for m in 0..nmode {
                block[m * n + j] *= r;
            }
        }
    }
}

/// PyNX `CG_Poisson_Gamma` (no background) on detector-plane `po = FT(P·O·q)`,
/// `a = FT(dP·O·q + P·dO·q)` and `b = FT(dP·dO·q)`: the numerator and denominator of the
/// step `γ = −L′(0) / L″(0)` of the Poisson sum `L(γ)` of [`llk`] at `O + γ·dO`, `P + γ·dP`.
/// Over unmasked pixels, with `icalc = max(Σ_modes |po|², 1e-12)`, `r = Σ Re(conj(po)·a)`,
/// `s = Σ (|a|² + 2·Re(conj(po)·b))` and `f = 1 − iobs / icalc`:
/// `num = Σ −2·r·f`, `den = Σ 2·s·f + iobs·(2·r / icalc)²`. The `1e-12` floor is
/// [`poisson_grad`]'s, so `num = −Re Σ conj(g)·d` for the gradient `g` it yields; PyNX floors at
/// `1e-32`, and its `s` has `Re(PdO)·Im(dPO)` where `Im(PdO)·Im(dPO)` belongs.
pub fn cg_poisson_gamma(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    po: &[C64],
    a: &[C64],
    b: &[C64],
) -> (f64, f64) {
    cg_poisson_gamma_with(obs, frames, nmode, [po, a, b], Model::default())
}

/// [`cg_poisson_gamma`] of [`llk_with`]'s Poisson sum: `icalc` `model`'s, `r` and `s` times
/// the frame's intensity scale.
pub(crate) fn cg_poisson_gamma_with(
    obs: &Observed,
    frames: Range<usize>,
    nmode: usize,
    [po, a, b]: [&[C64]; 3],
    model: Model<'_>,
) -> (f64, f64) {
    let n = obs.plane_len();
    let len = frames.len() * nmode * n;
    assert!(po.len() == len && a.len() == len && b.len() == len);
    let (mut num, mut den) = (0.0, 0.0);
    for (k, (f, o)) in frames
        .clone()
        .zip(obs.frames(frames).chunks_exact(n))
        .enumerate()
    {
        let at = |m: usize, j: usize| (k * nmode + m) * n + j;
        for (j, iobs) in o.iter().enumerate() {
            let Some(iobs) = *iobs else { continue };
            let (mut calc, mut r, mut s) = (0.0, 0.0, 0.0);
            for m in 0..nmode {
                let (p, a, b) = (po[at(m, j)], a[at(m, j)], b[at(m, j)]);
                calc += p.norm_sqr();
                r += (p.conj() * a).re;
                s += a.norm_sqr() + 2.0 * (p.conj() * b).re;
            }
            let calc = model.calc(f, j, calc).max(1e-12);
            let (r, s) = (model.scaled(f, r), model.scaled(f, s));
            let d = 1.0 - iobs / calc;
            num -= 2.0 * r * d;
            den += 2.0 * s * d + iobs * (2.0 * r / calc).powi(2);
        }
    }
    (num, den)
}

/// Log-likelihood values: sums from [`llk`], or per-observation values from
/// [`Llk::normalised`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Llk {
    pub poisson: f64,
    pub gaussian: f64,
    pub euclidian: f64,
    pub nb_photons_calc: f64,
}

impl Llk {
    /// The three LLK values divided by `nb_obs`; `nb_photons_calc` stays a total (PyNX
    /// `get_llk(norm=True)`, the values `update_history` records).
    pub fn normalised(self, nb_obs: usize) -> Llk {
        let n = nb_obs as f64;
        Llk {
            poisson: self.poisson / n,
            gaussian: self.gaussian / n,
            euclidian: self.euclidian / n,
            nb_photons_calc: self.nb_photons_calc,
        }
    }
}

/// Object back-projection, the adjoint of [`obj_probe_to_psi`] as a map of the object: over
/// views in order, `num[obj_k, win_k] += fftshift(psi[k]) · conj(q · P[probe_k])` and
/// `norm[obj_k, win_k] += |P[probe_k]|²`. Returns `(num, norm)` with shape `obj_dims`.
pub fn accumulate_obj(
    views: &[View],
    obj_dims: [usize; 3],
    probe: &Probe,
    quad: &[C64],
    psi: &[C64],
) -> (Stack<C64>, Stack<f64>) {
    let [ny, nx] = probe.plane_shape();
    let n = ny * nx;
    let nxo = obj_dims[2];
    assert_eq!(psi.len(), views.len() * n);
    let mut num = Stack::filled(obj_dims, C64::default());
    let mut norm = Stack::filled(obj_dims, 0.0);
    for (v, w) in views.iter().zip(psi.chunks_exact(n)) {
        let p = probe.plane(v.probe as usize);
        let (oy, ox) = (v.oy as usize, v.ox as usize);
        let (num, norm) = (
            num.plane_mut(v.obj as usize),
            norm.plane_mut(v.obj as usize),
        );
        for (j, w) in w.iter().enumerate() {
            let (i, y, x) = natural(j, ny, nx);
            let o = (oy + y) * nxo + ox + x;
            num[o] += w * (quad[i] * p[i]).conj();
            norm[o] += p[i].norm_sqr();
        }
    }
    (num, norm)
}

/// Probe back-projection, the adjoint of [`obj_probe_to_psi`] as a map of the probe: over views
/// in order, `num[probe_k] += fftshift(psi[k]) · conj(q · O[obj_k, win_k])` and
/// `norm[probe_k] += |O[obj_k, win_k]|²`. Returns `(num, norm)` with shape `probe_dims`.
pub fn accumulate_probe(
    views: &[View],
    object: &Object,
    probe_dims: [usize; 3],
    quad: &[C64],
    psi: &[C64],
) -> (Stack<C64>, Stack<f64>) {
    let [_, ny, nx] = probe_dims;
    let n = ny * nx;
    let nxo = object.plane_shape()[1];
    assert_eq!(psi.len(), views.len() * n);
    let mut num = Stack::filled(probe_dims, C64::default());
    let mut norm = Stack::filled(probe_dims, 0.0);
    for (v, w) in views.iter().zip(psi.chunks_exact(n)) {
        let ob = object.plane(v.obj as usize);
        let (oy, ox) = (v.oy as usize, v.ox as usize);
        let (num, norm) = (
            num.plane_mut(v.probe as usize),
            norm.plane_mut(v.probe as usize),
        );
        for (j, w) in w.iter().enumerate() {
            let (i, y, x) = natural(j, ny, nx);
            let o = ob[(oy + y) * nxo + ox + x];
            num[i] += w * (quad[i] * o).conj();
            norm[i] += o.norm_sqr();
        }
    }
    (num, norm)
}

/// Psi2ObjProbe on object-plane `psi`. An array given `Some(inertia)` becomes
/// `(num + reg·old) / max(norm + reg, 1e-12)` with `(num, norm)` from [`accumulate_obj`] /
/// [`accumulate_probe`] and `reg = max(norm) · inertia` (max over every mode and pixel).
///
/// Both back-projections read `object` and `probe` as they are on entry: with both updates the
/// probe is rebuilt from the old object (PyNX `Psi2ObjProbe(update_object=True,
/// update_probe=True)`, used by AP). DM calls it twice, object first.
pub fn psi_to_obj_probe(
    views: &[View],
    quad: &[C64],
    psi: &[C64],
    object: &mut Object,
    probe: &mut Probe,
    obj_inertia: Option<f64>,
    probe_inertia: Option<f64>,
) {
    let obj_acc = obj_inertia.map(|_| accumulate_obj(views, object.dims(), probe, quad, psi));
    if let Some(inertia) = probe_inertia {
        let (num, norm) = accumulate_probe(views, object, probe.dims(), quad, psi);
        blend(probe, &num, &norm, inertia);
    }
    if let (Some(inertia), Some((num, norm))) = (obj_inertia, obj_acc) {
        blend(object, &num, &norm, inertia);
    }
}

fn blend(old: &mut Stack<C64>, num: &Stack<C64>, norm: &Stack<f64>, inertia: f64) {
    let reg = norm.as_slice().iter().fold(0.0, |a: f64, &b| a.max(b)) * inertia;
    for ((o, a), w) in old
        .as_mut_slice()
        .iter_mut()
        .zip(num.as_slice())
        .zip(norm.as_slice())
    {
        *o = (a + *o * reg) / (w + reg).max(1e-12);
    }
}

#[cfg(test)]
mod tests {
    use ptyrs_core::{AddrTable, Counts, MASKED, Positions};

    use super::*;
    use crate::{ObjectToPsi, ProbeToPsi, register};

    fn wave(n: usize, seed: f64) -> Vec<C64> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.731 + seed;
                C64::new(t.sin() + 0.3, (1.7 * t).cos())
            })
            .collect()
    }

    /// Three frames, two probe modes, overlapping views, 8x6 frames in a 14x12 object.
    fn setup() -> (AddrTable, Object, Probe, Vec<C64>) {
        let pos = Positions::new(vec![0.0, 2.0, -3.0], vec![1.0, -2.0, 2.0]).unwrap();
        let addr = AddrTable::new(&pos, [1.0, 1.0], [8, 6], [14, 12], 2, 1).unwrap();
        let o = Object::new(Stack::from_vec([1, 14, 12], wave(168, 0.2)).unwrap());
        let p = Probe::new(Stack::from_vec([2, 8, 6], wave(96, 1.9)).unwrap());
        let g = Geometry::new(1e-10, 1.0, 1e-4, [8, 6]).unwrap();
        (addr, o, p, quad_phase(&g))
    }

    #[test]
    fn quad_phase_is_one_at_centre_and_symmetric() {
        let g = Geometry::new(1e-10, 2.0, 5e-5, [8, 8]).unwrap();
        let q = quad_phase(&g);
        assert_eq!(q[4 * 8 + 4], C64::new(1.0, 0.0));
        assert!((q[3 * 8 + 4] - q[5 * 8 + 4]).norm() < 1e-15);
        let px = g.pixel_size_object()[1];
        let want = C64::from_polar(1.0, PI / 2e-10 * 2.0 * px * px);
        assert!((q[3 * 8 + 5] - want).norm() < 1e-14);
    }

    #[test]
    fn back_projections_register_as_adjoints_of_obj_probe_to_psi() {
        let (addr, o, p, q) = setup();
        let views = addr.views();
        register(ObjectToPsi {
            views,
            obj_dims: o.dims(),
            probe: &p,
            quad: &q,
        })
        .unwrap();
        register(ProbeToPsi {
            views,
            object: &o,
            probe_dims: p.dims(),
            quad: &q,
        })
        .unwrap();
    }

    #[test]
    fn psi_to_obj_probe_keeps_a_consistent_pair() {
        // psi = P·O·q exactly: both updates must return the same object and probe.
        let (addr, o, p, q) = setup();
        let views = addr.views();
        let mut psi = vec![C64::default(); views.len() * 48];
        obj_probe_to_psi(views, &o, &p, &q, &mut psi);
        let (mut o2, mut p2) = (o.clone(), p.clone());
        psi_to_obj_probe(views, &q, &psi, &mut o2, &mut p2, Some(0.01), Some(0.001));
        for (a, b) in o2.as_slice().iter().zip(o.as_slice()) {
            assert!((a - b).norm() < 1e-12);
        }
        for (a, b) in p2.as_slice().iter().zip(p.as_slice()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    fn observed(raw: Vec<u16>, dims: [usize; 3]) -> Observed {
        Observed::from_counts(&Counts::from_raw(dims, raw).unwrap())
    }

    #[test]
    fn apply_amplitude_sets_modulus_and_skips_masked() {
        let mut raw: Vec<u16> = (0..48).map(|i| (i * 7 % 50) as u16).collect();
        raw[20] = MASKED;
        let obs = observed(raw, [2, 4, 6]);
        let before = wave(2 * 2 * 24, 0.7);
        let mut psi = before.clone();
        apply_amplitude(&obs, 0..2, 2, &mut psi);
        let o = obs.frames(0..2);
        for f in 0..2 {
            for j in 0..24 {
                let k = |m: usize| (f * 2 + m) * 24 + j;
                match o[f * 24 + j] {
                    Some(i) => {
                        let c = psi[k(0)].norm_sqr() + psi[k(1)].norm_sqr();
                        assert!((c - i).abs() < 1e-12 * i.max(1.0));
                        // phase and mode ratio are kept
                        let r = psi[k(0)] / before[k(0)];
                        assert!(r.im.abs() < 1e-12 && (psi[k(1)] / before[k(1)] - r).norm() < 1e-9);
                    }
                    None => assert_eq!((psi[k(0)], psi[k(1)]), (before[k(0)], before[k(1)])),
                }
            }
        }
    }

    #[test]
    fn llk_matches_hand_values_and_skips_masked() {
        // one frame, one mode, 2x2: iobs [4, 0, MASKED, 9] in shifted order
        let raw = vec![9, MASKED, 0, 4]; // detector order; shift swaps both axes for 2x2
        let obs = observed(raw, [1, 2, 2]);
        assert_eq!(obs.frames(0..1), &[Some(4.0), Some(0.0), None, Some(9.0)]);
        let psi = [2.0, 1.0, 5.0, 2.0].map(|a| C64::new(0.0, a));
        let l = llk(&obs, 0..1, 1, &psi);
        let p = (4.0 - 4.0) + 1.0 + (4.0 - 9.0 + 9.0 * (9.0f64 / 4.0).ln());
        assert!((l.poisson - p).abs() < 1e-14);
        assert!((l.gaussian - (0.0 + 1.0 + 25.0 / 10.0)).abs() < 1e-14);
        assert!((l.euclidian - 4.0 * (0.0 + 1.0 + 1.0)).abs() < 1e-14);
        assert_eq!(l.nb_photons_calc, 4.0 + 1.0 + 4.0);
        let n = l.normalised(obs.nb_obs());
        assert_eq!((n.poisson * 3.0, n.nb_photons_calc), (l.poisson, 9.0));
    }
}
