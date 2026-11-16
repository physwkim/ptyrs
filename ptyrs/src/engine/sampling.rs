//! How a frame's [`crate::engine::Features::position`] offset enters the exit waves: pty-chi's
//! two object-patch models, `ObjectPosOriginCoordsMethods`' `PATCH` and `FULL_OBJECT`.

use std::f64::consts::PI;

use crate::base::{C64, Object, Probe, Stack, View};
use crate::engine::features::freq;
use crate::engine::lsqml::illumination;
use crate::engine::{Fft2, Optics};

/// The sub-pixel position model. Without offsets both read every view's object window at its
/// integer corner, and only the offsets tell them apart.
///
/// Under [`Self::Patch`] the window stays on whole object pixels and the probe, with the plane
/// phase it carries, is Fourier-moved by the offset on the frame's periodic grid. In the near
/// field a moved probe displaces the hologram with it, which is pty-chi `PATCH`'s model but
/// not the sample's; [`Self::auto`] takes [`Self::FullObject`] only where the probe spans most
/// of the object, and setting it explicitly is the remedy elsewhere.
///
/// Under [`Self::FullObject`] the probe stays put and each view samples its object at the
/// corner plus the offset by Fourier interpolation of the whole object plane:
/// `patch[u] = DFT⁻¹(DFT(O)·exp(+2πi f·(corner + d)))[u]` over `u` in the window, `f` numpy's
/// `fftfreq` of the object plane, a sample past the plane's edge wrapping. Its adjoint embeds
/// a patch at the plane's origin, applies the conjugate phase in the spectrum and sums over
/// the views before one inverse transform per plane.
///
/// The transforms run at the object's exact shape: a side with a prime factor above 7 takes
/// the slow prime-size FFT (a GPU DM cycle at 404² costs 1.7 times one at 420²), so a caller
/// who picks the shape should take sides whose prime factors are all at most 7. Padding the
/// transform instead would turn the wrap above into a zero seam and change the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sampling {
    #[default]
    Patch,
    FullObject,
}

impl Sampling {
    /// pty-chi's `AUTO` rule: [`Self::FullObject`] in the near field (a propagator other than
    /// [`crate::engine::Propagator::Fourier`]) when `5·frame > 4·object` on both axes, else
    /// [`Self::Patch`].
    pub fn auto(optics: &Optics, object: [usize; 2], frame: [usize; 2]) -> Self {
        let large = (0..2).all(|a| 5 * frame[a] > 4 * object[a]);
        if optics.near_field() && large {
            Self::FullObject
        } else {
            Self::Patch
        }
    }

    /// pty-chi's illumination map over `views` (`Σ |P[probe_k]|²` on each view's window,
    /// shaped `obj_dims`) under this model: [`illumination`] without offsets or under
    /// [`Self::Patch`], whose probe move leaves it unmoved; under [`Self::FullObject`] each
    /// view's `|P|²` placed at its corner plus `offsets[frame]` by the sampling's adjoint, its
    /// real part floored at 0.
    pub fn illumination(
        self,
        views: &[View],
        obj_dims: [usize; 3],
        probe: &Probe,
        offsets: Option<&[[f64; 2]]>,
    ) -> Stack<f64> {
        let (Self::FullObject, Some(offsets)) = (self, offsets) else {
            return illumination(views, obj_dims, probe);
        };
        let [_, nyo, nxo] = obj_dims;
        let d = views.iter().map(|v| offsets[v.frame as usize]);
        let sampler = Sampler::new([nyo, nxo], probe.plane_shape(), views, d);
        let patches: Vec<f64> = (views.iter())
            .flat_map(|v| probe.plane(v.probe as usize).iter().map(|p| p.norm_sqr()))
            .collect();
        sampler.place_norm(&patches, obj_dims[0])
    }
}

/// [`Sampling::FullObject`]'s sampling of a set of views: per view its object plane and
/// position `corner + offset`.
#[derive(Clone)]
pub(crate) struct Sampler {
    /// The object plane's transform.
    fft: Fft2,
    /// `[py, px]`.
    window: [usize; 2],
    points: Vec<(usize, [f64; 2])>,
}

impl Sampler {
    /// The views `views` of `[nyo, nxo]` object planes through `[py, px]` windows, view `k` at
    /// its corner plus the `k`th of `offsets`.
    pub(crate) fn new(
        obj: [usize; 2],
        window: [usize; 2],
        views: &[View],
        offsets: impl IntoIterator<Item = [f64; 2]>,
    ) -> Self {
        let points: Vec<_> = (views.iter().zip(offsets))
            .map(|(v, d)| (v.obj as usize, [v.oy as f64 + d[0], v.ox as f64 + d[1]]))
            .collect();
        assert_eq!(points.len(), views.len(), "one offset per view");
        Self {
            fft: Fft2::new(obj),
            window,
            points,
        }
    }

    /// The window's dims over every view, `[nview, py, px]`.
    pub(crate) fn dims(&self) -> [usize; 3] {
        [self.points.len(), self.window[0], self.window[1]]
    }

    /// The sampling phase of spectrum index `j` at position `t`, differentiated over `t`'s
    /// components `axes`: `exp(+2πi f·t)·Π_a 2πi f_a`.
    fn phase(&self, j: usize, t: [f64; 2], axes: &[usize]) -> C64 {
        let [ny, nx] = self.fft.shape();
        let f = [freq(j / nx, ny), freq(j % nx, nx)];
        let e = C64::from_polar(1.0, 2.0 * PI * (f[0] * t[0] + f[1] * t[1]));
        axes.iter()
            .fold(e, |e, &a| e * C64::new(0.0, 2.0 * PI * f[a]))
    }

    /// Every view's window of `object` (natural layout, one plane per view) at its position,
    /// or with `axes` (`0` for `y`, `1` for `x`, repeats for higher orders) its derivative
    /// over the position.
    pub(crate) fn windows(&self, object: &Object, axes: &[usize]) -> Object {
        let [nyo, nxo] = self.fft.shape();
        let n = nyo * nxo;
        let [py, px] = self.window;
        let mut spectra = object.as_slice().to_vec();
        self.fft.forward(&mut spectra);
        let mut out = Vec::with_capacity(self.points.len() * py * px);
        let mut plane = vec![C64::default(); n];
        // Views at one point (a frame's modes on one object mode) share one transform.
        for group in self.points.chunk_by(|a, b| a == b) {
            let (obj, t) = group[0];
            for (j, (p, s)) in plane.iter_mut().zip(&spectra[obj * n..][..n]).enumerate() {
                *p = s * self.phase(j, t, axes);
            }
            self.fft.inverse(&mut plane);
            for _ in group {
                (0..py).for_each(|y| out.extend_from_slice(&plane[y * nxo..][..px]));
            }
        }
        Object::new(Stack::from_vec(self.dims(), out).unwrap())
    }

    /// The adjoint of [`Self::windows`] without derivative: `patches` (one natural-layout
    /// window per view) placed onto `nobj` object planes.
    pub(crate) fn place(&self, patches: &[C64], nobj: usize) -> Stack<C64> {
        self.convolve(patches, nobj, |t, m| {
            (m.iter_mut().enumerate()).for_each(|(j, m)| *m = self.phase(j, t, &[]).conj())
        })
    }

    /// Per view point, `patches` summed at the plane's origin and circularly convolved by the
    /// kernel whose spectrum `kernel(t, spectrum)` writes, summed onto `nobj` object planes.
    fn convolve(
        &self,
        patches: &[C64],
        nobj: usize,
        kernel: impl Fn([f64; 2], &mut [C64]),
    ) -> Stack<C64> {
        let [nyo, nxo] = self.fft.shape();
        let n = nyo * nxo;
        let [py, px] = self.window;
        assert_eq!(patches.len(), self.points.len() * py * px);
        let mut out = Stack::filled([nobj, nyo, nxo], C64::default());
        let mut plane = vec![C64::default(); n];
        let mut spectrum = vec![C64::default(); n];
        let mut start = 0;
        for group in self.points.chunk_by(|a, b| a == b) {
            let (obj, t) = group[0];
            plane.fill(C64::default());
            let end = start + group.len() * py * px;
            for patch in patches[start..end].chunks_exact(py * px) {
                for (y, row) in patch.chunks_exact(px).enumerate() {
                    (plane[y * nxo..][..px].iter_mut().zip(row)).for_each(|(p, r)| *p += r);
                }
            }
            start = end;
            self.fft.forward(&mut plane);
            kernel(t, &mut spectrum);
            for ((o, p), m) in out.plane_mut(obj).iter_mut().zip(&plane).zip(&spectrum) {
                *o += p * m;
            }
        }
        self.fft.inverse(out.as_mut_slice());
        out
    }

    /// [`Self::place`] of the real `patches`, its real part floored at 0: a placed sum of
    /// squared moduli, pty-chi's illumination map.
    pub(crate) fn place_norm(&self, patches: &[f64], nobj: usize) -> Stack<f64> {
        let patches: Vec<C64> = patches.iter().map(|&p| C64::new(p, 0.0)).collect();
        floored(self.place(&patches, nobj))
    }

    /// The diagonal of `AᴴA`, `A` the views' windows each weighted by a `W` with `|W|²` the
    /// real `patches` (natural layout), its real part floored at 0: per object pixel `i`,
    /// `Σ_views Σ_u |W[u]|²·|h_t[u − i]|²` with `h_t` the kernel that samples a window at `t`.
    /// The spectrum of `|h_t[−·]|²` is [`square_spectrum`]'s per axis, which at a whole-pixel
    /// `t` is [`Self::place`]'s conjugate phase, where this is [`Self::place_norm`].
    pub(crate) fn diag(&self, patches: &[f64], nobj: usize) -> Stack<f64> {
        let nxo = self.fft.shape()[1];
        let patches: Vec<C64> = patches.iter().map(|&p| C64::new(p, 0.0)).collect();
        floored(self.convolve(&patches, nobj, |t, m| {
            let [my, mx] = [0, 1].map(|a| square_spectrum(t[a], self.fft.shape()[a]));
            (m.iter_mut().enumerate()).for_each(|(j, m)| *m = my[j / nxo] * mx[j % nxo])
        }))
    }
}

/// The steps of [`Sampling::FullObject`]'s object solve in AP, DM and RAAR.
pub(crate) const OBJ_PCG: usize = 10;

/// The object solve holds a pixel whose `D + reg` is at most this times `max D`: one no
/// window samples, whose `D` and residual are the transforms' rounding, in `f32` up to
/// `1e-6·max D`, which a floor on `D + reg` would divide into the steps, and one the probe's
/// tails light as dimly inside a window. As `reg = inertia·max D`, it holds none once the
/// object inertia exceeds this.
pub(crate) const OBJ_HELD: f64 = 1e-5;

/// The spectrum over `n` points of `|h_t[−v]|²`, `h_t[v] = (1/n)·Σ_k exp(2πi k (t + v)/n)`
/// over numpy's `fftfreq` bins `k` the one-axis kernel that samples at `t`: at index
/// `m < n`, `(1 − m/n)·exp(−2πi t m/n) + (m/n)·exp(−2πi t (m − n)/n)`, the bin pairs
/// `k − k'` of `m` and `m − n`.
fn square_spectrum(t: f64, n: usize) -> Vec<C64> {
    (0..n)
        .map(|m| {
            let a = m as f64 / n as f64;
            C64::from_polar(1.0 - a, -2.0 * PI * t * a)
                + C64::from_polar(a, -2.0 * PI * t * (a - 1.0))
        })
        .collect()
}

/// The real part of `placed`, floored at 0.
fn floored(placed: Stack<C64>) -> Stack<f64> {
    let data = placed.as_slice().iter().map(|c| c.re.max(0.0)).collect();
    Stack::from_vec(placed.dims(), data).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{AddrTable, Geometry, Positions};
    use crate::engine::Wavefield;
    use crate::engine::linop::noise;
    use crate::worst::worst_of;

    /// Odd rows: index 5 of 11 is frequency +5/11, which one term of each mode reads. Even
    /// columns: index 5 of 10 is the Nyquist, −1/2, which object mode 1's `k_x = −5` term reads
    /// as `exp(−iπx)`; its sign shows only off whole pixels and in its derivatives.
    const OBJ: [usize; 2] = [11, 10];
    const WINDOW: [usize; 2] = [6, 5];

    /// Two probe modes at one point of object mode 0, then object mode 1 past the plane's
    /// corner and at a whole pixel.
    fn views() -> Vec<View> {
        let view = |probe, obj, oy, ox| View {
            frame: 0,
            mode: 0,
            probe,
            obj,
            oy,
            ox,
        };
        vec![
            view(0, 0, 1, 2),
            view(1, 0, 1, 2),
            view(0, 1, 8, 7),
            view(1, 1, 3, 0),
        ]
    }

    const OFFSETS: [[f64; 2]; 4] = [[0.3, -0.2], [0.3, -0.2], [-0.45, 0.35], [0.0, 0.0]];

    /// Per object mode, the terms `(c, k)` of a trigonometric polynomial on the planes'
    /// frequencies.
    fn terms(mode: usize) -> &'static [(C64, [f64; 2])] {
        const TERMS: [&[(C64, [f64; 2])]; 2] = [
            &[
                (C64::new(1.0, 0.0), [0.0, 0.0]),
                (C64::new(0.3, -0.2), [1.0, -2.0]),
                (C64::new(-0.25, 0.4), [-5.0, 4.0]),
                (C64::new(0.1, 0.15), [3.0, 1.0]),
            ],
            &[
                (C64::new(0.7, 0.2), [0.0, 1.0]),
                (C64::new(-0.2, 0.35), [-2.0, 3.0]),
                (C64::new(0.45, 0.05), [5.0, -4.0]),
                (C64::new(-0.3, 0.25), [3.0, -5.0]),
            ],
        ];
        TERMS[mode]
    }

    /// Object mode `mode`'s polynomial at `t`, differentiated over `t`'s components `axes`.
    fn value(mode: usize, t: [f64; 2], axes: &[usize]) -> C64 {
        let n = OBJ.map(|n| n as f64);
        (terms(mode).iter())
            .map(|&(c, k)| {
                let w = [0, 1].map(|a| 2.0 * PI * k[a] / n[a]);
                let d: C64 = axes.iter().map(|&a| C64::new(0.0, w[a])).product();
                c * d * C64::from_polar(1.0, w[0] * t[0] + w[1] * t[1])
            })
            .sum()
    }

    /// The windows and their derivatives of a band-limited object are its interpolant's,
    /// past the plane's edge too.
    #[test]
    fn windows_are_the_interpolant_and_its_derivatives() {
        let [nyo, nxo] = OBJ;
        let [py, px] = WINDOW;
        let data = (0..2 * nyo * nxo)
            .map(|i| {
                let j = i % (nyo * nxo);
                value(i / (nyo * nxo), [(j / nxo) as f64, (j % nxo) as f64], &[])
            })
            .collect();
        let object = Object::new(Stack::from_vec([2, nyo, nxo], data).unwrap());
        let views = views();
        let sm = Sampler::new(OBJ, WINDOW, &views, OFFSETS);
        assert_eq!(sm.dims(), [4, py, px]);
        for axes in [&[][..], &[0], &[1], &[0, 1], &[0, 0], &[1, 1]] {
            let got = sm.windows(&object, axes);
            let mut scale: f64 = 0.0;
            let err = worst_of(
                (views.iter().zip(OFFSETS).enumerate())
                    .flat_map(|(k, (v, d))| {
                        let plane = got.plane(k).to_vec();
                        (0..py * px).map(move |u| {
                            let t = [
                                v.oy as f64 + d[0] + (u / px) as f64,
                                v.ox as f64 + d[1] + (u % px) as f64,
                            ];
                            (plane[u], value(v.obj as usize, t, axes))
                        })
                    })
                    .map(|(g, w)| {
                        scale = scale.max(w.norm());
                        (g - w).norm()
                    }),
            );
            assert!(
                err < 1e-13 * scale,
                "axes {axes:?}: {err:.3e} of {scale:.3e}"
            );
        }
    }

    #[test]
    fn place_is_the_adjoint_of_windows() {
        let sm = Sampler::new(OBJ, WINDOW, &views(), OFFSETS);
        let n = OBJ[0] * OBJ[1];
        let object = Object::new(Stack::from_vec([2, OBJ[0], OBJ[1]], noise(2 * n, 1)).unwrap());
        let patches = noise(4 * WINDOW[0] * WINDOW[1], 2);
        let dot = |a: &[C64], b: &[C64]| a.iter().zip(b).map(|(a, b)| a.conj() * b).sum::<C64>();
        let lhs = dot(sm.windows(&object, &[]).as_slice(), &patches);
        let rhs = dot(object.as_slice(), sm.place(&patches, 2).as_slice());
        assert!(
            (lhs - rhs).norm() < 1e-14 * lhs.norm(),
            "{lhs} against {rhs}"
        );
    }

    /// `diag` against `Σ_views Σ_u |W[u]|²·|windows(e_i)[u]|²` per unit object `e_i`, at
    /// fractional offsets and at whole-pixel ones, where it is also `place_norm`.
    #[test]
    fn diag_is_the_diagonal_of_the_normal_operator() {
        let [nyo, nxo] = OBJ;
        let n = nyo * nxo;
        let views = views();
        let power: Vec<f64> = (noise(4 * WINDOW[0] * WINDOW[1], 4).iter())
            .map(|c| c.norm_sqr())
            .collect();
        let whole = OFFSETS.map(|d| d.map(f64::round));
        for offsets in [OFFSETS, whole] {
            let sm = Sampler::new(OBJ, WINDOW, &views, offsets);
            let got = sm.diag(&power, 2);
            let mut unit = Object::new(Stack::filled([2, nyo, nxo], C64::default()));
            let want: Vec<f64> = (0..2 * n)
                .map(|i| {
                    unit.as_mut_slice()[i] = C64::new(1.0, 0.0);
                    let w = sm.windows(&unit, &[]);
                    unit.as_mut_slice()[i] = C64::default();
                    let o = w.as_slice().iter().zip(&power);
                    o.map(|(w, p)| p * w.norm_sqr()).sum()
                })
                .collect();
            let scale = want.iter().copied().fold(0.0, f64::max);
            let err = worst_of((got.as_slice().iter().zip(&want)).map(|(g, w)| (g - w).abs()));
            assert!(err < 1e-14 * scale, "{offsets:?}: {err:.3e} of {scale:.3e}");
            if offsets == whole {
                let norm = sm.place_norm(&power, 2);
                let it = got.as_slice().iter().zip(norm.as_slice());
                let err = worst_of(it.map(|(g, w)| (g - w).abs()));
                assert!(err < 1e-14 * scale, "place_norm: {err:.3e} of {scale:.3e}");
            }
        }
    }

    /// At zero offsets the placed `|P|²` is the illumination map summed on the windows.
    #[test]
    fn full_object_illumination_without_offsets_is_the_patch_one() {
        let pos = Positions::new(vec![0.0, 2.0, -3.0], vec![1.0, -2.0, 2.0]).unwrap();
        let addr = AddrTable::new(&pos, [1.0, 1.0], [8, 6], [14, 12], 2, 2).unwrap();
        let probe = Probe::new(Stack::from_vec([2, 8, 6], noise(96, 3)).unwrap());
        let dims = [2, 14, 12];
        let want = illumination(addr.views(), dims, &probe);
        let zero = [[0.0; 2]; 3];
        let got = Sampling::FullObject.illumination(addr.views(), dims, &probe, Some(&zero));
        let scale = want.as_slice().iter().copied().fold(0.0, f64::max);
        let err =
            worst_of((got.as_slice().iter().zip(want.as_slice())).map(|(g, w)| (g - w).abs()));
        assert!(err < 1e-14 * scale, "{err:.3e} of {scale:.3e}");
        assert_eq!(
            Sampling::Patch
                .illumination(addr.views(), dims, &probe, Some(&zero))
                .as_slice(),
            want.as_slice()
        );
    }

    #[test]
    fn auto_takes_full_object_for_a_near_field_probe_over_most_of_the_object() {
        let far = Optics::far_field(&Geometry::new(1e-10, 1.0, 1e-4, [8, 6]).unwrap());
        let near = Optics::free_space(&Wavefield::new([8, 6], 1e-10, [5e-8; 2], 1e-4).unwrap());
        assert_eq!(Sampling::auto(&far, [9, 7], [8, 6]), Sampling::Patch);
        assert_eq!(Sampling::auto(&near, [9, 7], [8, 6]), Sampling::FullObject);
        // `5·8 = 4·10` and `5·6 < 4·8`.
        assert_eq!(Sampling::auto(&near, [10, 7], [8, 6]), Sampling::Patch);
        assert_eq!(Sampling::auto(&near, [9, 8], [8, 6]), Sampling::Patch);
    }
}
