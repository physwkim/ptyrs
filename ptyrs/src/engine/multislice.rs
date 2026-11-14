//! pty-chi's multislice exit wave (`PlanarPtychographyForwardModel.propagate_through_object`)
//! over the arrays and layout of [`crate::engine::ops`], with its linearisation and adjoint.
//!
//! The wave entering slice 0 is the view's probe plane. Slice `s` multiplies the wave entering
//! it by its object window, giving `psi_s`, which reaches slice `s + 1` through
//! [`Optics::gaps`]`[s]`; the last slice's `psi` is the exit wave. [`Optics::quad`] multiplies
//! the last slice only, where the wave leaves the object for [`Optics::exit`]. One slice
//! is [`obj_probe_to_psi`] itself.
//!
//! The probe and the plane phase are fixed in the lab frame and the slices sit on the window,
//! so a frame whose probe is moved by an offset takes the phase displaced by it. At one
//! slice the probe's move carries the phase; with more, the last slice applies the frame's
//! [`Optics::moved_quad`].

use std::borrow::Cow;

use crate::base::{C64, Object, Probe, Stack, View};

use crate::engine::ops::{accumulate_obj, accumulate_probe, obj_probe_to_psi};
use crate::engine::{MovedQuad, Optics, shift};

/// The frames' displaced plane phase on the last slice.
struct Moved {
    /// Per view, its frame's entry of `quads`.
    frame: Vec<usize>,
    quads: Vec<MovedQuad>,
}

/// The waves of `views` through the slices of an [`Optics`] at one object and probe: the wave
/// entering each slice and the exit wave.
pub struct SliceWaves<'a> {
    optics: &'a Optics,
    objects: &'a [Object],
    views: Vec<View>,
    probe_dims: [usize; 3],
    /// `views` with view `k` reading plane `k`, the layout of [`Self::incident`].
    own: Vec<View>,
    /// Per slice, the wave entering it: one natural-layout plane per view.
    incident: Vec<Probe>,
    /// The exit waves, one shifted plane per view.
    psi: Vec<C64>,
    unit: Vec<C64>,
    /// `Some` where the last slice's phase differs between the frames.
    moved: Option<Moved>,
}

impl<'a> SliceWaves<'a> {
    /// `objects` holds one object per slice of `optics`, all of one shape; view `k` reads
    /// probe plane `views[k].probe` and object mode `views[k].obj` of every slice. `offsets`
    /// holds, per frame of `views` in their order, the offset its probe planes were moved by
    /// (`None` for unmoved planes).
    pub fn new(
        optics: &'a Optics,
        views: &[View],
        objects: &'a [Object],
        probe: &Probe,
        offsets: Option<&[[f64; 2]]>,
    ) -> Self {
        assert_eq!(
            objects.len(),
            optics.nslice(),
            "one object per slice of the optics"
        );
        assert!(objects.iter().all(|o| o.dims() == objects[0].dims()));
        assert_eq!(probe.plane_shape(), optics.exit().shape());
        let [ny, nx] = probe.plane_shape();
        let n = ny * nx;
        let own: Vec<View> = views
            .iter()
            .enumerate()
            .map(|(k, v)| View {
                probe: k as u32,
                ..*v
            })
            .collect();
        let entry: Vec<C64> = views
            .iter()
            .flat_map(|v| probe.plane(v.probe as usize).iter().copied())
            .collect();
        let dims = [views.len(), ny, nx];
        let mut incident = vec![Probe::new(Stack::from_vec(dims, entry).unwrap())];
        let unit = vec![C64::new(1.0, 0.0); n];
        let mut psi = vec![C64::default(); views.len() * n];
        for (s, gap) in optics.gaps().iter().enumerate() {
            obj_probe_to_psi(&own, &objects[s], &incident[s], &unit, &mut psi);
            gap.forward(&mut psi);
            incident.push(Probe::new(reshift(&psi, dims)));
        }
        let moved = offsets.and_then(|offsets| {
            let quads: Option<Vec<MovedQuad>> =
                offsets.iter().map(|&d| optics.moved_quad(d)).collect();
            let mut frame = Vec::with_capacity(views.len());
            for (k, v) in views.iter().enumerate() {
                let next = k > 0 && v.frame != views[k - 1].frame;
                frame.push(frame.last().map_or(0, |&f| f + next as usize));
            }
            assert_eq!(frame.last().map_or(0, |f| f + 1), offsets.len());
            quads.map(|quads| Moved { frame, quads })
        });
        let mut waves = Self {
            optics,
            objects,
            views: views.to_vec(),
            probe_dims: probe.dims(),
            own,
            incident,
            psi: Vec::new(),
            unit,
            moved,
        };
        let last = objects.len() - 1;
        waves.to_psi(last, &objects[last], &waves.incident[last], &mut psi);
        waves.psi = psi;
        waves
    }

    /// `q_s·o·w` per view on slice `s`: `o` a slice's object, `w` one natural-layout plane
    /// per view.
    fn to_psi(&self, s: usize, o: &Object, w: &Probe, out: &mut [C64]) {
        obj_probe_to_psi(&self.own, o, w, self.shared(s), out);
        if let Some(m) = self.frames(s) {
            let [ny, nx] = w.plane_shape();
            for (k, plane) in out.chunks_exact_mut(ny * nx).enumerate() {
                let q = &m.quads[m.frame[k]].quad;
                (plane.iter_mut().enumerate()).for_each(|(j, p)| *p *= q[shift(j, ny, nx)]);
            }
        }
    }

    /// `chi` (one shifted plane per view) for the adjoints on slice `s`, which apply
    /// `conj(shared(s))`: times the conjugate of what [`Self::to_psi`] applies past it.
    fn dephased<'c>(&self, s: usize, chi: &'c [C64]) -> Cow<'c, [C64]> {
        let Some(m) = self.frames(s) else {
            return Cow::Borrowed(chi);
        };
        let [_, ny, nx] = self.incident[0].dims();
        let mut out = chi.to_vec();
        for (k, plane) in out.chunks_exact_mut(ny * nx).enumerate() {
            let q = &m.quads[m.frame[k]].quad;
            (plane.iter_mut().enumerate()).for_each(|(j, p)| *p *= q[shift(j, ny, nx)].conj());
        }
        Cow::Owned(out)
    }

    /// The frames' displaced phase where slice `s` applies it.
    fn frames(&self, s: usize) -> Option<&Moved> {
        self.moved.as_ref().filter(|_| s + 1 == self.objects.len())
    }

    /// The phase slice `s` applies to every view alike: 1 where it applies none or the
    /// frames' own.
    fn shared(&self, s: usize) -> &[C64] {
        match self.frames(s) {
            Some(_) => &self.unit,
            None => self.optics.slice_quad(s).unwrap_or(&self.unit),
        }
    }

    /// `w` (one shifted plane per view) times the derivative of the last slice's phase over
    /// its frame's offset components `axes` (one or two) over the phase itself: applied to
    /// [`Self::psi`], the exit waves' derivative over the offsets through the phase. `None`
    /// where the phase does not move with the offsets.
    pub fn phase_derivative(&self, axes: &[usize], w: &[C64]) -> Option<Vec<C64>> {
        let m = self.moved.as_ref()?;
        let [_, ny, nx] = self.incident[0].dims();
        let mut out = w.to_vec();
        for (k, plane) in out.chunks_exact_mut(ny * nx).enumerate() {
            let q = &m.quads[m.frame[k]];
            for (j, p) in plane.iter_mut().enumerate() {
                let i = shift(j, ny, nx);
                *p *= match *axes {
                    [a] => C64::new(0.0, q.slope[a][i]),
                    [a, c] => {
                        let curve = if a == c { q.curve[a] } else { 0.0 };
                        C64::new(-q.slope[a][i] * q.slope[c][i], curve)
                    }
                    _ => panic!("one or two offset components"),
                };
            }
        }
        Some(out)
    }

    /// The exit waves, one shifted plane per view, before [`Optics::exit`].
    pub fn psi(&self) -> &[C64] {
        &self.psi
    }

    /// The object-plane phase slice `s` applies to view `k`'s wave, natural layout: 1 where
    /// it applies none.
    pub fn quad(&self, s: usize, k: usize) -> &[C64] {
        match self.frames(s) {
            Some(m) => &m.quads[m.frame[k]].quad,
            None => self.shared(s),
        }
    }

    /// The wave entering slice `s`, one natural-layout plane per view; slice 0's is the views'
    /// probe planes.
    pub fn incident(&self, s: usize) -> &Probe {
        &self.incident[s]
    }

    /// `chi` (one shifted plane per view, a change of the exit waves) taken back through the
    /// slices: per slice `s`, its change of `psi_s`. Slice `n − 1` keeps `chi`; slice `s` takes
    /// `gaps[s]ᴴ(conj(q_{s+1}·O_{s+1})·chi_{s+1})`, pty-chi's `propagate_to_previous_slice` of
    /// its `delta_p_i`.
    pub fn back_propagate(&self, chi: &[C64]) -> Vec<Vec<C64>> {
        assert_eq!(chi.len(), self.psi.len());
        let n = self.objects.len();
        let dims = self.incident[0].dims();
        let mut out = vec![chi.to_vec()];
        for s in (0..n - 1).rev() {
            let up = self.dephased(s + 1, out.last().unwrap());
            let (h, _) = accumulate_probe(
                &self.own,
                &self.objects[s + 1],
                dims,
                self.shared(s + 1),
                &up,
            );
            let mut g = reshift(h.as_slice(), dims).into_vec();
            self.optics.gaps()[s].adjoint(&mut g);
            out.push(g);
        }
        out.reverse();
        out
    }

    /// The change of the exit waves along a change `d_objects` of every slice and `d_probe`
    /// of the probe (planes as the probe's): the linearisation of [`Self::new`] at this state.
    pub fn tangent(&self, d_objects: &[Object], d_probe: &Probe) -> Vec<C64> {
        self.tangent_waves(d_objects, d_probe).1
    }

    /// The derivatives of the exit waves along two changes `u` and `v`, each
    /// `(d_objects, d_probe)` as in [`Self::tangent`]: `[J u, J v, J″(u, v)]` with `J″(u, v)`
    /// the mixed second derivative. Slice `s` adds `q_s·(u_s·w_v,s + v_s·w_u,s)` to
    /// `q_s·O_s` times the mixed change of the wave entering it, `w_u,s` being that wave's
    /// change along `u`; at one slice `J″(u, v) = q·(u_O·v_P + v_O·u_P)`.
    pub fn second_order(&self, u: (&[Object], &Probe), v: (&[Object], &Probe)) -> [Vec<C64>; 3] {
        let (wu, ju) = self.tangent_waves(u.0, u.1);
        let (wv, jv) = self.tangent_waves(v.0, v.1);
        let dims = self.incident[0].dims();
        let mut out = vec![C64::default(); self.psi.len()];
        let mut cross = out.clone();
        // the mixed change of the wave entering slice s, zero at slice 0
        let mut wave: Option<Probe> = None;
        for (s, o) in self.objects.iter().enumerate() {
            match &wave {
                Some(w) => self.to_psi(s, o, w, &mut out),
                None => out.fill(C64::default()),
            }
            for (d, w) in [(&u.0[s], &wv[s]), (&v.0[s], &wu[s])] {
                self.to_psi(s, d, w, &mut cross);
                out.iter_mut().zip(&cross).for_each(|(o, c)| *o += c);
            }
            if let Some(gap) = self.optics.gaps().get(s) {
                gap.forward(&mut out);
                wave = Some(Probe::new(reshift(&out, dims)));
            }
        }
        [ju, jv, out]
    }

    /// Per slice, the change of the wave entering it along `(d_objects, d_probe)` (natural
    /// layout, one plane per view), and the change of the exit waves.
    fn tangent_waves(&self, d_objects: &[Object], d_probe: &Probe) -> (Vec<Probe>, Vec<C64>) {
        assert_eq!(d_objects.len(), self.objects.len());
        assert_eq!(d_probe.dims(), self.probe_dims);
        let dims = self.incident[0].dims();
        let entry: Vec<C64> = self
            .views
            .iter()
            .flat_map(|v| d_probe.plane(v.probe as usize).iter().copied())
            .collect();
        let mut waves = vec![Probe::new(Stack::from_vec(dims, entry).unwrap())];
        let mut out = vec![C64::default(); self.psi.len()];
        for (s, (o, d)) in self.objects.iter().zip(d_objects).enumerate() {
            self.to_psi(s, o, &waves[s], &mut out);
            self.obj_tangent(s, d)
                .iter()
                .zip(out.iter_mut())
                .for_each(|(t, o)| *o += t);
            if let Some(gap) = self.optics.gaps().get(s) {
                gap.forward(&mut out);
                waves.push(Probe::new(reshift(&out, dims)));
            }
        }
        (waves, out)
    }

    /// The change of `psi_s` along `d` on slice `s` with the wave entering it held: `dop` of
    /// pty-chi's step sizes.
    pub fn obj_tangent(&self, s: usize, d: &Object) -> Vec<C64> {
        let mut out = vec![C64::default(); self.psi.len()];
        self.to_psi(s, d, &self.incident[s], &mut out);
        out
    }

    /// Slice `s`'s object direction from its `chi_s` of [`Self::back_propagate`]: pty-chi's
    /// `delta_o_comb`, `Σ_views conj(q_s·in_s)·chi_s` scattered onto the slice
    /// ([`accumulate_obj`]).
    pub fn obj_adjoint(&self, s: usize, chi_s: &[C64]) -> Stack<C64> {
        let dims = self.objects[s].dims();
        let chi = self.dephased(s, chi_s);
        accumulate_obj(&self.own, dims, &self.incident[s], self.shared(s), &chi).0
    }

    /// The probe direction from slice 0's `chi_0` and `Σ|O_0|²` per plane:
    /// [`accumulate_probe`] over the views on slice 0.
    pub fn probe_adjoint(&self, chi_0: &[C64]) -> (Stack<C64>, Stack<f64>) {
        accumulate_probe(
            &self.views,
            &self.objects[0],
            self.probe_dims,
            self.shared(0),
            &self.dephased(0, chi_0),
        )
    }
}

/// `planes` (`dims`) with every plane's index `j` moved to [`shift`]`(j)`: the natural layout
/// of shifted planes and the reverse.
fn reshift(planes: &[C64], dims: [usize; 3]) -> Stack<C64> {
    let [_, ny, nx] = dims;
    let data = planes
        .chunks_exact(ny * nx)
        .flat_map(|p| (0..ny * nx).map(move |j| p[shift(j, ny, nx)]))
        .collect();
    Stack::from_vec(dims, data).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::base::{AddrTable, Geometry, Positions};

    use super::*;
    use crate::engine::linop::{dot_error, noise};
    use crate::engine::{SliceJacobian, Wavefield, register};
    use crate::worst::worst_of;

    const NSLICE: usize = 3;

    /// Three frames, two probe and two object modes, overlapping 8x6 views in 14x12 slices.
    fn setup() -> (AddrTable, Vec<Object>, Probe) {
        let pos = Positions::new(vec![0.0, 2.0, -3.0], vec![1.0, -2.0, 2.0]).unwrap();
        let addr = AddrTable::new(&pos, [1.0, 1.0], [8, 6], [14, 12], 2, 2).unwrap();
        let objects = (0..NSLICE as u64)
            .map(|s| Object::new(Stack::from_vec([2, 14, 12], noise(336, 10 + s)).unwrap()))
            .collect();
        let probe = Probe::new(Stack::from_vec([2, 8, 6], noise(96, 3)).unwrap());
        (addr, objects, probe)
    }

    /// No offsets, and the three frames'.
    const OFFSETS: [Option<&[[f64; 2]]>; 2] =
        [None, Some(&[[0.3, -0.2], [-0.45, 0.1], [0.2, 0.4]])];

    fn far_geometry() -> Geometry {
        Geometry::new(1e-10, 1.0, 1e-4, [8, 6]).unwrap()
    }

    /// Far field with its object-plane phase, and near field, each through two unequal gaps
    /// of Fresnel number near 1.
    fn optics() -> Vec<Optics> {
        let gaps = |pixel, d: [f64; 2]| d.map(|d| Wavefield::new([8, 6], 1e-10, pixel, d).unwrap());
        let g = far_geometry();
        let far = Optics::far_field(&g).with_slices(&gaps(g.pixel_size_object(), [5e-5, 1e-4]));
        let w = Wavefield::new([8, 6], 1e-10, [5e-8, 5e-8], 1e-4).unwrap();
        let near = Optics::free_space(&w).with_slices(&gaps([5e-8, 5e-8], [2e-5, 3e-5]));
        vec![far.unwrap(), near.unwrap()]
    }

    #[test]
    fn slice_jacobian_registers_as_the_adjoint_of_the_tangent() {
        let (addr, objects, probe) = setup();
        for (optics, offsets) in optics().iter().flat_map(|o| OFFSETS.map(|d| (o, d))) {
            let waves = SliceWaves::new(optics, addr.views(), &objects, &probe, offsets);
            let op = SliceJacobian {
                waves: &waves,
                obj_dims: objects[0].dims(),
                nslice: NSLICE,
                probe_dims: probe.dims(),
            };
            let err = dot_error(&op);
            assert!(err < 1e-14, "dot test error {err:.3e}");
            register(op).unwrap();
        }
    }

    #[test]
    fn tangent_is_the_derivative_of_the_exit_waves() {
        let (addr, objects, probe) = setup();
        let d_obj: Vec<Object> = (0..NSLICE as u64)
            .map(|s| Object::new(Stack::from_vec([2, 14, 12], noise(336, 20 + s)).unwrap()))
            .collect();
        let d_probe = Probe::new(Stack::from_vec([2, 8, 6], noise(96, 30)).unwrap());
        let moved = |a: &[C64], d: &[C64], t: f64| -> Vec<C64> {
            a.iter().zip(d).map(|(a, d)| a + d * t).collect()
        };
        let h = 1e-5;
        for (optics, offsets) in optics().iter().flat_map(|o| OFFSETS.map(|d| (o, d))) {
            let psi_at = |t: f64| {
                let o: Vec<Object> = objects
                    .iter()
                    .zip(&d_obj)
                    .map(|(o, d)| {
                        let v = moved(o.as_slice(), d.as_slice(), t);
                        Object::new(Stack::from_vec(o.dims(), v).unwrap())
                    })
                    .collect();
                let v = moved(probe.as_slice(), d_probe.as_slice(), t);
                let p = Probe::new(Stack::from_vec(probe.dims(), v).unwrap());
                SliceWaves::new(optics, addr.views(), &o, &p, offsets)
                    .psi()
                    .to_vec()
            };
            let (plus, minus) = (psi_at(h), psi_at(-h));
            let tangent = SliceWaves::new(optics, addr.views(), &objects, &probe, offsets)
                .tangent(&d_obj, &d_probe);
            let scale = tangent.iter().map(|t| t.norm()).fold(0.0, f64::max);
            let err = worst_of(
                plus.iter()
                    .zip(&minus)
                    .zip(&tangent)
                    .map(|((p, m), t)| ((p - m) / (2.0 * h) - t).norm()),
            );
            assert!(err / scale < 1e-8, "tangent error {:.3e}", err / scale);
        }
    }

    /// [`SliceWaves::quad`] is the phase the forward model applies: on every slice, `psi_s`
    /// of every view is its object patch times the wave entering the slice times
    /// `quad(s, k)`, with the frames moved and not, through three slices and through two,
    /// where LSQML's position slice `nslice / 2` is the last.
    #[test]
    fn quad_is_the_phase_of_every_slices_psi() {
        let (addr, objects, probe) = setup();
        let (views, [ny, nx], nxo) = (addr.views(), [8, 6], 12);
        let gap = |pixel| [Wavefield::new([ny, nx], 1e-10, pixel, 5e-5).unwrap()];
        let g = far_geometry();
        let far = Optics::far_field(&g).with_slices(&gap(g.pixel_size_object()));
        let w = Wavefield::new([ny, nx], 1e-10, [5e-8, 5e-8], 1e-3).unwrap();
        let fresnel = Optics::free_space(&w).with_slices(&gap([5e-8, 5e-8]));
        let mut all = optics();
        all.extend([far.unwrap(), fresnel.unwrap()]);
        assert_eq!(all.iter().filter(|o| o.nslice() == 2).count(), 2);
        let mut displaced = 0;
        for (optics, offsets) in all.iter().flat_map(|o| OFFSETS.map(|d| (o, d))) {
            let objects = &objects[..optics.nslice()];
            let last = objects.len() - 1;
            let waves = SliceWaves::new(optics, views, objects, &probe, offsets);
            for (s, object) in objects.iter().enumerate() {
                let psi = waves.obj_tangent(s, object);
                if s == last {
                    assert_eq!(psi, waves.psi());
                }
                for (k, (v, psi)) in views.iter().zip(psi.chunks_exact(ny * nx)).enumerate() {
                    let (o, q) = (object.plane(v.obj as usize), waves.quad(s, k));
                    let w = waves.incident(s).plane(k);
                    displaced += (s == last && q != optics.quad()) as usize;
                    for (j, got) in psi.iter().enumerate() {
                        let i = shift(j, ny, nx);
                        let patch = o[(v.oy as usize + i / nx) * nxo + v.ox as usize + i % nx];
                        let want = patch * w[i] * q[i];
                        assert!(
                            (got - want).norm() <= 1e-15 * want.norm(),
                            "slice {s} view {k} pixel {j}: {got} against {want}"
                        );
                    }
                }
            }
        }
        // Every view of the three exits with a plane phase, under the offsets.
        assert_eq!(displaced, 3 * views.len());
    }

    #[test]
    fn one_slice_is_obj_probe_to_psi() {
        let (addr, objects, probe) = setup();
        let optics = Optics::far_field(&far_geometry());
        let views = addr.views();
        let waves = SliceWaves::new(&optics, views, &objects[..1], &probe, None);
        let mut psi = vec![C64::default(); views.len() * 48];
        obj_probe_to_psi(views, &objects[0], &probe, optics.quad(), &mut psi);
        assert_eq!(waves.psi(), &psi[..]);
        let chi = noise(psi.len(), 5);
        assert_eq!(waves.back_propagate(&chi), vec![chi.clone()]);
        let (num, _) = accumulate_obj(views, objects[0].dims(), &probe, optics.quad(), &chi);
        assert_eq!(waves.obj_adjoint(0, &chi).as_slice(), num.as_slice());
    }

    /// A two-slice sample displaced by a fraction of a pixel under a fixed probe and plane
    /// phase is what an offset stands for: the probe moved ahead of the first slice and the
    /// phase displaced on the last ([`Optics::moved_quad`]) give its intensities under a
    /// Fresnel transform either way in distance and in the far field; the phase left
    /// unmoved does not.
    #[test]
    fn a_moved_probe_is_the_displaced_sample_through_two_slices() {
        use std::f64::consts::PI;

        use crate::engine::Fft2;
        use crate::engine::features::move_planes;

        const N: usize = 64;
        const OFFSET: [f64; 2] = [0.3, -0.45];
        let centred = |i: usize| [i / N, i % N].map(|v| v as f64 - (N / 2) as f64);
        // Band-limited slices of period 256, so they have a value between their pixels.
        let slices = |at: [f64; 2]| -> Vec<Object> {
            (0..2)
                .map(|s| {
                    let terms: Vec<(C64, [f64; 2])> = noise(24, 3 + s)
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|[c, k]| (0.15 * c, [k.re, k.im].map(|k| (12.0 * k).round())))
                        .collect();
                    let v = (0..N * N)
                        .map(|i| {
                            let [y, x] = centred(i);
                            let wave = |&(c, k): &(C64, [f64; 2])| {
                                let t = k[0] * (y + at[0]) + k[1] * (x + at[1]);
                                c * C64::from_polar(1.0, 2.0 * PI * t / 256.0)
                            };
                            C64::new(1.0, 0.0) + terms.iter().map(wave).sum::<C64>()
                        })
                        .collect();
                    Object::new(Stack::from_vec([1, N, N], v).unwrap())
                })
                .collect()
        };
        let probe: Vec<C64> = (0..N * N)
            .map(|i| {
                let [y, x] = centred(i);
                let r2 = y * y + x * x;
                C64::from_polar(
                    (-r2 / 49.0).exp(),
                    0.04 * r2 + 0.5 * (y / 3.0).sin() - 0.3 * (x / 4.0).cos(),
                )
            })
            .collect();
        let view = [View {
            frame: 0,
            mode: 0,
            probe: 0,
            obj: 0,
            oy: 0,
            ox: 0,
        }];
        let pixel = [50e-9; 2];
        let near = |d| Optics::free_space(&Wavefield::new([N, N], 1e-10, pixel, d).unwrap());
        let far = Geometry::new(1e-10, 5e-3, 1e-10 * 5e-3 / (N as f64 * pixel[0]), [N, N]);
        let far = Optics::far_field(&far.unwrap());
        assert!((far.pixel()[0] / pixel[0] - 1.0).abs() < 1e-12);
        let gap = [Wavefield::new([N, N], 1e-10, pixel, 2e-4).unwrap()];
        for (name, optics) in [("Fresnel", near(5e-3)), ("back", near(-8e-3)), ("far", far)] {
            assert!(
                optics.quad().iter().any(|q| (q - 1.0).norm() > 0.5),
                "{name}"
            );
            let optics = optics.with_slices(&gap).unwrap();
            let intensity = |objects: &[Object], probe: &[C64], offsets: Option<&[[f64; 2]]>| {
                let probe = Probe::new(Stack::from_vec([1, N, N], probe.to_vec()).unwrap());
                let waves = SliceWaves::new(&optics, &view, objects, &probe, offsets);
                let mut psi = waves.psi().to_vec();
                optics.exit().forward(&mut psi);
                psi.iter().map(|c| c.norm_sqr()).collect::<Vec<f64>>()
            };
            let truth = intensity(&slices(OFFSET), &probe, None);
            let mut moved = probe.clone();
            move_planes(&mut moved, optics.slice_quad(0), OFFSET, &Fft2::new([N, N]));
            let window = slices([0.0; 2]);
            let err = |offsets: Option<&[[f64; 2]]>| {
                let model = intensity(&window, &moved, offsets);
                let d = model.iter().zip(&truth).map(|(a, b)| (a - b).abs());
                worst_of(d) / truth.iter().copied().fold(0.0, f64::max)
            };
            let (with, without) = (err(Some(&[OFFSET])), err(None));
            assert!(with < 1e-6, "{name}: {with:.3e}");
            assert!(without > 1e-3, "{name}: {without:.3e}");
        }
    }
}
