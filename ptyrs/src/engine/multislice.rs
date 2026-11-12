//! pty-chi's multislice exit wave (`PlanarPtychographyForwardModel.propagate_through_object`)
//! over the arrays and layout of [`crate::engine::ops`], with its linearisation and adjoint.
//!
//! The wave entering slice 0 is the view's probe plane. Slice `s` multiplies the wave entering
//! it by its object window, giving `psi_s`, which reaches slice `s + 1` through
//! [`Optics::gaps`]`[s]`; the last slice's `psi` is the exit wave. [`Optics::quad`] multiplies
//! the last slice only, where the wave leaves the object for [`Optics::propagator`]. One slice
//! is [`obj_probe_to_psi`] itself.

use crate::base::{C64, Object, Probe, Stack, View};

use crate::engine::ops::{accumulate_obj, accumulate_probe, obj_probe_to_psi};
use crate::engine::{Optics, shift};

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
}

impl<'a> SliceWaves<'a> {
    /// `objects` holds one object per slice of `optics`, all of one shape; view `k` reads
    /// probe plane `views[k].probe` and object mode `views[k].obj` of every slice.
    pub fn new(optics: &'a Optics, views: &[View], objects: &'a [Object], probe: &Probe) -> Self {
        assert_eq!(
            objects.len(),
            optics.nslice(),
            "one object per slice of the optics"
        );
        assert!(objects.iter().all(|o| o.dims() == objects[0].dims()));
        assert_eq!(probe.plane_shape(), optics.propagator().shape());
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
        let last = objects.len() - 1;
        obj_probe_to_psi(
            &own,
            &objects[last],
            &incident[last],
            optics.quad(),
            &mut psi,
        );
        Self {
            optics,
            objects,
            views: views.to_vec(),
            probe_dims: probe.dims(),
            own,
            incident,
            psi,
            unit,
        }
    }

    /// The exit waves, one shifted plane per view, before [`Optics::propagator`].
    pub fn psi(&self) -> &[C64] {
        &self.psi
    }

    /// The object-plane phase slice `s` applies: [`Optics::quad`] on the last, 1 elsewhere.
    pub fn quad(&self, s: usize) -> &[C64] {
        if s + 1 == self.objects.len() {
            self.optics.quad()
        } else {
            &self.unit
        }
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
            let up = out.last().unwrap();
            let (h, _) =
                accumulate_probe(&self.own, &self.objects[s + 1], dims, self.quad(s + 1), up);
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
            let q = self.quad(s);
            match &wave {
                Some(w) => obj_probe_to_psi(&self.own, o, w, q, &mut out),
                None => out.fill(C64::default()),
            }
            for (d, w) in [(&u.0[s], &wv[s]), (&v.0[s], &wu[s])] {
                obj_probe_to_psi(&self.own, d, w, q, &mut cross);
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
            obj_probe_to_psi(&self.own, o, &waves[s], self.quad(s), &mut out);
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
        obj_probe_to_psi(&self.own, d, &self.incident[s], self.quad(s), &mut out);
        out
    }

    /// Slice `s`'s object direction from its `chi_s` of [`Self::back_propagate`]: pty-chi's
    /// `delta_o_comb`, `Σ_views conj(q_s·in_s)·chi_s` scattered onto the slice
    /// ([`accumulate_obj`]).
    pub fn obj_adjoint(&self, s: usize, chi_s: &[C64]) -> Stack<C64> {
        let dims = self.objects[s].dims();
        accumulate_obj(&self.own, dims, &self.incident[s], self.quad(s), chi_s).0
    }

    /// The probe direction from slice 0's `chi_0` and `Σ|O_0|²` per plane:
    /// [`accumulate_probe`] over the views on slice 0.
    pub fn probe_adjoint(&self, chi_0: &[C64]) -> (Stack<C64>, Stack<f64>) {
        accumulate_probe(
            &self.views,
            &self.objects[0],
            self.probe_dims,
            self.quad(0),
            chi_0,
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
        for optics in &optics() {
            let waves = SliceWaves::new(optics, addr.views(), &objects, &probe);
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
        for optics in &optics() {
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
                SliceWaves::new(optics, addr.views(), &o, &p).psi().to_vec()
            };
            let (plus, minus) = (psi_at(h), psi_at(-h));
            let tangent =
                SliceWaves::new(optics, addr.views(), &objects, &probe).tangent(&d_obj, &d_probe);
            let scale = tangent.iter().map(|t| t.norm()).fold(0.0, f64::max);
            let err = plus
                .iter()
                .zip(&minus)
                .zip(&tangent)
                .map(|((p, m), t)| ((p - m) / (2.0 * h) - t).norm())
                .fold(0.0, f64::max);
            assert!(err / scale < 1e-8, "tangent error {:.3e}", err / scale);
        }
    }

    #[test]
    fn one_slice_is_obj_probe_to_psi() {
        let (addr, objects, probe) = setup();
        let optics = Optics::far_field(&far_geometry());
        let views = addr.views();
        let waves = SliceWaves::new(&optics, views, &objects[..1], &probe);
        let mut psi = vec![C64::default(); views.len() * 48];
        obj_probe_to_psi(views, &objects[0], &probe, optics.quad(), &mut psi);
        assert_eq!(waves.psi(), &psi[..]);
        let chi = noise(psi.len(), 5);
        assert_eq!(waves.back_propagate(&chi), vec![chi.clone()]);
        let (num, _) = accumulate_obj(views, objects[0].dims(), &probe, optics.quad(), &chi);
        assert_eq!(waves.obj_adjoint(0, &chi).as_slice(), num.as_slice());
    }
}
