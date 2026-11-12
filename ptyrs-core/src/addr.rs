use crate::{CoreError, Positions};

/// One exit-wave view: frame `frame`, coherent mode `mode`, built from probe mode `probe` and
/// the object window of mode `obj` whose top-left corner is `[oy, ox]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub frame: u32,
    pub mode: u32,
    pub probe: u32,
    pub obj: u32,
    pub oy: u32,
    pub ox: u32,
}

/// Address table (the PtyPy `addr` idea): every (frame, mode) view as plain indices, so
/// operators and kernels never look at positions or geometry.
///
/// Views are ordered frame-major, mode-minor; view `k` is exit-wave plane `k`
/// (`k = frame * nmode + mode`). This order is also the accumulation order of every
/// object/probe sum (invariant 8). Mode `m` uses object mode `m / nprobe` and probe mode
/// `m % nprobe` (PyNX `_obj_psidx`, `_probe_psidx`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddrTable {
    views: Vec<View>,
    nframe: usize,
    nprobe: usize,
    nobj: usize,
    frame: [usize; 2],
    obj: [usize; 2],
}

impl AddrTable {
    /// `positions` must already be preprocessed ([`Positions::pynx_preprocess`]); `pixel` is
    /// `[py, px]` in metres, `frame` the probe/frame shape, `obj` the object shape.
    pub fn new(
        positions: &Positions,
        pixel: [f64; 2],
        frame: [usize; 2],
        obj: [usize; 2],
        nprobe: usize,
        nobj: usize,
    ) -> Result<Self, CoreError> {
        if nprobe == 0 || nobj == 0 {
            return Err(CoreError::Invalid("nprobe and nobj must be >= 1".into()));
        }
        if obj[0] < frame[0] || obj[1] < frame[1] {
            return Err(CoreError::Shape(format!(
                "object {obj:?} smaller than frame {frame:?}"
            )));
        }
        let nmode = nprobe * nobj;
        let mut views = Vec::with_capacity(positions.len() * nmode);
        for (f, [dy, dx]) in positions.in_pixels(pixel).into_iter().enumerate() {
            let corner = |axis: usize, d: f64| view_corner(obj[axis], frame[axis], d);
            let (Some(oy), Some(ox)) = (corner(0, dy), corner(1, dx)) else {
                return Err(CoreError::Shape(format!(
                    "frame {f} at ({dy:.2}, {dx:.2}) object pixels leaves object {obj:?}; \
                     size it with required_obj_shape"
                )));
            };
            let (oy, ox) = (oy as u32, ox as u32);
            for m in 0..nmode {
                views.push(View {
                    frame: f as u32,
                    mode: m as u32,
                    probe: (m % nprobe) as u32,
                    obj: (m / nprobe) as u32,
                    oy,
                    ox,
                });
            }
        }
        Ok(Self {
            views,
            nframe: positions.len(),
            nprobe,
            nobj,
            frame,
            obj,
        })
    }

    pub fn views(&self) -> &[View] {
        &self.views
    }

    /// The `nmode` views of frame `f`.
    pub fn frame_views(&self, f: usize) -> &[View] {
        let n = self.nmode();
        &self.views[f * n..(f + 1) * n]
    }

    pub fn nframe(&self) -> usize {
        self.nframe
    }

    pub fn nmode(&self) -> usize {
        self.nprobe * self.nobj
    }

    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    pub fn nobj(&self) -> usize {
        self.nobj
    }

    pub fn frame_shape(&self) -> [usize; 2] {
        self.frame
    }

    pub fn obj_shape(&self) -> [usize; 2] {
        self.obj
    }
}

/// Corner of the illuminated window along one axis (PyNX `shape.get_view_coord`):
/// `(obj - frame) // 2 + shift` rounded half-to-even, or `None` when the window leaves the
/// object. `shift` is the position in object pixels relative to the object centre. PyNX
/// clamps such a window into the object, which illuminates the wrong place; an object sized
/// by [`required_obj_shape`] holds every window.
pub fn view_corner(obj: usize, frame: usize, shift: f64) -> Option<usize> {
    let span = obj.checked_sub(frame)?;
    let c = ((span / 2) as f64 + shift).round_ties_even();
    (0.0..=span as f64).contains(&c).then_some(c as usize)
}

/// Object shape covering the scan (PyNX `calc_obj_shape(..., margin, multiple=4)`), from
/// preprocessed positions; `pixel` is `[py, px]`.
pub fn required_obj_shape(
    positions: &Positions,
    pixel: [f64; 2],
    frame: [usize; 2],
    margin: usize,
) -> [usize; 2] {
    let axis = |v: &[f64], p: f64, n: usize| {
        let ext = v
            .iter()
            .map(|x| (x / p).ceil().abs() + 1.0)
            .fold(0.0, f64::max);
        let len = (2.0 * ext) as usize + n + margin;
        len.div_ceil(4) * 4
    };
    [
        axis(&positions.y, pixel[0], frame[0]),
        axis(&positions.x, pixel[1], frame[1]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_rounds_half_even_and_rejects_windows_outside() {
        // (40 - 16) // 2 = 12
        assert_eq!(view_corner(40, 16, 0.5), Some(12));
        assert_eq!(view_corner(40, 16, 1.5), Some(14));
        assert_eq!(view_corner(40, 16, -0.4), Some(12));
        // both edges: c == 0 and c + frame == obj
        assert_eq!(view_corner(40, 16, -12.5), Some(0));
        assert_eq!(view_corner(40, 16, 12.0), Some(24));
        assert_eq!(view_corner(40, 16, -12.6), None);
        assert_eq!(view_corner(40, 16, 12.6), None);
        assert_eq!(view_corner(40, 16, -20.0), None);
        assert_eq!(view_corner(40, 16, 30.0), None);
        assert_eq!(view_corner(40, 16, f64::NAN), None);
        assert_eq!(view_corner(15, 16, 0.0), None);
    }

    #[test]
    fn table_rejects_a_frame_outside_the_object() {
        // (20 - 8) // 2 = 6: a shift of 6 puts the window on the last row
        let p = Positions::new(vec![0.0, 6.0], vec![0.0, 0.0]).unwrap();
        assert!(AddrTable::new(&p, [1.0, 1.0], [8, 8], [20, 20], 1, 1).is_ok());
        let p = Positions::new(vec![0.0, 6.6], vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            AddrTable::new(&p, [1.0, 1.0], [8, 8], [20, 20], 1, 1),
            Err(CoreError::Shape(_))
        ));
    }

    #[test]
    fn required_shape_holds_every_view_and_its_neighbour() {
        let p = Positions::new(vec![-3.2, 1.0, 9.49], vec![0.0, 5.1, -7.5]).unwrap();
        for margin in [0, 1, 4, 16] {
            let obj = required_obj_shape(&p, [1.0, 1.0], [16, 16], margin);
            for (y, x) in p.y.iter().zip(&p.x) {
                for (axis, d) in [(0, *y), (1, *x)] {
                    let c = view_corner(obj[axis], 16, d).unwrap();
                    assert!(c + 16 < obj[axis], "margin {margin} axis {axis} shift {d}");
                }
            }
        }
    }

    #[test]
    fn table_orders_frame_major_and_maps_modes() {
        let p = Positions::new(vec![0.0, 2.0], vec![0.0, -3.0]).unwrap();
        let t = AddrTable::new(&p, [1.0, 1.0], [8, 8], [20, 20], 2, 1).unwrap();
        assert_eq!(t.views().len(), 4);
        let v = t.frame_views(1);
        assert_eq!((v[0].frame, v[0].mode, v[0].probe, v[0].obj), (1, 0, 0, 0));
        assert_eq!((v[1].mode, v[1].probe), (1, 1));
        assert_eq!((v[0].oy, v[0].ox), (8, 3));
    }

    #[test]
    fn obj_shape_matches_pynx_formula() {
        // ny = 2*(|ceil(-3.2)|+1) + 16 + 8 = 32, nx = 2*(ceil(5.1)+1) + 16 + 8 = 38 -> 40
        let p = Positions::new(vec![-3.2, 1.0], vec![0.0, 5.1]).unwrap();
        assert_eq!(required_obj_shape(&p, [1.0, 1.0], [16, 16], 8), [32, 40]);
    }
}
