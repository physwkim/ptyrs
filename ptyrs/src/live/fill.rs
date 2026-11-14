//! Predicted first values for object pixels (DESIGN.md §3 B6, §5 inv. 7).
//!
//! A [`PatchPredictor`] turns frames' counts into one complex patch per
//! frame, the centre of that frame's window on the object; one call takes
//! the frames of every batch queued for it. [`PatchFill`] stitches a call's
//! patches into a [`Fill`]: the values of the pixels no earlier call issued
//! a prediction for, as rectangles in one pinned buffer, which
//! [`crate::live::object::LiveObject::write_fill`] copies onto the object.
//! The prediction is a first value only: the epochs that follow update it
//! like any other pixel.
//!
//! Rule: a pixel takes its first prediction when it lands, replacing
//! whatever earlier epochs left there, including the epochs its own batch
//! ran while waiting for the call. It takes one, from the first call whose
//! patches cover it; `PatchFill` alone records which pixels those are.

use std::f64::consts::PI;
use std::sync::{Arc, Mutex, PoisonError};

use cuda_core::{Context, DriverError, PinnedHostBuffer};

/// Predicts the object under the centre of each frame of a call. It lives
/// outside the engine: only arrays cross (counts in, patches out).
pub trait PatchPredictor: Send {
    /// `[py, px]`: the patch is the centre of a frame's window.
    fn patch(&self) -> [usize; 2];

    /// `counts` is `n` frames of `frame = [ny, nx]`, detector order, masked
    /// pixels as the sentinel, in landing order; `out` is `n * py * px`
    /// `[re, im]` values of object mode 0, patch for patch in the same
    /// order.
    fn predict(
        &mut self,
        counts: &[u16],
        frame: [usize; 2],
        out: &mut [[f32; 2]],
    ) -> Result<(), String>;
}

/// The object pixel of the corner of the patch of a frame whose window
/// starts at `view` (`[oy, ox]`). It saturates at `u32::MAX`, where a view
/// placed outside every object stays, so [`PatchFill::stitch`] clips its
/// patch whole.
pub fn patch_corner(view: [u32; 2], frame: [usize; 2], patch: [usize; 2]) -> [u32; 2] {
    [0, 1].map(|a| view[a].saturating_add(((frame[a] - patch[a]) / 2) as u32))
}

/// Object pixels `y..y + h`, `x..x + w`, whose values are the `h * w`
/// entries from `at` of [`Fill::values`], row-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub y: u32,
    pub x: u32,
    pub h: u32,
    pub w: u32,
    pub at: usize,
}

struct Buffers {
    values: PinnedHostBuffer<[f32; 2]>,
    rects: Vec<Rect>,
}

type Pool = Mutex<Vec<Buffers>>;

/// Buffers out of the pool, back in it when the last [`Fill`] of them drops.
struct Lent {
    buffers: Option<Buffers>,
    pool: Arc<Pool>,
}

impl Drop for Lent {
    fn drop(&mut self) {
        if let Some(b) = self.buffers.take() {
            self.pool
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(b);
        }
    }
}

/// One call's predictions for the pixels that had none: disjoint
/// rectangles over one pinned buffer. Clones share the buffer, one for each
/// object the fill is written to; it returns to its [`PatchFill`] when the
/// last of them drops, which an object that queued copies of it delays
/// until the copies ran.
#[derive(Clone)]
pub struct Fill(Arc<Lent>);

impl Fill {
    fn buffers(&self) -> &Buffers {
        // Taken only by `Lent::drop`.
        self.0.buffers.as_ref().expect("a live fill has buffers")
    }

    pub fn rects(&self) -> &[Rect] {
        &self.buffers().rects
    }

    /// The values the rectangles index.
    pub fn values(&self) -> &[[f32; 2]] {
        self.buffers().values.as_slice()
    }

    /// Pixels this fill has values for.
    pub fn pixels(&self) -> usize {
        self.rects().iter().map(|r| (r.h * r.w) as usize).sum()
    }
}

/// Stitches patches into fills and owns the record of which object pixels a
/// prediction was issued for.
pub struct PatchFill {
    /// `[nyo, nxo]`.
    shape: [usize; 2],
    patch: [usize; 2],
    max_batch: usize,
    /// One bit per pixel of mode 0: a prediction was issued for it. Set by
    /// [`Self::stitch`], never cleared, whether or not the fill reaches the
    /// object.
    issued: Vec<u64>,
    /// The `sin²` taper along each axis of a patch.
    taper: [Vec<f64>; 2],
    /// `[re, im, weight]` sums of one rectangle.
    acc: Vec<[f64; 3]>,
    /// Scratch: the columns the patches cover on one row, and the
    /// rectangles the previous and the current row extended.
    spans: Vec<[u32; 2]>,
    open: [Vec<usize>; 2],
    pool: Arc<Pool>,
    no_buffer: u64,
}

impl PatchFill {
    /// For an object of `shape`, `patch`-sized predictions and calls of up
    /// to `max_batch` frames, with `pool` fills in flight at most: one
    /// is in flight from [`Self::stitch`] until the objects it was written
    /// to ready tiles or take a fill after its copies ran.
    /// Everything [`Self::stitch`] needs is allocated here, pinned in `ctx`.
    pub fn new(
        ctx: &Arc<Context>,
        shape: [usize; 2],
        patch: [usize; 2],
        max_batch: usize,
        pool: usize,
    ) -> Result<Self, DriverError> {
        let most = max_batch * patch[0] * patch[1];
        let buffers = (0..pool)
            .map(|_| {
                Ok(Buffers {
                    values: PinnedHostBuffer::zeroed(ctx, most)?,
                    // Issued blocks on a row are a patch wide or end at the
                    // object's edge, so a row has at most one open run per
                    // patch on it: `max_batch * patch[0]` rectangles at most.
                    rects: Vec::with_capacity(4 * max_batch * patch[0]),
                })
            })
            .collect::<Result<Vec<_>, DriverError>>()?;
        let taper = |n: usize| -> Vec<f64> {
            (0..n)
                .map(|i| (PI * (i as f64 + 0.5) / n as f64).sin().powi(2))
                .collect()
        };
        Ok(Self {
            shape,
            patch,
            max_batch,
            issued: vec![0; (shape[0] * shape[1]).div_ceil(64)],
            taper: patch.map(taper),
            acc: vec![[0.0; 3]; most],
            spans: Vec::with_capacity(max_batch),
            open: [(); 2].map(|_| Vec::with_capacity(4 * max_batch)),
            pool: Arc::new(Mutex::new(buffers)),
            no_buffer: 0,
        })
    }

    /// `[py, px]`.
    pub fn patch(&self) -> [usize; 2] {
        self.patch
    }

    /// Calls [`Self::stitch`] turned away because every buffer was in
    /// flight.
    pub fn no_buffer(&self) -> u64 {
        self.no_buffer
    }

    /// Whether a prediction was issued for object pixel `[y, x]`.
    pub fn issued(&self, [y, x]: [usize; 2]) -> bool {
        let i = y * self.shape[1] + x;
        self.issued[i / 64] >> (i % 64) & 1 == 1
    }

    /// The fill of one call: `patches` holds a patch for each of
    /// `corners`, the object pixel of its corner ([`patch_corner`]), in
    /// landing order. The pixels the patches cover that no prediction was
    /// issued for take the mean of the patches over them under the taper
    /// weight `wy * wx`, summed in f64 in frame order as
    /// `nn::predictor::stitch` does, and count as issued from here on, so
    /// a fill that is dropped instead of written leaves them at what they
    /// had. Parts of a patch outside the object are ignored.
    ///
    /// `None`, with nothing issued, when no buffer is free: the pixels stay
    /// open for a later call.
    pub fn stitch(&mut self, corners: &[[u32; 2]], patches: &[[f32; 2]]) -> Option<Fill> {
        let ([nyo, nxo], [py, px]) = (self.shape, self.patch);
        assert!(corners.len() <= self.max_batch, "call over max_batch");
        assert_eq!(patches.len(), corners.len() * py * px);
        let taken = self
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let Some(mut b) = taken else {
            self.no_buffer += 1;
            return None;
        };
        b.rects.clear();
        // Each patch clipped to the object: rows and columns `[start, end)`.
        let clip = |c: u32, len: usize, n: usize| (c as usize).min(n)..(c as usize + len).min(n);
        let rows = |c: &[u32; 2]| clip(c[0], py, nyo);
        let cols = |c: &[u32; 2]| clip(c[1], px, nxo);
        let y0 = corners.iter().map(|c| rows(c).start).min().unwrap_or(0);
        let y1 = corners.iter().map(|c| rows(c).end).max().unwrap_or(0);
        let [prev, cur] = &mut self.open;
        prev.clear();
        for y in y0..y1 {
            self.spans.clear();
            self.spans.extend(
                corners
                    .iter()
                    .filter(|c| rows(c).contains(&y))
                    .map(|c| [cols(c).start as u32, cols(c).end as u32]),
            );
            self.spans.sort_unstable();
            cur.clear();
            let (mut x, mut next) = (0, 0);
            while let Some(&[s, mut e]) = self.spans.get(next) {
                // The union of the spans that touch this one.
                next += 1;
                while let Some(&[_, e2]) = self.spans.get(next).filter(|t| t[0] <= e) {
                    e = e.max(e2);
                    next += 1;
                }
                x = x.max(s);
                while x < e {
                    let i = y * nxo + x as usize;
                    if self.issued[i / 64] >> (i % 64) & 1 == 1 {
                        x += 1;
                        continue;
                    }
                    // A run of open pixels: issue it.
                    let start = x;
                    while x < e {
                        let i = y * nxo + x as usize;
                        let (word, bit) = (&mut self.issued[i / 64], 1u64 << (i % 64));
                        if *word & bit != 0 {
                            break;
                        }
                        *word |= bit;
                        x += 1;
                    }
                    let (y, w) = (y as u32, x - start);
                    // It extends the rectangle that ended on the row above
                    // over the same columns, or opens one.
                    let above = prev.iter().copied().find(|&r| {
                        let r = &b.rects[r];
                        (r.x, r.w, r.y + r.h) == (start, w, y)
                    });
                    cur.push(match above {
                        Some(r) => {
                            b.rects[r].h += 1;
                            r
                        }
                        None => {
                            b.rects.push(Rect {
                                y,
                                x: start,
                                h: 1,
                                w,
                                at: 0,
                            });
                            b.rects.len() - 1
                        }
                    });
                }
            }
            std::mem::swap(prev, cur);
        }
        let mut at = 0;
        let values = b.values.as_mut_slice();
        for r in &mut b.rects {
            r.at = at;
            let (ry, rx) = (
                r.y as usize..(r.y + r.h) as usize,
                r.x as usize..(r.x + r.w) as usize,
            );
            let (h, w) = (ry.len(), rx.len());
            at += h * w;
            let acc = &mut self.acc[..h * w];
            acc.fill([0.0; 3]);
            for (f, c) in corners.iter().enumerate() {
                let (fy, fx) = (rows(c), cols(c));
                let (ys, xs) = (
                    ry.start.max(fy.start)..ry.end.min(fy.end),
                    rx.start.max(fx.start)..rx.end.min(fx.end),
                );
                for y in ys {
                    let (v, wy) = (y - fy.start, self.taper[0][y - fy.start]);
                    for x in xs.clone() {
                        let u = x - fx.start;
                        let weight = wy * self.taper[1][u];
                        let [re, im] = patches[(f * py + v) * px + u].map(f64::from);
                        let a = &mut acc[(y - ry.start) * w + x - rx.start];
                        a[0] += re * weight;
                        a[1] += im * weight;
                        a[2] += weight;
                    }
                }
            }
            for (v, [re, im, weight]) in values[r.at..at].iter_mut().zip(acc.iter()) {
                *v = [(re / weight) as f32, (im / weight) as f32];
            }
        }
        Some(Fill(Arc::new(Lent {
            buffers: Some(b),
            pool: self.pool.clone(),
        })))
    }
}

/// A stand-in predictor for the engines' tests.
#[cfg(test)]
pub(crate) mod standin {
    use super::*;
    use crate::gpu::ViewAddr;

    /// Predicts what no object is: unit pixels of a phase hashed from the
    /// pixel's place in the stream of predictions.
    pub(crate) struct Wrong {
        pub(crate) patch: [usize; 2],
        pub(crate) issued: u64,
    }

    impl PatchPredictor for Wrong {
        fn patch(&self) -> [usize; 2] {
            self.patch
        }

        fn predict(
            &mut self,
            counts: &[u16],
            frame: [usize; 2],
            out: &mut [[f32; 2]],
        ) -> Result<(), String> {
            let n = counts.len() / (frame[0] * frame[1]);
            if out.len() != n * self.patch[0] * self.patch[1] {
                return Err(format!("{} values for {n} frames", out.len()));
            }
            for v in out {
                self.issued += 1;
                let h = self.issued.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
                let phase = h as f32 / (1u32 << 24) as f32 * std::f32::consts::TAU;
                *v = [phase.cos(), phase.sin()];
            }
            Ok(())
        }
    }

    /// The fill of one call as the wiring builds it: `counts` and `views`,
    /// `nmode` per frame, of its frames in landing order.
    pub(crate) fn batch_fill(
        pf: &mut PatchFill,
        pred: &mut dyn PatchPredictor,
        counts: &[u16],
        views: &[ViewAddr],
        nmode: usize,
        frame: [usize; 2],
    ) -> Fill {
        let patch = pred.patch();
        let corners: Vec<[u32; 2]> = views
            .chunks(nmode)
            .map(|v| patch_corner([v[0].oy, v[0].ox], frame, patch))
            .collect();
        let mut out = vec![[0.0; 2]; corners.len() * patch[0] * patch[1]];
        pred.predict(counts, frame, &mut out).unwrap();
        pf.stitch(&corners, &out).expect("a buffer is free")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::object::LiveObject;

    const PATCH: [usize; 2] = [8, 8];

    fn taper(i: usize, n: usize) -> f64 {
        (PI * (i as f64 + 0.5) / n as f64).sin().powi(2)
    }

    /// A patch of one value.
    fn flat(v: [f32; 2]) -> Vec<[f32; 2]> {
        vec![v; PATCH[0] * PATCH[1]]
    }

    /// The value `fill` holds for object pixel `[y, x]`.
    fn at(fill: &Fill, [y, x]: [u32; 2]) -> Option<[f32; 2]> {
        let r = fill
            .rects()
            .iter()
            .find(|r| (r.y..r.y + r.h).contains(&y) && (r.x..r.x + r.w).contains(&x))?;
        Some(fill.values()[r.at + ((y - r.y) * r.w + x - r.x) as usize])
    }

    fn stitcher(pool: usize) -> PatchFill {
        let ctx = Context::new(0).expect("GPU 0");
        PatchFill::new(&ctx, [64, 64], PATCH, 4, pool).unwrap()
    }

    #[test]
    fn patch_corner_centres_the_patch_in_the_window() {
        assert_eq!(patch_corner([10, 20], [16, 32], [8, 8]), [14, 32]);
    }

    /// A view at `u32::MAX`, where `ViewPlacement::views` puts a corner
    /// outside `0..u32::MAX`, or just short of it gives the patch corner
    /// `u32::MAX`, a patch the stitch leaves out whole beside one inside
    /// the object.
    #[test]
    fn a_corner_at_u32_max_stays_outside_the_object() {
        let far = patch_corner([u32::MAX, u32::MAX - 1], [16, 32], PATCH);
        assert_eq!(far, [u32::MAX, u32::MAX]);
        let mut pf = stitcher(1);
        let fill = pf.stitch(&[[4, 6], far], &flat([1.0, 0.0]).repeat(2));
        let fill = fill.expect("a buffer is free");
        assert_eq!(
            fill.rects(),
            [Rect {
                y: 4,
                x: 6,
                h: 8,
                w: 8,
                at: 0
            }]
        );
    }

    /// One patch gives its own values on one rectangle; two patches of a
    /// batch average under the taper where they overlap; pixels outside
    /// both are in no rectangle.
    #[test]
    fn a_batch_stitches_to_the_taper_mean() {
        let mut pf = stitcher(2);
        let mut ramp = flat([0.0, 0.0]);
        for (i, v) in ramp.iter_mut().enumerate() {
            *v = [i as f32, -(i as f32)];
        }
        let one = pf.stitch(&[[4, 6]], &ramp).unwrap();
        assert_eq!(
            one.rects(),
            [Rect {
                y: 4,
                x: 6,
                h: 8,
                w: 8,
                at: 0
            }]
        );
        assert_eq!(one.values()[..64], ramp[..]);
        assert_eq!(at(&one, [5, 7]), Some([9.0, -9.0]));
        assert_eq!(at(&one, [3, 6]), None);

        // Two patches, the second 3 columns right of the first.
        let (a, b) = ([2.0f32, 1.0], [4.0f32, -3.0]);
        let both = [flat(a), flat(b)].concat();
        let two = pf.stitch(&[[30, 20], [30, 23]], &both).unwrap();
        assert_eq!(two.pixels(), 8 * 11);
        assert_eq!(two.rects().len(), 1);
        assert_eq!(at(&two, [31, 21]), Some(a));
        assert_eq!(at(&two, [31, 29]), Some(b));
        // Column 25 is column 5 of the first patch and 2 of the second.
        let (wa, wb) = (taper(5, 8), taper(2, 8));
        let mean = |i: usize| ((f64::from(a[i]) * wa + f64::from(b[i]) * wb) / (wa + wb)) as f32;
        let got = at(&two, [33, 25]).unwrap();
        assert!((got[0] - mean(0)).abs() < 1e-6 && (got[1] - mean(1)).abs() < 1e-6);
        assert_eq!(at(&two, [38, 25]), None);
    }

    /// A later batch gets values only for the pixels no earlier batch was
    /// issued, also when the earlier fill was dropped unwritten.
    #[test]
    fn the_first_patch_wins() {
        let mut pf = stitcher(1);
        let first = pf.stitch(&[[10, 10]], &flat([1.0, 0.0])).unwrap();
        assert_eq!(first.pixels(), 64);
        drop(first);
        assert!(pf.issued([10, 10]) && pf.issued([17, 17]) && !pf.issued([18, 17]));
        // Shifted 2 down and 3 right: an L of 8 * 8 - 6 * 5 new pixels.
        let later = pf.stitch(&[[12, 13]], &flat([5.0, 5.0])).unwrap();
        assert_eq!(later.pixels(), 64 - 30);
        assert_eq!(at(&later, [12, 13]), None);
        assert_eq!(at(&later, [17, 17]), None);
        assert_eq!(at(&later, [12, 18]), Some([5.0, 5.0]));
        assert_eq!(at(&later, [19, 13]), Some([5.0, 5.0]));
        // Rows 12..18 right of the old patch, rows 18..20 full width.
        assert_eq!(
            later.rects(),
            [
                Rect {
                    y: 12,
                    x: 18,
                    h: 6,
                    w: 3,
                    at: 0
                },
                Rect {
                    y: 18,
                    x: 13,
                    h: 2,
                    w: 8,
                    at: 18
                }
            ]
        );
        drop(later);
        assert_eq!(
            pf.stitch(&[[12, 13]], &flat([9.0, 9.0])).unwrap().pixels(),
            0
        );
    }

    /// A patch over the object's edge is clipped, and with every buffer in
    /// flight a batch is turned away without issuing anything.
    #[test]
    fn the_pool_bounds_fills_in_flight() {
        let mut pf = stitcher(1);
        let held = pf.stitch(&[[60, 60]], &flat([1.0, 2.0])).unwrap();
        assert_eq!(held.pixels(), 16);
        let shared = held.clone();
        assert!(pf.stitch(&[[0, 0]], &flat([3.0, 3.0])).is_none());
        assert_eq!(pf.no_buffer(), 1);
        assert!(!pf.issued([0, 0]));
        drop(held);
        // A clone still holds the buffer.
        assert!(pf.stitch(&[[0, 0]], &flat([3.0, 3.0])).is_none());
        drop(shared);
        let again = pf.stitch(&[[0, 0]], &flat([3.0, 3.0])).unwrap();
        assert_eq!(at(&again, [7, 7]), Some([3.0, 3.0]));
    }

    /// Consumer-thread cost of `write_fill` for 16-frame batches of
    /// 256-pixel frames with 128-pixel patches; prints, checks nothing.
    #[test]
    #[ignore = "timing; run alone on a quiet GPU"]
    fn write_fill_cost() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (n, p, batch) = (4096usize, 128usize, 16usize);
        let nt = (n / 16) as u32;
        let patches = vec![[0.5f32, 0.25]; batch * p * p];
        // Corners of each batch: a raster stepped 30% of a frame, 16 to a
        // row; disjoint patches (the most bytes); and a 1-pixel diagonal
        // staircase, alone and with every other column issued (the most
        // rectangles).
        let raster = |b: usize| -> Vec<[u32; 2]> {
            (0..batch)
                .map(|i| [(b * 77) as u32, (i * 77) as u32])
                .collect()
        };
        let disjoint = |b: usize| -> Vec<[u32; 2]> {
            (0..batch)
                .map(|i| [(b * p) as u32, (i * p) as u32])
                .collect()
        };
        let stair = |b: usize| -> Vec<[u32; 2]> {
            (0..batch)
                .map(|i| [(b * 160 + i) as u32, (i * 130 + i) as u32])
                .collect()
        };
        // Corners scattered over a 1500-pixel square that fills up.
        let scatter = |b: usize| -> Vec<[u32; 2]> {
            let mut r = 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(b as u64 + 1);
            let mut next = || {
                r = r
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (r >> 33) as u32 % 1500
            };
            (0..batch).map(|_| [next(), next()]).collect()
        };
        type Corners<'a> = &'a dyn Fn(usize) -> Vec<[u32; 2]>;
        let cases: [(&str, Corners, bool); 5] = [
            ("raster 70% overlap", &raster, false),
            ("disjoint", &disjoint, false),
            ("staircase", &stair, false),
            ("scattered", &scatter, false),
            ("staircase over comb", &stair, true),
        ];
        for (name, corners, comb) in cases {
            let mut o = LiveObject::new(&s, 1, [n, n], 4, [1.0, 0.0], 1).unwrap();
            let tiles: Vec<[u32; 2]> = (0..nt)
                .flat_map(|ty| (0..nt).map(move |tx| [ty, tx]))
                .collect();
            o.backer().back(tiles.iter().copied()).unwrap();
            o.init_tiles(tiles).unwrap();
            s.synchronize().unwrap();
            let mut pf = PatchFill::new(&ctx, [n, n], [p, p], batch, 4).unwrap();
            if comb {
                // Every other column already issued: strips one pixel wide.
                pf.issued.fill(0x5555_5555_5555_5555);
            }
            let (mut stitch, mut write, mut worst) = (0.0f64, 0.0f64, 0.0f64);
            let (mut rects, mut px, rounds) = (0usize, 0usize, 20usize);
            for b in 0..rounds {
                let t = std::time::Instant::now();
                let fill = pf.stitch(&corners(b), &patches).unwrap();
                stitch += t.elapsed().as_secs_f64();
                rects += fill.rects().len();
                px += fill.pixels();
                let t = std::time::Instant::now();
                assert_eq!(o.write_fill(fill, [0, 0]).unwrap(), 0);
                let dt = t.elapsed().as_secs_f64();
                write += dt;
                worst = worst.max(dt);
                s.synchronize().unwrap();
            }
            let r = rounds as f64;
            println!(
                "{name}: {:.1} rects, {:.1} copies, {:.0} px per batch; write_fill mean {:.1} us max {:.1} us; stitch mean {:.1} us",
                rects as f64 / r,
                o.fill_copies() as f64 / r,
                px as f64 / r,
                write / r * 1e6,
                worst * 1e6,
                stitch / r * 1e6,
            );
        }
    }
}
