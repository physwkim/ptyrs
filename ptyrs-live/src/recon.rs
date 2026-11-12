//! Live AP (DESIGN.md §3 B2, B6; §7 M5): the object of a live scan iterated
//! over the frames landed so far, with the probe fixed, from three graphs
//! captured before the first frame lands and only replayed afterwards.
//!
//! An epoch is one AP cycle over every landed frame: the ingest graph bins
//! the frames landed since the last epoch, the sweep graph takes the next
//! stack of the sweep entry through PyNX AP (exit waves, FFT, amplitude
//! projection, inverse FFT, object sums) and the update graph applies the
//! sums to the object and rewinds the sweep. Frames land only between
//! epochs, so an epoch's sweep covers exactly the frames landed when it was
//! queued, as the engine's AP cycle covers its scan.

use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;

use cuda_core::{DeviceBuffer, Stream};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{C32, Kernels, UpdateSums, ViewAddr};

use crate::graph::LiveGraph;
use crate::grow::GrowError;
use crate::object::Snapshot;
use crate::scan::{LiveScan, ScanShape};

/// A live AP run's fixed parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApConfig {
    pub shape: ScanShape,
    /// Frames per stack.
    pub nf_max: NonZeroU32,
    /// View slots per object tile.
    pub cap: u32,
    /// Fresh object pixels.
    pub fill: [f32; 2],
    /// Device memory is backed this many bytes at a time.
    pub chunk: usize,
    /// PyNX `ObjNorm` inertia (the engine's AP default is 0.01).
    pub inertia: f32,
}

pub struct LiveAp {
    // The graphs drop before the buffers and kernels they replay on.
    ingest: LiveGraph,
    sweep: LiveGraph,
    update: LiveGraph,
    scan: LiveScan,
    _kernels: Kernels,
    _probe: DeviceBuffer<C32>,
    _psi: DeviceBuffer<C32>,
    _plan: Plan2d,
    _sums: UpdateSums,
    _counts: DeviceBuffer<u16>,
    /// Frames the ingest graph has binned.
    binned: u32,
    epochs: u64,
}

impl LiveAp {
    /// An empty scan of `cfg.shape` on `stream` with the fixed `probe`
    /// (`[nprobe, ny, nx]`, natural order), reading frame `f` of the scan
    /// from element `f * ny * nx` of `counts`, which holds at least
    /// `nframe` frames; captures its three graphs.
    pub fn new(
        stream: &Arc<Stream>,
        cfg: &ApConfig,
        probe: &[C32],
        counts: DeviceBuffer<u16>,
    ) -> Result<Self, GrowError> {
        let ctx = stream.context();
        let kernels = Kernels::load(ctx)?;
        let mut scan = LiveScan::new(stream, &cfg.shape, cfg.nf_max, cfg.cap, cfg.fill, cfg.chunk)?;
        let d = *scan.dims();
        let n = d.frame[0] as usize * d.frame[1] as usize;
        let planes = cfg.nf_max.get() as usize * d.nmode as usize;
        if d.frame[0] != d.frame[1] {
            return Err(ptyrs_gpu::Error::Shape(format!(
                "live AP needs square frames, got {:?}",
                d.frame
            ))
            .into());
        }
        let probe = DeviceBuffer::from_host(stream, probe)?;
        let mut psi = DeviceBuffer::<C32>::zeroed(stream, planes * n)?;
        let mut plan =
            Plan2d::new(stream, d.frame[0] as usize, planes).map_err(ptyrs_gpu::Error::from)?;
        let mut sums = UpdateSums::zeroed(
            stream,
            d.nobj as usize * d.obj[0] as usize * d.obj[1] as usize,
        )?;
        let k = &kernels;
        let ingest = {
            let s = &mut scan;
            LiveGraph::capture(stream, move |c| s.advance(k, c))?
        };
        let sweep = {
            let (s, psi, plan, sums) = (&scan, &mut psi, &mut plan, &mut sums);
            let (probe, counts) = (&probe, &counts);
            LiveGraph::capture(stream, move |c| {
                let (views, f) = (s.views(), s.frames());
                f.sweep_step(c)?;
                let st = f.sweep();
                k.obj_probe_to_psi(c, views, st, s.pixels(), probe, psi)?;
                plan.forward_inplace(c, psi)?;
                k.apply_amplitude(c, &d, st, counts, psi)?;
                plan.inverse_inplace(c, psi)?;
                k.psi_to_obj(c, views, st, psi, probe, sums)
            })?
        };
        let update = {
            let (s, sums, inertia) = (&mut scan, &mut sums, cfg.inertia);
            LiveGraph::capture(stream, move |c| {
                let (views, obj) = s.object_mut();
                k.update_obj_from_sums(c, views, obj, sums, inertia)?;
                s.frames().rewind(c)
            })?
        };
        Ok(Self {
            ingest,
            sweep,
            update,
            scan,
            _kernels: kernels,
            _probe: probe,
            _psi: psi,
            _plan: plan,
            _sums: sums,
            _counts: counts,
            binned: 0,
            epochs: 0,
        })
    }

    /// Lands the frames whose views are `views` as the next frames of the
    /// scan ([`LiveScan::push`]); their counts must be in `counts` when the
    /// next epoch runs.
    pub fn push(&mut self, views: &[ViewAddr]) -> Result<u32, GrowError> {
        self.scan.push(views)
    }

    /// Queues one epoch over every frame landed so far, without waiting.
    pub fn epoch(&mut self) -> Result<(), GrowError> {
        let (landed, nf) = (self.scan.landed(), self.scan.frames().nf_max());
        while self.binned < landed {
            self.ingest.replay()?;
            self.binned = (self.binned + nf).min(landed);
        }
        for _ in 0..landed.div_ceil(nf) {
            self.sweep.replay()?;
        }
        self.update.replay()?;
        self.epochs += 1;
        Ok(())
    }

    /// Queues the object's copy into `snap` after the epochs queued so far.
    pub fn snapshot(&self, snap: &mut Snapshot) -> Result<(), GrowError> {
        self.scan.obj().snapshot(snap)
    }

    /// A buffer for [`Self::snapshot`].
    pub fn new_snapshot(&self) -> Result<Snapshot, GrowError> {
        Ok(Snapshot::new(self.scan.obj())?)
    }

    pub fn scan(&self) -> &LiveScan {
        &self.scan
    }

    /// Epochs queued so far.
    pub fn epochs(&self) -> u64 {
        self.epochs
    }

    pub fn stream(&self) -> &Arc<Stream> {
        self.scan.stream()
    }
}

/// Where a scan position puts a frame in the live object: the corner of its
/// window is the position in object pixels plus a fixed origin, rounded half
/// to even like the engine's views.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewPlacement {
    /// `[py, px]`, m.
    pub pixel: [f64; 2],
    /// Corner of the window at position `(0, 0)`, `[oy, ox]` object pixels.
    pub origin: [f64; 2],
}

impl ViewPlacement {
    /// The placement of a planned scan, positions `(x, y)` in m: the windows
    /// of its first-to-last-planned extent start at row and column 0, and the
    /// returned reach is the largest corner, for [`ScanShape::reach`].
    pub fn plan(pixel: [f64; 2], positions: &[(f64, f64)]) -> (Self, [u32; 2]) {
        let px = |p: &(f64, f64), a: usize| [p.1, p.0][a] / pixel[a];
        let lo = [0, 1].map(|a| {
            positions
                .iter()
                .map(|p| px(p, a).round_ties_even())
                .fold(f64::INFINITY, f64::min)
        });
        let place = Self {
            pixel,
            origin: lo.map(|l| -l),
        };
        let reach = [0, 1].map(|a| {
            positions
                .iter()
                .map(|p| place.corner(*p)[a])
                .fold(0.0, f64::max) as u32
        });
        (place, reach)
    }

    /// `[oy, ox]` of the window at `(x, y)`, m; may lie outside the object.
    pub fn corner(&self, (x, y): (f64, f64)) -> [f64; 2] {
        [
            (y / self.pixel[0] + self.origin[0]).round_ties_even(),
            (x / self.pixel[1] + self.origin[1]).round_ties_even(),
        ]
    }

    /// Rows and columns of the bounding box of the window centres of frames
    /// of `frame` pixels at `positions`, `(x, y)` m: the scanned region the
    /// [`Clock`](crate::converge::Clock) compares. Empty for no position.
    pub fn scanned(&self, positions: &[(f64, f64)], frame: [u32; 2]) -> [Range<usize>; 2] {
        [0, 1].map(|a| {
            let c = positions
                .iter()
                .map(|&p| self.corner(p)[a] + f64::from(frame[a] / 2));
            let lo = c.clone().fold(f64::INFINITY, f64::min);
            let hi = c.fold(f64::NEG_INFINITY, f64::max);
            if lo > hi {
                0..0
            } else {
                lo.max(0.0) as usize..hi.max(0.0) as usize + 1
            }
        })
    }

    /// The `nprobe * nobj` views of the frame at `(x, y)` into `out`, mode
    /// `m` from probe `m % nprobe` and object `m / nprobe` (the engine's
    /// order). A corner outside `0..u32::MAX` becomes `u32::MAX`, which
    /// [`LiveScan::push`] refuses.
    pub fn views(&self, pos: (f64, f64), nprobe: u32, nobj: u32, out: &mut Vec<ViewAddr>) {
        let [oy, ox] = self.corner(pos).map(|c| {
            if (0.0..u32::MAX as f64).contains(&c) {
                c as u32
            } else {
                u32::MAX
            }
        });
        out.extend((0..nprobe * nobj).map(|m| ViewAddr {
            oy,
            ox,
            fy: 0.0,
            fx: 0.0,
            probe: m % nprobe,
            obj: m / nprobe,
        }));
    }
}

#[cfg(test)]
mod tests {
    use cuda_core::Context;
    use ptyrs_core::{
        AddrTable, C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack, required_obj_shape,
    };
    use ptyrs_engine::{ApParams, GpuOptions, GpuPtycho};

    use super::*;
    use crate::converge::Clock;
    use crate::farfield::{Ptycho, PtychoSpec, ScanPath};
    use crate::graph::WarmUp;

    const FILL: [f32; 2] = [1.0, 0.0];

    fn sim(n: usize, cols: usize, points: usize, step: f64) -> Ptycho {
        Ptycho::new(PtychoSpec {
            n,
            path: ScanPath::Raster { cols },
            points,
            step,
            photons: 1e6,
            seed: 5,
            wavelength: 1.5498e-10,
            distance: 5.0,
            det_pixel: 75e-6,
        })
    }

    fn frames(p: &Ptycho) -> Vec<u16> {
        (0..p.points()).flat_map(|k| p.frame(k)).collect()
    }

    fn probe(p: &Ptycho) -> Vec<C32> {
        p.probe()
            .iter()
            .map(|c| C32 { re: c.re, im: c.im })
            .collect()
    }

    fn config(shape: ScanShape, nf_max: u32) -> ApConfig {
        ApConfig {
            shape,
            nf_max: NonZeroU32::new(nf_max).unwrap(),
            cap: 256,
            fill: FILL,
            chunk: 1,
            inertia: 0.01,
        }
    }

    /// Plane `z` of a snapshot.
    fn plane(s: &mut Snapshot, z: usize) -> (Vec<C64>, [usize; 2]) {
        let [ny, nx] = s.shape();
        let px = s.wait().unwrap();
        let v = px[z * ny * nx..][..ny * nx]
            .iter()
            .map(|&[re, im]| C64::new(re.into(), im.into()))
            .collect();
        (v, [ny, nx])
    }

    /// The placement of a planned scan puts every window where the
    /// simulator cut it, its scanned region spans the window centres, and a
    /// position off the object gives views `push` refuses.
    #[test]
    fn a_planned_placement_puts_each_window_at_its_simulated_offset() {
        let p = sim(32, 7, 45, 5.0);
        let pos: Vec<(f64, f64)> = (0..p.points()).map(|k| p.position(k)).collect();
        let dx = p.spec().dx();
        let (place, reach) = ViewPlacement::plan([dx, dx], &pos);
        let (_, [ny, nx]) = p.object();
        assert_eq!(reach, [ny as u32 - 32, nx as u32 - 32]);
        let mut v = Vec::new();
        for (k, &q) in pos.iter().enumerate() {
            v.clear();
            place.views(q, 2, 1, &mut v);
            let [oy, ox] = p.offset(k).map(|o| o as u32);
            assert_eq!(v.len(), 2);
            for (m, w) in v.iter().enumerate() {
                assert_eq!((w.oy, w.ox, w.probe, w.obj), (oy, ox, m as u32, 0), "{k}");
            }
        }
        assert_eq!(place.scanned(&pos, [32, 32]), [16..ny - 15, 16..nx - 15]);
        assert_eq!(place.scanned(&[], [32, 32]), [0..0, 0..0]);
        v.clear();
        place.views((pos[0].0 - dx, pos[0].1), 1, 1, &mut v);
        assert_eq!((v[0].oy, v[0].ox), (0, u32::MAX));
    }

    /// Epochs over every frame landed at once are the engine's AP cycles
    /// with the probe fixed: same views, stacks, frames and start, so the
    /// same object after each of a few cycles.
    #[test]
    fn epochs_over_a_whole_scan_are_the_engine_ap_cycles() {
        let (n, nf_max) = (64usize, 16u32);
        let p = sim(n, 6, 36, 12.0);
        let s = p.spec();
        let counts = frames(&p);
        let g = Geometry::new(s.wavelength, s.distance, s.det_pixel, [n, n]).unwrap();
        let (x, y): (Vec<f64>, Vec<f64>) = (0..p.points()).map(|k| p.position(k)).unzip();
        let scan = Scan::new(
            g,
            Positions::new(y, x).unwrap(),
            Counts::from_raw([p.points(), n, n], counts.clone()).unwrap(),
        )
        .unwrap();
        let pixel = g.pixel_size_object();
        let pre = scan.positions.pynx_preprocess(pixel);
        let obj = required_obj_shape(&pre, pixel, [n, n], 0);
        let addr = AddrTable::new(&pre, pixel, [n, n], obj, 2, 1).unwrap();
        let views: Vec<ViewAddr> = addr
            .views()
            .iter()
            .map(|v| ViewAddr {
                oy: v.oy,
                ox: v.ox,
                fy: 0.0,
                fx: 0.0,
                probe: v.probe,
                obj: v.obj,
            })
            .collect();
        let reach = [0, 1].map(|a| views.iter().map(|v| [v.oy, v.ox][a]).max().unwrap());
        let quad_phase =
            (std::f64::consts::PI * pixel[1] * pixel[1] / (g.wavelength * g.distance)) as f32;

        let ctx = Context::new(0).expect("GPU 0");
        let start = Object::new(Stack::filled([1, obj[0], obj[1]], C64::new(1.0, 0.0)));
        let truth = Probe::new(
            Stack::from_vec(
                [2, n, n],
                p.probe()
                    .iter()
                    .map(|c| C64::new(c.re.into(), c.im.into()))
                    .collect(),
            )
            .unwrap(),
        );
        let opts = GpuOptions {
            stack_size: nf_max as usize,
            graph: true,
            llk_slots: 8,
        };
        let mut engine = GpuPtycho::new(&ctx, &scan, start, truth, opts).unwrap();
        let params = ApParams {
            obj_inertia: Some(0.01),
            probe_inertia: None,
            calc_llk: 0,
            zero_phase_ramp: false,
        };

        let st = ctx.new_stream().unwrap();
        let shape = ScanShape {
            nframe: p.points() as u32,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [n as u32; 2],
            reach,
            quad_phase,
        };
        let dev = DeviceBuffer::from_host(&st, &counts).unwrap();
        let mut live = LiveAp::new(&st, &config(shape, nf_max), &probe(&p), dev).unwrap();
        assert_eq!(live.push(&views).unwrap(), p.points() as u32);
        let mut snap = live.new_snapshot().unwrap();
        let [ly, lx] = snap.shape();
        // Every window lies inside both objects' common corner.
        let [cy, cx] = [ly.min(obj[0]), lx.min(obj[1])];
        assert!(reach[0] as usize + n <= cy && reach[1] as usize + n <= cx);
        for cycles in [1, 2, 5] {
            for _ in 0..cycles {
                live.epoch().unwrap();
            }
            engine.ap(&params, cycles).unwrap();
            live.snapshot(&mut snap).unwrap();
            let (got, _) = plane(&mut snap, 0);
            let want = engine.object().unwrap();
            let want = want.as_slice();
            let mut worst = 0f64;
            for y in 0..cy {
                for x in 0..cx {
                    let (a, b) = (got[y * lx + x], want[y * obj[1] + x]);
                    worst = worst.max((a - b).norm() / b.norm().max(1e-3));
                }
            }
            assert_eq!(worst, 0.0, "after {} epochs", live.epochs());
        }
    }

    /// The views of a simulated scan at its planned placement, two probe
    /// modes, its shape and its scanned region.
    fn planned(p: &Ptycho) -> (Vec<ViewAddr>, ScanShape, [Range<usize>; 2]) {
        let pos: Vec<(f64, f64)> = (0..p.points()).map(|k| p.position(k)).collect();
        let dx = p.spec().dx();
        let (place, reach) = ViewPlacement::plan([dx, dx], &pos);
        let mut views = Vec::new();
        for &q in &pos {
            place.views(q, 2, 1, &mut views);
        }
        let shape = ScanShape {
            nframe: p.points() as u32,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [p.spec().n as u32; 2],
            reach,
            quad_phase: 0.0,
        };
        (views, shape, place.scanned(&pos, shape.frame))
    }

    /// Frames of two columns of 250 rows, an object of three 2 MiB chunks,
    /// landed in uneven batches with epochs between: the object backed as
    /// the scan reaches new rows holds, after every batch, exactly the
    /// object of the same run backed whole at its first frame, from graphs
    /// captured before either landed a frame.
    #[test]
    fn an_object_backed_as_the_scan_grows_is_the_object_backed_whole() {
        let p = sim(64, 2, 500, 24.0);
        let (views, shape, _) = planned(&p);
        let ctx = Context::new(0).expect("GPU 0");
        let run = |chunk: usize| {
            let st = ctx.new_stream().unwrap();
            let dev = DeviceBuffer::from_host(&st, &frames(&p)).unwrap();
            let cfg = ApConfig {
                chunk,
                ..config(shape, 32)
            };
            LiveAp::new(&st, &cfg, &probe(&p), dev).unwrap()
        };
        let (mut grown, mut whole) = (run(1), run(64 << 20));
        let warm = WarmUp::end();
        let (mut sg, mut sw) = (grown.new_snapshot().unwrap(), whole.new_snapshot().unwrap());
        let mut mapped = vec![grown.scan().obj().mapped_bytes()];
        let mut at = 0;
        for batch in [40, 7, 130, 0, 90, 233] {
            let v = &views[2 * at..2 * (at + batch)];
            for live in [&mut grown, &mut whole] {
                assert_eq!(live.push(v).unwrap(), batch as u32);
                for _ in 0..3 {
                    live.epoch().unwrap();
                }
            }
            at += batch;
            mapped.push(grown.scan().obj().mapped_bytes());
            grown.snapshot(&mut sg).unwrap();
            whole.snapshot(&mut sw).unwrap();
            assert!(
                sg.wait().unwrap() == sw.wait().unwrap(),
                "after {at} frames"
            );
        }
        assert_eq!(at, p.points());
        assert!(mapped.windows(2).all(|w| w[0] <= w[1]), "{mapped:?}");
        assert!(mapped[1] < *mapped.last().unwrap(), "{mapped:?}");
        assert_eq!(whole.scan().obj().mapped_bytes(), 64 << 20);
        assert_eq!((grown.scan().lost(), whole.scan().lost()), (0, 0));
        assert_eq!(warm.recaptures(), 0);
    }

    /// Frames landed in uneven batches with epochs between, from graphs
    /// captured before the first frame: once the scan ends, the live object
    /// converges over the scanned region to the offline run over every
    /// frame (settled to 0.9994 between its epochs 900 and 1000), with no
    /// recapture.
    #[test]
    fn a_scan_landed_in_batches_converges_to_the_offline_run() {
        let p = sim(64, 6, 36, 12.0);
        let (views, shape, region) = planned(&p);
        let ctx = Context::new(0).expect("GPU 0");
        let run = || {
            let st = ctx.new_stream().unwrap();
            let dev = DeviceBuffer::from_host(&st, &frames(&p)).unwrap();
            LiveAp::new(&st, &config(shape, 16), &probe(&p), dev).unwrap()
        };

        let (settle, every) = (1000, 100);
        let mut offline = run();
        offline.push(&views).unwrap();
        let mut snap = offline.new_snapshot().unwrap();
        for _ in 0..settle - every {
            offline.epoch().unwrap();
        }
        offline.snapshot(&mut snap).unwrap();
        let (earlier, sh) = plane(&mut snap, 0);
        for _ in 0..every {
            offline.epoch().unwrap();
        }
        offline.snapshot(&mut snap).unwrap();
        // Epochs after the scan's end stand in for wall time.
        let clock = Clock::new(&plane(&mut snap, 0).0, &earlier, sh, region, 0).unwrap();

        let mut live = run();
        let warm = WarmUp::end();
        let mut at = 0;
        for batch in [5, 1, 13, 0, 17] {
            assert_eq!(
                live.push(&views[2 * at..2 * (at + batch)]).unwrap(),
                batch as u32
            );
            at += batch;
            for _ in 0..3 {
                live.epoch().unwrap();
            }
        }
        assert_eq!(at, p.points());
        let (scan_end, mut converged) = (live.epochs(), None);
        let mut snap = live.new_snapshot().unwrap();
        while converged.is_none() && live.epochs() < scan_end + 2 * settle {
            for _ in 0..25 {
                live.epoch().unwrap();
            }
            live.snapshot(&mut snap).unwrap();
            let after = (live.epochs() - scan_end) as i64;
            converged = clock.check(&plane(&mut snap, 0).0, after).unwrap();
        }
        assert!(
            converged.is_some(),
            "not converged in {} epochs",
            2 * settle
        );
        assert_eq!(live.scan().lost(), 0);
        assert_eq!(warm.recaptures(), 0);
    }
}
