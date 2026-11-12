//! One live scan on the device (DESIGN.md §3 B2, B3, B6): the address
//! table, the object with its bins, and the stack entry, all on one stream,
//! fed through [`LiveScan::push`] and read by the kernels through
//! [`Views::from_parts`].
//!
//! `push` is the one write site of frames into the scan. It lands a frame
//! only if its index is inside the scan, every view of it passes the
//! checks [`Views::new`] makes and every object tile it touches has a free
//! slot in its slab; it readies those tiles first. So every view the device
//! reads is in range, `bin_views` never overflows a slab and the stack entry
//! never needs the kernels' clamp. Frames land in order: the first frame
//! refused closes the scan, and it and every frame after it count as lost.
//!
//! `push` never backs device memory: a [`ScanBacker`], on a thread of its
//! own, backs the tiles and address records of frames before they are
//! pushed, and a frame pushed unbacked fails.

use std::num::NonZeroU32;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{DeviceBuffer, Stream};
use ptyrs_gpu::{C32, Kernels, StackRef, UpdateSums, ViewAddr, ViewDims, Views, view_tiles};

use crate::grow::{Backer, GrowError, GrowTable};
use crate::object::{LiveObject, ObjectBacker};
use crate::stack::LiveFrames;

/// A scan's geometry with the object sized from the largest view offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanShape {
    pub nframe: u32,
    /// Views per frame.
    pub nmode: u32,
    pub nprobe: u32,
    pub nobj: u32,
    /// `[ny, nx]`, both even.
    pub frame: [u32; 2],
    /// The largest `[oy, ox]` a view may have.
    pub reach: [u32; 2],
    pub quad_phase: f32,
}

impl ScanShape {
    /// The views' geometry. The object ends one pixel past the farthest
    /// frame, so a view at `reach` with a fractional offset still reads
    /// inside it and no view wraps.
    pub fn dims(&self) -> ViewDims {
        ViewDims {
            nframe: self.nframe,
            nmode: self.nmode,
            nprobe: self.nprobe,
            nobj: self.nobj,
            frame: self.frame,
            obj: [0, 1].map(|a| self.reach[a] + self.frame[a] + 1),
            quad_phase: self.quad_phase,
        }
    }

    /// The most of `views` that touch one object tile: the smallest slab
    /// `cap` under which [`LiveScan::push`] takes them all. Views outside
    /// the object touch no tile.
    pub fn tile_cap(&self, views: &[ViewAddr]) -> u32 {
        let d = self.dims();
        let [nty, ntx] = d.tiles().map(|t| t as usize);
        let mut n = vec![0u32; nty * ntx];
        for v in views.iter().filter(|v| d.check_view(v).is_ok()) {
            for [ty, tx] in view_tiles(&d, v) {
                n[ty as usize * ntx + tx as usize] += 1;
            }
        }
        n.into_iter().max().unwrap_or(0)
    }
}

/// A [`LiveScan`] split for the kernels that sweep it and update its object.
pub struct ScanParts<'a> {
    pub views: &'a Views,
    pub frames: &'a LiveFrames,
    pub pixels: &'a mut DeviceBuffer<C32>,
    pub sums: &'a mut UpdateSums,
}

/// The first frame a scan refused; it closed the scan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refusal {
    /// Its index is past the scan's last frame.
    PastScan { frame: u32 },
    /// One of its views fails the checks [`Views::new`] makes: inside the
    /// object, and mode `m` multiplying probe `m % nprobe` and object
    /// `m / nprobe` at the frame's position.
    View { frame: u32, view: ViewAddr },
    /// A tile it touches has no free slot left in its slab.
    Slab { frame: u32, tile: [u32; 2] },
    /// Backing or copying it on the device failed.
    Failed { frame: u32 },
}

/// Backs a [`LiveScan`]'s memory for frames ahead of [`LiveScan::push`],
/// on any thread.
pub struct ScanBacker {
    dims: ViewDims,
    obj: ObjectBacker,
    addr: Backer,
    /// Frames backed so far, in push order.
    next: u64,
}

impl ScanBacker {
    /// Backs the scan's plan before its first frame: what pushing `views`,
    /// `nmode` per frame, as the scan's first frames needs. It counts no
    /// frame as backed; [`Self::back`] then backs only what the frames
    /// pushed need beyond it.
    pub fn plan(&mut self, views: &[ViewAddr]) -> Result<(), GrowError> {
        self.back_from(0, views)
    }

    /// Backs what pushing `views`, `nmode` per frame, as the frames after
    /// those already backed needs. Frames land in push order, so backing
    /// every batch here before pushing it backs every frame that lands.
    pub fn back(&mut self, views: &[ViewAddr]) -> Result<(), GrowError> {
        self.back_from(self.next, views)?;
        self.next += (views.len() / self.dims.nmode as usize) as u64;
        Ok(())
    }

    /// Backs the object tiles of every view of `views`, frames `first` on,
    /// that passes [`ViewDims::check_view`] and the address records of
    /// those frames inside the scan.
    fn back_from(&mut self, first: u64, views: &[ViewAddr]) -> Result<(), GrowError> {
        let d = self.dims;
        let nmode = d.nmode as usize;
        let n = (views.len() / nmode) as u64;
        let inside = u64::from(d.nframe).clamp(first, first + n);
        let ins = &views[..(inside - first) as usize * nmode];
        self.obj.back(
            ins.iter()
                .filter(|v| d.check_view(v).is_ok())
                .flat_map(|v| view_tiles(&d, v)),
        )?;
        if inside > first {
            let sz = size_of::<ViewAddr>() * nmode;
            self.addr.back(first as usize * sz..inside as usize * sz)?;
        }
        Ok(())
    }
}

pub struct LiveScan {
    dims: ViewDims,
    views: Views,
    obj: LiveObject,
    addr: GrowTable<ViewAddr>,
    frames: LiveFrames,
    overflow: DeviceBuffer<u32>,
    /// Per tile, the slots its slab has given out: views pushed onto it.
    slots: Vec<u32>,
    stage: Vec<ViewAddr>,
    lost: u64,
    closed: Option<Refusal>,
}

impl LiveScan {
    /// An empty scan of `shape` on `stream`: stacks of at most `nf_max`
    /// frames, `cap` view slots per object tile, fresh object pixels at
    /// `fill`, device memory backed `chunk` bytes at a time.
    pub fn new(
        stream: &Arc<Stream>,
        shape: &ScanShape,
        nf_max: NonZeroU32,
        cap: u32,
        fill: [f32; 2],
        chunk: usize,
    ) -> Result<Self, GrowError> {
        let dims = shape.dims();
        let nview = dims.nframe as usize * dims.nmode as usize;
        let obj = LiveObject::new(
            stream,
            dims.nobj as usize,
            dims.obj.map(|n| n as usize),
            cap as usize,
            fill,
            chunk,
        )?;
        let addr = GrowTable::new(stream, nview, chunk)?;
        let frames = LiveFrames::new(stream, &dims, nf_max)?;
        // SAFETY: the kernels read the views and slabs of landed frames only
        // (the stack entry stays within them), and `push` lands a frame
        // after its records are queued and its tiles backed, and `bin_views`
        // appends each stack once, in frame order.
        let views = unsafe {
            let slabs = obj.tile_views()?;
            Views::from_parts(stream, dims, addr.typed(nview)?, slabs, cap)?
        };
        let ntiles = obj.tiles()[0] * obj.tiles()[1];
        Ok(Self {
            dims,
            views,
            obj,
            addr,
            frames,
            overflow: DeviceBuffer::zeroed(stream, 1)?,
            slots: vec![0; ntiles],
            stage: Vec::new(),
            lost: 0,
            closed: None,
        })
    }

    /// A backer of this scan's frames, for another thread. Back each batch
    /// through it before pushing the batch.
    pub fn backer(&self) -> ScanBacker {
        ScanBacker {
            dims: self.dims,
            obj: self.obj.backer(),
            addr: self.addr.backer(),
            next: u64::from(self.frames.landed()),
        }
    }

    /// Lands the frames whose views are `views`, `nmode` per frame, as the
    /// next frames of the scan, up to the first one refused; returns how
    /// many landed. Kernels queued on the stream afterwards see them. A
    /// [`ScanBacker`] must have backed them first.
    pub fn push(&mut self, views: &[ViewAddr]) -> Result<u32, GrowError> {
        let nmode = self.dims.nmode as usize;
        if !views.len().is_multiple_of(nmode) {
            return Err(ptyrs_gpu::Error::Shape(format!(
                "{} views are not whole frames of {nmode}",
                views.len()
            ))
            .into());
        }
        let n = (views.len() / nmode) as u64;
        if self.closed.is_some() {
            self.lost += n;
            return Ok(0);
        }
        let first = self.frames.landed();
        let pushed = self.stage_and_land(views);
        let landed = u64::from(self.frames.landed() - first);
        self.lost += n - landed;
        pushed.map(|()| landed as u32)
    }

    /// Stages the accepted prefix of `views`, then copies and lands it;
    /// closes the scan at the first refused frame or failure.
    fn stage_and_land(&mut self, views: &[ViewAddr]) -> Result<(), GrowError> {
        let first = self.frames.landed();
        let ntx = self.dims.tiles()[1] as usize;
        self.stage.clear();
        for (i, frame) in views.chunks(self.dims.nmode as usize).enumerate() {
            let f = first + i as u32;
            if let Some(r) = self.refusal(f, frame, ntx) {
                self.closed = Some(r);
                break;
            }
            let dims = self.dims;
            if let Err(e) = self
                .obj
                .init_tiles(frame.iter().flat_map(|v| view_tiles(&dims, v)))
            {
                self.closed = Some(Refusal::Failed { frame: f });
                return Err(e);
            }
            self.stage.extend_from_slice(frame);
        }
        let k = (self.stage.len() / self.dims.nmode as usize) as u32;
        if k == 0 {
            return Ok(());
        }
        let landed = self
            .addr
            .push(&self.stage)
            .and_then(|_| self.frames.land(first + k));
        if landed.is_err() {
            self.closed = Some(Refusal::Failed { frame: first });
        }
        landed
    }

    /// Why frame `f` with views `frame` cannot land, if it cannot; takes its
    /// slots when it can. Nothing lands after a refusal, so slots taken by
    /// the refused frame's earlier views are never given back.
    fn refusal(&mut self, f: u32, frame: &[ViewAddr], ntx: usize) -> Option<Refusal> {
        if f >= self.dims.nframe {
            return Some(Refusal::PastScan { frame: f });
        }
        let (np, at) = (self.dims.nprobe, |v: &ViewAddr| (v.oy, v.ox, v.fy, v.fx));
        let off = |(m, v): &(u32, &ViewAddr)| {
            self.dims.check_view(v).is_err()
                || v.probe != m % np
                || v.obj != m / np
                || at(v) != at(&frame[0])
        };
        if let Some((_, &view)) = (0..).zip(frame).find(off) {
            return Some(Refusal::View { frame: f, view });
        }
        let cap = self.obj.cap() as u32;
        for v in frame {
            for tile in view_tiles(&self.dims, v) {
                let s = &mut self.slots[tile[0] as usize * ntx + tile[1] as usize];
                if *s == cap {
                    return Some(Refusal::Slab { frame: f, tile });
                }
                *s += 1;
            }
        }
        None
    }

    /// Queues on `ctx`'s stream, or records into its capture, the step of
    /// the stack entry onto the next landed frames and the binning of their
    /// views; the stack kernels recorded after it take [`Self::stack`].
    pub fn advance(&mut self, k: &Kernels, ctx: &ExecutionContext) -> Result<(), ptyrs_gpu::Error> {
        self.frames.step(ctx)?;
        k.bin_views(
            ctx,
            &mut self.views,
            self.frames.stack(),
            &mut self.overflow,
        )
    }

    pub fn views(&self) -> &Views {
        &self.views
    }

    pub fn stack(&self) -> StackRef<'_> {
        self.frames.stack()
    }

    pub fn frames(&self) -> &LiveFrames {
        &self.frames
    }

    /// The views and frames with the object and its update sums.
    pub fn parts_mut(&mut self) -> ScanParts<'_> {
        let (pixels, sums) = self.obj.pixels_and_sums_mut();
        ScanParts {
            views: &self.views,
            frames: &self.frames,
            pixels,
            sums,
        }
    }

    pub fn obj(&self) -> &LiveObject {
        &self.obj
    }

    /// Views `bin_views` found no slot for; `push` keeps it 0.
    pub fn overflow(&self) -> &DeviceBuffer<u32> {
        &self.overflow
    }

    /// Frames refused or pushed after the scan closed.
    pub fn lost(&self) -> u64 {
        self.lost
    }

    pub fn closed(&self) -> Option<Refusal> {
        self.closed
    }

    pub fn landed(&self) -> u32 {
        self.frames.landed()
    }

    pub fn dims(&self) -> &ViewDims {
        &self.dims
    }

    pub fn stream(&self) -> &Arc<Stream> {
        self.frames.stream()
    }
}

#[cfg(test)]
mod tests {
    use cuda_bindings as sys;
    use cuda_core::{Context, IntoResult};
    use ptyrs_gpu::StackCtl;

    use super::*;
    use crate::graph::LiveGraph;

    fn shape(nframe: u32, reach: [u32; 2]) -> ScanShape {
        ScanShape {
            nframe,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [8, 8],
            reach,
            quad_phase: 0.0,
        }
    }

    /// Both probe modes of frame `f` at one raster position, with
    /// fractional offsets on some frames; the raster starts on the row at
    /// `reach[0]` and its last column sits at `reach[1]`.
    fn raster(f: u32, reach: [u32; 2]) -> [ViewAddr; 2] {
        let (cols, step) = (6u32, [reach[0] / 4, reach[1] / 5]);
        let (r, c) = (4 - (f / cols) % 5, f % cols);
        let frac = |k: u32| [0.0, 0.25, 0.5, 0.75][(k % 4) as usize];
        let v = ViewAddr {
            oy: (r * step[0]).min(reach[0]),
            ox: (c * step[1]).min(reach[1]),
            fy: frac(f),
            fx: frac(f / 2 + 1),
            probe: 0,
            obj: 0,
        };
        [v, ViewAddr { probe: 1, ..v }]
    }

    /// Backs `views`, then pushes them.
    fn push(scan: &mut LiveScan, b: &mut ScanBacker, views: &[ViewAddr]) -> Result<u32, GrowError> {
        b.back(views)?;
        scan.push(views)
    }

    fn read_ctl(scan: &LiveScan) -> StackCtl {
        let mut got = StackCtl { f0: 0, nf: 0 };
        scan.stream().synchronize().unwrap();
        // SAFETY: the stream is idle and the entry is one StackCtl.
        unsafe { sys::cuMemcpyDtoH_v2((&raw mut got).cast(), scan.stack().ctl(), 8) }
            .result()
            .unwrap();
        got
    }

    /// Frames pushed in uneven batches, with views on the last row and
    /// column and fractional offsets there, are binned and gathered by one
    /// graph captured before the first push, with no wait between batches.
    /// The device tile lists match the host lists of `view_tiles`, no slab
    /// overflows, and the object sums cover exactly the pixels the views
    /// read. A frame with a view past the reach closes the scan: it and
    /// every later frame are lost, the frames before it land.
    #[test]
    fn pushed_frames_reach_the_far_edge_through_one_graph() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let k = Kernels::load(&ctx).unwrap();
        let reach = [40, 70];
        let sh = shape(30, reach);
        let dims = sh.dims();
        assert_eq!(dims.obj, [49, 79]);
        let nf_max = NonZeroU32::new(4).unwrap();
        let fill = [0.75f32, -0.5];
        let mut scan = LiveScan::new(&s, &sh, nf_max, 64, fill, 1).unwrap();
        let n = 64usize;
        let one = C32 { re: 1.0, im: 0.0 };
        let probe = DeviceBuffer::from_host(&s, &vec![one; 2 * n]).unwrap();
        let mut psi = DeviceBuffer::<C32>::zeroed(&s, 4 * 2 * n).unwrap();
        let npx = (dims.obj[0] * dims.obj[1]) as usize;
        let graph = LiveGraph::capture(&s, |c| {
            scan.advance(&k, c)?;
            let p = scan.parts_mut();
            let st = p.frames.stack();
            k.obj_probe_to_psi(c, p.views, st, p.pixels, &probe, &mut psi)?;
            k.psi_to_obj(c, p.views, st, &psi, &probe, p.sums)
        })
        .unwrap();

        let views: Vec<ViewAddr> = (0..32).flat_map(|f| raster(f, reach)).collect();
        let mut bad = views.clone();
        bad[2 * 27] = ViewAddr {
            ox: reach[1] + 1,
            ..bad[2 * 27]
        };
        let (mut f, mut covered, mut backer) = (0usize, 0u32, scan.backer());
        for b in [1usize, 7, 3, 9, 5, 4, 3] {
            let got = push(&mut scan, &mut backer, &bad[2 * f..2 * (f + b)]).unwrap();
            f += b;
            while covered < scan.landed() {
                graph.replay().unwrap();
                covered = (covered + nf_max.get()).min(scan.landed());
            }
            assert_eq!(
                got,
                (f.min(27) - (f - b).min(27)) as u32,
                "batch ending {f}"
            );
        }
        assert_eq!(scan.landed(), 27);
        assert_eq!(scan.lost(), 32 - 27);
        assert_eq!(
            scan.closed(),
            Some(Refusal::View {
                frame: 27,
                view: bad[2 * 27]
            })
        );
        // The last replay covered frames 25 and 26, the last landed.
        assert_eq!(read_ctl(&scan), StackCtl { f0: 25, nf: 2 });
        assert_eq!(scan.overflow().to_host_vec(&s).unwrap(), vec![0]);

        // Tile lists: per tile, the ascending landed views view_tiles puts on it.
        let views = &views[..2 * 27];
        let [nty, ntx] = dims.tiles();
        let mut want = vec![Vec::new(); (nty * ntx) as usize];
        for (i, v) in views.iter().enumerate() {
            for [ty, tx] in view_tiles(&dims, v) {
                want[(ty * ntx + tx) as usize].push(i as u32);
            }
        }
        let len = scan.views().tile_len().to_host_vec(&s).unwrap();
        let list = scan.views().tile_views().cu_deviceptr();
        for (t, w) in want.iter().enumerate() {
            assert_eq!(len[t] as usize, w.len(), "tile {t}");
            let mut run = vec![0u32; w.len()];
            // SAFETY: the stream is idle and a tile with views has its slab backed.
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    run.as_mut_ptr().cast(),
                    list + (t * 64 * 4) as u64,
                    run.len() * 4,
                )
            }
            .result()
            .unwrap();
            assert_eq!(&run, w, "tile {t}");
        }

        // Sums: den > 0 exactly on the pixels the views read, num = fill * den.
        let mut read = vec![false; npx];
        for v in views {
            let rows = v.oy..v.oy + 8 + (v.fy > 0.0) as u32;
            let cols = v.ox..v.ox + 8 + (v.fx > 0.0) as u32;
            for y in rows {
                for x in cols.clone() {
                    read[(y * dims.obj[1] + x) as usize] = true;
                }
            }
        }
        assert!(read[npx - 1], "a landed view reads the far corner pixel");
        // Every tile row has a view, so the sums are backed whole.
        let sums = scan.obj().sums();
        let den = sums.den.to_host_vec(&s).unwrap();
        let num = sums.num.to_host_vec(&s).unwrap();
        for p in 0..npx {
            assert_eq!(den[p] > 0.0, read[p], "pixel {p}");
            let want = [fill[0] * den[p], fill[1] * den[p]];
            let err = (num[p].re - want[0]).abs() + (num[p].im - want[1]).abs();
            assert!(
                err <= 1e-5 * den[p].max(1.0),
                "pixel {p}: {:?} vs {want:?}",
                num[p]
            );
        }
    }

    /// A frame past the scan, one with no free slab slot or one pushed
    /// unbacked closes the scan the same way; nothing lands afterwards.
    #[test]
    fn a_full_slab_or_a_frame_past_the_scan_closes_it() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let one = NonZeroU32::new(1).unwrap();
        let v = raster(0, [16, 16]);
        let three: Vec<ViewAddr> = (0..3).flat_map(|_| v).collect();

        let mut scan = LiveScan::new(&s, &shape(2, [16, 16]), one, 64, [1.0, 0.0], 1).unwrap();
        let mut b = scan.backer();
        assert_eq!(push(&mut scan, &mut b, &three).unwrap(), 2);
        assert_eq!(scan.closed(), Some(Refusal::PastScan { frame: 2 }));
        assert_eq!((scan.landed(), scan.lost()), (2, 1));
        // Backing past the scan backs nothing more.
        assert_eq!(push(&mut scan, &mut b, &three).unwrap(), 0);

        // Both modes of a frame take a slot each: two frames fill a 4-slot slab.
        let mut scan = LiveScan::new(&s, &shape(5, [16, 16]), one, 4, [1.0, 0.0], 1).unwrap();
        let mut b = scan.backer();
        assert_eq!(push(&mut scan, &mut b, &three).unwrap(), 2);
        let tile = view_tiles(scan.dims(), &v[0]).next().unwrap();
        assert_eq!(scan.closed(), Some(Refusal::Slab { frame: 2, tile }));
        assert_eq!(push(&mut scan, &mut b, &three[..2]).unwrap(), 0);
        assert_eq!((scan.landed(), scan.lost()), (2, 2));
        assert!(matches!(
            push(&mut scan, &mut b, &three[..1]),
            Err(GrowError::Gpu(_))
        ));

        let mut scan = LiveScan::new(&s, &shape(5, [16, 16]), one, 64, [1.0, 0.0], 1).unwrap();
        assert!(matches!(
            scan.push(&three[..2]),
            Err(GrowError::Unbacked { .. })
        ));
        assert_eq!(scan.closed(), Some(Refusal::Failed { frame: 0 }));
        assert_eq!((scan.landed(), scan.lost()), (0, 1));
        assert_eq!(scan.obj().mapped_bytes(), 0);
    }

    /// A frame whose modes are not probe `m % nprobe` x object `m / nprobe`
    /// at one position closes the scan as a refused view, like a view
    /// outside the object: the kernels index modes by that layout.
    #[test]
    fn a_frame_whose_views_are_not_canonical_closes_the_scan() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let one = NonZeroU32::new(1).unwrap();
        let [a, b] = raster(1, [16, 16]);
        let moved = ViewAddr { ox: b.ox + 1, ..b };
        let shifted = ViewAddr { fx: 0.0, ..b };
        for (bad, view) in [
            ([b, a], b),
            ([a, moved], moved),
            ([a, shifted], shifted),
            ([a, a], a),
        ] {
            let mut scan = LiveScan::new(&s, &shape(3, [16, 16]), one, 64, [1.0, 0.0], 1).unwrap();
            let mut backer = scan.backer();
            let views = [a, b, bad[0], bad[1], a, b];
            assert_eq!(push(&mut scan, &mut backer, &views).unwrap(), 1, "{bad:?}");
            assert_eq!(scan.closed(), Some(Refusal::View { frame: 1, view }));
            assert_eq!((scan.landed(), scan.lost()), (1, 2));
        }
    }

    /// A plan backed before the first frame maps what backing its frames as
    /// pushed maps; its frames then push without mapping more, a frame past
    /// it grows the object, and an empty plan maps nothing.
    #[test]
    fn a_planned_scan_grows_only_past_its_plan() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let one = NonZeroU32::new(1).unwrap();
        // About four 2 MiB chunks of object rows.
        let sh = shape(12, [1024, 1024]);
        let at = |f: u32, oy| {
            let v = ViewAddr {
                oy,
                ox: f * 64,
                fy: 0.0,
                fx: 0.0,
                probe: 0,
                obj: 0,
            };
            [v, ViewAddr { probe: 1, ..v }]
        };
        // Ten frames on the top rows are planned; two on the bottom are not.
        let views: Vec<ViewAddr> = (0..10)
            .flat_map(|f| at(f, 0))
            .chain((10..12).flat_map(|f| at(f, 1024)))
            .collect();
        let (plan, past) = views.split_at(20);
        let new = || LiveScan::new(&s, &sh, one, 64, [1.0, 0.0], 1).unwrap();
        let mapped = |scan: &LiveScan| (scan.obj().mapped_bytes(), scan.addr.mapped_bytes());

        let mut scan = new();
        let mut b = scan.backer();
        b.plan(plan).unwrap();
        let planned = mapped(&scan);
        let pushed = new();
        pushed.backer().back(plan).unwrap();
        assert_eq!(planned, mapped(&pushed));
        assert!(planned.0 > 0 && planned.1 > 0, "{planned:?}");
        assert_eq!(push(&mut scan, &mut b, &plan[..6]).unwrap(), 3);
        assert_eq!(push(&mut scan, &mut b, &plan[6..]).unwrap(), 7);
        assert_eq!(mapped(&scan), planned);
        assert_eq!(push(&mut scan, &mut b, past).unwrap(), 2);
        assert!(mapped(&scan).0 > planned.0, "{:?}", mapped(&scan));
        assert_eq!((scan.landed(), scan.lost()), (12, 0));

        let empty = new();
        empty.backer().plan(&[]).unwrap();
        assert_eq!(mapped(&empty), (0, 0));
    }

    /// A slab of `tile_cap` slots takes every view; one slot fewer refuses
    /// the last frame. A view off the object adds to no tile.
    #[test]
    fn the_tile_cap_is_the_smallest_slab_that_takes_every_view() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let one = NonZeroU32::new(1).unwrap();
        let sh = shape(5, [16, 16]);
        let v = raster(0, [16, 16]);
        let three: Vec<ViewAddr> = (0..3).flat_map(|_| v).collect();
        let cap = sh.tile_cap(&three);
        assert_eq!(cap, 6);
        let off = ViewAddr {
            ox: u32::MAX,
            ..v[0]
        };
        assert_eq!(sh.tile_cap(&[three.as_slice(), &[off, off]].concat()), cap);
        let mut scan = LiveScan::new(&s, &sh, one, cap, [1.0, 0.0], 1).unwrap();
        let mut b = scan.backer();
        assert_eq!(push(&mut scan, &mut b, &three).unwrap(), 3);
        assert_eq!(scan.closed(), None);
        let mut scan = LiveScan::new(&s, &sh, one, cap - 1, [1.0, 0.0], 1).unwrap();
        let mut b = scan.backer();
        assert_eq!(push(&mut scan, &mut b, &three).unwrap(), 2);
        assert!(matches!(
            scan.closed(),
            Some(Refusal::Slab { frame: 2, .. })
        ));
    }
}
