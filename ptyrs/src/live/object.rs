//! The live object and its spatial bins (DESIGN.md §3 B2, B3, B6): both keep
//! their device addresses while the scan grows, so one captured graph covers
//! the whole measurement.
//!
//! The object is `[nobj, nyo, nxo]` interleaved complex f32, row-major, the
//! kernels' layout; rows are backed in whole tile rows by an
//! [`ObjectBacker`], those the scan's plan reaches before its first frame and
//! the rest on a thread of its own as views reach them, and fresh pixels
//! start at a flat value. Its update sums, pixel for pixel in the same
//! layout, are backed with those rows and start at zero. The
//! `tile_views` slabs of [`crate::gpu::Views`] are in storage the live side
//! owns: tile
//! `t = ty * ntx + tx` of [`OBJ_TILE`] pixels per side has the slots
//! `tile_views[start[t]..start[t + 1]]`, `cap` for each tile the object bins
//! and none for the others, packed in tile order; its run is the first
//! `tile_len[t]` of them, appended on the device by `bin_views`. Slots past
//! the run are never read. [`LiveObject::init_tiles`] readies a backed tile
//! before any view on it is binned, so `tile_len[t] > 0` only for ready tiles.
//!
//! Every access goes through the storage's lease, on the object's stream.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use crate::gpu::{C32, OBJ_TILE, UpdateSums};
use cuda_async::device_operation::ExecutionContext;
use cuda_bindings::{self as sys, CUdeviceptr};
use cuda_core::{Context, DeviceBuffer, DriverError, Event, IntoResult, PinnedHostBuffer, Stream};

use crate::live::fill::Fill;
use crate::live::grow::{Backer, GrowBuf, GrowError};

/// Where a [`LiveObject`]'s tiles sit in its buffers.
#[derive(Clone)]
struct Layout {
    nobj: usize,
    /// `[nyo, nxo]`.
    shape: [usize; 2],
    /// `[nty, ntx]`.
    tiles: [usize; 2],
    /// Tile `t`'s slab is slots `start[t]..start[t + 1]` of `tile_views`.
    start: Arc<[u32]>,
}

impl Layout {
    /// Tile `t` as indices, if it is in the object.
    fn tile(&self, t: [u32; 2]) -> Result<[usize; 2], GrowError> {
        let [y, x] = t.map(|i| i as usize);
        if y >= self.tiles[0] || x >= self.tiles[1] {
            return Err(GrowError::Tile {
                tile: t,
                tiles: self.tiles,
            });
        }
        Ok([y, x])
    }

    /// The pixels of tile row `ty`, one range per plane.
    fn rows(&self, ty: usize) -> impl Iterator<Item = Range<usize>> {
        let ([nyo, nxo], t) = (self.shape, OBJ_TILE as usize);
        let rows = ty * t..((ty + 1) * t).min(nyo);
        (0..self.nobj).map(move |z| (z * nyo + rows.start) * nxo..(z * nyo + rows.end) * nxo)
    }

    /// The bytes of tile `[ty, tx]`'s slab in `tile_views`, empty for a tile
    /// without one.
    fn slab(&self, [ty, tx]: [usize; 2]) -> Range<usize> {
        let t = ty * self.tiles[1] + tx;
        self.start[t] as usize * 4..self.start[t + 1] as usize * 4
    }
}

/// Backs a [`LiveObject`]'s tiles on any thread, ahead of
/// [`LiveObject::init_tiles`], so the thread queuing work on the object
/// never waits for the driver to grow it.
pub struct ObjectBacker {
    obj: Backer,
    num: Backer,
    den: Backer,
    tile_views: Backer,
    layout: Layout,
    /// Tile rows whose pixels and sums are backed, and tiles whose slab is.
    rows: Vec<bool>,
    slabs: Vec<bool>,
}

impl ObjectBacker {
    /// Backs the object rows with their sums and the slab of each tile
    /// `[ty, tx]` of `tiles`, in every plane: what `init_tiles` of the same
    /// tiles readies.
    pub fn back(&mut self, tiles: impl IntoIterator<Item = [u32; 2]>) -> Result<(), GrowError> {
        for t in tiles {
            let [y, x] = self.layout.tile(t)?;
            if !self.rows[y] {
                for px in self.layout.rows(y) {
                    self.obj.back(px.start * 8..px.end * 8)?;
                    self.num.back(px.start * 8..px.end * 8)?;
                    self.den.back(px.start * 4..px.end * 4)?;
                }
                self.rows[y] = true;
            }
            let i = y * self.layout.tiles[1] + x;
            if !self.slabs[i] {
                self.tile_views.back(self.layout.slab([y, x]))?;
                self.slabs[i] = true;
            }
        }
        Ok(())
    }
}

pub struct LiveObject {
    obj: GrowBuf,
    /// `obj` as the kernels take it, on the same tracker.
    pixels: DeviceBuffer<C32>,
    /// The object's update sums, backed row for row with `obj`.
    num: GrowBuf,
    den: GrowBuf,
    /// `num` and `den` as the kernels take them.
    sums: UpdateSums,
    tile_views: GrowBuf,
    layout: Layout,
    fill: [f32; 2],
    /// Fills whose copies are queued, oldest first. One leaves only through
    /// [`Self::release_fills`], once its event passed, or through `Drop`
    /// behind a stream synchronize: its buffer is never back in the pool
    /// while a copy may still read it.
    held: VecDeque<HeldFill>,
    /// Events of fills that left, for the next ones.
    spare: Vec<Event>,
    fill_copies: u64,
    stream: Arc<Stream>,
}

/// A fill the stream may still be copying from.
struct HeldFill {
    fill: Fill,
    /// Recorded on the stream behind the fill's copies.
    done: Event,
    /// False until `done` is recorded: an event never recorded reads as
    /// passed.
    recorded: bool,
}

impl LiveObject {
    /// [`Self::owned`] with a slab on every tile.
    pub fn new(
        stream: &Arc<Stream>,
        nobj: usize,
        shape: [usize; 2],
        cap: usize,
        fill: [f32; 2],
        chunk: usize,
    ) -> Result<Self, GrowError> {
        let [nty, ntx] = shape.map(|n| n.div_ceil(OBJ_TILE as usize));
        let all: Vec<u32> = (0..(nty * ntx) as u32).collect();
        Self::owned(stream, nobj, shape, cap, fill, chunk, &all)
    }

    /// Reserves `nobj` planes of `shape` pixels, their update sums and `cap`
    /// view slots for each tile of `own`, ascending tile ids, backed `chunk`
    /// bytes at a time by [`Self::backer`]; fills and copies run on
    /// `stream`. Nothing of the object, the sums or the slabs is backed yet.
    pub fn owned(
        stream: &Arc<Stream>,
        nobj: usize,
        shape: [usize; 2],
        cap: usize,
        fill: [f32; 2],
        chunk: usize,
        own: &[u32],
    ) -> Result<Self, GrowError> {
        let ctx = stream.context();
        let [nyo, nxo] = shape;
        let t = OBJ_TILE as usize;
        let tiles = [nyo.div_ceil(t), nxo.div_ceil(t)];
        let ntiles = tiles[0] * tiles[1];
        let bad = |m: String| GrowError::from(crate::gpu::Error::Shape(m));
        let (mut owned, mut start) = (own.iter().peekable(), vec![0u32]);
        for t in 0..ntiles as u32 {
            let slab = if owned.next_if_eq(&&t).is_some() {
                cap
            } else {
                0
            };
            let end = start[t as usize] as usize + slab;
            let end = u32::try_from(end).map_err(|_| {
                bad(format!(
                    "{} slabs of {cap} slots pass u32 offsets",
                    own.len()
                ))
            })?;
            start.push(end);
        }
        if owned.peek().is_some() {
            return Err(bad(format!(
                "own tiles {own:?} are not ascending ids under {ntiles}"
            )));
        }
        let len = nobj * nyo * nxo;
        let obj = GrowBuf::new(ctx, len * 8, chunk)?;
        let num = GrowBuf::new(ctx, len * 8, chunk)?;
        let den = GrowBuf::new(ctx, len * 4, chunk)?;
        // SAFETY: kernels reach the pixels and sums of tiles with views
        // only, and `init_tiles` readies those before a view on them is
        // binned.
        let (pixels, n, d) = unsafe { (obj.typed(len)?, num.typed(len)?, den.typed(len)?) };
        let sums = UpdateSums::new(n, d, stream)?;
        Ok(Self {
            obj,
            pixels,
            num,
            den,
            sums,
            tile_views: GrowBuf::new(ctx, start[ntiles] as usize * 4, chunk)?,
            layout: Layout {
                nobj,
                shape,
                tiles,
                start: start.into(),
            },
            fill,
            held: VecDeque::new(),
            spare: Vec::new(),
            fill_copies: 0,
            stream: stream.clone(),
        })
    }

    /// A backer of this object's tiles, for another thread.
    pub fn backer(&self) -> ObjectBacker {
        let l = self.layout.clone();
        ObjectBacker {
            obj: self.obj.backer(),
            num: self.num.backer(),
            den: self.den.backer(),
            tile_views: self.tile_views.backer(),
            rows: vec![false; l.tiles[0]],
            slabs: vec![false; l.tiles[0] * l.tiles[1]],
            layout: l,
        }
    }

    /// Object base: pixel `(z, y, x)` at `ptr() + ((z * nyo + y) * nxo + x) * 8`.
    pub fn ptr(&self) -> CUdeviceptr {
        self.obj.ptr()
    }

    /// The object as the kernels take it.
    pub fn pixels(&self) -> &DeviceBuffer<C32> {
        &self.pixels
    }

    /// The object and its update sums, for the kernels that write them.
    pub fn pixels_and_sums_mut(&mut self) -> (&mut DeviceBuffer<C32>, &mut UpdateSums) {
        (&mut self.pixels, &mut self.sums)
    }

    pub fn sums(&self) -> &UpdateSums {
        &self.sums
    }

    /// The slabs, typed for `Views::from_slabs` with [`Self::slab_starts`].
    ///
    /// # Safety
    /// Work through the views may only touch the slab of a tile
    /// [`Self::init_tiles`] readied.
    pub unsafe fn tile_views(&self) -> Result<DeviceBuffer<u32>, DriverError> {
        let n = self.layout.start[self.layout.start.len() - 1] as usize;
        // SAFETY: forwarded to the caller.
        unsafe { self.tile_views.typed(n) }
    }

    /// Tile `t`'s slab is slots `start[t]..start[t + 1]` of [`Self::tile_views`].
    pub fn slab_starts(&self) -> &[u32] {
        &self.layout.start
    }

    pub fn nobj(&self) -> usize {
        self.layout.nobj
    }

    /// `[nyo, nxo]`.
    pub fn shape(&self) -> [usize; 2] {
        self.layout.shape
    }

    /// `[nty, ntx]` tiles of [`OBJ_TILE`] pixels per side.
    pub fn tiles(&self) -> [usize; 2] {
        self.layout.tiles
    }

    /// Object bytes backed so far, every plane.
    pub fn mapped_bytes(&self) -> usize {
        self.obj.mapped_bytes()
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Readies the object rows with their sums and the slab of each tile
    /// `[ty, tx]` of `tiles`, in every plane, which a [`Self::backer`]
    /// backed: queues the fill of fresh pixels and the zeroing of fresh sums
    /// on the stream, so kernels queued there afterwards may read the
    /// pixels, add to the sums and append to the slabs. Never backs memory.
    /// Call it with [`crate::gpu::view_tiles`] of a view before the view
    /// enters the address table: those are the slabs `bin_views` writes it
    /// into and the tiles whose pixels the kernels touch for it.
    pub fn init_tiles(
        &mut self,
        tiles: impl IntoIterator<Item = [u32; 2]>,
    ) -> Result<(), GrowError> {
        // Every batch passes here, so a fill is back within one batch of
        // its copies, with or without a next fill.
        self.release_fills();
        self.stream.context().bind_to_thread()?;
        let s = self.stream.cu_stream();
        let [re, im] = self.fill.map(f32::to_bits);
        let (obj, num, den) = (self.obj.ptr(), self.num.ptr(), self.den.ptr());
        let stream = &self.stream;
        let writes = [
            self.pixels.storage(),
            self.sums.num.storage(),
            self.sums.den.storage(),
        ];
        // Leased once, at the first fresh chunk.
        let mut lease: Option<ExecutionContext> = None;
        let mut leased = || -> Result<(), GrowError> {
            if lease.is_none() {
                let c = ExecutionContext::new(stream.clone());
                crate::gpu::lease(&c, &[], &writes)?;
                lease = Some(c);
            }
            Ok(())
        };
        // SAFETY: callers pass a chunk `init_with` found backed.
        let zero = |at: CUdeviceptr, r: Range<usize>| unsafe {
            sys::cuMemsetD32Async(at + r.start as u64, 0, r.len() / 4, s).result()
        };
        let mut last_row = None;
        for t in tiles {
            let [y, x] = self.layout.tile(t)?;
            if last_row != Some(y) {
                for px in self.layout.rows(y) {
                    self.obj.init_with(px.start * 8..px.end * 8, |r| {
                        leased()?;
                        let dst = obj + r.start as u64;
                        // SAFETY: `r` is backed; chunks are 8-byte aligned.
                        unsafe {
                            sys::cuMemsetD2D32Async(dst, 8, re, 1, r.len() / 8, s).result()?;
                            sys::cuMemsetD2D32Async(dst + 4, 8, im, 1, r.len() / 8, s).result()?;
                        }
                        Ok(())
                    })?;
                    self.num.init_with(px.start * 8..px.end * 8, |r| {
                        leased()?;
                        Ok(zero(num, r)?)
                    })?;
                    self.den.init_with(px.start * 4..px.end * 4, |r| {
                        leased()?;
                        Ok(zero(den, r)?)
                    })?;
                }
                last_row = Some(y);
            }
            self.tile_views
                .init_with(self.layout.slab([y, x]), |_| Ok(()))?;
        }
        Ok(())
    }

    /// Queues `fill` onto mode 0 on the stream, behind the work already
    /// queued there, so kernels queued afterwards read the predictions:
    /// call it for a batch after its tiles are readied and before its
    /// first epoch. `origin` is the object pixel `[y, x]` of this object's
    /// pixel `(0, 0)`, `[0, 0]` unless the object is a window of a larger
    /// one; what `fill` holds outside the window is another object's.
    /// Rows not ready are skipped, never backed, and the pixels skipped
    /// are returned: they start flat when they are readied. The object has
    /// one mode: who builds a run with a predictor refuses any other.
    ///
    /// The object keeps `fill` until an event behind its copies passed, so
    /// its buffer cannot be reused under a pending copy, whatever this
    /// call returns.
    pub fn write_fill(&mut self, fill: Fill, origin: [usize; 2]) -> Result<usize, GrowError> {
        self.release_fills();
        self.stream.context().bind_to_thread()?;
        let done = match self.spare.pop() {
            Some(e) => e,
            None => sleeping_event(self.stream.context())?,
        };
        // Held before the first copy is queued.
        self.held.push_back(HeldFill {
            fill,
            done,
            recorded: false,
        });
        let queued = self.queue_fill(origin);
        // Also after a failure: copies queued before it are pending.
        let held = self.held.back_mut().expect("pushed above");
        held.done.record(&self.stream)?;
        held.recorded = true;
        queued
    }

    /// Host-to-device copies [`Self::write_fill`] has queued so far.
    pub fn fill_copies(&self) -> u64 {
        self.fill_copies
    }

    /// Hands back the fills whose copies ran.
    fn release_fills(&mut self) {
        while let Some(h) = self.held.front() {
            if !(h.recorded && matches!(h.done.query(), Ok(true))) {
                break;
            }
            let h = self.held.pop_front().expect("front exists");
            self.spare.push(h.done);
        }
    }

    /// Queues the copies of the newest held fill; returns the pixels on
    /// rows not ready.
    fn queue_fill(&mut self, [oy, ox]: [usize; 2]) -> Result<usize, GrowError> {
        let fill = &self.held.back().expect("write_fill holds one").fill;
        let [nyo, nxo] = self.layout.shape;
        let (s, src) = (self.stream.cu_stream(), fill.values().as_ptr());
        let mut lease: Option<ExecutionContext> = None;
        let (mut skipped, mut copies) = (0, 0);
        for r in fill.rects() {
            // The rectangle inside this object, in its own pixels.
            let clip = |at: u32, len: u32, o: usize, n: usize| {
                let (a, b) = (at as usize, (at + len) as usize);
                a.max(o).min(o + n) - o..b.max(o).min(o + n) - o
            };
            let (ys, xs) = (clip(r.y, r.h, oy, nyo), clip(r.x, r.w, ox, nxo));
            if ys.is_empty() || xs.is_empty() {
                continue;
            }
            let ready = |y: usize| {
                self.obj
                    .is_ready((y * nxo + xs.start) * 8..(y * nxo + xs.end) * 8)
            };
            let mut y = ys.start;
            while y < ys.end {
                if !ready(y) {
                    skipped += xs.len();
                    y += 1;
                    continue;
                }
                let band = y..(y + 1..ys.end).find(|&y| !ready(y)).unwrap_or(ys.end);
                y = band.end;
                if lease.is_none() {
                    let c = ExecutionContext::new(self.stream.clone());
                    crate::gpu::lease(&c, &[], &[self.pixels.storage()])?;
                    lease = Some(c);
                }
                let first =
                    (band.start + oy - r.y as usize) * r.w as usize + xs.start + ox - r.x as usize;
                // SAFETY: integers and raw pointers, zero is a value of
                // each; the fields left zero are the unused ones.
                let mut copy: sys::CUDA_MEMCPY2D = unsafe { std::mem::zeroed() };
                copy.srcMemoryType = sys::CUmemorytype_enum_CU_MEMORYTYPE_HOST;
                // SAFETY: inside the fill's values, by `Rect`.
                copy.srcHost = unsafe { src.add(r.at + first) }.cast();
                copy.srcPitch = r.w as usize * 8;
                copy.dstMemoryType = sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE;
                copy.dstDevice = self.obj.ptr() + ((band.start * nxo + xs.start) * 8) as u64;
                copy.dstPitch = nxo * 8;
                copy.WidthInBytes = xs.len() * 8;
                copy.Height = band.len();
                // SAFETY: the rows are ready and stay so while `self`
                // lives; the source is pinned and held in `self.held`
                // until an event behind this copy passed.
                unsafe { sys::cuMemcpy2DAsync_v2(&copy, s) }.result()?;
                copies += 1;
            }
        }
        self.fill_copies += copies;
        Ok(skipped)
    }

    /// Queues a copy of every ready object row into `snap` on the stream,
    /// behind the work already queued there, and returns without waiting
    /// for it; [`Snapshot::wait`] reads the result.
    pub fn snapshot(&self, snap: &mut Snapshot) -> Result<(), GrowError> {
        let l = &self.layout;
        assert_eq!(
            (snap.nobj, snap.shape),
            (l.nobj, l.shape),
            "snapshot of a differently shaped object"
        );
        let [nyo, nxo] = l.shape;
        let row = nxo * 8;
        let mut bands: Vec<Range<usize>> = Vec::new();
        for r in (0..l.nobj * nyo).filter(|r| self.obj.is_ready(r * row..(r + 1) * row)) {
            match bands.last_mut() {
                Some(b) if b.end == r * nxo => b.end += nxo,
                _ => bands.push(r * nxo..(r + 1) * nxo),
            }
        }
        snap.copied.clear();
        let queued = self.queue_copy(&bands, snap);
        if queued.is_err() {
            // Copies queued before the failure may still be writing `snap`.
            self.stream.synchronize()?;
        }
        queued?;
        snap.copied = bands;
        Ok(())
    }

    fn queue_copy(&self, bands: &[Range<usize>], snap: &mut Snapshot) -> Result<(), GrowError> {
        let lease = ExecutionContext::new(self.stream.clone());
        crate::gpu::lease(&lease, &[self.pixels.storage()], &[])?;
        self.stream.context().bind_to_thread()?;
        let (s, dst) = (self.stream.cu_stream(), snap.buf.as_mut_ptr());
        for b in bands {
            // SAFETY: `b` is ready and stays so while `self` lives; `snap`
            // holds the whole object and is neither read nor freed before
            // `done`, recorded below, completes.
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    dst.add(b.start).cast(),
                    self.obj.ptr() + (b.start * 8) as u64,
                    b.len() * 8,
                    s,
                )
            }
            .result()?;
        }
        Ok(snap.done.record(&self.stream)?)
    }
}

impl Drop for LiveObject {
    fn drop(&mut self) {
        // Queued fills and snapshot copies use the mappings the fields'
        // drops release, and held fills' buffers.
        if self.stream.synchronize().is_err() {
            // Their copies may be pending for good: the buffers stay out
            // of the pool.
            self.held.drain(..).for_each(std::mem::forget);
        }
    }
}

/// An event whose synchronize sleeps the waiting thread. With more CPUs
/// than contexts the driver otherwise spins it, and the live consumer waits
/// on the CPUs the receive runtime runs on.
pub(crate) fn sleeping_event(ctx: &Arc<Context>) -> Result<Event, DriverError> {
    ctx.new_event_with_flags(Some(
        sys::CUevent_flags_enum_CU_EVENT_BLOCKING_SYNC
            | sys::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
    ))
}

/// A [`sleeping_event`] that also records when it completed, for
/// [`Event::elapsed_time`].
pub(crate) fn timed_sleeping_event(ctx: &Arc<Context>) -> Result<Event, DriverError> {
    ctx.new_event_with_flags(Some(sys::CUevent_flags_enum_CU_EVENT_BLOCKING_SYNC))
}

/// A host copy of a [`LiveObject`], filled by [`LiveObject::snapshot`] and
/// read only through [`wait`](Self::wait); dropping it also waits, so a
/// queued copy never writes freed memory.
pub struct Snapshot {
    buf: PinnedHostBuffer<[f32; 2]>,
    done: Event,
    /// Pixel ranges the last queued copy writes, ascending.
    copied: Vec<Range<usize>>,
    fill: [f32; 2],
    nobj: usize,
    shape: [usize; 2],
}

impl Snapshot {
    pub fn new(o: &LiveObject) -> Result<Self, DriverError> {
        let ctx = o.stream.context();
        let l = &o.layout;
        let [nyo, nxo] = l.shape;
        Ok(Self {
            buf: PinnedHostBuffer::zeroed(ctx, l.nobj * nyo * nxo)?,
            done: sleeping_event(ctx)?,
            copied: Vec::new(),
            fill: o.fill,
            nobj: l.nobj,
            shape: l.shape,
        })
    }

    /// Waits for the last queued copy and returns the object's pixels in its
    /// layout. Rows the device has not readied read as the flat fill value,
    /// which is what readying them would have given.
    pub fn wait(&mut self) -> Result<&[[f32; 2]], DriverError> {
        self.done.synchronize()?;
        let px = self.buf.as_mut_slice();
        let end = px.len()..px.len();
        let mut at = 0;
        for b in self.copied.iter().chain([&end]) {
            px[at..b.start].fill(self.fill);
            at = b.end;
        }
        Ok(self.buf.as_slice())
    }

    pub fn nobj(&self) -> usize {
        self.nobj
    }

    /// `[nyo, nxo]`.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        // A queued copy writes `buf` until `done` completes.
        let _ = self.done.synchronize();
    }
}
#[cfg(test)]
mod tests {
    use cuda_core::Context;

    use super::*;
    use crate::live::fill::PatchFill;

    /// Holds `s` behind a gate until the returned sender is used or
    /// dropped, or 10 s passed, so a call that waits on the stream fails
    /// its check instead of hanging. Join the handle before `gate` drops.
    fn hold(
        s: &Stream,
        gate: &mut PinnedHostBuffer<u32>,
    ) -> (std::sync::mpsc::Sender<()>, std::thread::JoinHandle<()>) {
        // SAFETY: the caller keeps `gate` until the stream passed it.
        unsafe {
            sys::cuStreamWaitValue32_v2(
                s.cu_stream(),
                gate.as_ptr() as u64,
                1,
                sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
            )
        }
        .result()
        .unwrap();
        let (release, rx) = std::sync::mpsc::channel::<()>();
        let g = gate.as_mut_ptr() as usize;
        let watchdog = std::thread::spawn(move || {
            let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
            // SAFETY: `gate` outlives the thread, which the caller joins.
            unsafe { (g as *mut u32).write_volatile(1) };
        });
        (release, watchdog)
    }

    const PATCH: [usize; 2] = [8, 8];

    /// Every tile of a 64-pixel object.
    fn all_tiles() -> impl Iterator<Item = [u32; 2]> {
        (0..4).flat_map(|ty| (0..4).map(move |tx| [ty, tx]))
    }

    /// Fills reach the object as stitched: a pixel in one patch takes its
    /// value, one in two patches of a batch their taper mean, one a later
    /// batch covers again keeps the first value and one never covered what
    /// it had; on a ready object nothing is skipped.
    #[test]
    fn write_fill_writes_each_pixel_its_first_prediction() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (n, flat) = (64usize, [1.0f32, 0.5]);
        let mut o = LiveObject::new(&s, 1, [n, n], 4, flat, 1).unwrap();
        grow(&mut o, all_tiles());
        let mut pf = PatchFill::new(&ctx, [n, n], PATCH, 4, 2).unwrap();
        let patch = |v: [f32; 2]| vec![v; 64];
        let (a, b, c) = ([2.0f32, 1.0], [4.0f32, -3.0], [9.0f32, 9.0]);
        let first = pf
            .stitch(&[[30, 20], [30, 23]], &[patch(a), patch(b)].concat())
            .unwrap();
        let mean = first.values()[5];
        assert!(mean != a && mean != b);
        assert_eq!(o.write_fill(first, [0, 0]).unwrap(), 0);
        // Over the first batch's patches and 4 rows below them.
        let later = pf.stitch(&[[34, 21]], &patch(c)).unwrap();
        assert_eq!(o.write_fill(later, [0, 0]).unwrap(), 0);
        assert_eq!(o.fill_copies(), 2);
        let mut snap = Snapshot::new(&o).unwrap();
        o.snapshot(&mut snap).unwrap();
        let px = snap.wait().unwrap();
        let at = |y: usize, x: usize| px[y * n + x];
        assert_eq!(at(31, 21), a);
        assert_eq!(at(31, 29), b);
        assert_eq!(at(33, 25), mean);
        assert_eq!(at(36, 25), mean, "a later batch replaced a prediction");
        assert_eq!(at(40, 25), c);
        let covered = |y: usize, x: usize| {
            ((30..38).contains(&y) && (20..31).contains(&x))
                || ((38..42).contains(&y) && (21..29).contains(&x))
        };
        let moved = (0..n * n)
            .filter(|i| !covered(i / n, i % n) && px[*i] != flat)
            .count();
        assert_eq!(moved, 0);
    }

    /// A fill over rows not ready writes the ready ones, reports the
    /// pixels of the others and backs nothing; in a window it writes its
    /// part of the fill at the window's own pixels.
    #[test]
    fn write_fill_skips_rows_not_ready_and_clips_to_the_window() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (n, flat) = (1024usize, [1.0f32, 0.5]);
        let mut o = LiveObject::new(&s, 1, [n, n], 4, flat, 1).unwrap();
        // 2 MiB chunks back rows 256..512.
        grow(&mut o, [[300 / 16, 2]].into_iter());
        let mapped = o.mapped_bytes();
        let mut pf = PatchFill::new(&ctx, [2 * n, 2 * n], PATCH, 4, 4).unwrap();
        let v = [3.0f32, -2.0];
        // Rows 508..516 of the object: 4 ready, 4 not.
        let fill = pf.stitch(&[[508, 40]], &vec![v; 64]).unwrap();
        assert_eq!(o.write_fill(fill, [0, 0]).unwrap(), 4 * 8);
        assert_eq!(o.mapped_bytes(), mapped);
        // A window at [100, 1020] holds columns 1020..2044: 4 of the 8.
        let fill = pf.stitch(&[[400, 1016]], &vec![v; 64]).unwrap();
        assert_eq!(o.write_fill(fill, [100, 1020]).unwrap(), 0);
        // Wholly outside the window.
        let fill = pf.stitch(&[[2000, 0]], &vec![v; 64]).unwrap();
        assert_eq!(o.write_fill(fill, [100, 1020]).unwrap(), 0);
        assert_eq!(o.fill_copies(), 2);
        let mut snap = Snapshot::new(&o).unwrap();
        o.snapshot(&mut snap).unwrap();
        let px = snap.wait().unwrap();
        let filled = |y: usize, x: usize| {
            ((508..512).contains(&y) && (40..48).contains(&x)) || ((300..308).contains(&y) && x < 4)
        };
        let bad = (0..n * n)
            .filter(|i| px[*i] != if filled(i / n, i % n) { v } else { flat })
            .count();
        assert_eq!(bad, 0);
    }

    /// A fill's buffer is out of the pool while the stream has not run its
    /// copies: through the call that queued them, a later call and the
    /// object's drop, and back once they ran.
    #[test]
    fn write_fill_keeps_the_buffer_until_its_copies_ran() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let n = 64usize;
        let mut o = LiveObject::new(&s, 1, [n, n], 4, [1.0, 0.5], 1).unwrap();
        grow(&mut o, all_tiles());
        s.synchronize().unwrap();
        let mut pf = PatchFill::new(&ctx, [n, n], PATCH, 4, 2).unwrap();
        let patch = vec![[2.0f32, 2.0]; 64];
        let mut gate = PinnedHostBuffer::<u32>::zeroed(&ctx, 1).unwrap();
        let (release, watchdog) = hold(&s, &mut gate);
        let fill = pf.stitch(&[[0, 0]], &patch).unwrap();
        o.write_fill(fill, [0, 0]).unwrap();
        // The second call finds the first one's copies pending.
        let fill = pf.stitch(&[[8, 0]], &patch).unwrap();
        o.write_fill(fill, [0, 0]).unwrap();
        let pending = o.held.iter().all(|h| !h.done.query().unwrap());
        let held = (o.held.len(), pf.stitch(&[[16, 0]], &patch).is_none());
        release.send(()).unwrap();
        watchdog.join().unwrap();
        assert!(pending, "write_fill waited for the stream");
        assert_eq!(held, (2, true), "a buffer was back before its copies ran");
        s.synchronize().unwrap();
        // The copies ran: readying a tile hands both back.
        o.init_tiles([[0, 0]]).unwrap();
        assert_eq!((o.held.len(), o.spare.len()), (0, 2));

        // Dropped behind a held stream, the object keeps its fills until
        // the stream is through.
        let (release, watchdog) = {
            gate.as_mut_slice()[0] = 0;
            hold(&s, &mut gate)
        };
        let fill = pf.stitch(&[[24, 0]], &patch).unwrap();
        o.write_fill(fill, [0, 0]).unwrap();
        let fill = pf.stitch(&[[32, 0]], &patch).unwrap();
        o.write_fill(fill, [0, 0]).unwrap();
        let (dropped, done) = std::sync::mpsc::channel::<()>();
        let dropper = std::thread::spawn(move || {
            drop(o);
            let _ = dropped.send(());
        });
        let waited = done
            .recv_timeout(std::time::Duration::from_millis(200))
            .is_err();
        let kept = pf.stitch(&[[40, 0]], &patch).is_none();
        release.send(()).unwrap();
        watchdog.join().unwrap();
        dropper.join().unwrap();
        assert!(waited && kept, "the drop handed a buffer back early");
        assert!(pf.stitch(&[[40, 0]], &patch).is_some());
    }

    /// Backs `tiles` from another thread, as a mapper would, then readies
    /// them.
    fn grow(o: &mut LiveObject, tiles: impl Iterator<Item = [u32; 2]>) {
        let tiles: Vec<_> = tiles.collect();
        let (mut b, t) = (o.backer(), tiles.clone());
        std::thread::spawn(move || b.back(t))
            .join()
            .unwrap()
            .unwrap();
        o.init_tiles(tiles).unwrap();
    }

    /// A snapshot is queued behind work the stream has not reached, without
    /// waiting for it, copies ready rows only and reads the rest as fill.
    #[test]
    fn snapshot_copies_backed_rows_behind_queued_work() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (n, fill) = (1024usize, [1.0f32, 0.5]);
        let mut o = LiveObject::new(&s, 2, [n, n], 4, fill, 1).unwrap();
        let mut snap = Snapshot::new(&o).unwrap();
        o.snapshot(&mut snap).unwrap();
        assert!(snap.copied.is_empty());
        assert!(snap.wait().unwrap().iter().all(|&p| p == fill));

        // The tiles over rows y0..y1; 2 MiB chunks back 256 rows of a plane.
        let rows = |y0: u32, y1: u32| (y0 / 16..=(y1 - 1) / 16).map(|ty| [ty, 2]);
        // Readying never backs.
        assert!(matches!(
            o.init_tiles(rows(300, 340)),
            Err(GrowError::Unbacked { .. })
        ));
        assert_eq!(o.mapped_bytes(), 0);
        grow(&mut o, rows(300, 340));
        let mut gate = PinnedHostBuffer::<u32>::zeroed(&ctx, 1).unwrap();
        let mark = 7.0f32.to_bits();
        unsafe {
            sys::cuStreamWaitValue32_v2(
                s.cu_stream(),
                gate.as_ptr() as u64,
                1,
                sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
            )
            .result()
            .unwrap();
            // Plane 1, rows 300..340: real part 7.
            let at = o.ptr() + ((n + 300) * n * 8) as u64;
            sys::cuMemsetD2D32Async(at, 8, mark, 1, 40 * n, s.cu_stream())
                .result()
                .unwrap();
        }
        // Opens the gate after 10 s, so a snapshot that waits on the stream
        // fails the check below instead of hanging.
        let (release, rx) = std::sync::mpsc::channel::<()>();
        let g = gate.as_mut_ptr() as usize;
        let watchdog = std::thread::spawn(move || {
            let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
            // SAFETY: `gate` outlives the thread, which is joined below.
            unsafe { (g as *mut u32).write_volatile(1) };
        });
        o.snapshot(&mut snap).unwrap();
        let pending = !snap.done.query().unwrap();
        release.send(()).unwrap();
        watchdog.join().unwrap();
        assert!(
            pending,
            "the copy ran before the queued memset was released"
        );
        assert_eq!(
            snap.copied,
            vec![256 * n..512 * n, (n + 256) * n..(n + 512) * n]
        );
        let px = snap.wait().unwrap();
        let marked = (n + 300) * n..(n + 340) * n;
        let bad = (0..px.len())
            .filter(|i| px[*i] != if marked.contains(i) { [7.0, 0.5] } else { fill })
            .count();
        assert_eq!(bad, 0);

        // Growth adds rows 512..768; touching bands merge.
        grow(&mut o, rows(700, 740));
        o.snapshot(&mut snap).unwrap();
        assert_eq!(
            snap.copied,
            vec![256 * n..768 * n, (n + 256) * n..(n + 768) * n]
        );
        let px = snap.wait().unwrap();
        let bad = (0..px.len())
            .filter(|i| px[*i] != if marked.contains(i) { [7.0, 0.5] } else { fill })
            .count();
        assert_eq!(bad, 0);
    }

    /// The sums are backed with the object rows a tile needs and nowhere
    /// else, and read zero although the fresh backing is dirtied before the
    /// zeroing runs (fresh driver memory reads zero even unfilled).
    #[test]
    fn the_sums_are_backed_and_zeroed_with_the_object_rows() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (n, mib) = (1024usize, 1usize << 20);
        let mut o = LiveObject::new(&s, 2, [n, n], 4, [1.0, 0.5], 1).unwrap();
        let mut gate = PinnedHostBuffer::<u32>::zeroed(&ctx, 1).unwrap();
        // SAFETY: the gate outlives the stream's work, which is synchronized
        // below.
        unsafe {
            sys::cuStreamWaitValue32_v2(
                s.cu_stream(),
                gate.as_ptr() as u64,
                1,
                sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
            )
            .result()
            .unwrap();
        }
        // Opens the gate after 10 s, so an `init_tiles` that waits on the stream
        // fails the check below instead of hanging.
        let (release, rx) = std::sync::mpsc::channel::<()>();
        let g = gate.as_mut_ptr() as usize;
        let watchdog = std::thread::spawn(move || {
            let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
            // SAFETY: `gate` outlives the thread, which is joined below.
            unsafe { (g as *mut u32).write_volatile(1) };
        });
        grow(&mut o, (300 / 16..=339 / 16).map(|ty| [ty, 2]));
        let zeroed = ctx.new_event_with_flags(None).unwrap();
        zeroed.record(&s).unwrap();
        // 2 MiB chunks: rows 256..512 of each plane of the object and
        // `num`, rows 0..512 of `den`.
        assert_eq!(o.mapped_bytes(), 4 * mib);
        assert_eq!(
            (o.num.mapped_bytes(), o.den.mapped_bytes()),
            (4 * mib, 4 * mib)
        );
        let fresh: Vec<_> = (0..2)
            .flat_map(|z| {
                let rows = move |r: Range<usize>, b: usize| {
                    (z * n + r.start) * n * b..(z * n + r.end) * n * b
                };
                [(&o.num, rows(256..512, 8)), (&o.den, rows(0..512, 4))]
            })
            .collect();
        let dirt = ctx.new_stream().unwrap();
        for (buf, r) in &fresh {
            assert!(buf.is_ready(r.clone()) && !buf.is_ready(r.end..r.end + 4));
            // SAFETY: `r` is backed; the object's stream is held at the gate.
            unsafe {
                sys::cuMemsetD32Async(
                    buf.ptr() + r.start as u64,
                    !0,
                    r.len() / 4,
                    dirt.cu_stream(),
                )
            }
            .result()
            .unwrap();
        }
        dirt.synchronize().unwrap();
        let held = !zeroed.query().unwrap();
        release.send(()).unwrap();
        watchdog.join().unwrap();
        assert!(held, "the zeroing ran before the backing was dirtied");
        s.synchronize().unwrap();
        for (buf, r) in fresh {
            let mut got = vec![1u32; r.len() / 4];
            // SAFETY: the stream is idle and `r` is backed.
            unsafe {
                sys::cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), buf.ptr() + r.start as u64, r.len())
            }
            .result()
            .unwrap();
            assert!(
                got.iter().all(|&w| w == 0),
                "{} bytes from {}",
                r.len(),
                r.start
            );
        }
    }
}
