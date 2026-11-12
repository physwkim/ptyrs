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
//! `tile_views` slabs of [`ptyrs_gpu::Views`] are in storage the live side
//! owns: tile
//! `t = ty * ntx + tx` of [`OBJ_TILE`] pixels per side has the `cap` slots
//! `tile_views[t * cap..]`, and its run is the first `tile_len[t]` of them,
//! appended on the device by `bin_views`. Slots past the run are never read.
//! [`LiveObject::init_tiles`] readies a backed tile before any view on it is
//! binned, so `tile_len[t] > 0` only for ready tiles.
//!
//! Every access goes through the storage's lease, on the object's stream.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_bindings::{self as sys, CUdeviceptr};
use cuda_core::{Context, DeviceBuffer, DriverError, Event, IntoResult, PinnedHostBuffer, Stream};
use ptyrs_gpu::{C32, OBJ_TILE, UpdateSums};

use crate::grow::{Backer, GrowBuf, GrowError};

/// Where a [`LiveObject`]'s tiles sit in its buffers.
#[derive(Clone, Copy)]
struct Layout {
    nobj: usize,
    /// `[nyo, nxo]`.
    shape: [usize; 2],
    /// `[nty, ntx]`.
    tiles: [usize; 2],
    cap: usize,
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

    /// The bytes of tile `[ty, tx]`'s slab in `tile_views`.
    fn slab(&self, [ty, tx]: [usize; 2]) -> Range<usize> {
        let (at, slab) = ((ty * self.tiles[1] + tx) * self.cap * 4, self.cap * 4);
        at..at + slab
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
    stream: Arc<Stream>,
}

impl LiveObject {
    /// Reserves `nobj` planes of `shape` pixels, their update sums and `cap`
    /// view slots per tile, backed `chunk` bytes at a time by
    /// [`Self::backer`]; fills and copies run on `stream`. Nothing of the
    /// object, the sums or the slabs is backed yet.
    pub fn new(
        stream: &Arc<Stream>,
        nobj: usize,
        shape: [usize; 2],
        cap: usize,
        fill: [f32; 2],
        chunk: usize,
    ) -> Result<Self, GrowError> {
        let ctx = stream.context();
        let [nyo, nxo] = shape;
        let t = OBJ_TILE as usize;
        let tiles = [nyo.div_ceil(t), nxo.div_ceil(t)];
        let ntiles = tiles[0] * tiles[1];
        assert!(
            ntiles * cap <= u32::MAX as usize,
            "{ntiles} tiles of {cap} slots pass u32 offsets"
        );
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
            tile_views: GrowBuf::new(ctx, ntiles * cap * 4, chunk)?,
            layout: Layout {
                nobj,
                shape,
                tiles,
                cap,
            },
            fill,
            stream: stream.clone(),
        })
    }

    /// A backer of this object's tiles, for another thread.
    pub fn backer(&self) -> ObjectBacker {
        let l = self.layout;
        ObjectBacker {
            obj: self.obj.backer(),
            num: self.num.backer(),
            den: self.den.backer(),
            tile_views: self.tile_views.backer(),
            layout: l,
            rows: vec![false; l.tiles[0]],
            slabs: vec![false; l.tiles[0] * l.tiles[1]],
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

    /// The slabs, typed for `Views::from_parts` with [`Self::cap`].
    ///
    /// # Safety
    /// Work through the views may only touch the slab of a tile
    /// [`Self::init_tiles`] readied.
    pub unsafe fn tile_views(&self) -> Result<DeviceBuffer<u32>, DriverError> {
        let ntiles = self.layout.tiles[0] * self.layout.tiles[1];
        // SAFETY: forwarded to the caller.
        unsafe { self.tile_views.typed(ntiles * self.layout.cap) }
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

    pub fn cap(&self) -> usize {
        self.layout.cap
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
    /// Call it with [`ptyrs_gpu::view_tiles`] of a view before the view
    /// enters the address table: those are the slabs `bin_views` writes it
    /// into and the tiles whose pixels the kernels touch for it.
    pub fn init_tiles(
        &mut self,
        tiles: impl IntoIterator<Item = [u32; 2]>,
    ) -> Result<(), GrowError> {
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
                ptyrs_gpu::lease(&c, &[], &writes)?;
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

    /// Queues a copy of every ready object row into `snap` on the stream,
    /// behind the work already queued there, and returns without waiting
    /// for it; [`Snapshot::wait`] reads the result.
    pub fn snapshot(&self, snap: &mut Snapshot) -> Result<(), GrowError> {
        let l = self.layout;
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
        ptyrs_gpu::lease(&lease, &[self.pixels.storage()], &[])?;
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
        // drops release.
        let _ = self.stream.synchronize();
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
        let l = o.layout;
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
