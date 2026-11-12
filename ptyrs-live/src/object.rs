//! The live object and its spatial bins (DESIGN.md §3 B2, B3, B6): both keep
//! their device addresses while the scan grows, so one captured graph covers
//! the whole measurement.
//!
//! The object is `[nobj, nyo, nxo]` interleaved complex f32, row-major, the
//! kernels' layout; rows are backed in whole tile rows as views reach them,
//! and fresh pixels start at a flat value. Its update sums, pixel for pixel
//! in the same layout, are backed with those rows and start at zero. The
//! bins are the `tile_start`, `tile_views` and `tile_len` of
//! [`ptyrs_gpu::Views`], in storage the live side owns: tile
//! `t = ty * ntx + tx` of [`OBJ_TILE`] pixels per side has the `cap` slots
//! `tile_views[t * cap..]`, and its run is the first `tile_len[t]` of them,
//! appended on the device by `bin_views`. Slots past the run are never read. [`LiveObject::ensure`] backs a tile before any
//! view on it is binned, so `tile_len[t] > 0` only for backed tiles.
//!
//! Every access goes through the storage's lease, on the object's stream.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_bindings::{self as sys, CUdeviceptr};
use cuda_core::{Context, DeviceBuffer, DriverError, Event, IntoResult, PinnedHostBuffer, Stream};
use ptyrs_gpu::{C32, OBJ_TILE, UpdateSums};

use crate::grow::{GrowBuf, GrowError};

/// The `tile_start`, `tile_views` and `tile_len` of [`ptyrs_gpu::Views`].
pub struct Bins {
    pub start: DeviceBuffer<u32>,
    pub views: DeviceBuffer<u32>,
    pub len: DeviceBuffer<u32>,
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
    tile_len: GrowBuf,
    tile_start: GrowBuf,
    nobj: usize,
    shape: [usize; 2],
    tiles: [usize; 2],
    cap: usize,
    fill: [f32; 2],
    stream: Arc<Stream>,
}

impl LiveObject {
    /// Reserves `nobj` planes of `shape` pixels, their update sums and `cap`
    /// view slots per tile, backed `chunk` bytes at a time; fills and copies
    /// run on `stream`. Nothing of the object, the sums or the slabs is
    /// backed yet.
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
        // only, and `ensure` backs those before a view on them is binned.
        let (pixels, n, d) = unsafe { (obj.typed(len)?, num.typed(len)?, den.typed(len)?) };
        let sums = UpdateSums::new(n, d, stream)?;
        let mut o = Self {
            obj,
            pixels,
            num,
            den,
            sums,
            tile_views: GrowBuf::new(ctx, ntiles * cap * 4, chunk)?,
            tile_len: GrowBuf::new(ctx, ntiles * 4, 1)?,
            tile_start: GrowBuf::new(ctx, (ntiles + 1) * 4, 1)?,
            nobj,
            shape,
            tiles,
            cap,
            fill,
            stream: stream.clone(),
        };
        o.tile_len.map(0..ntiles * 4)?;
        o.tile_start.map(0..(ntiles + 1) * 4)?;
        let starts: Vec<u32> = (0..=ntiles).map(|t| (t * cap) as u32).collect();
        let starts = PinnedHostBuffer::from_slice(ctx, &starts)?;
        let lease = ExecutionContext::new(stream.clone());
        ptyrs_gpu::lease(&lease, &[], &[o.tile_len.storage(), o.tile_start.storage()])?;
        ctx.bind_to_thread()?;
        let s = stream.cu_stream();
        // SAFETY: both destinations were mapped above with these sizes; the
        // stream is synchronised before `starts` is freed.
        unsafe {
            sys::cuMemsetD32Async(o.tile_len.ptr(), 0, ntiles, s).result()?;
            sys::cuMemcpyHtoDAsync_v2(
                o.tile_start.ptr(),
                starts.as_ptr().cast(),
                starts.len() * 4,
                s,
            )
            .result()?;
        }
        stream.synchronize()?;
        Ok(o)
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

    /// Typed views of the bins for `Views::from_parts`: `tile_start[t] = t *
    /// cap` and `tile_len` zeroed at [`Self::new`].
    ///
    /// # Safety
    /// Work through the views may only touch the slab of a tile
    /// [`Self::ensure`] backed.
    pub unsafe fn bins(&self) -> Result<Bins, DriverError> {
        let ntiles = self.tiles[0] * self.tiles[1];
        // SAFETY: `tile_start` and `tile_len` are backed whole; the slabs
        // are forwarded to the caller.
        unsafe {
            Ok(Bins {
                start: self.tile_start.typed(ntiles + 1)?,
                views: self.tile_views.typed(ntiles * self.cap)?,
                len: self.tile_len.typed(ntiles)?,
            })
        }
    }

    pub fn nobj(&self) -> usize {
        self.nobj
    }

    /// `[nyo, nxo]`.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// `[nty, ntx]` tiles of [`OBJ_TILE`] pixels per side.
    pub fn tiles(&self) -> [usize; 2] {
        self.tiles
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Object bytes backed so far, every plane.
    pub fn mapped_bytes(&self) -> usize {
        self.obj.mapped_bytes()
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Backs the object rows with their sums and the slab of each tile
    /// `[ty, tx]` of `tiles`, in every plane, and queues the fill of fresh
    /// pixels and the zeroing of fresh sums on the stream, so kernels queued
    /// there afterwards may read the pixels, add to the sums and append to
    /// the slabs. Call it with [`ptyrs_gpu::view_tiles`] of a view before the
    /// view enters the address table: those are the slabs `bin_views`
    /// writes it into and the tiles whose pixels the kernels touch for it.
    pub fn ensure(&mut self, tiles: impl IntoIterator<Item = [u32; 2]>) -> Result<(), GrowError> {
        let [nyo, nxo] = self.shape;
        let [nty, ntx] = self.tiles;
        let (t, slab) = (OBJ_TILE as usize, self.cap * 4);
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
        // SAFETY: callers pass a chunk `map_with` just backed.
        let zero = |at: CUdeviceptr, r: Range<usize>| unsafe {
            sys::cuMemsetD32Async(at + r.start as u64, 0, r.len() / 4, s).result()
        };
        let mut last_row = None;
        for [ty, tx] in tiles {
            let (y, x) = (ty as usize, tx as usize);
            if y >= nty || x >= ntx {
                return Err(GrowError::Tile {
                    tile: [ty, tx],
                    tiles: self.tiles,
                });
            }
            if last_row != Some(y) {
                let rows = y * t..((y + 1) * t).min(nyo);
                for z in 0..self.nobj {
                    let px = (z * nyo + rows.start) * nxo..(z * nyo + rows.end) * nxo;
                    self.obj.map_with(px.start * 8..px.end * 8, |r| {
                        leased()?;
                        let dst = obj + r.start as u64;
                        // SAFETY: `r` was just mapped; chunks are 8-byte aligned.
                        unsafe {
                            sys::cuMemsetD2D32Async(dst, 8, re, 1, r.len() / 8, s).result()?;
                            sys::cuMemsetD2D32Async(dst + 4, 8, im, 1, r.len() / 8, s).result()?;
                        }
                        Ok(())
                    })?;
                    self.num.map_with(px.start * 8..px.end * 8, |r| {
                        leased()?;
                        Ok(zero(num, r)?)
                    })?;
                    self.den.map_with(px.start * 4..px.end * 4, |r| {
                        leased()?;
                        Ok(zero(den, r)?)
                    })?;
                }
                last_row = Some(y);
            }
            let at = (y * ntx + x) * slab;
            self.tile_views.map(at..at + slab)?;
        }
        Ok(())
    }

    /// Queues a copy of every backed object row into `snap` on the stream,
    /// behind the work already queued there, and returns without waiting
    /// for it; [`Snapshot::wait`] reads the result.
    pub fn snapshot(&self, snap: &mut Snapshot) -> Result<(), GrowError> {
        assert_eq!(
            (snap.nobj, snap.shape),
            (self.nobj, self.shape),
            "snapshot of a differently shaped object"
        );
        let [nyo, nxo] = self.shape;
        let row = nxo * 8;
        let mut bands: Vec<Range<usize>> = Vec::new();
        for r in (0..self.nobj * nyo).filter(|r| self.obj.is_mapped(r * row..(r + 1) * row)) {
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
            // SAFETY: `b` is backed and stays so while `self` lives; `snap`
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
        let [nyo, nxo] = o.shape;
        Ok(Self {
            buf: PinnedHostBuffer::zeroed(ctx, o.nobj * nyo * nxo)?,
            done: sleeping_event(ctx)?,
            copied: Vec::new(),
            fill: o.fill,
            nobj: o.nobj,
            shape: o.shape,
        })
    }

    /// Waits for the last queued copy and returns the object's pixels in its
    /// layout. Rows the device has not backed read as the flat fill value,
    /// which is what backing them would have given.
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

    /// A snapshot is queued behind work the stream has not reached, without
    /// waiting for it, copies backed rows only and reads the rest as fill.
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
        o.ensure(rows(300, 340)).unwrap();
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
        o.ensure(rows(700, 740)).unwrap();
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
    /// else, and read zero (as fresh driver memory does even unfilled).
    #[test]
    fn the_sums_are_backed_and_zeroed_with_the_object_rows() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (n, mib) = (1024usize, 1usize << 20);
        let mut o = LiveObject::new(&s, 2, [n, n], 4, [1.0, 0.5], 1).unwrap();
        o.ensure((300 / 16..=339 / 16).map(|ty| [ty, 2])).unwrap();
        // 2 MiB chunks: rows 256..512 of each plane of the object and
        // `num`, rows 0..512 of `den`.
        assert_eq!(o.mapped_bytes(), 4 * mib);
        assert_eq!(
            (o.num.mapped_bytes(), o.den.mapped_bytes()),
            (4 * mib, 4 * mib)
        );
        s.synchronize().unwrap();
        for z in 0..2 {
            let rows =
                |r: Range<usize>, b: usize| (z * n + r.start) * n * b..(z * n + r.end) * n * b;
            for (buf, r, b) in [(&o.num, 256..512, 8), (&o.den, 0..512, 4)] {
                let r = rows(r, b);
                assert!(buf.is_mapped(r.clone()) && !buf.is_mapped(r.end..r.end + 4));
                let mut got = vec![1u32; r.len() / 4];
                // SAFETY: the stream is idle and `r` is backed.
                unsafe {
                    sys::cuMemcpyDtoH_v2(
                        got.as_mut_ptr().cast(),
                        buf.ptr() + r.start as u64,
                        r.len(),
                    )
                }
                .result()
                .unwrap();
                assert!(
                    got.iter().all(|&w| w == 0),
                    "plane {z}, {} bytes from {}",
                    r.len(),
                    r.start
                );
            }
        }
    }
}
