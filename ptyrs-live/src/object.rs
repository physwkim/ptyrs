//! The live object and its spatial bins (DESIGN.md §3 B2, B3, B6): both keep
//! their device addresses while the scan grows, so one captured graph covers
//! the whole measurement.
//!
//! The object is `[nobj, nyo, nxo]` interleaved complex f32, row-major, the
//! M1 kernels' layout; rows are backed in whole tile rows as views reach
//! them, and fresh pixels start at a flat value. The bins are the
//! `tile_start`, `tile_views` and `tile_len` of the M1 `Views`, in storage
//! the live side owns: tile `t = ty * ntx + tx` has the `cap` slots
//! `tile_views[t * cap..]`, and its run is the first `tile_len[t]` of them,
//! appended on the device in ascending view order. Slots past the run are
//! never read. [`LiveObject::ensure`] backs a tile before any view on it is
//! binned, so `tile_len[t] > 0` only for backed tiles.

use std::ops::Range;
use std::sync::Arc;

use cuda_bindings::{self as sys, CUdeviceptr};
use cuda_core::{DriverError, Event, IntoResult, PinnedHostBuffer, Stream};

use crate::grow::{GrowBuf, GrowError};

pub struct LiveObject {
    obj: GrowBuf,
    tile_views: GrowBuf,
    tile_len: GrowBuf,
    tile_start: GrowBuf,
    nobj: usize,
    shape: [usize; 2],
    tile: usize,
    tiles: [usize; 2],
    cap: usize,
    fill: [f32; 2],
    stream: Arc<Stream>,
}

impl LiveObject {
    /// Reserves `nobj` planes of `shape` pixels and `cap` view slots per
    /// `tile x tile` tile, backed `chunk` bytes at a time; fills and copies
    /// run on `stream`. Nothing of the object or the slabs is backed yet.
    pub fn new(
        stream: &Arc<Stream>,
        nobj: usize,
        shape: [usize; 2],
        tile: usize,
        cap: usize,
        fill: [f32; 2],
        chunk: usize,
    ) -> Result<Self, GrowError> {
        let ctx = stream.context();
        let [nyo, nxo] = shape;
        let tiles = [nyo.div_ceil(tile), nxo.div_ceil(tile)];
        let ntiles = tiles[0] * tiles[1];
        assert!(
            ntiles * cap <= u32::MAX as usize,
            "{ntiles} tiles of {cap} slots pass u32 offsets"
        );
        let mut o = Self {
            obj: GrowBuf::new(ctx, nobj * nyo * nxo * 8, chunk)?,
            tile_views: GrowBuf::new(ctx, ntiles * cap * 4, chunk)?,
            tile_len: GrowBuf::new(ctx, ntiles * 4, 1)?,
            tile_start: GrowBuf::new(ctx, (ntiles + 1) * 4, 1)?,
            nobj,
            shape,
            tile,
            tiles,
            cap,
            fill,
            stream: stream.clone(),
        };
        o.tile_len.map(0..ntiles * 4)?;
        o.tile_start.map(0..(ntiles + 1) * 4)?;
        let starts: Vec<u32> = (0..=ntiles).map(|t| (t * cap) as u32).collect();
        let starts = PinnedHostBuffer::from_slice(ctx, &starts)?;
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

    /// `ntiles + 1` u32 offsets, `t * cap`.
    pub fn tile_start_ptr(&self) -> CUdeviceptr {
        self.tile_start.ptr()
    }

    /// `ntiles * cap` u32 view indices.
    pub fn tile_views_ptr(&self) -> CUdeviceptr {
        self.tile_views.ptr()
    }

    /// `ntiles` u32 run lengths.
    pub fn tile_len_ptr(&self) -> CUdeviceptr {
        self.tile_len.ptr()
    }

    pub fn nobj(&self) -> usize {
        self.nobj
    }

    /// `[nyo, nxo]`.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn tile(&self) -> usize {
        self.tile
    }

    /// `[nty, ntx]`.
    pub fn tiles(&self) -> [usize; 2] {
        self.tiles
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Backs the object rows and the slab of each tile `[ty, tx]` of `tiles`,
    /// in every plane, and queues the fill of fresh pixels on the stream, so
    /// kernels queued there afterwards may read the pixels and append to the
    /// slabs. Call it with `ViewDims::view_tiles` of a view before the view
    /// enters the address table: the bin append writes those slabs and the
    /// kernels touch those pixels, the bilinear neighbour's wrapped tile
    /// included.
    pub fn ensure(&mut self, tiles: impl IntoIterator<Item = [u32; 2]>) -> Result<(), GrowError> {
        let [nyo, nxo] = self.shape;
        let [nty, ntx] = self.tiles;
        let (t, row, plane, slab) = (self.tile, nxo * 8, nyo * nxo * 8, self.cap * 4);
        self.stream.context().bind_to_thread()?;
        let s = self.stream.cu_stream();
        let [re, im] = self.fill.map(f32::to_bits);
        let base = self.obj.ptr();
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
                    let at = z * plane;
                    self.obj
                        .map_with(at + rows.start * row..at + rows.end * row, |r| {
                            let dst = base + r.start as u64;
                            // SAFETY: `r` was just mapped; chunks are 8-byte aligned.
                            unsafe {
                                sys::cuMemsetD2D32Async(dst, 8, re, 1, r.len() / 8, s).result()?;
                                sys::cuMemsetD2D32Async(dst + 4, 8, im, 1, r.len() / 8, s).result()
                            }
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
    pub fn snapshot(&self, snap: &mut Snapshot) -> Result<(), DriverError> {
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

    fn queue_copy(&self, bands: &[Range<usize>], snap: &mut Snapshot) -> Result<(), DriverError> {
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
        snap.done.record(&self.stream)
    }
}

impl Drop for LiveObject {
    fn drop(&mut self) {
        // Queued fills and snapshot copies use the mappings the fields'
        // drops release.
        let _ = self.stream.synchronize();
    }
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
            done: ctx.new_event_with_flags(None)?,
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
    use std::collections::BTreeSet;
    use std::ffi::c_void;
    use std::num::NonZeroU32;

    use cuda_core::{Context, launch_kernel_on_stream};

    use super::*;
    use crate::consumer::{Graph, memset32};
    use crate::grow::GrowTable;
    use crate::stack::NextStack;

    const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// A stand-in for the M1 bin_views. Thread per tile t = ty * ntx + tx. For
// frames f in f0..f0 + nf of the StackCtl at ctl, ascending, whose w x w view
// at wins[f] = (oy, ox) touches the tile (on each axis, its tiles o / tile ..=
// (o + w - 1) / tile or the wrapped neighbour's ((o + w) % n) / tile): store
// f at views[tstart[t] + lens[t]] and add 1 to lens[t] while that slot is
// below tstart[t + 1], else add 1 to overflow.
.visible .entry bin_append(.param .u64 wins, .param .u64 ctl, .param .u64 tstart, .param .u64 views, .param .u64 lens, .param .u64 overflow, .param .u32 ntx, .param .u32 ntiles, .param .u32 tile, .param .u32 w, .param .u32 nyo, .param .u32 nxo)
{
    .reg .pred p;
    .reg .b32 r<22>;
    .reg .b64 rd<10>;
    ld.param.u64 rd1, [wins];
    ld.param.u64 rd2, [ctl];
    ld.param.u64 rd3, [tstart];
    ld.param.u64 rd4, [views];
    ld.param.u64 rd5, [lens];
    ld.param.u64 rd6, [overflow];
    ld.param.u32 r1, [ntx];
    ld.param.u32 r2, [ntiles];
    ld.param.u32 r3, [tile];
    ld.param.u32 r4, [w];
    ld.param.u32 r5, [nyo];
    ld.param.u32 r6, [nxo];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    cvta.to.global.u64 rd3, rd3;
    cvta.to.global.u64 rd4, rd4;
    cvta.to.global.u64 rd5, rd5;
    cvta.to.global.u64 rd6, rd6;
    mov.u32 r7, %ctaid.x;
    mov.u32 r8, %ntid.x;
    mov.u32 r9, %tid.x;
    mad.lo.u32 r7, r7, r8, r9;
    setp.ge.u32 p, r7, r2;
    @p bra BEND;
    div.u32 r8, r7, r1;
    rem.u32 r9, r7, r1;
    mul.wide.u32 rd7, r7, 4;
    add.s64 rd8, rd3, rd7;
    ld.global.u32 r10, [rd8];
    ld.global.u32 r11, [rd8+4];
    add.s64 rd7, rd5, rd7;
    ld.global.u32 r12, [rd7];
    ld.global.u32 r13, [rd2];
    ld.global.u32 r14, [rd2+4];
    add.u32 r14, r13, r14;
BLOOP:
    setp.ge.u32 p, r13, r14;
    @p bra BSTORE;
    mul.wide.u32 rd9, r13, 8;
    add.s64 rd9, rd1, rd9;
    ld.global.u32 r15, [rd9];
    ld.global.u32 r16, [rd9+4];
    div.u32 r17, r15, r3;
    add.u32 r18, r15, r4;
    sub.u32 r19, r18, 1;
    div.u32 r19, r19, r3;
    rem.u32 r18, r18, r5;
    div.u32 r18, r18, r3;
    setp.ge.u32 p, r8, r17;
    setp.le.and.u32 p, r8, r19, p;
    setp.eq.or.u32 p, r8, r18, p;
    @!p bra BNEXT;
    div.u32 r17, r16, r3;
    add.u32 r18, r16, r4;
    sub.u32 r19, r18, 1;
    div.u32 r19, r19, r3;
    rem.u32 r18, r18, r6;
    div.u32 r18, r18, r3;
    setp.ge.u32 p, r9, r17;
    setp.le.and.u32 p, r9, r19, p;
    setp.eq.or.u32 p, r9, r18, p;
    @!p bra BNEXT;
    add.u32 r20, r10, r12;
    setp.ge.u32 p, r20, r11;
    @p bra BOVER;
    mul.wide.u32 rd9, r20, 4;
    add.s64 rd9, rd4, rd9;
    st.global.u32 [rd9], r13;
    add.u32 r12, r12, 1;
    bra BNEXT;
BOVER:
    atom.global.add.u32 r21, [rd6], 1;
BNEXT:
    add.u32 r13, r13, 1;
    bra BLOOP;
BSTORE:
    st.global.u32 [rd7], r12;
BEND:
    ret;
}

// Block per tile, grid (ntx, nty, nobj); a tile with lens[t] == 0 returns
// before touching anything. Pixel (y, x) of plane z adds to its real part
// the number of frames f in f0..f0 + nf with f % 2 == z whose window covers
// it, read from the run tstart[t]..tstart[t] + lens[t] as the M1
// psi_to_obj reads it: binary search for the first f >= f0, stop at the
// first f >= f0 + nf.
.visible .entry owner_add(.param .u64 obj, .param .u64 wins, .param .u64 ctl, .param .u64 tstart, .param .u64 views, .param .u64 lens, .param .u32 nyo, .param .u32 nxo, .param .u32 w)
{
    .reg .pred p;
    .reg .b32 r<28>;
    .reg .f32 f<3>;
    .reg .b64 rd<14>;
    ld.param.u64 rd1, [obj];
    ld.param.u64 rd2, [wins];
    ld.param.u64 rd3, [ctl];
    ld.param.u64 rd4, [tstart];
    ld.param.u64 rd5, [views];
    ld.param.u64 rd6, [lens];
    ld.param.u32 r1, [nyo];
    ld.param.u32 r2, [nxo];
    ld.param.u32 r3, [w];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    cvta.to.global.u64 rd3, rd3;
    cvta.to.global.u64 rd4, rd4;
    cvta.to.global.u64 rd5, rd5;
    cvta.to.global.u64 rd6, rd6;
    mov.u32 r4, %ctaid.x;
    mov.u32 r5, %ctaid.y;
    mov.u32 r6, %nctaid.x;
    mad.lo.u32 r7, r5, r6, r4;
    mul.wide.u32 rd7, r7, 4;
    add.s64 rd8, rd6, rd7;
    ld.global.u32 r8, [rd8];
    setp.eq.u32 p, r8, 0;
    @p bra OEND;
    mov.u32 r9, %ntid.x;
    mov.u32 r10, %tid.x;
    mad.lo.u32 r11, r4, r9, r10;
    mov.u32 r9, %ntid.y;
    mov.u32 r10, %tid.y;
    mad.lo.u32 r12, r5, r9, r10;
    setp.ge.u32 p, r11, r2;
    @p bra OEND;
    setp.ge.u32 p, r12, r1;
    @p bra OEND;
    mov.u32 r13, %ctaid.z;
    add.s64 rd9, rd4, rd7;
    ld.global.u32 r14, [rd9];
    add.u32 r15, r14, r8;
    mov.u32 r16, r15;
    ld.global.u32 r17, [rd3];
    ld.global.u32 r18, [rd3+4];
    add.u32 r18, r17, r18;
SEARCH:
    setp.ge.u32 p, r14, r16;
    @p bra FOUND;
    sub.u32 r19, r16, r14;
    shr.u32 r19, r19, 1;
    add.u32 r19, r14, r19;
    mul.wide.u32 rd10, r19, 4;
    add.s64 rd10, rd5, rd10;
    ld.global.u32 r20, [rd10];
    setp.lt.u32 p, r20, r17;
    @p add.u32 r14, r19, 1;
    @!p mov.u32 r16, r19;
    bra SEARCH;
FOUND:
    mov.u32 r21, 0;
WALK:
    setp.ge.u32 p, r14, r15;
    @p bra WDONE;
    mul.wide.u32 rd10, r14, 4;
    add.s64 rd10, rd5, rd10;
    ld.global.u32 r20, [rd10];
    setp.ge.u32 p, r20, r18;
    @p bra WDONE;
    add.u32 r14, r14, 1;
    and.b32 r22, r20, 1;
    setp.ne.u32 p, r22, r13;
    @p bra WALK;
    mul.wide.u32 rd11, r20, 8;
    add.s64 rd11, rd2, rd11;
    ld.global.u32 r23, [rd11];
    ld.global.u32 r24, [rd11+4];
    setp.lt.u32 p, r12, r23;
    @p bra WALK;
    add.u32 r25, r23, r3;
    setp.ge.u32 p, r12, r25;
    @p bra WALK;
    setp.lt.u32 p, r11, r24;
    @p bra WALK;
    add.u32 r25, r24, r3;
    setp.ge.u32 p, r11, r25;
    @p bra WALK;
    add.u32 r21, r21, 1;
    bra WALK;
WDONE:
    mad.lo.u32 r26, r13, r1, r12;
    mul.wide.u32 rd12, r26, r2;
    cvt.u64.u32 rd13, r11;
    add.s64 rd12, rd12, rd13;
    shl.b64 rd12, rd12, 3;
    add.s64 rd12, rd1, rd12;
    ld.global.f32 f1, [rd12];
    cvt.rn.f32.u32 f2, r21;
    add.f32 f1, f1, f2;
    st.global.f32 [rd12], f1;
OEND:
    ret;
}
"#;

    fn dtoh(src: CUdeviceptr, n: usize) -> Vec<u32> {
        let mut v = vec![0u32; n];
        unsafe { sys::cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), src, n * 4) }
            .result()
            .unwrap();
        v
    }

    /// The test's `ViewDims::view_tiles` for a `w x w` view at `(oy, ox)` of
    /// an `n x n` object in 16-pixel tiles: the tiles the view covers and,
    /// on each axis, the tile of the bilinear neighbour at `(o + w) % n`.
    fn view_tiles([oy, ox]: [u32; 2], w: u32, n: u32) -> impl Iterator<Item = [u32; 2]> {
        fn axis(o: u32, w: u32, n: u32) -> impl Iterator<Item = u32> {
            let (first, last, wrap) = (o / 16, (o + w - 1) / 16, ((o + w) % n) / 16);
            (first..=last).chain((wrap < first || wrap > last).then_some(wrap))
        }
        axis(oy, w, n).flat_map(move |ty| axis(ox, w, n).map(move |tx| [ty, tx]))
    }

    /// Views on a raster, and views on the far edges whose bilinear
    /// neighbour wraps to tile row or column 0, reach new object rows and
    /// tiles in uneven batches. Each view's tiles are ensured before its
    /// record enters the table, and one graph, captured before anything was
    /// backed, advances the StackCtl, bins the new views and adds them tile
    /// by tile, reading each run up to `tile_len`. A batch larger than
    /// `nf_max` takes several replays; nothing is synchronised until the end.
    #[test]
    fn bins_and_object_grow_while_one_graph_reads_them() {
        let ctx = Context::new(0).expect("GPU 0");
        let m = ctx.load_module_from_ptx_src(PTX).unwrap();
        let (bin, add) = (
            m.load_function("bin_append").unwrap(),
            m.load_function("owner_add").unwrap(),
        );
        let s = ctx.new_stream().unwrap();
        let (n, tile, cap, w, nf_max) = (2048usize, 16usize, 1024usize, 64usize, 16u32);
        let fill = [1.0f32, 0.5];
        let mut o = LiveObject::new(&s, 2, [n, n], tile, cap, fill, 1).unwrap();
        let [nty, ntx] = o.tiles();
        let ntiles = nty * ntx;
        let mut wins: Vec<[u32; 2]> = (300..=700)
            .step_by(16)
            .flat_map(|oy| (100..=700).step_by(16).map(move |ox| [oy, ox]))
            .collect();
        let edge = (n - w) as u32;
        for (at, v) in [
            (900, [edge, edge]),
            (600, [600, edge]),
            (300, [edge, 1000]),
            (100, [edge, 500]),
        ] {
            wins.insert(at, v);
        }
        let total = wins.len();
        let mut table = GrowTable::<[u32; 2]>::new(&s, total, 1).unwrap();
        let step = NextStack::load(&ctx, NonZeroU32::new(nf_max).unwrap(), total as u32).unwrap();
        // The StackCtl, the landed count, the overflow count.
        let mut ctl = GrowBuf::new(&ctx, 16, 1).unwrap();
        ctl.map(0..16).unwrap();
        let (landed, overflow) = (ctl.ptr() + 8, ctl.ptr() + 12);
        memset32(ctl.ptr(), 0, 4, &s).unwrap();

        let (mut b_wins, mut b_ctl, mut b_start, mut b_views, mut b_lens, mut b_over) = (
            table.ptr(),
            ctl.ptr(),
            o.tile_start_ptr(),
            o.tile_views_ptr(),
            o.tile_len_ptr(),
            overflow,
        );
        let (mut b_ntx, mut b_ntiles, mut b_tile, mut b_w, mut b_nyo, mut b_nxo) = (
            ntx as u32,
            ntiles as u32,
            tile as u32,
            w as u32,
            n as u32,
            n as u32,
        );
        let mut bin_params = [
            (&raw mut b_wins).cast::<c_void>(),
            (&raw mut b_ctl).cast(),
            (&raw mut b_start).cast(),
            (&raw mut b_views).cast(),
            (&raw mut b_lens).cast(),
            (&raw mut b_over).cast(),
            (&raw mut b_ntx).cast(),
            (&raw mut b_ntiles).cast(),
            (&raw mut b_tile).cast(),
            (&raw mut b_w).cast(),
            (&raw mut b_nyo).cast(),
            (&raw mut b_nxo).cast(),
        ];
        let (mut a_obj, mut a_wins, mut a_ctl, mut a_start, mut a_views, mut a_lens) = (
            o.ptr(),
            table.ptr(),
            ctl.ptr(),
            o.tile_start_ptr(),
            o.tile_views_ptr(),
            o.tile_len_ptr(),
        );
        let (mut a_nyo, mut a_nxo, mut a_w) = (n as u32, n as u32, w as u32);
        let mut add_params = [
            (&raw mut a_obj).cast::<c_void>(),
            (&raw mut a_wins).cast(),
            (&raw mut a_ctl).cast(),
            (&raw mut a_start).cast(),
            (&raw mut a_views).cast(),
            (&raw mut a_lens).cast(),
            (&raw mut a_nyo).cast(),
            (&raw mut a_nxo).cast(),
            (&raw mut a_w).cast(),
        ];
        let t = tile as u32;
        // SAFETY: every buffer the graph reads or writes outlives it; the
        // parameter arrays match the kernels and outlive the capture.
        let graph = Graph::capture(&s, || unsafe {
            step.queue(&s, ctl.ptr(), landed)?;
            let blocks = ntiles.div_ceil(256) as u32;
            launch_kernel_on_stream(&bin, (blocks, 1, 1), (256, 1, 1), 0, &s, &mut bin_params)?;
            let grid = (ntx as u32, nty as u32, 2);
            launch_kernel_on_stream(&add, grid, (t, t, 1), 0, &s, &mut add_params)
        })
        .unwrap();

        let sizes = [1usize, 5, 13, 2, 40, 7, 3];
        let (mut f, mut covered, mut k, mut launches) = (0usize, 0usize, 0usize, 0usize);
        while f < total {
            let b = sizes[k % sizes.len()].min(total - f);
            k += 1;
            for &v in &wins[f..f + b] {
                o.ensure(view_tiles(v, w as u32, n as u32)).unwrap();
            }
            assert_eq!(table.push(&wins[f..f + b]).unwrap(), f..f + b);
            f += b;
            memset32(landed, f as u32, 1, &s).unwrap();
            while covered < f {
                graph.launch(&s).unwrap();
                launches += 1;
                covered = (covered + nf_max as usize).min(f);
            }
        }
        // Nothing new: this replay has nf = 0 and must change nothing.
        graph.launch(&s).unwrap();
        s.synchronize().unwrap();
        assert!(launches > k, "no batch needed several replays");
        assert_eq!(
            (o.ptr(), o.tile_views_ptr(), o.tile_len_ptr()),
            (a_obj, b_views, b_lens)
        );
        assert_eq!(dtoh(ctl.ptr(), 4), [total as u32, 0, total as u32, 0]);
        let starts: Vec<u32> = (0..=ntiles).map(|t| (t * cap) as u32).collect();
        assert_eq!(dtoh(o.tile_start_ptr(), ntiles + 1), starts);

        // Host model: runs in view order and per-plane cover counts.
        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); ntiles];
        let mut cover = vec![0u32; 2 * n * n];
        for (fi, &v) in wins.iter().enumerate() {
            for [ty, tx] in view_tiles(v, w as u32, n as u32) {
                bins[ty as usize * ntx + tx as usize].push(fi as u32);
            }
            let [oy, ox] = v.map(|c| c as usize);
            for y in oy..oy + w {
                for x in ox..ox + w {
                    cover[((fi % 2) * n + y) * n + x] += 1;
                }
            }
        }
        // Tile row 0 and column 0 are reached only through the wrap.
        for t in [0, 500 / 16, 1000 / 16, (600 / 16) * ntx] {
            assert!(!bins[t].is_empty(), "tile {t} not reached");
        }
        let lens = dtoh(o.tile_len_ptr(), ntiles);
        let bad_lens = (0..ntiles)
            .filter(|&t| lens[t] as usize != bins[t].len())
            .count();
        assert_eq!(bad_lens, 0, "tiles with a wrong run length");
        let reached: Vec<usize> = (0..ntiles).filter(|&t| !bins[t].is_empty()).collect();
        let bad_runs = reached
            .iter()
            .filter(|&&t| dtoh(o.tile_views_ptr() + (t * cap * 4) as u64, bins[t].len()) != bins[t])
            .count();
        assert_eq!(bad_runs, 0, "of {} reached tiles", reached.len());

        // Each plane backs the whole chunks over the reached tile rows, and
        // every backed pixel is the fill plus its cover count.
        let ck = o.obj.chunk_bytes();
        let (row, plane) = (n * 8, n * n * 8);
        assert_eq!(plane % ck, 0);
        let rows: BTreeSet<usize> = reached.iter().map(|t| t / ntx).collect();
        let mut chunks = BTreeSet::new();
        for z in 0..2 {
            for ty in &rows {
                let at = z * plane + ty * tile * row;
                chunks.extend(at / ck..=(at + tile * row - 1) / ck);
            }
        }
        assert!(chunks.contains(&0) && chunks.contains(&(plane / ck)));
        assert_eq!(o.obj.mapped_bytes(), chunks.len() * ck);
        let mut bad = 0;
        for c in &chunks {
            assert!(o.obj.is_mapped(c * ck..(c + 1) * ck));
            let got = dtoh(o.ptr() + (c * ck) as u64, ck / 4);
            let px0 = c * ck / 8;
            bad += (0..ck / 8)
                .filter(|&i| {
                    f32::from_bits(got[2 * i]) != fill[0] + cover[px0 + i] as f32
                        || f32::from_bits(got[2 * i + 1]) != fill[1]
                })
                .count();
        }
        assert_eq!(bad, 0, "pixels off fill + cover in {} chunks", chunks.len());
        // Slabs of reached tiles only, in whole chunks.
        let (sck, slab) = (o.tile_views.chunk_bytes(), cap * 4);
        let slabs: BTreeSet<usize> = reached
            .iter()
            .flat_map(|t| t * slab / sck..=((t + 1) * slab - 1) / sck)
            .collect();
        assert_eq!(o.tile_views.mapped_bytes(), slabs.len() * sck);

        assert!(o.ensure([]).is_ok());
        for past in [[nty as u32, 0], [0, ntx as u32]] {
            assert!(matches!(
                o.ensure([past]),
                Err(GrowError::Tile { tile, tiles }) if tile == past && tiles == [nty, ntx]
            ));
        }
        eprintln!(
            "object: {total} views in {k} batches, {launches} replays; object {} MiB, slabs {} MiB backed",
            o.obj.mapped_bytes() >> 20,
            o.tile_views.mapped_bytes() >> 20
        );
    }

    /// A snapshot is queued behind work the stream has not reached, without
    /// waiting for it, copies backed rows only and reads the rest as fill.
    #[test]
    fn snapshot_copies_backed_rows_behind_queued_work() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (n, fill) = (1024usize, [1.0f32, 0.5]);
        let mut o = LiveObject::new(&s, 2, [n, n], 16, 4, fill, 1).unwrap();
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
}
