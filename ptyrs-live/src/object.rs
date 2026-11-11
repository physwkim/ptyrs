//! The live object and its spatial bins (DESIGN.md §3 B2, B3, B6): both keep
//! their device addresses while the scan grows, so one captured graph covers
//! the whole measurement.
//!
//! The object is `[nobj, nyo, nxo]` interleaved complex f32, row-major, the
//! M1 kernels' layout; rows are backed in whole tile rows as frames reach
//! them, and fresh pixels start at a flat value. The bins give tile
//! `t = ty * ntx + tx` a slab of `cap` view indices at `t * cap`, appended in
//! ascending order and padded with `u32::MAX`; with the fixed offsets
//! `tile_start[t] = t * cap` a slab reads as the sorted run of a CSR index,
//! because the padding sorts after every view index. `lens[t]` counts the
//! appends, and is 0 for every tile whose slab or pixels are not backed.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use cuda_bindings::{self as sys, CUdeviceptr};
use cuda_core::{DriverError, Event, IntoResult, PinnedHostBuffer, Stream};

use crate::grow::{GrowBuf, GrowError};

/// Object pixels `oy..oy + ny` by `ox..ox + nx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub oy: usize,
    pub ox: usize,
    pub ny: usize,
    pub nx: usize,
}

impl fmt::Display for Window {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { oy, ox, ny, nx } = *self;
        write!(f, "rows {oy}..{} x cols {ox}..{}", oy + ny, ox + nx)
    }
}

pub struct LiveObject {
    obj: GrowBuf,
    lists: GrowBuf,
    lens: GrowBuf,
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
            lists: GrowBuf::new(ctx, ntiles * cap * 4, chunk)?,
            lens: GrowBuf::new(ctx, ntiles * 4, 1)?,
            tile_start: GrowBuf::new(ctx, (ntiles + 1) * 4, 1)?,
            nobj,
            shape,
            tile,
            tiles,
            cap,
            fill,
            stream: stream.clone(),
        };
        o.lens.map(0..ntiles * 4)?;
        o.tile_start.map(0..(ntiles + 1) * 4)?;
        let starts: Vec<u32> = (0..=ntiles).map(|t| (t * cap) as u32).collect();
        let starts = PinnedHostBuffer::from_slice(ctx, &starts)?;
        ctx.bind_to_thread()?;
        let s = stream.cu_stream();
        // SAFETY: both destinations were mapped above with these sizes; the
        // stream is synchronised before `starts` is freed.
        unsafe {
            sys::cuMemsetD32Async(o.lens.ptr(), 0, ntiles, s).result()?;
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
    pub fn lists_ptr(&self) -> CUdeviceptr {
        self.lists.ptr()
    }

    /// `ntiles` u32 append counts.
    pub fn lens_ptr(&self) -> CUdeviceptr {
        self.lens.ptr()
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

    /// Backs the object rows and the slabs of every tile `w` touches, in
    /// every plane, and queues the fill of the fresh parts on the stream, so
    /// kernels queued there afterwards may read and append to them. Call it
    /// before a view reading `w` enters the address table.
    pub fn ensure(&mut self, w: Window) -> Result<(), GrowError> {
        let [nyo, nxo] = self.shape;
        if w.oy + w.ny > nyo || w.ox + w.nx > nxo {
            return Err(GrowError::Outside {
                window: w,
                shape: self.shape,
            });
        }
        if w.ny == 0 || w.nx == 0 {
            return Ok(());
        }
        let t = self.tile;
        let ty = w.oy / t..(w.oy + w.ny - 1) / t + 1;
        let tx = w.ox / t..(w.ox + w.nx - 1) / t + 1;
        let (row, plane) = (nxo * 8, nyo * nxo * 8);
        let rows = ty.start * t..(ty.end * t).min(nyo);
        self.stream.context().bind_to_thread()?;
        let s = self.stream.cu_stream();
        let [re, im] = self.fill.map(f32::to_bits);
        let base = self.obj.ptr();
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
        let (ntx, slab, base) = (self.tiles[1], self.cap * 4, self.lists.ptr());
        for y in ty {
            let span = (y * ntx + tx.start) * slab..(y * ntx + tx.end) * slab;
            self.lists.map_with(span, |r| {
                // SAFETY: `r` was just mapped.
                unsafe { sys::cuMemsetD32Async(base + r.start as u64, u32::MAX, r.len() / 4, s) }
                    .result()
            })?;
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
    use std::ffi::c_void;

    use cuda_core::{Context, launch_kernel_on_stream};

    use super::*;
    use crate::consumer::{Graph, memset32};
    use crate::grow::GrowTable;

    const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// Thread per tile t = ty * ntx + tx. For frames f in ctl[0]..ctl[1], in
// ascending f, whose w x w window at wins[f] = (oy, ox) overlaps the tile:
// lists[t * cap + lens[t]] = f while lens[t] < cap, and lens[t] += 1.
.visible .entry bin_append(.param .u64 wins, .param .u64 ctl, .param .u64 lists, .param .u64 lens, .param .u32 ntx, .param .u32 ntiles, .param .u32 tile, .param .u32 w, .param .u32 cap)
{
    .reg .pred p;
    .reg .b32 r<20>;
    .reg .b64 rd<8>;
    ld.param.u64 rd1, [wins];
    ld.param.u64 rd2, [ctl];
    ld.param.u64 rd3, [lists];
    ld.param.u64 rd4, [lens];
    ld.param.u32 r1, [ntx];
    ld.param.u32 r2, [ntiles];
    ld.param.u32 r3, [tile];
    ld.param.u32 r4, [w];
    ld.param.u32 r5, [cap];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    cvta.to.global.u64 rd3, rd3;
    cvta.to.global.u64 rd4, rd4;
    mov.u32 r6, %ctaid.x;
    mov.u32 r7, %ntid.x;
    mov.u32 r8, %tid.x;
    mad.lo.u32 r6, r6, r7, r8;
    setp.ge.u32 p, r6, r2;
    @p bra BEND;
    div.u32 r9, r6, r1;
    rem.u32 r10, r6, r1;
    mul.lo.u32 r9, r9, r3;
    mul.lo.u32 r10, r10, r3;
    mul.wide.u32 rd5, r6, 4;
    add.s64 rd5, rd4, rd5;
    ld.global.u32 r11, [rd5];
    mul.lo.u32 r12, r6, r5;
    ld.global.u32 r13, [rd2];
    ld.global.u32 r14, [rd2+4];
BLOOP:
    setp.ge.u32 p, r13, r14;
    @p bra BSTORE;
    mul.wide.u32 rd6, r13, 8;
    add.s64 rd6, rd1, rd6;
    ld.global.u32 r15, [rd6];
    ld.global.u32 r16, [rd6+4];
    add.u32 r17, r9, r3;
    setp.ge.u32 p, r15, r17;
    @p bra BNEXT;
    add.u32 r17, r15, r4;
    setp.ge.u32 p, r9, r17;
    @p bra BNEXT;
    add.u32 r17, r10, r3;
    setp.ge.u32 p, r16, r17;
    @p bra BNEXT;
    add.u32 r17, r16, r4;
    setp.ge.u32 p, r10, r17;
    @p bra BNEXT;
    setp.ge.u32 p, r11, r5;
    @p bra BCOUNT;
    add.u32 r17, r12, r11;
    mul.wide.u32 rd7, r17, 4;
    add.s64 rd7, rd3, rd7;
    st.global.u32 [rd7], r13;
BCOUNT:
    add.u32 r11, r11, 1;
BNEXT:
    add.u32 r13, r13, 1;
    bra BLOOP;
BSTORE:
    st.global.u32 [rd5], r11;
BEND:
    ret;
}

// Block per tile, grid (ntx, nty, nobj); a tile with lens[t] == 0 returns
// before touching anything. Pixel (y, x) of plane z adds to its real part
// the number of frames f in ctl[0]..ctl[1] with f % 2 == z whose window
// covers it, read from tile_start[t]..tile_start[t + 1] as the M1
// psi_to_obj reads its run: binary search for the first f >= ctl[0], stop
// at the first f >= ctl[1].
.visible .entry owner_add(.param .u64 obj, .param .u64 wins, .param .u64 ctl, .param .u64 tstart, .param .u64 lists, .param .u64 lens, .param .u32 nyo, .param .u32 nxo, .param .u32 w)
{
    .reg .pred p;
    .reg .b32 r<28>;
    .reg .f32 f<3>;
    .reg .b64 rd<14>;
    ld.param.u64 rd1, [obj];
    ld.param.u64 rd2, [wins];
    ld.param.u64 rd3, [ctl];
    ld.param.u64 rd4, [tstart];
    ld.param.u64 rd5, [lists];
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
    ld.global.u32 r15, [rd9+4];
    mov.u32 r16, r15;
    ld.global.u32 r17, [rd3];
    ld.global.u32 r18, [rd3+4];
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

// ctl[0] = ctl[1].
.visible .entry advance(.param .u64 ctl)
{
    .reg .b32 r1;
    .reg .b64 rd1;
    ld.param.u64 rd1, [ctl];
    cvta.to.global.u64 rd1, rd1;
    ld.global.u32 r1, [rd1+4];
    st.global.u32 [rd1], r1;
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

    /// Frames on a raster reach new object rows and tiles in uneven batches;
    /// each window is ensured, its record appended to a table, and one graph,
    /// captured before anything was backed, bins the new frames and adds them
    /// tile by tile, reading the bins as a CSR run. Nothing is synchronised
    /// until the end.
    #[test]
    fn bins_and_object_grow_while_one_graph_reads_them() {
        let ctx = Context::new(0).expect("GPU 0");
        let m = ctx.load_module_from_ptx_src(PTX).unwrap();
        let (bin, add, adv) = (
            m.load_function("bin_append").unwrap(),
            m.load_function("owner_add").unwrap(),
            m.load_function("advance").unwrap(),
        );
        let s = ctx.new_stream().unwrap();
        let (n, tile, cap, w) = (1024usize, 16usize, 1024usize, 64usize);
        let fill = [1.0f32, 0.5];
        let mut o = LiveObject::new(&s, 2, [n, n], tile, cap, fill, 1).unwrap();
        let [nty, ntx] = o.tiles();
        let ntiles = nty * ntx;
        let wins: Vec<[u32; 2]> = (0..=400)
            .step_by(16)
            .flat_map(|oy| (100..=700).step_by(16).map(move |ox| [oy, ox]))
            .collect();
        let total = wins.len();
        let mut table = GrowTable::<[u32; 2]>::new(&s, total, 1).unwrap();
        let mut ctl = GrowBuf::new(&ctx, 8, 1).unwrap();
        ctl.map(0..8).unwrap();
        memset32(ctl.ptr(), 0, 2, &s).unwrap();

        let (mut b_wins, mut b_ctl, mut b_lists, mut b_lens) =
            (table.ptr(), ctl.ptr(), o.lists_ptr(), o.lens_ptr());
        let (mut b_ntx, mut b_ntiles, mut b_tile, mut b_w, mut b_cap) =
            (ntx as u32, ntiles as u32, tile as u32, w as u32, cap as u32);
        let mut bin_params = [
            (&raw mut b_wins).cast::<c_void>(),
            (&raw mut b_ctl).cast(),
            (&raw mut b_lists).cast(),
            (&raw mut b_lens).cast(),
            (&raw mut b_ntx).cast(),
            (&raw mut b_ntiles).cast(),
            (&raw mut b_tile).cast(),
            (&raw mut b_w).cast(),
            (&raw mut b_cap).cast(),
        ];
        let (mut a_obj, mut a_wins, mut a_ctl, mut a_start, mut a_lists, mut a_lens) = (
            o.ptr(),
            table.ptr(),
            ctl.ptr(),
            o.tile_start_ptr(),
            o.lists_ptr(),
            o.lens_ptr(),
        );
        let (mut a_nyo, mut a_nxo, mut a_w) = (n as u32, n as u32, w as u32);
        let mut add_params = [
            (&raw mut a_obj).cast::<c_void>(),
            (&raw mut a_wins).cast(),
            (&raw mut a_ctl).cast(),
            (&raw mut a_start).cast(),
            (&raw mut a_lists).cast(),
            (&raw mut a_lens).cast(),
            (&raw mut a_nyo).cast(),
            (&raw mut a_nxo).cast(),
            (&raw mut a_w).cast(),
        ];
        let mut v_ctl = ctl.ptr();
        let mut adv_params = [(&raw mut v_ctl).cast::<c_void>()];
        let t = tile as u32;
        let graph = Graph::capture(&s, || unsafe {
            let blocks = ntiles.div_ceil(256) as u32;
            launch_kernel_on_stream(&bin, (blocks, 1, 1), (256, 1, 1), 0, &s, &mut bin_params)?;
            let grid = (ntx as u32, nty as u32, 2);
            launch_kernel_on_stream(&add, grid, (t, t, 1), 0, &s, &mut add_params)?;
            launch_kernel_on_stream(&adv, (1, 1, 1), (1, 1, 1), 0, &s, &mut adv_params)
        })
        .unwrap();

        let sizes = [1usize, 5, 13, 2, 40, 7, 3];
        let (mut f, mut k) = (0usize, 0usize);
        while f < total {
            let b = sizes[k % sizes.len()].min(total - f);
            k += 1;
            for &[oy, ox] in &wins[f..f + b] {
                let (oy, ox) = (oy as usize, ox as usize);
                o.ensure(Window {
                    oy,
                    ox,
                    ny: w,
                    nx: w,
                })
                .unwrap();
            }
            assert_eq!(table.push(&wins[f..f + b]).unwrap(), f..f + b);
            f += b;
            memset32(ctl.ptr() + 4, f as u32, 1, &s).unwrap();
            graph.launch(&s).unwrap();
        }
        s.synchronize().unwrap();
        assert_eq!(
            (o.ptr(), o.lists_ptr(), o.lens_ptr()),
            (a_obj, b_lists, b_lens)
        );
        assert_eq!(dtoh(ctl.ptr(), 2), vec![total as u32; 2]);

        // Host model: bins in frame order and per-plane cover counts.
        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); ntiles];
        let mut cover = vec![0u32; 2 * n * n];
        for (fi, &[oy, ox]) in wins.iter().enumerate() {
            let (oy, ox) = (oy as usize, ox as usize);
            for ty in oy / tile..=(oy + w - 1) / tile {
                for tx in ox / tile..=(ox + w - 1) / tile {
                    bins[ty * ntx + tx].push(fi as u32);
                }
            }
            for y in oy..oy + w {
                for x in ox..ox + w {
                    cover[((fi % 2) * n + y) * n + x] += 1;
                }
            }
        }
        let lens = dtoh(o.lens_ptr(), ntiles);
        let bad_lens = (0..ntiles)
            .filter(|&t| lens[t] as usize != bins[t].len())
            .count();
        assert_eq!(bad_lens, 0, "tiles with a wrong append count");
        let (mut checked, mut bad_runs) = (0, 0);
        for (t, run) in bins.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
            let got = dtoh(o.lists_ptr() + (t * cap * 4) as u64, cap);
            let want: Vec<u32> = run
                .iter()
                .copied()
                .chain(std::iter::repeat(u32::MAX))
                .take(cap)
                .collect();
            checked += 1;
            bad_runs += usize::from(got != want);
        }
        assert_eq!(bad_runs, 0, "of {checked} binned tiles");

        // Rows 0..464 are reached; whole chunks back rows 0..512 per plane.
        let ck = o.obj.chunk_bytes();
        let (row, plane) = (n * 8, n * n * 8);
        assert_eq!(plane % ck, 0);
        let backed = (464 * row).div_ceil(ck) * ck / row;
        assert_eq!(o.obj.mapped_bytes(), 2 * backed * row);
        for z in 0..2 {
            let at = z * plane;
            assert!(o.obj.is_mapped(at..at + backed * row));
            assert!(!o.obj.is_mapped(at + backed * row..at + backed * row + 8));
            let got = dtoh(o.ptr() + at as u64, backed * n * 2);
            let bad = (0..backed * n)
                .filter(|&i| {
                    let want_re = fill[0] + cover[z * n * n + i] as f32;
                    f32::from_bits(got[2 * i]) != want_re
                        || f32::from_bits(got[2 * i + 1]) != fill[1]
                })
                .count();
            assert_eq!(bad, 0, "plane {z}: pixels off fill + cover count");
        }
        // Slabs of reached tiles only, in whole chunks.
        let lck = o.lists.chunk_bytes();
        let mut chunks = std::collections::BTreeSet::new();
        for (t, run) in bins.iter().enumerate() {
            if !run.is_empty() {
                chunks.extend(t * cap * 4 / lck..=((t + 1) * cap * 4 - 1) / lck);
            }
        }
        assert_eq!(o.lists.mapped_bytes(), chunks.len() * lck);

        assert!(
            o.ensure(Window {
                oy: 3,
                ox: 9,
                ny: 0,
                nx: 5
            })
            .is_ok()
        );
        let past = Window {
            oy: n - 63,
            ox: 0,
            ny: 64,
            nx: 64,
        };
        assert!(matches!(
            o.ensure(past),
            Err(GrowError::Outside { window, shape }) if window == past && shape == [n, n]
        ));
        eprintln!(
            "object: {total} frames in {k} batches; object {} MiB, slabs {} MiB backed",
            o.obj.mapped_bytes() >> 20,
            o.lists.mapped_bytes() >> 20
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

        // 2 MiB chunks back 256 rows of a plane: rows 256..512.
        let win = |oy| Window {
            oy,
            ox: 40,
            ny: 40,
            nx: 64,
        };
        o.ensure(win(300)).unwrap();
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
        o.ensure(win(700)).unwrap();
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
