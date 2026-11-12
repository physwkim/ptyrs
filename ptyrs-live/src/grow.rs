//! Device buffers that grow without moving (DESIGN.md §3 B2, B6; §5
//! invariant 9). One virtual range is reserved up front and physical chunks
//! are mapped into it as the scan reaches them, so a device pointer captured
//! in a CUDA graph stays valid for the whole measurement. Mapping a chunk
//! touches no mapped chunk, so kernels and graph replays reading the mapped
//! part keep running.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use cuda_bindings::{self as sys, CUdeviceptr};
use cuda_core::vmm::{self, Mapping, PhysicalAllocation, VirtualReservation};
use cuda_core::{Context, DeviceCopy, DriverError, IntoResult, PinnedHostBuffer, Stream};

#[derive(Debug)]
pub enum GrowError {
    /// The requested range ends past the reservation.
    Capacity {
        end: usize,
        capacity: usize,
    },
    /// A [`LiveObject`](crate::object::LiveObject) tile `[ty, tx]` past its
    /// `[nty, ntx]` tiles.
    Tile {
        tile: [u32; 2],
        tiles: [usize; 2],
    },
    /// An earlier chunk's initialisation failed; the buffer maps no more.
    Failed,
    Driver(DriverError),
}

impl fmt::Display for GrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity { end, capacity } => {
                write!(f, "range end {end} past the {capacity}-byte reservation")
            }
            Self::Tile { tile, tiles } => {
                write!(f, "tile {tile:?} past the {}x{} tiles", tiles[0], tiles[1])
            }
            Self::Failed => write!(f, "an earlier chunk initialisation failed"),
            Self::Driver(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GrowError {}

impl From<DriverError> for GrowError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// A fixed device address range whose chunks are backed on demand. Chunks
/// are never unmapped before drop, so every byte of a mapped chunk keeps its
/// address and contents; drop waits for all work queued on the context.
pub struct GrowBuf {
    // Declaration order is drop order: mappings before the reservation.
    chunks: Vec<Option<Mapping>>,
    va: VirtualReservation,
    chunk: usize,
    mapped: usize,
    failed: bool,
    ctx: Arc<Context>,
}

impl GrowBuf {
    /// Reserves `capacity` bytes, rounded up to whole chunks of at least
    /// `chunk` bytes (rounded up to the allocation granularity). Nothing is
    /// backed yet.
    pub fn new(ctx: &Arc<Context>, capacity: usize, chunk: usize) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let chunk = vmm::align_size(chunk.max(1), vmm::allocation_granularity(ctx.cu_device())?);
        let n = capacity.div_ceil(chunk).max(1);
        let va = VirtualReservation::new(n * chunk, chunk)?;
        Ok(Self {
            chunks: (0..n).map(|_| None).collect(),
            va,
            chunk,
            mapped: 0,
            failed: false,
            ctx: ctx.clone(),
        })
    }

    /// Base address; the same for the buffer's whole life.
    pub fn ptr(&self) -> CUdeviceptr {
        self.va.base()
    }

    pub fn capacity(&self) -> usize {
        self.va.size()
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk
    }

    pub fn mapped_bytes(&self) -> usize {
        self.mapped * self.chunk
    }

    /// Whether every byte of `r` is backed.
    pub fn is_mapped(&self, r: Range<usize>) -> bool {
        r.end <= self.capacity() && self.chunk_span(&r).all(|c| self.chunks[c].is_some())
    }

    /// [`map_with`](Self::map_with) without initialisation, for buffers whose
    /// new bytes are written before anything reads them.
    pub fn map(&mut self, r: Range<usize>) -> Result<(), GrowError> {
        self.map_with(r, |_| Ok(()))
    }

    /// Backs every chunk that overlaps `r`, ascending, and hands each newly
    /// backed chunk's byte range to `init` (to queue its first contents)
    /// before the next. A chunk that fails to back stays unbacked and the
    /// buffer usable. A chunk whose `init` fails stays backed, since work
    /// `init` queued may target it, and fails the buffer: every later call
    /// returns [`GrowError::Failed`], so no chunk is ever skipped as backed
    /// without having been initialised.
    pub fn map_with(
        &mut self,
        r: Range<usize>,
        mut init: impl FnMut(Range<usize>) -> Result<(), DriverError>,
    ) -> Result<(), GrowError> {
        if self.failed {
            return Err(GrowError::Failed);
        }
        if r.end > self.capacity() {
            return Err(GrowError::Capacity {
                end: r.end,
                capacity: self.capacity(),
            });
        }
        for c in self.chunk_span(&r) {
            if self.chunks[c].is_some() {
                continue;
            }
            self.ctx.bind_to_thread()?;
            let dev = self.ctx.cu_device();
            let va = self.va.base() + (c * self.chunk) as u64;
            // The mapping retains the physical memory; the handle can go.
            let phys = PhysicalAllocation::new(dev, self.chunk)?;
            let m = Mapping::new(va, self.chunk, &phys, 0)?;
            vmm::set_access(va, self.chunk, &[dev])?;
            self.chunks[c] = Some(m);
            self.mapped += 1;
            if let Err(e) = init(c * self.chunk..(c + 1) * self.chunk) {
                self.failed = true;
                return Err(e.into());
            }
        }
        Ok(())
    }

    fn chunk_span(&self, r: &Range<usize>) -> Range<usize> {
        if r.is_empty() {
            return 0..0;
        }
        r.start / self.chunk..r.end.div_ceil(self.chunk)
    }
}

impl Drop for GrowBuf {
    fn drop(&mut self) {
        // The fields' own drops unmap and free, with the context current. Unlike
        // cuMemFree, cuMemUnmap does not wait for queued work, and a copy or kernel
        // reaching an unmapped chunk is an MMU fault (Xid 31); no stream is known here.
        let _ = self.ctx.synchronize();
    }
}

/// Append-only device table of `T`, such as the live address table
/// (DESIGN.md §3 B6): record `i` stays at `ptr() + i * size_of::<T>()`.
/// Records are copied from a pinned mirror they never leave, so a push
/// never waits for the stream and no queued copy outlives its source.
pub struct GrowTable<T: DeviceCopy> {
    buf: GrowBuf,
    mirror: PinnedHostBuffer<T>,
    len: usize,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> GrowTable<T> {
    /// Room for `max_len` records, backed `chunk` bytes at a time and
    /// copied on `stream`.
    pub fn new(stream: &Arc<Stream>, max_len: usize, chunk: usize) -> Result<Self, DriverError> {
        let ctx = stream.context();
        Ok(Self {
            buf: GrowBuf::new(ctx, max_len * size_of::<T>(), chunk)?,
            mirror: PinnedHostBuffer::zeroed(ctx, max_len)?,
            len: 0,
            stream: stream.clone(),
        })
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.buf.ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn max_len(&self) -> usize {
        self.mirror.len()
    }

    pub fn mapped_bytes(&self) -> usize {
        self.buf.mapped_bytes()
    }

    /// Appends `recs` and queues their copy on the table's stream, so
    /// kernels queued there afterwards see them. Returns their indices.
    pub fn push(&mut self, recs: &[T]) -> Result<Range<usize>, GrowError> {
        let sz = size_of::<T>();
        let idx = self.len..self.len + recs.len();
        if idx.end > self.max_len() {
            return Err(GrowError::Capacity {
                end: idx.end * sz,
                capacity: self.max_len() * sz,
            });
        }
        self.buf.map(idx.start * sz..idx.end * sz)?;
        if recs.is_empty() {
            return Ok(idx);
        }
        self.mirror[idx.clone()].copy_from_slice(recs);
        self.stream.context().bind_to_thread()?;
        // SAFETY: the destination was mapped above; the source is the
        // mirror's slots `idx`, never written again while the table lives
        // (Drop waits for the stream before the mirror is freed).
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                self.ptr() + (idx.start * sz) as u64,
                self.mirror[idx.clone()].as_ptr().cast(),
                size_of_val(recs),
                self.stream.cu_stream(),
            )
        }
        .result()?;
        self.len = idx.end;
        Ok(idx)
    }
}

impl<T: DeviceCopy> Drop for GrowTable<T> {
    fn drop(&mut self) {
        // The mirror and the mapping go next; no queued copy may still use them.
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::time::Instant;

    use cuda_core::{Function, launch_kernel_on_stream};

    use super::*;
    use crate::consumer::Graph;

    const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// buf[i] += 1 for every i < *count, grid-stride.
.visible .entry bump(.param .u64 buf, .param .u64 count)
{
    .reg .pred p;
    .reg .b32 r<8>;
    .reg .b64 rd<6>;
    ld.param.u64 rd1, [buf];
    ld.param.u64 rd2, [count];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    ld.global.u32 r1, [rd2];
    mov.u32 r2, %ctaid.x;
    mov.u32 r3, %ntid.x;
    mov.u32 r4, %tid.x;
    mad.lo.u32 r5, r2, r3, r4;
    mov.u32 r6, %nctaid.x;
    mul.lo.u32 r6, r6, r3;
LOOP:
    setp.ge.u32 p, r5, r1;
    @p bra DONE;
    mul.wide.u32 rd3, r5, 4;
    add.s64 rd4, rd1, rd3;
    ld.global.u32 r7, [rd4];
    add.u32 r7, r7, 1;
    st.global.u32 [rd4], r7;
    add.u32 r5, r5, r6;
    bra LOOP;
DONE:
    ret;
}

// One thread sets gate[1], then re-reads buf[0..n) until the host sets
// gate[0] or 20 s pass, then stores the sum of buf[0..n) to out[0] and
// whether it was released to out[1]. gate is pinned host memory.
.visible .entry hold_sum(.param .u64 buf, .param .u32 n, .param .u64 gate, .param .u64 out)
{
    .reg .pred p, q;
    .reg .b32 r<9>;
    .reg .b64 rd<10>;
    ld.param.u64 rd1, [buf];
    ld.param.u32 r1, [n];
    ld.param.u64 rd2, [gate];
    ld.param.u64 rd3, [out];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    cvta.to.global.u64 rd3, rd3;
    mov.u32 r7, 1;
    st.volatile.global.u32 [rd2+4], r7;
    mov.u64 rd4, %globaltimer;
    mov.u32 r2, 0;
    mov.u32 r3, 0;
SPIN:
    mul.wide.u32 rd5, r2, 4;
    add.s64 rd6, rd1, rd5;
    ld.volatile.global.u32 r4, [rd6];
    xor.b32 r3, r3, r4;
    add.u32 r2, r2, 1;
    setp.ge.u32 p, r2, r1;
    @p mov.u32 r2, 0;
    ld.volatile.global.u32 r8, [rd2];
    setp.ne.u32 q, r8, 0;
    @q bra SUMS;
    mov.u64 rd7, %globaltimer;
    sub.s64 rd8, rd7, rd4;
    setp.lt.s64 p, rd8, 20000000000;
    @p bra SPIN;
SUMS:
    selp.u32 r7, 1, 0, q;
    mov.u32 r5, 0;
    mov.u32 r6, 0;
SUM:
    setp.ge.u32 p, r5, r1;
    @p bra END;
    mul.wide.u32 rd5, r5, 4;
    add.s64 rd6, rd1, rd5;
    ld.global.u32 r4, [rd6];
    add.u32 r6, r6, r4;
    add.u32 r5, r5, 1;
    bra SUM;
END:
    st.global.u32 [rd3], r6;
    st.global.u32 [rd3+4], r7;
    ret;
}

// ctl = [done, avail]. out[i] = sum of the six u32 of record i, for i in
// done..avail, grid-stride.
.visible .entry rec_sum(.param .u64 recs, .param .u64 ctl, .param .u64 out)
{
    .reg .pred p;
    .reg .b32 r<10>;
    .reg .b64 rd<6>;
    ld.param.u64 rd1, [recs];
    ld.param.u64 rd2, [ctl];
    ld.param.u64 rd3, [out];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    cvta.to.global.u64 rd3, rd3;
    ld.global.u32 r1, [rd2];
    ld.global.u32 r2, [rd2+4];
    mov.u32 r3, %ctaid.x;
    mov.u32 r4, %ntid.x;
    mov.u32 r5, %tid.x;
    mad.lo.u32 r3, r3, r4, r5;
    add.u32 r3, r3, r1;
    mov.u32 r6, %nctaid.x;
    mul.lo.u32 r6, r6, r4;
RLOOP:
    setp.ge.u32 p, r3, r2;
    @p bra RDONE;
    mul.wide.u32 rd4, r3, 24;
    add.s64 rd4, rd1, rd4;
    ld.global.u32 r7, [rd4];
    ld.global.u32 r8, [rd4+4];
    add.u32 r7, r7, r8;
    ld.global.u32 r8, [rd4+8];
    add.u32 r7, r7, r8;
    ld.global.u32 r8, [rd4+12];
    add.u32 r7, r7, r8;
    ld.global.u32 r8, [rd4+16];
    add.u32 r7, r7, r8;
    ld.global.u32 r8, [rd4+20];
    add.u32 r7, r7, r8;
    mul.wide.u32 rd5, r3, 4;
    add.s64 rd5, rd3, rd5;
    st.global.u32 [rd5], r7;
    add.u32 r3, r3, r6;
    bra RLOOP;
RDONE:
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

    fn ctx() -> Arc<Context> {
        Context::new(0).expect("GPU 0")
    }

    fn htod(dst: CUdeviceptr, src: &[u32]) {
        unsafe { sys::cuMemcpyHtoD_v2(dst, src.as_ptr().cast(), src.len() * 4) }
            .result()
            .unwrap();
    }

    fn dtoh(src: CUdeviceptr, n: usize) -> Vec<u32> {
        let mut v = vec![0u32; n];
        unsafe { sys::cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), src, n * 4) }
            .result()
            .unwrap();
        v
    }

    fn memset32(dst: CUdeviceptr, v: u32, n: usize, s: &Stream) {
        crate::consumer::memset32(dst, v, n, s).unwrap();
    }

    /// Maps `r` and returns the chunk ranges `init` saw.
    fn fresh(b: &mut GrowBuf, r: Range<usize>) -> Result<Vec<Range<usize>>, GrowError> {
        let mut seen = Vec::new();
        b.map_with(r, |c| {
            seen.push(c);
            Ok(())
        })?;
        Ok(seen)
    }

    /// `hold_sum` running on a stream until [`Hold::release`]. Released by
    /// the host, not by a timer, so a driver call delayed by other processes
    /// on the GPU cannot outlast it; only a call that waits for it can.
    struct Hold {
        gate: PinnedHostBuffer<u32>,
    }

    impl Hold {
        fn launch(f: &Function, s: &Stream, buf: CUdeviceptr, n: u32, out: CUdeviceptr) -> Self {
            let gate = PinnedHostBuffer::<u32>::zeroed(s.context(), 2).unwrap();
            // Pinned memory has one address for host and device (UVA).
            let (mut p_buf, mut p_n, mut p_gate, mut p_out) = (buf, n, gate.as_ptr() as u64, out);
            let mut params = [
                (&raw mut p_buf).cast::<c_void>(),
                (&raw mut p_n).cast(),
                (&raw mut p_gate).cast(),
                (&raw mut p_out).cast(),
            ];
            unsafe { launch_kernel_on_stream(f, (1, 1, 1), (1, 1, 1), 0, s, &mut params) }.unwrap();
            let t = Instant::now();
            while unsafe { std::ptr::read_volatile(gate.as_ptr().add(1)) } == 0 {
                assert!(t.elapsed().as_secs() < 30, "hold_sum did not start");
                std::thread::yield_now();
            }
            Self { gate }
        }

        fn release(&self) {
            unsafe { std::ptr::write_volatile(self.gate.as_ptr().cast_mut(), 1) }
        }
    }

    #[test]
    fn growth_keeps_the_address_and_the_mapped_bytes() {
        let ctx = ctx();
        let mut b = GrowBuf::new(
            &ctx,
            6 * vmm::allocation_granularity(ctx.cu_device()).unwrap(),
            1,
        )
        .unwrap();
        let ck = b.chunk_bytes();
        let base = b.ptr();
        assert_eq!(b.capacity() % ck, 0);
        assert_eq!(b.capacity(), 6 * ck);
        assert!(fresh(&mut b, 0..0).unwrap().is_empty());
        assert_eq!(fresh(&mut b, 0..1).unwrap(), vec![0..ck]);
        let pat: Vec<u32> = (0..(ck / 4) as u32)
            .map(|i| i.wrapping_mul(2654435761))
            .collect();
        htod(base, &pat);
        // A sparse chunk, then the gap and one more, ascending.
        assert_eq!(
            fresh(&mut b, 3 * ck..3 * ck + 8).unwrap(),
            vec![3 * ck..4 * ck]
        );
        assert_eq!(
            fresh(&mut b, ck / 2..5 * ck).unwrap(),
            vec![ck..2 * ck, 2 * ck..3 * ck, 4 * ck..5 * ck]
        );
        assert!(fresh(&mut b, 0..5 * ck).unwrap().is_empty());
        assert!(b.is_mapped(0..5 * ck));
        assert!(!b.is_mapped(4 * ck..5 * ck + 1));
        assert_eq!(b.mapped_bytes(), 5 * ck);
        assert_eq!(b.ptr(), base);
        assert_eq!(dtoh(base, pat.len()), pat);
        let tail: Vec<u32> = (0..(4 * ck / 4) as u32).collect();
        htod(base + ck as u64, &tail);
        assert_eq!(dtoh(base + ck as u64, tail.len()), tail);
        let cap = b.capacity();
        assert!(matches!(
            b.map(cap - 4..cap + 1),
            Err(GrowError::Capacity { end, capacity }) if end == cap + 1 && capacity == cap
        ));
        assert!(!b.is_mapped(cap - 4..cap));
        // A failed init leaves its chunk backed and the buffer failed.
        let bad = DriverError(sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
        assert!(matches!(
            b.map_with(5 * ck..5 * ck + 1, |_| Err(bad)),
            Err(GrowError::Driver(e)) if e == bad
        ));
        assert!(b.is_mapped(5 * ck..6 * ck));
        assert_eq!(b.mapped_bytes(), 6 * ck);
        assert!(matches!(b.map(0..1), Err(GrowError::Failed)));
    }

    /// Chunks are mapped while a running kernel keeps reading the mapped
    /// part; the mapping must not wait for the kernel, and the kernel's reads
    /// stay correct.
    #[test]
    fn mapping_does_not_wait_for_a_kernel_reading_the_buffer() {
        let ctx = ctx();
        let m = ctx.load_module_from_ptx_src(PTX).unwrap();
        let hold = m.load_function("hold_sum").unwrap();
        let s = ctx.new_stream().unwrap();
        let mut b = GrowBuf::new(&ctx, 256 << 20, 2 << 20).unwrap();
        let ck = b.chunk_bytes();
        b.map(0..ck).unwrap();
        let n = ck / 4;
        let vals: Vec<u32> = (0..n as u32).map(|i| i % 1000).collect();
        htod(b.ptr(), &vals);
        let mut out = GrowBuf::new(&ctx, 1, 1).unwrap();
        out.map(0..8).unwrap();
        let done = ctx.new_event_with_flags(None).unwrap();
        let h = Hold::launch(&hold, &s, b.ptr(), n as u32, out.ptr());
        done.record(&s).unwrap();
        let t = Instant::now();
        let cap = b.capacity();
        let fresh = fresh(&mut b, ck..cap).unwrap();
        let map_ms = t.elapsed().as_secs_f64() * 1e3;
        let running = !done.query().unwrap();
        h.release();
        s.synchronize().unwrap();
        eprintln!(
            "grow: mapped {} MiB in {map_ms:.1} ms under a running kernel",
            (fresh.len() * ck) >> 20
        );
        let want: Vec<_> = (1..b.capacity() / ck)
            .map(|c| c * ck..(c + 1) * ck)
            .collect();
        assert_eq!(fresh, want);
        assert!(running, "mapping returned only after the kernel finished");
        let want = vals.iter().fold(0u32, |a, &v| a.wrapping_add(v));
        assert_eq!(dtoh(out.ptr(), 2), vec![want, 1]);
        let tail: Vec<u32> = (0..1024).collect();
        let at = b.ptr() + (b.capacity() - 4096) as u64;
        htod(at, &tail);
        assert_eq!(dtoh(at, 1024), tail);
    }

    /// One graph, captured once with the base pointer and a device-side
    /// count, is replayed while the buffer grows underneath it; nothing is
    /// recaptured, updated or synchronised between rounds.
    #[test]
    fn a_graph_captured_once_replays_across_growth() {
        let ctx = ctx();
        let m = ctx.load_module_from_ptx_src(PTX).unwrap();
        let bump = m.load_function("bump").unwrap();
        let s = ctx.new_stream().unwrap();
        let mut b = GrowBuf::new(&ctx, 64 << 20, 1 << 20).unwrap();
        let mut count = GrowBuf::new(&ctx, 4, 1).unwrap();
        count.map(0..4).unwrap();
        let (mut p_buf, mut p_cnt) = (b.ptr(), count.ptr());
        let mut params = [(&raw mut p_buf).cast::<c_void>(), (&raw mut p_cnt).cast()];
        let graph = Graph::capture(&s, || unsafe {
            launch_kernel_on_stream(&bump, (128, 1, 1), (256, 1, 1), 0, &s, &mut params)
        })
        .unwrap();
        // Element counts per round: growth by uneven steps, a round with no
        // growth, and a final step to the whole reservation.
        let cap_elems = b.capacity() / 4;
        let rounds = [1000, 300_000, 300_000, 1_000_000, 4_000_000, cap_elems];
        for &n in &rounds {
            let base = b.ptr();
            b.map_with(0..n * 4, |r| {
                crate::consumer::memset32(base + r.start as u64, 0, r.len() / 4, &s)
            })
            .unwrap();
            memset32(count.ptr(), n as u32, 1, &s);
            graph.launch(&s).unwrap();
        }
        s.synchronize().unwrap();
        assert_eq!(b.ptr(), p_buf);
        let got = dtoh(b.ptr(), cap_elems);
        let bad = got
            .iter()
            .enumerate()
            .filter(|&(i, &v)| v != rounds.iter().filter(|&&n| i < n).count() as u32)
            .count();
        assert_eq!(bad, 0, "elements with a wrong replay count");
    }

    /// Records are pushed in uneven batches across chunk boundaries while one
    /// graph, captured before the first push, sums every new record. A kernel
    /// running ahead on the same stream is held until all the pushes return:
    /// none waits for it.
    #[test]
    fn a_table_grows_while_one_graph_reads_it() {
        let ctx = ctx();
        let m = ctx.load_module_from_ptx_src(PTX).unwrap();
        let (hold, sum, adv) = (
            m.load_function("hold_sum").unwrap(),
            m.load_function("rec_sum").unwrap(),
            m.load_function("advance").unwrap(),
        );
        let s = ctx.new_stream().unwrap();
        let total = 300_000usize;
        let mut t = GrowTable::<[u32; 6]>::new(&s, total, 1).unwrap();
        let ck = t.buf.chunk_bytes();
        assert!(total * 24 > 3 * ck, "the table spans several chunks");
        let mut out = GrowBuf::new(&ctx, total * 4, 1).unwrap();
        out.map(0..total * 4).unwrap();
        let mut ctl = GrowBuf::new(&ctx, 8, 1).unwrap();
        ctl.map(0..8).unwrap();
        memset32(ctl.ptr(), 0, 2, &s);
        let (mut p_recs, mut p_ctl, mut p_out) = (t.ptr(), ctl.ptr(), out.ptr());
        let mut sum_params = [
            (&raw mut p_recs).cast::<c_void>(),
            (&raw mut p_ctl).cast(),
            (&raw mut p_out).cast(),
        ];
        let mut a_ctl = ctl.ptr();
        let mut adv_params = [(&raw mut a_ctl).cast::<c_void>()];
        let graph = Graph::capture(&s, || unsafe {
            launch_kernel_on_stream(&sum, (64, 1, 1), (256, 1, 1), 0, &s, &mut sum_params)?;
            launch_kernel_on_stream(&adv, (1, 1, 1), (1, 1, 1), 0, &s, &mut adv_params)
        })
        .unwrap();

        // Hold the stream on a mapped chunk of its own, ahead of every push.
        let mut scratch = GrowBuf::new(&ctx, 1, 1).unwrap();
        scratch.map(0..4096).unwrap();
        let h = Hold::launch(&hold, &s, scratch.ptr(), 1000, scratch.ptr() + 4000);
        let held = ctx.new_event_with_flags(None).unwrap();
        held.record(&s).unwrap();

        let rec = |i: usize| {
            let i = i as u32;
            [i, i.wrapping_mul(3), i ^ 0x5555, 7, i >> 3, 1]
        };
        let sizes = [1usize, 7, 5000, 33, 60_000, 2, 100_000, 999];
        let (mut n, mut k) = (0usize, 0usize);
        let t0 = Instant::now();
        while n < total {
            let b = sizes[k % sizes.len()].min(total - n);
            k += 1;
            let recs: Vec<[u32; 6]> = (n..n + b).map(rec).collect();
            assert_eq!(t.push(&recs).unwrap(), n..n + b);
            n += b;
            memset32(ctl.ptr() + 4, n as u32, 1, &s);
            graph.launch(&s).unwrap();
        }
        let push_ms = t0.elapsed().as_secs_f64() * 1e3;
        let holding = !held.query().unwrap();
        h.release();
        s.synchronize().unwrap();
        eprintln!("grow: {k} pushes of {total} records in {push_ms:.1} ms behind a running kernel");
        assert!(holding, "a push waited for the kernel ahead of it");
        assert_eq!(
            dtoh(scratch.ptr() + 4004, 1),
            vec![1],
            "the held kernel timed out"
        );
        let got = dtoh(out.ptr(), total);
        let bad = (0..total)
            .filter(|&i| got[i] != rec(i).iter().fold(0u32, |a, &v| a.wrapping_add(v)))
            .count();
        assert_eq!(bad, 0, "records summed wrong");
        assert_eq!(dtoh(ctl.ptr(), 2), vec![total as u32; 2]);
        assert_eq!(t.len(), total);
        assert_eq!(t.mapped_bytes(), (total * 24).div_ceil(ck) * ck);
        assert!(matches!(
            t.push(&[rec(0)]),
            Err(GrowError::Capacity { end, capacity }) if end == (total + 1) * 24 && capacity == total * 24
        ));
        assert_eq!(t.len(), total);
    }
}
