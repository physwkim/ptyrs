//! Device buffers that grow without moving (DESIGN.md §3 B2, B6; §5
//! invariant 9). One virtual range is reserved up front and physical chunks
//! are mapped into it as the scan reaches them, so a device pointer captured
//! in a CUDA graph stays valid for the whole measurement. Mapping a chunk
//! touches no mapped chunk, so kernels and graph replays reading the mapped
//! part keep running.
//!
//! Backing a chunk is a driver call that can take far longer than an epoch
//! (cuMemCreate stalls about 130 ms every ~900 MiB a process allocates), so
//! during a scan a [`Backer`] does it on a thread of its own, and the thread
//! that queues work on the buffer only queues each chunk's first contents
//! ([`GrowBuf::init_with`]), which never backs.

use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use cuda_async::device_operation::ExecutionContext;
use cuda_bindings::{self as sys, CUdeviceptr};
use cuda_core::vmm::{self, Mapping, PhysicalAllocation, VirtualReservation};
use cuda_core::{
    Context, DeviceAllocation, DeviceBuffer, DeviceCopy, DriverError, IntoResult, PinnedHostBuffer,
    RawDeviceBuffer, Stream,
};

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
    /// Bytes `start..end` were to be initialised but a chunk of them was
    /// not backed: no [`Backer`] ran ahead of the push.
    Unbacked {
        start: usize,
        end: usize,
    },
    /// A landed count below the last one or past the scan's `nframe`
    /// ([`LiveFrames::land`](crate::stack::LiveFrames::land)).
    Land {
        n: u32,
        landed: u32,
        nframe: u32,
    },
    Driver(DriverError),
    /// A leased access or a kernel launch failed.
    Gpu(ptyrs_gpu::Error),
    /// An LLK record read back wrong.
    Llk(ptyrs_gpu::RingError),
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
            Self::Unbacked { start, end } => {
                write!(f, "bytes {start}..{end} were not backed ahead of their use")
            }
            Self::Land { n, landed, nframe } => write!(
                f,
                "landing {n} frames: {landed} already landed, the scan has {nframe}"
            ),
            Self::Driver(e) => write!(f, "{e}"),
            Self::Gpu(e) => write!(f, "{e}"),
            Self::Llk(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GrowError {}

impl From<DriverError> for GrowError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<ptyrs_gpu::Error> for GrowError {
    fn from(e: ptyrs_gpu::Error) -> Self {
        Self::Gpu(e)
    }
}

impl From<ptyrs_gpu::RingError> for GrowError {
    fn from(e: ptyrs_gpu::RingError) -> Self {
        match e {
            ptyrs_gpu::RingError::Gpu(e) => Self::Gpu(e),
            e => Self::Llk(e),
        }
    }
}

impl From<cuda_async::error::DeviceError> for GrowError {
    fn from(e: cuda_async::error::DeviceError) -> Self {
        Self::Gpu(e.into())
    }
}

/// The reservation and its backed chunks, the owner behind
/// [`GrowBuf::storage`]: every view and every lease taken through it holds
/// this, so a mapping lives while any work queued through it may reach it.
struct Reserved {
    // Declaration order is drop order: mappings before the reservation.
    /// Each chunk's mapping, set once by a holder of `backing`; reading it
    /// never waits for a chunk being backed.
    chunks: Vec<OnceLock<Mapping>>,
    /// Held while backing, so no chunk is mapped twice.
    backing: Mutex<()>,
    chunk: usize,
    va: VirtualReservation,
    ctx: Arc<Context>,
}

impl Reserved {
    fn span(&self, r: &Range<usize>) -> Range<usize> {
        if r.is_empty() {
            return 0..0;
        }
        r.start / self.chunk..r.end.div_ceil(self.chunk)
    }

    fn check(&self, r: &Range<usize>) -> Result<(), GrowError> {
        if r.end > self.va.size() {
            return Err(GrowError::Capacity {
                end: r.end,
                capacity: self.va.size(),
            });
        }
        Ok(())
    }

    fn is_backed(&self, c: usize) -> bool {
        self.chunks[c].get().is_some()
    }

    /// Backs every chunk overlapping `r` that is not backed yet, ascending.
    /// A chunk that fails to back stays unbacked.
    fn back(&self, r: Range<usize>) -> Result<(), GrowError> {
        self.check(&r)?;
        let span = self.span(&r);
        if span.clone().all(|c| self.is_backed(c)) {
            return Ok(());
        }
        let _backing = self.backing.lock().unwrap();
        self.ctx.bind_to_thread()?;
        let dev = self.ctx.cu_device();
        for c in span.filter(|&c| !self.is_backed(c)) {
            let va = self.va.base() + (c * self.chunk) as u64;
            // The mapping retains the physical memory; the handle can go.
            let phys = PhysicalAllocation::new(dev, self.chunk)?;
            let m = Mapping::new(va, self.chunk, &phys, 0)?;
            vmm::set_access(va, self.chunk, &[dev])?;
            assert!(self.chunks[c].set(m).is_ok(), "chunk {c} backed twice");
        }
        Ok(())
    }
}

// SAFETY: the base and size are the reservation's, fixed for its life, in
// `ctx`, which the owner keeps alive.
unsafe impl DeviceAllocation for Reserved {
    fn device_ptr(&self) -> CUdeviceptr {
        self.va.base()
    }

    fn len_bytes(&self) -> usize {
        self.va.size()
    }

    fn context(&self) -> Arc<Context> {
        self.ctx.clone()
    }
}

impl Drop for Reserved {
    fn drop(&mut self) {
        // Leased work holds this until it completes, but raw copies (the
        // uploader's into the frame store) do not. Unlike cuMemFree, cuMemUnmap
        // does not wait for queued work, and a copy or kernel reaching an
        // unmapped chunk is an MMU fault (Xid 31); no stream is known here.
        let _ = self.ctx.synchronize();
    }
}

/// Backs chunks of a [`GrowBuf`] on any thread; its owner then queues
/// their first contents with [`GrowBuf::init_with`] before using them.
#[derive(Clone)]
pub struct Backer(Arc<Reserved>);

impl Backer {
    /// Backs every chunk that overlaps `r` and is not backed yet, ascending.
    /// A chunk that fails to back stays unbacked and the buffer usable.
    pub fn back(&self, r: Range<usize>) -> Result<(), GrowError> {
        self.0.back(r)
    }
}

/// A fixed device address range whose chunks are backed on demand. Chunks
/// are never unmapped before the last holder of [`Self::storage`] (a view,
/// a lease, a captured graph) is gone, so every byte of a mapped chunk keeps
/// its address and contents. A chunk is ready once backed and its first
/// contents queued; only ready chunks are the owner's to use.
pub struct GrowBuf {
    alloc: Arc<Reserved>,
    /// The one access tracker over the whole reservation.
    storage: RawDeviceBuffer,
    /// Per chunk, whether it is ready; set by the owner only.
    ready: Vec<bool>,
    failed: bool,
}

impl GrowBuf {
    /// Reserves `capacity` bytes, rounded up to whole chunks of at least
    /// `chunk` bytes (rounded up to the allocation granularity). Nothing is
    /// backed yet.
    pub fn new(ctx: &Arc<Context>, capacity: usize, chunk: usize) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let granule = vmm::allocation_granularity(ctx.cu_device())?;
        let chunk = vmm::align_size(chunk.max(1), granule);
        let n = capacity.div_ceil(chunk).max(1);
        let alloc = Arc::new(Reserved {
            chunks: (0..n).map(|_| OnceLock::new()).collect(),
            backing: Mutex::new(()),
            chunk,
            // Mapping needs granule alignment only, and the driver refuses
            // an alignment that is not a power of two, as a chunk may be.
            va: VirtualReservation::new(n * chunk, granule)?,
            ctx: ctx.clone(),
        });
        Ok(Self {
            storage: RawDeviceBuffer::foreign(alloc.clone(), n * chunk),
            alloc,
            ready: vec![false; n],
            failed: false,
        })
    }

    /// A handle that backs this buffer's chunks from another thread.
    pub fn backer(&self) -> Backer {
        Backer(self.alloc.clone())
    }

    /// Base address; the same for the buffer's whole life.
    pub fn ptr(&self) -> CUdeviceptr {
        self.alloc.va.base()
    }

    pub fn capacity(&self) -> usize {
        self.alloc.va.size()
    }

    pub fn chunk_bytes(&self) -> usize {
        self.alloc.chunk
    }

    /// Bytes backed so far, ready or not.
    pub fn mapped_bytes(&self) -> usize {
        let a = &self.alloc;
        (0..a.chunks.len()).filter(|&c| a.is_backed(c)).count() * a.chunk
    }

    /// The whole reservation, for leasing every access to it.
    pub fn storage(&self) -> &RawDeviceBuffer {
        &self.storage
    }

    /// The first `len` `T`s as a typed view sharing [`Self::storage`]'s
    /// tracker and owner.
    ///
    /// # Safety
    /// Work through the view may only touch elements whose bytes are backed
    /// when it runs.
    pub unsafe fn typed<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>, DriverError> {
        let bytes = self.storage.slice_bytes(0, len * size_of::<T>())?;
        // SAFETY: the reservation is chunk-aligned; the caller bounds the use.
        unsafe { DeviceBuffer::from_storage(bytes, len) }
    }

    /// Whether every byte of `r` is ready.
    pub fn is_ready(&self, r: Range<usize>) -> bool {
        r.end <= self.capacity() && self.alloc.span(&r).all(|c| self.ready[c])
    }

    /// Backs `r` on this thread and readies it without initialisation, for
    /// buffers set up before a scan and buffers whose new bytes are written
    /// before anything reads them.
    pub fn map(&mut self, r: Range<usize>) -> Result<(), GrowError> {
        self.alloc.back(r.clone())?;
        self.init_with(r, |_| Ok(()))
    }

    /// Readies every chunk that overlaps `r`, ascending, handing each one
    /// not ready yet to `init` (to queue its first contents) before the
    /// next. Never backs: a chunk no [`Backer`] backed fails the call with
    /// [`GrowError::Unbacked`] and leaves the buffer usable. A chunk whose
    /// `init` fails fails the buffer: every later call returns
    /// [`GrowError::Failed`], so no chunk is ever used without its first
    /// contents.
    pub fn init_with(
        &mut self,
        r: Range<usize>,
        mut init: impl FnMut(Range<usize>) -> Result<(), GrowError>,
    ) -> Result<(), GrowError> {
        if self.failed {
            return Err(GrowError::Failed);
        }
        self.alloc.check(&r)?;
        let ck = self.alloc.chunk;
        for c in self.alloc.span(&r) {
            if self.ready[c] {
                continue;
            }
            if !self.alloc.is_backed(c) {
                return Err(GrowError::Unbacked {
                    start: r.start,
                    end: r.end,
                });
            }
            if let Err(e) = init(c * ck..(c + 1) * ck) {
                self.failed = true;
                return Err(e);
            }
            self.ready[c] = true;
        }
        Ok(())
    }
}

/// Append-only device table of `T`, such as the live address table
/// (DESIGN.md §3 B6): record `i` stays at `ptr() + i * size_of::<T>()`.
/// Records are copied from a pinned mirror they never leave, so a push
/// never waits for the stream and no queued copy outlives its source. A
/// [`Self::backer`] backs their bytes before they are pushed.
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

    /// Backs the bytes of records, record `i` at `i * size_of::<T>()`.
    pub fn backer(&self) -> Backer {
        self.buf.backer()
    }

    /// The whole reservation, for leasing reads of the records.
    pub fn storage(&self) -> &RawDeviceBuffer {
        self.buf.storage()
    }

    /// The first `len` records as a typed view; see [`GrowBuf::typed`].
    ///
    /// # Safety
    /// Work through the view may only read records already pushed when it
    /// runs.
    pub unsafe fn typed(&self, len: usize) -> Result<DeviceBuffer<T>, DriverError> {
        // SAFETY: forwarded to the caller.
        unsafe { self.buf.typed(len) }
    }

    /// Appends `recs`, which a [`Self::backer`] backed, and queues their
    /// copy on the table's stream, leased as a write, so kernels queued
    /// there afterwards see them. Returns their indices.
    pub fn push(&mut self, recs: &[T]) -> Result<Range<usize>, GrowError> {
        let sz = size_of::<T>();
        let idx = self.len..self.len + recs.len();
        if idx.end > self.max_len() {
            return Err(GrowError::Capacity {
                end: idx.end * sz,
                capacity: self.max_len() * sz,
            });
        }
        self.buf
            .init_with(idx.start * sz..idx.end * sz, |_| Ok(()))?;
        if recs.is_empty() {
            return Ok(idx);
        }
        self.mirror[idx.clone()].copy_from_slice(recs);
        let ctx = ExecutionContext::new(self.stream.clone());
        ptyrs_gpu::lease(&ctx, &[], &[self.buf.storage()])?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: the destination is ready, checked above; the source is the
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
    use crate::graph::LiveGraph;

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

    fn memset32_async(dst: CUdeviceptr, v: u32, n: usize, s: &Stream) -> Result<(), DriverError> {
        unsafe { sys::cuMemsetD32Async(dst, v, n, s.cu_stream()) }.result()
    }

    fn memset32(dst: CUdeviceptr, v: u32, n: usize, s: &Stream) {
        memset32_async(dst, v, n, s).unwrap();
    }

    /// Backs and readies `r` and returns the chunk ranges `init` saw.
    fn fresh(b: &mut GrowBuf, r: Range<usize>) -> Result<Vec<Range<usize>>, GrowError> {
        let mut seen = Vec::new();
        b.backer().back(r.clone())?;
        b.init_with(r, |c| {
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
        assert!(b.is_ready(0..5 * ck));
        assert!(!b.is_ready(4 * ck..5 * ck + 1));
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
        assert!(!b.is_ready(cap - 4..cap));
        // Initialising never backs: an unbacked chunk fails the call, hands
        // `init` nothing and leaves the buffer usable.
        let mut seen = 0;
        assert!(matches!(
            b.init_with(4 * ck..6 * ck, |_| {
                seen += 1;
                Ok(())
            }),
            Err(GrowError::Unbacked { start, end }) if (start, end) == (4 * ck, 6 * ck)
        ));
        assert_eq!((seen, b.mapped_bytes()), (0, 5 * ck));
        assert!(!b.is_ready(5 * ck..6 * ck));
        // A failed init leaves its chunk backed, not ready, and the buffer
        // failed.
        let bad = DriverError(sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
        b.backer().back(5 * ck..5 * ck + 1).unwrap();
        assert!(matches!(
            b.init_with(5 * ck..5 * ck + 1, |_| Err(bad.into())),
            Err(GrowError::Driver(e)) if e == bad
        ));
        assert!(!b.is_ready(5 * ck..6 * ck));
        assert_eq!(b.mapped_bytes(), 6 * ck);
        assert!(matches!(b.map(0..1), Err(GrowError::Failed)));
    }

    /// A chunk of three granules: the reservation is not chunk-aligned, and
    /// data crosses the chunk boundary intact.
    #[test]
    fn a_chunk_need_not_be_a_power_of_two() {
        let ctx = ctx();
        let g = vmm::allocation_granularity(ctx.cu_device()).unwrap();
        let mut b = GrowBuf::new(&ctx, 4 * g, 3 * g).unwrap();
        assert_eq!((b.chunk_bytes(), b.capacity()), (3 * g, 6 * g));
        b.map(0..b.capacity()).unwrap();
        assert_eq!(b.mapped_bytes(), 6 * g);
        let pat: Vec<u32> = (0..(2 * g / 4) as u32).collect();
        let at = b.ptr() + (2 * g) as u64;
        htod(at, &pat);
        assert_eq!(dtoh(at, pat.len()), pat);
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
        let (p_buf, p_cnt) = (b.ptr(), count.ptr());
        let (bs, cs) = (b.storage(), count.storage());
        let graph = LiveGraph::capture(&s, |ctx| {
            ptyrs_gpu::lease(ctx, &[cs], &[bs])?;
            let (mut p_buf, mut p_cnt) = (p_buf, p_cnt);
            let mut params = [(&raw mut p_buf).cast::<c_void>(), (&raw mut p_cnt).cast()];
            let s = ctx.get_cuda_stream();
            unsafe { launch_kernel_on_stream(&bump, (128, 1, 1), (256, 1, 1), 0, s, &mut params) }?;
            Ok(())
        })
        .unwrap();
        // Element counts per round: growth by uneven steps, a round with no
        // growth, and a final step to the whole reservation.
        let cap_elems = b.capacity() / 4;
        let rounds = [1000, 300_000, 300_000, 1_000_000, 4_000_000, cap_elems];
        for &n in &rounds {
            let base = b.ptr();
            b.backer().back(0..n * 4).unwrap();
            b.init_with(0..n * 4, |r| {
                Ok(memset32_async(base + r.start as u64, 0, r.len() / 4, &s)?)
            })
            .unwrap();
            memset32(count.ptr(), n as u32, 1, &s);
            graph.replay().unwrap();
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
        let (p_recs, p_ctl, p_out) = (t.ptr(), ctl.ptr(), out.ptr());
        let (ts, cs, os) = (t.storage(), ctl.storage(), out.storage());
        let graph = LiveGraph::capture(&s, |ctx| {
            ptyrs_gpu::lease(ctx, &[ts], &[cs, os])?;
            let (mut p_recs, mut p_ctl, mut p_out) = (p_recs, p_ctl, p_out);
            let mut sum_params = [
                (&raw mut p_recs).cast::<c_void>(),
                (&raw mut p_ctl).cast(),
                (&raw mut p_out).cast(),
            ];
            let mut a_ctl = p_ctl;
            let mut adv_params = [(&raw mut a_ctl).cast::<c_void>()];
            let s = ctx.get_cuda_stream();
            unsafe {
                launch_kernel_on_stream(&sum, (64, 1, 1), (256, 1, 1), 0, s, &mut sum_params)?;
                launch_kernel_on_stream(&adv, (1, 1, 1), (1, 1, 1), 0, s, &mut adv_params)?;
            }
            Ok(())
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
        // Pushing never backs: records no backer reached are refused.
        assert!(matches!(
            t.push(&[rec(0)]),
            Err(GrowError::Unbacked { start: 0, end: 24 })
        ));
        assert_eq!((t.len(), t.mapped_bytes()), (0, 0));
        // Another thread backs each batch before it is pushed.
        let (backer, (tx, rx)) = (t.backer(), std::sync::mpsc::channel());
        let ahead = std::thread::spawn(move || {
            let (mut n, mut k) = (0usize, 0usize);
            while n < total {
                let b = sizes[k % sizes.len()].min(total - n);
                k += 1;
                backer.back(n * 24..(n + b) * 24).unwrap();
                tx.send(b).unwrap();
                n += b;
            }
        });
        let (mut n, mut k) = (0usize, 0usize);
        let t0 = Instant::now();
        for b in rx.iter() {
            k += 1;
            let recs: Vec<[u32; 6]> = (n..n + b).map(rec).collect();
            assert_eq!(t.push(&recs).unwrap(), n..n + b);
            n += b;
            memset32(ctl.ptr() + 4, n as u32, 1, &s);
            graph.replay().unwrap();
        }
        let push_ms = t0.elapsed().as_secs_f64() * 1e3;
        let holding = !held.query().unwrap();
        ahead.join().unwrap();
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
