//! Device buffers that grow without moving (DESIGN.md §3 B2, B6; §5
//! invariant 9). One virtual range is reserved up front and physical chunks
//! are mapped into it as the scan reaches them, so a device pointer captured
//! in a CUDA graph stays valid for the whole measurement. Mapping a chunk
//! touches no mapped chunk, so kernels and graph replays reading the mapped
//! part keep running.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use cuda_bindings::CUdeviceptr;
use cuda_core::vmm::{self, Mapping, PhysicalAllocation, VirtualReservation};
use cuda_core::{Context, DriverError};

#[derive(Debug)]
pub enum GrowError {
    /// The requested range ends past the reservation.
    Capacity {
        end: usize,
        capacity: usize,
    },
    Driver(DriverError),
}

impl fmt::Display for GrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity { end, capacity } => {
                write!(f, "range end {end} past the {capacity}-byte reservation")
            }
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
/// address and contents.
pub struct GrowBuf {
    // Declaration order is drop order: mappings before the reservation.
    chunks: Vec<Option<Mapping>>,
    va: VirtualReservation,
    chunk: usize,
    mapped: usize,
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

    /// Backs every chunk that overlaps `r`. Returns the byte ranges that were
    /// newly mapped (coalesced, ascending); their contents are undefined
    /// until the caller writes them.
    pub fn map(&mut self, r: Range<usize>) -> Result<Vec<Range<usize>>, GrowError> {
        if r.end > self.capacity() {
            return Err(GrowError::Capacity {
                end: r.end,
                capacity: self.capacity(),
            });
        }
        let mut fresh: Vec<Range<usize>> = Vec::new();
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
            let b = c * self.chunk..(c + 1) * self.chunk;
            match fresh.last_mut() {
                Some(l) if l.end == b.start => l.end = b.end,
                _ => fresh.push(b),
            }
        }
        Ok(fresh)
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
        // The fields' own drops unmap and free; they need the context current.
        let _ = self.ctx.bind_to_thread();
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::time::Instant;

    use cuda_bindings as sys;
    use cuda_core::{IntoResult, Stream, launch_kernel_on_stream};

    use super::*;

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

// One thread re-reads buf[0..n) until `cycles` clocks have passed, then
// stores the sum of buf[0..n) to out[0].
.visible .entry spin_sum(.param .u64 buf, .param .u32 n, .param .u64 cycles, .param .u64 out)
{
    .reg .pred p;
    .reg .b32 r<8>;
    .reg .b64 rd<10>;
    ld.param.u64 rd1, [buf];
    ld.param.u32 r1, [n];
    ld.param.u64 rd2, [cycles];
    ld.param.u64 rd3, [out];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd3, rd3;
    mov.u64 rd4, %clock64;
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
    mov.u64 rd7, %clock64;
    sub.s64 rd8, rd7, rd4;
    setp.lt.s64 p, rd8, rd2;
    @p bra SPIN;
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
    st.global.u32 [rd3+4], r3;
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
        unsafe { sys::cuMemsetD32Async(dst, v, n, s.cu_stream()) }
            .result()
            .unwrap();
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
        assert!(b.map(0..0).unwrap().is_empty());
        assert_eq!(b.map(0..1).unwrap(), vec![0..ck]);
        let pat: Vec<u32> = (0..(ck / 4) as u32)
            .map(|i| i.wrapping_mul(2654435761))
            .collect();
        htod(base, &pat);
        // A sparse chunk, then the gap and one more: coalesced, ascending.
        assert_eq!(b.map(3 * ck..3 * ck + 8).unwrap(), vec![3 * ck..4 * ck]);
        assert_eq!(
            b.map(ck / 2..5 * ck).unwrap(),
            vec![ck..3 * ck, 4 * ck..5 * ck]
        );
        assert!(b.map(0..5 * ck).unwrap().is_empty());
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
    }

    /// Chunks are mapped while a kernel keeps reading the mapped part; the
    /// mapping must not wait for the kernel, and the kernel's reads stay
    /// correct.
    #[test]
    fn mapping_does_not_wait_for_a_kernel_reading_the_buffer() {
        let ctx = ctx();
        let m = ctx.load_module_from_ptx_src(PTX).unwrap();
        let spin = m.load_function("spin_sum").unwrap();
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
        let (mut p_buf, mut p_n, mut p_cyc, mut p_out) =
            (b.ptr(), n as u32, 1_000_000_000u64, out.ptr());
        let mut params = [
            (&raw mut p_buf).cast::<c_void>(),
            (&raw mut p_n).cast(),
            (&raw mut p_cyc).cast(),
            (&raw mut p_out).cast(),
        ];
        let t = Instant::now();
        unsafe { launch_kernel_on_stream(&spin, (1, 1, 1), (1, 1, 1), 0, &s, &mut params) }
            .unwrap();
        done.record(&s).unwrap();
        let fresh = b.map(ck..b.capacity()).unwrap();
        let map_ms = t.elapsed().as_secs_f64() * 1e3;
        let running = !done.query().unwrap();
        s.synchronize().unwrap();
        let kernel_ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "grow: mapped {} MiB in {map_ms:.1} ms while the kernel ran {kernel_ms:.1} ms",
            fresh[0].len() >> 20
        );
        assert_eq!(fresh, vec![ck..b.capacity()]);
        assert!(running, "mapping returned only after the kernel finished");
        let want = vals.iter().fold(0u32, |a, &v| a.wrapping_add(v));
        assert_eq!(dtoh(out.ptr(), 1)[0], want);
        let tail: Vec<u32> = (0..1024).collect();
        let at = b.ptr() + (b.capacity() - 4096) as u64;
        htod(at, &tail);
        assert_eq!(dtoh(at, 1024), tail);
    }

    /// RAII for a captured graph and its executable.
    struct Graph(sys::CUgraph, sys::CUgraphExec);

    impl Drop for Graph {
        fn drop(&mut self) {
            unsafe {
                let _ = sys::cuGraphExecDestroy(self.1);
                let _ = sys::cuGraphDestroy(self.0);
            }
        }
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
        let graph = unsafe {
            s.begin_capture(sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .unwrap();
            launch_kernel_on_stream(&bump, (128, 1, 1), (256, 1, 1), 0, &s, &mut params).unwrap();
            let g = s.end_capture().unwrap();
            let mut exec = MaybeUninit::uninit();
            sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), g, 0)
                .result()
                .unwrap();
            Graph(g, exec.assume_init())
        };
        // Element counts per round: growth by uneven steps, a round with no
        // growth, and a final step to the whole reservation.
        let cap_elems = b.capacity() / 4;
        let rounds = [1000, 300_000, 300_000, 1_000_000, 4_000_000, cap_elems];
        for &n in &rounds {
            for r in b.map(0..n * 4).unwrap() {
                memset32(b.ptr() + r.start as u64, 0, r.len() / 4, &s);
            }
            memset32(count.ptr(), n as u32, 1, &s);
            unsafe { sys::cuGraphLaunch(graph.1, s.cu_stream()) }
                .result()
                .unwrap();
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
}
