//! Peer reads checked by a kernel, and copy timings for every ordered device pair.
//!
//! The driver can report peer access as supported and enabled while the platform corrupts the
//! traffic (an IOMMU without the peer mapping returns all-ones loads). The halo pull and the
//! gather read peer memory with SM loads, so [`verify_reads`] checks that path before peer
//! access counts as enabled.

use std::ffi::c_void;
use std::sync::Arc;

use cuda_core::{
    Context, DeviceBuffer, Function, IntoResult, Stream, launch_kernel_on_stream, sys,
};

use crate::DistError;

struct Kernels {
    copy: Function,
    chase: Function,
}

const COPY_THREADS: u32 = 256;

impl Kernels {
    fn load(ctx: &Arc<Context>) -> Result<Self, DistError> {
        let m = ctx.load_module_from_ptx_src(PTX)?;
        Ok(Self {
            copy: m.load_function("peer_copy")?,
            chase: m.load_function("peer_chase")?,
        })
    }

    /// `dst[i] = src[i]` for `n` 16-byte vectors, loaded and stored by the stream's device.
    ///
    /// # Safety
    /// `dst` and `src` must address `n` 16-byte-aligned vectors on this device or on a peer
    /// with access enabled.
    unsafe fn copy(
        &self,
        stream: &Stream,
        dst: u64,
        src: u64,
        n: u64,
        blocks: u32,
    ) -> Result<(), DistError> {
        let (mut dst, mut src, mut n) = (dst, src, n);
        let mut params = [
            (&raw mut dst).cast::<c_void>(),
            (&raw mut src).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
        ];
        // SAFETY: the caller guarantees the extents.
        unsafe {
            launch_kernel_on_stream(
                &self.copy,
                (blocks, 1, 1),
                (COPY_THREADS, 1, 1),
                0,
                stream,
                &mut params,
            )?
        };
        Ok(())
    }

    /// One thread follows `hops` pointers from `base`, stopping at the first pointer outside
    /// `[base, base + len)` or not 8-byte aligned, and stores where it stopped in `out`.
    ///
    /// # Safety
    /// `base` must address `len` bytes on this device or a peer with access enabled, and
    /// `out` one u64 on this device.
    unsafe fn chase(
        &self,
        stream: &Stream,
        base: u64,
        len: u64,
        hops: u32,
        out: u64,
    ) -> Result<(), DistError> {
        let (mut base, mut len, mut hops, mut out) = (base, len, hops, out);
        let mut params = [
            (&raw mut base).cast::<c_void>(),
            (&raw mut len).cast::<c_void>(),
            (&raw mut hops).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
        ];
        // SAFETY: the kernel only dereferences pointers it has bounds-checked against `base`.
        unsafe {
            launch_kernel_on_stream(&self.chase, (1, 1, 1), (1, 1, 1), 0, stream, &mut params)?
        };
        Ok(())
    }
}

/// Words that differ between devices and between positions, so neither a stale buffer nor an
/// all-ones load matches.
fn pattern(ordinal: usize, words: usize) -> Vec<u32> {
    let salt = (ordinal as u32 + 1).wrapping_mul(0x85eb_ca77);
    (0..words as u32)
        .map(|i| i.wrapping_mul(0x9e37_79b1) ^ salt)
        .collect()
}

const PROBE_WORDS: usize = 1 << 16;

/// Every device of `ctxs` reads a pattern from every other device's memory with a kernel and
/// compares it. Peer access must already be enabled.
pub(crate) fn verify_reads(ctxs: &[Arc<Context>]) -> Result<(), DistError> {
    if ctxs.len() < 2 {
        return Ok(());
    }
    let streams = ctxs
        .iter()
        .map(|c| c.new_stream())
        .collect::<Result<Vec<_>, _>>()?;
    let kernels = ctxs
        .iter()
        .map(Kernels::load)
        .collect::<Result<Vec<_>, _>>()?;
    let src = ctxs
        .iter()
        .zip(&streams)
        .map(|(c, s)| DeviceBuffer::from_host(s, &pattern(c.ordinal(), PROBE_WORDS)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut bad = Vec::new();
    for (a, s) in streams.iter().enumerate() {
        for b in (0..ctxs.len()).filter(|&b| b != a) {
            let (owner, reader) = (ctxs[b].ordinal(), ctxs[a].ordinal());
            // SAFETY: access a -> b is enabled.
            if !unsafe { pull_matches(&kernels[a], s, &src[b], owner, PROBE_WORDS)? } {
                bad.push((reader, owner));
            }
        }
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(DistError::PeerReadFault(bad))
    }
}

/// The device of `stream` loads the first `words` words of `src` with a kernel and compares
/// them with `owner`'s [`pattern`].
///
/// # Safety
/// Access from the device of `stream` to the device holding `src` must be enabled.
unsafe fn pull_matches(
    kernels: &Kernels,
    stream: &Stream,
    src: &DeviceBuffer<u32>,
    owner: usize,
    words: usize,
) -> Result<bool, DistError> {
    assert!(words <= src.len() && words.is_multiple_of(4));
    let dst = DeviceBuffer::<u32>::zeroed(stream, words)?;
    // SAFETY: both buffers hold at least `words` words; the caller enabled access.
    unsafe {
        kernels.copy(
            stream,
            dst.cu_deviceptr(),
            src.cu_deviceptr(),
            (words / 4) as u64,
            64,
        )?
    };
    Ok(dst.to_host_vec(stream)? == pattern(owner, words))
}

/// The copy engine of the device holding `src` pushes its first `words` words into a new
/// buffer on the device of `dst_stream`, and the result is compared with `owner`'s [`pattern`].
fn push_matches(
    src_stream: &Stream,
    dst_stream: &Stream,
    src: &DeviceBuffer<u32>,
    owner: usize,
    words: usize,
) -> Result<bool, DistError> {
    assert!(words <= src.len());
    let dst = DeviceBuffer::<u32>::zeroed(dst_stream, words)?;
    // SAFETY: both buffers hold at least `words` words in their own contexts.
    unsafe {
        sys::cuMemcpyPeerAsync(
            dst.cu_deviceptr(),
            dst.context().cu_ctx(),
            src.cu_deviceptr(),
            src.context().cu_ctx(),
            words * 4,
            src_stream.cu_stream(),
        )
        .result()?
    };
    src_stream.synchronize()?;
    Ok(dst.to_host_vec(dst_stream)? == pattern(owner, words))
}

/// What [`bench`] found for one ordered pair: data moves from device `src` to device `dst`.
#[derive(Clone, Copy, Debug)]
pub struct PairResult {
    pub src: usize,
    pub dst: usize,
    pub outcome: Outcome,
}

#[derive(Clone, Copy, Debug)]
pub enum Outcome {
    /// A probe-sized engine copy or SM pull arrived wrong, so the pair was not timed.
    ProbeFailed,
    /// The probe passed but a word of a bulk copy or a hop of the chase arrived wrong; the
    /// timings measure corrupted traffic.
    Corrupt(Timings),
    /// Every word of both bulk copies and every hop of the chase arrived intact.
    Intact(Timings),
}

#[derive(Clone, Copy, Debug)]
pub struct Timings {
    /// Copy engine of `src` pushing the buffer to `dst` (`cuMemcpyPeerAsync`), GB/s.
    pub copy_gbps: f64,
    /// One 4-byte `cuMemcpyPeerAsync` on `src`'s stream, mean over a queue of copies, µs.
    pub copy_latency_us: f64,
    /// SMs of `dst` loading `src`'s memory, the halo pull and gather path, GB/s.
    pub pull_gbps: f64,
    /// One dependent 8-byte load by `dst` from `src`'s memory (pointer chase), µs.
    pub load_latency_us: f64,
}

const LATENCY_COPIES: u32 = 1000;
const CHASE_SLOTS: usize = 1024;
const CHASE_STRIDE: usize = 4096;
const CHASE_HOPS: u32 = 4096;
const PULL_BLOCKS: u32 = 1024;

/// Times every ordered pair of `ctxs` in turn, `reps` transfers of `bytes` per bulk timing,
/// after enabling peer access without [`verify_reads`] so a broken pair is reported instead of
/// failing the whole run. Each pair first moves a probe of at most 256 KiB by both paths, and
/// only a pair whose probe arrives intact carries bulk traffic.
///
/// # Panics
/// If `bytes` is not a positive multiple of 16 or `reps` is 0.
pub fn bench(ctxs: &[Arc<Context>], bytes: usize, reps: u32) -> Result<Vec<PairResult>, DistError> {
    assert!(
        bytes > 0 && bytes.is_multiple_of(16) && reps > 0,
        "bench of {reps} x {bytes} bytes: bytes must be a positive multiple of 16"
    );
    let words = bytes / 4;
    let streams = ctxs
        .iter()
        .map(|c| c.new_stream())
        .collect::<Result<Vec<_>, _>>()?;
    let kernels = ctxs
        .iter()
        .map(Kernels::load)
        .collect::<Result<Vec<_>, _>>()?;
    let src = ctxs
        .iter()
        .zip(&streams)
        .map(|(c, s)| DeviceBuffer::from_host(s, &pattern(c.ordinal(), words)))
        .collect::<Result<Vec<_>, _>>()?;
    let dst = streams
        .iter()
        .map(|s| DeviceBuffer::<u32>::zeroed(s, words))
        .collect::<Result<Vec<_>, _>>()?;
    let chain = streams
        .iter()
        .map(|s| {
            let mut buf = DeviceBuffer::<u64>::zeroed(s, CHASE_SLOTS * CHASE_STRIDE / 8)?;
            let base = buf.cu_deviceptr();
            let mut host = vec![0u64; buf.len()];
            for i in 0..CHASE_SLOTS {
                host[i * CHASE_STRIDE / 8] = base + (((i + 1) % CHASE_SLOTS) * CHASE_STRIDE) as u64;
            }
            buf.copy_from_host(s, &host)?;
            Ok::<_, DistError>(buf)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ends = streams
        .iter()
        .map(|s| DeviceBuffer::<u64>::zeroed(s, 1))
        .collect::<Result<Vec<_>, _>>()?;

    let mut out = Vec::new();
    for (i, a) in ctxs.iter().enumerate() {
        for (j, b) in ctxs.iter().enumerate().filter(|(j, _)| *j != i) {
            cuda_core::peer::enable_peer_access(a, b)?;
            cuda_core::peer::enable_peer_access(b, a)?;
            let (ss, ds) = (&streams[i], &streams[j]);
            let probe = words.min(PROBE_WORDS);
            // SAFETY: access j -> i is enabled.
            if !(push_matches(ss, ds, &src[i], a.ordinal(), probe)?
                && unsafe { pull_matches(&kernels[j], ds, &src[i], a.ordinal(), probe)? })
            {
                out.push(PairResult {
                    src: a.ordinal(),
                    dst: b.ordinal(),
                    outcome: Outcome::ProbeFailed,
                });
                continue;
            }
            let want = pattern(a.ordinal(), words);
            let peer_copy = |n: usize| {
                // SAFETY: both buffers hold `words >= n / 4` words in their own contexts.
                unsafe {
                    sys::cuMemcpyPeerAsync(
                        dst[j].cu_deviceptr(),
                        b.cu_ctx(),
                        src[i].cu_deviceptr(),
                        a.cu_ctx(),
                        n,
                        ss.cu_stream(),
                    )
                    .result()
                }
            };

            peer_copy(bytes)?;
            let copy_ms = timed(ss, || {
                for _ in 0..reps {
                    peer_copy(bytes)?;
                }
                Ok(())
            })?;
            let copy_ok = dst[j].to_host_vec(ds)? == want;
            let lat_ms = timed(ss, || {
                for _ in 0..LATENCY_COPIES {
                    peer_copy(4)?;
                }
                Ok(())
            })?;

            let mut d = DeviceBuffer::<u32>::zeroed(ds, words)?;
            let pull = |d: &mut DeviceBuffer<u32>| {
                // SAFETY: both buffers hold `words` words; access j -> i is enabled.
                unsafe {
                    kernels[j].copy(
                        ds,
                        d.cu_deviceptr(),
                        src[i].cu_deviceptr(),
                        (bytes / 16) as u64,
                        PULL_BLOCKS,
                    )
                }
            };
            pull(&mut d)?;
            let pull_ok = d.to_host_vec(ds)? == want;
            d.zero_async(ds)?;
            let pull_ms = timed(ds, || {
                for _ in 0..reps {
                    pull(&mut d)?;
                }
                Ok(())
            })?;

            let base = chain[i].cu_deviceptr();
            // SAFETY: `chain[i]` holds CHASE_SLOTS * CHASE_STRIDE bytes on device i, `ends[j]`
            // one u64 on device j.
            let chase = || unsafe {
                kernels[j].chase(
                    ds,
                    base,
                    chain[i].num_bytes() as u64,
                    CHASE_HOPS,
                    ends[j].cu_deviceptr(),
                )
            };
            chase()?;
            let chase_ms = timed(ds, chase)?;
            let end = base + (CHASE_HOPS as usize % CHASE_SLOTS * CHASE_STRIDE) as u64;
            let chase_ok = ends[j].to_host_vec(ds)? == [end];

            let gbps = |ms: f32| (bytes as f64 * reps as f64) / (ms as f64 * 1e-3) / 1e9;
            let timings = Timings {
                copy_gbps: gbps(copy_ms),
                copy_latency_us: lat_ms as f64 * 1e3 / LATENCY_COPIES as f64,
                pull_gbps: gbps(pull_ms),
                load_latency_us: chase_ms as f64 * 1e3 / CHASE_HOPS as f64,
            };
            out.push(PairResult {
                src: a.ordinal(),
                dst: b.ordinal(),
                outcome: if copy_ok && pull_ok && chase_ok {
                    Outcome::Intact(timings)
                } else {
                    Outcome::Corrupt(timings)
                },
            });
        }
    }
    Ok(out)
}

/// Milliseconds between timing events recorded on `stream` around `work`.
fn timed(stream: &Stream, work: impl FnOnce() -> Result<(), DistError>) -> Result<f32, DistError> {
    let flags = Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let start = stream.record_event(flags)?;
    work()?;
    let end = stream.record_event(flags)?;
    Ok(start.elapsed_ms(&end)?)
}

const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// dst[i] = src[i] for n 16-byte vectors, grid-stride.
.visible .entry peer_copy(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u64 p_n
)
{
    .reg .pred %p;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;
    ld.param.u64 %rd1, [p_dst];
    ld.param.u64 %rd2, [p_src];
    ld.param.u64 %rd3, [p_n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd5, %r3;
    mad.wide.u32 %rd4, %r1, %r2, %rd5;
    mov.u32 %r4, %nctaid.x;
    mul.wide.u32 %rd6, %r4, %r2;
LOOP:
    setp.ge.u64 %p, %rd4, %rd3;
    @%p bra DONE;
    shl.b64 %rd7, %rd4, 4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd8];
    add.u64 %rd9, %rd1, %rd7;
    st.global.v4.u32 [%rd9], {%r5, %r6, %r7, %r8};
    add.u64 %rd4, %rd4, %rd6;
    bra LOOP;
DONE:
    ret;
}

// p = base; up to hops times: p = *p, stopping before a p outside [base, base + len) or not
// 8-byte aligned; *out = p.
.visible .entry peer_chase(
    .param .u64 p_base,
    .param .u64 p_len,
    .param .u32 p_hops,
    .param .u64 p_out
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<7>;
    ld.param.u64 %rd1, [p_base];
    ld.param.u64 %rd2, [p_len];
    ld.param.u32 %r1, [p_hops];
    ld.param.u64 %rd3, [p_out];
    mov.u64 %rd4, %rd1;
LOOP:
    setp.eq.u32 %p0, %r1, 0;
    @%p0 bra DONE;
    sub.u64 %rd5, %rd4, %rd1;
    setp.ge.u64 %p1, %rd5, %rd2;
    and.b64 %rd6, %rd5, 7;
    setp.ne.u64 %p2, %rd6, 0;
    or.pred %p1, %p1, %p2;
    @%p1 bra DONE;
    ld.volatile.global.u64 %rd4, [%rd4];
    sub.u32 %r1, %r1, 1;
    bra LOOP;
DONE:
    st.global.u64 [%rd3], %rd4;
    ret;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_times_every_ordered_pair_and_checks_the_data() {
        let ctxs: Vec<_> = (0..4).map(|o| Context::new(o).unwrap()).collect();
        let t = bench(&ctxs, 1 << 20, 2).unwrap();
        assert_eq!(t.len(), 12);
        for p in &t {
            let Outcome::Intact(m) = p.outcome else {
                panic!("pair {} -> {}: {:?}", p.src, p.dst, p.outcome);
            };
            assert!(
                m.copy_gbps > 0.0 && m.pull_gbps > 0.0 && m.load_latency_us > 0.0,
                "{p:?}"
            );
        }
    }
}
