//! Peer reads checked by a kernel.
//!
//! The driver can report peer access as supported and enabled while the platform corrupts the
//! traffic (an IOMMU without the peer mapping returns all-ones loads). The halo pull and the
//! gather read peer memory with SM loads, so [`verify_reads`] checks that path before peer
//! access counts as enabled.

use std::ffi::c_void;
use std::sync::Arc;

use cuda_core::{Context, DeviceBuffer, Function, Stream, launch_kernel_on_stream};

use crate::DistError;

struct Kernels {
    copy: Function,
}

const COPY_THREADS: u32 = 256;

impl Kernels {
    fn load(ctx: &Arc<Context>) -> Result<Self, DistError> {
        let m = ctx.load_module_from_ptx_src(PTX)?;
        Ok(Self {
            copy: m.load_function("peer_copy")?,
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
"#;
