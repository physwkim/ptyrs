//! Stand-in for the engine's live graph (DESIGN.md §3 B2, §5 invariant 9):
//! one CUDA graph, captured before the first frame, consumes the device
//! store batch by batch while it grows. Which frames a replay covers is
//! read on the GPU from a counter the graph advances itself; the host only
//! writes how many frames have landed and launches the same graph again.
//!
//! The per-frame work is a position-weighted checksum, so an end-of-run
//! check covers every stored frame.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

use crossbeam_channel::Receiver;
use cuda_bindings as sys;
use cuda_bindings::CUdeviceptr;
use cuda_core::{Context, DriverError, IntoResult, Stream, launch_kernel_on_stream};

use crate::grow::{GrowBuf, GrowError};
use crate::ring::{DeviceStore, FrameBatch};

const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// ctl = [done, avail]. Block b sums frame done + b if it is below
// min(done + batch, avail): out[f] = sum_j frame[j] * (j + 1).
.visible .entry checksum(.param .u64 frames, .param .u32 elems, .param .u64 ctl,
                         .param .u32 batch, .param .u64 out)
{
    .reg .pred p, q;
    .reg .b32 r<12>;
    .reg .b64 rd<12>;
    .shared .align 8 .u64 acc;
    ld.param.u64 rd1, [frames];
    ld.param.u32 r1, [elems];
    ld.param.u64 rd2, [ctl];
    ld.param.u32 r2, [batch];
    ld.param.u64 rd3, [out];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    cvta.to.global.u64 rd3, rd3;
    ld.global.u32 r3, [rd2];
    ld.global.u32 r4, [rd2+4];
    add.u32 r5, r3, r2;
    min.u32 r5, r5, r4;
    mov.u32 r6, %ctaid.x;
    add.u32 r6, r6, r3;
    setp.ge.u32 p, r6, r5;
    @p bra DONE;
    mov.u32 r7, %tid.x;
    setp.eq.u32 q, r7, 0;
    mov.u64 rd4, 0;
    @q st.shared.u64 [acc], rd4;
    bar.sync 0;
    mul.wide.u32 rd5, r6, r1;
    shl.b64 rd5, rd5, 1;
    add.s64 rd5, rd1, rd5;
    mov.u32 r8, r7;
    mov.u32 r9, %ntid.x;
LOOP:
    setp.ge.u32 p, r8, r1;
    @p bra REDUCE;
    mul.wide.u32 rd6, r8, 2;
    add.s64 rd6, rd5, rd6;
    ld.global.u16 r10, [rd6];
    add.u32 r11, r8, 1;
    mul.wide.u32 rd7, r10, r11;
    add.u64 rd4, rd4, rd7;
    add.u32 r8, r8, r9;
    bra LOOP;
REDUCE:
    atom.shared.add.u64 rd8, [acc], rd4;
    bar.sync 0;
    @!q bra DONE;
    ld.shared.u64 rd9, [acc];
    mul.wide.u32 rd10, r6, 8;
    add.s64 rd10, rd3, rd10;
    st.global.u64 [rd10], rd9;
DONE:
    ret;
}

// done = min(done + batch, avail)
.visible .entry advance(.param .u64 ctl, .param .u32 batch)
{
    .reg .b32 r<5>;
    .reg .b64 rd<2>;
    ld.param.u64 rd1, [ctl];
    ld.param.u32 r1, [batch];
    cvta.to.global.u64 rd1, rd1;
    ld.global.u32 r2, [rd1];
    ld.global.u32 r3, [rd1+4];
    add.u32 r2, r2, r1;
    min.u32 r2, r2, r3;
    st.global.u32 [rd1], r2;
    ret;
}
"#;

/// Host twin of the `checksum` kernel.
pub fn checksum(frame: &[u16]) -> u64 {
    frame
        .iter()
        .zip(1u64..)
        .fold(0u64, |a, (&v, j)| a.wrapping_add(u64::from(v) * j))
}

/// A captured, instantiated CUDA graph.
pub struct Graph {
    exec: sys::CUgraphExec,
    graph: sys::CUgraph,
    ctx: Arc<Context>,
}

impl Graph {
    /// Records into a graph what `record` issues on `stream`; the capture
    /// is ended on every return path.
    pub fn capture(
        stream: &Stream,
        record: impl FnOnce() -> Result<(), DriverError>,
    ) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        ctx.bind_to_thread()?;
        // SAFETY: the capture is ended below before `stream` is used again.
        unsafe {
            stream
                .begin_capture(sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?
        };
        let recorded = record();
        // SAFETY: the stream is capturing (begun above).
        let graph = unsafe { stream.end_capture() }?;
        let mut exec = MaybeUninit::uninit();
        // SAFETY: `graph` is the valid graph just captured.
        let inst = recorded.and_then(|()| {
            unsafe { sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), graph, 0) }.result()
        });
        if let Err(e) = inst {
            // SAFETY: `graph` is owned here and not used afterwards.
            let _ = unsafe { sys::cuGraphDestroy(graph) };
            return Err(e);
        }
        Ok(Self {
            // SAFETY: instantiation succeeded.
            exec: unsafe { exec.assume_init() },
            graph,
            ctx,
        })
    }

    pub fn launch(&self, stream: &Stream) -> Result<(), DriverError> {
        // SAFETY: `exec` lives as long as `self`.
        unsafe { sys::cuGraphLaunch(self.exec, stream.cu_stream()) }.result()
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: both handles are owned by `self` and destroyed once.
        unsafe {
            let _ = sys::cuGraphExecDestroy(self.exec);
            let _ = sys::cuGraphDestroy(self.graph);
        }
    }
}

pub fn memset32(dst: CUdeviceptr, v: u32, n: usize, s: &Stream) -> Result<(), DriverError> {
    // SAFETY: callers pass a mapped range of at least `n` u32.
    unsafe { sys::cuMemsetD32Async(dst, v, n, s.cu_stream()) }.result()
}

pub struct ConsumerReport {
    /// Replays of the one captured graph.
    pub launches: u64,
    /// Frames the graph consumed, read back from its device counter.
    pub consumed: u64,
    /// Checksum of every consumed frame, by sequence number.
    pub checksums: Vec<u64>,
    /// Device memory behind `checksums` when the consumer stopped.
    pub out_mapped_bytes: usize,
}

/// Consume `batches` until the uploader drops its sender, `batch` frames per
/// replay at most.
pub fn run_checksum_consumer(
    ctx: Arc<Context>,
    store: Arc<DeviceStore>,
    batch: u32,
    batches: Receiver<FrameBatch>,
) -> Result<ConsumerReport, GrowError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let module = ctx.load_module_from_ptx_src(PTX)?;
    let (sum, advance) = (
        module.load_function("checksum")?,
        module.load_function("advance")?,
    );
    let mut ctl = GrowBuf::new(&ctx, 8, 1)?;
    ctl.map(0..8)?;
    memset32(ctl.ptr(), 0, 2, &stream)?;
    let mut out = GrowBuf::new(&ctx, store.max_frames() as usize * 8, 1)?;
    let (mut p_frames, mut p_elems, mut p_ctl, mut p_batch, mut p_out) = (
        store.frames_ptr(),
        store.frame_elems() as u32,
        ctl.ptr(),
        batch,
        out.ptr(),
    );
    let mut sum_params = [
        (&raw mut p_frames).cast::<c_void>(),
        (&raw mut p_elems).cast(),
        (&raw mut p_ctl).cast(),
        (&raw mut p_batch).cast(),
        (&raw mut p_out).cast(),
    ];
    let (mut a_ctl, mut a_batch) = (ctl.ptr(), batch);
    let mut adv_params = [(&raw mut a_ctl).cast::<c_void>(), (&raw mut a_batch).cast()];
    let graph = Graph::capture(&stream, || {
        // SAFETY: the parameter arrays match the kernels' signatures and
        // point at locals that outlive the capture.
        unsafe {
            launch_kernel_on_stream(
                &sum,
                (batch, 1, 1),
                (256, 1, 1),
                0,
                &stream,
                &mut sum_params,
            )?;
            launch_kernel_on_stream(&advance, (1, 1, 1), (1, 1, 1), 0, &stream, &mut adv_params)
        }
    })?;

    let (mut landed, mut covered, mut launches) = (0u64, 0u64, 0u64);
    let fed = (|| -> Result<(), GrowError> {
        for b in batches.iter() {
            assert_eq!(b.seq0, landed, "batches arrive in order and without gaps");
            landed += b.n;
            out.map(0..landed as usize * 8)?;
            memset32(ctl.ptr() + 4, landed as u32, 1, &stream)?;
            while covered < landed {
                graph.launch(&stream)?;
                launches += 1;
                covered = (covered + u64::from(batch)).min(landed);
            }
        }
        Ok(())
    })();
    // Nothing may still read the store or write `out` once this returns.
    stream.synchronize()?;
    fed?;

    let mut done = 0u32;
    let mut checksums = vec![0u64; landed as usize];
    // SAFETY: the stream is idle; both sources are mapped and hold at least
    // the bytes copied.
    unsafe {
        sys::cuMemcpyDtoH_v2((&raw mut done).cast(), ctl.ptr(), 4).result()?;
        sys::cuMemcpyDtoH_v2(
            checksums.as_mut_ptr().cast(),
            out.ptr(),
            checksums.len() * 8,
        )
        .result()?;
    }
    Ok(ConsumerReport {
        launches,
        consumed: u64::from(done),
        checksums,
        out_mapped_bytes: out.mapped_bytes(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::ring::{Ring, UploaderConfig, run_uploader};

    /// The input path feeds a graph captured before the first frame: the
    /// store grows 2 MiB at a time under it and every frame is consumed
    /// exactly once, the last replay covering a partial batch.
    #[test]
    fn one_graph_consumes_every_frame_while_the_store_grows() {
        let ctx = Context::new(0).expect("GPU 0");
        let (se, total, batch) = (65536, 203u64, 7u32);
        let ring = Arc::new(Ring::new(&ctx, 16, se).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, total as usize, se, 1).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let consumer = {
            let (ctx, store) = (ctx.clone(), store.clone());
            std::thread::spawn(move || run_checksum_consumer(ctx, store, batch, rx))
        };
        let uploader = {
            let (ring, ctx, stop, store) = (ring.clone(), ctx.clone(), stop.clone(), store.clone());
            let cfg = UploaderConfig {
                batch: 5,
                flush_ns: 100_000,
                idle: Duration::from_micros(20),
                events: 4,
            };
            std::thread::spawn(move || run_uploader(ring, ctx, store, cfg, stop, Some(tx)))
        };
        let frame = |seq: u64| -> Vec<u16> {
            (0..se)
                .map(|j| ((seq * 131 + j as u64 * 7) % 65521) as u16)
                .collect()
        };
        for _ in 0..total {
            let mut s = ring.reserve();
            let seq = s.seq();
            s.frame().copy_from_slice(&frame(seq));
            s.commit(seq as i32, 0.0, 0.0, 0, 0);
            std::thread::sleep(Duration::from_micros(200));
        }
        stop.store(true, Ordering::Release);
        let up = uploader.join().unwrap().unwrap();
        let rep = consumer.join().unwrap().unwrap();

        assert_eq!(up.frames, total);
        assert_eq!(rep.consumed, total);
        assert!(rep.launches >= total.div_ceil(u64::from(batch)));
        let bad = (0..total)
            .filter(|&s| rep.checksums[s as usize] != checksum(&frame(s)))
            .count();
        assert_eq!(bad, 0, "frames with a wrong checksum");
    }
}
