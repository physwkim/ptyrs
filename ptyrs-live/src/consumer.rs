//! Stand-in for the engine's live graph (DESIGN.md §3 B2, §5 invariant 9):
//! one CUDA graph, captured before the first frame, consumes the device
//! store batch by batch as frames land in it. Each replay starts with
//! [`LiveFrames::step`], so which frames it covers is read on the GPU from
//! the stack entry the graph advances itself; the host only lands the frames
//! that arrived and launches the same graph again.
//!
//! The per-frame work is a position-weighted checksum, so an end-of-run
//! check covers every stored frame.

use std::ffi::c_void;
use std::num::NonZeroU32;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use cuda_bindings as sys;
use cuda_core::{Context, IntoResult, launch_kernel_on_stream};
use ptyrs_gpu::StackCtl;

use crate::graph::LiveGraph;
use crate::grow::{GrowBuf, GrowError};
use crate::ring::{DeviceStore, FrameBatch};
use crate::stack::{LiveFrames, frames_only};

const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// ctl = StackCtl {f0, nf}. Block b < nf sums frame f0 + b:
// out[f] = sum_j frame[j] * (j + 1).
.visible .entry checksum(.param .u64 frames, .param .u32 elems, .param .u64 ctl,
                         .param .u64 out)
{
    .reg .pred p, q;
    .reg .b32 r<12>;
    .reg .b64 rd<12>;
    .shared .align 8 .u64 acc;
    ld.param.u64 rd1, [frames];
    ld.param.u32 r1, [elems];
    ld.param.u64 rd2, [ctl];
    ld.param.u64 rd3, [out];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    cvta.to.global.u64 rd3, rd3;
    ld.global.u32 r3, [rd2];
    ld.global.u32 r4, [rd2+4];
    mov.u32 r6, %ctaid.x;
    setp.ge.u32 p, r6, r4;
    @p bra DONE;
    add.u32 r6, r6, r3;
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
"#;

/// Host twin of the `checksum` kernel.
pub fn checksum(frame: &[u16]) -> u64 {
    frame
        .iter()
        .zip(1u64..)
        .fold(0u64, |a, (&v, j)| a.wrapping_add(u64::from(v) * j))
}
pub struct ConsumerReport {
    /// Replays of the one captured graph.
    pub launches: u64,
    /// Frames the graph consumed, read back from its stack entry.
    pub consumed: u64,
    /// Checksum of every consumed frame, by sequence number.
    pub checksums: Vec<u64>,
    /// Device memory behind `checksums` when the consumer stopped.
    pub out_mapped_bytes: usize,
}

/// Consume `batches` until the uploader drops its sender, `batch` frames per
/// replay at most. Sends on `captured` once its one graph is captured, and
/// drops it unsent if the capture fails.
pub fn run_checksum_consumer(
    ctx: Arc<Context>,
    store: Arc<DeviceStore>,
    batch: NonZeroU32,
    batches: Receiver<FrameBatch>,
    captured: Sender<()>,
) -> Result<ConsumerReport, GrowError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let sum = ctx
        .load_module_from_ptx_src(PTX)?
        .load_function("checksum")?;
    let nframe = u32::try_from(store.max_frames())
        .map_err(|_| ptyrs_gpu::Error::Shape(format!("{} frames pass u32", store.max_frames())))?;
    let nf_max = NonZeroU32::new(batch.get().min(nframe)).unwrap_or(batch);
    let mut frames = LiveFrames::new(&stream, &frames_only(nframe), nf_max)?;
    let mut out = GrowBuf::new(&ctx, nframe as usize * 8, 1)?;
    // Whole, now: the loop below must not wait for the driver to grow it.
    out.map(0..nframe as usize * 8)?;
    // SAFETY: a replay reads only landed frames, which the uploader sent,
    // and writes their checksums, which `out` backs whole.
    let (counts, sums) = unsafe { (store.frames_view()?, out.typed::<u64>(nframe as usize)?) };
    let elems = store.frame_elems() as u32;
    let graph = LiveGraph::capture(&stream, |ctx| {
        frames.step(ctx)?;
        let st = frames.stack();
        ptyrs_gpu::lease(ctx, &[counts.storage(), st.storage()], &[sums.storage()])?;
        let (mut p_frames, mut p_elems, mut p_ctl, mut p_out) =
            (counts.cu_deviceptr(), elems, st.ctl(), sums.cu_deviceptr());
        let mut params = [
            (&raw mut p_frames).cast::<c_void>(),
            (&raw mut p_elems).cast(),
            (&raw mut p_ctl).cast(),
            (&raw mut p_out).cast(),
        ];
        let s = ctx.get_cuda_stream();
        // SAFETY: the parameters match `checksum` and are copied at the
        // launch; every buffer is leased above.
        unsafe {
            launch_kernel_on_stream(&sum, (nf_max.get(), 1, 1), (256, 1, 1), 0, s, &mut params)
        }?;
        Ok(())
    })?;
    // The waiter may have given up; the run goes on either way.
    let _ = captured.send(());

    let (mut landed, mut covered, mut launches) = (0u64, 0u64, 0u64);
    let fed = (|| -> Result<(), GrowError> {
        for b in batches.iter() {
            assert_eq!(b.f0, landed, "batches arrive in order and without gaps");
            landed += b.n;
            frames.land(u32::try_from(landed).unwrap_or(u32::MAX))?;
            while covered < landed {
                graph.replay()?;
                launches += 1;
                covered = (covered + u64::from(nf_max.get())).min(landed);
            }
        }
        Ok(())
    })();
    // Nothing may still read the store or write `out` once this returns.
    stream.synchronize()?;
    fed?;

    let mut done = StackCtl { f0: 0, nf: 0 };
    let mut checksums = vec![0u64; landed as usize];
    // SAFETY: the stream is idle; both sources are mapped and hold at least
    // the bytes copied.
    unsafe {
        sys::cuMemcpyDtoH_v2((&raw mut done).cast(), frames.stack().ctl(), 8).result()?;
        sys::cuMemcpyDtoH_v2(
            checksums.as_mut_ptr().cast(),
            out.ptr(),
            checksums.len() * 8,
        )
        .result()?;
    }
    Ok(ConsumerReport {
        launches,
        consumed: u64::from(done.f0 + done.nf),
        checksums,
        out_mapped_bytes: out.mapped_bytes(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::graph::WarmUp;
    use crate::pos::{Pos, PosTable};
    use crate::ring::{Ring, UploaderConfig, run_uploader};
    use std::num::NonZeroUsize;

    /// The input path feeds a graph captured before the first frame: frames
    /// fill the store under it and every frame is consumed exactly once, the
    /// last replay covering a partial batch.
    #[test]
    fn one_graph_consumes_every_frame_as_the_store_fills() {
        let ctx = Context::new(0).expect("GPU 0");
        let (se, total, batch) = (65536, 203u64, NonZeroU32::new(7).unwrap());
        let ring = Arc::new(Ring::new(&ctx, 16, se).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, total as usize, se).unwrap());
        let pos = Arc::new(PosTable::new(256));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let (cap_tx, cap_rx) = crossbeam_channel::bounded(1);
        let consumer = {
            let (ctx, store) = (ctx.clone(), store.clone());
            std::thread::spawn(move || run_checksum_consumer(ctx, store, batch, rx, cap_tx))
        };
        cap_rx.recv().expect("the graph is captured");
        let warm = WarmUp::end();
        let uploader = {
            let (ring, ctx, stop, store) = (ring.clone(), ctx.clone(), stop.clone(), store.clone());
            let pos = pos.clone();
            let cfg = UploaderConfig {
                batch: NonZeroUsize::new(5).unwrap(),
                flush_ns: 100_000,
                idle: Duration::from_micros(20),
                events: 4,
                pos_wait_ns: 1_000_000_000,
            };
            std::thread::spawn(move || run_uploader(ring, ctx, store, pos, cfg, stop, Some(tx)))
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
            pos.insert(
                seq as i32,
                Pos {
                    x: 0.0,
                    y: 0.0,
                    t_rx: 0,
                },
            );
            s.commit(seq as i32, 0, 0);
            std::thread::sleep(Duration::from_micros(200));
        }
        stop.store(true, Ordering::Release);
        let up = uploader.join().unwrap().unwrap();
        let rep = consumer.join().unwrap().unwrap();

        assert_eq!(warm.recaptures(), 0);
        assert_eq!(up.frames, total);
        assert_eq!(rep.consumed, total);
        assert!(rep.launches >= total.div_ceil(u64::from(batch.get())));
        let bad = (0..total)
            .filter(|&s| rep.checksums[s as usize] != checksum(&frame(s)))
            .count();
        assert_eq!(bad, 0, "frames with a wrong checksum");
    }
}
