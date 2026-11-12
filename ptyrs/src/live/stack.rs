//! The frames a live graph replay works on (DESIGN.md §3 B2). The kernels
//! read a [`StackTable`] entry when they run; the host only lands frames,
//! through [`LiveFrames::land`], and replays graphs that move the entries.
//! Two entries walk the landed frames: the ingest entry ([`LiveFrames::step`])
//! takes each landed frame once, for binning; the sweep entry
//! ([`LiveFrames::sweep_step`]) walks them from the first on after each
//! [`LiveFrames::rewind`], one sweep per iteration.
//!
//! `land` is the one write of the landed count and refuses a count past the
//! scan or below the last one, so the count is monotone and at most
//! `nframe`. Both steps keep `f0 + nf <= landed` from `{0, 0}` on, and so
//! does a rewind, so every entry they write is in range and the kernels' own
//! clamp to `nframe` never drops a landed frame.

use std::ffi::c_void;
use std::num::NonZeroU32;
use std::sync::Arc;

use crate::gpu::{StackRef, StackTable, ViewDims};
use cuda_async::device_operation::ExecutionContext;
use cuda_bindings as sys;
use cuda_core::{DeviceBuffer, Function, IntoResult, Stream, launch_kernel_on_stream};

use crate::live::grow::GrowError;

const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// ctl = {f0, nf}: f0 += nf, then nf = min(landed - f0, nf_max).
.visible .entry next_stack(.param .u64 ctl, .param .u64 landed, .param .u32 nf_max)
{
    .reg .b32 r<6>;
    .reg .b64 rd<3>;
    ld.param.u64 rd1, [ctl];
    ld.param.u64 rd2, [landed];
    ld.param.u32 r1, [nf_max];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    ld.global.u32 r2, [rd1];
    ld.global.u32 r3, [rd1+4];
    add.u32 r2, r2, r3;
    ld.global.u32 r4, [rd2];
    sub.u32 r5, r4, r2;
    min.u32 r5, r5, r1;
    st.global.u32 [rd1], r2;
    st.global.u32 [rd1+4], r5;
    ret;
}
"#;

/// The landed count and the stack entries of one live scan, on one stream:
/// `land` and graphs the steps are recorded into must run there.
pub struct LiveFrames {
    table: StackTable,
    sweep: StackTable,
    landed: DeviceBuffer<u32>,
    func: Function,
    count: u32,
    nframe: u32,
    stream: Arc<Stream>,
}

impl LiveFrames {
    /// No frame landed, both entries `{0, 0}`; stacks of at most `nf_max`
    /// frames, `nf_max` in `1..=dims.nframe`.
    pub fn new(
        stream: &Arc<Stream>,
        dims: &ViewDims,
        nf_max: NonZeroU32,
    ) -> Result<Self, GrowError> {
        let ctx = stream.context();
        let table = StackTable::device(stream, dims, nf_max.get())?;
        let sweep = StackTable::device(stream, dims, nf_max.get())?;
        let landed = DeviceBuffer::zeroed(stream, 1)?;
        let func = ctx
            .load_module_from_ptx_src(PTX)?
            .load_function("next_stack")?;
        Ok(Self {
            table,
            sweep,
            landed,
            func,
            count: 0,
            nframe: dims.nframe,
            stream: stream.clone(),
        })
    }

    /// The ingest entry.
    pub fn stack(&self) -> StackRef<'_> {
        self.table.at(0)
    }

    /// The sweep entry.
    pub fn sweep(&self) -> StackRef<'_> {
        self.sweep.at(0)
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Frames landed so far.
    pub fn landed(&self) -> u32 {
        self.count
    }

    pub fn nframe(&self) -> u32 {
        self.nframe
    }

    pub fn nf_max(&self) -> u32 {
        self.table.nf_max()
    }

    /// Lands frames `landed()..n`: steps queued afterwards cover them. The
    /// only write of the landed count; `n` below it or past `nframe` is
    /// refused and changes nothing.
    pub fn land(&mut self, n: u32) -> Result<(), GrowError> {
        if n < self.count || n > self.nframe {
            return Err(GrowError::Land {
                n,
                landed: self.count,
                nframe: self.nframe,
            });
        }
        if n == self.count {
            return Ok(());
        }
        let ctx = ExecutionContext::new(self.stream.clone());
        crate::gpu::lease(&ctx, &[], &[self.landed.storage()])?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: `landed` holds one u32, leased for writing on this stream.
        unsafe { sys::cuMemsetD32Async(self.landed.cu_deviceptr(), n, 1, self.stream.cu_stream()) }
            .result()?;
        self.count = n;
        Ok(())
    }

    /// Queues on `ctx`'s stream, or records into its capture, the step of
    /// the ingest entry onto the next at most `nf_max` landed frames; `nf` is
    /// 0 when none is new.
    pub fn step(&self, ctx: &ExecutionContext) -> Result<(), crate::gpu::Error> {
        self.next(ctx, &self.table)
    }

    /// [`Self::step`] of the sweep entry.
    pub fn sweep_step(&self, ctx: &ExecutionContext) -> Result<(), crate::gpu::Error> {
        self.next(ctx, &self.sweep)
    }

    /// Queues or records the sweep entry's return to `{0, 0}`: the next
    /// `sweep_step` starts a sweep at the first frame.
    pub fn rewind(&self, ctx: &ExecutionContext) -> Result<(), crate::gpu::Error> {
        self.same_context(ctx)?;
        crate::gpu::lease(ctx, &[], &[self.sweep.storage()])?;
        ctx.context().bind_to_thread()?;
        // SAFETY: the entry is two u32 of the sweep table, leased for writing
        // on this stream.
        unsafe { sys::cuMemsetD32Async(self.sweep.ptr(), 0, 2, ctx.get_cuda_stream().cu_stream()) }
            .result()?;
        Ok(())
    }

    fn same_context(&self, ctx: &ExecutionContext) -> Result<(), crate::gpu::Error> {
        if ctx.context().as_ref() != self.stream.context().as_ref() {
            return Err(crate::gpu::Error::ContextMismatch);
        }
        Ok(())
    }

    fn next(&self, ctx: &ExecutionContext, table: &StackTable) -> Result<(), crate::gpu::Error> {
        self.same_context(ctx)?;
        crate::gpu::lease(ctx, &[self.landed.storage()], &[table.storage()])?;
        let (mut ctl, mut landed, mut nf_max) =
            (table.ptr(), self.landed.cu_deviceptr(), table.nf_max());
        let mut params = [
            (&raw mut ctl).cast::<c_void>(),
            (&raw mut landed).cast(),
            (&raw mut nf_max).cast(),
        ];
        let s = ctx.get_cuda_stream();
        // SAFETY: the parameters match `next_stack`'s signature and are copied
        // at the launch; both buffers are leased above.
        unsafe { launch_kernel_on_stream(&self.func, (1, 1, 1), (1, 1, 1), 0, s, &mut params) }?;
        Ok(())
    }
}

/// [`ViewDims`] for a stack table only: `nframe` frames, every other field
/// the smallest [`ViewDims`] accepts. For consumers that run no view kernel.
pub fn frames_only(nframe: u32) -> ViewDims {
    ViewDims {
        nframe,
        nmode: 1,
        nprobe: 1,
        nobj: 1,
        frame: [2, 2],
        obj: [2, 2],
        quad_phase: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::StackCtl;

    use super::*;

    fn read(f: &LiveFrames) -> StackCtl {
        read_entry(f, f.stack())
    }

    fn read_entry(f: &LiveFrames, e: StackRef<'_>) -> StackCtl {
        let mut got = StackCtl { f0: 0, nf: 0 };
        f.stream.synchronize().unwrap();
        // SAFETY: the stream is idle and the entry is one StackCtl.
        unsafe { sys::cuMemcpyDtoH_v2((&raw mut got).cast(), e.ctl(), 8) }
            .result()
            .unwrap();
        got
    }

    /// From `{0, 0}`, each step under lands that stall and grow by less and
    /// more than `nf_max`, up to the whole scan, gives the host model's
    /// range, and none reaches past the landed count or holds more than
    /// `nf_max`. Lands below the count or past the scan are refused.
    #[test]
    fn each_step_covers_the_next_landed_frames_within_the_limits() {
        let ctx = cuda_core::Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (nf_max, nframe) = (4u32, 20u32);
        let mut f =
            LiveFrames::new(&s, &frames_only(nframe), NonZeroU32::new(nf_max).unwrap()).unwrap();
        let mut want = StackCtl { f0: 0, nf: 0 };
        let lands = [0u32, 3, 3, 10, 10, 10, 19, 20, 20, 20, 20, 20];
        for (i, &l) in lands.iter().enumerate() {
            f.land(l).unwrap();
            f.step(&ExecutionContext::new(s.clone())).unwrap();
            let got = read(&f);
            want.f0 += want.nf;
            want.nf = (l - want.f0).min(nf_max);
            assert_eq!(got, want, "step {i}, landed {l}");
            assert!(got.f0 + got.nf <= l && got.nf <= nf_max);
        }
        assert_eq!(want, StackCtl { f0: 20, nf: 0 });
        for bad in [19, nframe + 1] {
            assert!(matches!(
                f.land(bad),
                Err(GrowError::Land { n, landed: 20, nframe: 20 }) if n == bad
            ));
        }
        assert_eq!(f.landed(), 20);
        f.step(&ExecutionContext::new(s.clone())).unwrap();
        assert_eq!(read(&f), StackCtl { f0: 20, nf: 0 });
    }

    /// Sweeps of `ceil(landed / nf_max)` steps, each after a rewind, cover
    /// the landed frames `0..landed` in order, with lands between sweeps that
    /// grow by less and more than `nf_max`; a step past the sweep's end gives
    /// `nf == 0`, and the ingest entry is left alone.
    #[test]
    fn each_sweep_after_a_rewind_covers_every_landed_frame_once() {
        let ctx = cuda_core::Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (nf_max, nframe) = (4u32, 20u32);
        let f = &mut LiveFrames::new(&s, &frames_only(nframe), NonZeroU32::new(nf_max).unwrap())
            .unwrap();
        let c = &ExecutionContext::new(s.clone());
        f.land(3).unwrap();
        f.step(c).unwrap();
        for l in [3u32, 3, 9, 13, 20, 20] {
            f.land(l).unwrap();
            f.rewind(c).unwrap();
            assert_eq!(read_entry(f, f.sweep()), StackCtl { f0: 0, nf: 0 });
            let mut next = 0;
            for _ in 0..l.div_ceil(nf_max) {
                f.sweep_step(c).unwrap();
                let got = read_entry(f, f.sweep());
                assert_eq!(got.f0, next, "landed {l}");
                assert_eq!(got.nf, (l - next).min(nf_max), "landed {l}");
                next += got.nf;
            }
            assert_eq!(next, l);
            f.sweep_step(c).unwrap();
            assert_eq!(read_entry(f, f.sweep()), StackCtl { f0: l, nf: 0 });
        }
        assert_eq!(read(f), StackCtl { f0: 0, nf: 3 });
    }
}
