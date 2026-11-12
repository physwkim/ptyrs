//! The frames a live graph replay works on (DESIGN.md §3 B2). The kernels
//! read one [`StackTable`] entry when they run; the host only lands frames,
//! through [`LiveFrames::land`], and replays the graph, which starts with
//! [`LiveFrames::step`] moving the entry onto the next landed frames.
//!
//! `land` is the one write of the landed count and refuses a count past the
//! scan or below the last one, so the count is monotone and at most
//! `nframe`. `step` keeps `f0 + nf <= landed` from `{0, 0}` on, so every
//! entry it writes is in range and the kernels' own clamp to `nframe` never
//! drops a landed frame.

use std::ffi::c_void;
use std::num::NonZeroU32;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_bindings::{self as sys, CUdeviceptr};
use cuda_core::{
    DeviceBuffer, Function, IntoResult, RawDeviceBuffer, Stream, launch_kernel_on_stream,
};
use ptyrs_gpu::{StackRef, StackTable, ViewDims};

use crate::grow::GrowError;

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

/// The landed count and the stack entry of one live scan, on one stream:
/// `land` and graphs that `step` is recorded into must run there.
pub struct LiveFrames {
    table: StackTable,
    landed: DeviceBuffer<u32>,
    func: Function,
    count: u32,
    nframe: u32,
    stream: Arc<Stream>,
}

impl LiveFrames {
    /// No frame landed, the entry `{0, 0}`; stacks of at most `nf_max`
    /// frames, `nf_max` in `1..=dims.nframe`.
    pub fn new(
        stream: &Arc<Stream>,
        dims: &ViewDims,
        nf_max: NonZeroU32,
    ) -> Result<Self, GrowError> {
        let ctx = stream.context();
        let table = StackTable::device(stream, dims, nf_max.get())?;
        let landed = DeviceBuffer::zeroed(stream, 1)?;
        let func = ctx
            .load_module_from_ptx_src(PTX)?
            .load_function("next_stack")?;
        Ok(Self {
            table,
            landed,
            func,
            count: 0,
            nframe: dims.nframe,
            stream: stream.clone(),
        })
    }

    /// The entry the stack kernels take.
    pub fn stack(&self) -> StackRef<'_> {
        self.table.at(0)
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

    /// The entry's [`StackCtl`](ptyrs_gpu::StackCtl), for kernels of the
    /// caller's own that read it; they lease it through
    /// [`Self::ctl_storage`].
    pub fn ctl_ptr(&self) -> CUdeviceptr {
        self.table.ptr()
    }

    pub fn ctl_storage(&self) -> &RawDeviceBuffer {
        self.table.storage()
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
        ptyrs_gpu::lease(&ctx, &[], &[self.landed.storage()])?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: `landed` holds one u32, leased for writing on this stream.
        unsafe { sys::cuMemsetD32Async(self.landed.cu_deviceptr(), n, 1, self.stream.cu_stream()) }
            .result()?;
        self.count = n;
        Ok(())
    }

    /// Queues on `ctx`'s stream, or records into its capture, the step of
    /// the entry onto the next at most `nf_max` landed frames; `nf` is 0
    /// when none is new.
    pub fn step(&self, ctx: &ExecutionContext) -> Result<(), ptyrs_gpu::Error> {
        if ctx.context().as_ref() != self.stream.context().as_ref() {
            return Err(ptyrs_gpu::Error::ContextMismatch);
        }
        ptyrs_gpu::lease(ctx, &[self.landed.storage()], &[self.table.storage()])?;
        let (mut ctl, mut landed, mut nf_max) = (
            self.table.ptr(),
            self.landed.cu_deviceptr(),
            self.table.nf_max(),
        );
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
    use ptyrs_gpu::StackCtl;

    use super::*;

    fn read(f: &LiveFrames) -> StackCtl {
        let mut got = StackCtl { f0: 0, nf: 0 };
        f.stream.synchronize().unwrap();
        // SAFETY: the stream is idle and the table holds one StackCtl.
        unsafe { sys::cuMemcpyDtoH_v2((&raw mut got).cast(), f.table.ptr(), 8) }
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
}
