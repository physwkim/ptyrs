//! The frame range a live graph replay works on, advanced on the device
//! (DESIGN.md §3 B2). The kernels read it through `*const StackCtl`; the
//! host only writes how many frames have landed and replays the graph, which
//! starts with [`NextStack`]. The kernels do not clamp a range written on the
//! device, so `next_stack` is what holds `nf <= nf_max` and
//! `f0 + nf <= nframe`.

use std::ffi::c_void;
use std::num::NonZeroU32;
use std::sync::Arc;

use cuda_bindings::CUdeviceptr;
use cuda_core::{Context, DriverError, Function, Stream, launch_kernel_on_stream};

/// Frames `f0..f0 + nf`; the layout of `ptyrs_kernels::pixel::StackCtl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StackCtl {
    pub f0: u32,
    pub nf: u32,
}

const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// ctl = {f0, nf}: f0 += nf, then nf = min(max(min(landed, nframe), f0) - f0, nf_max).
.visible .entry next_stack(.param .u64 ctl, .param .u64 landed, .param .u32 nf_max,
                           .param .u32 nframe)
{
    .reg .b32 r<7>;
    .reg .b64 rd<3>;
    ld.param.u64 rd1, [ctl];
    ld.param.u64 rd2, [landed];
    ld.param.u32 r1, [nf_max];
    ld.param.u32 r2, [nframe];
    cvta.to.global.u64 rd1, rd1;
    cvta.to.global.u64 rd2, rd2;
    ld.global.u32 r3, [rd1];
    ld.global.u32 r4, [rd1+4];
    add.u32 r3, r3, r4;
    ld.global.u32 r5, [rd2];
    min.u32 r5, r5, r2;
    max.u32 r5, r5, r3;
    sub.u32 r6, r5, r3;
    min.u32 r6, r6, r1;
    st.global.u32 [rd1], r3;
    st.global.u32 [rd1+4], r6;
    ret;
}
"#;

/// Moves a [`StackCtl`] past the frames it covered and onto the next at
/// most `nf_max` of the first `min(landed, nframe)`; `nf` is 0 when none
/// is new. Starting from `{0, 0}`, every value it writes keeps
/// `f0 + nf <= nframe` whatever `landed` holds.
pub struct NextStack {
    func: Function,
    nf_max: u32,
    nframe: u32,
}

impl NextStack {
    pub fn load(ctx: &Arc<Context>, nf_max: NonZeroU32, nframe: u32) -> Result<Self, DriverError> {
        Ok(Self {
            func: ctx
                .load_module_from_ptx_src(PTX)?
                .load_function("next_stack")?,
            nf_max: nf_max.get(),
            nframe,
        })
    }

    pub fn nf_max(&self) -> u32 {
        self.nf_max
    }

    /// Queues the step on `s`, or records it into the graph `s` captures.
    ///
    /// # Safety
    /// `ctl` and `landed` must address a [`StackCtl`] and a `u32` that stay
    /// backed while the step, or a graph it is captured into, may run, and
    /// nothing else may write `ctl` meanwhile.
    pub unsafe fn queue(
        &self,
        s: &Stream,
        ctl: CUdeviceptr,
        landed: CUdeviceptr,
    ) -> Result<(), DriverError> {
        let (mut ctl, mut landed, mut nf_max, mut nframe) = (ctl, landed, self.nf_max, self.nframe);
        let mut params = [
            (&raw mut ctl).cast::<c_void>(),
            (&raw mut landed).cast(),
            (&raw mut nf_max).cast(),
            (&raw mut nframe).cast(),
        ];
        // SAFETY: the parameters match `next_stack`'s signature and are
        // copied at the launch; the caller keeps the pointees backed.
        unsafe { launch_kernel_on_stream(&self.func, (1, 1, 1), (1, 1, 1), 0, s, &mut params) }
    }
}

#[cfg(test)]
mod tests {
    use cuda_bindings::{self as sys};
    use cuda_core::IntoResult;

    use super::*;
    use crate::consumer::memset32;
    use crate::grow::GrowBuf;

    /// From `{0, 0}`, each step under a `landed` that stalls, grows by less
    /// and more than `nf_max`, and passes `nframe` gives the host model's
    /// range, and none reaches past `nframe` or holds more than `nf_max`.
    #[test]
    fn each_step_covers_the_next_landed_frames_within_the_limits() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let (nf_max, nframe) = (4u32, 20u32);
        let step = NextStack::load(&ctx, NonZeroU32::new(nf_max).unwrap(), nframe).unwrap();
        let mut buf = GrowBuf::new(&ctx, 12, 1).unwrap();
        buf.map(0..12).unwrap();
        let (ctl, landed) = (buf.ptr(), buf.ptr() + 8);
        memset32(ctl, 0, 3, &s).unwrap();

        let mut want = StackCtl::default();
        let lands = [0u32, 3, 3, 10, 10, 10, 25, 25, 25, 25, 25, 30, 7];
        for (i, &l) in lands.iter().enumerate() {
            memset32(landed, l, 1, &s).unwrap();
            // SAFETY: `buf` backs both words until the synchronize below.
            unsafe { step.queue(&s, ctl, landed) }.unwrap();
            s.synchronize().unwrap();
            let mut got = StackCtl::default();
            // SAFETY: the stream is idle and `ctl` holds a StackCtl.
            unsafe { sys::cuMemcpyDtoH_v2((&raw mut got).cast(), ctl, 8) }
                .result()
                .unwrap();
            want.f0 += want.nf;
            want.nf = l.min(nframe).saturating_sub(want.f0).min(nf_max);
            assert_eq!(got, want, "step {i}, landed {l}");
            assert!(got.f0 + got.nf <= nframe && got.nf <= nf_max);
        }
        assert_eq!(want, StackCtl { f0: 20, nf: 0 });
    }
}
