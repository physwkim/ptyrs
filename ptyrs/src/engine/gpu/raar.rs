//! PyNX RAAR on the GPU.

use crate::gpu::Error as GpuError;
use cuda_async::device_operation::ExecutionContext;

use super::device::{M1Bufs, PassesMut};
use super::dm::{Carried, Mix};
use super::{Bufs, Epoch, GpuPtycho};
use crate::engine::{EngineError, RaarParams};

impl GpuPtycho {
    /// [`crate::engine::Ptycho::raar`]. Each stack carries its exit waves `psi = P(R)` and
    /// RAAR's iterate `R`; cycle 0 of a call starts from `R = PO`, later cycles mix `R = (1 -
    /// beta / 2) psi + beta / 2 (2 PO - R)` (M1: `raar_iterate` from `PO` in the scratch
    /// stack) and project it. The updates, the LLK record and balance, the features and the
    /// position model are DM's ([`Self::dm`]).
    pub fn raar(&mut self, params: &RaarParams, ncycle: usize) -> Result<(), EngineError> {
        let epoch = |first| Epoch::Raar {
            params: *params,
            first,
        };
        self.carried(ncycle, params.calc_llk, params.zero_phase_ramp, epoch)
    }
}

impl Bufs<'_> {
    /// One RAAR cycle enqueued on `ctx`; `first` for cycle 0 of a call.
    pub(super) fn raar_cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        params: &RaarParams,
        first: bool,
    ) -> Result<(), GpuError> {
        let c = Carried {
            mix: Mix::Raar {
                beta: params.beta as f32,
            },
            first,
            inertia: [params.obj_inertia, params.probe_inertia],
            loops: params.loop_obj_probe,
        };
        self.carried_cycle(ctx, passes, &c)
    }

    /// Stack `s`'s RAAR exit waves and iterate by M1's launches.
    pub(super) fn raar_m1(
        &mut self,
        ctx: &ExecutionContext,
        m1: &mut M1Bufs<'_>,
        s: usize,
        beta: f32,
        first: bool,
    ) -> Result<(), GpuError> {
        let (st, iterate) = self.stacks.raar(s);
        let len = st.frames.len;
        let (views, stack) = (self.views, self.table.at(s));
        let (obj, probe) = (&*self.obj, &*self.probe);
        if first {
            // `R = PO`, into both sets.
            for out in [&mut *st.psi, &mut *iterate] {
                self.kernels
                    .obj_probe_to_psi(ctx, views, stack, obj, probe, out)?;
            }
        } else {
            self.kernels
                .obj_probe_to_psi(ctx, views, stack, obj, probe, m1.scratch)?;
            self.kernels
                .raar_iterate(ctx, st.psi, iterate, m1.scratch, len, beta)?;
        }
        self.project(ctx, m1, s, None)
    }
}
