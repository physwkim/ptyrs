//! PyNX DM on the GPU, and the cycle it shares with RAAR.

use crate::gpu::{Error as GpuError, Exit, FusedBufs, FusedInputs, FusedSums};
use cuda_async::device_operation::ExecutionContext;

use super::device::{M1Bufs, PassesMut};
use super::{Bufs, Epoch, GpuPtycho};
use crate::engine::ptycho::llk_due;
use crate::engine::{DmParams, EngineError};

impl GpuPtycho {
    /// [`crate::engine::Ptycho::dm`]. Cycle 0 of a call projects `PO` itself (the reflection maps
    /// `psi = PO` to itself and adds nothing back); later cycles reflect each stack's carried
    /// exit waves against `PO`, project, and add the remainder back (M1: `dm_reflect` into
    /// the scratch stack, `add_c32`). Then `loop_obj_probe` object-only and probe-only
    /// updates, the probe one from the new object; the fused projection adds the first
    /// object-only update's sums itself. The LLK record is of the estimate the cycle starts
    /// from; after the update of each cycle that records one, `Σ|O|` and `Σ|P|` are balanced
    /// with `O P` unchanged (PyNX `ScaleObjProbe(absolute=False)`). With stepped features, a
    /// pass of their own first takes the far sums of `PO` under the cycle-start model and
    /// steps them, and the projection models the stepped values. With a position, every pass
    /// models the probe moved by each frame's offsets, the projection also takes each frame's
    /// position sums of the waves it leaves against the arrays it read, the later updates
    /// take their sums of the carried waves by `shifted_psi_sums`, and the offsets step after
    /// the updates.
    pub fn dm(&mut self, params: &DmParams, ncycle: usize) -> Result<(), EngineError> {
        let epoch = |first| Epoch::Dm {
            params: *params,
            first,
        };
        self.carried(ncycle, params.calc_llk, params.zero_phase_ramp, epoch)
    }

    /// `ncycle` cycles of DM or RAAR, `epoch(first)` each: the LLK record due by `calc_llk`
    /// before the cycle and the balance after it, then the phase ramp's removal.
    pub(super) fn carried(
        &mut self,
        ncycle: usize,
        calc_llk: usize,
        zero_phase_ramp: bool,
        epoch: impl Fn(bool) -> Epoch,
    ) -> Result<(), EngineError> {
        for ic in 0..ncycle {
            let epoch = epoch(ic == 0);
            let llk = llk_due(calc_llk, ic, ncycle);
            if llk {
                self.dev.llk_pass(Some((self.cycle, epoch.algorithm())))?;
            }
            self.dev.steady(epoch, self.cycle)?;
            if llk {
                self.dev.balance_obj_probe()?;
            }
            self.cycle += 1;
        }
        self.collect()?;
        if zero_phase_ramp {
            self.zero_phase_ramp()?;
        }
        Ok(())
    }
}

/// What a DM or RAAR cycle makes of each stack's carried sets and `PO` before the projection.
#[derive(Clone, Copy)]
pub(super) enum Mix {
    /// DM's reflection ([`Exit::Dm`]).
    Dm { alpha: f32 },
    /// RAAR's iterate ([`Exit::Raar`]).
    Raar { beta: f32 },
}

/// A DM or RAAR cycle: its [`Mix`], `first` for cycle 0 of a call, and `loops` object-only
/// then probe-only updates with `inertia` (object, probe).
pub(super) struct Carried {
    pub mix: Mix,
    pub first: bool,
    pub inertia: [Option<f64>; 2],
    pub loops: usize,
}

impl Bufs<'_> {
    /// One DM cycle enqueued on `ctx`; `first` for cycle 0 of a call.
    pub(super) fn dm_cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        params: &DmParams,
        first: bool,
    ) -> Result<(), GpuError> {
        let c = Carried {
            mix: Mix::Dm {
                alpha: params.alpha as f32,
            },
            first,
            inertia: [params.obj_inertia, params.probe_inertia],
            loops: params.loop_obj_probe,
        };
        self.carried_cycle(ctx, passes, &c)
    }

    /// One DM or RAAR cycle enqueued on `ctx`.
    pub(super) fn carried_cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        c: &Carried,
    ) -> Result<(), GpuError> {
        let [obj_inertia, probe_inertia] = c.inertia;
        // Whether the projection added the first object-only update's sums, which read the
        // projected exit waves and the probe the projection read.
        let mut summed = false;
        match passes {
            PassesMut::M1(m1) => {
                for s in 0..self.stacks.len() {
                    match c.mix {
                        Mix::Dm { alpha } => self.dm_m1(ctx, m1, s, alpha, c.first)?,
                        Mix::Raar { beta } => self.raar_m1(ctx, m1, s, beta, c.first)?,
                    }
                }
            }
            PassesMut::Fused(f) => {
                summed = c.loops > 0 && obj_inertia.is_some();
                if self.features.stepped() {
                    // No record, so nothing ended.
                    let _ = self.step_features(ctx, f, None)?;
                }
                let dims = *self.views.dims();
                let mut m = self.features.lend(ctx, self.kernels, &dims, self.probe)?;
                for s in 0..self.stacks.len() {
                    let stack = self.table.at(s);
                    let exit = match c.mix {
                        Mix::Dm { alpha } => {
                            let psi = self.stacks.get(s).psi;
                            match c.first {
                                true => Exit::DmFirst { psi },
                                false => Exit::Dm { psi, alpha },
                            }
                        }
                        Mix::Raar { beta } => {
                            let (st, iterate) = self.stacks.raar(s);
                            let psi = st.psi;
                            match c.first {
                                true => Exit::RaarFirst { psi, iterate },
                                false => Exit::Raar { psi, iterate, beta },
                            }
                        }
                    };
                    let inputs = FusedInputs {
                        views: self.views,
                        counts: self.counts,
                        obj: self.obj,
                        probe: self.probe,
                        model: m.model,
                    };
                    let sums = FusedSums {
                        obj: summed.then_some(&mut *self.obj_sums),
                        shift: m.shift_sums(false, true),
                        ..FusedSums::default()
                    };
                    self.kernels
                        .fused_project(ctx, &inputs, stack, exit, f, sums)?;
                }
            }
        }
        for _ in 0..c.loops {
            for (o, p) in [(obj_inertia, None), (None, probe_inertia)] {
                if o.is_none() && p.is_none() {
                    continue;
                }
                if !std::mem::take(&mut summed) {
                    match passes {
                        PassesMut::Fused(f) if self.features.shifted() => {
                            self.shifted_sums(ctx, f, o.is_some(), p.is_some())?;
                        }
                        _ => {
                            for s in 0..self.stacks.len() {
                                self.accumulate(ctx, s, o.is_some(), p.is_some())?;
                            }
                        }
                    }
                }
                self.apply_sums(ctx, o, p)?;
            }
        }
        self.features.position_step(ctx, self.grad)?;
        Ok(())
    }

    /// The object and/or probe update sums of every stack's carried exit waves under the
    /// position model, against the current arrays: the probe moved anew, since an update
    /// may have changed it; the shifted probe's sums become probe sums after the last stack.
    fn shifted_sums(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
        obj: bool,
        probe: bool,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let mut m = self.features.lend(ctx, self.kernels, &dims, self.probe)?;
        for s in 0..self.stacks.len() {
            let inputs = FusedInputs {
                views: self.views,
                counts: self.counts,
                obj: self.obj,
                probe: self.probe,
                model: m.model,
            };
            let sums = m.psi_sums(obj.then_some(&mut *self.obj_sums), probe);
            let psi = self.stacks.get(s).psi;
            self.kernels
                .shifted_psi_sums(ctx, &inputs, self.table.at(s), psi, f, sums)?;
        }
        if probe {
            m.probe_sums(ctx, self.kernels, &dims, self.probe_sums)?;
        }
        Ok(())
    }

    /// Stack `s`'s DM exit waves by M1's launches.
    fn dm_m1(
        &mut self,
        ctx: &ExecutionContext,
        m1: &mut M1Bufs<'_>,
        s: usize,
        alpha: f32,
        first: bool,
    ) -> Result<(), GpuError> {
        let st = self.stacks.get(s);
        let len = st.frames.len;
        let po = if first {
            &mut *st.psi
        } else {
            &mut *m1.scratch
        };
        self.kernels.obj_probe_to_psi(
            ctx,
            self.views,
            self.table.at(s),
            self.obj,
            self.probe,
            po,
        )?;
        if !first {
            self.kernels
                .dm_reflect(ctx, st.psi, m1.scratch, len, alpha)?;
        }
        self.project(ctx, m1, s, None)?;
        if !first {
            self.kernels
                .add_c32(ctx, self.stacks.get(s).psi, m1.scratch, len)?;
        }
        Ok(())
    }
}
