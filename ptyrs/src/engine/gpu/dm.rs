//! PyNX DM on the GPU, and the cycle it shares with RAAR.

use crate::gpu::{Error as GpuError, Exit, FusedBufs, FusedInputs, FusedSums, ShiftProbe};
use cuda_async::device_operation::ExecutionContext;

use super::device::{CarriedMut, M1Bufs, OneSlice, PassesMut, Sample, another_model};
use super::{Bufs, Epoch, GpuPtycho};
use crate::engine::ptycho::llk_due;
use crate::engine::{DmParams, EngineError};
use crate::fault;

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
    /// the updates; under [`crate::engine::Sampling::FullObject`] every pass reads each
    /// stack's windows of the object sampled at the offsets instead, the object sums are
    /// placed back from them, and each object update divides by the illumination map.
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
/// then probe-only updates with `inertia` (object, probe), on an object of `one` slice.
pub(super) struct Carried {
    pub mix: Mix,
    pub first: bool,
    pub inertia: [Option<f64>; 2],
    pub loops: usize,
    pub one: OneSlice,
}

impl Bufs<'_> {
    /// One DM cycle on `w`'s stacks enqueued on `ctx`; `first` for cycle 0 of a call.
    pub(super) fn dm_cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        w: &mut CarriedMut<'_>,
        params: &DmParams,
        first: bool,
        one: OneSlice,
    ) -> Result<(), GpuError> {
        let c = Carried {
            mix: Mix::Dm {
                alpha: params.alpha as f32,
            },
            first,
            inertia: [params.obj_inertia, params.probe_inertia],
            loops: params.loop_obj_probe,
            one,
        };
        self.carried_cycle(ctx, passes, w, &c)
    }

    /// One DM or RAAR cycle on `w`'s stacks enqueued on `ctx`.
    pub(super) fn carried_cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        w: &mut CarriedMut<'_>,
        c: &Carried,
    ) -> Result<(), GpuError> {
        let [obj_inertia, probe_inertia] = c.inertia;
        // Whether the projection added the first object-only update's sums, which read the
        // projected exit waves and the probe the projection read.
        let mut summed = false;
        match passes {
            PassesMut::M1(m1) => {
                for s in 0..w.cut.len() {
                    fault::at(fault::Exit::DmStack)?;
                    match c.mix {
                        Mix::Dm { alpha } => self.dm_m1(ctx, m1, w, s, alpha, c.first)?,
                        Mix::Raar { beta } => self.raar_m1(ctx, m1, w, s, beta, c.first)?,
                    }
                }
            }
            PassesMut::Fused(f) => {
                summed = c.loops > 0 && obj_inertia.is_some();
                if self.features.stepped() {
                    self.step_features(ctx, f, None, c.one)?;
                }
                let dims = *self.views.dims();
                let mut m = (self.features)
                    .lend(ctx, self.kernels, &dims, self.obj, self.probe, self.exit)?
                    .one_slice(c.one);
                for s in 0..w.cut.len() {
                    fault::at(fault::Exit::DmStack)?;
                    let stack = w.cut.at(s);
                    let (psi, iterate, windows) = w.fused(s);
                    let exit = match c.mix {
                        Mix::Dm { alpha } => match c.first {
                            true => Exit::DmFirst { psi: &mut *psi },
                            false => Exit::Dm {
                                psi: &mut *psi,
                                alpha,
                            },
                        },
                        Mix::Raar { beta } => {
                            let iterate = iterate.expect("Device::new_run built the RAAR iterate");
                            match c.first {
                                true => Exit::RaarFirst {
                                    psi: &mut *psi,
                                    iterate,
                                },
                                false => Exit::Raar {
                                    psi: &mut *psi,
                                    iterate,
                                    beta,
                                },
                            }
                        }
                    };
                    let sampled = Sample::Derivatives;
                    if let Some(mut win) = m.windows(ctx, self.views, stack, windows, sampled)? {
                        let read = win.read;
                        let inputs = FusedInputs {
                            views: read.views,
                            counts: self.counts,
                            obj: read.blocks,
                            probe: self.probe,
                            model: m.model,
                        };
                        let sums = FusedSums {
                            obj: summed.then(|| win.zeroed_sums(ctx)).transpose()?,
                            ..FusedSums::default()
                        };
                        (self.kernels).fused_project(ctx, &inputs, stack, exit, f, sums)?;
                        m.window_position_sums(ctx, self.grad, read, stack, &*psi, self.probe)?;
                        if summed {
                            m.place(
                                ctx,
                                self.views,
                                stack,
                                win.patches(),
                                &mut self.obj_sums.num,
                            )?;
                        }
                        continue;
                    }
                    let inputs = FusedInputs {
                        views: self.views,
                        counts: self.counts,
                        obj: self.obj,
                        probe: self.probe,
                        model: m.model,
                    };
                    let sums = FusedSums {
                        obj: summed.then_some(&mut *self.obj_sums),
                        shift: m.shift_sums(None, true),
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
                let mut p = p;
                if !std::mem::take(&mut summed) {
                    match passes {
                        PassesMut::Fused(f) if self.features.shifted() => {
                            // The probe's step is taken there.
                            self.shifted_sums(ctx, f, w, o.is_some(), p.take(), c.one)?;
                        }
                        PassesMut::Fused(_) if self.features.samples_object() => {
                            self.windowed_sums(ctx, w, o.is_some(), p.is_some(), c.one)?;
                        }
                        _ => {
                            for s in 0..w.cut.len() {
                                let (stack, psi) = (w.cut.at(s), w.psi(s));
                                self.accumulate(ctx, stack, psi, o.is_some(), p.is_some())?;
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

    /// The object update sums of every stack's carried exit waves in `w` under the position
    /// model, against the current arrays: the probe moved anew, since an update may have
    /// changed it. With `probe` its inertia, the probe's step of
    /// [`crate::gpu::Kernels::probe_step`] from them instead, taken here against the current
    /// object.
    fn shifted_sums(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
        w: &CarriedMut<'_>,
        obj: bool,
        probe: Option<f64>,
        one: OneSlice,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let mut m = (self.features)
            .lend(ctx, self.kernels, &dims, self.obj, self.probe, self.exit)?
            .one_slice(one);
        for s in 0..w.cut.len() {
            let inputs = FusedInputs {
                views: self.views,
                counts: self.counts,
                obj: self.obj,
                probe: self.probe,
                model: m.model,
            };
            let (stack, psi) = (w.cut.at(s), w.psi(s));
            if obj {
                let sums = m.psi_sums(Some(&mut *self.obj_sums), None);
                self.kernels
                    .shifted_psi_sums(ctx, &inputs, stack, psi, f, sums)?;
            }
            if probe.is_some() {
                let sums = m.psi_sums(None, Some(ShiftProbe::Step));
                self.kernels
                    .shifted_psi_sums(ctx, &inputs, stack, psi, f, sums)?;
            }
        }
        let Some(inertia) = probe else {
            return Ok(());
        };
        m.probe_sums(ctx, self.kernels, &dims, self.probe_sums)?;
        let model = m.model;
        let Some(ShiftProbe::Step(step)) = m.shift_sums(Some(ShiftProbe::Step), false).probe else {
            return Ok(());
        };
        let inputs = FusedInputs {
            views: self.views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model,
        };
        let stacks = (0..w.cut.len()).map(|s| w.cut.at(s));
        let sums = &mut *self.probe_sums;
        self.kernels
            .probe_step(ctx, &inputs, stacks, f, step, sums, inertia as f32)?;
        self.kernels
            .probe_step_add(ctx, self.probe, self.probe_sums)
    }

    /// The update sums of every stack's carried exit waves in `w` under
    /// [`crate::engine::Sampling::FullObject`], against the stack's windows of the current
    /// arrays, sampled anew since an update may have changed the object: with `obj` the
    /// object's, placed back from the windows, with `probe` the probe's.
    fn windowed_sums(
        &mut self,
        ctx: &ExecutionContext,
        w: &mut CarriedMut<'_>,
        obj: bool,
        probe: bool,
        one: OneSlice,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let mut m = (self.features)
            .lend(ctx, self.kernels, &dims, self.obj, self.probe, self.exit)?
            .one_slice(one);
        // The object's sums read the views and the probe alone.
        let sample = match probe {
            true => Sample::Windows,
            false => Sample::Views,
        };
        for s in 0..w.cut.len() {
            let stack = w.cut.at(s);
            let (psi, _, windows) = w.fused(s);
            let win = m.windows(ctx, self.views, stack, windows, sample)?;
            let mut win = win.ok_or_else(another_model)?;
            let read = win.read;
            if obj {
                let sums = win.zeroed_sums(ctx)?;
                (self.kernels).psi_to_obj(ctx, read.views, stack, &*psi, self.probe, sums)?;
                m.place(
                    ctx,
                    self.views,
                    stack,
                    win.patches(),
                    &mut self.obj_sums.num,
                )?;
            }
            if probe {
                let sums = &mut *self.probe_sums;
                (self.kernels).psi_to_probe(ctx, read.views, stack, &*psi, read.blocks, sums)?;
            }
        }
        Ok(())
    }

    /// `w`'s stack `s`'s DM exit waves by M1's launches, `PO` through the shared scratch.
    fn dm_m1(
        &mut self,
        ctx: &ExecutionContext,
        m1: &mut M1Bufs<'_>,
        w: &mut CarriedMut<'_>,
        s: usize,
        alpha: f32,
        first: bool,
    ) -> Result<(), GpuError> {
        let (stack, len) = (w.cut.at(s), w.cut.frames[s].len);
        let (plan, psi, _) = w.m1(s);
        let po = if first { &mut *psi } else { &mut *m1.scratch };
        self.kernels
            .obj_probe_to_psi(ctx, self.views, stack, self.obj, self.probe, po)?;
        if !first {
            self.kernels.dm_reflect(ctx, psi, m1.scratch, len, alpha)?;
        }
        self.project(ctx, stack, plan, psi, None)?;
        if !first {
            self.kernels.add_c32(ctx, psi, m1.scratch, len)?;
        }
        Ok(())
    }
}
