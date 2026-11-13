//! pty-chi rPIE on the GPU, in the call sequence of `crate::gpu::RpieKernels`.

use std::sync::Arc;

use crate::gpu::{
    C32, EpochOrder, Error as GpuError, FusedBufs, FusedInputs, Kernels, Model, PositionKernels,
    PositionRule, Positions, PowerMax, ProbeSpectrum, PsiSums, Shift, ShiftedProbeSums, Views,
    Weight,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{FeaturesMut, PositionLent, RpieBufs};
use super::{Bufs, Epoch, GpuPtycho, plane_len};
use crate::engine::features::{POSITION_LIMIT, POSITION_STEP};
use crate::engine::ptycho::{Algorithm, llk_due};
use crate::engine::{EngineError, RpieParams};

impl GpuPtycho {
    /// [`crate::engine::Ptycho::rpie`]: consecutive batches of `batch_size` frames, each
    /// stepping the object and probe from the batch-start arrays, and under
    /// [`crate::engine::Features::position`] its frames' offsets; one LLK record after every
    /// due epoch.
    pub fn rpie(&mut self, params: &RpieParams, nepoch: usize) -> Result<(), EngineError> {
        if self.dev.dims().nobj != 1 {
            return Err(EngineError::Invalid(
                "rPIE needs exactly one object mode".into(),
            ));
        }
        if params.batch_size == 0 {
            return Err(EngineError::Invalid("rPIE batch_size must be >= 1".into()));
        }
        if self.dev.has_stepped() {
            return Err(EngineError::Invalid(
                "rPIE models no intensity scale or background".into(),
            ));
        }
        for ic in 0..nepoch {
            self.dev.steady(Epoch::Rpie(*params), self.cycle)?;
            if llk_due(params.calc_llk, ic, nepoch) {
                self.dev.llk_pass(Some((self.cycle, Algorithm::Rpie)))?;
            }
            self.cycle += 1;
        }
        self.collect()?;
        Ok(())
    }
}

/// rPIE's state under the position model: a batch's moved probe `[nf_max * nprobe, ny, nx]`,
/// the positions, the scan order they step in, their kernels, and the shifted sums' scratch.
pub(super) struct RpieMoved {
    w: DeviceBuffer<C32>,
    pos: Positions,
    /// Never set: rPIE's batches take the frames in scan order.
    order: EpochOrder,
    kernels: PositionKernels,
    /// Its own, as a batch is one stack, which may outgrow the device's.
    fused: FusedBufs,
}

impl RpieMoved {
    /// For `views` in batches of up to `nf_max` frames.
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        nf_max: u32,
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        let wlen = nf_max as usize * dims.nprobe as usize * plane_len(dims);
        Ok(Self {
            w: DeviceBuffer::zeroed(stream, wlen)?,
            pos: Positions::zeroed(stream, dims, nf_max)?,
            order: EpochOrder::new(kernels, stream, views)?,
            kernels: PositionKernels::load(kernels)?,
            fused: FusedBufs::with_shift(stream, dims, nf_max)?,
        })
    }
}

/// [`RpieMoved`] with the device's position buffers.
struct MovedLent<'m> {
    moved: &'m mut RpieMoved,
    lent: PositionLent<'m>,
}

impl<'m> MovedLent<'m> {
    /// For rPIE's `moved` state with the device's `features`: none without a position, an
    /// error when the two disagree.
    fn lend(
        moved: Option<&'m mut RpieMoved>,
        features: &'m mut FeaturesMut<'_>,
    ) -> Result<Option<Self>, GpuError> {
        match (moved, features.position_lent()) {
            (None, None) => Ok(None),
            (Some(moved), Some(lent)) => Ok(Some(Self { moved, lent })),
            _ => Err(GpuError::Shape(
                "rPIE state built for another position model".into(),
            )),
        }
    }

    /// The lend for one batch, with the spectra of `probe` it starts from (two launches).
    fn batch(
        self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        probe: &'m DeviceBuffer<C32>,
    ) -> Result<MovedBatch<'m>, GpuError> {
        let PositionLent {
            offsets,
            spectrum,
            probe_sums,
        } = self.lent;
        let dims = self.moved.order.views().dims();
        Ok(MovedBatch {
            spectrum: kernels.probe_spectrum(ctx, dims, probe, spectrum)?,
            moved: self.moved,
            offsets,
            probe_sums,
        })
    }
}

/// [`MovedLent`] for one batch, with the spectra of the batch-start probe, which they borrow.
struct MovedBatch<'m> {
    moved: &'m mut RpieMoved,
    /// By frame: the device's offsets, which the batch steps.
    offsets: &'m mut DeviceBuffer<f32>,
    probe_sums: &'m mut ShiftedProbeSums,
    spectrum: ProbeSpectrum<'m>,
}

/// `base` under the batch's position model: the epoch's offsets `pos` and the batch-start
/// `spectrum`.
fn moved_inputs<'a>(
    base: &FusedInputs<'a>,
    pos: &'a Positions,
    spectrum: ProbeSpectrum<'a>,
) -> FusedInputs<'a> {
    let shift = Shift {
        offsets: pos.offsets(),
        spectrum,
    };
    FusedInputs {
        model: Model {
            shift: Some(shift),
            ..Model::default()
        },
        ..*base
    }
}

impl Bufs<'_> {
    /// One epoch enqueued on `ctx`.
    pub(super) fn rpie_epoch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        rb: &mut RpieBufs<'_>,
    ) -> Result<(), GpuError> {
        // Each frame is in one batch, stepped after its batch's passes: the offsets the epoch
        // starts from serve every batch.
        let moved = rb.moved.as_deref_mut();
        if let Some(m) = MovedLent::lend(moved, &mut self.features)? {
            let RpieMoved {
                pos,
                order,
                kernels,
                ..
            } = m.moved;
            kernels.gather(ctx, order, m.lent.offsets, pos)?;
        }
        for b in 0..rb.batches.len() {
            self.rpie_batch(ctx, params, rb, b)?;
        }
        Ok(())
    }

    /// One batch of an epoch enqueued on `ctx`.
    fn rpie_batch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        rb: &mut RpieBufs<'_>,
        b: usize,
    ) -> Result<(), GpuError> {
        let frames = &rb.batches[b];
        let stack = rb.table.at(b);
        let plan = rb.plans.get(frames.plan);
        let dims = *self.views.dims();
        let n = rb.n;
        let nk = frames.nf() * dims.nmode;
        let len = frames.len;
        let (k, rk) = (self.kernels, self.rpie);
        let moved = MovedLent::lend(rb.moved.as_deref_mut(), &mut self.features)?;
        let mut moved = (moved.map(|m| m.batch(ctx, k, self.probe))).transpose()?;
        let base = FusedInputs {
            views: self.views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model: Model::default(),
        };

        // Δ = IFT(ψ') − ψ, ψ = O_v W with W the moved probe under positions.
        match &mut moved {
            Some(m) => {
                let RpieMoved {
                    w, pos, kernels, ..
                } = &mut *m.moved;
                let inputs = moved_inputs(&base, pos, m.spectrum);
                k.shifted_rows_fwd(ctx, &inputs, stack, w, rb.dlt)?;
                kernels.shift_w(ctx, &dims, stack, w)?;
                rk.obj_wave_to_psi(ctx, self.views, stack, self.obj, w, rb.psi)?;
            }
            None => k.obj_probe_to_psi(ctx, self.views, stack, self.obj, self.probe, rb.psi)?,
        }
        plan.forward(ctx, rb.psi, rb.dlt)?;
        rk.amplitude(ctx, &dims, stack, self.counts, rb.dlt)?;
        plan.inverse_inplace(ctx, rb.dlt)?;
        k.scale_c32(ctx, rb.psi, len, -1.0)?;
        k.add_c32(ctx, rb.dlt, rb.psi, len)?;
        if let Some(m) = &mut moved {
            // The slopes of the object the batch started from, against Δ before the probe
            // step weights it.
            let RpieMoved {
                w, pos, kernels, ..
            } = &mut *m.moved;
            kernels.sums(ctx, self.views, stack, self.obj, [&*rb.dlt, &*w], pos)?;
        }

        if params.obj_step.is_some() {
            match &mut moved {
                // `conj(W) Δ / ((1 − a) |W|² + a max Σ_m |W_m|²)`, the max over the batch.
                Some(m) => {
                    let RpieMoved { w, pos, fused, .. } = &mut *m.moved;
                    let groups = PowerMax {
                        ngroup: frames.nf(),
                        nsum: dims.nprobe,
                        n,
                    };
                    rk.power_max_all(ctx, w, groups, rb.max_chunks, rb.pmax)?;
                    let spec = Weight {
                        nk,
                        n,
                        per_plane: false,
                        alpha: params.obj_alpha as f32,
                    };
                    rk.weight(ctx, rb.dlt, w, rb.pmax, rb.psi, spec)?;
                    let sums = PsiSums {
                        obj: Some(&mut *self.obj_sums),
                        probe: None,
                    };
                    let inputs = moved_inputs(&base, pos, m.spectrum);
                    k.shifted_psi_sums(ctx, &inputs, stack, rb.psi, fused, sums)?;
                }
                None => {
                    let groups = PowerMax {
                        ngroup: 1,
                        nsum: dims.nprobe,
                        n,
                    };
                    rk.power_max(ctx, self.probe, groups, rb.max_chunks, rb.pmax)?;
                    let spec = Weight {
                        nk: dims.nprobe,
                        n,
                        per_plane: false,
                        alpha: params.obj_alpha as f32,
                    };
                    rk.weight(ctx, self.probe, self.probe, rb.pmax, rb.ptilde, spec)?;
                    k.psi_to_obj(ctx, self.views, stack, rb.dlt, rb.ptilde, self.obj_sums)?;
                }
            }
        }
        if params.probe_step.is_some() {
            k.obj_probe_to_psi(ctx, self.views, stack, self.obj, rb.ones, rb.psi)?;
            let groups = PowerMax {
                ngroup: nk,
                nsum: 1,
                n,
            };
            rk.power_max(ctx, rb.psi, groups, rb.max_chunks, rb.omax)?;
            let spec = Weight {
                nk,
                n,
                per_plane: true,
                alpha: params.probe_alpha as f32,
            };
            rk.weight_inplace(ctx, rb.dlt, rb.psi, rb.omax, spec)?;
            match &mut moved {
                // `conj(q) T_−d(conj(O) Δ / ((1 − a) |O|² + a max |O|²))`.
                Some(m) => {
                    let sums = PsiSums {
                        obj: None,
                        probe: Some(&mut *m.probe_sums),
                    };
                    let RpieMoved { pos, fused, .. } = &mut *m.moved;
                    let inputs = moved_inputs(&base, pos, m.spectrum);
                    k.shifted_psi_sums(ctx, &inputs, stack, rb.dlt, fused, sums)?;
                    k.shifted_probe_sums(ctx, &dims, m.probe_sums, self.probe_sums)?;
                }
                None => {
                    k.psi_to_probe(ctx, self.views, stack, rb.dlt, self.obj, self.probe_sums)?
                }
            }
        }
        // The offsets' step reads neither array, so it may precede their steps, which end
        // the borrow of the batch-start probe.
        if let Some(m) = moved {
            let rule = PositionRule {
                step: POSITION_STEP as f32,
                limit: POSITION_LIMIT as f32,
            };
            let RpieMoved {
                pos,
                order,
                kernels,
                ..
            } = m.moved;
            kernels.step(ctx, frames.range.clone(), order, m.offsets, rule, pos)?;
        }
        if let Some(step) = params.obj_step {
            let s = step as f32;
            rk.step(ctx, self.obj, self.obj_sums, self.obj_len, s)?;
        }
        if let Some(step) = params.probe_step {
            let s = (step / frames.nf() as f64) as f32;
            rk.step(ctx, self.probe, self.probe_sums, self.probe_len, s)?;
        }
        Ok(())
    }
}
