//! pty-chi LSQML on the GPU ([`crate::engine::Lsqml`]) on the far-field Fourier path, incoherent
//! probe modes and no probe variation, in the call sequence of `crate::gpu::LsqmlKernels`:
//! the exit waves through the fused row passes around LSQML's column pass, the step sizes
//! and every frame's `a_far` kept on the device. Under the position model the probe moves by
//! each frame's offsets, and each batch steps its frames' offsets after the update.

use std::ops::Range;
use std::sync::Arc;

use crate::gpu::{
    BhScan, C32, EpochOrder, Error as GpuError, FusedBufs, FusedInputs, Kernels, LSQML_REC,
    LsqmlAt, LsqmlChi, LsqmlKernels, LsqmlNoise, LsqmlPositions, LsqmlScan, LsqmlVar, Model,
    PsiSums, Shift, ShiftedProbeSums, SpectrumBuf, StackTable, Views,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{FeaturesMut, PassesMut};
use super::{Bufs, Frames, GpuPtycho, batch_stacks, plane_len, stack_buf_len};
use crate::engine::features::{POSITION_LIMIT, POSITION_STEP};
use crate::engine::ptycho::{Algorithm, llk_due};
use crate::engine::{Cost, EngineError, LsqmlParams, LsqmlStep, random_batches};

/// What an epoch does to the object preconditioner before its batches, on
/// [`crate::engine::Lsqml::epoch`]'s schedule: the first epoch of a call builds it, epochs 1–9 and
/// every tenth update it, the others keep it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pre {
    Init,
    Update,
    Keep,
}

impl Pre {
    fn of(epoch: usize) -> Self {
        match epoch {
            0 => Self::Init,
            e if e < 10 || e.is_multiple_of(10) => Self::Update,
            _ => Self::Keep,
        }
    }
}

impl GpuPtycho {
    /// [`crate::engine::Lsqml::epoch`] `nepoch` times from a fresh [`crate::engine::Lsqml`] with no probe
    /// variation, refining the modelled position, over the same [`random_batches`]; one LLK
    /// record after every due epoch.
    pub fn lsqml(
        &mut self,
        params: &LsqmlParams,
        nepoch: usize,
        calc_llk: usize,
    ) -> Result<(), EngineError> {
        if params.batch_size == 0 {
            return Err(EngineError::Invalid("LSQML batch_size must be >= 1".into()));
        }
        if self.dev.has_stepped() {
            return Err(EngineError::Invalid(
                "LSQML refines no intensity scale or background, as pty-chi's".into(),
            ));
        }
        let nframe = self.dev.dims().nframe as usize;
        let size = params.batch_size.min(nframe);
        for ic in 0..nepoch {
            let order: Vec<u32> = random_batches(nframe, size, params.seed, ic as u64)
                .concat()
                .into_iter()
                .map(|f| f as u32)
                .collect();
            self.dev
                .lsqml_epoch(*params, Pre::of(ic), self.cycle, &order)?;
            if llk_due(calc_llk, ic, nepoch) {
                self.dev.llk_pass(Some((self.cycle, Algorithm::Lsqml)))?;
            }
            self.cycle += 1;
        }
        self.collect()?;
        Ok(())
    }

    /// The steps of the last LSQML epoch, as [`crate::engine::Lsqml::epoch`] returns them, read back
    /// from the device; none before an LSQML call.
    pub fn lsqml_steps(&self) -> Result<Vec<LsqmlStep>, EngineError> {
        self.dev.lsqml_steps()
    }
}

/// LSQML's batches over epoch positions and its state, sized by the device and the batch
/// size alone, so one state serves the graphs of every LSQML epoch of that size; the device
/// sets each epoch's frame order before its cycle.
pub(super) struct LsqmlState {
    kernels: LsqmlKernels,
    /// Frames per batch, the batch size capped at the frame count.
    size: u32,
    /// Each batch's range of `stacks`.
    batches: Vec<Range<usize>>,
    /// The batches cut into stacks of the device's stack size, one table entry each.
    stacks: Vec<Frames>,
    table: StackTable,
    /// A batch's row transforms, then exit-wave updates, one per stack of a batch.
    work: Vec<DeviceBuffer<C32>>,
    /// A batch's model intensities, one per stack, for the Poisson `a_far`.
    icalc: Vec<DeviceBuffer<f32>>,
    /// The scan frame at each epoch position, and the views in that order.
    order: EpochOrder,
    scan: LsqmlScan,
    /// The position model's state, if the device models positions.
    moved: Option<Moved>,
}

/// LSQML's state under the position model: a batch's shifted probes, one per stack, and the
/// positions.
struct Moved {
    w: Vec<DeviceBuffer<C32>>,
    pos: LsqmlPositions,
}

/// An [`LsqmlState`] lent to an epoch.
pub(super) struct LsqmlBufs<'a> {
    kernels: &'a LsqmlKernels,
    batches: &'a [Range<usize>],
    stacks: &'a [Frames],
    table: &'a StackTable,
    work: &'a mut [DeviceBuffer<C32>],
    icalc: &'a mut [DeviceBuffer<f32>],
    order: &'a EpochOrder,
    scan: &'a mut LsqmlScan,
    moved: Option<&'a mut Moved>,
}

/// The position model lent to one LSQML batch: the device's position buffers, the fused
/// passes' scratch and LSQML's own [`Moved`].
struct MovedLent<'m> {
    fused: &'m mut FusedBufs,
    /// By frame: the device's offsets, which the batch steps.
    offsets: &'m mut DeviceBuffer<f32>,
    spectrum: &'m mut SpectrumBuf,
    probe_sums: &'m mut ShiftedProbeSums,
    w: &'m mut [DeviceBuffer<C32>],
    pos: &'m mut LsqmlPositions,
}

impl<'m> MovedLent<'m> {
    /// For LSQML's `moved` state with the device's `passes` and `features`: none without a
    /// position, an error when the two disagree.
    fn lend(
        moved: Option<&'m mut Moved>,
        passes: &'m mut PassesMut<'_>,
        features: &'m mut FeaturesMut<'_>,
    ) -> Result<Option<Self>, GpuError> {
        match (moved, features.position_lent(), passes) {
            (None, None, _) => Ok(None),
            (Some(m), Some(p), PassesMut::Fused(fused)) => Ok(Some(Self {
                fused,
                offsets: p.offsets,
                spectrum: p.spectrum,
                probe_sums: p.probe_sums,
                w: &mut m.w,
                pos: &mut m.pos,
            })),
            _ => Err(GpuError::Shape(
                "LSQML state built for another position model".into(),
            )),
        }
    }

    /// The batch's waves under the position model, the spectra of `probe` taken.
    fn waves<'w>(
        &'w mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        views: &Views,
        probe: &'w DeviceBuffer<C32>,
    ) -> Result<MovedWaves<'w>, GpuError> {
        let spectrum = kernels.probe_spectrum(ctx, views.dims(), probe, self.spectrum)?;
        Ok(MovedWaves {
            shift: Shift {
                offsets: self.pos.offsets(),
                spectrum,
            },
            fused: self.fused,
            probe_sums: self.probe_sums,
            w: self.w,
        })
    }
}

/// [`MovedLent`] lent to a batch's exit waves and their update sums: the model of the probe
/// they move, which borrows it.
struct MovedWaves<'w> {
    shift: Shift<'w>,
    fused: &'w mut FusedBufs,
    probe_sums: &'w mut ShiftedProbeSums,
    w: &'w mut [DeviceBuffer<C32>],
}

impl LsqmlState {
    /// For `views` in batches of `size` frames cut into stacks of `stack_size`, `moved` under
    /// the position model; the epoch order is the scan's.
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        size: u32,
        stack_size: u32,
        moved: bool,
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        let ((stacks, table, _), batches) = batch_stacks(stream, dims, size, stack_size)?;
        let len = stack_buf_len(dims, &table);
        let nf_max = table.nf_max();
        let icalc_len = nf_max as usize * plane_len(dims);
        let slots = batches.iter().map(|b| b.len()).max().unwrap_or(0);
        let bufs = |len: usize| -> Result<Vec<DeviceBuffer<C32>>, EngineError> {
            Ok((0..slots)
                .map(|_| DeviceBuffer::zeroed(stream, len))
                .collect::<Result<_, _>>()?)
        };
        let moved = match moved {
            true => Some(Moved {
                w: bufs(nf_max as usize * dims.nprobe as usize * plane_len(dims))?,
                pos: LsqmlPositions::zeroed(stream, dims, nf_max)?,
            }),
            false => None,
        };
        Ok(Self {
            kernels: LsqmlKernels::load(kernels)?,
            size,
            work: bufs(len)?,
            icalc: (0..slots)
                .map(|_| DeviceBuffer::zeroed(stream, icalc_len))
                .collect::<Result<_, _>>()?,
            order: EpochOrder::new(kernels, stream, views)?,
            scan: LsqmlScan::zeroed(stream, dims)?,
            moved,
            batches,
            stacks,
            table,
        })
    }

    pub(super) fn size(&self) -> u32 {
        self.size
    }

    /// Whether the state is the position model's.
    pub(super) fn moved(&self) -> bool {
        self.moved.is_some()
    }

    /// Sets the epoch's frame order: `order` holds the scan frame at each epoch position.
    pub(super) fn set_order(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        views: &Views,
        order: &[u32],
    ) -> Result<(), GpuError> {
        self.order.set(ctx, kernels, views, order)
    }

    pub(super) fn lend(&mut self) -> LsqmlBufs<'_> {
        LsqmlBufs {
            kernels: &self.kernels,
            batches: &self.batches,
            stacks: &self.stacks,
            table: &self.table,
            work: &mut self.work,
            icalc: &mut self.icalc,
            order: &self.order,
            scan: &mut self.scan,
            moved: self.moved.as_mut(),
        }
    }

    /// The last epoch's steps, one per batch, read back: each batch's frames from the epoch
    /// order and each position's `(a_far, a_obj, a_probe)` record.
    pub(super) fn steps(&self, stream: &CudaStream) -> Result<Vec<LsqmlStep>, EngineError> {
        let ids = self.order.ids().to_host_vec(stream)?;
        let records = self.scan.records().to_host_vec(stream)?;
        let records = records.as_chunks::<{ LSQML_REC as usize }>().0;
        Ok(self
            .batches
            .iter()
            .map(|b| {
                let at = self.stacks[b.start].range.start as usize
                    ..self.stacks[b.end - 1].range.end as usize;
                let rec = &records[at.clone()];
                let col = |i: usize| rec.iter().map(|r| f64::from(r[i])).collect();
                LsqmlStep {
                    frames: ids[at].iter().map(|&f| f as usize).collect(),
                    a_far: col(0),
                    a_obj: vec![col(1)],
                    a_probe: col(2),
                }
            })
            .collect())
    }
}

impl<'a> Bufs<'a> {
    /// One epoch enqueued on `ctx`: the preconditioner as `pre` says, then batch by batch over
    /// the frames in the LSQML state's order.
    pub(super) fn lsqml_epoch(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        params: &LsqmlParams,
        pre: Pre,
    ) -> Result<(), GpuError> {
        let mut l = self
            .lsqml
            .take()
            .expect("Device::build_lsqml built the LSQML state");
        let r = self.lsqml_batches(ctx, passes, &mut l, params, pre);
        self.lsqml = Some(l);
        r
    }

    fn lsqml_batches(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        l: &mut LsqmlBufs<'a>,
        params: &LsqmlParams,
        pre: Pre,
    ) -> Result<(), GpuError> {
        match pre {
            Pre::Init => {
                l.kernels.start(ctx, self.counts, l.scan)?;
                l.kernels
                    .precondition(ctx, self.views, self.probe, l.scan, true)?;
            }
            Pre::Update => {
                l.kernels
                    .precondition(ctx, self.views, self.probe, l.scan, false)?;
            }
            Pre::Keep => {}
        }
        // Each frame is in one batch, stepped after its batch's passes: the offsets the
        // epoch starts from serve every pass.
        let moved = l.moved.as_deref_mut();
        if let Some(m) = MovedLent::lend(moved, passes, &mut self.features)? {
            l.kernels.gather_offsets(ctx, l.order, m.offsets, m.pos)?;
        }
        let noise = match params.cost {
            Cost::Amplitude => LsqmlNoise::Gaussian,
            Cost::Poisson => LsqmlNoise::Poisson,
        };
        for b in 0..l.batches.len() {
            self.lsqml_batch(ctx, passes, l, b, noise)?;
        }
        Ok(())
    }

    /// Batch `b` of an epoch: its exit-wave updates, the directions in the update sums, the
    /// step sizes and the update, which consumes the sums; under the position model then the
    /// step of its frames' offsets.
    fn lsqml_batch(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        l: &mut LsqmlBufs<'a>,
        b: usize,
        noise: LsqmlNoise,
    ) -> Result<(), GpuError> {
        let batch = l.batches[b].clone();
        let positions = l.stacks[batch.start].range.start..l.stacks[batch.end - 1].range.end;
        let scan = BhScan {
            order: l.order,
            counts: self.counts,
        };
        let views = l.order.views();
        let mut moved = MovedLent::lend(l.moved.as_deref_mut(), passes, &mut self.features)?;
        let mut waves = (moved.as_mut())
            .map(|m| m.waves(ctx, self.kernels, views, self.probe))
            .transpose()?;
        let inputs = |model| FusedInputs {
            views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model,
        };
        for (k, s) in batch.clone().enumerate() {
            let (stack, work) = (l.table.at(s), &mut l.work[k]);
            match &mut waves {
                None => l
                    .kernels
                    .forward(ctx, views, stack, self.obj, self.probe, work)?,
                Some(mw) => {
                    let model = Model {
                        shift: Some(mw.shift),
                        ..Model::default()
                    };
                    let w = &mut mw.w[k];
                    self.kernels
                        .shifted_rows_fwd(ctx, &inputs(model), stack, w, work)?;
                    l.kernels.shift_w(ctx, views.dims(), stack, w)?;
                }
            }
            match noise {
                LsqmlNoise::Gaussian => l.kernels.chi(ctx, scan, stack, noise, l.scan, work)?,
                LsqmlNoise::Poisson => l.kernels.icalc(ctx, scan, stack, work, &mut l.icalc[k])?,
            }
        }
        if noise == LsqmlNoise::Poisson {
            let icalc: Vec<_> = batch
                .clone()
                .zip(l.icalc.iter())
                .map(|(s, buf)| (l.table.at(s), buf))
                .collect();
            l.kernels
                .far(ctx, scan, &icalc, positions.clone(), l.scan)?;
            for (k, s) in batch.clone().enumerate() {
                let stack = l.table.at(s);
                l.kernels
                    .chi(ctx, scan, stack, noise, l.scan, &mut l.work[k])?;
            }
        }

        match waves {
            None => {
                for (k, s) in batch.clone().enumerate() {
                    let stack = l.table.at(s);
                    self.kernels.psi_to_obj(
                        ctx,
                        views,
                        stack,
                        &l.work[k],
                        self.probe,
                        self.obj_sums,
                    )?;
                    self.kernels.psi_to_probe(
                        ctx,
                        views,
                        stack,
                        &l.work[k],
                        self.obj,
                        self.probe_sums,
                    )?;
                }
            }
            Some(mw) => {
                let model = Model {
                    shift: Some(mw.shift),
                    ..Model::default()
                };
                for (k, s) in batch.clone().enumerate() {
                    let sums = PsiSums {
                        obj: Some(&mut *self.obj_sums),
                        probe: Some(&mut *mw.probe_sums),
                    };
                    let stack = l.table.at(s);
                    self.kernels.shifted_psi_sums(
                        ctx,
                        &inputs(model),
                        stack,
                        &l.work[k],
                        mw.fused,
                        sums,
                    )?;
                }
                self.kernels.shifted_probe_sums(
                    ctx,
                    views.dims(),
                    mw.probe_sums,
                    self.probe_sums,
                )?;
            }
        }
        l.kernels.dir_obj(ctx, l.scan, self.obj_sums)?;
        let nb = positions.len() as f32;
        self.kernels
            .scale_c32(ctx, &mut self.probe_sums.num, self.probe_len, 1.0 / nb)?;
        for (k, s) in batch.enumerate() {
            let at = LsqmlAt {
                x: [&*self.obj, &*self.probe],
                d: [&self.obj_sums.num, &self.probe_sums.num],
            };
            let chi = match &mut moved {
                None => LsqmlChi::Unmoved(&l.work[k]),
                Some(m) => LsqmlChi::Moved {
                    chi: &l.work[k],
                    w: &m.w[k],
                    pos: &mut *m.pos,
                },
            };
            l.kernels.dots(ctx, views, l.table.at(s), at, chi, l.scan)?;
        }
        l.kernels.step(ctx, positions.clone(), l.scan)?;
        let obj = LsqmlVar {
            x: &mut *self.obj,
            sums: &mut *self.obj_sums,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            sums: &mut *self.probe_sums,
        };
        l.kernels.update(ctx, l.scan, obj, probe)?;
        if let Some(m) = moved {
            let limits = [POSITION_STEP as f32, POSITION_LIMIT as f32];
            l.kernels
                .position_step(ctx, positions, l.order, m.offsets, limits, m.pos)?;
        }
        Ok(())
    }
}
