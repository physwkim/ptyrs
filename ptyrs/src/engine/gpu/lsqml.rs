//! pty-chi LSQML on the GPU ([`crate::engine::Lsqml`]) through the device's propagator, incoherent
//! probe modes and no probe variation, in the call sequence of `crate::gpu::LsqmlKernels`:
//! the exit waves through the fused row passes and the propagator's stage around LSQML's
//! column pass, the step sizes and every frame's `a_far` kept on the device. Under the position model the probe moves by
//! each frame's offsets, and each batch steps its frames' offsets after its passes; under
//! [`crate::engine::Sampling::FullObject`] the probe stays and the views read windows of
//! the object the sampler interpolates at each frame's corner and offset instead.
//!
//! A multislice object ([`crate::engine::SliceWaves`]) keeps, per stack of a batch, the wave
//! entering every slice past the first and every slice's exit-wave update: slice 0 then
//! steps with the probe as the one slice does, from its own update, and every other slice
//! alone, from the wave that entered it.

use std::ops::Range;
use std::sync::Arc;

use crate::base::{C64, Feature, Stack};
use crate::gpu::sampling::{Modes, ObjectSampler, StackPoints, WindowBlock, window_views};
use crate::gpu::{
    BandCarry, BhScan, C32, EpochOrder, Error as GpuError, ExitPass, ExitStage, FrameDirs,
    FrameMoves, FrameProbe, FrameWaves, FusedInputs, Kernels, LSQML_REC, LsqmlAt, LsqmlChi,
    LsqmlKernels, LsqmlNoise, LsqmlScan, LsqmlVar, Model, Opr, OprKernels, OprPlanes, OprShape,
    OwnModes, OwnSpectrumBuf, PositionRule, PositionStep, Positions, PsiSums, ShiftProbe,
    SliceSource, SliceSteps, Solve, StackRef, StackTable, ViewDims, Views,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{
    FeaturesMut, Moved, MovedLent, MovedSampling, MovedScratch, PassesMut, PositionLent,
    moved_inputs, moved_shift, slice_phase,
};
use super::{
    Bufs, Cut, Device, Frames, GpuPtycho, StackRule, Streamed, batch_stacks, plane_bytes,
    plane_len, wave_bytes,
};
use crate::engine::features::{POSITION_LIMIT, POSITION_STEP};
use crate::engine::ptycho::{Algorithm, llk_due};
use crate::engine::{
    Cost, EngineError, LsqmlParams, LsqmlStep, ProbeVariation, Sampling, random_batches,
};

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
    /// [`crate::engine::Lsqml::epoch`] `nepoch` times from a fresh [`crate::engine::Lsqml`]
    /// with the probe variation `var`, refining the modelled position, over the same
    /// [`random_batches`]; one LLK record after every due epoch, of each frame's own probe as
    /// [`crate::engine::Lsqml::llk`]'s. `var` comes back stepped with the object and probe;
    /// [`ProbeVariation::none`] runs the shared probe alone. LSQML runs fused row and column
    /// passes whatever [`super::GpuOptions::fused`] says, so frames of a side they cannot take
    /// ([`crate::gpu::FUSED_SIDES`]) are [`EngineError::Invalid`]; so is another variation
    /// without `GpuOptions { fused: true, .. }`, even under `calc_llk` 0, since only the fused
    /// passes take the LLK of each frame's own probe, a variation's batches of more frames than
    /// its launches' grid y takes, 65535, and a variation of other probe planes or of weight
    /// rows other than the frames.
    pub fn lsqml(
        &mut self,
        params: &LsqmlParams,
        var: &mut ProbeVariation,
        nepoch: usize,
        calc_llk: usize,
    ) -> Result<(), EngineError> {
        let nframe = self.dev.checked()?.dims().nframe as usize;
        let [_, ny, nx] = self.probe_dims;
        let mut opr = OprValues::of(var, [ny, nx], nframe)?;
        let varied = opr.is_some();
        let check = |dev: &Device| {
            params.validate()?;
            dev.lsqml_runs(params.batch_size, varied)?;
            if dev.models_scale_or_background() {
                return Err(EngineError::Invalid(
                    "LSQML models no intensity scale or background, as pty-chi's".into(),
                ));
            }
            Ok(())
        };
        self.call(false, check, |dev, cycle| {
            let size = params.batch_size.min(nframe);
            for ic in 0..nepoch {
                let order: Vec<u32> = random_batches(nframe, size, params.seed, ic as u64)
                    .concat()
                    .into_iter()
                    .map(|f| f as u32)
                    .collect();
                dev.lsqml_epoch(*params, Pre::of(ic), opr.as_ref(), *cycle, &order)?;
                if llk_due(calc_llk, ic, nepoch) {
                    dev.lsqml_llk(Some((*cycle, Algorithm::Lsqml)))?;
                }
                *cycle += 1;
            }
            match opr.as_mut() {
                Some(v) if nepoch > 0 => dev.lsqml_read_opr(v),
                _ => Ok(()),
            }
        })?;
        if let Some(v) = opr.filter(|_| nepoch > 0) {
            *var = v.variation();
        }
        Ok(())
    }

    /// The steps of the last LSQML epoch, as [`crate::engine::Lsqml::epoch`] returns them, read back
    /// from the device; none before an LSQML call.
    pub fn lsqml_steps(&self) -> Result<Vec<LsqmlStep>, EngineError> {
        self.dev.checked()?.lsqml_steps()
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
    /// The batches cut into stacks by the bytes a stack's passes touch, one table entry each.
    stacks: Vec<Frames>,
    table: StackTable,
    /// A batch's row transforms, then exit-wave updates, one per stack of a batch.
    work: Vec<DeviceBuffer<C32>>,
    /// Under an [`ExitStage::Banded`] stage, the band of each stack's waves, one per stack of
    /// a batch, from its forward stage to its adjoint ([`exit_stage`]); none under another.
    band: Vec<DeviceBuffer<C32>>,
    /// A batch's model intensities, one per stack, for the Poisson `a_far`.
    icalc: Vec<DeviceBuffer<f32>>,
    /// The scan frame at each epoch position, and the views in that order.
    order: EpochOrder,
    scan: LsqmlScan,
    /// What the views read of the object.
    reading: Reading,
    /// The probe variation's state, if the epochs run one.
    opr: Option<OprBufs>,
    /// A multislice object's waves, if the device's has more than one slice.
    sliced: Option<SliceBufs>,
}

/// What an LSQML state's views read of the object; under the position model with the
/// offsets' step, if the position is refined.
enum Reading {
    /// Each view's patch at its corner.
    Patch,
    /// The patches through the probe moved by each frame's modelled offset
    /// ([`crate::engine::Sampling::Patch`]).
    Moved {
        moved: Box<Moved>,
        step: Option<PositionStep>,
    },
    /// Each view's window of the object at its corner and modelled offset
    /// ([`crate::engine::Sampling::FullObject`]).
    Windows {
        moved: Box<Moved>,
        windows: Box<Windows>,
        step: Option<PositionStep>,
    },
}

/// [`Reading::split`]: the position model's state, the windows' and the offsets' step.
type ReadingSplit<'r> = (
    Option<&'r mut Moved>,
    Option<&'r mut Windows>,
    Option<&'r mut PositionStep>,
);

impl Reading {
    /// The position model's state, under [`crate::engine::Sampling::FullObject`] the windows',
    /// and the offsets' step as `features` refine the position: an error when the state was
    /// built for the other.
    fn split(&mut self, features: &FeaturesMut<'_>) -> Result<ReadingSplit<'_>, GpuError> {
        let (moved, windows, step) = match self {
            Self::Patch => (None, None, None),
            Self::Moved { moved, step } => (Some(&mut **moved), None, step.as_mut()),
            Self::Windows {
                moved,
                windows,
                step,
            } => (Some(&mut **moved), Some(&mut **windows), step.as_mut()),
        };
        Ok((moved, windows, features.own_step(step)?))
    }
}

/// [`Reading::Windows`]' views on its windows and what a batch samples through the
/// device's sampler, which takes the spectra it lends of one slice at the batch's start,
/// then of each object direction.
struct Windows {
    kernels: Kernels,
    /// [`window_views`] of the stacks, which read `win`.
    views: Views,
    /// Per stack of a batch, every slice's windows.
    win: Vec<Vec<DeviceBuffer<C32>>>,
    /// A stack's object numerator on its windows, which the sampler places on the object.
    patches: DeviceBuffer<C32>,
    /// A stack's windows of an object direction.
    dir: DeviceBuffer<C32>,
    /// The illumination map the preconditioner takes.
    illum: DeviceBuffer<f32>,
}

impl Windows {
    /// For `stacks` of views of `dims`, `slots` per batch, through `nslice` slices.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        stacks: &[Frames],
        slots: usize,
        nslice: usize,
    ) -> Result<Self, EngineError> {
        let ranges: Vec<Range<u32>> = stacks.iter().map(|f| f.range.clone()).collect();
        let views = window_views(stream, dims, &ranges)?;
        let len = views.dims().obj_len();
        let win = (0..slots)
            .map(|_| {
                (0..nslice)
                    .map(|_| DeviceBuffer::zeroed(stream, len))
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            kernels: kernels.clone(),
            win,
            patches: DeviceBuffer::zeroed(stream, len)?,
            dir: DeviceBuffer::zeroed(stream, len)?,
            illum: DeviceBuffer::zeroed(stream, dims.obj_len())?,
            views,
        })
    }

    /// The windows of `obj` as slice `i`'s, at each of a batch's stacks `at` in turn, the
    /// object's spectra taken once for the batch into `spectra`.
    fn sample<'b>(
        &mut self,
        ctx: &ExecutionContext,
        (sampler, spectra): (&mut ObjectSampler, &mut DeviceBuffer<C32>),
        at: impl IntoIterator<Item = StackPoints<'b>>,
        obj: &DeviceBuffer<C32>,
        i: usize,
    ) -> Result<(), GpuError> {
        sampler.spectra(ctx, obj, spectra)?;
        for (k, at) in at.into_iter().enumerate() {
            sampler.windows(ctx, at, spectra, WINDOWS, &mut self.win[k][i])?;
        }
        Ok(())
    }

    /// Adds to `num` the object sums of `psi` through `probe` on the stack's windows at `at`,
    /// placed by the sampler's adjoint.
    fn place(
        &mut self,
        ctx: &ExecutionContext,
        sampler: &mut ObjectSampler,
        at: StackPoints<'_>,
        psi: &DeviceBuffer<C32>,
        probe: FrameProbe<'_>,
        num: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        crate::gpu::zero(ctx, &mut self.patches)?;
        let (views, patches) = (&self.views, &mut self.patches);
        (self.kernels).psi_to_obj_frames(ctx, views, at.stack, psi, probe, patches)?;
        sampler.place(ctx, at, &self.patches, 0, num)
    }

    /// The stack's windows at `at` of the object direction whose spectra `spectra` holds,
    /// into `dir`.
    fn direction(
        &mut self,
        ctx: &ExecutionContext,
        (sampler, spectra): (&mut ObjectSampler, &DeviceBuffer<C32>),
        at: StackPoints<'_>,
    ) -> Result<(), GpuError> {
        sampler.windows(ctx, at, spectra, WINDOWS, &mut self.dir)
    }

    /// The derivatives along `y` then `x` of the stack's windows at `at` of object mode 0
    /// of the slice whose spectra `spectra` holds, into `slopes`.
    fn derivatives(
        &self,
        ctx: &ExecutionContext,
        (sampler, spectra): (&mut ObjectSampler, &DeviceBuffer<C32>),
        at: StackPoints<'_>,
        slopes: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        for (block, order) in [[1, 0], [0, 1]].into_iter().enumerate() {
            let block = WindowBlock {
                slice: 0,
                order,
                modes: Modes::First,
                block: block as u32,
            };
            sampler.windows(ctx, at, spectra, block, slopes)?;
        }
        Ok(())
    }
}

/// The windows of every object mode, the block [`Windows::win`] and [`Windows::dir`] hold.
const WINDOWS: WindowBlock = WindowBlock {
    slice: 0,
    order: [0, 0],
    modes: Modes::All,
    block: 0,
};

/// Where the stack's views of `views`, in epoch order, read the object: at the offsets by
/// epoch position the epoch starts from.
fn points<'b>(views: &'b Views, stack: StackRef<'b>, pos: &'b Positions) -> StackPoints<'b> {
    StackPoints {
        views,
        stack,
        offsets: pos.offsets(),
    }
}

/// The step of a batch's offsets, as the CPU's position model takes it.
fn position_solve(scale: f32) -> Solve {
    Solve {
        rule: PositionRule {
            step: POSITION_STEP as f32,
            limit: POSITION_LIMIT as f32,
        },
        scale,
    }
}

fn another_reading() -> GpuError {
    GpuError::Shape("LSQML state built for another position model".into())
}

/// `exit`'s stage on `stack`'s row transforms in `work`, forward or with `adjoint` back. A
/// banded stage's forward saves the band of the waves `w` into the stack's `band` and its
/// adjoint takes it back: a correction `Δ` then comes back as pty-chi's `Uᴴ (U w + Δ) - w`,
/// not `Uᴴ Δ`.
fn exit_stage(
    ctx: &ExecutionContext,
    exit: ExitPass<'_>,
    d: &ViewDims,
    stack: StackRef<'_>,
    work: &mut DeviceBuffer<C32>,
    band: Option<&mut DeviceBuffer<C32>>,
    adjoint: bool,
) -> Result<(), GpuError> {
    let ExitStage::Banded(stage) = exit.stage else {
        return (exit.kernels).exit(ctx, d, stack, exit.stage, work, adjoint);
    };
    let band =
        band.ok_or_else(|| GpuError::Shape("LSQML state built for another exit stage".into()))?;
    let carry = match adjoint {
        false => BandCarry::Save(band),
        true => BandCarry::Restore(band),
    };
    (exit.kernels).exit_banded(ctx, d, stack, stage, work, carry)
}

/// The waves of a batch through a multislice object of `n` slices, and the steps of its
/// slices past the first.
struct SliceBufs {
    /// Per stack of a batch, the waves entering slices `1 .. n`.
    incident: Vec<Vec<DeviceBuffer<C32>>>,
    /// Per stack of a batch, the exit-wave updates of slices `0 .. n - 1`; the last slice's
    /// is the stack's `work`.
    chi: Vec<Vec<DeviceBuffer<C32>>>,
    /// A stack's `conj(incident) chi` of one slice, which the object sums take.
    pair: DeviceBuffer<C32>,
    /// The steps of slices `1 .. n`.
    steps: Vec<SliceSteps>,
}

/// An [`LsqmlState`] lent to an epoch.
pub(super) struct LsqmlBufs<'a> {
    kernels: &'a LsqmlKernels,
    batches: &'a [Range<usize>],
    stacks: &'a [Frames],
    table: &'a StackTable,
    work: &'a mut [DeviceBuffer<C32>],
    band: &'a mut [DeviceBuffer<C32>],
    icalc: &'a mut [DeviceBuffer<f32>],
    order: &'a EpochOrder,
    scan: &'a mut LsqmlScan,
    reading: &'a mut Reading,
    pub(super) opr: Option<&'a mut OprBufs>,
    sliced: Option<&'a mut SliceBufs>,
}

impl LsqmlState {
    /// For `views` in batches of `size` frames, each cut into stacks by `rule`; under the
    /// position model, `scratch` holds the frames per stack of the shared scratch and the
    /// device's sampling: under [`crate::engine::Sampling::Patch`] the moved probe's sums
    /// borrow its shift planes, which caps the state's stacks. `deeper` counts the object's
    /// slices past the first, and `stage` is the device's exit stage. The epoch order is the
    /// scan's.
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        size: u32,
        rule: StackRule,
        scratch: Option<MovedScratch>,
        (deeper, stage): (usize, &ExitStage),
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        // work: the forward pass (0), the exit wave, `chi`, the adjoint exit and back pass (4),
        // then `psi_to_obj` and `psi_to_probe` (5, 6); icalc, which `far` reads batch-wide,
        // streams. Under the position: work from `shifted_rows_fwd` (0, with the moved probes
        // to `shift_w`) to the back pass (5), `shifted_psi_sums` from 6, then `dots` and a refined
        // position's sums (11, 12) read work and the moved probes again. The multislice loops hand
        // on one wave per pass boundary, two under the position: no more than these. Under
        // windows, every slice's: a loop of sampler calls (0 to 2) writes the windows; the waves'
        // loop (3 to 7) hands on one wave as above. Then work from `psi_to_probe` (8) to
        // `psi_to_obj` (18): between them, refined, the derivatives over `y` and `x` (two sampler
        // calls, 9 to 14) for the position sums (15), the window sums zeroed (16, 17); then
        // `place` zeroes its accumulator (19), runs a sampler call (20 to 22), transforms the
        // accumulator and adds it (23, 24); a slice past the first takes its pair (8) instead.
        // A sampler call (25 to 27) writes the direction's windows for `dots` (28). A banded
        // stage's band goes from the exit to the adjoint exit: 1 to 3, 2 to 4 under the
        // position, 4 to 6 under windows.
        let wave = wave_bytes(dims);
        let band = match stage {
            ExitStage::Banded(b) => Some(b),
            _ => None,
        };
        let band_bytes = band.map_or(0, |b| b.carry_len(dims, 1) * size_of::<C32>());
        let streamed = match scratch {
            None => vec![
                Streamed::new(wave, 0..=4),
                Streamed::new(wave, 5..=6),
                Streamed::new(band_bytes, 1..=3),
            ],
            Some(MovedScratch {
                sampling: MovedSampling::Patch,
                refined,
                ..
            }) => {
                let probes = plane_bytes::<C32>(dims, dims.nprobe);
                let mut s = vec![
                    Streamed::new(wave, 0..=5),
                    Streamed::new(probes, 0..=1),
                    Streamed::new(band_bytes, 2..=4),
                ];
                if refined {
                    s.extend([Streamed::new(wave, 11..=12), Streamed::new(probes, 11..=12)]);
                }
                s.extend(Streamed::shifted_psi_sums(dims, 6));
                s
            }
            Some(MovedScratch {
                sampling: MovedSampling::FullObject { held, scratch, acc },
                refined,
                ..
            }) => {
                let plane1 = plane_bytes::<C32>(dims, 1);
                let win = plane_bytes::<C32>(dims, dims.nobj);
                let den = plane_bytes::<f32>(dims, dims.nobj);
                let mut s = [
                    Streamed::chunked(held, 0..=2, &[(win, 2)], &[]),
                    vec![
                        Streamed::new(wave, 3..=7),
                        Streamed::new(band_bytes, 4..=6),
                        Streamed::new(wave, 8..=18),
                    ],
                    vec![Streamed::new(den, 16..=18), Streamed::fixed(acc, 19..=24)],
                    Streamed::chunked(scratch, 20..=22, &[], &[(win, 17)]),
                    Streamed::chunked(held, 25..=27, &[(win, 28)], &[]),
                ]
                .concat();
                if refined {
                    s.extend(Streamed::chunked(held, 9..=11, &[(plane1, 15)], &[]));
                    s.extend(Streamed::chunked(held, 12..=14, &[(plane1, 15)], &[]));
                }
                s
            }
        };
        let frames = rule.frames(&streamed);
        let frames = match scratch {
            Some(MovedScratch {
                frames: shared,
                sampling: MovedSampling::Patch,
                ..
            }) => frames.min(shared),
            _ => frames,
        };
        let (cut, batches) = batch_stacks(stream, dims, size, frames)?;
        let len = cut.wave_len(dims);
        let Cut {
            frames: stacks,
            table,
            ..
        } = cut;
        let nf_max = table.nf_max();
        let icalc_len = nf_max as usize * plane_len(dims);
        let slots = batches.iter().map(|b| b.len()).max().unwrap_or(0);
        let bufs = |len: usize| -> Result<Vec<DeviceBuffer<C32>>, EngineError> {
            Ok((0..slots)
                .map(|_| DeviceBuffer::zeroed(stream, len))
                .collect::<Result<_, _>>()?)
        };
        let reading = match scratch {
            None => Reading::Patch,
            Some(s) => {
                let moved = Box::new(Moved::new(
                    stream,
                    kernels,
                    dims,
                    nf_max,
                    slots,
                    s.sampling.into(),
                )?);
                let step = (s.refined)
                    .then(|| PositionStep::zeroed(stream, dims, nf_max))
                    .transpose()?;
                match s.sampling {
                    MovedSampling::Patch => Reading::Moved { moved, step },
                    MovedSampling::FullObject { .. } => Reading::Windows {
                        moved,
                        windows: Box::new(Windows::new(
                            stream,
                            kernels,
                            dims,
                            &stacks,
                            slots,
                            deeper + 1,
                        )?),
                        step,
                    },
                }
            }
        };
        let sliced = match deeper {
            0 => None,
            _ => {
                let per_slot = || -> Result<Vec<Vec<DeviceBuffer<C32>>>, EngineError> {
                    (0..slots)
                        .map(|_| (0..deeper).map(|_| Ok(DeviceBuffer::zeroed(stream, len)?)))
                        .map(Iterator::collect)
                        .collect()
                };
                Some(SliceBufs {
                    incident: per_slot()?,
                    chi: per_slot()?,
                    pair: DeviceBuffer::zeroed(stream, len)?,
                    steps: (0..deeper)
                        .map(|_| SliceSteps::zeroed(stream, dims))
                        .collect::<Result<_, _>>()?,
                })
            }
        };
        Ok(Self {
            kernels: LsqmlKernels::load(kernels)?,
            size,
            work: bufs(len)?,
            band: match band {
                Some(b) => bufs(b.carry_len(dims, nf_max))?,
                None => Vec::new(),
            },
            icalc: (0..slots)
                .map(|_| DeviceBuffer::zeroed(stream, icalc_len))
                .collect::<Result<_, _>>()?,
            order: EpochOrder::new(kernels, stream, views)?,
            scan: LsqmlScan::zeroed(stream, dims)?,
            reading,
            opr: None,
            sliced,
            batches,
            stacks,
            table,
        })
    }

    pub(super) fn size(&self) -> u32 {
        self.size
    }

    /// The sampling its views read the object under, if the state is the position model's.
    pub(super) fn reads(&self) -> Option<Sampling> {
        match self.reading {
            Reading::Patch => None,
            Reading::Moved { .. } => Some(Sampling::Patch),
            Reading::Windows { .. } => Some(Sampling::FullObject),
        }
    }

    /// The position's [`crate::base::Feature::refine`] the state was built for, none without
    /// the position model.
    pub(super) fn position(&self) -> Option<bool> {
        match &self.reading {
            Reading::Patch => None,
            Reading::Moved { step, .. } | Reading::Windows { step, .. } => Some(step.is_some()),
        }
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

    /// Builds the probe variation's state for `shape` unless it is built, none without one;
    /// an LLK stack holds up to `llk_frames` frames.
    pub(super) fn fit_opr(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        shape: Option<OprShape>,
        llk_frames: u32,
    ) -> Result<(), GpuError> {
        if self.opr.as_ref().map(|o| o.opr.shape()) != shape {
            self.opr = None;
            let shift = matches!(self.reading, Reading::Moved { .. });
            self.opr = (shape
                .map(|s| OprBufs::new(stream, kernels, views, s, self.size, llk_frames, shift)))
            .transpose()?;
        }
        Ok(())
    }

    /// Uploads `v` into the probe variation's state, which is built for its shape.
    pub(super) fn upload_opr(
        &mut self,
        stream: &CudaStream,
        v: &OprValues,
    ) -> Result<(), GpuError> {
        let o = self
            .opr
            .as_mut()
            .expect("Device::build_lsqml fitted the OPR state");
        o.opr
            .upload(stream, &v.eigenmodes, &v.weights, v.intensity.as_deref())
    }

    /// Reads the probe variation's state back into `v`, of its shape.
    pub(super) fn read_opr(
        &self,
        stream: &CudaStream,
        v: &mut OprValues,
    ) -> Result<(), EngineError> {
        let o = self
            .opr
            .as_ref()
            .expect("Device::build_lsqml fitted the OPR state");
        v.eigenmodes = o.opr.eigenmodes().to_host_vec(stream)?;
        v.weights = o.opr.weights().values.to_host_vec(stream)?;
        if let (Some(c), Some(dev)) = (v.intensity.as_mut(), o.opr.intensity()) {
            *c = dev.values.to_host_vec(stream)?;
        }
        Ok(())
    }

    pub(super) fn has_opr(&self) -> bool {
        self.opr.is_some()
    }

    pub(super) fn lend(&mut self) -> LsqmlBufs<'_> {
        LsqmlBufs {
            kernels: &self.kernels,
            batches: &self.batches,
            stacks: &self.stacks,
            table: &self.table,
            work: &mut self.work,
            band: &mut self.band,
            icalc: &mut self.icalc,
            order: &self.order,
            scan: &mut self.scan,
            reading: &mut self.reading,
            opr: self.opr.as_mut(),
            sliced: self.sliced.as_mut(),
        }
    }

    /// The last epoch's steps, one per batch, read back: each batch's frames from the epoch
    /// order and each position's `(a_far, a_obj, a_probe)` record, with the `a_obj` of every
    /// slice past the first after it.
    pub(super) fn steps(&self, stream: &CudaStream) -> Result<Vec<LsqmlStep>, EngineError> {
        let ids = self.order.ids().to_host_vec(stream)?;
        let records = self.scan.records().to_host_vec(stream)?;
        let records = records.as_chunks::<{ LSQML_REC as usize }>().0;
        let deeper: Vec<Vec<f32>> = (self.sliced.iter())
            .flat_map(|s| &s.steps)
            .map(|s| s.a_obj().to_host_vec(stream))
            .collect::<Result<_, _>>()?;
        Ok(self
            .batches
            .iter()
            .map(|b| {
                let at = self.stacks[b.start].range.start as usize
                    ..self.stacks[b.end - 1].range.end as usize;
                let rec = &records[at.clone()];
                let col = |i: usize| rec.iter().map(|r| f64::from(r[i])).collect();
                let deeper = deeper
                    .iter()
                    .map(|a| a[at.clone()].iter().map(|&a| f64::from(a)).collect());
                LsqmlStep {
                    a_far: col(0),
                    a_obj: std::iter::once(col(1)).chain(deeper).collect(),
                    a_probe: col(2),
                    frames: ids[at].iter().map(|&f| f as usize).collect(),
                }
            })
            .collect())
    }
}

/// A [`ProbeVariation`] of eigenmodes as the device holds it: `f32`, its shape the key of
/// the LSQML runs.
pub(super) struct OprValues {
    pub(super) shape: OprShape,
    plane: [usize; 2],
    eigenmodes: Vec<C32>,
    weights: Vec<f32>,
    intensity: Option<Vec<f32>>,
}

impl OprValues {
    /// `var` for a probe of planes `plane` over `nframe` frames; none for a variation of no
    /// eigenmodes, which runs the shared probe alone. [`EngineError::Invalid`] for planes of
    /// another shape or weight rows other than the frames, which
    /// [`crate::engine::Lsqml::epoch`] refuses too.
    pub(super) fn of(
        var: &ProbeVariation,
        plane: [usize; 2],
        nframe: usize,
    ) -> Result<Option<Self>, EngineError> {
        let modes = var.eigenmodes();
        if modes.plane_shape() != plane {
            return Err(EngineError::Invalid(format!(
                "probe variation of {:?} planes on a probe of {plane:?}",
                modes.plane_shape()
            )));
        }
        let k = modes.len();
        if k == 0 {
            return Ok(None);
        }
        let w = var.weights();
        if w.values.len() != nframe * k {
            return Err(EngineError::Invalid(format!(
                "{} OPR weights for {k} eigenmodes over {nframe} frames",
                w.values.len()
            )));
        }
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect();
        Ok(Some(Self {
            shape: OprShape {
                k: super::to_u32("eigenmodes", k)?,
                refine_weights: w.refine,
                intensity: var.intensity().map(|c| c.refine),
            },
            plane,
            eigenmodes: super::to_c32(modes.as_slice()),
            weights: f32s(&w.values),
            intensity: var.intensity().map(|c| f32s(&c.values)),
        }))
    }

    /// The variation these values hold.
    pub(super) fn variation(self) -> ProbeVariation {
        let [ny, nx] = self.plane;
        let f64s = |v: Vec<f32>| v.into_iter().map(f64::from).collect();
        let modes = self.eigenmodes.into_iter();
        let modes = modes.map(|c| C64::new(c.re.into(), c.im.into())).collect();
        let modes = Stack::from_vec([self.shape.k as usize, ny, nx], modes)
            .expect("OprValues::of took k planes of the probe's");
        let weights = Feature {
            values: f64s(self.weights),
            refine: self.shape.refine_weights,
        };
        let var = ProbeVariation::new(modes, weights);
        match (self.intensity, self.shape.intensity) {
            (Some(values), Some(refine)) => var.with_intensity(Feature {
                values: f64s(values),
                refine,
            }),
            _ => var,
        }
    }
}

/// LSQML's probe variation on the device, and the planes its batches and LLK stacks take.
pub(super) struct OprBufs {
    kernels: OprKernels,
    opr: Opr,
    /// The own mode-0 planes of a batch or of an LLK stack.
    planes: DeviceBuffer<C32>,
    /// Their spectra, which the moved probe reads.
    spectra: Option<OwnSpectrumBuf>,
    /// A batch's [`OprPlanes`].
    d: DeviceBuffer<C32>,
    vv: DeviceBuffer<f32>,
    moved: DeviceBuffer<C32>,
    /// The scan's order, which the LLK's stacks follow.
    scan: EpochOrder,
}

impl OprBufs {
    /// For batches of up to `size` frames and LLK stacks of up to `llk_frames`; with `shift`
    /// the planes' spectra too.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        shape: OprShape,
        size: u32,
        llk_frames: u32,
        shift: bool,
    ) -> Result<Self, GpuError> {
        let dims = views.dims();
        let n = dims.n();
        let nb = size.max(llk_frames);
        let batch = size as usize * n;
        Ok(Self {
            kernels: OprKernels::load(kernels)?,
            opr: Opr::zeroed(stream, n, dims.nframe, shape, size)?,
            planes: DeviceBuffer::zeroed(stream, nb as usize * n)?,
            spectra: shift
                .then(|| OwnSpectrumBuf::zeroed(stream, dims, nb))
                .transpose()?,
            d: DeviceBuffer::zeroed(stream, batch)?,
            vv: DeviceBuffer::zeroed(stream, batch)?,
            moved: DeviceBuffer::zeroed(stream, batch)?,
            scan: EpochOrder::new(kernels, stream, views)?,
        })
    }

    /// The own modes of the frames at the epoch `positions` of `order`, each frame's
    /// [`ProbeVariation::unique`] mode 0 of `probe`, with their spectra under `shift`.
    fn modes<'o>(
        planes: &'o mut DeviceBuffer<C32>,
        spectra: &'o mut Option<OwnSpectrumBuf>,
        (ctx, kernels, opr): (&ExecutionContext, (&Kernels, &OprKernels), &Opr),
        (order, positions): (&EpochOrder, Range<u32>),
        probe: &DeviceBuffer<C32>,
        shift: bool,
    ) -> Result<OwnModes<'o>, GpuError> {
        let (k, ok) = kernels;
        ok.unique(ctx, opr, order, positions.clone(), probe, planes)?;
        let own = OwnModes::new(planes, positions);
        match (spectra, shift) {
            (_, false) => Ok(own),
            (Some(buf), true) => Ok(k.own_spectrum(ctx, order.views().dims(), &own, buf)?),
            (None, true) => Err(GpuError::Shape(
                "OPR state built without the moved probe's spectra".into(),
            )),
        }
    }

    /// The batch at the epoch `positions` of `order` from `probe`, the shared probe it starts
    /// from; `shift` when its probe moves.
    fn batch(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        order: &EpochOrder,
        positions: Range<u32>,
        probe: &DeviceBuffer<C32>,
        shift: bool,
    ) -> Result<BatchOpr<'_>, GpuError> {
        let Self {
            kernels: ok,
            opr,
            planes,
            spectra,
            d,
            vv,
            moved,
            ..
        } = self;
        let at = (ctx, (kernels, &*ok), &*opr);
        let own = Self::modes(planes, spectra, at, (order, positions), probe, shift)?;
        Ok(BatchOpr {
            own,
            kernels: ok,
            opr,
            d,
            vv,
            moved,
        })
    }

    /// The own modes of the LLK stack `stack` of the scan, as [`Self::batch`]'s.
    pub(super) fn scan_modes(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        stack: StackRef<'_>,
        probe: &DeviceBuffer<C32>,
        shift: bool,
    ) -> Result<OwnModes<'_>, GpuError> {
        let frames = stack.frames().ok_or_else(|| {
            GpuError::Shape("the LLK's own probe modes need a StackTable entry".into())
        })?;
        let Self {
            kernels: ok,
            opr,
            planes,
            spectra,
            scan,
            ..
        } = self;
        let at = (ctx, (kernels, &*ok), &*opr);
        Self::modes(planes, spectra, at, (&*scan, frames), probe, shift)
    }
}

/// A batch's probe variation: its frames' own probe modes, then its step from their
/// directions.
struct BatchOpr<'o> {
    own: OwnModes<'o>,
    kernels: &'o OprKernels,
    opr: &'o mut Opr,
    d: &'o mut DeviceBuffer<C32>,
    vv: &'o mut DeviceBuffer<f32>,
    moved: &'o mut DeviceBuffer<C32>,
}

impl BatchOpr<'_> {
    /// Where [`LsqmlKernels::frame_dirs`] writes the frames at the epoch `positions`.
    fn dirs(&mut self, positions: Range<u32>) -> FrameDirs<'_> {
        FrameDirs {
            d: self.d,
            vv: self.vv,
            positions,
        }
    }

    /// The step of the variation at the epoch `positions`, from the shared `probe` the batch
    /// started from. A probe that does not move takes its directions back unmoved, as
    /// [`crate::engine::LsqmlBatch::adjoint_move`] does; a moved one has had them taken back
    /// into `moved` by [`ShiftProbe::Frames`].
    fn step(
        self,
        ctx: &ExecutionContext,
        (order, positions): (&EpochOrder, Range<u32>),
        probe: &DeviceBuffer<C32>,
        moved: bool,
    ) -> Result<(), GpuError> {
        if !moved {
            crate::gpu::copy(ctx, self.d, self.moved)?;
        }
        let planes = OprPlanes {
            d: self.d,
            vv: self.vv,
            moved: self.moved,
        };
        (self.kernels).step(ctx, self.opr, order, positions, probe, planes)
    }
}

/// `probe` as the frames read it: with `opr`, each frame's own mode 0.
fn frame_probe<'p>(probe: &'p DeviceBuffer<C32>, opr: Option<&'p BatchOpr<'_>>) -> FrameProbe<'p> {
    match opr {
        None => FrameProbe::Shared(probe),
        Some(o) => FrameProbe::Own(probe, &o.own),
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
        let k = self.exit.scale;
        if pre == Pre::Init {
            l.kernels.start(ctx, self.counts, l.scan)?;
        }
        // Each frame is in one batch, stepped after its batch's passes: the offsets the
        // epoch starts from serve every pass, and the windows' illumination.
        {
            let (moved, windows, _) = l.reading.split(&self.features)?;
            let moved = moved.map(|m| (m, ()));
            let mut lent = PositionLent::lend(moved, passes.fused(), &mut self.features)?;
            if let Some(m) = &mut lent {
                m.gather(ctx, l.order)?;
            }
            if pre != Pre::Keep {
                let init = pre == Pre::Init;
                match (windows, &mut lent) {
                    (Some(w), Some(PositionLent::FullObject(s))) => {
                        let offsets = &*s.offsets;
                        (s.sampler).illumination(
                            ctx,
                            self.views,
                            offsets,
                            self.probe,
                            &mut w.illum,
                        )?;
                        l.kernels
                            .precondition_placed(ctx, &w.illum, l.scan, init, k)?;
                    }
                    (None, None | Some(PositionLent::Patch(_))) => l
                        .kernels
                        .precondition(ctx, self.views, self.probe, l.scan, init, k)?,
                    _ => return Err(another_reading()),
                }
            }
        }
        let noise = match params.cost {
            Cost::Amplitude => LsqmlNoise::Gaussian,
            Cost::Poisson => LsqmlNoise::Poisson,
        };
        for b in 0..l.batches.len() {
            match (&*l.reading, &l.sliced) {
                (Reading::Windows { .. }, None) => self.lsqml_batch_windows(ctx, l, b, noise)?,
                (Reading::Windows { .. }, Some(_)) => {
                    self.lsqml_batch_sliced_windows(ctx, l, b, noise)?
                }
                (_, None) => self.lsqml_batch(ctx, passes, l, b, noise)?,
                (_, Some(_)) => self.lsqml_batch_sliced(ctx, passes, l, b, noise)?,
            }
        }
        Ok(())
    }

    /// Batch `b` of an epoch: its exit-wave updates, the directions in the update sums, the
    /// step sizes and the update, which consumes the sums; under the position model the step
    /// of its frames' offsets before the update.
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
        let (moved, _, mut step) = l.reading.split(&self.features)?;
        let moved = MovedLent::lend(moved.map(|m| (m, ())), passes.fused(), &mut self.features)?;
        let mut moved =
            (moved.map(|m| m.batch(ctx, self.kernels, views.dims(), self.probe))).transpose()?;
        let shift = moved.is_some();
        let mut opr = (l.opr.as_deref_mut())
            .map(|o| {
                o.batch(
                    ctx,
                    self.kernels,
                    l.order,
                    positions.clone(),
                    self.probe,
                    shift,
                )
            })
            .transpose()?;
        let base = FusedInputs {
            views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model: Model::default(),
        };
        // Row transforms at the detector plane into `chi`: the column pass, the propagator's
        // adjoint stage and the inverse row pass.
        let (lk, exit) = (l.kernels, self.exit);
        let chi = |st: &LsqmlScan,
                   stack: StackRef<'_>,
                   work: &mut DeviceBuffer<C32>,
                   band: Option<&mut DeviceBuffer<C32>>| {
            lk.chi(ctx, scan, stack, noise, st, work)?;
            exit_stage(ctx, exit, views.dims(), stack, work, band, true)?;
            lk.back(ctx, views, stack, work)
        };
        for (k, s) in batch.clone().enumerate() {
            let (stack, work) = (l.table.at(s), &mut l.work[k]);
            match &mut moved {
                None => {
                    let probe = frame_probe(self.probe, opr.as_ref());
                    (l.kernels).forward_frames(ctx, views, stack, self.obj, probe, work)?
                }
                Some(m) => {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let w = &mut m.w[k];
                    // Its row pass is the stack's forward pass: `exit` goes on from `work`.
                    match &opr {
                        None => (self.kernels).shifted_rows_fwd(ctx, &inputs, stack, w, work)?,
                        Some(o) => (self.kernels)
                            .shifted_rows_fwd_own(ctx, &inputs, &o.own, stack, w, work)?,
                    }
                    m.kernels.shift_w(ctx, views.dims(), stack, w)?;
                }
            }
            exit_stage(
                ctx,
                exit,
                views.dims(),
                stack,
                work,
                l.band.get_mut(k),
                false,
            )?;
            match noise {
                LsqmlNoise::Gaussian => chi(l.scan, stack, work, l.band.get_mut(k))?,
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
                chi(l.scan, l.table.at(s), &mut l.work[k], l.band.get_mut(k))?;
            }
        }

        match &mut moved {
            None => {
                for (k, s) in batch.clone().enumerate() {
                    let stack = l.table.at(s);
                    self.kernels.psi_to_obj_frames(
                        ctx,
                        views,
                        stack,
                        &l.work[k],
                        frame_probe(self.probe, opr.as_ref()),
                        &mut self.obj_sums.num,
                    )?;
                    self.kernels.psi_to_probe(
                        ctx,
                        views,
                        stack,
                        &l.work[k],
                        self.obj,
                        &mut self.probe_sums.num,
                    )?;
                }
            }
            Some(m) => {
                for (k, s) in batch.clone().enumerate() {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let (stack, psi) = (l.table.at(s), &l.work[k]);
                    match &mut opr {
                        None => {
                            let probe = Some(ShiftProbe::Num(&mut *m.probe_sums));
                            let sums = PsiSums {
                                obj: Some(&mut self.obj_sums.num),
                                probe,
                            };
                            (self.kernels)
                                .shifted_psi_sums(ctx, &inputs, stack, psi, m.fused, sums)?
                        }
                        Some(o) => {
                            let probe = Some(ShiftProbe::Frames {
                                sums: &mut *m.probe_sums,
                                moved: FrameMoves {
                                    planes: &mut *o.moved,
                                    positions: positions.clone(),
                                },
                            });
                            let sums = PsiSums {
                                obj: Some(&mut self.obj_sums.num),
                                probe,
                            };
                            let at = (&o.own, stack);
                            (self.kernels)
                                .shifted_psi_sums_own(ctx, &inputs, at, psi, m.fused, sums)?
                        }
                    }
                }
                self.kernels.shifted_probe_sums(
                    ctx,
                    views.dims(),
                    m.probe_sums,
                    &mut self.probe_sums.num,
                )?;
            }
        }
        l.kernels.dir_obj(ctx, l.scan, &mut self.obj_sums.num)?;
        let nb = positions.len() as f32;
        self.kernels
            .scale_c32(ctx, &mut self.probe_sums.num, self.probe_len, 1.0 / nb)?;
        for (k, s) in batch.enumerate() {
            let (stack, chi) = (l.table.at(s), &l.work[k]);
            let at = LsqmlAt {
                x: [&*self.obj, &*self.probe],
                d: [&self.obj_sums.num, &self.probe_sums.num],
            };
            let c = match (&moved, &opr) {
                (None, None) => LsqmlChi::Unmoved(chi),
                (None, Some(o)) => LsqmlChi::Own {
                    chi,
                    own: o.own.clone(),
                },
                (Some(m), _) => LsqmlChi::Moved {
                    chi,
                    w: FrameWaves::Probes(&m.w[k]),
                },
            };
            l.kernels.dots(ctx, views, stack, at, c, l.scan)?;
            if let (Some(m), Some(step)) = (&moved, step.as_deref_mut()) {
                // The slopes of the object the batch started from.
                let at = (chi, FrameWaves::Probes(&m.w[k]));
                m.kernels.sums(ctx, views, stack, self.obj, at, step)?;
            }
            if let Some(o) = &mut opr {
                let dirs = o.dirs(positions.clone());
                l.kernels
                    .frame_dirs(ctx, views, stack, self.obj, chi, dirs)?;
            }
        }
        l.kernels
            .step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        // Before the update, as the batch's spectra borrow the probe; the step reads neither.
        if let (Some(m), Some(step)) = (moved, step) {
            let solve = position_solve(self.exit.scale);
            let offsets = m.offsets;
            m.kernels
                .step(ctx, positions.clone(), l.order, offsets, solve, step)?;
        }
        if let Some(o) = opr {
            o.step(ctx, (l.order, positions), self.probe, shift)?;
        }
        let obj = LsqmlVar {
            x: &mut *self.obj,
            num: &mut self.obj_sums.num,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            num: &mut self.probe_sums.num,
        };
        l.kernels.update(ctx, l.scan, obj, probe)?;
        Ok(())
    }

    /// [`Self::lsqml_batch`] on the windows of the object the sampler reads at each frame's
    /// corner and offset: the views read a stack's windows, its sums on them go back onto the
    /// object through the sampler's adjoint, the dot products read the windows of the object
    /// direction, and the position sums the windows' derivatives, the probe staying put.
    fn lsqml_batch_windows(
        &mut self,
        ctx: &ExecutionContext,
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
        let (views, table) = (l.order.views(), l.table);
        let d = views.dims();
        let (moved, w, mut step) = l.reading.split(&self.features)?;
        let lent = PositionLent::lend(moved.map(|m| (m, ())), None, &mut self.features)?;
        let (Some(PositionLent::FullObject(p)), Some(w)) = (lent, w) else {
            return Err(another_reading());
        };
        let (lk, exit) = (l.kernels, self.exit);
        let mut opr = (l.opr.as_deref_mut())
            .map(|o| {
                o.batch(
                    ctx,
                    self.kernels,
                    l.order,
                    positions.clone(),
                    self.probe,
                    false,
                )
            })
            .transpose()?;
        let at = batch.clone().map(|s| points(views, table.at(s), &*p.pos));
        w.sample(ctx, (p.sampler, p.spectra), at, self.obj, 0)?;
        let wv = &w.views;
        let chi = |st: &LsqmlScan,
                   stack: StackRef<'_>,
                   work: &mut DeviceBuffer<C32>,
                   band: Option<&mut DeviceBuffer<C32>>| {
            lk.chi(ctx, scan, stack, noise, st, work)?;
            exit_stage(ctx, exit, d, stack, work, band, true)?;
            lk.back(ctx, wv, stack, work)
        };
        for (k, s) in batch.clone().enumerate() {
            let (stack, work) = (table.at(s), &mut l.work[k]);
            let probe = frame_probe(self.probe, opr.as_ref());
            lk.forward_frames(ctx, wv, stack, &w.win[k][0], probe, work)?;
            exit_stage(ctx, exit, d, stack, work, l.band.get_mut(k), false)?;
            match noise {
                LsqmlNoise::Gaussian => chi(l.scan, stack, work, l.band.get_mut(k))?,
                LsqmlNoise::Poisson => lk.icalc(ctx, scan, stack, work, &mut l.icalc[k])?,
            }
        }
        if noise == LsqmlNoise::Poisson {
            let icalc: Vec<_> = batch
                .clone()
                .zip(l.icalc.iter())
                .map(|(s, buf)| (table.at(s), buf))
                .collect();
            lk.far(ctx, scan, &icalc, positions.clone(), l.scan)?;
            for (k, s) in batch.clone().enumerate() {
                chi(l.scan, table.at(s), &mut l.work[k], l.band.get_mut(k))?;
            }
        }
        for (k, s) in batch.clone().enumerate() {
            let (stack, work) = (table.at(s), &l.work[k]);
            let probe = frame_probe(self.probe, opr.as_ref());
            (self.kernels).psi_to_probe(
                ctx,
                &w.views,
                stack,
                work,
                &w.win[k][0],
                &mut self.probe_sums.num,
            )?;
            if let Some(step) = step.as_deref_mut() {
                // The slopes of the windows the batch started from.
                let derivatives = |out: &mut _| {
                    let at = points(views, stack, p.pos);
                    w.derivatives(ctx, (p.sampler, p.spectra), at, out)
                };
                let at = (work, SliceSource::from(probe));
                (p.kernels).derivative_sums(ctx, views, stack, derivatives, at, step)?;
            }
            let at = points(views, stack, p.pos);
            w.place(ctx, p.sampler, at, work, probe, &mut self.obj_sums.num)?;
        }
        lk.dir_obj(ctx, l.scan, &mut self.obj_sums.num)?;
        let nb = positions.len() as f32;
        self.kernels
            .scale_c32(ctx, &mut self.probe_sums.num, self.probe_len, 1.0 / nb)?;
        p.sampler.spectra(ctx, &self.obj_sums.num, p.spectra)?;
        for (k, s) in batch.enumerate() {
            let stack = table.at(s);
            w.direction(ctx, (p.sampler, p.spectra), points(views, stack, p.pos))?;
            let at = LsqmlAt {
                x: [&w.win[k][0], &*self.probe],
                d: [&w.dir, &self.probe_sums.num],
            };
            let chi = &l.work[k];
            let c = match &opr {
                None => LsqmlChi::Unmoved(chi),
                Some(o) => LsqmlChi::Own {
                    chi,
                    own: o.own.clone(),
                },
            };
            lk.dots(ctx, &w.views, stack, at, c, l.scan)?;
            if let Some(o) = &mut opr {
                let dirs = o.dirs(positions.clone());
                lk.frame_dirs(ctx, &w.views, stack, &w.win[k][0], chi, dirs)?;
            }
        }
        lk.step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        if let Some(step) = step {
            let solve = position_solve(self.exit.scale);
            let offsets = p.offsets;
            (p.kernels).step(ctx, positions.clone(), l.order, offsets, solve, step)?;
        }
        if let Some(o) = opr {
            o.step(ctx, (l.order, positions), self.probe, false)?;
        }
        let obj = LsqmlVar {
            x: &mut *self.obj,
            num: &mut self.obj_sums.num,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            num: &mut self.probe_sums.num,
        };
        lk.update(ctx, l.scan, obj, probe)
    }

    /// [`Self::lsqml_batch`] through a multislice object: every stack's waves through the
    /// slices and their updates back; then the slices from the last to the second, each from
    /// the wave that entered it, alone; and slice 0 with the probe as the one slice, its
    /// object step shared with the others. Slice 0's passes read none of the others.
    fn lsqml_batch_sliced(
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
        let d = views.dims();
        let (moved, _, mut step) = l.reading.split(&self.features)?;
        let moved = MovedLent::lend(moved.map(|m| (m, ())), passes.fused(), &mut self.features)?;
        let mut moved = (moved.map(|m| m.batch(ctx, self.kernels, d, self.probe))).transpose()?;
        let shift = moved.is_some();
        let mut opr = (l.opr.as_deref_mut())
            .map(|o| {
                o.batch(
                    ctx,
                    self.kernels,
                    l.order,
                    positions.clone(),
                    self.probe,
                    shift,
                )
            })
            .transpose()?;
        let base = FusedInputs {
            views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model: Model::default(),
        };
        let (lk, exit, ok) = (l.kernels, self.exit, self.exit.kernels);
        let (Some(sl), Some(w)) = (self.slices.as_mut(), l.sliced.as_deref_mut()) else {
            return Err(GpuError::Shape(
                "LSQML state built for another slice count".into(),
            ));
        };
        let last = self.deeper.len();
        let share = (last + 1) as f32;
        // The last slice's plane phase, which moves with the probe.
        let quad = sl.quad;
        for (k, s) in batch.clone().enumerate() {
            let (stack, work, inc) = (l.table.at(s), &mut l.work[k], &mut w.incident[k]);
            let offsets = match &mut moved {
                None => {
                    let probe = SliceSource::from(frame_probe(self.probe, opr.as_ref()));
                    ok.slice_mul(ctx, views, stack, self.obj, probe, &mut inc[0])?;
                    None
                }
                Some(m) => {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let pw = &mut m.w[k];
                    match &opr {
                        None => self.kernels.shifted_probe(ctx, &inputs, stack, pw)?,
                        Some(o) => {
                            (self.kernels).shifted_probe_own(ctx, &inputs, &o.own, stack, pw)?
                        }
                    }
                    m.kernels.shift_w(ctx, d, stack, pw)?;
                    let probe = SliceSource::Waves(FrameWaves::Probes(pw));
                    ok.slice_mul(ctx, views, stack, self.obj, probe, &mut inc[0])?;
                    Some(m.pos.offsets())
                }
            };
            for i in 1..=last {
                ok.gap(ctx, d, stack, &sl.gaps[i - 1], &mut inc[i - 1], false)?;
                let (src, rest) = inc[i - 1..].split_first_mut().expect("i <= last");
                let dst = rest.first_mut().unwrap_or(&mut *work);
                let src = SliceSource::Waves(FrameWaves::Views(src));
                ok.slice_mul(ctx, views, stack, &self.deeper[i - 1], src, dst)?;
            }
            if let Some(p) = slice_phase(quad, offsets) {
                ok.slice_phase(ctx, d, stack, p, work, false)?;
            }
            ok.rows(ctx, d, stack, work, false)?;
            exit_stage(ctx, exit, d, stack, work, l.band.get_mut(k), false)?;
            if noise == LsqmlNoise::Poisson {
                lk.icalc(ctx, scan, stack, work, &mut l.icalc[k])?;
            }
        }
        if noise == LsqmlNoise::Poisson {
            let icalc: Vec<_> = batch
                .clone()
                .zip(l.icalc.iter())
                .map(|(s, buf)| (l.table.at(s), buf))
                .collect();
            lk.far(ctx, scan, &icalc, positions.clone(), l.scan)?;
        }
        // Every slice's update: the last's from the detector plane, then back through the
        // slices; the position sums, on the slice pty-chi corrects (`nslice / 2`), before
        // any slice steps.
        let pivot = last.div_ceil(2);
        for (k, s) in batch.clone().enumerate() {
            let (stack, work, chi) = (l.table.at(s), &mut l.work[k], &mut w.chi[k]);
            lk.chi(ctx, scan, stack, noise, l.scan, work)?;
            exit_stage(ctx, exit, d, stack, work, l.band.get_mut(k), true)?;
            lk.back(ctx, views, stack, work)?;
            let offsets = moved.as_ref().map(|m| m.pos.offsets());
            if let Some(p) = slice_phase(quad, offsets) {
                ok.slice_phase(ctx, d, stack, p, work, true)?;
            }
            for i in (0..last).rev() {
                let (dst, rest) = chi[i..].split_first_mut().expect("i < last");
                let src = rest.first().unwrap_or(&*work);
                ok.slice_adjoint(ctx, views, stack, &self.deeper[i], src, dst)?;
                ok.gap(ctx, d, stack, &sl.gaps[i], dst, true)?;
            }
            if let (Some(m), Some(step)) = (&moved, step.as_deref_mut()) {
                let c = chi.get(pivot).unwrap_or(&*work);
                let at = (c, FrameWaves::Views(&w.incident[k][pivot - 1]));
                m.kernels
                    .sums(ctx, views, stack, &self.deeper[pivot - 1], at, step)?;
            }
        }
        for i in (1..=last).rev() {
            for (k, s) in batch.clone().enumerate() {
                let stack = l.table.at(s);
                let pair = [&w.incident[k][i - 1], w.chi[k].get(i).unwrap_or(&l.work[k])];
                ok.slice_pair(ctx, d, stack, pair, &mut w.pair)?;
                self.kernels.psi_to_obj(
                    ctx,
                    views,
                    stack,
                    &w.pair,
                    sl.ones,
                    &mut self.obj_sums.num,
                )?;
            }
            lk.dir_obj(ctx, l.scan, &mut self.obj_sums.num)?;
            for (k, s) in batch.clone().enumerate() {
                let at = LsqmlAt {
                    x: [&self.deeper[i - 1], &*self.probe],
                    d: [&self.obj_sums.num, &self.probe_sums.num],
                };
                let c = LsqmlChi::Moved {
                    chi: w.chi[k].get(i).unwrap_or(&l.work[k]),
                    w: FrameWaves::Views(&w.incident[k][i - 1]),
                };
                lk.dots(ctx, views, l.table.at(s), at, c, l.scan)?;
            }
            let steps = &mut w.steps[i - 1];
            lk.step_alone(
                ctx,
                positions.clone(),
                l.scan,
                share * d.nmode as f32,
                steps,
            )?;
            let obj = LsqmlVar {
                x: &mut self.deeper[i - 1],
                num: &mut self.obj_sums.num,
            };
            lk.update_alone(ctx, l.scan, obj, steps)?;
        }

        match &mut moved {
            None => {
                for (k, s) in batch.clone().enumerate() {
                    let (stack, chi) = (l.table.at(s), &w.chi[k][0]);
                    self.kernels.psi_to_obj_frames(
                        ctx,
                        views,
                        stack,
                        chi,
                        frame_probe(self.probe, opr.as_ref()),
                        &mut self.obj_sums.num,
                    )?;
                    self.kernels.psi_to_probe(
                        ctx,
                        views,
                        stack,
                        chi,
                        self.obj,
                        &mut self.probe_sums.num,
                    )?;
                }
            }
            Some(m) => {
                for (k, s) in batch.clone().enumerate() {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let (stack, chi) = (l.table.at(s), &w.chi[k][0]);
                    match &mut opr {
                        None => {
                            let probe = Some(ShiftProbe::Num(&mut *m.probe_sums));
                            let sums = PsiSums {
                                obj: Some(&mut self.obj_sums.num),
                                probe,
                            };
                            (self.kernels)
                                .shifted_psi_sums(ctx, &inputs, stack, chi, m.fused, sums)?
                        }
                        Some(o) => {
                            let probe = Some(ShiftProbe::Frames {
                                sums: &mut *m.probe_sums,
                                moved: FrameMoves {
                                    planes: &mut *o.moved,
                                    positions: positions.clone(),
                                },
                            });
                            let sums = PsiSums {
                                obj: Some(&mut self.obj_sums.num),
                                probe,
                            };
                            let at = (&o.own, stack);
                            (self.kernels)
                                .shifted_psi_sums_own(ctx, &inputs, at, chi, m.fused, sums)?
                        }
                    }
                }
                self.kernels
                    .shifted_probe_sums(ctx, d, m.probe_sums, &mut self.probe_sums.num)?;
            }
        }
        lk.dir_obj(ctx, l.scan, &mut self.obj_sums.num)?;
        let nb = positions.len() as f32;
        self.kernels
            .scale_c32(ctx, &mut self.probe_sums.num, self.probe_len, 1.0 / nb)?;
        for (k, s) in batch.enumerate() {
            let at = LsqmlAt {
                x: [&*self.obj, &*self.probe],
                d: [&self.obj_sums.num, &self.probe_sums.num],
            };
            let (stack, chi) = (l.table.at(s), &w.chi[k][0]);
            let c = match (&moved, &opr) {
                (None, None) => LsqmlChi::Unmoved(chi),
                (None, Some(o)) => LsqmlChi::Own {
                    chi,
                    own: o.own.clone(),
                },
                (Some(m), _) => LsqmlChi::Moved {
                    chi,
                    w: FrameWaves::Probes(&m.w[k]),
                },
            };
            lk.dots(ctx, views, stack, at, c, l.scan)?;
            if let Some(o) = &mut opr {
                let dirs = o.dirs(positions.clone());
                lk.frame_dirs(ctx, views, stack, self.obj, chi, dirs)?;
            }
        }
        lk.step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        lk.share(ctx, positions.clone(), l.scan, share)?;
        if let (Some(m), Some(step)) = (moved, step) {
            let solve = position_solve(self.exit.scale);
            let offsets = m.offsets;
            m.kernels
                .step(ctx, positions.clone(), l.order, offsets, solve, step)?;
        }
        if let Some(o) = opr {
            o.step(ctx, (l.order, positions), self.probe, shift)?;
        }
        let obj = LsqmlVar {
            x: &mut *self.obj,
            num: &mut self.obj_sums.num,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            num: &mut self.probe_sums.num,
        };
        lk.update(ctx, l.scan, obj, probe)
    }

    /// [`Self::lsqml_batch_sliced`] on every slice's windows, as
    /// [`Self::lsqml_batch_windows`] reads the one slice's: each slice's windows at the batch's
    /// start, then its sums placed and the windows of its direction for its dot products; the
    /// position sums take the derivatives of the windows of the slice pty-chi corrects.
    fn lsqml_batch_sliced_windows(
        &mut self,
        ctx: &ExecutionContext,
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
        let (views, table) = (l.order.views(), l.table);
        let d = views.dims();
        let (lk, exit, ok) = (l.kernels, self.exit, self.exit.kernels);
        let (moved, w, mut step) = l.reading.split(&self.features)?;
        let lent = PositionLent::lend(moved.map(|m| (m, ())), None, &mut self.features)?;
        let (Some(PositionLent::FullObject(p)), Some(w)) = (lent, w) else {
            return Err(another_reading());
        };
        let (Some(sl), Some(sw)) = (self.slices.as_mut(), l.sliced.as_deref_mut()) else {
            return Err(GpuError::Shape(
                "LSQML state built for another slice count".into(),
            ));
        };
        let mut opr = (l.opr.as_deref_mut())
            .map(|o| {
                o.batch(
                    ctx,
                    self.kernels,
                    l.order,
                    positions.clone(),
                    self.probe,
                    false,
                )
            })
            .transpose()?;
        let last = self.deeper.len();
        let share = (last + 1) as f32;
        let quad = sl.quad;
        let pivot = last.div_ceil(2);
        // The pivot's last: its spectra stay for the derivatives.
        for i in (0..=last).filter(|&i| i != pivot).chain([pivot]) {
            let obj = match i {
                0 => &*self.obj,
                i => &self.deeper[i - 1],
            };
            let at = batch.clone().map(|s| points(views, table.at(s), &*p.pos));
            w.sample(ctx, (p.sampler, p.spectra), at, obj, i)?;
        }
        for (k, s) in batch.clone().enumerate() {
            let (stack, work, inc, wk) =
                (table.at(s), &mut l.work[k], &mut sw.incident[k], &w.win[k]);
            let probe = SliceSource::from(frame_probe(self.probe, opr.as_ref()));
            ok.slice_mul(ctx, &w.views, stack, &wk[0], probe, &mut inc[0])?;
            for i in 1..=last {
                ok.gap(ctx, d, stack, &sl.gaps[i - 1], &mut inc[i - 1], false)?;
                let (src, rest) = inc[i - 1..].split_first_mut().expect("i <= last");
                let dst = rest.first_mut().unwrap_or(&mut *work);
                let src = SliceSource::Waves(FrameWaves::Views(src));
                ok.slice_mul(ctx, &w.views, stack, &wk[i], src, dst)?;
            }
            if let Some(phase) = slice_phase(quad, None) {
                ok.slice_phase(ctx, d, stack, phase, work, false)?;
            }
            ok.rows(ctx, d, stack, work, false)?;
            exit_stage(ctx, exit, d, stack, work, l.band.get_mut(k), false)?;
            if noise == LsqmlNoise::Poisson {
                lk.icalc(ctx, scan, stack, work, &mut l.icalc[k])?;
            }
        }
        if noise == LsqmlNoise::Poisson {
            let icalc: Vec<_> = batch
                .clone()
                .zip(l.icalc.iter())
                .map(|(s, buf)| (table.at(s), buf))
                .collect();
            lk.far(ctx, scan, &icalc, positions.clone(), l.scan)?;
        }
        for (k, s) in batch.clone().enumerate() {
            let (stack, work, chi) = (table.at(s), &mut l.work[k], &mut sw.chi[k]);
            lk.chi(ctx, scan, stack, noise, l.scan, work)?;
            exit_stage(ctx, exit, d, stack, work, l.band.get_mut(k), true)?;
            lk.back(ctx, &w.views, stack, work)?;
            if let Some(phase) = slice_phase(quad, None) {
                ok.slice_phase(ctx, d, stack, phase, work, true)?;
            }
            for i in (0..last).rev() {
                let (dst, rest) = chi[i..].split_first_mut().expect("i < last");
                let src = rest.first().unwrap_or(&*work);
                ok.slice_adjoint(ctx, &w.views, stack, &w.win[k][i + 1], src, dst)?;
                ok.gap(ctx, d, stack, &sl.gaps[i], dst, true)?;
            }
            if let Some(step) = step.as_deref_mut() {
                let derivatives = |out: &mut _| {
                    let at = points(views, stack, p.pos);
                    w.derivatives(ctx, (p.sampler, p.spectra), at, out)
                };
                let c = chi.get(pivot).unwrap_or(&*work);
                let at = (
                    c,
                    SliceSource::Waves(FrameWaves::Views(&sw.incident[k][pivot - 1])),
                );
                (p.kernels).derivative_sums(ctx, views, stack, derivatives, at, step)?;
            }
        }
        for i in (1..=last).rev() {
            for (k, s) in batch.clone().enumerate() {
                let stack = table.at(s);
                let pair = [
                    &sw.incident[k][i - 1],
                    sw.chi[k].get(i).unwrap_or(&l.work[k]),
                ];
                ok.slice_pair(ctx, d, stack, pair, &mut sw.pair)?;
                let at = points(views, stack, p.pos);
                w.place(
                    ctx,
                    p.sampler,
                    at,
                    &sw.pair,
                    FrameProbe::Shared(sl.ones),
                    &mut self.obj_sums.num,
                )?;
            }
            lk.dir_obj(ctx, l.scan, &mut self.obj_sums.num)?;
            p.sampler.spectra(ctx, &self.obj_sums.num, p.spectra)?;
            for (k, s) in batch.clone().enumerate() {
                let stack = table.at(s);
                w.direction(ctx, (p.sampler, p.spectra), points(views, stack, p.pos))?;
                let at = LsqmlAt {
                    x: [&w.win[k][i], &*self.probe],
                    d: [&w.dir, &self.probe_sums.num],
                };
                let c = LsqmlChi::Moved {
                    chi: sw.chi[k].get(i).unwrap_or(&l.work[k]),
                    w: FrameWaves::Views(&sw.incident[k][i - 1]),
                };
                lk.dots(ctx, &w.views, stack, at, c, l.scan)?;
            }
            let steps = &mut sw.steps[i - 1];
            lk.step_alone(
                ctx,
                positions.clone(),
                l.scan,
                share * d.nmode as f32,
                steps,
            )?;
            let obj = LsqmlVar {
                x: &mut self.deeper[i - 1],
                num: &mut self.obj_sums.num,
            };
            lk.update_alone(ctx, l.scan, obj, steps)?;
        }

        for (k, s) in batch.clone().enumerate() {
            let (stack, chi) = (table.at(s), &sw.chi[k][0]);
            (self.kernels).psi_to_probe(
                ctx,
                &w.views,
                stack,
                chi,
                &w.win[k][0],
                &mut self.probe_sums.num,
            )?;
            let at = points(views, stack, p.pos);
            let probe = frame_probe(self.probe, opr.as_ref());
            w.place(ctx, p.sampler, at, chi, probe, &mut self.obj_sums.num)?;
        }
        lk.dir_obj(ctx, l.scan, &mut self.obj_sums.num)?;
        let nb = positions.len() as f32;
        self.kernels
            .scale_c32(ctx, &mut self.probe_sums.num, self.probe_len, 1.0 / nb)?;
        p.sampler.spectra(ctx, &self.obj_sums.num, p.spectra)?;
        for (k, s) in batch.enumerate() {
            let stack = table.at(s);
            w.direction(ctx, (p.sampler, p.spectra), points(views, stack, p.pos))?;
            let at = LsqmlAt {
                x: [&w.win[k][0], &*self.probe],
                d: [&w.dir, &self.probe_sums.num],
            };
            let chi = &sw.chi[k][0];
            let c = match &opr {
                None => LsqmlChi::Unmoved(chi),
                Some(o) => LsqmlChi::Own {
                    chi,
                    own: o.own.clone(),
                },
            };
            lk.dots(ctx, &w.views, stack, at, c, l.scan)?;
            if let Some(o) = &mut opr {
                let dirs = o.dirs(positions.clone());
                lk.frame_dirs(ctx, &w.views, stack, &w.win[k][0], chi, dirs)?;
            }
        }
        lk.step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        lk.share(ctx, positions.clone(), l.scan, share)?;
        if let Some(step) = step {
            let solve = position_solve(self.exit.scale);
            let offsets = p.offsets;
            (p.kernels).step(ctx, positions.clone(), l.order, offsets, solve, step)?;
        }
        if let Some(o) = opr {
            o.step(ctx, (l.order, positions), self.probe, false)?;
        }
        let obj = LsqmlVar {
            x: &mut *self.obj,
            num: &mut self.obj_sums.num,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            num: &mut self.probe_sums.num,
        };
        lk.update(ctx, l.scan, obj, probe)
    }
}
