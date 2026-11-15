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

use crate::gpu::sampling::{Modes, ObjectSampler, StackPoints, WindowBlock, window_views};
use crate::gpu::{
    BhScan, C32, EpochOrder, Error as GpuError, FrameWaves, FusedInputs, Kernels, LSQML_REC,
    LsqmlAt, LsqmlChi, LsqmlKernels, LsqmlNoise, LsqmlScan, LsqmlVar, Model, PositionRule,
    PositionStep, Positions, PsiSums, ShiftProbe, SliceSource, SliceSteps, Solve, StackRef,
    StackTable, UpdateSums, ViewDims, Views,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{
    FeaturesMut, Moved, MovedLent, MovedSampling, MovedScratch, PassesMut, PositionLent,
    moved_inputs, moved_shift, slice_phase,
};
use super::{
    Bufs, Cut, Frames, GpuPtycho, StackRule, Streamed, batch_stacks, plane_bytes, plane_len,
    wave_bytes,
};
use crate::engine::features::{POSITION_LIMIT, POSITION_STEP};
use crate::engine::ptycho::{Algorithm, llk_due};
use crate::engine::{Cost, EngineError, LsqmlParams, LsqmlStep, Sampling, random_batches};

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
        params.validate()?;
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
    /// The batches cut into stacks by the bytes a stack's passes touch, one table entry each.
    stacks: Vec<Frames>,
    table: StackTable,
    /// A batch's row transforms, then exit-wave updates, one per stack of a batch.
    work: Vec<DeviceBuffer<C32>>,
    /// A batch's model intensities, one per stack, for the Poisson `a_far`.
    icalc: Vec<DeviceBuffer<f32>>,
    /// The scan frame at each epoch position, and the views in that order.
    order: EpochOrder,
    scan: LsqmlScan,
    /// What the views read of the object.
    reading: Reading,
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
    /// A stack's object sums on its windows, which the sampler places on the object.
    sums: UpdateSums,
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
            sums: UpdateSums::zeroed(stream, len)?,
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
        probe: &DeviceBuffer<C32>,
        num: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        crate::gpu::zero(ctx, &mut self.sums.den)?;
        crate::gpu::zero(ctx, &mut self.sums.num)?;
        (self.kernels).psi_to_obj(ctx, &self.views, at.stack, psi, probe, &mut self.sums)?;
        sampler.place(ctx, at, &self.sums.num, 0, num)
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
    /// of the slice whose spectra `spectra` holds, into `step`'s slopes.
    fn derivatives(
        &self,
        ctx: &ExecutionContext,
        (sampler, spectra): (&mut ObjectSampler, &DeviceBuffer<C32>),
        at: StackPoints<'_>,
        step: &mut PositionStep,
    ) -> Result<(), GpuError> {
        for (block, order) in [[1, 0], [0, 1]].into_iter().enumerate() {
            let block = WindowBlock {
                slice: 0,
                order,
                modes: Modes::First,
                block: block as u32,
            };
            sampler.windows(ctx, at, spectra, block, step.slopes_mut())?;
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
    icalc: &'a mut [DeviceBuffer<f32>],
    order: &'a EpochOrder,
    scan: &'a mut LsqmlScan,
    reading: &'a mut Reading,
    sliced: Option<&'a mut SliceBufs>,
}

impl LsqmlState {
    /// For `views` in batches of `size` frames, each cut into stacks by `rule`; under the
    /// position model, `scratch` holds the frames per stack of the shared scratch and the
    /// device's sampling: under [`crate::engine::Sampling::Patch`] the moved probe's sums
    /// borrow its shift planes, which caps the state's stacks. `deeper` counts the object's
    /// slices past the first. The epoch order is the scan's.
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        size: u32,
        rule: StackRule,
        scratch: Option<MovedScratch>,
        deeper: usize,
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        // work: the forward pass (0), the exit wave, `chi`, the adjoint exit and back pass (4),
        // then `psi_to_obj` and `psi_to_probe` (5, 6); icalc, which `far` reads batch-wide,
        // streams. Under the position: work from `shifted_rows_fwd` (0, with the moved probes
        // to `shift_w`) to the back pass (5), `shifted_psi_sums` from 6, then `dots` and the
        // position sums (11, 12) read work and the moved probes again. The multislice loops
        // hand on one wave per pass boundary, two under the position: no more than these.
        // Under windows, every slice's: a loop of sampler calls (0 to 2) writes the windows;
        // the waves' loop (3 to 7) hands on one wave as above. Then work from `psi_to_probe`
        // (8) to `psi_to_obj` (18): between them the derivatives over `y` and `x` (two sampler
        // calls, 9 to 14) for the position sums (15), the window sums zeroed (16, 17); then
        // `place` zeroes its accumulator (19), runs a sampler call (20 to 22), transforms the
        // accumulator and adds it (23, 24); a slice past the first takes its pair (8) instead.
        // A sampler call (25 to 27) writes the direction's windows for `dots` (28).
        let wave = wave_bytes(dims);
        let streamed = match scratch {
            None => vec![Streamed::new(wave, 0..=4), Streamed::new(wave, 5..=6)],
            Some(MovedScratch {
                sampling: MovedSampling::Patch,
                ..
            }) => {
                let probes = plane_bytes::<C32>(dims, dims.nprobe);
                let mut s = vec![
                    Streamed::new(wave, 0..=5),
                    Streamed::new(probes, 0..=1),
                    Streamed::new(wave, 11..=12),
                    Streamed::new(probes, 11..=12),
                ];
                s.extend(Streamed::shifted_psi_sums(dims, 6));
                s
            }
            Some(MovedScratch {
                sampling: MovedSampling::FullObject { held, scratch, acc },
                ..
            }) => {
                let plane1 = plane_bytes::<C32>(dims, 1);
                let win = plane_bytes::<C32>(dims, dims.nobj);
                let den = plane_bytes::<f32>(dims, dims.nobj);
                [
                    Streamed::chunked(held, 0..=2, &[(win, 2)], &[]),
                    vec![Streamed::new(wave, 3..=7), Streamed::new(wave, 8..=18)],
                    Streamed::chunked(held, 9..=11, &[(plane1, 15)], &[]),
                    Streamed::chunked(held, 12..=14, &[(plane1, 15)], &[]),
                    vec![Streamed::new(den, 16..=18), Streamed::fixed(acc, 19..=24)],
                    Streamed::chunked(scratch, 20..=22, &[], &[(win, 17)]),
                    Streamed::chunked(held, 25..=27, &[(win, 28)], &[]),
                ]
                .concat()
            }
        };
        let frames = rule.frames(&streamed)?;
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
            icalc: (0..slots)
                .map(|_| DeviceBuffer::zeroed(stream, icalc_len))
                .collect::<Result<_, _>>()?,
            order: EpochOrder::new(kernels, stream, views)?,
            scan: LsqmlScan::zeroed(stream, dims)?,
            reading,
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
            reading: &mut self.reading,
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
                        let offsets = &*s.offsets.values;
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
        let chi = |st: &LsqmlScan, stack: StackRef<'_>, work: &mut DeviceBuffer<C32>| {
            lk.chi(ctx, scan, stack, noise, st, work)?;
            exit.kernels
                .exit(ctx, views.dims(), stack, exit.stage, work, true)?;
            lk.back(ctx, views, stack, work)
        };
        for (k, s) in batch.clone().enumerate() {
            let (stack, work) = (l.table.at(s), &mut l.work[k]);
            match &mut moved {
                None => l
                    .kernels
                    .forward(ctx, views, stack, self.obj, self.probe, work)?,
                Some(m) => {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let w = &mut m.w[k];
                    // Its row pass is the stack's forward pass: `exit` goes on from `work`.
                    self.kernels
                        .shifted_rows_fwd(ctx, &inputs, stack, w, work)?;
                    m.kernels.shift_w(ctx, views.dims(), stack, w)?;
                }
            }
            exit.kernels
                .exit(ctx, views.dims(), stack, exit.stage, work, false)?;
            match noise {
                LsqmlNoise::Gaussian => chi(l.scan, stack, work)?,
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
                chi(l.scan, l.table.at(s), &mut l.work[k])?;
            }
        }

        match &mut moved {
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
            Some(m) => {
                for (k, s) in batch.clone().enumerate() {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let sums = PsiSums {
                        obj: Some(&mut *self.obj_sums),
                        probe: Some(ShiftProbe::Sums(&mut *m.probe_sums)),
                    };
                    let stack = l.table.at(s);
                    self.kernels
                        .shifted_psi_sums(ctx, &inputs, stack, &l.work[k], m.fused, sums)?;
                }
                self.kernels.shifted_probe_sums(
                    ctx,
                    views.dims(),
                    m.probe_sums,
                    self.probe_sums,
                )?;
            }
        }
        l.kernels.dir_obj(ctx, l.scan, self.obj_sums)?;
        let nb = positions.len() as f32;
        self.kernels
            .scale_c32(ctx, &mut self.probe_sums.num, self.probe_len, 1.0 / nb)?;
        for (k, s) in batch.enumerate() {
            let (stack, chi) = (l.table.at(s), &l.work[k]);
            let at = LsqmlAt {
                x: [&*self.obj, &*self.probe],
                d: [&self.obj_sums.num, &self.probe_sums.num],
            };
            let c = match &moved {
                None => LsqmlChi::Unmoved(chi),
                Some(m) => LsqmlChi::Moved {
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
        }
        l.kernels
            .step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        // Before the update, as the batch's spectra borrow the probe; the step reads neither.
        if let (Some(m), Some(step)) = (moved, step) {
            let solve = position_solve(self.exit.scale);
            let offsets = m.offsets.values;
            m.kernels
                .step(ctx, positions, l.order, offsets, solve, step)?;
        }
        let obj = LsqmlVar {
            x: &mut *self.obj,
            sums: &mut *self.obj_sums,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            sums: &mut *self.probe_sums,
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
        let at = batch.clone().map(|s| points(views, table.at(s), &*p.pos));
        w.sample(ctx, (p.sampler, p.spectra), at, self.obj, 0)?;
        let wv = &w.views;
        let chi = |st: &LsqmlScan, stack: StackRef<'_>, work: &mut DeviceBuffer<C32>| {
            lk.chi(ctx, scan, stack, noise, st, work)?;
            exit.kernels.exit(ctx, d, stack, exit.stage, work, true)?;
            lk.back(ctx, wv, stack, work)
        };
        for (k, s) in batch.clone().enumerate() {
            let (stack, work) = (table.at(s), &mut l.work[k]);
            lk.forward(ctx, wv, stack, &w.win[k][0], self.probe, work)?;
            exit.kernels.exit(ctx, d, stack, exit.stage, work, false)?;
            match noise {
                LsqmlNoise::Gaussian => chi(l.scan, stack, work)?,
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
                chi(l.scan, table.at(s), &mut l.work[k])?;
            }
        }
        for (k, s) in batch.clone().enumerate() {
            let (stack, work) = (table.at(s), &l.work[k]);
            (self.kernels).psi_to_probe(
                ctx,
                &w.views,
                stack,
                work,
                &w.win[k][0],
                self.probe_sums,
            )?;
            if let Some(step) = step.as_deref_mut() {
                // The slopes of the windows the batch started from.
                w.derivatives(
                    ctx,
                    (p.sampler, p.spectra),
                    points(views, stack, p.pos),
                    step,
                )?;
                let at = (work, SliceSource::Probe(self.probe));
                (p.kernels).derivative_sums(ctx, views, stack, at, step)?;
            }
            let at = points(views, stack, p.pos);
            w.place(ctx, p.sampler, at, work, self.probe, &mut self.obj_sums.num)?;
        }
        lk.dir_obj(ctx, l.scan, self.obj_sums)?;
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
            let chi = LsqmlChi::Unmoved(&l.work[k]);
            lk.dots(ctx, &w.views, stack, at, chi, l.scan)?;
        }
        lk.step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        if let Some(step) = step {
            let solve = position_solve(self.exit.scale);
            let offsets = p.offsets.values;
            (p.kernels).step(ctx, positions, l.order, offsets, solve, step)?;
        }
        let obj = LsqmlVar {
            x: &mut *self.obj,
            sums: &mut *self.obj_sums,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            sums: &mut *self.probe_sums,
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
                    let probe = SliceSource::Probe(self.probe);
                    ok.slice_mul(ctx, views, stack, self.obj, probe, &mut inc[0])?;
                    None
                }
                Some(m) => {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let pw = &mut m.w[k];
                    self.kernels.shifted_probe(ctx, &inputs, stack, pw)?;
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
            ok.exit(ctx, d, stack, exit.stage, work, false)?;
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
            ok.exit(ctx, d, stack, exit.stage, work, true)?;
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
                self.kernels
                    .psi_to_obj(ctx, views, stack, &w.pair, sl.ones, self.obj_sums)?;
            }
            lk.dir_obj(ctx, l.scan, self.obj_sums)?;
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
                sums: &mut *self.obj_sums,
            };
            lk.update_alone(ctx, l.scan, obj, steps)?;
        }

        match &mut moved {
            None => {
                for (k, s) in batch.clone().enumerate() {
                    let (stack, chi) = (l.table.at(s), &w.chi[k][0]);
                    self.kernels
                        .psi_to_obj(ctx, views, stack, chi, self.probe, self.obj_sums)?;
                    self.kernels
                        .psi_to_probe(ctx, views, stack, chi, self.obj, self.probe_sums)?;
                }
            }
            Some(m) => {
                for (k, s) in batch.clone().enumerate() {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let sums = PsiSums {
                        obj: Some(&mut *self.obj_sums),
                        probe: Some(ShiftProbe::Sums(&mut *m.probe_sums)),
                    };
                    let (stack, chi) = (l.table.at(s), &w.chi[k][0]);
                    self.kernels
                        .shifted_psi_sums(ctx, &inputs, stack, chi, m.fused, sums)?;
                }
                self.kernels
                    .shifted_probe_sums(ctx, d, m.probe_sums, self.probe_sums)?;
            }
        }
        lk.dir_obj(ctx, l.scan, self.obj_sums)?;
        let nb = positions.len() as f32;
        self.kernels
            .scale_c32(ctx, &mut self.probe_sums.num, self.probe_len, 1.0 / nb)?;
        for (k, s) in batch.enumerate() {
            let at = LsqmlAt {
                x: [&*self.obj, &*self.probe],
                d: [&self.obj_sums.num, &self.probe_sums.num],
            };
            let chi = &w.chi[k][0];
            let c = match &moved {
                None => LsqmlChi::Unmoved(chi),
                Some(m) => LsqmlChi::Moved {
                    chi,
                    w: FrameWaves::Probes(&m.w[k]),
                },
            };
            lk.dots(ctx, views, l.table.at(s), at, c, l.scan)?;
        }
        lk.step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        lk.share(ctx, positions.clone(), l.scan, share)?;
        if let (Some(m), Some(step)) = (moved, step) {
            let solve = position_solve(self.exit.scale);
            let offsets = m.offsets.values;
            m.kernels
                .step(ctx, positions, l.order, offsets, solve, step)?;
        }
        let obj = LsqmlVar {
            x: &mut *self.obj,
            sums: &mut *self.obj_sums,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            sums: &mut *self.probe_sums,
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
            let probe = SliceSource::Probe(self.probe);
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
            ok.exit(ctx, d, stack, exit.stage, work, false)?;
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
            ok.exit(ctx, d, stack, exit.stage, work, true)?;
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
                w.derivatives(
                    ctx,
                    (p.sampler, p.spectra),
                    points(views, stack, p.pos),
                    step,
                )?;
                let c = chi.get(pivot).unwrap_or(&*work);
                let at = (
                    c,
                    SliceSource::Waves(FrameWaves::Views(&sw.incident[k][pivot - 1])),
                );
                (p.kernels).derivative_sums(ctx, views, stack, at, step)?;
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
                    sl.ones,
                    &mut self.obj_sums.num,
                )?;
            }
            lk.dir_obj(ctx, l.scan, self.obj_sums)?;
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
                sums: &mut *self.obj_sums,
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
                self.probe_sums,
            )?;
            let at = points(views, stack, p.pos);
            w.place(ctx, p.sampler, at, chi, self.probe, &mut self.obj_sums.num)?;
        }
        lk.dir_obj(ctx, l.scan, self.obj_sums)?;
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
            let chi = LsqmlChi::Unmoved(&sw.chi[k][0]);
            lk.dots(ctx, &w.views, stack, at, chi, l.scan)?;
        }
        lk.step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        lk.share(ctx, positions.clone(), l.scan, share)?;
        if let Some(step) = step {
            let solve = position_solve(self.exit.scale);
            let offsets = p.offsets.values;
            (p.kernels).step(ctx, positions, l.order, offsets, solve, step)?;
        }
        let obj = LsqmlVar {
            x: &mut *self.obj,
            sums: &mut *self.obj_sums,
        };
        let probe = LsqmlVar {
            x: &mut *self.probe,
            sums: &mut *self.probe_sums,
        };
        lk.update(ctx, l.scan, obj, probe)
    }
}
