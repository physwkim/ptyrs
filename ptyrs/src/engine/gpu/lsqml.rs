//! pty-chi LSQML on the GPU ([`crate::engine::Lsqml`]) through the device's propagator, incoherent
//! probe modes and no probe variation, in the call sequence of `crate::gpu::LsqmlKernels`:
//! the exit waves through the fused row passes and the propagator's stage around LSQML's
//! column pass, the step sizes and every frame's `a_far` kept on the device. Under the position model the probe moves by
//! each frame's offsets, and each batch steps its frames' offsets after its passes.
//!
//! A multislice object ([`crate::engine::SliceWaves`]) keeps, per stack of a batch, the wave
//! entering every slice past the first and every slice's exit-wave update: slice 0 then
//! steps with the probe as the one slice does, from its own update, and every other slice
//! alone, from the wave that entered it.

use std::ops::Range;
use std::sync::Arc;

use crate::gpu::{
    BhScan, C32, EpochOrder, Error as GpuError, FrameWaves, FusedInputs, Kernels, LSQML_REC,
    LsqmlAt, LsqmlChi, LsqmlKernels, LsqmlNoise, LsqmlScan, LsqmlVar, Model, PositionRule, PsiSums,
    ShiftProbe, SliceSource, SliceSteps, Solve, StackRef, StackTable, Views,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{
    Moved, MovedLent, MovedScratch, PassesMut, moved_inputs, moved_shift, slice_phase,
};
use super::{
    Bufs, Cut, Frames, GpuPtycho, StackRule, Streamed, batch_stacks, plane_bytes, plane_len,
    wave_bytes,
};
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
    /// The position model's state, if the device models positions.
    moved: Option<Moved>,
    /// A multislice object's waves, if the device's has more than one slice.
    sliced: Option<SliceBufs>,
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
    moved: Option<&'a mut Moved>,
    sliced: Option<&'a mut SliceBufs>,
}

impl LsqmlState {
    /// For `views` in batches of `size` frames, each cut into stacks by `rule`; under the
    /// position model, `scratch` holds the frames per stack of the shared scratch, whose shift
    /// planes its sums borrow, which caps its stacks, and the device's sampling; `deeper`
    /// counts the object's slices past the first. The epoch order is the scan's.
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
        let wave = wave_bytes(dims);
        let streamed = match scratch {
            None => vec![Streamed::new(wave, 0..=4), Streamed::new(wave, 5..=6)],
            Some(_) => {
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
        };
        let frames = rule.frames(&streamed)?;
        let frames = scratch.map_or(frames, |s| frames.min(s.frames));
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
        let moved = match scratch {
            Some(s) => Some(Moved::new(
                stream,
                kernels,
                dims,
                nf_max,
                slots,
                s.sampling.into(),
            )?),
            None => None,
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
            moved,
            sliced,
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
        match pre {
            Pre::Init => {
                l.kernels.start(ctx, self.counts, l.scan)?;
                l.kernels
                    .precondition(ctx, self.views, self.probe, l.scan, true, k)?;
            }
            Pre::Update => {
                l.kernels
                    .precondition(ctx, self.views, self.probe, l.scan, false, k)?;
            }
            Pre::Keep => {}
        }
        // Each frame is in one batch, stepped after its batch's passes: the offsets the
        // epoch starts from serve every pass.
        let moved = l.moved.as_deref_mut().map(|m| (m, ()));
        if let Some(mut m) = MovedLent::lend(moved, passes.fused(), &mut self.features)? {
            m.gather(ctx, l.order)?;
        }
        let noise = match params.cost {
            Cost::Amplitude => LsqmlNoise::Gaussian,
            Cost::Poisson => LsqmlNoise::Poisson,
        };
        for b in 0..l.batches.len() {
            match l.sliced {
                None => self.lsqml_batch(ctx, passes, l, b, noise)?,
                Some(_) => self.lsqml_batch_sliced(ctx, passes, l, b, noise)?,
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
        let moved = l.moved.as_deref_mut().map(|m| (m, ()));
        let moved = MovedLent::lend(moved, passes.fused(), &mut self.features)?;
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
            if let Some(m) = &mut moved {
                // The slopes of the object the batch started from.
                let at = (chi, FrameWaves::Probes(&m.w[k]));
                m.kernels.sums(ctx, views, stack, self.obj, at, m.pos)?;
            }
        }
        l.kernels
            .step(ctx, positions.clone(), l.scan, self.exit.scale)?;
        // Before the update, as the batch's spectra borrow the probe; the step reads neither.
        if let Some(m) = moved {
            let solve = Solve {
                rule: PositionRule {
                    step: POSITION_STEP as f32,
                    limit: POSITION_LIMIT as f32,
                },
                scale: self.exit.scale,
            };
            m.kernels
                .step(ctx, positions, l.order, m.offsets, solve, m.pos)?;
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
        let moved = l.moved.as_deref_mut().map(|m| (m, ()));
        let moved = MovedLent::lend(moved, passes.fused(), &mut self.features)?;
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
        let last = sl.deeper.len();
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
                ok.slice_mul(ctx, views, stack, &sl.deeper[i - 1], src, dst)?;
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
                ok.slice_adjoint(ctx, views, stack, &sl.deeper[i], src, dst)?;
                ok.gap(ctx, d, stack, &sl.gaps[i], dst, true)?;
            }
            if let Some(m) = &mut moved {
                let c = chi.get(pivot).unwrap_or(&*work);
                let at = (c, FrameWaves::Views(&w.incident[k][pivot - 1]));
                m.kernels
                    .sums(ctx, views, stack, &sl.deeper[pivot - 1], at, m.pos)?;
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
                    x: [&sl.deeper[i - 1], &*self.probe],
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
                x: &mut sl.deeper[i - 1],
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
        if let Some(m) = moved {
            let solve = Solve {
                rule: PositionRule {
                    step: POSITION_STEP as f32,
                    limit: POSITION_LIMIT as f32,
                },
                scale: self.exit.scale,
            };
            m.kernels
                .step(ctx, positions, l.order, m.offsets, solve, m.pos)?;
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
