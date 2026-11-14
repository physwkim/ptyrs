//! The device state a captured cycle addresses, and the graph captured from it, under one
//! owner. Nothing outside this module can reach a field of [`Device`] or of a stack's
//! buffers, and nothing here replaces one after [`Device::new`] but
//! [`Device::set_features`], which drops every graph before and recaptures the LLK pass's
//! after: a graph replays on the buffers the eager cycle would use, and other buffers mean
//! another `Device`, or other features, which start with no graph. The device also owns the
//! LLK ring and its count of records: a record is counted once per enqueued run of a cycle
//! or pass that ended it, as the [`Ended`] its body returned shows, eagerly or kept from the
//! capture of its graph.

use std::ops::Range;
use std::sync::Arc;

use crate::fft::Plan2d;
use crate::gpu::{
    BH_STEP, BackgroundSums, BhKernels, C32, CG_BLOCKS, Ended, EpochOrder, Error as GpuError,
    ExitPass, ExitStage, FarSums, FusedBufs, FusedInputs, GammaStacks, GradKernels, Kernels,
    LlkPartial, LlkRing, LlkScratch, LlkTotals, Model, OpticsKernels, PositionKernels,
    PositionPartial, Positions, PowerMax, ProbeSpectrum, PsiSums, RingDevice, RpieKernels,
    ScalePartial, Shift, ShiftSums, ShiftedProbeSums, SlicePhase, SliceSource, SpectrumBuf,
    StackRef, StackTable, UpdateSums, ViewAddr, ViewDims, Views,
};
use crate::tile::TileDots;
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use super::bh::BhMoved;
use super::lsqml::{LsqmlBufs, LsqmlState, Pre};
use super::ml::MlMoved;
use super::rpie::RpieMoved;
use super::{
    Cut, Epoch, Frames, GpuOptions, StackRule, Streamed, Tag, batch_stacks, plane_bytes, plane_len,
    to_c32, to_u32, wave_bytes,
};
use crate::engine::features::{POSITION_LIMIT, POSITION_STEP};
use crate::engine::ptycho::Algorithm;
use crate::engine::{
    ApParams, BhParams, BhStep, DmParams, EngineError, Features, LsqmlParams, LsqmlStep, MlParams,
    PlaneStage, RaarParams, RpieParams,
};

/// What [`Device::new`] uploads: the propagator's stage, the views' addresses, the counts
/// and the starting arrays.
pub(super) struct Uploads<'a> {
    pub stage: PlaneStage<'a>,
    /// The propagator's intensity scale.
    pub scale: f32,
    pub addrs: &'a [ViewAddr],
    pub counts: &'a [u16],
    /// The object, of a multislice object its first slice.
    pub obj: &'a [C32],
    pub probe: &'a [C32],
    /// The rest of a multislice object.
    pub slices: Option<SliceUploads>,
}

/// A multislice object past its first slice.
pub(super) struct SliceUploads {
    /// The slices after the first, each shaped as it.
    pub deeper: Vec<Vec<C32>>,
    /// Gap `s`'s transfer function, from slice `s` to `s + 1`: one per deeper slice.
    pub gaps: Vec<Vec<C32>>,
    /// The last slice's plane phase, radians per squared pixel.
    pub quad: f32,
}

/// A multislice object's device state past its first slice, which is [`Device::obj`]; only
/// LSQML and the LLK pass run it, and [`Device::one_slice`] gates every other pass.
struct Slices {
    deeper: Vec<DeviceBuffer<C32>>,
    gaps: Vec<DeviceBuffer<C32>>,
    /// A probe of ones `[nprobe, ny, nx]`: [`Kernels::psi_to_obj`] over it is a slice's
    /// adjoint of waves already times the conjugate of what entered it.
    ones: DeviceBuffer<C32>,
    quad: f32,
    /// Moves the probe the LLK pass sends through the slices.
    position: PositionKernels,
}

/// [`Slices`] lent to a cycle, with [`Shared::slice_waves`].
pub(super) struct SlicesMut<'a> {
    pub deeper: &'a mut [DeviceBuffer<C32>],
    pub gaps: &'a [DeviceBuffer<C32>],
    pub ones: &'a DeviceBuffer<C32>,
    pub quad: f32,
    position: &'a PositionKernels,
    waves: &'a mut [DeviceBuffer<C32>; 2],
}

/// The last slice's plane phase of `quad` radians per squared pixel, moved with the probe by
/// `offsets`; none of no phase.
pub(super) fn slice_phase(
    quad: f32,
    offsets: Option<&DeviceBuffer<f32>>,
) -> Option<SlicePhase<'_>> {
    (quad != 0.0).then_some(SlicePhase { quad, offsets })
}

impl SlicesMut<'_> {
    /// `stack`'s waves leaving the last slice, its plane phase on them, in a scratch stack's
    /// buffer: of `inputs` the views, the first slice as its object, and the probe, moved
    /// under a [`Model::shift`] by its offsets.
    pub fn exit_waves(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        optics: &OpticsKernels,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
    ) -> Result<&mut DeviceBuffer<C32>, GpuError> {
        let (views, d) = (inputs.views, inputs.views.dims());
        let [a, b] = &mut *self.waves;
        let offsets = match inputs.model.shift {
            None => {
                let probe = SliceSource::Probe(inputs.probe);
                optics.slice_mul(ctx, views, stack, inputs.obj, probe, a)?;
                None
            }
            Some(shift) => {
                kernels.shifted_probe(ctx, inputs, stack, b)?;
                self.position.shift_w(ctx, d, stack, b)?;
                let probe = SliceSource::Waves(b);
                optics.slice_mul(ctx, views, stack, inputs.obj, probe, a)?;
                Some(shift.offsets)
            }
        };
        let (mut src, mut dst) = (a, b);
        for (obj, gap) in self.deeper.iter().zip(self.gaps) {
            optics.gap(ctx, d, stack, gap, src, false)?;
            optics.slice_mul(ctx, views, stack, obj, SliceSource::Waves(src), dst)?;
            std::mem::swap(&mut src, &mut dst);
        }
        if let Some(p) = slice_phase(self.quad, offsets) {
            optics.slice_phase(ctx, d, stack, p, src, false)?;
        }
        Ok(src)
    }
}

/// Every buffer, module and plan a cycle launches on, and the graphs captured from them.
pub(super) struct Device {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    rpie: RpieKernels,
    grad: GradKernels,
    bh: BhKernels,
    optics: OpticsKernels,
    /// The propagator's stage ahead of the transform every pass takes.
    exit: ExitStage,
    /// The propagator's intensity scale.
    scale: f32,
    /// BH's dot products, compiled by [`Device::new`] so no capture compiles a Tile kernel.
    tile: TileDots,
    views: Views,
    counts: DeviceBuffer<u16>,
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
    /// A multislice object's slices past `obj`; none of one slice.
    slices: Option<Slices>,
    obj_sums: UpdateSums,
    probe_sums: UpdateSums,
    obj_len: u32,
    probe_len: u32,
    /// [`GpuOptions::stack_size`] on this device's L2: every cut below is its.
    rule: StackRule,
    shared: Shared,
    /// DM's or RAAR's carried stacks: built by the first run of either and dropped by
    /// [`Self::steady_with`] once no run, whose graphs bind them, is left.
    carried: Option<Box<CarriedBufs>>,
    norms: NormBufs,
    ring: LlkRing<Tag>,
    use_graph: bool,
    /// The epochs run since the parameters last changed, all of one [`Epoch::run`].
    runs: Vec<EpochRun>,
    /// With graphs, the LLK pass's graph.
    llk_graph: Option<Captured>,
    /// ML's state, built by the first ML call and replaced only for another position model.
    ml: Option<Box<MlState>>,
    /// BH's state, built by the first BH call and replaced only for another batch size.
    bh_state: Option<Box<BhState>>,
    /// LSQML's state, built by the first LSQML call and replaced only for another batch size.
    lsqml: Option<Box<LsqmlState>>,
    features: FeatureBufs,
}

/// The features' device state as [`Device::set_features`] last set it: each modelled
/// feature's values and the far sums its step reads.
#[derive(Default)]
struct FeatureBufs {
    scale: Option<ScaleBufs>,
    background: Option<BackgroundBufs>,
    position: Option<PositionBufs>,
}

/// A per-frame intensity scale `[nframe]` and its far sums `[nframe]`.
struct ScaleBufs {
    values: DeviceBuffer<f32>,
    sums: DeviceBuffer<ScalePartial>,
}

/// A detector background `[ny, nx]` in the counts' order, its far sums, each detector
/// pixel's unmasked frames and the background's photons over them.
struct BackgroundBufs {
    values: DeviceBuffer<f32>,
    sums: BackgroundSums,
    /// A copy of the ring's [`RingDevice::pixel_frames`], final once [`Device::new`] has
    /// counted every frame.
    nobs: DeviceBuffer<u32>,
    /// `[1]`, written by [`FeaturesMut::background_photons`].
    photons: DeviceBuffer<f32>,
}

/// Per-frame offsets `[nframe, 2]` of the position model, the spectra of the probe they
/// move, the shifted probe's update sums and each frame's position sums `[nframe]`.
struct PositionBufs {
    offsets: DeviceBuffer<f32>,
    spectrum: SpectrumBuf,
    probe_sums: ShiftedProbeSums,
    num: DeviceBuffer<PositionPartial>,
    den: DeviceBuffer<PositionPartial>,
}

/// [`FeatureBufs`] lent to a cycle.
pub(super) struct FeaturesMut<'a> {
    scale: Option<ScaleMut<'a>>,
    background: Option<BackgroundMut<'a>>,
    position: Option<PositionMut<'a>>,
}

/// [`ScaleBufs`] lent to a cycle.
struct ScaleMut<'a> {
    values: &'a mut DeviceBuffer<f32>,
    sums: &'a mut DeviceBuffer<ScalePartial>,
}

/// [`BackgroundBufs`] lent to a cycle.
struct BackgroundMut<'a> {
    values: &'a mut DeviceBuffer<f32>,
    sums: &'a mut BackgroundSums,
    nobs: &'a DeviceBuffer<u32>,
    photons: &'a mut DeviceBuffer<f32>,
}

/// [`PositionBufs`] lent to a cycle.
struct PositionMut<'a> {
    offsets: &'a mut DeviceBuffer<f32>,
    spectrum: &'a mut SpectrumBuf,
    probe_sums: &'a mut ShiftedProbeSums,
    num: &'a mut DeviceBuffer<PositionPartial>,
    den: &'a mut DeviceBuffer<PositionPartial>,
}

/// The position model's state a cycle that runs its own passes owns: per batch slot the moved
/// probes `[nf_max * nprobe, ny, nx]`, the offsets by epoch position and their kernels.
pub(super) struct Moved {
    w: Vec<DeviceBuffer<C32>>,
    pos: Positions,
    kernels: PositionKernels,
}

impl Moved {
    /// For frames of `dims` in `slots` stacks per batch of up to `nf_max` frames each.
    pub fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        nf_max: u32,
        slots: usize,
    ) -> Result<Self, EngineError> {
        let wlen = nf_max as usize * dims.nprobe as usize * plane_len(dims);
        Ok(Self {
            w: (0..slots)
                .map(|_| DeviceBuffer::zeroed(stream, wlen))
                .collect::<Result<_, _>>()?,
            pos: Positions::zeroed(stream, dims, nf_max)?,
            kernels: PositionKernels::load(kernels)?,
        })
    }
}

/// A cycle's [`Moved`] lent with the device's position buffers: the offsets by frame, the
/// probe spectra and the shifted probe's sums; `own` the cycle's other buffers of the model,
/// `fused` the scratch of the shifted sums over its stacks.
pub(super) struct MovedLent<'m, X> {
    pub own: X,
    fused: &'m mut FusedBufs,
    w: &'m mut [DeviceBuffer<C32>],
    pos: &'m mut Positions,
    kernels: &'m PositionKernels,
    offsets: &'m mut DeviceBuffer<f32>,
    spectrum: &'m mut SpectrumBuf,
    probe_sums: &'m mut ShiftedProbeSums,
}

impl<'m, X> MovedLent<'m, X> {
    /// `moved` with `fused` and `features`' position buffers: none without a position, an
    /// error when they disagree, as for a cycle's state built for another position model.
    pub fn lend(
        moved: Option<(&'m mut Moved, X)>,
        fused: Option<&'m mut FusedBufs>,
        features: &'m mut FeaturesMut<'_>,
    ) -> Result<Option<Self>, GpuError> {
        match (moved, features.position.as_mut(), fused) {
            (None, None, _) => Ok(None),
            (Some((m, own)), Some(p), Some(fused)) => Ok(Some(Self {
                own,
                fused,
                w: &mut m.w,
                pos: &mut m.pos,
                kernels: &m.kernels,
                offsets: &mut *p.offsets,
                spectrum: &mut *p.spectrum,
                probe_sums: &mut *p.probe_sums,
            })),
            _ => Err(GpuError::Shape(
                "cycle state built for another position model".into(),
            )),
        }
    }

    /// The offsets by epoch position from the device's, through `order`, the epoch's.
    pub fn gather(&mut self, ctx: &ExecutionContext, order: &EpochOrder) -> Result<(), GpuError> {
        self.kernels.gather(ctx, order, self.offsets, self.pos)
    }

    /// The lend for one batch, with the spectra of `probe` it starts from (two launches).
    pub fn batch(
        self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        dims: &ViewDims,
        probe: &'m DeviceBuffer<C32>,
    ) -> Result<MovedBatch<'m, X>, GpuError> {
        Ok(MovedBatch {
            spectrum: kernels.probe_spectrum(ctx, dims, probe, self.spectrum)?,
            own: self.own,
            fused: self.fused,
            w: self.w,
            pos: self.pos,
            kernels: self.kernels,
            offsets: self.offsets,
            probe_sums: self.probe_sums,
        })
    }
}

/// [`MovedLent`] for one batch, with the spectra of the batch-start probe, which they borrow.
pub(super) struct MovedBatch<'m, X> {
    pub own: X,
    pub fused: &'m mut FusedBufs,
    /// The moved probes, one per stack of the batch.
    pub w: &'m mut [DeviceBuffer<C32>],
    pub pos: &'m mut Positions,
    pub kernels: &'m PositionKernels,
    /// By frame: the device's offsets, which the batch steps.
    pub offsets: &'m mut DeviceBuffer<f32>,
    pub probe_sums: &'m mut ShiftedProbeSums,
    pub spectrum: ProbeSpectrum<'m>,
}

/// A batch's model of the probe, from its [`MovedBatch::pos`] and [`MovedBatch::spectrum`]
/// alone, so its other fields stay free to borrow: the spectra moved by the offsets by epoch
/// position.
pub(super) fn moved_shift<'a>(pos: &'a Positions, spectrum: ProbeSpectrum<'a>) -> Shift<'a> {
    Shift {
        offsets: pos.offsets(),
        spectrum,
    }
}

/// `base` under the position model `shift`.
pub(super) fn moved_inputs<'a>(base: &FusedInputs<'a>, shift: Shift<'a>) -> FusedInputs<'a> {
    FusedInputs {
        model: Model {
            shift: Some(shift),
            ..Model::default()
        },
        ..*base
    }
}

/// The features lent to the passes of one cycle body by [`FeaturesMut::lend`]: the model
/// every pass that compares with the counts reads, and the sums the passes write.
pub(super) struct Modelled<'b> {
    pub model: Model<'b>,
    scale: Option<&'b mut DeviceBuffer<ScalePartial>>,
    background: Option<&'b mut BackgroundSums>,
    position: Option<PositionSums<'b>>,
}

/// The sums of the position model, lent by [`Modelled`].
struct PositionSums<'b> {
    probe: &'b mut ShiftedProbeSums,
    num: &'b mut DeviceBuffer<PositionPartial>,
    den: &'b mut DeviceBuffer<PositionPartial>,
}

impl Modelled<'_> {
    /// The far sums of `llk` and of every stepped feature.
    pub fn far<'c>(&'c mut self, llk: Option<&'c mut DeviceBuffer<LlkPartial>>) -> FarSums<'c> {
        FarSums {
            llk,
            scale: self.scale.as_deref_mut(),
            background: self.background.as_deref_mut(),
        }
    }

    /// A fused pass's sums of the position model: with `probe` the shifted probe's update
    /// sums, with `position` each frame's position sums; none without a position.
    pub fn shift_sums(&mut self, probe: bool, position: bool) -> ShiftSums<'_> {
        match &mut self.position {
            Some(p) => ShiftSums {
                probe: probe.then_some(&mut *p.probe),
                position: position.then_some(&mut *p.num),
                position_den: position.then_some(&mut *p.den),
            },
            None => ShiftSums::default(),
        }
    }

    /// Whether the model moves the probe, whose update sums then come from
    /// [`Self::shift_sums`] or [`Self::psi_sums`], and [`Self::probe_sums`].
    pub fn shifted(&self) -> bool {
        self.position.is_some()
    }

    /// The update sums of stored waves under the position model: `obj`, and with `probe`
    /// the shifted probe's; the shifted probe's are none without a position.
    pub fn psi_sums<'c>(&'c mut self, obj: Option<&'c mut UpdateSums>, probe: bool) -> PsiSums<'c> {
        PsiSums {
            obj,
            probe: (self.position.as_mut())
                .filter(|_| probe)
                .map(|p| &mut *p.probe),
        }
    }

    /// Adds the shifted probe's update sums the projections added into `out`, consuming
    /// them; nothing without a position.
    pub fn probe_sums(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        dims: &ViewDims,
        out: &mut UpdateSums,
    ) -> Result<(), GpuError> {
        if let Some(p) = &mut self.position {
            kernels.shifted_probe_sums(ctx, dims, p.probe, out)?;
        }
        Ok(())
    }
}

impl FeaturesMut<'_> {
    /// Whether some feature takes a step from far sums.
    pub fn stepped(&self) -> bool {
        self.scale.is_some() || self.background.is_some()
    }

    /// Whether the position is modelled, as [`Modelled::shifted`] of a lend.
    pub fn shifted(&self) -> bool {
        self.position.is_some()
    }

    /// The per-frame intensity scale, if modelled.
    pub fn scale(&self) -> Option<&DeviceBuffer<f32>> {
        self.scale.as_ref().map(|s| &*s.values)
    }

    /// The features lent to the passes of one cycle body over `probe`, which cannot change
    /// while they are: the model of every feature set, the position's after the spectra of
    /// `probe` it moves (two launches) and the propagator's stage `exit`, and the sums.
    pub fn lend<'b>(
        &'b mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        dims: &ViewDims,
        probe: &'b DeviceBuffer<C32>,
        exit: ExitPass<'b>,
    ) -> Result<Modelled<'b>, GpuError> {
        let (scale, scale_sums) = match &mut self.scale {
            Some(s) => (Some(&*s.values), Some(&mut *s.sums)),
            None => (None, None),
        };
        let (background, background_sums) = match &mut self.background {
            Some(b) => (Some(&*b.values), Some(&mut *b.sums)),
            None => (None, None),
        };
        let (shift, position) = match &mut self.position {
            Some(p) => {
                let shift = Shift {
                    offsets: &*p.offsets,
                    spectrum: kernels.probe_spectrum(ctx, dims, probe, p.spectrum)?,
                };
                let sums = PositionSums {
                    probe: &mut *p.probe_sums,
                    num: &mut *p.num,
                    den: &mut *p.den,
                };
                (Some(shift), Some(sums))
            }
            None => (None, None),
        };
        Ok(Modelled {
            model: Model {
                scale,
                background,
                shift,
                exit: Some(exit),
            },
            scale: scale_sums,
            background: background_sums,
            position,
        })
    }

    /// Every stepped feature's step from the far sums [`Modelled::far`] lent the passes
    /// before.
    pub fn step(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        grad: &GradKernels,
    ) -> Result<(), GpuError> {
        if let Some(s) = &mut self.scale {
            grad.scale_step(ctx, s.values, s.sums)?;
        }
        if let Some(b) = &mut self.background {
            kernels.background_ap(ctx, b.values, b.sums, b.nobs)?;
        }
        Ok(())
    }

    /// The position step from the position sums the AP projections wrote, after the
    /// updates; nothing without a position.
    pub fn position_step(
        &mut self,
        ctx: &ExecutionContext,
        grad: &GradKernels,
    ) -> Result<(), GpuError> {
        if let Some(p) = &mut self.position {
            let limits = [POSITION_STEP as f32, POSITION_LIMIT as f32];
            grad.position_step(ctx, p.offsets, [&*p.num, &*p.den], limits)?;
        }
        Ok(())
    }

    /// The background's photons over the unmasked pixels of every frame, for
    /// [`crate::gpu::ObjProbeScale::Absolute`]; `None` without a background.
    pub fn background_photons(
        &mut self,
        ctx: &ExecutionContext,
        grad: &GradKernels,
    ) -> Result<Option<&DeviceBuffer<f32>>, GpuError> {
        let Some(b) = &mut self.background else {
            return Ok(None);
        };
        grad.background_photons(ctx, b.values, b.nobs, b.photons)?;
        Ok(Some(&*b.photons))
    }
}

/// How AP and DM project a stack and the LLK pass takes its partials, with the buffers only
/// that way uses ([`GpuOptions::fused`]).
enum Passes {
    M1 {
        /// One plan per stack length, indexed by [`Frames::plan`].
        plans: Vec<Plan2d>,
        /// One stack's worth of exit waves: DM's `PO`, the LLK pass's forward stack.
        scratch: DeviceBuffer<C32>,
    },
    /// M2's fused passes, one scratch set for every stack in turn.
    Fused(Box<FusedBufs>),
}

/// [`Passes`] lent to a cycle.
pub(super) enum PassesMut<'a> {
    M1(M1Bufs<'a>),
    Fused(&'a mut FusedBufs),
}

impl PassesMut<'_> {
    /// The fused passes' scratch, if the passes are fused.
    pub fn fused(&mut self) -> Option<&mut FusedBufs> {
        match self {
            Self::M1(_) => None,
            Self::Fused(f) => Some(f),
        }
    }
}

/// [`Passes::M1`] lent to a cycle.
pub(super) struct M1Bufs<'a> {
    pub plans: Plans<'a>,
    pub scratch: &'a mut DeviceBuffer<C32>,
}

/// The [`Passes`] every algorithm shares, the stacks of the passes that stream their scratch
/// alone, and the LLK buffers those passes fill, built together, so the scratch and the LLK
/// buffers hold the longest of those stacks. DM's and RAAR's stacks count the scratch's
/// buffers with their own, and the position model's BH and LSQML stacks, whose sums borrow its
/// shift planes, are capped at its stacks, so no stack that touches it is longer.
struct Shared {
    /// The scratch's buffers.
    streamed: Vec<Streamed>,
    /// AP's, the LLK pass's, ML's gradient's and opening scale's stacks.
    cut: Cut,
    passes: Passes,
    /// One stack's LLK partials, on their way into the ring.
    llk: DeviceBuffer<LlkPartial>,
    /// The chunk sums M1's [`Kernels::llk`] reduces a stack's frames through.
    llk_scratch: LlkScratch,
    /// A multislice object's waves of one stack, from slice to slice in turn, each
    /// `[nf_max * nmode, ny, nx]`.
    slice_waves: Option<[DeviceBuffer<C32>; 2]>,
}

impl Shared {
    /// M2's fused passes (with `shift`, the position model's) or M1's scratch, `fused`
    /// picking, and with `sliced` a multislice object's waves, cut by `rule`.
    fn new(
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        rule: StackRule,
        fused: bool,
        shift: bool,
        sliced: bool,
    ) -> Result<Self, EngineError> {
        // M1's scratch: `obj_probe_to_psi`, the transforms and `apply_amplitude`; the fused
        // work planes: the row, column and inverse row passes; each shift plane: the pass that
        // writes it and a later one.
        let mut streamed = vec![Streamed::new(wave_bytes(dims), if fused { 3 } else { 4 })];
        if shift {
            streamed.push(Streamed::new(FusedBufs::shift_bytes(dims), 2));
        }
        if sliced {
            streamed.extend([Streamed::new(wave_bytes(dims), 3); 2]);
        }
        let cut = rule.cut(stream, dims, &streamed)?;
        let nf_max = cut.table.nf_max();
        let wave = || DeviceBuffer::zeroed(stream, cut.wave_len(dims));
        let slice_waves = sliced.then(|| Ok::<_, EngineError>([wave()?, wave()?]));
        let passes = if fused {
            let f = match shift {
                true => FusedBufs::with_shift(stream, dims, nf_max),
                false => FusedBufs::new(stream, dims, nf_max),
            };
            let f = f.map_err(|e| match e {
                GpuError::Shape(m) if shift => EngineError::Invalid(format!("position model: {m}")),
                GpuError::Shape(m) => EngineError::Invalid(format!(
                    "{m}; GpuOptions {{ fused: false, .. }} runs M1's launches instead"
                )),
                e => e.into(),
            })?;
            Passes::Fused(Box::new(f))
        } else {
            Passes::M1 {
                plans: cut.plans(stream, dims)?,
                scratch: DeviceBuffer::zeroed(stream, cut.wave_len(dims))?,
            }
        };
        Ok(Self {
            streamed,
            llk: DeviceBuffer::zeroed(stream, nf_max as usize)?,
            llk_scratch: LlkScratch::new(stream, dims, nf_max)?,
            slice_waves: slice_waves.transpose()?,
            cut,
            passes,
        })
    }
}

/// DM's or RAAR's stacks and the exit waves each carries across cycles, with RAAR's iterate,
/// cut by the carried exit waves and the shared scratch; the iterate, which one pass reads and
/// writes, does not count. Cycle 0 of a call overwrites both sets, so nothing is carried from
/// one call to the next.
struct CarriedBufs {
    cut: Cut,
    /// M1's plans, one per stack length of `cut`; none for the fused passes.
    plans: Vec<Plan2d>,
    /// Exit waves of each stack's views; planes past `nf * nmode` stay zero.
    psi: Vec<DeviceBuffer<C32>>,
    /// RAAR's iterate, laid out as `psi`.
    iterate: Option<Vec<DeviceBuffer<C32>>>,
}

impl CarriedBufs {
    /// DM's, or with `raar` RAAR's, against `shared`.
    fn new(
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        rule: StackRule,
        shared: &Shared,
        raar: bool,
    ) -> Result<Self, EngineError> {
        // The exit waves: read by the forward pass, written by the inverse one (M1: also
        // `dm_reflect`, `add_c32`); the iterate: the forward row pass (M1: `raar_iterate`).
        let wave = wave_bytes(dims);
        let mut streamed = shared.streamed.clone();
        streamed.push(Streamed::new(wave, 2));
        if raar {
            streamed.push(Streamed::new(wave, 1));
        }
        let cut = rule.cut(stream, dims, &streamed)?;
        let plans = match shared.passes {
            Passes::M1 { .. } => cut.plans(stream, dims)?,
            Passes::Fused(_) => Vec::new(),
        };
        let len = cut.wave_len(dims);
        let set = || -> Result<Vec<DeviceBuffer<C32>>, EngineError> {
            Ok((0..cut.len())
                .map(|_| DeviceBuffer::zeroed(stream, len))
                .collect::<Result<_, _>>()?)
        };
        Ok(Self {
            psi: set()?,
            iterate: raar.then(set).transpose()?,
            plans,
            cut,
        })
    }

    /// Whether these are the buffers `epoch`'s cycle binds.
    fn serves(&self, epoch: Epoch) -> bool {
        match epoch {
            Epoch::Dm { .. } => self.iterate.is_none(),
            Epoch::Raar { .. } => self.iterate.is_some(),
            _ => false,
        }
    }

    fn lend(&mut self) -> CarriedMut<'_> {
        CarriedMut {
            cut: &self.cut,
            plans: Plans(&mut self.plans),
            psi: &mut self.psi,
            iterate: self.iterate.as_deref_mut(),
        }
    }
}

/// [`CarriedBufs`] lent to a cycle.
pub(super) struct CarriedMut<'a> {
    pub cut: &'a Cut,
    pub plans: Plans<'a>,
    pub psi: &'a mut [DeviceBuffer<C32>],
    iterate: Option<&'a mut [DeviceBuffer<C32>]>,
}

impl<'a> CarriedMut<'a> {
    /// The plans, and stack `s`'s carried exit waves and RAAR iterate, for a RAAR cycle.
    pub fn raar(
        &mut self,
        s: usize,
    ) -> (
        &mut Plans<'a>,
        &mut DeviceBuffer<C32>,
        &mut DeviceBuffer<C32>,
    ) {
        let iterate =
            (self.iterate.as_deref_mut()).expect("Device::new_run built the RAAR iterate");
        (&mut self.plans, &mut self.psi[s], &mut iterate[s])
    }
}

/// A cycle with the buffers only it uses, and with graphs the graph captured for it.
struct EpochRun {
    cycle: Cycle,
    graph: Option<Captured>,
}

/// A graph and the record its body ended when captured, if any: each replay ends it again.
struct Captured {
    graph: CudaGraph<()>,
    ended: Option<Ended>,
}

/// A cycle's [`Epoch`] and the buffers only it uses, built together, so a graph captured
/// from it bakes in sizes that match its buffers.
enum Cycle {
    Ap {
        params: ApParams,
        llk: bool,
    },
    /// Its buffers are the device's [`CarriedBufs`], which every DM epoch shares.
    Dm {
        params: DmParams,
        first: bool,
    },
    /// Its buffers are the device's [`CarriedBufs`], which every RAAR epoch shares.
    Raar {
        params: RaarParams,
        first: bool,
    },
    Rpie(RpieParams, Box<RpieScratch>),
    /// Its buffers are the device's [`MlState`], which every ML epoch shares.
    Ml {
        params: MlParams,
        first: bool,
        llk: bool,
    },
    /// Its buffers are the device's [`BhState`], which every BH epoch of its batch size
    /// shares.
    Bh {
        params: BhParams,
        cg: bool,
    },
    /// Its buffers are the device's [`LsqmlState`], which every LSQML epoch of its batch
    /// size shares.
    Lsqml {
        params: LsqmlParams,
        pre: Pre,
    },
}

impl Cycle {
    fn epoch(&self) -> Epoch {
        match *self {
            Self::Ap { params, llk } => Epoch::Ap { params, llk },
            Self::Dm { params, first } => Epoch::Dm { params, first },
            Self::Raar { params, first } => Epoch::Raar { params, first },
            Self::Rpie(p, _) => Epoch::Rpie(p),
            Self::Ml { params, first, llk } => Epoch::Ml { params, first, llk },
            Self::Bh { params, cg } => Epoch::Bh { params, cg },
            Self::Lsqml { params, pre } => Epoch::Lsqml { params, pre },
        }
    }
}

/// ML's conjugate-gradient state and line-search scratch, sized by the device and the
/// position model alone, so one state serves the graphs of every ML epoch of that model; each
/// call's first epoch resets the state.
struct MlState {
    /// The line search's stacks, cut by its fields and moved probes.
    cut: Cut,
    /// One plan per stack length of `cut`, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    /// The last cycle's gradient, object then probe.
    h: [DeviceBuffer<C32>; 2],
    /// The search direction, object then probe.
    d: [DeviceBuffer<C32>; 2],
    /// Block partial pairs: the β sums of the object then the probe.
    parts: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    /// The line-search pairs of every frame's pixel chunks ([`crate::gpu::gamma_parts`]).
    gamma_parts: DeviceBuffer<f32>,
    gamma: DeviceBuffer<f32>,
    /// The calculated photons over the unmasked pixels of each frame.
    photons: DeviceBuffer<f32>,
    /// Their sum over the scan.
    icalc: DeviceBuffer<f32>,
    fields: GammaStacks,
    /// The line search's moved probes, once an ML call has run under the position model.
    moved: Option<MlMoved>,
}

/// An [`MlState`] lent to an epoch.
pub(super) struct MlBufs<'a> {
    pub cut: &'a Cut,
    pub plans: Plans<'a>,
    pub h: &'a mut [DeviceBuffer<C32>; 2],
    pub d: &'a mut [DeviceBuffer<C32>; 2],
    pub parts: &'a mut DeviceBuffer<f32>,
    pub beta: &'a mut DeviceBuffer<f32>,
    pub gamma_parts: &'a mut DeviceBuffer<f32>,
    pub gamma: &'a mut DeviceBuffer<f32>,
    pub photons: &'a mut DeviceBuffer<f32>,
    pub icalc: &'a mut DeviceBuffer<f32>,
    pub fields: &'a mut GammaStacks,
    pub moved: Option<&'a mut MlMoved>,
}

impl MlState {
    /// For frames of `dims` cut by `rule`, arrays of `lens = [object, probe]` values and with
    /// `moved` the position model.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        rule: StackRule,
        lens: [usize; 2],
        moved: bool,
    ) -> Result<Self, EngineError> {
        // The fields po, a, b: `gamma_fields`, the transform, `gamma_frames`; under the
        // position the moved probes of P and dP: `shifted_rows_fwd`, `shift_w`,
        // `obj_wave_to_psi`.
        let mut streamed = vec![Streamed::new(3 * wave_bytes(dims), 3)];
        if moved {
            streamed.push(Streamed::new(2 * plane_bytes::<C32>(dims, dims.nprobe), 3));
        }
        let cut = rule.cut(stream, dims, &streamed)?;
        let zeroed = |len| DeviceBuffer::<C32>::zeroed(stream, len);
        let pair = || DeviceBuffer::<f32>::zeroed(stream, 2);
        Ok(Self {
            plans: cut.plans(stream, dims)?,
            h: [zeroed(lens[0])?, zeroed(lens[1])?],
            d: [zeroed(lens[0])?, zeroed(lens[1])?],
            parts: DeviceBuffer::zeroed(stream, 4 * CG_BLOCKS as usize)?,
            beta: pair()?,
            gamma_parts: DeviceBuffer::zeroed(stream, 2 * crate::gpu::gamma_parts(dims)? as usize)?,
            gamma: pair()?,
            photons: DeviceBuffer::zeroed(stream, dims.nframe as usize)?,
            icalc: DeviceBuffer::zeroed(stream, 1)?,
            fields: GammaStacks::zeroed(stream, cut.wave_len(dims))?,
            moved: (moved.then(|| MlMoved::new(stream, kernels, dims, cut.table.nf_max())))
                .transpose()?,
            cut,
        })
    }

    fn lend(&mut self) -> MlBufs<'_> {
        MlBufs {
            cut: &self.cut,
            plans: Plans(&mut self.plans),
            h: &mut self.h,
            d: &mut self.d,
            parts: &mut self.parts,
            beta: &mut self.beta,
            gamma_parts: &mut self.gamma_parts,
            gamma: &mut self.gamma,
            photons: &mut self.photons,
            icalc: &mut self.icalc,
            fields: &mut self.fields,
            moved: self.moved.as_mut(),
        }
    }
}

/// The `(sum |x|, sum |x|^2)` pairs of the object and the probe that
/// [`Bufs::scale_obj_probe`] scales them by, and the block partials it sums them through.
pub(super) struct NormBufs {
    pub parts: DeviceBuffer<f32>,
    pub pairs: [DeviceBuffer<f32>; 2],
}

impl NormBufs {
    fn zeroed(stream: &CudaStream) -> Result<Self, EngineError> {
        let pair = || DeviceBuffer::<f32>::zeroed(stream, 2);
        Ok(Self {
            parts: DeviceBuffer::zeroed(stream, 2 * CG_BLOCKS as usize)?,
            pairs: [pair()?, pair()?],
        })
    }
}

/// BH's batches over epoch positions and its state, sized by the device, the batch size and
/// whether the position is modelled alone, so one state serves the graphs of every BH epoch
/// of that size and model; [`Device::bh_epoch`] sets each epoch's frame order before its
/// cycle.
struct BhState {
    /// Frames per batch, the batch size capped at the frame count.
    size: u32,
    /// Each batch's range of `stacks`.
    batches: Vec<Range<usize>>,
    /// The batches cut into stacks, one table entry each.
    stacks: Vec<Frames>,
    table: StackTable,
    /// One plan per stack length, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    /// A batch's forward exit waves and inverse-FFT'd residuals, one per stack of a batch.
    psi: Vec<DeviceBuffer<C32>>,
    r: Vec<DeviceBuffer<C32>>,
    /// One stack's forward-FFT'd derivatives along the scaled gradient and the last direction.
    lu: DeviceBuffer<C32>,
    lv: DeviceBuffer<C32>,
    /// The last direction, object then probe.
    prev: [DeviceBuffer<C32>; 2],
    /// The scan frame at each epoch position, and the views in that order.
    order: EpochOrder,
    /// One step's partial sums ([`crate::gpu::bh_parts`]).
    parts: DeviceBuffer<f32>,
    /// The epoch's step records, one per batch.
    steps: DeviceBuffer<f32>,
    /// The state of the position model, if modelled.
    moved: Option<BhMoved>,
}

/// A [`BhState`] lent to an epoch.
pub(super) struct BhBufs<'a> {
    pub size: u32,
    pub batches: &'a [Range<usize>],
    pub stacks: &'a [Frames],
    pub table: &'a StackTable,
    pub plans: Plans<'a>,
    pub psi: &'a mut [DeviceBuffer<C32>],
    pub r: &'a mut [DeviceBuffer<C32>],
    pub lu: &'a mut DeviceBuffer<C32>,
    pub lv: &'a mut DeviceBuffer<C32>,
    pub prev: &'a mut [DeviceBuffer<C32>; 2],
    pub order: &'a EpochOrder,
    pub parts: &'a mut DeviceBuffer<f32>,
    pub steps: &'a mut DeviceBuffer<f32>,
    pub moved: Option<&'a mut BhMoved>,
}

impl BhState {
    /// For `views` in batches of `size` frames, each cut into stacks by `rule`, and arrays of
    /// `lens = [object, probe]` values; under the position model, `scratch` holds the frames
    /// per stack of the shared scratch, whose shift planes its sums borrow, which caps its
    /// stacks. The epoch order is the scan's.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        size: u32,
        rule: StackRule,
        lens: [usize; 2],
        scratch: Option<u32>,
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        // psi, r, lu, lv, each written by one pass and transformed or read by later ones;
        // under the position also y (`obj_conj`, the transform, `pos_grad`), the moved probes
        // (`shifted_rows_fwd`, `shift_w`, `obj_wave_to_psi`) and the shift planes of the
        // shared scratch, which `shifted_psi_sums` writes and reads.
        let wave = wave_bytes(dims);
        let mut streamed = vec![Streamed::new(4 * wave, 3)];
        if scratch.is_some() {
            streamed.extend([
                Streamed::new(wave, 3),
                Streamed::new(plane_bytes::<C32>(dims, dims.nprobe), 3),
                Streamed::new(FusedBufs::shift_bytes(dims), 2),
            ]);
        }
        let frames = rule.frames(&streamed)?;
        let frames = scratch.map_or(frames, |s| frames.min(s));
        let (cut, batches) = batch_stacks(stream, dims, size, frames)?;
        let len = cut.wave_len(dims);
        let plans = cut.plans(stream, dims)?;
        let Cut {
            frames: stacks,
            table,
            ..
        } = cut;
        let slots = batches.iter().map(|b| b.len()).max().unwrap_or(0);
        let zeroed = |len| DeviceBuffer::<C32>::zeroed(stream, len);
        let moved = (scratch.is_some())
            .then(|| BhMoved::new(stream, kernels, dims, table.nf_max(), slots, len))
            .transpose()?;
        Ok(Self {
            size,
            plans,
            psi: (0..slots).map(|_| zeroed(len)).collect::<Result<_, _>>()?,
            r: (0..slots).map(|_| zeroed(len)).collect::<Result<_, _>>()?,
            lu: zeroed(len)?,
            lv: zeroed(len)?,
            prev: [zeroed(lens[0])?, zeroed(lens[1])?],
            order: EpochOrder::new(kernels, stream, views)?,
            parts: DeviceBuffer::zeroed(stream, crate::gpu::bh_parts(dims))?,
            steps: DeviceBuffer::zeroed(stream, BH_STEP as usize * batches.len())?,
            batches,
            stacks,
            table,
            moved,
        })
    }

    fn lend(&mut self) -> BhBufs<'_> {
        BhBufs {
            size: self.size,
            batches: &self.batches,
            stacks: &self.stacks,
            table: &self.table,
            plans: Plans(&mut self.plans),
            psi: &mut self.psi,
            r: &mut self.r,
            lu: &mut self.lu,
            lv: &mut self.lv,
            prev: &mut self.prev,
            order: &self.order,
            parts: &mut self.parts,
            steps: &mut self.steps,
            moved: self.moved.as_mut(),
        }
    }
}

/// rPIE's batches (one table entry and plan length each) and buffers, sized for the longest
/// batch.
struct RpieScratch {
    batches: Vec<Frames>,
    /// Values per frame plane.
    n: u32,
    table: StackTable,
    /// One plan per batch length, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    psi: DeviceBuffer<C32>,
    dlt: DeviceBuffer<C32>,
    max_chunks: DeviceBuffer<f32>,
    omax: DeviceBuffer<f32>,
    ptilde: DeviceBuffer<C32>,
    pmax: DeviceBuffer<f32>,
    ones: DeviceBuffer<C32>,
    /// The position model's state, if the device models positions.
    moved: Option<RpieMoved>,
}

/// An [`RpieScratch`] lent to an epoch.
pub(super) struct RpieBufs<'a> {
    pub batches: &'a [Frames],
    /// Values per frame plane.
    pub n: u32,
    pub table: &'a StackTable,
    pub plans: Plans<'a>,
    /// The batch's exit waves, then its object windows `O_v q`.
    pub psi: &'a mut DeviceBuffer<C32>,
    /// pty-chi's `Δ`, then the probe step's weighted `Δ`.
    pub dlt: &'a mut DeviceBuffer<C32>,
    /// [`crate::gpu::RpieKernels::power_max`]'s per-chunk maxes, for up to `nk_max` groups.
    pub max_chunks: &'a mut DeviceBuffer<f32>,
    /// Per-view `max |O_v|^2`.
    pub omax: &'a mut DeviceBuffer<f32>,
    /// `P / ((1 - a_o) |P|^2 + a_o max_i sum_m |P_m|^2)`.
    pub ptilde: &'a mut DeviceBuffer<C32>,
    pub pmax: &'a mut DeviceBuffer<f32>,
    /// A probe of ones, so `obj_probe_to_psi` yields the object windows.
    pub ones: &'a DeviceBuffer<C32>,
    pub moved: Option<&'a mut RpieMoved>,
}

impl RpieScratch {
    /// For `views` in `params`' batches, `moved` under the position model.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        params: &RpieParams,
        nprobe_values: usize,
        moved: bool,
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        let batch = to_u32("batch_size", params.batch_size)?;
        let (cut, _) = batch_stacks(stream, dims, batch, batch)?;
        let len = cut.wave_len(dims);
        let plans = cut.plans(stream, dims)?;
        let Cut {
            frames: batches,
            table,
            ..
        } = cut;
        let nk_max = table.nf_max() as usize * dims.nmode as usize;
        let n = to_u32("frame size", plane_len(dims))?;
        let max_groups = PowerMax {
            ngroup: to_u32("planes per batch", nk_max)?,
            nsum: 1,
            n,
        };
        Ok(Self {
            batches,
            n,
            plans,
            psi: DeviceBuffer::zeroed(stream, len)?,
            dlt: DeviceBuffer::zeroed(stream, len)?,
            max_chunks: DeviceBuffer::zeroed(stream, max_groups.chunks())?,
            omax: DeviceBuffer::zeroed(stream, nk_max)?,
            ptilde: DeviceBuffer::zeroed(stream, nprobe_values)?,
            pmax: DeviceBuffer::zeroed(stream, 1)?,
            ones: DeviceBuffer::from_host(stream, &vec![C32 { re: 1.0, im: 0.0 }; nprobe_values])?,
            moved: (moved.then(|| RpieMoved::new(stream, kernels, views, table.nf_max())))
                .transpose()?,
            table,
        })
    }

    fn lend(&mut self) -> RpieBufs<'_> {
        RpieBufs {
            batches: &self.batches,
            n: self.n,
            table: &self.table,
            plans: Plans(&mut self.plans),
            psi: &mut self.psi,
            dlt: &mut self.dlt,
            max_chunks: &mut self.max_chunks,
            omax: &mut self.omax,
            ptilde: &mut self.ptilde,
            pmax: &mut self.pmax,
            ones: &self.ones,
            moved: self.moved.as_mut(),
        }
    }
}

/// The [`Device`] lent to a cycle, but for its [`PassesMut`]: references only, so a cycle
/// launches on every buffer but rebinding a field leaves the `Device` as it was.
pub(super) struct Bufs<'a> {
    pub kernels: &'a Kernels,
    pub rpie: &'a RpieKernels,
    pub grad: &'a GradKernels,
    pub bh: &'a BhKernels,
    pub exit: ExitPass<'a>,
    pub tile: &'a TileDots,
    pub views: &'a Views,
    pub counts: &'a DeviceBuffer<u16>,
    pub obj: &'a mut DeviceBuffer<C32>,
    pub probe: &'a mut DeviceBuffer<C32>,
    /// A multislice object's slices past `obj`.
    pub slices: Option<SlicesMut<'a>>,
    pub obj_sums: &'a mut UpdateSums,
    pub probe_sums: &'a mut UpdateSums,
    pub obj_len: u32,
    pub probe_len: u32,
    /// The stacks of the passes that stream the shared scratch alone.
    pub scratch: &'a Cut,
    /// DM's or RAAR's carried stacks, once a run of either has built them.
    pub carried: Option<CarriedMut<'a>>,
    /// A stack's LLK partials, which [`crate::gpu::RecordWriter::push`] then copies into the
    /// record.
    pub llk: &'a mut DeviceBuffer<LlkPartial>,
    /// The chunk sums of M1's [`Kernels::llk`].
    pub llk_scratch: &'a mut LlkScratch,
    pub norms: &'a mut NormBufs,
    /// ML's state, once an ML call has built it.
    pub ml: Option<MlBufs<'a>>,
    /// BH's state, once a BH call has built it.
    pub bh_state: Option<BhBufs<'a>>,
    /// LSQML's state, once an LSQML call has built it.
    pub lsqml: Option<LsqmlBufs<'a>>,
    pub features: FeaturesMut<'a>,
}

/// The cuFFT plans lent to a cycle, by [`Frames::plan`].
pub(super) struct Plans<'a>(&'a mut [Plan2d]);

impl Plans<'_> {
    pub fn get(&mut self, i: usize) -> &mut Plan2d {
        &mut self.0[i]
    }
}

impl Device {
    /// Loads the kernels and uploads the views, counts and starting arrays on a new stream of
    /// `ctx`; every pass cuts `0 .. nframe` by the [`StackRule`] of `ctx`'s L2, the LLK ring
    /// holds `options.llk_slots` records of every frame, its unmasked pixels counted here, and
    /// `options.fused` picks the [`Passes`], of which only the fused run a propagator's stage.
    pub fn new(
        ctx: &Arc<CudaContext>,
        dims: ViewDims,
        uploads: Uploads<'_>,
        options: GpuOptions,
    ) -> Result<Self, EngineError> {
        let Uploads {
            stage,
            scale,
            addrs,
            counts,
            obj,
            probe,
            slices,
        } = uploads;
        let stream = ctx.new_stream()?;
        let kernels = Kernels::load(ctx)?;
        let rpie = RpieKernels::load(&kernels)?;
        let grad = GradKernels::load(&kernels)?;
        let bh = BhKernels::load(&kernels)?;
        let optics = OpticsKernels::load(&kernels)?;
        let exit = match stage {
            PlaneStage::None => ExitStage::None,
            _ if !options.fused => {
                return Err(EngineError::Invalid(
                    "a propagator with a plane stage needs GpuOptions { fused: true, .. }".into(),
                ));
            }
            PlaneStage::Reverse => ExitStage::Reverse,
            PlaneStage::Transfer(tf) => {
                ExitStage::Transfer(DeviceBuffer::from_host(&stream, &to_c32(tf))?)
            }
        };
        let views = Views::new(&stream, dims, addrs)?;
        let tile = TileDots::new(&stream, &dims)?;
        let rule = StackRule::new(options.stack_size, ctx.l2_cache_size_bytes()?);
        let shared = Shared::new(&stream, &dims, rule, options.fused, false, slices.is_some())?;
        let slices = slices
            .map(|s| {
                let upload = |v: &[Vec<C32>]| {
                    v.iter()
                        .map(|v| DeviceBuffer::from_host(&stream, v))
                        .collect::<Result<Vec<_>, _>>()
                };
                let ones = vec![C32 { re: 1.0, im: 0.0 }; probe.len()];
                Ok::<_, EngineError>(Slices {
                    deeper: upload(&s.deeper)?,
                    gaps: upload(&s.gaps)?,
                    ones: DeviceBuffer::from_host(&stream, &ones)?,
                    quad: s.quad,
                    position: PositionKernels::load(&kernels)?,
                })
            })
            .transpose()?;
        let mut dev = Self {
            obj_len: to_u32("object length", obj.len())?,
            probe_len: to_u32("probe length", probe.len())?,
            norms: NormBufs::zeroed(&stream)?,
            ring: LlkRing::new(
                &kernels,
                &stream,
                &dims,
                to_u32("llk_slots", options.llk_slots)?,
            )?,
            counts: DeviceBuffer::from_host(&stream, counts)?,
            obj: DeviceBuffer::from_host(&stream, obj)?,
            probe: DeviceBuffer::from_host(&stream, probe)?,
            slices,
            obj_sums: UpdateSums::zeroed(&stream, obj.len())?,
            probe_sums: UpdateSums::zeroed(&stream, probe.len())?,
            kernels,
            rpie,
            grad,
            bh,
            optics,
            exit,
            scale,
            tile,
            views,
            rule,
            shared,
            carried: None,
            use_graph: options.graph,
            runs: Vec::new(),
            llk_graph: None,
            ml: None,
            bh_state: None,
            lsqml: None,
            features: FeatureBufs::default(),
            stream,
        };
        let ctx = dev.context();
        dev.ring.device().count(&ctx, 0..dims.nframe, &dev.counts)?;
        dev.capture_llk()?;
        Ok(dev)
    }

    /// With graphs, captures the LLK pass's graph.
    fn capture_llk(&mut self) -> Result<(), EngineError> {
        if self.use_graph {
            self.llk_graph = Some(self.capture(|bufs, passes, ring, ctx| {
                bufs.forward_llk(ctx, passes, ring.record()).map(Some)
            })?);
        }
        Ok(())
    }

    /// Models `features`, which the caller checked against the scan, from here on: the
    /// values of the features already set are written in place; other features replace the
    /// buffers, and so first drop every graph, which addresses them, and then recapture the
    /// LLK pass's. Only the fused passes model features; a position also takes their
    /// scratch of a shift, which they drop again with it, and so recuts the stacks that
    /// stream that scratch and drops DM's or RAAR's carried stacks, cut against it.
    pub fn set_features(&mut self, features: &Features) -> Result<(), EngineError> {
        if matches!(self.shared.passes, Passes::M1 { .. }) && *features != Features::default() {
            return Err(EngineError::Invalid(
                "features need GpuOptions { fused: true, .. }".into(),
            ));
        }
        let f32s = |v: &Option<Vec<f64>>| v.as_ref().map(|v| v.iter().map(|&x| x as f32).collect());
        let (scale, background): (Option<Vec<f32>>, Option<Vec<f32>>) =
            (f32s(&features.scale), f32s(&features.background));
        let position: Option<Vec<f32>> =
            (features.position.as_ref()).map(|d| d.iter().flatten().map(|&x| x as f32).collect());
        let have = &mut self.features;
        if have.scale.is_some() == scale.is_some()
            && have.background.is_some() == background.is_some()
            && have.position.is_some() == position.is_some()
        {
            if let (Some(b), Some(v)) = (&mut have.scale, scale) {
                b.values.copy_from_host(&self.stream, &v)?;
            }
            if let (Some(b), Some(v)) = (&mut have.background, background) {
                b.values.copy_from_host(&self.stream, &v)?;
            }
            if let (Some(b), Some(v)) = (&mut have.position, position) {
                b.offsets.copy_from_host(&self.stream, &v)?;
            }
            return Ok(());
        }
        let stream = &self.stream;
        let dims = self.views.dims();
        let shared = (have.position.is_some() != position.is_some())
            .then(|| {
                let (moved, sliced) = (position.is_some(), self.slices.is_some());
                Shared::new(stream, dims, self.rule, true, moved, sliced)
            })
            .transpose()?;
        let nf_max = shared.as_ref().unwrap_or(&self.shared).cut.table.nf_max();
        let scale = scale
            .map(|v| {
                Ok::<_, EngineError>(ScaleBufs {
                    values: DeviceBuffer::from_host(stream, &v)?,
                    sums: DeviceBuffer::zeroed(stream, v.len())?,
                })
            })
            .transpose()?;
        let background = background
            .map(|v| {
                let pixel_frames = self.ring.device().pixel_frames();
                let mut nobs = DeviceBuffer::zeroed(stream, pixel_frames.len())?;
                nobs.copy_from_device_async(pixel_frames, stream)?;
                Ok::<_, EngineError>(BackgroundBufs {
                    values: DeviceBuffer::from_host(stream, &v)?,
                    sums: BackgroundSums::zeroed(stream, dims, nf_max)?,
                    nobs,
                    photons: DeviceBuffer::zeroed(stream, 1)?,
                })
            })
            .transpose()?;
        let position = position
            .map(|v| {
                Ok::<_, EngineError>(PositionBufs {
                    offsets: DeviceBuffer::from_host(stream, &v)?,
                    spectrum: SpectrumBuf::zeroed(stream, dims)?,
                    probe_sums: ShiftedProbeSums::zeroed(stream, dims)?,
                    num: DeviceBuffer::zeroed(stream, dims.nframe as usize)?,
                    den: DeviceBuffer::zeroed(stream, dims.nframe as usize)?,
                })
            })
            .transpose()?;
        self.runs.clear();
        self.llk_graph = None;
        self.features = FeatureBufs {
            scale,
            background,
            position,
        };
        if let Some(shared) = shared {
            self.shared = shared;
            self.carried = None;
        }
        self.capture_llk()?;
        Ok(())
    }

    /// The modelled features, read back.
    pub fn features(&self) -> Result<Features, EngineError> {
        let f64s = |b: Option<&DeviceBuffer<f32>>| {
            b.map(|b| Ok::<_, EngineError>(b.to_host_vec(&self.stream)?))
                .transpose()
                .map(|v| v.map(|v| v.into_iter().map(f64::from).collect()))
        };
        Ok(Features {
            scale: f64s(self.features.scale.as_ref().map(|b| &b.values))?,
            background: f64s(self.features.background.as_ref().map(|b| &b.values))?,
            position: (self.features.position.as_ref())
                .map(|p| {
                    let d = p.offsets.to_host_vec(&self.stream)?;
                    let d = d.as_chunks::<2>().0.iter();
                    Ok::<_, EngineError>(d.map(|o| o.map(f64::from)).collect())
                })
                .transpose()?,
        })
    }

    /// Whether some feature takes a step from far sums: the scale or the background.
    pub fn has_stepped(&self) -> bool {
        let f = &self.features;
        f.scale.is_some() || f.background.is_some()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// A fresh eager context on the stream: one per cycle, so leases end with its work.
    pub fn context(&self) -> ExecutionContext {
        ExecutionContext::new(self.stream.clone())
    }

    pub fn dims(&self) -> ViewDims {
        *self.views.dims()
    }

    pub fn obj(&self) -> &DeviceBuffer<C32> {
        &self.obj
    }

    pub fn probe(&self) -> &DeviceBuffer<C32> {
        &self.probe
    }

    /// A multislice object's slices past [`Self::obj`]; empty of one slice.
    pub fn deeper(&self) -> &[DeviceBuffer<C32>] {
        self.slices.as_ref().map_or(&[], |s| &s.deeper)
    }

    /// Fails on a multislice object for `what`, which models one slice: every pass but
    /// LSQML's and the LLK pass is enqueued behind this.
    fn one_slice(&self, what: &str) -> Result<(), EngineError> {
        match &self.slices {
            None => Ok(()),
            Some(s) => Err(EngineError::Invalid(format!(
                "{what} runs one object slice, got {}: only LSQML and the LLK pass run a \
                 multislice object",
                s.deeper.len() + 1
            ))),
        }
    }

    /// Overwrites the object in place, so every captured graph reads the new values.
    pub fn write_object(&mut self, obj: &[C32]) -> Result<(), EngineError> {
        Ok(self.obj.copy_from_host(&self.stream, obj)?)
    }

    /// Overwrites the object and probe in place.
    pub fn write_arrays(&mut self, obj: &[C32], probe: &[C32]) -> Result<(), EngineError> {
        self.one_slice("the phase ramp's removal")?;
        self.write_object(obj)?;
        self.probe.copy_from_host(&self.stream, probe)?;
        Ok(())
    }

    /// Lends every buffer, and apart the passes' own and the ring the cycle writes its records
    /// through.
    fn lend(&mut self) -> (Bufs<'_>, PassesMut<'_>, &mut RingDevice) {
        let bufs = Bufs {
            kernels: &self.kernels,
            rpie: &self.rpie,
            grad: &self.grad,
            bh: &self.bh,
            exit: ExitPass {
                kernels: &self.optics,
                stage: &self.exit,
                scale: self.scale,
            },
            tile: &self.tile,
            views: &self.views,
            counts: &self.counts,
            obj: &mut self.obj,
            probe: &mut self.probe,
            slices: (self.slices.as_mut())
                .zip(self.shared.slice_waves.as_mut())
                .map(|(s, waves)| SlicesMut {
                    deeper: &mut s.deeper,
                    gaps: &s.gaps,
                    ones: &s.ones,
                    quad: s.quad,
                    position: &s.position,
                    waves,
                }),
            obj_sums: &mut self.obj_sums,
            probe_sums: &mut self.probe_sums,
            obj_len: self.obj_len,
            probe_len: self.probe_len,
            scratch: &self.shared.cut,
            carried: self.carried.as_deref_mut().map(CarriedBufs::lend),
            llk: &mut self.shared.llk,
            llk_scratch: &mut self.shared.llk_scratch,
            norms: &mut self.norms,
            ml: self.ml.as_deref_mut().map(MlState::lend),
            bh_state: self.bh_state.as_deref_mut().map(BhState::lend),
            lsqml: self.lsqml.as_deref_mut().map(LsqmlState::lend),
            features: FeaturesMut {
                scale: self.features.scale.as_mut().map(|s| ScaleMut {
                    values: &mut s.values,
                    sums: &mut s.sums,
                }),
                background: self.features.background.as_mut().map(|b| BackgroundMut {
                    values: &mut b.values,
                    sums: &mut b.sums,
                    nobs: &b.nobs,
                    photons: &mut b.photons,
                }),
                position: self.features.position.as_mut().map(|p| PositionMut {
                    offsets: &mut p.offsets,
                    spectrum: &mut p.spectrum,
                    probe_sums: &mut p.probe_sums,
                    num: &mut p.num,
                    den: &mut p.den,
                }),
            },
        };
        let passes = match &mut self.shared.passes {
            Passes::M1 { plans, scratch } => PassesMut::M1(M1Bufs {
                plans: Plans(plans),
                scratch,
            }),
            Passes::Fused(f) => PassesMut::Fused(f),
        };
        (bufs, passes, self.ring.device())
    }

    /// Enqueues one `epoch` cycle, numbered `cycle`: replays the graph captured for it, or
    /// launches it eagerly without graphs, and counts the LLK record it ends, if any. The
    /// epochs of one [`Epoch::run`] keep their buffers and graphs until an epoch of another
    /// run replaces them.
    pub fn steady(&mut self, epoch: Epoch, cycle: usize) -> Result<(), EngineError> {
        self.steady_with(epoch, cycle, |_| Ok(()))
    }

    /// One BH epoch as [`Self::steady`] enqueues it, after setting its frame order: `order`
    /// holds the scan frame at each epoch position.
    pub fn bh_epoch(
        &mut self,
        params: BhParams,
        cg: bool,
        cycle: usize,
        order: &[u32],
    ) -> Result<(), EngineError> {
        self.steady_with(Epoch::Bh { params, cg }, cycle, |dev| {
            let ctx = dev.context();
            let bh = dev
                .bh_state
                .as_deref_mut()
                .expect("new_run built the BH state");
            Ok(bh.order.set(&ctx, &dev.kernels, &dev.views, order)?)
        })
    }

    /// The last BH epoch's steps, one per batch, read back: each batch's frames from the
    /// device's epoch order and its `(β, α, top, bottom)` record; none before a BH call.
    pub fn bh_steps(&self) -> Result<Vec<BhStep>, EngineError> {
        let Some(bh) = self.bh_state.as_deref() else {
            return Ok(Vec::new());
        };
        let ids = bh.order.ids().to_host_vec(&self.stream)?;
        let records = bh.steps.to_host_vec(&self.stream)?;
        let steps = bh
            .batches
            .iter()
            .zip(records.as_chunks::<{ BH_STEP as usize }>().0);
        Ok(steps
            .map(|(b, r)| {
                let at = bh.stacks[b.start].range.start as usize
                    ..bh.stacks[b.end - 1].range.end as usize;
                let [beta, alpha, top, bottom] = r.map(f64::from);
                BhStep {
                    frames: ids[at].iter().map(|&f| f as usize).collect(),
                    beta,
                    alpha,
                    top,
                    bottom,
                }
            })
            .collect())
    }

    /// One LSQML epoch as [`Self::steady`] enqueues it, after setting its frame order:
    /// `order` holds the scan frame at each epoch position.
    pub fn lsqml_epoch(
        &mut self,
        params: LsqmlParams,
        pre: Pre,
        cycle: usize,
        order: &[u32],
    ) -> Result<(), EngineError> {
        self.steady_with(Epoch::Lsqml { params, pre }, cycle, |dev| {
            let ctx = dev.context();
            let l = dev
                .lsqml
                .as_deref_mut()
                .expect("new_run built the LSQML state");
            Ok(l.set_order(&ctx, &dev.kernels, &dev.views, order)?)
        })
    }

    /// The last LSQML epoch's steps, one per batch; none before an LSQML call.
    pub fn lsqml_steps(&self) -> Result<Vec<LsqmlStep>, EngineError> {
        match self.lsqml.as_deref() {
            Some(l) => l.steps(&self.stream),
            None => Ok(Vec::new()),
        }
    }

    /// [`Self::steady`] with `stage` enqueued just before the cycle, once its buffers are
    /// built.
    fn steady_with(
        &mut self,
        epoch: Epoch,
        cycle: usize,
        stage: impl FnOnce(&mut Self) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let mut runs = std::mem::take(&mut self.runs);
        runs.retain(|r| r.cycle.epoch().run() == epoch.run());
        if self.carried.as_ref().is_some_and(|c| !c.serves(epoch)) {
            // The runs left are all of `epoch.run()`, so none binds them.
            self.carried = None;
        }
        let at = match runs.iter().position(|r| r.cycle.epoch() == epoch) {
            Some(at) => Ok(at),
            None => self.new_run(epoch).map(|run| {
                runs.push(run);
                runs.len() - 1
            }),
        };
        let done = at.and_then(|at| {
            stage(self)?;
            let run = &mut runs[at];
            let tag = Some((cycle, epoch.algorithm()));
            match &run.graph {
                Some(c) => self.replay(c, tag),
                None => {
                    let ctx = self.context();
                    let (mut bufs, mut passes, ring) = self.lend();
                    match bufs.cycle(&ctx, &mut passes, ring, &mut run.cycle) {
                        Ok(ended) => self.count(ended.as_ref(), tag),
                        Err(e) => Err(e.into()),
                    }
                }
            }
        });
        self.runs = runs;
        done
    }

    /// Enqueues the LLK record of the current estimate under `tag`, from the graph captured
    /// in [`Self::new`] or eagerly without graphs.
    pub fn llk_pass(&mut self, tag: Tag) -> Result<(), EngineError> {
        let graph = self.llk_graph.take();
        let done = match &graph {
            Some(c) => self.replay(c, tag),
            None => {
                let ctx = self.context();
                let (mut bufs, mut passes, ring) = self.lend();
                match bufs.forward_llk(&ctx, &mut passes, ring.record()) {
                    Ok(ended) => self.count(Some(&ended), tag),
                    Err(e) => Err(e.into()),
                }
            }
        };
        self.llk_graph = graph;
        done
    }

    /// The LLK records counted since the last flush, in order; waits for their copies.
    pub fn flush(&mut self) -> Result<Vec<(Tag, LlkTotals)>, EngineError> {
        Ok(self.ring.flush(&self.context())?)
    }

    /// Fails under a propagator with a plane stage for the passes that run none: the LLK
    /// pass, LSQML's, BH's and rPIE's do, AP's, DM's and RAAR's at fixed positions only, and
    /// every other is enqueued behind this.
    fn stageless(&self, algorithm: Algorithm) -> Result<(), EngineError> {
        let moved = self.features.position.is_some();
        match (&self.exit, algorithm) {
            (ExitStage::None, _) | (_, Algorithm::Lsqml | Algorithm::Bh | Algorithm::Rpie) => {
                Ok(())
            }
            (_, Algorithm::Ap | Algorithm::Dm | Algorithm::Raar) if !moved => Ok(()),
            _ => Err(EngineError::Invalid(format!(
                "{algorithm:?}{} runs no propagator with a plane stage: angular spectrum, or a \
                 Fresnel transform back in distance",
                if moved { " with a position" } else { "" }
            ))),
        }
    }

    /// The buffers of `epoch`'s cycle, and with graphs its graph, captured from them.
    fn new_run(&mut self, epoch: Epoch) -> Result<EpochRun, EngineError> {
        self.stageless(epoch.algorithm())?;
        if !matches!(epoch, Epoch::Lsqml { .. }) {
            self.one_slice(&format!("{:?}", epoch.algorithm()))?;
        }
        let mut cycle = match epoch {
            Epoch::Ap { params, llk } => Cycle::Ap { params, llk },
            Epoch::Dm { params, first } => {
                self.build_carried(false)?;
                Cycle::Dm { params, first }
            }
            Epoch::Raar { params, first } => {
                self.build_carried(true)?;
                Cycle::Raar { params, first }
            }
            Epoch::Rpie(p) => {
                let moved = self.features.position.is_some();
                let scratch = RpieScratch::new(
                    &self.stream,
                    &self.kernels,
                    &self.views,
                    &p,
                    self.probe.len(),
                    moved,
                )?;
                Cycle::Rpie(p, Box::new(scratch))
            }
            Epoch::Ml { params, first, llk } => {
                self.build_ml()?;
                Cycle::Ml { params, first, llk }
            }
            Epoch::Bh { params, cg } => {
                self.build_bh(&params)?;
                Cycle::Bh { params, cg }
            }
            Epoch::Lsqml { params, pre } => {
                self.build_lsqml(&params)?;
                Cycle::Lsqml { params, pre }
            }
        };
        let graph = if self.use_graph {
            Some(self.capture(|bufs, passes, ring, ctx| bufs.cycle(ctx, passes, ring, &mut cycle))?)
        } else {
            None
        };
        Ok(EpochRun { cycle, graph })
    }

    /// Builds DM's, or with `raar` RAAR's, [`CarriedBufs`] unless [`Self::steady_with`] kept
    /// them for this run.
    fn build_carried(&mut self, raar: bool) -> Result<(), EngineError> {
        if self.carried.is_none() {
            let dims = self.dims();
            let c = CarriedBufs::new(&self.stream, &dims, self.rule, &self.shared, raar)?;
            self.carried = Some(Box::new(c));
        }
        Ok(())
    }

    /// Builds the [`MlState`] for the position model unless it is built. One of the other
    /// model is replaced only after [`Self::set_features`] has dropped every run, so no
    /// graph captured from it survives.
    fn build_ml(&mut self) -> Result<(), EngineError> {
        let moved = self.features.position.is_some();
        if (self.ml.as_ref()).is_none_or(|ml| ml.moved.is_some() != moved) {
            let lens = [self.obj.len(), self.probe.len()];
            let dims = self.dims();
            let state = MlState::new(&self.stream, &self.kernels, &dims, self.rule, lens, moved)?;
            self.ml = Some(Box::new(state));
        }
        Ok(())
    }

    /// Builds the [`BhState`] for `params`' batch size and the position model unless it is
    /// built. A state of another size or model is replaced only here, when [`Self::steady`]
    /// has already dropped the runs of other parameters and [`Self::set_features`] those of
    /// other features, so no graph captured from it survives.
    fn build_bh(&mut self, params: &BhParams) -> Result<(), EngineError> {
        let dims = self.dims();
        let size = to_u32("batch_size", params.batch_size.min(dims.nframe as usize))?;
        let moved = self.features.position.is_some();
        if (self.bh_state.as_ref()).is_none_or(|s| s.size != size || s.moved.is_some() != moved) {
            let lens = [self.obj.len(), self.probe.len()];
            let state = BhState::new(
                &self.stream,
                &self.kernels,
                &self.views,
                size,
                self.rule,
                lens,
                moved.then(|| self.shared.cut.table.nf_max()),
            )?;
            self.bh_state = Some(Box::new(state));
        }
        Ok(())
    }

    /// Builds the [`LsqmlState`] for `params`' batch size and the position model unless it is
    /// built, replaced as [`Self::build_bh`] replaces BH's.
    fn build_lsqml(&mut self, params: &LsqmlParams) -> Result<(), EngineError> {
        let size = to_u32(
            "batch_size",
            params.batch_size.min(self.dims().nframe as usize),
        )?;
        let moved = self.features.position.is_some();
        if (self.lsqml.as_ref()).is_none_or(|s| s.size() != size || s.moved() != moved) {
            let state = LsqmlState::new(
                &self.stream,
                &self.kernels,
                &self.views,
                size,
                self.rule,
                moved.then(|| self.shared.cut.table.nf_max()),
                self.deeper().len(),
            )?;
            self.lsqml = Some(Box::new(state));
        }
        Ok(())
    }

    /// [`Bufs::scale_to_photons`] enqueued eagerly, with `total` the unmasked observed photons.
    pub fn scale_to_photons(&mut self, total: f32) -> Result<(), EngineError> {
        self.stageless(Algorithm::Ml)?;
        self.one_slice("ML's scaling")?;
        self.build_ml()?;
        let ctx = self.context();
        let (mut bufs, mut passes, _) = self.lend();
        Ok(bufs.with_ml(|b, ml| b.scale_to_photons(&ctx, &mut passes, ml, total))?)
    }

    /// PyNX `ScaleObjProbe(absolute=False)` enqueued eagerly: `Σ|O|` and `Σ|P|` balanced,
    /// `O P` unchanged.
    pub fn balance_obj_probe(&mut self) -> Result<(), EngineError> {
        self.one_slice("the object and probe balance")?;
        let ctx = self.context();
        let (mut bufs, _, _) = self.lend();
        Ok(bufs.scale_obj_probe(&ctx, None)?)
    }

    /// The ML gradients of the current estimate, object then probe, as
    /// [`Bufs::ml_grad`] leaves them in the update sums; downloads them and zeroes the sums.
    pub fn ml_grads(&mut self) -> Result<[Vec<C32>; 2], EngineError> {
        self.stageless(Algorithm::Ml)?;
        self.one_slice("ML's gradient")?;
        {
            let ctx = self.context();
            let (mut bufs, mut passes, _) = self.lend();
            bufs.ml_grad(&ctx, &mut passes, [true, true], None)?;
        }
        let grads = [
            self.obj_sums.num.to_host_vec(&self.stream)?,
            self.probe_sums.num.to_host_vec(&self.stream)?,
        ];
        let ctx = self.context();
        for sums in [&mut self.obj_sums, &mut self.probe_sums] {
            crate::gpu::zero(&ctx, &mut sums.num)?;
            crate::gpu::zero(&ctx, &mut sums.den)?;
        }
        Ok(grads)
    }

    /// `f` on the lent buffers, captured on a side stream, so nothing runs; with the record
    /// it ended.
    fn capture(
        &mut self,
        f: impl FnOnce(
            &mut Bufs<'_>,
            &mut PassesMut<'_>,
            &mut RingDevice,
            &ExecutionContext,
        ) -> Result<Option<Ended>, GpuError>
        + Send,
    ) -> Result<Captured, EngineError> {
        let side = self.kernels.context().new_stream()?;
        let (mut bufs, mut passes, ring) = self.lend();
        let mut ended = None;
        let op = crate::gpu::record(|ctx| {
            ended = Some(f(&mut bufs, &mut passes, ring, ctx)?);
            Ok(())
        });
        let graph = CudaGraph::capture(side, op).map_err(GpuError::from)?;
        let ended = ended.expect("a capture that succeeded ran its body once");
        Ok(Captured { graph, ended })
    }

    /// Enqueues `c`'s graph on the stream and counts the record it ends, if any, under `tag`.
    fn replay(&mut self, c: &Captured, tag: Tag) -> Result<(), EngineError> {
        // SAFETY: the context holds the replay's leases on every captured buffer until the
        // stream has run it, so no other stream can touch them meanwhile.
        unsafe { c.graph.launch().execute(&self.context()) }.map_err(GpuError::from)?;
        self.count(c.ended.as_ref(), tag)
    }

    /// Counts under `tag` the record a run just enqueued ended, if it ended one.
    fn count(&mut self, ended: Option<&Ended>, tag: Tag) -> Result<(), EngineError> {
        if let Some(ended) = ended {
            let nframe = self.dims().nframe;
            self.ring.enqueued(&self.context(), ended, tag, nframe)?;
        }
        Ok(())
    }
}

impl<'a> Bufs<'a> {
    /// `f` on these buffers and ML's state, lent apart.
    pub(super) fn with_ml<R>(&mut self, f: impl FnOnce(&mut Self, &mut MlBufs<'a>) -> R) -> R {
        let mut ml = self.ml.take().expect("Device::build_ml built the ML state");
        let r = f(self, &mut ml);
        self.ml = Some(ml);
        r
    }

    /// `f` on these buffers and DM's or RAAR's carried stacks, lent apart.
    pub(super) fn with_carried<R>(
        &mut self,
        f: impl FnOnce(&mut Self, &mut CarriedMut<'a>) -> R,
    ) -> R {
        let mut c = (self.carried.take()).expect("Device::new_run built the carried stacks");
        let r = f(self, &mut c);
        self.carried = Some(c);
        r
    }

    /// `f` on these buffers and BH's state, lent apart.
    pub(super) fn with_bh<R>(&mut self, f: impl FnOnce(&mut Self, &mut BhBufs<'a>) -> R) -> R {
        let mut bh = self
            .bh_state
            .take()
            .expect("Device::build_bh built the BH state");
        let r = f(self, &mut bh);
        self.bh_state = Some(bh);
        r
    }

    /// One cycle of `cycle` enqueued on `ctx`, and the LLK record it ends: an AP cycle with
    /// `llk` writes one through `ring`, no other cycle touches it.
    fn cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        ring: &mut RingDevice,
        cycle: &mut Cycle,
    ) -> Result<Option<Ended>, GpuError> {
        match cycle {
            Cycle::Ap { params, llk } => {
                self.ap_cycle(ctx, passes, params, llk.then(|| ring.record()))
            }
            Cycle::Dm { params, first } => self
                .with_carried(|b, c| b.dm_cycle(ctx, passes, c, params, *first))
                .map(|()| None),
            Cycle::Raar { params, first } => self
                .with_carried(|b, c| b.raar_cycle(ctx, passes, c, params, *first))
                .map(|()| None),
            Cycle::Rpie(p, scratch) => self.rpie_epoch(ctx, p, &mut scratch.lend()).map(|()| None),
            Cycle::Ml { params, first, llk } => self.with_ml(|b, ml| {
                b.ml_epoch(ctx, passes, ml, params, *first, llk.then(|| ring.record()))
            }),
            Cycle::Bh { params, cg } => self
                .with_bh(|b, bh| b.bh_epoch(ctx, passes, bh, params, *cg))
                .map(|()| None),
            Cycle::Lsqml { params, pre } => {
                self.lsqml_epoch(ctx, passes, params, *pre).map(|()| None)
            }
        }
    }
}
