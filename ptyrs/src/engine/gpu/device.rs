//! The device state a captured cycle addresses, and the graph captured from it, under one
//! owner. Nothing outside this module can reach a field of [`Device`] or of a stack's
//! buffers, and nothing here replaces one after [`Device::new`] but
//! [`Device::set_features`] and [`Device::set_sampling`], which drop every graph, the LLK
//! pass's recaptured at its next pass: a graph replays on the buffers the eager cycle would
//! use, and other buffers mean another `Device`, or other features, which start with no graph. The device also owns the
//! LLK ring, which counts a record once per enqueued run of a cycle or pass that ends one:
//! an eager run through [`LlkRing::record`], a graph's replay through [`LlkRing::replay`]
//! with the [`Ended`] its capture left.

use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use crate::fft::Plan2d;
use crate::gpu::sampling::{
    Modes, Normal, ObjectSampler, ObjectSolve, StackPoints, WindowBlock, window_views,
};
use crate::gpu::{
    BH_STEP, BackgroundSums, BandedTransfer, BhKernels, BhParts, C32, CG_BLOCKS, Ended, EpochOrder,
    Error as GpuError, ExitPass, ExitStage, FarSums, FrameWaves, FusedBufs, FusedInputs,
    GammaStacks, GradKernels, HalfWaves, Kernels, LlkPartial, LlkRing, LlkScratch, LlkTotals,
    MASKED, Model, OprShape, OpticsKernels, OwnModes, PositionKernels, PositionPartial, Positions,
    PowerMax, ProbeSpectrum, RecordWriter, RpieKernels, ScalePartial, Shape, Shift, ShiftProbe,
    ShiftSums, ShiftedNumSums, ShiftedProbes, ShiftedSums, SlicePhase, SliceSource, SpectrumBuf,
    StackRef, StackTable, StackWindows, SumTarget, UpdateSums, ViewAddr, ViewDims, Views, Waves,
    half_scale, stage_fft,
};
use crate::tile::TileDots;
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use super::bh::{BhMoved, BhSliced, update_nums};
use super::lsqml::{LsqmlBufs, LsqmlState, OprValues, Pre};
use super::ml::{MlMoved, MlPosition, MlWindows};
use super::rpie::RpieMoved;
use super::{
    Cut, Epoch, Frames, GpuOptions, StackRule, Streamed, Tag, batch_frames, batch_stacks,
    plane_bytes, plane_len, to_c32, to_u32, wave_bytes,
};
use crate::engine::features::{POSITION_LIMIT, POSITION_STEP};
use crate::engine::sampling::{OBJ_HELD, OBJ_PCG};
use crate::engine::{
    ApParams, BhParams, BhStep, DmParams, EngineError, Feature, Features, LsqmlParams, LsqmlStep,
    MlParams, PlaneStage, RaarParams, RpieParams, Sampling,
};
use crate::fault;

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
/// LSQML, BH and the LLK pass run it; every other call checks [`Device::one_unitary`] first.
struct Slices {
    deeper: Vec<DeviceBuffer<C32>>,
    gaps: Vec<DeviceBuffer<C32>>,
    /// A probe of ones `[nprobe, ny, nx]`: [`Kernels::psi_to_obj`] over it is a slice's
    /// adjoint of waves already times the conjugate of what entered it.
    ones: DeviceBuffer<C32>,
    quad: f32,
    /// Moves the probe the LLK pass sends through the slices.
    position: PositionKernels,
    /// One stack's waves, from slice to slice in turn, each a [`Cut::wave_len`] of
    /// [`Shared::cut`], which counts them.
    waves: [DeviceBuffer<C32>; 2],
}

impl Slices {
    /// The two waves of a stack of `cut`.
    fn waves(
        stream: &Arc<CudaStream>,
        cut: &Cut,
        dims: &ViewDims,
    ) -> Result<[DeviceBuffer<C32>; 2], EngineError> {
        let wave = || DeviceBuffer::zeroed(stream, cut.wave_len(dims));
        Ok([wave()?, wave()?])
    }
}

/// [`Slices`] lent to a cycle, but for the slices, which [`Bufs::deeper`] lends.
pub(super) struct SlicesMut<'a> {
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
    /// buffer: of `inputs` the views, the first slice as its object, and the probe, with
    /// `own` each frame's own mode 0, moved under a [`Model::shift`] by its offsets, and
    /// `deeper`, the slices past the first as those views read them ([`Modelled::slices`]).
    pub fn exit_waves(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        optics: &OpticsKernels,
        inputs: &FusedInputs<'_>,
        deeper: &[DeviceBuffer<C32>],
        (own, stack): (Option<&OwnModes<'_>>, StackRef<'_>),
    ) -> Result<&mut DeviceBuffer<C32>, GpuError> {
        let (views, d) = (inputs.views, inputs.views.dims());
        let [a, b] = &mut *self.waves;
        let offsets = match inputs.model.shift {
            None => {
                let probe = match own {
                    None => SliceSource::Probe(inputs.probe),
                    Some(own) => SliceSource::OwnProbe(inputs.probe, own),
                };
                optics.slice_mul(ctx, views, stack, inputs.obj, probe, a)?;
                None
            }
            Some(shift) => {
                match own {
                    None => kernels.shifted_probe(ctx, inputs, stack, b)?,
                    Some(own) => kernels.shifted_probe_own(ctx, inputs, own, stack, b)?,
                }
                self.position.shift_w(ctx, d, stack, b)?;
                let probe = SliceSource::Waves(FrameWaves::Probes(b));
                optics.slice_mul(ctx, views, stack, inputs.obj, probe, a)?;
                Some(shift.offsets)
            }
        };
        let (mut src, mut dst) = (a, b);
        for (obj, gap) in deeper.iter().zip(self.gaps) {
            optics.gap(ctx, d, stack, gap, src, false)?;
            let waves = SliceSource::Waves(FrameWaves::Views(src));
            optics.slice_mul(ctx, views, stack, obj, waves, dst)?;
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
    /// Under [`GpuOptions::psi_f16`], the scale of the carried stacks' [`HalfWaves`].
    half: Option<f32>,
    norms: NormBufs,
    ring: LlkRing<Tag>,
    graphs: Graphs,
    /// The epochs run since the parameters last changed, all of one [`Epoch::run`].
    runs: Vec<EpochRun>,
    /// ML's state, built by the first ML call and replaced only for another position model.
    ml: Option<Box<MlState>>,
    /// BH's state, built by the first BH call and replaced for another batch size, and for
    /// another reading of the object: with or without the position model, and its sampling.
    bh_state: Option<Box<BhState>>,
    /// LSQML's state, built by the first LSQML call and replaced for another batch size, and for
    /// another reading of the object: with or without the position model, and its sampling.
    lsqml: Option<Box<LsqmlState>>,
    features: FeatureBufs,
    /// A call through [`Self::summing`] began adding to the update sums and failed before
    /// consuming them; the next such call discards what it left.
    begun: bool,
    /// How a position's offsets enter the exit waves. Every state and graph built under a
    /// position model was built under it: [`Self::set_sampling`] drops the others.
    sampling: Sampling,
}

/// The features' device state as [`Device::set_features`] last set it: each modelled
/// feature's values and, refined, the sums its step reads, whose presence is its
/// [`Feature::refine`].
#[derive(Default)]
struct FeatureBufs {
    scale: Option<ScaleBufs>,
    background: Option<BackgroundBufs>,
    position: Option<PositionBufs>,
}

impl FeatureBufs {
    fn positioned(&self) -> bool {
        self.position.is_some()
    }

    /// Whether the position is modelled and refined.
    fn position_refined(&self) -> bool {
        (self.position.as_ref()).is_some_and(|p| p.partials.is_some())
    }

    /// The scale's, background's and position's [`Feature::refine`], none for one not
    /// modelled.
    fn refine(&self) -> [Option<bool>; 3] {
        [
            self.scale.as_ref().map(|s| s.sums.is_some()),
            self.background.as_ref().map(|b| b.sums.is_some()),
            self.position.as_ref().map(|p| p.partials.is_some()),
        ]
    }

    /// What the position model adds to the shared passes.
    fn scratch(&self) -> PositionScratch {
        let refined = self.position_refined();
        PositionScratch::of(self.position.as_ref().map(|p| &p.mover), refined)
    }

    /// Builds [`Mover::Object`]'s solve unless built, before AP's, DM's or RAAR's graph
    /// captures [`FeaturesMut::solve_obj`].
    fn build_solve(&mut self, stream: &CudaStream) -> Result<(), EngineError> {
        if let Some(PositionBufs {
            mover: Mover::Object { sampler, solve, .. },
            ..
        }) = &mut self.position
            && solve.is_none()
        {
            *solve = Some(Box::new(ObjectSolve::new(stream, sampler)?));
        }
        Ok(())
    }

    /// Zeroes the sums the passes add to across stacks; the per-frame ones each pass
    /// overwrites.
    fn clear_sums(&mut self, ctx: &ExecutionContext) -> Result<(), GpuError> {
        if let Some(sums) = self.background.as_mut().and_then(|b| b.sums.as_mut()) {
            sums.clear(ctx)?;
        }
        if let Some(PositionBufs {
            mover: Mover::Probe { probe_sums, .. },
            ..
        }) = &mut self.position
        {
            probe_sums.clear(ctx)?;
        }
        Ok(())
    }
}

/// What the position model adds to the passes over a cut, by what its offsets move and
/// whether they are `refined`: only the position sums read what a refined position adds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PositionScratch {
    /// No position.
    None,
    /// [`Mover::Probe`]: the fused passes' scratch of a shift.
    Shift { refined: bool },
    /// [`Mover::Object`]: the cut's [`Windows`], beside the bytes the sampler keeps across a
    /// sampling, `sample` ([`ObjectSampler::windows_held`]), and across a placing, `place`
    /// (its scratch and accumulator).
    Windows {
        sample: usize,
        place: usize,
        refined: bool,
    },
}

impl PositionScratch {
    fn of(mover: Option<&Mover>, refined: bool) -> Self {
        match mover {
            None => Self::None,
            Some(Mover::Probe { .. }) => Self::Shift { refined },
            Some(Mover::Object { sampler, .. }) => Self::Windows {
                sample: sampler.windows_held(),
                place: sampler.scratch_bytes() + sampler.acc_bytes(),
                refined,
            },
        }
    }
}

/// Groups [`ObjectSampler`] interpolates per pass.
const SAMPLER_CHUNK: u32 = 4;

/// A cut's windows under [`Sampling::FullObject`]: the views that read a stack's windows, the
/// blocks [`ObjectSampler::windows`] writes them to, every object mode, and the object update
/// numerator over them.
pub(super) struct Windows {
    views: Views,
    /// Blocks `[nobj, nf_max, ny, nx]`, each the views' object: the windows, and for the
    /// position sums of a refined position their derivatives over its `y` and `x`.
    blocks: DeviceBuffer<C32>,
    /// A multislice object's windows of each slice past the first, a block each.
    deeper: Vec<DeviceBuffer<C32>>,
    /// `[nobj, nyo, nxo]`, the spectra of each slice past the first, which
    /// [`Modelled::sliced`] takes for its windows.
    spectra: Vec<DeviceBuffer<C32>>,
    /// `[nobj, nf_max, ny, nx]`, the numerator alone: the full sums' `den` is the sampler's
    /// diagonal, which [`ObjectSampler::solve`] spreads itself.
    patches: DeviceBuffer<C32>,
}

impl Windows {
    /// For `cut`'s stacks of the views of `dims` into an object of `slices` slices, with the
    /// `derivatives` blocks or without.
    fn new(
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        cut: &Cut,
        slices: usize,
        derivatives: bool,
    ) -> Result<Self, EngineError> {
        let stacks: Vec<_> = cut.frames.iter().map(|f| f.range.clone()).collect();
        let views = window_views(stream, dims, &stacks)?;
        let len = views.dims().obj_len();
        let blocks = if derivatives { 3 } else { 1 };
        Ok(Self {
            views,
            blocks: DeviceBuffer::zeroed(stream, blocks * len)?,
            deeper: (1..slices)
                .map(|_| DeviceBuffer::zeroed(stream, len))
                .collect::<Result<_, _>>()?,
            spectra: (1..slices)
                .map(|_| DeviceBuffer::zeroed(stream, dims.obj_len()))
                .collect::<Result<_, _>>()?,
            patches: DeviceBuffer::zeroed(stream, len)?,
        })
    }

    /// Whether the blocks hold the derivatives.
    fn derivatives(&self) -> bool {
        self.blocks.len() > self.views.dims().obj_len()
    }

    /// Bytes per frame of a stack's block, and of its patches.
    fn bytes(dims: &ViewDims) -> usize {
        plane_bytes::<C32>(dims, dims.nobj)
    }
}

/// What [`Modelled::windows`] samples of a stack's windows.
#[derive(Clone, Copy, Debug)]
pub(super) enum Sample {
    /// Nothing: the passes read the views alone.
    Views,
    /// The windows, block 0.
    Windows,
    /// The windows and their derivatives over the position's `y` and `x`, blocks 0 to 2; the
    /// windows alone while the position is held, as only its sums read the derivatives.
    Derivatives,
}

/// [`Windows`] lent to one stack's passes by [`Modelled::windows`]: the views and the blocks
/// they read, and the update numerator.
pub(super) struct WindowsMut<'w> {
    pub read: StackWindows<'w>,
    patches: &'w mut DeviceBuffer<C32>,
}

impl WindowsMut<'_> {
    /// The update numerator over the windows, zeroed for the passes to add to.
    pub fn zeroed_patches(
        &mut self,
        ctx: &ExecutionContext,
    ) -> Result<&mut dyn SumTarget, GpuError> {
        crate::gpu::zero(ctx, self.patches)?;
        Ok(self.patches)
    }

    /// The numerator the passes added since [`Self::zeroed_patches`], which
    /// [`Modelled::place`] places.
    pub fn patches(&self) -> &DeviceBuffer<C32> {
        self.patches
    }
}

/// [`Shared`] rebuilt for another position model, with a multislice object's waves cut by
/// it.
struct Reshared {
    shared: Shared,
    waves: Option<[DeviceBuffer<C32>; 2]>,
}

/// What [`Device::set_features`] models, built by [`Device::build_features`].
pub(super) struct NewFeatures(FeaturesUpdate);

enum FeaturesUpdate {
    /// The values of the features already set, refined or held as they are: scale,
    /// background, position.
    InPlace([Option<Vec<f32>>; 3]),
    /// The buffers of other features, or another refine, with the shared passes their position
    /// takes unless the current ones serve it.
    Replace(Box<(FeatureBufs, Option<Reshared>)>),
}

/// What [`Device::set_sampling`] enters, built by [`Device::build_sampling`].
pub(super) struct NewSampling {
    sampling: Sampling,
    /// Of another sampling while a position is modelled, what its offsets move under it, with
    /// the shared passes it takes unless the current ones serve it.
    moved: Option<(Mover, Option<Reshared>)>,
}

/// A per-frame intensity scale `[nframe]` and, refined, its far sums `[nframe]`.
struct ScaleBufs {
    values: DeviceBuffer<f32>,
    sums: Option<DeviceBuffer<ScalePartial>>,
}

/// A detector background `[ny, nx]` in the counts' order, refined its far sums, each
/// detector pixel's unmasked frames and the background's photons over them.
struct BackgroundBufs {
    values: DeviceBuffer<f32>,
    sums: Option<BackgroundSums>,
    /// A copy of the ring's [`LlkRing::pixel_frames`], final once [`Device::new`] has
    /// counted every frame.
    nobs: DeviceBuffer<u32>,
    /// `[1]`, written by [`FeaturesMut::background_photons`].
    photons: DeviceBuffer<f32>,
}

/// Per-frame offsets `[nframe, 2]` of the position model, refined each frame's position
/// sums, and what the offsets move.
struct PositionBufs {
    offsets: DeviceBuffer<f32>,
    partials: Option<PositionPartials>,
    mover: Mover,
}

/// Each frame's position sums `[nframe]`, which the fused passes write and the position step
/// reads.
struct PositionPartials {
    num: DeviceBuffer<PositionPartial>,
    den: DeviceBuffer<PositionPartial>,
}

/// What a position's offsets move, by the device's [`Sampling`].
enum Mover {
    /// [`Sampling::Patch`]: the probe, by the spectra of the probe it moves, with the shifted
    /// probe's update sums.
    Probe {
        spectrum: SpectrumBuf,
        probe_sums: Box<ShiftedProbes>,
    },
    /// [`Sampling::FullObject`]: the object's windows, which the passes sample from the
    /// spectra of the object they read; the probe stays.
    Object {
        sampler: Box<ObjectSampler>,
        /// `[nobj, nyo, nxo]`.
        spectra: DeviceBuffer<C32>,
        /// [`ObjectSampler::solve`]'s buffers, which only [`FeatureBufs::build_solve`] builds.
        solve: Option<Box<ObjectSolve>>,
    },
}

impl Mover {
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        sampling: Sampling,
    ) -> Result<Self, EngineError> {
        Ok(match sampling {
            Sampling::Patch => Self::Probe {
                spectrum: SpectrumBuf::zeroed(stream, dims)?,
                probe_sums: Box::new(ShiftedProbes::zeroed(stream, dims)?),
            },
            Sampling::FullObject => Self::Object {
                sampler: Box::new(ObjectSampler::new(
                    kernels,
                    stream,
                    *dims,
                    1,
                    SAMPLER_CHUNK,
                )?),
                spectra: DeviceBuffer::zeroed(stream, dims.obj_len())?,
                solve: None,
            },
        })
    }

    fn moves_probe(&self) -> bool {
        matches!(self, Self::Probe { .. })
    }
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
    sums: Option<&'a mut DeviceBuffer<ScalePartial>>,
}

/// [`BackgroundBufs`] lent to a cycle.
struct BackgroundMut<'a> {
    values: &'a mut DeviceBuffer<f32>,
    sums: Option<&'a mut BackgroundSums>,
    nobs: &'a DeviceBuffer<u32>,
    photons: &'a mut DeviceBuffer<f32>,
}

/// [`PositionBufs`] lent to a cycle.
struct PositionMut<'a> {
    offsets: &'a mut DeviceBuffer<f32>,
    partials: Option<&'a mut PositionPartials>,
    mover: &'a mut Mover,
}

/// The position model's state a cycle that runs its own passes owns: the offsets by epoch
/// position, their kernels and the moved probes, by the device's [`Sampling`].
pub(super) struct Moved {
    pos: Positions,
    kernels: PositionKernels,
    probes: MovedProbes,
}

/// A cycle's moved probes.
enum MovedProbes {
    /// [`Sampling::Patch`]: per batch slot, `[nf_max * nprobe, ny, nx]`.
    Patch(Vec<DeviceBuffer<C32>>),
    /// [`Sampling::FullObject`]: none, the probe stays.
    FullObject,
}

impl Moved {
    /// For frames of `dims` in `slots` stacks per batch of up to `nf_max` frames each, under
    /// `sampling`.
    pub fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        nf_max: u32,
        slots: usize,
        sampling: Sampling,
    ) -> Result<Self, EngineError> {
        let wlen = nf_max as usize * dims.nprobe as usize * plane_len(dims);
        Ok(Self {
            pos: Positions::zeroed(stream, dims)?,
            kernels: PositionKernels::load(kernels)?,
            probes: match sampling {
                Sampling::Patch => MovedProbes::Patch(
                    (0..slots)
                        .map(|_| DeviceBuffer::zeroed(stream, wlen))
                        .collect::<Result<_, _>>()?,
                ),
                Sampling::FullObject => MovedProbes::FullObject,
            },
        })
    }
}

/// What a cycle that runs its own passes under the position model builds by: the frames per
/// stack of the shared scratch, the device's [`Sampling`] and whether the position is
/// refined, for the cycle's own step of the offsets.
#[derive(Clone, Copy, Debug)]
pub(super) struct MovedScratch {
    pub frames: u32,
    pub sampling: MovedSampling,
    pub refined: bool,
}

/// The device's [`Sampling`] as [`MovedScratch`] holds it: under [`Sampling::FullObject`],
/// the bytes the device's sampler keeps across its calls.
#[derive(Clone, Copy, Debug)]
pub(super) enum MovedSampling {
    Patch,
    FullObject {
        /// [`ObjectSampler::windows_held`].
        held: usize,
        /// [`ObjectSampler::scratch_bytes`].
        scratch: usize,
        /// [`ObjectSampler::acc_bytes`].
        acc: usize,
    },
}

impl From<MovedSampling> for Sampling {
    fn from(s: MovedSampling) -> Self {
        match s {
            MovedSampling::Patch => Self::Patch,
            MovedSampling::FullObject { .. } => Self::FullObject,
        }
    }
}

/// A cycle's [`Moved`] lent with the device's position buffers, by the device's [`Sampling`]:
/// a cycle with both paths matches it.
pub(super) enum PositionLent<'m, X> {
    Patch(MovedLent<'m, X>),
    FullObject(SampledLent<'m, X>),
}

impl<'m, X> PositionLent<'m, X> {
    /// `moved` with `features`' position buffers and, under [`Sampling::Patch`], `fused`:
    /// none without a position, an error when they disagree, as for a cycle's state built
    /// for another position model. The offsets are lent alike refined or held: only the
    /// cycle's own step, which [`FeaturesMut::own_step`] pairs with the refine, moves them.
    pub fn lend(
        moved: Option<(&'m mut Moved, X)>,
        fused: Option<&'m mut FusedBufs>,
        features: &'m mut FeaturesMut<'_>,
    ) -> Result<Option<Self>, GpuError> {
        let (m, own, p) = match (moved, features.position.as_mut()) {
            (None, None) => return Ok(None),
            (Some((m, own)), Some(p)) => (m, own, p),
            _ => return Err(another_model()),
        };
        let Moved {
            pos,
            kernels,
            probes,
        } = m;
        match (probes, &mut *p.mover, fused) {
            (
                MovedProbes::Patch(w),
                Mover::Probe {
                    spectrum,
                    probe_sums,
                },
                Some(fused),
            ) => Ok(Some(Self::Patch(MovedLent {
                own,
                fused,
                w,
                pos,
                kernels,
                offsets: &mut *p.offsets,
                spectrum,
                probe_sums: probe_sums.get(),
            }))),
            (
                MovedProbes::FullObject,
                Mover::Object {
                    sampler, spectra, ..
                },
                _,
            ) => Ok(Some(Self::FullObject(SampledLent {
                own,
                pos,
                kernels,
                offsets: &mut *p.offsets,
                sampler,
                spectra,
            }))),
            _ => Err(another_model()),
        }
    }

    /// The cycle's other buffers of the model.
    pub fn own(&self) -> &X {
        match self {
            Self::Patch(m) => &m.own,
            Self::FullObject(s) => &s.own,
        }
    }

    /// The offsets by epoch position from the device's, through `order`, the epoch's.
    pub fn gather(&mut self, ctx: &ExecutionContext, order: &EpochOrder) -> Result<(), GpuError> {
        match self {
            Self::Patch(m) => m.gather(ctx, order),
            Self::FullObject(s) => s.gather(ctx, order),
        }
    }
}

pub(super) fn another_model() -> GpuError {
    GpuError::Shape("cycle state built for another position model".into())
}

impl<'m, X> TryFrom<PositionLent<'m, X>> for MovedLent<'m, X> {
    type Error = GpuError;

    /// The Patch lend; an error under [`Sampling::FullObject`], which keeps the probe.
    fn try_from(lent: PositionLent<'m, X>) -> Result<Self, GpuError> {
        match lent {
            PositionLent::Patch(m) => Ok(m),
            PositionLent::FullObject(_) => Err(GpuError::Shape(
                "this cycle moves the probe by the offsets, which sample the object under \
                 Sampling::FullObject; Sampling::Patch runs it"
                    .into(),
            )),
        }
    }
}

/// [`PositionLent::FullObject`]: a cycle's [`Moved`] lent with the device's offsets by frame,
/// which sample the object's windows, and its sampler; `own` the cycle's other buffers of the
/// model.
pub(super) struct SampledLent<'m, X> {
    pub own: X,
    pub pos: &'m mut Positions,
    pub kernels: &'m PositionKernels,
    /// By frame: the device's offsets, which only the cycle's own step moves
    /// ([`FeaturesMut::own_step`]).
    pub offsets: &'m mut DeviceBuffer<f32>,
    pub sampler: &'m mut ObjectSampler,
    /// `[nobj, nyo, nxo]`, the spectra [`ObjectSampler::windows`] reads.
    pub spectra: &'m mut DeviceBuffer<C32>,
}

impl<X> SampledLent<'_, X> {
    /// The offsets by epoch position from the device's, through `order`, the epoch's.
    pub fn gather(&mut self, ctx: &ExecutionContext, order: &EpochOrder) -> Result<(), GpuError> {
        self.kernels.gather(ctx, order, self.offsets, self.pos)
    }
}

/// [`PositionLent::Patch`]: a cycle's [`Moved`] lent with the device's position buffers: the
/// offsets by frame, the probe spectra and the shifted probe's sums; `own` the cycle's other
/// buffers of the model, `fused` the scratch of the shifted sums over its stacks.
pub(super) struct MovedLent<'m, X> {
    pub own: X,
    fused: &'m mut FusedBufs,
    w: &'m mut [DeviceBuffer<C32>],
    pos: &'m mut Positions,
    kernels: &'m PositionKernels,
    offsets: &'m mut DeviceBuffer<f32>,
    spectrum: &'m mut SpectrumBuf,
    probe_sums: &'m mut ShiftedNumSums,
}

impl<'m, X> MovedLent<'m, X> {
    /// [`PositionLent::lend`] for a cycle with the [`Sampling::Patch`] path alone: an error
    /// under [`Sampling::FullObject`], which keeps the probe this path moves.
    pub fn lend(
        moved: Option<(&'m mut Moved, X)>,
        fused: Option<&'m mut FusedBufs>,
        features: &'m mut FeaturesMut<'_>,
    ) -> Result<Option<Self>, GpuError> {
        (PositionLent::lend(moved, fused, features)?)
            .map(Self::try_from)
            .transpose()
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
    /// By frame: the device's offsets, which only the cycle's own step moves
    /// ([`FeaturesMut::own_step`]).
    pub offsets: &'m mut DeviceBuffer<f32>,
    pub probe_sums: &'m mut ShiftedNumSums,
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

/// Proof that the device's object has one slice, made only by [`Device::one_slice`]: a lend
/// narrowed by one ([`Modelled::one_slice`]) reads the object's first slice as the whole
/// object.
#[derive(Clone, Copy)]
pub(super) struct OneSlice(());

/// A lend over an object of any slice count, which reads its slices by [`Modelled::sliced`].
#[derive(Clone, Copy)]
pub(super) struct AnySlices;

/// The features lent to the passes of one cycle body by [`FeaturesMut::lend`]: the model
/// every pass that compares with the counts reads, what samples the object it reads, and the
/// sums the passes write; `K`, [`OneSlice`] or [`AnySlices`], is what the lend may read of
/// the object.
pub(super) struct Modelled<'b, K> {
    model: Model<'b>,
    sampled: Option<Sampled<'b>>,
    scale: Option<&'b mut DeviceBuffer<ScalePartial>>,
    background: Option<&'b mut BackgroundSums>,
    position: Option<PositionSums<'b>>,
    slices: PhantomData<K>,
}

/// An object's first slice and the slices past it, none of one slice.
#[derive(Clone, Copy)]
pub(super) struct Sliced<'a> {
    pub first: &'a DeviceBuffer<C32>,
    pub deeper: &'a [DeviceBuffer<C32>],
}

/// What the passes over each stack of a cut read of a multislice object, by
/// [`Modelled::sliced`].
pub(super) enum SlicedRead<'w> {
    /// The scan's views and the object's slices.
    Base(&'w Views, Sliced<'w>),
    /// Under [`Sampling::FullObject`], the scan's views and the cut's windows, beside whose
    /// deeper windows the spectra of the slices past the first are taken.
    Windows(&'w Views, &'w mut Windows),
}

/// [`Mover::Object`] lent by [`Modelled`]: the sampler, the spectra of the object the passes
/// read and the offsets by frame they sample it at.
struct Sampled<'b> {
    sampler: &'b mut ObjectSampler,
    spectra: &'b DeviceBuffer<C32>,
    offsets: &'b DeviceBuffer<f32>,
}

/// What a [`Modelled`] lend's position model moves, by [`Modelled::moves`].
#[derive(Clone, Copy)]
pub(super) enum Moves<'b> {
    /// Nothing: no position.
    Nothing,
    /// The probe, by its shift ([`Mover::Probe`]).
    Probe(Shift<'b>),
    /// The object, sampled at the offsets ([`Mover::Object`]).
    Object,
}

/// The sums of the position model lent by [`Modelled`]: each frame's position sums while
/// the offsets are refined. The shifted probe's sums are lent beside it, by
/// [`FeaturesMut::lend_shifted`].
struct PositionSums<'b> {
    partials: Option<&'b mut PositionPartials>,
}

impl Modelled<'_, OneSlice> {
    /// The views and object a fused pass over `stack` reads, of `base`, the scan's views and
    /// the object the lend took: `base`, or under [`Sampling::FullObject`] the stack's windows
    /// sampled into `windows`, its cut's, by [`Self::windows`].
    pub fn object<'w>(
        &mut self,
        ctx: &ExecutionContext,
        base: (&'w Views, &'w DeviceBuffer<C32>),
        stack: StackRef<'_>,
        windows: Option<&'w mut Windows>,
    ) -> Result<(&'w Views, &'w DeviceBuffer<C32>), GpuError> {
        Ok(
            match self.windows(ctx, base.0, stack, windows, Sample::Windows)? {
                None => base,
                Some(w) => (w.read.views, w.read.blocks),
            },
        )
    }

    /// Under [`Sampling::FullObject`], `stack`'s windows over the scan's `views` lent from
    /// `w`, its cut's, after sampling what `sample` names from the spectra of the object the
    /// lend took, three launches per [`SAMPLER_CHUNK`] groups a block; none without either.
    /// Their disagreement is an error, as for a cut built under another position model.
    pub fn windows<'w>(
        &mut self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        w: Option<&'w mut Windows>,
        sample: Sample,
    ) -> Result<Option<WindowsMut<'w>>, GpuError> {
        let (s, w) = match (&mut self.sampled, w) {
            (None, None) => return Ok(None),
            (Some(s), Some(w)) => (s, w),
            _ => return Err(another_model()),
        };
        let at = StackPoints {
            views,
            stack,
            offsets: s.offsets,
        };
        let refined = (self.position.as_ref()).is_some_and(|p| p.partials.is_some());
        let orders: &[[u32; 2]] = match sample {
            Sample::Views => &[],
            Sample::Derivatives if refined => &[[0, 0], [1, 0], [0, 1]],
            Sample::Windows | Sample::Derivatives => &[[0, 0]],
        };
        for (block, &order) in (0..).zip(orders) {
            let block = WindowBlock {
                slice: 0,
                order,
                modes: Modes::All,
                block,
            };
            s.sampler
                .windows(ctx, at, s.spectra, block, &mut w.blocks)?;
        }
        Ok(Some(WindowsMut {
            read: StackWindows {
                views: &w.views,
                blocks: &w.blocks,
            },
            patches: &mut w.patches,
        }))
    }

    /// Under [`Sampling::FullObject`], adds `patches`, a stack's object sums over its windows
    /// ([`WindowsMut::patches`]), placed back at the stack's offsets over the scan's `views`
    /// to `out` `[nobj, nyo, nxo]` ([`ObjectSampler::place`]); an error otherwise.
    pub fn place(
        &mut self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        patches: &DeviceBuffer<C32>,
        out: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        let s = self.sampled.as_mut().ok_or_else(another_model)?;
        let at = StackPoints {
            views,
            stack,
            offsets: s.offsets,
        };
        s.sampler.place(ctx, at, patches, 0, out)
    }

    /// Under [`Sampling::FullObject`], `block` of `stack`'s windows over the scan's `views`
    /// into `out`, sampled at the stack's offsets from `spectra` ([`Self::spectra_of`]), or
    /// without, from those of the object the lend took ([`ObjectSampler::windows`], three
    /// launches per [`SAMPLER_CHUNK`] groups); an error otherwise.
    pub fn windows_of(
        &mut self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        spectra: Option<&DeviceBuffer<C32>>,
        block: WindowBlock,
        out: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        self.sample_windows(ctx, views, stack, spectra, block, out)
    }

    /// Under [`Sampling::FullObject`], each of `stack`'s frames' position sums of `psi`
    /// against its windows `w`, sampled with their derivatives
    /// ([`GradKernels::window_position_sums`]), nothing while the position is held; an error
    /// otherwise.
    pub fn window_position_sums(
        &mut self,
        ctx: &ExecutionContext,
        grad: &GradKernels,
        w: StackWindows<'_>,
        stack: StackRef<'_>,
        psi: &dyn Waves,
        probe: &DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        let (Some(_), Some(p)) = (&self.sampled, &mut self.position) else {
            return Err(another_model());
        };
        let Some(q) = p.partials.as_deref_mut() else {
            return Ok(());
        };
        grad.window_position_sums(ctx, w, stack, psi, probe, [&mut q.num, &mut q.den])
    }
}

impl<'b> Modelled<'b, AnySlices> {
    /// This lend, reading the object it took as the whole object, which `_one` proves.
    pub fn one_slice(self, _one: OneSlice) -> Modelled<'b, OneSlice> {
        let Self {
            model,
            sampled,
            scale,
            background,
            position,
            slices: _,
        } = self;
        Modelled {
            model,
            sampled,
            scale,
            background,
            position,
            slices: PhantomData,
        }
    }

    /// What the passes over each stack read of `base`, the scan's views and the object's
    /// slices, the first the one the lend took: `base`, or under [`Sampling::FullObject`]
    /// `windows`, its cut's, after the spectra of each slice past the first beside their
    /// windows ([`Self::spectra_of`], one transform a slice). Their disagreement is an error,
    /// as for a cut built under another position model, and so are windows of another slice
    /// count.
    pub fn sliced<'w>(
        &mut self,
        ctx: &ExecutionContext,
        base: (&'w Views, Sliced<'w>),
        windows: Option<&'w mut Windows>,
    ) -> Result<SlicedRead<'w>, GpuError> {
        let (views, obj) = base;
        let w = match (&self.sampled, windows) {
            (None, None) => return Ok(SlicedRead::Base(views, obj)),
            (Some(_), Some(w)) => w,
            _ => return Err(another_model()),
        };
        if w.deeper.len() != obj.deeper.len() {
            return Err(GpuError::Shape(format!(
                "windows of {} slices for an object of {}",
                1 + w.deeper.len(),
                1 + obj.deeper.len()
            )));
        }
        for (slice, spectra) in obj.deeper.iter().zip(&mut w.spectra) {
            self.spectra_of(ctx, slice, spectra)?;
        }
        Ok(SlicedRead::Windows(views, w))
    }

    /// The views and every slice the passes over `stack` read of `read`: its base, or each
    /// slice's windows of the stack sampled into its windows, three launches per
    /// [`SAMPLER_CHUNK`] groups a slice, as [`Modelled::windows_of`].
    pub fn slices<'r>(
        &mut self,
        ctx: &ExecutionContext,
        read: &'r mut SlicedRead<'_>,
        stack: StackRef<'_>,
    ) -> Result<(&'r Views, Sliced<'r>), GpuError> {
        let (views, w) = match read {
            SlicedRead::Base(views, obj) => return Ok((*views, *obj)),
            SlicedRead::Windows(views, w) => (*views, &mut **w),
        };
        let block = WindowBlock {
            slice: 0,
            order: [0, 0],
            modes: Modes::All,
            block: 0,
        };
        self.sample_windows(ctx, views, stack, None, block, &mut w.blocks)?;
        for (spectra, win) in w.spectra.iter().zip(&mut w.deeper) {
            self.sample_windows(ctx, views, stack, Some(spectra), block, win)?;
        }
        let deeper = &w.deeper;
        Ok((
            &w.views,
            Sliced {
                first: &w.blocks,
                deeper,
            },
        ))
    }
}

impl<'b, K> Modelled<'b, K> {
    /// The model the passes that compare with the counts read, its shift none while the
    /// lend samples the object ([`Self::moves`]).
    pub fn model(&self) -> Model<'b> {
        self.model
    }

    /// What the lend's position model moves, as [`FeaturesMut::lend`] took it from one
    /// [`Mover`], which gives the shift or the sampler, never both.
    pub fn moves(&self) -> Moves<'b> {
        match (self.model.shift, &self.sampled) {
            (Some(shift), _) => Moves::Probe(shift),
            (None, Some(_)) => Moves::Object,
            (None, None) => Moves::Nothing,
        }
    }
}

impl<K> Modelled<'_, K> {
    /// Under [`Sampling::FullObject`], the spectra of the object planes `obj` into `out`,
    /// which [`Self::windows_of`] samples from, by the lend's sampler
    /// ([`ObjectSampler::spectra`], one transform); an error otherwise.
    pub fn spectra_of(
        &mut self,
        ctx: &ExecutionContext,
        obj: &DeviceBuffer<C32>,
        out: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        let s = self.sampled.as_mut().ok_or_else(another_model)?;
        s.sampler.spectra(ctx, obj, out)
    }

    /// [`Modelled::windows_of`] for a lend of any slice count, whose `None` reads the slice
    /// the lend took.
    fn sample_windows(
        &mut self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        spectra: Option<&DeviceBuffer<C32>>,
        block: WindowBlock,
        out: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        let s = self.sampled.as_mut().ok_or_else(another_model)?;
        let at = StackPoints {
            views,
            stack,
            offsets: s.offsets,
        };
        s.sampler
            .windows(ctx, at, spectra.unwrap_or(s.spectra), block, out)
    }

    /// The far sums of `llk` and of every stepped feature.
    pub fn far<'c>(&'c mut self, llk: Option<&'c mut DeviceBuffer<LlkPartial>>) -> FarSums<'c> {
        FarSums {
            llk,
            scale: self.scale.as_deref_mut(),
            background: self.background.as_deref_mut(),
        }
    }

    /// A fused pass's sums of the position model: `probe`, the shifted probe's sums
    /// [`FeaturesMut::lend_shifted`] lent, and with `position` each frame's position sums if
    /// the offsets are refined. Position sums without a moved probe fail the pass, which takes
    /// them from the shift alone.
    pub fn shift_sums<'c>(
        &'c mut self,
        probe: Option<ShiftProbe<'c>>,
        position: bool,
    ) -> ShiftSums<'c> {
        let partials = (self.position.as_mut()).and_then(|p| p.partials.as_deref_mut());
        let (num, den) = match partials.filter(|_| position) {
            Some(q) => (Some(&mut q.num), Some(&mut q.den)),
            None => (None, None),
        };
        ShiftSums {
            probe,
            position: num,
            position_den: den,
        }
    }

    /// Whether the model moves the probe, whose update sums
    /// [`FeaturesMut::lend_shifted`] then lends.
    pub fn shifted(&self) -> bool {
        self.model.shift.is_some()
    }
}

impl FeaturesMut<'_> {
    /// Whether some refined feature takes a step from far sums.
    pub fn stepped(&self) -> bool {
        (self.scale.as_ref()).is_some_and(|s| s.sums.is_some())
            || (self.background.as_ref()).is_some_and(|b| b.sums.is_some())
    }

    /// Whether the position is modelled and refined.
    pub fn position_refined(&self) -> bool {
        (self.position.as_ref()).is_some_and(|p| p.partials.is_some())
    }

    /// `step`, a cycle's own buffers of the offsets' step, as the position's refine asks for
    /// it: an error when the cycle's state was built for the other, a step without it or none
    /// with it.
    pub fn own_step<'s, T>(&self, step: Option<&'s mut T>) -> Result<Option<&'s mut T>, GpuError> {
        match step.is_some() == self.position_refined() {
            true => Ok(step),
            false => Err(another_model()),
        }
    }

    /// Whether the position model moves the probe, as [`Modelled::shifted`] of a lend.
    pub fn shifted(&self) -> bool {
        (self.position.as_ref()).is_some_and(|p| p.mover.moves_probe())
    }

    /// Whether the position model samples the object's windows ([`Sampling::FullObject`]).
    pub fn samples_object(&self) -> bool {
        (self.position.as_ref()).is_some_and(|p| matches!(*p.mover, Mover::Object { .. }))
    }

    /// The per-frame intensity scale, if modelled.
    pub fn scale(&self) -> Option<&DeviceBuffer<f32>> {
        self.scale.as_ref().map(|s| &*s.values)
    }

    /// The features lent to the passes of one cycle body over `obj` and `probe`, which
    /// cannot change while they are: the model of every feature set, the position's shift
    /// after the spectra of `probe` it moves (two launches) if it moves the probe, or the
    /// sampler after the spectra of `obj` (one transform) if it samples the object, and the
    /// propagator's stage `exit`, and the sums. It reads `obj` as one slice of any count;
    /// [`Modelled::one_slice`] narrows it to read `obj` as the whole object.
    pub fn lend<'b>(
        &'b mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        dims: &ViewDims,
        obj: &DeviceBuffer<C32>,
        probe: &'b DeviceBuffer<C32>,
        exit: ExitPass<'b>,
    ) -> Result<Modelled<'b, AnySlices>, GpuError> {
        Ok(self.lend_sums(ctx, kernels, dims, obj, probe, exit)?.0)
    }

    /// [`Self::lend`] with the shifted probe's sums of kind `S` if the model moves the probe:
    /// the one accumulator the passes add to and [`Kernels::shifted_probe_sums`] consumes.
    pub fn lend_shifted<'b, S: ShiftedSums>(
        &'b mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        dims: &ViewDims,
        obj: &DeviceBuffer<C32>,
        probe: &'b DeviceBuffer<C32>,
        exit: ExitPass<'b>,
    ) -> Result<(Modelled<'b, AnySlices>, Option<&'b mut S>), GpuError> {
        let (m, sums) = self.lend_sums(ctx, kernels, dims, obj, probe, exit)?;
        Ok((m, sums.map(ShiftedProbes::get::<S>)))
    }

    /// [`Self::lend`] with the shifted probe's sums of every kind.
    fn lend_sums<'b>(
        &'b mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        dims: &ViewDims,
        obj: &DeviceBuffer<C32>,
        probe: &'b DeviceBuffer<C32>,
        exit: ExitPass<'b>,
    ) -> Result<(Modelled<'b, AnySlices>, Option<&'b mut ShiftedProbes>), GpuError> {
        let (scale, scale_sums) = match &mut self.scale {
            Some(s) => (Some(&*s.values), s.sums.as_deref_mut()),
            None => (None, None),
        };
        let (background, background_sums) = match &mut self.background {
            Some(b) => (Some(&*b.values), b.sums.as_deref_mut()),
            None => (None, None),
        };
        let (shift, sampled, position, probe_sums) = match &mut self.position {
            Some(p) => {
                let (shift, sampled, sums) = match &mut *p.mover {
                    Mover::Probe {
                        spectrum,
                        probe_sums,
                    } => {
                        let shift = Shift {
                            offsets: &*p.offsets,
                            spectrum: kernels.probe_spectrum(ctx, dims, probe, spectrum)?,
                        };
                        (Some(shift), None, Some(&mut **probe_sums))
                    }
                    Mover::Object {
                        sampler, spectra, ..
                    } => {
                        sampler.spectra(ctx, obj, spectra)?;
                        let sampled = Sampled {
                            sampler,
                            spectra: &*spectra,
                            offsets: &*p.offsets,
                        };
                        (None, Some(sampled), None)
                    }
                };
                let partials = p.partials.as_deref_mut();
                (shift, sampled, Some(PositionSums { partials }), sums)
            }
            None => (None, None, None, None),
        };
        let m = Modelled {
            model: Model {
                scale,
                background,
                shift,
                exit: Some(exit),
            },
            sampled,
            scale: scale_sums,
            background: background_sums,
            position,
            slices: PhantomData,
        };
        Ok((m, probe_sums))
    }

    /// Every refined feature's step from the far sums [`Modelled::far`] lent the passes
    /// before.
    pub fn step(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        grad: &GradKernels,
    ) -> Result<(), GpuError> {
        if let Some(ScaleMut {
            values,
            sums: Some(sums),
        }) = &mut self.scale
        {
            grad.scale_step(ctx, values, sums)?;
        }
        if let Some(BackgroundMut {
            values,
            sums: Some(sums),
            nobs,
            ..
        }) = &mut self.background
        {
            kernels.background_ap(ctx, values, sums, nobs)?;
        }
        Ok(())
    }

    /// Under [`Sampling::FullObject`], the object update of `obj` from `sums` by the normal
    /// equations of `probe`'s windows over the scan's `views` at the offsets
    /// ([`ObjectSampler::solve`], [`OBJ_PCG`] steps), the object's spectra its scratch: `true`
    /// then, `false` with nothing touched otherwise; an error if [`FeatureBufs::build_solve`]
    /// has not built the solve.
    pub fn solve_obj(
        &mut self,
        ctx: &ExecutionContext,
        views: &Views,
        probe: &DeviceBuffer<C32>,
        obj: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        inertia: f32,
    ) -> Result<bool, GpuError> {
        let Some(p) = &mut self.position else {
            return Ok(false);
        };
        let Mover::Object {
            sampler,
            spectra,
            solve,
        } = &mut *p.mover
        else {
            return Ok(false);
        };
        let Some(solve) = solve else {
            return Err(GpuError::Shape("object solve not built".into()));
        };
        let a = Normal {
            views,
            offsets: p.offsets,
            probe,
        };
        let held = (inertia, OBJ_HELD as f32);
        sampler.solve::<OBJ_PCG>(ctx, (solve, spectra), a, obj, sums, held)?;
        Ok(true)
    }

    /// The position step from the position sums the AP projections wrote, after the
    /// updates; nothing without a refined position.
    pub fn position_step(
        &mut self,
        ctx: &ExecutionContext,
        grad: &GradKernels,
    ) -> Result<(), GpuError> {
        if let Some(PositionMut {
            offsets,
            partials: Some(q),
            ..
        }) = &mut self.position
        {
            let limits = [POSITION_STEP as f32, POSITION_LIMIT as f32];
            grad.position_step(ctx, offsets, [&q.num, &q.den], limits)?;
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
    /// AP's, the LLK pass's, and ML's gradient's, position sums' and opening scale's stacks.
    cut: Cut,
    passes: Passes,
    /// One stack's LLK partials, on their way into the ring.
    llk: DeviceBuffer<LlkPartial>,
    /// The chunk sums M1's [`Kernels::llk`] reduces a stack's frames through.
    llk_scratch: LlkScratch,
    /// The windows of the cut's stacks under [`PositionScratch::Windows`].
    windows: Option<Windows>,
}

impl Shared {
    /// M2's fused passes (with [`PositionScratch::Shift`], the position model's) or M1's
    /// scratch, `fused` picking, cut by `rule` with `scratch` and, past one of `slices`, a
    /// multislice object's [`Slices::waves`] and under [`PositionScratch::Windows`] its
    /// windows.
    fn new(
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        rule: StackRule,
        fused: bool,
        scratch: PositionScratch,
        slices: usize,
    ) -> Result<Self, EngineError> {
        // M1's scratch: `obj_probe_to_psi` (0), the transforms and `apply_amplitude` (to 3);
        // the fused work planes, first: the row (0), column and inverse row passes and the sums
        // over the waves (3). Under the position's shift `shift_pre` writes `T`, and for a refined
        // position's sums `T_y` into the adjoint planes, first (0); the inverse row pass (3) reads
        // both and writes the adjoint, object planes and power, `shift_post` and the probe sums
        // take the adjoint (4), `gather_shifted` the rest (5). Under its windows, as AP runs a
        // stack: the sampler's calls, three launches each, write the windows (0 to 2) and,
        // refined, their derivatives over `y` (3 to 5) and `x` (6 to 8); a memset zeroes the
        // window numerator (9), a block's size, which the fused passes (10 to 13) add to; a
        // refined position's sums (14) read the work planes and the three blocks, the probe sums
        // (15) the work planes and the windows; placing the numerator back takes six launches
        // (16 to 21). The LLK pass, ML's gradient and position sums and DM's and RAAR's updates
        // run a subset. Of a multislice object, the two waves its slices pass on, numbered
        // after; under the windows, as the LLK pass runs a stack, the sampler's calls write each
        // slice's windows in turn (from 22), which the slice passes after them read beside the
        // waves.
        let shift = matches!(scratch, PositionScratch::Shift { .. });
        let wave = wave_bytes(dims);
        let mut streamed = match scratch {
            PositionScratch::None => vec![Streamed::new(wave, 0..=3)],
            PositionScratch::Shift { refined } => {
                let probe = plane_bytes::<C32>(dims, dims.nprobe);
                let obj = plane_bytes::<C32>(dims, dims.nobj) + plane_bytes::<f32>(dims, 1);
                let adjoint = if refined { 0 } else { 3 };
                vec![
                    Streamed::new(wave, 1..=3),
                    Streamed::new(probe, 0..=3),
                    Streamed::new(probe, adjoint..=4),
                    Streamed::new(obj, 3..=5),
                ]
            }
            PositionScratch::Windows {
                sample,
                place,
                refined,
            } => {
                let block = Windows::bytes(dims);
                let calls: &[_] = if refined { &[3..=5, 6..=8] } else { &[] };
                let derivatives = (calls.iter())
                    .flat_map(|c| Streamed::chunked(sample, c.clone(), &[(block, 14)], &[]));
                [
                    vec![Streamed::new(wave, 10..=15)],
                    Streamed::chunked(sample, 0..=2, &[(block, 15)], &[]),
                    derivatives.collect(),
                    Streamed::chunked(place, 16..=21, &[], &[(block, 9)]),
                ]
                .concat()
            }
        };
        if slices > 1 {
            streamed.extend(match scratch {
                PositionScratch::Windows { sample, .. } => {
                    let (block, n) = (Windows::bytes(dims), to_u32("slices", slices)?);
                    let pass = 22 + 3 * n;
                    let sampled = (0..n).flat_map(|k| {
                        let call = 22 + 3 * k..=24 + 3 * k;
                        Streamed::chunked(sample, call, &[(block, pass + 1)], &[])
                    });
                    (sampled.chain([Streamed::new(wave, pass..=pass + 1); 2])).collect()
                }
                PositionScratch::None | PositionScratch::Shift { .. } => {
                    vec![Streamed::new(wave, 6..=7); 2]
                }
            });
        }
        let cut = rule.cut(stream, dims, &streamed)?;
        let windows = match scratch {
            PositionScratch::Windows { refined, .. } => {
                Some(Windows::new(stream, dims, &cut, slices, refined)?)
            }
            PositionScratch::None | PositionScratch::Shift { .. } => None,
        };
        let nf_max = cut.table.nf_max();
        let passes = if fused {
            let f = match shift {
                true => FusedBufs::with_shift(stream, dims, nf_max),
                false => FusedBufs::new(stream, dims, nf_max),
            };
            let f = f.map_err(|e| match e {
                GpuError::Shape(m) if shift => EngineError::Invalid(format!("position model: {m}")),
                GpuError::Shape(m) => EngineError::Invalid(format!(
                    "{m}; GpuOptions {{ fused: false, .. }} runs every algorithm but LSQML by \
                     M1's launches instead"
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
            cut,
            passes,
            windows,
        })
    }
}

/// DM's or RAAR's stacks and the exit waves each carries across cycles, with RAAR's iterate,
/// cut by the carried exit waves and the shared scratch; the iterate, which one pass reads and
/// writes, does not count. Cycle 0 of a call overwrites both sets, so nothing is carried from
/// one call to the next. Under [`PositionScratch::Windows`], the windows of its stacks.
struct CarriedBufs {
    cut: Cut,
    sets: Sets,
    windows: Option<Windows>,
}

/// One stack's carried waves for the fused passes, in either format.
type AnyWaves = Box<dyn Waves + Send + Sync>;

/// Each stack's exit waves, planes past `nf * nmode` zero, and RAAR's iterate laid out alike.
enum Sets {
    /// M1's: f32, which its cuFFT plans, one per stack length of the cut, transform in place.
    M1 {
        plans: Vec<Plan2d>,
        psi: Vec<DeviceBuffer<C32>>,
        iterate: Option<Vec<DeviceBuffer<C32>>>,
    },
    /// The fused passes': f32, or [`HalfWaves`] under [`GpuOptions::psi_f16`].
    Fused {
        psi: Vec<AnyWaves>,
        iterate: Option<Vec<AnyWaves>>,
    },
}

impl CarriedBufs {
    /// DM's, or with `raar` RAAR's, against `shared`; for the fused passes [`HalfWaves`] at
    /// `half` if some.
    fn new(
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        rule: StackRule,
        shared: &Shared,
        raar: bool,
        half: Option<f32>,
    ) -> Result<Self, EngineError> {
        // The exit waves: read by the forward pass, written by the inverse one, the passes of
        // the work planes (M1: also `dm_reflect`, `add_c32`); the iterate: the forward row
        // pass alone (M1: `raar_iterate`). `HalfWaves` hold a `u32` per value.
        let wave = match half {
            Some(_) => plane_bytes::<u32>(dims, dims.nmode),
            None => wave_bytes(dims),
        };
        let work = shared.streamed[0];
        let mut streamed = shared.streamed.clone();
        streamed.push(Streamed {
            bytes: wave,
            ..work
        });
        if raar {
            streamed.push(Streamed::new(wave, work.first..=work.first));
        }
        let cut = rule.cut(stream, dims, &streamed)?;
        let len = cut.wave_len(dims);
        let full = || DeviceBuffer::<C32>::zeroed(stream, len);
        let sets = match shared.passes {
            Passes::M1 { .. } => {
                let set = || {
                    (0..cut.len())
                        .map(|_| full())
                        .collect::<Result<Vec<_>, _>>()
                };
                Sets::M1 {
                    plans: cut.plans(stream, dims)?,
                    psi: set()?,
                    iterate: raar.then(set).transpose()?,
                }
            }
            Passes::Fused(_) => {
                let one = || -> Result<AnyWaves, GpuError> {
                    Ok(match half {
                        None => Box::new(full()?),
                        Some(scale) => Box::new(HalfWaves::zeroed(stream, len, scale)?),
                    })
                };
                let set = || (0..cut.len()).map(|_| one()).collect::<Result<Vec<_>, _>>();
                Sets::Fused {
                    psi: set()?,
                    iterate: raar.then(set).transpose()?,
                }
            }
        };
        let windows = (shared.windows.as_ref())
            .map(|w| Windows::new(stream, dims, &cut, 1 + w.deeper.len(), w.derivatives()))
            .transpose()?;
        Ok(Self { cut, sets, windows })
    }

    /// Whether these are the buffers `epoch`'s cycle binds.
    fn serves(&self, epoch: Epoch) -> bool {
        let raar = match &self.sets {
            Sets::M1 { iterate, .. } => iterate.is_some(),
            Sets::Fused { iterate, .. } => iterate.is_some(),
        };
        match epoch {
            Epoch::Dm { .. } => !raar,
            Epoch::Raar { .. } => raar,
            _ => false,
        }
    }

    fn lend(&mut self) -> CarriedMut<'_> {
        CarriedMut {
            cut: &self.cut,
            sets: &mut self.sets,
            windows: self.windows.as_mut(),
        }
    }
}

/// [`CarriedBufs`] lent to a cycle.
pub(super) struct CarriedMut<'a> {
    pub cut: &'a Cut,
    sets: &'a mut Sets,
    windows: Option<&'a mut Windows>,
}

impl CarriedMut<'_> {
    /// Stack `s`'s carried exit waves.
    pub fn psi(&self, s: usize) -> &dyn Waves {
        match &*self.sets {
            Sets::M1 { psi, .. } => &psi[s],
            Sets::Fused { psi, .. } => &*psi[s],
        }
    }

    /// The fused passes' exit waves of stack `s`, RAAR's iterate if the sets are RAAR's, and
    /// the cut's windows under [`PositionScratch::Windows`].
    pub fn fused(
        &mut self,
        s: usize,
    ) -> (&mut dyn Waves, Option<&mut dyn Waves>, Option<&mut Windows>) {
        let Sets::Fused { psi, iterate } = &mut *self.sets else {
            unreachable!("CarriedBufs::new builds the fused passes' sets for theirs");
        };
        (
            &mut *psi[s],
            iterate.as_mut().map(|i| &mut *i[s] as &mut dyn Waves),
            self.windows.as_deref_mut(),
        )
    }

    /// M1's plan for stack `s`, its exit waves, and RAAR's iterate if the sets are RAAR's.
    pub fn m1(
        &mut self,
        s: usize,
    ) -> (
        &mut Plan2d,
        &mut DeviceBuffer<C32>,
        Option<&mut DeviceBuffer<C32>>,
    ) {
        let Sets::M1 {
            plans,
            psi,
            iterate,
        } = &mut *self.sets
        else {
            unreachable!("CarriedBufs::new builds M1's sets for M1's passes");
        };
        let plan = &mut plans[self.cut.frames[s].plan];
        (plan, &mut psi[s], iterate.as_mut().map(|i| &mut i[s]))
    }
}

/// A cycle with the buffers only it uses, and with graphs the graph captured for it.
struct EpochRun {
    cycle: Cycle,
    graph: Option<Captured>,
}

/// A graph and, if its body ended a record when captured, the [`Ended`] each replay counts.
struct Captured {
    graph: CudaGraph<()>,
    ended: Option<Ended>,
}

impl Captured {
    /// Enqueues the graph on `ctx`'s stream, counting the record it ends, if any, in `ring`
    /// under `tag`.
    fn replay(
        &self,
        ring: &mut LlkRing<Tag>,
        ctx: &ExecutionContext,
        nframe: u32,
        tag: Tag,
    ) -> Result<(), EngineError> {
        // SAFETY: the context holds the replay's leases on every captured buffer until the
        // stream has run it, so no other stream can touch them meanwhile.
        let launch = || Ok(unsafe { self.graph.launch().execute(ctx) }.map_err(GpuError::from)?);
        match &self.ended {
            Some(ended) => ring.replay(ctx, ended, tag, nframe, launch),
            None => launch(),
        }
    }
}

/// Whether the passes replay captured graphs ([`GpuOptions::graph`]).
enum Graphs {
    /// Every pass runs eagerly.
    Off,
    /// Every pass replays a graph. The LLK pass's lives with the device state: none only while
    /// its capture is pending, from a replacement of the buffers it binds until the next LLK
    /// pass captures it.
    On { llk: Option<Captured> },
}

impl Graphs {
    /// Leaves the LLK pass's capture pending, as the buffers its graph binds are replaced.
    fn pend_llk(&mut self) {
        if let Self::On { llk } = self {
            *llk = None;
        }
    }
}

/// A cycle's [`Epoch`] and the buffers only it uses, built together, so a graph captured
/// from it bakes in sizes that match its buffers.
enum Cycle {
    Ap {
        params: ApParams,
        llk: bool,
        one: OneSlice,
    },
    /// Its buffers are the device's [`CarriedBufs`], which every DM epoch shares.
    Dm {
        params: DmParams,
        first: bool,
        one: OneSlice,
    },
    /// Its buffers are the device's [`CarriedBufs`], which every RAAR epoch shares.
    Raar {
        params: RaarParams,
        first: bool,
        one: OneSlice,
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
        opr: Option<OprShape>,
    },
}

impl Cycle {
    /// Whether each cycle ends an LLK record.
    fn records(&self) -> bool {
        matches!(
            self,
            Self::Ap { llk: true, .. } | Self::Ml { llk: true, .. }
        )
    }

    fn epoch(&self) -> Epoch {
        match *self {
            Self::Ap { params, llk, .. } => Epoch::Ap { params, llk },
            Self::Dm { params, first, .. } => Epoch::Dm { params, first },
            Self::Raar { params, first, .. } => Epoch::Raar { params, first },
            Self::Rpie(p, _) => Epoch::Rpie(p),
            Self::Ml { params, first, llk } => Epoch::Ml { params, first, llk },
            Self::Bh { params, cg } => Epoch::Bh { params, cg },
            Self::Lsqml { params, pre, opr } => Epoch::Lsqml { params, pre, opr },
        }
    }
}

/// ML's conjugate-gradient state and line-search scratch, sized by the device and the
/// position model alone, so one state serves the graphs of every ML epoch of that model; each
/// call's first epoch resets the state. ML runs an object of `one` slice.
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
    /// What the line search adds under the position model it was built for.
    position: Option<MlPosition>,
    one: OneSlice,
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
    pub position: Option<&'a mut MlPosition>,
    pub one: OneSlice,
}

impl MlState {
    /// For frames of `dims` cut by `rule`, arrays of `lens = [object, probe]` values and what
    /// `position`, the position model's, adds to the line search.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        rule: StackRule,
        lens: [usize; 2],
        position: PositionScratch,
        one: OneSlice,
    ) -> Result<Self, EngineError> {
        // The fields po, a, b: `gamma_fields` (0), the transforms, `gamma_frames` (4). Under
        // the moved probe those of P and dP (`shifted_probe` 0 and 2, `shift_w` 1 and 3) give
        // po, a (4, 5) and b (6, and 8 after `add_c32` 7) by `obj_wave_to_psi`, then the
        // transforms (9 to 11) and `gamma_frames` (12). Under the windows those of O (0 to 2,
        // from the lend's spectra) and of dO (3 to 5, from dO's) give the fields by
        // `gamma_fields` (6), then the transforms (7 to 9) and `gamma_frames` (10).
        let wave = wave_bytes(dims);
        let streamed = match position {
            PositionScratch::None => vec![Streamed::new(wave, 0..=4); 3],
            PositionScratch::Shift { .. } => {
                let probe = plane_bytes::<C32>(dims, dims.nprobe);
                vec![
                    Streamed::new(probe, 0..=5),
                    Streamed::new(probe, 2..=8),
                    Streamed::new(wave, 4..=12),
                    Streamed::new(wave, 5..=12),
                    Streamed::new(wave, 6..=12),
                ]
            }
            PositionScratch::Windows { sample, .. } => {
                let block = plane_bytes::<C32>(dims, dims.nobj);
                let mut streamed = Streamed::chunked(sample, 0..=2, &[(block, 6)], &[]);
                streamed.extend(Streamed::chunked(sample, 3..=5, &[(block, 6)], &[]));
                streamed.extend([Streamed::new(wave, 6..=10); 3]);
                streamed
            }
        };
        let cut = rule.cut(stream, dims, &streamed)?;
        let position = match position {
            PositionScratch::None => None,
            PositionScratch::Shift { .. } => Some(MlPosition::Moved(Box::new(MlMoved::new(
                stream,
                kernels,
                dims,
                cut.table.nf_max(),
            )?))),
            PositionScratch::Windows { .. } => Some(MlPosition::Sampled(Box::new(MlWindows::new(
                stream, dims, &cut, lens[0],
            )?))),
        };
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
            position,
            cut,
            one,
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
            position: self.position.as_mut(),
            one: self.one,
        }
    }

    /// Whether what the line search adds is `position`'s.
    fn fits(&self, position: PositionScratch) -> bool {
        matches!(
            (&self.position, position),
            (None, PositionScratch::None)
                | (Some(MlPosition::Moved(_)), PositionScratch::Shift { .. })
                | (
                    Some(MlPosition::Sampled(_)),
                    PositionScratch::Windows { .. }
                )
        )
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
    /// One step's partial sums.
    parts: BhParts,
    /// The epoch's step records, one per batch.
    steps: DeviceBuffer<f32>,
    /// The state of the position model, if modelled.
    moved: Option<BhMoved>,
    /// A multislice object's waves and slice sums, if the device's has more than one slice.
    sliced: Option<BhSliced>,
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
    pub parts: &'a mut BhParts,
    pub steps: &'a mut DeviceBuffer<f32>,
    pub moved: Option<&'a mut BhMoved>,
    pub sliced: Option<&'a mut BhSliced>,
}

impl BhState {
    /// For `views` in batches of `size` frames, each cut into stacks by `rule`, through an
    /// object of `deeper` slices past the first; under the position model, `scratch` holds
    /// the frames per stack of the shared scratch, whose shift planes its sums borrow, which
    /// caps its stacks, the device's sampling and whether the offsets are refined. The epoch
    /// order is the scan's.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        size: u32,
        rule: StackRule,
        deeper: usize,
        scratch: Option<MovedScratch>,
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        let frames = rule.frames(&Self::streamed(dims, deeper, scratch));
        // Only the shift's sums borrow the shared scratch.
        let frames = match scratch {
            Some(
                s @ MovedScratch {
                    sampling: MovedSampling::Patch,
                    ..
                },
            ) => frames.min(s.frames),
            _ => frames,
        };
        let (cut, batches) = batch_stacks(stream, dims, size, frames)?;
        let len = cut.wave_len(dims);
        let plans = cut.plans(stream, dims)?;
        let slots = batches.iter().map(|b| b.len()).max().unwrap_or(0);
        let moved = (scratch
            .map(|s| BhMoved::new(stream, kernels, dims, &cut, slots, (s, deeper))))
        .transpose()?;
        let sliced = (deeper > 0)
            .then(|| BhSliced::new(stream, dims, len, slots, deeper))
            .transpose()?;
        let Cut {
            frames: stacks,
            table,
            ..
        } = cut;
        let zeroed = |len| DeviceBuffer::<C32>::zeroed(stream, len);
        Ok(Self {
            size,
            plans,
            psi: (0..slots).map(|_| zeroed(len)).collect::<Result<_, _>>()?,
            r: (0..slots).map(|_| zeroed(len)).collect::<Result<_, _>>()?,
            lu: zeroed(len)?,
            lv: zeroed(len)?,
            prev: [
                zeroed(dims.obj_len())?,
                zeroed(dims.nprobe as usize * dims.n())?,
            ],
            order: EpochOrder::new(kernels, stream, views)?,
            parts: BhParts::zeroed(stream, dims, deeper)?,
            steps: DeviceBuffer::zeroed(stream, BH_STEP as usize * batches.len())?,
            batches,
            stacks,
            table,
            moved,
            sliced,
        })
    }

    /// The buffers of a stack's two loops through an object of `deeper` slices past the first,
    /// under `scratch`'s position model if any, by the passes that touch them.
    fn streamed(dims: &ViewDims, deeper: usize, scratch: Option<MovedScratch>) -> Vec<Streamed> {
        // The batch's two loops in turn. Without the position: psi (`obj_probe_to_psi` 0, the
        // transform, the residual 2), r (the adjoint, the sums 4), lu and lv (`far` 5, the
        // transforms, `hess_frames` 8). With it: the moved probes (`shifted_probe` 0,
        // `shift_w`, `obj_wave_to_psi` 2), psi (the transform, the residual 4), r (the
        // adjoint, `shifted_psi_sums` from 6, `obj_conj` 11), y (the transform, `pos_grad`
        // 13); then lu and lv (`pos_spec` 14, the inverses, `pos_near`, the transforms,
        // `pos_far` 22), r (`pos_near` 17, `obj_conj` 20) and y (the transform, `pos_far`).
        // Under full-object sampling: the estimate's windows (three sampler calls from 0, 3, 6
        // writing blocks 0 to 2), psi (`obj_probe_to_psi` 9, the transform, the residual 11),
        // r (the adjoint, `psi_to_probe` 13, `win_grad` 14, which reads blocks 1 and 2), the
        // patches (`win_grad`, `place` 15 to 20: zeroing, a chunk's embed 16, transform and
        // accumulate 18, the inverse and the add); then the windows of `u_O` and `v_O` (six
        // sampler calls from 21), `win_lift` 39, lu and lv (`far` 40, the transforms,
        // `hess_frames` 43), the second derivatives (three sampler calls from 44) and r and
        // the derivatives' windows to `win_hess` 53. Through a multislice object, two slices: the
        // wave entering the second (`slice_mul` 0, the gap, `slice_mul` 2, `slice_pair` 8), psi
        // (the phase, the transform, the residual 5), r (the adjoint, the phase's, `slice_pair`,
        // `slice_adjoint` 10), the work wave from `slice_pair` to `psi_to_obj` 9 and from
        // `slice_adjoint` (the gap) to `psi_to_obj` and `psi_to_probe` 13; then lu, lv and the
        // three second derivatives from `slice_tangent` 14 through the gaps and the second's
        // `slice_tangent` 20, the phases and the transforms to `sliced_hess_frames` 25. Under
        // full-object sampling as well, through two slices: a slice's window (sampler call 0, 5)
        // ahead of its wave (`slice_mul` 3, 8; the gap 4), psi (the transform, the residual 10), r
        // (the adjoint, `slice_pair` 12, `slice_adjoint` 26), then per slice from the last its
        // derivatives' windows (13 and 16, 29 and 32) to `win_grad` (19, 35), whose patches
        // `place` takes (20, 36), and the work wave from `slice_pair` to `win_grad` and from
        // `slice_adjoint` 26 to the first slice's `psi_to_probe` 28 and `win_grad`; then per slice
        // the windows of `u_O` and `v_O` (42 and 45, 55 and 58) to `win_lift` and `slice_tangent`
        // (49, 62), the second's window and entering wave read there again by `slice_adjoint` 89
        // and `slice_pair` 66, lu, lv and the three second derivatives from 49 to
        // `sliced_hess_frames` 65, and per slice from the last its derivatives' windows (four
        // calls from 67, 91) and second derivatives (three from 79, 103) to `win_hess` (88,
        // 112), r to `slice_adjoint` 89, the work wave from `slice_pair` 66 and from it to the
        // first slice's `win_hess`. With the moved probes through two slices: the moved probes
        // (`shifted_probe` 0, `shift_w`, `slice_mul` 2), the wave entering the second slice
        // (the gap 3, `slice_mul` 4, `phase_grad` 10, `slice_pair` 11), psi (the phase 5, the
        // transform, the residual 7), r (the adjoint, the phase's, `phase_grad`, `slice_pair`,
        // `slice_adjoint` 13), the work wave from `slice_pair` to `psi_to_obj` 12 and from
        // `slice_adjoint` (the gap) to `shifted_psi_sums` from 15 and `obj_conj` 20, y (the
        // transform, `pos_grad` 22); then lu and lv (`pos_spec` 23, the inverses, the
        // slices' `slice_tangent` 30 and 36 with the gaps between, `phase_tangent` 37, the
        // phase, the transforms, `sliced_hess_frames` 42), the three second derivatives alike
        // from `pos_second` 24, and the entering wave again from 36 to 37. A slice past the
        // second repeats the second's passes and holds its entering wave, and under full-object
        // sampling its window, to the backward sweep: each is booked across the second's passes
        // from its first forward one to past its last backward one (1 to 12; with the moved probes
        // 3 to 15; under full-object sampling 4 to 28 and, refined, through the second loop's 50 to
        // 91), so a boundary books the most that any slice's turn at it holds. Held, the offsets'
        // passes drop out and the others keep their numbers: r ends at `shifted_psi_sums`' 7, and
        // lu and lv at `moved_far`, in `pos_far`'s place right after their transforms (20); under
        // full-object sampling only block 0's windows are sampled, `win_obj_grad` takes
        // `win_grad`'s place, and `win_lift`, the second derivatives and `win_hess` drop out,
        // through a multislice object everything from 66; with the moved probes through a
        // multislice object the work wave ends at `shifted_psi_sums`' 16, `moved_spec` and the
        // second derivatives' zeroing take `pos_spec`'s and `pos_second`'s places, and the
        // entering wave is not read again.
        let wave = wave_bytes(dims);
        let refined = scratch.is_some_and(|s| s.refined);
        // The slices past the second.
        let more = deeper.saturating_sub(1);
        match scratch.map(|s| s.sampling) {
            None if deeper > 0 => {
                let mut s = vec![
                    Streamed::new(wave, 0..=8),
                    Streamed::new(wave, 2..=5),
                    Streamed::new(wave, 5..=10),
                    Streamed::new(wave, 8..=9),
                    Streamed::new(wave, 10..=13),
                ];
                s.extend(vec![Streamed::new(wave, 1..=12); more]);
                s.extend([Streamed::new(wave, 14..=25); 5]);
                s
            }
            None => vec![
                Streamed::new(wave, 0..=2),
                Streamed::new(wave, 2..=4),
                Streamed::new(wave, 5..=8),
                Streamed::new(wave, 5..=8),
            ],
            Some(MovedSampling::FullObject { held, scratch, acc }) if deeper > 0 => {
                let win = Windows::bytes(dims);
                let calls: &[_] = if refined {
                    &[
                        (0, 28),
                        (5, 26),
                        (13, 19),
                        (16, 19),
                        (29, 35),
                        (32, 35),
                        (42, 49),
                        (45, 49),
                        (55, 62),
                        (58, 62),
                        (67, 88),
                        (70, 88),
                        (73, 88),
                        (76, 88),
                        (79, 88),
                        (82, 88),
                        (85, 88),
                        (91, 112),
                        (94, 112),
                        (97, 112),
                        (100, 112),
                        (103, 112),
                        (106, 112),
                        (109, 112),
                    ]
                } else {
                    &[(0, 28), (5, 26), (42, 49), (45, 49), (55, 62), (58, 62)]
                };
                let mut s: Vec<_> = (calls.iter())
                    .flat_map(|&(p, read)| Streamed::chunked(held, p..=p + 2, &[(win, read)], &[]))
                    .collect();
                for p in [20, 36] {
                    s.extend(Streamed::chunked(acc, p..=p + 5, &[], &[(win, p - 1)]));
                    s.push(Streamed::fixed(scratch, p + 1..=p + 3));
                }
                s.extend([
                    Streamed::new(wave, 3..=12),
                    Streamed::new(wave, 8..=10),
                    Streamed::new(wave, 10..=26),
                    Streamed::new(wave, 12..=19),
                    Streamed::new(wave, 26..=35),
                ]);
                s.extend(vec![Streamed::new(wave + win, 4..=28); more]);
                if refined {
                    s.extend([
                        Streamed::new(win, 62..=89),
                        Streamed::new(wave, 62..=66),
                        Streamed::new(wave, 65..=89),
                        Streamed::new(wave, 66..=88),
                        Streamed::new(wave, 89..=112),
                    ]);
                    s.extend(vec![Streamed::new(wave + win, 50..=91); more]);
                }
                s.extend([Streamed::new(wave, 49..=65); 5]);
                s
            }
            Some(MovedSampling::FullObject { held, scratch, acc }) => {
                let win = Windows::bytes(dims);
                // Each sampler call's first pass, and the last to read the block it writes.
                let calls: &[_] = if refined {
                    &[
                        (0, 13),
                        (3, 14),
                        (6, 14),
                        (21, 43),
                        (24, 53),
                        (27, 53),
                        (30, 43),
                        (33, 53),
                        (36, 53),
                        (44, 53),
                        (47, 53),
                        (50, 53),
                    ]
                } else {
                    &[(0, 13), (21, 43), (30, 43)]
                };
                let mut s: Vec<_> = (calls.iter())
                    .flat_map(|&(p, read)| Streamed::chunked(held, p..=p + 2, &[(win, read)], &[]))
                    .collect();
                s.extend(Streamed::chunked(acc, 15..=20, &[], &[(win, 14)]));
                s.extend([
                    Streamed::fixed(scratch, 16..=18),
                    Streamed::new(wave, 9..=11),
                    Streamed::new(wave, 11..=14),
                    Streamed::new(wave, 40..=43),
                    Streamed::new(wave, 40..=43),
                ]);
                if refined {
                    s.push(Streamed::new(wave, 43..=53));
                }
                s
            }
            Some(MovedSampling::Patch) if deeper > 0 => {
                let mut s = vec![
                    Streamed::new(plane_bytes::<C32>(dims, dims.nprobe), 0..=2),
                    Streamed::new(wave, 2..=11),
                    Streamed::new(wave, 4..=7),
                    Streamed::new(wave, 7..=13),
                    Streamed::new(wave, 11..=12),
                ];
                s.extend(vec![Streamed::new(wave, 3..=15); more]);
                if refined {
                    s.extend([
                        Streamed::new(wave, 13..=20),
                        Streamed::new(wave, 20..=22),
                        Streamed::new(wave, 36..=37),
                    ]);
                } else {
                    s.push(Streamed::new(wave, 13..=16));
                }
                s.extend(Streamed::shifted_psi_sums(dims, 15));
                s.extend([Streamed::new(wave, 23..=42); 2]);
                s.extend([Streamed::new(wave, 24..=42); 3]);
                s
            }
            Some(MovedSampling::Patch) => {
                let mut s = vec![
                    Streamed::new(plane_bytes::<C32>(dims, dims.nprobe), 0..=2),
                    Streamed::new(wave, 2..=4),
                ];
                if refined {
                    s.extend([
                        Streamed::new(wave, 4..=11),
                        Streamed::new(wave, 11..=13),
                        Streamed::new(wave, 14..=22),
                        Streamed::new(wave, 14..=22),
                        Streamed::new(wave, 17..=20),
                        Streamed::new(wave, 20..=22),
                    ]);
                } else {
                    s.extend([
                        Streamed::new(wave, 4..=7),
                        Streamed::new(wave, 14..=20),
                        Streamed::new(wave, 14..=20),
                    ]);
                }
                s.extend(Streamed::shifted_psi_sums(dims, 6));
                s
            }
        }
    }

    /// The sampling its views read the object under, if the state is the position model's.
    fn reads(&self) -> Option<Sampling> {
        self.moved.as_ref().map(BhMoved::reads)
    }

    /// The position's [`crate::base::Feature::refine`] the state was built for, none without
    /// the position model.
    fn position(&self) -> Option<bool> {
        self.moved.as_ref().map(BhMoved::refined)
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
            sliced: self.sliced.as_mut(),
        }
    }
}

/// rPIE's batches (one table entry and plan length each) and buffers, sized for the longest
/// batch, for an object of `one` slice.
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
    one: OneSlice,
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
    pub one: OneSlice,
}

impl RpieScratch {
    /// For `views` in `params`' batches, under a position model by `moved`.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        params: &RpieParams,
        nprobe_values: usize,
        moved: Option<MovedScratch>,
        one: OneSlice,
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        let batch = batch_frames(dims, params.batch_size);
        let (cut, _) = batch_stacks(stream, dims, batch, batch)?;
        let len = cut.wave_len(dims);
        let plans = cut.plans(stream, dims)?;
        let moved = (moved.map(|s| RpieMoved::new(stream, kernels, views, &cut, s))).transpose()?;
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
            moved,
            table,
            one,
        })
    }

    /// Its buffers, and apart the position model's state.
    fn lend(&mut self) -> (RpieBufs<'_>, Option<&mut RpieMoved>) {
        let bufs = RpieBufs {
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
            one: self.one,
        };
        (bufs, self.moved.as_mut())
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
    /// The object's slices past `obj`; none of one slice.
    pub deeper: &'a mut [DeviceBuffer<C32>],
    /// A multislice object's [`Slices`] but for `deeper`.
    pub slices: Option<SlicesMut<'a>>,
    pub obj_sums: &'a mut UpdateSums,
    pub probe_sums: &'a mut UpdateSums,
    pub obj_len: u32,
    pub probe_len: u32,
    /// The stacks of the passes that stream the shared scratch alone.
    pub scratch: &'a Cut,
    /// The windows of `scratch`'s stacks while the position samples the object.
    pub windows: Option<&'a mut Windows>,
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

/// `e` as [`EngineError::Invalid`] if it is a launch shape `what` cannot take, for a check
/// that runs the launches' own shape rule before they run.
fn shape_refused(what: &str, e: GpuError) -> EngineError {
    match e {
        GpuError::Shape(m) => EngineError::Invalid(format!("{what}: {m}")),
        e => e.into(),
    }
}

impl Device {
    /// Loads the kernels and uploads the views, counts and starting arrays on a new stream of
    /// `ctx`; every pass cuts `0 .. nframe` by the [`StackRule`] of `ctx`'s L2, the LLK ring
    /// holds `options.llk_slots` records of every frame, its unmasked pixels counted here, and
    /// `options.fused` picks the [`Passes`], of which only the fused run a propagator's stage
    /// or hold the carried waves in f16 ([`GpuOptions::psi_f16`]); a position model's offsets
    /// enter as `sampling`.
    pub fn new(
        ctx: &Arc<CudaContext>,
        dims: ViewDims,
        uploads: Uploads<'_>,
        options: GpuOptions,
        sampling: Sampling,
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
            PlaneStage::Banded(tf) => ExitStage::Banded(BandedTransfer::new(&stream, &to_c32(tf))?),
        };
        let half = match (options.psi_f16, options.fused) {
            (false, _) => None,
            (true, false) => {
                return Err(EngineError::Invalid(
                    "psi_f16 needs GpuOptions { fused: true, .. }".into(),
                ));
            }
            (true, true) => {
                let unmasked = |f: &[u16]| -> u64 {
                    f.iter()
                        .filter(|&&c| c != MASKED)
                        .map(|&c| u64::from(c))
                        .sum()
                };
                let frames = counts.chunks_exact(plane_len(&dims));
                Some(half_scale(frames.map(unmasked).max().unwrap_or(0)))
            }
        };
        // a multislice object's gaps run the propagator stage whatever `options.fused` says
        if slices.is_some() {
            for n in dims.frame {
                stage_fft(&dims, n).map_err(|e| shape_refused("a multislice object", e))?;
            }
        }
        let views = Views::new(&stream, dims, addrs)?;
        let tile = TileDots::new(&stream, &dims)?;
        let rule = StackRule::on(options.stack_size, ctx, &dims)?;
        let shared = Shared::new(
            &stream,
            &dims,
            rule,
            options.fused,
            PositionScratch::None,
            slices.as_ref().map_or(1, |s| 1 + s.deeper.len()),
        )?;
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
                    waves: Slices::waves(&stream, &shared.cut, &dims)?,
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
            half,
            graphs: if options.graph {
                Graphs::On { llk: None }
            } else {
                Graphs::Off
            },
            runs: Vec::new(),
            ml: None,
            bh_state: None,
            lsqml: None,
            features: FeatureBufs::default(),
            begun: false,
            sampling,
            stream,
        };
        let ctx = dev.context();
        dev.ring.device().count(&ctx, 0..dims.nframe, &dev.counts)?;
        if let Graphs::On { llk: None } = dev.graphs {
            dev.graphs = Graphs::On {
                llk: Some(dev.capture_llk()?),
            };
        }
        Ok(dev)
    }

    /// Captures the LLK pass's graph.
    fn capture_llk(&mut self) -> Result<Captured, EngineError> {
        self.capture(|bufs, passes, ring, ended, ctx| {
            bufs.forward_llk(ctx, passes, ring.capture(ended), None)
        })
    }

    /// The buffers [`Self::set_features`] models `features` with, which the caller checked
    /// against the scan; fails, building nothing, unless these passes model them (only the
    /// fused ones model any) and every buffer is built. The values of the features already set,
    /// refined or held as they are, are kept to write in place; other features, or another
    /// refine, take buffers of their own, and a position also takes the fused passes' scratch
    /// ([`PositionScratch`]), by what its offsets move and whether they are refined, so one
    /// added, dropped or switched between refined and held takes the stacks recut for it.
    pub fn build_features(&self, features: &Features) -> Result<NewFeatures, EngineError> {
        if matches!(self.shared.passes, Passes::M1 { .. }) && *features != Features::default() {
            return Err(EngineError::Invalid(
                "features need GpuOptions { fused: true, .. }".into(),
            ));
        }
        let f32s = |v: &Option<Feature<Vec<f64>>>| -> Option<Feature<Vec<f32>>> {
            (v.as_ref()).map(|v| v.as_ref().map(|v| v.iter().map(|&x| x as f32).collect()))
        };
        let (scale, background) = (f32s(&features.scale), f32s(&features.background));
        let position: Option<Feature<Vec<f32>>> = (features.position.as_ref()).map(|d| {
            d.as_ref()
                .map(|d| d.iter().flatten().map(|&x| x as f32).collect())
        });
        let refine = |f: &Option<Feature<Vec<f32>>>| f.as_ref().map(|f| f.refine);
        if self.features.refine() == [refine(&scale), refine(&background), refine(&position)] {
            let values = [scale, background, position].map(|f| f.map(|f| f.values));
            return Ok(NewFeatures(FeaturesUpdate::InPlace(values)));
        }
        let stream = &self.stream;
        let dims = self.views.dims();
        let mover = (position.as_ref())
            .map(|_| Mover::new(stream, &self.kernels, dims, self.sampling))
            .transpose()?;
        let refined = position.as_ref().is_some_and(|p| p.refine);
        let shared = self.reshared(mover.as_ref(), refined)?;
        let nf_max = shared
            .as_ref()
            .map_or(&self.shared, |s| &s.shared)
            .cut
            .table
            .nf_max();
        let scale = scale
            .map(|v| {
                let sums = (v.refine)
                    .then(|| DeviceBuffer::zeroed(stream, v.values.len()))
                    .transpose()?;
                Ok::<_, EngineError>(ScaleBufs {
                    values: DeviceBuffer::from_host(stream, &v.values)?,
                    sums,
                })
            })
            .transpose()?;
        let background = background
            .map(|v| {
                let pixel_frames = self.ring.pixel_frames();
                let mut nobs = DeviceBuffer::zeroed(stream, pixel_frames.len())?;
                nobs.copy_from_device_async(pixel_frames, stream)?;
                let sums = (v.refine)
                    .then(|| BackgroundSums::zeroed(stream, dims, nf_max))
                    .transpose()?;
                Ok::<_, EngineError>(BackgroundBufs {
                    values: DeviceBuffer::from_host(stream, &v.values)?,
                    sums,
                    nobs,
                    photons: DeviceBuffer::zeroed(stream, 1)?,
                })
            })
            .transpose()?;
        let position = position
            .zip(mover)
            .map(|(v, mover)| {
                let partials = (v.refine)
                    .then(|| {
                        Ok::<_, EngineError>(PositionPartials {
                            num: DeviceBuffer::zeroed(stream, dims.nframe as usize)?,
                            den: DeviceBuffer::zeroed(stream, dims.nframe as usize)?,
                        })
                    })
                    .transpose()?;
                Ok::<_, EngineError>(PositionBufs {
                    offsets: DeviceBuffer::from_host(stream, &v.values)?,
                    partials,
                    mover,
                })
            })
            .transpose()?;
        let built = FeatureBufs {
            scale,
            background,
            position,
        };
        Ok(NewFeatures(FeaturesUpdate::Replace(Box::new((
            built, shared,
        )))))
    }

    /// Models the features `new` was built for from here on: the values of the features
    /// already set are written in place; other features' buffers replace the current ones, and
    /// so drop every graph, which addresses them, with the shared passes they take
    /// ([`Self::reshare`]). A failed write in place may leave the values before it written.
    pub fn set_features(&mut self, new: NewFeatures) -> Result<(), EngineError> {
        match new.0 {
            FeaturesUpdate::InPlace([scale, background, position]) => {
                let have = &mut self.features;
                if let (Some(b), Some(v)) = (&mut have.scale, scale) {
                    b.values.copy_from_host(&self.stream, &v)?;
                }
                fault::at(fault::Exit::FeatureCopy)?;
                if let (Some(b), Some(v)) = (&mut have.background, background) {
                    b.values.copy_from_host(&self.stream, &v)?;
                }
                if let (Some(b), Some(v)) = (&mut have.position, position) {
                    b.offsets.copy_from_host(&self.stream, &v)?;
                }
            }
            FeaturesUpdate::Replace(built) => {
                let (features, shared) = *built;
                self.runs.clear();
                self.graphs.pend_llk();
                self.features = features;
                if let Some(r) = shared {
                    self.reshare(r);
                }
            }
        }
        Ok(())
    }

    /// The shared passes, with a multislice object's waves, for a position that moves as
    /// `mover` (none without a position), `refined` or held: `None` if the current ones serve
    /// it, as they do whenever both take the same [`PositionScratch`].
    fn reshared(
        &self,
        mover: Option<&Mover>,
        refined: bool,
    ) -> Result<Option<Reshared>, EngineError> {
        let scratch = PositionScratch::of(mover, refined);
        if scratch == self.features.scratch() {
            return Ok(None);
        }
        let (stream, dims) = (&self.stream, self.views.dims());
        let shared = Shared::new(stream, dims, self.rule, true, scratch, self.nslice())?;
        // A new cut resizes the slices' waves with it.
        let waves = (self.slices.as_ref())
            .map(|_| Slices::waves(stream, &shared.cut, dims))
            .transpose()?;
        fault::at(fault::Exit::Reshare)?;
        Ok(Some(Reshared { shared, waves }))
    }

    /// Installs `r`, dropping the states cut against the shared passes it replaces: DM's or
    /// RAAR's carried stacks, and BH's and LSQML's built under a position model, whose stacks
    /// their frames cap. The caller has dropped every graph, which binds them.
    fn reshare(&mut self, r: Reshared) {
        self.shared = r.shared;
        if let Some((s, waves)) = self.slices.as_mut().zip(r.waves) {
            s.waves = waves;
        }
        self.carried = None;
        if self.bh_state.as_ref().is_some_and(|s| s.reads().is_some()) {
            self.bh_state = None;
        }
        if self.lsqml.as_ref().is_some_and(|s| s.reads().is_some()) {
            self.lsqml = None;
        }
    }

    /// The modelled features, read back, each refined while it keeps its sums.
    pub fn features(&self) -> Result<Features, EngineError> {
        let [scale, background, position] = self.features.refine();
        let f64s = |b: Option<&DeviceBuffer<f32>>, refine: Option<bool>| {
            (b.zip(refine))
                .map(|(b, refine)| {
                    let values = b.to_host_vec(&self.stream)?;
                    Ok::<_, EngineError>(Feature {
                        values: values.into_iter().map(f64::from).collect(),
                        refine,
                    })
                })
                .transpose()
        };
        Ok(Features {
            scale: f64s(self.features.scale.as_ref().map(|b| &b.values), scale)?,
            background: f64s(
                self.features.background.as_ref().map(|b| &b.values),
                background,
            )?,
            position: (self.features.position.as_ref().zip(position))
                .map(|(p, refine)| {
                    let d = p.offsets.to_host_vec(&self.stream)?;
                    let d = d.as_chunks::<2>().0.iter();
                    Ok::<_, EngineError>(Feature {
                        values: d.map(|o| o.map(f64::from)).collect(),
                        refine,
                    })
                })
                .transpose()?,
        })
    }

    pub fn sampling(&self) -> Sampling {
        self.sampling
    }

    /// What [`Self::set_sampling`] enters `sampling` with: of another sampling while a
    /// position is modelled, what its offsets move under it, with the shared passes as
    /// [`Self::build_features`] builds them; fails, building nothing, if a buffer cannot be
    /// built.
    pub fn build_sampling(&self, sampling: Sampling) -> Result<NewSampling, EngineError> {
        let moved = (self.features.position.as_ref())
            .filter(|_| sampling != self.sampling)
            .map(|_| {
                let mover = Mover::new(&self.stream, &self.kernels, self.views.dims(), sampling)?;
                let shared = self.reshared(Some(&mover), self.features.position_refined())?;
                Ok::<_, EngineError>((mover, shared))
            })
            .transpose()?;
        Ok(NewSampling { sampling, moved })
    }

    /// Enters a position's offsets as `new`'s sampling from here on. Another sampling drops
    /// ML's, BH's and LSQML's states built under a position model, which the next call of
    /// their algorithm rebuilds under it, and, while a position is modelled, replaces what its
    /// offsets move, with the shared passes as [`Self::set_features`] does, and every run,
    /// whose graphs bind them.
    pub fn set_sampling(&mut self, new: NewSampling) {
        if new.sampling == self.sampling {
            return;
        }
        self.sampling = new.sampling;
        if self.ml.as_ref().is_some_and(|s| s.position.is_some()) {
            self.ml = None;
        }
        if let Some((p, (mover, shared))) = self.features.position.as_mut().zip(new.moved) {
            p.mover = mover;
            self.runs.clear();
            self.graphs.pend_llk();
            if let Some(r) = shared {
                self.reshare(r);
            }
        }
    }

    /// Fails unless LSQML runs here: its fused row and column passes run whatever
    /// [`GpuOptions::fused`] says, so take the fused passes' launch shapes alone, and the LLK
    /// of each frame's own probe under a probe variation (`varied`) runs on the fused passes
    /// alone.
    pub fn lsqml_runs(&self, varied: bool) -> Result<(), EngineError> {
        Shape::of(&self.dims()).map_err(|e| shape_refused("LSQML", e))?;
        if varied && matches!(self.shared.passes, Passes::M1 { .. }) {
            return Err(EngineError::Invalid(
                "LSQML's probe variation needs GpuOptions { fused: true, .. }".into(),
            ));
        }
        Ok(())
    }

    /// Whether an intensity scale or a detector background is modelled, refined or held.
    pub fn models_scale_or_background(&self) -> bool {
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

    /// The object's slices.
    fn nslice(&self) -> usize {
        1 + self.deeper().len()
    }

    /// The proof that the object has one slice, which `what` models; fails on a multislice
    /// object. Every pass but LSQML's, BH's and the LLK pass is enqueued behind this, which
    /// holds while the device lives: nothing changes its slice count.
    fn one_slice(&self, what: &str) -> Result<OneSlice, EngineError> {
        match &self.slices {
            None => Ok(OneSlice(())),
            Some(s) => Err(EngineError::Invalid(format!(
                "{what} runs one object slice, got {}: only LSQML, BH and the LLK pass run a \
                 multislice object",
                s.deeper.len() + 1
            ))),
        }
    }

    /// Fails on a propagator that drops an evanescent band, for `what` of an algorithm that takes
    /// its adjoint for its inverse: only LSQML, which carries the band, BH, whose gradients take
    /// the adjoint as such, and the LLK pass, forward alone, run it.
    fn unitary(&self, what: &str) -> Result<(), EngineError> {
        match self.exit {
            ExitStage::Banded(_) => Err(EngineError::Invalid(format!(
                "{what} needs a unitary propagator; this one drops an evanescent band: only \
                 LSQML, BH and the LLK pass run it"
            ))),
            _ => Ok(()),
        }
    }

    /// [`Self::one_slice`] and [`Self::unitary`] for `what`: checked before every call that
    /// enqueues a pass but LSQML's, BH's and the LLK pass changes anything.
    pub fn one_unitary(&self, what: &str) -> Result<(), EngineError> {
        self.one_slice(what)?;
        self.unitary(what)
    }

    /// Overwrites the object in place, so every captured graph reads the new values.
    pub fn write_object(&mut self, obj: &[C32]) -> Result<(), EngineError> {
        Ok(self.obj.copy_from_host(&self.stream, obj)?)
    }

    /// [`Self::write_object`] from a device buffer of the object's length in this context,
    /// copied on the stream and waited for, so `src` may be released or rewritten on return.
    pub fn copy_object(&mut self, src: &DeviceBuffer<C32>) -> Result<(), EngineError> {
        self.obj.copy_from_device_async(src, &self.stream)?;
        Ok(self.stream.synchronize()?)
    }

    /// Overwrites the object and probe in place.
    pub fn write_arrays(&mut self, obj: &[C32], probe: &[C32]) -> Result<(), EngineError> {
        self.write_object(obj)?;
        fault::at(fault::Exit::RampWrite)?;
        self.probe.copy_from_host(&self.stream, probe)?;
        Ok(())
    }

    /// Lends every buffer, and apart the passes' own and the ring the cycle writes its records
    /// through.
    fn lend(&mut self) -> (Bufs<'_>, PassesMut<'_>, &mut LlkRing<Tag>) {
        let (deeper, slices) = match self.slices.as_mut() {
            Some(s) => {
                let slices = SlicesMut {
                    gaps: &s.gaps,
                    ones: &s.ones,
                    quad: s.quad,
                    position: &s.position,
                    waves: &mut s.waves,
                };
                (&mut s.deeper[..], Some(slices))
            }
            None => (&mut [][..], None),
        };
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
            deeper,
            slices,
            obj_sums: &mut self.obj_sums,
            probe_sums: &mut self.probe_sums,
            obj_len: self.obj_len,
            probe_len: self.probe_len,
            scratch: &self.shared.cut,
            windows: self.shared.windows.as_mut(),
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
                    sums: s.sums.as_mut(),
                }),
                background: self.features.background.as_mut().map(|b| BackgroundMut {
                    values: &mut b.values,
                    sums: b.sums.as_mut(),
                    nobs: &b.nobs,
                    photons: &mut b.photons,
                }),
                position: self.features.position.as_mut().map(|p| PositionMut {
                    offsets: &mut p.offsets,
                    partials: p.partials.as_mut(),
                    mover: &mut p.mover,
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
        (bufs, passes, &mut self.ring)
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
    /// `order` holds the scan frame at each epoch position. With `opr` the epoch runs that
    /// probe variation, which the first epoch of a call ([`Pre::Init`]) uploads.
    pub fn lsqml_epoch(
        &mut self,
        params: LsqmlParams,
        pre: Pre,
        opr: Option<&OprValues>,
        cycle: usize,
        order: &[u32],
    ) -> Result<(), EngineError> {
        let epoch = Epoch::Lsqml {
            params,
            pre,
            opr: opr.map(|v| v.shape),
        };
        self.steady_with(epoch, cycle, |dev| {
            let ctx = dev.context();
            let l = dev
                .lsqml
                .as_deref_mut()
                .expect("new_run built the LSQML state");
            l.set_order(&ctx, &dev.kernels, &dev.views, order)?;
            if let (Some(v), Pre::Init) = (opr, pre) {
                l.upload_opr(&dev.stream, v)?;
            }
            Ok(())
        })
    }

    /// Reads the probe variation the LSQML epochs stepped back into `v`, of their shape.
    pub fn lsqml_read_opr(&self, v: &mut OprValues) -> Result<(), EngineError> {
        let l = (self.lsqml.as_deref()).expect("new_run built the LSQML state");
        l.read_opr(&self.stream, v)
    }

    /// The LLK record of the current estimate under `tag` as [`crate::engine::Lsqml::llk`]
    /// takes it: under the LSQML state's probe variation, if it has one, by an eager pass of
    /// each frame's own probe; otherwise [`Self::llk_pass`].
    pub fn lsqml_llk(&mut self, tag: Tag) -> Result<(), EngineError> {
        if !self.lsqml.as_ref().is_some_and(|l| l.has_opr()) {
            return self.llk_pass(tag);
        }
        let (ctx, nframe) = (self.context(), self.dims().nframe);
        let (mut bufs, mut passes, ring) = self.lend();
        let rec = ring.record(&ctx, tag, nframe)?;
        let opr = bufs.lsqml.as_mut().and_then(|l| l.opr.take());
        Ok(bufs.forward_llk(&ctx, &mut passes, rec, opr)?)
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
            fault::at(fault::Exit::Launch)?;
            let run = &mut runs[at];
            let tag = Some((cycle, epoch.algorithm()));
            self.summing(|dev| {
                let (ctx, nframe) = (dev.context(), dev.dims().nframe);
                match &run.graph {
                    Some(c) => c.replay(&mut dev.ring, &ctx, nframe, tag),
                    None => {
                        let records = run.cycle.records();
                        let (mut bufs, mut passes, ring) = dev.lend();
                        let llk = records
                            .then(|| ring.record(&ctx, tag, nframe))
                            .transpose()?;
                        Ok(bufs.cycle(&ctx, &mut passes, llk, &mut run.cycle)?)
                    }
                }
            })
        });
        self.runs = runs;
        done
    }

    /// Enqueues the LLK record of the current estimate under `tag`: with graphs, from the
    /// LLK pass's graph, captured first while its capture is pending, which a failed capture
    /// leaves pending; eagerly without.
    pub fn llk_pass(&mut self, tag: Tag) -> Result<(), EngineError> {
        let (ctx, nframe) = (self.context(), self.dims().nframe);
        match &self.graphs {
            Graphs::Off => {
                let (mut bufs, mut passes, ring) = self.lend();
                ring.record(&ctx, tag, nframe)
                    .map_err(EngineError::from)
                    .and_then(|rec| Ok(bufs.forward_llk(&ctx, &mut passes, rec, None)?))
            }
            Graphs::On { llk: Some(c) } => c.replay(&mut self.ring, &ctx, nframe, tag),
            Graphs::On { llk: None } => {
                let c = self.capture_llk()?;
                let done = c.replay(&mut self.ring, &ctx, nframe, tag);
                self.graphs = Graphs::On { llk: Some(c) };
                done
            }
        }
    }

    /// The LLK records counted since the last flush, in order; waits for their copies.
    pub fn flush(&mut self) -> Result<Vec<(Tag, LlkTotals)>, EngineError> {
        Ok(self.ring.flush(&self.context())?)
    }

    /// The buffers of `epoch`'s cycle, and with graphs its graph, captured from them.
    fn new_run(&mut self, epoch: Epoch) -> Result<EpochRun, EngineError> {
        fault::at(fault::Exit::RunBuild)?;
        let what = format!("{:?}", epoch.algorithm());
        let mut cycle = match epoch {
            Epoch::Ap { params, llk } => {
                let one = self.one_slice(&what)?;
                self.unitary(&what)?;
                self.features.build_solve(&self.stream)?;
                Cycle::Ap { params, llk, one }
            }
            Epoch::Dm { params, first } => {
                let one = self.one_slice(&what)?;
                self.unitary(&what)?;
                self.build_carried(false)?;
                self.features.build_solve(&self.stream)?;
                Cycle::Dm { params, first, one }
            }
            Epoch::Raar { params, first } => {
                let one = self.one_slice(&what)?;
                self.unitary(&what)?;
                self.build_carried(true)?;
                self.features.build_solve(&self.stream)?;
                Cycle::Raar { params, first, one }
            }
            Epoch::Rpie(p) => {
                let one = self.one_slice(&what)?;
                self.unitary(&what)?;
                let moved = self.features.positioned().then(|| self.moved_scratch());
                let scratch = RpieScratch::new(
                    &self.stream,
                    &self.kernels,
                    &self.views,
                    &p,
                    self.probe.len(),
                    moved,
                    one,
                )?;
                Cycle::Rpie(p, Box::new(scratch))
            }
            Epoch::Ml { params, first, llk } => {
                let one = self.one_slice(&what)?;
                self.unitary(&what)?;
                self.build_ml(one)?;
                Cycle::Ml { params, first, llk }
            }
            Epoch::Bh { params, cg } => {
                self.build_bh(&params)?;
                Cycle::Bh { params, cg }
            }
            Epoch::Lsqml { params, pre, opr } => {
                self.build_lsqml(&params, opr)?;
                Cycle::Lsqml { params, pre, opr }
            }
        };
        let graph = match self.graphs {
            Graphs::On { .. } => Some(self.capture(|bufs, passes, ring, ended, ctx| {
                let llk = cycle.records().then(|| ring.capture(ended));
                bufs.cycle(ctx, passes, llk, &mut cycle)
            })?),
            Graphs::Off => None,
        };
        Ok(EpochRun { cycle, graph })
    }

    /// Builds DM's, or with `raar` RAAR's, [`CarriedBufs`] unless [`Self::steady_with`] kept
    /// them for this run.
    fn build_carried(&mut self, raar: bool) -> Result<(), EngineError> {
        if self.carried.is_none() {
            let dims = self.dims();
            let (rule, half) = (self.rule, self.half);
            let c = CarriedBufs::new(&self.stream, &dims, rule, &self.shared, raar, half)?;
            self.carried = Some(Box::new(c));
        }
        Ok(())
    }

    /// Builds the [`MlState`] for the position model unless it is built: with the line
    /// search's moved probes if it moves the probe, its windows if it samples the object.
    /// One of another model is replaced only after [`Self::set_features`] or
    /// [`Self::set_sampling`] has dropped every run, so no graph captured from it survives.
    fn build_ml(&mut self, one: OneSlice) -> Result<(), EngineError> {
        let position = self.features.scratch();
        if !(self.ml.as_ref()).is_some_and(|ml| ml.fits(position)) {
            let lens = [self.obj.len(), self.probe.len()];
            let dims = self.dims();
            let (stream, kernels) = (&self.stream, &self.kernels);
            let state = MlState::new(stream, kernels, &dims, self.rule, lens, position, one)?;
            self.ml = Some(Box::new(state));
        }
        Ok(())
    }

    /// Builds the [`BhState`] for `params`' batch size, the position model, refined or held,
    /// and its sampling unless it is built. A state of another size, model or sampling is
    /// replaced only here, when [`Self::steady`] has already dropped the runs of other
    /// parameters, [`Self::set_features`] those of other features and [`Self::set_sampling`]
    /// those of another sampling, so no graph captured from it survives.
    fn build_bh(&mut self, params: &BhParams) -> Result<(), EngineError> {
        let dims = self.dims();
        let size = batch_frames(&dims, params.batch_size);
        let reads = self.features.positioned().then_some(self.sampling);
        let [.., position] = self.features.refine();
        let deeper = self.deeper().len();
        if (self.bh_state.as_ref())
            .is_none_or(|s| s.size != size || s.reads() != reads || s.position() != position)
        {
            let state = BhState::new(
                &self.stream,
                &self.kernels,
                &self.views,
                size,
                self.rule,
                deeper,
                reads.map(|_| self.moved_scratch()),
            )?;
            self.bh_state = Some(Box::new(state));
        }
        Ok(())
    }

    /// Builds the [`LsqmlState`] for `params`' batch size, the position model, refined or
    /// held, its sampling and the probe variation of shape `opr` unless it is built, replaced
    /// as [`Self::build_bh`] replaces BH's.
    fn build_lsqml(
        &mut self,
        params: &LsqmlParams,
        opr: Option<OprShape>,
    ) -> Result<(), EngineError> {
        let size = batch_frames(&self.dims(), params.batch_size);
        let reads = self.features.positioned().then_some(self.sampling);
        let [.., position] = self.features.refine();
        if (self.lsqml.as_ref())
            .is_none_or(|s| s.size() != size || s.reads() != reads || s.position() != position)
        {
            let state = LsqmlState::new(
                &self.stream,
                &self.kernels,
                &self.views,
                size,
                self.rule,
                reads.map(|_| self.moved_scratch()),
                (self.deeper().len(), &self.exit),
            )?;
            self.lsqml = Some(Box::new(state));
        }
        let l = self.lsqml.as_deref_mut().expect("built above");
        let llk_frames = self.shared.cut.table.nf_max();
        Ok(l.fit_opr(&self.stream, &self.kernels, &self.views, opr, llk_frames)?)
    }

    /// The position model's [`MovedScratch`] for a cycle that runs its own passes.
    fn moved_scratch(&self) -> MovedScratch {
        let sampling = match self.features.position.as_ref().map(|p| &p.mover) {
            Some(Mover::Object { sampler, .. }) => MovedSampling::FullObject {
                held: sampler.windows_held(),
                scratch: sampler.scratch_bytes(),
                acc: sampler.acc_bytes(),
            },
            _ => MovedSampling::Patch,
        };
        MovedScratch {
            frames: self.shared.cut.table.nf_max(),
            sampling,
            refined: self.features.position_refined(),
        }
    }

    /// [`Bufs::scale_to_photons`] enqueued eagerly, with `total` the unmasked observed photons.
    pub fn scale_to_photons(&mut self, total: f32) -> Result<(), EngineError> {
        let one = self.one_slice("ML's scaling")?;
        self.unitary("ML's scaling")?;
        self.build_ml(one)?;
        let ctx = self.context();
        let (mut bufs, mut passes, _) = self.lend();
        Ok(bufs.with_ml(|b, ml| b.scale_to_photons(&ctx, &mut passes, ml, total))?)
    }

    /// PyNX `ScaleObjProbe(absolute=False)` enqueued eagerly: `Σ|O|` and `Σ|P|` balanced,
    /// `O P` unchanged.
    pub fn balance_obj_probe(&mut self) -> Result<(), EngineError> {
        let ctx = self.context();
        let (mut bufs, _, _) = self.lend();
        Ok(bufs.scale_obj_probe(&ctx, None)?)
    }

    /// The ML gradients of the current estimate, object then probe, as
    /// [`Bufs::ml_grad`] leaves them in the update sums; downloads them and zeroes the sums.
    pub fn ml_grads(&mut self) -> Result<[Vec<C32>; 2], EngineError> {
        self.ml_grad_sums([true, true], |stream, [obj, probe]| {
            Ok([obj.to_host_vec(stream)?, probe.to_host_vec(stream)?])
        })
    }

    /// The object's ML gradient alone, copied into `dst`, a buffer of the object's length in
    /// this context; the stream is waited for on every exit, so another stream may read `dst`
    /// once this returns.
    pub fn ml_obj_grad_into(&mut self, dst: &mut DeviceBuffer<C32>) -> Result<(), EngineError> {
        let copied = self.ml_grad_sums([true, false], |stream, [obj, _]| {
            Ok(dst.copy_from_device_async(obj, stream)?)
        });
        let waited = self.stream.synchronize();
        copied?;
        Ok(waited?)
    }

    /// [`Bufs::ml_grad`] for `update` (object, probe), then `read` on the stream and the update
    /// sums' numerators it leaves, object then probe. The sums are zeroed after `read`; a pass or
    /// `read` that failed leaves them for [`Self::summing`] to discard before the next adds.
    fn ml_grad_sums<R>(
        &mut self,
        update: [bool; 2],
        read: impl FnOnce(&Arc<CudaStream>, [&DeviceBuffer<C32>; 2]) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let one = self.one_slice("ML's gradient")?;
        self.unitary("ML's gradient")?;
        self.summing(|dev| {
            {
                let ctx = dev.context();
                let (mut bufs, mut passes, _) = dev.lend();
                bufs.ml_grad(&ctx, &mut passes, update, None, one)?;
            }
            let r = read(&dev.stream, [&dev.obj_sums.num, &dev.probe_sums.num])?;
            let ctx = dev.context();
            dev.zero_sums(&ctx)?;
            Ok(r)
        })
    }

    /// `f`, which adds to the update sums and consumes them, from zero sums: a call that
    /// failed before consuming them leaves them begun, and this discards them first, so a
    /// retry restarts the work rather than adding to what the failed call left.
    fn summing<R>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if self.begun {
            let ctx = self.context();
            self.zero_sums(&ctx)?;
            self.features.clear_sums(&ctx)?;
        }
        self.begun = true;
        let r = f(self)?;
        self.begun = false;
        Ok(r)
    }

    /// Zeroes [`update_nums`] and the two denominators.
    fn zero_sums(&mut self, ctx: &ExecutionContext) -> Result<(), GpuError> {
        crate::gpu::zero(ctx, &mut self.obj_sums.den)?;
        crate::gpu::zero(ctx, &mut self.probe_sums.den)?;
        let sliced = self.bh_state.as_deref_mut().and_then(|s| s.sliced.as_mut());
        let grads = sliced.map_or(&mut [][..], BhSliced::grads);
        for num in update_nums(&mut self.obj_sums, &mut self.probe_sums, grads) {
            crate::gpu::zero(ctx, num)?;
        }
        Ok(())
    }

    /// `f` on the lent buffers, captured on a side stream, so nothing runs; with the
    /// [`Ended`] its record's end left in the slot `f` is lent, if it ended one through
    /// [`LlkRing::capture`].
    fn capture(
        &mut self,
        f: impl FnOnce(
            &mut Bufs<'_>,
            &mut PassesMut<'_>,
            &mut LlkRing<Tag>,
            &mut Option<Ended>,
            &ExecutionContext,
        ) -> Result<(), GpuError>
        + Send,
    ) -> Result<Captured, EngineError> {
        fault::at(fault::Exit::Capture)?;
        let side = self.kernels.context().new_stream()?;
        let (mut bufs, mut passes, ring) = self.lend();
        let mut ended = None;
        let op = crate::gpu::record(|ctx| f(&mut bufs, &mut passes, ring, &mut ended, ctx));
        let graph = CudaGraph::capture(side, op).map_err(GpuError::from)?;
        Ok(Captured { graph, ended })
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

    /// One cycle of `cycle` enqueued on `ctx`; `llk`, given when [`Cycle::records`], is the
    /// record it writes and ends, which no other cycle touches.
    fn cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        llk: Option<RecordWriter<'_, Tag>>,
        cycle: &mut Cycle,
    ) -> Result<(), GpuError> {
        match cycle {
            Cycle::Ap { params, one, .. } => self.ap_cycle(ctx, passes, params, llk, *one),
            Cycle::Dm { params, first, one } => {
                self.with_carried(|b, c| b.dm_cycle(ctx, passes, c, params, *first, *one))
            }
            Cycle::Raar { params, first, one } => {
                self.with_carried(|b, c| b.raar_cycle(ctx, passes, c, params, *first, *one))
            }
            Cycle::Rpie(p, scratch) => {
                let (mut rb, moved) = scratch.lend();
                self.rpie_epoch(ctx, p, &mut rb, moved)
            }
            Cycle::Ml { params, first, .. } => {
                self.with_ml(|b, ml| b.ml_epoch(ctx, passes, ml, params, *first, llk))
            }
            Cycle::Bh { params, cg } => {
                self.with_bh(|b, bh| b.bh_epoch(ctx, passes, bh, params, *cg))
            }
            Cycle::Lsqml { params, pre, .. } => self.lsqml_epoch(ctx, passes, params, *pre),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::base::required_obj_shape;
    use crate::base::{C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack};
    use crate::engine::{
        Backend, BhParams, Feature, Features, GpuPtycho, Optics, Params, ProbeVariation, Step,
        Wavefield,
    };
    use crate::live::farfield::{Ptycho, PtychoSpec, ScanPath};

    /// An engine on a simulated raster of 36 points, its object flat: 32-pixel frames, or the
    /// fused passes' smallest, 64.
    fn engine(ctx: &Arc<CudaContext>, graph: bool, fused: bool) -> GpuPtycho {
        let (scan, object, probe, options) = start(graph, fused);
        GpuPtycho::new(ctx, &scan, object, probe, options).unwrap()
    }

    /// [`engine`]'s scan, object, probe and options.
    fn start(graph: bool, fused: bool) -> (Scan, Object, Probe, GpuOptions) {
        let n = if fused { 64 } else { 32 };
        let p = Ptycho::new(PtychoSpec {
            n,
            path: ScanPath::Raster { cols: 6 },
            points: 36,
            step: 6.0,
            photons: 1e6,
            seed: 5,
            wavelength: 1.5498e-10,
            distance: 5.0,
            det_pixel: 75e-6,
        });
        let s = p.spec();
        let g = Geometry::new(s.wavelength, s.distance, s.det_pixel, [n, n]).unwrap();
        let (x, y): (Vec<f64>, Vec<f64>) = (0..p.points()).map(|k| p.position(k)).unzip();
        let counts = (0..p.points()).flat_map(|k| p.frame(k)).collect();
        let scan = Scan::new(
            g,
            Positions::new(y, x).unwrap(),
            Counts::from_raw([p.points(), n, n], counts).unwrap(),
        )
        .unwrap();
        let pixel = g.pixel_size_object();
        let obj = required_obj_shape(&scan.positions.pynx_preprocess(pixel), pixel, [n, n], 0);
        let object = Object::new(Stack::filled([1, obj[0], obj[1]], C64::new(1.0, 0.0)));
        let probe = p.probe().iter().map(|c| C64::new(c.re.into(), c.im.into()));
        let probe = Probe::new(Stack::from_vec([2, n, n], probe.collect()).unwrap());
        let options = GpuOptions {
            stack_size: Some(8),
            graph,
            llk_slots: 4,
            fused,
            psi_f16: false,
        };
        (scan, object, probe, options)
    }

    /// What a call through [`Device::summing`] added to the update sums before failing never
    /// reaches the next: AP cycles after it, launched or replayed, are bit for bit those of
    /// an engine it never ran on, also with the features whose sums the fused passes add to
    /// across stacks, the background's and a position's shifted probe's.
    #[test]
    fn a_failed_summing_call_leaves_its_retry_no_sums() {
        let ctx = CudaContext::new(0).expect("GPU 0");
        let step = Step {
            params: Params::Ap(ApParams {
                obj_inertia: Some(0.01),
                probe_inertia: Some(0.01),
                calc_llk: 1,
                zero_phase_ramp: false,
            }),
            count: 2,
        };
        let summed = Features {
            background: Some(Feature::refined(vec![1.0; 64 * 64])),
            position: Some(Feature::refined(vec![[0.1, -0.1]; 36])),
            ..Features::default()
        };
        for (graph, fused) in [(false, false), (true, false), (false, true), (true, true)] {
            let [mut failed, mut clean] = [(); 2].map(|()| {
                let mut e = engine(&ctx, graph, fused);
                if fused {
                    e.set_features(&summed).unwrap();
                }
                e
            });
            let r = failed.dev.ungated().summing(|dev| {
                for sums in [&mut dev.obj_sums, &mut dev.probe_sums] {
                    let num = vec![C32 { re: 1.0, im: 1.0 }; sums.num.len()];
                    sums.num.copy_from_host(&dev.stream, &num)?;
                    sums.den
                        .copy_from_host(&dev.stream, &vec![1.0; sums.den.len()])?;
                }
                if let Some(BackgroundBufs {
                    sums: Some(sums), ..
                }) = &mut dev.features.background
                {
                    let sum = &mut sums.sum;
                    sum.copy_from_host(&dev.stream, &vec![1.0; sum.len()])?;
                }
                if let Some(PositionBufs {
                    mover: Mover::Probe { probe_sums, .. },
                    ..
                }) = &mut dev.features.position
                {
                    let (nums, den) = probe_sums.sums_mut();
                    for num in nums {
                        let ones = vec![C32 { re: 1.0, im: 1.0 }; num.len()];
                        num.copy_from_host(&dev.stream, &ones)?;
                    }
                    den.copy_from_host(&dev.stream, &vec![1.0; den.len()])?;
                }
                Err::<(), _>(EngineError::Invalid("injected".into()))
            });
            let what = format!("graph {graph}, fused {fused}");
            assert!(matches!(r, Err(EngineError::Invalid(_))), "{what}");
            for e in [&mut failed, &mut clean] {
                e.run(&step).unwrap();
            }
            let bits = |e: &GpuPtycho| {
                let (o, p) = (e.object().unwrap(), e.probe().unwrap());
                let llk = e.history().iter().map(|r| r.llk.poisson.to_bits());
                let px = o.as_slice().iter().chain(p.as_slice());
                let px = px.flat_map(|c| [c.re.to_bits(), c.im.to_bits()]);
                (px.collect::<Vec<_>>(), llk.collect::<Vec<_>>())
            };
            let (f, c) = (bits(&failed), bits(&clean));
            assert_eq!(f.1.len(), 2, "{what}");
            assert!(f == c, "{what}: the retry differs from a clean run");
            let features = |e: &GpuPtycho| {
                let f = e.features().unwrap();
                let pos = f.position.map(|p| p.values).into_iter().flatten().flatten();
                let px = (f.background.map(|b| b.values).into_iter().flatten()).chain(pos);
                px.map(f64::to_bits).collect::<Vec<_>>()
            };
            assert!(
                features(&failed) == features(&clean),
                "{what}: the retry's features differ from a clean run's"
            );
        }
    }

    /// What a failed call through [`Device::summing`] added to BH's sums of a multislice
    /// object's slices past the first never reaches the next: a BH epoch after it, launched or
    /// replayed, is bit for bit that of an engine it never ran on.
    #[test]
    fn a_failed_summing_call_leaves_bh_no_slice_sums() {
        let ctx = CudaContext::new(0).expect("GPU 0");
        let params = BhParams {
            batch_size: 12,
            probe_rho: 0.01,
            position_rho: 0.1,
            seed: 7,
        };
        for graph in [false, true] {
            let [mut failed, mut clean] = [(); 2].map(|()| {
                // The slices' propagator stage has in-kernel FFTs of the fused passes' sides alone.
                let (scan, object, probe, options) = start(graph, true);
                let g = scan.geometry;
                let gap =
                    Wavefield::new(g.frame.sides(), g.wavelength, g.pixel_size_object(), 2e-5);
                let optics = Optics::far_field(&g).with_slices(&[gap.unwrap()]);
                let objects = [object.clone(), object];
                let mut e =
                    GpuPtycho::with_slices(&ctx, &scan, &optics.unwrap(), &objects, probe, options)
                        .unwrap();
                e.bh(&params, 1, 1).unwrap();
                e
            });
            let r = failed.dev.ungated().summing(|dev| {
                let bh = dev.bh_state.as_deref_mut().expect("BH's state");
                let g = &mut bh.sliced.as_mut().expect("two slices").grads()[0];
                g.copy_from_host(&dev.stream, &vec![C32 { re: 1.0, im: 1.0 }; g.len()])?;
                Err::<(), _>(EngineError::Invalid("injected".into()))
            });
            assert!(matches!(r, Err(EngineError::Invalid(_))), "graph {graph}");
            for e in [&mut failed, &mut clean] {
                e.bh(&params, 1, 1).unwrap();
            }
            let bits = |e: &GpuPtycho| -> Vec<Vec<u64>> {
                let (o, p) = (e.objects().unwrap(), e.probe().unwrap());
                let px = |s: &[C64]| -> Vec<u64> {
                    s.iter()
                        .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                        .collect()
                };
                let llk = e.history().iter().map(|r| r.llk.poisson.to_bits());
                let arrays = o.iter().map(|o| px(o.as_slice()));
                arrays.chain([px(p.as_slice()), llk.collect()]).collect()
            };
            let (f, c) = (bits(&failed), bits(&clean));
            assert_eq!((f.len(), f[3].len()), (4, 2), "graph {graph}");
            for (what, (f, c)) in ["slice 0", "slice 1", "the probe", "the LLK"]
                .iter()
                .zip(f.iter().zip(&c))
            {
                assert!(f == c, "graph {graph}: {what} differs from a clean run's");
            }
        }
    }

    /// rPIE, ML, LSQML and BH divide by no denominator, so their passes add to the numerators
    /// alone: a sentinel in every denominator, the update sums' and a moved probe's, survives
    /// their cycles under either sampling, with no position and with one held or refined.
    #[test]
    fn numerator_updates_leave_every_denominator() {
        const SENTINEL: f32 = 7.0;
        let ctx = CudaContext::new(0).expect("GPU 0");
        fn dens(dev: &mut Device) -> Vec<&mut DeviceBuffer<f32>> {
            let mut dens = vec![&mut dev.obj_sums.den, &mut dev.probe_sums.den];
            if let Some(PositionBufs {
                mover: Mover::Probe { probe_sums, .. },
                ..
            }) = &mut dev.features.position
            {
                dens.push(probe_sums.sums_mut().1);
            }
            dens
        }
        let run = |e: &mut GpuPtycho, algo| match algo {
            "rpie" => e.rpie(
                &RpieParams {
                    obj_step: Some(1.0),
                    probe_step: Some(1.0),
                    obj_alpha: 0.5,
                    probe_alpha: 0.5,
                    batch_size: 12,
                    calc_llk: 1,
                },
                1,
            ),
            "ml" => e.ml(
                &MlParams {
                    update_object: true,
                    update_probe: true,
                    calc_llk: 1,
                    zero_phase_ramp: false,
                },
                2,
            ),
            "lsqml" => e.lsqml(
                &LsqmlParams {
                    batch_size: 12,
                    cost: crate::engine::Cost::Poisson,
                    seed: 7,
                },
                &mut ProbeVariation::none([e.probe_dims[1], e.probe_dims[2]]),
                1,
                1,
            ),
            _ => e.bh(
                &BhParams {
                    batch_size: 12,
                    probe_rho: 0.01,
                    position_rho: 0.1,
                    seed: 7,
                },
                2,
                1,
            ),
        };
        let offsets = vec![[0.1, -0.1]; 36];
        let positions = [
            None,
            Some(Feature::held(offsets.clone())),
            Some(Feature::refined(offsets)),
        ];
        for sampling in [Sampling::Patch, Sampling::FullObject] {
            for position in &positions {
                for algo in ["rpie", "ml", "lsqml", "bh"] {
                    let mut e = engine(&ctx, true, true);
                    e.set_sampling(sampling).unwrap();
                    let features = Features {
                        position: position.clone(),
                        ..Features::default()
                    };
                    e.set_features(&features).unwrap();
                    let what = format!("{algo} {sampling:?} position {position:?}");
                    run(&mut e, algo).unwrap_or_else(|err| panic!("{what}: {err}"));
                    let dev = e.dev.ungated();
                    let stream = Arc::clone(&dev.stream);
                    for den in dens(dev) {
                        den.copy_from_host(&stream, &vec![SENTINEL; den.len()])
                            .unwrap();
                    }
                    run(&mut e, algo).unwrap_or_else(|err| panic!("{what}: {err}"));
                    for (k, den) in dens(e.dev.ungated()).into_iter().enumerate() {
                        let kept = den.to_host_vec(&stream).unwrap();
                        assert!(
                            kept.iter().all(|&d| d == SENTINEL),
                            "{what}: denominator {k} changed"
                        );
                    }
                }
            }
        }
    }

    /// ML's line search refuses a lend that samples the object to a state built for no
    /// position, which [`Device::build_ml`]'s rebuild alone kept out: the state is swapped in
    /// after it.
    #[test]
    fn ml_refuses_windows_lent_to_a_fixed_line() {
        let ctx = CudaContext::new(0).expect("GPU 0");
        let mut e = engine(&ctx, false, true);
        e.set_sampling(Sampling::FullObject).unwrap();
        let position = Features {
            position: Some(Feature::held(vec![[0.1, -0.1]; 36])),
            ..Features::default()
        };
        e.set_features(&position).unwrap();
        let epoch = Epoch::Ml {
            params: MlParams::default(),
            first: true,
            llk: false,
        };
        let r = e.dev.ungated().steady_with(epoch, 0, |dev| {
            let (lens, dims) = ([dev.obj.len(), dev.probe.len()], dev.dims());
            let one = dev.one_slice("ML")?;
            let none = PositionScratch::None;
            let fixed = MlState::new(&dev.stream, &dev.kernels, &dims, dev.rule, lens, none, one)?;
            dev.ml = Some(Box::new(fixed));
            Ok(())
        });
        assert!(
            matches!(&r, Err(EngineError::Gpu(GpuError::Shape(s)))
                if s == "ML state built for another position model"),
            "{r:?}"
        );
    }

    /// A stack's first loop through `n` slices holds the wave entering each slice past the
    /// first and the last one's exit wave to the backward sweep, `n` waves, and under
    /// full-object sampling each slice's window beside its wave: 1 MiB and 0.5 MiB a frame at
    /// 256x256 against a 24 MiB budget. Whatever the position model, held or refined, the
    /// stack [`BhState`] books holds them through 10 slices; without full-object sampling it
    /// books them exactly from 7, where the first loop outweighs the second. Refined under
    /// full-object sampling, the second loop holds five waves besides the wave and window of
    /// each slice past the first, and the stack holds those too.
    #[test]
    fn bh_books_a_held_wave_per_slice() {
        let dims = ViewDims {
            nframe: 400,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [256; 2],
            obj: [1024; 2],
            quad_phase: 0.0,
        };
        let rule = StackRule::new(None, 64 << 20, 1, &dims).unwrap();
        let (wave, win) = (wave_bytes(&dims), Windows::bytes(&dims));
        let full = MovedSampling::FullObject {
            held: 0,
            scratch: 0,
            acc: 0,
        };
        let moved = |sampling, refined| {
            Some(MovedScratch {
                frames: dims.nframe,
                sampling,
                refined,
            })
        };
        let patch = MovedSampling::Patch;
        let arms = [
            ("unmoved", None, wave, false),
            ("held patch", moved(patch, false), wave, false),
            ("refined patch", moved(patch, true), wave, false),
            ("held full object", moved(full, false), wave + win, false),
            ("refined full object", moved(full, true), wave + win, true),
        ];
        for (name, scratch, per_slice, sliced_second) in arms {
            for n in 2..=10 {
                let frames = rule.frames(&BhState::streamed(&dims, n - 1, scratch)) as usize;
                let held = n * per_slice;
                assert!(
                    frames * held <= rule.budget,
                    "{name}, {n} slices: {frames} frames hold {} bytes",
                    frames * held
                );
                if n >= 7 && per_slice == wave {
                    assert_eq!(frames, rule.budget / held, "{name}, {n} slices");
                }
                if sliced_second {
                    let second = 5 * wave + (n - 1) * per_slice;
                    assert!(
                        frames * second <= rule.budget,
                        "{name}, {n} slices: {frames} frames hold {} bytes in the second loop",
                        frames * second
                    );
                }
            }
        }
    }
}
