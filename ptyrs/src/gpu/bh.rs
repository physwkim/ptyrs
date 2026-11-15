//! Checked launches of the BH kernels (`ptyrs_kernels::bh`), pty-chi's bilinear-Hessian step
//! on the amplitude cost (ptyrs::engine `bh`).
//!
//! # One BH batch in these calls
//!
//! A batch is a run of stacks whose table entries hold epoch positions; an [`EpochOrder`]
//! maps a position to its scan frame and holds the views in epoch order. With zeroed update
//! sums, per stack: [`Kernels::obj_probe_to_psi`] into its `psi`, forward FFT,
//! [`BhKernels::residual`] into its `r`, inverse FFT of `r`, [`Kernels::psi_to_obj`] and
//! [`Kernels::psi_to_probe`] of `r`: the sums' `num` is the gradient `g`. Then, with `u = g`
//! scaled by `ρ²` on the probe and for CG `v` the last direction, `crate::tile::TileDots::dots`;
//! per stack [`BhKernels::far`], forward FFTs of `lu` (and `lv`) and
//! [`BhKernels::hess_frames`]; [`BhKernels::step`], then [`BhKernels::update`] of each array,
//! which consumes the sums. The step never leaves the device.
//!
//! # Under the position model
//!
//! The offsets join the step as a third variable ([`BhOffsets`]), on the moved probes `W` of a
//! [`Shift`] in epoch order (`crate::gpu::PositionKernels::gather`): the exit waves are `O W`
//! (`Kernels::shifted_probe`, `PositionKernels::shift_w`, `RpieKernels::obj_wave_to_psi`),
//! the object and probe gradients `Kernels::shifted_psi_sums` of `r`. Per stack of the
//! gradient pass, [`BhKernels::obj_conj`] of `r`, its forward FFT and [`BhKernels::pos_grad`];
//! [`BhKernels::pos_dots`] after the dots. Per stack of the Hessian pass, in place of
//! [`BhKernels::far`] and [`BhKernels::hess_frames`]: [`BhKernels::pos_spec`], inverse FFTs,
//! [`BhKernels::pos_near`], forward FFTs, [`BhKernels::obj_conj`] and its forward FFT, and
//! [`BhKernels::pos_far`]. After the step, [`BhKernels::pos_update`].
//!
//! # With the offsets held
//!
//! The probes stay moved by the offsets, which the step does not take: no
//! [`BhKernels::obj_conj`], [`BhKernels::pos_grad`], [`BhKernels::pos_dots`] or
//! [`BhKernels::pos_update`]. Per stack of the Hessian pass, [`BhKernels::moved_spec`] in place
//! of [`BhKernels::pos_spec`] and [`BhKernels::moved_far`] in place of
//! [`BhKernels::pos_far`]; under full-object sampling, [`BhKernels::win_obj_grad`] in place of
//! [`BhKernels::win_grad`], no second windows and no [`BhKernels::win_lift`] or
//! [`BhKernels::win_hess`].
//!
//! # Under full-object sampling
//!
//! The views read windows of the object at their offsets (`crate::gpu::sampling`), the
//! estimate's taken per stack with their derivatives over `y` and `x` into three blocks, and
//! [`BhFieldsRef::views`] are the windows' views (`crate::gpu::sampling::window_views`). Per
//! stack of the gradient pass, after `r`'s inverse FFT: [`Kernels::psi_to_probe`] of `r` on
//! the windows, [`BhKernels::win_grad`] and the patches placed onto the object sums. Per
//! stack of the Hessian pass: the windows of `u_O` (and `v_O`) with their derivatives, of the
//! estimate's second derivatives, [`BhKernels::win_lift`], then [`BhKernels::far`] and
//! [`BhKernels::hess_frames`] on the lifted windows and [`BhKernels::win_hess`]. After the
//! step, [`BhKernels::pos_update`].

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, Event, LaunchConfig, PinnedHostBuffer, RawDeviceBuffer};
use ptyrs_kernels::bh::{
    BH_ENTRY_MOVED, BH_ENTRY_PROBE, BH_ENTRY_WAVES, BhDirs, BhFar, BhFields, BhMovedFar,
    BhMovedProbes, BhMoves, BhNear, BhObs, BhPartials, BhPosDirs, BhSecond, BhSliceIn, BhTangents,
    BhWinDerivs, BhWinFields, bh_kernels,
};
pub use ptyrs_kernels::bh::{BH_HESS, BH_STEP};
use ptyrs_kernels::grad::{CG_BLOCKS, GAMMA_CHUNK, ObjProbePtrs};
use ptyrs_kernels::pixel::{PIXEL_THREADS, StackGeom, SumPtrs};
use ptyrs_kernels::position::BatchSpan;

use crate::gpu::ops::grid_1d;
use crate::gpu::position::{ONE_BLOCK, chunk_grid, same_dims, span};
use crate::gpu::{
    C32, Error, Kernels, Permutation, ProbeSpectrum, Shift, StackCtl, StackRef, UpdateSums,
    ViewAddr, ViewDims, Views, check_len, dptr,
};

/// The BH kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct BhKernels {
    base: Kernels,
    module: bh_kernels::LoadedModule,
}

/// The scan in an epoch's order: the views and frame ids of `order`, and the counts in the
/// scan's own order, which the kernels read through the ids.
#[derive(Clone, Copy)]
pub struct BhScan<'a> {
    pub order: &'a EpochOrder,
    pub counts: &'a DeviceBuffer<u16>,
}

impl BhScan<'_> {
    pub(crate) fn check(&self) -> Result<&ViewDims, Error> {
        let dims = self.order.views.dims();
        check_len("counts", self.counts, dims.nframe as usize * dims.n())?;
        Ok(dims)
    }

    pub(crate) fn obs(&self) -> BhObs {
        BhObs {
            counts: dptr(self.counts),
            order: dptr(self.order.ids.order()),
        }
    }
}

/// Host slots of [`EpochOrder::set`]'s uploads: the host waits for a copy only when it comes
/// round to that copy's slot again.
const ORDER_SLOTS: usize = 4;

/// A scan's frames in an epoch's order: `nframe` scan frame ids, one per epoch position, and
/// the scan's views reordered by them. [`Self::set`] is the only writer of either, and the
/// ids are a [`Permutation`], so every id indexes a frame. Its uploads go through pinned host
/// slots, so the host does not wait on the stream.
pub struct EpochOrder {
    ids: Permutation,
    views: Views,
    /// The ids the device holds; empty after a [`Self::set`] that failed midway.
    last: Vec<u32>,
    host: PinnedHostBuffer<u32>,
    /// Per host slot, the event after the last copy out of it.
    copies: [Option<Event>; ORDER_SLOTS],
    next: usize,
}

impl EpochOrder {
    /// `src`'s frames in their own order, set on `stream`.
    pub fn new(kernels: &Kernels, stream: &Arc<CudaStream>, src: &Views) -> Result<Self, Error> {
        let nframe = src.dims().nframe as usize;
        let ids: Vec<u32> = (0..nframe as u32).collect();
        let mut order = Self {
            ids: Permutation::new(stream, &ids)?,
            views: Views::reordered(stream, src)?,
            last: Vec::new(),
            host: PinnedHostBuffer::zeroed(kernels.context(), ORDER_SLOTS * nframe)?,
            copies: Default::default(),
            next: 0,
        };
        order.set(&ExecutionContext::new(stream.clone()), kernels, src, &ids)?;
        Ok(order)
    }

    /// The views in epoch order.
    pub fn views(&self) -> &Views {
        &self.views
    }

    /// The scan frame at each epoch position.
    pub fn ids(&self) -> &DeviceBuffer<u32> {
        self.ids.order()
    }

    /// Enqueues on `ctx` the upload of `ids`, the scan frame at each epoch position, and the
    /// reorder of `src`, the views this order was made from, by them; nothing when the
    /// device already holds `ids`. Waits only for the copy `ORDER_SLOTS` uploads back.
    pub fn set(
        &mut self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        src: &Views,
        ids: &[u32],
    ) -> Result<(), Error> {
        if ids == self.last {
            return Ok(());
        }
        let n = self.ids.order().len();
        if ids.len() != n {
            return Err(Error::Shape(format!(
                "epoch order of {} ids for {n} frames",
                ids.len()
            )));
        }
        if ctx.context().as_ref() != self.host.context().as_ref() {
            return Err(Error::ContextMismatch);
        }
        self.last.clear();
        let slot = self.next;
        if let Some(copy) = self.copies[slot].take() {
            copy.synchronize()?;
        }
        let host = &mut self.host.as_mut_slice()[slot * n..(slot + 1) * n];
        host.copy_from_slice(ids);
        // SAFETY: the host slot is pinned and not written again before this copy's event
        // completes.
        unsafe { self.ids.set_async(ctx, host)? };
        self.copies[slot] = Some(ctx.get_cuda_stream().record_event(None)?);
        self.next = (slot + 1) % ORDER_SLOTS;
        kernels.reorder_views(ctx, src, &self.ids, &mut self.views)?;
        self.last.extend_from_slice(ids);
        Ok(())
    }
}

impl Drop for EpochOrder {
    fn drop(&mut self) {
        // The copies read `host`, freed after this.
        for copy in self.copies.iter().flatten() {
            let _ = copy.synchronize();
        }
    }
}

/// A step's displacements, each `[object, probe]`: `u` with its probe scaled by `u_scale`
/// and, for CG, `v`. Without `v` the step is steepest descent (`β = 0`).
#[derive(Clone, Copy)]
pub struct BhDisp<'a> {
    pub u: [&'a DeviceBuffer<C32>; 2],
    pub u_scale: f32,
    pub v: Option<[&'a DeviceBuffer<C32>; 2]>,
}

impl BhDisp<'_> {
    fn check(&self, dims: &ViewDims) -> Result<(), Error> {
        let lens = lens(dims);
        for (pair, what) in [(Some(self.u), "u"), (self.v, "v")] {
            for (b, len) in pair.into_iter().flatten().zip(lens) {
                check_len(what, b, len as usize)?;
            }
        }
        Ok(())
    }

    fn storages(&self) -> impl Iterator<Item = &RawDeviceBuffer> {
        self.u
            .into_iter()
            .chain(self.v.into_iter().flatten())
            .map(|b| b.storage())
    }

    /// Without `v`, `u`'s pointers stand in for it; the kernels never read them.
    fn ptrs(&self) -> BhDirs {
        let pair = |p: [&DeviceBuffer<C32>; 2]| ObjProbePtrs {
            obj: dptr(p[0]),
            probe: dptr(p[1]),
        };
        BhDirs {
            u: pair(self.u),
            v: pair(self.v.unwrap_or(self.u)),
            u_scale: self.u_scale,
            cg: self.v.is_some().into(),
        }
    }
}

/// One stack's fields the Hessian reads, each of `nf_max * nmode` planes like a stack's
/// `psi`: the forward-FFT'd exit waves, the residual after its inverse FFT, and the
/// forward-FFT'd derivatives along `u` and, with CG, `v`; and `views`, those the object is
/// read through: the epoch's, or under full-object sampling the windows'.
pub struct BhFieldsRef<'a> {
    pub views: &'a Views,
    pub psi: &'a DeviceBuffer<C32>,
    pub r: &'a DeviceBuffer<C32>,
    pub lu: &'a DeviceBuffer<C32>,
    pub lv: Option<&'a DeviceBuffer<C32>>,
}

impl BhFieldsRef<'_> {
    /// The fields a Hessian launch with `cg` or without reads, each checked as `len` long
    /// ([`touched`]): their storages and pointers.
    fn read(&self, cg: bool, len: usize) -> Result<(Vec<&RawDeviceBuffer>, BhFields), Error> {
        for (what, b) in [("psi", self.psi), ("r", self.r)] {
            check_len(what, b, len)?;
        }
        let lv = (["lv"], self.lv.map(|b| [b]));
        let (l, lu, [lv]) = touched(("lu", self.lu), lv, cg, len)?;
        let mut storages = vec![self.psi.storage(), self.r.storage()];
        storages.extend(l);
        let far = BhFar {
            psi: dptr(self.psi),
            lu,
            lv,
        };
        let r = dptr(self.r);
        Ok((storages, BhFields { far, r }))
    }
}

/// The wave entering a slice of [`BhKernels::slice_tangent`].
#[derive(Clone, Copy)]
pub enum BhEntry<'a> {
    /// The first slice's, the probe `[nprobe, ny, nx]`.
    Probe(&'a DeviceBuffer<C32>),
    /// The first slice's under the position model, the stack's moved probes of `nf_max *
    /// nprobe` planes (`crate::gpu::PositionKernels::shift_w`).
    Moved(&'a DeviceBuffer<C32>),
    /// A later slice's, the stack's waves that entered it, of `nf_max * nmode` planes like
    /// its `psi`.
    Waves(&'a DeviceBuffer<C32>),
}

impl<'a> BhEntry<'a> {
    /// Slice `x` entered by `self` as the kernels read them, checked for `g` under `dims`,
    /// with the two buffers' storages.
    fn slice_in(
        self,
        x: &'a DeviceBuffer<C32>,
        dims: &ViewDims,
        g: &StackGeom,
    ) -> Result<(BhSliceIn, [&'a RawDeviceBuffer; 2]), Error> {
        let nf_max = g.nf_max as usize;
        let (what, e, planes, kind) = match self {
            Self::Probe(p) => ("probe", p, dims.nprobe as usize, BH_ENTRY_PROBE),
            Self::Moved(w) => (
                "moved probes",
                w,
                nf_max * dims.nprobe as usize,
                BH_ENTRY_MOVED,
            ),
            Self::Waves(w) => ("waves", w, nf_max * dims.nmode as usize, BH_ENTRY_WAVES),
        };
        check_len("slice", x, dims.obj_len())?;
        check_len(what, e, planes * dims.n())?;
        let s = BhSliceIn {
            slice: dptr(x),
            entry: dptr(e),
            kind,
        };
        Ok((s, [x.storage(), e.storage()]))
    }
}

/// A multislice object's last slice under the position model, whose exit plane's phase moves
/// with the offsets ([`crate::gpu::SlicePhase::offsets`]): `slice`, the wave `entry` entering
/// it, and `quad`, the phase's ([`crate::gpu::SlicePhase::quad`]).
#[derive(Clone, Copy)]
pub struct BhLastSlice<'a> {
    pub slice: &'a DeviceBuffer<C32>,
    pub entry: BhEntry<'a>,
    pub quad: f32,
}

/// One stack's derivatives through a multislice object, each of `nf_max * nmode` planes
/// like its `psi`: `lu` along `u`, `muu` the second along `(u, u)`, and with CG `cg`: `lv`
/// along `v` and the second along `(u, v)` and `(v, v)`.
pub struct BhTangentsMut<'a> {
    pub lu: &'a mut DeviceBuffer<C32>,
    pub muu: &'a mut DeviceBuffer<C32>,
    pub cg: Option<[&'a mut DeviceBuffer<C32>; 3]>,
}

impl BhTangentsMut<'_> {
    /// `lu` and with CG `lv`, as [`BhKernels::pos_spec`] writes them.
    pub fn first(&mut self) -> (&mut DeviceBuffer<C32>, Option<&mut DeviceBuffer<C32>>) {
        (&mut *self.lu, self.cg.as_mut().map(|[lv, ..]| &mut **lv))
    }

    /// `muu` and with CG the second derivatives along `(u, v)` and `(v, v)`, as
    /// [`BhKernels::pos_second`] writes them.
    pub fn second(&mut self) -> (&mut DeviceBuffer<C32>, Option<[&mut DeviceBuffer<C32>; 2]>) {
        let m = self.cg.as_mut().map(|[_, uv, vv]| [&mut **uv, &mut **vv]);
        (&mut *self.muu, m)
    }

    /// The second derivatives as [`BhKernels::sliced_hess_frames`] reads them.
    pub fn second_ref(&self) -> BhSecondRef<'_> {
        BhSecondRef {
            uu: self.muu,
            cg: self.cg.as_ref().map(|[_, uv, vv]| [&**uv, &**vv]),
        }
    }

    /// The tangents a launch with `cg` or without touches, each checked as `len` long
    /// ([`touched`]); their storages and pointers.
    fn touched(&self, cg: bool, len: usize) -> Result<(Vec<&RawDeviceBuffer>, BhTangents), Error> {
        let lv = (["lv"], self.cg.as_ref().map(|[lv, ..]| [&**lv]));
        let (mut s, lu, [lv]) = touched(("lu", &*self.lu), lv, cg, len)?;
        let m = self.second_ref();
        let (m, muu, [muv, mvv]) = touched(("muu", m.uu), (["muv", "mvv"], m.cg), cg, len)?;
        s.extend(m);
        Ok((
            s,
            BhTangents {
                lu,
                lv,
                muu,
                muv,
                mvv,
            },
        ))
    }
}

/// The exit waves' second derivatives [`BhKernels::sliced_hess_frames`] reads, each of
/// `nf_max * nmode` planes like `psi`: along `(u, u)`, and with CG along `(u, v)` and `(v, v)`.
pub struct BhSecondRef<'a> {
    pub uu: &'a DeviceBuffer<C32>,
    pub cg: Option<[&'a DeviceBuffer<C32>; 2]>,
}

/// One batch of an epoch: its epoch positions, the share of the epoch's frames its step
/// stands for (`α` is scaled by it), its record in the step records, and whether its step
/// takes the offsets ([`BhKernels::pos_dots`]).
#[derive(Clone, Debug)]
pub struct BhBatch {
    pub frames: Range<u32>,
    pub frac: f32,
    pub record: u32,
    pub positions: bool,
}

impl BhBatch {
    /// Errors unless the step takes the offsets.
    fn positions(&self) -> Result<(), Error> {
        if !self.positions {
            return Err(Error::Shape(format!(
                "batch {:?} steps no offsets",
                self.frames
            )));
        }
        Ok(())
    }
}

/// The offsets' buffers of BH under the position model for a scan of `dims`: the gradient
/// partials in `nseg` segments of each frame's row, each written by one of
/// [`BhKernels::pos_grad`], [`BhKernels::win_grad`] (a segment per slice) or
/// [`BhKernels::phase_grad`], `u`'s offsets by epoch position, and the last direction's by
/// scan frame, which with CG are `v`'s.
pub struct BhPositions {
    dims: ViewDims,
    nseg: usize,
    /// `[nframe, nseg, nchunk, 2]` by epoch position.
    grads: DeviceBuffer<f32>,
    /// `[nframe, 2]` by epoch position.
    ud: DeviceBuffer<f32>,
    /// `[nframe, 2]` by scan frame.
    prev: DeviceBuffer<f32>,
}

impl BhPositions {
    pub fn zeroed(stream: &CudaStream, dims: &ViewDims, nseg: usize) -> Result<Self, Error> {
        dims.check()?;
        if nseg == 0 {
            return Err(Error::Shape("no gradient segment".into()));
        }
        let nframe = dims.nframe as usize;
        Ok(Self {
            dims: *dims,
            nseg,
            grads: DeviceBuffer::zeroed(stream, 2 * nframe * nseg * chunks(dims.n()))?,
            ud: DeviceBuffer::zeroed(stream, 2 * nframe)?,
            prev: DeviceBuffer::zeroed(stream, 2 * nframe)?,
        })
    }

    /// Segment `seg` of the partials, as the kernels write it.
    fn partials(&self, seg: usize) -> Result<BhPartials, Error> {
        if seg >= self.nseg {
            return Err(Error::Shape(format!("segment {seg} of {}", self.nseg)));
        }
        Ok(BhPartials {
            out: dptr(&self.grads),
            seg: seg as u32,
            nseg: self.nseg as u32,
        })
    }
}

/// The offsets as a variable of a step: `bufs` through `order`, the epoch's; `u`'s offsets
/// are their gradient times `scale` (pty-chi's position `ρ²`), and the update moves them by
/// `step α D` (pty-chi's position step size).
pub struct BhOffsets<'a> {
    pub bufs: &'a mut BhPositions,
    pub order: &'a EpochOrder,
    pub scale: f32,
    pub step: f32,
}

impl BhOffsets<'_> {
    fn check(&self) -> Result<&ViewDims, Error> {
        let dims = self.order.views.dims();
        same_dims(&self.bufs.dims, dims)?;
        Ok(dims)
    }

    fn dirs(&self, cg: bool) -> BhPosDirs {
        BhPosDirs {
            ud: dptr(&self.bufs.ud),
            prev: dptr(&self.bufs.prev),
            order: dptr(self.order.ids.order()),
            cg: cg.into(),
        }
    }
}

/// What the Hessian of a step along a [`BhDisp`] reads of the moved probes: `shift`, the
/// estimate's in epoch order; the spectra of the displacement's probe parts, `u`'s unscaled
/// and with CG `v`'s; and the offsets.
pub struct BhMovesRef<'a> {
    shift: Shift<'a>,
    au: &'a DeviceBuffer<C32>,
    av: Option<&'a DeviceBuffer<C32>>,
    off: &'a BhOffsets<'a>,
    /// `d.u_scale`, `u`'s probe part over `au`.
    probe_scale: f32,
}

impl<'a> BhMovesRef<'a> {
    /// For a step along `d`: `au` the spectra of `d.u`'s probe part (unscaled) and `av` those
    /// of `d.v`'s, given exactly when `d.v` is.
    pub fn new(
        d: &BhDisp<'_>,
        shift: Shift<'a>,
        au: ProbeSpectrum<'a>,
        av: Option<ProbeSpectrum<'a>>,
        off: &'a BhOffsets<'a>,
    ) -> Result<Self, Error> {
        if av.is_some() != d.v.is_some() {
            return Err(Error::Shape(
                "v's probe spectra are given exactly with v".into(),
            ));
        }
        Ok(Self {
            shift,
            au: au.spectra(),
            av: av.map(|s| s.spectra()),
            off,
            probe_scale: d.u_scale,
        })
    }

    fn check(&self) -> Result<&ViewDims, Error> {
        let dims = self.off.check()?;
        let len = dims.nprobe as usize * dims.n();
        check_len("spectra", self.shift.spectrum.spectra(), len)?;
        check_len("u spectra", self.au, len)?;
        if let Some(av) = self.av {
            check_len("v spectra", av, len)?;
        }
        check_len("offsets", self.shift.offsets, 2 * dims.nframe as usize)?;
        Ok(dims)
    }

    fn storages(&self) -> Vec<&RawDeviceBuffer> {
        let b = &self.off.bufs;
        let mut s = vec![
            self.shift.spectrum.spectra().storage(),
            self.shift.offsets.storage(),
            self.au.storage(),
            b.ud.storage(),
            b.prev.storage(),
            self.off.order.ids.order().storage(),
            self.off.order.views.addr().storage(),
        ];
        s.extend(self.av.map(|b| b.storage()));
        s
    }

    /// Without `av`, `au` stands in for it; the kernels never read it.
    fn ptrs(&self) -> BhMoves {
        BhMoves {
            a: dptr(self.shift.spectrum.spectra()),
            au: dptr(self.au),
            av: dptr(self.av.unwrap_or(self.au)),
            offsets: dptr(self.shift.offsets),
            probe_scale: self.probe_scale,
            dirs: self.off.dirs(self.av.is_some()),
        }
    }
}

/// What the Hessian of a step along a [`BhDisp`] reads of the moved probes with the offsets
/// held: the spectra of the displacement's probe parts, `u`'s unscaled and with CG `v`'s; the
/// offsets of the exit waves by epoch position (a [`Shift`]'s); and `order`, the epoch's.
pub struct BhMovedRef<'a> {
    order: &'a EpochOrder,
    offsets: &'a DeviceBuffer<f32>,
    au: &'a DeviceBuffer<C32>,
    av: Option<&'a DeviceBuffer<C32>>,
    /// `d.u_scale`, `u`'s probe part over `au`.
    probe_scale: f32,
}

impl<'a> BhMovedRef<'a> {
    /// For a step along `d`: `au` the spectra of `d.u`'s probe part (unscaled) and `av` those
    /// of `d.v`'s, given exactly when `d.v` is.
    pub fn new(
        d: &BhDisp<'_>,
        order: &'a EpochOrder,
        offsets: &'a DeviceBuffer<f32>,
        au: ProbeSpectrum<'a>,
        av: Option<ProbeSpectrum<'a>>,
    ) -> Result<Self, Error> {
        if av.is_some() != d.v.is_some() {
            return Err(Error::Shape(
                "v's probe spectra are given exactly with v".into(),
            ));
        }
        Ok(Self {
            order,
            offsets,
            au: au.spectra(),
            av: av.map(|s| s.spectra()),
            probe_scale: d.u_scale,
        })
    }

    fn check(&self) -> Result<&ViewDims, Error> {
        let dims = self.order.views.dims();
        let len = dims.nprobe as usize * dims.n();
        check_len("u spectra", self.au, len)?;
        if let Some(av) = self.av {
            check_len("v spectra", av, len)?;
        }
        check_len("offsets", self.offsets, 2 * dims.nframe as usize)?;
        Ok(dims)
    }

    fn storages(&self) -> Vec<&RawDeviceBuffer> {
        let mut s = vec![
            self.offsets.storage(),
            self.au.storage(),
            self.order.views.addr().storage(),
        ];
        s.extend(self.av.map(|b| b.storage()));
        s
    }

    /// Without `av`, `au` stands in for it; the kernel never reads it.
    fn ptrs(&self) -> BhMovedProbes {
        BhMovedProbes {
            au: dptr(self.au),
            av: dptr(self.av.unwrap_or(self.au)),
            offsets: dptr(self.offsets),
            probe_scale: self.probe_scale,
            cg: self.av.is_some().into(),
        }
    }
}

/// One stack under the position model as [`BhKernels::pos_near`] reads it, each in
/// [`Kernels::obj_probe_to_psi`]'s layout: the object, the moved probes `w` (`[nf_max *
/// nprobe, ny, nx]`, as `crate::gpu::PositionKernels::shift_w` leaves them), the residual
/// after its inverse FFT, and `lu` and with CG `lv` the inverse-FFT'd [`BhKernels::pos_spec`].
pub struct BhNearMut<'a> {
    pub obj: &'a DeviceBuffer<C32>,
    pub w: &'a DeviceBuffer<C32>,
    pub r: &'a DeviceBuffer<C32>,
    pub lu: &'a mut DeviceBuffer<C32>,
    pub lv: Option<&'a mut DeviceBuffer<C32>>,
}

/// One stack's far fields [`BhKernels::pos_far`] reads, each of `nf_max * nmode` planes: the
/// forward-FFT'd exit waves, the forward-FFT'd [`BhKernels::pos_near`] `lu` and with CG `lv`,
/// and the forward-FFT'd [`BhKernels::obj_conj`] of the residual.
pub struct BhMovedFarRef<'a> {
    pub psi: &'a DeviceBuffer<C32>,
    pub lu: &'a DeviceBuffer<C32>,
    pub lv: Option<&'a DeviceBuffer<C32>>,
    pub y: &'a DeviceBuffer<C32>,
}

/// One stack's far fields [`BhKernels::moved_far`] reads, each of `nf_max * nmode` planes:
/// the forward-FFT'd exit waves and the forward-FFT'd [`BhKernels::pos_near`] `lu` and, with
/// CG, `lv`. Without `lv` the step is steepest descent.
#[derive(Clone, Copy)]
pub struct BhFarRef<'a> {
    pub psi: &'a DeviceBuffer<C32>,
    pub lu: &'a DeviceBuffer<C32>,
    pub lv: Option<&'a DeviceBuffer<C32>>,
}

/// One stack under full-object sampling as [`BhKernels::win_grad`] and
/// [`BhKernels::win_hess`] read it: the residual after its inverse FFT, in
/// [`Kernels::obj_probe_to_psi`]'s layout, and the estimate's probe.
#[derive(Clone, Copy)]
pub struct BhWinRef<'a> {
    pub r: &'a DeviceBuffer<C32>,
    pub probe: &'a DeviceBuffer<C32>,
}

/// The windows [`BhKernels::win_hess`] reads, each three blocks of a stack's windows: the
/// estimate's second derivatives over `(y, y)`, `(y, x)` and `(x, x)`, and `u_O`'s and for
/// CG `v_O`'s windows, blocks 1 and 2 their derivatives over `y` and `x`. Without `v` the
/// step is steepest descent.
#[derive(Clone, Copy)]
pub struct BhWinSecond<'a> {
    pub second: &'a DeviceBuffer<C32>,
    pub u: &'a DeviceBuffer<C32>,
    pub v: Option<&'a DeviceBuffer<C32>>,
}

/// One array BH updates, of `n` values: the estimate, the last direction and the gradient
/// sums, the gradient scaled by `scale` (the probe's `ρ²`).
pub struct BhVar<'a> {
    pub x: &'a mut DeviceBuffer<C32>,
    pub prev: &'a mut DeviceBuffer<C32>,
    pub g: &'a mut UpdateSums,
    pub n: u32,
    pub scale: f32,
}

/// An array with dot pairs in [`BhParts`].
#[derive(Clone, Copy, Debug)]
pub enum BhArray {
    Object,
    Probe,
    /// Slice `s` past the first of a multislice object.
    Slice(usize),
}

/// The partial sums of one step over a scan: [`CG_BLOCKS`] dot pairs per [`BhArray`] (the
/// object, the probe, then each slice past the first in order), the offsets' pair, then a
/// [`BH_HESS`] triple per chunk of [`GAMMA_CHUNK`] pixels of each epoch position. It alone
/// places them.
pub struct BhParts {
    buf: DeviceBuffer<f32>,
    /// The arrays with dot pairs: the object, the probe and the slices past the first.
    arrays: usize,
    nframe: u32,
    /// Triples per epoch position.
    nchunk: usize,
}

impl BhParts {
    /// Zeroed partial sums over a scan of `dims` through `deeper` slices past the first.
    pub fn zeroed(stream: &Arc<CudaStream>, dims: &ViewDims, deeper: usize) -> Result<Self, Error> {
        let (arrays, nchunk) = (2 + deeper, chunks(dims.n()));
        Ok(Self {
            buf: DeviceBuffer::zeroed(stream, Self::hess_at(arrays, nchunk, dims.nframe))?,
            arrays,
            nframe: dims.nframe,
            nchunk,
        })
    }

    pub fn buf(&self) -> &DeviceBuffer<f32> {
        &self.buf
    }

    pub fn buf_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.buf
    }

    /// The first of the `k`-th array's dot pairs; the offsets' pair follows the last array's.
    fn pair(k: usize) -> usize {
        k * CG_BLOCKS as usize
    }

    /// The first of `a`'s dot pairs, an error for a slice the parts have none for.
    pub(crate) fn dots(&self, a: BhArray) -> Result<usize, Error> {
        let k = match a {
            BhArray::Object => 0,
            BhArray::Probe => 1,
            BhArray::Slice(s) => 2 + s,
        };
        match k < self.arrays {
            true => Ok(Self::pair(k)),
            false => Err(Error::Shape(format!(
                "dot pairs of {a:?} in parts of {} slices past the first",
                self.arrays - 2
            ))),
        }
    }

    /// The pairs a step sums: every array's, and with `positions` the offsets'.
    fn ndots(&self, positions: bool) -> u32 {
        (Self::pair(self.arrays) + usize::from(positions)) as u32
    }

    /// The first float of the offsets' pair.
    fn pos(&self) -> usize {
        2 * Self::pair(self.arrays)
    }

    /// The first float of epoch position `p`'s triples.
    fn hess(&self, p: u32) -> usize {
        Self::hess_at(self.arrays, self.nchunk, p)
    }

    /// [`Self::hess`] past `arrays` arrays' pairs and `nchunk` triples a position; at `p =
    /// nframe` the length.
    fn hess_at(arrays: usize, nchunk: usize, p: u32) -> usize {
        2 * Self::pair(arrays) + 2 + BH_HESS as usize * p as usize * nchunk
    }

    fn check(&self, dims: &ViewDims) -> Result<(), Error> {
        match (self.nframe, self.nchunk) == (dims.nframe, chunks(dims.n())) {
            true => Ok(()),
            false => Err(Error::Shape(format!(
                "parts of {} frames of {} chunks for a scan of {dims:?}",
                self.nframe, self.nchunk
            ))),
        }
    }

    /// The device address of float `at`.
    fn ptr(&self, at: usize) -> *mut f32 {
        dptr::<f32, f32>(&self.buf).wrapping_add(at)
    }
}

/// Chunks of [`GAMMA_CHUNK`] pixels per frame of `n` pixels.
fn chunks(n: usize) -> usize {
    n.div_ceil(GAMMA_CHUNK as usize)
}

/// One block of [`PIXEL_THREADS`] per chunk of a frame of `n` pixels and per frame of
/// `nf_max`, as the Hessian triples are summed.
fn hess_grid(nf_max: u32, n: usize) -> Result<LaunchConfig, Error> {
    if nf_max > u16::MAX as u32 {
        return Err(Error::Shape(format!("{nf_max} frames exceed grid y")));
    }
    let nchunk =
        u32::try_from(chunks(n)).map_err(|_| Error::Shape(format!("{n} pixels exceed grid x")))?;
    Ok(LaunchConfig {
        grid_dim: (nchunk, nf_max, 1),
        block_dim: (PIXEL_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Values of one block of a stack's windows of `g.nf_max` frames over a scan of `dims`.
fn window_block(dims: &ViewDims, g: &StackGeom) -> usize {
    dims.nobj as usize * g.nf_max as usize * dims.n()
}

/// A launch's buffers along a step, `cg` its CG: `first`, and `with_cg`, named, which every BH
/// kernel touches only with CG and a launch takes exactly with it. Each is checked as `len`
/// long under its name; returns their storages, to lease, and their pointers, `first`'s
/// standing in for `with_cg`'s without CG, which the kernels never read.
fn touched<'b, const N: usize>(
    first: (&str, &'b DeviceBuffer<C32>),
    (names, with_cg): ([&str; N], Option<[&'b DeviceBuffer<C32>; N]>),
    cg: bool,
    len: usize,
) -> Result<(Vec<&'b RawDeviceBuffer>, *mut f32, [*mut f32; N]), Error> {
    if with_cg.is_some() != cg {
        let names = names.join(", ");
        return Err(Error::Shape(format!("{names} given exactly with CG")));
    }
    let named = with_cg.map(|b| names.into_iter().zip(b));
    let mut storages = Vec::with_capacity(1 + N);
    for (what, b) in std::iter::once(first).chain(named.into_iter().flatten()) {
        check_len(what, b, len)?;
        storages.push(b.storage());
    }
    let ptrs = with_cg.map_or([dptr(first.1); N], |b| b.map(dptr));
    Ok((storages, dptr(first.1), ptrs))
}

/// Values of the object and of the probe.
fn lens(dims: &ViewDims) -> [u32; 2] {
    // `Views::new` checked both fit the kernels' u32 indices.
    [
        dims.obj_len() as u32,
        (dims.nprobe as usize * dims.n()) as u32,
    ]
}

impl BhKernels {
    /// Binds the BH kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = bh_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// The stack's residual from its forward-FFT'd (unnormalized) exit waves `psi`: `r =
    /// psi · 2 (1 − √obs / a) / n`, `a = √(Σ_m |psi|² / n) + 1e-8`, zero where masked. The
    /// unnormalized inverse FFT of `r`, back-projected by [`Kernels::psi_to_obj`] and
    /// [`Kernels::psi_to_probe`], gives the sums' `num` as the gradient `2 ∂F/∂x*`.
    pub fn residual(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        stack: StackRef<'_>,
        psi: &DeviceBuffer<C32>,
        r: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let dims = scan.check()?;
        let g = dims.geom(stack)?;
        let (n, nf_max) = (dims.n(), g.nf_max as usize);
        check_len("psi", psi, nf_max * dims.nmode as usize * n)?;
        check_len("r", r, nf_max * dims.nmode as usize * n)?;
        let config = grid_1d(nf_max * n)?;
        let reads = [
            scan.counts.storage(),
            scan.order.ids.order().storage(),
            psi.storage(),
            stack.storage(),
        ];
        self.base.launch(ctx, &reads, &[r.storage()], |stream| {
            // SAFETY: lengths checked above; `EpochOrder` holds frame ids below `nframe`.
            unsafe {
                self.module.bh_residual(
                    stream,
                    config,
                    dptr(psi),
                    dptr(r),
                    scan.obs(),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The stack's far-field derivatives before their forward FFT, in
    /// [`Kernels::obj_probe_to_psi`]'s layout, at `x = [O, P]`: `lu = q (u_P O + P u_O)` and,
    /// with `d.v`, `lv` likewise along `v`.
    pub fn far(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        x: [&DeviceBuffer<C32>; 2],
        d: &BhDisp<'_>,
        (lu, lv): (&mut DeviceBuffer<C32>, Option<&mut DeviceBuffer<C32>>),
    ) -> Result<(), Error> {
        let dims = views.dims();
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        for (b, n, what) in [(x[0], lens(dims)[0], "obj"), (x[1], lens(dims)[1], "probe")] {
            check_len(what, b, n as usize)?;
        }
        d.check(dims)?;
        let lv = (["lv"], lv.as_deref().map(|b| [b]));
        let (writes, lu, [lv]) = touched(("lu", &*lu), lv, d.v.is_some(), len)?;
        let config = grid_1d(len)?;
        let mut reads = vec![views.addr().storage(), stack.storage()];
        reads.extend(x.iter().map(|b| b.storage()));
        reads.extend(d.storages());
        self.base.launch(ctx, &reads, &writes, |stream| {
            let x = ObjProbePtrs {
                obj: dptr(x[0]),
                probe: dptr(x[1]),
            };
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object with valid mode indices.
            unsafe {
                self.module.bh_far(
                    stream,
                    config,
                    x,
                    d.ptrs(),
                    dptr::<ViewAddr, ViewAddr>(views.addr()),
                    lu,
                    lv,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The Hessian triples `(H(u, u), H(u, v), H(v, v))` along `d` of the stack's frames (the
    /// last two 0 without `d.v`), each over a chunk of [`GAMMA_CHUNK`] pixels in a fixed-order
    /// block sum, into the triples of [`BhParts`] at the frames' epoch positions; other
    /// positions' triples are left as they are. `d`'s object parts are read through
    /// `f.views`, which differ from the scan's only in their object.
    pub fn hess_frames(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        stack: StackRef<'_>,
        f: &BhFieldsRef<'_>,
        d: &BhDisp<'_>,
        parts: &mut BhParts,
    ) -> Result<(), Error> {
        let sd = scan.check()?;
        let dims = f.views.dims();
        if (ViewDims {
            obj: sd.obj,
            ..*dims
        }) != *sd
        {
            return Err(Error::Shape(format!(
                "views of {dims:?} for a scan of {sd:?}"
            )));
        }
        let g = dims.geom(stack)?;
        let n = dims.n();
        let len = g.nf_max as usize * dims.nmode as usize * n;
        let (read, fields) = f.read(d.v.is_some(), len)?;
        d.check(dims)?;
        parts.check(dims)?;
        let config = hess_grid(g.nf_max, n)?;
        let mut reads = vec![
            f.views.addr().storage(),
            scan.counts.storage(),
            scan.order.ids.order().storage(),
            stack.storage(),
        ];
        reads.extend(read);
        reads.extend(d.storages());
        let out = parts.ptr(parts.hess(0));
        self.base
            .launch(ctx, &reads, &[parts.buf.storage()], |stream| {
                // SAFETY: lengths checked above; one block of PIXEL_THREADS per chunk and frame;
                // `EpochOrder` holds frame ids below `nframe`.
                unsafe {
                    self.module.bh_hess_frames(
                        stream,
                        config,
                        fields,
                        d.ptrs(),
                        dptr::<ViewAddr, ViewAddr>(f.views.addr()),
                        scan.obs(),
                        out,
                        stack.ctl() as *const StackCtl,
                        g,
                    )
                }
            })
    }

    /// One slice `x` of a multislice object, which `entry` enters, on the stack's
    /// derivatives `t` along `d`, in place: the slice's exit `x w` of the wave `w` entering it
    /// changes along `u` by `x l_u + u_x w` and to second order along `(u, v)` by `x m_uv +
    /// u_x l_v + v_x l_u`, `u_x` and `v_x` the slice's parts `d.u[0]`, `d.v[0]` through
    /// `views`. Entered by the probe, `l_u` and `l_v` are `d`'s probe parts (`u`'s times
    /// `d.u_scale`), `m` is 0 and `t` only written; else `t` holds `w`'s and `d`'s probe parts
    /// are unread (entered by the moved probes, `t` holds theirs: the inverse FFTs of
    /// [`Self::pos_spec`] and [`Self::pos_second`]). `t.cg` is given exactly with `d.v`.
    pub fn slice_tangent(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        (x, entry): (&DeviceBuffer<C32>, BhEntry<'_>),
        d: &BhDisp<'_>,
        t: &mut BhTangentsMut<'_>,
    ) -> Result<(), Error> {
        let dims = views.dims();
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        let (s, entered) = entry.slice_in(x, dims, &g)?;
        d.check(dims)?;
        let (writes, t) = t.touched(d.v.is_some(), len)?;
        let config = grid_1d(len)?;
        let mut reads = vec![views.addr().storage(), stack.storage()];
        reads.extend(entered);
        reads.extend(d.storages());
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object with valid mode indices.
            unsafe {
                self.module.bh_slice_tangent(
                    stream,
                    config,
                    s,
                    d.ptrs(),
                    t,
                    dptr::<ViewAddr, ViewAddr>(views.addr()),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// [`Self::hess_frames`] through a multislice object: the near term from `m`, the exit
    /// waves' second derivatives of [`Self::slice_tangent`], and `f.r` with the adjoint of
    /// the last slice's plane phase already on it, as on `f.lu` and `f.lv` its phase; `f.lv`
    /// is given exactly with `m.cg`.
    pub fn sliced_hess_frames(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        stack: StackRef<'_>,
        f: &BhFieldsRef<'_>,
        m: &BhSecondRef<'_>,
        parts: &mut BhParts,
    ) -> Result<(), Error> {
        let dims = scan.check()?;
        let g = dims.geom(stack)?;
        let n = dims.n();
        let len = g.nf_max as usize * dims.nmode as usize * n;
        let cg = m.cg.is_some();
        let (read, fields) = f.read(cg, len)?;
        let (second, uu, [uv, vv]) = touched(("muu", m.uu), (["muv", "mvv"], m.cg), cg, len)?;
        parts.check(dims)?;
        let config = hess_grid(g.nf_max, n)?;
        let mut reads = vec![
            scan.counts.storage(),
            scan.order.ids.order().storage(),
            stack.storage(),
        ];
        reads.extend(read);
        reads.extend(second);
        let second = BhSecond { uu, uv, vv };
        let out = parts.ptr(parts.hess(0));
        self.base
            .launch(ctx, &reads, &[parts.buf.storage()], |stream| {
                // SAFETY: lengths checked above; one block of PIXEL_THREADS per chunk and frame;
                // `EpochOrder` holds frame ids below `nframe`.
                unsafe {
                    self.module.bh_sliced_hess_frames(
                        stream,
                        config,
                        fields,
                        second,
                        cg.into(),
                        scan.obs(),
                        out,
                        stack.ctl() as *const StackCtl,
                        g,
                    )
                }
            })
    }

    /// The step of `batch` along `d` into its record of `steps` (`(β, α, top, bottom)`,
    /// [`BH_STEP`] floats each) from the partials [`crate::tile::TileDots::dots`] (with
    /// `batch.positions` and [`Self::pos_dots`]) and [`Self::hess_frames`] (or the position
    /// model's) left in `parts`: `β = H(u, v) / H(v, v)` with `d.v`, else 0;
    /// `top = ⟨u, Pu⟩ − β ⟨u, v⟩ = −⟨g, D⟩` and `bottom = H(D, D)` for `D = β v − Pu`;
    /// `α = top / bottom * frac`.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        parts: &BhParts,
        d: &BhDisp<'_>,
        batch: &BhBatch,
        steps: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let frames = &batch.frames;
        if frames.start > frames.end || frames.end > dims.nframe {
            return Err(Error::Shape(format!(
                "batch {frames:?} outside {} frames",
                dims.nframe
            )));
        }
        parts.check(dims)?;
        check_len("steps", steps, step_len(batch.record))?;
        let nchunk = chunks(dims.n());
        let nhess = u32::try_from(frames.len() * nchunk)
            .map_err(|_| Error::Shape(format!("batch {frames:?} exceeds u32 triples")))?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let out = dptr::<f32, f32>(steps).wrapping_add(step_len(batch.record) - BH_STEP as usize);
        self.base
            .launch(ctx, &[parts.buf.storage()], &[steps.storage()], |stream| {
                // SAFETY: lengths checked above; the batch's `nhess` triples lie in `parts`
                // (`frames.end <= nframe`); one PIXEL_THREADS block.
                unsafe {
                    self.module.bh_step(
                        stream,
                        config,
                        parts.ptr(0),
                        parts.ndots(batch.positions),
                        parts.ptr(parts.hess(frames.start)),
                        nhess,
                        d.v.is_some().into(),
                        batch.frac,
                        out,
                    )
                }
            })
    }

    /// `var`'s step from record `record` of `steps`: `D = β prev − scale g`, `x += α D`,
    /// `prev = D`; consumes the gradient sums (zero after).
    pub fn update(
        &self,
        ctx: &ExecutionContext,
        var: BhVar<'_>,
        steps: &DeviceBuffer<f32>,
        record: u32,
    ) -> Result<(), Error> {
        let n = var.n as usize;
        check_len("x", var.x, n)?;
        check_len("prev", var.prev, n)?;
        var.g.check(n)?;
        check_len("steps", steps, step_len(record))?;
        if n == 0 {
            return Ok(());
        }
        let config = LaunchConfig::for_num_elems(var.n);
        let step = dptr::<f32, f32>(steps).wrapping_add(step_len(record) - BH_STEP as usize);
        let writes = [
            var.x.storage(),
            var.prev.storage(),
            var.g.num.storage(),
            var.g.den.storage(),
        ];
        self.base
            .launch(ctx, &[steps.storage()], &writes, |stream| {
                let g = SumPtrs {
                    num: dptr(&var.g.num),
                    den: dptr(&var.g.den),
                };
                // SAFETY: lengths checked above.
                unsafe {
                    self.module.bh_update(
                        stream,
                        config,
                        dptr(var.x),
                        dptr(var.prev),
                        g,
                        var.n,
                        var.scale,
                        step,
                    )
                }
            })
    }

    /// `buf *= T / √(ny nx)` on the stack's planes `[nf_max * nmode, ny, nx]`, or with `conj`
    /// by `conj(T)`: between a forward and an inverse FFT of `buf` this makes them angular
    /// spectrum of the transfer function `tf`, `[ny, nx]` on the unshifted frequency grid,
    /// at the far field's norm, or its adjoint.
    pub fn transfer(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        tf: &DeviceBuffer<C32>,
        buf: &mut DeviceBuffer<C32>,
        conj: bool,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        check_len("transfer function", tf, dims.n())?;
        check_len("buf", buf, len)?;
        let config = grid_1d(len)?;
        let scale = 1.0 / (dims.n() as f32).sqrt();
        let reads = [tf.storage(), stack.storage()];
        self.base.launch(ctx, &reads, &[buf.storage()], |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.bh_transfer(
                    stream,
                    config,
                    dptr(buf),
                    dptr::<_, f32>(tf).cast_const(),
                    scale,
                    conj as u32,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// `y = conj(O_v) r` of each view `v` of the stack, `[nf_max * nmode, ny, nx]` in
    /// [`Kernels::obj_probe_to_psi`]'s layout like `r`, `O_v` the object `v` samples at each
    /// stored pixel: before its forward FFT, what [`Self::pos_grad`] and [`Self::pos_far`]
    /// read of the residual.
    pub fn obj_conj(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        r: &DeviceBuffer<C32>,
        y: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let dims = views.dims();
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        check_len("obj", obj, dims.obj_len())?;
        check_len("r", r, len)?;
        check_len("y", y, len)?;
        let config = grid_1d(len)?;
        let reads = [
            views.addr().storage(),
            obj.storage(),
            r.storage(),
            stack.storage(),
        ];
        self.base.launch(ctx, &reads, &[y.storage()], |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object.
            unsafe {
                self.module.bh_obj_conj(
                    stream,
                    config,
                    dptr(obj),
                    dptr::<ViewAddr, ViewAddr>(views.addr()),
                    dptr(r),
                    dptr(y),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The offsets' gradient partials of the stack's frames into segment `seg` of `off`'s,
    /// from `y` the forward-FFT'd [`Self::obj_conj`] of its residual and `shift` the
    /// estimate's moved probes: per chunk of [`GAMMA_CHUNK`] pixels of each frame, the pair
    /// [`Self::pos_dots`] sums with the other segments' to `∂F/∂d`; other frames' partials,
    /// and other segments', are left as they are.
    pub fn pos_grad(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        y: &DeviceBuffer<C32>,
        shift: &Shift<'_>,
        (off, seg): (&mut BhOffsets<'_>, usize),
    ) -> Result<(), Error> {
        let dims = off.check()?;
        let part = off.bufs.partials(seg)?;
        let g = dims.geom(stack)?;
        let n = dims.n();
        check_len("y", y, g.nf_max as usize * dims.nmode as usize * n)?;
        let a = shift.spectrum.spectra();
        check_len("spectra", a, dims.nprobe as usize * n)?;
        check_len("offsets", shift.offsets, 2 * dims.nframe as usize)?;
        let config = chunk_grid(&g, n)?;
        let views = off.order.views.addr();
        let reads = [
            y.storage(),
            a.storage(),
            shift.offsets.storage(),
            views.storage(),
            stack.storage(),
        ];
        let grads = &off.bufs.grads;
        self.base.launch(ctx, &reads, &[grads.storage()], |stream| {
            // SAFETY: lengths checked above; `BhPositions::zeroed` sized `grads` for every
            // epoch position's chunks of every segment, of which `part` is one; one block of
            // PIXEL_THREADS per chunk and frame.
            unsafe {
                self.module.bh_pos_grad(
                    stream,
                    config,
                    dptr(y),
                    dptr(a),
                    dptr(shift.offsets),
                    dptr::<ViewAddr, ViewAddr>(views),
                    part,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The offsets' part of `batch`'s displacements and dots from [`Self::pos_grad`]'s
    /// partials: `u`'s offsets, `off.scale` times the gradient `g`, and into the offsets'
    /// pair of [`BhParts`] `(Σ g u_d, Σ g v_d)`, the second 0 without `d.v`.
    pub fn pos_dots(
        &self,
        ctx: &ExecutionContext,
        batch: &BhBatch,
        d: &BhDisp<'_>,
        off: &mut BhOffsets<'_>,
        parts: &mut BhParts,
    ) -> Result<(), Error> {
        batch.positions()?;
        let dims = off.check()?;
        let span = span(dims, &batch.frames)?;
        // A frame's row of partials holds every segment's chunks.
        let span = BatchSpan {
            nchunk: span.nchunk * off.bufs.nseg as u32,
            ..span
        };
        parts.check(dims)?;
        let dirs = off.dirs(d.v.is_some());
        let b = &off.bufs;
        let reads = [
            b.grads.storage(),
            b.prev.storage(),
            off.order.ids.order().storage(),
        ];
        let writes = [b.ud.storage(), parts.buf.storage()];
        let out = parts.ptr(parts.pos());
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: `span` lies in the epoch's positions, whose buffers `BhPositions::zeroed`
            // sized; the pair lies in `parts`; one PIXEL_THREADS block.
            unsafe {
                self.module.bh_pos_dots(
                    stream,
                    ONE_BLOCK,
                    dptr(&b.grads),
                    span,
                    dirs,
                    off.scale,
                    out,
                )
            }
        })
    }

    /// The spectra of the stack's moved probes' first-order changes along the step's `u`
    /// into `lu` and with CG `v` into `lv`, one per view in [`Kernels::obj_probe_to_psi`]'s
    /// layout, ready for their inverse FFT.
    pub fn pos_spec(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        mv: &BhMovesRef<'_>,
        (lu, lv): (&mut DeviceBuffer<C32>, Option<&mut DeviceBuffer<C32>>),
    ) -> Result<(), Error> {
        let dims = mv.check()?;
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        let lv = (["lv"], lv.as_deref().map(|b| [b]));
        let (writes, lu, [lv]) = touched(("lu", &*lu), lv, mv.av.is_some(), len)?;
        let config = grid_1d(len)?;
        let mut reads = mv.storages();
        reads.push(stack.storage());
        let views = mv.off.order.views.addr();
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view's probe mode.
            unsafe {
                self.module.bh_pos_spec(
                    stream,
                    config,
                    mv.ptrs(),
                    dptr::<ViewAddr, ViewAddr>(views),
                    lu,
                    lv,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The spectra of the stack's moved probes' second-order changes along the step's `(u,
    /// u)` into `muu` and with CG `(u, v)`, `(v, v)` into `m`'s two, laid out as
    /// [`Self::pos_spec`]'s and ready for their inverse FFT.
    pub fn pos_second(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        mv: &BhMovesRef<'_>,
        (muu, m): (&mut DeviceBuffer<C32>, Option<[&mut DeviceBuffer<C32>; 2]>),
    ) -> Result<(), Error> {
        let dims = mv.check()?;
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        let m = (["muv", "mvv"], m.as_ref().map(|[uv, vv]| [&**uv, &**vv]));
        let (writes, muu, [muv, mvv]) = touched(("muu", &*muu), m, mv.av.is_some(), len)?;
        let config = grid_1d(len)?;
        let mut reads = mv.storages();
        reads.push(stack.storage());
        let views = mv.off.order.views.addr();
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view's probe mode.
            unsafe {
                self.module.bh_pos_second(
                    stream,
                    config,
                    mv.ptrs(),
                    dptr::<ViewAddr, ViewAddr>(views),
                    muu,
                    muv,
                    mvv,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The near Hessian triples along `d` of the stack's frames under the position model
    /// into those of [`BhParts`], as [`Self::hess_frames`] places them, from `f.lu` and with
    /// `d.v` `f.lv`, the inverse-FFT'd [`Self::pos_spec`]; then in place the exit waves'
    /// first-order changes `lu = u_O W + O δ_u W` and `lv` likewise, ready for their forward
    /// FFT. `views` are the epoch's.
    pub fn pos_near(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        f: BhNearMut<'_>,
        d: &BhDisp<'_>,
        parts: &mut BhParts,
    ) -> Result<(), Error> {
        let dims = views.dims();
        let g = dims.geom(stack)?;
        let n = dims.n();
        let len = g.nf_max as usize * dims.nmode as usize * n;
        check_len("obj", f.obj, dims.obj_len())?;
        check_len("w", f.w, g.nf_max as usize * dims.nprobe as usize * n)?;
        check_len("r", f.r, len)?;
        let lv = (["lv"], f.lv.as_deref().map(|b| [b]));
        let (mut writes, lu, [lv]) = touched(("lu", &*f.lu), lv, d.v.is_some(), len)?;
        d.check(dims)?;
        parts.check(dims)?;
        let config = chunk_grid(&g, n)?;
        let mut reads = vec![
            views.addr().storage(),
            f.obj.storage(),
            f.w.storage(),
            f.r.storage(),
            stack.storage(),
        ];
        reads.extend(d.storages());
        writes.push(parts.buf.storage());
        let out = parts.ptr(parts.hess(0));
        self.base.launch(ctx, &reads, &writes, |stream| {
            let near = BhNear {
                obj: dptr(f.obj),
                w: dptr(f.w),
                r: dptr(f.r),
                lu,
                lv,
            };
            // SAFETY: lengths checked above; one block of PIXEL_THREADS per chunk and frame;
            // `Views::new` checked every view lies inside the object with valid modes.
            unsafe {
                self.module.bh_pos_near(
                    stream,
                    config,
                    near,
                    d.ptrs(),
                    dptr::<ViewAddr, ViewAddr>(views.addr()),
                    out,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Adds to the stack's frames' triples [`Self::pos_near`] left the far terms and the
    /// moved probes' second-order terms along the step, from the stack's far fields `f`;
    /// `counts` are the scan's, in its own order.
    pub fn pos_far(
        &self,
        ctx: &ExecutionContext,
        counts: &DeviceBuffer<u16>,
        stack: StackRef<'_>,
        f: &BhMovedFarRef<'_>,
        mv: &BhMovesRef<'_>,
        parts: &mut BhParts,
    ) -> Result<(), Error> {
        let dims = mv.check()?;
        let g = dims.geom(stack)?;
        let n = dims.n();
        check_len("counts", counts, dims.nframe as usize * n)?;
        let len = g.nf_max as usize * dims.nmode as usize * n;
        for (what, b) in [("psi", f.psi), ("y", f.y)] {
            check_len(what, b, len)?;
        }
        let lv = (["lv"], f.lv.map(|b| [b]));
        let (l, lu, [lv]) = touched(("lu", f.lu), lv, mv.av.is_some(), len)?;
        parts.check(dims)?;
        let config = chunk_grid(&g, n)?;
        let mut reads = mv.storages();
        reads.extend([
            counts.storage(),
            f.psi.storage(),
            f.y.storage(),
            stack.storage(),
        ]);
        reads.extend(l);
        let far = BhMovedFar {
            far: BhFar {
                psi: dptr(f.psi),
                lu,
                lv,
            },
            y: dptr(f.y),
        };
        let views = mv.off.order.views.addr();
        let out = parts.ptr(parts.hess(0));
        self.base
            .launch(ctx, &reads, &[parts.buf.storage()], |stream| {
                // SAFETY: lengths checked above; one block of PIXEL_THREADS per chunk and frame;
                // `EpochOrder` holds frame ids below `nframe`.
                unsafe {
                    self.module.bh_pos_far(
                        stream,
                        config,
                        far,
                        mv.ptrs(),
                        dptr::<ViewAddr, ViewAddr>(views),
                        dptr(counts),
                        out,
                        stack.ctl() as *const StackCtl,
                        g,
                    )
                }
            })
    }

    /// [`Self::pos_spec`] with the offsets held: `lu = e s Au_p` and with CG `lv = e Av_p`, `e`
    /// each frame's ramp of `mv`'s offsets.
    pub fn moved_spec(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        mv: &BhMovedRef<'_>,
        (lu, lv): (&mut DeviceBuffer<C32>, Option<&mut DeviceBuffer<C32>>),
    ) -> Result<(), Error> {
        let dims = mv.check()?;
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        let lv = (["lv"], lv.as_deref().map(|b| [b]));
        let (writes, lu, [lv]) = touched(("lu", &*lu), lv, mv.av.is_some(), len)?;
        let config = grid_1d(len)?;
        let mut reads = mv.storages();
        reads.push(stack.storage());
        let views = mv.order.views.addr();
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view's probe mode.
            unsafe {
                self.module.bh_moved_spec(
                    stream,
                    config,
                    mv.ptrs(),
                    dptr::<ViewAddr, ViewAddr>(views),
                    lu,
                    lv,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// [`Self::pos_far`] with the offsets held: adds to the stack's frames' triples
    /// [`Self::pos_near`] left only the far terms, from the stack's far fields `f`, the last
    /// two of each triple without `f.lv` left as they are.
    pub fn moved_far(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        stack: StackRef<'_>,
        f: &BhFarRef<'_>,
        parts: &mut BhParts,
    ) -> Result<(), Error> {
        let dims = scan.check()?;
        let g = dims.geom(stack)?;
        let n = dims.n();
        let len = g.nf_max as usize * dims.nmode as usize * n;
        for (what, b) in [("psi", Some(f.psi)), ("lu", Some(f.lu)), ("lv", f.lv)] {
            if let Some(b) = b {
                check_len(what, b, len)?;
            }
        }
        parts.check(dims)?;
        let config = chunk_grid(&g, n)?;
        let mut reads = vec![
            scan.counts.storage(),
            scan.order.ids.order().storage(),
            f.psi.storage(),
            f.lu.storage(),
            stack.storage(),
        ];
        reads.extend(f.lv.map(|b| b.storage()));
        let far = BhFar {
            psi: dptr(f.psi),
            lu: dptr(f.lu),
            // Without `lv`, `lu` stands in; the kernel never reads it.
            lv: dptr(f.lv.unwrap_or(f.lu)),
        };
        let cg = f.lv.is_some().into();
        let out = parts.ptr(parts.hess(0));
        self.base
            .launch(ctx, &reads, &[parts.buf.storage()], |stream| {
                // SAFETY: lengths checked above; one block of PIXEL_THREADS per chunk and frame;
                // `EpochOrder` holds frame ids below `nframe`.
                unsafe {
                    self.module.bh_moved_far(
                        stream,
                        config,
                        far,
                        cg,
                        scan.obs(),
                        out,
                        stack.ctl() as *const StackCtl,
                        g,
                    )
                }
            })
    }

    /// The offsets' update of `batch` from its record of `steps`: per frame `f` of the batch
    /// `D = β prev − u_d`, `offsets[f] += off.step α D` and `prev = D`; `offsets` `[nframe,
    /// 2]` by scan frame.
    pub fn pos_update(
        &self,
        ctx: &ExecutionContext,
        batch: &BhBatch,
        off: &mut BhOffsets<'_>,
        offsets: &mut DeviceBuffer<f32>,
        steps: &DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        batch.positions()?;
        let dims = off.check()?;
        let span = span(dims, &batch.frames)?;
        check_len("offsets", offsets, 2 * dims.nframe as usize)?;
        check_len("steps", steps, step_len(batch.record))?;
        let config = grid_1d(2 * span.nb as usize)?;
        let step = dptr::<f32, f32>(steps).wrapping_add(step_len(batch.record) - BH_STEP as usize);
        // β = 0 outside CG, so the direction's `cg` changes nothing here.
        let dirs = off.dirs(false);
        let b = &off.bufs;
        let reads = [
            b.ud.storage(),
            off.order.ids.order().storage(),
            steps.storage(),
        ];
        let writes = [b.prev.storage(), offsets.storage()];
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: `span` lies in the epoch's positions, whose buffers `BhPositions::zeroed`
            // sized; `EpochOrder` holds frame ids below `nframe`.
            unsafe {
                self.module
                    .bh_pos_update(stream, config, span, dirs, dptr(offsets), step, off.step)
            }
        })
    }

    /// Under full-object sampling, the stack's object gradient patches into `patch` and its
    /// offsets' gradient partials into segment `seg` of `off`'s, from `f` and `win`, the
    /// estimate's windows of a slice and their derivatives over `y` and `x` in blocks 0 to 2:
    /// per frame and object mode, `G = Σ_p conj(q P_p) r` on the window, which
    /// [`crate::gpu::sampling::ObjectSampler::place`] adds to the slice as
    /// [`Kernels::psi_to_obj`] would, and per chunk of [`GAMMA_CHUNK`] pixels of each frame
    /// the pair [`Self::pos_dots`] sums with the other segments' to `∂F/∂d`; other frames'
    /// patches and partials, and other segments' partials, are left as they are.
    pub fn win_grad(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        f: BhWinRef<'_>,
        (win, seg): (&DeviceBuffer<C32>, usize),
        patch: &mut DeviceBuffer<C32>,
        off: &mut BhOffsets<'_>,
    ) -> Result<(), Error> {
        let dims = off.check()?;
        let g = dims.geom(stack)?;
        let blk = window_block(dims, &g);
        f.check(dims, &g)?;
        let part = off.bufs.partials(seg)?;
        check_len("win", win, 3 * blk)?;
        check_len("patch", patch, blk)?;
        let config = chunk_grid(&g, dims.n())?;
        let reads = [
            f.r.storage(),
            f.probe.storage(),
            win.storage(),
            stack.storage(),
        ];
        let grads = &off.bufs.grads;
        let writes = [patch.storage(), grads.storage()];
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above; `BhPositions::zeroed` sized `grads` for every
            // epoch position's chunks of every segment, of which `part` is one; one block of
            // PIXEL_THREADS per chunk and frame.
            unsafe {
                self.module.bh_win_grad(
                    stream,
                    config,
                    f.ptrs(),
                    dptr(win),
                    dptr(patch),
                    part,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The stack's offsets' gradient partials of `last`'s plane phase into segment `seg` of
    /// `off`'s, from `r` the residual after its inverse FFT and the phase's adjoint: per
    /// chunk of [`GAMMA_CHUNK`] pixels of each frame the pair [`Self::pos_dots`] sums with
    /// the other segments' to `∂F/∂d`; other frames' partials, and other segments', are left
    /// as they are. `off`'s views are those the slice is read through.
    pub fn phase_grad(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        last: BhLastSlice<'_>,
        r: &DeviceBuffer<C32>,
        (off, seg): (&mut BhOffsets<'_>, usize),
    ) -> Result<(), Error> {
        let dims = off.check()?;
        let g = dims.geom(stack)?;
        let n = dims.n();
        let (s, entered) = last.entry.slice_in(last.slice, dims, &g)?;
        check_len("r", r, g.nf_max as usize * dims.nmode as usize * n)?;
        let part = off.bufs.partials(seg)?;
        let config = chunk_grid(&g, n)?;
        let views = off.order.views.addr();
        let mut reads = vec![r.storage(), views.storage(), stack.storage()];
        reads.extend(entered);
        let grads = &off.bufs.grads;
        self.base.launch(ctx, &reads, &[grads.storage()], |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object with valid mode indices; `BhPositions::zeroed` sized `grads` for every
            // epoch position's chunks of every segment, of which `part` is one; one block of
            // PIXEL_THREADS per chunk and frame.
            unsafe {
                self.module.bh_phase_grad(
                    stream,
                    config,
                    s,
                    dptr(r),
                    dptr::<ViewAddr, ViewAddr>(views),
                    last.quad,
                    part,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The stack's derivatives `t` along `d` through `last`'s plane phase, in place with the
    /// phase itself left out: on entry those of the slice's exit ([`Self::slice_tangent`]),
    /// after it those of the exit waves once [`crate::gpu::OpticsKernels::slice_phase`] puts
    /// the phase on `t.lu` and its `lv` and its adjoint is on the residual, as
    /// [`Self::sliced_hess_frames`] reads them. Of `d` only whether it has `v` is read, and
    /// `t.cg` is given exactly with it. `off`'s views are those the slice is read through.
    pub fn phase_tangent(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        last: BhLastSlice<'_>,
        off: &BhOffsets<'_>,
        d: &BhDisp<'_>,
        t: &mut BhTangentsMut<'_>,
    ) -> Result<(), Error> {
        let dims = off.check()?;
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        let (s, entered) = last.entry.slice_in(last.slice, dims, &g)?;
        let (writes, t) = t.touched(d.v.is_some(), len)?;
        let config = grid_1d(len)?;
        let b = &off.bufs;
        let views = off.order.views.addr();
        let mut reads = vec![
            views.storage(),
            stack.storage(),
            b.ud.storage(),
            b.prev.storage(),
            off.order.ids.order().storage(),
        ];
        reads.extend(entered);
        let dirs = off.dirs(d.v.is_some());
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object with valid mode indices; `EpochOrder` holds frame ids below `nframe`.
            unsafe {
                self.module.bh_phase_tangent(
                    stream,
                    config,
                    s,
                    dirs,
                    t,
                    dptr::<ViewAddr, ViewAddr>(views),
                    last.quad,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// [`Self::win_grad`]'s object gradient patches alone, for the offsets held: per frame and
    /// object mode, `G = Σ_p conj(q P_p) r` on the window into `patch`; other frames' patches
    /// are left as they are. `dims` are the scan's.
    pub fn win_obj_grad(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        f: BhWinRef<'_>,
        patch: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        let blk = window_block(dims, &g);
        f.check(dims, &g)?;
        check_len("patch", patch, blk)?;
        let config = grid_1d(blk)?;
        let reads = [f.r.storage(), f.probe.storage(), stack.storage()];
        self.base.launch(ctx, &reads, &[patch.storage()], |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.bh_win_obj_grad(
                    stream,
                    config,
                    f.ptrs(),
                    dptr(patch),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Under full-object sampling, lifts the windows of the step's object parts to the
    /// windows' first-order changes: `hu += Σ_a u_da ∂_a S(O)` and with `hv` `hv += Σ_a v_da
    /// ∂_a S(O)` on block 0 of each, `win` the estimate's windows and their derivatives over
    /// `y` and `x`, `u_d` and `v_d` [`Self::pos_dots`]' and the last direction's offsets of
    /// `off`. Frames outside the stack are left as they are.
    pub fn win_lift(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        win: &DeviceBuffer<C32>,
        hu: &mut DeviceBuffer<C32>,
        hv: Option<&mut DeviceBuffer<C32>>,
        off: &BhOffsets<'_>,
    ) -> Result<(), Error> {
        let dims = off.check()?;
        let g = dims.geom(stack)?;
        let blk = window_block(dims, &g);
        check_len("win", win, 3 * blk)?;
        check_len("hu", hu, blk)?;
        if let Some(hv) = &hv {
            check_len("hv", hv, blk)?;
        }
        let hv = hv.map(|b| &*b);
        let dirs = off.dirs(hv.is_some());
        let config = grid_1d(blk)?;
        let b = &off.bufs;
        let reads = [
            win.storage(),
            b.ud.storage(),
            b.prev.storage(),
            off.order.ids.order().storage(),
            stack.storage(),
        ];
        let mut writes = vec![hu.storage()];
        writes.extend(hv.map(|b| b.storage()));
        // Without `hv`, `hu`'s pointer stands in for it; the kernel never writes it.
        let hv = dptr(hv.unwrap_or(hu));
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above; `EpochOrder` holds frame ids below `nframe`.
            unsafe {
                self.module.bh_win_lift(
                    stream,
                    config,
                    dptr(win),
                    dptr(hu),
                    hv,
                    dirs,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Under full-object sampling, adds the windows' second-order terms to the Hessian triples
    /// of [`Self::hess_frames`] along the step whose object parts' windows `w` holds: per
    /// chunk of [`GAMMA_CHUNK`] pixels of each frame, `Σ Re(conj(r) q P o″)` with `o″(u, v) =
    /// Σ_a (v_da ∂_a S(u_O) + u_da ∂_a S(v_O)) + Σ_ac u_da v_dc ∂_a ∂_c S(O)`, the last two of
    /// the triple without `w.v` left as they are; `u_d` and `v_d` as [`Self::win_lift`] reads
    /// them.
    pub fn win_hess(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        f: BhWinRef<'_>,
        w: &BhWinSecond<'_>,
        off: &BhOffsets<'_>,
        parts: &mut BhParts,
    ) -> Result<(), Error> {
        let dims = off.check()?;
        let g = dims.geom(stack)?;
        let blk = window_block(dims, &g);
        f.check(dims, &g)?;
        check_len("second", w.second, 3 * blk)?;
        check_len("u windows", w.u, 3 * blk)?;
        if let Some(v) = w.v {
            check_len("v windows", v, 3 * blk)?;
        }
        parts.check(dims)?;
        let config = chunk_grid(&g, dims.n())?;
        let b = &off.bufs;
        let mut reads = vec![
            f.r.storage(),
            f.probe.storage(),
            w.second.storage(),
            w.u.storage(),
            b.ud.storage(),
            b.prev.storage(),
            off.order.ids.order().storage(),
            stack.storage(),
        ];
        reads.extend(w.v.map(|b| b.storage()));
        let dirs = off.dirs(w.v.is_some());
        let derivs = BhWinDerivs {
            second: dptr(w.second),
            du: dptr(w.u),
            // Without `v`, `u`'s stand in; the kernel never reads them.
            dv: dptr(w.v.unwrap_or(w.u)),
        };
        let out = parts.ptr(parts.hess(0));
        self.base
            .launch(ctx, &reads, &[parts.buf.storage()], |stream| {
                // SAFETY: lengths checked above; the triples lie in `parts`; one block of
                // PIXEL_THREADS per chunk and frame; `EpochOrder` holds frame ids below `nframe`.
                unsafe {
                    self.module.bh_win_hess(
                        stream,
                        config,
                        f.ptrs(),
                        derivs,
                        dirs,
                        out,
                        stack.ctl() as *const StackCtl,
                        g,
                    )
                }
            })
    }
}

impl BhWinRef<'_> {
    fn check(&self, dims: &ViewDims, g: &StackGeom) -> Result<(), Error> {
        let n = dims.n();
        check_len("r", self.r, g.nf_max as usize * dims.nmode as usize * n)?;
        check_len("probe", self.probe, dims.nprobe as usize * n)
    }

    fn ptrs(&self) -> BhWinFields {
        BhWinFields {
            r: dptr(self.r),
            probe: dptr(self.probe),
        }
    }
}

/// Floats of step records `0..=record`.
fn step_len(record: u32) -> usize {
    BH_STEP as usize * (record as usize + 1)
}

#[cfg(test)]
mod tests;
