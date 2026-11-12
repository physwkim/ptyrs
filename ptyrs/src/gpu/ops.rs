//! M1 pixel operators: checked launches of the non-fused exit-wave kernels.
//!
//! # Contract (shared with the engine's CPU oracle)
//!
//! * Shapes are row-major. A frame, the probe and the detector are `[ny, nx]` with `ny`, `nx`
//!   even; `n = ny * nx`. `probe` is `[nprobe, ny, nx]`, `obj` `[nobj, nyo, nxo]`, both in
//!   natural order (no shift). Complex values are [`C32`].
//! * Views are frame-major, `k = frame * nmode + mode`; [`ViewAddr`] says which probe and
//!   object mode a view multiplies and where it samples the object. They are canonical
//!   ([`Views::new`] checks it): `nmode = nprobe * nobj`, mode `m` multiplies probe mode
//!   `m % nprobe` and object mode `m / nprobe`, and all views of a frame sample the object
//!   at one position. A stack is frames
//!   `f0 .. f0 + nf`, a [`StackCtl`] entry of a [`StackTable`] that the kernels read when they
//!   run, not when they are launched or captured. Per-stack buffers are sized by the table's
//!   `nf_max`: `psi` holds `nf_max * nmode` planes, plane `kl < nf * nmode` being view
//!   `f0 * nmode + kl`; planes and records past the stack are left as they are, and a stack
//!   with `nf == 0` writes nothing.
//! * Real-space exit waves have their origin at `[0, 0]`: probe pixel `(y, x)` is stored at
//!   `((y + ny/2) % ny, (x + nx/2) % nx)`. So a plain 2D FFT of each plane
//!   (`crate::fft::Plan2d`, unnormalized, forward `exp(-2 pi i jk / n)`) gives the far field in
//!   FFT order, and frequency `(u, v)` of that FFT is detector pixel
//!   `((u + ny/2) % ny, (v + nx/2) % nx)` of the counts (zero frequency at `[ny/2, nx/2]`).
//!   The kernels apply that map themselves; nobody shifts data.
//! * Scaling: PyNX uses an orthonormal FFT. Here both FFTs are unnormalized, the calculated
//!   intensity is `sum_m |Psi_m|^2 / n`, and [`Kernels::apply_amplitude`] folds the round
//!   trip's `1 / n` into its update, so FFT, `apply_amplitude`, inverse FFT gives PyNX's
//!   projected exit wave exactly (in real arithmetic).
//! * Far field: [`ViewDims::quad_phase`] `= pi * px^2 / (lambda * z)` (`px` the object pixel
//!   size); the forward operator multiplies by `exp(+i quad r^2)` with `r^2 = (y - ny/2)^2 +
//!   (x - nx/2)^2`, the update operators by its conjugate. Zero for near field.
//! * Counts are `u16` `[nframe, ny, nx]` in detector order for the whole scan; [`MASKED`]
//!   marks a pixel to ignore. M1 has no background.
//!
//! # PyNX AP / DM in these calls (per stack, then per cycle)
//!
//! AP: `obj_probe_to_psi`, FFT, `apply_amplitude`, inverse FFT, `psi_to_obj` and/or
//! `psi_to_probe` into the sums; after the last stack, `update_obj_from_sums` with inertia
//! 0.01 (object) or `update_from_sums` with 0.001 (probe), which consume the sums.
//!
//! DM (`alpha` 0.02, `psi` kept per stack across cycles): `obj_probe_to_psi` into a scratch
//! `po`, then on the first cycle copy `po` into `psi` (PyNX starts from `psi = PO`, which
//! [`Kernels::dm_reflect`] maps to itself), `dm_reflect(psi, po)`, FFT, `apply_amplitude`,
//! inverse FFT, `add_c32(psi, po)`; then the object/probe sums and updates as in AP, from the
//! new `psi`.
//!
//! LLK: forward FFT of `obj_probe_to_psi`, then [`Kernels::llk`] per stack; PyNX's reported
//! (normalized) values are [`LlkSums::from_partials`] over the scan's partials indexed by
//! frame, a tree fixed by frame index whatever the stacks or GPUs, divided by the number of
//! unmasked count pixels of the scan.

use std::ops::Range;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::sys::CUdeviceptr;
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig, RawDeviceBuffer};
use ptyrs_kernels::pixel::{
    LLK_CHUNK, PIXEL_THREADS, STACK_TILE_CANDIDATES, SumPtrs, TilePtrs, TileSlabs, WARP,
    axis_tiles, axis_tiles_max,
};
pub use ptyrs_kernels::pixel::{
    LlkPartial, MASKED, OBJ_TILE, PositionPartial, ScalePartial, StackCtl, StackGeom, ViewAddr,
};

use crate::gpu::{C32, Error, Kernels, check_len, dptr, lease};

/// Geometry of a scan's views.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewDims {
    pub nframe: u32,
    /// Views per frame.
    pub nmode: u32,
    pub nprobe: u32,
    pub nobj: u32,
    /// `[ny, nx]`, both even.
    pub frame: [u32; 2],
    /// `[nyo, nxo]`, at least the frame.
    pub obj: [u32; 2],
    /// `pi * px^2 / (lambda * z)`, radians per squared object pixel; 0 for near field.
    pub quad_phase: f32,
}

impl ViewDims {
    pub(crate) fn n(&self) -> usize {
        self.frame[0] as usize * self.frame[1] as usize
    }

    pub(crate) fn obj_len(&self) -> usize {
        self.nobj as usize * self.obj[0] as usize * self.obj[1] as usize
    }

    /// `[nty, ntx]` object tiles of `OBJ_TILE` pixels per side; tile `[ty, tx]` is entry
    /// `ty * ntx + tx` of the [`Views`] tile lists.
    pub fn tiles(&self) -> [u32; 2] {
        self.obj.map(|n| n.div_ceil(OBJ_TILE))
    }

    /// The per-view check of [`Views::new`]: `fy` and `fx` are in `[0, 1)`, every object
    /// pixel the view's samples read lies inside the object (the `+ 1` neighbour of a
    /// non-zero `fy` or `fx` included; nothing wraps) and its probe and object modes exist.
    pub fn check_view(&self, v: &ViewAddr) -> Result<(), Error> {
        let frac = |f: f32| (0.0..1.0).contains(&f);
        let fits =
            |o: u32, f: f32, n: u32, no: u32| o as u64 + n as u64 + (f > 0.0) as u64 <= no as u64;
        let inside = fits(v.oy, v.fy, self.frame[0], self.obj[0])
            && fits(v.ox, v.fx, self.frame[1], self.obj[1]);
        if !frac(v.fy) || !frac(v.fx) || !inside || v.probe >= self.nprobe || v.obj >= self.nobj {
            return Err(Error::Shape(format!("{v:?} is outside {self:?}")));
        }
        Ok(())
    }

    fn ntiles(&self) -> usize {
        let [nty, ntx] = self.tiles();
        nty as usize * ntx as usize
    }

    /// One `(OBJ_TILE, OBJ_TILE)` block per object tile and object mode.
    pub(crate) fn tile_grid(&self) -> LaunchConfig {
        let [nty, ntx] = self.tiles();
        LaunchConfig {
            grid_dim: (ntx, nty, self.nobj),
            block_dim: (OBJ_TILE, OBJ_TILE, 1),
            shared_mem_bytes: 0,
        }
    }

    /// The launch of the per-stack object kernels for stacks of at most `nf_max` frames:
    /// [`STACK_TILE_CANDIDATES`] tiles of the frames' [`axis_tiles`] boxes per
    /// `(OBJ_TILE, OBJ_TILE)` block and object mode, so the grid scales with the stack, not
    /// with the object.
    pub(crate) fn stack_tile_grid(&self, nf_max: u32) -> Result<LaunchConfig, Error> {
        let [ny, nx] = self.frame;
        let ncand = u64::from(nf_max) * u64::from(axis_tiles_max(ny) * axis_tiles_max(nx));
        let gx = ncand.div_ceil(STACK_TILE_CANDIDATES.into());
        if ncand > u64::from(u32::MAX) || gx > i32::MAX as u64 {
            return Err(Error::Shape(format!(
                "{self:?}: {nf_max} frames per stack exceed the object-gather grid"
            )));
        }
        Ok(LaunchConfig {
            grid_dim: (gx as u32, 1, self.nobj),
            block_dim: (OBJ_TILE, OBJ_TILE, 1),
            shared_mem_bytes: 0,
        })
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        let [ny, nx] = self.frame;
        let [nyo, nxo] = self.obj;
        let bad = |what: &str| Err(Error::Shape(format!("{self:?}: {what}")));
        if self.nframe == 0 || self.nmode == 0 || self.nprobe == 0 || self.nobj == 0 {
            return bad("a count is zero");
        }
        if self.nmode as u64 != self.nprobe as u64 * self.nobj as u64 {
            return bad("nmode is not nprobe * nobj");
        }
        // `r^2` of the quadratic phase is an i32 on the device.
        if ny == 0 || nx == 0 || ny % 2 != 0 || nx % 2 != 0 || ny > 32768 || nx > 32768 {
            return bad("frame sides must be even, 2..=32768");
        }
        if nyo < ny || nxo < nx {
            return bad("object smaller than a frame");
        }
        if nyo.div_ceil(OBJ_TILE) > 65535 || self.nobj > 65535 {
            return bad("object exceeds the object-gather grid");
        }
        if u32::try_from(self.nframe as u64 * self.nmode as u64).is_err() {
            return bad("more than u32::MAX views");
        }
        if !self.quad_phase.is_finite() {
            return bad("quad_phase is not finite");
        }
        Ok(())
    }

    /// The stack kernels' geometry for `stack`, whose table must have been made for a scan
    /// of `nframe` frames: the [`StackGeom`] a stack kernel takes by value.
    pub fn geom(&self, stack: StackRef<'_>) -> Result<StackGeom, Error> {
        self.check()?;
        let t = stack.table;
        if t.nframe != self.nframe {
            return Err(Error::Shape(format!(
                "stack table made for {} frames, dims have {}",
                t.nframe, self.nframe
            )));
        }
        Ok(StackGeom {
            nf_max: t.nf_max,
            ..self.scan_geom()
        })
    }

    /// The geometry of the whole scan as one stack, for checked dims.
    pub(crate) fn scan_geom(&self) -> StackGeom {
        StackGeom {
            nframe: self.nframe,
            nf_max: self.nframe,
            nmode: self.nmode,
            nprobe: self.nprobe,
            ny: self.frame[0],
            nx: self.frame[1],
            nyo: self.obj[0],
            nxo: self.obj[1],
            quad: self.quad_phase,
        }
    }
}

/// [`StackCtl`] entries on the device, for a scan of `nframe` frames, each stack at most
/// `nf_max` frames. A kernel reads its entry when it runs and clamps it to `nf <= nf_max`,
/// `f0 + nf <= nframe`.
pub struct StackTable {
    ctl: DeviceBuffer<StackCtl>,
    nframe: u32,
    nf_max: u32,
}

impl StackTable {
    /// One entry per range, each a non-empty part of `0 .. dims.nframe`; `nf_max` is the
    /// longest.
    pub fn new(stream: &CudaStream, dims: &ViewDims, stacks: &[Range<u32>]) -> Result<Self, Error> {
        dims.check()?;
        let mut ctl = Vec::with_capacity(stacks.len());
        for r in stacks {
            if r.start >= r.end || r.end > dims.nframe {
                return Err(Error::Shape(format!(
                    "stack {r:?} is not a non-empty part of 0..{}",
                    dims.nframe
                )));
            }
            ctl.push(StackCtl {
                f0: r.start,
                nf: r.end - r.start,
            });
        }
        let nf_max = ctl
            .iter()
            .map(|c| c.nf)
            .max()
            .ok_or_else(|| Error::Shape("no stacks".into()))?;
        Ok(Self {
            ctl: DeviceBuffer::from_host(stream, &ctl)?,
            nframe: dims.nframe,
            nf_max,
        })
    }

    /// One entry, `{ f0: 0, nf: 0 }`, for the caller's own device code to rewrite between
    /// runs (through [`Self::ptr`], leased through [`Self::storage`]); `nf_max` in
    /// `1 ..= dims.nframe`.
    pub fn device(stream: &CudaStream, dims: &ViewDims, nf_max: u32) -> Result<Self, Error> {
        dims.check()?;
        if nf_max == 0 || nf_max > dims.nframe {
            return Err(Error::Shape(format!(
                "nf_max {nf_max} is not in 1..={}",
                dims.nframe
            )));
        }
        Ok(Self {
            ctl: DeviceBuffer::zeroed(stream, 1)?,
            nframe: dims.nframe,
            nf_max,
        })
    }

    pub fn len(&self) -> usize {
        self.ctl.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ctl.is_empty()
    }

    pub fn nf_max(&self) -> u32 {
        self.nf_max
    }

    /// Entry `s`; panics if `s >= self.len()`.
    pub fn at(&self, s: usize) -> StackRef<'_> {
        assert!(s < self.len(), "stack {s} of a {}-entry table", self.len());
        StackRef { table: self, s }
    }

    /// Device address of entry 0; entry `s` is `s * size_of::<StackCtl>()` bytes on.
    pub fn ptr(&self) -> CUdeviceptr {
        self.ctl.cu_deviceptr()
    }

    /// The entries' storage, for leasing a write to them.
    pub fn storage(&self) -> &RawDeviceBuffer {
        self.ctl.storage()
    }
}

/// Entry `s` of a [`StackTable`], what a stack launch takes.
#[derive(Clone, Copy)]
pub struct StackRef<'a> {
    table: &'a StackTable,
    s: usize,
}

impl StackRef<'_> {
    pub fn nf_max(&self) -> u32 {
        self.table.nf_max
    }

    /// Device address of this entry's [`StackCtl`], a stack kernel's `ctl` argument.
    pub fn ctl(&self) -> CUdeviceptr {
        self.table.ptr() + (self.s * size_of::<StackCtl>()) as CUdeviceptr
    }

    /// The whole table's storage, for leasing a read of the entry around a launch.
    pub fn storage(&self) -> &RawDeviceBuffer {
        self.table.ctl.storage()
    }
}

/// The views of a scan on the device, with the per-object-tile view lists the object
/// gather walks: tile `t`'s run is `tile_views[tile_start[t] .. tile_start[t] + tile_len[t]]`.
///
/// The owned tiles are those whose slab `tile_start[t] .. tile_start[t + 1]` has a slot;
/// binning lists views in them only. A tile outside the owned set has an empty run: the
/// launches may sample its object pixels, which the forward and probe passes read through
/// each view, but never write them or its sums.
pub struct Views {
    dims: ViewDims,
    addr: DeviceBuffer<ViewAddr>,
    /// `ntiles + 1` ascending offsets into `tile_views`; tile `ty * ntx + tx` of `OBJ_TILE`
    /// pixels.
    tile_start: DeviceBuffer<u32>,
    /// Per owned tile, ascending indices of the views whose samples may touch it.
    tile_views: DeviceBuffer<u32>,
    /// `ntiles` run lengths, at most `tile_start[t + 1] - tile_start[t]`. No launch writes
    /// the object pixels of a tile whose run is empty, nor reads or writes its sums; the
    /// passes that sample the object per view read its object pixels.
    tile_len: DeviceBuffer<u32>,
}

impl Views {
    /// Checks `addr` (`nframe * nmode` canonical views, each inside the object with `fy`,
    /// `fx` in `[0, 1)`) and uploads it with its tile lists; it owns every tile a view
    /// touches.
    pub fn new(stream: &CudaStream, dims: ViewDims, addr: &[ViewAddr]) -> Result<Self, Error> {
        dims.check()?;
        Self::build(stream, dims, addr, &vec![true; dims.ntiles()])
    }

    /// [`Self::new`] owning only the tiles of `own` (`ty * ntx + tx`, any order, repeats
    /// allowed), a region's tiles: each gets the run `new` gives it, every other tile an
    /// empty slab and run. The update launches leave the other tiles' object pixels and sums
    /// as they are, for their owner to write; the forward and probe passes still sample
    /// them. A tile index past the object's tiles is an error.
    pub fn owned(
        stream: &CudaStream,
        dims: ViewDims,
        addr: &[ViewAddr],
        own: &[u32],
    ) -> Result<Self, Error> {
        dims.check()?;
        let mut owned = vec![false; dims.ntiles()];
        for &t in own {
            let ntiles = owned.len();
            *owned
                .get_mut(t as usize)
                .ok_or_else(|| Error::Shape(format!("owned tile {t} of {ntiles}")))? = true;
        }
        Self::build(stream, dims, addr, &owned)
    }

    /// [`Self::owned`] of the tiles `owned` marks, `dims` checked.
    fn build(
        stream: &CudaStream,
        dims: ViewDims,
        addr: &[ViewAddr],
        owned: &[bool],
    ) -> Result<Self, Error> {
        let nview = dims.nframe as usize * dims.nmode as usize;
        if addr.len() != nview {
            return Err(Error::Shape(format!(
                "{} views, dims need {nview}",
                addr.len()
            )));
        }
        for (k, v) in addr.iter().enumerate() {
            dims.check_view(v)
                .map_err(|e| Error::Shape(format!("view {k}: {e}")))?;
        }
        for (f, frame) in addr.chunks_exact(dims.nmode as usize).enumerate() {
            let at = |v: &ViewAddr| (v.oy, v.ox, v.fy, v.fx);
            for (m, v) in (0..).zip(frame) {
                if v.probe != m % dims.nprobe || v.obj != m / dims.nprobe || at(v) != at(&frame[0])
                {
                    return Err(Error::Shape(format!(
                        "frame {f} mode {m}: {v:?} is not probe {} x object {} at the frame's \
                         position",
                        m % dims.nprobe,
                        m / dims.nprobe
                    )));
                }
            }
        }
        let t = tile_lists(&dims, addr, owned)?;
        Ok(Self {
            dims,
            addr: DeviceBuffer::from_host(stream, addr)?,
            tile_start: DeviceBuffer::from_host(stream, &t.start)?,
            tile_views: DeviceBuffer::from_host(stream, &t.views)?,
            tile_len: DeviceBuffer::from_host(stream, &t.len)?,
        })
    }

    /// Views over device storage the caller keeps filling and backing: `addr` takes the
    /// records of frames as they land, and `tile_views` holds `ntiles` slabs of `cap` slots,
    /// tile `t`'s from slot `t * cap`, so it owns every tile, or none when `cap` is 0. Every
    /// run starts empty; only binning ([`Kernels::bin_views`], [`Kernels::reorder_views`])
    /// writes runs. Checks `dims`, that `addr` holds `nframe * nmode` records and
    /// `tile_views` `ntiles * cap` slots.
    ///
    /// # Safety
    ///
    /// Whenever a launch runs, every record it reads passes the checks of [`Self::new`], and
    /// the slab of every tile those views touch is device memory; a launch touches no other
    /// slab.
    pub unsafe fn from_parts(
        stream: &CudaStream,
        dims: ViewDims,
        addr: DeviceBuffer<ViewAddr>,
        tile_views: DeviceBuffer<u32>,
        cap: u32,
    ) -> Result<Self, Error> {
        dims.check()?;
        let ntiles = dims.ntiles();
        check_len("addr", &addr, dims.nframe as usize * dims.nmode as usize)?;
        let start = (0..=ntiles as u64)
            .map(|t| u32::try_from(t * u64::from(cap)))
            .collect::<Result<Vec<u32>, _>>()
            .map_err(|_| Error::Shape(format!("{ntiles} slabs of {cap} slots overflow u32")))?;
        check_len("tile_views", &tile_views, start[ntiles] as usize)?;
        Ok(Self {
            dims,
            addr,
            tile_start: DeviceBuffer::from_host(stream, &start)?,
            tile_views,
            tile_len: DeviceBuffer::zeroed(stream, ntiles)?,
        })
    }

    /// Views for [`Kernels::reorder_views`] of `src`: its dims, frames and tile slabs, so
    /// each tile's capacity and the owned set, with every run empty until a reorder fills
    /// them.
    pub fn reordered(stream: &CudaStream, src: &Views) -> Result<Self, Error> {
        fn copy<T: DeviceCopy>(
            stream: &CudaStream,
            from: &DeviceBuffer<T>,
        ) -> Result<DeviceBuffer<T>, Error> {
            let mut to = DeviceBuffer::zeroed(stream, from.len())?;
            to.copy_from_device_async(from, stream)?;
            Ok(to)
        }
        Ok(Self {
            dims: src.dims,
            addr: copy(stream, &src.addr)?,
            tile_start: copy(stream, &src.tile_start)?,
            tile_views: DeviceBuffer::zeroed(stream, src.tile_views.len())?,
            tile_len: DeviceBuffer::zeroed(stream, src.tile_len.len())?,
        })
    }

    pub fn dims(&self) -> &ViewDims {
        &self.dims
    }

    /// The `nframe * nmode` view records, view `k` at index `k`.
    pub fn addr(&self) -> &DeviceBuffer<ViewAddr> {
        &self.addr
    }

    /// `ntiles + 1` ascending offsets into [`Self::tile_views`]; object tile `ty * ntx + tx`
    /// covers `OBJ_TILE` pixels per side, `ntx = nxo.div_ceil(OBJ_TILE)`.
    pub fn tile_start(&self) -> &DeviceBuffer<u32> {
        &self.tile_start
    }

    /// Per object tile, the ascending indices of the views whose samples may touch it.
    pub fn tile_views(&self) -> &DeviceBuffer<u32> {
        &self.tile_views
    }

    /// Per object tile, the length of its run in [`Self::tile_views`].
    pub fn tile_len(&self) -> &DeviceBuffer<u32> {
        &self.tile_len
    }

    pub(crate) fn tile_ptrs(&self) -> TilePtrs {
        TilePtrs {
            start: dptr(&self.tile_start),
            len: dptr(&self.tile_len),
            views: dptr(&self.tile_views),
        }
    }

    fn tile_slabs(&self) -> TileSlabs {
        TileSlabs {
            start: dptr(&self.tile_start),
            len: dptr(&self.tile_len),
            views: dptr(&self.tile_views),
        }
    }

    pub(crate) fn tile_storages(&self) -> [&RawDeviceBuffer; 3] {
        [
            self.tile_start.storage(),
            self.tile_len.storage(),
            self.tile_views.storage(),
        ]
    }
}

/// A permutation of the frames `0 .. n` on the device, the order
/// [`Kernels::reorder_views`] takes.
pub struct Permutation {
    order: DeviceBuffer<u32>,
}

impl Permutation {
    /// Checks that `order` holds each of `0 .. order.len()` once and uploads it.
    pub fn new(stream: &CudaStream, order: &[u32]) -> Result<Self, Error> {
        Self::check(order)?;
        Ok(Self {
            order: DeviceBuffer::from_host(stream, order)?,
        })
    }

    fn check(order: &[u32]) -> Result<(), Error> {
        let mut seen = vec![false; order.len()];
        for (p, &q) in order.iter().enumerate() {
            match seen.get_mut(q as usize) {
                Some(s) if !*s => *s = true,
                _ => {
                    return Err(Error::Shape(format!(
                        "order[{p}] = {q} repeats a frame or is past 0..{}",
                        order.len()
                    )));
                }
            }
        }
        Ok(())
    }

    /// The frame at each position.
    pub(crate) fn order(&self) -> &DeviceBuffer<u32> {
        &self.order
    }

    /// Checks `order` as [`Self::new`] does, of this permutation's length, and enqueues on
    /// `ctx` its copy over this one.
    ///
    /// # Safety
    ///
    /// `order` is pinned host memory, not written until the copy completes.
    pub(crate) unsafe fn set_async(
        &mut self,
        ctx: &ExecutionContext,
        order: &[u32],
    ) -> Result<(), Error> {
        if order.len() != self.order.len() {
            return Err(Error::Shape(format!(
                "order of {} frames over a permutation of {}",
                order.len(),
                self.order.len()
            )));
        }
        Self::check(order)?;
        lease(ctx, &[], &[self.order.storage()])?;
        ctx.context().bind_to_thread()?;
        // SAFETY: `self.order` holds `order.len()` ids, leased for writing on this stream;
        // the caller keeps `order` as it is until the copy completes.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                self.order.cu_deviceptr(),
                order.as_ptr(),
                size_of_val(order),
                ctx.get_cuda_stream().cu_stream(),
            )?
        };
        Ok(())
    }
}

/// The object tiles `[ty, tx]` holding the object pixels that the samples of view `v` read,
/// ascending (ty-major); for a view [`ViewDims::check_view`] accepts, all inside the object.
/// [`Views::new`] lists view `v` in exactly these tiles' runs; [`Views::owned`] and
/// [`Kernels::bin_views`] in those of the owned ones.
pub fn view_tiles(dims: &ViewDims, v: &ViewAddr) -> impl Iterator<Item = [u32; 2]> + use<> {
    let [ny, nx] = dims.frame;
    let (x0, x1) = axis_tiles(v.ox, nx, v.fx);
    let (y0, y1) = axis_tiles(v.oy, ny, v.fy);
    (y0..=y1).flat_map(move |ty| (x0..=x1).map(move |tx| [ty, tx]))
}

/// Host form of the [`Views`] tile lists.
struct TileLists {
    start: Vec<u32>,
    len: Vec<u32>,
    views: Vec<u32>,
}

/// CSR lists of the views touching each object tile `owned` marks, ascending per tile; every
/// other tile gets an empty slab.
fn tile_lists(dims: &ViewDims, addr: &[ViewAddr], owned: &[bool]) -> Result<TileLists, Error> {
    let ntx = dims.tiles()[1] as usize;
    let mut lists = vec![Vec::new(); dims.ntiles()];
    for (k, v) in addr.iter().enumerate() {
        for [ty, tx] in view_tiles(dims, v) {
            let t = ty as usize * ntx + tx as usize;
            if owned[t] {
                lists[t].push(k as u32);
            }
        }
    }
    let mut start = Vec::with_capacity(lists.len() + 1);
    let len = lists.iter().map(|l| l.len() as u32).collect();
    let mut flat = Vec::new();
    start.push(0);
    for l in lists {
        flat.extend(l);
        start.push(
            u32::try_from(flat.len())
                .map_err(|_| Error::Shape("tile lists exceed u32::MAX entries".into()))?,
        );
    }
    Ok(TileLists {
        start,
        len,
        views: flat,
    })
}

/// Leaves of one level of the [`LlkSums`] tree: each level sums blocks of this many nodes.
pub const LLK_TREE_LEAVES: usize = 256;

/// `f64` totals of [`LlkPartial`]s over a tree fixed by frame index.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LlkSums {
    pub poisson: f64,
    pub gaussian: f64,
    pub euclidean: f64,
    /// Total calculated intensity (PyNX `nb_photons_calc`).
    pub icalc: f64,
}

impl LlkSums {
    /// Pairwise sums, adjacent nodes first, over the partials padded with `+0.0` to
    /// `LLK_TREE_LEAVES^L` (smallest `L >= 1`). The tree depends only on frame indices, not
    /// on how frames are split across launches or GPUs, so each level is one program per
    /// block of [`LLK_TREE_LEAVES`] nodes.
    pub fn from_partials(partials: &[LlkPartial]) -> Self {
        let mut level: Vec<[f64; 4]> = partials
            .iter()
            .map(|p| [p.poisson, p.gaussian, p.euclidean, p.icalc].map(f64::from))
            .collect();
        loop {
            level = if level.is_empty() {
                vec![[0.0; 4]]
            } else {
                level.chunks(LLK_TREE_LEAVES).map(pair_tree).collect()
            };
            if level.len() == 1 {
                let [poisson, gaussian, euclidean, icalc] = level[0];
                return Self {
                    poisson,
                    gaussian,
                    euclidean,
                    icalc,
                };
            }
        }
    }
}

/// The per-chunk sums [`Kernels::llk`] reduces each frame through, for stacks of up to
/// `nf_max` frames of one scan's [`ViewDims`]; one serves every stack in turn.
pub struct LlkScratch {
    dims: ViewDims,
    /// `[nf_max, llk_chunks(n)]`.
    chunks: DeviceBuffer<LlkPartial>,
}

impl LlkScratch {
    pub fn new(stream: &CudaStream, dims: &ViewDims, nf_max: u32) -> Result<Self, Error> {
        dims.check()?;
        if nf_max == 0 {
            return Err(Error::Shape("nf_max is zero".into()));
        }
        let len = nf_max as usize * llk_chunks(dims.n()) as usize;
        Ok(Self {
            dims: *dims,
            chunks: DeviceBuffer::zeroed(stream, len)?,
        })
    }
}

/// Chunks of [`LLK_CHUNK`] pixels per frame of `n` pixels.
fn llk_chunks(n: usize) -> u32 {
    // `n <= 32768^2`, so at most 2^19 chunks.
    n.div_ceil(LLK_CHUNK as usize) as u32
}

/// Pairwise total of `block` padded with `+0.0` to [`LLK_TREE_LEAVES`] nodes.
fn pair_tree(block: &[[f64; 4]]) -> [f64; 4] {
    let mut node = [[0.0; 4]; LLK_TREE_LEAVES];
    node[..block.len()].copy_from_slice(block);
    let mut width = LLK_TREE_LEAVES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            node[i] = std::array::from_fn(|t| node[2 * i][t] + node[2 * i + 1][t]);
        }
    }
    node[0]
}

/// `PIXEL_THREADS`-thread blocks covering `total` elements.
pub(crate) fn grid_1d(total: usize) -> Result<LaunchConfig, Error> {
    let blocks = total.div_ceil(PIXEL_THREADS as usize);
    let blocks = u32::try_from(blocks)
        .ok()
        .filter(|&b| b <= i32::MAX as u32)
        .ok_or_else(|| Error::Shape(format!("{total} elements exceed a 1D grid")))?;
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (PIXEL_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

impl Kernels {
    /// Exit waves of the stack: `psi[kl] = shift(O_v * P_v * exp(i quad r^2))` for view
    /// `v = f0 * nmode + kl`, `O_v` the object bilinearly sampled at the view's offset (PyNX
    /// `ObjProbe2Psi`). `psi` holds `nf_max * nmode` planes.
    pub fn obj_probe_to_psi(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        probe: &DeviceBuffer<C32>,
        psi: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let d = &views.dims;
        let g = d.geom(stack)?;
        let (n, nk_max) = (d.n(), g.nf_max as usize * d.nmode as usize);
        check_len("obj", obj, d.obj_len())?;
        check_len("probe", probe, d.nprobe as usize * n)?;
        check_len("psi", psi, nk_max * n)?;
        let config = grid_1d(nk_max * n)?;
        let reads = [
            views.addr.storage(),
            obj.storage(),
            probe.storage(),
            stack.storage(),
        ];
        self.launch(ctx, &reads, &[psi.storage()], |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object with valid mode indices.
            unsafe {
                self.module.obj_probe_to_psi(
                    stream,
                    config,
                    dptr(obj),
                    dptr(probe),
                    dptr(&views.addr),
                    dptr(psi),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Calculated intensity of the stack in detector order: `icalc[fl, d] = sum_m
    /// |psi[fl * nmode + m, fft(d)]|^2 / n` from the forward-FFT'd stack (PyNX `Calc2Obs`,
    /// no background). `icalc` holds `nf_max` frames.
    pub fn mode_intensity(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        psi: &DeviceBuffer<C32>,
        icalc: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        let (n, nf_max) = (dims.n(), g.nf_max as usize);
        check_len("psi", psi, nf_max * dims.nmode as usize * n)?;
        check_len("icalc", icalc, nf_max * n)?;
        let config = grid_1d(nf_max * n)?;
        let reads = [psi.storage(), stack.storage()];
        self.launch(ctx, &reads, &[icalc.storage()], |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.mode_intensity(
                    stream,
                    config,
                    dptr(psi),
                    dptr(icalc),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Fourier-modulus projection of the forward-FFT'd stack (PyNX `ApplyAmplitude`,
    /// no background, independent of the noise model): each pixel's modes are scaled by
    /// `sqrt(obs / max(icalc, 1e-12)) / n`, or by `1 / n` where the count is [`MASKED`];
    /// the unnormalized inverse FFT then yields the projected exit wave. `counts` holds
    /// the whole scan.
    pub fn apply_amplitude(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        counts: &DeviceBuffer<u16>,
        psi: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        let (n, nf_max) = (dims.n(), g.nf_max as usize);
        check_len("counts", counts, dims.nframe as usize * n)?;
        check_len("psi", psi, nf_max * dims.nmode as usize * n)?;
        let config = grid_1d(nf_max * n)?;
        let reads = [counts.storage(), stack.storage()];
        self.launch(ctx, &reads, &[psi.storage()], |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.apply_amplitude(
                    stream,
                    config,
                    dptr(psi),
                    dptr(counts),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Per-frame log-likelihood sums of the forward-FFT'd stack against the counts (PyNX
    /// `LLKAll`, no background): `out[fl]`, `fl < nf`, gets, over the unmasked pixels of
    /// frame `f0 + fl`, with `calc = sum_m |Psi_m|^2 / n`: Poisson `calc` if `obs == 0`
    /// else `calc - obs + obs ln(obs / calc)`; Gaussian `(obs - calc)^2 / (obs + 1)`;
    /// Euclidean `4 (sqrt obs - sqrt calc)^2`; and `calc`. Each frame's sum is fixed-order:
    /// one block reduction per chunk of `LLK_CHUNK` contiguous pixels into `scratch`, then
    /// the chunks in ascending order; total with [`LlkSums::from_partials`]. The scan's
    /// dims are the ones `scratch` was made for; `out` holds `nf_max` records. Two launches.
    pub fn llk(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        counts: &DeviceBuffer<u16>,
        psi: &DeviceBuffer<C32>,
        scratch: &mut LlkScratch,
        out: &mut DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        let dims = &scratch.dims;
        let g = dims.geom(stack)?;
        let n = dims.n();
        let nchunk = llk_chunks(n);
        check_len("counts", counts, dims.nframe as usize * n)?;
        check_len("psi", psi, g.nf_max as usize * dims.nmode as usize * n)?;
        check_len(
            "scratch",
            &scratch.chunks,
            g.nf_max as usize * nchunk as usize,
        )?;
        check_len("out", out, g.nf_max as usize)?;
        if g.nf_max > u16::MAX as u32 {
            return Err(Error::Shape(format!("{} frames exceed grid y", g.nf_max)));
        }
        let config = LaunchConfig {
            grid_dim: (nchunk, g.nf_max, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let reads = [counts.storage(), psi.storage(), stack.storage()];
        self.launch(ctx, &reads, &[scratch.chunks.storage()], |stream| {
            // SAFETY: lengths checked above; one block of PIXEL_THREADS per chunk and frame.
            unsafe {
                self.module.llk(
                    stream,
                    config,
                    dptr(psi),
                    dptr(counts),
                    dptr(&scratch.chunks),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })?;
        self.llk_groups(ctx, stack, g, &scratch.chunks, nchunk, out)
    }

    /// `out[fl]`, `fl < nf`, gets the sum of `groups[fl * ngroup .. (fl + 1) * ngroup]` in
    /// ascending order.
    pub(crate) fn llk_groups(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        g: StackGeom,
        groups: &DeviceBuffer<LlkPartial>,
        ngroup: u32,
        out: &mut DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        let nf_max = g.nf_max as usize;
        check_len("groups", groups, nf_max * ngroup as usize)?;
        check_len("out", out, nf_max)?;
        let config = grid_1d(nf_max)?;
        let reads = [groups.storage(), stack.storage()];
        self.launch(ctx, &reads, &[out.storage()], |stream| {
            // SAFETY: `groups` holds `nf_max * ngroup` records, `out` `nf_max`.
            unsafe {
                self.module.llk_groups(
                    stream,
                    config,
                    dptr(groups),
                    ngroup,
                    dptr(out),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Appends the stack's views to the tile lists: each view of frames `f0 .. f0 + nf` goes
    /// after the run of every owned tile [`view_tiles`] lists it in, in ascending view order.
    /// Stacks binned once each in ascending frame order leave the runs [`Views::new`] (or
    /// [`Views::owned`] of the same owned set) makes of the frames binned; in any other
    /// order the launches stay in bounds, but the runs and what the passes add over them are
    /// unspecified. It keeps the owned set: a tile without a slot gets no view and counts
    /// none in `overflow`. A view whose owned tile's slab `tile_start[t] .. tile_start[t +
    /// 1]` is full is not written and adds 1 to `overflow[0]`. A stack with `nf == 0` writes
    /// nothing.
    pub fn bin_views(
        &self,
        ctx: &ExecutionContext,
        views: &mut Views,
        stack: StackRef<'_>,
        overflow: &mut DeviceBuffer<u32>,
    ) -> Result<(), Error> {
        let d = &views.dims;
        let g = d.geom(stack)?;
        check_len("overflow", overflow, 1)?;
        let config = grid_1d(d.ntiles() * WARP as usize)?;
        let reads = [
            views.addr.storage(),
            views.tile_start.storage(),
            stack.storage(),
        ];
        let writes = [
            views.tile_len.storage(),
            views.tile_views.storage(),
            overflow.storage(),
        ];
        let slabs = views.tile_slabs();
        self.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: `Views::new` or the `from_parts` contract keeps the stack's views
            // inside the object; every constructor leaves each run within its slab, and
            // binning keeps it there; `overflow` holds one u32.
            unsafe {
                self.module.bin_views(
                    stream,
                    config,
                    dptr(&views.addr),
                    slabs,
                    dptr(overflow),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Fills `dst` with `src`'s frames in `order`, frame `p` of `dst` being frame `order[p]`
    /// of `src`, and rebuilds every run of `dst` as [`Self::bin_views`] over all frames
    /// would, so they are the runs [`Views::new`] (or [`Views::owned`] of `dst`'s owned set)
    /// makes of the reordered frames. It keeps `dst`'s owned set, which [`Views::reordered`]
    /// takes from `src`. A gather, a memset and a bin launch, stream-ordered with no host
    /// sync, so a graph can capture them. When they run, `dst`'s slabs hold every owned
    /// tile's views, as [`Views::reordered`] of `src` gives when `src`'s runs hold all its
    /// views; otherwise the launches stay in bounds but `dst`'s runs are unspecified.
    pub fn reorder_views(
        &self,
        ctx: &ExecutionContext,
        src: &Views,
        order: &Permutation,
        dst: &mut Views,
    ) -> Result<(), Error> {
        let d = &src.dims;
        if dst.dims != *d || order.order.len() != d.nframe as usize {
            return Err(Error::Shape(format!(
                "reorder of {d:?} by {} frames into {:?}",
                order.order.len(),
                dst.dims
            )));
        }
        let order = &order.order;
        let config = grid_1d(d.nframe as usize * d.nmode as usize)?;
        let reads = [src.addr.storage(), order.storage()];
        self.launch(ctx, &reads, &[dst.addr.storage()], |stream| {
            // SAFETY: `src` and `dst` own separate `nframe * nmode` records; `order` holds
            // `nframe` entries.
            unsafe {
                self.module.gather_frames(
                    stream,
                    config,
                    dptr(&src.addr),
                    dptr(order),
                    dptr(&dst.addr),
                    d.nframe,
                    d.nmode,
                )
            }
        })?;
        crate::gpu::zero(ctx, &mut dst.tile_len)?;
        let config = grid_1d(d.ntiles() * WARP as usize)?;
        let reads = [dst.addr.storage(), dst.tile_start.storage()];
        let writes = [dst.tile_len.storage(), dst.tile_views.storage()];
        let (slabs, g) = (dst.tile_slabs(), d.scan_geom());
        self.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: every view of `dst` is one it held or one of `src`'s, each of the same
            // dims and inside the object by `Views::new` or the `from_parts` contract; the
            // runs start empty, each within its slab.
            unsafe {
                self.module
                    .bin_all_views(stream, config, dptr(&dst.addr), slabs, g)
            }
        })
    }

    /// Adds the stack's object-update sums (PyNX `Psi2Obj`) to `sums` `[nobj, nyo, nxo]`:
    /// into `num` the exact adjoint of [`Self::obj_probe_to_psi`] applied to `psi`, into
    /// `den` the same bilinear spread of `|P_v|^2`. Each object pixel adds its views in
    /// ascending view order, so splitting the frames into stacks does not change a bit of
    /// the result. Only the pixels of tiles whose run holds a view of the stack are read and
    /// written.
    pub fn psi_to_obj(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        psi: &DeviceBuffer<C32>,
        probe: &DeviceBuffer<C32>,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let d = &views.dims;
        let g = d.geom(stack)?;
        let n = d.n();
        check_len("psi", psi, g.nf_max as usize * d.nmode as usize * n)?;
        check_len("probe", probe, d.nprobe as usize * n)?;
        sums.check(d.obj_len())?;
        let config = d.stack_tile_grid(g.nf_max)?;
        let [start, len, list] = views.tile_storages();
        let reads = [
            psi.storage(),
            probe.storage(),
            views.addr.storage(),
            start,
            len,
            list,
            stack.storage(),
        ];
        self.launch(ctx, &reads, &sums.storages(), |stream| {
            // SAFETY: lengths checked above; the tile lists were built from these views.
            unsafe {
                self.module.psi_to_obj(
                    stream,
                    config,
                    dptr(psi),
                    dptr(probe),
                    dptr(&views.addr),
                    views.tile_ptrs(),
                    sums.ptrs(),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Adds the stack's probe-update sums (PyNX `Psi2Probe`) to `sums` `[nprobe, ny, nx]`:
    /// `num += conj(O_v exp(i quad r^2)) psi_v` and `den += |O_v|^2` over the views of each
    /// probe mode, in ascending view order (stack-split invariant); a mode without a view in
    /// the stack is not written.
    pub fn psi_to_probe(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        psi: &DeviceBuffer<C32>,
        obj: &DeviceBuffer<C32>,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let d = &views.dims;
        let g = d.geom(stack)?;
        let n = d.n();
        check_len("psi", psi, g.nf_max as usize * d.nmode as usize * n)?;
        check_len("obj", obj, d.obj_len())?;
        sums.check(d.nprobe as usize * n)?;
        let config = grid_1d(d.nprobe as usize * n)?;
        let reads = [
            psi.storage(),
            obj.storage(),
            views.addr.storage(),
            stack.storage(),
        ];
        self.launch(ctx, &reads, &sums.storages(), |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view.
            unsafe {
                self.module.psi_to_probe(
                    stream,
                    config,
                    dptr(psi),
                    dptr(obj),
                    dptr(&views.addr),
                    sums.ptrs(),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Applies accumulated sums to the first `n` elements of `x` (PyNX `ObjNorm`):
    /// `x = (num + reg x) / max(den + reg, 1e-12)` with `reg = inertia * max(den[..n])`,
    /// every operation rounded on its own; then zeroes `sums` over `..n`, which it consumed.
    /// `sums.max_bits` is its scratch.
    pub fn update_from_sums(
        &self,
        ctx: &ExecutionContext,
        x: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        n: u32,
        inertia: f32,
    ) -> Result<(), Error> {
        check_len("x", x, n as usize)?;
        sums.check(n as usize)?;
        if n == 0 {
            return Ok(());
        }
        let config = grid_1d(n as usize)?;
        // Enough blocks to fill the device; the kernel strides over the rest.
        let max_config = LaunchConfig {
            grid_dim: (config.grid_dim.0.min(1024), 1, 1),
            ..config
        };
        let UpdateSums { num, den, max_bits } = sums;
        crate::gpu::zero(ctx, max_bits)?;
        self.launch(ctx, &[den.storage()], &[max_bits.storage()], |stream| {
            // SAFETY: `den` holds `n` floats, `max_bits` one zeroed u32.
            unsafe {
                self.module
                    .max_nonneg_f32(stream, max_config, dptr(den), n, dptr(max_bits))
            }
        })?;
        let writes = [x.storage(), num.storage(), den.storage()];
        let ptrs = SumPtrs {
            num: dptr(num),
            den: dptr(den),
        };
        self.launch(ctx, &[max_bits.storage()], &writes, |stream| {
            // SAFETY: all lengths checked above.
            unsafe {
                self.module.normalize_update(
                    stream,
                    config,
                    dptr(x),
                    ptrs,
                    n,
                    dptr(max_bits),
                    inertia,
                )
            }
        })
    }

    /// [`Self::update_from_sums`] of the object `[nobj, nyo, nxo]` on the object tiles with a
    /// non-empty run only: `reg = inertia * max(den)` over their pixels, then the update of each,
    /// zeroing its sums. A tile whose run is empty is neither read nor written, in `obj` or
    /// `sums`, so it keeps its bits; when its `den` is zero, as in sums built zeroed (no stack
    /// adds to such a tile), every other pixel gets the bits `update_from_sums` gives it.
    /// It is [`Self::obj_den_max`] then [`Self::normalize_obj`] with `sums.max_bits`.
    pub fn update_obj_from_sums(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        obj: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        inertia: f32,
    ) -> Result<(), Error> {
        self.obj_den_max(ctx, views, sums)?;
        let UpdateSums { num, den, max_bits } = sums;
        self.normalize_obj_tiles(ctx, views, obj, (num, den), max_bits, inertia)
    }

    /// The first step of [`Self::update_obj_from_sums`]: the bits of `max(den)` over the
    /// pixels of the object tiles with a non-empty run into `sums.max_bits`. The max is exact and
    /// independent of order, so the largest of several devices' words is the max over the
    /// union of their tiles.
    pub fn obj_den_max(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let d = &views.dims;
        sums.check(d.obj_len())?;
        let config = d.tile_grid();
        let [nyo, nxo] = d.obj;
        let tile_len = &views.tile_len;
        let UpdateSums { den, max_bits, .. } = sums;
        crate::gpu::zero(ctx, max_bits)?;
        let reads = [den.storage(), tile_len.storage()];
        self.launch(ctx, &reads, &[max_bits.storage()], |stream| {
            // SAFETY: `den` holds the object, `tile_len` its tiles, `max_bits` one zeroed u32.
            unsafe {
                self.module.max_den_tiles(
                    stream,
                    config,
                    dptr(den),
                    dptr(tile_len),
                    dptr(max_bits),
                    nyo,
                    nxo,
                )
            }
        })
    }

    /// The second step of [`Self::update_obj_from_sums`], with `reg = inertia *
    /// f32::from_bits(max_bits[0])` read on the device when it runs (a captured graph reads
    /// whatever an exchange of [`Self::obj_den_max`] words left there): the update of each
    /// pixel of the object tiles with a non-empty run, zeroing its sums. `sums.max_bits` is
    /// not used.
    pub fn normalize_obj(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        obj: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        max_bits: &DeviceBuffer<u32>,
        inertia: f32,
    ) -> Result<(), Error> {
        check_len("max_bits", max_bits, 1)?;
        let UpdateSums { num, den, .. } = sums;
        self.normalize_obj_tiles(ctx, views, obj, (num, den), max_bits, inertia)
    }

    /// [`Self::normalize_obj`] over `sums` as `(num, den)`, so that
    /// [`Self::update_obj_from_sums`] can pass the max word of the same [`UpdateSums`].
    fn normalize_obj_tiles(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        obj: &mut DeviceBuffer<C32>,
        (num, den): (&mut DeviceBuffer<C32>, &mut DeviceBuffer<f32>),
        max_bits: &DeviceBuffer<u32>,
        inertia: f32,
    ) -> Result<(), Error> {
        let d = &views.dims;
        check_len("obj", obj, d.obj_len())?;
        check_len("sums.num", num, d.obj_len())?;
        check_len("sums.den", den, d.obj_len())?;
        let config = d.tile_grid();
        let [nyo, nxo] = d.obj;
        let tile_len = &views.tile_len;
        let reads = [tile_len.storage(), max_bits.storage()];
        let writes = [obj.storage(), num.storage(), den.storage()];
        let ptrs = SumPtrs {
            num: dptr(num),
            den: dptr(den),
        };
        self.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: all lengths checked above.
            unsafe {
                self.module.normalize_tiles(
                    stream,
                    config,
                    dptr(obj),
                    ptrs,
                    dptr(tile_len),
                    dptr(max_bits),
                    inertia,
                    nyo,
                    nxo,
                )
            }
        })
    }

    /// DM reflection over `n` elements (PyNX `DM1`, plus the term `DM2` adds back): with
    /// `psi1 = (2 - alpha) po - (1 - alpha) psi`, sets `psi = psi1` and
    /// `po = (1 - alpha) (po - psi1)`. `po` holds `obj_probe_to_psi` of the current object
    /// and probe; after the Fourier projection of `psi`, [`Self::add_c32`]`(psi, po)` gives
    /// PyNX's `DM2` result.
    pub fn dm_reflect(
        &self,
        ctx: &ExecutionContext,
        psi: &mut DeviceBuffer<C32>,
        po: &mut DeviceBuffer<C32>,
        n: u32,
        alpha: f32,
    ) -> Result<(), Error> {
        check_len("psi", psi, n as usize)?;
        check_len("po", po, n as usize)?;
        if n == 0 {
            return Ok(());
        }
        let config = grid_1d(n as usize)?;
        self.launch(ctx, &[], &[psi.storage(), po.storage()], |stream| {
            // SAFETY: both hold `n` values.
            unsafe {
                self.module
                    .dm_reflect(stream, config, dptr(psi), dptr(po), n, alpha)
            }
        })
    }

    /// `y += x` for the first `n` values.
    pub fn add_c32(
        &self,
        ctx: &ExecutionContext,
        y: &mut DeviceBuffer<C32>,
        x: &DeviceBuffer<C32>,
        n: u32,
    ) -> Result<(), Error> {
        check_len("y", y, n as usize)?;
        check_len("x", x, n as usize)?;
        if n == 0 {
            return Ok(());
        }
        let config = grid_1d(n as usize)?;
        self.launch(ctx, &[x.storage()], &[y.storage()], |stream| {
            // SAFETY: both hold `n` values.
            unsafe { self.module.add_c32(stream, config, dptr(y), dptr(x), n) }
        })
    }
}

/// Numerator and denominator of an object or probe update (PyNX `obj_new`, `obj_norm`),
/// element `i` of `den` belonging to element `i` of `num`, with the scratch the update
/// needs, so that no launch allocates. They hold what was accumulated since the last
/// [`Kernels::update_from_sums`], [`Kernels::update_obj_from_sums`] or
/// [`Kernels::normalize_obj`] consumed them; the only other reset is construction.
pub struct UpdateSums {
    pub num: DeviceBuffer<C32>,
    pub den: DeviceBuffer<f32>,
    /// One word: [`Kernels::update_from_sums`] and [`Kernels::obj_den_max`] write the bits of
    /// `max(den)` here.
    pub max_bits: DeviceBuffer<u32>,
}

impl UpdateSums {
    pub fn zeroed(stream: &CudaStream, len: usize) -> Result<Self, Error> {
        Self::new(
            DeviceBuffer::zeroed(stream, len)?,
            DeviceBuffer::zeroed(stream, len)?,
            stream,
        )
    }

    /// `num` and `den` with a fresh `max_bits`.
    pub fn new(
        num: DeviceBuffer<C32>,
        den: DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<Self, Error> {
        Ok(Self {
            num,
            den,
            max_bits: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub(crate) fn check(&self, need: usize) -> Result<(), Error> {
        check_len("sums.num", &self.num, need)?;
        check_len("sums.den", &self.den, need)?;
        check_len("sums.max_bits", &self.max_bits, 1)
    }

    pub(crate) fn storages(&self) -> [&cuda_core::RawDeviceBuffer; 2] {
        [self.num.storage(), self.den.storage()]
    }

    pub(crate) fn ptrs(&self) -> SumPtrs {
        SumPtrs {
            num: dptr(&self.num),
            den: dptr(&self.den),
        }
    }
}
