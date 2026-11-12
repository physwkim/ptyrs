//! M1 pixel operators: checked launches of the non-fused exit-wave kernels.
//!
//! # Contract (shared with the engine's CPU oracle)
//!
//! * Shapes are row-major. A frame, the probe and the detector are `[ny, nx]` with `ny`, `nx`
//!   even; `n = ny * nx`. `probe` is `[nprobe, ny, nx]`, `obj` `[nobj, nyo, nxo]`, both in
//!   natural order (no shift). Complex values are [`C32`].
//! * Views are frame-major, `k = frame * nmode + mode`; [`ViewAddr`] says which probe and
//!   object mode a view multiplies and where it samples the object. A stack is frames
//!   `f0 .. f0 + nf`, a [`StackCtl`] entry of a [`StackTable`] that the kernels read when they
//!   run, not when they are launched or captured. Per-stack buffers are sized by the table's
//!   `nf_max`: `psi` holds `nf_max * nmode` planes, plane `kl < nf * nmode` being view
//!   `f0 * nmode + kl`; planes and records past the stack are left as they are, and a stack
//!   with `nf == 0` writes nothing.
//! * Real-space exit waves have their origin at `[0, 0]`: probe pixel `(y, x)` is stored at
//!   `((y + ny/2) % ny, (x + nx/2) % nx)`. So a plain 2D FFT of each plane
//!   (`ptyrs_fft::Plan2d`, unnormalized, forward `exp(-2 pi i jk / n)`) gives the far field in
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
//! `psi_to_probe` into zeroed sums; after the last stack, `update_obj_from_sums` with inertia
//! 0.01 (object) or `update_from_sums` with 0.001 (probe).
//!
//! DM (`alpha` 0.02, `psi` kept per stack across cycles): `obj_probe_to_psi` into a scratch
//! `po`, then on the first cycle copy `po` into `psi` (PyNX starts from `psi = PO`, which
//! [`Kernels::dm_reflect`] maps to itself), `dm_reflect(psi, po)`, FFT, `apply_amplitude`,
//! inverse FFT, `add_c32(psi, po)`; then the object/probe sums and updates as in AP, from the
//! new `psi`.
//!
//! LLK: forward FFT of `obj_probe_to_psi`, then [`Kernels::llk`] per stack; PyNX's reported
//! (normalized) values are [`LlkSums::from_partials`] over all frames divided by the number
//! of unmasked count pixels of the scan.

use std::ops::Range;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::sys::CUdeviceptr;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig, RawDeviceBuffer};
pub use ptyrs_kernels::pixel::{LlkPartial, MASKED, OBJ_TILE, StackCtl, ViewAddr};
use ptyrs_kernels::pixel::{
    PIXEL_THREADS, StackGeom, SumPtrs, TilePtrs, TileSlabs, WARP, axis_tiles,
};

use crate::{C32, Error, Kernels, check_len, dptr};

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
    fn n(&self) -> usize {
        self.frame[0] as usize * self.frame[1] as usize
    }

    fn obj_len(&self) -> usize {
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
    fn tile_grid(&self) -> LaunchConfig {
        let [nty, ntx] = self.tiles();
        LaunchConfig {
            grid_dim: (ntx, nty, self.nobj),
            block_dim: (OBJ_TILE, OBJ_TILE, 1),
            shared_mem_bytes: 0,
        }
    }

    fn check(&self) -> Result<(), Error> {
        let [ny, nx] = self.frame;
        let [nyo, nxo] = self.obj;
        let bad = |what: &str| Err(Error::Shape(format!("{self:?}: {what}")));
        if self.nframe == 0 || self.nmode == 0 || self.nprobe == 0 || self.nobj == 0 {
            return bad("a count is zero");
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
    /// of `nframe` frames.
    fn geom(&self, stack: StackRef<'_>) -> Result<StackGeom, Error> {
        self.check()?;
        let t = stack.table;
        if t.nframe != self.nframe {
            return Err(Error::Shape(format!(
                "stack table made for {} frames, dims have {}",
                t.nframe, self.nframe
            )));
        }
        Ok(StackGeom {
            nframe: self.nframe,
            nf_max: t.nf_max,
            nmode: self.nmode,
            nprobe: self.nprobe,
            ny: self.frame[0],
            nx: self.frame[1],
            nyo: self.obj[0],
            nxo: self.obj[1],
            quad: self.quad_phase,
        })
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

    fn ctl(&self) -> *const StackCtl {
        (self.table.ptr() + (self.s * size_of::<StackCtl>()) as CUdeviceptr) as *const StackCtl
    }

    fn storage(&self) -> &RawDeviceBuffer {
        self.table.ctl.storage()
    }
}

/// The views of a scan on the device, with the per-object-tile view lists the object
/// gather walks: tile `t`'s run is `tile_views[tile_start[t] .. tile_start[t] + tile_len[t]]`.
pub struct Views {
    dims: ViewDims,
    addr: DeviceBuffer<ViewAddr>,
    /// `ntiles + 1` ascending offsets into `tile_views`; tile `ty * ntx + tx` of `OBJ_TILE`
    /// pixels.
    tile_start: DeviceBuffer<u32>,
    /// Per tile, ascending indices of the views whose samples may touch it.
    tile_views: DeviceBuffer<u32>,
    /// `ntiles` run lengths, at most `tile_start[t + 1] - tile_start[t]`. No launch reads or
    /// writes the object or sum pixels of a tile whose run is empty.
    tile_len: DeviceBuffer<u32>,
}

impl Views {
    /// Checks `addr` (`nframe * nmode` views, each inside the object with `fy`, `fx` in
    /// `[0, 1)` and valid mode indices) and uploads it with its tile lists.
    pub fn new(stream: &CudaStream, dims: ViewDims, addr: &[ViewAddr]) -> Result<Self, Error> {
        dims.check()?;
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
        let t = tile_lists(&dims, addr)?;
        Ok(Self {
            dims,
            addr: DeviceBuffer::from_host(stream, addr)?,
            tile_start: DeviceBuffer::from_host(stream, &t.start)?,
            tile_views: DeviceBuffer::from_host(stream, &t.views)?,
            tile_len: DeviceBuffer::from_host(stream, &t.len)?,
        })
    }

    /// Views over device storage the caller keeps filling, e.g. through
    /// [`Kernels::bin_views`] into a slab of `tile_cap` slots per tile (`tile_start[t] = t *
    /// tile_cap`) with `tile_len` starting zeroed. Checks `dims`
    /// and that `addr` holds `nframe * nmode` records, `tile_start` `ntiles + 1` and
    /// `tile_len` `ntiles` entries.
    ///
    /// # Safety
    ///
    /// Whenever a launch reads them: `tile_start` is ascending with `tile_start[ntiles] <=
    /// tile_views.len()`; `tile_len[t] <= tile_start[t + 1] - tile_start[t]`; each run holds
    /// ascending indices of views that may touch the tile, every view of the launch's stack
    /// that does among them; and every view the launch reads passes the checks of
    /// [`Self::new`].
    pub unsafe fn from_parts(
        dims: ViewDims,
        addr: DeviceBuffer<ViewAddr>,
        tile_start: DeviceBuffer<u32>,
        tile_views: DeviceBuffer<u32>,
        tile_len: DeviceBuffer<u32>,
    ) -> Result<Self, Error> {
        dims.check()?;
        let ntiles = dims.ntiles();
        check_len("addr", &addr, dims.nframe as usize * dims.nmode as usize)?;
        check_len("tile_start", &tile_start, ntiles + 1)?;
        check_len("tile_len", &tile_len, ntiles)?;
        Ok(Self {
            dims,
            addr,
            tile_start,
            tile_views,
            tile_len,
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

    fn tile_ptrs(&self) -> TilePtrs {
        TilePtrs {
            start: dptr(&self.tile_start),
            len: dptr(&self.tile_len),
            views: dptr(&self.tile_views),
        }
    }

    fn tile_storages(&self) -> [&RawDeviceBuffer; 3] {
        [
            self.tile_start.storage(),
            self.tile_len.storage(),
            self.tile_views.storage(),
        ]
    }
}

/// The object tiles `[ty, tx]` holding the object pixels that the samples of view `v` read,
/// ascending (ty-major); for a view [`ViewDims::check_view`] accepts, all inside the object.
/// [`Views::new`] and [`Kernels::bin_views`] list view `v` in exactly these tiles' runs.
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

/// CSR lists of the views touching each object tile, ascending per tile.
fn tile_lists(dims: &ViewDims, addr: &[ViewAddr]) -> Result<TileLists, Error> {
    let ntx = dims.tiles()[1] as usize;
    let mut lists = vec![Vec::new(); dims.ntiles()];
    for (k, v) in addr.iter().enumerate() {
        for [ty, tx] in view_tiles(dims, v) {
            lists[ty as usize * ntx + tx as usize].push(k as u32);
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

/// Frame-ordered `f64` totals of [`LlkPartial`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LlkSums {
    pub poisson: f64,
    pub gaussian: f64,
    pub euclidean: f64,
    /// Total calculated intensity (PyNX `nb_photons_calc`).
    pub icalc: f64,
}

impl LlkSums {
    /// Sums in slice order, so a fixed frame order gives a fixed result.
    pub fn from_partials(partials: &[LlkPartial]) -> Self {
        partials.iter().fold(Self::default(), |s, p| Self {
            poisson: s.poisson + p.poisson as f64,
            gaussian: s.gaussian + p.gaussian as f64,
            euclidean: s.euclidean + p.euclidean as f64,
            icalc: s.icalc + p.icalc as f64,
        })
    }
}

/// `PIXEL_THREADS`-thread blocks covering `total` elements.
fn grid_1d(total: usize) -> Result<LaunchConfig, Error> {
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
                    stack.ctl(),
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
                self.module
                    .mode_intensity(stream, config, dptr(psi), dptr(icalc), stack.ctl(), g)
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
                self.module
                    .apply_amplitude(stream, config, dptr(psi), dptr(counts), stack.ctl(), g)
            }
        })
    }

    /// Per-frame log-likelihood sums of the forward-FFT'd stack against the counts (PyNX
    /// `LLKAll`, no background): `out[fl]`, `fl < nf`, gets, over the unmasked pixels of
    /// frame `f0 + fl`, with `calc = sum_m |Psi_m|^2 / n`: Poisson `calc` if `obs == 0`
    /// else `calc - obs + obs ln(obs / calc)`; Gaussian `(obs - calc)^2 / (obs + 1)`;
    /// Euclidean `4 (sqrt obs - sqrt calc)^2`; and `calc`. Each frame's sum is a fixed-order
    /// block reduction; total with [`LlkSums::from_partials`]. `out` holds `nf_max` records.
    pub fn llk(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        counts: &DeviceBuffer<u16>,
        psi: &DeviceBuffer<C32>,
        out: &mut DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        let n = dims.n();
        check_len("counts", counts, dims.nframe as usize * n)?;
        check_len("psi", psi, g.nf_max as usize * dims.nmode as usize * n)?;
        check_len("out", out, g.nf_max as usize)?;
        let config = LaunchConfig {
            grid_dim: (g.nf_max, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let reads = [counts.storage(), psi.storage(), stack.storage()];
        self.launch(ctx, &reads, &[out.storage()], |stream| {
            // SAFETY: lengths checked above; one block of PIXEL_THREADS per frame.
            unsafe {
                self.module.llk(
                    stream,
                    config,
                    dptr(psi),
                    dptr(counts),
                    dptr(out),
                    stack.ctl(),
                    g,
                )
            }
        })
    }

    /// Appends the stack's views to the tile lists: each view of frames `f0 .. f0 + nf` goes
    /// after the run of every tile [`view_tiles`] lists it in, in ascending view order, so
    /// runs stay ascending while stacks are binned once each in ascending frame order (the
    /// order [`Views::from_parts`] requires of them). A view whose tile's slab
    /// `tile_start[t] .. tile_start[t + 1]` is full is not written and adds 1 to
    /// `overflow[0]`. A stack with `nf == 0` writes nothing.
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
        let slabs = TileSlabs {
            start: dptr(&views.tile_start),
            len: dptr(&views.tile_len),
            views: dptr(&views.tile_views),
        };
        self.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: `Views::new` or the `from_parts` contract keeps the views inside the
            // object and each run within its slab; `overflow` holds one u32.
            unsafe {
                self.module.bin_views(
                    stream,
                    config,
                    dptr(&views.addr),
                    slabs,
                    dptr(overflow),
                    stack.ctl(),
                    g,
                )
            }
        })
    }

    /// Adds the stack's object-update sums (PyNX `Psi2Obj`) to `sums` `[nobj, nyo, nxo]`:
    /// into `num` the exact adjoint of [`Self::obj_probe_to_psi`] applied to `psi`, into
    /// `den` the same bilinear spread of `|P_v|^2`. Each object pixel adds its views in
    /// ascending view order, so splitting the frames into stacks does not change a bit of
    /// the result. Only the pixels of tiles the stack's views touch are read and written.
    /// Zero `sums` before the first stack.
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
        let config = d.tile_grid();
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
                    stack.ctl(),
                    g,
                )
            }
        })
    }

    /// Adds the stack's probe-update sums (PyNX `Psi2Probe`) to `sums` `[nprobe, ny, nx]`:
    /// `num += conj(O_v exp(i quad r^2)) psi_v` and `den += |O_v|^2` over the views of each
    /// probe mode, in ascending view order (stack-split invariant); a mode without a view in
    /// the stack is not written. Zero `sums` before the first stack.
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
                    stack.ctl(),
                    g,
                )
            }
        })
    }

    /// Applies accumulated sums to the first `n` elements of `x` (PyNX `ObjNorm`):
    /// `x = (num + reg x) / max(den + reg, 1e-12)` with `reg = inertia * max(den[..n])`,
    /// every operation rounded on its own. `sums.max_bits` is its scratch.
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
        crate::zero(ctx, max_bits)?;
        self.launch(ctx, &[den.storage()], &[max_bits.storage()], |stream| {
            // SAFETY: `den` holds `n` floats, `max_bits` one zeroed u32.
            unsafe {
                self.module
                    .max_nonneg_f32(stream, max_config, dptr(den), n, dptr(max_bits))
            }
        })?;
        let reads = [num.storage(), den.storage(), max_bits.storage()];
        self.launch(ctx, &reads, &[x.storage()], |stream| {
            // SAFETY: all lengths checked above.
            unsafe {
                self.module.normalize_update(
                    stream,
                    config,
                    dptr(x),
                    dptr(num),
                    dptr(den),
                    n,
                    dptr(max_bits),
                    inertia,
                )
            }
        })
    }

    /// [`Self::update_from_sums`] of the object `[nobj, nyo, nxo]` on the object tiles with
    /// views only: `reg = inertia * max(den)` over their pixels, then the update of each. A
    /// tile whose run is empty is neither read nor written, in `obj` or `sums`, so it keeps
    /// its bits; with `den` zero there, as after [`UpdateSums::zero`], every other pixel
    /// gets the bits `update_from_sums` gives it.
    pub fn update_obj_from_sums(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        obj: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        inertia: f32,
    ) -> Result<(), Error> {
        let d = &views.dims;
        check_len("obj", obj, d.obj_len())?;
        sums.check(d.obj_len())?;
        let config = d.tile_grid();
        let [nyo, nxo] = d.obj;
        let tile_len = &views.tile_len;
        let UpdateSums { num, den, max_bits } = sums;
        crate::zero(ctx, max_bits)?;
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
        })?;
        let reads = [
            num.storage(),
            den.storage(),
            tile_len.storage(),
            max_bits.storage(),
        ];
        let ptrs = SumPtrs {
            num: dptr(num),
            den: dptr(den),
        };
        self.launch(ctx, &reads, &[obj.storage()], |stream| {
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
/// needs, so that no launch allocates.
pub struct UpdateSums {
    pub num: DeviceBuffer<C32>,
    pub den: DeviceBuffer<f32>,
    /// One word: [`Kernels::update_from_sums`] writes the bits of `max(den)` here.
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

    /// Zeroes `num` and `den` on `ctx`'s stream.
    pub fn zero(&mut self, ctx: &ExecutionContext) -> Result<(), Error> {
        crate::zero(ctx, &mut self.num)?;
        crate::zero(ctx, &mut self.den)
    }

    fn check(&self, need: usize) -> Result<(), Error> {
        check_len("sums.num", &self.num, need)?;
        check_len("sums.den", &self.den, need)?;
        check_len("sums.max_bits", &self.max_bits, 1)
    }

    fn storages(&self) -> [&cuda_core::RawDeviceBuffer; 2] {
        [self.num.storage(), self.den.storage()]
    }

    fn ptrs(&self) -> SumPtrs {
        SumPtrs {
            num: dptr(&self.num),
            den: dptr(&self.den),
        }
    }
}
