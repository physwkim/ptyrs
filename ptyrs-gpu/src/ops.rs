//! M1 pixel operators: checked launches of the non-fused exit-wave kernels.
//!
//! # Contract (shared with the engine's CPU oracle)
//!
//! * Shapes are row-major. A frame, the probe and the detector are `[ny, nx]` with `ny`, `nx`
//!   even; `n = ny * nx`. `probe` is `[nprobe, ny, nx]`, `obj` `[nobj, nyo, nxo]`, both in
//!   natural order (no shift). Complex values are [`C32`].
//! * Views are frame-major, `k = frame * nmode + mode`; [`ViewAddr`] says which probe and
//!   object mode a view multiplies and where it samples the object. A stack is a frame range
//!   `f0 .. f1`; its exit-wave buffer `psi` holds planes `0 .. (f1 - f0) * nmode`, plane `kl`
//!   being view `f0 * nmode + kl`.
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
//! `psi_to_probe` into zeroed sums; after the last stack, `update_from_sums` with inertia
//! 0.01 (object) or 0.001 (probe).
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

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
pub use ptyrs_kernels::pixel::{LlkPartial, MASKED, OBJ_TILE, ViewAddr};
use ptyrs_kernels::pixel::{PIXEL_THREADS, StackGeom, SumPtrs};

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

/// A frame range resolved against [`ViewDims`].
struct Stack {
    f0: u32,
    nf: u32,
    k0: u32,
    nk: u32,
}

impl ViewDims {
    fn n(&self) -> usize {
        self.frame[0] as usize * self.frame[1] as usize
    }

    fn obj_len(&self) -> usize {
        self.nobj as usize * self.obj[0] as usize * self.obj[1] as usize
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

    fn stack(&self, frames: &Range<u32>) -> Result<Stack, Error> {
        self.check()?;
        if frames.start >= frames.end || frames.end > self.nframe {
            return Err(Error::Shape(format!(
                "frames {frames:?} is not a non-empty part of 0..{}",
                self.nframe
            )));
        }
        let nf = frames.end - frames.start;
        Ok(Stack {
            f0: frames.start,
            nf,
            k0: frames.start * self.nmode,
            nk: nf * self.nmode,
        })
    }

    /// The view kernels' geometry for stack `s`.
    fn geom(&self, s: &Stack) -> StackGeom {
        StackGeom {
            k0: s.k0,
            nk: s.nk,
            nprobe: self.nprobe,
            ny: self.frame[0],
            nx: self.frame[1],
            nyo: self.obj[0],
            nxo: self.obj[1],
            quad: self.quad_phase,
        }
    }
}

/// The views of a scan on the device, with the per-object-tile view lists the object
/// gather walks.
pub struct Views {
    dims: ViewDims,
    addr: DeviceBuffer<ViewAddr>,
    /// `ntiles + 1` offsets into `tile_views`; tile `ty * ntx + tx` of `OBJ_TILE` pixels.
    tile_start: DeviceBuffer<u32>,
    /// Per tile, ascending indices of the views whose samples may touch it.
    tile_views: DeviceBuffer<u32>,
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
        let [ny, nx] = dims.frame;
        let [nyo, nxo] = dims.obj;
        let frac = |f: f32| (0.0..1.0).contains(&f);
        for (k, v) in addr.iter().enumerate() {
            let inside =
                v.oy as u64 + ny as u64 <= nyo as u64 && v.ox as u64 + nx as u64 <= nxo as u64;
            if !inside || !frac(v.fy) || !frac(v.fx) || v.probe >= dims.nprobe || v.obj >= dims.nobj
            {
                return Err(Error::Shape(format!("view {k}: {v:?} is outside {dims:?}")));
            }
        }
        let (start, list) = tile_lists(&dims, addr)?;
        Ok(Self {
            dims,
            addr: DeviceBuffer::from_host(stream, addr)?,
            tile_start: DeviceBuffer::from_host(stream, &start)?,
            tile_views: DeviceBuffer::from_host(stream, &list)?,
        })
    }

    pub fn dims(&self) -> &ViewDims {
        &self.dims
    }

    /// The `nframe * nmode` view records, view `k` at index `k`.
    pub fn addr(&self) -> &DeviceBuffer<ViewAddr> {
        &self.addr
    }

    /// `ntiles + 1` CSR offsets into [`Self::tile_views`]; object tile `ty * ntx + tx`
    /// covers `OBJ_TILE` pixels per side, `ntx = nxo.div_ceil(OBJ_TILE)`.
    pub fn tile_start(&self) -> &DeviceBuffer<u32> {
        &self.tile_start
    }

    /// Per object tile, the ascending indices of the views whose samples may touch it.
    pub fn tile_views(&self) -> &DeviceBuffer<u32> {
        &self.tile_views
    }
}

/// Object tiles (of `OBJ_TILE` pixels per side) that a view's samples may touch along one
/// axis: offsets `o .. o + len` plus the bilinear `+ 1` neighbour of the last, wrapped.
fn axis_tiles(o: u32, len: u32, n: u32) -> impl Iterator<Item = u32> {
    let (first, last) = (o / OBJ_TILE, (o + len - 1) / OBJ_TILE);
    let wrap = ((o + len) % n) / OBJ_TILE;
    (first..=last).chain((wrap < first || wrap > last).then_some(wrap))
}

/// CSR lists of the views touching each object tile, ascending per tile.
fn tile_lists(dims: &ViewDims, addr: &[ViewAddr]) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let [ny, nx] = dims.frame;
    let [nyo, nxo] = dims.obj;
    let ntx = nxo.div_ceil(OBJ_TILE) as usize;
    let ntiles = nyo.div_ceil(OBJ_TILE) as usize * ntx;
    let mut lists = vec![Vec::new(); ntiles];
    for (k, v) in addr.iter().enumerate() {
        for ty in axis_tiles(v.oy, ny, nyo) {
            for tx in axis_tiles(v.ox, nx, nxo) {
                lists[ty as usize * ntx + tx as usize].push(k as u32);
            }
        }
    }
    let mut start = Vec::with_capacity(ntiles + 1);
    let mut flat = Vec::new();
    start.push(0);
    for l in lists {
        flat.extend(l);
        start.push(
            u32::try_from(flat.len())
                .map_err(|_| Error::Shape("tile lists exceed u32::MAX entries".into()))?,
        );
    }
    Ok((start, flat))
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
    /// Exit waves of `frames`: `psi[kl] = shift(O_v * P_v * exp(i quad r^2))` for view
    /// `v = frames.start * nmode + kl`, `O_v` the object bilinearly sampled at the view's
    /// offset (PyNX `ObjProbe2Psi`). `psi` holds `frames.len() * nmode` planes.
    pub fn obj_probe_to_psi(
        &self,
        stream: &CudaStream,
        views: &Views,
        frames: Range<u32>,
        obj: &DeviceBuffer<C32>,
        probe: &DeviceBuffer<C32>,
        psi: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let d = &views.dims;
        let s = d.stack(&frames)?;
        let n = d.n();
        check_len("obj", obj, d.obj_len())?;
        check_len("probe", probe, d.nprobe as usize * n)?;
        check_len("psi", psi, s.nk as usize * n)?;
        let config = grid_1d(s.nk as usize * n)?;
        let reads = [views.addr.storage(), obj.storage(), probe.storage()];
        self.launch(stream, &reads, &[psi.storage()], || {
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
                    d.geom(&s),
                )
            }
        })
    }

    /// Calculated intensity of `frames` in detector order: `icalc[fl, d] = sum_m
    /// |psi[fl * nmode + m, fft(d)]|^2 / n` from the forward-FFT'd stack (PyNX `Calc2Obs`,
    /// no background). `icalc` holds `frames.len()` frames.
    pub fn mode_intensity(
        &self,
        stream: &CudaStream,
        dims: &ViewDims,
        frames: Range<u32>,
        psi: &DeviceBuffer<C32>,
        icalc: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let s = dims.stack(&frames)?;
        let n = dims.n();
        check_len("psi", psi, s.nk as usize * n)?;
        check_len("icalc", icalc, s.nf as usize * n)?;
        let config = grid_1d(s.nf as usize * n)?;
        self.launch(stream, &[psi.storage()], &[icalc.storage()], || {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.mode_intensity(
                    stream,
                    config,
                    dptr(psi),
                    dptr(icalc),
                    s.nf,
                    dims.nmode,
                    dims.frame[0],
                    dims.frame[1],
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
        stream: &CudaStream,
        dims: &ViewDims,
        frames: Range<u32>,
        counts: &DeviceBuffer<u16>,
        psi: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let s = dims.stack(&frames)?;
        let n = dims.n();
        check_len("counts", counts, dims.nframe as usize * n)?;
        check_len("psi", psi, s.nk as usize * n)?;
        let config = grid_1d(s.nf as usize * n)?;
        self.launch(stream, &[counts.storage()], &[psi.storage()], || {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.apply_amplitude(
                    stream,
                    config,
                    dptr(psi),
                    dptr(counts),
                    s.f0,
                    s.nf,
                    dims.nmode,
                    dims.frame[0],
                    dims.frame[1],
                )
            }
        })
    }

    /// Per-frame log-likelihood sums of the forward-FFT'd stack against the counts (PyNX
    /// `LLKAll`, no background): `out[fl]` gets, over the unmasked pixels of frame
    /// `frames.start + fl`, with `calc = sum_m |Psi_m|^2 / n`: Poisson `calc` if `obs == 0`
    /// else `calc - obs + obs ln(obs / calc)`; Gaussian `(obs - calc)^2 / (obs + 1)`;
    /// Euclidean `4 (sqrt obs - sqrt calc)^2`; and `calc`. Each frame's sum is a fixed-order
    /// block reduction; total with [`LlkSums::from_partials`].
    pub fn llk(
        &self,
        stream: &CudaStream,
        dims: &ViewDims,
        frames: Range<u32>,
        counts: &DeviceBuffer<u16>,
        psi: &DeviceBuffer<C32>,
        out: &mut DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        let s = dims.stack(&frames)?;
        let n = dims.n();
        check_len("counts", counts, dims.nframe as usize * n)?;
        check_len("psi", psi, s.nk as usize * n)?;
        check_len("out", out, s.nf as usize)?;
        let config = LaunchConfig {
            grid_dim: (s.nf, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.launch(
            stream,
            &[counts.storage(), psi.storage()],
            &[out.storage()],
            || {
                // SAFETY: lengths checked above; one block of PIXEL_THREADS per frame.
                unsafe {
                    self.module.llk(
                        stream,
                        config,
                        dptr(psi),
                        dptr(counts),
                        dptr(out),
                        s.f0,
                        dims.nmode,
                        dims.frame[0],
                        dims.frame[1],
                    )
                }
            },
        )
    }

    /// Adds the stack's object-update sums (PyNX `Psi2Obj`) to `sums` `[nobj, nyo, nxo]`:
    /// into `num` the exact adjoint of [`Self::obj_probe_to_psi`] applied to `psi`, into
    /// `den` the same bilinear spread of `|P_v|^2`. Each object pixel adds its views in
    /// ascending view order, so splitting the frames into stacks does not change a bit of
    /// the result. Zero `sums` before the first stack.
    pub fn psi_to_obj(
        &self,
        stream: &CudaStream,
        views: &Views,
        frames: Range<u32>,
        psi: &DeviceBuffer<C32>,
        probe: &DeviceBuffer<C32>,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let d = &views.dims;
        let s = d.stack(&frames)?;
        let n = d.n();
        check_len("psi", psi, s.nk as usize * n)?;
        check_len("probe", probe, d.nprobe as usize * n)?;
        sums.check(d.obj_len())?;
        let config = LaunchConfig {
            grid_dim: (
                d.obj[1].div_ceil(OBJ_TILE),
                d.obj[0].div_ceil(OBJ_TILE),
                d.nobj,
            ),
            block_dim: (OBJ_TILE, OBJ_TILE, 1),
            shared_mem_bytes: 0,
        };
        let reads = [
            psi.storage(),
            probe.storage(),
            views.addr.storage(),
            views.tile_start.storage(),
            views.tile_views.storage(),
        ];
        self.launch(stream, &reads, &sums.storages(), || {
            // SAFETY: lengths checked above; the tile lists were built from these views.
            unsafe {
                self.module.psi_to_obj(
                    stream,
                    config,
                    dptr(psi),
                    dptr(probe),
                    dptr(&views.addr),
                    dptr(&views.tile_start),
                    dptr(&views.tile_views),
                    sums.ptrs(),
                    d.geom(&s),
                )
            }
        })
    }

    /// Adds the stack's probe-update sums (PyNX `Psi2Probe`) to `sums` `[nprobe, ny, nx]`:
    /// `num += conj(O_v exp(i quad r^2)) psi_v` and `den += |O_v|^2` over the views of each
    /// probe mode, in ascending view order (stack-split invariant). Zero `sums` before the
    /// first stack.
    pub fn psi_to_probe(
        &self,
        stream: &CudaStream,
        views: &Views,
        frames: Range<u32>,
        psi: &DeviceBuffer<C32>,
        obj: &DeviceBuffer<C32>,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let d = &views.dims;
        let s = d.stack(&frames)?;
        let n = d.n();
        check_len("psi", psi, s.nk as usize * n)?;
        check_len("obj", obj, d.obj_len())?;
        sums.check(d.nprobe as usize * n)?;
        let config = grid_1d(d.nprobe as usize * n)?;
        let reads = [psi.storage(), obj.storage(), views.addr.storage()];
        self.launch(stream, &reads, &sums.storages(), || {
            // SAFETY: lengths checked above; `Views::new` checked every view.
            unsafe {
                self.module.psi_to_probe(
                    stream,
                    config,
                    dptr(psi),
                    dptr(obj),
                    dptr(&views.addr),
                    sums.ptrs(),
                    d.geom(&s),
                )
            }
        })
    }

    /// Applies accumulated sums to the first `n` elements of `x` (PyNX `ObjNorm`):
    /// `x = (num + reg x) / max(den + reg, 1e-12)` with `reg = inertia * max(den[..n])`.
    pub fn update_from_sums(
        &self,
        stream: &CudaStream,
        x: &mut DeviceBuffer<C32>,
        sums: &UpdateSums,
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
        let max_bits = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        let UpdateSums { num, den } = sums;
        self.launch(stream, &[den.storage()], &[max_bits.storage()], || {
            // SAFETY: `den` holds `n` floats, `max_bits` one zeroed u32.
            unsafe {
                self.module
                    .max_nonneg_f32(stream, max_config, dptr(den), n, dptr(&max_bits))
            }
        })?;
        let reads = [num.storage(), den.storage(), max_bits.storage()];
        self.launch(stream, &reads, &[x.storage()], || {
            // SAFETY: all lengths checked above.
            unsafe {
                self.module.normalize_update(
                    stream,
                    config,
                    dptr(x),
                    dptr(num),
                    dptr(den),
                    n,
                    dptr(&max_bits),
                    inertia,
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
        stream: &CudaStream,
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
        self.launch(stream, &[], &[psi.storage(), po.storage()], || {
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
        stream: &CudaStream,
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
        self.launch(stream, &[x.storage()], &[y.storage()], || {
            // SAFETY: both hold `n` values.
            unsafe { self.module.add_c32(stream, config, dptr(y), dptr(x), n) }
        })
    }
}

/// Numerator and denominator of an object or probe update (PyNX `obj_new`, `obj_norm`),
/// element `i` of `den` belonging to element `i` of `num`.
pub struct UpdateSums {
    pub num: DeviceBuffer<C32>,
    pub den: DeviceBuffer<f32>,
}

impl UpdateSums {
    pub fn zeroed(stream: &CudaStream, len: usize) -> Result<Self, Error> {
        Ok(Self {
            num: DeviceBuffer::zeroed(stream, len)?,
            den: DeviceBuffer::zeroed(stream, len)?,
        })
    }

    /// Zeroes both on `stream`.
    pub fn zero(&mut self, stream: &CudaStream) -> Result<(), Error> {
        self.num.zero_async(stream)?;
        self.den.zero_async(stream)?;
        Ok(())
    }

    fn check(&self, need: usize) -> Result<(), Error> {
        check_len("sums.num", &self.num, need)?;
        check_len("sums.den", &self.den, need)
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
