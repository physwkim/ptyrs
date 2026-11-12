//! Checked launches of the rPIE kernels (`ptyrs_kernels::rpie`).
//!
//! # pty-chi `RPIEReconstructor` in these calls (per batch of frames, one object mode)
//!
//! `obj_probe_to_psi` into `psi`; forward FFT of a copy `dlt`; [`RpieKernels::amplitude`];
//! inverse FFT; `dlt -= psi` (`scale_c32(psi, -1)`, `add_c32`), so `dlt` is pty-chi's
//! `Δ`. Object step: [`RpieKernels::power_max`] of the probe (all modes summed) and
//! [`RpieKernels::weight`] of the probe by itself give `P~ = P / ((1 - a) |P|^2 + a max)`,
//! and `psi_to_obj(dlt, P~)` into the (consumed, so zero) sums yields the object step's
//! direction in `num`. Probe step: `obj_probe_to_psi` with a probe of ones gives the windows
//! `O_v q`; [`RpieKernels::power_max`] per plane and [`RpieKernels::weight_inplace`] of `dlt`
//! by them, then `psi_to_probe(dlt, O)`. Both steps read the batch-start arrays: apply
//! [`RpieKernels::step`], which consumes the sums, only after both sums are taken.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{DeviceBuffer, LaunchConfig};
use ptyrs_kernels::pixel::{PIXEL_THREADS, SumPtrs};
use ptyrs_kernels::rpie::{WeightGeom, rpie_kernels};

pub use ptyrs_kernels::rpie::POWER_CHUNK;

use crate::{C32, Error, Kernels, StackCtl, StackRef, UpdateSums, ViewDims, check_len, dptr};

/// The rPIE kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct RpieKernels {
    base: Kernels,
    module: rpie_kernels::LoadedModule,
}

/// A 1D launch over `total` elements; errors if it does not fit `u32`.
fn elems(total: usize) -> Result<(u32, LaunchConfig), Error> {
    let n = u32::try_from(total)
        .map_err(|_| Error::Shape(format!("{total} elements exceed a 1D launch")))?;
    Ok((n, LaunchConfig::for_num_elems(n)))
}

impl RpieKernels {
    /// Binds the rPIE kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = rpie_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// pty-chi's modulus constraint on the forward-FFT'd stack: each pixel's modes are
    /// scaled by `sqrt(obs + 1e-7) / (sqrt(icalc) + 1e-7) / n`, `icalc = sum_m |Psi_m|^2 /
    /// n`, or by `1 / n` where the count is `MASKED`; the unnormalized inverse FFT then
    /// yields pty-chi's corrected exit wave. `counts` holds the whole scan.
    pub fn amplitude(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        counts: &DeviceBuffer<u16>,
        psi: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        let (n, nf_max) = (g.ny as usize * g.nx as usize, g.nf_max as usize);
        check_len("counts", counts, dims.nframe as usize * n)?;
        check_len("psi", psi, nf_max * dims.nmode as usize * n)?;
        let (_, config) = elems(nf_max * n)?;
        let reads = [counts.storage(), stack.storage()];
        self.base.launch(ctx, &reads, &[psi.storage()], |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.rpie_amplitude(
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

    /// `out[b] = max_i sum_{m < nsum} |x[b * nsum + m, i]|^2` over the groups of `spec`,
    /// through the per-chunk maxes in `chunks` ([`PowerMax::chunks`] floats).
    pub fn power_max(
        &self,
        ctx: &ExecutionContext,
        x: &DeviceBuffer<C32>,
        spec: PowerMax,
        chunks: &mut DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let PowerMax { ngroup, nsum, n } = spec;
        if ngroup == 0 || nsum == 0 || n == 0 || ngroup > u16::MAX as u32 {
            return Err(Error::Shape(format!("empty or too many groups {spec:?}")));
        }
        check_len("x", x, ngroup as usize * nsum as usize * n as usize)?;
        check_len("chunks", chunks, spec.chunks())?;
        check_len("out", out, ngroup as usize)?;
        let nchunk = n.div_ceil(POWER_CHUNK);
        let config = LaunchConfig {
            grid_dim: (nchunk, ngroup, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.base
            .launch(ctx, &[x.storage()], &[chunks.storage()], |stream| {
                // SAFETY: lengths checked above; one PIXEL_THREADS block per (chunk, group).
                unsafe {
                    self.module
                        .rpie_power_max(stream, config, dptr(x), nsum, n, dptr(chunks))
                }
            })?;
        let config = LaunchConfig::for_num_elems(ngroup);
        self.base
            .launch(ctx, &[chunks.storage()], &[out.storage()], |stream| {
                // SAFETY: lengths checked above; one thread per group.
                unsafe {
                    self.module.rpie_max_chunks(
                        stream,
                        config,
                        dptr(chunks),
                        nchunk,
                        ngroup,
                        dptr(out),
                    )
                }
            })
    }

    /// `out[k, i] = x[k, i] / ((1 - alpha) |w[k, i]|^2 + alpha mx[k'])` over the planes of
    /// `spec`.
    pub fn weight(
        &self,
        ctx: &ExecutionContext,
        x: &DeviceBuffer<C32>,
        w: &DeviceBuffer<C32>,
        mx: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<C32>,
        spec: Weight,
    ) -> Result<(), Error> {
        let total = spec.check(w, mx)?;
        check_len("x", x, total)?;
        check_len("out", out, total)?;
        let reads = [x.storage(), w.storage(), mx.storage()];
        let ptrs = [dptr(x), dptr(w), dptr(mx), dptr(out)];
        self.launch_weight(ctx, spec, ptrs, &reads, &[out.storage()])
    }

    /// [`Self::weight`] in place on `x`.
    pub fn weight_inplace(
        &self,
        ctx: &ExecutionContext,
        x: &mut DeviceBuffer<C32>,
        w: &DeviceBuffer<C32>,
        mx: &DeviceBuffer<f32>,
        spec: Weight,
    ) -> Result<(), Error> {
        let total = spec.check(w, mx)?;
        check_len("x", x, total)?;
        let reads = [w.storage(), mx.storage()];
        let ptrs = [dptr(x), dptr(w), dptr(mx), dptr(x)];
        self.launch_weight(ctx, spec, ptrs, &reads, &[x.storage()])
    }

    /// The rPIE step of the `n` values of `x` from `sums`, consuming them: `x += s * num`,
    /// then `num` and `den` are zero.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        x: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        n: u32,
        s: f32,
    ) -> Result<(), Error> {
        check_len("x", x, n as usize)?;
        check_len("sums.num", &sums.num, n as usize)?;
        check_len("sums.den", &sums.den, n as usize)?;
        if n == 0 {
            return Ok(());
        }
        let (_, config) = elems(n as usize)?;
        let writes = [x.storage(), sums.num.storage(), sums.den.storage()];
        self.base.launch(ctx, &[], &writes, |stream| {
            let sums = SumPtrs {
                num: dptr(&sums.num),
                den: dptr(&sums.den),
            };
            // SAFETY: lengths checked above.
            unsafe { self.module.rpie_step(stream, config, dptr(x), sums, n, s) }
        })
    }

    /// Launches `rpie_weight` on `[x, w, mx, out]`, which [`Weight::check`] and the caller
    /// checked; `out` may be `x`, since each thread reads and writes only its own element.
    fn launch_weight(
        &self,
        ctx: &ExecutionContext,
        spec: Weight,
        [x, w, mx, out]: [*mut f32; 4],
        reads: &[&cuda_core::RawDeviceBuffer],
        writes: &[&cuda_core::RawDeviceBuffer],
    ) -> Result<(), Error> {
        let (_, config) = elems(spec.nk as usize * spec.n as usize)?;
        self.base.launch(ctx, reads, writes, |stream| {
            // SAFETY: the buffers behind the pointers hold what the callers checked.
            unsafe {
                self.module.rpie_weight(
                    stream,
                    config,
                    x,
                    w,
                    mx,
                    out,
                    WeightGeom {
                        nk: spec.nk,
                        n: spec.n,
                        mstride: spec.per_plane as u32,
                        alpha: spec.alpha,
                    },
                )
            }
        })
    }
}

/// Groups of [`RpieKernels::power_max`]: `ngroup` groups of `nsum` planes of `n` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerMax {
    pub ngroup: u32,
    pub nsum: u32,
    pub n: u32,
}

impl PowerMax {
    /// The floats of the `chunks` scratch: one per group and `POWER_CHUNK` values of a plane.
    pub fn chunks(&self) -> usize {
        self.ngroup as usize * self.n.div_ceil(POWER_CHUNK) as usize
    }
}

/// Planes and regulariser of [`RpieKernels::weight`]: `nk` planes of `n` values, divided by
/// `(1 - alpha) |w|^2 + alpha mx[k']` with `k' = k` if `per_plane` else 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weight {
    pub nk: u32,
    pub n: u32,
    pub per_plane: bool,
    pub alpha: f32,
}

impl Weight {
    /// Elements per complex buffer, after checking `w` and `mx`.
    fn check(&self, w: &DeviceBuffer<C32>, mx: &DeviceBuffer<f32>) -> Result<usize, Error> {
        if self.nk == 0 || self.n == 0 {
            return Err(Error::Shape(format!("empty weight {self:?}")));
        }
        let (total, _) = elems(self.nk as usize * self.n as usize)?;
        check_len("w", w, total as usize)?;
        check_len("mx", mx, if self.per_plane { self.nk as usize } else { 1 })?;
        Ok(total as usize)
    }
}
