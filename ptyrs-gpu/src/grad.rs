//! Checked launches of the Poisson ML conjugate-gradient kernels (`ptyrs_kernels::grad`).
//!
//! # One CG cycle (PyNX `ML`) in these calls
//!
//! Per updated array (object, probe): the gradient `g` accumulates into zeroed sums, `h`
//! holds the last cycle's gradient (zeros before the first) and `d` the direction (zeros).
//! [`GradKernels::beta_parts`] of every array into its own `CG_BLOCKS` pairs of one parts
//! buffer, then [`GradKernels::sum_parts`] of them all gives the Polak–Ribière pair; it is
//! `(sum |g|^2, 0)` on the first cycle, so `β = 0`. [`GradKernels::direction`] per array,
//! then the line-search pair `γ`, then [`GradKernels::step`] per array. The coefficients
//! never leave the device, so a cycle is one host-sync-free sequence.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{DeviceBuffer, LaunchConfig};
pub use ptyrs_kernels::grad::CG_BLOCKS;
use ptyrs_kernels::grad::grad_kernels;
use ptyrs_kernels::pixel::{PIXEL_THREADS, SumPtrs};

use crate::{C32, Error, Kernels, UpdateSums, check_len, dptr};

/// The CG kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct GradKernels {
    base: Kernels,
    module: grad_kernels::LoadedModule,
}

impl GradKernels {
    /// Binds the CG kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = grad_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// One array's Polak–Ribière partial sums `(sum Re(conj(g) (g - h)), sum |h|^2)` into
    /// the `CG_BLOCKS` pairs of `parts` from pair `first`, then `h = g`.
    pub fn beta_parts(
        &self,
        ctx: &ExecutionContext,
        g: &DeviceBuffer<C32>,
        h: &mut DeviceBuffer<C32>,
        n: u32,
        parts: &mut DeviceBuffer<f32>,
        first: u32,
    ) -> Result<(), Error> {
        check_len("g", g, n as usize)?;
        check_len("h", h, n as usize)?;
        check_len("parts", parts, 2 * (first as usize + CG_BLOCKS as usize))?;
        let config = LaunchConfig {
            grid_dim: (CG_BLOCKS, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let out = dptr::<f32, f32>(parts).wrapping_add(2 * first as usize);
        let writes = [h.storage(), parts.storage()];
        self.base.launch(ctx, &[g.storage()], &writes, |stream| {
            // SAFETY: lengths checked above; CG_BLOCKS blocks of PIXEL_THREADS.
            unsafe {
                self.module
                    .cg_beta_parts(stream, config, dptr(g), dptr(h), n, out)
            }
        })
    }

    /// `pair = sum` of the first `npart` pairs of `parts`, in a fixed order.
    pub fn sum_parts(
        &self,
        ctx: &ExecutionContext,
        parts: &DeviceBuffer<f32>,
        npart: u32,
        pair: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        check_len("parts", parts, 2 * npart as usize)?;
        check_len("pair", pair, 2)?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.base
            .launch(ctx, &[parts.storage()], &[pair.storage()], |stream| {
                // SAFETY: lengths checked above; one PIXEL_THREADS block.
                unsafe {
                    self.module
                        .cg_sum_parts(stream, config, dptr(parts), npart, dptr(pair))
                }
            })
    }

    /// `d = β d - g` with `β = beta[0] / beta[1]`, or 0 where that is negative or not finite;
    /// consumes the gradient sums `g` (zero after).
    pub fn direction(
        &self,
        ctx: &ExecutionContext,
        d: &mut DeviceBuffer<C32>,
        g: &mut UpdateSums,
        n: u32,
        beta: &DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        check_len("d", d, n as usize)?;
        check_len("g.num", &g.num, n as usize)?;
        check_len("g.den", &g.den, n as usize)?;
        check_len("beta", beta, 2)?;
        if n == 0 {
            return Ok(());
        }
        let config = LaunchConfig::for_num_elems(n);
        let writes = [d.storage(), g.num.storage(), g.den.storage()];
        self.base.launch(ctx, &[beta.storage()], &writes, |stream| {
            let g = SumPtrs {
                num: dptr(&g.num),
                den: dptr(&g.den),
            };
            // SAFETY: lengths checked above.
            unsafe {
                self.module
                    .cg_direction(stream, config, dptr(d), g, n, dptr(beta))
            }
        })
    }

    /// With `γ = gamma[0] / gamma[1]`: `x += γ d` if `γ` is finite, else `x` stays and
    /// `d = -h`, `h` being this cycle's gradient as [`Self::beta_parts`] leaves it.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        x: &mut DeviceBuffer<C32>,
        d: &mut DeviceBuffer<C32>,
        h: &DeviceBuffer<C32>,
        n: u32,
        gamma: &DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        check_len("x", x, n as usize)?;
        check_len("d", d, n as usize)?;
        check_len("h", h, n as usize)?;
        check_len("gamma", gamma, 2)?;
        if n == 0 {
            return Ok(());
        }
        let config = LaunchConfig::for_num_elems(n);
        let reads = [h.storage(), gamma.storage()];
        let writes = [x.storage(), d.storage()];
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module
                    .cg_step(stream, config, dptr(x), dptr(d), dptr(h), n, dptr(gamma))
            }
        })
    }
}
