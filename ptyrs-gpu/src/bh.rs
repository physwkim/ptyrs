//! Checked launches of the BH kernels (`ptyrs_kernels::bh`), pty-chi's bilinear-Hessian step
//! on the amplitude cost (ptyrs-engine `bh`).
//!
//! # One BH batch in these calls
//!
//! A batch is a run of stacks whose table entries hold epoch positions; [`BhScan::order`]
//! maps a position to its scan frame, and the views are in epoch order. With zeroed update
//! sums, per stack: [`Kernels::obj_probe_to_psi`] into its `psi`, forward FFT,
//! [`BhKernels::residual`] into its `r`, inverse FFT of `r`, [`Kernels::psi_to_obj`] and
//! [`Kernels::psi_to_probe`] of `r`: the sums' `num` is the gradient `g`. Then, with `u = g`
//! scaled by `ρ²` on the probe and for CG `v` the last direction, [`BhKernels::dots`]; per
//! stack [`BhKernels::far`], forward FFTs of `lu` (and `lv`) and [`BhKernels::hess_frames`];
//! [`BhKernels::step`], then [`BhKernels::update`] of each array, which consumes the sums.
//! The step never leaves the device.

use std::ops::Range;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{DeviceBuffer, LaunchConfig, RawDeviceBuffer};
pub use ptyrs_kernels::bh::{BH_HESS, BH_STEP};
use ptyrs_kernels::bh::{BhDirs, BhFields, BhObs, bh_kernels};
use ptyrs_kernels::grad::{CG_BLOCKS, GAMMA_CHUNK, ObjProbePtrs};
use ptyrs_kernels::pixel::{PIXEL_THREADS, SumPtrs};

use crate::ops::grid_1d;
use crate::{
    C32, Error, Kernels, StackCtl, StackRef, UpdateSums, ViewAddr, ViewDims, Views, check_len, dptr,
};

/// The BH kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct BhKernels {
    base: Kernels,
    module: bh_kernels::LoadedModule,
}

/// The scan in epoch order: the views at each epoch position and, through `order`
/// (`nframe` scan frame ids, one per position), the counts in the scan's own order.
#[derive(Clone, Copy)]
pub struct BhScan<'a> {
    pub views: &'a Views,
    pub counts: &'a DeviceBuffer<u16>,
    pub order: &'a DeviceBuffer<u32>,
}

impl BhScan<'_> {
    fn check(&self) -> Result<&ViewDims, Error> {
        let dims = self.views.dims();
        check_len("counts", self.counts, dims.nframe as usize * dims.n())?;
        check_len("order", self.order, dims.nframe as usize)?;
        Ok(dims)
    }

    fn obs(&self) -> BhObs {
        BhObs {
            counts: dptr(self.counts),
            order: dptr(self.order),
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
/// forward-FFT'd derivatives along `u` and `v` (`lv` unread without `v`).
pub struct BhFieldsRef<'a> {
    pub psi: &'a DeviceBuffer<C32>,
    pub r: &'a DeviceBuffer<C32>,
    pub lu: &'a DeviceBuffer<C32>,
    pub lv: &'a DeviceBuffer<C32>,
}

/// One batch of an epoch: its epoch positions, the share of the epoch's frames its step
/// stands for (`α` is scaled by it), and its record in the step records.
#[derive(Clone, Debug)]
pub struct BhBatch {
    pub frames: Range<u32>,
    pub frac: f32,
    pub record: u32,
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

/// Floats of the partial sums of one step over a scan of `dims`: the dot products' `2 *
/// CG_BLOCKS` pairs (object, then probe), then a [`BH_HESS`] triple per chunk of
/// [`GAMMA_CHUNK`] pixels of each epoch position.
pub fn bh_parts(dims: &ViewDims) -> usize {
    HESS_FIRST + BH_HESS as usize * dims.nframe as usize * chunks(dims.n())
}

/// The first float of the Hessian triples in [`bh_parts`].
const HESS_FIRST: usize = 2 * 2 * CG_BLOCKS as usize;

/// Chunks of [`GAMMA_CHUNK`] pixels per frame of `n` pixels.
fn chunks(n: usize) -> usize {
    n.div_ceil(GAMMA_CHUNK as usize)
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
            scan.order.storage(),
            psi.storage(),
            stack.storage(),
        ];
        self.base.launch(ctx, &reads, &[r.storage()], |stream| {
            // SAFETY: lengths checked above; `order` holds frame ids below `nframe`.
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

    /// The dot products of the step along `d`, `⟨u, Pu⟩` and (CG) `⟨u, v⟩`, as block partials
    /// into the first `2 * CG_BLOCKS` pairs of `parts`, object then probe.
    pub fn dots(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        d: &BhDisp<'_>,
        parts: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        d.check(dims)?;
        check_len("parts", parts, bh_parts(dims))?;
        let config = LaunchConfig {
            grid_dim: (CG_BLOCKS, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let scales = [1.0, d.u_scale];
        for (i, n) in lens(dims).into_iter().enumerate() {
            let (g, prev) = (d.u[i], d.v.map(|v| v[i]));
            let out = dptr::<f32, f32>(parts).wrapping_add(2 * i * CG_BLOCKS as usize);
            let reads: Vec<_> = [Some(g), prev]
                .into_iter()
                .flatten()
                .map(|b| b.storage())
                .collect();
            self.base
                .launch(ctx, &reads, &[parts.storage()], |stream| {
                    // SAFETY: lengths checked above; CG_BLOCKS blocks of PIXEL_THREADS write the
                    // array's CG_BLOCKS pairs; `prev` is read only when given.
                    unsafe {
                        self.module.bh_dots(
                            stream,
                            config,
                            dptr(g),
                            dptr(prev.unwrap_or(g)),
                            n,
                            scales[i],
                            prev.is_some().into(),
                            out,
                        )
                    }
                })?;
        }
        Ok(())
    }

    /// The stack's far-field derivatives before their forward FFT, in
    /// [`Kernels::obj_probe_to_psi`]'s layout, at `x = [O, P]`: `l[0] = q (u_P O + P u_O)` and,
    /// with `d.v`, `l[1]` likewise along `v` (else `l[1]` is left as it is).
    pub fn far(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        x: [&DeviceBuffer<C32>; 2],
        d: &BhDisp<'_>,
        l: [&mut DeviceBuffer<C32>; 2],
    ) -> Result<(), Error> {
        let dims = views.dims();
        let g = dims.geom(stack)?;
        let len = g.nf_max as usize * dims.nmode as usize * dims.n();
        for (b, n, what) in [(x[0], lens(dims)[0], "obj"), (x[1], lens(dims)[1], "probe")] {
            check_len(what, b, n as usize)?;
        }
        d.check(dims)?;
        let [lu, lv] = l;
        check_len("lu", lu, len)?;
        check_len("lv", lv, len)?;
        let config = grid_1d(len)?;
        let mut reads = vec![views.addr().storage(), stack.storage()];
        reads.extend(x.iter().map(|b| b.storage()));
        reads.extend(d.storages());
        let writes = [lu.storage(), lv.storage()];
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
                    dptr(lu),
                    dptr(lv),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The Hessian triples `(H(u, u), H(u, v), H(v, v))` along `d` of the stack's frames (the
    /// last two 0 without `d.v`), each over a chunk of [`GAMMA_CHUNK`] pixels in a fixed-order
    /// block sum, into the triples of [`bh_parts`] at the frames' epoch positions; other
    /// positions' triples are left as they are.
    pub fn hess_frames(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        stack: StackRef<'_>,
        f: &BhFieldsRef<'_>,
        d: &BhDisp<'_>,
        parts: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let dims = scan.check()?;
        let g = dims.geom(stack)?;
        let n = dims.n();
        let len = g.nf_max as usize * dims.nmode as usize * n;
        for (what, b) in [("psi", f.psi), ("r", f.r), ("lu", f.lu), ("lv", f.lv)] {
            check_len(what, b, len)?;
        }
        d.check(dims)?;
        check_len("parts", parts, bh_parts(dims))?;
        if g.nf_max > u16::MAX as u32 {
            return Err(Error::Shape(format!("{} frames exceed grid y", g.nf_max)));
        }
        let nchunk = u32::try_from(chunks(n))
            .map_err(|_| Error::Shape(format!("{n} pixels exceed grid x")))?;
        let config = LaunchConfig {
            grid_dim: (nchunk, g.nf_max, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut reads = vec![
            scan.views.addr().storage(),
            scan.counts.storage(),
            scan.order.storage(),
            f.psi.storage(),
            f.r.storage(),
            f.lu.storage(),
            f.lv.storage(),
            stack.storage(),
        ];
        reads.extend(d.storages());
        let out = dptr::<f32, f32>(parts).wrapping_add(HESS_FIRST);
        self.base.launch(ctx, &reads, &[parts.storage()], |stream| {
            let fields = BhFields {
                psi: dptr(f.psi),
                r: dptr(f.r),
                lu: dptr(f.lu),
                lv: dptr(f.lv),
            };
            // SAFETY: lengths checked above; one block of PIXEL_THREADS per chunk and frame;
            // `order` holds frame ids below `nframe`.
            unsafe {
                self.module.bh_hess_frames(
                    stream,
                    config,
                    fields,
                    d.ptrs(),
                    dptr::<ViewAddr, ViewAddr>(scan.views.addr()),
                    scan.obs(),
                    out,
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The step of `batch` along `d` into its record of `steps` (`(β, α, top, bottom)`,
    /// [`BH_STEP`] floats each) from the partials [`Self::dots`] and [`Self::hess_frames`]
    /// left in `parts`: `β = H(u, v) / H(v, v)` with `d.v`, else 0; `top = ⟨u, Pu⟩ − β ⟨u,
    /// v⟩ = −⟨g, D⟩` and `bottom = H(D, D)` for `D = β v − Pu`; `α = top / bottom * frac`.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        parts: &DeviceBuffer<f32>,
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
        check_len("parts", parts, bh_parts(dims))?;
        check_len("steps", steps, step_len(batch.record))?;
        let nchunk = chunks(dims.n());
        let nhess = u32::try_from(frames.len() * nchunk)
            .map_err(|_| Error::Shape(format!("batch {frames:?} exceeds u32 triples")))?;
        let hess = HESS_FIRST + BH_HESS as usize * frames.start as usize * nchunk;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let out = dptr::<f32, f32>(steps).wrapping_add(step_len(batch.record) - BH_STEP as usize);
        self.base
            .launch(ctx, &[parts.storage()], &[steps.storage()], |stream| {
                // SAFETY: lengths checked above; the batch's `nhess` triples lie in `parts`
                // (`frames.end <= nframe`); one PIXEL_THREADS block.
                unsafe {
                    self.module.bh_step(
                        stream,
                        config,
                        dptr(parts),
                        2 * CG_BLOCKS,
                        dptr::<f32, f32>(parts).wrapping_add(hess),
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
}

/// Floats of step records `0..=record`.
fn step_len(record: u32) -> usize {
    BH_STEP as usize * (record as usize + 1)
}
