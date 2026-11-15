//! Checked launches of the full-object sampling kernels (`ptyrs_kernels::sampling`), the
//! device side of [`crate::engine::Sampling::FullObject`].
//!
//! A view reads its object plane at its corner `(oy + fy, ox + fx)` plus its frame's offset
//! `(dy, dx)` (`offsets` `[nframe, 2]`, frame order of the views) by Fourier interpolation of
//! the whole `[nyo, nxo]` plane: `W O = crop(DFT⁻¹(DFT(O)·exp(+2πi f·t)))`, `f` the plane's
//! `fftfreq`, a sample past the plane's edge wrapping. An object is `[nslice, nobj, nyo,
//! nxo]`, and its [`ObjectSampler::spectra`] are taken once for the windows of every stack.
//!
//! The views of a frame on one object mode share one position, a group.
//! [`ObjectSampler::windows`] writes a block `[modes, nf_max, ny, nx]` of a stack's windows,
//! natural layout, `modes` every object mode ([`Modes::All`]) or the first: plane
//! `o * nf_max + fl` holds frame `f0 + fl`'s window on object mode `o`. A block is so an
//! object `[modes, nf_max * ny, nx]` of [`window_views`], which view `kl` reads as it reads
//! the object, at object mode `v.obj`, corner `(kl / nmode * ny, 0)` and no fraction. The
//! blocks of one buffer hold the slices, or the derivatives over `y` and `x`.
//! [`ObjectSampler::place`] adds `W^H` of patches shaped as an all-modes block, each group's
//! views summed as a scatter onto the window views leaves them, to one slice of the object:
//! the exact adjoint of the windows.
//!
//! Each pass transforms `chunk` groups in a scratch of `chunk` object planes; a stack takes
//! `ceil(nf_max * modes / chunk)` passes whatever its `nf`, so the launches can be captured
//! once for every stack.
//!
//! [`ObjectSampler::solve`] updates an object from its placed sums by the normal equations
//! of its windows over the whole scan, the passes of one stack of every frame weighting each
//! window by the probe's power in place of a window buffer.

use std::ops::Range;
use std::slice;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use ptyrs_kernels::pixel::{PIXEL_THREADS, StackGeom};
use ptyrs_kernels::sampling::{PCG_PAP, PCG_RZ, PCG_SCALARS, Pcg, SamplePass, sampling_kernels};

use crate::fft::Plan2d;
use crate::gpu::{
    C32, CG_BLOCKS, Error, Kernels, StackCtl, StackRef, StackTable, UpdateSums, ViewAddr, ViewDims,
    Views, check_len, dptr,
};

/// `buf`'s device address `at` complex values in, as the kernels take it.
fn dptr_at(buf: &DeviceBuffer<C32>, at: usize) -> *mut f32 {
    dptr::<C32, f32>(buf).wrapping_add(2 * at)
}

/// A 1D launch over `total` elements; errors if it does not fit `u32`.
fn elems(total: usize) -> Result<(u32, LaunchConfig), Error> {
    let n = u32::try_from(total)
        .map_err(|_| Error::Shape(format!("{total} elements exceed a 1D launch")))?;
    Ok((n, LaunchConfig::for_num_elems(n)))
}

/// `blocks` blocks of `threads`.
fn blocks(blocks: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The views over a stack's windows where each view samples the object at its offset: an
/// all-modes block of [`ObjectSampler::windows`], plane `o * nf_max + fl` frame `f0 + fl`'s
/// window on object mode `o`, is these views' object `[nobj, nf_max * ny, nx]`, epoch
/// position `f0 + fl` at row `fl * ny`. `stacks` are the stack table's, consecutive from
/// position 0 to `nframe`; every stack's views share the rows, and a stack launch reads its
/// own.
pub fn window_views(
    stream: &CudaStream,
    dims: &ViewDims,
    stacks: &[Range<u32>],
) -> Result<Views, Error> {
    dims.check()?;
    let mut next = 0;
    for r in stacks {
        if r.start != next || r.is_empty() {
            return Err(Error::Shape(format!(
                "stack {r:?} does not follow position {next}"
            )));
        }
        next = r.end;
    }
    if next != dims.nframe {
        return Err(Error::Shape(format!(
            "stacks end at {next}, the scan at {}",
            dims.nframe
        )));
    }
    let nf_max = stacks.iter().map(ExactSizeIterator::len).max().unwrap_or(0);
    let [ny, nx] = dims.frame;
    let rows = u32::try_from(nf_max * ny as usize)
        .map_err(|_| Error::Shape(format!("{nf_max} windows of {ny} rows exceed u32")))?;
    let addr: Vec<ViewAddr> = stacks
        .iter()
        .flat_map(|r| 0..r.end - r.start)
        .flat_map(|fl| {
            (0..dims.nmode).map(move |m| ViewAddr {
                oy: fl * ny,
                ox: 0,
                fy: 0.0,
                fx: 0.0,
                probe: m % dims.nprobe,
                obj: m / dims.nprobe,
            })
        })
        .collect();
    let windows = ViewDims {
        obj: [rows, nx],
        ..*dims
    };
    Views::new(stream, windows, &addr)
}

/// Where a stack's views read the object: the scan's views, the stack and every frame's
/// offset `(dy, dx)`, `[nframe, 2]`.
#[derive(Clone, Copy)]
pub struct StackPoints<'a> {
    pub views: &'a Views,
    pub stack: StackRef<'a>,
    pub offsets: &'a DeviceBuffer<f32>,
}

/// What [`ObjectSampler::solve`]'s `AᴴA` reads: the scan's views, every frame's offset
/// `(dy, dx)`, `[nframe, 2]`, and the probe `[nprobe, ny, nx]`.
#[derive(Clone, Copy)]
pub struct Normal<'a> {
    pub views: &'a Views,
    pub offsets: &'a DeviceBuffer<f32>,
    pub probe: &'a DeviceBuffer<C32>,
}

/// The object modes of a frame that [`ObjectSampler::windows`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modes {
    /// Every object mode, `nobj` planes per frame.
    All,
    /// Object mode 0, the frame's first view, one plane per frame.
    First,
}

/// One block of windows [`ObjectSampler::windows`] writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowBlock {
    /// The object slice read.
    pub slice: u32,
    /// The derivative's order over the position's `y` and `x`, `[0, 0]` for the windows.
    pub order: [u32; 2],
    pub modes: Modes,
    /// The block of the output, `[.., modes, nf_max, ny, nx]`, written.
    pub block: u32,
}

/// Full-object sampling of the views of one [`ViewDims`]: the cuFFT plans over its object
/// planes and the scratch they work in.
pub struct ObjectSampler {
    base: Kernels,
    module: sampling_kernels::LoadedModule,
    dims: ViewDims,
    nslice: u32,
    /// `nslice * nobj` object planes.
    slices: Plan2d,
    /// `nobj` object planes, those of `acc`.
    planes: Plan2d,
    /// `chunk` object planes, those of `scratch`.
    pass: Plan2d,
    scratch: DeviceBuffer<C32>,
    /// The placed spectra, then the illumination, `[nobj, nyo, nxo]`.
    acc: DeviceBuffer<C32>,
    chunk: u32,
    /// The scan as one stack, which [`Self::solve`]'s `AᴴA` passes over.
    whole: StackTable,
    /// [`Self::solve`]'s direction, `[nobj, nyo, nxo]`.
    dir: DeviceBuffer<C32>,
    /// [`Self::solve`]'s block sums, `CG_BLOCKS`.
    parts: DeviceBuffer<f32>,
    /// [`Self::solve`]'s scalars, [`Pcg`]'s `s`.
    scalars: DeviceBuffer<f32>,
}

/// What [`ObjectSampler::spread`] sums over the frames.
#[derive(Clone, Copy)]
enum Spread {
    /// Each frame's conjugate sampling phase: `W^H` of the probe's power.
    Illumination,
    /// The spectrum of each frame's squared sampling kernel: `AᴴA`'s diagonal.
    Diagonal,
}

impl ObjectSampler {
    /// Plans and allocates for objects of `nslice` slices of `dims` on `stream`,
    /// interpolating at most `chunk` groups per pass: the scratch holds `chunk` object planes.
    pub fn new(
        kernels: &Kernels,
        stream: &CudaStream,
        dims: ViewDims,
        nslice: u32,
        chunk: u32,
    ) -> Result<Self, Error> {
        dims.check()?;
        if nslice == 0 || chunk == 0 {
            return Err(Error::Shape(format!("{nslice} slices, passes of {chunk}")));
        }
        let shape = dims.obj.map(|n| n as usize);
        let plane = shape[0] * shape[1];
        elems(chunk as usize * plane)?;
        elems(nslice as usize * dims.obj_len())?;
        let module = sampling_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
            dims,
            nslice,
            slices: Plan2d::with_shape(stream, shape, nslice as usize * dims.nobj as usize)?,
            planes: Plan2d::with_shape(stream, shape, dims.nobj as usize)?,
            pass: Plan2d::with_shape(stream, shape, chunk as usize)?,
            scratch: DeviceBuffer::zeroed(stream, chunk as usize * plane)?,
            acc: DeviceBuffer::zeroed(stream, dims.obj_len())?,
            chunk,
            whole: StackTable::new(stream, &dims, slice::from_ref(&(0..dims.nframe)))?,
            dir: DeviceBuffer::zeroed(stream, dims.obj_len())?,
            parts: DeviceBuffer::zeroed(stream, CG_BLOCKS as usize)?,
            scalars: DeviceBuffer::zeroed(stream, PCG_SCALARS)?,
        })
    }

    pub fn dims(&self) -> &ViewDims {
        &self.dims
    }

    pub fn nslice(&self) -> u32 {
        self.nslice
    }

    /// The bytes [`Self::windows`] holds across a chunk's shift, transform and crop whatever
    /// the stack's frames: the scratch, and the one slice of spectra every chunk's shift reads.
    pub fn windows_held(&self) -> usize {
        self.scratch_bytes() + self.dims.obj_len() * size_of::<C32>()
    }

    /// The bytes of the scratch, `chunk` object planes, which [`Self::place`] holds across a
    /// chunk's embed, transform and accumulate.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.len() * size_of::<C32>()
    }

    /// The bytes of the accumulator, `nobj` object planes, which [`Self::place`] holds from
    /// zeroing it through every chunk's accumulate to adding it onto the object.
    pub fn acc_bytes(&self) -> usize {
        self.acc.len() * size_of::<C32>()
    }

    /// The unnormalized spectra `DFT(obj)` of the object planes, `[nslice, nobj, nyo, nxo]`,
    /// which [`Self::windows`] reads.
    pub fn spectra(
        &mut self,
        ctx: &ExecutionContext,
        obj: &DeviceBuffer<C32>,
        spectra: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        check_len("obj", obj, self.object_len())?;
        check_len("spectra", spectra, self.object_len())?;
        Ok(self.slices.forward(ctx, obj, spectra)?)
    }

    /// `block` of the windows of the stack's groups at `at` on the object whose
    /// [`Self::spectra`] `spectra` holds, into `win` (see the module docs); its planes past
    /// the stack's frames, and the other blocks, are left as they are.
    pub fn windows(
        &mut self,
        ctx: &ExecutionContext,
        at: StackPoints<'_>,
        spectra: &DeviceBuffer<C32>,
        block: WindowBlock,
        win: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let StackPoints {
            views,
            stack,
            offsets,
        } = at;
        let g = self.geom(at)?;
        let modes = match block.modes {
            Modes::All => self.dims.nobj,
            Modes::First => 1,
        };
        let len = g.nf_max as usize * modes as usize * self.dims.n();
        check_len("spectra", spectra, self.object_len())?;
        check_len("win", win, (block.block as usize + 1) * len)?;
        let slice = self.slice(block.slice)?;
        let out = block.block as usize * len;
        let reads = [
            views.addr().storage(),
            offsets.storage(),
            spectra.storage(),
            stack.storage(),
        ];
        for mut s in self.passes(g, modes) {
            [s.ord_y, s.ord_x] = block.order;
            let (_, config) = elems(s.npass as usize * self.plane())?;
            self.base
                .launch(ctx, &reads, &[self.scratch.storage()], |stream| {
                    // SAFETY: lengths checked above; `Views::new` checked every view's modes.
                    unsafe {
                        self.module.sample_shift(
                            stream,
                            config,
                            dptr_at(spectra, slice),
                            dptr::<ViewAddr, ViewAddr>(views.addr()),
                            dptr(offsets),
                            dptr(&self.scratch),
                            stack.ctl() as *const StackCtl,
                            g,
                            s,
                        )
                    }
                })?;
            self.pass.inverse_inplace(ctx, &mut self.scratch)?;
            let (_, config) = elems(s.npass as usize * self.dims.n())?;
            let reads = [self.scratch.storage(), stack.storage()];
            self.base.launch(ctx, &reads, &[win.storage()], |stream| {
                // SAFETY: lengths checked above.
                unsafe {
                    self.module.sample_crop(
                        stream,
                        config,
                        dptr(&self.scratch),
                        dptr_at(win, out),
                        stack.ctl() as *const StackCtl,
                        g,
                        s,
                    )
                }
            })?;
        }
        Ok(())
    }

    /// Adds `W^H patches` to `slice` of `obj`, `[nslice, nobj, nyo, nxo]`: the patches of the
    /// stack's groups at `at`, `[nobj, nf_max, ny, nx]` natural layout, plane `o * nf_max + fl`
    /// frame `f0 + fl`'s on object mode `o`, placed back onto the object planes; the adjoint
    /// of [`Self::windows`] of every mode without derivative.
    pub fn place(
        &mut self,
        ctx: &ExecutionContext,
        at: StackPoints<'_>,
        patches: &DeviceBuffer<C32>,
        slice: u32,
        obj: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let StackPoints {
            views,
            stack,
            offsets,
        } = at;
        let g = self.geom(at)?;
        let len = self.dims.obj_len();
        check_len(
            "patches",
            patches,
            self.dims.nobj as usize * g.nf_max as usize * self.dims.n(),
        )?;
        check_len("obj", obj, self.object_len())?;
        let slice = self.slice(slice)?;
        crate::gpu::zero(ctx, &mut self.acc)?;
        for s in self.passes(g, self.dims.nobj) {
            let (_, config) = elems(s.npass as usize * self.plane())?;
            let reads = [patches.storage(), stack.storage()];
            self.base
                .launch(ctx, &reads, &[self.scratch.storage()], |stream| {
                    // SAFETY: lengths checked above.
                    unsafe {
                        self.module.place_embed(
                            stream,
                            config,
                            dptr(patches),
                            dptr(&self.scratch),
                            stack.ctl() as *const StackCtl,
                            g,
                            s,
                        )
                    }
                })?;
            self.pass.forward_inplace(ctx, &mut self.scratch)?;
            let (_, config) = elems(len)?;
            let reads = [
                self.scratch.storage(),
                views.addr().storage(),
                offsets.storage(),
                stack.storage(),
            ];
            self.base
                .launch(ctx, &reads, &[self.acc.storage()], |stream| {
                    // SAFETY: lengths checked above; `Views::new` checked every view's modes.
                    unsafe {
                        self.module.place_accumulate(
                            stream,
                            config,
                            dptr(&self.scratch),
                            dptr::<ViewAddr, ViewAddr>(views.addr()),
                            dptr(offsets),
                            dptr(&self.acc),
                            stack.ctl() as *const StackCtl,
                            g,
                            s,
                        )
                    }
                })?;
        }
        self.planes.inverse_inplace(ctx, &mut self.acc)?;
        let (n, config) = elems(len)?;
        let scale = 1.0 / self.plane() as f32;
        self.base
            .launch(ctx, &[self.acc.storage()], &[obj.storage()], |stream| {
                // SAFETY: `acc` holds `len` values, `obj` `len` from `slice` on.
                unsafe {
                    self.module.place_add(
                        stream,
                        config,
                        dptr_at(obj, slice),
                        dptr(&self.acc),
                        n,
                        scale,
                    )
                }
            })
    }

    /// pty-chi's full-object illumination map into `out` `[nobj, nyo, nxo]`: `re(W^H |P|^2)`
    /// floored at 0 over every view of the scan, `|P|^2` its probe mode's on each view, the
    /// same on every object mode. One transform pair serves the scan: `W^H` of a patch that
    /// depends only on the view's probe mode is the embedded `sum_p |P_p|^2` transformed
    /// once, times the frames' conjugate phases summed, transformed back.
    pub fn illumination(
        &mut self,
        ctx: &ExecutionContext,
        views: &Views,
        offsets: &DeviceBuffer<f32>,
        probe: &DeviceBuffer<C32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let at = Normal {
            views,
            offsets,
            probe,
        };
        self.spread(ctx, at, out, Spread::Illumination)
    }

    /// Updates the object `obj` `[nobj, nyo, nxo]` from its sums, `sums.num` the placed
    /// `W^H` of `conj(P) psi` over every view, to `min ‖psi − A O'‖² + reg ‖O' − O‖²`, `A`
    /// the windows weighted by the probe and `reg = inertia max D`: `STEPS` conjugate-gradient
    /// steps on `(AᴴA + reg) O' = num + reg O` from `O`, preconditioned by `D + reg`, `D`
    /// `AᴴA`'s diagonal into `sums.den`. A step along a direction of no positive curvature,
    /// or after a residual of no positive norm, is 0. Every scalar stays on the device, so the
    /// launches capture; `spectra`, one object of them, is scratch. Zeroes `sums.num` and
    /// `sums.den`, which it consumed.
    pub fn solve<const STEPS: usize>(
        &mut self,
        ctx: &ExecutionContext,
        a: Normal<'_>,
        obj: &mut DeviceBuffer<C32>,
        spectra: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        inertia: f32,
    ) -> Result<(), Error> {
        let len = self.dims.obj_len();
        check_len("obj", obj, len)?;
        check_len("spectra", spectra, len)?;
        sums.check(len)?;
        let (n, config) = elems(len)?;
        self.spread(ctx, a, &mut sums.den, Spread::Diagonal)?;
        let UpdateSums { num, den, max_bits } = sums;
        crate::gpu::zero(ctx, max_bits)?;
        let max = blocks(config.grid_dim.0.min(1024), PIXEL_THREADS);
        let m = &self.base.module;
        self.base
            .launch(ctx, &[den.storage()], &[max_bits.storage()], |stream| {
                // SAFETY: `den` holds `n` floats, `max_bits` one zeroed u32.
                unsafe { m.max_nonneg_f32(stream, max, dptr(den), n, dptr(max_bits)) }
            })?;
        let pcg = Pcg {
            x: dptr(obj),
            res: dptr(num),
            dir: dptr(&self.dir),
            ap: dptr(&self.acc),
            den: dptr(den),
            part: dptr(&self.parts),
            s: dptr(&self.scalars),
            scale: 1.0 / self.plane() as f32,
            n,
        };
        let reads = [max_bits.storage()];
        self.base
            .launch(ctx, &reads, &[self.scalars.storage()], |stream| {
                // SAFETY: `max_bits` holds one word, `pcg` as `Pcg` says.
                unsafe { (self.module).pcg_reg(stream, blocks(1, 1), dptr(max_bits), inertia, pcg) }
            })?;
        let (cg, sum) = (blocks(CG_BLOCKS, PIXEL_THREADS), blocks(1, PIXEL_THREADS));
        // `num - AᴴA O` in `num`, the residual from here on.
        self.planes.forward(ctx, obj, spectra)?;
        self.normal(ctx, a, spectra)?;
        let reads = [self.acc.storage(), den.storage()];
        let writes = [
            num.storage(),
            self.dir.storage(),
            self.parts.storage(),
            self.scalars.storage(),
        ];
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: `pcg` as `Pcg` says, `parts` `CG_BLOCKS`.
            unsafe { self.module.pcg_start(stream, cg, pcg) }
        })?;
        self.sum(ctx, sum, PCG_RZ[0])?;
        for step in 0..STEPS {
            let (rz, next) = (PCG_RZ[step % 2], PCG_RZ[(step + 1) % 2]);
            self.planes.forward(ctx, &self.dir, spectra)?;
            self.normal(ctx, a, spectra)?;
            let reads = [self.dir.storage()];
            let writes = [
                self.acc.storage(),
                self.parts.storage(),
                self.scalars.storage(),
            ];
            self.base.launch(ctx, &reads, &writes, |stream| {
                // SAFETY: as `pcg_start`.
                unsafe { self.module.pcg_curv(stream, cg, pcg) }
            })?;
            self.sum(ctx, sum, PCG_PAP)?;
            let reads = [self.dir.storage(), self.acc.storage(), den.storage()];
            let writes = [
                obj.storage(),
                num.storage(),
                self.parts.storage(),
                self.scalars.storage(),
            ];
            self.base.launch(ctx, &reads, &writes, |stream| {
                // SAFETY: as `pcg_start`, `rz` a slot of `PCG_RZ`.
                unsafe { self.module.pcg_step(stream, cg, pcg, rz) }
            })?;
            if step + 1 == STEPS {
                break;
            }
            self.sum(ctx, sum, next)?;
            let reads = [num.storage(), den.storage(), self.scalars.storage()];
            self.base
                .launch(ctx, &reads, &[self.dir.storage()], |stream| {
                    // SAFETY: as `pcg_start`, `rz` and `next` slots of `PCG_RZ`.
                    unsafe { (self.module).pcg_dir(stream, cg, pcg, rz, next) }
                })?;
        }
        crate::gpu::zero(ctx, num)?;
        crate::gpu::zero(ctx, den)
    }

    /// The scalar at `slot` of `scalars`, the sum of the block sums.
    fn sum(
        &mut self,
        ctx: &ExecutionContext,
        config: LaunchConfig,
        slot: u32,
    ) -> Result<(), Error> {
        let out = dptr::<f32, f32>(&self.scalars).wrapping_add(slot as usize);
        let reads = [self.parts.storage()];
        self.base
            .launch(ctx, &reads, &[self.scalars.storage()], |stream| {
                // SAFETY: `parts` holds `CG_BLOCKS` floats, `scalars` `slot`.
                unsafe { (self.module).pcg_sum(stream, config, dptr(&self.parts), CG_BLOCKS, out) }
            })
    }

    /// `acc = plane AᴴA x` of `x` `[nobj, nyo, nxo]`, `spectra` its [`Self::spectra`]: each
    /// frame's window on each object mode, as [`Self::windows`] samples it, weighted by
    /// `sum_p |P_p|^2` and placed back as [`Self::place`] does, over the scan as one stack.
    fn normal(
        &mut self,
        ctx: &ExecutionContext,
        a: Normal<'_>,
        spectra: &DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let stack = self.whole.at(0);
        let g = self.dims.geom(stack)?;
        check_len("probe", a.probe, self.dims.nprobe as usize * self.dims.n())?;
        crate::gpu::zero(ctx, &mut self.acc)?;
        for s in self.passes(g, self.dims.nobj) {
            let (_, config) = elems(s.npass as usize * self.plane())?;
            let reads = [
                a.views.addr().storage(),
                a.offsets.storage(),
                spectra.storage(),
                stack.storage(),
            ];
            self.base
                .launch(ctx, &reads, &[self.scratch.storage()], |stream| {
                    // SAFETY: `solve` checked the lengths; `Views::new` checked every view's
                    // modes.
                    unsafe {
                        self.module.sample_shift(
                            stream,
                            config,
                            dptr(spectra),
                            dptr::<ViewAddr, ViewAddr>(a.views.addr()),
                            dptr(a.offsets),
                            dptr(&self.scratch),
                            stack.ctl() as *const StackCtl,
                            g,
                            s,
                        )
                    }
                })?;
            self.pass.inverse_inplace(ctx, &mut self.scratch)?;
            self.base.launch(
                ctx,
                &[a.probe.storage()],
                &[self.scratch.storage()],
                |stream| {
                    // SAFETY: `probe` holds `nprobe` frames, `scratch` `npass` planes.
                    unsafe {
                        (self.module).normal_mask(
                            stream,
                            config,
                            dptr(a.probe),
                            dptr(&self.scratch),
                            g,
                            s,
                        )
                    }
                },
            )?;
            self.pass.forward_inplace(ctx, &mut self.scratch)?;
            let (_, config) = elems(self.dims.obj_len())?;
            let reads = [
                self.scratch.storage(),
                a.views.addr().storage(),
                a.offsets.storage(),
                stack.storage(),
            ];
            self.base
                .launch(ctx, &reads, &[self.acc.storage()], |stream| {
                    // SAFETY: as `sample_shift`; `acc` holds `nobj` planes.
                    unsafe {
                        self.module.place_accumulate(
                            stream,
                            config,
                            dptr(&self.scratch),
                            dptr::<ViewAddr, ViewAddr>(a.views.addr()),
                            dptr(a.offsets),
                            dptr(&self.acc),
                            stack.ctl() as *const StackCtl,
                            g,
                            s,
                        )
                    }
                })?;
        }
        Ok(self.planes.inverse_inplace(ctx, &mut self.acc)?)
    }

    /// [`Spread::Illumination`] or [`Spread::Diagonal`] of the probe's power over the scan's
    /// views at the offsets into `out` `[nobj, nyo, nxo]`: the embedded `sum_p |P_p|^2`
    /// transformed, times the frames' phases summed, transformed back, its real part floored
    /// at 0.
    fn spread(
        &mut self,
        ctx: &ExecutionContext,
        a: Normal<'_>,
        out: &mut DeviceBuffer<f32>,
        kind: Spread,
    ) -> Result<(), Error> {
        let Normal {
            views,
            offsets,
            probe,
        } = a;
        self.check_views(views, offsets)?;
        let d = self.dims;
        let g = d.scan_geom();
        let len = d.obj_len();
        check_len("probe", probe, d.nprobe as usize * d.n())?;
        check_len("out", out, len)?;
        let (_, config) = elems(len)?;
        self.base
            .launch(ctx, &[probe.storage()], &[self.acc.storage()], |stream| {
                // SAFETY: lengths checked above.
                unsafe {
                    self.module
                        .illum_embed(stream, config, dptr(probe), dptr(&self.acc), g, d.nobj)
                }
            })?;
        self.planes.forward_inplace(ctx, &mut self.acc)?;
        let (_, config) = elems(self.plane())?;
        let reads = [views.addr().storage(), offsets.storage()];
        self.base
            .launch(ctx, &reads, &[self.acc.storage()], |stream| {
                let (addr, offsets) = (dptr::<ViewAddr, ViewAddr>(views.addr()), dptr(offsets));
                let acc = dptr(&self.acc);
                // SAFETY: lengths checked above; `Views::new` checked every view's modes.
                unsafe {
                    match kind {
                        Spread::Illumination => {
                            (self.module).illum_phase(stream, config, addr, offsets, acc, g, d.nobj)
                        }
                        Spread::Diagonal => {
                            (self.module).diag_phase(stream, config, addr, offsets, acc, g, d.nobj)
                        }
                    }
                }
            })?;
        self.planes.inverse_inplace(ctx, &mut self.acc)?;
        let (n, config) = elems(len)?;
        let scale = 1.0 / self.plane() as f32;
        self.base
            .launch(ctx, &[self.acc.storage()], &[out.storage()], |stream| {
                // SAFETY: `acc` holds `len` complex values, `out` `len` floats.
                unsafe {
                    self.module
                        .illum_clamp(stream, config, dptr(&self.acc), dptr(out), n, scale)
                }
            })
    }

    fn plane(&self) -> usize {
        self.dims.obj[0] as usize * self.dims.obj[1] as usize
    }

    fn object_len(&self) -> usize {
        self.nslice as usize * self.dims.obj_len()
    }

    /// The first value of `slice` in an object.
    fn slice(&self, slice: u32) -> Result<usize, Error> {
        if slice >= self.nslice {
            return Err(Error::Shape(format!("slice {slice} of {}", self.nslice)));
        }
        Ok(slice as usize * self.dims.obj_len())
    }

    /// `views` are of this sampler's dims and `offsets` holds a pair per frame.
    fn check_views(&self, views: &Views, offsets: &DeviceBuffer<f32>) -> Result<(), Error> {
        if *views.dims() != self.dims {
            return Err(Error::Shape(format!(
                "views of {:?}, sampler of {:?}",
                views.dims(),
                self.dims
            )));
        }
        check_len("offsets", offsets, 2 * self.dims.nframe as usize)
    }

    fn geom(&self, at: StackPoints<'_>) -> Result<StackGeom, Error> {
        self.check_views(at.views, at.offsets)?;
        self.dims.geom(at.stack)
    }

    /// The passes over the groups of a stack of `g.nf_max` frames, `modes` per frame.
    fn passes(&self, g: StackGeom, modes: u32) -> impl Iterator<Item = SamplePass> + use<> {
        let chunk = self.chunk;
        let npoint = g.nf_max * modes;
        (0..npoint)
            .step_by(chunk as usize)
            .map(move |p0| SamplePass {
                p0,
                npass: chunk.min(npoint - p0),
                modes,
                ord_y: 0,
                ord_x: 0,
            })
    }
}
