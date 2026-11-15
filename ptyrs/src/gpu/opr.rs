//! Checked launches of the OPR kernels (`ptyrs_kernels::opr`): pty-chi's orthogonal probe
//! relaxation of probe mode 0 as `engine::opr`'s `ProbeVariation` defines it, in f32 on the
//! device. [`OprKernels::unique`] gives a run of epoch positions their own mode-0 planes;
//! [`OprKernels::step`] moves the eigenmodes and the refined weights and main-mode weights
//! from a batch's per-frame planes. Every plane is `n` pixels in the probe's own layout, the
//! operations pixelwise or sums over the pixels, so any layout serves that every plane shares.

use std::ops::Range;
use std::ptr;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig, RawDeviceBuffer};
use ptyrs_kernels::fused::{ProbeSrc, shared_probe};
use ptyrs_kernels::opr::{OPR_SUMS, OprFrames, OprScan, opr_kernels};
use ptyrs_kernels::pixel::PIXEL_THREADS;
use ptyrs_kernels::position::BatchSpan;

use crate::base::Feature;
use crate::gpu::fused::{GRID_Y_MAX, OwnSpectra};
use crate::gpu::ops::grid_1d;
use crate::gpu::position::{ONE_BLOCK, chunks};
use crate::gpu::{C32, EpochOrder, Error, Kernels, ViewDims, check_len, dptr};

#[cfg(test)]
mod tests;

/// The OPR kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct OprKernels {
    base: Kernels,
    module: opr_kernels::LoadedModule,
}

/// The OPR state of one scan: `k ≥ 1` eigenmodes `[k, n]`, their weights `w_fi` at `f k + i`
/// and, when modelled, the main mode's weights `c_f` at `f`, each refined or held, both by
/// scan frame, with the scratch of [`OprKernels::step`] for batches of up to `nb_max` frames.
pub struct Opr {
    n: usize,
    k: u32,
    nframe: u32,
    nb_max: u32,
    eigenmodes: DeviceBuffer<C32>,
    weights: Feature<DeviceBuffer<f32>>,
    intensity: Option<Feature<DeviceBuffer<f32>>>,
    /// [`OPR_SUMS`] per chunk of each batch frame.
    sums: DeviceBuffer<f32>,
    /// Two plane sums, per chunk.
    part: DeviceBuffer<f32>,
    e: DeviceBuffer<C32>,
    /// Eigenmode `i`'s update coefficient per batch frame: `opr_coef` writes it, `opr_e` reads
    /// it.
    coef: DeviceBuffer<f32>,
    /// The deflation's complex coefficient per batch frame: `opr_deflate_coef` writes it,
    /// `opr_deflate` reads it.
    deflate_coef: DeviceBuffer<f32>,
}

impl Opr {
    /// Uploads `eigenmodes` (`k n` values, `k ≥ 1`), `weights` (`k` per scan frame) and, when
    /// modelled, `intensity` (one per scan frame), for planes of `n` pixels and batches of up
    /// to `nb_max` frames.
    pub fn new(
        stream: &CudaStream,
        n: usize,
        eigenmodes: &[C32],
        weights: Feature<&[f32]>,
        intensity: Option<Feature<&[f32]>>,
        nb_max: u32,
    ) -> Result<Self, Error> {
        let bad = |what: String| Err(Error::Shape(what));
        if n == 0 || u32::try_from(n).is_err() || !eigenmodes.len().is_multiple_of(n) {
            return bad(format!(
                "{} eigenmode values in planes of {n}",
                eigenmodes.len()
            ));
        }
        let (k, w) = (eigenmodes.len() / n, weights.values);
        if k == 0 || !w.len().is_multiple_of(k) || w.is_empty() {
            return bad(format!("{} weights for {k} eigenmodes", w.len()));
        }
        let nframe = w.len() / k;
        if let Some(c) = intensity
            && c.values.len() != nframe
        {
            let have = c.values.len();
            return bad(format!("{have} main-mode weights for {nframe} frames"));
        }
        if nb_max == 0 || nb_max as usize > nframe || nb_max as u64 > GRID_Y_MAX {
            return bad(format!("batches of {nb_max} of {nframe} frames"));
        }
        let (k, nframe, nb) = (
            u32::try_from(k).map_err(|_| Error::Shape(format!("{k} eigenmodes")))?,
            u32::try_from(nframe).map_err(|_| Error::Shape(format!("{nframe} frames")))?,
            nb_max as usize,
        );
        Ok(Self {
            n,
            k,
            nframe,
            nb_max,
            eigenmodes: DeviceBuffer::from_host(stream, eigenmodes)?,
            weights: Feature {
                values: DeviceBuffer::from_host(stream, w)?,
                refine: weights.refine,
            },
            intensity: match intensity {
                Some(c) => Some(Feature {
                    values: DeviceBuffer::from_host(stream, c.values)?,
                    refine: c.refine,
                }),
                None => None,
            },
            sums: DeviceBuffer::zeroed(stream, OPR_SUMS as usize * nb * chunks(n))?,
            part: DeviceBuffer::zeroed(stream, 2 * chunks(n))?,
            e: DeviceBuffer::zeroed(stream, n)?,
            coef: DeviceBuffer::zeroed(stream, nb)?,
            deflate_coef: DeviceBuffer::zeroed(stream, 2 * nb)?,
        })
    }

    /// The eigenmodes `[k, n]`.
    pub fn eigenmodes(&self) -> &DeviceBuffer<C32> {
        &self.eigenmodes
    }

    /// `w_fi` at `f k + i`, by scan frame.
    pub fn weights(&self) -> &Feature<DeviceBuffer<f32>> {
        &self.weights
    }

    /// `c_f` by scan frame, when modelled.
    pub fn intensity(&self) -> Option<&Feature<DeviceBuffer<f32>>> {
        self.intensity.as_ref()
    }

    /// The epoch positions `p` through `order`, a permutation of this scan's frames.
    fn span(&self, order: &EpochOrder, p: &Range<u32>) -> Result<BatchSpan, Error> {
        let have = order.ids().len();
        if have != self.nframe as usize {
            return Err(Error::Shape(format!(
                "epoch order of {have} frames, OPR of {}",
                self.nframe
            )));
        }
        if p.start >= p.end || p.end > self.nframe {
            return Err(Error::Shape(format!(
                "positions {p:?} outside {} frames",
                self.nframe
            )));
        }
        Ok(BatchSpan {
            p0: p.start,
            nb: p.end - p.start,
            nchunk: chunks(self.n) as u32,
        })
    }

    /// The weights by scan frame, with `ids`, the scan frame at each epoch position.
    fn scan(&self, ids: &DeviceBuffer<u32>) -> OprScan {
        OprScan {
            order: dptr::<u32, u32>(ids).cast_const(),
            weights: dptr(&self.weights.values),
            intensity: self
                .intensity
                .as_ref()
                .map_or(ptr::null_mut(), |c| dptr(&c.values)),
            k: self.k,
        }
    }

    /// Eigenmode `i`'s plane.
    fn eigenmode(&self, i: u32) -> *mut f32 {
        // In bounds: `i < k` and the buffer holds `k n` values.
        dptr::<C32, f32>(&self.eigenmodes).wrapping_add(2 * i as usize * self.n)
    }
}

/// Mode 0 of each frame's own probe over a run of epoch positions: [`OprKernels::unique`]'s
/// `planes`, `[nb, n]`, plane `b` at position `positions.start + b`; from
/// [`Kernels::own_spectrum`], with their spectra, which the shifted passes read instead.
#[derive(Clone)]
pub struct OwnModes<'a> {
    planes: &'a DeviceBuffer<C32>,
    positions: Range<u32>,
    spectra: Option<OwnSpectra<'a>>,
}

impl<'a> OwnModes<'a> {
    pub fn new(planes: &'a DeviceBuffer<C32>, positions: Range<u32>) -> Self {
        Self {
            planes,
            positions,
            spectra: None,
        }
    }

    /// These modes with `spectra`, the spectra of their planes.
    pub(super) fn with_spectra(&self, spectra: OwnSpectra<'a>) -> Self {
        Self {
            spectra: Some(spectra),
            ..self.clone()
        }
    }

    pub(super) fn planes(&self) -> &'a DeviceBuffer<C32> {
        self.planes
    }

    /// `probe` with these planes of `n` pixels for its mode 0, as a kernel reads it.
    pub(super) fn src(&self, probe: &DeviceBuffer<C32>, n: usize) -> Result<ProbeSrc, Error> {
        self.src_of(probe, self.planes, n)
    }

    /// The spectra `spectrum` of the shared probe with those of these planes for its mode 0,
    /// as the shifted passes of `d` read them, and the buffers they address.
    pub(super) fn spectrum_src<'b>(
        &'b self,
        spectrum: &'b DeviceBuffer<C32>,
        d: &ViewDims,
    ) -> Result<(ProbeSrc, [&'b RawDeviceBuffer; 2]), Error> {
        let Some(own) = &self.spectra else {
            return Err(Error::Shape(
                "own probe modes under a Model::shift need their spectra, from \
                 Kernels::own_spectrum"
                    .into(),
            ));
        };
        let a = own.of(d)?;
        let src = self.src_of(spectrum, a, d.n())?;
        Ok((src, [spectrum.storage(), a.storage()]))
    }

    /// The number of planes, checked against `self`'s.
    pub(super) fn nb(&self, n: usize) -> Result<u32, Error> {
        self.checked_nb(self.planes, n)
    }

    /// The number of positions, refused for none or a `mode0` short of their planes.
    fn checked_nb(&self, mode0: &DeviceBuffer<C32>, n: usize) -> Result<u32, Error> {
        let nb = self.positions.len();
        if nb == 0 {
            let p = &self.positions;
            return Err(Error::Shape(format!(
                "own probe modes at no positions, {p:?}"
            )));
        }
        check_len("own probe modes", mode0, nb * n)?;
        Ok(nb as u32)
    }

    fn src_of(
        &self,
        probe: &DeviceBuffer<C32>,
        mode0: &DeviceBuffer<C32>,
        n: usize,
    ) -> Result<ProbeSrc, Error> {
        let nb = self.checked_nb(mode0, n)?;
        Ok(ProbeSrc {
            probe: dptr::<C32, f32>(probe).cast_const(),
            mode0: dptr::<C32, f32>(mode0).cast_const(),
            p0: self.positions.start,
            nb,
        })
    }
}

/// The probe `[nprobe, ny, nx]` a pass's frames read: shared by all of them, or with each
/// frame's mode 0 from its [`OwnModes`] plane instead.
#[derive(Clone, Copy)]
pub enum FrameProbe<'a> {
    Shared(&'a DeviceBuffer<C32>),
    Own(&'a DeviceBuffer<C32>, &'a OwnModes<'a>),
}

impl<'a> FrameProbe<'a> {
    /// The probe planes; `Own` reads only modes `1..` of them.
    pub(super) fn probe(self) -> &'a DeviceBuffer<C32> {
        match self {
            Self::Shared(p) | Self::Own(p, _) => p,
        }
    }

    /// The buffers a kernel reading [`Self::src`] reads.
    pub(super) fn storages(self) -> impl Iterator<Item = &'a RawDeviceBuffer> {
        let own = match self {
            Self::Shared(_) => None,
            Self::Own(_, own) => Some(own.planes.storage()),
        };
        std::iter::once(self.probe().storage()).chain(own)
    }

    /// The probe of planes of `n` pixels as a kernel reads it.
    pub(super) fn src(self, n: usize) -> Result<ProbeSrc, Error> {
        match self {
            Self::Shared(p) => Ok(shared_src(p)),
            Self::Own(p, own) => own.src(p, n),
        }
    }
}

/// `probe` shared by every frame, as a kernel reads it.
pub(super) fn shared_src(probe: &DeviceBuffer<C32>) -> ProbeSrc {
    shared_probe(dptr::<C32, f32>(probe).cast_const())
}

/// A batch's per-frame mode-0 planes for [`OprKernels::step`], each `[nb, n]` by batch frame
/// `b` (epoch position `p0 + b`): the probe direction `d_b = conj(q V_b) chi_b0`, `vv_b =
/// |V_b|²` at it and `moved`, `d` taken back through the frames' probe moves (`d` itself
/// without), `V_b` the frame's object view. The step leaves its residues in `moved`.
pub struct OprPlanes<'a> {
    pub d: &'a DeviceBuffer<C32>,
    pub vv: &'a DeviceBuffer<f32>,
    pub moved: &'a mut DeviceBuffer<C32>,
}

impl OprKernels {
    /// Binds the OPR kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = opr_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// `ProbeVariation::unique`'s mode-0 planes of the epoch positions `frames` into `out`,
    /// `[nb, n]` by position `p0 + b`: `c_f P_0 + w_f0 U_0 + w_f1 U_1 + …` in that order, `f`
    /// the scan frame `order` puts there and `P_0` the first `n` values of `probe`; the other
    /// probe modes stay shared.
    pub fn unique(
        &self,
        ctx: &ExecutionContext,
        opr: &Opr,
        order: &EpochOrder,
        frames: Range<u32>,
        probe: &DeviceBuffer<C32>,
        out: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let span = opr.span(order, &frames)?;
        let (n, total) = (opr.n, span.nb as usize * opr.n);
        check_len("probe", probe, n)?;
        check_len("out", out, total)?;
        let config = grid_1d(total)?;
        let ids = order.ids();
        let mut reads = vec![
            probe.storage(),
            opr.eigenmodes.storage(),
            opr.weights.values.storage(),
            ids.storage(),
        ];
        reads.extend(opr.intensity.as_ref().map(|c| c.values.storage()));
        let scan = opr.scan(ids);
        self.base.launch(ctx, &reads, &[out.storage()], |stream| {
            // SAFETY: `span` lies in the scan's positions and `order` maps them to its
            // frames; `probe` and `out` checked above, the state sized by `Opr::new`.
            unsafe {
                self.module.opr_unique(
                    stream,
                    config,
                    dptr::<C32, f32>(probe).cast_const(),
                    dptr::<C32, f32>(&opr.eigenmodes).cast_const(),
                    scan,
                    span,
                    n as u32,
                    dptr::<C32, f32>(out),
                )
            }
        })
    }

    /// `ProbeVariation::step` on the batch of epoch positions `batch` through `order`, from
    /// its [`OprPlanes`] and `P_0`, the first `n` values of `probe`: the refined main-mode
    /// weights first, then, for two frames or more, each eigenmode and its refined weights in
    /// turn. Every sum over a frame's pixels is totalled in chunk order and every sum over the
    /// batch in frame order; `Σ |U″|²` is its normalization's `n²`.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        opr: &mut Opr,
        order: &EpochOrder,
        batch: Range<u32>,
        probe: &DeviceBuffer<C32>,
        planes: OprPlanes<'_>,
    ) -> Result<(), Error> {
        let span = opr.span(order, &batch)?;
        if span.nb > opr.nb_max {
            return Err(Error::Shape(format!(
                "batch of {} frames, OPR scratch for {}",
                span.nb, opr.nb_max
            )));
        }
        let OprPlanes { d, vv, moved } = planes;
        let (n, nb) = (opr.n, span.nb as usize);
        check_len("probe", probe, n)?;
        check_len("d", d, nb * n)?;
        check_len("vv", vv, nb * n)?;
        check_len("moved", moved, nb * n)?;
        let ids = order.ids();
        let scan = opr.scan(ids);
        let (nu, k, line) = (n as u32, opr.k, grid_1d(n)?);
        if let Some(c) = opr.intensity.as_ref().filter(|c| c.refine) {
            let of = SumsOf {
                a: Shared::Probe(probe),
                dv: Some((d, vv)),
                r: None,
            };
            self.frame_sums(ctx, opr, span, of)?;
            let reads = [opr.sums.storage(), ids.storage()];
            self.base
                .launch(ctx, &reads, &[c.values.storage()], |stream| {
                    // SAFETY: `span` lies in the scan's positions, `sums` and `c` sized by
                    // `Opr::new`; one block.
                    unsafe {
                        self.module.opr_intensity(
                            stream,
                            ONE_BLOCK,
                            dptr::<f32, f32>(&opr.sums).cast_const(),
                            span,
                            scan,
                        )
                    }
                })?;
        }
        if nb < 2 {
            return Ok(());
        }
        let r = dptr::<C32, f32>(moved);
        self.base.launch(ctx, &[], &[moved.storage()], |stream| {
            // SAFETY: `moved` holds `nb n` values.
            unsafe { self.module.opr_residual(stream, line, r, span.nb, nu) }
        })?;
        let a_u = 0.1f64.min(nb as f64 / opr.nframe as f64) as f32;
        let chunk_grid = LaunchConfig {
            grid_dim: (span.nchunk, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        for i in 0..k {
            let u = opr.eigenmode(i);
            let of = SumsOf {
                a: Shared::Eigenmode(i),
                dv: None,
                r: Some(moved),
            };
            self.frame_sums(ctx, opr, span, of)?;
            let reads = [
                opr.sums.storage(),
                ids.storage(),
                opr.weights.values.storage(),
            ];
            self.base
                .launch(ctx, &reads, &[opr.coef.storage()], |stream| {
                    // SAFETY: `span` lies in the scan's positions, `i < k`, the state sized by
                    // `Opr::new`; one block.
                    unsafe {
                        self.module.opr_coef(
                            stream,
                            ONE_BLOCK,
                            dptr::<f32, f32>(&opr.sums).cast_const(),
                            span,
                            scan,
                            i,
                            nu,
                            dptr(&opr.coef),
                        )
                    }
                })?;
            let reads = [moved.storage(), opr.coef.storage()];
            let writes = [opr.e.storage(), opr.part.storage()];
            self.base.launch(ctx, &reads, &writes, |stream| {
                // SAFETY: `moved` holds `nb n` values, the state sized by `Opr::new`; one
                // block per chunk.
                unsafe {
                    self.module.opr_e(
                        stream,
                        chunk_grid,
                        r.cast_const(),
                        dptr::<f32, f32>(&opr.coef).cast_const(),
                        span.nb,
                        nu,
                        dptr(&opr.e),
                        dptr(&opr.part),
                    )
                }
            })?;
            let writes = [opr.eigenmodes.storage(), opr.part.storage()];
            self.base
                .launch(ctx, &[opr.e.storage()], &writes, |stream| {
                    // SAFETY: eigenmode `i` and `e` hold `n` values, `part` `2 nchunk`; one
                    // block per chunk.
                    unsafe {
                        self.module.opr_move(
                            stream,
                            chunk_grid,
                            u,
                            dptr::<C32, f32>(&opr.e).cast_const(),
                            a_u,
                            nu,
                            dptr(&opr.part),
                        )
                    }
                })?;
            self.base.launch(
                ctx,
                &[opr.part.storage()],
                &[opr.eigenmodes.storage()],
                |stream| {
                    // SAFETY: eigenmode `i` holds `n` values, `part` `2 nchunk`.
                    unsafe {
                        self.module.opr_scale(
                            stream,
                            line,
                            u,
                            nu,
                            dptr::<f32, f32>(&opr.part).cast_const(),
                        )
                    }
                },
            )?;
            let (refine, deflate) = (opr.weights.refine, i + 1 < k);
            if !(refine || deflate) {
                continue;
            }
            let of = SumsOf {
                a: Shared::Eigenmode(i),
                dv: refine.then_some((d, vv)),
                r: deflate.then_some(&*moved),
            };
            self.frame_sums(ctx, opr, span, of)?;
            if refine {
                let reads = [opr.sums.storage(), ids.storage()];
                let writes = [opr.weights.values.storage()];
                self.base.launch(ctx, &reads, &writes, |stream| {
                    // SAFETY: `span` lies in the scan's positions, `i < k`, the state sized by
                    // `Opr::new`; one block.
                    unsafe {
                        self.module.opr_weights(
                            stream,
                            ONE_BLOCK,
                            dptr::<f32, f32>(&opr.sums).cast_const(),
                            span,
                            scan,
                            i,
                        )
                    }
                })?;
            }
            if deflate {
                self.deflate(ctx, opr, span, i, moved)?;
            }
        }
        Ok(())
    }

    /// `r_b −= coef_b U″` for every batch frame `b` of `moved`, `U″` eigenmode `i` and
    /// `coef_b = Σ conj(U″) r_b / n²` from the [`OPR_SUMS`] in `opr`'s sums.
    fn deflate(
        &self,
        ctx: &ExecutionContext,
        opr: &Opr,
        span: BatchSpan,
        i: u32,
        moved: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let (nu, per_frame, line) = (opr.n as u32, grid_1d(span.nb as usize)?, grid_1d(opr.n)?);
        self.base.launch(
            ctx,
            &[opr.sums.storage()],
            &[opr.deflate_coef.storage()],
            |stream| {
                // SAFETY: `span.nb ≤ nb_max`, `sums` and `deflate_coef` sized by `Opr::new` for
                // it.
                unsafe {
                    self.module.opr_deflate_coef(
                        stream,
                        per_frame,
                        dptr::<f32, f32>(&opr.sums).cast_const(),
                        span,
                        nu,
                        dptr(&opr.deflate_coef),
                    )
                }
            },
        )?;
        let reads = [opr.eigenmodes.storage(), opr.deflate_coef.storage()];
        let (r, u) = (dptr::<C32, f32>(moved), opr.eigenmode(i).cast_const());
        self.base.launch(ctx, &reads, &[moved.storage()], |stream| {
            // SAFETY: `moved` holds `nb n` values, eigenmode `i` `n`, `deflate_coef` `2 nb`.
            unsafe {
                self.module.opr_deflate(
                    stream,
                    line,
                    r,
                    u,
                    dptr::<f32, f32>(&opr.deflate_coef).cast_const(),
                    span.nb,
                    nu,
                )
            }
        })
    }

    /// The [`OPR_SUMS`] of `of` per chunk of each batch frame into `opr`'s sums.
    fn frame_sums(
        &self,
        ctx: &ExecutionContext,
        opr: &Opr,
        span: BatchSpan,
        of: SumsOf<'_>,
    ) -> Result<(), Error> {
        let config = LaunchConfig {
            grid_dim: (span.nchunk, span.nb, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let a = match of.a {
            Shared::Probe(p) => (dptr::<C32, f32>(p), p.storage()),
            Shared::Eigenmode(i) => (opr.eigenmode(i), opr.eigenmodes.storage()),
        };
        let mut reads = vec![a.1];
        reads.extend(of.dv.iter().flat_map(|(d, vv)| [d.storage(), vv.storage()]));
        reads.extend(of.r.map(|r| r.storage()));
        let planes = OprFrames {
            d: of
                .dv
                .map_or(ptr::null(), |(d, _)| dptr::<C32, f32>(d).cast_const()),
            vv: of
                .dv
                .map_or(ptr::null(), |(_, vv)| dptr::<f32, f32>(vv).cast_const()),
            r: of
                .r
                .map_or(ptr::null(), |r| dptr::<C32, f32>(r).cast_const()),
        };
        self.base
            .launch(ctx, &reads, &[opr.sums.storage()], |stream| {
                // SAFETY: the caller checked the shared plane holds `n` values and every
                // frame plane `nb n`; `nb ≤ nb_max ≤ GRID_Y_MAX`, `sums` sized by `Opr::new`.
                unsafe {
                    self.module.opr_frame_sums(
                        stream,
                        config,
                        a.0.cast_const(),
                        planes,
                        opr.n as u32,
                        dptr(&opr.sums),
                    )
                }
            })
    }
}

/// The shared plane `A` of [`OprKernels::frame_sums`].
#[derive(Clone, Copy)]
enum Shared<'a> {
    /// The first `n` values: the probe's mode 0.
    Probe(&'a DeviceBuffer<C32>),
    Eigenmode(u32),
}

/// What [`OprKernels::frame_sums`] sums: `A` against the batch's `d` and `vv`, and against
/// its residues `r`; a plane left out sums to 0.
#[derive(Clone, Copy)]
struct SumsOf<'a> {
    a: Shared<'a>,
    dv: Option<(&'a DeviceBuffer<C32>, &'a DeviceBuffer<f32>)>,
    r: Option<&'a DeviceBuffer<C32>>,
}
