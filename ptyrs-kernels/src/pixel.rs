//! Records the M1 pixel kernels share with the host, and their device helpers.
//!
//! Index conventions (all kernels): a probe/frame plane is `[ny, nx]` row-major with even
//! `ny`, `nx`; `n = ny * nx`. Probe and object are in natural (centred) order. A real-space
//! exit wave is stored with its origin at `[0, 0]`: probe pixel `(pry, prx)` lands at
//! [`psi_index`]. The same map takes an FFT-order frequency to its detector pixel (zero
//! frequency at `[ny/2, nx/2]`), because for even sizes the shift is its own inverse.

use cuda_device::device;

/// Object tiles of the deterministic object gather are `OBJ_TILE x OBJ_TILE` pixels.
pub const OBJ_TILE: u32 = 16;
/// Block size of every 1D/2D pixel kernel and of the reductions.
pub const PIXEL_THREADS: u32 = 256;
/// Count value of a masked pixel (`ptyrs-core` `MASKED`).
pub const MASKED: u16 = u16::MAX;

/// One exit-wave view (PyNX `cx, cy` + `_obj_psidx`, `_probe_psidx`): probe mode `probe`
/// times object mode `obj` sampled at `(oy + fy + pry, ox + fx + prx)` for probe pixel
/// `(pry, prx)`. `oy + ny <= nyo`, `ox + nx <= nxo`, `fy`, `fx` in `[0, 1)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewAddr {
    pub oy: u32,
    pub ox: u32,
    pub fy: f32,
    pub fx: f32,
    pub probe: u32,
    pub obj: u32,
}

// SAFETY: repr(C) of four u32 and two f32, no padding; every bit pattern is a valid value.
unsafe impl cuda_core::DeviceCopy for ViewAddr {}

/// The frames `f0 .. f0 + nf` of one stack. A kernel reads it from device memory when it
/// runs, so one launch, or one captured graph, serves whichever stack the entry holds then.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StackCtl {
    pub f0: u32,
    pub nf: u32,
}

// SAFETY: repr(C) of two u32, no padding; every bit pattern is a valid value.
unsafe impl cuda_core::DeviceCopy for StackCtl {}

/// What a stack kernel needs besides its buffers and [`StackCtl`], passed by value: the
/// scan's `nframe` and `nmode`, `nf_max` (the most frames a stack may have, which sizes the
/// launch and the per-stack buffers), the probe mode count, the frame `[ny, nx]`, the object
/// `[nyo, nxo]` and the far-field phase `quad` (radians per squared pixel).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StackGeom {
    pub nframe: u32,
    pub nf_max: u32,
    pub nmode: u32,
    pub nprobe: u32,
    pub ny: u32,
    pub nx: u32,
    pub nyo: u32,
    pub nxo: u32,
    pub quad: f32,
}

/// `(f0, nf)` of the stack `ctl` holds, clamped to `nf <= nf_max` and `f0 + nf <= nframe`:
/// the one place a kernel reads a [`StackCtl`], so no entry, however written, makes a launch
/// address past its per-stack buffers or the scan.
///
/// # Safety
///
/// `ctl` must point to a readable `StackCtl`.
#[device]
pub unsafe fn stack_frames(ctl: *const StackCtl, g: StackGeom) -> (u32, u32) {
    let c = unsafe { *ctl };
    let f0 = c.f0.min(g.nframe);
    (f0, c.nf.min(g.nf_max).min(g.nframe - f0))
}

/// Device addresses of an update's numerator (complex, interleaved) and denominator (real);
/// element `i` of one belongs to element `i` of the other.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SumPtrs {
    pub num: *mut f32,
    pub den: *mut f32,
}

/// Device addresses of the per-object-tile view lists: tile `t`'s run is
/// `views[start[t] .. start[t + 1]]`, ascending view indices.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TilePtrs {
    pub start: *const u32,
    pub views: *const u32,
}

/// Per-frame log-likelihood sums over the frame's unmasked pixels (PyNX `LLKAll`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LlkPartial {
    pub poisson: f32,
    pub gaussian: f32,
    pub euclidean: f32,
    /// Sum of the calculated intensity.
    pub icalc: f32,
}

// SAFETY: repr(C) of four f32, no padding; every bit pattern is a valid value.
unsafe impl cuda_core::DeviceCopy for LlkPartial {}

/// Offset in an origin-at-`[0, 0]` plane of natural pixel `(y, x)`.
#[device]
pub fn psi_index(y: u32, x: u32, ny: u32, nx: u32) -> usize {
    ((y + ny / 2) % ny) as usize * nx as usize + ((x + nx / 2) % nx) as usize
}

/// `exp(i * quad * r2)` as `(cos, sin)`, `r2 = (y - ny/2)^2 + (x - nx/2)^2` in pixels.
#[device]
pub fn quad_phase(quad: f32, y: u32, x: u32, ny: u32, nx: u32) -> (f32, f32) {
    let dy = y as i32 - (ny / 2) as i32;
    let dx = x as i32 - (nx / 2) as i32;
    let a = quad * (dy * dy + dx * dx) as f32;
    (a.cos(), a.sin())
}

#[device]
pub fn cmul(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// `conj(a) * b`.
#[device]
pub fn cmul_conj(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    (a.0 * b.0 + a.1 * b.1, a.0 * b.1 - a.1 * b.0)
}

/// Complex value `i` of an interleaved buffer.
///
/// # Safety
///
/// `p` must hold `i + 1` complex values.
#[device]
pub unsafe fn load_c(p: *const f32, i: usize) -> (f32, f32) {
    unsafe { (*p.add(2 * i), *p.add(2 * i + 1)) }
}

/// # Safety
///
/// `p` must hold `i + 1` complex values.
#[device]
pub unsafe fn store_c(p: *mut f32, i: usize, v: (f32, f32)) {
    unsafe {
        *p.add(2 * i) = v.0;
        *p.add(2 * i + 1) = v.1;
    }
}

/// Bilinear weights of the four object pixels a sample reads, in the order
/// `(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)`.
#[device]
pub fn bilinear_weights(fy: f32, fx: f32) -> [f32; 4] {
    [
        (1.0 - fy) * (1.0 - fx),
        (1.0 - fy) * fx,
        fy * (1.0 - fx),
        fy * fx,
    ]
}

/// Object value probe pixel `(pry, prx)` of view `v` sees: bilinear over the object pixels
/// `(oy + pry + {0, 1}, ox + prx + {0, 1})`, the `+ 1` neighbours wrapped periodically
/// (PyNX `bilinear`). Corners whose weight is zero (`fy == 0` or `fx == 0`) are not read.
///
/// # Safety
///
/// `obj` must hold `[nobj, nyo, nxo]` with `v.obj < nobj`, `v.oy + pry < nyo` and
/// `v.ox + prx < nxo`.
#[device]
pub unsafe fn sample_obj(
    obj: *const f32,
    v: ViewAddr,
    pry: u32,
    prx: u32,
    nyo: u32,
    nxo: u32,
) -> (f32, f32) {
    let plane = v.obj as usize * nyo as usize * nxo as usize;
    let y0 = v.oy + pry;
    let x0 = v.ox + prx;
    let y1 = if y0 + 1 == nyo { 0 } else { y0 + 1 };
    let x1 = if x0 + 1 == nxo { 0 } else { x0 + 1 };
    let w = bilinear_weights(v.fy, v.fx);
    let at = |y: u32, x: u32| plane + y as usize * nxo as usize + x as usize;
    unsafe {
        let o = load_c(obj, at(y0, x0));
        let mut acc = (w[0] * o.0, w[0] * o.1);
        if v.fx != 0.0 {
            let o = load_c(obj, at(y0, x1));
            acc = (acc.0 + w[1] * o.0, acc.1 + w[1] * o.1);
        }
        if v.fy != 0.0 {
            let o = load_c(obj, at(y1, x0));
            acc = (acc.0 + w[2] * o.0, acc.1 + w[2] * o.1);
            if v.fx != 0.0 {
                let o = load_c(obj, at(y1, x1));
                acc = (acc.0 + w[3] * o.0, acc.1 + w[3] * o.1);
            }
        }
        acc
    }
}

/// The probe pixels of view `v` whose bilinear sample reads object pixel `(y, x)`, as
/// `(pry, prx, weight)` in [`bilinear_weights`] order; `None` where no such pixel exists or
/// the weight is zero. This is the exact transpose of [`sample_obj`].
#[device]
pub fn obj_pixel_sources(
    v: ViewAddr,
    y: u32,
    x: u32,
    ny: u32,
    nx: u32,
    nyo: u32,
    nxo: u32,
) -> [Option<(u32, u32, f32)>; 4] {
    // Row offset of `y` in the view (`y0 = oy + pry`), and of the row whose `+ 1`
    // neighbour is `y`; likewise for columns.
    let r = (y + nyo - v.oy) % nyo;
    let c = (x + nxo - v.ox) % nxo;
    let rm = if r == 0 { nyo - 1 } else { r - 1 };
    let cm = if c == 0 { nxo - 1 } else { c - 1 };
    let w = bilinear_weights(v.fy, v.fx);
    let pick = |pry: u32, prx: u32, wt: f32, used: bool| {
        if used && pry < ny && prx < nx {
            Some((pry, prx, wt))
        } else {
            None
        }
    };
    [
        pick(r, c, w[0], true),
        pick(r, cm, w[1], v.fx != 0.0),
        pick(rm, c, w[2], v.fy != 0.0),
        pick(rm, cm, w[3], v.fy != 0.0 && v.fx != 0.0),
    ]
}
