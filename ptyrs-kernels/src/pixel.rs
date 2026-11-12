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
/// Pixels per block of the M1 LLK kernel: a frame of `n` pixels is `ceil(n / LLK_CHUNK)`
/// contiguous chunks, the last one short.
pub const LLK_CHUNK: u32 = 2048;

/// One exit-wave view (PyNX `cx, cy` + `_obj_psidx`, `_probe_psidx`): probe mode `probe`
/// times object mode `obj` sampled at `(oy + fy + pry, ox + fx + prx)` for probe pixel
/// `(pry, prx)`. `fy`, `fx` in `[0, 1)`; `oy + ny + (fy > 0) <= nyo` and `ox + nx + (fx > 0)
/// <= nxo`, so every object pixel a sample reads lies inside the object.
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
/// runs, so one launch, or one captured graph, serves whichever stack the entry holds then;
/// with `nf == 0` it writes nothing.
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

// An object tile is one block of the object kernels and of the reductions.
const _: () = assert!(OBJ_TILE * OBJ_TILE == PIXEL_THREADS);

/// Threads of a warp; `bin_views` runs one warp per object tile.
pub const WARP: u32 = 32;
// A 1D launch's warps lie whole in its blocks.
const _: () = assert!(PIXEL_THREADS.is_multiple_of(WARP));

/// Device addresses of the per-object-tile view lists: tile `t`'s run is
/// `views[start[t] .. start[t] + len[t]]`, ascending view indices, `len[t] <=
/// start[t + 1] - start[t]`. No kernel reads or writes the object or sum pixels of a tile
/// with `len[t] == 0`, nor its run.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TilePtrs {
    pub start: *const u32,
    pub len: *const u32,
    pub views: *const u32,
}

/// Tile `t`'s run `lo .. end` of `tiles.views`.
///
/// # Safety
///
/// `tiles.start` and `tiles.len` must hold entry `t`.
#[device]
pub unsafe fn tile_run(tiles: TilePtrs, t: usize) -> (u32, u32) {
    let lo = unsafe { *tiles.start.add(t) };
    (lo, lo + unsafe { *tiles.len.add(t) })
}

/// The part of tile `t`'s run from its first view `>= k0` to its end, if that view is
/// `< k1`; the caller stops at the first view `>= k1`. `None` when no view of the stack
/// `k0 .. k1` touches the tile, so the tile's sums must be left untouched.
///
/// # Safety
///
/// As [`tile_run`], and the run must lie in `tiles.views`.
#[device]
pub unsafe fn tile_stack_run(tiles: TilePtrs, t: usize, k0: u32, k1: u32) -> Option<(u32, u32)> {
    let (mut lo, end) = unsafe { tile_run(tiles, t) };
    let mut hi = end;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if unsafe { *tiles.views.add(mid as usize) } < k0 {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == end || unsafe { *tiles.views.add(lo as usize) } >= k1 {
        return None;
    }
    Some((lo, end))
}

/// [`TilePtrs`] with the runs writable, for `bin_views` to append to: tile `t`'s slab is
/// `views[start[t] .. start[t + 1]]`, its run the first `len[t]` slots.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TileSlabs {
    pub start: *const u32,
    pub len: *mut u32,
    pub views: *mut u32,
}

/// First and last object tile, of `OBJ_TILE` pixels, holding an object pixel that a view's
/// samples read along one axis: offsets `o .. o + len`, and the bilinear `+ 1` neighbour of
/// the last when `f > 0`. The host tile lists and `bin_views` both bin by this rule.
#[device]
pub fn axis_tiles(o: u32, len: u32, f: f32) -> (u32, u32) {
    let last = if f > 0.0 { o + len } else { o + len - 1 };
    (o / OBJ_TILE, last / OBJ_TILE)
}

/// Log-likelihood sums over unmasked pixels of one frame, or of a chunk or column group of
/// one (PyNX `LLKAll`).
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
/// `(oy + pry + {0, 1}, ox + prx + {0, 1})` (PyNX `bilinear`, on an object sized so that it
/// never wraps). Corners whose weight is zero (`fy == 0` or `fx == 0`) are not read.
///
/// # Safety
///
/// `obj` must hold `[nobj, nyo, nxo]` with `v.obj < nobj`, `v.oy + pry + (v.fy > 0) < nyo`
/// and `v.ox + prx + (v.fx > 0) < nxo`.
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
    let (y1, x1) = (y0 + 1, x0 + 1);
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

/// Calls `f(pry, prx, weight)` for each probe pixel of view `v` whose bilinear sample reads
/// object pixel `(y, x)`, in [`bilinear_weights`] order, skipping corners outside the frame
/// or of zero weight. This is the exact transpose of [`sample_obj`]. A visitor, not an
/// array of sources: kernels kept such an array in local memory.
#[device]
pub fn obj_pixel_sources<F: FnMut(u32, u32, f32)>(
    v: ViewAddr,
    y: u32,
    x: u32,
    ny: u32,
    nx: u32,
    mut f: F,
) {
    // Row offset of `y` in the view (`y0 = oy + pry`), and of the row whose `+ 1`
    // neighbour is `y`; likewise for columns. An offset below zero wraps past every frame
    // side, so `pick` drops it.
    let r = y.wrapping_sub(v.oy);
    let c = x.wrapping_sub(v.ox);
    let (rm, cm) = (r.wrapping_sub(1), c.wrapping_sub(1));
    let w = bilinear_weights(v.fy, v.fx);
    let mut pick = |pry: u32, prx: u32, wt: f32, used: bool| {
        if used && pry < ny && prx < nx {
            f(pry, prx, wt);
        }
    };
    pick(r, c, w[0], true);
    pick(r, cm, w[1], v.fx != 0.0);
    pick(rm, c, w[2], v.fy != 0.0);
    pick(rm, cm, w[3], v.fy != 0.0 && v.fx != 0.0);
}
