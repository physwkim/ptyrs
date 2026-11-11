//! ptyrs-tile: cuTile (Tile IR) alternatives to `ptyrs-gpu` SIMT pieces.
//!
//! Each launch takes the same buffers as its `ptyrs_gpu` counterpart, enqueues on the
//! caller's stream and returns; like `ptyrs_gpu`, it leases every buffer it touches on that
//! stream until the kernel completes. cutile JIT-compiles a kernel on its first launch.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_buffer::DevicePointer;
use cuda_async::error::DeviceError;
use cuda_core::{AllocationLease, CudaStream, DeviceBuffer, DeviceCopy, RawDeviceBuffer};
use cutile::prelude::*;
use ptyrs_gpu::{C32, LlkPartial, OBJ_TILE, UpdateSums, Views};

#[cutile::module]
mod device {
    use cutile::core::*;

    unsafe fn load_i32(p: *const i32, i: i32) -> i32 {
        let base: PointerTile<*const i32, { [] }> = pointer_to_tile(p);
        let at: PointerTile<*const i32, { [] }> = base.offset(i);
        let (v, _t): (Tile<i32, { [] }>, Token) = unsafe {
            load_ptr_tko(
                at,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        let s: i32 = tile_to_scalar(v);
        s
    }

    unsafe fn load_f32(p: *const f32, i: i32) -> f32 {
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
        let at: PointerTile<*const f32, { [] }> = base.offset(i);
        let (v, _t): (Tile<f32, { [] }>, Token) = unsafe {
            load_ptr_tko(
                at,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        let s: f32 = tile_to_scalar(v);
        s
    }

    /// First index in `lo .. hi` of the ascending `list` whose value is `>= key`.
    unsafe fn lower_bound(list: *const i32, lo: i32, hi: i32, key: i32) -> i32 {
        let mut a: i32 = lo;
        let mut b: i32 = hi;
        while a < b {
            let mid: i32 = a + (b - a) / 2i32;
            let v: i32 = unsafe { load_i32(list, mid) };
            if v < key {
                a = mid + 1i32;
            } else {
                b = mid;
            }
        }
        a
    }

    /// `p[at]` where `m`; other lanes are unspecified.
    unsafe fn gather(
        p: *const f32,
        at: Tile<i32, { [16, 16] }>,
        m: Tile<bool, { [16, 16] }>,
    ) -> Tile<f32, { [16, 16] }> {
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*const f32, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*const f32, { [16, 16] }> = base.broadcast(shape![16, 16]);
        let ptrs: PointerTile<*const f32, { [16, 16] }> = base.offset_tile(at);
        // A padding value on a rank-2 pointer load fails tileiras; masked lanes are
        // discarded by the caller instead.
        let (v, _t): (Tile<f32, { [16, 16] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(m),
                None,
                None,
                Latency::<0>,
            )
        };
        v
    }

    unsafe fn scatter(
        p: *mut f32,
        at: Tile<i32, { [16, 16] }>,
        v: Tile<f32, { [16, 16] }>,
        m: Tile<bool, { [16, 16] }>,
    ) {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*mut f32, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*mut f32, { [16, 16] }> = base.broadcast(shape![16, 16]);
        let ptrs: PointerTile<*mut f32, { [16, 16] }> = base.offset_tile(at);
        let _t: Token = unsafe {
            store_ptr_tko(
                ptrs,
                v,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(m),
                None,
                Latency::<0>,
            )
        };
    }

    /// `ptyrs_kernels::psi_to_obj` with one tile program per object tile instead of one
    /// thread per pixel: program `(tx, ty, iobj)` owns pixels `[16 ty, 16 tx] + [0..16,
    /// 0..16]` of object mode `iobj`, walks the tile's views `k0 .. k0 + nk` in ascending
    /// order and, per view, its four bilinear sources in `obj_pixel_sources` order, with
    /// the SIMT kernel's arithmetic, then writes the pixels it owns. Complex buffers are
    /// interleaved `f32`; `views` is the `ViewAddr` records as words (`oy, ox, fy, fx,
    /// probe, obj`), read through `view_words` and `view_frac`.
    #[cutile::entry()]
    unsafe fn psi_to_obj(
        psi: *const f32,
        probe: *const f32,
        view_words: *const i32,
        view_frac: *const f32,
        tile_start: *const i32,
        tile_views: *const i32,
        num: *mut f32,
        den: *mut f32,
        k0: i32,
        nk: i32,
        ny: i32,
        nx: i32,
        nyo: i32,
        nxo: i32,
        quad: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();
        let iobj: i32 = pid.2;
        let tile: i32 = pid.1 * grid.0 + pid.0;
        let start: i32 = unsafe { load_i32(tile_start, tile) };
        let end: i32 = unsafe { load_i32(tile_start, tile + 1i32) };
        let lo: i32 = unsafe { lower_bound(tile_views, start, end, k0) };
        let hi: i32 = unsafe { lower_bound(tile_views, lo, end, k0 + nk) };

        let lane: Tile<i32, { [16] }> = iota(shape![16]);
        let row: Tile<i32, { [16, 1] }> = lane.reshape(shape![16, 1]);
        let col: Tile<i32, { [1, 16] }> = lane.reshape(shape![1, 16]);
        let ys: Tile<i32, { [16, 16] }> =
            row.broadcast(shape![16, 16]) + broadcast_scalar(pid.1 * 16i32, shape![16, 16]);
        let xs: Tile<i32, { [16, 16] }> =
            col.broadcast(shape![16, 16]) + broadcast_scalar(pid.0 * 16i32, shape![16, 16]);
        let ny_t: Tile<i32, { [16, 16] }> = broadcast_scalar(ny, shape![16, 16]);
        let nx_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nx, shape![16, 16]);
        let nyo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nyo, shape![16, 16]);
        let nxo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nxo, shape![16, 16]);
        let hy_t: Tile<i32, { [16, 16] }> = broadcast_scalar(ny / 2i32, shape![16, 16]);
        let hx_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nx / 2i32, shape![16, 16]);
        let one_i: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let two_i: Tile<i32, { [16, 16] }> = constant(2i32, shape![16, 16]);
        let zero_f: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
        let quad_t: Tile<f32, { [16, 16] }> = broadcast_scalar(quad, shape![16, 16]);

        let owned: Tile<bool, { [16, 16] }> = lt_tile(ys, nyo_t) & lt_tile(xs, nxo_t);
        let px: Tile<i32, { [16, 16] }> =
            (broadcast_scalar(iobj * nyo, shape![16, 16]) + ys) * nxo_t + xs;
        let px_re: Tile<i32, { [16, 16] }> = px * two_i;
        let px_im: Tile<i32, { [16, 16] }> = px_re + one_i;
        let num_in: *const f32 = cast_const(num);
        let den_in: *const f32 = cast_const(den);
        let mut acc_re: Tile<f32, { [16, 16] }> = unsafe { gather(num_in, px_re, owned) };
        let mut acc_im: Tile<f32, { [16, 16] }> = unsafe { gather(num_in, px_im, owned) };
        let mut norm: Tile<f32, { [16, 16] }> = unsafe { gather(den_in, px, owned) };

        let n: i32 = ny * nx;
        for j in lo..hi {
            let k: i32 = unsafe { load_i32(tile_views, j) };
            let w0: i32 = 6i32 * k;
            let obj: i32 = unsafe { load_i32(view_words, w0 + 5i32) };
            if obj != iobj {
                continue;
            }
            let oy: i32 = unsafe { load_i32(view_words, w0) };
            let ox: i32 = unsafe { load_i32(view_words, w0 + 1i32) };
            let fy: f32 = unsafe { load_f32(view_frac, w0 + 2i32) };
            let fx: f32 = unsafe { load_f32(view_frac, w0 + 3i32) };
            let iprobe: i32 = unsafe { load_i32(view_words, w0 + 4i32) };
            let fy_t: Tile<f32, { [16, 16] }> = broadcast_scalar(fy, shape![16, 16]);
            let fx_t: Tile<f32, { [16, 16] }> = broadcast_scalar(fx, shape![16, 16]);
            let gy_t: Tile<f32, { [16, 16] }> = broadcast_scalar(1.0f32 - fy, shape![16, 16]);
            let gx_t: Tile<f32, { [16, 16] }> = broadcast_scalar(1.0f32 - fx, shape![16, 16]);
            let fy_nz: Tile<bool, { [16, 16] }> = ne_tile(fy_t, zero_f);
            let fx_nz: Tile<bool, { [16, 16] }> = ne_tile(fx_t, zero_f);
            // Offset of each pixel in the view (`y0 = oy + pry`), as obj_pixel_sources.
            let r: Tile<i32, { [16, 16] }> =
                (ys + broadcast_scalar(nyo - oy, shape![16, 16])) % nyo_t;
            let c: Tile<i32, { [16, 16] }> =
                (xs + broadcast_scalar(nxo - ox, shape![16, 16])) % nxo_t;
            let plane: Tile<i32, { [16, 16] }> = broadcast_scalar((k - k0) * n, shape![16, 16]);
            let poff: Tile<i32, { [16, 16] }> = broadcast_scalar(iprobe * n, shape![16, 16]);
            // Sources in bilinear_weights order: `s = 2 sy + sx` reads the row `sy` and the
            // column `sx` before this pixel, with weight `(sy ? fy : 1 - fy) * (sx ? fx :
            // 1 - fx)`, and only where that factor's offset is non-zero.
            for s in 0i32..4i32 {
                let sy_t: Tile<i32, { [16, 16] }> = broadcast_scalar(s / 2i32, shape![16, 16]);
                let sx_t: Tile<i32, { [16, 16] }> = broadcast_scalar(s % 2i32, shape![16, 16]);
                let by: Tile<bool, { [16, 16] }> = eq_tile(sy_t, one_i);
                let bx: Tile<bool, { [16, 16] }> = eq_tile(sx_t, one_i);
                let pry: Tile<i32, { [16, 16] }> = (r + nyo_t - sy_t) % nyo_t;
                let prx: Tile<i32, { [16, 16] }> = (c + nxo_t - sx_t) % nxo_t;
                let wy: Tile<f32, { [16, 16] }> = select(by, fy_t, gy_t);
                let wx: Tile<f32, { [16, 16] }> = select(bx, fx_t, gx_t);
                let w: Tile<f32, { [16, 16] }> = wy * wx;
                let used: Tile<bool, { [16, 16] }> =
                    (ne_tile(sy_t, one_i) | fy_nz) & (ne_tile(sx_t, one_i) | fx_nz);
                let m: Tile<bool, { [16, 16] }> =
                    owned & used & lt_tile(pry, ny_t) & lt_tile(prx, nx_t);

                let pi: Tile<i32, { [16, 16] }> = (poff + pry * nx_t + prx) * two_i;
                let p_re: Tile<f32, { [16, 16] }> = unsafe { gather(probe, pi, m) };
                let p_im: Tile<f32, { [16, 16] }> = unsafe { gather(probe, pi + one_i, m) };
                let dy: Tile<i32, { [16, 16] }> = pry - hy_t;
                let dx: Tile<i32, { [16, 16] }> = prx - hx_t;
                let r2: Tile<f32, { [16, 16] }> = convert_tile(dy * dy + dx * dx);
                let a: Tile<f32, { [16, 16] }> = quad_t * r2;
                let q_re: Tile<f32, { [16, 16] }> = cos(a);
                let q_im: Tile<f32, { [16, 16] }> = sin(a);
                let si: Tile<i32, { [16, 16] }> =
                    (plane + ((pry + hy_t) % ny_t) * nx_t + (prx + hx_t) % nx_t) * two_i;
                let s_re: Tile<f32, { [16, 16] }> = unsafe { gather(psi, si, m) };
                let s_im: Tile<f32, { [16, 16] }> = unsafe { gather(psi, si + one_i, m) };

                let pq_re: Tile<f32, { [16, 16] }> = p_re * q_re - p_im * q_im;
                let pq_im: Tile<f32, { [16, 16] }> = p_re * q_im + p_im * q_re;
                let u_re: Tile<f32, { [16, 16] }> = pq_re * s_re + pq_im * s_im;
                let u_im: Tile<f32, { [16, 16] }> = pq_re * s_im - pq_im * s_re;
                acc_re = select(m, acc_re + w * u_re, acc_re);
                acc_im = select(m, acc_im + w * u_im, acc_im);
                norm = select(m, norm + w * (p_re * p_re + p_im * p_im), norm);
            }
        }
        unsafe { scatter(num, px_re, acc_re, owned) };
        unsafe { scatter(num, px_im, acc_im, owned) };
        unsafe { scatter(den, px, norm, owned) };
    }

    /// `p[at]` where `m`; other lanes are unspecified.
    unsafe fn gather_1d(
        p: *const f32,
        at: Tile<i32, { [1024] }>,
        m: Tile<bool, { [1024] }>,
    ) -> Tile<f32, { [1024] }> {
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [1024] }> = base.broadcast(shape![1024]);
        let ptrs: PointerTile<*const f32, { [1024] }> = base.offset_tile(at);
        let (v, _t): (Tile<f32, { [1024] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(m),
                None,
                None,
                Latency::<0>,
            )
        };
        v
    }

    /// `ptyrs_kernels::max_nonneg_f32`: `*out = max(*out, x[0 .. n] ...)` as `f32` bits
    /// over the non-negative values (negatives and NaN skipped), `*out` starting as a
    /// non-negative float. Program `i` takes `x[1024 i ..][.. 1024]`.
    #[cutile::entry()]
    unsafe fn max_nonneg_f32(x: *const f32, n: i32, out: *mut i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let lane: Tile<i32, { [1024] }> = iota(shape![1024]);
        let at: Tile<i32, { [1024] }> = lane + broadcast_scalar(pid.0 * 1024i32, shape![1024]);
        let inside: Tile<bool, { [1024] }> = lt_tile(at, broadcast_scalar(n, shape![1024]));
        let v: Tile<f32, { [1024] }> = unsafe { gather_1d(x, at, inside) };
        let zero: Tile<f32, { [1024] }> = constant(0.0f32, shape![1024]);
        // Lane-wise `if v > m { m = v }` from `m = 0`, as the SIMT loop: no NaN or negative
        // value survives, so the max-reduce below is exact in any order.
        let m: Tile<f32, { [1024] }> = select(inside & gt_tile(v, zero), v, zero);
        let top: Tile<f32, { [] }> = reduce_max(m, 0i32);
        // Non-negative floats order like their bit patterns.
        let bits: Tile<i32, { [] }> = bitcast(top);
        let bits: Tile<i32, { [1] }> = bits.reshape(shape![1]);
        let ptr: PointerTile<*mut i32, { [] }> = pointer_to_tile(out);
        let ptr: PointerTile<*mut i32, { [1] }> = ptr.reshape(shape![1]);
        let _r: (Tile<i32, { [1] }>, Token) = unsafe {
            atomic_rmw_tko(
                ptr,
                bits,
                atomic::Max,
                ordering::Relaxed,
                scope::Device,
                None,
                None,
            )
        };
    }

    /// `ptyrs_kernels::normalize_update`: `x = (num + reg x) / max(den + reg, 1e-12)` with
    /// `reg = inertia * *max`, for `n` complex values. Program `i` takes the interleaved
    /// floats `[1024 i ..][.. 1024]` of `x` and `num`.
    #[cutile::entry()]
    unsafe fn normalize_update(
        x: *mut f32,
        num: *const f32,
        den: *const f32,
        n: i32,
        max: *const f32,
        inertia: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let reg: f32 = inertia * unsafe { load_f32(max, 0i32) };
        let lane: Tile<i32, { [1024] }> = iota(shape![1024]);
        let at: Tile<i32, { [1024] }> = lane + broadcast_scalar(pid.0 * 1024i32, shape![1024]);
        let inside: Tile<bool, { [1024] }> = lt_tile(at, broadcast_scalar(2i32 * n, shape![1024]));
        let e: Tile<i32, { [1024] }> = at / constant(2i32, shape![1024]);
        let x_in: *const f32 = cast_const(x);
        let a: Tile<f32, { [1024] }> = unsafe { gather_1d(num, at, inside) };
        let b: Tile<f32, { [1024] }> = unsafe { gather_1d(x_in, at, inside) };
        let dv: Tile<f32, { [1024] }> = unsafe { gather_1d(den, e, inside) };
        let reg_t: Tile<f32, { [1024] }> = broadcast_scalar(reg, shape![1024]);
        let eps: Tile<f32, { [1024] }> = constant(1e-12f32, shape![1024]);
        // `f32::max` keeps the non-NaN operand: a NaN `den + reg` becomes `1e-12`.
        let s: Tile<f32, { [1024] }> = dv + reg_t;
        let d: Tile<f32, { [1024] }> = select(gt_tile(s, eps), s, eps);
        let r: Tile<f32, { [1024] }> = divf(a + reg_t * b, d, rounding::NearestEven, ftz::Disabled);
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(x);
        let base: PointerTile<*mut f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut f32, { [1024] }> = base.broadcast(shape![1024]);
        let ptrs: PointerTile<*mut f32, { [1024] }> = base.offset_tile(at);
        let _t: Token = unsafe {
            store_ptr_tko(
                ptrs,
                r,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(inside),
                None,
                Latency::<0>,
            )
        };
    }

    /// `LlkSums::from_partials` over `partials[0 .. n]` (four `f32` each: Poisson, Gaussian,
    /// Euclidean, calculated intensity): lane `t` adds term `t` of each partial in `f64`,
    /// in slice order, from zero, and `out[t]` takes the total.
    #[cutile::entry()]
    unsafe fn llk_sums(partials: *const f32, n: i32, out: *mut f64) {
        let term: Tile<i32, { [4] }> = iota(shape![4]);
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(partials);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [4] }> = base.broadcast(shape![4]);
        let mut acc: Tile<f64, { [4] }> = constant(0.0f64, shape![4]);
        for f in 0i32..n {
            let at: Tile<i32, { [4] }> = term + broadcast_scalar(4i32 * f, shape![4]);
            let ptrs: PointerTile<*const f32, { [4] }> = base.offset_tile(at);
            let (v, _t): (Tile<f32, { [4] }>, Token) = unsafe {
                load_ptr_tko(
                    ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                )
            };
            let v: Tile<f64, { [4] }> = convert_tile(v);
            acc = acc + v;
        }
        let dst: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let dst: PointerTile<*mut f64, { [1] }> = dst.reshape(shape![1]);
        let dst: PointerTile<*mut f64, { [4] }> = dst.broadcast(shape![4]);
        let dst: PointerTile<*mut f64, { [4] }> = dst.offset_tile(term);
        let _t: Token = unsafe {
            store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            )
        };
    }
}

#[derive(Debug)]
pub enum Error {
    /// A shape, context or driver failure, as `ptyrs_gpu` reports it.
    Gpu(ptyrs_gpu::Error),
    /// cutile failed to compile or launch a kernel.
    Tile(DeviceError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu(e) => e.fmt(f),
            Self::Tile(e) => write!(f, "cutile: {e:?}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<ptyrs_gpu::Error> for Error {
    fn from(e: ptyrs_gpu::Error) -> Self {
        Self::Gpu(e)
    }
}

impl From<cuda_core::DriverError> for Error {
    fn from(e: cuda_core::DriverError) -> Self {
        Self::Gpu(e.into())
    }
}

impl From<DeviceError> for Error {
    fn from(e: DeviceError) -> Self {
        Self::Tile(e)
    }
}

fn shape(what: String) -> Error {
    Error::Gpu(ptyrs_gpu::Error::Shape(what))
}

fn check_len<T: DeviceCopy>(what: &str, buf: &DeviceBuffer<T>, need: usize) -> Result<(), Error> {
    if buf.len() < need {
        return Err(shape(format!(
            "{what} holds {} elements, needs {need}",
            buf.len()
        )));
    }
    Ok(())
}

/// `v` as a Tile kernel's `i32` offset.
fn offset(what: &str, v: usize) -> Result<i32, Error> {
    i32::try_from(v).map_err(|_| shape(format!("{what} {v} exceeds the i32 offsets")))
}

/// Device address of `buf` as a kernel pointer to `U`.
fn dptr<T: DeviceCopy, U>(buf: &DeviceBuffer<T>) -> DevicePointer<U> {
    // SAFETY: a launch hands the kernel this address with a contract that keeps every
    // access inside `buf` (checked by its caller).
    unsafe { DevicePointer::from_cu_deviceptr(buf.cu_deviceptr()) }
}

/// Leases `reads` and `writes` on `stream`, runs `enqueue`, and keeps the leases until the
/// work it enqueued completes (as `ptyrs_gpu::Kernels` does for its launches).
fn launch(
    stream: &Arc<CudaStream>,
    reads: &[&RawDeviceBuffer],
    writes: &[&RawDeviceBuffer],
    enqueue: impl FnOnce() -> Result<(), Error>,
) -> Result<(), Error> {
    let mut leases: Vec<AllocationLease> = Vec::with_capacity(reads.len() + writes.len());
    for (bufs, write) in [(reads, false), (writes, true)] {
        for buf in bufs {
            if buf.context().as_ref() != stream.context().as_ref() {
                return Err(ptyrs_gpu::Error::ContextMismatch.into());
            }
            leases.push(buf.acquire(stream, write, false)?);
        }
    }
    enqueue()?;
    for lease in leases {
        lease.retain_until_complete(stream)?;
    }
    Ok(())
}

/// [`ptyrs_gpu::Kernels::psi_to_obj`] as a Tile kernel: adds the stack's object-update sums
/// to `sums` `[nobj, nyo, nxo]`, each object pixel adding its views in ascending view
/// order from the same tile lists, so the result does not depend on the stack split.
pub fn psi_to_obj(
    stream: &Arc<CudaStream>,
    views: &Views,
    frames: Range<u32>,
    psi: &DeviceBuffer<C32>,
    probe: &DeviceBuffer<C32>,
    sums: &mut UpdateSums,
) -> Result<(), Error> {
    let d = views.dims();
    if frames.start >= frames.end || frames.end > d.nframe {
        return Err(shape(format!(
            "frames {frames:?} is not a non-empty part of 0..{}",
            d.nframe
        )));
    }
    let [ny, nx] = d.frame;
    let [nyo, nxo] = d.obj;
    let n = ny as usize * nx as usize;
    let (k0, nk) = (
        frames.start * d.nmode,
        (frames.end - frames.start) * d.nmode,
    );
    let obj_len = d.nobj as usize * nyo as usize * nxo as usize;
    check_len("psi", psi, nk as usize * n)?;
    check_len("probe", probe, d.nprobe as usize * n)?;
    check_len("sums.num", &sums.num, obj_len)?;
    check_len("sums.den", &sums.den, obj_len)?;
    // Every offset the kernel forms, in f32 or ViewAddr-word units, is an i32.
    offset("psi length", 2 * nk as usize * n)?;
    offset("probe length", 2 * d.nprobe as usize * n)?;
    offset("object length", 2 * obj_len)?;
    offset("view words", 6 * views.addr().len())?;
    offset("tile list length", views.tile_views().len())?;
    let geom = [k0, nk, ny, nx, nyo, nxo]
        .iter()
        .map(|&v| offset("geometry", v as usize))
        .collect::<Result<Vec<_>, _>>()?;
    let reads = [
        psi.storage(),
        probe.storage(),
        views.addr().storage(),
        views.tile_start().storage(),
        views.tile_views().storage(),
    ];
    let UpdateSums { num, den } = sums;
    launch(stream, &reads, &[num.storage(), den.storage()], || {
        let grid = (nxo.div_ceil(OBJ_TILE), nyo.div_ceil(OBJ_TILE), d.nobj);
        // SAFETY: lengths and i32 offsets checked above; `Views::new` checked every view
        // lies inside the object with valid mode indices and built the tile lists from
        // them; the leases order every access on `stream`, and each program writes only
        // the pixels of its own tile.
        unsafe {
            device::psi_to_obj(
                dptr(psi),
                dptr(probe),
                dptr(views.addr()),
                dptr(views.addr()),
                dptr(views.tile_start()),
                dptr(views.tile_views()),
                dptr(num),
                dptr(den),
                geom[0],
                geom[1],
                geom[2],
                geom[3],
                geom[4],
                geom[5],
                d.quad_phase,
            )
            .grid(grid)
            .async_on(stream)?;
        }
        Ok(())
    })
}

/// [`ptyrs_gpu::Kernels::update_from_sums`] as Tile kernels: `x = (num + reg x) /
/// max(den + reg, 1e-12)` for the first `n` values, `reg = inertia * max(0, den[0 .. n]
/// ...)` over the non-NaN values of `den`.
pub fn update_from_sums(
    stream: &Arc<CudaStream>,
    x: &mut DeviceBuffer<C32>,
    sums: &UpdateSums,
    n: u32,
    inertia: f32,
) -> Result<(), Error> {
    check_len("x", x, n as usize)?;
    check_len("sums.num", &sums.num, n as usize)?;
    check_len("sums.den", &sums.den, n as usize)?;
    // The kernels index interleaved floats, `2 n` of them, as i32.
    let n = offset("n", n as usize)?;
    offset("2 n", 2 * n as usize)?;
    if n == 0 {
        return Ok(());
    }
    let max_bits = DeviceBuffer::<u32>::zeroed(stream, 1)?;
    let UpdateSums { num, den } = sums;
    launch(stream, &[den.storage()], &[max_bits.storage()], || {
        // SAFETY: `den` holds `n` floats, `max_bits` one zeroed u32.
        unsafe {
            device::max_nonneg_f32(dptr(den), n, dptr(&max_bits))
                .grid(((n as u32).div_ceil(1024), 1, 1))
                .async_on(stream)?;
        }
        Ok(())
    })?;
    let reads = [num.storage(), den.storage(), max_bits.storage()];
    launch(stream, &reads, &[x.storage()], || {
        // SAFETY: `x`, `num` hold `n` complex values, `den` `n` floats, `max_bits` one.
        unsafe {
            device::normalize_update(dptr(x), dptr(num), dptr(den), n, dptr(&max_bits), inertia)
                .grid(((2 * n as u32).div_ceil(1024), 1, 1))
                .async_on(stream)?;
        }
        Ok(())
    })
}

/// [`ptyrs_gpu::LlkSums::from_partials`] of `partials[0 .. n]` on the device, into `out` as
/// `[poisson, gaussian, euclidean, icalc]`: the same `f64` adds in the same (slice) order,
/// so the totals equal the host's bit for bit. One program; `n` is a frame count.
pub fn llk_sums(
    stream: &Arc<CudaStream>,
    partials: &DeviceBuffer<LlkPartial>,
    n: u32,
    out: &mut DeviceBuffer<f64>,
) -> Result<(), Error> {
    check_len("partials", partials, n as usize)?;
    check_len("out", out, 4)?;
    // The kernel indexes the partials' floats, `4 n` of them, as i32.
    let n = offset("n", n as usize)?;
    offset("4 n", 4 * n as usize)?;
    launch(stream, &[partials.storage()], &[out.storage()], || {
        // SAFETY: `partials` holds `n` records of four f32, `out` four f64.
        unsafe {
            device::llk_sums(dptr(partials), n, dptr(out))
                .grid((1, 1, 1))
                .async_on(stream)?;
        }
        Ok(())
    })
}
