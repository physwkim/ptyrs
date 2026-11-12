//! ptyrs::tile: cuTile (Tile IR) alternatives to `ptyrs::gpu` SIMT pieces.
//!
//! The engine builds [`TileDots`] alone, BH's dot products, which have no SIMT form, so it
//! compiles only the Tile kernels it launches. The other sets are the Tile forms of kernels
//! the engine runs in SIMT, built by this crate's tests and examples to compare them:
//! [`TileKernels`], the kernels of one scan, its [`TileObjUpdate`] and [`TileProbeUpdate`],
//! and [`TileCgDots`].
//!
//! Every launch is a method of a set, each compiling its kernels when it is built, so no
//! launch compiles, eagerly or under a graph capture. A launch takes the context, stack and
//! buffers of its `crate::gpu` counterpart, the scan's views being the set's (the object
//! update takes them, as its counterpart does), leases the buffers with
//! [`crate::gpu::lease`] and enqueues on the context's stream, so it runs eagerly or records
//! into a capture the same way.

use std::sync::Arc;

pub use crate::gpu::Error;
use crate::gpu::{
    BhDisp, C32, CG_BLOCKS, LLK_TREE_LEAVES, LlkPartial, StackGeom, StackRef, StackTable,
    UpdateSums, ViewDims, Views, gamma_parts, lease,
};
use cuda_async::device_buffer::DevicePointer;
use cuda_core::sys::CUdeviceptr;
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use cutile::prelude::*;

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
    unsafe fn gather<const R: i32, const C: i32>(
        p: *const f32,
        at: Tile<i32, { [R, C] }>,
        m: Tile<bool, { [R, C] }>,
    ) -> Tile<f32, { [R, C] }> {
        let (v, _t): (Tile<f32, { [R, C] }>, Token) = unsafe { gather_tok(p, at, m) };
        v
    }

    /// [`gather`] and the load's token, which a store to the same pixels must take: Tile IR
    /// orders memory operations only through tokens and data dependencies.
    unsafe fn gather_tok<const R: i32, const C: i32>(
        p: *const f32,
        at: Tile<i32, { [R, C] }>,
        m: Tile<bool, { [R, C] }>,
    ) -> (Tile<f32, { [R, C] }>, Token) {
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*const f32, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*const f32, { [R, C] }> = base.broadcast(shape![R, C]);
        let ptrs: PointerTile<*const f32, { [R, C] }> = base.offset_tile(at);
        // A padding value on a rank-2 pointer load fails tileiras; masked lanes are
        // discarded by the caller instead.
        let (v, t): (Tile<f32, { [R, C] }>, Token) = unsafe {
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
        (v, t)
    }

    /// `(p[at], p[at + 1])` where `m`, as one load of `[R, C, 2]` so each lane reads its
    /// complex value in one access; other lanes are unspecified.
    unsafe fn gather_c<const R: i32, const C: i32>(
        p: *const f32,
        at: Tile<i32, { [R, C] }>,
        m: Tile<bool, { [R, C] }>,
    ) -> (Tile<f32, { [R, C] }>, Tile<f32, { [R, C] }>) {
        let (re, im, _t): (Tile<f32, { [R, C] }>, Tile<f32, { [R, C] }>, Token) =
            unsafe { gather_c_tok(p, at, m) };
        (re, im)
    }

    /// [`gather_c`] and the load's token, as [`gather_tok`].
    unsafe fn gather_c_tok<const R: i32, const C: i32>(
        p: *const f32,
        at: Tile<i32, { [R, C] }>,
        m: Tile<bool, { [R, C] }>,
    ) -> (Tile<f32, { [R, C] }>, Tile<f32, { [R, C] }>, Token) {
        let part: Tile<i32, { [2] }> = iota(shape![2]);
        let part: Tile<i32, { [1, 1, 2] }> = part.reshape(shape![1, 1, 2]);
        let part: Tile<i32, { [R, C, 2] }> = part.broadcast(shape![R, C, 2]);
        let at: Tile<i32, { [R, C, 1] }> = at.reshape(shape![R, C, 1]);
        let at: Tile<i32, { [R, C, 2] }> = at.broadcast(shape![R, C, 2]) + part;
        let m: Tile<bool, { [R, C, 1] }> = m.reshape(shape![R, C, 1]);
        let m: Tile<bool, { [R, C, 2] }> = m.broadcast(shape![R, C, 2]);
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*const f32, { [1, 1, 1] }> = base.reshape(shape![1, 1, 1]);
        let base: PointerTile<*const f32, { [R, C, 2] }> = base.broadcast(shape![R, C, 2]);
        let ptrs: PointerTile<*const f32, { [R, C, 2] }> = base.offset_tile(at);
        let (v, t): (Tile<f32, { [R, C, 2] }>, Token) = unsafe {
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
        let z: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let o: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let re: Tile<f32, { [R, C, 1] }> = extract(v, [z, z, z]);
        let im: Tile<f32, { [R, C, 1] }> = extract(v, [z, z, o]);
        (re.reshape(shape![R, C]), im.reshape(shape![R, C]), t)
    }

    unsafe fn scatter<const R: i32, const C: i32>(
        p: *mut f32,
        at: Tile<i32, { [R, C] }>,
        v: Tile<f32, { [R, C] }>,
        m: Tile<bool, { [R, C] }>,
    ) {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*mut f32, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*mut f32, { [R, C] }> = base.broadcast(shape![R, C]);
        let ptrs: PointerTile<*mut f32, { [R, C] }> = base.offset_tile(at);
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

    /// `p[at] = 0` where `m`, ordered after the operation that returned `after`.
    unsafe fn zero_after<const R: i32, const C: i32>(
        p: *mut f32,
        at: Tile<i32, { [R, C] }>,
        m: Tile<bool, { [R, C] }>,
        after: Token,
    ) {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*mut f32, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*mut f32, { [R, C] }> = base.broadcast(shape![R, C]);
        let ptrs: PointerTile<*mut f32, { [R, C] }> = base.offset_tile(at);
        let zero: Tile<f32, { [R, C] }> = constant(0.0f32, shape![R, C]);
        let _t: Token = unsafe {
            store_ptr_tko(
                ptrs,
                zero,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(m),
                Some(after),
                Latency::<0>,
            )
        };
    }

    /// `ptyrs_kernels::pixel::stack_frames`: `(f0, nf)` of the `StackCtl` words at `ctl`,
    /// clamped to `nf <= nf_max` and `f0 + nf <= nframe`. The words are `u32`, so a
    /// negative one lies above every bound.
    unsafe fn stack_frames(ctl: *const i32, nframe: i32, nf_max: i32) -> (i32, i32) {
        let mut f0: i32 = unsafe { load_i32(ctl, 0i32) };
        let mut nf: i32 = unsafe { load_i32(ctl, 1i32) };
        if f0 < 0i32 {
            f0 = nframe;
        }
        if nf < 0i32 {
            nf = nf_max;
        }
        let f0: i32 = min(f0, nframe);
        let nf: i32 = min(min(nf, nf_max), nframe - f0);
        (f0, nf)
    }

    /// Object tile `ty * ntx + tx` of program `(tx, ty, ..)` in a `(ntx, nty, ..)` grid, and
    /// the rows and columns of its pixels `[16 ty, 16 tx] + [0..16, 0..16]`.
    fn obj_tile() -> (i32, Tile<i32, { [16, 16] }>, Tile<i32, { [16, 16] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();
        let lane: Tile<i32, { [16] }> = iota(shape![16]);
        let row: Tile<i32, { [16, 1] }> = lane.reshape(shape![16, 1]);
        let col: Tile<i32, { [1, 16] }> = lane.reshape(shape![1, 16]);
        let ys: Tile<i32, { [16, 16] }> =
            row.broadcast(shape![16, 16]) + broadcast_scalar(pid.1 * 16i32, shape![16, 16]);
        let xs: Tile<i32, { [16, 16] }> =
            col.broadcast(shape![16, 16]) + broadcast_scalar(pid.0 * 16i32, shape![16, 16]);
        (pid.1 * grid.0 + pid.0, ys, xs)
    }

    /// `ptyrs_kernels::psi_to_obj` with one tile program per object tile instead of one
    /// thread per pixel: program `(tx, ty, iobj)` owns pixels `[16 ty, 16 tx] + [0..16,
    /// 0..16]` of object mode `iobj`, walks the views of the stack at `ctl` in the tile's
    /// run in ascending order and, per view, its four bilinear sources in
    /// `obj_pixel_sources` order, with the SIMT kernel's arithmetic, then writes the pixels
    /// it owns. A tile whose run holds no view of the stack is neither read nor written.
    /// Complex buffers are interleaved `f32`; `views` is the `ViewAddr` records as words
    /// (`oy, ox, fy, fx, probe, obj`), read through `view_words` and `view_frac`.
    #[cutile::entry()]
    unsafe fn psi_to_obj(
        psi: *const f32,
        probe: *const f32,
        view_words: *const i32,
        view_frac: *const f32,
        tile_start: *const i32,
        tile_len: *const i32,
        tile_views: *const i32,
        num: *mut f32,
        den: *mut f32,
        ctl: *const i32,
        nframe: i32,
        nf_max: i32,
        nmode: i32,
        ny: i32,
        nx: i32,
        nyo: i32,
        nxo: i32,
        quad: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let iobj: i32 = pid.2;
        let (tile, ys, xs): (i32, Tile<i32, { [16, 16] }>, Tile<i32, { [16, 16] }>) = obj_tile();
        let (f0, nf): (i32, i32) = unsafe { stack_frames(ctl, nframe, nf_max) };
        let k0: i32 = f0 * nmode;
        let start: i32 = unsafe { load_i32(tile_start, tile) };
        let end: i32 = start + unsafe { load_i32(tile_len, tile) };
        let lo: i32 = unsafe { lower_bound(tile_views, start, end, k0) };
        let hi: i32 = unsafe { lower_bound(tile_views, lo, end, (f0 + nf) * nmode) };
        if lo < hi {
            let ny_t: Tile<i32, { [16, 16] }> = broadcast_scalar(ny, shape![16, 16]);
            let nx_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nx, shape![16, 16]);
            let nyo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nyo, shape![16, 16]);
            let nxo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nxo, shape![16, 16]);
            let hy_t: Tile<i32, { [16, 16] }> = broadcast_scalar(ny / 2i32, shape![16, 16]);
            let hx_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nx / 2i32, shape![16, 16]);
            let zero_i: Tile<i32, { [16, 16] }> = constant(0i32, shape![16, 16]);
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
                // Offset of each pixel in the view (`y0 = oy + pry`), as obj_pixel_sources;
                // nothing wraps, so a negative offset is no source.
                let r: Tile<i32, { [16, 16] }> = ys - broadcast_scalar(oy, shape![16, 16]);
                let c: Tile<i32, { [16, 16] }> = xs - broadcast_scalar(ox, shape![16, 16]);
                let plane: Tile<i32, { [16, 16] }> = broadcast_scalar((k - k0) * n, shape![16, 16]);
                let poff: Tile<i32, { [16, 16] }> = broadcast_scalar(iprobe * n, shape![16, 16]);
                // Sources in bilinear_weights order: `s = 2 sy + sx` reads the row `sy` and
                // the column `sx` before this pixel, with weight `(sy ? fy : 1 - fy) * (sx ?
                // fx : 1 - fx)`, and only where that factor's offset is non-zero.
                for s in 0i32..4i32 {
                    let sy_t: Tile<i32, { [16, 16] }> = broadcast_scalar(s / 2i32, shape![16, 16]);
                    let sx_t: Tile<i32, { [16, 16] }> = broadcast_scalar(s % 2i32, shape![16, 16]);
                    let by: Tile<bool, { [16, 16] }> = eq_tile(sy_t, one_i);
                    let bx: Tile<bool, { [16, 16] }> = eq_tile(sx_t, one_i);
                    let pry: Tile<i32, { [16, 16] }> = r - sy_t;
                    let prx: Tile<i32, { [16, 16] }> = c - sx_t;
                    let wy: Tile<f32, { [16, 16] }> = select(by, fy_t, gy_t);
                    let wx: Tile<f32, { [16, 16] }> = select(bx, fx_t, gx_t);
                    let w: Tile<f32, { [16, 16] }> = wy * wx;
                    let used: Tile<bool, { [16, 16] }> =
                        (ne_tile(sy_t, one_i) | fy_nz) & (ne_tile(sx_t, one_i) | fx_nz);
                    let inside: Tile<bool, { [16, 16] }> = ge_tile(pry, zero_i)
                        & lt_tile(pry, ny_t)
                        & ge_tile(prx, zero_i)
                        & lt_tile(prx, nx_t);
                    let m: Tile<bool, { [16, 16] }> = owned & used & inside;

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
    }

    /// `ptyrs_kernels::psi_to_probe` with one tile program per probe tile: program `(tx, ty,
    /// mode)` owns pixels `[8 ty, 32 tx] + [0..8, 0..32]` of probe mode `mode`, walks the
    /// views of the stack at `ctl` in ascending order, skipping those of other modes, and
    /// adds each with the SIMT kernel's arithmetic (`sample_obj`'s corner order), then
    /// writes the pixels it owns if it added a view. Buffers as in `psi_to_obj`.
    #[cutile::entry()]
    unsafe fn psi_to_probe(
        psi: *const f32,
        obj: *const f32,
        view_words: *const i32,
        view_frac: *const f32,
        num: *mut f32,
        den: *mut f32,
        ctl: *const i32,
        nframe: i32,
        nf_max: i32,
        nmode: i32,
        ny: i32,
        nx: i32,
        nyo: i32,
        nxo: i32,
        quad: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mode: i32 = pid.2;
        let (f0, nf): (i32, i32) = unsafe { stack_frames(ctl, nframe, nf_max) };
        let k0: i32 = f0 * nmode;
        let nk: i32 = nf * nmode;
        // Rows of 32 pixels make each view's psi and object reads 256-byte runs; the
        // 64-byte runs of 16 x 16 tiles left the kernel at two thirds of the bandwidth.
        let row: Tile<i32, { [8] }> = iota(shape![8]);
        let row: Tile<i32, { [8, 1] }> = row.reshape(shape![8, 1]);
        let col: Tile<i32, { [32] }> = iota(shape![32]);
        let col: Tile<i32, { [1, 32] }> = col.reshape(shape![1, 32]);
        let ys: Tile<i32, { [8, 32] }> =
            row.broadcast(shape![8, 32]) + broadcast_scalar(pid.1 * 8i32, shape![8, 32]);
        let xs: Tile<i32, { [8, 32] }> =
            col.broadcast(shape![8, 32]) + broadcast_scalar(pid.0 * 32i32, shape![8, 32]);
        let ny_t: Tile<i32, { [8, 32] }> = broadcast_scalar(ny, shape![8, 32]);
        let nx_t: Tile<i32, { [8, 32] }> = broadcast_scalar(nx, shape![8, 32]);
        let nxo_t: Tile<i32, { [8, 32] }> = broadcast_scalar(nxo, shape![8, 32]);
        let hy_t: Tile<i32, { [8, 32] }> = broadcast_scalar(ny / 2i32, shape![8, 32]);
        let hx_t: Tile<i32, { [8, 32] }> = broadcast_scalar(nx / 2i32, shape![8, 32]);
        let one_i: Tile<i32, { [8, 32] }> = constant(1i32, shape![8, 32]);
        let two_i: Tile<i32, { [8, 32] }> = constant(2i32, shape![8, 32]);

        let owned: Tile<bool, { [8, 32] }> = lt_tile(ys, ny_t) & lt_tile(xs, nx_t);
        let px: Tile<i32, { [8, 32] }> =
            (broadcast_scalar(mode * ny, shape![8, 32]) + ys) * nx_t + xs;
        let px_re: Tile<i32, { [8, 32] }> = px * two_i;
        let px_im: Tile<i32, { [8, 32] }> = px_re + one_i;
        let num_in: *const f32 = cast_const(num);
        let den_in: *const f32 = cast_const(den);
        let mut acc_re: Tile<f32, { [8, 32] }> = unsafe { gather(num_in, px_re, owned) };
        let mut acc_im: Tile<f32, { [8, 32] }> = unsafe { gather(num_in, px_im, owned) };
        let mut norm: Tile<f32, { [8, 32] }> = unsafe { gather(den_in, px, owned) };

        // `quad_phase` and `psi_index` of each pixel, the same for every view.
        let dy: Tile<i32, { [8, 32] }> = ys - hy_t;
        let dx: Tile<i32, { [8, 32] }> = xs - hx_t;
        let r2: Tile<f32, { [8, 32] }> = convert_tile(dy * dy + dx * dx);
        let a: Tile<f32, { [8, 32] }> = broadcast_scalar(quad, shape![8, 32]) * r2;
        let q_re: Tile<f32, { [8, 32] }> = cos(a);
        let q_im: Tile<f32, { [8, 32] }> = sin(a);
        let at: Tile<i32, { [8, 32] }> = ((ys + hy_t) % ny_t) * nx_t + (xs + hx_t) % nx_t;

        let n: i32 = ny * nx;
        let mut added: i32 = 0i32;
        for kl in 0i32..nk {
            let w0: i32 = 6i32 * (k0 + kl);
            let iprobe: i32 = unsafe { load_i32(view_words, w0 + 4i32) };
            if iprobe != mode {
                continue;
            }
            added = 1i32;
            let oy: i32 = unsafe { load_i32(view_words, w0) };
            let ox: i32 = unsafe { load_i32(view_words, w0 + 1i32) };
            let fy: f32 = unsafe { load_f32(view_frac, w0 + 2i32) };
            let fx: f32 = unsafe { load_f32(view_frac, w0 + 3i32) };
            let iobj: i32 = unsafe { load_i32(view_words, w0 + 5i32) };
            // `bilinear_weights`, and which corners `sample_obj` reads.
            let wt0: Tile<f32, { [8, 32] }> =
                broadcast_scalar((1.0f32 - fy) * (1.0f32 - fx), shape![8, 32]);
            let wt1: Tile<f32, { [8, 32] }> = broadcast_scalar((1.0f32 - fy) * fx, shape![8, 32]);
            let wt2: Tile<f32, { [8, 32] }> = broadcast_scalar(fy * (1.0f32 - fx), shape![8, 32]);
            let wt3: Tile<f32, { [8, 32] }> = broadcast_scalar(fy * fx, shape![8, 32]);
            let zero_f: Tile<f32, { [8, 32] }> = constant(0.0f32, shape![8, 32]);
            let fy_nz: Tile<bool, { [8, 32] }> =
                ne_tile(broadcast_scalar(fy, shape![8, 32]), zero_f);
            let fx_nz: Tile<bool, { [8, 32] }> =
                ne_tile(broadcast_scalar(fx, shape![8, 32]), zero_f);
            let m01: Tile<bool, { [8, 32] }> = owned & fx_nz;
            let m10: Tile<bool, { [8, 32] }> = owned & fy_nz;
            let m11: Tile<bool, { [8, 32] }> = m10 & fx_nz;

            let y0: Tile<i32, { [8, 32] }> = broadcast_scalar(oy, shape![8, 32]) + ys;
            let x0: Tile<i32, { [8, 32] }> = broadcast_scalar(ox, shape![8, 32]) + xs;
            let y1: Tile<i32, { [8, 32] }> = y0 + one_i;
            let x1: Tile<i32, { [8, 32] }> = x0 + one_i;
            let plane: Tile<i32, { [8, 32] }> = broadcast_scalar(iobj * nyo, shape![8, 32]);
            let i00: Tile<i32, { [8, 32] }> = ((plane + y0) * nxo_t + x0) * two_i;
            let i01: Tile<i32, { [8, 32] }> = ((plane + y0) * nxo_t + x1) * two_i;
            let i10: Tile<i32, { [8, 32] }> = ((plane + y1) * nxo_t + x0) * two_i;
            let i11: Tile<i32, { [8, 32] }> = ((plane + y1) * nxo_t + x1) * two_i;

            let (c_re, c_im): (Tile<f32, { [8, 32] }>, Tile<f32, { [8, 32] }>) =
                unsafe { gather_c(obj, i00, owned) };
            let o_re: Tile<f32, { [8, 32] }> = wt0 * c_re;
            let o_im: Tile<f32, { [8, 32] }> = wt0 * c_im;
            let (c_re, c_im): (Tile<f32, { [8, 32] }>, Tile<f32, { [8, 32] }>) =
                unsafe { gather_c(obj, i01, m01) };
            let o_re: Tile<f32, { [8, 32] }> = select(fx_nz, o_re + wt1 * c_re, o_re);
            let o_im: Tile<f32, { [8, 32] }> = select(fx_nz, o_im + wt1 * c_im, o_im);
            let (c_re, c_im): (Tile<f32, { [8, 32] }>, Tile<f32, { [8, 32] }>) =
                unsafe { gather_c(obj, i10, m10) };
            let o_re: Tile<f32, { [8, 32] }> = select(fy_nz, o_re + wt2 * c_re, o_re);
            let o_im: Tile<f32, { [8, 32] }> = select(fy_nz, o_im + wt2 * c_im, o_im);
            let (c_re, c_im): (Tile<f32, { [8, 32] }>, Tile<f32, { [8, 32] }>) =
                unsafe { gather_c(obj, i11, m11) };
            let both: Tile<bool, { [8, 32] }> = fy_nz & fx_nz;
            let o_re: Tile<f32, { [8, 32] }> = select(both, o_re + wt3 * c_re, o_re);
            let o_im: Tile<f32, { [8, 32] }> = select(both, o_im + wt3 * c_im, o_im);

            let si: Tile<i32, { [8, 32] }> = (broadcast_scalar(kl * n, shape![8, 32]) + at) * two_i;
            let (s_re, s_im): (Tile<f32, { [8, 32] }>, Tile<f32, { [8, 32] }>) =
                unsafe { gather_c(psi, si, owned) };
            let oq_re: Tile<f32, { [8, 32] }> = o_re * q_re - o_im * q_im;
            let oq_im: Tile<f32, { [8, 32] }> = o_re * q_im + o_im * q_re;
            acc_re = acc_re + (oq_re * s_re + oq_im * s_im);
            acc_im = acc_im + (oq_re * s_im - oq_im * s_re);
            norm = norm + (o_re * o_re + o_im * o_im);
        }
        // As the SIMT kernel: a mode without a view in the stack is not written.
        if added != 0i32 {
            unsafe { scatter(num, px_re, acc_re, owned) };
            unsafe { scatter(num, px_im, acc_im, owned) };
            unsafe { scatter(den, px, norm, owned) };
        }
    }

    /// `p[at]` where `m`; other lanes are unspecified.
    unsafe fn gather_1d(
        p: *const f32,
        at: Tile<i32, { [1024] }>,
        m: Tile<bool, { [1024] }>,
    ) -> Tile<f32, { [1024] }> {
        let (v, _t): (Tile<f32, { [1024] }>, Token) = unsafe { gather_1d_tok(p, at, m) };
        v
    }

    /// [`gather_1d`] and the load's token, as [`gather_tok`].
    unsafe fn gather_1d_tok(
        p: *const f32,
        at: Tile<i32, { [1024] }>,
        m: Tile<bool, { [1024] }>,
    ) -> (Tile<f32, { [1024] }>, Token) {
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [1024] }> = base.broadcast(shape![1024]);
        let ptrs: PointerTile<*const f32, { [1024] }> = base.offset_tile(at);
        let (v, t): (Tile<f32, { [1024] }>, Token) = unsafe {
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
        (v, t)
    }

    /// [`zero_after`] of a `[1024]` tile.
    unsafe fn zero_1d_after(
        p: *mut f32,
        at: Tile<i32, { [1024] }>,
        m: Tile<bool, { [1024] }>,
        after: Token,
    ) {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*mut f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut f32, { [1024] }> = base.broadcast(shape![1024]);
        let ptrs: PointerTile<*mut f32, { [1024] }> = base.offset_tile(at);
        let zero: Tile<f32, { [1024] }> = constant(0.0f32, shape![1024]);
        let _t: Token = unsafe {
            store_ptr_tko(
                ptrs,
                zero,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(m),
                Some(after),
                Latency::<0>,
            )
        };
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
        unsafe { max_into(out, top) };
    }

    /// `*out = max(*out, top)` as `f32` bits, `top` and `*out` non-negative floats, which
    /// order like their bit patterns.
    unsafe fn max_into(out: *mut i32, top: Tile<f32, { [] }>) {
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
    /// `reg = inertia * *max`, for `n` complex values, then zeroes the `num` and `den` it
    /// consumed. Program `i` takes the interleaved floats `[1024 i ..][.. 1024]` of `x` and
    /// `num`, and the `den` values `[512 i ..][.. 512]`.
    #[cutile::entry()]
    unsafe fn normalize_update(
        x: *mut f32,
        num: *mut f32,
        den: *mut f32,
        n: i32,
        max: *const f32,
        inertia: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let reg: f32 = inertia * unsafe { load_f32(max, 0i32) };
        let lane: Tile<i32, { [1024] }> = iota(shape![1024]);
        let at: Tile<i32, { [1024] }> = lane + broadcast_scalar(pid.0 * 1024i32, shape![1024]);
        let inside: Tile<bool, { [1024] }> = lt_tile(at, broadcast_scalar(2i32 * n, shape![1024]));
        let two: Tile<i32, { [1024] }> = constant(2i32, shape![1024]);
        let e: Tile<i32, { [1024] }> = at / two;
        let x_in: *const f32 = cast_const(x);
        let (a, num_t): (Tile<f32, { [1024] }>, Token) =
            unsafe { gather_1d_tok(cast_const(num), at, inside) };
        let b: Tile<f32, { [1024] }> = unsafe { gather_1d(x_in, at, inside) };
        let (dv, den_t): (Tile<f32, { [1024] }>, Token) =
            unsafe { gather_1d_tok(cast_const(den), e, inside) };
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
        // Each `den` value is read by the lanes of both its floats and zeroed by the even one.
        let even: Tile<bool, { [1024] }> = eq_tile(e * two, at);
        unsafe { zero_1d_after(num, at, inside, num_t) };
        unsafe { zero_1d_after(den, e, inside & even, den_t) };
    }

    /// `ptyrs_kernels::max_den_tiles`: `max_nonneg_f32` of `den` `[nobj, nyo, nxo]` over the
    /// pixels of the object tiles with views, one program per tile and object mode; a
    /// program whose tile has none reads nothing.
    #[cutile::entry()]
    unsafe fn max_den_tiles(
        den: *const f32,
        tile_len: *const i32,
        out: *mut i32,
        nyo: i32,
        nxo: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let (tile, ys, xs): (i32, Tile<i32, { [16, 16] }>, Tile<i32, { [16, 16] }>) = obj_tile();
        let len: i32 = unsafe { load_i32(tile_len, tile) };
        if len != 0i32 {
            let nyo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nyo, shape![16, 16]);
            let nxo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nxo, shape![16, 16]);
            let owned: Tile<bool, { [16, 16] }> = lt_tile(ys, nyo_t) & lt_tile(xs, nxo_t);
            let px: Tile<i32, { [16, 16] }> =
                (broadcast_scalar(pid.2 * nyo, shape![16, 16]) + ys) * nxo_t + xs;
            let v: Tile<f32, { [16, 16] }> = unsafe { gather(den, px, owned) };
            let zero: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
            let m: Tile<f32, { [16, 16] }> = select(owned & gt_tile(v, zero), v, zero);
            let m: Tile<f32, { [256] }> = m.reshape(shape![256]);
            let top: Tile<f32, { [] }> = reduce_max(m, 0i32);
            unsafe { max_into(out, top) };
        }
    }

    /// `ptyrs_kernels::normalize_tiles`: `normalize_update` of `x` `[nobj, nyo, nxo]` on the
    /// pixels of the object tiles with views, zeroing their sums, one program per tile and
    /// object mode, every operation rounded on its own as `normalize_at`; a program whose
    /// tile has none reads and writes nothing.
    #[cutile::entry()]
    unsafe fn normalize_tiles(
        x: *mut f32,
        num: *mut f32,
        den: *mut f32,
        tile_len: *const i32,
        max: *const f32,
        inertia: f32,
        nyo: i32,
        nxo: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let (tile, ys, xs): (i32, Tile<i32, { [16, 16] }>, Tile<i32, { [16, 16] }>) = obj_tile();
        let len: i32 = unsafe { load_i32(tile_len, tile) };
        if len != 0i32 {
            let nyo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nyo, shape![16, 16]);
            let nxo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nxo, shape![16, 16]);
            let owned: Tile<bool, { [16, 16] }> = lt_tile(ys, nyo_t) & lt_tile(xs, nxo_t);
            let px: Tile<i32, { [16, 16] }> =
                (broadcast_scalar(pid.2 * nyo, shape![16, 16]) + ys) * nxo_t + xs;
            let at: Tile<i32, { [16, 16] }> = px * constant(2i32, shape![16, 16]);
            let mx: Tile<f32, { [16, 16] }> =
                broadcast_scalar(unsafe { load_f32(max, 0i32) }, shape![16, 16]);
            let reg: Tile<f32, { [16, 16] }> = mulf(
                broadcast_scalar(inertia, shape![16, 16]),
                mx,
                rounding::NearestEven,
                ftz::Disabled,
            );
            let x_in: *const f32 = cast_const(x);
            let (a_re, a_im, num_t): (Tile<f32, { [16, 16] }>, Tile<f32, { [16, 16] }>, Token) =
                unsafe { gather_c_tok(cast_const(num), at, owned) };
            let (b_re, b_im): (Tile<f32, { [16, 16] }>, Tile<f32, { [16, 16] }>) =
                unsafe { gather_c(x_in, at, owned) };
            let (dv, den_t): (Tile<f32, { [16, 16] }>, Token) =
                unsafe { gather_tok(cast_const(den), px, owned) };
            let eps: Tile<f32, { [16, 16] }> = constant(1e-12f32, shape![16, 16]);
            // `f32::max` keeps the non-NaN operand: a NaN `den + reg` becomes `1e-12`.
            let s: Tile<f32, { [16, 16] }> = addf(dv, reg, rounding::NearestEven, ftz::Disabled);
            let d: Tile<f32, { [16, 16] }> = select(gt_tile(s, eps), s, eps);
            let rb_re: Tile<f32, { [16, 16] }> =
                mulf(reg, b_re, rounding::NearestEven, ftz::Disabled);
            let rb_im: Tile<f32, { [16, 16] }> =
                mulf(reg, b_im, rounding::NearestEven, ftz::Disabled);
            let t_re: Tile<f32, { [16, 16] }> =
                addf(a_re, rb_re, rounding::NearestEven, ftz::Disabled);
            let t_im: Tile<f32, { [16, 16] }> =
                addf(a_im, rb_im, rounding::NearestEven, ftz::Disabled);
            let r_re: Tile<f32, { [16, 16] }> = divf(t_re, d, rounding::NearestEven, ftz::Disabled);
            let r_im: Tile<f32, { [16, 16] }> = divf(t_im, d, rounding::NearestEven, ftz::Disabled);
            let at_im: Tile<i32, { [16, 16] }> = at + constant(1i32, shape![16, 16]);
            unsafe { scatter(x, at, r_re, owned) };
            unsafe { scatter(x, at_im, r_im, owned) };
            unsafe { zero_after(num, at, owned, num_t) };
            unsafe { zero_after(num, at_im, owned, num_t) };
            unsafe { zero_after(den, px, owned, den_t) };
        }
    }

    /// Element offsets `4 (256 p + row) + term` of program `p`'s block of four-term nodes,
    /// and which rows lie below `n`.
    fn llk_block(n: i32) -> (Tile<i32, { [256, 4] }>, Tile<bool, { [256, 4] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row: Tile<i32, { [256] }> = iota(shape![256]);
        let row: Tile<i32, { [256, 1] }> = row.reshape(shape![256, 1]);
        let row: Tile<i32, { [256, 4] }> = row.broadcast(shape![256, 4]);
        let node: Tile<i32, { [256, 4] }> = row + broadcast_scalar(pid.0 * 256i32, shape![256, 4]);
        let term: Tile<i32, { [4] }> = iota(shape![4]);
        let term: Tile<i32, { [1, 4] }> = term.reshape(shape![1, 4]);
        let term: Tile<i32, { [256, 4] }> = term.broadcast(shape![256, 4]);
        let inside: Tile<bool, { [256, 4] }> = lt_tile(node, broadcast_scalar(n, shape![256, 4]));
        (node * broadcast_scalar(4i32, shape![256, 4]) + term, inside)
    }

    /// Pairwise total over the 256 rows, adjacent rows first (`LlkSums`' tree).
    fn llk_pairs(x: Tile<f64, { [256, 4] }>) -> Tile<f64, { [4] }> {
        let z: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let o: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let x: Tile<f64, { [128, 2, 4] }> = x.reshape(shape![128, 2, 4]);
        let a: Tile<f64, { [128, 1, 4] }> = extract(x, [z, z, z]);
        let b: Tile<f64, { [128, 1, 4] }> = extract(x, [z, o, z]);
        let x: Tile<f64, { [64, 2, 4] }> = (a + b).reshape(shape![64, 2, 4]);
        let a: Tile<f64, { [64, 1, 4] }> = extract(x, [z, z, z]);
        let b: Tile<f64, { [64, 1, 4] }> = extract(x, [z, o, z]);
        let x: Tile<f64, { [32, 2, 4] }> = (a + b).reshape(shape![32, 2, 4]);
        let a: Tile<f64, { [32, 1, 4] }> = extract(x, [z, z, z]);
        let b: Tile<f64, { [32, 1, 4] }> = extract(x, [z, o, z]);
        let x: Tile<f64, { [16, 2, 4] }> = (a + b).reshape(shape![16, 2, 4]);
        let a: Tile<f64, { [16, 1, 4] }> = extract(x, [z, z, z]);
        let b: Tile<f64, { [16, 1, 4] }> = extract(x, [z, o, z]);
        let x: Tile<f64, { [8, 2, 4] }> = (a + b).reshape(shape![8, 2, 4]);
        let a: Tile<f64, { [8, 1, 4] }> = extract(x, [z, z, z]);
        let b: Tile<f64, { [8, 1, 4] }> = extract(x, [z, o, z]);
        let x: Tile<f64, { [4, 2, 4] }> = (a + b).reshape(shape![4, 2, 4]);
        let a: Tile<f64, { [4, 1, 4] }> = extract(x, [z, z, z]);
        let b: Tile<f64, { [4, 1, 4] }> = extract(x, [z, o, z]);
        let x: Tile<f64, { [2, 2, 4] }> = (a + b).reshape(shape![2, 2, 4]);
        let a: Tile<f64, { [2, 1, 4] }> = extract(x, [z, z, z]);
        let b: Tile<f64, { [2, 1, 4] }> = extract(x, [z, o, z]);
        let x: Tile<f64, { [1, 2, 4] }> = (a + b).reshape(shape![1, 2, 4]);
        let a: Tile<f64, { [1, 1, 4] }> = extract(x, [z, z, z]);
        let b: Tile<f64, { [1, 1, 4] }> = extract(x, [z, o, z]);
        let x: Tile<f64, { [4] }> = (a + b).reshape(shape![4]);
        x
    }

    /// `out[4 p ..][.. 4] = v` for program `p`.
    unsafe fn llk_store(out: *mut f64, v: Tile<f64, { [4] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let term: Tile<i32, { [4] }> = iota(shape![4]);
        let at: Tile<i32, { [4] }> = term + broadcast_scalar(pid.0 * 4i32, shape![4]);
        let dst: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let dst: PointerTile<*mut f64, { [1] }> = dst.reshape(shape![1]);
        let dst: PointerTile<*mut f64, { [4] }> = dst.broadcast(shape![4]);
        let dst: PointerTile<*mut f64, { [4] }> = dst.offset_tile(at);
        let _t: Token = unsafe {
            store_ptr_tko(
                dst,
                v,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            )
        };
    }

    /// First `LlkSums` tree level: program `p` stores the pairwise total of the four-term
    /// `f32` partials `256 p ..` (`+0.0` from `n` on) as `f64` in `out[4 p ..][.. 4]`.
    #[cutile::entry()]
    unsafe fn llk_leaves(partials: *const f32, n: i32, out: *mut f64) {
        let (at, inside): (Tile<i32, { [256, 4] }>, Tile<bool, { [256, 4] }>) = llk_block(n);
        let src: PointerTile<*const f32, { [] }> = pointer_to_tile(partials);
        let src: PointerTile<*const f32, { [1, 1] }> = src.reshape(shape![1, 1]);
        let src: PointerTile<*const f32, { [256, 4] }> = src.broadcast(shape![256, 4]);
        let src: PointerTile<*const f32, { [256, 4] }> = src.offset_tile(at);
        let (v, _t): (Tile<f32, { [256, 4] }>, Token) = unsafe {
            load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(inside),
                None,
                None,
                Latency::<0>,
            )
        };
        let v: Tile<f64, { [256, 4] }> = convert_tile(v);
        let zero: Tile<f64, { [256, 4] }> = constant(0.0f64, shape![256, 4]);
        let v: Tile<f64, { [256, 4] }> = select(inside, v, zero);
        unsafe { llk_store(out, llk_pairs(v)) };
    }

    /// A later `LlkSums` tree level: as `llk_leaves` over `n` four-term `f64` nodes.
    #[cutile::entry()]
    unsafe fn llk_nodes(nodes: *const f64, n: i32, out: *mut f64) {
        let (at, inside): (Tile<i32, { [256, 4] }>, Tile<bool, { [256, 4] }>) = llk_block(n);
        let src: PointerTile<*const f64, { [] }> = pointer_to_tile(nodes);
        let src: PointerTile<*const f64, { [1, 1] }> = src.reshape(shape![1, 1]);
        let src: PointerTile<*const f64, { [256, 4] }> = src.broadcast(shape![256, 4]);
        let src: PointerTile<*const f64, { [256, 4] }> = src.offset_tile(at);
        let (v, _t): (Tile<f64, { [256, 4] }>, Token) = unsafe {
            load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(inside),
                None,
                None,
                Latency::<0>,
            )
        };
        let zero: Tile<f64, { [256, 4] }> = constant(0.0f64, shape![256, 4]);
        let v: Tile<f64, { [256, 4] }> = select(inside, v, zero);
        unsafe { llk_store(out, llk_pairs(v)) };
    }

    /// Program `b`'s 256 lanes as SIMT block `b`'s threads: element `256 b + t` of lane
    /// `t`, the grid stride `256 * grid` and how many strides start below `n`.
    fn dot_lanes(n: i32) -> (Tile<i32, { [256] }>, i32, i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();
        let first: i32 = pid.0 * 256i32;
        let stride: i32 = grid.0 * 256i32;
        let lane: Tile<i32, { [256] }> = iota(shape![256]);
        let at: Tile<i32, { [256] }> = lane + broadcast_scalar(first, shape![256]);
        (at, stride, ceil_div(max(n - first, 0i32), stride))
    }

    /// Float offsets `2 at + [0, 1]` of the complex values at `at`, and `m` for both.
    fn c_at(
        at: Tile<i32, { [256] }>,
        m: Tile<bool, { [256] }>,
    ) -> (Tile<i32, { [256, 2] }>, Tile<bool, { [256, 2] }>) {
        let part: Tile<i32, { [2] }> = iota(shape![2]);
        let part: Tile<i32, { [1, 2] }> = part.reshape(shape![1, 2]);
        let part: Tile<i32, { [256, 2] }> = part.broadcast(shape![256, 2]);
        let at: Tile<i32, { [256] }> = at * constant(2i32, shape![256]);
        let at: Tile<i32, { [256, 1] }> = at.reshape(shape![256, 1]);
        let m: Tile<bool, { [256, 1] }> = m.reshape(shape![256, 1]);
        (
            at.broadcast(shape![256, 2]) + part,
            m.broadcast(shape![256, 2]),
        )
    }

    /// `p[at]` where `m`, as one load; other lanes are unspecified.
    unsafe fn load_c(
        p: *const f32,
        at: Tile<i32, { [256, 2] }>,
        m: Tile<bool, { [256, 2] }>,
    ) -> (Tile<f32, { [256, 2] }>, Token) {
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
        let base: PointerTile<*const f32, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*const f32, { [256, 2] }> = base.broadcast(shape![256, 2]);
        let ptrs: PointerTile<*const f32, { [256, 2] }> = base.offset_tile(at);
        let (v, t): (Tile<f32, { [256, 2] }>, Token) = unsafe {
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
        (v, t)
    }

    /// The real and imaginary columns of complex values.
    fn re_im(v: Tile<f32, { [256, 2] }>) -> (Tile<f32, { [256] }>, Tile<f32, { [256] }>) {
        let z: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let o: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let re: Tile<f32, { [256, 1] }> = extract(v, [z, z]);
        let im: Tile<f32, { [256, 1] }> = extract(v, [z, o]);
        (re.reshape(shape![256]), im.reshape(shape![256]))
    }

    /// `ptyrs_kernels::block_sum` of one value per lane: `s[t] += s[t + half]` for `half =
    /// 128, 64, .., 1`, the first half plus the second at every level.
    fn block_sum(x: Tile<f32, { [256] }>) -> Tile<f32, { [1] }> {
        let z: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let o: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let x: Tile<f32, { [2, 128] }> = x.reshape(shape![2, 128]);
        let a: Tile<f32, { [1, 128] }> = extract(x, [z, z]);
        let b: Tile<f32, { [1, 128] }> = extract(x, [o, z]);
        let x: Tile<f32, { [2, 64] }> = (a + b).reshape(shape![2, 64]);
        let a: Tile<f32, { [1, 64] }> = extract(x, [z, z]);
        let b: Tile<f32, { [1, 64] }> = extract(x, [o, z]);
        let x: Tile<f32, { [2, 32] }> = (a + b).reshape(shape![2, 32]);
        let a: Tile<f32, { [1, 32] }> = extract(x, [z, z]);
        let b: Tile<f32, { [1, 32] }> = extract(x, [o, z]);
        let x: Tile<f32, { [2, 16] }> = (a + b).reshape(shape![2, 16]);
        let a: Tile<f32, { [1, 16] }> = extract(x, [z, z]);
        let b: Tile<f32, { [1, 16] }> = extract(x, [o, z]);
        let x: Tile<f32, { [2, 8] }> = (a + b).reshape(shape![2, 8]);
        let a: Tile<f32, { [1, 8] }> = extract(x, [z, z]);
        let b: Tile<f32, { [1, 8] }> = extract(x, [o, z]);
        let x: Tile<f32, { [2, 4] }> = (a + b).reshape(shape![2, 4]);
        let a: Tile<f32, { [1, 4] }> = extract(x, [z, z]);
        let b: Tile<f32, { [1, 4] }> = extract(x, [o, z]);
        let x: Tile<f32, { [2, 2] }> = (a + b).reshape(shape![2, 2]);
        let a: Tile<f32, { [1, 2] }> = extract(x, [z, z]);
        let b: Tile<f32, { [1, 2] }> = extract(x, [o, z]);
        let x: Tile<f32, { [2, 1] }> = (a + b).reshape(shape![2, 1]);
        let a: Tile<f32, { [1, 1] }> = extract(x, [z, z]);
        let b: Tile<f32, { [1, 1] }> = extract(x, [o, z]);
        (a + b).reshape(shape![1])
    }

    /// `p[i] = v`.
    unsafe fn store_1(p: *mut f32, i: i32, v: Tile<f32, { [1] }>) {
        let at: PointerTile<*mut f32, { [] }> = pointer_to_tile(p);
        let at: PointerTile<*mut f32, { [] }> = at.offset(i);
        let at: PointerTile<*mut f32, { [1] }> = at.reshape(shape![1]);
        let _t: Token = unsafe {
            store_ptr_tko(
                at,
                v,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            )
        };
    }

    /// `out[2 b ..][.. 2] = (block_sum(s0), block_sum(s1))` for program `b`, as
    /// `ptyrs_kernels::store_pair`.
    unsafe fn store_pair(out: *mut f32, s0: Tile<f32, { [256] }>, s1: Tile<f32, { [256] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        unsafe { store_1(out, 2i32 * pid.0, block_sum(s0)) };
        unsafe { store_1(out, 2i32 * pid.0 + 1i32, block_sum(s1)) };
    }

    /// `ptyrs_kernels::cg_beta_parts`: program `b`'s pair `(sum Re(conj(g) (g - h)), sum
    /// |h|^2)` over its SIMT block's grid-stride elements of `g` and `h` (`n` complex), into
    /// `part[2 b ..][.. 2]`, then `h = g` there. Each lane accumulates as the SIMT build
    /// contracts it, `acc = fma(x0, y0, fma(x1, y1, acc))`.
    #[cutile::entry()]
    unsafe fn cg_beta_parts(g: *const f32, h: *mut f32, n: i32, part: *mut f32) {
        let (at0, stride, steps): (Tile<i32, { [256] }>, i32, i32) = dot_lanes(n);
        let n_t: Tile<i32, { [256] }> = broadcast_scalar(n, shape![256]);
        let mut acc0: Tile<f32, { [256] }> = constant(0.0f32, shape![256]);
        let mut acc1: Tile<f32, { [256] }> = constant(0.0f32, shape![256]);
        for k in 0i32..steps {
            let at: Tile<i32, { [256] }> = at0 + broadcast_scalar(k * stride, shape![256]);
            let inside: Tile<bool, { [256] }> = lt_tile(at, n_t);
            let (ac, mc): (Tile<i32, { [256, 2] }>, Tile<bool, { [256, 2] }>) = c_at(at, inside);
            let (va, _ta): (Tile<f32, { [256, 2] }>, Token) = unsafe { load_c(g, ac, mc) };
            let (vb, tb): (Tile<f32, { [256, 2] }>, Token) =
                unsafe { load_c(cast_const(h), ac, mc) };
            let (a0, a1): (Tile<f32, { [256] }>, Tile<f32, { [256] }>) = re_im(va);
            let (b0, b1): (Tile<f32, { [256] }>, Tile<f32, { [256] }>) = re_im(vb);
            let d0: Tile<f32, { [256] }> = subf(a0, b0, rounding::NearestEven, ftz::Disabled);
            let d1: Tile<f32, { [256] }> = subf(a1, b1, rounding::NearestEven, ftz::Disabled);
            let s0: Tile<f32, { [256] }> = fma(a1, d1, acc0, rounding::NearestEven, ftz::Disabled);
            let s0: Tile<f32, { [256] }> = fma(a0, d0, s0, rounding::NearestEven, ftz::Disabled);
            let s1: Tile<f32, { [256] }> = fma(b1, b1, acc1, rounding::NearestEven, ftz::Disabled);
            let s1: Tile<f32, { [256] }> = fma(b0, b0, s1, rounding::NearestEven, ftz::Disabled);
            acc0 = select(inside, s0, acc0);
            acc1 = select(inside, s1, acc1);
            let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(h);
            let base: PointerTile<*mut f32, { [1, 1] }> = base.reshape(shape![1, 1]);
            let base: PointerTile<*mut f32, { [256, 2] }> = base.broadcast(shape![256, 2]);
            let ptrs: PointerTile<*mut f32, { [256, 2] }> = base.offset_tile(ac);
            let _t: Token = unsafe {
                store_ptr_tko(
                    ptrs,
                    va,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(mc),
                    Some(tb),
                    Latency::<0>,
                )
            };
        }
        unsafe { store_pair(part, acc0, acc1) };
    }

    /// `ptyrs_kernels::cg_sum_parts`: `out[0 .. 2]` = the sum of the `npart` pairs of
    /// `part`, lane `t` adding pairs `t, t + 256, ..` in order, then `block_sum`. One
    /// program.
    #[cutile::entry()]
    unsafe fn cg_sum_parts(part: *const f32, npart: i32, out: *mut f32) {
        let lane: Tile<i32, { [256] }> = iota(shape![256]);
        let n_t: Tile<i32, { [256] }> = broadcast_scalar(npart, shape![256]);
        let mut acc0: Tile<f32, { [256] }> = constant(0.0f32, shape![256]);
        let mut acc1: Tile<f32, { [256] }> = constant(0.0f32, shape![256]);
        for k in 0i32..ceil_div(npart, 256i32) {
            let at: Tile<i32, { [256] }> = lane + broadcast_scalar(256i32 * k, shape![256]);
            let inside: Tile<bool, { [256] }> = lt_tile(at, n_t);
            let (ac, mc): (Tile<i32, { [256, 2] }>, Tile<bool, { [256, 2] }>) = c_at(at, inside);
            let (v, _t): (Tile<f32, { [256, 2] }>, Token) = unsafe { load_c(part, ac, mc) };
            let (v0, v1): (Tile<f32, { [256] }>, Tile<f32, { [256] }>) = re_im(v);
            acc0 = select(inside, acc0 + v0, acc0);
            acc1 = select(inside, acc1 + v1, acc1);
        }
        unsafe { store_pair(out, acc0, acc1) };
    }

    /// `ptyrs_kernels::cg_norm_parts`: adds program `b`'s `(sum |x|, sum |x|^2)` over its
    /// SIMT block's grid-stride elements of `x` (`n` complex) to `part[2 b ..][.. 2]`, with
    /// `|x|^2 = fma(re, re, im im)` as the SIMT build contracts it and an IEEE square root.
    #[cutile::entry()]
    unsafe fn cg_norm_parts(x: *const f32, n: i32, part: *mut f32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let (at0, stride, steps): (Tile<i32, { [256] }>, i32, i32) = dot_lanes(n);
        let n_t: Tile<i32, { [256] }> = broadcast_scalar(n, shape![256]);
        let mut acc0: Tile<f32, { [256] }> = constant(0.0f32, shape![256]);
        let mut acc1: Tile<f32, { [256] }> = constant(0.0f32, shape![256]);
        for k in 0i32..steps {
            let at: Tile<i32, { [256] }> = at0 + broadcast_scalar(k * stride, shape![256]);
            let inside: Tile<bool, { [256] }> = lt_tile(at, n_t);
            let (ac, mc): (Tile<i32, { [256, 2] }>, Tile<bool, { [256, 2] }>) = c_at(at, inside);
            let (v, _t): (Tile<f32, { [256, 2] }>, Token) = unsafe { load_c(x, ac, mc) };
            let (re, im): (Tile<f32, { [256] }>, Tile<f32, { [256] }>) = re_im(v);
            let im2: Tile<f32, { [256] }> = mulf(im, im, rounding::NearestEven, ftz::Disabled);
            let s: Tile<f32, { [256] }> = fma(re, re, im2, rounding::NearestEven, ftz::Disabled);
            let r: Tile<f32, { [256] }> = sqrt(s, rounding::NearestEven, ftz::Disabled);
            acc0 = select(inside, acc0 + r, acc0);
            acc1 = select(inside, acc1 + s, acc1);
        }
        let i0: i32 = 2i32 * pid.0;
        let p0: f32 = unsafe { load_f32(cast_const(part), i0) };
        let p1: f32 = unsafe { load_f32(cast_const(part), i0 + 1i32) };
        let s0: Tile<f32, { [1] }> = block_sum(acc0) + broadcast_scalar(p0, shape![1]);
        let s1: Tile<f32, { [1] }> = block_sum(acc1) + broadcast_scalar(p1, shape![1]);
        unsafe { store_1(part, i0, s0) };
        unsafe { store_1(part, i0 + 1i32, s1) };
    }

    /// Program `b`'s pair `(scale sum |g|^2, sum Re(conj(g) prev))` over its SIMT block's
    /// grid-stride elements (`n` complex) into `part[2 b ..][.. 2]`, the second sum only if
    /// `cg != 0` (else `0`, and `prev` is not read). Each lane accumulates `acc = fma(x0,
    /// y0, fma(x1, y1, acc))` and is scaled before the tree.
    #[cutile::entry()]
    unsafe fn bh_dots(
        g: *const f32,
        prev: *const f32,
        n: i32,
        scale: f32,
        cg: i32,
        part: *mut f32,
    ) {
        let (at0, stride, steps): (Tile<i32, { [256] }>, i32, i32) = dot_lanes(n);
        let n_t: Tile<i32, { [256] }> = broadcast_scalar(n, shape![256]);
        let cg_i: Tile<i32, { [256] }> = broadcast_scalar(cg, shape![256]);
        let zero_i: Tile<i32, { [256] }> = constant(0i32, shape![256]);
        let cg_t: Tile<bool, { [256] }> = ne_tile(cg_i, zero_i);
        let mut acc0: Tile<f32, { [256] }> = constant(0.0f32, shape![256]);
        let mut acc1: Tile<f32, { [256] }> = constant(0.0f32, shape![256]);
        for k in 0i32..steps {
            let at: Tile<i32, { [256] }> = at0 + broadcast_scalar(k * stride, shape![256]);
            let inside: Tile<bool, { [256] }> = lt_tile(at, n_t);
            let both: Tile<bool, { [256] }> = inside & cg_t;
            let (ac, mc): (Tile<i32, { [256, 2] }>, Tile<bool, { [256, 2] }>) = c_at(at, inside);
            let (_ab, mb): (Tile<i32, { [256, 2] }>, Tile<bool, { [256, 2] }>) = c_at(at, both);
            let (va, _ta): (Tile<f32, { [256, 2] }>, Token) = unsafe { load_c(g, ac, mc) };
            let (vb, _tb): (Tile<f32, { [256, 2] }>, Token) = unsafe { load_c(prev, ac, mb) };
            let (a0, a1): (Tile<f32, { [256] }>, Tile<f32, { [256] }>) = re_im(va);
            let (b0, b1): (Tile<f32, { [256] }>, Tile<f32, { [256] }>) = re_im(vb);
            let s0: Tile<f32, { [256] }> = fma(a1, a1, acc0, rounding::NearestEven, ftz::Disabled);
            let s0: Tile<f32, { [256] }> = fma(a0, a0, s0, rounding::NearestEven, ftz::Disabled);
            let s1: Tile<f32, { [256] }> = fma(a1, b1, acc1, rounding::NearestEven, ftz::Disabled);
            let s1: Tile<f32, { [256] }> = fma(a0, b0, s1, rounding::NearestEven, ftz::Disabled);
            acc0 = select(inside, s0, acc0);
            acc1 = select(both, s1, acc1);
        }
        let sc: Tile<f32, { [256] }> = broadcast_scalar(scale, shape![256]);
        let acc0: Tile<f32, { [256] }> = mulf(acc0, sc, rounding::NearestEven, ftz::Disabled);
        unsafe { store_pair(part, acc0, acc1) };
    }
}

fn shape(what: String) -> Error {
    Error::Shape(what)
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

/// A device address as a kernel pointer to `U`.
fn ptr<U>(addr: CUdeviceptr) -> DevicePointer<U> {
    // SAFETY: a launch hands the kernel this address with a contract that keeps every
    // access inside the buffer it came from (checked by its caller); a compile-only run
    // reads nothing.
    unsafe { DevicePointer::from_cu_deviceptr(addr) }
}

/// Device address of `buf` as a kernel pointer to `U`.
fn dptr<T: DeviceCopy, U>(buf: &DeviceBuffer<T>) -> DevicePointer<U> {
    ptr(buf.cu_deviceptr())
}

/// Address of a buffer a launch is handed. cutile specializes a kernel on the alignment of
/// each pointer, up to 16 bytes, and [`TileKernels::new`] compiles the 16-byte case only.
fn addr<T: DeviceCopy>(what: &str, buf: &DeviceBuffer<T>) -> Result<CUdeviceptr, Error> {
    let a = buf.cu_deviceptr();
    if !a.is_multiple_of(16) {
        return Err(shape(format!("{what} at {a:#x} is not 16-byte aligned")));
    }
    Ok(a)
}

/// What a compile-only run passes for a buffer a launch is handed: aligned like every
/// address [`addr`] accepts, and never read.
const UNBOUND: CUdeviceptr = 0;

/// What a compile-only run passes for a stack entry, one address of each alignment [`ctl`]
/// accepts (16 and 8 bytes), never read.
const UNBOUND_CTL: [CUdeviceptr; 2] = [0, 8];

/// Address of `stack`'s entry. Entry `s` lies `8 s` bytes into its table's allocation, so
/// it is aligned to 16 or 8 bytes, the two cases [`TileKernels::new`] compiles.
fn ctl(stack: StackRef<'_>) -> Result<CUdeviceptr, Error> {
    let a = stack.ctl();
    if !a.is_multiple_of(8) {
        return Err(shape(format!(
            "stack entry at {a:#x} is not 8-byte aligned"
        )));
    }
    Ok(a)
}

fn obj_len(d: &ViewDims) -> usize {
    d.nobj as usize * d.obj[0] as usize * d.obj[1] as usize
}

/// Node counts of the [`crate::gpu::LlkSums`] tree levels above `n` leaves, the root last.
fn llk_levels(n: u32) -> Vec<usize> {
    let mut levels = Vec::new();
    let mut m = n as usize;
    loop {
        m = m.div_ceil(LLK_TREE_LEAVES).max(1);
        levels.push(m);
        if m == 1 {
            return levels;
        }
    }
}

/// `f64`s of scratch [`TileKernels::llk_sums`] needs for `n` frames: the tree levels below
/// the root.
pub fn llk_sums_scratch_len(n: u32) -> usize {
    let levels = llk_levels(n);
    4 * levels[..levels.len() - 1].iter().sum::<usize>()
}

/// Where a launch sequence goes: compiled only, while a [`TileKernels`] is built, or onto a
/// context's stream.
#[derive(Clone, Copy)]
enum Run<'c> {
    Compile(&'c Arc<CudaStream>),
    Launch(&'c ExecutionContext),
}

/// Compiles or launches a launcher, as `$run` says. Expands to unsafe calls.
macro_rules! run {
    ($run:expr, $launcher:expr) => {
        match $run {
            Run::Compile(stream) => $launcher.compile_on(stream)?,
            Run::Launch(ctx) => {
                $launcher.execute(ctx)?;
            }
        }
    };
}

/// The Tile kernels of one scan's [`Views`] and stack geometry ([`StackGeom`]: the views'
/// sizes and a table's `nf_max`), compiled when the set is built. cutile compiles a kernel
/// once per specialization, keyed by the power-of-two divisor of each integer argument and
/// the alignment of each pointer. Every integer here comes from the geometry, which a launch
/// checks its stack against, the scan's buffers are the set's, a stack entry (`8 s` bytes
/// into its table) is aligned to 16 or 8 bytes and every other buffer must be 16-byte
/// aligned; so [`Self::new`]
/// compiles every specialization a launch can reach, and no launch, eager or captured,
/// compiles.
pub struct TileKernels<'a> {
    views: &'a Views,
    /// What the kernels were compiled for; each of its counts and sides is an `i32`.
    geom: StackGeom,
    /// Complex values of a stack's psi, `nf_max * nmode * ny * nx`.
    psi_len: usize,
    /// Complex values of the probe, `nprobe * ny * nx`.
    probe_len: usize,
    llk_levels: Vec<usize>,
    obj_update: TileObjUpdate,
    probe_update: TileProbeUpdate,
}

impl<'a> TileKernels<'a> {
    /// Checks the scan's sizes against the kernels' `i32` offsets and compiles every kernel
    /// on `stream`, without launching any, for the stacks of `table` and of every other
    /// table of its `nf_max` over `views`.
    pub fn new(
        stream: &Arc<CudaStream>,
        views: &'a Views,
        table: &StackTable,
    ) -> Result<Self, Error> {
        let d = views.dims();
        let geom = d.geom(table.at(0))?;
        let n = geom.ny as usize * geom.nx as usize;
        let psi_len = geom.nf_max as usize * geom.nmode as usize * n;
        let probe_len = geom.nprobe as usize * n;
        // Every offset the kernels form, in `f32` or `ViewAddr`-word units, is an i32.
        offset("psi length", 2 * psi_len)?;
        offset("probe length", 2 * probe_len)?;
        offset("object length", 2 * obj_len(d))?;
        offset("view words", 6 * views.addr().len())?;
        offset("tile list length", views.tile_views().len())?;
        let g = geom;
        for v in [g.nframe, g.nf_max, g.nmode, g.ny, g.nx, g.nyo, g.nxo] {
            offset("geometry", v as usize)?;
        }
        let llk_levels = llk_levels(geom.nframe);
        // The first level's programs index 256 four-term nodes each, as i32.
        offset("leaf offsets", 4 * LLK_TREE_LEAVES * llk_levels[0])?;
        let k = Self {
            views,
            geom,
            psi_len,
            probe_len,
            llk_levels,
            obj_update: TileObjUpdate::new(stream, views)?,
            probe_update: TileProbeUpdate::new(stream, views)?,
        };
        let run = Run::Compile(stream);
        for ctl in UNBOUND_CTL {
            k.psi_to_obj_run(run, ctl, [UNBOUND; 4])?;
            k.psi_to_probe_run(run, ctl, [UNBOUND; 4])?;
        }
        k.llk_run(run, [UNBOUND; 3])?;
        Ok(k)
    }

    /// The scan's object update, compiled with the set.
    pub fn obj_update(&self) -> &TileObjUpdate {
        &self.obj_update
    }

    /// The scan's probe update, compiled with the set.
    pub fn probe_update(&self) -> &TileProbeUpdate {
        &self.probe_update
    }

    /// The stack kernels' scalar arguments: `nframe`, `nf_max`, `nmode`, the frame and
    /// object sides, each checked by [`Self::new`] to be an `i32`.
    fn args(&self) -> [i32; 7] {
        let g = self.geom;
        [g.nframe, g.nf_max, g.nmode, g.ny, g.nx, g.nyo, g.nxo].map(|v| v as i32)
    }

    /// The address of `stack`'s entry, a stack of the geometry the set was compiled for.
    fn stack(&self, stack: StackRef<'_>) -> Result<CUdeviceptr, Error> {
        let g = self.views.dims().geom(stack)?;
        if g != self.geom {
            return Err(shape(format!(
                "a stack of {g:?}, the kernels were built for {:?}",
                self.geom
            )));
        }
        ctl(stack)
    }

    /// [`crate::gpu::Kernels::psi_to_obj`] for `stack`: adds the stack's object-update sums
    /// to `sums` `[nobj, nyo, nxo]`, each object pixel adding its views in ascending view
    /// order from the same tile lists, so the result does not depend on the stack split.
    /// Only the pixels of tiles the stack's views touch are read and written. `psi` holds
    /// `nf_max` frames. The kernel reads the entry when it runs, so a captured launch serves
    /// whichever stack the entry holds at replay.
    pub fn psi_to_obj(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        psi: &DeviceBuffer<C32>,
        probe: &DeviceBuffer<C32>,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let (v, len) = (self.views, obj_len(self.views.dims()));
        let ctl = self.stack(stack)?;
        check_len("psi", psi, self.psi_len)?;
        check_len("probe", probe, self.probe_len)?;
        check_len("sums.num", &sums.num, len)?;
        check_len("sums.den", &sums.den, len)?;
        let at = [
            addr("psi", psi)?,
            addr("probe", probe)?,
            addr("sums.num", &sums.num)?,
            addr("sums.den", &sums.den)?,
        ];
        let reads = [
            psi.storage(),
            probe.storage(),
            v.addr().storage(),
            v.tile_start().storage(),
            v.tile_len().storage(),
            v.tile_views().storage(),
            stack.storage(),
        ];
        lease(ctx, &reads, &[sums.num.storage(), sums.den.storage()])?;
        self.psi_to_obj_run(Run::Launch(ctx), ctl, at)
    }

    fn psi_to_obj_run(
        &self,
        run: Run,
        ctl: CUdeviceptr,
        [psi, probe, num, den]: [CUdeviceptr; 4],
    ) -> Result<(), Error> {
        let (v, d, g) = (self.views, self.views.dims(), self.args());
        let [nty, ntx] = d.tiles();
        // SAFETY: a launch comes from `psi_to_obj`, which checked the stack's geometry and
        // the lengths and leased the buffers on `ctx`; `new` checked the i32 offsets; `Views`
        // checked every view lies inside the object with valid mode indices and holds tile
        // runs built from them; the kernel clamps the stack entry to `nf_max` frames inside
        // the scan, and each program writes only the pixels of its own tile.
        unsafe {
            run!(
                run,
                device::psi_to_obj(
                    ptr(psi),
                    ptr(probe),
                    dptr(v.addr()),
                    dptr(v.addr()),
                    dptr(v.tile_start()),
                    dptr(v.tile_len()),
                    dptr(v.tile_views()),
                    ptr(num),
                    ptr(den),
                    ptr(ctl),
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    g[4],
                    g[5],
                    g[6],
                    self.geom.quad,
                )
                .grid((ntx, nty, d.nobj))
            );
        }
        Ok(())
    }

    /// [`crate::gpu::Kernels::psi_to_probe`] for `stack`: adds the stack's probe-update sums
    /// to `sums` `[nprobe, ny, nx]`, each probe pixel adding the views of its mode in
    /// ascending view order, so the result does not depend on the stack split; a mode
    /// without a view in the stack is not written. `psi` holds `nf_max` frames. The entry is
    /// read as in [`Self::psi_to_obj`].
    pub fn psi_to_probe(
        &self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        psi: &DeviceBuffer<C32>,
        obj: &DeviceBuffer<C32>,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let ctl = self.stack(stack)?;
        check_len("psi", psi, self.psi_len)?;
        check_len("obj", obj, obj_len(self.views.dims()))?;
        check_len("sums.num", &sums.num, self.probe_len)?;
        check_len("sums.den", &sums.den, self.probe_len)?;
        let at = [
            addr("psi", psi)?,
            addr("obj", obj)?,
            addr("sums.num", &sums.num)?,
            addr("sums.den", &sums.den)?,
        ];
        let reads = [
            psi.storage(),
            obj.storage(),
            self.views.addr().storage(),
            stack.storage(),
        ];
        lease(ctx, &reads, &[sums.num.storage(), sums.den.storage()])?;
        self.psi_to_probe_run(Run::Launch(ctx), ctl, at)
    }

    fn psi_to_probe_run(
        &self,
        run: Run,
        ctl: CUdeviceptr,
        [psi, obj, num, den]: [CUdeviceptr; 4],
    ) -> Result<(), Error> {
        let (v, g) = (self.views, self.args());
        let (ny, nx, nprobe) = (self.geom.ny, self.geom.nx, self.geom.nprobe);
        // SAFETY: a launch comes from `psi_to_probe`, which checked the stack's geometry and
        // the lengths and leased the buffers on `ctx`; `new` checked the i32 offsets; `Views`
        // checked every view lies inside the object with valid mode indices; the kernel clamps
        // the stack entry to `nf_max` frames inside the scan, and each program writes only the
        // pixels of its own tile.
        unsafe {
            run!(
                run,
                device::psi_to_probe(
                    ptr(psi),
                    ptr(obj),
                    dptr(v.addr()),
                    dptr(v.addr()),
                    ptr(num),
                    ptr(den),
                    ptr(ctl),
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    g[4],
                    g[5],
                    g[6],
                    self.geom.quad,
                )
                .grid((nx.div_ceil(32), ny.div_ceil(8), nprobe))
            );
        }
        Ok(())
    }

    /// [`crate::gpu::LlkSums::from_partials`] of the scan's `nframe` partials on the device,
    /// into `out` as `[poisson, gaussian, euclidean, icalc]`, bit for bit: one launch per
    /// tree level, the levels below the root in `scratch` ([`llk_sums_scratch_len`] long).
    pub fn llk_sums(
        &self,
        ctx: &ExecutionContext,
        partials: &DeviceBuffer<LlkPartial>,
        scratch: &mut DeviceBuffer<f64>,
        out: &mut DeviceBuffer<f64>,
    ) -> Result<(), Error> {
        let n = self.geom.nframe;
        check_len("partials", partials, n as usize)?;
        check_len("scratch", scratch, llk_sums_scratch_len(n))?;
        check_len("out", out, 4)?;
        let at = [
            addr("partials", partials)?,
            addr("scratch", scratch)?,
            addr("out", out)?,
        ];
        lease(
            ctx,
            &[partials.storage()],
            &[scratch.storage(), out.storage()],
        )?;
        self.llk_run(Run::Launch(ctx), at)
    }

    fn llk_run(&self, run: Run, [partials, scratch, out]: [CUdeviceptr; 3]) -> Result<(), Error> {
        let n = self.geom.nframe as i32;
        let node = |at: usize| scratch + (at * size_of::<f64>()) as CUdeviceptr;
        // Scratch offset and node count of the level below.
        let mut below: Option<(usize, i32)> = None;
        let mut at = 0;
        for (l, &m) in self.llk_levels.iter().enumerate() {
            let dst = if l + 1 == self.llk_levels.len() {
                out
            } else {
                node(at)
            };
            // SAFETY: a launch comes from `llk_sums`, which checked that `partials` holds `n`
            // records of four f32 and `scratch` every level below the root, each of `m`
            // four-f64 nodes at `at`, and leased them on `ctx`; program `p` reads nodes `256 p
            // ..` below `n` (or the level's count) and writes node `p`.
            unsafe {
                match below {
                    None => run!(
                        run,
                        device::llk_leaves(ptr(partials), n, ptr(dst)).grid((m as u32, 1, 1))
                    ),
                    Some((src, count)) => run!(
                        run,
                        device::llk_nodes(ptr(node(src)), count, ptr(dst)).grid((m as u32, 1, 1))
                    ),
                }
            }
            below = Some((at, m as i32));
            at += 4 * m;
        }
        Ok(())
    }
}

/// The Tile object update of one scan's views, compiled when it is built, as [`TileKernels`]
/// compiles its kernels; a launch's integers are the views' sizes, which it checks, and every
/// buffer, `tile_len` included, must be 16-byte aligned. It keeps the sizes, not the views,
/// so the owner of a [`Views`] can hold it beside them.
pub struct TileObjUpdate {
    /// What the update was compiled for.
    dims: ViewDims,
}

impl TileObjUpdate {
    /// Checks the object's length against the kernels' `i32` offsets and compiles the update
    /// on `stream`, without launching it, for views of `views`' sizes.
    pub fn new(stream: &Arc<CudaStream>, views: &Views) -> Result<Self, Error> {
        let dims = *views.dims();
        offset("object length", 2 * obj_len(&dims))?;
        let u = Self { dims };
        u.obj_run(Run::Compile(stream), [UNBOUND; 5], 0.0)?;
        Ok(u)
    }

    /// [`crate::gpu::Kernels::update_obj_from_sums`]:
    /// [`TileProbeUpdate::update_probe_from_sums`] of the object `[nobj, nyo, nxo]` on the
    /// object tiles with views only, `reg = inertia * max(den)` over their pixels, zeroing
    /// their sums. A tile whose run is empty is neither read nor written, in `obj` or `sums`,
    /// so it keeps its bits. `sums.max_bits` is the scratch for the maximum.
    pub fn update_obj_from_sums(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        obj: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        inertia: f32,
    ) -> Result<(), Error> {
        let (d, at) = (views.dims(), &self.dims);
        if d.nobj != at.nobj || d.obj != at.obj {
            return Err(shape(format!(
                "views of {} objects of {:?}, the update was built for {} of {:?}",
                d.nobj, d.obj, at.nobj, at.obj
            )));
        }
        let len = obj_len(at);
        check_len("obj", obj, len)?;
        check_len("sums.num", &sums.num, len)?;
        check_len("sums.den", &sums.den, len)?;
        check_len("sums.max_bits", &sums.max_bits, 1)?;
        let UpdateSums { num, den, max_bits } = sums;
        let tile_len = views.tile_len();
        let at = [
            addr("obj", obj)?,
            addr("sums.num", num)?,
            addr("sums.den", den)?,
            addr("sums.max_bits", max_bits)?,
            addr("views.tile_len", tile_len)?,
        ];
        crate::gpu::zero(ctx, max_bits)?;
        let writes = [
            max_bits.storage(),
            obj.storage(),
            num.storage(),
            den.storage(),
        ];
        lease(ctx, &[tile_len.storage()], &writes)?;
        self.obj_run(Run::Launch(ctx), at, inertia)
    }

    fn obj_run(
        &self,
        run: Run,
        [obj, num, den, max_bits, tile_len]: [CUdeviceptr; 5],
        inertia: f32,
    ) -> Result<(), Error> {
        let d = &self.dims;
        // `new` checked that the object's `2 nobj nyo nxo` floats index as i32.
        let [nyo, nxo] = d.obj.map(|v| v as i32);
        let [nty, ntx] = d.tiles();
        let grid = (ntx, nty, d.nobj);
        // SAFETY: a launch comes from `update_obj_from_sums`, which checked that `obj`, `num`
        // and `den` hold the object and `max_bits` one u32, zeroed it and leased the buffers
        // on `ctx`, and that `tile_len` holds the tiles of views of the object's sizes; each
        // program reads and writes only the pixels of its own tile.
        unsafe {
            run!(
                run,
                device::max_den_tiles(ptr(den), ptr(tile_len), ptr(max_bits), nyo, nxo).grid(grid)
            );
            run!(
                run,
                device::normalize_tiles(
                    ptr(obj),
                    ptr(num),
                    ptr(den),
                    ptr(tile_len),
                    ptr(max_bits),
                    inertia,
                    nyo,
                    nxo,
                )
                .grid(grid)
            );
        }
        Ok(())
    }
}

/// The Tile probe update of one scan's views, compiled when it is built, as [`TileKernels`]
/// compiles its kernels; every buffer must be 16-byte aligned.
pub struct TileProbeUpdate {
    /// Complex values of the probe, `nprobe * ny * nx`.
    probe_len: usize,
}

impl TileProbeUpdate {
    /// Checks the probe's length against the kernels' `i32` offsets and compiles the update
    /// on `stream`, without launching it, for views of `views`' sizes.
    pub fn new(stream: &Arc<CudaStream>, views: &Views) -> Result<Self, Error> {
        let d = views.dims();
        let probe_len = d.nprobe as usize * d.frame[0] as usize * d.frame[1] as usize;
        offset("probe length", 2 * probe_len)?;
        let u = Self { probe_len };
        u.probe_run(Run::Compile(stream), [UNBOUND; 4], 0.0)?;
        Ok(u)
    }

    /// [`crate::gpu::Kernels::update_from_sums`] of the probe: `x = (num + reg x) / max(den +
    /// reg, 1e-12)` for its `nprobe * ny * nx` values, `reg = inertia * max(0, den ...)`
    /// over the non-NaN values of `den`, then zeroes `sums.num` and `sums.den`, which it has
    /// consumed. `sums.max_bits` is the scratch for that maximum, zeroed on `ctx`'s stream
    /// here, so the call allocates nothing and never synchronizes.
    pub fn update_probe_from_sums(
        &self,
        ctx: &ExecutionContext,
        probe: &mut DeviceBuffer<C32>,
        sums: &mut UpdateSums,
        inertia: f32,
    ) -> Result<(), Error> {
        let n = self.probe_len;
        check_len("probe", probe, n)?;
        check_len("sums.num", &sums.num, n)?;
        check_len("sums.den", &sums.den, n)?;
        check_len("sums.max_bits", &sums.max_bits, 1)?;
        let UpdateSums { num, den, max_bits } = sums;
        let at = [
            addr("probe", probe)?,
            addr("sums.num", num)?,
            addr("sums.den", den)?,
            addr("sums.max_bits", max_bits)?,
        ];
        crate::gpu::zero(ctx, max_bits)?;
        let writes = [
            max_bits.storage(),
            probe.storage(),
            num.storage(),
            den.storage(),
        ];
        lease(ctx, &[], &writes)?;
        self.probe_run(Run::Launch(ctx), at, inertia)
    }

    fn probe_run(
        &self,
        run: Run,
        [x, num, den, max_bits]: [CUdeviceptr; 4],
        inertia: f32,
    ) -> Result<(), Error> {
        // `new` checked that the probe's `2 n` floats index as i32.
        let n = self.probe_len as i32;
        // SAFETY: a launch comes from `update_probe_from_sums`, which checked that `x`, `num`
        // hold `n` complex values, `den` `n` floats and `max_bits` one, zeroed `max_bits` and
        // leased the buffers on `ctx`.
        unsafe {
            run!(
                run,
                device::max_nonneg_f32(ptr(den), n, ptr(max_bits)).grid((
                    (n as u32).div_ceil(1024),
                    1,
                    1
                ))
            );
            run!(
                run,
                device::normalize_update(ptr(x), ptr(num), ptr(den), n, ptr(max_bits), inertia)
                    .grid(((2 * n as u32).div_ceil(1024), 1, 1))
            );
        }
        Ok(())
    }
}

/// Elements a dot product's `CG_BLOCKS` programs of 256 lanes visit per grid stride.
const DOT_STRIDE: usize = 256 * CG_BLOCKS as usize;

/// The object's and the probe's lengths in complex values, checked against the dot
/// kernels' `i32` offsets, which reach `2 (n + DOT_STRIDE)` floats.
fn dot_lens(dims: &ViewDims) -> Result<[i32; 2], Error> {
    let probe_len = dims.nprobe as usize * dims.frame[0] as usize * dims.frame[1] as usize;
    let mut lens = [0; 2];
    for (l, (what, n)) in lens.iter_mut().zip([
        ("object length", obj_len(dims)),
        ("probe length", probe_len),
    ]) {
        offset(what, 2 * (n + DOT_STRIDE))?;
        *l = n as i32;
    }
    Ok(lens)
}

/// BH's dot products over a scan of one [`ViewDims`], [`Self::dots`], which has no SIMT
/// form. [`Self::new`] compiles it for the object's and the probe's lengths, with and
/// without `v`, and nothing else.
pub struct TileDots {
    /// Complex values of the object and of the probe.
    lens: [i32; 2],
}

impl TileDots {
    /// Checks the lengths against the kernels' `i32` offsets and compiles [`Self::dots`] on
    /// `stream` for a scan of `dims`, without launching it.
    pub fn new(stream: &Arc<CudaStream>, dims: &ViewDims) -> Result<Self, Error> {
        let t = Self {
            lens: dot_lens(dims)?,
        };
        for n in t.lens {
            for cg in [false, true] {
                t.dots_run(Run::Compile(stream), [UNBOUND; 3], n, 1.0, cg)?;
            }
        }
        Ok(t)
    }

    /// BH's dot products of the step along `d`, `⟨u, Pu⟩` and (CG) `⟨u, v⟩`, as block
    /// partials into the first `2 CG_BLOCKS` pairs of `parts`, object then probe.
    pub fn dots(
        &self,
        ctx: &ExecutionContext,
        d: &BhDisp<'_>,
        parts: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        check_len("parts", parts, 4 * CG_BLOCKS as usize)?;
        for (i, n) in self.lens.into_iter().enumerate() {
            let (g, prev) = (d.u[i], d.v.map(|v| v[i]));
            check_len("u", g, n as usize)?;
            if let Some(v) = prev {
                check_len("v", v, n as usize)?;
            }
            // Without `v`, `u` stands in for it, unread.
            let at = [
                addr("u", g)?,
                addr("v", prev.unwrap_or(g))?,
                pairs_at(parts, i * CG_BLOCKS as usize)?,
            ];
            let reads: Vec<_> = [Some(g), prev]
                .into_iter()
                .flatten()
                .map(|b| b.storage())
                .collect();
            lease(ctx, &reads, &[parts.storage()])?;
            let scale = if i == 0 { 1.0 } else { d.u_scale };
            self.dots_run(Run::Launch(ctx), at, n, scale, prev.is_some())?;
        }
        Ok(())
    }

    fn dots_run(
        &self,
        run: Run,
        [g, prev, out]: [CUdeviceptr; 3],
        n: i32,
        scale: f32,
        cg: bool,
    ) -> Result<(), Error> {
        // SAFETY: a launch comes from `dots`, which checked that `g` holds `n` complex values,
        // `prev` too when `cg`, and `out` `CG_BLOCKS` pairs, and leased them on `ctx`.
        unsafe {
            run!(
                run,
                device::bh_dots(ptr(g), ptr(prev), n, scale, i32::from(cg), ptr(out))
                    .grid((CG_BLOCKS, 1, 1))
            );
        }
        Ok(())
    }
}

/// The Tile forms of [`crate::gpu::GradKernels`]' CG dot products `beta_parts`, `norm_parts`
/// and `sum_parts` over a scan of one [`ViewDims`], in their block order and float order, so
/// their sums are bit-identical. [`Self::new`] compiles each for the lengths a scan's calls
/// take, the object's and the probe's `n` and the `npart`s `CG_BLOCKS`, `2 CG_BLOCKS` and
/// [`gamma_parts`]; a launch of another length is refused rather than compiled.
pub struct TileCgDots {
    /// Complex values of the object and of the probe.
    lens: [i32; 2],
    /// The `npart`s of [`Self::sum_parts`].
    nparts: [i32; 3],
}

impl TileCgDots {
    /// Checks the lengths against the kernels' `i32` offsets and compiles every kernel on
    /// `stream` for a scan of `dims`, without launching any.
    pub fn new(stream: &Arc<CudaStream>, dims: &ViewDims) -> Result<Self, Error> {
        let lens = dot_lens(dims)?;
        let gamma = gamma_parts(dims)? as usize;
        offset("line-search pairs", 2 * (gamma + 256))?;
        let c = CG_BLOCKS as i32;
        let t = Self {
            lens,
            nparts: [c, 2 * c, gamma as i32],
        };
        let run = Run::Compile(stream);
        for n in lens {
            t.beta_run(run, [UNBOUND; 3], n)?;
            t.norm_run(run, [UNBOUND; 2], n)?;
        }
        for npart in t.nparts {
            t.sum_run(run, [UNBOUND; 2], npart)?;
        }
        Ok(t)
    }

    /// `n` as compiled: one of the object's and the probe's lengths.
    fn len(&self, n: u32) -> Result<i32, Error> {
        self.lens
            .into_iter()
            .find(|&l| l as u32 == n)
            .ok_or_else(|| shape(format!("no Tile dot product compiled for {n} values")))
    }

    /// [`crate::gpu::GradKernels::beta_parts`]: one array's Polak–Ribière partial sums `(sum
    /// Re(conj(g) (g - h)), sum |h|^2)` over its first `n` values into the `CG_BLOCKS` pairs
    /// of `parts` from pair `first`, then `h = g`.
    pub fn beta_parts(
        &self,
        ctx: &ExecutionContext,
        g: &DeviceBuffer<C32>,
        h: &mut DeviceBuffer<C32>,
        n: u32,
        parts: &mut DeviceBuffer<f32>,
        first: u32,
    ) -> Result<(), Error> {
        let n = self.len(n)?;
        check_len("g", g, n as usize)?;
        check_len("h", h, n as usize)?;
        check_len("parts", parts, 2 * (first as usize + CG_BLOCKS as usize))?;
        let at = [
            addr("g", g)?,
            addr("h", h)?,
            pairs_at(parts, first as usize)?,
        ];
        lease(ctx, &[g.storage()], &[h.storage(), parts.storage()])?;
        self.beta_run(Run::Launch(ctx), at, n)
    }

    /// [`crate::gpu::GradKernels::norm_parts`]: adds `(sum |x|, sum |x|^2)` over the first `n`
    /// values of `x` to the first `CG_BLOCKS` pairs of `parts`.
    pub fn norm_parts(
        &self,
        ctx: &ExecutionContext,
        x: &DeviceBuffer<C32>,
        n: u32,
        parts: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let n = self.len(n)?;
        check_len("x", x, n as usize)?;
        check_len("parts", parts, 2 * CG_BLOCKS as usize)?;
        let at = [addr("x", x)?, addr("parts", parts)?];
        lease(ctx, &[x.storage()], &[parts.storage()])?;
        self.norm_run(Run::Launch(ctx), at, n)
    }

    /// [`crate::gpu::GradKernels::sum_parts`]: `pair = sum` of the first `npart` pairs of
    /// `parts`, in a fixed order.
    pub fn sum_parts(
        &self,
        ctx: &ExecutionContext,
        parts: &DeviceBuffer<f32>,
        npart: u32,
        pair: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let npart = self
            .nparts
            .into_iter()
            .find(|&p| p as u32 == npart)
            .ok_or_else(|| shape(format!("no Tile sum compiled for {npart} pairs")))?;
        check_len("parts", parts, 2 * npart as usize)?;
        check_len("pair", pair, 2)?;
        let at = [addr("parts", parts)?, addr("pair", pair)?];
        lease(ctx, &[parts.storage()], &[pair.storage()])?;
        self.sum_run(Run::Launch(ctx), at, npart)
    }

    fn beta_run(&self, run: Run, [g, h, out]: [CUdeviceptr; 3], n: i32) -> Result<(), Error> {
        // SAFETY: a launch comes from `beta_parts`, which checked that `g` and `h` hold `n`
        // complex values and `out` `CG_BLOCKS` pairs, and leased them on `ctx`.
        unsafe {
            run!(
                run,
                device::cg_beta_parts(ptr(g), ptr(h), n, ptr(out)).grid((CG_BLOCKS, 1, 1))
            );
        }
        Ok(())
    }

    fn norm_run(&self, run: Run, [x, out]: [CUdeviceptr; 2], n: i32) -> Result<(), Error> {
        // SAFETY: a launch comes from `norm_parts`, which checked that `x` holds `n` complex
        // values and `out` `CG_BLOCKS` pairs, and leased them on `ctx`.
        unsafe {
            run!(
                run,
                device::cg_norm_parts(ptr(x), n, ptr(out)).grid((CG_BLOCKS, 1, 1))
            );
        }
        Ok(())
    }

    fn sum_run(&self, run: Run, [parts, out]: [CUdeviceptr; 2], npart: i32) -> Result<(), Error> {
        // SAFETY: a launch comes from `sum_parts`, which checked that `parts` holds `npart`
        // pairs and `out` one, and leased them on `ctx`.
        unsafe {
            run!(
                run,
                device::cg_sum_parts(ptr(parts), npart, ptr(out)).grid((1, 1, 1))
            );
        }
        Ok(())
    }
}

/// Address of pair `first` of `parts`, which must be 16-byte aligned like every address
/// [`addr`] accepts.
fn pairs_at(parts: &DeviceBuffer<f32>, first: usize) -> Result<CUdeviceptr, Error> {
    let a = addr("parts", parts)? + 8 * first as CUdeviceptr;
    if !a.is_multiple_of(16) {
        return Err(shape(format!(
            "pair {first} of parts is not 16-byte aligned"
        )));
    }
    Ok(a)
}
